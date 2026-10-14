"""The gated verifier head (Grug) is opt-in: every other verifier's targets are computed exactly as before."""

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dspark import DSparkSpeculatorConfig
from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.utils import apply_verifier_gate, uses_gated_verifier_head

HIDDEN = 16


def _tl_config() -> Qwen3Config:
    return Qwen3Config(
        hidden_size=HIDDEN, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, vocab_size=64, _attn_implementation="eager",  # type: ignore[call-arg]
    )


def _dflash() -> DFlashDraftModel:
    config = DFlashSpeculatorConfig(
        transformer_layer_config=_tl_config(), draft_vocab_size=64, block_size=4,
        aux_hidden_state_layer_ids=[0, 1], mask_token_id=0, sample_from_anchor=True,
    )
    model = DFlashDraftModel(config)
    torch.nn.init.normal_(model.verifier_lm_head.weight)
    torch.nn.init.normal_(model.verifier_norm.weight)
    return model.eval()


def test_only_grug_architectures_are_gated():
    assert uses_gated_verifier_head(["GrugMoeForCausalLM"])
    assert not uses_gated_verifier_head(["Qwen3ForCausalLM"])
    assert not uses_gated_verifier_head(["LlamaForCausalLM", "Qwen3MoeForCausalLM"])
    assert not uses_gated_verifier_head([])
    assert not uses_gated_verifier_head(None)
    # a config without `architectures` is still recognised by its model_type
    assert uses_gated_verifier_head(None, "grug_moe")
    assert uses_gated_verifier_head([], "grug_moe")
    assert not uses_gated_verifier_head(None, "qwen3")
    assert not uses_gated_verifier_head(["Qwen3ForCausalLM"], "qwen3")


def test_non_gated_verifier_logits_are_the_old_computation_exactly():
    model = _dflash()
    assert model.verifier_gate_down_weight is None and model.verifier_gate_up_weight is None
    h = torch.randn(1, 12, HIDDEN)
    with torch.no_grad():
        old = model.verifier_lm_head(model.verifier_norm(h))  # what the code computed before the patch
        new = model._verifier_logits(h)
    assert torch.equal(old, new)


def test_gate_buffers_are_not_saved_in_checkpoints():
    state = _dflash().state_dict()
    assert not [k for k in state if "verifier_gate" in k]  # a Qwen/Llama drafter's checkpoint is byte-for-byte unchanged in layout


def test_gate_applies_only_when_weights_are_loaded():
    model = _dflash()
    h = torch.randn(1, 12, HIDDEN)
    down, up = torch.randn(4, HIDDEN), torch.randn(HIDDEN, 4)
    with torch.no_grad():
        ungated = model._verifier_logits(h)
        model.verifier_gate_down_weight, model.verifier_gate_up_weight = down, up
        gated = model._verifier_logits(h)
        normed = model.verifier_norm(h)
        expected = model.verifier_lm_head(normed * torch.sigmoid(torch.nn.functional.linear(torch.nn.functional.silu(torch.nn.functional.linear(normed, down)), up)))
    assert not torch.allclose(ungated, gated)
    assert torch.allclose(gated, expected, atol=1e-6)
    assert torch.allclose(apply_verifier_gate(normed, down, up), normed * torch.sigmoid(torch.nn.functional.linear(torch.nn.functional.silu(torch.nn.functional.linear(normed, down)), up)))


def test_dspark_inherits_the_same_opt_in_behaviour():
    config = DSparkSpeculatorConfig(
        transformer_layer_config=_tl_config(), draft_vocab_size=64, block_size=4,
        aux_hidden_state_layer_ids=[0, 1], mask_token_id=0,
    )
    model = DSparkDraftModel(config).eval()
    torch.nn.init.normal_(model.verifier_lm_head.weight)
    torch.nn.init.normal_(model.verifier_norm.weight)
    assert isinstance(model, DFlashDraftModel)
    assert model.verifier_gate_down_weight is None
    h = torch.randn(1, 8, HIDDEN)
    with torch.no_grad():
        assert torch.equal(model.verifier_lm_head(model.verifier_norm(h)), model._verifier_logits(h))


def test_missing_architectures_does_not_silently_skip_the_gate(monkeypatch):
    """No `architectures` on a Grug verifier: the gate is still loaded (and a warning logged); Qwen still gets none."""
    from types import SimpleNamespace

    import speculators.models.dflash.core as core
    import speculators.utils.loading as loading

    down, up = torch.randn(4, HIDDEN), torch.randn(HIDDEN, 4)
    names = ("model.final_gated_norm.down_proj.weight", "model.final_gated_norm.up_proj.weight")
    monkeypatch.setattr(loading, "load_model_layers", lambda layer_names, path: dict(zip(names, (down, up), strict=True)))
    for model_type, expect_gate in (("grug_moe", True), ("qwen3", False)):
        model = _dflash()
        monkeypatch.setattr(core.DFlashDraftModel, "_load_verifier_weights", lambda self, **kw: None)
        monkeypatch.setattr(core, "get_verifier_config", lambda path, mt=model_type: SimpleNamespace(model_type=mt))
        model.config.speculators_config = SimpleNamespace(verifier=SimpleNamespace(architectures=None, name_or_path="x"))
        model.load_verifier_weights()
        assert (model.verifier_gate_down_weight is not None) is expect_gate


def test_gate_weights_in_another_dtype_are_cast_to_the_hidden_dtype():
    """The checkpoint's gate is bf16; the draft model may be fp32 (or the reverse)."""
    down, up = torch.randn(4, HIDDEN), torch.randn(HIDDEN, 4)
    for hidden_dtype, weight_dtype in ((torch.float32, torch.bfloat16), (torch.bfloat16, torch.float32)):
        h = torch.randn(1, 6, HIDDEN).to(hidden_dtype)
        out = apply_verifier_gate(h, down.to(weight_dtype), up.to(weight_dtype))
        assert out.dtype == hidden_dtype and torch.isfinite(out.float()).all()
