"""DSpark with ``layer_type="grug"``: Grug-style layers instead of Qwen3 ones."""

import logging
import math
from typing import Any, cast

import pytest
import torch
from torch.nn import functional
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash.grug_definitions import (
    GrugDSparkDecoderLayer,
    grug_rms_norm_no_weight,
    partial_rotary_config,
)
from speculators.models.dflash.model_definitions import Qwen3DFlashDecoderLayer
from speculators.models.dspark import DSparkSpeculatorConfig
from speculators.models.dspark.core import DSparkDraftModel
from speculators.proposals.greedy import GreedyTokenProposalConfig

HIDDEN = 32
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 16
PARTIAL = 0.5
QK_MULT = 1.75


def _tl_config(**overrides: Any) -> Qwen3Config:
    kwargs: dict[str, Any] = {
        "hidden_size": HIDDEN,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": 64,
        "_attn_implementation": "eager",
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": PARTIAL,
        },
    }
    kwargs.update(overrides)
    return Qwen3Config(**kwargs)


def _config(layer_type: str = "grug", **overrides: Any) -> DSparkSpeculatorConfig:
    kwargs: dict[str, Any] = {
        "transformer_layer_config": _tl_config(),
        "draft_vocab_size": 64,
        "block_size": 4,
        "aux_hidden_state_layer_ids": [0, 1],
        "mask_token_id": 0,
        "layer_type": layer_type,
        "speculators_config": SpeculatorsConfig(
            algorithm="dspark",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=4)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(name_or_path="dummy", architectures=[]),
        ),
    }
    kwargs.update(overrides)
    return DSparkSpeculatorConfig(**kwargs)


def _randomize(module: torch.nn.Module) -> None:
    """Give every parameter a non-trivial value (attn_gate starts at zero)."""
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(torch.randn(param.shape, generator=gen) * 0.2)


def test_default_layer_type_is_unchanged_qwen3():
    config = DSparkSpeculatorConfig(transformer_layer_config=_tl_config())
    assert config.layer_type == "qwen3"
    model = DSparkDraftModel(_config("qwen3"))
    assert all(isinstance(layer, Qwen3DFlashDecoderLayer) for layer in model.layers)


def test_grug_layers_are_built_with_zero_initialised_output_gate():
    model = DSparkDraftModel(_config())
    assert len(model.layers) == 2
    for layer in model.layers:
        assert isinstance(layer, GrugDSparkDecoderLayer)
        assert not layer.self_attn.attn_gate.weight.any()  # untrained gate = identity
    assert isinstance(model, DFlashDraftModel)
    names = set(model.state_dict())
    assert "layers.0.self_attn.attn_gate.weight" in names
    assert "layers.0.shared_expert.gate_proj.weight" in names
    assert not [
        n for n in names if "q_norm" in n or "k_norm" in n
    ]  # q/k norm has no weight


def test_grug_config_round_trips_through_json():
    config = _config(grug_qk_mult=1.5, grug_gated_norm_rank=64)
    restored = DSparkSpeculatorConfig.model_validate_json(config.model_dump_json())
    assert restored.layer_type == "grug"
    assert restored.grug_qk_mult == 1.5
    assert restored.grug_gated_norm_rank == 64
    assert (
        restored.transformer_layer_config.rope_parameters["partial_rotary_factor"]
        == PARTIAL
    )


def test_grug_rejects_sliding_window_layers_and_routed_experts():
    sliding = _tl_config(
        layer_types=["full_attention", "sliding_attention"], sliding_window=8
    )
    with pytest.raises(ValueError, match="full attention"):
        _config(transformer_layer_config=sliding)
    with pytest.raises(ValueError, match="not implemented"):
        _config(grug_num_experts=8)


def test_partial_rotary_table_covers_only_the_rotary_channels():
    model = DSparkDraftModel(_config())
    cos, _ = model.rotary_emb(torch.zeros(1, 3, HIDDEN), torch.arange(3)[None])
    assert cos.shape[-1] == int(HEAD_DIM * PARTIAL)
    # Qwen3 layers keep the full-head table
    qwen_cos, _ = DSparkDraftModel(_config("qwen3")).rotary_emb(
        torch.zeros(1, 3, HIDDEN), torch.arange(3)[None]
    )
    assert qwen_cos.shape[-1] == HEAD_DIM
    assert partial_rotary_config(_tl_config()).head_dim == HEAD_DIM * PARTIAL
    with pytest.raises(ValueError, match="even"):
        partial_rotary_config(
            _tl_config(
                rope_parameters={
                    "rope_type": "default",
                    "rope_theta": 1e4,
                    "partial_rotary_factor": 0.2,
                }
            )
        )


def _reference_layer(layer: GrugDSparkDecoderLayer, ctx, x, cos, sin, qk_mult):
    """Grug layer math written out with plain ops (bidirectional ctx + block)."""
    bsz, q_len, _ = x.shape
    ctx_len = ctx.shape[1]

    def rms(t, weight):
        t32 = t.float()
        out = t32 * torch.rsqrt(t32.square().mean(-1, keepdim=True) + 1e-5)
        return out * weight

    def gated(t, norm):
        return t * torch.sigmoid(norm.up_proj(functional.silu(norm.down_proj(t))))

    def rope(t):  # t: [b, heads, len, hd]; rotate the first rotary_dim channels
        rot = cos.shape[-1]
        head, tail = t[..., :rot], t[..., rot:]
        half = rot // 2
        rotated = torch.cat((-head[..., half:], head[..., :half]), dim=-1)
        c, s_ = (
            cos[:, None, -t.shape[2] :],
            sin[:, None, -t.shape[2] :],
        )  # queries use the block's rows
        return torch.cat((head * c + rotated * s_, tail), dim=-1)

    attn = layer.self_attn
    h = gated(rms(x, layer.input_layernorm.weight), layer.attn_gated_norm)
    q = attn.q_proj(h).view(bsz, q_len, HEADS, HEAD_DIM)
    k = torch.cat([attn.k_proj(ctx), attn.k_proj(h)], dim=1).view(
        bsz, -1, KV_HEADS, HEAD_DIM
    )
    v = torch.cat([attn.v_proj(ctx), attn.v_proj(h)], dim=1).view(
        bsz, -1, KV_HEADS, HEAD_DIM
    )
    q, k = (
        grug_rms_norm_no_weight(q).transpose(1, 2),
        grug_rms_norm_no_weight(k).transpose(1, 2),
    )
    v = v.transpose(1, 2)
    q, k = rope(q), rope(k)
    q = q * qk_mult
    group = HEADS // KV_HEADS
    scores = (
        q @ k.repeat_interleave(group, dim=1).transpose(-1, -2) / math.sqrt(HEAD_DIM)
    )
    out = torch.softmax(scores, dim=-1) @ v.repeat_interleave(
        group, dim=1
    )  # [b, H, q, hd]
    own_v = v[:, :, ctx_len:].repeat_interleave(group, dim=1)
    out = (
        out
        - (out * own_v).sum(-1, keepdim=True)
        / (own_v.square().sum(-1, keepdim=True) + 1e-6)
        * own_v
    )
    gate = 2.0 * torch.sigmoid(attn.attn_gate(h)).transpose(1, 2)  # [b, H, q]
    out = (out * gate[..., None]).transpose(1, 2).reshape(bsz, q_len, -1)
    x = x + attn.o_proj(out)
    m = gated(rms(x, layer.post_attention_layernorm.weight), layer.mlp_gated_norm)
    expert = layer.shared_expert
    return x + expert.down_proj(
        functional.silu(expert.gate_proj(m)) * expert.up_proj(m)
    )


def test_grug_layer_matches_reference_math():
    model = DSparkDraftModel(_config())
    _randomize(model)
    layer = cast("GrugDSparkDecoderLayer", model.layers[0].eval())
    ctx_len, q_len = 6, 4
    torch.manual_seed(1)
    ctx = torch.randn(1, ctx_len, HIDDEN)
    x = torch.randn(1, q_len, HIDDEN)
    # positions of [ctx | block]; the table is the partial (half-RoPE) one
    position_ids = torch.arange(ctx_len + q_len)[None]
    cos, sin = model.rotary_emb(x, position_ids)
    with torch.no_grad():
        got = layer(
            target_hidden=ctx,
            hidden_states=x,
            attention_mask=None,
            position_embeddings=(cos, sin),
        )
        want = _reference_layer(layer, ctx, x, cos, sin, QK_MULT)
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-4)


def test_grug_layer_is_position_aware_inside_the_block():
    """Identical mask tokens must differ by position (no NoPE block collapse)."""
    model = DSparkDraftModel(_config())
    _randomize(model)
    layer = cast("GrugDSparkDecoderLayer", model.layers[0].eval())
    ctx = torch.randn(1, 5, HIDDEN)
    x = (
        torch.randn(1, 1, HIDDEN).expand(1, 4, HIDDEN).contiguous()
    )  # identical block tokens
    cos, sin = model.rotary_emb(x, torch.arange(9)[None])
    with torch.no_grad():
        out = layer(
            target_hidden=ctx,
            hidden_states=x,
            attention_mask=None,
            position_embeddings=(cos, sin),
        )
    assert not torch.allclose(out[:, 0], out[:, 1], atol=1e-4)


def test_untrained_output_gate_is_identity_and_xsa_removes_the_own_value_direction():
    model = DSparkDraftModel(_config())
    layer = cast("GrugDSparkDecoderLayer", model.layers[0].eval())
    attn = layer.self_attn
    torch.nn.init.normal_(attn.v_proj.weight)
    ctx = torch.randn(1, 5, HIDDEN)
    x = torch.randn(1, 3, HIDDEN)
    cos, sin = model.rotary_emb(x, torch.arange(8)[None])
    captured: dict[str, torch.Tensor] = {}
    attn.o_proj.register_forward_hook(
        lambda _m, inp, _out: captured.update(pre_o=inp[0])
    )
    with torch.no_grad():
        attn(
            hidden_states=x,
            target_hidden=ctx,
            attention_mask=None,
            position_embeddings=(cos, sin),
        )
        own_v = (
            attn.v_proj(x)
            .view(1, 3, KV_HEADS, HEAD_DIM)
            .repeat_interleave(HEADS // KV_HEADS, dim=2)
        )
    pre_o = captured["pre_o"].view(1, 3, HEADS, HEAD_DIM)
    along_v = (pre_o * own_v).sum(-1) / own_v.norm(dim=-1)
    assert along_v.abs().max() < 1e-3  # the gate is 1, so only XSA acts on the output


def test_partial_rotary_config_drops_the_factor_from_its_copy():
    config = _tl_config()
    rotary = partial_rotary_config(config)
    assert rotary.head_dim == HEAD_DIM * PARTIAL
    # head_dim already carries the factor, so the copy must not apply it again
    assert rotary.rope_parameters is not None
    assert "partial_rotary_factor" not in rotary.rope_parameters
    assert config.rope_parameters is not None
    assert config.rope_parameters["partial_rotary_factor"] == PARTIAL
    assert config.head_dim == HEAD_DIM


def test_grug_without_partial_rotary_factor_logs_a_warning(caplog):
    no_factor = _tl_config(
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0}
    )
    logger_name = "speculators.models.dspark.config"
    with caplog.at_level(logging.WARNING, logger=logger_name):
        _config(transformer_layer_config=no_factor)
    assert "partial_rotary_factor" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=logger_name):
        _config()
        _config("qwen3", transformer_layer_config=no_factor)
    assert "partial_rotary_factor" not in caplog.text


def test_dspark_forward_backward_reaches_every_trainable_parameter():
    """A full DSpark training step through the real draft attention mask (CPU)."""
    torch.manual_seed(0)
    model = DSparkDraftModel(_config())
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.isnan().any():
                torch.nn.init.normal_(parameter, std=0.02)
    seq_len = 32
    trainer_kwargs, _ = DSparkDraftModel.get_trainer_kwargs(
        loss_fn='{"ce": 0.1, "tv": 0.9}',
        loss_implementation="eager",
        max_anchors=4,
    )
    _, loss, _ = model(
        hidden_states=torch.randn(1, seq_len, 2 * HIDDEN),
        input_ids=torch.randint(0, model.verifier_vocab_size, (1, seq_len)),
        loss_mask=torch.ones(1, seq_len),
        verifier_last_hidden_states=torch.randn(1, seq_len, HIDDEN),
        document_ids=torch.zeros(1, seq_len, dtype=torch.long),
        **trainer_kwargs,
    )
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        assert parameter.grad is not None, f"missing gradient for {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient for {name}"
        assert torch.count_nonzero(parameter.grad), f"zero gradient for {name}"
