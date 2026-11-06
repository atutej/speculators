"""Grug-style decoder layer for DFlash-family drafts (``layer_type="grug"``).

Mirrors the Grug target layer (``GrugMoeDecoderLayer``) with the DFlash twist that
the keys/values of the context come from the target's hidden states. Differences
from ``Qwen3DFlashDecoderLayer``:

* q/k use a parameter-free RMSNorm, and q is scaled by ``qk_mult``.
* The attention output has its component along the token's own value removed
  (XSA) and is scaled by a per-head ``2 * sigmoid`` gate that starts at identity.
* Every RMSNorm is followed by a low-rank sigmoid "gated norm".
* The FFN is the dense shared expert (routed experts are not implemented yet).

Attention is always full (no sliding window) and positions use RoPE, which may be
partial through ``rope_parameters["partial_rotary_factor"]`` (Grug uses 0.5).
"""

from copy import copy

import torch
from torch import nn
from torch.nn import functional
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Config,
    eager_attention_forward,
)
from typing_extensions import Unpack

from speculators.models.eagle3.rotary_partial import partial_neox_apply_rotary_pos_emb

__all__ = [
    "GRUG_ATTN_GATE_SCALE",
    "GrugDSparkAttention",
    "GrugDSparkDecoderLayer",
    "GrugDenseMLP",
    "GrugGatedNorm",
    "GrugRMSNorm",
    "grug_rms_norm_no_weight",
    "partial_rotary_config",
]

# Per-head output gate: sigmoid scaled so an untrained (zero) gate is the identity.
GRUG_ATTN_GATE_SCALE = 2.0
GRUG_QK_RMS_NORM_EPS = 1e-6
GRUG_XSA_EPS = 1e-6


def partial_rotary_config(config: Qwen3Config) -> Qwen3Config:
    """Config whose rotary table covers only ``partial_rotary_factor`` of each head.

    ``Qwen3RotaryEmbedding`` always builds a full-head table, so shrink ``head_dim`` to
    the rotary dim; ``partial_neox_apply_rotary_pos_emb`` then rotates just those
    leading channels (the vLLM ``partial_rotary_factor`` behaviour).
    """
    factor = (config.rope_parameters or {}).get("partial_rotary_factor")
    if factor is None:
        factor = 1.0
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    rotary_dim = int(head_dim * factor)
    if rotary_dim <= 0 or rotary_dim % 2:
        raise ValueError(
            f"partial_rotary_factor={factor} gives a rotary dim of {rotary_dim} for "
            f"head_dim={head_dim}; it must be a positive even number"
        )
    rotary_config = copy(config)
    rotary_config.head_dim = rotary_dim
    # The factor is already applied through head_dim: drop it from the copy so a
    # rotary class that honors it cannot apply it a second time.
    rope_parameters = dict(config.rope_parameters or {})
    rope_parameters.pop("partial_rotary_factor", None)
    rotary_config.rope_parameters = rope_parameters
    return rotary_config


def grug_rms_norm_no_weight(hidden_states: torch.Tensor) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    fp32 = hidden_states.float()
    normed = fp32 * torch.rsqrt(
        fp32.square().mean(dim=-1, keepdim=True) + GRUG_QK_RMS_NORM_EPS
    )
    return normed.to(input_dtype)


class GrugRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        fp32 = hidden_states.float()
        normed = fp32 * torch.rsqrt(fp32.square().mean(dim=-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).to(input_dtype)


class GrugGatedNorm(nn.Module):
    """Elementwise ``sigmoid(up(silu(down(x))))`` gate through a rank bottleneck."""

    def __init__(self, hidden_size: int, rank: int) -> None:
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, rank, bias=False)
        self.up_proj = nn.Linear(rank, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(
            self.up_proj(functional.silu(self.down_proj(hidden_states)))
        )
        return hidden_states * gate.to(hidden_states.dtype)


class GrugDenseMLP(nn.Module):
    """SwiGLU FFN; with no routed experts this is Grug's shared expert."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            functional.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class GrugDSparkAttention(nn.Module):
    """Grug attention over ``[target context | draft block]`` keys and values."""

    def __init__(self, config: Qwen3Config, layer_idx: int, qk_mult: float):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,  # type: ignore[operator]
        )
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads  # type: ignore[operator]
        self.scaling = self.head_dim**-0.5
        self.qk_mult = qk_mult
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        hidden_size = config.hidden_size
        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)  # type: ignore[arg-type,operator]
        self.k_proj = nn.Linear(
            hidden_size,  # type: ignore[arg-type]
            self.num_kv_heads * self.head_dim,  # type: ignore[operator]
            bias=False,
        )
        self.v_proj = nn.Linear(
            hidden_size,  # type: ignore[arg-type]
            self.num_kv_heads * self.head_dim,  # type: ignore[operator]
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)  # type: ignore[arg-type,operator]
        self.attn_gate = nn.Linear(hidden_size, self.num_heads, bias=False)  # type: ignore[arg-type]
        # Zero gate logits make the output gate exactly 1 at init. Mark the module
        # so HF's weight init does not overwrite them.
        nn.init.zeros_(self.attn_gate.weight)
        self.attn_gate._is_hf_initialized = True  # type: ignore[assignment] # noqa: SLF001

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim)
        q = grug_rms_norm_no_weight(q).transpose(1, 2)
        # As in DFlash, the target hidden states supply the context keys/values.
        v_noise = self.v_proj(hidden_states).view(bsz, q_len, -1, self.head_dim)
        k = torch.cat(
            [self.k_proj(target_hidden), self.k_proj(hidden_states)], dim=1
        ).view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = torch.cat(
            [self.v_proj(target_hidden).view(bsz, ctx_len, -1, self.head_dim), v_noise],
            dim=1,
        )
        k = grug_rms_norm_no_weight(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        # cos/sin cover [context | block]; the queries are the block's last q_len rows.
        q, _ = partial_neox_apply_rotary_pos_emb(q, q, cos[:, -q_len:], sin[:, -q_len:])
        _, k = partial_neox_apply_rotary_pos_emb(k, k, cos, sin)
        q = q * self.qk_mult
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        attn_fn = eager_attention_forward
        if (
            self.config._attn_implementation is not None  # noqa: SLF001
            and self.config._attn_implementation != "eager"  # noqa: SLF001
        ):
            attn_fn = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation  # noqa: SLF001
            ]
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )  # [bsz, q_len, heads, head_dim]

        # XSA: drop the part of the output along the query token's own value.
        own_v = v_noise.repeat_interleave(self.num_key_value_groups, dim=2)
        dot = (attn_output * own_v).sum(dim=-1, keepdim=True)
        v_norm_sq = own_v.square().sum(dim=-1, keepdim=True)
        attn_output = attn_output - (dot / (v_norm_sq + GRUG_XSA_EPS)) * own_v
        gate = GRUG_ATTN_GATE_SCALE * torch.sigmoid(self.attn_gate(hidden_states))
        attn_output = attn_output * gate.unsqueeze(-1).to(attn_output.dtype)
        return self.o_proj(attn_output.reshape(bsz, q_len, -1)), attn_weights


class GrugDSparkDecoderLayer(GradientCheckpointingLayer):
    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        *,
        qk_mult: float,
        gated_norm_rank: int,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GrugDSparkAttention(config, layer_idx, qk_mult)
        self.input_layernorm = GrugRMSNorm(config.hidden_size, config.rms_norm_eps)  # type: ignore[arg-type]
        self.attn_gated_norm = GrugGatedNorm(config.hidden_size, gated_norm_rank)  # type: ignore[arg-type]
        self.post_attention_layernorm = GrugRMSNorm(
            config.hidden_size,  # type: ignore[arg-type]
            config.rms_norm_eps,  # type: ignore[arg-type]
        )
        self.mlp_gated_norm = GrugGatedNorm(config.hidden_size, gated_norm_rank)  # type: ignore[arg-type]
        self.shared_expert = GrugDenseMLP(config.hidden_size, config.intermediate_size)  # type: ignore[arg-type]

    def forward(
        self,
        target_hidden: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,  # noqa: ARG002
        past_key_value: Cache | None = None,
        output_attentions: bool | None = False,  # noqa: ARG002
        use_cache: bool | None = False,  # noqa: ARG002
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        assert hidden_states is not None  # noqa: S101
        attn_input = self.attn_gated_norm(self.input_layernorm(hidden_states))
        hidden_states = (
            hidden_states
            + self.self_attn(
                hidden_states=attn_input,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                past_key_values=past_key_value,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )[0]
        )
        mlp_input = self.mlp_gated_norm(self.post_attention_layernorm(hidden_states))
        return hidden_states + self.shared_expert(mlp_input)
