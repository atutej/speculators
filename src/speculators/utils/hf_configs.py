"""Hugging Face config classes for verifier architectures that Transformers does not ship.

Grug (Marin's MoE family, model_type ``grug_moe``) is exported with a config.json but no config code, so
``AutoConfig.from_pretrained`` raises "does not recognize this architecture". Marin's Levanter and the Marin vLLM fork
each register a class at import time; this environment has neither, so register a minimal one here. A bare
``PretrainedConfig`` keeps every key of config.json as an attribute (hidden_size, num_hidden_layers,
num_attention_heads, num_key_value_heads, head_dim, vocab_size, ...), which is all the drafter sizing needs.
Registration uses ``exist_ok=True`` and is skipped if a richer class was already registered.
"""

from transformers import AutoConfig, PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING


class GrugMoeConfig(PretrainedConfig):
    model_type = "grug_moe"


def register_verifier_configs() -> None:
    if "grug_moe" not in CONFIG_MAPPING:
        AutoConfig.register("grug_moe", GrugMoeConfig, exist_ok=True)
