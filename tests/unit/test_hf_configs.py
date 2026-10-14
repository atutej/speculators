from transformers import AutoConfig

import speculators  # noqa: F401  (registers verifier configs on import)


def test_grug_moe_config_loads_from_a_config_dict():
    config = AutoConfig.for_model("grug_moe", hidden_size=2560, num_hidden_layers=26, num_attention_heads=20,
                                  num_key_value_heads=5, head_dim=128, vocab_size=128256)
    assert config.model_type == "grug_moe"
    assert (config.hidden_size, config.num_hidden_layers, config.head_dim) == (2560, 26, 128)
