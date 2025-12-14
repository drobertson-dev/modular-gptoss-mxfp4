from __future__ import annotations

from dataclasses import dataclass, field

import pytest

pytest.importorskip("max")

from gpt_oss.model_config import GptOssConfig
from max.pipelines.lib import (  # type: ignore
    KVCacheConfig,
    PipelineConfig,
    YarnScalingParams,
)

config = {
    "architectures": ["GptOssForCausalLM"],
    "attention_bias": True,
    "attention_dropout": 0.0,
    "eos_token_id": 200002,
    "experts_per_token": 4,
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": 2880,
    "initial_context_length": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 2880,
    "layer_types": [
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
    ],
    "max_position_embeddings": 131072,
    "model_type": "gpt_oss",
    "num_attention_heads": 64,
    "num_experts_per_tok": 4,
    "num_hidden_layers": 24,
    "num_key_value_heads": 8,
    "num_local_experts": 32,
    "output_router_logits": False,
    "pad_token_id": 199999,
    "quantization_config": {
        "modules_to_not_convert": [
            "model.layers.*.self_attn",
            "model.layers.*.mlp.router",
            "model.embed_tokens",
            "lm_head",
        ],
        "quant_method": "mxfp4",
    },
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "factor": 32.0,
        "original_max_position_embeddings": 4096,
        "rope_type": "yarn",
        "truncate": False,
    },
    "rope_theta": 150000,
    "router_aux_loss_coef": 0.9,
    "sliding_window": 128,
    "swiglu_limit": 7.0,
    "tie_word_embeddings": False,
    "transformers_version": "4.55.0.dev0",
    "use_cache": True,
    "vocab_size": 201088,
}


@dataclass
class FakeHFConfig:
    vocab_size: int = 201_088
    hidden_size: int = 2_880
    intermediate_size: int = 2_880
    num_hidden_layers: int = 24
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 64
    max_position_embeddings: int = 131_072
    rope_theta: float = 150_000
    rms_norm_eps: float = 1e-5
    sliding_window: int = 128
    rope_scaling: dict = field(
        default_factory=lambda: {
            "rope_type": "yarn",
            "factor": 32.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "original_max_position_embeddings": 4096,
            "truncate": False,
        }
    )
    layer_types: list[str] | None = field(
        default_factory=lambda: [
            "sliding_attention",
            "full_attention",
        ]
        * 12
    )
    hidden_act: str = "gelu_pytorch_tanh"
    tie_word_embeddings: bool = False
    router_aux_loss_coef: float = 0.9
    num_experts_per_tok: int = 4
    num_local_experts: int = 32
    swiglu_limit: float = 7.0
    attention_bias: bool = True


def _minimal_pipeline_config(hf: FakeHFConfig) -> PipelineConfig:
    return PipelineConfig(
        model_config=hf,
        tokenizer=None,
        devices=[],
    )


def test_get_kv_params_matches_hf_fields():
    hf = FakeHFConfig()
    kv_cache_config = KVCacheConfig(kv_cache_page_size=16)
    params = GptOssConfig.get_kv_params(
        hf_config=hf,
        n_devices=2,
        kv_cache_config=kv_cache_config,
        cache_dtype="bfloat16",
    )

    assert params.num_layers == hf.num_hidden_layers
    assert params.n_kv_heads == hf.num_key_value_heads
    assert params.head_dim == hf.head_dim
    assert params.n_devices == 2
    assert params.dtype == "bfloat16"
    assert params.page_size == kv_cache_config.kv_cache_page_size


def test_generate_populates_core_fields_and_types():
    hf = FakeHFConfig()
    pipeline_config = _minimal_pipeline_config(hf)
    cfg = GptOssConfig.generate(
        pipeline_config=pipeline_config,
        hf_config=hf,
        state_dict={"language_model.lm_head.weight": object()},
        return_logits=None,
        kv_cache_config=KVCacheConfig(kv_cache_page_size=16),
        devices=[],
    )

    assert cfg.vocab_size == hf.vocab_size
    assert cfg.hidden_size == hf.hidden_size
    assert cfg.intermediate_size == hf.intermediate_size
    assert cfg.num_hidden_layers == hf.num_hidden_layers
    assert cfg.num_attention_heads == hf.num_attention_heads
    assert cfg.num_key_value_heads == hf.num_key_value_heads
    assert cfg.head_dim == hf.head_dim
    assert cfg.max_position_embeddings == hf.max_position_embeddings
    assert cfg.rope_theta == hf.rope_theta
    assert cfg.rms_norm_eps == hf.rms_norm_eps
    assert cfg.sliding_window == hf.sliding_window
    assert isinstance(cfg.rope_scaling, YarnScalingParams)
    assert cfg.hidden_activation == "gelu_tanh"
    assert cfg.tie_word_embeddings is False
    assert cfg.layer_types == hf.layer_types


def test_generate_tie_embeddings_default_when_missing_head():
    hf = FakeHFConfig()
    pipeline_config = _minimal_pipeline_config(hf)
    cfg = GptOssConfig.generate(
        pipeline_config=pipeline_config,
        hf_config=hf,
        state_dict={},  # lm_head missing
        return_logits=None,
        kv_cache_config=KVCacheConfig(kv_cache_page_size=16),
        devices=[],
    )

    assert cfg.tie_word_embeddings is True


def test_generate_layer_types_fallback_when_missing():
    hf = FakeHFConfig(layer_types=None)
    pipeline_config = _minimal_pipeline_config(hf)
    cfg = GptOssConfig.generate(
        pipeline_config=pipeline_config,
        hf_config=hf,
        state_dict={},
        return_logits=None,
        kv_cache_config=KVCacheConfig(kv_cache_page_size=16),
        devices=[],
    )

    assert len(cfg.layer_types) == hf.num_hidden_layers
    assert all(
        t in ("sliding_attention", "full_attention") for t in cfg.layer_types
    )


@pytest.mark.parametrize(
    "rope_scaling",
    [None, {"rope_type": "linear"}, {"rope_type": "unknown"}],
)
def test_generate_rejects_invalid_rope_scaling(rope_scaling):
    hf = FakeHFConfig(rope_scaling=rope_scaling)
    pipeline_config = _minimal_pipeline_config(hf)

    with pytest.raises(ValueError):
        GptOssConfig.generate(
            pipeline_config=pipeline_config,
            hf_config=hf,
            state_dict={},
            return_logits=None,
            kv_cache_config=KVCacheConfig(kv_cache_page_size=16),
            devices=[],
        )
