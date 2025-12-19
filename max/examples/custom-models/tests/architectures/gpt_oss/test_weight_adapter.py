from __future__ import annotations

import pytest

pytest.importorskip("max")

from gpt_oss_mxfp4 import weight_adapters
from max.dtype import DType


class _DummyCastable:
    def __init__(self) -> None:
        self.cast_dtype: DType | None = None

    def astype(self, dtype: DType):
        self.cast_dtype = dtype
        return self


def test_prefix_and_router_mapping() -> None:
    state = {
        "model.layers.0.mlp.router.weight": object(),
        "model.layers.0.mlp.router.bias": object(),
        "model.embed_tokens.weight": object(),
        "model.norm.weight": object(),
        "lm_head.weight": object(),
    }
    mapped = weight_adapters.convert_safetensor_state_dict(state)

    assert "language_model.layers.0.mlp.gate.gate_score.weight" in mapped
    assert "language_model.layers.0.mlp.gate.gate_score.bias" in mapped
    assert "language_model.embed_tokens.weight" in mapped
    assert "language_model.norm.weight" in mapped
    assert "language_model.lm_head.weight" in mapped


def test_moe_expert_biases_cast_to_f32() -> None:
    gate_bias = _DummyCastable()
    down_bias = _DummyCastable()
    state = {
        "model.layers.0.mlp.experts.gate_up_proj_bias": gate_bias,
        "model.layers.0.mlp.experts.down_proj_bias": down_bias,
    }
    mapped = weight_adapters.convert_safetensor_state_dict(state)

    assert (
        mapped["language_model.layers.0.mlp.experts.gate_up_proj_bias"].cast_dtype
        == DType.float32
    )
    assert (
        mapped["language_model.layers.0.mlp.experts.down_proj_bias"].cast_dtype
        == DType.float32
    )

