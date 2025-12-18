"""Weight adapters for GPT-OSS MXFP4 (ModuleV3) checkpoints.

These adapters rename HuggingFace safetensors keys into the Weight names used
by the MAX ModuleV3 tree under `examples/custom-models/gpt_oss_mxfp4_v3/`.

Key design choice:
- Keep expert weights as checkpoint-native `uint8` blocks + `uint8` scales.
- Keep expert biases as BF16 (checkpoint dtype).
"""

from __future__ import annotations

from collections import OrderedDict

from max.graph.weights import WeightData, Weights

# Ordered so that more specific mappings happen before similarly-prefixed weights.
GPT_OSS_SAFETENSOR_MAP: OrderedDict[str, str] = OrderedDict(
    [
        ("model.embed_tokens.", "language_model.embed_tokens."),
        ("model.norm.", "language_model.norm."),
        ("lm_head.", "language_model.lm_head."),
        ("model.layers.", "language_model.layers."),
        # Router uses bias-enabled Linear layer under `.mlp.gate.gate_score`.
        (".mlp.router", ".mlp.gate.gate_score"),
    ]
)


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights], **kwargs
) -> dict[str, WeightData]:
    """Convert safetensor state dict to MAX format."""

    _ = kwargs

    new_state_dict: dict[str, WeightData] = {}
    for weight_name, value in state_dict.items():
        mapped = weight_name
        for before, after in GPT_OSS_SAFETENSOR_MAP.items():
            mapped = mapped.replace(before, after)
        new_state_dict[mapped] = value.data() if hasattr(value, "data") else value
    return new_state_dict


__all__ = ["GPT_OSS_SAFETENSOR_MAP", "convert_safetensor_state_dict"]
