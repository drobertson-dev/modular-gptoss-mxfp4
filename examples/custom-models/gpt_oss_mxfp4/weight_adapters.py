"""Weight adapters for GPT-OSS MXFP4 checkpoints.

These adapters rename HuggingFace safetensors keys into the Weight names used
by the MAX Module tree under `examples/custom-models/gpt_oss_mxfp4/`.
"""

from __future__ import annotations

from collections import OrderedDict

from max.graph.weights import WeightData, Weights

# Ordered so that bias mappings happen before similarly-prefixed weights.
GPT_OSS_SAFETENSOR_MAP: OrderedDict[str, str] = OrderedDict(
    [
        ("model.embed_tokens.", "language_model.embed_tokens."),
        ("model.norm.", "language_model.norm."),
        ("lm_head.", "language_model.lm_head."),
        ("model.layers.", "language_model.layers."),
        # Router uses bias-enabled Linear layer under `.mlp.gate.gate_score`.
        (".mlp.router", ".mlp.gate.gate_score"),
        # MoE expert parameters (MXFP4 blocks/scales + BF16 biases).
        ("experts.gate_up_proj_bias", "_experts_gate_up_proj_bias"),
        ("experts.down_proj_bias", "_experts_down_proj_bias"),
        # The following entries must be listed after the bias weights.
        ("experts.gate_up_proj", "_experts_gate_up_proj_weight"),
        ("experts.down_proj", "_experts_down_proj_weight"),
    ]
)


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights], **kwargs
) -> dict[str, WeightData]:
    """Convert safetensor state dict to MAX format."""

    new_state_dict: dict[str, WeightData] = {}
    for weight_name, value in state_dict.items():
        mapped = weight_name
        for before, after in GPT_OSS_SAFETENSOR_MAP.items():
            mapped = mapped.replace(before, after)
        new_state_dict[mapped] = (
            value.data() if hasattr(value, "data") else value
        )
    return new_state_dict


__all__ = ["GPT_OSS_SAFETENSOR_MAP", "convert_safetensor_state_dict"]
