"""Weight adapters for GPT-OSS MXFP4 checkpoints.

These adapters rename HuggingFace safetensors keys into the Weight names used
by the MAX Module tree under `examples/custom-models/gpt_oss_mxfp4/`.
"""

from __future__ import annotations

from collections import OrderedDict
import os

from max.dtype import DType
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
    ]
)

_USE_GROUPED_RS = os.environ.get("MXFP4_LEGACY_GROUPED_RS", "0") == "1"


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights], **kwargs
) -> dict[str, WeightData]:
    """Convert safetensor state dict to MAX format."""

    if _USE_GROUPED_RS:
        # Reuse the v3 adapter for Hopper swizzle + packbits preprocessing.
        from gpt_oss_mxfp4_v3.weight_adapters import (
            convert_safetensor_state_dict as _convert_v3_state_dict,
        )
        has_converted = any(
            key.startswith(
                (
                    "model.layers.",
                    "model.embed_tokens.",
                    "model.norm.",
                    "lm_head.",
                )
            )
            for key in state_dict
        )
        filtered_state_dict = (
            {
                key: value
                for key, value in state_dict.items()
                if not key.startswith(
                    ("block.", "embedding.", "unembedding.", "norm.")
                )
            }
            if has_converted
            else state_dict
        )
        return _convert_v3_state_dict(filtered_state_dict, **kwargs)

    # OpenAI gpt-oss repos contain both:
    #  - "original/" weights with `block.*`/`embedding.*` naming, and
    #  - HF-converted weights with `model.*`/`lm_head.*` naming.
    # We prefer the HF-converted keys because they already match the split
    # q/k/v + MoE expert layout expected by this architecture.
    has_converted = any(
        key.startswith(
            (
                "model.layers.",
                "model.embed_tokens.",
                "model.norm.",
                "lm_head.",
            )
        )
        for key in state_dict
    )
    has_original = any(
        key.startswith(("block.", "embedding.", "unembedding.", "norm."))
        for key in state_dict
    )
    if has_original and not has_converted:
        raise ValueError(
            "Detected original GPT-OSS checkpoint layout (`block.*` keys only). "
            "This architecture expects HF-converted `model.*` safetensors "
            "with split q/k/v tensors."
        )
    has_mxfp4_experts = any(
        key.endswith(
            (
                ".mlp.experts.gate_up_proj_blocks",
                ".mlp.experts.down_proj_blocks",
            )
        )
        for key in state_dict
    )
    if not has_mxfp4_experts:
        raise ValueError(
            "Checkpoint does not contain MXFP4 expert blocks. "
            "This architecture requires OpenAI GPT-OSS MXFP4 weights "
            "(for example `openai/gpt-oss-20b`), not BF16-only variants."
        )

    new_state_dict: dict[str, WeightData] = {}
    for weight_name, value in state_dict.items():
        if has_converted and weight_name.startswith(
            ("block.", "embedding.", "unembedding.", "norm.")
        ):
            # Skip "original" weight names when converted weights exist.
            continue
        mapped = weight_name
        for before, after in GPT_OSS_SAFETENSOR_MAP.items():
            mapped = mapped.replace(before, after)
        data = value.data() if hasattr(value, "data") else value
        # MoE expert biases are BF16 in the checkpoint but the Mojo custom ops expect FP32.
        # Cast once at load time so we don't insert per-step casts into the graph.
        if mapped.endswith(
            ("experts.gate_up_proj_bias", "experts.down_proj_bias")
        ):
            data = data.astype(DType.float32)
        new_state_dict[mapped] = data
    return new_state_dict


__all__ = ["GPT_OSS_SAFETENSOR_MAP", "convert_safetensor_state_dict"]
