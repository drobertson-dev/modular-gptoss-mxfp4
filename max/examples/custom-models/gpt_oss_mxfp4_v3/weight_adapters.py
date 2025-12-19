"""Weight adapters for GPT-OSS MXFP4 (ModuleV3) checkpoints.

These adapters rename HuggingFace safetensors keys into the Weight names used
by the MAX ModuleV3 tree under `examples/custom-models/gpt_oss_mxfp4_v3/`.

Key design choice:
- Keep expert weights as checkpoint-native `uint8` blocks + `uint8` scales.
- Keep expert biases as BF16 (checkpoint dtype).
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
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


def _mxfp4_pack_bits_u8(x: np.ndarray) -> np.ndarray:
    """Apply the Hopper MXFP4 `_pack_bits` transform in numpy.

    This is the local (per 4 bytes) bit-interleaving trick used by Triton
    `matmul_ogs` to enable a very cheap FP4->BF16 unpack sequence.

    Input/Output:
      - dtype: uint8
      - shape: unchanged
      - last dimension must be divisible by 4
    """
    x = np.ascontiguousarray(x)
    if x.dtype != np.uint8:
        raise TypeError(f"_mxfp4_pack_bits_u8 expects uint8, got {x.dtype}")
    if (x.shape[-1] % 4) != 0:
        raise ValueError("MXFP4 packed bytes last dim must be divisible by 4")

    x2 = x.reshape(x.shape[:-1] + (x.shape[-1] // 4, 4))

    def _compress_fp4(b: np.ndarray) -> np.ndarray:
        b = b.astype(np.uint32, copy=False)
        return ((b & np.uint32(0x8)) << np.uint32(12)) | (
            (b & np.uint32(0x7)) << np.uint32(6)
        )

    def _compress_fourth(b: np.ndarray) -> np.ndarray:
        b = b.astype(np.uint32, copy=False)
        return (
            ((b & np.uint32(0x8)) << np.uint32(11))
            | ((b & np.uint32(0x6)) << np.uint32(9))
            | ((b & np.uint32(0x1)) << np.uint32(13))
        )

    def _pack_two_nibbles(comp_fn, b: np.ndarray) -> np.ndarray:
        lo = comp_fn(b)
        hi = comp_fn((b >> np.uint32(4)) & np.uint32(0xF))
        return lo | (hi << np.uint32(16))

    # Shape: (..., groups)
    packed = _pack_two_nibbles(_compress_fp4, x2[..., 0])
    packed |= _pack_two_nibbles(_compress_fp4, x2[..., 1]) >> np.uint32(3)
    packed |= _pack_two_nibbles(_compress_fp4, x2[..., 2]) >> np.uint32(6)
    packed |= _pack_two_nibbles(_compress_fourth, x2[..., 3])

    # View as uint8 to restore the original last dimension (4 bytes per uint32).
    return packed.view(np.uint8).reshape(x.shape)


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights], **kwargs
) -> dict[str, WeightData]:
    """Convert safetensor state dict to MAX format."""

    _ = kwargs

    def _to_numpy(arr) -> np.ndarray:  # noqa: ANN001
        if isinstance(arr, np.ndarray):
            return arr
        try:
            return np.from_dlpack(arr)
        except Exception:
            return np.asarray(arr)

    new_state_dict: dict[str, WeightData] = {}
    for weight_name, value in state_dict.items():
        mapped = weight_name
        for before, after in GPT_OSS_SAFETENSOR_MAP.items():
            mapped = mapped.replace(before, after)

        weight = value.data() if hasattr(value, "data") else value
        if not isinstance(weight, WeightData):
            weight = WeightData.from_numpy(_to_numpy(weight), name=mapped)
        elif weight.name != mapped:
            weight = WeightData(
                data=weight.data,
                name=mapped,
                dtype=weight.dtype,
                shape=weight.shape,
                quantization_encoding=weight.quantization_encoding,
            )

        # Prepack expert MXFP4 tensors to a kernel-friendly layout:
        # [E, N, K/32, 16] -> [E, K/32, N, 16] and [E, N, K/32] -> [E, K/32, N].
        if mapped.endswith(
            ".mlp.experts.gate_up_proj_blocks"
        ) or mapped.endswith(".mlp.experts.down_proj_blocks"):
            arr = _to_numpy(weight.data)
            arr = np.ascontiguousarray(np.transpose(arr, (0, 2, 1, 3)))
            arr = _mxfp4_pack_bits_u8(arr)
            weight = WeightData.from_numpy(arr, name=mapped)
        elif mapped.endswith(
            ".mlp.experts.gate_up_proj_scales"
        ) or mapped.endswith(".mlp.experts.down_proj_scales"):
            arr = _to_numpy(weight.data)
            arr = np.ascontiguousarray(np.transpose(arr, (0, 2, 1)))
            weight = WeightData.from_numpy(arr, name=mapped)

        new_state_dict[mapped] = weight
    return new_state_dict


__all__ = ["GPT_OSS_SAFETENSOR_MAP", "convert_safetensor_state_dict"]
