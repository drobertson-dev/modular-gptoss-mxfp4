"""Weight adapters for GPT-OSS MXFP4 (ModuleV3) checkpoints.

These adapters rename HuggingFace safetensors keys into the Weight names used
by the MAX ModuleV3 tree under `examples/custom-models/gpt_oss_mxfp4_v3/`.

Key design choice:
- Keep expert weights as checkpoint-native `uint8` blocks + `uint8` scales.
- Keep expert biases as BF16 (checkpoint dtype).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable

import numpy as np
from max.graph.weights import WeightData, Weights

# Hopper MXFP4 scale swizzle parameters (keep in sync with Mojo).
HOPPER_SCALE_NUM_WARPS = 4
HOPPER_SCALE_ALIGN_M = 32 * HOPPER_SCALE_NUM_WARPS
HOPPER_SCALE_ALIGN_K = 2

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

    def _pack_two_nibbles(
        comp_fn: Callable[[np.ndarray], np.ndarray],
        b: np.ndarray,
    ) -> np.ndarray:
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


def _mxfp4_swizzle_scales_hopper(
    scales: np.ndarray,
    *,
    num_warps: int = HOPPER_SCALE_NUM_WARPS,
) -> np.ndarray:
    """Swizzle MXFP4 scales into the Hopper layout used by matmul_ogs.

    Input shape: (..., M, Kblocks) where Kblocks = K/32.
    Output shape: (..., M_pad/32, Kblocks*32) with M padded to 32*num_warps.
    """
    scales = np.ascontiguousarray(scales)
    if scales.dtype != np.uint8:
        raise TypeError(
            f"_mxfp4_swizzle_scales_hopper expects uint8, got {scales.dtype}"
        )
    if scales.ndim < 2:
        raise ValueError("MXFP4 scales must have at least 2 dimensions")

    m = int(scales.shape[-2])
    kblocks = int(scales.shape[-1])
    if (kblocks % HOPPER_SCALE_ALIGN_K) != 0:
        raise ValueError(
            "MXFP4 scale Kblocks must be divisible by 2 for Hopper swizzle"
        )

    align_m = 32 * num_warps
    m_pad = ((m + align_m - 1) // align_m) * align_m
    pad_m = m_pad - m
    if pad_m:
        pad = [(0, 0)] * scales.ndim
        pad[-2] = (0, pad_m)
        scales = np.pad(scales, pad, mode="constant")

    m0 = m_pad // (32 * num_warps)
    k0 = kblocks // 2
    leading = scales.shape[:-2]

    # (..., m0, t1=2, w, t3=2, c=8, k0, d=2)
    reshaped = scales.reshape(
        *leading, m0, 2, num_warps, 2, 8, k0, 2
    )

    # -> (..., m0, w, k0, t1, c, d, t3) then flatten.
    base = len(leading)
    perm = (
        list(range(base))
        + [base, base + 2, base + 5, base + 1, base + 4, base + 6, base + 3]
    )
    swizzled = np.transpose(reshaped, perm)
    out = swizzled.reshape(*leading, m0 * num_warps, k0 * 32)
    return np.ascontiguousarray(out)


def _mxfp4_unswizzle_scales_hopper(
    swizzled: np.ndarray,
    *,
    num_warps: int = HOPPER_SCALE_NUM_WARPS,
    m: int | None = None,
    kblocks: int | None = None,
) -> np.ndarray:
    """Invert Hopper scale swizzle back to logical (..., M, Kblocks)."""
    swizzled = np.ascontiguousarray(swizzled)
    if swizzled.dtype != np.uint8:
        raise TypeError(
            f"_mxfp4_unswizzle_scales_hopper expects uint8, got {swizzled.dtype}"
        )
    if swizzled.ndim < 2:
        raise ValueError("MXFP4 swizzled scales must have at least 2 dims")

    m2 = int(swizzled.shape[-2])
    k2 = int(swizzled.shape[-1])
    if (m2 % num_warps) != 0:
        raise ValueError("Swizzled scale M2 must be divisible by num_warps")
    if (k2 % 32) != 0:
        raise ValueError("Swizzled scale K2 must be divisible by 32")

    kblocks_pad = k2 // 32
    if (kblocks_pad % 2) != 0:
        raise ValueError("Swizzled scale Kblocks must be even")

    m0 = m2 // num_warps
    k0 = kblocks_pad // 2
    leading = swizzled.shape[:-2]

    # (..., m0, w, k0, t1=2, c=8, d=2, t3=2)
    reshaped = swizzled.reshape(
        *leading, m0, num_warps, k0, 2, 8, 2, 2
    )

    # -> (..., m0, t1, w, t3, c, k0, d) then flatten.
    base = len(leading)
    perm = (
        list(range(base))
        + [
            base,
            base + 3,
            base + 1,
            base + 6,
            base + 4,
            base + 2,
            base + 5,
        ]
    )
    unswizzled = np.transpose(reshaped, perm)
    out = unswizzled.reshape(*leading, m0 * 32 * num_warps, k0 * 2)

    if m is not None:
        slicer = (slice(None),) * (out.ndim - 2) + (slice(0, m), slice(None))
        out = out[slicer]
    if kblocks is not None:
        slicer = (slice(None),) * (out.ndim - 2) + (slice(None), slice(0, kblocks))
        out = out[slicer]

    return np.ascontiguousarray(out)


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
        # [E, N, K/32, 16] -> [E, K/32, N, 16] for values and
        # [E, N, K/32] -> [E, N/32, K] with Hopper scale swizzle.
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
            arr = _mxfp4_swizzle_scales_hopper(arr)
            weight = WeightData.from_numpy(arr, name=mapped)

        new_state_dict[mapped] = weight
    return new_state_dict


__all__ = [
    "GPT_OSS_SAFETENSOR_MAP",
    "convert_safetensor_state_dict",
    "_mxfp4_pack_bits_u8",
    "_mxfp4_swizzle_scales_hopper",
    "_mxfp4_unswizzle_scales_hopper",
]
