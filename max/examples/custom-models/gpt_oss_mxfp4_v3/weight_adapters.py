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
import os

import numpy as np
from max.graph.weights import WeightData, Weights

# Hopper MXFP4 scale swizzle parameters (keep in sync with Mojo).
HOPPER_SCALE_NUM_WARPS = 4
HOPPER_SCALE_ALIGN_M = 32 * HOPPER_SCALE_NUM_WARPS
HOPPER_SCALE_ALIGN_K = 2

# Hopper MXFP4 value swizzle constants.
HOPPER_VALUE_MMA_VERSION = 3
_KEEP_OGS_RAW = os.environ.get("MXFP4_V3_OGS_KEEP_RAW", "") == "1"

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


def _right_shift_u32(x: np.ndarray, shift: int) -> np.ndarray:
    x = x.astype(np.uint32, copy=False)
    mask = (np.uint32(1) << np.uint32(32 - shift)) - np.uint32(1)
    return (x >> np.uint32(shift)) & mask


def _mxfp4_unpack_bits_u8(x: np.ndarray) -> np.ndarray:
    """Inverse of `_mxfp4_pack_bits_u8` (Hopper bit untwiddle)."""
    x = np.ascontiguousarray(x)
    if x.dtype != np.uint8:
        raise TypeError(f"_mxfp4_unpack_bits_u8 expects uint8, got {x.dtype}")

    x32 = x.view(np.uint32)
    m = np.uint32(0b10000001110000001000000111000000)
    a = (x32 << np.uint32(1)) & np.uint32(0b10000000000000001000000000000000)
    b = _right_shift_u32(x32, 3) & np.uint32(0b00000001100000000000000110000000)
    c = _right_shift_u32(x32, 7) & np.uint32(0b00000000010000000000000001000000)

    unpacked = np.stack(
        [x32 & m, (x32 << np.uint32(3)) & m, (x32 << np.uint32(6)) & m, a | b | c],
        axis=-1,
    )
    unpacked = unpacked.reshape(*unpacked.shape[:-2], -1)

    lo = unpacked & np.uint32(0xFFFF)
    hi = _right_shift_u32(unpacked, 16) & np.uint32(0xFFFF)

    def _bf16_to_fp4e2m1(bits: np.ndarray) -> np.ndarray:
        s = (_right_shift_u32(bits, 15) & np.uint32(0x1)) << np.uint32(3)
        em = _right_shift_u32(bits, 6) & np.uint32(0x7)
        return (s | em).astype(np.uint8, copy=False)

    lo_fp4 = _bf16_to_fp4e2m1(lo)
    hi_fp4 = _bf16_to_fp4e2m1(hi)
    return np.ascontiguousarray(lo_fp4 | (hi_fp4 << np.uint8(4)))


def _mxfp4_swizzle_values_hopper(
    values: np.ndarray,
    *,
    mx_axis: int,
    mma_version: int = HOPPER_VALUE_MMA_VERSION,
) -> np.ndarray:
    """Swizzle MXFP4 value bytes into the Hopper value layout (HOPPER_VALUE).

    Input shape: (..., M, K) packed uint8 bytes (2 FP4 per byte).
    Output shape: (..., M/4, K*4) with Hopper swizzle + packbits applied.
    """
    values = np.ascontiguousarray(values)
    if values.dtype != np.uint8:
        raise TypeError(f"_mxfp4_swizzle_values_hopper expects uint8, got {values.dtype}")
    if values.ndim < 2:
        raise ValueError("MXFP4 values must have at least 2 dimensions")
    if (values.shape[-1] % 4) != 0:
        raise ValueError("MXFP4 value bytes last dim must be divisible by 4")
    if mma_version not in (2, 3):
        raise ValueError("mma_version must be 2 or 3")

    leading = values.shape[:-2]
    data = values
    if mx_axis == len(leading):
        data = np.swapaxes(data, -1, -2)

    # We are loading 8 bf16 elements per thread to use ld.global.v4.
    u8_kwidth = 4 if mma_version == 2 else 1

    contig = (1, u8_kwidth)
    scott_trick = (2, 1)
    threads = (4, 4)
    warp_tile = (2, 2)
    k_tile = (1, 4 // u8_kwidth)

    sizes = list(data.shape[:-2])
    pads = [0, 0]
    for i, (a, b, c, s, d) in enumerate(
        zip(k_tile, warp_tile, threads, scott_trick, contig)
    ):
        pack = a * b * c * s * d
        size = data.shape[len(leading) + i]
        pad = (pack - size % pack) % pack
        pads[i] = pad
        sizes.append((size + pad) // pack)
        sizes += [a, b, c, s, d]

    if pads[0] or pads[1]:
        pad_width = [(0, 0)] * data.ndim
        pad_width[-2] = (0, pads[0])
        pad_width[-1] = (0, pads[1])
        data = np.pad(data, pad_width, mode="constant")

    data = data.reshape(*sizes)
    perm = [0, 3, 6, 10, 4, 9, 7, 1, 8, 2, 5, 11]
    perm = list(range(len(leading))) + [len(leading) + p for p in perm]
    data = np.transpose(data, perm)

    shape = data.shape
    merged = int(np.prod(shape[-10:]))
    data = data.reshape(*shape[:-10], merged)
    shape = data.shape
    data = data.reshape(*shape[:-3], shape[-3] * shape[-2], shape[-1])

    data = _mxfp4_pack_bits_u8(data)
    if mx_axis == len(leading):
        data = np.swapaxes(data, -1, -2)
    return np.ascontiguousarray(data)


def _mxfp4_unswizzle_values_hopper(
    swizzled: np.ndarray,
    *,
    mx_axis: int,
    mma_version: int = HOPPER_VALUE_MMA_VERSION,
    m: int | None = None,
    k: int | None = None,
) -> np.ndarray:
    """Invert Hopper value swizzle back to logical (..., M, K) bytes."""
    swizzled = np.ascontiguousarray(swizzled)
    if swizzled.dtype != np.uint8:
        raise TypeError(
            f"_mxfp4_unswizzle_values_hopper expects uint8, got {swizzled.dtype}"
        )
    if swizzled.ndim < 2:
        raise ValueError("MXFP4 values must have at least 2 dimensions")
    if mma_version not in (2, 3):
        raise ValueError("mma_version must be 2 or 3")

    data = swizzled
    if mx_axis == len(swizzled.shape[:-2]):
        data = np.swapaxes(data, -1, -2)

    data = _mxfp4_unpack_bits_u8(data)
    *leading, m_dim, k_dim = data.shape

    mult = 1
    u8_kwidth = 4 if mma_version == 2 else 1
    if (m_dim % 4) != 0:
        raise ValueError("M must be divisible by 4")
    stride_k = 4 * 8 * 2 * 2 * mult
    if (k_dim % stride_k) != 0:
        raise ValueError(f"K must be divisible by {stride_k}")

    data = data.reshape(
        *leading,
        m_dim // 4,
        4,
        k_dim // stride_k,
        2,
        4,
        8 // u8_kwidth,
        2,
        u8_kwidth * mult,
    )
    base = len(leading)
    perm = [0, 6, 1, 3, 2, 5, 4, 7]
    perm = list(range(base)) + [base + p for p in perm]
    data = np.transpose(data, perm)
    data = data.reshape(*leading, m_dim * 4, k_dim // 4)

    if mx_axis == len(leading):
        data = np.swapaxes(data, -1, -2)
    if m is not None:
        slicer = (slice(None),) * (data.ndim - 2) + (slice(0, m), slice(None))
        data = data[slicer]
    if k is not None:
        slicer = (slice(None),) * (data.ndim - 2) + (slice(None), slice(0, k))
        data = data[slicer]
    return np.ascontiguousarray(data)

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
    pad_k = (HOPPER_SCALE_ALIGN_K - (kblocks % HOPPER_SCALE_ALIGN_K)) % HOPPER_SCALE_ALIGN_K
    if pad_m or pad_k:
        pad = [(0, 0)] * scales.ndim
        pad[-2] = (0, pad_m)
        pad[-1] = (0, pad_k)
        scales = np.pad(scales, pad, mode="constant")

    *leading, m_pad, kblocks_pad = scales.shape
    m0 = m_pad // (32 * num_warps)
    k0 = kblocks_pad // 2

    # (..., m0, t1=2, w, t3=2, c=8, k0, d=2)
    reshaped = scales.reshape(
        *leading, m0, 2, num_warps, 2, 8, k0, 2
    )

    # permute to (..., m0, w, k0, t1, c, d, t3) then flatten.
    base = len(leading)
    perm = list(range(base)) + [base, base + 2, base + 5, base + 1, base + 4, base + 6, base + 3]
    swizzled = np.transpose(reshaped, perm)

    # flatten last 5 dims (k0, t1, c, d, t3) and then m0,w
    swizzled = swizzled.reshape(*leading, m0, num_warps, k0 * 64)
    out = swizzled.reshape(*leading, m0 * num_warps, k0 * 64)
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
    if (k2 % 64) != 0:
        raise ValueError("Swizzled scale K2 must be divisible by 64")

    m0 = m2 // num_warps
    k0 = k2 // 64
    leading = swizzled.shape[:-2]

    # (..., m0, w, k0, t1=2, c=8, d=2, t3=2)
    reshaped = swizzled.reshape(
        *leading, m0, num_warps, k0, 2, 8, 2, 2
    )

    # -> (..., m0, k0, t1, c, d, w, t3) then flatten.
    base = len(leading)
    perm = list(range(base)) + [
        base,          # m0
        base + 3,      # t1
        base + 1,      # w
        base + 6,      # t3
        base + 4,      # c
        base + 2,      # k0
        base + 5,      # d
    ]
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
    if has_converted:
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith(("block.", "embedding.", "unembedding.", "norm."))
        }

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

        # Prepack expert MXFP4 tensors to a kernel-friendly Hopper layout:
        # - values: [E, N, K/32, 16] -> [E, N/4, Kbytes*4] with value swizzle + packbits
        # - scales: [E, N, K/32] -> [E, N/32, K] with Hopper scale swizzle
        if mapped.endswith(
            ".mlp.experts.gate_up_proj_blocks"
        ) or mapped.endswith(".mlp.experts.down_proj_blocks"):
            arr = _to_numpy(weight.data)
            if arr.ndim != 4:
                raise ValueError(
                    f"Expected MXFP4 blocks rank-4, got {arr.shape}"
                )
            # Flatten Kblocks and bytes into Kbytes so swizzle can operate on
            # contiguous packed bytes.
            kbytes = arr.shape[2] * arr.shape[3]
            arr = np.ascontiguousarray(arr.reshape(arr.shape[0], arr.shape[1], kbytes))
            if _KEEP_OGS_RAW:
                raw_name = f"{mapped}_raw"
                new_state_dict[raw_name] = WeightData.from_numpy(
                    arr, name=raw_name
                )
            arr = _mxfp4_swizzle_values_hopper(arr, mx_axis=2)
            weight = WeightData.from_numpy(arr, name=mapped)
        elif mapped.endswith(
            ".mlp.experts.gate_up_proj_scales"
        ) or mapped.endswith(".mlp.experts.down_proj_scales"):
            arr = _to_numpy(weight.data)
            if _KEEP_OGS_RAW:
                raw_name = f"{mapped}_raw"
                new_state_dict[raw_name] = WeightData.from_numpy(
                    np.ascontiguousarray(arr), name=raw_name
                )
            arr = _mxfp4_swizzle_scales_hopper(arr)
            weight = WeightData.from_numpy(arr, name=mapped)

        new_state_dict[mapped] = weight
    return new_state_dict


__all__ = [
    "GPT_OSS_SAFETENSOR_MAP",
    "convert_safetensor_state_dict",
    "_mxfp4_pack_bits_u8",
    "_mxfp4_unpack_bits_u8",
    "_mxfp4_swizzle_values_hopper",
    "_mxfp4_unswizzle_values_hopper",
    "_mxfp4_swizzle_scales_hopper",
    "_mxfp4_unswizzle_scales_hopper",
]
