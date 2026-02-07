"""Contract test for Hopper-swizzled MXFP4 value/scale decode indexing.

This isolates decode/index mapping from WGMMA. If this test passes, weight bytes
and scale bytes are being interpreted correctly and correctness drift is in the
RS fragment/epilogue path.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from gpt_oss_mxfp4.weight_adapters import (
    _mxfp4_swizzle_scales_hopper,
    _mxfp4_swizzle_values_hopper,
    _mxfp4_unpack_bits_u8,
)
from safetensors import safe_open


MXFP4_VALUES_PER_BLOCK = 32
MXFP4_BYTES_PER_BLOCK = 16

FP4_VALUES = np.array(
    [
        +0.0,
        +0.5,
        +1.0,
        +1.5,
        +2.0,
        +3.0,
        +4.0,
        +6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=np.float32,
)


def _bf16_round_to_f32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    rounding_bias = ((u >> 16) & 1).astype(np.uint32) + np.uint32(0x7FFF)
    rounded = (u + rounding_bias) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def _find_gpt_oss_20b_file() -> Path | None:
    hub_root = (
        Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        / "hub"
    )
    snapshots = hub_root / "models--openai--gpt-oss-20b" / "snapshots"
    if not snapshots.exists():
        return None
    for snap in sorted(snapshots.iterdir(), reverse=True):
        candidate = snap / "model-00000-of-00002.safetensors"
        if candidate.exists():
            return candidate
    return None


def _decode_reference(
    blocks_kbn16: np.ndarray, scales_kbn: np.ndarray
) -> np.ndarray:
    """Decode logical MXFP4 rows into dense BF16-rounded float32 [N, K]."""
    k_blocks, n_rows, _ = blocks_kbn16.shape
    k = k_blocks * MXFP4_VALUES_PER_BLOCK
    out = np.empty((n_rows, k), dtype=np.float32)
    for kb in range(k_blocks):
        for r in range(n_rows):
            scale = np.exp2(np.float32(np.int32(scales_kbn[kb, r]) - 127))
            blk = blocks_kbn16[kb, r]
            for b in range(MXFP4_BYTES_PER_BLOCK):
                packed = int(blk[b])
                lo = packed & 0x0F
                hi = packed >> 4
                k0 = kb * MXFP4_VALUES_PER_BLOCK + 2 * b
                out[r, k0] = FP4_VALUES[lo] * scale
                out[r, k0 + 1] = FP4_VALUES[hi] * scale
    return _bf16_round_to_f32(out)


def _hopper_value_swizzle_index(m: int, kbyte: int) -> tuple[int, int]:
    m0 = m >> 4
    m_rem = m & 15
    b0 = m_rem >> 3
    c0 = (m_rem & 7) >> 1
    s0 = m_rem & 1

    k0 = kbyte >> 5
    k_rem = kbyte & 31
    c1 = k_rem & 3
    b1 = (k_rem >> 2) & 1
    a1 = (k_rem >> 3) & 3

    m2 = m0 * 4 + c0
    k2 = (((((k0 * 2 + s0) * 4 + c1) * 4 + a1) * 2 + b1) * 2) + b0
    return m2, k2


def _hopper_scale_swizzle_index_fast(m: int, kblock: int) -> tuple[int, int]:
    m0 = m >> 7
    r = m & 127
    t1 = r >> 6
    w = (r >> 4) & 3
    t3 = (r >> 3) & 1
    c = r & 7

    k0 = kblock >> 1
    d = kblock & 1
    m2 = m0 * 4 + w
    k2 = (((((k0 * 2 + t1) * 8 + c) * 2 + d) * 2) + t3)
    return m2, k2


def _decode_from_swizzled_fastpath(
    swizzled_values: np.ndarray, swizzled_scales: np.ndarray, n_cols: int, k: int
) -> np.ndarray:
    """Emulate kernel fast path: load_swizzled_pack_u32 + byte_idx extraction."""
    out = np.zeros((n_cols, k), dtype=np.float32)
    for row in range(n_cols):
        for col in range(k):
            kb_rel = col >> 5
            sm2, sk2 = _hopper_scale_swizzle_index_fast(row, kb_rel)
            scale_exp = int(swizzled_scales[sm2, sk2])
            scale = np.exp2(np.float32(scale_exp - 127))

            kbyte = col >> 1
            vm2, vk2 = _hopper_value_swizzle_index(row, kbyte)
            base_k2 = vk2 & ~3
            pack4 = np.array(
                [
                    int(swizzled_values[vm2, base_k2 + 0]),
                    int(swizzled_values[vm2, base_k2 + 1]),
                    int(swizzled_values[vm2, base_k2 + 2]),
                    int(swizzled_values[vm2, base_k2 + 3]),
                ],
                dtype=np.uint8,
            )
            unpacked = _mxfp4_unpack_bits_u8(pack4)  # 4 bytes, pre-packbits
            byte_idx = vk2 & 3
            packed_byte = int(unpacked[byte_idx])
            lo = packed_byte & 0x0F
            hi = packed_byte >> 4
            out[row, col] = (FP4_VALUES[lo] if (col & 1) == 0 else FP4_VALUES[hi]) * scale
    return _bf16_round_to_f32(out)


def test_hopper_swizzled_fast_decode_contract_matches_reference() -> None:
    ckpt = _find_gpt_oss_20b_file()
    if ckpt is None:
        pytest.skip(
            "GPT-OSS checkpoint not found in HF cache; run `pixi run generate` once"
        )

    n_cols = 128
    k = 128
    k_blocks = k // MXFP4_VALUES_PER_BLOCK

    with safe_open(str(ckpt), framework="numpy") as f:
        w_blocks_all = f.get_tensor("model.layers.0.mlp.experts.down_proj_blocks")
        w_scales_all = f.get_tensor("model.layers.0.mlp.experts.down_proj_scales")

    # Expert 0 only. Raw shape from checkpoint is [E, N, K/32, 16].
    raw_blocks_nkb16 = np.ascontiguousarray(w_blocks_all[0, :n_cols, :k_blocks, :])
    raw_scales_nkb = np.ascontiguousarray(w_scales_all[0, :n_cols, :k_blocks])

    # Reference decode expects [K/32, N, 16] blocks and [K/32, N] scales.
    ref_blocks_kbn16 = np.ascontiguousarray(np.transpose(raw_blocks_nkb16, (1, 0, 2)))
    ref_scales_kbn = np.ascontiguousarray(raw_scales_nkb.T)
    ref = _decode_reference(ref_blocks_kbn16, ref_scales_kbn)

    # Hopper swizzled representations consumed by kernel.
    swz_values = _mxfp4_swizzle_values_hopper(
        raw_blocks_nkb16.reshape(1, n_cols, k_blocks * MXFP4_BYTES_PER_BLOCK),
        mx_axis=2,
    )[0]
    swz_scales = _mxfp4_swizzle_scales_hopper(raw_scales_nkb[None, ...])[0]

    got = _decode_from_swizzled_fastpath(swz_values, swz_scales, n_cols, k)
    assert np.array_equal(got, ref), (
        f"decode contract mismatch: max abs diff {np.max(np.abs(got - ref))}"
    )
