"""Validate Hopper MXFP4 scale swizzle against the Triton-style reference."""

from __future__ import annotations

import numpy as np

from gpt_oss_mxfp4.weight_adapters import _mxfp4_swizzle_scales_hopper


def _ref_swizzle_index(m: int, k: int, num_warps: int) -> tuple[int, int]:
    """Reference mapping from Triton hopper_scale.py (HopperMXScaleLayout)."""
    m0 = m // (32 * num_warps)
    r = m - m0 * (32 * num_warps)

    t1 = r // (num_warps * 16)
    r1 = r - t1 * (num_warps * 16)

    w = r1 // 16
    r2 = r1 - w * 16

    t3 = r2 // 8
    c = r2 - t3 * 8

    k0 = k // 2
    d = k - k0 * 2

    m2 = m0 * num_warps + w
    k2 = (((k0 * 2 + t1) * 8 + c) * 2 + d) * 2 + t3
    return m2, k2


def _check_swizzle(m: int, kblocks: int, num_warps: int) -> None:
    rng = np.random.default_rng(0)
    scales = rng.integers(0, 256, size=(m, kblocks), dtype=np.uint8)
    swz = _mxfp4_swizzle_scales_hopper(scales, num_warps=num_warps)

    for _ in range(512):
        mm = int(rng.integers(0, m))
        kk = int(rng.integers(0, kblocks))
        m2, k2 = _ref_swizzle_index(mm, kk, num_warps)
        assert swz[m2, k2] == scales[mm, kk]


def test_scale_swizzle_reference_matches() -> None:
    for num_warps in (1, 2, 4, 8):
        # Use a non-aligned M to exercise padding.
        _check_swizzle(m=32 * num_warps + 7, kblocks=6, num_warps=num_warps)
