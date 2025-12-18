"""Small MXFP4 CPU helpers for tests and debugging.

This module is intentionally lightweight: it provides NumPy reference decode
and synthetic packed-weight generation used by `examples/custom-models/tests/`.

It is *not* the main safetensors loading path for end-to-end inference; that is
handled by MAX pipeline weight loading + `weight_adapters.py`.
"""

from __future__ import annotations

import numpy as np

MXFP4_VALUES_PER_BLOCK = 32
MXFP4_BYTES_PER_BLOCK = 16

_FP4_E2M1_TABLE = np.array(
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


def make_dummy_mxfp4_weights(
    k: int, n: int, *, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic MXFP4 packed weights for CPU-path correctness tests.

    Returns:
        blocks: uint8 array shaped [K/32, N, 16]
        scales: float32 array shaped [K/32, N]
    """
    if k % MXFP4_VALUES_PER_BLOCK != 0:
        raise ValueError("k must be divisible by 32 for MXFP4 packing")
    rng = np.random.default_rng(seed)

    k_blocks = k // MXFP4_VALUES_PER_BLOCK
    blocks = rng.integers(
        0, 256, size=(k_blocks, n, MXFP4_BYTES_PER_BLOCK), dtype=np.uint8
    )
    # Keep scales as powers-of-two so values are exactly representable in the E8M0 scheme,
    # even though the current CPU debug op takes float32 scales.
    exponents = rng.integers(-2, 3, size=(k_blocks, n), dtype=np.int32)
    scales = np.exp2(exponents.astype(np.float32))
    return blocks, scales.astype(np.float32)


def decode_mxfp4(blocks: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Decode packed MXFP4 weights to dense float32.

    Args:
        blocks: [K/32, N, 16] packed FP4(E2M1) bytes (two values per byte).
        scales: [K/32, N] float32 scales (debug path).
    Returns:
        Dense weights shaped [K, N] float32.
    """
    blocks = np.asarray(blocks, dtype=np.uint8)
    scales = np.asarray(scales, dtype=np.float32)

    if blocks.ndim != 3:
        raise ValueError(
            f"blocks must be rank-3 [K/32, N, 16], got {blocks.shape}"
        )
    if scales.ndim != 2:
        raise ValueError(f"scales must be rank-2 [K/32, N], got {scales.shape}")
    if blocks.shape[:2] != scales.shape:
        raise ValueError(
            f"blocks/scales mismatch: {blocks.shape[:2]} vs {scales.shape}"
        )
    if blocks.shape[2] != MXFP4_BYTES_PER_BLOCK:
        raise ValueError(
            f"blocks last dim must be 16 bytes, got {blocks.shape[2]}"
        )

    k_blocks, n, _ = blocks.shape
    k = k_blocks * MXFP4_VALUES_PER_BLOCK
    out = np.empty((k, n), dtype=np.float32)

    for kb in range(k_blocks):
        scale_row = scales[kb]
        blk = blocks[kb]
        for col in range(n):
            s = scale_row[col]
            for byte_idx in range(MXFP4_BYTES_PER_BLOCK):
                packed = int(blk[col, byte_idx])
                lo = packed & 0x0F
                hi = packed >> 4
                k0 = kb * MXFP4_VALUES_PER_BLOCK + 2 * byte_idx
                out[k0, col] = _FP4_E2M1_TABLE[lo] * s
                out[k0 + 1, col] = _FP4_E2M1_TABLE[hi] * s

    return out


__all__ = [
    "MXFP4_BYTES_PER_BLOCK",
    "MXFP4_VALUES_PER_BLOCK",
    "decode_mxfp4",
    "make_dummy_mxfp4_weights",
]
