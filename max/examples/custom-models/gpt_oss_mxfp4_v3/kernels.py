"""Python wrappers for MXFP4 custom ops used by ModuleV3 MoE."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from max.dtype import DType
from max.graph import TensorType, TensorValue, ops

MXFP4_VALUES_PER_BLOCK = 32
MXFP4_BYTES_PER_BLOCK = 16


def get_mxfp4_kernels_path() -> Path:
    """Return the Mojo package path containing the registered MXFP4 custom ops."""
    examples_dir = Path(__file__).resolve().parents[2]
    return examples_dir / "custom_ops" / "kernels"


def _as_tensor(value: Any) -> TensorValue:
    return value.tensor if hasattr(value, "tensor") else value


def _try_int(dim: Any) -> int | None:
    try:
        return int(dim)
    except Exception:
        return None


def mxfp4_grouped_matmul_ragged_bf16(
    a_bf16: Any,
    w_blocks: Any,
    w_scales: Any,
    expert_start_indices: Any,
    expert_ids: Any,
    expert_usage_stats: Any,
    *,
    target: str = "gpu",
) -> TensorValue:
    """Compute grouped matmul with MXFP4 weights (BF16 in/out, FP32 accum in kernel).

    Calls `mxfp4_grouped_matmul_ragged_bf16` registered by
    `examples/custom_ops/kernels/grouped_matmul_mxfp4_ops.mojo`.
    """

    a_t = _as_tensor(a_bf16)
    w_blocks_t = _as_tensor(w_blocks)
    w_scales_t = _as_tensor(w_scales)
    expert_start_t = _as_tensor(expert_start_indices)
    expert_ids_t = _as_tensor(expert_ids)
    expert_usage_stats_t = _as_tensor(expert_usage_stats)

    if a_t.dtype != DType.bfloat16:
        raise ValueError(
            f"mxfp4_grouped_matmul_ragged_bf16 expects BF16 activations, got {a_t.dtype}"
        )
    if w_blocks_t.dtype != DType.uint8:
        raise ValueError(
            f"mxfp4_grouped_matmul_ragged_bf16 expects uint8 blocks, got {w_blocks_t.dtype}"
        )
    if w_scales_t.dtype != DType.uint8:
        raise ValueError(
            f"mxfp4_grouped_matmul_ragged_bf16 expects uint8 scales, got {w_scales_t.dtype}"
        )
    if expert_start_t.dtype != DType.uint32:
        raise ValueError(
            f"mxfp4_grouped_matmul_ragged_bf16 expects uint32 expert_start_indices, got {expert_start_t.dtype}"
        )
    if expert_ids_t.dtype != DType.int32:
        raise ValueError(
            f"mxfp4_grouped_matmul_ragged_bf16 expects int32 expert_ids, got {expert_ids_t.dtype}"
        )
    if expert_usage_stats_t.dtype != DType.uint32:
        raise ValueError(
            f"mxfp4_grouped_matmul_ragged_bf16 expects uint32 expert_usage_stats, got {expert_usage_stats_t.dtype}"
        )

    bytes_per_block = _try_int(w_blocks_t.shape[3])
    if bytes_per_block is not None and bytes_per_block != MXFP4_BYTES_PER_BLOCK:
        raise ValueError(
            f"MXFP4 packed byte dim must be 16, got {bytes_per_block}"
        )

    P = a_t.shape[0]
    K = a_t.shape[1]
    N = w_blocks_t.shape[1]

    k_blocks_dim = _try_int(w_blocks_t.shape[2])
    k_dim = _try_int(K)
    if k_dim is not None and k_blocks_dim is not None:
        if k_dim != k_blocks_dim * MXFP4_VALUES_PER_BLOCK:
            raise ValueError(
                "K must be divisible by 32 and match w_blocks Kblocks dimension"
            )

    out_type = TensorType(dtype=DType.bfloat16, shape=[P, N], device=a_t.device)

    return ops.custom(
        "mxfp4_grouped_matmul_ragged_bf16",
        device=a_t.device,
        values=[
            a_t,
            w_blocks_t,
            w_scales_t,
            expert_start_t,
            expert_ids_t,
            expert_usage_stats_t,
        ],
        out_types=[out_type],
        parameters={"target": target},
    )[0].tensor


__all__ = [
    "MXFP4_BYTES_PER_BLOCK",
    "MXFP4_VALUES_PER_BLOCK",
    "get_mxfp4_kernels_path",
    "mxfp4_grouped_matmul_ragged_bf16",
]
