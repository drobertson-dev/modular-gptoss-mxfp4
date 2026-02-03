"""Python wrappers for MXFP4 custom ops used by ModuleV3 MoE."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from max import functional as F
from max.dtype import DType
from max.graph import TensorType, TensorValue
from max.tensor import Tensor

MXFP4_VALUES_PER_BLOCK = 32
MXFP4_BYTES_PER_BLOCK = 16
HOPPER_SCALE_NUM_WARPS = 4


def get_mxfp4_kernels_path() -> Path:
    """Return the Mojo package path containing the registered MXFP4 custom ops."""
    examples_dir = Path(__file__).resolve().parents[2]
    # Use a minimal Mojo package to avoid importing unrelated example kernels
    # from `custom_ops/kernels/__init__.mojo`.
    return examples_dir / "custom_ops" / "mxfp4"


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
    expert_usage_stats_host: Any,
    *,
    target: str = "gpu",
    custom_extensions: str | Path | Sequence[str | Path] | None = None,
) -> Tensor | TensorValue:
    """Compute grouped matmul with MXFP4 weights (BF16 in/out, FP32 accum in kernel).

    Calls `mxfp4_grouped_matmul_ragged_bf16` registered by
    `examples/custom_ops/mxfp4/grouped_matmul_sm90_entrypoints.mojo`.
    """

    a_t = _as_tensor(a_bf16)
    w_blocks_t = _as_tensor(w_blocks)
    w_scales_t = _as_tensor(w_scales)
    expert_start_t = _as_tensor(expert_start_indices)
    expert_ids_t = _as_tensor(expert_ids)
    expert_usage_stats_host_t = _as_tensor(expert_usage_stats_host)

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
    if expert_usage_stats_host_t.dtype != DType.uint32:
        raise ValueError(
            "mxfp4_grouped_matmul_ragged_bf16 expects uint32 expert_usage_stats_host, got"
            f" {expert_usage_stats_host_t.dtype}"
        )
    host_stats_len = _try_int(expert_usage_stats_host_t.shape[0])
    if host_stats_len is not None and host_stats_len != 2:
        raise ValueError(
            f"mxfp4_grouped_matmul_ragged_bf16 expects expert_usage_stats_host shape [2], got {expert_usage_stats_host_t.shape}"
        )

    bytes_per_block = _try_int(w_blocks_t.shape[3])
    if bytes_per_block is not None and bytes_per_block != MXFP4_BYTES_PER_BLOCK:
        raise ValueError(
            f"MXFP4 packed byte dim must be 16, got {bytes_per_block}"
        )

    P = a_t.shape[0]
    K = a_t.shape[1]
    N = w_blocks_t.shape[2]

    k_blocks_dim = _try_int(w_blocks_t.shape[1])
    k_dim = _try_int(K)
    if k_dim is not None and k_blocks_dim is not None:
        if k_dim != k_blocks_dim * MXFP4_VALUES_PER_BLOCK:
            raise ValueError(
                "K must be divisible by 32 and match w_blocks Kblocks dimension"
            )
        if (k_blocks_dim % 2) != 0:
            raise ValueError(
                "Hopper scale swizzle requires Kblocks to be even (K % 64 == 0)"
            )

    scales_m2 = _try_int(w_scales_t.shape[1])
    scales_k = _try_int(w_scales_t.shape[2])
    n_dim = _try_int(N)
    if scales_m2 is not None and n_dim is not None:
        if scales_m2 * 32 < n_dim:
            raise ValueError(
                "w_scales dim1 must cover N (expected >= N/32, with padding)"
            )
        if (scales_m2 % HOPPER_SCALE_NUM_WARPS) != 0:
            raise ValueError(
                "w_scales dim1 must be a multiple of HOPPER_SCALE_NUM_WARPS"
            )
    if scales_k is not None and k_dim is not None:
        if scales_k != k_dim:
            raise ValueError(
                "w_scales dim2 must match K (Kblocks*32)"
            )

    out_type = TensorType(dtype=DType.bfloat16, shape=[P, N], device=a_t.device)
    if custom_extensions is None:
        custom_extensions = get_mxfp4_kernels_path()

    return F.custom(
        "mxfp4_grouped_matmul_ragged_bf16",
        device=a_t.device,
        values=[
            a_t,
            w_blocks_t,
            w_scales_t,
            expert_start_t,
            expert_ids_t,
            expert_usage_stats_host_t[0],
            expert_usage_stats_host_t[1],
        ],
        out_types=[out_type],
        parameters={"target": target},
        custom_extensions=custom_extensions,
    )[0]


def mxfp4_grouped_matmul_ragged_bf16_swizzled(
    a_bf16: Any,
    w_blocks: Any,
    w_scales: Any,
    expert_start_indices: Any,
    expert_ids: Any,
    expert_usage_stats_host: Any,
    *,
    n_cols: int | None = None,
    target: str = "gpu",
    custom_extensions: str | Path | Sequence[str | Path] | None = None,
) -> Tensor | TensorValue:
    """Grouped matmul with Hopper swizzled MXFP4 values + scales."""

    a_t = _as_tensor(a_bf16)
    w_blocks_t = _as_tensor(w_blocks)
    w_scales_t = _as_tensor(w_scales)
    expert_start_t = _as_tensor(expert_start_indices)
    expert_ids_t = _as_tensor(expert_ids)
    expert_usage_stats_host_t = _as_tensor(expert_usage_stats_host)

    if a_t.dtype != DType.bfloat16:
        raise ValueError(
            "mxfp4_grouped_matmul_ragged_bf16_swizzled expects BF16 activations, "
            f"got {a_t.dtype}"
        )
    if w_blocks_t.dtype != DType.uint8:
        raise ValueError(
            "mxfp4_grouped_matmul_ragged_bf16_swizzled expects uint8 blocks, "
            f"got {w_blocks_t.dtype}"
        )
    if w_scales_t.dtype != DType.uint8:
        raise ValueError(
            "mxfp4_grouped_matmul_ragged_bf16_swizzled expects uint8 scales, "
            f"got {w_scales_t.dtype}"
        )
    if expert_start_t.dtype != DType.uint32:
        raise ValueError(
            "mxfp4_grouped_matmul_ragged_bf16_swizzled expects uint32 expert_start_indices, "
            f"got {expert_start_t.dtype}"
        )
    if expert_ids_t.dtype != DType.int32:
        raise ValueError(
            "mxfp4_grouped_matmul_ragged_bf16_swizzled expects int32 expert_ids, "
            f"got {expert_ids_t.dtype}"
        )
    if expert_usage_stats_host_t.dtype != DType.uint32:
        raise ValueError(
            "mxfp4_grouped_matmul_ragged_bf16_swizzled expects uint32 expert_usage_stats_host, "
            f"got {expert_usage_stats_host_t.dtype}"
        )

    P = a_t.shape[0]
    K = a_t.shape[1]
    N = n_cols if n_cols is not None else w_blocks_t.shape[1] * 4

    k_dim = _try_int(K)
    if k_dim is not None:
        if (k_dim % MXFP4_VALUES_PER_BLOCK) != 0:
            raise ValueError("K must be divisible by 32 for MXFP4 packing")
        if ((k_dim // MXFP4_VALUES_PER_BLOCK) % 2) != 0:
            raise ValueError("Hopper swizzles require K % 64 == 0")

    w_m2 = _try_int(w_blocks_t.shape[1])
    w_k2 = _try_int(w_blocks_t.shape[2])
    if w_m2 is not None and w_m2 * 4 < N:
        raise ValueError("w_blocks dim1 must cover N (>= N/4 with padding)")
    if w_k2 is not None and k_dim is not None:
        if w_k2 < (k_dim // 2) * 4:
            raise ValueError("w_blocks dim2 must cover K bytes (>= K/2 * 4)")

    scales_m2 = _try_int(w_scales_t.shape[1])
    scales_k = _try_int(w_scales_t.shape[2])
    if scales_m2 is not None:
        if (scales_m2 % HOPPER_SCALE_NUM_WARPS) != 0:
            raise ValueError(
                "w_scales dim1 must be a multiple of HOPPER_SCALE_NUM_WARPS"
            )
    if scales_k is not None and k_dim is not None:
        if scales_k != k_dim:
            raise ValueError("w_scales dim2 must match K (Kblocks*32)")

    out_type = TensorType(dtype=DType.bfloat16, shape=[P, N], device=a_t.device)
    if custom_extensions is None:
        custom_extensions = get_mxfp4_kernels_path()

    return F.custom(
        "mxfp4_grouped_matmul_ragged_bf16_swizzled",
        device=a_t.device,
        values=[
            a_t,
            w_blocks_t,
            w_scales_t,
            expert_start_t,
            expert_ids_t,
            expert_usage_stats_host_t[0],
            expert_usage_stats_host_t[1],
        ],
        out_types=[out_type],
        parameters={"target": target},
        custom_extensions=custom_extensions,
    )[0]


__all__ = [
    "MXFP4_BYTES_PER_BLOCK",
    "MXFP4_VALUES_PER_BLOCK",
    "get_mxfp4_kernels_path",
    "mxfp4_grouped_matmul_ragged_bf16",
    "mxfp4_grouped_matmul_ragged_bf16_swizzled",
]
