"""Python wrappers for MXFP4 Mojo custom ops.

These wrappers exist to centralize:
- the registered custom op names
- shape/dtype expectations (Python must match Mojo, not vice versa)
- the `custom_extensions` lookup path for MAX Graph construction
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from max.dtype import DType
from max.graph import DeviceRef, TensorType, TensorValue, ops

MXFP4_VALUES_PER_BLOCK = 32
MXFP4_BYTES_PER_BLOCK = 16
MXFP4_TOPK = 4


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


def mxfp4_matmul_swiglu(
    a: Any,
    b_packed: Any,
    b_scales: Any,
    bias: Any,
    *,
    alpha: float = 1.702,
    limit: float = 7.0,
    target: str = "cpu",
    custom_extensions: list[str] | None = None,  # compatibility; Graph owns this
) -> TensorValue:
    """Call `gpt_oss.mxfp4.matmul.sm90` (currently CPU-only correctness op).

    Shapes/dtypes must match `examples/custom_ops/kernels/mxfp4_matmul_sm90.mojo`.
    """

    del custom_extensions  # Graph-level `custom_extensions` controls op loading.

    a_t = _as_tensor(a)
    b_packed_t = _as_tensor(b_packed)
    b_scales_t = _as_tensor(b_scales)
    bias_t = _as_tensor(bias)

    m_dim, k_dim = a_t.shape
    k_blocks_dim, n_full_dim, bytes_per_block_dim = b_packed_t.shape

    bytes_per_block = _try_int(bytes_per_block_dim)
    k = _try_int(k_dim)
    k_blocks = _try_int(k_blocks_dim)
    n_full = _try_int(n_full_dim)

    if bytes_per_block is not None and bytes_per_block != MXFP4_BYTES_PER_BLOCK:
        raise ValueError(f"MXFP4 packed byte dim must be 16, got {bytes_per_block}")
    if k is not None and k_blocks is not None and k != k_blocks * MXFP4_VALUES_PER_BLOCK:
        raise ValueError("K must be divisible by 32 (MXFP4 block)")
    if n_full is not None and n_full % 2 != 0:
        raise ValueError("N must be even (interleaved gate/up columns)")

    if n_full is None:
        raise ValueError("mxfp4_matmul_swiglu requires a statically known N dimension")
    out_type = TensorType(
        dtype=a_t.dtype,
        shape=[m_dim, n_full // 2],
        device=a_t.device,
    )
    alpha_val = ops.constant(alpha, dtype=DType.float32, device=DeviceRef.CPU())
    limit_val = ops.constant(limit, dtype=DType.float32, device=DeviceRef.CPU())
    return ops.custom(
        "gpt_oss.mxfp4.matmul.sm90",
        device=a_t.device,
        values=[a_t, b_packed_t, b_scales_t, bias_t, alpha_val, limit_val],
        out_types=[out_type],
        parameters={"target": target},
    )[0].tensor


def mxfp4_moe_w1_swiglu(
    x_bf16: Any,
    token_expert_order: Any,
    expert_start_indices: Any,
    expert_ids: Any,
    expert_usage_stats: Any,
    w_blocks: Any,
    w_scales: Any,
    bias_f32: Any,
    *,
    alpha: float = 1.702,
    limit: float = 7.0,
    target: str = "gpu",
) -> TensorValue:
    """Call `mxfp4_moe_w1_swiglu` (GPU-only correctness-first MoE W1+SwiGLU)."""

    x_t = _as_tensor(x_bf16)
    token_expert_order_t = _as_tensor(token_expert_order)
    expert_start_indices_t = _as_tensor(expert_start_indices)
    expert_ids_t = _as_tensor(expert_ids)
    expert_usage_stats_t = _as_tensor(expert_usage_stats)
    w_blocks_t = _as_tensor(w_blocks)
    w_scales_t = _as_tensor(w_scales)
    bias_t = _as_tensor(bias_f32)

    p = token_expert_order_t.shape[0]
    bias_cols = _try_int(bias_t.shape[1])
    if bias_cols is None:
        raise ValueError("mxfp4_moe_w1_swiglu requires a statically known bias width")
    i = bias_cols // 2

    out_type = TensorType(dtype=DType.bfloat16, shape=[p, i], device=x_t.device)
    alpha_val = ops.constant(alpha, dtype=DType.float32, device=DeviceRef.CPU())
    limit_val = ops.constant(limit, dtype=DType.float32, device=DeviceRef.CPU())

    return ops.custom(
        "mxfp4_moe_w1_swiglu",
        device=x_t.device,
        values=[
            x_t,
            token_expert_order_t,
            expert_start_indices_t,
            expert_ids_t,
            expert_usage_stats_t,
            w_blocks_t,
            w_scales_t,
            bias_t,
            alpha_val,
            limit_val,
        ],
        out_types=[out_type],
        parameters={"target": target},
    )[0].tensor


def mxfp4_moe_w2_pairs_bf16(
    x_like: Any,
    h_sorted: Any,
    token_expert_order: Any,
    expert_start_indices: Any,
    expert_ids: Any,
    expert_usage_stats: Any,
    gate_weights_bf16: Any,
    w_blocks: Any,
    w_scales: Any,
    bias_f32: Any,
    *,
    target: str = "gpu",
) -> TensorValue:
    """Call `mxfp4_moe_w2_pairs_bf16` (GPU-only MoE W2 writing BF16 y_pairs)."""

    x_t = _as_tensor(x_like)
    h_sorted_t = _as_tensor(h_sorted)
    token_expert_order_t = _as_tensor(token_expert_order)
    expert_start_indices_t = _as_tensor(expert_start_indices)
    expert_ids_t = _as_tensor(expert_ids)
    expert_usage_stats_t = _as_tensor(expert_usage_stats)
    gate_weights_t = _as_tensor(gate_weights_bf16)
    w_blocks_t = _as_tensor(w_blocks)
    w_scales_t = _as_tensor(w_scales)
    bias_t = _as_tensor(bias_f32)

    if gate_weights_t.dtype != DType.bfloat16:
        raise ValueError(f"mxfp4_moe_w2_pairs_bf16 expects BF16 gate weights, got {gate_weights_t.dtype}")

    p_pairs = token_expert_order_t.shape[0]
    d_hidden = x_t.shape[1]

    out_type = TensorType(dtype=DType.bfloat16, shape=[p_pairs, d_hidden], device=x_t.device)

    return ops.custom(
        "mxfp4_moe_w2_pairs_bf16",
        device=x_t.device,
        values=[
            h_sorted_t,
            token_expert_order_t,
            expert_start_indices_t,
            expert_ids_t,
            expert_usage_stats_t,
            gate_weights_t,
            w_blocks_t,
            w_scales_t,
            bias_t,
        ],
        out_types=[out_type],
        parameters={"target": target},
    )[0].tensor


def mxfp4_moe_topk_reduce_bf16(
    x_like: Any,
    y_pairs: Any,
    *,
    target: str = "gpu",
) -> TensorValue:
    """Call `mxfp4_moe_topk_reduce_bf16` (GPU-only TOPK reduction, BF16 in/out)."""

    x_t = _as_tensor(x_like)
    y_pairs_t = _as_tensor(y_pairs)

    t_tokens = x_t.shape[0]
    d_hidden = x_t.shape[1]

    out_type = TensorType(dtype=DType.bfloat16, shape=[t_tokens, d_hidden], device=x_t.device)

    return ops.custom(
        "mxfp4_moe_topk_reduce_bf16",
        device=x_t.device,
        values=[y_pairs_t],
        out_types=[out_type],
        parameters={"target": target},
    )[0].tensor


__all__ = [
    "MXFP4_BYTES_PER_BLOCK",
    "MXFP4_TOPK",
    "MXFP4_VALUES_PER_BLOCK",
    "get_mxfp4_kernels_path",
    "mxfp4_matmul_swiglu",
    "mxfp4_moe_topk_reduce_bf16",
    "mxfp4_moe_w1_swiglu",
    "mxfp4_moe_w2_pairs_bf16",
]
