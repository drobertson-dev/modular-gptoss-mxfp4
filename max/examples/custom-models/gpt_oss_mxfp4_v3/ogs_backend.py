"""Eager-only OGS (Triton matmul_ogs) backend for MXFP4 MoE.

This is intended for profiling/benchmarking against the vLLM OGS path. It is
NOT graph-compilable and requires torch + triton_kernels to be available in the
Python environment.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

from max.tensor import Tensor

_TRITON_KERNELS_PATH = os.environ.get(
    "MXFP4_V3_TRITON_KERNELS_PATH", "/workspace/triton/python/triton_kernels"
)


def _ensure_triton_kernels() -> None:
    if _TRITON_KERNELS_PATH and _TRITON_KERNELS_PATH not in sys.path:
        if os.path.isdir(_TRITON_KERNELS_PATH):
            sys.path.insert(0, _TRITON_KERNELS_PATH)


def _import_ogs() -> Any:
    _ensure_triton_kernels()
    import torch
    import triton  # noqa: F401
    import triton.language as tl
    from triton_kernels import swiglu
    from triton_kernels.matmul_ogs import (
        FlexCtx,
        FnSpecs,
        FusedActivation,
        PrecisionConfig,
        matmul_ogs,
    )
    from triton_kernels.matmul_ogs_details import opt_flags
    from triton_kernels.numerics import InFlexData
    from triton_kernels.routing import routing_from_bitmatrix
    from triton_kernels.tensor import FP4, Tensor as TKTensor, convert_layout, wrap_torch_tensor
    from triton_kernels.tensor_details import layout

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if major == 9:
            opt_flags.update_opt_flags_constraints({"split_k": 1})
        elif major >= 10:
            opt_flags.update_opt_flags_constraints(
                {"is_persistent": True, "epilogue_subtile": 1}
            )

    return {
        "torch": torch,
        "triton": triton,
        "tl": tl,
        "swiglu": swiglu,
        "FlexCtx": FlexCtx,
        "FnSpecs": FnSpecs,
        "FusedActivation": FusedActivation,
        "PrecisionConfig": PrecisionConfig,
        "matmul_ogs": matmul_ogs,
        "opt_flags": opt_flags,
        "InFlexData": InFlexData,
        "routing_from_bitmatrix": routing_from_bitmatrix,
        "FP4": FP4,
        "TKTensor": TKTensor,
        "convert_layout": convert_layout,
        "wrap_torch_tensor": wrap_torch_tensor,
        "layout": layout,
    }


def _torch_from_dlpack(t: Tensor) -> Any:
    import torch
    from torch.utils.dlpack import from_dlpack

    return from_dlpack(t.__dlpack__()).contiguous()


def _maybe_float32(t: Any) -> Any:
    import torch

    return t if t.dtype == torch.float32 else t.to(torch.float32)


def _swizzle_mxfp4(quant_tensor: Any, scale: Any, *, num_warps: int) -> tuple[Any, Any, Any]:
    ogs = _import_ogs()
    layout = ogs["layout"]
    wrap_torch_tensor = ogs["wrap_torch_tensor"]
    convert_layout = ogs["convert_layout"]
    FP4 = ogs["FP4"]
    InFlexData = ogs["InFlexData"]

    value_layout, value_layout_opts = layout.make_default_matmul_mxfp4_w_layout(
        mx_axis=1
    )
    scale_layout, scale_layout_opts = layout.make_default_matmul_mxfp4_w_scale_layout(
        mx_axis=1, num_warps=num_warps
    )

    # transpose so the quantization axis is on dim=1
    quant_tensor = quant_tensor.transpose(-2, -1)
    scale = scale.transpose(-2, -1)

    quant_tensor = convert_layout(
        wrap_torch_tensor(quant_tensor, dtype=FP4),
        value_layout,
        **value_layout_opts,
    )
    scale = convert_layout(
        wrap_torch_tensor(scale),
        scale_layout,
        **scale_layout_opts,
    )
    return quant_tensor, InFlexData(), scale


def _make_routing_data(
    topk_ids: Any, topk_weights: Any, *, num_experts: int, num_topk: int
) -> tuple[Any, Any, Any]:
    ogs = _import_ogs()
    torch = ogs["torch"]
    triton = ogs["triton"]
    routing_from_bitmatrix = ogs["routing_from_bitmatrix"]
    TKTensor = ogs["TKTensor"]

    tl = ogs["tl"]

    @ogs["triton"].jit
    def pack_bitmatrix(
        bitmatrix,
        topk_ids,
        n_rows,
        bm_cols: tl.constexpr,
        n_expts_act,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offsets_k = tl.arange(0, BLOCK_SIZE_K)
        offsets = offsets_m[:, None] * n_expts_act + offsets_k[None, :]
        mask = (offsets_m < n_rows)[:, None] & (offsets_k < n_expts_act)[None, :]
        indices = tl.load(topk_ids + offsets, mask=mask, other=-1)
        div = indices // 32
        rem = indices % 32
        one = tl.full([], 1, tl.uint32)
        for i in range(bm_cols):
            offs = tl.arange(0, BLOCK_SIZE_K // 32) + i * (BLOCK_SIZE_K // 32)
            x = tl.where(
                div[:, :, None] == offs[None, None, :],
                (one << rem)[:, :, None],
                0,
            )
            y = tl.reduce_or(x, axis=1)
            bitmatrix_ptrs = bitmatrix + offsets_m[:, None] * bm_cols + offs[None, :]
            tl.store(bitmatrix_ptrs, y, mask=offsets_m[:, None] < n_rows)

    topk_ids = topk_ids.to(torch.int16)
    topk_weights = topk_weights.to(torch.bfloat16)
    n_rows = topk_ids.shape[0]

    BLOCK_SIZE_M = 512
    BLOCK_SIZE_K = 32
    bm_cols = triton.cdiv(num_experts, BLOCK_SIZE_K)
    bitmatrix = torch.zeros(
        (n_rows, bm_cols), dtype=torch.uint32, device=topk_ids.device
    )
    grid = (triton.cdiv(n_rows, BLOCK_SIZE_M),)
    pack_bitmatrix[grid](
        bitmatrix,
        topk_ids,
        n_rows,
        bm_cols,
        num_topk,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    bitmatrix_shape = [n_rows, bm_cols * 32]
    bitmatrix_shape_max = [n_rows, None]
    from triton_kernels.tensor import Bitmatrix

    bitmatrix = Bitmatrix(
        bitmatrix, shape=bitmatrix_shape, shape_max=bitmatrix_shape_max, scratchpad=None
    )

    # matmul_ogs expects invalid topk_weights to be -1
    topk_weights = torch.where(topk_ids == -1, -1.0, topk_weights)
    return routing_from_bitmatrix(
        bitmatrix, topk_weights, topk_ids, num_experts, num_topk
    )


@dataclass
class _OgsWeights:
    w1: Any
    w2: Any
    pc1: Any
    pc2: Any


def _build_ogs_weights(
    *,
    w1_blocks_raw: Tensor,
    w1_scales_raw: Tensor,
    w2_blocks_raw: Tensor,
    w2_scales_raw: Tensor,
    num_warps: int,
) -> _OgsWeights:
    ogs = _import_ogs()
    torch = ogs["torch"]
    FlexCtx = ogs["FlexCtx"]
    PrecisionConfig = ogs["PrecisionConfig"]
    InFlexData = ogs["InFlexData"]

    w1_blocks = _torch_from_dlpack(w1_blocks_raw)
    w1_scales = _torch_from_dlpack(w1_scales_raw)
    w2_blocks = _torch_from_dlpack(w2_blocks_raw)
    w2_scales = _torch_from_dlpack(w2_scales_raw)

    w1_weight, w1_flex, w1_scale = _swizzle_mxfp4(
        w1_blocks, w1_scales, num_warps=num_warps
    )
    w2_weight, w2_flex, w2_scale = _swizzle_mxfp4(
        w2_blocks, w2_scales, num_warps=num_warps
    )

    pc1 = PrecisionConfig(
        weight_scale=w1_scale, flex_ctx=FlexCtx(rhs_data=w1_flex)
    )
    pc2 = PrecisionConfig(
        weight_scale=w2_scale, flex_ctx=FlexCtx(rhs_data=w2_flex)
    )

    return _OgsWeights(w1=w1_weight, w2=w2_weight, pc1=pc1, pc2=pc2)


def ogs_moe_forward(
    *,
    x: Tensor,
    gate_weight: Tensor,
    gate_bias: Tensor,
    w1_blocks_raw: Tensor,
    w1_scales_raw: Tensor,
    w1_bias: Tensor,
    w2_blocks_raw: Tensor,
    w2_scales_raw: Tensor,
    w2_bias: Tensor,
    topk: int,
    swiglu_alpha: float,
    swiglu_limit: float,
    num_experts: int,
    num_warps: int,
    _cache: dict[str, Any] | None = None,
) -> Tensor:
    ogs = _import_ogs()
    torch = ogs["torch"]
    FnSpecs = ogs["FnSpecs"]
    FusedActivation = ogs["FusedActivation"]
    matmul_ogs = ogs["matmul_ogs"]
    swiglu = ogs["swiglu"]

    if _cache is None:
        _cache = {}

    if "weights" not in _cache:
        _cache["weights"] = _build_ogs_weights(
            w1_blocks_raw=w1_blocks_raw,
            w1_scales_raw=w1_scales_raw,
            w2_blocks_raw=w2_blocks_raw,
            w2_scales_raw=w2_scales_raw,
            num_warps=num_warps,
        )

    weights: _OgsWeights = _cache["weights"]

    x_torch = _torch_from_dlpack(x)
    if "gate_w" not in _cache:
        _cache["gate_w"] = _torch_from_dlpack(gate_weight)
        _cache["gate_b"] = _torch_from_dlpack(gate_bias)
    gate_w = _cache["gate_w"].to(x_torch.dtype)
    gate_b = _cache["gate_b"].to(x_torch.dtype)

    if "w1_bias" not in _cache:
        _cache["w1_bias"] = _maybe_float32(_torch_from_dlpack(w1_bias))
        _cache["w2_bias"] = _maybe_float32(_torch_from_dlpack(w2_bias))
    w1_bias_t = _cache["w1_bias"]
    w2_bias_t = _cache["w2_bias"]

    # Router: topk + softmax (renormalize)
    logits = torch.nn.functional.linear(x_torch, gate_w, gate_b)
    topk_vals, topk_ids = torch.topk(logits, k=topk, dim=-1)
    topk_weights = torch.softmax(topk_vals, dim=-1)

    routing_data, gather_idx, scatter_idx = _make_routing_data(
        topk_ids, topk_weights, num_experts=num_experts, num_topk=topk
    )

    fused_act = FusedActivation(
        FnSpecs("swiglu", swiglu.swiglu_fn, ("alpha", "limit")),
        (swiglu_alpha, swiglu_limit),
        2,
    )

    # W1 + SwiGLU
    intermediate = matmul_ogs(
        x_torch,
        weights.w1,
        w1_bias_t,
        routing_data,
        gather_indx=gather_idx,
        precision_config=weights.pc1,
        gammas=None,
        fused_activation=fused_act,
    )

    # W2 + reduce
    if intermediate.ndim == 3:
        inter2d = intermediate.view(
            intermediate.shape[1], intermediate.shape[2]
        ).contiguous()
    else:
        inter2d = intermediate

    out = matmul_ogs(
        inter2d,
        weights.w2,
        w2_bias_t,
        routing_data,
        scatter_indx=scatter_idx,
        precision_config=weights.pc2,
        gammas=routing_data.gate_scal,
    )
    out2d = (
        out.view(out.shape[1], out.shape[2]) if out.ndim == 3 else out
    )
    return Tensor.from_dlpack(out2d.contiguous())


__all__ = ["ogs_moe_forward"]
