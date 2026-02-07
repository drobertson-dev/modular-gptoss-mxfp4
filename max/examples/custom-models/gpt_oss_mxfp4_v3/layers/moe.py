"""GPT-OSS MoE layer swapping expert GEMMs to MXFP4 grouped matmul."""

from __future__ import annotations

import os
from typing import Any, cast

from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.tensor import Tensor
from max.graph import ops
from max.nn import Linear
from max.nn.legacy.kernels import scatter_nd_skip_oob_indices
from max.nn.module import Module
from max.pipelines.architectures.gpt_oss.layers.functional_kernels import (
    moe_create_indices,
)
from max.pipelines.architectures.gpt_oss.layers.moe_base import (
    MoE,
    MoEGate,
)

from gpt_oss_mxfp4_v3.kernels import (
    MXFP4_VALUES_PER_BLOCK,
)
from gpt_oss_mxfp4_v3.kernels import (
    mxfp4_grouped_matmul_ragged_bf16_swizzled as _mxfp4_grouped_matmul_ragged_bf16,
)
from gpt_oss_mxfp4_v3.model_config import GptOssConfig

mxfp4_grouped_matmul_ragged_bf16 = F.functional(
    _mxfp4_grouped_matmul_ragged_bf16
)

HOPPER_SCALE_NUM_WARPS = 4
HOPPER_SCALE_ALIGN_M = 32 * HOPPER_SCALE_NUM_WARPS

_MOE_DEBUG = os.environ.get("MXFP4_V3_MOE_DEBUG", "") == "1"
_MOE_DEBUG_LAYER = int(os.environ.get("MXFP4_V3_MOE_DEBUG_LAYER", "-1"))
_MOE_NAN_SCAN = os.environ.get("MXFP4_V3_MOE_NAN_SCAN", "") == "1"
_MOE_NAN_CLAMP = os.environ.get("MXFP4_V3_MOE_NAN_CLAMP", "") == "1"
_USE_OGS = os.environ.get("MXFP4_V3_USE_OGS", "") == "1"
_KEEP_OGS_RAW = os.environ.get("MXFP4_V3_OGS_KEEP_RAW", "") == "1"


class GptOssMoEGate(MoEGate):
    """GptOss-style Gate module for MoE with bias support."""

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
        )
        self.gate_score = Linear(
            in_dim=hidden_dim,
            out_dim=num_experts,
            bias=True,
        )

    def forward(self, hidden_state: Tensor) -> tuple[Tensor, Tensor]:
        scores = self.gate_score(hidden_state)
        topk_scores, topk_indices = F.top_k(
            scores, k=self.num_experts_per_token, axis=-1
        )
        topk_scores = F.softmax(topk_scores)
        return topk_indices, topk_scores


class MXFP4Experts(Module):
    """Holds MXFP4 expert weights in checkpoint-compatible names."""

    def __init__(
        self, *, num_experts: int, hidden_dim: int, moe_dim: int
    ) -> None:
        super().__init__()
        if hidden_dim % MXFP4_VALUES_PER_BLOCK != 0:
            raise ValueError(
                "hidden_dim must be divisible by 32 for MXFP4 packing"
            )
        if moe_dim % MXFP4_VALUES_PER_BLOCK != 0:
            raise ValueError(
                "intermediate_size must be divisible by 32 for MXFP4 packing"
            )
        if (hidden_dim // MXFP4_VALUES_PER_BLOCK) % 2 != 0:
            raise ValueError(
                "hidden_dim must be divisible by 64 for Hopper scale swizzle"
            )
        if (moe_dim // MXFP4_VALUES_PER_BLOCK) % 2 != 0:
            raise ValueError(
                "intermediate_size must be divisible by 64 for Hopper scale swizzle"
            )

        kblocks_w1 = hidden_dim // MXFP4_VALUES_PER_BLOCK
        kblocks_w2 = moe_dim // MXFP4_VALUES_PER_BLOCK

        def _value_swizzle_shape(m: int, k: int) -> tuple[int, int]:
            # Matches _mxfp4_swizzle_values_hopper padding for mma v3:
            # M padded to 16, Kbytes padded to 32.
            kbytes = k // 2
            m_pad = ((m + 15) // 16) * 16
            k_pad = ((kbytes + 31) // 32) * 32
            return m_pad // 4, k_pad * 4

        n_pad_w1 = (
            (2 * moe_dim + HOPPER_SCALE_ALIGN_M - 1)
            // HOPPER_SCALE_ALIGN_M
        ) * HOPPER_SCALE_ALIGN_M
        scale_m2_w1 = n_pad_w1 // 32

        # W1 (prepacked): [E, (2*I)/4, (D/2)*4] swizzled value bytes +
        # [E, ceildiv(2*I, 32*num_warps)*num_warps, D] scales + BF16 bias.
        # This matches the SM90 kernel's preferred access pattern: fixed `kb` reads
        # contiguous columns.
        w1_m2, w1_k2 = _value_swizzle_shape(2 * moe_dim, hidden_dim)
        self.gate_up_proj_blocks = Tensor.zeros(
            shape=[num_experts, w1_m2, w1_k2],
            dtype=DType.uint8,
        )
        self.gate_up_proj_scales = Tensor.zeros(
            shape=[num_experts, scale_m2_w1, hidden_dim],
            dtype=DType.uint8,
        )
        self.gate_up_proj_bias = Tensor.zeros(
            shape=[num_experts, 2 * moe_dim],
            dtype=DType.bfloat16,
        )

        n_pad_w2 = (
            (hidden_dim + HOPPER_SCALE_ALIGN_M - 1) // HOPPER_SCALE_ALIGN_M
        ) * HOPPER_SCALE_ALIGN_M
        scale_m2_w2 = n_pad_w2 // 32

        # W2 (prepacked): [E, D/4, (I/2)*4] swizzled value bytes +
        # [E, ceildiv(D, 32*num_warps)*num_warps, I] scales + BF16 bias.
        w2_m2, w2_k2 = _value_swizzle_shape(hidden_dim, moe_dim)
        self.down_proj_blocks = Tensor.zeros(
            shape=[num_experts, w2_m2, w2_k2],
            dtype=DType.uint8,
        )
        self.down_proj_scales = Tensor.zeros(
            shape=[num_experts, scale_m2_w2, moe_dim],
            dtype=DType.uint8,
        )
        self.down_proj_bias = Tensor.zeros(
            shape=[num_experts, hidden_dim],
            dtype=DType.bfloat16,
        )

        if _KEEP_OGS_RAW:
            kbytes_w1 = hidden_dim // 2
            kbytes_w2 = moe_dim // 2
            # Raw (unswizzled) MXFP4 blocks/scales for OGS layout conversion.
            self.gate_up_proj_blocks_raw = Tensor.zeros(
                shape=[num_experts, 2 * moe_dim, kbytes_w1],
                dtype=DType.uint8,
            )
            self.gate_up_proj_scales_raw = Tensor.zeros(
                shape=[num_experts, 2 * moe_dim, kblocks_w1],
                dtype=DType.uint8,
            )
            self.down_proj_blocks_raw = Tensor.zeros(
                shape=[num_experts, hidden_dim, kbytes_w2],
                dtype=DType.uint8,
            )
            self.down_proj_scales_raw = Tensor.zeros(
                shape=[num_experts, hidden_dim, kblocks_w2],
                dtype=DType.uint8,
            )


class GptOssMoE(MoE):
    """GptOss-style MoE that swaps expert GEMMs to MXFP4 grouped matmul."""

    def __init__(self, config: GptOssConfig, *, layer_idx: int | None = None):
        self.alpha = 1.702
        self.limit = float(getattr(config, "swiglu_limit", 7.0))
        self.config = config
        self.layer_idx = layer_idx

        super().__init__(
            hidden_dim=config.hidden_size,
            num_experts=config.num_local_experts,
            num_experts_per_token=config.num_experts_per_tok,
            moe_dim=config.intermediate_size,
            gate_cls=GptOssMoEGate,
            has_shared_experts=False,
            ep_size=1,
            apply_router_weight_first=False,
        )

    def _init_experts(self) -> None:
        # Replace the dense expert weights with MXFP4 blocks/scales under a child
        # module named `experts` so checkpoint keys map directly.
        self.experts = cast(
            Any,
            MXFP4Experts(
                num_experts=self.num_experts,
                hidden_dim=self.hidden_dim,
                moe_dim=self.moe_dim,
            ),
        )

    def _mxfp4_experts(self) -> MXFP4Experts:
        return cast(MXFP4Experts, self.experts)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[0]

        x_bf16 = x if x.dtype == DType.bfloat16 else F.cast(x, DType.bfloat16)

        if _USE_OGS:
            if F.in_graph_context():
                raise RuntimeError(
                    "MXFP4_V3_USE_OGS requires eager execution (no graph context)."
                )
            experts = self._mxfp4_experts()
            if not hasattr(experts, "gate_up_proj_blocks_raw"):
                raise RuntimeError(
                    "OGS path requires raw MXFP4 weights. "
                    "Set MXFP4_V3_OGS_KEEP_RAW=1 before loading weights."
                )
            from gpt_oss_mxfp4_v3.ogs_backend import ogs_moe_forward

            if not hasattr(self, "_ogs_cache"):
                self._ogs_cache = {}

            return ogs_moe_forward(
                x=x_bf16,
                gate_weight=self.gate.gate_score.weight,
                gate_bias=self.gate.gate_score.bias,
                w1_blocks_raw=experts.gate_up_proj_blocks_raw,
                w1_scales_raw=experts.gate_up_proj_scales_raw,
                w1_bias=experts.gate_up_proj_bias,
                w2_blocks_raw=experts.down_proj_blocks_raw,
                w2_scales_raw=experts.down_proj_scales_raw,
                w2_bias=experts.down_proj_bias,
                topk=self.num_experts_per_token,
                swiglu_alpha=self.alpha,
                swiglu_limit=self.limit,
                num_experts=self.num_experts,
                num_warps=HOPPER_SCALE_NUM_WARPS,
                _cache=self._ogs_cache,
            )

        def _debug_any_nan(label: str, t: Tensor) -> None:
            t_flat = F.reshape(t, [1, -1])
            nan = F.max(F.cast(F.is_nan(t_flat), DType.int32), axis=0)
            nan = F.max(nan, axis=1)
            ops.print(nan.__tensorvalue__(), label=label)

        def _debug_abs_max(label: str, t: Tensor) -> None:
            # Cast to FP32 so debug reductions don't depend on BF16 shuffle support.
            t_flat = F.reshape(F.abs(F.cast(t, DType.float32)), [1, -1])
            v = F.max(t_flat, axis=0)
            v = F.max(v, axis=1)
            ops.print(v.__tensorvalue__(), label=label)

        debug_enabled = _MOE_DEBUG and (
            _MOE_DEBUG_LAYER < 0 or self.layer_idx == _MOE_DEBUG_LAYER
        )

        # Minimal always-on-per-layer tracing for isolating the first layer that
        # introduces NaNs (kept separate from full debug to reduce noise).
        if _MOE_NAN_SCAN:
            ops.print(
                F.constant(
                    int(self.layer_idx) if self.layer_idx is not None else -1,
                    DType.int32,
                    device=x.device,
                ).__tensorvalue__(),
                label="mxfp4_v3_nan_scan_layer_idx",
            )
            _debug_any_nan("mxfp4_v3_nan_scan_x_any_nan", x_bf16)

        if debug_enabled:
            ops.print(
                F.constant(
                    int(self.layer_idx) if self.layer_idx is not None else -1,
                    DType.int32,
                    device=x.device,
                ).__tensorvalue__(),
                label="mxfp4_v3_moe_layer_idx",
            )
            _debug_any_nan("mxfp4_v3_moe_x_any_nan", x_bf16)
            _debug_abs_max("mxfp4_v3_moe_x_abs_max", x_bf16)

        router_idx, router_weight = self.gate(x_bf16)
        router_idx = F.reshape(router_idx, [-1])
        router_weight = F.reshape(router_weight, [-1])
        router_weight = F.reshape(
            router_weight, [seq_len, self.num_experts_per_token]
        )
        router_idx_i32 = F.cast(router_idx, DType.int32)
        # Clamp router indices to valid expert range to avoid downstream OOB.
        router_min = ops.constant(
            0, dtype=router_idx_i32.dtype, device=router_idx_i32.device
        )
        router_max = ops.constant(
            self.num_experts - 1,
            dtype=router_idx_i32.dtype,
            device=router_idx_i32.device,
        )
        router_idx_i32 = F.min(F.max(router_idx_i32, router_min), router_max)

        if debug_enabled:
            ops.print(
                F.min(router_idx_i32, axis=0).__tensorvalue__(),
                label="mxfp4_v3_router_idx_min",
            )
            ops.print(
                F.max(router_idx_i32, axis=0).__tensorvalue__(),
                label="mxfp4_v3_router_idx_max",
            )

        (
            token_expert_order,
            expert_start_indices,
            restore_token_order,
            expert_ids,
            expert_usage_stats,
        ) = moe_create_indices(router_idx_i32, self.num_experts)
        # Defensive clamp to keep expert ids inside weight tensor bounds.
        expert_ids = F.min(
            F.max(F.cast(expert_ids, DType.int32), router_min),
            router_max,
        )
        token_expert_order_i32 = F.cast(token_expert_order, DType.int32)
        expert_usage_stats_host = expert_usage_stats.to(CPU())

        if debug_enabled:
            ops.print(
                F.min(
                    F.cast(token_expert_order, DType.int32), axis=0
                ).__tensorvalue__(),
                label="mxfp4_v3_token_expert_order_min",
            )
            ops.print(
                F.max(
                    F.cast(token_expert_order, DType.int32), axis=0
                ).__tensorvalue__(),
                label="mxfp4_v3_token_expert_order_max",
            )
            ops.print(
                F.min(
                    F.cast(restore_token_order, DType.int32), axis=0
                ).__tensorvalue__(),
                label="mxfp4_v3_restore_token_order_min",
            )
            ops.print(
                F.max(
                    F.cast(restore_token_order, DType.int32), axis=0
                ).__tensorvalue__(),
                label="mxfp4_v3_restore_token_order_max",
            )
            # Indices used to gather `x_bf16` into per-expert order.
            token_rows = token_expert_order // self.num_experts_per_token
            ops.print(
                F.min(
                    F.cast(token_rows, DType.int32), axis=0
                ).__tensorvalue__(),
                label="mxfp4_v3_token_rows_min",
            )
            ops.print(
                F.max(
                    F.cast(token_rows, DType.int32), axis=0
                ).__tensorvalue__(),
                label="mxfp4_v3_token_rows_max",
            )

        permutated_states = F.gather(
            x_bf16,
            F.cast(
                token_expert_order_i32 // self.num_experts_per_token, DType.int32
            ),
            axis=0,
        )

        if self.apply_router_weight_first:
            permutated_states = permutated_states * F.gather(
                router_weight.reshape([-1, 1]), token_expert_order_i32, axis=0
            ).cast(x_bf16.dtype)

        experts = self._mxfp4_experts()

        # W1: grouped GEMM with MXFP4 weights.
        gate_up_output = mxfp4_grouped_matmul_ragged_bf16(
            permutated_states,
            experts.gate_up_proj_blocks,
            experts.gate_up_proj_scales,
            expert_start_indices,
            expert_ids,
            expert_usage_stats_host,
        )
        if debug_enabled:
            _debug_any_nan("mxfp4_v3_gate_up_matmul_any_nan", gate_up_output)
            _debug_abs_max("mxfp4_v3_gate_up_matmul_abs_max", gate_up_output)

        # Bias per token based on expert assignment.
        expert_assignments = F.gather(
            router_idx_i32, token_expert_order_i32, axis=0
        )
        if debug_enabled:
            ops.print(
                F.min(expert_assignments, axis=0).__tensorvalue__(),
                label="mxfp4_v3_expert_assignments_min",
            )
            ops.print(
                F.max(expert_assignments, axis=0).__tensorvalue__(),
                label="mxfp4_v3_expert_assignments_max",
            )
        bias_per_token = F.gather(
            experts.gate_up_proj_bias, expert_assignments, axis=0
        )
        gate_up_output = gate_up_output + bias_per_token
        if debug_enabled:
            _debug_any_nan("mxfp4_v3_gate_up_any_nan", gate_up_output)
            _debug_abs_max("mxfp4_v3_gate_up_abs_max", gate_up_output)

        gate = gate_up_output[:, 0::2]
        up = gate_up_output[:, 1::2]

        gate = F.min(gate, self.limit)
        up = up.clip(min=-self.limit, max=self.limit)

        glu = gate * F.sigmoid(gate * self.alpha)
        gated_output = (up + 1.0) * glu
        if debug_enabled:
            _debug_any_nan("mxfp4_v3_gated_any_nan", gated_output)
            _debug_abs_max("mxfp4_v3_gated_abs_max", gated_output)

        # W2: grouped GEMM with MXFP4 weights.
        down_output = mxfp4_grouped_matmul_ragged_bf16(
            gated_output,
            experts.down_proj_blocks,
            experts.down_proj_scales,
            expert_start_indices,
            expert_ids,
            expert_usage_stats_host,
        )

        down_bias_per_token = F.gather(
            experts.down_proj_bias, expert_assignments, axis=0
        )
        down_output = down_output + down_bias_per_token
        if _MOE_NAN_CLAMP:
            zero = ops.constant(
                0, dtype=down_output.dtype, device=down_output.device
            )
            down_output = F.where(F.is_nan(down_output), zero, down_output)
        if debug_enabled:
            _debug_any_nan("mxfp4_v3_down_any_nan", down_output)
            _debug_abs_max("mxfp4_v3_down_abs_max", down_output)

        # Restore token order using scatter with OOB-safe indices to avoid
        # gather out-of-bounds failures when indices are corrupted.
        restored_shape0 = seq_len * self.num_experts_per_token
        restore_indices = F.cast(restore_token_order, DType.int32)
        restore_indices_2d = ops.unsqueeze(restore_indices, -1)
        restored = scatter_nd_skip_oob_indices(
            input=ops.broadcast_to(
                ops.constant(
                    0,
                    dtype=down_output.dtype,
                    device=down_output.device,
                ),
                [restored_shape0, down_output.shape[1]],
            ),
            updates=down_output,
            indices=restore_indices_2d,
        )
        down_output = restored.reshape(
            [
                seq_len,
                self.num_experts_per_token,
                -1,
            ]
        )

        if not self.apply_router_weight_first:
            routed_expert_out = F.unsqueeze(router_weight, axis=1) @ down_output
            routed_expert_out = F.squeeze(routed_expert_out, axis=1).cast(
                x.dtype
            )
        else:
            routed_expert_out = down_output.transpose(1, 2)
            routed_expert_out = F.squeeze(
                F.sum(routed_expert_out, axis=2), axis=2
            ).cast(x.dtype)

        if self.has_shared_experts:
            routed_expert_out += self.shared_experts(x)

        if debug_enabled:
            _debug_any_nan("mxfp4_v3_routed_any_nan", routed_expert_out)
            _debug_abs_max("mxfp4_v3_routed_abs_max", routed_expert_out)

        if _MOE_NAN_SCAN:
            _debug_any_nan(
                "mxfp4_v3_nan_scan_routed_any_nan", routed_expert_out
            )

        return routed_expert_out


__all__ = ["GptOssMoE", "GptOssMoEGate"]
