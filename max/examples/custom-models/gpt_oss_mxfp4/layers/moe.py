"""GPT-OSS Mixture of Experts Layer (MXFP4 custom ops).

This matches the Triton MoE structure:
  router -> routing indices -> W1 MXFP4 GEMM + fused SwiGLU -> W2 MXFP4 GEMM (per-pair) -> TOPK reduce

Python is responsible for producing routing tensors and passing weights in the
exact layout expected by the Mojo custom ops. Kernel optimization (SM90 wgmma)
must not require Python interface churn.
"""

from __future__ import annotations

from collections.abc import Iterable
import os

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, Weight, ops
from max.nn.legacy.kernels import moe_create_indices
from max.nn.legacy.layer import LayerList, Shardable
from max.nn.legacy.linear import Linear
from max.nn.legacy.moe import MoE, MoEGate

from ..kernels import (
    HOPPER_SCALE_NUM_WARPS,
    MXFP4_VALUES_PER_BLOCK,
    mxfp4_grouped_matmul_ragged_bf16_swizzled,
    mxfp4_moe_topk_reduce_bf16,
    mxfp4_moe_w1_swiglu,
    mxfp4_moe_w2_pairs_bf16,
)
from ..model_config import GptOssConfig

# Keep RS grouped path opt-in until index/OOB issues are resolved in end-to-end generate.
_USE_GROUPED_RS = os.environ.get("MXFP4_LEGACY_GROUPED_RS", "0") == "1"
_DISABLE_SMALL_M = os.environ.get("MXFP4_LEGACY_NO_SMALL_M", "1") == "1"


class GptOssMoEGate(MoEGate):
    """GptOss-style Gate module for MoE with bias support."""

    def __init__(
        self,
        devices: list[DeviceRef],
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        dtype: DType,
    ) -> None:
        super().__init__(
            devices=devices,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            dtype=dtype,
        )
        # Bias-enabled router linear.
        self.gate_score = Linear(
            in_dim=hidden_dim,
            out_dim=num_experts,
            dtype=dtype,
            device=devices[0],
            has_bias=True,
        )

    def __call__(
        self, hidden_state: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        scores = self.gate_score(hidden_state)
        topk_scores, topk_indices = ops.top_k(
            scores, k=self.num_experts_per_token, axis=-1
        )
        topk_scores = ops.softmax(topk_scores)
        return topk_indices, topk_scores


class GptOssMoE(MoE, Shardable):
    """GptOss-style MoE implementation backed by MXFP4 Mojo custom ops."""

    def __init__(self, config: GptOssConfig):
        self.alpha = 1.702
        self.limit = float(getattr(config, "swiglu_limit", 7.0))
        self._use_grouped_rs = _USE_GROUPED_RS

        self.config = config
        self._sharding_strategy: ShardingStrategy | None = None

        super().__init__(
            devices=config.devices,
            hidden_dim=config.hidden_size,
            num_experts=config.num_local_experts,
            num_experts_per_token=config.num_experts_per_tok,
            moe_dim=config.intermediate_size,
            gate_cls=GptOssMoEGate,
            has_shared_experts=False,
            ep_size=1,
            dtype=config.dtype,
            apply_router_weight_first=False,
        )

    def _init_experts(self) -> None:
        # Experts are represented as packed MXFP4 tensors, not per-expert Modules.
        self.experts = LayerList([])

        if self.hidden_dim % MXFP4_VALUES_PER_BLOCK != 0:
            raise ValueError(
                "hidden_dim must be divisible by 32 for MXFP4 packing"
            )
        if self.moe_dim % MXFP4_VALUES_PER_BLOCK != 0:
            raise ValueError(
                "intermediate_size must be divisible by 32 for MXFP4 packing"
            )
        kblocks_w1 = self.hidden_dim // MXFP4_VALUES_PER_BLOCK
        kblocks_w2 = self.moe_dim // MXFP4_VALUES_PER_BLOCK

        if self._use_grouped_rs:
            if (kblocks_w1 % 2) != 0 or (kblocks_w2 % 2) != 0:
                raise ValueError(
                    "Grouped RS path requires hidden/intermediate divisible by 64"
                )
            hopper_align_m = 32 * HOPPER_SCALE_NUM_WARPS

            def _value_swizzle_shape(m_rows: int, k_vals: int) -> tuple[int, int]:
                # Matches Hopper value swizzle layout: M padded to 16, Kbytes padded to 32.
                kbytes = k_vals // 2
                m_pad = ((m_rows + 15) // 16) * 16
                k_pad = ((kbytes + 31) // 32) * 32
                return m_pad // 4, k_pad * 4

            n_pad_w1 = (
                (2 * self.moe_dim + hopper_align_m - 1) // hopper_align_m
            ) * hopper_align_m
            scale_m2_w1 = n_pad_w1 // 32
            w1_m2, w1_k2 = _value_swizzle_shape(2 * self.moe_dim, self.hidden_dim)

            self._experts_gate_up_proj_weight_blocks = Weight(
                "experts.gate_up_proj_blocks",
                shape=[self.num_experts, w1_m2, w1_k2],
                dtype=DType.uint8,
                device=self.devices[0],
            )
            self._experts_gate_up_proj_weight_scales = Weight(
                "experts.gate_up_proj_scales",
                shape=[self.num_experts, scale_m2_w1, self.hidden_dim],
                dtype=DType.uint8,
                device=self.devices[0],
            )
            self._experts_gate_up_proj_bias = Weight(
                "experts.gate_up_proj_bias",
                shape=[self.num_experts, 2 * self.moe_dim],
                dtype=DType.bfloat16,
                device=self.devices[0],
            )

            n_pad_w2 = (
                (self.hidden_dim + hopper_align_m - 1) // hopper_align_m
            ) * hopper_align_m
            scale_m2_w2 = n_pad_w2 // 32
            w2_m2, w2_k2 = _value_swizzle_shape(self.hidden_dim, self.moe_dim)
            self._experts_down_proj_weight_blocks = Weight(
                "experts.down_proj_blocks",
                shape=[self.num_experts, w2_m2, w2_k2],
                dtype=DType.uint8,
                device=self.devices[0],
            )
            self._experts_down_proj_weight_scales = Weight(
                "experts.down_proj_scales",
                shape=[self.num_experts, scale_m2_w2, self.moe_dim],
                dtype=DType.uint8,
                device=self.devices[0],
            )
            self._experts_down_proj_bias = Weight(
                "experts.down_proj_bias",
                shape=[self.num_experts, self.hidden_dim],
                dtype=DType.bfloat16,
                device=self.devices[0],
            )
            return

        # Legacy fallback path: W1/W2 fused custom ops with non-swizzled layouts.
        self._experts_gate_up_proj_weight_blocks = Weight(
            "experts.gate_up_proj_blocks",
            shape=[self.num_experts, 2 * self.moe_dim, kblocks_w1, 16],
            dtype=DType.uint8,
            device=self.devices[0],
        )
        self._experts_gate_up_proj_weight_scales = Weight(
            "experts.gate_up_proj_scales",
            shape=[self.num_experts, 2 * self.moe_dim, kblocks_w1],
            dtype=DType.uint8,
            device=self.devices[0],
        )
        self._experts_gate_up_proj_bias = Weight(
            "experts.gate_up_proj_bias",
            shape=[self.num_experts, 2 * self.moe_dim],
            dtype=DType.float32,
            device=self.devices[0],
        )
        self._experts_down_proj_weight_blocks = Weight(
            "experts.down_proj_blocks",
            shape=[self.num_experts, self.hidden_dim, kblocks_w2, 16],
            dtype=DType.uint8,
            device=self.devices[0],
        )
        self._experts_down_proj_weight_scales = Weight(
            "experts.down_proj_scales",
            shape=[self.num_experts, self.hidden_dim, kblocks_w2],
            dtype=DType.uint8,
            device=self.devices[0],
        )
        self._experts_down_proj_bias = Weight(
            "experts.down_proj_bias",
            shape=[self.num_experts, self.hidden_dim],
            dtype=DType.float32,
            device=self.devices[0],
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        # Enforce that these ops run on GPU for now.
        if x.device == DeviceRef.CPU():
            raise ValueError("MXFP4 MoE custom ops are GPU-only")
        if os.environ.get("MXFP4_MOE_BYPASS", "") == "1":
            # Debug/bring-up path: bypass MXFP4 MoE and return activations.
            return x
        debug_graph = os.environ.get("MXFP4_MOE_DEBUG_GRAPH", "") == "1"
        debug_stats = os.environ.get("MXFP4_MOE_DEBUG_STATS", "") == "1"
        # Routing.
        router_idx, router_weight = self.gate(x)  # [T, TOPK]
        router_idx_flat = ops.reshape(router_idx, [-1])  # [P]
        gate_weights_flat = ops.reshape(
            router_weight, [-1]
        )  # [P] in original pair order
        router_idx_i32 = ops.cast(router_idx_flat, DType.int32)
        router_min = ops.constant(
            0, dtype=DType.int32, device=router_idx_i32.device
        )
        router_max = ops.constant(
            self.num_experts - 1,
            dtype=DType.int32,
            device=router_idx_i32.device,
        )
        # Defensive clamp so invalid router ids don't poison gather/scatter indices.
        router_idx_i32 = ops.min(ops.max(router_idx_i32, router_min), router_max)
        (
            token_expert_order,
            expert_start_indices,
            restore_token_order,
            expert_ids,
            expert_usage_stats,
        ) = moe_create_indices(router_idx_i32, self.num_experts)
        # Defensive clamp to keep expert ids inside weight tensor bounds.
        expert_ids = ops.min(
            ops.max(
                ops.cast(expert_ids, DType.int32),
                router_min,
            ),
            router_max,
        )
        if debug_graph:
            ops.print("mxfp4_moe: routing done")
        if debug_stats:
            ops.print(
                ops.min(token_expert_order, axis=0),
                label="mxfp4_moe_token_expert_order_min",
            )
            ops.print(
                ops.max(token_expert_order, axis=0),
                label="mxfp4_moe_token_expert_order_max",
            )
            ops.print(
                ops.min(expert_start_indices, axis=0),
                label="mxfp4_moe_expert_start_min",
            )
            ops.print(
                ops.max(expert_start_indices, axis=0),
                label="mxfp4_moe_expert_start_max",
            )
            ops.print(
                expert_usage_stats, label="mxfp4_moe_expert_usage_stats"
            )
        # Mojo kernels expect BF16 activations, BF16 gate weights, and FP32 biases.
        x_bf16 = ops.cast(x, DType.bfloat16) if x.dtype != DType.bfloat16 else x
        gate_weights_bf16 = (
            ops.cast(gate_weights_flat, DType.bfloat16)
            if gate_weights_flat.dtype != DType.bfloat16
            else gate_weights_flat
        )
        if self._use_grouped_rs:
            token_expert_order_i32 = ops.cast(token_expert_order, DType.int32)
            token_rows = ops.cast(
                token_expert_order_i32 // self.num_experts_per_token,
                DType.int32,
            )
            permutated_states = ops.gather(x_bf16, token_rows, axis=0)

            if debug_graph:
                ops.print("mxfp4_moe: grouped w1")
            gate_up_output = mxfp4_grouped_matmul_ragged_bf16_swizzled(
                permutated_states,
                self._experts_gate_up_proj_weight_blocks,
                self._experts_gate_up_proj_weight_scales,
                expert_start_indices,
                expert_ids,
                expert_usage_stats,
                n_cols=2 * self.moe_dim,
                target="gpu",
                no_small_m=_DISABLE_SMALL_M,
            )
            expert_assignments = ops.gather(
                router_idx_i32, token_expert_order_i32, axis=0
            )
            gate_bias = ops.gather(
                self._experts_gate_up_proj_bias, expert_assignments, axis=0
            )
            gate_up_output = gate_up_output + gate_bias

            gate = gate_up_output[:, 0::2]
            up = gate_up_output[:, 1::2]
            limit_pos = ops.constant(
                self.limit, dtype=gate.dtype, device=gate.device
            )
            limit_neg = ops.constant(
                -self.limit, dtype=up.dtype, device=up.device
            )
            alpha = ops.constant(
                self.alpha, dtype=gate.dtype, device=gate.device
            )
            one = ops.constant(1.0, dtype=up.dtype, device=up.device)
            gate = ops.min(gate, limit_pos)
            up = ops.min(ops.max(up, limit_neg), limit_pos)
            glu = gate * ops.sigmoid(gate * alpha)
            gated_output = (up + one) * glu

            if debug_graph:
                ops.print("mxfp4_moe: grouped w2")
            down_output = mxfp4_grouped_matmul_ragged_bf16_swizzled(
                gated_output,
                self._experts_down_proj_weight_blocks,
                self._experts_down_proj_weight_scales,
                expert_start_indices,
                expert_ids,
                expert_usage_stats,
                n_cols=self.hidden_dim,
                target="gpu",
                no_small_m=_DISABLE_SMALL_M,
            )
            down_bias = ops.gather(
                self._experts_down_proj_bias, expert_assignments, axis=0
            )
            down_output = down_output + down_bias

            gate_weights_sorted = ops.gather(
                gate_weights_bf16, token_expert_order_i32, axis=0
            )
            down_output = down_output * ops.unsqueeze(gate_weights_sorted, -1)

            restore_indices = ops.cast(restore_token_order, DType.int32)
            # `restore_token_order` is the inverse permutation of
            # `token_expert_order`. Gather restores pair order directly and
            # avoids the extra scatter staging path.
            y_pairs = ops.gather(down_output, restore_indices, axis=0)

            if debug_graph:
                ops.print("mxfp4_moe: grouped reduce")
            y = mxfp4_moe_topk_reduce_bf16(x_bf16, y_pairs, target="gpu")
            return y if y.dtype == x.dtype else ops.cast(y, x.dtype)

        w1_bias_f32 = self._experts_gate_up_proj_bias
        w2_bias_f32 = self._experts_down_proj_bias
        if debug_graph:
            ops.print("mxfp4_moe: entering w1")
        h_sorted = mxfp4_moe_w1_swiglu(
            x_bf16,
            token_expert_order,
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
            self._experts_gate_up_proj_weight_blocks,
            self._experts_gate_up_proj_weight_scales,
            w1_bias_f32,
            alpha=self.alpha,
            limit=self.limit,
            target="gpu",
        )
        if debug_graph:
            ops.print("mxfp4_moe: w1 done")
        y_pairs = mxfp4_moe_w2_pairs_bf16(
            x_bf16,
            h_sorted,
            token_expert_order,
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
            gate_weights_bf16,
            self._experts_down_proj_weight_blocks,
            self._experts_down_proj_weight_scales,
            w2_bias_f32,
            target="gpu",
        )
        if debug_graph:
            ops.print("mxfp4_moe: w2 done")
        y = mxfp4_moe_topk_reduce_bf16(x_bf16, y_pairs, target="gpu")
        if debug_graph:
            ops.print("mxfp4_moe: reduce done")
        return y if y.dtype == x.dtype else ops.cast(y, x.dtype)

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        # The current Mojo kernels are single-device; keep the contract explicit.
        if not strategy.is_tensor_parallel:
            raise ValueError(
                "Only tensor-parallel sharding is supported for MoE modules"
            )
        if strategy.num_devices != 1:
            raise ValueError(
                "MXFP4 MoE sharding across multiple devices is not implemented yet"
            )
        self._sharding_strategy = strategy

    def shard(self, devices: Iterable[DeviceRef]) -> list[GptOssMoE]:
        devices = list(devices)
        if not self._sharding_strategy:
            raise ValueError(
                "MoE module cannot be sharded because no sharding strategy was provided."
            )
        if len(devices) != 1:
            raise ValueError(
                "MXFP4 MoE currently supports only single-device execution"
            )
        return [self]
