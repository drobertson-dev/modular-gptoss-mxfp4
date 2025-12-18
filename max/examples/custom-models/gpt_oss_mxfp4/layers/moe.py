"""GPT-OSS Mixture of Experts Layer (MXFP4 custom ops).

This matches the Triton MoE structure:
  router -> routing indices -> W1 MXFP4 GEMM + fused SwiGLU -> W2 MXFP4 GEMM (per-pair) -> TOPK reduce

Python is responsible for producing routing tensors and passing weights in the
exact layout expected by the Mojo custom ops. Kernel optimization (SM90 wgmma)
must not require Python interface churn.
"""

from __future__ import annotations

from collections.abc import Iterable

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, Weight, ops
from max.nn.kernels import moe_create_indices
from max.nn.layer import LayerList, Shardable
from max.nn.linear import Linear
from max.nn.moe import MoE, MoEGate

from ..kernels import (
    MXFP4_VALUES_PER_BLOCK,
    mxfp4_moe_topk_reduce_bf16,
    mxfp4_moe_w1_swiglu,
    mxfp4_moe_w2_pairs_bf16,
)
from ..model_config import GptOssConfig


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

    def __call__(self, hidden_state: TensorValue) -> tuple[TensorValue, TensorValue]:
        scores = self.gate_score(hidden_state)
        topk_scores, topk_indices = ops.top_k(scores, k=self.num_experts_per_token, axis=-1)
        topk_scores = ops.softmax(topk_scores)
        return topk_indices, topk_scores


class GptOssMoE(MoE, Shardable):
    """GptOss-style MoE implementation backed by MXFP4 Mojo custom ops."""

    def __init__(self, config: GptOssConfig):
        self.alpha = 1.702
        self.limit = float(getattr(config, "swiglu_limit", 7.0))

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
            raise ValueError("hidden_dim must be divisible by 32 for MXFP4 packing")
        if self.moe_dim % MXFP4_VALUES_PER_BLOCK != 0:
            raise ValueError("intermediate_size must be divisible by 32 for MXFP4 packing")
        kblocks_w1 = self.hidden_dim // MXFP4_VALUES_PER_BLOCK
        kblocks_w2 = self.moe_dim // MXFP4_VALUES_PER_BLOCK
        # W1: [E, 2*I, D/32, 16] blocks + [E, 2*I, D/32] scales (E8M0 exponent bytes).
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
        # Keep biases as FP32 so we don't do per-step graph casts on the hot path.
        self._experts_gate_up_proj_bias = Weight(
            "experts.gate_up_proj_bias",
            shape=[self.num_experts, 2 * self.moe_dim],
            dtype=DType.float32,
            device=self.devices[0],
        )
        # W2: [E, D, I/32, 16] blocks + [E, D, I/32] scales (E8M0 exponent bytes).
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
        # Routing.
        router_idx, router_weight = self.gate(x)  # [T, TOPK]
        router_idx_flat = ops.reshape(router_idx, [-1])  # [P]
        gate_weights_flat = ops.reshape(router_weight, [-1])  # [P] in original pair order
        (
            token_expert_order,
            expert_start_indices,
            _restore_token_order,
            expert_ids,
            expert_usage_stats,
        ) = moe_create_indices(ops.cast(router_idx_flat, DType.int32), self.num_experts)
        # Mojo kernels expect BF16 activations, BF16 gate weights, and FP32 biases.
        x_bf16 = ops.cast(x, DType.bfloat16) if x.dtype != DType.bfloat16 else x
        gate_weights_bf16 = (
            ops.cast(gate_weights_flat, DType.bfloat16)
            if gate_weights_flat.dtype != DType.bfloat16
            else gate_weights_flat
        )
        w1_bias_f32 = self._experts_gate_up_proj_bias
        w2_bias_f32 = self._experts_down_proj_bias
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
        y = mxfp4_moe_topk_reduce_bf16(x_bf16, y_pairs, target="gpu")
        return y if y.dtype == x.dtype else ops.cast(y, x.dtype)

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        # The current Mojo kernels are single-device; keep the contract explicit.
        if not strategy.is_tensor_parallel:
            raise ValueError("Only tensor-parallel sharding is supported for MoE modules")
        if strategy.num_devices != 1:
            raise ValueError("MXFP4 MoE sharding across multiple devices is not implemented yet")
        self._sharding_strategy = strategy

    def shard(self, devices: Iterable[DeviceRef]) -> list[GptOssMoE]:
        devices = list(devices)
        if not self._sharding_strategy:
            raise ValueError("MoE module cannot be sharded because no sharding strategy was provided.")
        if len(devices) != 1:
            raise ValueError("MXFP4 MoE currently supports only single-device execution")
        return [self]
