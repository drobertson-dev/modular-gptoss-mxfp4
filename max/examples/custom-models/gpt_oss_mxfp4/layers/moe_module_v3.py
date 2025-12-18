"""ModuleV3 GPT-OSS MoE layer backed by MXFP4 Mojo custom ops."""

from __future__ import annotations

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.nn.module_v3 import Linear
from max.nn.module_v3.module import Module
from max.pipelines.architectures.gpt_oss_module_v3.layers.functional_kernels import (
    moe_create_indices,
)
from max.pipelines.architectures.gpt_oss_module_v3.layers.moe_base import (
    MoE,
    MoEGate,
)

from ..kernels import (
    MXFP4_VALUES_PER_BLOCK,
)
from ..kernels import (
    mxfp4_moe_topk_reduce_bf16 as _mxfp4_moe_topk_reduce_bf16,
)
from ..kernels import (
    mxfp4_moe_w1_swiglu as _mxfp4_moe_w1_swiglu,
)
from ..kernels import (
    mxfp4_moe_w2_pairs_bf16 as _mxfp4_moe_w2_pairs_bf16,
)

mxfp4_moe_w1_swiglu = F.functional(_mxfp4_moe_w1_swiglu)
mxfp4_moe_w2_pairs_bf16 = F.functional(_mxfp4_moe_w2_pairs_bf16)
mxfp4_moe_topk_reduce_bf16 = F.functional(_mxfp4_moe_topk_reduce_bf16)


class GptOssMoEGate(MoEGate):
    """GptOss-style Gate module for MoE with bias support (ModuleV3)."""

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

    def __call__(self, hidden_state: Tensor) -> tuple[Tensor, Tensor]:
        scores = self.gate_score(hidden_state)
        topk_scores, topk_indices = F.top_k(scores, k=self.num_experts_per_token, axis=-1)
        topk_scores = F.softmax(topk_scores)
        return topk_indices, topk_scores


class MXFP4Experts(Module):
    """Holds MXFP4 expert weights in checkpoint-compatible names.

    Parameter names must match `...mlp.experts.*` so the safetensors adapter can
    stay minimal and avoid BF16 materialization of expert weights.
    """

    def __init__(self, *, num_experts: int, hidden_dim: int, moe_dim: int) -> None:
        super().__init__()
        if hidden_dim % MXFP4_VALUES_PER_BLOCK != 0:
            raise ValueError("hidden_dim must be divisible by 32 for MXFP4 packing")
        if moe_dim % MXFP4_VALUES_PER_BLOCK != 0:
            raise ValueError("intermediate_size must be divisible by 32 for MXFP4 packing")

        kblocks_w1 = hidden_dim // MXFP4_VALUES_PER_BLOCK
        kblocks_w2 = moe_dim // MXFP4_VALUES_PER_BLOCK

        # W1: [E, 2*I, D/32, 16] blocks + [E, 2*I, D/32] scales.
        self.gate_up_proj_blocks = Tensor.zeros(
            shape=[num_experts, 2 * moe_dim, kblocks_w1, 16],
            dtype=DType.uint8,
        )
        self.gate_up_proj_scales = Tensor.zeros(
            shape=[num_experts, 2 * moe_dim, kblocks_w1],
            dtype=DType.uint8,
        )
        # Biases are kept FP32 to match current Mojo op signatures; they are small
        # and only used in epilogue math (FP32 in registers).
        self.gate_up_proj_bias = Tensor.zeros(
            shape=[num_experts, 2 * moe_dim],
            dtype=DType.float32,
        )

        # W2: [E, D, I/32, 16] blocks + [E, D, I/32] scales.
        self.down_proj_blocks = Tensor.zeros(
            shape=[num_experts, hidden_dim, kblocks_w2, 16],
            dtype=DType.uint8,
        )
        self.down_proj_scales = Tensor.zeros(
            shape=[num_experts, hidden_dim, kblocks_w2],
            dtype=DType.uint8,
        )
        self.down_proj_bias = Tensor.zeros(
            shape=[num_experts, hidden_dim],
            dtype=DType.float32,
        )


class GptOssMoEMXFP4(MoE):
    """GPT-OSS MoE that routes via `moe_create_indices` and runs MXFP4 Mojo ops."""

    def __init__(self, config) -> None:  # noqa: ANN001
        self.alpha = 1.702
        self.limit = float(getattr(config, "swiglu_limit", 7.0))

        self.config = config
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
        # module named `experts` to match checkpoint key paths.
        self.experts = MXFP4Experts(
            num_experts=self.num_experts,
            hidden_dim=self.hidden_dim,
            moe_dim=self.moe_dim,
        )

    def __call__(self, x: Tensor) -> Tensor:
        if x.dtype != DType.bfloat16:
            x_bf16 = F.cast(x, DType.bfloat16)
        else:
            x_bf16 = x

        router_idx, router_weight = self.gate(x_bf16)  # [T, TOPK]
        router_idx_flat = F.reshape(router_idx, [-1])  # [P]
        gate_weights_flat = F.reshape(router_weight, [-1])  # [P]

        (
            token_expert_order,
            expert_start_indices,
            _restore_token_order,
            expert_ids,
            expert_usage_stats,
        ) = moe_create_indices(F.cast(router_idx_flat, DType.int32), self.num_experts)

        gate_weights_bf16 = (
            gate_weights_flat
            if gate_weights_flat.dtype == DType.bfloat16
            else F.cast(gate_weights_flat, DType.bfloat16)
        )

        h_sorted = mxfp4_moe_w1_swiglu(
            x_bf16,
            token_expert_order,
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
            self.experts.gate_up_proj_blocks,
            self.experts.gate_up_proj_scales,
            self.experts.gate_up_proj_bias,
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
            self.experts.down_proj_blocks,
            self.experts.down_proj_scales,
            self.experts.down_proj_bias,
            target="gpu",
        )
        y = mxfp4_moe_topk_reduce_bf16(x_bf16, y_pairs, target="gpu")
        return y if y.dtype == x.dtype else F.cast(y, x.dtype)


__all__ = ["GptOssMoEGate", "GptOssMoEMXFP4"]
