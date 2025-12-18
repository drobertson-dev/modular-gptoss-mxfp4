"""ModuleV3 GPT-OSS MoE layer swapping expert GEMMs to MXFP4 grouped matmul."""

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
    mxfp4_grouped_matmul_ragged_bf16 as _mxfp4_grouped_matmul_ragged_bf16,
)
from ..model_config import GptOssConfig

mxfp4_grouped_matmul_ragged_bf16 = F.functional(_mxfp4_grouped_matmul_ragged_bf16)


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

    def __call__(self, hidden_state: Tensor) -> tuple[Tensor, Tensor]:
        scores = self.gate_score(hidden_state)
        topk_scores, topk_indices = F.top_k(scores, k=self.num_experts_per_token, axis=-1)
        topk_scores = F.softmax(topk_scores)
        return topk_indices, topk_scores


class MXFP4Experts(Module):
    """Holds MXFP4 expert weights in checkpoint-compatible names."""

    def __init__(self, *, num_experts: int, hidden_dim: int, moe_dim: int) -> None:
        super().__init__()
        if hidden_dim % MXFP4_VALUES_PER_BLOCK != 0:
            raise ValueError("hidden_dim must be divisible by 32 for MXFP4 packing")
        if moe_dim % MXFP4_VALUES_PER_BLOCK != 0:
            raise ValueError("intermediate_size must be divisible by 32 for MXFP4 packing")

        kblocks_w1 = hidden_dim // MXFP4_VALUES_PER_BLOCK
        kblocks_w2 = moe_dim // MXFP4_VALUES_PER_BLOCK

        # W1: [E, 2*I, D/32, 16] blocks + [E, 2*I, D/32] scales + BF16 bias.
        self.gate_up_proj_blocks = Tensor.zeros(
            shape=[num_experts, 2 * moe_dim, kblocks_w1, 16],
            dtype=DType.uint8,
        )
        self.gate_up_proj_scales = Tensor.zeros(
            shape=[num_experts, 2 * moe_dim, kblocks_w1],
            dtype=DType.uint8,
        )
        self.gate_up_proj_bias = Tensor.zeros(
            shape=[num_experts, 2 * moe_dim],
            dtype=DType.bfloat16,
        )

        # W2: [E, D, I/32, 16] blocks + [E, D, I/32] scales + BF16 bias.
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
            dtype=DType.bfloat16,
        )


class GptOssMoE(MoE):
    """GptOss-style MoE that swaps expert GEMMs to MXFP4 grouped matmul."""

    def __init__(self, config: GptOssConfig):
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
        # module named `experts` so checkpoint keys map directly.
        self.experts = MXFP4Experts(
            num_experts=self.num_experts,
            hidden_dim=self.hidden_dim,
            moe_dim=self.moe_dim,
        )

    def __call__(self, x: Tensor) -> Tensor:
        seq_len = x.shape[0]

        x_bf16 = x if x.dtype == DType.bfloat16 else F.cast(x, DType.bfloat16)

        router_idx, router_weight = self.gate(x_bf16)
        router_idx = F.reshape(router_idx, [-1])

        (
            token_expert_order,
            expert_start_indices,
            restore_token_order,
            expert_ids,
            expert_usage_stats,
        ) = moe_create_indices(F.cast(router_idx, DType.int32), self.num_experts)

        permutated_states = F.gather(
            x_bf16,
            F.cast(token_expert_order // self.num_experts_per_token, DType.int32),
            axis=0,
        )

        if self.apply_router_weight_first:
            permutated_states = permutated_states * F.gather(
                router_weight.reshape([-1, 1]), token_expert_order, axis=0
            ).cast(x_bf16.dtype)

        # W1: grouped GEMM with MXFP4 weights.
        gate_up_output = mxfp4_grouped_matmul_ragged_bf16(
            permutated_states,
            self.experts.gate_up_proj_blocks,
            self.experts.gate_up_proj_scales,
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
        )

        # Bias per token based on expert assignment.
        expert_assignments = F.gather(router_idx, token_expert_order, axis=0)
        bias_per_token = F.gather(self.experts.gate_up_proj_bias, expert_assignments, axis=0)
        gate_up_output = gate_up_output + bias_per_token

        gate = gate_up_output[:, 0::2]
        up = gate_up_output[:, 1::2]

        gate = F.min(gate, self.limit)
        up = up.clip(min=-self.limit, max=self.limit)

        glu = gate * F.sigmoid(gate * self.alpha)
        gated_output = (up + 1.0) * glu

        # W2: grouped GEMM with MXFP4 weights.
        down_output = mxfp4_grouped_matmul_ragged_bf16(
            gated_output,
            self.experts.down_proj_blocks,
            self.experts.down_proj_scales,
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
        )

        down_bias_per_token = F.gather(self.experts.down_proj_bias, expert_assignments, axis=0)
        down_output = down_output + down_bias_per_token

        down_output = F.gather(down_output, restore_token_order, axis=0).reshape([
            seq_len,
            self.num_experts_per_token,
            -1,
        ])

        if not self.apply_router_weight_first:
            routed_expert_out = F.unsqueeze(router_weight, axis=1) @ down_output
            routed_expert_out = F.squeeze(routed_expert_out, axis=1).cast(x.dtype)
        else:
            routed_expert_out = down_output.transpose(1, 2)
            routed_expert_out = F.squeeze(F.sum(routed_expert_out, axis=2), axis=2).cast(x.dtype)

        if self.has_shared_experts:
            routed_expert_out += self.shared_experts(x)

        return routed_expert_out


__all__ = ["GptOssMoE", "GptOssMoEGate"]
