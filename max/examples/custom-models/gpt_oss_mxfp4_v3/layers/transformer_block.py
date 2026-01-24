"""Implements the GPT OSS transformer block for MXFP4."""

from __future__ import annotations

from max.nn import Module
from max.nn.legacy.kv_cache import PagedCacheValues
from max.nn.norm import RMSNorm
from max.tensor import Tensor
from max.pipelines.architectures.gpt_oss.layers.attention import (
    GptOssAttention,
)
from max.pipelines.architectures.gpt_oss.layers.moe_base import MoE


class GptOssTransformerBlock(Module[..., Tensor]):
    """Stack of Attention, MoE, and RMSNorm layers for GPT OSS."""

    def __init__(
        self,
        attention: GptOssAttention,
        mlp: MoE,
        input_layernorm: RMSNorm,
        post_attention_layernorm: RMSNorm,
    ) -> None:
        super().__init__()
        self.self_attn = attention
        self.mlp = mlp

        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm

    def forward(
        self,
        layer_idx: Tensor,
        x: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
        **kwargs,
    ) -> Tensor:
        residual = x
        norm_xs = self.input_layernorm(x)
        attn_out = self.self_attn(
            norm_xs,
            kv_collection,
            input_row_offsets=input_row_offsets,
            **kwargs,
        )

        # Add residual connection after attention
        hidden_states = residual + attn_out

        # Apply post-attention layer norm and then MoE
        residual = hidden_states
        norm_xs = self.post_attention_layernorm(hidden_states)

        # Apply MoE - it returns (output, router_logits)
        mlp_outputs = self.mlp(norm_xs)

        # Add residual connection
        hidden_states = residual + mlp_outputs
        return hidden_states


__all__ = ["GptOssTransformerBlock"]
