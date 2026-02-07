"""Implements the GPT OSS transformer block."""

from __future__ import annotations

import os

from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    ops,
)
from max.nn.legacy.comm.allreduce import Allreduce
from max.nn.legacy.kv_cache import PagedCacheValues
from max.nn.legacy.layer import Module
from max.nn.legacy.transformer.distributed_transformer import (
    ShardableCallable,
    forward_sharded_layers,
)

from .attention import GptOssAttention
from .moe import GptOssMoE


class GptOssTransformerBlock(Module):
    """Stack of Attention, MoE, and RMSNorm layers for GPT OSS.

    This is a distributed transformer block that uses a Mixture of Experts (MoE)
    layer instead of a standard feedforward network.
    Block's attention type (full or window) is specified in the model config.
    """

    def __init__(
        self,
        attention: GptOssAttention,
        mlp: GptOssMoE,
        input_layernorm: ShardableCallable,
        post_attention_layernorm: ShardableCallable,
        devices: list[DeviceRef],
    ) -> None:
        super().__init__()

        # TODO: Figure out a better way to indicate to the type checker that these
        # are Shardable Modules. (Probably need a protocol called ShardableModule)
        self.self_attn = attention
        self.self_attn.sharding_strategy = ShardingStrategy.tensor_parallel(
            len(devices)
        )
        self.self_attn_shards = attention.shard(devices)

        self.mlp = mlp
        self.mlp.sharding_strategy = ShardingStrategy.tensor_parallel(
            len(devices)
        )
        self.mlp_shards = mlp.shard(devices)

        self.input_layernorm = input_layernorm
        self.input_layernorm.sharding_strategy = ShardingStrategy.replicate(
            len(devices)
        )
        self.input_layernorm_shards = input_layernorm.shard(devices)

        self.post_attention_layernorm = post_attention_layernorm
        self.post_attention_layernorm.sharding_strategy = (
            ShardingStrategy.replicate(len(devices))
        )
        self.post_attention_layernorm_shards = post_attention_layernorm.shard(
            devices
        )

        self.devices = devices
        self.allreduce = Allreduce(num_accelerators=len(devices))

    def __call__(
        self,
        layer_idx: TensorValue,
        xs: list[TensorValue],
        signal_buffers: list[BufferValue],
        kv_collections: list[PagedCacheValues],
        input_row_offsets: list[TensorValue],
        **kwargs,
    ) -> list[TensorValue]:
        debug_blocks = os.environ.get("MXFP4_MOE_DEBUG_BLOCKS", "") == "1"
        residual = xs
        norm_xs = forward_sharded_layers(self.input_layernorm_shards, xs)
        if debug_blocks:
            ops.print(layer_idx, label="mxfp4_block_idx")
            ops.print("mxfp4_block: attn start")
        attn_out = [
            shard(
                norm_xs[i],
                kv_collections[i],
                input_row_offsets=input_row_offsets[i],
                **kwargs,
            )
            for i, shard in enumerate(self.self_attn_shards)
        ]
        if len(attn_out) > 1:
            attn_out = self.allreduce(attn_out, signal_buffers)
        if debug_blocks:
            ops.print("mxfp4_block: attn done")

        # Add residual connection after attention
        hidden_states = [
            residual[i] + attn_out[i] for i in range(len(attn_out))
        ]

        # Apply post-attention layer norm and then MoE
        residual = hidden_states
        norm_xs = forward_sharded_layers(
            self.post_attention_layernorm_shards, hidden_states
        )

        # Apply MoE - it returns (output, router_logits)
        if debug_blocks:
            ops.print("mxfp4_block: moe start")
        mlp_results = [
            self.mlp_shards[i](norm_xs[i]) for i in range(len(norm_xs))
        ]
        if debug_blocks:
            ops.print("mxfp4_block: moe done")

        # Separate outputs and router logits
        mlp_outputs = [result for result in mlp_results]

        # Allreduce MoE outputs
        if len(mlp_outputs) > 1:
            mlp_outputs = self.allreduce(mlp_outputs, signal_buffers)
        if debug_blocks:
            ops.print("mxfp4_block: moe allreduce done")

        # Add residual connection
        hidden_states = [
            residual[i] + mlp_outputs[i] for i in range(len(mlp_outputs))
        ]
        if debug_blocks:
            ops.print("mxfp4_block: return")
        return hidden_states
