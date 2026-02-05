"""Implements the GPT-OSS model.

This mirrors MAX's built-in GPT-OSS module structure, but our MoE layer is
customized to call the MXFP4 Mojo ops (see `layers/moe.py`).
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
import os

from max.dtype import DType
from max.graph import BufferValue, ShardingStrategy, TensorValue, ops
from max.nn.legacy import ColumnParallelLinear, Embedding, LayerList, Module
from max.nn.legacy.kv_cache import PagedCacheValues
from max.nn.legacy.norm.rms_norm import RMSNorm
from max.nn.legacy.rotary_embedding import YarnRotaryEmbedding, YarnScalingParams

from .layers.attention import GptOssAttention
from .layers.moe import GptOssMoE
from .layers.transformer_block import GptOssTransformerBlock
from .model_config import GptOssConfig


class GptOssTextModel(Module):
    """Decoder-only Transformer with MoE feed-forward and YARN RoPE."""

    def __init__(self, config: GptOssConfig) -> None:
        super().__init__()
        self.devices = config.devices

        assert config.rope_scaling is not None, (
            "RoPE scaling is required for GPT-OSS models"
        )
        assert isinstance(config.rope_scaling, YarnScalingParams), (
            "Only YARN scaling is supported for GPT-OSS models"
        )
        yarn_scaling_params: YarnScalingParams = config.rope_scaling

        rope = YarnRotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_position_embeddings,
            head_dim=config.head_dim,
            interleaved=False,
            scaling_params=yarn_scaling_params,
        )

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.dtype,
            device=config.devices[0],
        )

        self.norm = RMSNorm(
            config.hidden_size,
            config.dtype,
            config.rms_norm_eps,
            multiply_before_cast=True,
        )
        self.norm.sharding_strategy = ShardingStrategy.replicate(
            len(config.devices)
        )
        self.norm_shards = self.norm.shard(config.devices)

        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            dtype=config.dtype,
            devices=config.devices,
            tied_weight=(
                self.embed_tokens.weight if config.tie_word_embeddings else None
            ),
        )

        create_norm = functools.partial(
            RMSNorm,
            config.hidden_size,
            config.dtype,
            eps=config.rms_norm_eps,
            multiply_before_cast=True,
        )

        layers = [
            GptOssTransformerBlock(
                attention=GptOssAttention(
                    rope=rope,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    hidden_size=config.hidden_size,
                    kv_params=config.kv_params,
                    layer_idx=i,
                    dtype=config.dtype,
                    devices=config.devices,
                    local_window_size=config.sliding_window,
                    has_bias=config.attention_bias,
                    layer_type=config.layer_types[i]
                    if i < len(config.layer_types)
                    else "full_attention",
                ),
                mlp=GptOssMoE(config),
                input_layernorm=create_norm(),
                post_attention_layernorm=create_norm(),
                devices=config.devices,
            )
            for i in range(config.num_hidden_layers)
        ]

        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.layers = LayerList(layers)
        self.kv_params = config.kv_params
        self.return_logits = config.return_logits

    def __call__(
        self,
        tokens: TensorValue,
        signal_buffers: Sequence[BufferValue],
        kv_collections: Sequence[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: Sequence[TensorValue],
        **kwargs,
    ) -> tuple[TensorValue, ...]:
        debug_model = os.environ.get("MXFP4_MODEL_DEBUG", "") == "1"
        debug_layer = int(os.environ.get("MXFP4_MODEL_DEBUG_LAYER", "-1"))
        debug_last_token = os.environ.get("MXFP4_MODEL_DEBUG_LAST_TOKEN", "") == "1"
        clamp_last_token = (
            os.environ.get("MXFP4_MODEL_CLAMP_LAST_TOKEN", "") == "1"
        )
        h_embed = self.embed_tokens(tokens)
        h = [h_embed.to(device) for device in self.devices]

        for idx, layer in enumerate(self.layers):
            if debug_model and (debug_layer < 0 or idx == debug_layer):
                ops.print(
                    ops.constant(idx, DType.int32, device=self.devices[0]),
                    label="mxfp4_model: enter_layer",
                )
            layer_idx_tensor = ops.constant(
                idx, DType.uint32, device=self.devices[0]
            )
            h = layer(
                layer_idx_tensor,
                h,
                signal_buffers,
                kv_collections,
                input_row_offsets=input_row_offsets,
                **kwargs,
            )
            if debug_model and (debug_layer < 0 or idx == debug_layer):
                ops.print(
                    ops.constant(idx, DType.int32, device=self.devices[0]),
                    label="mxfp4_model: exit_layer",
                )

        last_token_indices = [offsets[1:] - 1 for offsets in input_row_offsets]
        last_token_h: list[TensorValue] = []
        if h:
            if clamp_last_token:
                # Clamp indices defensively to avoid gather OOB when row offsets are off.
                clamped_indices: list[TensorValue] = []
                for h_device, indices, offsets in zip(
                    h, last_token_indices, input_row_offsets, strict=True
                ):
                    total_len = ops.cast(offsets[-1], DType.int32)
                    indices_i32 = ops.cast(indices, DType.int32)
                    zero = ops.constant(
                        0, DType.int32, device=h_device.device
                    )
                    upper = total_len - ops.constant(
                        1, DType.int32, device=h_device.device
                    )
                    clamped = ops.min(ops.max(indices_i32, zero), upper)
                    if debug_last_token:
                        ops.print(
                            ops.min(clamped, axis=0),
                            label="mxfp4_model: last_token_min",
                        )
                        ops.print(
                            ops.max(clamped, axis=0),
                            label="mxfp4_model: last_token_max",
                        )
                    clamped_indices.append(clamped)
                last_token_h = [
                    ops.gather(h_device, indices, axis=0)
                    for h_device, indices in zip(
                        h, clamped_indices, strict=True
                    )
                ]
            else:
                last_token_h = [
                    ops.gather(h_device, indices, axis=0)
                    for h_device, indices in zip(
                        h, last_token_indices, strict=True
                    )
                ]

        if debug_model:
            ops.print("mxfp4_model: pre_lm_head")
        last_logits = ops.cast(
            self.lm_head(
                [
                    self.norm_shards[i](last_token_h[i])
                    for i in range(len(last_token_h))
                ],
                signal_buffers,
            )[0],
            DType.float32,
        )
        if debug_model:
            ops.print("mxfp4_model: post_lm_head")
            ops.print(
                ops.argmax(last_logits, axis=-1),
                label="mxfp4_model: argmax_token",
            )
        return (last_logits,)


class GptOss(Module):
    """Top-level GPT-OSS module."""

    def __init__(self, config: GptOssConfig) -> None:
        super().__init__()
        self.language_model = GptOssTextModel(config)

    def __call__(
        self,
        tokens: TensorValue,
        signal_buffers: Sequence[BufferValue],
        kv_cache_inputs_per_dev: Sequence[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: Sequence[TensorValue],
    ) -> tuple[TensorValue, ...]:
        return self.language_model(
            tokens,
            signal_buffers,
            kv_cache_inputs_per_dev,
            return_n_logits,
            input_row_offsets,
        )


__all__ = ["GptOss", "GptOssTextModel"]
