"""Pipeline model for GPT-OSS with MXFP4 Mojo custom ops.

We subclass MAX's built-in GPT-OSS pipeline model to preserve the full pipeline
contract, and only override graph construction to:
- instantiate our local `GptOss` module (whose MoE layer calls MXFP4 custom ops)
- add `custom_extensions` so the Mojo ops are registered when compiling the Graph
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
from max.nn import Signals
from max.pipelines.architectures.gpt_oss.model import (
    GptOssInputs,
)
from max.pipelines.architectures.gpt_oss.model import (
    GptOssModel as _BaseGptOssModel,
)

from .gpt_oss import GptOss
from .kernels import get_mxfp4_kernels_path
from .model_config import GptOssConfig


class GptOssModel(_BaseGptOssModel):
    """GPT-OSS pipeline model that wires in MXFP4 custom ops."""

    def _build_graph(self):  # noqa: ANN202
        device0 = self.devices[0]
        device_ref = DeviceRef(device0.label, device0.id)
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )

        input_row_offsets_types = [
            TensorType(
                DType.uint32,
                shape=["input_row_offsets_len"],
                device=DeviceRef(device.label, device.id),
            )
            for device in self.devices
        ]
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )

        huggingface_config = self.huggingface_config
        if self.adapter:
            state_dict = self.adapter(
                dict(self.weights.items()),
                huggingface_config=huggingface_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            state_dict = {
                key: value.data() for key, value in self.weights.items()
            }

        model_config = GptOssConfig.generate(
            pipeline_config=self.pipeline_config,
            huggingface_config=huggingface_config,
            state_dict=state_dict,
            dtype=self.dtype,
            n_devices=len(self.devices),
            cache_dtype=self.encoding.cache_dtype,
            kv_cache_config=self.kv_cache_config,
            return_logits=self.return_logits,
        )
        nn_model = GptOss(model_config)
        nn_model.load_state_dict(
            state_dict,
            weight_alignment=1,
            strict=self._strict_state_dict_loading,
        )
        self.state_dict = nn_model.state_dict(auto_initialize=False)

        signals = Signals(
            devices=(DeviceRef(d.label, d.id) for d in self.devices)
        )

        # MAX nightlies migrated KV cache graph inputs from `kv_manager` to
        # `kv_params`. Use `kv_params` so this custom architecture stays aligned
        # with the built-in GPT-OSS model wiring.
        kv_inputs = self.kv_params.get_symbolic_inputs()
        flattened_kv_types = [
            kv_type for sublist in kv_inputs for kv_type in sublist
        ]

        with Graph(
            getattr(self.huggingface_config, "model_type", "GptOss"),
            input_types=[
                tokens_type,
                return_n_logits_type,
                *input_row_offsets_types,
                *signals.input_types(),
                *flattened_kv_types,
            ],
            custom_extensions=[get_mxfp4_kernels_path()],
        ) as graph:
            tokens, return_n_logits, *variadic_args = graph.inputs

            input_row_offsets = [
                v.tensor for v in variadic_args[: len(self.devices)]
            ]
            variadic_args = variadic_args[len(self.devices) :]

            signal_buffers = [
                v.buffer for v in variadic_args[: len(self.devices)]
            ]
            variadic_args = variadic_args[len(self.devices) :]

            kv_cache = self._unflatten_kv_inputs(variadic_args)

            outputs = nn_model(
                tokens=tokens.tensor,
                signal_buffers=signal_buffers,
                kv_cache_inputs_per_dev=kv_cache,
                return_n_logits=return_n_logits.tensor,
                input_row_offsets=input_row_offsets,
            )
            graph.output(*outputs)

        return graph


__all__ = ["GptOssInputs", "GptOssModel"]
