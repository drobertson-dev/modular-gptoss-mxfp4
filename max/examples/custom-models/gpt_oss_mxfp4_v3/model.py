"""ModuleV3 pipeline model for GPT-OSS with MXFP4 expert GEMMs."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from max.driver import Device, Tensor
from max.dtype import DType
from max.graph import DeviceRef, TensorType
from max.pipelines.architectures.gpt_oss_module_v3.model import (
    GptOssModel as _BaseGptOssModel,
)

from .gpt_oss import GptOss
from .kernels import get_mxfp4_kernels_path
from .model_config import GptOssConfig
from .module_v3_compile import compile_with_custom_extensions

logger = logging.getLogger("max.pipelines")


class GptOssModelModuleV3MXFP4(_BaseGptOssModel):
    """GPT-OSS ModuleV3 pipeline model that loads MXFP4 custom ops."""

    def load_model(self) -> Callable[..., Any]:
        assert self.pipeline_config.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        self._input_row_offsets_prealloc = Tensor.from_numpy(
            np.arange(self.pipeline_config.max_batch_size + 1, dtype=np.uint32)
        ).to(self.devices[0])

        logger.info("Building and compiling model...")
        before = time.perf_counter()

        device0: Device = self.devices[0]
        device_ref = DeviceRef(device0.label, device0.id)
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32,
            shape=["input_row_offsets_len"],
            device=device0,
        )
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

        nn_model = GptOss(model_config, self.kv_manager)
        nn_model.to(self.devices[0])

        kv_inputs = self.kv_params.get_symbolic_inputs()
        flattened_kv_types = [
            kv_type for sublist in kv_inputs for kv_type in sublist
        ]

        compiled_model = compile_with_custom_extensions(
            nn_model,
            tokens_type,
            return_n_logits_type,
            input_row_offsets_type,
            *flattened_kv_types,
            weights=state_dict,
            custom_extensions=[get_mxfp4_kernels_path()],
        )

        after = time.perf_counter()
        logger.info(
            f"Building and compiling model took {after - before:.6f} seconds"
        )
        return compiled_model


__all__ = ["GptOssModelModuleV3MXFP4"]
