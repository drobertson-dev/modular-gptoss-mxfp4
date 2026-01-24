"""GPT-OSS Module architecture registration (MXFP4 experts + Mojo custom ops)."""

from __future__ import annotations

from max.graph.weights import WeightsFormat
from max.interfaces import PipelineTask
from max.nn.legacy.kv_cache import KVCacheStrategy
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    RopeType,
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
)

from . import weight_adapters
from .model_module_v3 import GptOssModelModuleV3

gpt_oss_module_v3_arch = SupportedArchitecture(
    # Match the built-in module architecture name so this package overrides it
    # when used via `--custom-architectures ... --no-use-legacy-module`.
    name="GptOssForCausalLM",
    example_repo_ids=[
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
    ],
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
    },
    pipeline_model=GptOssModelModuleV3,
    task=PipelineTask.TEXT_GENERATION,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=False,
    rope_type=RopeType.yarn,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
)

__all__ = ["WeightsFormat", "gpt_oss_module_v3_arch"]
