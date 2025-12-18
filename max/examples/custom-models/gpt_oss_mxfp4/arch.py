"""GPT-OSS architecture registration (MXFP4 weights + custom Mojo ops)."""

from __future__ import annotations

from max.graph.weights import WeightsFormat
from max.interfaces import PipelineTask
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    RopeType,
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
)

from . import weight_adapters
from .model import GptOssModel

gpt_oss_arch = SupportedArchitecture(
    # Keep the same name as the built-in architecture so this custom package
    # overrides it when passed via `--custom-architectures`.
    name="GptOssForCausalLM",
    example_repo_ids=[
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
    ],
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
    },
    pipeline_model=GptOssModel,
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

__all__ = ["WeightsFormat", "gpt_oss_arch"]
