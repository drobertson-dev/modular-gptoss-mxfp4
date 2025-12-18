"""GPT-OSS config re-export.

We reuse the built-in GPT-OSS config derivation logic from MAX to avoid
diverging from the known-correct HuggingFace mapping. The MXFP4-specific
behavior lives in the MoE layer and Mojo custom ops, not the config.
"""

from __future__ import annotations

from max.pipelines.architectures.gpt_oss.model_config import (
    GptOssConfig,
    GptOssConfigBase,
)

__all__ = ["GptOssConfig", "GptOssConfigBase"]
