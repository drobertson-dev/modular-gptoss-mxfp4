# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from collections import defaultdict

from max.dtype import DType
from max.graph.quantization import QuantizationEncoding
from max.graph.weights import WeightData, Weights

GPT_OSS_SAFETENSOR_MAP: dict[str, str] = {
    "model.embed_tokens.": "language_model.embed_tokens.",
    "model.norm.": "language_model.norm.",
    "lm_head.": "language_model.lm_head.",
    "model.layers.": "language_model.layers.",
    # MoE weight mappings
    ".mlp.router": ".mlp.gate.gate_score",
}

_EXPERT_RENAMES: dict[str, str] = {
    ".mlp.mlp1_weight": ".mlp.experts.gate_up_proj",
    ".mlp.mlp1_bias": ".mlp.experts.gate_up_proj_bias",
    ".mlp.mlp2_weight": ".mlp.experts.down_proj",
    ".mlp.mlp2_bias": ".mlp.experts.down_proj_bias",
}

_MXFP4_REQUIRED_SUFFIXES = {"blocks", "scales"}


def _maybe_mark_mxfp4_weight(
    name: str, weight_data: WeightData, tracking: dict[str, set[str]]
) -> WeightData:
    if ".mlp.experts." not in name:
        return weight_data

    if name.endswith(".blocks"):
        suffix = "blocks"
    elif name.endswith(".scales"):
        suffix = "scales"
    else:
        return weight_data

    base_name = name[: -(len(suffix) + 1)]
    tracking[base_name].add(suffix)

    if weight_data.dtype != DType.uint8:
        weight_data = weight_data.astype(DType.uint8)
        weight_data.name = name

    weight_data.quantization_encoding = QuantizationEncoding.MXFP4
    return weight_data

def convert_safetensor_state_dict(
    state_dict: dict[str, Weights], **kwargs
) -> dict[str, WeightData]:
    """Convert safetensor state dict to MAX format.

    Args:
        state_dict: Dictionary of weight tensors

    Returns:
        Dictionary of converted weight data
    """

    # Now remap all weight names from HuggingFace to MAX format
    new_state_dict: dict[str, WeightData] = {}
    mxfp4_pairs: dict[str, set[str]] = defaultdict(set)

    for weight_name, value in state_dict.items():
        max_name: str = weight_name
        for before, after in GPT_OSS_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)
        for before, after in _EXPERT_RENAMES.items():
            max_name = max_name.replace(before, after)
        # Normalize MXFP4 suffix naming to use dotted separators.
        if max_name.endswith("_blocks"):
            max_name = max_name[: -len("_blocks")] + ".blocks"
        elif max_name.endswith("_scales"):
            max_name = max_name[: -len("_scales")] + ".scales"
        weight_data = value.data()
        weight_data.name = max_name
        weight_data = _maybe_mark_mxfp4_weight(
            max_name, weight_data, mxfp4_pairs
        )
        new_state_dict[max_name] = weight_data

    missing_pairs = [
        base_name
        for base_name, suffixes in mxfp4_pairs.items()
        if suffixes != _MXFP4_REQUIRED_SUFFIXES
    ]
    if missing_pairs:
        formatted = ", ".join(sorted(missing_pairs))
        raise ValueError(
            "Incomplete MXFP4 tensors detected; found only one of blocks/scales for "
            f"{formatted}. Please ensure the checkpoint includes both tensors."
        )

    return new_state_dict
