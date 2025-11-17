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

import numpy as np
from max.dtype import DType as MaxDType
from max.graph.weights import WeightData, Weights

GPT_OSS_SAFETENSOR_MAP: dict[str, str] = {
    "model.embed_tokens.": "language_model.embed_tokens.",
    "model.norm.": "language_model.norm.",
    "lm_head.": "language_model.lm_head.",
    "model.layers.": "language_model.layers.",
    # MoE weight mappings
    ".mlp.router": ".mlp.gate.gate_score",
}


def _map_weight_name(name: str) -> str:
    mapped = name
    for before, after in GPT_OSS_SAFETENSOR_MAP.items():
        mapped = mapped.replace(before, after)
    if mapped.endswith("_blocks"):
        mapped = f"{mapped[: -len('_blocks')]}.weight.blocks"
    elif mapped.endswith("_scales"):
        mapped = f"{mapped[: -len('_scales')]}.weight.scales"
    return mapped


def _as_numpy(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "to_numpy"):
        tensor = value
        orig_dtype = getattr(tensor, "dtype", None)
        try:
            return np.asarray(tensor.to_numpy())
        except RuntimeError:
            cast = tensor.view(MaxDType.int32)
            arr = np.asarray(cast.to_numpy())
            if orig_dtype == MaxDType.uint8:
                return arr.view(np.uint8)
            return arr
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _convert_blocks_to_qe(
    blocks: np.ndarray, scales: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert MXFP4 block/scales tensors to packed Q/E layout."""
    blocks_np = np.asarray(blocks, dtype=np.uint8)
    scales_np = np.asarray(scales, dtype=np.uint8)
    if blocks_np.shape[-1] != 16:
        raise ValueError(
            f"Expected MXFP4 blocks to have trailing dimension 16, got {blocks_np.shape}."
        )
    prefix = blocks_np.shape[:-2]
    group = blocks_np.shape[-2]
    bytes_per_group = blocks_np.shape[-1]
    q = blocks_np.reshape(*prefix, group * bytes_per_group)
    e = scales_np.reshape(*prefix, group)
    return q, e


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights], **kwargs
) -> dict[str, WeightData]:
    """Convert safetensor state dict to MAX format.

    Args:
        state_dict: Dictionary of weight tensors

    Returns:
        Dictionary of converted weight data
    """

    new_state: dict[str, WeightData] = {}
    pending_mxfp4: dict[str, dict[str, np.ndarray]] = defaultdict(dict)

    for weight_name, accessor in state_dict.items():
        mapped_name = _map_weight_name(weight_name)
        weight_data = accessor.data()
        value_np = _as_numpy(weight_data.data)

        if mapped_name.endswith(".blocks"):
            base = mapped_name[: -len(".blocks")]
            pending_mxfp4[base]["blocks"] = value_np
            continue
        if mapped_name.endswith(".scales"):
            base = mapped_name[: -len(".scales")]
            pending_mxfp4[base]["scales"] = value_np
            continue

        new_state[mapped_name] = weight_data

    for base, tensors in pending_mxfp4.items():
        if "blocks" not in tensors or "scales" not in tensors:
            raise ValueError(
                f"Incomplete MXFP4 tensor for '{base}': found {list(tensors.keys())}"
            )
        q, e = _convert_blocks_to_qe(tensors["blocks"], tensors["scales"])
        new_state[f"{base}.q"] = WeightData.from_numpy(
            np.ascontiguousarray(q, dtype=np.uint8), f"{base}.q"
        )
        new_state[f"{base}.e"] = WeightData.from_numpy(
            np.ascontiguousarray(e, dtype=np.uint8), f"{base}.e"
        )

    return new_state
