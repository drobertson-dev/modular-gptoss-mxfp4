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

import os

import numpy as np
import pytest
from max.dtype import DType
from max.driver import Tensor
from max.driver import accelerator_count
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.kernels import grouped_mxfp4_matmul
from max.pipelines.lib.custom_extensions import (
    collect_custom_extensions_from_env,
)

_CUSTOM_EXTENSION_ENV = (
    os.environ.get("MAX_CUSTOM_EXTENSIONS")
    or os.environ.get("MXFP4_KERNEL_PACKAGE")
)
_CUSTOM_EXTENSIONS = collect_custom_extensions_from_env(
    _CUSTOM_EXTENSION_ENV,
    include_runtime_dependencies=True,
)

if not _CUSTOM_EXTENSIONS:
    pytest.skip(
        "MXFP4 kernel package unavailable. Set MAX_CUSTOM_EXTENSIONS or "
        "MXFP4_KERNEL_PACKAGE to point at MOGGKernelAPI.mojopkg.",
        allow_module_level=True,
    )

_FP4_VALUES = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def _quantize_row(row: np.ndarray, scale_exp: int) -> tuple[np.ndarray, np.uint8]:
    """Quantizes a row of 32 values into packed nibbles."""
    scale = np.float32(2.0**scale_exp)
    scaled = row / scale
    idxs = np.abs(_FP4_VALUES[None, :] - scaled[:, None]).argmin(axis=1).astype(np.uint8)
    bytes_view = idxs.reshape(-1, 2)
    packed = (bytes_view[:, 0] & 0x0F) | ((bytes_view[:, 1] & 0x0F) << 4)
    return packed.astype(np.uint8), np.uint8(scale_exp + 127)


def _quantize_mxfp4(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (blocks, scales, dense_reference)."""
    experts, out_features, in_features = weights.shape
    assert in_features % 32 == 0

    blocks = np.zeros((experts, out_features, in_features // 2), dtype=np.uint8)
    scales = np.zeros((experts, out_features, in_features // 32), dtype=np.uint8)
    dense = np.zeros_like(weights, dtype=np.float32)

    for e in range(experts):
        for row in range(out_features):
            for blk in range(in_features // 32):
                chunk = weights[e, row, blk * 32 : (blk + 1) * 32]
                max_abs = np.max(np.abs(chunk))
                if max_abs == 0:
                    scale_exp = 0
                else:
                    scale_exp = int(np.floor(np.log2(max_abs / 6.0)))
                    scale_exp = min(max(scale_exp, -127), 127)
                packed, scale_byte = _quantize_row(chunk, scale_exp)
                blocks[e, row, blk * 16 : (blk + 1) * 16] = packed
                scales[e, row, blk] = scale_byte

                decoded = np.empty(32, dtype=np.float32)
                for idx in range(32):
                    byte = packed[idx // 2]
                    if idx % 2 == 0:
                        nibble = byte & 0x0F
                    else:
                        nibble = byte >> 4
                    decoded[idx] = _FP4_VALUES[nibble] * np.float32(2.0**scale_exp)
                dense[e, row, blk * 32 : (blk + 1) * 32] = decoded

    return blocks, scales, dense


def _build_reference_outputs(
    hidden_states: np.ndarray,
    dense_weights: np.ndarray,
    bias: np.ndarray,
    expert_starts: np.ndarray,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for expert, (start, end) in enumerate(zip(expert_starts[:-1], expert_starts[1:], strict=True)):
        if end <= start:
            continue
        act = hidden_states[start:end]
        weight = dense_weights[expert]
        outputs.append(act @ weight.T + bias[expert])
    return np.vstack(outputs)


def test_mxfp4_grouped_matmul_matches_dense(session: InferenceSession) -> None:
    rng = np.random.default_rng(0)
    num_experts = 2
    tokens_per_expert = [3, 2]
    in_features = 64
    out_features = 16

    total_tokens = sum(tokens_per_expert)
    hidden = rng.standard_normal((total_tokens, in_features), dtype=np.float32).astype(np.float32)

    dense_weights = rng.standard_normal(
        (num_experts, out_features, in_features), dtype=np.float32
    ).astype(np.float32)
    blocks, scales, decoded = _quantize_mxfp4(dense_weights)
    biases = rng.standard_normal((num_experts, out_features), dtype=np.float32)

    expert_offsets = np.zeros(num_experts + 1, dtype=np.uint32)
    cursor = 0
    for idx, count in enumerate(tokens_per_expert, start=1):
        cursor += count
        expert_offsets[idx] = cursor

    expert_ids = np.arange(num_experts, dtype=np.int32)
    stats = np.array([max(tokens_per_expert), num_experts], dtype=np.uint32)

    graph = Graph(
        "mxfp4_grouped_matmul",
        input_types=[
            TensorType(DType.float32, hidden.shape, device=DeviceRef.CPU()),
            TensorType(DType.uint8, blocks.shape, device=DeviceRef.CPU()),
            TensorType(DType.uint8, scales.shape, device=DeviceRef.CPU()),
            TensorType(DType.float32, biases.shape, device=DeviceRef.CPU()),
            TensorType(DType.uint32, expert_offsets.shape, device=DeviceRef.CPU()),
            TensorType(DType.int32, expert_ids.shape, device=DeviceRef.CPU()),
            TensorType(DType.uint32, stats.shape, device=DeviceRef.CPU()),
        ],
        output_types=[
            TensorType(DType.float32, (total_tokens, out_features), device=DeviceRef.CPU()),
        ],
        custom_extensions=_CUSTOM_EXTENSIONS,
    )

    with graph:
        out = grouped_mxfp4_matmul(
            graph.inputs[0].tensor,
            graph.inputs[1].tensor,
            graph.inputs[2].tensor,
            graph.inputs[3].tensor,
            graph.inputs[4].tensor,
            graph.inputs[5].tensor,
            graph.inputs[6].tensor,
        )
        graph.output(out)

    compiled = session.load(graph, custom_extensions=_CUSTOM_EXTENSIONS)
    (result,) = compiled.execute(hidden, blocks, scales, biases, expert_offsets, expert_ids, stats)
    produced = result.to_numpy()

    reference = _build_reference_outputs(hidden, decoded, biases, expert_offsets)
    np.testing.assert_allclose(produced, reference, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(
    accelerator_count() == 0, reason="GPU required for MXFP4 kernel test"
)
def test_mxfp4_grouped_matmul_matches_dense_gpu(
    session: InferenceSession,
) -> None:
    rng = np.random.default_rng(1)
    num_experts = 2
    tokens_per_expert = [4, 1]
    in_features = 64
    out_features = 16

    total_tokens = sum(tokens_per_expert)
    hidden = rng.standard_normal((total_tokens, in_features), dtype=np.float32)

    dense_weights = rng.standard_normal(
        (num_experts, out_features, in_features), dtype=np.float32
    ).astype(np.float32)
    blocks, scales, decoded = _quantize_mxfp4(dense_weights)
    biases = rng.standard_normal((num_experts, out_features), dtype=np.float32)

    expert_offsets = np.zeros(num_experts + 1, dtype=np.uint32)
    cursor = 0
    for idx, count in enumerate(tokens_per_expert, start=1):
        cursor += count
        expert_offsets[idx] = cursor

    expert_ids = np.arange(num_experts, dtype=np.int32)
    stats = np.array([max(tokens_per_expert), num_experts], dtype=np.uint32)

    graph = Graph(
        "mxfp4_grouped_matmul_gpu",
        input_types=[
            TensorType(DType.float32, hidden.shape, device=DeviceRef.GPU()),
            TensorType(DType.uint8, blocks.shape, device=DeviceRef.GPU()),
            TensorType(DType.uint8, scales.shape, device=DeviceRef.GPU()),
            TensorType(DType.float32, biases.shape, device=DeviceRef.GPU()),
            TensorType(DType.uint32, expert_offsets.shape, device=DeviceRef.GPU()),
            TensorType(DType.int32, expert_ids.shape, device=DeviceRef.GPU()),
            TensorType(DType.uint32, stats.shape, device=DeviceRef.CPU()),
        ],
        output_types=[
            TensorType(
                DType.float32,
                (total_tokens, out_features),
                device=DeviceRef.GPU(),
            ),
        ],
        custom_extensions=_CUSTOM_EXTENSIONS,
    )

    with graph:
        out = grouped_mxfp4_matmul(
            graph.inputs[0].tensor,
            graph.inputs[1].tensor,
            graph.inputs[2].tensor,
            graph.inputs[3].tensor,
            graph.inputs[4].tensor,
            graph.inputs[5].tensor,
            graph.inputs[6].tensor,
        )
        graph.output(out)

    compiled = session.load(graph, custom_extensions=_CUSTOM_EXTENSIONS)

    numpy_inputs = [
        hidden,
        blocks,
        scales,
        biases,
        expert_offsets,
        expert_ids,
        stats,
    ]
    tensor_inputs: list[Tensor] = []
    for arr, device in zip(numpy_inputs, compiled.input_devices, strict=True):
        tensor_inputs.append(Tensor.from_dlpack(arr).to(device))

    (result,) = compiled.execute(*tensor_inputs)
    produced = result.to_numpy()

    reference = _build_reference_outputs(hidden, decoded, biases, expert_offsets)
    np.testing.assert_allclose(produced, reference, rtol=2e-2, atol=2e-2)
