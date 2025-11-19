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

from math import ceildiv, ldexp

from buffer.buffer import NDBuffer
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    global_idx,
    grid_dim,
)
from gpu.host import DeviceContext
from gpu.host.info import is_cpu, is_gpu
from layout import Layout, LayoutTensor
from layout._ndbuffer_stub import from_ndbuffer_row_major
from memory import LegacyUnsafePointer as UnsafePointer
from utils.static_tuple import StaticTuple

alias _FP4_VALUES = StaticTuple[
    Float32,
    16,
](
    Float32(0.0),
    Float32(0.5),
    Float32(1.0),
    Float32(1.5),
    Float32(2.0),
    Float32(3.0),
    Float32(4.0),
    Float32(6.0),
    Float32(-0.0),
    Float32(-0.5),
    Float32(-1.0),
    Float32(-1.5),
    Float32(-2.0),
    Float32(-3.0),
    Float32(-4.0),
    Float32(-6.0),
)


@always_inline
fn _scale_multiplier(scale_byte: UInt8) -> Float32:
    var exponent = Int(scale_byte) - 127
    return ldexp(Float32(1.0), exponent)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](512)
)
fn mxfp4_grouped_matmul_kernel[
    c_type: DType,
    a_type: DType,
    c_layout: Layout,
    a_layout: Layout,
    packed_layout: Layout,
    scale_layout: Layout,
    offsets_layout: Layout,
    ids_layout: Layout,
](
    c: LayoutTensor[mut=True, c_type, c_layout, MutAnyOrigin],
    a: LayoutTensor[a_type, a_layout, MutAnyOrigin],
    packed_b: LayoutTensor[DType.uint8, packed_layout, MutAnyOrigin],
    scales: LayoutTensor[DType.uint8, scale_layout, MutAnyOrigin],
    expert_offsets: LayoutTensor[DType.uint32, offsets_layout, MutAnyOrigin],
    expert_ids: LayoutTensor[DType.int32, ids_layout, MutAnyOrigin],
) raises:
    var expert_block = Int(block_idx.z)
    var offsets_ptr = expert_offsets.data
    var ids_ptr = expert_ids.data
    var start_offset = Int(offsets_ptr[expert_block].value)
    var end_offset = Int(offsets_ptr[expert_block + 1].value)
    var tokens_for_expert = end_offset - start_offset
    if tokens_for_expert <= 0:
        return

    var num_outputs = packed_b.dim(1)
    var packed_stride = packed_b.dim(2)
    var in_features = packed_stride * 2
    var scale_stride = scales.dim(2)

    var a_data = a.data + UInt(start_offset * in_features)
    var expert = Int(ids_ptr[expert_block].value)

    if expert == -1:
        return

    var packed_by_expert = (
        packed_b.data + UInt(expert * num_outputs * packed_stride)
    )
    var scales_by_expert = (
        scales.data + UInt(expert * num_outputs * scale_stride)
    )

    var n = Int(global_idx.x)
    var m = Int(global_idx.y)

    if n >= num_outputs or m >= tokens_for_expert:
        return

    var accum = Float32(0.0)

    for k in range(in_features):
        var packed_index = k >> 1
        var packed_byte = packed_by_expert[
            n * packed_stride + packed_index
        ]
        var nibble: UInt8
        if (k & 1) == 0:
            nibble = packed_byte & UInt8(0x0F)
        else:
            nibble = packed_byte >> 4

        var weight = _FP4_VALUES[Int(nibble)]
        var scale_block = k // 32
        var scale_byte = scales_by_expert[
            n * scale_stride + scale_block
        ]
        var scaled_weight = weight * _scale_multiplier(scale_byte)
        var a_val = a_data[
            UInt(m * in_features + k)
        ].cast[DType.float32]()
        accum += a_val * scaled_weight

    var c_data = c.data + UInt(start_offset * num_outputs)
    c_data[UInt(m * num_outputs + n)] = Scalar[c_type](accum)


fn _mxfp4_grouped_matmul_gpu[
    c_type: DType,
    a_type: DType,
](
    c: NDBuffer[mut=True, c_type, 2, MutAnyOrigin],
    a: NDBuffer[a_type, 2, MutAnyOrigin],
    packed_b: NDBuffer[DType.uint8, 3, MutAnyOrigin],
    scales: NDBuffer[DType.uint8, 3, MutAnyOrigin],
    expert_offsets: NDBuffer[DType.uint32, 1, MutAnyOrigin],
    expert_ids: NDBuffer[DType.int32, 1, MutAnyOrigin],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    var c_tensor = from_ndbuffer_row_major(c)
    var a_tensor = from_ndbuffer_row_major(a)
    var packed_tensor = from_ndbuffer_row_major(packed_b)
    var scale_tensor = from_ndbuffer_row_major(scales)
    var offsets_tensor = from_ndbuffer_row_major(expert_offsets)
    var ids_tensor = from_ndbuffer_row_major(expert_ids)

    alias kernel = mxfp4_grouped_matmul_kernel[
        c_type,
        a_type,
        c_tensor.layout,
        a_tensor.layout,
        packed_tensor.layout,
        scale_tensor.layout,
        offsets_tensor.layout,
        ids_tensor.layout,
    ]
    ctx.enqueue_function_checked[kernel, kernel](
        c_tensor,
        a_tensor,
        packed_tensor,
        scale_tensor,
        offsets_tensor,
        ids_tensor,
        grid_dim=(
            ceildiv(c_tensor.dim(1), 32),
            ceildiv(max_num_tokens_per_expert, 16),
            num_active_experts,
        ),
        block_dim=(32, 16, 1),
    )


fn _decode_weight(
    packed_row: UnsafePointer[UInt8],
    scale_row: UnsafePointer[UInt8],
    packed_stride: Int,
    scale_stride: Int,
    out_idx: Int,
    k: Int,
) -> Float32:
    var packed_byte = packed_row[out_idx * packed_stride + (k >> 1)]
    var nibble: UInt8
    if (k & 1) == 0:
        nibble = packed_byte & UInt8(0x0F)
    else:
        nibble = packed_byte >> 4
    var base = _FP4_VALUES[Int(nibble)]
    var scale_byte = scale_row[out_idx * scale_stride + (k // 32)]
    return base * _scale_multiplier(scale_byte)


fn _mxfp4_grouped_matmul_cpu[
    c_type: DType,
    a_type: DType,
](
    c: NDBuffer[mut=True, c_type, 2, MutAnyOrigin],
    a: NDBuffer[a_type, 2, MutAnyOrigin],
    packed_b: NDBuffer[DType.uint8, 3, MutAnyOrigin],
    scales: NDBuffer[DType.uint8, 3, MutAnyOrigin],
    expert_offsets: NDBuffer[DType.uint32, 1, MutAnyOrigin],
    expert_ids: NDBuffer[DType.int32, 1, MutAnyOrigin],
    num_active_experts: Int,
) raises:
    var num_outputs = packed_b.dim[1]()
    var packed_stride = packed_b.dim[2]()
    var in_features = packed_stride * 2
    var scale_stride = scales.dim[2]()

    for expert_idx in range(num_active_experts):
        var expert = Int(expert_ids[expert_idx])
        var token_start = Int(expert_offsets[expert_idx])
        var token_end = Int(expert_offsets[expert_idx + 1])
        var tokens = token_end - token_start

        if expert == -1 or tokens <= 0:
            continue

        var a_slice = a.data + UInt(token_start * in_features)
        var out_slice = c.data + UInt(token_start * num_outputs)
        var packed_row = packed_b.data + expert * num_outputs * packed_stride
        var scale_row = scales.data + expert * num_outputs * scale_stride

        for m in range(tokens):
            for n in range(num_outputs):
                var accum = Float32(0.0)
                for k in range(in_features):
                    var weight = _decode_weight(
                        packed_row,
                        scale_row,
                        packed_stride,
                        scale_stride,
                        n,
                        k,
                    )
                    var a_val = a_slice[
                        UInt(m * in_features + k)
                    ].cast[DType.float32]()
                    accum += a_val * weight
                out_slice[UInt(m * num_outputs + n)] = Scalar[c_type](accum)


fn mxfp4_grouped_matmul[
    c_type: DType,
    a_type: DType,
    target: StaticString,
](
    c: NDBuffer[mut=True, c_type, 2, MutAnyOrigin],
    a: NDBuffer[a_type, 2, MutAnyOrigin],
    packed_b: NDBuffer[DType.uint8, 3, MutAnyOrigin],
    scales: NDBuffer[DType.uint8, 3, MutAnyOrigin],
    expert_offsets: NDBuffer[DType.uint32, 1, MutAnyOrigin],
    expert_ids: NDBuffer[DType.int32, 1, MutAnyOrigin],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    if is_gpu[target]():
        _mxfp4_grouped_matmul_gpu[c_type, a_type](
            c,
            a,
            packed_b,
            scales,
            expert_offsets,
            expert_ids,
            max_num_tokens_per_expert,
            num_active_experts,
            ctx,
        )
    elif is_cpu[target]():
        _mxfp4_grouped_matmul_cpu[c_type, a_type](
            c,
            a,
            packed_b,
            scales,
            expert_offsets,
            expert_ids,
            num_active_experts,
        )
    else:
        constrained[False, "Unsupported target for MXFP4 matmul"]()
