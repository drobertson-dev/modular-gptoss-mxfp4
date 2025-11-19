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
    thread_idx,
)
from gpu.host import DeviceContext
from gpu.host.info import is_cpu, is_gpu
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
](
    c: NDBuffer[mut=True, c_type, 2, MutAnyOrigin],
    a: NDBuffer[a_type, 2, MutAnyOrigin],
    packed_b: NDBuffer[DType.uint8, 3, MutAnyOrigin],
    scales: NDBuffer[DType.uint8, 3, MutAnyOrigin],
    expert_offsets: NDBuffer[DType.uint32, 1, MutAnyOrigin],
    expert_ids: NDBuffer[DType.int32, 1, MutAnyOrigin],
) raises:
    var tokens_for_expert: UInt = UInt(
        expert_offsets[Int(block_idx.z) + 1] - expert_offsets[Int(block_idx.z)]
    )
    var num_outputs = packed_b.dim[1]()
    var packed_stride = packed_b.dim[2]()
    var in_features = Int(packed_stride) * 2
    var scale_stride = scales.dim[2]()

    var a_start_row = expert_offsets[Int(block_idx.z)]
    var a_data = a.data + a_start_row * UInt(in_features)

    var expert = expert_ids[Int(block_idx.z)]
    var packed_by_expert = packed_b.data + expert * num_outputs * packed_stride
    var scales_by_expert = scales.data + expert * num_outputs * scale_stride

    var n = global_idx.x
    var m = global_idx.y

    if n >= UInt(num_outputs) or m >= tokens_for_expert:
        return

    alias accum_type = Float32
    var accum = Scalar[accum_type](0.0)

    if expert != -1:
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
            var a_val = a_data[m * UInt(in_features) + UInt(k)].cast[accum_type]()
            accum += a_val * scaled_weight
        endfor
    endif

    var c_data = c.data + a_start_row * num_outputs
    c_data[m * UInt(num_outputs) + n] = accum.cast[c_type]()


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
    alias kernel = mxfp4_grouped_matmul_kernel[
        c_type,
        a_type,
    ]
    ctx.enqueue_function_checked[kernel, kernel](
        c,
        a,
        packed_b,
        scales,
        expert_offsets,
        expert_ids,
        grid_dim=(
            ceildiv(c.dim[1](), 32),
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
    var in_features = Int(packed_stride) * 2
    var scale_stride = scales.dim[2]()

    for expert_idx in range(num_active_experts):
        var expert = expert_ids[expert_idx]
        var token_start = expert_offsets[expert_idx]
        var token_end = expert_offsets[expert_idx + 1]
        var tokens = token_end - token_start

        if expert == -1 or tokens <= 0:
            continue

        var a_slice = a.data + token_start * UInt(in_features)
        var out_slice = c.data + token_start * num_outputs
        var packed_row = packed_b.data + expert * num_outputs * packed_stride
        var scale_row = scales.data + expert * num_outputs * scale_stride

        for m in range(Int(tokens)):
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
                    var a_val = a_slice[m * UInt(in_features) + UInt(k)].cast[
                        Float32
                    ]()
                    accum += a_val * weight
                endfor
                out_slice[m * num_outputs + n] = Scalar[c_type](accum)
            endfor
        endfor
    endfor


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
