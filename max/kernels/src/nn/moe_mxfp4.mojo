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
    barrier,
)
from gpu.host import DeviceContext
from gpu.host.info import is_cpu, is_gpu
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from layout._ndbuffer_stub import from_ndbuffer_row_major
from memory import LegacyUnsafePointer as UnsafePointer, stack_allocation
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

alias _FP4_PER_BLOCK = 32
alias _PACKED_BYTES_PER_BLOCK = 16
alias _OUTPUT_TILE = 64
alias _TOKEN_TILE = 2
alias _K_BLOCKS_PER_TILE = 4
alias _K_TILE = _K_BLOCKS_PER_TILE * _FP4_PER_BLOCK
alias _WEIGHT_TILE_SIZE = _OUTPUT_TILE * _K_TILE
alias _ACT_TILE_SIZE = _TOKEN_TILE * _K_TILE


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
    bias_layout: Layout,
    offsets_layout: Layout,
    ids_layout: Layout,
](
    c: LayoutTensor[mut=True, c_type, c_layout, MutAnyOrigin],
    a: LayoutTensor[a_type, a_layout, MutAnyOrigin],
    packed_b: LayoutTensor[DType.uint8, packed_layout, MutAnyOrigin],
    scales: LayoutTensor[DType.uint8, scale_layout, MutAnyOrigin],
    bias: LayoutTensor[c_type, bias_layout, MutAnyOrigin],
    expert_offsets: LayoutTensor[DType.uint32, offsets_layout, MutAnyOrigin],
    expert_ids: LayoutTensor[DType.int32, ids_layout, MutAnyOrigin],
):
    var expert_block = Int(block_idx.z)
    var offsets_ptr = expert_offsets.ptr
    var ids_ptr = expert_ids.ptr
    var start_offset = Int(offsets_ptr[expert_block])
    var end_offset = Int(offsets_ptr[expert_block + 1])
    var tokens_for_expert = end_offset - start_offset
    if tokens_for_expert <= 0:
        return

    var num_outputs = packed_b.dim(1)
    var packed_stride = packed_b.dim(2)
    var in_features = packed_stride * 2
    var scale_stride = scales.dim(2)

    var a_data = a.ptr + UInt(start_offset * in_features)
    var expert = Int(ids_ptr[expert_block])

    if expert == -1:
        return

    var packed_by_expert = (
        packed_b.ptr + UInt(expert * num_outputs * packed_stride)
    )
    var scales_by_expert = (
        scales.ptr + UInt(expert * num_outputs * scale_stride)
    )
    var bias_by_expert = (
        bias.ptr + UInt(expert * num_outputs)
    )

    var n = Int(global_idx.x)
    var m = Int(global_idx.y)
    var local_n = Int(thread_idx.x)
    var active = n < num_outputs and m < tokens_for_expert

    var accum = Float32(0.0)

    var shared_weights = stack_allocation[_WEIGHT_TILE_SIZE, Float32, address_space = AddressSpace.SHARED]()
    var shared_act = stack_allocation[_ACT_TILE_SIZE, Float32, address_space = AddressSpace.SHARED]()

    var num_blocks = in_features // _FP4_PER_BLOCK
    var k_block_start = 0
    while k_block_start < num_blocks:
        var blocks_this_tile = num_blocks - k_block_start
        if blocks_this_tile > _K_BLOCKS_PER_TILE:
            blocks_this_tile = _K_BLOCKS_PER_TILE
        var k_tile = blocks_this_tile * _FP4_PER_BLOCK

        if thread_idx.y == 0 and n < num_outputs:
            var packed_row = packed_by_expert + UInt(n * packed_stride)
            var scale_row = scales_by_expert + UInt(n * scale_stride)
            var shared_base = local_n * _K_TILE

            for block_inner in range(blocks_this_tile):
                var scale_block = k_block_start + block_inner
                var scale_mul = _scale_multiplier(scale_row[scale_block])
                var packed_block = packed_row + UInt(scale_block * _PACKED_BYTES_PER_BLOCK)
                var block_offset = shared_base + block_inner * _FP4_PER_BLOCK

                for byte_base in range(0, _PACKED_BYTES_PER_BLOCK, 4):
                    var b0 = packed_block[byte_base]
                    var b1 = packed_block[byte_base + 1]
                    var b2 = packed_block[byte_base + 2]
                    var b3 = packed_block[byte_base + 3]

                    var base_idx = block_offset + byte_base * 2
                    shared_weights[base_idx] = _FP4_VALUES[Int(b0 & UInt8(0x0F))] * scale_mul
                    shared_weights[base_idx + 1] = _FP4_VALUES[Int(b0 >> 4)] * scale_mul
                    shared_weights[base_idx + 2] = _FP4_VALUES[Int(b1 & UInt8(0x0F))] * scale_mul
                    shared_weights[base_idx + 3] = _FP4_VALUES[Int(b1 >> 4)] * scale_mul
                    shared_weights[base_idx + 4] = _FP4_VALUES[Int(b2 & UInt8(0x0F))] * scale_mul
                    shared_weights[base_idx + 5] = _FP4_VALUES[Int(b2 >> 4)] * scale_mul
                    shared_weights[base_idx + 6] = _FP4_VALUES[Int(b3 & UInt8(0x0F))] * scale_mul
                    shared_weights[base_idx + 7] = _FP4_VALUES[Int(b3 >> 4)] * scale_mul

        if m < tokens_for_expert:
            var a_row = a_data + UInt(m * in_features)
            var k_index = Int(thread_idx.x)
            while k_index < k_tile:
                shared_act[
                    Int(thread_idx.y) * _K_TILE + k_index
                ] = a_row[
                    UInt(k_block_start * _FP4_PER_BLOCK + k_index)
                ].cast[DType.float32]()
                k_index += _OUTPUT_TILE

        barrier()

        if active:
            var shared_base = local_n * _K_TILE

            var act_base = Int(thread_idx.y) * _K_TILE
            for k_offset in range(0, k_tile, 4):
                var a0 = shared_act[act_base + k_offset]
                var a1 = shared_act[act_base + k_offset + 1]
                var a2 = shared_act[act_base + k_offset + 2]
                var a3 = shared_act[act_base + k_offset + 3]

                var w0 = shared_weights[shared_base + k_offset]
                var w1 = shared_weights[shared_base + k_offset + 1]
                var w2 = shared_weights[shared_base + k_offset + 2]
                var w3 = shared_weights[shared_base + k_offset + 3]

                accum += a0 * w0 + a1 * w1 + a2 * w2 + a3 * w3

        k_block_start += _K_BLOCKS_PER_TILE

    var c_data = c.ptr + UInt(start_offset * num_outputs)
    if active:
        accum += bias_by_expert[UInt(n)].cast[DType.float32]()
        c_data[UInt(m * num_outputs + n)] = Scalar[c_type](accum)


fn _mxfp4_grouped_matmul_gpu[
    c_type: DType,
    a_type: DType,
](
    c: NDBuffer[mut=True, c_type, 2, MutAnyOrigin],
    a: NDBuffer[a_type, 2, MutAnyOrigin],
    packed_b: NDBuffer[DType.uint8, 3, MutAnyOrigin],
    scales: NDBuffer[DType.uint8, 3, MutAnyOrigin],
    bias: NDBuffer[c_type, 2, MutAnyOrigin],
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
    var bias_tensor = from_ndbuffer_row_major(bias)
    var offsets_tensor = from_ndbuffer_row_major(expert_offsets)
    var ids_tensor = from_ndbuffer_row_major(expert_ids)

    alias kernel = mxfp4_grouped_matmul_kernel[
        c_type,
        a_type,
        c_tensor.layout,
        a_tensor.layout,
        packed_tensor.layout,
        scale_tensor.layout,
        bias_tensor.layout,
        offsets_tensor.layout,
        ids_tensor.layout,
    ]
    ctx.enqueue_function_checked[kernel, kernel](
        c_tensor,
        a_tensor,
        packed_tensor,
        scale_tensor,
        bias_tensor,
        offsets_tensor,
        ids_tensor,
        grid_dim=(
            ceildiv(c_tensor.dim(1), _OUTPUT_TILE),
            ceildiv(max_num_tokens_per_expert, _TOKEN_TILE),
            num_active_experts,
        ),
        block_dim=(_OUTPUT_TILE, _TOKEN_TILE, 1),
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
    var scale_byte = scale_row[out_idx * scale_stride + (k // _FP4_PER_BLOCK)]
    return base * _scale_multiplier(scale_byte)


fn _mxfp4_grouped_matmul_cpu[
    c_type: DType,
    a_type: DType,
](
    c: NDBuffer[mut=True, c_type, 2, MutAnyOrigin],
    a: NDBuffer[a_type, 2, MutAnyOrigin],
    packed_b: NDBuffer[DType.uint8, 3, MutAnyOrigin],
    scales: NDBuffer[DType.uint8, 3, MutAnyOrigin],
    bias: NDBuffer[c_type, 2, MutAnyOrigin],
    expert_offsets: NDBuffer[DType.uint32, 1, MutAnyOrigin],
    expert_ids: NDBuffer[DType.int32, 1, MutAnyOrigin],
    num_active_experts: Int,
) raises:
    var num_outputs = packed_b.dim[1]()
    var packed_stride = packed_b.dim[2]()
    var in_features = packed_stride * 2
    var scale_stride = scales.dim[2]()
    var bias_stride = bias.dim[1]()

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
        var bias_row = bias.data + expert * bias_stride

        for m in range(tokens):
            var a_row = a_slice + UInt(m * in_features)
            for n in range(num_outputs):
                var accum = Float32(0.0)
                var packed_row_n = packed_row + n * packed_stride
                var scale_row_n = scale_row + n * scale_stride
                var num_blocks = in_features // _FP4_PER_BLOCK
                for scale_block in range(num_blocks):
                    var scale_mul = _scale_multiplier(
                        scale_row_n[scale_block]
                    )
                    var k_base = scale_block * _FP4_PER_BLOCK
                    var packed_block = (
                        packed_row_n
                        + scale_block * _PACKED_BYTES_PER_BLOCK
                    )
                    for byte_idx in range(_PACKED_BYTES_PER_BLOCK):
                        var packed_byte = packed_block[byte_idx]
                        var w0 = _FP4_VALUES[
                            Int(packed_byte & UInt8(0x0F))
                        ] * scale_mul
                        var w1 = _FP4_VALUES[Int(packed_byte >> 4)] * scale_mul

                        var k0 = k_base + byte_idx * 2
                        var a0 = a_row[UInt(k0)].cast[DType.float32]()
                        var a1 = a_row[UInt(k0 + 1)].cast[DType.float32]()
                        accum += a0 * w0 + a1 * w1
                accum += bias_row[UInt(n)].cast[DType.float32]()
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
    bias: NDBuffer[c_type, 2, MutAnyOrigin],
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
            bias,
            expert_offsets,
            expert_ids,
            max_num_tokens_per_expert,
            num_active_experts,
            ctx,
        )
    else:
        # Default to the CPU path when no GPU target is available. This keeps
        # host-side testing working even when the target string is not one of
        # the expected CPU/GPU enumerations.
        _mxfp4_grouped_matmul_cpu[c_type, a_type](
            c,
            a,
            packed_b,
            scales,
            bias,
            expert_offsets,
            expert_ids,
            num_active_experts,
        )
