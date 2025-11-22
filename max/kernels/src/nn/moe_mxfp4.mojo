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

from math import ceildiv, ldexp, exp
from sys import align_of, env_get_bool

from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    block_dim,
    global_idx,
    grid_dim,
    thread_idx,
    barrier,
)
from gpu.host import DeviceContext
from gpu.host.info import H100, is_cpu, is_gpu
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from layout._ndbuffer_stub import from_ndbuffer_row_major
from layout.runtime_layout import RuntimeLayout
from layout.tensor_core_async import (
    TensorCoreAsync,
    tile_layout_k_major,
    warpgroup_fence,
    wgmma_c_layout,
)
from memory import LegacyUnsafePointer as UnsafePointer, stack_allocation
from utils.index import Index
from utils.static_tuple import StaticTuple
from utils.numerics import get_accum_type

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
alias _TOKEN_TILE = 4
alias _K_BLOCKS_PER_TILE = 4
alias _K_TILE = _K_BLOCKS_PER_TILE * _FP4_PER_BLOCK
alias _WEIGHT_TILE_SIZE = _OUTPUT_TILE * _K_TILE
alias _ACT_TILE_SIZE = _TOKEN_TILE * _K_TILE
alias _WGMMA_M = 64
alias _WGMMA_N = 64
alias _WGMMA_K = 16


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](128))
fn mxfp4_grouped_matmul_sm90_kernel[
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
    alias accum_type = get_accum_type[c_type]()
    alias block_m = _WGMMA_M  # 64
    alias block_n = _WGMMA_N  # 64
    alias block_k = _WGMMA_K  # 16

    var a_width = a.dim(1)
    var out_features = packed_b.dim(1)
    var packed_stride = packed_b.dim(2)
    var scale_stride = scales.dim(2)

    var expert_block = Int(block_idx.z)
    var offsets_ptr = expert_offsets.ptr
    var ids_ptr = expert_ids.ptr

    var start_offset = Int(offsets_ptr[expert_block])
    var end_offset = Int(offsets_ptr[expert_block + 1])
    var tokens_for_expert = end_offset - start_offset
    if tokens_for_expert <= 0:
        return

    var expert = Int(ids_ptr[expert_block])
    if expert == -1:
        return

    var n_tile = Int(block_idx.x)
    var m_tile = Int(block_idx.y)
    var n_start = n_tile * block_n
    var m_start = m_tile * block_m

    if n_start >= out_features:
        return

    var a_ptr = a.ptr + UInt((start_offset + m_start) * a_width)

    var packed_base = packed_b.ptr + UInt(
        expert * out_features * packed_stride + n_start * packed_stride
    )
    var scale_base = scales.ptr + UInt(
        expert * out_features * scale_stride + n_start * scale_stride
    )
    var bias_base = bias.ptr + UInt(expert * bias.dim(1))

    # --- shared tiles for A and B in bf16 ---

    alias a_smem_layout = tile_layout_k_major[
        DType.bfloat16, block_m, block_k, TensorMapSwizzle.SWIZZLE_NONE
    ]()
    var a_smem = LayoutTensor[
        DType.bfloat16,
        a_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    alias b_layout = tile_layout_k_major[
        DType.bfloat16, block_n, block_k, TensorMapSwizzle.SWIZZLE_NONE
    ]()
    var b_smem = LayoutTensor[
        DType.bfloat16,
        b_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var c_smem = LayoutTensor[
        accum_type,
        Layout.row_major(block_m, block_n),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # --- C layouts: registers + mapping to shared ---

    alias c_layouts = wgmma_c_layout[
        _WGMMA_M,  # mma_m
        _WGMMA_N,  # mma_n
        c_smem.layout,  # final layout in shared
    ]()
    alias c_reg_layout = c_layouts[0]
    alias tv_tile_to_idx_const = c_layouts[2]
    alias tv_to_idx_const = tv_tile_to_idx_const[0]
    alias tile_to_idx = tv_to_idx_const[1]
    alias t_to_idx_const = tv_to_idx_const[0]
    alias v_to_idx = tv_to_idx_const[1]

    var c_reg = LayoutTensor[
        accum_type,
        c_reg_layout,
        MutAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation()
    _ = c_reg.fill(Scalar[accum_type](0))

    alias wgmma_op = TensorCoreAsync[
        accum_type,
        DType.bfloat16,
        DType.bfloat16,
        Index(block_m, block_n, block_k),
        a_swizzle = TensorMapSwizzle.SWIZZLE_NONE,
        b_swizzle = TensorMapSwizzle.SWIZZLE_NONE,
        transpose_b=False,
    ]

    var k_tiles = a_width // block_k
    var block_size = Int(block_dim.x * block_dim.y)
    var warpgroup_tid = Int(thread_idx.y) * Int(block_dim.x) + Int(thread_idx.x)

    for k_tile in range(k_tiles):
        var k_base = k_tile * block_k

        # --- load A tile (bf16) into shared ---

        var total_a = block_m * block_k
        var idx = warpgroup_tid
        while idx < total_a:
            var m_local = idx // block_k
            var k_local = idx - m_local * block_k
            var m_global = m_start + m_local
            var value = Float32(0.0)
            if m_global < tokens_for_expert:
                var a_offset = UInt(m_local * a_width + k_base + k_local)
                value = a_ptr[a_offset].cast[DType.float32]()
            a_smem[m_local, k_local] = Scalar[DType.bfloat16](value)
            idx += block_size

        # --- decode packed B tile (FP4 -> bf16) into shared ---

        var total_b = block_n * block_k
        idx = warpgroup_tid
        while idx < total_b:
            var n_local = idx // block_k
            var k_local = idx - n_local * block_k
            var n_global = n_start + n_local
            var k_global = k_base + k_local
            var decoded = Float32(0.0)
            if n_global < out_features:
                var packed_row = packed_base + UInt(n_local * packed_stride)
                var packed_byte = packed_row[UInt(k_global >> 1)]
                var scale_row = scale_base + UInt(n_local * scale_stride)
                var scale_byte = scale_row[UInt(k_global >> 5)]
                var nibble = (
                    packed_byte & UInt8(0x0F) if (k_global & 1)
                    == 0 else packed_byte >> 4
                )
                var scale_mul = _scale_multiplier(scale_byte)
                decoded = _FP4_VALUES[Int(nibble)] * scale_mul
            b_smem[n_local, k_local] = Scalar[DType.bfloat16](decoded)
            idx += block_size

        barrier()

        # --- WGMMA accumulate into c_reg ---

        warpgroup_fence(c_reg)
        wgmma_op.arrive()
        wgmma_op.wgmma(
            a_smem,
            b_smem,
            c_reg,
            wg_idx=0,
        )
        wgmma_op.commit_group()
        warpgroup_fence(c_reg)
        wgmma_op.wait_group()

        barrier()

    # --- move C fragments from registers -> shared row-major ---

    var t_to_idx = RuntimeLayout[t_to_idx_const]()
    var linear_tid = Int(thread_idx.y) * Int(block_dim.x) + Int(thread_idx.x)
    var lane_offset = t_to_idx(linear_tid)
    var c_smem_ptr = c_smem.ptr.offset(lane_offset)

    var c_reg_vec2 = c_reg.vectorize[1, 2]()
    alias VecType = c_reg_vec2.element_type

    @parameter
    for mma_id in range(tile_to_idx.size()):
        alias mma_idx = tile_to_idx(mma_id)

        @parameter
        for frag_idx_v2 in range(c_reg_vec2.layout[1].size()):
            alias frag_idx = frag_idx_v2 * 2
            alias v_idx = v_to_idx(frag_idx)
            alias dst_idx = v_idx + mma_idx
            c_smem_ptr.offset(dst_idx).store[alignment = align_of[VecType]()](
                (c_reg_vec2[mma_id, frag_idx_v2])
            )

    barrier()

    # --- epilogue: add bias and write out ---

    var total_c = block_m * block_n
    var idx2 = warpgroup_tid
    while idx2 < total_c:
        var m_local = idx2 // block_n
        var n_local = idx2 - m_local * block_n
        var m_global = m_start + m_local
        var n_global = n_start + n_local
        if m_global < tokens_for_expert and n_global < out_features:
            var out_row = start_offset + m_global
            var bias_val = bias_base[UInt(n_global)].cast[accum_type]()
            var acc = c_smem[m_local, n_local] + bias_val
            var out_val = acc[0].cast[c_type]()
            var out_ptr = c.ptr + UInt(out_row * out_features + n_global)
            out_ptr[0] = out_val
        idx2 += block_size


@always_inline
fn _scale_multiplier(scale_byte: UInt8) -> Float32:
    var exponent = Int(scale_byte) - 127
    return ldexp(Float32(1.0), exponent)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256)
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
    debug_assert(
        in_features % _FP4_PER_BLOCK == 0,
        "in_features must be a multiple of _FP4_PER_BLOCK (32) for MXFP4.",
    )

    var a_data = a.ptr + UInt(start_offset * in_features)
    var expert = Int(ids_ptr[expert_block])

    if expert == -1:
        return

    var packed_by_expert = packed_b.ptr + UInt(
        expert * num_outputs * packed_stride
    )
    var scales_by_expert = scales.ptr + UInt(
        expert * num_outputs * scale_stride
    )
    var bias_by_expert = bias.ptr + UInt(expert * num_outputs)

    var n = Int(global_idx.x)
    var m = Int(global_idx.y)
    var local_n = Int(thread_idx.x)
    var active = n < num_outputs and m < tokens_for_expert

    var accum = Float32(0.0)

    var shared_weights = stack_allocation[
        _WEIGHT_TILE_SIZE, Float32, address_space = AddressSpace.SHARED
    ]()
    var shared_act = stack_allocation[
        _ACT_TILE_SIZE, Float32, address_space = AddressSpace.SHARED
    ]()

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
                var packed_block = packed_row + UInt(
                    scale_block * _PACKED_BYTES_PER_BLOCK
                )
                var block_offset = shared_base + block_inner * _FP4_PER_BLOCK

                for byte_base in range(0, _PACKED_BYTES_PER_BLOCK, 4):
                    var b0 = packed_block[byte_base]
                    var b1 = packed_block[byte_base + 1]
                    var b2 = packed_block[byte_base + 2]
                    var b3 = packed_block[byte_base + 3]

                    var base_idx = block_offset + byte_base * 2
                    shared_weights[base_idx] = (
                        _FP4_VALUES[Int(b0 & UInt8(0x0F))] * scale_mul
                    )
                    shared_weights[base_idx + 1] = (
                        _FP4_VALUES[Int(b0 >> 4)] * scale_mul
                    )
                    shared_weights[base_idx + 2] = (
                        _FP4_VALUES[Int(b1 & UInt8(0x0F))] * scale_mul
                    )
                    shared_weights[base_idx + 3] = (
                        _FP4_VALUES[Int(b1 >> 4)] * scale_mul
                    )
                    shared_weights[base_idx + 4] = (
                        _FP4_VALUES[Int(b2 & UInt8(0x0F))] * scale_mul
                    )
                    shared_weights[base_idx + 5] = (
                        _FP4_VALUES[Int(b2 >> 4)] * scale_mul
                    )
                    shared_weights[base_idx + 6] = (
                        _FP4_VALUES[Int(b3 & UInt8(0x0F))] * scale_mul
                    )
                    shared_weights[base_idx + 7] = (
                        _FP4_VALUES[Int(b3 >> 4)] * scale_mul
                    )

        if m < tokens_for_expert:
            var a_row = a_data + UInt(m * in_features)
            var k_index = Int(thread_idx.x)
            while k_index < k_tile:
                shared_act[Int(thread_idx.y) * _K_TILE + k_index] = a_row[
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


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
fn mxfp4_grouped_matmul_swiglu_kernel[
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
    alpha: Float32,
    limit: Float32,
):
    var expert_block = Int(block_idx.z)
    var offsets_ptr = expert_offsets.ptr
    var ids_ptr = expert_ids.ptr
    var start_offset = Int(offsets_ptr[expert_block])
    var end_offset = Int(offsets_ptr[expert_block + 1])
    var tokens_for_expert = end_offset - start_offset
    if tokens_for_expert <= 0:
        return

    var num_pairs = c.dim(1)
    var packed_stride = packed_b.dim(2)
    var in_features = packed_stride * 2
    var scale_stride = scales.dim(2)
    var bias_stride = bias.dim(1)
    debug_assert(
        in_features % _FP4_PER_BLOCK == 0,
        "in_features must be a multiple of _FP4_PER_BLOCK (32) for MXFP4.",
    )

    var a_data = a.ptr + UInt(start_offset * in_features)
    var expert = Int(ids_ptr[expert_block])

    if expert == -1:
        return

    var gate_row_base = UInt(expert * num_pairs * 2 * packed_stride)
    var scale_row_base = UInt(expert * num_pairs * 2 * scale_stride)
    var bias_row_base = UInt(expert * bias_stride)

    var n_pair = Int(global_idx.x)
    var m = Int(global_idx.y)
    var local_n = Int(thread_idx.x)
    var active = n_pair < num_pairs and m < tokens_for_expert

    var accum_gate = Float32(0.0)
    var accum_up = Float32(0.0)

    var shared_weights = stack_allocation[
        _WEIGHT_TILE_SIZE, Float32, address_space = AddressSpace.SHARED
    ]()
    var shared_act = stack_allocation[
        _ACT_TILE_SIZE, Float32, address_space = AddressSpace.SHARED
    ]()

    var num_blocks = in_features // _FP4_PER_BLOCK
    var k_block_start = 0
    while k_block_start < num_blocks:
        var blocks_this_tile = num_blocks - k_block_start
        if blocks_this_tile > _K_BLOCKS_PER_TILE:
            blocks_this_tile = _K_BLOCKS_PER_TILE
        var k_tile = blocks_this_tile * _FP4_PER_BLOCK

        if thread_idx.y == 0 and n_pair < num_pairs:
            var gate_idx = n_pair * 2
            var up_idx = gate_idx + 1

            var gate_packed_row = (
                packed_b.ptr + gate_row_base + UInt(gate_idx * packed_stride)
            )
            var up_packed_row = (
                packed_b.ptr + gate_row_base + UInt(up_idx * packed_stride)
            )
            var gate_scale_row = (
                scales.ptr + scale_row_base + UInt(gate_idx * scale_stride)
            )
            var up_scale_row = (
                scales.ptr + scale_row_base + UInt(up_idx * scale_stride)
            )

            var shared_base = local_n * _K_TILE

            for block_inner in range(blocks_this_tile):
                var scale_block = k_block_start + block_inner
                var gate_scale_mul = _scale_multiplier(
                    gate_scale_row[scale_block]
                )
                var gate_packed_block = gate_packed_row + UInt(
                    scale_block * _PACKED_BYTES_PER_BLOCK
                )
                var block_offset = shared_base + block_inner * _FP4_PER_BLOCK

                for byte_base in range(0, _PACKED_BYTES_PER_BLOCK, 4):
                    var gb0 = gate_packed_block[byte_base]
                    var gb1 = gate_packed_block[byte_base + 1]
                    var gb2 = gate_packed_block[byte_base + 2]
                    var gb3 = gate_packed_block[byte_base + 3]

                    var base_idx = block_offset + byte_base * 2
                    shared_weights[base_idx] = (
                        _FP4_VALUES[Int(gb0 & UInt8(0x0F))] * gate_scale_mul
                    )
                    shared_weights[base_idx + 1] = (
                        _FP4_VALUES[Int(gb0 >> 4)] * gate_scale_mul
                    )
                    shared_weights[base_idx + 2] = (
                        _FP4_VALUES[Int(gb1 & UInt8(0x0F))] * gate_scale_mul
                    )
                    shared_weights[base_idx + 3] = (
                        _FP4_VALUES[Int(gb1 >> 4)] * gate_scale_mul
                    )
                    shared_weights[base_idx + 4] = (
                        _FP4_VALUES[Int(gb2 & UInt8(0x0F))] * gate_scale_mul
                    )
                    shared_weights[base_idx + 5] = (
                        _FP4_VALUES[Int(gb2 >> 4)] * gate_scale_mul
                    )
                    shared_weights[base_idx + 6] = (
                        _FP4_VALUES[Int(gb3 & UInt8(0x0F))] * gate_scale_mul
                    )
                    shared_weights[base_idx + 7] = (
                        _FP4_VALUES[Int(gb3 >> 4)] * gate_scale_mul
                    )

        if m < tokens_for_expert:
            var a_row = a_data + UInt(m * in_features)
            var k_index = Int(thread_idx.x)
            while k_index < k_tile:
                shared_act[Int(thread_idx.y) * _K_TILE + k_index] = a_row[
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

                var wg0 = shared_weights[shared_base + k_offset]
                var wg1 = shared_weights[shared_base + k_offset + 1]
                var wg2 = shared_weights[shared_base + k_offset + 2]
                var wg3 = shared_weights[shared_base + k_offset + 3]

                accum_gate += a0 * wg0 + a1 * wg1 + a2 * wg2 + a3 * wg3

        barrier()

        if thread_idx.y == 0 and n_pair < num_pairs:
            var gate_idx = n_pair * 2
            var up_idx = gate_idx + 1

            var up_packed_row = (
                packed_b.ptr + gate_row_base + UInt(up_idx * packed_stride)
            )
            var up_scale_row = (
                scales.ptr + scale_row_base + UInt(up_idx * scale_stride)
            )

            var shared_base_up = local_n * _K_TILE

            for block_inner in range(blocks_this_tile):
                var scale_block = k_block_start + block_inner
                var up_scale_mul = _scale_multiplier(up_scale_row[scale_block])
                var up_packed_block = up_packed_row + UInt(
                    scale_block * _PACKED_BYTES_PER_BLOCK
                )
                var block_offset = shared_base_up + block_inner * _FP4_PER_BLOCK

                for byte_base in range(0, _PACKED_BYTES_PER_BLOCK, 4):
                    var gu0 = up_packed_block[byte_base]
                    var gu1 = up_packed_block[byte_base + 1]
                    var gu2 = up_packed_block[byte_base + 2]
                    var gu3 = up_packed_block[byte_base + 3]

                    var base_idx = block_offset + byte_base * 2
                    shared_weights[base_idx] = (
                        _FP4_VALUES[Int(gu0 & UInt8(0x0F))] * up_scale_mul
                    )
                    shared_weights[base_idx + 1] = (
                        _FP4_VALUES[Int(gu0 >> 4)] * up_scale_mul
                    )
                    shared_weights[base_idx + 2] = (
                        _FP4_VALUES[Int(gu1 & UInt8(0x0F))] * up_scale_mul
                    )
                    shared_weights[base_idx + 3] = (
                        _FP4_VALUES[Int(gu1 >> 4)] * up_scale_mul
                    )
                    shared_weights[base_idx + 4] = (
                        _FP4_VALUES[Int(gu2 & UInt8(0x0F))] * up_scale_mul
                    )
                    shared_weights[base_idx + 5] = (
                        _FP4_VALUES[Int(gu2 >> 4)] * up_scale_mul
                    )
                    shared_weights[base_idx + 6] = (
                        _FP4_VALUES[Int(gu3 & UInt8(0x0F))] * up_scale_mul
                    )
                    shared_weights[base_idx + 7] = (
                        _FP4_VALUES[Int(gu3 >> 4)] * up_scale_mul
                    )

        barrier()

        if active:
            var shared_base_up = local_n * _K_TILE
            var act_base_up = Int(thread_idx.y) * _K_TILE
            for k_offset in range(0, k_tile, 4):
                var a0 = shared_act[act_base_up + k_offset]
                var a1 = shared_act[act_base_up + k_offset + 1]
                var a2 = shared_act[act_base_up + k_offset + 2]
                var a3 = shared_act[act_base_up + k_offset + 3]

                var wu0 = shared_weights[shared_base_up + k_offset]
                var wu1 = shared_weights[shared_base_up + k_offset + 1]
                var wu2 = shared_weights[shared_base_up + k_offset + 2]
                var wu3 = shared_weights[shared_base_up + k_offset + 3]

                accum_up += a0 * wu0 + a1 * wu1 + a2 * wu2 + a3 * wu3

        k_block_start += _K_BLOCKS_PER_TILE
        barrier()

    var c_data = c.ptr + UInt(start_offset * num_pairs)
    if active:
        var gate_bias = bias.ptr[bias_row_base + UInt(n_pair * 2)].cast[
            DType.float32
        ]()
        var up_bias = bias.ptr[bias_row_base + UInt(n_pair * 2 + 1)].cast[
            DType.float32
        ]()

        var gate_val = accum_gate + gate_bias
        var up_val = accum_up + up_bias

        if gate_val > limit:
            gate_val = limit
        if up_val > limit:
            up_val = limit
        if up_val < -limit:
            up_val = -limit

        var sig = Float32(1.0) / (Float32(1.0) + exp(-gate_val * alpha))
        var glu = gate_val * sig
        var out = glu * (up_val + Float32(1.0))
        c_data[UInt(m * num_pairs + n_pair)] = Scalar[c_type](out)


fn _mxfp4_grouped_matmul_sm90[
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

    alias kernel = mxfp4_grouped_matmul_sm90_kernel[
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
            ceildiv(c_tensor.dim(1), _WGMMA_N),
            ceildiv(max_num_tokens_per_expert, _WGMMA_M),
            num_active_experts,
        ),
        block_dim=(32, 4, 1),
    )


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


fn _mxfp4_grouped_matmul_swiglu_gpu[
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
    alpha: Float32,
    limit: Float32,
    ctx: DeviceContext,
) raises:
    var c_tensor = from_ndbuffer_row_major(c)
    var a_tensor = from_ndbuffer_row_major(a)
    var packed_tensor = from_ndbuffer_row_major(packed_b)
    var scale_tensor = from_ndbuffer_row_major(scales)
    var bias_tensor = from_ndbuffer_row_major(bias)
    var offsets_tensor = from_ndbuffer_row_major(expert_offsets)
    var ids_tensor = from_ndbuffer_row_major(expert_ids)

    alias kernel = mxfp4_grouped_matmul_swiglu_kernel[
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
        alpha,
        limit,
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
    debug_assert(
        in_features % _FP4_PER_BLOCK == 0,
        "in_features must be a multiple of _FP4_PER_BLOCK (32) for MXFP4.",
    )

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
                    var scale_mul = _scale_multiplier(scale_row_n[scale_block])
                    var k_base = scale_block * _FP4_PER_BLOCK
                    var packed_block = (
                        packed_row_n + scale_block * _PACKED_BYTES_PER_BLOCK
                    )
                    for byte_idx in range(_PACKED_BYTES_PER_BLOCK):
                        var packed_byte = packed_block[byte_idx]
                        var w0 = (
                            _FP4_VALUES[Int(packed_byte & UInt8(0x0F))]
                            * scale_mul
                        )
                        var w1 = _FP4_VALUES[Int(packed_byte >> 4)] * scale_mul

                        var k0 = k_base + byte_idx * 2
                        var a0 = a_row[UInt(k0)].cast[DType.float32]()
                        var a1 = a_row[UInt(k0 + 1)].cast[DType.float32]()
                        accum += a0 * w0 + a1 * w1
                accum += bias_row[UInt(n)].cast[DType.float32]()
                out_slice[UInt(m * num_outputs + n)] = Scalar[c_type](accum)


fn _mxfp4_grouped_matmul_swiglu_cpu[
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
    alpha: Float32,
    limit: Float32,
) raises:
    var num_pairs = c.dim[1]()
    var packed_stride = packed_b.dim[2]()
    var in_features = packed_stride * 2
    var scale_stride = scales.dim[2]()
    var bias_stride = bias.dim[1]()
    debug_assert(
        in_features % _FP4_PER_BLOCK == 0,
        "in_features must be a multiple of _FP4_PER_BLOCK (32) for MXFP4.",
    )

    for expert_idx in range(num_active_experts):
        var expert = Int(expert_ids[expert_idx])
        var token_start = Int(expert_offsets[expert_idx])
        var token_end = Int(expert_offsets[expert_idx + 1])
        var tokens = token_end - token_start

        if expert == -1 or tokens <= 0:
            continue

        var a_slice = a.data + UInt(token_start * in_features)
        var out_slice = c.data + UInt(token_start * num_pairs)
        var packed_row = packed_b.data + expert * num_pairs * 2 * packed_stride
        var scale_row = scales.data + expert * num_pairs * 2 * scale_stride
        var bias_row = bias.data + expert * bias_stride

        for m in range(tokens):
            var a_row = a_slice + UInt(m * in_features)
            for pair in range(num_pairs):
                var gate_accum = Float32(0.0)
                var up_accum = Float32(0.0)
                var gate_idx = pair * 2
                var up_idx = gate_idx + 1
                var gate_packed_row = packed_row + gate_idx * packed_stride
                var up_packed_row = packed_row + up_idx * packed_stride
                var gate_scale_row = scale_row + gate_idx * scale_stride
                var up_scale_row = scale_row + up_idx * scale_stride
                var num_blocks = in_features // _FP4_PER_BLOCK
                for scale_block in range(num_blocks):
                    var gate_scale_mul = _scale_multiplier(
                        gate_scale_row[scale_block]
                    )
                    var up_scale_mul = _scale_multiplier(
                        up_scale_row[scale_block]
                    )
                    var k_base = scale_block * _FP4_PER_BLOCK
                    var gate_packed_block = (
                        gate_packed_row + scale_block * _PACKED_BYTES_PER_BLOCK
                    )
                    var up_packed_block = (
                        up_packed_row + scale_block * _PACKED_BYTES_PER_BLOCK
                    )
                    for byte_idx in range(_PACKED_BYTES_PER_BLOCK):
                        var packed_gate = gate_packed_block[byte_idx]
                        var packed_up = up_packed_block[byte_idx]
                        var w0_gate = (
                            _FP4_VALUES[Int(packed_gate & UInt8(0x0F))]
                            * gate_scale_mul
                        )
                        var w1_gate = (
                            _FP4_VALUES[Int(packed_gate >> 4)] * gate_scale_mul
                        )
                        var w0_up = (
                            _FP4_VALUES[Int(packed_up & UInt8(0x0F))]
                            * up_scale_mul
                        )
                        var w1_up = (
                            _FP4_VALUES[Int(packed_up >> 4)] * up_scale_mul
                        )

                        var k0 = k_base + byte_idx * 2
                        var a0 = a_row[UInt(k0)].cast[DType.float32]()
                        var a1 = a_row[UInt(k0 + 1)].cast[DType.float32]()
                        gate_accum += a0 * w0_gate + a1 * w1_gate
                        up_accum += a0 * w0_up + a1 * w1_up
                gate_accum += bias_row[UInt(gate_idx)].cast[DType.float32]()
                up_accum += bias_row[UInt(up_idx)].cast[DType.float32]()

                if gate_accum > limit:
                    gate_accum = limit
                if up_accum > limit:
                    up_accum = limit
                if up_accum < -limit:
                    up_accum = -limit

                var sig = Float32(1.0) / (
                    Float32(1.0) + exp(-gate_accum * alpha)
                )
                var glu = gate_accum * sig
                var out = glu * (up_accum + Float32(1.0))
                out_slice[UInt(m * num_pairs + pair)] = Scalar[c_type](out)


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
        var force_generic = env_get_bool[
            "MAX_MXFP4_FORCE_GENERIC_GPU",
            False,
        ]()
        var is_h100 = False
        if not force_generic:
            is_h100 = materialize[ctx.default_device_info is H100]()

        if is_h100 and not force_generic:
            _mxfp4_grouped_matmul_sm90[c_type, a_type](
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


fn mxfp4_grouped_matmul_swiglu[
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
    alpha: Float32,
    limit: Float32,
    ctx: DeviceContext,
) raises:
    if is_gpu[target]():
        _mxfp4_grouped_matmul_swiglu_gpu[c_type, a_type](
            c,
            a,
            packed_b,
            scales,
            bias,
            expert_offsets,
            expert_ids,
            max_num_tokens_per_expert,
            num_active_experts,
            alpha,
            limit,
            ctx,
        )
    else:
        _mxfp4_grouped_matmul_swiglu_cpu[c_type, a_type](
            c,
            a,
            packed_b,
            scales,
            bias,
            expert_offsets,
            expert_ids,
            num_active_experts,
            alpha,
            limit,
        )
