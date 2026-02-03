# grouped_matmul_sm90_common.mojo
#
# Shared constants + helpers for SM90 MXFP4 grouped matmul kernels.

from math import ceildiv

import compiler
from sys import simd_width_of, size_of
from gpu import (
    WARP_SIZE,
    barrier,
    block_dim,
    block_idx,
    lane_id,
    thread_idx,
    warp_id,
)
from gpu.globals import WARPGROUP_SIZE
from gpu.intrinsics import Scope, warpgroup_reg_alloc, warpgroup_reg_dealloc
from gpu.intrinsics import threadfence
from gpu.host import DeviceBuffer, FuncAttribute
from gpu.host.info import is_cpu
from gpu.memory import (
    AddressSpace,
    external_memory,
)
from layout import IntTuple
from layout.int_tuple import UNKNOWN_VALUE
from layout.layout_tensor import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from layout.swizzle import Swizzle, make_swizzle
from layout.tensor_core_async import (
    TensorCoreAsync,
    tile_layout_k_major,
    warpgroup_fence,
)
from layout.tensor_core import TensorCore
from layout.tma_async import (
    PipelineState,
    SharedMemBarrier,
    TMATensorTile,
    create_tma_tile,
)
from memory import bitcast, stack_allocation
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from utils.index import Index, IndexList

from gpu.host.nvidia.tma import TensorMapSwizzle

from .decode import (
    decode_mxfp4_packbits_u32_to_8xbf16_scaled,
    decode_mxfp4_packbits_u32_to_8xbf16_scaled_e8m0,
    e8m0_to_bf16_bits,
)
from .layout_hopper import (
    HOPPER_SCALE_NUM_WARPS,
    HOPPER_SCALE_ALIGN_M,
    HOPPER_SCALE_ALIGN_K,
    hopper_scale_swizzle_index,
    hopper_scale_swizzle_index_fast,
    hopper_value_swizzle_index,
)


comptime BF16 = DType.bfloat16
comptime F32 = DType.float32
comptime U8 = DType.uint8
comptime U32 = DType.uint32
comptime U64 = DType.uint64
comptime I32 = DType.int32

comptime BYTES_PER_BLOCK = 16
comptime VALUES_PER_BLOCK = 32
comptime TRANSPOSE_WRITER_MAP = 0


@always_inline
fn _u32_from_u8x4(b0: UInt8, b1: UInt8, b2: UInt8, b3: UInt8) -> UInt32:
    return (
        UInt32(b0)
        | (UInt32(b1) << UInt32(8))
        | (UInt32(b2) << UInt32(16))
        | (UInt32(b3) << UInt32(24))
    )


@always_inline
fn u32_from_u8x4(b0: UInt8, b1: UInt8, b2: UInt8, b3: UInt8) -> UInt32:
    return _u32_from_u8x4(b0, b1, b2, b3)


@always_inline
fn _decode_chunk_from_packed_u8x16(
    packed0: SIMD[U8, 8],
    packed1: SIMD[U8, 8],
    chunk: Int,
    scale: Scalar[BF16],
) -> SIMD[BF16, 8]:
    var off = chunk * 4
    var src = packed0
    if chunk >= 2:
        src = packed1
        off = (chunk - 2) * 4
    var p = u32_from_u8x4(
        UInt8(src[off + 0]),
        UInt8(src[off + 1]),
        UInt8(src[off + 2]),
        UInt8(src[off + 3]),
    )
    return decode_mxfp4_packbits_u32_to_8xbf16_scaled(p, scale)


@always_inline
fn _decode_mxfp4_unshuffle_value[
    USE_VALUE_SWIZZLE: Bool = False,
](
    w_blocks_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    w_blocks_stride0: Int,
    w_blocks_stride1: Int,
    w_blocks_stride2: Int,
    expert_id: Int,
    base_m2: Int,
    base_k2: Int,
    row_rel: Int,
    row_abs: Int,
    col_rel: Int,
    k0: Int,
    K: Int,
    N: Int,
    scales: SIMD[BF16, 4],
) -> Scalar[BF16]:
    if row_abs >= N:
        return Scalar[BF16](0)
    var col_abs = k0 + col_rel
    if col_abs >= K:
        return Scalar[BF16](0)

    var kb_rel = col_rel >> 5
    var scale = scales[kb_rel]

    # Inverse of Triton _unshuffle_triton (mma_version=3) for one element.
    var d0 = row_rel & 1
    var b0 = (row_rel >> 1) & 3
    var g0 = (row_rel >> 3) & 1
    var a0 = row_rel >> 4
    var m_in = a0 * 4 + b0

    var h0 = col_rel & 1
    var e0 = (col_rel >> 1) & 3
    var f0 = (col_rel >> 3) & 7
    var c0 = col_rel >> 6
    var k_in = (
        (((((c0 * 2 + d0) * 4 + e0) * 8 + f0) * 2 + g0) * 2 + h0)
    )

    var off = k_in & 7
    var p: UInt32 = 0

    @parameter
    if USE_VALUE_SWIZZLE:
        # Hopper value swizzle: map logical (row_abs, kbyte) -> swizzled coords.
        var kbyte = (k0 + k_in) >> 1
        var kbyte_group = kbyte - (kbyte & 3)

        var idx0 = hopper_value_swizzle_index(row_abs, kbyte_group)
        var idx1 = hopper_value_swizzle_index(row_abs, kbyte_group + 1)
        var idx2 = hopper_value_swizzle_index(row_abs, kbyte_group + 2)
        var idx3 = hopper_value_swizzle_index(row_abs, kbyte_group + 3)

        var base = expert_id * w_blocks_stride0
        var p0 = w_blocks_ptr[
            base + idx0[0] * w_blocks_stride1 + idx0[1] * w_blocks_stride2
        ]
        var p1 = w_blocks_ptr[
            base + idx1[0] * w_blocks_stride1 + idx1[1] * w_blocks_stride2
        ]
        var p2 = w_blocks_ptr[
            base + idx2[0] * w_blocks_stride1 + idx2[1] * w_blocks_stride2
        ]
        var p3 = w_blocks_ptr[
            base + idx3[0] * w_blocks_stride1 + idx3[1] * w_blocks_stride2
        ]
        p = u32_from_u8x4(UInt8(p0), UInt8(p1), UInt8(p2), UInt8(p3))
    else:
        var m2 = base_m2 + m_in
        var kbyte = k_in >> 1
        var kbyte_group = kbyte - (kbyte & 3)

        var base = (
            expert_id * w_blocks_stride0
            + m2 * w_blocks_stride1
            + (base_k2 + kbyte_group) * w_blocks_stride2
        )
        p = u32_from_u8x4(
            w_blocks_ptr[base],
            w_blocks_ptr[base + w_blocks_stride2],
            w_blocks_ptr[base + 2 * w_blocks_stride2],
            w_blocks_ptr[base + 3 * w_blocks_stride2],
        )
    var vals = decode_mxfp4_packbits_u32_to_8xbf16_scaled(p, scale)
    return vals[off]


@always_inline
fn decode_mxfp4_unshuffle_value[
    USE_VALUE_SWIZZLE: Bool = False,
](
    w_blocks_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    w_blocks_stride0: Int,
    w_blocks_stride1: Int,
    w_blocks_stride2: Int,
    expert_id: Int,
    base_m2: Int,
    base_k2: Int,
    row_rel: Int,
    row_abs: Int,
    col_rel: Int,
    k0: Int,
    K: Int,
    N: Int,
    scales: SIMD[BF16, 4],
) -> Scalar[BF16]:
    return _decode_mxfp4_unshuffle_value[
        USE_VALUE_SWIZZLE=USE_VALUE_SWIZZLE
    ](
        w_blocks_ptr,
        w_blocks_stride0,
        w_blocks_stride1,
        w_blocks_stride2,
        expert_id,
        base_m2,
        base_k2,
        row_rel,
        row_abs,
        col_rel,
        k0,
        K,
        N,
        scales,
    )


@always_inline
fn _apply_xor_swizzle(swizzle: Swizzle, idx: Int) -> Int:
    # Matches layout_tensor swizzle application (canonical SWIZZLE_128B mapping).
    var base = idx % swizzle.size()
    return swizzle(base) + (idx - base)


@always_inline
fn apply_xor_swizzle(swizzle: Swizzle, idx: Int) -> Int:
    return _apply_xor_swizzle(swizzle, idx)


@always_inline
fn _load_swizzled_block_u8x16(
    w_blocks_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    w_blocks_stride0: Int,
    w_blocks_stride1: Int,
    w_blocks_stride2: Int,
    expert_id: Int,
    col: Int,
    kb: Int,
) -> Tuple[SIMD[U8, 8], SIMD[U8, 8]]:
    var packed0 = SIMD[U8, 8](0)
    var packed1 = SIMD[U8, 8](0)
    var base = expert_id * w_blocks_stride0
    var kbyte_base = kb * BYTES_PER_BLOCK

    @parameter
    for i in range(BYTES_PER_BLOCK):
        var kbyte = kbyte_base + i
        var idx = hopper_value_swizzle_index(col, kbyte)
        var byte = w_blocks_ptr[
            base + idx[0] * w_blocks_stride1 + idx[1] * w_blocks_stride2
        ]
        if i < 8:
            packed0[i] = byte
        else:
            packed1[i - 8] = byte

    return (packed0, packed1)


@always_inline
fn load_swizzled_block_u8x16(
    w_blocks_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    w_blocks_stride0: Int,
    w_blocks_stride1: Int,
    w_blocks_stride2: Int,
    expert_id: Int,
    col: Int,
    kb: Int,
) -> Tuple[SIMD[U8, 8], SIMD[U8, 8]]:
    return _load_swizzled_block_u8x16(
        w_blocks_ptr,
        w_blocks_stride0,
        w_blocks_stride1,
        w_blocks_stride2,
        expert_id,
        col,
        kb,
    )


@always_inline
fn _unshuffle_k_in(row_rel: Int, col_rel: Int) -> Int:
    var d0 = row_rel & 1
    var g0 = (row_rel >> 3) & 1
    var h0 = col_rel & 1
    var e0 = (col_rel >> 1) & 3
    var f0 = (col_rel >> 3) & 7
    var c0 = col_rel >> 6
    return (
        (((((c0 * 2 + d0) * 4 + e0) * 8 + f0) * 2 + g0) * 2 + h0)
    )


@always_inline
fn unshuffle_k_in(row_rel: Int, col_rel: Int) -> Int:
    return _unshuffle_k_in(row_rel, col_rel)


@always_inline
fn _compute_kbyte_row0_from_col(
    row_rel: Int, col_rel: Int, k0_half: Int
) -> Int:
    # General inverse of Triton unshuffle mapping: kbyte = (k0 + k_in) >> 1.
    # This handles g0/h0 and avoids row-parity aliasing for partial K tiles.
    var k_in = unshuffle_k_in(row_rel, col_rel)
    return k0_half + (k_in >> 1)


@always_inline
fn compute_kbyte_row0_from_col(
    row_rel: Int, col_rel: Int, k0_half: Int
) -> Int:
    return _compute_kbyte_row0_from_col(row_rel, col_rel, k0_half)


@always_inline
fn _load_swizzled_pack_u32(
    w_blocks_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    w_blocks_stride0: Int,
    w_blocks_stride1: Int,
    w_blocks_stride2: Int,
    expert_id: Int,
    row_abs: Int,
    kbyte: Int,
) -> Tuple[UInt32, Int]:
    var idx = hopper_value_swizzle_index(row_abs, kbyte)
    var k2 = idx[1]
    var base_k2 = k2 & ~3
    var base16 = base_k2 & ~15
    var base = (
        expert_id * w_blocks_stride0
        + idx[0] * w_blocks_stride1
        + base16 * w_blocks_stride2
    )
    var w_u32 = w_blocks_ptr.address_space_cast[
        AddressSpace.GLOBAL
    ]().bitcast[Scalar[U32]]()
    var words = w_u32.load[width=4, alignment=16](base >> 2)
    var word_idx = (base_k2 - base16) >> 2
    var p32 = words[0]
    if word_idx == 1:
        p32 = words[1]
    elif word_idx == 2:
        p32 = words[2]
    elif word_idx == 3:
        p32 = words[3]
    return (p32, k2 & 3)


@always_inline
fn load_swizzled_pack_u32(
    w_blocks_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    w_blocks_stride0: Int,
    w_blocks_stride1: Int,
    w_blocks_stride2: Int,
    expert_id: Int,
    row_abs: Int,
    kbyte: Int,
) -> Tuple[UInt32, Int]:
    return _load_swizzled_pack_u32(
        w_blocks_ptr,
        w_blocks_stride0,
        w_blocks_stride1,
        w_blocks_stride2,
        expert_id,
        row_abs,
        kbyte,
    )
