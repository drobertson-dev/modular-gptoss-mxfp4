# moe_mxfp4_ops_common.mojo
#
# Shared constants + helpers for MXFP4 MoE ops (W1/W2).
#
# Contract (matches `examples/custom-models/triton_example/moe.py` at a high level):
# - Routing is provided as:
#     token_expert_order : [P] u32  (pair_idx sorted by expert)
#     expert_start_indices: [E+1] u32 (segment starts for each expert id 0..E-1)
# - W1 kernel computes: h_sorted[pair, :] = SwiGLU(x[token] @ W1_expert + b1_expert)
# - W2 kernel computes: y[token, :] += gate_weight[pair_idx] * (h_sorted[pair] @ W2_expert + b2_expert)
#
# Notes:
# - This implementation currently uses TensorCore MMA for the matmul inner loops.
# - The W2 custom op zero-initializes the output buffer before launching, since
#   the kernel scatter-adds into it.

from math import ceildiv
from os import Atomic

import compiler
from gpu import barrier, block_dim, block_idx, grid_dim, thread_idx, warp_id
from gpu.host import DeviceBuffer, FuncAttribute
from gpu.host.info import is_cpu
from gpu.memory import (
    AddressSpace,
    async_copy,
    async_copy_commit_group,
    async_copy_wait_all,
    external_memory,
)
from layout.layout_tensor import Layout, LayoutTensor
from layout.swizzle import Swizzle, make_swizzle
from layout.tensor_core import TensorCore
from layout.tensor_core_async import (
    TensorCoreAsync,
    tile_layout_k_major,
    warpgroup_fence,
)
from memory import bitcast, stack_allocation
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from utils.index import Index, IndexList
from gpu.host.nvidia.tma import TensorMapSwizzle

from .decode import (
    decode_mxfp4_byte_to_2xbf16_e8m0,
    decode_mxfp4_byte_to_2xbf16_scaled,
    decode_mxfp4_packbits_u32_to_8xbf16_scaled,
    e8m0_to_bf16_bits,
    swiglu_pair,
)
from .layout_hopper import hopper_value_swizzle_index


comptime BF16 = DType.bfloat16
comptime F16 = DType.float16
comptime F32 = DType.float32
comptime U8 = DType.uint8
comptime U32 = DType.uint32
comptime U64 = DType.uint64
comptime I32 = DType.int32

comptime BYTES_PER_BLOCK = 16
comptime VALUES_PER_BLOCK = 32
comptime TOPK = 4
comptime WRITER_MAP = 2  # 0=baseline, 1=swap pairs, 2=lane-rotated (reduces bank conflicts)


@always_inline
fn _u32_from_u8x4(b0: UInt8, b1: UInt8, b2: UInt8, b3: UInt8) -> UInt32:
    return (
        UInt32(b0)
        | (UInt32(b1) << UInt32(8))
        | (UInt32(b2) << UInt32(16))
        | (UInt32(b3) << UInt32(24))
    )


@always_inline
fn _u64_from_u8x8(v: SIMD[U8, 8]) -> UInt64:
    var lo = _u32_from_u8x4(v[0], v[1], v[2], v[3])
    var hi = _u32_from_u8x4(v[4], v[5], v[6], v[7])
    return UInt64(lo) | (UInt64(hi) << UInt64(32))


@always_inline
fn _apply_swizzle(swizzle: Swizzle, idx: Int) -> Int:
    # Match the canonical swizzle application used by layout_tensor helpers.
    var base = idx % swizzle.size()
    return swizzle(base) + (idx - base)


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
