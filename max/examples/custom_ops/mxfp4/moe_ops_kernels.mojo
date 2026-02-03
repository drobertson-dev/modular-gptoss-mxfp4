# moe_mxfp4_ops_kernels.mojo
#
# Kernel implementations for MXFP4 MoE ops (W1/W2).
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


from .moe_ops_common import (
    BF16,
    F16,
    F32,
    U8,
    U32,
    U64,
    I32,
    BYTES_PER_BLOCK,
    VALUES_PER_BLOCK,
    TOPK,
    WRITER_MAP,
    _u32_from_u8x4,
    _u64_from_u8x8,
    _apply_swizzle,
    hopper_value_swizzle_index,
    _load_swizzled_block_u8x16,
)

@parameter
fn moe_w1_mxfp4_swiglu_wgmma[
    BM: Int = 64,
    BN_RAW: Int = 64,
    BK: Int = 64,
    WGMMA_M: Int = 64,
    WGMMA_N: Int = 64,
    WGMMA_K: Int = 16,
    NUM_WARP_GROUPS: Int = 1,
    USE_VALUE_SWIZZLE: Bool = False,
](
    x_ptr: UnsafePointer[BFloat16, MutAnyOrigin],
    T: Int,
    D: Int,
    token_expert_order_ptr: UnsafePointer[UInt32, MutAnyOrigin],
    P: Int,
    expert_start_ptr: UnsafePointer[UInt32, MutAnyOrigin],
    expert_ids_ptr: UnsafePointer[Int32, MutAnyOrigin],
    expert_usage_stats_ptr: UnsafePointer[UInt32, MutAnyOrigin],
    w_blocks_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    w_scales_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    Kblocks: Int,
    N_raw_total: Int,
    b_ptr: UnsafePointer[Float32, MutAnyOrigin],
    h_ptr: UnsafePointer[BFloat16, MutAnyOrigin],
    I: Int,
    alpha: Scalar[F32],
    limit: Scalar[F32],
):
    constrained[WGMMA_M == 64, "SM90 WGMMA requires M=64 for this kernel"]()
    constrained[WGMMA_K in (8, 16, 32)]()
    constrained[BM % WGMMA_M == 0]()
    constrained[BN_RAW % WGMMA_N == 0]()
    constrained[BK % WGMMA_K == 0]()
    constrained[BK % VALUES_PER_BLOCK == 0, "BK must be a multiple of 32"]()
    constrained[BM % (WGMMA_M * NUM_WARP_GROUPS) == 0]()

    var expert_idx = Int(block_idx.z)
    var num_active_experts = Int(expert_usage_stats_ptr[1])
    if expert_idx >= num_active_experts:
        return

    var expert_id = Int(expert_ids_ptr[expert_idx])
    if expert_id < 0:
        return

    var n_tile_act = Int(block_idx.x)
    var n_act0 = n_tile_act * (BN_RAW // 2)
    var n_raw0 = n_tile_act * BN_RAW

    var wg_idx = Int(thread_idx.x) >> 7
    var warp_in_wg = Int(warp_id() & UInt(3))

    var seg_start = Int(expert_start_ptr[expert_idx])
    var seg_end = Int(expert_start_ptr[expert_idx + 1])
    var row_base = seg_start + Int(block_idx.y) * BM
    if row_base >= seg_end or row_base >= P:
        return
    var row_stride = BM * Int(grid_dim.y)

    # Routing: token ids for BM rows (pair_idx sorted by expert, TOPK packed).
    var token_ids_s = stack_allocation[
        BM, Scalar[U32], address_space = AddressSpace.SHARED
    ]()

    comptime blocks_per_tile = BK // VALUES_PER_BLOCK

    # Shared tiles in WGMMA-friendly layouts.
    #
    # Use dynamic shared memory for the 2-stage pipeline so we can exceed the
    # 48KB static shared memory limit (needed for BM=128/BN=128).
    comptime b_swizzle = (
        TensorMapSwizzle.SWIZZLE_128B if USE_VALUE_SWIZZLE else TensorMapSwizzle.SWIZZLE_NONE
    )
    comptime a_smem_layout = tile_layout_k_major[BF16, BM, BK]()
    comptime b_smem_layout = tile_layout_k_major[BF16, BN_RAW, BK, b_swizzle]()
    comptime b_smem_swizzle = make_swizzle[BF16, b_swizzle]()

    comptime a_bytes = a_smem_layout.size() * 2  # BF16 = 2 bytes
    comptime b_bytes = b_smem_layout.size() * 2  # BF16 = 2 bytes
    comptime a1_off = ((a_bytes + 255) // 256) * 256
    comptime b0_off = ((a1_off + a_bytes + 255) // 256) * 256
    comptime b1_off = ((b0_off + b_bytes + 255) // 256) * 256
    comptime pack_bytes = BN_RAW * blocks_per_tile * BYTES_PER_BLOCK
    comptime pack0_off = ((b1_off + b_bytes + 255) // 256) * 256
    comptime pack1_off = ((pack0_off + pack_bytes + 255) // 256) * 256

    var smem = external_memory[
        Scalar[U8],
        address_space = AddressSpace.SHARED,
        alignment=256,
        name="moe_mxfp4_w1_dynamic_smem",
    ]()
    var smem_ptr = smem.address_space_cast[AddressSpace.SHARED]().mut_cast[
        True
    ]()

    var A_s0 = LayoutTensor[
        BF16,
        a_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=256,
    ](smem_ptr.bitcast[Scalar[BF16]]())
    var A_s1 = LayoutTensor[
        BF16,
        a_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=256,
    ]((smem_ptr + a1_off).bitcast[Scalar[BF16]]())

    # B is stored as [N, K] (transpose_b=True) for WGMMA.
    var B_s0 = LayoutTensor[
        BF16,
        b_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=256,
    ]((smem_ptr + b0_off).bitcast[Scalar[BF16]]())
    var B_s1 = LayoutTensor[
        BF16,
        b_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=256,
    ]((smem_ptr + b1_off).bitcast[Scalar[BF16]]())
    var B_data0 = (smem_ptr + b0_off).bitcast[Scalar[BF16]]()
    var B_data1 = (smem_ptr + b1_off).bitcast[Scalar[BF16]]()
    var b_strides = B_s1.runtime_layout.stride.value

    # Packed MXFP4 staging buffers: [BN_RAW * (BK/32) blocks] * 16 bytes.
    var B_pack0 = smem_ptr + pack0_off
    var B_pack1 = smem_ptr + pack1_off

    var w_blocks_u8 = w_blocks_ptr.address_space_cast[
        AddressSpace.GLOBAL
    ]().bitcast[Scalar[U8]]()
    var B_pack0_u64 = (
        B_pack0.bitcast[Scalar[U64]]()
        .address_space_cast[AddressSpace.SHARED]()
        .mut_cast[True]()
    )
    var B_pack1_u64 = (
        B_pack1.bitcast[Scalar[U64]]()
        .address_space_cast[AddressSpace.SHARED]()
        .mut_cast[True]()
    )
    var B_pack0_u32 = (
        B_pack0.bitcast[Scalar[U32]]()
        .address_space_cast[AddressSpace.SHARED]()
        .mut_cast[True]()
    )
    var B_pack1_u32 = (
        B_pack1.bitcast[Scalar[U32]]()
        .address_space_cast[AddressSpace.SHARED]()
        .mut_cast[True]()
    )
    var w_blocks_stride0 = 0
    var w_blocks_stride1 = 0
    var w_blocks_stride2 = 0

    @parameter
    if USE_VALUE_SWIZZLE:
        var kbytes = Kblocks * BYTES_PER_BLOCK
        var m_pad = ((N_raw_total + 15) // 16) * 16
        var k_pad = ((kbytes + 31) // 32) * 32
        var w_m2 = m_pad // 4
        var w_k2 = k_pad * 4
        w_blocks_stride0 = w_m2 * w_k2
        w_blocks_stride1 = w_k2
        w_blocks_stride2 = 1

    comptime num_m_mmas_total = BM // WGMMA_M
    comptime num_m_mmas = num_m_mmas_total // NUM_WARP_GROUPS
    comptime num_n_mmas = BN_RAW // WGMMA_N
    comptime c_frag_size = (WGMMA_M * WGMMA_N) // 128

    var c_reg_tile = (
        LayoutTensor[
            F32,
            Layout.row_major(num_m_mmas * num_n_mmas, c_frag_size),
            MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .fill(0.0)
    )

    var wgmma = TensorCoreAsync[
        F32,
        BF16,
        BF16,
        IndexList[3](WGMMA_M, WGMMA_N, WGMMA_K),
        b_swizzle=b_swizzle,
        transpose_b=True,
    ]()

    comptime a_chunks = BK // 4

    for row0 in range(row_base, seg_end, row_stride):
        for r in range(Int(thread_idx.x), BM, Int(block_dim.x)):
            var global_row = row0 + r
            if global_row < seg_end and global_row < P:
                var pair_idx = Int(token_expert_order_ptr[global_row])
                token_ids_s[r] = UInt32(pair_idx // TOPK)
            else:
                token_ids_s[r] = UInt32(0)
        barrier()

        _ = c_reg_tile.fill(0.0)

        # Preload first K tile into buffer 0.
        var kb0 = 0
        for idx in range(
            Int(thread_idx.x),
            BN_RAW * blocks_per_tile,
            Int(block_dim.x),
        ):
            var c = idx // blocks_per_tile
            var block_in_tile = idx - c * blocks_per_tile
            var n_raw = n_raw0 + c
            var kb = kb0 + block_in_tile

            if n_raw < N_raw_total and kb < Kblocks:

                @parameter
                if USE_VALUE_SWIZZLE:
                    var packed0, packed1 = _load_swizzled_block_u8x16(
                        w_blocks_ptr,
                        w_blocks_stride0,
                        w_blocks_stride1,
                        w_blocks_stride2,
                        expert_id,
                        n_raw,
                        kb,
                    )
                    var base_u64 = idx * 2
                    B_pack0_u64.store[alignment=8](
                        base_u64 + 0, Scalar[U64](_u64_from_u8x8(packed0))
                    )
                    B_pack0_u64.store[alignment=8](
                        base_u64 + 1, Scalar[U64](_u64_from_u8x8(packed1))
                    )
                else:
                    var packed_base = (
                        (expert_id * N_raw_total + n_raw) * Kblocks + kb
                    ) * BYTES_PER_BLOCK
                    async_copy[16](
                        w_blocks_u8 + packed_base,
                        B_pack0 + idx * BYTES_PER_BLOCK,
                    )
            else:
                var base_u64 = idx * 2
                B_pack0_u64.store[alignment=8](base_u64 + 0, Scalar[U64](0))
                B_pack0_u64.store[alignment=8](base_u64 + 1, Scalar[U64](0))

        @parameter
        if not USE_VALUE_SWIZZLE:
            async_copy_commit_group()

        for idx in range(Int(thread_idx.x), BM * a_chunks, Int(block_dim.x)):
            var r = idx // a_chunks
            var chunk = idx - r * a_chunks
            var kk = chunk * 4
            var global_row = row0 + r
            var global_k = kk

            if global_row < seg_end and global_row < P and global_k + 3 < D:
                var tok = Int(token_ids_s[r][0])
                if tok < T:
                    var p64 = rebind[UnsafePointer[UInt64, MutAnyOrigin]](
                        x_ptr + (tok * D + global_k)
                    )
                    var v4 = bitcast[DType.bfloat16, 4](p64[0])
                    A_s0[r, kk + 0] = v4[0]
                    A_s0[r, kk + 1] = v4[1]
                    A_s0[r, kk + 2] = v4[2]
                    A_s0[r, kk + 3] = v4[3]
                else:
                    A_s0[r, kk + 0] = 0
                    A_s0[r, kk + 1] = 0
                    A_s0[r, kk + 2] = 0
                    A_s0[r, kk + 3] = 0
            else:

                @parameter
                for i in range(4):
                    var k_i = global_k + i
                    if global_row < seg_end and global_row < P and k_i < D:
                        var tok = Int(token_ids_s[r][0])
                        if tok < T:
                            A_s0[r, kk + i] = x_ptr[tok * D + k_i]
                        else:
                            A_s0[r, kk + i] = 0
                    else:
                        A_s0[r, kk + i] = 0

        @parameter
        if USE_VALUE_SWIZZLE:
            barrier()
        else:
            async_copy_wait_all()

        @parameter
        if USE_VALUE_SWIZZLE and WRITER_MAP == 2:
            for gidx in range(
                Int(thread_idx.x),
                BN_RAW * blocks_per_tile * 4,
                Int(block_dim.x),
            ):
                var block_idx = gidx >> 2
                var g = gidx & 3
                var c = block_idx // blocks_per_tile
                var block_in_tile = block_idx - c * blocks_per_tile
                var n_raw = n_raw0 + c
                var kb = kb0 + block_in_tile
                var scale_exp = UInt8(0)
                if n_raw < N_raw_total and kb < Kblocks:
                    scale_exp = w_scales_ptr[
                        ((expert_id * N_raw_total + n_raw) * Kblocks) + kb
                    ]
                var scale = e8m0_to_bf16_bits(scale_exp)

                var base_u32 = block_idx * 4
                var p = UInt32(B_pack0_u32.load[alignment=4](base_u32 + g))
                var outv = decode_mxfp4_packbits_u32_to_8xbf16_scaled(
                    p, scale
                )
                var k_base = block_in_tile * VALUES_PER_BLOCK + g * 8
                var ridx = B_s0.idx_list_t[2](fill=0)
                ridx[0] = c
                ridx[1] = k_base
                var lin = B_s0._get_offset(b_strides, ridx)
                var swz = _apply_swizzle(b_smem_swizzle, lin)
                B_data0.store[width=8, alignment=16](swz, outv)
        else:
            for idx in range(
                Int(thread_idx.x),
                BN_RAW * blocks_per_tile,
                Int(block_dim.x),
            ):
                var c = idx // blocks_per_tile
                var block_in_tile = idx - c * blocks_per_tile
                var n_raw = n_raw0 + c
                var kb = kb0 + block_in_tile
                var scale_exp = UInt8(0)
                if n_raw < N_raw_total and kb < Kblocks:
                    scale_exp = w_scales_ptr[
                        ((expert_id * N_raw_total + n_raw) * Kblocks) + kb
                    ]
                var scale = e8m0_to_bf16_bits(scale_exp)

                var base_u64 = idx * 2
                var packed0 = bitcast[DType.uint8, 8](
                    UInt64(B_pack0_u64.load[alignment=8](base_u64 + 0))
                )
                var packed1 = bitcast[DType.uint8, 8](
                    UInt64(B_pack0_u64.load[alignment=8](base_u64 + 1))
                )

                @parameter
                if USE_VALUE_SWIZZLE:

                    @parameter
                    for group in range(2):
                        var byte_base = group * 4
                        var k_base = block_in_tile * VALUES_PER_BLOCK + group * 8
                        var p = _u32_from_u8x4(
                            packed0[byte_base + 0],
                            packed0[byte_base + 1],
                            packed0[byte_base + 2],
                            packed0[byte_base + 3],
                        )
                        var outv = decode_mxfp4_packbits_u32_to_8xbf16_scaled(
                            p, scale
                        )
                        var ridx = B_s0.idx_list_t[2](fill=0)
                        ridx[0] = c
                        ridx[1] = k_base
                        var lin = B_s0._get_offset(b_strides, ridx)
                        var swz = _apply_swizzle(b_smem_swizzle, lin)
                        B_data0.store[width=8, alignment=16](swz, outv)

                    @parameter
                    for group in range(2):
                        var byte_base = group * 4
                        var k_base = (
                            block_in_tile * VALUES_PER_BLOCK + 16 + group * 8
                        )
                        var p = _u32_from_u8x4(
                            packed1[byte_base + 0],
                            packed1[byte_base + 1],
                            packed1[byte_base + 2],
                            packed1[byte_base + 3],
                        )
                        var outv = decode_mxfp4_packbits_u32_to_8xbf16_scaled(
                            p, scale
                        )
                        var ridx = B_s0.idx_list_t[2](fill=0)
                        ridx[0] = c
                        ridx[1] = k_base
                        var lin = B_s0._get_offset(b_strides, ridx)
                        var swz = _apply_swizzle(b_smem_swizzle, lin)
                        B_data0.store[width=8, alignment=16](swz, outv)
                else:

                    @parameter
                    for byte_in_block in range(8):
                        var r0 = (
                            block_in_tile * VALUES_PER_BLOCK + byte_in_block * 2
                        )
                        var v2 = decode_mxfp4_byte_to_2xbf16_scaled(
                            packed0[byte_in_block], scale
                        )
                        B_s0[c, r0 + 0] = v2[0]
                        B_s0[c, r0 + 1] = v2[1]

                    @parameter
                    for byte_in_block in range(8):
                        var byte = byte_in_block + 8
                        var r0 = block_in_tile * VALUES_PER_BLOCK + byte * 2
                        var v2 = decode_mxfp4_byte_to_2xbf16_scaled(
                            packed1[byte_in_block], scale
                        )
                        B_s0[c, r0 + 0] = v2[0]
                        B_s0[c, r0 + 1] = v2[1]

        barrier()

        var num_k_tiles = ceildiv(D, BK)
        for k_tile in range(num_k_tiles):
            var use_buf0 = (k_tile & 1) == 0

            warpgroup_fence(c_reg_tile)
            wgmma.arrive()
            if use_buf0:
                wgmma.wgmma[num_warp_groups=NUM_WARP_GROUPS](
                    a_smem_tile=A_s0,
                    b_smem_tile=B_s0,
                    c_reg_tile=c_reg_tile,
                    wg_idx=wg_idx,
                )
            else:
                wgmma.wgmma[num_warp_groups=NUM_WARP_GROUPS](
                    a_smem_tile=A_s1,
                    b_smem_tile=B_s1,
                    c_reg_tile=c_reg_tile,
                    wg_idx=wg_idx,
                )
            wgmma.commit_group()
            warpgroup_fence(c_reg_tile)

            if k_tile + 1 < num_k_tiles:
                # Allow at most one group to remain pending before overwriting the
                # next buffer (2-stage pipeline).
                wgmma.wait_group[1]()

                var k0_next = (k_tile + 1) * BK
                var kb0_next = k0_next // VALUES_PER_BLOCK

                if use_buf0:
                    # Load next tile into buffer 1.
                    for idx in range(
                        Int(thread_idx.x),
                        BN_RAW * blocks_per_tile,
                        Int(block_dim.x),
                    ):
                        var c = idx // blocks_per_tile
                        var block_in_tile = idx - c * blocks_per_tile
                        var n_raw = n_raw0 + c
                        var kb = kb0_next + block_in_tile

                        if n_raw < N_raw_total and kb < Kblocks:

                            @parameter
                            if USE_VALUE_SWIZZLE:
                                var packed0, packed1 = (
                                    _load_swizzled_block_u8x16(
                                        w_blocks_ptr,
                                        w_blocks_stride0,
                                        w_blocks_stride1,
                                        w_blocks_stride2,
                                        expert_id,
                                        n_raw,
                                        kb,
                                    )
                                )
                                var base_u64 = idx * 2
                                B_pack1_u64.store[alignment=8](
                                    base_u64 + 0,
                                    Scalar[U64](_u64_from_u8x8(packed0)),
                                )
                                B_pack1_u64.store[alignment=8](
                                    base_u64 + 1,
                                    Scalar[U64](_u64_from_u8x8(packed1)),
                                )
                            else:
                                var packed_base = (
                                    (expert_id * N_raw_total + n_raw) * Kblocks
                                    + kb
                                ) * BYTES_PER_BLOCK
                                async_copy[16](
                                    w_blocks_u8 + packed_base,
                                    B_pack1 + idx * BYTES_PER_BLOCK,
                                )
                        else:
                            var base_u64 = idx * 2
                            B_pack1_u64.store[alignment=8](
                                base_u64 + 0, Scalar[U64](0)
                            )
                            B_pack1_u64.store[alignment=8](
                                base_u64 + 1, Scalar[U64](0)
                            )

                    @parameter
                    if not USE_VALUE_SWIZZLE:
                        async_copy_commit_group()

                    for idx in range(
                        Int(thread_idx.x), BM * a_chunks, Int(block_dim.x)
                    ):
                        var r = idx // a_chunks
                        var chunk = idx - r * a_chunks
                        var kk = chunk * 4
                        var global_row = row0 + r
                        var global_k = k0_next + kk

                        if (
                            global_row < seg_end
                            and global_row < P
                            and global_k + 3 < D
                        ):
                            var tok = Int(token_ids_s[r][0])
                            if tok < T:
                                var p64 = rebind[
                                    UnsafePointer[UInt64, MutAnyOrigin]
                                ](x_ptr + (tok * D + global_k))
                                var v4 = bitcast[DType.bfloat16, 4](p64[0])
                                A_s1[r, kk + 0] = v4[0]
                                A_s1[r, kk + 1] = v4[1]
                                A_s1[r, kk + 2] = v4[2]
                                A_s1[r, kk + 3] = v4[3]
                            else:
                                A_s1[r, kk + 0] = 0
                                A_s1[r, kk + 1] = 0
                                A_s1[r, kk + 2] = 0
                                A_s1[r, kk + 3] = 0
                        else:

                            @parameter
                            for i in range(4):
                                var k_i = global_k + i
                                if (
                                    global_row < seg_end
                                    and global_row < P
                                    and k_i < D
                                ):
                                    var tok = Int(token_ids_s[r][0])
                                    if tok < T:
                                        A_s1[r, kk + i] = x_ptr[tok * D + k_i]
                                    else:
                                        A_s1[r, kk + i] = 0
                                else:
                                    A_s1[r, kk + i] = 0

                    @parameter
                    if USE_VALUE_SWIZZLE:
                        barrier()
                    else:
                        async_copy_wait_all()

                    for idx in range(
                        Int(thread_idx.x),
                        BN_RAW * blocks_per_tile,
                        Int(block_dim.x),
                    ):
                        var c = idx // blocks_per_tile
                        var block_in_tile = idx - c * blocks_per_tile
                        var n_raw = n_raw0 + c
                        var kb = kb0_next + block_in_tile
                        var scale_exp = UInt8(0)
                        if n_raw < N_raw_total and kb < Kblocks:
                            scale_exp = w_scales_ptr[
                                ((expert_id * N_raw_total + n_raw) * Kblocks)
                                + kb
                            ]
                        var scale = e8m0_to_bf16_bits(scale_exp)

                        var base_u64 = idx * 2
                        var packed0 = bitcast[DType.uint8, 8](
                            UInt64(B_pack1_u64.load[alignment=8](base_u64 + 0))
                        )
                        var packed1 = bitcast[DType.uint8, 8](
                            UInt64(B_pack1_u64.load[alignment=8](base_u64 + 1))
                        )

                        @parameter
                        if USE_VALUE_SWIZZLE:
                            @parameter
                            for group in range(2):
                                var byte_base = group * 4
                                var k_base = (
                                    block_in_tile * VALUES_PER_BLOCK + group * 8
                                )
                                var p = _u32_from_u8x4(
                                    packed0[byte_base + 0],
                                    packed0[byte_base + 1],
                                    packed0[byte_base + 2],
                                    packed0[byte_base + 3],
                                )
                                var outv = (
                                    decode_mxfp4_packbits_u32_to_8xbf16_scaled(
                                        p, scale
                                    )
                                )
                                var ridx = B_s1.idx_list_t[2](fill=0)
                                ridx[0] = c
                                ridx[1] = k_base
                                var lin = B_s1._get_offset(b_strides, ridx)
                                var swz = _apply_swizzle(b_smem_swizzle, lin)
                                B_data1.store[width=8, alignment=16](swz, outv)

                            @parameter
                            for group in range(2):
                                var byte_base = group * 4
                                var k_base = (
                                    block_in_tile * VALUES_PER_BLOCK
                                    + 16
                                    + group * 8
                                )
                                var p = _u32_from_u8x4(
                                    packed1[byte_base + 0],
                                    packed1[byte_base + 1],
                                    packed1[byte_base + 2],
                                    packed1[byte_base + 3],
                                )
                                var outv = (
                                    decode_mxfp4_packbits_u32_to_8xbf16_scaled(
                                        p, scale
                                    )
                                )
                                var ridx = B_s1.idx_list_t[2](fill=0)
                                ridx[0] = c
                                ridx[1] = k_base
                                var lin = B_s1._get_offset(b_strides, ridx)
                                var swz = _apply_swizzle(b_smem_swizzle, lin)
                                B_data1.store[width=8, alignment=16](swz, outv)
                        else:

                            @parameter
                            for byte_in_block in range(8):
                                var r0 = (
                                    block_in_tile * VALUES_PER_BLOCK
                                    + byte_in_block * 2
                                )
                                var v2 = decode_mxfp4_byte_to_2xbf16_scaled(
                                    packed0[byte_in_block], scale
                                )
                                B_s1[c, r0 + 0] = v2[0]
                                B_s1[c, r0 + 1] = v2[1]

                            @parameter
                            for byte_in_block in range(8):
                                var byte = byte_in_block + 8
                                var r0 = (
                                    block_in_tile * VALUES_PER_BLOCK + byte * 2
                                )
                                var v2 = decode_mxfp4_byte_to_2xbf16_scaled(
                                    packed1[byte_in_block], scale
                                )
                                B_s1[c, r0 + 0] = v2[0]
                                B_s1[c, r0 + 1] = v2[1]
                    barrier()
                else:
                    # Load next tile into buffer 0.
                    for idx in range(
                        Int(thread_idx.x),
                        BN_RAW * blocks_per_tile,
                        Int(block_dim.x),
                    ):
                        var c = idx // blocks_per_tile
                        var block_in_tile = idx - c * blocks_per_tile
                        var n_raw = n_raw0 + c
                        var kb = kb0_next + block_in_tile

                        if n_raw < N_raw_total and kb < Kblocks:

                            @parameter
                            if USE_VALUE_SWIZZLE:
                                var packed0, packed1 = (
                                    _load_swizzled_block_u8x16(
                                        w_blocks_ptr,
                                        w_blocks_stride0,
                                        w_blocks_stride1,
                                        w_blocks_stride2,
                                        expert_id,
                                        n_raw,
                                        kb,
                                    )
                                )
                                var base_u64 = idx * 2
                                B_pack0_u64.store[alignment=8](
                                    base_u64 + 0,
                                    Scalar[U64](_u64_from_u8x8(packed0)),
                                )
                                B_pack0_u64.store[alignment=8](
                                    base_u64 + 1,
                                    Scalar[U64](_u64_from_u8x8(packed1)),
                                )
                            else:
                                var packed_base = (
                                    (expert_id * N_raw_total + n_raw) * Kblocks
                                    + kb
                                ) * BYTES_PER_BLOCK
                                async_copy[16](
                                    w_blocks_u8 + packed_base,
                                    B_pack0 + idx * BYTES_PER_BLOCK,
                                )
                        else:
                            var base_u64 = idx * 2
                            B_pack0_u64.store[alignment=8](
                                base_u64 + 0, Scalar[U64](0)
                            )
                            B_pack0_u64.store[alignment=8](
                                base_u64 + 1, Scalar[U64](0)
                            )

                    @parameter
                    if not USE_VALUE_SWIZZLE:
                        async_copy_commit_group()

                    for idx in range(
                        Int(thread_idx.x), BM * a_chunks, Int(block_dim.x)
                    ):
                        var r = idx // a_chunks
                        var chunk = idx - r * a_chunks
                        var kk = chunk * 4
                        var global_row = row0 + r
                        var global_k = k0_next + kk

                        if (
                            global_row < seg_end
                            and global_row < P
                            and global_k + 3 < D
                        ):
                            var tok = Int(token_ids_s[r][0])
                            if tok < T:
                                var p64 = rebind[
                                    UnsafePointer[UInt64, MutAnyOrigin]
                                ](x_ptr + (tok * D + global_k))
                                var v4 = bitcast[DType.bfloat16, 4](p64[0])
                                A_s0[r, kk + 0] = v4[0]
                                A_s0[r, kk + 1] = v4[1]
                                A_s0[r, kk + 2] = v4[2]
                                A_s0[r, kk + 3] = v4[3]
                            else:
                                A_s0[r, kk + 0] = 0
                                A_s0[r, kk + 1] = 0
                                A_s0[r, kk + 2] = 0
                                A_s0[r, kk + 3] = 0
                        else:

                            @parameter
                            for i in range(4):
                                var k_i = global_k + i
                                if (
                                    global_row < seg_end
                                    and global_row < P
                                    and k_i < D
                                ):
                                    var tok = Int(token_ids_s[r][0])
                                    if tok < T:
                                        A_s0[r, kk + i] = x_ptr[tok * D + k_i]
                                    else:
                                        A_s0[r, kk + i] = 0
                                else:
                                    A_s0[r, kk + i] = 0

                    @parameter
                    if USE_VALUE_SWIZZLE:
                        barrier()
                    else:
                        async_copy_wait_all()

                    @parameter
                    if USE_VALUE_SWIZZLE and WRITER_MAP == 2:
                        for gidx in range(
                            Int(thread_idx.x),
                            BN_RAW * blocks_per_tile * 4,
                            Int(block_dim.x),
                        ):
                            var block_idx = gidx >> 2
                            var g = gidx & 3
                            var c = block_idx // blocks_per_tile
                            var block_in_tile = block_idx - c * blocks_per_tile
                            var n_raw = n_raw0 + c
                            var kb = kb0_next + block_in_tile
                            var scale_exp = UInt8(0)
                            if n_raw < N_raw_total and kb < Kblocks:
                                scale_exp = w_scales_ptr[
                                    ((expert_id * N_raw_total + n_raw) * Kblocks)
                                    + kb
                                ]
                            var scale = e8m0_to_bf16_bits(scale_exp)

                            var base_u32 = block_idx * 4
                            var p = UInt32(
                                B_pack0_u32.load[alignment=4](base_u32 + g)
                            )
                            var outv = (
                                decode_mxfp4_packbits_u32_to_8xbf16_scaled(
                                    p, scale
                                )
                            )
                            var k_base = (
                                block_in_tile * VALUES_PER_BLOCK + g * 8
                            )
                            var ridx = B_s0.idx_list_t[2](fill=0)
                            ridx[0] = c
                            ridx[1] = k_base
                            var lin = B_s0._get_offset(b_strides, ridx)
                            var swz = _apply_swizzle(b_smem_swizzle, lin)
                            B_data0.store[width=8, alignment=16](swz, outv)
                    else:
                        for idx in range(
                            Int(thread_idx.x),
                            BN_RAW * blocks_per_tile,
                            Int(block_dim.x),
                        ):
                            var c = idx // blocks_per_tile
                            var block_in_tile = idx - c * blocks_per_tile
                            var n_raw = n_raw0 + c
                            var kb = kb0_next + block_in_tile
                            var scale_exp = UInt8(0)
                            if n_raw < N_raw_total and kb < Kblocks:
                                scale_exp = w_scales_ptr[
                                    ((expert_id * N_raw_total + n_raw) * Kblocks)
                                    + kb
                                ]
                            var scale = e8m0_to_bf16_bits(scale_exp)

                            var base_u64 = idx * 2
                            var packed0 = bitcast[DType.uint8, 8](
                                UInt64(
                                    B_pack0_u64.load[alignment=8](base_u64 + 0)
                                )
                            )
                            var packed1 = bitcast[DType.uint8, 8](
                                UInt64(
                                    B_pack0_u64.load[alignment=8](base_u64 + 1)
                                )
                            )

                            @parameter
                            if USE_VALUE_SWIZZLE:
                                @parameter
                                if WRITER_MAP == 0:

                                    @parameter
                                    for group in range(4):
                                        var g = group
                                        var byte_base = (g & 1) * 4
                                        var b0 = packed0[byte_base + 0]
                                        var b1 = packed0[byte_base + 1]
                                        var b2 = packed0[byte_base + 2]
                                        var b3 = packed0[byte_base + 3]
                                        if g >= 2:
                                            b0 = packed1[byte_base + 0]
                                            b1 = packed1[byte_base + 1]
                                            b2 = packed1[byte_base + 2]
                                            b3 = packed1[byte_base + 3]
                                        var p = _u32_from_u8x4(b0, b1, b2, b3)
                                        var outv = (
                                            decode_mxfp4_packbits_u32_to_8xbf16_scaled(
                                                p, scale
                                            )
                                        )
                                        var k_base = (
                                            block_in_tile * VALUES_PER_BLOCK + g * 8
                                        )
                                        var ridx = B_s0.idx_list_t[2](fill=0)
                                        ridx[0] = c
                                        ridx[1] = k_base
                                        var lin = B_s0._get_offset(b_strides, ridx)
                                        var swz = _apply_swizzle(b_smem_swizzle, lin)
                                        B_data0.store[width=8, alignment=16](
                                            swz, outv
                                        )
                                elif WRITER_MAP == 1:
                                    var phase = Int(thread_idx.x) & 1
                                    if phase == 0:
                                        @parameter
                                        for group in range(4):
                                            var g = group
                                            var byte_base = (g & 1) * 4
                                            var b0 = packed0[byte_base + 0]
                                            var b1 = packed0[byte_base + 1]
                                            var b2 = packed0[byte_base + 2]
                                            var b3 = packed0[byte_base + 3]
                                            if g >= 2:
                                                b0 = packed1[byte_base + 0]
                                                b1 = packed1[byte_base + 1]
                                                b2 = packed1[byte_base + 2]
                                                b3 = packed1[byte_base + 3]
                                            var p = _u32_from_u8x4(b0, b1, b2, b3)
                                            var outv = (
                                                decode_mxfp4_packbits_u32_to_8xbf16_scaled(
                                                    p, scale
                                                )
                                            )
                                            var k_base = (
                                                block_in_tile * VALUES_PER_BLOCK
                                                + g * 8
                                            )
                                            var ridx = B_s0.idx_list_t[2](fill=0)
                                            ridx[0] = c
                                            ridx[1] = k_base
                                            var lin = B_s0._get_offset(
                                                b_strides, ridx
                                            )
                                            var swz = _apply_swizzle(
                                                b_smem_swizzle, lin
                                            )
                                            B_data0.store[width=8, alignment=16](
                                                swz, outv
                                            )
                                    else:
                                        @parameter
                                        for group in range(4):
                                            var g = group
                                            if g == 0:
                                                g = 1
                                            elif g == 1:
                                                g = 0
                                            elif g == 2:
                                                g = 3
                                            else:
                                                g = 2
                                            var byte_base = (g & 1) * 4
                                            var b0 = packed0[byte_base + 0]
                                            var b1 = packed0[byte_base + 1]
                                            var b2 = packed0[byte_base + 2]
                                            var b3 = packed0[byte_base + 3]
                                            if g >= 2:
                                                b0 = packed1[byte_base + 0]
                                                b1 = packed1[byte_base + 1]
                                                b2 = packed1[byte_base + 2]
                                                b3 = packed1[byte_base + 3]
                                            var p = _u32_from_u8x4(b0, b1, b2, b3)
                                            var outv = (
                                                decode_mxfp4_packbits_u32_to_8xbf16_scaled(
                                                    p, scale
                                                )
                                            )
                                            var k_base = (
                                                block_in_tile * VALUES_PER_BLOCK
                                                + g * 8
                                            )
                                            var ridx = B_s0.idx_list_t[2](fill=0)
                                            ridx[0] = c
                                            ridx[1] = k_base
                                            var lin = B_s0._get_offset(
                                                b_strides, ridx
                                            )
                                            var swz = _apply_swizzle(
                                                b_smem_swizzle, lin
                                            )
                                            B_data0.store[width=8, alignment=16](
                                                swz, outv
                                            )
                                else:
                                    var lane_phase = Int(thread_idx.x) & 3
                                    @parameter
                                    for step in range(4):
                                        var g = (step + lane_phase) & 3
                                        var byte_base = (g & 1) * 4
                                        var b0 = packed0[byte_base + 0]
                                        var b1 = packed0[byte_base + 1]
                                        var b2 = packed0[byte_base + 2]
                                        var b3 = packed0[byte_base + 3]
                                        if g >= 2:
                                            b0 = packed1[byte_base + 0]
                                            b1 = packed1[byte_base + 1]
                                            b2 = packed1[byte_base + 2]
                                            b3 = packed1[byte_base + 3]
                                        var p = _u32_from_u8x4(b0, b1, b2, b3)
                                        var outv = (
                                            decode_mxfp4_packbits_u32_to_8xbf16_scaled(
                                                p, scale
                                            )
                                        )
                                        var k_base = (
                                            block_in_tile * VALUES_PER_BLOCK + g * 8
                                        )
                                        var ridx = B_s0.idx_list_t[2](fill=0)
                                        ridx[0] = c
                                        ridx[1] = k_base
                                        var lin = B_s0._get_offset(b_strides, ridx)
                                        var swz = _apply_swizzle(b_smem_swizzle, lin)
                                        B_data0.store[width=8, alignment=16](
                                            swz, outv
                                        )
                            else:

                                @parameter
                                for byte_in_block in range(8):
                                    var r0 = (
                                        block_in_tile * VALUES_PER_BLOCK
                                        + byte_in_block * 2
                                    )
                                    var v2 = decode_mxfp4_byte_to_2xbf16_scaled(
                                        packed0[byte_in_block], scale
                                    )
                                    B_s0[c, r0 + 0] = v2[0]
                                    B_s0[c, r0 + 1] = v2[1]

                                @parameter
                                for byte_in_block in range(8):
                                    var byte = byte_in_block + 8
                                    var r0 = (
                                        block_in_tile * VALUES_PER_BLOCK + byte * 2
                                    )
                                    var v2 = decode_mxfp4_byte_to_2xbf16_scaled(
                                        packed1[byte_in_block], scale
                                    )
                                    B_s0[c, r0 + 0] = v2[0]
                                    B_s0[c, r0 + 1] = v2[1]
                    barrier()

                # Next iteration can proceed after B_s0/B_s1 is decoded.

        # Ensure all WGMMA groups are complete before the epilogue.
        wgmma.wait_group()

        # Epilogue: bias + SwiGLU directly from register fragments.
        var lane = Int(thread_idx.x & 31)
        var lane_row = lane // 4
        var lane_col = lane - lane_row * 4

        comptime warp_rows = WGMMA_M // 4
        comptime row_iters = warp_rows // 8
        comptime col_iters = (WGMMA_N // 2) // 4

        @parameter
        for m_mma in range(num_m_mmas):

            @parameter
            for n_mma in range(num_n_mmas):
                comptime mma_id = n_mma * num_m_mmas + m_mma
                var c_frag = c_reg_tile.tile[1, c_frag_size](mma_id, 0)
                var c_pairs = c_frag.vectorize[1, 2]()

                @parameter
                for r_it in range(row_iters):
                    var row_in_warp = lane_row + r_it * 8
                    var row_in_cta = (
                        wg_idx * (num_m_mmas * WGMMA_M)
                        + m_mma * WGMMA_M
                        + warp_in_wg * warp_rows
                        + row_in_warp
                    )
                    var global_row = row0 + row_in_cta
                    if global_row >= seg_end or global_row >= P:
                        continue

                    @parameter
                    for c_it in range(col_iters):
                        var col_pair = lane_col + c_it * 4
                        var out_col = n_act0 + n_mma * (WGMMA_N // 2) + col_pair
                        if out_col >= I:
                            continue

                        var raw0 = n_raw0 + n_mma * WGMMA_N + col_pair * 2
                        var raw1 = raw0 + 1

                        var pair_idx = c_it * row_iters + r_it
                        var v2 = c_pairs[0, pair_idx]

                        var glu = v2[0] + b_ptr[expert_id * N_raw_total + raw0]
                        var lin = v2[1] + b_ptr[expert_id * N_raw_total + raw1]
                        var y = swiglu_pair(glu, lin, alpha, limit)
                        h_ptr.store(global_row * I + out_col, y.cast[BF16]())


@parameter
fn moe_w1_mxfp4_swiglu_tc[
    BM: Int = 128,
    BN_RAW: Int = 64,
    BK: Int = 32,
    WM: Int = 32,
    WN: Int = 32,
    MMA_M: Int = 16,
    MMA_N: Int = 8,
    MMA_K: Int = 16,
](
    x_ptr: UnsafePointer[BFloat16, MutAnyOrigin],
    T: Int,
    D: Int,
    token_expert_order_ptr: UnsafePointer[UInt32, MutAnyOrigin],
    P: Int,
    expert_start_ptr: UnsafePointer[UInt32, MutAnyOrigin],
    expert_ids_ptr: UnsafePointer[Int32, MutAnyOrigin],
    num_active_experts: Int,
    w_blocks_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    w_scales_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    Kblocks: Int,
    N_raw_total: Int,
    b_ptr: UnsafePointer[Float32, MutAnyOrigin],
    h_ptr: UnsafePointer[BFloat16, MutAnyOrigin],
    I: Int,
    alpha: Scalar[F32],
    limit: Scalar[F32],
):
    var expert_idx = Int(block_idx.z)
    if expert_idx >= num_active_experts:
        return

    var expert_id = Int(expert_ids_ptr[expert_idx])
    if expert_id < 0:
        return

    var n_tile_act = Int(block_idx.x)
    var n_act0 = n_tile_act * (BN_RAW // 2)
    var n_raw0 = n_tile_act * BN_RAW

    var seg_start = Int(expert_start_ptr[expert_idx])
    var seg_end = Int(expert_start_ptr[expert_idx + 1])
    var row0 = seg_start + Int(block_idx.y) * BM

    # Shared staging.
    var token_ids_s = stack_allocation[
        BM, Scalar[U32], address_space = AddressSpace.SHARED
    ]()

    var A_s = LayoutTensor[
        BF16,
        Layout.row_major(BM, BK),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var B_s = LayoutTensor[
        BF16,
        Layout.row_major(BK, BN_RAW),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var C_s = LayoutTensor[
        F32,
        Layout.row_major(BM, BN_RAW),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Precompute token ids for the BM rows.
    for r in range(Int(thread_idx.x), BM, Int(block_dim.x)):
        var global_row = row0 + r
        if global_row < seg_end and global_row < P:
            var pair_idx = Int(token_expert_order_ptr[global_row])
            token_ids_s[r] = UInt32(pair_idx // TOPK)
        else:
            token_ids_s[r] = UInt32(0)
    barrier()

    constrained[
        WM % MMA_M == 0 and WN % MMA_N == 0 and BK % MMA_K == 0,
        "Warp tile should be an integer multiple of instruction shape",
    ]()

    # TensorCore op: bf16 * bf16 -> f32 accum.
    var mma = TensorCore[F32, BF16, Index(MMA_M, MMA_N, MMA_K)]()

    var warp_y = warp_id() // UInt(BN_RAW // WN)
    var warp_x = warp_id() % UInt(BN_RAW // WN)

    var c_reg = (
        LayoutTensor[
            F32,
            Layout.row_major(WM // MMA_M, (WN * 4) // MMA_N),
            MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .fill(0)
    )

    for k0 in range(0, D, BK):
        # Load A tile: [BM, BK]
        for idx in range(Int(thread_idx.x), BM * BK, Int(block_dim.x)):
            var r = idx // BK
            var kk = idx - r * BK
            var global_row = row0 + r
            var global_k = k0 + kk

            if global_row < seg_end and global_row < P and global_k < D:
                var tok = Int(token_ids_s[r][0])
                if tok < T:
                    A_s[r, kk] = x_ptr[tok * D + global_k]
                else:
                    A_s[r, kk] = 0
            else:
                A_s[r, kk] = 0

        # Decode B tile: [BK, BN_RAW]
        var kb = k0 // VALUES_PER_BLOCK
        for idx in range(
            Int(thread_idx.x),
            BN_RAW * BYTES_PER_BLOCK,
            Int(block_dim.x),
        ):
            var c = idx // BYTES_PER_BLOCK
            var byte_in_block = idx - c * BYTES_PER_BLOCK
            var n_raw = n_raw0 + c
            var r0 = byte_in_block * 2

            if n_raw < N_raw_total:
                var scale_exp = w_scales_ptr[
                    ((expert_id * N_raw_total + n_raw) * Kblocks) + kb
                ]
                var packed = w_blocks_ptr[
                    (
                        ((expert_id * N_raw_total + n_raw) * Kblocks + kb)
                        * BYTES_PER_BLOCK
                    )
                    + byte_in_block
                ]
                var v2 = decode_mxfp4_byte_to_2xbf16_e8m0(packed, scale_exp)
                B_s[r0 + 0, c] = v2[0]
                B_s[r0 + 1, c] = v2[1]
            else:
                B_s[r0 + 0, c] = 0
                B_s[r0 + 1, c] = 0

        barrier()

        # MMA accumulate into c_reg (warp tile WM x WN).
        var A_warp_tile = A_s.tile[WM, BK](Int(warp_y), 0)
        var B_warp_tile = B_s.tile[BK, WN](0, Int(warp_x))

        @parameter
        for mma_k in range(BK // MMA_K):

            @parameter
            for mma_m in range(WM // MMA_M):

                @parameter
                for mma_n in range(WN // MMA_N):
                    var c_tile = c_reg.tile[1, 4](mma_m, mma_n)
                    var A_mma = A_warp_tile.tile[MMA_M, MMA_K](mma_m, mma_k)
                    var B_mma = B_warp_tile.tile[MMA_K, MMA_N](mma_k, mma_n)
                    var a_reg = mma.load_a(A_mma)
                    var b_reg = mma.load_b(B_mma)
                    var d_reg = mma.mma_op(a_reg, b_reg, c_tile)
                    c_tile.copy_from(d_reg)

        barrier()

    # Store warp accumulators into shared C tile.
    var C_warp = C_s.tile[WM, WN](Int(warp_y), Int(warp_x))

    @parameter
    for mma_m in range(WM // MMA_M):

        @parameter
        for mma_n in range(WN // MMA_N):
            var C_mma = C_warp.tile[MMA_M, MMA_N](mma_m, mma_n)
            var c_tile = c_reg.tile[1, 4](mma_m, mma_n)
            mma.store_d(C_mma, c_tile)
    barrier()

    # Epilogue: bias + SwiGLU -> write activated columns [I] to H.
    var BN_ACT = BN_RAW // 2
    for idx in range(Int(thread_idx.x), BM * BN_ACT, Int(block_dim.x)):
        var r = idx // BN_ACT
        var c_act = idx - r * BN_ACT
        var global_row = row0 + r
        var out_col = n_act0 + c_act
        if global_row < seg_end and global_row < P and out_col < I:
            var raw0 = c_act * 2
            var raw1 = raw0 + 1
            var col_raw0 = n_raw0 + raw0
            var col_raw1 = n_raw0 + raw1

            var glu = (
                C_s[r, raw0][0] + b_ptr[expert_id * N_raw_total + col_raw0]
            )
            var lin = (
                C_s[r, raw1][0] + b_ptr[expert_id * N_raw_total + col_raw1]
            )

            var y = swiglu_pair(glu, lin, alpha, limit)
            h_ptr.store(global_row * I + out_col, y.cast[BF16]())


@parameter
fn moe_w2_mxfp4_scatter_tc[
    BM: Int = 128,
    BN: Int = 64,
    BK: Int = 32,
    WM: Int = 32,
    WN: Int = 32,
    MMA_M: Int = 16,
    MMA_N: Int = 8,
    MMA_K: Int = 16,
](
    h_ptr: UnsafePointer[BFloat16, MutAnyOrigin],
    P: Int,
    I: Int,
    token_expert_order_ptr: UnsafePointer[UInt32, MutAnyOrigin],
    expert_start_ptr: UnsafePointer[UInt32, MutAnyOrigin],
    expert_ids_ptr: UnsafePointer[Int32, MutAnyOrigin],
    num_active_experts: Int,
    gate_w_ptr: UnsafePointer[Float32, MutAnyOrigin],
    w_blocks_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    w_scales_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    Kblocks: Int,
    b_ptr: UnsafePointer[Float32, MutAnyOrigin],
    y_ptr: UnsafePointer[Float32, MutAnyOrigin],
    T: Int,
    D: Int,
):
    var expert_idx = Int(block_idx.z)
    if expert_idx >= num_active_experts:
        return

    var expert_id = Int(expert_ids_ptr[expert_idx])
    if expert_id < 0:
        return

    var n0 = Int(block_idx.x) * BN

    var seg_start = Int(expert_start_ptr[expert_idx])
    var seg_end = Int(expert_start_ptr[expert_idx + 1])
    var row0 = seg_start + Int(block_idx.y) * BM

    var token_ids_s = stack_allocation[
        BM, Scalar[U32], address_space = AddressSpace.SHARED
    ]()
    var gamma_s = stack_allocation[
        BM, Scalar[F32], address_space = AddressSpace.SHARED
    ]()

    var A_s = LayoutTensor[
        BF16,
        Layout.row_major(BM, BK),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var B_s = LayoutTensor[
        BF16,
        Layout.row_major(BK, BN),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var C_s = LayoutTensor[
        F32,
        Layout.row_major(BM, BN),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Precompute token ids + gamma for the BM rows.
    for r in range(Int(thread_idx.x), BM, Int(block_dim.x)):
        var global_row = row0 + r
        if global_row < seg_end and global_row < P:
            var pair_idx = Int(token_expert_order_ptr[global_row])
            token_ids_s[r] = UInt32(pair_idx // TOPK)
            gamma_s[r] = gate_w_ptr[pair_idx]
        else:
            token_ids_s[r] = UInt32(0)
            gamma_s[r] = 0.0
    barrier()

    constrained[
        WM % MMA_M == 0 and WN % MMA_N == 0 and BK % MMA_K == 0,
        "Warp tile should be an integer multiple of instruction shape",
    ]()

    var mma = TensorCore[F32, BF16, Index(MMA_M, MMA_N, MMA_K)]()

    var warp_y = warp_id() // UInt(BN // WN)
    var warp_x = warp_id() % UInt(BN // WN)

    var c_reg = (
        LayoutTensor[
            F32,
            Layout.row_major(WM // MMA_M, (WN * 4) // MMA_N),
            MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .fill(0)
    )

    for k0 in range(0, I, BK):
        # Load A tile from H: [BM, BK]
        for idx in range(Int(thread_idx.x), BM * BK, Int(block_dim.x)):
            var r = idx // BK
            var kk = idx - r * BK
            var global_row = row0 + r
            var global_k = k0 + kk
            if global_row < seg_end and global_row < P and global_k < I:
                A_s[r, kk] = h_ptr[global_row * I + global_k]
            else:
                A_s[r, kk] = 0

        # Decode B tile: [BK, BN] from FP4 blocks (packed along K=I).
        var kb = k0 // VALUES_PER_BLOCK
        for idx in range(
            Int(thread_idx.x),
            BN * BYTES_PER_BLOCK,
            Int(block_dim.x),
        ):
            var c = idx // BYTES_PER_BLOCK
            var byte_in_block = idx - c * BYTES_PER_BLOCK
            var col = n0 + c
            var r0 = byte_in_block * 2

            if col < D:
                var scale_exp = w_scales_ptr[
                    ((expert_id * D + col) * Kblocks) + kb
                ]
                var packed = w_blocks_ptr[
                    (((expert_id * D + col) * Kblocks + kb) * BYTES_PER_BLOCK)
                    + byte_in_block
                ]
                var v2 = decode_mxfp4_byte_to_2xbf16_e8m0(packed, scale_exp)
                B_s[r0 + 0, c] = v2[0]
                B_s[r0 + 1, c] = v2[1]
            else:
                B_s[r0 + 0, c] = 0
                B_s[r0 + 1, c] = 0

        barrier()

        var A_warp_tile = A_s.tile[WM, BK](Int(warp_y), 0)
        var B_warp_tile = B_s.tile[BK, WN](0, Int(warp_x))

        @parameter
        for mma_k in range(BK // MMA_K):

            @parameter
            for mma_m in range(WM // MMA_M):

                @parameter
                for mma_n in range(WN // MMA_N):
                    var c_tile = c_reg.tile[1, 4](mma_m, mma_n)
                    var A_mma = A_warp_tile.tile[MMA_M, MMA_K](mma_m, mma_k)
                    var B_mma = B_warp_tile.tile[MMA_K, MMA_N](mma_k, mma_n)
                    var a_reg = mma.load_a(A_mma)
                    var b_reg = mma.load_b(B_mma)
                    var d_reg = mma.mma_op(a_reg, b_reg, c_tile)
                    c_tile.copy_from(d_reg)

        barrier()

    # Store to shared C tile
    var C_warp = C_s.tile[WM, WN](Int(warp_y), Int(warp_x))

    @parameter
    for mma_m in range(WM // MMA_M):

        @parameter
        for mma_n in range(WN // MMA_N):
            var C_mma = C_warp.tile[MMA_M, MMA_N](mma_m, mma_n)
            var c_tile = c_reg.tile[1, 4](mma_m, mma_n)
            mma.store_d(C_mma, c_tile)
    barrier()

    # Epilogue: bias + gamma scaling + scatter-add into Y[token_id, n]
    for idx in range(Int(thread_idx.x), BM * BN, Int(block_dim.x)):
        var r = idx // BN
        var c = idx - r * BN
        var global_row = row0 + r
        var col = n0 + c
        if global_row < seg_end and global_row < P and col < D:
            var tok = Int(token_ids_s[r][0])
            if tok < T:
                var v = C_s[r, c][0] + b_ptr[expert_id * D + col]
                v *= gamma_s[r][0]

                _ = Atomic.fetch_add(y_ptr + (tok * D + col), v)


@parameter
fn moe_w2_mxfp4_scatter_wgmma[
    BM: Int = 64,
    BN: Int = 64,
    BK: Int = 64,
    WGMMA_M: Int = 64,
    WGMMA_N: Int = 64,
    WGMMA_K: Int = 16,
    NUM_WARP_GROUPS: Int = 1,
    WRITE_PAIRS: Bool = False,
    PAIR_OUT_BF16: Bool = False,
    USE_VALUE_SWIZZLE: Bool = False,
](
    h_ptr: UnsafePointer[BFloat16, MutAnyOrigin],
    P: Int,
    I: Int,
    token_expert_order_ptr: UnsafePointer[UInt32, MutAnyOrigin],
    expert_start_ptr: UnsafePointer[UInt32, MutAnyOrigin],
    expert_ids_ptr: UnsafePointer[Int32, MutAnyOrigin],
    expert_usage_stats_ptr: UnsafePointer[UInt32, MutAnyOrigin],
    gate_w_ptr: UnsafePointer[BFloat16, MutAnyOrigin],
    w_blocks_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    w_scales_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    Kblocks: Int,
    b_ptr: UnsafePointer[Float32, MutAnyOrigin],
    y_ptr: UnsafePointer[Float32, MutAnyOrigin],
    T: Int,
    D: Int,
):
    constrained[WGMMA_M == 64, "SM90 WGMMA requires M=64 for this kernel"]()
    constrained[WGMMA_K in (8, 16, 32)]()
    constrained[BM % WGMMA_M == 0]()
    constrained[BN % WGMMA_N == 0]()
    constrained[BK % WGMMA_K == 0]()
    constrained[BK % VALUES_PER_BLOCK == 0, "BK must be a multiple of 32"]()
    constrained[BM % (WGMMA_M * NUM_WARP_GROUPS) == 0]()
    constrained[
        not PAIR_OUT_BF16 or WRITE_PAIRS,
        "PAIR_OUT_BF16 is only supported when WRITE_PAIRS=True",
    ]()

    var expert_idx = Int(block_idx.z)
    var num_active_experts = Int(expert_usage_stats_ptr[1])
    if expert_idx >= num_active_experts:
        return

    var expert_id = Int(expert_ids_ptr[expert_idx])
    if expert_id < 0:
        return

    var wg_idx = Int(thread_idx.x) >> 7
    var warp_in_wg = Int(warp_id() & UInt(3))

    var n0 = Int(block_idx.x) * BN

    var seg_start = Int(expert_start_ptr[expert_idx])
    var seg_end = Int(expert_start_ptr[expert_idx + 1])
    var row_base = seg_start + Int(block_idx.y) * BM
    if row_base >= seg_end or row_base >= P:
        return
    var row_stride = BM * Int(grid_dim.y)

    var pair_idx_s = stack_allocation[
        BM, Scalar[U32], address_space = AddressSpace.SHARED
    ]()
    var gamma_s = stack_allocation[
        BM, Scalar[BF16], address_space = AddressSpace.SHARED
    ]()

    # Shared tiles in WGMMA-friendly layouts.
    #
    # Use dynamic shared memory for the 2-stage pipeline so we can exceed the
    # 48KB static shared memory limit (needed for BM=128/BN=128).
    comptime b_swizzle = (
        TensorMapSwizzle.SWIZZLE_128B if USE_VALUE_SWIZZLE else TensorMapSwizzle.SWIZZLE_NONE
    )
    comptime a_smem_layout = tile_layout_k_major[BF16, BM, BK]()
    comptime b_smem_layout = tile_layout_k_major[BF16, BN, BK, b_swizzle]()
    comptime b_smem_swizzle = make_swizzle[BF16, b_swizzle]()
    comptime blocks_per_tile = BK // VALUES_PER_BLOCK

    comptime a_bytes = a_smem_layout.size() * 2  # BF16 = 2 bytes
    comptime b_bytes = b_smem_layout.size() * 2  # BF16 = 2 bytes
    comptime a1_off = ((a_bytes + 255) // 256) * 256
    comptime b0_off = ((a1_off + a_bytes + 255) // 256) * 256
    comptime b1_off = ((b0_off + b_bytes + 255) // 256) * 256
    comptime pack_bytes = BN * blocks_per_tile * BYTES_PER_BLOCK
    comptime pack0_off = ((b1_off + b_bytes + 255) // 256) * 256
    comptime pack1_off = ((pack0_off + pack_bytes + 255) // 256) * 256

    var smem = external_memory[
        Scalar[U8],
        address_space = AddressSpace.SHARED,
        alignment=256,
        name="moe_mxfp4_w2_dynamic_smem",
    ]()
    var smem_ptr = smem.address_space_cast[AddressSpace.SHARED]().mut_cast[
        True
    ]()

    var A_s0 = LayoutTensor[
        BF16,
        a_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=256,
    ](smem_ptr.bitcast[Scalar[BF16]]())
    var A_s1 = LayoutTensor[
        BF16,
        a_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=256,
    ]((smem_ptr + a1_off).bitcast[Scalar[BF16]]())

    var B_s0 = LayoutTensor[
        BF16,
        b_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=256,
    ]((smem_ptr + b0_off).bitcast[Scalar[BF16]]())
    var B_s1 = LayoutTensor[
        BF16,
        b_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=256,
    ]((smem_ptr + b1_off).bitcast[Scalar[BF16]]())
    var B_data0 = (smem_ptr + b0_off).bitcast[Scalar[BF16]]()
    var B_data1 = (smem_ptr + b1_off).bitcast[Scalar[BF16]]()
    var b_strides = B_s1.runtime_layout.stride.value

    # Packed MXFP4 staging buffers: [BN * (BK/32) blocks] * 16 bytes.
    var B_pack0 = smem_ptr + pack0_off
    var B_pack1 = smem_ptr + pack1_off

    var w_blocks_u8 = w_blocks_ptr.address_space_cast[
        AddressSpace.GLOBAL
    ]().bitcast[Scalar[U8]]()
    var B_pack0_u64 = (
        B_pack0.bitcast[Scalar[U64]]()
        .address_space_cast[AddressSpace.SHARED]()
        .mut_cast[True]()
    )
    var B_pack1_u64 = (
        B_pack1.bitcast[Scalar[U64]]()
        .address_space_cast[AddressSpace.SHARED]()
        .mut_cast[True]()
    )
    var B_pack0_u32 = (
        B_pack0.bitcast[Scalar[U32]]()
        .address_space_cast[AddressSpace.SHARED]()
        .mut_cast[True]()
    )
    var B_pack1_u32 = (
        B_pack1.bitcast[Scalar[U32]]()
        .address_space_cast[AddressSpace.SHARED]()
        .mut_cast[True]()
    )

    var w_blocks_stride0 = 0
    var w_blocks_stride1 = 0
    var w_blocks_stride2 = 0

    @parameter
    if USE_VALUE_SWIZZLE:
        var kbytes = Kblocks * BYTES_PER_BLOCK
        var m_pad = ((D + 15) // 16) * 16
        var k_pad = ((kbytes + 31) // 32) * 32
        var w_m2 = m_pad // 4
        var w_k2 = k_pad * 4
        w_blocks_stride0 = w_m2 * w_k2
        w_blocks_stride1 = w_k2
        w_blocks_stride2 = 1

    comptime num_m_mmas_total = BM // WGMMA_M
    comptime num_m_mmas = num_m_mmas_total // NUM_WARP_GROUPS
    comptime num_n_mmas = BN // WGMMA_N
    comptime c_frag_size = (WGMMA_M * WGMMA_N) // 128

    var c_reg_tile = (
        LayoutTensor[
            F32,
            Layout.row_major(num_m_mmas * num_n_mmas, c_frag_size),
            MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .fill(0.0)
    )

    var wgmma = TensorCoreAsync[
        F32,
        BF16,
        BF16,
        IndexList[3](WGMMA_M, WGMMA_N, WGMMA_K),
        b_swizzle=b_swizzle,
        transpose_b=True,
    ]()

    comptime a_chunks = BK // 4

    # Epilogue thread mapping for the register fragments.
    var lane = Int(thread_idx.x & 31)
    var lane_row = lane // 4
    var lane_col = lane - lane_row * 4

    comptime warp_rows = WGMMA_M // 4
    comptime row_iters = warp_rows // 8
    comptime col_iters = (WGMMA_N // 2) // 4

    for row0 in range(row_base, seg_end, row_stride):
        # Precompute original pair indices + gamma for the BM rows.
        for r in range(Int(thread_idx.x), BM, Int(block_dim.x)):
            var global_row = row0 + r
            if global_row < seg_end and global_row < P:
                var pair_idx = UInt32(token_expert_order_ptr[global_row])
                pair_idx_s[r] = pair_idx
                gamma_s[r] = gate_w_ptr[Int(pair_idx)]
            else:
                pair_idx_s[r] = UInt32(0)
                gamma_s[r] = 0.0
        barrier()

        _ = c_reg_tile.fill(0.0)

        # Preload first K tile into buffer 0.
        var kb0 = 0
        for idx in range(
            Int(thread_idx.x),
            BN * blocks_per_tile,
            Int(block_dim.x),
        ):
            var c = idx // blocks_per_tile
            var block_in_tile = idx - c * blocks_per_tile
            var col = n0 + c
            var kb = kb0 + block_in_tile

            if col < D and kb < Kblocks:

                @parameter
                if USE_VALUE_SWIZZLE:
                    var packed0, packed1 = _load_swizzled_block_u8x16(
                        w_blocks_ptr,
                        w_blocks_stride0,
                        w_blocks_stride1,
                        w_blocks_stride2,
                        expert_id,
                        col,
                        kb,
                    )
                    var base_u64 = idx * 2
                    B_pack0_u64.store[alignment=8](
                        base_u64 + 0, Scalar[U64](_u64_from_u8x8(packed0))
                    )
                    B_pack0_u64.store[alignment=8](
                        base_u64 + 1, Scalar[U64](_u64_from_u8x8(packed1))
                    )
                else:
                    var packed_base = (
                        ((expert_id * D + col) * Kblocks) + kb
                    ) * BYTES_PER_BLOCK
                    async_copy[16](
                        w_blocks_u8 + packed_base,
                        B_pack0 + idx * BYTES_PER_BLOCK,
                    )
            else:
                var base_u64 = idx * 2
                B_pack0_u64.store[alignment=8](base_u64 + 0, Scalar[U64](0))
                B_pack0_u64.store[alignment=8](base_u64 + 1, Scalar[U64](0))

        @parameter
        if not USE_VALUE_SWIZZLE:
            async_copy_commit_group()

        for idx in range(Int(thread_idx.x), BM * a_chunks, Int(block_dim.x)):
            var r = idx // a_chunks
            var chunk = idx - r * a_chunks
            var kk = chunk * 4
            var global_row = row0 + r
            var global_k = kk
            if global_row < seg_end and global_row < P and global_k + 3 < I:
                var p64 = rebind[UnsafePointer[UInt64, MutAnyOrigin]](
                    h_ptr + (global_row * I + global_k)
                )
                var v4 = bitcast[DType.bfloat16, 4](p64[0])
                A_s0[r, kk + 0] = v4[0]
                A_s0[r, kk + 1] = v4[1]
                A_s0[r, kk + 2] = v4[2]
                A_s0[r, kk + 3] = v4[3]
            else:

                @parameter
                for i in range(4):
                    var k_i = global_k + i
                    if global_row < seg_end and global_row < P and k_i < I:
                        A_s0[r, kk + i] = h_ptr[global_row * I + k_i]
                    else:
                        A_s0[r, kk + i] = 0

        @parameter
        if USE_VALUE_SWIZZLE:
            barrier()
        else:
            async_copy_wait_all()

        @parameter
        if USE_VALUE_SWIZZLE and WRITER_MAP == 2:
            for gidx in range(
                Int(thread_idx.x),
                BN * blocks_per_tile * 4,
                Int(block_dim.x),
            ):
                var block_idx = gidx >> 2
                var g = gidx & 3
                var c = block_idx // blocks_per_tile
                var block_in_tile = block_idx - c * blocks_per_tile
                var col = n0 + c
                var kb = kb0 + block_in_tile
                var scale_exp = UInt8(0)
                if col < D and kb < Kblocks:
                    scale_exp = w_scales_ptr[
                        ((expert_id * D + col) * Kblocks) + kb
                    ]
                var scale = e8m0_to_bf16_bits(scale_exp)

                var base_u32 = block_idx * 4
                var p = UInt32(B_pack0_u32.load[alignment=4](base_u32 + g))
                var outv = decode_mxfp4_packbits_u32_to_8xbf16_scaled(
                    p, scale
                )
                var k_base = block_in_tile * VALUES_PER_BLOCK + g * 8
                var ridx = B_s0.idx_list_t[2](fill=0)
                ridx[0] = c
                ridx[1] = k_base
                var lin = B_s0._get_offset(b_strides, ridx)
                var swz = _apply_swizzle(b_smem_swizzle, lin)
                B_data0.store[width=8, alignment=16](swz, outv)
        else:
            for idx in range(
                Int(thread_idx.x),
                BN * blocks_per_tile,
                Int(block_dim.x),
            ):
                var c = idx // blocks_per_tile
                var block_in_tile = idx - c * blocks_per_tile
                var col = n0 + c
                var kb = kb0 + block_in_tile
                var scale_exp = UInt8(0)
                if col < D and kb < Kblocks:
                    scale_exp = w_scales_ptr[
                        ((expert_id * D + col) * Kblocks) + kb
                    ]
                var scale = e8m0_to_bf16_bits(scale_exp)

                var base_u64 = idx * 2
                var packed0 = bitcast[DType.uint8, 8](
                    UInt64(B_pack0_u64.load[alignment=8](base_u64 + 0))
                )
                var packed1 = bitcast[DType.uint8, 8](
                    UInt64(B_pack0_u64.load[alignment=8](base_u64 + 1))
                )

                @parameter
                if USE_VALUE_SWIZZLE:
                    @parameter
                    if WRITER_MAP == 0:

                        @parameter
                        for group in range(4):
                            var g = group
                            var byte_base = (g & 1) * 4
                            var b0 = packed0[byte_base + 0]
                            var b1 = packed0[byte_base + 1]
                            var b2 = packed0[byte_base + 2]
                            var b3 = packed0[byte_base + 3]
                            if g >= 2:
                                b0 = packed1[byte_base + 0]
                                b1 = packed1[byte_base + 1]
                                b2 = packed1[byte_base + 2]
                                b3 = packed1[byte_base + 3]
                            var p = _u32_from_u8x4(b0, b1, b2, b3)
                            var outv = decode_mxfp4_packbits_u32_to_8xbf16_scaled(
                                p, scale
                            )
                            var k_base = (
                                block_in_tile * VALUES_PER_BLOCK + g * 8
                            )
                            var ridx = B_s0.idx_list_t[2](fill=0)
                            ridx[0] = c
                            ridx[1] = k_base
                            var lin = B_s0._get_offset(b_strides, ridx)
                            var swz = _apply_swizzle(b_smem_swizzle, lin)
                            B_data0.store[width=8, alignment=16](swz, outv)
                    elif WRITER_MAP == 1:
                        var phase = Int(thread_idx.x) & 1
                        if phase == 0:
                            @parameter
                            for group in range(4):
                                var g = group
                                var byte_base = (g & 1) * 4
                                var b0 = packed0[byte_base + 0]
                                var b1 = packed0[byte_base + 1]
                                var b2 = packed0[byte_base + 2]
                                var b3 = packed0[byte_base + 3]
                                if g >= 2:
                                    b0 = packed1[byte_base + 0]
                                    b1 = packed1[byte_base + 1]
                                    b2 = packed1[byte_base + 2]
                                    b3 = packed1[byte_base + 3]
                                var p = _u32_from_u8x4(b0, b1, b2, b3)
                                var outv = (
                                    decode_mxfp4_packbits_u32_to_8xbf16_scaled(
                                        p, scale
                                    )
                                )
                                var k_base = (
                                    block_in_tile * VALUES_PER_BLOCK + g * 8
                                )
                                var ridx = B_s0.idx_list_t[2](fill=0)
                                ridx[0] = c
                                ridx[1] = k_base
                                var lin = B_s0._get_offset(b_strides, ridx)
                                var swz = _apply_swizzle(b_smem_swizzle, lin)
                                B_data0.store[width=8, alignment=16](
                                    swz, outv
                                )
                        else:
                            @parameter
                            for group in range(4):
                                var g = group
                                if g == 0:
                                    g = 1
                                elif g == 1:
                                    g = 0
                                elif g == 2:
                                    g = 3
                                else:
                                    g = 2
                                var byte_base = (g & 1) * 4
                                var b0 = packed0[byte_base + 0]
                                var b1 = packed0[byte_base + 1]
                                var b2 = packed0[byte_base + 2]
                                var b3 = packed0[byte_base + 3]
                                if g >= 2:
                                    b0 = packed1[byte_base + 0]
                                    b1 = packed1[byte_base + 1]
                                    b2 = packed1[byte_base + 2]
                                    b3 = packed1[byte_base + 3]
                                var p = _u32_from_u8x4(b0, b1, b2, b3)
                                var outv = (
                                    decode_mxfp4_packbits_u32_to_8xbf16_scaled(
                                        p, scale
                                    )
                                )
                                var k_base = (
                                    block_in_tile * VALUES_PER_BLOCK + g * 8
                                )
                                var ridx = B_s0.idx_list_t[2](fill=0)
                                ridx[0] = c
                                ridx[1] = k_base
                                var lin = B_s0._get_offset(b_strides, ridx)
                                var swz = _apply_swizzle(b_smem_swizzle, lin)
                                B_data0.store[width=8, alignment=16](
                                    swz, outv
                                )
                    else:
                        var lane_phase = Int(thread_idx.x) & 3
                        @parameter
                        for step in range(4):
                            var g = (step + lane_phase) & 3
                            var byte_base = (g & 1) * 4
                            var b0 = packed0[byte_base + 0]
                            var b1 = packed0[byte_base + 1]
                            var b2 = packed0[byte_base + 2]
                            var b3 = packed0[byte_base + 3]
                            if g >= 2:
                                b0 = packed1[byte_base + 0]
                                b1 = packed1[byte_base + 1]
                                b2 = packed1[byte_base + 2]
                                b3 = packed1[byte_base + 3]
                            var p = _u32_from_u8x4(b0, b1, b2, b3)
                            var outv = decode_mxfp4_packbits_u32_to_8xbf16_scaled(
                                p, scale
                            )
                            var k_base = (
                                block_in_tile * VALUES_PER_BLOCK + g * 8
                            )
                            var ridx = B_s0.idx_list_t[2](fill=0)
                            ridx[0] = c
                            ridx[1] = k_base
                            var lin = B_s0._get_offset(b_strides, ridx)
                            var swz = _apply_swizzle(b_smem_swizzle, lin)
                            B_data0.store[width=8, alignment=16](swz, outv)
                else:

                    @parameter
                    for byte_in_block in range(8):
                        var r0 = (
                            block_in_tile * VALUES_PER_BLOCK + byte_in_block * 2
                        )
                        var v2 = decode_mxfp4_byte_to_2xbf16_scaled(
                            packed0[byte_in_block], scale
                        )
                        B_s0[c, r0 + 0] = v2[0]
                        B_s0[c, r0 + 1] = v2[1]

                    @parameter
                    for byte_in_block in range(8):
                        var byte = byte_in_block + 8
                        var r0 = block_in_tile * VALUES_PER_BLOCK + byte * 2
                        var v2 = decode_mxfp4_byte_to_2xbf16_scaled(
                            packed1[byte_in_block], scale
                        )
                        B_s0[c, r0 + 0] = v2[0]
                        B_s0[c, r0 + 1] = v2[1]
        barrier()

        var num_k_tiles = ceildiv(I, BK)
        for k_tile in range(num_k_tiles):
            var use_buf0 = (k_tile & 1) == 0

            warpgroup_fence(c_reg_tile)
            wgmma.arrive()
            if use_buf0:
                wgmma.wgmma[num_warp_groups=NUM_WARP_GROUPS](
                    a_smem_tile=A_s0,
                    b_smem_tile=B_s0,
                    c_reg_tile=c_reg_tile,
                    wg_idx=wg_idx,
                )
            else:
                wgmma.wgmma[num_warp_groups=NUM_WARP_GROUPS](
                    a_smem_tile=A_s1,
                    b_smem_tile=B_s1,
                    c_reg_tile=c_reg_tile,
                    wg_idx=wg_idx,
                )
            wgmma.commit_group()
            warpgroup_fence(c_reg_tile)

            if k_tile + 1 < num_k_tiles:
                wgmma.wait_group[1]()

                var k0_next = (k_tile + 1) * BK
                var kb0_next = k0_next // VALUES_PER_BLOCK

                if use_buf0:
                    # Load next tile into buffer 1.
                    for idx in range(
                        Int(thread_idx.x),
                        BN * blocks_per_tile,
                        Int(block_dim.x),
                    ):
                        var c = idx // blocks_per_tile
                        var block_in_tile = idx - c * blocks_per_tile
                        var col = n0 + c
                        var kb = kb0_next + block_in_tile

                        if col < D and kb < Kblocks:

                            @parameter
                            if USE_VALUE_SWIZZLE:
                                var packed0, packed1 = (
                                    _load_swizzled_block_u8x16(
                                        w_blocks_ptr,
                                        w_blocks_stride0,
                                        w_blocks_stride1,
                                        w_blocks_stride2,
                                        expert_id,
                                        col,
                                        kb,
                                    )
                                )
                                var base_u64 = idx * 2
                                B_pack1_u64.store[alignment=8](
                                    base_u64 + 0,
                                    Scalar[U64](_u64_from_u8x8(packed0)),
                                )
                                B_pack1_u64.store[alignment=8](
                                    base_u64 + 1,
                                    Scalar[U64](_u64_from_u8x8(packed1)),
                                )
                            else:
                                var packed_base = (
                                    (expert_id * D + col) * Kblocks + kb
                                ) * BYTES_PER_BLOCK
                                async_copy[16](
                                    w_blocks_u8 + packed_base,
                                    B_pack1 + idx * BYTES_PER_BLOCK,
                                )
                        else:
                            var base_u64 = idx * 2
                            B_pack1_u64.store[alignment=8](
                                base_u64 + 0, Scalar[U64](0)
                            )
                            B_pack1_u64.store[alignment=8](
                                base_u64 + 1, Scalar[U64](0)
                            )

                    @parameter
                    if not USE_VALUE_SWIZZLE:
                        async_copy_commit_group()

                    for idx in range(
                        Int(thread_idx.x), BM * a_chunks, Int(block_dim.x)
                    ):
                        var r = idx // a_chunks
                        var chunk = idx - r * a_chunks
                        var kk = chunk * 4
                        var global_row = row0 + r
                        var global_k = k0_next + kk
                        if (
                            global_row < seg_end
                            and global_row < P
                            and global_k + 3 < I
                        ):
                            var p64 = rebind[
                                UnsafePointer[UInt64, MutAnyOrigin]
                            ](h_ptr + (global_row * I + global_k))
                            var v4 = bitcast[DType.bfloat16, 4](p64[0])
                            A_s1[r, kk + 0] = v4[0]
                            A_s1[r, kk + 1] = v4[1]
                            A_s1[r, kk + 2] = v4[2]
                            A_s1[r, kk + 3] = v4[3]
                        else:

                            @parameter
                            for i in range(4):
                                var k_i = global_k + i
                                if (
                                    global_row < seg_end
                                    and global_row < P
                                    and k_i < I
                                ):
                                    A_s1[r, kk + i] = h_ptr[
                                        global_row * I + k_i
                                    ]
                                else:
                                    A_s1[r, kk + i] = 0

                    @parameter
                    if USE_VALUE_SWIZZLE:
                        barrier()
                    else:
                        async_copy_wait_all()

                    @parameter
                    if USE_VALUE_SWIZZLE and WRITER_MAP == 2:
                        for gidx in range(
                            Int(thread_idx.x),
                            BN * blocks_per_tile * 4,
                            Int(block_dim.x),
                        ):
                            var block_idx = gidx >> 2
                            var g = gidx & 3
                            var c = block_idx // blocks_per_tile
                            var block_in_tile = block_idx - c * blocks_per_tile
                            var col = n0 + c
                            var kb = kb0_next + block_in_tile
                            var scale_exp = UInt8(0)
                            if col < D and kb < Kblocks:
                                scale_exp = w_scales_ptr[
                                    ((expert_id * D + col) * Kblocks) + kb
                                ]
                            var scale = e8m0_to_bf16_bits(scale_exp)

                            var base_u32 = block_idx * 4
                            var p = UInt32(
                                B_pack1_u32.load[alignment=4](base_u32 + g)
                            )
                            var outv = (
                                decode_mxfp4_packbits_u32_to_8xbf16_scaled(
                                    p, scale
                                )
                            )
                            var k_base = (
                                block_in_tile * VALUES_PER_BLOCK + g * 8
                            )
                            var ridx = B_s1.idx_list_t[2](fill=0)
                            ridx[0] = c
                            ridx[1] = k_base
                            var lin = B_s1._get_offset(b_strides, ridx)
                            var swz = _apply_swizzle(b_smem_swizzle, lin)
                            B_data1.store[width=8, alignment=16](swz, outv)
                    else:
                        for idx in range(
                            Int(thread_idx.x),
                            BN * blocks_per_tile,
                            Int(block_dim.x),
                        ):
                            var c = idx // blocks_per_tile
                            var block_in_tile = idx - c * blocks_per_tile
                            var col = n0 + c
                            var kb = kb0_next + block_in_tile
                            var scale_exp = UInt8(0)
                            if col < D and kb < Kblocks:
                                scale_exp = w_scales_ptr[
                                    ((expert_id * D + col) * Kblocks) + kb
                                ]
                            var scale = e8m0_to_bf16_bits(scale_exp)

                            var base_u64 = idx * 2
                            var packed0 = bitcast[DType.uint8, 8](
                                UInt64(
                                    B_pack1_u64.load[alignment=8](base_u64 + 0)
                                )
                            )
                            var packed1 = bitcast[DType.uint8, 8](
                                UInt64(
                                    B_pack1_u64.load[alignment=8](base_u64 + 1)
                                )
                            )

                            @parameter
                            if USE_VALUE_SWIZZLE:
                                @parameter
                                if WRITER_MAP == 0:

                                    @parameter
                                    for group in range(4):
                                        var g = group
                                        var byte_base = (g & 1) * 4
                                        var b0 = packed0[byte_base + 0]
                                        var b1 = packed0[byte_base + 1]
                                        var b2 = packed0[byte_base + 2]
                                        var b3 = packed0[byte_base + 3]
                                        if g >= 2:
                                            b0 = packed1[byte_base + 0]
                                            b1 = packed1[byte_base + 1]
                                            b2 = packed1[byte_base + 2]
                                            b3 = packed1[byte_base + 3]
                                        var p = _u32_from_u8x4(b0, b1, b2, b3)
                                        var outv = (
                                            decode_mxfp4_packbits_u32_to_8xbf16_scaled(
                                                p, scale
                                            )
                                        )
                                        var k_base = (
                                            block_in_tile * VALUES_PER_BLOCK + g * 8
                                        )
                                        var ridx = B_s1.idx_list_t[2](fill=0)
                                        ridx[0] = c
                                        ridx[1] = k_base
                                        var lin = B_s1._get_offset(b_strides, ridx)
                                        var swz = _apply_swizzle(b_smem_swizzle, lin)
                                        B_data1.store[width=8, alignment=16](
                                            swz, outv
                                        )
                                elif WRITER_MAP == 1:
                                    var phase = Int(thread_idx.x) & 1
                                    if phase == 0:
                                        @parameter
                                        for group in range(4):
                                            var g = group
                                            var byte_base = (g & 1) * 4
                                            var b0 = packed0[byte_base + 0]
                                            var b1 = packed0[byte_base + 1]
                                            var b2 = packed0[byte_base + 2]
                                            var b3 = packed0[byte_base + 3]
                                            if g >= 2:
                                                b0 = packed1[byte_base + 0]
                                                b1 = packed1[byte_base + 1]
                                                b2 = packed1[byte_base + 2]
                                                b3 = packed1[byte_base + 3]
                                            var p = _u32_from_u8x4(b0, b1, b2, b3)
                                            var outv = (
                                                decode_mxfp4_packbits_u32_to_8xbf16_scaled(
                                                    p, scale
                                                )
                                            )
                                            var k_base = (
                                                block_in_tile * VALUES_PER_BLOCK
                                                + g * 8
                                            )
                                            var ridx = B_s1.idx_list_t[2](fill=0)
                                            ridx[0] = c
                                            ridx[1] = k_base
                                            var lin = B_s1._get_offset(
                                                b_strides, ridx
                                            )
                                            var swz = _apply_swizzle(
                                                b_smem_swizzle, lin
                                            )
                                            B_data1.store[width=8, alignment=16](
                                                swz, outv
                                            )
                                    else:
                                        @parameter
                                        for group in range(4):
                                            var g = group
                                            if g == 0:
                                                g = 1
                                            elif g == 1:
                                                g = 0
                                            elif g == 2:
                                                g = 3
                                            else:
                                                g = 2
                                            var byte_base = (g & 1) * 4
                                            var b0 = packed0[byte_base + 0]
                                            var b1 = packed0[byte_base + 1]
                                            var b2 = packed0[byte_base + 2]
                                            var b3 = packed0[byte_base + 3]
                                            if g >= 2:
                                                b0 = packed1[byte_base + 0]
                                                b1 = packed1[byte_base + 1]
                                                b2 = packed1[byte_base + 2]
                                                b3 = packed1[byte_base + 3]
                                            var p = _u32_from_u8x4(b0, b1, b2, b3)
                                            var outv = (
                                                decode_mxfp4_packbits_u32_to_8xbf16_scaled(
                                                    p, scale
                                                )
                                            )
                                            var k_base = (
                                                block_in_tile * VALUES_PER_BLOCK
                                                + g * 8
                                            )
                                            var ridx = B_s1.idx_list_t[2](fill=0)
                                            ridx[0] = c
                                            ridx[1] = k_base
                                            var lin = B_s1._get_offset(
                                                b_strides, ridx
                                            )
                                            var swz = _apply_swizzle(
                                                b_smem_swizzle, lin
                                            )
                                            B_data1.store[width=8, alignment=16](
                                                swz, outv
                                            )
                                else:
                                    var lane_phase = Int(thread_idx.x) & 3
                                    @parameter
                                    for step in range(4):
                                        var g = (step + lane_phase) & 3
                                        var byte_base = (g & 1) * 4
                                        var b0 = packed0[byte_base + 0]
                                        var b1 = packed0[byte_base + 1]
                                        var b2 = packed0[byte_base + 2]
                                        var b3 = packed0[byte_base + 3]
                                        if g >= 2:
                                            b0 = packed1[byte_base + 0]
                                            b1 = packed1[byte_base + 1]
                                            b2 = packed1[byte_base + 2]
                                            b3 = packed1[byte_base + 3]
                                        var p = _u32_from_u8x4(b0, b1, b2, b3)
                                        var outv = (
                                            decode_mxfp4_packbits_u32_to_8xbf16_scaled(
                                                p, scale
                                            )
                                        )
                                        var k_base = (
                                            block_in_tile * VALUES_PER_BLOCK + g * 8
                                        )
                                        var ridx = B_s1.idx_list_t[2](fill=0)
                                        ridx[0] = c
                                        ridx[1] = k_base
                                        var lin = B_s1._get_offset(b_strides, ridx)
                                        var swz = _apply_swizzle(b_smem_swizzle, lin)
                                        B_data1.store[width=8, alignment=16](
                                            swz, outv
                                        )
                            else:

                                @parameter
                                for byte_in_block in range(8):
                                    var r0 = (
                                        block_in_tile * VALUES_PER_BLOCK
                                        + byte_in_block * 2
                                    )
                                    var v2 = decode_mxfp4_byte_to_2xbf16_scaled(
                                        packed0[byte_in_block], scale
                                    )
                                    B_s1[c, r0 + 0] = v2[0]
                                    B_s1[c, r0 + 1] = v2[1]

                                @parameter
                                for byte_in_block in range(8):
                                    var byte = byte_in_block + 8
                                    var r0 = (
                                        block_in_tile * VALUES_PER_BLOCK + byte * 2
                                    )
                                    var v2 = decode_mxfp4_byte_to_2xbf16_scaled(
                                        packed1[byte_in_block], scale
                                    )
                                    B_s1[c, r0 + 0] = v2[0]
                                    B_s1[c, r0 + 1] = v2[1]
                else:
                    # Load next tile into buffer 0.
                    for idx in range(
                        Int(thread_idx.x),
                        BN * blocks_per_tile,
                        Int(block_dim.x),
                    ):
                        var c = idx // blocks_per_tile
                        var block_in_tile = idx - c * blocks_per_tile
                        var col = n0 + c
                        var kb = kb0_next + block_in_tile

                        if col < D and kb < Kblocks:

                            @parameter
                            if USE_VALUE_SWIZZLE:
                                var packed0, packed1 = (
                                    _load_swizzled_block_u8x16(
                                        w_blocks_ptr,
                                        w_blocks_stride0,
                                        w_blocks_stride1,
                                        w_blocks_stride2,
                                        expert_id,
                                        col,
                                        kb,
                                    )
                                )
                                var base_u64 = idx * 2
                                B_pack0_u64.store[alignment=8](
                                    base_u64 + 0,
                                    Scalar[U64](_u64_from_u8x8(packed0)),
                                )
                                B_pack0_u64.store[alignment=8](
                                    base_u64 + 1,
                                    Scalar[U64](_u64_from_u8x8(packed1)),
                                )
                            else:
                                var packed_base = (
                                    (expert_id * D + col) * Kblocks + kb
                                ) * BYTES_PER_BLOCK
                                async_copy[16](
                                    w_blocks_u8 + packed_base,
                                    B_pack0 + idx * BYTES_PER_BLOCK,
                                )
                        else:
                            var base_u64 = idx * 2
                            B_pack0_u64.store[alignment=8](
                                base_u64 + 0, Scalar[U64](0)
                            )
                            B_pack0_u64.store[alignment=8](
                                base_u64 + 1, Scalar[U64](0)
                            )

                    @parameter
                    if not USE_VALUE_SWIZZLE:
                        async_copy_commit_group()

                    for idx in range(
                        Int(thread_idx.x), BM * a_chunks, Int(block_dim.x)
                    ):
                        var r = idx // a_chunks
                        var chunk = idx - r * a_chunks
                        var kk = chunk * 4
                        var global_row = row0 + r
                        var global_k = k0_next + kk
                        if (
                            global_row < seg_end
                            and global_row < P
                            and global_k + 3 < I
                        ):
                            var p64 = rebind[
                                UnsafePointer[UInt64, MutAnyOrigin]
                            ](h_ptr + (global_row * I + global_k))
                            var v4 = bitcast[DType.bfloat16, 4](p64[0])
                            A_s0[r, kk + 0] = v4[0]
                            A_s0[r, kk + 1] = v4[1]
                            A_s0[r, kk + 2] = v4[2]
                            A_s0[r, kk + 3] = v4[3]
                        else:

                            @parameter
                            for i in range(4):
                                var k_i = global_k + i
                                if (
                                    global_row < seg_end
                                    and global_row < P
                                    and k_i < I
                                ):
                                    A_s0[r, kk + i] = h_ptr[
                                        global_row * I + k_i
                                    ]
                                else:
                                    A_s0[r, kk + i] = 0

                    @parameter
                    if USE_VALUE_SWIZZLE:
                        barrier()
                    else:
                        async_copy_wait_all()

                    @parameter
                    if USE_VALUE_SWIZZLE and WRITER_MAP == 2:
                        for gidx in range(
                            Int(thread_idx.x),
                            BN * blocks_per_tile * 4,
                            Int(block_dim.x),
                        ):
                            var block_idx = gidx >> 2
                            var g = gidx & 3
                            var c = block_idx // blocks_per_tile
                            var block_in_tile = block_idx - c * blocks_per_tile
                            var col = n0 + c
                            var kb = kb0_next + block_in_tile
                            var scale_exp = UInt8(0)
                            if col < D and kb < Kblocks:
                                scale_exp = w_scales_ptr[
                                    ((expert_id * D + col) * Kblocks) + kb
                                ]
                            var scale = e8m0_to_bf16_bits(scale_exp)

                            var base_u32 = block_idx * 4
                            var p = UInt32(
                                B_pack0_u32.load[alignment=4](base_u32 + g)
                            )
                            var outv = (
                                decode_mxfp4_packbits_u32_to_8xbf16_scaled(
                                    p, scale
                                )
                            )
                            var k_base = (
                                block_in_tile * VALUES_PER_BLOCK + g * 8
                            )
                            var ridx = B_s0.idx_list_t[2](fill=0)
                            ridx[0] = c
                            ridx[1] = k_base
                            var lin = B_s0._get_offset(b_strides, ridx)
                            var swz = _apply_swizzle(b_smem_swizzle, lin)
                            B_data0.store[width=8, alignment=16](swz, outv)
                    else:
                        for idx in range(
                            Int(thread_idx.x),
                            BN * blocks_per_tile,
                            Int(block_dim.x),
                        ):
                            var c = idx // blocks_per_tile
                            var block_in_tile = idx - c * blocks_per_tile
                            var col = n0 + c
                            var kb = kb0_next + block_in_tile
                            var scale_exp = UInt8(0)
                            if col < D and kb < Kblocks:
                                scale_exp = w_scales_ptr[
                                    ((expert_id * D + col) * Kblocks) + kb
                                ]
                            var scale = e8m0_to_bf16_bits(scale_exp)

                            var base_u64 = idx * 2
                            var packed0 = bitcast[DType.uint8, 8](
                                UInt64(
                                    B_pack0_u64.load[alignment=8](base_u64 + 0)
                                )
                            )
                            var packed1 = bitcast[DType.uint8, 8](
                                UInt64(
                                    B_pack0_u64.load[alignment=8](base_u64 + 1)
                                )
                            )

                            @parameter
                            if USE_VALUE_SWIZZLE:
                                @parameter
                                if WRITER_MAP == 0:

                                    @parameter
                                    for group in range(4):
                                        var g = group
                                        var byte_base = (g & 1) * 4
                                        var b0 = packed0[byte_base + 0]
                                        var b1 = packed0[byte_base + 1]
                                        var b2 = packed0[byte_base + 2]
                                        var b3 = packed0[byte_base + 3]
                                        if g >= 2:
                                            b0 = packed1[byte_base + 0]
                                            b1 = packed1[byte_base + 1]
                                            b2 = packed1[byte_base + 2]
                                            b3 = packed1[byte_base + 3]
                                        var p = _u32_from_u8x4(b0, b1, b2, b3)
                                        var outv = (
                                            decode_mxfp4_packbits_u32_to_8xbf16_scaled(
                                                p, scale
                                            )
                                        )
                                        var k_base = (
                                            block_in_tile * VALUES_PER_BLOCK + g * 8
                                        )
                                        var ridx = B_s0.idx_list_t[2](fill=0)
                                        ridx[0] = c
                                        ridx[1] = k_base
                                        var lin = B_s0._get_offset(b_strides, ridx)
                                        var swz = _apply_swizzle(b_smem_swizzle, lin)
                                        B_data0.store[width=8, alignment=16](
                                            swz, outv
                                        )
                                elif WRITER_MAP == 1:
                                    var phase = Int(thread_idx.x) & 1
                                    if phase == 0:
                                        @parameter
                                        for group in range(4):
                                            var g = group
                                            var byte_base = (g & 1) * 4
                                            var b0 = packed0[byte_base + 0]
                                            var b1 = packed0[byte_base + 1]
                                            var b2 = packed0[byte_base + 2]
                                            var b3 = packed0[byte_base + 3]
                                            if g >= 2:
                                                b0 = packed1[byte_base + 0]
                                                b1 = packed1[byte_base + 1]
                                                b2 = packed1[byte_base + 2]
                                                b3 = packed1[byte_base + 3]
                                            var p = _u32_from_u8x4(b0, b1, b2, b3)
                                            var outv = (
                                                decode_mxfp4_packbits_u32_to_8xbf16_scaled(
                                                    p, scale
                                                )
                                            )
                                            var k_base = (
                                                block_in_tile * VALUES_PER_BLOCK
                                                + g * 8
                                            )
                                            var ridx = B_s0.idx_list_t[2](fill=0)
                                            ridx[0] = c
                                            ridx[1] = k_base
                                            var lin = B_s0._get_offset(
                                                b_strides, ridx
                                            )
                                            var swz = _apply_swizzle(
                                                b_smem_swizzle, lin
                                            )
                                            B_data0.store[width=8, alignment=16](
                                                swz, outv
                                            )
                                    else:
                                        @parameter
                                        for group in range(4):
                                            var g = group
                                            if g == 0:
                                                g = 1
                                            elif g == 1:
                                                g = 0
                                            elif g == 2:
                                                g = 3
                                            else:
                                                g = 2
                                            var byte_base = (g & 1) * 4
                                            var b0 = packed0[byte_base + 0]
                                            var b1 = packed0[byte_base + 1]
                                            var b2 = packed0[byte_base + 2]
                                            var b3 = packed0[byte_base + 3]
                                            if g >= 2:
                                                b0 = packed1[byte_base + 0]
                                                b1 = packed1[byte_base + 1]
                                                b2 = packed1[byte_base + 2]
                                                b3 = packed1[byte_base + 3]
                                            var p = _u32_from_u8x4(b0, b1, b2, b3)
                                            var outv = (
                                                decode_mxfp4_packbits_u32_to_8xbf16_scaled(
                                                    p, scale
                                                )
                                            )
                                            var k_base = (
                                                block_in_tile * VALUES_PER_BLOCK
                                                + g * 8
                                            )
                                            var ridx = B_s0.idx_list_t[2](fill=0)
                                            ridx[0] = c
                                            ridx[1] = k_base
                                            var lin = B_s0._get_offset(
                                                b_strides, ridx
                                            )
                                            var swz = _apply_swizzle(
                                                b_smem_swizzle, lin
                                            )
                                            B_data0.store[width=8, alignment=16](
                                                swz, outv
                                            )
                                else:
                                    var lane_phase = Int(thread_idx.x) & 3
                                    @parameter
                                    for step in range(4):
                                        var g = (step + lane_phase) & 3
                                        var byte_base = (g & 1) * 4
                                        var b0 = packed0[byte_base + 0]
                                        var b1 = packed0[byte_base + 1]
                                        var b2 = packed0[byte_base + 2]
                                        var b3 = packed0[byte_base + 3]
                                        if g >= 2:
                                            b0 = packed1[byte_base + 0]
                                            b1 = packed1[byte_base + 1]
                                            b2 = packed1[byte_base + 2]
                                            b3 = packed1[byte_base + 3]
                                        var p = _u32_from_u8x4(b0, b1, b2, b3)
                                        var outv = (
                                            decode_mxfp4_packbits_u32_to_8xbf16_scaled(
                                                p, scale
                                            )
                                        )
                                        var k_base = (
                                            block_in_tile * VALUES_PER_BLOCK + g * 8
                                        )
                                        var ridx = B_s0.idx_list_t[2](fill=0)
                                        ridx[0] = c
                                        ridx[1] = k_base
                                        var lin = B_s0._get_offset(b_strides, ridx)
                                        var swz = _apply_swizzle(b_smem_swizzle, lin)
                                        B_data0.store[width=8, alignment=16](
                                            swz, outv
                                        )
                            else:

                                @parameter
                                for byte_in_block in range(8):
                                    var r0 = (
                                        block_in_tile * VALUES_PER_BLOCK
                                        + byte_in_block * 2
                                    )
                                    var v2 = decode_mxfp4_byte_to_2xbf16_scaled(
                                        packed0[byte_in_block], scale
                                    )
                                    B_s0[c, r0 + 0] = v2[0]
                                    B_s0[c, r0 + 1] = v2[1]

                                @parameter
                                for byte_in_block in range(8):
                                    var byte = byte_in_block + 8
                                    var r0 = (
                                        block_in_tile * VALUES_PER_BLOCK + byte * 2
                                    )
                                    var v2 = decode_mxfp4_byte_to_2xbf16_scaled(
                                        packed1[byte_in_block], scale
                                    )
                                    B_s0[c, r0 + 0] = v2[0]
                                    B_s0[c, r0 + 1] = v2[1]

                barrier()

        wgmma.wait_group()

        # Epilogue: bias + gamma scaling.
        #
        # - Scatter mode (WRITE_PAIRS=False): atomic scatter-add into Y[token, n]
        # - Pair-buffer mode (WRITE_PAIRS=True): write one row per pair into Y_pairs[pair_idx, n]
        #   for a later TOPK reduction kernel.
        @parameter
        for m_mma in range(num_m_mmas):

            @parameter
            for n_mma in range(num_n_mmas):
                comptime mma_id = n_mma * num_m_mmas + m_mma
                var c_frag = c_reg_tile.tile[1, c_frag_size](mma_id, 0)
                var c_pairs = c_frag.vectorize[1, 2]()

                @parameter
                for r_it in range(row_iters):
                    var row_in_warp = lane_row + r_it * 8
                    var row_in_cta = (
                        wg_idx * (num_m_mmas * WGMMA_M)
                        + m_mma * WGMMA_M
                        + warp_in_wg * warp_rows
                        + row_in_warp
                    )
                    var global_row = row0 + row_in_cta
                    if global_row >= seg_end or global_row >= P:
                        continue

                    var pair_id = Int(pair_idx_s[row_in_cta][0])
                    var gamma = gamma_s[row_in_cta][0].cast[F32]()
                    var tok = 0

                    @parameter
                    if not WRITE_PAIRS:
                        tok = pair_id // TOPK
                        if tok >= T:
                            continue

                    @parameter
                    for c_it in range(col_iters):
                        var col_pair = lane_col + c_it * 4
                        var col0 = n0 + n_mma * WGMMA_N + col_pair * 2
                        var col1 = col0 + 1
                        var frag_idx = c_it * row_iters + r_it
                        var v2 = c_pairs[0, frag_idx]

                        if col0 < D:
                            var v0 = v2[0] + b_ptr[expert_id * D + col0]
                            v0 *= gamma

                            @parameter
                            if WRITE_PAIRS:

                                @parameter
                                if PAIR_OUT_BF16:
                                    var y_bf16_ptr = rebind[
                                        UnsafePointer[BFloat16, MutAnyOrigin]
                                    ](y_ptr)
                                    y_bf16_ptr.store(
                                        pair_id * D + col0, v0.cast[BF16]()
                                    )
                                else:
                                    y_ptr.store(pair_id * D + col0, v0)
                            else:
                                _ = Atomic.fetch_add(
                                    y_ptr + (tok * D + col0), v0
                                )

                        if col1 < D:
                            var v1 = v2[1] + b_ptr[expert_id * D + col1]
                            v1 *= gamma

                            @parameter
                            if WRITE_PAIRS:

                                @parameter
                                if PAIR_OUT_BF16:
                                    var y_bf16_ptr = rebind[
                                        UnsafePointer[BFloat16, MutAnyOrigin]
                                    ](y_ptr)
                                    y_bf16_ptr.store(
                                        pair_id * D + col1, v1.cast[BF16]()
                                    )
                                else:
                                    y_ptr.store(pair_id * D + col1, v1)
                            else:
                                _ = Atomic.fetch_add(
                                    y_ptr + (tok * D + col1), v1
                                )

        # Synchronize before reusing the shared routing buffers in the next loop iteration.
        barrier()



@parameter
fn moe_topk_reduce_pairs[
    BN: Int = 256,
](
    y_pairs_ptr: UnsafePointer[Float32, MutAnyOrigin],
    P: Int,
    y_ptr: UnsafePointer[Float32, MutAnyOrigin],
    T: Int,
    D: Int,
):
    var tok = Int(block_idx.y)
    if tok >= T:
        return

    var pair0 = tok * TOPK
    if pair0 + (TOPK - 1) >= P:
        return

    var col = Int(block_idx.x) * BN + Int(thread_idx.x)
    if col >= D:
        return

    var base = pair0 * D + col
    var s = y_pairs_ptr[base]
    s += y_pairs_ptr[base + D]
    s += y_pairs_ptr[base + 2 * D]
    s += y_pairs_ptr[base + 3 * D]
    y_ptr.store(tok * D + col, s)



@parameter
fn moe_topk_reduce_pairs_bf16[
    BN: Int = 256,
](
    y_pairs_ptr: UnsafePointer[BFloat16, MutAnyOrigin],
    P: Int,
    y_ptr: UnsafePointer[BFloat16, MutAnyOrigin],
    T: Int,
    D: Int,
):
    var tok = Int(block_idx.y)
    if tok >= T:
        return

    var pair0 = tok * TOPK
    if pair0 + (TOPK - 1) >= P:
        return

    var col = Int(block_idx.x) * BN + Int(thread_idx.x)
    if col >= D:
        return

    var base = pair0 * D + col
    var s = y_pairs_ptr[base].cast[F32]()
    s += y_pairs_ptr[base + D].cast[F32]()
    s += y_pairs_ptr[base + 2 * D].cast[F32]()
    s += y_pairs_ptr[base + 3 * D].cast[F32]()
    y_ptr.store(tok * D + col, s.cast[BF16]())
