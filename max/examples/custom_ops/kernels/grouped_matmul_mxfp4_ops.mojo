# grouped_matmul_mxfp4_ops.mojo
#
# MXFP4 grouped matmul for ModuleV3 MoE expert GEMMs.
#
# This is intended to be a drop-in replacement for `max.nn.kernels.grouped_matmul_ragged`
# in the GPT-OSS ModuleV3 MoE layer:
#   C[p, n] = A[p, :] @ W_expert[n, :].T
# where W_expert is stored as MXFP4 blocks (packed FP4 E2M1 nibbles) + per-block E8M0 scales.
#
# Hard precision rule:
# - FP32 only in registers (accum + scalar temps).
# - Everything in memory is BF16 (or U8/U32 metadata). No FP32 shared tiles.

from math import ceildiv

import compiler
from gpu import (
    WARP_SIZE,
    barrier,
    block_dim,
    block_idx,
    lane_id,
    thread_idx,
    warp_id,
)
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

from .mxfp4_decode import decode_mxfp4_byte_to_2xbf16_scaled, e8m0_to_bf16_bits


comptime BF16 = DType.bfloat16
comptime F32 = DType.float32
comptime U8 = DType.uint8
comptime U32 = DType.uint32
comptime U64 = DType.uint64
comptime I32 = DType.int32

comptime BYTES_PER_BLOCK = 16
comptime VALUES_PER_BLOCK = 32


@parameter
fn grouped_matmul_mxfp4_bf16_tc[
    BM: Int = 16,
    BN: Int = 64,
    BK: Int = 64,
    MMA_M: Int = 16,
    MMA_N: Int = 8,
    MMA_K: Int = 16,
](
    a_ptr: UnsafePointer[BFloat16, MutAnyOrigin],
    P: Int,
    K: Int,
    expert_start_ptr: UnsafePointer[UInt32, MutAnyOrigin],
    expert_ids_ptr: UnsafePointer[Int32, MutAnyOrigin],
    expert_usage_stats_ptr: UnsafePointer[UInt32, MutAnyOrigin],
    w_blocks_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    w_scales_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    Kblocks: Int,
    out_ptr: UnsafePointer[BFloat16, MutAnyOrigin],
    N: Int,
):
    constrained[BK % VALUES_PER_BLOCK == 0, "BK must be a multiple of 32"]()
    constrained[BK % MMA_K == 0]()
    constrained[BM == MMA_M]()
    constrained[BN % MMA_N == 0]()

    var expert_idx = Int(block_idx.z)
    var num_active_experts = Int(expert_usage_stats_ptr[1])
    if expert_idx >= num_active_experts:
        return
    var expert_id = Int(expert_ids_ptr[expert_idx])
    if expert_id < 0:
        return

    var seg_start = Int(expert_start_ptr[expert_idx])
    var seg_end = Int(expert_start_ptr[expert_idx + 1])
    if seg_start >= seg_end:
        return

    var n0 = Int(block_idx.x) * BN
    var row0 = seg_start + Int(block_idx.y) * BM
    if row0 >= seg_end or row0 >= P:
        return

    # Shared tiles (BF16 only).
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
        BF16,
        Layout.row_major(BM, BN),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Warp mapping: BN=64 with MMA_N=8 -> 8 warps along N.
    var warp = Int(warp_id())
    var warp_y = 0
    var warp_x = warp

    # Mixed-precision MMA: BF16 inputs, FP32 accumulator in registers.
    var mma = TensorCore[F32, BF16, Index(MMA_M, MMA_N, MMA_K)]()
    var c_reg = mma.c_reg_tile_type.stack_allocation().fill(0.0)

    comptime blocks_per_tile = BK // VALUES_PER_BLOCK
    var w_blocks_u8 = w_blocks_ptr.address_space_cast[
        AddressSpace.GLOBAL
    ]().bitcast[Scalar[U8]]()

    var num_k_tiles = ceildiv(K, BK)
    for k_tile in range(num_k_tiles):
        barrier()
        var k0 = k_tile * BK
        var kb0 = k0 // VALUES_PER_BLOCK

        # Load A tile.
        for idx in range(Int(thread_idx.x), BM * BK, Int(block_dim.x)):
            var r = idx // BK
            var c = idx - r * BK
            var global_row = row0 + r
            var global_k = k0 + c
            if global_row < seg_end and global_row < P and global_k < K:
                A_s[r, c] = a_ptr[global_row * K + global_k]
            else:
                A_s[r, c] = 0.0

        # Decode B tile from MXFP4 blocks/scales.
        for idx in range(
            Int(thread_idx.x), BN * blocks_per_tile, Int(block_dim.x)
        ):
            var c = idx // blocks_per_tile
            var block_in_tile = idx - c * blocks_per_tile
            var col = n0 + c
            var kb = kb0 + block_in_tile
            var scale_exp = UInt8(0)
            if col < N and kb < Kblocks:
                scale_exp = w_scales_ptr[((expert_id * N + col) * Kblocks) + kb]
            var scale = e8m0_to_bf16_bits(scale_exp)

            # Load packed bytes (16 bytes per 32 values) and decode into BF16.
            var packed_base = (
                ((expert_id * N + col) * Kblocks) + kb
            ) * BYTES_PER_BLOCK

            @parameter
            for byte_in_block in range(BYTES_PER_BLOCK):
                var packed = UInt8(0)
                if col < N and kb < Kblocks:
                    packed = UInt8(w_blocks_u8[packed_base + byte_in_block])
                var r0 = (
                    block_in_tile * VALUES_PER_BLOCK + byte_in_block * 2
                )
                var v2 = decode_mxfp4_byte_to_2xbf16_scaled(
                    packed, scale
                )
                B_s[r0 + 0, c] = v2[0]
                B_s[r0 + 1, c] = v2[1]

        barrier()

        # Each warp computes one MMA tile (16x16) for its N-slice.
        @parameter
        for mma_k in range(BK // MMA_K):
            var a_tile = A_s.tile[MMA_M, MMA_K](warp_y, mma_k)
            var b_tile = B_s.tile[MMA_K, MMA_N](mma_k, warp_x)
            var a_reg = mma.load_a(a_tile)
            var b_reg = mma.load_b(b_tile)
            var d_reg = mma.mma_op(a_reg, b_reg, c_reg)
            c_reg.copy_from(d_reg)

    # Store warp results into BF16 shared C tile.
    var C_warp_tile = C_s.tile[MMA_M, MMA_N](warp_y, warp_x)
    var dst = C_warp_tile.vectorize[1, 2]().distribute[
        Layout.row_major(8, 4)
    ](lane_id())

    comptime c_reg_size = mma.c_reg_type.size
    var c_bf16 = LayoutTensor[
        BF16,
        Layout.col_major(1, c_reg_size),
        MutAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation()
    @parameter
    for i in range(c_reg_size):
        c_bf16[0, i] = c_reg[0, i].cast[BF16]()

    dst.copy_from(c_bf16.vectorize[1, 2]())

    barrier()

    # Copy C_s to global output with global leading dimension N.
    for idx in range(Int(thread_idx.x), BM * BN, Int(block_dim.x)):
        var r = idx // BN
        var c = idx - r * BN
        var global_row = row0 + r
        var global_col = n0 + c
        if global_row < seg_end and global_row < P and global_col < N:
            out_ptr[global_row * N + global_col] = C_s[r, c][0]


@parameter
fn grouped_matmul_mxfp4_bf16_wgmma[
    BM: Int = 128,
    BN: Int = 128,
    BK: Int = 64,
    WGMMA_M: Int = 64,
    WGMMA_N: Int = 128,
    WGMMA_K: Int = 16,
    NUM_WARP_GROUPS: Int = 2,
](
    a_ptr: UnsafePointer[BFloat16, MutAnyOrigin],
    P: Int,
    K: Int,
    expert_start_ptr: UnsafePointer[UInt32, MutAnyOrigin],
    expert_ids_ptr: UnsafePointer[Int32, MutAnyOrigin],
    expert_usage_stats_ptr: UnsafePointer[UInt32, MutAnyOrigin],
    w_blocks_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    w_scales_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    Kblocks: Int,
    out_ptr: UnsafePointer[BFloat16, MutAnyOrigin],
    N: Int,
):
    constrained[WGMMA_M == 64, "SM90 WGMMA requires M=64"]()
    constrained[WGMMA_K in (8, 16, 32)]()
    constrained[BM % WGMMA_M == 0]()
    constrained[BN % WGMMA_N == 0]()
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

    var seg_start = Int(expert_start_ptr[expert_idx])
    var seg_end = Int(expert_start_ptr[expert_idx + 1])
    if seg_start >= seg_end:
        return

    var n0 = Int(block_idx.x) * BN
    var row0 = seg_start + Int(block_idx.y) * BM
    if row0 >= seg_end or row0 >= P:
        return

    var wg_idx = Int(thread_idx.x) >> 7
    var warp_in_wg = Int(warp_id() & UInt(3))

    # Shared tiles in WGMMA-friendly layouts (BF16 only).
    comptime a_smem_layout = tile_layout_k_major[BF16, BM, BK]()
    comptime b_smem_layout = tile_layout_k_major[BF16, BN, BK]()
    comptime blocks_per_tile = BK // VALUES_PER_BLOCK

    comptime a_bytes = a_smem_layout.size() * 2
    comptime b_bytes = b_smem_layout.size() * 2
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
        name="mxfp4_grouped_matmul_dynamic_smem",
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

    # Packed MXFP4 staging buffers.
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
        transpose_b=True,
    ]()

    comptime a_chunks = BK // 4

    # Epilogue mapping (stores output in pairs of columns).
    var lane = Int(thread_idx.x & 31)
    var lane_row = lane // 4
    var lane_col = lane - lane_row * 4
    comptime warp_rows = WGMMA_M // 4
    comptime row_iters = warp_rows // 8
    comptime col_iters = (WGMMA_N // 2) // 4

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

        if col < N and kb < Kblocks:
            var packed_base = (((expert_id * N + col) * Kblocks) + kb) * (
                BYTES_PER_BLOCK
            )
            async_copy[16](
                w_blocks_u8 + packed_base,
                B_pack0 + idx * BYTES_PER_BLOCK,
            )
        else:
            var base_u64 = idx * 2
            B_pack0_u64.store[alignment=8](base_u64 + 0, Scalar[U64](0))
            B_pack0_u64.store[alignment=8](base_u64 + 1, Scalar[U64](0))

    async_copy_commit_group()

    for idx in range(Int(thread_idx.x), BM * a_chunks, Int(block_dim.x)):
        var r = idx // a_chunks
        var chunk = idx - r * a_chunks
        var kk = chunk * 4
        var global_row = row0 + r
        var global_k = kk
        if global_row < seg_end and global_row < P and global_k + 3 < K:
            var p64 = rebind[UnsafePointer[UInt64, MutAnyOrigin]](
                a_ptr + (global_row * K + global_k)
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
                if global_row < seg_end and global_row < P and k_i < K:
                    A_s0[r, kk + i] = a_ptr[global_row * K + k_i]
                else:
                    A_s0[r, kk + i] = 0

    async_copy_wait_all()

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
        if col < N and kb < Kblocks:
            scale_exp = w_scales_ptr[((expert_id * N + col) * Kblocks) + kb]
        var scale = e8m0_to_bf16_bits(scale_exp)

        var base_u64 = idx * 2
        var packed0 = bitcast[DType.uint8, 8](
            UInt64(B_pack0_u64.load[alignment=8](base_u64 + 0))
        )
        var packed1 = bitcast[DType.uint8, 8](
            UInt64(B_pack0_u64.load[alignment=8](base_u64 + 1))
        )

        @parameter
        for byte_in_block in range(8):
            var r0 = block_in_tile * VALUES_PER_BLOCK + byte_in_block * 2
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

    var num_k_tiles = ceildiv(K, BK)
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
                    BN * blocks_per_tile,
                    Int(block_dim.x),
                ):
                    var c = idx // blocks_per_tile
                    var block_in_tile = idx - c * blocks_per_tile
                    var col = n0 + c
                    var kb = kb0_next + block_in_tile

                    if col < N and kb < Kblocks:
                        var packed_base = (
                            (expert_id * N + col) * Kblocks + kb
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

                async_copy_commit_group()

                for idx in range(
                    Int(thread_idx.x), BM * a_chunks, Int(block_dim.x)
                ):
                    var r = idx // a_chunks
                    var chunk = idx - r * a_chunks
                    var kk = chunk * 4
                    var global_row = row0 + r
                    var global_k = k0_next + kk
                    if global_row < seg_end and global_row < P and global_k + 3 < K:
                        var p64 = rebind[
                            UnsafePointer[UInt64, MutAnyOrigin]
                        ](a_ptr + (global_row * K + global_k))
                        var v4 = bitcast[DType.bfloat16, 4](p64[0])
                        A_s1[r, kk + 0] = v4[0]
                        A_s1[r, kk + 1] = v4[1]
                        A_s1[r, kk + 2] = v4[2]
                        A_s1[r, kk + 3] = v4[3]
                    else:

                        @parameter
                        for i in range(4):
                            var k_i = global_k + i
                            if global_row < seg_end and global_row < P and k_i < K:
                                A_s1[r, kk + i] = a_ptr[global_row * K + k_i]
                            else:
                                A_s1[r, kk + i] = 0

                async_copy_wait_all()

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
                    if col < N and kb < Kblocks:
                        scale_exp = w_scales_ptr[
                            ((expert_id * N + col) * Kblocks) + kb
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
                        var r0 = block_in_tile * VALUES_PER_BLOCK + byte * 2
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
                    BN * blocks_per_tile,
                    Int(block_dim.x),
                ):
                    var c = idx // blocks_per_tile
                    var block_in_tile = idx - c * blocks_per_tile
                    var col = n0 + c
                    var kb = kb0_next + block_in_tile

                    if col < N and kb < Kblocks:
                        var packed_base = (
                            (expert_id * N + col) * Kblocks + kb
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

                async_copy_commit_group()

                for idx in range(
                    Int(thread_idx.x), BM * a_chunks, Int(block_dim.x)
                ):
                    var r = idx // a_chunks
                    var chunk = idx - r * a_chunks
                    var kk = chunk * 4
                    var global_row = row0 + r
                    var global_k = k0_next + kk
                    if global_row < seg_end and global_row < P and global_k + 3 < K:
                        var p64 = rebind[
                            UnsafePointer[UInt64, MutAnyOrigin]
                        ](a_ptr + (global_row * K + global_k))
                        var v4 = bitcast[DType.bfloat16, 4](p64[0])
                        A_s0[r, kk + 0] = v4[0]
                        A_s0[r, kk + 1] = v4[1]
                        A_s0[r, kk + 2] = v4[2]
                        A_s0[r, kk + 3] = v4[3]
                    else:

                        @parameter
                        for i in range(4):
                            var k_i = global_k + i
                            if global_row < seg_end and global_row < P and k_i < K:
                                A_s0[r, kk + i] = a_ptr[global_row * K + k_i]
                            else:
                                A_s0[r, kk + i] = 0

                async_copy_wait_all()

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
                    if col < N and kb < Kblocks:
                        scale_exp = w_scales_ptr[
                            ((expert_id * N + col) * Kblocks) + kb
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
                        var r0 = block_in_tile * VALUES_PER_BLOCK + byte * 2
                        var v2 = decode_mxfp4_byte_to_2xbf16_scaled(
                            packed1[byte_in_block], scale
                        )
                        B_s0[c, r0 + 0] = v2[0]
                        B_s0[c, r0 + 1] = v2[1]

                barrier()

    wgmma.wait_group()

    # Epilogue: FP32 accum -> BF16 store.
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
                    var col0 = n0 + n_mma * WGMMA_N + col_pair * 2
                    var col1 = col0 + 1
                    var frag_idx = c_it * row_iters + r_it
                    var v2 = c_pairs[0, frag_idx]

                    if col0 < N:
                        out_ptr[global_row * N + col0] = v2[0].cast[BF16]()
                    if col1 < N:
                        out_ptr[global_row * N + col1] = v2[1].cast[BF16]()

    barrier()


@compiler.register("mxfp4_grouped_matmul_ragged_bf16")
struct MXFP4GroupedMatmulRaggedBF16:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        c: OutputTensor[dtype=BF16, rank=2],
        a: InputTensor[dtype=BF16, rank=2],
        w_blocks: InputTensor[dtype=U8, rank=4],
        w_scales: InputTensor[dtype=U8, rank=3],
        expert_start_indices: InputTensor[dtype=U32, rank=1],
        expert_ids: InputTensor[dtype=I32, rank=1],
        expert_usage_stats: InputTensor[dtype=U32, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if is_cpu[target]():
            raise Error("mxfp4_grouped_matmul_ragged_bf16: GPU only")

        var P = a.dim_size(0)
        var K = a.dim_size(1)
        var N = c.dim_size(1)
        var Kblocks = w_blocks.dim_size(2)

        if P == 0 or K == 0 or N == 0:
            return

        # Validate MXFP4 block shape matches K.
        if Kblocks * VALUES_PER_BLOCK != K:
            raise Error(
                "mxfp4_grouped_matmul_ragged_bf16: K must be divisible by 32 and match w_blocks Kblocks"
            )

        var gpu_ctx = ctx.get_device_context()
        var a_dev = DeviceBuffer[a.dtype](
            gpu_ctx, a.unsafe_ptr(), a.size(), owning=False
        )
        var c_dev = DeviceBuffer[c.dtype](
            gpu_ctx, c.unsafe_ptr(), c.size(), owning=False
        )
        var w_blocks_dev = DeviceBuffer[w_blocks.dtype](
            gpu_ctx, w_blocks.unsafe_ptr(), w_blocks.size(), owning=False
        )
        var w_scales_dev = DeviceBuffer[w_scales.dtype](
            gpu_ctx, w_scales.unsafe_ptr(), w_scales.size(), owning=False
        )
        var expert_start_dev = DeviceBuffer[expert_start_indices.dtype](
            gpu_ctx,
            expert_start_indices.unsafe_ptr(),
            expert_start_indices.size(),
            owning=False,
        )
        var expert_ids_dev = DeviceBuffer[expert_ids.dtype](
            gpu_ctx, expert_ids.unsafe_ptr(), expert_ids.size(), owning=False
        )
        var expert_usage_stats_dev = DeviceBuffer[expert_usage_stats.dtype](
            gpu_ctx,
            expert_usage_stats.unsafe_ptr(),
            expert_usage_stats.size(),
            owning=False,
        )

        # Dispatch by total routed tokens (P) to avoid host-visible routing stats, which
        # can be unsafe to read without synchronization.
        var grid_z = expert_ids.dim_size(0)
        # Use TC for very small P (decode/single-request) where kernel launch
        # overhead dominates; use WGMMA for prefill-sized P.
        if P <= 256:
            comptime BN = 64
            comptime BM = 16
            var grid_x = ceildiv(N, BN)
            var grid_y = ceildiv(P, BM)
            if grid_x == 0 or grid_y == 0:
                return

            comptime tc_kernel = grouped_matmul_mxfp4_bf16_tc[
                BM=16,
                BN=64,
                BK=64,
                MMA_M=16,
                MMA_N=8,
                MMA_K=16,
            ]
            gpu_ctx.enqueue_function_checked[tc_kernel, tc_kernel](
                a_dev,
                P,
                K,
                expert_start_dev,
                expert_ids_dev,
                expert_usage_stats_dev,
                w_blocks_dev,
                w_scales_dev,
                Kblocks,
                c_dev,
                N,
                grid_dim=(grid_x, grid_y, grid_z),
                block_dim=(256, 1, 1),
            )
            return

        comptime BN = 128
        comptime BM = 128
        var grid_x = ceildiv(N, BN)
        var grid_y = ceildiv(P, BM)
        if grid_x == 0 or grid_y == 0:
            return

        comptime wgmma_kernel = grouped_matmul_mxfp4_bf16_wgmma[
            BM=128,
            BN=128,
            BK=64,
            WGMMA_M=64,
            WGMMA_N=128,
            WGMMA_K=16,
            NUM_WARP_GROUPS=2,
        ]
        comptime a_smem_layout = tile_layout_k_major[BF16, 128, 64]()
        comptime b_smem_layout = tile_layout_k_major[BF16, 128, 64]()
        comptime a_bytes = a_smem_layout.size() * 2
        comptime b_bytes = b_smem_layout.size() * 2
        comptime a1_off = ((a_bytes + 255) // 256) * 256
        comptime b0_off = ((a1_off + a_bytes + 255) // 256) * 256
        comptime b1_off = ((b0_off + b_bytes + 255) // 256) * 256
        comptime blocks_per_tile = 64 // VALUES_PER_BLOCK
        comptime pack_bytes = 128 * blocks_per_tile * BYTES_PER_BLOCK
        comptime pack0_off = ((b1_off + b_bytes + 255) // 256) * 256
        comptime pack1_off = ((pack0_off + pack_bytes + 255) // 256) * 256
        comptime smem_use = pack1_off + pack_bytes
        gpu_ctx.enqueue_function_checked[wgmma_kernel, wgmma_kernel](
            a_dev,
            P,
            K,
            expert_start_dev,
            expert_ids_dev,
            expert_usage_stats_dev,
            w_blocks_dev,
            w_scales_dev,
            Kblocks,
            c_dev,
            N,
            grid_dim=(grid_x, grid_y, grid_z),
            block_dim=(256, 1, 1),
            shared_mem_bytes=Int(smem_use),
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                smem_use
            ),
        )
