from .grouped_matmul_sm90_common import *


@parameter
fn grouped_matmul_mxfp4_bf16_warp_mma_sm90[
    BM: Int = 64,
    BN: Int = 64,
    BK: Int = 64,
    WM: Int = 64,
    WN: Int = 64,
    MMA_M: Int = 16,
    MMA_N: Int = 8,
    MMA_K: Int = 16,
](
    a_ptr: UnsafePointer[BFloat16, MutAnyOrigin],
    a_stride0: Int,
    a_stride1: Int,
    P: Int,
    K: Int,
    expert_start_ptr: UnsafePointer[UInt32, MutAnyOrigin],
    expert_ids_ptr: UnsafePointer[Int32, MutAnyOrigin],
    num_active_experts: Int,
    w_blocks_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    w_scales_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    Kblocks: Int,
    out_ptr: UnsafePointer[BFloat16, MutAnyOrigin],
    out_stride0: Int,
    out_stride1: Int,
    N: Int,
):
    constrained[BM == WM]()
    constrained[BN == WN]()
    constrained[BM % MMA_M == 0]()
    constrained[BN % MMA_N == 0]()
    constrained[BK % MMA_K == 0]()
    constrained[
        BK % VALUES_PER_BLOCK == 0,
        "BK must be a multiple of 32",
    ]()
    constrained[
        (HOPPER_SCALE_NUM_WARPS & (HOPPER_SCALE_NUM_WARPS - 1)) == 0,
        "Hopper scale swizzle requires power-of-two num_warps",
    ]()
    constrained[
        BN % (HOPPER_SCALE_ALIGN_M) == 0,
        "BN must be a multiple of 32 * num_warps for Hopper scale swizzle",
    ]()
    constrained[
        BK % 64 == 0,
        "BK must be a multiple of 64 for Hopper scale swizzle",
    ]()

    var expert_idx = Int(block_idx.z)
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

    var w_blocks_u64 = w_blocks_ptr.address_space_cast[
        AddressSpace.GLOBAL
    ]().bitcast[Scalar[U64]]()

    var n_pad = (
        (N + HOPPER_SCALE_ALIGN_M - 1) // HOPPER_SCALE_ALIGN_M
    ) * HOPPER_SCALE_ALIGN_M
    var scale_m2 = n_pad // 32
    var w_scales_stride0 = scale_m2 * K
    var w_scales_stride1 = K
    var w_scales_stride2 = 1

    var A_s = LayoutTensor[
        BF16,
        Layout.row_major(BM, BK),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var C_s = LayoutTensor[
        F32,
        Layout.row_major(BM, BN),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var mma = TensorCore[F32, BF16, Index(MMA_M, MMA_N, MMA_K)]()

    var c_reg = (
        LayoutTensor[
            F32,
            Layout.row_major(WM // MMA_M, (WN * 4) // MMA_N),
            MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .fill(0.0)
    )

    var num_k_tiles = ceildiv(K, BK)
    for k_tile in range(num_k_tiles):
        var k_base = k_tile * BK

        for idx in range(Int(thread_idx.x), BM * BK, Int(block_dim.x)):
            var r = idx // BK
            var c = idx - r * BK
            var global_row = row0 + r
            var global_k = k_base + c
            if global_row < seg_end and global_row < P and global_k < K:
                A_s[r, c] = a_ptr[global_row * a_stride0 + global_k * a_stride1]
            else:
                A_s[r, c] = 0
        barrier()

        var A_warp_tile = A_s.tile[WM, BK](0, 0)

        @parameter
        for mma_k in range(BK // MMA_K):
            var k_frag_base = k_base + mma_k * MMA_K
            var kb = k_frag_base // VALUES_PER_BLOCK
            var k_in_block = k_frag_base - kb * VALUES_PER_BLOCK
            var byte_start = k_in_block // 2

            @parameter
            for mma_n in range(WN // MMA_N):
                var n_frag_base = n0 + mma_n * MMA_N
                __comptime_assert (
                    MMA_M == 16 and MMA_N == 8 and MMA_K == 16
                ), "Register decode mapping assumes MMA 16x8x16"

                var lane = Int(lane_id())
                var lane_k = lane & 3
                var lane_n = lane >> 2
                var col = n_frag_base + lane_n

                comptime b_reg_elems = (MMA_K * MMA_N) // WARP_SIZE
                var b_reg_tile = LayoutTensor[
                    BF16,
                    Layout.row_major(b_reg_elems, 1),
                    MutAnyOrigin,
                    address_space = AddressSpace.LOCAL,
                ].stack_allocation()

                b_reg_tile[0, 0] = 0
                b_reg_tile[1, 0] = 0
                b_reg_tile[2, 0] = 0
                b_reg_tile[3, 0] = 0

                if col < N and kb < Kblocks:
                    var idx = hopper_scale_swizzle_index[
                        HOPPER_SCALE_NUM_WARPS
                    ](col, kb)
                    var scale_exp = w_scales_ptr[
                        expert_id * w_scales_stride0
                        + idx[0] * w_scales_stride1
                        + idx[1] * w_scales_stride2
                    ]
                    var scale = e8m0_to_bf16_bits(scale_exp)
                    var packed_base = (
                        ((expert_id * Kblocks + kb) * N) + col
                    ) * BYTES_PER_BLOCK + byte_start
                    var base_u64 = packed_base // 8
                    var byte_offset = packed_base - base_u64 * 8
                    var packed = bitcast[DType.uint8, 8](
                        UInt64(w_blocks_u64.load[alignment=8](base_u64))
                    )

                    var byte_base = Int(byte_offset)
                    var p0 = u32_from_u8x4(
                        UInt8(packed[byte_base + 0]),
                        UInt8(packed[byte_base + 1]),
                        UInt8(packed[byte_base + 2]),
                        UInt8(packed[byte_base + 3]),
                    )
                    var p1 = u32_from_u8x4(
                        UInt8(packed[byte_base + 4]),
                        UInt8(packed[byte_base + 5]),
                        UInt8(packed[byte_base + 6]),
                        UInt8(packed[byte_base + 7]),
                    )
                    var out0 = decode_mxfp4_packbits_u32_to_8xbf16_scaled(
                        p0, scale
                    )
                    var out1 = decode_mxfp4_packbits_u32_to_8xbf16_scaled(
                        p1, scale
                    )
                    var idx0 = lane_k * 2
                    b_reg_tile[0, 0] = out0[idx0 + 0]
                    b_reg_tile[1, 0] = out0[idx0 + 1]
                    b_reg_tile[2, 0] = out1[idx0 + 0]
                    b_reg_tile[3, 0] = out1[idx0 + 1]

                @parameter
                for mma_m in range(WM // MMA_M):
                    var c_tile = c_reg.tile[1, 4](mma_m, mma_n)
                    var A_mma = A_warp_tile.tile[MMA_M, MMA_K](mma_m, mma_k)
                    var a_reg = mma.load_a(A_mma)
                    var d_reg = mma.mma_op(a_reg, b_reg_tile, c_tile)
                    c_tile.copy_from(d_reg)
        barrier()

    var C_warp = C_s.tile[WM, WN](0, 0)

    @parameter
    for mma_m in range(WM // MMA_M):

        @parameter
        for mma_n in range(WN // MMA_N):
            var C_mma = C_warp.tile[MMA_M, MMA_N](mma_m, mma_n)
            var c_tile = c_reg.tile[1, 4](mma_m, mma_n)
            mma.store_d(C_mma, c_tile)
    barrier()

    for idx in range(Int(thread_idx.x), BM * BN, Int(block_dim.x)):
        var r = idx // BN
        var c = idx - r * BN
        var global_row = row0 + r
        var global_col = n0 + c
        if global_row < seg_end and global_row < P and global_col < N:
            out_ptr[global_row * out_stride0 + global_col * out_stride1] = C_s[
                r, c
            ][0].cast[BF16]()


@parameter
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
