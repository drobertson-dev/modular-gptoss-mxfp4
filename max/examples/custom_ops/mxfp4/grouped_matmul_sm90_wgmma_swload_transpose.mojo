from .grouped_matmul_sm90_common import *
from .grouped_matmul_sm90_common import (
    apply_xor_swizzle,
    decode_mxfp4_unshuffle_value,
    compute_kbyte_row0_from_col,
    load_swizzled_pack_u32,
)


@parameter
fn grouped_matmul_mxfp4_bf16_wgmma_sm90_pipeline_swload_transpose[
    BM: Int = 256,
    BN: Int = 64,
    BK: Int = 64,
    WGMMA_M: Int = 64,
    WGMMA_N: Int = 64,
    WGMMA_K: Int = 16,
    NUM_WARP_GROUPS: Int = 1,  # number of consumer warp groups
    NUM_PIPELINE_STAGES: Int = 2,
    USE_VALUE_SWIZZLE: Bool = False,
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
    w_blocks_stride0: Int,
    w_blocks_stride1: Int,
    w_blocks_stride2: Int,
    w_scales_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    w_scales_stride0: Int,
    w_scales_stride1: Int,
    w_scales_stride2: Int,
    Kblocks: Int,
    out_ptr: UnsafePointer[BFloat16, MutAnyOrigin],
    out_stride0: Int,
    out_stride1: Int,
    N: Int,
):
    constrained[WGMMA_M == 64, "SM90 WGMMA requires M=64"]()
    constrained[WGMMA_K in (8, 16, 32)]()
    constrained[BM % WGMMA_M == 0]()
    constrained[BN % WGMMA_N == 0]()
    constrained[BK % WGMMA_K == 0]()
    constrained[BK % VALUES_PER_BLOCK == 0, "BK must be a multiple of 32"]()
    constrained[BM % (WGMMA_M * NUM_WARP_GROUPS) == 0]()
    constrained[NUM_PIPELINE_STAGES >= 2]()

    var expert_idx = Int(block_idx.z)
    if expert_idx >= num_active_experts:
        return

    var seg_start = Int(expert_start_ptr[expert_idx])
    var seg_end = Int(expert_start_ptr[expert_idx + 1])
    # Guard against undefined tail entries when host passes a conservative
    # num_active_experts bound. Only segments fully inside [0, P] are valid.
    if seg_start < 0 or seg_end < 0 or seg_start > P or seg_end > P:
        return
    if seg_start >= seg_end:
        return
    var expert_id = Int(expert_ids_ptr[expert_idx])
    if expert_id < 0 or expert_id >= num_active_experts:
        return
    # Grouped tile scheduler guard: skip tiles beyond this expert's segment.
    var seg_len = seg_end - seg_start
    var max_tiles = ceildiv(seg_len, BN)
    if Int(block_idx.y) >= max_tiles:
        return

    var n0 = Int(block_idx.x) * BM  # N-tile
    if n0 >= N:
        return
    var row0 = seg_start + Int(block_idx.y) * BN  # M-tile
    if row0 >= seg_end or row0 >= P:
        return
    var kbytes = K >> 1

    var warp_group_idx, warp_group_thread_idx = divmod(
        thread_idx.x, UInt(WARPGROUP_SIZE)
    )
    var local_wg_idx = Int(warp_group_idx) - 1  # consumer-local index

    # Shared tiles (BF16 only).
    # RS path: activations (B) live in shared; weights (A) are decoded to regs.
    comptime a_swizzle = TensorMapSwizzle.SWIZZLE_NONE
    comptime b_swizzle = TensorMapSwizzle.SWIZZLE_128B
    comptime b_smem_layout = tile_layout_k_major[BF16, BN, BK, b_swizzle]()
    comptime b_smem_swizzle = make_swizzle[BF16, b_swizzle]()
    comptime B_smem_tile = LayoutTensor[
        BF16,
        b_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=1024,
    ]

    comptime b_bytes = b_smem_layout.size() * 2
    comptime b_stage_bytes = ((b_bytes + 255) // 256) * 256
    comptime a_stage_bytes = 0
    comptime stage_bytes = b_stage_bytes

    var smem = external_memory[
        Scalar[U8],
        address_space = AddressSpace.SHARED,
        alignment=1024,
        name="mxfp4_grouped_matmul_pipeline_smem_swload_transpose",
    ]()
    var smem_ptr = smem.address_space_cast[AddressSpace.SHARED]().mut_cast[
        True
    ]()

    var full_mbar = stack_allocation[
        NUM_PIPELINE_STAGES,
        SharedMemBarrier,
        address_space = AddressSpace.SHARED,
        alignment=8,
    ]()
    var empty_mbar = stack_allocation[
        NUM_PIPELINE_STAGES,
        SharedMemBarrier,
        address_space = AddressSpace.SHARED,
        alignment=8,
    ]()

    if thread_idx.x == 0:

        @parameter
        for i in range(NUM_PIPELINE_STAGES):
            full_mbar[i].init(Int32(WARPGROUP_SIZE))
            empty_mbar[i].init(Int32(NUM_WARP_GROUPS))

    barrier()

    comptime num_m_mmas_total = BM // WGMMA_M
    comptime num_m_mmas = num_m_mmas_total // NUM_WARP_GROUPS
    comptime num_n_mmas = BN // WGMMA_N
    comptime num_k_mmas = BK // WGMMA_K
    comptime c_frag_size = (WGMMA_M * WGMMA_N) // 128
    comptime a_frag_size = (WGMMA_M * WGMMA_K) // 128

    comptime warp_rows = WGMMA_M // 4
    comptime row_iters = warp_rows // 8
    comptime col_iters = (WGMMA_N // 2) // 4

    if warp_group_idx != 0:
        if warp_group_thread_idx == 0:

            @parameter
            for i in range(NUM_PIPELINE_STAGES):
                _ = empty_mbar[i].arrive()

    var num_k_tiles = ceildiv(K, BK)

    if warp_group_idx == 0:

        @parameter
        if NUM_WARP_GROUPS <= 2:
            warpgroup_reg_dealloc[24]()
        else:
            warpgroup_reg_dealloc[32]()

        var write_state = PipelineState[NUM_PIPELINE_STAGES]()

        for k_tile in range(num_k_tiles):
            var slot = write_state.index()
            empty_mbar[Int(slot)].wait(write_state.phase())

            var stage_base = smem_ptr + Int(slot) * stage_bytes
            var B_s = B_smem_tile(stage_base.bitcast[Scalar[BF16]]())
            var B_data = stage_base.bitcast[Scalar[BF16]]()
            var b_strides = B_s.runtime_layout.stride.value

            var k0 = k_tile * BK
            var kb0 = k0 // VALUES_PER_BLOCK
            comptime blocks_per_tile = BK // VALUES_PER_BLOCK

            # Load activations into B_s (swizzled shared), vectorized 8-wide.
            comptime k_vec = BK // 8
            for idx in range(
                Int(warp_group_thread_idx), BN * k_vec, Int(WARPGROUP_SIZE)
            ):
                var r = idx // k_vec
                var base_k8 = idx - r * k_vec
                var k8 = base_k8
                @parameter
                if TRANSPOSE_WRITER_MAP == 1:
                    var phase = r & (k_vec - 1)
                    k8 = (base_k8 + phase) & (k_vec - 1)
                elif TRANSPOSE_WRITER_MAP == 2:
                    k8 = ((base_k8 & 1) << 2) | (base_k8 >> 1)
                var k = k8 * 8
                var global_row = row0 + r
                var global_col = k0 + k

                var outv = SIMD[BF16, 8](0)
                if global_row < seg_end and global_row < P:
                    if global_col + 7 < K:
                        outv = a_ptr.load[width=8, alignment=16](
                            global_row * a_stride0 + global_col * a_stride1
                        )
                    else:
                        @parameter
                        for t in range(8):
                            var col = global_col + t
                            if col < K:
                                outv[t] = a_ptr[
                                    global_row * a_stride0 + col * a_stride1
                                ].cast[BF16]()

                var ridx = B_smem_tile.idx_list_t[2](fill=0)
                ridx[0] = r
                ridx[1] = k
                var lin = B_smem_tile._get_offset(b_strides, ridx)
                var swz = apply_xor_swizzle(b_smem_swizzle, lin)
                B_data.store[width=8, alignment=16](swz, outv)

            _ = full_mbar[Int(slot)].arrive()
            write_state.step()

    else:

        @parameter
        fn consumer_regs() -> Int:
            if NUM_WARP_GROUPS == 1:
                return 224
            if NUM_WARP_GROUPS == 2:
                return 192
            return 160

        warpgroup_reg_alloc[consumer_regs()]()

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
            a_swizzle=a_swizzle,
            b_swizzle=b_swizzle,
            transpose_b=True,
        ]()

        var lane = Int(thread_idx.x & UInt(31))
        var lane_row = lane // 4
        var lane_col = lane - lane_row * 4

        var read_state = PipelineState[NUM_PIPELINE_STAGES]()

        for k_tile in range(num_k_tiles):
            var slot = read_state.index()
            full_mbar[Int(slot)].wait_acquire[Scope.BLOCK](read_state.phase())

            var stage_base = smem_ptr + Int(slot) * stage_bytes
            var B_s = B_smem_tile(stage_base.bitcast[Scalar[BF16]]())

            # Decode weights directly into A fragments (RS WGMMA path).
            var a_frag_tile = (
                LayoutTensor[
                    BF16,
                    Layout.row_major(num_m_mmas * num_k_mmas, a_frag_size),
                    MutAnyOrigin,
                    address_space = AddressSpace.LOCAL,
                ]
                .stack_allocation()
            )
            var a_frags = a_frag_tile.vectorize[1, a_frag_size]()

            # Use warp/lane coordinates local to this consumer warpgroup.
            # Global warp_id() offsets producer warps and breaks fragment row mapping.
            var warp = Int(warp_group_thread_idx) // Int(WARP_SIZE)
            var lane = Int(warp_group_thread_idx) % Int(WARP_SIZE)
            var row_block = warp * (WGMMA_M // 4)
            var lane_row = lane >> 2
            var lane_col = lane & 3

            var k0 = k_tile * BK
            comptime blocks_per_tile = BK // VALUES_PER_BLOCK
            var kb0 = k_tile * blocks_per_tile
            var base_m2 = Int(block_idx.x) * (BM // 4)
            var base_k2 = k_tile * (2 * BK)

            var w_scales_u32 = w_scales_ptr.address_space_cast[
                AddressSpace.GLOBAL
            ]().bitcast[Scalar[U32]]()

            @parameter
            for m_mma in range(num_m_mmas):
                var row_base = m_mma * WGMMA_M + row_block
                var row0_rel = row_base + lane_row
                var row1_rel = row0_rel + 8
                var row0_abs = n0 + row0_rel
                var row1_abs = n0 + row1_rel

                var row0_scale_exp = SIMD[U8, 4](0)
                var row1_scale_exp = SIMD[U8, 4](0)

                if row0_abs < N:
                    var kb_abs0 = kb0
                    if kb_abs0 < Kblocks:
                        var idx0 = hopper_scale_swizzle_index_fast(
                            row0_abs, kb_abs0
                        )
                        var base0 = idx0[1] & ~3
                        var byte0 = idx0[1] & 3
                        var p0 = w_scales_u32[
                            (expert_id * w_scales_stride0
                             + idx0[0] * w_scales_stride1
                             + base0) >> 2
                        ]
                        var exp0 = UInt8(p0 >> UInt32(byte0 * 8))
                        row0_scale_exp[0] = exp0
                        if kb_abs0 + 1 < Kblocks:
                            var exp1 = UInt8(
                                p0 >> UInt32((byte0 + 2) * 8)
                            )
                            row0_scale_exp[1] = exp1

                    var kb_abs2 = kb0 + 2
                    if kb_abs2 < Kblocks:
                        var idx2 = hopper_scale_swizzle_index_fast(
                            row0_abs, kb_abs2
                        )
                        var base2 = idx2[1] & ~3
                        var byte2 = idx2[1] & 3
                        var p2 = w_scales_u32[
                            (expert_id * w_scales_stride0
                             + idx2[0] * w_scales_stride1
                             + base2) >> 2
                        ]
                        var exp2 = UInt8(p2 >> UInt32(byte2 * 8))
                        row0_scale_exp[2] = exp2
                        if kb_abs2 + 1 < Kblocks:
                            var exp3 = UInt8(
                                p2 >> UInt32((byte2 + 2) * 8)
                            )
                            row0_scale_exp[3] = exp3

                if row1_abs < N:
                    var kb_abs0b = kb0
                    if kb_abs0b < Kblocks:
                        var idx1 = hopper_scale_swizzle_index_fast(
                            row1_abs, kb_abs0b
                        )
                        var base1 = idx1[1] & ~3
                        var byte1 = idx1[1] & 3
                        var p1 = w_scales_u32[
                            (expert_id * w_scales_stride0
                             + idx1[0] * w_scales_stride1
                             + base1) >> 2
                        ]
                        var exp4 = UInt8(p1 >> UInt32(byte1 * 8))
                        row1_scale_exp[0] = exp4
                        if kb_abs0b + 1 < Kblocks:
                            var exp5 = UInt8(
                                p1 >> UInt32((byte1 + 2) * 8)
                            )
                            row1_scale_exp[1] = exp5

                    var kb_abs2b = kb0 + 2
                    if kb_abs2b < Kblocks:
                        var idx3 = hopper_scale_swizzle_index_fast(
                            row1_abs, kb_abs2b
                        )
                        var base3 = idx3[1] & ~3
                        var byte3 = idx3[1] & 3
                        var p3 = w_scales_u32[
                            (expert_id * w_scales_stride0
                             + idx3[0] * w_scales_stride1
                             + base3) >> 2
                        ]
                        var exp6 = UInt8(p3 >> UInt32(byte3 * 8))
                        row1_scale_exp[2] = exp6
                        if kb_abs2b + 1 < Kblocks:
                            var exp7 = UInt8(
                                p3 >> UInt32((byte3 + 2) * 8)
                            )
                            row1_scale_exp[3] = exp7

                @parameter
                for k_mma in range(num_k_mmas):
                    var col_base = k_mma * WGMMA_K
                    var colg0 = lane_col
                    var colg1 = lane_col + 4
                    var col0 = col_base + colg0 * 2
                    var col1 = col0 + 1
                    var col2 = col_base + colg1 * 2
                    var col3 = col2 + 1

                    var v00 = Scalar[BF16](0)
                    var v01 = Scalar[BF16](0)
                    var v02 = Scalar[BF16](0)
                    var v03 = Scalar[BF16](0)
                    var v04 = Scalar[BF16](0)
                    var v05 = Scalar[BF16](0)
                    var v06 = Scalar[BF16](0)
                    var v07 = Scalar[BF16](0)

                    @parameter
                    if USE_VALUE_SWIZZLE:
                        if row0_abs < N:
                            var col0_abs = k0 + col0
                            if col0_abs < K:
                                var kb_rel0 = col0 >> 5
                                var kbyte0 = compute_kbyte_row0_from_col(
                                    row0_rel,
                                    col0,
                                    k0 >> 1,
                                )
                                if kbyte0 < kbytes:
                                    # Partial K tile: treat out-of-range swizzled bytes as zero.
                                    # This prevents OOB loads when K is not a multiple of BK.
                                    var p0, byte_idx0 = load_swizzled_pack_u32(
                                        w_blocks_ptr,
                                        w_blocks_stride0,
                                        w_blocks_stride1,
                                        w_blocks_stride2,
                                        expert_id,
                                        row0_abs,
                                        kbyte0,
                                    )
                                    var scale0: UInt8 = 0
                                    if kb_rel0 < Kblocks:
                                        var sidx0 = hopper_scale_swizzle_index_fast(
                                            row0_abs, kb_rel0
                                        )
                                        var sbase0 = sidx0[1] & ~3
                                        var sbyte0 = sidx0[1] & 3
                                        var sp0 = w_scales_u32[
                                            (expert_id * w_scales_stride0
                                             + sidx0[0] * w_scales_stride1
                                             + sbase0) >> 2
                                        ]
                                        scale0 = UInt8(
                                            sp0 >> UInt32(sbyte0 * 8)
                                        )
                                    var vals0 = (
                                        decode_mxfp4_packbits_u32_to_8xbf16_scaled_e8m0(
                                            p0,
                                            scale0,
                                        )
                                    )
                                    var base_idx0 = byte_idx0 * 2
                                    v00 = vals0[base_idx0]
                                    if col0_abs + 1 < K:
                                        v01 = vals0[base_idx0 + 1]

                                if row1_abs < N:
                                    var kbyte0b = compute_kbyte_row0_from_col(
                                        row1_rel,
                                        col0,
                                        k0 >> 1,
                                    )
                                    if kbyte0b < kbytes:
                                        var p0b, byte_idx0b = load_swizzled_pack_u32(
                                            w_blocks_ptr,
                                            w_blocks_stride0,
                                            w_blocks_stride1,
                                            w_blocks_stride2,
                                            expert_id,
                                            row1_abs,
                                            kbyte0b,
                                        )
                                        var scale0b: UInt8 = 0
                                        if kb_rel0 < Kblocks:
                                            var sidx0b = hopper_scale_swizzle_index_fast(
                                                row1_abs, kb_rel0
                                            )
                                            var sbase0b = sidx0b[1] & ~3
                                            var sbyte0b = sidx0b[1] & 3
                                            var sp0b = w_scales_u32[
                                                (expert_id * w_scales_stride0
                                                 + sidx0b[0] * w_scales_stride1
                                                 + sbase0b) >> 2
                                            ]
                                            scale0b = UInt8(
                                                sp0b >> UInt32(sbyte0b * 8)
                                            )
                                        var vals0b = (
                                            decode_mxfp4_packbits_u32_to_8xbf16_scaled_e8m0(
                                                p0b,
                                                scale0b,
                                            )
                                        )
                                        var base_idx0b = byte_idx0b * 2
                                        v04 = vals0b[base_idx0b]
                                        if col0_abs + 1 < K:
                                            v05 = vals0b[base_idx0b + 1]

                            var col2_abs = k0 + col2
                            if col2_abs < K:
                                var kb_rel2 = col2 >> 5
                                var kbyte2 = compute_kbyte_row0_from_col(
                                    row0_rel,
                                    col2,
                                    k0 >> 1,
                                )
                                if kbyte2 < kbytes:
                                    var p2, byte_idx2 = load_swizzled_pack_u32(
                                        w_blocks_ptr,
                                        w_blocks_stride0,
                                        w_blocks_stride1,
                                        w_blocks_stride2,
                                        expert_id,
                                        row0_abs,
                                        kbyte2,
                                    )
                                    var scale2: UInt8 = 0
                                    if kb_rel2 < Kblocks:
                                        var sidx2 = hopper_scale_swizzle_index_fast(
                                            row0_abs, kb_rel2
                                        )
                                        var sbase2 = sidx2[1] & ~3
                                        var sbyte2 = sidx2[1] & 3
                                        var sp2 = w_scales_u32[
                                            (expert_id * w_scales_stride0
                                             + sidx2[0] * w_scales_stride1
                                             + sbase2) >> 2
                                        ]
                                        scale2 = UInt8(
                                            sp2 >> UInt32(sbyte2 * 8)
                                        )
                                    var vals2 = (
                                        decode_mxfp4_packbits_u32_to_8xbf16_scaled_e8m0(
                                            p2,
                                            scale2,
                                        )
                                    )
                                    var base_idx2 = byte_idx2 * 2
                                    v02 = vals2[base_idx2]
                                    if col2_abs + 1 < K:
                                        v03 = vals2[base_idx2 + 1]

                                if row1_abs < N:
                                    var kbyte2b = compute_kbyte_row0_from_col(
                                        row1_rel,
                                        col2,
                                        k0 >> 1,
                                    )
                                    if kbyte2b < kbytes:
                                        var p2b, byte_idx2b = load_swizzled_pack_u32(
                                            w_blocks_ptr,
                                            w_blocks_stride0,
                                            w_blocks_stride1,
                                            w_blocks_stride2,
                                            expert_id,
                                            row1_abs,
                                            kbyte2b,
                                        )
                                        var scale2b: UInt8 = 0
                                        if kb_rel2 < Kblocks:
                                            var sidx2b = hopper_scale_swizzle_index_fast(
                                                row1_abs, kb_rel2
                                            )
                                            var sbase2b = sidx2b[1] & ~3
                                            var sbyte2b = sidx2b[1] & 3
                                            var sp2b = w_scales_u32[
                                                (expert_id * w_scales_stride0
                                                 + sidx2b[0] * w_scales_stride1
                                                 + sbase2b) >> 2
                                            ]
                                            scale2b = UInt8(
                                                sp2b >> UInt32(sbyte2b * 8)
                                            )
                                        var vals2b = (
                                            decode_mxfp4_packbits_u32_to_8xbf16_scaled_e8m0(
                                                p2b,
                                                scale2b,
                                            )
                                        )
                                        var base_idx2b = byte_idx2b * 2
                                        v06 = vals2b[base_idx2b]
                                        if col2_abs + 1 < K:
                                            v07 = vals2b[base_idx2b + 1]
                    else:
                        v00 = decode_mxfp4_unshuffle_value[
                            USE_VALUE_SWIZZLE=USE_VALUE_SWIZZLE
                        ](
                            w_blocks_ptr,
                            w_blocks_stride0,
                            w_blocks_stride1,
                            w_blocks_stride2,
                            expert_id,
                            base_m2,
                            base_k2,
                            row0_rel,
                            row0_abs,
                            col0,
                            k0,
                            K,
                            N,
                            row0_scale_exp,
                        )
                        v01 = decode_mxfp4_unshuffle_value[
                            USE_VALUE_SWIZZLE=USE_VALUE_SWIZZLE
                        ](
                            w_blocks_ptr,
                            w_blocks_stride0,
                            w_blocks_stride1,
                            w_blocks_stride2,
                            expert_id,
                            base_m2,
                            base_k2,
                            row0_rel,
                            row0_abs,
                            col1,
                            k0,
                            K,
                            N,
                            row0_scale_exp,
                        )
                        v02 = decode_mxfp4_unshuffle_value[
                            USE_VALUE_SWIZZLE=USE_VALUE_SWIZZLE
                        ](
                            w_blocks_ptr,
                            w_blocks_stride0,
                            w_blocks_stride1,
                            w_blocks_stride2,
                            expert_id,
                            base_m2,
                            base_k2,
                            row0_rel,
                            row0_abs,
                            col2,
                            k0,
                            K,
                            N,
                            row0_scale_exp,
                        )
                        v03 = decode_mxfp4_unshuffle_value[
                            USE_VALUE_SWIZZLE=USE_VALUE_SWIZZLE
                        ](
                            w_blocks_ptr,
                            w_blocks_stride0,
                            w_blocks_stride1,
                            w_blocks_stride2,
                            expert_id,
                            base_m2,
                            base_k2,
                            row0_rel,
                            row0_abs,
                            col3,
                            k0,
                            K,
                            N,
                            row0_scale_exp,
                        )
                        v04 = decode_mxfp4_unshuffle_value[
                            USE_VALUE_SWIZZLE=USE_VALUE_SWIZZLE
                        ](
                            w_blocks_ptr,
                            w_blocks_stride0,
                            w_blocks_stride1,
                            w_blocks_stride2,
                            expert_id,
                            base_m2,
                            base_k2,
                            row1_rel,
                            row1_abs,
                            col0,
                            k0,
                            K,
                            N,
                            row1_scale_exp,
                        )
                        v05 = decode_mxfp4_unshuffle_value[
                            USE_VALUE_SWIZZLE=USE_VALUE_SWIZZLE
                        ](
                            w_blocks_ptr,
                            w_blocks_stride0,
                            w_blocks_stride1,
                            w_blocks_stride2,
                            expert_id,
                            base_m2,
                            base_k2,
                            row1_rel,
                            row1_abs,
                            col1,
                            k0,
                            K,
                            N,
                            row1_scale_exp,
                        )
                        v06 = decode_mxfp4_unshuffle_value[
                            USE_VALUE_SWIZZLE=USE_VALUE_SWIZZLE
                        ](
                            w_blocks_ptr,
                            w_blocks_stride0,
                            w_blocks_stride1,
                            w_blocks_stride2,
                            expert_id,
                            base_m2,
                            base_k2,
                            row1_rel,
                            row1_abs,
                            col2,
                            k0,
                            K,
                            N,
                            row1_scale_exp,
                        )
                        v07 = decode_mxfp4_unshuffle_value[
                            USE_VALUE_SWIZZLE=USE_VALUE_SWIZZLE
                        ](
                            w_blocks_ptr,
                            w_blocks_stride0,
                            w_blocks_stride1,
                            w_blocks_stride2,
                            expert_id,
                            base_m2,
                            base_k2,
                            row1_rel,
                            row1_abs,
                            col3,
                            k0,
                            K,
                            N,
                            row1_scale_exp,
                        )

                    var idx = m_mma + k_mma * num_m_mmas
                    a_frags[idx, 0] = rebind[type_of(a_frags[idx, 0])](
                        SIMD[BF16, 8](
                            v00,
                            v04,
                            v01,
                            v05,
                            v03,
                            v07,
                            v02,
                            v06,
                            )
                    )

            warpgroup_fence(c_reg_tile)
            wgmma.arrive()
            wgmma.wgmma(
                a_frag_tile=a_frag_tile,
                b_smem_tile=B_s,
                c_reg_tile=c_reg_tile,
            )
            wgmma.commit_group()
            warpgroup_fence(c_reg_tile)
            wgmma.wait_group()

            if warp_group_thread_idx == 0:
                _ = empty_mbar[Int(slot)].arrive()
            read_state.step()

        wgmma.wait_group()

        # Epilogue: store C^T into output (M x N).
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
                    var warp_in_wg = (
                        Int(warp_group_thread_idx) // Int(WARP_SIZE)
                    )
                    var n_in_cta = (
                        local_wg_idx * (num_m_mmas * WGMMA_M)
                        + m_mma * WGMMA_M
                        + warp_in_wg * warp_rows
                        + row_in_warp
                    )
                    var global_col = n0 + n_in_cta
                    if global_col >= N:
                        continue

                    @parameter
                    for c_it in range(col_iters):
                        var col_pair = lane_col + c_it * 4
                        var row_base = row0 + n_mma * WGMMA_N + col_pair * 2
                        var row1 = row_base + 1
                        var frag_idx = c_it * row_iters + r_it
                        var v2 = c_pairs[0, frag_idx]

                        if row_base < seg_end and row_base < P:
                            out_ptr[
                                row_base * out_stride0
                                + global_col * out_stride1
                            ] = v2[0].cast[BF16]()
                        if row1 < seg_end and row1 < P:
                            out_ptr[
                                row1 * out_stride0 + global_col * out_stride1
                            ] = v2[1].cast[BF16]()
