from .grouped_matmul_sm90_common import *
from .grouped_matmul_sm90_common import load_swizzled_pack_u32
from .grouped_matmul_sm90_common import (
    apply_xor_swizzle,
)


@parameter
fn grouped_matmul_mxfp4_bf16_wgmma_sm90_pipeline_swload[
    BM: Int = 128,
    BN: Int = 256,
    BK: Int = 64,
    WGMMA_M: Int = 64,
    WGMMA_N: Int = 256,
    WGMMA_K: Int = 16,
    NUM_WARP_GROUPS: Int = 2,  # number of consumer warp groups
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
    var expert_id = Int(expert_ids_ptr[expert_idx])
    if expert_id < 0:
        return

    var seg_start = Int(expert_start_ptr[expert_idx])
    var seg_end = Int(expert_start_ptr[expert_idx + 1])
    if seg_start >= seg_end:
        return
    # Grouped tile scheduler guard: skip tiles beyond this expert's segment.
    var seg_len = seg_end - seg_start
    var max_tiles = ceildiv(seg_len, BM)
    if Int(block_idx.y) >= max_tiles:
        return

    var n0 = Int(block_idx.x) * BN
    var row0 = seg_start + Int(block_idx.y) * BM
    if row0 >= seg_end or row0 >= P:
        return

    var warp_group_idx, warp_group_thread_idx = divmod(
        thread_idx.x, UInt(WARPGROUP_SIZE)
    )
    var local_wg_idx = Int(warp_group_idx) - 1  # consumer-local index

    # Shared tiles (BF16 only).
    #
    # Note: `tile_layout_*[..., SWIZZLE_128B]` selects the canonical core-matrix
    # row width used by WGMMA descriptors, but software-produced tiles must still
    # apply the same XOR swizzle mapping on store. See `make_swizzle` and
    # `copy_local_to_shared` in upstream (read-only) references.
    comptime a_swizzle = TensorMapSwizzle.SWIZZLE_NONE
    comptime b_swizzle = TensorMapSwizzle.SWIZZLE_128B
    comptime a_smem_layout = tile_layout_k_major[BF16, BM, BK, a_swizzle]()
    comptime b_smem_layout = tile_layout_k_major[BF16, BN, BK, b_swizzle]()
    comptime b_smem_swizzle = make_swizzle[BF16, b_swizzle]()
    comptime B_smem_tile = LayoutTensor[
        BF16,
        b_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=1024,
    ]

    comptime a_bytes = a_smem_layout.size() * 2
    comptime b_bytes = b_smem_layout.size() * 2
    comptime a_stage_bytes = ((a_bytes + 255) // 256) * 256
    comptime b_stage_bytes = ((b_bytes + 255) // 256) * 256
    comptime stage_bytes = a_stage_bytes + b_stage_bytes

    var smem = external_memory[
        Scalar[U8],
        address_space = AddressSpace.SHARED,
        alignment=1024,
        name="mxfp4_grouped_matmul_pipeline_smem_swload",
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
            var A_s = LayoutTensor[
                BF16,
                a_smem_layout,
                MutAnyOrigin,
                address_space = AddressSpace.SHARED,
                alignment=1024,
            ](stage_base.bitcast[Scalar[BF16]]())
            var B_s = B_smem_tile(
                (stage_base + a_stage_bytes).bitcast[Scalar[BF16]]()
            )
            var B_data = (stage_base + a_stage_bytes).bitcast[Scalar[BF16]]()
            var b_strides = B_s.runtime_layout.stride.value

            var k0 = k_tile * BK
            var kb0 = k0 // VALUES_PER_BLOCK
            comptime blocks_per_tile = BK // VALUES_PER_BLOCK

            # Load A tile into shared (zero-fill out-of-range rows).
            for idx in range(
                Int(warp_group_thread_idx), BM * BK, Int(WARPGROUP_SIZE)
            ):
                var r = idx // BK
                var k = idx - r * BK

                var global_row = row0 + r
                var global_col = k0 + k

                var v = Scalar[BF16](0)
                if global_row < seg_end and global_row < P and global_col < K:
                    v = a_ptr[
                        global_row * a_stride0 + global_col * a_stride1
                    ].cast[BF16]()

                A_s[r, k] = v

            # Decode B tile (MXFP4 -> BF16 in shared).
            # BN can exceed WARPGROUP_SIZE (e.g. BN=256, WARPGROUP_SIZE=128), so
            # each producer thread decodes multiple columns deterministically.
            @parameter
            for col_it in range(ceildiv(BN, Int(WARPGROUP_SIZE))):
                var c = Int(warp_group_thread_idx) + col_it * Int(
                    WARPGROUP_SIZE
                )
                if c >= BN:
                    continue

                var col = n0 + c

                @parameter
                for block_in_tile in range(blocks_per_tile):
                    var kb = kb0 + block_in_tile
                    var k_tile_base = block_in_tile * VALUES_PER_BLOCK

                    var scale_exp: UInt8 = 0
                    var words = SIMD[U32, 4](0)
                    var base16 = 0
                    var kbyte_base = kb * BYTES_PER_BLOCK
                    if col < N and kb < Kblocks:
                        var idx = hopper_scale_swizzle_index_fast(col, kb)
                        scale_exp = w_scales_ptr[
                            expert_id * w_scales_stride0
                            + idx[0] * w_scales_stride1
                            + idx[1] * w_scales_stride2
                        ]

                        @parameter
                        if USE_VALUE_SWIZZLE:
                            var idx0 = hopper_value_swizzle_index(
                                col, kbyte_base
                            )
                            var k2 = idx0[1]
                            base16 = k2 & ~15
                            var base = (
                                expert_id * w_blocks_stride0
                                + idx0[0] * w_blocks_stride1
                                + base16 * w_blocks_stride2
                            )
                            var w_u32 = w_blocks_ptr.address_space_cast[
                                AddressSpace.GLOBAL
                            ]().bitcast[Scalar[U32]]()
                            words = w_u32.load[
                                width=4, alignment=16
                            ](base >> 2)
                        else:
                            var packed_base = (
                                expert_id * w_blocks_stride0
                                + kb * w_blocks_stride1
                                + col * w_blocks_stride2
                            )
                            var w_u32 = w_blocks_ptr.address_space_cast[
                                AddressSpace.GLOBAL
                            ]().bitcast[Scalar[U32]]()
                            words = w_u32.load[
                                width=4, alignment=16
                            ](packed_base >> 2)

                    @parameter
                    for chunk in range(4):
                        var p: UInt32 = 0
                        if col < N and kb < Kblocks:
                            @parameter
                            if USE_VALUE_SWIZZLE:
                                var kbyte = kbyte_base + chunk * 4
                                var idx = hopper_value_swizzle_index(
                                    col, kbyte
                                )
                                var base_k2 = idx[1] & ~3
                                var word_idx = (base_k2 - base16) >> 2
                                p = words[word_idx]
                            else:
                                p = words[chunk]

                        var o = decode_mxfp4_packbits_u32_to_8xbf16_scaled_e8m0(
                            p, scale_exp
                        )

                        var kk = k_tile_base + chunk * 8
                        var ridx = B_smem_tile.idx_list_t[2](fill=0)
                        ridx[0] = c
                        ridx[1] = kk
                        var lin = B_smem_tile._get_offset(b_strides, ridx)
                        var swz = apply_xor_swizzle(b_smem_swizzle, lin)
                        B_data.store[width=8, alignment=16](swz, o)

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
            var A_s = LayoutTensor[
                BF16,
                a_smem_layout,
                MutAnyOrigin,
                address_space = AddressSpace.SHARED,
                alignment=1024,
            ](stage_base.bitcast[Scalar[BF16]]())
            var B_s = B_smem_tile(
                (stage_base + a_stage_bytes).bitcast[Scalar[BF16]]()
            )

            warpgroup_fence(c_reg_tile)
            wgmma.arrive()
            wgmma.wgmma[num_warp_groups=NUM_WARP_GROUPS](
                a_smem_tile=A_s,
                b_smem_tile=B_s,
                c_reg_tile=c_reg_tile,
                wg_idx=local_wg_idx,
            )
            wgmma.commit_group()
            warpgroup_fence(c_reg_tile)
            wgmma.wait_group()

            if warp_group_thread_idx == 0:
                _ = empty_mbar[Int(slot)].arrive()
            read_state.step()

        wgmma.wait_group()

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
                        local_wg_idx * (num_m_mmas * WGMMA_M)
                        + m_mma * WGMMA_M
                        + (Int(warp_id() & UInt(3)) * warp_rows)
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
                            out_ptr[
                                global_row * out_stride0 + col0 * out_stride1
                            ] = v2[0].cast[BF16]()
                        if col1 < N:
                            out_ptr[
                                global_row * out_stride0 + col1 * out_stride1
                            ] = v2[1].cast[BF16]()
