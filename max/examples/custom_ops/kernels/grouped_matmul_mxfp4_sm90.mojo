# grouped_matmul_mxfp4_sm90.mojo
#
# SM90-optimized MXFP4 grouped matmul for ModuleV3 GPT‑OSS MoE expert GEMMs.
#
# Goal: match upstream SM90 grouped matmul intent:
# - BF16 activations in/out
# - MXFP4 expert weights stored as packed FP4 (uint8) + E8M0 scales (uint8)
# - Decode weights inside the K-tile loop into BF16 shared memory
# - FP32 only for register accumulators (+ tiny scalar epilogue temps)
#
# Design: producer/consumer warpgroup pipeline with mbarrier barriers.
# - Warpgroup 0: producer (loads A tile, decodes B tile)
# - Warpgroup 1..N: consumers (WGMMA + epilogue store)

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

from .mxfp4_decode import (
    decode_mxfp4_packbits_u32_to_8xbf16_scaled,
    e8m0_to_bf16_bits,
)


comptime BF16 = DType.bfloat16
comptime F32 = DType.float32
comptime U8 = DType.uint8
comptime U32 = DType.uint32
comptime U64 = DType.uint64
comptime I32 = DType.int32

comptime BYTES_PER_BLOCK = 16
comptime VALUES_PER_BLOCK = 32


@always_inline
fn _u32_from_u8x4(b0: UInt8, b1: UInt8, b2: UInt8, b3: UInt8) -> UInt32:
    return (
        UInt32(b0)
        | (UInt32(b1) << UInt32(8))
        | (UInt32(b2) << UInt32(16))
        | (UInt32(b3) << UInt32(24))
    )


@always_inline
fn _apply_xor_swizzle(swizzle: Swizzle, idx: Int) -> Int:
    var base = idx % swizzle.size()
    return swizzle(base) + (idx - base)


@parameter
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
fn grouped_matmul_mxfp4_bf16_wgmma_sm90_pipeline[
    BM: Int = 128,
    BN: Int = 256,
    BK: Int = 64,
    WGMMA_M: Int = 64,
    WGMMA_N: Int = 256,
    WGMMA_K: Int = 16,
    NUM_WARP_GROUPS: Int = 2,  # number of consumer warp groups
    NUM_PIPELINE_STAGES: Int = 2,
](
    a_tma_op: TMATensorTile[
        BF16,
        Layout.row_major(BM, BK),
        Layout.row_major(BM, BK),
    ],
    P: Int,
    K: Int,
    expert_start_ptr: UnsafePointer[UInt32, MutAnyOrigin],
    expert_ids_ptr: UnsafePointer[Int32, MutAnyOrigin],
    num_active_experts: Int,
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

    var n0 = Int(block_idx.x) * BN
    var row0 = seg_start + Int(block_idx.y) * BM
    if row0 >= seg_end or row0 >= P:
        return

    var warp_group_idx, warp_group_thread_idx = divmod(
        thread_idx.x, UInt(WARPGROUP_SIZE)
    )
    var local_wg_idx = Int(warp_group_idx) - 1  # consumer-local index

    # Shared tiles (BF16 only).
    # Use 128B swizzle for A so TMA can copy the entire BK=64 slice in one
    # transaction per tile. B is also 128B-swizzled so consumers stay on the
    # fast WGMMA path; because B is software-produced (decoded), producer
    # stores must apply the XOR swizzle mapping on write.
    comptime a_swizzle = TensorMapSwizzle.SWIZZLE_128B
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
    __comptime_assert (a_stage_bytes % 1024) == 0
    __comptime_assert (b_stage_bytes % 1024) == 0
    __comptime_assert (stage_bytes % 1024) == 0

    var smem = external_memory[
        Scalar[U8],
        address_space = AddressSpace.SHARED,
        alignment=1024,
        name="mxfp4_grouped_matmul_pipeline_smem",
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
            # `expect_bytes()` uses `mbarrier.arrive.expect_tx`, which counts
            # as an arrival. We still need the producer warpgroup to arrive
            # *after* it finishes decoding B into shared; otherwise consumers
            # may observe partially-written tiles and produce NaNs.
            full_mbar[i].init(Int32(Int(WARPGROUP_SIZE) + 1))
            empty_mbar[i].init(Int32(NUM_WARP_GROUPS))

    # Ensure all threads see initialized barriers before divergence.
    barrier()

    var w_blocks_u64 = w_blocks_ptr.address_space_cast[
        AddressSpace.GLOBAL
    ]().bitcast[Scalar[U64]]()

    # WGMMA accumulator shape (FP32 in registers only; allocated in consumers).
    comptime num_m_mmas_total = BM // WGMMA_M
    comptime num_m_mmas = num_m_mmas_total // NUM_WARP_GROUPS
    comptime num_n_mmas = BN // WGMMA_N
    comptime c_frag_size = (WGMMA_M * WGMMA_N) // 128

    comptime a_chunks = BK // 4

    comptime warp_rows = WGMMA_M // 4
    comptime row_iters = warp_rows // 8
    comptime col_iters = (WGMMA_N // 2) // 4

    # Consumer warpgroups pre-arrive empty barriers to "prime" the pipeline.
    if warp_group_idx != 0:
        if warp_group_thread_idx == 0:

            @parameter
            for i in range(NUM_PIPELINE_STAGES):
                _ = empty_mbar[i].arrive()

    var num_k_tiles = ceildiv(K, BK)

    if warp_group_idx == 0:
        # Producer warpgroup: load A and decode B into the next free stage.
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
            comptime vec_len = simd_width_of[BF16]()
            __comptime_assert vec_len == 8

            var k0 = k_tile * BK
            var kb0 = k0 // VALUES_PER_BLOCK
            comptime blocks_per_tile = BK // VALUES_PER_BLOCK

            # Load A tile (BF16) via TMA and combine its completion with the
            # producer arrival count on `full_mbar`.
            if warp_group_thread_idx == 0:
                full_mbar[Int(slot)].expect_bytes(Int32(a_bytes))
                a_tma_op.async_copy(
                    A_s,
                    full_mbar[Int(slot)],
                    (UInt(k0), UInt(row0)),
                )

            # Decode B tile (MXFP4 -> BF16 in shared).
            # Work is distributed across producer warpgroup threads with
            # coalesced global loads: keep `kb` fixed so each warp loads a
            # contiguous span of columns in `w_blocks/w_scales`.
            var c = Int(warp_group_thread_idx)
            if c < BN:
                var col = n0 + c

                @parameter
                for block_in_tile in range(blocks_per_tile):
                    var kb = kb0 + block_in_tile
                    var k_tile_base = block_in_tile * VALUES_PER_BLOCK

                    var scale_exp: UInt8 = 0
                    if col < N and kb < Kblocks:
                        scale_exp = w_scales_ptr[
                            ((expert_id * Kblocks + kb) * N) + col
                        ]
                    var scale = e8m0_to_bf16_bits(scale_exp)

                    var packed0: SIMD[U8, 8] = SIMD[U8, 8](0)
                    var packed1: SIMD[U8, 8] = SIMD[U8, 8](0)
                    if col < N and kb < Kblocks:
                        var packed_base = (
                            ((expert_id * Kblocks + kb) * N) + col
                        ) * BYTES_PER_BLOCK
                        var base_u64 = packed_base // 8
                        packed0 = bitcast[DType.uint8, 8](
                            UInt64(w_blocks_u64.load[alignment=8](base_u64 + 0))
                        )
                        packed1 = bitcast[DType.uint8, 8](
                            UInt64(w_blocks_u64.load[alignment=8](base_u64 + 1))
                        )

                    @parameter
                    for group in range(2):
                        var byte_base = group * 4
                        var k_base = k_tile_base + group * 8
                        var p = _u32_from_u8x4(
                            UInt8(packed0[byte_base + 0]),
                            UInt8(packed0[byte_base + 1]),
                            UInt8(packed0[byte_base + 2]),
                            UInt8(packed0[byte_base + 3]),
                        )
                        var outv = decode_mxfp4_packbits_u32_to_8xbf16_scaled(
                            p, scale
                        )
                        var ridx = B_smem_tile.idx_list_t[2](fill=0)
                        ridx[0] = c
                        ridx[1] = k_base
                        var lin = B_smem_tile._get_offset(b_strides, ridx)
                        var swz = _apply_xor_swizzle(b_smem_swizzle, lin)
                        B_data.store[width=8, alignment=16](swz, outv)

                    @parameter
                    for group in range(2):
                        var byte_base = group * 4
                        var k_base = k_tile_base + 16 + group * 8
                        var p = _u32_from_u8x4(
                            UInt8(packed1[byte_base + 0]),
                            UInt8(packed1[byte_base + 1]),
                            UInt8(packed1[byte_base + 2]),
                            UInt8(packed1[byte_base + 3]),
                        )
                        var outv = decode_mxfp4_packbits_u32_to_8xbf16_scaled(
                            p, scale
                        )
                        var ridx = B_smem_tile.idx_list_t[2](fill=0)
                        ridx[0] = c
                        ridx[1] = k_base
                        var lin = B_smem_tile._get_offset(b_strides, ridx)
                        var swz = _apply_xor_swizzle(b_smem_swizzle, lin)
                        B_data.store[width=8, alignment=16](swz, outv)

            # Signal stage ready for consumers (all producer threads must arrive).
            # Ensure software-written B tile stores are visible before consumers
            # pass the acquire-wait and issue WGMMA loads.
            threadfence[Scope.BLOCK]()
            _ = full_mbar[Int(slot)].arrive()
            write_state.step()

    else:
        # Consumer warpgroups: WGMMA + store.
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

        # Epilogue mapping (stores output in BF16).
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

        # Ensure all WGMMA groups completed before epilogue stores.
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
                            out_ptr[global_row * N + col0] = v2[0].cast[BF16]()
                        if col1 < N:
                            out_ptr[global_row * N + col1] = v2[1].cast[BF16]()


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
    comptime c_frag_size = (WGMMA_M * WGMMA_N) // 128

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
                    if col < N and kb < Kblocks:
                        scale_exp = w_scales_ptr[
                            expert_id * w_scales_stride0
                            + kb * w_scales_stride1
                            + col * w_scales_stride2
                        ]
                    var scale = e8m0_to_bf16_bits(scale_exp)

                    var packed0: SIMD[U8, 8] = SIMD[U8, 8](0)
                    var packed1: SIMD[U8, 8] = SIMD[U8, 8](0)
                    if col < N and kb < Kblocks:
                        var packed_base = (
                            expert_id * w_blocks_stride0
                            + kb * w_blocks_stride1
                            + col * w_blocks_stride2
                        )
                        packed0 = w_blocks_ptr.load[width=8, alignment=1](
                            packed_base
                        )
                        packed1 = w_blocks_ptr.load[width=8, alignment=1](
                            packed_base + 8
                        )

                    @parameter
                    for chunk in range(2):
                        var p = _u32_from_u8x4(
                            UInt8(packed0[chunk * 4 + 0]),
                            UInt8(packed0[chunk * 4 + 1]),
                            UInt8(packed0[chunk * 4 + 2]),
                            UInt8(packed0[chunk * 4 + 3]),
                        )
                        var o = decode_mxfp4_packbits_u32_to_8xbf16_scaled(
                            p, scale
                        )

                        @parameter
                        for t in range(8):
                            var kk = k_tile_base + chunk * 8 + t
                            var ridx = B_smem_tile.idx_list_t[2](fill=0)
                            ridx[0] = c
                            ridx[1] = kk
                            var lin = B_smem_tile._get_offset(b_strides, ridx)
                            var swz = _apply_xor_swizzle(b_smem_swizzle, lin)
                            B_data[swz] = o[t]

                    @parameter
                    for chunk in range(2):
                        var p = _u32_from_u8x4(
                            UInt8(packed1[chunk * 4 + 0]),
                            UInt8(packed1[chunk * 4 + 1]),
                            UInt8(packed1[chunk * 4 + 2]),
                            UInt8(packed1[chunk * 4 + 3]),
                        )
                        var o = decode_mxfp4_packbits_u32_to_8xbf16_scaled(
                            p, scale
                        )

                        @parameter
                        for t in range(8):
                            var kk = k_tile_base + 16 + chunk * 8 + t
                            var ridx = B_smem_tile.idx_list_t[2](fill=0)
                            ridx[0] = c
                            ridx[1] = kk
                            var lin = B_smem_tile._get_offset(b_strides, ridx)
                            var swz = _apply_xor_swizzle(b_smem_swizzle, lin)
                            B_data[swz] = o[t]

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
        max_num_tokens_per_expert: UInt32,
        num_active_experts: UInt32,
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if is_cpu[target]():
            raise Error("mxfp4_grouped_matmul_ragged_bf16: GPU only")

        var P = a.dim_size(0)
        var K = a.dim_size(1)
        var N = c.dim_size(1)
        var a_stride0 = a.stride_length[0]()
        var a_stride1 = a.stride_length[1]()
        var out_stride0 = c.stride_length[0]()
        var out_stride1 = c.stride_length[1]()
        var w_blocks_stride0 = w_blocks.stride_length[0]()
        var w_blocks_stride1 = w_blocks.stride_length[1]()
        var w_blocks_stride2 = w_blocks.stride_length[2]()
        var w_scales_stride0 = w_scales.stride_length[0]()
        var w_scales_stride1 = w_scales.stride_length[1]()
        var w_scales_stride2 = w_scales.stride_length[2]()
        var Kblocks = w_blocks.dim_size(1)

        if P == 0 or K == 0 or N == 0:
            return
        # Match upstream grouped_matmul_ragged fast-path assumptions:
        # - A activations are contiguous row-major so TMA can be used safely.
        if a_stride1 != 1 or a_stride0 != K:
            raise Error(
                "mxfp4_grouped_matmul_ragged_bf16: activations must be"
                " contiguous"
            )
        if out_stride1 != 1 or out_stride0 != N:
            raise Error(
                "mxfp4_grouped_matmul_ragged_bf16: output must be contiguous"
            )

        if Kblocks * VALUES_PER_BLOCK != K:
            raise Error(
                "mxfp4_grouped_matmul_ragged_bf16: K must be divisible by 32"
                " and match w_blocks Kblocks"
            )

        var gpu_ctx = ctx.get_device_context()
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

        # IMPORTANT: `mo.moe.create.indices` compacts `expert_start_indices` and
        # `expert_ids` so only the first `num_active_experts` entries are valid.
        # Reading beyond that yields undefined values and can cause OOB reads on
        # expert weights (NaNs -> invalid router indices -> gather OOB).
        var max_M = Int(max_num_tokens_per_expert)
        var n_active = Int(num_active_experts)
        if max_M <= 0 or n_active <= 0:
            return
        if max_M > P:
            max_M = P

        var grid_z = n_active
        var max_experts = expert_ids.dim_size(0)
        var start_len = expert_start_indices.dim_size(0)
        if start_len <= 1:
            return
        var start_max = start_len - 1
        if max_experts > start_max:
            max_experts = start_max
        if grid_z > max_experts:
            grid_z = max_experts
        if grid_z <= 0:
            return

        # Deterministic baseline: 1 producer warpgroup + 1 consumer warpgroup.
        # Keep BN=128 so producer has 1 thread per output column.
        comptime BN = 128
        comptime BM = 64
        comptime BK = 64
        comptime NUM_WARP_GROUPS = 1
        comptime NUM_PIPELINE_STAGES = 2
        var grid_x = ceildiv(N, BN)
        var grid_y = ceildiv(max_M, BM)
        if grid_x == 0 or grid_y == 0:
            return

        # The fast path assumes checkpoint-prepacked contiguous expert weights:
        #   w_blocks: [E, K/32, N, 16] row-major contiguous
        #   w_scales: [E, K/32, N] row-major contiguous
        # since the kernel uses fixed-stride pointer math for coalesced loads.
        if (
            w_blocks_stride2 != BYTES_PER_BLOCK
            or w_blocks.stride_length[3]() != 1
        ):
            raise Error("w_blocks must be contiguous with inner dims [N,16]")
        if w_scales_stride2 != 1:
            raise Error("w_scales must be contiguous with inner dim [N]")
        if w_blocks_stride1 != N * BYTES_PER_BLOCK:
            raise Error("w_blocks must be contiguous with dim stride1 == N*16")
        if w_blocks_stride0 != Kblocks * N * BYTES_PER_BLOCK:
            raise Error(
                "w_blocks must be contiguous with dim stride0 == Kblocks*N*16"
            )
        if w_scales_stride1 != N:
            raise Error("w_scales must be contiguous with dim stride1 == N")
        if w_scales_stride0 != Kblocks * N:
            raise Error(
                "w_scales must be contiguous with dim stride0 == Kblocks*N"
            )

        # Build a TMA op for A so the producer can async-copy BF16 tiles into
        # SWIZZLE_128B shared memory.
        comptime a_gmem_layout = Layout.row_major[2]()
        var a_shape = IndexList[2](P, K)
        var a_runtime_layout = RuntimeLayout[a_gmem_layout].row_major(a_shape)
        var a_tensor = LayoutTensor[
            BF16,
            a_gmem_layout,
            MutAnyOrigin,
            address_space = AddressSpace.GENERIC,
        ](a.unsafe_ptr(), a_runtime_layout)
        comptime a_swizzle = TensorMapSwizzle.SWIZZLE_128B
        var a_tma_op = create_tma_tile[Index(BM, BK), swizzle_mode=a_swizzle](
            gpu_ctx, a_tensor
        )

        comptime kernel = grouped_matmul_mxfp4_bf16_wgmma_sm90_pipeline[
            BM=BM,
            BN=BN,
            BK=BK,
            WGMMA_M=64,
            WGMMA_N=128,
            WGMMA_K=16,
            NUM_WARP_GROUPS=NUM_WARP_GROUPS,
            NUM_PIPELINE_STAGES=NUM_PIPELINE_STAGES,
        ]

        comptime a_smem_layout = tile_layout_k_major[
            BF16, BM, BK, TensorMapSwizzle.SWIZZLE_128B
        ]()
        comptime b_smem_layout = tile_layout_k_major[
            BF16, BN, BK, TensorMapSwizzle.SWIZZLE_128B
        ]()
        comptime a_bytes = a_smem_layout.size() * 2
        comptime b_bytes = b_smem_layout.size() * 2
        comptime a_stage_bytes = ((a_bytes + 255) // 256) * 256
        comptime b_stage_bytes = ((b_bytes + 255) // 256) * 256
        comptime stage_bytes = a_stage_bytes + b_stage_bytes
        comptime smem_use = NUM_PIPELINE_STAGES * stage_bytes

        gpu_ctx.enqueue_function_checked[kernel, kernel](
            a_tma_op,
            P,
            K,
            expert_start_dev,
            expert_ids_dev,
            grid_z,
            w_blocks_dev,
            w_scales_dev,
            Kblocks,
            c_dev,
            N,
            grid_dim=(grid_x, grid_y, grid_z),
            block_dim=(Int(WARPGROUP_SIZE) * (NUM_WARP_GROUPS + 1), 1, 1),
            shared_mem_bytes=Int(smem_use),
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                Int(smem_use)
            ),
        )
