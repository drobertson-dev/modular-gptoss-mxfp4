# moe_mxfp4_ops.mojo
#
# Correctness-first MXFP4 MoE kernels + custom op entrypoints.
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
from gpu import barrier, block_dim, block_idx, thread_idx, warp_id
from gpu.host import DeviceBuffer
from gpu.host.info import is_cpu
from gpu.memory import AddressSpace
from layout.layout_tensor import Layout, LayoutTensor
from layout.tensor_core import TensorCore
from layout.tensor_core_async import (
    TensorCoreAsync,
    tile_layout_k_major,
    warpgroup_fence,
)
from memory import stack_allocation
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from utils.index import Index, IndexList

from .mxfp4_decode import decode_mxfp4_byte_to_2xbf16_e8m0, swiglu_pair


comptime BF16 = DType.bfloat16
comptime F16 = DType.float16
comptime F32 = DType.float32
comptime U8 = DType.uint8
comptime U32 = DType.uint32
comptime I32 = DType.int32

comptime BYTES_PER_BLOCK = 16
comptime VALUES_PER_BLOCK = 32
comptime TOPK = 4


@parameter
fn moe_w1_mxfp4_swiglu_wgmma[
    BM: Int = 64,
    BN_RAW: Int = 64,
    BK: Int = 64,
    WGMMA_M: Int = 64,
    WGMMA_N: Int = 64,
    WGMMA_K: Int = 16,
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
    constrained[WGMMA_M == 64, "SM90 WGMMA requires M=64 for this kernel"]()
    constrained[WGMMA_K in (8, 16, 32)]()
    constrained[BM % WGMMA_M == 0]()
    constrained[BN_RAW % WGMMA_N == 0]()
    constrained[BK % WGMMA_K == 0]()
    constrained[BK % VALUES_PER_BLOCK == 0, "BK must be a multiple of 32"]()

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

    # Routing: token ids for BM rows (pair_idx sorted by expert, TOPK packed).
    var token_ids_s = stack_allocation[
        BM, Scalar[U32], address_space = AddressSpace.SHARED
    ]()
    for r in range(Int(thread_idx.x), BM, Int(block_dim.x)):
        var global_row = row0 + r
        if global_row < seg_end and global_row < P:
            var pair_idx = Int(token_expert_order_ptr[global_row])
            token_ids_s[r] = UInt32(pair_idx // TOPK)
        else:
            token_ids_s[r] = UInt32(0)
    barrier()

    # Shared tiles in WGMMA-friendly layouts.
    comptime a_smem_layout = tile_layout_k_major[BF16, BM, BK]()
    var A_s0 = LayoutTensor[
        BF16,
        a_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()
    var A_s1 = LayoutTensor[
        BF16,
        a_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()

    # B is stored as [N, K] (transpose_b=True) for WGMMA.
    comptime b_smem_layout = tile_layout_k_major[BF16, BN_RAW, BK]()
    var B_s0 = LayoutTensor[
        BF16,
        b_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()
    var B_s1 = LayoutTensor[
        BF16,
        b_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()

    comptime num_m_mmas = BM // WGMMA_M
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
        transpose_b=True,
    ]()

    comptime blocks_per_tile = BK // VALUES_PER_BLOCK
    comptime bytes_per_col = BYTES_PER_BLOCK * blocks_per_tile

    # Preload first K tile into buffer 0.
    for idx in range(Int(thread_idx.x), BM * BK, Int(block_dim.x)):
        var r = idx // BK
        var kk = idx - r * BK
        var global_row = row0 + r
        var global_k = kk

        if global_row < seg_end and global_row < P and global_k < D:
            var tok = Int(token_ids_s[r][0])
            if tok < T:
                A_s0[r, kk] = x_ptr[tok * D + global_k]
            else:
                A_s0[r, kk] = 0
        else:
            A_s0[r, kk] = 0

    var kb0 = 0
    for idx in range(
        Int(thread_idx.x),
        BN_RAW * bytes_per_col,
        Int(block_dim.x),
    ):
        var c = idx // bytes_per_col
        var byte_off = idx - c * bytes_per_col
        var block_in_tile = byte_off // BYTES_PER_BLOCK
        var byte_in_block = byte_off - block_in_tile * BYTES_PER_BLOCK

        var n_raw = n_raw0 + c
        var r0 = block_in_tile * VALUES_PER_BLOCK + byte_in_block * 2
        var kb = kb0 + block_in_tile

        if n_raw < N_raw_total and kb < Kblocks:
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
            B_s0[c, r0 + 0] = v2[0]
            B_s0[c, r0 + 1] = v2[1]
        else:
            B_s0[c, r0 + 0] = 0
            B_s0[c, r0 + 1] = 0
    barrier()

    var num_k_tiles = ceildiv(D, BK)
    for k_tile in range(num_k_tiles):
        var use_buf0 = (k_tile & 1) == 0

        warpgroup_fence(c_reg_tile)
        wgmma.arrive()
        if use_buf0:
            wgmma.wgmma(
                a_smem_tile=A_s0, b_smem_tile=B_s0, c_reg_tile=c_reg_tile
            )
        else:
            wgmma.wgmma(
                a_smem_tile=A_s1, b_smem_tile=B_s1, c_reg_tile=c_reg_tile
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
                for idx in range(Int(thread_idx.x), BM * BK, Int(block_dim.x)):
                    var r = idx // BK
                    var kk = idx - r * BK
                    var global_row = row0 + r
                    var global_k = k0_next + kk

                    if global_row < seg_end and global_row < P and global_k < D:
                        var tok = Int(token_ids_s[r][0])
                        if tok < T:
                            A_s1[r, kk] = x_ptr[tok * D + global_k]
                        else:
                            A_s1[r, kk] = 0
                    else:
                        A_s1[r, kk] = 0

                for idx in range(
                    Int(thread_idx.x),
                    BN_RAW * bytes_per_col,
                    Int(block_dim.x),
                ):
                    var c = idx // bytes_per_col
                    var byte_off = idx - c * bytes_per_col
                    var block_in_tile = byte_off // BYTES_PER_BLOCK
                    var byte_in_block = (
                        byte_off - block_in_tile * BYTES_PER_BLOCK
                    )

                    var n_raw = n_raw0 + c
                    var r0 = (
                        block_in_tile * VALUES_PER_BLOCK + byte_in_block * 2
                    )
                    var kb = kb0_next + block_in_tile

                    if n_raw < N_raw_total and kb < Kblocks:
                        var scale_exp = w_scales_ptr[
                            ((expert_id * N_raw_total + n_raw) * Kblocks) + kb
                        ]
                        var packed = w_blocks_ptr[
                            (
                                (
                                    (expert_id * N_raw_total + n_raw) * Kblocks
                                    + kb
                                )
                                * BYTES_PER_BLOCK
                            )
                            + byte_in_block
                        ]
                        var v2 = decode_mxfp4_byte_to_2xbf16_e8m0(
                            packed, scale_exp
                        )
                        B_s1[c, r0 + 0] = v2[0]
                        B_s1[c, r0 + 1] = v2[1]
                    else:
                        B_s1[c, r0 + 0] = 0
                        B_s1[c, r0 + 1] = 0
            else:
                # Load next tile into buffer 0.
                for idx in range(Int(thread_idx.x), BM * BK, Int(block_dim.x)):
                    var r = idx // BK
                    var kk = idx - r * BK
                    var global_row = row0 + r
                    var global_k = k0_next + kk

                    if global_row < seg_end and global_row < P and global_k < D:
                        var tok = Int(token_ids_s[r][0])
                        if tok < T:
                            A_s0[r, kk] = x_ptr[tok * D + global_k]
                        else:
                            A_s0[r, kk] = 0
                    else:
                        A_s0[r, kk] = 0

                for idx in range(
                    Int(thread_idx.x),
                    BN_RAW * bytes_per_col,
                    Int(block_dim.x),
                ):
                    var c = idx // bytes_per_col
                    var byte_off = idx - c * bytes_per_col
                    var block_in_tile = byte_off // BYTES_PER_BLOCK
                    var byte_in_block = (
                        byte_off - block_in_tile * BYTES_PER_BLOCK
                    )

                    var n_raw = n_raw0 + c
                    var r0 = (
                        block_in_tile * VALUES_PER_BLOCK + byte_in_block * 2
                    )
                    var kb = kb0_next + block_in_tile

                    if n_raw < N_raw_total and kb < Kblocks:
                        var scale_exp = w_scales_ptr[
                            ((expert_id * N_raw_total + n_raw) * Kblocks) + kb
                        ]
                        var packed = w_blocks_ptr[
                            (
                                (
                                    (expert_id * N_raw_total + n_raw) * Kblocks
                                    + kb
                                )
                                * BYTES_PER_BLOCK
                            )
                            + byte_in_block
                        ]
                        var v2 = decode_mxfp4_byte_to_2xbf16_e8m0(
                            packed, scale_exp
                        )
                        B_s0[c, r0 + 0] = v2[0]
                        B_s0[c, r0 + 1] = v2[1]
                    else:
                        B_s0[c, r0 + 0] = 0
                        B_s0[c, r0 + 1] = 0

            barrier()

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
                    m_mma * WGMMA_M + Int(warp_id()) * warp_rows + row_in_warp
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
    constrained[WGMMA_M == 64, "SM90 WGMMA requires M=64 for this kernel"]()
    constrained[WGMMA_K in (8, 16, 32)]()
    constrained[BM % WGMMA_M == 0]()
    constrained[BN % WGMMA_N == 0]()
    constrained[BK % WGMMA_K == 0]()
    constrained[BK % VALUES_PER_BLOCK == 0, "BK must be a multiple of 32"]()

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

    # Shared tiles in WGMMA-friendly layouts.
    comptime a_smem_layout = tile_layout_k_major[BF16, BM, BK]()
    var A_s0 = LayoutTensor[
        BF16,
        a_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()
    var A_s1 = LayoutTensor[
        BF16,
        a_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()

    comptime b_smem_layout = tile_layout_k_major[BF16, BN, BK]()
    var B_s0 = LayoutTensor[
        BF16,
        b_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()
    var B_s1 = LayoutTensor[
        BF16,
        b_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ].stack_allocation()

    comptime num_m_mmas = BM // WGMMA_M
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

    comptime blocks_per_tile = BK // VALUES_PER_BLOCK
    comptime bytes_per_col = BYTES_PER_BLOCK * blocks_per_tile

    # Preload first K tile into buffer 0.
    for idx in range(Int(thread_idx.x), BM * BK, Int(block_dim.x)):
        var r = idx // BK
        var kk = idx - r * BK
        var global_row = row0 + r
        var global_k = kk
        if global_row < seg_end and global_row < P and global_k < I:
            A_s0[r, kk] = h_ptr[global_row * I + global_k]
        else:
            A_s0[r, kk] = 0

    var kb0 = 0
    for idx in range(
        Int(thread_idx.x),
        BN * bytes_per_col,
        Int(block_dim.x),
    ):
        var c = idx // bytes_per_col
        var byte_off = idx - c * bytes_per_col
        var block_in_tile = byte_off // BYTES_PER_BLOCK
        var byte_in_block = byte_off - block_in_tile * BYTES_PER_BLOCK

        var col = n0 + c
        var r0 = block_in_tile * VALUES_PER_BLOCK + byte_in_block * 2
        var kb = kb0 + block_in_tile

        if col < D and kb < Kblocks:
            var scale_exp = w_scales_ptr[((expert_id * D + col) * Kblocks) + kb]
            var packed = w_blocks_ptr[
                (((expert_id * D + col) * Kblocks + kb) * BYTES_PER_BLOCK)
                + byte_in_block
            ]
            var v2 = decode_mxfp4_byte_to_2xbf16_e8m0(packed, scale_exp)
            B_s0[c, r0 + 0] = v2[0]
            B_s0[c, r0 + 1] = v2[1]
        else:
            B_s0[c, r0 + 0] = 0
            B_s0[c, r0 + 1] = 0
    barrier()

    var num_k_tiles = ceildiv(I, BK)
    for k_tile in range(num_k_tiles):
        var use_buf0 = (k_tile & 1) == 0

        warpgroup_fence(c_reg_tile)
        wgmma.arrive()
        if use_buf0:
            wgmma.wgmma(
                a_smem_tile=A_s0, b_smem_tile=B_s0, c_reg_tile=c_reg_tile
            )
        else:
            wgmma.wgmma(
                a_smem_tile=A_s1, b_smem_tile=B_s1, c_reg_tile=c_reg_tile
            )
        wgmma.commit_group()
        warpgroup_fence(c_reg_tile)

        if k_tile + 1 < num_k_tiles:
            wgmma.wait_group[1]()

            var k0_next = (k_tile + 1) * BK
            var kb0_next = k0_next // VALUES_PER_BLOCK

            if use_buf0:
                # Load next tile into buffer 1.
                for idx in range(Int(thread_idx.x), BM * BK, Int(block_dim.x)):
                    var r = idx // BK
                    var kk = idx - r * BK
                    var global_row = row0 + r
                    var global_k = k0_next + kk
                    if global_row < seg_end and global_row < P and global_k < I:
                        A_s1[r, kk] = h_ptr[global_row * I + global_k]
                    else:
                        A_s1[r, kk] = 0

                for idx in range(
                    Int(thread_idx.x),
                    BN * bytes_per_col,
                    Int(block_dim.x),
                ):
                    var c = idx // bytes_per_col
                    var byte_off = idx - c * bytes_per_col
                    var block_in_tile = byte_off // BYTES_PER_BLOCK
                    var byte_in_block = (
                        byte_off - block_in_tile * BYTES_PER_BLOCK
                    )

                    var col = n0 + c
                    var r0 = (
                        block_in_tile * VALUES_PER_BLOCK + byte_in_block * 2
                    )
                    var kb = kb0_next + block_in_tile

                    if col < D and kb < Kblocks:
                        var scale_exp = w_scales_ptr[
                            ((expert_id * D + col) * Kblocks) + kb
                        ]
                        var packed = w_blocks_ptr[
                            (
                                ((expert_id * D + col) * Kblocks + kb)
                                * BYTES_PER_BLOCK
                            )
                            + byte_in_block
                        ]
                        var v2 = decode_mxfp4_byte_to_2xbf16_e8m0(
                            packed, scale_exp
                        )
                        B_s1[c, r0 + 0] = v2[0]
                        B_s1[c, r0 + 1] = v2[1]
                    else:
                        B_s1[c, r0 + 0] = 0
                        B_s1[c, r0 + 1] = 0
            else:
                # Load next tile into buffer 0.
                for idx in range(Int(thread_idx.x), BM * BK, Int(block_dim.x)):
                    var r = idx // BK
                    var kk = idx - r * BK
                    var global_row = row0 + r
                    var global_k = k0_next + kk
                    if global_row < seg_end and global_row < P and global_k < I:
                        A_s0[r, kk] = h_ptr[global_row * I + global_k]
                    else:
                        A_s0[r, kk] = 0

                for idx in range(
                    Int(thread_idx.x),
                    BN * bytes_per_col,
                    Int(block_dim.x),
                ):
                    var c = idx // bytes_per_col
                    var byte_off = idx - c * bytes_per_col
                    var block_in_tile = byte_off // BYTES_PER_BLOCK
                    var byte_in_block = (
                        byte_off - block_in_tile * BYTES_PER_BLOCK
                    )

                    var col = n0 + c
                    var r0 = (
                        block_in_tile * VALUES_PER_BLOCK + byte_in_block * 2
                    )
                    var kb = kb0_next + block_in_tile

                    if col < D and kb < Kblocks:
                        var scale_exp = w_scales_ptr[
                            ((expert_id * D + col) * Kblocks) + kb
                        ]
                        var packed = w_blocks_ptr[
                            (
                                ((expert_id * D + col) * Kblocks + kb)
                                * BYTES_PER_BLOCK
                            )
                            + byte_in_block
                        ]
                        var v2 = decode_mxfp4_byte_to_2xbf16_e8m0(
                            packed, scale_exp
                        )
                        B_s0[c, r0 + 0] = v2[0]
                        B_s0[c, r0 + 1] = v2[1]
                    else:
                        B_s0[c, r0 + 0] = 0
                        B_s0[c, r0 + 1] = 0

            barrier()

    wgmma.wait_group()

    # Epilogue: bias + gamma scaling + scatter-add into Y[token_id, n] directly
    # from register fragments.
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
                    m_mma * WGMMA_M + Int(warp_id()) * warp_rows + row_in_warp
                )
                var global_row = row0 + row_in_cta
                if global_row >= seg_end or global_row >= P:
                    continue

                var tok = Int(token_ids_s[row_in_cta][0])
                if tok >= T:
                    continue
                var gamma = gamma_s[row_in_cta][0]

                @parameter
                for c_it in range(col_iters):
                    var col_pair = lane_col + c_it * 4
                    var col0 = n0 + n_mma * WGMMA_N + col_pair * 2
                    var col1 = col0 + 1
                    var pair_idx = c_it * row_iters + r_it
                    var v2 = c_pairs[0, pair_idx]

                    if col0 < D:
                        var v0 = v2[0] + b_ptr[expert_id * D + col0]
                        v0 *= gamma
                        _ = Atomic.fetch_add(y_ptr + (tok * D + col0), v0)

                    if col1 < D:
                        var v1 = v2[1] + b_ptr[expert_id * D + col1]
                        v1 *= gamma
                        _ = Atomic.fetch_add(y_ptr + (tok * D + col1), v1)


@compiler.register("mxfp4_moe_w1_swiglu")
struct MXFP4MoEW1SwiGlu:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        h_sorted: OutputTensor[dtype=BF16, rank=2],
        x: InputTensor[dtype=BF16, rank=2],
        token_expert_order: InputTensor[dtype=U32, rank=1],
        expert_start_indices: InputTensor[dtype=U32, rank=1],
        expert_ids: InputTensor[dtype=I32, rank=1],
        max_num_tokens_per_expert: UInt32,
        num_active_experts: UInt32,
        w_blocks: InputTensor[dtype=U8, rank=4],
        w_scales: InputTensor[dtype=U8, rank=3],
        bias: InputTensor[dtype=F32, rank=2],
        alpha: Float32,
        limit: Float32,
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if is_cpu[target]():
            raise Error("mxfp4_moe_w1_swiglu: GPU only")

        var T = x.dim_size(0)
        var D = x.dim_size(1)
        var P = token_expert_order.dim_size(0)

        var num_active_experts_i = Int(num_active_experts)
        var max_tokens_i = Int(max_num_tokens_per_expert)
        var num_experts = w_blocks.dim_size(0)
        var kblocks = w_blocks.dim_size(2)

        var n_raw_total = bias.dim_size(1)
        var I = n_raw_total // 2

        if P == 0 or num_active_experts_i == 0 or max_tokens_i == 0:
            return

        var grid_x = ceildiv(I, 32)  # BN_ACT = 32
        var grid_y = ceildiv(max_tokens_i, 64)  # BM = 64
        var grid_z = num_active_experts_i

        var gpu_ctx = ctx.get_device_context()

        var x_dev = DeviceBuffer[x.dtype](
            gpu_ctx, x.unsafe_ptr(), x.size(), owning=False
        )
        var token_expert_order_dev = DeviceBuffer[token_expert_order.dtype](
            gpu_ctx,
            token_expert_order.unsafe_ptr(),
            token_expert_order.size(),
            owning=False,
        )
        var expert_start_indices_dev = DeviceBuffer[expert_start_indices.dtype](
            gpu_ctx,
            expert_start_indices.unsafe_ptr(),
            expert_start_indices.size(),
            owning=False,
        )
        var expert_ids_dev = DeviceBuffer[expert_ids.dtype](
            gpu_ctx,
            expert_ids.unsafe_ptr(),
            expert_ids.size(),
            owning=False,
        )
        var w_blocks_dev = DeviceBuffer[w_blocks.dtype](
            gpu_ctx, w_blocks.unsafe_ptr(), w_blocks.size(), owning=False
        )
        var w_scales_dev = DeviceBuffer[w_scales.dtype](
            gpu_ctx, w_scales.unsafe_ptr(), w_scales.size(), owning=False
        )
        var bias_dev = DeviceBuffer[bias.dtype](
            gpu_ctx, bias.unsafe_ptr(), bias.size(), owning=False
        )
        var h_sorted_dev = DeviceBuffer[h_sorted.dtype](
            gpu_ctx, h_sorted.unsafe_ptr(), h_sorted.size(), owning=False
        )

        comptime w1_kernel = moe_w1_mxfp4_swiglu_wgmma[
            BM=64,
            BN_RAW=64,
            BK=64,
            WGMMA_M=64,
            WGMMA_N=64,
            WGMMA_K=16,
        ]
        gpu_ctx.enqueue_function_checked[w1_kernel, w1_kernel](
            x_dev,
            T,
            D,
            token_expert_order_dev,
            P,
            expert_start_indices_dev,
            expert_ids_dev,
            num_active_experts_i,
            w_blocks_dev,
            w_scales_dev,
            kblocks,
            n_raw_total,
            bias_dev,
            h_sorted_dev,
            I,
            Scalar[F32](alpha),
            Scalar[F32](limit),
            grid_dim=(grid_x, grid_y, grid_z),
            block_dim=(128, 1, 1),
        )


@compiler.register("mxfp4_moe_w2_scatter")
struct MXFP4MoEW2Scatter:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        y: OutputTensor[dtype=F32, rank=2],
        h_sorted: InputTensor[dtype=BF16, rank=2],
        token_expert_order: InputTensor[dtype=U32, rank=1],
        expert_start_indices: InputTensor[dtype=U32, rank=1],
        expert_ids: InputTensor[dtype=I32, rank=1],
        max_num_tokens_per_expert: UInt32,
        num_active_experts: UInt32,
        gate_weights: InputTensor[dtype=F32, rank=1],
        w_blocks: InputTensor[dtype=U8, rank=4],
        w_scales: InputTensor[dtype=U8, rank=3],
        bias: InputTensor[dtype=F32, rank=2],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if is_cpu[target]():
            raise Error("mxfp4_moe_w2_scatter: GPU only")

        var P = h_sorted.dim_size(0)
        var I = h_sorted.dim_size(1)
        var D = y.dim_size(1)

        var T = y.dim_size(0)
        var num_active_experts_i = Int(num_active_experts)
        var max_tokens_i = Int(max_num_tokens_per_expert)
        var num_experts = w_blocks.dim_size(0)
        var kblocks = w_blocks.dim_size(2)

        var grid_x = ceildiv(D, 64)  # BN = 64
        var grid_y = ceildiv(max_tokens_i, 64)  # BM = 64
        var grid_z = num_active_experts_i

        var gpu_ctx = ctx.get_device_context()

        var h_sorted_dev = DeviceBuffer[h_sorted.dtype](
            gpu_ctx, h_sorted.unsafe_ptr(), h_sorted.size(), owning=False
        )
        var token_expert_order_dev = DeviceBuffer[token_expert_order.dtype](
            gpu_ctx,
            token_expert_order.unsafe_ptr(),
            token_expert_order.size(),
            owning=False,
        )
        var expert_start_indices_dev = DeviceBuffer[expert_start_indices.dtype](
            gpu_ctx,
            expert_start_indices.unsafe_ptr(),
            expert_start_indices.size(),
            owning=False,
        )
        var expert_ids_dev = DeviceBuffer[expert_ids.dtype](
            gpu_ctx,
            expert_ids.unsafe_ptr(),
            expert_ids.size(),
            owning=False,
        )
        var gate_weights_dev = DeviceBuffer[gate_weights.dtype](
            gpu_ctx,
            gate_weights.unsafe_ptr(),
            gate_weights.size(),
            owning=False,
        )
        var w_blocks_dev = DeviceBuffer[w_blocks.dtype](
            gpu_ctx, w_blocks.unsafe_ptr(), w_blocks.size(), owning=False
        )
        var w_scales_dev = DeviceBuffer[w_scales.dtype](
            gpu_ctx, w_scales.unsafe_ptr(), w_scales.size(), owning=False
        )
        var bias_dev = DeviceBuffer[bias.dtype](
            gpu_ctx, bias.unsafe_ptr(), bias.size(), owning=False
        )
        var y_dev = DeviceBuffer[y.dtype](
            gpu_ctx, y.unsafe_ptr(), y.size(), owning=False
        )

        # Zero-initialize output (kernel scatter-adds into Y).
        gpu_ctx.enqueue_memset(y_dev, 0)

        if P == 0 or num_active_experts_i == 0 or max_tokens_i == 0:
            return

        comptime w2_kernel = moe_w2_mxfp4_scatter_wgmma[
            BM=64,
            BN=64,
            BK=64,
            WGMMA_M=64,
            WGMMA_N=64,
            WGMMA_K=16,
        ]
        gpu_ctx.enqueue_function_checked[w2_kernel, w2_kernel](
            h_sorted_dev,
            P,
            I,
            token_expert_order_dev,
            expert_start_indices_dev,
            expert_ids_dev,
            num_active_experts_i,
            gate_weights_dev,
            w_blocks_dev,
            w_scales_dev,
            kblocks,
            bias_dev,
            y_dev,
            T,
            D,
            grid_dim=(grid_x, grid_y, grid_z),
            block_dim=(128, 1, 1),
        )
