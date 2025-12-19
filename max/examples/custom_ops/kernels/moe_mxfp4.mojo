# moe_mxfp4.mojo
#
# Correctness-first grouped MXFP4 GEMM kernels used by the Mojo test suite.
#
# These are intended as the stable interface for the eventual SM90 `wgmma`
# implementation. Keep the `execute(...)` signatures stable and swap the
# internal MMA path to `wgmma` later.

from math import ceildiv

from gpu import block_idx, thread_idx
from gpu.host import DeviceBuffer, DeviceContext
from layout.layout_tensor import Layout, LayoutTensor

from .fp4_utils import (
    MXFP4_SF_DTYPE,
    MXFP4_SF_VECTOR_SIZE,
    SF_ATOM_K,
    SF_ATOM_M,
    SF_MN_GROUP_SIZE,
)
from .mxfp4 import (
    MXFP4_BLOCK_K,
    MXFP4_PACKED_BYTES_PER_BLOCK,
)
from .mxfp4_decode import decode_mxfp4_byte_to_2xbf16_e8m0, swiglu_pair


comptime BF16 = DType.bfloat16
comptime F32 = DType.float32
comptime U8 = DType.uint8
comptime U32 = DType.uint32
comptime I32 = DType.int32

comptime SWIGLU_ALPHA: Float32 = 1.702
comptime SWIGLU_LIMIT: Float32 = 7.0


@always_inline
fn swiglu_activation(gate: Float32, up: Float32) -> Float32:
    return swiglu_pair(
        gate,
        up,
        Scalar[F32](SWIGLU_ALPHA),
        Scalar[F32](SWIGLU_LIMIT),
    )


@always_inline
fn swiglu_activation_simd[
    width: Int,
](
    gate: SIMD[DType.float32, width],
    up: SIMD[DType.float32, width],
) -> SIMD[
    DType.float32, width
]:
    var out = SIMD[DType.float32, width]()

    @parameter
    for i in range(width):
        out[i] = swiglu_activation(gate[i], up[i])
    return out


@parameter
fn grouped_mxfp4_matmul_ref[
    BM: Int = 16,
    BN: Int = 16,
](
    out_ptr: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    M: Int,
    N: Int,
    a_ptr: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    K: Int,
    w_packed_ptr: UnsafePointer[Scalar[U8], MutAnyOrigin],
    num_experts_total: Int,
    Kblocks: Int,
    scales_ptr: UnsafePointer[Scalar[MXFP4_SF_DTYPE], MutAnyOrigin],
    col_groups: Int,
    bias_ptr: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    offsets_ptr: UnsafePointer[Scalar[U32], MutAnyOrigin],
    expert_ids_ptr: UnsafePointer[Scalar[I32], MutAnyOrigin],
    num_active_experts: Int,
):
    var expert_idx = Int(block_idx.z)
    if expert_idx >= num_active_experts:
        return

    var expert_id = Int(expert_ids_ptr[expert_idx])
    if expert_id < 0 or expert_id >= num_experts_total:
        return

    var seg_start = Int(offsets_ptr[expert_idx])
    var seg_end = Int(offsets_ptr[expert_idx + 1])

    var row0 = seg_start + Int(block_idx.y) * BM
    var col0 = Int(block_idx.x) * BN

    var tid = Int(thread_idx.x)
    var r = tid // BN
    var c = tid - r * BN
    if r >= BM:
        return

    var row = row0 + r
    var col = col0 + c
    if row >= seg_end or row >= M or col >= N:
        return

    var acc: Scalar[F32] = 0.0
    for kb in range(Kblocks):
        var k_block_start = kb * MXFP4_BLOCK_K
        var row_group = col // SF_MN_GROUP_SIZE
        var col_group = k_block_start // (MXFP4_SF_VECTOR_SIZE * SF_ATOM_K)
        var m0 = col % SF_ATOM_M[0]
        var m1 = (col % SF_MN_GROUP_SIZE) // SF_ATOM_M[0]
        var k0 = (k_block_start // MXFP4_SF_VECTOR_SIZE) % SF_ATOM_K

        var scale_idx = (
            (((row_group * col_groups + col_group) * 32 + m0) * 4 + m1)
            * SF_ATOM_K
        ) + k0
        var scale_exp = rebind[Scalar[U8]](scales_ptr[scale_idx])

        @parameter
        for byte_in_block in range(MXFP4_PACKED_BYTES_PER_BLOCK):
            var kk = kb * MXFP4_BLOCK_K + byte_in_block * 2
            if kk + 1 >= K:
                continue
            var a0 = a_ptr[row * K + kk].cast[F32]()
            var a1 = a_ptr[row * K + kk + 1].cast[F32]()
            var packed = w_packed_ptr[
                (
                    (
                        ((expert_id * N + col) * Kblocks + kb)
                        * MXFP4_PACKED_BYTES_PER_BLOCK
                    )
                    + byte_in_block
                )
            ]
            var v2 = decode_mxfp4_byte_to_2xbf16_e8m0(packed, scale_exp)
            acc += a0 * v2[0].cast[F32]() + a1 * v2[1].cast[F32]()

    var b = bias_ptr[expert_id * N + col].cast[F32]()
    out_ptr.store(row * N + col, (acc + b).cast[BF16]())


@parameter
fn grouped_mxfp4_swiglu_ref[
    BM: Int = 16,
    BN: Int = 16,
](
    out_ptr: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    M: Int,
    I: Int,
    a_ptr: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    K: Int,
    w_packed_ptr: UnsafePointer[Scalar[U8], MutAnyOrigin],
    num_experts_total: Int,
    Kblocks: Int,
    scales_ptr: UnsafePointer[Scalar[MXFP4_SF_DTYPE], MutAnyOrigin],
    col_groups: Int,
    bias_ptr: UnsafePointer[Scalar[BF16], MutAnyOrigin],
    offsets_ptr: UnsafePointer[Scalar[U32], MutAnyOrigin],
    expert_ids_ptr: UnsafePointer[Scalar[I32], MutAnyOrigin],
    num_active_experts: Int,
):
    var expert_idx = Int(block_idx.z)
    if expert_idx >= num_active_experts:
        return

    var expert_id = Int(expert_ids_ptr[expert_idx])
    if expert_id < 0 or expert_id >= num_experts_total:
        return

    var seg_start = Int(offsets_ptr[expert_idx])
    var seg_end = Int(offsets_ptr[expert_idx + 1])
    var row0 = seg_start + Int(block_idx.y) * BM

    var col0 = Int(block_idx.x) * BN

    var tid = Int(thread_idx.x)
    var r = tid // BN
    var c = tid - r * BN
    if r >= BM:
        return

    var row = row0 + r
    var out_col = col0 + c
    if row >= seg_end or row >= M or out_col >= I:
        return

    var col_gate = out_col * 2
    var col_up = col_gate + 1

    var acc_gate: Scalar[F32] = 0.0
    var acc_up: Scalar[F32] = 0.0
    for kb in range(Kblocks):
        var k_block_start = kb * MXFP4_BLOCK_K
        var col_group = k_block_start // (MXFP4_SF_VECTOR_SIZE * SF_ATOM_K)
        var k0 = (k_block_start // MXFP4_SF_VECTOR_SIZE) % SF_ATOM_K

        var row_group_gate = col_gate // SF_MN_GROUP_SIZE
        var m0_gate = col_gate % SF_ATOM_M[0]
        var m1_gate = (col_gate % SF_MN_GROUP_SIZE) // SF_ATOM_M[0]

        var row_group_up = col_up // SF_MN_GROUP_SIZE
        var m0_up = col_up % SF_ATOM_M[0]
        var m1_up = (col_up % SF_MN_GROUP_SIZE) // SF_ATOM_M[0]

        var scale_idx_gate = (
            (
                ((row_group_gate * col_groups + col_group) * 32 + m0_gate) * 4
                + m1_gate
            )
            * SF_ATOM_K
        ) + k0
        var scale_idx_up = (
            (((row_group_up * col_groups + col_group) * 32 + m0_up) * 4 + m1_up)
            * SF_ATOM_K
        ) + k0

        var scale_gate = rebind[Scalar[U8]](scales_ptr[scale_idx_gate])
        var scale_up = rebind[Scalar[U8]](scales_ptr[scale_idx_up])

        @parameter
        for byte_in_block in range(MXFP4_PACKED_BYTES_PER_BLOCK):
            var kk = kb * MXFP4_BLOCK_K + byte_in_block * 2
            if kk + 1 >= K:
                continue
            var a0 = a_ptr[row * K + kk].cast[F32]()
            var a1 = a_ptr[row * K + kk + 1].cast[F32]()

            var packed_gate = w_packed_ptr[
                (
                    (
                        ((expert_id * (2 * I) + col_gate) * Kblocks + kb)
                        * MXFP4_PACKED_BYTES_PER_BLOCK
                    )
                    + byte_in_block
                )
            ]
            var packed_up = w_packed_ptr[
                (
                    (
                        ((expert_id * (2 * I) + col_up) * Kblocks + kb)
                        * MXFP4_PACKED_BYTES_PER_BLOCK
                    )
                    + byte_in_block
                )
            ]

            var g2 = decode_mxfp4_byte_to_2xbf16_e8m0(packed_gate, scale_gate)
            var u2 = decode_mxfp4_byte_to_2xbf16_e8m0(packed_up, scale_up)

            acc_gate += a0 * g2[0].cast[F32]() + a1 * g2[1].cast[F32]()
            acc_up += a0 * u2[0].cast[F32]() + a1 * u2[1].cast[F32]()

    var gate_bias = bias_ptr[expert_id * (2 * I) + col_gate].cast[F32]()
    var up_bias = bias_ptr[expert_id * (2 * I) + col_up].cast[F32]()
    var y = swiglu_activation(acc_gate + gate_bias, acc_up + up_bias)
    out_ptr.store(row * I + out_col, y.cast[BF16]())


struct GroupedMXFP4Matmul:
    @staticmethod
    fn execute[
        out_layout: Layout,
        a_layout: Layout,
        w_layout: Layout,
        scales_layout: Layout,
        bias_layout: Layout,
        offsets_layout: Layout,
        ids_layout: Layout,
        stats_layout: Layout,
        //,
    ](
        output: LayoutTensor[BF16, out_layout, MutAnyOrigin],
        a: LayoutTensor[BF16, a_layout, MutAnyOrigin],
        w_packed: LayoutTensor[U8, w_layout, MutAnyOrigin],
        scales: LayoutTensor[MXFP4_SF_DTYPE, scales_layout, MutAnyOrigin],
        bias: LayoutTensor[BF16, bias_layout, MutAnyOrigin],
        offsets: LayoutTensor[U32, offsets_layout, MutAnyOrigin],
        expert_ids: LayoutTensor[I32, ids_layout, MutAnyOrigin],
        stats: LayoutTensor[U32, stats_layout, MutAnyOrigin],
        ctx: DeviceContext,
    ) raises:
        var N = output.dim(1)
        var max_tokens = Int(stats[0])
        var num_active_experts = Int(stats[1])

        var M = output.dim(0)
        var K = a.dim(1)
        var num_experts_total = w_packed.dim(0)
        var Kblocks = w_packed.dim(2)
        var col_groups = scales.dim(1)

        var out_dev = DeviceBuffer[BF16](
            ctx, output.ptr, output.size(), owning=False
        )
        var a_dev = DeviceBuffer[BF16](ctx, a.ptr, a.size(), owning=False)
        var w_dev = DeviceBuffer[U8](
            ctx, w_packed.ptr, w_packed.size(), owning=False
        )
        var scales_dev = DeviceBuffer[MXFP4_SF_DTYPE](
            ctx, scales.ptr, scales.size(), owning=False
        )
        var bias_dev = DeviceBuffer[BF16](
            ctx, bias.ptr, bias.size(), owning=False
        )
        var offsets_dev = DeviceBuffer[U32](
            ctx, offsets.ptr, offsets.size(), owning=False
        )
        var ids_dev = DeviceBuffer[I32](
            ctx, expert_ids.ptr, expert_ids.size(), owning=False
        )

        comptime kernel = grouped_mxfp4_matmul_ref[
            BM=16,
            BN=16,
        ]

        ctx.enqueue_function_checked[kernel, kernel](
            out_dev,
            M,
            N,
            a_dev,
            K,
            w_dev,
            num_experts_total,
            Kblocks,
            scales_dev,
            col_groups,
            bias_dev,
            offsets_dev,
            ids_dev,
            num_active_experts,
            grid_dim=(
                ceildiv(N, 16),
                ceildiv(max_tokens, 16),
                num_active_experts,
            ),
            block_dim=(256),
        )


struct GroupedMXFP4MatmulSwiGLU:
    @staticmethod
    fn execute[
        out_layout: Layout,
        a_layout: Layout,
        w_layout: Layout,
        scales_layout: Layout,
        bias_layout: Layout,
        offsets_layout: Layout,
        ids_layout: Layout,
        stats_layout: Layout,
        //,
    ](
        output: LayoutTensor[BF16, out_layout, MutAnyOrigin],
        a: LayoutTensor[BF16, a_layout, MutAnyOrigin],
        w_packed: LayoutTensor[U8, w_layout, MutAnyOrigin],
        scales: LayoutTensor[MXFP4_SF_DTYPE, scales_layout, MutAnyOrigin],
        bias: LayoutTensor[BF16, bias_layout, MutAnyOrigin],
        offsets: LayoutTensor[U32, offsets_layout, MutAnyOrigin],
        expert_ids: LayoutTensor[I32, ids_layout, MutAnyOrigin],
        stats: LayoutTensor[U32, stats_layout, MutAnyOrigin],
        ctx: DeviceContext,
    ) raises:
        var I = output.dim(1)
        var max_tokens = Int(stats[0])
        var num_active_experts = Int(stats[1])

        var M = output.dim(0)
        var K = a.dim(1)
        var num_experts_total = w_packed.dim(0)
        var Kblocks = w_packed.dim(2)
        var col_groups = scales.dim(1)

        var out_dev = DeviceBuffer[BF16](
            ctx, output.ptr, output.size(), owning=False
        )
        var a_dev = DeviceBuffer[BF16](ctx, a.ptr, a.size(), owning=False)
        var w_dev = DeviceBuffer[U8](
            ctx, w_packed.ptr, w_packed.size(), owning=False
        )
        var scales_dev = DeviceBuffer[MXFP4_SF_DTYPE](
            ctx, scales.ptr, scales.size(), owning=False
        )
        var bias_dev = DeviceBuffer[BF16](
            ctx, bias.ptr, bias.size(), owning=False
        )
        var offsets_dev = DeviceBuffer[U32](
            ctx, offsets.ptr, offsets.size(), owning=False
        )
        var ids_dev = DeviceBuffer[I32](
            ctx, expert_ids.ptr, expert_ids.size(), owning=False
        )

        comptime kernel = grouped_mxfp4_swiglu_ref[
            BM=16,
            BN=16,
        ]

        ctx.enqueue_function_checked[kernel, kernel](
            out_dev,
            M,
            I,
            a_dev,
            K,
            w_dev,
            num_experts_total,
            Kblocks,
            scales_dev,
            col_groups,
            bias_dev,
            offsets_dev,
            ids_dev,
            num_active_experts,
            grid_dim=(
                ceildiv(I, 16),
                ceildiv(max_tokens, 16),
                num_active_experts,
            ),
            block_dim=(256),
        )
