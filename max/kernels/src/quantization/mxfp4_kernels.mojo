# ===----------------------------------------------------------------------=== #
# MXFP4 Quantization + MoE grouped matmul (ragged) kernels (minimal build)
# ===----------------------------------------------------------------------=== #

import compiler_internal as compiler
from gpu import (
    WARP_SIZE,
    block_dim,
    block_idx,
    global_idx,
    lane_id,
    thread_idx,
    warp_id,
)
from layout import Layout
from layout.layout_tensor import LayoutTensor
from layout.runtime_layout import RuntimeLayout, UNKNOWN_VALUE
from math import ldexp, log2
from memory import AddressSpace, LegacyUnsafePointer as UnsafePointer
from os.atomic import Atomic, Consistency
from runtime.asyncrt import DeviceContext, DeviceContextPtr
from sys import align_of, env_get_bool, env_get_int, size_of
from tensor import InputTensor, OutputTensor
from utils.index import Index, IndexList

# Constants
alias QK_MXFP4 = 32
alias BLOCK_TOKENS: Int = 32
alias BLOCK_OUT: Int = 32
alias THREADS_TOKENS: Int = 32
alias THREADS_OUT: Int = 4
alias ASSIGN_THREADS: Int = 128
alias TILE_DTYPE = DType.bfloat16

alias MXFP4_LUT = SIMD[DType.float32, 16,](
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

# Helpers
fn min_int(a: Int, b: Int) -> Int:
    if a < b:
        return a
    return b

fn ceil_div(a: Int, b: Int) -> Int:
    return (a + b - 1) // b

fn e8m0_to_fp32(e: UInt8) -> Float32:
    if e == 0:
        return 0.0
    return ldexp(Float32(1.0), Int(e) - 127)

fn fp32_to_e8m0_from_block_max(max_val: Float32) -> UInt8:
    if max_val <= 0.0:
        return 0
    if max_val < 1e-38:
        return 0
    var scaled = max_val / 6.0
    if scaled <= 0.0:
        return 0
    var e_est: Float32 = log2(scaled) + 127.0
    return UInt8(min_int(Int(e_est), 254))

fn decode_byte_pair(byte_val: UInt8, scale: Float32) -> SIMD[DType.float32, 2]:
    var lo = byte_val & 0x0F
    var hi = (byte_val >> 4) & 0x0F
    return SIMD[DType.float32, 2](
        MXFP4_LUT[Int(lo)] * scale, MXFP4_LUT[Int(hi)] * scale
    )

fn encode_mxfp4(v: Float32, d: Float32) -> UInt8:
    if d == 0.0:
        return 0
    var a = v / d
    var sign: UInt8 = 0
    if a < 0.0:
        sign = 0x8
        a = -a
    var k: UInt8 = 0
    if a >= 0.5:
        k = 1
    if a >= 1.0:
        k = 2
    if a >= 1.5:
        k = 3
    if a >= 2.0:
        k = 4
    if a >= 3.0:
        k = 5
    if a >= 4.0:
        k = 6
    if a >= 6.0:
        k = 7
    return sign | k

# CPU encode/decode paths used by tests
@compiler.register("modular_ops::mxfp4_quantize_exq")
struct MXFP4QuantizeEXQ:
    @staticmethod
    fn execute[
        in_dtype: DType, rank: Int, target: StaticString
    ](
        out_q: OutputTensor[dtype = DType.uint8, rank=rank],
        out_e: OutputTensor[dtype = DType.uint8, rank=rank],
        x: InputTensor[dtype=in_dtype, rank=rank],
        ctx: DeviceContextPtr,
    ) raises:
        constrained[rank == 2, "rank must be 2"]()
        constrained[in_dtype == DType.float32, "quantize expects float32 input"]()
        var X = x.to_layout_tensor()
        var Q = out_q.to_layout_tensor()
        var E = out_e.to_layout_tensor()
        alias W = X.shape[1]()
        constrained[W % QK_MXFP4 == 0, "W must be divisible by 32"]()
        _mxfp4_quantize_cpu(X, Q, E)

fn _mxfp4_quantize_cpu(
    X: LayoutTensor, mut Q: LayoutTensor, mut E: LayoutTensor
):
    alias H = X.shape[0]()
    alias W = X.shape[1]()
    var blocks_per_row = W // QK_MXFP4
    for r in range(H):
        for b in range(blocks_per_row):
            var c0 = b * QK_MXFP4
            var m: Float32 = 0.0
            for j in range(QK_MXFP4):
                var f: Float32 = X[r, c0 + j].cast[DType.float32]()[0]
                var af = f if f >= 0.0 else -f
                if af > m:
                    m = af
            var e: UInt8 = fp32_to_e8m0_from_block_max(m)
            E[r, b] = e.cast[E.dtype]()
            var d: Float32 = e8m0_to_fp32(e)
            var q_base = b * (QK_MXFP4 // 2)
            for j in range(QK_MXFP4 // 2):
                var v0: Float32 = X[r, c0 + j].cast[DType.float32]()[0]
                var v1: Float32 = X[r, c0 + j + QK_MXFP4 // 2].cast[
                    DType.float32
                ]()[0]
                var i0: UInt8 = encode_mxfp4(v0, d)
                var i1: UInt8 = encode_mxfp4(v1, d)
                var packed: UInt8 = (UInt8(i1) << 4) | (i0 & 0x0F)
                Q[r, q_base + j] = packed.cast[Q.dtype]()

@compiler.register("modular_ops::mxfp4_dequantize_exq")
struct MXFP4DequantizeEXQ:
    @staticmethod
    fn execute[
        out_dtype: DType, rank: Int, target: StaticString
    ](
        out_x: OutputTensor[dtype=out_dtype, rank=rank],
        q: InputTensor[dtype = DType.uint8, rank=rank],
        e: InputTensor[dtype = DType.uint8, rank=rank],
        ctx: DeviceContextPtr,
    ) raises:
        constrained[rank == 2, "rank must be 2"]()
        constrained[out_dtype == DType.float32, "dequantize outputs float32"]()
        var Q = q.to_layout_tensor()
        var E = e.to_layout_tensor()
        var X = out_x.to_layout_tensor()
        _mxfp4_dequantize_cpu(Q, E, X)

fn _mxfp4_dequantize_cpu(
    Q: LayoutTensor, E: LayoutTensor, mut X: LayoutTensor
):
    alias H = Q.shape[0]()
    alias W2 = Q.shape[1]()
    var W = W2 * 2
    var blocks_per_row = W // QK_MXFP4
    for r in range(H):
        for b in range(blocks_per_row):
            var d: Float32 = e8m0_to_fp32(E[r, b].cast[DType.uint8]()[0])
            var q_base = b * (QK_MXFP4 // 2)
            var x_base = b * QK_MXFP4
            for j in range(QK_MXFP4 // 2):
                var byte0: UInt8 = Q[r, q_base + j].cast[DType.uint8]()[0]
                var pair = decode_byte_pair(byte0, d)
                X[r, x_base + j] = pair[0].cast[X.dtype]()
                X[r, x_base + j + QK_MXFP4 // 2] = pair[1].cast[X.dtype]()

# ----------------------------------------------------------------------------
# GROUPED MATMUL (RAGGED) WITH MXFP4 WEIGHTS
# ----------------------------------------------------------------------------

fn _mxfp4_grouped_matmul_cpu(
    hidden: LayoutTensor,
    Q: LayoutTensor,
    E: LayoutTensor,
    assignments: LayoutTensor,
    mut Y: LayoutTensor,
) raises:
    alias tokens = hidden.shape[0]()
    alias out_features = Y.shape[1]()
    alias packed_width = Q.shape[2]()
    var hidden_width = hidden.shape[1]()
    if hidden_width != packed_width * 2:
        raise Error(
            "mxfp4_grouped_matmul: hidden width ",
            hidden_width,
            " does not match packed weights (",
            packed_width * 2,
            ")",
        )
    alias entries_per_block = QK_MXFP4 // 2
    var num_blocks = packed_width // entries_per_block
    for t in range(tokens):
        var expert_idx_raw = assignments[t].cast[DType.int32]()[0]
        if expert_idx_raw < 0 or expert_idx_raw >= Int(Q.shape[0]()):
            raise Error(
                "mxfp4_grouped_matmul: assignment index ",
                expert_idx_raw,
                " is outside valid expert range [0, ",
                Q.shape[0]() - 1,
                "]",
            )
        var expert_idx = Int(expert_idx_raw)
        for m in range(out_features):
            var acc: Float32 = 0.0
            for block in range(num_blocks):
                var scale: Float32 = e8m0_to_fp32(
                    E[expert_idx, m, block].cast[DType.uint8]()[0]
                )
                var q_base = block * entries_per_block
                var hidden_base = block * QK_MXFP4
                for j in range(entries_per_block):
                    var byte0: UInt8 = Q[expert_idx, m, q_base + j].cast[
                        DType.uint8
                    ]()[0]
                    var idx0 = hidden_base + j
                    var idx1 = idx0 + entries_per_block
                    var x0: Float32 = hidden[t, idx0].cast[DType.float32]()[0]
                    var x1: Float32 = hidden[t, idx1].cast[DType.float32]()[0]
                    var pair = decode_byte_pair(byte0, scale)
                    acc += pair[0] * x0 + pair[1] * x1
            Y[t, m] = acc.cast[Y.dtype]()

alias hidden_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
alias weight_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE)
alias assign_layout = Layout.row_major(UNKNOWN_VALUE)

fn _mxfp4_grouped_matmul_kernel(
    hidden: LayoutTensor[DType.bfloat16, hidden_layout, MutAnyOrigin],
    Q: LayoutTensor[DType.uint8, weight_layout, MutAnyOrigin],
    E: LayoutTensor[DType.uint8, weight_layout, MutAnyOrigin],
    assignments: LayoutTensor[DType.int32, assign_layout, MutAnyOrigin],
    mut Y: LayoutTensor[DType.bfloat16, hidden_layout, MutAnyOrigin],
) raises:
    var token_tile = Int(block_idx.x) * BLOCK_TOKENS
    var out_tile = Int(block_idx.y) * BLOCK_OUT
    alias entries_per_block = QK_MXFP4 // 2
    var num_blocks = Q.shape[2]() // entries_per_block

    var token_idx = token_tile + Int(thread_idx.x)
    while token_idx < token_tile + BLOCK_TOKENS:
        if token_idx >= hidden.shape[0]():
            break
        var expert_idx_raw = assignments[token_idx].cast[DType.int32]()[0]
        if expert_idx_raw < 0 or expert_idx_raw >= Int(Q.shape[0]()):
            return
        var expert_idx = Int(expert_idx_raw)

        var out_idx = out_tile + Int(thread_idx.y)
        while out_idx < out_tile + BLOCK_OUT:
            if out_idx >= Y.shape[1]():
                break
            var acc: Float32 = 0.0
            for block in range(num_blocks):
                var scale: Float32 = e8m0_to_fp32(
                    E[expert_idx, out_idx, block].cast[DType.uint8]()[0]
                )
                var q_base = block * entries_per_block
                var hidden_base = block * QK_MXFP4
                for j in range(entries_per_block):
                    var byte0: UInt8 = Q[expert_idx, out_idx, q_base + j].cast[
                        DType.uint8
                    ]()[0]
                    var idx0 = hidden_base + j
                    var idx1 = idx0 + entries_per_block
                    var x0: Float32 = hidden[token_idx, idx0].cast[
                        DType.float32
                    ]()[0]
                    var x1: Float32 = hidden[token_idx, idx1].cast[
                        DType.float32
                    ]()[0]
                    var pair = decode_byte_pair(byte0, scale)
                    acc += pair[0] * x0 + pair[1] * x1
            Y[token_idx, out_idx] = acc.cast[Y.dtype]()
            out_idx += Int(block_dim.y)
        token_idx += Int(block_dim.x)

fn _mxfp4_grouped_matmul_gpu(
    ctx: DeviceContext,
    hidden: LayoutTensor,
    Q: LayoutTensor,
    E: LayoutTensor,
    assignments: LayoutTensor,
    mut Y: LayoutTensor,
) raises:
    var grid_x = ceil_div(hidden.shape[0](), BLOCK_TOKENS)
    var grid_y = ceil_div(Y.shape[1](), BLOCK_OUT)
    alias kernel = _mxfp4_grouped_matmul_kernel
    ctx.enqueue_function_checked[kernel, kernel](
        hidden,
        Q,
        E,
        assignments,
        Y,
        grid_dim=(grid_x, grid_y),
        block_dim=(THREADS_TOKENS, THREADS_OUT),
    )

fn _mxfp4_assignments_kernel(
    assignments: LayoutTensor[DType.int32, assign_layout, MutAnyOrigin],
    expert_start_indices: LayoutTensor[DType.uint32, assign_layout, MutAnyOrigin],
    expert_ids: LayoutTensor[DType.int32, assign_layout, MutAnyOrigin],
    max_tokens_per_expert: Int,
) raises:
    var expert = Int(block_idx.y)
    var token_offset = Int(global_idx.x)
    if expert >= expert_ids.shape[0]():
        return
    if token_offset >= max_tokens_per_expert:
        return
    var start = Int(expert_start_indices[expert])
    var end = Int(expert_start_indices[expert + 1])
    if start + token_offset >= end:
        return
    assignments[start + token_offset] = expert_ids[expert]

fn _mxfp4_grouped_matmul_ragged_gpu(
    ctx: DeviceContext,
    hidden: LayoutTensor,
    Q: LayoutTensor,
    E: LayoutTensor,
    expert_start_indices: LayoutTensor,
    expert_ids: LayoutTensor,
    max_tokens_per_expert: Int,
    num_active_experts: Int,
    mut Y: LayoutTensor,
) raises:
    if hidden.shape[0]() == 0 or num_active_experts == 0:
        return

    var assignments_buf = ctx.enqueue_create_buffer[DType.int32](
        hidden.shape[0]()
    )
    alias assignments_layout = Layout.row_major(1, UNKNOWN_VALUE)
    var assignments = LayoutTensor[
        DType.int32, assignments_layout
    ](
        rebind[UnsafePointer[Scalar[DType.int32]]](assignments_buf.unsafe_ptr()),
        RuntimeLayout[assignments_layout].row_major(
            IndexList[2](1, hidden.shape[0]())
        ),
    )

    var grid_x = ceil_div(max_tokens_per_expert, ASSIGN_THREADS)
    alias assign_kernel = _mxfp4_assignments_kernel
    ctx.enqueue_function_checked[assign_kernel, assign_kernel](
        assignments,
        expert_start_indices,
        expert_ids,
        max_tokens_per_expert,
        grid_dim=(grid_x, num_active_experts, 1),
        block_dim=(ASSIGN_THREADS, 1, 1),
    )

    _mxfp4_grouped_matmul_gpu(ctx, hidden, Q, E, assignments, Y)

fn _mxfp4_grouped_matmul_ragged(
    ctx: DeviceContext,
    hidden: LayoutTensor,
    Q: LayoutTensor,
    E: LayoutTensor,
    expert_start_indices: LayoutTensor,
    expert_ids: LayoutTensor,
    max_tokens_per_expert: Int,
    num_active_experts: Int,
    mut Y: LayoutTensor,
) raises:
    _mxfp4_grouped_matmul_ragged_gpu(
        ctx,
        hidden,
        Q,
        E,
        expert_start_indices,
        expert_ids,
        max_tokens_per_expert,
        num_active_experts,
        Y,
    )

@compiler.register("modular_ops::mxfp4_grouped_matmul_f32_exq")
struct MXFP4GroupedMatMulF32EXQ:
    @staticmethod
    fn execute[
        out_dtype: DType,
        in_dtype: DType,
        target: StaticString,
    ](
        out_y: OutputTensor[dtype=out_dtype, rank=2],
        hidden: InputTensor[dtype=in_dtype, rank=2],
        q: InputTensor[dtype = DType.uint8, rank=3],
        e: InputTensor[dtype = DType.uint8, rank=3],
        assignments: InputTensor[dtype = DType.int32, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        var hidden_lt = hidden.to_layout_tensor()
        var q_lt = q.to_layout_tensor()
        var e_lt = e.to_layout_tensor()
        var assignments_lt = assignments.to_layout_tensor()
        var out_lt = out_y.to_layout_tensor()

        @parameter
        if target == "cpu":
            _mxfp4_grouped_matmul_cpu(
                hidden_lt, q_lt, e_lt, assignments_lt, out_lt
            )
        else:
            var dev = ctx.get_device_context()
            _mxfp4_grouped_matmul_gpu(
                dev, hidden_lt, q_lt, e_lt, assignments_lt, out_lt
            )

@compiler.register("modular_ops::mxfp4_grouped_matmul_exq")
struct MXFP4GroupedMatMulRaggedEXQ:
    @staticmethod
    fn execute[
        out_dtype: DType,
        in_dtype: DType,
        target: StaticString,
    ](
        out_y: OutputTensor[dtype=out_dtype, rank=2],
        hidden: InputTensor[dtype=in_dtype, rank=2],
        q: InputTensor[dtype = DType.uint8, rank=3],
        e: InputTensor[dtype = DType.uint8, rank=3],
        expert_start_indices: InputTensor[dtype = DType.uint32, rank=1],
        expert_ids: InputTensor[dtype = DType.int32, rank=1],
        max_tokens_per_expert: UInt32,
        num_active_experts: UInt32,
        ctx: DeviceContextPtr,
    ) raises:
        var hidden_lt = hidden.to_layout_tensor()
        var q_lt = q.to_layout_tensor()
        var e_lt = e.to_layout_tensor()
        var start_lt = expert_start_indices.to_layout_tensor()
        var expert_lt = expert_ids.to_layout_tensor()
        var out_lt = out_y.to_layout_tensor()

        @parameter
        if target == "cpu":
            raise Error(
                "modular_ops::mxfp4_grouped_matmul_exq only supports GPU"
            )
        else:
            var dev = ctx.get_device_context()
            _mxfp4_grouped_matmul_ragged(
                dev,
                hidden_lt,
                q_lt,
                e_lt,
                start_lt,
                expert_lt,
                Int(max_tokens_per_expert),
                Int(num_active_experts),
                out_lt,
            )
