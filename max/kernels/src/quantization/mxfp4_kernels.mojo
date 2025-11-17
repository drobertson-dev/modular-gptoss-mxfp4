# ===----------------------------------------------------------------------=== #
# MXFP4 MoE grouped matmul (ragged) GPU kernel
# ===----------------------------------------------------------------------=== #

import compiler_internal as compiler
from gpu import block_dim, block_idx, thread_idx
from layout import Layout, LayoutTensor
from math import floor, ldexp, log2
from runtime.asyncrt import DeviceContext, DeviceContextPtr
from tensor import InputTensor, OutputTensor

# Kernel parameters
alias QK_MXFP4 = 32
alias BLOCK_TOKENS: Int = 32
alias BLOCK_OUT: Int = 32
alias THREADS_TOKENS: Int = 32
alias THREADS_OUT: Int = 4
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


@always_inline
fn ceil_div(a: Int, b: Int) -> Int:
    return (a + b - 1) // b


@always_inline
fn fp32_to_e8m0_from_block_max(max_val: Float32) -> UInt8:
    if max_val <= 0.0:
        return 0
    if max_val < 1e-38:
        return 0
    var scaled = max_val / 6.0
    if scaled <= 0.0:
        return 0
    var e_est: Float32 = floor(log2(scaled)) + 127.0
    if e_est < 1.0:
        return 0
    if e_est > 254.0:
        return 254
    return UInt8(e_est)


@always_inline
fn e8m0_to_fp32(e: UInt8) -> Float32:
    if e == 0:
        return 0.0
    return ldexp(Float32(1.0), Int(e) - 127)


@always_inline
fn encode_mxfp4(val: Float32, d: Float32) -> UInt8:
    if d == 0.0:
        return 0
    var a = val / d
    var sign: UInt8 = 0
    if a < 0.0:
        sign = 0x8
        a = -a
    var k: UInt8 = 0
    if a >= 0.25:
        k = 1
    if a >= 0.75:
        k = 2
    if a >= 1.25:
        k = 3
    if a >= 1.75:
        k = 4
    if a >= 2.5:
        k = 5
    if a >= 3.5:
        k = 6
    if a >= 5.0:
        k = 7
    return sign | k


@always_inline
fn decode_byte_pair(byte_val: UInt8, scale: Float32) -> SIMD[DType.float32, 2]:
    var lo = byte_val & 0x0F
    var hi = (byte_val >> 4) & 0x0F
    return SIMD[DType.float32, 2](
        MXFP4_LUT[Int(lo)] * scale, MXFP4_LUT[Int(hi)] * scale
    )


fn _mxfp4_quantize_cpu(
    X: LayoutTensor, mut Q: LayoutTensor, mut E: LayoutTensor
):
    alias H = X.shape[0]()
    alias W = X.shape[1]()
    constrained[W % QK_MXFP4 == 0, "W must be divisible by 32"]()
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
        constrained[
            in_dtype == DType.float32, "quantize expects float32 input"
        ]()
        var X = x.to_layout_tensor()
        var Q = out_q.to_layout_tensor()
        var E = out_e.to_layout_tensor()

        @parameter
        if target == "cpu":
            _mxfp4_quantize_cpu(X, Q, E)
        else:
            raise Error("mxfp4_quantize: GPU target not implemented")


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


# GPU kernel with flat indexing into LayoutTensors.
fn _mxfp4_grouped_matmul_kernel[
    y_layout: Layout,
    hidden_layout: Layout,
    q_layout: Layout,
    e_layout: Layout,
    start_layout: Layout,
    ids_layout: Layout,
](
    Y: LayoutTensor[mut=True, TILE_DTYPE, y_layout, MutAnyOrigin],
    hidden: LayoutTensor[TILE_DTYPE, hidden_layout, MutAnyOrigin],
    Q: LayoutTensor[DType.uint8, q_layout, MutAnyOrigin],
    E: LayoutTensor[DType.uint8, e_layout, MutAnyOrigin],
    expert_start_indices: LayoutTensor[
        DType.uint32, start_layout, MutAnyOrigin
    ],
    expert_ids: LayoutTensor[DType.int32, ids_layout, MutAnyOrigin],
    max_tokens_per_expert: Int,
):
    var expert_block = Int(block_idx.z)
    var num_expert_starts = Int(expert_start_indices.shape[0]())
    if expert_block + 1 >= num_expert_starts:
        return
    var expert_idx = Int(expert_ids[expert_block])
    if expert_idx < 0 or expert_idx >= Int(Q.shape[0]()):
        return
    var start = Int(expert_start_indices[expert_block])
    var end = Int(expert_start_indices[expert_block + 1])

    var entries_per_block = QK_MXFP4 // 2
    var num_blocks = Int(Q.shape[2]()) // entries_per_block

    var q_blocks = Int(Q.shape[2]())
    var q_expert_stride = Int(Q.shape[1]()) * q_blocks
    var e_blocks = Int(E.shape[2]())
    var e_expert_stride = Int(E.shape[1]()) * e_blocks

    var token_offset = Int(block_idx.x) * BLOCK_TOKENS + Int(thread_idx.x)
    if token_offset >= max_tokens_per_expert:
        return
    var token_idx = start + token_offset
    if token_idx >= end or token_idx >= hidden.shape[0]():
        return

    var hidden_stride = Int(hidden.shape[1]())
    var y_stride = Int(Y.shape[1]())

    # Keep writes inside the current BLOCK_OUT tile so adjacent blocks don't overlap.
    var block_base = Int(block_idx.y) * BLOCK_OUT
    var out_idx = block_base + Int(thread_idx.y)
    while out_idx < y_stride and out_idx < block_base + BLOCK_OUT:
        var acc: Float32 = 0.0

        var block = 0
        while block < num_blocks:
            # flat index into E: [expert_idx, out_idx, block]
            var e_idx = (
                expert_idx * e_expert_stride + out_idx * e_blocks + block
            )
            var scale: Float32 = e8m0_to_fp32(
                E.ptr[e_idx].cast[DType.uint8]()[0]
            )

            var q_base = block * entries_per_block
            var hidden_base = block * QK_MXFP4
            var j = 0
            while j < entries_per_block:
                var q_idx = (
                    expert_idx * q_expert_stride
                    + out_idx * q_blocks
                    + q_base
                    + j
                )
                var byte0: UInt8 = Q.ptr[q_idx].cast[DType.uint8]()[0]
                var idx0 = hidden_base + j
                var idx1 = idx0 + entries_per_block

                var h_idx0 = token_idx * hidden_stride + idx0
                var h_idx1 = token_idx * hidden_stride + idx1
                var x0: Float32 = hidden.ptr[h_idx0].cast[DType.float32]()[0]
                var x1: Float32 = hidden.ptr[h_idx1].cast[DType.float32]()[0]
                var pair = decode_byte_pair(byte0, scale)
                acc += pair[0] * x0 + pair[1] * x1
                j += 1
            block += 1

        var y_idx = token_idx * y_stride + out_idx
        Y.ptr[y_idx] = acc.cast[Y.dtype]()
        out_idx += Int(block_dim.y)


fn _mxfp4_grouped_matmul_gpu(
    ctx: DeviceContext,
    hidden: InputTensor,
    Q: InputTensor,
    E: InputTensor,
    expert_start_indices: InputTensor,
    expert_ids: InputTensor,
    max_tokens_per_expert: Int,
    num_active_experts: Int,
    Y: OutputTensor,
) raises:
    var hidden_lt = hidden.to_layout_tensor()
    var q_lt = Q.to_layout_tensor()
    var e_lt = E.to_layout_tensor()
    var start_lt = expert_start_indices.to_layout_tensor()
    var ids_lt = expert_ids.to_layout_tensor()
    var y_lt = Y.to_layout_tensor()

    var grid_x = ceil_div(max_tokens_per_expert, BLOCK_TOKENS)
    var grid_y = ceil_div(y_lt.shape[1](), BLOCK_OUT)

    alias Kernel = _mxfp4_grouped_matmul_kernel[
        y_lt.layout,
        hidden_lt.layout,
        q_lt.layout,
        e_lt.layout,
        start_lt.layout,
        ids_lt.layout,
    ]

    ctx.enqueue_function_checked[Kernel, Kernel](
        y_lt,
        hidden_lt,
        q_lt,
        e_lt,
        start_lt,
        ids_lt,
        max_tokens_per_expert,
        grid_dim=(grid_x, grid_y, num_active_experts),
        block_dim=(THREADS_TOKENS, THREADS_OUT, 1),
    )


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
    constrained[hidden.dtype == TILE_DTYPE, "hidden must be bf16"]()
    constrained[Y.dtype == TILE_DTYPE, "output must be bf16"]()
    var grid_x = ceil_div(max_tokens_per_expert, BLOCK_TOKENS)
    var grid_y = ceil_div(Y.shape[1](), BLOCK_OUT)

    alias Kernel = _mxfp4_grouped_matmul_kernel[
        Y.layout,
        hidden.layout,
        Q.layout,
        E.layout,
        expert_start_indices.layout,
        expert_ids.layout,
    ]

    ctx.enqueue_function_checked[Kernel, Kernel](
        Y,
        hidden,
        Q,
        E,
        expert_start_indices,
        expert_ids,
        max_tokens_per_expert,
        grid_dim=(grid_x, grid_y, num_active_experts),
        block_dim=(THREADS_TOKENS, THREADS_OUT, 1),
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
        constrained[in_dtype == DType.bfloat16, "hidden must be bf16"]()
        constrained[out_dtype == DType.bfloat16, "out must be bf16"]()

        @parameter
        if target == "cpu":
            raise Error(
                "modular_ops::mxfp4_grouped_matmul_exq only supports GPU"
            )
        else:
            var dev = ctx.get_device_context()
            _mxfp4_grouped_matmul_gpu(
                dev,
                hidden,
                q,
                e,
                expert_start_indices,
                expert_ids,
                Int(max_tokens_per_expert),
                Int(num_active_experts),
                out_y,
            )
