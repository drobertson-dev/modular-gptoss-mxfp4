# ===----------------------------------------------------------------------=== #
# MXFP4 MoE grouped matmul (ragged) GPU kernel
# ===----------------------------------------------------------------------=== #

import compiler_internal as compiler
from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import block_dim, block_idx, thread_idx
from math import ldexp, log2
from runtime.asyncrt import DeviceContext, DeviceContextPtr
from tensor import InputTensor, OutputTensor
from tensor.transitional import managed_tensor_slice_to_ndbuffer

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
fn e8m0_to_fp32(e: UInt8) -> Float32:
    if e == 0:
        return 0.0
    return ldexp(Float32(1.0), Int(e) - 127)

@always_inline
fn decode_byte_pair(byte_val: UInt8, scale: Float32) -> SIMD[DType.float32, 2]:
    var lo = byte_val & 0x0F
    var hi = (byte_val >> 4) & 0x0F
    return SIMD[DType.float32, 2](
        MXFP4_LUT[Int(lo)] * scale, MXFP4_LUT[Int(hi)] * scale
    )

# GPU kernel: grid_z = num_active_experts, grid_x/y tile tokens/out_features per expert.
fn _mxfp4_grouped_matmul_kernel[
    y_shape: DimList,
    hidden_shape: DimList,
    q_shape: DimList,
    e_shape: DimList,
    start_shape: DimList,
    ids_shape: DimList,
](
    Y: NDBuffer[mut=True, TILE_DTYPE, 2, MutAnyOrigin, y_shape],
    hidden: NDBuffer[TILE_DTYPE, 2, MutAnyOrigin, hidden_shape],
    Q: NDBuffer[DType.uint8, 3, MutAnyOrigin, q_shape],
    E: NDBuffer[DType.uint8, 3, MutAnyOrigin, e_shape],
    expert_start_indices: NDBuffer[DType.uint32, 1, MutAnyOrigin, start_shape],
    expert_ids: NDBuffer[DType.int32, 1, MutAnyOrigin, ids_shape],
    max_tokens_per_expert: Int,
) raises:
    var expert_block = Int(block_idx.z)
    if expert_block + 1 >= expert_start_indices.dim[0]():
        return
    var expert_idx = expert_ids[expert_block].cast[DType.int32]()[0]
    if expert_idx < 0 or expert_idx >= Int(Q.dim[0]()):
        return
    var start = Int(expert_start_indices[expert_block])
    var end = Int(expert_start_indices[expert_block + 1])

    var entries_per_block = QK_MXFP4 // 2
    var num_blocks = Q.dim[2]() // entries_per_block

    var token_offset = Int(block_idx.x) * BLOCK_TOKENS + Int(thread_idx.x)
    if token_offset >= max_tokens_per_expert:
        return
    var token_idx = start + token_offset
    if token_idx >= end or token_idx >= hidden.dim[0]():
        return

    var out_idx = Int(block_idx.y) * BLOCK_OUT + Int(thread_idx.y)
    while out_idx < Y.dim[1]():
        var acc: Float32 = 0.0
        var block = 0
        while block < num_blocks:
            var scale: Float32 = e8m0_to_fp32(
                E[expert_idx, out_idx, block].cast[DType.uint8]()[0]
            )
            var q_base = block * entries_per_block
            var hidden_base = block * QK_MXFP4
            var j = 0
            while j < entries_per_block:
                var byte0: UInt8 = Q[expert_idx, out_idx, q_base + j].cast[
                    DType.uint8
                ]()[0]
                var idx0 = hidden_base + j
                var idx1 = idx0 + entries_per_block
                var x0: Float32 = hidden[token_idx, idx0].cast[DType.float32]()[0]
                var x1: Float32 = hidden[token_idx, idx1].cast[DType.float32]()[0]
                var pair = decode_byte_pair(byte0, scale)
                acc += pair[0] * x0 + pair[1] * x1
                j += 1
            block += 1
        Y[token_idx, out_idx] = acc.cast[Y.dtype]()
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
    mut Y: OutputTensor,
) raises:
    var hidden_buf = managed_tensor_slice_to_ndbuffer(hidden)
    var q_buf = managed_tensor_slice_to_ndbuffer(Q)
    var e_buf = managed_tensor_slice_to_ndbuffer(E)
    var start_buf = managed_tensor_slice_to_ndbuffer(expert_start_indices)
    var ids_buf = managed_tensor_slice_to_ndbuffer(expert_ids)
    var y_buf = managed_tensor_slice_to_ndbuffer(Y)

    var grid_x = ceil_div(max_tokens_per_expert, BLOCK_TOKENS)
    var grid_y = ceil_div(y_buf.dim[1](), BLOCK_OUT)

    ctx.enqueue_function_checked[
        _mxfp4_grouped_matmul_kernel, _mxfp4_grouped_matmul_kernel
    ](
        y_buf,
        hidden_buf,
        q_buf,
        e_buf,
        start_buf,
        ids_buf,
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
