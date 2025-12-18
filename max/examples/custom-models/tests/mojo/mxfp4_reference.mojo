# ===----------------------------------------------------------------------=== #
# MXFP4 CPU Reference Utilities
#
# Shared helpers used by the MXFP4 tests to:
#   - Decode packed FP4(E2M1) bytes on CPU
#   - Convert E8M0 scales to float32
#   - Build scale tensors for arbitrary (row, K) ranges
#   - Generate packed weights with custom nibble patterns
#   - Compute reference grouped GEMM and SwiGLU outputs in float32
#
# These CPU-only helpers keep scalar reference logic out of the GPU kernel
# sources while giving the tests a consistent oracle.
# ===----------------------------------------------------------------------=== #

from buffer import Dim
from buffer.dimlist import DimList
from ndbuffer_utils import HostNDBuffer, zero
from kernels.fp4_utils import E2M1_TO_FLOAT32, SF_MN_GROUP_SIZE
from kernels.mxfp4 import (
    MXFP4_BLOCK_K,
    MXFP4_PACKED_BYTES_PER_BLOCK,
    MXFP4_SF_DTYPE,
    get_mxfp4_scale,
    set_mxfp4_scale,
)
from kernels.mxfp4.primitives import e8m0_to_float32, float32_to_e8m0
from layout import Layout, LayoutTensor
from layout._ndbuffer_stub import from_ndbuffer_row_major
from math import ceildiv, exp

comptime SWIGLU_LIMIT: Float32 = 7.0
comptime SWIGLU_ALPHA: Float32 = 1.702


fn decode_e2m1_pair_cpu(packed: Scalar[DType.uint8]) -> SIMD[DType.float32, 2]:
    """Decode a packed FP4(E2M1) byte into float32 without GPU intrinsics."""
    var raw = UInt8(packed.cast[DType.uint8]().to_bits())
    var lo = Int(raw & UInt8(0xF))
    var hi = Int((raw >> 4) & UInt8(0xF))
    return SIMD[DType.float32, 2](E2M1_TO_FLOAT32[lo], E2M1_TO_FLOAT32[hi])


fn scale_f32_cpu[
    scales_layout: Layout,
](
    scales: LayoutTensor[MXFP4_SF_DTYPE, scales_layout, MutAnyOrigin],
    row: Int,
    k_block_start: Int,
) -> Float32:
    """Fetch one MXFP4 scale as float32 for the provided row/K-block anchor."""
    return e8m0_to_float32(get_mxfp4_scale(scales, row, k_block_start))


fn encode_e2m1_code(val: Float32) -> UInt8:
    """Return the FP4(E2M1) nibble code for an exactly representable value."""

    @parameter
    for i in range(16):
        if E2M1_TO_FLOAT32[i] == val:
            return UInt8(i)
    return UInt8(0)


fn pack_nibbles(lo: UInt8, hi: UInt8) -> UInt8:
    """Pack two 4-bit codes into one byte (lo nibble + hi nibble)."""
    return lo | (hi << 4)


fn make_scales_host[
    num_rows: Int,
    k: Int,
]() -> HostNDBuffer[
    MXFP4_SF_DTYPE,
    5,
    DimList(
        Dim(max(1, ceildiv(num_rows, SF_MN_GROUP_SIZE))),
        Dim(max(1, ceildiv(k, MXFP4_BLOCK_K * 4))),
        Dim(32),
        Dim(4),
        Dim(4),
    ),
]:
    """Create a zero-initialized scale tensor sized for (num_rows, K)."""
    alias row_groups = max(1, ceildiv(num_rows, SF_MN_GROUP_SIZE))
    alias col_groups = max(1, ceildiv(k, MXFP4_BLOCK_K * 4))
    alias scale_shape = DimList(
        Dim(row_groups), Dim(col_groups), Dim(32), Dim(4), Dim(4)
    )
    var scales = HostNDBuffer[MXFP4_SF_DTYPE, 5, scale_shape](scale_shape)
    zero(scales.tensor)
    return scales^


fn fill_scales_constant[
    scales_layout: Layout,
](
    scales: LayoutTensor[MXFP4_SF_DTYPE, scales_layout, MutAnyOrigin],
    num_rows: Int,
    k_blocks: Int,
    value: Float32,
):
    """Set every used (row, k_block) scale to the provided constant value."""
    var sf = float32_to_e8m0(value)
    for row in range(num_rows):
        for kb in range(k_blocks):
            set_mxfp4_scale(scales, row, kb * MXFP4_BLOCK_K, sf)


fn fill_scales_table[
    scales_layout: Layout,
    k_blocks: Int,
](
    scales: LayoutTensor[MXFP4_SF_DTYPE, scales_layout, MutAnyOrigin],
    num_rows: Int,
    values: SIMD[DType.float32, 4],
):
    """Fill up to four k-blocks per row with explicit float32 scale values."""
    for row in range(num_rows):

        @parameter
        for kb in range(min(k_blocks, 4)):
            var sf = float32_to_e8m0(values[kb])
            set_mxfp4_scale(scales, row, kb * MXFP4_BLOCK_K, sf)


fn fill_packed_by_nibble[
    packed_layout: Layout,
](
    packed: LayoutTensor[DType.uint8, packed_layout, MutAnyOrigin],
    k_blocks: Int,
    nibble_provider: fn (Int, Int, Int) -> UInt8,
):
    """Populate packed weights via a nibble generator (expert, out_col, k idx).
    """
    for expert in range(packed.dim(0)):
        for out_col in range(packed.dim(1)):
            for kb in range(k_blocks):

                @parameter
                for byte_idx in range(MXFP4_PACKED_BYTES_PER_BLOCK):
                    var k0 = kb * MXFP4_BLOCK_K + byte_idx * 2
                    var lo = nibble_provider(expert, out_col, k0)
                    var hi = nibble_provider(expert, out_col, k0 + 1)
                    packed[expert, out_col, kb, byte_idx] = pack_nibbles(lo, hi)


fn swiglu_reference(gate: Float32, up: Float32) -> Float32:
    """Scalar SwiGLU reference that mirrors the Python implementation."""
    var gate_clamped = gate if gate < SWIGLU_LIMIT else SWIGLU_LIMIT
    var up_clamped = up
    if up_clamped < -SWIGLU_LIMIT:
        up_clamped = -SWIGLU_LIMIT
    if up_clamped > SWIGLU_LIMIT:
        up_clamped = SWIGLU_LIMIT

    var sigmoid_val = Float32(1.0) / (
        Float32(1.0) + exp(-(gate_clamped * SWIGLU_ALPHA))
    )
    var glu = gate_clamped * sigmoid_val
    return (up_clamped + Float32(1.0)) * glu


fn swiglu_reference_simd[
    width: Int,
](gate: SIMD[DType.float32, width], up: SIMD[DType.float32, width]) -> SIMD[
    DType.float32, width
]:
    """Vectorized SwiGLU using the scalar reference per lane."""
    var result = SIMD[DType.float32, width]()

    @parameter
    for i in range(width):
        result[i] = swiglu_reference(gate[i], up[i])
    return result


fn ref_grouped_matmul_cpu[
    a_type: DType,
    a_layout: Layout,
    packed_layout: Layout,
    scales_layout: Layout,
    bias_type: DType,
    bias_layout: Layout,
    offsets_layout: Layout,
    ids_layout: Layout,
    stats_layout: Layout,
    out_layout: Layout,
](
    a: LayoutTensor[a_type, a_layout, MutAnyOrigin],
    b_packed: LayoutTensor[DType.uint8, packed_layout, MutAnyOrigin],
    scales: LayoutTensor[MXFP4_SF_DTYPE, scales_layout, MutAnyOrigin],
    bias: LayoutTensor[bias_type, bias_layout, MutAnyOrigin],
    expert_offsets: LayoutTensor[DType.uint32, offsets_layout, MutAnyOrigin],
    expert_ids: LayoutTensor[DType.int32, ids_layout, MutAnyOrigin],
    expert_usage_stats: LayoutTensor[DType.uint32, stats_layout, MutAnyOrigin],
    output: LayoutTensor[DType.float32, out_layout, MutAnyOrigin],
):
    """Float32 reference for grouped MXFP4 GEMM (no activation)."""
    var active_experts = Int(expert_usage_stats[1])
    var k_blocks = b_packed.dim(2)
    var input_k = a.dim(1)

    for expert_idx in range(active_experts):
        var token_start = Int(expert_offsets[expert_idx])
        var token_end = Int(expert_offsets[expert_idx + 1])
        var num_tokens = token_end - token_start
        if num_tokens <= 0:
            continue

        var expert = Int(expert_ids[expert_idx])
        for token_rel in range(num_tokens):
            var token = token_start + token_rel
            for n in range(b_packed.dim(1)):
                var acc = Float32(0.0)
                for kb in range(k_blocks):
                    var scale = scale_f32_cpu(scales, n, kb * MXFP4_BLOCK_K)

                    @parameter
                    for byte_idx in range(MXFP4_PACKED_BYTES_PER_BLOCK):
                        var k0 = kb * MXFP4_BLOCK_K + byte_idx * 2
                        var packed = b_packed[expert, n, kb, byte_idx]
                        var pair = decode_e2m1_pair_cpu(
                            rebind[Scalar[DType.uint8]](packed)
                        )
                        if k0 < input_k:
                            var a0 = rebind[Scalar[a_type]](a[token, k0]).cast[
                                DType.float32
                            ]()
                            acc += a0 * pair[0] * scale
                        if k0 + 1 < input_k:
                            var a1 = rebind[Scalar[a_type]](
                                a[token, k0 + 1]
                            ).cast[DType.float32]()
                            acc += a1 * pair[1] * scale

                var bias_val = rebind[Scalar[bias_type]](bias[expert, n]).cast[
                    DType.float32
                ]()
                output[token, n] = acc + bias_val


fn ref_grouped_swiglu_cpu[
    a_type: DType,
    a_layout: Layout,
    packed_layout: Layout,
    scales_layout: Layout,
    bias_type: DType,
    bias_layout: Layout,
    offsets_layout: Layout,
    ids_layout: Layout,
    stats_layout: Layout,
    out_layout: Layout,
](
    a: LayoutTensor[a_type, a_layout, MutAnyOrigin],
    gate_up_packed: LayoutTensor[DType.uint8, packed_layout, MutAnyOrigin],
    scales: LayoutTensor[MXFP4_SF_DTYPE, scales_layout, MutAnyOrigin],
    bias: LayoutTensor[bias_type, bias_layout, MutAnyOrigin],
    expert_offsets: LayoutTensor[DType.uint32, offsets_layout, MutAnyOrigin],
    expert_ids: LayoutTensor[DType.int32, ids_layout, MutAnyOrigin],
    expert_usage_stats: LayoutTensor[DType.uint32, stats_layout, MutAnyOrigin],
    output: LayoutTensor[DType.float32, out_layout, MutAnyOrigin],
):
    """Float32 reference for grouped MXFP4 GEMM + fused SwiGLU."""
    var active_experts = Int(expert_usage_stats[1])
    var k_blocks = gate_up_packed.dim(2)
    var input_k = a.dim(1)
    var two_moe_dim = gate_up_packed.dim(1)
    var moe_dim = two_moe_dim // 2

    for expert_idx in range(active_experts):
        var token_start = Int(expert_offsets[expert_idx])
        var token_end = Int(expert_offsets[expert_idx + 1])
        var num_tokens = token_end - token_start
        if num_tokens <= 0:
            continue

        var expert = Int(expert_ids[expert_idx])
        for token_rel in range(num_tokens):
            var token = token_start + token_rel
            for j in range(moe_dim):
                var gate_row = j * 2
                var up_row = gate_row + 1
                var gate_acc = Float32(0.0)
                var up_acc = Float32(0.0)

                for kb in range(k_blocks):
                    var gate_scale = scale_f32_cpu(
                        scales, gate_row, kb * MXFP4_BLOCK_K
                    )
                    var up_scale = scale_f32_cpu(
                        scales, up_row, kb * MXFP4_BLOCK_K
                    )

                    @parameter
                    for byte_idx in range(MXFP4_PACKED_BYTES_PER_BLOCK):
                        var k0 = kb * MXFP4_BLOCK_K + byte_idx * 2
                        var gate_pair = decode_e2m1_pair_cpu(
                            rebind[Scalar[DType.uint8]](
                                gate_up_packed[expert, gate_row, kb, byte_idx]
                            )
                        )
                        var up_pair = decode_e2m1_pair_cpu(
                            rebind[Scalar[DType.uint8]](
                                gate_up_packed[expert, up_row, kb, byte_idx]
                            )
                        )

                        if k0 < input_k:
                            var a0 = rebind[Scalar[a_type]](a[token, k0]).cast[
                                DType.float32
                            ]()
                            gate_acc += a0 * gate_pair[0] * gate_scale
                            up_acc += a0 * up_pair[0] * up_scale
                        if k0 + 1 < input_k:
                            var a1 = rebind[Scalar[a_type]](
                                a[token, k0 + 1]
                            ).cast[DType.float32]()
                            gate_acc += a1 * gate_pair[1] * gate_scale
                            up_acc += a1 * up_pair[1] * up_scale

                gate_acc += rebind[Scalar[bias_type]](
                    bias[expert, gate_row]
                ).cast[DType.float32]()
                up_acc += rebind[Scalar[bias_type]](bias[expert, up_row]).cast[
                    DType.float32
                ]()

                output[token, j] = swiglu_reference(gate_acc, up_acc)
