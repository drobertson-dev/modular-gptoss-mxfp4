# mxfp4/__init__.mojo
#
# MXFP4 packing conventions + scale layout helpers for tests.

from ..fp4_utils import (
    E2M1_TO_FLOAT32,
    MXFP4_SF_DTYPE,
    MXFP4_SF_VECTOR_SIZE,
    SF_ATOM_K,
    SF_ATOM_M,
    SF_MN_GROUP_SIZE,
)
from .primitives import e8m0_to_float32

from layout import Layout, LayoutTensor

comptime U8 = DType.uint8
comptime F32 = DType.float32

comptime MXFP4_BLOCK_K = 32
comptime MXFP4_PACKED_BYTES_PER_BLOCK = 16


@always_inline
fn mxfp4_address(k: Int, block_k: Int = MXFP4_BLOCK_K) -> SIMD[DType.int32, 3]:
    """Return (k_block, byte_in_block, nibble_in_byte) for logical k."""
    var k_block = k // block_k
    var k_in_block = k - k_block * block_k
    var byte_in_block = k_in_block // 2
    var nibble = k_in_block - byte_in_block * 2
    return SIMD[DType.int32, 3](k_block, byte_in_block, nibble)


@always_inline
fn set_mxfp4_scale[
    scales_layout: Layout,
    //,
](
    scales: LayoutTensor[MXFP4_SF_DTYPE, scales_layout, MutAnyOrigin],
    row: Int,
    k_block_start: Int,
    value: Scalar[MXFP4_SF_DTYPE],
):
    scales[
        row // SF_MN_GROUP_SIZE,
        k_block_start // (MXFP4_SF_VECTOR_SIZE * SF_ATOM_K),
        row % SF_ATOM_M[0],
        (row % SF_MN_GROUP_SIZE) // SF_ATOM_M[0],
        (k_block_start // MXFP4_SF_VECTOR_SIZE) % SF_ATOM_K,
    ] = rebind[Scalar[MXFP4_SF_DTYPE]](value)


@always_inline
fn get_mxfp4_scale[
    scales_layout: Layout,
    //,
](
    scales: LayoutTensor[MXFP4_SF_DTYPE, scales_layout, MutAnyOrigin],
    row: Int,
    k_block_start: Int,
) -> Scalar[MXFP4_SF_DTYPE]:
    return rebind[Scalar[MXFP4_SF_DTYPE]](
        scales[
            row // SF_MN_GROUP_SIZE,
            k_block_start // (MXFP4_SF_VECTOR_SIZE * SF_ATOM_K),
            row % SF_ATOM_M[0],
            (row % SF_MN_GROUP_SIZE) // SF_ATOM_M[0],
            (k_block_start // MXFP4_SF_VECTOR_SIZE) % SF_ATOM_K,
        ]
    )


fn dequant_row_cpu[
    packed_layout: Layout,
    scales_layout: Layout,
    out_layout: Layout,
    //,
](
    packed: LayoutTensor[DType.uint8, packed_layout, MutAnyOrigin],
    scales: LayoutTensor[MXFP4_SF_DTYPE, scales_layout, MutAnyOrigin],
    expert: Int,
    row: Int,
    k: Int,
    block_k: Int,
    output: LayoutTensor[DType.float32, out_layout, MutAnyOrigin],
):
    """Decode one MXFP4 row into dense float32 (CPU helper)."""
    var k_blocks = (k + block_k - 1) // block_k
    for kb in range(k_blocks):
        var sf = get_mxfp4_scale(scales, row, kb * block_k)
        var scale = e8m0_to_float32(sf)

        @parameter
        for byte_idx in range(MXFP4_PACKED_BYTES_PER_BLOCK):
            var k0 = kb * block_k + byte_idx * 2
            if k0 >= k:
                continue
            var packed_byte = rebind[Scalar[U8]](
                packed[expert, row, kb, byte_idx]
            )
            var raw = UInt8(packed_byte.cast[U8]().to_bits())
            var lo = Int(raw & UInt8(0xF))
            var hi = Int((raw >> 4) & UInt8(0xF))

            output[k0] = E2M1_TO_FLOAT32[lo] * scale
            if k0 + 1 < k:
                output[k0 + 1] = E2M1_TO_FLOAT32[hi] * scale
