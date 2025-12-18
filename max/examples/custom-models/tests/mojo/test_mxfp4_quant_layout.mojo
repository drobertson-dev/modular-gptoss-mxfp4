# ===----------------------------------------------------------------------=== #
# MXFP4 quantization + layout tests (CPU only).
#
# These tests validate the fundamental FP4/E8M0 decode and scale math, along
# with the packed block addressing used by the MXFP4 kernels. They serve as the
# canonical scalar reference separate from the GPU kernel sources.
# ===----------------------------------------------------------------------=== #

from buffer import Dim
from buffer.dimlist import DimList
from ndbuffer_utils import HostNDBuffer
from kernels.fp4_utils import E2M1_TO_FLOAT32
from kernels.mxfp4 import (
    dequant_row_cpu,
    mxfp4_address,
    MXFP4_BLOCK_K,
    MXFP4_PACKED_BYTES_PER_BLOCK,
    MXFP4_SF_DTYPE,
    set_mxfp4_scale,
)
from kernels.mxfp4.primitives import float32_to_e8m0
from layout._ndbuffer_stub import from_ndbuffer_row_major
from testing import assert_almost_equal, assert_equal, TestSuite

from mxfp4_reference import (
    decode_e2m1_pair_cpu,
    fill_packed_by_nibble,
    make_scales_host,
    scale_f32_cpu,
)


fn _nibble_from_k(expert: Int, out_col: Int, k: Int) -> UInt8:
    """Helper for fill_packed_by_nibble: returns k % 16 as nibble."""
    return UInt8(k % 16)


fn test_e2m1_decode_table_cpu() raises:
    # Every nibble code should decode exactly to the LUT entry.
    # Pack each code in both low and high nibble positions to test both.
    @parameter
    for code in range(16):
        # Pack code into both nibbles: low nibble = code, high nibble = code
        var packed = Scalar[DType.uint8](UInt8(code | (code << 4)))
        var decoded = decode_e2m1_pair_cpu(packed)
        assert_equal(decoded[0], E2M1_TO_FLOAT32[code])
        assert_equal(decoded[1], E2M1_TO_FLOAT32[code])


fn test_decode_e2m1_pair_known_bytes() raises:
    # Spot-check a few packed bytes that mix low/high nibbles.
    var cases = SIMD[DType.uint8, 4](0x00, 0xFF, 0x0F, 0xF0)

    @parameter
    for i in range(4):
        var packed = Scalar[DType.uint8](cases[i])
        var pair = decode_e2m1_pair_cpu(packed)
        var lo = Int(cases[i] & UInt8(0xF))
        var hi = Int((cases[i] >> 4) & UInt8(0xF))
        assert_equal(pair[0], E2M1_TO_FLOAT32[lo])
        assert_equal(pair[1], E2M1_TO_FLOAT32[hi])


fn test_e8m0_scale_to_float() raises:
    alias num_rows = 3
    alias K = MXFP4_BLOCK_K
    var scales_host = make_scales_host[num_rows, K]()
    var scales_tensor = from_ndbuffer_row_major(scales_host.tensor)

    # Choose exponents around zero, mid-range, and a large value.
    var exponents = SIMD[DType.uint8, num_rows](127, 120, 200)

    @parameter
    for row in range(num_rows):
        var sf = rebind[Scalar[MXFP4_SF_DTYPE]](exponents[row])
        set_mxfp4_scale(scales_tensor, row, 0, sf)
        var expected = Float32(2.0) ** Float32(Int(exponents[row]) - 127)
        var got = scale_f32_cpu(scales_tensor, row, 0)
        assert_almost_equal(got, expected, rtol=1e-7, atol=1e-7)


fn test_mxfp4_block_addressing_and_scales() raises:
    # Verify the mapping (k -> byte/nibble) and matching scale selection.
    alias num_experts = 1
    alias N = 4
    alias K = MXFP4_BLOCK_K * 2  # two MXFP4 blocks
    alias packed_shape = DimList(
        num_experts, N, K // MXFP4_BLOCK_K, MXFP4_PACKED_BYTES_PER_BLOCK
    )

    var packed_host = HostNDBuffer[DType.uint8, 4, packed_shape](packed_shape)
    var packed_tensor = from_ndbuffer_row_major(packed_host.tensor)
    fill_packed_by_nibble(packed_tensor, K // MXFP4_BLOCK_K, _nibble_from_k)

    # Distinct scales per block (must be powers of 2 for E8M0 exactness).
    var block_scales = SIMD[DType.float32, 4](1.0, 2.0, 4.0, 8.0)
    var scales_host = make_scales_host[N, K]()
    var scales_tensor = from_ndbuffer_row_major(scales_host.tensor)

    @parameter
    for kb in range(K // MXFP4_BLOCK_K):
        set_mxfp4_scale(
            scales_tensor,
            0,
            kb * MXFP4_BLOCK_K,
            float32_to_e8m0(block_scales[kb]),
        )
        set_mxfp4_scale(
            scales_tensor,
            1,
            kb * MXFP4_BLOCK_K,
            float32_to_e8m0(block_scales[kb]),
        )
        set_mxfp4_scale(
            scales_tensor,
            2,
            kb * MXFP4_BLOCK_K,
            float32_to_e8m0(block_scales[kb]),
        )
        set_mxfp4_scale(
            scales_tensor,
            3,
            kb * MXFP4_BLOCK_K,
            float32_to_e8m0(block_scales[kb]),
        )

    # Dequantize on CPU and confirm each k maps to the right nibble and scale.
    alias out_shape = DimList(Dim(K))
    var out_row = HostNDBuffer[DType.float32, 1, out_shape](out_shape)
    var out_tensor = from_ndbuffer_row_major(out_row.tensor)
    for n in range(N):
        # Clear and dequantize this row using the shared helper.
        for k in range(K):
            out_tensor[k] = Float32(0.0)
        dequant_row_cpu(
            packed_tensor,
            scales_tensor,
            0,  # expert
            n,  # row/out_col
            K,
            MXFP4_BLOCK_K,
            out_tensor,
        )

        for k in range(K):
            var expected = (
                E2M1_TO_FLOAT32[k % 16] * block_scales[k // MXFP4_BLOCK_K]
            )
            assert_almost_equal(out_tensor[k], expected, rtol=0.0, atol=0.0)
            # Addressing sanity: nibble index must match k parity within the block.
            var addr = mxfp4_address(k, MXFP4_BLOCK_K)
            assert_equal(addr[2], k % 2)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
