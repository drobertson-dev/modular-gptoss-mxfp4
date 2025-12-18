# ===----------------------------------------------------------------------=== #
# SwiGLU activation semantics (CPU only).
#
# These tests pin down the scalar and SIMD SwiGLU helpers to match the Python
# reference exactly, including clamp behavior at the ±SWIGLU_LIMIT boundary.
# ===----------------------------------------------------------------------=== #

from kernels.moe_mxfp4 import swiglu_activation, swiglu_activation_simd
from testing import assert_almost_equal, assert_equal, TestSuite

from mxfp4_reference import (
    SWIGLU_LIMIT,
    swiglu_reference,
    swiglu_reference_simd,
)


fn test_swiglu_scalar_matches_reference() raises:
    var samples = SIMD[DType.float32, 9](
        -8.0,
        -7.0,
        -6.5,
        -1.25,
        0.0,
        1.0,
        6.9,
        7.0,
        7.1,
    )

    @parameter
    for i in range(9):

        @parameter
        for j in range(9):
            var gate = samples[i]
            var up = samples[j]
            var expected = swiglu_reference(gate, up)
            var got = swiglu_activation(gate, up)
            assert_almost_equal(got, expected, rtol=1e-6, atol=1e-6)


fn test_swiglu_simd_lane_equivalence() raises:
    var gate = SIMD[DType.float32, 8](
        -7.5,
        -7.0,
        -3.5,
        -0.5,
        0.5,
        3.25,
        6.75,
        7.5,
    )
    var up = SIMD[DType.float32, 8](
        -7.5,
        -6.9,
        -1.0,
        0.0,
        1.0,
        5.5,
        7.0,
        7.25,
    )

    var simd_out = swiglu_activation_simd[8](gate, up)
    var ref_out = swiglu_reference_simd[8](gate, up)

    @parameter
    for i in range(8):
        assert_almost_equal(simd_out[i], ref_out[i], rtol=1e-6, atol=1e-6)
        assert_equal(simd_out[i], swiglu_activation(gate[i], up[i]))


fn test_swiglu_clamp_edges() raises:
    # Values exactly at the clamp boundaries and just across them.
    var gate = SIMD[DType.float32, 6](
        -SWIGLU_LIMIT - Float32(0.5),
        -SWIGLU_LIMIT,
        -SWIGLU_LIMIT + Float32(0.01),
        SWIGLU_LIMIT - Float32(0.01),
        SWIGLU_LIMIT,
        SWIGLU_LIMIT + Float32(0.5),
    )
    var up = SIMD[DType.float32, 6](
        -SWIGLU_LIMIT - Float32(0.25),
        -SWIGLU_LIMIT,
        -SWIGLU_LIMIT + Float32(0.05),
        SWIGLU_LIMIT - Float32(0.05),
        SWIGLU_LIMIT,
        SWIGLU_LIMIT + Float32(0.25),
    )

    @parameter
    for i in range(6):
        var expected = swiglu_reference(gate[i], up[i])
        var got = swiglu_activation(gate[i], up[i])
        assert_almost_equal(got, expected, rtol=1e-6, atol=1e-6)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
