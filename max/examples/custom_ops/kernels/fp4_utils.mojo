# fp4_utils.mojo
#
# Small FP4(E2M1) + MXFP4 scale helpers used by the MXFP4 test suite.
#
# Keep this file self-contained (no imports from upstream `max/` sources) so it
# can be compiled from within the `examples/custom_ops/kernels` package.

comptime SF_ATOM_M = (32, 4)
comptime SF_ATOM_K = 4
comptime SF_MN_GROUP_SIZE = SF_ATOM_M[0] * SF_ATOM_M[1]  # 128

comptime MXFP4_SF_VECTOR_SIZE = 32
comptime MXFP4_SF_DTYPE = DType.uint8

comptime E2M1_TO_FLOAT32 = SIMD[DType.float32, 16](
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
