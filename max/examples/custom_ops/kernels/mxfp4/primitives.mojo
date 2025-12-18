# primitives.mojo
#
# Minimal MXFP4 scalar utilities required by the Mojo test suite.

from math import exp2, log2

from ..fp4_utils import MXFP4_SF_DTYPE

comptime U8 = DType.uint8
comptime F32 = DType.float32


@always_inline
fn e8m0_to_float32(sf: Scalar[MXFP4_SF_DTYPE]) -> Float32:
    # MXFP4 uses an exponent-only E8M0 scale (bias 127) stored as a raw byte.
    # Convert the exponent to a power-of-two scale.
    var exp_u8 = UInt8(rebind[Scalar[U8]](sf))
    if exp_u8 == 0xFF:
        return Float32(0.0)
    return exp2(Float32(Int(exp_u8) - 127))


@always_inline
fn float32_to_e8m0(x: Float32) -> Scalar[MXFP4_SF_DTYPE]:
    # This helper is only used in tests, which set `x` to exact powers-of-two.
    if x <= 0.0:
        return rebind[Scalar[MXFP4_SF_DTYPE]](UInt8(0))

    var e = Int(log2(x))
    var exp = e + 127
    if exp < 0:
        exp = 0
    if exp > 254:
        exp = 254
    return rebind[Scalar[MXFP4_SF_DTYPE]](UInt8(exp))
