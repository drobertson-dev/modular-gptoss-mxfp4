# mxfp4_decode.mojo
#
# Minimal helpers for MXFP4 (FP4 e2m1 packed in uint8 + per-32-value MX scale in float8_e8m0).
#
# This file intentionally avoids importing MAX internal modules so it can be
# built as part of the standalone `examples/custom_ops/kernels` Mojo package.

from math import exp
from memory import bitcast

comptime BF16 = DType.bfloat16
comptime F16 = DType.float16
comptime F32 = DType.float32
comptime U8 = DType.uint8

comptime FP4_E2M1_LUT_BF16 = SIMD[BF16, 16](
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

comptime FP4_E2M1_LUT_F16 = SIMD[F16, 16](
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
fn sigmoid_f32(x: Scalar[F32]) -> Scalar[F32]:
    return 1.0 / (1.0 + exp(-x))


@always_inline
fn swiglu_pair(
    glu: Scalar[F32],
    lin: Scalar[F32],
    alpha: Scalar[F32],
    limit: Scalar[F32],
) -> Scalar[F32]:
    var x_glu = glu
    if x_glu > limit:
        x_glu = limit

    var x_lin = lin
    if x_lin > limit:
        x_lin = limit
    if x_lin < -limit:
        x_lin = -limit

    var out_glu = x_glu * sigmoid_f32(alpha * x_glu)
    return out_glu * (x_lin + 1.0)


@always_inline
fn fp4_e2m1_to_f16(code: UInt8) -> Scalar[F16]:
    return FP4_E2M1_LUT_F16[Int(code & UInt8(0xF))]


@always_inline
fn e8m0_to_bf16_bits(scale_exp: Scalar[U8]) -> Scalar[BF16]:
    # E8M0 is an exponent-only power-of-two scale with bias 127.
    #
    # Exact BF16 bit construction:
    # - scale_exp == 0xFF: reserved/NaN -> map to BF16(0)
    # - scale_exp == 0x00: represent 2^-127 as BF16 subnormal (bits 0x0040)
    # - otherwise: BF16 bits = UInt16(scale_exp) << 7 (zero mantissa)
    var exp_u8 = UInt8(scale_exp)
    var bits: UInt16
    if exp_u8 == 0xFF:
        bits = UInt16(0)
    elif exp_u8 == 0x00:
        bits = UInt16(0x0040)
    else:
        bits = UInt16(exp_u8) << 7
    return bitcast[BF16, 1](bits)


@always_inline
fn decode_mxfp4_byte_to_2xbf16_e8m0(
    packed: Scalar[U8],
    scale_exp: Scalar[U8],
) -> SIMD[BF16, 2]:
    var lo = UInt8(packed) & 0x0F
    var hi = UInt8(packed) >> 4
    var s = e8m0_to_bf16_bits(scale_exp)
    var out = SIMD[BF16, 2](0)
    out[0] = (FP4_E2M1_LUT_BF16[Int(lo)] * s).cast[BF16]()
    out[1] = (FP4_E2M1_LUT_BF16[Int(hi)] * s).cast[BF16]()
    return out


@always_inline
fn decode_mxfp4_byte_to_2xbf16_scaled(
    packed: Scalar[U8],
    scale: Scalar[BF16],
) -> SIMD[BF16, 2]:
    var lo = UInt8(packed) & 0x0F
    var hi = UInt8(packed) >> 4
    var out = SIMD[BF16, 2](0)
    out[0] = (FP4_E2M1_LUT_BF16[Int(lo)] * scale).cast[BF16]()
    out[1] = (FP4_E2M1_LUT_BF16[Int(hi)] * scale).cast[BF16]()
    return out


@always_inline
fn decode_mxfp4_byte_to_2xf16[
    scale_dtype: DType,
](packed: Scalar[U8], scale: Scalar[scale_dtype],) -> SIMD[F16, 2]:
    # Two FP4 values per byte (low/high nibble), scaled by the shared E8M0 scale.
    var lo = UInt8(packed) & 0x0F
    var hi = UInt8(packed) >> 4

    var s: Scalar[F16] = scale.cast[F16]()

    var out = SIMD[F16, 2](0)
    out[0] = fp4_e2m1_to_f16(lo) * s
    out[1] = fp4_e2m1_to_f16(hi) * s
    return out
