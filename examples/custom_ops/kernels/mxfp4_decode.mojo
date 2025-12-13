# mxfp4_decode.mojo
#
# Minimal helpers for MXFP4 (FP4 e2m1 packed in uint8 + per-32-value MX scale in float8_e8m0).
#
# This file intentionally avoids importing MAX internal modules so it can be
# built as part of the standalone `examples/custom_ops/kernels` Mojo package.

from math import exp, exp2

comptime BF16 = DType.bfloat16
comptime F16 = DType.float16
comptime F32 = DType.float32
comptime U8 = DType.uint8


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
    # Canonical FP4(E2M1) table used by GPT-OSS MXFP4:
    #   0..7  => +{0,0.5,1,1.5,2,3,4,6}
    #   8..15 => -{0,0.5,1,1.5,2,3,4,6}
    var sign = (code & 0x8) != 0
    var mag = code & 0x7

    var v: Scalar[F16]
    if mag == 0:
        v = 0.0
    elif mag == 1:
        v = 0.5
    elif mag == 2:
        v = 1.0
    elif mag == 3:
        v = 1.5
    elif mag == 4:
        v = 2.0
    elif mag == 5:
        v = 3.0
    elif mag == 6:
        v = 4.0
    else:
        v = 6.0

    return -v if sign else v


@always_inline
fn fp4_e2m1_to_f32(code: UInt8) -> Scalar[F32]:
    var sign = (code & 0x8) != 0
    var mag = code & 0x7

    var v: Scalar[F32]
    if mag == 0:
        v = 0.0
    elif mag == 1:
        v = 0.5
    elif mag == 2:
        v = 1.0
    elif mag == 3:
        v = 1.5
    elif mag == 4:
        v = 2.0
    elif mag == 5:
        v = 3.0
    elif mag == 6:
        v = 4.0
    else:
        v = 6.0

    return -v if sign else v


@always_inline
fn e8m0_to_f32(scale_exp: Scalar[U8]) -> Scalar[F32]:
    # E8M0 is an exponent-only power-of-two scale with bias 127.
    # E8M0 reserves 0xFF as NaN for the whole block; map to a quiet NaN BF16.
    if UInt8(scale_exp) == 0xFF:
        return 0.0

    var e = Int(UInt8(scale_exp)) - 127
    return exp2(Scalar[F32](e))


@always_inline
fn decode_mxfp4_byte_to_2xbf16_e8m0(
    packed: Scalar[U8],
    scale_exp: Scalar[U8],
) -> SIMD[BF16, 2]:
    var lo = UInt8(packed) & 0x0F
    var hi = UInt8(packed) >> 4
    var s = e8m0_to_f32(scale_exp)
    var out = SIMD[BF16, 2](0)
    out[0] = (fp4_e2m1_to_f32(lo) * s).cast[BF16]()
    out[1] = (fp4_e2m1_to_f32(hi) * s).cast[BF16]()
    return out


@always_inline
fn decode_mxfp4_byte_to_2xbf16_scaled(
    packed: Scalar[U8],
    scale: Scalar[F32],
) -> SIMD[BF16, 2]:
    var lo = UInt8(packed) & 0x0F
    var hi = UInt8(packed) >> 4
    var out = SIMD[BF16, 2](0)
    out[0] = (fp4_e2m1_to_f32(lo) * scale).cast[BF16]()
    out[1] = (fp4_e2m1_to_f32(hi) * scale).cast[BF16]()
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
