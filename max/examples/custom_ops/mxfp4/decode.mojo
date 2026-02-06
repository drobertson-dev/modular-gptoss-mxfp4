# mxfp4_decode.mojo
#
# Minimal helpers for MXFP4 (FP4 e2m1 packed in uint8 + per-32-value MX scale in float8_e8m0).
#
# This file intentionally avoids importing MAX internal modules so it can be
# built as part of the standalone `examples/custom_ops/mxfp4` Mojo package.

from math import exp
from memory import bitcast

from .layout_hopper import (
    MXFP4_PACK_MASK_U32,
    MXFP4_D1_MASK_U32,
    MXFP4_D3_MASK_U32,
    MXFP4_D6_MASK_U32,
)

comptime BF16 = DType.bfloat16
comptime F16 = DType.float16
comptime F32 = DType.float32
comptime U8 = DType.uint8
comptime U16 = DType.uint16
comptime U32 = DType.uint32

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


comptime _MXFP4_FP4_BIAS_BF16 = bitcast[BF16, 1](UInt16(0x7E80))  # 2**126


@always_inline
fn mxfp4_bias_scale2(scale: Scalar[BF16]) -> SIMD[BF16, 2]:
    var bias2 = SIMD[BF16, 2](_MXFP4_FP4_BIAS_BF16, _MXFP4_FP4_BIAS_BF16)
    var scale2 = SIMD[BF16, 2](scale, scale)
    return bias2 * scale2


@always_inline
fn _msb_index_u7(mant: UInt16) -> UInt16:
    if (mant & UInt16(0x40)) != UInt16(0):
        return UInt16(6)
    if (mant & UInt16(0x20)) != UInt16(0):
        return UInt16(5)
    if (mant & UInt16(0x10)) != UInt16(0):
        return UInt16(4)
    if (mant & UInt16(0x08)) != UInt16(0):
        return UInt16(3)
    if (mant & UInt16(0x04)) != UInt16(0):
        return UInt16(2)
    if (mant & UInt16(0x02)) != UInt16(0):
        return UInt16(1)
    return UInt16(0)


@always_inline
fn _bf16_scale_e8m0_bits(
    value_bits: UInt16,
    scale_exp: Scalar[U8],
) -> UInt16:
    var sign = value_bits & UInt16(0x8000)
    var exp = (value_bits >> UInt16(7)) & UInt16(0xFF)
    var mant = value_bits & UInt16(0x7F)

    if scale_exp == UInt8(0xFF):
        return sign
    if scale_exp >= UInt8(129):
        if exp == UInt16(0) and mant == UInt16(0):
            return UInt16(0x7FC0)
        if sign != UInt16(0):
            return UInt16(0xFF80)
        return UInt16(0x7F80)
    if exp == UInt16(0):
        if mant == UInt16(0):
            return sign
        if scale_exp == UInt8(0):
            var mant_new = mant >> UInt16(1)
            if (mant & UInt16(1)) != UInt16(0) and (mant_new & UInt16(1)) != UInt16(0):
                mant_new += UInt16(1)
            return sign | (mant_new & UInt16(0x7F))
        var k = UInt16(scale_exp) - UInt16(1)
        var p = _msb_index_u7(mant)
        var kp = k + p
        if kp >= UInt16(7):
            var exp_new = kp - UInt16(6)
            var mant_new = (
                (mant - (UInt16(1) << p)) << (UInt16(7) - p)
            )
            return sign | (exp_new << UInt16(7)) | (mant_new & UInt16(0x7F))
        var mant_new = mant << k
        return sign | (mant_new & UInt16(0x7F))

    if scale_exp == UInt8(0):
        if exp > UInt16(1):
            var exp_new = exp - UInt16(1)
            return sign | (exp_new << UInt16(7)) | mant
        var mant_full = UInt16(0x80) | mant
        var mant_new = mant_full >> UInt16(1)
        if (mant_full & UInt16(1)) != UInt16(0) and (mant_new & UInt16(1)) != UInt16(0):
            mant_new += UInt16(1)
        return sign | (mant_new & UInt16(0x7F))

    var k = UInt16(scale_exp) - UInt16(1)
    var exp_new = exp + k
    if exp_new >= UInt16(0xFF):
        return sign | UInt16(0x7F80)
    return sign | (exp_new << UInt16(7)) | mant


@always_inline
fn _bf16x2_scale_e8m0_bits(
    packed_bits: UInt32,
    scale_exp: Scalar[U8],
) -> SIMD[BF16, 2]:
    var lo_bits = UInt16(packed_bits & UInt32(0xFFFF))
    var hi_bits = UInt16(packed_bits >> UInt32(16))
    var lo_out = _bf16_scale_e8m0_bits(lo_bits, scale_exp)
    var hi_out = _bf16_scale_e8m0_bits(hi_bits, scale_exp)
    return SIMD[BF16, 2](
        bitcast[BF16, 1](lo_out),
        bitcast[BF16, 1](hi_out),
    )


@always_inline
fn decode_mxfp4_packbits_u32_to_8xbf16_bias_scaled(
    packed_bits: UInt32,
    bias2: SIMD[BF16, 2],
    scale2: SIMD[BF16, 2],
) -> SIMD[BF16, 8]:
    var x = packed_bits

    var y0_bits = x & MXFP4_PACK_MASK_U32
    var y1_bits = (x << UInt32(3)) & MXFP4_PACK_MASK_U32
    var y2_bits = (x << UInt32(6)) & MXFP4_PACK_MASK_U32

    var d1 = (x << UInt32(1)) & MXFP4_D1_MASK_U32
    var d3 = (x >> UInt32(3)) & MXFP4_D3_MASK_U32
    var d6 = (x >> UInt32(7)) & MXFP4_D6_MASK_U32
    var y3_bits = d1 | d3 | d6

    # Keep multiplication in two stages to avoid overflowing (2**126 * scale)
    # when scale exponent bytes exceed 128.
    var v0 = (bitcast[BF16, 2](y0_bits) * bias2) * scale2
    var v1 = (bitcast[BF16, 2](y1_bits) * bias2) * scale2
    var v2 = (bitcast[BF16, 2](y2_bits) * bias2) * scale2
    var v3 = (bitcast[BF16, 2](y3_bits) * bias2) * scale2

    return SIMD[BF16, 8](
        v0[0],
        v0[1],
        v1[0],
        v1[1],
        v2[0],
        v2[1],
        v3[0],
        v3[1],
    )


@always_inline
fn decode_mxfp4_packbits_u32_to_8xbf16_scaled_e8m0(
    packed_bits: UInt32,
    scale_exp: Scalar[U8],
) -> SIMD[BF16, 8]:
    """Decode 4 packed MXFP4 bytes (8 FP4 values) using E8M0 exponent-add.

    This is the arithmetic-free path for MXFP4 E8M0 scales:
    unpack FP4 bits to BF16 bit patterns, then apply the shared scale exponent
    by bit-level exponent adjustment (no BF16 multiply in this path).
    """
    var x = packed_bits

    var y0_bits = x & MXFP4_PACK_MASK_U32
    var y1_bits = (x << UInt32(3)) & MXFP4_PACK_MASK_U32
    var y2_bits = (x << UInt32(6)) & MXFP4_PACK_MASK_U32

    var d1 = (x << UInt32(1)) & MXFP4_D1_MASK_U32
    var d3 = (x >> UInt32(3)) & MXFP4_D3_MASK_U32
    var d6 = (x >> UInt32(7)) & MXFP4_D6_MASK_U32
    var y3_bits = d1 | d3 | d6

    var v0 = _bf16x2_scale_e8m0_bits(y0_bits, scale_exp)
    var v1 = _bf16x2_scale_e8m0_bits(y1_bits, scale_exp)
    var v2 = _bf16x2_scale_e8m0_bits(y2_bits, scale_exp)
    var v3 = _bf16x2_scale_e8m0_bits(y3_bits, scale_exp)

    return SIMD[BF16, 8](
        v0[0],
        v0[1],
        v1[0],
        v1[1],
        v2[0],
        v2[1],
        v3[0],
        v3[1],
    )


@always_inline
fn decode_mxfp4_packbits_u32_to_8xbf16_scaled(
    packed_bits: UInt32,
    scale: Scalar[BF16],
) -> SIMD[BF16, 8]:
    """Decode 4 packed MXFP4 bytes (8 FP4 values) into 8 BF16 values.

    This expects the input bytes have been preprocessed by the Hopper `_pack_bits`
    transform (offline, in the weight adapter). The unpack sequence mirrors the
    Triton `matmul_ogs` Hopper path: mask/shift to BF16 bit patterns, then
    multiply by a BF16 bias (2**126) to add the missing exponent bias, and then
    multiply by the per-32 E8M0 scale (BF16).
    """
    var bias2 = SIMD[BF16, 2](_MXFP4_FP4_BIAS_BF16, _MXFP4_FP4_BIAS_BF16)
    var scale2 = SIMD[BF16, 2](scale, scale)
    return decode_mxfp4_packbits_u32_to_8xbf16_bias_scaled(
        packed_bits,
        bias2,
        scale2,
    )




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
