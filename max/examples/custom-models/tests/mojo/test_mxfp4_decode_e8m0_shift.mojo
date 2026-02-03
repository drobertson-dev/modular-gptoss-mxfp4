# ===----------------------------------------------------------------------=== #
# MXFP4 E8M0 exponent-add decode tests (CPU only).
#
# Confirms the integer exponent-add path matches the existing bias*scale decode
# for Hopper packbits input.
# ===----------------------------------------------------------------------=== #

from memory import bitcast
from testing import assert_equal, TestSuite

from mxfp4.decode import (
    decode_mxfp4_packbits_u32_to_8xbf16_scaled,
    decode_mxfp4_packbits_u32_to_8xbf16_scaled_e8m0,
    e8m0_to_bf16_bits,
)

comptime BF16 = DType.bfloat16
comptime U8 = DType.uint8
comptime U16 = DType.uint16
comptime U32 = DType.uint32


@always_inline
fn _bf16_bits(val: Scalar[BF16]) -> UInt16:
    return bitcast[U16, 1](val)


fn test_mxfp4_packbits_e8m0_shift_matches_bias_mul() raises:
    var scale_exps = SIMD[U8, 8](0, 1, 2, 10, 127, 200, 254, 255)
    var packed_cases = SIMD[U32, 6](
        0x00000000,
        0xFFFFFFFF,
        0x12345678,
        0x75417657,
        0xCD55C80A,
        0x80204010,
    )

    @parameter
    for case_idx in range(6):
        var packed = UInt32(packed_cases[case_idx])

        @parameter
        for s in range(8):
            var exp = UInt8(scale_exps[s])
            var scale = e8m0_to_bf16_bits(exp)
            var ref_vals = decode_mxfp4_packbits_u32_to_8xbf16_scaled(
                packed,
                scale,
            )
            var got_vals = decode_mxfp4_packbits_u32_to_8xbf16_scaled_e8m0(
                packed,
                exp,
            )

            @parameter
            for i in range(8):
                assert_equal(
                    _bf16_bits(ref_vals[i]), _bf16_bits(got_vals[i])
                )


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
