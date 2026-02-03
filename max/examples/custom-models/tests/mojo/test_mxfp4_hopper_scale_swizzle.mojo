# ===----------------------------------------------------------------------=== #
# Hopper MXFP4 scale swizzle tests (CPU only).
# Validates forward + inverse index mappings for the swizzled scale layout.
# ===----------------------------------------------------------------------=== #

from testing import assert_equal, assert_true, TestSuite

from mxfp4.layout_hopper import (
    HOPPER_SCALE_NUM_WARPS,
    hopper_scale_swizzle_index,
    hopper_scale_unswizzle_index,
)


fn test_hopper_scale_swizzle_roundtrip() raises:
    # Cover multiple num_warps values to match Triton reference behavior.
    @parameter
    for num_warps in range(1, 9, 1):
        if num_warps != 1 and num_warps != 2 and num_warps != 4 and num_warps != 8:
            continue

        # Use a minimal aligned shape: M must be multiple of 32*num_warps.
        var M = 32 * num_warps
        var Kblocks = 4  # K = 128, Kblocks must be even for Hopper swizzle
        var M2 = M // 32
        var K2 = Kblocks * 32

        # Forward swizzle: logical (m, k) -> stored (m2, k2).
        for m in range(M):
            for k in range(Kblocks):
                var idx = hopper_scale_swizzle_index[num_warps](m, k)
                assert_true(idx[0] >= 0)
                assert_true(idx[0] < M2)
                assert_true(idx[1] >= 0)
                assert_true(idx[1] < K2)
                var back = hopper_scale_unswizzle_index[num_warps](
                    idx[0], idx[1]
                )
                assert_equal(back[0], m)
                assert_equal(back[1], k)

        # Inverse: stored (m2, k2) -> logical (m, k), then forward again.
        for m2 in range(M2):
            for k2 in range(K2):
                var back = hopper_scale_unswizzle_index[num_warps](m2, k2)
                assert_true(back[0] >= 0)
                assert_true(back[0] < M)
                assert_true(back[1] >= 0)
                assert_true(back[1] < Kblocks)
                var idx = hopper_scale_swizzle_index[num_warps](
                    back[0], back[1]
                )
                assert_equal(idx[0], m2)
                assert_equal(idx[1], k2)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
