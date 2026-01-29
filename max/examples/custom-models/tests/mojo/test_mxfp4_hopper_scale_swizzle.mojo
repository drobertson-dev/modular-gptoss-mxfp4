# ===----------------------------------------------------------------------=== #
# Hopper MXFP4 scale swizzle tests (CPU only).
# Validates forward + inverse index mappings for the swizzled scale layout.
# ===----------------------------------------------------------------------=== #

from testing import assert_equal, assert_true, TestSuite

from mxfp4_grouped_kernels.hopper_mxfp4_layout import (
    HOPPER_SCALE_NUM_WARPS,
    hopper_scale_swizzle_index,
    hopper_scale_unswizzle_index,
)


fn test_hopper_scale_swizzle_roundtrip() raises:
    # Use a minimal aligned shape: M must be multiple of 32*num_warps.
    comptime M = 128
    comptime Kblocks = 4  # K = 128, Kblocks must be even for Hopper swizzle
    comptime M2 = M // 32
    comptime K2 = Kblocks * 32

    # Forward swizzle: logical (m, k) -> stored (m2, k2).
    for m in range(M):
        for k in range(Kblocks):
            var idx = hopper_scale_swizzle_index[
                HOPPER_SCALE_NUM_WARPS
            ](m, k)
            assert_true(idx[0] >= 0)
            assert_true(idx[0] < M2)
            assert_true(idx[1] >= 0)
            assert_true(idx[1] < K2)
            var back = hopper_scale_unswizzle_index[
                HOPPER_SCALE_NUM_WARPS
            ](idx[0], idx[1])
            assert_equal(back[0], m)
            assert_equal(back[1], k)

    # Inverse: stored (m2, k2) -> logical (m, k), then forward again.
    for m2 in range(M2):
        for k2 in range(K2):
            var back = hopper_scale_unswizzle_index[
                HOPPER_SCALE_NUM_WARPS
            ](m2, k2)
            assert_true(back[0] >= 0)
            assert_true(back[0] < M)
            assert_true(back[1] >= 0)
            assert_true(back[1] < Kblocks)
            var idx = hopper_scale_swizzle_index[
                HOPPER_SCALE_NUM_WARPS
            ](back[0], back[1])
            assert_equal(idx[0], m2)
            assert_equal(idx[1], k2)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
