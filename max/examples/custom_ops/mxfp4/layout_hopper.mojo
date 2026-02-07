# layout_hopper.mojo
#
# Hopper MXFP4 layout contract helpers for value packbits + scale swizzle.
#
# This module centralizes the bit-interleave (packbits) constants and the
# Hopper MXFP4 scale swizzle mapping so packers, decoders, and kernels stay
# aligned.

from utils.index import IndexList

comptime U32 = DType.uint32

# NOTE: Update both Mojo and Python if this changes.
comptime HOPPER_SCALE_NUM_WARPS = 4
comptime HOPPER_SCALE_ALIGN_M = 32 * HOPPER_SCALE_NUM_WARPS
comptime HOPPER_SCALE_ALIGN_K = 2

# Packbits mask constants (Hopper MXFP4 value layout).
comptime MXFP4_PACK_MASK_U32 = UInt32(0x81C081C0)
comptime MXFP4_D1_MASK_U32 = UInt32(0x80008000)
comptime MXFP4_D3_MASK_U32 = UInt32(0x01800180)
comptime MXFP4_D6_MASK_U32 = UInt32(0x00400040)

@always_inline
fn hopper_value_swizzle_index(m: Int, k: Int) -> IndexList[2]:
    """Forward map logical value coords (m, k_byte) -> swizzled (m2, k2).

    Layout matches HopperMXValueLayout for MMA v3 (u8_kwidth=1):
      - Input bytes: shape (M, Kbytes)
      - Output bytes: shape (M/4, Kbytes*4)
    """
    var m0 = m >> 4  # M // 16
    var m_rem = m & 15
    var b0 = m_rem >> 3          # warp_tile[0]
    var c0 = (m_rem & 7) >> 1    # threads[0]
    var s0 = m_rem & 1           # scott_trick[0]

    var k0 = k >> 5              # Kbytes // 32
    var k_rem = k & 31
    var c1 = k_rem & 3           # threads[1]
    var b1 = (k_rem >> 2) & 1    # warp_tile[1]
    var a1 = (k_rem >> 3) & 3    # k_tile[1]

    var m2 = m0 * 4 + c0
    var k2 = (
        (((k0 * 2 + s0) * 4 + c1) * 4 + a1) * 2 + b1
    ) * 2 + b0

    return IndexList[2](m2, k2)


@always_inline
fn hopper_scale_swizzle_index[
    NUM_WARPS: Int = HOPPER_SCALE_NUM_WARPS,
](m: Int, k: Int) -> IndexList[2]:
    """Forward map logical scale coords (m, k) -> stored swizzled (m2, k2).

    Logical shape: (M, K) where M is N (output columns), K is K/32.
    Stored shape: (M/32, K*32).
    """
    var m0 = m // (32 * NUM_WARPS)
    var r = m - m0 * (32 * NUM_WARPS)

    var t1 = r // (NUM_WARPS * 16)
    var r1 = r - t1 * (NUM_WARPS * 16)

    var w = r1 // 16
    var r2 = r1 - w * 16

    var t3 = r2 // 8
    var c = r2 - t3 * 8

    var k0 = k // 2
    var d = k - k0 * 2

    var m2 = m0 * NUM_WARPS + w
    var k2 = (
        (((k0 * 2 + t1) * 8 + c) * 2 + d) * 2 + t3
    )

    return IndexList[2](m2, k2)


@always_inline
fn hopper_scale_swizzle_index_fast(m: Int, k: Int) -> IndexList[2]:
    """Fast path for NUM_WARPS=4 using shifts/masks (M aligned to 128)."""
    # NUM_WARPS == 4 so 32 * NUM_WARPS == 128 and NUM_WARPS * 16 == 64.
    var m0 = m >> 7
    var r = m & 127
    var t1 = r >> 6
    var w = (r >> 4) & 3
    var t3 = (r >> 3) & 1
    var c = r & 7

    var k0 = k >> 1
    var d = k & 1

    var m2 = m0 * 4 + w
    var k2 = (
        (((k0 * 2 + t1) * 8 + c) * 2 + d) * 2 + t3
    )
    return IndexList[2](m2, k2)


@always_inline
fn hopper_scale_unswizzle_index[
    NUM_WARPS: Int = HOPPER_SCALE_NUM_WARPS,
](m2: Int, k2: Int) -> IndexList[2]:
    """Inverse map stored swizzled coords (m2, k2) -> logical (m, k)."""
    var m0 = m2 // NUM_WARPS
    var w = m2 - m0 * NUM_WARPS

    var t3 = k2 & 1
    var q = k2 >> 1
    var d = q & 1
    q = q >> 1
    var c = q & 7
    q = q >> 3
    var t1 = q & 1
    var k0 = q >> 1

    var k = k0 * 2 + d
    var m = (
        m0 * (32 * NUM_WARPS)
        + t1 * (NUM_WARPS * 16)
        + w * 16
        + t3 * 8
        + c
    )

    return IndexList[2](m, k)
