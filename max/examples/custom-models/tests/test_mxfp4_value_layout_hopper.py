import numpy as np

from gpt_oss_mxfp4_v3.weight_adapters import (
    _mxfp4_swizzle_values_hopper,
    _mxfp4_unswizzle_values_hopper,
)


def _roundtrip(shape, *, mx_axis):
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, size=shape, dtype=np.uint8)
    swz = _mxfp4_swizzle_values_hopper(data, mx_axis=mx_axis)
    unswz = _mxfp4_unswizzle_values_hopper(
        swz, mx_axis=mx_axis, m=shape[-2], k=shape[-1]
    )
    assert np.array_equal(unswz, data)


def test_hopper_value_swizzle_roundtrip_rhs():
    _roundtrip((64, 128), mx_axis=1)


def test_hopper_value_swizzle_roundtrip_lhs():
    _roundtrip((64, 128), mx_axis=0)
