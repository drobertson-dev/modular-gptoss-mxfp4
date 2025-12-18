# ===----------------------------------------------------------------------=== #
# Lightweight NDBuffer helpers for MXFP4 Mojo tests.
#
# The upstream `internal_utils` package no longer exports `HostNDBuffer` /
# `DeviceNDBuffer`. Keep these wrappers local to the test suite so the tests
# stay runnable with `mojo run`.
# ===----------------------------------------------------------------------=== #

from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceBuffer, DeviceContext
from memory import LegacyUnsafePointer


fn zero(mut buffer: NDBuffer[mut=True, *_, **_]):
    buffer.zero()


struct HostNDBuffer[
    dtype: DType,
    rank: Int,
    shape: DimList,
](Movable):
    var _ptr: LegacyUnsafePointer[Scalar[Self.dtype]]
    var tensor: NDBuffer[
        mut=True, Self.dtype, Self.rank, MutAnyOrigin, Self.shape
    ]

    fn __init__(out self, dynamic_shape: DimList):
        var numel = Int(dynamic_shape.product())
        self._ptr = LegacyUnsafePointer[Scalar[Self.dtype]].alloc(numel)
        self.tensor = NDBuffer[
            mut=True, Self.dtype, Self.rank, MutAnyOrigin, Self.shape
        ](self._ptr, dynamic_shape)

    fn __moveinit__(out self, deinit existing: Self):
        self._ptr = existing._ptr
        self.tensor = existing.tensor
        existing._ptr = LegacyUnsafePointer[Scalar[Self.dtype]]()
        existing.tensor = NDBuffer[
            mut=True, Self.dtype, Self.rank, MutAnyOrigin, Self.shape
        ]()

    fn __del__(deinit self):
        if self._ptr:
            self._ptr.free()


struct DeviceNDBuffer[
    dtype: DType,
    rank: Int,
    shape: DimList,
]:
    var buffer: DeviceBuffer[Self.dtype]
    var tensor: NDBuffer[
        mut=True, Self.dtype, Self.rank, MutAnyOrigin, Self.shape
    ]

    fn __init__(out self, dynamic_shape: DimList, ctx: DeviceContext) raises:
        var numel = Int(dynamic_shape.product())
        self.buffer = ctx.enqueue_create_buffer[Self.dtype](numel)
        self.tensor = NDBuffer[
            mut=True, Self.dtype, Self.rank, MutAnyOrigin, Self.shape
        ](self.buffer.unsafe_ptr(), dynamic_shape)
