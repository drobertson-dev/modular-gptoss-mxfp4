# ===----------------------------------------------------------------------=== #
# MXFP4 kernel sanity tests using the TestSuite runner (no Bazel required).
# These are intentionally small and self contained for quick iteration.
# ===----------------------------------------------------------------------=== #

from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from gpu.host.info import H100
from internal_utils import DeviceNDBuffer
from nn.moe_mxfp4 import (
    _FP4_VALUES,
    _mxfp4_grouped_matmul_cpu,
    _scale_multiplier,
    mxfp4_grouped_matmul,
)
from sys.info import _accelerator_arch
from testing import TestSuite, assert_almost_equal


alias _TOKENS = 64
alias _IN_FEATURES = 128
alias _OUT_FEATURES = 64
alias _NUM_EXPERTS = 1


struct Inputs:
    var hidden_storage: InlineArray[Float32, _TOKENS * _IN_FEATURES]
    var packed_storage: InlineArray[
        UInt8, _NUM_EXPERTS * _OUT_FEATURES * (_IN_FEATURES // 2)
    ]
    var scales_storage: InlineArray[
        UInt8, _NUM_EXPERTS * _OUT_FEATURES * (_IN_FEATURES // 32)
    ]
    var bias_storage: InlineArray[Float32, _NUM_EXPERTS * _OUT_FEATURES]
    var offsets_storage: InlineArray[UInt32, _NUM_EXPERTS + 1]
    var ids_storage: InlineArray[Int32, _NUM_EXPERTS]

    var hidden: NDBuffer[
        DType.float32, 2, MutAnyOrigin, DimList(_TOKENS, _IN_FEATURES)
    ]
    var packed: NDBuffer[
        DType.uint8,
        3,
        MutAnyOrigin,
        DimList(_NUM_EXPERTS, _OUT_FEATURES, _IN_FEATURES // 2),
    ]
    var scales: NDBuffer[
        DType.uint8,
        3,
        MutAnyOrigin,
        DimList(_NUM_EXPERTS, _OUT_FEATURES, _IN_FEATURES // 32),
    ]
    var bias: NDBuffer[
        DType.float32, 2, MutAnyOrigin, DimList(_NUM_EXPERTS, _OUT_FEATURES)
    ]
    var offsets: NDBuffer[
        DType.uint32, 1, MutAnyOrigin, DimList(_NUM_EXPERTS + 1)
    ]
    var ids: NDBuffer[DType.int32, 1, MutAnyOrigin, DimList(_NUM_EXPERTS)]

    fn __init__(out self):
        self.hidden_storage = InlineArray[Float32, _TOKENS * _IN_FEATURES](
            uninitialized=True
        )
        for i in range(_TOKENS * _IN_FEATURES):
            self.hidden_storage[i] = Float32(i % 17)
        self.hidden = NDBuffer[
            DType.float32,
            2,
            _,
            DimList(_TOKENS, _IN_FEATURES),
        ](self.hidden_storage.unsafe_ptr())

        self.packed_storage = InlineArray[
            UInt8,
            _NUM_EXPERTS * _OUT_FEATURES * (_IN_FEATURES // 2),
        ](uninitialized=True)
        for i in range(self.packed_storage.size):
            var lo = UInt8(i % 16)
            var hi = UInt8((i + 1) % 16)
            self.packed_storage[i] = lo | (hi << 4)
        self.packed = NDBuffer[
            DType.uint8,
            3,
            _,
            DimList(_NUM_EXPERTS, _OUT_FEATURES, _IN_FEATURES // 2),
        ](self.packed_storage.unsafe_ptr())

        self.scales_storage = InlineArray[
            UInt8,
            _NUM_EXPERTS * _OUT_FEATURES * (_IN_FEATURES // 32),
        ](uninitialized=True)
        for i in range(self.scales_storage.size):
            self.scales_storage[i] = UInt8(127 + (i % 2))
        self.scales = NDBuffer[
            DType.uint8,
            3,
            _,
            DimList(_NUM_EXPERTS, _OUT_FEATURES, _IN_FEATURES // 32),
        ](self.scales_storage.unsafe_ptr())

        self.bias_storage = InlineArray[Float32, _NUM_EXPERTS * _OUT_FEATURES](
            uninitialized=True
        )
        for i in range(self.bias_storage.size):
            self.bias_storage[i] = Float32(i)
        self.bias = NDBuffer[
            DType.float32,
            2,
            _,
            DimList(_NUM_EXPERTS, _OUT_FEATURES),
        ](self.bias_storage.unsafe_ptr())

        self.offsets_storage = InlineArray[UInt32, _NUM_EXPERTS + 1](
            uninitialized=True
        )
        self.offsets_storage[0] = 0
        self.offsets_storage[1] = _TOKENS
        self.offsets = NDBuffer[
            DType.uint32,
            1,
            _,
            DimList(_NUM_EXPERTS + 1),
        ](self.offsets_storage.unsafe_ptr())

        self.ids_storage = InlineArray[Int32, _NUM_EXPERTS](uninitialized=True)
        self.ids_storage[0] = 0
        self.ids = NDBuffer[
            DType.int32,
            1,
            _,
            DimList(_NUM_EXPERTS),
        ](self.ids_storage.unsafe_ptr())


fn _make_inputs() -> Inputs:
    return Inputs()


struct DeviceInputs:
    var hidden: DeviceNDBuffer[
        DType.float32,
        2,
        DimList(_TOKENS, _IN_FEATURES),
    ]
    var packed: DeviceNDBuffer[
        DType.uint8,
        3,
        DimList(_NUM_EXPERTS, _OUT_FEATURES, _IN_FEATURES // 2),
    ]
    var scales: DeviceNDBuffer[
        DType.uint8,
        3,
        DimList(_NUM_EXPERTS, _OUT_FEATURES, _IN_FEATURES // 32),
    ]
    var bias: DeviceNDBuffer[
        DType.float32,
        2,
        DimList(_NUM_EXPERTS, _OUT_FEATURES),
    ]
    var offsets: DeviceNDBuffer[
        DType.uint32,
        1,
        DimList(_NUM_EXPERTS + 1),
    ]
    var ids: DeviceNDBuffer[
        DType.int32,
        1,
        DimList(_NUM_EXPERTS),
    ]

    fn __init__(out self, host_inputs: Inputs, ctx: DeviceContext) raises:
        self.hidden = DeviceNDBuffer[
            DType.float32,
            2,
            DimList(_TOKENS, _IN_FEATURES),
        ](ctx=ctx)
        ctx.enqueue_copy(self.hidden.buffer, host_inputs.hidden.data)

        self.packed = DeviceNDBuffer[
            DType.uint8,
            3,
            DimList(_NUM_EXPERTS, _OUT_FEATURES, _IN_FEATURES // 2),
        ](ctx=ctx)
        ctx.enqueue_copy(self.packed.buffer, host_inputs.packed.data)

        self.scales = DeviceNDBuffer[
            DType.uint8,
            3,
            DimList(_NUM_EXPERTS, _OUT_FEATURES, _IN_FEATURES // 32),
        ](ctx=ctx)
        ctx.enqueue_copy(self.scales.buffer, host_inputs.scales.data)

        self.bias = DeviceNDBuffer[
            DType.float32,
            2,
            DimList(_NUM_EXPERTS, _OUT_FEATURES),
        ](ctx=ctx)
        ctx.enqueue_copy(self.bias.buffer, host_inputs.bias.data)

        self.offsets = DeviceNDBuffer[
            DType.uint32,
            1,
            DimList(_NUM_EXPERTS + 1),
        ](ctx=ctx)
        ctx.enqueue_copy(self.offsets.buffer, host_inputs.offsets.data)

        self.ids = DeviceNDBuffer[
            DType.int32,
            1,
            DimList(_NUM_EXPERTS),
        ](ctx=ctx)
        ctx.enqueue_copy(self.ids.buffer, host_inputs.ids.data)


fn _decode_weights(
    packed: NDBuffer[
        DType.uint8,
        3,
        _,
        DimList(_NUM_EXPERTS, _OUT_FEATURES, _IN_FEATURES // 2),
    ],
    scales: NDBuffer[
        DType.uint8,
        3,
        _,
        DimList(_NUM_EXPERTS, _OUT_FEATURES, _IN_FEATURES // 32),
    ],
) raises -> InlineArray[Float32, _NUM_EXPERTS * _OUT_FEATURES * _IN_FEATURES]:
    var dense = InlineArray[
        Float32, _NUM_EXPERTS * _OUT_FEATURES * _IN_FEATURES
    ](uninitialized=True)
    for e in range(_NUM_EXPERTS):
        for n in range(_OUT_FEATURES):
            for k in range(_IN_FEATURES):
                var packed_byte = packed[e, n, k >> 1]
                var scale_byte = scales[e, n, k >> 5]
                var nibble = (
                    packed_byte & UInt8(0x0F) if (k & 1)
                    == 0 else packed_byte >> 4
                )
                var scale_mul = _scale_multiplier(scale_byte)
                var decoded = _FP4_VALUES[Int(nibble)] * scale_mul
                dense[(e * _OUT_FEATURES + n) * _IN_FEATURES + k] = decoded
    return dense


fn _reference_matmul(
    hidden: NDBuffer[DType.float32, 2, _, DimList(_TOKENS, _IN_FEATURES)],
    dense_weights: InlineArray[
        Float32, _NUM_EXPERTS * _OUT_FEATURES * _IN_FEATURES
    ],
    bias: NDBuffer[DType.float32, 2, _, DimList(_NUM_EXPERTS, _OUT_FEATURES)],
) raises -> InlineArray[Float32, _TOKENS * _OUT_FEATURES]:
    var out = InlineArray[Float32, _TOKENS * _OUT_FEATURES](uninitialized=True)
    for m in range(_TOKENS):
        for n in range(_OUT_FEATURES):
            var acc = Float32(0.0)
            for k in range(_IN_FEATURES):
                var a_val = hidden[m, k]
                var w_val = dense_weights[(n) * _IN_FEATURES + k]
                acc += a_val * w_val
            acc += bias[0, n]
            out[m * _OUT_FEATURES + n] = acc
    return out


fn test_cpu_reference_match() raises:
    var inputs = _make_inputs()
    var dense = _decode_weights(inputs.packed, inputs.scales)
    var expected = _reference_matmul(inputs.hidden, dense, inputs.bias)

    var out_backing = InlineArray[Float32, _TOKENS * _OUT_FEATURES](
        uninitialized=True
    )
    var out = NDBuffer[DType.float32, 2, _, DimList(_TOKENS, _OUT_FEATURES)](
        out_backing.unsafe_ptr()
    )
    _mxfp4_grouped_matmul_cpu[DType.float32, DType.float32](
        out,
        inputs.hidden,
        inputs.packed,
        inputs.scales,
        inputs.bias,
        inputs.offsets,
        inputs.ids,
        _NUM_EXPERTS,
    )

    for i in range(out_backing.size):
        assert_almost_equal(out_backing[i], expected[i], atol=1e-3, rtol=1e-3)


fn test_gpu_matches_cpu_if_available() raises:
    if _accelerator_arch() == "":
        return  # skip cleanly on CPU-only systems

    var ctx = DeviceContext()
    print("Accelerator:", _accelerator_arch())
    print("Device:", materialize[ctx.default_device_info]())
    var inputs = _make_inputs()
    var device_inputs = DeviceInputs(inputs, ctx=ctx)
    var dense = _decode_weights(inputs.packed, inputs.scales)
    var expected = _reference_matmul(inputs.hidden, dense, inputs.bias)

    var out_backing = InlineArray[Float32, _TOKENS * _OUT_FEATURES](
        uninitialized=True
    )
    var out = NDBuffer[DType.float32, 2, _, DimList(_TOKENS, _OUT_FEATURES)](
        out_backing.unsafe_ptr()
    )
    var out_device = DeviceNDBuffer[
        DType.float32,
        2,
        DimList(_TOKENS, _OUT_FEATURES),
    ](ctx=ctx)
    mxfp4_grouped_matmul[DType.float32, DType.float32, "gpu"](
        out_device.tensor,
        device_inputs.hidden.tensor,
        device_inputs.packed.tensor,
        device_inputs.scales.tensor,
        device_inputs.bias.tensor,
        device_inputs.offsets.tensor,
        device_inputs.ids.tensor,
        _TOKENS,
        _NUM_EXPERTS,
        ctx,
    )
    ctx.enqueue_copy(out.data, out_device.buffer)
    ctx.synchronize()

    for i in range(out_backing.size):
        assert_almost_equal(out_backing[i], expected[i], atol=1e-2, rtol=1e-2)


fn test_sm90_path_matches_cpu_if_h100() raises:
    alias is_h100 = materialize[DeviceContext.default_device_info is H100]()
    if not is_h100:
        return

    var ctx = DeviceContext()
    print("Accelerator:", _accelerator_arch())
    print("Device:", materialize[ctx.default_device_info]())
    var inputs = _make_inputs()
    var device_inputs = DeviceInputs(inputs, ctx=ctx)
    var dense = _decode_weights(inputs.packed, inputs.scales)
    var expected = _reference_matmul(inputs.hidden, dense, inputs.bias)

    var out_backing = InlineArray[Float32, _TOKENS * _OUT_FEATURES](
        uninitialized=True
    )
    var out = NDBuffer[DType.float32, 2, _, DimList(_TOKENS, _OUT_FEATURES)](
        out_backing.unsafe_ptr()
    )
    # Invoke the gated GPU entrypoint (will select sm90 on H100).
    var out_device = DeviceNDBuffer[
        DType.float32,
        2,
        DimList(_TOKENS, _OUT_FEATURES),
    ](ctx=ctx)
    mxfp4_grouped_matmul[DType.float32, DType.float32, "gpu"](
        out_device.tensor,
        device_inputs.hidden.tensor,
        device_inputs.packed.tensor,
        device_inputs.scales.tensor,
        device_inputs.bias.tensor,
        device_inputs.offsets.tensor,
        device_inputs.ids.tensor,
        _TOKENS,
        _NUM_EXPERTS,
        ctx,
    )
    ctx.enqueue_copy(out.data, out_device.buffer)
    ctx.synchronize()

    for i in range(out_backing.size):
        assert_almost_equal(out_backing[i], expected[i], atol=1e-2, rtol=1e-2)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
