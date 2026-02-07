# activation.mojo
#
# MXFP4 SwiGLU activation helpers (CPU + GPU friendly).

from .decode import swiglu_pair

comptime F32 = DType.float32
comptime SWIGLU_ALPHA: Float32 = 1.702
comptime SWIGLU_LIMIT: Float32 = 7.0


@always_inline
fn swiglu_activation(gate: Float32, up: Float32) -> Float32:
    return swiglu_pair(
        gate,
        up,
        Scalar[F32](SWIGLU_ALPHA),
        Scalar[F32](SWIGLU_LIMIT),
    )


@always_inline
fn swiglu_activation_simd[width: Int](
    gate: SIMD[DType.float32, width],
    up: SIMD[DType.float32, width],
) -> SIMD[DType.float32, width]:
    var out = SIMD[DType.float32, width]()

    @parameter
    for i in range(width):
        out[i] = swiglu_activation(gate[i], up[i])
    return out
