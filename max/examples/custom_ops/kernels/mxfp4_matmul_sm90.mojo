# mxfp4_matmul_sm90.mojo
#
# Correctness-first MXFP4 matmul + fused SwiGLU custom op.
#
# This is primarily used for debugging and CPU-side correctness testing.
# Performance work should focus on the MoE kernels in `moe_mxfp4_ops.mojo`.

import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor

from .mxfp4_decode import decode_mxfp4_byte_to_2xf16, swiglu_pair


comptime F32 = DType.float32
comptime U8 = DType.uint8
comptime MX_SF = DType.float8_e8m0fnu

comptime VALUES_PER_BLOCK = 32
comptime BYTES_PER_BLOCK = 16


@compiler.register("gpt_oss.mxfp4.matmul.sm90")
struct MXFP4MatmulSwiGlu:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor[rank=2],
        a: InputTensor[dtype = output.dtype, rank=2],
        b_packed: InputTensor[dtype=U8, rank=3],
        b_scales: InputTensor[dtype=F32, rank=2],
        bias: InputTensor[dtype = output.dtype, rank=1],
        alpha: Float32,
        limit: Float32,
        ctx: DeviceContextPtr,
    ) raises:
        # Shapes:
        #   a:        [M, K]
        #   b_packed: [K/32, N, 16]
        #   b_scales: [K/32, N]
        #   bias:     [N]
        #   output:   [M, N/2]

        var M = a.dim_size(0)
        var K = a.dim_size(1)
        var k_blocks = b_packed.dim_size(0)
        var N = b_packed.dim_size(1)
        var byte_dim = b_packed.dim_size(2)

        if byte_dim != BYTES_PER_BLOCK:
            raise Error(
                "MXFP4 packed byte dim must be 16, got:",
                byte_dim,
            )
        if K != k_blocks * VALUES_PER_BLOCK:
            raise Error("K must be divisible by 32 (MXFP4 block)")
        if (N % 2) != 0:
            raise Error("N must be even (interleaved gate/up columns)")
        if b_scales.dim_size(0) != k_blocks or b_scales.dim_size(1) != N:
            raise Error("b_scales must match b_packed leading dims")
        if bias.dim_size(0) != N:
            raise Error("bias length must match packed N")
        if output.dim_size(0) != M or output.dim_size(1) != (N // 2):
            raise Error("output shape must be [M, N/2]")

        @parameter
        if target != "cpu":
            raise Error("gpt_oss.mxfp4.matmul.sm90: only CPU path implemented")

        var alpha_f32 = Scalar[F32](alpha)
        var limit_f32 = Scalar[F32](limit)

        for m in range(M):
            for n_out in range(N // 2):
                var col_gate = n_out * 2
                var col_up = col_gate + 1

                var acc_gate: Scalar[F32] = 0.0
                var acc_up: Scalar[F32] = 0.0

                for kb in range(k_blocks):
                    var scale_gate = b_scales[kb, col_gate][0]
                    var scale_up = b_scales[kb, col_up][0]
                    var k_base = kb * VALUES_PER_BLOCK

                    for byte_idx in range(BYTES_PER_BLOCK):
                        var k0 = k_base + 2 * byte_idx

                        var a0 = a[m, k0][0].cast[F32]()
                        var a1 = a[m, k0 + 1][0].cast[F32]()

                        var packed_gate = b_packed[kb, col_gate, byte_idx][0]
                        var packed_up = b_packed[kb, col_up, byte_idx][0]

                        var g2 = decode_mxfp4_byte_to_2xf16(
                            packed_gate, scale_gate
                        )
                        var u2 = decode_mxfp4_byte_to_2xf16(packed_up, scale_up)

                        acc_gate += (
                            a0 * g2[0].cast[F32]() + a1 * g2[1].cast[F32]()
                        )
                        acc_up += (
                            a0 * u2[0].cast[F32]() + a1 * u2[1].cast[F32]()
                        )

                # Bias + fused SwiGLU.
                var gate_bias = bias[col_gate][0].cast[F32]()
                var up_bias = bias[col_up][0].cast[F32]()

                var y = swiglu_pair(
                    acc_gate + gate_bias,
                    acc_up + up_bias,
                    alpha_f32,
                    limit_f32,
                )
                output[m, n_out] = y.cast[output.dtype]()
