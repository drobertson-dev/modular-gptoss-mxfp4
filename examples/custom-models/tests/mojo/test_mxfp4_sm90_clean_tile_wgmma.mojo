# ===----------------------------------------------------------------------=== #
# SM90 MXFP4 clean-tile WGMMA smoke test.
# Exercises the guarded TMA+WGMMA path: CTA_M=64, CTA_N=64, CTA_K=64
# single expert, dequantized BF16 weights.
# ===----------------------------------------------------------------------=== #

from buffer import Dim
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from ndbuffer_utils import DeviceNDBuffer, HostNDBuffer, zero
from kernels.moe_mxfp4 import GroupedMXFP4Matmul
from kernels.mxfp4 import (
    MXFP4_BLOCK_K,
    MXFP4_PACKED_BYTES_PER_BLOCK,
    MXFP4_SF_DTYPE,
    set_mxfp4_scale,
)
from kernels.mxfp4.primitives import float32_to_e8m0
from layout._ndbuffer_stub import from_ndbuffer_row_major
from testing import assert_almost_equal, assert_true
from testing import TestSuite

from mxfp4_reference import (
    fill_packed_by_nibble,
    ref_grouped_matmul_cpu,
)


fn _nibble_from_k_mod16(expert: Int, out_col: Int, k: Int) -> UInt8:
    return UInt8(k % 16)


fn _skip_if_no_sm90(ctx: DeviceContext) -> Bool:
    from sys import has_nvidia_gpu_accelerator

    if not has_nvidia_gpu_accelerator():
        print("Skipping clean SM90 test (no NVIDIA GPU detected)")
        return True
    if ctx.api() != "cuda":
        print("Skipping clean SM90 test (non-CUDA context)")
        return True
    if ctx.default_device_info.compute < 9.0:
        print("Skipping clean SM90 test (requires sm90+)")
        return True
    return False


fn test_mxfp4_sm90_clean_tile_wgmma() raises:
    var ctx = DeviceContext()
    if _skip_if_no_sm90(ctx):
        return
    alias num_experts = 1
    alias tokens = 64
    alias K = 64  # k_dim spans 4 CTA_K tiles (CTA_K=16)
    alias N = 64  # matches CTA_N

    # Activations: simple pattern
    alias a_shape = DimList(tokens, K)
    var a_host = HostNDBuffer[DType.bfloat16, 2, a_shape](a_shape)
    var a_tensor_host = from_ndbuffer_row_major(a_host.tensor)
    for m in range(tokens):
        for k in range(K):
            var val = Float32(m + k)
            a_tensor_host[m, k] = SIMD[DType.bfloat16, 1](
                val.cast[DType.bfloat16]()
            )

    # Packed weights: k % 16 nibble pattern
    alias packed_shape = DimList(
        num_experts, N, K // MXFP4_BLOCK_K, MXFP4_PACKED_BYTES_PER_BLOCK
    )
    var w_host = HostNDBuffer[DType.uint8, 4, packed_shape](packed_shape)
    var w_tensor_host = from_ndbuffer_row_major(w_host.tensor)
    fill_packed_by_nibble(
        w_tensor_host, K // MXFP4_BLOCK_K, _nibble_from_k_mod16
    )

    # Scales: all ones
    alias row_groups = 1
    alias col_groups = 1
    alias scale_shape = DimList(
        Dim(row_groups), Dim(col_groups), Dim(32), Dim(4), Dim(4)
    )
    var s_host = HostNDBuffer[MXFP4_SF_DTYPE, 5, scale_shape](scale_shape)
    zero(s_host.tensor)
    var s_tensor = from_ndbuffer_row_major(s_host.tensor)

    @parameter
    for kb in range(K // MXFP4_BLOCK_K):
        for row in range(N):
            set_mxfp4_scale(
                s_tensor, row, kb * MXFP4_BLOCK_K, float32_to_e8m0(1.0)
            )

    # Bias: zeros
    alias bias_shape = DimList(num_experts, N)
    var bias_host = HostNDBuffer[DType.bfloat16, 2, bias_shape](bias_shape)
    zero(bias_host.tensor)

    # Routing metadata
    alias offsets_shape = DimList(num_experts + 1)
    alias ids_shape = DimList(num_experts)
    alias stats_shape = DimList(2)
    var offsets_host = HostNDBuffer[DType.uint32, 1, offsets_shape](
        offsets_shape
    )
    var ids_host = HostNDBuffer[DType.int32, 1, ids_shape](ids_shape)
    var stats_host = HostNDBuffer[DType.uint32, 1, stats_shape](stats_shape)
    offsets_host.tensor.data[0] = UInt32(0)
    offsets_host.tensor.data[1] = UInt32(tokens)
    ids_host.tensor.data[0] = Int32(0)
    stats_host.tensor.data[0] = UInt32(tokens)
    stats_host.tensor.data[1] = UInt32(num_experts)

    # Reference
    alias out_shape = DimList(tokens, N)
    var ref_out = HostNDBuffer[DType.float32, 2, out_shape](out_shape)
    zero(ref_out.tensor)
    ref_grouped_matmul_cpu(
        a_tensor_host,
        w_tensor_host,
        s_tensor,
        from_ndbuffer_row_major(bias_host.tensor),
        from_ndbuffer_row_major(offsets_host.tensor),
        from_ndbuffer_row_major(ids_host.tensor),
        from_ndbuffer_row_major(stats_host.tensor),
        from_ndbuffer_row_major(ref_out.tensor),
    )

    # Device copies
    var a_dev = DeviceNDBuffer[DType.bfloat16, 2, a_shape](a_shape, ctx=ctx)
    ctx.enqueue_copy(a_dev.buffer, a_host.tensor.data)
    var w_dev = DeviceNDBuffer[DType.uint8, 4, packed_shape](
        packed_shape, ctx=ctx
    )
    ctx.enqueue_copy(w_dev.buffer, w_host.tensor.data)
    var s_dev = DeviceNDBuffer[MXFP4_SF_DTYPE, 5, scale_shape](
        scale_shape, ctx=ctx
    )
    ctx.enqueue_copy(s_dev.buffer, s_host.tensor.data)
    var bias_dev = DeviceNDBuffer[DType.bfloat16, 2, bias_shape](
        bias_shape, ctx=ctx
    )
    ctx.enqueue_copy(bias_dev.buffer, bias_host.tensor.data)
    var offsets_dev = DeviceNDBuffer[DType.uint32, 1, offsets_shape](
        offsets_shape, ctx=ctx
    )
    ctx.enqueue_copy(offsets_dev.buffer, offsets_host.tensor.data)
    var ids_dev = DeviceNDBuffer[DType.int32, 1, ids_shape](ids_shape, ctx=ctx)
    ctx.enqueue_copy(ids_dev.buffer, ids_host.tensor.data)
    var out_dev = DeviceNDBuffer[DType.bfloat16, 2, out_shape](
        out_shape, ctx=ctx
    )
    ctx.enqueue_memset(out_dev.buffer, 0)

    GroupedMXFP4Matmul.execute(
        from_ndbuffer_row_major(out_dev.tensor),
        from_ndbuffer_row_major(a_dev.tensor),
        from_ndbuffer_row_major(w_dev.tensor),
        from_ndbuffer_row_major(s_dev.tensor),
        from_ndbuffer_row_major(bias_dev.tensor),
        from_ndbuffer_row_major(offsets_dev.tensor),
        from_ndbuffer_row_major(ids_dev.tensor),
        from_ndbuffer_row_major(stats_host.tensor),
        ctx,
    )
    var out_host = HostNDBuffer[DType.bfloat16, 2, out_shape](out_shape)
    ctx.enqueue_copy(out_host.tensor.data, out_dev.buffer)
    ctx.synchronize()

    var out_tensor = from_ndbuffer_row_major(out_host.tensor)
    var ref_tensor = from_ndbuffer_row_major(ref_out.tensor)
    var all_ok = True
    for m in range(tokens):
        for n in range(N):
            var expected = ref_tensor[m, n].cast[DType.float32]()[0]
            var got = out_tensor[m, n].cast[DType.float32]()[0]
            all_ok = all_ok and abs(got - expected) < 1e-2
            assert_almost_equal(got, expected, rtol=1e-2, atol=1e-2)
    assert_true(all_ok)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
