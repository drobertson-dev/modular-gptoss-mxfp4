# ===----------------------------------------------------------------------=== #
# SM90 MXFP4 MoE kernel targets.
#
# GPU tests that pin down:
#   - Single-expert SM90 tile GEMM (clean tile + tail K case)
#   - Grouped MoE routing for down-projection (per-expert signatures, ragged)
#   - Grouped MoE gate_up + fused SwiGLU (even/odd row mapping + clamps)
#
# Each test pairs the GPU path with a CPU float32 reference from
# tests/mojo/mxfp4_reference.mojo. The only way to satisfy these tests is a
# correct SM90 implementation; no CPU fallback is exercised here.
# ===----------------------------------------------------------------------=== #

from buffer import Dim
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from ndbuffer_utils import DeviceNDBuffer, HostNDBuffer, zero
from kernels.moe_mxfp4 import (
    GroupedMXFP4Matmul,
    GroupedMXFP4MatmulSwiGLU,
)
from kernels.mxfp4 import (
    MXFP4_BLOCK_K,
    MXFP4_PACKED_BYTES_PER_BLOCK,
    MXFP4_SF_DTYPE,
    set_mxfp4_scale,
)
from kernels.mxfp4.primitives import float32_to_e8m0
from layout import Layout, LayoutTensor
from layout._ndbuffer_stub import from_ndbuffer_row_major
from math import ceildiv
from sys import has_nvidia_gpu_accelerator
from testing import assert_almost_equal, assert_true

from mxfp4_reference import (
    SWIGLU_LIMIT,
    encode_e2m1_code,
    fill_packed_by_nibble,
    pack_nibbles,
    ref_grouped_matmul_cpu,
    ref_grouped_swiglu_cpu,
    swiglu_reference,
)


# ===----------------------------------------------------------------------=== #
# HELPER FUNCTIONS for fill_packed_by_nibble callbacks
# ===----------------------------------------------------------------------=== #
# Mojo doesn't support inline lambdas with captures, so we define named functions.


fn _nibble_from_k_mod16(expert: Int, out_col: Int, k: Int) -> UInt8:
    """Return k % 16 as nibble (ignores expert and out_col)."""
    return UInt8(k % 16)


fn _skip_if_no_sm90(ctx: DeviceContext) -> Bool:
    if not has_nvidia_gpu_accelerator():
        print("Skipping MXFP4 SM90 test (no NVIDIA GPU detected)")
        return True
    if ctx.api() != "cuda":
        print("Skipping MXFP4 SM90 test (non-CUDA context)")
        return True
    # DeviceInfo.compute reports major.minor (e.g., 9.0 for H100), not 90.
    if ctx.default_device_info.compute < 9.0:
        print("Skipping MXFP4 SM90 test (requires sm90+)")
        return True
    return False


fn _fill_offsets_ids[
    offsets_layout: Layout,
    ids_layout: Layout,
    stats_layout: Layout,
](
    offsets: LayoutTensor[DType.uint32, offsets_layout, MutAnyOrigin],
    ids: LayoutTensor[DType.int32, ids_layout, MutAnyOrigin],
    stats: LayoutTensor[DType.uint32, stats_layout, MutAnyOrigin],
    expert_ids: SIMD[DType.int32, 8],
    counts: SIMD[DType.int32, 8],
    num_experts: Int,
):
    var cursor = UInt32(0)
    offsets[0] = cursor
    for i in range(num_experts):
        cursor += UInt32(counts[i])
        offsets[i + 1] = cursor
        ids[i] = expert_ids[i]

    var max_tokens = Int(0)
    for i in range(num_experts):
        if Int(counts[i]) > max_tokens:
            max_tokens = Int(counts[i])

    stats[0] = UInt32(max_tokens)
    stats[1] = UInt32(num_experts)


fn test_mxfp4_sm90_tile_gemm_clean() raises:
    var ctx = DeviceContext()
    if _skip_if_no_sm90(ctx):
        return

    alias num_experts = 1
    alias tokens = 128
    alias K = 128
    alias N = 128

    # Activations with structured values (unique per (m, k)).
    alias a_shape = DimList(tokens, K)
    var a_host = HostNDBuffer[DType.bfloat16, 2, a_shape](a_shape)
    var a_tensor_host = from_ndbuffer_row_major(a_host.tensor)
    for m in range(tokens):
        for k in range(K):
            var val = Float32(m) + Float32(k) * Float32(0.01)
            a_tensor_host[m, k] = SIMD[DType.bfloat16, 1](
                val.cast[DType.bfloat16]()
            )

    # Packed weights: nibble encodes k % 16 so each element is analytically unique.
    alias packed_shape = DimList(
        num_experts, N, K // MXFP4_BLOCK_K, MXFP4_PACKED_BYTES_PER_BLOCK
    )
    var w_host = HostNDBuffer[DType.uint8, 4, packed_shape](packed_shape)
    var w_tensor_host = from_ndbuffer_row_major(w_host.tensor)
    fill_packed_by_nibble(
        w_tensor_host, K // MXFP4_BLOCK_K, _nibble_from_k_mod16
    )

    # Scales: all ones across both K blocks.
    alias row_groups = ceildiv(N, 128)
    alias col_groups = ceildiv(K, MXFP4_BLOCK_K * 4)
    alias scale_shape = DimList(
        Dim(row_groups), Dim(col_groups), Dim(32), Dim(4), Dim(4)
    )
    var s_host = HostNDBuffer[MXFP4_SF_DTYPE, 5, scale_shape](scale_shape)
    zero(s_host.tensor)
    var s_tensor = from_ndbuffer_row_major(s_host.tensor)
    for row in range(N):

        @parameter
        for kb in range(K // MXFP4_BLOCK_K):
            set_mxfp4_scale(
                s_tensor, row, kb * MXFP4_BLOCK_K, float32_to_e8m0(1.0)
            )

    # Bias: zeros.
    alias bias_shape = DimList(num_experts, N)
    var bias_host = HostNDBuffer[DType.bfloat16, 2, bias_shape](bias_shape)
    zero(bias_host.tensor)
    var bias_tensor_host = from_ndbuffer_row_major(bias_host.tensor)

    # Routing metadata (single expert covering all tokens).
    alias offsets_shape = DimList(num_experts + 1)
    alias ids_shape = DimList(num_experts)
    alias stats_shape = DimList(2)
    var offsets_host = HostNDBuffer[DType.uint32, 1, offsets_shape](
        offsets_shape
    )
    var ids_host = HostNDBuffer[DType.int32, 1, ids_shape](ids_shape)
    var stats_host = HostNDBuffer[DType.uint32, 1, stats_shape](stats_shape)
    var expert_ids = SIMD[DType.int32, 8](0, 0, 0, 0, 0, 0, 0, 0)
    var counts = SIMD[DType.int32, 8](tokens, 0, 0, 0, 0, 0, 0, 0)
    _fill_offsets_ids(
        from_ndbuffer_row_major(offsets_host.tensor),
        from_ndbuffer_row_major(ids_host.tensor),
        from_ndbuffer_row_major(stats_host.tensor),
        expert_ids,
        counts,
        num_experts,
    )
    var offsets_tensor = from_ndbuffer_row_major(offsets_host.tensor)
    var ids_tensor = from_ndbuffer_row_major(ids_host.tensor)
    var stats_tensor = from_ndbuffer_row_major(stats_host.tensor)

    # CPU reference.
    alias out_shape = DimList(tokens, N)
    var ref_out = HostNDBuffer[DType.float32, 2, out_shape](out_shape)
    zero(ref_out.tensor)
    var ref_tensor = from_ndbuffer_row_major(ref_out.tensor)
    ref_grouped_matmul_cpu(
        a_tensor_host,
        w_tensor_host,
        s_tensor,
        bias_tensor_host,
        from_ndbuffer_row_major(offsets_host.tensor),
        from_ndbuffer_row_major(ids_host.tensor),
        from_ndbuffer_row_major(stats_host.tensor),
        ref_tensor,
    )

    # Device copies.
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
    var all_ok = True
    for m in range(tokens):
        for n in range(N):
            var expected_bf16 = ref_tensor[m, n].cast[DType.bfloat16]()
            var expected = expected_bf16.cast[DType.float32]()[0]
            var got = out_tensor[m, n].cast[DType.float32]()[0]
            var err = abs(got - expected)
            all_ok = all_ok and err < 1e-2
            assert_almost_equal(got, expected, rtol=1e-2, atol=1e-2)
    assert_true(all_ok)


fn test_mxfp4_sm90_tile_gemm_tail_k() raises:
    var ctx = DeviceContext()
    if _skip_if_no_sm90(ctx):
        return

    alias num_experts = 1
    alias tokens = 96
    alias K = 96  # three MXFP4 blocks, not a full 128-tile
    alias N = 80  # exercises N tail handling

    alias a_shape = DimList(tokens, K)
    var a_host = HostNDBuffer[DType.bfloat16, 2, a_shape](a_shape)
    var a_tensor_host = from_ndbuffer_row_major(a_host.tensor)
    for m in range(tokens):
        for k in range(K):
            var val = Float32(m) + Float32(k) * Float32(0.05)
            a_tensor_host[m, k] = SIMD[DType.bfloat16, 1](
                val.cast[DType.bfloat16]()
            )

    alias packed_shape = DimList(
        num_experts, N, K // MXFP4_BLOCK_K, MXFP4_PACKED_BYTES_PER_BLOCK
    )
    var w_host = HostNDBuffer[DType.uint8, 4, packed_shape](packed_shape)
    var w_tensor_host = from_ndbuffer_row_major(w_host.tensor)
    fill_packed_by_nibble(
        w_tensor_host, K // MXFP4_BLOCK_K, _nibble_from_k_mod16
    )

    alias row_groups = ceildiv(N, 128)
    alias col_groups = ceildiv(K, MXFP4_BLOCK_K * 4)
    alias scale_shape = DimList(
        Dim(row_groups), Dim(col_groups), Dim(32), Dim(4), Dim(4)
    )
    var s_host = HostNDBuffer[MXFP4_SF_DTYPE, 5, scale_shape](scale_shape)
    zero(s_host.tensor)
    var s_tensor = from_ndbuffer_row_major(s_host.tensor)
    var block_scales = SIMD[DType.float32, 4](1.0, 0.5, 2.0, 1.0)

    @parameter
    for kb in range(K // MXFP4_BLOCK_K):
        var sf = float32_to_e8m0(block_scales[kb])
        for row in range(N):
            set_mxfp4_scale(s_tensor, row, kb * MXFP4_BLOCK_K, sf)

    alias bias_shape = DimList(num_experts, N)
    var bias_host = HostNDBuffer[DType.bfloat16, 2, bias_shape](bias_shape)
    zero(bias_host.tensor)
    var bias_tensor_host = from_ndbuffer_row_major(bias_host.tensor)

    alias offsets_shape = DimList(num_experts + 1)
    alias ids_shape = DimList(num_experts)
    alias stats_shape = DimList(2)
    var offsets_host = HostNDBuffer[DType.uint32, 1, offsets_shape](
        offsets_shape
    )
    var ids_host = HostNDBuffer[DType.int32, 1, ids_shape](ids_shape)
    var stats_host = HostNDBuffer[DType.uint32, 1, stats_shape](stats_shape)
    var expert_ids = SIMD[DType.int32, 8](0, 0, 0, 0, 0, 0, 0, 0)
    var counts = SIMD[DType.int32, 8](tokens, 0, 0, 0, 0, 0, 0, 0)
    _fill_offsets_ids(
        from_ndbuffer_row_major(offsets_host.tensor),
        from_ndbuffer_row_major(ids_host.tensor),
        from_ndbuffer_row_major(stats_host.tensor),
        expert_ids,
        counts,
        num_experts,
    )

    alias out_shape = DimList(tokens, N)
    var ref_out = HostNDBuffer[DType.float32, 2, out_shape](out_shape)
    zero(ref_out.tensor)
    var ref_tensor = from_ndbuffer_row_major(ref_out.tensor)
    ref_grouped_matmul_cpu(
        a_tensor_host,
        w_tensor_host,
        s_tensor,
        bias_tensor_host,
        from_ndbuffer_row_major(offsets_host.tensor),
        from_ndbuffer_row_major(ids_host.tensor),
        from_ndbuffer_row_major(stats_host.tensor),
        ref_tensor,
    )

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
    var all_ok = True
    for m in range(tokens):
        for n in range(N):
            var expected_bf16 = ref_tensor[m, n].cast[DType.bfloat16]()
            var expected = expected_bf16.cast[DType.float32]()[0]
            var got = out_tensor[m, n].cast[DType.float32]()[0]
            all_ok = all_ok and abs(got - expected) < 1e-2
            assert_almost_equal(got, expected, rtol=1e-2, atol=1e-2)
    assert_true(all_ok)


fn test_mxfp4_sm90_grouped_gemm_routing() raises:
    var ctx = DeviceContext()
    if _skip_if_no_sm90(ctx):
        return

    alias num_experts = 4
    alias tokens_e0 = 128
    alias tokens_e2 = 128
    alias tokens_e1 = 0
    alias tokens_e3 = 64
    alias tokens = tokens_e0 + tokens_e2 + tokens_e1 + tokens_e3
    alias K = 128
    alias N = 256

    # Activations: token t filled with (t + 1) for easy row sums.
    alias a_shape = DimList(tokens, K)
    var a_host = HostNDBuffer[DType.bfloat16, 2, a_shape](a_shape)
    var a_tensor_host = from_ndbuffer_row_major(a_host.tensor)
    for t in range(tokens):
        var val = Float32(t + 1)
        for k in range(K):
            a_tensor_host[t, k] = SIMD[DType.bfloat16, 1](
                val.cast[DType.bfloat16]()
            )

    # Per-expert constant signatures via nibble codes.
    alias packed_shape = DimList(
        num_experts, N, K // MXFP4_BLOCK_K, MXFP4_PACKED_BYTES_PER_BLOCK
    )
    var w_host = HostNDBuffer[DType.uint8, 4, packed_shape](packed_shape)
    var w_tensor_host = from_ndbuffer_row_major(w_host.tensor)
    var nibble_codes = SIMD[DType.uint8, num_experts](
        encode_e2m1_code(1.0),
        encode_e2m1_code(2.0),
        encode_e2m1_code(4.0),
        encode_e2m1_code(6.0),
    )
    # Fill packed weights with expert-based nibbles (inline to avoid closure)
    for expert in range(num_experts):
        for out_col in range(N):
            for kb in range(K // MXFP4_BLOCK_K):

                @parameter
                for byte_idx in range(MXFP4_PACKED_BYTES_PER_BLOCK):
                    var nibble = nibble_codes[expert]
                    w_tensor_host[expert, out_col, kb, byte_idx] = pack_nibbles(
                        nibble, nibble
                    )

    # Scales: all ones across all k-blocks (shared across experts).
    alias row_groups = ceildiv(N, 128)
    alias col_groups = ceildiv(K, MXFP4_BLOCK_K * 4)
    alias scale_shape = DimList(
        Dim(row_groups), Dim(col_groups), Dim(32), Dim(4), Dim(4)
    )
    var s_host = HostNDBuffer[MXFP4_SF_DTYPE, 5, scale_shape](scale_shape)
    zero(s_host.tensor)
    var s_tensor = from_ndbuffer_row_major(s_host.tensor)

    @parameter
    for kb in range(K // MXFP4_BLOCK_K):
        var sf = float32_to_e8m0(1.0)
        for row in range(N):
            set_mxfp4_scale(s_tensor, row, kb * MXFP4_BLOCK_K, sf)

    # Bias zeros.
    alias bias_shape = DimList(num_experts, N)
    var bias_host = HostNDBuffer[DType.bfloat16, 2, bias_shape](bias_shape)
    zero(bias_host.tensor)
    var bias_tensor_host = from_ndbuffer_row_major(bias_host.tensor)

    # Routing metadata (expert_ids ordering: [0, 2, 1, 3]).
    alias offsets_shape = DimList(num_experts + 1)
    alias ids_shape = DimList(num_experts)
    alias stats_shape = DimList(2)
    var offsets_host = HostNDBuffer[DType.uint32, 1, offsets_shape](
        offsets_shape
    )
    var ids_host = HostNDBuffer[DType.int32, 1, ids_shape](ids_shape)
    var stats_host = HostNDBuffer[DType.uint32, 1, stats_shape](stats_shape)
    var expert_ids = SIMD[DType.int32, 8](0, 2, 1, 3, 0, 0, 0, 0)
    var counts = SIMD[DType.int32, 8](
        tokens_e0, tokens_e2, tokens_e1, tokens_e3, 0, 0, 0, 0
    )
    _fill_offsets_ids(
        from_ndbuffer_row_major(offsets_host.tensor),
        from_ndbuffer_row_major(ids_host.tensor),
        from_ndbuffer_row_major(stats_host.tensor),
        expert_ids,
        counts,
        num_experts,
    )

    alias out_shape = DimList(tokens, N)
    var ref_out = HostNDBuffer[DType.float32, 2, out_shape](out_shape)
    zero(ref_out.tensor)
    var ref_tensor = from_ndbuffer_row_major(ref_out.tensor)
    ref_grouped_matmul_cpu(
        a_tensor_host,
        w_tensor_host,
        s_tensor,
        bias_tensor_host,
        from_ndbuffer_row_major(offsets_host.tensor),
        from_ndbuffer_row_major(ids_host.tensor),
        from_ndbuffer_row_major(stats_host.tensor),
        ref_tensor,
    )

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
    var all_ok = True
    for t in range(tokens):
        for n in range(N):
            var expected_bf16 = ref_tensor[t, n].cast[DType.bfloat16]()
            var expected = expected_bf16.cast[DType.float32]()[0]
            var got = out_tensor[t, n].cast[DType.float32]()[0]
            all_ok = all_ok and abs(got - expected) < 1e-2
            assert_almost_equal(got, expected, rtol=1e-2, atol=1e-2)
    assert_true(all_ok)

    # Unused expert (expert 1) should be ignored even if weights/bias change.
    var expert1_code = encode_e2m1_code(6.0)
    # Re-fill packed weights with modified expert 1 (inline to avoid closure)
    for expert in range(num_experts):
        for out_col in range(N):
            for kb in range(K // MXFP4_BLOCK_K):

                @parameter
                for byte_idx in range(MXFP4_PACKED_BYTES_PER_BLOCK):
                    var nibble = (
                        expert1_code if expert == 1 else nibble_codes[expert]
                    )
                    w_tensor_host[expert, out_col, kb, byte_idx] = pack_nibbles(
                        nibble, nibble
                    )
    for n in range(N):
        bias_tensor_host[1, n] = SIMD[DType.bfloat16, 1](
            Float32(42.0).cast[DType.bfloat16]()
        )

    ctx.enqueue_copy(w_dev.buffer, w_host.tensor.data)
    ctx.enqueue_copy(bias_dev.buffer, bias_host.tensor.data)
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

    var out_mut = HostNDBuffer[DType.bfloat16, 2, out_shape](out_shape)
    ctx.enqueue_copy(out_mut.tensor.data, out_dev.buffer)
    ctx.synchronize()

    var out_mut_tensor = from_ndbuffer_row_major(out_mut.tensor)
    for t in range(tokens):
        for n in range(N):
            var base = out_tensor[t, n].cast[DType.float32]()
            var mutated = out_mut_tensor[t, n].cast[DType.float32]()
            assert_almost_equal(mutated, base, rtol=1e-3, atol=1e-3)


fn test_mxfp4_sm90_grouped_swiglu_gate_up() raises:
    var ctx = DeviceContext()
    if _skip_if_no_sm90(ctx):
        return

    alias num_experts = 4
    alias tokens_e0 = 96
    alias tokens_e2 = 96
    alias tokens_e1 = 0
    alias tokens_e3 = 48
    alias tokens = tokens_e0 + tokens_e2 + tokens_e1 + tokens_e3
    alias K = 96  # tail K to exercise block indexing
    alias moe_dim = 120  # tail moe_dim → two_moe_dim = 240 (>1 row group)
    alias two_moe_dim = moe_dim * 2

    alias a_shape = DimList(tokens, K)
    var a_host = HostNDBuffer[DType.bfloat16, 2, a_shape](a_shape)
    var a_tensor_host = from_ndbuffer_row_major(a_host.tensor)
    for t in range(tokens):
        var val = Float32(t + 1)
        for k in range(K):
            a_tensor_host[t, k] = SIMD[DType.bfloat16, 1](
                val.cast[DType.bfloat16]()
            )

    alias packed_shape = DimList(
        num_experts,
        two_moe_dim,
        K // MXFP4_BLOCK_K,
        MXFP4_PACKED_BYTES_PER_BLOCK,
    )
    var gate_up_host = HostNDBuffer[DType.uint8, 4, packed_shape](packed_shape)
    var gate_up_tensor_host = from_ndbuffer_row_major(gate_up_host.tensor)
    var gate_code = encode_e2m1_code(1.0)
    var up_code = encode_e2m1_code(0.0)
    # Fill gate_up weights with row-based nibbles (inline to avoid closure)
    for expert in range(num_experts):
        for row in range(two_moe_dim):
            var nibble = gate_code if row % 2 == 0 else up_code
            for kb in range(K // MXFP4_BLOCK_K):

                @parameter
                for byte_idx in range(MXFP4_PACKED_BYTES_PER_BLOCK):
                    gate_up_tensor_host[
                        expert, row, kb, byte_idx
                    ] = pack_nibbles(nibble, nibble)

    alias row_groups = ceildiv(two_moe_dim, 128)
    alias col_groups = ceildiv(K, MXFP4_BLOCK_K * 4)
    alias scale_shape = DimList(
        Dim(row_groups), Dim(col_groups), Dim(32), Dim(4), Dim(4)
    )
    var s_host = HostNDBuffer[MXFP4_SF_DTYPE, 5, scale_shape](scale_shape)
    zero(s_host.tensor)
    var s_tensor = from_ndbuffer_row_major(s_host.tensor)

    @parameter
    for kb in range(K // MXFP4_BLOCK_K):
        var sf = float32_to_e8m0(1.0)
        for row in range(two_moe_dim):
            set_mxfp4_scale(s_tensor, row, kb * MXFP4_BLOCK_K, sf)

    alias bias_shape = DimList(num_experts, two_moe_dim)
    var bias_host = HostNDBuffer[DType.bfloat16, 2, bias_shape](bias_shape)
    var bias_tensor_host = from_ndbuffer_row_major(bias_host.tensor)
    for e in range(num_experts):
        for j in range(two_moe_dim):
            var idx = j // 2
            if j % 2 == 0:
                # Gate bias nudges some rows over the clamp boundary.
                var gate_bias = Float32(idx) * Float32(0.25) + Float32(
                    e
                ) * Float32(0.05)
                if idx % 8 == 0:
                    gate_bias = SWIGLU_LIMIT + Float32(1.0)
                bias_tensor_host[e, j] = SIMD[DType.bfloat16, 1](
                    gate_bias.cast[DType.bfloat16]()
                )
            else:
                # Up bias exercises ±limit clamping.
                var up_bias = -Float32(idx) * Float32(0.1)
                if idx % 5 == 0:
                    up_bias = SWIGLU_LIMIT + Float32(0.5)
                if idx % 7 == 0:
                    up_bias = -SWIGLU_LIMIT - Float32(0.5)
                bias_tensor_host[e, j] = SIMD[DType.bfloat16, 1](
                    up_bias.cast[DType.bfloat16]()
                )

    alias offsets_shape = DimList(num_experts + 1)
    alias ids_shape = DimList(num_experts)
    alias stats_shape = DimList(2)
    var offsets_host = HostNDBuffer[DType.uint32, 1, offsets_shape](
        offsets_shape
    )
    var ids_host = HostNDBuffer[DType.int32, 1, ids_shape](ids_shape)
    var stats_host = HostNDBuffer[DType.uint32, 1, stats_shape](stats_shape)
    var expert_ids = SIMD[DType.int32, 8](0, 2, 1, 3, 0, 0, 0, 0)
    var counts = SIMD[DType.int32, 8](
        tokens_e0, tokens_e2, tokens_e1, tokens_e3, 0, 0, 0, 0
    )
    _fill_offsets_ids(
        from_ndbuffer_row_major(offsets_host.tensor),
        from_ndbuffer_row_major(ids_host.tensor),
        from_ndbuffer_row_major(stats_host.tensor),
        expert_ids,
        counts,
        num_experts,
    )

    alias out_shape = DimList(tokens, moe_dim)
    var ref_out = HostNDBuffer[DType.float32, 2, out_shape](out_shape)
    zero(ref_out.tensor)
    var ref_tensor = from_ndbuffer_row_major(ref_out.tensor)
    var offsets_tensor = from_ndbuffer_row_major(offsets_host.tensor)
    var ids_tensor = from_ndbuffer_row_major(ids_host.tensor)
    var stats_tensor = from_ndbuffer_row_major(stats_host.tensor)
    ref_grouped_swiglu_cpu(
        a_tensor_host,
        gate_up_tensor_host,
        s_tensor,
        bias_tensor_host,
        offsets_tensor,
        ids_tensor,
        stats_tensor,
        ref_tensor,
    )

    var a_dev = DeviceNDBuffer[DType.bfloat16, 2, a_shape](a_shape, ctx=ctx)
    ctx.enqueue_copy(a_dev.buffer, a_host.tensor.data)
    var gate_up_dev = DeviceNDBuffer[DType.uint8, 4, packed_shape](
        packed_shape, ctx=ctx
    )
    ctx.enqueue_copy(gate_up_dev.buffer, gate_up_host.tensor.data)
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

    GroupedMXFP4MatmulSwiGLU.execute(
        from_ndbuffer_row_major(out_dev.tensor),
        from_ndbuffer_row_major(a_dev.tensor),
        from_ndbuffer_row_major(gate_up_dev.tensor),
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
    var all_ok = True
    for t in range(tokens):
        for j in range(moe_dim):
            var expected_bf16 = ref_tensor[t, j].cast[DType.bfloat16]()
            var expected = expected_bf16.cast[DType.float32]()[0]
            var got = out_tensor[t, j].cast[DType.float32]()[0]
            all_ok = all_ok and abs(got - expected) < 1e-2
            assert_almost_equal(got, expected, rtol=1e-2, atol=1e-2)
    assert_true(all_ok)

    # Spot-check the even/odd (gate/up) mapping with analytic signatures.
    var gate_value = Float32(1.0)
    for expert_idx in range(num_experts):
        var start = Int(offsets_tensor[expert_idx])
        var end = Int(offsets_tensor[expert_idx + 1])
        if end <= start:
            continue
        var token_val = Float32(start + 1)
        var row_sum = token_val * Float32(K)
        for j in range(min(moe_dim, 4)):
            var gate_bias_idx = j * 2
            var up_bias_idx = j * 2 + 1
            var gate_bias: Float32 = bias_tensor_host[
                expert_idx, gate_bias_idx
            ].cast[DType.float32]()[0]
            var up_bias: Float32 = bias_tensor_host[
                expert_idx, up_bias_idx
            ].cast[DType.float32]()[0]
            var gate_acc: Float32 = row_sum * gate_value + gate_bias
            var up_acc: Float32 = up_bias
            var expected = swiglu_reference(gate_acc, up_acc)
            var got: Float32 = out_tensor[start, j].cast[DType.float32]()[0]
            assert_almost_equal(got, expected, rtol=1e-2, atol=1e-2)


fn test_mxfp4_sm90_k_loop_multi_tile() raises:
    """Test K-loop with multiple K-tiles (K > CTA_K).

    This test uses K=256 which requires 2 K-tiles with CTA_K=128.
    Validates that the K-loop accumulation correctly sums partial products
    across multiple tiles. Uses varying scales per K-block to ensure
    each block contributes uniquely to the final result.
    """
    var ctx = DeviceContext()
    if _skip_if_no_sm90(ctx):
        return

    alias num_experts = 1
    alias tokens = 64
    alias K = 256  # Two K-tiles with CTA_K=128
    alias N = 64

    # Activations: distinct pattern per row and k
    alias a_shape = DimList(tokens, K)
    var a_host = HostNDBuffer[DType.bfloat16, 2, a_shape](a_shape)
    var a_tensor_host = from_ndbuffer_row_major(a_host.tensor)
    for m in range(tokens):
        for k in range(K):
            # Use row + k offset to create unique patterns
            var val = Float32(m + 1) + Float32(k) * Float32(0.001)
            a_tensor_host[m, k] = SIMD[DType.bfloat16, 1](
                val.cast[DType.bfloat16]()
            )

    # Packed weights with k % 16 pattern
    alias packed_shape = DimList(
        num_experts, N, K // MXFP4_BLOCK_K, MXFP4_PACKED_BYTES_PER_BLOCK
    )
    var w_host = HostNDBuffer[DType.uint8, 4, packed_shape](packed_shape)
    var w_tensor_host = from_ndbuffer_row_major(w_host.tensor)
    fill_packed_by_nibble(
        w_tensor_host, K // MXFP4_BLOCK_K, _nibble_from_k_mod16
    )

    # Scales: different per K-block to verify accumulation
    # K=256 → 8 blocks of 32 elements each
    alias row_groups = ceildiv(N, 128)
    alias col_groups = ceildiv(K, MXFP4_BLOCK_K * 4)
    alias scale_shape = DimList(
        Dim(row_groups), Dim(col_groups), Dim(32), Dim(4), Dim(4)
    )
    var s_host = HostNDBuffer[MXFP4_SF_DTYPE, 5, scale_shape](scale_shape)
    zero(s_host.tensor)
    var s_tensor = from_ndbuffer_row_major(s_host.tensor)

    # Varying scales per K-block: [1.0, 0.5, 2.0, 0.25, 1.0, 0.5, 2.0, 0.25]
    var block_scales = SIMD[DType.float32, 8](
        1.0, 0.5, 2.0, 0.25, 1.0, 0.5, 2.0, 0.25
    )
    for row in range(N):

        @parameter
        for kb in range(K // MXFP4_BLOCK_K):
            set_mxfp4_scale(
                s_tensor,
                row,
                kb * MXFP4_BLOCK_K,
                float32_to_e8m0(block_scales[kb]),
            )

    # Bias: small values to ensure they're added correctly
    alias bias_shape = DimList(num_experts, N)
    var bias_host = HostNDBuffer[DType.bfloat16, 2, bias_shape](bias_shape)
    var bias_tensor_host = from_ndbuffer_row_major(bias_host.tensor)
    for n in range(N):
        var b = Float32(n) * Float32(0.1)
        bias_tensor_host[0, n] = SIMD[DType.bfloat16, 1](
            b.cast[DType.bfloat16]()
        )

    # Routing: single expert
    alias offsets_shape = DimList(num_experts + 1)
    alias ids_shape = DimList(num_experts)
    alias stats_shape = DimList(2)
    var offsets_host = HostNDBuffer[DType.uint32, 1, offsets_shape](
        offsets_shape
    )
    var ids_host = HostNDBuffer[DType.int32, 1, ids_shape](ids_shape)
    var stats_host = HostNDBuffer[DType.uint32, 1, stats_shape](stats_shape)
    var expert_ids = SIMD[DType.int32, 8](0, 0, 0, 0, 0, 0, 0, 0)
    var counts = SIMD[DType.int32, 8](tokens, 0, 0, 0, 0, 0, 0, 0)
    _fill_offsets_ids(
        from_ndbuffer_row_major(offsets_host.tensor),
        from_ndbuffer_row_major(ids_host.tensor),
        from_ndbuffer_row_major(stats_host.tensor),
        expert_ids,
        counts,
        num_experts,
    )

    # CPU reference
    alias out_shape = DimList(tokens, N)
    var ref_out = HostNDBuffer[DType.float32, 2, out_shape](out_shape)
    zero(ref_out.tensor)
    var ref_tensor = from_ndbuffer_row_major(ref_out.tensor)
    ref_grouped_matmul_cpu(
        a_tensor_host,
        w_tensor_host,
        s_tensor,
        bias_tensor_host,
        from_ndbuffer_row_major(offsets_host.tensor),
        from_ndbuffer_row_major(ids_host.tensor),
        from_ndbuffer_row_major(stats_host.tensor),
        ref_tensor,
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
    var all_ok = True
    var max_error = Float32(0.0)
    for m in range(tokens):
        for n in range(N):
            var expected_bf16 = ref_tensor[m, n].cast[DType.bfloat16]()
            var expected = expected_bf16.cast[DType.float32]()[0]
            var got = out_tensor[m, n].cast[DType.float32]()[0]
            var error = abs(got - expected)
            if error > max_error:
                max_error = error
            all_ok = all_ok and error < 1e-1  # Slightly relaxed for multi-tile
            assert_almost_equal(got, expected, rtol=1e-1, atol=1e-1)
    assert_true(all_ok)


fn main() raises:
    """Entry point for MXFP4 SM90 GPU tests."""
    print("Running MXFP4 SM90 GPU tests...")

    print("  test_mxfp4_sm90_tile_gemm_clean...")
    test_mxfp4_sm90_tile_gemm_clean()
    print("  PASSED")

    print("  test_mxfp4_sm90_tile_gemm_tail_k...")
    test_mxfp4_sm90_tile_gemm_tail_k()
    print("  PASSED")

    print("  test_mxfp4_sm90_grouped_gemm_routing...")
    test_mxfp4_sm90_grouped_gemm_routing()
    print("  PASSED")

    print("  test_mxfp4_sm90_grouped_swiglu_gate_up...")
    test_mxfp4_sm90_grouped_swiglu_gate_up()
    print("  PASSED")

    print("  test_mxfp4_sm90_k_loop_multi_tile...")
    test_mxfp4_sm90_k_loop_multi_tile()
    print("  PASSED")

    print("All MXFP4 SM90 GPU tests passed!")
