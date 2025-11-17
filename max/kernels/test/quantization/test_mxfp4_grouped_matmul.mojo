# ===----------------------------------------------------------------------=== #
# MXFP4 grouped matmul tests (CPU + ragged helper)
# ===----------------------------------------------------------------------=== #
# These tests exercise the MXFP4 grouped matmul CPU path and the ragged wrapper
# that materializes assignments from expert_start_indices/expert_ids. They keep
# shapes small to stay deterministic and compare against a float reference.

from math import ceildiv

from buffer import NDBuffer
from buffer.dimlist import DimList
from layout import Layout, LayoutTensor, RuntimeLayout
from memory import InlineArray, LegacyUnsafePointer as UnsafePointer
from runtime.asyncrt import DeviceContext
from testing import assert_almost_equal, assert_true

from quantization.mxfp4_kernels import (
    MXFP4QuantizeEXQ,
    _mxfp4_grouped_matmul_cpu,
    _mxfp4_grouped_matmul_ragged_gpu,
)
from utils.index import Index, IndexList


fn _make_layout_tensor[
    dtype: DType,
    layout: Layout,
    origin: Inferred = MutAnyOrigin,
](ptr: UnsafePointer[Scalar[dtype], **origin]) -> LayoutTensor[dtype, layout, origin]:
    alias rt = RuntimeLayout[layout].row_major(IndexList[layout.rank()](layout.shape()))
    return LayoutTensor[dtype, layout, origin](ptr, rt)


def _quantize_weights(
    w: LayoutTensor[DType.float32, Layout.row_major(2)()],
    q: LayoutTensor[DType.uint8, Layout.row_major(2)()],
    e: LayoutTensor[DType.uint8, Layout.row_major(2)()],
):
    alias kernel = MXFP4QuantizeEXQ
    kernel.execute[
        in_dtype=DType.float32,
        rank=2,
        target="cpu",
    ](q, e, w, DeviceContext())


fn _reference_matmul(
    hidden: LayoutTensor[DType.float32, Layout.row_major(2)()],
    weight: LayoutTensor[DType.float32, Layout.row_major(3)()],
    assignments: LayoutTensor[DType.int32, Layout.row_major(2)()],
    mut out: LayoutTensor[DType.float32, Layout.row_major(2)()],
):
    alias tokens = hidden.shape[0]()
    alias out_dim = out.shape[1]()
    alias k = hidden.shape[1]()
    for t in range(tokens):
        var e = assignments[t].cast[DType.int32]()[0]
        if e < 0:
            continue
        for n in range(out_dim):
            var acc: Float32 = 0.0
            for kk in range(k):
                acc += hidden[t, kk].cast[DType.float32]()[0] * weight[e, n, kk].cast[DType.float32]()[0]
            out[t, n] = acc


fn _build_hidden(tokens: Int, k: Int) -> LayoutTensor[DType.float32, Layout.row_major(2)()]:
    var buf = InlineArray[Float32, tokens * k](uninitialized=True)
    for i in range(tokens * k):
        buf[i] = Float32(i % 13) - 6.5
    return LayoutTensor[DType.float32, Layout.row_major(tokens, k)()(MutAnyOrigin)](
        buf.unsafe_ptr()
    )


fn _build_assignments(assigns: List[Int32]) -> LayoutTensor[DType.int32, Layout.row_major(2)()]:
    var buf = InlineArray[Int32, len(assigns)](uninitialized=True)
    for i in range(len(assigns)):
        buf[i] = assigns[i]
    return LayoutTensor[DType.int32, Layout.row_major(len(assigns), 1)()(MutAnyOrigin)](
        buf.unsafe_ptr()
    )


fn test_mxfp4_grouped_matmul_cpu_matches_reference() raises:
    alias tokens = 4
    alias k = 32
    alias out_dim = 8
    alias experts = 2

    var hidden = _build_hidden(tokens, k)

    # Build fp32 weights [E, N, K]
    var w_buf = InlineArray[Float32, experts * out_dim * k](uninitialized=True)
    for i in range(len(w_buf)):
        w_buf[i] = Float32((i % 7) - 3)
    var w = LayoutTensor[DType.float32, Layout.row_major(experts, out_dim, k)()(MutAnyOrigin)](w_buf.unsafe_ptr())

    # Allocate packed Q/E
    var q_buf = InlineArray[UInt8, experts * out_dim * (k // 2)](uninitialized=True)
    var e_buf = InlineArray[UInt8, experts * out_dim * (k // 32)](uninitialized=True)
    var q = LayoutTensor[DType.uint8, Layout.row_major(experts, out_dim, k // 2)()(MutAnyOrigin)](q_buf.unsafe_ptr())
    var e = LayoutTensor[DType.uint8, Layout.row_major(experts, out_dim, k // 32)()(MutAnyOrigin)](e_buf.unsafe_ptr())

    _quantize_weights(w, q, e)

    # Assignments: first two tokens -> expert 0, next two -> expert 1
    var assignments = _build_assignments(List[Int32](0, 0, 1, 1))

    var out_mxfp4_buf = InlineArray[Float32, tokens * out_dim](uninitialized=True)
    var out_mxfp4 = LayoutTensor[DType.float32, Layout.row_major(tokens, out_dim)()(MutAnyOrigin)](out_mxfp4_buf.unsafe_ptr())
    _mxfp4_grouped_matmul_cpu(hidden, q, e, assignments, out_mxfp4)

    var out_ref_buf = InlineArray[Float32, tokens * out_dim](uninitialized=True)
    var out_ref = LayoutTensor[DType.float32, Layout.row_major(tokens, out_dim)()(MutAnyOrigin)](out_ref_buf.unsafe_ptr())
    _reference_matmul(hidden, w, assignments, out_ref)

    assert_almost_equal(out_mxfp4, out_ref, atol=1e-2, rtol=1e-3)


fn test_mxfp4_grouped_matmul_ragged_matches_assignments() raises:
    alias tokens = 6
    alias k = 64
    alias out_dim = 8
    alias experts = 3
    alias max_tokens_per_expert = 4
    alias num_active_experts = 3

    var hidden = _build_hidden(tokens, k)

    # Simple weights same across experts to make comparison straightforward
    var w_buf = InlineArray[Float32, experts * out_dim * k](uninitialized=True)
    for i in range(len(w_buf)):
        w_buf[i] = Float32((i % 5) - 2)
    var w = LayoutTensor[DType.float32, Layout.row_major(experts, out_dim, k)()(MutAnyOrigin)](w_buf.unsafe_ptr())

    var q_buf = InlineArray[UInt8, experts * out_dim * (k // 2)](uninitialized=True)
    var e_buf = InlineArray[UInt8, experts * out_dim * (k // 32)](uninitialized=True)
    var q = LayoutTensor[DType.uint8, Layout.row_major(experts, out_dim, k // 2)()(MutAnyOrigin)](q_buf.unsafe_ptr())
    var e = LayoutTensor[DType.uint8, Layout.row_major(experts, out_dim, k // 32)()(MutAnyOrigin)](e_buf.unsafe_ptr())
    _quantize_weights(w, q, e)

    # Ragged packing: expert 0 gets tokens 0,1; expert 1 gets 2,3; expert 2 gets 4; slot 5 unused (-1)
    var expert_start = InlineArray[UInt32, num_active_experts + 1](uninitialized=True)
    expert_start[0] = 0
    expert_start[1] = 2
    expert_start[2] = 4
    expert_start[3] = 5
    var expert_ids = InlineArray[Int32, num_active_experts](uninitialized=True)
    expert_ids[0] = 0
    expert_ids[1] = 1
    expert_ids[2] = -1  # inactive expert block should be skipped

    alias start_layout = Layout.row_major(num_active_experts + 1)
    alias id_layout = Layout.row_major(num_active_experts)
    var start_lt = _make_layout_tensor[DType.uint32, start_layout](expert_start.unsafe_ptr())
    var ids_lt = _make_layout_tensor[DType.int32, id_layout](expert_ids.unsafe_ptr())

    # Reference assignments for comparison
    var assignments = _build_assignments(List[Int32](0, 0, 1, 1, 2, 2))

    var out_direct_buf = InlineArray[Float32, tokens * out_dim](uninitialized=True)
    var out_direct = LayoutTensor[DType.float32, Layout.row_major(tokens, out_dim)()(MutAnyOrigin)](out_direct_buf.unsafe_ptr())
    _mxfp4_grouped_matmul_cpu(hidden, q, e, assignments, out_direct)

    var out_ragged_buf = InlineArray[Float32, tokens * out_dim](uninitialized=True)
    var out_ragged = LayoutTensor[DType.float32, Layout.row_major(tokens, out_dim)()(MutAnyOrigin)](out_ragged_buf.unsafe_ptr())

    var ctx = DeviceContext()
    _mxfp4_grouped_matmul_ragged_gpu(
        ctx,
        hidden,
        q,
        e,
        start_lt,
        ids_lt,
        max_tokens_per_expert,
        num_active_experts,
        out_ragged,
    )

    assert_almost_equal(out_ragged, out_direct, atol=1e-2, rtol=1e-3)


fn test_mxfp4_grouped_matmul_raises_on_bad_width() raises:
    alias tokens = 1
    alias k = 30  # not divisible by 32*1/2
    alias out_dim = 2

    var hidden = _build_hidden(tokens, k)
    var q_buf = InlineArray[UInt8, out_dim * (k // 2 + 1)](uninitialized=True)
    var e_buf = InlineArray[UInt8, out_dim * (k // 32 + 1)](uninitialized=True)
    var q = LayoutTensor[DType.uint8, Layout.row_major(1, out_dim, k // 2 + 1)()(MutAnyOrigin)](q_buf.unsafe_ptr())
    var e = LayoutTensor[DType.uint8, Layout.row_major(1, out_dim, k // 32 + 1)()(MutAnyOrigin)](e_buf.unsafe_ptr())
    var assignments = _build_assignments(List[Int32](0))
    var out_buf = InlineArray[Float32, tokens * out_dim](uninitialized=True)
    var out = LayoutTensor[DType.float32, Layout.row_major(tokens, out_dim)()(MutAnyOrigin)](out_buf.unsafe_ptr())

    var raised: Bool = False
    try:
        _mxfp4_grouped_matmul_cpu(hidden, q, e, assignments, out)
    except e:
        raised = True
    assert_true(raised)
