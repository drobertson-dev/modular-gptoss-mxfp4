# ===----------------------------------------------------------------------=== #
# MXFP4 MoE microbenchmark (SM90).
#
# Uses Mojo's benchmark runner + DeviceContext GPU timers to measure:
#   - `mxfp4_moe_w1_swiglu` (W1 GEMM + fused SwiGLU)
#   - `mxfp4_moe_w2_scatter` (W2 GEMM + gamma scatter-add; baseline w/ atomics)
#   - `mxfp4_moe_w2_pairs` (W2 GEMM writing y_pairs[P, D] with no atomics)
#   - `mxfp4_moe_topk_reduce` (tiny TOPK reduction: y[token, :] = sum_k y_pairs[token*TOPK+k, :])
#
# Run (from `examples/custom-models/`):
#   pixi run mxfp4-moe-bench
# ===----------------------------------------------------------------------=== #

from benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from buffer.dimlist import DimList
from gpu.host import DeviceBuffer, DeviceContext
from kernels import (
    MXFP4MoETopKReduce,
    MXFP4MoEW1SwiGlu,
    MXFP4MoEW2Pairs,
    MXFP4MoEW2Scatter,
)
from sys import argv, has_nvidia_gpu_accelerator
from tensor import Input, IOSpec, ManagedTensorSlice, Output, StaticTensorSpec


@fieldwise_init
struct Tensor[
    dtype: DType,
    rank: Int, //,
    io_spec: IOSpec,
    static_spec: StaticTensorSpec[dtype, rank],
](ImplicitlyCopyable):
    comptime size = Int(Self.static_spec.shape.product())

    var slice: ManagedTensorSlice[
        io_spec = Self.io_spec, static_spec = Self.static_spec
    ]
    var buffer: DeviceBuffer[Self.dtype]

    fn __init__(out self, ctx: DeviceContext) raises:
        self.buffer = ctx.enqueue_create_buffer[Self.dtype](Self.size)
        self.slice = ManagedTensorSlice[
            io_spec = Self.io_spec, static_spec = Self.static_spec
        ](
            self.buffer.unsafe_ptr(),
            Self.static_spec.shape.into_index_list[Self.rank](),
            Self.static_spec.strides.into_index_list[Self.rank](),
        )


fn _skip_if_no_sm90(ctx: DeviceContext) -> Bool:
    if not has_nvidia_gpu_accelerator():
        print("Skipping MXFP4 MoE bench (no NVIDIA GPU detected)")
        return True
    if ctx.api() != "cuda":
        print("Skipping MXFP4 MoE bench (non-CUDA context)")
        return True
    if ctx.default_device_info.compute < 9.0:
        print("Skipping MXFP4 MoE bench (requires sm90+)")
        return True
    return False


fn _fill_routing_buffers[
    TOKENS: Int,
    TOPK: Int,
    num_experts: Int,
](
    token_expert_order: DeviceBuffer[DType.uint32],
    expert_start_indices: DeviceBuffer[DType.uint32],
    expert_ids: DeviceBuffer[DType.int32],
    gate_weights: DeviceBuffer[DType.float32],
) raises -> Int:
    # Fill on host: routing arrays are small; costs are dominated by GEMM anyway.
    comptime P = TOKENS * TOPK
    var max_tokens_per_expert = 0
    with token_expert_order.map_to_host() as order_host:
        with expert_start_indices.map_to_host() as start_host:
            var order_ptr = order_host.unsafe_ptr()
            var start_ptr = start_host.unsafe_ptr()

            var offset = 0
            for e in range(num_experts):
                start_ptr[e] = UInt32(offset)
                var expert_start = offset
                for t in range(TOKENS):
                    for k in range(TOPK):
                        var pair = t * TOPK + k
                        # Deterministic "realistic" routing: each token hits TOPK different experts.
                        var expert = (t + k) % num_experts
                        if expert == e:
                            order_ptr[offset] = UInt32(pair)
                            offset += 1
                var expert_len = offset - expert_start
                if expert_len > max_tokens_per_expert:
                    max_tokens_per_expert = expert_len
            start_ptr[num_experts] = UInt32(offset)

    with expert_ids.map_to_host() as host:
        var ptr = host.unsafe_ptr()
        for e in range(num_experts):
            ptr[e] = Int32(e)

    with gate_weights.map_to_host() as host:
        var ptr = host.unsafe_ptr()
        for i in range(P):
            ptr[i] = 1.0

    return max_tokens_per_expert


fn run_bench[
    TOKENS: Int = 256,
    HIDDEN: Int = 4096,
    INTERMEDIATE: Int = 4096,
    NUM_EXPERTS: Int = 8,
    TOPK: Int = 4,
]() raises:
    var ctx = DeviceContext()
    if _skip_if_no_sm90(ctx):
        return

    constrained[HIDDEN % 32 == 0, "HIDDEN must be divisible by 32 for MXFP4"]()
    constrained[
        INTERMEDIATE % 32 == 0, "INTERMEDIATE must be divisible by 32 for MXFP4"
    ]()

    comptime P = TOKENS * TOPK
    constrained[
        P % NUM_EXPERTS == 0,
        "Default bench expects uniform expert segments (P % NUM_EXPERTS == 0)",
    ]()

    # Tensors (allocate once, reuse across measurements).
    comptime x_spec = StaticTensorSpec[DType.bfloat16, 2](DimList(TOKENS, HIDDEN))
    var x = Tensor[Input, x_spec](ctx)
    ctx.enqueue_memset(x.buffer, 0)

    comptime order_spec = StaticTensorSpec[DType.uint32, 1](DimList(P))
    var token_expert_order = Tensor[Input, order_spec](ctx)

    comptime start_spec = StaticTensorSpec[DType.uint32, 1](DimList(NUM_EXPERTS + 1))
    var expert_start = Tensor[Input, start_spec](ctx)

    comptime ids_spec = StaticTensorSpec[DType.int32, 1](DimList(NUM_EXPERTS))
    var expert_ids = Tensor[Input, ids_spec](ctx)

    comptime gamma_spec = StaticTensorSpec[DType.float32, 1](DimList(P))
    var gate_weights = Tensor[Input, gamma_spec](ctx)

    var max_tokens_per_expert = _fill_routing_buffers[TOKENS, TOPK, NUM_EXPERTS](
        token_expert_order.buffer,
        expert_start.buffer,
        expert_ids.buffer,
        gate_weights.buffer,
    )

    # W1 weights: [E, 2I, D/32, 16] blocks + [E, 2I, D/32] scales, plus FP32 bias.
    comptime kblocks_w1 = HIDDEN // 32
    comptime w1_blocks_spec = StaticTensorSpec[DType.uint8, 4](
        DimList(NUM_EXPERTS, 2 * INTERMEDIATE, kblocks_w1, 16)
    )
    var w1_blocks = Tensor[Input, w1_blocks_spec](ctx)
    ctx.enqueue_memset(w1_blocks.buffer, 0)

    comptime w1_scales_spec = StaticTensorSpec[DType.uint8, 3](
        DimList(NUM_EXPERTS, 2 * INTERMEDIATE, kblocks_w1)
    )
    var w1_scales = Tensor[Input, w1_scales_spec](ctx)
    ctx.enqueue_memset(w1_scales.buffer, 0)

    comptime w1_bias_spec = StaticTensorSpec[DType.float32, 2](
        DimList(NUM_EXPERTS, 2 * INTERMEDIATE)
    )
    var w1_bias = Tensor[Input, w1_bias_spec](ctx)
    ctx.enqueue_memset(w1_bias.buffer, 0)

    # W2 weights: [E, D, I/32, 16] blocks + [E, D, I/32] scales, plus FP32 bias.
    comptime kblocks_w2 = INTERMEDIATE // 32
    comptime w2_blocks_spec = StaticTensorSpec[DType.uint8, 4](
        DimList(NUM_EXPERTS, HIDDEN, kblocks_w2, 16)
    )
    var w2_blocks = Tensor[Input, w2_blocks_spec](ctx)
    ctx.enqueue_memset(w2_blocks.buffer, 0)

    comptime w2_scales_spec = StaticTensorSpec[DType.uint8, 3](
        DimList(NUM_EXPERTS, HIDDEN, kblocks_w2)
    )
    var w2_scales = Tensor[Input, w2_scales_spec](ctx)
    ctx.enqueue_memset(w2_scales.buffer, 0)

    comptime w2_bias_spec = StaticTensorSpec[DType.float32, 2](DimList(NUM_EXPERTS, HIDDEN))
    var w2_bias = Tensor[Input, w2_bias_spec](ctx)
    ctx.enqueue_memset(w2_bias.buffer, 0)

    # Outputs.
    comptime h_spec = StaticTensorSpec[DType.bfloat16, 2](DimList(P, INTERMEDIATE))
    var h_sorted = Tensor[Output, h_spec](ctx)
    ctx.enqueue_memset(h_sorted.buffer, 0)
    var h_sorted_in = ManagedTensorSlice[
        io_spec = Input, static_spec = h_spec
    ](
        h_sorted.buffer.unsafe_ptr(),
        h_spec.shape.into_index_list[2](),
        h_spec.strides.into_index_list[2](),
    )

    comptime y_spec = StaticTensorSpec[DType.float32, 2](DimList(TOKENS, HIDDEN))
    var y = Tensor[Output, y_spec](ctx)

    # Pair-buffer output for W2 (Triton-style "compute then reduce TOPK").
    comptime y_pairs_spec = StaticTensorSpec[DType.float32, 2](DimList(P, HIDDEN))
    var y_pairs = Tensor[Output, y_pairs_spec](ctx)
    var y_pairs_in = ManagedTensorSlice[
        io_spec = Input, static_spec = y_pairs_spec
    ](
        y_pairs.buffer.unsafe_ptr(),
        y_pairs_spec.shape.into_index_list[2](),
        y_pairs_spec.strides.into_index_list[2](),
    )

    # Warmup (jit/compile + first-run caches).
    MXFP4MoEW1SwiGlu.execute[target="gpu"](
        h_sorted.slice,
        x.slice,
        token_expert_order.slice,
        expert_start.slice,
        expert_ids.slice,
        UInt32(max_tokens_per_expert),
        UInt32(NUM_EXPERTS),
        w1_blocks.slice,
        w1_scales.slice,
        w1_bias.slice,
        1.702,
        7.0,
        ctx,
    )
    MXFP4MoEW2Pairs.execute[target="gpu"](
        y_pairs.slice,
        h_sorted_in,
        token_expert_order.slice,
        expert_start.slice,
        expert_ids.slice,
        UInt32(max_tokens_per_expert),
        UInt32(NUM_EXPERTS),
        gate_weights.slice,
        w2_blocks.slice,
        w2_scales.slice,
        w2_bias.slice,
        ctx,
    )
    MXFP4MoETopKReduce.execute[target="gpu"](
        y.slice,
        y_pairs_in,
        ctx,
    )
    MXFP4MoEW2Scatter.execute[target="gpu"](
        y.slice,
        h_sorted_in,
        token_expert_order.slice,
        expert_start.slice,
        expert_ids.slice,
        UInt32(max_tokens_per_expert),
        UInt32(NUM_EXPERTS),
        gate_weights.slice,
        w2_blocks.slice,
        w2_scales.slice,
        w2_bias.slice,
        ctx,
    )
    ctx.synchronize()

    # Benchmark.
    var bench = Bench()
    bench.config.verbose_metric_names = False

    comptime flops_w1 = P * (2 * INTERMEDIATE) * (2 * HIDDEN - 1)
    comptime flops_w2 = P * HIDDEN * (2 * INTERMEDIATE - 1)
    var w1_metrics = [ThroughputMeasure(BenchMetric.flops, flops_w1)]
    var w2_metrics = [ThroughputMeasure(BenchMetric.flops, flops_w2)]

    @parameter
    @always_inline
    fn bench_w1(mut b: Bencher) raises:
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            MXFP4MoEW1SwiGlu.execute[target="gpu"](
                h_sorted.slice,
                x.slice,
                token_expert_order.slice,
                expert_start.slice,
                expert_ids.slice,
                UInt32(max_tokens_per_expert),
                UInt32(NUM_EXPERTS),
                w1_blocks.slice,
                w1_scales.slice,
                w1_bias.slice,
                1.702,
                7.0,
                ctx,
            )

        b.iter_custom[kernel_launch](ctx)

    @parameter
    @always_inline
    fn bench_w2(mut b: Bencher) raises:
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            MXFP4MoEW2Scatter.execute[target="gpu"](
                y.slice,
                h_sorted_in,
                token_expert_order.slice,
                expert_start.slice,
                expert_ids.slice,
                UInt32(max_tokens_per_expert),
                UInt32(NUM_EXPERTS),
                gate_weights.slice,
                w2_blocks.slice,
                w2_scales.slice,
                w2_bias.slice,
                ctx,
            )

        b.iter_custom[kernel_launch](ctx)

    @parameter
    @always_inline
    fn bench_w2_pairs(mut b: Bencher) raises:
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            MXFP4MoEW2Pairs.execute[target="gpu"](
                y_pairs.slice,
                h_sorted_in,
                token_expert_order.slice,
                expert_start.slice,
                expert_ids.slice,
                UInt32(max_tokens_per_expert),
                UInt32(NUM_EXPERTS),
                gate_weights.slice,
                w2_blocks.slice,
                w2_scales.slice,
                w2_bias.slice,
                ctx,
            )

        b.iter_custom[kernel_launch](ctx)

    @parameter
    @always_inline
    fn bench_w2_pairs_reduce(mut b: Bencher) raises:
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            MXFP4MoEW2Pairs.execute[target="gpu"](
                y_pairs.slice,
                h_sorted_in,
                token_expert_order.slice,
                expert_start.slice,
                expert_ids.slice,
                UInt32(max_tokens_per_expert),
                UInt32(NUM_EXPERTS),
                gate_weights.slice,
                w2_blocks.slice,
                w2_scales.slice,
                w2_bias.slice,
                ctx,
            )
            MXFP4MoETopKReduce.execute[target="gpu"](
                y.slice,
                y_pairs_in,
                ctx,
            )

        b.iter_custom[kernel_launch](ctx)

    bench.bench_function[bench_w1](BenchId("mxfp4_moe_w1_swiglu", "gpu"), w1_metrics)
    bench.bench_function[bench_w2](BenchId("mxfp4_moe_w2_scatter", "gpu"), w2_metrics)
    bench.bench_function[bench_w2_pairs](BenchId("mxfp4_moe_w2_pairs", "gpu"), w2_metrics)
    bench.bench_function[bench_w2_pairs_reduce](
        BenchId("mxfp4_moe_w2_pairs_reduce", "gpu"), w2_metrics
    )

    print(bench)


def main():
    # Placeholder for future shape selection via CLI; keep defaults for now.
    _ = argv()
    run_bench[]()
