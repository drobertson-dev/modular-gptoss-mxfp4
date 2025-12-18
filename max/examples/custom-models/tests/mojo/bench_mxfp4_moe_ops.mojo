# ===----------------------------------------------------------------------=== #
# MXFP4 MoE microbenchmark (SM90).
#
# Uses Mojo's benchmark runner + DeviceContext GPU timers to measure:
#   - `mxfp4_moe_w1_swiglu` (W1 GEMM + fused SwiGLU)
#   - `mxfp4_moe_w2_pairs_bf16` (W2 GEMM writing y_pairs[P, D] as BF16)
#   - `mxfp4_moe_topk_reduce_bf16` (TOPK reduction: y[token, :] = sum_k y_pairs[token*TOPK+k, :])
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
    MXFP4MoETopKReduceBF16,
    MXFP4MoEW1SwiGlu,
    MXFP4MoEW2PairsBF16,
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
    gate_weights: DeviceBuffer[DType.bfloat16],
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
            ptr[i] = BFloat16(1.0)

    return max_tokens_per_expert


fn run_bench[
    TOKENS: Int = 256,
    # GPT-OSS (20B/120B) uses hidden_size=2880 and intermediate_size=2880.
    HIDDEN: Int = 2880,
    INTERMEDIATE: Int = 2880,
    # 20B uses 32 experts; 120B uses 128. The CLI selects the preset.
    NUM_EXPERTS: Int = 32,
    TOPK: Int = 4,
](run_w1: Bool = True, run_w2: Bool = True, run_reduce: Bool = True) raises:
    var ctx = DeviceContext()
    if _skip_if_no_sm90(ctx):
        return

    constrained[HIDDEN % 32 == 0, "HIDDEN must be divisible by 32 for MXFP4"]()
    constrained[
        INTERMEDIATE % 32 == 0, "INTERMEDIATE must be divisible by 32 for MXFP4"
    ]()

    comptime P = TOKENS * TOPK

    # Tensors (allocate once, reuse across measurements).
    comptime x_spec = StaticTensorSpec[DType.bfloat16, 2](
        DimList(TOKENS, HIDDEN)
    )
    var x = Tensor[Input, x_spec](ctx)
    ctx.enqueue_memset(x.buffer, 0)

    comptime order_spec = StaticTensorSpec[DType.uint32, 1](DimList(P))
    var token_expert_order = Tensor[Input, order_spec](ctx)

    comptime start_spec = StaticTensorSpec[DType.uint32, 1](
        DimList(NUM_EXPERTS + 1)
    )
    var expert_start = Tensor[Input, start_spec](ctx)

    comptime ids_spec = StaticTensorSpec[DType.int32, 1](DimList(NUM_EXPERTS))
    var expert_ids = Tensor[Input, ids_spec](ctx)

    comptime gamma_spec = StaticTensorSpec[DType.bfloat16, 1](DimList(P))
    var gate_weights = Tensor[Input, gamma_spec](ctx)

    # Mirrors `moe_create_indices` output: [max_tokens_per_expert, num_active_experts].
    comptime stats_spec = StaticTensorSpec[DType.uint32, 1](DimList(2))
    var expert_usage_stats = Tensor[Input, stats_spec](ctx)

    var max_tokens_per_expert = _fill_routing_buffers[
        TOKENS, TOPK, NUM_EXPERTS
    ](
        token_expert_order.buffer,
        expert_start.buffer,
        expert_ids.buffer,
        gate_weights.buffer,
    )
    with expert_usage_stats.buffer.map_to_host() as host:
        var ptr = host.unsafe_ptr()
        ptr[0] = UInt32(max_tokens_per_expert)
        ptr[1] = UInt32(NUM_EXPERTS)

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

    comptime w2_bias_spec = StaticTensorSpec[DType.float32, 2](
        DimList(NUM_EXPERTS, HIDDEN)
    )
    var w2_bias = Tensor[Input, w2_bias_spec](ctx)
    ctx.enqueue_memset(w2_bias.buffer, 0)

    # Outputs.
    comptime h_spec = StaticTensorSpec[DType.bfloat16, 2](
        DimList(P, INTERMEDIATE)
    )
    var h_sorted = Tensor[Output, h_spec](ctx)
    ctx.enqueue_memset(h_sorted.buffer, 0)
    var h_sorted_in = ManagedTensorSlice[io_spec=Input, static_spec=h_spec](
        h_sorted.buffer.unsafe_ptr(),
        h_spec.shape.into_index_list[2](),
        h_spec.strides.into_index_list[2](),
    )

    # Pair-buffer output for W2 (Triton-style "compute then reduce TOPK").
    comptime y_pairs_bf16_spec = StaticTensorSpec[DType.bfloat16, 2](
        DimList(P, HIDDEN)
    )
    var y_pairs_bf16 = Tensor[Output, y_pairs_bf16_spec](ctx)
    var y_pairs_bf16_in = ManagedTensorSlice[
        io_spec=Input, static_spec=y_pairs_bf16_spec
    ](
        y_pairs_bf16.buffer.unsafe_ptr(),
        y_pairs_bf16_spec.shape.into_index_list[2](),
        y_pairs_bf16_spec.strides.into_index_list[2](),
    )

    comptime y_bf16_spec = StaticTensorSpec[DType.bfloat16, 2](
        DimList(TOKENS, HIDDEN)
    )
    var y_bf16 = Tensor[Output, y_bf16_spec](ctx)

    # Warmup (jit/compile + first-run caches).
    MXFP4MoEW1SwiGlu.execute[target="gpu"](
        h_sorted.slice,
        x.slice,
        token_expert_order.slice,
        expert_start.slice,
        expert_ids.slice,
        expert_usage_stats.slice,
        w1_blocks.slice,
        w1_scales.slice,
        w1_bias.slice,
        1.702,
        7.0,
        ctx,
    )
    MXFP4MoEW2PairsBF16.execute[target="gpu"](
        y_pairs_bf16.slice,
        h_sorted_in,
        token_expert_order.slice,
        expert_start.slice,
        expert_ids.slice,
        expert_usage_stats.slice,
        gate_weights.slice,
        w2_blocks.slice,
        w2_scales.slice,
        w2_bias.slice,
        ctx,
    )
    MXFP4MoETopKReduceBF16.execute[target="gpu"](
        y_bf16.slice,
        y_pairs_bf16_in,
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
                expert_usage_stats.slice,
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
    fn bench_w2_pairs_bf16(mut b: Bencher) raises:
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            MXFP4MoEW2PairsBF16.execute[target="gpu"](
                y_pairs_bf16.slice,
                h_sorted_in,
                token_expert_order.slice,
                expert_start.slice,
                expert_ids.slice,
                expert_usage_stats.slice,
                gate_weights.slice,
                w2_blocks.slice,
                w2_scales.slice,
                w2_bias.slice,
                ctx,
            )

        b.iter_custom[kernel_launch](ctx)

    @parameter
    @always_inline
    fn bench_w2_pairs_bf16_reduce(mut b: Bencher) raises:
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            MXFP4MoEW2PairsBF16.execute[target="gpu"](
                y_pairs_bf16.slice,
                h_sorted_in,
                token_expert_order.slice,
                expert_start.slice,
                expert_ids.slice,
                expert_usage_stats.slice,
                gate_weights.slice,
                w2_blocks.slice,
                w2_scales.slice,
                w2_bias.slice,
                ctx,
            )
            MXFP4MoETopKReduceBF16.execute[target="gpu"](
                y_bf16.slice,
                y_pairs_bf16_in,
                ctx,
            )

        b.iter_custom[kernel_launch](ctx)

    if run_w1:
        bench.bench_function[bench_w1](
            BenchId("mxfp4_moe_w1_swiglu", "gpu"), w1_metrics
        )
    if run_w2:
        bench.bench_function[bench_w2_pairs_bf16](
            BenchId("mxfp4_moe_w2_pairs_bf16", "gpu"), w2_metrics
        )
    if run_w2 and run_reduce:
        bench.bench_function[bench_w2_pairs_bf16_reduce](
            BenchId("mxfp4_moe_w2_pairs_bf16_reduce", "gpu"), w2_metrics
        )

    print(bench)


def main():
    # Presets:
    #   --20b (default): hidden=2880, intermediate=2880, experts=32
    #   --120b          : hidden=2880, intermediate=2880, experts=128
    #
    # Kernel selection:
    #   --w1-only
    #   --w2-only
    #   --no-reduce
    #
    # Token-count presets (compile-time specializations):
    #   --tokens1
    #   --tokens64
    #   --tokens256 (default)
    var use_120b = False
    var tokens1 = False
    var tokens64 = False
    var run_w1 = True
    var run_w2 = True
    var run_reduce = True

    for arg in argv():
        if arg == "--120b":
            use_120b = True
        if arg == "--20b":
            use_120b = False
        if arg == "--tokens1":
            tokens1 = True
        if arg == "--tokens64":
            tokens64 = True
        if arg == "--w1-only":
            run_w2 = False
            run_reduce = False
        if arg == "--w2-only":
            run_w1 = False
        if arg == "--no-reduce":
            run_reduce = False

    if use_120b:
        if tokens1:
            run_bench[TOKENS=1, NUM_EXPERTS=128](
                run_w1=run_w1,
                run_w2=run_w2,
                run_reduce=run_reduce,
            )
        elif tokens64:
            run_bench[TOKENS=64, NUM_EXPERTS=128](
                run_w1=run_w1,
                run_w2=run_w2,
                run_reduce=run_reduce,
            )
        else:
            run_bench[NUM_EXPERTS=128](
                run_w1=run_w1,
                run_w2=run_w2,
                run_reduce=run_reduce,
            )
    else:
        if tokens1:
            run_bench[TOKENS=1, NUM_EXPERTS=32](
                run_w1=run_w1,
                run_w2=run_w2,
                run_reduce=run_reduce,
            )
        elif tokens64:
            run_bench[TOKENS=64, NUM_EXPERTS=32](
                run_w1=run_w1,
                run_w2=run_w2,
                run_reduce=run_reduce,
            )
        else:
            run_bench[NUM_EXPERTS=32](
                run_w1=run_w1,
                run_w2=run_w2,
                run_reduce=run_reduce,
            )
