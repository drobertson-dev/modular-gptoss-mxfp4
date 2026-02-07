# ===----------------------------------------------------------------------=== #
# MXFP4 grouped matmul microbenchmark (SM90).
#
# Runs grouped_matmul_ragged with Hopper swizzled layouts.
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
from mxfp4 import (
    MXFP4GroupedMatmulRaggedBF16Swizzled,
    MXFP4GroupedMatmulRaggedBF16SwizzledNoSmallM,
)
from mxfp4.layout_hopper import HOPPER_SCALE_NUM_WARPS
from sys import argv, has_nvidia_gpu_accelerator
from tensor import Input, IOSpec, ManagedTensorSlice, Output, StaticTensorSpec


@fieldwise_init
struct Tensor[
    dtype: DType,
    rank: Int,
    //,
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
        print("Skipping MXFP4 grouped bench (no NVIDIA GPU detected)")
        return True
    if ctx.api() != "cuda":
        print("Skipping MXFP4 grouped bench (non-CUDA context)")
        return True
    if ctx.default_device_info.compute < 9.0:
        print("Skipping MXFP4 grouped bench (requires sm90+)")
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
) raises -> Int:
    # Deterministic routing: each token hits TOPK different experts.
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

    return max_tokens_per_expert


fn run_bench[
    TOKENS: Int = 256,
    HIDDEN: Int = 2880,
    INTERMEDIATE: Int = 2880,
    NUM_EXPERTS: Int = 32,
    TOPK: Int = 4,
](
    skip_warmup: Bool = False,
    disable_small_m: Bool = False,
) raises:
    var ctx = DeviceContext()
    if _skip_if_no_sm90(ctx):
        return

    comptime P = TOKENS * TOPK

    # Inputs.
    comptime a_spec = StaticTensorSpec[DType.bfloat16, 2](
        DimList(P, INTERMEDIATE)
    )
    var a = Tensor[Input, a_spec](ctx)
    ctx.enqueue_memset(a.buffer, 0)

    comptime order_spec = StaticTensorSpec[DType.uint32, 1](DimList(P))
    var token_expert_order = Tensor[Input, order_spec](ctx)

    comptime start_spec = StaticTensorSpec[DType.uint32, 1](
        DimList(NUM_EXPERTS + 1)
    )
    var expert_start = Tensor[Input, start_spec](ctx)

    comptime ids_spec = StaticTensorSpec[DType.int32, 1](DimList(NUM_EXPERTS))
    var expert_ids = Tensor[Input, ids_spec](ctx)

    var max_tokens_per_expert = _fill_routing_buffers[
        TOKENS, TOPK, NUM_EXPERTS
    ](
        token_expert_order.buffer,
        expert_start.buffer,
        expert_ids.buffer,
    )

    # Weights: swizzled Hopper value layout + Hopper scales.
    comptime wgm_kbytes = INTERMEDIATE // 2
    comptime wgm_m_pad = ((HIDDEN + 15) // 16) * 16
    comptime wgm_k_pad = ((wgm_kbytes + 31) // 32) * 32
    comptime wgm_m2 = wgm_m_pad // 4
    comptime wgm_k2 = wgm_k_pad * 4
    comptime wgm_blocks_spec = StaticTensorSpec[DType.uint8, 3](
        DimList(NUM_EXPERTS, wgm_m2, wgm_k2)
    )
    var wgm_blocks = Tensor[Input, wgm_blocks_spec](ctx)
    ctx.enqueue_memset(wgm_blocks.buffer, 0)

    comptime scales_m2 = (
        (HIDDEN + (32 * HOPPER_SCALE_NUM_WARPS) - 1)
        // (32 * HOPPER_SCALE_NUM_WARPS)
    ) * HOPPER_SCALE_NUM_WARPS
    comptime wgm_scales_spec = StaticTensorSpec[DType.uint8, 3](
        DimList(NUM_EXPERTS, scales_m2, INTERMEDIATE)
    )
    var wgm_scales = Tensor[Input, wgm_scales_spec](ctx)
    ctx.enqueue_memset(wgm_scales.buffer, 0)

    # Output.
    comptime y_grouped_spec = StaticTensorSpec[DType.bfloat16, 2](
        DimList(P, HIDDEN)
    )
    var y_grouped = Tensor[Output, y_grouped_spec](ctx)

    @parameter
    @always_inline
    fn launch_grouped_matmul(ctx: DeviceContext) raises:
        if disable_small_m:
            MXFP4GroupedMatmulRaggedBF16SwizzledNoSmallM.execute[target="gpu"](
                y_grouped.slice,
                a.slice,
                wgm_blocks.slice,
                wgm_scales.slice,
                expert_start.slice,
                expert_ids.slice,
                UInt32(max_tokens_per_expert),
                UInt32(NUM_EXPERTS),
                ctx,
            )
        else:
            MXFP4GroupedMatmulRaggedBF16Swizzled.execute[target="gpu"](
                y_grouped.slice,
                a.slice,
                wgm_blocks.slice,
                wgm_scales.slice,
                expert_start.slice,
                expert_ids.slice,
                UInt32(max_tokens_per_expert),
                UInt32(NUM_EXPERTS),
                ctx,
            )

    if not skip_warmup:
        launch_grouped_matmul(ctx)
        ctx.synchronize()

    var bench = Bench()
    bench.config.verbose_metric_names = False

    comptime flops = P * HIDDEN * (2 * INTERMEDIATE - 1)
    var metrics = [ThroughputMeasure(BenchMetric.flops, flops)]

    @parameter
    @always_inline
    fn bench_grouped(mut b: Bencher) raises:
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            launch_grouped_matmul(ctx)
        b.iter_custom[kernel_launch](ctx)

    bench.bench_function[bench_grouped](
        BenchId("mxfp4_grouped_matmul_ragged_bf16", "gpu"), metrics
    )

    print(bench)
    bench.dump_report()


fn main() raises:
    # Presets:
    #   --20b (default): hidden=2880, intermediate=2880, experts=32
    #   --120b          : hidden=2880, intermediate=2880, experts=128
    #
    # Token-count presets (compile-time specializations):
    #   --tokens1
    #   --tokens64
    #   --tokens256 (default)
    #   --tokens512
    #   --tokens1024
    #
    # Options:
    #   --skip-warmup
    #   --no-small-m
    var use_120b = False
    var tokens1 = False
    var tokens64 = False
    var tokens512 = False
    var tokens1024 = False
    var skip_warmup = False
    var disable_small_m = False

    for arg in argv():
        if arg == "--120b":
            use_120b = True
        if arg == "--20b":
            use_120b = False
        if arg == "--tokens1":
            tokens1 = True
        if arg == "--tokens64":
            tokens64 = True
        if arg == "--tokens512":
            tokens512 = True
        if arg == "--tokens1024":
            tokens1024 = True
        if arg == "--skip-warmup":
            skip_warmup = True
        if arg == "--no-small-m":
            disable_small_m = True

    if use_120b:
        if tokens1:
            run_bench[TOKENS=1, NUM_EXPERTS=128](
                skip_warmup=skip_warmup,
                disable_small_m=disable_small_m,
            )
        elif tokens64:
            run_bench[TOKENS=64, NUM_EXPERTS=128](
                skip_warmup=skip_warmup,
                disable_small_m=disable_small_m,
            )
        elif tokens512:
            run_bench[TOKENS=512, NUM_EXPERTS=128](
                skip_warmup=skip_warmup,
                disable_small_m=disable_small_m,
            )
        elif tokens1024:
            run_bench[TOKENS=1024, NUM_EXPERTS=128](
                skip_warmup=skip_warmup,
                disable_small_m=disable_small_m,
            )
        else:
            run_bench[NUM_EXPERTS=128](
                skip_warmup=skip_warmup,
                disable_small_m=disable_small_m,
            )
    else:
        if tokens1:
            run_bench[TOKENS=1, NUM_EXPERTS=32](
                skip_warmup=skip_warmup,
                disable_small_m=disable_small_m,
            )
        elif tokens64:
            run_bench[TOKENS=64, NUM_EXPERTS=32](
                skip_warmup=skip_warmup,
                disable_small_m=disable_small_m,
            )
        elif tokens512:
            run_bench[TOKENS=512, NUM_EXPERTS=32](
                skip_warmup=skip_warmup,
                disable_small_m=disable_small_m,
            )
        elif tokens1024:
            run_bench[TOKENS=1024, NUM_EXPERTS=32](
                skip_warmup=skip_warmup,
                disable_small_m=disable_small_m,
            )
        else:
            run_bench[NUM_EXPERTS=32](
                skip_warmup=skip_warmup,
                disable_small_m=disable_small_m,
            )
