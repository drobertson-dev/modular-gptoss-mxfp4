---
title: "GPU Synchronization Primitives"
description: "GPU synchronization is the foundation that enables complex parallel algorithms to work correctly and efficiently. This chapter explores three fundamental synchronization patterns that appear throughout high-performance GPU computing: pipeline coordination, memory barrier management, and streaming computation."
---

# GPU Synchronization Primitives

GPU synchronization is the foundation that enables complex parallel algorithms to work correctly and efficiently. This chapter explores three fundamental synchronization patterns that appear throughout high-performance GPU computing: pipeline coordination, memory barrier management, and streaming computation.

> **Beyond Simple Parallelism**
>
> This chapter introduces **synchronization patterns**that enable complex GPU algorithms requiring precise coordination between threads. Unlike previous puzzles that focused on simple parallel operations, these challenges explore **architectural approaches**used in production GPU software.
>
> **What you'll learn:**
>
> - **Thread specialization**: Different thread groups executing distinct algorithms within a single block
> - **Producer-consumer pipelines**: Multi-stage processing with explicit data dependencies
> - **Advanced barrier APIs**: Fine-grained synchronization control beyond basic `barrier()` calls
> - **Memory barrier coordination**: Explicit control over memory visibility and ordering
> - **Iterative algorithm patterns**: Double-buffering and pipeline coordination for complex computations
>
> **Why this matters:**Most GPU tutorials teach simple data-parallel patterns, but real-world applications require **sophisticated coordination**between different processing phases, memory access patterns, and algorithmic stages. These puzzles bridge the gap between academic examples and production GPU computing.

## Overview

GPU synchronization is the foundation that enables complex parallel algorithms to work correctly and efficiently. This chapter explores three fundamental synchronization patterns that appear throughout high-performance GPU computing: **pipeline coordination**, **memory barrier management**, and **streaming computation**.

**Core learning objectives:**

- **Understand when and why**different synchronization primitives are needed
- **Design multi-stage algorithms**with proper thread specialization
- **Implement iterative patterns**that require precise memory coordination
- **Optimize synchronization overhead**while maintaining correctness guarantees

**Architectural progression:**These puzzles follow a carefully designed progression from basic pipeline coordination to advanced memory barrier management, culminating in streaming computation patterns used in high-throughput applications.

## Key concepts

**Thread coordination paradigms:**

- **Simple parallelism**: All threads execute identical operations (previous puzzles)
- **Specialized parallelism**: Different thread groups execute distinct algorithms (this chapter)
- **Pipeline parallelism**: Sequential stages with producer-consumer relationships
- **Iterative parallelism**: Multiple passes with careful buffer management

**Synchronization primitive hierarchy:**

- **Basic [`barrier()`](https://docs.modular.com/mojo/stdlib/gpu/sync/#barrier)**: Simple thread synchronization within blocks
- **Advanced [mbarrier APIs](https://docs.modular.com/mojo/stdlib/gpu/sync/)**: Fine-grained memory barrier control with state tracking
- **Streaming coordination**: Asynchronous copy and bulk transfer synchronization

**Memory consistency models:**

- **Shared memory coordination**: Fast on-chip memory for inter-thread communication
- **Global memory ordering**: Ensuring visibility of writes across different memory spaces
- **Buffer management**: Double-buffering and ping-pong patterns for iterative algorithms

## Configuration

**System architecture:**

- **Block size**: `TPB = 256` threads per block for optimal occupancy
- **Grid configuration**: Multiple blocks processing different data tiles
- **Memory hierarchy**: Strategic use of shared memory, registers, and global memory
- **Data types**: `DType.float32` for numerical computations

**Synchronization patterns covered:**

1. **Multi-stage pipelines**: Thread specialization with barrier coordination
2. **Double-buffered iterations**: Memory barrier management for iterative algorithms
3. **Streaming computation**: Asynchronous copy coordination for high-throughput processing

**Performance considerations:**

- **Synchronization overhead**: Understanding the cost of different barrier types
- **Memory bandwidth**: Optimizing access patterns for maximum throughput
- **Thread utilization**: Balancing specialized roles with overall efficiency

## Puzzle structure

This chapter contains three interconnected puzzles that build upon each other:

### **[Multi-Stage Pipeline Coordination](#multi-stage-pipeline-coordination)**

**Focus**: Thread specialization and pipeline architecture

Learn how to design GPU kernels where different thread groups execute completely different algorithms within the same block. This puzzle introduces **producer-consumer relationships**and strategic barrier placement for coordinating between different algorithmic stages.

**Key concepts**:

- Thread role specialization (Stage 1: load, Stage 2: process, Stage 3: output)
- Producer-consumer data flow between processing stages
- Strategic barrier placement between different algorithms

**Real-world applications**: Image processing pipelines, multi-stage scientific computations, neural network layer coordination

### **[Double-Buffered Stencil Computation](#double-buffered-stencil-computation)**

**Focus**: Advanced memory barrier APIs and iterative processing

Explore **fine-grained synchronization control**using [mbarrier APIs](https://docs.modular.com/mojo/stdlib/gpu/sync/) for iterative algorithms that require precise memory coordination. This puzzle demonstrates double-buffering patterns essential for iterative solvers and simulation algorithms.

**Key concepts**:

- Advanced [mbarrier APIs](https://docs.modular.com/mojo/stdlib/gpu/sync/) vs basic [`barrier()`](https://docs.modular.com/mojo/stdlib/gpu/sync/#barrier)
- Double-buffering with alternating read/write buffer roles
- Iterative algorithm coordination with explicit memory barriers

**Real-world applications**: Iterative solvers (Jacobi, Gauss-Seidel), cellular automata, simulation time-stepping

## Getting started

**Recommended approach:**

1. **Start with [Pipeline Coordination](#multi-stage-pipeline-coordination)**: Understand thread specialization basics
2. **Progress to [Memory Barriers](#double-buffered-stencil-computation)**: Learn fine-grained synchronization control
3. **Apply to streaming patterns**: Combine concepts for high-throughput applications

**Prerequisites:**

- Comfort with basic GPU programming concepts (threads, blocks, shared memory)
- Understanding of memory hierarchies and access patterns
- Familiarity with barrier synchronization from previous puzzles

**Learning outcomes:**
By completing this chapter, you'll have the foundation to design and implement sophisticated GPU algorithms that require precise coordination, preparing you for the architectural complexity found in production GPU computing applications.

**Ready to dive in?**Start with **[Multi-Stage Pipeline Coordination](#multi-stage-pipeline-coordination)**to learn thread specialization fundamentals, then advance to **[Double-Buffered Stencil Computation](#double-buffered-stencil-computation)**to explore advanced memory barrier techniques.

## Multi-Stage Pipeline Coordination

### Overview

Implement a kernel that processes an image through a coordinated 3-stage pipeline where different thread groups handle
specialized processing stages, synchronized with explicit barriers.

**Note:**_You have specialized thread roles: Stage 1 (threads 0-127) loads and preprocesses data, Stage 2 (threads
128-255) applies blur operations, and Stage 3 (all threads) performs final smoothing._

**Algorithm architecture:**This puzzle implements a **producer-consumer pipeline**where different thread groups
execute completely different algorithms within a single GPU block. Unlike traditional GPU programming where all threads
execute the same algorithm on different data, this approach divides threads by **functional specialization**.

**Pipeline concept:**The algorithm processes data through three distinct stages, where each stage has specialized
thread groups that execute different algorithms. Each stage produces data that the next stage consumes, creating
explicit **producer-consumer relationships**that must be carefully synchronized with barriers.

**Data dependencies and synchronization:**Each stage produces data that the next stage consumes:

- **Stage 1  Stage 2**: First stage produces preprocessed data for blur processing
- **Stage 2  Stage 3**: Second stage produces blur results for final smoothing
- **Barriers prevent race conditions**by ensuring complete stage completion before dependent stages begin

Concretely, the multi-stage pipeline implements a coordinated image processing algorithm with three mathematical
operations:

**Stage 1 - Preprocessing Enhancement:**

[P[i] = I[i] times 1.1]

where (P[i]) is the preprocessed data and (I[i]) is the input data.

**Stage 2 - Horizontal Blur Filter:**

[B[i] = frac{1}{N_i} sum_{k=-2}^{2} P[i+k] quad text{where } i+k in [0, 255]]

where (B[i]) is the blur result, and (N_i) is the count of valid neighbors within the tile boundary.

**Stage 3 - Cascading Neighbor Smoothing:**

[F[i] = begin{cases}
(B[i] + B[i+1]) times 0.6 & text{if } i = 0
((B[i] + B[i-1]) times 0.6 + B[i+1]) times 0.6 & text{if } 0 < i < 255
(B[i] + B[i-1]) times 0.6 & text{if } i = 255
end{cases}]

where (F[i]) is the final output with cascading smoothing applied.

**Thread Specialization:**

- **Threads 0-127**: Compute (P[i]) for (i in {0, 1, 2, ldots, 255}) (2 elements per thread)
- **Threads 128-255**: Compute (B[i]) for (i in {0, 1, 2, ldots, 255}) (2 elements per thread)
- **All 256 threads**: Compute (F[i]) for (i in {0, 1, 2, ldots, 255}) (1 element per thread)

**Synchronization Points:**

[text{barrier}_1  P[i] text{ complete}  text{barrier}_2  B[i] text{ complete}  text{barrier}_3  F[i]
text{complete}]

### Key concepts

In this puzzle, you'll learn about:

- Implementing thread role specialization within a single GPU block
- Coordinating producer-consumer relationships between processing stages
- Using barriers to synchronize between different algorithms (not just within the same algorithm)

The key insight is understanding how to design multi-stage pipelines where different thread groups execute completely
different algorithms, coordinated through strategic barrier placement.

**Why this matters:**Most GPU tutorials teach barrier usage within a single algorithm - synchronizing threads during
reductions or shared memory operations. But real-world GPU algorithms often require **architectural complexity**with
multiple distinct processing stages that must be carefully orchestrated. This puzzle demonstrates how to transform
monolithic algorithms into specialized, coordinated processing pipelines.

**Previous vs. current barrier usage:**

- **Previous
  puzzles (P8, P12, P15):**All
  threads execute the same algorithm, barriers sync within algorithm steps
- **This puzzle:**Different thread groups execute different algorithms, barriers coordinate between different
  algorithms

**Thread specialization architecture:**Unlike data parallelism where threads differ only in their data indices, this
puzzle implements **algorithmic parallelism**where threads execute fundamentally different code paths based on their
role in the pipeline.

### Configuration

**System parameters:**

- **Image size**: `SIZE = 1024` elements (1D for simplicity)
- **Threads per block**: `TPB = 256` threads organized as `(256, 1)` block dimension
- **Grid configuration**: `(4, 1)` blocks to process entire image in tiles (4 blocks total)
- **Data type**: `DType.float32` for all computations

**Thread specialization architecture:**

- **Stage 1 threads**: `STAGE1_THREADS = 128` (threads 0-127, first half of block)
  - **Responsibility**: Load input data from global memory and apply preprocessing
  - **Work distribution**: Each thread processes 2 elements for efficient load balancing
  - **Output**: Populates `input_shared[256]` with preprocessed data

- **Stage 2 threads**: `STAGE2_THREADS = 128` (threads 128-255, second half of block)
  - **Responsibility**: Apply horizontal blur filter on preprocessed data
  - **Work distribution**: Each thread processes 2 blur operations
  - **Output**: Populates `blur_shared[256]` with blur results

- **Stage 3 threads**: All 256 threads collaborate
  - **Responsibility**: Final smoothing and output to global memory
  - **Work distribution**: One-to-one mapping (thread `i` processes element `i`)
  - **Output**: Writes final results to global `output` array

### Running the code

To test your solution, run the following command in your terminal:

```bash
pixi run p29 --multi-stage
```

```bash
pixi run -e amd p29 --multi-stage
```

After completing the puzzle successfully, you should see output similar to:

```bash
Puzzle 29: GPU Synchronization Primitives
==================================================
TPB: 256
SIZE: 1024
STAGE1_THREADS: 128
STAGE2_THREADS: 128
BLUR_RADIUS: 2

Testing Puzzle 29A: Multi-Stage Pipeline Coordination
============================================================
Multi-stage pipeline blur completed
Input sample: 0.0 1.01 2.02
Output sample: 1.6665002 2.3331003 3.3996604
OK Multi-stage pipeline coordination test PASSED!
```

### Reference implementation (example)

```mojo
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.sync import (
    mbarrier_init,
    mbarrier_arrive,
    mbarrier_test_wait,
)
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from layout.layout_tensor import copy_dram_to_sram_async
from sys import size_of, argv, info
from testing import assert_true, assert_almost_equal

comptime
TPB = 256  # Threads per block for pipeline stages
comptime
SIZE = 1024  # Image size (1D for simplicity)
comptime
BLOCKS_PER_GRID = (4, 1)
comptime
THREADS_PER_BLOCK = (TPB, 1)
comptime
dtype = DType.float32
comptime
layout = Layout.row_major(SIZE)

# Multi-stage processing configuration
comptime
STAGE1_THREADS = TPB // 2
comptime
STAGE2_THREADS = TPB // 2
comptime
BLUR_RADIUS = 2

# ANCHOR: multi_stage_pipeline_solution
fn
multi_stage_image_blur_pipeline
    layout: Layout
:
"""Multi-stage image blur pipeline with barrier coordination.

Stage 1 (threads 0-127): Load input data and apply 1.1x preprocessing
Stage 2 (threads 128-255): Apply 5-point blur with BLUR_RADIUS=2
Stage 3 (all threads): Final neighbor smoothing and output
"""

# Shared memory buffers for pipeline stages
input_shared = LayoutTensor[
    dtype,
    Layout.row_major(TPB),
    MutAnyOrigin,
    address_space = AddressSpace.SHARED,
].stack_allocation()
blur_shared = LayoutTensor[
    dtype,
    Layout.row_major(TPB),
    MutAnyOrigin,
    address_space = AddressSpace.SHARED,
].stack_allocation()

global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
local_i = Int(thread_idx.x)

# Stage 1: Load and preprocess (threads 0-127)
if local_i < STAGE1_THREADS:
    if global_i < size:
        input_shared[local_i] = input[global_i] * 1.1
        # Each thread loads 2 elements
        if local_i + STAGE1_THREADS < size:
            input_shared[local_i + STAGE1_THREADS] = (
                    input[global_i + STAGE1_THREADS] * 1.1
            )
    else:
        # Zero-padding for out-of-bounds
        input_shared[local_i] = 0.0
        if local_i + STAGE1_THREADS < TPB:
            input_shared[local_i + STAGE1_THREADS] = 0.0

barrier()  # Wait for Stage 1 completion

# Stage 2: Apply blur (threads 128-255)
if local_i >= STAGE1_THREADS:
    blur_idx = local_i - STAGE1_THREADS
    var
    blur_sum: Scalar[dtype] = 0.0
    blur_count = 0

    # 5-point blur kernel
    for offset in range(-BLUR_RADIUS, BLUR_RADIUS + 1):
        sample_idx = blur_idx + offset
        if sample_idx >= 0 and sample_idx < TPB:
            blur_sum += rebind[Scalar[dtype]](input_shared[sample_idx])
            blur_count += 1

    if blur_count > 0:
        blur_shared[blur_idx] = blur_sum / blur_count
    else:
        blur_shared[blur_idx] = 0.0

    # Process second element
    second_idx = blur_idx + STAGE1_THREADS
    if second_idx < TPB:
        blur_sum = 0.0
        blur_count = 0
        for offset in range(-BLUR_RADIUS, BLUR_RADIUS + 1):
            sample_idx = second_idx + offset
            if sample_idx >= 0 and sample_idx < TPB:
                blur_sum += rebind[Scalar[dtype]](input_shared[sample_idx])
                blur_count += 1

        if blur_count > 0:
            blur_shared[second_idx] = blur_sum / blur_count
        else:
            blur_shared[second_idx] = 0.0

barrier()  # Wait for Stage 2 completion

# Stage 3: Final smoothing (all threads)
if global_i < size:
    final_value = blur_shared[local_i]

    # Neighbor smoothing with 0.6 scaling
    if local_i > 0:
        final_value = (final_value + blur_shared[local_i - 1]) * 0.6
    if local_i < TPB - 1:
        final_value = (final_value + blur_shared[local_i + 1]) * 0.6

    output[global_i] = final_value

barrier()  # Ensure all writes complete

# ANCHOR_END: multi_stage_pipeline_solution

# Double-buffered stencil configuration
comptime
STENCIL_ITERATIONS = 3
comptime
BUFFER_COUNT = 2

# ANCHOR: double_buffered_stencil_solution
fn
double_buffered_stencil_computation
    layout: Layout
:
"""Double-buffered stencil computation with memory barrier coordination.

Iteratively applies 3-point stencil using alternating buffers.
Uses mbarrier APIs for precise buffer swap coordination.
"""

# Double-buffering: Two shared memory buffers
buffer_A = LayoutTensor[
    dtype,
    Layout.row_major(TPB),
    MutAnyOrigin,
    address_space = AddressSpace.SHARED,
].stack_allocation()
buffer_B = LayoutTensor[
    dtype,
    Layout.row_major(TPB),
    MutAnyOrigin,
    address_space = AddressSpace.SHARED,
].stack_allocation()

# Memory barriers for coordinating buffer swaps
init_barrier = LayoutTensor[
    DType.uint64,
    Layout.row_major(1),
    MutAnyOrigin,
    address_space = AddressSpace.SHARED,
].stack_allocation()
iter_barrier = LayoutTensor[
    DType.uint64,
    Layout.row_major(1),
    MutAnyOrigin,
    address_space = AddressSpace.SHARED,
].stack_allocation()
final_barrier = LayoutTensor[
    DType.uint64,
    Layout.row_major(1),
    MutAnyOrigin,
    address_space = AddressSpace.SHARED,
].stack_allocation()

global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
local_i = Int(thread_idx.x)

# Initialize barriers (only thread 0)
if local_i == 0:
    mbarrier_init(init_barrier.ptr, TPB)
    mbarrier_init(iter_barrier.ptr, TPB)
    mbarrier_init(final_barrier.ptr, TPB)

# Initialize buffer_A with input data
if local_i < TPB and global_i < size:
    buffer_A[local_i] = input[global_i]
else:
    buffer_A[local_i] = 0.0

# Wait for buffer_A initialization
_ = mbarrier_arrive(init_barrier.ptr)
_ = mbarrier_test_wait(init_barrier.ptr, TPB)

# Iterative stencil processing with double-buffering
@parameter

for iteration in range(STENCIL_ITERATIONS):

    @parameter

    if iteration % 2 == 0:
        # Even iteration: Read from A, Write to B
        if local_i < TPB:
            var
            stencil_sum: Scalar[dtype] = 0.0
            var
            stencil_count: Int = 0

            # 3-point stencil: [i-1, i, i+1]
            for offset in range(-1, 2):
                sample_idx = local_i + offset
                if sample_idx >= 0 and sample_idx < TPB:
                    stencil_sum += rebind[Scalar[dtype]](
                        buffer_A[sample_idx]
                    )
                    stencil_count += 1

            if stencil_count > 0:
                buffer_B[local_i] = stencil_sum / stencil_count
            else:
                buffer_B[local_i] = buffer_A[local_i]

    else:
        # Odd iteration: Read from B, Write to A
        if local_i < TPB:
            var
            stencil_sum: Scalar[dtype] = 0.0
            var
            stencil_count: Int = 0

            # 3-point stencil: [i-1, i, i+1]
            for offset in range(-1, 2):
                sample_idx = local_i + offset
                if sample_idx >= 0 and sample_idx < TPB:
                    stencil_sum += rebind[Scalar[dtype]](
                        buffer_B[sample_idx]
                    )
                    stencil_count += 1

            if stencil_count > 0:
                buffer_A[local_i] = stencil_sum / stencil_count
            else:
                buffer_A[local_i] = buffer_B[local_i]

    # Memory barrier: wait for all writes before buffer swap
    _ = mbarrier_arrive(iter_barrier.ptr)
    _ = mbarrier_test_wait(iter_barrier.ptr, TPB)

    # Reinitialize barrier for next iteration
    if local_i == 0:
        mbarrier_init(iter_barrier.ptr, TPB)

# Write final results from active buffer
if local_i < TPB and global_i < size:

    @parameter

    if STENCIL_ITERATIONS % 2 == 0:
        # Even iterations end in buffer_A
        output[global_i] = buffer_A[local_i]
    else:
        # Odd iterations end in buffer_B
        output[global_i] = buffer_B[local_i]

# Final barrier
_ = mbarrier_arrive(final_barrier.ptr)
_ = mbarrier_test_wait(final_barrier.ptr, TPB)

# ANCHOR_END: double_buffered_stencil_solution

def test_multi_stage_pipeline():
    """Test Puzzle 26A: Multi-Stage Pipeline Coordination."""
    with DeviceContext() as ctx:
        out = ctx.enqueue_create_bufferdtype
        out.enqueue_fill(0)
        inp = ctx.enqueue_create_bufferdtype
        inp.enqueue_fill(0)

        # Initialize input with a simple pattern
        with inp.map_to_host() as inp_host:
            for i in range(SIZE):
                # Create a simple wave pattern for blurring
                inp_host[i] = Float32(i % 10) + Float32(i / 100.0)

        # Create LayoutTensors
        out_tensor = LayoutTensordtype, layout, MutAnyOrigin
        inp_tensor = LayoutTensordtype, layout, ImmutAnyOrigin

        comptime
        kernel = multi_stage_image_blur_pipeline[layout]
        ctx.enqueue_functionkernel, kernel

        ctx.synchronize()

        # Simple verification - check that output differs from input and values are reasonable
        with out.map_to_host() as out_host, inp.map_to_host() as inp_host:
            print("Multi-stage pipeline blur completed")
            print("Input sample:", inp_host[0], inp_host[1], inp_host[2])
            print("Output sample:", out_host[0], out_host[1], out_host[2])

            # Basic verification - output should be different from input (pipeline processed them)
            assert_true(
                abs(out_host[0] - inp_host[0]) > 0.001,
                "Pipeline should modify values",
            )
            assert_true(
                abs(out_host[1] - inp_host[1]) > 0.001,
                "Pipeline should modify values",
            )
            assert_true(
                abs(out_host[2] - inp_host[2]) > 0.001,
                "Pipeline should modify values",
            )

            # Values should be reasonable (not NaN, not extreme)
            for i in range(10):
                assert_true(
                    out_host[i] >= 0.0, "Output values should be non-negative"
                )
                assert_true(
                    out_host[i] < 1000.0, "Output values should be reasonable"
                )

            print("OK Multi-stage pipeline coordination test PASSED!")

def test_double_buffered_stencil():
    """Test Puzzle 26B: Double-Buffered Stencil Computation."""
    with DeviceContext() as ctx:
        # Test Puzzle 26B: Double-Buffered Stencil Computation
        out = ctx.enqueue_create_bufferdtype
        out.enqueue_fill(0)
        inp = ctx.enqueue_create_bufferdtype
        inp.enqueue_fill(0)

        # Initialize input with a different pattern for stencil testing
        with inp.map_to_host() as inp_host:
            for i in range(SIZE):
                # Create a step pattern that will be smoothed by stencil
                inp_host[i] = Float32(1.0 if i % 20 < 10 else 0.0)

        # Create LayoutTensors for Puzzle 26B
        out_tensor = LayoutTensordtype, layout, MutAnyOrigin
        inp_tensor = LayoutTensordtype, layout, ImmutAnyOrigin

        comptime
        kernel = double_buffered_stencil_computation[layout]
        ctx.enqueue_functionkernel, kernel

        ctx.synchronize()

        # Simple verification - check that GPU implementation works correctly
        with inp.map_to_host() as inp_host, out.map_to_host() as out_host:
            print("Double-buffered stencil completed")
            print("Input sample:", inp_host[0], inp_host[1], inp_host[2])
            print("GPU output sample:", out_host[0], out_host[1], out_host[2])

            # Basic sanity checks
            var
            processing_occurred = False
            var
            all_values_valid = True

            for i in range(SIZE):
                # Check if processing occurred (output should differ from step pattern)
                if abs(out_host[i] - inp_host[i]) > 0.001:
                    processing_occurred = True

                # Check for invalid values (NaN, infinity, or out of reasonable range)
                if out_host[i] < 0.0 or out_host[i] > 1.0:
                    all_values_valid = False
                    break

            # Verify the stencil smoothed the step pattern
            assert_true(
                processing_occurred, "Stencil should modify the input values"
            )
            assert_true(
                all_values_valid,
                "All output values should be in valid range [0,1]",
            )

            # Check that values are smoothed (no sharp transitions)
            var
            smooth_transitions = True
            for i in range(1, SIZE - 1):
                # Check if transitions are reasonably smooth (not perfect step function)
                var
                left_diff = abs(out_host[i] - out_host[i - 1])
                var
                right_diff = abs(out_host[i + 1] - out_host[i])
                # After 3 stencil iterations, sharp 0->1 transitions should be smoothed
                if left_diff > 0.8 or right_diff > 0.8:
                    smooth_transitions = False
                    break

            assert_true(
                smooth_transitions, "Stencil should smooth sharp transitions"
            )

            print("OK Double-buffered stencil test PASSED!")

def main():
    """Run GPU synchronization tests based on command line arguments."""
    print("Puzzle 26: GPU Synchronization Primitives")
    print("=" * 50)

    # Parse command line arguments
    if len(argv()) != 2:
        print("Usage: p26.mojo [--multi-stage | --double-buffer]")
        print("  --multi-stage: Test multi-stage pipeline coordination")
        print("  --double-buffer: Test double-buffered stencil computation")
        return

    if argv()[1] == "--multi-stage":
        print("TPB:", TPB)
        print("SIZE:", SIZE)
        print("STAGE1_THREADS:", STAGE1_THREADS)
        print("STAGE2_THREADS:", STAGE2_THREADS)
        print("BLUR_RADIUS:", BLUR_RADIUS)
        print("")
        print("Testing Puzzle 26A: Multi-Stage Pipeline Coordination")
        print("=" * 60)
        test_multi_stage_pipeline()
    elif argv()[1] == "--double-buffer":
        print("TPB:", TPB)
        print("SIZE:", SIZE)
        print("STENCIL_ITERATIONS:", STENCIL_ITERATIONS)
        print("BUFFER_COUNT:", BUFFER_COUNT)
        print("")
        print("Testing Puzzle 26B: Double-Buffered Stencil Computation")
        print("=" * 60)
        test_double_buffered_stencil()
    else:
        print("Usage: p26.mojo [--multi-stage | --double-buffer]")

```

### Explanation

The key insight is recognizing this as a **pipeline architecture problem**with thread role specialization:

1. **Design stage-specific thread groups**: Divide threads by function, not just by data
2. **Implement producer-consumer chains**: Stage 1 produces for Stage 2, Stage 2 produces for Stage 3
3. **Use strategic barrier placement**: Synchronize between different algorithms, not within the same algorithm
4. **Optimize memory access patterns**: Ensure coalesced reads and efficient shared memory usage

**Complete Solution with Detailed Explanation**

The multi-stage pipeline solution demonstrates sophisticated thread specialization and barrier coordination. This
approach transforms a traditional monolithic GPU algorithm into a specialized, coordinated processing pipeline.

### **Pipeline architecture design**

The fundamental breakthrough in this puzzle is **thread specialization by role**rather than by data:

**Traditional approach:**All threads execute the same algorithm on different data

- Everyone performs identical operations (like reductions or matrix operations)
- Barriers synchronize threads within the same algorithm steps
- Thread roles differ only by data indices they process

**This puzzle's innovation:**Different thread groups execute completely different algorithms

- Threads 0-127 execute loading and preprocessing algorithms
- Threads 128-255 execute blur processing algorithms
- All threads collaborate in final smoothing algorithm
- Barriers coordinate between different algorithms, not within the same algorithm

### **Producer-consumer coordination**

Unlike previous puzzles where threads were peers in the same algorithm, this establishes explicit producer-consumer
relationships:

- **Stage 1**: Producer (creates preprocessed data for Stage 2)
- **Stage 2**: Consumer (uses Stage 1 data) + Producer (creates blur data for Stage 3)
- **Stage 3**: Consumer (uses Stage 2 data)

### **Strategic barrier placement**

Understanding when barriers are necessary vs. wasteful:

- **Necessary**: Between dependent stages to prevent race conditions
- **Wasteful**: Within independent operations of the same stage
- **Performance insight**: Each barrier has a cost - use them strategically

**Critical synchronization points:**

1. **After Stage 1**: Prevent Stage 2 from reading incomplete preprocessed data
2. **After Stage 2**: Prevent Stage 3 from reading incomplete blur results
3. **After Stage 3**: Ensure all output writes complete before block termination

### **Thread utilization patterns**

- **Stage 1**: 50% utilization (128/256 threads active, 128 idle)
- **Stage 2**: 50% utilization (128 active, 128 idle)
- **Stage 3**: 100% utilization (all 256 threads active)

This demonstrates sophisticated **algorithmic parallelism**where different thread groups specialize in different
computational tasks within a coordinated pipeline, moving beyond simple data parallelism to architectural thinking
required for real-world GPU algorithms.

### **Memory hierarchy optimization**

**Shared memory architecture:**

- Two specialized buffers handle data flow between stages
- Global memory access minimized to boundary operations only
- All intermediate processing uses fast shared memory

**Access pattern benefits:**

- **Stage 1**: Coalesced global memory reads for input loading
- **Stage 2**: Fast shared memory reads for blur processing
- **Stage 3**: Coalesced global memory writes for output

### **Real-world applications**

This pipeline architecture pattern is fundamental to:

**Image processing pipelines:**

- Multi-stage filters (blur, sharpen, edge detection in sequence)
- Color space conversions (RGB  HSV  processing  RGB)
- Noise reduction with multiple algorithm passes

**Scientific computing:**

- Stencil computations with multi-stage finite difference methods
- Signal processing with filtering, transformation, and analysis pipelines
- Computational fluid dynamics with multi-stage solver iterations

**Machine learning:**

- Neural network layers with specialized thread groups for different operations
- Data preprocessing pipelines (load, normalize, augment in coordinated stages)
- Batch processing where different thread groups handle different operations

### **Key technical insights**

**Algorithmic vs. data parallelism:**

- **Data parallelism**: Threads execute identical code on different data elements
- **Algorithmic parallelism**: Threads execute fundamentally different algorithms based on their specialized roles

**Barrier usage philosophy:**

- **Strategic placement**: Barriers only where necessary to prevent race conditions between dependent stages
- **Performance consideration**: Each barrier incurs synchronization overhead - use sparingly but correctly
- **Correctness guarantee**: Proper barrier placement ensures deterministic results regardless of thread execution
  timing

**Thread specialization benefits:**

- **Algorithmic optimization**: Each stage can be optimized for its specific computational pattern
- **Memory access optimization**: Different stages can use different memory access strategies
- **Resource utilization**: Complex algorithms can be decomposed into specialized, efficient components

This solution demonstrates how to design sophisticated GPU algorithms that leverage thread specialization and strategic
synchronization for complex multi-stage computations, moving beyond simple parallel loops to architectural approaches
used in production GPU software.

## Double-Buffered Stencil Computation

> ** Fine-Grained Synchronization: mbarrier vs barrier()**
>
> This puzzle introduces **explicit memory barrier APIs**that provide significantly more control than the basic [`barrier()`](https://docs.modular.com/mojo/stdlib/gpu/sync/#barrier) function used in previous puzzles.
>
> **Basic `barrier()` limitations:**
>
> - **Fire-and-forget**: Single synchronization point with no state tracking
> - **Block-wide only**: All threads in the block must participate simultaneously
> - **No reusability**: Each barrier() call creates a new synchronization event
> - **Coarse-grained**: Limited control over memory ordering and timing
> - **Static coordination**: Cannot adapt to different thread participation patterns
>
> **Advanced [`mbarrier APIs`](https://docs.modular.com/mojo/stdlib/gpu/sync/) capabilities:**
>
> - **Precise control**: [`mbarrier_init()`](https://docs.modular.com/mojo/stdlib/gpu/sync/#mbarrier_init) sets up reusable barrier objects with specific thread counts
> - **State tracking**: [`mbarrier_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/sync/#mbarrier_arrive) signals individual thread completion and maintains arrival count
> - **Flexible waiting**: [`mbarrier_test_wait()`](https://docs.modular.com/mojo/stdlib/gpu/sync/#mbarrier_test_wait) allows threads to wait for specific completion states
> - **Reusable objects**: Same barrier can be reinitialized and reused across multiple iterations
> - **Multiple barriers**: Different barrier objects for different synchronization points (initialization, iteration, finalization)
> - **Hardware optimization**: Maps directly to GPU hardware synchronization primitives for better performance
> - **Memory semantics**: Explicit control over memory visibility and ordering guarantees
>
> **Why this matters for iterative algorithms:**
> In double-buffering patterns, you need **precise coordination**between buffer swap phases. Basic `barrier()` cannot provide the fine-grained control required for:
>
> - **Buffer role alternation**: Ensuring all writes to buffer_A complete before reading from buffer_A begins
> - **Iteration boundaries**: Coordinating multiple synchronization points within a single kernel
> - **State management**: Tracking which threads have completed which phase of processing
> - **Performance optimization**: Minimizing synchronization overhead through reusable barrier objects
>
> This puzzle demonstrates **synchronization patterns**used in real-world GPU computing applications like iterative solvers, simulation frameworks, and high-performance image processing pipelines.

### Overview

Implement a kernel that performs iterative stencil operations using double-buffered shared memory, coordinated with explicit memory barriers to ensure safe buffer swapping between iterations. A stencil operation is a computational pattern where the value of each element in an array is calculated based on a fixed pattern of its neighbors.

**Note:**_You have alternating buffer roles: `buffer_A` and `buffer_B` swap between read and write operations each iteration, with mbarrier synchronization ensuring all threads complete writes before buffer swaps._

**Algorithm architecture:**This puzzle implements a **double-buffering pattern**where two shared memory buffers alternate roles as read and write targets across multiple iterations. Unlike simple stencil operations that process data once, this approach performs iterative refinement with careful memory barrier coordination to prevent race conditions during buffer transitions.

**Pipeline concept:**The algorithm processes data through iterative stencil refinement, where each iteration reads from one buffer and writes to another. The buffers alternate roles each iteration, creating a ping-pong pattern that enables continuous processing without data corruption.

**Data dependencies and synchronization:**Each iteration depends on the complete results of the previous iteration:

- **Iteration N  Iteration N+1**: Current iteration produces refined data that next iteration consumes
- **Buffer coordination**: Read and write buffers swap roles each iteration
- **Memory barriers prevent race conditions**by ensuring all writes complete before any thread begins reading from the newly written buffer

Concretely, the double-buffered stencil implements an iterative smoothing algorithm with three mathematical operations:

**Iteration Pattern - Buffer Alternation:**

\\[\\text{Iteration } i: \\begin{cases}
\\text{Read from buffer\_A, Write to buffer\_B} & \\text{if } i \\bmod 2 = 0 \\\\
\\text{Read from buffer\_B, Write to buffer\_A} & \\text{if } i \\bmod 2 = 1
\\end{cases}\\]

**Stencil Operation - 3-Point Average:**

\\[S^{(i+1)}[j] = \\frac{1}{N_j} \\sum_{k=-1}^{1} S^{(i)}[j+k] \\quad \\text{where } j+k \\in [0, 255]\\]

where \\(S^{(i)}[j]\\) is the stencil value at position \\(j\\) after iteration \\(i\\), and \\(N_j\\) is the count of valid neighbors.

**Memory Barrier Coordination:**

\\[\\text{mbarrier\_arrive}() \\Rightarrow \\text{mbarrier\_test\_wait}() \\Rightarrow \\text{buffer swap} \\Rightarrow \\text{next iteration}\\]

**Final Output Selection:**

\\[\\text{Output}[j] = \\begin{cases}
\\text{buffer\_A}[j] & \\text{if STENCIL\_ITERATIONS } \\bmod 2 = 0 \\\\
\\text{buffer\_B}[j] & \\text{if STENCIL\_ITERATIONS } \\bmod 2 = 1
\\end{cases}\\]

### Key concepts

In this puzzle, you'll learn about:

- Implementing double-buffering patterns for iterative algorithms
- Coordinating explicit memory barriers using [mbarrier APIs](https://docs.modular.com/mojo/stdlib/gpu/sync/)
- Managing alternating read/write buffer roles across iterations

The key insight is understanding how to safely coordinate buffer swapping in iterative algorithms where race conditions between read and write operations can corrupt data if not properly synchronized.

**Why this matters:**Most GPU tutorials show simple one-pass algorithms, but real-world applications often require **iterative refinement**with multiple passes over data. Double-buffering is essential for algorithms like iterative solvers, image processing filters, and simulation updates where each iteration depends on the complete results of the previous iteration.

**Previous vs. current synchronization:**

- **Previous puzzles (P8, P12, P15):**Simple [`barrier()`](https://docs.modular.com/mojo/stdlib/gpu/sync/#barrier) calls for single-pass algorithms
- **This puzzle:**Explicit [mbarrier APIs](https://docs.modular.com/mojo/stdlib/gpu/sync/) for precise control over buffer swap timing

**Memory barrier specialization:**Unlike basic thread synchronization, this puzzle uses **explicit memory barriers**that provide fine-grained control over when memory operations complete, essential for complex memory access patterns.

### Configuration

**System parameters:**

- **Image size**: `SIZE = 1024` elements (1D for simplicity)
- **Threads per block**: `TPB = 256` threads organized as `(256, 1)` block dimension
- **Grid configuration**: `(4, 1)` blocks to process entire image in tiles (4 blocks total)
- **Data type**: `DType.float32` for all computations

**Iteration parameters:**

- **Stencil iterations**: `STENCIL_ITERATIONS = 3` refinement passes
- **Buffer count**: `BUFFER_COUNT = 2` (double-buffering)
- **Stencil kernel**: 3-point averaging with radius 1

**Buffer architecture:**

- **buffer_A**: Primary shared memory buffer (`[256]` elements)
- **buffer_B**: Secondary shared memory buffer (`[256]` elements)
- **Role alternation**: Buffers swap between read source and write target each iteration

**Processing requirements:**

**Initialization phase:**

- **Buffer setup**: Initialize buffer_A with input data, buffer_B with zeros
- **Barrier initialization**: Set up [mbarrier objects](https://docs.modular.com/mojo/stdlib/gpu/sync/#mbarrier_init) for synchronization points
- **Thread coordination**: All threads participate in initialization

**Iterative processing:**

- **Even iterations**(0, 2, 4...): Read from buffer_A, write to buffer_B
- **Odd iterations**(1, 3, 5...): Read from buffer_B, write to buffer_A
- **Stencil operation**: 3-point average \\((\\text{left} + \\text{center} + \\text{right}) / 3\\)
- **Boundary handling**: Use adaptive averaging for elements at buffer edges

**Memory barrier coordination:**

- **[mbarrier_arrive()](https://docs.modular.com/mojo/stdlib/gpu/sync/#mbarrier_arrive)**: Each thread signals completion of write phase
- **[mbarrier_test_wait()](https://docs.modular.com/mojo/stdlib/gpu/sync/#mbarrier_test_wait)**: All threads wait until everyone completes writes
- **Buffer swap safety**: Prevents reading from buffer while others still writing
- **Barrier reinitialization**: Reset barrier state between iterations

**Output phase:**

- **Final buffer selection**: Choose active buffer based on iteration parity
- **Global memory write**: Copy final results to output array
- **Completion barrier**: Ensure all writes finish before block termination

### Running the code

To test your solution, run the following command in your terminal:

  
    pixi NVIDIA (default)
    pixi AMD
    uv
  
  

```bash
pixi run p29 --double-buffer
```

  
  

```bash
pixi run -e amd p29 --double-buffer
```

  
  

```bash
uv run poe p29 --double-buffer
```

  

After completing the puzzle successfully, you should see output similar to:

```
Puzzle 29: GPU Synchronization Primitives
==================================================
TPB: 256
SIZE: 1024
STENCIL_ITERATIONS: 3
BUFFER_COUNT: 2

Testing Puzzle 29B: Double-Buffered Stencil Computation
============================================================
Double-buffered stencil completed
Input sample: 1.0 1.0 1.0
GPU output sample: 1.0 1.0 1.0
OK Double-buffered stencil test PASSED!
```

### Reference implementation (example)


```mojo
fn double_buffered_stencil_computation
    layout: Layout
:
    """Double-buffered stencil computation with memory barrier coordination.

    Iteratively applies 3-point stencil using alternating buffers.
    Uses mbarrier APIs for precise buffer swap coordination.
    """

    # Double-buffering: Two shared memory buffers
    buffer_A = LayoutTensor[
        dtype,
        Layout.row_major(TPB),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    buffer_B = LayoutTensor[
        dtype,
        Layout.row_major(TPB),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Memory barriers for coordinating buffer swaps
    init_barrier = LayoutTensor[
        DType.uint64,
        Layout.row_major(1),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    iter_barrier = LayoutTensor[
        DType.uint64,
        Layout.row_major(1),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    final_barrier = LayoutTensor[
        DType.uint64,
        Layout.row_major(1),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    local_i = Int(thread_idx.x)

    # Initialize barriers (only thread 0)
    if local_i == 0:
        mbarrier_init(init_barrier.ptr, TPB)
        mbarrier_init(iter_barrier.ptr, TPB)
        mbarrier_init(final_barrier.ptr, TPB)

    # Initialize buffer_A with input data
    if local_i < TPB and global_i < size:
        buffer_A[local_i] = input[global_i]
    else:
        buffer_A[local_i] = 0.0

    # Wait for buffer_A initialization
    _ = mbarrier_arrive(init_barrier.ptr)
    _ = mbarrier_test_wait(init_barrier.ptr, TPB)

    # Iterative stencil processing with double-buffering
    @parameter
    for iteration in range(STENCIL_ITERATIONS):

        @parameter
        if iteration % 2 == 0:
            # Even iteration: Read from A, Write to B
            if local_i < TPB:
                var stencil_sum: Scalar[dtype] = 0.0
                var stencil_count: Int = 0

                # 3-point stencil: [i-1, i, i+1]
                for offset in range(-1, 2):
                    sample_idx = local_i + offset
                    if sample_idx >= 0 and sample_idx < TPB:
                        stencil_sum += rebind[Scalar[dtype]](
                            buffer_A[sample_idx]
                        )
                        stencil_count += 1

                if stencil_count > 0:
                    buffer_B[local_i] = stencil_sum / stencil_count
                else:
                    buffer_B[local_i] = buffer_A[local_i]

        else:
            # Odd iteration: Read from B, Write to A
            if local_i < TPB:
                var stencil_sum: Scalar[dtype] = 0.0
                var stencil_count: Int = 0

                # 3-point stencil: [i-1, i, i+1]
                for offset in range(-1, 2):
                    sample_idx = local_i + offset
                    if sample_idx >= 0 and sample_idx < TPB:
                        stencil_sum += rebind[Scalar[dtype]](
                            buffer_B[sample_idx]
                        )
                        stencil_count += 1

                if stencil_count > 0:
                    buffer_A[local_i] = stencil_sum / stencil_count
                else:
                    buffer_A[local_i] = buffer_B[local_i]

        # Memory barrier: wait for all writes before buffer swap
        _ = mbarrier_arrive(iter_barrier.ptr)
        _ = mbarrier_test_wait(iter_barrier.ptr, TPB)

        # Reinitialize barrier for next iteration
        if local_i == 0:
            mbarrier_init(iter_barrier.ptr, TPB)

    # Write final results from active buffer
    if local_i < TPB and global_i < size:

        @parameter
        if STENCIL_ITERATIONS % 2 == 0:
            # Even iterations end in buffer_A
            output[global_i] = buffer_A[local_i]
        else:
            # Odd iterations end in buffer_B
            output[global_i] = buffer_B[local_i]

    # Final barrier
    _ = mbarrier_arrive(final_barrier.ptr)
    _ = mbarrier_test_wait(final_barrier.ptr, TPB)

```

The key insight is recognizing this as a **double-buffering architecture problem**with explicit memory barrier coordination:

1. **Design alternating buffer roles**: Swap read/write responsibilities each iteration
2. **Implement explicit memory barriers**: Use mbarrier APIs for precise synchronization control
3. **Coordinate iterative processing**: Ensure complete iteration results before buffer swaps
4. **Optimize memory access patterns**: Keep all processing in fast shared memory

Complete Solution with Detailed Explanation

The double-buffered stencil solution demonstrates sophisticated memory barrier coordination and iterative processing patterns. This approach enables safe iterative refinement algorithms that require precise control over memory access timing.

### **Double-buffering architecture design**

The fundamental breakthrough in this puzzle is **explicit memory barrier control**rather than simple thread synchronization:

**Traditional approach:**Use basic [`barrier()`](https://docs.modular.com/mojo/stdlib/gpu/sync/#barrier) for simple thread coordination

- All threads execute same operation on different data
- Single barrier call synchronizes thread completion
- No control over specific memory operation timing

**This puzzle's innovation:**Different buffer roles coordinated with explicit memory barriers

- buffer_A and buffer_B alternate between read source and write target
- [mbarrier APIs](https://docs.modular.com/mojo/stdlib/gpu/sync/) provide precise control over memory operation completion
- Explicit coordination prevents race conditions during buffer transitions

### **Iterative processing coordination**

Unlike single-pass algorithms, this establishes iterative refinement with careful buffer management:

- **Iteration 0**: Read from buffer_A (initialized with input), write to buffer_B
- **Iteration 1**: Read from buffer_B (previous results), write to buffer_A
- **Iteration 2**: Read from buffer_A (previous results), write to buffer_B
- **Continue alternating**: Each iteration refines results from previous iteration

### **Memory barrier API usage**

Understanding the mbarrier coordination pattern:

- **[mbarrier_init()](https://docs.modular.com/mojo/stdlib/gpu/sync/#mbarrier_init)**: Initialize barrier for specific thread count (TPB)
- **[mbarrier_arrive()](https://docs.modular.com/mojo/stdlib/gpu/sync/#mbarrier_arrive)**: Signal individual thread completion of write phase
- **[mbarrier_test_wait()](https://docs.modular.com/mojo/stdlib/gpu/sync/#mbarrier_test_wait)**: Block until all threads signal completion
- **Reinitialization**: Reset barrier state between iterations for reuse

**Critical timing sequence:**

1. **All threads write**: Each thread updates its assigned buffer element
2. **Signal completion**: Each thread calls [`mbarrier_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/sync/#mbarrier_arrive)
3. **Wait for all**: All threads call [`mbarrier_test_wait()`](https://docs.modular.com/mojo/stdlib/gpu/sync/#mbarrier_test_wait)
4. **Safe to proceed**: Now safe to swap buffer roles for next iteration

### **Stencil operation mechanics**

The 3-point stencil operation with adaptive boundary handling:

**Interior elements**(indices 1 to 254):

```mojo
# Average with left, center, and right neighbors
stencil_sum = buffer[i-1] + buffer[i] + buffer[i+1]
result[i] = stencil_sum / 3.0
```

**Boundary elements**(indices 0 and 255):

```mojo
# Only include valid neighbors in average
stencil_count = 0
for neighbor in valid_neighbors:
    stencil_sum += buffer[neighbor]
    stencil_count += 1
result[i] = stencil_sum / stencil_count
```

### **Buffer role alternation**

The ping-pong buffer pattern ensures data integrity:

**Even iterations**(0, 2, 4...):

- **Read source**: buffer_A contains current data
- **Write target**: buffer_B receives updated results
- **Memory flow**: buffer_A  stencil operation  buffer_B

**Odd iterations**(1, 3, 5...):

- **Read source**: buffer_B contains current data
- **Write target**: buffer_A receives updated results
- **Memory flow**: buffer_B  stencil operation  buffer_A

### **Race condition prevention**

Memory barriers eliminate multiple categories of race conditions:

**Without barriers (broken)**:

```mojo
# Thread A writes to buffer_B[10]
buffer_B[10] = stencil_result_A

# Thread B immediately reads buffer_B[10] for its stencil
# RACE CONDITION: Thread B might read old value before Thread A's write completes
stencil_input = buffer_B[10]  // Undefined behavior!
```

**With barriers (correct)**:

```mojo
# All threads write their results
buffer_B[local_i] = stencil_result

# Signal write completion
mbarrier_arrive(barrier)

# Wait for ALL threads to complete writes
mbarrier_test_wait(barrier, TPB)

# Now safe to read - all writes guaranteed complete
stencil_input = buffer_B[neighbor_index]  // Always sees correct values
```

### **Output buffer selection**

Final result location depends on iteration parity:

**Mathematical determination**:

- **STENCIL_ITERATIONS = 3**(odd number)
- **Final active buffer**: Iteration 2 writes to buffer_B
- **Output source**: Copy from buffer_B to global memory

**Implementation pattern**:

```mojo
@parameter
if STENCIL_ITERATIONS % 2 == 0:
    # Even total iterations end in buffer_A
    output[global_i] = buffer_A[local_i]
else:
    # Odd total iterations end in buffer_B
    output[global_i] = buffer_B[local_i]
```

### **Performance characteristics**

**Memory hierarchy optimization:**

- **Global memory**: Accessed only for input loading and final output
- **Shared memory**: All iterative processing uses fast shared memory
- **Register usage**: Minimal due to shared memory focus

**Synchronization overhead:**

- **mbarrier cost**: Higher than basic barrier() but provides essential control
- **Iteration scaling**: Overhead increases linearly with iteration count
- **Thread efficiency**: All threads remain active throughout processing

### **Real-world applications**

This double-buffering pattern is fundamental to:

**Iterative solvers:**

- Gauss-Seidel and Jacobi methods for linear systems
- Iterative refinement for numerical accuracy
- Multigrid methods with level-by-level processing

**Image processing:**

- Multi-pass filters (bilateral, guided, edge-preserving)
- Iterative denoising algorithms
- Heat diffusion and anisotropic smoothing

**Simulation algorithms:**

- Cellular automata with state evolution
- Particle systems with position updates
- Fluid dynamics with iterative pressure solving

### **Key technical insights**

**Memory barrier philosophy:**

- **Explicit control**: Precise timing control over memory operations vs automatic synchronization
- **Race prevention**: Essential for any algorithm with alternating read/write patterns
- **Performance trade-off**: Higher synchronization cost for guaranteed correctness

**Double-buffering benefits:**

- **Data integrity**: Eliminates read-while-write hazards
- **Algorithm clarity**: Clean separation between current and next iteration state
- **Memory efficiency**: No need for global memory intermediate storage

**Iteration management:**

- **Compile-time unrolling**: `@parameter for` enables optimization opportunities
- **State tracking**: Buffer role alternation must be deterministic
- **Boundary handling**: Adaptive stencil operations handle edge cases gracefully

This solution demonstrates how to design iterative GPU algorithms that require precise memory access control, moving beyond simple parallel loops to sophisticated memory management patterns used in production numerical software.
