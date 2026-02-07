---
title: "GPU Cluster Programming (SM90+)"
description: "Building on your journey from warp-level programming (Puzzles 24-26) through block-level programming (Puzzle 27), you'll now learn cluster-level programming - coordinating multiple thread blocks to solve problems that exceed single-block capabilities."
---

# GPU Cluster Programming (SM90+)

Building on your journey from warp-level programming (Puzzles 24-26) through block-level programming (Puzzle 27), you'll now learn cluster-level programming - coordinating multiple thread blocks to solve problems that exceed single-block capabilities.

## Introduction

> **Hardware requirement:  NVIDIA SM90+ Only**
>
> This puzzle requires **NVIDIA Hopper architecture**(H100, H200) or newer GPUs with SM90+ compute capability. The cluster programming APIs are hardware-accelerated and will raise errors on unsupported hardware. If you're unsure about the underlying architecture, run `pixi run gpu-specs` and must have at least `Compute Cap: 9.0` (see GPU profiling basics for hardware identification)

Building on your journey from **warp-level programming (Puzzles 24-26)**through **block-level programming (Puzzle 27)**, you'll now learn **cluster-level programming**- coordinating multiple thread blocks to solve problems that exceed single-block capabilities.

## What are thread block clusters?

Thread Block Clusters are a revolutionary SM90+ feature that enable **multiple thread blocks to cooperate**on a single computational task with hardware-accelerated synchronization and communication primitives.

**Key capabilities:**
- **Inter-block synchronization**: Coordinate multiple blocks with [`cluster_sync`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_sync), [`cluster_arrive`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive), [`cluster_wait`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait)
- **Block identification**: Use [`block_rank_in_cluster`](https://docs.modular.com/mojo/stdlib/gpu/cluster/block_rank_in_cluster) for unique block coordination
- **Efficient coordination**: [`elect_one_sync`](https://docs.modular.com/mojo/stdlib/gpu/cluster/elect_one_sync) for optimized warp-level cooperation
- **Advanced patterns**: [`cluster_mask_base`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_mask_base) for selective block coordination

## The cluster programming model

### Traditional GPU programming hierarchy:

```
Grid (Multiple Blocks)
|-- Block (Multiple Warps) - barrier() synchronization
    |-- Warp (32 Threads) - SIMT lockstep execution
    |   |-- Lane 0  -+
    |   |-- Lane 1   | All execute same instruction
    |   |-- Lane 2   | at same time (SIMT)
    |   |   ...      | warp.sum(), warp.broadcast()
    |   `-- Lane 31 -+
        `-- Thread (SIMD operations within each thread)
```

### **New: Cluster programming hierarchy:**
```
Grid (Multiple Clusters)
|-- NEW Cluster (Multiple Blocks) - cluster_sync(), cluster_arrive()
    |-- Block (Multiple Warps) - barrier() synchronization
        |-- Warp (32 Threads) - SIMT lockstep execution
        |   |-- Lane 0  -+
        |   |-- Lane 1   | All execute same instruction
        |   |-- Lane 2   | at same time (SIMT)
        |   |   ...      | warp.sum(), warp.broadcast()
        |   `-- Lane 31 -+
            `-- Thread (SIMD operations within each thread)
```

**Execution Model Details:**
- **Thread Level**: SIMD operations within individual threads
- **Warp Level**: SIMT execution - 32 threads in lockstep coordination
- **Block Level**: Multi-warp coordination with shared memory and barriers
- ** Cluster Level**: Multi-block coordination with SM90+ cluster APIs

## Learning progression

This puzzle follows a carefully designed **3-part progression**that builds your cluster programming expertise:

### **[ Multi-Block Coordination Basics](#multi-block-coordination-basics)**

**Focus**: Understanding fundamental cluster synchronization patterns

Learn how multiple thread blocks coordinate their execution using [`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive) and [`cluster_wait()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait) for basic inter-block communication and data distribution.

**Key APIs**: [`block_rank_in_cluster()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/block_rank_in_cluster), [`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive), [`cluster_wait()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait)

---

### **[ Cluster-Wide Collective Operations](#cluster-wide-collective-operations)**

**Focus**: Extending block-level patterns to cluster scale

Learn cluster-wide reductions and collective operations that extend familiar `block.sum()` concepts to coordinate across multiple thread blocks for large-scale computations.

**Key APIs**: [`cluster_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_sync), [`elect_one_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/elect_one_sync) for efficient cluster coordination

---

### **[ Advanced Cluster Algorithms](#advanced-cluster-algorithms)**

**Focus**: Production-ready multi-level coordination patterns

Implement sophisticated algorithms combining warp-level, block-level, and cluster-level coordination for maximum GPU utilization and complex computational workflows.

**Key APIs**: [`elect_one_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/elect_one_sync), [`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive), advanced coordination patterns

## Why cluster programming matters

**Problem Scale**: Modern AI and scientific workloads often require computations that exceed single thread block capabilities:
- **Large matrix operations**requiring inter-block coordination (like matrix multiplication from Puzzle 16)
- **Multi-stage algorithms**with producer-consumer dependencies from Puzzle 29
- **Global statistics**across datasets larger than shared memory from Puzzle 8
- **Advanced stencil computations**requiring neighbor block communication

**Hardware Evolution**: As GPUs gain more compute units (see GPU architecture profiling in Puzzle 30), **cluster programming becomes essential**for utilizing next-generation hardware efficiently.

## Educational value

By completing this puzzle, you'll have learned the complete **GPU programming hierarchy**:

- **Thread-level**: Individual computation units with SIMD operations
- **Warp-level**: 32-thread SIMT coordination (Puzzles 24-26)
- **Block-level**: Multi-warp coordination with shared memory (Puzzle 27)
- ** Cluster-level**: Multi-block coordination (Puzzle 34)
- **Grid-level**: Independent block execution across multiple streaming multiprocessors

This progression prepares you for **next-generation GPU programming**and **large-scale parallel computing**challenges, building on the performance optimization techniques from Puzzles 30-32.

## Getting started

**Prerequisites**:
- Complete understanding of block-level programming (Puzzle 27)
- Experience with warp-level programming (Puzzles 24-26)
- Familiarity with GPU memory hierarchy from shared memory concepts (Puzzle 8)
- Understanding of GPU synchronization from barriers (Puzzle 29)
- Access to NVIDIA SM90+ hardware or compatible environment

**Recommended approach**: Follow the 3-part progression sequentially, as each part builds essential concepts for the next level of complexity.

**Hardware note**: If running on non-SM90+ hardware, the puzzles serve as **educational examples**of cluster programming concepts and API usage patterns.

Ready to learn the future of GPU programming? Start with **[Multi-Block Coordination Basics](#multi-block-coordination-basics)**to learn fundamental cluster synchronization patterns!

## Multi-Block Coordination Basics

### Overview

Welcome to your first **cluster programming challenge**! This section introduces the fundamental building blocks of inter-block coordination using SM90+ cluster APIs.

**Example scenario**: Implement a multi-block histogram algorithm where **4 thread blocks coordinate**to process different ranges of data and store results in a shared output array.

**Key Learning**: Learn the essential cluster synchronization pattern: [`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive)  process  [`cluster_wait()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait), extending the synchronization concepts from barrier() in Puzzle 29.

### The problem: multi-block histogram binning

Traditional single-block algorithms like those in Puzzle 27 can only process data that fits within one block's thread capacity (e.g., 256 threads). For larger datasets exceeding shared memory capacity from Puzzle 8, we need **multiple blocks to cooperate**.

**Example goal**: Implement a histogram where each of 4 blocks processes a different data range, scales values by its unique block rank, and coordinates with other blocks using synchronization patterns from Puzzle 29 to ensure all processing completes before any block reads the final results.

#### Problem specification

**Multi-Block Data Distribution:**

- **Block 0**: Processes elements 0-255, scales by 1
- **Block 1**: Processes elements 256-511, scales by 2
- **Block 2**: Processes elements 512-767, scales by 3
- **Block 3**: Processes elements 768-1023, scales by 4

**Coordination Requirements:**

1. Each block must signal completion using [`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive)
2. All blocks must wait for others using [`cluster_wait()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait)
3. Final output shows each block's processed sum in a 4-element array

### Configuration

- **Problem Size**: `SIZE = 1024` elements (1D array)
- **Block Configuration**: `TPB = 256` threads per block `(256, 1)`
- **Grid Configuration**: `CLUSTER_SIZE = 4` blocks per cluster `(4, 1)`
- **Data Type**: `DType.float32`
- **Memory Layout**: Input `Layout.row_major(SIZE)`, Output `Layout.row_major(CLUSTER_SIZE)`

**Thread Block Distribution:**

- Block 0: threads 0-255  elements 0-255
- Block 1: threads 0-255  elements 256-511
- Block 2: threads 0-255  elements 512-767
- Block 3: threads 0-255  elements 768-1023

### Running the code

  
    pixi NVIDIA (default)
    uv
  
  

```bash
pixi run p34 --coordination
```

  
  

```bash
uv run poe p34 --coordination
```

  

**Expected Output:**

```
Testing Multi-Block Coordination
SIZE: 1024 TPB: 256 CLUSTER_SIZE: 4
Block coordination results:
  Block 0 : 127.5
  Block 1 : 255.0
  Block 2 : 382.5
  Block 3 : 510.0
OK Multi-block coordination tests passed!
```

**Success Criteria:**

- All 4 blocks produce **non-zero results**
- Results show **scaling pattern**: Block 1 > Block 0, Block 2 > Block 1, etc.
- No race conditions or coordination failures

### Reference implementation (example)


```mojo
fn cluster_coordination_basics
    in_layout: Layout, out_layout: Layout, tpb: Int
:
    """Real cluster coordination using SM90+ cluster APIs."""
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    local_i = thread_idx.x

    # Check what's happening with cluster ranks
    my_block_rank = Int(block_rank_in_cluster())
    block_id = Int(block_idx.x)

    shared_data = LayoutTensor[
        dtype,
        Layout.row_major(tpb),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Each block should process different portions of the data
    var data_scale = Float32(
        block_id + 1
    )  # Use block_idx instead of cluster rank

    # Phase 1: Each block processes its portion
    if global_i < size:
        shared_data[local_i] = input[global_i] * data_scale
    else:
        shared_data[local_i] = 0.0

    barrier()

    # Phase 2: Use cluster_arrive() for inter-block coordination
    cluster_arrive()  # Signal this block has completed processing

    # Block-level aggregation (only thread 0)
    if local_i == 0:
        var block_sum: Float32 = 0.0
        for i in range(tpb):
            block_sum += shared_data[i][0]
        output[block_id] = block_sum

    # Wait for all blocks in cluster to complete
    cluster_wait()

```

**The cluster coordination solution demonstrates the fundamental multi-block synchronization pattern using a carefully orchestrated two-phase approach:**

### **Phase 1: Independent block processing**

**Thread and block identification:**

```mojo
global_i = block_dim.x * block_idx.x + thread_idx.x  # Global thread index
local_i = thread_idx.x                               # Local thread index within block
my_block_rank = Int(block_rank_in_cluster())         # Cluster rank (0-3)
block_id = Int(block_idx.x)                          # Block index for reliable addressing
```

**Shared memory allocation and data processing:**

- Each block allocates its own shared memory workspace: `LayoutTensor[dtype, Layout.row_major(tpb), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()`
- **Scaling strategy**: `data_scale = Float32(block_id + 1)` ensures each block processes data differently
  - Block 0: multiplies by 1.0, Block 1: by 2.0, Block 2: by 3.0, Block 3: by 4.0
- **Bounds checking**: `if global_i < size:` prevents out-of-bounds memory access
- **Data processing**: `shared_data[local_i] = input[global_i] * data_scale` scales input data per block

**Intra-block synchronization:**

- `barrier()` ensures all threads within each block complete data loading before proceeding
- This prevents race conditions between data loading and subsequent cluster coordination

### **Phase 2: Cluster coordination**

**Inter-block signaling:**

- [`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive) signals that this block has completed its local processing phase
- This is a **non-blocking**operation that registers completion with the cluster hardware

**Local aggregation (Thread 0 only):**

```mojo
if local_i == 0:
    var block_sum: Float32 = 0.0
    for i in range(tpb):
        block_sum += shared_data[i][0]  # Sum all elements in shared memory
    output[block_id] = block_sum        # Store result at unique block position
```

- Only thread 0 performs the sum to avoid race conditions
- Results stored at `output[block_id]` ensures each block writes to unique location

**Final synchronization:**

- [`cluster_wait()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait) blocks until ALL blocks in the cluster have completed their work
- This ensures deterministic completion order across the entire cluster

### **Key technical insights**

**Why use `block_id` instead of `my_block_rank`?**

- `block_idx.x` provides reliable grid-launch indexing (0, 1, 2, 3)
- [`block_rank_in_cluster()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/block_rank_in_cluster) may behave differently depending on cluster configuration
- Using `block_id` guarantees each block gets unique data portions and output positions

**Memory access pattern:**

- **Global memory**: Each thread reads `input[global_i]` exactly once
- **Shared memory**: Used for intra-block communication and aggregation
- **Output memory**: Each block writes to `output[block_id]` exactly once

**Synchronization hierarchy:**

1. **`barrier()`**: Synchronizes threads within each block (intra-block)
2. **[`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive)**: Signals completion to other blocks (inter-block, non-blocking)
3. **[`cluster_wait()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait)**: Waits for all blocks to complete (inter-block, blocking)

**Performance characteristics:**

- **Compute complexity**: O(TPB) per block for local sum, O(1) for cluster coordination
- **Memory bandwidth**: Each input element read once, minimal inter-block communication
- **Scalability**: Pattern scales to larger cluster sizes with minimal overhead

### Understanding the pattern

The essential cluster coordination pattern follows a simple but powerful structure:

1. **Phase 1**: Each block processes its assigned data portion independently
2. **Signal**: [`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive) announces completion of processing
3. **Phase 2**: Blocks can safely perform operations that depend on other blocks' results
4. **Synchronize**: [`cluster_wait()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait) ensures all blocks finish before proceeding

##  Cluster-Wide Collective Operations

### Overview

Building on basic cluster coordination from the previous section, this challenge teaches you to implement **cluster-wide collective operations**- extending the familiar [`block.sum`](https://docs.modular.com/mojo/stdlib/gpu/primitives/block/sum) pattern from Puzzle 27 to coordinate across **multiple thread blocks**.

**Example scenario**: Implement a cluster-wide reduction that processes 1024 elements across 4 coordinated blocks, combining their individual reductions into a single global result.

**Key Learning**: Learn [`cluster_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_sync) for full cluster coordination and [`elect_one_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/elect_one_sync) for efficient final reductions.

### The problem: large-scale global sum

Single blocks (as learned in Puzzle 27) are limited by their thread count and shared memory capacity from Puzzle 8. For **large datasets**requiring global statistics (mean, variance, sum) beyond single-block reductions, we need **cluster-wide collective operations**.

**Example goal**: Implement a cluster-wide sum reduction where:

1. Each block performs local reduction (like `block.sum()` from Puzzle 27)
2. Blocks coordinate to combine their partial results using synchronization from Puzzle 29
3. One elected thread computes the final global sum using warp election patterns

#### Problem specification

**Algorithmic Flow:**

**Phase 1 - Local Reduction (within each block):**
\\[R_i = \sum_{j=0}^{TPB-1} input[i \times TPB + j] \quad \text{for block } i\\]

**Phase 2 - Global Aggregation (across cluster):**
\\[\text{Global Sum} = \sum_{i=0}^{\text{CLUSTER_SIZE}-1} R_i\\]

**Coordination Requirements:**

1. **Local reduction**: Each block computes partial sum using tree reduction
2. **Cluster sync**: [`cluster_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_sync) ensures all partial results are ready
3. **Final aggregation**: One elected thread combines all partial results

### Configuration

- **Problem Size**: `SIZE = 1024` elements
- **Block Configuration**: `TPB = 256` threads per block `(256, 1)`
- **Grid Configuration**: `CLUSTER_SIZE = 4` blocks per cluster `(4, 1)`
- **Data Type**: `DType.float32`
- **Memory Layout**: Input `Layout.row_major(SIZE)`, Output `Layout.row_major(1)`
- **Temporary Storage**: `Layout.row_major(CLUSTER_SIZE)` for partial results

**Expected Result**: Sum of sequence `0, 0.01, 0.02, ..., 10.23` = **523,776**

### Cluster APIs reference

**From [`gpu.primitives.cluster`](https://docs.modular.com/mojo/stdlib/gpu/primitives/cluster/) module:**

- **[`cluster_sync()`](https://docs.modular.com/mojo/stdlib/gpu/primitives/cluster/cluster_sync)**: Full cluster synchronization - stronger than arrive/wait pattern
- **[`elect_one_sync()`](https://docs.modular.com/mojo/stdlib/gpu/primitives/cluster/elect_one_sync)**: Elects single thread within warp for efficient coordination
- **[`block_rank_in_cluster()`](https://docs.modular.com/mojo/stdlib/gpu/primitives/cluster/block_rank_in_cluster)**: Returns unique block identifier within cluster

### Tree reduction pattern

Recall the **tree reduction pattern**from Puzzle 27's traditional dot product:

```
Stride 128: [T0] += [T128], [T1] += [T129], [T2] += [T130], ...
Stride 64:  [T0] += [T64],  [T1] += [T65],  [T2] += [T66],  ...
Stride 32:  [T0] += [T32],  [T1] += [T33],  [T2] += [T34],  ...
Stride 16:  [T0] += [T16],  [T1] += [T17],  [T2] += [T18],  ...
...
Stride 1:   [T0] += [T1] -> Final result at T0
```

**Now extend this pattern to cluster scale**where each block produces one partial result, then combine across blocks.

### Running the code

  
    pixi NVIDIA (default)
    uv
  
  

```bash
pixi run p34 --reduction
```

  
  

```bash
uv run poe p34 --reduction
```

  

**Expected Output:**

```
Testing Cluster-Wide Reduction
SIZE: 1024 TPB: 256 CLUSTER_SIZE: 4
Expected sum: 523776.0
Cluster reduction result: 523776.0
Expected: 523776.0
Error: 0.0
OK Passed: Cluster reduction accuracy test
OK Cluster-wide collective operations tests passed!
```

**Success Criteria:**

- **Perfect accuracy**: Result exactly matches expected sum (523,776)
- **Cluster coordination**: All 4 blocks contribute their partial sums
- **Efficient final reduction**: Single elected thread computes final result

### Reference implementation (example)


```mojo
fn cluster_collective_operations
    in_layout: Layout, out_layout: Layout, tpb: Int
, MutAnyOrigin
    ],
    size: Int,
):
    """Cluster-wide collective operations using real cluster APIs."""
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    local_i = Int(thread_idx.x)
    my_block_rank = Int(block_rank_in_cluster())
    block_id = Int(block_idx.x)

    # Each thread accumulates its data
    var my_value: Float32 = 0.0
    if global_i < size:
        my_value = input[global_i][0]

    # Block-level reduction using shared memory
    shared_mem = LayoutTensor[
        dtype,
        Layout.row_major(tpb),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    shared_mem[local_i] = my_value
    barrier()

    # Tree reduction within block
    var stride = tpb // 2
    while stride > 0:
        if local_i < stride and local_i + stride < tpb:
            shared_mem[local_i] += shared_mem[local_i + stride]
        barrier()
        stride = stride // 2

    if local_i == 0:
        temp_storage[block_id] = shared_mem[0]

    # Use cluster_sync() for full cluster synchronization
    cluster_sync()

    # Final cluster reduction (elect one thread to do the final work)
    if elect_one_sync() and my_block_rank == 0:
        var total: Float32 = 0.0
        for i in range(CLUSTER_SIZE):
            total += temp_storage[i][0]
        output[0] = total

```

**The cluster collective operations solution demonstrates the classic distributed computing pattern: local reduction  global coordination  final aggregation:**

### **Phase 1: Local block reduction (traditional tree reduction)**

**Data loading and initialization:**

```mojo
var my_value: Float32 = 0.0
if global_i < size:
    my_value = input[global_i][0]  # Load with bounds checking
shared_mem[local_i] = my_value     # Store in shared memory
barrier()                          # Ensure all threads complete loading
```

**Tree reduction algorithm:**

```mojo
var stride = tpb // 2  # Start with half the threads (128)
while stride > 0:
    if local_i < stride and local_i + stride < tpb:
        shared_mem[local_i] += shared_mem[local_i + stride]
    barrier()          # Synchronize after each reduction step
    stride = stride // 2
```

**Tree reduction visualization (TPB=256):**

```
Step 1: stride=128  [T0]+=T128, [T1]+=T129, ..., [T127]+=T255
Step 2: stride=64   [T0]+=T64,  [T1]+=T65,  ..., [T63]+=T127
Step 3: stride=32   [T0]+=T32,  [T1]+=T33,  ..., [T31]+=T63
Step 4: stride=16   [T0]+=T16,  [T1]+=T17,  ..., [T15]+=T31
Step 5: stride=8    [T0]+=T8,   [T1]+=T9,   ..., [T7]+=T15
Step 6: stride=4    [T0]+=T4,   [T1]+=T5,   [T2]+=T6,  [T3]+=T7
Step 7: stride=2    [T0]+=T2,   [T1]+=T3
Step 8: stride=1    [T0]+=T1    -> Final result at shared_mem[0]
```

**Partial result storage:**

- Only thread 0 writes: `temp_storage[block_id] = shared_mem[0]`
- Each block stores its sum at `temp_storage[0]`, `temp_storage[1]`, `temp_storage[2]`, `temp_storage[3]`

### **Phase 2: Cluster synchronization**

**Full cluster barrier:**

- [`cluster_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_sync) provides **stronger guarantees**than [`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive)/[`cluster_wait()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait)
- Ensures **all blocks complete their local reductions**before any block proceeds
- Hardware-accelerated synchronization across all blocks in the cluster

### **Phase 3: Final global aggregation**

**Thread election for efficiency:**

```mojo
if elect_one_sync() and my_block_rank == 0:
    var total: Float32 = 0.0
    for i in range(CLUSTER_SIZE):
        total += temp_storage[i][0]  # Sum: temp[0] + temp[1] + temp[2] + temp[3]
    output[0] = total
```

**Why this election strategy?**

- **[`elect_one_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/elect_one_sync)**: Hardware primitive that selects exactly one thread per warp
- **`my_block_rank == 0`**: Only elect from the first block to ensure single writer
- **Result**: Only ONE thread across the entire cluster performs the final summation
- **Efficiency**: Avoids redundant computation across all 1024 threads

### **Key technical insights**

**Three-level reduction hierarchy:**

1. **Thread  Warp**: Individual threads contribute to warp-level partial sums
2. **Warp  Block**: Tree reduction combines warps into single block result (256  1)
3. **Block  Cluster**: Simple loop combines block results into final sum (4  1)

**Memory access patterns:**

- **Input**: Each element read exactly once (`input[global_i]`)
- **Shared memory**: High-speed workspace for intra-block tree reduction
- **Temp storage**: Low-overhead inter-block communication (only 4 values)
- **Output**: Single global result written once

**Synchronization guarantees:**

- **`barrier()`**: Ensures all threads in block complete each tree reduction step
- **[`cluster_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_sync)**: **Global barrier**- all blocks reach same execution point
- **Single writer**: Election prevents race conditions on final output

**Algorithm complexity analysis:**

- **Tree reduction**: O(log TPB) = O(log 256) = 8 steps per block
- **Cluster coordination**: O(1) synchronization overhead
- **Final aggregation**: O(CLUSTER_SIZE) = O(4) simple additions
- **Total**: Logarithmic within blocks, linear across blocks

**Scalability characteristics:**

- **Block level**: Scales to thousands of threads with logarithmic complexity
- **Cluster level**: Scales to dozens of blocks with linear complexity
- **Memory**: Temp storage requirements scale linearly with cluster size
- **Communication**: Minimal inter-block data movement (one value per block)

### Understanding the collective pattern

This puzzle demonstrates the classic **two-phase reduction pattern**used in distributed computing:

1. **Local aggregation**: Each processing unit (block) reduces its data portion
2. **Global coordination**: Processing units synchronize and exchange results
3. **Final reduction**: One elected unit combines all partial results

**Comparison to single-block approaches:**

- **Traditional `block.sum()`**: Works within 256 threads maximum
- **Cluster collective**: Scales to 1000+ threads across multiple blocks
- **Same accuracy**: Both produce identical mathematical results
- **Different scale**: Cluster approach handles larger datasets

**Performance benefits**:

- **Larger datasets**: Process arrays that exceed single-block capacity
- **Better utilization**: Use more GPU compute units simultaneously
- **Scalable patterns**: Foundation for complex multi-stage algorithms

##  Advanced Cluster Algorithms

### Overview

This final challenge combines **all levels of GPU programming hierarchy**from warp-level (Puzzles 24-26), block-level (Puzzle 27), and cluster coordination - to implement a sophisticated multi-level algorithm that maximizes GPU utilization.

**Example scenario**: Implement a hierarchical cluster algorithm using **warp-level optimization**(`elect_one_sync()`), **block-level aggregation**, and **cluster-level coordination**in a single unified pattern.

**Key Learning**: Learn the complete GPU programming stack with production-ready coordination patterns used in advanced computational workloads.

### The problem: multi-level data processing pipeline

Real-world GPU algorithms often require **hierarchical coordination**where different levels of the GPU hierarchy (warps from Puzzle 24, blocks from Puzzle 27, clusters) perform specialized roles in a coordinated computation pipeline, extending multi-stage processing from Puzzle 29.

**Example goal**: Implement a multi-stage algorithm where:

1. **Warp-level**: Use [`elect_one_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/elect_one_sync) for efficient intra-warp coordination (from SIMT execution)
2. **Block-level**: Aggregate warp results using shared memory coordination
3. **Cluster-level**: Coordinate between blocks using [`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive) / [`cluster_wait()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait) staged synchronization from Puzzle 29

#### Algorithm specification

**Multi-Stage Processing Pipeline:**

1. **Stage 1 (Warp-level)**: Each warp elects one thread to sum 32 consecutive elements
2. **Stage 2 (Block-level)**: Aggregate all warp sums within each block
3. **Stage 3 (Cluster-level)**: Coordinate between blocks with [`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive) / [`cluster_wait()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait)

**Input**: 1024 float values with pattern `(i % 50) * 0.02` for testing
**Output**: 4 block results showing hierarchical processing effects

### Configuration

- **Problem Size**: `SIZE = 1024` elements
- **Block Configuration**: `TPB = 256` threads per block `(256, 1)`
- **Grid Configuration**: `CLUSTER_SIZE = 4` blocks `(4, 1)`
- **Warp Size**: `WARP_SIZE = 32` threads per warp (NVIDIA standard)
- **Warps per Block**: `TPB / WARP_SIZE = 8` warps
- **Data Type**: `DType.float32`
- **Memory Layout**: Input `Layout.row_major(SIZE)`, Output `Layout.row_major(CLUSTER_SIZE)`

**Processing Distribution:**

- **Block 0**: 256 threads  8 warps  elements 0-255
- **Block 1**: 256 threads  8 warps  elements 256-511
- **Block 2**: 256 threads  8 warps  elements 512-767
- **Block 3**: 256 threads  8 warps  elements 768-1023

### Advanced cluster APIs

**From [`gpu.primitives.cluster`](https://docs.modular.com/mojo/stdlib/gpu/cluster/) module:**

- **[`elect_one_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/elect_one_sync)**: Warp-level thread election for efficient computation
- **[`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive)**: Signal completion for staged cluster coordination
- **[`cluster_wait()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait)**: Wait for all blocks to reach synchronization point
- **[`block_rank_in_cluster()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/block_rank_in_cluster)**: Get unique block identifier within cluster

### Hierarchical coordination pattern

This puzzle demonstrates **three-level coordination hierarchy**:

#### **Level 1: Warp Coordination**(Puzzle 24)

```
Warp (32 threads) -> elect_one_sync() -> 1 elected thread -> processes 32 elements
```

#### **Level 2: Block Coordination**(Puzzle 27)

```
Block (8 warps) -> aggregate warp results -> 1 block total
```

#### **Level 3: Cluster Coordination**(This puzzle)

```
Cluster (4 blocks) -> cluster_arrive/wait -> synchronized completion
```

**Combined Effect:**1024 threads  32 warp leaders  4 block results  coordinated cluster completion

### Running the code

  
    pixi NVIDIA (default)
    uv
  
  

```bash
pixi run p34 --advanced
```

  
  

```bash
uv run poe p34 --advanced
```

  

**Expected Output:**

```
Testing Advanced Cluster Algorithms
SIZE: 1024 TPB: 256 CLUSTER_SIZE: 4
Advanced cluster algorithm results:
  Block 0 : 122.799995
  Block 1 : 247.04001
  Block 2 : 372.72
  Block 3 : 499.83997
OK Advanced cluster patterns tests passed!
```

**Success Criteria:**

- **Hierarchical scaling**: Results show multi-level coordination effects
- **Warp optimization**: `elect_one_sync()` reduces redundant computation
- **Cluster coordination**: All blocks complete processing successfully
- **Performance pattern**: Higher block IDs produce proportionally larger results

### Reference implementation (example)


```mojo
fn advanced_cluster_patterns
    in_layout: Layout, out_layout: Layout, tpb: Int
:
    """Advanced cluster programming using cluster masks and relaxed synchronization.
    """
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    local_i = Int(thread_idx.x)
    my_block_rank = Int(block_rank_in_cluster())
    block_id = Int(block_idx.x)

    shared_data = LayoutTensor[
        dtype,
        Layout.row_major(tpb),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Compute cluster mask for advanced coordination
    # base_mask = cluster_mask_base()  # Requires cluster_shape parameter

    var data_scale = Float32(block_id + 1)
    if global_i < size:
        shared_data[local_i] = input[global_i] * data_scale
    else:
        shared_data[local_i] = 0.0

    barrier()

    # Advanced pattern: Use elect_one_sync for efficient coordination
    if elect_one_sync():  # Only one thread per warp does this work
        var warp_sum: Float32 = 0.0
        var warp_start = (local_i // 32) * 32  # Get warp start index
        for i in range(32):  # Sum across warp
            if warp_start + i < tpb:
                warp_sum += shared_data[warp_start + i][0]
        shared_data[local_i] = warp_sum

    barrier()

    # Use cluster_arrive for staged synchronization in sm90+
    cluster_arrive()

    # Only first thread in each block stores result
    if local_i == 0:
        var block_total: Float32 = 0.0
        for i in range(0, tpb, 32):  # Sum warp results
            if i < tpb:
                block_total += shared_data[i][0]
        output[block_id] = block_total

    # Wait for all blocks to complete their calculations in sm90+
    cluster_wait()

```

**The advanced cluster patterns solution demonstrates a sophisticated three-level hierarchical optimization that combines warp, block, and cluster coordination for maximum GPU utilization:**

### **Level 1: Warp-Level Optimization (Thread Election)**

**Data preparation and scaling:**

```mojo
var data_scale = Float32(block_id + 1)  # Block-specific scaling factor
if global_i < size:
    shared_data[local_i] = input[global_i] * data_scale
else:
    shared_data[local_i] = 0.0  # Zero-pad for out-of-bounds
barrier()  # Ensure all threads complete data loading
```

**Warp-level thread election:**

```mojo
if elect_one_sync():  # Hardware elects exactly 1 thread per warp
    var warp_sum: Float32 = 0.0
    var warp_start = (local_i // 32) * 32  # Calculate warp boundary
    for i in range(32):  # Process entire warp's data
        if warp_start + i < tpb:
            warp_sum += shared_data[warp_start + i][0]
    shared_data[local_i] = warp_sum  # Store result at elected thread's position
```

**Warp boundary calculation explained:**

- **Thread 37**(in warp 1): `warp_start = (37 // 32) * 32 = 1 * 32 = 32`
- **Thread 67**(in warp 2): `warp_start = (67 // 32) * 32 = 2 * 32 = 64`
- **Thread 199**(in warp 6): `warp_start = (199 // 32) * 32 = 6 * 32 = 192`

**Election pattern visualization (TPB=256, 8 warps):**

```
Warp 0 (threads 0-31):   elect_one_sync() -> Thread 0   processes elements 0-31
Warp 1 (threads 32-63):  elect_one_sync() -> Thread 32  processes elements 32-63
Warp 2 (threads 64-95):  elect_one_sync() -> Thread 64  processes elements 64-95
Warp 3 (threads 96-127): elect_one_sync() -> Thread 96  processes elements 96-127
Warp 4 (threads 128-159):elect_one_sync() -> Thread 128 processes elements 128-159
Warp 5 (threads 160-191):elect_one_sync() -> Thread 160 processes elements 160-191
Warp 6 (threads 192-223):elect_one_sync() -> Thread 192 processes elements 192-223
Warp 7 (threads 224-255):elect_one_sync() -> Thread 224 processes elements 224-255
```

### **Level 2: Block-level aggregation (Warp Leader Coordination)**

**Inter-warp synchronization:**

```mojo
barrier()  # Ensure all warps complete their elected computations
```

**Warp leader aggregation (Thread 0 only):**

```mojo
if local_i == 0:
    var block_total: Float32 = 0.0
    for i in range(0, tpb, 32):  # Iterate through warp leader positions
        if i < tpb:
            block_total += shared_data[i][0]  # Sum warp results
    output[block_id] = block_total
```

**Memory access pattern:**

- Thread 0 reads from: `shared_data[0]`, `shared_data[32]`, `shared_data[64]`, `shared_data[96]`, `shared_data[128]`, `shared_data[160]`, `shared_data[192]`, `shared_data[224]`
- These positions contain the warp sums computed by elected threads
- Result: 8 warp sums  1 block total

### **Level 3: Cluster-level staged synchronization**

**Staged synchronization approach:**

```mojo
cluster_arrive()  # Non-blocking: signal this block's completion
# ... Thread 0 computes and stores block result ...
cluster_wait()    # Blocking: wait for all blocks to complete
```

**Why staged synchronization?**

- **[`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive)**called **before**final computation allows overlapping work
- Block can compute its result while other blocks are still processing
- **[`cluster_wait()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait)**ensures deterministic completion order
- More efficient than [`cluster_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_sync) for independent block computations

### **Advanced pattern characteristics**

**Hierarchical computation reduction:**

1. **256 threads** **8 elected threads**(32x reduction per block)
2. **8 warp sums** **1 block total**(8x reduction per block)
3. **4 blocks** **staged completion**(synchronized termination)
4. **Total efficiency**: 256x reduction in redundant computation per block

**Memory access optimization:**

- **Level 1**: Coalesced reads from `input[global_i]`, scaled writes to shared memory
- **Level 2**: Elected threads perform warp-level aggregation (8 computations vs 256)
- **Level 3**: Thread 0 performs block-level aggregation (1 computation vs 8)
- **Result**: Minimized memory bandwidth usage through hierarchical reduction

**Synchronization hierarchy:**

1. **`barrier()`**: Intra-block thread synchronization (after data loading and warp processing)
2. **[`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive)**: Inter-block signaling (non-blocking, enables work overlap)
3. **[`cluster_wait()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait)**: Inter-block synchronization (blocking, ensures completion order)

**Why this is "advanced":**

- **Multi-level optimization**: Combines warp, block, and cluster programming techniques
- **Hardware efficiency**: Leverages [`elect_one_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/elect_one_sync) for optimal warp utilization
- **Staged coordination**: Uses advanced cluster APIs for flexible synchronization
- **Production-ready**: Demonstrates patterns used in real-world GPU libraries

**Real-world performance benefits:**

- **Reduced memory pressure**: Fewer threads accessing shared memory simultaneously
- **Better warp utilization**: Elected threads perform focused computation
- **Scalable coordination**: Staged synchronization handles larger cluster sizes
- **Algorithm flexibility**: Foundation for complex multi-stage processing pipelines

**Complexity analysis:**

- **Warp level**: O(32) operations per elected thread = O(256) total per block
- **Block level**: O(8) aggregation operations per block
- **Cluster level**: O(1) synchronization overhead per block
- **Total**: Linear complexity with massive parallelization benefits

### The complete GPU hierarchy

Congratulations! By completing this puzzle, you've learned **the complete GPU programming stack**:

 **Thread-level programming**: Individual execution units
 **Warp-level programming**: 32-thread SIMT coordination
 **Block-level programming**: Multi-warp coordination and shared memory
 ** Cluster-level programming**: Multi-block coordination with SM90+ APIs
 **Coordinate multiple thread blocks**with cluster synchronization primitives
 **Scale algorithms beyond single-block limitations**using cluster APIs
 **Implement hierarchical algorithms**combining warp + block + cluster coordination
 **Utilize next-generation GPU hardware**with SM90+ cluster programming

### Real-world applications

The hierarchical coordination patterns from this puzzle are fundamental to:

**High-Performance Computing:**

- **Multi-grid solvers**: Different levels handle different resolution grids
- **Domain decomposition**: Hierarchical coordination across problem subdomains
- **Parallel iterative methods**: Warp-level local operations, cluster-level global communication

**Deep Learning:**

- **Model parallelism**: Different blocks process different model components
- **Pipeline parallelism**: Staged processing across multiple transformer layers
- **Gradient aggregation**: Hierarchical reduction across distributed training nodes

**Graphics and Visualization:**

- **Multi-pass rendering**: Staged processing for complex visual effects
- **Hierarchical culling**: Different levels cull at different granularities
- **Parallel geometry processing**: Coordinated transformation pipelines

You've now learned the **cutting-edge GPU programming techniques**available on modern hardware!

**Ready for more challenges?**Explore other advanced GPU programming topics, revisit performance optimization techniques from Puzzles 30-32, apply profiling methodologies from NVIDIA tools, or build upon these cluster programming patterns for your own computational workloads!
