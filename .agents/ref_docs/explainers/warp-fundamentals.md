---
title: "Warp Fundamentals"
description: "Understand the fundamental hardware unit of GPU parallelism:"
---

# Warp Fundamentals

Understand the fundamental hardware unit of GPU parallelism:

## Overview

**Part VI: GPU Warp Programming**introduces GPU **warp-level primitives**- hardware-accelerated operations that leverage synchronized thread execution within warps. You'll learn to use built-in warp operations to replace complex shared memory patterns with simple, efficient function calls.

**Goal:**Replace complex shared memory + barrier + tree reduction patterns with efficient warp primitive calls that leverage hardware synchronization.

**Key insight:**_GPU warps execute in lockstep - Mojo's warp operations use this synchronization to provide powerful parallel primitives with zero explicit synchronization._

## What you'll learn

### **GPU warp execution model**

Understand the fundamental hardware unit of GPU parallelism:

```text
GPU Block (e.g., 256 threads)
|-- Warp 0 (32 threads, SIMT lockstep execution)
|   |-- Lane 0  -+
|   |-- Lane 1   | All execute same instruction
|   |-- Lane 2   | at same time (SIMT)
|   |   ...      |
|   `-- Lane 31 -+
|-- Warp 1 (32 threads, independent)
|-- Warp 2 (32 threads, independent)
`-- ...
```

**Hardware reality:**

- **32 threads per warp**on NVIDIA GPUs (`WARP_SIZE=32`)
- **32 or 64 threads per warp**on AMD GPUs (`WARP_SIZE=32 or 64`)
- **Lockstep execution**: All threads in a warp execute the same instruction simultaneously
- **Zero synchronization cost**: Warp operations happen instantly within each warp

### **Warp operations available in Mojo**

Learn the core warp primitives from `gpu.primitives.warp`:

1. **`sum(value)`**: Sum all values across warp lanes
2. **`shuffle_idx(value, lane)`**: Get value from specific lane
3. **`shuffle_down(value, delta)`**: Get value from lane+delta
4. **`prefix_sum(value)`**: Compute prefix sum across lanes
5. **`lane_id()`**: Get current thread's lane number (0-31 or 0-63)

### **Performance transformation example**

```mojo
# 1. Reduction through shared memory
# Complex pattern we have seen earlier (from p12.mojo):
shared = LayoutTensor[
    dtype,
    Layout.row_major(WARP_SIZE),
    MutAnyOrigin,
    address_space = AddressSpace.SHARED,
].stack_allocation()
shared[local_i] = partial_product
barrier()

# Safe tree reduction through shared memory requires a barrier after each reduction phase:
stride = WARP_SIZE // 2
while stride > 0:
    if local_i < stride:
        shared[local_i] += shared[local_i + stride]

    barrier()
    stride //= 2

# 2. Reduction using warp primitives
# Safe tree reduction using warp primitives does not require shared memory or a barrier
# after each reduction phase.
# Mojo's warp-level sum operation uses warp primitives under the hood and hides all this
# complexity:
total = sum(partial_product)  # Internally no barriers, no race conditions!
```

### **When warp operations excel**

Learn the performance characteristics:

```text
Problem Scale         Traditional    Warp Operations
Single warp (32)      Fast          Fastest (no barriers)
Few warps (128)       Good          Excellent (minimal overhead)
Many warps (1024+)    Good          Outstanding (scales linearly)
Massive (16K+)        Bottlenecked  Memory-bandwidth limited
```

## Prerequisites

Before diving into warp programming, ensure you're comfortable with:

- **Part V functional patterns**: Elementwise, tiled, and vectorized approaches
- **GPU thread hierarchy**: Understanding blocks, warps, and threads
- **LayoutTensor operations**: Loading, storing, and tensor manipulation
- **Shared memory concepts**: Why barriers and tree reduction are complex

## Learning path

### **1. SIMT execution model**

** [Warp Lanes & SIMT Execution](#warp-lanes-simt-execution)**

Understand the hardware foundation that makes warp operations possible.

**What you'll learn:**

- Single Instruction, Multiple Thread (SIMT) execution model
- Warp divergence and convergence patterns
- Lane synchronization within warps
- Hardware vs software thread management

**Key insight:**Warps are the fundamental unit of GPU execution - understanding SIMT unlocks warp programming.

### **2. Warp sum fundamentals**

** [warp.sum() Essentials](#warpsum-essentials-warp-level-dot-product)**

Learn the most important warp operation through dot product implementation.

**What you'll learn:**

- Replacing shared memory + barriers with `sum()`
- Cross-GPU architecture compatibility (`WARP_SIZE`)
- Kernel vs functional programming patterns with warps
- Performance comparison with traditional approaches

**Key pattern:**

```mojo
partial_result = compute_per_lane_value()
total = sum(partial_result)  # Magic happens here!
if lane_id() == 0:
    output[0] = total
```

### **3. When to use warp programming**

** [When to Use Warp Programming](#when-to-use-warp-programming)**

Learn the decision framework for choosing warp operations over alternatives.

**What you'll learn:**

- Problem characteristics that favor warp operations
- Performance scaling patterns with warp count
- Memory bandwidth vs computation trade-offs
- Warp operation selection guidelines

**Decision framework:**When reduction operations become the bottleneck, warp primitives often provide the breakthrough.

## Key concepts to learn

### **Hardware-software alignment**

Understanding how Mojo's warp operations map to GPU hardware:

- **SIMT execution**: All lanes execute same instruction simultaneously
- **Built-in synchronization**: No explicit barriers needed within warps
- **Cross-architecture support**: `WARP_SIZE` handles NVIDIA vs AMD differences

### **Pattern transformation**

Converting complex parallel patterns to warp primitives:

- **Tree reduction** `sum()`
- **Prefix computation** `prefix_sum()`
- **Data shuffling** `shuffle_idx()`, `shuffle_down()`

### **Performance characteristics**

Recognizing when warp operations provide advantages:

- **Small to medium problems**: Eliminates barrier overhead
- **Large problems**: Reduces memory traffic and improves cache utilization
- **Regular patterns**: Warp operations excel with predictable access patterns

## Getting started

Start with understanding the SIMT execution model, then dive into practical warp sum implementation, and finish with the strategic decision framework.

 **Success tip**: Think of warps as **synchronized vector units**rather than independent threads. This mental model will guide you toward effective warp programming patterns.

**Learning objective**: By the end of Part VI, you'll recognize when warp operations can replace complex synchronization patterns, enabling you to write simpler, faster GPU code.

**Ready to begin?**Start with **[SIMT Execution Model](#warp-lanes-simt-execution)**and discover the power of warp-level programming!

##  Warp lanes & SIMT execution

### Mental model for warp programming vs SIMD

#### What is a warp?

A **warp**is a group of 32 (or 64) GPU threads that execute **the same instruction at the same time**on different data. Think of it as a **synchronized vector unit**where each thread acts like a "lane" in a vector processor.

**Simple example:**

```mojo
from gpu.primitives.warp import sum
# All 32 threads in the warp execute this simultaneously:
var my_value = input[my_thread_id]     # Each gets different data
var warp_total = sum(my_value)         # All contribute to one sum
```

What just happened? Instead of 32 separate threads doing complex coordination, the **warp**automatically synchronized them to produce a single result. This is **SIMT (Single Instruction, Multiple Thread)**execution.

#### SIMT vs SIMD comparison

If you're familiar with CPU vector programming (SIMD), GPU warps are similar but with key differences:

| Aspect | CPU SIMD (e.g., AVX) | GPU Warp (SIMT) |
|--------|---------------------|------------------|
| **Programming model**| Explicit vector operations | Thread-based programming |
| **Data width**| Fixed (256/512 bits) | Flexible (32/64 threads) |
| **Synchronization**| Implicit within instruction | Implicit within warp |
| **Communication**| Via memory/registers | Via shuffle operations |
| **Divergence handling**| Not applicable | Hardware masking |
| **Example**| `a + b` | `sum(thread_value)` |

**CPU SIMD approach (C++ intrinsics):**

```cpp
// Explicit vector operations - say 8 floats in parallel
__m256 result = _mm256_add_ps(a, b);   // Add 8 pairs simultaneously
```

**CPU SIMD approach (Mojo):**

```mojo
# SIMD in Mojo is first class citizen type so if a, b are of type SIMD then
# addition is performed in parallel
var result = a + b # Add 8 pairs simultaneously
```

**GPU SIMT approach (Mojo):**

```mojo
# Thread-based code that becomes vector operations
from gpu.primitives.warp import sum

var my_data = input[thread_id]         # Each thread gets its element
var partial = my_data * coefficient    # All threads compute simultaneously
var total = sum(partial)               # Hardware coordinates the sum
```

#### Core concepts that make warps powerful

**1. Lane identity:**Each thread has a "lane ID" (0 to 31) that's essentially free to access

```mojo
var my_lane = lane_id()  # Just reading a hardware register
```

**2. Implicit synchronization:**No barriers needed within a warp

```mojo
# This just works - all threads automatically synchronized
var sum = sum(my_contribution)
```

**3. Efficient communication:**Threads can share data without memory

```mojo
# Get value from lane 0 to all other lanes
var broadcasted = shuffle_idx(my_value, 0)
```

**Key insight:**SIMT lets you write natural thread code that executes as efficient vector operations, combining the ease of thread programming with the performance of vector processing.

#### Where warps fit in GPU execution hierarchy

For complete context on how warps relate to the overall GPU execution model, see GPU Threading vs SIMD. Here's where warps fit:

```
GPU Device
|-- Grid (your entire problem)
|   |-- Block 1 (group of threads, shared memory)
|   |   |-- Warp 1 (32 threads, lockstep execution) <- This level
|   |   |   |-- Thread 1 -> SIMD operations
|   |   |   |-- Thread 2 -> SIMD operations
|   |   |   `-- ... (32 threads total)
|   |   `-- Warp 2 (32 threads)
|   `-- Block 2 (independent group)
```

**Warp programming operates at the "Warp level"**- you work with operations that coordinate all 32 threads within a single warp, enabling powerful primitives like `sum()` that would otherwise require complex shared memory coordination.

This mental model supports recognizing when problems map naturally to warp operations versus requiring traditional shared memory approaches.

### The hardware foundation of warp programming

Understanding **Single Instruction, Multiple Thread (SIMT)**execution is crucial for effective warp programming. This isn't just a software abstraction - it's how GPU hardware actually works at the silicon level.

### What is SIMT execution?

**SIMT**means that within a warp, all threads execute the **same instruction**at the **same time**on **different data**. This is fundamentally different from CPU threads, which can execute completely different instructions independently.

#### CPU vs GPU Execution Models

| Aspect | CPU (MIMD) | GPU Warp (SIMT) |
|--------|------------|------------------|
| **Instruction Model**| Multiple Instructions, Multiple Data | Single Instruction, Multiple Thread |
| **Core 1**| `add r1, r2` | `add r1, r2` |
| **Core 2**| `load r3, [mem]` | `add r1, r2` (same instruction) |
| **Core 3**| `branch loop` | `add r1, r2` (same instruction) |
| **... Core 32**| `different instruction` | `add r1, r2` (same instruction) |
| **Execution**| Independent, asynchronous | Synchronized, lockstep |
| **Scheduling**| Complex, OS-managed | Simple, hardware-managed |
| **Data**| Independent data sets | Different data, same operation |

**GPU Warp Execution Pattern:**

- **Instruction**: Same for all 32 lanes: `add r1, r2`
- **Lane 0**: Operates on `Data0`  `Result0`
- **Lane 1**: Operates on `Data1`  `Result1`
- **Lane 2**: Operates on `Data2`  `Result2`
- **... (all lanes execute simultaneously)**
- **Lane 31**: Operates on `Data31`  `Result31`

**Key insight:**All lanes execute the **same instruction**at the **same time**on **different data**.

#### Why SIMT works for GPUs

GPUs are optimized for **throughput**, not latency. SIMT enables:

- **Hardware simplification**: One instruction decoder serves 32 or 64 threads
- **Execution efficiency**: No complex scheduling between warp threads
- **Memory bandwidth**: Coalesced memory access patterns
- **Power efficiency**: Shared control logic across lanes

### Warp execution mechanics

#### Lane numbering and identity

Each thread within a warp has a **lane ID**from 0 to `WARP_SIZE-1`:

```mojo
from gpu import lane_id
from gpu.primitives.warp import WARP_SIZE

# Within a kernel function:
my_lane = lane_id()  # Returns 0-31 (NVIDIA/RDNA) or 0-63 (CDNA)
```

**Key insight:**`lane_id()` is **free**- it's just reading a hardware register, not computing a value.

#### Synchronization within warps

The most powerful aspect of SIMT: **implicit synchronization**.

```mojo
# Example with thread_idx.x < WARP_SIZE

# 1. Traditional shared memory approach:
shared[thread_idx.x] = partial_result
barrier()  # Explicit synchronization required
var total = shared[0] + shared[1] + ... + shared[WARP_SIZE] # Sum reduction

# 2. Warp approach:
from gpu.primitives.warp import sum

var total = sum(partial_result)  # Implicit synchronization!
```

**Why no barriers needed?**All lanes execute each instruction at exactly the same time. When `sum()` starts, all lanes have already computed their `partial_result`.

### Warp divergence and convergence

#### What happens with conditional code?

```mojo
if lane_id() % 2 == 0:
    # Even lanes execute this path
    result = compute_even()
else:
    # Odd lanes execute this path
    result = compute_odd()
# All lanes converge here
```

**Hardware behaviour steps:**

| Step | Phase | Active Lanes | Waiting Lanes | Efficiency | Performance Cost |
|------|-------|--------------|---------------|------------|------------------|
| **1**| Condition evaluation | All 32 lanes | None | 100% | Normal speed |
| **2**| Even lanes branch | Lanes 0,2,4...30 (16 lanes) | Lanes 1,3,5...31 (16 lanes) | 50% | **2 slower**|
| **3**| Odd lanes branch | Lanes 1,3,5...31 (16 lanes) | Lanes 0,2,4...30 (16 lanes) | 50% | **2 slower**|
| **4**| Convergence | All 32 lanes | None | 100% | Normal speed resumed |

**Example breakdown:**

- **Step 2**: Only even lanes execute `compute_even()` while odd lanes wait
- **Step 3**: Only odd lanes execute `compute_odd()` while even lanes wait
- **Total time**: `time(compute_even) + time(compute_odd)` (sequential execution)
- **Without divergence**: `max(time(compute_even), time(compute_odd))` (parallel execution)

**Performance impact:**

1. **Divergence**: Warp splits execution - some lanes active, others wait
2. **Serial execution**: Different paths run sequentially, not in parallel
3. **Convergence**: All lanes reunite and continue together
4. **Cost**: Divergent warps take 2 time (or more) vs unified execution

#### Best practices for warp efficiency

#### Warp efficiency patterns

** EXCELLENT: Uniform execution (100% efficiency)**

```mojo
# All lanes do the same work - no divergence
var partial = a[global_i] * b[global_i]
var total = sum(partial)
```

*Performance: All 32 lanes active simultaneously*

** ACCEPTABLE: Predictable divergence (~95% efficiency)**

```mojo
# Divergence based on lane_id() - hardware optimized
if lane_id() == 0:
    output[block_idx] = sum(partial)
```

*Performance: Brief single-lane operation, predictable pattern*

** CAUTION: Structured divergence (~50-75% efficiency)**

```mojo
# Regular patterns can be optimized by compiler
if (global_i / 4) % 2 == 0:
    result = method_a()
else:
    result = method_b()
```

*Performance: Predictable groups, some optimization possible*

** AVOID: Data-dependent divergence (~25-50% efficiency)**

```mojo
# Different lanes may take different paths based on data
if input[global_i] > threshold:  # Unpredictable branching
    result = expensive_computation()
else:
    result = simple_computation()
```

*Performance: Random divergence kills warp efficiency*

** TERRIBLE: Nested data-dependent divergence (~10-25% efficiency)**

```mojo
# Multiple levels of unpredictable branching
if input[global_i] > threshold1:
    if input[global_i] > threshold2:
        result = very_expensive()
    else:
        result = expensive()
else:
    result = simple()
```

*Performance: Warp efficiency destroyed*

### Cross-architecture compatibility

#### NVIDIA vs AMD warp sizes

```mojo
from gpu.primitives.warp import WARP_SIZE

# NVIDIA GPUs:     WARP_SIZE = 32
# AMD RDNA GPUs:   WARP_SIZE = 32 (wavefront32 mode)
# AMD CDNA GPUs:   WARP_SIZE = 64 (traditional wavefront64)
```

**Why this matters:**

- **Memory patterns**: Coalesced access depends on warp size
- **Algorithm design**: Reduction trees must account for warp size
- **Performance scaling**: Twice as many lanes per warp on AMD

#### Writing portable warp code

#### Architecture adaptation strategies

** PORTABLE: Always use `WARP_SIZE`**

```mojo
comptime THREADS_PER_BLOCK = (WARP_SIZE, 1)  # Adapts automatically
comptime ELEMENTS_PER_WARP = WARP_SIZE        # Scales with hardware
```

*Result: Code works optimally on NVIDIA/AMD (32) and AMD (64)*

** BROKEN: Never hardcode warp size**

```mojo
comptime THREADS_PER_BLOCK = (32, 1)  # Breaks on AMD GPUs!
comptime REDUCTION_SIZE = 32           # Wrong on AMD!
```

*Result: Suboptimal on AMD, potential correctness issues*

#### Real hardware impact

| GPU Architecture | WARP_SIZE | Memory per Warp | Reduction Steps | Lane Pattern |
|------------------|-----------|-----------------|-----------------|--------------|
| **NVIDIA/AMD RDNA**| 32 | 128 bytes (432) | 5 steps: 32168421 | Lanes 0-31 |
| **AMD CDNA**| 64 | 256 bytes (464) | 6 steps: 6432168421 | Lanes 0-63 |

**Performance implications of 64 vs 32:**

- **CDNA advantage**: 2 memory bandwidth per warp
- **CDNA advantage**: 2 computation per warp
- **NVIDIA/RDNA advantage**: More warps per block (better occupancy)
- **Code portability**: Same source, optimal performance on both

### Memory access patterns with warps

#### Coalesced memory access patterns

** PERFECT: Coalesced access (100% bandwidth utilization)**

```mojo
# Adjacent lanes -> adjacent memory addresses
var value = input[global_i]  # Lane 0->input[0], Lane 1->input[1], etc.
```

**Memory access patterns:**

| Access Pattern | NVIDIA/RDNA (32 lanes) | CDNA (64 lanes) | Bandwidth Utilization | Performance |
|----------------|-------------------|----------------|----------------------|-------------|
| ** Coalesced**| Lane N  Address 4N | Lane N  Address 4N | 100% | Optimal |
| | 1 transaction: 128 bytes | 1 transaction: 256 bytes | Full bus width | Fast |
| ** Scattered**| Lane N  Random address | Lane N  Random address | ~6% | Terrible |
| | 32 separate transactions | 64 separate transactions | Mostly idle bus | **32 slower**|

**Example addresses:**

- **Coalesced**: Lane 00, Lane 14, Lane 28, Lane 312, ...
- **Scattered**: Lane 01000, Lane 152, Lane 2997, Lane 38, ...

#### Shared memory bank conflicts

**What is a bank conflict?**

Assume that a GPU shared memory is divided into 32 independent **banks**that can be accessed simultaneously. A **bank conflict**occurs when multiple threads in a warp try to access different addresses within the same bank at the same time. When this happens, the hardware must **serialize**these accesses, turning what should be a single-cycle operation into multiple cycles.

**Key concepts:**

- **No conflict**: Each thread accesses a different bank  All accesses happen simultaneously (1 cycle)
- **Bank conflict**: Multiple threads access the same bank  Accesses happen sequentially (N cycles for N threads)
- **Broadcast**: All threads access the same address  Hardware optimizes this to 1 cycle

**Shared memory bank organization:**

| Bank | Addresses (byte offsets) | Example Data (float32) |
|------|--------------------------|------------------------|
| Bank 0 | 0, 128, 256, 384, ... | `shared[0]`, `shared[32]`, `shared[64]`, ... |
| Bank 1 | 4, 132, 260, 388, ... | `shared[1]`, `shared[33]`, `shared[65]`, ... |
| Bank 2 | 8, 136, 264, 392, ... | `shared[2]`, `shared[34]`, `shared[66]`, ... |
| ... | ... | ... |
| Bank 31 | 124, 252, 380, 508, ... | `shared[31]`, `shared[63]`, `shared[95]`, ... |

**Bank conflict examples:**

| Access Pattern | Bank Usage | Cycles | Performance | Explanation |
|----------------|------------|--------|-------------|-------------|
| ** Sequential**| `shared[thread_idx.x]` | 1 cycle | 100% | Each lane hits different bank |
| | Lane 0Bank 0, Lane 1Bank 1, ... | | Optimal | No conflicts |
| ** Same index**| `shared[0]`| 1 cycle | 100% | All lanes broadcast from same address |
| | All 32 lanesBank 0 (Same address) | | Optimal | No conflicts |
| ** Stride 2**| `shared[thread_idx.x * 2]` | 2 cycles | 50% | 2 lanes per bank |
| | Lane 0,16Bank 0; Lane 1,17Bank 1 | | **2 slower**| Serialized access |
| ** Stride 32**| `shared[thread_idx.x * 32]` | 32 cycles | 3% | All lanes hit same bank |
| | All 32 lanesBank 0 (Different address) | | **32 slower**| Completely serialized |

### Practical implications for warp programming

#### When warp operations are most effective

1. **Reduction operations**: `sum()`, `max()`, etc.
2. **Broadcast operations**: `shuffle_idx()` to share values
3. **Neighbor communication**: `shuffle_down()` for sliding windows
4. **Prefix computations**: `prefix_sum()` for scan algorithms

#### Performance characteristics

| Operation Type | Traditional | Warp Operations |
|----------------|------------|-----------------|
| **Reduction (32 elements)**| ~20 instructions | 10 instructions |
| **Memory traffic**| High | Minimal |
| **Synchronization cost**| Expensive | Free |
| **Code complexity**| High | Low |

Now that you understand the SIMT foundation, you're ready to see how these concepts enable powerful warp operations. The next section will show you how `sum()` transforms complex reduction patterns into simple, efficient function calls.

** Continue to [warp.sum() Essentials](#warpsum-essentials-warp-level-dot-product)**

## warp.sum() Essentials - Warp-Level Dot Product

Implement the dot product we saw in puzzle 12 using Mojo's warp operations to replace complex shared memory patterns with simple function calls. Each warp lane will process one element and use `warp.sum()` to combine results automatically, demonstrating how warp programming transforms GPU synchronization.

**Key insight:**_The [warp.sum()](https://docs.modular.com/mojo/stdlib/gpu/warp/sum) operation leverages SIMT execution to replace shared memory + barriers + tree reduction with a single hardware-accelerated instruction._

### Key concepts

In this puzzle, you'll learn:

- **Warp-level reductions**with `warp.sum()`
- **SIMT execution model**and lane synchronization
- **Cross-architecture compatibility**with `WARP_SIZE`
- **Performance transformation**from complex to simple patterns
- **Lane ID management**and conditional writes

The mathematical operation is a dot product (inner product):
\\[\Large \text{output}[0] = \sum_{i=0}^{N-1} a[i] \times b[i]\\]

But the implementation teaches fundamental patterns for all warp-level GPU programming in Mojo.

### Configuration

- Vector size: `SIZE = WARP_SIZE` (32 or 64 depending on GPU architecture)
- Data type: `DType.float32`
- Block configuration: `(WARP_SIZE, 1)` threads per block
- Grid configuration: `(1, 1)` blocks per grid
- Layout: `Layout.row_major(SIZE)` (1D row-major)

### The traditional complexity (from Puzzle 12)

Recall the complex approach from solutions/p12/p12.mojo that required shared memory, barriers, and tree reduction:

```mojo
comptime SIZE = WARP_SIZE
comptime BLOCKS_PER_GRID = (1, 1)
comptime THREADS_PER_BLOCK = (WARP_SIZE, 1)
comptime dtype = DType.float32
comptime SIMD_WIDTH = simd_width_of[dtype]()
comptime in_layout = Layout.row_major(SIZE)
comptime out_layout = Layout.row_major(1)

fn traditional_dot_product_p12_style
    in_layout: Layout, out_layout: Layout, size: Int
:
    """
    This is the complex approach from p12_layout_tensor.mojo - kept for comparison.
    """
    shared = LayoutTensor[
        dtype,
        Layout.row_major(WARP_SIZE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    local_i = Int(thread_idx.x)

    if global_i < size:
        shared[local_i] = (a[global_i] * b[global_i]).reduce_add()
    else:
        shared[local_i] = 0.0

    barrier()

    stride = WARP_SIZE // 2
    while stride > 0:
        if local_i < stride:
            shared[local_i] += shared[local_i + stride]
        barrier()
        stride //= 2

    if local_i == 0:
        output[global_i // WARP_SIZE] = shared[0]

```

**What makes this complex:**

- **Shared memory allocation**: Manual memory management within blocks
- **Explicit barriers**: `barrier()` calls to synchronize threads
- **Tree reduction**: Complex loop with stride-based indexing
- **Conditional writes**: Only thread 0 writes the final result

This works, but it's verbose, error-prone, and requires deep understanding of GPU synchronization.

**Test the traditional approach:**

  
    pixi NVIDIA (default)
    pixi AMD
    pixi Apple
    uv
  
  

```bash
pixi run p24 --traditional
```

  
  

```bash
pixi run -e amd p24 --traditional
```

  
  

```bash
pixi run -e apple p24 --traditional
```

  
  

```bash
uv run poe p24 --traditional
```

  

### Performance comparison with benchmarks

Run comprehensive benchmarks to see how warp operations scale:

  
    uv
    pixi
  
  

```bash
uv run poe p24 --benchmark
```

  
  

```bash
pixi run p24 --benchmark
```

  

Here's example output from a complete benchmark run:

```
SIZE: 32
WARP_SIZE: 32
SIMD_WIDTH: 8
--------------------------------------------------------------------------------
Testing SIZE=1 x WARP_SIZE, BLOCKS=1
Running traditional_1x
Running simple_warp_1x
Running functional_warp_1x
--------------------------------------------------------------------------------
Testing SIZE=4 x WARP_SIZE, BLOCKS=4
Running traditional_4x
Running simple_warp_4x
Running functional_warp_4x
--------------------------------------------------------------------------------
Testing SIZE=32 x WARP_SIZE, BLOCKS=32
Running traditional_32x
Running simple_warp_32x
Running functional_warp_32x
--------------------------------------------------------------------------------
Testing SIZE=256 x WARP_SIZE, BLOCKS=256
Running traditional_256x
Running simple_warp_256x
Running functional_warp_256x
--------------------------------------------------------------------------------
Testing SIZE=2048 x WARP_SIZE, BLOCKS=2048
Running traditional_2048x
Running simple_warp_2048x
Running functional_warp_2048x
--------------------------------------------------------------------------------
Testing SIZE=16384 x WARP_SIZE, BLOCKS=16384 (Large Scale)
Running traditional_16384x
Running simple_warp_16384x
Running functional_warp_16384x
--------------------------------------------------------------------------------
Testing SIZE=65536 x WARP_SIZE, BLOCKS=65536 (Massive Scale)
Running traditional_65536x
Running simple_warp_65536x
Running functional_warp_65536x
| name                   | met (ms)              | iters |
| ---------------------- | --------------------- | ----- |
| traditional_1x         | 0.00460128            | 100   |
| simple_warp_1x         | 0.00574047            | 100   |
| functional_warp_1x     | 0.00484192            | 100   |
| traditional_4x         | 0.00492671            | 100   |
| simple_warp_4x         | 0.00485247            | 100   |
| functional_warp_4x     | 0.00587679            | 100   |
| traditional_32x        | 0.0062406399999999996 | 100   |
| simple_warp_32x        | 0.0054918400000000004 | 100   |
| functional_warp_32x    | 0.00552447            | 100   |
| traditional_256x       | 0.0050614300000000004 | 100   |
| simple_warp_256x       | 0.00488768            | 100   |
| functional_warp_256x   | 0.00461472            | 100   |
| traditional_2048x      | 0.01120031            | 100   |
| simple_warp_2048x      | 0.00884383            | 100   |
| functional_warp_2048x  | 0.007038720000000001  | 100   |
| traditional_16384x     | 0.038533750000000005  | 100   |
| simple_warp_16384x     | 0.0323264             | 100   |
| functional_warp_16384x | 0.01674271            | 100   |
| traditional_65536x     | 0.19784991999999998   | 100   |
| simple_warp_65536x     | 0.12870176            | 100   |
| functional_warp_65536x | 0.048680310000000004  | 100   |

Benchmarks completed!

WARP OPERATIONS PERFORMANCE ANALYSIS:
   GPU Architecture: NVIDIA (WARP_SIZE=32) vs AMD (WARP_SIZE=64)
   - 1,...,256 x WARP_SIZE: Grid size too small to benchmark
   - 2048 x WARP_SIZE: Warp primative benefits emerge
   - 16384 x WARP_SIZE: Large scale (512K-1M elements)
   - 65536 x WARP_SIZE: Massive scale (2M-4M elements)

   Expected Results at Large Scales:
   - Traditional: Slower due to more barrier overhead
   - Warp operations: Faster, scale better with problem size
   - Memory bandwidth becomes the limiting factor
```

**Performance insights from this example:**

- **Small scales (1x-4x)**: Warp operations show modest improvements (~10-15% faster)
- **Medium scale (32x-256x)**: Functional approach often performs best
- **Large scales (16K-65K)**: All approaches converge as memory bandwidth dominates
- **Variability**: Performance depends heavily on specific GPU architecture and memory subsystem

**Note:**Your results will vary significantly depending on your hardware (GPU model, memory bandwidth, `WARP_SIZE`). The key insight is observing the relative performance trends rather than absolute timings.

Once you've learned warp sum operations, you're ready for:

- **[When to Use Warp Programming](#when-to-use-warp-programming)**: Strategic decision framework for warp vs traditional approaches
- **Advanced warp operations**: `shuffle_idx()`, `shuffle_down()`, `prefix_sum()` for complex communication patterns
- **Multi-warp algorithms**: Combining warp operations with block-level synchronization
- **Part VII: Memory Coalescing**: Optimizing memory access patterns for maximum bandwidth

 **Key Takeaway**: Warp operations transform GPU programming by replacing complex synchronization patterns with hardware-accelerated primitives, demonstrating how understanding the execution model enables dramatic simplification without sacrificing performance.

## When to Use Warp Programming

### Quick decision guide

** Use warp operations when:**
- Reduction operations (`sum`, `max`, `min`) with 32+ elements
- Regular memory access patterns (adjacent lanes  adjacent addresses)
- Need cross-architecture portability (NVIDIA/RDNA 32 vs CDNA 64 threads)
- Want simpler, more maintainable code

** Use traditional approaches when:**
- Complex cross-warp synchronization required
- Irregular/scattered memory access patterns
- Variable work per thread (causes warp divergence)
- Problem `size < WARP_SIZE`

### Performance characteristics

#### Problem size scaling
| Elements | Warp Advantage | Notes |
|----------|---------------|-------|
| < 32 | None | Traditional better |
| 32-1K | 1.2-1.5 | Sweet spot begins |
| 1K-32K | 1.5-2.5 | **Warp operations excel**|
| > 32K | Memory-bound | Both approaches limited by bandwidth |

#### Key warp advantages
- **No synchronization overhead**: Eliminates barrier costs
- **Minimal memory usage**: No shared memory allocation needed
- **Better scaling**: Performance improves with more warps
- **Simpler code**: Fewer lines, less error-prone

### Algorithm-specific guidance

| Algorithm | Recommendation | Reason |
|-----------|---------------|--------|
| **Dot product**| Warp ops (1K+ elements) | Single reduction, regular access |
| **Matrix row/col sum**| Warp ops | Natural reduction pattern |
| **Prefix sum**| Always warp `prefix_sum()` | Hardware-optimized primitive |
| **Pooling (max/min)**| Warp ops (regular windows) | Efficient window reductions |
| **Histogram with large number of bins**| Traditional | Irregular writes, atomic updates |

### Code examples

####  Perfect for warps
```mojo
# Reduction operations
from gpu.primitives.warp import sum, max
var total = sum(partial_values)
var maximum = max(partial_values)

# Communication patterns
from gpu.primitives.warp import shuffle_idx, prefix_sum
var broadcast = shuffle_idx(my_value, 0)
var running_sum = prefix_sum(my_value)
```

####  Better with traditional approaches
```mojo
# Complex multi-stage synchronization
stage1_compute()
barrier()  # Need ALL threads to finish
stage2_depends_on_stage1()

# Irregular memory access
var value = input[random_indices[global_i]]  # Scattered reads

# Data-dependent work
if input[global_i] > threshold:
    result = expensive_computation()  # Causes warp divergence
```

### Performance measurement

```bash
# Always benchmark both approaches
mojo p22.mojo --benchmark

# Look for scaling patterns:
# traditional_1x:  X.XX ms
# warp_1x:         Y.YY ms  # Should be faster
# warp_32x:        Z.ZZ ms  # Advantage should increase
```

### Summary

**Start with warp operations for:**
- Reductions with regular access patterns
- Problems  1 warp in size
- Cross-platform compatibility needs

**Use traditional approaches for:**
- Complex synchronization requirements
- Irregular memory patterns
- Small problems or heavy divergence

**When in doubt:**Implement both and benchmark. The performance difference will guide your decision.
