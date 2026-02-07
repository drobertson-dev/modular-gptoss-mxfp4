---
title: "GPU Functional Programming Patterns"
description: "Understand the fundamental relationship between GPU threads and SIMD operations:"
---

# GPU Functional Programming Patterns

Understand the fundamental relationship between GPU threads and SIMD operations:

## Overview

**Part VI: Functional GPU Programming**introduces Mojo's high-level programming patterns for GPU computation. You'll learn functional approaches that automatically handle vectorization, memory optimization, and performance tuning, replacing manual GPU kernel programming.

**Key insight:**_Modern GPU programming doesn't require sacrificing elegance for performance - Mojo's functional patterns give you both._

## What you'll learn

### **GPU execution hierarchy**

Understand the fundamental relationship between GPU threads and SIMD operations:

```
GPU Device
|-- Grid (your entire problem)
|   |-- Block 1 (group of threads, shared memory)
|   |   |-- Warp 1 (32 threads, lockstep execution) --> We'll learn in Part VI
|   |   |   |-- Thread 1 -> SIMD
|   |   |   |-- Thread 2 -> SIMD
|   |   |   `-- ... (32 threads total)
|   |   `-- Warp 2 (32 threads)
|   `-- Block 2 (independent group)
```

**What Mojo abstracts for you:**

- Grid/Block configuration automatically calculated
- Warp management handled transparently
- Thread scheduling optimized automatically
- Memory hierarchy optimization built-in

 **Note**: While this Part focuses on functional patterns, **warp-level programming**and advanced GPU memory management will be covered in detail in **Part VII**.

### **Four fundamental patterns**

Learn the complete spectrum of GPU functional programming:

1. **Elementwise**: Maximum parallelism with automatic SIMD vectorization
2. **Tiled**: Memory-efficient processing with cache optimization
3. **Manual vectorization**: Expert-level control over SIMD operations
4. **Mojo vectorize**: Safe, automatic vectorization with bounds checking

### **Performance patterns you'll recognize**

```
Problem: Add two 1024-element vectors (SIZE=1024, SIMD_WIDTH=4)

Elementwise:     256 threads x 1 SIMD op   = High parallelism
Tiled:           32 threads  x 8 SIMD ops  = Cache optimization
Manual:          8 threads   x 32 SIMD ops = Maximum control
Mojo vectorize:  32 threads  x 8 SIMD ops  = Automatic safety
```

###  **Real performance insights**

Learn to interpret empirical benchmark results:

```
Benchmark Results (SIZE=1,048,576):
elementwise:        11.34ms  <- Maximum parallelism wins at scale
tiled:              12.04ms  <- Good balance of locality and parallelism
manual_vectorized:  15.75ms  <- Complex indexing hurts simple operations
vectorized:         13.38ms  <- Automatic optimization overhead
```

## Prerequisites

Before diving into functional patterns, ensure you're comfortable with:

- **Basic GPU concepts**: Memory hierarchy, thread execution, SIMD operations
- **Mojo fundamentals**: Parameter functions, compile-time specialization, capturing semantics
- **LayoutTensor operations**: Loading, storing, and tensor manipulation
- **GPU memory management**: Buffer allocation, host-device synchronization

## Learning path

### **1. Elementwise operations**

** [Elementwise - Basic GPU Functional Operations](#elementwise-basic-gpu-functional-operations)**

Start with the foundation: automatic thread management and SIMD vectorization.

**What you'll learn:**

- Functional GPU programming with `elementwise`
- Automatic SIMD vectorization within GPU threads
- LayoutTensor operations for safe memory access
- Capturing semantics in nested functions

**Key pattern:**

```mojo
elementwiseadd_function, SIMD_WIDTH, target="gpu"
```

### **2. Tiled processing**

** [Tile - Memory-Efficient Tiled Processing](#tile-memory-efficient-tiled-processing)**

Build on elementwise with memory-optimized tiling patterns.

**What you'll learn:**

- Tile-based memory organization for cache optimization
- Sequential SIMD processing within tiles
- Memory locality principles and cache-friendly access patterns
- Thread-to-tile mapping vs thread-to-element mapping

**Key insight:**Tiling trades parallel breadth for memory locality - fewer threads each doing more work with better cache utilization.

### **3. Advanced vectorization**

** [Vectorization - Fine-Grained SIMD Control](#vectorization-fine-grained-simd-control)**

Explore manual control and automatic vectorization strategies.

**What you'll learn:**

- Manual SIMD operations with explicit index management
- Mojo's vectorize function for safe, automatic vectorization
- Chunk-based memory organization for optimal SIMD alignment
- Performance trade-offs between manual control and safety

**Two approaches:**

- **Manual**: Direct control, maximum performance, complex indexing
- **Mojo vectorize**: Automatic optimization, built-in safety, clean code

###  **4. Threading vs SIMD concepts**

** [GPU Threading vs SIMD - Understanding the Execution Hierarchy](#gpu-threading-vs-simd-understanding-the-execution-hierarchy)**

Understand the fundamental relationship between parallelism levels.

**What you'll learn:**

- GPU threading hierarchy and hardware mapping
- SIMD operations within GPU threads
- Pattern comparison and thread-to-work mapping
- Choosing the right pattern for different workloads

**Key insight:**GPU threads provide the parallelism structure, while SIMD operations provide the vectorization within each thread.

###  **5. Performance benchmarking in Mojo**

** [Benchmarking in Mojo](#benchmarking-performance-analysis-and-optimization)**

Learn to measure, analyze, and optimize GPU performance scientifically.

**What you'll learn:**

- Mojo's built-in benchmarking framework
- GPU-specific timing and synchronization challenges
- Parameterized benchmark functions with compile-time specialization
- Empirical performance analysis and pattern selection

**Critical technique:**Using `keep()` to prevent compiler optimization of benchmarked code.

## Getting started

Start with the elementwise pattern and work through each section systematically. Each puzzle builds on the previous concepts while introducing new levels of sophistication.

 **Success tip**: Focus on understanding the **why**behind each pattern, not just the **how**. The conceptual framework you develop here will serve you throughout your GPU programming career.

**Learning objective**: By the end of Part VI, you'll think in terms of functional patterns rather than low-level GPU mechanics, enabling you to write more maintainable, performant, and portable GPU code.

**Begin with**: **[Elementwise Operations](#elementwise-basic-gpu-functional-operations)**to discover functional GPU programming.

## Elementwise - Basic GPU Functional Operations

This puzzle implements vector addition using Mojo's functional `elementwise` pattern. Each thread automatically processes multiple SIMD elements, showing how modern GPU programming abstracts low-level details while preserving high performance.

**Key insight:**_The [elementwise](https://docs.modular.com/mojo/stdlib/algorithm/functional/elementwise/) function automatically handles thread management, SIMD vectorization, and memory coalescing for you._

### Key concepts

This puzzle covers:

- **Functional GPU programming**with `elementwise`
- **Automatic SIMD vectorization**within GPU threads
- **LayoutTensor operations**for safe memory access
- **GPU thread hierarchy**vs SIMD operations
- **Capturing semantics**in nested functions

The mathematical operation is simple element-wise addition:
\\[\Large \text{output}[i] = a[i] + b[i]\\]

The implementation covers fundamental patterns applicable to all GPU functional programming in Mojo.

### Configuration

- Vector size: `SIZE = 1024`
- Data type: `DType.float32`
- SIMD width: Target-dependent (determined by GPU architecture and data type)
- Layout: `Layout.row_major(SIZE)` (1D row-major)

### Running the code

To test your solution, run the following command in your terminal:

  
    pixi NVIDIA (default)
    pixi AMD
    pixi Apple
    uv
  
  

```bash
pixi run p23 --elementwise
```

  
  

```bash
pixi run -e amd p23 --elementwise
```

  
  

```bash
pixi run -e apple p23 --elementwise
```

  
  

```bash
uv run poe p23 --elementwise
```

  

Your output will look like this if the puzzle isn't solved yet:

```txt
SIZE: 1024
simd_width: 4
...
idx: 404
idx: 408
idx: 412
idx: 416
...

out: HostBuffer([0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0])
expected: HostBuffer([1.0, 5.0, 9.0, ..., 4085.0, 4089.0, 4093.0])
```

### Reference implementation (example)


```mojo
fn elementwise_add
    layout: Layout, dtype: DType, simd_width: Int, rank: Int, size: Int
 raises:
    @parameter
    @always_inline
    fn add[
        simd_width: Int, rank: Int, alignment: Int = align_of[dtype]()
    ](indices: IndexList[rank]) capturing -> None:
        idx = indices[0]
        # Note: This is thread-local SIMD - each thread processes its own vector of data
        # we'll later better see this hierarchy in Mojo:
        # SIMD within threads, warp across threads, block across warps
        a_simd = a.aligned_loadwidth=simd_width)
        b_simd = b.aligned_loadwidth=simd_width)
        ret = a_simd + b_simd
        # print(
        #     "idx:", idx, ", a_simd:", a_simd, ", b_simd:", b_simd, " sum:", ret
        # )
        output.storesimd_width, ret)

    elementwiseadd, SIMD_WIDTH, target="gpu", ctx)

```

The elementwise functional pattern in Mojo introduces several fundamental concepts for modern GPU programming:

#### 1. **Functional abstraction philosophy**

The `elementwise` function represents a paradigm shift from traditional GPU programming:

**Traditional CUDA/HIP approach:**

```mojo
# Manual thread management
idx = thread_idx.x + block_idx.x * block_dim.x
if idx < size:
    output[idx] = a[idx] + b[idx];  // Scalar operation
```

**Mojo functional approach:**

```mojo
# Automatic management + SIMD vectorization
elementwiseadd_function, simd_width, target="gpu"
```

**What `elementwise` abstracts away:**

- **Thread grid configuration**: No need to calculate block/grid dimensions
- **Bounds checking**: Automatic handling of array boundaries
- **Memory coalescing**: Optimal memory access patterns built-in
- **SIMD orchestration**: Vectorization handled transparently
- **GPU target selection**: Works across different GPU architectures

#### 2. **Deep dive: nested function architecture**

```mojo
@parameter
@always_inline
fn addsimd_width: Int, rank: Int capturing -> None:
```

**Parameter Analysis:**

- **`@parameter`**: This decorator provides **compile-time specialization**. The function is generated separately for each unique `simd_width` and `rank`, allowing aggressive optimization.
- **`@always_inline`**: Critical for GPU performance - eliminates function call overhead by embedding the code directly into the kernel.
- **`capturing`**: Enables **lexical scoping**- the inner function can access variables from the outer scope without explicit parameter passing.
- **`IndexList[rank]`**: Provides **dimension-agnostic indexing**- the same pattern works for 1D vectors, 2D matrices, 3D tensors, etc.

#### 3. **SIMD execution model deep dive**

```mojo
idx = indices[0]                                  # Linear index: 0, 4, 8, 12... (GPU-dependent spacing)
a_simd = a.aligned_loadsimd_width)       # Load: [a[0:4], a[4:8], a[8:12]...] (4 elements per load)
b_simd = b.aligned_loadsimd_width)       # Load: [b[0:4], b[4:8], b[8:12]...] (4 elements per load)
ret = a_simd + b_simd                             # SIMD: 4 additions in parallel (GPU-dependent)
output.storesimd_width, ret)     # Store: 4 results simultaneously (GPU-dependent)
```

**Execution Hierarchy Visualization:**

```
GPU Architecture:
|-- Grid (entire problem)
|   |-- Block 1 (multiple warps)
|   |   |-- Warp 1 (32 threads) --> We'll learn about Warp in the next Part VI
|   |   |   |-- Thread 1 -> SIMD[4 elements]  <- Our focus (GPU-dependent width)
|   |   |   |-- Thread 2 -> SIMD[4 elements]
|   |   |   `-- ...
|   |   `-- Warp 2 (32 threads)
|   `-- Block 2 (multiple warps)
```

**For a 1024-element vector with SIMD_WIDTH=4 (example GPU):**

- **Total SIMD operations needed**: 1024  4 = 256
- **GPU launches**: 256 threads (1024  4)
- **Each thread processes**: Exactly 4 consecutive elements
- **Memory bandwidth**: SIMD_WIDTH improvement over scalar operations

**Note**: SIMD width varies by GPU architecture (e.g., 4 for some GPUs, 8 for RTX 4090, 16 for A100).

#### 4. **Memory access pattern analysis**

```mojo
a.aligned_loadsimd_width)  // Coalesced memory access
```

**Memory Coalescing Benefits:**

- **Sequential access**: Threads access consecutive memory locations
- **Cache optimization**: Maximizes L1/L2 cache hit rates
- **Bandwidth utilization**: Achieves near-theoretical memory bandwidth
- **Hardware efficiency**: GPU memory controllers optimized for this pattern

**Example for SIMD_WIDTH=4 (GPU-dependent):**

```
Thread 0: loads a[0:4]   -> Memory bank 0-3
Thread 1: loads a[4:8]   -> Memory bank 4-7
Thread 2: loads a[8:12]  -> Memory bank 8-11
...
Result: Optimal memory controller utilization
```

#### 5. **Performance characteristics & optimization**

**Computational Intensity Analysis (for SIMD_WIDTH=4):**

- **Arithmetic operations**: 1 SIMD addition per 4 elements
- **Memory operations**: 2 SIMD loads + 1 SIMD store per 4 elements
- **Arithmetic intensity**: 1 add  3 memory ops = 0.33 (memory-bound)

**Why This Is Memory-Bound:**

```
Memory bandwidth >>> Compute capability for simple operations
```

**Optimization Implications:**

- Focus on memory access patterns rather than arithmetic optimization
- SIMD vectorization provides the primary performance benefit
- Memory coalescing is critical for performance
- Cache locality matters more than computational complexity

#### 6. **Scaling and adaptability**

**Automatic Hardware Adaptation:**

```mojo
comptime SIMD_WIDTH = simd_width_of[dtype, target = _get_gpu_target()]()
```

- **GPU-specific optimization**: SIMD width adapts to hardware (e.g., 4 for some cards, 8 for RTX 4090, 16 for A100)
- **Data type awareness**: Different SIMD widths for float32 vs float16
- **Compile-time optimization**: Zero runtime overhead for hardware detection

**Scalability Properties:**

- **Thread count**: Automatically scales with problem size
- **Memory usage**: Linear scaling with input size
- **Performance**: Near-linear speedup until memory bandwidth saturation

#### 7. **Advanced insights: why this pattern matters**

**Foundation for Complex Operations:**
This elementwise pattern is the building block for:

- **Reduction operations**: Sum, max, min across large arrays
- **Broadcast operations**: Scalar-to-vector operations
- **Complex transformations**: Activation functions, normalization
- **Multi-dimensional operations**: Matrix operations, convolutions

**Compared to Traditional Approaches:**

```mojo
// Traditional: Error-prone, verbose, hardware-specific
__global__ void add_kernel(float* output, float* a, float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = a[idx] + b[idx];  // No vectorization
    }
}

// Mojo: Safe, concise, automatically vectorized
elementwiseadd, SIMD_WIDTH, target="gpu"
```

**Benefits of Functional Approach:**

- **Safety**: Automatic bounds checking prevents buffer overruns
- **Portability**: Same code works across GPU vendors/generations
- **Performance**: Compiler optimizations often exceed hand-tuned code
- **Maintainability**: Clean abstractions reduce debugging complexity
- **Composability**: Easy to combine with other functional operations

This pattern represents the future of GPU programming - high-level abstractions that don't sacrifice performance, making GPU computing accessible while maintaining optimal efficiency.

Once you've learned elementwise operations, you're ready for:

- **[Tile Operations](#tile-memory-efficient-tiled-processing)**: Memory-efficient tiled processing patterns
- **[Vectorization](#vectorization-fine-grained-simd-control)**: Fine-grained SIMD control
- **[ GPU Threading vs SIMD](#gpu-threading-vs-simd-understanding-the-execution-hierarchy)**: Understanding the execution hierarchy
- **[ Benchmarking](#benchmarking-performance-analysis-and-optimization)**: Performance analysis and optimization

 **Key Takeaway**: The `elementwise` pattern shows how Mojo combines functional programming elegance with GPU performance, automatically handling vectorization and thread management while maintaining full control over the computation.

## Tile - Memory-Efficient Tiled Processing

### Overview

Building on the **elementwise**pattern, this puzzle introduces **tiled processing**- a fundamental technique for optimizing memory access patterns and cache utilization on GPUs. Instead of each thread processing individual SIMD vectors across the entire array, tiling organizes data into smaller, manageable chunks that fit better in cache memory.

You've already seen tiling in action with **Puzzle 16's tiled matrix multiplication**, where we used tiles to process large matrices efficiently. Here, we apply the same tiling principles to vector operations, demonstrating how this technique scales from 2D matrices to 1D arrays.

Implement the same vector addition operation using Mojo's tiled approach. Each GPU thread will process an entire tile of data sequentially, demonstrating how memory locality can improve performance for certain workloads.

**Key insight:**_Tiling trades parallel breadth for memory locality - fewer threads each doing more work with better cache utilization._

### Key concepts

In this puzzle, you'll learn:

- **Tile-based memory organization**for cache optimization
- **Sequential SIMD processing**within tiles
- **Memory locality principles**and cache-friendly access patterns
- **Thread-to-tile mapping**vs thread-to-element mapping
- **Performance trade-offs**between parallelism and memory efficiency

The same mathematical operation as elementwise:
\\[\Large \text{output}[i] = a[i] + b[i]\\]

But with a completely different execution strategy optimized for memory hierarchy.

### Configuration

- Vector size: `SIZE = 1024`
- Tile size: `TILE_SIZE = 32`
- Data type: `DType.float32`
- SIMD width: GPU-dependent (for operations within tiles)
- Layout: `Layout.row_major(SIZE)` (1D row-major)

### Running the code

To test your solution, run the following command in your terminal:

  
    pixi NVIDIA (default)
    pixi AMD
    pixi Apple
    uv
  
  

```bash
pixi run p23 --tiled
```

  
  

```bash
pixi run -e amd p23 --tiled
```

  
  

```bash
pixi run -e apple p23 --tiled
```

  
  

```bash
uv run poe p23 --tiled
```

  

Your output will look like this when not yet solved:

```txt
SIZE: 1024
simd_width: 4
tile size: 32
tile_id: 0
tile_id: 1
tile_id: 2
tile_id: 3
...
tile_id: 29
tile_id: 30
tile_id: 31
out: HostBuffer([0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0])
expected: HostBuffer([1.0, 5.0, 9.0, ..., 4085.0, 4089.0, 4093.0])
```

### Reference implementation (example)


```mojo
comptime TILE_SIZE = 32

fn tiled_elementwise_add
    layout: Layout,
    dtype: DType,
    simd_width: Int,
    rank: Int,
    size: Int,
    tile_size: Int,
 raises:
    @parameter
    @always_inline
    fn process_tiles[
        simd_width: Int, rank: Int, alignment: Int = align_of[dtype]()
    ](indices: IndexList[rank]) capturing -> None:
        tile_id = indices[0]

        output_tile = output.tiletile_size
        a_tile = a.tiletile_size
        b_tile = b.tiletile_size

        @parameter
        for i in range(tile_size):
            a_vec = a_tile.loadsimd_width)
            b_vec = b_tile.loadsimd_width)
            ret = a_vec + b_vec
            output_tile.storesimd_width, ret)

    num_tiles = (size + tile_size - 1) // tile_size
    elementwiseprocess_tiles, 1, target="gpu"

```

The tiled processing pattern demonstrates advanced memory optimization techniques for GPU programming:

#### 1. **Tiling philosophy and memory hierarchy**

Tiling represents a fundamental shift in how we think about parallel processing:

**Elementwise approach:**

- **Wide parallelism**: Many threads, each doing minimal work
- **Global memory pressure**: Threads scattered across entire array
- **Cache misses**: Poor spatial locality across thread boundaries

**Tiled approach:**

- **Deep parallelism**: Fewer threads, each doing substantial work
- **Localized memory access**: Each thread works on contiguous data
- **Cache optimization**: Excellent spatial and temporal locality

#### 2. **Tile organization and indexing**

```mojo
tile_id = indices[0]
out_tile = output.tiletile_size
a_tile = a.tiletile_size
b_tile = b.tiletile_size
```

**Tile mapping visualization (TILE_SIZE=32):**

```
Original array: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ..., 1023]

Tile 0 (thread 0): [0, 1, 2, ..., 31]      <- Elements 0-31
Tile 1 (thread 1): [32, 33, 34, ..., 63]   <- Elements 32-63
Tile 2 (thread 2): [64, 65, 66, ..., 95]   <- Elements 64-95
...
Tile 31 (thread 31): [992, 993, ..., 1023] <- Elements 992-1023
```

**Key insights:**

- Each `tilesize` creates a **view**into the original tensor
- Views are zero-copy - no data movement, just pointer arithmetic
- Tile boundaries are always aligned to `tile_size` boundaries

#### 3. **Sequential processing deep dive**

```mojo
@parameter
for i in range(tile_size):
    a_vec = a_tile.loadsimd_width
    b_vec = b_tile.loadsimd_width
    ret = a_vec + b_vec
    out_tile.storesimd_width
```

**Why sequential processing?**

- **Cache optimization**: Consecutive memory accesses maximize cache hit rates
- **Compiler optimization**: `@parameter` loops unroll completely at compile-time
- **Memory bandwidth**: Sequential access aligns with memory controller design
- **Reduced coordination**: No need to synchronize between SIMD groups

**Execution pattern within one tile (TILE_SIZE=32, SIMD_WIDTH=4):**

```
Thread processes tile sequentially:
Step 0: Process elements [0:4] with SIMD
Step 1: Process elements [4:8] with SIMD
Step 2: Process elements [8:12] with SIMD
...
Step 7: Process elements [28:32] with SIMD
Total: 8 SIMD operations per thread (32 / 4 = 8)
```

#### 4. **Memory access pattern analysis**

**Cache behavior comparison:**

**Elementwise pattern:**

```
Thread 0: accesses global positions [0, 4, 8, 12, ...]    <- Stride = SIMD_WIDTH
Thread 1: accesses global positions [4, 8, 12, 16, ...]   <- Stride = SIMD_WIDTH
...
Result: Memory accesses spread across entire array
```

**Tiled pattern:**

```
Thread 0: accesses positions [0:32] sequentially         <- Contiguous 32-element block
Thread 1: accesses positions [32:64] sequentially       <- Next contiguous 32-element block
...
Result: Perfect spatial locality within each thread
```

**Cache efficiency implications:**

- **L1 cache**: Small tiles often fit better in L1 cache, reducing cache misses
- **Memory bandwidth**: Sequential access maximizes effective bandwidth
- **TLB efficiency**: Fewer translation lookbook buffer misses
- **Prefetching**: Hardware prefetchers work optimally with sequential patterns

#### 5. **Thread configuration strategy**

```mojo
elementwiseprocess_tiles, 1, target="gpu"
```

**Why `1` instead of `SIMD_WIDTH`?**

- **Thread count**: Launch exactly `num_tiles` threads, not `num_tiles  SIMD_WIDTH`
- **Work distribution**: Each thread handles one complete tile
- **Load balancing**: More work per thread, fewer threads total
- **Memory locality**: Each thread's work is spatially localized

**Performance trade-offs:**

- **Fewer logical threads**: May not fully utilize all GPU cores at low occupancy
- **More work per thread**: Better cache utilization and reduced coordination overhead
- **Sequential access**: Optimal memory bandwidth utilization within each thread
- **Reduced overhead**: Less thread launch and coordination overhead

**Important note**: "Fewer threads" refers to the logical programming model. The GPU scheduler can still achieve high hardware utilization by running multiple warps and efficiently switching between them during memory stalls.

#### 6. **Performance characteristics**

**When tiling helps:**

- **Memory-bound operations**: When memory bandwidth is the bottleneck
- **Cache-sensitive workloads**: Operations that benefit from data reuse
- **Complex operations**: When compute per element is higher
- **Limited parallelism**: When you have fewer threads than GPU cores

**When tiling hurts:**

- **Highly parallel workloads**: When you need maximum thread utilization
- **Simple operations**: When memory access dominates over computation
- **Irregular access patterns**: When tiling doesn't improve locality

**For our simple addition example (TILE_SIZE=32):**

- **Thread count**: 32 threads instead of 256 (8 fewer)
- **Work per thread**: 32 elements instead of 4 (8 more)
- **Memory pattern**: Sequential vs strided access
- **Cache utilization**: Much better spatial locality

#### 7. **Advanced tiling considerations**

**Tile size selection:**

- **Too small**: Poor cache utilization, more overhead
- **Too large**: May not fit in cache, reduced parallelism
- **Sweet spot**: Usually 16-64 elements for L1 cache optimization
- **Our choice**: 32 elements balances cache usage with parallelism

**Hardware considerations:**

- **Cache size**: Tiles should fit in L1 cache when possible
- **Memory bandwidth**: Consider memory controller width
- **Core count**: Ensure enough tiles to utilize all cores
- **SIMD width**: Tile size should be multiple of SIMD width

**Comparison summary:**

```
Elementwise: High parallelism, scattered memory access
Tiled:       Moderate parallelism, localized memory access
```

The choice between elementwise and tiled patterns depends on your specific workload characteristics, data access patterns, and target hardware capabilities.

Now that you understand both elementwise and tiled patterns:

- **[Vectorization](#vectorization-fine-grained-simd-control)**: Fine-grained control over SIMD operations
- **[ GPU Threading vs SIMD](#gpu-threading-vs-simd-understanding-the-execution-hierarchy)**: Understanding the execution hierarchy
- **[ Benchmarking](#benchmarking-performance-analysis-and-optimization)**: Performance analysis and optimization

 **Key takeaway**: Tiling demonstrates how memory access patterns often matter more than raw computational throughput. The best GPU code balances parallelism with memory hierarchy optimization.

## Vectorization - Fine-Grained SIMD Control

### Overview

This puzzle explores **advanced vectorization techniques**using manual vectorization and [vectorize](https://docs.modular.com/mojo/stdlib/algorithm/functional/vectorize/) that give you precise control over SIMD operations within GPU kernels. You'll implement two different approaches to vectorized computation:

1. **Manual vectorization**: Direct SIMD control with explicit index calculations
2. **Mojo's vectorize function**: High-level vectorization with automatic bounds checking

Both approaches build on tiling concepts but with different trade-offs between control, safety, and performance optimization.

**Key insight:**_Different vectorization strategies suit different performance requirements and complexity levels._

### Key concepts

In this puzzle, you'll learn:

- **Manual SIMD operations**with explicit index management
- **Mojo's vectorize function**for safe, automatic vectorization
- **Chunk-based memory organization**for optimal SIMD alignment
- **Bounds checking strategies**for edge cases
- **Performance trade-offs**between manual control and safety

The same mathematical operation as before:
\\[\Large \text{output}[i] = a[i] + b[i]\\]

But with sophisticated vectorization strategies for maximum performance.

### Configuration

- Vector size: `SIZE = 1024`
- Tile size: `TILE_SIZE = 32`
- Data type: `DType.float32`
- SIMD width: GPU-dependent
- Layout: `Layout.row_major(SIZE)` (1D row-major)

### 1. Manual vectorization approach

#### Tips

#### 1. **Understanding chunk organization**

```mojo
comptime chunk_size = tile_size * simd_width  # 32 * 4 = 128 elements per chunk
```

Each tile now contains multiple SIMD groups, not just sequential elements.

#### 2. **Global index calculation**

```mojo
global_start = tile_id * chunk_size + i * simd_width
```

This calculates the exact global position for each SIMD vector within the chunk.

#### 3. **Direct tensor access**

```mojo
a_vec = a.loadsimd_width     # Load from global tensor
output.storesimd_width  # Store to global tensor
```

Note: Access the original tensors, not the tile views.

#### 4. **Key characteristics**

- More control, more complexity, global tensor access
- Perfect SIMD alignment with hardware
- Manual bounds checking required

#### Running manual vectorization

  
    pixi NVIDIA (default)
    pixi AMD
    pixi Apple
    uv
  
  

```bash
pixi run p23 --manual-vectorized
```

  
  

```bash
pixi run -e amd p23 --manual-vectorized
```

  
  

```bash
pixi run -e apple p23 --manual-vectorized
```

  
  

```bash
uv run poe p23 --manual-vectorized
```

  

Your output will look like this when not yet solved:

```txt
SIZE: 1024
simd_width: 4
tile size: 32
tile_id: 0
tile_id: 1
tile_id: 2
tile_id: 3
tile_id: 4
tile_id: 5
tile_id: 6
tile_id: 7
out: HostBuffer([0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0])
expected: HostBuffer([1.0, 5.0, 9.0, ..., 4085.0, 4089.0, 4093.0])
```

#### Manual vectorization solution


```mojo
fn manual_vectorized_tiled_elementwise_add
    layout: Layout,
    dtype: DType,
    simd_width: Int,
    num_threads_per_tile: Int,
    rank: Int,
    size: Int,
    tile_size: Int,
 raises:
    # Each tile contains tile_size groups of simd_width elements
    comptime chunk_size = tile_size * simd_width

    @parameter
    @always_inline
    fn process_manual_vectorized_tiles[
        num_threads_per_tile: Int, rank: Int, alignment: Int = align_of[dtype]()
    ](indices: IndexList[rank]) capturing -> None:
        tile_id = indices[0]

        output_tile = output.tilechunk_size
        a_tile = a.tilechunk_size
        b_tile = b.tilechunk_size

        @parameter
        for i in range(tile_size):
            global_start = tile_id * chunk_size + i * simd_width

            a_vec = a.aligned_loadsimd_width)
            b_vec = b.aligned_loadsimd_width)
            ret = a_vec + b_vec
            # print("tile:", tile_id, "simd_group:", i, "global_start:", global_start, "a_vec:", a_vec, "b_vec:", b_vec, "result:", ret)

            output.storesimd_width, ret)

    # Number of tiles needed: each tile processes chunk_size elements
    num_tiles = (size + chunk_size - 1) // chunk_size
    elementwise
        process_manual_vectorized_tiles, num_threads_per_tile, target="gpu"
    

```

#### Manual vectorization deep dive

**Manual vectorization**gives you direct control over SIMD operations with explicit index calculations:

- **Chunk-based organization**: `chunk_size = tile_size * simd_width`
- **Global indexing**: Direct calculation of memory positions
- **Manual bounds management**: You handle edge cases explicitly

**Architecture and memory layout:**

```mojo
comptime chunk_size = tile_size * simd_width  # 32 * 4 = 128
```

**Chunk organization visualization (TILE_SIZE=32, SIMD_WIDTH=4):**

```
Original array: [0, 1, 2, 3, ..., 1023]

Chunk 0 (thread 0): [0:128]    <- 128 elements = 32 SIMD groups of 4
Chunk 1 (thread 1): [128:256]  <- Next 128 elements
Chunk 2 (thread 2): [256:384]  <- Next 128 elements
...
Chunk 7 (thread 7): [896:1024] <- Final 128 elements
```

**Processing within one chunk:**

```mojo
@parameter
for i in range(tile_size):  # i = 0, 1, 2, ..., 31
    global_start = tile_id * chunk_size + i * simd_width
    # For tile_id=0: global_start = 0, 4, 8, 12, ..., 124
    # For tile_id=1: global_start = 128, 132, 136, 140, ..., 252
```

**Performance characteristics:**

- **Thread count**: 8 threads (1024  128 = 8)
- **Work per thread**: 128 elements (32 SIMD operations of 4 elements each)
- **Memory pattern**: Large chunks with perfect SIMD alignment
- **Overhead**: Minimal - direct hardware mapping
- **Safety**: Manual bounds checking required

**Key advantages:**

- **Predictable indexing**: Exact control over memory access patterns
- **Optimal alignment**: SIMD operations perfectly aligned to hardware
- **Maximum throughput**: No overhead from safety checks
- **Hardware optimization**: Direct mapping to GPU SIMD units

**Key challenges:**

- **Index complexity**: Manual calculation of global positions
- **Bounds responsibility**: Must handle edge cases explicitly
- **Debugging difficulty**: More complex to verify correctness

### 2. Mojo vectorize approach

#### Tips

#### 1. **Tile boundary calculation**

```mojo
tile_start = tile_id * tile_size
tile_end = min(tile_start + tile_size, size)
actual_tile_size = tile_end - tile_start
```

Handle cases where the last tile might be smaller than `tile_size`.

#### 2. **Vectorized function pattern**

```mojo
fn vectorized_add
  width: Int
 unified {read tile_start, read a, read b, mut output}:
    global_idx = tile_start + i
    if global_idx + width <= size:  # Bounds checking
        # SIMD operations here
```

The `width` parameter is automatically determined by the vectorize function.

#### 3. **Calling vectorize**

```mojo
vectorizesimd_width
```

This automatically handles the vectorization loop with the provided SIMD width.

#### 4. **Key characteristics**

- Automatic remainder handling, built-in safety, tile-based access
- Takes explicit SIMD width parameter
- Built-in bounds checking and automatic remainder element processing

#### Running Mojo vectorize

  
    uv
    pixi
  
  

```bash
uv run poe p23 --vectorized
```

  
  

```bash
pixi run p23 --vectorized
```

  

Your output will look like this when not yet solved:

```txt
SIZE: 1024
simd_width: 4
tile size: 32
tile_id: 0 tile_start: 0 tile_end: 32 actual_tile_size: 32
tile_id: 1 tile_start: 32 tile_end: 64 actual_tile_size: 32
tile_id: 2 tile_start: 64 tile_end: 96 actual_tile_size: 32
tile_id: 3 tile_start: 96 tile_end: 128 actual_tile_size: 32
...
tile_id: 29 tile_start: 928 tile_end: 960 actual_tile_size: 32
tile_id: 30 tile_start: 960 tile_end: 992 actual_tile_size: 32
tile_id: 31 tile_start: 992 tile_end: 1024 actual_tile_size: 32
out: HostBuffer([0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0])
expected: HostBuffer([1.0, 5.0, 9.0, ..., 4085.0, 4089.0, 4093.0])
```

#### Mojo vectorize solution


```mojo
fn vectorize_within_tiles_elementwise_add
    layout: Layout,
    dtype: DType,
    simd_width: Int,
    num_threads_per_tile: Int,
    rank: Int,
    size: Int,
    tile_size: Int,
 raises:
    # Each tile contains tile_size elements (not SIMD groups)
    @parameter
    @always_inline
    fn process_tile_with_vectorize[
        num_threads_per_tile: Int, rank: Int, alignment: Int = align_of[dtype]()
    ](indices: IndexList[rank]) capturing -> None:
        tile_id = indices[0]
        tile_start = tile_id * tile_size
        tile_end = min(tile_start + tile_size, size)
        actual_tile_size = tile_end - tile_start

        fn vectorized_add
            width: Int
         unified {read tile_start, read a, read b, mut output}:
            global_idx = tile_start + i
            if global_idx + width <= size:
                a_vec = a.aligned_loadwidth)
                b_vec = b.aligned_loadwidth)
                result = a_vec + b_vec
                output.storewidth, result)

        # Use vectorize within each tile
        vectorizesimd_width

    num_tiles = (size + tile_size - 1) // tile_size
    elementwise
        process_tile_with_vectorize, num_threads_per_tile, target="gpu"
    

```

#### Mojo vectorize deep dive

**Mojo's vectorize function**provides automatic vectorization with built-in safety:

- **Explicit SIMD width parameter**: You provide the simd_width to use
- **Built-in bounds checking**: Prevents buffer overruns automatically
- **Automatic remainder handling**: Processes leftover elements automatically
- **Nested function pattern**: Clean separation of vectorization logic

**Tile-based organization:**

```mojo
tile_start = tile_id * tile_size    # 0, 32, 64, 96, ...
tile_end = min(tile_start + tile_size, size)
actual_tile_size = tile_end - tile_start
```

**Automatic vectorization mechanism:**

```mojo
fn vectorized_add
  width: Int
 unified {read tile_start, read a, read b, mut output}:
    global_idx = tile_start + i
    if global_idx + width <= size:
        # Automatic SIMD optimization
```

**How vectorize works:**

- **Automatic chunking**: Divides `actual_tile_size` into chunks of your provided `simd_width`
- **Remainder handling**: Automatically processes leftover elements with smaller widths
- **Bounds safety**: Automatically prevents buffer overruns
- **Loop management**: Handles the vectorization loop automatically

**Execution visualization (TILE_SIZE=32, SIMD_WIDTH=4):**

```
Tile 0 processing:
  vectorize call 0: processes elements [0:4]   with SIMD_WIDTH=4
  vectorize call 1: processes elements [4:8]   with SIMD_WIDTH=4
  ...
  vectorize call 7: processes elements [28:32] with SIMD_WIDTH=4
  Total: 8 automatic SIMD operations
```

**Performance characteristics:**

- **Thread count**: 32 threads (1024  32 = 32)
- **Work per thread**: 32 elements (automatic SIMD chunking)
- **Memory pattern**: Smaller tiles with automatic vectorization
- **Overhead**: Slight - automatic optimization and bounds checking
- **Safety**: Built-in bounds checking and edge case handling

### Performance comparison and best practices

#### When to use each approach

**Choose manual vectorization when:**

- **Maximum performance**is critical
- You have **predictable, aligned data**patterns
- **Expert-level control**over memory access is needed
- You can **guarantee bounds safety**manually
- **Hardware-specific optimization**is required

**Choose Mojo vectorize when:**

- **Development speed**and safety are priorities
- Working with **irregular or dynamic data sizes**
- You want **automatic remainder handling**instead of manual edge case management
- **Bounds checking**complexity would be error-prone
- You prefer **cleaner vectorization patterns**over manual loop management

#### Advanced optimization insights

**Memory bandwidth utilization:**

```
Manual:    8 threads x 32 SIMD ops = 256 total SIMD operations
Vectorize: 32 threads x 8 SIMD ops = 256 total SIMD operations
```

Both achieve similar total throughput but with different parallelism strategies.

**Cache behavior:**

- **Manual**: Large chunks may exceed L1 cache, but perfect sequential access
- **Vectorize**: Smaller tiles fit better in cache, with automatic remainder handling

**Hardware mapping:**

- **Manual**: Direct control over warp utilization and SIMD unit mapping
- **Vectorize**: Simplified vectorization with automatic loop and remainder management

#### Best practices summary

**Manual vectorization best practices:**

- Always validate index calculations carefully
- Use compile-time constants for `chunk_size` when possible
- Profile memory access patterns for cache optimization
- Consider alignment requirements for optimal SIMD performance

**Mojo vectorize best practices:**

- Choose appropriate SIMD width for your data and hardware
- Focus on algorithm clarity over micro-optimizations
- Use nested parameter functions for clean vectorization logic
- Trust automatic bounds checking and remainder handling for edge cases

Both approaches represent valid strategies in the GPU performance optimization toolkit, with manual vectorization offering maximum control and Mojo's vectorize providing safety and automatic remainder handling.

Now that you understand all three fundamental patterns:

- **[ GPU Threading vs SIMD](#gpu-threading-vs-simd-understanding-the-execution-hierarchy)**: Understanding the execution hierarchy
- **[ Benchmarking](#benchmarking-performance-analysis-and-optimization)**: Performance analysis and optimization

 **Key takeaway**: Different vectorization strategies suit different performance requirements. Manual vectorization gives maximum control, while Mojo's vectorize function provides safety and automatic remainder handling. Choose based on your specific performance needs and development constraints.

##  GPU Threading vs SIMD - Understanding the Execution Hierarchy

### Overview

After exploring **elementwise**, **tiled**, and **vectorization**patterns, you've seen different ways to organize GPU computation. This section clarifies the fundamental relationship between **GPU threads**and **SIMD operations**- two distinct but complementary levels of parallelism that work together for optimal performance.

> **Key insight:**_GPU threads provide the parallelism structure, while SIMD operations provide the vectorization within each thread._

### Core concepts

#### GPU threading hierarchy

GPU execution follows a well-defined hierarchy that abstracts hardware complexity:

```
GPU Device
|-- Grid (your entire problem)
|   |-- Block 1 (group of threads, shared memory)
|   |   |-- Warp 1 (32 threads, lockstep execution)
|   |   |   |-- Thread 1 -> SIMD operations
|   |   |   |-- Thread 2 -> SIMD operations
|   |   |   `-- ... (32 threads total)
|   |   `-- Warp 2 (32 threads)
|   `-- Block 2 (independent group)
```

 **Note**: While this Part focuses on functional patterns, **warp-level programming**and advanced GPU memory management will be covered in detail in **Part VII**.

**What Mojo abstracts for you:**
- **Grid/Block configuration**: Automatically calculated based on problem size
- **Warp management**: Hardware handles 32-thread groups transparently
- **Thread scheduling**: GPU scheduler manages execution automatically
- **Memory hierarchy**: Optimal access patterns built into functional operations

#### SIMD within GPU threads

Each GPU thread can process multiple data elements simultaneously using **SIMD (Single Instruction, Multiple Data)**operations:

```mojo
# Within one GPU thread:
a_simd = a.loadsimd_width      # Load 4 floats simultaneously
b_simd = b.loadsimd_width      # Load 4 floats simultaneously
result = a_simd + b_simd                 # Add 4 pairs simultaneously
output.storesimd_width # Store 4 results simultaneously
```

### Pattern comparison and thread-to-work mapping

> **Critical insight:**All patterns perform the **same total work**- 256 SIMD operations for 1024 elements with SIMD_WIDTH=4. The difference is in how this work is distributed across GPU threads.

#### Thread organization comparison (`SIZE=1024`, `SIMD_WIDTH=4`)

| Pattern | Threads | SIMD ops/thread | Memory pattern | Trade-off |
|---------|---------|-----------------|----------------|-----------|
| **Elementwise**| 256 | 1 | Distributed access | Max parallelism, poor locality |
| **Tiled**| 32 | 8 | Small blocks | Balanced parallelism + locality |
| **Manual vectorized**| 8 | 32 | Large chunks | High bandwidth, fewer threads |
| **Mojo vectorize**| 32 | 8 | Smart blocks | Automatic optimization |

#### Detailed execution patterns

**Elementwise pattern:**
```
Thread 0: [0,1,2,3] -> Thread 1: [4,5,6,7] -> ... -> Thread 255: [1020,1021,1022,1023]
256 threads x 1 SIMD op = 256 total SIMD operations
```

**Tiled pattern:**
```
Thread 0: [0:32] (8 SIMD) -> Thread 1: [32:64] (8 SIMD) -> ... -> Thread 31: [992:1024] (8 SIMD)
32 threads x 8 SIMD ops = 256 total SIMD operations
```

**Manual vectorized pattern:**
```
Thread 0: [0:128] (32 SIMD) -> Thread 1: [128:256] (32 SIMD) -> ... -> Thread 7: [896:1024] (32 SIMD)
8 threads x 32 SIMD ops = 256 total SIMD operations
```

**Mojo vectorize pattern:**
```
Thread 0: [0:32] auto-vectorized -> Thread 1: [32:64] auto-vectorized -> ... -> Thread 31: [992:1024] auto-vectorized
32 threads x 8 SIMD ops = 256 total SIMD operations
```

### Performance characteristics and trade-offs

#### Core trade-offs summary

| Aspect | High thread count (Elementwise) | Moderate threads (Tiled/Vectorize) | Low threads (Manual) |
|--------|--------------------------------|-----------------------------------|----------------------|
| **Parallelism**| Maximum latency hiding | Balanced approach | Minimal parallelism |
| **Cache locality**| Poor between threads | Good within tiles | Excellent sequential |
| **Memory bandwidth**| Good coalescing | Good + cache reuse | Maximum theoretical |
| **Complexity**| Simplest | Moderate | Most complex |

#### When to choose each pattern

**Use elementwise when:**
- Simple operations with minimal arithmetic per element
- Maximum parallelism needed for latency hiding
- Scalability across different problem sizes is important

**Use tiled/vectorize when:**
- Cache-sensitive operations that benefit from data reuse
- Balanced performance and maintainability desired
- Automatic optimization (vectorize) is preferred

**Use manual vectorization when:**
- Expert-level control over memory patterns is needed
- Maximum memory bandwidth utilization is critical
- Development complexity is acceptable

### Hardware considerations

Modern GPU architectures include several levels that Mojo abstracts:

**Hardware reality:**
- **Warps**: 32 threads execute in lockstep
- **Streaming Multiprocessors (SMs)**: Multiple warps execute concurrently
- **SIMD units**: Vector processing units within each SM
- **Memory hierarchy**: L1/L2 caches, shared memory, global memory

**Mojo's abstraction benefits:**
- Automatically handles warp alignment and scheduling
- Optimizes memory access patterns transparently
- Manages resource allocation across SMs
- Provides portable performance across GPU vendors

### Performance mental model

Think of GPU programming as managing two complementary types of parallelism:

**Thread-level parallelism:**
- Provides the parallel structure (how many execution units)
- Enables latency hiding through concurrent execution
- Managed by GPU scheduler automatically

**SIMD-level parallelism:**
- Provides vectorization within each thread
- Maximizes arithmetic throughput per thread
- Utilizes vector processing units efficiently

**Optimal performance formula:**
```
Performance = (Sufficient threads for latency hiding) x
              (Efficient SIMD utilization) x
              (Optimal memory access patterns)
```

### Scaling considerations

| Problem size | Optimal pattern | Reasoning |
|-------------|----------------|-----------|
| Small (< 1K) | Tiled/Vectorize | Lower launch overhead |
| Medium (1K-1M) | Any pattern | Similar performance |
| Large (> 1M) | Usually Elementwise | Parallelism dominates |

The optimal choice depends on your specific hardware, workload complexity, and development constraints.

With a solid understanding of GPU threading vs SIMD concepts:

- **[ Benchmarking](#benchmarking-performance-analysis-and-optimization)**: Measure and compare actual performance

 **Key takeaway**: GPU threads and SIMD operations work together as complementary levels of parallelism. Understanding their relationship allows you to choose the right pattern for your specific performance requirements and constraints.

##  Benchmarking - Performance Analysis and Optimization

### Overview

After learning **elementwise**, **tiled**, **manual vectorization**, and **Mojo vectorize**patterns, it's time to measure their actual performance. Here's how to use the built-in benchmarking system in `p21.mojo` to scientifically compare these approaches and understand their performance characteristics.

> **Key insight:**_Theoretical analysis is valuable, but empirical benchmarking reveals the true performance story on your specific hardware._

### Running benchmarks

To execute the comprehensive benchmark suite:

  
    pixi NVIDIA (default)
    pixi AMD
    pixi Apple
    uv
  
  

```bash
pixi run p23 --benchmark
```

  
  

```bash
pixi run -e amd p23 --benchmark
```

  
  

```bash
pixi run -e apple p23 --benchmark
```

  
  

```bash
uv run poe p23 --benchmark
```

  

Your output will show performance measurements for each pattern:

```txt
SIZE: 1024
simd_width: 4
Running P21 GPU Benchmarks...
SIMD width: 4
--------------------------------------------------------------------------------
Testing SIZE=16, TILE=4
Running elementwise_16_4
Running tiled_16_4
Running manual_vectorized_16_4
Running vectorized_16_4
--------------------------------------------------------------------------------
Testing SIZE=128, TILE=16
Running elementwise_128_16
Running tiled_128_16
Running manual_vectorized_128_16
--------------------------------------------------------------------------------
Testing SIZE=128, TILE=16, Vectorize within tiles
Running vectorized_128_16
--------------------------------------------------------------------------------
Testing SIZE=1048576 (1M), TILE=1024
Running elementwise_1M_1024
Running tiled_1M_1024
Running manual_vectorized_1M_1024
Running vectorized_1M_1024
| name                      | met (ms)              | iters |
| ------------------------- | --------------------- | ----- |
| elementwise_16_4          | 0.0033248             | 100   |
| tiled_16_4                | 0.00327392            | 100   |
| manual_vectorized_16_4    | 0.0036169600000000002 | 100   |
| vectorized_16_4           | 0.0037209599999999997 | 100   |
| elementwise_128_16        | 0.00351999            | 100   |
| tiled_128_16              | 0.00370431            | 100   |
| manual_vectorized_128_16  | 0.0043696             | 100   |
| vectorized_128_16         | 0.00378048            | 100   |
| elementwise_1M_1024       | 0.03130143            | 100   |
| tiled_1M_1024             | 0.6892189000000001    | 100   |
| manual_vectorized_1M_1024 | 0.5923888             | 100   |
| vectorized_1M_1024        | 0.1876688             | 100   |

Benchmarks completed!
```

### Benchmark configuration

The benchmarking system uses Mojo's built-in `benchmark` module:

```mojo
from benchmark import Bench, BenchConfig, Bencher, BenchId, keep
bench_config = BenchConfig(max_iters=10, num_warmup_iters=1)
```

- **`max_iters=10`**: Up to 10 iterations for statistical reliability
- **`num_warmup_iters=1`**: GPU warmup before measurement
- Check out the [benchmark documentation](https://docs.modular.com/mojo/stdlib/benchmark/)

### Benchmarking implementation essentials

#### Core workflow pattern

Each benchmark follows a streamlined pattern:

```mojo
@parameter
fn benchmark_pattern_parameterizedtest_size: Int, tile_size: Int raises:
    bench_ctx = DeviceContext()
    # Setup: Create buffers and initialize data
    @parameter
    fn pattern_workflow(ctx: DeviceContext) raises:
      # Compute: Execute the algorithm being measured

    b.iter_custompattern_workflow
    # Prevent optimization: keep(out.unsafe_ptr())
    # Synchronize: ctx.synchronize()
```

**Key phases:**

1. **Setup**: Buffer allocation and data initialization
2. **Computation**: The actual algorithm being benchmarked
3. **Prevent optimization**: Critical for accurate measurement
4. **Synchronization**: Ensure GPU work completes

> **Critical: The `keep()` function**
> `keep(out.unsafe_ptr())` prevents the compiler from optimizing away your computation as "unused code." Without this, you might measure nothing instead of your algorithm! This is essential for accurate GPU benchmarking because kernels are launched asynchronously.

#### Why custom iteration works for GPU

Standard benchmarking assumes CPU-style synchronous execution. GPU kernels launch asynchronously, so we need:

- **GPU context management**: Proper DeviceContext lifecycle
- **Memory management**: Buffer cleanup between iterations
- **Synchronization handling**: Accurate timing of async operations
- **Overhead isolation**: Separate setup cost from computation cost

### Test scenarios and thread analysis

The benchmark suite tests three scenarios to reveal performance characteristics:

#### Thread utilization summary

| Problem Size | Pattern | Threads | SIMD ops/thread | Total SIMD ops |
|-------------|---------|---------|-----------------|----------------|
| **SIZE=16**| Elementwise | 4 | 1 | 4 |
|             | Tiled | 4 | 1 | 4 |
|             | Manual | 1 | 4 | 4 |
|             | Vectorize | 4 | 1 | 4 |
| **SIZE=128**| Elementwise | 32 | 1 | 32 |
|              | Tiled | 8 | 4 | 32 |
|              | Manual | 2 | 16 | 32 |
|              | Vectorize | 8 | 4 | 32 |
| **SIZE=1M**| Elementwise | 262,144 | 1 | 262,144 |
|             | Tiled | 1,024 | 256 | 262,144 |
|             | Manual | 256 | 1,024 | 262,144 |
|             | Vectorize | 1,024 | 256 | 262,144 |

#### Performance characteristics by problem size

**Small problems (SIZE=16):**

- Launch overhead dominates (~0.003ms baseline)
- Thread count differences don't matter
- Tiled/vectorize show slightly lower overhead

**Medium problems (SIZE=128):**

- Still overhead-dominated (~0.003ms for all)
- Performance differences nearly disappear
- Transitional behaviour between overhead and computation

**Large problems (SIZE=1M):**

- Real algorithmic differences emerge
- Impact of uncoalesced loads becomes apparent
- Clear performance ranking appears

### What the data shows

Based on empirical benchmark results across different hardware:

#### Performance rankings (large problems)

| Rank | Pattern | Typical time | Key insight |
|------|---------|-------------|-------------|
|  | **Elementwise**| ~0.03ms | Coalesced memory access wins for memory-bound ops |
|  | **Mojo vectorize**| ~0.19ms | Uncoalesced memory access hurts performance |
|  | **Manual vectorized**| ~0.59ms | Uncoalesced memory access and manual optimization reduces performance |
| 4th | **Tiled**| ~0.69ms | Uncoalesced memory access, manual optimization without SIMD loads reduces performance further |

#### Key performance insights

> **For simple memory-bound operations:**Maximum parallelism (elementwise) outperforms complex memory optimizations at scale.

**Why elementwise wins:**

- **262,144 threads**provide excellent latency hiding
- **Simple memory patterns**achieve good coalescing
- **Minimal overhead**per thread
- **Scales naturally**with GPU core count

**Why tiled and vectorize are competitive:**

- **Balanced approach**between parallelism and memory locality
- **Automatic optimization**(vectorize) performs nearly as well as manual tiling
- **Good thread utilization**without excessive complexity

**Why manual vectorization struggles:**

- **Only 256 threads**limit parallelism
- **Complex indexing**adds computational overhead
- **Cache pressure**from large chunks per thread
- **Diminishing returns**for simple arithmetic

**Framework intelligence:**

- Automatic iteration count adjustment (91-100 iterations)
- Statistical reliability across different execution times
- Handles thermal throttling and system variation

### Interpreting your results

#### Reading the output table

```txt
| name                     | met (ms)           | iters |
| elementwise_1M_1024      | 0.03130143         | 100   |
```

- **`met (ms)`**: Execution time for a single iteration
- **`iters`**: Number of iterations performed
- **Compare within problem size**: Same-size comparisons are most meaningful

#### Making optimization decisions

**Choose patterns based on empirical evidence:**

**For production workloads:**

- **Large datasets (>100K elements)**: Elementwise typically optimal
- **Small/startup datasets (<1K elements)**: Tiled or vectorize for lower overhead
- **Development speed priority**: Mojo vectorize for automatic optimization
- **Avoid manual vectorization**: Complexity rarely pays off for simple operations

**Performance optimization workflow:**

1. **Profile first**: Measure before optimizing
2. **Test at scale**: Small problems mislead about real performance
3. **Consider total cost**: Include development and maintenance effort
4. **Validate improvements**: Confirm with benchmarks on target hardware

### Advanced benchmarking techniques

#### Custom test scenarios

Modify parameters to test different conditions:

```mojo
# Different problem sizes
benchmark_elementwise_parameterized[1024, 32]  # Large problem
benchmark_elementwise_parameterized[64, 8]     # Small problem

# Different tile sizes
benchmark_tiled_parameterized[256, 8]   # Small tiles
benchmark_tiled_parameterized[256, 64]  # Large tiles
```

#### Hardware considerations

Your results will vary based on:

- **GPU architecture**: SIMD width, core count, memory bandwidth
- **System configuration**: PCIe bandwidth, CPU performance
- **Thermal state**: GPU boost clocks vs sustained performance
- **Concurrent workloads**: Other processes affecting GPU utilization

### Best practices summary

**Benchmarking workflow:**

1. **Warm up GPU**before critical measurements
2. **Run multiple iterations**for statistical significance
3. **Test multiple problem sizes**to understand scaling
4. **Use `keep()` consistently**to prevent optimization artifacts
5. **Compare like with like**(same problem size, same hardware)

**Performance decision framework:**

- **Start simple**: Begin with elementwise for memory-bound operations
- **Measure don't guess**: Theoretical analysis guides, empirical data decides
- **Scale matters**: Small problem performance doesn't predict large problem behaviour
- **Total cost optimization**: Balance development time vs runtime performance

With benchmarking skills:

- **Profile real applications**: Apply these patterns to actual workloads
- **Advanced GPU patterns**: Explore reductions, convolutions, and matrix operations
- **Multi-GPU scaling**: Understand distributed GPU computing patterns
- **Memory optimization**: Dive deeper into shared memory and advanced caching

 **Key takeaway**: Benchmarking transforms theoretical understanding into practical performance optimization. Use empirical data to make informed decisions about which patterns work best for your specific hardware and workload characteristics.

### Looking ahead: when you need more control

The functional patterns in Part V provide excellent performance for most workloads, but some algorithms require **direct thread communication**:

#### **Algorithms that benefit from warp programming:**

- **Reductions**: Sum, max, min operations across thread groups
- **Prefix operations**: Cumulative sums, running maximums
- **Data shuffling**: Reorganizing data between threads
- **Cooperative algorithms**: Where threads must coordinate closely

#### **Performance preview:**

In Part VI, we'll revisit several algorithms from Part II and show how warp operations can:

- **Simplify code**: Replace complex shared memory patterns with single function calls
- **Improve performance**: Eliminate barriers and reduce memory traffic
- **Enable new algorithms**: Unlock patterns impossible with pure functional approaches
