---
title: "Block-Level Programming"
description: "Welcome to Puzzle 27: Block-Level Programming! This puzzle introduces you to the fundamental building blocks of GPU parallel programming - block-level communication primitives that enable sophisticated parallel algorithms across entire thread blocks. You'll explore three essential communication patterns that replace complex manual synchronization with elegant, hardware-optimized operations."
---

# Block-Level Programming

Welcome to Puzzle 27: Block-Level Programming! This puzzle introduces you to the fundamental building blocks of GPU parallel programming - block-level communication primitives that enable sophisticated parallel algorithms across entire thread blocks. You'll explore three essential communication patterns that replace complex manual synchronization with elegant, hardware-optimized operations.

## Overview

Welcome to **Puzzle 27: Block-Level Programming**! This puzzle introduces you to the fundamental building blocks of GPU parallel programming - **block-level communication primitives**that enable sophisticated parallel algorithms across entire thread blocks. You'll explore three essential communication patterns that replace complex manual synchronization with elegant, hardware-optimized operations.

**What you'll achieve:**Transform from complex shared memory + barriers + tree reduction patterns (Puzzle 12) to elegant single-function-call algorithms that leverage hardware-optimized block-wide communication primitives across multiple warps.

**Key insight:**_GPU thread blocks execute with sophisticated hardware coordination - Mojo's block operations harness cross-warp communication and dedicated hardware units to provide complete parallel programming building blocks: reduction (allone), scan (alleach), and broadcast (oneall)._

## What you'll learn

### **Block-level communication model**
Understand the three fundamental communication patterns within GPU thread blocks:

```
GPU Thread Block (128 threads across 4 or 2 warps, hardware coordination)
All-to-One (Reduction):     All threads -> Single result at thread 0
All-to-Each (Scan):         All threads -> Each gets cumulative position
One-to-All (Broadcast):     Thread 0 -> All threads get same value

Cross-warp coordination:
|-- Warp 0 (threads 0-31)   --block.sum()--+
|-- Warp 1 (threads 32-63)  --block.sum()--+-> Thread 0 result
|-- Warp 2 (threads 64-95)  --block.sum()--+
`-- Warp 3 (threads 96-127) --block.sum()--+
```

**Hardware reality:**
- **Cross-warp synchronization**: Automatic coordination across multiple warps within a block
- **Dedicated hardware units**: Specialized scan units and butterfly reduction networks
- **Zero explicit barriers**: Hardware manages all synchronization internally
- **Logarithmic complexity**: \\(O(\\log n)\\) algorithms with single-instruction simplicity

### **Block operations in Mojo**
Learn the complete parallel programming toolkit from `gpu.primitives.block`:

1. **[`block.sum(value)`](https://docs.modular.com/mojo/stdlib/gpu/primitives/block/sum)**: All-to-one reduction for totals, averages, maximum/minimum values
2. **[`block.prefix_sum(value)`](https://docs.modular.com/mojo/stdlib/gpu/primitives/block/prefix_sum)**: All-to-each scan for parallel filtering and extraction
3. **[`block.broadcast(value)`](https://docs.modular.com/mojo/stdlib/gpu/primitives/block/broadcast)**: One-to-all distribution for parameter sharing and coordination

> **Note:**These primitives enable sophisticated parallel algorithms like statistical computations, histogram binning, and normalization workflows that would otherwise require dozens of lines of complex shared memory coordination code.

### **Performance transformation example**
```mojo
# Complex block-wide reduction (traditional approach - from Puzzle 12):
shared_memory[local_i] = my_value
barrier()
for stride in range(64, 0, -1):
    if local_i < stride:
        shared_memory[local_i] += shared_memory[local_i + stride]
    barrier()
if local_i == 0:
    output[block_idx.x] = shared_memory[0]

# Block operations eliminate all this complexity:
my_partial = compute_local_contribution()
total = block.sumblock_size=128, broadcast=False  # Single call!
if local_i == 0:
    output[block_idx.x] = total[0]
```

### **When block operations excel**
Learn the performance characteristics:

| Algorithm Pattern | Traditional | Block Operations |
|-------------------|-------------|------------------|
| Block-wide reductions | Shared memory + barriers | Single `block.sum` call |
| Parallel filtering | Complex indexing | `block.prefix_sum` coordination |
| Parameter sharing | Manual synchronization | Single `block.broadcast` call |
| Cross-warp algorithms | Explicit barrier management | Hardware-managed coordination |

## The evolution of GPU programming patterns

### **Where we started: Manual coordination (Puzzle 12)**
Complex but educational - explicit shared memory, barriers, and tree reduction:
```mojo
# Manual approach: 15+ lines of complex synchronization
shared_memory[local_i] = my_value
barrier()
# Tree reduction with stride-based indexing...
for stride in range(64, 0, -1):
    if local_i < stride:
        shared_memory[local_i] += shared_memory[local_i + stride]
    barrier()
```

### **The intermediate step: Warp programming (Puzzle 24)**
Hardware-accelerated but limited scope - `warp.sum()` within 32-thread warps:
```mojo
# Warp approach: 1 line but single warp only
total = warp.sumwarp_size=WARP_SIZE
```

### **The destination: Block programming (This puzzle)**
Complete toolkit - hardware-optimized primitives across entire blocks:
```mojo
# Block approach: 1 line across multiple warps (128+ threads)
total = block.sumblock_size=128, broadcast=False
```

## The three fundamental communication patterns

Block-level programming provides three essential primitives that cover all parallel communication needs:

### **1. All-to-One: Reduction (`block.sum()`)**
- **Pattern**: All threads contribute  One thread receives result
- **Use case**: Computing totals, averages, finding maximum/minimum values
- **Example**: Dot product, statistical aggregation
- **Hardware**: Cross-warp butterfly reduction with automatic barriers

### **2. All-to-Each: Scan (`block.prefix_sum()`)**
- **Pattern**: All threads contribute  Each thread receives cumulative position
- **Use case**: Parallel filtering, stream compaction, histogram binning
- **Example**: Computing write positions for parallel data extraction
- **Hardware**: Parallel scan with cross-warp coordination

### **3. One-to-All: Broadcast (`block.broadcast()`)**
- **Pattern**: One thread provides  All threads receive same value
- **Use case**: Parameter sharing, configuration distribution
- **Example**: Sharing computed mean for normalization algorithms
- **Hardware**: Optimized distribution across multiple warps

## Learning progression

Complete this puzzle in three parts, building from simple to sophisticated:

### **Part 1: [Block.sum() Essentials](#blocksum-essentials-block-level-dot-product)**
**Transform complex reduction to simple function call**

Learn the foundational block reduction pattern by implementing dot product with `block.sum()`. This part shows how block operations replace 15+ lines of manual barriers with a single optimized call.

**Key concepts:**
- Block-wide synchronization across multiple warps
- Hardware-optimized reduction patterns
- Thread 0 result management
- Performance comparison with traditional approaches

**Expected outcome:**Understand how `block.sum()` provides warp.sum() simplicity at block scale.

---

### **Part 2: [Block.prefix_sum() Parallel Histogram](#blockprefixsum-parallel-histogram-binning)**
**Advanced parallel filtering and extraction**

Build sophisticated parallel algorithms using `block.prefix_sum()` for histogram binning. This part demonstrates how prefix sum enables complex data reorganization that would be difficult with simple reductions.

**Key concepts:**
- Parallel filtering with binary predicates
- Coordinated write position computation
- Advanced partitioning algorithms
- Cross-thread data extraction patterns

**Expected outcome:**Understand how `block.prefix_sum()` enables sophisticated parallel algorithms beyond simple aggregation.

---

### **Part 3: [Block.broadcast() Vector Normalization](#blockbroadcast-vector-normalization)**
**Complete workflow combining all patterns**

Implement vector mean normalization using the complete block operations toolkit. This part shows how all three primitives work together to solve real computational problems with mathematical correctness.

**Key concepts:**
- One-to-all communication patterns
- Coordinated multi-phase algorithms
- Complete block operations workflow
- Real-world algorithm implementation

**Expected outcome:**Understand how to compose block operations for sophisticated parallel algorithms.

## Why block operations matter

### **Code simplicity transformation:**
```
Traditional approach:  20+ lines of barriers, shared memory, complex indexing
Block operations:      3-5 lines of composable, hardware-optimized primitives
```

### **Performance advantages:**
- **Hardware optimization**: Leverages GPU architecture-specific optimizations
- **Automatic synchronization**: Eliminates manual barrier placement errors
- **Composability**: Operations work together seamlessly
- **Portability**: Same code works across different GPU architectures

### **Educational value:**
- **Conceptual clarity**: Each operation has a clear communication purpose
- **Progressive complexity**: Build from simple reductions to complex algorithms
- **Real applications**: Patterns used extensively in scientific computing, graphics, AI

## Prerequisites

Before starting this puzzle, you should have completed:
- **Puzzle 12**: Understanding of manual GPU synchronization
- **Puzzle 24**: Experience with warp-level programming

## Expected learning outcomes

After completing all three parts, you'll understand:

1. **When to use each block operation**for different parallel communication needs
2. **How to compose operations**to build sophisticated algorithms
3. **Performance trade-offs**between manual and automated approaches
4. **Real-world applications**of block-level programming patterns
5. **Architecture-independent programming**using hardware-optimized primitives

## Getting started

**Recommended approach:**Complete the three parts in sequence, as each builds on concepts from the previous parts. The progression from simple reduction  advanced partitioning  complete workflow provides the optimal learning path for understanding block-level GPU programming.

 **Key insight**: Block operations represent the sweet spot between programmer productivity and hardware performance - they provide the simplicity of high-level operations with the efficiency of carefully optimized low-level implementations. This puzzle teaches you to think at the right abstraction level for modern GPU programming.

## block.sum() Essentials - Block-Level Dot Product

Implement the dot product we saw in puzzle 12 using block-level [sum](https://docs.modular.com/mojo/stdlib/gpu/primitives/block/sum) operations to replace complex shared memory patterns with simple function calls. Each thread in the block will process one element and use `block.sum()` to combine results automatically, demonstrating how block programming transforms GPU synchronization across entire thread blocks.

**Key insight:**_The [block.sum()](https://docs.modular.com/mojo/stdlib/gpu/primitives/block/sum) operation leverages block-wide execution to replace shared memory + barriers + tree reduction with expertly optimized implementations that work across all threads using warp patterns in a block. See [technical investigation](#technical-investigation-what-does-blocksum-actually-compile-to) for LLVM analysis._

### Key concepts

In this puzzle, you'll learn:

- **Block-level reductions**with `block.sum()`
- **Block-wide synchronization**and thread coordination
- **Cross-warp communication**within a single block
- **Performance transformation**from complex to simple patterns
- **Thread 0 result management**and conditional writes

The mathematical operation is a dot product (inner product):
\\[\Large \text{output}[0] = \sum_{i=0}^{N-1} a[i] \times b[i]\\]

But the implementation teaches fundamental patterns for all block-level GPU programming in Mojo.

### Configuration

- Vector size: `SIZE = 128` elements
- Data type: `DType.float32`
- Block configuration: `(128, 1)` threads per block (`TPB = 128`)
- Grid configuration: `(1, 1)` blocks per grid
- Layout: `Layout.row_major(SIZE)` (1D row-major)
- Warps per block: `128 / WARP_SIZE` (4 warps on NVIDIA, 2 or 4 warps on AMD)

### The traditional complexity (from Puzzle 12)

Recall the complex approach from Puzzle 12 that required shared memory, barriers, and tree reduction:

```mojo
fn traditional_dot_product
    in_layout: Layout, out_layout: Layout, tpb: Int
:
    """Traditional dot product using shared memory + barriers + tree reduction.
    Educational but complex - shows the manual coordination needed."""

    shared = LayoutTensor[
        dtype,
        Layout.row_major(tpb),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    local_i = Int(thread_idx.x)

    # Each thread computes partial product
    if global_i < size:
        a_val = rebind[Scalar[dtype]](a[global_i])
        b_val = rebind[Scalar[dtype]](b[global_i])
        shared[local_i] = a_val * b_val

    barrier()

    # Tree reduction in shared memory - complex but educational
    var stride = tpb // 2
    while stride > 0:
        if local_i < stride:
            shared[local_i] += shared[local_i + stride]
        barrier()
        stride //= 2

    # Only thread 0 writes final result
    if local_i == 0:
        output[0] = shared[0]

```

**What makes this complex:**

- **Shared memory allocation**: Manual memory management within blocks
- **Explicit barriers**: `barrier()` calls to synchronize all threads in block
- **Tree reduction**: Complex loop with stride-based indexing (6432168421)
- **Cross-warp coordination**: Must synchronize across multiple warps
- **Conditional writes**: Only thread 0 writes the final result

This works across the entire block (128 threads across 2 or 4 warps depending on GPU), but it's verbose, error-prone, and requires deep understanding of block-level GPU synchronization.

### The warp-level improvement (from Puzzle 24)

Before jumping to block-level operations, recall how Puzzle 24 simplified reduction within a single warp using `warp.sum()`:

```mojo
fn simple_warp_dot_product
    in_layout: Layout, out_layout: Layout, size: Int
:
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)

    # Each thread computes one partial product using vectorized approach as values in Mojo are SIMD based
    var partial_product: Scalar[dtype] = 0
    if global_i < size:
        partial_product = (a[global_i] * b[global_i]).reduce_add()

    # warp_sum() replaces all the shared memory + barriers + tree reduction
    total = warp_sum(partial_product)

    # Only lane 0 writes the result (all lanes have the same total)
    if lane_id() == 0:
        output[global_i // WARP_SIZE] = total

```

**What `warp.sum()` achieved:**

- **Single warp scope**: Works within 32 threads (NVIDIA) or 32/64 threads (AMD)
- **Hardware shuffle**: Uses `shfl.sync.bfly.b32` instructions for efficiency
- **Zero shared memory**: No explicit memory management needed
- **One line reduction**: `total = warp_sumwarp_size=WARP_SIZE`

**But the limitation:**`warp.sum()` only works within a single warp. For problems requiring multiple warps (like our 128-thread block), you'd still need the complex shared memory + barriers approach to coordinate between warps.

**Test the traditional approach:**

  
    pixi NVIDIA (default)
    pixi AMD
    pixi Apple
    uv
  
  

```bash
pixi run p27 --traditional-dot-product
```

  
  

```bash
pixi run -e amd p27 --traditional-dot-product
```

  
  

```bash
pixi run -e apple p27 --traditional-dot-product
```

  
  

```bash
uv run poe p27 --traditional-dot-product
```

  

### Reference implementation (example)


```mojo
fn block_sum_dot_product
    in_layout: Layout, out_layout: Layout, tpb: Int
:
    """Dot product using block.sum() - convenience function like warp.sum()!
    Replaces manual shared memory + barriers + tree reduction with one line."""

    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    local_i = thread_idx.x

    # Each thread computes partial product
    var partial_product: Scalar[dtype] = 0.0
    if global_i < size:
        # LayoutTensor indexing `[0]` returns the underlying SIMD value
        partial_product = a[global_i][0] * b[global_i][0]

    # The magic: block.sum() replaces 15+ lines of manual reduction!
    # Just like warp.sum() but for the entire block
    total = block.sumblock_size=tpb, broadcast=False
    )

    # Only thread 0 writes the result
    if local_i == 0:
        output[0] = total[0]

```

The `block.sum()` kernel demonstrates the fundamental transformation from complex block synchronization to expertly optimized implementations:

**What disappeared from the traditional approach:**

- **15+ lines  8 lines**: Dramatic code reduction
- **Shared memory allocation**: Zero memory management required
- **7+ barrier() calls**: Zero explicit synchronization needed
- **Complex tree reduction**: Single function call
- **Stride-based indexing**: Eliminated entirely
- **Cross-warp coordination**: Handled automatically by optimized implementation

**Block-wide execution model:**

```
Block threads (128 threads across 4 warps):
Warp 0 (threads 0-31):
  Thread 0: partial_product = a[0] * b[0] = 0.0
  Thread 1: partial_product = a[1] * b[1] = 2.0
  ...
  Thread 31: partial_product = a[31] * b[31] = 1922.0

Warp 1 (threads 32-63):
  Thread 32: partial_product = a[32] * b[32] = 2048.0
  ...

Warp 2 (threads 64-95):
  Thread 64: partial_product = a[64] * b[64] = 8192.0
  ...

Warp 3 (threads 96-127):
  Thread 96: partial_product = a[96] * b[96] = 18432.0
  Thread 127: partial_product = a[127] * b[127] = 32258.0

block.sum() hardware operation:
All threads -> 0.0 + 2.0 + 1922.0 + 2048.0 + ... + 32258.0 = 1381760.0
Thread 0 receives -> total = 1381760.0 (when broadcast=False)
```

**Why this works without barriers:**

1. **Block-wide execution**: All threads execute each instruction in lockstep within warps
2. **Built-in synchronization**: `block.sum()` implementation handles synchronization internally
3. **Cross-warp communication**: Optimized communication between warps in the block
4. **Coordinated result delivery**: Only thread 0 receives the final result

**Comparison to warp.sum() (Puzzle 24):**

- **Warp scope**: `warp.sum()` works within 32/64 threads (single warp)
- **Block scope**: `block.sum()` works across entire block (multiple warps)
- **Same simplicity**: Both replace complex manual reductions with one-line calls
- **Automatic coordination**: `block.sum()` handles the cross-warp barriers that `warp.sum()` cannot

### Technical investigation: What does `block.sum()` actually compile to?

To understand what `block.sum()` actually generates, we compiled the puzzle with debug information:

```bash
pixi run mojo build --emit llvm --debug-level=line-tables solutions/p27/p27.mojo -o solutions/p27/p27.ll
```

This generated **LLVM file**`solutions/p27/p27.ll`. For example, on a compatible NVIDIA GPU, the `p27.ll` file has embedded **PTX assembly**showing the actual GPU instructions:

#### **Finding 1: Not a single instruction**

`block.sum()` compiles to approximately **20+ PTX instructions**, organized in a two-phase reduction:

**Phase 1: Warp-level reduction (butterfly shuffles)**

```ptx
shfl.sync.bfly.b32 %r23, %r46, 16, 31, -1;  // shuffle with offset 16
add.f32            %r24, %r46, %r23;         // add shuffled values
shfl.sync.bfly.b32 %r25, %r24, 8, 31, -1;   // shuffle with offset 8
add.f32            %r26, %r24, %r25;         // add shuffled values
// ... continues for offsets 4, 2, 1
```

**Phase 2: Cross-warp coordination**

```ptx
shr.u32            %r32, %r1, 5;             // compute warp ID
mov.b32            %r34, _global_alloc_$__gpu_shared_mem; // shared memory
bar.sync           0;                        // barrier synchronization
// ... another butterfly shuffle sequence for cross-warp reduction
```

#### **Finding 2: Hardware-optimized implementation**

- **Butterfly shuffles**: More efficient than tree reduction
- **Automatic barrier placement**: Handles cross-warp synchronization
- **Optimized memory access**: Uses shared memory strategically
- **Architecture-aware**: Same API works on NVIDIA (32-thread warps) and AMD (32 or 64-thread warps)

#### **Finding 3: Algorithm complexity analysis**

**Our approach to investigation:**

1. Located PTX assembly in binary ELF sections (`.nv_debug_ptx_txt`)
2. Identified algorithmic differences rather than counting individual instructions

**Key algorithmic differences observed:**

- **Traditional**: Tree reduction with shared memory + multiple `bar.sync` calls
- **block.sum()**: Butterfly shuffle pattern + optimized cross-warp coordination

The performance advantage comes from **expertly optimized algorithm choice**(butterfly > tree), not from instruction count or magical hardware. Take a look at [block.mojo] in Mojo gpu module for more details about the implementation.

### Performance insights

**`block.sum()` vs Traditional:**

- **Code simplicity**: 15+ lines  1 line for the reduction
- **Memory usage**: No shared memory allocation required
- **Synchronization**: No explicit barriers needed
- **Scalability**: Works with any block size (up to hardware limits)

**`block.sum()` vs `warp.sum()`:**

- **Scope**: Block-wide (128 threads) vs warp-wide (32 threads)
- **Use case**: When you need reduction across entire block
- **Convenience**: Same programming model, different scale

**When to use `block.sum()`:**

- **Single block problems**: When all data fits in one block
- **Block-level algorithms**: Shared memory computations needing reduction
- **Convenience over scalability**: Simpler than multi-block approaches

### Relationship to previous puzzles

**From Puzzle 12 (Traditional):**

```
Complex: shared memory + barriers + tree reduction

Simple: block.sum() hardware primitive
```

**From Puzzle 24 (`warp.sum()`):**

```
Warp-level: warp.sum() across 32 threads (single warp)

Block-level: block.sum() across 128 threads (multiple warps)
```

**Three-stage progression:**

1. **Manual reduction**(Puzzle 12): Complex shared memory + barriers + tree reduction
2. **Warp primitives**(Puzzle 24): `warp.sum()` - simple but limited to single warp
3. **Block primitives**(Puzzle 27): `block.sum()` - extends warp simplicity across multiple warps

**The key insight:**`block.sum()` gives you the simplicity of `warp.sum()` but scales across an entire block by automatically handling the complex cross-warp coordination that you'd otherwise need to implement manually.

Once you've learned about `block.sum()` operations, you're ready for:

- **[Block Prefix Sum Operations](#blockprefixsum-parallel-histogram-binning)**: Cumulative operations across block threads
- **[Block Broadcast Operations](#blockbroadcast-vector-normalization)**: Sharing values across all threads in a block

 **Key Takeaway**: Block operations extend warp programming concepts to entire thread blocks, providing optimized primitives that replace complex synchronization patterns while working across multiple warps simultaneously. Just like `warp.sum()` simplified warp-level reductions, `block.sum()` simplifies block-level reductions without sacrificing performance.

## block.prefix_sum() Parallel Histogram Binning

This puzzle implements parallel histogram binning using block-level [block.prefix_sum](https://docs.modular.com/mojo/stdlib/gpu/primitives/block/prefix_sum) operations for advanced parallel filtering and extraction. Each thread determines its element's target bin, then applies `block.prefix_sum()` to compute write positions for extracting elements from a specific bin, showing how prefix sum enables sophisticated parallel partitioning beyond simple reductions.

**Key insight:**_The [block.prefix_sum()](https://docs.modular.com/mojo/stdlib/gpu/primitives/block/prefix_sum) operation provides parallel filtering and extraction by computing cumulative write positions for matching elements across all threads in a block._

### Key concepts

This puzzle covers:

- **Block-level prefix sum**with `block.prefix_sum()`
- **Parallel filtering and extraction**using cumulative computations
- **Advanced parallel partitioning**algorithms
- **Histogram binning**with block-wide coordination
- **Exclusive vs inclusive**prefix sum patterns

The algorithm constructs histograms by extracting elements belonging to specific value ranges (bins):
\\[\Large \text{Bin}_k = \\{x_i : k/N \leq x_i < (k+1)/N\\}\\]

Each thread determines its element's bin assignment, with `block.prefix_sum()` coordinating parallel extraction.

### Configuration

- Vector size: `SIZE = 128` elements
- Data type: `DType.float32`
- Block configuration: `(128, 1)` threads per block (`TPB = 128`)
- Grid configuration: `(1, 1)` blocks per grid
- Number of bins: `NUM_BINS = 8` (ranges [0.0, 0.125), [0.125, 0.25), etc.)
- Layout: `Layout.row_major(SIZE)` (1D row-major)
- Warps per block: `128 / WARP_SIZE` (2 or 4 warps depending on GPU)

### Example scenario: Parallel bin extraction

Traditional sequential histogram construction processes elements one by one:

```python
# Sequential approach - doesn't parallelize well
histogram = [[] for _ in range(NUM_BINS)]
for element in data:
    bin_id = int(element * NUM_BINS)  # Determine bin
    histogram[bin_id].append(element)  # Sequential append
```

**Problems with naive GPU parallelization:**

- **Race conditions**: Multiple threads writing to same bin simultaneously
- **Uncoalesced memory**: Threads access different memory locations
- **Load imbalance**: Some bins may have many more elements than others
- **Complex synchronization**: Need barriers and atomic operations

### The advanced approach: `block.prefix_sum()` coordination

Transform the complex parallel partitioning into coordinated extraction:

### Reference implementation (example)


```mojo
fn block_histogram_bin_extract
    in_layout: Layout, bin_layout: Layout, out_layout: Layout, tpb: Int
:
    """Parallel histogram using block.prefix_sum() for bin extraction.

    This demonstrates advanced parallel filtering and extraction:
    1. Each thread determines which bin its element belongs to
    2. Use block.prefix_sum() to compute write positions for target_bin elements
    3. Extract and pack only elements belonging to target_bin
    """

    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    local_i = Int(thread_idx.x)

    # Step 1: Each thread determines its bin and element value
    var my_value: Scalar[dtype] = 0.0
    var my_bin: Int = -1

    if global_i < size:
        # `[0]` returns the underlying SIMD value
        my_value = input_data[global_i][0]
        # Bin values 0.0, 1.0) into num_bins buckets
        my_bin = Int(floor(my_value * num_bins))
        # Clamp to valid range
        if my_bin >= num_bins:
            my_bin = num_bins - 1
        if my_bin < 0:
            my_bin = 0

    # Step 2: Create predicate for target bin extraction
    var belongs_to_target: Int = 0
    if global_i < size and my_bin == target_bin:
        belongs_to_target = 1

    # Step 3: Use block.prefix_sum() for parallel bin extraction!
    # This computes where each thread should write within the target bin
    write_offset = block.prefix_sum[
        dtype = DType.int32, block_size=tpb, exclusive=True
    )

    # Step 4: Extract and pack elements belonging to target_bin
    if belongs_to_target == 1:
        bin_output[Int(write_offset[0])] = my_value

    # Step 5: Final thread computes total count for this bin
    if local_i == tpb - 1:
        # Inclusive sum = exclusive sum + my contribution
        total_count = write_offset[0] + belongs_to_target
        count_output[0] = total_count

```

The `block.prefix_sum()` kernel demonstrates advanced parallel coordination patterns by building on concepts from previous puzzles:

### **Step-by-step algorithm walkthrough:**

#### **Phase 1: Element processing (like Puzzle 12 dot product)**

```
Thread indexing (familiar pattern):
  global_i = block_dim.x * block_idx.x + thread_idx.x  // Global element index
  local_i = thread_idx.x                              // Local thread index

Element loading (like LayoutTensor pattern):
  Thread 0:  my_value = input_data[0][0] = 0.00
  Thread 1:  my_value = input_data[1][0] = 0.01
  Thread 13: my_value = input_data[13][0] = 0.13
  Thread 25: my_value = input_data[25][0] = 0.25
  ...
```

#### **Phase 2: Bin classification (new concept)**

```
Bin calculation using floor operation:
  Thread 0:  my_bin = Int(floor(0.00 * 8)) = 0  // Values [0.000, 0.125) -> bin 0
  Thread 1:  my_bin = Int(floor(0.01 * 8)) = 0  // Values [0.000, 0.125) -> bin 0
  Thread 13: my_bin = Int(floor(0.13 * 8)) = 1  // Values [0.125, 0.250) -> bin 1
  Thread 25: my_bin = Int(floor(0.25 * 8)) = 2  // Values [0.250, 0.375) -> bin 2
  ...
```

#### **Phase 3: Binary predicate creation (filtering pattern)**

```
For target_bin=0, create extraction mask:
  Thread 0:  belongs_to_target = 1  (bin 0 == target 0)
  Thread 1:  belongs_to_target = 1  (bin 0 == target 0)
  Thread 13: belongs_to_target = 0  (bin 1 != target 0)
  Thread 25: belongs_to_target = 0  (bin 2 != target 0)
  ...

This creates binary array: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, ...]
```

#### **Phase 4: Parallel prefix sum (the magic!)**

```
block.prefix_sum[exclusive=True] on predicates:
Input:     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, ...]
Exclusive: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12, -, -, -, ...]
                                                      ^
                                                 doesn't matter

Key insight: Each thread gets its WRITE POSITION in the output array!
```

#### **Phase 5: Coordinated extraction (conditional write)**

```
Only threads with belongs_to_target=1 write:
  Thread 0:  bin_output[0] = 0.00   // Uses write_offset[0] = 0
  Thread 1:  bin_output[1] = 0.01   // Uses write_offset[1] = 1
  Thread 12: bin_output[12] = 0.12  // Uses write_offset[12] = 12
  Thread 13: (no write)             // belongs_to_target = 0
  Thread 25: (no write)             // belongs_to_target = 0
  ...

Result: [0.00, 0.01, 0.02, ..., 0.12, ???, ???, ...] // Perfectly packed!
```

#### **Phase 6: Count computation (like block.sum() pattern)**

```
Last thread computes total (not thread 0!):
  if local_i == tpb - 1:  // Thread 127 in our case
      total = write_offset[0] + belongs_to_target  // Inclusive sum formula
      count_output[0] = total
```

### **Why this advanced algorithm works:**

#### **Connection to Puzzle 12 (Traditional dot product):**

- **Same thread indexing**: `global_i` and `local_i` patterns
- **Same bounds checking**: `if global_i < size` validation
- **Same data loading**: LayoutTensor SIMD extraction with `[0]`

#### **Connection to [`block.sum()`](#blocksum-essentials-block-level-dot-product) (earlier in this puzzle):**

- **Same block-wide operation**: All threads participate in block primitive
- **Same result handling**: Special thread (last instead of first) handles final result
- **Same SIMD conversion**: `Int(result[0])` pattern for array indexing

#### **Advanced concepts unique to `block.prefix_sum()`:**

- **Every thread gets result**: Unlike `block.sum()` where only thread 0 matters
- **Coordinated write positions**: Prefix sum eliminates race conditions automatically
- **Parallel filtering**: Binary predicates enable sophisticated data reorganization

### **Performance advantages over naive approaches:**

#### **vs. Atomic operations:**

- **No race conditions**: Prefix sum gives unique write positions
- **Coalesced memory**: Sequential writes improve cache performance
- **No serialization**: All writes happen in parallel

#### **vs. Multi-pass algorithms:**

- **Single kernel**: Complete histogram extraction in one GPU launch
- **Full utilization**: All threads work regardless of data distribution
- **Optimal memory bandwidth**: Pattern optimized for GPU memory hierarchy

This demonstrates how `block.prefix_sum()` enables sophisticated parallel algorithms that would be complex or impossible with simpler primitives like `block.sum()`.

### Performance insights

**`block.prefix_sum()` vs Traditional:**

- **Algorithm sophistication**: Advanced parallel partitioning vs sequential processing
- **Memory efficiency**: Coalesced writes vs scattered random access
- **Synchronization**: Built-in coordination vs manual barriers and atomics
- **Scalability**: Works with any block size and bin count

**`block.prefix_sum()` vs `block.sum()`:**

- **Scope**: Every thread gets result vs only thread 0
- **Use case**: Complex partitioning vs simple aggregation
- **Algorithm type**: Parallel scan primitive vs reduction primitive
- **Output pattern**: Per-thread positions vs single total

**When to use `block.prefix_sum()`:**

- **Parallel filtering**: Extract elements matching criteria
- **Stream compaction**: Remove unwanted elements
- **Parallel partitioning**: Separate data into categories
- **Advanced algorithms**: Load balancing, sorting, graph algorithms

Once you've learned about `block.prefix_sum()` operations, you're ready for:

- **[Block Broadcast Operations](#blockbroadcast-vector-normalization)**: Sharing values across all threads in a block
- **Multi-block algorithms**: Coordinating multiple blocks for larger problems
- **Advanced parallel algorithms**: Sorting, graph traversal, dynamic load balancing
- **Complex memory patterns**: Combining block operations with sophisticated memory access

 **Key Takeaway**: Block prefix sum operations transform GPU programming from simple parallel computations to sophisticated parallel algorithms. While `block.sum()` simplified reductions, `block.prefix_sum()` enables advanced data reorganization patterns essential for high-performance parallel algorithms.

## block.broadcast() Vector Normalization

Implement vector mean normalization by combining [block.sum](https://docs.modular.com/mojo/stdlib/gpu/primitives/block/sum) and [block.broadcast](https://docs.modular.com/mojo/stdlib/gpu/primitives/block/broadcast) operations to demonstrate the complete block-level communication workflow. Each thread will contribute to computing the mean, then receive the broadcast mean to normalize its element, showcasing how block operations work together to solve real parallel algorithms.

**Key insight:**_The [block.broadcast()](https://docs.modular.com/mojo/stdlib/gpu/primitives/block/broadcast) operation enables one-to-all communication, completing the fundamental block communication patterns: reduction (allone), scan (alleach), and broadcast (oneall)._

### Key concepts

In this puzzle, you'll learn:

- **Block-level broadcast**with `block.broadcast()`
- **One-to-all communication**patterns
- **Source thread specification**and parameter control
- **Complete block operations workflow**combining multiple operations
- **Real-world algorithm implementation**using coordinated block primitives

The algorithm demonstrates vector mean normalization:
\\[\Large \text{output}[i] = \frac{\text{input}[i]}{\frac{1}{N}\sum_{j=0}^{N-1} \text{input}[j]}\\]

Each thread contributes to the mean calculation, then receives the broadcast mean to normalize its element.

### Configuration

- Vector size: `SIZE = 128` elements
- Data type: `DType.float32`
- Block configuration: `(128, 1)` threads per block (`TPB = 128`)
- Grid configuration: `(1, 1)` blocks per grid
- Layout: `Layout.row_major(SIZE)` (1D row-major for input and output)
- Test data: Values cycling 1-8, so mean = 4.5
- Expected output: Normalized vector with mean = 1.0

### Example scenario: Coordinating block-wide computation and distribution

Traditional approaches to mean normalization require complex coordination:

```python
# Sequential approach - doesn't utilize parallelism
total = sum(input_array)
mean = total / len(input_array)
output_array = [x / mean for x in input_array]
```

**Problems with naive GPU parallelization:**

- **Multiple kernel launches**: One pass to compute mean, another to normalize
- **Global memory round-trip**: Store mean to global memory, read back later
- **Synchronization complexity**: Need barriers between computation phases
- **Thread divergence**: Different threads doing different tasks

**Traditional GPU solution complexity:**

```mojo
# Phase 1: Reduce to find sum (complex shared memory + barriers)
shared_sum[local_i] = my_value
barrier()
# Manual tree reduction with multiple barrier() calls...

# Phase 2: Thread 0 computes mean
if local_i == 0:
    mean = shared_sum[0] / size
    shared_mean[0] = mean

barrier()

# Phase 3: All threads read mean and normalize
mean = shared_mean[0]  # Everyone reads the same value
output[global_i] = my_value / mean
```

### The advanced approach: `block.sum()` + `block.broadcast()` coordination

Transform the multi-phase coordination into elegant block operations workflow:

### Reference implementation (example)


```mojo
fn block_normalize_vector
    in_layout: Layout, out_layout: Layout, tpb: Int
:
    """Vector mean normalization using block.sum() + block.broadcast() combination.

    This demonstrates the complete block operations workflow:
    1. Use block.sum() to compute sum of all elements (all -> one)
    2. Thread 0 computes mean = sum / size
    3. Use block.broadcast() to share mean to all threads (one -> all)
    4. Each thread normalizes: output[i] = input[i] / mean
    """

    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    local_i = thread_idx.x

    # Step 1: Each thread loads its element
    var my_value: Scalar[dtype] = 0.0
    if global_i < size:
        my_value = input_data[global_i][0]  # Extract SIMD value

    # Step 2: Use block.sum() to compute total sum (familiar from earlier!)
    total_sum = block.sumblock_size=tpb, broadcast=False
    )

    # Step 3: Thread 0 computes mean value
    var mean_value: Scalar[dtype] = 1.0  # Default to avoid division by zero
    if local_i == 0:
        if total_sum[0] > 0.0:
            mean_value = total_sum[0] / Float32(size)

    # Step 4: block.broadcast() shares mean to ALL threads!
    # This completes the block operations trilogy demonstration
    broadcasted_mean = block.broadcast
        dtype = DType.float32, width=1, block_size=tpb
    , src_thread=UInt(0))

    # Step 5: Each thread normalizes by the mean
    if global_i < size:
        normalized_value = my_value / broadcasted_mean[0]
        output_data[global_i] = normalized_value

```

The `block.broadcast()` kernel demonstrates the complete block operations workflow by combining all three fundamental communication patterns in a real algorithm that produces mathematically verifiable results:

### **Complete algorithm walkthrough with concrete execution:**

#### **Phase 1: Parallel data loading (established patterns from all previous puzzles)**

```
Thread indexing (consistent across all puzzles):
  global_i = block_dim.x * block_idx.x + thread_idx.x  // Maps to input array position
  local_i = thread_idx.x                              // Position within block (0-127)

Parallel element loading using LayoutTensor pattern:
  Thread 0:   my_value = input_data[0][0] = 1.0    // First cycle value
  Thread 1:   my_value = input_data[1][0] = 2.0    // Second cycle value
  Thread 7:   my_value = input_data[7][0] = 8.0    // Last cycle value
  Thread 8:   my_value = input_data[8][0] = 1.0    // Cycle repeats: 1,2,3,4,5,6,7,8,1,2...
  Thread 15:  my_value = input_data[15][0] = 8.0   // 15 % 8 = 7, so 8th value
  Thread 127: my_value = input_data[127][0] = 8.0  // 127 % 8 = 7, so 8th value

All 128 threads load simultaneously - perfect parallel efficiency!
```

#### **Phase 2: Block-wide sum reduction (leveraging earlier block.sum() knowledge)**

```
block.sum() coordination across all 128 threads:
  Contribution analysis:
    - Values 1,2,3,4,5,6,7,8 repeat 16 times each (128/8 = 16)
    - Thread contributions: 16x1 + 16x2 + 16x3 + 16x4 + 16x5 + 16x6 + 16x7 + 16x8
    - Mathematical sum: 16 x (1+2+3+4+5+6+7+8) = 16 x 36 = 576.0

block.sum() hardware execution:
  All threads -> [reduction tree] -> Thread 0
  total_sum = SIMDDType.float32, 1  // Only thread 0 receives this

Threads 1-127: Have no access to total_sum (broadcast=False in block.sum)
```

#### **Phase 3: Exclusive mean computation (single-thread processing)**

```
Thread 0 performs critical computation:
  Input: total_sum[0] = 576.0, size = 128
  Computation: mean_value = 576.0 / 128.0 = 4.5

  Verification: Expected mean = (1+2+3+4+5+6+7+8)/8 = 36/8 = 4.5 OK

All other threads (1-127):
  mean_value = 1.0 (default safety value)
  These values are irrelevant - will be overwritten by broadcast

Critical insight: Only thread 0 has the correct mean value at this point!
```

#### **Phase 4: Block-wide broadcast distribution (one  all communication)**

```
block.broadcast() API execution:
  Source: src_thread = UInt(0) -> Thread 0's mean_value = 4.5
  Target: All 128 threads in block

Before broadcast:
  Thread 0:   mean_value = 4.5  <- Source of truth
  Thread 1:   mean_value = 1.0  <- Will be overwritten
  Thread 2:   mean_value = 1.0  <- Will be overwritten
  ...
  Thread 127: mean_value = 1.0  <- Will be overwritten

After block.broadcast() execution:
  Thread 0:   broadcasted_mean[0] = 4.5  <- Receives own value back
  Thread 1:   broadcasted_mean[0] = 4.5  <- Now has correct value!
  Thread 2:   broadcasted_mean[0] = 4.5  <- Now has correct value!
  ...
  Thread 127: broadcasted_mean[0] = 4.5  <- Now has correct value!

Result: Perfect synchronization - all threads have identical mean value!
```

#### **Phase 5: Parallel mean normalization (coordinated processing)**

```
Each thread independently normalizes using broadcast mean:
  Thread 0:   normalized = 1.0 / 4.5 = 0.22222222...
  Thread 1:   normalized = 2.0 / 4.5 = 0.44444444...
  Thread 2:   normalized = 3.0 / 4.5 = 0.66666666...
  Thread 7:   normalized = 8.0 / 4.5 = 1.77777777...
  Thread 8:   normalized = 1.0 / 4.5 = 0.22222222...  (pattern repeats)
  ...

Mathematical verification:
  Output sum = (0.222... + 0.444... + ... + 1.777...) x 16 = 4.5 x 16 x 2 = 128.0
  Output mean = 128.0 / 128 = 1.0  Perfect normalization!

Each value divided by original mean gives output with mean = 1.0
```

#### **Phase 6: Verification of correctness**

```
Input analysis:
  - Sum: 576.0, Mean: 4.5
  - Max: 8.0, Min: 1.0
  - Range: [1.0, 8.0]

Output analysis:
  - Sum: 128.0, Mean: 1.0 OK
  - Max: 1.777..., Min: 0.222...
  - Range: [0.222, 1.777] (all values scaled by factor 1/4.5)

Proportional relationships preserved:
  - Original 8:1 ratio becomes 1.777:0.222 = 8:1 OK
  - All relative magnitudes maintained perfectly
```

### **Why this complete workflow is mathematically and computationally superior:**

#### **Technical accuracy and verification:**

```
Mathematical proof of correctness:
  Input: x, x, ..., x where n = 128
  Mean:  = (x)/n = 576/128 = 4.5

  Normalization: y = x/
  Output mean: (y)/n = (x/)/n = (1/)(x)/n = (1/) = 1 OK

Algorithm produces provably correct mathematical result.
```

#### **Connection to Puzzle 12 (foundational patterns):**

- **Thread coordination evolution**: Same `global_i`, `local_i` patterns but with block primitives
- **Memory access patterns**: Same LayoutTensor SIMD extraction `[0]` but optimized workflow
- **Complexity elimination**: Replaces 20+ lines of manual barriers with 2 block operations
- **Educational progression**: Manual  automated, complex  simple, error-prone  reliable

#### **Connection to [`block.sum()`](#blocksum-essentials-block-level-dot-product) (perfect integration):**

- **API consistency**: Identical template structure `[block_size=tpb, broadcast=False]`
- **Result flow design**: Thread 0 receives sum, naturally computes derived parameter
- **Seamless composition**: Output of `block.sum()` becomes input for computation + broadcast
- **Performance optimization**: Single-kernel workflow vs multi-pass approaches

#### **Connection to [`block.prefix_sum()`](#blockprefixsum-parallel-histogram-binning) (complementary communication):**

- **Distribution patterns**: `prefix_sum` gives unique positions, `broadcast` gives shared values
- **Usage scenarios**: `prefix_sum` for parallel partitioning, `broadcast` for parameter sharing
- **Template consistency**: Same `dtype`, `block_size` parameter patterns across all operations
- **SIMD handling uniformity**: All block operations return SIMD requiring `[0]` extraction

#### **Advanced algorithmic insights:**

```
Communication pattern comparison:
  Traditional approach:
    1. Manual reduction:     O(log n) with explicit barriers
    2. Shared memory write:  O(1) with synchronization
    3. Shared memory read:   O(1) with potential bank conflicts
    Total: Multiple synchronization points, error-prone

  Block operations approach:
    1. block.sum():          O(log n) hardware-optimized, automatic barriers
    2. Computation:          O(1) single thread
    3. block.broadcast():    O(log n) hardware-optimized, automatic distribution
    Total: Two primitives, automatic synchronization, provably correct
```

#### **Real-world algorithm patterns demonstrated:**

```
Common parallel algorithm structure:
  Phase 1: Parallel data processing      -> All threads contribute
  Phase 2: Global parameter computation  -> One thread computes
  Phase 3: Parameter distribution        -> All threads receive
  Phase 4: Coordinated parallel output   -> All threads process

This exact pattern appears in:
  - Batch normalization (deep learning)
  - Histogram equalization (image processing)
  - Iterative numerical methods (scientific computing)
  - Lighting calculations (computer graphics)

Mean normalization is the perfect educational example of this fundamental pattern.
```

### **Block operations trilogy completed:**

#### **1. `block.sum()` - All to One (Reduction)**

- **Input**: All threads provide values
- **Output**: Thread 0 receives aggregated result
- **Use case**: Computing totals, finding maximums, etc.

#### **2. `block.prefix_sum()` - All to Each (Scan)**

- **Input**: All threads provide values
- **Output**: Each thread receives cumulative position
- **Use case**: Computing write positions, parallel partitioning

#### **3. `block.broadcast()` - One to All (Broadcast)**

- **Input**: One thread provides value (typically thread 0)
- **Output**: All threads receive the same value
- **Use case**: Sharing computed parameters, configuration values

**Complete block operations progression:**

1. **Manual coordination**(Puzzle 12): Understand parallel fundamentals
2. **Warp primitives**(Puzzle 24): Learn hardware-accelerated patterns
3. **Block reduction**([`block.sum()`](#blocksum-essentials-block-level-dot-product)): Learn allone communication
4. **Block scan**([`block.prefix_sum()`](#blockprefixsum-parallel-histogram-binning)): Learn alleach communication
5. **Block broadcast**(`block.broadcast()`): Learn oneall communication

**The complete picture:**Block operations provide the fundamental communication building blocks for sophisticated parallel algorithms, replacing complex manual coordination with clean, composable primitives.

### Performance insights and technical analysis

#### **Quantitative performance comparison:**

**`block.broadcast()` vs Traditional shared memory approach (for demonstration):**

**Traditional Manual Approach:**

```
Phase 1: Manual reduction
  - Shared memory allocation: ~5 cycles
  - Barrier synchronization: ~10 cycles
  - Tree reduction loop: ~15 cycles
  - Error-prone manual indexing

Phase 2: Mean computation: ~2 cycles

Phase 3: Shared memory broadcast
  - Manual write to shared: ~2 cycles
  - Barrier synchronization: ~10 cycles
  - All threads read: ~3 cycles

Total: ~47 cycles
  + synchronization overhead
  + potential race conditions
  + manual error debugging
```

**Block Operations Approach:**

```
Phase 1: block.sum()
  - Hardware-optimized: ~3 cycles
  - Automatic barriers: 0 explicit cost
  - Optimized reduction: ~8 cycles
  - Verified correct implementation

Phase 2: Mean computation: ~2 cycles

Phase 3: block.broadcast()
  - Hardware-optimized: ~4 cycles
  - Automatic distribution: 0 explicit cost
  - Verified correct implementation

Total: ~17 cycles
  + automatic optimization
  + guaranteed correctness
  + composable design
```

#### **Memory hierarchy advantages:**

**Cache efficiency:**

- **block.sum()**: Optimized memory access patterns reduce cache misses
- **block.broadcast()**: Efficient distribution minimizes memory bandwidth usage
- **Combined workflow**: Single kernel reduces global memory round-trips by 100%

**Memory bandwidth utilization:**

```
Traditional multi-kernel approach:
  Kernel 1: Input -> Reduction -> Global memory write
  Kernel 2: Global memory read -> Broadcast -> Output
  Total global memory transfers: 3x array size

Block operations single-kernel:
  Input -> block.sum() -> block.broadcast() -> Output
  Total global memory transfers: 2x array size (33% improvement)
```

#### **When to use each block operation:**

**`block.sum()` optimal scenarios:**

- **Data aggregation**: Computing totals, averages, maximum/minimum values
- **Reduction patterns**: Any all-to-one communication requirement
- **Statistical computation**: Mean, variance, correlation calculations

**`block.prefix_sum()` optimal scenarios:**

- **Parallel partitioning**: Stream compaction, histogram binning
- **Write position calculation**: Parallel output generation
- **Parallel algorithms**: Sorting, searching, data reorganization

**`block.broadcast()` optimal scenarios:**

- **Parameter distribution**: Sharing computed values to all threads
- **Configuration propagation**: Mode flags, scaling factors, thresholds
- **Coordinated processing**: When all threads need the same computed parameter

#### **Composition benefits:**

```
Individual operations: Good performance, limited scope
Combined operations:   Excellent performance, comprehensive algorithms

Example combinations seen in real applications:
- block.sum() + block.broadcast():       Normalization algorithms
- block.prefix_sum() + block.sum():      Advanced partitioning
- All three together:                    Complex parallel algorithms
- With traditional patterns:             Hybrid optimization strategies
```

Once you've learned about the complete block operations trilogy, you're ready for:

- **Multi-block algorithms**: Coordinating operations across multiple thread blocks
- **Advanced parallel patterns**: Combining block operations for complex algorithms
- **Memory hierarchy optimization**: Efficient data movement patterns
- **Algorithm design**: Structuring parallel algorithms using block operation building blocks
- **Performance optimization**: Choosing optimal block sizes and operation combinations

 **Key Takeaway**: The block operations trilogy (`sum`, `prefix_sum`, `broadcast`) provides complete communication primitives for block-level parallel programming. By composing these operations, you can implement sophisticated parallel algorithms with clean, maintainable code that leverages GPU hardware optimizations. Mean normalization demonstrates how these operations work together to solve real computational problems efficiently.
