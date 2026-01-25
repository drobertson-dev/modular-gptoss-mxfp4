---
title: "Advanced Warp Patterns"
description: "Welcome to Puzzle 26: Advanced Warp Communication Primitives! This puzzle introduces you to sophisticated GPU warp-level butterfly communication and parallel scan operations - hardware-accelerated primitives that enable efficient tree-based algorithms and parallel reductions within warps. You'll learn about using shuffle_xor for butterfly networks and prefix_sum for hardware-optimized parallel scan without complex multi-phase shared memory algorithms."
---

# Advanced Warp Patterns

Welcome to Puzzle 26: Advanced Warp Communication Primitives! This puzzle introduces you to sophisticated GPU warp-level butterfly communication and parallel scan operations - hardware-accelerated primitives that enable efficient tree-based algorithms and parallel reductions within warps. You'll learn about using shuffle_xor for butterfly networks and prefix_sum for hardware-optimized parallel scan without complex multi-phase shared memory algorithms.

## Overview

Welcome to **Puzzle 26: Advanced Warp Communication Primitives**! This puzzle introduces you to sophisticated GPU **warp-level butterfly communication and parallel scan operations**- hardware-accelerated primitives that enable efficient tree-based algorithms and parallel reductions within warps. You'll learn about using [shuffle_xor](https://docs.modular.com/mojo/stdlib/gpu/warp/shuffle_xor) for butterfly networks and [prefix_sum](https://docs.modular.com/mojo/stdlib/gpu/warp/prefix_sum) for hardware-optimized parallel scan without complex multi-phase shared memory algorithms.

**What you'll achieve:**Transform from complex shared memory + barrier + multi-phase reduction patterns to elegant single-function-call algorithms that leverage hardware-optimized butterfly networks and parallel scan units.

**Key insight:**_GPU warps can perform sophisticated tree-based communication and parallel scan operations in hardware - Mojo's advanced warp primitives harness butterfly networks and dedicated scan units to provide \\(O(\\log n)\\) algorithms with single-instruction simplicity._

## What you'll learn

### **Advanced warp communication model**

Understand sophisticated communication patterns within GPU warps:

```
GPU Warp Butterfly Network (32 threads, XOR-based communication)
Offset 16: Lane 0 <-> Lane 16, Lane 1 <-> Lane 17, ..., Lane 15 <-> Lane 31
Offset 8:  Lane 0 <-> Lane 8,  Lane 1 <-> Lane 9,  ..., Lane 23 <-> Lane 31
Offset 4:  Lane 0 <-> Lane 4,  Lane 1 <-> Lane 5,  ..., Lane 27 <-> Lane 31
Offset 2:  Lane 0 <-> Lane 2,  Lane 1 <-> Lane 3,  ..., Lane 29 <-> Lane 31
Offset 1:  Lane 0 <-> Lane 1,  Lane 2 <-> Lane 3,  ..., Lane 30 <-> Lane 31

Hardware Prefix Sum (parallel scan acceleration)
Input:  [1, 2, 3, 4, 5, 6, 7, 8, ...]
Output: [1, 3, 6, 10, 15, 21, 28, 36, ...] (inclusive scan)
```

**Hardware reality:**

- **Butterfly networks**: XOR-based communication creates optimal tree topologies
- **Dedicated scan units**: Hardware-accelerated parallel prefix operations
- **Logarithmic complexity**: \\(O(\\log n)\\) algorithms replace \\(O(n)\\) sequential patterns
- **Single-cycle operations**: Complex reductions happen in specialized hardware

### **Advanced warp operations in Mojo**

Learn the sophisticated communication primitives from `gpu.primitives.warp`:

1. **[`shuffle_xor(value, mask)`](https://docs.modular.com/mojo/stdlib/gpu/warp/shuffle_xor)**: XOR-based butterfly communication for tree algorithms
2. **[`prefix_sum(value)`](https://docs.modular.com/mojo/stdlib/gpu/warp/prefix_sum)**: Hardware-accelerated parallel scan operations
3. **Advanced coordination patterns**: Combining multiple primitives for complex algorithms

> **Note:**These primitives enable sophisticated parallel algorithms like parallel reductions, stream compaction, quicksort partitioning, and FFT operations that would otherwise require dozens of lines of shared memory coordination code.

### **Performance transformation example**

```mojo
# Complex parallel reduction (traditional approach - from Puzzle 14):
shared = LayoutTensor[
    dtype,
    Layout.row_major(WARP_SIZE),
    MutAnyOrigin,
    address_space = AddressSpace.SHARED,
].stack_allocation()
shared[local_i] = input[global_i]
barrier()
offset = 1
for i in range(Int(log2(Scalardtype))):
    var current_val: output.element_type = 0
    if local_i >= offset and local_i < WARP_SIZE:
        current_val = shared[local_i - offset]
    barrier()
    if local_i >= offset and local_i < WARP_SIZE:
        shared[local_i] += current_val
    barrier()
    offset *= 2

# Advanced warp primitives eliminate all this complexity:
current_val = input[global_i]
scan_result = prefix_sumexclusive=False  # Single call!
output[global_i] = scan_result
```

### **When advanced warp operations excel**

Learn the performance characteristics:

| Algorithm Pattern | Traditional | Advanced Warp Operations |
|------------------|-------------|-------------------------|
| Parallel reductions | Shared memory + barriers | Single `shuffle_xor` tree |
| Prefix/scan operations | Multi-phase algorithms | Hardware `prefix_sum` |
| Stream compaction | Complex indexing | `prefix_sum` + coordination |
| Quicksort partition | Manual position calculation | Combined primitives |
| Tree algorithms | Recursive shared memory | Butterfly communication |

## Prerequisites

Before diving into advanced warp communication, ensure you're comfortable with:

- **Part VII warp fundamentals**: Understanding SIMT execution and basic warp operations (see Puzzle 24 and Puzzle 25)
- **Parallel algorithm theory**: Tree reductions, parallel scan, and butterfly networks
- **GPU memory hierarchy**: Shared memory patterns and synchronization (see Puzzle 14)
- **Mathematical operations**: Understanding XOR operations and logarithmic complexity

## Learning path

### **1. Butterfly communication with shuffle_xor**

** [Warp Shuffle XOR](#warpshufflexor-butterfly-communication)**

Learn XOR-based butterfly communication patterns for efficient tree algorithms and parallel reductions.

**What you'll learn:**

- Using `shuffle_xor()` for creating butterfly network topologies
- Implementing \\(O(\\log n)\\) parallel reductions with tree communication
- Understanding XOR-based lane pairing and communication patterns
- Advanced conditional butterfly operations for multi-value reductions

**Key pattern:**

```mojo
max_val = input[global_i]
offset = WARP_SIZE // 2
while offset > 0:
    max_val = max(max_val, shuffle_xor(max_val, offset))
    offset //= 2
# All lanes now have global maximum
```

### **2. Hardware-accelerated parallel scan with prefix_sum**

** [Warp Prefix Sum](#warpprefixsum-hardware-optimized-parallel-scan)**

Learn hardware-optimized parallel scan operations that replace complex multi-phase algorithms with single function calls.

**What you'll learn:**

- Using `prefix_sum()` for hardware-accelerated cumulative operations
- Implementing stream compaction and parallel partitioning
- Combining `prefix_sum` with `shuffle_xor` for advanced coordination
- Understanding inclusive vs exclusive scan patterns

**Key pattern:**

```mojo
current_val = input[global_i]
scan_result = prefix_sumexclusive=False
output[global_i] = scan_result  # Hardware-optimized cumulative sum
```

## Key concepts

### **Butterfly network communication**

Understanding XOR-based communication topologies:

- **XOR pairing**: `lane_id  mask` creates symmetric communication pairs
- **Tree reduction**: Logarithmic complexity through hierarchical data exchange
- **Parallel coordination**: All lanes participate simultaneously in reduction
- **Dynamic algorithms**: Works for any power-of-2 `WARP_SIZE` (32, 64, etc.)

### **Hardware-accelerated parallel scan**

Recognizing dedicated scan unit capabilities:

- **Prefix sum operations**: Cumulative operations with hardware acceleration
- **Stream compaction**: Parallel filtering and data reorganization
- **Single-function simplicity**: Complex algorithms become single calls
- **Zero synchronization**: Hardware handles all coordination internally

### **Algorithm complexity transformation**

Converting traditional patterns to advanced warp operations:

- **Sequential reductions**(\\(O(n)\\))  **Butterfly reductions**(\\(O(\\log n)\\))
- **Multi-phase scan algorithms** **Single hardware prefix_sum**
- **Complex shared memory patterns** **Register-only operations**
- **Explicit synchronization** **Hardware-managed coordination**

### **Advanced coordination patterns**

Combining multiple primitives for sophisticated algorithms:

- **Dual reductions**: Simultaneous min/max tracking with butterfly patterns
- **Parallel partitioning**: `shuffle_xor` + `prefix_sum` for quicksort-style operations
- **Conditional operations**: Lane-based output selection with global coordination
- **Multi-primitive algorithms**: Complex parallel patterns with optimal performance

## Getting started

Ready to harness advanced GPU warp-level communication? Start with butterfly network operations to understand tree-based communication, then progress to hardware-accelerated parallel scan for optimal algorithm performance.

 **Success tip**: Think of advanced warp operations as **hardware-accelerated parallel algorithm building blocks**. These primitives replace entire categories of complex shared memory algorithms with single, optimized function calls.

**Learning objective**: By the end of Puzzle 24, you'll recognize when advanced warp primitives can replace complex multi-phase algorithms, enabling you to write dramatically simpler and faster tree-based reductions, parallel scans, and coordination patterns.

**Ready to begin?**Start with **[Warp Shuffle XOR Operations](#warpshufflexor-butterfly-communication)**to learn butterfly communication, then advance to **[Warp Prefix Sum Operations](#warpprefixsum-hardware-optimized-parallel-scan)**for hardware-accelerated parallel scan patterns!

## `warp.shuffle_xor()` Butterfly Communication

For warp-level butterfly communication we can use `shuffle_xor()` to create sophisticated tree-based communication patterns within a warp. This powerful primitive enables efficient parallel reductions, sorting networks, and advanced coordination algorithms without shared memory or explicit synchronization.

**Key insight:**_The [shuffle_xor()](https://docs.modular.com/mojo/stdlib/gpu/warp/shuffle_xor) operation leverages SIMT execution to create XOR-based communication trees, enabling efficient butterfly networks and parallel algorithms that scale with \\(O(\\log n)\\) complexity relative to warp size._

> **What are butterfly networks?**[Butterfly networks](https://en.wikipedia.org/wiki/Butterfly_network) are communication topologies where threads exchange data based on XOR patterns of their indices. The name comes from the visual pattern when drawn - connections that look like butterfly wings. These networks are fundamental to parallel algorithms like FFT, bitonic sort, and parallel reductions because they enable \\(O(\\log n)\\) communication complexity.

### Key concepts

In this puzzle, you'll learn:

- **XOR-based communication patterns**with `shuffle_xor()`
- **Butterfly network topologies**for parallel algorithms
- **Tree-based parallel reductions**with \\(O(\\log n)\\) complexity
- **Conditional butterfly operations**for advanced coordination
- **Hardware-optimized parallel primitives**replacing complex shared memory

The `shuffle_xor` operation enables each lane to exchange data with lanes based on [XOR](https://en.wikipedia.org/wiki/Exclusive_or) patterns:
\\[\\Large \text{shuffle\_xor}(\text{value}, \text{mask}) = \text{value_from_lane}(\text{lane\_id} \oplus \text{mask})\\]

This transforms complex parallel algorithms into elegant butterfly communication patterns, enabling efficient tree reductions and sorting networks without explicit coordination.

### 1. Basic butterfly pair swap

#### Configuration

- Vector size: `SIZE = WARP_SIZE` (32 or 64 depending on GPU)
- Grid configuration: `(1, 1)` blocks per grid
- Block configuration: `(WARP_SIZE, 1)` threads per block
- Data type: `DType.float32`
- Layout: `Layout.row_major(SIZE)` (1D row-major)

#### The `shuffle_xor` concept

Traditional pair swapping requires complex indexing and coordination:

```mojo
# Traditional approach - complex and requires synchronization
shared_memory[lane] = input[global_i]
barrier()
if lane % 2 == 0:
    partner = lane + 1
else:
    partner = lane - 1
if partner < WARP_SIZE:
    swapped_val = shared_memory[partner]
```

**Problems with traditional approach:**

- **Memory overhead**: Requires shared memory allocation
- **Synchronization**: Needs explicit barriers
- **Complex logic**: Manual partner calculation and bounds checking
- **Poor scaling**: Doesn't leverage hardware communication

With `shuffle_xor()`, pair swapping becomes elegant:

```mojo
# Butterfly XOR approach - simple and hardware-optimized
current_val = input[global_i]
swapped_val = shuffle_xor(current_val, 1)  # XOR with 1 creates pairs
output[global_i] = swapped_val
```

**Benefits of shuffle_xor:**

- **Zero memory overhead**: Direct register-to-register communication
- **No synchronization**: SIMT execution guarantees correctness
- **Hardware optimized**: Single instruction for all lanes
- **Butterfly foundation**: Building block for complex parallel algorithms

#### Tips

#### 1. **Understanding shuffle_xor**

The `shuffle_xor(value, mask)` operation allows each lane to exchange data with a lane whose ID differs by the XOR mask. Think about what happens when you XOR a lane ID with different mask values.

**Key question to explore:**

- What partner does lane 0 get when you XOR with mask 1?
- What partner does lane 1 get when you XOR with mask 1?
- Do you see a pattern forming?

**Hint**: Try working out the XOR operation manually for the first few lane IDs to understand the pairing pattern.

#### 2. **XOR pair pattern**

Think about the binary representation of lane IDs and what happens when you flip the least significant bit.

**Questions to consider:**

- What happens to even-numbered lanes when you XOR with 1?
- What happens to odd-numbered lanes when you XOR with 1?
- Why does this create perfect pairs?

#### 3. **No boundary checking needed**

Unlike `shuffle_down()`, `shuffle_xor()` operations stay within warp boundaries. Consider why XOR with small masks never creates out-of-bounds lane IDs.

**Think about**: What's the maximum lane ID you can get when XORing any valid lane ID with 1?

**Test the butterfly pair swap:**

  
    pixi NVIDIA (default)
    pixi AMD
    pixi Apple
    uv
  
  

```bash
pixi run p26 --pair-swap
```

  
  

```bash
pixi run -e amd p26 --pair-swap
```

  
  

```bash
pixi run -e apple p26 --pair-swap
```

  
  

```bash
uv run poe p26 --pair-swap
```

  

Expected output when solved:

```txt
WARP_SIZE:  32
SIZE:  32
output: [1.0, 0.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0, 8.0, 11.0, 10.0, 13.0, 12.0, 15.0, 14.0, 17.0, 16.0, 19.0, 18.0, 21.0, 20.0, 23.0, 22.0, 25.0, 24.0, 27.0, 26.0, 29.0, 28.0, 31.0, 30.0]
expected: [1.0, 0.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0, 8.0, 11.0, 10.0, 13.0, 12.0, 15.0, 14.0, 17.0, 16.0, 19.0, 18.0, 21.0, 20.0, 23.0, 22.0, 25.0, 24.0, 27.0, 26.0, 29.0, 28.0, 31.0, 30.0]
OK Butterfly pair swap test passed!
```

#### Reference implementation (example)


```mojo
fn butterfly_pair_swap
    layout: Layout, size: Int
:
    """
    Basic butterfly pair swap: Exchange values between adjacent pairs using XOR pattern.
    Each thread exchanges its value with its XOR-1 neighbor, creating pairs: (0,1), (2,3), (4,5), etc.
    Uses shuffle_xor(val, 1) to swap values within each pair.
    This is the foundation of butterfly network communication patterns.
    """
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)

    if global_i < size:
        current_val = input[global_i]

        # Exchange with XOR-1 neighbor using butterfly pattern
        # Lane 0 exchanges with lane 1, lane 2 with lane 3, etc.
        swapped_val = shuffle_xor(current_val, 1)

        # For demonstration, we'll store the swapped value
        # In real applications, this might be used for sorting, reduction, etc.
        output[global_i] = swapped_val

```

This solution demonstrates how `shuffle_xor()` creates perfect pair exchanges through XOR communication patterns.

**Algorithm breakdown:**

```mojo
if global_i < size:
    current_val = input[global_i]              # Each lane reads its element
    swapped_val = shuffle_xor(current_val, 1)  # XOR creates pair exchange

    # For demonstration, store the swapped value
    output[global_i] = swapped_val
```

**SIMT execution deep dive:**

```
Cycle 1: All lanes load their values simultaneously
  Lane 0: current_val = input[0] = 0
  Lane 1: current_val = input[1] = 1
  Lane 2: current_val = input[2] = 2
  Lane 3: current_val = input[3] = 3
  ...
  Lane 31: current_val = input[31] = 31

Cycle 2: shuffle_xor(current_val, 1) executes on all lanes
  Lane 0: receives from Lane 1 (01=1) -> swapped_val = 1
  Lane 1: receives from Lane 0 (11=0) -> swapped_val = 0
  Lane 2: receives from Lane 3 (21=3) -> swapped_val = 3
  Lane 3: receives from Lane 2 (31=2) -> swapped_val = 2
  ...
  Lane 30: receives from Lane 31 (301=31) -> swapped_val = 31
  Lane 31: receives from Lane 30 (311=30) -> swapped_val = 30

Cycle 3: Store results
  Lane 0: output[0] = 1
  Lane 1: output[1] = 0
  Lane 2: output[2] = 3
  Lane 3: output[3] = 2
  ...
```

**Mathematical insight:**This implements perfect pair exchange using XOR properties:
\\[\\Large \\text{XOR}(i, 1) = \\begin{cases}
i + 1 & \\text{if } i \\bmod 2 = 0 \\\\
i - 1 & \\text{if } i \\bmod 2 = 1
\\end{cases}\\]

**Why shuffle_xor is superior:**

1. **Perfect symmetry**: Every lane participates in exactly one pair
2. **No coordination**: All pairs exchange simultaneously
3. **Hardware optimized**: Single instruction for entire warp
4. **Butterfly foundation**: Building block for complex parallel algorithms

**Performance characteristics:**

- **Latency**: 1 cycle (hardware register exchange)
- **Bandwidth**: 0 bytes (no memory traffic)
- **Parallelism**: All WARP_SIZE lanes exchange simultaneously
- **Scalability**: \\(O(1)\\) complexity regardless of data size

### 2. Butterfly parallel maximum

#### Configuration

- Vector size: `SIZE = WARP_SIZE` (32 or 64 depending on GPU)
- Grid configuration: `(1, 1)` blocks per grid
- Block configuration: `(WARP_SIZE, 1)` threads per block

#### Tips

#### 1. **Understanding butterfly reduction**

The butterfly reduction creates a binary tree communication pattern. Think about how you can systematically reduce the problem size at each step.

**Key questions:**

- What should be your starting offset to cover the maximum range?
- How should the offset change between steps?
- When should you stop the reduction?

**Hint**: The name "butterfly" comes from the communication pattern - try sketching it out for a small example.

#### 2. **XOR reduction properties**

XOR creates non-overlapping communication pairs at each step. Consider why this is important for parallel reductions.

**Think about:**

- How does XOR with different offsets create different communication patterns?
- Why don't lanes interfere with each other at the same step?
- What makes XOR particularly well-suited for tree reductions?

#### 3. **Accumulating maximum values**

Each lane needs to progressively build up knowledge of the maximum value in its "region".

**Algorithm structure:**

- Start with your own value
- At each step, compare with a neighbor's value
- Keep the maximum and continue

**Key insight**: After each step, your "region of knowledge" doubles in size.

- After final step: Each lane knows global maximum

#### 4. **Why this pattern works**

The butterfly reduction guarantees that after \\(\\log_2(\\text{WARP\\_SIZE})\\) steps:

- **Every lane**has seen **every other lane's**value indirectly
- **No redundant communication**: Each pair exchanges exactly once per step
- **Optimal complexity**: \\(O(\\log n)\\) steps instead of \\(O(n)\\) sequential comparison

**Trace example**(4 lanes, values [3, 1, 7, 2]):

```
Initial: Lane 0=3, Lane 1=1, Lane 2=7, Lane 3=2

Step 1 (offset=2): 0 <-> 2, 1 <-> 3
  Lane 0: max(3, 7) = 7
  Lane 1: max(1, 2) = 2
  Lane 2: max(7, 3) = 7
  Lane 3: max(2, 1) = 2

Step 2 (offset=1): 0 <-> 1, 2 <-> 3
  Lane 0: max(7, 2) = 7
  Lane 1: max(2, 7) = 7
  Lane 2: max(7, 2) = 7
  Lane 3: max(2, 7) = 7

Result: All lanes have global maximum = 7
```

**Test the butterfly parallel maximum:**

  
    pixi NVIDIA (default)
    pixi AMD
    uv
  
  

```bash
pixi run p26 --parallel-max
```

  
  

```bash
pixi run -e amd p26 --parallel-max
```

  
  

```bash
uv run poe p26 --parallel-max
```

  

Expected output when solved:

```txt
WARP_SIZE:  32
SIZE:  32
output: [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
expected: [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
OK Butterfly parallel max test passed!
```

#### Reference implementation (example)


```mojo
fn butterfly_parallel_max
    layout: Layout, size: Int
:
    """
    Parallel maximum reduction using butterfly pattern.
    Uses shuffle_xor with decreasing offsets (16, 8, 4, 2, 1) to perform tree-based reduction.
    Each step reduces the active range by half until all threads have the maximum value.
    This implements an efficient O(log n) parallel reduction algorithm.
    """
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)

    if global_i < size:
        max_val = input[global_i]

        # Butterfly reduction tree: dynamic for any WARP_SIZE (32, 64, etc.)
        # Start with half the warp size and reduce by half each step
        offset = WARP_SIZE // 2
        while offset > 0:
            max_val = max(max_val, shuffle_xor(max_val, offset))
            offset //= 2

        # All threads now have the maximum value across the entire warp
        output[global_i] = max_val

```

This solution demonstrates how `shuffle_xor()` creates efficient parallel reduction trees with \\(O(\\log n)\\) complexity.

**Complete algorithm analysis:**

```mojo
if global_i < size:
    max_val = input[global_i]  # Start with local value

    # Butterfly reduction tree: dynamic for any WARP_SIZE
    offset = WARP_SIZE // 2
    while offset > 0:
        max_val = max(max_val, shuffle_xor(max_val, offset))
        offset //= 2

    output[global_i] = max_val  # All lanes have global maximum
```

**Butterfly execution trace (8-lane example, values [0,2,4,6,8,10,12,1000]):**

```
Initial state:
  Lane 0: max_val = 0,    Lane 1: max_val = 2
  Lane 2: max_val = 4,    Lane 3: max_val = 6
  Lane 4: max_val = 8,    Lane 5: max_val = 10
  Lane 6: max_val = 12,   Lane 7: max_val = 1000

Step 1: shuffle_xor(max_val, 4) - Halves exchange
  Lane 0<->4: max(0,8)=8,     Lane 1<->5: max(2,10)=10
  Lane 2<->6: max(4,12)=12,   Lane 3<->7: max(6,1000)=1000
  Lane 4<->0: max(8,0)=8,     Lane 5<->1: max(10,2)=10
  Lane 6<->2: max(12,4)=12,   Lane 7<->3: max(1000,6)=1000

Step 2: shuffle_xor(max_val, 2) - Quarters exchange
  Lane 0<->2: max(8,12)=12,   Lane 1<->3: max(10,1000)=1000
  Lane 2<->0: max(12,8)=12,   Lane 3<->1: max(1000,10)=1000
  Lane 4<->6: max(8,12)=12,   Lane 5<->7: max(10,1000)=1000
  Lane 6<->4: max(12,8)=12,   Lane 7<->5: max(1000,10)=1000

Step 3: shuffle_xor(max_val, 1) - Pairs exchange
  Lane 0<->1: max(12,1000)=1000,  Lane 1<->0: max(1000,12)=1000
  Lane 2<->3: max(12,1000)=1000,  Lane 3<->2: max(1000,12)=1000
  Lane 4<->5: max(12,1000)=1000,  Lane 5<->4: max(1000,12)=1000
  Lane 6<->7: max(12,1000)=1000,  Lane 7<->6: max(1000,12)=1000

Final result: All lanes have max_val = 1000
```

**Mathematical insight:**This implements the parallel reduction operator with butterfly communication:
\\[\\Large \\text{Reduce}(\\oplus, [a_0, a_1, \\ldots, a_{n-1}]) = a_0 \\oplus a_1 \\oplus \\cdots \\oplus a_{n-1}\\]

Where \\(\\oplus\\) is the `max` operation and the butterfly pattern ensures optimal \\(O(\\log n)\\) complexity.

**Why butterfly reduction is superior:**

1. **Logarithmic complexity**: \\(O(\\log n)\\) vs \\(O(n)\\) for sequential reduction
2. **Perfect load balancing**: Every lane participates equally at each step
3. **No memory bottlenecks**: Pure register-to-register communication
4. **Hardware optimized**: Maps directly to GPU butterfly networks

**Performance characteristics:**

- **Steps**: \\(\\log_2(\\text{WARP\_SIZE})\\) (e.g., 5 for 32-thread, 6 for 64-thread warp)
- **Latency per step**: 1 cycle (register exchange + comparison)
- **Total latency**: \\(\\log_2(\\text{WARP\_SIZE})\\) cycles vs \\((\\text{WARP\_SIZE}-1)\\) cycles for sequential
- **Parallelism**: All lanes active throughout the algorithm

### 3. Butterfly conditional maximum

#### Configuration

- Vector size: `SIZE_2 = 64` (multi-block scenario)
- Grid configuration: `BLOCKS_PER_GRID_2 = (2, 1)` blocks per grid
- Block configuration: `THREADS_PER_BLOCK_2 = (WARP_SIZE, 1)` threads per block

#### Tips

#### 1. **Dual-track butterfly reduction**

This puzzle requires tracking TWO different values simultaneously through the butterfly tree. Think about how you can run multiple reductions in parallel.

**Key questions:**

- How can you maintain both maximum and minimum values during the reduction?
- Can you use the same butterfly pattern for both operations?
- What variables do you need to track?

#### 2. **Conditional output logic**

After completing the butterfly reduction, you need to output different values based on lane parity.

**Consider:**

- How do you determine if a lane is even or odd?
- Which lanes should output the maximum vs minimum?
- How do you access the lane ID?

#### 3. **Butterfly reduction for both min and max**

Example scenario is efficiently computing both min and max in parallel using the same butterfly communication pattern.

**Think about:**

- Do you need separate shuffle operations for min and max?
- Can you reuse the same neighbor values for both operations?
- How do you ensure both reductions complete correctly?

#### 4. **Multi-block boundary considerations**

This puzzle uses multiple blocks. Consider how this affects the reduction scope.

**Important considerations:**

- What's the scope of each butterfly reduction?
- How does the block structure affect lane numbering?
- Are you computing global or per-block min/max values?

**Test the butterfly conditional maximum:**

  
    pixi NVIDIA (default)
    pixi AMD
    uv
  
  

```bash
pixi run p26 --conditional-max
```

  
  

```bash
pixi run -e amd p26 --conditional-max
```

  
  

```bash
uv run poe p26 --conditional-max
```

  

Expected output when solved:

```txt
WARP_SIZE:  32
SIZE_2:  64
output: [9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0]
expected: [9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0]
OK Butterfly conditional max test passed!
```

#### Reference implementation (example)


```mojo
fn butterfly_conditional_max
    layout: Layout, size: Int
:
    """
    Conditional butterfly maximum: Perform butterfly max reduction, but only store result
    in even-numbered lanes. Odd-numbered lanes store the minimum value seen.
    Demonstrates conditional logic combined with butterfly communication patterns.
    """
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    lane = lane_id()

    if global_i < size:
        current_val = input[global_i]
        min_val = current_val

        # Butterfly reduction for both maximum and minimum: dynamic for any WARP_SIZE
        offset = WARP_SIZE // 2
        while offset > 0:
            neighbor_val = shuffle_xor(current_val, offset)
            current_val = max(current_val, neighbor_val)

            min_neighbor_val = shuffle_xor(min_val, offset)
            min_val = min(min_val, min_neighbor_val)

            offset //= 2

        # Conditional output: max for even lanes, min for odd lanes
        if lane % 2 == 0:
            output[global_i] = current_val  # Maximum
        else:
            output[global_i] = min_val  # Minimum

```

This solution demonstrates advanced butterfly reduction with dual tracking and conditional output.

**Complete algorithm analysis:**

```mojo
if global_i < size:
    current_val = input[global_i]
    min_val = current_val  # Track minimum separately

    # Butterfly reduction for both max and min log_2(WARP_SIZE}) steps)
    offset = WARP_SIZE // 2
    while offset > 0:
        neighbor_val = shuffle_xor(current_val, offset)
        current_val = max(current_val, neighbor_val)    # Max reduction

        min_neighbor_val = shuffle_xor(min_val, offset)
        min_val = min(min_val, min_neighbor_val)        # Min reduction

        offset //= 2

    # Conditional output based on lane parity
    if lane % 2 == 0:
        output[global_i] = current_val  # Even lanes: maximum
    else:
        output[global_i] = min_val      # Odd lanes: minimum
```

**Dual reduction execution trace (4-lane example, values [3, 1, 7, 2]):**

```
Initial state:
  Lane 0: current_val=3, min_val=3
  Lane 1: current_val=1, min_val=1
  Lane 2: current_val=7, min_val=7
  Lane 3: current_val=2, min_val=2

Step 1: shuffle_xor(current_val, 2) and shuffle_xor(min_val, 2) - Halves exchange
  Lane 0<->2: max_neighbor=7, min_neighbor=7 -> current_val=max(3,7)=7, min_val=min(3,7)=3
  Lane 1<->3: max_neighbor=2, min_neighbor=2 -> current_val=max(1,2)=2, min_val=min(1,2)=1
  Lane 2<->0: max_neighbor=3, min_neighbor=3 -> current_val=max(7,3)=7, min_val=min(7,3)=3
  Lane 3<->1: max_neighbor=1, min_neighbor=1 -> current_val=max(2,1)=2, min_val=min(2,1)=1

Step 2: shuffle_xor(current_val, 1) and shuffle_xor(min_val, 1) - Pairs exchange
  Lane 0<->1: max_neighbor=2, min_neighbor=1 -> current_val=max(7,2)=7, min_val=min(3,1)=1
  Lane 1<->0: max_neighbor=7, min_neighbor=3 -> current_val=max(2,7)=7, min_val=min(1,3)=1
  Lane 2<->3: max_neighbor=2, min_neighbor=1 -> current_val=max(7,2)=7, min_val=min(3,1)=1
  Lane 3<->2: max_neighbor=7, min_neighbor=3 -> current_val=max(2,7)=7, min_val=min(1,3)=1

Final result: All lanes have current_val=7 (global max) and min_val=1 (global min)
```

**Dynamic algorithm**(works for any WARP_SIZE):

```mojo
offset = WARP_SIZE // 2
while offset > 0:
    neighbor_val = shuffle_xor(current_val, offset)
    current_val = max(current_val, neighbor_val)

    min_neighbor_val = shuffle_xor(min_val, offset)
    min_val = min(min_val, min_neighbor_val)

    offset //= 2
```

**Mathematical insight:**This implements dual parallel reduction with conditional demultiplexing:
\\[\\Large \\begin{align}
\\text{max\_result} &= \\max_{i=0}^{n-1} \\text{input}[i] \\\\
\\text{min\_result} &= \\min_{i=0}^{n-1} \\text{input}[i] \\\\
\\text{output}[i] &= \\text{lane\_parity}(i) \\; \text{?} \\; \\text{min\_result} : \\text{max\_result}
\\end{align}\\]

**Why dual butterfly reduction works:**

1. **Independent reductions**: Max and min reductions are mathematically independent
2. **Parallel execution**: Both can use the same butterfly communication pattern
3. **Shared communication**: Same shuffle operations serve both reductions
4. **Conditional output**: Lane parity determines which result to output

**Performance characteristics:**

- **Communication steps**: \\(\\log_2(\\text{WARP\_SIZE})\\) (same as single reduction)
- **Computation per step**: 2 operations (max + min) vs 1 for single reduction
- **Memory efficiency**: 2 registers per thread vs complex shared memory approaches
- **Output flexibility**: Different lanes can output different reduction results

### Summary

The `shuffle_xor()` primitive enables powerful butterfly communication patterns that form the foundation of efficient parallel algorithms. Through these three problems, you've learned:

#### **Core Butterfly Patterns**

1. **Pair Exchange**(`shuffle_xor(value, 1)`):
   - Creates perfect adjacent pairs: (0,1), (2,3), (4,5), ...
   - \\(O(1)\\) complexity with zero memory overhead
   - Foundation for sorting networks and data reorganization

2. **Tree Reduction**(dynamic offsets: `WARP_SIZE/2`  `1`):
   - Logarithmic parallel reduction: \\(O(\\log n)\\) vs \\(O(n)\\) sequential
   - Works for any associative operation (max, min, sum, etc.)
   - Optimal load balancing across all warp lanes

3. **Conditional Multi-Reduction**(dual tracking + lane parity):
   - Simultaneous multiple reductions in parallel
   - Conditional output based on thread characteristics
   - Advanced coordination without explicit synchronization

#### **Key Algorithmic Insights**

**XOR Communication Properties:**

- `shuffle_xor(value, mask)` creates symmetric, non-overlapping pairs
- Each mask creates a unique communication topology
- Butterfly networks emerge naturally from binary XOR patterns

**Dynamic Algorithm Design:**

```mojo
offset = WARP_SIZE // 2
while offset > 0:
    neighbor_val = shuffle_xor(current_val, offset)
    current_val = operation(current_val, neighbor_val)
    offset //= 2
```

**Performance Advantages:**

- **Hardware optimization**: Direct register-to-register communication
- **No synchronization**: SIMT execution guarantees correctness
- **Scalable complexity**: \\(O(\\log n)\\) for any WARP_SIZE (32, 64, etc.)
- **Memory efficiency**: Zero shared memory requirements

#### **Practical Applications**

These butterfly patterns are fundamental to:

- **Parallel reductions**: Sum, max, min, logical operations
- **Prefix/scan operations**: Cumulative sums, parallel sorting
- **FFT algorithms**: Signal processing and convolution
- **Bitonic sorting**: Parallel sorting networks
- **Graph algorithms**: Tree traversals and connectivity

The `shuffle_xor()` primitive transforms complex parallel coordination into elegant, hardware-optimized communication patterns that scale efficiently across different GPU architectures.

## `warp.prefix_sum()` Hardware-Optimized Parallel Scan

For warp-level parallel scan operations we can use `prefix_sum()` to replace complex shared memory algorithms with hardware-optimized primitives. This powerful operation enables efficient cumulative computations, parallel partitioning, and advanced coordination algorithms that would otherwise require dozens of lines of shared memory and synchronization code.

**Key insight:**_The [prefix_sum()](https://docs.modular.com/mojo/stdlib/gpu/warp/prefix_sum) operation leverages hardware-accelerated parallel scan to compute cumulative operations across warp lanes with \\(O(\\log n)\\) complexity, replacing complex multi-phase algorithms with single function calls._

> **What is parallel scan?**[Parallel scan (prefix sum)](https://en.wikipedia.org/wiki/Prefix_sum) is a fundamental parallel primitive that computes cumulative operations across data elements. For addition, it transforms `[a, b, c, d]` into `[a, a+b, a+b+c, a+b+c+d]`. This operation is essential for parallel algorithms like stream compaction, quicksort partitioning, and parallel sorting.

### Key concepts

In this puzzle, you'll learn:

- **Hardware-optimized parallel scan**with `prefix_sum()`
- **Inclusive vs exclusive prefix sum**patterns
- **Warp-level stream compaction**for data reorganization
- **Advanced parallel partitioning**combining multiple warp primitives
- **Single-warp algorithm optimization**replacing complex shared memory

This transforms multi-phase shared memory algorithms into elegant single-function calls, enabling efficient parallel scan operations without explicit synchronization.

### 1. Warp inclusive prefix sum

#### Configuration

- Vector size: `SIZE = WARP_SIZE` (32 or 64 depending on GPU)
- Grid configuration: `(1, 1)` blocks per grid
- Block configuration: `(WARP_SIZE, 1)` threads per block
- Data type: `DType.float32`
- Layout: `Layout.row_major(SIZE)` (1D row-major)

#### The `prefix_sum` advantage

Traditional prefix sum requires complex multi-phase shared memory algorithms. In Puzzle 14, we implemented this the hard way with explicit shared memory management:

```mojo
fn prefix_sum_simple
    layout: Layout
:
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x
    shared = LayoutTensor[
        dtype,
        Layout.row_major(TPB),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    if global_i < size:
        shared[local_i] = a[global_i]

    barrier()

    offset = UInt(1)
    for i in range(Int(log2(Scalardtype))):
        var current_val: output.element_type = 0
        if local_i >= offset and local_i < size:
            current_val = shared[local_i - offset]  # read

        barrier()
        if local_i >= offset and local_i < size:
            shared[local_i] += current_val

        barrier()
        offset *= 2

    if global_i < size:
        output[global_i] = shared[local_i]

```

**Problems with traditional approach:**

- **Memory overhead**: Requires shared memory allocation
- **Multiple barriers**: Complex multi-phase synchronization
- **Complex indexing**: Manual stride calculation and boundary checking
- **Poor scaling**: \\(O(\\log n)\\) phases with barriers between each

With `prefix_sum()`, parallel scan becomes trivial:

```mojo
# Hardware-optimized approach - single function call!
current_val = input[global_i]
scan_result = prefix_sumexclusive=False
output[global_i] = scan_result
```

**Benefits of prefix_sum:**

- **Zero memory overhead**: Hardware-accelerated computation
- **No synchronization**: Single atomic operation
- **Hardware optimized**: Leverages specialized scan units
- **Perfect scaling**: Works for any `WARP_SIZE` (32, 64, etc.)

#### Tips

#### 1. **Understanding prefix_sum parameters**

The `prefix_sum()` function has an important template parameter that controls the scan type.

**Key questions:**

- What's the difference between inclusive and exclusive prefix sum?
- Which parameter controls this behavior?
- For inclusive scan, what should each lane output?

**Hint**: Look at the function signature and consider what "inclusive" means for cumulative operations.

#### 2. **Single warp limitation**

This hardware primitive only works within a single warp. Consider the implications.

**Think about:**

- What happens if you have multiple warps?
- Why is this limitation important to understand?
- How would you extend this to multi-warp scenarios?

#### 3. **Data type considerations**

The `prefix_sum` function may require specific data types for optimal performance.

**Consider:**

- What data type does your input use?
- Does `prefix_sum` expect a specific scalar type?
- How do you handle type conversions if needed?

**Test the warp inclusive prefix sum:**

  
    pixi NVIDIA (default)
    pixi AMD
    pixi Apple
    uv
  
  

```bash
pixi run p26 --prefix-sum
```

  
  

```bash
pixi run -e amd p26 --prefix-sum
```

  
  

```bash
pixi run -e apple p26 --prefix-sum
```

  
  

```bash
uv run poe p26 --prefix-sum
```

  

Expected output when solved:

```txt
WARP_SIZE:  32
SIZE:  32
output: [1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0, 45.0, 55.0, 66.0, 78.0, 91.0, 105.0, 120.0, 136.0, 153.0, 171.0, 190.0, 210.0, 231.0, 253.0, 276.0, 300.0, 325.0, 351.0, 378.0, 406.0, 435.0, 465.0, 496.0, 528.0]
expected: [1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0, 45.0, 55.0, 66.0, 78.0, 91.0, 105.0, 120.0, 136.0, 153.0, 171.0, 190.0, 210.0, 231.0, 253.0, 276.0, 300.0, 325.0, 351.0, 378.0, 406.0, 435.0, 465.0, 496.0, 528.0]
OK Warp inclusive prefix sum test passed!
```

#### Reference implementation (example)


```mojo
fn warp_inclusive_prefix_sum
    layout: Layout, size: Int
:
    """
    Inclusive prefix sum using warp primitive: Each thread gets sum of all elements up to and including its position.
    Compare this to Puzzle 12's complex shared memory + barrier approach.

    Puzzle 12 approach:
    - Shared memory allocation
    - Multiple barrier synchronizations
    - Log(n) iterations with manual tree reduction
    - Complex multi-phase algorithm

    Warp prefix_sum approach:
    - Single function call!
    - Hardware-optimized parallel scan
    - Automatic synchronization
    - O(log n) complexity, but implemented in hardware.

    NOTE: This implementation only works correctly within a single warp (WARP_SIZE threads).
    For multi-warp scenarios, additional coordination would be needed.
    """
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)

    if global_i < size:
        current_val = input[global_i]

        # This one call replaces ~30 lines of complex shared memory logic from Puzzle 12!
        # But it only works within the current warp (WARP_SIZE threads)
        scan_result = prefix_sumexclusive=False
        )

        output[global_i] = scan_result

```

This solution demonstrates how `prefix_sum()` replaces complex multi-phase algorithms with a single hardware-optimized function call.

**Algorithm breakdown:**

```mojo
if global_i < size:
    current_val = input[global_i]

    # This one call replaces ~30 lines of complex shared memory logic from Puzzle 14!
    # But it only works within the current warp (WARP_SIZE threads)
    scan_result = prefix_sumexclusive=False
    )

    output[global_i] = scan_result
```

**SIMT execution deep dive:**

```
Input: [1, 2, 3, 4, 5, 6, 7, 8, ...]

Cycle 1: All lanes load their values simultaneously
  Lane 0: current_val = 1
  Lane 1: current_val = 2
  Lane 2: current_val = 3
  Lane 3: current_val = 4
  ...
  Lane 31: current_val = 32

Cycle 2: prefix_sum[exclusive=False] executes (hardware-accelerated)
  Lane 0: scan_result = 1 (sum of elements 0 to 0)
  Lane 1: scan_result = 3 (sum of elements 0 to 1: 1+2)
  Lane 2: scan_result = 6 (sum of elements 0 to 2: 1+2+3)
  Lane 3: scan_result = 10 (sum of elements 0 to 3: 1+2+3+4)
  ...
  Lane 31: scan_result = 528 (sum of elements 0 to 31)

Cycle 3: Store results
  Lane 0: output[0] = 1
  Lane 1: output[1] = 3
  Lane 2: output[2] = 6
  Lane 3: output[3] = 10
  ...
```

**Mathematical insight:**This implements the inclusive prefix sum operation:
\\[\\Large \\text{output}[i] = \\sum_{j=0}^{i} \\text{input}[j]\\]

**Comparison with Puzzle 14's approach:**

- **Puzzle 14**: ~30 lines of shared memory + multiple barriers + complex indexing
- **Warp primitive**: 1 function call with hardware acceleration
- **Performance**: Same \\(O(\\log n)\\) complexity, but implemented in specialized hardware
- **Memory**: Zero shared memory usage vs explicit allocation

**Evolution from Puzzle 12:**This demonstrates the power of modern GPU architectures - what required careful manual implementation in Puzzle 12 is now a single hardware-accelerated primitive. The warp-level `prefix_sum()` gives you the same algorithmic benefits with zero implementation complexity.

**Why prefix_sum is superior:**

1. **Hardware acceleration**: Dedicated scan units on modern GPUs
2. **Zero memory overhead**: No shared memory allocation required
3. **Automatic synchronization**: No explicit barriers needed
4. **Perfect scaling**: Works optimally for any `WARP_SIZE`

**Performance characteristics:**

- **Latency**: ~1-2 cycles (hardware scan units)
- **Bandwidth**: Zero memory traffic (register-only operation)
- **Parallelism**: All `WARP_SIZE` lanes participate simultaneously
- **Scalability**: \\(O(\\log n)\\) complexity with hardware optimization

**Important limitation**: This primitive only works within a single warp. For multi-warp scenarios, you would need additional coordination between warps.

### 2. Warp partition

#### Configuration

- Vector size: `SIZE = WARP_SIZE` (32 or 64 depending on GPU)
- Grid configuration: `(1, 1)` blocks per grid
- Block configuration: `(WARP_SIZE, 1)` threads per block

#### Tips

#### 1. **Multi-phase algorithm structure**

This algorithm requires several coordinated phases. Think about the logical steps needed for partitioning.

**Key phases to consider:**

- How do you identify which elements belong to which partition?
- How do you calculate positions within each partition?
- How do you determine the total size of the left partition?
- How do you write elements to their final positions?

#### 2. **Predicate creation**

You need to create boolean predicates to identify partition membership.

**Think about:**

- How do you represent "this element belongs to the left partition"?
- How do you represent "this element belongs to the right partition"?
- What data type should you use for predicates that work with `prefix_sum`?

#### 3. **Combining shuffle_xor and prefix_sum**

This algorithm uses both warp primitives for different purposes.

**Consider:**

- What is `shuffle_xor` used for in this context?
- What is `prefix_sum` used for in this context?
- How do these two operations work together?

#### 4. **Position calculation**

The trickiest part is calculating where each element should be written in the output.

**Key insights:**

- Left partition elements: What determines their final position?
- Right partition elements: How do you offset them correctly?
- How do you combine local positions with partition boundaries?

**Test the warp partition:**

  
    uv
    pixi
  
  

```bash
uv run poe p26 --partition
```

  
  

```bash
pixi run p26 --partition
```

  

Expected output when solved:

```txt
WARP_SIZE:  32
SIZE:  32
output: HostBuffer([3.0, 1.0, 2.0, 4.0, 0.0, 3.0, 1.0, 4.0, 3.0, 1.0, 2.0, 4.0, 0.0, 3.0, 1.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0, 11.0, 12.0, 13.0, 7.0, 8.0, 9.0, 6.0, 10.0, 11.0, 12.0, 13.0])
expected: HostBuffer([3.0, 1.0, 2.0, 4.0, 0.0, 3.0, 1.0, 4.0, 3.0, 1.0, 2.0, 4.0, 0.0, 3.0, 1.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0, 11.0, 12.0, 13.0, 7.0, 8.0, 9.0, 6.0, 10.0, 11.0, 12.0, 13.0])
pivot: 5.0
OK Warp partition test passed!
```

#### Reference implementation (example)


```mojo
fn warp_partition
    layout: Layout, size: Int
:
    """
    Single-warp parallel partitioning using BOTH shuffle_xor AND prefix_sum.
    This implements a warp-level quicksort partition step that places elements < pivot
    on the left and elements >= pivot on the right.

    ALGORITHM COMPLEXITY - combines two advanced warp primitives:
    1. shuffle_xor(): Butterfly pattern for warp-level reductions
    2. prefix_sum(): Warp-level exclusive scan for position calculation.

    This demonstrates the power of warp primitives for sophisticated parallel algorithms
    within a single warp (works for any WARP_SIZE: 32, 64, etc.).

    Example with pivot=5:
    Input:  [3, 7, 1, 8, 2, 9, 4, 6]
    Result: [3, 1, 2, 4, 7, 8, 9, 6] (< pivot | >= pivot).
    """
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)

    if global_i < size:
        current_val = input[global_i]

        # Phase 1: Create warp-level predicates
        predicate_left = Float32(1.0) if current_val < pivot else Float32(0.0)
        predicate_right = Float32(1.0) if current_val >= pivot else Float32(0.0)

        # Phase 2: Warp-level prefix sum to get positions within warp
        warp_left_pos = prefix_sumexclusive=True
        warp_right_pos = prefix_sumexclusive=True

        # Phase 3: Get total left count using shuffle_xor reduction
        warp_left_total = predicate_left

        # Butterfly reduction to get total across the warp: dynamic for any WARP_SIZE
        offset = WARP_SIZE // 2
        while offset > 0:
            warp_left_total += shuffle_xor(warp_left_total, offset)
            offset //= 2

        # Phase 4: Write to output positions
        if current_val < pivot:
            # Left partition: use warp-level position
            output[Int(warp_left_pos)] = current_val
        else:
            # Right partition: offset by total left count + right position
            output[Int(warp_left_total + warp_right_pos)] = current_val

```

This solution demonstrates advanced coordination between multiple warp primitives to implement sophisticated parallel algorithms.

**Complete algorithm analysis:**

```mojo
if global_i < size:
    current_val = input[global_i]

    # Phase 1: Create warp-level predicates
    predicate_left = Float32(1.0) if current_val < pivot else Float32(0.0)
    predicate_right = Float32(1.0) if current_val >= pivot else Float32(0.0)

    # Phase 2: Warp-level prefix sum to get positions within warp
    warp_left_pos = prefix_sumexclusive=True
    warp_right_pos = prefix_sumexclusive=True

    # Phase 3: Get total left count using shuffle_xor reduction
    warp_left_total = predicate_left

    # Butterfly reduction to get total across the warp: dynamic for any WARP_SIZE
    offset = WARP_SIZE // 2
    while offset > 0:
        warp_left_total += shuffle_xor(warp_left_total, offset)
        offset //= 2

    # Phase 4: Write to output positions
    if current_val < pivot:
        # Left partition: use warp-level position
        output[Int(warp_left_pos)] = current_val
    else:
        # Right partition: offset by total left count + right position
        output[Int(warp_left_total + warp_right_pos)] = current_val
```

**Multi-phase execution trace (8-lane example, pivot=5, values [3,7,1,8,2,9,4,6]):**

```
Initial state:
  Lane 0: current_val=3 (< 5)  Lane 1: current_val=7 (>= 5)
  Lane 2: current_val=1 (< 5)  Lane 3: current_val=8 (>= 5)
  Lane 4: current_val=2 (< 5)  Lane 5: current_val=9 (>= 5)
  Lane 6: current_val=4 (< 5)  Lane 7: current_val=6 (>= 5)

Phase 1: Create predicates
  Lane 0: predicate_left=1.0, predicate_right=0.0
  Lane 1: predicate_left=0.0, predicate_right=1.0
  Lane 2: predicate_left=1.0, predicate_right=0.0
  Lane 3: predicate_left=0.0, predicate_right=1.0
  Lane 4: predicate_left=1.0, predicate_right=0.0
  Lane 5: predicate_left=0.0, predicate_right=1.0
  Lane 6: predicate_left=1.0, predicate_right=0.0
  Lane 7: predicate_left=0.0, predicate_right=1.0

Phase 2: Exclusive prefix sum for positions
  warp_left_pos:  [0, 0, 1, 1, 2, 2, 3, 3]
  warp_right_pos: [0, 0, 0, 1, 1, 2, 2, 3]

Phase 3: Butterfly reduction for left total
  Initial: [1, 0, 1, 0, 1, 0, 1, 0]
  After reduction: all lanes have warp_left_total = 4

Phase 4: Write to output positions
  Lane 0: current_val=3 < pivot -> output[0] = 3
  Lane 1: current_val=7 >= pivot -> output[4+0] = output[4] = 7
  Lane 2: current_val=1 < pivot -> output[1] = 1
  Lane 3: current_val=8 >= pivot -> output[4+1] = output[5] = 8
  Lane 4: current_val=2 < pivot -> output[2] = 2
  Lane 5: current_val=9 >= pivot -> output[4+2] = output[6] = 9
  Lane 6: current_val=4 < pivot -> output[3] = 4
  Lane 7: current_val=6 >= pivot -> output[4+3] = output[7] = 6

Final result: [3, 1, 2, 4, 7, 8, 9, 6] (< pivot | >= pivot)
```

**Mathematical insight:**This implements parallel partitioning with dual warp primitives:
\\[\\Large \\begin{align}
\\text{left\\_pos}[i] &= \\text{prefix\\_sum}_{\\text{exclusive}}(\\text{predicate\\_left}[i]) \\\\
\\text{right\\_pos}[i] &= \\text{prefix\\_sum}_{\\text{exclusive}}(\\text{predicate\\_right}[i]) \\\\
\\text{left\\_total} &= \\text{butterfly\\_reduce}(\\text{predicate\\_left}) \\\\
\\text{final\\_pos}[i] &= \\begin{cases}
\\text{left\\_pos}[i] & \\text{if } \\text{input}[i] < \\text{pivot} \\\\
\\text{left\\_total} + \\text{right\\_pos}[i] & \\text{if } \\text{input}[i] \\geq \\text{pivot}
\\end{cases}
\\end{align}\\]

**Why this multi-primitive approach works:**

1. **Predicate creation**: Identifies partition membership for each element
2. **Exclusive prefix sum**: Calculates relative positions within each partition
3. **Butterfly reduction**: Computes partition boundary (total left count)
4. **Coordinated write**: Combines local positions with global partition structure

**Algorithm complexity:**

- **Phase 1**: \\(O(1)\\) - Predicate creation
- **Phase 2**: \\(O(\\log n)\\) - Hardware-accelerated prefix sum
- **Phase 3**: \\(O(\\log n)\\) - Butterfly reduction with `shuffle_xor`
- **Phase 4**: \\(O(1)\\) - Coordinated write
- **Total**: \\(O(\\log n)\\) with excellent constants

**Performance characteristics:**

- **Communication steps**: \\(2 \\times \\log_2(\\text{WARP\_SIZE})\\) (prefix sum + butterfly reduction)
- **Memory efficiency**: Zero shared memory, all register-based
- **Parallelism**: All lanes active throughout algorithm
- **Scalability**: Works for any `WARP_SIZE` (32, 64, etc.)

**Practical applications:**This pattern is fundamental to:

- **Quicksort partitioning**: Core step in parallel sorting algorithms
- **Stream compaction**: Removing null/invalid elements from data streams
- **Parallel filtering**: Separating data based on complex predicates
- **Load balancing**: Redistributing work based on computational requirements

### Summary

The `prefix_sum()` primitive enables hardware-accelerated parallel scan operations that replace complex multi-phase algorithms with single function calls. Through these two problems, you've learned:

#### **Core Prefix Sum Patterns**

1. **Inclusive Prefix Sum**(`prefix_sum[exclusive=False]`):
   - Hardware-accelerated cumulative operations
   - Replaces ~30 lines of shared memory code with single function call
   - \\(O(\\log n)\\) complexity with specialized hardware optimization

2. **Advanced Multi-Primitive Coordination**(combining `prefix_sum` + `shuffle_xor`):
   - Sophisticated parallel algorithms within single warp
   - Exclusive scan for position calculation + butterfly reduction for totals
   - Complex partitioning operations with optimal parallel efficiency

#### **Key Algorithmic Insights**

**Hardware Acceleration Benefits:**

- `prefix_sum()` leverages dedicated scan units on modern GPUs
- Zero shared memory overhead compared to traditional approaches
- Automatic synchronization without explicit barriers

**Multi-Primitive Coordination:**

```mojo
# Phase 1: Create predicates for partition membership
predicate = 1.0 if condition else 0.0

# Phase 2: Use prefix_sum for local positions
local_pos = prefix_sumexclusive=True

# Phase 3: Use shuffle_xor for global totals
global_total = butterfly_reduce(predicate)

# Phase 4: Combine for final positioning
final_pos = local_pos + partition_offset
```

**Performance Advantages:**

- **Hardware optimization**: Specialized scan units vs software implementation
- **Memory efficiency**: Register-only operations vs shared memory allocation
- **Scalable complexity**: \\(O(\\log n)\\) with hardware acceleration
- **Single-warp optimization**: Perfect for algorithms within `WARP_SIZE` limits

#### **Practical Applications**

These prefix sum patterns are fundamental to:

- **Parallel scan operations**: Cumulative sums, products, min/max scans
- **Stream compaction**: Parallel filtering and data reorganization
- **Quicksort partitioning**: Core parallel sorting algorithm building block
- **Parallel algorithms**: Load balancing, work distribution, data restructuring

The combination of `prefix_sum()` and `shuffle_xor()` demonstrates how modern GPU warp primitives can implement sophisticated parallel algorithms with minimal code complexity and optimal performance characteristics.
