---
title: "Warp Communication"
description: "Understand the fundamental communication patterns within GPU warps:"
---

# Warp Communication

Understand the fundamental communication patterns within GPU warps:

## Overview

**Puzzle 25: Warp Communication Primitives**introduces advanced GPU **warp-level communication operations**- hardware-accelerated primitives that enable efficient data exchange and coordination patterns within warps. You'll learn about using [shuffle_down](https://docs.modular.com/mojo/stdlib/gpu/warp/shuffle_down) and [broadcast](https://docs.modular.com/mojo/stdlib/gpu/warp/broadcast) to implement neighbor communication and collective coordination without complex shared memory patterns.

**Part VII: GPU Warp Communication**introduces warp-level data movement operations within thread groups. You'll learn to replace complex shared memory + indexing + boundary checking patterns with efficient warp communication calls that leverage hardware-optimized data movement.

**Key insight:**_GPU warps execute in lockstep - Mojo's warp communication operations use this synchronization to provide efficient data exchange primitives with automatic boundary handling and zero explicit synchronization._

## What you'll learn

### **Warp communication model**

Understand the fundamental communication patterns within GPU warps:

```
GPU Warp (32 threads, SIMT lockstep execution)
|-- Lane 0  --shuffle_down--> Lane 1  --shuffle_down--> Lane 2
|-- Lane 1  --shuffle_down--> Lane 2  --shuffle_down--> Lane 3
|-- Lane 2  --shuffle_down--> Lane 3  --shuffle_down--> Lane 4
|   ...
`-- Lane 31 --shuffle_down--> undefined (boundary)

Broadcast pattern:
Lane 0 --broadcast--> All lanes (0, 1, 2, ..., 31)
```

**Hardware reality:**

- **Register-to-register communication**: Data moves directly between thread registers
- **Zero memory overhead**: No shared memory allocation required
- **Automatic boundary handling**: Hardware manages warp edge cases
- **Single-cycle operations**: Communication happens in one instruction cycle

### **Warp communication operations in Mojo**

Learn the core communication primitives from `gpu.primitives.warp`:

1. **[`shuffle_down(value, offset)`](https://docs.modular.com/mojo/stdlib/gpu/warp/shuffle_down)**: Get value from lane at higher index (neighbor access)
2. **[`broadcast(value)`](https://docs.modular.com/mojo/stdlib/gpu/warp/broadcast)**: Share lane 0's value with all other lanes (one-to-many)
3. **[`shuffle_idx(value, lane)`](https://docs.modular.com/mojo/stdlib/gpu/warp/shuffle_idx)**: Get value from specific lane (random access)
4. **[`shuffle_up(value, offset)`](https://docs.modular.com/mojo/stdlib/gpu/warp/shuffle_up)**: Get value from lane at lower index (reverse neighbor)

> **Note:**This puzzle focuses on `shuffle_down()` and `broadcast()` as the most commonly used communication patterns. For complete coverage of all warp operations, see the [Mojo GPU Warp Documentation](https://docs.modular.com/mojo/stdlib/gpu/warp/).

### **Performance transformation example**

```mojo
# Complex neighbor access pattern (traditional approach):
shared = LayoutTensor[
    dtype,
    Layout.row_major(WARP_SIZE),
    MutAnyOrigin,
    address_space = AddressSpace.SHARED,
].stack_allocation()
shared[local_i] = input[global_i]
barrier()
if local_i < WARP_SIZE - 1:
    next_value = shared[local_i + 1]  # Neighbor access
    result = next_value - shared[local_i]
else:
    result = 0  # Boundary handling
barrier()

# Warp communication eliminates all this complexity:
current_val = input[global_i]
next_val = shuffle_down(current_val, 1)  # Direct neighbor access
if lane < WARP_SIZE - 1:
    result = next_val - current_val
else:
    result = 0
```

### **When warp communication excels**

Learn the performance characteristics:

| Communication Pattern | Traditional | Warp Operations |
|----------------------|-------------|-----------------|
| Neighbor access | Shared memory | Register-to-register |
| Stencil operations | Complex indexing | Simple shuffle patterns |
| Block coordination | Barriers + shared | Single broadcast |
| Boundary handling | Manual checks | Hardware automatic |

## Prerequisites

Before diving into warp communication, ensure you're comfortable with:

- **Part VII warp fundamentals**: Understanding SIMT execution and basic warp operations (see Puzzle 24)
- **GPU thread hierarchy**: Blocks, warps, and lane numbering
- **LayoutTensor operations**: Loading, storing, and tensor manipulation
- **Boundary condition handling**: Managing edge cases in parallel algorithms

## Learning path

### **1. Neighbor communication with shuffle_down**

** [Warp Shuffle Down](#warpshuffledown-one-to-one-communication)**

Learn neighbor-based communication patterns for stencil operations and finite differences.

**What you'll learn:**

- Using `shuffle_down()` for accessing adjacent lane data
- Implementing finite differences and moving averages
- Handling warp boundaries automatically
- Multi-offset shuffling for extended neighbor access

**Key pattern:**

```mojo
current_val = input[global_i]
next_val = shuffle_down(current_val, 1)
if lane < WARP_SIZE - 1:
    result = compute_with_neighbors(current_val, next_val)
```

### **2. Collective coordination with broadcast**

** [Warp Broadcast](#warpbroadcast-one-to-many-communication)**

Learn one-to-many communication patterns for block-level coordination and collective decision-making.

**What you'll learn:**

- Using `broadcast()` for sharing computed values across lanes
- Implementing block-level statistics and collective decisions
- Combining broadcast with conditional logic
- Advanced broadcast-shuffle coordination patterns

**Key pattern:**

```mojo
var shared_value = 0.0
if lane == 0:
    shared_value = compute_block_statistic()
shared_value = broadcast(shared_value)
result = use_shared_value(shared_value, local_data)
```

## Key concepts

### **Communication patterns**

Understanding fundamental warp communication paradigms:

- **Neighbor communication**: Lane-to-adjacent-lane data exchange
- **Collective coordination**: One-lane-to-all-lanes information sharing
- **Stencil operations**: Accessing fixed patterns of neighboring data
- **Boundary handling**: Managing communication at warp edges

### **Hardware optimization**

Recognizing how warp communication maps to GPU hardware:

- **Register file communication**: Direct inter-thread register access
- **SIMT execution**: All lanes execute communication simultaneously
- **Zero latency**: Communication happens within the execution unit
- **Automatic synchronization**: No explicit barriers needed

### **Algorithm transformation**

Converting traditional parallel patterns to warp communication:

- **Array neighbor access** `shuffle_down()`
- **Shared memory coordination** `broadcast()`
- **Complex boundary logic** Hardware-handled edge cases
- **Multi-stage synchronization** Single communication operations

## Getting started

Start with neighbor-based shuffle operations to understand the foundation, then progress to collective broadcast patterns for advanced coordination.

 **Success tip**: Think of warp communication as **hardware-accelerated message passing**between threads in the same warp. This mental model will guide you toward efficient communication patterns that leverage the GPU's SIMT architecture.

**Learning objective**: By the end of Puzzle 25, you'll recognize when warp communication can replace complex shared memory patterns, enabling you to write simpler, faster neighbor-based and coordination algorithms.

**Begin with**: **[Warp Shuffle Down Operations](#warpshuffledown-one-to-one-communication)**to learn neighbor communication, then advance to **[Warp Broadcast Operations](#warpbroadcast-one-to-many-communication)**for collective coordination patterns.

## `warp.shuffle_down()` One-to-One Communication

For warp-level neighbor communication we can use `shuffle_down()` to access data from adjacent lanes within a warp. This powerful primitive enables efficient finite differences, moving averages, and neighbor-based computations without shared memory or explicit synchronization.

**Key insight:**_The [shuffle_down()](https://docs.modular.com/mojo/stdlib/gpu/warp/shuffle_down) operation leverages SIMT execution to let each lane access data from its neighbors within the same warp, enabling efficient stencil patterns and sliding window operations._

> **What are stencil operations?**[Stencil](https://en.wikipedia.org/wiki/Iterative_Stencil_Loops) operations are computations where each output element depends on a fixed pattern of neighboring input elements. Common examples include finite differences (derivatives), convolutions, and moving averages. The "stencil" refers to the pattern of neighbor access - like a 3-point stencil that reads `[i-1, i, i+1]` or a 5-point stencil that reads `[i-2, i-1, i, i+1, i+2]`.

### Key concepts

In this puzzle, you'll learn:

- **Warp-level data shuffling**with `shuffle_down()`
- **Neighbor access patterns**for stencil computations
- **Boundary handling**at warp edges
- **Multi-offset shuffling**for extended neighbor access
- **Cross-warp coordination**in multi-block scenarios

The `shuffle_down` operation enables each lane to access data from lanes at higher indices:
\\[\\Large \text{shuffle\_down}(\text{value}, \text{offset}) = \text{value_from_lane}(\text{lane\_id} + \text{offset})\\]

This transforms complex neighbor access patterns into simple warp-level operations, enabling efficient stencil computations without explicit memory indexing.

### 1. Basic neighbor difference

#### Configuration

- Vector size: `SIZE = WARP_SIZE` (32 or 64 depending on GPU)
- Grid configuration: `(1, 1)` blocks per grid
- Block configuration: `(WARP_SIZE, 1)` threads per block
- Data type: `DType.float32`
- Layout: `Layout.row_major(SIZE)` (1D row-major)

#### The shuffle_down concept

Traditional neighbor access requires complex indexing and bounds checking:

```mojo
# Traditional approach - complex and error-prone
if global_i < size - 1:
    next_value = input[global_i + 1]  # Potential out-of-bounds
    result = next_value - current_value
```

**Problems with traditional approach:**

- **Bounds checking**: Must manually verify array bounds
- **Memory access**: Requires separate memory loads
- **Synchronization**: May need barriers for shared memory patterns
- **Complex logic**: Handling edge cases becomes verbose

With `shuffle_down()`, neighbor access becomes elegant:

```mojo
# Warp shuffle approach - simple and safe
current_val = input[global_i]
next_val = shuffle_down(current_val, 1)  # Get value from lane+1
if lane < WARP_SIZE - 1:
    result = next_val - current_val
```

**Benefits of shuffle_down:**

- **Zero memory overhead**: No additional memory accesses
- **Automatic bounds**: Hardware handles warp boundaries
- **No synchronization**: SIMT execution guarantees correctness
- **Composable**: Easy to combine with other warp operations

#### Tips

#### 1. **Understanding shuffle_down**

The `shuffle_down(value, offset)` operation allows each lane to receive data from a lane at a higher index. Study how this can give you access to neighboring elements without explicit memory loads.

**What `shuffle_down(val, 1)` does:**

- Lane 0 gets value from Lane 1
- Lane 1 gets value from Lane 2
- ...
- Lane 30 gets value from Lane 31
- Lane 31 gets undefined value (handled by boundary check)

#### 2. **Warp boundary considerations**

Consider what happens at the edges of a warp. Some lanes may not have valid neighbors to access via shuffle operations.

**Design prompt:**Design your algorithm to handle cases where shuffle operations may return undefined data for lanes at warp boundaries.

For neighbor difference with `WARP_SIZE = 32`:

- **Valid difference**(`lane < WARP_SIZE - 1`): **Lanes 0-30**(31 lanes)
  - **When**: \\(\text{lane\_id}() \in \{0, 1, \cdots, 30\}\\)
  - **Why**: `shuffle_down(current_val, 1)` successfully gets next neighbor's value
  - **Result**: `output[i] = input[i+1] - input[i]` (finite difference)

- **Boundary case**(else): **Lane 31**(1 lane)
  - **When**: \\(\text{lane\_id}() = 31\\)
  - **Why**: `shuffle_down(current_val, 1)` returns undefined data (no lane 32)
  - **Result**: `output[i] = 0` (cannot compute difference)

#### 3. **Lane identification**

```mojo
lane = lane_id()  # Returns 0 to WARP_SIZE-1
```

**Lane numbering:**Within each warp, lanes are numbered 0, 1, 2, ..., `WARP_SIZE-1`

**Test the neighbor difference:**

  
    pixi NVIDIA (default)
    pixi AMD
    pixi Apple
    uv
  
  

```bash
pixi run p25 --neighbor
```

  
  

```bash
pixi run -e amd p25 --neighbor
```

  
  

```bash
pixi run -e apple p25 --neighbor
```

  
  

```bash
uv run poe p25 --neighbor
```

  

Expected output when solved:

```txt
WARP_SIZE:  32
SIZE:  32
output: [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0, 31.0, 33.0, 35.0, 37.0, 39.0, 41.0, 43.0, 45.0, 47.0, 49.0, 51.0, 53.0, 55.0, 57.0, 59.0, 61.0, 0.0]
expected: [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0, 31.0, 33.0, 35.0, 37.0, 39.0, 41.0, 43.0, 45.0, 47.0, 49.0, 51.0, 53.0, 55.0, 57.0, 59.0, 61.0, 0.0]
OK Basic neighbor difference test passed!
```

#### Reference implementation (example)


```mojo
fn neighbor_difference
    layout: Layout, size: Int
:
    """
    Compute finite differences: output[i] = input[i+1] - input[i]
    Uses shuffle_down(val, 1) to get the next neighbor's value.
    Works across multiple blocks, each processing one warp worth of data.
    """
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    lane = Int(lane_id())

    if global_i < size:
        # Get current value
        current_val = input[global_i]

        # Get next neighbor's value using shuffle_down
        next_val = shuffle_down(current_val, 1)

        # Compute difference - valid within warp boundaries
        # Last lane of each warp has no valid neighbor within the warp
        # Note there's only one warp in this test, so we don't need to check global_i < size - 1
        # We'll see how this works with multiple blocks in the next tests
        if lane < WARP_SIZE - 1:
            output[global_i] = next_val - current_val
        else:
            # Last thread in warp or last thread overall, set to 0
            output[global_i] = 0

```

This solution demonstrates how `shuffle_down()` transforms traditional array indexing into efficient warp-level communication.

**Algorithm breakdown:**

```mojo
if global_i < size:
    current_val = input[global_i]           # Each lane reads its element
    next_val = shuffle_down(current_val, 1) # Hardware shifts data right

    if lane < WARP_SIZE - 1:
        output[global_i] = next_val - current_val  # Compute difference
    else:
        output[global_i] = 0                       # Boundary handling
```

**SIMT execution deep dive:**

```
Cycle 1: All lanes load their values simultaneously
  Lane 0: current_val = input[0] = 0
  Lane 1: current_val = input[1] = 1
  Lane 2: current_val = input[2] = 4
  ...
  Lane 31: current_val = input[31] = 961

Cycle 2: shuffle_down(current_val, 1) executes on all lanes
  Lane 0: receives current_val from Lane 1 -> next_val = 1
  Lane 1: receives current_val from Lane 2 -> next_val = 4
  Lane 2: receives current_val from Lane 3 -> next_val = 9
  ...
  Lane 30: receives current_val from Lane 31 -> next_val = 961
  Lane 31: receives undefined (no Lane 32) -> next_val = ?

Cycle 3: Difference computation (lanes 0-30 only)
  Lane 0: output[0] = 1 - 0 = 1
  Lane 1: output[1] = 4 - 1 = 3
  Lane 2: output[2] = 9 - 4 = 5
  ...
  Lane 31: output[31] = 0 (boundary condition)
```

**Mathematical insight:**This implements the discrete derivative operator \\(D\\):
\\\\Large D[f = f(i+1) - f(i)\\]

For our quadratic input \\(f(i) = i^2\\):
\\[\\Large D[i^2] = (i+1)^2 - i^2 = i^2 + 2i + 1 - i^2 = 2i + 1\\]

**Why shuffle_down is superior:**

1. **Memory efficiency**: Traditional approach requires `input[global_i + 1]` load, potentially causing cache misses
2. **Bounds safety**: No risk of out-of-bounds access; hardware handles warp boundaries
3. **SIMT optimization**: Single instruction processes all lanes simultaneously
4. **Register communication**: Data moves between registers, not through memory hierarchy

**Performance characteristics:**

- **Latency**: 1 cycle (vs 100+ cycles for memory access)
- **Bandwidth**: 0 bytes (vs 4 bytes per thread for traditional)
- **Parallelism**: All 32 lanes process simultaneously

### 2. Multi-offset moving average

#### Configuration

- Vector size: `SIZE_2 = 64` (multi-block scenario)
- Grid configuration: `BLOCKS_PER_GRID = (2, 1)` blocks per grid
- Block configuration: `THREADS_PER_BLOCK = (WARP_SIZE, 1)` threads per block

#### Tips

#### 1. **Multi-offset shuffle patterns**

This puzzle requires accessing multiple neighbors simultaneously. You'll need to use shuffle operations with different offsets.

**Key questions:**

- How can you get both `input[i+1]` and `input[i+2]` using shuffle operations?
- What's the relationship between shuffle offset and neighbor distance?
- Can you perform multiple shuffles on the same source value?

**Visualization concept:**

```
Your lane needs:  current_val, next_val, next_next_val
Shuffle offsets:  0 (direct),  1,        2
```

**Think about:**How many shuffle operations do you need, and what offsets should you use?

#### 2. **Tiered boundary handling**

Unlike the simple neighbor difference, this puzzle has multiple boundary scenarios because you need access to 2 neighbors.

**Boundary scenarios to consider:**

- **Full window:**Lane can access both neighbors  use all 3 values
- **Partial window:**Lane can access 1 neighbor  use 2 values
- **No window:**Lane can't access any neighbors  use 1 value

**Critical thinking:**

- Which lanes fall into each category?
- How should you weight the averages when you have fewer values?
- What boundary conditions should you check?

**Pattern to consider:**

```
if (can_access_both_neighbors):
    # 3-point average
elif (can_access_one_neighbor):
    # 2-point average
else:
    # 1-point (no averaging)
```

#### 3. **Multi-block coordination**

This puzzle uses multiple blocks, each processing a different section of the data.

**Important considerations:**

- Each block has its own warp with lanes 0 to WARP_SIZE-1
- Boundary conditions apply within each warp independently
- Lane numbering resets for each block

**Questions to think about:**

- Does your boundary logic work correctly for both Block 0 and Block 1?
- Are you checking both lane boundaries AND global array boundaries?
- How does `global_i` relate to `lane_id()` in different blocks?

**Debugging tip:**Test your logic by tracing through what happens at the boundary lanes of each block.

**Test the moving average:**

  
    pixi NVIDIA (default)
    pixi AMD
    uv
  
  

```bash
pixi run p25 --average
```

  
  

```bash
pixi run -e amd p25 --average
```

  
  

```bash
uv run poe p25 --average
```

  

Expected output when solved:

```txt
WARP_SIZE:  32
SIZE_2:  64
output: HostBuffer([3.3333333, 6.3333335, 10.333333, 15.333333, 21.333334, 28.333334, 36.333332, 45.333332, 55.333332, 66.333336, 78.333336, 91.333336, 105.333336, 120.333336, 136.33333, 153.33333, 171.33333, 190.33333, 210.33333, 231.33333, 253.33333, 276.33334, 300.33334, 325.33334, 351.33334, 378.33334, 406.33334, 435.33334, 465.33334, 496.33334, 512.0, 528.0, 595.3333, 630.3333, 666.3333, 703.3333, 741.3333, 780.3333, 820.3333, 861.3333, 903.3333, 946.3333, 990.3333, 1035.3334, 1081.3334, 1128.3334, 1176.3334, 1225.3334, 1275.3334, 1326.3334, 1378.3334, 1431.3334, 1485.3334, 1540.3334, 1596.3334, 1653.3334, 1711.3334, 1770.3334, 1830.3334, 1891.3334, 1953.3334, 2016.3334, 2048.0, 2080.0])
expected: HostBuffer([3.3333333, 6.3333335, 10.333333, 15.333333, 21.333334, 28.333334, 36.333332, 45.333332, 55.333332, 66.333336, 78.333336, 91.333336, 105.333336, 120.333336, 136.33333, 153.33333, 171.33333, 190.33333, 210.33333, 231.33333, 253.33333, 276.33334, 300.33334, 325.33334, 351.33334, 378.33334, 406.33334, 435.33334, 465.33334, 496.33334, 512.0, 528.0, 595.3333, 630.3333, 666.3333, 703.3333, 741.3333, 780.3333, 820.3333, 861.3333, 903.3333, 946.3333, 990.3333, 1035.3334, 1081.3334, 1128.3334, 1176.3334, 1225.3334, 1275.3334, 1326.3334, 1378.3334, 1431.3334, 1485.3334, 1540.3334, 1596.3334, 1653.3334, 1711.3334, 1770.3334, 1830.3334, 1891.3334, 1953.3334, 2016.3334, 2048.0, 2080.0])
OK Moving average test passed!
```

#### Reference implementation (example)


```mojo
fn moving_average_3
    layout: Layout, size: Int
:
    """
    Compute 3-point moving average: output[i] = (input[i] + input[i+1] + input[i+2]) / 3
    Uses shuffle_down with offsets 1 and 2 to access neighbors.
    Works within warp boundaries across multiple blocks.
    """
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    lane = Int(lane_id())

    if global_i < size:
        # Get current, next, and next+1 values
        current_val = input[global_i]
        next_val = shuffle_down(current_val, 1)
        next_next_val = shuffle_down(current_val, 2)

        # Compute 3-point average - valid within warp boundaries
        if lane < WARP_SIZE - 2 and global_i < size - 2:
            output[global_i] = (current_val + next_val + next_next_val) / 3.0
        elif lane < WARP_SIZE - 1 and global_i < size - 1:
            # Second-to-last in warp: only current + next available
            output[global_i] = (current_val + next_val) / 2.0
        else:
            # Last thread in warp or boundary cases: only current available
            output[global_i] = current_val

```

This solution demonstrates advanced multi-offset shuffling for complex stencil operations.

**Complete algorithm analysis:**

```mojo
if global_i < size:
    # Step 1: Acquire all needed data via multiple shuffles
    current_val = input[global_i]                   # Direct access
    next_val = shuffle_down(current_val, 1)         # Right neighbor
    next_next_val = shuffle_down(current_val, 2)    # Right+1 neighbor

    # Step 2: Adaptive computation based on available data
    if lane < WARP_SIZE - 2 and global_i < size - 2:
        # Full 3-point stencil available
        output[global_i] = (current_val + next_val + next_next_val) / 3.0
    elif lane < WARP_SIZE - 1 and global_i < size - 1:
        # Only 2-point stencil available (near warp boundary)
        output[global_i] = (current_val + next_val) / 2.0
    else:
        # No stencil possible (at warp boundary)
        output[global_i] = current_val
```

**Multi-offset execution trace (`WARP_SIZE = 32`):**

```
Initial state (Block 0, elements 0-31):
  Lane 0: current_val = input[0] = 1
  Lane 1: current_val = input[1] = 2
  Lane 2: current_val = input[2] = 4
  ...
  Lane 31: current_val = input[31] = X

First shuffle: shuffle_down(current_val, 1)
  Lane 0: next_val = input[1] = 2
  Lane 1: next_val = input[2] = 4
  Lane 2: next_val = input[3] = 7
  ...
  Lane 30: next_val = input[31] = X
  Lane 31: next_val = undefined

Second shuffle: shuffle_down(current_val, 2)
  Lane 0: next_next_val = input[2] = 4
  Lane 1: next_next_val = input[3] = 7
  Lane 2: next_next_val = input[4] = 11
  ...
  Lane 29: next_next_val = input[31] = X
  Lane 30: next_next_val = undefined
  Lane 31: next_next_val = undefined

Computation phase:
  Lanes 0-29: Full 3-point average -> (current + next + next_next) / 3
  Lane 30:    2-point average -> (current + next) / 2
  Lane 31:    1-point average -> current (passthrough)
```

**Mathematical foundation:**This implements a variable-width discrete convolution:
\\[\\Large h[i] = \\sum_{k=0}^{K(i)-1} w_k^{(i)} \\cdot f[i+k]\\]

Where the kernel adapts based on position:

- **Interior points**: \\(K(i) = 3\\), \\(\\mathbf{w}^{(i)} = [\\frac{1}{3}, \\frac{1}{3}, \\frac{1}{3}]\\)
- **Near boundary**: \\(K(i) = 2\\), \\(\\mathbf{w}^{(i)} = [\\frac{1}{2}, \\frac{1}{2}]\\)
- **At boundary**: \\(K(i) = 1\\), \\(\\mathbf{w}^{(i)} = [1]\\)

**Multi-block coordination:**With `SIZE_2 = 64` and 2 blocks:

```
Block 0 (global indices 0-31):
  Lane boundaries apply to global indices 29, 30, 31

Block 1 (global indices 32-63):
  Lane boundaries apply to global indices 61, 62, 63
  Lane numbers reset: global_i=32 -> lane=0, global_i=63 -> lane=31
```

**Performance optimizations:**

1. **Parallel data acquisition**: Both shuffle operations execute simultaneously
2. **Conditional branching**: GPU handles divergent lanes efficiently via predication
3. **Memory coalescing**: Sequential global memory access pattern optimal for GPU
4. **Register reuse**: All intermediate values stay in registers

**Signal processing perspective:**This is a causal FIR filter with impulse response \\(h[n] = \\frac{1}{3}[\\delta[n] + \\delta[n-1] + \\delta[n-2]]\\), providing smoothing with a cutoff frequency at \\(f_c \\approx 0.25f_s\\).

### Summary

Here is what the core pattern of this section looks like

```mojo
current_val = input[global_i]
neighbor_val = shuffle_down(current_val, offset)
if lane < WARP_SIZE - offset:
    result = compute(current_val, neighbor_val)
```

**Key benefits:**

- **Hardware efficiency**: Register-to-register communication
- **Boundary safety**: Automatic warp limit handling
- **SIMT optimization**: Single instruction, all lanes parallel

**Applications**: Finite differences, stencil operations, moving averages, convolutions.

## `warp.broadcast()` One-to-Many Communication

For warp-level coordination we can use `broadcast()` to share data from one lane to all other lanes within a warp. This powerful primitive enables efficient block-level computations, conditional logic coordination, and one-to-many communication patterns without shared memory or explicit synchronization.

**Key insight:**_The [broadcast()](https://docs.modular.com/mojo/stdlib/gpu/warp/broadcast) operation leverages SIMT execution to let one lane (typically lane 0) share its computed value with all other lanes in the same warp, enabling efficient coordination patterns and collective decision-making._

> **What are broadcast operations?**Broadcast operations are communication patterns where one thread computes a value and shares it with all other threads in a group. This is essential for coordination tasks like computing block-level statistics, making collective decisions, or sharing configuration parameters across all threads in a warp.

### Key concepts

In this puzzle, you'll learn:

- **Warp-level broadcasting**with `broadcast()`
- **One-to-many communication**patterns
- **Collective computation**strategies
- **Conditional coordination**across lanes
- **Combined broadcast-shuffle**operations

The `broadcast` operation enables one lane (by default lane 0) to share its value with all other lanes:
\\[\\Large \text{broadcast}(\text{value}) = \text{value_from_lane_0_to_all_lanes}\\]

This transforms complex coordination patterns into simple warp-level operations, enabling efficient collective computations without explicit synchronization.

### The broadcast concept

Traditional coordination requires complex shared memory patterns:

```mojo
# Traditional approach - complex and error-prone
shared_memory[lane] = local_computation()
sync_threads()  # Expensive synchronization
if lane == 0:
    result = compute_from_shared_memory()
sync_threads()  # Another expensive synchronization
final_result = shared_memory[0]  # All threads read
```

**Problems with traditional approach:**

- **Memory overhead**: Requires shared memory allocation
- **Synchronization**: Multiple expensive barrier operations
- **Complex logic**: Managing shared memory indices and access patterns
- **Error-prone**: Easy to introduce race conditions

With `broadcast()`, coordination becomes elegant:

```mojo
# Warp broadcast approach - simple and safe
collective_value = 0
if lane == 0:
    collective_value = compute_block_statistic()
collective_value = broadcast(collective_value)  # Share with all lanes
result = use_collective_value(collective_value)
```

**Benefits of broadcast:**

- **Zero memory overhead**: No shared memory required
- **Automatic synchronization**: SIMT execution guarantees correctness
- **Simple pattern**: One lane computes, all lanes receive
- **Composable**: Easy to combine with other warp operations

### 1. Basic broadcast

Implement a basic broadcast pattern where lane 0 computes a block-level statistic and shares it with all lanes.

**Requirements:**

- Lane 0 should compute the sum of the first 4 elements in the current block
- This computed value must be shared with all other lanes in the warp using `broadcast()`
- Each lane should then add this shared value to its own input element

**Test data:**Input `[1, 2, 3, 4, 5, 6, 7, 8, ...]` should produce output `[11, 12, 13, 14, 15, 16, 17, 18, ...]`

**Design prompt:**How do you coordinate so that only one lane does the block-level computation, but all lanes can use the result in their individual operations?

#### Configuration

- Vector size: `SIZE = WARP_SIZE` (32 or 64 depending on GPU)
- Grid configuration: `(1, 1)` blocks per grid
- Block configuration: `(WARP_SIZE, 1)` threads per block
- Data type: `DType.float32`
- Layout: `Layout.row_major(SIZE)` (1D row-major)

#### Tips

#### 1. **Understanding broadcast mechanics**

The `broadcast(value)` operation takes the value from lane 0 and distributes it to all lanes in the warp.

**Key insight:**Only lane 0's value matters for the broadcast. Other lanes' values are ignored, but all lanes receive lane 0's value.

**Visualization:**

```
Before broadcast: Lane 0 has \(\text{val}_0\), Lane 1 has \(\text{val}_1\), Lane 2 has \(\text{val}_2\), ...
After broadcast:  Lane 0 has \(\text{val}_0\), Lane 1 has \(\text{val}_0\), Lane 2 has \(\text{val}_0\), ...
```

**Think about:**How can you ensure only lane 0 computes the value you want to broadcast?

#### 2. **Lane-specific computation**

Design your algorithm so that lane 0 performs the special computation while other lanes wait.

**Pattern to consider:**

```
var shared_value = initial_value
if lane == 0:
    # Only lane 0 computes
    shared_value = special_computation()
# All lanes participate in broadcast
shared_value = broadcast(shared_value)
```

**Critical questions:**

- What should other lanes' values be before the broadcast?
- How do you ensure lane 0 has the correct value to broadcast?

#### 3. **Collective usage**

After broadcasting, all lanes have the same value and can use it in their individual computations.

**Think about:**How does each lane combine the broadcast value with its own local data?

**Test the basic broadcast:**

  
    pixi NVIDIA (default)
    pixi AMD
    pixi Apple
    uv
  
  

```bash
pixi run p25 --broadcast-basic
```

  
  

```bash
pixi run -e amd p25 --broadcast-basic
```

  
  

```bash
pixi run -e apple p25 --broadcast-basic
```

  
  

```bash
uv run poe p25 --broadcast-basic
```

  

Expected output when solved:

```txt
WARP_SIZE:  32
SIZE:  32
output: HostBuffer([11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0])
expected: HostBuffer([11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0])
OK Basic broadcast test passed!
```

#### Reference implementation (example)


```mojo
fn basic_broadcast
    layout: Layout, size: Int
:
    """
    Basic broadcast: Lane 0 computes a block-local value, broadcasts it to all lanes.
    Each lane then uses this broadcast value in its own computation.
    """
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    lane = Int(lane_id())

    if global_i < size:
        # Step 1: Lane 0 computes special value (sum of first 4 elements in this block)
        var broadcast_value: output.element_type = 0.0
        if lane == 0:
            block_start = Int(block_idx.x * block_dim.x)
            var sum: output.element_type = 0.0
            for i in range(4):
                if block_start + i < size:
                    sum += input[block_start + i]
            broadcast_value = sum

        # Step 2: Broadcast lane 0's value to all lanes in this warp
        broadcast_value = broadcast(broadcast_value)

        # Step 3: All lanes use broadcast value in their computation
        output[global_i] = broadcast_value + input[global_i]

```

This solution demonstrates the fundamental broadcast pattern for warp-level coordination.

**Algorithm breakdown:**

```mojo
if global_i < size:
    # Step 1: Lane 0 computes special value
    var broadcast_value: output.element_type = 0.0
    if lane == 0:
        # Only lane 0 performs this computation
        block_start = block_idx.x * block_dim.x
        var sum: output.element_type = 0.0
        for i in range(4):
            if block_start + i < size:
                sum += input[block_start + i]
        broadcast_value = sum

    # Step 2: Share lane 0's value with all lanes
    broadcast_value = broadcast(broadcast_value)

    # Step 3: All lanes use the broadcast value
    output[global_i] = broadcast_value + input[global_i]
```

**SIMT execution trace:**

```
Cycle 1: Lane-specific computation
  Lane 0: Computes sum of input[0] + input[1] + input[2] + input[3] = 1+2+3+4 = 10
  Lane 1: broadcast_value remains 0.0 (not lane 0)
  Lane 2: broadcast_value remains 0.0 (not lane 0)
  ...
  Lane 31: broadcast_value remains 0.0 (not lane 0)

Cycle 2: broadcast(broadcast_value) executes
  Lane 0: Keeps its value -> broadcast_value = 10.0
  Lane 1: Receives lane 0's value -> broadcast_value = 10.0
  Lane 2: Receives lane 0's value -> broadcast_value = 10.0
  ...
  Lane 31: Receives lane 0's value -> broadcast_value = 10.0

Cycle 3: Individual computation with broadcast value
  Lane 0: output[0] = 10.0 + input[0] = 10.0 + 1.0 = 11.0
  Lane 1: output[1] = 10.0 + input[1] = 10.0 + 2.0 = 12.0
  Lane 2: output[2] = 10.0 + input[2] = 10.0 + 3.0 = 13.0
  ...
  Lane 31: output[31] = 10.0 + input[31] = 10.0 + 32.0 = 42.0
```

**Why broadcast is superior:**

1. **Coordination efficiency**: Single operation coordinates all lanes
2. **Memory efficiency**: No shared memory allocation required
3. **Synchronization-free**: SIMT execution handles coordination automatically
4. **Scalable pattern**: Works identically regardless of warp size

**Performance characteristics:**

- **Latency**: 1 cycle for broadcast operation
- **Bandwidth**: 0 bytes (register-to-register communication)
- **Coordination**: All 32 lanes synchronized automatically

### 2. Conditional broadcast

Implement conditional coordination where lane 0 analyzes block data and makes a decision that affects all lanes.

**Requirements:**

- Lane 0 should analyze the first 8 elements in the current block and find their maximum value
- This maximum value must be broadcast to all other lanes using `broadcast()`
- Each lane should then apply conditional logic: if their element is above half the maximum, double it; otherwise, halve it

**Test data:**Input `[3, 1, 7, 2, 9, 4, 6, 8, ...]` (repeating pattern) should produce output `[1.5, 0.5, 14.0, 1.0, 18.0, 2.0, 12.0, 16.0, ...]`

**Design prompt:**How do you coordinate block-level analysis with element-wise conditional transformations across all lanes?

#### Configuration

- Vector size: `SIZE = WARP_SIZE` (32 or 64 depending on GPU)
- Grid configuration: `(1, 1)` blocks per grid
- Block configuration: `(WARP_SIZE, 1)` threads per block

#### Tips

#### 1. **Analysis and decision-making**

Lane 0 needs to analyze multiple data points and make a decision that will guide all other lanes.

**Key questions:**

- How can lane 0 efficiently analyze multiple elements?
- What kind of decision should be broadcast to coordinate lane behavior?
- How do you handle boundary conditions when analyzing data?

**Pattern to consider:**

```
var decision = default_value
if lane == 0:
    # Analyze block-local data
    decision = analyze_and_decide()
decision = broadcast(decision)
```

#### 2. **Conditional execution coordination**

After receiving the broadcast decision, all lanes need to apply different logic based on the decision.

**Think about:**

- How do lanes use the broadcast value to make local decisions?
- What operations should be applied in each conditional branch?
- How do you ensure consistent behavior across all lanes?

**Conditional pattern:**

```
if (local_data meets_broadcast_criteria):
    # Apply one transformation
else:
    # Apply different transformation
```

#### 3. **Data analysis strategies**

Consider efficient ways for lane 0 to analyze multiple data points.

**Approaches to consider:**

- Finding maximum/minimum values
- Computing averages or sums
- Detecting patterns or thresholds
- Making binary decisions based on data characteristics

**Test the conditional broadcast:**

  
    pixi NVIDIA (default)
    pixi AMD
    uv
  
  

```bash
pixi run p25 --broadcast-conditional
```

  
  

```bash
pixi run -e amd p25 --broadcast-conditional
```

  
  

```bash
uv run poe p25 --broadcast-conditional
```

  

Expected output when solved:

```txt
WARP_SIZE:  32
SIZE:  32
output: HostBuffer([1.5, 0.5, 14.0, 1.0, 18.0, 2.0, 12.0, 16.0, 1.5, 0.5, 14.0, 1.0, 18.0, 2.0, 12.0, 16.0, 1.5, 0.5, 14.0, 1.0, 18.0, 2.0, 12.0, 16.0, 1.5, 0.5, 14.0, 1.0, 18.0, 2.0, 12.0, 16.0])
expected: HostBuffer([1.5, 0.5, 14.0, 1.0, 18.0, 2.0, 12.0, 16.0, 1.5, 0.5, 14.0, 1.0, 18.0, 2.0, 12.0, 16.0, 1.5, 0.5, 14.0, 1.0, 18.0, 2.0, 12.0, 16.0, 1.5, 0.5, 14.0, 1.0, 18.0, 2.0, 12.0, 16.0])
OK Conditional broadcast test passed!
```

#### Reference implementation (example)


```mojo
fn conditional_broadcast
    layout: Layout, size: Int
:
    """
    Conditional broadcast: Lane 0 makes a decision based on block-local data, broadcasts it to all lanes.
    All lanes apply different logic based on the broadcast decision.
    """
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    lane = Int(lane_id())

    if global_i < size:
        # Step 1: Lane 0 analyzes block-local data and makes decision (find max of first 8 in block)
        var decision_value: output.element_type = 0.0
        if lane == 0:
            block_start = Int(block_idx.x * block_dim.x)
            decision_value = input[block_start] if block_start < size else 0.0
            for i in range(1, min(8, min(WARP_SIZE, size - block_start))):
                if block_start + i < size:
                    current_val = input[block_start + i]
                    if current_val > decision_value:
                        decision_value = current_val

        # Step 2: Broadcast decision to all lanes in this warp
        decision_value = broadcast(decision_value)

        # Step 3: All lanes apply conditional logic based on broadcast decision
        current_input = input[global_i]
        threshold = decision_value / 2.0
        if current_input >= threshold:
            output[global_i] = current_input * 2.0  # Double if >= threshold
        else:
            output[global_i] = current_input / 2.0  # Halve if < threshold

```

This solution demonstrates advanced broadcast patterns for conditional coordination across lanes.

**Complete algorithm analysis:**

```mojo
if global_i < size:
    # Step 1: Lane 0 analyzes block data and makes decision
    var decision_value: output.element_type = 0.0
    if lane == 0:
        # Find maximum among first 8 elements in block
        block_start = block_idx.x * block_dim.x
        decision_value = input[block_start] if block_start < size else 0.0
        for i in range(1, min(8, min(WARP_SIZE, size - block_start))):
            if block_start + i < size:
                current_val = input[block_start + i]
                if current_val > decision_value:
                    decision_value = current_val

    # Step 2: Broadcast decision to coordinate all lanes
    decision_value = broadcast(decision_value)

    # Step 3: All lanes apply conditional logic based on broadcast
    current_input = input[global_i]
    threshold = decision_value / 2.0
    if current_input >= threshold:
        output[global_i] = current_input * 2.0  # Double if >= threshold
    else:
        output[global_i] = current_input / 2.0  # Halve if < threshold
```

**Decision-making execution trace:**

```
Input data: [3.0, 1.0, 7.0, 2.0, 9.0, 4.0, 6.0, 8.0, ...]

Step 1: Lane 0 finds maximum of first 8 elements
  Lane 0 analysis:
    Start with input[0] = 3.0
    Compare with input[1] = 1.0 -> keep 3.0
    Compare with input[2] = 7.0 -> update to 7.0
    Compare with input[3] = 2.0 -> keep 7.0
    Compare with input[4] = 9.0 -> update to 9.0
    Compare with input[5] = 4.0 -> keep 9.0
    Compare with input[6] = 6.0 -> keep 9.0
    Compare with input[7] = 8.0 -> keep 9.0
    Final decision_value = 9.0

Step 2: Broadcast decision_value = 9.0 to all lanes
  All lanes now have: decision_value = 9.0, threshold = 4.5

Step 3: Conditional execution per lane
  Lane 0: input[0] = 3.0 < 4.5 -> output[0] = 3.0 / 2.0 = 1.5
  Lane 1: input[1] = 1.0 < 4.5 -> output[1] = 1.0 / 2.0 = 0.5
  Lane 2: input[2] = 7.0 >= 4.5 -> output[2] = 7.0 * 2.0 = 14.0
  Lane 3: input[3] = 2.0 < 4.5 -> output[3] = 2.0 / 2.0 = 1.0
  Lane 4: input[4] = 9.0 >= 4.5 -> output[4] = 9.0 * 2.0 = 18.0
  Lane 5: input[5] = 4.0 < 4.5 -> output[5] = 4.0 / 2.0 = 2.0
  Lane 6: input[6] = 6.0 >= 4.5 -> output[6] = 6.0 * 2.0 = 12.0
  Lane 7: input[7] = 8.0 >= 4.5 -> output[7] = 8.0 * 2.0 = 16.0
  ...pattern repeats for remaining lanes
```

**Mathematical foundation:**This implements a threshold-based transformation:
\\[\\Large f(x) = \\begin{cases}
2x & \\text{if } x \\geq \\tau \\\\
\\frac{x}{2} & \\text{if } x < \\tau
\\end{cases}\\]

Where \\(\\tau = \\frac{\\max(\\text{block\_data})}{2}\\) is the broadcast threshold.

**Coordination pattern benefits:**

1. **Centralized analysis**: One lane analyzes, all lanes benefit
2. **Consistent decisions**: All lanes use the same threshold
3. **Adaptive behavior**: Threshold adapts to block-local data characteristics
4. **Efficient coordination**: Single broadcast coordinates complex conditional logic

**Applications:**

- **Adaptive algorithms**: Adjusting parameters based on local data characteristics
- **Quality control**: Applying different processing based on data quality metrics
- **Load balancing**: Distributing work based on block-local complexity analysis

### 3. Broadcast-shuffle coordination

Implement advanced coordination combining both `broadcast()` and `shuffle_down()` operations.

**Requirements:**

- Lane 0 should compute the average of the first 4 elements in the block and broadcast this scaling factor to all lanes
- Each lane should use `shuffle_down(offset=1)` to get their next neighbor's value
- For most lanes: multiply the scaling factor by `(current_value + next_neighbor_value)`
- For the last lane in the warp: multiply the scaling factor by just `current_value` (no valid neighbor)

**Test data:**Input follows pattern `[2, 4, 6, 8, 1, 3, 5, 7, ...]` (first 4 elements: 2,4,6,8 then repeating 1,3,5,7)

- Lane 0 computes scaling factor: `(2+4+6+8)/4 = 5.0`
- Expected output: `[30.0, 50.0, 70.0, 45.0, 20.0, 40.0, 60.0, 40.0, ...]`

**Design prompt:**How do you coordinate multiple warp primitives so that one lane's computation affects all lanes, while each lane also accesses its neighbor's data?

#### Configuration

- Vector size: `SIZE = WARP_SIZE` (32 or 64 depending on GPU)
- Grid configuration: `(1, 1)` blocks per grid
- Block configuration: `(WARP_SIZE, 1)` threads per block

#### Tips

#### 1. **Multi-primitive coordination**

This puzzle requires orchestrating both broadcast and shuffle operations in sequence.

**Think about the flow:**

1. One lane computes a value for the entire warp
2. This value is broadcast to all lanes
3. Each lane uses shuffle to access neighbor data
4. The broadcast value influences how neighbor data is processed

**Coordination pattern:**

```
# Phase 1: Broadcast coordination
var shared_param = compute_if_lane_0()
shared_param = broadcast(shared_param)

# Phase 2: Shuffle neighbor access
current_val = input[global_i]
neighbor_val = shuffle_down(current_val, offset)

# Phase 3: Combined computation
result = combine(current_val, neighbor_val, shared_param)
```

#### 2. **Parameter computation strategy**

Consider what kind of block-level parameter would be useful for scaling neighbor operations.

**Questions to explore:**

- What statistic should lane 0 compute from the block data?
- How should this parameter influence the neighbor-based computation?
- What happens at warp boundaries when shuffle operations are involved?

#### 3. **Combined operation design**

Think about how to meaningfully combine broadcast parameters with shuffle-based neighbor access.

**Pattern considerations:**

- Should the broadcast parameter scale the inputs, outputs, or computation?
- How do you handle boundary cases where shuffle returns undefined data?
- What's the most efficient order of operations?

**Test the broadcast-shuffle coordination:**

  
    pixi NVIDIA (default)
    pixi AMD
    uv
  
  

```bash
pixi run p25 --broadcast-shuffle-coordination
```

  
  

```bash
pixi run -e amd p25 --broadcast-shuffle-coordination
```

  
  

```bash
uv run poe p25 --broadcast-shuffle-coordination
```

  

Expected output when solved:

```txt
WARP_SIZE:  32
SIZE:  32
output: HostBuffer([30.0, 50.0, 70.0, 45.0, 20.0, 40.0, 60.0, 40.0, 20.0, 40.0, 60.0, 40.0, 20.0, 40.0, 60.0, 40.0, 20.0, 40.0, 60.0, 40.0, 20.0, 40.0, 60.0, 40.0, 20.0, 40.0, 60.0, 40.0, 20.0, 40.0, 60.0, 35.0])
expected: HostBuffer([30.0, 50.0, 70.0, 45.0, 20.0, 40.0, 60.0, 40.0, 20.0, 40.0, 60.0, 40.0, 20.0, 40.0, 60.0, 40.0, 20.0, 40.0, 60.0, 40.0, 20.0, 40.0, 60.0, 40.0, 20.0, 40.0, 60.0, 40.0, 20.0, 40.0, 60.0, 35.0])
OK Broadcast + Shuffle coordination test passed!
```

#### Reference implementation (example)


```mojo
fn broadcast_shuffle_coordination
    layout: Layout, size: Int
:
    """
    Combine broadcast() and shuffle_down() for advanced warp coordination.
    Lane 0 computes block-local scaling factor, broadcasts it to all lanes in the warp.
    Each lane uses shuffle_down() for neighbor access and applies broadcast factor.
    """
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    lane = Int(lane_id())

    if global_i < size:
        # Step 1: Lane 0 computes block-local scaling factor
        var scale_factor: output.element_type = 0.0
        if lane == 0:
            # Compute average of first 4 elements in this block's data
            block_start = Int(block_idx.x * block_dim.x)
            var sum: output.element_type = 0.0
            for i in range(4):
                if block_start + i < size:
                    sum += input[block_start + i]
            scale_factor = sum / 4.0

        # Step 2: Broadcast scaling factor to all lanes in this warp
        scale_factor = broadcast(scale_factor)

        # Step 3: Each lane gets current and next values
        current_val = input[global_i]
        next_val = shuffle_down(current_val, 1)

        # Step 4: Apply broadcast factor with neighbor coordination
        if lane < WARP_SIZE - 1 and global_i < size - 1:
            # Combine current + next, then scale by broadcast factor
            output[global_i] = (current_val + next_val) * scale_factor
        else:
            # Last lane in warp or last element: only current value, scaled by broadcast factor
            output[global_i] = current_val * scale_factor

```

This solution demonstrates the most advanced warp coordination pattern, combining broadcast and shuffle primitives.

**Complete algorithm analysis:**

```mojo
if global_i < size:
    # Step 1: Lane 0 computes block-local scaling factor
    var scale_factor: output.element_type = 0.0
    if lane == 0:
        block_start = block_idx.x * block_dim.x
        var sum: output.element_type = 0.0
        for i in range(4):
            if block_start + i < size:
                sum += input[block_start + i]
        scale_factor = sum / 4.0

    # Step 2: Broadcast scaling factor to all lanes
    scale_factor = broadcast(scale_factor)

    # Step 3: Each lane gets current and next values via shuffle
    current_val = input[global_i]
    next_val = shuffle_down(current_val, 1)

    # Step 4: Apply broadcast factor with neighbor coordination
    if lane < WARP_SIZE - 1 and global_i < size - 1:
        output[global_i] = (current_val + next_val) * scale_factor
    else:
        output[global_i] = current_val * scale_factor
```

**Multi-primitive execution trace:**

```
Input data: [2, 4, 6, 8, 1, 3, 5, 7, ...]

Phase 1: Lane 0 computes scaling factor
  Lane 0 computes: (input[0] + input[1] + input[2] + input[3]) / 4
                 = (2 + 4 + 6 + 8) / 4 = 20 / 4 = 5.0
  Other lanes: scale_factor remains 0.0

Phase 2: Broadcast scale_factor = 5.0 to all lanes
  All lanes now have: scale_factor = 5.0

Phase 3: Shuffle operations for neighbor access
  Lane 0: current_val = input[0] = 2, next_val = shuffle_down(2, 1) = input[1] = 4
  Lane 1: current_val = input[1] = 4, next_val = shuffle_down(4, 1) = input[2] = 6
  Lane 2: current_val = input[2] = 6, next_val = shuffle_down(6, 1) = input[3] = 8
  Lane 3: current_val = input[3] = 8, next_val = shuffle_down(8, 1) = input[4] = 1
  ...
  Lane 31: current_val = input[31], next_val = undefined

Phase 4: Combined computation with broadcast scaling
  Lane 0: output[0] = (2 + 4) * 5.0 = 6 * 5.0 = 30.0
  Lane 1: output[1] = (4 + 6) * 5.0 = 10 * 5.0 = 50.0... wait, expected is 30.0

  Let me recalculate based on the expected pattern:
  Expected: [30.0, 30.0, 35.0, 45.0, 30.0, 40.0, 35.0, 40.0, ...]

  Lane 0: (2 + 4) * 5 = 30 OK
  Lane 1: (4 + 6) * 5 = 50, but expected 30...

  Hmm, let me check if the input pattern is different or if there's an error in my understanding.
```

**Communication pattern analysis:**
This algorithm implements a **hierarchical coordination pattern**:

1. **Vertical coordination**(broadcast): Lane 0  All lanes
2. **Horizontal coordination**(shuffle): Lane i  Lane i+1
3. **Combined computation**: Uses both broadcast and shuffle data

**Mathematical foundation:**
\\[\\Large \\text{output}[i] = \\begin{cases}
(\\text{input}[i] + \\text{input}[i+1]) \\cdot \\beta & \\text{if lane } i < \\text{WARP\_SIZE} - 1 \\\\
\\text{input}[i] \\cdot \\beta & \\text{if lane } i = \\text{WARP\_SIZE} - 1
\\end{cases}\\]

Where \\(\\beta = \\frac{1}{4}\\sum_{k=0}^{3} \\text{input}[\\text{block\_start} + k]\\) is the broadcast scaling factor.

**Advanced coordination benefits:**

1. **Multi-level communication**: Combines global (broadcast) and local (shuffle) coordination
2. **Adaptive scaling**: Block-level parameters influence neighbor operations
3. **Efficient composition**: Two primitives work together seamlessly
4. **Complex algorithms**: Enables sophisticated parallel algorithms

**Real-world applications:**

- **Adaptive filtering**: Block-level noise estimation with neighbor-based filtering
- **Dynamic load balancing**: Global work distribution with local coordination
- **Multi-scale processing**: Global parameters controlling local stencil operations

### Summary

Here is what the core pattern of this section looks like

```mojo
var shared_value = initial_value
if lane == 0:
    shared_value = compute_block_statistic()
shared_value = broadcast(shared_value)
result = use_shared_value(shared_value, local_data)
```

**Key benefits:**

- **One-to-many coordination**: Single lane computes, all lanes benefit
- **Zero synchronization overhead**: SIMT execution handles coordination
- **Composable patterns**: Easily combines with shuffle and other warp operations

**Applications**: Block statistics, collective decisions, parameter sharing, adaptive algorithms.
