---
title: "Prefix Sum"
description: "Prefix sum (also known as _scan_) is a fundamental parallel algorithm that computes running totals of a sequence. Found at the heart of many parallel applications - from sorting algorithms to scientific simulations - it transforms a sequence of numbers into their running totals. While simple to compute sequentially, making this efficient on a GPU requires clever parallel thinking!"
---

# Prefix Sum

Prefix sum (also known as _scan_) is a fundamental parallel algorithm that computes running totals of a sequence. Found at the heart of many parallel applications - from sorting algorithms to scientific simulations - it transforms a sequence of numbers into their running totals. While simple to compute sequentially, making this efficient on a GPU requires clever parallel thinking!

## Overview

Prefix sum (also known as _scan_) is a fundamental parallel algorithm that computes running totals of a sequence. Found at the heart of many parallel applications - from sorting algorithms to scientific simulations - it transforms a sequence of numbers into their running totals. While simple to compute sequentially, making this efficient on a GPU requires clever parallel thinking!

Implement a kernel that computes a prefix-sum over 1D LayoutTensor `a` and stores it in 1D LayoutTensor `output`.

**Note:**_If the size of `a` is greater than the block size, only store the sum of each block._

## Key concepts

In this puzzle, you'll learn about:

- Parallel algorithms with logarithmic complexity
- Shared memory coordination patterns
- Multi-phase computation strategies

The key insight is understanding how to transform a sequential operation into an efficient parallel algorithm using shared memory.

For example, given an input sequence \\([3, 1, 4, 1, 5, 9]\\), the prefix sum would produce:

- \\([3]\\) (just the first element)
- \\([3, 4]\\) (3 + 1)
- \\([3, 4, 8]\\) (previous sum + 4)
- \\([3, 4, 8, 9]\\) (previous sum + 1)
- \\([3, 4, 8, 9, 14]\\) (previous sum + 5)
- \\([3, 4, 8, 9, 14, 23]\\) (previous sum + 9)

Mathematically, for a sequence \\([x_0, x_1, ..., x_n]\\), the prefix sum produces:
\\[[x_0, x_0+x_1, x_0+x_1+x_2, ..., \sum_{i=0}^n x_i] \\]

While a sequential algorithm would need \\(O(n)\\) steps, our parallel approach will use a clever two-phase algorithm that completes in \\(O(\log n)\\) steps! Here's a visualization of this process:

This puzzle is split into two parts to help you learn the concept:

- [Simple Version](#simple-version)
  Start with a single block implementation where all data fits in shared memory. This helps understand the core parallel algorithm.

- [Complete Version](#complete-version)
  Then tackle the more challenging case of handling larger arrays that span multiple blocks, requiring coordination between blocks.

Each version builds on the previous one, helping you develop a deep understanding of parallel prefix sum computation. The simple version establishes the fundamental algorithm, while the complete version shows how to scale it to larger datasets - a common requirement in real-world GPU applications.

## Simple Version

Implement a kernel that computes a prefix-sum over 1D LayoutTensor `a` and stores it in 1D LayoutTensor `output`.

**Note:**_If the size of `a` is greater than the block size, only store the sum of each block._

### Configuration

- Array size: `SIZE = 8` elements
- Threads per block: `TPB = 8`
- Number of blocks: 1
- Shared memory: `TPB` elements

Notes:

- **Data loading**: Each thread loads one element using LayoutTensor access
- **Memory pattern**: Shared memory for intermediate results using LayoutTensor with address_space
- **Thread sync**: Coordination between computation phases
- **Access pattern**: Stride-based parallel computation
- **Type safety**: Leveraging LayoutTensor's type system

### Reference implementation (example)


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

The parallel (inclusive) prefix-sum algorithm works as follows:

#### Setup & Configuration

- `TPB` (Threads Per Block) = 8
- `SIZE` (Array Size) = 8

#### Race condition prevention

The algorithm uses explicit synchronization to prevent read-write hazards:

- **Read Phase**: All threads first read the values they need into a local variable `current_val`
- **Synchronization**: `barrier()` ensures all reads complete before any writes begin
- **Write Phase**: All threads then safely write their computed values back to shared memory

This prevents the race condition that would occur if threads simultaneously read from and write to the same shared memory locations.

**Alternative approach**: Another solution to prevent race conditions is through _double buffering_, where you allocate twice the shared memory and alternate between reading from one buffer and writing to another. While this approach eliminates race conditions completely, it requires more shared memory and adds complexity. For educational purposes, we use the explicit synchronization approach as it's more straightforward to understand.

#### Thread mapping

- `thread_idx.x`: \\([0, 1, 2, 3, 4, 5, 6, 7]\\) (`local_i`)
- `block_idx.x`: \\([0, 0, 0, 0, 0, 0, 0, 0]\\)
- `global_i`: \\([0, 1, 2, 3, 4, 5, 6, 7]\\) (`block_idx.x * TPB + thread_idx.x`)

#### Initial load to shared memory

```txt
Threads:      T   T   T   T   T   T   T   T
Input array:  [0    1    2    3    4    5    6    7]
shared:       [0    1    2    3    4    5    6    7]
                                           
              T   T   T   T   T   T   T   T
```

#### Offset = 1: First Parallel Step

Active threads: \\(T_1 \ldots T_7\\) (where `local_i  1`)

**Read Phase**: Each thread reads the value it needs:

```txt
T reads shared[0] = 0    T reads shared[4] = 4
T reads shared[1] = 1    T reads shared[5] = 5
T reads shared[2] = 2    T reads shared[6] = 6
T reads shared[3] = 3
```

**Synchronization**: `barrier()` ensures all reads complete

**Write Phase**: Each thread adds its read value to its current position:

```txt
Before:      [0    1    2    3    4    5    6    7]
Add:              +0   +1   +2   +3   +4   +5   +6
                   |    |    |    |    |    |    |
Result:      [0    1    3    5    7    9    11   13]
                                           
                  T   T   T   T   T   T   T
```

#### Offset = 2: Second Parallel Step

Active threads: \\(T_2 \ldots T_7\\) (where `local_i  2`)

**Read Phase**: Each thread reads the value it needs:

```txt
T reads shared[0] = 0    T reads shared[3] = 5
T reads shared[1] = 1    T reads shared[4] = 7
T reads shared[2] = 3    T reads shared[5] = 9
```

**Synchronization**: `barrier()` ensures all reads complete

**Write Phase**: Each thread adds its read value:

```txt
Before:      [0    1    3    5    7    9    11   13]
Add:                   +0   +1   +3   +5   +7   +9
                        |    |    |    |    |    |
Result:      [0    1    3    6    10   14   18   22]
                                            
                       T   T   T   T   T   T
```

#### Offset = 4: Third Parallel Step

Active threads: \\(T_4 \ldots T_7\\) (where `local_i  4`)

**Read Phase**: Each thread reads the value it needs:

```txt
T reads shared[0] = 0    T reads shared[2] = 3
T reads shared[1] = 1    T reads shared[3] = 6
```

**Synchronization**: `barrier()` ensures all reads complete

**Write Phase**: Each thread adds its read value:

```txt
Before:      [0    1    3    6    10   14   18   22]
Add:                              +0   +1   +3   +6
                                  |    |    |    |
Result:      [0    1    3    6    10   15   21   28]
                                              
                                  T   T   T   T
```

#### Final write to output

```txt
Threads:      T   T   T   T   T   T   T   T
global_i:     0    1    2    3    4    5    6    7
output:       [0    1    3    6    10   15   21   28]
                                          
              T   T   T   T   T   T   T   T
```

#### Key implementation details

**Synchronization Pattern**: Each iteration follows a strict read  sync  write pattern:

1. `var current_val: out.element_type = 0` - Initialize local variable
2. `current_val = shared[local_i - offset]` - Read phase (if conditions met)
3. `barrier()` - Explicit synchronization to prevent race conditions
4. `shared[local_i] += current_val` - Write phase (if conditions met)
5. `barrier()` - Standard synchronization before next iteration

**Race Condition Prevention**: Without the explicit read-write separation, multiple threads could simultaneously access the same shared memory location, leading to undefined behavior. The two-phase approach with explicit synchronization ensures correctness.

**Memory Safety**: The algorithm maintains memory safety through:

- Bounds checking with `if local_i >= offset and local_i < size`
- Proper initialization of the temporary variable
- Coordinated access patterns that prevent data races

The solution ensures correct synchronization between phases using `barrier()` and handles array bounds checking with `if global_i < size`. The final result produces the inclusive prefix sum where each element \\(i\\) contains \\(\sum_{j=0}^{i} a[j]\\).

## Complete Version

Implement a kernel that computes a prefix-sum over 1D LayoutTensor `a` and stores it in 1D LayoutTensor `output`.

**Note:**_If the size of `a` is greater than the block size, we need to synchronize across multiple blocks to get the correct result._

### Configuration

- Array size: `SIZE_2 = 15` elements
- Threads per block: `TPB = 8`
- Number of blocks: 2
- Shared memory: `TPB` elements per block

Notes:

- **Multiple blocks**: When the input array is larger than one block, we need a multi-phase approach
- **Block-level sync**: Within a block, use `barrier()` to synchronize threads
- **Host-level sync**: Between blocks, Mojo's `DeviceContext` ensures kernel launches are ordered, which means they start in the order they where scheduled and wait for the previous kernel to finish before starting. You may need to use `ctx.synchronize()` to ensure all GPU work is complete before reading results back to the host.
- **Auxiliary storage**: Use extra space to store block sums for cross-block communication

### Reference implementation (example)


```mojo

# Kernel 1: Compute local prefix sums and store block sums in out
fn prefix_sum_local_phase
    out_layout: Layout, in_layout: Layout
:
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x
    shared = LayoutTensor[
        dtype,
        Layout.row_major(TPB),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Load data into shared memory
    # Example with SIZE_2=15, TPB=8, BLOCKS=2:
    # Block 0 shared mem: [0,1,2,3,4,5,6,7]
    # Block 1 shared mem: [8,9,10,11,12,13,14,uninitialized]
    # Note: The last position remains uninitialized since global_i >= size,
    # but this is safe because that thread doesn't participate in computation
    if global_i < size:
        shared[local_i] = a[global_i]

    barrier()

    # Compute local prefix sum using parallel reduction
    # This uses a tree-based algorithm with log(TPB) iterations
    # Iteration 1 (offset=1):
    #   Block 0: [0,0+1,2+1,3+2,4+3,5+4,6+5,7+6] = [0,1,3,5,7,9,11,13]
    # Iteration 2 (offset=2):
    #   Block 0: [0,1,3+0,5+1,7+3,9+5,11+7,13+9] = [0,1,3,6,10,14,18,22]
    # Iteration 3 (offset=4):
    #   Block 0: [0,1,3,6,10+0,14+1,18+3,22+6] = [0,1,3,6,10,15,21,28]
    #   Block 1 follows same pattern to get [8,17,27,38,50,63,77,???]
    offset = UInt(1)
    for i in range(Int(log2(Scalardtype))):
        var current_val: output.element_type = 0
        if local_i >= offset and local_i < TPB:
            current_val = shared[local_i - offset]  # read

        barrier()
        if local_i >= offset and local_i < TPB:
            shared[local_i] += current_val  # write

        barrier()
        offset *= 2

    # Write local results to output
    # Block 0 writes: [0,1,3,6,10,15,21,28]
    # Block 1 writes: [8,17,27,38,50,63,77,???]
    if global_i < size:
        output[global_i] = shared[local_i]

    # Store block sums in auxiliary space
    # Block 0: Thread 7 stores shared[7] == 28 at position size+0 (position 15)
    # Block 1: Thread 7 stores shared[7] == ??? at position size+1 (position 16).  This sum is not needed for the final output.
    # This gives us: [0,1,3,6,10,15,21,28, 8,17,27,38,50,63,77, 28,???]
    #                                                             
    #                                                     Block sums here
    if local_i == TPB - 1:
        output[size + block_idx.x] = shared[local_i]

# Kernel 2: Add block sums to their respective blocks
fn prefix_sum_block_sum_phase
    layout: Layout
:
    global_i = block_dim.x * block_idx.x + thread_idx.x

    # Second pass: add previous block's sum to each element
    # Block 0: No change needed - already correct
    # Block 1: Add Block 0's sum (28) to each element
    #   Before: [8,17,27,38,50,63,77]
    #   After: [36,45,55,66,78,91,105]
    # Final result combines both blocks:
    # [0,1,3,6,10,15,21,28, 36,45,55,66,78,91,105]
    if block_idx.x > 0 and global_i < size:
        prev_block_sum = output[size + block_idx.x - 1]
        output[global_i] += prev_block_sum

```

This solution implements a multi-block prefix sum using a two-kernel approach to handle an array that spans multiple thread blocks. Let's break down each aspect in detail:

### Example scenario of cross-block communication

The fundamental limitation in GPU programming is that threads can only synchronize within a block using `barrier()`. When data spans multiple blocks, we face Example scenario: **How do we ensure blocks can communicate their partial results to other blocks?**

#### Memory layout visualization

For our test case with `SIZE_2 = 15` and `TPB = 8`:

```
Input array:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

Block 0 processes: [0, 1, 2, 3, 4, 5, 6, 7]
Block 1 processes: [8, 9, 10, 11, 12, 13, 14] (7 valid elements)
```

We extend the output buffer to include space for block sums:

```
Extended buffer: [data values (15 elements)] + [block sums (2 elements)]
                 [0...14] + [block0_sum, block1_sum]
```

The size of this extended buffer is: `EXTENDED_SIZE = SIZE_2 + num_blocks = 15 + 2 = 17`

### Phase 1 kernel: Local prefix sums

#### Race condition prevention in local phase

The local phase uses the same explicit synchronization pattern as the simple version to prevent read-write hazards:

- **Read Phase**: All threads first read the values they need into a local variable `current_val`
- **Synchronization**: `barrier()` ensures all reads complete before any writes begin
- **Write Phase**: All threads then safely write their computed values back to shared memory

This prevents race conditions that could occur when multiple threads simultaneously access the same shared memory locations during the parallel reduction.

#### Step-by-step execution for Block 0

1. **Load values into shared memory**:

   ```
   shared = [0, 1, 2, 3, 4, 5, 6, 7]
   ```

2. **Iterations of parallel reduction**(\\(\log_2(TPB) = 3\\) iterations):

   **Iteration 1**(offset=1):

   **Read Phase**: Each active thread reads the value it needs:

   ```
   T reads shared[0] = 0    T reads shared[4] = 4
   T reads shared[1] = 1    T reads shared[5] = 5
   T reads shared[2] = 2    T reads shared[6] = 6
   T reads shared[3] = 3
   ```

   **Synchronization**: `barrier()` ensures all reads complete

   **Write Phase**: Each thread adds its read value:

   ```
   shared[0] = 0              (unchanged)
   shared[1] = 1 + 0 = 1
   shared[2] = 2 + 1 = 3
   shared[3] = 3 + 2 = 5
   shared[4] = 4 + 3 = 7
   shared[5] = 5 + 4 = 9
   shared[6] = 6 + 5 = 11
   shared[7] = 7 + 6 = 13
   ```

   After barrier: `shared = [0, 1, 3, 5, 7, 9, 11, 13]`

   **Iteration 2**(offset=2):

   **Read Phase**: Each active thread reads the value it needs:

   ```
   T reads shared[0] = 0    T reads shared[3] = 5
   T reads shared[1] = 1    T reads shared[4] = 7
   T reads shared[2] = 3    T reads shared[5] = 9
   ```

   **Synchronization**: `barrier()` ensures all reads complete

   **Write Phase**: Each thread adds its read value:

   ```
   shared[0] = 0              (unchanged)
   shared[1] = 1              (unchanged)
   shared[2] = 3 + 0 = 3      (unchanged)
   shared[3] = 5 + 1 = 6
   shared[4] = 7 + 3 = 10
   shared[5] = 9 + 5 = 14
   shared[6] = 11 + 7 = 18
   shared[7] = 13 + 9 = 22
   ```

   After barrier: `shared = [0, 1, 3, 6, 10, 14, 18, 22]`

   **Iteration 3**(offset=4):

   **Read Phase**: Each active thread reads the value it needs:

   ```
   T reads shared[0] = 0    T reads shared[2] = 3
   T reads shared[1] = 1    T reads shared[3] = 6
   ```

   **Synchronization**: `barrier()` ensures all reads complete

   **Write Phase**: Each thread adds its read value:

   ```
   shared[0] = 0              (unchanged)
   shared[1] = 1              (unchanged)
   shared[2] = 3              (unchanged)
   shared[3] = 6              (unchanged)
   shared[4] = 10 + 0 = 10    (unchanged)
   shared[5] = 14 + 1 = 15
   shared[6] = 18 + 3 = 21
   shared[7] = 22 + 6 = 28
   ```

   After barrier: `shared = [0, 1, 3, 6, 10, 15, 21, 28]`

3. **Write local results back to global memory**:

   ```
   output[0...7] = [0, 1, 3, 6, 10, 15, 21, 28]
   ```

4. **Store block sum in auxiliary space**(only last thread):

   ```
   output[15] = 28  // at position size + block_idx.x = 15 + 0
   ```

#### Step-by-step execution for Block 1

1. **Load values into shared memory**:

   ```
   shared = [8, 9, 10, 11, 12, 13, 14, uninitialized]
   ```

   Note: Thread 7 doesn't load anything since `global_i = 15 >= SIZE_2`, leaving `shared[7]` uninitialized. This is safe because Thread 7 won't participate in the final output.

2. **Iterations of parallel reduction**(\\(\log_2(TPB) = 3\\) iterations):

   Only the first 7 threads participate in meaningful computation. After all three iterations:

   ```
   shared = [8, 17, 27, 38, 50, 63, 77, uninitialized]
   ```

3. **Write local results back to global memory**:

   ```
   output[8...14] = [8, 17, 27, 38, 50, 63, 77]  // Only 7 valid outputs
   ```

4. **Store block sum in auxiliary space**(only last thread in block):

   ```
   output[16] = shared[7]  // Thread 7 (TPB-1) stores whatever is in shared[7]
   ```

   Note: Even though Thread 7 doesn't load valid input data, it still participates in the prefix sum computation within the block. The `shared[7]` position gets updated during the parallel reduction iterations, but since it started uninitialized, the final value is unpredictable. However, this doesn't affect correctness because Block 1 is the last block, so this block sum is never used in Phase 2.

After Phase 1, the output buffer contains:

```
[0, 1, 3, 6, 10, 15, 21, 28, 8, 17, 27, 38, 50, 63, 77, 28, ???]
                                                        ^   ^
                                                Block sums stored here
```

Note: The last block sum (???) is unpredictable since it's based on uninitialized memory, but this doesn't affect the final result.

### Host-device synchronization: When it's actually needed

The two kernel phases execute sequentially **without any explicit synchronization**between them:

```mojo
# Phase 1: Local prefix sums
ctx.enqueue_function[prefix_sum_local_phase[...], prefix_sum_local_phase[...]](...)

# Phase 2: Add block sums (automatically waits for Phase 1)
ctx.enqueue_function[prefix_sum_block_sum_phase[...], prefix_sum_block_sum_phase[...]](...)
```

**Key insight**: Mojo's `DeviceContext` uses a single execution stream (CUDA stream on NVIDIA GPUs, HIP stream on AMD ROCm GPUs), which guarantees that kernel launches execute in the exact order they are enqueued. No explicit synchronization is needed between kernels.

**When `ctx.synchronize()` is needed**:

```mojo
# After both kernels complete, before reading results on host
ctx.synchronize()  # Host waits for GPU to finish

with out.map_to_host() as out_host:  # Now safe to read GPU results
    print("out:", out_host)
```

The `ctx.synchronize()` call serves its traditional purpose:

- **Host-device synchronization**: Ensures the host waits for all GPU work to complete before accessing results
- **Memory safety**: Prevents reading GPU memory before computations finish

**Execution model**: Unlike `barrier()` which synchronizes threads within a block, kernel ordering comes from Mojo's single-stream execution model, while `ctx.synchronize()` handles host-device coordination.

### Phase 2 kernel: Block sum addition

1. **Block 0**: No changes needed (it's already correct).

2. **Block 1**: Each thread adds Block 0's sum to its element:

   ```
   prev_block_sum = output[size + block_idx.x - 1] = output[15] = 28
   output[global_i] += prev_block_sum
   ```

   Block 1 values are transformed:

   ```
   Before: [8, 17, 27, 38, 50, 63, 77]
   After:  [36, 45, 55, 66, 78, 91, 105]
   ```

### Performance and optimization considerations

#### Key implementation details

**Local phase synchronization pattern**: Each iteration within a block follows a strict read  sync  write pattern:

1. `var current_val: out.element_type = 0` - Initialize local variable
2. `current_val = shared[local_i - offset]` - Read phase (if conditions met)
3. `barrier()` - Explicit synchronization to prevent race conditions
4. `shared[local_i] += current_val` - Write phase (if conditions met)
5. `barrier()` - Standard synchronization before next iteration

**Cross-block synchronization**: The algorithm uses two levels of synchronization:

- **Intra-block**: `barrier()` synchronizes threads within each block during local prefix sum computation
- **Inter-block**: The `DeviceContext` context manager that launches enqueued kernels sequentially to ensure Phase 1 completes before Phase 2 begins. To explicitly enforce host-device synchronization before reading results, `ctx.synchronize()` is used.

**Race condition prevention**: The explicit read-write separation in the local phase prevents the race condition that would occur if threads simultaneously read from and write to the same shared memory locations during parallel reduction.

1. **Work efficiency**: This implementation has \\(O(n \log n)\\) work complexity, while the sequential algorithm is \\(O(n)\\). This is a classic space-time tradeoff in parallel algorithms.

2. **Memory overhead**: The extra space for block sums is minimal (just one element per block).

This two-kernel approach is a fundamental pattern in GPU programming for algorithms that require cross-block communication. The same strategy can be applied to other parallel algorithms like radix sort, histogram calculation, and reduction operations.
