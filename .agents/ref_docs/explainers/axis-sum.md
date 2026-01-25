---
title: "Axis Sum"
description: "Implement a kernel that computes a sum over each row of 2D matrix `a` and stores it in `output` using LayoutTensor."
---

# Axis Sum

Implement a kernel that computes a sum over each row of 2D matrix `a` and stores it in `output` using LayoutTensor.

## Overview

Implement a kernel that computes a sum over each row of 2D matrix `a` and stores it in `output` using LayoutTensor.

## Key concepts

This puzzle covers:

- Parallel reduction along matrix dimensions using LayoutTensor
- Using block coordinates for data partitioning
- Efficient shared memory reduction patterns
- Working with multi-dimensional tensor layouts

The key insight is understanding how to map thread blocks to matrix rows and perform efficient parallel reduction within each block while leveraging LayoutTensor's dimensional indexing.

## Configuration

- Matrix dimensions: \\(\\text{BATCH} \\times \\text{SIZE} = 4 \\times 6\\)
- Threads per block: \\(\\text{TPB} = 8\\)
- Grid dimensions: \\(1 \\times \\text{BATCH}\\)
- Shared memory: \\(\\text{TPB}\\) elements per block
- Input layout: `Layout.row_major(BATCH, SIZE)`
- Output layout: `Layout.row_major(BATCH, 1)`

Matrix visualization:

```txt
Row 0: [0, 1, 2, 3, 4, 5]       -> Block(0,0)
Row 1: [6, 7, 8, 9, 10, 11]     -> Block(0,1)
Row 2: [12, 13, 14, 15, 16, 17] -> Block(0,2)
Row 3: [18, 19, 20, 21, 22, 23] -> Block(0,3)
```

## Running the Code

To test your solution, run the following command in your terminal:

  
    pixi NVIDIA (default)
    pixi AMD
    pixi Apple
    uv
  
  

```bash
pixi run p15
```

  
  

```bash
pixi run -e amd p15
```

  
  

```bash
pixi run -e apple p15
```

  
  

```bash
uv run poe p15
```

  

Your output will look like this if the puzzle isn't solved yet:

```txt
out: DeviceBuffer([0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([15.0, 51.0, 87.0, 123.0])
```

## Reference implementation (example)


```mojo
fn axis_sum
    in_layout: Layout, out_layout: Layout
:
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x
    batch = block_idx.y
    cache = LayoutTensor[
        dtype,
        Layout.row_major(TPB),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Visualize:
    # Block(0,0): [T0,T1,T2,T3,T4,T5,T6,T7] -> Row 0: [0,1,2,3,4,5]
    # Block(0,1): [T0,T1,T2,T3,T4,T5,T6,T7] -> Row 1: [6,7,8,9,10,11]
    # Block(0,2): [T0,T1,T2,T3,T4,T5,T6,T7] -> Row 2: [12,13,14,15,16,17]
    # Block(0,3): [T0,T1,T2,T3,T4,T5,T6,T7] -> Row 3: [18,19,20,21,22,23]

    # each row is handled by each block bc we have grid_dim=(1, BATCH)

    if local_i < size:
        cache[local_i] = a[batch, local_i]
    else:
        # Add zero-initialize padding elements for later reduction
        cache[local_i] = 0

    barrier()

    # do reduction sum per each block
    stride = UInt(TPB // 2)
    while stride > 0:
        # Read phase: all threads read the values they need first to avoid race conditions
        var temp_val: output.element_type = 0
        if local_i < stride:
            temp_val = cache[local_i + stride]

        barrier()

        # Write phase: all threads safely write their computed values
        if local_i < stride:
            cache[local_i] += temp_val

        barrier()
        stride //= 2

    # writing with local thread = 0 that has the sum for each batch
    if local_i == 0:
        output[batch, 0] = cache[0]

```

The solution implements a parallel row-wise sum reduction for a 2D matrix using LayoutTensor. Here's a comprehensive breakdown:

### Matrix layout and block mapping

```txt
Input Matrix (4x6) with LayoutTensor:                Block Assignment:
[[ a[0,0]  a[0,1]  a[0,2]  a[0,3]  a[0,4]  a[0,5] ] -> Block(0,0)
 [ a[1,0]  a[1,1]  a[1,2]  a[1,3]  a[1,4]  a[1,5] ] -> Block(0,1)
 [ a[2,0]  a[2,1]  a[2,2]  a[2,3]  a[2,4]  a[2,5] ] -> Block(0,2)
 [ a[3,0]  a[3,1]  a[3,2]  a[3,3]  a[3,4]  a[3,5] ] -> Block(0,3)
```

### Parallel reduction process

1. **Initial Data Loading**:

   ```txt
   Block(0,0): cache = [a[0,0] a[0,1] a[0,2] a[0,3] a[0,4] a[0,5] * *]  // * = padding
   Block(0,1): cache = [a[1,0] a[1,1] a[1,2] a[1,3] a[1,4] a[1,5] * *]
   Block(0,2): cache = [a[2,0] a[2,1] a[2,2] a[2,3] a[2,4] a[2,5] * *]
   Block(0,3): cache = [a[3,0] a[3,1] a[3,2] a[3,3] a[3,4] a[3,5] * *]
   ```

2. **Reduction Steps**(for Block 0,0):

   ```txt
   Initial:  [0  1  2  3  4  5  *  *]
   Stride 4: [4  5  6  7  4  5  *  *]
   Stride 2: [10 12 6  7  4  5  *  *]
   Stride 1: [15 12 6  7  4  5  *  *]
   ```

### Key implementation features

1. **Layout Configuration**:
   - Input: row-major layout (BATCH  SIZE)
   - Output: row-major layout (BATCH  1)
   - Each block processes one complete row

2. **Memory Access Pattern**:
   - LayoutTensor 2D indexing for input: `a[batch, local_i]`
   - Shared memory for efficient reduction
   - LayoutTensor 2D indexing for output: `output[batch, 0]`

3. **Parallel Reduction Logic**:

   ```mojo
   stride = TPB // 2
   while stride > 0:
       if local_i < stride:
           cache[local_i] += cache[local_i + stride]
       barrier()
       stride //= 2
   ```

   **Note**: This implementation has a potential race condition where threads simultaneously read from and write to shared memory during the same iteration. A safer approach would separate the read and write phases:

   ```mojo
   stride = TPB // 2
   while stride > 0:
       var temp_val: output.element_type = 0
       if local_i < stride:
           temp_val = cache[local_i + stride]  # Read phase
       barrier()
       if local_i < stride:
           cache[local_i] += temp_val  # Write phase
       barrier()
       stride //= 2
   ```

4. **Output Writing**:

   ```mojo
   if local_i == 0:
       output[batch, 0] = cache[0]  --> One result per batch
   ```

### Performance optimizations

1. **Memory Efficiency**:
   - Coalesced memory access through LayoutTensor
   - Shared memory for fast reduction
   - Single write per row result

2. **Thread Utilization**:
   - Perfect load balancing across rows
   - No thread divergence in main computation
   - Efficient parallel reduction pattern

3. **Synchronization**:
   - Minimal barriers (only during reduction)
   - Independent processing between rows
   - No inter-block communication needed
   - **Race condition consideration**: The current implementation may have read-write hazards during parallel reduction that could be resolved with explicit read-write phase separation

### Complexity analysis

- Time: \\(O(\log n)\\) per row, where n is row length
- Space: \\(O(TPB)\\) shared memory per block
- Total parallel time: \\(O(\log n)\\) with sufficient threads
