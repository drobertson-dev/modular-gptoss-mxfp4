---
title: "Pooling"
description: "Implement a kernel that computes the running sum of the last 3 positions of vector `a` and stores it in vector `output`."
---

# Pooling

Implement a kernel that computes the running sum of the last 3 positions of vector `a` and stores it in vector `output`.

## Overview

Implement a kernel that computes the running sum of the last 3 positions of vector `a` and stores it in vector `output`.

**Note:**_You have 1 thread per position. You only need 1 global read and 1 global write per thread._

## Implementation approaches

###  Raw memory approach
Learn how to implement sliding window operations with manual memory management and synchronization.

###  LayoutTensor Version
Use LayoutTensor's features for efficient window-based operations and shared memory management.

 **Note**: See how LayoutTensor simplifies sliding window operations while maintaining efficient memory access patterns.

### Overview

Implement a kernel that compute the running sum of the last 3 positions of vector `a` and stores it in vector `output`.

**Note:**_You have 1 thread per position. You only need 1 global read and 1 global write per thread._

### Key concepts

In this puzzle, you'll learn about:

- Using shared memory for sliding window operations
- Handling boundary conditions in pooling
- Coordinating thread access to neighboring elements

The key insight is understanding how to efficiently access a window of elements using shared memory, with special handling for the first elements in the sequence.

### Configuration

- Array size: `SIZE = 8` elements
- Threads per block: `TPB = 8`
- Window size: 3 elements
- Shared memory: `TPB` elements

Notes:

- **Window access**: Each output depends on up to 3 previous elements
- **Edge handling**: First two positions need special treatment
- **Memory pattern**: One shared memory load per thread
- **Thread sync**: Coordination before window operations

### Running the code

To test your solution, run the following command in your terminal:

  
    pixi NVIDIA (default)
    pixi AMD
    pixi Apple
    uv
  
  

```bash
pixi run p11
```

  
  

```bash
pixi run -e amd p11
```

  
  

```bash
pixi run -e apple p11
```

  
  

```bash
uv run poe p11
```

  

Your output will look like this if the puzzle isn't solved yet:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([0.0, 1.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0])
```

### Reference implementation (example)


```mojo
fn pooling(
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    size: UInt,
):
    shared = stack_allocation[
        TPB,
        Scalar[dtype],
        address_space = AddressSpace.SHARED,
    ]()
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x
    if global_i < size:
        shared[local_i] = a[global_i]

    barrier()

    if global_i == 0:
        output[0] = shared[0]
    elif global_i == 1:
        output[1] = shared[0] + shared[1]
    elif UInt(1) < global_i < size:
        output[global_i] = (
            shared[local_i - 2] + shared[local_i - 1] + shared[local_i]
        )

```

The solution implements a sliding window sum using shared memory with these key steps:

1. **Shared memory setup**
   - Allocates `TPB` elements in shared memory:

     ```txt
     Input array:  [0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0]
     Block shared: [0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0]
     ```

   - Each thread loads one element from global memory
   - `barrier()` ensures all data is loaded

2. **Boundary cases**
   - Position 0: Single element

     ```txt
     output[0] = shared[0] = 0.0
     ```

   - Position 1: Sum of first two elements

     ```txt
     output[1] = shared[0] + shared[1] = 0.0 + 1.0 = 1.0
     ```

3. **Main window operation**
   - For positions 2 and beyond:

     ```txt
     Position 2: shared[0] + shared[1] + shared[2] = 0.0 + 1.0 + 2.0 = 3.0
     Position 3: shared[1] + shared[2] + shared[3] = 1.0 + 2.0 + 3.0 = 6.0
     Position 4: shared[2] + shared[3] + shared[4] = 2.0 + 3.0 + 4.0 = 9.0
     ...
     ```

   - Window calculation using local indices:

     ```txt
     # Sliding window of 3 elements
     window_sum = shared[i-2] + shared[i-1] + shared[i]
     ```

4. **Memory access pattern**
   - One global read per thread into shared memory
   - One global write per thread from shared memory
   - Uses shared memory for efficient neighbor access
   - Maintains coalesced memory access pattern

This approach optimizes performance through:

- Minimal global memory access
- Fast shared memory neighbor lookups
- Clean boundary handling
- Efficient memory coalescing

The final output shows the cumulative window sums:

```txt
[0.0, 1.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0]
```

### Overview

Implement a kernel that compute the running sum of the last 3 positions of 1D LayoutTensor `a` and stores it in 1D LayoutTensor `output`.

**Note:**_You have 1 thread per position. You only need 1 global read and 1 global write per thread._

### Key concepts

In this puzzle, you'll learn about:

- Using LayoutTensor for sliding window operations
- Managing shared memory with LayoutTensor address_space that we saw in puzzle_08
- Efficient neighbor access patterns
- Boundary condition handling

The key insight is how LayoutTensor simplifies shared memory management while maintaining efficient window-based operations.

### Configuration

- Array size: `SIZE = 8` elements
- Threads per block: `TPB = 8`
- Window size: 3 elements
- Shared memory: `TPB` elements

Notes:

- **LayoutTensor allocation**: Use `LayoutTensor[dtype, Layout.row_major(TPB), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()`
- **Window access**: Natural indexing for 3-element windows
- **Edge handling**: Special cases for first two positions
- **Memory pattern**: One shared memory load per thread

### Running the code

To test your solution, run the following command in your terminal:

  
    pixi NVIDIA (default)
    pixi AMD
    pixi Apple
    uv
  
  

```bash
pixi run p11_layout_tensor
```

  
  

```bash
pixi run -e amd p11_layout_tensor
```

  
  

```bash
pixi run -e apple p11_layout_tensor
```

  
  

```bash
uv run poe p11_layout_tensor
```

  

Your output will look like this if the puzzle isn't solved yet:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([0.0, 1.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0])
```

### Reference implementation (example)


```mojo
fn pooling
    layout: Layout
:
    # Allocate shared memory using tensor builder
    shared = LayoutTensor[
        dtype,
        Layout.row_major(TPB),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x

    # Load data into shared memory
    if global_i < size:
        shared[local_i] = a[global_i]

    # Synchronize threads within block
    barrier()

    # Handle first two special cases
    if global_i == 0:
        output[0] = shared[0]
    elif global_i == 1:
        output[1] = shared[0] + shared[1]
    # Handle general case
    elif UInt(1) < global_i < size:
        output[global_i] = (
            shared[local_i - 2] + shared[local_i - 1] + shared[local_i]
        )

```

The solution implements a sliding window sum using LayoutTensor with these key steps:

1. **Shared memory setup**
   - LayoutTensor creates block-local storage with address_space:

     ```txt
     shared = LayoutTensor[dtype, Layout.row_major(TPB), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()
     ```

   - Each thread loads one element:

     ```txt
     Input array:  [0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0]
     Block shared: [0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0]
     ```

   - `barrier()` ensures all data is loaded

2. **Boundary cases**
   - Position 0: Single element

     ```txt
     output[0] = shared[0] = 0.0
     ```

   - Position 1: Sum of first two elements

     ```txt
     output[1] = shared[0] + shared[1] = 0.0 + 1.0 = 1.0
     ```

3. **Main window operation**
   - For positions 2 and beyond:

     ```txt
     Position 2: shared[0] + shared[1] + shared[2] = 0.0 + 1.0 + 2.0 = 3.0
     Position 3: shared[1] + shared[2] + shared[3] = 1.0 + 2.0 + 3.0 = 6.0
     Position 4: shared[2] + shared[3] + shared[4] = 2.0 + 3.0 + 4.0 = 9.0
     ...
     ```

   - Natural indexing with LayoutTensor:

     ```txt
     # Sliding window of 3 elements
     window_sum = shared[i-2] + shared[i-1] + shared[i]
     ```

4. **Memory access pattern**
   - One global read per thread into shared tensor
   - Efficient neighbor access through shared memory
   - LayoutTensor benefits:
     - Automatic bounds checking
     - Natural window indexing
     - Layout-aware memory access
     - Type safety throughout

This approach combines the performance of shared memory with LayoutTensor's safety and ergonomics:

- Minimizes global memory access
- Simplifies window operations
- Handles boundaries cleanly
- Maintains coalesced access patterns

The final output shows the cumulative window sums:

```txt
[0.0, 1.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0]
```
