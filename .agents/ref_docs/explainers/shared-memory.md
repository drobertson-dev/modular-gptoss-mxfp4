---
title: "Shared Memory"
description: "Implement a kernel that adds 10 to each position of a vector `a` and stores it in vector `output`."
---

# Shared Memory

Implement a kernel that adds 10 to each position of a vector `a` and stores it in vector `output`.

## Overview

Implement a kernel that adds 10 to each position of a vector `a` and stores it in vector `output`.

**Note:**_You have fewer threads per block than the size of `a`._

## Implementation approaches

###  Raw memory approach
Learn how to manually manage shared memory and synchronization.

###  LayoutTensor Version
Use LayoutTensor's built-in shared memory management features.

 **Note**: Experience how LayoutTensor simplifies shared memory operations while maintaining performance.

### Overview

Implement a kernel that adds 10 to each position of a vector `a` and stores it in `output`.

**Note:**_You have fewer threads per block than the size of `a`._

### Key concepts

In this puzzle, you'll learn about:

- Using shared memory within thread blocks
- Synchronizing threads with barriers
- Managing block-local data storage

The key insight is understanding how shared memory provides fast, block-local storage that all threads in a block can access, requiring careful coordination between threads.

### Configuration

- Array size: `SIZE = 8` elements
- Threads per block: `TPB = 4`
- Number of blocks: 2
- Shared memory: `TPB` elements per block

Notes:

- **Shared memory**: Fast storage shared by threads in a block
- **Thread sync**: Coordination using `barrier()`
- **Memory scope**: Shared memory only visible within block
- **Access pattern**: Local vs global indexing

> **Warning**: Each block can only have a _constant_ amount of shared memory that threads in that block can read and write to. This needs to be a literal python constant, not a variable. After writing to shared memory you need to call [barrier](https://docs.modular.com/mojo/stdlib/gpu/sync/barrier/) to ensure that threads do not cross.

**Educational Note**: In this specific puzzle, the `barrier()` isn't strictly necessary since each thread only accesses its own shared memory location. However, it's included to teach proper shared memory synchronization patterns for more complex scenarios where threads need to coordinate access to shared data.

### Running the code

To test your solution, run the following command in your terminal:

  
    pixi NVIDIA (default)
    pixi AMD
    pixi Apple
    uv
  
  

```bash
pixi run p08
```

  
  

```bash
pixi run -e amd p08
```

  
  

```bash
pixi run -e apple p08
```

  
  

```bash
uv run poe p08
```

  

Your output will look like this if the puzzle isn't solved yet:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0])
```

### Reference implementation (example)


```mojo
fn add_10_shared(
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
    # local data into shared memory
    if global_i < size:
        shared[local_i] = a[global_i]

    # wait for all threads to complete
    # works within a thread block
    # Note: barrier is not strictly needed here since each thread only accesses its own shared memory location.
    # However, it's included to teach proper shared memory synchronization patterns
    # for more complex scenarios where threads need to coordinate access to shared data.
    # For this specific puzzle, we can remove the barrier since each thread only accesses its own shared memory location.
    barrier()

    # process using shared memory
    if global_i < size:
        output[global_i] = shared[local_i] + 10

```

This solution demonstrates key concepts of shared memory usage in GPU programming:

1. **Memory hierarchy**
   - Global memory: `a` and `output` arrays (slow, visible to all blocks)
   - Shared memory: `shared` array (fast, thread-block local)
   - Example for 8 elements with 4 threads per block:

     ```txt
     Global array a: [1 1 1 1 | 1 1 1 1]  # Input: all ones

     Block (0):      Block (1):
     shared[0..3]    shared[0..3]
     [1 1 1 1]       [1 1 1 1]
     ```

2. **Thread coordination**
   - Load phase:

     ```txt
     Thread 0: shared[0] = a[0]=1    Thread 2: shared[2] = a[2]=1
     Thread 1: shared[1] = a[1]=1    Thread 3: shared[3] = a[3]=1
     barrier()                                 # Wait for all loads
     ```

   - Process phase: Each thread adds 10 to its shared memory value
   - Result: `output[i] = shared[local_i] + 10 = 11`

   **Note**: In this specific case, the `barrier()` isn't strictly necessary since each thread only writes to and reads from its own shared memory location (`shared[local_i]`). However, it's included for educational purposes to demonstrate proper shared memory synchronization patterns that are essential when threads need to access each other's data.

3. **Index mapping**
   - Global index: `block_dim.x * block_idx.x + thread_idx.x`

     ```txt
     Block 0 output: [11 11 11 11]
     Block 1 output: [11 11 11 11]
     ```

   - Local index: `thread_idx.x` for shared memory access

     ```txt
     Both blocks process: 1 + 10 = 11
     ```

4. **Memory access pattern**
   - Load: Global  Shared (coalesced reads of 1s)
   - Sync: `barrier()` ensures all loads complete
   - Process: Add 10 to shared values
   - Store: Write 11s back to global memory

This pattern shows how to use shared memory to optimize data access while maintaining thread coordination within blocks.

### Overview

Implement a kernel that adds 10 to each position of a 1D LayoutTensor `a` and stores it in 1D LayoutTensor `output`.

**Note:**_You have fewer threads per block than the size of `a`._

### Key concepts

In this puzzle, you'll learn about:

- Using LayoutTensor's shared memory features with address_space
- Thread synchronization with shared memory
- Block-local data management with LayoutTensor

The key insight is how LayoutTensor simplifies shared memory management while maintaining the performance benefits of block-local storage.

### Configuration

- Array size: `SIZE = 8` elements
- Threads per block: `TPB = 4`
- Number of blocks: 2
- Shared memory: `TPB` elements per block

### Key differences from raw approach

1. **Memory allocation**: We will use [LayoutTensor](https://docs.modular.com/mojo/stdlib/layout/layout_tensor/LayoutTensor) with address_space instead of [stack_allocation](https://docs.modular.com/mojo/stdlib/memory/memory/stack_allocation/)

   ```mojo
   # Raw approach
   shared = stack_allocation[TPB, Scalar[dtype]]()

   # LayoutTensor approach
   shared = LayoutTensor[dtype, Layout.row_major(TPB), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()
   ```

2. **Memory access**: Same syntax

   ```mojo
   # Raw approach
   shared[local_i] = a[global_i]

   # LayoutTensor approach
   shared[local_i] = a[global_i]
   ```

3. **Safety features**:

   - Type safety
   - Layout management
   - Memory alignment handling

> **Note**: LayoutTensor handles memory layout, but you still need to manage thread synchronization with `barrier()` when using shared memory.

**Educational Note**: In this specific puzzle, the `barrier()` isn't strictly necessary since each thread only accesses its own shared memory location. However, it's included to teach proper shared memory synchronization patterns for more complex scenarios where threads need to coordinate access to shared data.

### Running the code

To test your solution, run the following command in your terminal:

  
    pixi NVIDIA (default)
    pixi AMD
    pixi Apple
    uv
  
  

```bash
pixi run p08_layout_tensor
```

  
  

```bash
pixi run -e amd p08_layout_tensor
```

  
  

```bash
pixi run -e apple p08_layout_tensor
```

  
  

```bash
uv run poe p08_layout_tensor
```

  

Your output will look like this if the puzzle isn't solved yet:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0])
```

### Reference implementation (example)


```mojo
fn add_10_shared_layout_tensor
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

    if global_i < size:
        shared[local_i] = a[global_i]

    # Note: barrier is not strictly needed here since each thread only accesses its own shared memory location.
    # However, it's included to teach proper shared memory synchronization patterns
    # for more complex scenarios where threads need to coordinate access to shared data.
    # For this specific puzzle, we can remove the barrier since each thread only accesses its own shared memory location.
    barrier()

    if global_i < size:
        output[global_i] = shared[local_i] + 10

```

This solution demonstrates how LayoutTensor simplifies shared memory usage while maintaining performance:

1. **Memory hierarchy with LayoutTensor**
   - Global tensors: `a` and `output` (slow, visible to all blocks)
   - Shared tensor: `shared` (fast, thread-block local)
   - Example for 8 elements with 4 threads per block:

     ```txt
     Global tensor a: [1 1 1 1 | 1 1 1 1]  # Input: all ones

     Block (0):         Block (1):
     shared[0..3]       shared[0..3]
     [1 1 1 1]          [1 1 1 1]
     ```

2. **Thread coordination**
   - Load phase with natural indexing:

     ```txt
     Thread 0: shared[0] = a[0]=1    Thread 2: shared[2] = a[2]=1
     Thread 1: shared[1] = a[1]=1    Thread 3: shared[3] = a[3]=1
     barrier()                                 # Wait for all loads
     ```

   - Process phase: Each thread adds 10 to its shared tensor value
   - Result: `output[global_i] = shared[local_i] + 10 = 11`

   **Note**: In this specific case, the `barrier()` isn't strictly necessary since each thread only writes to and reads from its own shared memory location (`shared[local_i]`). However, it's included for educational purposes to demonstrate proper shared memory synchronization patterns that are essential when threads need to access each other's data.

3. **LayoutTensor benefits**
   - Shared memory allocation:

     ```txt
     # Clean LayoutTensor API with address_space
     shared = LayoutTensor[dtype, Layout.row_major(TPB), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()
     ```

   - Natural indexing for both global and shared:

     ```txt
     Block 0 output: [11 11 11 11]
     Block 1 output: [11 11 11 11]
     ```

   - Built-in layout management and type safety

4. **Memory access pattern**
   - Load: Global tensor  Shared tensor (optimized)
   - Sync: Same `barrier()` requirement as raw version
   - Process: Add 10 to shared values
   - Store: Write 11s back to global tensor

This pattern shows how LayoutTensor maintains the performance benefits of shared memory while providing a more ergonomic API and built-in features.
