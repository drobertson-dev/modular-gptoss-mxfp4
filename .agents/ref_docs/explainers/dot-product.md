---
title: "Dot Product"
description: "Implement a kernel that computes the dot product of vector `a` and vector `b` and stores it in `output` (single number). The dot product is an operation that takes two vectors of the same size and returns a single number (a scalar). It is calculated by multiplying corresponding elements from each vector and then summing those products."
---

# Dot Product

Implement a kernel that computes the dot product of vector `a` and vector `b` and stores it in `output` (single number). The dot product is an operation that takes two vectors of the same size and returns a single number (a scalar). It is calculated by multiplying corresponding elements from each vector and then summing those products.

## Overview

Implement a kernel that computes the dot product of vector `a` and vector `b` and stores it in `output` (single number).  The dot product is an operation that takes two vectors of the same size and returns a single number (a scalar). It is calculated by multiplying corresponding elements from each vector and then summing those products.

For example, if you have two vectors:

\\[a = [a_{1}, a_{2}, ..., a_{n}] \\]
\\[b = [b_{1}, b_{2}, ..., b_{n}] \\]

Their dot product is:
\\[a \\cdot b = a_{1}b_{1} +  a_{2}b_{2} + ... + a_{n}b_{n}\\]

**Note:**_You have 1 thread per position. You only need 2 global reads per thread and 1 global write per thread block._

## Implementation approaches

###  Raw memory approach
Learn how to implement the reduction with manual memory management and synchronization.

###  LayoutTensor Version
Use LayoutTensor's features for efficient reduction and shared memory management.

 **Note**: See how LayoutTensor simplifies efficient memory access patterns.

### Overview

Implement a kernel that computes the dot product of vector `a` and vector `b` and stores it in `output` (single number).

**Note:**_You have 1 thread per position. You only need 2 global reads per thread and 1 global write per thread block._

### Key concepts

This puzzle covers:

- Implementing parallel reduction operations
- Using shared memory for intermediate results
- Coordinating threads for collective operations

The key insight is understanding how to efficiently combine multiple values into a single result using parallel computation and shared memory.

### Configuration

- Vector size: `SIZE = 8` elements
- Threads per block: `TPB = 8`
- Number of blocks: 1
- Output size: 1 element
- Shared memory: `TPB` elements

Notes:

- **Element access**: Each thread reads corresponding elements from `a` and `b`
- **Partial results**: Computing and storing intermediate values
- **Thread coordination**: Synchronizing before combining results
- **Final reduction**: Converting partial results to scalar output

_Note: For this problem, you don't need to worry about number of shared reads. We will
handle that challenge later._

### Running the code

To test your solution, run the following command in your terminal:

  
    pixi NVIDIA (default)
    pixi AMD
    pixi Apple
    uv
  
  

```bash
pixi run p12
```

  
  

```bash
pixi run -e amd p12
```

  
  

```bash
pixi run -e apple p12
```

  
  

```bash
uv run poe p12
```

  

Your output will look like this if the puzzle isn't solved yet:

```txt
out: HostBuffer([0.0])
expected: HostBuffer([140.0])
```

### Reference implementation (example)


```mojo
fn dot_product(
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
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
        shared[local_i] = a[global_i] * b[global_i]

    barrier()

    # The following causes race condition: all threads writing to the same location
    # out[0] += shared[local_i]

    # Instead can do parallel reduction in shared memory as opposed to
    # global memory which has no guarantee on synchronization.
    # Loops using global memory can cause thread divergence because
    # fundamentally GPUs execute threads in warps (groups of 32 threads typically)
    # and warps can be scheduled independently.
    # However, shared memory does not have such issues as long as we use `barrier()`
    # correctly when we're in the same thread block.
    stride = UInt(TPB // 2)
    while stride > 0:
        if local_i < stride:
            shared[local_i] += shared[local_i + stride]

        barrier()
        stride //= 2

    # only thread 0 writes the final result
    if local_i == 0:
        output[0] = shared[0]

```

The solution implements a parallel reduction algorithm for dot product computation using shared memory. Here's a detailed breakdown:

#### Phase 1: Element-wise Multiplication

Each thread performs one multiplication:

```txt
Thread i: shared[i] = a[i] * b[i]
```

#### Phase 2: Parallel Reduction

The reduction uses a tree-based approach that halves active threads in each step:

```txt
Initial:  [0*0  1*1  2*2  3*3  4*4  5*5  6*6  7*7]
        = [0    1    4    9    16   25   36   49]

Step 1:   [0+16 1+25 4+36 9+49  16   25   36   49]
        = [16   26   40   58   16   25   36   49]

Step 2:   [16+40 26+58 40   58   16   25   36   49]
        = [56   84   40   58   16   25   36   49]

Step 3:   [56+84  84   40   58   16   25   36   49]
        = [140   84   40   58   16   25   36   49]
```

#### Key implementation features

1. **Memory Access Pattern**:
   - Each thread loads exactly two values from global memory (`a[i]`, `b[i]`)
   - Uses shared memory for intermediate results
   - Final result written once to global memory

2. **Thread Synchronization**:
   - `barrier()` after initial multiplication
   - `barrier()` after each reduction step
   - Prevents race conditions between reduction steps

3. **Reduction Logic**:

   ```mojo
   stride = TPB // 2
   while stride > 0:
       if local_i < stride:
           shared[local_i] += shared[local_i + stride]
       barrier()
       stride //= 2
   ```

   - Halves stride in each step
   - Only active threads perform additions
   - Maintains work efficiency

4. **Performance Considerations**:
   - \\(\log_2(n)\\) steps for \\(n\\) elements
   - Coalesced memory access pattern
   - Minimal thread divergence
   - Efficient use of shared memory

This implementation achieves \\(O(\log n)\\) time complexity compared to \\(O(n)\\) in sequential execution, demonstrating the power of parallel reduction algorithms.

#### Barrier synchronization importance

The `barrier()` between reduction steps is critical for correctness. Here's why:

Without `barrier()`, race conditions occur:

```text
Initial shared memory: [0 1 4 9 16 25 36 49]

Step 1 (stride = 4):
Thread 0 reads: shared[0] = 0, shared[4] = 16
Thread 1 reads: shared[1] = 1, shared[5] = 25
Thread 2 reads: shared[2] = 4, shared[6] = 36
Thread 3 reads: shared[3] = 9, shared[7] = 49

Without barrier:
- Thread 0 writes: shared[0] = 0 + 16 = 16
  and reads old value shared[0] = 0 instead of 16!
```

With `barrier()`:

```text
Step 1 (stride = 4):
All threads write their sums:
[16 26 40 58 16 25 36 49]
barrier() ensures ALL threads see these values

Step 2 (stride = 2):
Now threads safely read the updated values:
Thread 0: shared[0] = 16 + 40 = 56
Thread 1: shared[1] = 26 + 58 = 84
```

The `barrier()` ensures:

1. All writes from current step complete
2. All threads see updated values
3. No thread starts next iteration early
4. Consistent shared memory state

Without these synchronization points, we could get:

- Memory race conditions
- Threads reading stale values
- Non-deterministic results
- Incorrect final sum

### Overview

Implement a kernel that computes the dot product of 1D LayoutTensor `a` and 1D LayoutTensor `b` and stores it in 1D LayoutTensor `output` (single number).

**Note:**_You have 1 thread per position. You only need 2 global reads per thread and 1 global write per thread block._

### Key concepts

This puzzle covers:

- Similar to the puzzle 8 and puzzle 11, implementing parallel reduction with LayoutTensor
- Managing shared memory using LayoutTensor with address_space
- Coordinating threads for collective operations
- Using layout-aware tensor operations

The key insight is how LayoutTensor simplifies memory management while maintaining efficient parallel reduction patterns.

### Configuration

- Vector size: `SIZE = 8` elements
- Threads per block: `TPB = 8`
- Number of blocks: 1
- Output size: 1 element
- Shared memory: `TPB` elements

Notes:

- **LayoutTensor allocation**: Use `LayoutTensor[dtype, Layout.row_major(TPB), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()`
- **Element access**: Natural indexing with bounds checking
- **Layout handling**: Separate layouts for input and output
- **Thread coordination**: Same synchronization patterns with `barrier()`

### Running the code

To test your solution, run the following command in your terminal:

  
    pixi NVIDIA (default)
    pixi AMD
    pixi Apple
    uv
  
  

```bash
pixi run p12_layout_tensor
```

  
  

```bash
pixi run -e amd p12_layout_tensor
```

  
  

```bash
pixi run -e apple p12_layout_tensor
```

  
  

```bash
uv run poe p12_layout_tensor
```

  

Your output will look like this if the puzzle isn't solved yet:

```txt
out: HostBuffer([0.0])
expected: HostBuffer([140.0])
```

### Reference implementation (example)


```mojo
fn dot_product
    in_layout: Layout, out_layout: Layout
:
    shared = LayoutTensor[
        dtype,
        Layout.row_major(TPB),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x

    # Compute element-wise multiplication into shared memory
    if global_i < size:
        shared[local_i] = a[global_i] * b[global_i]

    # Synchronize threads within block
    barrier()

    # Parallel reduction in shared memory
    stride = UInt(TPB // 2)
    while stride > 0:
        if local_i < stride:
            shared[local_i] += shared[local_i + stride]

        barrier()
        stride //= 2

    # Only thread 0 writes the final result
    if local_i == 0:
        output[0] = shared[0]

```

The solution implements a parallel reduction for dot product using LayoutTensor. Here's the detailed breakdown:

#### Phase 1: Element-wise Multiplication

Each thread performs one multiplication with natural indexing:

```mojo
shared[local_i] = a[global_i] * b[global_i]
```

#### Phase 2: Parallel Reduction

Tree-based reduction with layout-aware operations:

```txt
Initial:  [0*0  1*1  2*2  3*3  4*4  5*5  6*6  7*7]
        = [0    1    4    9    16   25   36   49]

Step 1:   [0+16 1+25 4+36 9+49  16   25   36   49]
        = [16   26   40   58   16   25   36   49]

Step 2:   [16+40 26+58 40   58   16   25   36   49]
        = [56   84   40   58   16   25   36   49]

Step 3:   [56+84  84   40   58   16   25   36   49]
        = [140   84   40   58   16   25   36   49]
```

#### Key implementation features

1. **Memory Management**:
   - Clean shared memory allocation with LayoutTensor address_space parameter
   - Type-safe operations with LayoutTensor
   - Automatic bounds checking
   - Layout-aware indexing

2. **Thread Synchronization**:
   - `barrier()` after initial multiplication
   - `barrier()` between reduction steps
   - Safe thread coordination

3. **Reduction Logic**:

   ```mojo
   stride = TPB // 2
   while stride > 0:
       if local_i < stride:
           shared[local_i] += shared[local_i + stride]
       barrier()
       stride //= 2
   ```

4. **Performance Benefits**:
   - \\(O(\log n)\\) time complexity
   - Coalesced memory access
   - Minimal thread divergence
   - Efficient shared memory usage

The LayoutTensor version maintains the same efficient parallel reduction while providing:

- Better type safety
- Cleaner memory management
- Layout awareness
- Natural indexing syntax

#### Barrier synchronization importance

The `barrier()` between reduction steps is critical for correctness. Here's why:

Without `barrier()`, race conditions occur:

```text
Initial shared memory: [0 1 4 9 16 25 36 49]

Step 1 (stride = 4):
Thread 0 reads: shared[0] = 0, shared[4] = 16
Thread 1 reads: shared[1] = 1, shared[5] = 25
Thread 2 reads: shared[2] = 4, shared[6] = 36
Thread 3 reads: shared[3] = 9, shared[7] = 49

Without barrier:
- Thread 0 writes: shared[0] = 0 + 16 = 16
  and reads old value shared[0] = 0 instead of 16!
```

With `barrier()`:

```text
Step 1 (stride = 4):
All threads write their sums:
[16 26 40 58 16 25 36 49]
barrier() ensures ALL threads see these values

Step 2 (stride = 2):
Now threads safely read the updated values:
Thread 0: shared[0] = 16 + 40 = 56
Thread 1: shared[1] = 26 + 58 = 84
```

The `barrier()` ensures:

1. All writes from current step complete
2. All threads see updated values
3. No thread starts next iteration early
4. Consistent shared memory state

Without these synchronization points, we could get:

- Memory race conditions
- Threads reading stale values
- Non-deterministic results
- Incorrect final sum
