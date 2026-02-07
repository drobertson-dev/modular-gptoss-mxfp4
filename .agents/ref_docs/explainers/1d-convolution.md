---
title: "1D Convolution"
description: "In signal processing and image analysis, convolution is a fundamental operation that combines two sequences to produce a third sequence. This puzzle challenges you to implement a 1D convolution on the GPU, where each output element is computed by sliding a kernel over an input array."
---

# 1D Convolution

In signal processing and image analysis, convolution is a fundamental operation that combines two sequences to produce a third sequence. This puzzle challenges you to implement a 1D convolution on the GPU, where each output element is computed by sliding a kernel over an input array.

> ## Moving to LayoutTensor
>
> So far in our GPU puzzle journey, we've been exploring two parallel approaches to GPU memory management:
>
> 1. Raw memory management with direct pointer manipulation using [UnsafePointer](https://docs.modular.com/mojo/stdlib/memory/unsafe_pointer/UnsafePointer/)
> 2. The more structured [LayoutTensor](https://docs.modular.com/mojo/kernels/layout/layout_tensor/LayoutTensor/) with its powerful address_space parameter for memory allocation
>
> Starting from this puzzle, we're transitioning exclusively to using `LayoutTensor`. This abstraction provides several benefits:
> - Type-safe memory access patterns
> - Clear representation of data layouts
> - Better code maintainability
> - Reduced chance of memory-related bugs
> - More expressive code that better represents the underlying computations
> - A lot more ... that we'll uncover gradually!
>
> This transition aligns with best practices in modern GPU programming in Mojo , where higher-level abstractions help manage complexity without sacrificing performance.

## Overview

In signal processing and image analysis, convolution is a fundamental operation that combines two sequences to produce a third sequence. This puzzle challenges you to implement a 1D convolution on the GPU, where each output element is computed by sliding a kernel over an input array.

Implement a kernel that computes a 1D convolution between vector `a` and vector `b` and stores it in `output` using the `LayoutTensor` abstraction.

**Note:**_You need to handle the general case. You only need 2 global reads and 1 global write per thread._

For those new to convolution, think of it as a weighted sliding window operation. At each position, we multiply the kernel values with the corresponding input values and sum the results. In mathematical notation, this is often written as:

\\[\Large output[i] = \sum_{j=0}^{\text{CONV}-1} a[i+j] \cdot b[j] \\]

In pseudocode, 1D convolution is:

```python
for i in range(SIZE):
    for j in range(CONV):
        if i + j < SIZE:
            ret[i] += a_host[i + j] * b_host[j]
```

This puzzle is split into two parts to help you build understanding progressively:

- [Simple Version with Single Block](#simple-case-with-single-block)
  Start here to learn the basics of implementing convolution with shared memory in a single block using LayoutTensor.

- [Block Boundary Version](#block-boundary-version)
  Then tackle the more challenging case where data needs to be shared across block boundaries, leveraging LayoutTensor's capabilities.

Each version presents unique challenges in terms of memory access patterns and thread coordination. The simple version helps you understand the basic convolution operation, while the complete version tests your ability to handle more complex scenarios that arise in real-world GPU programming.

## Simple Case with Single Block

Implement a kernel that computes a 1D convolution between 1D LayoutTensor `a` and 1D LayoutTensor `b` and stores it in 1D LayoutTensor `output`.

**Note:**_You need to handle the general case. You only need 2 global reads and 1 global write per thread._

### Key concepts

This puzzle covers:

- Implementing sliding window operations on GPUs
- Managing data dependencies across threads
- Using shared memory for overlapping regions

The key insight is understanding how to efficiently access overlapping elements while maintaining correct boundary conditions.

### Configuration

- Input array size: `SIZE = 6` elements
- Kernel size: `CONV = 3` elements
- Threads per block: `TPB = 8`
- Number of blocks: 1
- Shared memory: Two arrays of size `SIZE` and `CONV`

Notes:

- **Data loading**: Each thread loads one element from input and kernel
- **Memory pattern**: Shared arrays for input and convolution kernel
- **Thread sync**: Coordination before computation

### Reference implementation (example)


```mojo
fn conv_1d_simple
    in_layout: Layout, out_layout: Layout, conv_layout: Layout
:
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = Int(thread_idx.x)
    shared_a = LayoutTensor[
        dtype,
        Layout.row_major(SIZE),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    shared_b = LayoutTensor[
        dtype,
        Layout.row_major(CONV),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    if global_i < SIZE:
        shared_a[local_i] = a[global_i]

    if global_i < CONV:
        shared_b[local_i] = b[global_i]

    barrier()

    # Note: this is unsafe as it enforces no guard so could access `shared_a` beyond its bounds
    # local_sum = Scalardtype
    # for j in range(CONV):
    #     if local_i + j < SIZE:
    #         local_sum += shared_a[local_i + j] * shared_b[j]

    # if global_i < SIZE:
    #     out[global_i] = local_sum

    # Safe and correct:
    if global_i < SIZE:
        # Note: using `var` allows us to include the type in the type inference
        # `out.element_type` is available in LayoutTensor
        var local_sum: output.element_type = 0

        # Note: `@parameter` decorator unrolls the loop at compile time given `CONV` is a compile-time constant
        # See: https://docs.modular.com/mojo/manual/decorators/parameter/#parametric-for-statement
        @parameter
        for j in range(CONV):
            # Bonus: do we need this check for this specific example with fixed SIZE, CONV
            if local_i + j < SIZE:
                local_sum += shared_a[local_i + j] * shared_b[j]

        output[global_i] = local_sum

```

The solution implements a 1D convolution using shared memory for efficient access to overlapping elements. Here's a detailed breakdown:

#### Memory layout

```txt
Input array a:   [0  1  2  3  4  5]
Kernel b:        [0  1  2]
```

#### Computation steps

1. **Data Loading**:

   ```txt
   shared_a: [0  1  2  3  4  5]  // Input array
   shared_b: [0  1  2]           // Convolution kernel
   ```

2. **Convolution Process**for each position i:

   ```txt
   output[0] = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] = 0*0 + 1*1 + 2*2 = 5
   output[1] = a[1]*b[0] + a[2]*b[1] + a[3]*b[2] = 1*0 + 2*1 + 3*2 = 8
   output[2] = a[2]*b[0] + a[3]*b[1] + a[4]*b[2] = 2*0 + 3*1 + 4*2 = 11
   output[3] = a[3]*b[0] + a[4]*b[1] + a[5]*b[2] = 3*0 + 4*1 + 5*2 = 14
   output[4] = a[4]*b[0] + a[5]*b[1] + 0*b[2]    = 4*0 + 5*1 + 0*2 = 5
   output[5] = a[5]*b[0] + 0*b[1]   + 0*b[2]     = 5*0 + 0*1 + 0*2 = 0
   ```

#### Implementation details

1. **Thread Participation and Efficiency Considerations**:
   - The inefficient approach without proper thread guard:

     ```mojo
     # Inefficient version - all threads compute even when results won't be used
     local_sum = Scalardtype
     for j in range(CONV):
         if local_i + j < SIZE:
             local_sum += shared_a[local_i + j] * shared_b[j]
     # Only guard the final write
     if global_i < SIZE:
         output[global_i] = local_sum
     ```

   - The efficient and correct implementation:

     ```mojo
     if global_i < SIZE:
         var local_sum: output.element_type = 0  # Using var allows type inference
         @parameter  # Unrolls loop at compile time since CONV is constant
         for j in range(CONV):
             if local_i + j < SIZE:
                 local_sum += shared_a[local_i + j] * shared_b[j]
         output[global_i] = local_sum
     ```

   The key difference is that the inefficient version has **all threads perform the convolution computation**(including those where `global_i >= SIZE`), and only the final write is guarded. This leads to:
   - **Wasteful computation**: Threads beyond the valid range still perform unnecessary work
   - **Reduced efficiency**: Extra computations that won't be used
   - **Poor resource utilization**: GPU cores working on meaningless calculations

   The efficient version ensures that only threads with valid `global_i` values perform any computation, making better use of GPU resources.

2. **Key Implementation Features**:
   - Uses `var` for proper type inference with `output.element_type`
   - Employs `@parameter` decorator to unroll the convolution loop at compile time
   - Maintains strict bounds checking for memory safety
   - Leverages LayoutTensor's type system for better code safety

3. **Memory Management**:
   - Uses shared memory for both input array and kernel
   - Single load per thread from global memory
   - Efficient reuse of loaded data

4. **Thread Coordination**:
   - `barrier()` ensures all data is loaded before computation
   - Each thread computes one output element
   - Maintains coalesced memory access pattern

5. **Performance Optimizations**:
   - Minimizes global memory access
   - Uses shared memory for fast data access
   - Avoids thread divergence in main computation loop
   - Loop unrolling through `@parameter` decorator

## Block Boundary Version

Implement a kernel that computes a 1D convolution between 1D LayoutTensor `a` and 1D LayoutTensor `b` and stores it in 1D LayoutTensor `output`.

**Note:**_You need to handle the general case. You only need 2 global reads and 1 global write per thread._

### Configuration

- Input array size: `SIZE_2 = 15` elements
- Kernel size: `CONV_2 = 4` elements
- Threads per block: `TPB = 8`
- Number of blocks: 2
- Shared memory: `TPB + CONV_2 - 1` elements for input

Notes:

- **Extended loading**: Account for boundary overlap
- **Block edges**: Handle data across block boundaries
- **Memory layout**: Efficient shared memory usage
- **Synchronization**: Proper thread coordination

### Reference implementation (example)


```mojo
fn conv_1d_block_boundary
    in_layout: Layout, out_layout: Layout, conv_layout: Layout, dtype: DType
:
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    local_i = Int(thread_idx.x)
    # first: need to account for padding
    shared_a = LayoutTensor[
        dtype,
        Layout.row_major(TPB + CONV_2 - 1),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    shared_b = LayoutTensor[
        dtype,
        Layout.row_major(CONV_2),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    if global_i < SIZE_2:
        shared_a[local_i] = a[global_i]
    else:
        shared_a[local_i] = 0

    # second: load elements needed for convolution at block boundary
    if local_i < CONV_2 - 1:
        # indices from next block
        next_idx = global_i + TPB
        if next_idx < SIZE_2:
            shared_a[TPB + local_i] = a[next_idx]
        else:
            # Initialize out-of-bounds elements to 0 to avoid reading from uninitialized memory
            # which is an undefined behavior
            shared_a[TPB + local_i] = 0

    if local_i < CONV_2:
        shared_b[local_i] = b[local_i]

    barrier()

    if global_i < SIZE_2:
        var local_sum: output.element_type = 0

        @parameter
        for j in range(CONV_2):
            if global_i + j < SIZE_2:
                local_sum += shared_a[local_i + j] * shared_b[j]

        output[global_i] = local_sum

```

The solution handles block boundary cases in 1D convolution using extended shared memory. Here's a detailed analysis:

#### Memory layout and sizing

```txt
Test Configuration:
- Full array size: SIZE_2 = 15 elements
- Grid: 2 blocks x 8 threads
- Convolution kernel: CONV_2 = 4 elements

Block 0 shared memory:  [0 1 2 3 4 5 6 7|8 9 10]  // TPB(8) + (CONV_2-1)(3) padding
Block 1 shared memory:  [8 9 10 11 12 13 14 0|0 0 0]  // Second block. data(7) + padding to fill grid(1) + (CONV_2-1)(3) padding

Size calculation:
- Main data: TPB elements (8)
- Overlap: CONV_2 - 1 elements (4 - 1 = 3)
- Total: TPB + CONV_2 - 1 = 8 + 4 - 1 = 11 elements
```

#### Implementation details

1. **Shared Memory Allocation**:

   ```mojo
   # First: account for padding needed for convolution window
   shared_a = LayoutTensor[dtype, Layout.row_major(TPB + CONV_2 - 1), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()
   shared_b = LayoutTensor[dtype, Layout.row_major(CONV_2), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()
   ```

   This allocation pattern ensures we have enough space for both the block's data and the overlap region.

2. **Data Loading Strategy**:

   ```mojo
   # Main block data
   if global_i < SIZE_2:
       shared_a[local_i] = a[global_i]
   else:
       shared_a[local_i] = 0

   # Boundary data from next block
   if local_i < CONV_2 - 1:
       next_idx = global_i + TPB
       if next_idx < SIZE_2:
           shared_a[TPB + local_i] = a[next_idx]
       else:
           # Initialize out-of-bounds elements to 0 to avoid reading from uninitialized memory
           # which is an undefined behavior
           shared_a[TPB + local_i] = 0
   ```

   - Only threads with `local_i < CONV_2 - 1` load boundary data
   - Prevents unnecessary thread divergence
   - Maintains memory coalescing for main data load
   - Explicitly zeroes out-of-bounds elements to avoid undefined behavior

3. **Kernel Loading**:

   ```mojo
   if local_i < b_size:
       shared_b[local_i] = b[local_i]
   ```

   - Single load per thread
   - Bounded by kernel size

4. **Convolution Computation**:

   ```mojo
   if global_i < SIZE_2:
       var local_sum: output.element_type = 0
       @parameter
       for j in range(CONV_2):
           if global_i + j < SIZE_2:
               local_sum += shared_a[local_i + j] * shared_b[j]
   ```

   - Uses `@parameter` for compile-time loop unrolling
   - Proper type inference with `output.element_type`
   - Semantically correct bounds check: only compute convolution for valid input positions

#### Memory access pattern analysis

1. **Block 0 Access Pattern**:

   ```txt
   Thread 0: [0 1 2 3] x [0 1 2 3]
   Thread 1: [1 2 3 4] x [0 1 2 3]
   Thread 2: [2 3 4 5] x [0 1 2 3]
   ...
   Thread 7: [7 8 9 10] x [0 1 2 3]  // Uses overlap data
   ```

2. **Block 1 Access Pattern**:
Note how starting from thread 4, `global_i + j < SIZE_2` evaluates to `False` and hence iterations are skipped.
   ```txt
   Thread 0: [8  9 10 11] x [0 1 2 3]
   Thread 1: [9 10 11 12] x [0 1 2 3]
   ...
   Thread 4: [12 13 14] x [0 1 2]       // Zero padding at end
   Thread 5: [13 14]    x [0 1]
   Thread 6: [14]       x [0]
   Thread 7: skipped                    // global_i + j < SIZE_2 evaluates to false for all j, no computation
   ```

#### Performance optimizations

1. **Memory Coalescing**:
   - Main data load: Consecutive threads access consecutive memory
   - Boundary data: Only necessary threads participate
   - Single barrier synchronization point

2. **Thread Divergence Minimization**:
   - Clean separation of main and boundary loading
   - Uniform computation pattern within warps
   - Efficient bounds checking

3. **Shared Memory Usage**:
   - Optimal sizing to handle block boundaries
   - No bank conflicts in access pattern
   - Efficient reuse of loaded data

4. **Boundary Handling**:
   - Explicit zero initialization for out-of-bounds elements which prevents reading from uninitialized shared memory
   - Semantically correct boundary checking using `global_i + j < SIZE_2` instead of shared memory bounds
   - Proper handling of edge cases without over-computation

#### Boundary condition improvement

The solution uses `if global_i + j < SIZE_2:` rather than checking shared memory bounds. This pattern is:

- **Mathematically correct**: Only computes convolution where input data actually exists
- **More efficient**: Avoids unnecessary computations for positions beyond the input array
- **Safer**: Prevents reliance on zero-padding behavior in shared memory

This implementation achieves efficient cross-block convolution while maintaining:

- Memory safety through proper bounds checking
- High performance through optimized memory access
- Clean code structure using LayoutTensor abstractions
- Minimal synchronization overhead
- Mathematically sound boundary handling
