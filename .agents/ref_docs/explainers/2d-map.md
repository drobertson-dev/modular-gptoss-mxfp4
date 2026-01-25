---
title: "2D Map"
description: "{{ youtube EjmBmwgdAT0 breakpoint-lg }}"
---

# 2D Map

{{ youtube EjmBmwgdAT0 breakpoint-lg }}

{{ youtube EjmBmwgdAT0 breakpoint-lg }}

## Overview
Implement a kernel that adds 10 to each position of 2D square matrix `a` and stores it in 2D square matrix `output`.

**Note:**_You have more threads than positions_.

{{ youtube EjmBmwgdAT0 breakpoint-sm }}

## Key concepts
- 2D thread indexing
- Matrix operations on GPU
- Handling excess threads
- Memory layout patterns

For each position \\((i,j)\\):
\\[\Large output[i,j] = a[i,j] + 10\\]

> ## Thread indexing convention
>
> When working with 2D matrices in GPU programming, we follow a natural mapping between thread indices and matrix coordinates:
> - `thread_idx.y` corresponds to the row index
> - `thread_idx.x` corresponds to the column index
> 
> 
>
> This convention aligns with:
>
> 1. The standard mathematical notation where matrix positions are specified as (row, column)
> 2. The visual representation of matrices where rows go top-to-bottom (y-axis) and columns go left-to-right (x-axis)
> 3. Common GPU programming patterns where thread blocks are organized in a 2D grid matching the matrix structure
>
> ### Historical origins
>
> While graphics and image processing typically use \\((x,y)\\) coordinates, matrix operations in computing have historically used (row, column) indexing. This comes from how early computers stored and processed 2D data: line by line, top to bottom, with each line read left to right. This row-major memory layout proved efficient for both CPUs and GPUs, as it matches how they access memory sequentially. When GPU programming adopted thread blocks for parallel processing, it was natural to map `thread_idx.y` to rows and `thread_idx.x` to columns, maintaining consistency with established matrix indexing conventions.

## Implementation approaches

###  Raw memory approach
Learn how 2D indexing works with manual memory management.

### [ Learn about LayoutTensor](#introduction-to-layouttensor)
Discover a powerful abstraction that simplifies multi-dimensional array operations and memory management on GPU.

### [ Modern 2D operations](#layouttensor-version)
Put LayoutTensor into practice with natural 2D indexing and automatic bounds checking.

 **Note**: From this puzzle onward, we'll primarily use LayoutTensor for cleaner, safer GPU code.

### Overview

Implement a kernel that adds 10 to each position of 2D square matrix `a` and stores it in 2D square matrix `output`.

**Note:**_You have more threads than positions_.

### Key concepts

In this puzzle, you'll learn about:

- Working with 2D thread indices (`thread_idx.x`, `thread_idx.y`)
- Converting 2D coordinates to 1D memory indices
- Handling boundary checks in two dimensions

The key insight is understanding how to map from 2D thread coordinates \\((i,j)\\) to elements in a row-major matrix of size \\(n \times n\\), while ensuring thread indices are within bounds.

- **2D indexing**: Each thread has a unique \\((i,j)\\) position
- **Memory layout**: Row-major ordering maps 2D to 1D memory
- **Guard condition**: Need bounds checking in both dimensions
- **Thread bounds**: More threads \\((3 \times 3)\\) than matrix elements \\((2 \times 2)\\)

### Running the code

To test your solution, run the following command in your terminal:

  
    pixi NVIDIA (default)
    pixi AMD
    pixi Apple
    uv
  
  

```bash
pixi run p04
```

  
  

```bash
pixi run -e amd p04
```

  
  

```bash
pixi run -e apple p04
```

  
  

```bash
uv run poe p04
```

  

Your output will look like this if the puzzle isn't solved yet:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([10.0, 11.0, 12.0, 13.0])
```

### Reference implementation (example)


```mojo
fn add_10_2d(
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    size: UInt,
):
    row = thread_idx.y
    col = thread_idx.x
    if row < size and col < size:
        output[row * size + col] = a[row * size + col] + 10.0

```

This solution:

1. Get 2D indices:  `row = thread_idx.y`, `col = thread_idx.x`
2. Add guard: `if row < size and col < size`
3. Inside guard: `output[row * size + col] = a[row * size + col] + 10.0`

## Introduction to LayoutTensor

Let's take a quick break from solving puzzles to preview a powerful abstraction that will make our GPU programming journey more enjoyable:
 ... the **[LayoutTensor](https://docs.modular.com/mojo/kernels/layout/layout_tensor/LayoutTensor/)**.

>  _This is a motivational overview of LayoutTensor's capabilities. Don't worry about understanding everything now - we'll explore each feature in depth as we progress through the puzzles_.

### Example scenario: Growing complexity

Let's look at the challenges we've faced so far:

```mojo
# Puzzle 1: Simple indexing
output[i] = a[i] + 10.0

# Puzzle 2: Multiple array management
output[i] = a[i] + b[i]

# Puzzle 3: Bounds checking
if i < size:
    output[i] = a[i] + 10.0
```

As dimensions grow, code becomes more complex:

```mojo
# Traditional 2D indexing for row-major 2D matrix
idx = row * WIDTH + col
if row < height and col < width:
    output[idx] = a[idx] + 10.0
```

### The solution: A peek at LayoutTensor

LayoutTensor will help us tackle these challenges with elegant solutions. Here's a glimpse of what's coming:

1. **Natural Indexing**: Use `tensor[i, j]` instead of manual offset calculations
3. **Flexible Memory Layouts**: Support for row-major, column-major, and tiled organizations
4. **Performance Optimization**: Efficient memory access patterns for GPU

### A taste of what's ahead

Let's look at a few examples of what LayoutTensor can do. Don't worry about understanding all the details now - we'll cover each feature thoroughly in upcoming puzzles.

#### Basic usage example

```mojo
from layout import Layout, LayoutTensor

# Define layout
comptime HEIGHT = 2
comptime WIDTH = 3
comptime layout = Layout.row_major(HEIGHT, WIDTH)

# Create tensor
tensor = LayoutTensordtype, layout)

# Access elements naturally
tensor[0, 0] = 1.0  # First element
tensor[1, 2] = 2.0  # Last element
```

To learn more about `Layout` and `LayoutTensor`, see these guides from the [Mojo manual](https://docs.modular.com/mojo/manual/)

- [Introduction to layouts](https://docs.modular.com/mojo/manual/layout/layouts)
- [Using LayoutTensor](https://docs.modular.com/mojo/manual/layout/tensors)

### Quick example

Let's put everything together with a simple example that demonstrates the basics of LayoutTensor:

```mojo
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor

comptime HEIGHT = 2
comptime WIDTH = 3
comptime dtype = DType.float32
comptime layout = Layout.row_major(HEIGHT, WIDTH)

fn kernel
    dtype: DType, layout: Layout
:
    print("Before:")
    print(tensor)
    tensor[0, 0] += 1
    print("After:")
    print(tensor)

def main():
    ctx = DeviceContext()

    a = ctx.enqueue_create_bufferdtype
    a.enqueue_fill(0)
    tensor = LayoutTensordtype, layout, MutAnyOrigin
    # Note: since `tensor` is a device tensor we can't print it without the kernel wrapper
    ctx.enqueue_function[kernel[dtype, layout], kernel[dtype, layout]](
        tensor, grid_dim=1, block_dim=1
    )

    ctx.synchronize()

```

When we run this code with:

  
    pixi NVIDIA (default)
    pixi AMD
    pixi Apple
    uv
  
  

```bash
pixi run layout_tensor_intro
```

  
  

```bash
pixi run -e amd layout_tensor_intro
```

  
  

```bash
pixi run -e apple layout_tensor_intro
```

  
  

```bash
uv run poe layout_tensor_intro
```

  

```txt
Before:
0.0 0.0 0.0
0.0 0.0 0.0
After:
1.0 0.0 0.0
0.0 0.0 0.0
```

Let's break down what's happening:

1. We create a `2 x 3` tensor with row-major layout
2. Initially, all elements are zero
3. Using natural indexing, we modify a single element
4. The change is reflected in our output

This simple example demonstrates key LayoutTensor benefits:

- Clean syntax for tensor creation and access
- Automatic memory layout handling
- Natural multi-dimensional indexing

While this example is straightforward, the same patterns will scale to complex GPU operations in upcoming puzzles. You'll see how these basic concepts extend to:

- Multi-threaded GPU operations
- Shared memory optimizations
- Complex tiling strategies
- Hardware-accelerated computations

Ready to start your GPU programming journey with LayoutTensor? Let's dive into the puzzles!

 **Tip**: Keep this example in mind as we progress - we'll build upon these fundamental concepts to create increasingly sophisticated GPU programs.

## LayoutTensor Version

### Overview

Implement a kernel that adds 10 to each position of 2D _LayoutTensor_ `a` and stores it in 2D _LayoutTensor_ `output`.

**Note:**_You have more threads than positions_.

### Key concepts

In this puzzle, you'll learn about:

- Using `LayoutTensor` for 2D array access
- Direct 2D indexing with `tensor[i, j]`
- Handling bounds checking with `LayoutTensor`

The key insight is that `LayoutTensor` provides a natural 2D indexing interface, abstracting away the underlying memory layout while still requiring bounds checking.

- **2D access**: Natural \\((i,j)\\) indexing with `LayoutTensor`
- **Memory abstraction**: No manual row-major calculation needed
- **Guard condition**: Still need bounds checking in both dimensions
- **Thread bounds**: More threads \\((3 \times 3)\\) than tensor elements \\((2 \times 2)\\)

### Running the code

To test your solution, run the following command in your terminal:

  
    pixi NVIDIA (default)
    pixi AMD
    pixi Apple
    uv
  
  

```bash
pixi run p04_layout_tensor
```

  
  

```bash
pixi run -e amd p04_layout_tensor
```

  
  

```bash
pixi run -e apple p04_layout_tensor
```

  
  

```bash
uv run poe p04_layout_tensor
```

  

Your output will look like this if the puzzle isn't solved yet:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([10.0, 11.0, 12.0, 13.0])
```

### Reference implementation (example)


```mojo
fn add_10_2d(
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    a: LayoutTensor[dtype, layout, MutAnyOrigin],
    size: UInt,
):
    row = thread_idx.y
    col = thread_idx.x
    if col < size and row < size:
        output[row, col] = a[row, col] + 10.0

```

This solution:

- Gets 2D thread indices with `row = thread_idx.y`, `col = thread_idx.x`
- Guards against out-of-bounds with `if row < size and col < size`
- Uses `LayoutTensor`'s 2D indexing: `output[row, col] = a[row, col] + 10.0`
