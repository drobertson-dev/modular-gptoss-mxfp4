---
title: "Broadcast"
description: "Implement a kernel that broadcast adds vector `a` and vector `b` and stores it in 2D matrix `output`."
---

# Broadcast

Implement a kernel that broadcast adds vector `a` and vector `b` and stores it in 2D matrix `output`.

## Overview

Implement a kernel that broadcast adds vector `a` and vector `b` and stores it in 2D matrix `output`.

**Note:**_You have more threads than positions._

## Key concepts
- Broadcasting vectors to matrix
- 2D thread management
- Mixed dimension operations
- Memory layout patterns

## Implementation approaches

###  Raw memory approach
Learn how to handle broadcasting with manual memory indexing.

### [ LayoutTensor Version](#layouttensor-version)
Use LayoutTensor to handle mixed-dimension operations.

 **Note**: Notice how LayoutTensor simplifies broadcasting compared to manual indexing.

### Overview

Implement a kernel that broadcast adds vector `a` and vector `b` and stores it in 2D matrix `output`.

**Note:**_You have more threads than positions._

### Key concepts

In this puzzle, you'll learn about:

- Broadcasting 1D vectors across different dimensions
- Using 2D thread indices for broadcast operations
- Handling boundary conditions in broadcast patterns

The key insight is understanding how to map elements from two 1D vectors to create a 2D output matrix through broadcasting, while handling thread bounds correctly.

- **Broadcasting**: Each element of `a` combines with each element of `b`
- **Thread mapping**: 2D thread grid \\((3 \times 3)\\) for \\(2 \times 2\\) output
- **Vector access**: Different access patterns for `a` and `b`
- **Bounds checking**: Guard against threads outside matrix dimensions

### Running the code

To test your solution, run the following command in your terminal:

  
    pixi NVIDIA (default)
    pixi AMD
    pixi Apple
    uv
  
  

```bash
pixi run p05
```

  
  

```bash
pixi run -e amd p05
```

  
  

```bash
pixi run -e apple p05
```

  
  

```bash
uv run poe p05
```

  

Your output will look like this if the puzzle isn't solved yet:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([1.0, 2.0, 11.0, 12.0])
```

### Reference implementation (example)


```mojo
fn broadcast_add(
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    size: UInt,
):
    row = thread_idx.y
    col = thread_idx.x
    if row < size and col < size:
        output[row * size + col] = a[col] + b[row]

```

This solution demonstrates fundamental GPU broadcasting concepts without LayoutTensor abstraction:

1. **Thread to matrix mapping**
   - Uses `thread_idx.y` for row access and `thread_idx.x` for column access
   - Direct mapping from 2D thread grid to output matrix elements
   - Handles excess threads (33 grid) for 22 output matrix

2. **Broadcasting mechanics**
   - Vector `a` broadcasts horizontally: same `a[col]` used across each row
   - Vector `b` broadcasts vertically: same `b[row]` used across each column
   - Output combines both vectors through addition

   ```txt
   [ a0 a1 ]  +  [ b0 ]  =  [ a0+b0  a1+b0 ]
                 [ b1 ]     [ a0+b1  a1+b1 ]
   ```

3. **Bounds checking**
   - Single guard condition `row < size and col < size` handles both dimensions
   - Prevents out-of-bounds access for both input vectors and output matrix
   - Required due to 33 thread grid being larger than 22 data

Compare this with the LayoutTensor version to see how the abstraction simplifies broadcasting operations while maintaining the same underlying concepts.

## LayoutTensor Version

### Overview

Implement a kernel that broadcast adds 1D LayoutTensor `a` and 1D LayoutTensor `b` and stores it in 2D LayoutTensor `output`.

**Note:**_You have more threads than positions._

### Key concepts

In this puzzle, you'll learn about:

- Using `LayoutTensor` for broadcast operations
- Working with different tensor shapes
- Handling 2D indexing with `LayoutTensor`

The key insight is that `LayoutTensor` allows natural broadcasting through different tensor shapes: \\((1, n)\\) and \\((n, 1)\\) to \\((n,n)\\), while still requiring bounds checking.

- **Tensor shapes**: Input vectors have shapes \\((1, n)\\) and \\((n, 1)\\)
- **Broadcasting**: Output combines both dimensions to \\((n,n)\\)
- **Guard condition**: Still need bounds checking for output size
- **Thread bounds**: More threads \\((3 \times 3)\\) than tensor elements \\((2 \times 2)\\)

### Running the code

To test your solution, run the following command in your terminal:

  
    pixi NVIDIA (default)
    pixi AMD
    pixi Apple
    uv
  
  

```bash
pixi run p05_layout_tensor
```

  
  

```bash
pixi run -e amd p05_layout_tensor
```

  
  

```bash
pixi run -e apple p05_layout_tensor
```

  
  

```bash
uv run poe p05_layout_tensor
```

  

Your output will look like this if the puzzle isn't solved yet:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([1.0, 2.0, 11.0, 12.0])
```

### Reference implementation (example)


```mojo
fn broadcast_add
    out_layout: Layout,
    a_layout: Layout,
    b_layout: Layout,
:
    row = thread_idx.y
    col = thread_idx.x
    if row < size and col < size:
        output[row, col] = a[0, col] + b[row, 0]

```

This solution demonstrates key concepts of LayoutTensor broadcasting and GPU thread mapping:

1. **Thread to matrix mapping**

   - Uses `thread_idx.y` for row access and `thread_idx.x` for column access
   - Natural 2D indexing matches the output matrix structure
   - Excess threads (33 grid) are handled by bounds checking

2. **Broadcasting mechanics**
   - Input `a` has shape `(1,n)`: `a[0,col]` broadcasts across rows
   - Input `b` has shape `(n,1)`: `b[row,0]` broadcasts across columns
   - Output has shape `(n,n)`: Each element is sum of corresponding broadcasts

   ```txt
   [ a0 a1 ]  +  [ b0 ]  =  [ a0+b0  a1+b0 ]
                 [ b1 ]     [ a0+b1  a1+b1 ]
   ```

3. **Bounds Checking**
   - Guard condition `row < size and col < size` prevents out-of-bounds access
   - Handles both matrix bounds and excess threads efficiently
   - No need for separate checks for `a` and `b` due to broadcasting

This pattern forms the foundation for more complex tensor operations we'll explore in later puzzles.
