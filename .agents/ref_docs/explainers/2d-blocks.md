---
title: "2D Blocks"
description: "Implement a kernel that adds 10 to each position of matrix `a` and stores it in `output`."
---

# 2D Blocks

Implement a kernel that adds 10 to each position of matrix `a` and stores it in `output`.

## Overview

Implement a kernel that adds 10 to each position of matrix `a` and stores it in `output`.

**Note:**_You have fewer threads per block than the size of `a` in both directions._

## Key concepts

- Block-based processing
- Grid-block coordination
- Multi-block indexing
- Memory access patterns

>  **2D thread indexing convention**
>
> We extend the block-based indexing from puzzle 4 to 2D:
>
> ```txt
> Global position calculation:
> row = block_dim.y * block_idx.y + thread_idx.y
> col = block_dim.x * block_idx.x + thread_idx.x
> ```
>
> For example, with 22 blocks in a 44 grid:
> ```txt
> Block (0,0):   Block (1,0):
> [0,0  0,1]     [0,2  0,3]
> [1,0  1,1]     [1,2  1,3]
>
> Block (0,1):   Block (1,1):
> [2,0  2,1]     [2,2  2,3]
> [3,0  3,1]     [3,2  3,3]
> ```
>
> Each position shows (row, col) for that thread's global index.
> The block dimensions and indices work together to ensure:
> - Continuous coverage of the 2D space
> - No overlap between blocks
> - Efficient memory access patterns

## Implementation approaches

###  Raw memory approach
Learn how to handle multi-block operations with manual indexing.

### [ LayoutTensor Version](#layouttensor-version)
Use LayoutTensor features to elegantly handle block-based processing.

 **Note**: See how LayoutTensor simplifies block coordination and memory access patterns.

### Overview

Implement a kernel that adds 10 to each position of matrix `a` and stores it in `output`.

**Note:**_You have fewer threads per block than the size of `a` in both directions._

### Key concepts

In this puzzle, you'll learn about:

- Working with 2D block and thread arrangements
- Handling matrix data larger than block size
- Converting between 2D and linear memory access

The key insight is understanding how to coordinate multiple blocks of threads to process a 2D matrix that's larger than a single block's dimensions.

### Configuration

- **Matrix size**: \\(5 \times 5\\) elements
- **2D blocks**: Each block processes a \\(3 \times 3\\) region
- **Grid layout**: Blocks arranged in \\(2 \times 2\\) grid
- **Total threads**: \\(36\\) for \\(25\\) elements
- **Memory pattern**: Row-major storage for 2D data
- **Coverage**: Ensuring all matrix elements are processed

### Running the code

To test your solution, run the following command in your terminal:

  
    pixi NVIDIA (default)
    pixi AMD
    pixi Apple
    uv
  
  

```bash
pixi run p07
```

  
  

```bash
pixi run -e amd p07
```

  
  

```bash
pixi run -e apple p07
```

  
  

```bash
uv run poe p07
```

  

Your output will look like this if the puzzle isn't solved yet:

```txt
out: HostBuffer([0.0, 0.0, 0.0, ... , 0.0])
expected: HostBuffer([10.0, 11.0, 12.0, ... , 34.0])
```

### Reference implementation (example)


```mojo
fn add_10_blocks_2d(
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    size: UInt,
):
    row = block_dim.y * block_idx.y + thread_idx.y
    col = block_dim.x * block_idx.x + thread_idx.x
    if row < size and col < size:
        output[row * size + col] = a[row * size + col] + 10.0

```

This solution demonstrates key concepts of 2D block-based processing with raw memory:

1. **2D thread indexing**
   - Global row: `block_dim.y * block_idx.y + thread_idx.y`
   - Global col: `block_dim.x * block_idx.x + thread_idx.x`
   - Maps thread grid to matrix elements:

     ```txt
     5x5 matrix with 3x3 blocks:

     Block (0,0)         Block (1,0)
     [(0,0) (0,1) (0,2)] [(0,3) (0,4)    *  ]
     [(1,0) (1,1) (1,2)] [(1,3) (1,4)    *  ]
     [(2,0) (2,1) (2,2)] [(2,3) (2,4)    *  ]

     Block (0,1)         Block (1,1)
     [(3,0) (3,1) (3,2)] [(3,3) (3,4)    *  ]
     [(4,0) (4,1) (4,2)] [(4,3) (4,4)    *  ]
     [  *     *     *  ] [  *     *      *  ]
     ```

     (* = thread exists but outside matrix bounds)

2. **Memory layout**
   - Row-major linear memory: `index = row * size + col`
   - Example for 55 matrix:

     ```txt
     2D indices:    Linear memory:
     (2,1) -> 11   [00 01 02 03 04]
                   [05 06 07 08 09]
                   [10 11 12 13 14]
                   [15 16 17 18 19]
                   [20 21 22 23 24]
     ```

3. **Bounds checking**
   - Guard `row < size and col < size` handles:
     - Excess threads in partial blocks
     - Edge cases at matrix boundaries
     - 22 block grid with 33 threads each = 36 threads for 25 elements

4. **Block coordination**
   - Each 33 block processes part of 55 matrix
   - 22 grid of blocks ensures full coverage
   - Overlapping threads handled by bounds check
   - Efficient parallel processing across blocks

This pattern shows how to handle 2D data larger than block size while maintaining efficient memory access and thread coordination.

## LayoutTensor Version

### Overview

Implement a kernel that adds 10 to each position of 2D LayoutTensor `a` and stores it in 2D LayoutTensor `output`.

**Note:**_You have fewer threads per block than the size of `a` in both directions._

### Key concepts

In this puzzle, you'll learn about:

- Using `LayoutTensor` with multiple blocks
- Handling large matrices with 2D block organization
- Combining block indexing with `LayoutTensor` access

The key insight is that `LayoutTensor` simplifies 2D indexing while still requiring proper block coordination for large matrices.

### Configuration

- **Matrix size**: \\(5 \times 5\\) elements
- **Layout handling**: `LayoutTensor` manages row-major organization
- **Block coordination**: Multiple blocks cover the full matrix
- **2D indexing**: Natural \\((i,j)\\) access with bounds checking
- **Total threads**: \\(36\\) for \\(25\\) elements
- **Thread mapping**: Each thread processes one matrix element

### Running the code

To test your solution, run the following command in your terminal:

  
    pixi NVIDIA (default)
    pixi AMD
    pixi Apple
    uv
  
  

```bash
pixi run p07_layout_tensor
```

  
  

```bash
pixi run -e amd p07_layout_tensor
```

  
  

```bash
pixi run -e apple p07_layout_tensor
```

  
  

```bash
uv run poe p07_layout_tensor
```

  

Your output will look like this if the puzzle isn't solved yet:

```txt
out: HostBuffer([0.0, 0.0, 0.0, ... , 0.0])
expected: HostBuffer([10.0, 11.0, 12.0, ... , 34.0])
```

### Reference implementation (example)


```mojo
fn add_10_blocks_2d
    out_layout: Layout,
    a_layout: Layout,
:
    row = block_dim.y * block_idx.y + thread_idx.y
    col = block_dim.x * block_idx.x + thread_idx.x
    if row < size and col < size:
        output[row, col] = a[row, col] + 10.0

```

This solution demonstrates how LayoutTensor simplifies 2D block-based processing:

1. **2D thread indexing**
   - Global row: `block_dim.y * block_idx.y + thread_idx.y`
   - Global col: `block_dim.x * block_idx.x + thread_idx.x`
   - Maps thread grid to tensor elements:

     ```txt
     5x5 tensor with 3x3 blocks:

     Block (0,0)         Block (1,0)
     [(0,0) (0,1) (0,2)] [(0,3) (0,4)    *  ]
     [(1,0) (1,1) (1,2)] [(1,3) (1,4)    *  ]
     [(2,0) (2,1) (2,2)] [(2,3) (2,4)    *  ]

     Block (0,1)         Block (1,1)
     [(3,0) (3,1) (3,2)] [(3,3) (3,4)    *  ]
     [(4,0) (4,1) (4,2)] [(4,3) (4,4)    *  ]
     [  *     *     *  ] [  *     *      *  ]
     ```

     (* = thread exists but outside tensor bounds)

2. **LayoutTensor benefits**
   - Natural 2D indexing: `tensor[row, col]` instead of manual offset calculation
   - Automatic memory layout optimization
   - Example access pattern:

     ```txt
     Raw memory:         LayoutTensor:
     row * size + col    tensor[row, col]
     (2,1) -> 11        (2,1) -> same element
     ```

3. **Bounds checking**
   - Guard `row < size and col < size` handles:
     - Excess threads in partial blocks
     - Edge cases at tensor boundaries
     - Automatic memory layout handling by LayoutTensor
     - 36 threads (22 blocks of 33) for 25 elements

4. **Block coordination**
   - Each 33 block processes part of 55 tensor
   - LayoutTensor handles:
     - Memory layout optimization
     - Efficient access patterns
     - Block boundary coordination
     - Cache-friendly data access

This pattern shows how LayoutTensor simplifies 2D block processing while maintaining optimal memory access patterns and thread coordination.
