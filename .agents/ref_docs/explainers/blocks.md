---
title: "Blocks"
description: "Implement a kernel that adds 10 to each position of vector `a` and stores it in `output`."
---

# Blocks

Implement a kernel that adds 10 to each position of vector `a` and stores it in `output`.

## Overview

Implement a kernel that adds 10 to each position of vector `a` and stores it in `output`.

**Note:**_You have fewer threads per block than the size of a._

## Key concepts

This puzzle covers:

- Processing data larger than thread block size
- Coordinating multiple blocks of threads
- Computing global thread positions

The key insight is understanding how blocks of threads work together to process data that's larger than a single block's capacity, while maintaining correct element-to-thread mapping.

## Running the code

To test your solution, run the following command in your terminal:

  
    pixi NVIDIA (default)
    pixi AMD
    pixi Apple
    uv
  
  

```bash
pixi run p06
```

  
  

```bash
pixi run -e amd p06
```

  
  

```bash
pixi run -e apple p06
```

  
  

```bash
uv run poe p06
```

  

Your output will look like this if the puzzle isn't solved yet:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0])
```

## Reference implementation (example)


```mojo
fn add_10_blocks(
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    size: UInt,
):
    i = block_dim.x * block_idx.x + thread_idx.x
    if i < size:
        output[i] = a[i] + 10.0

```

This solution covers key concepts of block-based GPU processing:

1. **Global thread indexing**
   - Combines block and thread indices: `block_dim.x * block_idx.x + thread_idx.x`
   - Maps each thread to a unique global position
   - Example for 3 threads per block:

     ```txt
     Block 0: [0 1 2]
     Block 1: [3 4 5]
     Block 2: [6 7 8]
     ```

2. **Block coordination**
   - Each block processes a contiguous chunk of data
   - Block size (3) < Data size (9) requires multiple blocks
   - Automatic work distribution across blocks:

     ```txt
     Data:    [0 1 2 3 4 5 6 7 8]
     Block 0: [0 1 2]
     Block 1:       [3 4 5]
     Block 2:             [6 7 8]
     ```

3. **Bounds checking**
   - Guard condition `i < size` handles edge cases
   - Prevents out-of-bounds access when size isn't perfectly divisible by block size
   - Essential for handling partial blocks at the end of data

4. **Memory access pattern**
   - Coalesced memory access: threads in a block access contiguous memory
   - Each thread processes one element: `output[i] = a[i] + 10.0`
   - Block-level parallelism provides efficient memory bandwidth utilization

This pattern forms the foundation for processing large datasets that exceed the size of a single thread block.
