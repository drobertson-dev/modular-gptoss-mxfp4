---
title: "Map"
description: "{{ youtube rLhjprX8Nck breakpoint-lg }}"
---

# Map

{{ youtube rLhjprX8Nck breakpoint-lg }}

{{ youtube rLhjprX8Nck breakpoint-lg }}

## Overview

This puzzle introduces the fundamental concept of GPU parallelism: mapping individual threads to data elements for concurrent processing.
Example goal:  implement a kernel that adds 10 to each element of vector `a`, storing the results in vector `output`.

**Note:**_You have 1 thread per position._

{{ youtube rLhjprX8Nck breakpoint-sm }}

## Key concepts

- Basic GPU kernel structure
- One-to-one thread to data mapping
- Memory access patterns
- Array operations on GPU

For each position \\(i\\):
\\[\Large output[i] = a[i] + 10\\]

## What we cover

###  Raw Memory Approach

Start with direct memory manipulation to understand GPU fundamentals.

### [ Preview: Modern Approach with LayoutTensor](#2d-indexing-coming-in-later-puzzles)

See how LayoutTensor simplifies GPU programming with safer, cleaner code.

 **Tip**: Understanding both approaches leads to better appreciation of modern GPU programming patterns.

### Key concepts

In this puzzle, you'll learn about:

- Basic GPU kernel structure
- Thread indexing with `thread_idx.x`
- Simple parallel operations

- **Parallelism**: Each thread executes independently
- **Thread indexing**: Access element at position `i = thread_idx.x`
- **Memory access**: Read from `a[i]` and write to `output[i]`
- **Data independence**: Each output depends only on its corresponding input

### Running the code

To test your solution, run the following command in your terminal:

  
    pixi NVIDIA (default)
    pixi AMD
    pixi Apple
    uv
  
  

```bash
pixi run p01
```

  
  

```bash
pixi run -e amd p01
```

  
  

```bash
pixi run -e apple p01
```

  
  

```bash
uv run poe p01
```

  

Your output will look like this if the puzzle isn't solved yet:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([10.0, 11.0, 12.0, 13.0])
```

### Reference implementation (example)


```mojo
fn add_10(
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    i = thread_idx.x
    output[i] = a[i] + 10.0

```

This solution:

- Gets thread index with `i = thread_idx.x`
- Adds 10 to input value: `output[i] = a[i] + 10.0`

### Why consider LayoutTensor?

Looking at our traditional implementation below, you might notice some potential issues:

#### Current approach

```mojo
i = thread_idx.x
output[i] = a[i] + 10.0
```

This works for 1D arrays, but what happens when we need to:

- Handle 2D or 3D data?
- Deal with different memory layouts?
- Ensure coalesced memory access?

#### Preview of future challenges

As we progress through the puzzles, array indexing will become more complex:

```mojo
# 2D indexing coming in later puzzles
idx = row * WIDTH + col

# 3D indexing
idx = (batch * HEIGHT + row) * WIDTH + col

# With padding
idx = (batch * padded_height + row) * padded_width + col
```

#### LayoutTensor preview

[LayoutTensor](https://docs.modular.com/mojo/kernels/layout/layout_tensor/LayoutTensor/) will help us handle these cases more elegantly:

```mojo
# Future preview - don't worry about this syntax yet!
output[i, j] = a[i, j] + 10.0  # 2D indexing
output[b, i, j] = a[b, i, j] + 10.0  # 3D indexing
```

We'll learn about LayoutTensor in detail in Puzzle 4, where these concepts become essential. For now, focus on understanding:

- Basic thread indexing
- Simple memory access patterns
- One-to-one mapping of threads to data

 **Key Takeaway**: While direct indexing works for simple cases, we'll soon need more sophisticated tools for complex GPU programming patterns.
