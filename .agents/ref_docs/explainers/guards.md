---
title: "Guards"
description: "{{ youtube YFKutZbRYSM breakpoint-lg }}"
---

# Guards

{{ youtube YFKutZbRYSM breakpoint-lg }}

{{ youtube YFKutZbRYSM breakpoint-lg }}

## Overview

Implement a kernel that adds 10 to each position of vector `a` and stores it in vector `output`.

**Note**: _You have more threads than positions. This means you need to protect against out-of-bounds memory access._

{{ youtube YFKutZbRYSM breakpoint-sm }}

## Key concepts

This puzzle covers:

- Handling thread/data size mismatches
- Preventing out-of-bounds memory access
- Using conditional execution in GPU kernels
- Safe memory access patterns

### Mathematical description

For each thread \\(i\\):
\\[\Large \text{if}\\ i < \text{size}: output[i] = a[i] + 10\\]

### Memory safety pattern

```txt
Thread 0 (i=0):  if 0 < size:  output[0] = a[0] + 10  OK Valid
Thread 1 (i=1):  if 1 < size:  output[1] = a[1] + 10  OK Valid
Thread 2 (i=2):  if 2 < size:  output[2] = a[2] + 10  OK Valid
Thread 3 (i=3):  if 3 < size:  output[3] = a[3] + 10  OK Valid
Thread 4 (i=4):  if 4 < size:  X Skip (out of bounds)
Thread 5 (i=5):  if 5 < size:  X Skip (out of bounds)
```

 **Note**: Boundary checking becomes increasingly complex with:

- Multi-dimensional arrays
- Different array shapes
- Complex access patterns

## Running the code

To test your solution, run the following command in your terminal:

  
    pixi NVIDIA (default)
    pixi AMD
    pixi Apple
    uv
  
  

```bash
pixi run p03
```

  
  

```bash
pixi run -e amd p03
```

  
  

```bash
pixi run -e apple p03
```

  
  

```bash
uv run poe p03
```

  

Your output will look like this if the puzzle isn't solved yet:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([10.0, 11.0, 12.0, 13.0])
```

## Reference implementation (example)


```mojo
fn add_10_guard(
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    size: UInt,
):
    i = thread_idx.x
    if i < size:
        output[i] = a[i] + 10.0

```

This solution:

- Gets thread index with `i = thread_idx.x`
- Guards against out-of-bounds access with `if i < size`
- Inside guard: adds 10 to input value

> You might wonder why it passes the test even without the bound-check!
> Always remember that passing the tests doesn't necessarily mean the code
> is sound and free of Undefined Behavoirs. In puzzle 10 we'll examine such cases and use some tools to catch such
> soundness bugs.

### Looking ahead

While simple boundary checks work here, consider these challenges:

- What about 2D/3D array boundaries?
- How to handle different shapes efficiently?
- What if we need padding or edge handling?

Example of growing complexity:

```mojo
# Current: 1D bounds check
if i < size: ...

# Coming soon: 2D bounds check
if i < height and j < width: ...

# Later: 3D with padding
if i < height and j < width and k < depth and
   i >= padding and j >= padding: ...
```

These boundary handling patterns will become more elegant when we learn about LayoutTensor in Puzzle 4, which provides built-in shape management.
