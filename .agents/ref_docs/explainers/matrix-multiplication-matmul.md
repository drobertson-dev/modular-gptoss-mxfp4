---
title: "Matrix Multiplication (MatMul)"
description: "Matrix multiplication is a fundamental operation in scientific computing, machine learning, and graphics. Given two matrices \\(A\\) and \\(B\\), we want to compute their product \\(C = A \\times B.\\)"
---

# Matrix Multiplication (MatMul)

Matrix multiplication is a fundamental operation in scientific computing, machine learning, and graphics. Given two matrices \\(A\\) and \\(B\\), we want to compute their product \\(C = A \\times B.\\)

## Overview

Matrix multiplication is a fundamental operation in scientific computing, machine learning, and graphics. Given two matrices \\(A\\) and \\(B\\), we want to compute their product \\(C = A \\times B.\\)

For matrices \\(A_{m\\times k}\\) and \\(B_{k\\times n}\\), each element of the result \\(C_{m\\times n}\\) is computed as:

\\[\Large C_{ij} = \sum_{l=0}^{k-1} A_{il} \\cdot B_{lj} \\]

This puzzle explores different approaches to implementing matrix multiplication on GPUs, each with its own performance characteristics:

- [Naive Version](#nave-matrix-multiplication)
  The straightforward implementation where each thread computes one element of the output matrix. While simple to understand, this approach makes many redundant memory accesses.

- [Shared Memory Version](#shared-memory-matrix-multiplication)
  Improves performance by loading blocks of input matrices into fast shared memory, reducing global memory accesses. Each thread still computes one output element but reads from shared memory.

- [Tiled Version](#tiled-matrix-multiplication)
  Further optimizes by dividing the computation into tiles, allowing threads to cooperate on loading and computing blocks of the output matrix. This approach better utilizes memory hierarchy and thread cooperation.

Each version builds upon the previous one, introducing new optimization techniques common in GPU programming. You'll learn how different memory access patterns and thread cooperation strategies affect performance.

The progression illustrates a common pattern in GPU optimization:
1. Start with a correct but naive implementation
2. Reduce global memory access with shared memory
3. Improve data locality and thread cooperation with tiling
4. Use high-level abstractions while maintaining performance

Choose a version to begin your matrix multiplication journey!

## Nave Matrix Multiplication

### Overview

Implement a kernel that multiplies square matrices \\(A\\) and \\(B\\) and stores the result in \\(\text{output}\\).
This is the most straightforward implementation where each thread computes one element of the output matrix.

### Key concepts

This puzzle covers:

- 2D thread organization for matrix operations
- Global memory access patterns
- Matrix indexing in row-major layout
- Thread-to-output element mapping

The key insight is understanding how to map 2D thread indices to matrix elements and compute dot products in parallel.

### Configuration

- Matrix size: \\(\\text{SIZE} \\times \\text{SIZE} = 2 \\times 2\\)
- Threads per block: \\(\\text{TPB} \\times \\text{TPB} = 3 \\times 3\\)
- Grid dimensions: \\(1 \\times 1\\)

Layout configuration:

- Input A: `Layout.row_major(SIZE, SIZE)`
- Input B: `Layout.row_major(SIZE, SIZE)`
- Output: `Layout.row_major(SIZE, SIZE)`

### Running the code

To test your solution, run the following command in your terminal:

  
    pixi NVIDIA (default)
    pixi AMD
    pixi Apple
    uv
  
  

```bash
pixi run p16 --naive
```

  
  

```bash
pixi run -e amd p16 --naive
```

  
  

```bash
pixi run -e apple p16 --naive
```

  
  

```bash
uv run poe p16 --naive
```

  

Your output will look like this if the puzzle isn't solved yet:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([4.0, 6.0, 12.0, 22.0])
```

### Reference implementation (example)


```mojo
fn naive_matmul
    layout: Layout, size: UInt
:
    row = block_dim.y * block_idx.y + thread_idx.y
    col = block_dim.x * block_idx.x + thread_idx.x

    if row < size and col < size:
        var acc: output.element_type = 0

        @parameter
        for k in range(size):
            acc += a[row, k] * b[k, col]

        output[row, col] = acc

```

The naive matrix multiplication using LayoutTensor follows this basic approach:

#### Matrix layout (22 example)

```txt
Matrix A:          Matrix B:                   Output C:
[a[0,0] a[0,1]]    [b[0,0] b[0,1]]             [c[0,0] c[0,1]]
[a[1,0] a[1,1]]    [b[1,0] b[1,1]]             [c[1,0] c[1,1]]
```

#### Implementation details

1. **Thread mapping**:

   ```mojo
   row = block_dim.y * block_idx.y + thread_idx.y
   col = block_dim.x * block_idx.x + thread_idx.x
   ```

2. **Memory access pattern**:
   - Direct 2D indexing: `a[row, k]`
   - Transposed access: `b[k, col]`
   - Output writing: `output[row, col]`

3. **Computation flow**:

   ```mojo
   # Use var for mutable accumulator with tensor's element type
   var acc: output.element_type = 0

   # @parameter for compile-time loop unrolling
   @parameter
   for k in range(size):
       acc += a[row, k] * b[k, col]
   ```

#### Key language features

1. **Variable declaration**:
   - The use of `var` in `var acc: output.element_type = 0` allows for type inference with `output.element_type` ensures type compatibility with the output tensor
   - Initialized to zero before accumulation

2. **Loop pptimization**:
   - [`@parameter`](https://docs.modular.com/mojo/manual/decorators/parameter/#parametric-for-statement) decorator unrolls the loop at compile time
   - Improves performance for small, known matrix sizes
   - Enables better instruction scheduling

#### Performance characteristics

1. **Memory access**:
   - Each thread makes `2 x SIZE` global memory reads
   - One global memory write per thread
   - No data reuse between threads

2. **Computational efficiency**:
   - Simple implementation but suboptimal performance
   - Many redundant global memory accesses
   - No use of fast shared memory

3. **Limitations**:
   - High global memory bandwidth usage
   - Poor data locality
   - Limited scalability for large matrices

This naive implementation serves as a baseline for understanding matrix multiplication on GPUs, highlighting the need for optimization in memory access patterns.

## Understanding GPU Performance: The Roofline Model

Having implemented the naive matrix multiplication, you might be wondering: *How well is our kernel actually performing?* Is it limited by the GPU's computational power, or is something else holding it back?

The **roofline model**is your compass for GPU optimizationit reveals which hardware bottleneck limits your kernel's performance and guides you toward the most impactful optimizations. Rather than guessing at improvements, the roofline model shows you exactly where to focus your efforts.

### 1. Two ceilings for every GPU kernel

Every GPU kernel operates under two fundamental constraints:

- **Compute ceiling** how quickly the cores can execute floating-point operations (peak FLOPs/s)
- **Memory ceiling** how quickly the memory system can feed those cores with data (peak bytes/s)

Understanding which ceiling constrains your kernel is crucial for optimization strategy. The roofline model visualizes this relationship by plotting two key metrics:

**X-axis: Arithmetic Intensity** How much computation you extract per byte of data

\\[\Large I = \frac{\text{Total FLOPs}}{\text{Total Bytes from Memory}} \quad [\text{FLOP/B}]\\]

**Y-axis: Sustained Performance** How fast your kernel actually runs

\\[\Large P_{\text{sustained}} = \frac{\text{Total FLOPs}}{\text{Elapsed Time}} \quad [\text{GFLOP/s}]\\]

Two "roofs" bound all achievable performance:

| Roof             | Equation                         | Meaning                                            |
| ---------------- | -------------------------------- | -------------------------------------------------- |
| **Memory roof**| \\(P = B_{\text{peak}} \cdot I\\) | Sloped line; performance limited by memory bandwidth |
| **Compute roof**| \\(P = P_{\text{peak}}\\)         | Horizontal line; performance limited by compute throughput |

The **critical intensity**

\\[\Large I^* = \frac{P_{\text{peak}}}{B_{\text{peak}}}\\]

marks where a kernel transitions from memory-bound (\\(I  I^* \\)).

### 2. Hardware example: NVIDIA A100 specifications

Let's ground this theory in concrete numbers using the NVIDIA A100:

**Peak FP32 throughput**
\\[\Large P_{\text{peak}} = 19.5 \text{ TFLOP/s} = 19{,}500 \text{ GFLOP/s}\\]

**Peak HBM2 bandwidth**
\\[\Large B_{\text{peak}} = 1{,}555 \text{ GB/s}\\]

**Critical intensity**
\\[\Large I^* = \frac{19{,}500}{1{,}555} \approx 12.5 \text{ FLOP/B}\\]

*Source: [NVIDIA A100 Tensor Core GPU Architecture](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)*

This means kernels with arithmetic intensity below 12.5 FLOP/B are memory-bound, while those above are compute-bound.

### 3. Visualizing our matrix multiplication implementations

The animation below shows how our puzzle implementations map onto the A100's roofline model:

!Roofline Model Visualization

The visualization demonstrates the optimization journey we'll take in this puzzle:

1. **Hardware constraints** The red memory roof and blue compute roof define performance limits
2. **Our starting point** The naive implementation (orange dot) sitting firmly on the memory roof
3. **Optimization target** The shared memory version (teal dot) with improved arithmetic intensity
4. **Ultimate goal** The golden arrow pointing toward the critical intensity where kernels become compute-bound

### 4. Analyzing our naive implementation

Let's examine why our naive kernel from the previous section performs as it does. For our \\(2 \times 2\\) matrix multiplication:

**Computation per output element**: \\(\text{SIZE} + (\text{SIZE}-1) = 3 \text{ FLOPs }\\)

 > Each element requires \\(\text{SIZE}\\) multiplications and \\(\text{SIZE} - 1\\) additions:
 > \\[C_{00} = A_{00} \cdot B_{00} + A_{01} \cdot B_{10}\\]
 > For \\(\text{SIZE} = 2\\) it is 2 multiplications + 1 addition = 3 FLOPs

**Memory accesses per output element**:
- Row from matrix A: \\(2 \times 4 = 8\\) bytes (FP32)
- Column from matrix B: \\(2 \times 4 = 8\\) bytes (FP32)
- Total: \\(16\\) bytes per output element

**Arithmetic intensity**:
\\[\Large I_{\text{naive}} = \frac{3 \text{ FLOPs}}{16 \text{ bytes}} = 0.1875 \text{ FLOP/B}\\]

This arithmetic intensity is far below the compute roof of an A100, indicating that our naive kernel is **severely memory-bound**.

\\[\Large I_{\text{naive}} = 0.1875 \ll I^* = 12.5\\]

**Expected performance**:
\\[\Large P \approx B_{\text{peak}} \times I_{\text{naive}} = 1{,}555 \times 0.1875 \approx 292 \text{ GFLOP/s}\\]

This represents only \\(\frac{292}{19{,}500} \approx 1.5\%\\) of the GPU's computational potential! The visualization clearly shows this as the yellow dot sitting squarely on the memory roofwe're nowhere near the compute ceiling.

### 5. The path forward: shared memory optimization

The roofline model reveals our optimization strategy: **increase arithmetic intensity**by reducing redundant memory accesses. This is exactly what the shared memory approach accomplishes:

**Shared memory benefits**:
- **Cooperative loading**: Threads work together to load matrix blocks into fast shared memory
- **Data reuse**: Each loaded element serves multiple computations
- **Reduced global memory traffic**: Fewer accesses to slow global memory

**Expected arithmetic intensity improvement**:
\\[\Large I_{\text{shared}} = \frac{12 \text{ FLOPs}}{32 \text{ bytes}} = 0.375 \text{ FLOP/B}\\]

While still memory-bound for our small \\(2 \times 2\\) case, this 2 improvement in arithmetic intensity scales dramatically for larger matrices where shared memory tiles can be reused many more times.

### 6. Optimization strategies revealed by the roofline

The roofline model not only diagnoses current performance but also illuminates optimization paths. Here are the key techniques we'll explore in later puzzles:

| Technique                       | Roofline effect                                               | Implementation approach                                        |
| ------------------------------- | ------------------------------------------------------------ | -------------------------------------------------------------- |
| **Shared memory tiling**|  Arithmetic intensity through data reuse                    | Cooperative loading, block-wise computation                    |
| **Register blocking**| Reduce memory traffic with register accumulation             | Loop unrolling with register variables                         |
| **Kernel fusion**| More FLOPs per byte by combining operations                   | Single kernel handling multiple computation stages             |
| **Memory coalescing**| Maximize effective bandwidth utilization                      | Structured access patterns, proper thread organization         |
| **Asynchronous memory copies**| Dedicated copy engine enables compute-memory overlap          | `copy_dram_to_sram_async` with computation overlap            |
| **Mixed precision**| Smaller data types reduce memory pressure                     | FP16/BF16 input with FP32 accumulation                        |

Each technique moves kernels along the rooflineeither up the memory roof (better bandwidth utilization) or rightward toward the compute roof (higher arithmetic intensity).

**Note on asynchronous operations**: Standard GPU memory loads (`ld.global`) are already asynchronous - warps continue executing until they need the loaded data. Specialized async copy instructions like `cp.async` (CUDA) or [copy_dram_to_sram_async](https://docs.modular.com/mojo/kernels/layout/layout_tensor/copy_dram_to_sram_async/) (Mojo) provide additional benefits by using dedicated copy engines, bypassing registers, and enabling better resource utilization rather than simply making synchronous operations asynchronous.

### 7. Beyond simple rooflines

**Multi-level memory**: Advanced rooflines include separate ceilings for L2 cache, shared memory, and register bandwidth to identify which memory hierarchy level constrains performance.

**Communication rooflines**: For multi-GPU applications, replace memory bandwidth with interconnect bandwidth (NVLink, InfiniBand) to analyze scaling efficiency.

**Specialized units**: Modern GPUs include tensor cores with their own performance characteristics, requiring specialized roofline analysis.

### 8. Using the roofline in practice

1. **Profile your kernel**: Use tools like Nsight Compute to measure actual FLOPs and memory traffic
2. **Plot the data point**: Calculate arithmetic intensity and sustained performance
3. **Identify the bottleneck**: Memory-bound kernels sit on the memory roof; compute-bound kernels approach the compute roof
4. **Choose optimizations**: Focus on bandwidth improvements for memory-bound kernels, algorithmic changes for compute-bound ones
5. **Measure and iterate**: Verify that optimizations move kernels in the expected direction

### Connection to our shared memory puzzle

In the next section, we'll implement the **shared memory optimization**that begins moving our kernel up the roofline. As the visualization shows, this takes us from the orange dot (naive) to the teal dot (shared memory)a clear performance improvement through better data reuse.

While our \\(2 \times 2\\) example won't reach the compute roof, you'll see how the same principles scale to larger matrices where shared memory becomes crucial for performance. The roofline model provides the theoretical foundation for understanding **why**shared memory helps and **how much**improvement to expect.

Understanding the roofline model transforms GPU optimization from guesswork into systematic engineering. Every optimization technique in this book can be understood through its effect on this simple but powerful performance model.

## Shared Memory Matrix Multiplication

### Overview

This puzzle implements matrix multiplication for square matrices \\(A\\) and \\(B\\), storing results in \\(\text{output}\\) while leveraging shared memory to optimize memory access patterns. The implementation preloads matrix blocks into shared memory before performing computations.

### Key concepts

This puzzle covers:

- Block-local memory management with LayoutTensor
- Thread synchronization patterns
- Memory access optimization using shared memory
- Collaborative data loading with 2D indexing
- Efficient use of LayoutTensor for matrix operations

The central concept involves utilizing fast shared memory through LayoutTensor to minimize costly global memory accesses.

### Configuration

- Matrix size: \\(\\text{SIZE} \\times \\text{SIZE} = 2 \\times 2\\)
- Threads per block: \\(\\text{TPB} \\times \\text{TPB} = 3 \\times 3\\)
- Grid dimensions: \\(1 \\times 1\\)

Layout configuration:

- Input A: `Layout.row_major(SIZE, SIZE)`
- Input B: `Layout.row_major(SIZE, SIZE)`
- Output: `Layout.row_major(SIZE, SIZE)`
- Shared Memory: Two `TPB  TPB` LayoutTensors

Memory organization:

```txt
Global Memory (LayoutTensor):          Shared Memory (LayoutTensor):
A[i,j]: Direct access                  a_shared[local_row, local_col]
B[i,j]: Direct access                  b_shared[local_row, local_col]
```

### Running the code

To test your solution, run the following command in your terminal:

  
    pixi NVIDIA (default)
    pixi AMD
    pixi Apple
    uv
  
  

```bash
pixi run p16 --single-block
```

  
  

```bash
pixi run -e amd p16 --single-block
```

  
  

```bash
pixi run -e apple p16 --single-block
```

  
  

```bash
uv run poe p16 --single-block
```

  

Your output will look like this if the puzzle isn't solved yet:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([4.0, 6.0, 12.0, 22.0])
```

### Reference implementation (example)


```mojo
fn single_block_matmul
    layout: Layout, size: UInt
:
    row = block_dim.y * block_idx.y + thread_idx.y
    col = block_dim.x * block_idx.x + thread_idx.x
    local_row = thread_idx.y
    local_col = thread_idx.x

    a_shared = LayoutTensor[
        dtype,
        Layout.row_major(TPB, TPB),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    b_shared = LayoutTensor[
        dtype,
        Layout.row_major(TPB, TPB),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    if row < size and col < size:
        a_shared[local_row, local_col] = a[row, col]
        b_shared[local_row, local_col] = b[row, col]

    barrier()

    if row < size and col < size:
        var acc: output.element_type = 0

        @parameter
        for k in range(size):
            acc += a_shared[local_row, k] * b_shared[k, local_col]

        output[row, col] = acc

```

The shared memory implementation with LayoutTensor improves performance through efficient memory access patterns:

#### Memory organization

```txt
Input Tensors (2x2):                Shared Memory (3x3):
Matrix A:                           a_shared:
 [a[0,0] a[0,1]]                     [s[0,0] s[0,1] s[0,2]]
 [a[1,0] a[1,1]]                     [s[1,0] s[1,1] s[1,2]]
                                     [s[2,0] s[2,1] s[2,2]]
Matrix B:                           b_shared: (similar layout)
 [b[0,0] b[0,1]]                     [t[0,0] t[0,1] t[0,2]]
 [b[1,0] b[1,1]]                     [t[1,0] t[1,1] t[1,2]]
                                     [t[2,0] t[2,1] t[2,2]]
```

#### Implementation phases

1. **Shared Memory Setup**:

   ```mojo
   # Create 2D shared memory tensors using LayoutTensor with address_space
   a_shared = LayoutTensor[dtype, Layout.row_major(TPB, TPB), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()
   b_shared = LayoutTensor[dtype, Layout.row_major(TPB, TPB), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()
   ```

2. **Thread Indexing**:

   ```mojo
   # Global indices for matrix access
   row = block_dim.y * block_idx.y + thread_idx.y
   col = block_dim.x * block_idx.x + thread_idx.x

   # Local indices for shared memory
   local_row = thread_idx.y
   local_col = thread_idx.x
   ```

3. **Data Loading**:

   ```mojo
   # Load data into shared memory using LayoutTensor indexing
   if row < size and col < size:
       a_shared[local_row, local_col] = a[row, col]
       b_shared[local_row, local_col] = b[row, col]
   ```

4. **Computation with Shared Memory**:

   ```mojo
   # Guard ensures we only compute for valid matrix elements
   if row < size and col < size:
       # Initialize accumulator with output tensor's type
       var acc: output.element_type = 0

       # Compile-time unrolled loop for matrix multiplication
       @parameter
       for k in range(size):
           acc += a_shared[local_row, k] * b_shared[k, local_col]

       # Write result only for threads within matrix bounds
       output[row, col] = acc
   ```

   Key aspects:
   - **Boundary check**: `if row < size and col < size`
     - Prevents out-of-bounds computation
     - Only valid threads perform work
     - Essential because TPB (33) > SIZE (22)

   - **Accumulator Type**: `var acc: output.element_type`
     - Uses output tensor's element type for type safety
     - Ensures consistent numeric precision
     - Initialized to zero before accumulation

   - **Loop Optimization**: `@parameter for k in range(size)`
     - Unrolls the loop at compile time
     - Enables better instruction scheduling
     - Efficient for small, known matrix sizes

   - **Result Writing**: `output[row, col] = acc`
     - Protected by the same guard condition
     - Only valid threads write results
     - Maintains matrix bounds safety

#### Thread safety and synchronization

1. **Guard conditions**:
   - Input Loading: `if row < size and col < size`
   - Computation: Same guard ensures thread safety
   - Output Writing: Protected by the same condition
   - Prevents invalid memory access and race conditions

2. **Memory access safety**:
   - Shared memory: Accessed only within TPB bounds
   - Global memory: Protected by size checks
   - Output: Guarded writes prevent corruption

#### Key language features

1. **LayoutTensor benefits**:
   - Direct 2D indexing simplifies code
   - Type safety through `element_type`
   - Efficient memory layout handling

2. **Shared memory allocation**:
   - LayoutTensor with address_space for structured allocation
   - Row-major layout matching input tensors
   - Proper alignment for efficient access

3. **Synchronization**:
   - `barrier()` ensures shared memory consistency
   - Proper synchronization between load and compute
   - Thread cooperation within block

#### Performance optimizations

1. **Memory Access Efficiency**:
   - Single global memory load per element
   - Multiple reuse through shared memory
   - Coalesced memory access patterns

2. **Thread cooperation**:
   - Collaborative data loading
   - Shared data reuse
   - Efficient thread synchronization

3. **Computational benefits**:
   - Reduced global memory traffic
   - Better cache utilization
   - Improved instruction throughput

This implementation significantly improves performance over the naive version by:

- Reducing global memory accesses
- Enabling data reuse through shared memory
- Using efficient 2D indexing with LayoutTensor
- Maintaining proper thread synchronization

## Tiled Matrix Multiplication

### Overview

Implement a kernel that multiplies square matrices \\(A\\) and \\(B\\) using tiled matrix multiplication with LayoutTensor. This approach handles large matrices by processing them in smaller chunks (tiles).

### Key concepts

- Matrix tiling with LayoutTensor for efficient computation
- Multi-block coordination with proper layouts
- Efficient shared memory usage through TensorBuilder
- Boundary handling for tiles with LayoutTensor indexing

### Configuration

- Matrix size: \\(\\text{SIZE\_TILED} = 9\\)
- Threads per block: \\(\\text{TPB} \times \\text{TPB} = 3 \times 3\\)
- Grid dimensions: \\(3 \times 3\\) blocks
- Shared memory: Two \\(\\text{TPB} \times \\text{TPB}\\) LayoutTensors per block

Layout configuration:

- Input A: `Layout.row_major(SIZE_TILED, SIZE_TILED)`
- Input B: `Layout.row_major(SIZE_TILED, SIZE_TILED)`
- Output: `Layout.row_major(SIZE_TILED, SIZE_TILED)`
- Shared Memory: Two `TPB  TPB` LayoutTensors using TensorBuilder

### Tiling strategy

#### Block organization

```txt
Grid Layout (3x3):           Thread Layout per Block (3x3):
[B00][B01][B02]               [T00 T01 T02]
[B10][B11][B12]               [T10 T11 T12]
[B20][B21][B22]               [T20 T21 T22]

Each block processes a tile using LayoutTensor indexing
```

#### Tile processing steps

1. Calculate global and local indices for thread position
2. Allocate shared memory for A and B tiles
3. For each tile:
   - Load tile from matrix A and B
   - Compute partial products
   - Accumulate results in registers
4. Write final accumulated result

#### Memory access pattern

```txt
Matrix A (8x8)                 Matrix B (8x8)               Matrix C (8x8)
+---+---+---+                  +---+---+---+                +---+---+---+
|T00|T01|T02| ...              |T00|T01|T02| ...            |T00|T01|T02| ...
+---+---+---+                  +---+---+---+                +---+---+---+
|T10|T11|T12|                  |T10|T11|T12|                |T10|T11|T12|
+---+---+---+                  +---+---+---+                +---+---+---+
|T20|T21|T22|                  |T20|T21|T22|                |T20|T21|T22|
+---+---+---+                  +---+---+---+                +---+---+---+
  ...                            ...                          ...

Tile Processing (for computing C[T11]):
1. Load tiles from A and B:
   +---+      +---+
   |A11| x    |B11|     For each phase k:
   +---+      +---+     C[T11] += A[row, k] x B[k, col]

2. Tile movement:
   Phase 1     Phase 2     Phase 3
   A: [T10]    A: [T11]    A: [T12]
   B: [T01]    B: [T11]    B: [T21]

3. Each thread (i,j) in tile computes:
   C[i,j] =  (A[i,k] x B[k,j]) for k in tile width

Synchronization required:
* After loading tiles to shared memory
* After computing each phase
```

### Running the code

To test your solution, run the following command in your terminal:

  
    pixi NVIDIA (default)
    pixi AMD
    pixi Apple
    uv
  
  

```bash
pixi run p16 --tiled
```

  
  

```bash
pixi run -e amd p16 --tiled
```

  
  

```bash
pixi run -e apple p16 --tiled
```

  
  

```bash
uv run poe p16 --tiled
```

  

Your output will look like this if the puzzle isn't solved yet:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([3672.0, 3744.0, 3816.0, 3888.0, 3960.0, 4032.0, 4104.0, 4176.0, 4248.0, 9504.0, 9738.0, 9972.0, 10206.0, 10440.0, 10674.0, 10908.0, 11142.0, 11376.0, 15336.0, 15732.0, 16128.0, 16524.0, 16920.0, 17316.0, 17712.0, 18108.0, 18504.0, 21168.0, 21726.0, 22284.0, 22842.0, 23400.0, 23958.0, 24516.0, 25074.0, 25632.0, 27000.0, 27720.0, 28440.0, 29160.0, 29880.0, 30600.0, 31320.0, 32040.0, 32760.0, 32832.0, 33714.0, 34596.0, 35478.0, 36360.0, 37242.0, 38124.0, 39006.0, 39888.0, 38664.0, 39708.0, 40752.0, 41796.0, 42840.0, 43884.0, 44928.0, 45972.0, 47016.0, 44496.0, 45702.0, 46908.0, 48114.0, 49320.0, 50526.0, 51732.0, 52938.0, 54144.0, 50328.0, 51696.0, 53064.0, 54432.0, 55800.0, 57168.0, 58536.0, 59904.0, 61272.0])
```

### Solution: Manual tiling


```mojo
fn matmul_tiled
    layout: Layout, size: UInt
:
    local_row = thread_idx.y
    local_col = thread_idx.x
    tiled_row = block_idx.y * TPB + local_row
    tiled_col = block_idx.x * TPB + local_col

    a_shared = LayoutTensor[
        dtype,
        Layout.row_major(TPB, TPB),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    b_shared = LayoutTensor[
        dtype,
        Layout.row_major(TPB, TPB),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var acc: output.element_type = 0

    # Iterate over tiles to compute matrix product
    @parameter
    for tile in range((size + TPB - 1) // TPB):
        # Load A tile - global row stays the same, col determined by tile
        if tiled_row < size and (tile * TPB + local_col) < size:
            a_shared[local_row, local_col] = a[
                tiled_row, tile * TPB + local_col
            ]

        # Load B tile - row determined by tile, global col stays the same
        if (tile * TPB + local_row) < size and tiled_col < size:
            b_shared[local_row, local_col] = b[
                tile * TPB + local_row, tiled_col
            ]

        barrier()

        # Matrix multiplication within the tile
        if tiled_row < size and tiled_col < size:

            @parameter
            for k in range(min(TPB, size - tile * TPB)):
                acc += a_shared[local_row, k] * b_shared[k, local_col]

        barrier()

    # Write out final result
    if tiled_row < size and tiled_col < size:
        output[tiled_row, tiled_col] = acc

```

The tiled matrix multiplication implementation demonstrates efficient handling of matrices \\((9 \times 9)\\) using small tiles \\((3 \times 3)\\). Here's how it works:

1. **Shared memory allocation**

   ```txt
   Input matrices (9x9) - Perfect fit for (3x3) tiling:
   A = [0  1  2  3  4  5  6  7  8 ]    B = [0  2  4  6  8  10 12 14 16]
       [9  10 11 12 13 14 15 16 17]        [18 20 22 24 26 28 30 32 34]
       [18 19 20 21 22 23 24 25 26]        [36 38 40 42 44 46 48 50 52]
       [27 28 29 30 31 32 33 34 35]        [54 56 58 60 62 64 66 68 70]
       [36 37 38 39 40 41 42 43 44]        [72 74 76 78 80 82 84 86 88]
       [45 46 47 48 49 50 51 52 53]        [90 92 94 96 98 100 102 104 106]
       [54 55 56 57 58 59 60 61 62]        [108 110 112 114 116 118 120 122 124]
       [63 64 65 66 67 68 69 70 71]        [126 128 130 132 134 136 138 140 142]
       [72 73 74 75 76 77 78 79 80]        [144 146 148 150 152 154 156 158 160]

   Shared memory per block (3x3):
   a_shared[TPB, TPB]  b_shared[TPB, TPB]
   ```

2. **Tile processing loop**

   ```txt
   Number of tiles = 9 // 3 = 3 tiles (perfect division!)

   For each tile:
   1. Load tile from A and B
   2. Compute partial products
   3. Accumulate in register
   ```

3. **Memory loading pattern**
   - With perfect \\((9 \times 9)\\) tiling, bounds check is technically unnecessary but included for defensive programming and consistency with other matrix sizes.

     ```mojo
        # Load A tile - global row stays the same, col determined by tile
        if tiled_row < size and (tile * TPB + local_col) < size:
            a_shared[local_row, local_col] = a[
                tiled_row, tile * TPB + local_col
            ]

        # Load B tile - row determined by tile, global col stays the same
        if (tile * TPB + local_row) < size and tiled_col < size:
            b_shared[local_row, local_col] = b[
                tile * TPB + local_row, tiled_col
            ]
     ```

4. **Computation within tile**

   ```mojo
   for k in range(min(TPB, size - tile * TPB)):
       acc += a_shared[local_row, k] * b_shared[k, local_col]
   ```

   - Avoids shared memory bank conflicts:

     ```txt
     Bank Conflict Free (Good):        Bank Conflicts (Bad):
     Thread0: a_shared[0,k] b_shared[k,0]  Thread0: a_shared[k,0] b_shared[0,k]
     Thread1: a_shared[0,k] b_shared[k,1]  Thread1: a_shared[k,0] b_shared[1,k]
     Thread2: a_shared[0,k] b_shared[k,2]  Thread2: a_shared[k,0] b_shared[2,k]
                                          
     Parallel access to different banks    Serialized access to same bank of b_shared
     (or broadcast for a_shared)           if shared memory was column-major
     ```

     **Shared memory bank conflicts explained:**
     - **Left (Good)**: `b_shared[k,threadIdx.x]` accesses different banks, `a_shared[0,k]` broadcasts to all threads
     - **Right (Bad)**: If b_shared were column-major, threads would access same bank simultaneously
     - **Key insight**: This is about shared memory access patterns, not global memory coalescing
     - **Bank structure**: Shared memory has 32 banks; conflicts occur when multiple threads access different addresses in the same bank simultaneously

5. **Synchronization points**

   ```txt
   barrier() after:
   1. Tile loading
   2. Tile computation
   ```

Key performance features:

- Processes \\((9 \times 9)\\) matrix using \\((3 \times 3)\\) tiles (perfect fit!)
- Uses shared memory for fast tile access
- Minimizes global memory transactions with coalesced memory access
- Optimized shared memory layout and access pattern to avoid shared memory bank conflicts

6. **Result writing**:

   ```mojo
   if tiled_row < size and tiled_col < size:
      output[tiled_row, tiled_col] = acc
   ```

   - Defensive bounds checking included for other matrix sizes and tiling strategies
   - Direct assignment to output matrix
   - All threads write valid results

#### Key optimizations

1. **Layout optimization**:
   - Row-major layout for all tensors
   - Efficient 2D indexing

2. **Memory access**:
   - Coalesced global memory loads
   - Efficient shared memory usage

3. **Computation**:
   - Register-based accumulation i.e. `var acc: output.element_type = 0`
   - Compile-time loop unrolling via `@parameter`

This implementation achieves high performance through:

- Efficient use of LayoutTensor for memory access
- Optimal tiling strategy
- Proper thread synchronization
- Careful boundary handling

### Solution: Idiomatic LayoutTensor tiling


```mojo
from gpu.memory import async_copy_wait_all
from layout.layout_tensor import copy_dram_to_sram_async

comptime NUM_THREADS = TPB * TPB
comptime BLOCK_DIM_COUNT = 2

fn matmul_idiomatic_tiled
    layout: Layout, size: UInt
:
    local_row = thread_idx.y
    local_col = thread_idx.x
    tiled_row = block_idx.y * TPB + local_row
    tiled_col = block_idx.x * TPB + local_col

    # Get the tile of the output matrix that this thread block is responsible for
    out_tile = output.tileTPB, TPB, Int(block_idx.x))
    a_shared = LayoutTensor[
        dtype,
        Layout.row_major(TPB, TPB),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    b_shared = LayoutTensor[
        dtype,
        Layout.row_major(TPB, TPB),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var acc: output.element_type = 0

    comptime load_a_layout = Layout.row_major(1, TPB)  # Coalesced loading
    comptime load_b_layout = Layout.row_major(1, TPB)  # Coalesced loading
    # Note: Both matrices stored in same orientation for correct matrix multiplication
    # Transposed loading would be useful if B were pre-transposed in global memory

    @parameter
    for idx in range(size // TPB):  # Perfect division: 9 // 3 = 3 tiles
        # Get tiles from A and B matrices
        a_tile = a.tileTPB, TPB, Int(idx))
        b_tile = b.tileTPB, TPB, Int(block_idx.x))

        # Asynchronously copy tiles to shared memory with consistent orientation
        copy_dram_to_sram_async
            thread_layout=load_a_layout,
            num_threads=NUM_THREADS,
            block_dim_count=BLOCK_DIM_COUNT,
        
        copy_dram_to_sram_async
            thread_layout=load_b_layout,
            num_threads=NUM_THREADS,
            block_dim_count=BLOCK_DIM_COUNT,
        

        # Wait for all async copies to complete
        async_copy_wait_all()
        barrier()

        # Compute partial matrix multiplication for this tile
        @parameter
        for k in range(TPB):
            acc += a_shared[local_row, k] * b_shared[k, local_col]

        barrier()

    # Write final result to output tile
    if tiled_row < size and tiled_col < size:
        out_tile[local_row, local_col] = acc

```

The idiomatic tiled matrix multiplication leverages Mojo's LayoutTensor API and asynchronous memory operations for a beautifully clean implementation.

** Key Point: This implementation performs standard matrix multiplication A  B using coalesced loading for both matrices.**

**What this implementation does:**

- **Matrix operation**: Standard \\(A \times B\\) multiplication (not \\(A \times B^T\\))
- **Loading pattern**: Both matrices use `Layout.row_major(1, TPB)` for coalesced access
- **Computation**: `acc += a_shared[local_row, k] * b_shared[k, local_col]`
- **Data layout**: No transposition during loading - both matrices loaded in same orientation

**What this implementation does NOT do:**

- Does NOT perform \\(A \times B^T\\) multiplication
- Does NOT use transposed loading patterns
- Does NOT transpose data during copy operations

With the \\((9 \times 9)\\) matrix size, we get perfect tiling that eliminates all boundary checks:

1. **LayoutTensor tile API**

   ```mojo
   out_tile = output.tileTPB, TPB
   a_tile = a.tileTPB, TPB
   b_tile = b.tileTPB, TPB
   ```

   This directly expresses "get the tile at position (block_idx.y, block_idx.x)" without manual coordinate calculation. See the [documentation](https://docs.modular.com/mojo/kernels/layout/layout_tensor/LayoutTensor/#tile) for more details.

2. **Asynchronous memory operations**

   ```mojo
   copy_dram_to_sram_async
      thread_layout = load_a_layout,
      num_threads = NUM_THREADS,
      block_dim_count = BLOCK_DIM_COUNT
   
   copy_dram_to_sram_async
      thread_layout = load_b_layout,
      num_threads = NUM_THREADS,
      block_dim_count = BLOCK_DIM_COUNT
   
   async_copy_wait_all()
   ```

   These operations:
   - Use dedicated copy engines that bypass registers and enable compute-memory overlap via [copy_dram_to_sram_async](https://docs.modular.com/mojo/kernels/layout/layout_tensor/copy_dram_to_sram_async/)
   - Use specialized thread layouts for optimal memory access patterns
   - Eliminate the need for manual memory initialization
   - **Important**:
      - Standard GPU loads are already asynchronous; these provide better resource utilization and register bypass
      - `copy_dram_to_sram_async` assumes that you are using a 1d thread block (`block_dim.y == block_dim.z == 1`) and all the threads from a thread block participate in the copy unless you specify otherwise.  This behaviour in overridden by specifying:
         - `block_dim_count`: the dimensionality of the thread block (`2` for the 2d thread block `THREADS_PER_BLOCK_TILED = (TPB, TPB)`)
         - `num_threads`: the number of threads in the thread block (`TPB*TPB == 9`)

3. **Optimized memory access layouts**

   ```mojo
   comptime load_a_layout = Layout.row_major(1, TPB)    # Coalesced loading
   comptime load_b_layout = Layout.row_major(1, TPB)    # Coalesced loading
   # Note: Both matrices use the same layout for standard A x B multiplication
   ```

   **Memory Access Analysis for Current Implementation:**

   Both matrices use `Layout.row_major(1, TPB)` for coalesced loading from global memory:
   - `load_a_layout`: Threads cooperate to load consecutive elements from matrix A rows
   - `load_b_layout`: Threads cooperate to load consecutive elements from matrix B rows
   - **Key insight**: Thread layout determines how threads cooperate during copy, not the final data layout

   **Actual Computation Pattern (proves this is A  B):**

   ```mojo
   # This is the actual computation in the current implementation
   acc += a_shared[local_row, k] * b_shared[k, local_col]

   # This corresponds to: C[i,j] = (A[i,k] * B[k,j])
   # Which is standard matrix multiplication A x B
   ```

   **Why both matrices use the same coalesced loading pattern:**

   ```txt
   Loading tiles from global memory:
   - Matrix A tile: threads load A[block_row, k], A[block_row, k+1], A[block_row, k+2]... (consecutive)
   - Matrix B tile: threads load B[k, block_col], B[k, block_col+1], B[k, block_col+2]... (consecutive)

   Both patterns are coalesced with Layout.row_major(1, TPB)
   ```

   **Three separate memory concerns:**
   1. **Global-to-shared coalescing**: `Layout.row_major(1, TPB)` ensures coalesced global memory access
   2. **Shared memory computation**: `a_shared[local_row, k] * b_shared[k, local_col]` avoids bank conflicts
   3. **Matrix operation**: The computation pattern determines this is A  B, not A  B^T

4. **Perfect tiling eliminates boundary checks**

   ```mojo
   @parameter
   for idx in range(size // TPB):  # Perfect division: 9 // 3 = 3
   ```

   With \\((9 \times 9)\\) matrices and \\((3 \times 3)\\) tiles, every tile is exactly full-sized. No boundary checking needed!

5. **Clean tile processing with defensive bounds checking**

   ```mojo
   # Defensive bounds checking included even with perfect tiling
   if tiled_row < size and tiled_col < size:
       out_tile[local_row, local_col] = acc
   ```

   With perfect \\((9 \times 9)\\) tiling, this bounds check is technically unnecessary but included for defensive programming and consistency with other matrix sizes.

#### Performance considerations

The idiomatic implementation maintains the performance benefits of tiling while providing cleaner abstractions:

1. **Memory locality**: Exploits spatial and temporal locality through tiling
2. **Coalesced access**: Specialized load layouts ensure coalesced memory access patterns
3. **Compute-memory overlap**: Potential overlap through asynchronous memory operations
4. **Shared memory efficiency**: No redundant initialization of shared memory
5. **Register pressure**: Uses accumulation registers for optimal compute throughput

This implementation shows how high-level abstractions can express complex GPU algorithms without sacrificing performance. It's a prime example of Mojo's philosophy: combining high-level expressiveness with low-level performance control.

#### Key differences from manual tiling

| Feature | Manual Tiling | Idiomatic Tiling |
|---------|--------------|------------------|
| Memory access | Direct indexing with bounds checks | LayoutTensor tile API |
| Tile loading | Explicit element-by-element copying | Dedicated copy engine bulk transfers |
| Shared memory | Manual initialization (defensive) | Managed by copy functions |
| Code complexity | More verbose with explicit indexing | More concise with higher-level APIs |
| Bounds checking | Multiple checks during loading and computing | Single defensive check at final write |
| Matrix orientation | Both A and B in same orientation (standard A  B) | Both A and B in same orientation (standard A  B) |
| Performance | Explicit control over memory patterns | Optimized layouts with register bypass |

The idiomatic approach is not just cleaner but also potentially more performant due to the use of specialized memory layouts and asynchronous operations.

#### Educational: When would transposed loading be useful?

The current implementation does NOT use transposed loading. This section is purely educational to show what's possible with the layout system.

**Current implementation recap:**

- Uses `Layout.row_major(1, TPB)` for both matrices
- Performs standard A  B multiplication
- No data transposition during copy

**Educational scenarios where you WOULD use transposed loading:**

While this puzzle uses standard coalesced loading for both matrices, the layout system's flexibility enables powerful optimizations in other scenarios:

```mojo
# Example: Loading pre-transposed matrix B^T to compute A x B
# (This is NOT what the current implementation does)
comptime load_b_layout = Layout.row_major(TPB, 1)   # Load B^T with coalesced access
comptime store_b_layout = Layout.row_major(1, TPB)  # Store as B in shared memory
copy_dram_to_sram_asyncsrc_thread_layout=load_b_layout, dst_thread_layout=store_b_layout
```

**Use cases for transposed loading (not used in this puzzle):**

1. **Pre-transposed input matrices**: When \\(B\\) is already stored transposed in global memory
2. **Different algorithms**: Computing \\(A^T \times B\\), \\(A \times B^T\\), or \\(A^T \times B^T\\)
3. **Memory layout conversion**: Converting between row-major and column-major layouts
4. **Avoiding transpose operations**: Loading data directly in the required orientation

**Key distinction:**

- **Current implementation**: Both matrices use `Layout.row_major(1, TPB)` for standard \\(A \times B\\) multiplication
- **Transposed loading example**: Would use different layouts to handle pre-transposed data or different matrix operations

This demonstrates Mojo's philosophy: providing low-level control when needed while maintaining high-level abstractions for common cases.

---

### Summary: Key takeaways

**What the idiomatic tiled implementation actually does:**

1. **Matrix Operation**: Standard A  B multiplication
2. **Memory Loading**: Both matrices use `Layout.row_major(1, TPB)` for coalesced access
3. **Computation Pattern**: `acc += a_shared[local_row, k] * b_shared[k, local_col]`
4. **Data Layout**: No transposition during loading

**Why this is optimal:**

- **Coalesced global memory access**: `Layout.row_major(1, TPB)` ensures efficient loading
- **Bank conflict avoidance**: Shared memory access pattern avoids conflicts
- **Standard algorithm**: Implements the most common matrix multiplication pattern
