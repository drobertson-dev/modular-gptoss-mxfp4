---
title: "Embedding Op"
description: "In this puzzle, you'll implement two different GPU kernels for embedding operations - a fundamental component in neural networks. While both kernels produce identical results, they use different memory access patterns that lead to significant performance differences."
---

# Embedding Op

In this puzzle, you'll implement two different GPU kernels for embedding operations - a fundamental component in neural networks. While both kernels produce identical results, they use different memory access patterns that lead to significant performance differences.

> ## Memory access patterns and performance
>
> We're continuing Part IV with a focus on **memory-bound operations**and **GPU memory access optimization**.
>
> Building on Puzzle 20, you'll now explore how different kernel implementations of the same operation can have dramatically different performance characteristics. You'll learn:
> - How GPU memory coalescing affects performance
> - Why grid configuration matters for memory-bound operations
> - How to design kernels with optimal memory access patterns
> - The performance implications of different threading strategies
>
> This puzzle demonstrates that **how you access memory**can be more important than **what computation you perform**.

## Overview

In this puzzle, you'll implement two different GPU kernels for embedding operations - a fundamental component in neural networks. While both kernels produce identical results, they use different memory access patterns that lead to significant performance differences.

You'll compare:
- **1D coalesced kernel**: Optimized for memory bandwidth with consecutive memory accesses
- **2D non-coalesced kernel**: Suboptimal memory access pattern for comparison

This comparison teaches the critical importance of memory coalescing in GPU kernel performance.

## Background: Embedding operations

An embedding operation converts discrete token indices into dense vector representations:

```python
# Input: token indices
indices = [[1, 5, 2], [7, 1, 9]]           # Shape: [batch_size, seq_len]

# Embedding table (learned parameters)
embedding_table = [                        # Shape: [vocab_size, embed_dim]
    [0.1, 0.2, 0.3, 0.4],  # Token 0
    [0.5, 0.6, 0.7, 0.8],  # Token 1
    [0.9, 1.0, 1.1, 1.2],  # Token 2
    # ... more tokens
]

# Output: embedded vectors
output[0,0] = embedding_table[1]  # [0.5, 0.6, 0.7, 0.8]
output[0,1] = embedding_table[5]  # lookup token 5's embedding
output[0,2] = embedding_table[2]  # [0.9, 1.0, 1.1, 1.2]
# ... and so on
```

This operation is **memory-bound**- performance depends on how efficiently you can read from the embedding table and write to the output tensor.

## Learning path

This puzzle is structured in two parts to build your understanding systematically:

### **[Simple embedding kernel](#embedding-kernels-coalesced-vs-non-coalesced)**

Start here to implement the actual puzzle code and understand the kernel implementations.

**What you'll do:**
- Complete two different GPU embedding kernels (1D coalesced vs 2D non-coalesced)
- Learn fundamental memory access patterns for GPU programming
- See the same algorithm implemented with different threading strategies
- Understand custom operation registration in Mojo

### **[Performance comparison](#performance-coalesced-vs-non-coalesced-memory-access)**

Deep dive into why the kernels perform differently and the theory behind memory coalescing.

**What you'll learn:**
- Why memory coalescing matters for GPU performance
- How thread organization affects memory bandwidth utilization
- Real-world implications for neural network optimization
- Optimization strategies for memory-bound operations

## Getting started

Ready to explore GPU memory optimization? Start with the **[Simple embedding kernel](#embedding-kernels-coalesced-vs-non-coalesced)**to implement the code, then move to **[Performance comparison](#performance-coalesced-vs-non-coalesced-memory-access)**to understand the performance implications.

 **Success tip:**Pay attention to how the different grid configurations (1D vs 2D) affect memory access patterns - this insight applies to many GPU programming scenarios beyond embeddings.

## Embedding Kernels: Coalesced vs Non-Coalesced

In this puzzle, you'll implement two different GPU kernels for embedding operations that produce identical results but use different memory access patterns, demonstrating the critical importance of memory coalescing in GPU performance.

### 1D coalesced kernel (optimized approach)

This kernel uses a simple 1D grid where each thread processes exactly one output element. The key insight is that consecutive threads will access consecutive memory locations, leading to optimal memory coalescing.

**Thread organization:**

- **Grid configuration**: `[total_elements // 256]` blocks, `256` threads per block
- **Thread mapping**: Each thread handles one `(batch, seq, embed)` position
- **Memory pattern**: Consecutive threads access consecutive embedding dimensions

**What you need to implement:**

1. Calculate the global thread index from block and thread indices
2. Convert the flat index to 3D coordinates `(batch_idx, seq_idx, embed_idx)`
3. Look up the token index from the indices tensor
4. Copy the appropriate embedding vector element to the output

#### Tips

- Start with `global_idx = block_idx.x * block_dim.x + thread_idx.x`
- Convert to 3D coordinates using division and modulo: `batch_idx = global_idx // (seq_len * embed_dim)`
- Use `remaining = global_idx % (seq_len * embed_dim)` to simplify further calculations
- Always check bounds: `if global_idx >= total_elements: return`
- Handle invalid token indices by setting output to 0
- The embedding lookup is: `output[batch_idx, seq_idx, embed_idx] = weights[token_idx, embed_idx]`

### 2D non-coalesced kernel (comparison approach)

This kernel uses a 2D grid where the X dimension spans `(batch  seq)` positions and the Y dimension spans embedding dimensions. This can lead to non-coalesced memory access patterns.

**Thread organization:**

- **Grid configuration**: `[batch x seq // 16, embed_dim // 16]` blocks, `16 x 16` threads per block
- **Thread mapping**: `thread_idx.x` maps to batch/sequence, `thread_idx.y` maps to embedding dimension
- **Memory pattern**: Threads in a warp may access scattered memory locations

**What you need to implement:**

1. Calculate both X and Y coordinates from the 2D grid
2. Convert the X coordinate to separate batch and sequence indices
3. Use the Y coordinate directly as the embedding dimension
4. Perform the same embedding lookup with bounds checking

#### Tips

- Use both X and Y thread coordinates: `batch_seq_idx = block_idx.x * block_dim.x + thread_idx.x`
- And: `embed_idx = block_idx.y * block_dim.y + thread_idx.y`
- Convert `batch_seq_idx` to separate batch and sequence indices: `batch_idx = batch_seq_idx // seq_len`
- Remember to check bounds for both dimensions: `if batch_seq_idx >= total_positions or embed_idx >= embed_dim`
- The token lookup is the same as 1D, but you're only handling one embedding dimension per thread
- This kernel processes one embedding dimension per thread instead of entire vectors

### Custom ops registration

The kernels are wrapped in PyTorch custom operations for easy integration. The registration pattern is the same as MAX custom ops explained in Understanding MAX Graph custom ops:

#### 1D coalesced operation

This operation registers the optimized 1D embedding kernel as `"embedding"`:

```mojo
import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from memory import UnsafePointer
from gpu.host import DeviceBuffer

@compiler.register("embedding")
struct EmbeddingCustomOp:
    @staticmethod
    fn execute
        target: StaticString,
        batch_size: Int,
        seq_len: Int,
        vocab_size: Int,
        embed_dim: Int,
     raises:
        output_tensor = output.to_layout_tensor()
        indices_tensor = indices.to_layout_tensor()
        weights_tensor = weights.to_layout_tensor()

        comptime indices_layout = indices_tensor.layout
        comptime weights_layout = weights_tensor.layout
        comptime out_layout = output_tensor.layout

        @parameter
        if target == "gpu":
            gpu_ctx = ctx.get_device_context()

            # Zero out output tensor
            gpu_ctx.enqueue_memset(
                DeviceBufferoutput.dtype,
                0,
            )

            # Calculate 1D grid dimensions (matching kernel's flat indexing)
            total_elements = batch_size * seq_len * embed_dim
            blocks = max(1, ceildiv(total_elements, THREADS_PER_BLOCK))

            # Compile and launch optimized kernel
            comptime kernel = embedding_kernel_coalesced[
                indices_layout,
                weights_layout,
                out_layout,
                batch_size,
                seq_len,
                vocab_size,
                embed_dim,
                output.dtype,
            ]
            compiled_kernel = gpu_ctx.compile_function[kernel, kernel]()

            gpu_ctx.enqueue_function(
                compiled_kernel,
                output_tensor,
                indices_tensor,
                weights_tensor,
                grid_dim=(blocks,),
                block_dim=(THREADS_PER_BLOCK,),
            )

        elif target == "cpu":
            for batch in range(batch_size):
                for seq in range(seq_len):
                    token_idx_val = Int(indices_tensor[batch, seq])
                    if token_idx_val >= 0 and token_idx_val < vocab_size:
                        for emb in range(embed_dim):
                            output_tensor[batch, seq, emb] = weights_tensor[
                                token_idx_val, emb
                            ]
        else:
            raise Error("Unsupported target: " + target)

```

**Key aspects of this registration:**

- **Simple grid configuration**: Uses a straightforward 1D grid with `ceildiv(total_elements, THREADS_PER_BLOCK)` blocks
- **Memory optimization**: Single `enqueue_memset` call to zero the output buffer efficiently
- **Compile-time parameters**: All tensor dimensions passed as compile-time parameters for optimal performance
- **Device abstraction**: Handles both GPU execution and CPU fallback seamlessly

#### 2D non-coalesced operation

This operation registers the comparison 2D embedding kernel as `"embedding_2d"`:

```mojo
@compiler.register("embedding_2d")
struct Embedding2DCustomOp:
    @staticmethod
    fn execute
        target: StaticString,
        batch_size: Int,
        seq_len: Int,
        vocab_size: Int,
        embed_dim: Int,
     raises:
        output_tensor = output.to_layout_tensor()
        indices_tensor = indices.to_layout_tensor()
        weights_tensor = weights.to_layout_tensor()

        comptime indices_layout = indices_tensor.layout
        comptime weights_layout = weights_tensor.layout
        comptime out_layout = output_tensor.layout

        @parameter
        if target == "gpu":
            gpu_ctx = ctx.get_device_context()

            # Zero out output tensor
            gpu_ctx.enqueue_memset(
                DeviceBufferoutput.dtype,
                0,
            )

            # Calculate 2D grid dimensions for non-coalesced access
            total_positions = batch_size * seq_len
            comptime BLOCK_X = 16  # batch*seq dimension
            comptime BLOCK_Y = 16  # embed dimension
            blocks_x = max(1, ceildiv(total_positions, BLOCK_X))
            blocks_y = max(1, ceildiv(embed_dim, BLOCK_Y))

            # Compile and launch 2D kernel
            comptime kernel = embedding_kernel_2d[
                indices_layout,
                weights_layout,
                out_layout,
                batch_size,
                seq_len,
                vocab_size,
                embed_dim,
                output.dtype,
            ]

            compiled_kernel = gpu_ctx.compile_function[kernel, kernel]()

            gpu_ctx.enqueue_function(
                compiled_kernel,
                output_tensor,
                indices_tensor,
                weights_tensor,
                grid_dim=(blocks_x, blocks_y),
                block_dim=(BLOCK_X, BLOCK_Y),
            )

        elif target == "cpu":
            # Same CPU fallback as 1D version
            for batch in range(batch_size):
                for seq in range(seq_len):
                    token_idx_val = Int(indices_tensor[batch, seq])
                    if token_idx_val >= 0 and token_idx_val < vocab_size:
                        for emb in range(embed_dim):
                            output_tensor[batch, seq, emb] = weights_tensor[
                                token_idx_val, emb
                            ]
        else:
            raise Error("Unsupported target: " + target)

```

**Key differences from the 1D operation:**

- **Complex grid configuration**: Uses a 2D grid with separate calculations for `blocks_x` and `blocks_y`
- **Fixed block dimensions**: Hard-coded `BLOCK_X = 16` and `BLOCK_Y = 16` for 2D thread organization
- **Same memory management**: Identical memory initialization and CPU fallback logic
- **Different kernel call**: Passes 2D grid dimensions `(blocks_x, blocks_y)` and block dimensions `(BLOCK_X, BLOCK_Y)`

#### Common wrapper functionality

Both custom operations provide essential infrastructure:

1. **Memory management**:
   - Zero-initialization of output tensors with `enqueue_memset`
   - Proper buffer creation and memory layout handling
   - Automatic cleanup and resource management

2. **Device abstraction**:
   - GPU execution with optimized kernels
   - CPU fallback for compatibility and debugging
   - Consistent interface regardless of execution target

3. **Parameter passing**:
   - Compile-time tensor dimensions for kernel optimization
   - Runtime tensor data through layout tensor conversion
   - Type-safe parameter validation

4. **Grid configuration**:
   - Automatic calculation of optimal grid dimensions
   - Different strategies optimized for each kernel's access pattern
   - Proper block dimension management

#### Integration with PyTorch

These registered operations can be called from Python using the [CustomOpLibrary](https://docs.modular.com/max/api/python/torch/CustomOpLibrary/):

```python
# Load the custom operations
ops = CustomOpLibrary(mojo_kernels)

# Call the 1D coalesced version
result_1d = ops.embedding{"batch_size": B, "seq_len": L, "vocab_size": V, "embed_dim": E}

# Call the 2D non-coalesced version
result_2d = ops.embedding_2d{"batch_size": B, "seq_len": L, "vocab_size": V, "embed_dim": E}
```

The power of this approach is that the same kernel implementations can be used across different Python frameworks while maintaining optimal performance characteristics.

### Run the code

You can run the puzzle with:

  
    pixi NVIDIA (default)
    pixi AMD
    uv
  
  

```bash
pixi run p21
```

  
  

```bash
pixi run -e amd p21
```

  
  

```bash
uv run poe p21
```

  

When successful, you should see output similar to:

```
Puzzle 21: Mojo Embedding Kernel Comparison
======================================================================
Configuration: B=8, L=512, V=10000, E=512
------------------------------------------------------------

Testing Correctness...
   1D Coalesced - Max difference: 1.19e-07
   2D Non-coalesced - Max difference: 1.19e-07
   OK Both implementations CORRECT

Benchmarking Mojo Kernels...

Performance Results:
   1D Coalesced:     2.145 ms
   2D Non-coalesced: 3.867 ms
   1D is 1.80x faster than 2D

Key Learning Points:
- Compare different GPU kernel implementations
- 1D vs 2D grid patterns have different memory access
- Coalesced memory access should be faster
- Grid configuration affects GPU utilization
```

### Reference implementation (example)


The solution involves implementing the coordinate transformations and memory operations for both kernels:

### 1D Coalesced Kernel

```mojo
fn embedding_kernel_coalesced
    indices_layout: Layout,
    weights_layout: Layout,
    out_layout: Layout,
    batch_size: Int,
    seq_len: Int,
    vocab_size: Int,
    embed_dim: Int,
    dtype: DType = DType.float32,
:
    """
    Memory-coalescing focused embedding kernel.

    Key insight: The bottleneck is memory access patterns, not computation.
    - Each thread handles one (batch, seq, embed) position
    - Simple 1D grid for maximum simplicity and correctness
    - Focus on getting memory access right first
    """

    # Simple 1D indexing - each thread = one output element
    global_idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    total_elements = batch_size * seq_len * embed_dim

    if global_idx >= total_elements:
        return

    # Convert to (batch, seq, embed) coordinates
    batch_idx = global_idx // (seq_len * embed_dim)
    remaining = global_idx % (seq_len * embed_dim)
    seq_idx = remaining // embed_dim
    embed_idx = remaining % embed_dim

    # Get token index
    token_idx_val = Int(indices[batch_idx, seq_idx])

    # Simple, correct assignment
    if token_idx_val >= 0 and token_idx_val < vocab_size:
        output[batch_idx, seq_idx, embed_idx] = weights[
            token_idx_val, embed_idx
        ]
    else:
        output[batch_idx, seq_idx, embed_idx] = 0

```

### 2D Non-Coalesced Kernel

```mojo
fn embedding_kernel_2d
    indices_layout: Layout,
    weights_layout: Layout,
    out_layout: Layout,
    batch_size: Int,
    seq_len: Int,
    vocab_size: Int,
    embed_dim: Int,
    dtype: DType = DType.float32,
:
    """
    2D grid non-coalesced embedding kernel.

    Non-optimal approach for comparison:
    - 2D grid: (batch*seq, embed_dim)
    - More complex indexing
    - Potentially worse memory access patterns
    """

    # 2D grid indexing
    batch_seq_idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    embed_idx = Int(block_idx.y * block_dim.y + thread_idx.y)

    total_positions = batch_size * seq_len

    # Bounds check
    if batch_seq_idx >= total_positions or embed_idx >= embed_dim:
        return

    # Convert to (batch, seq) coordinates
    batch_idx = batch_seq_idx // seq_len
    seq_idx = batch_seq_idx % seq_len

    # Get token index
    token_idx_val = Int(indices[batch_idx, seq_idx])

    # Assignment with 2D grid pattern
    if token_idx_val >= 0 and token_idx_val < vocab_size:
        output[batch_idx, seq_idx, embed_idx] = weights[
            token_idx_val, embed_idx
        ]
    else:
        output[batch_idx, seq_idx, embed_idx] = 0

```

Both solutions implement the same embedding lookup logic but with different thread organizations:

#### Key differences

1. **Thread mapping**:
   - **1D kernel**: One thread per output element, simple flat indexing
   - **2D kernel**: 2D grid mapping to (batchseq, embed_dim) coordinates

2. **Memory access patterns**:
   - **1D kernel**: Consecutive threads access consecutive embedding dimensions  coalesced
   - **2D kernel**: Thread access pattern depends on block configuration  potentially non-coalesced

3. **Indexing complexity**:
   - **1D kernel**: Single division/modulo chain to get 3D coordinates
   - **2D kernel**: Separate X/Y coordinate calculations

#### Performance implications

The 1D kernel typically performs better because:

- **Memory coalescing**: Consecutive threads access consecutive memory addresses
- **Simple indexing**: Lower computational overhead for coordinate calculations
- **Better cache utilization**: Predictable memory access patterns

The 2D kernel may perform worse due to:

- **Scattered memory accesses**: Threads within a warp may access different embedding vectors
- **Complex grid configuration**: 1616 blocks may not align optimally with memory layout
- **Warp divergence**: Different threads may follow different execution paths

### Key concepts

| Concept | 1D Coalesced | 2D Non-coalesced |
|---------|---------------|-------------------|
| **Thread organization**| 1D flat indexing | 2D grid (batchseq, embed) |
| **Memory access**| Consecutive addresses | Potentially scattered |
| **Grid configuration**| Simple: `[total_elements // 256]` | Complex: `[batchseq // 16, embed // 16]` |
| **Performance**| Optimized for memory bandwidth | Suboptimal memory pattern |
| **Use case**| Production kernels | Educational comparison |

The core lesson: **memory coalescing**can lead to 2-3x performance differences for memory-bound operations like embeddings.

## Performance: Coalesced vs non-coalesced memory access

Understanding memory access patterns is crucial for GPU performance optimization. This section explains why coalesced memory access patterns typically outperform non-coalesced patterns, particularly for memory-bound operations like embedding lookups.

### Memory coalescing basics

**Memory coalescing**occurs when consecutive threads in a warp access consecutive memory addresses. GPUs can combine these individual memory requests into fewer, larger memory transactions, dramatically improving bandwidth utilization.

#### Coalesced vs non-coalesced access

**Coalesced (efficient):**
```
- Thread 0 -> Address 0x1000
- Thread 1 -> Address 0x1004
- Thread 2 -> Address 0x1008
- Thread 3 -> Address 0x100C
- ...
```

**Result**: 1 memory transaction for entire warp (32 threads)

**Non-coalesced (inefficient):**
```
- Thread 0 -> Address 0x1000
- Thread 1 -> Address 0x2000
- Thread 2 -> Address 0x3000
- Thread 3 -> Address 0x4000
- ...
```

**Result**: Up to 32 separate memory transactions

### Why embedding operations are memory-bound

Embedding lookups are **memory-bound**because they involve:
- **Minimal computation**: Just copying data from input to output
- **Large memory footprint**: Embedding tables can be gigabytes in size
- **High memory bandwidth requirements**: Need to transfer large amounts of data

For such operations, **memory access efficiency**determines performance more than computational complexity.

### Kernel comparison

#### 1D coalesced kernel
- **Thread organization**: `[total_elements // 256]` blocks, one thread per output element
- **Memory pattern**: Consecutive threads access consecutive embedding dimensions
- **Why it's coalesced**: `Thread 0: output[0,0,0]`, `Thread 1: output[0,0,1]`  consecutive addresses

#### 2D non-coalesced kernel
- **Thread organization**: `[batch*seq // 16, embed_dim // 16]` blocks with 1616 threads
- **Memory pattern**: Threads may access different embedding vectors
- **Why it's non-coalesced**: Thread access pattern can be scattered across memory

### Performance results

Typical benchmark results:
```
Performance Results:
   1D Coalesced:     2.145 ms
   2D Non-coalesced: 3.867 ms
   1D is 1.80x faster than 2D
```

### Memory access visualization

#### Coalesced pattern (1D kernel)

**Warp execution for output[0,0,0:32]:**

| Element | Thread ID | Memory Access | Address Pattern |
|---------|-----------|---------------|-----------------|
| `output[0,0,0]` | 0 | `[0,0]` | Base + 0 |
| `output[0,0,1]` | 1 | `[0,1]` | Base + 4 |
| `output[0,0,2]` | 2 | `[0,2]` | Base + 8 |
| `output[0,0,3]` | 3 | `[0,3]` | Base + 12 |
| ... | ... | ... | ... |
| `output[0,0,31]` | 31 | `[0,31]` | Base + 124 |

**Result**: Consecutive addresses  **1 memory transaction**for entire warp

#### Non-coalesced pattern (2D kernel)

**Warp execution with 1616 blocks:**

```
Block organization (16x16):
    X-dim: batch*seq positions (0-15)
    Y-dim: embed dimensions (0-15)

Warp threads might access:
    Thread 0:  batch=0, seq=0, embed=0  -> Address A
    Thread 1:  batch=0, seq=1, embed=0  -> Address B (different row)
    Thread 2:  batch=0, seq=2, embed=0  -> Address C (different row)
    ...
    Thread 31: batch=1, seq=15, embed=0 -> Address Z (scattered)
```

**Result**: Potentially scattered addresses  **Multiple memory transactions**

### Key optimization strategies

1. **Prefer 1D indexing**for memory-bound operations when possible
2. **Align data structures**to coalescing-friendly layouts
3. **Consider memory access patterns**during kernel design
4. **Profile memory bandwidth**to identify bottlenecks
5. **Use memory-bound benchmarks**to validate optimizations

The core insight: **memory access patterns**often determine GPU performance more than computational complexity, especially for memory-bound operations like embeddings.
