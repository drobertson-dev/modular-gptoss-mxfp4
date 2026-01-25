---
title: "Attention Op"
description: "In this puzzle, we'll implement the attention mechanism as a custom MAX Graph operation. Attention is a fundamental building block of modern neural networks, popularized particularly by transformers, that allows models to focus on relevant parts of the input when making predictions."
---

# Attention Op

In this puzzle, we'll implement the attention mechanism as a custom MAX Graph operation. Attention is a fundamental building block of modern neural networks, popularized particularly by transformers, that allows models to focus on relevant parts of the input when making predictions.

## Overview

In this puzzle, we'll implement the attention mechanism as a custom MAX Graph operation. Attention is a fundamental building block of modern neural networks, popularized particularly by [transformers](https://arxiv.org/abs/1706.03762), that allows models to focus on relevant parts of the input when making predictions.

Mathematically, the attention function is defined as:

$$\\Large \\text{Attention}(Q, K, V) = \\text{softmax}(Q \\cdot K^T) \\cdot V$$

Where:

- \\(Q\\) is the **query vector**of shape \\((d,)~\\) - represents what we're looking for
- \\(K\\) is the **key matrix**of shape \\((\text{seq\_len}, d)~\\) - represents what's available to match against
- \\(V\\) is the **value matrix**of shape \\((\text{seq\_len}, d)~\\) - represents the information to retrieve
- The output is a **weighted combination**vector of shape \\((d,)\\)

The computation involves three main steps:

1. **Attention Scores**: Compute \\(Q \cdot K^T\\) to measure how well the query matches each key vector
2. **Attention Weights**: Apply softmax to convert scores into a probability distribution (weights sum to 1)
3. **Weighted Sum**: Combine value vectors using attention weights to produce the final output

## Understanding attention: a step-by-step breakdown

Think of attention as a **smart lookup mechanism**. Given a query (what you're looking for), attention finds the most relevant information from a collection of key-value pairs:

1. **Step 1 - Similarity Matching**: Compare your query \\(Q\\) against all keys \\(K\\) to get similarity scores
   - Compute \\(Q \cdot K^T\\) where each score measures how well \\(Q\\) matches each key vector
   - Higher scores = better matches

2. **Step 2 - Probability Distribution**: Convert raw scores into normalized weights
   - Apply softmax to ensure all weights sum to 1.0
   - This creates a probability distribution over which values to focus on

3. **Step 3 - Weighted Retrieval**: Combine values using the attention weights
   - Multiply each value vector by its corresponding weight
   - Sum everything up to get the final output

**Real-world analogy**: Imagine searching a library. Your query is what you want to find, the book titles are keys, and the book contents are values. Attention computes how relevant each book is to your query, then gives you a summary weighted by relevance.

### Visual computation flow

```
Input:  Q(16,)    K(16,16)    V(16,16)
                               
Step 1: Q(1,16) @ K^T(16,16) -> Scores(1,16)
         
Step 2: softmax(Scores) -> Weights(1,16)  [sum = 1.0]
         
Step 3: Weights(1,16) @ V(16,16) -> Output(1,16) -> reshape -> Output(16,)
```

**Key insight**: We reshape the query vector \\(Q\\) from shape \\((16,)\\) to \\((1,16)\\) so we can use matrix multiplication instead of manual dot products. This allows us to leverage the highly optimized tiled matmul kernel from Puzzle 18!

Our GPU implementation **reuses and combines optimized kernels from previous puzzles**:

- **Tiled matrix multiplication from Puzzle 16**for efficient \\(Q \cdot K^T\\) and \\(\text{weights} \cdot V\\) operations
- **Shared memory transpose**for computing \\(K^T\\) efficiently
- **Parallel softmax from Puzzle 18**for numerically stable attention weight computation

> ** Kernel Reuse Strategy**: This puzzle demonstrates how to build complex operations by combining proven, optimized kernels from previous puzzles. Rather than writing everything from scratch, we leverage the `matmul_idiomatic_tiled` from Puzzle 16 and `softmax_kernel` from Puzzle 18, showcasing the power of modular GPU kernel design.

## Key concepts

- Vector attention mechanism for sequence processing
- **Kernel reuse**: Leveraging proven implementations from Puzzle 16 and Puzzle 18
- Efficient matrix multiplication using shared memory tiling
- Memory-optimized tensor reshaping to minimize buffer allocation
- Integration of multiple optimized kernels into a single operation
- Custom MAX Graph operation with multi-input support
- CPU fallback implementation for compatibility

## Configuration

- **Sequence length**: \\(\text{SEQ\_LEN} = 16~\\) - number of key/value vectors in our sequence
- **Model dimension**: \\(\text{D} = 16~\\) - dimensionality of each vector (query, keys, values)
- **Threads per block**: Individually optimized for each kernel
- **Grid dimensions**: Computed dynamically to handle different matrix sizes efficiently
- **Shared memory**: Utilized in transpose, matmul, and softmax kernels for performance

Layout configuration:

- Query tensor: `Layout.row_major(d)`
- Key tensor: `Layout.row_major(seq_len, d)`
- Value tensor: `Layout.row_major(seq_len, d)`
- Output tensor: `Layout.row_major(d)`
- Custom op parameters: `{"seq_len": seq_len, "d": d, "dtype": dtype}`

Key aspects of this puzzle include:

1. **Multi-kernel orchestration**: Combining transpose, matmul, and softmax operations
2. **Memory optimization**: Using reshape operations and buffer reuse to minimize allocations
3. **Numerical stability**: Leveraging the proven softmax implementation from Puzzle 18
4. **Performance optimization**: Using tiled algorithms from Puzzle 16 for all matrix operations
5. **Multi-input operations**: Handling three input tensors (Q, K, V) in a single custom op

Our attention custom operation will:

- Accept query, key, and value tensors from Python
- Process them efficiently on GPU using optimized kernels
- Return the attention-weighted output vector
- Match the results of NumPy reference implementation

## Reference implementation (example)


To solve this puzzle, we need to implement the transpose kernel in Mojo and complete the Python graph definition for our attention custom operation. This puzzle builds upon concepts from previous puzzles, combining **tiled matrix multiplication from Puzzle 16**and **softmax from Puzzle 18**into a complete attention mechanism.

### Reused kernels

Our implementation directly incorporates these proven kernels:

1. **`matmul_idiomatic_tiled`**from Puzzle 16 - Powers both \\(Q \\times K^T\\) and \\(\\text{weights} \\times V\\) operations
2. **`softmax_kernel`**from Puzzle 18 - Provides numerically stable attention weight computation

This exemplifies **modular GPU architecture**: complex neural network operations built by orchestrating proven, optimized components rather than monolithic implementations.

The attention operation follows the canonical mathematical definition:

$$\\Large \\text{Attention}(Q, K, V) = \\text{softmax}(Q \\cdot K^T) \\cdot V$$

**Breaking down the math**:

- \\(Q \cdot K^T~\\): Query-key similarity scores of shape: \\((1, \text{seq\_len})\\)
- \\(\text{softmax}(\cdot)~\\): Normalize scores to probabilities of shape: \\((1, \text{seq\_len})\\)
- \\(\text{weights} \cdot V~\\): Weighted combination of values of shape: \\((1, d)\\)

This involves several computational steps that we optimize using GPU kernels from previous puzzles.

### 1. Transpose kernel implementation

```mojo
fn transpose_kernel
    layout_in: Layout,  # Layout for input matrix (seq_len, d)
    layout_out: Layout,  # Layout for output matrix (d, seq_len)
    rows: Int,
    cols: Int,
    dtype: DType = DType.float32,
:
    """Transpose matrix using shared memory tiling for coalesced access."""
    shared_tile = LayoutTensor[
        dtype,
        Layout.row_major(TRANSPOSE_BLOCK_DIM_XY, TRANSPOSE_BLOCK_DIM_XY),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    local_row = Int(thread_idx.y)
    local_col = Int(thread_idx.x)

    global_row = Int(block_idx.y) * TRANSPOSE_BLOCK_DIM_XY + local_row
    global_col = Int(block_idx.x) * TRANSPOSE_BLOCK_DIM_XY + local_col

    if global_row < rows and global_col < cols:
        shared_tile[local_row, local_col] = inp[global_row, global_col]

    barrier()

    out_row = Int(block_idx.x) * TRANSPOSE_BLOCK_DIM_XY + local_row
    out_col = Int(block_idx.y) * TRANSPOSE_BLOCK_DIM_XY + local_col

    # Store data from shared memory to global memory (coalesced write)
    # Note: we transpose the shared memory access pattern
    if out_row < cols and out_col < rows:
        output[out_row, out_col] = shared_tile[local_col, local_row]

```

The transpose kernel uses **shared memory tiling**to achieve coalesced memory access patterns. Key implementation details:

#### Critical transpose pattern

```mojo
# Load with normal indexing
shared_tile[local_row, local_col] = inp[global_row, global_col]
barrier()
# Store with swapped indexing for transpose
output[out_row, out_col] = shared_tile[local_col, local_row]
```

The transpose happens through **swapped indexing**in shared memory access (`[local_col, local_row]` instead of `[local_row, local_col]`) and **swapped block coordinates**for output positioning. This ensures both reads and writes remain coalesced while achieving the transpose operation.

### 2. GPU kernel orchestration

```mojo

            # Step 1: Reshape Q from (d,) to (1, d) - no buffer needed
            q_2d = q_tensor.reshape[layout_q_2d]()

            # Step 2: Transpose K from (seq_len, d) to K^T (d, seq_len)\
            comptime kernel = transpose_kernel[
                layout_k, layout_k_t, seq_len, d, dtype
            ]
            gpu_ctx.enqueue_functionkernel, kernel

            # Step 3: Compute attention scores using matmul: Q @ K^T = (1, d) @ (d, seq_len) -> (1, seq_len)
            # This computes Q - K^T[i] = Q - K[i] for each column i of K^T (which is row i of K)
            # Reuse scores_weights_buf as (1, seq_len) for scores
            scores_2d = LayoutTensordtype, layout_scores_2d, MutAnyOrigin
            comptime kernel2 = matmul_idiomatic_tiled[
                layout_q_2d,
                layout_k_t,
                layout_scores_2d,
                1,
                seq_len,
                d,
                dtype,
            ]
            gpu_ctx.enqueue_functionkernel2, kernel2

            # Step 4: Reshape scores from (1, seq_len) to (seq_len,) for softmax
            weights = scores_2d.reshape[layout_scores]()

            # Step 5: Apply softmax to get attention weights
            comptime kernel3 = softmax_gpu_kernel[layout_scores, seq_len, dtype]
            gpu_ctx.enqueue_functionkernel3, kernel3

            # Step 6: Reshape weights from (seq_len,) to (1, seq_len) for final matmul
            weights_2d = weights.reshape[layout_weights_2d]()

            # Step 7: Compute final result using matmul: weights @ V = (1, seq_len) @ (seq_len, d) -> (1, d)
            # Reuse out_tensor reshaped as (1, d) for result
            result_2d = output_tensor.reshape[layout_result_2d]()
            comptime kernel4 = matmul_idiomatic_tiled[
                layout_weights_2d,
                layout_v,
                layout_result_2d,
                1,
                d,
                seq_len,
                dtype,
            ]
            gpu_ctx.enqueue_functionkernel4, kernel4

```

The GPU orchestration demonstrates **sophisticated kernel chaining**and **zero-copy memory optimization**:

#### Advanced memory optimization strategies

```mojo
# Zero-copy reshaping - no data movement, just reinterpret tensor shape
q_2d = q_tensor.reshape[layout_q_2d]()
# Aggressive buffer reuse - same memory, different interpretations
weights = scores_2d.reshape[layout_scores]()
```

The implementation achieves **maximum memory efficiency**through:

- **Zero-copy reshaping**: Reinterpreting tensor shapes without moving data in memory
- **Intelligent buffer reuse**: The same `scores_weights_buf` serves dual purposes as both scores \\((1,\\text{seq_len})\\) and weights \\((\\text{seq_len},)\\)
- **Minimal allocations**: Only 2 temporary buffers power the entire attention operation
- **Memory coalescing**: All operations maintain optimal memory access patterns

#### Strategic kernel reuse pattern

- **Steps 3 & 7**: Both leverage `matmul_idiomatic_tiled` from Puzzle 16
  - Step 3: \\(Q \\times K^T\\)  attention scores computation \\((1,d) \\times (d,\\text{seq_len}) \\rightarrow (1,\\text{seq_len})\\)
  - Step 7: \\(\\text{weights} \\times V\\)  final weighted output \\((1,\\text{seq_len}) \\times (\\text{seq_len},d) \\rightarrow (1,d)\\)
  - Both operations include bounds checking for robustness with variable matrix dimensions
- **Step 5**: Employs `softmax_kernel` from Puzzle 18
  - Converts raw scores into normalized probability distribution
  - Ensures numerical stability through max subtraction and parallel reduction
  - Guarantees \\(\\sum_{i} \\text{weights}[i] = 1.0\\)

This exemplifies **modular GPU architecture**: complex neural network operations built by orchestrating proven, optimized kernels rather than monolithic implementations!

### Key implementation insights

#### Memory optimization strategy

The implementation achieves **minimal memory allocation**through aggressive buffer reuse:

```mojo
# Only 2 temporary buffers needed for the entire operation
k_t_buf = gpu_ctx.enqueue_create_bufferdtype
scores_weights_buf = gpu_ctx.enqueue_create_bufferdtype
```

**Key optimization insights**:

- The same `scores_weights_buf` is reused for both attention scores and weights through reshape operations
- Zero-copy tensor reshaping eliminates unnecessary data movement

#### Kernel reuse architecture

This puzzle showcases **modular kernel design**by combining three specialized kernels:

- **`matmul_idiomatic_tiled`**(used twice) - Powers both \\(Q \\times K^T\\) and \\(\\text{weights} \\times V\\) operations
- **`softmax_kernel`**- Provides numerically stable attention weight computation with parallel reduction
- **`transpose_kernel`**- Enables efficient \\(K^T\\) computation with coalesced memory access

**Architectural benefits**:

- **Composability**: Complex operations built from proven components
- **Maintainability**: Each kernel has a single, well-defined responsibility
- **Performance**: Leverages highly optimized implementations from previous puzzles
- **Scalability**: Modular design enables easy extension to larger attention mechanisms

The implementation demonstrates that **sophisticated neural network operations**can be built by orchestrating simpler, well-tested GPU kernels rather than writing monolithic implementations.
