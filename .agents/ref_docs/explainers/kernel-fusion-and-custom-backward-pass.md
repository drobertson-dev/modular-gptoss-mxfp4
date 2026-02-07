---
title: "Kernel Fusion and Custom Backward Pass"
description: "In this puzzle, you'll implement fused LayerNorm + Linear operations with both forward and backward passes. While both fused and unfused implementations produce identical results, they use different strategies that lead to significant performance differences."
---

# Kernel Fusion and Custom Backward Pass

In this puzzle, you'll implement fused LayerNorm + Linear operations with both forward and backward passes. While both fused and unfused implementations produce identical results, they use different strategies that lead to significant performance differences.

> ## Kernel fusion and autograd integration
>
> We're continuing Part IV with a focus on **kernel fusion**and **autograd integration**.
>
> Building on Puzzle 21, you'll now explore how to combine multiple operations into a single efficient kernel and integrate it with PyTorch's autograd system. You'll learn:
>
> - How kernel fusion improves performance in both forward and backward passes
> - Why custom backward passes are crucial for fused operations
> - How to design fused kernels with proper gradient flow
> - The performance implications of different fusion strategies
>
> This puzzle demonstrates that **how you combine operations**can be as important as **how you implement them**.

## Overview

In this puzzle, you'll implement fused LayerNorm + Linear operations with both forward and backward passes. While both fused and unfused implementations produce identical results, they use different strategies that lead to significant performance differences.

You'll compare:

- **Unfused approach**: Separate kernels for LayerNorm and Linear
- **Fused kernel**: Combined operation in a single kernel
- **Custom backward pass**: Gradient computation for fused operations

This comparison teaches the critical importance of kernel fusion and proper gradient computation in deep learning operations.

## Background: LayerNorm + Linear operations

LayerNorm and Linear are fundamental operations in transformer architectures, particularly in attention mechanisms and feed-forward networks. Here's how they're typically used:

```python
import torch
import torch.nn.functional as F

# Input: hidden states
x = torch.randn(batch_size, seq_len, hidden_dim)

# LayerNorm parameters
ln_weight = torch.ones(hidden_dim)  # scale parameter ()
ln_bias = torch.zeros(hidden_dim)   # shift parameter ()

# Linear layer parameters
linear_weight = torch.randn(output_dim, hidden_dim)
linear_bias = torch.zeros(output_dim)

# Unfused operations (with autograd)
ln_output = F.layer_norm(x, [hidden_dim], weight=ln_weight, bias=ln_bias)
output = F.linear(ln_output, linear_weight, linear_bias)

# Fused operation (custom implementation)
# This is what you'll implement in this puzzle
output_fused = fused_layernorm_linear(x, ln_weight, ln_bias, linear_weight, linear_bias)
```

When fused, these operations are combined into a single efficient kernel that:

- Reduces memory bandwidth usage
- Minimizes kernel launch overhead
- Improves cache utilization
- Eliminates intermediate allocations

In practice, this fusion can provide up to 1.5-2x speedup in both forward and backward passes, which is crucial for transformer training efficiency.

### Why custom backward passes matter

PyTorch's autograd system automatically computes gradients for individual operations, but fused operations require custom backward passes to:

- Maintain numerical stability
- Ensure proper gradient flow
- Optimize memory access patterns
- Handle atomic operations for gradient accumulation

## Learning path

This puzzle is structured in two parts to build your understanding systematically:

### **[Forward pass implementation](#fused-vs-unfused-kernels)**

Start here to implement the fused forward kernel and understand kernel fusion benefits.

**What you'll do:**

- Implement both unfused and fused forward kernels
- Learn fundamental kernel fusion techniques
- See the same operations implemented with different strategies
- Understand performance implications of fusion
- Learn memory access patterns for optimal performance

### **[Backward pass implementation](#autograd-integration-backward-pass)**

Deep dive into autograd integration and gradient computation.

**What you'll learn:**

- How to implement custom backward passes
- Why proper gradient flow is crucial
- Real-world implications for training efficiency
- Optimization strategies for backward operations
- Mathematical foundations of gradient computation
- Atomic operations for gradient accumulation
- Numerical stability in backward passes

## Getting started

Ready to explore kernel fusion and autograd integration? Start with the **[Forward pass implementation](#fused-vs-unfused-kernels)**to implement the fused kernel, then move to **[Backward pass implementation](#autograd-integration-backward-pass)**to understand gradient computation.

The puzzle includes a comprehensive testing framework that verifies:

- Numerical correctness against PyTorch's implementation for both forward and backward passes
- Performance comparison between our CPU and GPU implementations
- Gradient computation accuracy for all parameters (input, LayerNorm weights/bias, Linear weights/bias)
- Memory usage optimization through kernel fusion

 **Success tip:**Pay attention to how the different implementations (fused vs unfused) affect both forward and backward pass performance - this insight applies to many deep learning operations beyond LayerNorm + Linear. The backward pass implementation is particularly important as it directly impacts training efficiency and numerical stability.

##  Fused vs Unfused Kernels

### Overview

In this puzzle, we explore the performance benefits of kernel fusion by implementing and comparing two approaches to the [LayerNorm](https://arxiv.org/abs/1607.06450) and Linear operation:

1. **Unfused approach**: Executes LayerNorm and Linear as separate operations
2. **Fused kernel**: Combines LayerNorm and Linear operations into a single GPU kernel

This comparison demonstrates how kernel fusion can significantly improve performance by:

- Reducing memory bandwidth usage
- Minimizing kernel launch overhead
- Improving cache utilization
- Eliminating intermediate memory allocations

### Key concepts

In this puzzle, you'll learn:

- **Kernel fusion techniques**for combining multiple operations
- **Memory bandwidth optimization**through fused operations
- **Performance benchmarking**of different kernel implementations
- **Numerical stability**in fused operations
- **PyTorch custom operation integration**

The mathematical operations we're fusing are:

1. LayerNorm:
\\[\Large \text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta \\]

2. Linear:
\\[\Large \text{Linear}(x) = Wx + b \\]

When fused, we compute:
\\[\Large \text{Fused}(x) = W(\gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta) + b \\]

### Understanding LayerNorm

LayerNorm is a normalization technique that helps stabilize and accelerate the training of deep neural networks. Let's break down its components and parameters:

#### What LayerNorm does

1. **Normalization**: LayerNorm normalizes the activations across the features (hidden dimensions) for each sample independently. This means:
   - For each sequence position, it computes statistics across the hidden dimension
   - Each sample in the batch is normalized independently
   - This is different from [BatchNorm](https://arxiv.org/abs/1502.03167), which normalizes across the batch dimension

2. **Parameters**:
   - \\(\gamma\\) (scale): A learnable parameter vector that allows the network to learn the optimal scale for each feature
   - \\(\beta\\) (shift): A learnable parameter vector that allows the network to learn the optimal shift for each feature
   - \\(\epsilon\\): A small constant (1e-5) added to the variance to prevent division by zero

#### What LayerNorm does in practice

LayerNorm performs several crucial functions in deep neural networks:

1. **Feature standardization**:
   - Transforms each feature to have zero mean and unit variance
   - Makes the network's learning process more stable
   - Helps prevent the "internal covariate shift" problem where the distribution of layer inputs changes during training

2. **Gradient flow**:
   - Improves gradient flow through the network
   - Prevents vanishing/exploding gradients
   - Makes training more efficient by allowing higher learning rates

3. **Regularization effect**:
   - Acts as a form of implicit regularization
   - Helps prevent overfitting by normalizing the feature distributions
   - Makes the network more robust to input variations

4. **Sequence modeling**:
   - Particularly effective in transformer architectures
   - Helps maintain consistent signal magnitude across different sequence lengths
   - Enables better handling of variable-length sequences

5. **Training dynamics**:
   - Accelerates training convergence
   - Reduces the need for careful learning rate tuning
   - Makes the network less sensitive to weight initialization

#### Mathematical components

1. **Mean Calculation**(\\(\mu\\)):
   \\[\Large \mu = \frac{1}{H} \sum_{i=1}^{H} x_i \\]
   - Computes the mean across the hidden dimension (H)
   - Each sequence position has its own mean

2. **Variance Calculation**(\\(\sigma^2\\)):
   \\[\Large \sigma^2 = \frac{1}{H} \sum_{i=1}^{H} (x_i - \mu)^2 \\]
   - Computes the variance across the hidden dimension
   - Used to scale the normalized values

3. **Normalization and Scaling**:
   \\[\Large \text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta \\]
   - First normalizes the input to have zero mean and unit variance
   - Then applies learnable scale (\\(\gamma\\)) and shift (\\(\beta\\)) parameters
   - The \\(\odot\\) symbol represents elementwise multiplication (Hadamard product)
   - For example, if \\(\gamma = [1.2, 0.8, 1.5]\\)  and normalized input is \\([0.5, -0.3, 0.7]\\), then \\(\gamma \odot x = [0.6, -0.24, 1.05]\\)

#### Why LayerNorm is important

1. **Training Stability**:
   - Prevents activations from growing too large or small
   - Helps maintain consistent signal magnitude throughout the network

2. **Feature Learning**:
   - The scale (\\(\gamma\\)) and shift (\\(\beta\\)) parameters allow the network to learn which features are important
   - Can effectively learn to ignore or emphasize certain features

3. **Independence**:
   - Unlike BatchNorm, LayerNorm's statistics are computed independently for each sample
   - Makes it more suitable for variable-length sequences and small batch sizes

### Configuration

- Batch size: `BATCH_SIZE = 4`
- Sequence length: `SEQ_LEN = 4`
- Hidden dimension: `HIDDEN_DIM = 8`
- Output dimension: `OUTPUT_DIM = 16`
- Epsilon: `EPS = 1e-5`
- Data type: `DType.float32`

### Implementation approaches

#### 1. Unfused implementation

The unfused approach executes operations separately using multiple kernels. Here are some of the kernels we wrote in the previous chapters:

##### Matrix multiplication kernel

From Puzzle 16, we reuse the tiled matrix multiplication kernel for the linear transformation. This kernel includes bounds checking to handle variable matrix dimensions safely:

```mojo
# Idiomatic tiled matmul from p19.mojo
fn matmul_idiomatic_tiled
    a_layout: Layout,
    b_layout: Layout,
    out_layout: Layout,
    rows: Int,
    cols: Int,
    inner: Int,
    dtype: DType = DType.float32,
:
    """Idiomatic tiled matrix multiplication from p19."""
    local_row = thread_idx.y
    local_col = thread_idx.x
    tiled_row = Int(block_idx.y * MATMUL_BLOCK_DIM_XY + local_row)
    tiled_col = Int(block_idx.x * MATMUL_BLOCK_DIM_XY + local_col)

    # Get the tile of the output matrix that this thread block is responsible for
    out_tile = output.tileMATMUL_BLOCK_DIM_XY, MATMUL_BLOCK_DIM_XY, Int(block_idx.x)
    )
    a_shared = LayoutTensor[
        dtype,
        Layout.row_major(MATMUL_BLOCK_DIM_XY, MATMUL_BLOCK_DIM_XY),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    b_shared = LayoutTensor[
        dtype,
        Layout.row_major(MATMUL_BLOCK_DIM_XY, MATMUL_BLOCK_DIM_XY),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    var acc: output.element_type = 0

    comptime load_a_layout = Layout.row_major(
        MATMUL_BLOCK_DIM_XY, MATMUL_BLOCK_DIM_XY
    )  # Coalesced loading
    comptime load_b_layout = Layout.row_major(
        MATMUL_BLOCK_DIM_XY, MATMUL_BLOCK_DIM_XY
    )  # Coalesced loading

    @parameter
    for idx in range((inner + MATMUL_BLOCK_DIM_XY - 1) // MATMUL_BLOCK_DIM_XY):
        # Get tiles from A and B matrices
        a_tile = a.tileMATMUL_BLOCK_DIM_XY, MATMUL_BLOCK_DIM_XY, idx
        )
        b_tile = b.tileMATMUL_BLOCK_DIM_XY, MATMUL_BLOCK_DIM_XY
        )

        # Asynchronously copy tiles to shared memory with consistent orientation
        copy_dram_to_sram_async
            thread_layout=load_a_layout,
            num_threads=MATMUL_NUM_THREADS,
            block_dim_count=MATMUL_BLOCK_DIM_COUNT,
        
        copy_dram_to_sram_async
            thread_layout=load_b_layout,
            num_threads=MATMUL_NUM_THREADS,
            block_dim_count=MATMUL_BLOCK_DIM_COUNT,
        

        # Wait for all async copies to complete
        async_copy_wait_all()
        barrier()

        # Compute partial matrix multiplication for this tile
        @parameter
        for k in range(MATMUL_BLOCK_DIM_XY):
            if (
                tiled_row < rows and tiled_col < cols
            ):  # Only perform calculation for valid outputs
                if k < a_tile.dim(
                    1
                ):  # Only perform calculation on valid inputs
                    acc += a_shared[local_row, k] * b_shared[k, local_col]

        barrier()

    # Write final result with bounds checking (needed for variable matrix sizes)
    if tiled_row < rows and tiled_col < cols:
        out_tile[local_row, local_col] = acc

```

##### Transpose kernel

For efficient memory access patterns, we use a transpose kernel with shared memory tiling:

```mojo
fn transpose_kernel
    layout_in: Layout,
    layout_out: Layout,
    rows: UInt,
    cols: UInt,
    dtype: DType = DType.float32,
:
    """Transpose matrix using shared memory tiling for coalesced access.
    We will learn more about coalesced access in the next part.
    """
    shared_tile = LayoutTensor[
        dtype,
        Layout.row_major(TRANSPOSE_BLOCK_DIM_XY, TRANSPOSE_BLOCK_DIM_XY),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    local_row = thread_idx.y
    local_col = thread_idx.x

    global_row = block_idx.y * TRANSPOSE_BLOCK_DIM_XY + local_row
    global_col = block_idx.x * TRANSPOSE_BLOCK_DIM_XY + local_col

    if global_row < rows and global_col < cols:
        shared_tile[local_row, local_col] = inp[global_row, global_col]

    barrier()

    out_row = block_idx.x * TRANSPOSE_BLOCK_DIM_XY + local_row
    out_col = block_idx.y * TRANSPOSE_BLOCK_DIM_XY + local_col

    # Store data from shared memory to global memory (coalesced write)
    # Note: we transpose the shared memory access pattern
    if out_row < cols and out_col < rows:
        output[out_row, out_col] = shared_tile[local_col, local_row]

```

##### Bias addition kernel

A simple elementwise addition kernel for adding the bias term:

```mojo
fn add_bias_kernel
    input_layout: Layout,
    bias_layout: Layout,
    output_layout: Layout,
    batch_size: Int,
    seq_len: Int,
    output_dim: Int,
:
    """Simple bias addition."""
    batch_idx = Int(block_idx.x)
    seq_idx = Int(block_idx.y)
    out_idx = Int(thread_idx.x)

    if batch_idx >= batch_size or seq_idx >= seq_len or out_idx >= output_dim:
        return

    output[batch_idx, seq_idx, out_idx] = input[
        batch_idx, seq_idx, out_idx
    ] + rebind[Scalar[dtype]](bias[out_idx])

```

##### LayerNorm kernel

Now complete this kernel to implement the LayerNorm operation. You'll need to:

1. Compute mean \\(\mu\\) and variance \\(\sigma^2\\) for each sequence position
2. Normalize the input using these statistics
3. Apply the scale \\(\gamma\\) and shift \\(\beta\\) parameters

```mojo
fn layernorm_kernel
    input_layout: Layout,
    ln_params_layout: Layout,
    output_layout: Layout,
    batch_size: Int,
    seq_len: Int,
    hidden_dim: Int,
:
    batch_idx = Int(block_idx.x)
    seq_idx = Int(block_idx.y)
    hidden_idx = Int(thread_idx.x)

    if (
        batch_idx >= batch_size
        or seq_idx >= seq_len
        or hidden_idx >= hidden_dim
    ):
        return

    # Compute statistics for this sequence position (redundant but simple)
    var sum_val: Scalar[dtype] = 0
    var sq_sum: Scalar[dtype] = 0

    # FILL ME IN (roughly 11 lines)

```

**Implementation steps:**

1. First, compute mean and variance using parallel reduction
2. Then normalize the input using these statistics
3. Finally, apply the scale and shift parameters

**Characteristics of unfused approach:**

- Multiple kernel launches (LayerNorm  MatMul  Bias)
- Intermediate tensor allocations between operations
- More memory bandwidth usage due to separate passes
- Simpler implementation with clear separation of concerns
- Easier to debug as each operation is isolated

#### Tips

1. **Thread organization**:
   - Use one thread block per sequence position (grid: `[batch_size, seq_len]`)
   - Each thread handles one hidden dimension element
   - Avoid redundant computation by computing statistics once per sequence

2. **Memory access**:
   - Access input tensor with `[batch_idx, seq_idx, hidden_idx]`
   - Access output tensor with `[batch_idx, seq_idx, hidden_idx]`
   - Access LayerNorm parameters with `[hidden_idx]`

3. **Numerical stability**:
   - Add epsilon (1e-5) before taking square root
   - Use `rebind[Scalar[dtype]]` for proper type casting
   - Compute variance as (sq_sum / hidden_dim) - (mean * mean)

4. **Performance**:
   - Compute mean and variance in a single pass
   - Reuse computed statistics for all elements in sequence
   - Avoid unnecessary memory barriers

#### Running the code

To test your unfused implementation, run:

  
    pixi NVIDIA (default)
    pixi AMD
    uv
  
  

```bash
pixi run p22 --unfused
```

  
  

```bash
pixi run -e amd p22 --unfused
```

  
  

```bash
uv run poe p22 --unfused
```

  

Your output will look like this:

```txt
Testing with dimensions: [4, 4, 8] -> [4, 4, 16]
OK Loaded Mojo operations library
============================================================
   Puzzle 22: UNFUSED Algorithm Test & Benchmark
============================================================

 Correctness Testing for UNFUSED Algorithm
====================================================

Testing Reference PyTorch Implementation
-----------------------------------------------
OK Reference PyTorch
   Max difference: 0.00e+00
   Result: OK CORRECT

Testing CPU Implementation
---------------------------------
OK Using Mojo fused kernel (CPU)
   Max difference: 1.86e-08
   Result: OK CORRECT

Testing GPU Unfused Implementation
-----------------------------------------
OK Using Mojo unfused kernel (GPU)
   Max difference: 1.86e-08
   Result: OK CORRECT

Correctness Summary:
   - Reference:   OK CORRECT
   - CPU:         OK CORRECT
   - GPU unfused: OK CORRECT

   Overall Correctness: OK ALL CORRECT

Benchmarking CPU vs GPU UNFUSED
------------------------------------------
   Testing CPU performance...
   CPU: 3173.70ms (50 iterations)
   Testing GPU unfused performance...
   GPU unfused: 3183.57ms (50 iterations)

   GPU unfused vs CPU: 1.00x slower
   CPU wins (GPU overhead > computation benefit)

UNFUSED Algorithm Test Completed!
```

### Reference implementation (example)


```mojo
fn layernorm_kernel
    input_layout: Layout,
    ln_params_layout: Layout,
    output_layout: Layout,
    batch_size: Int,
    seq_len: Int,
    hidden_dim: Int,
    dtype: DType = DType.float32,
:
    batch_idx = Int(block_idx.x)
    seq_idx = Int(block_idx.y)
    hidden_idx = Int(thread_idx.x)

    if (
        batch_idx >= batch_size
        or seq_idx >= seq_len
        or hidden_idx >= hidden_dim
    ):
        return

    # Compute statistics for this sequence position (redundant but simple)
    var sum_val: Scalar[dtype] = 0
    var sq_sum: Scalar[dtype] = 0

    @parameter
    for h in range(hidden_dim):
        val = input[batch_idx, seq_idx, h]
        sum_val += rebind[Scalar[dtype]](val)
        sq_sum += rebind[Scalar[dtype]](val * val)

    mean_val = sum_val / hidden_dim
    var_val = (sq_sum / hidden_dim) - (mean_val * mean_val)
    inv_std = 1.0 / sqrt(var_val + 1e-5)

    # Apply LayerNorm to this element
    input_val = input[batch_idx, seq_idx, hidden_idx]
    normalized = (input_val - mean_val) * inv_std * rebind[Scalar[dtype]](
        ln_weight[hidden_idx]
    ) + rebind[Scalar[dtype]](ln_bias[hidden_idx])
    output[batch_idx, seq_idx, hidden_idx] = normalized

```

The unfused implementation follows a straightforward approach where each thread handles one element of the output tensor. Let's break down the key components:

1. **Thread and Block Organization**:

   ```mojo
   batch_idx = block_idx.x
   seq_idx = block_idx.y
   hidden_idx = thread_idx.x
   ```

   - Each thread block handles one sequence position in the batch
   - Grid dimensions: `[batch_size, seq_len]`
   - Each thread processes one element in the hidden dimension
   - Early return if indices are out of bounds:

     ```mojo
     if (batch_idx >= batch_size or seq_idx >= seq_len or hidden_idx >= hidden_dim):
         return
     ```

2. **Statistics Computation**:

   ```mojo
   var sum_val: Scalar[dtype] = 0
   var sq_sum: Scalar[dtype] = 0

   @parameter
   for h in range(hidden_dim):
       val = input[batch_idx, seq_idx, h]
       sum_val += rebind[Scalar[dtype]](val)
       sq_sum += rebind[Scalar[dtype]](val * val)
   ```

   - Compute sum and squared sum in a single pass
   - Use `@parameter` for compile-time loop unrolling
   - Proper type casting with `rebind[Scalar[dtype]]`
   - Calculate mean and variance:

     ```mojo
     mean_val = sum_val / hidden_dim
     var_val = (sq_sum / hidden_dim) - (mean_val * mean_val)
     inv_std = 1.0 / sqrt(var_val + 1e-5)
     ```

3. **Normalization and Scaling**:

   ```mojo
   input_val = input[batch_idx, seq_idx, hidden_idx]
   normalized = (input_val - mean_val) * inv_std * rebind[Scalar[dtype]](
       ln_weight[hidden_idx]
   ) + rebind[Scalar[dtype]](ln_bias[hidden_idx])
   output[batch_idx, seq_idx, hidden_idx] = normalized
   ```

   - Apply normalization: \\[\Large \text{normalized} = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta \\]
   - Scale with learnable parameter `` (ln_weight)
   - Add learnable bias `` (ln_bias)
   - Store result in output tensor

4. **Performance Characteristics**:
   - Each thread computes statistics independently
   - No shared memory usage (simple but less efficient)
   - Memory access pattern:
     - Input: `[batch_idx, seq_idx, h]`
     - Output: `[batch_idx, seq_idx, hidden_idx]`
     - Parameters: `[hidden_idx]`
   - Numerical stability ensured by:
     - Adding epsilon (1e-5) before square root
     - Using proper type casting
     - Computing variance in a numerically stable way

5. **Implementation Details**:
   - **Type Safety**:
     - Use `Scalar[dtype]` for intermediate calculations
     - `rebind[Scalar[dtype]]` for proper type casting
     - Ensures consistent floating-point precision

   - **Memory Access**:
     - Coalesced reads from input tensor
     - Coalesced writes to output tensor
     - Sequential access to LayerNorm parameters

   - **Computation Flow**:
     - Statistics computation: \\[\Large O(H) \text{ operations per thread} \\]
     - Normalization: \\[\Large O(1) \text{ operations per thread} \\]
     - Total complexity: \\[\Large O(H) \text{ per output element} \\]

   - **Limitations**:
     - Redundant computation of statistics
     - No shared memory for intermediate results
     - High memory bandwidth usage
     - Multiple kernel launches required

This implementation is correct but not optimal for performance, as shown in the benchmark results where it's slightly slower than the CPU version. The fused implementation will address these performance limitations by:

- Computing statistics once per sequence
- Reusing normalized values
- Reducing memory traffic
- Eliminating intermediate tensor allocations

#### 2. Fused kernel implementation

The fused kernel combines LayerNorm and Linear operations into a single GPU kernel:

**Key optimizations:**

- Single kernel launch instead of two
- Shared memory for intermediate results
- Coalesced memory access patterns
- Reduced memory bandwidth usage
- No intermediate tensor allocations

#### Tips

1. **Thread organization**:
   - One thread block per sequence position (grid: `[batch_size, seq_len]`)
   - Single thread per sequence position to avoid redundancy
   - Compute all outputs for each sequence position in one thread

2. **Memory access**:
   - Access input tensor with `[batch_idx, seq_idx, h]`
   - Access output tensor with `[batch_idx, seq_idx, out_idx]`
   - Access weights with `[out_idx, h]` for linear layer

3. **Computation flow**:
   - Compute LayerNorm statistics once per sequence
   - Reuse normalized values for all output dimensions
   - Combine normalization and linear transformation

4. **Performance**:
   - Avoid redundant computation of statistics
   - Minimize memory traffic by fusing operations
   - Use proper type casting with `rebind[Scalar[dtype]]`

#### Running the code

To test your fused implementation, run:

  
    pixi NVIDIA (default)
    pixi AMD
    uv
  
  

```bash
pixi run p22 --fused
```

  
  

```bash
pixi run -e amd p22 --fused
```

  
  

```bash
uv run poe p22 --fused
```

  

Your output will look like this:

```txt
Testing with dimensions: [4, 4, 8] -> [4, 4, 16]
OK Loaded Mojo operations library
============================================================
   Puzzle 22: FUSED Algorithm Test & Benchmark
============================================================

 Correctness Testing for FUSED Algorithm
==================================================

Testing Reference PyTorch Implementation
-----------------------------------------------
OK Reference PyTorch
   Max difference: 0.00e+00
   Result: OK CORRECT

Testing CPU Implementation
---------------------------------
OK Using Mojo fused kernel (CPU)
   Max difference: 1.86e-08
   Result: OK CORRECT

Testing GPU Fused Implementation
---------------------------------------
OK Using Mojo fused kernel (GPU)
   Max difference: 1.86e-08
   Result: OK CORRECT

Correctness Summary:
   - Reference:   OK CORRECT
   - CPU:         OK CORRECT
   - GPU fused: OK CORRECT

   Overall Correctness: OK ALL CORRECT

 Benchmarking CPU vs GPU FUSED
----------------------------------------
   Testing CPU performance...
   CPU: 3144.75ms (50 iterations)
   Testing GPU fused performance...
   GPU fused: 3116.11ms (50 iterations)

   GPU fused vs CPU: 1.01x faster
   GPU fused wins!

FUSED Algorithm Test Completed!
```

### Reference implementation (example)


```mojo
fn minimal_fused_kernel
    input_layout: Layout,
    ln_params_layout: Layout,
    weight_layout: Layout,
    bias_layout: Layout,
    output_layout: Layout,
    batch_size: Int,
    seq_len: Int,
    hidden_dim: Int,
    output_dim: Int,
    dtype: DType = DType.float32,
:
    """Minimal fused kernel - one thread per sequence position to avoid redundancy.
    """
    # Grid: (batch_size, seq_len) - one thread block per sequence position
    # Block: (1,) - single thread per sequence position to avoid redundant computation
    batch_idx = Int(block_idx.x)
    seq_idx = Int(block_idx.y)

    if batch_idx >= batch_size or seq_idx >= seq_len:
        return

    # Step 1: Compute LayerNorm statistics once per sequence position
    var sum_val: Scalar[dtype] = 0
    var sq_sum: Scalar[dtype] = 0

    @parameter
    for h in range(hidden_dim):
        val = input[batch_idx, seq_idx, h]
        sum_val += rebind[Scalar[dtype]](val)
        sq_sum += rebind[Scalar[dtype]](val * val)

    mean_val = sum_val / hidden_dim
    var_val = (sq_sum / hidden_dim) - (mean_val * mean_val)
    inv_std = 1.0 / sqrt(var_val + 1e-5)

    # Step 2: Compute all outputs for this sequence position
    @parameter
    for out_idx in range(output_dim):
        var acc: Scalar[dtype] = 0

        @parameter
        for h in range(hidden_dim):
            input_val = input[batch_idx, seq_idx, h]
            normalized = (input_val - mean_val) * inv_std * rebind[
                Scalar[dtype]
            ](ln_weight[h]) + rebind[Scalar[dtype]](ln_bias[h])
            acc += rebind[Scalar[dtype]](normalized * linear_weight[out_idx, h])

        output[batch_idx, seq_idx, out_idx] = acc + rebind[Scalar[dtype]](
            linear_bias[out_idx]
        )

```

The fused implementation combines operations efficiently:

1. **Thread organization**:
   - One thread block per sequence position (grid: `[batch_size, seq_len]`)
   - Single thread per sequence position
   - Thread indices: `batch_idx = block_idx.x`, `seq_idx = block_idx.y`

2. **LayerNorm phase**:
   - Compute sum and squared sum for the sequence position
   - Calculate mean: \\[\Large \mu = \frac{1}{H} \sum_{i=1}^{H} x_i \\]
   - Calculate variance: \\[\Large \sigma^2 = \frac{1}{H} \sum_{i=1}^{H} (x_i - \mu)^2 \\]
   - Compute inverse standard deviation: \\[\Large \text{inv\_std} = \frac{1}{\sqrt{\sigma^2 + \epsilon}} \\]

3. **Linear phase**:
   - For each output dimension:
     - Compute normalized value: \\[\Large \text{normalized} = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta \\]
     - Multiply with linear weight and accumulate: \\[\Large \text{acc} = \sum_{h=1}^{H} \text{normalized}_h \cdot W_{out,h} \\]
     - Add linear bias: \\[\Large \text{output} = \text{acc} + b_{out} \\]
   - Store result in `output[batch_idx, seq_idx, out_idx]`

4. **Performance optimizations**:
   - Single kernel launch for both operations
   - Reuse computed statistics
   - Minimize memory traffic
   - No intermediate tensor allocations
   - Efficient memory access patterns

This implementation achieves better performance than the unfused version by reducing memory bandwidth usage and kernel launch overhead.

### Advantages of kernel fusion

In this puzzle, we've explored two approaches to implementing LayerNorm + Linear operations:

1. **Unfused implementation**:
   - Separate kernels for LayerNorm and Linear
   - Simpler implementation but less efficient
   - Higher memory bandwidth usage
   - Multiple kernel launches
   - Benchmark results: 3183.57ms (GPU)

2. **Fused implementation**:
   - Single kernel combining both operations
   - More complex but significantly more efficient
   - Reduced memory bandwidth usage
   - Single kernel launch
   - Benchmark results: 3116.11ms (GPU)

#### Memory bandwidth optimization

1. **Eliminated memory traffic**:
   - No intermediate tensor allocations between operations
   - Reduced global memory reads/writes
   - Reuse of normalized values for linear transformation
   - Memory bandwidth reduction: \\[\Large \text{reduction} = \frac{\text{unfused\_bandwidth} - \text{fused\_bandwidth}}{\text{unfused\_bandwidth}}\\]

2. **Cache efficiency**:
   - Better L1/L2 cache utilization
   - Reduced cache misses
   - Improved memory access patterns
   - Higher arithmetic intensity

#### Reduced overhead

1. **Kernel launch optimization**:
   - Single kernel launch instead of multiple
   - Lower driver overhead
   - Reduced synchronization points
   - Fewer memory allocations

2. **Resource management**:
   - Shared memory reuse between operations
   - Better register utilization
   - Improved thread occupancy
   - Higher GPU utilization

#### Performance characteristics

1. **Scalability**:
   - Better performance scaling with input size
   - Reduced memory bandwidth bottleneck
   - More efficient use of GPU resources
   - Improved throughput for large models

2. **Numerical efficiency**:
   - Maintained numerical stability
   - Reduced rounding errors
   - Better precision in intermediate results
   - Optimized computation order

 **Key insight**: Kernel fusion is particularly beneficial for operations that are frequently used together in neural networks, like LayerNorm + Linear in transformer architectures. The performance benefits become more significant with larger input sizes and more complex models.

##  Autograd Integration & Backward Pass

### Overview

In this puzzle, we explore the backward pass implementation of the fused LayerNorm + Linear operation. The backward pass computes gradients with respect to:

- Input tensor
- LayerNorm scale (\\(\gamma\\)) and shift (\\(\beta\\)) parameters
- Linear layer weight matrix and bias

The mathematical operations we're implementing are:

1. LayerNorm backward (details of derivation in [Detailed derivation of LayerNorm backward pass](#detailed-derivation-of-layernorm-backward-pass)):
\\[\Large \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \odot \gamma \odot \frac{1}{\sqrt{\sigma^2 + \epsilon}} (1 - \frac{1}{H} - \frac{(x - \mu)^2}{H(\sigma^2 + \epsilon)}) \\]

2. Linear backward:
\\[\Large \frac{\partial L}{\partial W} = \frac{\partial L}{\partial y}x^T \\]
\\[\Large \frac{\partial L}{\partial b} = \frac{\partial L}{\partial y} \\]
\\[\Large \frac{\partial L}{\partial x} = W^T\frac{\partial L}{\partial y} \\]

3. Chain Rule for Fused Operation:
\\[\Large \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y_{linear}} \frac{\partial y_{linear}}{\partial y_{norm}} \frac{\partial y_{norm}}{\partial x} \\]
where:

- \\(y_{norm}\\) is the LayerNorm output
- \\(y_{linear}\\) is the Linear layer output
- The chain rule ensures proper gradient flow through both operations

### Key concepts

- **Thread organization**:
  - One thread block per sequence position (grid: `[batch_size, seq_len]`)
  - Single thread per sequence position to avoid redundancy
  - Compute all gradients for each sequence position in one thread
  - Ensure proper thread synchronization for atomic operations

- **Memory access**:
  - Access input tensor with `[batch_idx, seq_idx, h]`
  - Access output tensor with `[batch_idx, seq_idx, out_idx]`
  - Access weights with `[out_idx, h]` for linear layer
  - Ensure memory alignment for atomic operations
  - Use shared memory for frequently accessed data

- **Computation flow**:
  - Compute LayerNorm statistics in same order as forward pass
  - Reuse normalized values for all output dimensions
  - Combine normalization and linear transformation
  - Maintain numerical stability throughout
  - Handle edge cases properly

- **Performance**:
  - Avoid redundant computation of statistics
  - Minimize memory traffic by fusing operations
  - Use proper type casting with `rebind[Scalar[dtype]]`
  - Ensure proper memory alignment
  - Optimize for autograd integration

### Configuration

- Batch size: `BATCH_SIZE = 4`
- Sequence length: `SEQ_LEN = 4`
- Hidden dimension: `HIDDEN_DIM = 8`
- Output dimension: `OUTPUT_DIM = 16`
- Epsilon: `EPS = 1e-5`
- Data type: `DType.float32`

### Implementation (challenging)

The fused backward kernel combines LayerNorm and Linear backward operations into a single GPU kernel. This is a challenging implementation that requires careful handling of:

- [Atomic operations](https://docs.modular.com/mojo/stdlib/os/atomic/Atomic/) for gradient accumulation
- Numerical stability in gradient computations
- Memory access patterns for efficient GPU utilization
- Proper synchronization between operations

**Key optimizations:**

- Single kernel launch for all gradient computations
- Atomic operations for safe gradient accumulation
- Coalesced memory access patterns
- Reduced memory bandwidth usage
- No intermediate tensor allocations

#### Tips

1. **Thread organization**:
   - One thread block per sequence position
   - Single thread per sequence position
   - Compute all gradients in one thread

2. **Memory access**:
   - Coalesced access for input/output tensors
   - Strided access for weight matrix
   - Proper alignment for atomic operations

3. **Computation flow**:
   - Compute statistics in same order as forward pass
   - Reuse normalized values
   - Maintain numerical stability

4. **Performance**:
   - Minimize memory traffic
   - Use proper type casting
   - Ensure proper alignment

#### Running the code

To test your fused backward implementation, run:

  
    pixi NVIDIA (default)
    pixi AMD
    uv
  
  

```bash
pixi run p22 --backward
```

  
  

```bash
pixi run -e amd p22 --backward
```

  
  

```bash
uv run poe p22 --backward
```

  

Your output will look like this:

```txt
Testing with dimensions: [4, 4, 8] -> [4, 4, 16]
OK Loaded Mojo operations library
============================================================
           Comprehensive Backward Pass Test
           Testing Custom LayerNorm + Linear Gradients
============================================================
Testing with dimensions: [4, 4, 8] -> [4, 4, 16]

Testing CPU Backward Pass:

Testing CPU Backward Implementation - Backward Pass
---------------------------------------------------------
   Computing PyTorch autograd reference...
   Computing Mojo backward implementation (CPU)...
OK CPU Backward Implementation backward completed
   Forward max difference: 1.49e-08
   grad_input: 2.98e-08 OK
   grad_ln_weight: 5.96e-08 OK
   grad_ln_bias: 2.38e-07 OK
   grad_linear_weight: 9.54e-07 OK
   grad_linear_bias: 0.00e+00 OK

   Forward pass: OK CORRECT
   Gradients:    OK CORRECT
   Overall:      OK CORRECT

Testing GPU Backward Pass:

Testing GPU Backward Implementation - Backward Pass
---------------------------------------------------------
   Computing PyTorch autograd reference...
   Computing Mojo backward implementation (GPU)...

OK GPU Backward Implementation backward completed
   Forward max difference: 1.86e-08
   grad_input: 4.47e-08 OK
   grad_ln_weight: 5.96e-08 OK
   grad_ln_bias: 3.58e-07 OK
   grad_linear_weight: 9.54e-07 OK
   grad_linear_bias: 0.00e+00 OK

   Forward pass: OK CORRECT
   Gradients:    OK CORRECT
   Overall:      OK CORRECT

Backward Pass Test Summary:
   - CPU Backward:  OK CORRECT
   - GPU Backward:  OK CORRECT

   Overall Result: OK ALL CORRECT

BACKWARD PASS Test Completed!
```

### Reference implementation (example)


```mojo
fn minimal_fused_kernel_backward
    grad_output_layout: Layout,
    input_layout: Layout,
    ln_params_layout: Layout,
    weight_layout: Layout,
    grad_input_layout: Layout,
    grad_ln_weight_layout: Layout,
    grad_ln_bias_layout: Layout,
    grad_weight_layout: Layout,
    grad_bias_layout: Layout,
    batch_size: Int,
    seq_len: Int,
    hidden_dim: Int,
    output_dim: Int,
    dtype: DType = DType.float32,
:
    """Fused backward kernel using atomic operations for safe gradient accumulation.
    """
    # Grid: (batch_size, seq_len) - one thread per sequence position
    # Block: (1,) - single thread per sequence position
    batch_idx = Int(block_idx.x)
    seq_idx = Int(block_idx.y)

    if batch_idx >= batch_size or seq_idx >= seq_len:
        return

    # Initialize gradient tensors to zero (block 0,0 only to avoid UB with atomic ops)
    if batch_idx == 0 and seq_idx == 0:
        # Initialize grad_ln_weight and grad_ln_bias
        @parameter
        for h in range(hidden_dim):
            (grad_ln_weight.ptr + h).init_pointee_copy(0)
            (grad_ln_bias.ptr + h).init_pointee_copy(0)

        # Initialize grad_weight and grad_bias
        @parameter
        for out_idx in range(output_dim):
            (grad_bias.ptr + out_idx).init_pointee_copy(0)

            @parameter
            for h in range(hidden_dim):
                (grad_weight.ptr + out_idx * hidden_dim + h).init_pointee_copy(
                    0
                )

    # Note: We cannot use barrier() here as it only synchronizes within a block.
    # The atomic operations will handle synchronization across blocks.

    # Step 1: Recompute forward pass statistics (needed for gradients)
    var sum_val: Scalar[dtype] = 0
    var sq_sum: Scalar[dtype] = 0

    @parameter
    for h in range(hidden_dim):
        val = input[batch_idx, seq_idx, h]
        sum_val += rebind[Scalar[dtype]](val)
        sq_sum += rebind[Scalar[dtype]](val * val)

    mean_val = sum_val / hidden_dim
    var_val = (sq_sum / hidden_dim) - (mean_val * mean_val)
    inv_std = 1.0 / sqrt(var_val + 1e-5)

    # Step 2: Atomically accumulate gradients w.r.t. linear bias
    @parameter
    for out_idx in range(output_dim):
        grad_bias_ptr = grad_bias.ptr + out_idx
        _ = Atomic[dtype].fetch_add(
            grad_bias_ptr,
            rebind[Scalar[dtype]](grad_output[batch_idx, seq_idx, out_idx]),
        )

    # Step 3: Atomically accumulate gradients w.r.t. linear weight
    @parameter
    for out_idx in range(output_dim):

        @parameter
        for h in range(hidden_dim):
            var input_val = input[batch_idx, seq_idx, h]
            var normalized = (input_val - mean_val) * inv_std
            var ln_output_val = normalized * rebind[Scalar[dtype]](
                ln_weight[h]
            ) + rebind[Scalar[dtype]](ln_bias[h])

            # Atomic gradient accumulation for linear weight
            var grad_w = (
                grad_output[batch_idx, seq_idx, out_idx] * ln_output_val
            )
            var grad_weight_ptr = grad_weight.ptr + out_idx * hidden_dim + h
            _ = Atomic.fetch_add(grad_weight_ptr, rebind[Scalar[dtype]](grad_w))

    # Step 4: Atomically accumulate gradients w.r.t. LayerNorm parameters
    @parameter
    for h in range(hidden_dim):
        input_val = input[batch_idx, seq_idx, h]
        normalized = (input_val - mean_val) * inv_std

        # Compute gradient w.r.t. LayerNorm output for this h
        var grad_ln_out: Scalar[dtype] = 0

        @parameter
        for out_idx in range(output_dim):
            grad_ln_out = grad_ln_out + rebind[Scalar[dtype]](
                grad_output[batch_idx, seq_idx, out_idx]
                * linear_weight[out_idx, h]
            )

        # Atomic accumulation of LayerNorm parameter gradients
        grad_ln_weight_ptr = grad_ln_weight.ptr + h
        grad_ln_bias_ptr = grad_ln_bias.ptr + h
        _ = Atomic[dtype].fetch_add(
            grad_ln_weight_ptr, rebind[Scalar[dtype]](grad_ln_out * normalized)
        )
        _ = Atomic[dtype].fetch_add(
            grad_ln_bias_ptr, rebind[Scalar[dtype]](grad_ln_out)
        )

    # Step 5: Compute gradients w.r.t. input (LayerNorm backward)
    # Compute sum terms needed for LayerNorm backward
    var sum_grad_normalized: Scalar[dtype] = 0
    var sum_grad_normalized_times_normalized: Scalar[dtype] = 0

    @parameter
    for h in range(hidden_dim):
        h_input_val = input[batch_idx, seq_idx, h]
        h_normalized = (h_input_val - mean_val) * inv_std

        var h_grad_ln_out: Scalar[dtype] = 0

        @parameter
        for out_idx in range(output_dim):
            h_grad_ln_out = h_grad_ln_out + rebind[Scalar[dtype]](
                grad_output[batch_idx, seq_idx, out_idx]
                * linear_weight[out_idx, h]
            )

        h_grad_norm = h_grad_ln_out * rebind[Scalar[dtype]](ln_weight[h])
        sum_grad_normalized = sum_grad_normalized + rebind[Scalar[dtype]](
            h_grad_norm
        )
        sum_grad_normalized_times_normalized = (
            sum_grad_normalized_times_normalized
            + rebind[Scalar[dtype]](h_grad_norm * h_normalized)
        )

    # Compute actual input gradients (no race conditions here - each thread writes to different positions)
    @parameter
    for h in range(hidden_dim):
        h_input_val = input[batch_idx, seq_idx, h]
        h_normalized = (h_input_val - mean_val) * inv_std

        var h_grad_ln_out: Scalar[dtype] = 0

        @parameter
        for out_idx in range(output_dim):
            h_grad_ln_out = h_grad_ln_out + rebind[Scalar[dtype]](
                grad_output[batch_idx, seq_idx, out_idx]
                * linear_weight[out_idx, h]
            )

        h_grad_norm = h_grad_ln_out * rebind[Scalar[dtype]](ln_weight[h])
        grad_input[batch_idx, seq_idx, h] = inv_std * (
            h_grad_norm
            - (sum_grad_normalized / hidden_dim)
            - (h_normalized * sum_grad_normalized_times_normalized / hidden_dim)
        )

```

The fused backward implementation combines operations efficiently:

1. **Thread organization and memory layout**:
   - Grid dimensions: `[batch_size, seq_len]` for one thread block per sequence position
   - Thread indices: `batch_idx = block_idx.x`, `seq_idx = block_idx.y`
   - Memory layout:
     - Input tensor: `[batch_size, seq_len, hidden_dim]`
     - Output tensor: `[batch_size, seq_len, output_dim]`
     - Weight matrix: `[output_dim, hidden_dim]`
     - Gradients: `[batch_size, seq_len, hidden_dim]` for input gradients
     - Parameter gradients: `[hidden_dim]` for LayerNorm, `[output_dim, hidden_dim]` for Linear

2. **LayerNorm backward phase**:
   - Recompute forward pass statistics in same order as forward pass:
     - Mean: \\[\Large \mu = \frac{1}{H} \sum_{i=1}^{H} x_i \\]
     - Variance: \\[\Large \sigma^2 = \frac{1}{H} \sum_{i=1}^{H} (x_i - \mu)^2 \\]
     - Inverse standard deviation: \\[\Large \text{inv\_std} = \frac{1}{\sqrt{\sigma^2 + \epsilon}} \\]
   - Compute normalized values: \\[\Large \hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \\]
   - Calculate gradients:
     - Input gradient: \\[\Large \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \odot \gamma \odot \frac{1}{\sqrt{\sigma^2 + \epsilon}} (1 - \frac{1}{H} - \frac{(x - \mu)^2}{H(\sigma^2 + \epsilon)}) \\]
     - Scale gradient: \\[\Large \frac{\partial L}{\partial \gamma} = \sum_{i=1}^{H} \frac{\partial L}{\partial y_i} \odot \hat{x}_i \\]
     - Shift gradient: \\[\Large \frac{\partial L}{\partial \beta} = \sum_{i=1}^{H} \frac{\partial L}{\partial y_i} \\]

3. **Linear backward phase**:
   - For each output dimension:
     - Bias gradient: \\[\Large \frac{\partial L}{\partial b} = \frac{\partial L}{\partial y} \\]
     - Weight gradient: \\[\Large \frac{\partial L}{\partial W} = \frac{\partial L}{\partial y}x^T \\]
     - Input gradient: \\[\Large \frac{\partial L}{\partial x} = W^T\frac{\partial L}{\partial y} \\]
   - Use atomic operations for gradient accumulation:
     - `atomic_add` for bias gradients with proper alignment
     - `atomic_add` for weight gradients with proper alignment
     - `atomic_add` for LayerNorm parameter gradients with proper alignment

4. **Memory access patterns**:
   - Coalesced access for input/output tensors
   - Strided access for weight matrix
   - Atomic operations for gradient accumulation
   - Shared memory for intermediate results
   - Register usage for frequently accessed values
   - Proper memory alignment for all operations

5. **Numerical stability**:
   - Careful handling of epsilon in denominator
   - Proper scaling of gradients
   - Stable computation of statistics
   - Type casting with `rebind[Scalar[dtype]]`
   - Proper handling of edge cases
   - Maintain same computation order as forward pass

6. **Performance optimizations**:
   - Single kernel launch for all operations
   - Reuse of computed statistics
   - Minimized memory traffic
   - No intermediate tensor allocations
   - Efficient thread utilization
   - Reduced synchronization points
   - Optimized memory access patterns
   - Proper memory alignment

7. **Implementation details**:
   - Use of `@parameter` for compile-time constants
   - Proper handling of tensor dimensions
   - Efficient type casting and conversions
   - Careful management of shared memory
   - Proper synchronization between operations
   - Error handling and boundary checks
   - Integration with PyTorch's autograd system

This implementation achieves better performance than the unfused version by:

- Reducing memory bandwidth usage through kernel fusion
- Minimizing kernel launch overhead
- Optimizing memory access patterns
- Efficient use of GPU resources
- Maintaining numerical stability
- Proper handling of gradient accumulation
- Ensuring proper memory alignment
- Efficient autograd integration

The fused backward pass is particularly important in transformer architectures where LayerNorm + Linear operations are frequently used together, making the performance benefits significant for real-world applications.

### Performance considerations

The backward pass implementation uses `torch.compile` with optimizations to minimize overhead:

```python
# Compilation configuration
torch._dynamo.config.cache_size_limit = 64  # Increase cache
torch._dynamo.config.suppress_errors = True  # Handle errors gracefully
torch._dynamo.config.automatic_dynamic_shapes = True  # Dynamic shapes
```

These optimizations are particularly important for the backward pass because:

- Small tensor operations benefit from compilation caching
- Dynamic shapes are common in backward passes
- Error handling needs to be robust for gradient computation
- Cache size helps with repeated backward operations
- Proper error handling is crucial for gradient computation
- Compilation overhead can significantly impact training time

The backward pass is compiled with `reduce-overhead` mode to minimize the compilation overhead while maintaining correctness. This is especially important because:

- Backward passes are called frequently during training
- Gradient computation needs to be numerically stable
- Memory access patterns need to be optimized
- Atomic operations require proper synchronization
- Autograd integration needs to be efficient

### Detailed derivation of LayerNorm backward pass

The backward pass gradient for LayerNorm is derived through careful application of the chain rule. Here's the step-by-step derivation:

#### Forward pass operations

- Mean: \\(\mu = \frac{1}{H} \sum_{i=1}^{H} x_i\\)
- Variance: \\(\sigma^2 = \frac{1}{H} \sum_{i=1}^{H} (x_i - \mu)^2\\)
- Normalized value: \\(\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}\\)
- Final output: \\(y = \gamma \odot \hat{x} + \beta\\)

#### Chain rule application

To compute \\(\frac{\partial L}{\partial x}\\), we apply the chain rule:
\\[\Large \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial \hat{x}} \frac{\partial \hat{x}}{\partial x}\\]

#### Gradient components

##### Output to normalized value

- \\(\frac{\partial y}{\partial \hat{x}} = \gamma\\) (element-wise multiplication)

##### Normalized value to input

The gradient \\(\frac{\partial \hat{x}}{\partial x}\\) has three components:

- Direct effect through numerator: \\(\frac{1}{\sqrt{\sigma^2 + \epsilon}}\\)
- Indirect effect through mean: \\(-\frac{1}{H} \frac{1}{\sqrt{\sigma^2 + \epsilon}}\\)
- Indirect effect through variance: \\(-\frac{(x - \mu)}{H(\sigma^2 + \epsilon)^{3/2}} (x - \mu)\\)

#### Combining terms

The gradient through the normalization term can be simplified to:
\\[\Large \frac{\partial \hat{x}}{\partial x} = \frac{1}{\sqrt{\sigma^2 + \epsilon}} (1 - \frac{1}{H} - \frac{(x - \mu)^2}{H(\sigma^2 + \epsilon)})\\]

#### Final gradient expression

Combining all terms:
\\[\Large \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \odot \gamma \odot \frac{1}{\sqrt{\sigma^2 + \epsilon}} (1 - \frac{1}{H} - \frac{(x - \mu)^2}{H(\sigma^2 + \epsilon)})\\]

#### Key insights

- The chain rule accounts for all paths through which x affects the output
- The normalization term \\(\sqrt{\sigma^2 + \epsilon}\\) appears in both numerator and denominator
- The mean and variance terms create additional paths for gradient flow
- The final expression combines all effects into a single efficient computation

#### Implementation considerations

- The gradient properly accounts for the scaling effect of \\(\gamma\\)
- The normalization effect of mean and variance is preserved
- The numerical stability term \\(\epsilon\\) is maintained
- Gradients are properly scaled across the hidden dimension H
- The computation order matches the forward pass for numerical stability

This derivation ensures that the backward pass maintains the same numerical properties as the forward pass while efficiently computing all necessary gradients.
