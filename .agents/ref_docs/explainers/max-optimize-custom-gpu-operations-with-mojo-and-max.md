---
title: "Optimize Custom GPU Operations with Mojo and MAX"
description: "Learn strategies to improve GPU custom op performance in MAX by optimizing a matrix multiplication kernel using Mojo."
---

# Optimize Custom GPU Operations with Mojo and MAX

Learn strategies to improve GPU custom op performance in MAX by optimizing a matrix multiplication kernel using Mojo.

Building high-performance AI workloads for GPUs can be a daunting task, but Modular simplifies the experience with our
custom op API that allows you to build graph operators for both GPUs and CPUs. In this tutorial, we'll teach you some
strategies you can use to improve the performance of your GPU custom ops written in Mojo.

For demonstration purposes, we'll show you how to incrementally improve the performance of a custom matrix
multiplication (matmul) op. We're not teaching you how to build a matmul op, because MAX already contains leading-edge
implementations of matmul that you can use with the MAX graph API. Rather, we're using a basic matmul operation (AKA "
GPU kernel") to teach you GPU programming strategies that might help with your other GPU code written in Mojo.

As you progress through this tutorial, you'll learn the following:

- How to define a custom matrix multiplication operation for a MAX graph.
- How to use Mojo high-performance GPU programming abstractions to progressively optimize a matrix multiplication.
- How to access GPU hardware features, like Tensor Cores, from MAX.

## Requirements

To use a GPU, your system must meet the GPU requirements.

## Run and compare the results

To get a sense of how each implementation of the custom op (kernel) performs, download the code and run the benchmark
script:

1. Get the example code from GitHub:

```bash
git clone https://github.com/modular/modular.git
```

2. If you don't have it, install `pixi`:

```sh
curl -fsSL https://pixi.sh/install.sh | sh
```

Then restart your terminal for the changes to take effect.

3. Run all the matmul examples:

```bash
cd modular/examples/custom_ops
pixi run python matrix_multiplication.py
```

As long as you have a compatible GPU, this will compile, run, and print the results of each matmul implementation that
we'll discuss below.

4. Run the `matmul` benchmarks to see the impact of each optimization:

```bash
pixi run mojo benchmarks.mojo --matmul
```

This also runs each implementation but also benchmarks them and prints a comparison table. For example (this is running
on a `g5-2xlarge` instance; your results will vary):

```output
   ---------------------------------------------------------------------------------------------------------
| name                       | met (ms)            | iters | GFLOPS/s           | GElems/s              |
   ---------------------------------------------------------------------------------------------------------
| cpu/naive                  | 1647.331583         | 2     | 1.3183084343256959 | 0.0006415126201098278 |
| gpu/naive                  | 2.842315817535545   | 422   | 764.0569378680037  | 0.37180386270949084   |
| gpu/coalescing             | 1.0952283930000002  | 1000  | 1982.8659792606384 | 0.9648982867448362    |
| gpu/tiled                  | 0.981560302         | 1000  | 2212.48874427279   | 1.076636858526905     |
| gpu/tiled_register         | 0.39235773577501637 | 3058  | 5534.977195518529  | 2.693419559863031     |
| gpu/block_tiled            | 0.38266733939393943 | 3135  | 5675.141033565811  | 2.7616258070879858    |
| gpu/block_tiled_vectorized | 0.3684924709677419  | 3255  | 5893.447739370804  | 2.8678577807157195    |
| gpu/tensor_core            | 0.18174263374734928 | 6602  | 11949.266252072646 | 5.814728103198369     |
   ---------------------------------------------------------------------------------------------------------
```

## Introduction

AI models in MAX are built as computational graphs using the MAX graph API. MAX contains within it a powerful graph
compiler that can take these graphs and optimize them for best performance on a wide range of hardware.

Each node in a MAX graph is defined by an operation that performs a calculation on zero or more inputs and produces one
or more outputs. These inputs and outputs tend to be in the form of tensors, and the operations are usually
data-parallel calculations that are accelerated on CPUs or GPUs. In MAX, these operations are written using Mojo, a
Python-family language built for high-performance computation.

Matrix multiplications are key components in modern AI models, accounting for a sizable fraction of the GPU workload
when running these models. Optimizations applied to matrix multiplication calculations can have a significant impact on
the throughput of models on GPUs.

To review, a matrix multiplication involves multiplying two matrices, A and B, to produce a new matrix C. Each value in
the output matrix is the dot product of a row from A and a column from B. In a worst case scenario, when multiplying an
MxK matrix by a KxN matrix, calculating one output value requires loading `2 * K` values and performing `K`
floating-point multiplications.

### Structure of the custom operation

The matrix multiplication algorithms demonstrated here are encapsulated within a custom MAX graph operation. AI models
in MAX are built from a graph of operations like this, and in this case we're demonstrating one of these operations
running in isolation. The `matrix_multiplication.py` file exercises seven different matrix multiplication algorithms
using a single-operation graph and shows that the results of multiplying two random matrices are the same for each.

The single-operation graph is constructed using the following function:

```python
def matrix_multiplication(
        a: NDArray[np.float32],
        b: NDArray[np.float32],
        algorithm: str,
        session: InferenceSession,
        device: Device,
) -> Buffer:
    dtype = DType.float32

    a_tensor = Buffer.from_numpy(a).to(device)
    b_tensor = Buffer.from_numpy(b).to(device)

    mojo_kernels = Path(__file__).parent / "kernels"

    with Graph(
            "matrix_multiplication_graph",
            input_types=[
                TensorType(
                    dtype,
                    shape=a_tensor.shape,
                    device=DeviceRef.from_device(device),
                ),
                TensorType(
                    dtype,
                    shape=b_tensor.shape,
                    device=DeviceRef.from_device(device),
                ),
            ],
            custom_extensions=[mojo_kernels],
    ) as graph:
        a_value, b_value = graph.inputs
        output = ops.custom(
            name="matrix_multiplication",
            device=DeviceRef.from_device(device),
            values=[a_value, b_value],
            out_types=[
                TensorType(
                    dtype=a_value.tensor.dtype,
                    shape=[a_value.tensor.shape[0], b_value.tensor.shape[1]],
                    device=DeviceRef.from_device(device),
                )
            ],
            parameters={"algorithm": algorithm},
        )[0].tensor
        graph.output(output)

    print("Compiling...")
    model = session.load(graph)

    print("Executing...")
    result = model.execute(a_tensor, b_tensor)[0]
    return result.to(CPU())
```

A single `matrix_multiplication` operation is used, and the algorithm variant is specified by the `algorithm`
compile-time parameter.

The custom operation itself is defined in Mojo within the `operations/matrix_multiplication.mojo` file. The
`MatrixMultiplication` struct hosts all the setup code for taking in the matrix tensors, branching execution based on
whether the operation is running on CPU or GPU, and then selecting and running a specific algorithm. Mojo supports
compile-time specialization of code based on parameters like target hardware, and that is also extended here to
user-supplied algorithm choice.

### Matrix multiplication algorithms

The algorithms demonstrated in this example follow a progression of optimizations:

1. **naive**: Naive matrix multiplication with no optimizations.
1. **coalescing**: Applying memory coalescing.
1. **tiled**: Reworking to use shared memory tiling.
1. **tiled_register**: Using shared memory tiling and register tiling.
1. **block_tiled**: Introducing block tiling.
1. **block_tiled_vectorized**: Block tiling with vectorized memory access.
1. **tensor_core**: Using Tensor Cores for matrix multiplication.

The results on an NVIDIA A100 GPU for 32-bit floats and input matrices sized to 4096x4096 look like the following:

| Algorithm              | GFLOPS/s |
|------------------------|----------|
| naive                  | 292      |
| coalescing             | 2936     |
| tiled                  | 3943     |
| tiled_register         | 7078     |
| block_tiled            | 10661    |
| block_tiled_vectorized | 10663    |

### Layouts and LayoutTensor

The `matrix_multiplication` custom operation uses layouts and `LayoutTensor` to represent the input and output matrices.
A layout represents a mapping from a set of logical coordinates to a single, one-dimensional coordinate.

```mojo
my_layout = Layout.row_major(2, 6)
print_layout(my_layout)
```

```plaintext
       0    1    2    3    4    5
    +----+----+----+----+----+----+
 0  |  0 |  1 |  2 |  3 |  4 |  5 |
    +----+----+----+----+----+----+
 1  |  6 |  7 |  8 |  9 | 10 | 11 |
    +----+----+----+----+----+----+
```

A `LayoutTensor` consists of a layout and a pointer to memory. One `LayoutTensor` method used frequently is `tile()`,
which returns a new `LayoutTensor` that is a subset of the original, pointing to the same underlying data.

## Kernel 1: Naive matrix multiplication with no optimizations

A basic matrix multiplication in Mojo looks like the following:

```mojo
fn naive_matrix_multiplication[
    dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    BM: Int,
    BN: Int,
](
    a: LayoutTensor[dtype, a_layout, MutAnyOrigin],
    b: LayoutTensor[dtype, b_layout, MutAnyOrigin],
    c: LayoutTensor[dtype, c_layout, MutAnyOrigin],
):
    var M = a.dim[0]()
    var N = b.dim[1]()
    var K = b.dim[0]()

    var row = block_dim.x * block_idx.x + thread_idx.x
    var col = block_dim.y * block_idx.y + thread_idx.y

    var dst_reg: c.element_type = 0

    if row < UInt(M) and col < UInt(N):
        for k_index in range(K):
            dst_reg = dst_reg + a[row, k_index] * b[k_index, col]

    c[row, col] = dst_reg
```

## Kernel 2: Applying memory coalescing

Global memory accesses can be coalesced by swapping the thread indices for columns and rows:

```mojo
var row = block_dim.y * block_idx.y + thread_idx.y
var col = block_dim.x * block_idx.x + thread_idx.x
```

With this change, adjacent threads access values in the same row of the input matrices, which are contiguous in memory.

## Kernel 3: Reworking to use shared memory tiling

Shared memory on the GPU is far faster to access than global memory. The input matrices A and B are loaded into shared
memory in tiles of size BM x BK and BK x BN.

```mojo
var col = thread_idx.x % UInt(BN)
var row = thread_idx.x // UInt(BN)

var dst = c.tile[BM, BN](Int(block_idx.y), Int(block_idx.x))

var a_smem = LayoutTensor[
    dtype,
    Layout.row_major(BM, BK),
    MutAnyOrigin,
    address_space = AddressSpace.SHARED,
].stack_allocation()
var b_smem = LayoutTensor[
    dtype,
    Layout.row_major(BK, BN),
    MutAnyOrigin,
    address_space = AddressSpace.SHARED,
].stack_allocation()

var dst_reg: c.element_type = 0

for block in range(b.dim[0]() // BK):
    comptime load_a_layout = Layout.row_major(NUM_THREADS // BK, BK)
    comptime load_b_layout = Layout.row_major(BK, NUM_THREADS // BK)

    var a_tile = a.tile[BM, BK](Int(block_idx.y), block)
    var b_tile = b.tile[BK, BN](block, Int(block_idx.x))

    copy_dram_to_sram_async[thread_layout=load_a_layout](a_smem, a_tile)
    copy_dram_to_sram_async[thread_layout=load_b_layout](b_smem, b_tile)

    async_copy_wait_all()
    barrier()

    @parameter
    for k in range(BK):
        dst_reg += a_smem[row, k] * b_smem[k, col]

    barrier()

dst[row, col] += dst_reg
```

## Kernel 4: Using shared memory tiling and register tiling

In this version, each thread is responsible for calculating multiple values of C, further reducing the memory bandwidth
required.

```mojo
var col = thread_idx.x % UInt(BN)
var row = thread_idx.x // UInt(BN)

var dst = c.tile[BM, BN](Int(block_idx.y), Int(block_idx.x)).tile[TM, 1](
    Int(row), Int(col)
)

var a_smem = tb[dtype]().row_major[BM, BK]().shared().alloc()
var b_smem = tb[dtype]().row_major[BK, BN]().shared().alloc()

var dst_reg = tb[dtype]().layout[TM]().local().alloc()
dst_reg.copy_from(dst)

for block in range(b.dim[0]() // BK):
    comptime load_a_layout = Layout.row_major(NUM_THREADS // BK, BK)
    comptime load_b_layout = Layout.row_major(BK, NUM_THREADS // BK)

    var a_tile = a.tile[BM, BK](Int(block_idx.y), block)
    var b_tile = b.tile[BK, BN](block, Int(block_idx.x))

    copy_dram_to_sram_async[thread_layout=load_a_layout](a_smem, a_tile)
    copy_dram_to_sram_async[thread_layout=load_b_layout](b_smem, b_tile)

    async_copy_wait_all()
    barrier()

    @parameter
    for k in range(BK):
        var a_tile = a_smem.tile[TM, 1](Int(row), k)
        var b_tile = b_smem.tile[1, BN](k, 0)
        var b_val = b_tile[0, col]

        @parameter
        for t in range(TM):
            dst_reg[t] += a_tile[t, 0] * b_val

    barrier()

dst.copy_from(dst_reg)
```

## Kernel 5: Introducing block tiling

This kernel uses a 2-D block tiling strategy where each thread calculates a `TM`x`TN` tile of the output tensor.

```mojo
var partition_col = Int(thread_idx.x % UInt(BN // TN))
var partition_row = Int(thread_idx.x // UInt(BN // TN))

var dst = c.tile[BM, BN](Int(block_idx.y), Int(block_idx.x)).tile[TM, TN](
    partition_row, partition_col
)

var a_smem = tb[dtype]().row_major[BM, BK]().shared().alloc()
var b_smem = tb[dtype]().row_major[BK, BN]().shared().alloc()

var dst_reg = tb[dtype]().row_major[TM, TN]().local().alloc()
dst_reg.copy_from(dst)
var a_reg = tb[dtype]().layout[TM]().local().alloc()
var b_reg = tb[dtype]().layout[TN]().local().alloc()

for block in range(ntiles):
    comptime load_a_layout = Layout.row_major(NUM_THREADS // BK, BK)
    comptime load_b_layout = Layout.row_major(BK, NUM_THREADS // BK)
    var a_tile = a.tile[BM, BK](Int(block_idx.y), block)
    var b_tile = b.tile[BK, BN](block, Int(block_idx.x))
    copy_dram_to_sram_async[thread_layout=load_a_layout](a_smem, a_tile)
    copy_dram_to_sram_async[thread_layout=load_b_layout](b_smem, b_tile)

    async_copy_wait_all()
    barrier()

    @parameter
    for k in range(BK):
        var a_tile = a_smem.tile[TM, 1](partition_row, k)
        var b_tile = b_smem.tile[1, TN](k, partition_col)
        a_reg.copy_from(a_tile)
        b_reg.copy_from(b_tile)
        outer_product_acc(dst_reg, a_reg, b_reg)
    barrier()

dst.copy_from(dst_reg)
```

## Kernel 6: Block tiling with vectorized memory access

Memory accesses can be vectorized to improve bandwidth using the `LayoutTensor.vectorize()` method.

```mojo
from sys.info import simd_width_of

comptime simd_width = simd_width_of[dtype]()
var partition_col = Int(thread_idx.x % UInt(BN // TN))
var partition_row = Int(thread_idx.x // UInt(BN // TN))

var dst = c.tile[BM, BN](Int(block_idx.y), Int(block_idx.x)).tile[TM, TN](
    partition_row, partition_col
)
var dst_vec = dst.vectorize[1, simd_width]()

var a_smem = tb[dtype]().col_major[BM, BK]().shared().alloc()
var b_smem = tb[dtype]().row_major[BK, BN]().shared().alloc()

var dst_reg = tb[dtype]().row_major[TM, TN]().local().alloc()
var dst_reg_vec = dst_reg.vectorize[1, simd_width]()
dst_reg_vec.copy_from(dst_vec)

for block in range(ntiles):
    comptime load_a_layout = Layout.row_major(NUM_THREADS // BK, BK)
    comptime load_b_layout = Layout.row_major(BK, NUM_THREADS // BK)
    var a_tile = a.tile[BM, BK](Int(block_idx.y), block)
    var b_tile = b.tile[BK, BN](block, Int(block_idx.x))

    copy_dram_to_sram_async[thread_layout=load_a_layout](
        a_smem.vectorize[simd_width, 1](), a_tile.vectorize[simd_width, 1]()
    )
    copy_dram_to_sram_async[thread_layout=load_b_layout](
        b_smem.vectorize[1, simd_width](), b_tile.vectorize[1, simd_width]()
    )

    async_copy_wait_all()
    barrier()

    @parameter
    for k in range(BK):
        var a_tile = a_smem.tile[TM, 1](partition_row, k)
        var b_tile = b_smem.tile[1, TN](k, partition_col)
        a_reg.copy_from(a_tile)
        b_reg.copy_from(b_tile)
        outer_product_acc(dst_reg, a_reg, b_reg)

    barrier()

dst_vec.copy_from(dst_reg_vec)
```

## Kernel 7: Using Tensor Cores for matrix multiplication

Modern GPUs have dedicated hardware units for performing accelerated matrix multiplication called Tensor Cores. MAX
contains interfaces that make it ergonomic to program these units.

```mojo
fn tensor_core_matrix_multiplication[
    dtype: DType,
    layout_a: Layout,
    layout_b: Layout,
    layout_c: Layout,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    MMA_M: Int,
    MMA_N: Int,
    MMA_K: Int,
](
    A: LayoutTensor[dtype, layout_a, MutAnyOrigin],
    B: LayoutTensor[dtype, layout_b, MutAnyOrigin],
    C: LayoutTensor[dtype, layout_c, MutAnyOrigin],
):
    comptime M = C.shape[0]()
    comptime N = C.shape[1]()
    comptime K = A.shape[1]()

    warp_y = warp_id() // UInt(BN // WN)
    warp_x = warp_id() % UInt(BN // WN)

    C_warp_tile = C.tile[BM, BN](Int(block_idx.y), Int(block_idx.x)).tile[WM, WN](
        Int(warp_y), Int(warp_x)
    )

    mma_op = TensorCore[A.dtype, C.dtype, Index(MMA_M, MMA_N, MMA_K)]()

    A_sram_tile = tb[A.dtype]().row_major[BM, BK]().shared().alloc()
    B_sram_tile = tb[B.dtype]().row_major[BK, BN]().shared().alloc()

    c_reg = (
        tb[C.dtype]()
        .row_major[WM // MMA_M, (WN * 4) // MMA_N]()
        .local()
        .alloc()
        .fill(0)
    )

    for k_i in range(K // BK):
        barrier()

        A_dram_tile = A.tile[BM, BK](Int(block_idx.y), k_i)
        B_dram_tile = B.tile[BK, BN](k_i, Int(block_idx.x))

        copy_dram_to_sram_async[thread_layout = Layout.row_major(4, 8)](
            A_sram_tile.vectorize[1, 4](), A_dram_tile.vectorize[1, 4]()
        )
        copy_dram_to_sram_async[thread_layout = Layout.row_major(4, 8)](
            B_sram_tile.vectorize[1, 4](), B_dram_tile.vectorize[1, 4]()
        )

        async_copy_wait_all()
        barrier()

        A_warp_tile = A_sram_tile.tile[WM, BK](Int(warp_y), 0)
        B_warp_tile = B_sram_tile.tile[BK, WN](0, Int(warp_x))

        @parameter
        for mma_k in range(BK // MMA_K):
            @parameter
            for mma_m in range(WM // MMA_M):
                @parameter
                for mma_n in range(WN // MMA_N):
                    c_reg_m_n = c_reg.tile[1, 4](mma_m, mma_n)
                    A_mma_tile = A_warp_tile.tile[MMA_M, MMA_K](mma_m, mma_k)
                    B_mma_tile = B_warp_tile.tile[MMA_K, MMA_N](mma_k, mma_n)

                    a_reg = mma_op.load_a(A_mma_tile)
                    b_reg = mma_op.load_b(B_mma_tile)

                    var d_reg_m_n = mma_op.mma_op(
                        a_reg,
                        b_reg,
                        c_reg_m_n,
                    )
                    c_reg_m_n.copy_from(d_reg_m_n)

    @parameter
    for mma_m in range(WM // MMA_M):
        @parameter
        for mma_n in range(WN // MMA_N):
            var C_mma_tile = C_warp_tile.tile[MMA_M, MMA_N](mma_m, mma_n)
            var c_reg_m_n = c_reg.tile[1, 4](mma_m, mma_n)
            mma_op.store_d(C_mma_tile, c_reg_m_n)
```
