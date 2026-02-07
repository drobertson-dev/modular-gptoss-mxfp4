---
title: "Build Custom GPU Operations for MAX Graphs with Mojo"
description: "A tutorial on writing hardware-independent custom operations in Mojo and integrating them into Python-based MAX computation graphs."
---

# Build Custom GPU Operations for MAX Graphs with Mojo

A tutorial on writing hardware-independent custom operations in Mojo and integrating them into Python-based MAX computation graphs.

Mojo is our not-so-secret weapon to achieve architecture-independent performance for all types of AI workloads. In this
tutorial, you'll learn to write custom graph operations (ops) in Mojo that run on GPUs and CPUs, and then load them into
a MAX graph written in Python.

We'll start with a simple custom op that just adds `1` to every element in the graph's input tensor, using an API that
abstracts-away the CPU and GPU device management. Then you'll learn to write specialized functions for CPUs and GPUs. (
GPU functions that run in parallel are also known as *kernels*.)

Before you begin, you should have a basic understanding of MAX graphs, which are computation graphs written in Python.
These graphs are the foundation for high-performance AI models that run on MAX. To learn more, see our tutorial to get
started with MAX graphs in Python. It also helps if you know the Mojo language basics.

## Requirements

Although these examples work on both CPUs and GPUs, in order for the GPU code paths to run, your system must meet the
GPU requirements.

## Get the examples

This tutorial is a walkthrough of a couple code examples from our GitHub repo. Start by cloning and running one of the
examples:

1. Clone the repo:

```sh
git clone https://github.com/modular/modular.git
```

2. To ensure you have a compatible developer environment, we recommend using `pixi` to create a virtual environment and
   manage the package dependencies. If you don't have it, install it:

```sh
curl -fsSL https://pixi.sh/install.sh | sh
```

Then restart your terminal for the changes to take effect.

3. Make sure everything works by running the first example:

```sh
cd modular/examples/custom_ops

pixi run python addition.py
```

The exact output will vary based on random initialization of the input tensor, but the "Graph result" and "Expected
result" should match:

```output
Graph result:
     [[1.7736697 1.4688652 1.7971799 1.4553597 1.8967733 1.3691401 1.1297637
     1.7047229 1.1314526 1.3924606]
       # ... shorten for brevity
Expected result:
     [[1.7736697 1.4688652 1.7971799 1.4553597 1.8967733 1.3691401 1.1297637
     1.7047229 1.1314526 1.3924606]
       # ... shorten for brevity
```

## Example 1: Learn the custom op basics

To learn how to create a custom op, let's look as simple "hello world" example that adds `1` to each element of a
tensor-doing so in parallel on a GPU, if available.

### Define the custom op in Mojo

Take a look at the custom op defined in `custom_ops/kernels/add_one.mojo`. You'll see a Mojo struct called `AddOne` with
an `execute()` function. Every custom op must be defined with this general format, as described below.

Depending on the purpose of your custom op, the `execute()` function will accept zero or more inputs and produce one or
more outputs, as specified by the function arguments.

Let's inspect the struct and function signatures:

**kernels/add_one.mojo**

```mojo
@compiler.register("add_one")
struct AddOne:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor,
        x: InputTensor[dtype = output.dtype, rank = output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        # See below for the rest
```

The struct must include the `@compiler.register()` decorator, which registers the custom op with MAX. The `add_one` name
we set here is the name we'll use to add the op to our Python graph in the next section.

The rest of the `execute()` signature describes the custom op's graph node to the graph compiler: the parameters,
inputs, and outputs:

- There's one compile-time parameter, `target`, which tells the function what kind of hardware it's being compiled for (
  either `"cpu"` or `"gpu"`; we'll use this in the next example).

- The runtime arguments include the op's inputs and outputs, which take the form of `InputTensor` and `OutputTensor`,
  respectively. These are specialized versions of the `ManagedTensorSlice` type, which represents a tensor of a specific
  rank and datatype whose memory is managed outside of the operation.

The `execute()` function must include `target` as the first parameter and `output` as the first argument.

Now let's look at the body of the `execute()` function. The op's core computation that adds `1` to each element in the
tensor happens in the `elementwise_add_one()` closure function.

**kernels/add_one.mojo**

```mojo
@compiler.register("add_one")
struct AddOne:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor,
        x: InputTensor[dtype = output.dtype, rank = output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn elementwise_add_one[
            width: Int
        ](idx: IndexList[x.rank]) -> SIMD[x.dtype, width]:
            return x.load[width](idx) + 1

        foreach[elementwise_add_one, target=target](output, ctx)
```

We call `elementwise_add_one()` using `foreach()`, which distributes an elementwise computation in parallel across all
elements in the output tensor. At compile time, `foreach()` optimizes the computation for the hardware its running on,
optimally distributing parallel workloads to make the most efficient use of computational resources. This means that
this same code runs with optimal performance on CPU or GPU with no changes required.

Also notice that we pass the `target` parameter to the `foreach()` function, which allows it to optimize for the
hardware device.

### Add the custom op to a Python graph

Let's now look at the corresponding `custom_ops/addition.py` file where we load the `add_one` custom op into a graph
using `ops.custom()`.

Here's the code that specifies the path to the Mojo custom op and adds it to the graph:

**addition.py**

```python
mojo_kernels = Path(__file__).parent / "kernels"

rows = 5
columns = 10
dtype = DType.float32
device = CPU() if accelerator_count() == 0 else Accelerator()

graph = Graph(
    "addition",
    forward=lambda x: ops.custom(
        name="add_one",
        device=DeviceRef.from_device(device),
        values=[x],
        out_types=[
            TensorType(
                dtype=x.dtype,
                shape=x.tensor.shape,
                device=DeviceRef.from_device(device),
            )
        ],
    )[0].tensor,
    input_types=[
        TensorType(
            dtype,
            shape=[rows, columns],
            device=DeviceRef.from_device(device),
        ),
    ],
    custom_extensions=[mojo_kernels],
)
```

Make sure the directory you pass to `custom_extensions` is a Mojo package containing an `__init__.mojo` file (which can
be empty).

The `Graph()` takes an input tensor with five rows and ten columns, runs the custom `add_one` operation on it, and
returns the result.

Now we can run an inference, using `InferenceSession`. We start by loading the graph onto the selected `device`:

**addition.py**

```python
session = InferenceSession(
    devices=[device],
)

model = session.load(graph)
```

Finally, we generate some random data and pass it as input:

**addition.py**

```python
# Fill an input matrix with random values.
x_values = np.random.uniform(size=(rows, columns)).astype(np.float32)

# Create a buffer and move it to the device (CPU or GPU).
x = Buffer.from_numpy(x_values).to(device)

# Run inference with the input tensor.
result = model.execute(x)[0]

# Copy values back to the CPU to be read.
assert isinstance(result, Buffer)
result = result.to(CPU())
```

Notice that the `Buffer` is initially resident on the host (CPU), so we move it to the accelerator to be ready for use
with the graph on that device. Likewise, after we get results, we move the result back to the CPU to read it.

## Example 2: Write device-specific kernels

Now let's write some GPU code of our own.

The `add_one` custom op above uses `foreach()` to run our custom op computation. This function is a device-independent
abstraction to perform calculations on each element of a tensor. However, there might also be situations in which you
want to write your own hardware-specific algorithms.

Let's look at the `vector_addition.mojo` custom op, which adds two vectors together in parallel on a GPU.

This example uses a programming model that may be familiar if you've used general purpose GPU programming in CUDA: We're
going to write separate functions for GPUs and CPUs, and the GPU function (kernel) is written specifically for
parallelism across GPU threads.

The vector addition op in `kernels/vector_addition.mojo` looks like this:

**kernels/vector_addition.mojo**

```mojo
@compiler.register("vector_addition")
struct VectorAddition:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor[rank=1],
        lhs: InputTensor[dtype = output.dtype, rank = output.rank],
        rhs: InputTensor[dtype = output.dtype, rank = output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if target == "cpu":
            _vector_addition_cpu(output, lhs, rhs, ctx)
        elif target == "gpu":
            _vector_addition_gpu(output, lhs, rhs, ctx)
        else:
            raise Error("No known target:", target)
```

Using the parametric `if`, Mojo checks at compile time if the target device is a GPU or not. If it is, it uses the
`_vector_addition_gpu()` function; otherwise, it uses `_vector_addition_cpu()`.

The `_vector_addition_gpu()` function looks like this:

**kernels/vector_addition.mojo**

```mojo
fn _vector_addition_gpu(
    output: ManagedTensorSlice[mut=True],
    lhs: ManagedTensorSlice[dtype = output.dtype, rank = output.rank],
    rhs: ManagedTensorSlice[dtype = output.dtype, rank = output.rank],
    ctx: DeviceContextPtr,
) raises:
    alias BLOCK_SIZE = 16
    var gpu_ctx = ctx.get_device_context()
    var vector_length = output.dim_size(0)

    @parameter
    fn vector_addition_gpu_kernel(length: Int):
        var tid = block_dim.x * block_idx.x + thread_idx.x
        if tid < UInt(length):
            var idx = IndexList[output.rank](Int(tid))
            var result = lhs.load[1](idx) + rhs.load[1](idx)
            output.store[1](idx, result)

    var num_blocks = ceildiv(vector_length, BLOCK_SIZE)
    gpu_ctx.enqueue_function_experimental[vector_addition_gpu_kernel](
        vector_length, grid_dim=num_blocks, block_dim=BLOCK_SIZE
    )
```

The `vector_addition_gpu_kernel()` closure function runs once per thread on the GPU, adding an element from the `lhs`
vector to the matching element in the `rhs` vector and then saving the result at the correct position in the `output`
vector. This function is then run across a grid of `BLOCK_SIZE` blocks of threads.

The block size is arbitrary here, and is not tuned for the specific GPU hardware this will be run on. The
previously-used `foreach()` abstraction will do hardware-specific tuning for this style of dispatch. However, this
example shows how you might mentally map CUDA C-style code to thread-level GPU operations in MAX.

You can execute this one by running the Python-based graph that uses the custom op:

```sh
pixi run python vector_addition.py
```
