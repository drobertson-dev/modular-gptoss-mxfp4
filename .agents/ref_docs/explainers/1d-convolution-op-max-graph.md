---
title: "1D Convolution Op (MAX Graph)"
description: "In Puzzle 13, we implemented a 1D convolution kernel that runs efficiently on the GPU. Now we'll take this kernel and transform it into a custom operation that can be called directly from Python using MAX Graph."
---

# 1D Convolution Op (MAX Graph)

In Puzzle 13, we implemented a 1D convolution kernel that runs efficiently on the GPU. Now we'll take this kernel and transform it into a custom operation that can be called directly from Python using MAX Graph.

> ## Bridging to Python with MAX Graph
>
> We're now entering Part IV of our GPU puzzle journey: **Interfacing with Python via MAX Graph Custom Ops**.
>
> In previous puzzles, we've learned how to write efficient GPU kernels in Mojo. Now we'll explore how to:
>
> - Package these kernels as custom operations that can be called from Python
> - Integrate with the MAX Graph system for accelerated machine learning
> - Bridge the gap between high-level Python APIs and low-level GPU code
>
> This integration allows us to leverage the performance of Mojo GPU kernels while working in familiar Python environments.

## Overview

In Puzzle 13, we implemented a 1D convolution kernel that runs efficiently on the GPU. Now we'll take this kernel and transform it into a custom operation that can be called directly from Python using [MAX Graph](https://docs.modular.com/max/api/python/graph/).

The 1D convolution kernel we'll be working with is already implemented:

```mojo
comptime TPB = 15
comptime BLOCKS_PER_GRID = (2, 1)

fn conv1d_kernel
    in_layout: Layout,
    out_layout: Layout,
    conv_layout: Layout,
    input_size: Int,
    conv_size: Int,
    dtype: DType = DType.float32,
:
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    local_i = Int(thread_idx.x)
    # first: need to account for padding
    shared_a = LayoutTensor[
        dtype,
        Layout.row_major(TPB + conv_size - 1),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    shared_b = LayoutTensor[
        dtype,
        Layout.row_major(conv_size),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    if global_i < input_size:
        shared_a[local_i] = input[global_i]

    # second: load elements needed for convolution at block boundary
    if local_i < conv_size - 1:
        # indices from next block
        next_idx = global_i + TPB
        if next_idx < input_size:
            shared_a[TPB + local_i] = input[next_idx]
        else:
            shared_a[TPB + local_i] = 0

    if local_i < conv_size:
        shared_b[local_i] = kernel[local_i]

    barrier()

    if global_i < input_size:
        var local_sum: output.element_type = 0

        @parameter
        for j in range(conv_size):
            if local_i + j < TPB + conv_size - 1:
                local_sum += shared_a[local_i + j] * shared_b[j]

        output[global_i] = local_sum

```

The key aspects of this puzzle include:

1. **Custom op registration**: Understanding how to expose Mojo functions to Python via the `@compiler.register` decorator
2. **Packaging custom ops**: Learning how to package Mojo code for use with MAX Graph
3. **Python integration**: Calling custom operations from Python through MAX Graph
4. **Cross-language data flow**: Managing data types and memory between Python and GPU

This custom operation will:

- Accept [NumPy](https://numpy.org/doc/stable/) arrays as input from Python
- Transfer this data to the GPU
- Execute our optimized convolution kernel
- Return the results back to Python

When you complete this puzzle, you'll have created a seamless bridge between Python's rich ecosystem and Mojo's powerful GPU performance.

## Reference implementation (example)


To solve this puzzle, we need to integrate our 1D convolution kernel with the MAX Graph system. The key is to properly call our kernel from the `execute` method in the `Conv1DCustomOp` struct.

The solution is:

```mojo
            comptime kernel = conv1d_kernel[
                in_layout, out_layout, conv_layout, input_size, conv_size
            ]
            gpu_ctx.enqueue_functionkernel, kernel,
            )

```

This single line does several important things:

1. Calls [enqueue_function](https://docs.modular.com/mojo/stdlib/gpu/host/device_context/DeviceContext/#enqueue_function) on the GPU context (`gpu_ctx` is of type [DeviceContext](https://docs.modular.com/mojo/stdlib/gpu/host/device_context/DeviceContext/)) to schedule our kernel execution
2. Passes the necessary layout and size information as **compile-time**parameters
3. Provides the output, input, and kernel tensors as runtime arguments
4. Configures the execution grid with the appropriate dimensions

Let's break down how this works in the larger context:

### Python-Mojo integration flow

1. **Python side (problems/p17/p17.py)**:
   - Creates NumPy arrays for input and kernel
   - Calls `conv_1d()` function which wraps our operation in MAX Graph
   - Converts NumPy arrays to [MAX driver](https://docs.modular.com/max/api/python/driver) Buffers with `Buffer.from_numpy(input).to(device)`
   - Loads the custom operation package with `custom_extensions=[mojo_kernels]`

2. **Graph building**:
   - Defines input and output tensor types with [TensorType](https://docs.modular.com/max/api/python/graph/type/#max.graph.type.TensorType)
   - Specifies parameters for our operation via `parameters={...}`
   - Creates a computation graph with [`Graph("conv_1d_graph", ...)`](https://docs.modular.com/max/api/python/graph/Graph)
   - Calls our operation using [`ops.custom(name="conv1d", ...)`](https://docs.modular.com/max/api/python/graph/ops#custom)

3. **Custom op registration**:
   - The `@compiler.register("conv1d")` decorator exposes our operation to MAX Graph. See [@compiler.register](https://docs.modular.com/mojo/manual/decorators/compiler-register/)
   - The `execute` method parameters define the interface (inputs, outputs, context)
   - Input/output tensors are converted to LayoutTensors for use in our kernel
   - Device context manages GPU memory allocation and kernel execution

4. **Kernel execution**:
   - When [model.execute(...)]() is called, our `conv1d_kernel` receives the data
   - GPU thread configuration is set with `grid_dim` and `block_dim`
   - Results are transferred back to CPU with `result.to(CPU())`
   - NumPy verification compares our results with the expected output

### Key components in detail

1. **Custom Op Structure**:

   ```mojo
   @compiler.register("conv1d")
   struct Conv1DCustomOp:
       @staticmethod
       fn executetarget: StaticString, input_size: Int, conv_size: Int, dtype: DType = DType.float32 raises:
           # Implementation
   ```

   - `target` indicates the device type ("gpu" or "cpu")
   - `input_size` and `conv_size` are parameters passed from Python
   - Tensor types ensure correct shape and type checking
   - Return type is `raises` for proper error handling

2. **Tensor Conversion**:

   ```mojo
   output_tensor = output.to_layout_tensor()
   input_tensor = input.to_layout_tensor()
   kernel_tensor = kernel.to_layout_tensor()
   ```

   - MAX Graph tensors are converted to Mojo LayoutTensors
   - This allows our kernel to work with them directly
   - The layouts are extracted for compile-time optimization

3. **Device Context Usage**:

   ```mojo
   gpu_ctx = ctx.get_device_context()
   gpu_ctx.enqueue_memset(...)  # Zero output buffer
   gpu_ctx.enqueue_function..., ... # Schedule kernel
   ```

   - Device context manages GPU resources
   - Memory operations ensure correct buffer state
   - Function enqueueing schedules our kernel for execution

This solution demonstrates the complete flow from Python data through MAX Graph to GPU execution and back, leveraging Mojo's powerful type system and parametric functions to create efficient, type-safe, accelerated operations.

## Understanding MAX Graph custom ops

> Check out the follow tutorials for more details:
>
> - [Get started with MAX Graph in Python](https://docs.modular.com/max/tutorials/get-started-with-max-graph-in-python/)
> - [MAX Graph custom op for GPUs](https://docs.modular.com/max/tutorials/build-custom-ops/)

### Custom op registration

The core of creating a custom operation is the `@compiler.register` decorator and the associated structure:

```mojo
@compiler.register("conv1d")
struct Conv1DCustomOp:
    @staticmethod
    fn execute... raises:
        # Implementation here
```

Key components of the registration:

- The **name**passed to the decorator (`"conv1d"`) is what Python code will use to call this operation
- The **struct**must have an `execute` method with the correct signature
- **OutputTensor**and **InputTensor**types define the interface for Python data
- **DeviceContextPtr**provides access to the execution environment

### Packaging custom ops

Before the custom operation can be used from Python, it needs to be packaged:

```bash
mojo package op -o op.mojopkg
```

This command:

1. Compiles the Mojo code into a deployable package
2. Creates the necessary metadata for MAX Graph to understand the operation
3. Produces a binary artifact (`op.mojopkg`) that can be loaded by Python

The package must be placed in a location where MAX Graph can find it, typically in a directory accessible to the Python code.

### Python integration

On the Python side, here's how the custom operation is used:

```python
# Path to the directory containing our Mojo operations
mojo_kernels = Path(__file__).parent / "op"

# Configure our graph with the custom conv1d operation
with Graph(
    "conv_1d_graph",
    input_types=[...],
    custom_extensions=[mojo_kernels],  # Load our custom op package
) as graph:
    # Define inputs to the graph
    input_value, kernel_value = graph.inputs

    # Use our custom operation by name
    output = ops.custom(
        name="conv1d",  # Must match the name in @compiler.register
        values=[input_value, kernel_value],
        out_types=[...],
        parameters={
            "input_size": input_tensor.shape[0],
            "conv_size": kernel_tensor.shape[0],
            "dtype": dtype,
        },
    )[0].tensor
```

The key elements are:

1. Specifying the path to our custom operations with `custom_extensions`
2. Calling `ops.custom` with the registered operation name
3. Passing input values and parameters that match our operation's signature
