---
title: "Introduction to MAX Custom Operations"
description: "Learn how to extend MAX Graph with custom Mojo kernels and integrate them into PyTorch workflows."
---

# Intro to custom ops

Custom operations (custom ops) extend MAX Graph's Python inference APIs with custom Mojo kernels. Whether you need to
optimize performance of functions, implement custom algorithms, or create hardware-specific versions of existing
operators, custom ops provide the flexibility you need.

The custom ops API provides complete control over MAX Graph while handling kernel integration and optimization pipelines
automatically.

Try it now with our [custom ops examples](https://github.com/modular/modular/tree/main/max/examples/custom_ops) on
GitHub or follow the [Build custom ops for GPUs](https://docs.modular.com/max/develop/build-custom-ops) tutorial.

### How it works

A custom op consists of two main components that work together to integrate your custom implementation into the MAX
execution pipeline:

1. A custom function implementation written in Mojo that defines your computation
1. A registration process that connects your function to the graph execution system

Under the hood, custom ops utilize high-level abstractions that handle memory management, device placement, and
optimization. The graph compiler integrates your custom op implementation into the execution flow.

For more information:

- Follow the [Build custom ops for GPUs tutorial](https://docs.modular.com/max/develop/build-custom-ops)
- Learn more about [GPU programming with Mojo](https://docs.modular.com/mojo/manual/gpu/basics)
- Explore the [Custom ops GitHub examples](https://github.com/modular/modular/tree/main/max/examples/custom_ops)
- Reference the [MAX Graph custom ops API](https://docs.modular.com/max/api/python/graph/ops#custom)

## Mojo custom ops in PyTorch

You can also use Mojo to write high-performance kernels for existing PyTorch models without migrating your entire
workflow to MAX. This approach allows you to replace specific performance bottlenecks in your PyTorch code with
optimized Mojo implementations.

Custom operations in PyTorch can now be written using Mojo, letting you experiment with new GPU algorithms in a familiar
PyTorch environment. These custom operations are registered using the [
`CustomOpLibrary`](https://docs.modular.com/max/api/python/torch#max.torch.CustomOpLibrary) class in the [
`max.torch`](https://docs.modular.com/max/api/python/torch) package.

### How it works

1. Write your kernel implementation in Mojo.
1. Register your custom operation using `CustomOpLibrary` from `max.torch`.
1. Replace specific operations in your existing PyTorch model with your Mojo implementation.

This allows you to keep your existing PyTorch workflows while gaining access to Mojo's performance capabilities for
targeted optimizations.

For more information, see
the [Extending PyTorch with custom operations in Mojo](https://github.com/modular/modular/tree/main/max/examples/pytorch_custom_ops)
example.
