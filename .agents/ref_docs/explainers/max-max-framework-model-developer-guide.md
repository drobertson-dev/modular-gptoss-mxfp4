---
title: "MAX Framework Model Developer Guide"
description: "A guide to developing and deploying neural networks with the MAX framework using eager-style execution and explicit graph construction."
---

# MAX Framework Model Developer Guide

A guide to developing and deploying neural networks with the MAX framework using eager-style execution and explicit graph construction.

MAX is a high-performance framework built for production-ready neural network
model development and deployment across a wide variety of hardware. It supports
two programming patterns for constructing an AI model:

- Eager-style execution for an enhanced developer experience during model development and debugging.
- Explicit graph construction when you need low-level control over compilation and deployment.

The eager-style execution pattern provides a familiar developer experience
inspired by PyTorch's eager execution, allowing you to write natural Python code
with familiar operators and syntax. Since most time spent in model development
is verifying model correctness, we recommend starting with eager-style
execution. You write PyTorch-style code and get feedback on shape and type
errors during development, while MAX uses lazy evaluation to automatically build
and optimize computation graphs behind the scenes, delivering better performance
than PyTorch.

Take, for example, the following code:

```python
from max import functional as F
from max.tensor import Tensor

# Create tensor from Python data
x = Tensor.constant([1.0, -2.0, 3.0, -4.0, 5.0])

y = F.relu(x)

# Results are available right away
print(y)
```

The expected output is:

```output
TensorType(dtype=float32, shape=[Dim(5)], device=cpu:0): [1.0, 0.0, 3.0, 0.0, 5.0]
```

When you run this code, Tensor.constant() creates a tensor and F.relu()
performs the ReLU activation. The print(y) statement triggers execution. MAX compiles
and runs the operations, then displays the result.

Eager-style execution performance improvements are in progress. Initial
compilation times may be higher than expected and should not be used as a
general replacement for numpy in production.

This is different from explicit graph construction where you define the complete
graph structure upfront, compile it separately, then execute it with data.
Here's the same relu() operation using explicit graph construction:

```python
import numpy as np
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, ops

# Step 1: Define the graph structure
cpu = CPU()
input_type = TensorType(DType.float32, shape=[5], device=cpu)

with Graph("relu_graph", input_types=[input_type]) as graph:
    x = graph.inputs[0]
    y = ops.relu(x)
    graph.output(y)

# Step 2: Compile the graph
session = InferenceSession(devices=[cpu])
model = session.load(graph)

# Step 3: Execute with data
input_data = np.array([1.0, -2.0, 3.0, -4.0, 5.0], dtype=np.float32)
result = model.execute(input_data)
print(np.from_dlpack(result[0]))
```

The expected output is:

```output
[1. 0. 3. 0. 5.]
```

Both produce the same result, but explicit graph construction gives you full
control over graph structure, data flow, and device placement. Eager-style
execution lets you write natural Python code while MAX handles graph building
and optimization automatically.

## Use standard Python operators

You can perform arithmetic operations using Python operators on tensors:

```python
from max.tensor import Tensor

a = Tensor.constant([1.0, 2.0, 3.0])
b = Tensor.constant([4.0, 5.0, 6.0])

c = a + b  # Addition
d = a * b  # Element-wise multiplication

print(c)
print(d)
```

For operations beyond basic arithmetic, use the functional API:

```python
from max import functional as F
from max.tensor import Tensor

x = Tensor.constant([[1.0, 2.0], [3.0, 4.0]])

y = F.sqrt(x)  # Element-wise square root
z = F.softmax(x, axis=-1)  # Softmax along last axis
```

The max.functional module (typically imported as F) provides
operations like relu(), softmax(), sqrt(), and many more.

## Inspect values while debugging

One of the biggest advantages of the eager-style API is that you can inspect
intermediate values at any point in your code.

This makes debugging straightforward:

```python
from max.driver import CPU
from max.dtype import DType
from max import functional as F
from max.tensor import Tensor

def debug_forward_pass(x: Tensor) -> Tensor:
    """Forward pass with intermediate inspection."""
    # Can print/inspect at any point
    print(f"Input: {x}")

    z = x * 2
    print(f"After multiply: {z}")

    h = F.relu(z)
    print(f"After ReLU: {h}")

    return h

x = Tensor.constant([-1.0, 0.0, 1.0, 2.0], dtype=DType.float32, device=CPU())
result = debug_forward_pass(x)
```

The expected output is:

```output
Input: TensorType(dtype=float32, shape=[Dim(4)], device=cpu:0): [-1.0, 0.0, 1.0, 2.0]
After multiply: TensorType(dtype=float32, shape=[Dim(4)], device=cpu:0): [-2.0, 0.0, 2.0, 4.0]
After ReLU: TensorType(dtype=float32, shape=[Dim(4)], device=cpu:0): [0.0, 0.0, 2.0, 4.0]
```

If shapes don't match or types are incompatible, you get clear errors showing
exactly which operation failed, right where you wrote it.

## Follow a complete workflow example

Here's a more realistic example showing a forward pass through a simple model:

```python
from max.driver import CPU
from max.dtype import DType
from max import functional as F
from max import random
from max.tensor import Tensor

# Create input data
x = Tensor.constant([[1.0, 2.0], [3.0, 4.0]], dtype=DType.float32, device=CPU())

# Create random weights
w = random.gaussian(
    [2, 2], mean=0.0, std=0.1, dtype=DType.float32, device=CPU()
)

# Forward pass - each operation executes as you write it
z = x @ w  # Matrix multiply
h = F.relu(z)  # Activation
out = h.mean()  # Reduce to scalar

# Inspect intermediate results anytime
print(f"Input shape: {x.shape}")
print(f"After matmul: {z.shape}")
print(f"Output: {out}")
```

Shape and type validation happens as you write operations. If the matrix
dimensions didn't align for x @ w, you'd get an error at that exact line
showing the shape mismatch.

## Understand when execution happens

With eager-style execution, MAX uses lazy evaluation to optimize performance.
When you write operations like y = F.relu(x), MAX doesn't compute the result
immediately. Instead, it builds up a computation graph by recording each
operation as you call it. This deferred execution allows MAX to analyze the
entire sequence of operations and optimize them before running anything.

Think of it like writing a recipe: as you add each step (slice vegetables, heat
oil, add ingredients), you're building instructions but not actually cooking
yet. MAX does the same thing, it records your tensor operations into a graph,
then compiles and executes that graph only when you need actual computed values.

Execution happens automatically when an operation requires concrete values:

- **Printing tensors**: print(x) needs real values to display them. MAX
  compiles and executes the graph to produce those values.
- **Accessing scalar values**: x.item() must return a concrete Python number
  (like 3.14) rather than a symbolic operation, which triggers execution.
- **Indexing tensors**: x[0] needs to extract the actual value at that
  position.
- **Converting to other formats**: Passing a MAX Tensor to np.from_dlpack(x)
  requires a real memory buffer for x, which forces execution.
- **Module forward passes**: Calling module(x) or module.forward(x) on an nn.Module executes the
  graph to produce the output.

This all happens automatically and transparently behind the scenes. MAX detects
when it needs actual values and handles compilation and execution for you.
