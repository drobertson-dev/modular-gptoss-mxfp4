---
title: "Get Started with MAX Graph in Python"
description: "Learn how to build, compile, and execute computational graphs using the MAX Python API."
---

# Get Started with MAX Graph in Python

Learn how to build, compile, and execute computational graphs using the MAX Python API.

MAX provides a high-performance computation framework that lets you build and execute efficient machine learning models.
It provides a flexible way to define computational workflows as graphs, where each node represents an operation (like
matrix multiplication or addition) and edges represent the flow of data. By using the MAX Python API, you can create
optimized machine learning models that run faster and more efficiently on modern hardware.

In this tutorial, you'll build a graph using the Python Graph API with an ops function.

To do this, you will complete the following steps:

1. Build a simple graph that adds two numbers
1. Create an inference session to load and compile the graph
1. Execute the graph with input data

By the end of this tutorial, you'll have an understanding of how to construct basic computational graphs, set up
inference sessions, and run computations using the MAX Python API.

## Set up your environment

Create a Python project to install our APIs and CLI tools.

### pixi

1. If you don't have it, install pixi:

```sh
curl -fsSL https://pixi.sh/install.sh | sh
```

Then restart your terminal for the changes to take effect.

2. Create a project:

```sh
pixi init example-project \
     -c https://conda.modular.com/max-nightly/ -c conda-forge \
     && cd example-project
```

3. Install the modular conda package:

**Nightly**

```sh
pixi add modular
```

**Stable**

```sh
pixi add "modular==25.7"
```

4. Start the virtual environment:

```sh
pixi shell
```

### uv

1. If you don't have it, install uv:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then restart your terminal to make uv accessible.

2. Create a project:

```sh
uv init example-project && cd example-project
```

3. Create and start a virtual environment:

```sh
uv venv && source .venv/bin/activate
```

4. Install the modular Python package:

**Nightly**

```sh
uv pip install modular \
  --index https://whl.modular.com/nightly/simple/ \
  --prerelease allow
```

**Stable**

```sh
uv pip install modular \
  --extra-index-url https://modular.gateway.scarf.sh/simple/
```

### pip

1. Create a project folder:

```sh
mkdir example-project && cd example-project
```

2. Create and activate a virtual environment:

```sh
python3 -m venv .venv/example-project \
     && source .venv/example-project/bin/activate
```

3. Install the modular Python package:

**Nightly**

```sh
pip install --pre modular \
  --extra-index-url https://whl.modular.com/nightly/simple/
```

**Stable**

```sh
pip install modular \
  --extra-index-url https://modular.gateway.scarf.sh/simple/
```

### conda

1. If you don't have it, install conda.

1. Initialize conda for shell interaction:

```sh
conda init
```

If you're on a Mac, instead use:

```sh
conda init zsh
```

Then restart your terminal for the changes to take effect.

3. Create a project:

```sh
conda create -n example-project
```

4. Start the virtual environment:

```sh
conda activate example-project
```

5. Install the modular conda package:

**Nightly**

```sh
conda install -c conda-forge -c https://conda.modular.com/max-nightly/ modular
```

**Stable**

```sh
conda install -c conda-forge -c https://conda.modular.com/max/ modular
```

### Working Directory Setup

**For pip and uv:**

Create a folder called `max_ops`:

```sh
mkdir max_ops
cd max_ops
```

You can check your MAX and Python versions:

```sh
max --version
python --version
```

**For pixi:**

Change folders to your working directory:

```sh
cd src/quickstart
```

You can check your MAX and Python versions:

```sh
pixi run max --version
pixi run python --version
```

To clear cached data while iterating on graph builds, you can use `pixi clean` to remove the MEF cache and other
environment data:

```sh
pixi clean
```

## 1. Build the graph

Now with our environment and packages setup, lets create the graph. This graph will define a computational workflow that
adds two tensors together.

Start by creating a new file called `addition.py` inside of your working directory and add the following libraries:

```python
import numpy as np
from max import engine
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops
```

To create a computational graph, use the `Graph()` class. When initializing, specify a name for the graph and define the
types of inputs it will accept.

```python
def add_tensors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # 1. Build the graph
    input_type = TensorType(
        dtype=DType.float32, shape=(1,), device=DeviceRef.CPU()
    )
    with Graph(
            "simple_add_graph", input_types=(input_type, input_type)
    ) as graph:
        lhs, rhs = graph.inputs
        out = ops.add(lhs, rhs)
        graph.output(out)
```

Inside the context manager, access the graph's inputs using the `inputs` property. This returns a symbolic tensor
representing the input arguments.

The symbolic tensor is a placeholder that represents the shape and type of data that will flow through the graph during
the execution, rather than containing the actual numeric values like in eager execution.

Then use the `add()` function from the `ops` package to add the two input tensors. This creates a new symbolic tensor
representing the sum.

Finally, set the output of the graph using the `output()` method. This specifies which tensors should be returned when
the graph is executed.

Now, add a `print()` function to the graph to see what's created.

```python
def add_tensors(a: np.ndarray, b: np.ndarray) -> dict[str, any]:
    # 1. Build the graph
    # ...
    print("final graph:", graph)
```

The output will show the structure of the graph, including the input it expects and the operations it will perform.

## 2. Create an inference session

Now that our graph is constructed, let's set up an environment where it can operate. This involves creating an inference
session and loading our graph into it.

Create an `InferenceSession()` instance that loads and runs the graph inside the `add_tensors()` function.

```python
def add_tensors(a: np.ndarray, b: np.ndarray) -> dict[str, any]:
    # 1. Build the graph
    # ...
    # 2. Create an inference session
    session = engine.InferenceSession(devices=[CPU()])
    model = session.load(graph)
```

This step transforms the abstract graph into a computational model that's ready for execution.

### Debugging graph compilation errors

If you encounter errors during `session.load(graph)`, you can enable detailed debugging information by setting the
`MODULAR_MAX_DEBUG` environment variable:

```bash
export MODULAR_MAX_DEBUG=True
python addition.py
```

To ensure the model is set up correctly, examine its input requirements by printing the `input_metadata` property.

```python
def add_tensors(a: np.ndarray, b: np.ndarray) -> dict[str, any]:
    # 1. Build the graph
    # ...
    # 2. Create an inference session
    session = engine.InferenceSession(devices=[CPU()])
    model = session.load(graph)
    for tensor in model.input_metadata:
        print(
            f"name: {tensor.name}, shape: {tensor.shape}, dtype: {tensor.dtype}"
        )
```

## 3. Execute the graph

To give the model something to add, create two inputs of a shape and a data type that match the graph's input
requirements. Then pass the inputs to the `execute()` function:

```python
def add_tensors(a: np.ndarray, b: np.ndarray) -> dict[str, any]:
    # ...
    # 2. Create an inference session
    # ...
    # 3. Execute the graph
    output = model.execute(a, b)[0]
    result = output.to_numpy()
    return result
```

The `execute()` function returns a list of outputs. We take the first element and convert it to a NumPy array.

## 4. Run the example

At the end of your `addition.py` file, add the following code:

```python
if __name__ == "__main__":
    input0 = np.array([1.0], dtype=np.float32)
    input1 = np.array([1.0], dtype=np.float32)
    result = add_tensors(input0, input1)
    print("result:", result)
```

Run the Python file from the command line:

**pip or uv**

```sh
python addition.py
```

**pixi**

```sh
pixi run python addition.py
```

### Output Analysis

The terminal output will show the graph structure:

```output
final graph: mo.graph @simple_add_graph(%arg0: !mo.tensor<[1], f32>, %arg1: !mo.tensor<[1], f32>) -> !mo.tensor<[1], f32> attributes {argument_names = ["input0", "input1"], result_names = ["output0"]} {
  %0 = rmo.add(%arg0, %arg1) : (!mo.tensor<[1], f32>, !mo.tensor<[1], f32>) -> !mo.tensor<[1], f32>
  mo.output %0 : !mo.tensor<[1], f32>
}
```

- Two input tensors (`%arg0`, `%arg1`) of shape `[1]` and float32 type
- The addition operation connecting them
- One output tensor of matching shape/type

The metadata lines confirm both input tensors match the required specifications:

```output
name: input0, shape: [1], dtype: DType.float32
name: input1, shape: [1], dtype: DType.float32
```

The result shows the addition worked correctly:

[1.0] + [1.0] = [2.0]

```output
result: [2.]
```
