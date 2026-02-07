---
title: "Tensor Fundamentals in MAX"
description: "An introduction to creating and managing multi-dimensional arrays using tensors in the MAX framework."
---

# Tensor fundamentals

Tensors are the fundamental building blocks of neural network models in MAX. They represent multi-dimensional arrays of
numbers used for model inputs, outputs, and parameters.

If you're coming from NumPy, think of a MAX `Tensor` like a NumPy `ndarray`.

A tensor is a multi-dimensional array of numbers. You can think of tensors as:

- Rank 0 (scalar): a single number (like `72.5`)
- Rank 1 (vector): a 1-D tensor or list of numbers (like `[1, 2, 3, 4]`)
- Rank 2 (matrix): a 2-D tensor or table of numbers (like a spreadsheet with rows and columns)
- Rank 3+ (higher-dimensional arrays): a 3-or-more dimensional tensor (stacks of matrices or more complex structures)

### Coming from PyTorch or NumPy?

There are some key differences between tensors in MAX and tensors in PyTorch or NumPy. For example, tensors in MAX are
created using the `Tensor.constant()` function instead of direct construction, many math operations use the functional
API (`F.sqrt()` instead of `.sqrt()`), and reduction operations keep dimensions by default.

## Create a tensor

You can create a tensor from a Python list using the `Tensor.constant()` function:

```python
from max.tensor import Tensor

# Create a simple 1-D tensor (a vector)
x = Tensor.constant([1, 2, 3, 4, 5])

print(x)
```

This imports the `Tensor` class and creates a 1-D tensor (a vector) with the values `[1, 2, 3, 4, 5]`. The `constant()`
function creates a tensor from a tensor-like object; meaning, it can be a list, a NumPy array, or a PyTorch tensor.

The expected output is:

```output
TensorType(dtype=float32, shape=[Dim(5)], device=cpu:0): [1.0, 2.0, 3.0, 4.0, 5.0]
```

### Performance note

`Tensor.constant()` performance optimizations are still being improved. In the meantime, if you need better performance
when creating tensors on accelerators, use:

```python
import numpy as np
from max.driver import Accelerator
from max.tensor import Tensor

np_array = np.array([1.0, 2.0, 3.0])
tensor = Tensor.from_numpy(np_array).to(Accelerator())
```

### About dtype and device

Notice that when creating the tensors, the output shows the dtype and device. The dtype is the data type of the tensor,
and the device is the device on which the tensor will be stored.

You can also create tensors with any number of dimensions by nesting lists:

```python
from max.tensor import Tensor

# Create a 2-D tensor (a matrix)
matrix = Tensor.constant([[1, 2, 3], [4, 5, 6]])
print(matrix)

# Create a 3-D tensor (a cube of numbers)
cube = Tensor.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(cube)
```

The expected output is:

```output
TensorType(dtype=float32, shape=[Dim(2), Dim(3)], device=cpu:0): [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
TensorType(dtype=float32, shape=[Dim(2), Dim(2), Dim(2)], device=cpu:0): [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
```

MAX provides convenient functions for creating tensors with specific patterns, such as `ones()` and `zeros()`:

```python
from max.driver import CPU
from max.dtype import DType
from max.tensor import Tensor

# Tensor filled with ones
ones = Tensor.ones([3, 4], dtype=DType.float32, device=CPU())
print(ones)

# Tensor filled with zeros
zeros = Tensor.zeros([2, 3], dtype=DType.float32, device=CPU())
print(zeros)
```

The expected output is:

```output
TensorType(dtype=float32, shape=[Dim(3), Dim(4)], device=cpu:0): [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
TensorType(dtype=float32, shape=[Dim(2), Dim(3)], device=cpu:0): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

For random data, you can use one of several functions from the `random` module, such as `random.normal()`:

```python
from max import random

# Random values from a normal distribution
random_tensor = random.normal([3, 3])
print(random_tensor)
```

The expected output is:

```output
TensorType(dtype=float32, shape=[Dim(3), Dim(3)], device=cpu:0): [1.6810914278030396, 2.3331382274627686, -0.25120288133621216, 0.8896129131317139, 1.6362168788909912, -1.9282348155975342, -0.4372555911540985, -0.8747910261154175, 0.5068135857582092]
```

You can adjust the mean and standard deviation of the normal distribution to create tensors with different values:

```python
random_tensor = random.normal([3, 3], mean=0.0, std=1.0)
print(random_tensor)
```

The expected output will vary (since it's random), but will look similar to this:

```output
TensorType(dtype=float32, shape=[Dim(3), Dim(3)], device=cpu:0): [-0.5, 1.2, -0.8, 0.3, -1.1, 0.7, 0.2, -0.4, 0.9]
```

The actual values will be different each time you run the code, as they are randomly generated from a normal (Gaussian)
distribution with mean 0.0 and standard deviation 1.0.

## Tensor properties

Every tensor has several key properties that tell you about its structure and contents.

### Shape

The *shape* tells you the size of each dimension of the tensor. It's a list where each number represents the size along
that dimension:

```python
from max.tensor import Tensor

# 1-D tensor
x = Tensor.constant([1, 2, 3, 4])
print(x.shape)

# 2-D tensor
matrix = Tensor.constant([[1, 2, 3], [4, 5, 6]])
print(matrix.shape)

# 3-D tensor
cube = Tensor.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(cube.shape)
```

The expected output is:

```output
[Dim(4)]
[Dim(2), Dim(3)]
[Dim(2), Dim(2), Dim(2)]
```

In this example, the `shape` property returns a list of `Dim` objects representing the size of each dimension.

### Rank

The *rank* (also called number of dimensions) tells you how many axes the tensor has:

```python
from max.tensor import Tensor

scalar = Tensor.constant([42])  # Rank 1 (it's a 1-element vector)
vector = Tensor.constant([1, 2, 3])  # Rank 1
matrix = Tensor.constant([[1, 2], [3, 4]])  # Rank 2
cube = Tensor.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # Rank 3

print(vector.rank)  # 1
print(matrix.rank)  # 2
print(cube.rank)  # 3
```

### Data type (dtype)

*Dtype*, or data type, is a property of a tensor that tells you the type of the data stored in the tensor. For example,
a tensor can contain integers: `int32`, floating-point numbers: `float32`, or booleans: `bool`.

```python
from max.dtype import DType
from max.tensor import Tensor

# Float tensor (default for most operations)
floats = Tensor.ones([2, 2], dtype=DType.float32)

# Integer tensor
integers = Tensor.ones([2, 2], dtype=DType.int32)
```

MAX supports a wide range of dtypes, including all standard NumPy and PyTorch dtypes.

### Device

The *device* tells you where the tensor operation occurs and is stored:

```python
from max.driver import CPU
from max.tensor import Tensor

# Tensor on CPU
cpu_tensor = Tensor.ones([2, 2], device=CPU())
```

When you need to perform operations on a tensor, you need to specify the device on which the tensor is stored.
Currently, MAX supports CPU and GPU devices by specifying either the `CPU()` or `Accelerator()` class.

### Total elements

You can check how many numbers are stored in the tensor:

```python
from max.tensor import Tensor

t = Tensor.constant([[1, 2, 3], [4, 5, 6]])
print(t.num_elements())  # 6 (2 rows × 3 columns)
```

In this example, the expected output is:

```output
6
```

The `num_elements()` function returns the total number of elements in the tensor, which is the product of the
dimensions.
