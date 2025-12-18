"""ModuleV3 compilation helpers.

Upstream `max.nn.module_v3.Module.compile()` does not pass `custom_extensions` to
`max.graph.Graph(...)`, so custom Mojo ops under `examples/custom_ops/kernels/`
won't be registered. This helper mirrors `Module.compile()` while injecting the
needed `custom_extensions` list.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from max import graph
from max.driver import CPU, DLPackArray
from max.experimental import functional as F
from max.experimental.tensor import Tensor, _session
from max.graph import Graph, TensorType
from max.nn.module_v3.module import Module


def compile_with_custom_extensions(
    module: Module,
    *input_types: graph.Type[Any],
    weights: Mapping[str, DLPackArray],
    custom_extensions: Sequence[str | Path],
) -> Callable[..., Any]:
    """Compile a ModuleV3 `Module` with `custom_extensions` enabled."""

    kernel_paths = [p if isinstance(p, Path) else Path(p) for p in custom_extensions]

    with Graph(
        type(module).__qualname__,
        input_types=input_types,
        custom_extensions=kernel_paths,
    ) as graph_:
        inputs = [Tensor.from_graph_value(v) for v in graph_.inputs]

        def as_weight(name: str, tensor: Tensor):  # noqa: ANN202
            # Match upstream Module.compile: weights are registered as CPU externals
            # and moved to the parameter device during init.
            type_ = TensorType(tensor.dtype, tensor.shape, CPU())
            return F.constant_external(name, type_).to(tensor.device)

        with module._mapped_parameters(as_weight):
            outputs: Tensor | Sequence[Tensor] = module(*inputs)

        unary = isinstance(outputs, Tensor)
        if unary:
            graph_.output(outputs)
        else:
            graph_.output(*outputs)

    session = _session()
    compiled = F.functional(session.load(graph_, weights_registry=weights))

    if unary:
        return functools.wraps(module)(lambda *inputs: compiled(*inputs)[0])
    return compiled


__all__ = ["compile_with_custom_extensions"]
