"""Regression test: keep GPT-OSS sink attention enabled.

GPT-OSS uses per-head sink weights ("attention sinks") and expects them to be
passed into `flash_attention_ragged(..., sink_weights=...)` for correctness.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from gpt_oss_mxfp4.layers.attention import GptOssAttention


def _find_flash_attention_call(tree: ast.AST) -> ast.Call | None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "flash_attention_ragged":
            return node
    return None


def test_flash_attention_sink_weights_enabled() -> None:
    src = textwrap.dedent(inspect.getsource(GptOssAttention.__call__))
    tree = ast.parse(src)
    call = _find_flash_attention_call(tree)
    assert call is not None, "Expected a call to flash_attention_ragged"

    kw = {k.arg: k.value for k in call.keywords if k.arg is not None}
    assert "sink_weights" in kw, "Expected flash_attention_ragged sink_weights="

    sink = kw["sink_weights"]
    assert isinstance(sink, ast.Attribute) and sink.attr == "sinks", (
        "Expected flash_attention_ragged(..., sink_weights=self.sinks) for GPT-OSS sink attention"
    )
