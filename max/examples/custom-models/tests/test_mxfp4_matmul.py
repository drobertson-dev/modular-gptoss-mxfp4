"""Correctness smoke test for the MXFP4 matmul+SwiGLU custom op (CPU path)."""

from __future__ import annotations

import numpy as np
import pytest
from gpt_oss_mxfp4.kernels import (
    MXFP4_VALUES_PER_BLOCK,
    get_mxfp4_kernels_path,
    mxfp4_matmul_swiglu,
)
from gpt_oss_mxfp4.weight_loader import decode_mxfp4, make_dummy_mxfp4_weights
from max.driver import CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType


def _fused_swiglu_ref(
    mat: np.ndarray, bias: np.ndarray, alpha: float, limit: float
) -> np.ndarray:
    """Reference fused SwiGLU for interleaved gate/up columns."""
    gate = mat[:, 0::2]
    up = mat[:, 1::2]
    gate = np.minimum(gate + bias[0::2], limit)
    up = np.clip(up + bias[1::2], -limit, limit)
    glu = gate * (1.0 / (1.0 + np.exp(-alpha * gate)))
    return glu * (up + 1.0)


@pytest.mark.parametrize("M,K,N_full", [(4, 32, 8), (2, 64, 16)])
def test_mxfp4_matmul_swiglu_cpu(M: int, K: int, N_full: int) -> None:
    if K % MXFP4_VALUES_PER_BLOCK != 0:
        pytest.skip("K must be divisible by 32 for MXFP4 packing")

    device = CPU()
    device_ref = DeviceRef.from_device(device)

    # Synth inputs
    a_np = np.random.uniform(-1, 1, size=(M, K)).astype(np.float32)
    w_blocks, w_scales = make_dummy_mxfp4_weights(K, N_full, seed=0)
    bias = np.random.uniform(-0.25, 0.25, size=(N_full,)).astype(np.float32)

    # Reference dense path
    dense = decode_mxfp4(w_blocks, w_scales)
    ref = a_np @ dense
    ref = _fused_swiglu_ref(ref, bias, alpha=1.702, limit=7.0).astype(
        np.float32
    )

    # Build graph
    with Graph(
        "mxfp4_matmul_swiglu_cpu",
        input_types=[
            TensorType(DType.float32, shape=[M, K], device=device_ref),
            TensorType(DType.uint8, shape=w_blocks.shape, device=device_ref),
            TensorType(DType.float32, shape=w_scales.shape, device=device_ref),
            TensorType(DType.float32, shape=[N_full], device=device_ref),
        ],
        custom_extensions=[get_mxfp4_kernels_path()],
    ) as graph:
        a_val, blocks_val, scales_val, bias_val = graph.inputs
        out = mxfp4_matmul_swiglu(
            a_val,
            blocks_val,
            scales_val,
            bias_val,
            target="cpu",
            custom_extensions=[get_mxfp4_kernels_path()],
        )
        graph.output(out)

    session = InferenceSession(devices=[device])
    model = session.load(graph)

    a_t = Tensor.from_numpy(a_np).to(device)
    blocks_t = Tensor.from_numpy(w_blocks).to(device)
    scales_t = Tensor.from_numpy(w_scales.astype(np.float32)).to(device)
    bias_t = Tensor.from_numpy(bias).to(device)

    out_t = model.execute(a_t, blocks_t, scales_t, bias_t)[0].to(CPU())
    got = out_t.to_numpy()

    assert got.shape == ref.shape
    atol = 1e-2
    rtol = 1e-2
    assert np.allclose(got, ref, atol=atol, rtol=rtol), (
        f"max diff {np.max(np.abs(got - ref))}"
    )
