"""Focused grouped MXFP4 swizzled path correctness tests.

Non-swizzled grouped tests were removed because they are not a target path for
the legacy-default + RS-cleanup branch.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from gpt_oss_mxfp4.kernels import (
    MXFP4_VALUES_PER_BLOCK,
    get_mxfp4_kernels_path,
    mxfp4_grouped_matmul_ragged_bf16_swizzled,
)
from gpt_oss_mxfp4.weight_adapters import (
    _mxfp4_swizzle_scales_hopper,
    _mxfp4_swizzle_values_hopper,
)
from max.driver import Buffer, CPU, Accelerator
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from safetensors import safe_open

Tensor = Buffer

pytestmark = pytest.mark.skipif(
    os.environ.get("MXFP4_GROUPED_TEST_ENABLE", "0") != "1",
    reason=(
        "Grouped MXFP4 matmul tests are opt-in while legacy fused MoE path is primary."
        " Set MXFP4_GROUPED_TEST_ENABLE=1 to run them."
    ),
)

FP4_VALUES = np.array(
    [
        +0.0,
        +0.5,
        +1.0,
        +1.5,
        +2.0,
        +3.0,
        +4.0,
        +6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=np.float32,
)


def _bf16_round_to_f32(x: np.ndarray) -> np.ndarray:
    """Round FP32 -> BF16 -> FP32 (round-to-nearest-even)."""
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    rounding_bias = ((u >> 16) & 1).astype(np.uint32) + np.uint32(0x7FFF)
    rounded = (u + rounding_bias) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def _decode_mxfp4_rows(
    blocks: np.ndarray, scales_exp: np.ndarray
) -> np.ndarray:
    """Decode MXFP4 rows into dense float32.

    Args:
        blocks: [K/32, N, 16] packed FP4 nibbles.
        scales_exp: [K/32, N] E8M0 exponent bytes (uint8).
    Returns:
        Dense weights [N, K].
    """
    if blocks.ndim != 3:
        raise ValueError(f"Expected blocks rank-3, got {blocks.shape}")
    if scales_exp.ndim != 2:
        raise ValueError(f"Expected scales rank-2, got {scales_exp.shape}")
    if blocks.shape[:2] != scales_exp.shape:
        raise ValueError(
            f"blocks/scales mismatch: {blocks.shape[:2]} vs {scales_exp.shape}"
        )
    if blocks.shape[-1] != 16:
        raise ValueError(f"Expected 16 bytes per block, got {blocks.shape[-1]}")

    k_blocks, n_rows, _ = blocks.shape
    k = k_blocks * MXFP4_VALUES_PER_BLOCK
    out = np.empty((n_rows, k), dtype=np.float32)

    for kb in range(k_blocks):
        for r in range(n_rows):
            scale = np.exp2(np.float32(np.int32(scales_exp[kb, r]) - 127))
            blk = blocks[kb, r]
            for b in range(16):
                packed = int(blk[b])
                lo = packed & 0x0F
                hi = packed >> 4
                k0 = kb * MXFP4_VALUES_PER_BLOCK + 2 * b
                out[r, k0] = FP4_VALUES[lo] * scale
                out[r, k0 + 1] = FP4_VALUES[hi] * scale
    return out


def _prepare_mxfp4_weights_swizzled(
    w_blocks_all: np.ndarray,
    w_scales_all: np.ndarray,
    *,
    num_experts: int,
    n_cols: int,
    k_blocks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (blocks_raw, blocks_swizzled, scales_swizzled, scales_ref)."""
    w_blocks_raw = np.ascontiguousarray(
        w_blocks_all[:num_experts, :n_cols, :k_blocks, :]
    )
    kbytes = k_blocks * 16
    w_blocks_2d = w_blocks_raw.reshape(num_experts, n_cols, kbytes)
    w_blocks_swz = _mxfp4_swizzle_values_hopper(w_blocks_2d, mx_axis=2)
    w_scales_logical = np.ascontiguousarray(
        w_scales_all[:num_experts, :n_cols, :k_blocks]
    )
    w_scales = _mxfp4_swizzle_scales_hopper(w_scales_logical)
    w_scales_ref = np.ascontiguousarray(
        np.transpose(w_scales_logical, (0, 2, 1))
    )
    return w_blocks_raw, w_blocks_swz, w_scales, w_scales_ref


def _find_gpt_oss_20b_file() -> Path | None:
    """Find the first shard that contains layer0 MoE weights."""
    hub_root = (
        Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        / "hub"
    )
    snapshots = hub_root / "models--openai--gpt-oss-20b" / "snapshots"
    if not snapshots.exists():
        return None

    for snap in sorted(snapshots.iterdir(), reverse=True):
        candidate = snap / "model-00000-of-00002.safetensors"
        if candidate.exists():
            return candidate
    return None


@pytest.mark.parametrize("P", [32])
def test_mxfp4_grouped_matmul_swizzled_matches_reference(P: int) -> None:
    try:
        device = Accelerator()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GPU not available: {exc}")

    ckpt = _find_gpt_oss_20b_file()
    if ckpt is None:
        pytest.skip(
            "GPT-OSS checkpoint not found in HF cache; run `pixi run generate` once"
        )

    num_experts = 2
    K = 128
    N = 128
    k_blocks = K // MXFP4_VALUES_PER_BLOCK

    with safe_open(str(ckpt), framework="numpy") as f:
        w_blocks_all = f.get_tensor(
            "model.layers.0.mlp.experts.down_proj_blocks"
        )
        w_scales_all = f.get_tensor(
            "model.layers.0.mlp.experts.down_proj_scales"
        )

    w_blocks_raw, w_blocks_swz, w_scales, w_scales_ref = (
        _prepare_mxfp4_weights_swizzled(
            w_blocks_all,
            w_scales_all,
            num_experts=num_experts,
            n_cols=N,
            k_blocks=k_blocks,
        )
    )

    rng = np.random.default_rng(0)
    a_f32 = rng.uniform(-1.0, 1.0, size=(P, K)).astype(np.float32)
    a_bf16 = _bf16_round_to_f32(a_f32)

    w_dense = _decode_mxfp4_rows(
        np.ascontiguousarray(np.transpose(w_blocks_raw[0], (1, 0, 2))),
        w_scales_ref[0],
    )
    w_dense = _bf16_round_to_f32(w_dense)
    ref = _bf16_round_to_f32(a_bf16 @ w_dense.T)

    expert_start = np.array([0, P, P], dtype=np.uint32)
    expert_ids = np.array([0, -1], dtype=np.int32)
    expert_usage_stats = np.array([P, 1], dtype=np.uint32)
    devref = DeviceRef.from_device(device)
    session = InferenceSession(devices=[device])

    with Graph(
        "mxfp4_grouped_matmul_swizzled",
        input_types=[
            TensorType(DType.float32, shape=[P, K], device=devref),
            TensorType(DType.uint8, shape=w_blocks_swz.shape, device=devref),
            TensorType(DType.uint8, shape=w_scales.shape, device=devref),
            TensorType(DType.uint32, shape=[num_experts + 1], device=devref),
            TensorType(DType.int32, shape=[num_experts], device=devref),
            TensorType(DType.uint32, shape=[2], device=DeviceRef.CPU()),
        ],
        custom_extensions=[get_mxfp4_kernels_path()],
    ) as graph:
        a_in, blocks_in, scales_in, start_in, ids_in, stats_in = graph.inputs
        a_bf16_in = ops.cast(a_in.tensor, DType.bfloat16)
        out_bf16 = mxfp4_grouped_matmul_ragged_bf16_swizzled(
            a_bf16_in,
            blocks_in.tensor,
            scales_in.tensor,
            start_in.tensor,
            ids_in.tensor,
            stats_in.tensor,
            n_cols=N,
            target="gpu",
            no_small_m=True,
        )
        graph.output(ops.cast(out_bf16, DType.float32))

    model = session.load(graph)
    got = (
        model.execute(
            Tensor.from_numpy(a_f32).to(device),
            Tensor.from_numpy(w_blocks_swz).to(device),
            Tensor.from_numpy(w_scales).to(device),
            Tensor.from_numpy(expert_start).to(device),
            Tensor.from_numpy(expert_ids).to(device),
            Tensor.from_numpy(expert_usage_stats).to(CPU()),
        )[0]
        .to(CPU())
        .to_numpy()
    )

    assert got.shape == ref.shape
    assert np.allclose(got, ref, atol=1e-1, rtol=1e-1), (
        f"max abs diff {np.max(np.abs(got - ref))}"
    )
