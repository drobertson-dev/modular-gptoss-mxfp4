"""Reference check for `mxfp4_grouped_matmul_ragged_bf16` against numpy decoding."""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest
from gpt_oss_mxfp4_v3.kernels import (
    MXFP4_VALUES_PER_BLOCK,
    get_mxfp4_kernels_path,
    mxfp4_grouped_matmul_ragged_bf16,
)
from max.driver import CPU, Accelerator, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from safetensors import safe_open

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


def _decode_mxfp4_rows(blocks: np.ndarray, scales_exp: np.ndarray) -> np.ndarray:
    """Decode MXFP4 rows into dense float32.

    Args:
        blocks: [N, K/32, 16] packed FP4 nibbles.
        scales_exp: [N, K/32] E8M0 exponent bytes (uint8).
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

    n_rows, k_blocks, _ = blocks.shape
    k = k_blocks * MXFP4_VALUES_PER_BLOCK
    out = np.empty((n_rows, k), dtype=np.float32)

    for r in range(n_rows):
        for kb in range(k_blocks):
            scale = np.exp2(np.float32(np.int32(scales_exp[r, kb]) - 127))
            blk = blocks[r, kb]
            for b in range(16):
                packed = int(blk[b])
                lo = packed & 0x0F
                hi = packed >> 4
                k0 = kb * MXFP4_VALUES_PER_BLOCK + 2 * b
                out[r, k0] = FP4_VALUES[lo] * scale
                out[r, k0 + 1] = FP4_VALUES[hi] * scale
    return out


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


def _load_safetensor_bf16_header(path: Path, key: str) -> tuple[str, list[int]]:
    """Return (dtype, shape) for a key without loading data."""
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    info = header[key]
    return info["dtype"], info["shape"]


@pytest.mark.parametrize("P", [32, 512])
def test_mxfp4_grouped_matmul_single_expert_matches_reference(P: int) -> None:
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
        w_blocks_all = f.get_tensor("model.layers.0.mlp.experts.down_proj_blocks")
        w_scales_all = f.get_tensor("model.layers.0.mlp.experts.down_proj_scales")

    w_blocks = w_blocks_all[:num_experts, :N, :k_blocks, :].copy()
    w_scales = w_scales_all[:num_experts, :N, :k_blocks].copy()

    # Bias is not part of this op, but confirm it exists and is BF16 (checkpoint contract).
    bias_dtype, bias_shape = _load_safetensor_bf16_header(
        ckpt, "model.layers.0.mlp.experts.down_proj_bias"
    )
    assert bias_dtype == "BF16"
    assert bias_shape[:2] == [32, 2880]

    rng = np.random.default_rng(0)
    a_f32 = rng.uniform(-1.0, 1.0, size=(P, K)).astype(np.float32)
    a_bf16 = _bf16_round_to_f32(a_f32)

    w_dense = _decode_mxfp4_rows(w_blocks[0], w_scales[0])
    w_dense = _bf16_round_to_f32(w_dense)

    ref = a_bf16 @ w_dense.T
    ref = _bf16_round_to_f32(ref)

    expert_start = np.array([0, P, P], dtype=np.uint32)
    expert_ids = np.array([0, -1], dtype=np.int32)
    expert_usage_stats = np.array([P, 1], dtype=np.uint32)
    devref = DeviceRef.from_device(device)
    session = InferenceSession(devices=[device])

    with Graph(
        "mxfp4_grouped_matmul_single_expert",
        input_types=[
            TensorType(DType.float32, shape=[P, K], device=devref),
            TensorType(DType.uint8, shape=w_blocks.shape, device=devref),
            TensorType(DType.uint8, shape=w_scales.shape, device=devref),
            TensorType(DType.uint32, shape=[num_experts + 1], device=devref),
            TensorType(DType.int32, shape=[num_experts], device=devref),
            TensorType(DType.uint32, shape=[2], device=devref),
        ],
        custom_extensions=[get_mxfp4_kernels_path()],
    ) as graph:
        a_in, blocks_in, scales_in, start_in, ids_in, stats_in = graph.inputs
        a_bf16_in = ops.cast(a_in.tensor, DType.bfloat16)
        out_bf16 = mxfp4_grouped_matmul_ragged_bf16(
            a_bf16_in,
            blocks_in.tensor,
            scales_in.tensor,
            start_in.tensor,
            ids_in.tensor,
            stats_in.tensor,
            target="gpu",
        )
        graph.output(ops.cast(out_bf16, DType.float32))

    model = session.load(graph)
    got = model.execute(
        Tensor.from_numpy(a_f32).to(device),
        Tensor.from_numpy(w_blocks).to(device),
        Tensor.from_numpy(w_scales).to(device),
        Tensor.from_numpy(expert_start).to(device),
        Tensor.from_numpy(expert_ids).to(device),
        Tensor.from_numpy(expert_usage_stats).to(device),
    )[0].to(CPU()).to_numpy()

    assert got.shape == ref.shape
    assert np.allclose(got, ref, atol=1e-1, rtol=1e-1), (
        f"max abs diff {np.max(np.abs(got - ref))}"
    )


def test_mxfp4_grouped_matmul_two_experts_segments_match_reference() -> None:
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
    P0 = 16
    P1 = 48
    P = P0 + P1
    K = 128
    N = 128
    k_blocks = K // MXFP4_VALUES_PER_BLOCK

    with safe_open(str(ckpt), framework="numpy") as f:
        w_blocks_all = f.get_tensor("model.layers.0.mlp.experts.down_proj_blocks")
        w_scales_all = f.get_tensor("model.layers.0.mlp.experts.down_proj_scales")

    w_blocks = w_blocks_all[:num_experts, :N, :k_blocks, :].copy()
    w_scales = w_scales_all[:num_experts, :N, :k_blocks].copy()

    rng = np.random.default_rng(1)
    a_f32 = rng.uniform(-1.0, 1.0, size=(P, K)).astype(np.float32)
    a_bf16 = _bf16_round_to_f32(a_f32)

    w0 = _bf16_round_to_f32(_decode_mxfp4_rows(w_blocks[0], w_scales[0]))
    w1 = _bf16_round_to_f32(_decode_mxfp4_rows(w_blocks[1], w_scales[1]))

    ref0 = a_bf16[:P0] @ w0.T
    ref1 = a_bf16[P0:] @ w1.T
    ref = np.concatenate([ref0, ref1], axis=0)
    ref = _bf16_round_to_f32(ref)

    expert_start = np.array([0, P0, P], dtype=np.uint32)
    expert_ids = np.array([0, 1], dtype=np.int32)
    expert_usage_stats = np.array([max(P0, P1), 2], dtype=np.uint32)
    devref = DeviceRef.from_device(device)
    session = InferenceSession(devices=[device])

    with Graph(
        "mxfp4_grouped_matmul_two_experts",
        input_types=[
            TensorType(DType.float32, shape=[P, K], device=devref),
            TensorType(DType.uint8, shape=w_blocks.shape, device=devref),
            TensorType(DType.uint8, shape=w_scales.shape, device=devref),
            TensorType(DType.uint32, shape=[num_experts + 1], device=devref),
            TensorType(DType.int32, shape=[num_experts], device=devref),
            TensorType(DType.uint32, shape=[2], device=devref),
        ],
        custom_extensions=[get_mxfp4_kernels_path()],
    ) as graph:
        a_in, blocks_in, scales_in, start_in, ids_in, stats_in = graph.inputs
        a_bf16_in = ops.cast(a_in.tensor, DType.bfloat16)
        out_bf16 = mxfp4_grouped_matmul_ragged_bf16(
            a_bf16_in,
            blocks_in.tensor,
            scales_in.tensor,
            start_in.tensor,
            ids_in.tensor,
            stats_in.tensor,
            target="gpu",
        )
        graph.output(ops.cast(out_bf16, DType.float32))

    model = session.load(graph)
    got = model.execute(
        Tensor.from_numpy(a_f32).to(device),
        Tensor.from_numpy(w_blocks).to(device),
        Tensor.from_numpy(w_scales).to(device),
        Tensor.from_numpy(expert_start).to(device),
        Tensor.from_numpy(expert_ids).to(device),
        Tensor.from_numpy(expert_usage_stats).to(device),
    )[0].to(CPU()).to_numpy()

    assert got.shape == ref.shape
    assert np.allclose(got, ref, atol=1e-1, rtol=1e-1), (
        f"max abs diff {np.max(np.abs(got - ref))}"
    )


def test_mxfp4_grouped_matmul_large_kblocks_matches_reference() -> None:
    """Regression guard: exercise many K tiles (K=2880) in WGMMA path."""
    try:
        device = Accelerator()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GPU not available: {exc}")

    ckpt = _find_gpt_oss_20b_file()
    if ckpt is None:
        pytest.skip(
            "GPT-OSS checkpoint not found in HF cache; run `pixi run generate` once"
        )

    num_experts = 1
    P = 257
    K = 2880
    N = 128
    k_blocks = K // MXFP4_VALUES_PER_BLOCK

    with safe_open(str(ckpt), framework="numpy") as f:
        w_blocks_all = f.get_tensor("model.layers.0.mlp.experts.down_proj_blocks")
        w_scales_all = f.get_tensor("model.layers.0.mlp.experts.down_proj_scales")

    w_blocks = w_blocks_all[:num_experts, :N, :k_blocks, :].copy()
    w_scales = w_scales_all[:num_experts, :N, :k_blocks].copy()

    rng = np.random.default_rng(2)
    a_f32 = rng.uniform(-1.0, 1.0, size=(P, K)).astype(np.float32)
    a_bf16 = _bf16_round_to_f32(a_f32)

    w_dense = _bf16_round_to_f32(_decode_mxfp4_rows(w_blocks[0], w_scales[0]))
    ref = _bf16_round_to_f32(a_bf16 @ w_dense.T)

    expert_start = np.array([0, P], dtype=np.uint32)
    expert_ids = np.array([0], dtype=np.int32)
    expert_usage_stats = np.array([P, 1], dtype=np.uint32)
    devref = DeviceRef.from_device(device)
    session = InferenceSession(devices=[device])

    with Graph(
        "mxfp4_grouped_matmul_large_k",
        input_types=[
            TensorType(DType.float32, shape=[P, K], device=devref),
            TensorType(DType.uint8, shape=w_blocks.shape, device=devref),
            TensorType(DType.uint8, shape=w_scales.shape, device=devref),
            TensorType(DType.uint32, shape=[num_experts + 1], device=devref),
            TensorType(DType.int32, shape=[num_experts], device=devref),
            TensorType(DType.uint32, shape=[2], device=devref),
        ],
        custom_extensions=[get_mxfp4_kernels_path()],
    ) as graph:
        a_in, blocks_in, scales_in, start_in, ids_in, stats_in = graph.inputs
        a_bf16_in = ops.cast(a_in.tensor, DType.bfloat16)
        out_bf16 = mxfp4_grouped_matmul_ragged_bf16(
            a_bf16_in,
            blocks_in.tensor,
            scales_in.tensor,
            start_in.tensor,
            ids_in.tensor,
            stats_in.tensor,
            target="gpu",
        )
        graph.output(ops.cast(out_bf16, DType.float32))

    model = session.load(graph)
    got = model.execute(
        Tensor.from_numpy(a_f32).to(device),
        Tensor.from_numpy(w_blocks).to(device),
        Tensor.from_numpy(w_scales).to(device),
        Tensor.from_numpy(expert_start).to(device),
        Tensor.from_numpy(expert_ids).to(device),
        Tensor.from_numpy(expert_usage_stats).to(device),
    )[0].to(CPU()).to_numpy()

    assert got.shape == ref.shape
    assert np.allclose(got, ref, atol=1e-1, rtol=1e-1), (
        f"max abs diff {np.max(np.abs(got - ref))}"
    )


def test_mxfp4_grouped_matmul_gate_up_wgmma_matches_reference() -> None:
    """Regression guard: WGMMA correctness for gate_up (N=5760 in model).

    We slice N down to 256 for test runtime but keep K=2880 and P=257 to
    exercise the WGMMA kernel path.
    """
    try:
        device = Accelerator()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GPU not available: {exc}")

    ckpt = _find_gpt_oss_20b_file()
    if ckpt is None:
        pytest.skip(
            "GPT-OSS checkpoint not found in HF cache; run `pixi run generate` once"
        )

    num_experts = 1
    P = 257
    K = 2880
    N = 256
    k_blocks = K // MXFP4_VALUES_PER_BLOCK

    with safe_open(str(ckpt), framework="numpy") as f:
        w_blocks_all = f.get_tensor("model.layers.0.mlp.experts.gate_up_proj_blocks")
        w_scales_all = f.get_tensor("model.layers.0.mlp.experts.gate_up_proj_scales")

    w_blocks = w_blocks_all[:num_experts, :N, :k_blocks, :].copy()
    w_scales = w_scales_all[:num_experts, :N, :k_blocks].copy()

    rng = np.random.default_rng(3)
    a_f32 = rng.uniform(-1.0, 1.0, size=(P, K)).astype(np.float32)
    a_bf16 = _bf16_round_to_f32(a_f32)

    w_dense = _bf16_round_to_f32(_decode_mxfp4_rows(w_blocks[0], w_scales[0]))
    ref = _bf16_round_to_f32(a_bf16 @ w_dense.T)

    expert_start = np.array([0, P], dtype=np.uint32)
    expert_ids = np.array([0], dtype=np.int32)
    expert_usage_stats = np.array([P, 1], dtype=np.uint32)
    devref = DeviceRef.from_device(device)
    session = InferenceSession(devices=[device])

    with Graph(
        "mxfp4_grouped_matmul_gate_up_wgmma",
        input_types=[
            TensorType(DType.float32, shape=[P, K], device=devref),
            TensorType(DType.uint8, shape=w_blocks.shape, device=devref),
            TensorType(DType.uint8, shape=w_scales.shape, device=devref),
            TensorType(DType.uint32, shape=[num_experts + 1], device=devref),
            TensorType(DType.int32, shape=[num_experts], device=devref),
            TensorType(DType.uint32, shape=[2], device=devref),
        ],
        custom_extensions=[get_mxfp4_kernels_path()],
    ) as graph:
        a_in, blocks_in, scales_in, start_in, ids_in, stats_in = graph.inputs
        a_bf16_in = ops.cast(a_in.tensor, DType.bfloat16)
        out_bf16 = mxfp4_grouped_matmul_ragged_bf16(
            a_bf16_in,
            blocks_in.tensor,
            scales_in.tensor,
            start_in.tensor,
            ids_in.tensor,
            stats_in.tensor,
            target="gpu",
        )
        graph.output(ops.cast(out_bf16, DType.float32))

    model = session.load(graph)
    got = model.execute(
        Tensor.from_numpy(a_f32).to(device),
        Tensor.from_numpy(w_blocks).to(device),
        Tensor.from_numpy(w_scales).to(device),
        Tensor.from_numpy(expert_start).to(device),
        Tensor.from_numpy(expert_ids).to(device),
        Tensor.from_numpy(expert_usage_stats).to(device),
    )[0].to(CPU()).to_numpy()

    assert got.shape == ref.shape
    assert np.allclose(got, ref, atol=1e-1, rtol=1e-1), (
        f"max abs diff {np.max(np.abs(got - ref))}"
    )
