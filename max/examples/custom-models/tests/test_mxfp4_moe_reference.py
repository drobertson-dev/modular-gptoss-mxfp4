"""Reference check for the MXFP4 MoE custom ops against numpy decoding.

This uses a small slice of the real `openai/gpt-oss-20b` checkpoint tensors to
validate the Triton-style MoE path:

  W1+SwiGLU -> W2 per-pair output -> TOPK reduction
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest
from gpt_oss_mxfp4.kernels import get_mxfp4_kernels_path
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


def _decode_mxfp4_rows(
    blocks: np.ndarray, scales_exp: np.ndarray
) -> np.ndarray:
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
    k = k_blocks * 32
    out = np.empty((n_rows, k), dtype=np.float32)

    for r in range(n_rows):
        for kb in range(k_blocks):
            scale = np.exp2(np.float32(np.int32(scales_exp[r, kb]) - 127))
            blk = blocks[r, kb]
            for b in range(16):
                packed = int(blk[b])
                lo = packed & 0x0F
                hi = packed >> 4
                k0 = kb * 32 + 2 * b
                out[r, k0] = FP4_VALUES[lo] * scale
                out[r, k0 + 1] = FP4_VALUES[hi] * scale
    return out


def _load_safetensor_bf16_as_f32(path: Path, key: str) -> np.ndarray:
    """Load BF16 tensor from a safetensors file into float32 numpy.

    `safetensors.safe_open(..., framework="numpy")` cannot load BF16 with
    NumPy<2.0, so we decode BF16 manually from the raw bytes.
    """
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        if key not in header:
            raise KeyError(f"{key} not found in {path}")

        info = header[key]
        if info["dtype"] != "BF16":
            raise ValueError(f"{key} expected BF16, got {info['dtype']}")

        start, end = info["data_offsets"]
        data_start = 8 + header_len
        f.seek(data_start + start)
        raw = f.read(end - start)

    u16 = np.frombuffer(raw, dtype=np.uint16).copy()
    u32 = (u16.astype(np.uint32) << 16).view(np.uint32)
    f32 = u32.view(np.float32)
    return f32.reshape(tuple(info["shape"]))


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


def _swiglu_interleaved(
    x: np.ndarray, bias: np.ndarray, *, alpha: float, limit: float
) -> np.ndarray:
    """Interleaved SwiGLU reference: even=gate, odd=linear."""
    gate = x[:, 0::2] + bias[0::2]
    lin = x[:, 1::2] + bias[1::2]
    gate = np.minimum(gate, limit)
    lin = np.clip(lin, -limit, limit)
    out_glu = gate * (1.0 / (1.0 + np.exp(-alpha * gate)))
    return out_glu * (lin + 1.0)


def test_mxfp4_moe_ops_match_numpy_reference() -> None:
    try:
        device = Accelerator()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GPU not available: {exc}")

    ckpt = _find_gpt_oss_20b_file()
    if ckpt is None:
        pytest.skip(
            "GPT-OSS checkpoint not found in HF cache; run `pixi run generate` once"
        )

    # Small slice sizes (must respect MXFP4 32-value blocks and kernel tile constraints).
    num_experts = 2
    t_tokens = 1
    d_hidden = 128
    i_moe = 128
    n_raw = 2 * i_moe
    kblocks_w1 = d_hidden // 32
    kblocks_w2 = i_moe // 32

    # Load checkpoint tensors (uint8 blocks/scales via safetensors; BF16 bias via raw bytes).
    with safe_open(str(ckpt), framework="numpy") as f:
        w1_blocks_all = f.get_tensor(
            "model.layers.0.mlp.experts.gate_up_proj_blocks"
        )
        w1_scales_all = f.get_tensor(
            "model.layers.0.mlp.experts.gate_up_proj_scales"
        )
        w2_blocks_all = f.get_tensor(
            "model.layers.0.mlp.experts.down_proj_blocks"
        )
        w2_scales_all = f.get_tensor(
            "model.layers.0.mlp.experts.down_proj_scales"
        )

    w1_bias_all = _load_safetensor_bf16_as_f32(
        ckpt, "model.layers.0.mlp.experts.gate_up_proj_bias"
    )
    w2_bias_all = _load_safetensor_bf16_as_f32(
        ckpt, "model.layers.0.mlp.experts.down_proj_bias"
    )

    w1_blocks = w1_blocks_all[:num_experts, :n_raw, :kblocks_w1, :].copy()
    w1_scales = w1_scales_all[:num_experts, :n_raw, :kblocks_w1].copy()
    w1_bias = w1_bias_all[:num_experts, :n_raw].copy()

    w2_blocks = w2_blocks_all[:num_experts, :d_hidden, :kblocks_w2, :].copy()
    w2_scales = w2_scales_all[:num_experts, :d_hidden, :kblocks_w2].copy()
    w2_bias = w2_bias_all[:num_experts, :d_hidden].copy()

    # Reference computation (bf16 inputs/weights, f32 accumulate, bf16 activation output).
    rng = np.random.default_rng(0)
    x_f32 = rng.uniform(-1.0, 1.0, size=(t_tokens, d_hidden)).astype(np.float32)
    x_bf16 = _bf16_round_to_f32(x_f32)

    w1_dense = _decode_mxfp4_rows(w1_blocks[0], w1_scales[0])
    w1_dense = _bf16_round_to_f32(w1_dense)
    raw = x_bf16 @ w1_dense.T
    h = _swiglu_interleaved(raw, w1_bias[0], alpha=1.702, limit=7.0)
    h_bf16 = _bf16_round_to_f32(h)

    w2_dense = _decode_mxfp4_rows(w2_blocks[0], w2_scales[0])
    w2_dense = _bf16_round_to_f32(w2_dense)
    ref = (h_bf16 @ w2_dense.T) + w2_bias[0]

    # One token with TOPK=4, all routed to expert 0 with weights that sum to 1.0.
    # This makes the final reduced output equal to the single-expert reference.
    token_expert_order = np.array([0, 1, 2, 3], dtype=np.uint32)
    expert_start = np.array([0, 4, 4], dtype=np.uint32)
    expert_ids = np.array([0, -1], dtype=np.int32)
    expert_usage_stats = np.array([4, 1], dtype=np.uint32)
    gate_weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)

    devref = DeviceRef.from_device(device)
    session = InferenceSession(devices=[device])

    with Graph(
        "mxfp4_moe_reference_bf16",
        input_types=[
            TensorType(
                DType.float32, shape=[t_tokens, d_hidden], device=devref
            ),
            TensorType(
                DType.uint32, shape=[4], device=devref
            ),  # token_expert_order
            TensorType(
                DType.uint32, shape=[num_experts + 1], device=devref
            ),  # expert_start
            TensorType(
                DType.int32, shape=[num_experts], device=devref
            ),  # expert_ids
            TensorType(
                DType.uint32, shape=[2], device=devref
            ),  # expert_usage_stats
            TensorType(
                DType.uint8,
                shape=[num_experts, n_raw, kblocks_w1, 16],
                device=devref,
            ),
            TensorType(
                DType.uint8,
                shape=[num_experts, n_raw, kblocks_w1],
                device=devref,
            ),
            TensorType(
                DType.float32,
                shape=[num_experts, n_raw],
                device=devref,
            ),
            TensorType(DType.float32, shape=[4], device=devref),  # gate_weights
            TensorType(
                DType.uint8,
                shape=[num_experts, d_hidden, kblocks_w2, 16],
                device=devref,
            ),
            TensorType(
                DType.uint8,
                shape=[num_experts, d_hidden, kblocks_w2],
                device=devref,
            ),
            TensorType(
                DType.float32,
                shape=[num_experts, d_hidden],
                device=devref,
            ),
        ],
        custom_extensions=[get_mxfp4_kernels_path()],
    ) as graph_bf16:
        (
            x_in,
            order_in,
            expert_start_in,
            expert_ids_in,
            expert_usage_stats_in,
            w1_blocks_in,
            w1_scales_in,
            w1_bias_in,
            gate_in,
            w2_blocks_in,
            w2_scales_in,
            w2_bias_in,
        ) = graph_bf16.inputs

        x_bf16_in = ops.cast(x_in.tensor, DType.bfloat16)
        gate_bf16 = ops.cast(gate_in.tensor, DType.bfloat16)
        alpha_val = ops.constant(
            1.702, dtype=DType.float32, device=DeviceRef.CPU()
        )
        limit_val = ops.constant(
            7.0, dtype=DType.float32, device=DeviceRef.CPU()
        )

        h_sorted_type = TensorType(
            dtype=DType.bfloat16,
            shape=[4, i_moe],
            device=devref,
        )
        h_sorted = ops.custom(
            "mxfp4_moe_w1_swiglu",
            device=devref,
            values=[
                x_bf16_in,
                order_in.tensor,
                expert_start_in.tensor,
                expert_ids_in.tensor,
                expert_usage_stats_in.tensor,
                w1_blocks_in.tensor,
                w1_scales_in.tensor,
                w1_bias_in.tensor,
                alpha_val,
                limit_val,
            ],
            out_types=[h_sorted_type],
            parameters={"target": "gpu"},
        )[0].tensor

        y_pairs_bf16_type = TensorType(
            dtype=DType.bfloat16,
            shape=[4, d_hidden],
            device=devref,
        )
        y_pairs_bf16 = ops.custom(
            "mxfp4_moe_w2_pairs_bf16",
            device=devref,
            values=[
                h_sorted,
                order_in.tensor,
                expert_start_in.tensor,
                expert_ids_in.tensor,
                expert_usage_stats_in.tensor,
                gate_bf16,
                w2_blocks_in.tensor,
                w2_scales_in.tensor,
                w2_bias_in.tensor,
            ],
            out_types=[y_pairs_bf16_type],
            parameters={"target": "gpu"},
        )[0].tensor

        y_bf16_type = TensorType(
            dtype=DType.bfloat16,
            shape=[t_tokens, d_hidden],
            device=devref,
        )
        y_bf16 = ops.custom(
            "mxfp4_moe_topk_reduce_bf16",
            device=devref,
            values=[y_pairs_bf16],
            out_types=[y_bf16_type],
            parameters={"target": "gpu"},
        )[0].tensor
        y_bf16_f32 = ops.cast(y_bf16, DType.float32)

        graph_bf16.output(y_bf16_f32)

    model_bf16 = session.load(graph_bf16)
    got_bf16 = model_bf16.execute(
        Tensor.from_numpy(x_f32).to(device),
        Tensor.from_numpy(token_expert_order).to(device),
        Tensor.from_numpy(expert_start).to(device),
        Tensor.from_numpy(expert_ids).to(device),
        Tensor.from_numpy(expert_usage_stats).to(device),
        Tensor.from_numpy(w1_blocks).to(device),
        Tensor.from_numpy(w1_scales).to(device),
        Tensor.from_numpy(w1_bias).to(device),
        Tensor.from_numpy(gate_weights).to(device),
        Tensor.from_numpy(w2_blocks).to(device),
        Tensor.from_numpy(w2_scales).to(device),
        Tensor.from_numpy(w2_bias).to(device),
    )[0].to(CPU()).to_numpy()

    bf16_atol = 1e-1
    bf16_rtol = 1e-1
    assert np.allclose(
        got_bf16, ref, atol=bf16_atol, rtol=bf16_rtol
    ), f"bf16 max abs diff {np.max(np.abs(got_bf16 - ref))}"
