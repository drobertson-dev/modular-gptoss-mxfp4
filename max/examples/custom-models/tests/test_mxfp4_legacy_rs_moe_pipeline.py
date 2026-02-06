"""RS-path MoE isolation tests for legacy GPT-OSS MXFP4 custom model."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from gpt_oss_mxfp4.kernels import (
    HOPPER_SCALE_NUM_WARPS,
    get_mxfp4_kernels_path,
    mxfp4_grouped_matmul_ragged_bf16_swizzled,
    mxfp4_moe_topk_reduce_bf16,
)
from gpt_oss_mxfp4_v3.weight_adapters import (
    _mxfp4_swizzle_scales_hopper,
    _mxfp4_swizzle_values_hopper,
    _mxfp4_unswizzle_scales_hopper,
    _mxfp4_unswizzle_values_hopper,
)
from max.driver import Buffer, CPU, Accelerator
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.legacy.kernels import moe_create_indices, scatter_nd_skip_oob_indices
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
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    rounding_bias = ((u >> 16) & 1).astype(np.uint32) + np.uint32(0x7FFF)
    rounded = (u + rounding_bias) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    ez = np.exp(z, dtype=np.float32)
    return ez / np.sum(ez, axis=axis, keepdims=True)


def _find_gpt_oss_20b_file() -> Path | None:
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


def _decode_dense_from_swizzled(
    blocks_swz: np.ndarray,
    scales_swz: np.ndarray,
    *,
    n_cols: int,
    k: int,
) -> np.ndarray:
    """Decode Hopper-swizzled MXFP4 weights into dense float32 [E, N, K]."""
    kblocks = k // 32
    kbytes = k // 2
    values = _mxfp4_unswizzle_values_hopper(
        blocks_swz, mx_axis=2, m=n_cols, k=kbytes
    )
    scales = _mxfp4_unswizzle_scales_hopper(
        scales_swz,
        num_warps=HOPPER_SCALE_NUM_WARPS,
        m=n_cols,
        kblocks=kblocks,
    )

    e_dim = values.shape[0]
    out = np.empty((e_dim, n_cols, k), dtype=np.float32)
    for e in range(e_dim):
        for n in range(n_cols):
            row_bytes = values[e, n]
            for kb in range(kblocks):
                scale = np.exp2(np.float32(np.int32(scales[e, n, kb]) - 127))
                base_k = kb * 32
                blk = row_bytes[kb * 16 : (kb + 1) * 16]
                for b in range(16):
                    packed = int(blk[b])
                    lo = packed & 0x0F
                    hi = packed >> 4
                    out[e, n, base_k + 2 * b] = FP4_VALUES[lo] * scale
                    out[e, n, base_k + 2 * b + 1] = FP4_VALUES[hi] * scale
    return out


def _build_case_synthetic(
    *,
    rng: np.random.Generator,
    tokens: int,
    topk: int,
    hidden: int,
    intermediate: int,
    num_experts: int,
) -> dict[str, np.ndarray]:
    x_f32 = rng.standard_normal((tokens, hidden), dtype=np.float32)
    router_scores = rng.standard_normal((tokens, num_experts), dtype=np.float32)
    router_topk = np.argsort(-router_scores, axis=1)[:, :topk].astype(np.int32)
    router_topk_scores = np.take_along_axis(router_scores, router_topk, axis=1)
    gate_weights = _softmax(router_topk_scores, axis=1).astype(np.float32)

    w1_kbytes = hidden // 2
    w1_kblocks = hidden // 32
    w1_blocks_logical = rng.integers(
        0,
        256,
        size=(num_experts, 2 * intermediate, w1_kbytes),
        dtype=np.uint8,
    )
    w1_scales_logical = rng.integers(
        120,
        134,
        size=(num_experts, 2 * intermediate, w1_kblocks),
        dtype=np.uint8,
    )
    w1_blocks_swz = _mxfp4_swizzle_values_hopper(w1_blocks_logical, mx_axis=2)
    w1_scales_swz = _mxfp4_swizzle_scales_hopper(w1_scales_logical)
    w1_bias = _bf16_round_to_f32(
        rng.standard_normal((num_experts, 2 * intermediate), dtype=np.float32)
    )

    w2_kbytes = intermediate // 2
    w2_kblocks = intermediate // 32
    w2_blocks_logical = rng.integers(
        0, 256, size=(num_experts, hidden, w2_kbytes), dtype=np.uint8
    )
    w2_scales_logical = rng.integers(
        120, 134, size=(num_experts, hidden, w2_kblocks), dtype=np.uint8
    )
    w2_blocks_swz = _mxfp4_swizzle_values_hopper(w2_blocks_logical, mx_axis=2)
    w2_scales_swz = _mxfp4_swizzle_scales_hopper(w2_scales_logical)
    w2_bias = _bf16_round_to_f32(
        rng.standard_normal((num_experts, hidden), dtype=np.float32)
    )

    return {
        "x_f32": x_f32,
        "router_topk": router_topk,
        "gate_weights": gate_weights,
        "w1_blocks_swz": w1_blocks_swz,
        "w1_scales_swz": w1_scales_swz,
        "w1_bias": w1_bias,
        "w2_blocks_swz": w2_blocks_swz,
        "w2_scales_swz": w2_scales_swz,
        "w2_bias": w2_bias,
    }


def _build_case_checkpoint(
    *,
    rng: np.random.Generator,
    tokens: int,
    topk: int,
    hidden: int,
    intermediate: int,
    num_experts: int,
) -> dict[str, np.ndarray]:
    ckpt = _find_gpt_oss_20b_file()
    if ckpt is None:
        pytest.skip(
            "GPT-OSS checkpoint not found in HF cache; run `pixi run generate` once"
        )

    x_f32 = rng.standard_normal((tokens, hidden), dtype=np.float32)
    router_scores = rng.standard_normal((tokens, num_experts), dtype=np.float32)
    router_topk = np.argsort(-router_scores, axis=1)[:, :topk].astype(np.int32)
    router_topk_scores = np.take_along_axis(router_scores, router_topk, axis=1)
    gate_weights = _softmax(router_topk_scores, axis=1).astype(np.float32)

    w1_kbytes = hidden // 2
    w1_kblocks = hidden // 32
    w2_kbytes = intermediate // 2
    w2_kblocks = intermediate // 32

    def _normalize_blocks_to_row_bytes(
        arr: np.ndarray, *, n_cols: int, kbytes: int, kblocks: int, name: str
    ) -> np.ndarray:
        arr = np.ascontiguousarray(arr)
        if arr.ndim == 4 and arr.shape[-1] == 16:
            if arr.shape[2] < kblocks:
                raise ValueError(
                    f"{name}: expected at least {kblocks} K-blocks, got {arr.shape[2]}"
                )
            view = arr[:num_experts, :n_cols, :kblocks, :]
            return np.ascontiguousarray(view.reshape(num_experts, n_cols, kbytes))
        if arr.ndim == 3:
            if arr.shape[2] < kbytes:
                raise ValueError(
                    f"{name}: expected at least {kbytes} K-bytes, got {arr.shape[2]}"
                )
            return np.ascontiguousarray(arr[:num_experts, :n_cols, :kbytes])
        raise ValueError(
            f"{name}: unsupported blocks rank/shape {arr.shape}; expected [E,N,K/2] or [E,N,K/32,16]"
        )

    try:
        import torch
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"torch is required for checkpoint BF16 tensors: {exc}")

    with safe_open(str(ckpt), framework="pt") as f:
        w1_blocks_logical = _normalize_blocks_to_row_bytes(
            f.get_tensor("model.layers.0.mlp.experts.gate_up_proj_blocks")
            .cpu()
            .numpy(),
            n_cols=2 * intermediate,
            kbytes=w1_kbytes,
            kblocks=w1_kblocks,
            name="gate_up_proj_blocks",
        )
        w1_scales_logical = np.ascontiguousarray(
            f.get_tensor("model.layers.0.mlp.experts.gate_up_proj_scales")
            .cpu()
            .numpy()[
                :num_experts, : 2 * intermediate, :w1_kblocks
            ]
        )
        w1_bias = _bf16_round_to_f32(
            np.ascontiguousarray(
                f.get_tensor("model.layers.0.mlp.experts.gate_up_proj_bias")
                .to(dtype=torch.float32)
                .cpu()
                .numpy()[
                    :num_experts, : 2 * intermediate
                ]
            ).astype(np.float32)
        )

        w2_blocks_logical = _normalize_blocks_to_row_bytes(
            f.get_tensor("model.layers.0.mlp.experts.down_proj_blocks")
            .cpu()
            .numpy(),
            n_cols=hidden,
            kbytes=w2_kbytes,
            kblocks=w2_kblocks,
            name="down_proj_blocks",
        )
        w2_scales_logical = np.ascontiguousarray(
            f.get_tensor("model.layers.0.mlp.experts.down_proj_scales")
            .cpu()
            .numpy()[
                :num_experts, :hidden, :w2_kblocks
            ]
        )
        w2_bias = _bf16_round_to_f32(
            np.ascontiguousarray(
                f.get_tensor("model.layers.0.mlp.experts.down_proj_bias")
                .to(dtype=torch.float32)
                .cpu()
                .numpy()[
                    :num_experts, :hidden
                ]
            ).astype(np.float32)
        )

    return {
        "x_f32": x_f32,
        "router_topk": router_topk,
        "gate_weights": gate_weights,
        "w1_blocks_swz": _mxfp4_swizzle_values_hopper(
            w1_blocks_logical, mx_axis=2
        ),
        "w1_scales_swz": _mxfp4_swizzle_scales_hopper(w1_scales_logical),
        "w1_bias": w1_bias,
        "w2_blocks_swz": _mxfp4_swizzle_values_hopper(
            w2_blocks_logical, mx_axis=2
        ),
        "w2_scales_swz": _mxfp4_swizzle_scales_hopper(w2_scales_logical),
        "w2_bias": w2_bias,
    }


def _run_rs_pipeline(
    *,
    device: Accelerator,
    tokens: int,
    topk: int,
    hidden: int,
    intermediate: int,
    num_experts: int,
    case: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    devref = DeviceRef.from_device(device)
    session = InferenceSession(devices=[device])

    with Graph(
        "legacy_rs_moe_pipeline_isolation",
        input_types=[
            TensorType(DType.float32, shape=[tokens, hidden], device=devref),
            TensorType(DType.int32, shape=[tokens, topk], device=devref),
            TensorType(DType.float32, shape=[tokens, topk], device=devref),
            TensorType(
                DType.uint8, shape=case["w1_blocks_swz"].shape, device=devref
            ),
            TensorType(
                DType.uint8, shape=case["w1_scales_swz"].shape, device=devref
            ),
            TensorType(DType.float32, shape=case["w1_bias"].shape, device=devref),
            TensorType(
                DType.uint8, shape=case["w2_blocks_swz"].shape, device=devref
            ),
            TensorType(
                DType.uint8, shape=case["w2_scales_swz"].shape, device=devref
            ),
            TensorType(DType.float32, shape=case["w2_bias"].shape, device=devref),
        ],
        custom_extensions=[get_mxfp4_kernels_path()],
    ) as graph:
        (
            x_in,
            router_idx_in,
            gate_w_in,
            w1_blocks_in,
            w1_scales_in,
            w1_bias_in,
            w2_blocks_in,
            w2_scales_in,
            w2_bias_in,
        ) = graph.inputs

        x_bf16 = ops.cast(x_in.tensor, DType.bfloat16)
        router_idx_flat = ops.reshape(router_idx_in.tensor, [-1])
        gate_weights_flat = ops.reshape(gate_w_in.tensor, [-1])
        gate_weights_bf16 = ops.cast(gate_weights_flat, DType.bfloat16)

        (
            token_expert_order,
            expert_start_indices,
            restore_token_order,
            expert_ids,
            expert_usage_stats,
        ) = moe_create_indices(router_idx_flat, num_experts)

        token_expert_order_i32 = ops.cast(token_expert_order, DType.int32)
        token_rows = ops.cast(token_expert_order_i32 // topk, DType.int32)
        permutated_states = ops.gather(x_bf16, token_rows, axis=0)

        gate_up_output = mxfp4_grouped_matmul_ragged_bf16_swizzled(
            permutated_states,
            w1_blocks_in.tensor,
            w1_scales_in.tensor,
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
            n_cols=2 * intermediate,
            target="gpu",
            no_small_m=True,
        )
        expert_assignments = ops.gather(
            router_idx_flat, token_expert_order_i32, axis=0
        )
        gate_bias = ops.gather(
            ops.cast(w1_bias_in.tensor, DType.bfloat16),
            expert_assignments,
            axis=0,
        )
        gate_up_output = gate_up_output + gate_bias

        gate = gate_up_output[:, 0::2]
        up = gate_up_output[:, 1::2]
        limit_pos = ops.constant(7.0, dtype=gate.dtype, device=gate.device)
        limit_neg = ops.constant(-7.0, dtype=up.dtype, device=up.device)
        alpha = ops.constant(1.702, dtype=gate.dtype, device=gate.device)
        one = ops.constant(1.0, dtype=up.dtype, device=up.device)
        gate = ops.min(gate, limit_pos)
        up = ops.min(ops.max(up, limit_neg), limit_pos)
        glu = gate * ops.sigmoid(gate * alpha)
        gated_output = (up + one) * glu

        down_output = mxfp4_grouped_matmul_ragged_bf16_swizzled(
            gated_output,
            w2_blocks_in.tensor,
            w2_scales_in.tensor,
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
            n_cols=hidden,
            target="gpu",
            no_small_m=True,
        )
        down_bias = ops.gather(
            ops.cast(w2_bias_in.tensor, DType.bfloat16),
            expert_assignments,
            axis=0,
        )
        down_output = down_output + down_bias

        gate_weights_sorted = ops.gather(
            gate_weights_bf16, token_expert_order_i32, axis=0
        )
        down_output = down_output * ops.unsqueeze(gate_weights_sorted, -1)

        restore_indices = ops.cast(restore_token_order, DType.int32)
        restore_indices_2d = ops.unsqueeze(restore_indices, -1)
        zeros = ops.broadcast_to(
            ops.constant(
                0, dtype=down_output.dtype, device=down_output.device
            ),
            [tokens * topk, down_output.shape[1]],
        )
        y_pairs = scatter_nd_skip_oob_indices(
            input=zeros,
            updates=down_output,
            indices=restore_indices_2d,
        )
        y = mxfp4_moe_topk_reduce_bf16(x_bf16, y_pairs, target="gpu")

        graph.output(
            ops.cast(y, DType.float32),
            ops.cast(gate_up_output, DType.float32),
            ops.cast(gated_output, DType.float32),
            ops.cast(down_output, DType.float32),
            ops.cast(y_pairs, DType.float32),
            token_expert_order,
            restore_token_order,
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
        )

    (
        y_gpu,
        gate_up_gpu,
        gated_gpu,
        down_gpu,
        y_pairs_gpu,
        token_expert_order_gpu,
        restore_order_gpu,
        expert_start_gpu,
        expert_ids_gpu,
        expert_usage_gpu,
    ) = session.load(graph).execute(
        Buffer.from_numpy(case["x_f32"]).to(device),
        Buffer.from_numpy(case["router_topk"]).to(device),
        Buffer.from_numpy(case["gate_weights"]).to(device),
        Buffer.from_numpy(case["w1_blocks_swz"]).to(device),
        Buffer.from_numpy(case["w1_scales_swz"]).to(device),
        Buffer.from_numpy(case["w1_bias"]).to(device),
        Buffer.from_numpy(case["w2_blocks_swz"]).to(device),
        Buffer.from_numpy(case["w2_scales_swz"]).to(device),
        Buffer.from_numpy(case["w2_bias"]).to(device),
    )

    return {
        "y": y_gpu.to(CPU()).to_numpy(),
        "gate_up": gate_up_gpu.to(CPU()).to_numpy(),
        "gated": gated_gpu.to(CPU()).to_numpy(),
        "down": down_gpu.to(CPU()).to_numpy(),
        "y_pairs": y_pairs_gpu.to(CPU()).to_numpy(),
        "token_expert_order": token_expert_order_gpu.to(CPU()).to_numpy(),
        "restore_order": restore_order_gpu.to(CPU()).to_numpy(),
        "expert_start": expert_start_gpu.to(CPU()).to_numpy(),
        "expert_ids": expert_ids_gpu.to(CPU()).to_numpy(),
        "expert_usage": expert_usage_gpu.to(CPU()).to_numpy(),
    }


def _dense_reference(
    *,
    tokens: int,
    topk: int,
    hidden: int,
    intermediate: int,
    case: dict[str, np.ndarray],
    token_expert_order: np.ndarray,
    restore_order: np.ndarray,
) -> np.ndarray:
    pairs = tokens * topk
    w1_dense = _bf16_round_to_f32(
        _decode_dense_from_swizzled(
            case["w1_blocks_swz"],
            case["w1_scales_swz"],
            n_cols=2 * intermediate,
            k=hidden,
        )
    )
    w2_dense = _bf16_round_to_f32(
        _decode_dense_from_swizzled(
            case["w2_blocks_swz"],
            case["w2_scales_swz"],
            n_cols=hidden,
            k=intermediate,
        )
    )

    x_bf16 = _bf16_round_to_f32(case["x_f32"])
    router_idx_flat = case["router_topk"].reshape(-1)
    gate_weights_flat = _bf16_round_to_f32(case["gate_weights"].reshape(-1))

    token_rows = token_expert_order // topk
    expert_assignments = router_idx_flat[token_expert_order]
    gate_weights_sorted = gate_weights_flat[token_expert_order]

    x_perm = x_bf16[token_rows]
    gate_up = np.empty((pairs, 2 * intermediate), dtype=np.float32)
    for i in range(pairs):
        e = int(expert_assignments[i])
        gate_up[i] = x_perm[i] @ w1_dense[e].T
    gate_up = _bf16_round_to_f32(gate_up + case["w1_bias"][expert_assignments])

    gate = np.minimum(gate_up[:, 0::2], 7.0)
    up = np.clip(gate_up[:, 1::2], -7.0, 7.0)
    # Keep stable dtype behavior for reference sigmoid.
    sigmoid = 1.0 / (1.0 + np.exp(-(gate * 1.702).astype(np.float32)))
    gated = _bf16_round_to_f32((up + 1.0) * (gate * sigmoid))

    down = np.empty((pairs, hidden), dtype=np.float32)
    for i in range(pairs):
        e = int(expert_assignments[i])
        down[i] = gated[i] @ w2_dense[e].T
    down = _bf16_round_to_f32(down + case["w2_bias"][expert_assignments])
    down = _bf16_round_to_f32(
        down * gate_weights_sorted[:, np.newaxis].astype(np.float32)
    )

    y_pairs = np.zeros((pairs, hidden), dtype=np.float32)
    y_pairs[restore_order] = down
    return _bf16_round_to_f32(y_pairs.reshape(tokens, topk, hidden).sum(axis=1))


def _assert_index_invariants(
    *,
    outputs: dict[str, np.ndarray],
    pairs: int,
    num_experts: int,
) -> None:
    token_expert_order = outputs["token_expert_order"]
    restore_order = outputs["restore_order"]
    expert_start = outputs["expert_start"]
    expert_usage = outputs["expert_usage"]

    assert token_expert_order.shape == (pairs,)
    assert restore_order.shape == (pairs,)
    assert expert_start.shape == (num_experts + 1,)
    assert np.min(token_expert_order) >= 0
    assert np.max(token_expert_order) < pairs
    assert np.min(restore_order) >= 0
    assert np.max(restore_order) < pairs
    assert np.all(expert_start[1:] >= expert_start[:-1])
    assert expert_start[0] == 0
    assert expert_start[-1] == pairs
    assert int(expert_usage[0]) <= pairs
    assert int(expert_usage[1]) <= num_experts


def _assert_finite_stages(outputs: dict[str, np.ndarray]) -> None:
    for stage in ("gate_up", "gated", "down", "y_pairs", "y"):
        arr = outputs[stage]
        finite = np.isfinite(arr)
        if not finite.all():
            bad = np.argwhere(~finite)
            first = tuple(int(v) for v in bad[0])
            raise AssertionError(
                f"first non-finite at stage={stage}, index={first}, "
                f"value={arr[first]}"
            )


def _run_and_compare(
    *,
    case: dict[str, np.ndarray],
    tokens: int,
    topk: int,
    hidden: int,
    intermediate: int,
    num_experts: int,
) -> None:
    try:
        device = Accelerator()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GPU not available: {exc}")

    outputs = _run_rs_pipeline(
        device=device,
        tokens=tokens,
        topk=topk,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        case=case,
    )
    pairs = tokens * topk
    _assert_index_invariants(outputs=outputs, pairs=pairs, num_experts=num_experts)
    _assert_finite_stages(outputs)

    y_ref = _dense_reference(
        tokens=tokens,
        topk=topk,
        hidden=hidden,
        intermediate=intermediate,
        case=case,
        token_expert_order=outputs["token_expert_order"],
        restore_order=outputs["restore_order"],
    )
    assert np.isfinite(y_ref).all()
    assert np.allclose(outputs["y"], y_ref, atol=0.6, rtol=0.2), (
        f"max abs diff {np.max(np.abs(outputs['y'] - y_ref))}"
    )


def test_legacy_rs_moe_pipeline_synthetic_matches_dense_reference() -> None:
    if os.environ.get("MXFP4_RS_ISOLATION_ENABLE", "0") != "1":
        pytest.skip(
            "RS isolation diagnostic is opt-in; set MXFP4_RS_ISOLATION_ENABLE=1"
        )

    tokens = 128
    topk = 4
    hidden = 128
    intermediate = 128
    num_experts = 8
    rng = np.random.default_rng(1234)

    case = _build_case_synthetic(
        rng=rng,
        tokens=tokens,
        topk=topk,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
    )
    _run_and_compare(
        case=case,
        tokens=tokens,
        topk=topk,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
    )


def test_legacy_rs_moe_pipeline_checkpoint_matches_dense_reference() -> None:
    if os.environ.get("MXFP4_RS_ISOLATION_CHECKPOINT", "0") != "1":
        pytest.skip(
            "Checkpoint RS isolation is opt-in; set MXFP4_RS_ISOLATION_CHECKPOINT=1"
        )

    tokens = 128
    topk = 4
    hidden = 128
    intermediate = 128
    num_experts = 8
    rng = np.random.default_rng(777)

    case = _build_case_checkpoint(
        rng=rng,
        tokens=tokens,
        topk=topk,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
    )
    _run_and_compare(
        case=case,
        tokens=tokens,
        topk=topk,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
    )
