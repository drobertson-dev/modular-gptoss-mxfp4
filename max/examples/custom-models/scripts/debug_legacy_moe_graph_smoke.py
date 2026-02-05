"""Graph-level smoke test for legacy MXFP4 MoE custom ops.

Builds a minimal graph with only:
  W1 fused SwiGLU -> W2 pair matmul -> TOPK reduce
using synthetic inputs so we can isolate custom-op execution from serving.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from safetensors import safe_open

from gpt_oss_mxfp4.kernels import (
    MXFP4_TOPK,
    MXFP4_VALUES_PER_BLOCK,
    get_mxfp4_kernels_path,
    mxfp4_moe_topk_reduce_bf16,
    mxfp4_moe_w1_swiglu,
    mxfp4_moe_w2_pairs_bf16,
)
from max.driver import Buffer, CPU, Accelerator
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.legacy.kernels import moe_create_indices


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Legacy MXFP4 MoE graph smoke test"
    )
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=2880)
    parser.add_argument("--intermediate", type=int, default=2880)
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--snapshot-dir", type=str, default="")
    parser.add_argument("--route-expert-id", type=int, default=0)
    parser.add_argument("--sweep-experts", action="store_true")
    parser.add_argument("--random-routing", action="store_true")
    return parser.parse_args()


def _mark(msg: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def _rand_bf16_like(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return rng.standard_normal(shape).astype(np.float32).astype(np.float16).astype(
        np.float32
    )


def _resolve_snapshot_dir(override: str) -> Path:
    if override:
        return Path(override)
    base = Path("/workspace/hf-cache/hub/models--openai--gpt-oss-20b")
    ref = (base / "refs" / "main").read_text().strip()
    return base / "snapshots" / ref


def _load_weight_from_snapshot(snapshot_dir: Path, key: str) -> np.ndarray:
    with (snapshot_dir / "model.safetensors.index.json").open("r") as f:
        weight_map = json.load(f)["weight_map"]
    shard = snapshot_dir / weight_map[key]
    try:
        with safe_open(str(shard), framework="numpy") as st:
            return st.get_tensor(key)
    except TypeError:
        # safetensors/numpy cannot decode BF16 directly; use torch fallback.
        with safe_open(str(shard), framework="pt") as st:
            return st.get_tensor(key).float().cpu().numpy()


def main() -> None:
    args = _parse_args()
    if args.hidden % MXFP4_VALUES_PER_BLOCK != 0:
        raise SystemExit("hidden must be divisible by 32")
    if args.intermediate % MXFP4_VALUES_PER_BLOCK != 0:
        raise SystemExit("intermediate must be divisible by 32")

    rng = np.random.default_rng(args.seed)
    device = Accelerator()
    devref = DeviceRef.from_device(device)
    session = InferenceSession(devices=[device])

    t = args.tokens
    d = args.hidden
    i = args.intermediate
    e = args.experts
    p = t * MXFP4_TOPK

    kblocks_w1 = d // MXFP4_VALUES_PER_BLOCK
    kblocks_w2 = i // MXFP4_VALUES_PER_BLOCK

    x = _rand_bf16_like(rng, (t, d))
    gate_weights = _rand_bf16_like(rng, (p,))

    if args.use_checkpoint:
        _mark("load checkpoint weights")
        snapshot_dir = _resolve_snapshot_dir(args.snapshot_dir)
        prefix = f"model.layers.{args.layer}.mlp.experts"
        w1_blocks = _load_weight_from_snapshot(
            snapshot_dir, f"{prefix}.gate_up_proj_blocks"
        ).astype(np.uint8, copy=False)
        w1_scales = _load_weight_from_snapshot(
            snapshot_dir, f"{prefix}.gate_up_proj_scales"
        ).astype(np.uint8, copy=False)
        w1_bias = _load_weight_from_snapshot(
            snapshot_dir, f"{prefix}.gate_up_proj_bias"
        ).astype(np.float32, copy=False)
        w2_blocks = _load_weight_from_snapshot(
            snapshot_dir, f"{prefix}.down_proj_blocks"
        ).astype(np.uint8, copy=False)
        w2_scales = _load_weight_from_snapshot(
            snapshot_dir, f"{prefix}.down_proj_scales"
        ).astype(np.uint8, copy=False)
        w2_bias = _load_weight_from_snapshot(
            snapshot_dir, f"{prefix}.down_proj_bias"
        ).astype(np.float32, copy=False)
        if w1_blocks.shape != (e, 2 * i, kblocks_w1, 16):
            raise ValueError(f"Unexpected W1 blocks shape: {w1_blocks.shape}")
        if w2_blocks.shape != (e, d, kblocks_w2, 16):
            raise ValueError(f"Unexpected W2 blocks shape: {w2_blocks.shape}")
    else:
        w1_blocks = rng.integers(
            0, 256, size=(e, 2 * i, kblocks_w1, 16), dtype=np.uint8
        )
        # Keep E8M0 exponents in a finite, moderate range for stable synthetic tests.
        w1_scales = rng.integers(
            120, 136, size=(e, 2 * i, kblocks_w1), dtype=np.uint8
        )
        w1_bias = rng.standard_normal((e, 2 * i), dtype=np.float32)

        w2_blocks = rng.integers(
            0, 256, size=(e, d, kblocks_w2, 16), dtype=np.uint8
        )
        w2_scales = rng.integers(
            120, 136, size=(e, d, kblocks_w2), dtype=np.uint8
        )
        w2_bias = rng.standard_normal((e, d), dtype=np.float32)

    _mark("build graph")
    with Graph(
        "legacy_moe_graph_smoke",
        input_types=[
            TensorType(DType.float32, shape=[t, d], device=devref),  # x
            TensorType(DType.int32, shape=[p], device=devref),  # router expert ids
            TensorType(DType.float32, shape=[p], device=devref),  # gate weights
            TensorType(DType.uint8, shape=[e, 2 * i, kblocks_w1, 16], device=devref),
            TensorType(DType.uint8, shape=[e, 2 * i, kblocks_w1], device=devref),
            TensorType(DType.float32, shape=[e, 2 * i], device=devref),
            TensorType(DType.uint8, shape=[e, d, kblocks_w2, 16], device=devref),
            TensorType(DType.uint8, shape=[e, d, kblocks_w2], device=devref),
            TensorType(DType.float32, shape=[e, d], device=devref),
        ],
        custom_extensions=[get_mxfp4_kernels_path()],
    ) as graph:
        (
            x_in,
            router_ids_in,
            gate_in,
            w1b_in,
            w1s_in,
            w1bias_in,
            w2b_in,
            w2s_in,
            w2bias_in,
        ) = graph.inputs

        token_expert_order, expert_start_indices, _, expert_ids, expert_usage_stats = (
            moe_create_indices(router_ids_in.tensor, e)
        )
        x_bf16 = ops.cast(x_in.tensor, DType.bfloat16)
        gate_bf16 = ops.cast(gate_in.tensor, DType.bfloat16)

        h_sorted = mxfp4_moe_w1_swiglu(
            x_bf16,
            token_expert_order,
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
            w1b_in.tensor,
            w1s_in.tensor,
            w1bias_in.tensor,
            target="gpu",
        )
        y_pairs = mxfp4_moe_w2_pairs_bf16(
            x_bf16,
            h_sorted,
            token_expert_order,
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
            gate_bf16,
            w2b_in.tensor,
            w2s_in.tensor,
            w2bias_in.tensor,
            target="gpu",
        )
        y_out = mxfp4_moe_topk_reduce_bf16(x_bf16, y_pairs, target="gpu")
        graph.output(
            ops.cast(h_sorted, DType.float32),
            ops.cast(y_pairs, DType.float32),
            ops.cast(y_out, DType.float32),
        )

    _mark("load graph")
    t0 = time.perf_counter()
    model = session.load(graph)
    _mark(f"graph loaded in {(time.perf_counter() - t0):.3f}s")

    def run_one(route_expert_id: int) -> tuple[bool, bool, bool]:
        if args.random_routing:
            router_ids = rng.integers(0, e, size=(p,), dtype=np.int32)
        else:
            router_ids = np.full((p,), np.int32(route_expert_id), dtype=np.int32)
        t0 = time.perf_counter()
        h_sorted_out, y_pairs_out, y_out = model.execute(
            Buffer.from_numpy(x).to(device),
            Buffer.from_numpy(router_ids).to(device),
            Buffer.from_numpy(gate_weights).to(device),
            Buffer.from_numpy(w1_blocks).to(device),
            Buffer.from_numpy(w1_scales).to(device),
            Buffer.from_numpy(w1_bias).to(device),
            Buffer.from_numpy(w2_blocks).to(device),
            Buffer.from_numpy(w2_scales).to(device),
            Buffer.from_numpy(w2_bias).to(device),
        )
        ms = (time.perf_counter() - t0) * 1e3
        h_np = h_sorted_out.to(CPU()).to_numpy()
        y_pairs_np = y_pairs_out.to(CPU()).to_numpy()
        y_np = y_out.to(CPU()).to_numpy()
        h_nan = bool(np.isnan(h_np).any())
        y_pairs_nan = bool(np.isnan(y_pairs_np).any())
        y_nan = bool(np.isnan(y_np).any())
        print(
            f"route_expert_id={route_expert_id} random_routing={args.random_routing} exec_ms={ms:.3f} h_nan={h_nan} y_pairs_nan={y_pairs_nan} y_nan={y_nan} output_max_abs={float(np.nanmax(np.abs(y_np))):.6f}",
            flush=True,
        )
        return h_nan, y_pairs_nan, y_nan

    _mark("execute")
    if args.sweep_experts:
        bad: list[int] = []
        for expert_id in range(e):
            h_nan, y_pairs_nan, y_nan = run_one(expert_id)
            if h_nan or y_pairs_nan or y_nan:
                bad.append(expert_id)
        print(f"sweep_bad_experts={bad}", flush=True)
    else:
        run_one(args.route_expert_id)


if __name__ == "__main__":
    main()
