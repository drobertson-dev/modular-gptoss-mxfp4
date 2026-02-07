"""Side-by-side diff harness for legacy MXFP4 MoE baseline vs grouped RS path.

This script builds one graph with both branches:
1) Baseline fused custom ops:
   mxfp4_moe_w1_swiglu -> mxfp4_moe_w2_pairs_bf16 -> mxfp4_moe_topk_reduce_bf16
2) Grouped RS path mirroring `layers/moe.py`:
   grouped W1/Gated -> grouped W2 -> restore/scatter -> topk reduce

It runs both on the same routed inputs/weights and reports stage-wise diffs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from safetensors import safe_open

from gpt_oss_mxfp4.kernels import (
    HOPPER_SCALE_NUM_WARPS,
    MXFP4_TOPK,
    MXFP4_VALUES_PER_BLOCK,
    get_mxfp4_kernels_path,
    mxfp4_grouped_matmul_ragged_bf16_swizzled,
    mxfp4_moe_topk_reduce_bf16,
    mxfp4_moe_w1_swiglu,
    mxfp4_moe_w2_pairs_bf16,
)
from gpt_oss_mxfp4.weight_adapters import (
    _mxfp4_swizzle_scales_hopper,
    _mxfp4_swizzle_values_hopper,
)
from max.driver import Buffer, CPU, Accelerator
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn.legacy.kernels import moe_create_indices, scatter_nd_skip_oob_indices


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Legacy MXFP4 MoE grouped-vs-baseline diff harness"
    )
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=2880)
    parser.add_argument("--intermediate", type=int, default=2880)
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--snapshot-dir", type=str, default="")
    parser.add_argument("--route-expert-id", type=int, default=0)
    parser.add_argument("--random-routing", action="store_true")
    parser.add_argument("--print-samples", action="store_true")
    return parser.parse_args()


def _mark(msg: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


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
        with safe_open(str(shard), framework="pt") as st:
            return st.get_tensor(key).float().cpu().numpy()


def _rand_f32(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return rng.standard_normal(shape).astype(np.float32)


def _bf16_like_from_f32(x: np.ndarray) -> np.ndarray:
    # Approximate BF16 rounding through FP16 path for deterministic host refs.
    return x.astype(np.float16).astype(np.float32)


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    ex = np.exp(x, dtype=np.float64)
    sm = ex / ex.sum(axis=-1, keepdims=True)
    return sm.astype(np.float32)


def _prepare_swizzled_weights(
    w_blocks: np.ndarray, w_scales: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # Raw blocks: [E, M, K/32, 16] -> [E, M, Kbytes]
    e, m, kblocks, _ = w_blocks.shape
    kbytes = kblocks * 16
    blocks_nk = np.ascontiguousarray(w_blocks.reshape(e, m, kbytes))
    blocks_swz = _mxfp4_swizzle_values_hopper(blocks_nk, mx_axis=2)
    scales_swz = _mxfp4_swizzle_scales_hopper(np.ascontiguousarray(w_scales))
    return blocks_swz, scales_swz


def _diff_stats(name: str, got: np.ndarray, ref: np.ndarray) -> None:
    d = got.astype(np.float32) - ref.astype(np.float32)
    max_abs = float(np.max(np.abs(d)))
    mean_abs = float(np.mean(np.abs(d)))
    rmse = float(np.sqrt(np.mean(d * d)))
    got_nan = bool(np.isnan(got).any())
    ref_nan = bool(np.isnan(ref).any())
    print(
        f"{name}: max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} rmse={rmse:.6f} got_nan={got_nan} ref_nan={ref_nan}",
        flush=True,
    )


def _build_graph(
    devref: DeviceRef,
    t: int,
    d: int,
    i: int,
    e: int,
    kblocks_w1: int,
    kblocks_w2: int,
    w1_blocks_swz_shape: tuple[int, ...],
    w1_scales_swz_shape: tuple[int, ...],
    w2_blocks_swz_shape: tuple[int, ...],
    w2_scales_swz_shape: tuple[int, ...],
) -> Graph:
    p = t * MXFP4_TOPK
    with Graph(
        "legacy_moe_grouped_diff",
        input_types=[
            TensorType(DType.float32, shape=[t, d], device=devref),  # x
            TensorType(DType.int32, shape=[p], device=devref),  # router ids flat
            TensorType(DType.float32, shape=[p], device=devref),  # gate weights flat
            # Baseline path weights (raw non-swizzled)
            TensorType(
                DType.uint8, shape=[e, 2 * i, kblocks_w1, 16], device=devref
            ),
            TensorType(DType.uint8, shape=[e, 2 * i, kblocks_w1], device=devref),
            TensorType(DType.float32, shape=[e, 2 * i], device=devref),
            TensorType(DType.uint8, shape=[e, d, kblocks_w2, 16], device=devref),
            TensorType(DType.uint8, shape=[e, d, kblocks_w2], device=devref),
            TensorType(DType.float32, shape=[e, d], device=devref),
            # Grouped path weights (swizzled)
            TensorType(DType.uint8, shape=list(w1_blocks_swz_shape), device=devref),
            TensorType(DType.uint8, shape=list(w1_scales_swz_shape), device=devref),
            TensorType(DType.uint8, shape=list(w2_blocks_swz_shape), device=devref),
            TensorType(DType.uint8, shape=list(w2_scales_swz_shape), device=devref),
        ],
        custom_extensions=[get_mxfp4_kernels_path()],
    ) as graph:
        (
            x_in,
            router_ids_in,
            gate_in,
            w1b_raw_in,
            w1s_raw_in,
            w1bias_f32_in,
            w2b_raw_in,
            w2s_raw_in,
            w2bias_f32_in,
            w1b_swz_in,
            w1s_swz_in,
            w2b_swz_in,
            w2s_swz_in,
        ) = graph.inputs

        token_expert_order, expert_start_indices, restore_token_order, expert_ids, expert_usage_stats = (
            moe_create_indices(router_ids_in.tensor, e)
        )
        x_bf16 = ops.cast(x_in.tensor, DType.bfloat16)
        gate_bf16 = ops.cast(gate_in.tensor, DType.bfloat16)

        # Baseline fused path.
        h_sorted_base = mxfp4_moe_w1_swiglu(
            x_bf16,
            token_expert_order,
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
            w1b_raw_in.tensor,
            w1s_raw_in.tensor,
            w1bias_f32_in.tensor,
            alpha=1.702,
            limit=7.0,
            target="gpu",
        )
        y_pairs_base = mxfp4_moe_w2_pairs_bf16(
            x_bf16,
            h_sorted_base,
            token_expert_order,
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
            gate_bf16,
            w2b_raw_in.tensor,
            w2s_raw_in.tensor,
            w2bias_f32_in.tensor,
            target="gpu",
        )
        y_base = mxfp4_moe_topk_reduce_bf16(x_bf16, y_pairs_base, target="gpu")

        # Grouped RS path mirroring layers/moe.py.
        token_expert_order_i32 = ops.cast(token_expert_order, DType.int32)
        token_rows = ops.cast(
            token_expert_order_i32 // MXFP4_TOPK,
            DType.int32,
        )
        permutated_states = ops.gather(x_bf16, token_rows, axis=0)

        gate_up = mxfp4_grouped_matmul_ragged_bf16_swizzled(
            permutated_states,
            w1b_swz_in.tensor,
            w1s_swz_in.tensor,
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
            n_cols=2 * i,
            target="gpu",
            no_small_m=True,
        )
        expert_assignments = ops.gather(router_ids_in.tensor, token_expert_order_i32, axis=0)
        gate_bias = ops.gather(
            ops.cast(w1bias_f32_in.tensor, DType.bfloat16),
            expert_assignments,
            axis=0,
        )
        gate_up = gate_up + gate_bias

        gate = gate_up[:, 0::2]
        up = gate_up[:, 1::2]
        limit_pos = ops.constant(7.0, dtype=gate.dtype, device=gate.device)
        limit_neg = ops.constant(-7.0, dtype=up.dtype, device=up.device)
        alpha = ops.constant(1.702, dtype=gate.dtype, device=gate.device)
        one = ops.constant(1.0, dtype=up.dtype, device=up.device)
        gate = ops.min(gate, limit_pos)
        up = ops.min(ops.max(up, limit_neg), limit_pos)
        glu = gate * ops.sigmoid(gate * alpha)
        h_sorted_group = (up + one) * glu

        down_sorted = mxfp4_grouped_matmul_ragged_bf16_swizzled(
            h_sorted_group,
            w2b_swz_in.tensor,
            w2s_swz_in.tensor,
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
            n_cols=d,
            target="gpu",
            no_small_m=True,
        )
        down_bias = ops.gather(
            ops.cast(w2bias_f32_in.tensor, DType.bfloat16),
            expert_assignments,
            axis=0,
        )
        down_sorted = down_sorted + down_bias

        gate_weights_sorted = ops.gather(gate_bf16, token_expert_order_i32, axis=0)
        down_sorted = down_sorted * ops.unsqueeze(gate_weights_sorted, -1)

        # Baseline y_pairs are expected in original pair order; gather into
        # token_expert_order to compare in the same sorted coordinate system.
        y_pairs_base_sorted = ops.gather(
            y_pairs_base,
            token_expert_order_i32,
            axis=0,
        )

        restore_indices = ops.cast(restore_token_order, DType.int32)
        restore_indices_2d = ops.unsqueeze(restore_indices, -1)
        restored_shape0 = x_bf16.shape[0] * MXFP4_TOPK
        zeros = ops.broadcast_to(
            ops.constant(
                0,
                dtype=down_sorted.dtype,
                device=down_sorted.device,
            ),
            [restored_shape0, down_sorted.shape[1]],
        )
        y_pairs_group = ops.gather(
            down_sorted,
            restore_indices,
            axis=0,
        )
        # Alternate restore variants for debugging index contract mismatches.
        token_order_2d = ops.unsqueeze(token_expert_order_i32, -1)
        y_pairs_group_scatter_token_order = scatter_nd_skip_oob_indices(
            input=zeros,
            updates=down_sorted,
            indices=token_order_2d,
        )
        y_pairs_group_scatter_restore = scatter_nd_skip_oob_indices(
            input=zeros,
            updates=down_sorted,
            indices=restore_indices_2d,
        )
        y_group = mxfp4_moe_topk_reduce_bf16(x_bf16, y_pairs_group, target="gpu")

        graph.output(
            ops.cast(h_sorted_base, DType.float32),
            ops.cast(h_sorted_group, DType.float32),
            ops.cast(y_pairs_base, DType.float32),
            ops.cast(y_pairs_group, DType.float32),
            ops.cast(y_pairs_group_scatter_token_order, DType.float32),
            ops.cast(y_pairs_group_scatter_restore, DType.float32),
            ops.cast(y_pairs_base_sorted, DType.float32),
            ops.cast(down_sorted, DType.float32),
            ops.cast(y_base, DType.float32),
            ops.cast(y_group, DType.float32),
            token_expert_order_i32,
            restore_indices,
        )
    return graph


def main() -> None:
    args = _parse_args()
    if args.hidden % MXFP4_VALUES_PER_BLOCK != 0:
        raise SystemExit("hidden must be divisible by 32")
    if args.intermediate % MXFP4_VALUES_PER_BLOCK != 0:
        raise SystemExit("intermediate must be divisible by 32")
    if args.experts <= 0:
        raise SystemExit("experts must be positive")

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

    x = _rand_f32(rng, (t, d))
    if args.random_routing:
        router_ids = rng.integers(0, e, size=(p,), dtype=np.int32)
    else:
        route_id = max(0, min(e - 1, args.route_expert_id))
        router_ids = np.full((p,), np.int32(route_id), dtype=np.int32)
    gate_logits = _rand_f32(rng, (t, MXFP4_TOPK))
    gate_weights_2d = _softmax_rows(gate_logits)
    gate_weights = gate_weights_2d.reshape(-1).astype(np.float32)
    gate_weights = _bf16_like_from_f32(gate_weights)

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
        w1_bias_f32 = _load_weight_from_snapshot(
            snapshot_dir, f"{prefix}.gate_up_proj_bias"
        ).astype(np.float32, copy=False)
        w2_blocks = _load_weight_from_snapshot(
            snapshot_dir, f"{prefix}.down_proj_blocks"
        ).astype(np.uint8, copy=False)
        w2_scales = _load_weight_from_snapshot(
            snapshot_dir, f"{prefix}.down_proj_scales"
        ).astype(np.uint8, copy=False)
        w2_bias_f32 = _load_weight_from_snapshot(
            snapshot_dir, f"{prefix}.down_proj_bias"
        ).astype(np.float32, copy=False)
    else:
        w1_blocks = rng.integers(
            0, 256, size=(e, 2 * i, kblocks_w1, 16), dtype=np.uint8
        )
        w1_scales = rng.integers(
            120, 136, size=(e, 2 * i, kblocks_w1), dtype=np.uint8
        )
        w1_bias_f32 = _rand_f32(rng, (e, 2 * i))
        w2_blocks = rng.integers(
            0, 256, size=(e, d, kblocks_w2, 16), dtype=np.uint8
        )
        w2_scales = rng.integers(
            120, 136, size=(e, d, kblocks_w2), dtype=np.uint8
        )
        w2_bias_f32 = _rand_f32(rng, (e, d))

    if w1_blocks.shape != (e, 2 * i, kblocks_w1, 16):
        raise ValueError(f"Unexpected W1 blocks shape: {w1_blocks.shape}")
    if w2_blocks.shape != (e, d, kblocks_w2, 16):
        raise ValueError(f"Unexpected W2 blocks shape: {w2_blocks.shape}")

    # Grouped RS path expects Hopper-swizzled weights.
    w1_blocks_swz, w1_scales_swz = _prepare_swizzled_weights(w1_blocks, w1_scales)
    w2_blocks_swz, w2_scales_swz = _prepare_swizzled_weights(w2_blocks, w2_scales)

    if (w1_scales_swz.shape[1] % HOPPER_SCALE_NUM_WARPS) != 0:
        raise ValueError(
            f"W1 swizzled scale M2 must be multiple of {HOPPER_SCALE_NUM_WARPS}, got {w1_scales_swz.shape}"
        )
    if (w2_scales_swz.shape[1] % HOPPER_SCALE_NUM_WARPS) != 0:
        raise ValueError(
            f"W2 swizzled scale M2 must be multiple of {HOPPER_SCALE_NUM_WARPS}, got {w2_scales_swz.shape}"
        )

    _mark("build graph")
    graph = _build_graph(
        devref=devref,
        t=t,
        d=d,
        i=i,
        e=e,
        kblocks_w1=kblocks_w1,
        kblocks_w2=kblocks_w2,
        w1_blocks_swz_shape=w1_blocks_swz.shape,
        w1_scales_swz_shape=w1_scales_swz.shape,
        w2_blocks_swz_shape=w2_blocks_swz.shape,
        w2_scales_swz_shape=w2_scales_swz.shape,
    )

    _mark("load graph")
    t0 = time.perf_counter()
    model = session.load(graph)
    _mark(f"graph loaded in {(time.perf_counter() - t0):.3f}s")

    _mark("execute")
    t0 = time.perf_counter()
    (
        h_sorted_base,
        h_sorted_group,
        y_pairs_base,
        y_pairs_group,
        y_pairs_group_scatter_token_order,
        y_pairs_group_scatter_restore,
        y_pairs_base_sorted,
        down_sorted,
        y_base,
        y_group,
        token_expert_order_out,
        restore_token_order_out,
    ) = model.execute(
        Buffer.from_numpy(x).to(device),
        Buffer.from_numpy(router_ids).to(device),
        Buffer.from_numpy(gate_weights).to(device),
        Buffer.from_numpy(w1_blocks).to(device),
        Buffer.from_numpy(w1_scales).to(device),
        Buffer.from_numpy(w1_bias_f32).to(device),
        Buffer.from_numpy(w2_blocks).to(device),
        Buffer.from_numpy(w2_scales).to(device),
        Buffer.from_numpy(w2_bias_f32).to(device),
        Buffer.from_numpy(w1_blocks_swz).to(device),
        Buffer.from_numpy(w1_scales_swz).to(device),
        Buffer.from_numpy(w2_blocks_swz).to(device),
        Buffer.from_numpy(w2_scales_swz).to(device),
    )
    exec_ms = (time.perf_counter() - t0) * 1e3

    h_sorted_base_np = h_sorted_base.to(CPU()).to_numpy()
    h_sorted_group_np = h_sorted_group.to(CPU()).to_numpy()
    y_pairs_base_np = y_pairs_base.to(CPU()).to_numpy()
    y_pairs_group_np = y_pairs_group.to(CPU()).to_numpy()
    y_pairs_group_scatter_token_order_np = (
        y_pairs_group_scatter_token_order.to(CPU()).to_numpy()
    )
    y_pairs_group_scatter_restore_np = (
        y_pairs_group_scatter_restore.to(CPU()).to_numpy()
    )
    y_pairs_base_sorted_np = y_pairs_base_sorted.to(CPU()).to_numpy()
    down_sorted_np = down_sorted.to(CPU()).to_numpy()
    y_base_np = y_base.to(CPU()).to_numpy()
    y_group_np = y_group.to(CPU()).to_numpy()
    token_expert_order_np = token_expert_order_out.to(CPU()).to_numpy()
    restore_token_order_np = restore_token_order_out.to(CPU()).to_numpy()

    print(
        f"exec_ms={exec_ms:.3f} tokens={t} hidden={d} intermediate={i} experts={e} checkpoint={args.use_checkpoint}",
        flush=True,
    )
    _diff_stats("h_sorted_group_vs_base", h_sorted_group_np, h_sorted_base_np)
    _diff_stats("down_sorted_vs_base_sorted_pairs", down_sorted_np, y_pairs_base_sorted_np)
    _diff_stats("y_pairs_group_vs_base", y_pairs_group_np, y_pairs_base_np)
    _diff_stats(
        "y_pairs_group_scatter_token_order_vs_base",
        y_pairs_group_scatter_token_order_np,
        y_pairs_base_np,
    )
    _diff_stats(
        "y_pairs_group_scatter_restore_vs_base",
        y_pairs_group_scatter_restore_np,
        y_pairs_base_np,
    )
    _diff_stats("y_group_vs_base", y_group_np, y_base_np)
    print(
        f"y_base_nan={bool(np.isnan(y_base_np).any())} y_group_nan={bool(np.isnan(y_group_np).any())}",
        flush=True,
    )
    # Permutation sanity: for a perfect inverse mapping this should be all rows.
    inv_hits = int(
        np.sum(
            np.arange(token_expert_order_np.shape[0], dtype=np.int32)
            == restore_token_order_np[token_expert_order_np]
        )
    )
    print(
        f"perm_inverse_hits={inv_hits}/{token_expert_order_np.shape[0]}",
        flush=True,
    )

    if args.print_samples:
        print("sample_y_base[0, :8] =", y_base_np[0, :8], flush=True)
        print("sample_y_group[0, :8] =", y_group_np[0, :8], flush=True)


if __name__ == "__main__":
    main()
