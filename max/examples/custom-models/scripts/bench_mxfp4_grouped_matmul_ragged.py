"""Microbench for MXFP4 vs BF16 grouped_matmul_ragged (ModuleV3 MoE expert GEMMs).

Runs a single-expert, contiguous-segment grouped matmul to isolate kernel cost.
"""

from __future__ import annotations

import argparse
import os

# Allow running as a script without requiring PYTHONPATH tweaks.
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpt_oss_mxfp4.kernels import (
    MXFP4_VALUES_PER_BLOCK,
    get_mxfp4_kernels_path,
    mxfp4_grouped_matmul_ragged_bf16_swizzled,
)
from gpt_oss_mxfp4.weight_adapters import (
    _mxfp4_pack_bits_u8,
    _mxfp4_swizzle_scales_hopper,
    _mxfp4_swizzle_values_hopper,
)
from max.driver import Buffer, CPU, Accelerator
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
try:
    from max.nn.kernels import grouped_matmul_ragged
except Exception:  # pragma: no cover
    try:
        from max.nn.legacy.kernels import grouped_matmul_ragged
    except Exception:  # pragma: no cover
        grouped_matmul_ragged = None
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

    blocks: [K/32, N, 16] uint8
    scales_exp: [K/32, N] uint8 (E8M0 exponent bytes)
    Returns: [N, K] float32
    """
    k_blocks, n_rows, bytes_per_block = blocks.shape
    if bytes_per_block != 16:
        raise ValueError(f"Expected 16 bytes per block, got {bytes_per_block}")
    if scales_exp.shape != (k_blocks, n_rows):
        raise ValueError(
            f"blocks/scales mismatch: {blocks.shape[:2]} vs {scales_exp.shape}"
        )

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


def _bench(
    model,
    device: Accelerator,
    inputs: list[Buffer],
    *,
    warmup: int,
    iters: int,
) -> float:
    for _ in range(warmup):
        _ = model.execute(*inputs)
    device.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model.execute(*inputs)
    device.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=int, default=257)
    ap.add_argument("--K", type=int, default=2880)
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument(
        "--which",
        choices=["gate_up", "down"],
        default="gate_up",
        help="Select which checkpoint MXFP4 weight tensor to use.",
    )
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    ap.add_argument(
        "--value-swizzle",
        action="store_true",
        default=True,
        help="Kept for compatibility; active path always uses swizzled values.",
    )
    ap.add_argument(
        "--skip-bf16",
        action="store_true",
        help="Skip building/running the BF16 baseline graph.",
    )
    args = ap.parse_args()

    ckpt = _find_gpt_oss_20b_file()
    if ckpt is None:
        raise SystemExit(
            "GPT-OSS checkpoint not found in HF cache; run `pixi run generate-v3-smoke` once"
        )

    w_blocks_key = (
        "model.layers.0.mlp.experts.gate_up_proj_blocks"
        if args.which == "gate_up"
        else "model.layers.0.mlp.experts.down_proj_blocks"
    )
    w_scales_key = (
        "model.layers.0.mlp.experts.gate_up_proj_scales"
        if args.which == "gate_up"
        else "model.layers.0.mlp.experts.down_proj_scales"
    )

    k_blocks = args.K // MXFP4_VALUES_PER_BLOCK
    if args.K != k_blocks * MXFP4_VALUES_PER_BLOCK:
        raise SystemExit("K must be divisible by 32 for MXFP4 weights.")
    if (k_blocks % 2) != 0:
        raise SystemExit("K must be divisible by 64 for Hopper scales.")

    with safe_open(str(ckpt), framework="numpy") as f:
        w_blocks_all = f.get_tensor(w_blocks_key)
        w_scales_all = f.get_tensor(w_scales_key)

    # Prepack for our MXFP4 kernel.
    w_blocks_raw = np.ascontiguousarray(
        np.transpose(w_blocks_all[:1, : args.N, :k_blocks, :], (0, 2, 1, 3))
    )
    if args.value_swizzle:
        w_blocks_raw_nk = np.ascontiguousarray(
            w_blocks_all[:1, : args.N, :k_blocks, :]
        )
        w_blocks_nk = w_blocks_raw_nk.reshape(1, args.N, k_blocks * 16)
        w_blocks = _mxfp4_swizzle_values_hopper(w_blocks_nk, mx_axis=2)
    else:
        # blocks -> [E, K/32, N, 16] packbits
        w_blocks = _mxfp4_pack_bits_u8(w_blocks_raw)
    w_scales_logical = np.ascontiguousarray(
        w_scales_all[:1, : args.N, :k_blocks]
    )
    w_scales = _mxfp4_swizzle_scales_hopper(w_scales_logical)
    w_scales_ref = np.ascontiguousarray(
        np.transpose(w_scales_logical, (0, 2, 1))
    )

    rng = np.random.default_rng(args.seed)
    a_f32 = rng.uniform(-1.0, 1.0, size=(args.P, args.K)).astype(np.float32)

    expert_start = np.array([0, args.P], dtype=np.uint32)
    expert_ids = np.array([0], dtype=np.int32)
    expert_usage_stats = np.array([args.P, 1], dtype=np.uint32)

    w_dense = None
    if not args.skip_bf16:
        # Dense BF16 baseline weights.
        w_dense = _decode_mxfp4_rows(w_blocks_raw[0], w_scales_ref[0])
        w_dense = _bf16_round_to_f32(w_dense).reshape(1, args.N, args.K)

    device = Accelerator()
    devref = DeviceRef.from_device(device)
    session = InferenceSession(devices=[device])

    # MXFP4 op graph.
    with Graph(
        "bench_mxfp4_grouped_matmul",
        input_types=[
            TensorType(DType.float32, shape=[args.P, args.K], device=devref),
            TensorType(DType.uint8, shape=w_blocks.shape, device=devref),
            TensorType(DType.uint8, shape=w_scales.shape, device=devref),
            TensorType(DType.uint32, shape=[2], device=devref),
            TensorType(DType.int32, shape=[1], device=devref),
            TensorType(DType.uint32, shape=[2], device=DeviceRef.CPU()),
        ],
        custom_extensions=[get_mxfp4_kernels_path()],
    ) as graph_mxfp4:
        a_in, blocks_in, scales_in, start_in, ids_in, stats_in = (
            graph_mxfp4.inputs
        )
        a_bf16 = ops.cast(a_in.tensor, DType.bfloat16)
        out = mxfp4_grouped_matmul_ragged_bf16_swizzled(
            a_bf16,
            blocks_in.tensor,
            scales_in.tensor,
            start_in.tensor,
            ids_in.tensor,
            stats_in.tensor,
            n_cols=args.N,
            target="gpu",
        )
        graph_mxfp4.output(ops.cast(out, DType.float32))

    graph_bf16 = None
    if (not args.skip_bf16) and grouped_matmul_ragged is not None:
        # BF16 baseline graph using upstream grouped_matmul_ragged.
        with Graph(
            "bench_bf16_grouped_matmul",
            input_types=[
                TensorType(DType.float32, shape=[args.P, args.K], device=devref),
                TensorType(DType.float32, shape=w_dense.shape, device=devref),
                TensorType(DType.uint32, shape=[2], device=devref),
                TensorType(DType.int32, shape=[1], device=devref),
                TensorType(DType.uint32, shape=[2], device=DeviceRef.CPU()),
            ],
        ) as graph_bf16_ctx:
            a_in, w_in, start_in, ids_in, stats_in = graph_bf16_ctx.inputs
            a_bf16 = ops.cast(a_in.tensor, DType.bfloat16)
            w_bf16 = ops.cast(w_in.tensor, DType.bfloat16)
            out = grouped_matmul_ragged(
                a_bf16,
                w_bf16,
                start_in.tensor,
                ids_in.tensor,
                stats_in.tensor,
            )
            graph_bf16_ctx.output(ops.cast(out, DType.float32))
        graph_bf16 = graph_bf16_ctx

    mxfp4_model = session.load(graph_mxfp4)
    bf16_model = session.load(graph_bf16) if graph_bf16 is not None else None

    a_dev = Buffer.from_numpy(a_f32).to(device)
    blocks_dev = Buffer.from_numpy(w_blocks).to(device)
    scales_dev = Buffer.from_numpy(w_scales).to(device)
    start_dev = Buffer.from_numpy(expert_start).to(device)
    ids_dev = Buffer.from_numpy(expert_ids).to(device)
    stats_cpu = Buffer.from_numpy(expert_usage_stats).to(CPU())
    w_dense_dev = Buffer.from_numpy(w_dense).to(device) if w_dense is not None else None

    t_mxfp4 = _bench(
        mxfp4_model,
        device,
        [a_dev, blocks_dev, scales_dev, start_dev, ids_dev, stats_cpu],
        warmup=args.warmup,
        iters=args.iters,
    )
    t_bf16 = None
    if bf16_model is not None and w_dense_dev is not None:
        t_bf16 = _bench(
            bf16_model,
            device,
            [a_dev, w_dense_dev, start_dev, ids_dev, stats_cpu],
            warmup=args.warmup,
            iters=args.iters,
        )

    print(f"P={args.P} K={args.K} N={args.N} which={args.which}")
    print(f"MXFP4 grouped matmul: {t_mxfp4 * 1e3:.3f} ms/iter")
    if t_bf16 is not None:
        print(f"BF16  grouped matmul: {t_bf16 * 1e3:.3f} ms/iter")
        print(f"Speedup (BF16/MXFP4): {t_bf16 / t_mxfp4:.2f}x")
    else:
        print("BF16  grouped matmul: skipped (max.nn.kernels.grouped_matmul_ragged not available)")

    if args.check:
        out_mxfp4 = (
            mxfp4_model.execute(
                a_dev, blocks_dev, scales_dev, start_dev, ids_dev, stats_cpu
            )[0]
            .to(CPU())
            .to_numpy()
        )
        if bf16_model is not None and w_dense_dev is not None:
            out_bf16 = (
                bf16_model.execute(
                    a_dev, w_dense_dev, start_dev, ids_dev, stats_cpu
                )[0]
                .to(CPU())
                .to_numpy()
            )
            diff = np.max(
                np.abs(
                    out_mxfp4.astype(np.float32) - out_bf16.astype(np.float32)
                )
            )
            print(f"max|MXFP4-BF16| = {diff}")
        else:
            print("max|MXFP4-BF16| = skipped (no BF16 baseline)")


if __name__ == "__main__":
    main()
