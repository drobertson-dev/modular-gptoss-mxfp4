"""Eager-style smoke harness for the MXFP4 grouped matmul custom op."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Allow running as a script without requiring PYTHONPATH tweaks.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from max import functional as F
from max.driver import Accelerator, CPU
from max.dtype import DType
from max.tensor import Tensor

from gpt_oss_mxfp4_v3.kernels import (
    MXFP4_VALUES_PER_BLOCK,
    mxfp4_grouped_matmul_ragged_bf16,
)

HOPPER_SCALE_NUM_WARPS = 4
HOPPER_SCALE_ALIGN_M = 32 * HOPPER_SCALE_NUM_WARPS


def _build_expert_segments(
    *, total_rows: int, num_experts: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (expert_start_indices, expert_ids, expert_usage_stats)."""
    if num_experts < 1:
        raise ValueError("num_experts must be >= 1")
    rows_per_expert = (total_rows + num_experts - 1) // num_experts
    starts = [0]
    for _ in range(num_experts):
        starts.append(min(total_rows, starts[-1] + rows_per_expert))
    starts[-1] = total_rows
    expert_start = np.asarray(starts, dtype=np.uint32)
    expert_ids = np.arange(num_experts, dtype=np.int32)
    max_tokens = int((expert_start[1:] - expert_start[:-1]).max())
    expert_usage_stats = np.asarray(
        [max_tokens, num_experts], dtype=np.uint32
    )
    return expert_start, expert_ids, expert_usage_stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MXFP4 grouped-matmul eager smoke test."
    )
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--intermediate", type=int, default=256)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.hidden % MXFP4_VALUES_PER_BLOCK != 0:
        raise ValueError("hidden must be divisible by 32 for MXFP4 packing")
    if args.intermediate % MXFP4_VALUES_PER_BLOCK != 0:
        raise ValueError(
            "intermediate must be divisible by 32 for MXFP4 packing"
        )
    if (args.hidden // MXFP4_VALUES_PER_BLOCK) % 2 != 0:
        raise ValueError("hidden must be divisible by 64 for Hopper scales")
    if (args.intermediate // MXFP4_VALUES_PER_BLOCK) % 2 != 0:
        raise ValueError(
            "intermediate must be divisible by 64 for Hopper scales"
        )

    try:
        device = Accelerator() if args.device == "gpu" else CPU()
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"Device init failed: {exc}") from exc

    rng = np.random.default_rng(args.seed)
    total_rows = args.tokens * args.topk
    k_blocks = args.hidden // MXFP4_VALUES_PER_BLOCK

    a_f32 = rng.standard_normal((total_rows, args.hidden), dtype=np.float32)
    w_blocks = np.zeros(
        (args.experts, k_blocks, args.intermediate, 16), dtype=np.uint8
    )
    scales_m2 = (
        (args.intermediate + HOPPER_SCALE_ALIGN_M - 1)
        // HOPPER_SCALE_ALIGN_M
    ) * HOPPER_SCALE_NUM_WARPS
    w_scales = np.zeros(
        (args.experts, scales_m2, args.hidden), dtype=np.uint8
    )
    expert_start, expert_ids, expert_usage_stats = _build_expert_segments(
        total_rows=total_rows, num_experts=args.experts
    )

    a = Tensor.from_dlpack(a_f32).to(device)
    a_bf16 = F.cast(a, DType.bfloat16)
    w_blocks_t = Tensor.from_dlpack(w_blocks).to(device)
    w_scales_t = Tensor.from_dlpack(w_scales).to(device)
    expert_start_t = Tensor.from_dlpack(expert_start).to(device)
    expert_ids_t = Tensor.from_dlpack(expert_ids).to(device)
    expert_usage_stats_t = Tensor.from_dlpack(expert_usage_stats).to(CPU())

    def _run_once() -> float:
        y = mxfp4_grouped_matmul_ragged_bf16(
            a_bf16,
            w_blocks_t,
            w_scales_t,
            expert_start_t,
            expert_ids_t,
            expert_usage_stats_t,
        )
        return float(y[0, 0].item())

    for _ in range(max(args.warmup, 0)):
        _run_once()

    start = time.perf_counter()
    out0 = None
    for _ in range(max(args.iters, 1)):
        out0 = _run_once()
    elapsed = time.perf_counter() - start

    print(
        "mxfp4 eager smoke",
        f"rows={total_rows}",
        f"hidden={args.hidden}",
        f"intermediate={args.intermediate}",
        f"experts={args.experts}",
        f"device={args.device}",
        f"iters={args.iters}",
        f"elapsed_s={elapsed:.6f}",
        f"out0={out0}",
    )


if __name__ == "__main__":
    main()
