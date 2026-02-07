"""End-to-end MXFP4 MoE bench (routing + W1/W2 + reduce)."""

from __future__ import annotations

import argparse
import time
from types import SimpleNamespace

import numpy as np

from max import functional as F
from max.driver import Accelerator
from max.dtype import DType
from max.tensor import Tensor

from gpt_oss_mxfp4_v3.layers.moe import GptOssMoE


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end MXFP4 MoE bench (routing + kernels)."
    )
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=2880)
    parser.add_argument("--intermediate", type=int, default=2880)
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.hidden % 32 != 0 or args.intermediate % 32 != 0:
        raise SystemExit("hidden/intermediate must be divisible by 32")
    if args.hidden % 64 != 0 or args.intermediate % 64 != 0:
        raise SystemExit("hidden/intermediate must be divisible by 64")

    device = Accelerator()
    rng = np.random.default_rng(args.seed)

    config = SimpleNamespace(
        hidden_size=args.hidden,
        num_local_experts=args.experts,
        num_experts_per_tok=args.topk,
        intermediate_size=args.intermediate,
        swiglu_limit=7.0,
    )

    moe = GptOssMoE(config, layer_idx=0)
    moe.to(device)

    x = rng.standard_normal((args.tokens, args.hidden), dtype=np.float32)
    x_t = Tensor.from_dlpack(x).to(device)
    x_bf16 = F.cast(x_t, DType.bfloat16)

    def _run_once() -> None:
        _ = moe(x_bf16)
        device.synchronize()

    for _ in range(max(args.warmup, 0)):
        _run_once()

    start = time.perf_counter()
    for _ in range(max(args.iters, 1)):
        _run_once()
    elapsed = time.perf_counter() - start

    iters = max(args.iters, 1)
    avg_ms = (elapsed / iters) * 1e3
    print(
        "mxfp4_e2e_moe_bench",
        f"tokens={args.tokens}",
        f"hidden={args.hidden}",
        f"intermediate={args.intermediate}",
        f"experts={args.experts}",
        f"topk={args.topk}",
        f"iters={iters}",
        f"avg_ms={avg_ms:.3f}",
    )


if __name__ == "__main__":
    main()
