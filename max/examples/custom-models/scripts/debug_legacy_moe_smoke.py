"""Debug smoke test for legacy MXFP4 MoE outside serving pipeline.

This script isolates the custom MoE path and prints milestone timestamps so we
can identify where execution stalls.
"""

from __future__ import annotations

import argparse
import time
from types import SimpleNamespace

import numpy as np

from max import functional as F
from max.driver import Accelerator
from max.dtype import DType
from max.tensor import Tensor

from gpt_oss_mxfp4.layers.moe import GptOssMoE


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Legacy MXFP4 MoE smoke test.")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=2880)
    parser.add_argument("--intermediate", type=int, default=2880)
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _mark(msg: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def main() -> None:
    args = _parse_args()
    if args.hidden % 32 != 0 or args.intermediate % 32 != 0:
        raise SystemExit("hidden and intermediate must be divisible by 32")

    _mark("start")
    rng = np.random.default_rng(args.seed)

    _mark("create accelerator")
    accel = Accelerator()

    _mark("build config")
    config = SimpleNamespace(
        devices=[accel],
        hidden_size=args.hidden,
        num_local_experts=args.experts,
        num_experts_per_tok=args.topk,
        intermediate_size=args.intermediate,
        swiglu_limit=7.0,
        dtype=DType.bfloat16,
    )

    _mark("construct GptOssMoE")
    moe = GptOssMoE(config)

    _mark("create input tensor")
    x = rng.standard_normal((args.tokens, args.hidden), dtype=np.float32)
    x_t = Tensor.from_dlpack(x).to(accel)
    x_bf16 = F.cast(x_t, DType.bfloat16)

    _mark("invoke moe")
    y = moe(x_bf16)
    _mark("moe returned")

    _mark("synchronize")
    accel.synchronize()
    _mark("done")
    print(f"output shape={y.shape} dtype={y.dtype}", flush=True)


if __name__ == "__main__":
    main()
