"""Eager smoke test for MXFP4 OGS MoE path (Triton matmul_ogs)."""

from __future__ import annotations

import argparse
import time

import numpy as np

from max import functional as F
from max.driver import Accelerator
from max.dtype import DType
from max.tensor import Tensor

from gpt_oss_mxfp4_v3.ogs_backend import ogs_moe_forward


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MXFP4 OGS MoE eager smoke test."
    )
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--intermediate", type=int, default=256)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _rand_u8(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def main() -> None:
    args = _parse_args()
    if args.hidden % 32 != 0 or args.intermediate % 32 != 0:
        raise SystemExit("hidden/intermediate must be divisible by 32")
    if args.hidden % 64 != 0 or args.intermediate % 64 != 0:
        raise SystemExit("hidden/intermediate must be divisible by 64")

    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"torch is required: {exc}") from exc

    rng = np.random.default_rng(args.seed)
    device = Accelerator()

    tokens = args.tokens
    hidden = args.hidden
    inter = args.intermediate
    experts = args.experts
    topk = args.topk

    x = rng.standard_normal((tokens, hidden), dtype=np.float32)
    gate_w = rng.standard_normal((experts, hidden), dtype=np.float32)
    gate_b = rng.standard_normal((experts,), dtype=np.float32)

    kbytes_w1 = hidden // 2
    kblocks_w1 = hidden // 32
    w1_blocks = _rand_u8((experts, 2 * inter, kbytes_w1), rng)
    w1_scales = _rand_u8((experts, 2 * inter, kblocks_w1), rng)
    w1_bias = rng.standard_normal((experts, 2 * inter), dtype=np.float32)

    kbytes_w2 = inter // 2
    kblocks_w2 = inter // 32
    w2_blocks = _rand_u8((experts, hidden, kbytes_w2), rng)
    w2_scales = _rand_u8((experts, hidden, kblocks_w2), rng)
    w2_bias = rng.standard_normal((experts, hidden), dtype=np.float32)

    x_t = Tensor.from_dlpack(x).to(device)
    x_bf16 = F.cast(x_t, DType.bfloat16)
    gate_w_t = Tensor.from_dlpack(gate_w).to(device)
    gate_b_t = Tensor.from_dlpack(gate_b).to(device)
    w1_blocks_t = Tensor.from_dlpack(w1_blocks).to(device)
    w1_scales_t = Tensor.from_dlpack(w1_scales).to(device)
    w1_bias_t = Tensor.from_dlpack(w1_bias).to(device)
    w2_blocks_t = Tensor.from_dlpack(w2_blocks).to(device)
    w2_scales_t = Tensor.from_dlpack(w2_scales).to(device)
    w2_bias_t = Tensor.from_dlpack(w2_bias).to(device)

    cache: dict[str, object] = {}

    def _run_once() -> Tensor:
        return ogs_moe_forward(
            x=x_bf16,
            gate_weight=gate_w_t,
            gate_bias=gate_b_t,
            w1_blocks_raw=w1_blocks_t,
            w1_scales_raw=w1_scales_t,
            w1_bias=w1_bias_t,
            w2_blocks_raw=w2_blocks_t,
            w2_scales_raw=w2_scales_t,
            w2_bias=w2_bias_t,
            topk=topk,
            swiglu_alpha=1.702,
            swiglu_limit=7.0,
            num_experts=experts,
            num_warps=args.num_warps,
            _cache=cache,
        )

    for _ in range(max(args.warmup, 0)):
        _ = _run_once()
        torch.cuda.synchronize()

    start = time.perf_counter()
    out = None
    for _ in range(max(args.iters, 1)):
        out = _run_once()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(
        "mxfp4_ogs_smoke",
        f"tokens={tokens}",
        f"hidden={hidden}",
        f"intermediate={inter}",
        f"experts={experts}",
        f"topk={topk}",
        f"iters={max(args.iters,1)}",
        f"elapsed_s={elapsed:.6f}",
        f"out0={(float(out[0,0].item()) if out is not None else 0.0):.4f}",
    )


if __name__ == "__main__":
    main()
