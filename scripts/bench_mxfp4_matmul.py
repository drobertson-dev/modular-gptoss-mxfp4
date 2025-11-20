#!/usr/bin/env python3
"""
Microbenchmark for MXFP4 grouped matmul using MAX.

This builds a small graph around `grouped_mxfp4_matmul` with synthetic data
matching GPT-OSS MoE shapes (interleaved gate/up layout). It can run against
the built-in kernel or a custom mojopkg and optionally emit MAX profiling spans
when MODULAR_ENABLE_PROFILING=1.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Iterable

import numpy as np

from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.graph import ops
from max.pipelines.lib.custom_extensions import (
    collect_custom_extensions_from_env,
)
from max.profiler import Tracer, is_profiling_enabled
from max.driver import Accelerator, CPU, Tensor

_FP4_VALUES = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def _quantize_row(row: np.ndarray, scale_exp: int) -> tuple[np.ndarray, np.uint8]:
    scale = np.float32(2.0**scale_exp)
    scaled = row / scale
    idxs = np.abs(_FP4_VALUES[None, :] - scaled[:, None]).argmin(axis=1).astype(np.uint8)
    bytes_view = idxs.reshape(-1, 2)
    packed = (bytes_view[:, 0] & 0x0F) | ((bytes_view[:, 1] & 0x0F) << 4)
    return packed.astype(np.uint8), np.uint8(scale_exp + 127)


def quantize_mxfp4(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize dense weights to MXFP4 blocks/scales."""
    experts, out_features, in_features = weights.shape
    if in_features % 32 != 0:
        raise ValueError(f"in_features must be divisible by 32, got {in_features}")

    blocks = np.zeros((experts, out_features, in_features // 2), dtype=np.uint8)
    scales = np.zeros((experts, out_features, in_features // 32), dtype=np.uint8)

    for e in range(experts):
        for row in range(out_features):
            for blk in range(in_features // 32):
                chunk = weights[e, row, blk * 32 : (blk + 1) * 32]
                max_abs = np.max(np.abs(chunk))
                if max_abs == 0:
                    scale_exp = 0
                else:
                    scale_exp = int(np.floor(np.log2(max_abs / 6.0)))
                    scale_exp = min(max(scale_exp, -127), 127)
                packed, scale_byte = _quantize_row(chunk, scale_exp)
                blocks[e, row, blk * 16 : (blk + 1) * 16] = packed
                scales[e, row, blk] = scale_byte
    return blocks, scales


def build_expert_offsets(tokens_per_expert: Iterable[int]) -> np.ndarray:
    offsets = np.zeros(len(tokens_per_expert) + 1, dtype=np.uint32)
    cursor = 0
    for idx, count in enumerate(tokens_per_expert, start=1):
        cursor += count
        offsets[idx] = cursor
    return offsets


def run_bench(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    tokens_per_expert = [args.tokens_per_expert] * args.num_experts
    expert_offsets = build_expert_offsets(tokens_per_expert)
    max_tokens_per_expert = max(tokens_per_expert)
    total_tokens = expert_offsets[-1]

    # Shapes mirror GPT-OSS gate/up matmul by default.
    in_features = args.in_features
    out_features = args.out_features

    hidden = rng.standard_normal((total_tokens, in_features), dtype=np.float32).astype(np.float32)
    dense_weights = rng.standard_normal((args.num_experts, out_features, in_features), dtype=np.float32)
    blocks, scales = quantize_mxfp4(dense_weights)
    bias = np.zeros((args.num_experts, out_features), dtype=np.float32)

    expert_ids = np.arange(args.num_experts, dtype=np.int32)
    max_tokens = np.uint32(max_tokens_per_expert)
    num_experts = np.uint32(args.num_experts)

    ext_env = (
        args.custom_extension
        or os.environ.get("MAX_CUSTOM_EXTENSIONS")
        or os.environ.get("MXFP4_KERNEL_PACKAGE")
    )
    extensions = collect_custom_extensions_from_env(
        ext_env,
        include_runtime_dependencies=True,
    )
    if not extensions:
        raise SystemExit(
            "No MXFP4 custom extensions found. Set --custom-extension or MAX_CUSTOM_EXTENSIONS/MXFP4_KERNEL_PACKAGE"
        )

    device_ref = DeviceRef.GPU() if args.device == "gpu" else DeviceRef.CPU()

    graph = Graph(
        "mxfp4_grouped_matmul_bench",
        input_types=[
            TensorType(DType.float32, hidden.shape, device=device_ref),
            TensorType(DType.uint8, blocks.shape, device=device_ref),
            TensorType(DType.uint8, scales.shape, device=device_ref),
            TensorType(DType.float32, bias.shape, device=device_ref),
            TensorType(DType.uint32, expert_offsets.shape, device=device_ref),
            TensorType(DType.int32, expert_ids.shape, device=device_ref),
            TensorType(DType.uint32, (), device=DeviceRef.CPU()),
            TensorType(DType.uint32, (), device=DeviceRef.CPU()),
        ],
        output_types=[
            TensorType(DType.float32, (total_tokens, out_features), device=device_ref),
        ],
        custom_extensions=extensions,
    )

    with graph:
        preferred_name = os.environ.get("MAX_MXFP4_KERNEL_OP", "mo.moe.mx4.matmul")
        candidate_names = []
        for name in (preferred_name, "custom.moe.mx4.matmul"):
            if name not in candidate_names:
                candidate_names.append(name)

        last_exc: Exception | None = None
        result = None
        for kernel_name in candidate_names:
            try:
                result = ops.custom(
                    kernel_name,
                    device=device_ref,
                    values=[
                        graph.inputs[0].tensor,
                        graph.inputs[1].tensor,
                        graph.inputs[2].tensor,
                        graph.inputs[3].tensor,
                        graph.inputs[4].tensor,
                        graph.inputs[5].tensor,
                        graph.inputs[6].tensor,
                        graph.inputs[7].tensor,
                    ],
                    out_types=[
                        TensorType(
                            dtype=DType.float32,
                            shape=[total_tokens, out_features],
                            device=device_ref,
                        )
                    ],
                )[0].tensor
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                continue
        if last_exc is not None or result is None:
            raise last_exc if last_exc else RuntimeError("No MXFP4 kernel succeeded")
        graph.output(result)

    device = Accelerator(0) if args.device == "gpu" else CPU(0)
    session = InferenceSession(devices=[device])
    compiled = session.load(graph, custom_extensions=extensions)

    hidden_t = Tensor.from_numpy(hidden).to(device)
    blocks_t = Tensor.from_numpy(blocks).to(device)
    scales_t = Tensor.from_numpy(scales).to(device)
    bias_t = Tensor.from_numpy(bias).to(device)
    offsets_t = Tensor.from_numpy(expert_offsets).to(device)
    expert_ids_t = Tensor.from_numpy(expert_ids).to(device)
    max_tokens_t = Tensor.scalar(max_tokens, DType.uint32, device=CPU())
    num_experts_t = Tensor.scalar(num_experts, DType.uint32, device=CPU())

    # Warmup
    for _ in range(args.warmup):
        compiled.execute(
            hidden_t,
            blocks_t,
            scales_t,
            bias_t,
            offsets_t,
            expert_ids_t,
            max_tokens_t,
            num_experts_t,
        )

    token_count = total_tokens * args.iters
    start = time.perf_counter()
    for _ in range(args.iters):
        if is_profiling_enabled():
            with Tracer("mxfp4_grouped_matmul_bench", color="green"):
                compiled.execute(
                    hidden_t,
                    blocks_t,
                    scales_t,
                    bias_t,
                    offsets_t,
                    expert_ids_t,
                    max_tokens_t,
                    num_experts_t,
                )
        else:
            compiled.execute(
                hidden_t,
                blocks_t,
                scales_t,
                bias_t,
                offsets_t,
                expert_ids_t,
                max_tokens_t,
                num_experts_t,
            )
    elapsed = time.perf_counter() - start

    tokens_per_sec = token_count / elapsed if elapsed > 0 else 0.0
    print(
        f"iters={args.iters}, tokens/iter={total_tokens}, elapsed={elapsed:.4f}s, "
        f"tokens/s={tokens_per_sec:.2f}"
    )
    print(
        f"shape: experts={args.num_experts}, tokens_per_expert={args.tokens_per_expert}, "
        f"in_features={in_features}, out_features={out_features}"
    )
    if is_profiling_enabled():
        print("Profiling enabled (MODULAR_ENABLE_PROFILING=1); inspect profiler output for spans.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-experts", type=int, default=32, help="Number of experts")
    p.add_argument("--tokens-per-expert", type=int, default=4, help="Tokens per expert (uniform)")
    p.add_argument("--in-features", type=int, default=2880, help="Input dim (hidden)")
    p.add_argument("--out-features", type=int, default=5760, help="Output dim (e.g., 2 * moe_dim for gate/up)")
    p.add_argument("--iters", type=int, default=10, help="Measured iterations")
    p.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    p.add_argument("--device", choices=["gpu", "cpu"], default="gpu", help="Device to run on")
    p.add_argument(
        "--custom-extension",
        type=str,
        default=None,
        help="Path to MXFP4 mojopkg (falls back to MAX_CUSTOM_EXTENSIONS/MXFP4_KERNEL_PACKAGE)",
    )
    return p.parse_args()


if __name__ == "__main__":
    run_bench(parse_args())
