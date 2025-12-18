#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[check] No FP32 in shared tiles (MXFP4 kernels)"

# Rule: LayoutTensor[F32, ..., address_space = AddressSpace.SHARED] is forbidden.
# Match within a single `LayoutTensor[ ... ]` generic instantiation (no nested `]`).
if rg -n -U "LayoutTensor\\[[^\\]]*\\bF32\\b[^\\]]*address_space\\s*=\\s*AddressSpace\\.SHARED[^\\]]*\\]" ../custom_ops/kernels/; then
  echo >&2
  echo "ERROR: Found F32 LayoutTensor in SHARED address space." >&2
  exit 1
fi

# Rule: No FP32 shared allocations for per-row scalars/tiles.
if rg -n -U "stack_allocation\\[[^\\]]*Scalar\\[F32\\][^\\]]*address_space\\s*=\\s*AddressSpace\\.SHARED" ../custom_ops/kernels/; then
  echo >&2
  echo "ERROR: Found Scalar[F32] stack_allocation in SHARED address space." >&2
  exit 1
fi

# Rule: No MoE ops should expose F32 output tensors (FP32 only for accumulators/epilogue regs).
if rg -n "OutputTensor\\[dtype=F32" ../custom_ops/kernels/moe_mxfp4_ops.mojo; then
  echo >&2
  echo "ERROR: Found MoE op with F32 OutputTensor." >&2
  exit 1
fi

echo "[ok] MXFP4 FP32 gates passed"
