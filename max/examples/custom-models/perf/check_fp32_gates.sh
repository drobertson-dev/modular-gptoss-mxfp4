#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

echo "[mxfp4] FP32-only-in-accumulators grep gates"

echo "== LayoutTensor[F32] usage (must not be SHARED) =="
rg -n "LayoutTensor\\[F32" examples/custom_ops/kernels/ || true
if rg -n "LayoutTensor\\[F32" examples/custom_ops/kernels/ | rg -n "AddressSpace\\.SHARED"; then
  echo "ERROR: Found LayoutTensor[F32] in SHARED address space"
  exit 1
fi

echo "== No F32 in SHARED allocations =="
if rg -n "address_space\\s*=\\s*AddressSpace\\.SHARED[\\s\\S]*F32" examples/custom_ops/kernels/; then
  echo "ERROR: Found F32 in SHARED allocations"
  exit 1
fi

echo "== No MoE OutputTensor[dtype=F32] =="
if rg -n "OutputTensor\\[dtype=F32" examples/custom_ops/kernels/moe_mxfp4_ops.mojo; then
  echo "ERROR: Found OutputTensor[dtype=F32] in MoE ops"
  exit 1
fi

echo "OK"

