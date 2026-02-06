# First-Principles Reset (SM90 MXFP4 MoE)

Date: 2026-02-06
Branch: `kernels/gptoss-mxfp4-swizzle`

## Verified Active Path

Legacy architecture path (`gpt_oss_mxfp4`) with grouped RS enabled:

1. `GptOssMoE.__call__` in `max/examples/custom-models/gpt_oss_mxfp4/layers/moe.py`
2. `mxfp4_grouped_matmul_ragged_bf16_swizzled(...)` in
   `max/examples/custom-models/gpt_oss_mxfp4/kernels.py`
3. Mojo op registration:
   `mxfp4_grouped_matmul_ragged_bf16_swizzled` in
   `max/examples/custom_ops/mxfp4/grouped_matmul_sm90_entrypoints.mojo`
4. Kernel:
   `grouped_matmul_mxfp4_bf16_wgmma_sm90_pipeline_swload_transpose` in
   `max/examples/custom_ops/mxfp4/grouped_matmul_sm90_wgmma_swload_transpose.mojo`

This is currently the hot path for RS decode + WGMMA when grouped path is enabled.

## What Is Confirmed Noise

- The swizzled entrypoint small-M branch was effectively dead for performance
  tuning (same transpose RS kernel shape path used either way).
- Legacy non-swizzled grouped op exists for compatibility/reference, but it is
  not the target path for Hopper MXFP4 work.

## Guardrails Added

1. Clamp `expert_ids` in both MoE callsites before launching grouped kernels:
   - `max/examples/custom-models/gpt_oss_mxfp4/layers/moe.py`
   - `max/examples/custom-models/gpt_oss_mxfp4_v3/layers/moe.py`

2. Force single-variant behavior in swizzled entrypoint by setting:
   - `SMALL_M_TRANSPOSE_THRESHOLD = 0` in
     `max/examples/custom_ops/mxfp4/grouped_matmul_sm90_entrypoints.mojo`

This reduces branch ambiguity and removes one class of invalid expert-id routing
from immediately poisoning expert weight reads.

## Weight-Loading Contract (Important)

- `gpt_oss_mxfp4_v3/weight_adapters.py` already performs Hopper value swizzle
  + packbits + Hopper scale swizzle.
- `gpt_oss_mxfp4/weight_adapters.py` can reuse that v3 adapter when
  `MXFP4_LEGACY_GROUPED_RS=1`.
- Adapter enforces converted `model.*` checkpoint layout and rejects original
  `block.*`-only layout for this architecture.

## Next Measurement Ladder

1. Correctness first:
   - Run grouped swizzled reference test and keep tracking max-abs diff.
2. Physical behavior:
   - NCU/SASS diff for active swizzled RS kernel only.
   - Focus: global load width (`LDG.64/128`), integer op pressure, tensor pipe
     activity.
3. If correctness still fails:
   - isolate mismatch between swizzled adapter output and kernel decode mapping
     (`_mxfp4_swizzle_values_hopper` + `hopper_value_swizzle_index_fast` +
     decode fragment ordering).
4. Only after correctness lock:
   - continue integer/decode tightening and throughput tuning.

## 2026-02-06 Addendum (Active Debug Results)

### Dispatch Reality (Critical)

- `mxfp4_grouped_matmul_ragged_bf16_swizzled` currently routes to the
  transpose RS kernel path by default:
  - `SMALL_M_TRANSPOSE_THRESHOLD = 0`
  - runtime takes the `else` branch (transpose RS) for all practical `max_M > 0`
  - file: `max/examples/custom_ops/mxfp4/grouped_matmul_sm90_entrypoints.mojo`

### Current Correctness Baseline

- Swizzled grouped test remains failing with stable error:
  - `MXFP4_GROUPED_TEST_ENABLE=1 pixi run pytest tests/test_mxfp4_grouped_matmul_ragged.py::test_mxfp4_grouped_matmul_swizzled_matches_reference[32] -q`
  - observed `max abs diff 2.703125`

### Identity Probe Findings (P=32, K=128, N=128)

- Probe setup: `A = [I_32 | 0]` so output should directly expose weight-column
  mapping.
- Baseline tuned RS fragment order (`v00,v04,v01,v05,v03,v07,v06,v02`) gives:
  - `max_abs 0.625`
  - `mean_abs 0.02808`
  - `row_identity 18/32`
  - `col_identity 12/128`
- Interpretation: not random corruption; mapping/order issue remains in RS
  fragment/epilogue path.

### Ruled-Out Changes

- Epi fragment index change:
  - `frag_idx = r_it * col_iters + c_it`
  - caused half-column zeroing and worse mapping; reverted.
- Epi lane->col mapping change:
  - `col_pair = c_it + lane_col * col_iters`
  - worsened identity mapping; reverted.
- Alternative RS pack order (`v00,v04,v01,v05,v02,v06,v03,v07`):
  - improved identity metrics (`row_identity 27`, `col_identity 15`) but worsened
    full random-input test (`max abs diff 3.691...`); not adopted.

### Practical Next Step

- Derive RS `a_frags` ordering from a mapping oracle instead of permutation
  guessing:
  1. build a tiny debug kernel that loads A-fragments via canonical loader
     (shared->reg path) for `m64n64k16`,
  2. capture per-lane fragment element order,
  3. mirror that exact order in transpose RS decode pack.

## 2026-02-06 Update (RS Fragment Order Fix)

### What Changed

- Updated transpose RS kernel `a_frags` packing order to match the canonical
  TensorCoreAsync lane contract produced by
  `vectorize[1,2] + distribute[Layout.row_major(8,4)]`.
- File:
  `max/examples/custom_ops/mxfp4/grouped_matmul_sm90_wgmma_swload_transpose.mojo`

### Correctness Result

- Previously failing test now passes:
  - `MXFP4_GROUPED_TEST_ENABLE=1 pixi run pytest tests/test_mxfp4_grouped_matmul_ragged.py::test_mxfp4_grouped_matmul_swizzled_matches_reference[32] -q`
- Added decode-contract isolation test (swizzle/index/decode only), passing:
  - `pixi run pytest tests/test_mxfp4_swizzled_decode_contract.py -q`

### Runtime Smoke (Legacy grouped RS path)

- Graph-level smoke with checkpoint weights shows no NaNs after fix:
  - `scripts/debug_legacy_moe_graph_smoke.py --tokens 64 --use-checkpoint --random-routing`
  - `scripts/debug_legacy_moe_graph_smoke.py --tokens 512 --use-checkpoint --random-routing`
  - `scripts/debug_legacy_moe_graph_smoke.py --tokens 1024 --use-checkpoint --random-routing`
- All runs: `h_nan=False`, `y_pairs_nan=False`, `y_nan=False`.

### Follow-up

- Continue with perf-only iteration now that RS grouped correctness is restored:
  1. NCU on active transpose RS kernel.
  2. Re-check global load width + integer pressure.
  3. Tighten decode/load path without changing fragment order contract.
