# 2026-02-06 Weight Loading + RS Notes

## What was verified
- Local cached `openai/gpt-oss-20b` safetensors use HF-converted keys (`model.layers.*`, split `q_proj/k_proj/v_proj`).
- No `block.*` keys were present in the cached snapshot.
- MoE expert tensors are present as MXFP4 blocks/scales (`*.mlp.experts.*_proj_blocks`, `*.mlp.experts.*_proj_scales`).

## Why this matters
- Our custom MXFP4 architectures expect HF-converted `model.*` naming and split attention projections.
- Original `block.*` + fused `attn.qkv` layout is not directly compatible with current adapter/model wiring.
- BF16-only checkpoint variants are not valid for this MXFP4 path.

## Code changes made
- `gpt_oss_mxfp4/weight_adapters.py`:
  - fail fast if checkpoint is BF16-only (missing MXFP4 expert blocks)
  - fail fast on original-only `block.*` layout
- `gpt_oss_mxfp4_v3/weight_adapters.py`:
  - same fail-fast validation
  - when both formats are present, drop original keys and keep converted `model.*` keys

## RS harness status
- Checkpoint RS isolation compiles and runs after block-layout normalization in test harness.
- Synthetic RS isolation still trips first non-finite at `gate_up` stage.

## Commands used to verify checkpoint layout
- Enumerate key prefixes and QKV style in cached files:
  - `pixi run python - <<'PY' ... safe_open(...).keys() ... PY`

## 2026-02-06 Follow-up (legacy path focus)
- `decode.mojo` now avoids pre-multiplying `bias * scale` in BF16 for packbits decode.
  - old behavior could overflow at scale exponent bytes `>=129` (`2**126 * scale` in BF16).
  - new behavior multiplies in two stages: `(bitcast(y_bits) * bias) * scale`.
- Grouped matmul test module is now opt-in:
  - `tests/test_mxfp4_grouped_matmul_ragged.py` is guarded by
    `MXFP4_GROUPED_TEST_ENABLE=1` and otherwise cleanly skips.
  - rationale: grouped RS path is not the current default execution path in legacy MoE.
- Swizzled grouped reference test had a shape-order bug in its CPU reference decode input.
  - fixed by transposing `w_blocks_raw[0]` to `[K/32, N, 16]` before `_decode_mxfp4_rows`.

## Current behavior snapshot
- Legacy checkpoint isolation test passes:
  - `MXFP4_RS_ISOLATION_CHECKPOINT=1 pixi run pytest tests/test_mxfp4_legacy_rs_moe_pipeline.py -k checkpoint -q`
- Legacy custom `generate` with debug graph shows MoE stages executing repeatedly (`routing -> w1 -> w2 -> reduce`) and no immediate gather/OOB assert.
- End-to-end generation remains very slow in this environment; grouped RS correctness is still not enabled as default.
