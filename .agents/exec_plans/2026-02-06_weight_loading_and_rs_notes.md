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

