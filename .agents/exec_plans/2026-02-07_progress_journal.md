# 2026-02-07 Progress Journal (MXFP4 / Hopper SM90)

## Purpose
- Keep a single running log of concrete findings, surprises, and decisions.
- Preserve known-good commands and baselines before more kernel iteration.

## Snapshot: Known-Good Serving Baseline (User-validated)

### Command
```bash
python -m max.entrypoints.pipelines serve \
  --custom-architectures /workspace/modular-gptoss-mxfp4/max/examples/custom-models/gpt_oss_mxfp4 \
  --model-path openai/gpt-oss-20b \
  --quantization-encoding bfloat16 \
  --devices gpu \
  --chat-template $PIXI_PROJECT_ROOT/gpt_oss_mxfp4_v3/templates/gpt-oss-20b.chat_template.jinja \
  --device-memory-utilization 0.90 \
  --max-batch-size 512 \
  --max-length 131072 \
  --enable-in-flight-batching \
  --pretty-print-config
```

### API Probe
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-20b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Write me an article why the Bald Eagle is an American Icon."}
    ]
  }'
```

### Observed Behavior
- Architecture selected: `GptOssForCausalLM_Legacy`.
- Model compiles and serves successfully.
- Output is coherent and non-garbled (baseline correctness at serve level).
- Reported steady generation throughput around `~26 tok/s` in this run.

### Important Noise vs Signal
- `pydantic` "extra inputs are not permitted" warnings/errors remain noisy in serve logs.
- Duplicate CLI `section-name/config-file` warnings still appear.
- These warnings are not blocking kernel execution or generation in this baseline.

## Kernel/Perf Findings Carried Forward
- Swizzled RS grouped path is the target path.
- Non-swizzled grouped path is de-emphasized and optionally gated for tests.
- NCU must be symbol-locked to avoid profiling first-launch/non-hot kernels.
- Recent grouped transpose decode optimization (scale-exp hoist) reduced:
  - global load instructions
  - integer instruction pressure
  - kernel duration on P=1024/P=4096 benchmark points.

## Surprises Worth Remembering
- Several "improvements" had zero runtime effect because they were not on the active hot path.
- A stable and coherent full serve path can coexist with warning-heavy logs; warnings alone are not enough evidence of runtime failure.
- Weight/key format confusion (converted vs original/or BF16-only variants) was a major source of false debugging direction.

## Guardrails
- Treat this baseline as control: if future changes break coherence, revert and bisect from here.
- Always validate active dispatch path before tuning (entrypoint -> kernel symbol -> NCU target).
- Keep Hopper constraint in mind: avoid undersized tile directions that push kernels below efficient operating regime.

## Next Work Items
1. Keep this branch baseline runnable while tightening RS swizzled path accuracy/perf.
2. Extend serve-level validation with longer prompts and larger generated lengths.
3. Continue NCU-driven tuning only on the confirmed active kernel symbol.
4. Capture each material change here with:
   - command
   - metric delta
   - pass/fail on coherence.

## 2026-02-07 Long Generate Probe (Legacy Path)

### Command
```bash
cd max/examples/custom-models && \
pixi run python -m max.entrypoints.pipelines generate \
  --custom-architectures /workspace/modular-gptoss-mxfp4/max/examples/custom-models/gpt_oss_mxfp4 \
  --model-path openai/gpt-oss-20b \
  --quantization-encoding bfloat16 \
  --devices gpu \
  --chat-template /workspace/modular-gptoss-mxfp4/max/examples/custom-models/gpt_oss_mxfp4/templates/gpt-oss-20b.chat_template.jinja \
  --device-memory-utilization 0.90 \
  --max-batch-size 512 \
  --max-length 4096 \
  --prompt "Write a detailed article on why the bald eagle is an American icon, including history, law, and conservation."
```

### Result
- Architecture used: `GptOssForCausalLM_Legacy`.
- Build graph: `6.4s`.
- Compile: `32.3s`.
- Build + compile total: `38.6s`.
- TTFT: `~6191.6 ms`.
- Prompt eval throughput: `~14.37 tok/s`.
- Decode throughput: `~26.12 tok/s`.
- Output remained coherent (non-garbled).

### Notes
- The output still includes internal `analysis` text in this CLI path because of template/adapter behavior (known issue; not a kernel math failure).
- Throughput aligns with the known-good serve baseline and is stable in this run.

## 2026-02-07 Kernel Baseline Refresh (P=4096)

### Grouped Bench
Command:
```bash
cd max/examples/custom-models && \
pixi run python scripts/bench_mxfp4_grouped_matmul_ragged.py \
  --P 4096 --K 2880 --N 256 --which gate_up --warmup 20 --iters 100
```

Result:
- `MXFP4 grouped matmul: 0.849 ms/iter`
- `BF16 grouped matmul: 0.065 ms/iter`

Secondary run:
```bash
cd max/examples/custom-models && \
pixi run python scripts/bench_mxfp4_grouped_matmul_ragged.py \
  --P 4096 --which down --warmup 20 --iters 100
```

Result:
- `MXFP4 grouped matmul: 0.855 ms/iter`
- `BF16 grouped matmul: 0.066 ms/iter`

### Correctness Check
Command:
```bash
cd max/examples/custom-models && \
MXFP4_GROUPED_TEST_ENABLE=1 \
pixi run pytest tests/test_mxfp4_grouped_matmul_ragged.py -q
```

Result:
- `1 passed` (swizzled target test path).

### Interpretation
- Current active swizzled grouped kernel is stable and reproducible on this branch.
- Serve-level coherence is good; grouped microbenchmark remains slower than BF16 baseline, so next optimization work remains performance-focused, not emergency correctness/debug triage.

## 2026-02-07 NCU Min-Set (P=4096, active symbol)

### Kernel Symbol (locked)
- `fn_UnsafePointer_Bool_TruoAqAgAqA6A6AcB6A6AoA_11d1b77c3aecf2dc`

### Command Pattern
```bash
cd max/examples/custom-models && \
/home/m1mbp/.codex/skills/ncu-max-profiling/scripts/ncu_run_mojo.sh \
  -o profiles/grouped_p4096_minset_int \
  -k 'fn_UnsafePointer_Bool_TruoAqAgAqA6A6AcB6A6AoA_11d1b77c3aecf2dc' \
  -m 'gpu__time_duration.sum,dram__bytes_read.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,smsp__sass_inst_executed_op_global_ld.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_elapsed' \
  -- .pixi/envs/default/bin/python scripts/bench_mxfp4_grouped_matmul_ragged.py \
       --P 4096 --K 2880 --N 256 --which gate_up --warmup 20 --iters 100 --skip-bf16
```

### Metrics
- `gpu__time_duration.sum`: `~1.05 ms`
- `dram__bytes_read.sum`: `~24.5 MB`
- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`: `6,682,368`
- `smsp__sass_inst_executed_op_global_ld.sum`: `462,592`
- `smsp__sass_thread_inst_executed_op_integer_pred_on.sum`: `312,242,848`
- `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`: `~0.71%`
- `sm__warps_active.avg.pct_of_peak_sustained_elapsed`: `~2.65%`

### Tensor-Op Confirmation
- `sm__inst_executed_pipe_tensor_op_hmma.sum`: `188,416`
- `sm__inst_executed_pipe_tensor_op_gmma.sum`: `188,416`
- Conclusion: this symbol does execute tensor ops, but overall activity remains low relative to non-tensor overhead.

## 2026-02-07 Micro-Optimization Attempt (No Measurable NCU Delta)

### Change
- In `grouped_matmul_sm90_wgmma_swload_transpose.mojo`:
  - hoisted `k0_half = k0 >> 1`,
  - removed redundant `kb_rel < 4` checks,
  - compile-time gated unused scale loads (`kb0+2/+3`) for `BK=64`.

### Outcome
- Correctness remained green:
  - `MXFP4_GROUPED_TEST_ENABLE=1 pixi run pytest tests/test_mxfp4_grouped_matmul_ragged.py -q` -> `1 passed`.
- Bench remained in same band (`~0.857 ms/iter` gate_up P=4096).
- NCU min-set metrics were effectively unchanged.

### Surprising Finding
- Source-level cleanup did not move instruction counters; compiler appears to have already optimized most of this path.
- Next optimization should target structural behavior (pipeline/scheduling/layout path) rather than further micro integer cleanup.

## 2026-02-07 Structural Shape Change (Major Win)

### What Changed
- Entry-point launch shape for grouped transpose RS path changed from:
  - `BN=128, BK=128`
  to:
  - `BN=64, BK=64`
- File:
  - `max/examples/custom_ops/mxfp4/grouped_matmul_sm90_entrypoints.mojo`
- Applied consistently to grouped entrypoint variants in this file so launches stay shape-consistent.

### Kernel Bench Impact (P=4096, K=2880, N=256)
- Before: `~0.85-0.92 ms/iter`
- After: `~0.36-0.39 ms/iter`
- Approx speedup: `~2.4x`

Observed commands:
```bash
cd max/examples/custom-models && \
pixi run python scripts/bench_mxfp4_grouped_matmul_ragged.py \
  --P 4096 --K 2880 --N 256 --which gate_up --warmup 20 --iters 100 --skip-bf16
```

and

```bash
cd max/examples/custom-models && \
pixi run python scripts/bench_mxfp4_grouped_matmul_ragged.py \
  --P 4096 --which down --warmup 20 --iters 100 --skip-bf16
```

Both now run in about the `0.36-0.39 ms/iter` band.

### NCU After Shape Change (new symbol hash)
- Active symbol changed to:
  - `fn_UnsafePointer_Bool_TruoAqAgAqA6A6AcB6A6AoA_4e28ce988b0909ef`
- Key metrics:
  - `gpu__time_duration.sum`: `391.008 us` (down from ~`1.05 ms`)
  - `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`: `1.873%` (up)
  - `sm__warps_active.avg.pct_of_peak_sustained_elapsed`: `5.883%` (up)
  - `launch__waves_per_multiprocessor`: `0.48` (up from `0.24`)
- Registers per thread remained high (`255`), but the smaller tile shape improved scheduling progress/waves.

## 2026-02-07 Expanded Correctness Coverage

### Change
- Extended grouped swizzled reference test parameterization:
  - from `P=[32]`
  - to `P=[32, 512, 1024]`
- File:
  - `max/examples/custom-models/tests/test_mxfp4_grouped_matmul_ragged.py`

### Result
- `MXFP4_GROUPED_TEST_ENABLE=1 pixi run pytest tests/test_mxfp4_grouped_matmul_ragged.py -q`
- Result: `3 passed`

## 2026-02-07 End-to-End Grouped RS Sanity

### Generate Probe
- Enabled grouped RS path:
  - `MXFP4_LEGACY_GROUPED_RS=1 MXFP4_LEGACY_NO_SMALL_M=1`
- Runtime observations:
  - Build+compile time increased substantially in this probe (`~257s` total).
  - Prompt/decode throughput improved (`~42 tok/s` decode in this probe).
  - Output became garbled/non-coherent.

### Interpretation
- Kernel-level grouped benchmark performance improved substantially.
- Full model grouped path still has an integration correctness issue even with passing grouped matmul reference tests.
- Current default legacy serve path (grouped disabled) remains coherent and stable.

## 2026-02-07 Grouped Integration Root Cause + Fix

### Root Cause Found
- Added side-by-side diff harness:
  - `max/examples/custom-models/scripts/debug_legacy_moe_grouped_diff.py`
- Harness compared baseline fused MoE path vs grouped path with the same routed inputs.
- Key finding:
  - Grouped branch matched baseline in sorted space (`down_sorted` close),
  - but diverged after pair restore (`y_pairs`) when using:
    - `scatter_nd_skip_oob_indices(..., indices=restore_token_order)`.

### Correct Restore Contract
- `restore_token_order` acts as inverse-permutation for gather restore in this flow.
- Replaced grouped restore in `layers/moe.py`:
  - from scatter restore
  - to gather restore:
    - `y_pairs = ops.gather(down_output, restore_indices, axis=0)`

### Verification
- Checkpoint diff harness (tokens=64, route_expert_id=0):
  - before fix:
    - `y_pairs_group_vs_base max_abs ~147`
    - `y_group_vs_base max_abs ~245`
  - after fix:
    - `y_pairs_group_vs_base max_abs ~1`
    - `y_group_vs_base max_abs ~2`
- Grouped generate probe remains fast and now returns coherent text (still with known template `analysis` leakage).

### Files Updated for the Fix
- `max/examples/custom-models/gpt_oss_mxfp4/layers/moe.py`
  - grouped path now restores pair order with:
    - `ops.gather(down_output, restore_token_order, axis=0)`
  - removed the scatter-based restore in this path.

### Post-Fix Generate Snapshot
- Grouped env:
  - `MXFP4_LEGACY_GROUPED_RS=1 MXFP4_LEGACY_NO_SMALL_M=1`
- Observed:
  - decode throughput around `~44.7 tok/s`
  - output stayed coherent (not garbled)
  - known template `analysis` prefix leak still present (separate issue).

### Validation Note
- `tests/test_mxfp4_legacy_rs_moe_pipeline.py::test_legacy_rs_moe_pipeline_checkpoint_matches_dense_reference`
  had one transient `.mojopkg` parser/load failure, then passed on immediate rerun.
