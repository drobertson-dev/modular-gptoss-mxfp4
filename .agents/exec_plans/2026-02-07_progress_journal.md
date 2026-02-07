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
