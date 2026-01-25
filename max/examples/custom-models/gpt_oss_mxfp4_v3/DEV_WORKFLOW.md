# MXFP4 Module-First Workflow

This note captures the current eager-first workflow, profiling hooks, and
benchmarking entry points for the MXFP4 GPT-OSS custom model work. Keep new
code under `max/examples/` and avoid editing core Modular kernels.

Eager smoke (fast iteration)
- `pixi run mxfp4-eager-smoke` runs a tiny MXFP4 grouped-matmul smoke using
  `gpt_oss_mxfp4_v3/eager_mxfp4_moe_smoke.py`.
- Flags: `--tokens`, `--topk`, `--hidden`, `--intermediate`, `--experts`,
  `--iters`, `--warmup`, `--device {gpu,cpu}`.
- Hidden/intermediate must be divisible by 32 (MXFP4 block size).

Serve/generate CLI
- Use `--model-path` (not `--model`) and `--no-use-legacy-module` for ModuleV3.
- The 20B serve tasks omit default flags that are now implicit in MAX.

Profiling hooks
- `MODULAR_ENABLE_PROFILING=detailed` enables NVTX markers (nsys/ncu).
- `pixi run profile-eager-mxfp4-nsys` profiles the eager smoke with Nsight
  Systems and writes to `max/examples/custom-models/profiles/`.
- `pixi run profile-eager-mxfp4-ncu` captures kernel-level metrics into
  `max/examples/custom-models/profiles/`.
- Serve/generate tasks can add `--gpu-profiling detailed` for MAX markers.

Kernel benchmarking (kbench)
- `pixi run mxfp4-moe-kbench` runs kbench using
  `max/examples/custom-models/benchmarks/mxfp4_moe_kbench.yaml`.
- The task sets `KERNEL_BENCHMARKS_ROOT` and uses `--build-opts "-I .../custom_ops"`
  so the Mojo compiler can find the MXFP4 kernels.
- Results land in `kbench-output/` with `.txt/.csv/.pkl` summaries.
- If you have no tunable params, use `params: - {}` in the YAML to produce a
  single empty spec (avoids the `file=./` default).

Decode strategy reminder
- The MXFP4 kernel pattern should stage packed weights/scales in shared
  memory and decode into registers right before WGMMA (see
  `.agents/OVERVIEW.md` and `max/examples/custom-models/triton_example/`).
