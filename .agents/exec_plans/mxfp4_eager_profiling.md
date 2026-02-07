# MXFP4 Eager Workflow + Profiling Baseline

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan must be maintained in accordance with `.agents/PLANS.md` and `.agents/AGENTS.md`. Key local rules to restate here: do not modify existing Modular kernel code, and put new code under `max/examples/` only. The environment targets H100 (SM90). MXFP4 kernels must follow the shared-memory staging and register decode pattern described in `.agents/OVERVIEW.md`.

## Purpose / Big Picture

Enable a faster, eager-first development loop for the MXFP4 GPT-OSS custom model, backed by repeatable profiling and kernel benchmarking. After this change, a developer can run a small eager smoke test without full graph compilation, profile it with Nsight, and run a kbench-driven kernel benchmark from the repo root with a stable config file.

## Feasibility, Preconditions, and Risks

Pre-flight checks performed:
- Inspected `max/examples/custom-models/pixi.toml` to identify serve/generate tasks and defaults.
- Inspected `max/examples/custom-models/gpt_oss_mxfp4_v3/kernels.py` to confirm custom op wrapping uses `ops.custom`.
- Inspected `max/python/max/functional.py` to confirm `F.custom(..., custom_extensions=...)` auto-loads kernels.
- Confirmed kbench tooling exists under `max/kernels/benchmarks/autotune/` and `bazelw` exists at repo root.
- Confirmed `bench_mxfp4_moe_ops.mojo` lacks `dump_report()` and so cannot emit kbench outputs.

Preconditions required:
- A working MAX nightly toolchain and Pixi environment (`pixi` + `modular`).
- `mojo` in PATH (Pixi env usually provides this).
- Nsight tools (`nsys`, `ncu`) for profiling tasks.
- H100 (SM90) GPU for MXFP4 kernel execution.

Verification status:
- `pixi run mxfp4-eager-smoke` succeeded (GPU) and printed output.
- `pixi run profile-eager-mxfp4-nsys` succeeded; report at `max/examples/custom-models/profiles/eager_mxfp4_moe.nsys-rep`.
- `pixi run mxfp4-moe-bench` succeeded; baseline GFLOPS captured.
- `kbench` run succeeded; outputs at `kbench-output/mxfp4_moe_kbench.{txt,csv,pkl}`.
- `ncu` available but not run (requires `sudo`).

Risks:
- Eager-mode custom ops require custom extensions to be loaded in the active graph; if not loaded, MXFP4 ops will fail at runtime.
- kbench requires `bench.dump_report()` output and correct Mojo include paths; missing either prevents result collection.
- If MAX CLI flags change again, task definitions in `pixi.toml` may drift.

Assumptions:
- The Pixi environment is active when running `pixi run` so PATH contains `mojo`.
- The H100 is accessible to this container.
- Bazel builds can run from repo root.

## Progress

- [x] (2026-01-25 19:10Z) Added eager-style smoke harness for MXFP4 grouped matmul in `max/examples/custom-models/gpt_oss_mxfp4_v3/eager_mxfp4_moe_smoke.py`.
- [x] (2026-01-25 19:12Z) Switched MXFP4 custom op wrapper to `F.custom(..., custom_extensions=...)` to support eager contexts.
- [x] (2026-01-25 19:15Z) Updated `bench_mxfp4_moe_ops.mojo` to emit `bench.dump_report()` for kbench compatibility.
- [x] (2026-01-25 19:20Z) Added kbench config at `max/examples/custom-models/benchmarks/mxfp4_moe_kbench.yaml`.
- [x] (2026-01-25 19:25Z) Updated `max/examples/custom-models/pixi.toml` to use `--model-path`, drop default serve flags, and add eager/kbench/profile tasks.
- [x] (2026-01-25 19:30Z) Added `max/examples/custom-models/gpt_oss_mxfp4_v3/DEV_WORKFLOW.md` to document the new workflow.
- [x] (2026-01-25 19:38Z) Ran eager smoke via pixi; GPU path executed and printed output.
- [x] (2026-01-25 19:41Z) Ran `profile-eager-mxfp4-nsys`; report generated under `max/examples/custom-models/profiles/`.
- [x] (2026-01-25 19:46Z) Ran Mojo MoE benchmark; recorded baseline GFLOPS for w1/w2 kernels.
- [x] (2026-01-25 19:58Z) Ran kbench; outputs generated in `kbench-output/`.
- [x] (2026-01-25 19:59Z) Updated this ExecPlan with validation outcomes.

## Surprises & Discoveries

- `max.functional.custom` can load custom kernel libraries directly via `custom_extensions`, so eager-mode MXFP4 ops can register themselves without explicit graph creation.
  Evidence: `max/python/max/functional.py` exposes `custom(..., custom_extensions=...)`.
- `bench_mxfp4_moe_ops.mojo` printed results but did not call `dump_report()`, which kbench expects for CSV/PKL output.
  Evidence: `max/examples/custom-models/tests/mojo/bench_mxfp4_moe_ops.mojo` lacked `dump_report()` before edits.
- `kbench` treats empty `params: []` as "no mesh" and uses a default instance with `file=./`, which breaks output naming.
  Evidence: `ValueError: PosixPath('.') has an empty name` from `kbench_model.py:file_stem`.
  Fix: use `params: - {}` to generate a single empty spec.
- MXFP4 decode currently writes decoded BF16 values into shared memory (not registers), which conflicts with the recommended Hopper path.
  Evidence: "Decode B tile (MXFP4 -> BF16 in shared)" blocks in `max/examples/custom_ops/kernels/grouped_matmul_mxfp4_sm90.mojo:284` and `max/examples/custom_ops/kernels/grouped_matmul_mxfp4_sm90.mojo:666`.

## Decision Log

- Decision: Switch `mxfp4_grouped_matmul_ragged_bf16` to use `F.custom` with `custom_extensions` defaulting to the MXFP4 kernel path.
  Rationale: Enables eager-style execution without explicit graph construction while staying compatible with pipeline compilation.
  Date/Author: 2026-01-25, Codex.
- Decision: Reuse `bench_mxfp4_moe_ops.mojo` with `dump_report()` and drive it via a simple kbench YAML config instead of authoring a new benchmark file.
  Rationale: Minimal code changes; preserves existing CLI flags and keeps new code under `max/examples/`.
  Date/Author: 2026-01-25, Codex.

## Outcomes & Retrospective

Validation complete. Baseline results captured:
- Eager smoke: ~3.25s for 1 iter, output scalar `0.0`.
- Mojo bench: w1 ~4952 GFLOPS, w2 ~4761 GFLOPS, w2+reduce ~4757 GFLOPS.
- kbench: ~6.9 ms (w1), ~3.6 ms (w2 / w2+reduce) with outputs in `kbench-output/`.
Remaining gap: ncu profiling requires `sudo`, not executed yet.

## Context and Orientation

The custom MXFP4 model lives under `max/examples/custom-models/gpt_oss_mxfp4_v3/`. The MXFP4 grouped matmul custom op wrapper is `max/examples/custom-models/gpt_oss_mxfp4_v3/kernels.py`, and the Mojo kernels live under `max/examples/custom_ops/`. The existing Mojo microbenchmark is `max/examples/custom-models/tests/mojo/bench_mxfp4_moe_ops.mojo`. Pixi task entry points are defined in `max/examples/custom-models/pixi.toml`. The kbench driver is in `max/kernels/benchmarks/autotune/` and expects benchmark binaries to emit `dump_report()` output.

## Plan of Work

Update the MXFP4 custom op wrapper to use `F.custom` with `custom_extensions` so it runs in eager contexts without explicit graph setup. Add a small eager smoke harness that allocates synthetic data, builds minimal routing buffers, and invokes the custom op, printing a single scalar to force compilation and execution. Update the Mojo MoE microbenchmark to call `bench.dump_report()` so kbench can capture results. Add a kbench YAML config under examples, and add Pixi tasks for the eager smoke, profiling with Nsight Systems, and running kbench from the repo root. Update serve/generate tasks in `pixi.toml` to use `--model-path` and drop default flags to align with the new CLI defaults. Document the workflow in a short DEV note under the v3 model directory.

## Concrete Steps

All commands assume the repo root as `/workspace/modular-gptoss-mxfp4` unless noted.

1) Eager smoke harness
   - Create `max/examples/custom-models/gpt_oss_mxfp4_v3/eager_mxfp4_moe_smoke.py`.
   - Run: `cd max/examples/custom-models && pixi run mxfp4-eager-smoke`
   - Expect output similar to:
     mxfp4 eager smoke rows=256 hidden=256 intermediate=256 experts=4 device=gpu iters=1 elapsed_s=... out0=0.0

2) Update MXFP4 custom op wrapper
   - Edit `max/examples/custom-models/gpt_oss_mxfp4_v3/kernels.py` so `mxfp4_grouped_matmul_ragged_bf16` calls `F.custom(..., custom_extensions=...)`.

3) Enable kbench output from Mojo benchmark
   - Edit `max/examples/custom-models/tests/mojo/bench_mxfp4_moe_ops.mojo` to call `bench.dump_report()` after printing results.

4) Add kbench config
   - Create `max/examples/custom-models/benchmarks/mxfp4_moe_kbench.yaml`.
   - Run: `cd max/examples/custom-models && pixi run mxfp4-moe-kbench`
   - Expect `kbench-output/` to include `output.txt`, `output.csv`, and `output.pkl`.

5) Update Pixi tasks and documentation
   - Edit `max/examples/custom-models/pixi.toml` to use `--model-path` in serve/generate tasks, drop default flags, and add eager/kbench/profile tasks.
   - Add `max/examples/custom-models/gpt_oss_mxfp4_v3/DEV_WORKFLOW.md` to document the new workflow.

## Validation and Acceptance

Manual validation (no automated tests added):
- `pixi run mxfp4-eager-smoke` completes without errors and prints a scalar output.
- `pixi run profile-eager-mxfp4-nsys` produces a `.nsys-rep` under `max/examples/custom-models/profiles/`.
- `pixi run mxfp4-moe-kbench` produces `kbench-output/output.csv` and `kbench-output/output.pkl`.
- `pixi run serve-20b-v3` starts a server without CLI flag errors and prints the config.

If any step fails, capture the exact error and update the `Surprises & Discoveries` section with evidence.

## Artifacts and Notes

Expected kbench output layout:
  kbench-output/output.txt
  kbench-output/output.csv
  kbench-output/output.pkl

Example eager smoke output:
  mxfp4 eager smoke rows=256 hidden=256 intermediate=256 experts=4 device=gpu iters=1 elapsed_s=0.012345 out0=0.0

## Interfaces and Dependencies

Dependencies:
- `max.functional.custom` for eager custom op execution.
- `max/examples/custom_ops/` for Mojo kernels and includes.
- `mojo` compiler and runtime.
- `nsys` and `ncu` for profiling.

Primary interfaces to end in this change:
- `mxfp4_grouped_matmul_ragged_bf16(..., custom_extensions=...)` in `max/examples/custom-models/gpt_oss_mxfp4_v3/kernels.py`.
- `mxfp4-eager-smoke`, `profile-eager-mxfp4-nsys`, and `mxfp4-moe-kbench` tasks in `max/examples/custom-models/pixi.toml`.
