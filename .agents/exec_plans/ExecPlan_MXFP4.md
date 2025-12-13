# MXFP4 GPT-OSS SM90 Kernel and Architecture

This ExecPlan is a living document. Maintain it in line with `.agents/PLANS.md` and `AGENTS.md`; keep `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` current as work proceeds.

Critical local rules (repeat of repo constraints):
- **Do not modify existing Modular kernel code under `max/`.** Import/reuse upstream kernels if needed; all new code goes under `examples/`.
- **Triton is the blueprint.** The reference implementation in `examples/custom-models/triton_example/` (and the rules in `.agents/ref_docs/MXFP4_KEY_TAKEAWAYS.md`) define the required kernel pattern.
- **MXFP4 must stay fused.** Stage packed bytes + scales, then decode per warp into register fragments as part of the GEMM K-loop; do not dequantize B into BF16/FP16 in shared and then run a “normal” BF16 matmul.
- **Per-subdir environments.** `examples/custom_ops/` and `examples/custom-models/` each have their own `pixi.toml`. Prefer `pixi run --manifest-path <dir> ...` (or run `pixi run ...` with CWD set to that directory) over `pixi shell` (known to reset CWD to `/workspace`).
- **Mojo tests with Mojo.** Use the Mojo runner (`mojo run`) for Mojo tests; Python tests are only for Python integration/graph wiring.

## Purpose / Big Picture

Enable GPT-OSS (20B/120B) to run natively in MXFP4 on H100 (SM90) with custom Mojo kernels that decode MXFP4 weights inside the GEMM and fuse the SwiGLU epilogue, wired into a MAX pipeline. After implementation, a user should be able to load MXFP4 checkpoints and execute generation via `max.entrypoints.pipelines` using the custom architecture, with MoE MLP1/MLP2 backed by an SM90 `wgmma` MXFP4 kernel rather than BF16 fallbacks.

## Feasibility, Preconditions, and Risks

Pre-flight checks refreshed 2025-12-10 02:26Z: `nvidia-smi` shows an H100 80GB (SM90); `mojo --version` is 0.26.1.0.dev2025120705; `python` imports `max`; `pixi --version` is 0.60.0.

Current repo state (audited 2025-12-13):
- Mojo custom ops live under `examples/custom_ops/kernels/`:
  - `examples/custom_ops/kernels/moe_mxfp4_ops.mojo` registers the MoE ops `mxfp4_moe_w1_swiglu` and `mxfp4_moe_w2_scatter` (GPU-only, correctness-first).
  - `examples/custom_ops/kernels/mxfp4_matmul_sm90.mojo` registers `gpt_oss.mxfp4.matmul.sm90` as a **CPU-only** correctness/debug op (not a performance path).
- Python custom architecture scaffolding lives under `examples/custom-models/gpt_oss_mxfp4/`; the remaining work is to ensure Python weights + routing tensors match the Mojo op interfaces exactly (Python must adapt to Mojo, not vice versa).
- Integration tests exist under `examples/custom-models/tests/` (Python) and `examples/custom-models/tests/mojo/` (Mojo).

Assumptions:
- MXFP4 safetensors checkpoints (blocks + scales + biases) are present in the HF cache (`$HF_HOME` or `~/.cache/huggingface`) or can be downloaded.

Risks:
- Correctness/perf of the eventual SM90 `wgmma` fragment mapping; mitigate by mirroring the Triton `matmul_ogs` pattern and the tiling/warp mapping documented in `.agents/OVERVIEW.md`, and by using the repository’s Mojo GPU debugging guides (`.agents/skills/debugging-mojo/`).

## Progress

- [x] (2025-12-10 00:46Z) Read `.agents/OVERVIEW.md`, `.agents/ref_docs/MXFP4_KEY_TAKEAWAYS.md`, and inspected the Triton reference (`examples/custom-models/triton_example/`).
- [x] (2025-12-10 01:41Z) Registered `gpt_oss.mxfp4.matmul.sm90` CPU reference op in `examples/custom_ops/kernels/mxfp4_matmul_sm90.mojo`.
- [x] (2025-12-10 02:40Z) Added Python env scaffold under `examples/custom-models/pixi.toml` (deps + tasks) and initial integration tests under `examples/custom-models/tests/`.
- [x] (2025-12-12 05:27Z) Reviewed `examples/custom_ops/kernels/moe_mxfp4_ops.mojo` against the Triton MoE path; confirmed high-level contracts match (remaining gaps are perf-only internals).
- [x] (2025-12-13) Updated this ExecPlan to incorporate per-subdir pixi usage, the “Python adapts to Mojo” rule, and the actual registered Mojo op signatures/weight layouts in the current tree.
- [x] (2025-12-13) Wired Python to the Mojo contracts: added wrappers (`gpt_oss_mxfp4/kernels.py`), switched MoE to call `mxfp4_moe_w1_swiglu`/`mxfp4_moe_w2_scatter`, added MXFP4 weight adapter, and ensured the pipeline graph loads custom ops via `custom_extensions`.
- [x] (2025-12-13) Implemented SM90 `wgmma` paths inside the MoE ops (`examples/custom_ops/kernels/moe_mxfp4_ops.mojo`) **without changing the Python-visible op signatures** (initial correctness-first WGMMA; perf work continues).
- [x] (2025-12-13) Added pixi tasks + ran smoke validations (Mojo runner for Mojo tests; Python tests for integration).

## Surprises & Discoveries

- `pixi shell` is disruptive in this repo (can reset CWD to `/workspace`); use `pixi run --manifest-path <dir> ...` (or run from that directory) instead.
- The dense debug op `gpt_oss.mxfp4.matmul.sm90` is CPU-only today; performance work must land in the GPU MoE ops and later the SM90 `wgmma` implementation.
- The MoE ops already implement the correct *high-level* Triton behavior (routing → W1+SwiGLU → W2 scatter-add with gammas); the remaining work is internal performance engineering and Python architecture wiring.
- Mojo test helpers in `examples/custom-models/tests/mojo/` used outdated `internal_utils.HostNDBuffer`/`DeviceNDBuffer`; replaced with a local `ndbuffer_utils.mojo` wrapper so tests run via `mojo run`.
- In Mojo GPU tests, keep tiny routing metadata like `stats = [max_tokens, num_active_experts]` on the host; reading device-resident LayoutTensor scalars on the host can crash under KGEN.
- The current MAX/Mojo CUDA toolchain is enforcing a **48KB static shared-memory cap** for these kernels (`0xc000 max` in `ptxas`). A BM=128 WGMMA version with an FP32 shared C tile overflowed this limit (`0xe400`), so the current WGMMA implementation uses **one warpgroup per CTA** with `BM=64` to stay under the cap.

## Decision Log

- Decision: Keep MoE custom op signatures stable (`mxfp4_moe_w1_swiglu`, `mxfp4_moe_w2_scatter`) and make Python adapt to those contracts.
  Rationale: Enables iterative kernel optimization (including the SM90 `wgmma` path) without repeated Python churn.
  Date/Author: 2025-12-13 Codex

## Outcomes & Retrospective

To be filled as milestones complete.

## Context and Orientation

Key directories:
- `examples/custom_ops/kernels/`: Mojo custom ops and kernels (MXFP4 kernels live here).
- `examples/custom-models/gpt_oss_mxfp4/`: Python custom architecture (must call Mojo ops with matching dtypes/layouts).
- `examples/custom-models/triton_example/`: OpenAI Triton reference (behavior + kernel structure to emulate).
- `.agents/ref_docs/MXFP4_KEY_TAKEAWAYS.md`: Critical performance constraints (must follow).
- `.agents/skills/debugging-mojo/` and `.agents/skills/testing-mojo/`: Required workflows for debugging and testing Mojo code.

Terminology:
- “Packed MXFP4 blocks”: FP4(E2M1) values packed as 2 nibbles per byte, 32 values per block → 16 bytes per (row, k_block).
- “Scale”: an E8M0 exponent byte per 32-value block (power-of-two scale). In `moe_mxfp4_ops.mojo` the scale tensor is passed as `uint8` exponent bytes.
- “Pair”: one (token, expert) assignment from top-k routing. With `TOPK=4`, there are `P = T * 4` pairs for `T` tokens.

## Plan of Work

1. **Python interface first (no kernel changes):**
   - Implement/repair `examples/custom-models/gpt_oss_mxfp4/kernels.py` wrappers that call:
     - `gpt_oss.mxfp4.matmul.sm90` for CPU-only debug checks.
     - `mxfp4_moe_w1_swiglu` and `mxfp4_moe_w2_scatter` for MoE.
   - Update the MoE layer in `examples/custom-models/gpt_oss_mxfp4/layers/moe.py` to:
     - Store MXFP4 weights as `uint8` blocks/scales with shapes matching the Mojo ops.
     - Route with `moe_create_indices` and feed `token_expert_order`/`expert_start_indices` to the custom ops.
     - Pass `gate_weights` as float32 in original pair order.

2. **Then kernel optimization (internal only):**
   - Replace the current implementation inside `examples/custom_ops/kernels/moe_mxfp4_ops.mojo` with an SM90 `wgmma` implementation that:
     - Stages packed blocks + scales (compact) and decodes per warp into register fragments in the K-loop.
     - Preserves the registered op names/signatures and the packed weight layout.
   - Current state: SM90 WGMMA is wired up and correct, but still correctness-first; next perf iteration should follow the Triton pattern more tightly (minimize dequant traffic and pipeline loads/compute).

## Concrete Steps

Run commands from repo root (`/workspace/modular-gptoss-mxfp4`), using per-subdir pixi envs:

Environment sanity:

    nvidia-smi
    mojo --version
    pixi --version

Mojo tests (use Mojo runner):

    cd examples/custom-models
    pixi run mxfp4-mojo-tests

    # SM90-only Mojo tests (optional; currently under active development)
    pixi run mxfp4-mojo-sm90-clean-tile-test
    pixi run mxfp4-mojo-sm90-moe-test

Benchmark (SM90-only):

    pixi run mxfp4-moe-bench

Python integration tests (verify wrappers + custom op loading):

    cd examples/custom-models
    pixi run mxfp4-py-tests

End-to-end smoke (once model wiring is in place):

    cd examples/custom-models
    pixi run generate

## Validation and Acceptance

Acceptance requires observable behavior:
1. Python can build a MAX Graph that calls the registered Mojo custom ops (verified by `tests/test_modular_home_bootstrap.py`).
2. `gpt_oss.mxfp4.matmul.sm90` (CPU debug op) matches a NumPy reference for synthetic weights (`tests/test_mxfp4_matmul.py`).
3. The MoE ops match a small NumPy reference against a real checkpoint slice on GPU (`tests/test_mxfp4_moe_reference.py`).
4. After SM90 optimization, kernel changes are internal-only: Python code does not change, and performance improves vs the prior kernel without dequantizing full B tiles to BF16 in shared.

## Artifacts and Notes

Evidence from pre-flight (historical):

    nvidia-smi -> H100 80GB HBM3, CUDA 13.0, no active procs.
    mojo --version -> 0.26.1.0.dev2025120705
    pixi --version -> 0.60.0

## Interfaces and Dependencies

**Mojo custom ops (source of truth):**

- `examples/custom_ops/kernels/mxfp4_matmul_sm90.mojo`
  - Op name: `gpt_oss.mxfp4.matmul.sm90`
  - Status: CPU-only correctness/debug.
  - Shapes/dtypes:
    - `a`: `[M, K]` (dtype = output dtype)
    - `b_packed`: `[K/32, N, 16]` `uint8`
    - `b_scales`: `[K/32, N]` `float32` (debug path; not the final perf format)
    - `bias`: `[N]` (dtype = output dtype)
    - `output`: `[M, N/2]` (interleaved gate/up columns fused with SwiGLU)

- `examples/custom_ops/kernels/moe_mxfp4_ops.mojo`
  - Op name: `mxfp4_moe_w1_swiglu`
    - Inputs:
      - `x`: `[T, D]` `bf16`
      - `token_expert_order`: `[P]` `uint32`
      - `expert_start_indices`: `[num_experts + 1]` `uint32` (compacted; only first `num_active_experts+1` entries are used)
      - `expert_ids`: `[num_experts]` `int32` (compacted; first `num_active_experts` entries are active expert ids)
      - `max_num_tokens_per_expert`: scalar `uint32` (host; from `expert_usage_stats[0]`)
      - `num_active_experts`: scalar `uint32` (host; from `expert_usage_stats[1]`)
      - `w_blocks`: `[num_experts, 2*I, D/32, 16]` `uint8`
      - `w_scales`: `[num_experts, 2*I, D/32]` `uint8` (E8M0 exponent bytes)
      - `bias`: `[num_experts, 2*I]` `float32`
      - `alpha`, `limit`: `float32`
    - Output: `h_sorted`: `[P, I]` `bf16`
  - Op name: `mxfp4_moe_w2_scatter`
    - Inputs:
      - `h_sorted`: `[P, I]` `bf16`
      - `token_expert_order`: `[P]` `uint32`
      - `expert_start_indices`: `[num_experts + 1]` `uint32` (compacted; only first `num_active_experts+1` entries are used)
      - `expert_ids`: `[num_experts]` `int32` (compacted; first `num_active_experts` entries are active expert ids)
      - `max_num_tokens_per_expert`: scalar `uint32` (host; from `expert_usage_stats[0]`)
      - `num_active_experts`: scalar `uint32` (host; from `expert_usage_stats[1]`)
      - `gate_weights`: `[P]` `float32` (original pair order; the op uses `pair_idx` to index it)
      - `w_blocks`: `[num_experts, D, I/32, 16]` `uint8`
      - `w_scales`: `[num_experts, D, I/32]` `uint8` (E8M0 exponent bytes)
      - `bias`: `[num_experts, D]` `float32`
    - Output: `y`: `[T, D]` `float32` (the op zero-initializes output and scatter-adds into it)

Internal kernel notes (implementation detail; not part of the Python contract):
- `examples/custom_ops/kernels/moe_mxfp4_ops.mojo` WGMMA kernels currently launch with `block_dim=(128,1,1)` (one warpgroup) and tile params `BM=64`, `BN=64`/`BN_RAW=64`, `BK=64`, `wgmma_shape=(64,64,16)`.

**Python wrappers (must match Mojo):**
- `examples/custom-models/gpt_oss_mxfp4/kernels.py` defines `get_mxfp4_kernels_path()` and thin wrappers around `ops.custom(...)` for the ops above.
- `examples/custom-models/gpt_oss_mxfp4/layers/moe.py` uses `moe_create_indices` and calls the MoE custom ops; Python code must adapt to the Mojo dtypes/layouts, not vice versa.

Update 2025-12-13: Rewrote the ExecPlan to incorporate repo workflow constraints (per-subdir pixi envs, Mojo runner), corrected stale references to non-existent paths/behaviors, and pinned the current Mojo op signatures/layouts as the Python contract to stabilize future SM90 work.
