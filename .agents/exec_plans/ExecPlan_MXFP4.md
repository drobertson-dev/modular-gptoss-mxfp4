# MXFP4 GPT-OSS SM90 Kernel and Architecture

This ExecPlan is a living document. Maintain it in line with `.agents/PLANS.md` and `AGENTS.md`; keep `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` current as work proceeds.

Critical local rules (repeat of repo constraints):

- **Do not modify existing Modular kernel code under `max/`.** Import/reuse upstream kernels if needed; all new code goes under `examples/`.
- **Triton is the blueprint.** The reference implementation in `examples/custom-models/triton_example/` (and the rules in `.agents/ref_docs/MXFP4_KEY_TAKEAWAYS.md`) define the required kernel pattern.
- **MXFP4 must stay fused.** Stage packed bytes + scales, then decode per-tile inside the GEMM K-loop. For SM90 WGMMA this can include decoding into BF16/FP16 tiles in shared (as required by WGMMA) immediately before compute; avoid any global pre-dequantization or separate BF16 matmul pass.
- **Per-subdir environments.** `examples/custom_ops/` and `examples/custom-models/` each have their own `pixi.toml`. Prefer `pixi run --manifest-path <dir> ...` (or run `pixi run ...` with CWD set to that directory) over `pixi shell` (known to reset CWD to `/workspace`).
- **Mojo tests with Mojo.** Use the Mojo runner (`mojo run`) for Mojo tests; Python tests are only for Python integration/graph wiring.

## Hard Requirements (Non-Negotiable)

Precision policy (enforced throughout this ExecPlan):

- **FP32 only in registers:** accumulator fragments + a few scalar epilogue temporaries (bias add / gamma multiply / activation math).
- **Everything else:** BF16 (or U8/U32 for metadata). **No FP32 tiles, no FP32 intermediate tensors.**
- **Default MoE path must match Triton intent:** low-precision weights/activations, FP32 accum, BF16 outputs.

## Execution Plan (P0–P3 Cleanup)

This section mirrors the “do-this-exactly” checklist for bringing drifted code back into spec.

P0 — Make W2 stop generating FP32 traffic

1) Model always uses BF16-pairs W2 path (no flags/branching)
2) Remove BF16-as-F32 aliasing in `MXFP4MoEW2PairsBF16` (true BF16 output pointer)
3) Delete FP32 W2 ops + wrappers (no dead code)

P1 — Remove FP32 “non-accumulator” tensors and tiles

4) Gate weights stored/read as BF16 (cast to FP32 only in registers)
5) Gamma shared-memory buffer is BF16 (not FP32)
6) Eliminate FP32 shared tiles (delete TC kernels; forbid `LayoutTensor[F32, ..., SHARED]`)

P2 — Remove FP32 from MXFP4 decode

7) Replace exp2-based E8M0 decode with exact BF16 bit construction (no `exp2()`)
8) Replace FP4 branch ladder with LUT; decode+scale using BF16 math only

P3 — Reduce decode/small-P launch waste

9) Fix single-token decode waste with a **kernel-family dispatch** (no “BM=64 WGMMA”):
   - SM90 WGMMA is **fixed M=64**; for tiny expert segments this wastes most work.
   - Add a small-M **HMMA / warp-MMA** kernel family for W1/W2 (M=16/32 tiles).
   - Dispatch deterministically (single threshold; no env flags):
     - if `M <= 64`: HMMA
     - else: WGMMA

Final gates (must pass): `examples/custom-models/scripts/check_mxfp4_fp32_gates.sh`

## Purpose / Big Picture

Enable GPT-OSS (20B/120B) to run natively in MXFP4 on H100 (SM90) with custom Mojo kernels that decode MXFP4 weights inside the GEMM and fuse the SwiGLU epilogue, wired into a MAX pipeline. After implementation, a user should be able to load MXFP4 checkpoints and execute generation via `max.entrypoints.pipelines` using the custom architecture, with MoE MLP1/MLP2 backed by an SM90 `wgmma` MXFP4 kernel rather than BF16 fallbacks.

## Feasibility, Preconditions, and Risks

Pre-flight checks refreshed 2025-12-10 02:26Z: `nvidia-smi` shows an H100 80GB (SM90); `mojo --version` is 0.26.1.0.dev2025120705; `python` imports `max`; `pixi --version` is 0.60.0.

Current repo state (audited 2025-12-13):

- Mojo custom ops live under `examples/custom_ops/kernels/`:
  - `examples/custom_ops/kernels/moe_mxfp4_ops.mojo` registers the MoE ops:
    - `mxfp4_moe_w1_swiglu` (W1 GEMM + fused SwiGLU)
    - `mxfp4_moe_w2_pairs_bf16` (W2 writes BF16 `y_pairs[P, D]` in original pair order)
    - `mxfp4_moe_topk_reduce_bf16` (TOPK reduction from BF16 pair-buffer → BF16 output)
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
- [x] (2025-12-13) Wired Python to the Mojo contracts: added wrappers (`gpt_oss_mxfp4/kernels.py`), switched MoE to call `mxfp4_moe_w1_swiglu` → `mxfp4_moe_w2_pairs_bf16` → `mxfp4_moe_topk_reduce_bf16` (Triton-style “compute then reduce TOPK”), added MXFP4 weight adapter, and ensured the pipeline graph loads custom ops via `custom_extensions`.
- [x] (2025-12-13) Implemented SM90 `wgmma` paths inside the MoE ops (`examples/custom_ops/kernels/moe_mxfp4_ops.mojo`) **without changing the Python-visible op signatures** (initial correctness-first WGMMA; perf work continues).
- [x] (2025-12-13) Added pixi tasks + ran smoke validations (Mojo runner for Mojo tests; Python tests for integration).
- [x] (2025-12-13) Vectorized activation (A) global loads (UInt64 -> 4x BF16) for W1/W2 WGMMA preload + prefetch, improving the `mxfp4-moe-bench` microbench substantially.
- [x] (2025-12-13) Reduced MXFP4 decode overhead by converting E8M0 scale once per 32-value block and reusing it for all 16 packed bytes during B-tile decode (Triton-like K-loop behavior).
- [x] (2025-12-14) Removed MoE CPU sync by passing `expert_usage_stats` as a GPU tensor into W1/W2 ops; updated W1/W2 kernels to use persistent-y loops with fixed launch geometry and no host-derived scalars.
- [x] (2025-12-14) Triton-like packed staging refactor: stage packed `w_blocks` + `w_scales` into shared (cp.async), then decode from staged packed blocks into BF16 tiles immediately before WGMMA for both W1 and W2; updated dynamic shared memory sizing; validated via `pixi run mxfp4-moe-bench`, `pixi run mxfp4-moe-reference-test`, and `pixi run mxfp4-mojo-sm90-moe-test`.
- [x] (2025-12-13) Switched MoE WGMMA kernels to an `.agents/OVERVIEW.md`-style CTA: `BM=128`, `BN/BN_RAW=128`, `BK=64`, `WGMMA_N=128`, `NUM_WARP_GROUPS=2` (`block_dim=(256,1,1)`), and updated host launch grids accordingly; validated via `mxfp4-moe-reference-test` + `mxfp4-moe-bench`.
- [x] (2025-12-13) Tried increasing CTA K (`BK=128`) for W1/W2 while keeping the OVErVIEW CTA; correctness passed but `mxfp4-moe-bench` regressed, so kept `BK=64`.
- [x] (2025-12-16) Implemented MXFP4 decode cleanup per precision policy: exact E8M0 → BF16 bit construction (no `exp2()`), FP4(E2M1) LUT decode, and BF16-only decode math for producing BF16 tiles; updated W1/W2 kernels to use the new helpers; validated with `pixi run mxfp4-moe-reference-test` and `pixi run mxfp4-mojo-tests`.

## Surprises & Discoveries

- `pixi shell` is disruptive in this repo (can reset CWD to `/workspace`); use `pixi run --manifest-path <dir> ...` (or run from that directory) instead.
- The dense debug op `gpt_oss.mxfp4.matmul.sm90` is CPU-only today; performance work must land in the GPU MoE ops and later the SM90 `wgmma` implementation.
- The MoE ops already implement the correct *high-level* Triton behavior (routing → W1+SwiGLU → W2 per-pair output + TOPK reduce); the remaining work is internal performance engineering and Python architecture wiring.
- Mojo test helpers in `examples/custom-models/tests/mojo/` used outdated `internal_utils.HostNDBuffer`/`DeviceNDBuffer`; replaced with a local `ndbuffer_utils.mojo` wrapper so tests run via `mojo run`.
- In Mojo GPU tests, keep tiny routing metadata like `stats = [max_tokens, num_active_experts]` on the host; reading device-resident LayoutTensor scalars on the host can crash under KGEN.
- The current MAX/Mojo CUDA toolchain is enforcing a **48KB static shared-memory cap** for these kernels (`0xc000 max` in `ptxas`). The `BM=128, BN=128` WGMMA path needs ~64KB of A/B shared tiles for a 2-stage pipeline, so A/B buffers are allocated in **dynamic shared memory** (`external_memory`) and the host launch sets `shared_mem_bytes` + `FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(...)`.
- Increasing `BK` from 64 → 128 inflated dynamic shared memory to ~128KB per block (A/B ping-pong), which appears to reduce occupancy enough that it regresses `mxfp4-moe-bench` (W1 ~1.64ms, W2 ~0.75ms vs BK=64 W1 ~1.52ms, W2 ~0.50ms).
- Attempted enabling `TensorMapSwizzle.SWIZZLE_128B` for A/B WGMMA tiles improved the microbench, but failed `tests/test_mxfp4_moe_reference.py` (max abs diff ~0.089). Reverted pending a targeted correctness test to isolate the mismatch.
- Decode-side micro-optimizations are not the primary limiter for single-token decode: the dominant issue is **small-M waste** (SM90 WGMMA uses fixed `M=64`). Use `tests/mojo/bench_mxfp4_moe_ops.mojo --tokens1` to track small-M latency while iterating on kernel families/dispatch.

## Decision Log

- Decision: Keep MoE custom op signatures stable (`mxfp4_moe_w1_swiglu`, `mxfp4_moe_w2_pairs_bf16`, `mxfp4_moe_topk_reduce_bf16`) and make Python adapt to those contracts.
  Rationale: Enables iterative kernel optimization (including the SM90 `wgmma` path) without repeated Python churn.
  Date/Author: 2025-12-13 Codex
- Decision: Use dynamic shared memory (`external_memory`) for WGMMA A/B ping-pong buffers at `BM=128, BN=128`.
  Rationale: Avoids the 48KB static shared memory cap while preserving a 2-stage pipeline at the OVErVIEW CTA tile size.
  Date/Author: 2025-12-13 Codex
- Decision: Defer 128B swizzle until correctness is reproduced in a minimal unit test.
  Rationale: The swizzle path showed measurable speedup but failed the end-to-end NumPy reference; fixing it requires tighter isolation than the MoE reference test provides.
  Date/Author: 2025-12-13 Codex
- Decision: Keep `BK=64` for the current OVErVIEW CTA until shared-memory/occupancy tradeoffs are revisited as part of the larger Triton-like refactor.
  Rationale: `BK=128` was correct but slower in `mxfp4-moe-bench` (likely due to ~128KB dynamic shared per block limiting residency).
  Date/Author: 2025-12-13 Codex
- Decision: Use exact E8M0 → BF16 bit construction for scales and LUT-based FP4(E2M1) decode in Mojo helpers (no `exp2()`).
  Rationale: Matches the hard precision policy (no FP32 intermediates outside accumulators/epilogues) and removes branchy FP4 decode; correctness is validated by the checkpoint-based MoE reference test.
  Date/Author: 2025-12-16 Codex

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
     - `mxfp4_moe_w1_swiglu`, `mxfp4_moe_w2_pairs_bf16`, and `mxfp4_moe_topk_reduce_bf16` for MoE.
   - Update the MoE layer in `examples/custom-models/gpt_oss_mxfp4/layers/moe.py` to:
     - Store MXFP4 weights as `uint8` blocks/scales with shapes matching the Mojo ops.
     - Route with `moe_create_indices` and feed `token_expert_order`/`expert_start_indices` to the custom ops.
     - Pass `gate_weights` as BF16 in original pair order (cast to FP32 in registers only).

2. **Then kernel optimization (internal only):**
   - Replace the current implementation inside `examples/custom_ops/kernels/moe_mxfp4_ops.mojo` with an SM90 `wgmma` implementation that:
     - Keeps MXFP4 decode fused inside the GEMM (no global BF16 weight materialization).
     - Uses WGMMA-required BF16/FP16 shared tiles for A/B inputs, with a pipelined K-loop.
     - Preserves the registered op names/signatures and the packed weight layout.
   - (Done) Triton-like packed staging: stage **packed** `w_blocks` + `w_scales` into shared with `async_copy`, then decode into the BF16 B tile immediately before WGMMA, so global traffic stays compact and decode work is overlapped with compute.
   - Next: tighten the pipeline (avoid `async_copy_wait_all` when possible), experiment with `async_copy_wait_group`/group sizing, and measure; only then consider 3-stage buffering or TMA.

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
4. After SM90 optimization, kernel changes are internal-only: Python code does not change, and performance improves while keeping MXFP4 decode fused in-kernel (no global BF16 weight materialization).

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
      - `expert_usage_stats`: `[2]` `uint32` (device; `expert_usage_stats[1]` is `num_active_experts`)
      - `w_blocks`: `[num_experts, 2*I, D/32, 16]` `uint8`
      - `w_scales`: `[num_experts, 2*I, D/32]` `uint8` (E8M0 exponent bytes)
      - `bias`: `[num_experts, 2*I]` `float32`
      - `alpha`, `limit`: `float32`
    - Output: `h_sorted`: `[P, I]` `bf16`
  - Op name: `mxfp4_moe_w2_pairs_bf16`
    - Inputs:
      - `h_sorted`: `[P, I]` `bf16`
      - `token_expert_order`: `[P]` `uint32`
      - `expert_start_indices`: `[num_experts + 1]` `uint32` (compacted; only first `num_active_experts+1` entries are used)
      - `expert_ids`: `[num_experts]` `int32` (compacted; first `num_active_experts` entries are active expert ids)
      - `expert_usage_stats`: `[2]` `uint32` (device; `expert_usage_stats[1]` is `num_active_experts`)
      - `gate_weights`: `[P]` `bf16` (original pair order; cast to FP32 in registers)
      - `w_blocks`: `[num_experts, D, I/32, 16]` `uint8`
      - `w_scales`: `[num_experts, D, I/32]` `uint8` (E8M0 exponent bytes)
      - `bias`: `[num_experts, D]` `float32`
    - Output: `y_pairs`: `[P, D]` `bf16` (written in original pair order; no atomics)
  - Op name: `mxfp4_moe_topk_reduce_bf16`
    - Inputs:
      - `y_pairs`: `[P, D]` `bf16`
    - Output:
      - `y`: `[T, D]` `bf16` (sums TOPK rows per token: `y[t] = sum_k y_pairs[t*TOPK+k]`)

Internal kernel notes (implementation detail; not part of the Python contract):

- `examples/custom_ops/kernels/moe_mxfp4_ops.mojo` WGMMA kernels launch with `block_dim=(256,1,1)` (2 warpgroups) and tile params `BM=128`, `BN=128`/`BN_RAW=128`, `BK=64`, `wgmma_shape=(64,128,16)`, `NUM_WARP_GROUPS=2`. Host launch uses a small heuristic: `grid_y=1` for tiny `P` else `grid_y=2`, and `grid_z=min(num_experts, P)` to avoid launching inactive experts on decode/small batches.
- A/B BF16 tiles and packed staging buffers (`B_pack{0,1}`) are in dynamic shared memory; packed `w_blocks` are staged via `gpu.memory.async_copy` and decoded to BF16 in shared immediately before WGMMA (host launch sets `shared_mem_bytes`); `w_scales` scale bytes are loaded directly and kept in registers during decode.
- Note on “full fusion”: a naïve fused W1→SwiGLU→W2 kernel would recompute W1 activations for each W2 output tile (N-tiling), so true fusion likely needs a different mapping (e.g., cluster/DSM reuse) and should be approached after closing simpler bandwidth gaps.

**Python wrappers (must match Mojo):**

- `examples/custom-models/gpt_oss_mxfp4/kernels.py` defines `get_mxfp4_kernels_path()` and thin wrappers around `ops.custom(...)` for the ops above.
- `examples/custom-models/gpt_oss_mxfp4/layers/moe.py` uses `moe_create_indices` and calls the MoE custom ops; Python code must adapt to the Mojo dtypes/layouts, not vice versa.

Update 2025-12-13: Rewrote the ExecPlan to incorporate repo workflow constraints (per-subdir pixi envs, Mojo runner), corrected stale references to non-existent paths/behaviors, and pinned the current Mojo op signatures/layouts as the Python contract to stabilize future SM90 work.
