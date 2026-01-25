# MXFP4 Register Decode Refactor (Grouped Matmul)

This ExecPlan tracks the refactor to avoid decoding full MXFP4 B tiles into shared memory by moving decode to fragment scope. The goal is to enable a true register-fragment decode path while preserving correctness and providing a testable baseline.

## Purpose / Big Picture

Replace the current WGMMA pipeline (producer decodes full B tile into shared) with a path that decodes MXFP4 per fragment and feeds tensor cores without a full BF16 shared tile. This should reduce shared memory bandwidth and align with the SM90/Triton register-decode pattern.

## Feasibility, Preconditions, and Risks

Preconditions required:
- H100 (SM90) GPU available for testing.
- Mojo compilation via Pixi (`pixi run` works).
- The MXFP4 packed weights/scales follow the checkpoint layout (contiguous).

Risks:
- Register fragment layout for BF16 MMA is non-trivial; incorrect mapping yields wrong outputs.
- If we fall back to shared scratch for B fragments, we may still carry some shared traffic (temporary compromise).
- Performance may regress while correctness-first kernels are in place.

Assumptions:
- Grouped matmul is only used for the MXFP4 MoE path (not general matmul).
- BF16 accumulation in registers and BF16 output are acceptable for baseline.

## Progress

- [x] (2026-01-25 20:22Z) Added a warp-level MMA kernel that decodes MXFP4 per fragment and stages only a small B fragment in shared (`grouped_matmul_mxfp4_sm90.mojo`).
- [x] (2026-01-25 20:25Z) Switched `mxfp4_grouped_matmul_ragged_bf16` to use the new kernel (BM/BN/BK = 64, single-warp block).
- [x] (2026-01-25 20:27Z) Verified eager smoke executes with the new kernel (`pixi run mxfp4-eager-smoke`).
- [ ] Build a correctness check comparing the grouped matmul output against numpy decode for small shapes.
- [ ] Replace shared fragment scratch with a true register-fragment decode path.
- [ ] Re-evaluate block sizes and tiling for performance once correctness is locked.

## Surprises & Discoveries

- No existing Mojo kernels decode MXFP4 directly into register fragments; all SM90 paths decode into shared (per repo search).
- TensorCore load paths (`load_a` / `load_b`) require shared memory; register-fragment mapping is not exposed directly.

## Decision Log

- Decision: Use a single-warp MMA kernel with fragment-sized B shared scratch as an interim step.
  Rationale: Enables per-fragment decode without full B tile in shared while retaining tensor core usage.
  Date/Author: 2026-01-25, Codex.

## Outcomes & Retrospective

Pending correctness validation for the new kernel. The eager smoke run succeeded (no compile errors).
