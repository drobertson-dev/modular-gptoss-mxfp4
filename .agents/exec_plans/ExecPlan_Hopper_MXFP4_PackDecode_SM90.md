# Hopper MXFP4 pack-and-decode pipeline for SM90 grouped matmul

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan follows `.agents/PLANS.md` and the repository `AGENTS.md` instructions. Critical rules restated here so this plan is self-contained: do not modify any code under `max/kernels/`; new kernel work must live under `max/examples/` (especially `max/examples/custom_ops/` and `max/examples/custom-models/`), and SM90 (H100) is the target device.

## Purpose / Big Picture

Deliver an SM90 MXFP4 grouped-matmul path that uses a “pack once, decode in the hot loop” contract. We will prepack MXFP4 weights into a Hopper-friendly bit layout and maintain a matching decode path in the kernel so FP4→BF16 expansion becomes a handful of integer bit ops plus BF16 multiplies, with scales applied as power-of-two exponent shifts. After this change, the existing grouped matmul kernel should match the Triton reference behavior for MXFP4 (packbits decode + E8M0 scaling) and be verifiable via existing MXFP4 tests and the kbench “--check” correctness path.

## Feasibility, Preconditions, and Risks

Pre-flight checks performed:
We read the local architecture and constraints docs (`.agents/OVERVIEW.md`, `.agents/ref_docs/MXFP4_KEY_TAKEAWAYS.md`, `.agents/ref_docs/MXFP4.md`) and inspected the SM90 MXFP4 kernel and decoder (`max/examples/custom_ops/mxfp4_grouped_kernels/grouped_matmul_mxfp4_sm90.mojo`, `max/examples/custom_ops/mxfp4_grouped_kernels/mxfp4_decode.mojo`). We inspected the v3 weight adapter’s packbits logic (`max/examples/custom-models/gpt_oss_mxfp4_v3/weight_adapters.py`) and the Triton example wrapper (`max/examples/custom-models/triton_example/moe.py`), noting that the actual Triton kernel/layout code is external to this repo. We also read the v3 dev workflow note (`max/examples/custom-models/gpt_oss_mxfp4_v3/DEV_WORKFLOW.md`) which explicitly says to stage packed weights/scales in shared memory and decode into registers right before WGMMA.

Preconditions that must hold:
The target environment must have an H100 (SM90) GPU, a working Mojo toolchain, and the MAX custom extension build pipeline. The plan assumes that the weight adapter’s `_mxfp4_pack_bits_u8` transform is the canonical packbits layout that the kernel should decode (the repo already treats it as such in tests).

Verified in this environment:
File inspection and code review only. We have not executed kernels or tests, and we have not verified GPU availability or the pixi environment.

Tests state:
There are existing Python correctness tests under `max/examples/custom-models/tests/test_mxfp4_grouped_matmul_ragged.py` and Mojo tests under `max/examples/custom-models/tests/mojo/`. The dev workflow document references `pixi run mxfp4-moe-bench -- --check` for correctness. We have not run any of these due to unknown GPU availability.

Risks and blockers:
The Triton layout implementation referenced by the example (`triton_kernels.tensor_details.layout`) is not present in the repository, so we cannot diff or import it directly. We will instead treat the existing Python packbits transform (`_mxfp4_pack_bits_u8`) and the Mojo decode helper (`decode_mxfp4_packbits_u32_to_8xbf16_scaled`) as the authoritative layout contract for this repo. If those are not equivalent to the Triton Hopper layout used in production, performance and accuracy may diverge.

Assumptions:
1) The packbits transform in `gpt_oss_mxfp4_v3/weight_adapters.py` is the desired Hopper-friendly layout and should remain the single source of truth for bit interleaving. If this assumption is wrong, we must redesign both packer and decoder.
2) Scale tensors use the Hopper swizzled layout `[E, N/32 (padded), K]` described below; packer and kernel must stay in lockstep.

Questions for the user (need confirmation before implementation):
Resolved: the exact Hopper scale swizzle mapping has been provided. Proceed with Hopper scale swizzle now.

## Progress

- [x] (2026-01-29) Read AGENTS/PLANS/overview and MXFP4 reference docs; confirmed constraints and target device.
- [x] (2026-01-29) Audited current MXFP4 packbits adapter, decode helpers, and grouped matmul kernel to identify current behavior and gaps.
- [x] (2026-01-29) Draft and finalize ExecPlan for pack-once layout and decode-in-kernel integration, including validation steps and decisions.
- [x] (2026-01-29) Implemented Hopper scale swizzle helpers and kernel indexing; updated weight adapters, model shapes, tests, and benches to match.

## Surprises & Discoveries

- Observation: The v3 weight adapter already applies a Hopper-style `_pack_bits` transform and transposes the weight layout to `[E, K/32, N, 16]`, which matches the kernel’s current indexing pattern. This means the core packbits contract already exists locally and does not require external Triton code.
  Evidence: `max/examples/custom-models/gpt_oss_mxfp4_v3/weight_adapters.py` and `max/examples/custom_ops/mxfp4_grouped_kernels/mxfp4_decode.mojo`.

- Observation: The SM90 WGMMA pipeline kernel currently decodes MXFP4 into BF16 in shared memory, even though comments and dev workflow suggest decode into registers immediately before WGMMA. This is acceptable for correctness but may not be the intended final performance profile.
  Evidence: `grouped_matmul_mxfp4_sm90.mojo` shows decode into shared `B_s` tiles in the producer warpgroup.

- Observation: Some tests slice larger weight tensors to create non-trivial strides; supporting these requires honoring runtime strides rather than recomputing contiguous strides.
  Evidence: `tests/test_mxfp4_grouped_matmul_ragged.py` (strided weight views test).

## Decision Log

- Decision: Adopt the vLLM-style Hopper value layout contract and enforce it end-to-end (packbits at load, unpack in-kernel).
  Rationale: The existing adapter already applies a Hopper packbits transform; matching this layout avoids optimizing around a known-slow layout.
  Date/Author: 2026-01-29 / Codex

- Decision: Implement Hopper scale swizzle now using the provided exact index mapping, and update packer + kernel together.
  Rationale: The mapping is precise and matches vLLM’s Hopper path; adopting it now avoids optimizing around a slow layout.
  Date/Author: 2026-01-29 / Codex

## Outcomes & Retrospective

Not yet executed. This section will be updated as milestones complete.

## Context and Orientation

MXFP4 weights are stored as FP4 E2M1 values packed two per byte (16 bytes per 32-element block) with an E8M0 scale per 32-element block. The v3 weight adapter (`max/examples/custom-models/gpt_oss_mxfp4_v3/weight_adapters.py`) transposes expert weights to `[E, K/32, N, 16]` and applies a packbits transform to interleave bits for cheap dequantization. The decoder helpers live in `max/examples/custom_ops/mxfp4_grouped_kernels/mxfp4_decode.mojo` and already implement a mask/shift + BF16 bias path compatible with that packbits transform. The grouped matmul kernel under `max/examples/custom_ops/mxfp4_grouped_kernels/grouped_matmul_mxfp4_sm90.mojo` contains both a warp-MMA path and a WGMMA pipeline path, and it expects weight blocks in `[E, K/32, N, 16]` and scales in the Hopper-swizzled `[E, N/32 (padded), K]` layout.

Terminology used in this plan:
“Packbits layout” means the Hopper-friendly interleaving of FP4 bits applied per 4 bytes that allows decoding via a few integer masks/shifts. “E8M0 scale” means an exponent-only power-of-two scale where the BF16 bits can be formed by shifting the exponent byte into the BF16 exponent field. “Producer warpgroup” refers to the warpgroup responsible for loading A tiles and decoding B tiles in the WGMMA pipeline. “Consumer warpgroup” refers to the warpgroup running WGMMA and writing results.

## Plan of Work

First, lock down the Hopper value layout contract and make it explicit in the kernel code. We will introduce a small “layout contract” module under `max/examples/custom_ops/mxfp4_grouped_kernels/` that documents the packbits mapping (the Hopper bit-interleaving) and the scale layout assumption in one place. Decode helpers will call into it rather than duplicating constants. This keeps the packer (Python) and decoder (Mojo) aligned.

Second, update the SM90 grouped matmul kernel to use the fast packbits decode path consistently in both the warp-MMA and WGMMA pipeline implementations. The kernel should continue to stage packed bytes and scales in shared memory but perform unpack + scale in registers immediately before storing to the BF16 shared-memory tile that feeds WGMMA. The store into shared must apply the XOR swizzle mapping (`make_swizzle`) so that WGMMA descriptors see the canonical layout.

Third, implement the Hopper scale swizzle contract and keep the scale conversion path explicit and cheap: E8M0 scales should be converted to BF16 via bit shifting, not via floating-point conversion. The Hopper scale swizzle changes indexing only; conversion stays the same.

Fourth, validate correctness using existing tests and the kbench “--check” path. If GPU tests cannot be run in this environment, the plan requires at minimum running CPU-side reference tests (where available) and documenting the limitation.

Finally, document any performance observations (if profiling is possible) and capture them in the ExecPlan’s “Surprises & Discoveries”.

## Concrete Steps

1) Add or update a layout contract module under `max/examples/custom_ops/mxfp4_grouped_kernels/` (for example, `hopper_mxfp4_layout.mojo`). This module should:
   - Define the packbits masks and unpack sequence for a 4-byte word (matching the Python packer).
   - Define a helper to convert E8M0 exponent bytes into BF16 scale bits.
   - Provide a short, plain-language comment describing the layout contract: value blocks are `[E, K/32, N, 16]`, scales are stored in Hopper-swizzled layout (see below), each block is 32 values.
   - Implement Hopper scale swizzle forward and inverse mappings (exact index math provided), with compile-time `num_warps` and alignment checks:
     - `SWIZZLE_ALIGN_M = 32 * num_warps` (M padded to multiple of this).
     - `SWIZZLE_ALIGN_K = 2` (K padded to multiple of 2).
     - Forward map: logical `S[m,k]` → stored `T[m2,k2]`.
     - Inverse map: stored `T[m2,k2]` → logical `S[m,k]`.

2) Update `max/examples/custom_ops/mxfp4_grouped_kernels/mxfp4_decode.mojo` to use the layout contract module for constants and helper functions. Keep the public decode helpers as they are (e.g., `decode_mxfp4_packbits_u32_to_8xbf16_scaled`) so existing code does not break, but ensure they share the canonical masks and scale conversion logic. Add a scale unswizzle helper that can operate on a tile in registers or shared memory.

3) Update `max/examples/custom_ops/mxfp4_grouped_kernels/grouped_matmul_mxfp4_sm90.mojo`:
   - Ensure both warp-MMA and WGMMA paths use `decode_mxfp4_packbits_u32_to_8xbf16_scaled` consistently.
   - Make the “producer” WGMMA path decode into registers, then store vectorized BF16 outputs into shared memory with the swizzle applied. Do not build a full BF16 B tile in registers; only hold fragment-sized vectors (8 BF16 values) at a time.
   - Replace scale indexing with Hopper scale unswizzle:
     - Stored scales are in layout `(N/32, K)` (per expert).
     - Kernel must unswizzle tiles to recover logical `(N, K/32)` for decoding.
   - Enforce Hopper scale constraints in compile-time checks:
     - `num_warps` is power-of-two.
     - `BLOCK_N % (32 * num_warps) == 0`.
     - `BLOCK_K % 64 == 0` when Hopper scale swizzle is enabled.
   - Keep staging of packed bytes and scales in shared memory or via coalesced global loads; the key improvement is minimizing arithmetic in the hot loop with packbits + scale shift.

4) Update or add tests:
   - If existing tests already exercise packbits decode (they do), ensure they still pass.
   - Add a small Mojo unit test under `max/examples/custom-models/tests/mojo/` if needed to compare the packbits decode against a reference decode for a fixed byte pattern and scale exponent.
   - Update Python tests that create scales to apply Hopper scale swizzle in the same way as the packer.
   - Keep tolerances tight but realistic for BF16: start with `max_abs <= 2e-2` and `max_rel <= 2e-2`, adjust only if justified by evidence.

5) Document usage:
   - Update `max/examples/custom-models/gpt_oss_mxfp4_v3/DEV_WORKFLOW.md` (if needed) to note that the packbits adapter is required and that the kernel assumes `[E, K/32, N, 16]` layout for blocks and Hopper-swizzled `[E, N/32 (padded), K]` for scales.

## Validation and Acceptance

Primary acceptance (GPU available):
Run `pixi run mxfp4-moe-bench -- --check` from `max/examples/custom-models/` and confirm the grouped matmul correctness check passes. This is the clearest end-to-end check that the packed layout and decode path agree. Then run a small kbench sweep and record throughput. The initial performance goal is to improve throughput by at least 5× over the current kernel on the same shapes; refine targets once a baseline is recorded. Use `ncu` to verify decode instruction count and memory coalescing once correctness passes.

Secondary acceptance (if only Python tests are possible):
Run `pytest max/examples/custom-models/tests/test_mxfp4_grouped_matmul_ragged.py -q` and confirm all tests pass. These tests use the Python packbits adapter and compare against a reference decode.

Manual spot-check:
Use `max/examples/custom-models/tests/test_mxfp4_grouped_matmul_ragged.py` to print a small decoded tile and verify that the same packed bytes produce the same BF16 values via both the numpy reference and the kernel path (a small local test can be added if not already present).

## Artifacts and Notes

Expected test commands (run from repo root unless noted):
  - `cd max/examples/custom-models`
  - `pixi run mxfp4-moe-bench -- --check`
  - `pytest tests/test_mxfp4_grouped_matmul_ragged.py -q`

Example expected output (abbreviated):
  - `... mxfp4_grouped_matmul_ragged_bf16 ... OK`
  - `1 passed` (for pytest run)
  - `kbench-output/...` with the new kernel showing multi‑x speedup over the previous baseline

If GPU tests cannot run, record that limitation in `Surprises & Discoveries`.

## Interfaces and Dependencies

Key modules to modify or depend on:
  - `max/examples/custom_ops/mxfp4_grouped_kernels/mxfp4_decode.mojo` (decode helpers)
  - `max/examples/custom_ops/mxfp4_grouped_kernels/grouped_matmul_mxfp4_sm90.mojo` (SM90 kernel)
  - `max/examples/custom-models/gpt_oss_mxfp4_v3/weight_adapters.py` (packbits transform)
  - `max/examples/custom-models/tests/test_mxfp4_grouped_matmul_ragged.py` (Python correctness, including swizzled scales)
  - `max/examples/custom-models/tests/mojo/` (Mojo correctness, if needed)

Key function contracts:
  - `decode_mxfp4_packbits_u32_to_8xbf16_scaled(packed_bits: UInt32, scale: BF16) -> SIMD[BF16,8]` must decode the packbits layout used by `_mxfp4_pack_bits_u8` exactly.
  - `e8m0_to_bf16_bits(scale_exp: UInt8) -> BF16` must construct the BF16 power-of-two scale without a floating-point conversion.
  - Kernel weight layouts are `[E, K/32, N, 16]` for blocks and Hopper-swizzled `[E, N/32 (padded), K]` for scales.
