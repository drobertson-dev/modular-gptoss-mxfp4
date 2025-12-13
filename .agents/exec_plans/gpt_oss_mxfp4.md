# MXFP4 GPT-OSS SM90 Kernel and Architecture

**Status:** Superseded by `.agents/exec_plans/ExecPlan_MXFP4.md`. This file is kept for historical context only; do not execute it without reconciling it with the canonical ExecPlan.

This ExecPlan is a living document. Maintain it in line with `.agents/PLANS.md` and `AGENTS.md`; keep `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` current as work proceeds. Critical local rules: do not modify existing Modular kernel code under `max/`; add all new code under `examples/`. MXFP4 dequantization must stay fused inside the GEMM (decode in registers after staging packed data in shared), mirroring the Triton reference.

## Purpose / Big Picture

Enable GPT-OSS (20B/120B) to run natively in MXFP4 on H100 (SM90) with a custom Mojo kernel that decodes MXFP4 weights inside the GEMM and fuses the SwiGLU epilogue, wired into a MAX pipeline. After implementation, a user should be able to load MXFP4 checkpoints and execute generation via `max.entrypoints.pipelines` using the custom architecture, with MoE MLP1/MLP2 backed by the MXFP4 SM90 kernel rather than BF16 fallbacks.

## Feasibility, Preconditions, and Risks

Pre-flight checks: `nvidia-smi` shows an H100 80GB (SM90) available; `mojo --version` reports 0.26.1.0.dev2025120705; `python` imports `max` successfully; `pixi --version` is 0.60.0. The `safetensors` package is missing (import error), so Python weight loading will need to add it to the environment (e.g., in `examples/custom-models/pixi.toml`). Existing `examples/custom-models/gpt_oss_mxfp4/` files are mostly empty placeholders; the Triton reference under `examples/custom-models/triton_example/` is present and uses `triton_kernels.matmul_ogs` with fused SwiGLU as the blueprint. No MXFP4-specific tests exist yet; new unit/integration checks must be added. Assumptions: MXFP4 checkpoints (blocks + scales) are available at execution time; if not, synthetic weights will be used for correctness tests only, not fidelity. Risk: correctness/perf of SM90 wgmma layout mapping; mitigate by mirroring `max/kernels/src/linalg/matmul/gpu/sm90/*` fragment ordering and Triton’s `matmul_ogs` tiling (BLOCK_M/N=128, BLOCK_K≈64, FRAG_K≈16–32). Another risk is API drift in MAX Python; favor local wrappers in `examples/custom-models/gpt_oss_mxfp4/` to insulate from upstream changes.

## Progress

- [x] (2025-12-10 00:46Z) Read AGENTS/OVERVIEW, MXFP4 key docs, Triton reference; inspected existing MAX GPT-OSS code and confirmed GPU/tools availability; noted missing `safetensors`.
- [ ] Flesh out MXFP4 decode utilities, SM90 kernel, and op registration under `examples/custom_ops/kernels/`.
- [ ] Implement Python wrappers, weight loader, and GPT-OSS MXFP4 architecture under `examples/custom-models/gpt_oss_mxfp4/`.
- [ ] Add tests/benchmarks and run smoke validations on H100.
- [ ] Update this plan with discoveries, decisions, and retrospective.

## Surprises & Discoveries

- H100 80GB is available locally; good for SM90 work without further setup.
- `safetensors` is not installed; weight loading will fail until the dependency is added to the `examples/custom-models` environment.
- `examples/custom-models/gpt_oss_mxfp4/` is effectively empty; only a weight name map exists, so the architecture and kernels must be built from scratch.
- Triton reference (`examples/custom-models/triton_example/moe.py`) already fuses matmul+SwiGLU with MXFP4 decode; it is the behavioral template to mirror in Mojo.

## Decision Log

- Decision: Target SM90 tiling BLOCK_M=128, BLOCK_N=128, BLOCK_K≈64 with FRAG_K 16–32, decode MXFP4 in registers per warp and fuse SwiGLU epilogue (limit=7.0, alpha=1.702, interleaved gate/up layout).  
  Rationale: Matches `.agents/OVERVIEW.md` and Triton `matmul_ogs` pattern to keep the kernel compute-bound and minimize shared-memory footprint.  
  Date/Author: 2025-12-10 Codex

## Outcomes & Retrospective

To be filled as milestones complete.

## Context and Orientation

Relevant docs: `.agents/OVERVIEW.md` (SM90 kernel shape, wgmma fragment handling, fused SwiGLU), `.agents/ref_docs/MXFP4_KEY_TAKEAWAYS.md` (must keep decode fused, block size 32 values, power-of-two float8 scales, avoid extra memory traffic), `.agents/ref_docs/MXFP4.md` (format details), `.agents/ref_docs/GPT_OSS_OVERVIEW.md` (model architecture: 36 layers, MoE top-4, alternating dense/banded attention, SWIGLU with clamp).

Repository layout for this task: new Mojo kernels live in `examples/custom_ops/kernels/`; custom Python architecture in `examples/custom-models/gpt_oss_mxfp4/`; Triton reference in `examples/custom-models/triton_example/` shows the desired behavior. Existing MAX GPT-OSS (BF16) lives under `max/python/max/pipelines/architectures/gpt_oss/` and can be used for control flow and config patterns but must not be modified. Use `examples/custom-models/pixi.toml` to manage Python deps/tasks; add `safetensors` and tests there.

MXFP4 layout: groups of 32 FP4(E2M1) values packed into 16 bytes with one float8_e8m0fnu scale per group (power-of-two). Decoding: nibble→LUT→scale (ldexp)→FP16; scales stay packed alongside blocks. Kernels must stage packed blocks and scales in shared, decode per warp into register fragments, then issue wgmma; dequantizing to BF16 in shared is forbidden for perf. SwiGLU epilogue clamps gate/up to ±limit, computes gate*sigmoid(alpha*gate)*(up+1) with interleaved even/odd columns.

## Plan of Work

First, add MXFP4 decode utilities in a new Mojo helper (e.g., `examples/custom_ops/kernels/mxfp4_utils.mojo`): define LUT for FP4 values, constants for block size (32), scale dtype (float8_e8m0fnu), and helpers to map (k,n) to block/byte indices and to decode 16 bytes + one scale into FP16 vectors. Provide a tiny CPU reference decode for tests and reuse BF16/NVFP4 helpers from `max/kernels/src/linalg/*` via imports where possible without modifying them.

Next, implement the SM90 MXFP4 matmul kernel in `examples/custom_ops/kernels/mxfp4_matmul_sm90.mojo`: CTA tile 128×128 over C, K loop in 64-wide slices with FRAG_K 16–32, 8 warps/CTA. Stage A tiles (BF16) and packed B blocks + scales into shared; for each FRAG_K chunk, load A fragment via existing SM90 loaders, decode B fragment from packed bytes/scales into the fragment layout expected by wgmma, accumulate FP32. After the K loop, fuse bias + SwiGLU epilogue on the register C fragment (interleaved even/odd columns) and store BF16. Guard for SM90; keep no shared-memory expanded B. Mirror fragment ordering from `max/kernels/src/linalg/matmul/gpu/sm90/*` and Triton `matmul_ogs`.

Register the kernel as a custom op (e.g., `gpt_oss.mxfp4.matmul.sm90`) in the same Mojo file: `@compiler.register` struct with `execute` taking C output, A BF16, B_packed uint8, B_scales float8, bias BF16, alpha/limit scalars, and DeviceContextPtr; dispatch SM90 path and optionally a CPU/TensorCore fallback using dequantized matmul for tests.

Add Python wrappers in `examples/custom-models/gpt_oss_mxfp4/kernels.py`: a function `mxfp4_matmul_swiglu(x, w_blocks, w_scales, bias, alpha=1.702, limit=7.0)` that calls `ops.custom("gpt_oss.mxfp4.matmul.sm90", ...)`, infers N from blocks/scales, and falls back to dequant+ops.matmul when SM90 is unavailable or for debug. Include shape validation (K multiple of 32, block/scales alignment).

Build MXFP4 weight loading/prep in `examples/custom-models/gpt_oss_mxfp4/weight_loader.py`: complete the checkpoint mapping, load blocks/scales from safetensors, cast scales to float8_e8m0fnu, reshape to `[K/32, N]` (or `[num_experts, K/32, N_local]`), and provide helpers to move them to devices. Include a Python dequant reference for tests. Add config dataclasses in `model_config.py` covering GPT-OSS hyperparameters (from `.agents/ref_docs/GPT_OSS_OVERVIEW.md`) for 20B and 120B, plus runtime fields (devices, dtype, rope, swiglu_limit, etc.).

Implement the model under `examples/custom-models/gpt_oss_mxfp4/`: attention layer (reuse MAX primitives for RMSNorm, linear, RoPE, dense/banded attention similar to `max/pipelines/architectures/gpt_oss/layers/attention.py`), and MoE layer that uses router top-4, `moe_create_indices`, and the new `mxfp4_matmul_swiglu` for MLP1 and the same kernel (without fused activation) or a simple dequant+matmul path for MLP2. Wire them into a `GptOssMXFP4` block stack in `gpt_oss.py`, add arch registration in `arch.py` (e.g., `GptOssMXFP4ForCausalLM`) and a pipeline model in `model.py` mirroring the existing MAX GPT-OSS model but sourcing weights/scales from the MXFP4 loader. Keep everything under `examples/custom-models/gpt_oss_mxfp4/`; import but do not edit `max/*`.

Add validation scaffolding: a small correctness test comparing the custom op against a Python dequant+torch matmul+SwiGLU for random tensors (K,N multiples of 128) and a smoke graph run using synthetic weights. Place tests under `examples/custom-models/gpt_oss_mxfp4/tests/` and wire pixi tasks in `examples/custom-models/pixi.toml` (e.g., `pixi run mxfp4-matmul-test`). For performance sanity, add an optional benchmark script that times the SM90 kernel vs dequant matmul on H100. Document how to run an end-to-end generation with a real checkpoint via `python -m max.entrypoints.pipelines generate --custom-architectures examples/custom-models/gpt_oss_mxfp4 --model <checkpoint>` once weights exist.

## Concrete Steps

Work in `/workspace/modular-gptoss-mxfp4`. Commands already run:

    nvidia-smi
    mojo --version
    python - <<'PY'\nimport sys\nimport max\nprint(sys.version)\nprint('max version', getattr(max,'__version__','unknown'))\nPY
    pixi --version
    python - <<'PY'\nimport safetensors\nPY   # currently fails (module missing)

Upcoming steps during execution (update as completed):

    # Edit Mojo helpers and kernel
    # Edit Python wrappers/architecture
    # Add pixi deps/tasks under examples/custom-models
    pixi run -p examples/custom-models mxfp4-matmul-test      # after tests exist
    pixi run -p examples/custom-models gpt-oss-mxfp4-smoke    # synthetic smoke
    python -m max.entrypoints.pipelines generate --custom-architectures examples/custom-models/gpt_oss_mxfp4 --model <checkpoint> --prompt "Hello"  # real weights

## Validation and Acceptance

Acceptance requires: (1) MXFP4 matmul+SwiGLU custom op matches a Python dequant+torch matmul reference within BF16 tolerance on representative shapes (including expert-partitioned layouts); (2) fallback path works on CPU/non-SM90 for tests; (3) end-to-end GPT-OSS MXFP4 graph runs a short prompt without errors using synthetic weights, and with real MXFP4 checkpoints once provided; (4) performance sanity shows SM90 kernel executes without expanding B to shared (confirm via code inspection and optional benchmark). Tests in the repo (pixi tasks) must pass.

## Artifacts and Notes

Evidence from pre-flight:

    nvidia-smi -> H100 80GB HBM3, CUDA 13.0, no active procs.
    mojo --version -> 0.26.1.0.dev2025120705
    python import max -> success; safetensors -> ModuleNotFoundError
    pixi --version -> 0.60.0

## Interfaces and Dependencies

Kernel interface (planned): custom op name `gpt_oss.mxfp4.matmul.sm90`; inputs `(C_out: OutputTensor[bf16, rank=2], A: InputTensor[bf16, rank=2], B_packed: InputTensor[uint8, rank=2 or 3 for experts], B_scales: InputTensor[float8_e8m0fnu, matching packed dims], bias: InputTensor[bf16, rank=1], alpha: Float32, limit: Float32, ctx: DeviceContextPtr)`; output `[M,N]` BF16. B layout expects K multiple of 32 with packed bytes shaped `[K/32, N]` (or `[experts, K/32, N_local]`), scales matching. Python wrapper signature `mxfp4_matmul_swiglu(x, w_blocks, w_scales, bias, alpha=1.702, limit=7.0) -> TensorValue`.

Dependencies to declare: `safetensors`, `torch` (for reference tests), `max` (already present), optional `pytest` for tests. Use MAX Graph ops (`ops.custom`, `ops.softmax`, `moe_create_indices`, etc.) and existing attention utilities from `max.nn`. Ensure pixi env under `examples/custom-models` includes new deps so tasks run reproducibly.

Update 2025-12-13: Marked this ExecPlan as superseded by `.agents/exec_plans/ExecPlan_MXFP4.md` (canonical).
