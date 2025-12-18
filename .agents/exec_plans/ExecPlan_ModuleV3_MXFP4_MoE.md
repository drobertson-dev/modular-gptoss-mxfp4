# ExecPlan: ModuleV3 GPT‑OSS with MXFP4 Expert GEMMs (Decode-In-GEMM)

## Purpose and Intent

Enable running `openai/gpt-oss-20b` (and 120B later) via MAX **ModuleV3** while keeping **all non-expert model weights and activations in BF16**, and storing **only MoE expert weights in MXFP4** (packed FP4 bytes + E8M0 scale bytes). The only architectural change is swapping the two MoE expert GEMM calls so weights are **decoded inside the GEMM tile loop** and never materialized as BF16 weights in VRAM.

Success looks like:

- `pixi run serve-20b --custom-architectures .../gpt_oss_mxfp4_v3 --use-module-v3` starts and serves coherent **chat completions**.
- The MoE path uses the upstream ModuleV3 routing/activation/reorder logic unchanged, except the two expert GEMMs call our MXFP4 kernel.
- Hard precision rule is enforced: FP32 only in registers (accum + tiny epilogue temps); no FP32 shared tiles; no FP32 intermediate tensors.

## Feasibility, Preconditions, and Risks

### Repo rules (from `AGENTS.md`)

- **Do not modify** upstream MAX/Modular code under `max/` (or installed packages). All changes must live under `examples/` and `.agents/`.
- MXFP4 kernels must **decode inside** the GEMM loop; avoid weight expansion in VRAM.

### Preconditions

- A GPU with SM90 (H100) is available (required for Hopper-specific kernel tuning).
- `pixi` environment can run MAX pipelines (`pixi run ...`).
- `openai/gpt-oss-20b` safetensors are available via HF cache or can be downloaded.

### Verified in this environment

- HF cache contains `openai/gpt-oss-20b` (found under `~/.cache/huggingface/hub/models--openai--gpt-oss-20b/...`).
- Checkpoint contains MXFP4 expert tensors:
  - `model.layers.*.mlp.experts.gate_up_proj_blocks/scales`
  - `model.layers.*.mlp.experts.down_proj_blocks/scales`

### Main risks

- **Small‑M decode**: MoE decode for single-token generation often has tiny per-expert row counts. A WGMMA-only design wastes work because WGMMA instruction-level **M is fixed at 64**. We must provide a small‑M kernel path (warp‑MMA / HMMA / “GEMV-like”) to avoid compute waste.
- **Kernel correctness**: a wrong interpretation of MXFP4 packing/scales or incorrect per-expert routing can produce garbled text or memory faults. We must validate the new op against a reference decode.
- **ModuleV3 custom ops registration**: upstream `Module.compile()` does not pass `custom_extensions`; custom Mojo ops must be injected during compilation.

### Questions for the user (blocking only if unanswered)

- None required to begin implementation. (We will target `openai/gpt-oss-20b` first, then generalize.)

## Progress

- [ ] (2025-12-18) Write new `gpt_oss_mxfp4_v3` custom architecture skeleton (ModuleV3 override).
- [ ] (2025-12-18) Implement MXFP4 `grouped_matmul_ragged_mxfp4_bf16` custom op (small‑M + large‑M kernels, deterministic dispatch).
- [ ] (2025-12-18) Wire MoE layer to swap only the two expert GEMMs.
- [ ] (2025-12-18) Add tests: op-vs-reference, end-to-end generate smoke.
- [ ] (2025-12-18) Add/update microbench to track single-token vs prefill performance.

## Surprises & Discoveries

- Observation: upstream ModuleV3 uses `grouped_matmul_ragged(...)` for expert GEMMs and expects a host-side `expert_usage_stats_host` tensor.
  Evidence: docstring of `max.nn.kernels.grouped_matmul_ragged` and usage in `max.pipelines.architectures.gpt_oss_module_v3.layers.moe`.

## Decision Log

- Decision: implement a new custom architecture package `examples/custom-models/gpt_oss_mxfp4_v3` instead of continuing in `gpt_oss_mxfp4`.
  Rationale: keep the prior custom MoE WGMMA path isolated; minimize drift by extending upstream ModuleV3 and swapping only the GEMMs.
  Date/Author: 2025-12-18 / Codex

## Outcomes & Retrospective

- (To be filled as milestones complete.)

## Context and Orientation

### Key upstream behavior we must preserve

Upstream ModuleV3 GPT‑OSS MoE implementation:

- File: `max.pipelines.architectures.gpt_oss_module_v3.layers.moe`
- In `GptOssMoE.__call__`, routing and activation logic:
  - `moe_create_indices(...)` produces:
    - `token_expert_order` (reordering of expert “pairs”)
    - `expert_start_indices` (segment boundaries for active experts)
    - `restore_token_order` (inverse permutation)
    - `expert_ids` (active expert ids)
    - `expert_usage_stats` (max tokens per expert, num active experts)
  - It then calls exactly two expert GEMMs via `grouped_matmul_ragged(...)`:
    - W1: `permutated_states @ gate_up_proj.T`
    - W2: `gated_output @ down_proj.T`

We will keep this structure, replacing only the GEMM calls with an MXFP4-backed op.

### Where new code must live

- New custom architecture package: `examples/custom-models/gpt_oss_mxfp4_v3/`
- New Mojo op(s): `examples/custom_ops/kernels/` (and imported from `examples/custom_ops/kernels/__init__.mojo`)
- Tests/bench: `examples/custom-models/tests/` and `examples/custom-models/tests/mojo/`

### Precision policy (hard rule)

- FP32 is permitted **only** for register accumulators and scalar epilogue temps.
- Everything stored in memory (global/shared) must be BF16 (or U8/U32 metadata).
- Forbidden: `LayoutTensor[F32, ..., address_space=SHARED]`.

## Plan of Work

### A) Create the `gpt_oss_mxfp4_v3` custom architecture (ModuleV3 override)

Implement a custom architecture package that overrides the upstream name `GptOssForCausalLM_ModuleV3` so it is selected by `--use-module-v3`:

- `examples/custom-models/gpt_oss_mxfp4_v3/arch.py`
  - Register `SupportedArchitecture(name="GptOssForCausalLM_ModuleV3", ...)`
  - Use a custom pipeline model that instantiates our custom ModuleV3 model.
- `examples/custom-models/gpt_oss_mxfp4_v3/model.py`
  - Subclass `max.pipelines.architectures.gpt_oss_module_v3.model.GptOssModel`
  - Override `load_model()` to build the upstream config, but instantiate **our** `GptOss` ModuleV3 module (same as upstream except MoE).
  - Compile via a helper that injects `custom_extensions=[.../examples/custom_ops/kernels]`.
- `examples/custom-models/gpt_oss_mxfp4_v3/model_config.py`
  - Re-export upstream `GptOssConfig` (do not fork config derivation).
- `examples/custom-models/gpt_oss_mxfp4_v3/weight_adapters.py`
  - Map HF keys to ModuleV3 module parameter names.
  - Keep expert weights as `uint8` blocks/scales (no BF16 expansion).
  - Cast MoE biases to BF16 (or FP32 only if the kernel requires it; prefer BF16 in memory).

### B) Implement `grouped_matmul_ragged_mxfp4_bf16` Mojo op

Add a new Mojo file:

- `examples/custom_ops/kernels/grouped_matmul_mxfp4_ops.mojo`

Register a custom op name:

- `"mxfp4_grouped_matmul_ragged_bf16"`

Inputs (conceptual):

- `A`: BF16 `[P, K]` (already permuted/grouped by expert)
- `B_blocks`: U8 `[E, N, K/32, 16]`
- `B_scales`: U8 `[E, N, K/32]`
- `expert_start_indices`: U32 `[E+1]` (only first `num_active_experts+1` are meaningful)
- `expert_ids`: I32 `[E]` (only first `num_active_experts` are meaningful)
- `expert_usage_stats_host`: U32 `[2]` on CPU (max tokens per expert, num active experts)

Output:

- `C`: BF16 `[P, N]` (same ordering as `A`, i.e. grouped by expert segments)

Kernel strategy:

- Provide **exactly two** GPU kernel variants inside the op:
  - **Small‑M** (decode step): avoids WGMMA M=64 waste (warp‑MMA / GEMV‑like).
  - **Large‑M** (prefill/throughput): WGMMA path.
- Deterministic dispatch based on `max_tokens_per_expert = expert_usage_stats_host[0]`:
  - If `max_tokens_per_expert <= 64`: small‑M kernel
  - Else: large‑M kernel
- Decode MXFP4 in the K‑loop:
  - Load packed bytes for a `(K/32)` block and the scale exponent byte.
  - FP4 nibble decode uses a compile-time LUT (16 entries).
  - E8M0 exponent decode uses BF16 bit construction (no `exp2()`).
  - Multiply in BF16; accumulate in FP32 registers.

### C) Swap only the two expert GEMMs in ModuleV3 MoE

Create a custom MoE layer file that is a minimal diff from upstream:

- `examples/custom-models/gpt_oss_mxfp4_v3/layers/moe.py`
  - Copy upstream `max.pipelines.architectures.gpt_oss_module_v3.layers.moe.GptOssMoE`.
  - Replace the two calls to `grouped_matmul_ragged(...)` with our `grouped_matmul_ragged_mxfp4_bf16(...)`.
  - Store expert weights as MXFP4 blocks/scales tensors under `self.experts.*` so keys match checkpoint names:
    - `experts.gate_up_proj_blocks`, `experts.gate_up_proj_scales`
    - `experts.down_proj_blocks`, `experts.down_proj_scales`

### D) Tests and benchmarks

Add tests that fail on incorrect decode/routing:

- Unit test: compare `mxfp4_grouped_matmul_ragged_bf16` against a NumPy decode reference on a small slice of the real checkpoint.
- End-to-end: `pipelines generate` with `--use-module-v3` produces coherent text (chat template).
- Microbench: time the two expert GEMMs for `tokens1` and `tokens64` (single-step decode vs prefill-like).

## Concrete Steps

Run from repo root unless noted.

### Environment / sanity

1) Ensure pixi env works:

    cd examples/custom-models
    pixi run python -c "import max; print(max.__version__)"

### Tests

2) Run python tests:

    cd examples/custom-models
    pixi run pytest -q

### Microbench (MoE-only)

3) Run Mojo benchmark (update once new op exists):

    cd examples/custom-models
    pixi run mojo run -I "$PIXI_PROJECT_ROOT/../custom_ops" "$PIXI_PROJECT_ROOT/tests/mojo/bench_mxfp4_moe_ops.mojo" --20b --tokens1
    pixi run mojo run -I "$PIXI_PROJECT_ROOT/../custom_ops" "$PIXI_PROJECT_ROOT/tests/mojo/bench_mxfp4_moe_ops.mojo" --20b --tokens64

### End-to-end smoke (ModuleV3)

4) Run generate using the custom architecture:

    cd examples/custom-models
    pixi run python -m max.entrypoints.pipelines generate \
      --custom-architectures "$PIXI_PROJECT_ROOT/gpt_oss_mxfp4_v3" \
      --use-module-v3 \
      --model openai/gpt-oss-20b \
      --devices gpu:0 \
      --chat-template "$PIXI_PROJECT_ROOT/gpt_oss_mxfp4_v3/templates/gpt-oss-20b.chat_template.jinja" \
      --prompt "Say hello in one sentence." \
      --max-new-tokens 32 \
      --max-length 256

Expected: the decoded text is coherent English; no CUDA launch failures.

## Validation and Acceptance

This ExecPlan is complete when:

- Running the end-to-end smoke command above completes successfully and emits coherent text.
- The new op has a reference test that passes on GPU.
- Grep-gates for FP32 tiles pass:
  - `rg "LayoutTensor\\[F32" examples/custom_ops/kernels/` matches only LOCAL accumulator fragments.
  - `rg -U "address_space\\s*=\\s*AddressSpace\\.SHARED[\\s\\S]*F32" examples/custom_ops/kernels/` returns zero matches.

## Artifacts and Notes

- The custom arch must be invoked with `--custom-architectures "$PIXI_PROJECT_ROOT/gpt_oss_mxfp4_v3" --use-module-v3`.
- The chat template for gpt-oss-20b must be passed (chat completions are the coherence check).

## Interfaces and Dependencies

### New python interface (must exist)

In `examples/custom-models/gpt_oss_mxfp4_v3/kernels.py` (or equivalent), define a wrapper callable by ModuleV3:

    def grouped_matmul_ragged_mxfp4_bf16(
        a_bf16: Any,
        w_blocks_u8: Any,
        w_scales_u8: Any,
        expert_start_indices: Any,
        expert_ids: Any,
        expert_usage_stats_host: Any,
        *,
        target: str = "gpu",
    ) -> TensorValue: ...

### New Mojo op (must be registered)

In `examples/custom_ops/kernels/grouped_matmul_mxfp4_ops.mojo`:

- `@compiler.register("mxfp4_grouped_matmul_ragged_bf16")`

and `examples/custom_ops/kernels/__init__.mojo` must import it:

- `from .grouped_matmul_mxfp4_ops import *`

---

Plan revised on 2025-12-18: rewritten to be self-contained and to target the new `gpt_oss_mxfp4_v3` package, keeping upstream ModuleV3 behavior and swapping only the two expert GEMMs.
