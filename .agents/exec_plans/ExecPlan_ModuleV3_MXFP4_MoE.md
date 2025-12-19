# ExecPlan: ModuleV3 GPT‑OSS with MXFP4 Expert GEMMs (Decode-In-GEMM)

## Purpose and Intent

Enable running `openai/gpt-oss-20b` (and 120B later) via MAX **ModuleV3** while keeping **all non-expert model weights and activations in BF16**, and storing **only MoE expert weights in MXFP4** (packed FP4 bytes + E8M0 scale bytes). The only architectural change is swapping the two MoE expert GEMM calls so weights are **decoded inside the GEMM tile loop** and never materialized as BF16 weights in VRAM.

Performance intent (H100 / Hopper): match the spirit of OpenAI/vLLM’s Triton `matmul_ogs` Hopper path:

- Feed **BF16 tensor cores** (WGMMA/HMMA) and keep **FP32 accumulators in registers** only.
- Make dequantization “nearly free” by using **offline bit-twiddling** of FP4 bytes (`_pack_bits`) and a **bit-unpack** decode sequence (mask/shift + `mul.bf16x2`-style math), rather than a branchy FP4 ladder or FP32 `exp2()`.
- Follow the **matmul_ogs recipe** exactly: offline swizzled weights/scales + packed-BF16 dequant + BF16 TC matmul + MoE-aware scheduling. Do not “approximate” this with kernels that are merely “close enough”.

Success looks like:

- `pixi run serve-20b-v3 --custom-architectures .../gpt_oss_mxfp4_v3 --use-module-v3` starts and serves coherent **chat completions**.
- The MoE path uses the upstream ModuleV3 routing/activation/reorder logic unchanged, except the two expert GEMMs call our MXFP4 kernel.
- Hard precision rule is enforced: FP32 only in registers (accum + tiny epilogue temps); no FP32 shared tiles; no FP32 intermediate tensors.

## Feasibility, Preconditions, and Risks

### Repo rules (from `AGENTS.md`)

- **Do not modify** upstream MAX/Modular code under `max/` (or installed packages). All changes must live under `max/examples/` and `.agents/`.
- MXFP4 kernels must **decode inside** the GEMM loop; avoid weight expansion in VRAM.

### Preconditions

- A GPU with SM90 (H100) is available (required for Hopper-specific kernel tuning).
- `pixi` environment can run MAX pipelines (`pixi run ...`).
- `openai/gpt-oss-20b` safetensors are available via HF cache or can be downloaded.
- The repo has a local Triton `triton_kernels` reference checkout (used only as a **spec**):
  - `/tmp/triton/python/triton_kernels/triton_kernels/tensor_details/layout_details/hopper_value.py`
    - `HopperMXValueLayout._pack_bits(...)`
    - `_unpack_fp4_to_bf16_triton(...)` bit-unpack masks we will port to Mojo
  - `/tmp/triton/python/triton_kernels/triton_kernels/matmul_ogs_details/opt_flags_details/opt_flags_nvidia.py`
    - Hopper FP4 uses `BLOCK_K=128` when no native FP4 tensor cores exist (H100).

### Verified in this environment

- HF cache contains `openai/gpt-oss-20b` (found under `~/.cache/huggingface/hub/models--openai--gpt-oss-20b/...`).
- Checkpoint contains MXFP4 expert tensors:
  - `model.layers.*.mlp.experts.gate_up_proj_blocks/scales`
  - `model.layers.*.mlp.experts.down_proj_blocks/scales`

### Main risks

- **Small‑M decode**: MoE decode for single-token generation often has tiny per-expert row counts. WGMMA instruction-level **M is fixed at 64**, so we waste compute for `max_M < 64`. We intentionally accept this for now because a naïve GEMV path was significantly slower; the next performance step is producing **swizzled B tiles** (and/or a purpose-built decode kernel) so the waste is amortized by higher tensor-core utilization.
  - Note: WGMMA’s instruction shape has **M fixed at 64 on SM90** (for everyone). vLLM/Triton avoid visible waste via MoE-aware tiling + scheduling (block shapes, split‑K/persistence) so SMs stay busy even when per-expert `M` is small.
- **Kernel correctness**: a wrong interpretation of MXFP4 packing/scales or incorrect per-expert routing can produce garbled text or memory faults. We must validate the new op against a reference decode.
- **ModuleV3 custom ops registration**: upstream `Module.compile()` does not pass `custom_extensions`; custom Mojo ops must be injected during compilation.
- **Performance cliff if dequant is “generic”**: decoding via BF16 LUT + scalar BF16 multiplies is correct but is usually stuck around ~0.25–0.3× BF16 throughput on H100. The hard-path requires the `_pack_bits` + bit-unpack decode so dequant is cheap enough to hide behind tensor-core math.

### Questions for the user (blocking only if unanswered)

- None required to begin implementation. (We will target `openai/gpt-oss-20b` first, then generalize.)

## Progress

- [x] (2025-12-18) Write new `gpt_oss_mxfp4_v3` custom architecture skeleton (ModuleV3 override).
- [x] (2025-12-18) Implement MXFP4 `mxfp4_grouped_matmul_ragged_bf16` custom op (TC + WGMMA variants, deterministic dispatch).
- [x] (2025-12-18) Wire ModuleV3 MoE layer to swap only the two expert GEMMs.
- [x] (2025-12-18) Add tests: op-vs-reference (incl. synthetic long-K stress) and smoke runs.
- [x] (2025-12-18) Fix correctness regression: respect runtime strides for A/C and MXFP4 blocks/scales to prevent NaNs and downstream gather OOB.
- [x] (2025-12-18) Implement offline `_pack_bits` for expert FP4 bytes in `weight_adapters.py` and use bit-unpack decode in the grouped matmul kernel.
- [x] (2025-12-18) Update tests/bench to feed `_pack_bits`-packed weights and keep reference decode based on the raw (unswizzled) MXFP4 blocks.
- [x] (2025-12-19) Fix in-flight batching crash: pass `expert_usage_stats_host` as a single CPU tensor (do not split into scalars) so host/device dependency ordering matches upstream `grouped_matmul_ragged`.
- [ ] (2025-12-18) Implement full matmul_ogs-style layout handshake: Hopper value reblocking + Hopper scale swizzle + producer->consumer shared-memory swizzle compatibility.
- [ ] (2025-12-18) Add/update microbench to track single-token vs prefill performance (and compare to BF16 ModuleV3 baseline) at realistic MoE shapes + distributions.

## Surprises & Discoveries

- Observation: upstream ModuleV3 uses `grouped_matmul_ragged(...)` for expert GEMMs and expects a host-side `expert_usage_stats_host` tensor.
  Evidence: docstring of `max.nn.kernels.grouped_matmul_ragged` and usage in `max.pipelines.architectures.gpt_oss_module_v3.layers.moe`.
- Root-cause of the `Gather index out of bounds` crash under ShareGPT/in-flight batching: splitting `expert_usage_stats_host` into independent CPU scalars at the Python wrapper boundary can break host/device dependency ordering, causing downstream ragged routing ops to observe inconsistent metadata.
  Fix: pass `expert_usage_stats_host` as a single CPU tensor `[max_M, num_active_experts]` into the custom op and read it inside the Mojo `execute`.
- Confirmed: Triton’s Hopper `matmul_ogs` dequant trick is local and testable — applying `_pack_bits` to 4 bytes and then the bit-unpack mask/shift sequence reproduces BF16-rounded FP4 decode exactly (scale=1). This is the core we will port into Mojo.
- Correctness gotcha (SM90 swizzle handshake): `TensorMapSwizzle.SWIZZLE_128B` is a **hardware address permutation** applied by TMA/WGMMA when *accessing* shared memory. If the producer writes BF16 tiles “plain” (unswizzled), but the consumer reads them with `*_swizzle=SWIZZLE_128B`, the math is wrong.
- Clarification (prevents “close enough” swizzle bugs): `tile_layout_k_major[..., swizzle_mode=SWIZZLE_128B]` / `tile_layout_mn_major[..., SWIZZLE_128B]` select the **canonical core-matrix row width** and satisfy descriptor constraints, but they do **not** magically fix software-produced BF16 tiles. If you decode in software, you must ensure the bytes in shared match the swizzled descriptor view.
- Approved ways to write swizzled shared tiles (pick one and do it exactly):
  - **TMA/CPAsync writers:** use TMA or async-copy tile loaders/writers that know about `TensorMapSwizzle.SWIZZLE_128B` so the producer never does “software swizzle”.
  - **Software swizzle (for decode):** compute the **layout-derived linear element index** and apply the XOR swizzle to that index during the store.
    - Helper: `make_swizzle[BF16, TensorMapSwizzle.SWIZZLE_128B]()` in `max/kernels/src/layout/swizzle.mojo`
    - Helper: `copy_local_to_shared[..., swizzle=make_swizzle[...]](...)` in `max/kernels/src/layout/layout_tensor.mojo`
    - Manual fallback: apply the same swizzle mapping as `copy_local_to_shared` (do not invent a new one; do not swizzle a row-major index if the tensor-core layout is not row-major).
  - Reference patterns (read-only): `max/kernels/test/gpu/linalg/test_tma_wgmma.mojo`, `max/kernels/test/gpu/linalg/test_async_copy_wgmma.mojo`, and `max/kernels/src/linalg/matmul/gpu/sm90/*` (Hopper-specific helpers).
- Performance reality check: `_pack_bits` + bit-unpack alone is not enough; without the full swizzle + TMA/persistent scheduling pattern the custom kernel remains far below BF16 throughput.
  Evidence: `max/examples/custom-models/scripts/bench_mxfp4_grouped_matmul_ragged.py` currently reports ~`0.18–0.22×` BF16 for the isolated grouped matmul on H100.

## Decision Log

- Decision: implement a new custom architecture package `max/examples/custom-models/gpt_oss_mxfp4_v3` instead of continuing in `gpt_oss_mxfp4`.
  Rationale: keep the prior custom MoE WGMMA path isolated; minimize drift by extending upstream ModuleV3 and swapping only the GEMMs.
  Date/Author: 2025-12-18 / Codex
- Decision: pivot away from LUT-based FP4 decode and implement matmul_ogs-style `_pack_bits` + bit-unpack decode.
  Rationale: LUT decode is “correct but slow” and hits the known ~0.25–0.3× BF16 ceiling on H100; matmul_ogs’ packed-bit decode is designed specifically to make dequant cheap enough to hide behind tensor-core math.
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

- New custom architecture package: `max/examples/custom-models/gpt_oss_mxfp4_v3/`
- New Mojo op(s): `max/examples/custom_ops/kernels/` (and imported from `max/examples/custom_ops/kernels/__init__.mojo`)
- Tests/bench: `max/examples/custom-models/tests/` and `max/examples/custom-models/tests/mojo/`

### Precision policy (hard rule)

- FP32 is permitted **only** for register accumulators and scalar epilogue temps.
- Everything stored in memory (global/shared) must be BF16 (or U8/U32 metadata).
- Forbidden: `LayoutTensor[F32, ..., address_space=SHARED]`.

## Plan of Work

### A) Create the `gpt_oss_mxfp4_v3` custom architecture (ModuleV3 override)

Implement a custom architecture package that overrides the upstream name `GptOssForCausalLM_ModuleV3` so it is selected by `--use-module-v3`:

- `max/examples/custom-models/gpt_oss_mxfp4_v3/arch.py`
  - Register `SupportedArchitecture(name="GptOssForCausalLM_ModuleV3", ...)`
  - Use a custom pipeline model that instantiates our custom ModuleV3 model.
- `max/examples/custom-models/gpt_oss_mxfp4_v3/model.py`
  - Subclass `max.pipelines.architectures.gpt_oss_module_v3.model.GptOssModel`
  - Override `load_model()` to build the upstream config, but instantiate **our** `GptOss` ModuleV3 module (same as upstream except MoE).
  - Compile via a helper that injects `custom_extensions=[.../max/examples/custom_ops/kernels]`.
- `max/examples/custom-models/gpt_oss_mxfp4_v3/model_config.py`
  - Re-export upstream `GptOssConfig` (do not fork config derivation).
- `max/examples/custom-models/gpt_oss_mxfp4_v3/weight_adapters.py`
  - Map HF keys to ModuleV3 module parameter names.
  - Keep expert weights as `uint8` blocks/scales (no BF16 expansion).
  - Cast MoE biases to BF16 (or FP32 only if the kernel requires it; prefer BF16 in memory).

### B) Implement `grouped_matmul_ragged_mxfp4_bf16` Mojo op

Hard-path performance requirement: “generic” dequant (LUT + scalar math, or float multiplies) is too slow for H100 and stalls at ~0.25–0.3× BF16 throughput. The design must follow the matmul_ogs recipe below; do not deviate.

#### The matmul_ogs recipe (non‑negotiable)

This section is intentionally repetitive and explicit so we do not drift back into stale ideas.

1) **Correct mental model for MXFP4**
   - MXFP4 weights are **FP4 E2M1 values** (two nibbles per byte) plus a **power-of-two microscale** shared over **groups of 32 values** (scale exponent stored in **E8M0** bytes).
   - Treat scale decode as **bit construction** (BF16 exponent bits), not `exp2()` and not FP32 scalar math in the inner loop.

2) **Offline swizzle/layout is the secret sauce (not arithmetic)**
   - Preprocess expert weights at load time into a kernel-friendly layout so each warp does:
     - contiguous global loads,
     - conflict-free shared writes/reads,
     - tensor-core-friendly fragments.
   - Concretely, implement (or mirror) the local reference layouts under `/tmp/triton/.../hopper_value.py` and `/tmp/triton/.../hopper_scale.py`:
     - **Values:** apply Hopper `_pack_bits` (already done) and then the Hopper “reblocking”/permute that groups 4 mma tiles along K and improves cacheline utilization.
     - **Scales:** implement Hopper scale swizzle that pads/reshapes scales so they align with the warp tiling (depends on `num_warps`).
   - Do not “kind of swizzle”. The layout handshake must match what the kernel expects.

3) **Dequant must be packed-bit operations + packed BF16 math**
   - Decode 4 bytes → 8 BF16 values with a mask/shift sequence and BF16 pair multiplies (the bias-add trick) like the Hopper unpack kernel.
   - Avoid branch ladders, avoid per-nibble float operations, and avoid intermediate FP32 vectors.

4) **The matmul must stay BF16 tensor core dominated**
   - Decode weights to BF16 fragments and feed **WGMMA/HMMA BF16** dot instructions.
   - Accumulate FP32 in registers only; do not create FP32 tiles/tensors.

5) **MoE scheduling is mandatory**
   - Small, ragged per-expert M is normal. Use scheduling to keep SMs busy:
     - deterministic block shapes based on “tokens per expert”,
     - split‑K when it increases parallelism (but keep deterministic),
     - program-ID reordering (swizzle) to improve locality/assignment,
     - persistent scheduling + TMA where applicable.

Keep the upstream-style SM90 producer/consumer pipeline structure, but change the B-path to:

1) Use **offline `_pack_bits`** on the raw FP4 byte stream in the weight adapter.
2) In the producer warpgroup, decode **4 bytes → 8 BF16 values** with a **bit-unpack mask/shift sequence** (matmul_ogs/Hopper) and then apply the E8M0 BF16 scale.
3) Store the decoded BF16 tile into **swizzled shared memory** so consumers can hit the fast WGMMA path.

Reference upstream files (do not modify; follow the pattern):

- `max/kernels/src/linalg/matmul/gpu/sm90/grouped_matmul.mojo` (`grouped_matmul_sm90`)
- `max/kernels/src/linalg/matmul/gpu/sm90/matmul_kernels.mojo` (`HopperMatmulSM90Kernel`)
- `max/kernels/src/linalg/matmul/gpu/sm90/ring_buffer.mojo` (multi-stage pipeline barriers)
- `max/kernels/src/linalg/matmul/gpu/sm90/tile_loader.mojo` (TMA/CPAsync loader patterns)
- `max/kernels/test/gpu/linalg/test_sm90_splitk.mojo` (split‑K scheduling reference)
- `max/kernels/test/gpu/linalg/test_tma_wgmma.mojo` (minimal TMA+WGMMA example)
- `max/kernels/test/gpu/linalg/test_tma_wgmma_with_multicast.mojo` (multicast variant; currently gated upstream)
- `max/kernels/test/gpu/numerics/test_ue8m0_conversion.mojo` (E8M0 helpers + correctness)
- `max/kernels/test/gpu/numerics/test_e2m1_conversion.mojo` (E2M1 helpers + correctness)
- `max/kernels/src/nn/mha_sm90.mojo` (SM90 notes + unresolved TODOs)

#### SWIZZLE_128B handshake (must not drift)

Hopper swizzle is a **descriptor interpretation**, not an automatic transform on raw bytes. If we store into shared “naïvely” (row-major) but build a TMA/WGMMA descriptor with `SWIZZLE_128B`, the WGMMA loads will interpret the bytes with the XOR permutation and the math will be wrong.

Hard rules for correctness (and to keep the kernel on the SM90 fast path):

- **Do not** allocate a row-major `LayoutTensor[..., AddressSpace.SHARED]` and then “flip on” `TensorMapSwizzle.SWIZZLE_128B` in the descriptor.
- Allocate shared tiles with the canonical SM90 layouts:
  - `tile_layout_k_major[..., TensorMapSwizzle.SWIZZLE_128B]` / `tile_layout_mn_major[..., TensorMapSwizzle.SWIZZLE_128B]`
  - Derive WGMMA/TMA descriptors from those layouts (`tile_to_descriptor` / `create_tma_tile`) rather than hand-encoding stride/swizzle bits.
- For **software-produced** shared tiles (our decoded BF16 `B_s`), apply the XOR swizzle on store:
  - Use `copy_local_to_shared[..., swizzle=make_swizzle[dtype, TensorMapSwizzle.SWIZZLE_128B]()]` (preferred; vectorized, compile-time indexing)
  - Or follow the same swizzled-index construction used by SM90 cp.async loaders in `max/kernels/src/linalg/matmul/gpu/sm90/tile_loader.mojo`.
- Enforce alignment for swizzled tiles:
  - BF16 core-matrix swizzle region is **1024B**, so the shared-memory base for swizzled tiles must be aligned accordingly (and each pipeline stage base must preserve that alignment).

Reference (read-only) helpers/patterns:

- `max/kernels/src/layout/swizzle.mojo` (`make_swizzle[dtype, TensorMapSwizzle]`)
- `max/kernels/src/layout/layout_tensor.mojo` (`copy_local_to_shared[...]` with `swizzle=...`)
- `max/kernels/src/linalg/matmul/gpu/sm90/tile_loader.mojo` (how cp.async applies swizzle to dst indices)

Target custom implementation location:

- `max/examples/custom_ops/kernels/grouped_matmul_mxfp4_sm90.mojo` (new, SM90-focused)
- Keep the registered op name: `"mxfp4_grouped_matmul_ragged_bf16"` (so Python graph stays unchanged)

Inputs/outputs (fixed contract):

- `A`: BF16 `[P, K]` (already permuted/grouped by expert segments)
- `W_blocks`: U8 `[E, K/32, N, 16]` (packed FP4; **kernel-prepacked** layout)
- `W_scales`: U8 `[E, K/32, N]` (E8M0 exponent bytes; **kernel-prepacked** layout)
- `expert_start_indices`: U32 `[E+1]` (only first `num_active_experts+1` are meaningful)
- `expert_ids`: I32 `[E]` (only first `num_active_experts` are meaningful)
- `expert_usage_stats_host`: U32 `[2]` on CPU (`max_M`, `num_active_experts`)
- Output `C`: BF16 `[P, N]` (same ordering as `A`)

Notes on weight layout:

- HF checkpoint layout for expert weights is typically `[E, N, K/32, 16]` / `[E, N, K/32]`.
- Our weight adapter **transposes once at load time** into `[E, K/32, N, 16]` / `[E, K/32, N]` so that the SM90 producer can keep `kb` fixed and issue **contiguous** loads across `N`.

Kernel strategy (final architecture; avoid “slow correctness kernel then refactor”):

- Use **producer/consumer warpgroups** and a **multi-stage pipeline** (match upstream defaults: 4 stages, 2 consumer warpgroups).
- Load A tiles with **TMA**; decode MXFP4 weights to BF16 directly into swizzled shared memory before WGMMA consumes B.
  - Optional later optimization (only if profiling shows it helps): stage `W_blocks/W_scales` via TMA/CPAsync into shared (still U8), then decode from shared → BF16 `B_s`.
- Decode rules:
  - FP4 bytes are preprocessed with `_pack_bits` offline.
  - Producer warpgroup decodes FP4 with the Hopper bit-unpack sequence (mask/shift) to BF16 values (no branch ladder).
  - E8M0 exponent decode via exact BF16 bit construction (no `exp2()`), then apply scale in BF16.
  - Keep all intermediate storage BF16/U8; only FP32 in register accumulators.
- Dispatch:
  - Keep deterministic dispatch on `max_M` (from `expert_usage_stats_host[0]`).
  - Default path should be the Hopper WGMMA kernel; add a small‑M kernel family only after profiling proves WGMMA waste dominates single-token generation.

Host-stats contract (prevents “gather index out of bounds” regressions):

- Match upstream `grouped_matmul_ragged` ordering rules by passing `expert_usage_stats_host` as a **single** CPU tensor `[max_M, num_active_experts]`.
- Do **not** split those into two CPU scalar values at the Python wrapper boundary; that can break host/device dependency ordering under in-flight batching and cause downstream gather/index tensors to go OOB.

Upstream-default SM90 block shape to mirror (BF16 baseline; adapt to MXFP4 decode):

- `BM=128`, `BN=256` (or `BN=128` for smaller-N cases), `BK=64`
- `a_swizzle=b_swizzle=TensorMapSwizzle.SWIZZLE_128B`
- `num_pipeline_stages=4`, `num_consumer=2`
- Reference: `max/kernels/src/linalg/matmul/gpu/sm90/grouped_matmul.mojo` (`default_config_sm90`)

Current baseline (implemented first; then tune toward upstream defaults):

- `BM=64`, `BN=128`, `BK=64`
- `a_swizzle=b_swizzle=TensorMapSwizzle.SWIZZLE_128B`
- `num_pipeline_stages=2`, `num_consumer=1`

### C) Swap only the two expert GEMMs in ModuleV3 MoE

Create a custom MoE layer file that is a minimal diff from upstream:

- `max/examples/custom-models/gpt_oss_mxfp4_v3/layers/moe.py`
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

    cd max/examples/custom-models
    pixi run python -c "import max; print(max.__version__)"

### Tests

2) Run python tests:

    cd max/examples/custom-models
    pixi run pytest -q

### Microbench (MoE-only)

3) Run Mojo benchmark (update once new op exists):

    cd max/examples/custom-models
    pixi run mojo run -I "$PIXI_PROJECT_ROOT/../custom_ops" "$PIXI_PROJECT_ROOT/tests/mojo/bench_mxfp4_moe_ops.mojo" --20b --tokens1
    pixi run mojo run -I "$PIXI_PROJECT_ROOT/../custom_ops" "$PIXI_PROJECT_ROOT/tests/mojo/bench_mxfp4_moe_ops.mojo" --20b --tokens64

### End-to-end smoke (ModuleV3)

4) End-to-end smoke (no debug):

    cd max/examples/custom-models
    PROMPT=$(for i in {1..900}; do printf 'hello '; done)
    pixi run python -m max.entrypoints.pipelines generate \
      --custom-architectures ./gpt_oss_mxfp4_v3 \
      --use-module-v3 \
      --model openai/gpt-oss-20b \
      --chat-template ./gpt_oss_mxfp4_v3/templates/gpt-oss-20b.chat_template.jinja \
      --devices gpu \
      --device-memory-utilization 0.90 \
      --max-length 2048 \
      --enable-chunked-prefill \
      --prefill-chunk-size 8192 \
      --no-enable-prefix-caching \
      --max-new-tokens 32 \
      --prompt "$PROMPT"

5) Debugging (prints per MoE call):

    cd max/examples/custom-models
    MXFP4_V3_MOE_DEBUG=1 MODULAR_DEVICE_CONTEXT_SYNC_MODE=true pixi run python -m max.entrypoints.pipelines generate \
      --custom-architectures ./gpt_oss_mxfp4_v3 \
      --use-module-v3 \
      --model openai/gpt-oss-20b \
      --chat-template ./gpt_oss_mxfp4_v3/templates/gpt-oss-20b.chat_template.jinja \
      --devices gpu \
      --max-length 2048 \
      --max-new-tokens 2 \
      --prompt "hello"

6) Quick smoke (short prompt):

    cd max/examples/custom-models
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
  - `rg "LayoutTensor\\[F32" max/examples/custom_ops/kernels/` matches only LOCAL accumulator fragments.
  - `rg -U "address_space\\s*=\\s*AddressSpace\\.SHARED[\\s\\S]*F32" max/examples/custom_ops/kernels/` returns zero matches.

## Artifacts and Notes

- The custom arch must be invoked with `--custom-architectures "$PIXI_PROJECT_ROOT/gpt_oss_mxfp4_v3" --use-module-v3`.
- The chat template for gpt-oss-20b must be passed (chat completions are the coherence check).

## Interfaces and Dependencies

### New python interface (must exist)

In `max/examples/custom-models/gpt_oss_mxfp4_v3/kernels.py` (or equivalent), define a wrapper callable by ModuleV3:

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

In `max/examples/custom_ops/kernels/grouped_matmul_mxfp4_sm90.mojo`:

- `@compiler.register("mxfp4_grouped_matmul_ragged_bf16")`

and `max/examples/custom_ops/kernels/__init__.mojo` must import it:

- `from .grouped_matmul_mxfp4_sm90 import *`

---

Plan revised on 2025-12-18: rewritten to be self-contained and to target the new `gpt_oss_mxfp4_v3` package, keeping upstream ModuleV3 behavior and swapping only the two expert GEMMs.
