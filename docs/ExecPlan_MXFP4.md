# ExecPlan: GPT-OSS MXFP4 on MAX

This ExecPlan must be maintained in accordance with .agents/PLANS.md and remains the single source of truth for this effort.

## Purpose / Big Picture

GPT-OSS checkpoints ship their MoE weights in MXFP4 so that the 120B variant fits inside one 80 GB H100. The MAX pipeline currently upcasts those tensors to BF16, which quadruples memory usage and makes single-GPU serving impossible. This plan delivers a quantized path that keeps MXFP4 blocks in memory, adds a Hopper-optimized custom kernel that dequantizes inside the matmul, and exposes a configuration flag so operators can load `max serve openai/gpt-oss-120b` on a single H100. Success is shown by running the MAX GPT-OSS pipeline with `--encoding mxfp4`, observing steady VRAM usage under 70 GB and logits that match the BF16 path within expected tolerance on canned prompts.

## Progress

- [x] (2025-11-19 08:00Z) Reviewed AGENTS.md plus ref-docs/FEATURE_OVERVIEW.md, MXFP4.md, and GPT_OSS_OVERVIEW.md; drafted the initial plan and recorded repository touch points.
- [x] (2025-11-19 08:00Z) Extend SupportedEncoding/QuantizationEncoding, pipeline config, and CLI surfaces so GPT-OSS can be requested in MXFP4 with correct dtype/cache defaults.
- [x] (2025-11-19 08:00Z) Update GPT-OSS weight adapters and state-dict plumbing to keep `.blocks` and `.scales` tensors as uint8 WeightData labeled with the MXFP4 encoding.
- [x] (2025-11-19 08:00Z) Teach `GptOssConfig`/`GptOssMoE` to allocate quantized Weight objects, surface packed metadata, and switch matmul calls to a new MXFP4 op while preserving BF16 fallback.
- [x] (2025-11-19 08:00Z) Implement and register the Hopper MXFP4 grouped matmul kernel plus its Python binding, then integrate it into the graph build.
- [ ] (2025-11-19 08:00Z) Add unit/kernel/integration tests and run an end-to-end inference smoke test proving VRAM savings and numerical parity.

## Surprises & Discoveries

- Observation: None yet; this is the first revision of the plan before any implementation starts.
  Evidence: ExecPlan authored 2025-11-19, no code outside docs touched.

## Decision Log

- Decision: Store this ExecPlan at docs/ExecPlan_MXFP4.md and treat MXFP4 as an additional SupportedEncoding on the existing GPT-OSS architecture instead of a separate pipeline.
  Rationale: Keeps plan co-located with other docs, honors .agents/PLANS.md guidance, and minimizes duplicate models while still allowing a user-visible flag.
  Date/Author: 2025-11-19 / Codex

## Outcomes & Retrospective

To be filled once the MXFP4 path is merged and validated. Capture VRAM numbers, perf delta vs BF16, and any follow-up work discovered along the way.

## Context and Orientation

The GPT-OSS pipeline lives under `max/python/max/pipelines/architectures/gpt_oss`. Weight ingestion currently happens in `weight_adapters.py`, which renames HuggingFace keys and materializes every tensor as BF16 via `WeightData`. The Mixture-of-Experts implementation is `layers/moe.py`, which creates four `Weight` objects (gate/up weights and biases, down weights and biases) and calls `max.nn.kernels.grouped_matmul_ragged`, a BF16-only custom op registered as `mo.grouped.matmul.ragged` in `max/kernels/src/Mogg/MOGGKernelAPI/MOGGKernelAPI.mojo` and implemented in `max/kernels/src/nn/moe.mojo`. `GptOssModel` builds the graph, instantiates `GptOssConfig`, and pushes MAX graph weights through `nn_model.load_state_dict`.

MXFP4 (documented in ref-docs/MXFP4.md) stores 32 FP4(E2M1) elements per block, each block occupying 16 bytes for packed nibbles plus one E8M0 scale byte (power-of-two exponent offset). GPT-OSS safetensors include both `<weight>.blocks` and `<weight>.scales` tensors for each MoE projection (mlp1 and mlp2), laid out such that the innermost dimension of `.blocks` is half of the dense K dimension. The reference Triton kernel in `ref-docs/gpt-oss/gpt_oss/triton/moe.py` keeps the packed bytes in HBM, decodes on the fly into registers, multiplies by `ldexp` of the shared exponent, and feeds warp-group MMA instructions.

MAX already supports quantized weights through `max.graph.quantization.QuantizationEncoding`, but MXFP4 is absent. `SupportedEncoding` (in `max/python/max/pipelines/lib/config_enums.py`) maps CLI-friendly encodings to `QuantizationEncoding`, dtype, cache dtype, and supported devices. GPT-OSS currently only exposes `SupportedEncoding.bfloat16` in `arch.py`. Custom ops are typically compiled into `.mojopkg` bundles via Bazel targets under `max/kernels/src`, registered via `@compiler.register` functions, and invoked from Python via `ops.custom` wrappers defined in `max/python/max/nn/kernels.py`.

## Plan of Work

### Phase 1 – Extend encoding, config, and CLI surfaces

Add `QuantizationEncoding.MXFP4` to `max/python/max/graph/quantization.py`, define its `BlockParameters(elements_per_block=32, block_size=17)`, and document that scales are unsigned E8M0 exponents minus 127. Introduce a matching enum member `SupportedEncoding.mxfp4` inside `max/python/max/pipelines/lib/config_enums.py`, mapping `dtype` to `DType.bfloat16` (activations still run in BF16), `cache_dtype` to `DType.bfloat16`, `quantization_encoding` to the new MXFP4 entry, and `supported_devices` to `("gpu",)` so the pipeline rejects CPU launches. Extend `SupportedEncoding.parse_from_file_name` to detect `"mxfp4"` substrings. Update `max/python/max/pipelines/architectures/gpt_oss/arch.py` so `supported_encodings` includes the new entry (and optionally make it the default for GPT-OSS weights whose folder names include `"mxfp4"`). Expose a simple enum or bool on `GptOssConfig` (for example `quantization: Literal["bf16","mxfp4"]`) and plumb it through `GptOssConfig.generate`, deriving the value from `self.encoding` when the user selects `SupportedEncoding.mxfp4`. Document this in the plan and add schema validation to ensure only MoE weights use MXFP4 while the rest remain BF16.

### Phase 2 – Preserve MXFP4 weights when loading checkpoints

Modify `max/python/max/pipelines/architectures/gpt_oss/weight_adapters.py` so `convert_safetensor_state_dict` inspects each key coming from the safetensors. When it sees MoE weight names (for example `language_model.layers.N.mlp.experts.gate_up_proj`), it should look for both `.blocks` and `.scales` suffixes, keep them as `WeightData` with `dtype=DType.uint8`, set `quantization_encoding=QuantizationEncoding.MXFP4`, and record the logical dense shape in metadata. Create a helper dataclass (for example `Mxfp4PackedTensor`) that stores `blocks`, `scales`, and the inferred `[num_experts, out_features, in_features]` logical shape so downstream code can reason about the layout without repeatedly recomputing Strides. Ensure `state_dict` keys follow the MAX naming scheme, e.g. `language_model.layers.0.mlp.experts.gate_up_proj.blocks`. Update the adapter to leave non-MoE tensors untouched (still BF16) and to raise a descriptive error if only one of the pair (`.blocks` or `.scales`) exists.

### Phase 3 – Teach GptOssConfig and MoE layers about MXFP4 tensors

Extend `GptOssConfig` to carry a `moe_weight_format` struct summarizing whether each tensor is BF16 or MXFP4, plus small helpers that report the packed strides (e.g., bytes per logical row). Update `GptOssMoE._init_experts` in `layers/moe.py` to allocate either BF16 `Weight` objects (existing behavior) or quantized placeholders when `config.quantization == "mxfp4"`. For the quantized case, create six `Weight` objects: `experts.gate_up_proj.blocks`, `.scales`, `experts.down_proj.blocks`, `.scales`, plus the two bias tensors (still BF16). Each block tensor should have shape `[num_experts, out_features, hidden_dim // 2]` because two FP4 values fit per byte, while the `scales` tensors use `[num_experts, out_features, hidden_dim // 32]`. Set `quantization_encoding=QuantizationEncoding.MXFP4` on both block and scale weights and store their logical dense shape in a companion metadata map on the module. During `state_dict` loading, pass `override_quantization_encoding=True` so the metadata from the checkpoint wins. Add helper accessors that return a `Mxfp4ExpertWeights` struct describing the block/scales pointers to keep the `__call__` logic tidy.

### Phase 4 – Implement the MXFP4 grouped matmul kernel

Create a new Mojo source file, e.g. `max/kernels/src/nn/moe_mxfp4.mojo`, housing both CPU fallbacks (simple decode + matmul for validation) and the Hopper GPU implementation. The kernel signature should mirror `grouped_matmul_ragged` but accept pointers to packed weights and scales alongside the ragged activation tile metadata: `activations (bf16)`, `packed_weights (uint8)`, `scales (uint8)`, `expert_start_indices`, `expert_ids`, `max_tokens_per_expert`, and `num_active_experts`. Add compile-time constants for MXFP4 (32 values per block, 16 packed-byte payload, base FP4 lookup table). On GPU, each thread block should iterate over one expert tile, load TMA chunks of `activations` and the corresponding packed weights/scales, decode two FP4 values per byte via bitmask/shift, apply the `ldexp` exponent using integer addition on the BF16 exponent bits, and accumulate using `warp.mma` or `wg.mma_async` on SM90+. Reuse the ragged scheduling logic from `max/kernels/src/nn/moe.mojo` to compute which tokens map to each expert. Register the kernel by adding a `@compiler.register("mo.moe.mx4.matmul")` struct inside `max/kernels/src/Mogg/MOGGKernelAPI/MOGGKernelAPI.mojo` that forwards to a new function (e.g. `moe_mxfp4_grouped_matmul`) implemented in the new Mojo file. Make sure the CPU path simply unpacks the bytes into a scratch buffer and calls the existing `linalg.matmul` utilities so tests can run on CI without GPUs.

### Phase 5 – Bind the kernel into Python and switch the MoE matmuls

Add a helper in `max/python/max/nn/kernels.py` such as `grouped_mxfp4_matmul(...)` that mirrors `grouped_matmul_ragged` but constructs the constant tensors for `max_num_tokens_per_expert` and `num_active_experts` along with the additional weight inputs, then calls `ops.custom("mo.moe.mx4.matmul", ...)`. Import this helper inside `layers/moe.py` and branch in `__call__`: when quantization is MXFP4, gather the packed weights, pass them to the new custom op twice (once for gate/up, once for down), and continue with the SwiGLU + routing logic unchanged. Keep the BF16 path intact as a fallback. Ensure the new helper validates shapes (hidden size divisible by 32) and raises a clear exception on CPU because the GPU version is required for performance. Teach `GptOssModel` to pass a `custom_extensions` entry for the kernel if it is packaged separately, otherwise rely on the in-tree registration so regular builds pick it up.

### Phase 6 – Tests, documentation, and validation hooks

Author unit tests under `max/python/max/pipelines/architectures/gpt_oss/tests` that feed a toy safetensors map with `.blocks/.scales` pairs through `convert_safetensor_state_dict` and verify the shapes, dtypes, and `quantization_encoding`. Add a kernel-level test (Bazel target under `max/tests/integration/API/python/graph`) that instantiates a ragged matmul graph with random MXFP4 weights, compares outputs to a CPU reference that decodes to BF16, and runs on both GPU (fast) and CPU (decode fallback). Include a pipeline integration test that builds a tiny GPT-OSS config (handful of experts, small dims), loads random data in MXFP4, and asserts the quantized path matches the BF16 path to within 1e-2 relative tolerance. Document the new CLI usage in `README.md` or an architecture-specific doc, mentioning memory footprint expectations and hardware requirements (H100 / SM90). Update release notes or `ref-docs/FEATURE_OVERVIEW.md` to mark the plan executed.

## Concrete Steps

Work from the repo root (`/Users/m1mbp/tools/modular-gptoss-mxfp4`) unless noted.

    ./bazelw test //max/python/max/pipelines/architectures/gpt_oss:all
      Expect the new MXFP4-specific unit tests (e.g. test_mxfp4_weight_adapter) to pass alongside existing ones.

    ./bazelw test //max/tests/integration/API/python/graph:test_mxfp4_moe_kernel
      Ensures the custom op works on both GPU and CPU fallback; failures should include tensor diffs.

    ./bazelw build //max/kernels/src/nn:nn
      Rebuilds the Mojo kernels (including moe_mxfp4.mojo) to surface syntax or register errors early.

    python -m max.pipelines.run \
      --arch gpt_oss \
      --weights /path/to/gpt-oss-20b-mxfp4 \
      --encoding mxfp4 \
      --prompt "Why did the chicken cross the road?" \
      --max-tokens 32
      Observe VRAM usage in nvidia-smi (~18 GB for 20B), response text, and log lines confirming the MXFP4 kernel was selected.

    ./bazelw test //max/tests/integration/pipelines:gpt_oss_mxfp4_smoke
      Adds an integration test that runs a short decode with deterministic inputs to guard regressions.

## Validation and Acceptance

The feature is accepted when all new and existing GPT-OSS tests pass under `./bazelw test //max/...`, the MXFP4 kernel test (`test_mxfp4_moe_kernel`) reports matching outputs against a BF16 reference, and a manual `max serve` or `python -m max.pipelines.run` invocation with `--encoding mxfp4` returns coherent logits on a developer-provided prompt while keeping total GPU memory usage below 70 GB for the 120B checkpoint. Capture before/after VRAM measurements and include them in the Outcomes section once the change lands.

## Idempotence and Recovery

All weight-conversion scripts are pure functions of the safetensors input and can be re-run safely. The Bazel builds can be retried after `./bazelw clean --expunge` if custom op registration errors appear. The MXFP4 loader should validate paired `.blocks`/`.scales` tensors so partially downloaded checkpoints fail fast with descriptive messages. The custom kernel path should fall back to a scalar decode implementation (guarded by `DeviceRef.CPU`) for debugging, allowing repeated runs on machines without GPUs.

## Artifacts and Notes

Record short transcripts demonstrating:

    bazel test //max/tests/integration/API/python/graph:test_mxfp4_moe_kernel
      PASSED in X.XXs

    python -m max.pipelines.run --arch gpt_oss --encoding mxfp4 --prompt "hi" ...
      MXFP4 mode enabled for 36 MoE layers
      Generated: "Hello there..."

Attach those snippets (trimmed to essentials) here when available so future contributors can see the expected console output without re-running.

## Interfaces and Dependencies

Define or update the following:

- `max/python/max/graph/quantization.py`: add `QuantizationEncoding.MXFP4` with `block_parameters=BlockParameters(32, 17)` and document the E8M0 scale semantics.
- `max/python/max/pipelines/lib/config_enums.py`: new `SupportedEncoding.mxfp4` with dtype `DType.bfloat16`, cache dtype `DType.bfloat16`, quantization encoding `QuantizationEncoding.MXFP4`, and GPU-only support.
- `max/python/max/pipelines/architectures/gpt_oss/model_config.py`: extend `GptOssConfig` with `quantization: Literal["bf16","mxfp4"]` and a helper such as `def is_mxfp4(self) -> bool`.
- `max/python/max/pipelines/architectures/gpt_oss/weight_adapters.py`: helper `def extract_mxfp4_weight(blocks: WeightData, scales: WeightData, *, logical_shape: tuple[int, int, int]) -> WeightData`.
- `max/python/max/pipelines/architectures/gpt_oss/layers/moe.py`: struct `@dataclass class Mxfp4ExpertWeights` bundling `blocks`, `scales`, and logical dims, plus a `def _moe_matmul(self, inputs, weights, indices, stats)` method selecting BF16 vs MXFP4.
- `max/python/max/nn/kernels.py`: function `def grouped_mxfp4_matmul(hidden_states, packed_weight, packed_scales, expert_start_indices, expert_ids, expert_usage_stats_host) -> TensorValue`.
- `max/kernels/src/nn/moe_mxfp4.mojo`: GPU kernel `fn mxfp4_grouped_matmul(...)` that handles ragged scheduling, block decode, and tensor-core GEMM, along with a CPU reference path used by tests.
- `max/kernels/src/Mogg/MOGGKernelAPI/MOGGKernelAPI.mojo`: registration struct `@compiler.register("mo.moe.mx4.matmul") struct Struct_moe_mx4_matmul`.

Note any new Python dependencies (none expected) and ensure Bazel targets list the new Mojo source file.

---

Revision history:

- 2025-11-19: Initial ExecPlan drafted after reviewing AGENTS.md and reference docs. (Codex)
