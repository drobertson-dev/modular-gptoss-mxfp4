---
title: "max warm-cache CLI Command Reference"
description: "Preload and compile models to optimize initialization time and warm up the Hugging Face cache."
---

# max warm-cache

Preloads and compiles the model to optimize initialization time by:

- Pre-compiling models before deployment
- Warming up the Hugging Face cache

This command is useful to run before serving a model.

For example:

```bash
max warm-cache \
  --model google/gemma-3-12b-it
```

The Modular Executable Format (MEF) is platform independent, but the serialized cache (MEF files) produced during
compilation is platform-dependent. This is because:

- Platform-dependent optimizations happen during compilation.
- Fallback operations assume a particular runtime environment.

Weight transformations and hashing during MEF caching can impact performance. While efforts to improve this through
weight externalization are ongoing, compiled MEF files remain platform-specific and are not generally portable.

## Usage

```shell
max warm-cache [OPTIONS]
```

## Options

- ###
  `--allow-safetensors-weights-fp32-bf6-bidirectional-cast, --no-allow-safetensors-weights-fp32-bf6-bidirectional-cast`

Whether to allow automatic float32 to/from bfloat16 safetensors weight type casting, if needed. Currently only supported
in Llama3 models.

- ### `--cache-strategy <cache_strategy>`

The cache strategy to use. This defaults to model_default, which selects the default strategy for the requested
architecture. You can also force a specific strategy: continuous or paged.

**Options:**

KVCacheStrategy.MODEL_DEFAULT | KVCacheStrategy.PAGED

- ### `--ce-delay-ms <ce_delay_ms>`

Duration of scheduler sleep prior to starting a prefill batch. Experimental for the TTS scheduler.

- ### `--chat-template <chat_template>`

Optional custom chat template to override the one shipped with the HuggingFace model config. If a path is provided, the
file is read during config resolution and the content stored as a string. If None, the model’s default chat template is
used.

- ### `--config-file <config_file>`
- ### `--custom-architectures <custom_architectures>`

Custom architecture implementations to register. Each input can either be a raw module name or an import path followed
by a colon and the module name. Each module must expose an ARCHITECTURES list of architectures to register.

- ### `--data-parallel-degree <data_parallel_degree>`

Data-parallelism parameter. The degree to which the model is replicated is dependent on the model type.

- ### `--defer-resolve, --no-defer-resolve`

Whether to defer resolving the pipeline config.

- ### `--device-memory-utilization <device_memory_utilization>`

The fraction of available device memory that the process should consume. This informs the KVCache workspace size:
kv_cache_workspace = (total_free_memory * device_memory_utilization) - model_weights_size.

- ### `--devices <devices>`

Whether to run the model on CPU (–devices=cpu), GPU (–devices=gpu) or a list of GPUs (–devices=gpu:0,1) etc. An ID value
can be provided optionally to indicate the device ID to target. If not provided, the model will run on the first
available GPU (–devices=gpu), or CPU if no GPUs are available (–devices=cpu).

- ###
  `--draft-allow-safetensors-weights-fp32-bf6-bidirectional-cast, --no-draft-allow-safetensors-weights-fp32-bf6-bidirectional-cast`

Whether to allow automatic float32 to/from bfloat16 safetensors weight type casting, if needed. Currently only supported
in Llama3 models.

- ### `--draft-config-file <draft_config_file>`
- ### `--draft-data-parallel-degree <draft_data_parallel_degree>`

Data-parallelism parameter. The degree to which the model is replicated is dependent on the model type.

- ### `--draft-devices <draft_devices>`

Whether to run the model on CPU (–devices=cpu), GPU (–devices=gpu) or a list of GPUs (–devices=gpu:0,1) etc. An ID value
can be provided optionally to indicate the device ID to target. If not provided, the model will run on the first
available GPU (–devices=gpu), or CPU if no GPUs are available (–devices=cpu).

- ### `--draft-force-download, --no-draft-force-download`

Whether to force download a given file if it’s already present in the local cache.

- ### `--draft-huggingface-model-revision <draft_huggingface_model_revision>`

Branch or Git revision of Hugging Face model repository to use.

- ### `--draft-huggingface-weight-revision <draft_huggingface_weight_revision>`

Branch or Git revision of Hugging Face model repository to use.

- ### `--draft-model-path <draft_model_path>`

repo_id of a Hugging Face model repository to use.

- ### `--draft-quantization-encoding <draft_quantization_encoding>`

Weight encoding type.

**Options:**

SupportedEncoding.float32 | SupportedEncoding.bfloat16 | SupportedEncoding.q4_k | SupportedEncoding.q4_0 |
SupportedEncoding.q6_k | SupportedEncoding.float8_e4m3fn | SupportedEncoding.float4_e2m1fnx2 | SupportedEncoding.gptq

- ### `--draft-rope-type <draft_rope_type>`

Force using a specific rope type: none, normal, or neox. Only matters for GGUF weights.

**Options:**

RopeType.none | RopeType.normal | RopeType.neox | RopeType.longrope | RopeType.yarn

- ### `--draft-section-name <draft_section_name>`
- ### `--draft-served-model-name <draft_served_model_name>`

Optional override for client-facing model name. Defaults to model_path.

- ### `--draft-trust-remote-code, --no-draft-trust-remote-code`

Whether or not to allow for custom modelling files on Hugging Face.

- ### `--draft-use-subgraphs, --no-draft-use-subgraphs`

Whether to use subgraphs for the model. This can significantly reduce compile time, especially for large models with
identical blocks. Default is true.

- ### `--draft-vision-config-overrides <draft_vision_config_overrides>`

Model-specific vision configuration overrides. For example, for InternVL: {“max_dynamic_patch”: 24}.

- ### `--draft-weight-path <draft_weight_path>`

Optional path or url of the model weights to use.

- ### `--enable-chunked-prefill, --no-enable-chunked-prefill`

Enable chunked prefill to split context encoding requests into multiple chunks based on max_batch_input_tokens.

- ### `--enable-echo, --no-enable-echo`

Whether the model should be built with echo capabilities.

- ### `--enable-in-flight-batching, --no-enable-in-flight-batching`

When enabled, prioritizes token generation by batching it with context encoding requests.

- ### `--enable-kvcache-swapping-to-host, --no-enable-kvcache-swapping-to-host`

Whether to swap paged KVCache blocks to host memory when device blocks are evicted.

- ### `--enable-lora, --no-enable-lora`

Enables LoRA on the server.

- ### `--enable-min-tokens, --no-enable-min-tokens`

Whether to enable min_tokens, which blocks the model from generating stopping tokens before the min_tokens count is
reached.

- ### `--enable-overlap-scheduler, --no-enable-overlap-scheduler`

Whether to enable the overlap scheduler. This feature allows the scheduler to run alongside GPU execution. This helps
improve GPU utilization. This is an experimental feature which may crash and burn.

- ### `--enable-penalties, --no-enable-penalties`

Whether to apply frequency and presence penalties to the model’s output.

- ### `--enable-prefix-caching, --no-enable-prefix-caching`

Whether to enable prefix caching for the paged KVCache.

- ### `--enable-prioritize-first-decode, --no-enable-prioritize-first-decode`

When enabled, the scheduler always runs a TG batch immediately after a CE batch with the same requests. This may reduce
time-to-first-chunk latency. Experimental for the TTS scheduler.

- ### `--enable-structured-output, --no-enable-structured-output`

Enable structured generation/guided decoding for the server. This allows the user to pass a json schema in the
response_format field, which the LLM will adhere to.

- ### `--enable-variable-logits, --no-enable-variable-logits`

Enable the sampling graph to accept a ragged tensor of different sequences as inputs, along with their associated
logit_offsets. This is needed to produce additional logits for echo and speculative decoding purposes.

- ### `--ep-size <ep_size>`

The expert parallelism size. Needs to be 1 (no expert parallelism) or the total number of GPUs across nodes.

- ### `--execute-empty-batches, --no-execute-empty-batches`

Whether the scheduler should execute empty batches.

- ### `--force, --no-force`

Skip validation of user provided flags against the architecture’s required arguments.

- ### `--force-download, --no-force-download`

Whether to force download a given file if it’s already present in the local cache.

- ### `--gpu-profiling <gpu_profiling>`

Whether to enable GPU profiling of the model.

**Options:**

GPUProfilingMode.OFF | GPUProfilingMode.ON | GPUProfilingMode.DETAILED

- ### `--host-kvcache-swap-space-gb <host_kvcache_swap_space_gb>`

The amount of host memory to use for the host KVCache in GiB. This space is only allocated when kvcache_swapping_to_host
is enabled.

- ### `--huggingface-model-revision <huggingface_model_revision>`

Branch or Git revision of Hugging Face model repository to use.

- ### `--huggingface-weight-revision <huggingface_weight_revision>`

Branch or Git revision of Hugging Face model repository to use.

- ### `--kv-cache-page-size <kv_cache_page_size>`

The number of tokens in a single page in the paged KVCache.

- ### `--kvcache-ce-watermark <kvcache_ce_watermark>`

Projected cache usage threshold for scheduling CE requests, considering current and incoming requests. CE is scheduled
if either projected usage stays below this threshold or no active requests exist. Higher values can cause more
preemptions.

- ### `--lora-paths <lora_paths>`

List of statically defined LoRA paths.

- ### `--max-batch-input-tokens <max_batch_input_tokens>`

The target number of un-encoded tokens to include in each batch. This value is used for chunked prefill and memory
estimation.

- ### `--max-batch-size <max_batch_size>`

Maximum batch size to execute with the model. When not specified (None), this value is determined dynamically. For
server launches, set this higher based on server capacity.

- ### `--max-batch-total-tokens <max_batch_total_tokens>`

Ensures that the sum of the context length in a batch does not exceed max_batch_total_tokens. If None, the sum is not
limited.

- ### `--max-length <max_length>`

Maximum sequence length of the model.

- ### `--max-lora-rank <max_lora_rank>`

Maximum rank of all possible LoRAs.

- ### `--max-num-loras <max_num_loras>`

The maximum number of active LoRAs in a batch. This controls how many LoRA adapters can be active simultaneously during
inference. Lower values reduce memory usage but limit concurrent adapter usage.

- ### `--max-num-steps <max_num_steps>`

The number of steps to run for multi-step scheduling. -1 specifies a default value based on configuration and platform.
Ignored for models which are not auto-regressive (e.g. embedding models).

- ### `--max-queue-size-tg <max_queue_size_tg>`

Maximum number of requests in decode queue. By default, this is max_batch_size.

- ### `--min-batch-size-tg <min_batch_size_tg>`

Soft floor on the decode batch size. If the TG batch size is larger, the scheduler continues TG batches; if it falls
below, the scheduler prioritizes CE. This is not a strict minimum. By default, this is max_queue_size_tg. Experimental
for the TTS scheduler.

- ### `--model-path <model_path>`

repo_id of a Hugging Face model repository to use.

- ### `--num-speculative-tokens <num_speculative_tokens>`

The number of speculative tokens.

- ### `--pipeline-role <pipeline_role>`

Whether the pipeline should serve both a prefill or decode role or both.

**Options:**

PipelineRole.PrefillAndDecode | PipelineRole.PrefillOnly | PipelineRole.DecodeOnly

- ### `--pool-embeddings, --no-pool-embeddings`

Whether to pool embedding outputs.

- ### `--quantization-encoding <quantization_encoding>`

Weight encoding type.

**Options:**

SupportedEncoding.float32 | SupportedEncoding.bfloat16 | SupportedEncoding.q4_k | SupportedEncoding.q4_0 |
SupportedEncoding.q6_k | SupportedEncoding.float8_e4m3fn | SupportedEncoding.float4_e2m1fnx2 | SupportedEncoding.gptq

- ### `--rope-type <rope_type>`

Force using a specific rope type: none, normal, or neox. Only matters for GGUF weights.

**Options:**

RopeType.none | RopeType.normal | RopeType.neox | RopeType.longrope | RopeType.yarn

- ### `--section-name <section_name>`
- ### `--served-model-name <served_model_name>`

Optional override for client-facing model name. Defaults to model_path.

- ### `--speculative-method <speculative_method>`

The speculative decoding method to use.

**Options:**

SpeculativeMethod.STANDALONE | SpeculativeMethod.EAGLE

- ### `--target <target>`

Target API and architecture to compile for (e.g., cuda, cuda:sm_90, hip:gfx942). When specified, uses virtual devices
for compilation without requiring physical hardware.

- ### `--trust-remote-code, --no-trust-remote-code`

Whether or not to allow for custom modelling files on Hugging Face.

- ### `--use-experimental-kernels <use_experimental_kernels>`

Enables using experimental mojo kernels with max serve. The kernels could be unstable or incorrect.

- ### `--use-legacy-module, --no-use-legacy-module`

Whether to use the legacy Module architecture (default=True for backward compatibility). Set to False to use the new
Module-based architecture when available.

- ### `--use-subgraphs, --no-use-subgraphs`

Whether to use subgraphs for the model. This can significantly reduce compile time, especially for large models with
identical blocks. Default is true.

- ### `--use-vendor-blas <use_vendor_blas>`

Enables using vendor BLAS libraries (cublas/hipblas/etc) with max serve. Currently, this just replaces matmul calls.

- ### `--vision-config-overrides <vision_config_overrides>`

Model-specific vision configuration overrides. For example, for InternVL: {“max_dynamic_patch”: 24}.

- ### `--weight-path <weight_path>`

Optional path or url of the model weights to use.

- ### `--zmq-endpoint-base <zmq_endpoint_base>`

Prefix for ZMQ endpoints used for IPC. This ensures unique endpoints across MAX Serve instances on the same host.
Example: lora_request_zmq_endpoint = f”{zmq_endpoint_base}-lora_request”.
