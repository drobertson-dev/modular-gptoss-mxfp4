---
title: "MAX CLI Reference: MAX serve Command"
description: "Documentation for the max serve command to launch OpenAI-compatible model servers with MAX."
---

# MAX CLI Reference: MAX serve Command

Documentation for the max serve command to launch OpenAI-compatible model servers with MAX.

Launches a model server with an OpenAI-compatible endpoint. Just specify the model as a Hugging Face model ID or a local path.

For example:

```bash
max serve \
  --model google/gemma-3-12b-it \
  --devices gpu:0 \
  --max-batch-size 8 \
  --device-memory-utilization 0.9
```

For details about the endpoint APIs provided by the server, see the MAX REST API reference.

The `max` CLI also supports loading custom model architectures through the `--custom-architectures` flag. This allows you to extend MAX's capabilities with your own model implementations:

```bash
max serve \
  --model google/gemma-3-12b-it \
  --custom-architectures path/to/module1:module1 \
  --custom-architectures path/to/module2:module2
```

### Custom architectures

The `--custom-architectures` flag allows you to load custom pipeline architectures from your own Python modules. You can set the `ARCHITECTURES` variable containing the architecture definitions. Each entry in `--custom-architectures` can be specified in two formats:

- A raw module name; for example: `my_module`.
- An import path followed by a colon and the module name; for example: `folder/path/to/import:my_module`.

The `ARCHITECTURES` variable in your module should be a list of implementations that conform to the SupportedArchitecture interface. These will be registered with the MAX pipeline registry automatically.

### Quantization encoding

When using GGUF models, quantization encoding formats are automatically detected. If no `--quantization-encoding` is specified, MAX Serve automatically detects and uses the first encoding option from the repository. If quantization encoding is provided, it must align with the available encoding options in the repository.

If the repository contains multiple quantization formats, specify which encoding type you want to use with the `--quantization-encoding` parameter.

## Usage

```shell
max serve [OPTIONS]
```

## Options

- ### `--allow-safetensors-weights-fp32-bf6-bidirectional-cast, --no-allow-safetensors-weights-fp32-bf6-bidirectional-cast`

Whether to allow automatic float32 to/from bfloat16 safetensors weight type casting, if needed. Currently only supported in Llama3 models.

- ### `--cache-strategy `

The cache strategy to use. This defaults to model_default, which selects the default strategy for the requested architecture. You can also force a specific strategy: continuous or paged.
**Options:** KVCacheStrategy.MODEL_DEFAULT | KVCacheStrategy.PAGED

- ### `--ce-delay-ms `

Duration of scheduler sleep prior to starting a prefill batch. Experimental for the TTS scheduler.

- ### `--chat-template `

Optional custom chat template to override the one shipped with the HuggingFace model config. If a path is provided, the file is read during config resolution and the content stored as a string. If None, the model's default chat template is used.

- ### `--config-file `

Path to a configuration file.

- ### `--custom-architectures `

Custom architecture implementations to register. Each input can either be a raw module name or an import path followed by a colon and the module name. Each module must expose an ARCHITECTURES list of architectures to register.

- ### `--data-parallel-degree `

Data-parallelism parameter. The degree to which the model is replicated is dependent on the model type.

- ### `--defer-resolve, --no-defer-resolve`

Whether to defer resolving the pipeline config.

- ### `--device-memory-utilization `

The fraction of available device memory that the process should consume. This informs the KVCache workspace size: kv_cache_workspace = (total_free_memory * device_memory_utilization) - model_weights_size.

- ### `--devices `

Whether to run the model on CPU (-devices=cpu), GPU (-devices=gpu) or a list of GPUs (-devices=gpu:0,1) etc. An ID value can be provided optionally to indicate the device ID to target. If not provided, the model will run on the first available GPU (-devices=gpu), or CPU if no GPUs are available (-devices=cpu).

- ### `--draft-allow-safetensors-weights-fp32-bf6-bidirectional-cast, --no-draft-allow-safetensors-weights-fp32-bf6-bidirectional-cast`

Whether to allow automatic float32 to/from bfloat16 safetensors weight type casting, if needed. Currently only supported in Llama3 models.

- ### `--draft-config-file `

Path to a configuration file for the draft model.

- ### `--draft-data-parallel-degree `

Data-parallelism parameter for the draft model.

- ### `--draft-devices `

Devices to use for the draft model.

- ### `--draft-force-download, --no-draft-force-download`

Whether to force download a given file if it's already present in the local cache.

- ### `--draft-huggingface-model-revision `

Branch or Git revision of Hugging Face model repository to use for the draft model.

- ### `--draft-huggingface-weight-revision `

Branch or Git revision of Hugging Face model repository to use for the draft model weights.

- ### `--draft-model-path `

repo_id of a Hugging Face model repository to use for the draft model.

- ### `--draft-quantization-encoding `

Weight encoding type for the draft model.
**Options:** SupportedEncoding.float32 | SupportedEncoding.bfloat16 | SupportedEncoding.q4_k | SupportedEncoding.q4_0 | SupportedEncoding.q6_k | SupportedEncoding.float8_e4m3fn | SupportedEncoding.float4_e2m1fnx2 | SupportedEncoding.gptq

- ### `--draft-rope-type `

Force using a specific rope type for the draft model: none, normal, or neox. Only matters for GGUF weights.
**Options:** RopeType.none | RopeType.normal | RopeType.neox | RopeType.longrope | RopeType.yarn

- ### `--draft-section-name `

Section name in the config file for the draft model.

- ### `--draft-served-model-name `

Optional override for client-facing draft model name. Defaults to model_path.

- ### `--draft-trust-remote-code, --no-draft-trust-remote-code`

Whether or not to allow for custom modelling files on Hugging Face for the draft model.

- ### `--draft-use-subgraphs, --no-draft-use-subgraphs`

Whether to use subgraphs for the draft model. This can significantly reduce compile time. Default is true.

- ### `--draft-vision-config-overrides `

Model-specific vision configuration overrides for the draft model.

- ### `--draft-weight-path `

Optional path or url of the draft model weights to use.

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

Whether to enable min_tokens, which blocks the model from generating stopping tokens before the min_tokens count is reached.

- ### `--enable-overlap-scheduler, --no-enable-overlap-scheduler`

Whether to enable the overlap scheduler. This feature allows the scheduler to run alongside GPU execution. This is an experimental feature.

- ### `--enable-penalties, --no-enable-penalties`

Whether to apply frequency and presence penalties to the model's output.

- ### `--enable-prefix-caching, --no-enable-prefix-caching`

Whether to enable prefix caching for the paged KVCache.

- ### `--enable-prioritize-first-decode, --no-enable-prioritize-first-decode`

When enabled, the scheduler always runs a TG batch immediately after a CE batch with the same requests. Experimental for the TTS scheduler.

- ### `--enable-structured-output, --no-enable-structured-output`

Enable structured generation/guided decoding for the server. This allows the user to pass a json schema in the response_format field.

- ### `--enable-variable-logits, --no-enable-variable-logits`

Enable the sampling graph to accept a ragged tensor of different sequences as inputs. Needed for echo and speculative decoding.

- ### `--ep-size `

The expert parallelism size. Needs to be 1 or the total number of GPUs across nodes.

- ### `--execute-empty-batches, --no-execute-empty-batches`

Whether the scheduler should execute empty batches.

- ### `--force, --no-force`

Skip validation of user provided flags against the architecture's required arguments.

- ### `--force-download, --no-force-download`

Whether to force download a given file if it's already present in the local cache.

- ### `--gpu-profiling `

Whether to enable GPU profiling of the model.
**Options:** GPUProfilingMode.OFF | GPUProfilingMode.ON | GPUProfilingMode.DETAILED

- ### `--headless`

Run only the dispatcher service and model worker without the API server.
**Default:** `False`

- ### `--host-kvcache-swap-space-gb `

The amount of host memory to use for the host KVCache in GiB.

- ### `--huggingface-model-revision `

Branch or Git revision of Hugging Face model repository to use.

- ### `--huggingface-weight-revision `

Branch or Git revision of Hugging Face model repository to use for weights.

- ### `--kv-cache-page-size `

The number of tokens in a single page in the paged KVCache.

- ### `--kvcache-ce-watermark `

Projected cache usage threshold for scheduling CE requests. Higher values can cause more preemptions.

- ### `--log-prefix `

Optional prefix to add to all log messages for this server instance.

- ### `--lora-paths `

List of statically defined LoRA paths.

- ### `--max-batch-input-tokens `

The target number of un-encoded tokens to include in each batch. Used for chunked prefill and memory estimation.

- ### `--max-batch-size `

Maximum batch size to execute with the model. Determined dynamically if not specified.

- ### `--max-batch-total-tokens `

Ensures that the sum of the context length in a batch does not exceed this value.

- ### `--max-length `

Maximum sequence length of the model.

- ### `--max-lora-rank `

Maximum rank of all possible LoRAs.

- ### `--max-num-loras `

The maximum number of active LoRAs in a batch.

- ### `--max-num-steps `

The number of steps to run for multi-step scheduling.

- ### `--max-queue-size-tg `

Maximum number of requests in decode queue. By default, this is max_batch_size.

- ### `--min-batch-size-tg `

Soft floor on the decode batch size. Experimental for the TTS scheduler.

- ### `--model-path `

repo_id of a Hugging Face model repository to use.

- ### `--num-speculative-tokens `

The number of speculative tokens.

- ### `--pipeline-role `

Whether the pipeline should serve both a prefill or decode role or both.
**Options:** PipelineRole.PrefillAndDecode | PipelineRole.PrefillOnly | PipelineRole.DecodeOnly

- ### `--pool-embeddings, --no-pool-embeddings`

Whether to pool embedding outputs.

- ### `--port `

Port to run the server on.

- ### `--pretty-print-config`

Pretty Print Entire Config.

- ### `--profile-serve`

Whether to enable pyinstrument profiling on the serving endpoint.
**Default:** `False`

- ### `--quantization-encoding `

Weight encoding type.
**Options:** SupportedEncoding.float32 | SupportedEncoding.bfloat16 | SupportedEncoding.q4_k | SupportedEncoding.q4_0 | SupportedEncoding.q6_k | SupportedEncoding.float8_e4m3fn | SupportedEncoding.float4_e2m1fnx2 | SupportedEncoding.gptq

- ### `--rope-type `

Force using a specific rope type: none, normal, or neox. Only matters for GGUF weights.
**Options:** RopeType.none | RopeType.normal | RopeType.neox | RopeType.longrope | RopeType.yarn

- ### `--section-name `

Section name in the config file.

- ### `--served-model-name `

Optional override for client-facing model name. Defaults to model_path.

- ### `--sim-failure `

Simulate fake-perf with failure percentage.

- ### `--speculative-method `

The speculative decoding method to use.
**Options:** SpeculativeMethod.STANDALONE | SpeculativeMethod.EAGLE

- ### `--task `

The task to run.

- ### `--task-arg `

Task-specific arguments to pass to the underlying model.

- ### `--trust-remote-code, --no-trust-remote-code`

Whether or not to allow for custom modelling files on Hugging Face.

- ### `--use-experimental-kernels `

Enables using experimental mojo kernels. Could be unstable or incorrect.

- ### `--use-legacy-module, --no-use-legacy-module`

Whether to use the legacy Module architecture (default=True). Set to False for the new Module-based architecture.

- ### `--use-subgraphs, --no-use-subgraphs`

Whether to use subgraphs for the model. Reduces compile time for large models. Default is true.

- ### `--use-vendor-blas `

Enables using vendor BLAS libraries (cublas/hipblas/etc).

- ### `--vision-config-overrides `

Model-specific vision configuration overrides.

- ### `--weight-path `

Optional path or url of the model weights to use.

- ### `--zmq-endpoint-base `

Prefix for ZMQ endpoints used for IPC. Ensures unique endpoints across MAX Serve instances.
