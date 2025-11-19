# MXFP4 Custom Kernel Bring-Up Notes

Repo root: `/workspace/modular-gptoss-mxfp4`

## Building the custom Mojo package
```
./bazelw build //max/kernels/src/custom_ops/mogg_mxfp4:mogg_mxfp4
```
Result: `bazel-bin/max/kernels/src/custom_ops/mogg_mxfp4/mogg_mxfp4.mojopkg`

## Running the integration test
```
MXFP4_KERNEL_PACKAGE=$PWD/bazel-bin/max/kernels/src/custom_ops/mogg_mxfp4/mogg_mxfp4.mojopkg \
PYTHONPATH=$PWD/max/python \
pixi run python -m pytest max/tests/integration/API/python/graph/test_mxfp4_moe_kernel.py \
    -k test_mxfp4_grouped_matmul_matches_dense -q
```

## Serving GPT-OSS with MXFP4
```
export MAX_CUSTOM_EXTENSIONS=$PWD/bazel-bin/max/kernels/src/custom_ops/mogg_mxfp4/mogg_mxfp4.mojopkg
export MAX_ALLOW_UNSUPPORTED_ENCODING=1
PYTHONPATH=$PWD/max/python pixi run python -m max.entrypoints.pipelines serve \
    --model openai/gpt-oss-120b \
    --quantization-encoding mxfp4 \
    --devices gpu \
    --force
```
During startup you should see `Loading 1 custom Mojo extension(s)` and the compile step should finish in ~10–12 seconds.

## Environment reminders
- Pixi environment is pinned to Python 3.11.14 (see `pixi.toml`).
- `MAX_CUSTOM_EXTENSIONS` can list multiple `.mojopkg` paths separated by `:`.
- `grouped_mxfp4_matmul` first tries `mo.moe.mx4.matmul` then `custom.moe.mx4.matmul`. Override via `MAX_MXFP4_KERNEL_OP` if needed.

## Clean-up performed
- Removed `serve.log` and generated `kvcache_agent_service_v1_pb2*.py` files.
- All MXFP4-related changes live under `max/kernels/src/custom_ops/mogg_mxfp4` and the new helper `max/python/max/pipelines/lib/custom_extensions.py`.
