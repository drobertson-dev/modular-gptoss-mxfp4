# MXFP4 Custom Kernel Bring-Up Notes

Repo root: `/workspace/modular-gptoss-mxfp4`

## Building the custom Mojo package

```bash
pixi run mxfp4-build
```

This wraps `scripts/mxfp4_build.sh`, which runs `./bazelw build //max/kernels/src/Mogg/MOGGKernelAPI:MOGGKernelAPI` (and the custom package) and prints the resulting `.mojopkg` path. No cache file is written; use the printed path directly when exporting `MAX_CUSTOM_EXTENSIONS`.

## Running the integration test

```bash
pixi run mxfp4-test
```

The task depends on `mxfp4-build`, sources `scripts/mxfp4_env.sh`, runs the Mojo unit test (`max/kernels/test/mxfp4_mojo/test_mxfp4.mojo`), and executes the Python integration check `test_mxfp4_grouped_matmul_matches_dense` for parity with the dense path.

## Serving GPT-OSS with MXFP4

```bash
pixi run mxfp4-serve
```

Like the other tasks, this rebuilds the kernel if needed, sources the env helper, and then launches `max.entrypoints.pipelines serve --model openai/gpt-oss-20b --quantization-encoding mxfp4 --devices gpu --force`. Because the environment hook sets `MAX_ALLOW_UNSUPPORTED_ENCODING=1`, there is no need to export it manually.

## Running the Mojo CLI tests

Pixi sources `scripts/mxfp4_env.sh` on every `pixi run`/`pixi shell`. The helper derives import paths from the repo root (no sentinel files), prepends them plus the SDK defaults to `MODULAR_MOJO_MAX_IMPORT_PATH`, exports `PYTHONPATH`, and injects the freshly built `.mojopkg` into `MAX_CUSTOM_EXTENSIONS` when present. Inside a Pixi-managed shell (including VS Code when launched via `pixi run code`), invoking `mojo` anywhere inside the workspace will locate `nn.moe_mxfp4`.

Outside Pixi, source the script manually:

```bash
source scripts/mxfp4_env.sh
```

With the activation hook enabled, the standard MXFP4 Mojo tests work directly:

```bash
pixi run mojo run -I max/kernels/src max/kernels/test/mxfp4_mojo/test_mxfp4.mojo
```

During startup you should see `Loading 1 custom Mojo extension(s)` and the compile step should finish in ~10–12 seconds.

## Environment reminders

- To avoid path drift, you can source `scripts/mxfp4_env.sh` to export the Mojo
    search paths (`MODULAR_MOJO_MAX_IMPORT_PATH`) and `PYTHONPATH` before running
    tests or launching the debugger. The script also sets `MAX_CUSTOM_EXTENSIONS`
    to the freshly built `.mojopkg` when available.
- Pixi environment is pinned to Python 3.11.14 (see `pixi.toml`).
- `MAX_CUSTOM_EXTENSIONS` can list multiple `.mojopkg` paths separated by `:`.
- `grouped_mxfp4_matmul` first tries `mo.moe.mx4.matmul` then `custom.moe.mx4.matmul`. Override via `MAX_MXFP4_KERNEL_OP` if needed.

## Runtime guard for MXFP4 quantization

`max.pipelines.lib.pipeline_variants.TextGenerationPipeline` still supports `--quantization-encoding mxfp4`, but the MXFP4 kernel must be available either from the built-in runtime or by pointing `MAX_CUSTOM_EXTENSIONS` at the freshly built `.mojopkg` (for local builds). The scalar fall-back kernel has been removed entirely; if MAX would otherwise run anything but the Hopper-optimized kernel, it raises a runtime error.

## Debugging from VS Code

The repo now ships with a ready-to-use Mojo launch configuration under `.vscode/launch.json` named **Debug MXFP4 Mojo tests**. It launches
`max/kernels/test/mxfp4_mojo/test_mxfp4.mojo` via the Mojo LLDB adapter with the right environment so we can breakpoint the GPU/CPU kernels
without reinventing the import plumbing each time.

Prerequisites:

- Install the Mojo VS Code extension (plus the Python extension it depends on).
- Point the Python extension at the Pixi interpreter (Command Palette → *Python: Select Interpreter* → choose `.pixi/envs/default/bin/python`).
- Make sure the Pixi environment already has the `modular` SDK installed (`pixi install` in the repo root covers this).

What the launch config does:

- Sets `PYTHONPATH=${workspaceFolder}/max/python` so the test suite can import MAX Python helpers.
- Extends `MODULAR_MOJO_MAX_IMPORT_PATH` with `${workspaceFolder}/max/kernels/src` so Mojo resolves the local kernel sources before falling
    back to the SDK copies.
- Enables `MODULAR_DEVICE_CONTEXT_SYNC_MODE=true` to force synchronous device execution (handy for debugging the CUDA illegal address).
- Runs inside `${workspaceFolder}` and spawns the debugger in a dedicated terminal so stdio is interactive.

Using it:

1. Open the Run and Debug view (`Ctrl+Shift+D`) and select **Debug MXFP4 Mojo tests**.
2. Hit F5. Execution pauses on your first breakpoint; otherwise it will run until the CUDA error reproduces and the debugger halts.

Because the configuration relies on VS Code's environment resolution, any extra tweaks (e.g. toggling kernel variants, DEBUG flags, etc.) can
be layered on by editing `.vscode/launch.json` or exporting variables in the terminal that launches VS Code.

## Clean-up performed

- Removed `serve.log` and generated `kvcache_agent_service_v1_pb2*.py` files.
- All MXFP4-related changes live under `max/kernels/src/custom_ops/mogg_mxfp4` and the new helper `max/python/max/pipelines/lib/custom_extensions.py`.
