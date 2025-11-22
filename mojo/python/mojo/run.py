# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ._package_root import get_package_root


def _workspace_import_sentinel() -> str:
    return os.environ.get("MODULAR_MOJO_MAX_IMPORT_SENTINEL", ".mojo-import-paths")


def _resolve_workspace_import_paths() -> list[str]:
    sentinel_name = _workspace_import_sentinel()
    cwd = Path.cwd()
    search_roots = [cwd] + list(cwd.parents)
    for root in search_roots:
        sentinel = root / sentinel_name
        if not sentinel.is_file():
            continue

        try:
            raw_lines = sentinel.read_text().splitlines()
        except OSError:
            return []

        paths: list[str] = []
        for line in raw_lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            path_obj = Path(stripped)
            if not path_obj.is_absolute():
                path_obj = (root / path_obj).resolve()
            paths.append(str(path_obj))
        return paths
    return []


# fmt: off
def _sdk_default_env() -> dict[str, str]:
    root = get_package_root()

    # Running in Bazel
    if not root:
        return {}

    bin = root / "bin"
    lib = root / "lib"

    ext = ".dylib" if sys.platform == "darwin" else ".so"

    # Special case for wheel entrypoint - in
    # the venv it is actually put in the root `bin`.
    # lib/python3.13/site-packages/max/
    # ->
    # bin/mblack
    extra_env = {}
    maybe_mblack_path = root.parent.parent.parent.parent / "bin" / "mblack"
    if maybe_mblack_path.exists():
        extra_env["MODULAR_MOJO_MAX_MBLACK_PATH"] = str(maybe_mblack_path)

    env = {
        "MODULAR_MAX_PACKAGE_ROOT": str(root),
        "MODULAR_MAX_CACHE_DIR": str(root / "share" / "max" / ".max_cache"),
        "MODULAR_MAX_ENABLE_MODEL_IR_CACHE": "true",
        "MODULAR_MAX_PATH": str(root),
        "MODULAR_MAX_NAME": "MAX Platform",
        # MODULAR_MAX_VERSION intentionally omitted

        "MODULAR_MOJO_MAX_PACKAGE_ROOT": str(root),
        "MODULAR_MOJO_MAX_COMPILERRT_PATH": (
            str(lib / "libKGENCompilerRTShared") + ext
        ),
        "MODULAR_MOJO_MAX_MGPRT_PATH": str(lib / "libMGPRT") + ext,
        "MODULAR_MOJO_MAX_SHARED_LIBS": (
            str(lib / "libAsyncRTMojoBindings")
            + ext
            + ","
            + str(lib / "libAsyncRTRuntimeGlobals")
            + ext
            + ","
            + str(lib / "libMSupportGlobals")
            + ext
            + ",-Xlinker,-rpath,-Xlinker,"
            + str(lib)
        ),
        "MODULAR_MOJO_MAX_DRIVER_PATH": str(bin / "mojo"),
        "MODULAR_MOJO_MAX_IMPORT_PATH": str(lib / "mojo"),
        # MODULAR_MOJO_MAX_JUPYTER_PATH
        "MODULAR_MOJO_MAX_LLDB_PATH": str(bin / "mojo-lldb"),
        "MODULAR_MOJO_MAX_LLDB_PLUGIN_PATH": str(lib / "libMojoLLDB") + ext,
        "MODULAR_MOJO_MAX_LLDB_VISUALIZERS_PATH": str(lib / "lldb-visualizers"),
        # env["MODULAR_MOJO_MAX_LLDB_VSCODE_PATH"] = str(bin / "mojo-lldb-dap")
        "MODULAR_MOJO_MAX_LSP_SERVER_PATH": str(bin / "mojo-lsp-server"),
        "MODULAR_MOJO_MAX_REPL_ENTRY_POINT": str(lib / "mojo-repl-entry-point"),
        "MODULAR_MOJO_MAX_LLD_PATH": str(bin / "lld"),
        "MODULAR_MOJO_MAX_SYSTEM_LIBS": (
            "-lm"
            if sys.platform == "darwin"
            else "-lrt,-ldl,-lpthread,-lm"
        ),

        "MODULAR_CRASH_REPORTING_HANDLER_PATH": str(
            bin / "modular-crashpad-handler"
        ),
    } | extra_env

    workspace_paths = _resolve_workspace_import_paths()
    if workspace_paths:
        base_path = env.get("MODULAR_MOJO_MAX_IMPORT_PATH", "")
        merged_paths = workspace_paths + ([base_path] if base_path else [])
        env["MODULAR_MOJO_MAX_IMPORT_PATH"] = ",".join(merged_paths)

    return env
# fmt: on


def _mojo_env() -> dict[str, str]:
    """Returns an environment variable set that uses the Mojo SDK environment
    paths by default, but with overrides from the ambient OS environment."""

    return _sdk_default_env() | dict(os.environ)


def subprocess_run_mojo(  # noqa: ANN201
    mojo_args: list[str],
    **kwargs: Any,
):
    """Launches the bundled `mojo` in a subprocess, configured to use the
    `mojo` assets in the `max` package.

    Arguments:
        mojo_args: Arguments supplied to the `mojo` command.
        kwargs: Additional arguments to pass to `subprocess.run()`
    """

    env = _mojo_env()
    mojo = env.get("MODULAR_MOJO_MAX_DRIVER_PATH") or shutil.which("mojo")
    if not mojo or not os.path.exists(mojo):
        raise RuntimeError("error: Could not find `mojo` executable")

    return subprocess.run(
        # Combine the `mojo` executable path with the provided argument list.
        [mojo] + mojo_args,
        env=env,
        **kwargs,
    )
