#!/usr/bin/env bash
# Sets up a predictable environment for MXFP4 Mojo work without relying on
# repo-sentinel files. Can be executed directly (Pixi activation) or sourced.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

read_sdk_env_var() {
  local key="$1"
  python - "$key" <<'PY'
from mojo.run import _sdk_default_env
import sys
env = _sdk_default_env()
print(env.get(sys.argv[1], ''), end='')
PY
}

default_import_path="$(read_sdk_env_var MODULAR_MOJO_MAX_IMPORT_PATH 2>/dev/null || printf '')"
existing_import_path="${MODULAR_MOJO_MAX_IMPORT_PATH:-}"

mojo_lib_dir="$(python - <<'PY'
import sys
from pathlib import Path
path = Path(sys.executable).resolve().parent.parent / "lib" / "mojo"
print(path if path.exists() else "")
PY
)"

local_kernel_path="$repo_root/max/kernels/src"
extra_paths=()
if [[ -n "$mojo_lib_dir" ]]; then
  extra_paths+=("$mojo_lib_dir")
fi
extra_paths+=("$local_kernel_path")

combine_paths() {
  local joined=()
  for value in "$@"; do
    [[ -z "$value" ]] && continue
    joined+=("$value")
  done
  (IFS=','; echo "${joined[*]}")
}

combine_colon_paths() {
  local joined=()
  for value in "$@"; do
    [[ -z "$value" ]] && continue
    joined+=("$value")
  done
  (IFS=':'; echo "${joined[*]}")
}

MODULAR_MOJO_MAX_IMPORT_PATH="$(combine_paths "${extra_paths[@]}" "$default_import_path" "$existing_import_path")"

export MODULAR_MOJO_MAX_IMPORT_PATH
export MOJO_PACKAGE_PATH="$(combine_colon_paths "$mojo_lib_dir" "${MOJO_PACKAGE_PATH:-}")"

py_paths=("$repo_root/max/python" "$repo_root/mojo/python")
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${py_paths[0]}:${py_paths[1]}:$PYTHONPATH"
else
  export PYTHONPATH="${py_paths[0]}:${py_paths[1]}"
fi

export MODULAR_DEVICE_CONTEXT_SYNC_MODE="${MODULAR_DEVICE_CONTEXT_SYNC_MODE:-true}"

# Prefer the in-tree Mojo package; fall back to the custom ops package if built.
pkg_candidates=(
  "$repo_root/bazel-bin/max/kernels/src/Mogg/MOGGKernelAPI/MOGGKernelAPI.mojopkg"
  "$repo_root/bazel-bin/max/kernels/src/Mogg/MOGGKernelAPI/MOGGMxfp4Extension.mojopkg"
  "$repo_root/bazel-bin/max/kernels/src/custom_ops/mogg_mxfp4/mogg_mxfp4.mojopkg"
)
mxfp4_pkg_path=""
for candidate in "${pkg_candidates[@]}"; do
  if [[ -f "$candidate" ]]; then
    mxfp4_pkg_path="$candidate"
    break
  fi
done

if [[ -n "$mxfp4_pkg_path" ]]; then
  export MXFP4_KERNEL_PACKAGE="$mxfp4_pkg_path"
  export MAX_CUSTOM_EXTENSIONS="$mxfp4_pkg_path"
else
  unset MXFP4_KERNEL_PACKAGE MAX_CUSTOM_EXTENSIONS
fi

export MAX_ALLOW_UNSUPPORTED_ENCODING=1

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Set MODULAR_MOJO_MAX_IMPORT_PATH=$MODULAR_MOJO_MAX_IMPORT_PATH"
  echo "Set PYTHONPATH=$PYTHONPATH"
  if [[ -n "$mxfp4_pkg_path" ]]; then
    echo "Set MXFP4 kernel package=$mxfp4_pkg_path"
  else
    echo "Warning: MXFP4 kernel package not found; build //max/kernels/src/Mogg/MOGGKernelAPI:MOGGKernelAPI" >&2
  fi
fi
