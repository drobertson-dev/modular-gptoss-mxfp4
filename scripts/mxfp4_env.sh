#!/usr/bin/env bash
# Sets up a predictable environment for MXFP4 Mojo work.
# Can be executed directly (Pixi activation) or sourced in a shell.

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
default_cache_dir="$(read_sdk_env_var MODULAR_MAX_CACHE_DIR 2>/dev/null || printf '')"
existing_import_path="${MODULAR_MOJO_MAX_IMPORT_PATH:-}"

sentinel="$repo_root/.mojo-import-paths"
sentinel_paths=()
if [[ -f "$sentinel" ]]; then
  while IFS= read -r line; do
    trimmed="${line%%#*}"
    trimmed="${trimmed##[[:space:]]}"
    trimmed="${trimmed%%[[:space:]]}"
    if [[ -z "$trimmed" ]]; then
      continue
    fi
    if [[ "$trimmed" != /* ]]; then
      trimmed="$repo_root/$trimmed"
    fi
    sentinel_paths+=("$trimmed")
  done < "$sentinel"
fi

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
for path in "${sentinel_paths[@]}"; do
  if [[ "$path" != "$local_kernel_path" ]]; then
    extra_paths+=("$path")
  fi
done

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
export MODULAR_MOJO_MAX_IMPORT_SENTINEL="$sentinel"

if [[ -n "$mojo_lib_dir" ]]; then
  export MOJO_PACKAGE_PATH="$(combine_colon_paths "$mojo_lib_dir" "${MOJO_PACKAGE_PATH:-}")"
else
  export MOJO_PACKAGE_PATH="${MOJO_PACKAGE_PATH:-}"
fi

py_paths=("$repo_root/max/python" "$repo_root/mojo/python")
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${py_paths[0]}:${py_paths[1]}:$PYTHONPATH"
else
  export PYTHONPATH="${py_paths[0]}:${py_paths[1]}"
fi

export MODULAR_DEVICE_CONTEXT_SYNC_MODE="${MODULAR_DEVICE_CONTEXT_SYNC_MODE:-true}"

pkg_cache_file="$repo_root/.mxfp4-package-path"
if [[ -f "$pkg_cache_file" ]]; then
  mxfp4_pkg_path="$(<"$pkg_cache_file")"
else
  mxfp4_pkg_path=""
fi

if [[ -n "$mxfp4_pkg_path" ]]; then
  export MXFP4_KERNEL_PACKAGE="$mxfp4_pkg_path"
  export MAX_CUSTOM_EXTENSIONS="$mxfp4_pkg_path"
else
  unset MXFP4_KERNEL_PACKAGE MAX_CUSTOM_EXTENSIONS
fi

export MAX_ALLOW_UNSUPPORTED_ENCODING=1

cache_dir="${MODULAR_MAX_CACHE_DIR:-}" 
if [[ -z "$cache_dir" ]]; then
  cache_dir="$default_cache_dir"
fi
if [[ -n "$cache_dir" ]]; then
  rm -rf "$cache_dir"
  export MODULAR_MAX_CACHE_DIR="$cache_dir"
fi

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Set MODULAR_MOJO_MAX_IMPORT_PATH=$MODULAR_MOJO_MAX_IMPORT_PATH"
  echo "Set PYTHONPATH=$PYTHONPATH"
  echo "Set MODULAR_DEVICE_CONTEXT_SYNC_MODE=$MODULAR_DEVICE_CONTEXT_SYNC_MODE"
  if [[ -n "$mxfp4_pkg_path" ]]; then
    echo "Set MXFP4 kernel package=$mxfp4_pkg_path"
  else
    echo "Warning: MXFP4 kernel package path not found; run pixi run mxfp4-build" >&2
  fi
  if [[ -n "$cache_dir" ]]; then
    echo "Cleared MAX cache at $cache_dir"
  fi
fi
