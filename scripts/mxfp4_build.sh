#!/usr/bin/env bash
# Builds the MXFP4 custom Mojo package and records its absolute path.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pkg_rel="bazel-bin/max/kernels/src/custom_ops/mogg_mxfp4/mogg_mxfp4.mojopkg"

cd "$repo_root"

./bazelw build //max/kernels/src/custom_ops/mogg_mxfp4:mogg_mxfp4
./bazelw build //max/kernels/src/Mogg/MOGGKernelAPI:MOGGKernelAPI

pkg_path="$repo_root/$pkg_rel"
if [[ ! -e "$pkg_path" ]]; then
  echo "error: expected package at $pkg_path but it was not produced" >&2
  exit 1
fi

pkg_abs="$(
  readlink -f "$pkg_path" 2>/dev/null || \
  python - "$pkg_path" <<'PY'
import os, sys
path = sys.argv[1]
  print(os.path.abspath(path))
PY
)"

echo "MXFP4 custom package: $pkg_abs"
builtin_pkg="$repo_root/bazel-bin/max/kernels/src/Mogg/MOGGKernelAPI/MOGGKernelAPI.mojopkg"
if [[ -f "$builtin_pkg" ]]; then
  echo "MOGGKernelAPI package: $builtin_pkg"
fi
echo
echo "Use one of the above paths with MAX_CUSTOM_EXTENSIONS if needed."
