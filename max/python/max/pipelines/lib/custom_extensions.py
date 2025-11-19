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
"""Helpers for loading Mojo custom extension packages."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

from mojo.paths import is_mojo_source_package_path

LOGGER = logging.getLogger(__name__)


def _workspace_root() -> Path | None:
    current = Path(__file__).resolve()
    for candidate in (current,) + tuple(current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _dedupe(paths: Iterable[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen:
            continue
        deduped.append(resolved)
        seen.add(resolved)
    return deduped


def _candidate_mojo_lib_dirs() -> list[Path]:
    candidates: list[Path] = []
    if override := os.environ.get("MAX_MOJO_LIB_DIR"):
        candidates.append(Path(override).expanduser())

    for prefix in {sys.prefix, sys.base_prefix}:
        candidates.append(Path(prefix) / "lib" / "mojo")

    if workspace := _workspace_root():
        pixi_envs = workspace / ".pixi" / "envs"
        if pixi_envs.is_dir():
            for env_dir in sorted(pixi_envs.iterdir()):
                candidates.append(env_dir / "lib" / "mojo")

    deduped = []
    seen: set[Path] = set()
    for path in candidates:
        key = path.resolve() if path.exists() else path
        if key in seen:
            continue
        deduped.append(path)
        seen.add(key)
    return deduped


def runtime_dependency_paths(logger: logging.Logger | None = None) -> list[Path]:
    """Locate stdlib/compiler_internal packages for custom ops."""

    logger = logger or LOGGER
    required = ("compiler_internal.mojopkg", "stdlib.mojopkg")
    found: list[Path] = []
    found_names: set[str] = set()

    for lib_dir in _candidate_mojo_lib_dirs():
        if not lib_dir.exists():
            continue
        for filename in required:
            if filename in found_names:
                continue
            candidate = lib_dir / filename
            if candidate.exists():
                found.append(candidate.resolve())
                found_names.add(filename)
        if len(found_names) == len(required):
            break

    if len(found_names) < len(required):
        missing = ", ".join(name for name in required if name not in found_names)
        logger.warning(
            "Unable to locate all Mojo runtime dependencies (%s); custom extensions may fail to load",
            missing,
        )

    return found


def _update_env_path_list(
    *,
    var_name: str,
    paths: Sequence[Path],
    separator: str,
    logger: logging.Logger | None = None,
) -> None:
    logger = logger or LOGGER
    if not paths:
        return

    existing = os.environ.get(var_name, "")
    current_entries = [
        Path(entry)
        for entry in existing.split(separator)
        if entry.strip()
    ]

    added = False
    for directory in paths:
        if directory in current_entries:
            continue
        current_entries.append(directory)
        added = True

    if not added:
        return

    os.environ[var_name] = separator.join(str(path) for path in current_entries)
    logger.info(
        "Set %s to include %s",
        var_name,
        ", ".join(str(path) for path in paths),
    )


def _extend_mojo_search_paths(
    packages: Iterable[Path], logger: logging.Logger | None = None
) -> None:
    resolved_dirs: list[Path] = []
    for path in packages:
        directory = path if path.is_dir() else path.parent
        if not directory.exists():
            continue
        resolved_dirs.append(directory.resolve())

    if not resolved_dirs:
        return

    _update_env_path_list(
        var_name="MOJO_PACKAGE_PATH",
        paths=resolved_dirs,
        separator=os.pathsep,
        logger=logger,
    )
    _update_env_path_list(
        var_name="MODULAR_MOJO_MAX_IMPORT_PATH",
        paths=resolved_dirs,
        separator=",",
        logger=logger,
    )


def parse_custom_extensions_env(
    env_value: str | None,
    *,
    logger: logging.Logger | None = None,
) -> list[Path]:
    """Parses MAX_CUSTOM_EXTENSIONS, expanding directories into packages."""

    logger = logger or LOGGER
    if not env_value:
        return []

    parsed: list[Path] = []
    for entry in env_value.split(os.pathsep):
        entry = entry.strip()
        if not entry:
            continue
        candidate = Path(entry).expanduser()
        if candidate.is_dir():
            if is_mojo_source_package_path(candidate):
                parsed.append(candidate)
                continue
            packages = sorted(candidate.glob("*.mojopkg"))
            if not packages:
                logger.warning(
                    "No Mojo packages found under custom extension directory %s",
                    candidate,
                )
            parsed.extend(packages)
            continue
        if not candidate.exists():
            logger.warning("Custom extension path %s does not exist; skipping", candidate)
            continue
        parsed.append(candidate)

    return _dedupe(parsed)


def collect_custom_extensions_from_env(
    env_value: str | None,
    *,
    include_runtime_dependencies: bool = True,
    logger: logging.Logger | None = None,
) -> list[Path]:
    """Returns parsed custom extensions plus runtime deps when requested."""

    parsed = parse_custom_extensions_env(env_value, logger=logger)
    if not parsed:
        return []

    if include_runtime_dependencies:
        deps = runtime_dependency_paths(logger=logger)
        _extend_mojo_search_paths(deps, logger=logger)
    else:
        _extend_mojo_search_paths(parsed, logger=logger)

    return parsed
