"""Custom GPT-OSS MXFP4 architecture for MAX pipelines."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_modular_home() -> None:
    """Ensure MAX can locate Mojo stdlib packages.

    Running the pixi env's `python` directly (without `pixi run` / `pixi shell`)
    does not set `MODULAR_HOME`, which can cause Mojo package loads to fail with:
    "unable to locate module 'stdlib'".
    """

    modular_home = os.environ.get("MODULAR_HOME")
    if modular_home and Path(modular_home).exists():
        return

    env_root = Path(sys.executable).resolve().parents[1]
    candidate = env_root / "share" / "max"
    if candidate.exists():
        os.environ["MODULAR_HOME"] = str(candidate)


_ensure_modular_home()

from .arch import gpt_oss_arch
from .arch_module_v3 import gpt_oss_module_v3_arch

ARCHITECTURES = [gpt_oss_arch, gpt_oss_module_v3_arch]

__all__ = ["ARCHITECTURES", "gpt_oss_arch", "gpt_oss_module_v3_arch"]
