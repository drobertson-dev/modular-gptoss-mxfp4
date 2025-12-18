import sys
from pathlib import Path


def _add_src_to_path():
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if src.is_dir():
        sys.path.insert(0, str(src))


_add_src_to_path()
