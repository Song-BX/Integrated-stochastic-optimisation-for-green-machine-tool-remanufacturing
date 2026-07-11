"""I/O helpers for Stage 10."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from stage6_selective_assembly.io_utils import CSV_ENCODING, read_stage6_tables, require_stage1_passed


def ensure_output_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


__all__ = [
    "CSV_ENCODING",
    "ensure_output_dirs",
    "read_stage6_tables",
    "require_stage1_passed",
]

