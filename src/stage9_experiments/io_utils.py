"""I/O helpers for Stage 9."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def require_stage1_passed(path: Path) -> Dict[str, object]:
    """Load Stage 1 validation report and require zero failures."""

    if not path.exists():
        raise FileNotFoundError(f"Stage 1 validation report not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    failed = int(payload.get("summary", {}).get("failed", 0))
    if failed != 0:
        raise RuntimeError(f"Stage 1 validation failed; failed={failed}")
    return payload
