"""I/O helpers for Stage 7."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from stage6_selective_assembly.io_utils import CSV_ENCODING, ensure_output_dirs, read_stage6_tables, require_stage1_passed


def read_stage6_solution_summary(stage6_results_dir: Path) -> Dict[str, object]:
    path = stage6_results_dir / "solution_summary.json"
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"status": "unreadable", "path": str(path), "error": str(exc)}
    solution = payload.get("solution", {})
    return {
        "status": "loaded",
        "path": str(path),
        "objective_value": solution.get("objective_value"),
        "summary_metrics": solution.get("summary_metrics", {}),
        "objective_breakdown": solution.get("objective_breakdown", {}),
        "cvar_summary": solution.get("cvar_summary", {}),
    }


__all__ = [
    "CSV_ENCODING",
    "ensure_output_dirs",
    "read_stage6_solution_summary",
    "read_stage6_tables",
    "require_stage1_passed",
]
