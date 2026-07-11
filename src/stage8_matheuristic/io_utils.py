"""I/O helpers for Stage 8."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from stage7_pareto.io_utils import CSV_ENCODING, ensure_output_dirs, read_stage6_tables, require_stage1_passed


def read_stage7_reference(stage7_results_dir: Path) -> Dict[str, object]:
    """Read exact Stage 7 Pareto results when available."""

    pareto_path = stage7_results_dir / "pareto_front.csv"
    grid_path = stage7_results_dir / "grid_solution_summary.csv"
    representatives_path = stage7_results_dir / "representative_solutions.json"
    if not pareto_path.exists():
        return {"status": "missing", "pareto_path": str(pareto_path)}
    try:
        pareto = pd.read_csv(pareto_path, encoding=CSV_ENCODING)
        grid = pd.read_csv(grid_path, encoding=CSV_ENCODING) if grid_path.exists() else pd.DataFrame()
        representatives = json.loads(representatives_path.read_text(encoding="utf-8")) if representatives_path.exists() else {}
    except (OSError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        return {"status": "unreadable", "pareto_path": str(pareto_path), "error": str(exc)}
    if pareto.empty:
        return {"status": "empty", "pareto_path": str(pareto_path)}
    return {
        "status": "loaded",
        "pareto_path": str(pareto_path),
        "grid_path": str(grid_path),
        "pareto_point_count": int(len(pareto)),
        "feasible_grid_points": int((grid.get("feasible", pd.Series(dtype=int)) == 1).sum()) if not grid.empty else None,
        "min_economic_risk": _safe_min(pareto, "economic_risk"),
        "min_environmental_impact": _safe_min(pareto, "environmental_impact"),
        "min_assembly_quality_loss": _safe_min(pareto, "assembly_quality_loss"),
        "pareto_records": pareto.to_dict(orient="records"),
        "representatives": representatives,
    }


def _safe_min(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns or frame.empty:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.min()) if not values.empty else None


__all__ = [
    "CSV_ENCODING",
    "ensure_output_dirs",
    "read_stage6_tables",
    "read_stage7_reference",
    "require_stage1_passed",
]
