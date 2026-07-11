"""I/O helpers and Stage 1 gate checks for Stage 6."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from stage5_risk_averse.io_utils import CSV_ENCODING, ensure_output_dirs, require_stage1_passed


REQUIRED_STAGE6_FILES = [
    "time_periods.csv",
    "machine_types.csv",
    "bom.csv",
    "returned_cores.csv",
    "component_inspection.csv",
    "routes.csv",
    "route_feasibility.csv",
    "processing_parameters.csv",
    "orders.csv",
    "capacity_calendar.csv",
    "initial_inventory.csv",
    "procurement_parameters.csv",
    "baseline_rules.csv",
    "scenarios.csv",
    "demand_scenarios.csv",
    "component_quality_scenarios.csv",
    "route_outcome_scenarios.csv",
    "reliability_parameters.csv",
    "risk_parameters.csv",
    "assembly_requirements.csv",
    "assembly_candidates.csv",
    "assembly_compatibility.csv",
]


def read_stage6_tables(raw_dir: Path) -> Dict[str, pd.DataFrame]:
    missing = [file_name for file_name in REQUIRED_STAGE6_FILES if not (raw_dir / file_name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing Stage 6 input files: {', '.join(missing)}")
    return {
        file_name.removesuffix(".csv"): pd.read_csv(raw_dir / file_name, encoding=CSV_ENCODING)
        for file_name in REQUIRED_STAGE6_FILES
    }


def read_stage5_solution_summary(stage5_results_dir: Path) -> Dict[str, object]:
    path = stage5_results_dir / "solution_summary.json"
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"status": "unreadable", "path": str(path), "error": str(exc)}
    solution = payload.get("solution", {})
    metrics = solution.get("summary_metrics", {})
    cvar = solution.get("cvar_summary", {})
    return {
        "status": "loaded",
        "path": str(path),
        "objective_value": solution.get("objective_value"),
        "expected_demand_units": metrics.get("expected_demand_units"),
        "expected_assembled_units": metrics.get("expected_assembled_units"),
        "expected_final_backlog_units": metrics.get("expected_final_backlog_units"),
        "selected_component_route_count": metrics.get("selected_component_route_count"),
        "cvar_value": cvar.get("cvar_value"),
        "worst_scenario_loss": cvar.get("worst_scenario_loss"),
        "worst_scenario_id": cvar.get("worst_scenario_id"),
    }


__all__ = [
    "CSV_ENCODING",
    "ensure_output_dirs",
    "read_stage5_solution_summary",
    "read_stage6_tables",
    "require_stage1_passed",
]
