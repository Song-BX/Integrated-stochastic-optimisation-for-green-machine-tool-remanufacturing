"""I/O helpers and Stage 1 gate checks for Stage 4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd


CSV_ENCODING = "utf-8-sig"

REQUIRED_STAGE4_FILES = [
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
]


def require_stage1_passed(report_path: Path) -> Dict[str, object]:
    """Read Stage 1 validation report and fail if it contains failures."""

    if not report_path.exists():
        raise FileNotFoundError(
            f"Stage 1 validation report not found: {report_path}. "
            "Run scripts/scan_data_catalogue.py before Stage 4."
        )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    failed = int(payload.get("summary", {}).get("failed", 0))
    if failed != 0:
        raise RuntimeError(f"Stage 1 validation gate failed: validation_report.json has failed={failed}.")
    return payload


def read_stage4_tables(raw_dir: Path) -> Dict[str, pd.DataFrame]:
    missing = [file_name for file_name in REQUIRED_STAGE4_FILES if not (raw_dir / file_name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing Stage 4 input files: {', '.join(missing)}")
    return {
        file_name.removesuffix(".csv"): pd.read_csv(raw_dir / file_name, encoding=CSV_ENCODING)
        for file_name in REQUIRED_STAGE4_FILES
    }


def ensure_output_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
