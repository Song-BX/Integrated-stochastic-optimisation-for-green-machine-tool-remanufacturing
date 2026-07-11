"""I/O helpers for Stage 12 finalization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .config import Stage12Config


def require_stage1_gate(config: Stage12Config) -> Dict[str, Any]:
    """Require Stage 1 validation to have zero failures."""

    if not config.stage1_report.exists():
        raise FileNotFoundError(f"Stage 1 validation report not found: {config.stage1_report}")
    payload = read_json(config.stage1_report)
    failed = int(payload.get("summary", {}).get("failed", 0))
    if failed != 0:
        raise ValueError(f"Stage 1 gate failed; validation failures={failed}.")
    return payload


def ensure_dirs(config: Stage12Config) -> Dict[str, Path]:
    """Create Stage 12 output directories."""

    dirs = {
        "processed": config.processed_dir,
        "results": config.results_dir,
        "runs": config.results_dir / "runs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def existing_stage11_paths(config: Stage12Config) -> Dict[str, Path]:
    """Return important Stage 11 paths."""

    processed = config.data_processed_dir / "stage11"
    results = config.data_results_dir / "stage11"
    return {
        "artifact_manifest": processed / "artifact_manifest.csv",
        "artifact_manifest_json": processed / "artifact_manifest.json",
        "table_source_map": processed / "table_source_map.csv",
        "figure_source_map": processed / "figure_source_map.csv",
        "source_metric_catalogue": processed / "source_metric_catalogue.csv",
        "artifact_checks": results / "artifact_checks.json",
        "tables_dir": results / "tables",
        "figures_dir": results / "figures",
    }
