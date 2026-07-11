"""Optional run-mode helpers for Stage 9."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

from .config import Stage9Config


def maybe_run_profile(config: Stage9Config, root: Path) -> Dict[str, object]:
    """Execute optional quick-run experiments when explicitly requested."""

    if config.execution_mode != "run":
        return {"execution_mode": config.execution_mode, "executed": []}
    if config.profile != "quick-run":
        return {
            "execution_mode": config.execution_mode,
            "executed": [],
            "warning": f"Run mode currently supports profile='quick-run'; received profile={config.profile}.",
        }
    sys.path.insert(0, str(root / "src"))
    from stage8_matheuristic.benchmark import run_single_instance
    from stage8_matheuristic.config import Stage8Config
    from stage8_matheuristic.io_utils import read_stage6_tables, require_stage1_passed

    run_root = config.results_dir / "runs" / "quick_stage8_matheuristic"
    processed_dir = config.processed_dir / "runs" / "quick_stage8_matheuristic"
    stage8_config = Stage8Config(
        raw_dir=root / "data/raw",
        stage1_report=config.stage1_report,
        processed_dir=processed_dir,
        results_dir=run_root,
        stage4_results_dir=config.data_results_dir / "stage4",
        stage5_results_dir=config.data_results_dir / "stage5",
        stage6_results_dir=config.data_results_dir / "stage6",
        stage7_results_dir=config.data_results_dir / "stage7",
        machine_type_id=config.machine_type_id,
        period_start=config.period_start,
        period_count=config.period_count,
        processing_window_periods=config.processing_window_periods,
        epsilon_grid_size=config.run_epsilon_grid_size,
        max_iterations=config.run_max_iterations,
        repair_time_limit=config.run_repair_time_limit,
    )
    require_stage1_passed(config.stage1_report)
    tables = read_stage6_tables(stage8_config.raw_dir)
    result, paths = run_single_instance(tables, stage8_config)
    summary_path = config.results_dir / "quick_run_summary.json"
    config.results_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "execution_mode": config.execution_mode,
        "executed": ["quick_stage8_matheuristic"],
        "success": result.success,
        "approx_pareto_points": int(len(result.approx_pareto_front)),
        "repair_solves": int(len(result.repair_solve_log)),
        "feasible_repair_solves": int((result.repair_solve_log["feasible"] == True).sum()) if not result.repair_solve_log.empty else 0,  # noqa: E712
        "paths": paths,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
