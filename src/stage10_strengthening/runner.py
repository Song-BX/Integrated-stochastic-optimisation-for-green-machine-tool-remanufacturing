"""Orchestrate Stage 10 targeted model strengthening."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .config import Stage10Config
from .io_utils import ensure_output_dirs, read_stage6_tables, require_stage1_passed
from .pair_carbon import analyze_pair_carbon
from .reporting import write_stage10_reports
from .shared_capacity import build_shared_capacity_instance, build_shared_capacity_model, solve_shared_capacity_experiment
from .structures import Stage10Result


def run_stage10(config: Stage10Config) -> tuple[Stage10Result, Dict[str, str]]:
    """Run pair-carbon audit and shared-capacity mini experiment."""

    ensure_output_dirs(config.processed_dir, config.results_dir)
    require_stage1_passed(config.stage1_report)
    tables = read_stage6_tables(config.raw_dir)
    pair_mapping, environmental_breakdown, pair_summary, _stage7_model_data = analyze_pair_carbon(tables, config)
    shared_instance = build_shared_capacity_instance(tables, config)
    shared_model = build_shared_capacity_model(shared_instance, config)
    solution_summary, comparison, utilization = solve_shared_capacity_experiment(shared_instance, shared_model, config)
    checks = _checks(pair_summary, shared_model, comparison)
    success = all(check["status"] != "failed" for check in checks)
    result = Stage10Result(
        success=success,
        status_message="Stage 10 completed." if success else "Stage 10 completed with failed checks.",
        pair_carbon_mapping=pair_mapping,
        environmental_objective_breakdown=environmental_breakdown,
        pair_carbon_summary=pair_summary,
        shared_capacity_instance_summary=shared_instance.to_summary_dict(),
        shared_capacity_model_summary=shared_model.to_summary_dict(),
        shared_capacity_solution_summary=solution_summary,
        shared_capacity_comparison=comparison,
        shared_capacity_utilization=utilization,
        checks=checks,
    )
    paths = write_stage10_reports(result, config)
    return result, paths


def _checks(pair_summary: object, shared_model: object, comparison: pd.DataFrame) -> List[Dict[str, object]]:
    checks = []
    checks.append(
        _check(
            "pair_carbon_coefficients_nonempty",
            pair_summary.pair_nonzero_coefficient_count > 0,
            f"nonzero pair carbon coefficients = {pair_summary.pair_nonzero_coefficient_count}",
        )
    )
    checks.append(
        _check(
            "environmental_vector_finite",
            bool(pair_summary.finite_objective_vector),
            f"finite environmental objective vector = {pair_summary.finite_objective_vector}",
        )
    )
    checks.append(
        _check(
            "shared_capacity_rows_nonempty",
            int(len(shared_model.shared_capacity_rows)) > 0,
            f"shared capacity rows = {len(shared_model.shared_capacity_rows)}",
        )
    )
    if comparison.empty:
        checks.append(_check("shared_capacity_solution_exists", False, "comparison table is empty"))
    else:
        shared = comparison[comparison["capacity_mode"] == "shared_capacity"]
        independent = comparison[comparison["capacity_mode"] == "independent_capacity_total"]
        checks.append(
            _check(
                "independent_capacity_solution_exists",
                not independent.empty and independent["success"].astype(bool).all(),
                f"independent aggregate rows = {len(independent)}",
            )
        )
        checks.append(
            _check(
                "shared_capacity_solution_exists",
                not shared.empty and shared["success"].astype(bool).all(),
                f"shared capacity rows = {len(shared)}",
            )
        )
        objective_values = pd.to_numeric(comparison.get("objective_value"), errors="coerce")
        checks.append(
            _check(
                "objective_values_finite",
                bool(np.isfinite(objective_values.dropna()).all()) and not objective_values.dropna().empty,
                "objective values are finite where reported",
            )
        )
    return checks


def _check(name: str, passed: bool, message: str) -> Dict[str, object]:
    return {"name": name, "status": "passed" if passed else "failed", "message": message}
