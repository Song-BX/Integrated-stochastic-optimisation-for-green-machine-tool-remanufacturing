"""Benchmark-suite orchestration for Stage 8."""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from .aggregation import build_stage8_instance
from .config import Stage8Config, TOP5_52W_MACHINE_TYPES
from .model import build_model_data
from .reporting import write_stage8_reports
from .solver import solve_matheuristic
from .structures import Stage8RunResult


def benchmark_machine_types(config: Stage8Config) -> Tuple[str, ...]:
    """Return machine types for the configured benchmark suite."""

    if config.benchmark_suite == "top5_52w":
        return tuple(config.benchmark_machine_types or TOP5_52W_MACHINE_TYPES)
    return (config.machine_type_id,)


def run_single_instance(tables: Dict[str, pd.DataFrame], config: Stage8Config) -> tuple[Stage8RunResult, Dict[str, str]]:
    """Build, solve, and report one Stage 8 instance."""

    instance = build_stage8_instance(tables, config)
    model_data = build_model_data(instance, config, tables)
    result = solve_matheuristic(instance, model_data, config)
    paths = write_stage8_reports(model_data, result, config)
    return result, paths


def run_benchmark_suite(tables: Dict[str, pd.DataFrame], config: Stage8Config) -> tuple[pd.DataFrame, Dict[str, Dict[str, str]]]:
    """Run each benchmark instance independently and return the summary table."""

    rows = []
    path_map: Dict[str, Dict[str, str]] = {}
    for machine_type in benchmark_machine_types(config):
        start = time.perf_counter()
        machine_config = _machine_config(config, machine_type)
        try:
            result, paths = run_single_instance(tables, machine_config)
            pareto = result.approx_pareto_front
            incumbent = result.incumbent_solution_summary
            row = {
                "machine_type_id": machine_type,
                "success": bool(result.success),
                "status_message": result.status_message,
                "solve_seconds": float(result.solve_seconds),
                "wall_seconds": float(time.perf_counter() - start),
                "repair_solves": int(len(result.repair_solve_log)),
                "feasible_repair_solves": int((result.repair_solve_log["feasible"] == True).sum()) if not result.repair_solve_log.empty else 0,  # noqa: E712
                "approx_pareto_points": int(len(pareto)),
                "best_economic_risk": _safe_min(pareto, "economic_risk"),
                "best_environmental_impact": _safe_min(pareto, "environmental_impact"),
                "best_assembly_quality_loss": _safe_min(pareto, "assembly_quality_loss"),
                "incumbent_economic_risk": incumbent.get("economic_risk"),
                "incumbent_environmental_impact": incumbent.get("environmental_impact"),
                "incumbent_assembly_quality_loss": incumbent.get("assembly_quality_loss"),
            }
            path_map[machine_type] = paths
        except Exception as exc:  # noqa: BLE001 - benchmark summary should report failed instances.
            row = {
                "machine_type_id": machine_type,
                "success": False,
                "status_message": f"{type(exc).__name__}: {exc}",
                "solve_seconds": None,
                "wall_seconds": float(time.perf_counter() - start),
                "repair_solves": 0,
                "feasible_repair_solves": 0,
                "approx_pareto_points": 0,
                "best_economic_risk": None,
                "best_environmental_impact": None,
                "best_assembly_quality_loss": None,
                "incumbent_economic_risk": None,
                "incumbent_environmental_impact": None,
                "incumbent_assembly_quality_loss": None,
            }
            path_map[machine_type] = {}
        rows.append(row)
    summary = pd.DataFrame(rows)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = config.results_dir / "large_benchmark_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary, path_map


def _machine_config(config: Stage8Config, machine_type: str) -> Stage8Config:
    processed_dir = config.processed_dir / machine_type
    results_dir = config.results_dir / machine_type
    return dataclasses.replace(
        config,
        machine_type_id=machine_type,
        processed_dir=processed_dir,
        results_dir=results_dir,
        benchmark_suite=None,
    )


def _safe_min(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.min()) if not values.empty else None
