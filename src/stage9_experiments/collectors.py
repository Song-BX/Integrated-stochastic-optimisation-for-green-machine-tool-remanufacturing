"""Metric collectors for Stage 9 experiment reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from .config import Stage9Config
from .structures import ExperimentResult


def collect_experiment_results(manifest: pd.DataFrame, config: Stage9Config) -> pd.DataFrame:
    """Collect standardized metrics for all manifest rows."""

    rows: List[ExperimentResult] = []
    for spec in manifest.to_dict(orient="records"):
        rows.append(_collect_one(spec, config))
    return pd.DataFrame([row.to_dict() for row in rows])


def _collect_one(spec: Dict[str, Any], config: Stage9Config) -> ExperimentResult:
    experiment_id = str(spec["experiment_id"])
    group = str(spec["experiment_group"])
    stage = str(spec["model_stage"])
    path = Path(str(spec["source_path"])) if pd.notna(spec.get("source_path")) and spec.get("source_path") else None
    source_type = str(spec.get("source_type", "existing_result"))
    if path is not None and source_type == "optional_run":
        summary_path = path.with_name("summary.json")
        if summary_path.exists():
            path = summary_path
    base = ExperimentResult(
        experiment_id=experiment_id,
        experiment_group=group,
        model_stage=stage,
        description=str(spec.get("description", "")),
        source_path=str(path) if path else None,
        status="missing",
    )
    if path is None or not path.exists():
        base.warning = "Source result is unavailable."
        if source_type == "optional_run":
            base.status = "optional_missing"
        return base
    try:
        if path.suffix.lower() == ".json":
            return _collect_json_result(base, path, spec, config)
        if path.suffix.lower() == ".csv":
            return _collect_csv_result(base, path, spec, config)
    except Exception as exc:  # noqa: BLE001 - collector should keep suite running.
        base.status = "unreadable"
        base.warning = f"{type(exc).__name__}: {exc}"
        return base
    base.status = "unsupported"
    base.warning = f"Unsupported source suffix: {path.suffix}"
    return base


def _collect_json_result(base: ExperimentResult, path: Path, spec: Dict[str, Any], config: Stage9Config) -> ExperimentResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(spec["experiment_id"]).startswith("rule_"):
        return _collect_rule_reference(base, payload, spec)
    solution = payload.get("solution", payload)
    metrics = solution.get("summary_metrics", {}) if isinstance(solution, dict) else {}
    cvar = solution.get("cvar_summary", {}) if isinstance(solution, dict) else {}
    breakdown = solution.get("objective_breakdown", {}) if isinstance(solution, dict) else {}
    flat = payload if isinstance(payload, dict) else {}
    source_status = str((solution.get("status") if isinstance(solution, dict) else flat.get("status")) or flat.get("status") or "")
    base.success = _as_bool(solution.get("success") if isinstance(solution, dict) else flat.get("success"))
    if base.success is None and source_status.lower() == "success":
        base.success = True
    failed_status = "failed" in source_status.lower()
    if base.success is False or failed_status:
        base.status = "failed_run_available"
        base.warning = f"Source run status={source_status or 'unknown'}; metrics retained for audit/appendix only."
    else:
        base.status = "collected"
    base.machine_type_id = _machine_type_from_solution(solution, config)
    base.period_start = config.period_start
    base.period_count = _as_int(metrics.get("period_count")) or config.period_count
    base.scenario_count = _as_int(metrics.get("scenario_count") or flat.get("scenario_count"))
    base.variable_count = _as_int(metrics.get("variable_count"))
    base.constraint_count = _as_int(metrics.get("constraint_count"))
    base.objective_value = _as_float(solution.get("objective_value") if isinstance(solution, dict) else flat.get("objective_value"))
    base.economic_risk = base.objective_value
    base.expected_assembled_units = _as_float(metrics.get("expected_assembled_units") or flat.get("expected_assembled_units"))
    base.expected_final_backlog_units = _as_float(
        metrics.get("expected_final_backlog_units") or flat.get("expected_backlog")
    )
    base.expected_assembly_shortfall_units = _as_float(
        metrics.get("expected_assembly_shortfall_units") or flat.get("assembly_shortfall")
    )
    base.cvar_value = _as_float(
        cvar.get("cvar_value")
        or metrics.get("cvar_value")
        or breakdown.get("cvar_value")
        or flat.get("cvar_value")
    )
    base.eta = _as_float(cvar.get("eta") or metrics.get("var_eta") or flat.get("eta"))
    base.worst_scenario_loss = _as_float(
        cvar.get("worst_scenario_loss") or metrics.get("worst_scenario_loss") or flat.get("worst_scenario_loss")
    )
    base.route_mix_summary = _json_compact(metrics.get("route_mix") or flat.get("route_mix"))
    base.solve_seconds = _as_float(solution.get("solve_seconds") if isinstance(solution, dict) else flat.get("runtime_seconds"))
    if base.solve_seconds is None:
        base.solve_seconds = _as_float(flat.get("runtime_seconds"))
    return base


def _collect_csv_result(base: ExperimentResult, path: Path, spec: Dict[str, Any], config: Stage9Config) -> ExperimentResult:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    experiment_id = str(spec["experiment_id"])
    base.status = "collected"
    base.success = not frame.empty
    base.machine_type_id = config.machine_type_id
    base.period_start = config.period_start
    base.period_count = config.period_count
    if "large_benchmark" in experiment_id:
        return _collect_benchmark_csv(base, frame)
    if "stage7" in experiment_id or "stage8" in experiment_id or "exact_vs" in experiment_id or "pareto" in path.name:
        return _collect_pareto_csv(base, frame, stage=str(spec["model_stage"]))
    base.warning = "CSV source collected without specialized metric mapping."
    return base


def _collect_pareto_csv(base: ExperimentResult, frame: pd.DataFrame, stage: str) -> ExperimentResult:
    feasible = _feasible_rows(frame)
    if feasible.empty:
        base.status = "collected_empty"
        base.success = False
        base.warning = "No feasible Pareto rows were found."
        return base
    base.pareto_points = int(len(feasible))
    base.economic_risk = _min_numeric(feasible, "economic_risk")
    base.environmental_impact = _min_numeric(feasible, "environmental_impact")
    base.assembly_quality_loss = _min_numeric(feasible, "assembly_quality_loss")
    base.objective_value = base.economic_risk
    base.success = True
    base.solve_seconds = _markdown_solve_seconds(Path(str(base.source_path)))
    if stage == "Stage8":
        repair_log = Path(str(base.source_path)).with_name("repair_solve_log.csv")
        if repair_log.exists():
            repairs = pd.read_csv(repair_log, encoding="utf-8-sig")
            base.feasible_repairs = int(_truthy_series(repairs.get("feasible", pd.Series(dtype=object))).sum())
            base.solve_seconds = base.solve_seconds or _sum_numeric(repairs, "solve_seconds")
    return base


def _collect_benchmark_csv(base: ExperimentResult, frame: pd.DataFrame) -> ExperimentResult:
    base.success = bool(_truthy_series(frame.get("success", pd.Series(dtype=object))).all()) if not frame.empty else False
    base.machine_type_id = "top5_52w"
    base.pareto_points = int(pd.to_numeric(frame.get("approx_pareto_points", 0), errors="coerce").fillna(0).sum()) if not frame.empty else 0
    base.feasible_repairs = int(pd.to_numeric(frame.get("feasible_repair_solves", 0), errors="coerce").fillna(0).sum()) if not frame.empty else 0
    base.economic_risk = _min_numeric(frame, "best_economic_risk")
    base.environmental_impact = _min_numeric(frame, "best_environmental_impact")
    base.assembly_quality_loss = _min_numeric(frame, "best_assembly_quality_loss")
    base.solve_seconds = _sum_numeric(frame, "solve_seconds")
    base.wall_seconds = _sum_numeric(frame, "wall_seconds")
    return base


def _collect_rule_reference(base: ExperimentResult, payload: Dict[str, Any], spec: Dict[str, Any]) -> ExperimentResult:
    solution = payload.get("solution", {})
    baseline = solution.get("baseline_comparison", {})
    rule_id = str(spec["model_stage"])
    base.status = "collected"
    base.success = True
    base.machine_type_id = "CK6150"
    base.warning = None
    if rule_id == "BR14":
        base.objective_value = _as_float(baseline.get("risk_baseline_objective"))
        base.cvar_value = _as_float(baseline.get("risk_baseline_cvar_value"))
    elif rule_id == "BR08":
        selective = baseline.get("selective_assembly_rule", {})
        base.expected_assembly_shortfall_units = _as_float(selective.get("expected_assembly_shortfall_units"))
        base.warning = None if selective else "BR08 detailed metrics unavailable in Stage 6 report."
    elif rule_id == "BR18":
        ablation = baseline.get("no_selective_assembly_ablation_rule", {})
        base.expected_assembly_shortfall_units = _as_float(ablation.get("expected_assembly_shortfall_units"))
        base.warning = None if ablation else "BR18 detailed metrics unavailable in Stage 6 report."
    if all(getattr(base, name) is None for name in ["objective_value", "cvar_value", "expected_assembly_shortfall_units"]):
        base.warning = f"{rule_id} metrics were not extractable from Stage 6 report."
    return base


def _feasible_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "feasible" not in frame.columns:
        return frame.copy()
    mask = _truthy_series(frame["feasible"])
    return frame[mask].copy()


def _truthy_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=bool)
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _machine_type_from_solution(solution: Dict[str, Any], config: Stage9Config) -> str:
    metrics = solution.get("summary_metrics", {}) if isinstance(solution, dict) else {}
    return str(metrics.get("machine_type_id") or config.machine_type_id)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = pd.to_numeric(value, errors="coerce")
        return float(parsed) if pd.notna(parsed) else None
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    parsed = _as_float(value)
    return int(parsed) if parsed is not None else None


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return str(value).lower() in ["true", "1", "yes"]


def _min_numeric(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns or frame.empty:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.min()) if not values.empty else None


def _sum_numeric(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns or frame.empty:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.sum()) if not values.empty else None


def _json_compact(value: Any) -> str | None:
    if value in [None, ""]:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _markdown_solve_seconds(source_path: Path) -> float | None:
    candidates = [
        source_path.with_name("stage7_pareto_report.md"),
        source_path.with_name("stage8_matheuristic_report.md"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if "Solve seconds" not in line:
                continue
            value = line.split("`")
            if len(value) >= 2:
                return _as_float(value[1])
    return None
