"""Final experiment completion logic for Stage 12."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from stage4_stochastic.aggregation import build_stage4_instance
from stage4_stochastic.config import Stage4Config, scenario_ids_for_count
from stage4_stochastic.io_utils import read_stage4_tables, require_stage1_passed
from stage4_stochastic.model import build_model_data as build_stage4_model_data
from stage4_stochastic.reporting import write_stage4_reports
from stage4_stochastic.solver import solve_model as solve_stage4_model
from stage5_risk_averse.aggregation import build_stage5_instance
from stage5_risk_averse.config import Stage5Config
from stage5_risk_averse.io_utils import read_stage5_tables
from stage5_risk_averse.model import build_model_data as build_stage5_model_data
from stage5_risk_averse.reporting import write_stage5_reports
from stage5_risk_averse.solver import solve_model as solve_stage5_model
from stage6_selective_assembly.aggregation import build_stage6_instance
from stage6_selective_assembly.config import Stage6Config
from stage6_selective_assembly.io_utils import read_stage6_tables
from stage6_selective_assembly.model import build_model_data as build_stage6_model_data
from stage6_selective_assembly.reporting import write_stage6_reports
from stage6_selective_assembly.solver import solve_model as solve_stage6_model
from stage8_matheuristic.benchmark import run_single_instance
from stage8_matheuristic.config import Stage8Config
from stage8_matheuristic.io_utils import read_stage6_tables as read_stage8_tables, require_stage1_passed as require_stage8_stage1_passed
from stage9_experiments.config import Stage9Config
from stage9_experiments.manifest import build_experiment_manifest

from .config import Stage12Config
from .structures import CompletionResult, CompletionSpec


SAA_COUNTS = (18, 27)
SENSITIVITY_AXES = {
    "cvar_lambda": [0.00, 0.11, 0.22, 0.44],
    "min_system_reliability": [0.90, 0.93, 0.95],
    "env_weight": [60, 120, 240],
    "assembly_shortfall_penalty_rmb": [125000, 250000, 500000],
}


def build_completion_manifest(config: Stage12Config) -> pd.DataFrame:
    """Build a final completion manifest from the Stage 9 experiment manifest."""

    stage9_config = _stage9_config(config)
    stage9_manifest = build_experiment_manifest(stage9_config)
    specs: List[CompletionSpec] = []
    seen_completion_ids: set[str] = set()
    for row in stage9_manifest.to_dict(orient="records"):
        experiment_id = str(row["experiment_id"])
        group = str(row["experiment_group"])
        source_path = Path(str(row.get("source_path", "")))
        source_type = str(row.get("source_type", "existing_result"))
        if source_type != "optional_run":
            continue
        if group == "saa_stability":
            count = _saa_count_from_experiment(experiment_id)
            target_path = _stage9_runs_dir(config) / f"saa_scenario_{count}" / "solution_summary.json"
            summary_path = target_path.with_name("summary.json")
            existing_path = summary_path if summary_path.exists() else target_path
            action = "collect-existing" if existing_path.exists() else "run-stage4-saa"
            status = "already_available" if existing_path.exists() else "planned"
            _append_spec(
                specs,
                seen_completion_ids,
                CompletionSpec(
                    completion_id=f"saa_scenario_{count}",
                    experiment_group="saa_stability",
                    source_experiment_id=f"saa_scenario_{count}_optional",
                    target_path=str(existing_path),
                    action=action,
                    status=status,
                    reason=f"{'Collect existing' if target_path.exists() else 'Run bounded'} Stage 4 SAA with {count} scenarios.",
                )
            )
        elif group == "sensitivity_analysis":
            axis, level = _parse_sensitivity_experiment(experiment_id)
            target_path = _stage9_runs_dir(config) / f"sensitivity_{axis}_{level}" / "summary.json"
            action = "collect-existing" if target_path.exists() else "run-sensitivity"
            status = "already_available" if target_path.exists() else "planned"
            _append_spec(
                specs,
                seen_completion_ids,
                CompletionSpec(
                    completion_id=f"sensitivity_{axis}_{level}",
                    experiment_group="sensitivity_analysis",
                    source_experiment_id=experiment_id,
                    target_path=str(target_path),
                    action=action,
                    status=status,
                    reason=f"{'Collect existing' if target_path.exists() else 'Run bounded'} sensitivity profile for {axis}={level}.",
                )
            )
        elif source_path.exists():
            _append_spec(
                specs,
                seen_completion_ids,
                CompletionSpec(
                    completion_id=f"complete_{experiment_id}",
                    experiment_group=group,
                    source_experiment_id=experiment_id,
                    target_path=str(source_path),
                    action="collect-existing",
                    status="already_available",
                    reason="Optional result already exists.",
                )
            )
        else:
            _append_spec(
                specs,
                seen_completion_ids,
                CompletionSpec(
                    completion_id=f"gap_{experiment_id}",
                    experiment_group=group,
                    source_experiment_id=experiment_id,
                    target_path=str(source_path),
                    action="register-gap",
                    status="planned",
                    reason="Optional completion item is not yet materialized.",
                )
            )
    _append_spec(
        specs,
        seen_completion_ids,
        CompletionSpec(
            completion_id="quick_stage8_matheuristic_validation",
            experiment_group="exact_vs_matheuristic",
            source_experiment_id="quick_stage8_matheuristic",
            target_path=str(config.results_dir / "runs" / "quick_stage8_matheuristic" / "approx_pareto_front.csv"),
            action="run-stage8-quick",
            status="planned",
            reason="Quick repair run validates bounded completion without touching canonical Stage 8 outputs.",
        )
    )
    return pd.DataFrame([spec.to_dict() for spec in specs])


def complete_missing_experiments(config: Stage12Config, root: Path, manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute supported completion items and record unsupported gaps."""

    results: List[CompletionResult] = []
    gaps: List[dict[str, object]] = []
    for row in manifest.to_dict(orient="records"):
        action = str(row["action"])
        if action == "collect-existing":
            results.append(_available_result(row))
            continue
        if action == "run-stage8-quick":
            result = _run_stage8_quick(config, root, row)
            results.append(result)
            if not result.success:
                gaps.append(_gap_row(row, "completion_failed", result.message or "Stage 8 quick validation failed."))
            continue
        if action == "run-stage4-saa":
            result = _run_stage4_saa(config, root, row)
            results.append(result)
            if not result.success:
                gaps.append(_gap_row(row, "completion_failed", result.message or "Stage 4 SAA completion failed."))
            continue
        if action == "run-sensitivity":
            result = _run_sensitivity(config, root, row)
            results.append(result)
            if not result.success:
                gaps.append(_gap_row(row, "completion_failed", result.message or "Sensitivity completion failed."))
            continue
        message = str(row.get("reason", "Unsupported completion item."))
        results.append(
            CompletionResult(
                completion_id=str(row["completion_id"]),
                experiment_group=str(row["experiment_group"]),
                source_experiment_id=str(row["source_experiment_id"]),
                action=action,
                status="blocking_gap",
                success=False,
                output_path=str(row["target_path"]),
                message=message,
                affects_main_claim=str(row["experiment_group"]) in {"saa_stability", "sensitivity_analysis"},
            )
        )
        gaps.append(
            _gap_row(row, "unsupported_quick_completion", message)
        )
    return pd.DataFrame([result.to_dict() for result in results]), pd.DataFrame(gaps)


def _available_result(row: dict[str, object]) -> CompletionResult:
    success = True
    status = "already_available"
    message = str(row.get("reason", ""))
    target = Path(str(row["target_path"]))
    if target.name == "summary.json" and target.exists():
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
            success = bool(payload.get("success", False))
            if not success:
                status = "available_failed"
                message = f"Existing summary reports status={payload.get('status', 'unknown')}."
        except json.JSONDecodeError:
            success = False
            status = "available_unreadable"
            message = "Existing summary.json could not be parsed."
    return CompletionResult(
        completion_id=str(row["completion_id"]),
        experiment_group=str(row["experiment_group"]),
        source_experiment_id=str(row["source_experiment_id"]),
        action=str(row["action"]),
        status=status,
        success=success,
        output_path=str(row["target_path"]),
        message=message,
    )


def _run_stage8_quick(config: Stage12Config, root: Path, row: dict[str, object]) -> CompletionResult:
    start = time.perf_counter()
    try:
        run_results = config.results_dir / "runs" / "quick_stage8_matheuristic"
        run_processed = config.processed_dir / "runs" / "quick_stage8_matheuristic"
        stage8_config = Stage8Config(
            raw_dir=root / "data/raw",
            stage1_report=config.stage1_report,
            processed_dir=run_processed,
            results_dir=run_results,
            stage4_results_dir=config.data_results_dir / "stage4",
            stage5_results_dir=config.data_results_dir / "stage5",
            stage6_results_dir=config.data_results_dir / "stage6",
            stage7_results_dir=config.data_results_dir / "stage7",
            machine_type_id=config.machine_type_id,
            period_start=config.period_start,
            period_count=config.period_count,
            processing_window_periods=config.processing_window_periods,
            epsilon_grid_size=config.quick_epsilon_grid_size,
            max_iterations=config.quick_max_iterations,
            repair_time_limit=config.quick_repair_time_limit,
        )
        require_stage8_stage1_passed(stage8_config.stage1_report)
        tables = read_stage8_tables(stage8_config.raw_dir)
        result, _paths = run_single_instance(tables, stage8_config)
        if result.success:
            _write_json_summary(run_results / "summary.json", {
                "status": "success",
                "success": True,
                "objective_value": result.incumbent_solution_summary.get("objective_value"),
                "expected_backlog": result.incumbent_solution_summary.get("expected_final_backlog_units"),
                "approx_pareto_points": int(len(result.approx_pareto_front)),
                "runtime_seconds": float(time.perf_counter() - start),
                "route_mix": result.incumbent_solution_summary.get("route_mix_summary"),
            })
        return CompletionResult(
            completion_id=str(row["completion_id"]),
            experiment_group=str(row["experiment_group"]),
            source_experiment_id=str(row["source_experiment_id"]),
            action=str(row["action"]),
            status="completed" if result.success else "completed_failed",
            success=bool(result.success),
            output_path=str(run_results / "approx_pareto_front.csv"),
            seconds=time.perf_counter() - start,
            message=result.status_message,
        )
    except Exception as exc:  # noqa: BLE001
        _write_json_summary(config.results_dir / "runs" / "quick_stage8_matheuristic" / "summary.json", {
            "status": "failed",
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
        })
        return CompletionResult(
            completion_id=str(row["completion_id"]),
            experiment_group=str(row["experiment_group"]),
            source_experiment_id=str(row["source_experiment_id"]),
            action=str(row["action"]),
            status="failed",
            success=False,
            output_path=str(row["target_path"]),
            seconds=time.perf_counter() - start,
            message=f"{type(exc).__name__}: {exc}",
            affects_main_claim=True,
        )


def _run_stage4_saa(config: Stage12Config, root: Path, row: dict[str, object]) -> CompletionResult:
    start = time.perf_counter()
    count = _saa_count_from_experiment(row["source_experiment_id"])
    try:
        run_results = _stage9_runs_dir(config) / f"saa_scenario_{count}"
        run_processed = _stage9_processed_runs_dir(config) / f"saa_scenario_{count}"
        stage4_config = Stage4Config(
            raw_dir=root / "data/raw",
            stage1_report=config.stage1_report,
            processed_dir=run_processed,
            results_dir=run_results,
            machine_type_id=config.machine_type_id,
            period_start=config.period_start,
            period_count=config.period_count,
            processing_window_periods=config.processing_window_periods,
            scenario_mode=f"macro_probability_representative_{count}",
            scenario_ids=_select_scenario_ids(root, config, count),
            baseline_rule_id="BR02",
            time_limit_seconds=config.quick_saa_time_limit,
        ).resolved(root)
        require_stage1_passed(stage4_config.stage1_report)
        tables = read_stage4_tables(stage4_config.raw_dir)
        instance = build_stage4_instance(tables, stage4_config)
        model_data = build_stage4_model_data(instance, stage4_config)
        solution = solve_stage4_model(instance, model_data, stage4_config, tables)
        write_stage4_reports(instance, model_data, solution, stage4_config)
        _write_json_summary(run_results / "summary.json", {
            "status": "success" if solution.success else "completed_failed",
            "success": bool(solution.success),
            "objective_value": solution.objective_value,
            "expected_backlog": solution.summary_metrics.get("expected_final_backlog_units"),
            "route_mix": solution.summary_metrics.get("route_mix"),
            "runtime_seconds": float(solution.solve_seconds or (time.perf_counter() - start)),
            "scenario_count": count,
            "saa_probability_sum": float(instance.scenario_probability_summary.get("saa_probability_sum", 0.0)),
        })
        return CompletionResult(
            completion_id=str(row["completion_id"]),
            experiment_group=str(row["experiment_group"]),
            source_experiment_id=str(row["source_experiment_id"]),
            action=str(row["action"]),
            status="completed" if solution.success else "completed_failed",
            success=bool(solution.success),
            output_path=str(run_results / "solution_summary.json"),
            seconds=time.perf_counter() - start,
            message=solution.status_message,
        )
    except Exception as exc:  # noqa: BLE001
        _write_json_summary(run_results / "summary.json", {
            "status": "failed",
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "scenario_count": count,
        })
        return CompletionResult(
            completion_id=str(row["completion_id"]),
            experiment_group=str(row["experiment_group"]),
            source_experiment_id=str(row["source_experiment_id"]),
            action=str(row["action"]),
            status="failed",
            success=False,
            output_path=str(row["target_path"]),
            seconds=time.perf_counter() - start,
            message=f"{type(exc).__name__}: {exc}",
            affects_main_claim=True,
        )


def _run_sensitivity(config: Stage12Config, root: Path, row: dict[str, object]) -> CompletionResult:
    start = time.perf_counter()
    experiment_id = str(row["source_experiment_id"])
    parameter, value = _parse_sensitivity_experiment(experiment_id)
    try:
        run_results = _stage9_runs_dir(config) / f"sensitivity_{parameter}_{value}"
        run_processed = _stage9_processed_runs_dir(config) / f"sensitivity_{parameter}_{value}"
        # Use a bounded Stage 6 run when possible, otherwise Stage 5 as fallback.
        if parameter in {"cvar_lambda", "min_system_reliability", "assembly_shortfall_penalty_rmb", "env_weight"}:
            summary = _run_stage6_sensitivity(root, config, parameter, value, run_results, run_processed)
        else:
            summary = _run_stage5_sensitivity(root, config, parameter, value, run_results, run_processed)
        _write_json_summary(run_results / "summary.json", summary)
        return CompletionResult(
            completion_id=str(row["completion_id"]),
            experiment_group=str(row["experiment_group"]),
            source_experiment_id=experiment_id,
            action=str(row["action"]),
            status="completed",
            success=bool(summary.get("success", False)),
            output_path=str(run_results / "summary.json"),
            seconds=float(time.perf_counter() - start),
            message="Sensitivity summary generated.",
        )
    except Exception as exc:  # noqa: BLE001
        _write_json_summary(run_results / "summary.json", {
            "status": "failed",
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "parameter": parameter,
            "value": value,
        })
        return CompletionResult(
            completion_id=str(row["completion_id"]),
            experiment_group=str(row["experiment_group"]),
            source_experiment_id=experiment_id,
            action=str(row["action"]),
            status="failed",
            success=False,
            output_path=str(run_results / "summary.json"),
            seconds=float(time.perf_counter() - start),
            message=f"{type(exc).__name__}: {exc}",
            affects_main_claim=True,
        )


def _run_stage6_sensitivity(
    root: Path,
    config: Stage12Config,
    parameter: str,
    value: str,
    run_results: Path,
    run_processed: Path,
) -> dict[str, object]:
    stage6_config = Stage6Config(
        raw_dir=root / "data/raw",
        stage1_report=config.stage1_report,
        processed_dir=run_processed,
        results_dir=run_results,
        stage4_results_dir=config.data_results_dir / "stage4",
        stage5_results_dir=config.data_results_dir / "stage5",
        machine_type_id=config.machine_type_id,
        period_start=config.period_start,
        period_count=config.period_count,
        processing_window_periods=config.processing_window_periods,
        cvar_lambda=float(value) if parameter == "cvar_lambda" else 0.22,
        min_system_reliability=float(value) if parameter == "min_system_reliability" else None,
        env_weight=float(value) if parameter == "env_weight" else 120.0,
        assembly_shortfall_penalty_rmb=float(value) if parameter == "assembly_shortfall_penalty_rmb" else 250000.0,
        time_limit_seconds=config.quick_sensitivity_time_limit,
    ).resolved(root)
    require_stage1_passed(stage6_config.stage1_report)
    tables = read_stage6_tables(stage6_config.raw_dir)
    instance = build_stage6_instance(tables, stage6_config)
    model_data = build_stage6_model_data(instance, stage6_config)
    solution = solve_stage6_model(instance, model_data, stage6_config, tables)
    write_stage6_reports(instance, model_data, solution, stage6_config)
    return {
        "status": "success" if solution.success else "completed_failed",
        "success": bool(solution.success),
        "parameter": parameter,
        "value": value,
        "objective_value": solution.objective_value,
        "expected_backlog": solution.summary_metrics.get("expected_final_backlog_units"),
        "cvar_value": solution.cvar_summary.get("cvar_value"),
        "assembly_shortfall": solution.summary_metrics.get("expected_assembly_shortfall_units"),
        "route_mix": solution.summary_metrics.get("route_mix"),
        "runtime_seconds": float(solution.solve_seconds or 0.0),
    }


def _run_stage5_sensitivity(
    root: Path,
    config: Stage12Config,
    parameter: str,
    value: str,
    run_results: Path,
    run_processed: Path,
) -> dict[str, object]:
    stage5_config = Stage5Config(
        raw_dir=root / "data/raw",
        stage1_report=config.stage1_report,
        processed_dir=run_processed,
        results_dir=run_results,
        stage4_results_dir=config.data_results_dir / "stage4",
        machine_type_id=config.machine_type_id,
        period_start=config.period_start,
        period_count=config.period_count,
        processing_window_periods=config.processing_window_periods,
        cvar_lambda=float(value) if parameter == "cvar_lambda" else 0.22,
        min_system_reliability=float(value) if parameter == "min_system_reliability" else None,
        env_weight=float(value) if parameter == "env_weight" else 120.0,
        time_limit_seconds=config.quick_sensitivity_time_limit,
    ).resolved(root)
    require_stage1_passed(stage5_config.stage1_report)
    tables = read_stage5_tables(stage5_config.raw_dir)
    instance = build_stage5_instance(tables, stage5_config)
    model_data = build_stage5_model_data(instance, stage5_config)
    solution = solve_stage5_model(instance, model_data, stage5_config, tables)
    write_stage5_reports(instance, model_data, solution, stage5_config)
    return {
        "status": "success" if solution.success else "completed_failed",
        "success": bool(solution.success),
        "parameter": parameter,
        "value": value,
        "objective_value": solution.objective_value,
        "expected_backlog": solution.summary_metrics.get("expected_final_backlog_units"),
        "cvar_value": solution.cvar_summary.get("cvar_value"),
        "route_mix": solution.summary_metrics.get("route_mix"),
        "runtime_seconds": float(solution.solve_seconds or 0.0),
    }


def _select_scenario_ids(root: Path, config: Stage12Config, count: int) -> tuple[str, ...]:
    tables = read_stage4_tables(root / "data/raw")
    ids = scenario_ids_for_count(tables["scenarios"], count)
    return ids


def _append_spec(specs: List[CompletionSpec], seen: set[str], spec: CompletionSpec) -> None:
    if spec.completion_id in seen:
        return
    seen.add(spec.completion_id)
    specs.append(spec)


def _stage9_runs_dir(config: Stage12Config) -> Path:
    return config.data_results_dir / "stage9" / "runs"


def _stage9_processed_runs_dir(config: Stage12Config) -> Path:
    return config.data_processed_dir / "stage9" / "runs"


def _stage9_config(config: Stage12Config) -> Stage9Config:
    return Stage9Config(
        stage1_report=config.stage1_report,
        processed_dir=config.data_processed_dir / "stage9",
        results_dir=config.data_results_dir / "stage9",
        data_results_dir=config.data_results_dir,
        data_processed_dir=config.data_processed_dir,
        profile=config.profile,
        execution_mode="collect-existing",
        machine_type_id=config.machine_type_id,
        period_start=config.period_start,
        period_count=config.period_count,
        processing_window_periods=config.processing_window_periods,
        run_epsilon_grid_size=config.quick_epsilon_grid_size,
        run_max_iterations=config.quick_max_iterations,
        run_repair_time_limit=config.quick_repair_time_limit,
    )


def _parse_sensitivity_experiment(experiment_id: object) -> tuple[str, str]:
    text = str(experiment_id)
    for parameter in SENSITIVITY_AXES:
        prefix = f"sensitivity_{parameter}_"
        if text.startswith(prefix):
            return parameter, text[len(prefix):]
    raise ValueError(f"Unrecognized sensitivity experiment id: {text}")


def _saa_count_from_experiment(experiment_id: object) -> int:
    text = str(experiment_id)
    for count in SAA_COUNTS:
        if f"saa_scenario_{count}" in text:
            return count
    raise ValueError(f"Unrecognized SAA experiment id: {text}")


def _gap_row(row: dict[str, object], gap_type: str, reason: str) -> dict[str, object]:
    return {
        "completion_id": row["completion_id"],
        "experiment_group": row["experiment_group"],
        "source_experiment_id": row["source_experiment_id"],
        "target_path": row["target_path"],
        "gap_type": gap_type,
        "affects_main_claim": str(row["experiment_group"]) in {"saa_stability", "sensitivity_analysis"},
        "reason": reason,
        "recommended_follow_up": _follow_up_for_group(str(row["experiment_group"])),
    }


def _follow_up_for_group(group: str) -> str:
    if group == "saa_stability":
        return "Add Stage 4 scenario-count profiles for 18/27 scenarios, then rerun into Stage 9 run directories."
    if group == "sensitivity_analysis":
        return "Add bounded Stage 5/6 sensitivity runners for the manifest axes, each writing isolated Stage 9 summaries."
    return "Add a bounded completion runner for this optional experiment group."


def _write_json_summary(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
