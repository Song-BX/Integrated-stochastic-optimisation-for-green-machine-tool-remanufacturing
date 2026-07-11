"""Report writers for Stage 4 stochastic runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd

from .checks import model_size_counts, summarize_checks
from .config import Stage4Config
from .io_utils import ensure_output_dirs
from .structures import Stage4Instance, Stage4ModelData, Stage4Solution


def write_stage4_reports(
    instance: Stage4Instance,
    model_data: Stage4ModelData,
    solution: Stage4Solution,
    config: Stage4Config,
) -> Dict[str, str]:
    ensure_output_dirs(config.processed_dir, config.results_dir)
    paths = {
        "instance_summary": config.processed_dir / "instance_summary.json",
        "scenario_sample": config.processed_dir / "scenario_sample.csv",
        "scenario_probability_summary": config.processed_dir / "scenario_probability_summary.json",
        "scenario_demand": config.processed_dir / "scenario_demand.csv",
        "component_route_period_scenario_table": config.processed_dir / "component_route_period_scenario_table.csv",
        "model_summary": config.processed_dir / "model_summary.json",
        "solution_json": config.results_dir / "solution_summary.json",
        "solution_checks": config.results_dir / "solution_checks.json",
        "first_stage_decisions": config.results_dir / "first_stage_decisions.csv",
        "scenario_selected_component_routes": config.results_dir / "scenario_selected_component_routes.csv",
        "scenario_assembly_plan": config.results_dir / "scenario_assembly_plan.csv",
        "scenario_inventory_trajectory": config.results_dir / "scenario_inventory_trajectory.csv",
        "scenario_capacity_utilization": config.results_dir / "scenario_capacity_utilization.csv",
        "report_md": config.results_dir / "stage4_saa_report.md",
    }
    _write_json(paths["instance_summary"], {"generated_at_utc": _now(), "instance": instance.to_summary_dict()})
    instance.scenario_sample.to_csv(paths["scenario_sample"], index=False, encoding="utf-8-sig")
    _write_json(paths["scenario_probability_summary"], {"generated_at_utc": _now(), **instance.scenario_probability_summary})
    instance.scenario_demand.to_csv(paths["scenario_demand"], index=False, encoding="utf-8-sig")
    instance.component_route_period_scenario_table.to_csv(paths["component_route_period_scenario_table"], index=False, encoding="utf-8-sig")
    _write_json(
        paths["model_summary"],
        {
            "generated_at_utc": _now(),
            **model_size_counts(model_data),
            "constraint_names": model_data.constraint_names,
            "config": _config_to_dict(config),
            "scenario_probability_summary": instance.scenario_probability_summary,
        },
    )
    _write_json(paths["solution_json"], {"generated_at_utc": _now(), "solution": solution.to_json_dict()})
    _write_json(
        paths["solution_checks"],
        {"generated_at_utc": _now(), "summary": summarize_checks(solution.solution_checks), "checks": solution.solution_checks},
    )
    solution.first_stage_decisions.to_csv(paths["first_stage_decisions"], index=False, encoding="utf-8-sig")
    solution.scenario_selected_component_routes.to_csv(paths["scenario_selected_component_routes"], index=False, encoding="utf-8-sig")
    solution.scenario_assembly_plan.to_csv(paths["scenario_assembly_plan"], index=False, encoding="utf-8-sig")
    solution.scenario_inventory_trajectory.to_csv(paths["scenario_inventory_trajectory"], index=False, encoding="utf-8-sig")
    solution.scenario_capacity_utilization.to_csv(paths["scenario_capacity_utilization"], index=False, encoding="utf-8-sig")
    paths["report_md"].write_text(_markdown_report(instance, model_data, solution), encoding="utf-8")
    return {name: str(path) for name, path in paths.items()}


def _markdown_report(instance: Stage4Instance, model_data: Stage4ModelData, solution: Stage4Solution) -> str:
    counts = model_size_counts(model_data)
    checks = summarize_checks(solution.solution_checks)
    metrics = solution.summary_metrics
    baseline = solution.baseline_comparison
    sample = instance.scenario_sample[["scenario_id", "scenario_name", "macro_group", "scenario_probability", "saa_probability"]]
    scenario_summary = (
        solution.scenario_assembly_plan.groupby("scenario_id")
        .agg(
            demand_units=("demand_units", "sum"),
            assembled_units=("assembled_units", "sum"),
            backlog_units=("backlog_units", "last"),
        )
        .reset_index()
        if not solution.scenario_assembly_plan.empty
        else pd.DataFrame()
    )
    lines = [
        "# Stage 4 Stochastic SAA MILP Report",
        "",
        f"Generated at UTC: `{_now()}`",
        "",
        "## Instance",
        "",
        f"- Machine type: `{instance.machine_type_id}`",
        f"- Period window: `{instance.periods[0]}` to `{instance.periods[-1]}`",
        f"- Scenario count: `{len(instance.scenario_ids)}`",
        f"- Demand units expected: `{_fmt(instance.demand_units_expected)}`",
        f"- Candidate cores: `{instance.candidate_core_count}`",
        f"- Component instances: `{instance.component_instance_count}`",
        f"- Route ids: `{', '.join(instance.route_ids)}`",
        "",
        "## Scenario Sample",
        "",
        "| Scenario | Macro group | Raw prob | SAA prob |",
        "|---|---|---:|---:|",
    ]
    for row in sample.itertuples(index=False):
        lines.append(f"| `{row.scenario_id}` | `{row.macro_group}` | {_fmt(row.scenario_probability)} | {_fmt(row.saa_probability)} |")
    lines.extend(
        [
            "",
            "## Model Size",
            "",
            f"- Variables: `{counts['variable_count']}`",
            f"- Constraints: `{counts['constraint_count']}`",
            f"- Binary variables: `{counts['binary_variable_count']}`",
            f"- General integer variables: `{counts['general_integer_variable_count']}`",
            "",
            "## Solver Result",
            "",
            f"- Success: `{solution.success}`",
            f"- Status: `{solution.status}`",
            f"- Message: `{solution.status_message}`",
            f"- Objective: `{_fmt(solution.objective_value)}`",
            f"- MIP gap: `{_fmt(solution.mip_gap)}`",
            f"- Solve seconds: `{solution.solve_seconds:.3f}`",
            f"- Solution checks: `{checks}`",
            "",
            "## First Stage",
            "",
            f"- Accepted cores: `{metrics.get('accepted_core_count')}`",
            f"- Acceptance rate: `{_fmt(metrics.get('acceptance_rate'))}`",
            "",
            "## Scenario Performance",
            "",
            "| Scenario | Demand | Assembled | Backlog |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in scenario_summary.itertuples(index=False):
        lines.append(f"| `{row.scenario_id}` | {_fmt(row.demand_units)} | {_fmt(row.assembled_units)} | {_fmt(row.backlog_units)} |")
    lines.extend(
        [
            "",
            "## Objective Breakdown",
            "",
            "| Term | Value |",
            "|---|---:|",
        ]
    )
    for key, value in solution.objective_breakdown.items():
        lines.append(f"| `{key}` | {_fmt(value)} |")
    lines.extend(
        [
            "",
            "## Baseline Comparison",
            "",
            f"- Baseline rule: `{baseline.get('baseline_rule_id')}` / `{baseline.get('baseline_rule_name')}`",
            f"- Baseline estimated assembly units: `{baseline.get('estimated_assembly_units')}`",
            f"- Baseline estimated backlog units: `{baseline.get('estimated_backlog_units')}`",
            f"- Baseline route mix: `{baseline.get('route_mix')}`",
            "",
            "## Stage Boundary",
            "",
            "This Stage 4 run is a deterministic equivalent SAA model. It does not include chance constraints, CVaR, selective-assembly pairwise compatibility, Pareto optimisation, or matheuristics.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _config_to_dict(config: Stage4Config) -> Dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in config.__dict__.items()}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):,.4f}"
    except (TypeError, ValueError):
        return str(value)
