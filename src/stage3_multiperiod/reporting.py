"""Report writers for Stage 3 multi-period runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd

from .checks import model_size_counts, summarize_checks
from .config import Stage3Config
from .io_utils import ensure_output_dirs
from .structures import Stage3Instance, Stage3ModelData, Stage3Solution


def write_stage3_reports(
    instance: Stage3Instance,
    model_data: Stage3ModelData,
    solution: Stage3Solution,
    config: Stage3Config,
) -> Dict[str, str]:
    ensure_output_dirs(config.processed_dir, config.results_dir)
    paths = {
        "instance_summary": config.processed_dir / "instance_summary.json",
        "component_route_period_table": config.processed_dir / "component_route_period_table.csv",
        "period_demand": config.processed_dir / "period_demand.csv",
        "bom_requirements": config.processed_dir / "bom_requirements.csv",
        "model_summary": config.processed_dir / "model_summary.json",
        "solution_json": config.results_dir / "solution_summary.json",
        "solution_checks": config.results_dir / "solution_checks.json",
        "selected_component_routes": config.results_dir / "selected_component_routes.csv",
        "assembly_plan": config.results_dir / "assembly_plan.csv",
        "inventory_trajectory": config.results_dir / "inventory_trajectory.csv",
        "capacity_utilization": config.results_dir / "capacity_utilization.csv",
        "report_md": config.results_dir / "stage3_report.md",
    }
    _write_json(paths["instance_summary"], {"generated_at_utc": _now(), "instance": instance.to_summary_dict()})
    instance.component_route_period_table.to_csv(paths["component_route_period_table"], index=False, encoding="utf-8-sig")
    instance.period_demand.to_csv(paths["period_demand"], index=False, encoding="utf-8-sig")
    instance.bom_requirements.to_csv(paths["bom_requirements"], index=False, encoding="utf-8-sig")
    _write_json(
        paths["model_summary"],
        {
            "generated_at_utc": _now(),
            **model_size_counts(model_data),
            "constraint_names": model_data.constraint_names,
            "config": _config_to_dict(config),
        },
    )
    _write_json(paths["solution_json"], {"generated_at_utc": _now(), "solution": solution.to_json_dict()})
    _write_json(
        paths["solution_checks"],
        {"generated_at_utc": _now(), "summary": summarize_checks(solution.solution_checks), "checks": solution.solution_checks},
    )
    solution.selected_component_routes.to_csv(paths["selected_component_routes"], index=False, encoding="utf-8-sig")
    solution.assembly_plan.to_csv(paths["assembly_plan"], index=False, encoding="utf-8-sig")
    solution.inventory_trajectory.to_csv(paths["inventory_trajectory"], index=False, encoding="utf-8-sig")
    solution.capacity_utilization.to_csv(paths["capacity_utilization"], index=False, encoding="utf-8-sig")
    paths["report_md"].write_text(_markdown_report(instance, model_data, solution), encoding="utf-8")
    return {name: str(path) for name, path in paths.items()}


def _markdown_report(instance: Stage3Instance, model_data: Stage3ModelData, solution: Stage3Solution) -> str:
    counts = model_size_counts(model_data)
    checks = summarize_checks(solution.solution_checks)
    metrics = solution.summary_metrics
    baseline = solution.baseline_comparison
    lines = [
        "# Stage 3 Multi-Period Deterministic MILP Report",
        "",
        f"Generated at UTC: `{_now()}`",
        "",
        "## Instance",
        "",
        f"- Machine type: `{instance.machine_type_id}`",
        f"- Period window: `{instance.periods[0]}` to `{instance.periods[-1]}`",
        f"- Demand units: `{instance.demand_units}`",
        f"- Candidate cores: `{instance.candidate_core_count}`",
        f"- Component instances: `{instance.component_instance_count}`",
        f"- Route ids: `{', '.join(instance.route_ids)}`",
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
        "## Operational Metrics",
        "",
        f"- Accepted cores: `{metrics.get('accepted_core_count')}`",
        f"- Acceptance rate: `{_fmt(metrics.get('acceptance_rate'))}`",
        f"- Selected component routes: `{metrics.get('selected_component_route_count')}`",
        f"- Assembled units: `{_fmt(metrics.get('assembled_units'))}`",
        f"- Final backlog units: `{_fmt(metrics.get('final_backlog_units'))}`",
        f"- Route mix: `{metrics.get('route_mix')}`",
        f"- Average selected quality: `{_fmt(metrics.get('average_selected_quality'))}`",
        f"- Average selected residual life h: `{_fmt(metrics.get('average_selected_residual_life_h'))}`",
        "",
        "## Objective Breakdown",
        "",
        "| Term | Value |",
        "|---|---:|",
    ]
    for key, value in solution.objective_breakdown.items():
        lines.append(f"| `{key}` | {_fmt(value)} |")
    lines.extend(
        [
            "",
            "## Assembly Plan",
            "",
            "| Period | Demand | Assembled | Backlog |",
            "|---|---:|---:|---:|",
        ]
    )
    active_plan = solution.assembly_plan[
        (solution.assembly_plan["demand_units"] > 0) | (solution.assembly_plan["assembled_units"] > 0) | (solution.assembly_plan["backlog_units"] > 0)
    ]
    for row in active_plan.itertuples(index=False):
        lines.append(f"| `{row.period_id}` | {_fmt(row.demand_units)} | {_fmt(row.assembled_units)} | {_fmt(row.backlog_units)} |")
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
            "This Stage 3 run is deterministic and multi-period. It does not include stochastic scenarios, SAA, chance constraints, CVaR, or selective-assembly pairwise compatibility.",
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


def _config_to_dict(config: Stage3Config) -> Dict[str, object]:
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
