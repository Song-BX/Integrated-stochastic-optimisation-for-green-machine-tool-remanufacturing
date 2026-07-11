"""Report writers for Stage 2 deterministic MILP runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd

from .checks import model_size_counts, summarize_checks
from .config import Stage2Config
from .io_utils import ensure_output_dirs
from .structures import Stage2Instance, Stage2ModelData, Stage2Solution


def write_stage2_reports(
    instance: Stage2Instance,
    model_data: Stage2ModelData,
    solution: Stage2Solution,
    config: Stage2Config,
) -> Dict[str, str]:
    """Write processed model artefacts and human-readable result reports."""

    ensure_output_dirs(config.processed_dir, config.results_dir)
    paths = {
        "instance_summary": config.processed_dir / "instance_summary.json",
        "core_route_table": config.processed_dir / "core_route_table.csv",
        "route_coefficients": config.processed_dir / "route_coefficients.csv",
        "model_summary": config.processed_dir / "model_summary.json",
        "solution_json": config.results_dir / "solution_summary.json",
        "solution_checks": config.results_dir / "solution_checks.json",
        "selected_routes": config.results_dir / "selected_routes.csv",
        "capacity_utilization": config.results_dir / "capacity_utilization.csv",
        "objective_terms": config.results_dir / "objective_terms.csv",
        "report_md": config.results_dir / "stage2_report.md",
    }

    _write_json(paths["instance_summary"], {"generated_at_utc": _now(), "instance": instance.to_summary_dict()})
    instance.core_route_table.to_csv(paths["core_route_table"], index=False, encoding="utf-8-sig")
    instance.route_coefficients.to_csv(paths["route_coefficients"], index=False, encoding="utf-8-sig")
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
        {
            "generated_at_utc": _now(),
            "summary": summarize_checks(solution.solution_checks),
            "checks": solution.solution_checks,
        },
    )
    solution.selected_routes.to_csv(paths["selected_routes"], index=False, encoding="utf-8-sig")
    solution.capacity_utilization.to_csv(paths["capacity_utilization"], index=False, encoding="utf-8-sig")
    model_data.objective_terms.to_csv(paths["objective_terms"], index=False, encoding="utf-8-sig")
    paths["report_md"].write_text(_markdown_report(instance, model_data, solution, config), encoding="utf-8")
    return {name: str(path) for name, path in paths.items()}


def _markdown_report(
    instance: Stage2Instance,
    model_data: Stage2ModelData,
    solution: Stage2Solution,
    config: Stage2Config,
) -> str:
    metrics = solution.summary_metrics
    baseline = solution.baseline_comparison
    counts = model_size_counts(model_data)
    check_summary = summarize_checks(solution.solution_checks)
    lines = [
        "# Stage 2 Deterministic MILP Report",
        "",
        f"Generated at UTC: `{_now()}`",
        "",
        "## Instance",
        "",
        f"- Machine type: `{instance.machine_type_id}`",
        f"- Demand units: `{instance.demand_units}`",
        f"- Candidate returned cores: `{instance.candidate_core_count}`",
        f"- Route classes: `{', '.join(instance.route_classes)}`",
        f"- Capacity share: `{instance.machine_summary.get('capacity_share'):.4f}`",
        "",
        "## Model Size",
        "",
        f"- Variables: `{counts['variable_count']}`",
        f"- Constraints: `{counts['constraint_count']}`",
        f"- Binary variables: `{counts['binary_variable_count']}`",
        f"- General integer variables: `{counts['general_integer_variable_count']}`",
        f"- Integer variables: `{counts['integer_variable_count']}`",
        "",
        "## Solver Result",
        "",
        f"- Success: `{solution.success}`",
        f"- Status: `{solution.status}`",
        f"- Message: `{solution.status_message}`",
        f"- Objective: `{_fmt(solution.objective_value)}`",
        f"- MIP gap: `{_fmt(solution.mip_gap)}`",
        f"- Solve seconds: `{solution.solve_seconds:.3f}`",
        "",
        "## Operational Metrics",
        "",
        f"- Accepted cores: `{metrics.get('accepted_core_count')}`",
        f"- Acceptance rate: `{_fmt(metrics.get('acceptance_rate'))}`",
        f"- Procurement units: `{_fmt(metrics.get('procurement_units'))}`",
        f"- Shortage units: `{_fmt(metrics.get('shortage_units'))}`",
        f"- Demand fill ratio: `{_fmt(metrics.get('demand_fill_ratio'))}`",
        f"- Route mix: `{metrics.get('route_mix')}`",
        f"- Average selected quality: `{_fmt(metrics.get('average_selected_quality'))}`",
        f"- Average selected residual life h: `{_fmt(metrics.get('average_selected_residual_life_h'))}`",
        f"- Solution checks: `{check_summary}`",
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
            "## Capacity Utilization",
            "",
            "| Resource | Used h | Regular h | Overtime h | Utilization |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in solution.capacity_utilization.itertuples(index=False):
        lines.append(
            f"| `{row.resource_type}` | {_fmt(row.used_hours)} | {_fmt(row.available_regular_hours)} | "
            f"{_fmt(row.overtime_hours)} | {_fmt(row.utilization_rate_regular)} |"
        )

    lines.extend(
        [
            "",
            "## Baseline Comparison",
            "",
            f"- Baseline rule: `{baseline.get('baseline_rule_id')}` / `{baseline.get('baseline_rule_name')}`",
            f"- Baseline status: `{baseline.get('status')}`",
            f"- Baseline objective: `{_fmt(baseline.get('objective_value'))}`",
            f"- Gap vs MILP (%): `{_fmt(baseline.get('gap_vs_milp_pct'))}`",
            f"- Baseline route mix: `{baseline.get('route_mix')}`",
            "",
            "## Stage Boundary",
            "",
            "This Stage 2 run is deterministic, single-period and aggregate. It does not include SAA, chance constraints, CVaR, pairwise selective assembly or explicit multi-period inventory flow.",
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


def _config_to_dict(config: Stage2Config) -> Dict[str, object]:
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
