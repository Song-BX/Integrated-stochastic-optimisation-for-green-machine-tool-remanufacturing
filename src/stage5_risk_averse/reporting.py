"""Report writers for Stage 5 risk-averse runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd

from .checks import model_size_counts, summarize_checks
from .config import Stage5Config
from .io_utils import ensure_output_dirs
from .structures import Stage5Instance, Stage5ModelData, Stage5Solution


def write_stage5_reports(
    instance: Stage5Instance,
    model_data: Stage5ModelData,
    solution: Stage5Solution,
    config: Stage5Config,
) -> Dict[str, str]:
    ensure_output_dirs(config.processed_dir, config.results_dir)
    paths = {
        "instance_summary": config.processed_dir / "instance_summary.json",
        "scenario_sample": config.processed_dir / "scenario_sample.csv",
        "component_route_reliability": config.processed_dir / "component_route_reliability.csv",
        "component_route_risk": config.processed_dir / "component_route_risk.csv",
        "component_route_period_scenario_risk_table": config.processed_dir / "component_route_period_scenario_risk_table.csv",
        "model_summary": config.processed_dir / "model_summary.json",
        "solution_json": config.results_dir / "solution_summary.json",
        "solution_checks": config.results_dir / "solution_checks.json",
        "first_stage_decisions": config.results_dir / "first_stage_decisions.csv",
        "scenario_selected_component_routes": config.results_dir / "scenario_selected_component_routes.csv",
        "scenario_assembly_plan": config.results_dir / "scenario_assembly_plan.csv",
        "scenario_inventory_trajectory": config.results_dir / "scenario_inventory_trajectory.csv",
        "scenario_capacity_utilization": config.results_dir / "scenario_capacity_utilization.csv",
        "chance_constraint_report": config.results_dir / "chance_constraint_report.csv",
        "scenario_risk_metrics": config.results_dir / "scenario_risk_metrics.csv",
        "cvar_summary": config.results_dir / "cvar_summary.json",
        "report_md": config.results_dir / "stage5_risk_averse_report.md",
    }
    _write_json(paths["instance_summary"], {"generated_at_utc": _now(), "instance": instance.to_summary_dict()})
    instance.scenario_sample.to_csv(paths["scenario_sample"], index=False, encoding="utf-8-sig")
    instance.component_route_reliability.to_csv(paths["component_route_reliability"], index=False, encoding="utf-8-sig")
    instance.component_route_risk.to_csv(paths["component_route_risk"], index=False, encoding="utf-8-sig")
    instance.component_route_period_scenario_table.to_csv(
        paths["component_route_period_scenario_risk_table"], index=False, encoding="utf-8-sig"
    )
    _write_json(
        paths["model_summary"],
        {
            "generated_at_utc": _now(),
            **model_size_counts(model_data),
            "constraint_names": model_data.constraint_names,
            "config": _config_to_dict(config),
            "scenario_probability_summary": instance.scenario_probability_summary,
            "chance_candidate_summary": {
                "stage4_route_candidate_count": instance.stage4_route_candidate_count,
                "stage5_route_candidate_count": int(len(instance.component_route_period_scenario_table)),
                "chance_pass_rate": instance.to_summary_dict().get("chance_pass_rate"),
            },
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
    solution.chance_constraint_report.to_csv(paths["chance_constraint_report"], index=False, encoding="utf-8-sig")
    solution.scenario_risk_metrics.to_csv(paths["scenario_risk_metrics"], index=False, encoding="utf-8-sig")
    _write_json(paths["cvar_summary"], {"generated_at_utc": _now(), **solution.cvar_summary})
    paths["report_md"].write_text(_markdown_report(instance, model_data, solution), encoding="utf-8")
    return {name: str(path) for name, path in paths.items()}


def _markdown_report(instance: Stage5Instance, model_data: Stage5ModelData, solution: Stage5Solution) -> str:
    counts = model_size_counts(model_data)
    checks = summarize_checks(solution.solution_checks)
    metrics = solution.summary_metrics
    baseline = solution.baseline_comparison
    cvar = solution.cvar_summary
    stage4 = solution.stage4_comparison
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
    scenario_summary = scenario_summary.merge(solution.scenario_risk_metrics, on="scenario_id", how="left")
    chance_excluded = int(instance.stage4_route_candidate_count - len(instance.component_route_period_scenario_table))
    lines = [
        "# Stage 5 Risk-Averse SAA MILP Report",
        "",
        f"Generated at UTC: `{_now()}`",
        "",
        "## Instance",
        "",
        f"- Machine type: `{instance.machine_type_id}`",
        f"- Period window: `{instance.periods[0]}` to `{instance.periods[-1]}`",
        f"- Scenario count: `{len(instance.scenario_ids)}`",
        f"- Candidate cores: `{instance.candidate_core_count}`",
        f"- Component instances: `{instance.component_instance_count}`",
        f"- Stage 4 candidate rows: `{instance.stage4_route_candidate_count}`",
        f"- Stage 5 chance-feasible rows: `{len(instance.component_route_period_scenario_table)}`",
        f"- Excluded rows by chance constraints: `{chance_excluded}`",
        f"- Min system reliability threshold: `{_fmt(instance.min_system_reliability)}`",
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
            f"- Objective with CVaR: `{_fmt(solution.objective_value)}`",
            f"- MIP gap: `{_fmt(solution.mip_gap)}`",
            f"- Solve seconds: `{solution.solve_seconds:.3f}`",
            f"- Solution checks: `{checks}`",
            "",
            "## CVaR Metrics",
            "",
            f"- Confidence: `{_fmt(cvar.get('confidence'))}`",
            f"- Lambda: `{_fmt(cvar.get('lambda'))}`",
            f"- Eta / VaR: `{_fmt(cvar.get('eta'))}`",
            f"- CVaR value: `{_fmt(cvar.get('cvar_value'))}`",
            f"- CVaR objective contribution: `{_fmt(cvar.get('cvar_objective_contribution'))}`",
            f"- Worst scenario: `{cvar.get('worst_scenario_id')}` / `{_fmt(cvar.get('worst_scenario_loss'))}`",
            "",
            "## Scenario Performance And Risk",
            "",
            "| Scenario | Demand | Assembled | Backlog | Loss | Tail excess |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in scenario_summary.itertuples(index=False):
        lines.append(
            f"| `{row.scenario_id}` | {_fmt(row.demand_units)} | {_fmt(row.assembled_units)} | "
            f"{_fmt(row.backlog_units)} | {_fmt(getattr(row, 'scenario_loss', None))} | {_fmt(getattr(row, 'tail_excess', None))} |"
        )
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
            "## Stage 4 Comparison",
            "",
            f"- Stage 4 status: `{stage4.get('status')}`",
            f"- Stage 4 objective: `{_fmt(stage4.get('objective_value'))}`",
            f"- Stage 4 expected assembled units: `{_fmt(stage4.get('expected_assembled_units'))}`",
            f"- Stage 4 expected final backlog units: `{_fmt(stage4.get('expected_final_backlog_units'))}`",
            f"- Stage 5 expected assembled units: `{_fmt(metrics.get('expected_assembled_units'))}`",
            f"- Stage 5 expected final backlog units: `{_fmt(metrics.get('expected_final_backlog_units'))}`",
            "",
            "## BR14 Baseline Reference",
            "",
            f"- Baseline rule: `{baseline.get('baseline_rule_id')}` / `{baseline.get('baseline_rule_name')}`",
            f"- Risk baseline rule: `{baseline.get('risk_baseline_rule_id')}`",
            f"- Risk weight: `{_fmt(baseline.get('risk_weight'))}`",
            f"- Reliability weight: `{_fmt(baseline.get('reliability_weight'))}`",
            f"- Uses scenarios / recourse: `{baseline.get('uses_scenarios')}` / `{baseline.get('allows_recourse')}`",
            f"- Uses chance constraints / CVaR: `{baseline.get('uses_chance_constraints')}` / `{baseline.get('uses_cvar')}`",
            f"- Baseline estimated assembly units: `{baseline.get('estimated_assembly_units')}`",
            f"- Baseline estimated backlog units: `{baseline.get('estimated_backlog_units')}`",
            "",
            "## Stage Boundary",
            "",
            "This Stage 5 run adds hard reliability chance constraints and a CVaR risk objective to Stage 4. It still excludes selective assembly, dimensional-chain constraints, Pareto optimisation, and matheuristics.",
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


def _config_to_dict(config: Stage5Config) -> Dict[str, object]:
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
