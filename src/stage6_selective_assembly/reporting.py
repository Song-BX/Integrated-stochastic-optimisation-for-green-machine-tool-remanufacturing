"""Report writers for Stage 6 selective-assembly runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd

from .checks import model_size_counts, summarize_checks
from .config import Stage6Config
from .io_utils import ensure_output_dirs
from .structures import Stage6Instance, Stage6ModelData, Stage6Solution


def write_stage6_reports(
    instance: Stage6Instance,
    model_data: Stage6ModelData,
    solution: Stage6Solution,
    config: Stage6Config,
) -> Dict[str, str]:
    ensure_output_dirs(config.processed_dir, config.results_dir)
    paths = {
        "instance_summary": config.processed_dir / "instance_summary.json",
        "assembly_requirements_filtered": config.processed_dir / "assembly_requirements_filtered.csv",
        "assembly_candidate_pool": config.processed_dir / "assembly_candidate_pool.csv",
        "assembly_pair_pool": config.processed_dir / "assembly_pair_pool.csv",
        "model_summary": config.processed_dir / "model_summary.json",
        "solution_json": config.results_dir / "solution_summary.json",
        "solution_checks": config.results_dir / "solution_checks.json",
        "first_stage_decisions": config.results_dir / "first_stage_decisions.csv",
        "scenario_selected_component_routes": config.results_dir / "scenario_selected_component_routes.csv",
        "scenario_assembly_plan": config.results_dir / "scenario_assembly_plan.csv",
        "scenario_inventory_trajectory": config.results_dir / "scenario_inventory_trajectory.csv",
        "scenario_capacity_utilization": config.results_dir / "scenario_capacity_utilization.csv",
        "selected_assembly_candidates": config.results_dir / "selected_assembly_candidates.csv",
        "selected_assembly_pairs": config.results_dir / "selected_assembly_pairs.csv",
        "feature_assembly_plan": config.results_dir / "feature_assembly_plan.csv",
        "dimension_chain_report": config.results_dir / "dimension_chain_report.csv",
        "assembly_quality_loss_report": config.results_dir / "assembly_quality_loss_report.csv",
        "scenario_assembly_risk_metrics": config.results_dir / "scenario_assembly_risk_metrics.csv",
        "chance_constraint_report": config.results_dir / "chance_constraint_report.csv",
        "scenario_risk_metrics": config.results_dir / "scenario_risk_metrics.csv",
        "cvar_summary": config.results_dir / "cvar_summary.json",
        "report_md": config.results_dir / "stage6_selective_assembly_report.md",
    }
    _write_json(paths["instance_summary"], {"generated_at_utc": _now(), "instance": instance.to_summary_dict()})
    instance.assembly_requirements.to_csv(paths["assembly_requirements_filtered"], index=False, encoding="utf-8-sig")
    instance.assembly_candidate_pool.to_csv(paths["assembly_candidate_pool"], index=False, encoding="utf-8-sig")
    instance.assembly_pair_pool.to_csv(paths["assembly_pair_pool"], index=False, encoding="utf-8-sig")
    _write_json(
        paths["model_summary"],
        {
            "generated_at_utc": _now(),
            **model_size_counts(model_data),
            "constraint_names": model_data.constraint_names,
            "config": _config_to_dict(config),
            "scenario_probability_summary": instance.scenario_probability_summary,
            "assembly_pool_summary": instance.assembly_pool_summary,
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
    solution.selected_assembly_candidates.to_csv(paths["selected_assembly_candidates"], index=False, encoding="utf-8-sig")
    solution.selected_assembly_pairs.to_csv(paths["selected_assembly_pairs"], index=False, encoding="utf-8-sig")
    solution.feature_assembly_plan.to_csv(paths["feature_assembly_plan"], index=False, encoding="utf-8-sig")
    solution.dimension_chain_report.to_csv(paths["dimension_chain_report"], index=False, encoding="utf-8-sig")
    solution.assembly_quality_loss_report.to_csv(paths["assembly_quality_loss_report"], index=False, encoding="utf-8-sig")
    solution.scenario_assembly_risk_metrics.to_csv(paths["scenario_assembly_risk_metrics"], index=False, encoding="utf-8-sig")
    solution.chance_constraint_report.to_csv(paths["chance_constraint_report"], index=False, encoding="utf-8-sig")
    solution.scenario_risk_metrics.to_csv(paths["scenario_risk_metrics"], index=False, encoding="utf-8-sig")
    _write_json(paths["cvar_summary"], {"generated_at_utc": _now(), **solution.cvar_summary})
    paths["report_md"].write_text(_markdown_report(instance, model_data, solution), encoding="utf-8")
    return {name: str(path) for name, path in paths.items()}


def _markdown_report(instance: Stage6Instance, model_data: Stage6ModelData, solution: Stage6Solution) -> str:
    counts = model_size_counts(model_data)
    checks = summarize_checks(solution.solution_checks)
    cvar = solution.cvar_summary
    metrics = solution.summary_metrics
    stage5 = solution.stage5_comparison
    pool = instance.assembly_pool_summary
    scenario_summary = (
        solution.scenario_assembly_plan.groupby("scenario_id")
        .agg(demand_units=("demand_units", "sum"), assembled_units=("assembled_units", "sum"), backlog_units=("backlog_units", "last"))
        .reset_index()
        .merge(solution.scenario_risk_metrics, on="scenario_id", how="left")
    )
    feature_summary = (
        solution.feature_assembly_plan.groupby("assembly_requirement_id")
        .agg(
            expected_feature_assembled=("feature_assembled_units", lambda s: _prob_weighted(s, solution.feature_assembly_plan.loc[s.index, "saa_probability"])),
            expected_shortfall=("assembly_shortfall_units", lambda s: _prob_weighted(s, solution.feature_assembly_plan.loc[s.index, "saa_probability"])),
            mean_coverage=("coverage_rate", "mean"),
        )
        .reset_index()
        if not solution.feature_assembly_plan.empty
        else pd.DataFrame()
    )
    lines = [
        "# Stage 6 Selective Assembly MILP Report",
        "",
        f"Generated at UTC: `{_now()}`",
        "",
        "## Instance",
        "",
        f"- Machine type: `{instance.machine_type_id}`",
        f"- Period window: `{instance.periods[0]}` to `{instance.periods[-1]}`",
        f"- Scenario count: `{len(instance.scenario_ids)}`",
        f"- Assembly requirements: `{len(instance.assembly_requirements)}`",
        f"- Candidate pool: `{pool.get('candidate_count')}`",
        f"- Pair pool: `{pool.get('pair_count')}`",
        f"- Hard / soft pairs: `{pool.get('hard_pair_count')}` / `{pool.get('soft_pair_count')}`",
        f"- Candidate pool mode: `{instance.candidate_pool_mode}`",
        f"- Pairwise mode: `{instance.pairwise_mode}`",
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
        "## CVaR Metrics",
        "",
        f"- Confidence: `{_fmt(cvar.get('confidence'))}`",
        f"- Lambda: `{_fmt(cvar.get('lambda'))}`",
        f"- Eta / VaR: `{_fmt(cvar.get('eta'))}`",
        f"- CVaR value: `{_fmt(cvar.get('cvar_value'))}`",
        f"- Worst scenario: `{cvar.get('worst_scenario_id')}` / `{_fmt(cvar.get('worst_scenario_loss'))}`",
        "",
        "## Scenario Performance",
        "",
        "| Scenario | Demand | Assembled | Backlog | Loss | Tail excess |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in scenario_summary.itertuples(index=False):
        lines.append(
            f"| `{row.scenario_id}` | {_fmt(row.demand_units)} | {_fmt(row.assembled_units)} | "
            f"{_fmt(row.backlog_units)} | {_fmt(getattr(row, 'scenario_loss', None))} | {_fmt(getattr(row, 'tail_excess', None))} |"
        )
    lines.extend(
        [
            "",
            "## Selective Assembly Coverage",
            "",
            "| Requirement | Expected covered | Expected shortfall | Mean coverage |",
            "|---|---:|---:|---:|",
        ]
    )
    if not feature_summary.empty:
        for row in feature_summary.itertuples(index=False):
            lines.append(
                f"| `{row.assembly_requirement_id}` | {_fmt(row.expected_feature_assembled)} | "
                f"{_fmt(row.expected_shortfall)} | {_fmt(row.mean_coverage)} |"
            )
    lines.extend(
        [
            "",
            "## Stage 5 Comparison",
            "",
            f"- Stage 5 status: `{stage5.get('status')}`",
            f"- Stage 5 objective: `{_fmt(stage5.get('objective_value'))}`",
            f"- Stage 5 CVaR: `{_fmt(stage5.get('cvar_value'))}`",
            f"- Stage 5 expected assembled units: `{_fmt(stage5.get('expected_assembled_units'))}`",
            f"- Stage 5 expected final backlog units: `{_fmt(stage5.get('expected_final_backlog_units'))}`",
            f"- Stage 6 expected assembled units: `{_fmt(metrics.get('expected_assembled_units'))}`",
            f"- Stage 6 expected final backlog units: `{_fmt(metrics.get('expected_final_backlog_units'))}`",
            f"- Stage 6 expected feature shortfall units: `{_fmt(metrics.get('expected_assembly_shortfall_units'))}`",
            "",
            "## Baseline References",
            "",
            f"- BR14 risk rule: `{solution.baseline_comparison.get('risk_baseline_rule_id')}`",
            f"- BR08 selective assembly rule: `{solution.baseline_comparison.get('selective_assembly_rule', {}).get('assembly_policy')}`",
            f"- BR18 no-selective ablation: `{solution.baseline_comparison.get('no_selective_assembly_ablation_rule', {}).get('assembly_policy')}`",
            "",
            "## Stage Boundary",
            "",
            "This Stage 6 run adds scenario-total selective assembly, sparse pairwise compatibility, dimensional-chain penalties, quality-loss penalties, life-gap penalties, and assembly shortfall penalties. It still excludes Pareto multi-objective solving and large-scale matheuristics.",
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


def _config_to_dict(config: Stage6Config) -> Dict[str, object]:
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


def _prob_weighted(values: pd.Series, probabilities: pd.Series) -> float:
    return float((pd.to_numeric(values, errors="coerce").fillna(0.0) * pd.to_numeric(probabilities, errors="coerce").fillna(0.0)).sum())
