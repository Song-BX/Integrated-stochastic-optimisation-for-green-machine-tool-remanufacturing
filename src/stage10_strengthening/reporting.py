"""Report writers for Stage 10 targeted model strengthening."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd

from .config import Stage10Config
from .structures import Stage10Result


def write_stage10_reports(result: Stage10Result, config: Stage10Config) -> Dict[str, str]:
    """Write Stage 10 processed and result artifacts."""

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "pair_carbon_mapping": config.processed_dir / "pair_carbon_mapping.csv",
        "environmental_objective_breakdown": config.processed_dir / "environmental_objective_breakdown.csv",
        "shared_capacity_instance_summary": config.processed_dir / "shared_capacity_instance_summary.json",
        "shared_capacity_model_summary": config.processed_dir / "shared_capacity_model_summary.json",
        "pair_carbon_summary": config.results_dir / "pair_carbon_summary.json",
        "shared_capacity_solution_summary": config.results_dir / "shared_capacity_solution_summary.json",
        "shared_capacity_comparison": config.results_dir / "shared_capacity_comparison.csv",
        "shared_capacity_utilization": config.results_dir / "shared_capacity_utilization.csv",
        "report_md": config.results_dir / "stage10_strengthening_report.md",
    }
    result.pair_carbon_mapping.to_csv(paths["pair_carbon_mapping"], index=False, encoding="utf-8-sig")
    result.environmental_objective_breakdown.to_csv(paths["environmental_objective_breakdown"], index=False, encoding="utf-8-sig")
    _write_json(paths["shared_capacity_instance_summary"], result.shared_capacity_instance_summary)
    _write_json(paths["shared_capacity_model_summary"], result.shared_capacity_model_summary)
    _write_json(paths["pair_carbon_summary"], result.pair_carbon_summary.to_dict())
    _write_json(paths["shared_capacity_solution_summary"], result.shared_capacity_solution_summary)
    result.shared_capacity_comparison.to_csv(paths["shared_capacity_comparison"], index=False, encoding="utf-8-sig")
    result.shared_capacity_utilization.to_csv(paths["shared_capacity_utilization"], index=False, encoding="utf-8-sig")
    paths["report_md"].write_text(_markdown_report(result, config), encoding="utf-8")
    return {name: str(path) for name, path in paths.items()}


def _markdown_report(result: Stage10Result, config: Stage10Config) -> str:
    pair = result.pair_carbon_summary
    comparison = result.shared_capacity_comparison
    lines = [
        "# Stage 10 Targeted Model Strengthening Report",
        "",
        f"Generated at UTC: `{_now()}`",
        "",
        "## Scope",
        "",
        "- Stage 10 adds assembly pair carbon to the Stage 7/8 environmental objective vectors.",
        "- Stage 10 also runs a small shared-capacity extension experiment based on Stage 4 SAA granularity.",
        "- It does not replace the Stage 6 selective-assembly model or the Stage 8 top5 independent benchmark.",
        "",
        "## Assembly Pair Carbon",
        "",
        f"- Machine type audited: `{pair.machine_type_id}`",
        f"- Pair carbon coefficients: `{pair.pair_coefficient_count}`",
        f"- Nonzero pair carbon coefficients: `{pair.pair_nonzero_coefficient_count}`",
        f"- Environmental nonzero count before pair carbon: `{pair.environmental_nonzero_before}`",
        f"- Environmental nonzero count after pair carbon: `{pair.environmental_nonzero_after}`",
        f"- Total weighted pair carbon coefficient: `{_fmt(pair.total_weighted_pair_carbon)}`",
        f"- Mean pair carbon kg: `{_fmt(pair.mean_pair_carbon_kg)}`",
        f"- Max pair carbon kg: `{_fmt(pair.max_pair_carbon_kg)}`",
        f"- Environmental vector finite: `{pair.finite_objective_vector}`",
        "",
        "## Shared-Capacity Mini Experiment",
        "",
        f"- Machine types: `{', '.join(config.machine_types)}`",
        f"- Period window: `{config.period_start}` / `{config.period_count}` periods",
        f"- Shared resources: `{', '.join(config.shared_resources)}`",
        "",
        "| Capacity mode | Machine type | Success | Objective | Backlog | Overtime | Mean util. | Max util. | Seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| `{row.capacity_mode}` | `{row.machine_type_id}` | `{row.success}` | {_fmt(row.objective_value)} | "
            f"{_fmt(row.expected_final_backlog_units)} | {_fmt(row.expected_overtime_hours)} | "
            f"{_fmt(row.mean_shared_resource_utilization)} | {_fmt(row.max_shared_resource_utilization)} | {_fmt(row.solve_seconds)} |"
        )
    lines.extend(
        [
            "",
            "## Checks",
            "",
        ]
    )
    for check in result.checks:
        lines.append(f"- `{check['status']}` `{check['name']}`: {check['message']}")
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "The shared-capacity rows are an extension experiment for reviewer-facing evidence that the formulation can express multi-type resource competition. The main paper model remains the Stage 6/7/8 selective-assembly CVaR SAA structure unless the manuscript explicitly promotes this extension.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps({"generated_at_utc": _now(), **payload}, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    return str(value)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fmt(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    try:
        return f"{float(value):,.4f}"
    except (TypeError, ValueError):
        return str(value)

