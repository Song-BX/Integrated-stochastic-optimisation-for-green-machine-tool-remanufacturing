"""Report and plot writers for Stage 7 Pareto analysis."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import Stage7Config
from .io_utils import ensure_output_dirs
from .structures import Stage7Instance, Stage7ModelData, Stage7Solution


def write_stage7_reports(
    instance: Stage7Instance,
    model_data: Stage7ModelData,
    solution: Stage7Solution,
    config: Stage7Config,
) -> Dict[str, str]:
    ensure_output_dirs(config.processed_dir, config.results_dir)
    paths = {
        "objective_vectors_summary": config.processed_dir / "objective_vectors_summary.json",
        "payoff_table": config.processed_dir / "payoff_table.csv",
        "epsilon_grid": config.processed_dir / "epsilon_grid.csv",
        "model_summary": config.processed_dir / "model_summary.json",
        "grid_solution_summary": config.results_dir / "grid_solution_summary.csv",
        "pareto_front": config.results_dir / "pareto_front.csv",
        "dominated_solutions": config.results_dir / "dominated_solutions.csv",
        "representative_solutions": config.results_dir / "representative_solutions.json",
        "pareto_cost_environment": config.results_dir / "pareto_cost_environment.png",
        "pareto_cost_assembly": config.results_dir / "pareto_cost_assembly.png",
        "pareto_3d": config.results_dir / "pareto_3d.png",
        "solution_checks": config.results_dir / "solution_checks.json",
        "report_md": config.results_dir / "stage7_pareto_report.md",
    }
    _write_json(paths["objective_vectors_summary"], {"generated_at_utc": _now(), **model_data.objective_vector_summary})
    solution.payoff_table.to_csv(paths["payoff_table"], index=False, encoding="utf-8-sig")
    solution.epsilon_grid.to_csv(paths["epsilon_grid"], index=False, encoding="utf-8-sig")
    _write_json(
        paths["model_summary"],
        {
            "generated_at_utc": _now(),
            "instance": instance.to_summary_dict(),
            "model_summary": solution.model_summary,
            "config": _config_to_dict(config),
        },
    )
    solution.grid_solution_summary.to_csv(paths["grid_solution_summary"], index=False, encoding="utf-8-sig")
    solution.pareto_front.to_csv(paths["pareto_front"], index=False, encoding="utf-8-sig")
    solution.dominated_solutions.to_csv(paths["dominated_solutions"], index=False, encoding="utf-8-sig")
    _write_json(paths["representative_solutions"], {"generated_at_utc": _now(), **solution.representative_solutions})
    _write_json(paths["solution_checks"], {"generated_at_utc": _now(), "checks": solution.solution_checks})
    _write_plots(solution, paths)
    paths["report_md"].write_text(_markdown_report(instance, solution), encoding="utf-8")
    return {name: str(path) for name, path in paths.items()}


def _write_plots(solution: Stage7Solution, paths: Dict[str, Path]) -> None:
    pareto = solution.pareto_front
    grid = solution.grid_solution_summary
    feasible = grid[grid["feasible"] == 1] if not grid.empty else pd.DataFrame()
    _scatter2d(
        feasible,
        pareto,
        "economic_risk",
        "environmental_impact",
        "Economic risk",
        "Environmental impact",
        paths["pareto_cost_environment"],
    )
    _scatter2d(
        feasible,
        pareto,
        "economic_risk",
        "assembly_quality_loss",
        "Economic risk",
        "Assembly quality loss",
        paths["pareto_cost_assembly"],
    )
    _scatter3d(feasible, pareto, paths["pareto_3d"])


def _scatter2d(feasible: pd.DataFrame, pareto: pd.DataFrame, x_col: str, y_col: str, x_label: str, y_label: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    if not feasible.empty:
        ax.scatter(feasible[x_col], feasible[y_col], c="#9aa4b2", s=28, label="Feasible grid")
    if not pareto.empty:
        ax.scatter(pareto[x_col], pareto[y_col], c="#d62728", s=42, label="Pareto")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _scatter3d(feasible: pd.DataFrame, pareto: pd.DataFrame, path: Path) -> None:
    fig = plt.figure(figsize=(7, 5), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    if not feasible.empty:
        ax.scatter(feasible["economic_risk"], feasible["environmental_impact"], feasible["assembly_quality_loss"], c="#9aa4b2", s=18, label="Feasible grid")
    if not pareto.empty:
        ax.scatter(pareto["economic_risk"], pareto["environmental_impact"], pareto["assembly_quality_loss"], c="#d62728", s=36, label="Pareto")
    ax.set_xlabel("Economic risk")
    ax.set_ylabel("Environmental impact")
    ax.set_zlabel("Assembly quality loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _markdown_report(instance: Stage7Instance, solution: Stage7Solution) -> str:
    checks = {key: sum(1 for check in solution.solution_checks if check["severity"] == key) for key in ["passed", "failed"]}
    infeasible_count = int((solution.grid_solution_summary["feasible"] != 1).sum()) if not solution.grid_solution_summary.empty else 0
    lines = [
        "# Stage 7 Pareto Analysis Report",
        "",
        f"Generated at UTC: `{_now()}`",
        "",
        "## Instance",
        "",
        f"- Machine type: `{instance.machine_type_id}`",
        f"- Period window: `{instance.periods[0]}` to `{instance.periods[-1]}`",
        f"- Scenarios: `{len(instance.scenario_ids)}`",
        f"- Method: `{instance.multiobjective_method}`",
        "",
        "## Payoff Table",
        "",
        "| Payoff | f1 economic risk | f2 environmental | f3 assembly quality | Fallback |",
        "|---|---:|---:|---:|---|",
    ]
    for row in solution.payoff_table.itertuples(index=False):
        lines.append(
            f"| `{row.payoff_name}` | {_fmt(row.economic_risk)} | {_fmt(row.environmental_impact)} | "
            f"{_fmt(row.assembly_quality_loss)} | `{row.fallback_used}` |"
        )
    lines.extend(
        [
            "",
            "## Pareto Summary",
            "",
            f"- Grid points: `{len(solution.epsilon_grid)}`",
            f"- Feasible grid points: `{int((solution.grid_solution_summary['feasible'] == 1).sum())}`",
            f"- Infeasible grid points: `{infeasible_count}`",
            f"- Pareto points: `{len(solution.pareto_front)}`",
            f"- Checks: `{checks}`",
            f"- Solve seconds: `{solution.solve_seconds:.3f}`",
            "",
            "## Representative Pareto Solutions",
            "",
            "| Representative | Grid | Economic risk | Environmental | Assembly quality |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for name, row in solution.representative_solutions.items():
        lines.append(
            f"| `{name}` | `{row.get('grid_id')}` | {_fmt(row.get('economic_risk'))} | "
            f"{_fmt(row.get('environmental_impact'))} | {_fmt(row.get('assembly_quality_loss'))} |"
        )
    stage6 = solution.stage6_comparison
    metrics = stage6.get("summary_metrics", {}) if isinstance(stage6, dict) else {}
    lines.extend(
        [
            "",
            "## Stage 6 Comparison",
            "",
            f"- Stage 6 status: `{stage6.get('status') if isinstance(stage6, dict) else 'n/a'}`",
            f"- Stage 6 objective: `{_fmt(stage6.get('objective_value') if isinstance(stage6, dict) else None)}`",
            f"- Stage 6 expected assembled units: `{_fmt(metrics.get('expected_assembled_units'))}`",
            f"- Stage 6 expected final backlog units: `{_fmt(metrics.get('expected_final_backlog_units'))}`",
            "",
            "## Notes",
            "",
            "Stage 7 keeps the Stage 6 feasible region unchanged. Environmental impact currently combines route/procurement carbon proxies; assembly-pair carbon is reported through Stage 6 pair data but is not added as a separate coefficient unless exposed in the Stage 6 loss table.",
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


def _config_to_dict(config: Stage7Config) -> Dict[str, object]:
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
