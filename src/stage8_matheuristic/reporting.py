"""Report and plot writers for Stage 8 matheuristic runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import Stage8Config
from .io_utils import ensure_output_dirs
from .structures import Stage8ModelData, Stage8RunResult


def write_stage8_reports(model_data: Stage8ModelData, result: Stage8RunResult, config: Stage8Config) -> Dict[str, str]:
    """Write Stage 8 processed and result artifacts."""

    ensure_output_dirs(config.processed_dir, config.results_dir)
    paths = {
        "instance_summary": config.processed_dir / "instance_summary.json",
        "heuristic_config": config.processed_dir / "heuristic_config.json",
        "initial_restriction_summary": config.processed_dir / "initial_restriction_summary.csv",
        "operator_catalogue": config.processed_dir / "operator_catalogue.csv",
        "benchmark_instances": config.processed_dir / "benchmark_instances.csv",
        "model_summary": config.processed_dir / "model_summary.json",
        "iteration_log": config.results_dir / "iteration_log.csv",
        "repair_solve_log": config.results_dir / "repair_solve_log.csv",
        "incumbent_solution_summary": config.results_dir / "incumbent_solution_summary.json",
        "approx_pareto_front": config.results_dir / "approx_pareto_front.csv",
        "dominated_solutions": config.results_dir / "dominated_solutions.csv",
        "operator_scores": config.results_dir / "operator_scores.csv",
        "large_benchmark_summary": config.results_dir / "large_benchmark_summary.csv",
        "convergence": config.results_dir / "convergence.png",
        "operator_scores_plot": config.results_dir / "operator_scores.png",
        "approx_pareto_3d": config.results_dir / "approx_pareto_3d.png",
        "benchmark_runtime_gap": config.results_dir / "benchmark_runtime_gap.png",
        "solution_checks": config.results_dir / "solution_checks.json",
        "report_md": config.results_dir / "stage8_matheuristic_report.md",
    }
    _write_json(paths["instance_summary"], {"generated_at_utc": _now(), "instance": result.instance_summary})
    _write_json(paths["heuristic_config"], {"generated_at_utc": _now(), "config": result.heuristic_config})
    result.initial_restriction_summary.to_csv(paths["initial_restriction_summary"], index=False, encoding="utf-8-sig")
    result.operator_catalogue.to_csv(paths["operator_catalogue"], index=False, encoding="utf-8-sig")
    result.benchmark_instances.to_csv(paths["benchmark_instances"], index=False, encoding="utf-8-sig")
    _write_json(
        paths["model_summary"],
        {
            "generated_at_utc": _now(),
            "model_summary": result.model_summary,
            "objective_vector_summary": model_data.objective_vector_summary,
            "restriction_summary": model_data.restriction_summary,
        },
    )
    result.iteration_log.to_csv(paths["iteration_log"], index=False, encoding="utf-8-sig")
    result.repair_solve_log.to_csv(paths["repair_solve_log"], index=False, encoding="utf-8-sig")
    _write_json(paths["incumbent_solution_summary"], {"generated_at_utc": _now(), **result.incumbent_solution_summary})
    result.approx_pareto_front.to_csv(paths["approx_pareto_front"], index=False, encoding="utf-8-sig")
    result.dominated_solutions.to_csv(paths["dominated_solutions"], index=False, encoding="utf-8-sig")
    result.operator_scores.to_csv(paths["operator_scores"], index=False, encoding="utf-8-sig")
    result.large_benchmark_summary.to_csv(paths["large_benchmark_summary"], index=False, encoding="utf-8-sig")
    _write_json(paths["solution_checks"], {"generated_at_utc": _now(), "checks": result.solution_checks})
    _write_plots(result, paths)
    paths["report_md"].write_text(_markdown_report(result), encoding="utf-8")
    return {name: str(path) for name, path in paths.items()}


def _write_plots(result: Stage8RunResult, paths: Dict[str, Path]) -> None:
    _plot_convergence(result.iteration_log, paths["convergence"])
    _plot_operator_scores(result.operator_scores, paths["operator_scores_plot"])
    _plot_pareto_3d(result.approx_pareto_front, result.repair_solve_log, paths["approx_pareto_3d"])
    _plot_benchmark(result.large_benchmark_summary, paths["benchmark_runtime_gap"])


def _plot_convergence(iteration_log: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    if not iteration_log.empty:
        feasible = iteration_log[iteration_log["feasible"] == True].copy()  # noqa: E712
        if not feasible.empty:
            feasible["best_economic_risk"] = feasible["economic_risk"].cummin()
            ax.plot(feasible["iteration"], feasible["best_economic_risk"], marker="o", linewidth=1.2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best economic risk")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_operator_scores(operator_scores: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    if not operator_scores.empty:
        ax.bar(operator_scores["operator_name"], operator_scores["score"], color="#4c78a8")
        ax.tick_params(axis="x", rotation=35)
    ax.set_ylabel("Adaptive score")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_pareto_3d(pareto: pd.DataFrame, repair_log: pd.DataFrame, path: Path) -> None:
    fig = plt.figure(figsize=(7, 5), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    feasible = repair_log[(repair_log.get("feasible", pd.Series(dtype=bool)) == True) & (repair_log.get("grid_id", "") != "ANCHOR")].copy() if not repair_log.empty else pd.DataFrame()  # noqa: E712
    if not feasible.empty:
        ax.scatter(feasible["economic_risk"], feasible["environmental_impact"], feasible["assembly_quality_loss"], c="#9aa4b2", s=18, label="Feasible repairs")
    if not pareto.empty:
        ax.scatter(pareto["economic_risk"], pareto["environmental_impact"], pareto["assembly_quality_loss"], c="#d62728", s=36, label="Approx Pareto")
    ax.set_xlabel("Economic risk")
    ax.set_ylabel("Environmental impact")
    ax.set_zlabel("Assembly quality loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_benchmark(benchmark: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    if not benchmark.empty and {"machine_type_id", "solve_seconds"}.issubset(benchmark.columns):
        ax.bar(benchmark["machine_type_id"], benchmark["solve_seconds"], color="#59a14f")
        ax.tick_params(axis="x", rotation=25)
    ax.set_ylabel("Solve seconds")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _markdown_report(result: Stage8RunResult) -> str:
    checks = {key: sum(1 for check in result.solution_checks if check["severity"] == key) for key in ["passed", "failed"]}
    comparison = result.stage7_comparison
    lines = [
        "# Stage 8 Matheuristic Report",
        "",
        f"Generated at UTC: `{_now()}`",
        "",
        "## Instance",
        "",
        f"- Machine type: `{result.instance_summary.get('machine_type_id')}`",
        f"- Period window: `{result.instance_summary.get('periods', ['n/a'])[0]}` to `{result.instance_summary.get('periods', ['n/a'])[-1]}`",
        f"- Scenarios: `{len(result.instance_summary.get('scenario_ids', []))}`",
        f"- Method: `{result.instance_summary.get('heuristic_method')}`",
        f"- Pareto mode: `{result.instance_summary.get('pareto_mode')}`",
        "",
        "## Search Summary",
        "",
        f"- Success: `{result.success}`",
        f"- Status: `{result.status_message}`",
        f"- Solve seconds: `{result.solve_seconds:.3f}`",
        f"- Repair solves: `{len(result.repair_solve_log)}`",
        f"- Feasible repair solves: `{int((result.repair_solve_log['feasible'] == True).sum()) if not result.repair_solve_log.empty else 0}`",
        f"- Approximate Pareto points: `{len(result.approx_pareto_front)}`",
        f"- Checks: `{checks}`",
        "",
        "## Stage 7 Exact Comparison",
        "",
        f"- Reference status: `{comparison.get('stage7_reference_status')}`",
        f"- Exact Pareto points: `{comparison.get('exact_pareto_point_count')}`",
        f"- Approx Pareto points: `{comparison.get('approx_pareto_point_count')}`",
        f"- Economic anchor gap (%): `{_fmt(comparison.get('economic_anchor_gap_pct'))}`",
        "",
        "## Operator Scores",
        "",
        "| Operator | Uses | Successes | Score | Success rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in result.operator_scores.itertuples(index=False):
        lines.append(f"| `{row.operator_name}` | {row.uses} | {row.successes} | {_fmt(row.score)} | {_fmt(row.success_rate)} |")
    lines.extend(
        [
            "",
            "## Managerial Notes",
            "",
            "- Stage 8 provides a scalable solution path for IJPR-scale experiments without relaxing the Stage 6/7 operating constraints.",
            "- Restricted MILP repair keeps procurement, backlog, overtime, inventory, CVaR, and assembly shortfall variables open, so infeasibility points to candidate-pool restrictions rather than a changed model.",
            "- Approximate Pareto points can be used to screen large instances before exact Stage 7 solves are attempted on selected representative cases.",
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):,.4f}"
    except (TypeError, ValueError):
        return str(value)
