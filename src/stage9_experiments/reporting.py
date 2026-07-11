"""Report writers for Stage 9 experiment suites."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd

from .config import Stage9Config
from .structures import ExperimentSuiteResult


def write_stage9_reports(result: ExperimentSuiteResult, config: Stage9Config) -> Dict[str, str]:
    """Write all Stage 9 processed/result artifacts."""

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "experiment_manifest_csv": config.processed_dir / "experiment_manifest.csv",
        "experiment_manifest_json": config.processed_dir / "experiment_manifest.json",
        "metric_dictionary": config.processed_dir / "metric_dictionary.json",
        "all_experiment_results": config.results_dir / "all_experiment_results.csv",
        "baseline_comparison": config.results_dir / "baseline_comparison.csv",
        "ablation_study": config.results_dir / "ablation_study.csv",
        "saa_stability": config.results_dir / "saa_stability.csv",
        "sensitivity_summary": config.results_dir / "sensitivity_summary.csv",
        "exact_vs_matheuristic_gap": config.results_dir / "exact_vs_matheuristic_gap.csv",
        "top5_benchmark_summary": config.results_dir / "top5_benchmark_summary.csv",
        "experiment_checks": config.results_dir / "experiment_checks.json",
        "report_md": config.results_dir / "stage9_experiment_report.md",
    }
    result.manifest.to_csv(paths["experiment_manifest_csv"], index=False, encoding="utf-8-sig")
    _write_json(paths["experiment_manifest_json"], {"generated_at_utc": _now(), "manifest": result.manifest.to_dict(orient="records")})
    _write_json(paths["metric_dictionary"], {"generated_at_utc": _now(), "metrics": result.metric_dictionary})
    result.all_experiment_results.to_csv(paths["all_experiment_results"], index=False, encoding="utf-8-sig")
    result.baseline_comparison.to_csv(paths["baseline_comparison"], index=False, encoding="utf-8-sig")
    result.ablation_study.to_csv(paths["ablation_study"], index=False, encoding="utf-8-sig")
    result.saa_stability.to_csv(paths["saa_stability"], index=False, encoding="utf-8-sig")
    result.sensitivity_summary.to_csv(paths["sensitivity_summary"], index=False, encoding="utf-8-sig")
    result.exact_vs_matheuristic_gap.to_csv(paths["exact_vs_matheuristic_gap"], index=False, encoding="utf-8-sig")
    result.top5_benchmark_summary.to_csv(paths["top5_benchmark_summary"], index=False, encoding="utf-8-sig")
    _write_json(paths["experiment_checks"], {"generated_at_utc": _now(), "checks": result.experiment_checks})
    paths["report_md"].write_text(_markdown_report(result, config), encoding="utf-8")
    return {name: str(path) for name, path in paths.items()}


def _markdown_report(result: ExperimentSuiteResult, config: Stage9Config) -> str:
    checks = {key: sum(1 for check in result.experiment_checks if check["severity"] == key) for key in ["passed", "warning", "failed"]}
    baseline_rows = result.baseline_comparison if not result.baseline_comparison.empty else pd.DataFrame()
    gap = result.exact_vs_matheuristic_gap.iloc[0].to_dict() if not result.exact_vs_matheuristic_gap.empty else {}
    top5 = result.top5_benchmark_summary
    lines = [
        "# Stage 9 Experiment Suite Report",
        "",
        f"Generated at UTC: `{_now()}`",
        "",
        "## Configuration",
        "",
        f"- Profile: `{config.profile}`",
        f"- Execution mode: `{config.execution_mode}`",
        f"- Machine type: `{config.machine_type_id}`",
        f"- Period window: `{config.period_start}` / `{config.period_count}` periods",
        "",
        "## Suite Summary",
        "",
        f"- Success: `{result.success}`",
        f"- Status: `{result.status_message}`",
        f"- Manifest rows: `{len(result.manifest)}`",
        f"- Collected result rows: `{int((result.all_experiment_results['status'] == 'collected').sum()) if not result.all_experiment_results.empty else 0}`",
        f"- Checks: `{checks}`",
        "",
        "## Baseline Comparison",
        "",
        "| Experiment | Stage | Status | Objective / economic risk | Backlog | CVaR | Assembly shortfall |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in baseline_rows.itertuples(index=False):
        lines.append(
            f"| `{row.experiment_id}` | `{row.model_stage}` | `{row.status}` | {_fmt(getattr(row, 'economic_risk', None) or getattr(row, 'objective_value', None))} | "
            f"{_fmt(getattr(row, 'expected_final_backlog_units', None))} | {_fmt(getattr(row, 'cvar_value', None))} | "
            f"{_fmt(getattr(row, 'expected_assembly_shortfall_units', None))} |"
        )
    lines.extend(
        [
            "",
            "## Exact vs Matheuristic",
            "",
            f"- Exact Pareto points: `{gap.get('exact_pareto_points')}`",
            f"- Approx Pareto points: `{gap.get('approx_pareto_points')}`",
            f"- Economic-risk gap (%): `{_fmt(gap.get('economic_risk_gap_pct'))}`",
            "",
            "## Top5 Benchmark",
            "",
            f"- Rows: `{len(top5)}`",
            f"- Successful instances: `{_success_count(top5)}`",
            f"- Total wall seconds: `{_fmt(_sum_col(top5, 'wall_seconds'))}`",
            "",
            "## Notes",
            "",
            "- Stage 9 is an experiment collection/orchestration layer. It does not introduce a new optimization model.",
            "- Missing optional SAA/sensitivity runs are recorded as warnings so the report can serve as a living experiment manifest.",
            "- Canonical Stage 3-Stage 8 outputs are not overwritten by default.",
            "",
        ]
    )
    return "\n".join(lines)


def _success_count(frame: pd.DataFrame) -> int:
    if frame.empty or "success" not in frame.columns:
        return 0
    return int(frame["success"].astype(str).str.lower().isin(["true", "1", "yes"]).sum())


def _sum_col(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.sum()) if not values.empty else None


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
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    try:
        return f"{float(value):,.4f}"
    except (TypeError, ValueError):
        return str(value)
