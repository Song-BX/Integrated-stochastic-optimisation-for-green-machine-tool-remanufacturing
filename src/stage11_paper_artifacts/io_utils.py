"""Input helpers and source discovery for Stage 11."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from pandas.errors import EmptyDataError

from .config import Stage11Config


CSV_ENCODING = "utf-8-sig"


def require_stage1_gate(config: Stage11Config) -> Dict[str, Any]:
    """Read and optionally enforce the Stage 1 validation gate."""

    payload = read_json(config.stage1_report)
    failed = int(payload.get("summary", {}).get("failed", 0))
    if config.stage1_gate_required and failed != 0:
        raise RuntimeError(f"Stage 1 validation gate failed: failed={failed}.")
    return payload


def collect_existing_sources(config: Stage11Config) -> Dict[str, Any]:
    """Collect existing Stage 1-10 outputs without rerunning any solver."""

    processed = config.data_processed_dir
    results = config.data_results_dir
    snapshot: Dict[str, Any] = {
        "stage1_validation": read_json(processed / "stage1" / "validation_report.json"),
        "stage1_catalogue": read_csv(processed / "stage1" / "data_catalogue.csv"),
        "model_summaries": {},
        "solution_summaries": {},
        "checks": {},
        "csv": {},
        "json": {},
        "missing_sources": [],
    }
    discovered_paths: Dict[str, str] = {}
    for stage in range(2, 11):
        stage_name = f"stage{stage}"
        model_path = processed / stage_name / "model_summary.json"
        instance_path = processed / stage_name / "instance_summary.json"
        solution_path = results / stage_name / "solution_summary.json"
        if model_path.exists():
            snapshot["model_summaries"][stage_name] = read_json(model_path)
            discovered_paths[f"{stage_name}_model_summary"] = str(model_path)
        if instance_path.exists():
            snapshot["json"][f"{stage_name}_instance_summary"] = read_json(instance_path)
            discovered_paths[f"{stage_name}_instance_summary"] = str(instance_path)
        if solution_path.exists():
            solution_payload = read_json(solution_path)
            snapshot["solution_summaries"][stage_name] = solution_payload
            snapshot["json"][f"{stage_name}_solution"] = solution_payload
            discovered_paths[f"{stage_name}_solution"] = str(solution_path)
        check_path = results / stage_name / "solution_checks.json"
        if check_path.exists():
            snapshot["checks"][stage_name] = read_json(check_path)
            discovered_paths[f"{stage_name}_checks"] = str(check_path)

    csv_sources = {
        "stage3_selected_routes": results / "stage3" / "selected_component_routes.csv",
        "stage4_selected_routes": results / "stage4" / "scenario_selected_component_routes.csv",
        "stage5_selected_routes": results / "stage5" / "scenario_selected_component_routes.csv",
        "stage6_selected_routes": results / "stage6" / "scenario_selected_component_routes.csv",
        "stage7_pareto_front": results / "stage7" / "pareto_front.csv",
        "stage7_grid": results / "stage7" / "grid_solution_summary.csv",
        "stage7_payoff": processed / "stage7" / "payoff_table.csv",
        "stage8_approx_pareto": results / "stage8" / "approx_pareto_front.csv",
        "stage8_iteration_log": results / "stage8" / "iteration_log.csv",
        "stage8_operator_scores": results / "stage8" / "operator_scores.csv",
        "stage8_large_benchmark": results / "stage8" / "large_benchmark_summary.csv",
        "stage9_baseline": results / "stage9" / "baseline_comparison.csv",
        "stage9_ablation": results / "stage9" / "ablation_study.csv",
        "stage9_saa": results / "stage9" / "saa_stability.csv",
        "stage9_sensitivity": results / "stage9" / "sensitivity_summary.csv",
        "stage9_exact_gap": results / "stage9" / "exact_vs_matheuristic_gap.csv",
        "stage9_top5": results / "stage9" / "top5_benchmark_summary.csv",
        "stage10_env_breakdown": processed / "stage10" / "environmental_objective_breakdown.csv",
        "stage10_shared_comparison": results / "stage10" / "shared_capacity_comparison.csv",
        "stage10_shared_utilization": results / "stage10" / "shared_capacity_utilization.csv",
        "stage12_operational_audit": results / "stage12" / "operational_reality_audit.csv",
    }
    json_sources = {
        "stage7_representatives": results / "stage7" / "representative_solutions.json",
        "stage8_incumbent": results / "stage8" / "incumbent_solution_summary.json",
        "stage10_pair_carbon": results / "stage10" / "pair_carbon_summary.json",
        "stage10_shared_solution": results / "stage10" / "shared_capacity_solution_summary.json",
    }
    for key, path in csv_sources.items():
        if path.exists():
            snapshot["csv"][key] = read_csv(path)
        else:
            snapshot["missing_sources"].append(str(path))
    for key, path in json_sources.items():
        if path.exists():
            snapshot["json"][key] = read_json(path)
        else:
            snapshot["missing_sources"].append(str(path))
    snapshot["source_paths"] = {
        **discovered_paths,
        **{key: str(path) for key, path in csv_sources.items()},
        **{key: str(path) for key, path in json_sources.items()},
    }
    return snapshot


def ensure_output_dirs(config: Stage11Config) -> Dict[str, Path]:
    """Create Stage 11 output directories."""

    tables_dir = config.results_dir / "tables"
    figures_dir = config.results_dir / "figures"
    for path in [config.processed_dir, config.results_dir, tables_dir, figures_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return {"tables": tables_dir, "figures": figures_dir}


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding=CSV_ENCODING)
    except EmptyDataError:
        return pd.DataFrame()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    return str(value)
