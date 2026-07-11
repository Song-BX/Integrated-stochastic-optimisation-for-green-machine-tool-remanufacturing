"""Stage 11 manuscript table builders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from .config import Stage11Config
from .structures import PaperArtifactSpec, PaperTable


def build_tables(snapshot: Dict[str, Any], config: Stage11Config) -> List[PaperTable]:
    """Build the Stage 11 manuscript tables."""

    tables: List[PaperTable] = []
    tables.append(_make_table_stagewise_complexity(snapshot, config))
    tables.append(_make_table_baseline_and_ablation(snapshot, config))
    tables.append(_make_table_pareto_payoff(snapshot, config))
    tables.append(_make_table_exact_vs_matheuristic(snapshot, config))
    tables.append(_make_table_risk_selective(snapshot, config))
    tables.append(_make_table_stage10(snapshot, config))
    tables.append(_make_table_saa_manifest(snapshot, config))
    return tables


def _make_table_stagewise_complexity(snapshot: Dict[str, Any], config: Stage11Config) -> PaperTable:
    rows: List[Dict[str, Any]] = []
    stage_info = {
        "Stage 1": _stage_row(snapshot, 1, "data validation / schema gate"),
        "Stage 2": _stage_row(snapshot, 2, "deterministic MILP skeleton"),
        "Stage 3": _stage_row(snapshot, 3, "multiperiod / multi-component / multi-route MILP"),
        "Stage 4": _stage_row(snapshot, 4, "SAA stochastic equivalent"),
        "Stage 5": _stage_row(snapshot, 5, "chance constraints / CVaR"),
        "Stage 6": _stage_row(snapshot, 6, "selective assembly / dimension-chain"),
        "Stage 7": _stage_row(snapshot, 7, "augmented epsilon-constraint Pareto"),
        "Stage 8": _stage_row(snapshot, 8, "ALNS + restricted MILP repair"),
        "Stage 9": {"status": "reporting", "variable_count": None, "constraint_count": None, "notes": "experiment suite"},
        "Stage 10": {"status": "extension", "variable_count": None, "constraint_count": None, "notes": "pair carbon + shared capacity"},
    }
    for stage, payload in stage_info.items():
        stage_num = _stage_number(stage)
        variable_count = payload.get("variable_count")
        constraint_count = payload.get("constraint_count")
        if variable_count is None and stage_num is not None:
            variable_count = _extract_number(snapshot, stage_num, "variable_count")
        if constraint_count is None and stage_num is not None:
            constraint_count = _extract_number(snapshot, stage_num, "constraint_count")
        rows.append(
            {
                "stage": stage,
                "status": payload.get("status", "collected"),
                "core_function": payload.get("notes"),
                "instance_label": _instance_label(snapshot, stage),
                "variable_count": variable_count,
                "constraint_count": constraint_count,
                "check_summary": _stage_check_summary(snapshot, stage),
            }
        )
    data = pd.DataFrame(rows)
    formatted = _format_numeric_table(data, integer_cols=["variable_count", "constraint_count"])
    spec = PaperArtifactSpec(
        artifact_id="T1_stagewise_model_complexity",
        artifact_type="table",
        title="Stage-wise model complexity and scope",
        claim="The formulation grows from validation to integrated stochastic optimization without breaking the data gate.",
        source_files=list(snapshot.get("source_paths", {}).values()),
        output_files=[],
    )
    return PaperTable(spec=spec, data=data, formatted=formatted)


def _make_table_baseline_and_ablation(snapshot: Dict[str, Any], config: Stage11Config) -> PaperTable:
    baseline = _read_csv(snapshot, "stage9_baseline")
    ablation = _read_csv(snapshot, "stage9_ablation")
    if not baseline.empty:
        baseline = baseline.copy()
        baseline["table_group"] = "baseline"
    if not ablation.empty:
        ablation = ablation.copy()
        ablation["table_group"] = "ablation"
    data = pd.concat([baseline, ablation], ignore_index=True, sort=False) if (not baseline.empty or not ablation.empty) else pd.DataFrame()
    data = _ensure_columns(data, ["experiment_id", "model_stage", "status", "objective_value", "expected_final_backlog_units", "cvar_value", "expected_assembly_shortfall_units", "runtime_delta_seconds", "table_group"])
    formatted = _format_numeric_table(data, float_cols=["objective_value", "expected_final_backlog_units", "cvar_value", "expected_assembly_shortfall_units", "runtime_delta_seconds"])
    spec = PaperArtifactSpec(
        artifact_id="T2_baseline_and_ablation",
        artifact_type="table",
        title="Baseline comparison and ablation study",
        claim="The integrated model improves or explains trade-offs relative to deterministic, stochastic, CVaR, selective-assembly, and matheuristic baselines.",
        source_files=[snapshot["source_paths"].get("stage9_baseline", ""), snapshot["source_paths"].get("stage9_ablation", "")],
        output_files=[],
        warnings=_missing_warnings([baseline, ablation]),
    )
    return PaperTable(spec=spec, data=data, formatted=formatted)


def _make_table_pareto_payoff(snapshot: Dict[str, Any], config: Stage11Config) -> PaperTable:
    payoff = _read_csv(snapshot, "stage7_payoff")
    reps = _read_json_dict(snapshot, "stage7_representatives")
    rep_rows = []
    for name, row in reps.items():
        if not isinstance(row, dict):
            continue
        rep_rows.append(
            {
                "representative": name,
                "grid_id": row.get("grid_id"),
                "economic_risk": row.get("economic_risk"),
                "environmental_impact": row.get("environmental_impact"),
                "assembly_quality_loss": row.get("assembly_quality_loss"),
                "solver_status": row.get("status"),
            }
        )
    rep_df = pd.DataFrame(rep_rows)
    if not payoff.empty:
        payoff = payoff.copy()
        payoff["payload"] = "payoff"
    if not rep_df.empty:
        rep_df["payload"] = "representative"
    data = pd.concat([payoff, rep_df], ignore_index=True, sort=False) if (not payoff.empty or not rep_df.empty) else pd.DataFrame()
    data = _ensure_columns(data, ["payoff_name", "representative", "grid_id", "economic_risk", "environmental_impact", "assembly_quality_loss", "fallback_used", "solver_status", "payload"])
    formatted = _format_numeric_table(data, float_cols=["economic_risk", "environmental_impact", "assembly_quality_loss"])
    spec = PaperArtifactSpec(
        artifact_id="T3_pareto_payoff_and_representatives",
        artifact_type="table",
        title="Payoff table and representative Pareto solutions",
        claim="Stage 7 yields a compact payoff table and representative Pareto points that support trade-off interpretation.",
        source_files=[snapshot["source_paths"].get("stage7_payoff", ""), snapshot["source_paths"].get("stage7_representatives", "")],
        output_files=[],
    )
    return PaperTable(spec=spec, data=data, formatted=formatted)


def _make_table_exact_vs_matheuristic(snapshot: Dict[str, Any], config: Stage11Config) -> PaperTable:
    gap = _read_csv(snapshot, "stage9_exact_gap")
    top5 = _read_csv(snapshot, "stage9_top5")
    data = pd.concat([gap.assign(table_group="exact_vs_matheuristic") if not gap.empty else gap, top5.assign(table_group="top5_benchmark") if not top5.empty else top5], ignore_index=True, sort=False)
    data = _ensure_columns(
        data,
        [
            "comparison_id",
            "machine_type_id",
            "success",
            "exact_pareto_points",
            "approx_pareto_points",
            "exact_min_economic_risk",
            "approx_min_economic_risk",
            "economic_risk_gap_pct",
            "solve_seconds",
            "wall_seconds",
            "table_group",
        ],
    )
    formatted = _format_numeric_table(
        data,
        float_cols=[
            "exact_pareto_points",
            "approx_pareto_points",
            "exact_min_economic_risk",
            "approx_min_economic_risk",
            "economic_risk_gap_pct",
            "solve_seconds",
            "wall_seconds",
        ],
    )
    spec = PaperArtifactSpec(
        artifact_id="T4_exact_vs_matheuristic_and_top5",
        artifact_type="table",
        title="Exact vs matheuristic gap and top5 benchmark summary",
        claim="The matheuristic closely tracks exact Pareto anchors and scales across the top5 benchmark suite.",
        source_files=[snapshot["source_paths"].get("stage9_exact_gap", ""), snapshot["source_paths"].get("stage9_top5", "")],
        output_files=[],
        warnings=_missing_warnings([gap, top5]),
    )
    return PaperTable(spec=spec, data=data, formatted=formatted)


def _make_table_risk_selective(snapshot: Dict[str, Any], config: Stage11Config) -> PaperTable:
    stage5 = _read_json_dict(snapshot, "stage5_solution")
    stage6 = _read_json_dict(snapshot, "stage6_solution")
    rows = []
    for stage_name, payload in [("Stage 5", stage5), ("Stage 6", stage6)]:
        solution = payload.get("solution", payload) if isinstance(payload, dict) else {}
        metrics = solution.get("summary_metrics", {}) if isinstance(solution, dict) else {}
        cvar = solution.get("cvar_summary", {}) if isinstance(solution, dict) else {}
        rows.append(
            {
                "stage": stage_name,
                "expected_assembled_units": metrics.get("expected_assembled_units"),
                "expected_final_backlog_units": metrics.get("expected_final_backlog_units"),
                "selected_component_route_count": metrics.get("selected_component_route_count"),
                "selected_assembly_pairs": metrics.get("selected_assembly_pairs_count"),
                "assembly_shortfall_units": metrics.get("expected_assembly_shortfall_units"),
                "cvar_value": cvar.get("cvar_value"),
                "worst_scenario_loss": cvar.get("worst_scenario_loss"),
                "route_mix": _compact_dict(metrics.get("route_mix", {})),
            }
        )
    data = pd.DataFrame(rows)
    formatted = _format_numeric_table(
        data,
        float_cols=["expected_assembled_units", "expected_final_backlog_units", "selected_component_route_count", "selected_assembly_pairs", "assembly_shortfall_units", "cvar_value", "worst_scenario_loss"],
    )
    spec = PaperArtifactSpec(
        artifact_id="T5_risk_selective_assembly_metrics",
        artifact_type="table",
        title="Risk-averse selective-assembly metrics",
        claim="Stage 5 and Stage 6 jointly quantify how reliability filtering and selective assembly reshape backlog and CVaR.",
        source_files=[snapshot["source_paths"].get("stage5_solution", ""), snapshot["source_paths"].get("stage6_solution", "")],
        output_files=[],
        warnings=_missing_warnings([stage5, stage6]),
    )
    return PaperTable(spec=spec, data=data, formatted=formatted)


def _make_table_stage10(snapshot: Dict[str, Any], config: Stage11Config) -> PaperTable:
    env_breakdown = _read_csv(snapshot, "stage10_env_breakdown")
    shared = _read_csv(snapshot, "stage10_shared_comparison")
    pair_summary = _read_json_dict(snapshot, "stage10_pair_carbon")
    shared_summary = _read_json_dict(snapshot, "stage10_shared_solution")
    rows = []
    if not env_breakdown.empty:
        for row in env_breakdown.itertuples(index=False):
            rows.append(
                {
                    "section": "environmental_breakdown",
                    "component": getattr(row, "component", None),
                    "nonzero_count": getattr(row, "nonzero_count", None),
                    "coefficient_sum": getattr(row, "coefficient_sum", None),
                    "coefficient_mean": getattr(row, "coefficient_mean", None),
                    "coefficient_max": getattr(row, "coefficient_max", None),
                }
            )
    if not shared.empty:
        for row in shared.itertuples(index=False):
            rows.append(
                {
                    "section": "shared_capacity",
                    "component": getattr(row, "capacity_mode", None),
                    "machine_type_id": getattr(row, "machine_type_id", None),
                    "objective_value": getattr(row, "objective_value", None),
                    "expected_final_backlog_units": getattr(row, "expected_final_backlog_units", None),
                    "expected_overtime_hours": getattr(row, "expected_overtime_hours", None),
                    "mean_shared_resource_utilization": getattr(row, "mean_shared_resource_utilization", None),
                    "max_shared_resource_utilization": getattr(row, "max_shared_resource_utilization", None),
                }
            )
    if pair_summary:
        rows.append(
            {
                "section": "pair_carbon_summary",
                "component": pair_summary.get("machine_type_id"),
                "pair_coefficient_count": pair_summary.get("pair_coefficient_count"),
                "pair_nonzero_coefficient_count": pair_summary.get("pair_nonzero_coefficient_count"),
                "environmental_nonzero_before": pair_summary.get("environmental_nonzero_before"),
                "environmental_nonzero_after": pair_summary.get("environmental_nonzero_after"),
                "total_weighted_pair_carbon": pair_summary.get("total_weighted_pair_carbon"),
            }
        )
    if shared_summary:
        rows.append(
            {
                "section": "shared_capacity_solution",
                "component": shared_summary.get("shared_capacity", {}).get("machine_type_id"),
                "status": shared_summary.get("shared_capacity", {}).get("status"),
                "success": shared_summary.get("shared_capacity", {}).get("success"),
                "objective_value": shared_summary.get("shared_capacity", {}).get("objective_value"),
                "expected_final_backlog_units": shared_summary.get("shared_capacity", {}).get("expected_final_backlog_units"),
            }
        )
    data = pd.DataFrame(rows)
    formatted = _format_numeric_table(
        data,
        float_cols=["nonzero_count", "coefficient_sum", "coefficient_mean", "coefficient_max", "objective_value", "expected_final_backlog_units", "expected_overtime_hours", "mean_shared_resource_utilization", "max_shared_resource_utilization", "pair_coefficient_count", "pair_nonzero_coefficient_count", "environmental_nonzero_before", "environmental_nonzero_after", "total_weighted_pair_carbon"],
    )
    spec = PaperArtifactSpec(
        artifact_id="T6_stage10_strengthening",
        artifact_type="table",
        title="Targeted strengthening results",
        claim="Stage 10 adds pair-carbon accounting and a shared-capacity extension without disturbing the canonical pipeline.",
        source_files=[snapshot["source_paths"].get("stage10_env_breakdown", ""), snapshot["source_paths"].get("stage10_shared_comparison", ""), snapshot["source_paths"].get("stage10_pair_carbon", ""), snapshot["source_paths"].get("stage10_shared_solution", "")],
        output_files=[],
        warnings=_missing_warnings([env_breakdown, shared]),
    )
    return PaperTable(spec=spec, data=data, formatted=formatted)


def _make_table_saa_manifest(snapshot: Dict[str, Any], config: Stage11Config) -> PaperTable:
    rows = [
        _json_manifest_row(snapshot, "stage4_solution"),
        _json_manifest_row(snapshot, "stage4_instance_summary"),
        _csv_manifest_row(snapshot, "stage9_saa", group_column="scenario_setting", expected_count=3),
        _csv_manifest_row(snapshot, "stage9_sensitivity", group_column="parameter", expected_count=4),
    ]
    data = pd.DataFrame(rows)
    formatted = data.copy()
    spec = PaperArtifactSpec(
        artifact_id="T7_saa_sensitivity_manifest",
        artifact_type="table",
        title="SAA stability and sensitivity manifest",
        claim="The report records what robustness evidence is available and what remains optional.",
        source_files=[snapshot["source_paths"].get("stage9_saa", ""), snapshot["source_paths"].get("stage9_sensitivity", "")],
        output_files=[],
        warnings=_missing_warnings([_read_csv(snapshot, "stage9_saa"), _read_csv(snapshot, "stage9_sensitivity")]),
    )
    return PaperTable(spec=spec, data=data, formatted=formatted)


def _json_manifest_row(snapshot: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = _read_json_dict(snapshot, key)
    status = "available" if value else "missing"
    return {
        "source_key": key,
        "status": status,
        "row_count": 1 if value else 0,
        "available_groups": "",
        "failed_rows": 0,
        "success_rows": 1 if value else 0,
        "summary": json.dumps(value, ensure_ascii=False)[:300] if value else "",
    }


def _csv_manifest_row(
    snapshot: Dict[str, Any],
    key: str,
    group_column: str,
    expected_count: int,
) -> Dict[str, Any]:
    frame = _read_csv(snapshot, key)
    if frame.empty:
        return {
            "source_key": key,
            "status": "missing",
            "row_count": 0,
            "available_groups": "",
            "failed_rows": 0,
            "success_rows": 0,
            "summary": "",
        }
    success_series = frame.get("success", pd.Series([True] * len(frame)))
    success_mask = success_series.astype(str).str.lower().isin(["true", "1", "yes"])
    failed_rows = int((~success_mask).sum())
    groups = []
    if group_column in frame.columns:
        groups = sorted({str(value) for value in frame[group_column].dropna().tolist()})
    status = "available"
    if failed_rows:
        status = "available_with_failure"
    elif expected_count and len(groups) < expected_count:
        status = "available_partial"
    summary_payload = {
        "rows": int(len(frame)),
        "groups": groups,
        "success_rows": int(success_mask.sum()),
        "failed_rows": failed_rows,
    }
    return {
        "source_key": key,
        "status": status,
        "row_count": int(len(frame)),
        "available_groups": ", ".join(groups),
        "failed_rows": failed_rows,
        "success_rows": int(success_mask.sum()),
        "summary": json.dumps(summary_payload, ensure_ascii=False),
    }


def _stage_row(snapshot: Dict[str, Any], stage_num: int, notes: str) -> Dict[str, Any]:
    stage_name = f"stage{stage_num}"
    model = snapshot.get("model_summaries", {}).get(stage_name, {})
    if stage_num in {1, 9, 10}:
        return {"status": "reporting", "variable_count": None, "constraint_count": None, "notes": notes}
    return {
        "status": "collected",
        "variable_count": _model_summary_value(model, "variable_count")
        or _model_summary_value(model, "objective_vector_summary", "economic_risk", "length")
        or _model_summary_value(model, "shared_capacity_row_count"),
        "constraint_count": _model_summary_value(model, "constraint_count") or _model_summary_value(model, "constraint_count_total"),
        "notes": notes,
    }


def _instance_label(snapshot: Dict[str, Any], stage: str) -> str:
    instance = snapshot.get("json", {}).get(f"{stage}_instance_summary", {})
    if not instance:
        model = snapshot.get("model_summaries", {}).get(stage, {})
        instance = model.get("instance", {}) if isinstance(model, dict) else {}
    machine = instance.get("machine_type_id") or instance.get("machine_types")
    if isinstance(machine, list):
        machine = "+".join(machine)
    if not machine:
        machine = "n/a"
    return str(machine)


def _stage_check_summary(snapshot: Dict[str, Any], stage: str) -> str:
    checks = snapshot.get("checks", {}).get(stage, {})
    if not checks:
        return "n/a"
    summary = checks.get("summary", {})
    if summary:
        return ", ".join(f"{k}={v}" for k, v in summary.items())
    return "available"


def _extract_number(snapshot: Dict[str, Any], stage_num: int, key: str) -> Any:
    stage_name = f"stage{stage_num}"
    model = snapshot.get("model_summaries", {}).get(stage_name, {})
    if not isinstance(model, dict):
        return None
    return _model_summary_value(model, key)


def _model_summary_value(model: Dict[str, Any], *keys: str) -> Any:
    """Read a value from either the model summary root or nested model_summary."""

    if not isinstance(model, dict) or not keys:
        return None
    for root in (model, model.get("model_summary", {})):
        current: Any = root
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if current is not None:
            return current
    return None


def _stage_number(stage: str) -> int | None:
    text = str(stage).strip().lower().replace("stage", "")
    try:
        return int(text)
    except ValueError:
        return None


def _read_csv(snapshot: Dict[str, Any], key: str) -> pd.DataFrame:
    value = snapshot.get("csv", {}).get(key)
    return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _read_json_dict(snapshot: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = snapshot.get("json", {}).get(key)
    return value if isinstance(value, dict) else {}


def _ensure_columns(frame: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=columns)
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    return frame


def _format_numeric_table(frame: pd.DataFrame, float_cols: List[str] | None = None, integer_cols: List[str] | None = None) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    formatted = frame.copy()
    float_cols = float_cols or []
    integer_cols = integer_cols or []
    for column in float_cols:
        if column in formatted.columns:
            formatted[column] = pd.to_numeric(formatted[column], errors="coerce").map(lambda value: _fmt_float(value))
    for column in integer_cols:
        if column in formatted.columns:
            formatted[column] = pd.to_numeric(formatted[column], errors="coerce").map(lambda value: _fmt_int(value))
    return formatted


def _fmt_float(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):,.4f}"


def _fmt_int(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{int(round(float(value))):,d}"


def _missing_warnings(frames: List[Any]) -> List[str]:
    warnings = []
    for index, frame in enumerate(frames, start=1):
        if isinstance(frame, pd.DataFrame) and frame.empty:
            warnings.append(f"source_{index}_missing_or_empty")
        elif isinstance(frame, dict) and not frame:
            warnings.append(f"source_{index}_missing_or_empty")
        elif frame is None:
            warnings.append(f"source_{index}_missing_or_empty")
    return warnings


def _compact_dict(value: Any) -> str:
    if not isinstance(value, dict):
        return "" if value is None else str(value)
    items = list(value.items())[:6]
    return "; ".join(f"{k}:{v}" for k, v in items)
