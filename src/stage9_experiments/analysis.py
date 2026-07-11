"""Derived experiment tables and checks for Stage 9."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .config import Stage9Config
from .manifest import metric_dictionary
from .structures import ExperimentSuiteResult


def build_suite_result(manifest: pd.DataFrame, all_results: pd.DataFrame, config: Stage9Config) -> ExperimentSuiteResult:
    """Build all Stage 9 derived tables."""

    baseline = _baseline_comparison(all_results)
    ablation = _ablation_study(all_results)
    saa = _saa_stability(all_results)
    sensitivity = _sensitivity_summary(all_results)
    exact_gap = _exact_vs_matheuristic(all_results)
    top5 = _top5_summary(all_results, config)
    checks = _checks(manifest, all_results, baseline, ablation, exact_gap, top5)
    success = not any(check["severity"] == "failed" for check in checks)
    return ExperimentSuiteResult(
        success=success,
        status_message="Stage 9 experiment suite completed." if success else "Stage 9 experiment suite completed with failed checks.",
        manifest=manifest,
        metric_dictionary=metric_dictionary(),
        all_experiment_results=all_results,
        baseline_comparison=baseline,
        ablation_study=ablation,
        saa_stability=saa,
        sensitivity_summary=sensitivity,
        exact_vs_matheuristic_gap=exact_gap,
        top5_benchmark_summary=top5,
        experiment_checks=checks,
    )


def _baseline_comparison(results: pd.DataFrame) -> pd.DataFrame:
    baseline = results[results["experiment_group"] == "baseline_comparison"].copy()
    if baseline.empty:
        return pd.DataFrame()
    order = {
        "baseline_stage3_deterministic": 1,
        "baseline_stage4_saa": 2,
        "baseline_stage5_cvar": 3,
        "baseline_stage6_selective_assembly": 4,
        "baseline_stage7_pareto_anchor": 5,
        "baseline_stage8_matheuristic": 6,
        "rule_br14_risk_aware": 7,
        "rule_br08_selective_assembly": 8,
        "rule_br18_no_selective_assembly": 9,
    }
    baseline["sort_order"] = baseline["experiment_id"].map(order).fillna(999).astype(int)
    columns = [
        "sort_order",
        "experiment_id",
        "model_stage",
        "status",
        "success",
        "objective_value",
        "economic_risk",
        "environmental_impact",
        "assembly_quality_loss",
        "expected_assembled_units",
        "expected_final_backlog_units",
        "cvar_value",
        "expected_assembly_shortfall_units",
        "solve_seconds",
        "pareto_points",
        "warning",
    ]
    return baseline[[column for column in columns if column in baseline.columns]].sort_values("sort_order").reset_index(drop=True)


def _ablation_study(results: pd.DataFrame) -> pd.DataFrame:
    lookup = results.set_index("experiment_id")
    pairs = [
        ("ablation_no_stochasticity", "baseline_stage3_deterministic", "baseline_stage4_saa"),
        ("ablation_no_cvar", "baseline_stage4_saa", "baseline_stage5_cvar"),
        ("ablation_no_selective_assembly", "baseline_stage5_cvar", "baseline_stage6_selective_assembly"),
        ("ablation_no_pareto", "baseline_stage6_selective_assembly", "baseline_stage7_pareto_anchor"),
        ("ablation_matheuristic_approximation", "baseline_stage7_pareto_anchor", "baseline_stage8_matheuristic"),
    ]
    rows = []
    for experiment_id, before_id, after_id in pairs:
        before = lookup.loc[before_id] if before_id in lookup.index else pd.Series(dtype=object)
        after = lookup.loc[after_id] if after_id in lookup.index else pd.Series(dtype=object)
        rows.append(
            {
                "experiment_id": experiment_id,
                "before_experiment_id": before_id,
                "after_experiment_id": after_id,
                "before_stage": before.get("model_stage"),
                "after_stage": after.get("model_stage"),
                "before_status": before.get("status"),
                "after_status": after.get("status"),
                "objective_delta": _delta(before, after, "objective_value"),
                "economic_risk_delta": _delta(before, after, "economic_risk"),
                "expected_backlog_delta": _delta(before, after, "expected_final_backlog_units"),
                "cvar_delta": _delta(before, after, "cvar_value"),
                "assembly_shortfall_delta": _delta(before, after, "expected_assembly_shortfall_units"),
                "runtime_delta_seconds": _delta(before, after, "solve_seconds"),
            }
        )
    return pd.DataFrame(rows)


def _saa_stability(results: pd.DataFrame) -> pd.DataFrame:
    saa = results[results["experiment_group"] == "saa_stability"].copy()
    if saa.empty:
        return pd.DataFrame()
    saa["scenario_setting"] = saa["experiment_id"].str.extract(r"scenario_(\d+)", expand=False)
    return saa[
        [
            "experiment_id",
            "scenario_setting",
            "scenario_count",
            "status",
            "success",
            "objective_value",
            "expected_final_backlog_units",
            "cvar_value",
            "solve_seconds",
            "warning",
        ]
    ].reset_index(drop=True)


def _sensitivity_summary(results: pd.DataFrame) -> pd.DataFrame:
    sensitivity = results[results["experiment_group"] == "sensitivity_analysis"].copy()
    if sensitivity.empty:
        return pd.DataFrame()
    extracted = sensitivity["experiment_id"].str.extract(r"sensitivity_(.+)_([^_]+)$")
    sensitivity["parameter"] = extracted[0]
    sensitivity["level"] = extracted[1]
    columns = [
        "experiment_id",
        "parameter",
        "level",
        "status",
        "success",
        "objective_value",
        "expected_final_backlog_units",
        "cvar_value",
        "expected_assembly_shortfall_units",
        "solve_seconds",
        "warning",
    ]
    return sensitivity[[column for column in columns if column in sensitivity.columns]].reset_index(drop=True)


def _exact_vs_matheuristic(results: pd.DataFrame) -> pd.DataFrame:
    lookup = results.set_index("experiment_id")
    exact = lookup.loc["baseline_stage7_pareto_anchor"] if "baseline_stage7_pareto_anchor" in lookup.index else pd.Series(dtype=object)
    approx = lookup.loc["baseline_stage8_matheuristic"] if "baseline_stage8_matheuristic" in lookup.index else pd.Series(dtype=object)
    row = {
        "comparison_id": "stage7_exact_vs_stage8_matheuristic",
        "exact_status": exact.get("status"),
        "approx_status": approx.get("status"),
        "exact_pareto_points": exact.get("pareto_points"),
        "approx_pareto_points": approx.get("pareto_points"),
        "exact_min_economic_risk": exact.get("economic_risk"),
        "approx_min_economic_risk": approx.get("economic_risk"),
        "economic_risk_gap_abs": _gap_abs(exact.get("economic_risk"), approx.get("economic_risk")),
        "economic_risk_gap_pct": _gap_pct(exact.get("economic_risk"), approx.get("economic_risk")),
        "environmental_gap_abs": _gap_abs(exact.get("environmental_impact"), approx.get("environmental_impact")),
        "assembly_quality_gap_abs": _gap_abs(exact.get("assembly_quality_loss"), approx.get("assembly_quality_loss")),
        "exact_runtime_seconds": exact.get("solve_seconds"),
        "approx_runtime_seconds": approx.get("solve_seconds"),
        "runtime_ratio_approx_over_exact": _ratio(approx.get("solve_seconds"), exact.get("solve_seconds")),
    }
    return pd.DataFrame([row])


def _top5_summary(results: pd.DataFrame, config: Stage9Config) -> pd.DataFrame:
    benchmark_path = config.data_results_dir / "stage8" / "large_benchmark_summary.csv"
    if not benchmark_path.exists():
        row = results[results["experiment_id"] == "large_benchmark_top5_52w"].copy()
        return row.reset_index(drop=True)
    return pd.read_csv(benchmark_path, encoding="utf-8-sig")


def _checks(
    manifest: pd.DataFrame,
    results: pd.DataFrame,
    baseline: pd.DataFrame,
    ablation: pd.DataFrame,
    exact_gap: pd.DataFrame,
    top5: pd.DataFrame,
) -> List[Dict[str, object]]:
    required = manifest[manifest["required"].astype(bool)]
    required_results = results[results["experiment_id"].isin(required["experiment_id"])]
    required_collected = required_results["status"].isin(["collected", "collected_empty"]).all() if not required_results.empty else False
    top5_success = _top5_success(top5)
    warnings = int(results["warning"].notna().sum()) if "warning" in results.columns else 0
    return [
        _check("experiment_manifest_nonempty", not manifest.empty, f"Manifest rows={len(manifest)}."),
        _check("required_results_detected", required_collected, f"Required collected={int(required_results['status'].isin(['collected', 'collected_empty']).sum())}/{len(required)}."),
        _check("baseline_table_generated", not baseline.empty, f"Baseline rows={len(baseline)}."),
        _check("ablation_table_generated", not ablation.empty, f"Ablation rows={len(ablation)}."),
        _check("exact_vs_matheuristic_generated", not exact_gap.empty, f"Exact-vs-matheuristic rows={len(exact_gap)}."),
        _check("top5_benchmark_success", top5_success, "Stage 8 top5 benchmark reports all instances successful when available."),
        _check("optional_missing_are_warnings", True, f"Rows with warnings={warnings}.", severity="warning" if warnings else "passed"),
    ]


def _top5_success(top5: pd.DataFrame) -> bool:
    if top5.empty or "success" not in top5.columns:
        return False
    values = top5["success"].astype(str).str.lower().isin(["true", "1", "yes"])
    return bool(len(values) >= 5 and values.all())


def _delta(before: pd.Series, after: pd.Series, column: str) -> float | None:
    before_value = _as_float(before.get(column))
    after_value = _as_float(after.get(column))
    if before_value is None or after_value is None:
        return None
    return after_value - before_value


def _gap_abs(exact: object, approx: object) -> float | None:
    exact_value = _as_float(exact)
    approx_value = _as_float(approx)
    if exact_value is None or approx_value is None:
        return None
    return approx_value - exact_value


def _gap_pct(exact: object, approx: object) -> float | None:
    exact_value = _as_float(exact)
    gap = _gap_abs(exact, approx)
    if exact_value in [None, 0] or gap is None:
        return None
    return gap / abs(exact_value) * 100.0


def _ratio(numerator: object, denominator: object) -> float | None:
    n = _as_float(numerator)
    d = _as_float(denominator)
    if n is None or d in [None, 0]:
        return None
    return n / d


def _as_float(value: object) -> float | None:
    try:
        parsed = pd.to_numeric(value, errors="coerce")
        return float(parsed) if pd.notna(parsed) and np.isfinite(float(parsed)) else None
    except (TypeError, ValueError):
        return None


def _check(name: str, passed: bool, message: str, severity: str | None = None) -> Dict[str, object]:
    return {
        "check_name": name,
        "severity": severity or ("passed" if passed else "failed"),
        "message": message,
    }
