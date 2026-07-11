"""Operational realism audit for remanufacturing manuscript results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .config import Stage12Config


STAGE_LABELS = {
    "stage3": "Stage3",
    "stage4": "Stage4",
    "stage5": "Stage5",
    "stage6": "Stage6",
}


def build_operational_reality_audit(config: Stage12Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a read-only audit of whether results match remanufacturing practice."""

    metrics = _collect_stage_metrics(config)
    rows: List[Dict[str, Any]] = []
    adjustments: List[Dict[str, Any]] = []

    _audit_remanufacturing_activity(rows, adjustments, metrics)
    _audit_procurement_fallback(rows, adjustments, metrics)
    _audit_delivery_realism(rows, adjustments, metrics)
    _audit_route_realism(rows, adjustments, metrics)
    _audit_selective_assembly_realism(rows, adjustments, metrics)
    _audit_robustness_realism(config, rows, adjustments)

    return pd.DataFrame(rows), pd.DataFrame(adjustments)


def _collect_stage_metrics(config: Stage12Config) -> Dict[str, Dict[str, Any]]:
    baseline = _safe_read_csv(config.data_results_dir / "stage9" / "baseline_comparison.csv")
    metrics: Dict[str, Dict[str, Any]] = {}
    for stage_key, label in STAGE_LABELS.items():
        stage_dir = config.data_results_dir / stage_key
        first_stage = _first_stage_metrics(stage_dir)
        summary = _summary_metrics(stage_dir / "solution_summary.json")
        assembly = _assembly_metrics(stage_key, stage_dir, baseline)
        route = _route_metrics(stage_key, stage_dir)
        metrics[stage_key] = {
            "label": label,
            "stage_dir": stage_dir,
            **first_stage,
            **summary,
            **assembly,
            **route,
        }
        if metrics[stage_key].get("core_count") is None and metrics[stage_key].get("json_core_count") is not None:
            metrics[stage_key]["core_count"] = metrics[stage_key].get("json_core_count")
            metrics[stage_key]["accepted_cores"] = metrics[stage_key].get("json_accepted_cores")
            metrics[stage_key]["acceptance_rate"] = metrics[stage_key].get("json_acceptance_rate")
    metrics["stage6"].update(_stage6_assembly_metrics(config.data_results_dir / "stage6"))
    return metrics


def _audit_remanufacturing_activity(
    rows: List[Dict[str, Any]],
    adjustments: List[Dict[str, Any]],
    metrics: Dict[str, Dict[str, Any]],
) -> None:
    for stage_key in ["stage3", "stage4", "stage5", "stage6"]:
        metric = metrics[stage_key]
        accepted = metric.get("accepted_cores")
        total = metric.get("core_count")
        route_count = int(metric.get("selected_route_count", 0) or 0)
        accept_rate = _safe_ratio(accepted, total)
        value = f"accepted={_fmt(accepted)}/{_fmt(total)}; accept_rate={_fmt(accept_rate)}; selected_routes={route_count}"
        if route_count > 0 and (accept_rate is None or accept_rate > 0):
            decision = "realistic_main_claim"
            evidence = "main_text"
            status = "passed"
            flag = "remanufacturing_activity_present"
            message = f"{metric['label']} contains old-core acceptance and selected remanufacturing routes."
            writing = "Use as evidence that the model can express active remanufacturing and route selection."
        else:
            decision = "explain_as_risk_averse_behavior"
            evidence = "main_text_with_caution"
            status = "warning"
            flag = "low_remanufacturing_activity"
            message = f"{metric['label']} has zero accepted cores or no selected old-part routes."
            writing = "Do not claim high reuse; explain the result as conservative reliability or assembly-risk behavior."
            adjustments.append(
                _adjustment(
                    "low_remanufacturing_activity",
                    metric["label"],
                    value,
                    "Frame this stage as the cost of strict risk controls, not as a high-reuse operating policy.",
                )
            )
        _row(
            rows,
            check_id=f"{stage_key}_remanufacturing_activity",
            dimension="remanufacturing_activity",
            model_stage=metric["label"],
            status=status,
            decision=decision,
            evidence=evidence,
            metric_name="acceptance_and_route_activity",
            metric_value=value,
            benchmark_value="positive accepted cores and selected routes",
            threshold="selected_routes > 0 and accepted_cores > 0 when acceptance is reported",
            flag=flag,
            message=message,
            sources=_stage_sources(metric, ["first_stage_decisions.csv", "selected routes", "solution_summary.json"]),
            writing=writing,
        )


def _audit_procurement_fallback(
    rows: List[Dict[str, Any]],
    adjustments: List[Dict[str, Any]],
    metrics: Dict[str, Dict[str, Any]],
) -> None:
    stage6 = metrics["stage6"]
    new_share = stage6.get("selected_candidate_new_share")
    old_share = stage6.get("selected_candidate_old_share")
    selected_candidates = stage6.get("selected_candidate_count", 0)
    metric_value = (
        f"selected_candidates={_fmt(selected_candidates)}; "
        f"new_or_replacement_share={_fmt(new_share)}; old_candidate_share={_fmt(old_share)}"
    )
    if selected_candidates and new_share is not None and new_share >= 0.8:
        status = "warning"
        decision = "explain_as_risk_averse_behavior"
        evidence = "main_text_with_caution"
        flag = "fallback_dominant_solution"
        message = "Stage 6 selective assembly is dominated by new or replacement candidates."
        writing = "Explain procurement fallback as a conservative response to strict reliability and compatibility screens."
        adjustments.append(
            _adjustment(
                "procurement_fallback_dominance",
                "Stage6",
                metric_value,
                "Avoid presenting Stage 6 as broad old-part reuse; present it as reliability-driven fallback.",
            )
        )
    else:
        status = "passed"
        decision = "realistic_main_claim"
        evidence = "main_text"
        flag = "mixed_source_solution"
        message = "Selected assembly candidates include a non-dominant fallback mix."
        writing = "Use as a mixed-source selective assembly result."
    _row(
        rows,
        check_id="stage6_procurement_fallback",
        dimension="procurement_fallback",
        model_stage="Stage6",
        status=status,
        decision=decision,
        evidence=evidence,
        metric_name="selected_candidate_source_mix",
        metric_value=metric_value,
        benchmark_value="new/replacement share below 0.80 for a strong reuse claim",
        threshold="new_or_replacement_share >= 0.80 indicates fallback dominance",
        flag=flag,
        message=message,
        sources=_stage_sources(stage6, ["selected_assembly_candidates.csv", "first_stage_decisions.csv"]),
        writing=writing,
    )


def _audit_delivery_realism(
    rows: List[Dict[str, Any]],
    adjustments: List[Dict[str, Any]],
    metrics: Dict[str, Dict[str, Any]],
) -> None:
    stage4 = metrics["stage4"]
    stage5 = metrics["stage5"]
    stage6 = metrics["stage6"]
    stage4_backlog = _to_float(stage4.get("expected_final_backlog_units"))
    stage5_backlog = _to_float(stage5.get("expected_final_backlog_units"))
    stage5_assembled = _to_float(stage5.get("expected_assembled_units"))
    stage6_assembled = _to_float(stage6.get("expected_assembled_units"))
    stage6_backlog = _to_float(stage6.get("expected_final_backlog_units"))
    assembled_ratio = _safe_ratio(stage6_assembled, stage5_assembled)

    if stage5_backlog is not None and stage4_backlog is not None and stage5_backlog < stage4_backlog:
        _row(
            rows,
            check_id="stage5_delivery_improvement",
            dimension="delivery_realism",
            model_stage="Stage5",
            status="passed",
            decision="realistic_main_claim",
            evidence="main_text",
            metric_name="stage4_to_stage5_backlog_change",
            metric_value=f"stage4_backlog={_fmt(stage4_backlog)}; stage5_backlog={_fmt(stage5_backlog)}",
            benchmark_value="Stage 5 backlog should not be worse if risk controls improve recourse choices",
            threshold="stage5_backlog < stage4_backlog",
            flag="risk_model_improves_delivery",
            message="CVaR/risk modeling improves expected delivery relative to the SAA baseline in the current instance.",
            sources=_stage_sources(stage4, ["scenario_assembly_plan.csv"]) + ";" + _stage_sources(stage5, ["scenario_assembly_plan.csv"]),
            writing="Use Stage 4 to Stage 5 as the cleanest operational improvement claim.",
        )
    if assembled_ratio is not None and assembled_ratio < 0.20:
        decision = "explain_as_risk_averse_behavior"
        evidence = "main_text_with_caution"
        status = "warning"
        flag = "selective_assembly_over_restrictive"
        writing = "Present Stage 6 as showing the delivery cost of strict selective assembly, not as a production-volume improvement."
        adjustments.append(
            _adjustment(
                "selective_assembly_delivery_drop",
                "Stage6",
                f"stage6/stage5 assembled ratio={_fmt(assembled_ratio)}; stage6 backlog={_fmt(stage6_backlog)}",
                writing,
            )
        )
    else:
        decision = "realistic_main_claim"
        evidence = "main_text"
        status = "passed"
        flag = "delivery_scale_preserved"
        writing = "Use as evidence that selective assembly preserves delivery scale."
    _row(
        rows,
        check_id="stage6_delivery_realism",
        dimension="delivery_realism",
        model_stage="Stage6",
        status=status,
        decision=decision,
        evidence=evidence,
        metric_name="stage6_assembled_relative_to_stage5",
        metric_value=(
            f"stage5_assembled={_fmt(stage5_assembled)}; stage6_assembled={_fmt(stage6_assembled)}; "
            f"ratio={_fmt(assembled_ratio)}; stage6_backlog={_fmt(stage6_backlog)}"
        ),
        benchmark_value="Stage 6 assembled output should stay above 20% of Stage 5 for a production-volume claim",
        threshold="stage6_assembled / stage5_assembled < 0.20 indicates over-restrictive assembly",
        flag=flag,
        message="Stage 6 sharply reduces assembled output relative to Stage 5." if status == "warning" else "Stage 6 delivery scale is comparable to Stage 5.",
        sources=_stage_sources(stage5, ["scenario_assembly_plan.csv"]) + ";" + _stage_sources(stage6, ["scenario_assembly_plan.csv"]),
        writing=writing,
    )


def _audit_route_realism(
    rows: List[Dict[str, Any]],
    adjustments: List[Dict[str, Any]],
    metrics: Dict[str, Dict[str, Any]],
) -> None:
    for stage_key in ["stage3", "stage4", "stage5", "stage6"]:
        metric = metrics[stage_key]
        mix = metric.get("route_mix", {})
        route_count = int(metric.get("selected_route_count", 0) or 0)
        if route_count > 0:
            decision = "realistic_main_claim"
            evidence = "main_text"
            status = "passed"
            flag = "route_mix_observed"
            message = f"{metric['label']} produces a traceable route mix."
            writing = "Use route mix to explain how model layers alter remanufacturing operations."
        else:
            decision = "explain_as_risk_averse_behavior"
            evidence = "main_text_with_caution"
            status = "warning"
            flag = "no_old_route_selected"
            message = f"{metric['label']} has no selected old-part route records."
            writing = "Use only to explain that reliability filters and fallback variables replace remanufacturing routes."
            adjustments.append(
                _adjustment(
                    "empty_route_mix",
                    metric["label"],
                    "selected_route_count=0",
                    "Avoid using this stage for route-preference managerial claims.",
                )
            )
        _row(
            rows,
            check_id=f"{stage_key}_route_realism",
            dimension="route_realism",
            model_stage=metric["label"],
            status=status,
            decision=decision,
            evidence=evidence,
            metric_name="route_mix",
            metric_value=_format_mix(mix),
            benchmark_value="nonzero selected route records for route-policy claims",
            threshold="selected_route_count > 0",
            flag=flag,
            message=message,
            sources=_stage_sources(metric, ["selected routes"]),
            writing=writing,
        )


def _audit_selective_assembly_realism(
    rows: List[Dict[str, Any]],
    adjustments: List[Dict[str, Any]],
    metrics: Dict[str, Dict[str, Any]],
) -> None:
    stage6 = metrics["stage6"]
    mean_coverage = _to_float(stage6.get("mean_feature_coverage_rate"))
    shortfall_features = _to_float(stage6.get("shortfall_feature_rows"))
    selected_pairs = _to_float(stage6.get("selected_pair_count"))
    hard_pair_share = _to_float(stage6.get("hard_pair_share"))
    metric_value = (
        f"mean_coverage={_fmt(mean_coverage)}; shortfall_feature_rows={_fmt(shortfall_features)}; "
        f"selected_pairs={_fmt(selected_pairs)}; hard_pair_share={_fmt(hard_pair_share)}"
    )
    if shortfall_features and shortfall_features > 0:
        status = "warning"
        decision = "explain_as_risk_averse_behavior"
        evidence = "main_text_with_caution"
        flag = "partial_selective_assembly_coverage"
        message = "Stage 6 covers most selective-assembly features but leaves recurring feature-level shortfall."
        writing = "State that strict compatibility improves quality screening but creates shortfall for difficult features."
        adjustments.append(
            _adjustment(
                "selective_assembly_shortfall",
                "Stage6",
                metric_value,
                "Use Stage 6 as evidence of compatibility-screening trade-offs rather than universal assembly success.",
            )
        )
    else:
        status = "passed"
        decision = "realistic_main_claim"
        evidence = "main_text"
        flag = "selective_assembly_coverage_complete"
        message = "Stage 6 selective assembly has complete feature coverage."
        writing = "Use as direct selective-assembly feasibility evidence."
    _row(
        rows,
        check_id="stage6_selective_assembly_realism",
        dimension="selective_assembly_realism",
        model_stage="Stage6",
        status=status,
        decision=decision,
        evidence=evidence,
        metric_name="feature_coverage_and_pair_quality",
        metric_value=metric_value,
        benchmark_value="zero feature shortfall for a strong assembly-coverage claim",
        threshold="shortfall_feature_rows > 0 requires risk-averse interpretation",
        flag=flag,
        message=message,
        sources=_stage_sources(stage6, ["feature_assembly_plan.csv", "selected_assembly_pairs.csv"]),
        writing=writing,
    )


def _audit_robustness_realism(
    config: Stage12Config,
    rows: List[Dict[str, Any]],
    adjustments: List[Dict[str, Any]],
) -> None:
    saa = _safe_read_csv(config.data_results_dir / "stage9" / "saa_stability.csv")
    if not saa.empty:
        successful = saa[saa.get("success", pd.Series(dtype=object)).astype(str).str.lower().isin(["true", "1", "yes"])]
        failed = saa[~saa.index.isin(successful.index)]
        _row(
            rows,
            check_id="saa_robustness_realism",
            dimension="robustness_realism",
            model_stage="Stage9",
            status="warning" if not failed.empty else "passed",
            decision="realistic_main_claim" if failed.empty else "explain_as_risk_averse_behavior",
            evidence="main_text_with_caution" if not failed.empty else "main_text",
            metric_name="saa_successful_scenario_counts",
            metric_value=(
                f"successful={','.join(successful.get('scenario_setting', pd.Series(dtype=object)).astype(str).tolist())}; "
                f"failed_or_audit_only={','.join(failed.get('scenario_setting', pd.Series(dtype=object)).astype(str).tolist())}"
            ),
            benchmark_value="failed SAA runs should not support main robustness claims",
            threshold="success must be true for main SAA evidence",
            flag="partial_saa_robustness",
            message="SAA robustness is usable for successful scenario counts; failed runs remain appendix/audit evidence.",
            sources=str(config.data_results_dir / "stage9" / "saa_stability.csv"),
            writing="Use 9/18-scenario stability in the main text; keep failed 27-scenario quick run out of main claims.",
        )
    sensitivity = _safe_read_csv(config.data_results_dir / "stage9" / "sensitivity_summary.csv")
    if not sensitivity.empty:
        for parameter in ["min_system_reliability", "env_weight"]:
            subset = sensitivity[sensitivity.get("parameter", pd.Series(dtype=object)).astype(str) == parameter]
            flat = bool(not subset.empty and _all_metrics_flat(subset))
            _row(
                rows,
                check_id=f"sensitivity_{parameter}_operational_realism",
                dimension="robustness_realism",
                model_stage="Stage9",
                status="warning" if flat else "passed",
                decision="do_not_use_as_main_claim" if flat else "realistic_main_claim",
                evidence="appendix_or_audit" if flat else "main_text",
                metric_name="sensitivity_response",
                metric_value="flat_response_detected" if flat else "nonflat_response_detected",
                benchmark_value="non-flat operational response for strong sensitivity claims",
                threshold="all key metrics flat means appendix-only or true-insensitive explanation",
                flag="flat_response_warning" if flat else "sensitivity_response_present",
                message=f"{parameter} sensitivity is flat in the current quick instance." if flat else f"{parameter} sensitivity changes model outputs.",
                sources=str(config.data_results_dir / "stage9" / "sensitivity_summary.csv"),
                writing="Do not use this axis as a strong robustness claim; mention only as flat-response appendix evidence." if flat else "Use as sensitivity evidence.",
            )
            if flat:
                adjustments.append(
                    _adjustment(
                        "flat_sensitivity_appendix_only",
                        parameter,
                        "all key metrics unchanged across levels",
                        "Keep this sensitivity axis out of the main managerial conclusion.",
                    )
                )
    top5 = _safe_read_csv(config.data_results_dir / "stage8" / "large_benchmark_summary.csv")
    if top5.empty:
        top5 = _safe_read_csv(config.data_results_dir / "stage9" / "top5_benchmark_summary.csv")
    if not top5.empty:
        zero_best = (
            pd.to_numeric(top5.get("best_environmental_impact", pd.Series(dtype=object)), errors="coerce").fillna(0).eq(0)
            & pd.to_numeric(top5.get("incumbent_environmental_impact", pd.Series(dtype=object)), errors="coerce").fillna(0).gt(0)
        )
        zero_quality = (
            pd.to_numeric(top5.get("best_assembly_quality_loss", pd.Series(dtype=object)), errors="coerce").fillna(0).eq(0)
            & pd.to_numeric(top5.get("incumbent_assembly_quality_loss", pd.Series(dtype=object)), errors="coerce").fillna(0).gt(0)
        )
        misleading_zero = bool((zero_best | zero_quality).any())
        _row(
            rows,
            check_id="stage8_top5_zero_objective_coordinates",
            dimension="robustness_realism",
            model_stage="Stage8",
            status="warning" if misleading_zero else "passed",
            decision="do_not_use_as_main_claim" if misleading_zero else "realistic_main_claim",
            evidence="appendix_or_audit" if misleading_zero else "main_text",
            metric_name="best_vs_incumbent_environmental_and_assembly_metrics",
            metric_value=f"rows_with_zero_coordinate_warning={int((zero_best | zero_quality).sum())}",
            benchmark_value="zero objective coordinates must not be narrated as literal zero impact",
            threshold="best objective coordinate equals 0 while incumbent metric is positive",
            flag="pareto_coordinate_not_literal_impact" if misleading_zero else "top5_metrics_consistent",
            message="Stage 8 top5 best environmental/assembly values include zero objective coordinates while incumbent impacts are positive.",
            sources=str(config.data_results_dir / "stage8" / "large_benchmark_summary.csv"),
            writing="Use Stage 8 for runtime, feasibility and approximate Pareto coverage; do not claim literal zero carbon or zero assembly loss.",
        )
        if misleading_zero:
            adjustments.append(
                _adjustment(
                    "stage8_zero_coordinate_caution",
                    "Stage8 top5 benchmark",
                    "best_environmental_impact or best_assembly_quality_loss equals 0 while incumbent metrics are positive",
                    "Interpret these as Pareto/objective-coordinate values, not physical zero impacts.",
                )
            )


def _first_stage_metrics(stage_dir: Path) -> Dict[str, Any]:
    path = stage_dir / "first_stage_decisions.csv"
    table = _safe_read_csv(path)
    if table.empty:
        return {
            "core_count": None,
            "accepted_cores": None,
            "acceptance_rate": None,
            "pre_procure_units": None,
        }
    core_rows = table[table.get("core_id", pd.Series(dtype=object)).notna()].copy()
    core_rows = core_rows[core_rows.get("core_id", pd.Series(dtype=object)).astype(str).str.strip() != ""]
    accepted = pd.to_numeric(core_rows.get("accept_core", pd.Series(dtype=object)), errors="coerce").fillna(0).sum()
    procure = pd.to_numeric(table.get("pre_procure_units", pd.Series(dtype=object)), errors="coerce").fillna(0).sum()
    total = len(core_rows)
    return {
        "core_count": int(total) if total else None,
        "accepted_cores": float(accepted) if total else None,
        "acceptance_rate": _safe_ratio(accepted, total),
        "pre_procure_units": float(procure),
        "first_stage_source": str(path),
    }


def _summary_metrics(path: Path) -> Dict[str, Any]:
    payload = _safe_read_json(path)
    if not payload:
        return {"solution_summary_source": str(path)}
    solution = payload.get("solution", {})
    variables = solution.get("variables", [])
    accept_values = [
        _to_float(var.get("value"))
        for var in variables
        if str(var.get("variable_name", "")).startswith("accept_core[")
    ]
    summary_metrics = payload.get("summary_metrics", {})
    accepted = sum(value for value in accept_values if value is not None)
    result = {
        "success": solution.get("success"),
        "objective_value": solution.get("objective_value"),
        "solve_seconds": solution.get("solve_seconds"),
        "solution_summary_source": str(path),
    }
    if accept_values:
        result.update(
            {
                "json_core_count": len(accept_values),
                "json_accepted_cores": accepted,
                "json_acceptance_rate": _safe_ratio(accepted, len(accept_values)),
            }
        )
    for key in [
        "expected_assembled_units",
        "expected_final_backlog_units",
        "expected_assembly_shortfall_units",
        "route_mix",
    ]:
        if key in summary_metrics:
            result[key] = summary_metrics[key]
    return result


def _assembly_metrics(stage_key: str, stage_dir: Path, baseline: pd.DataFrame) -> Dict[str, Any]:
    label = STAGE_LABELS[stage_key]
    row = _baseline_row(baseline, label)
    result: Dict[str, Any] = {}
    if row:
        for key in ["expected_assembled_units", "expected_final_backlog_units", "cvar_value", "expected_assembly_shortfall_units"]:
            value = _to_float(row.get(key))
            if value is not None:
                result[key] = value
    if stage_key == "stage3":
        plan = _safe_read_csv(stage_dir / "assembly_plan.csv")
        if not plan.empty:
            result.setdefault("expected_assembled_units", _sum_numeric(plan, "assembled_units"))
            result.setdefault("expected_final_backlog_units", _last_numeric(plan, "backlog_units"))
    return result


def _route_metrics(stage_key: str, stage_dir: Path) -> Dict[str, Any]:
    if stage_key == "stage3":
        path = stage_dir / "selected_component_routes.csv"
    else:
        path = stage_dir / "scenario_selected_component_routes.csv"
    table = _safe_read_csv(path)
    if table.empty:
        return {"selected_route_count": 0, "route_mix": {}, "route_source": str(path)}
    route_id = table.get("route_id", pd.Series(dtype=object)).dropna().astype(str)
    mix = route_id.value_counts().sort_index().to_dict()
    return {
        "selected_route_count": int(len(table)),
        "route_mix": {str(key): int(value) for key, value in mix.items()},
        "route_source": str(path),
    }


def _stage6_assembly_metrics(stage_dir: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    candidates = _safe_read_csv(stage_dir / "selected_assembly_candidates.csv")
    if not candidates.empty:
        selected = _selected_rows(candidates)
        source_types = selected.get("candidate_source_type", pd.Series(dtype=object)).astype(str).str.lower()
        old_flags = pd.to_numeric(selected.get("old_candidate_flag", pd.Series(dtype=object)), errors="coerce").fillna(0)
        new_mask = source_types.str.contains("new", na=False) | source_types.str.contains("replacement", na=False)
        new_count = int(new_mask.sum())
        old_count = int((old_flags > 0.5).sum())
        total = int(len(selected))
        result.update(
            {
                "selected_candidate_count": total,
                "selected_candidate_new_share": _safe_ratio(new_count, total),
                "selected_candidate_old_share": _safe_ratio(old_count, total),
                "candidate_source": str(stage_dir / "selected_assembly_candidates.csv"),
            }
        )
    pairs = _safe_read_csv(stage_dir / "selected_assembly_pairs.csv")
    if not pairs.empty:
        selected_pairs = _selected_rows(pairs)
        hard = selected_pairs.get("compatibility_status", pd.Series(dtype=object)).astype(str).str.lower().eq("hard_feasible").sum()
        total_pairs = int(len(selected_pairs))
        result.update(
            {
                "selected_pair_count": total_pairs,
                "hard_pair_share": _safe_ratio(hard, total_pairs),
                "pair_source": str(stage_dir / "selected_assembly_pairs.csv"),
            }
        )
    features = _safe_read_csv(stage_dir / "feature_assembly_plan.csv")
    if not features.empty:
        coverage = pd.to_numeric(features.get("coverage_rate", pd.Series(dtype=object)), errors="coerce").dropna()
        shortfall = pd.to_numeric(features.get("assembly_shortfall_units", pd.Series(dtype=object)), errors="coerce").fillna(0)
        result.update(
            {
                "mean_feature_coverage_rate": float(coverage.mean()) if not coverage.empty else None,
                "shortfall_feature_rows": int((shortfall > 1e-8).sum()),
                "total_feature_shortfall": float(shortfall.sum()),
                "feature_source": str(stage_dir / "feature_assembly_plan.csv"),
            }
        )
    chance = _safe_read_csv(stage_dir / "chance_constraint_report.csv")
    if not chance.empty:
        candidate_rows = _sum_numeric(chance, "candidate_rows")
        excluded_rows = _sum_numeric(chance, "excluded_rows")
        chance_pass = _sum_numeric(chance, "chance_pass_rows")
        result.update(
            {
                "chance_candidate_rows": candidate_rows,
                "chance_excluded_rows": excluded_rows,
                "chance_pass_rows": chance_pass,
                "chance_excluded_share": _safe_ratio(excluded_rows, candidate_rows),
                "chance_source": str(stage_dir / "chance_constraint_report.csv"),
            }
        )
    return result


def _baseline_row(baseline: pd.DataFrame, stage_label: str) -> Dict[str, Any]:
    if baseline.empty or "model_stage" not in baseline.columns:
        return {}
    matches = baseline[baseline["model_stage"].astype(str) == stage_label]
    if matches.empty:
        return {}
    return matches.iloc[0].to_dict()


def _selected_rows(table: pd.DataFrame) -> pd.DataFrame:
    if "selected_value" not in table.columns:
        return table
    selected = pd.to_numeric(table["selected_value"], errors="coerce").fillna(0) > 0.5
    return table[selected].copy()


def _all_metrics_flat(frame: pd.DataFrame) -> bool:
    columns = [
        "objective_value",
        "expected_final_backlog_units",
        "cvar_value",
        "expected_assembly_shortfall_units",
    ]
    flags = []
    for column in columns:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").dropna().round(8)
            if not values.empty:
                flags.append(values.nunique() <= 1)
    return bool(flags and all(flags))


def _stage_sources(metric: Dict[str, Any], expected: List[str]) -> str:
    sources = []
    for key in [
        "first_stage_source",
        "route_source",
        "solution_summary_source",
        "candidate_source",
        "pair_source",
        "feature_source",
        "chance_source",
    ]:
        value = str(metric.get(key, ""))
        if value and value != "None" and _source_exists(value):
            sources.append(value)
    if not sources and expected:
        sources.extend(expected)
    return ";".join(_dedupe(sources))


def _row(
    rows: List[Dict[str, Any]],
    check_id: str,
    dimension: str,
    model_stage: str,
    status: str,
    decision: str,
    evidence: str,
    metric_name: str,
    metric_value: str,
    benchmark_value: str,
    threshold: str,
    flag: str,
    message: str,
    sources: str,
    writing: str,
) -> None:
    rows.append(
        {
            "check_id": check_id,
            "audit_dimension": dimension,
            "model_stage": model_stage,
            "status": status,
            "operational_reality_decision": decision,
            "evidence_decision": evidence,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "benchmark_value": benchmark_value,
            "threshold": threshold,
            "flag": flag,
            "message": message,
            "source_files": sources,
            "recommended_writing": writing,
        }
    )


def _adjustment(adjustment_type: str, target: str, evidence: str, writing: str) -> Dict[str, Any]:
    return {
        "adjustment_type": adjustment_type,
        "target_result": target,
        "evidence": evidence,
        "interpretation_adjustment": writing,
    }


def _format_mix(mix: Any) -> str:
    if isinstance(mix, dict) and mix:
        return "; ".join(f"{key}:{value}" for key, value in sorted(mix.items()))
    if isinstance(mix, str) and mix:
        return mix
    return "none"


def _sum_numeric(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.sum())


def _last_numeric(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[-1])


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    num = _to_float(numerator)
    den = _to_float(denominator)
    if num is None or den is None or abs(den) < 1e-12:
        return None
    return float(num / den)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return "NA"
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.4f}"


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except (pd.errors.EmptyDataError, UnicodeDecodeError):
        return pd.DataFrame()


def _safe_read_json(path: Path) -> Dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _source_exists(value: str) -> bool:
    try:
        return Path(value).exists()
    except OSError:
        return False


def _dedupe(values: List[str]) -> List[str]:
    seen = set()
    output = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output
