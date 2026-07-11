"""Solver interface and solution extraction for Stage 6."""

from __future__ import annotations

import time
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from stage4_stochastic.solver import (
    _first_stage_decisions,
    _objective_breakdown,
    _scenario_assembly_plan,
    _scenario_capacity_utilization,
    _scenario_inventory_trajectory,
    _selected_component_routes,
    _summary_metrics,
)
from stage5_risk_averse.solver import _risk_objective_breakdown, _scenario_risk_metrics

from .baseline import evaluate_baseline
from .checks import run_solution_checks, summarize_checks
from .config import Stage6Config
from .io_utils import read_stage5_solution_summary
from .structures import Stage6Instance, Stage6ModelData, Stage6Solution


def solve_model(
    instance: Stage6Instance,
    model_data: Stage6ModelData,
    config: Stage6Config,
    tables: Dict[str, pd.DataFrame],
) -> Stage6Solution:
    start = time.perf_counter()
    result = milp(
        c=model_data.objective,
        integrality=model_data.integrality,
        bounds=Bounds(model_data.lower_bounds, model_data.upper_bounds),
        constraints=LinearConstraint(model_data.constraint_matrix, model_data.constraint_lhs, model_data.constraint_rhs),
        options={"time_limit": config.time_limit_seconds, "mip_rel_gap": config.mip_rel_gap},
    )
    solve_seconds = time.perf_counter() - start

    x = np.asarray(result.x if result.x is not None else np.zeros(len(model_data.variable_names)), dtype=float)
    variables = pd.DataFrame({"variable_name": model_data.variable_names, "value": x, "objective_coefficient": model_data.objective})
    first_stage_decisions = _first_stage_decisions(instance, model_data, x)
    selected_routes = _selected_component_routes(instance, model_data, x)
    assembly_plan = _scenario_assembly_plan(instance, model_data, x)
    inventory_trajectory = _scenario_inventory_trajectory(instance, model_data, x, selected_routes, assembly_plan)
    capacity_utilization = _scenario_capacity_utilization(instance, model_data, x)
    scenario_risk_metrics, cvar_summary = _scenario_risk_metrics(instance, model_data, x, config)
    selected_candidates = _selected_assembly_candidates(instance, model_data, x)
    selected_pairs = _selected_assembly_pairs(instance, model_data, x)
    feature_plan = _feature_assembly_plan(instance, model_data, x, assembly_plan)
    dimension_report = _dimension_chain_report(selected_pairs, feature_plan)
    quality_report = _assembly_quality_loss_report(selected_pairs, feature_plan)
    assembly_risk_metrics = _scenario_assembly_risk_metrics(model_data, x, selected_pairs, feature_plan)
    objective_breakdown = _objective_breakdown(model_data, x)
    objective_breakdown.update(_assembly_objective_breakdown(model_data, x))
    objective_breakdown.update(_risk_objective_breakdown(cvar_summary, config))
    objective_breakdown["total_with_selective_assembly_and_cvar"] = float(result.fun) if result.fun is not None else None
    baseline = evaluate_baseline(instance, config, tables)
    stage5_comparison = read_stage5_solution_summary(config.stage5_results_dir)
    summary_metrics = _summary_metrics(instance, model_data, x, first_stage_decisions, selected_routes, assembly_plan)
    summary_metrics.update(
        {
            "chance_pass_rate": instance.to_summary_dict().get("chance_pass_rate"),
            "assembly_requirement_count": int(len(instance.assembly_requirements)),
            "assembly_candidate_pool_size": int(len(instance.assembly_candidate_pool)),
            "assembly_pair_pool_size": int(len(instance.assembly_pair_pool)),
            "selected_assembly_candidate_count": int(len(selected_candidates)),
            "selected_assembly_pair_count": int(len(selected_pairs)),
            "expected_feature_assembled_units": _expected_feature_value(feature_plan, "feature_assembled_units"),
            "expected_assembly_shortfall_units": _expected_feature_value(feature_plan, "assembly_shortfall_units"),
            "cvar_value": cvar_summary.get("cvar_value"),
            "var_eta": cvar_summary.get("eta"),
            "worst_scenario_loss": cvar_summary.get("worst_scenario_loss"),
        }
    )
    checks = run_solution_checks(
        instance,
        model_data,
        x,
        first_stage_decisions,
        selected_routes,
        assembly_plan,
        inventory_trajectory,
        capacity_utilization,
        scenario_risk_metrics,
        selected_candidates,
        selected_pairs,
        feature_plan,
        processing_window_periods=config.processing_window_periods,
    )
    summary_metrics["solution_check_summary"] = summarize_checks(checks)

    return Stage6Solution(
        success=bool(result.success),
        status=int(result.status),
        status_message=str(result.message),
        objective_value=float(result.fun) if result.fun is not None else None,
        mip_gap=float(getattr(result, "mip_gap", np.nan)) if getattr(result, "mip_gap", None) is not None else None,
        solve_seconds=solve_seconds,
        variables=variables,
        first_stage_decisions=first_stage_decisions,
        scenario_selected_component_routes=selected_routes,
        scenario_assembly_plan=assembly_plan,
        scenario_inventory_trajectory=inventory_trajectory,
        scenario_capacity_utilization=capacity_utilization,
        objective_breakdown=objective_breakdown,
        summary_metrics=summary_metrics,
        baseline_comparison=baseline,
        solution_checks=checks,
        scenario_risk_metrics=scenario_risk_metrics,
        cvar_summary=cvar_summary,
        chance_constraint_report=instance.chance_constraint_report,
        stage4_comparison=baseline.get("stage4_comparison", {}),
        selected_assembly_candidates=selected_candidates,
        selected_assembly_pairs=selected_pairs,
        feature_assembly_plan=feature_plan,
        dimension_chain_report=dimension_report,
        assembly_quality_loss_report=quality_report,
        scenario_assembly_risk_metrics=assembly_risk_metrics,
        stage5_comparison=stage5_comparison,
        assembly_baseline_comparison=baseline,
    )


def _selected_assembly_candidates(instance: Stage6Instance, model_data: Stage6ModelData, x: np.ndarray) -> pd.DataFrame:
    pool = instance.assembly_candidate_pool.set_index("assembly_candidate_id")
    rows = []
    for key, index in model_data.variable_groups["select_candidate"].items():
        if x[index] <= 0.5:
            continue
        scenario_id, requirement_id, candidate_id = key.split("|", 2)
        row = pool.loc[candidate_id].to_dict()
        rows.append(row | {"scenario_id": scenario_id, "assembly_requirement_id": requirement_id, "assembly_candidate_id": candidate_id, "selected_value": float(x[index])})
    return pd.DataFrame(rows).sort_values(["scenario_id", "assembly_requirement_id", "assembly_candidate_id"]).reset_index(drop=True) if rows else pd.DataFrame()


def _selected_assembly_pairs(instance: Stage6Instance, model_data: Stage6ModelData, x: np.ndarray) -> pd.DataFrame:
    pool = instance.assembly_pair_pool.set_index("compatibility_id")
    rows = []
    for key, index in model_data.variable_groups["select_pair"].items():
        if x[index] <= 0.5:
            continue
        scenario_id, requirement_id, compatibility_id = key.split("|", 2)
        row = pool.loc[compatibility_id].to_dict()
        rows.append(row | {"scenario_id": scenario_id, "assembly_requirement_id": requirement_id, "compatibility_id": compatibility_id, "selected_value": float(x[index])})
    return pd.DataFrame(rows).sort_values(["scenario_id", "assembly_requirement_id", "compatibility_id"]).reset_index(drop=True) if rows else pd.DataFrame()


def _feature_assembly_plan(
    instance: Stage6Instance,
    model_data: Stage6ModelData,
    x: np.ndarray,
    assembly_plan: pd.DataFrame,
) -> pd.DataFrame:
    total_assemble = assembly_plan.groupby("scenario_id")["assembled_units"].sum().to_dict()
    probability = instance.scenario_sample.set_index("scenario_id")["saa_probability"].to_dict()
    rows = []
    for scenario_id in instance.scenario_ids:
        for req in instance.assembly_requirements.itertuples(index=False):
            key = f"{scenario_id}|{req.assembly_requirement_id}"
            feature_value = float(x[model_data.variable_groups["feature_assemble"][key]])
            shortfall_value = float(x[model_data.variable_groups["assembly_shortfall"][key]])
            scenario_total = float(total_assemble.get(scenario_id, 0.0))
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "saa_probability": float(probability[scenario_id]),
                    "assembly_requirement_id": req.assembly_requirement_id,
                    "assembly_feature": req.assembly_feature,
                    "required_component_types": req.required_component_types,
                    "required_component_count": float(req.required_component_count),
                    "pair_count_divisor": float(req.pair_count_divisor),
                    "scenario_total_assembled_units": scenario_total,
                    "feature_assembled_units": feature_value,
                    "assembly_shortfall_units": shortfall_value,
                    "coverage_residual": feature_value + shortfall_value - scenario_total,
                    "coverage_rate": feature_value / scenario_total if scenario_total > 1e-9 else None,
                }
            )
    return pd.DataFrame(rows)


def _dimension_chain_report(selected_pairs: pd.DataFrame, feature_plan: pd.DataFrame) -> pd.DataFrame:
    if selected_pairs.empty:
        return pd.DataFrame()
    grouped = (
        selected_pairs.assign(abs_dimension_error=lambda df: pd.to_numeric(df["pair_dimension_error_mm"], errors="coerce").abs())
        .groupby(["scenario_id", "assembly_requirement_id"], dropna=False)
        .agg(
            selected_pair_count=("compatibility_id", "count"),
            max_abs_dimension_error_mm=("abs_dimension_error", "max"),
            mean_abs_dimension_error_mm=("abs_dimension_error", "mean"),
            max_allowed_dimension_error_mm=("max_dimension_chain_error_mm", "max"),
            min_feature_pair_reliability=("feature_pair_reliability", "min"),
            mean_compatibility_score=("compatibility_score", "mean"),
            soft_pair_count=("soft_pair_flag", "sum"),
        )
        .reset_index()
    )
    return grouped.merge(
        feature_plan[["scenario_id", "assembly_requirement_id", "feature_assembled_units", "assembly_shortfall_units"]],
        on=["scenario_id", "assembly_requirement_id"],
        how="left",
    )


def _assembly_quality_loss_report(selected_pairs: pd.DataFrame, feature_plan: pd.DataFrame) -> pd.DataFrame:
    if selected_pairs.empty:
        return pd.DataFrame()
    grouped = (
        selected_pairs.groupby(["scenario_id", "assembly_requirement_id"], dropna=False)
        .agg(
            selected_pair_count=("compatibility_id", "count"),
            total_pair_quality_loss=("pair_quality_loss", "sum"),
            mean_pair_quality_loss=("pair_quality_loss", "mean"),
            max_pair_quality_loss=("pair_quality_loss", "max"),
            max_allowed_quality_loss=("max_quality_loss", "max"),
            mean_life_gap_h=("pairwise_life_gap_h", "mean"),
            max_life_gap_h=("pairwise_life_gap_h", "max"),
            max_allowed_life_gap_h=("max_pairwise_life_gap_h", "max"),
        )
        .reset_index()
    )
    return grouped.merge(
        feature_plan[["scenario_id", "assembly_requirement_id", "feature_assembled_units", "assembly_shortfall_units"]],
        on=["scenario_id", "assembly_requirement_id"],
        how="left",
    )


def _scenario_assembly_risk_metrics(
    model_data: Stage6ModelData,
    x: np.ndarray,
    selected_pairs: pd.DataFrame,
    feature_plan: pd.DataFrame,
) -> pd.DataFrame:
    name_to_value = dict(zip(model_data.variable_names, x))
    rows = []
    terms = model_data.assembly_loss_terms.copy()
    if terms.empty:
        return pd.DataFrame()
    for scenario_id, group in terms.groupby("scenario_id", dropna=False):
        selected_loss = 0.0
        pair_loss = 0.0
        shortfall_loss = 0.0
        tail_risk_loss = 0.0
        for term in group.itertuples(index=False):
            value = float(name_to_value.get(term.variable_name, 0.0))
            contribution = value * float(term.loss_coefficient_unweighted)
            selected_loss += contribution
            if term.loss_source == "assembly_pair":
                pair_loss += contribution
                tail_risk_loss += value * float(getattr(term, "assembly_tail_risk_loss_rmb", 0.0))
            elif term.loss_source == "assembly_shortfall":
                shortfall_loss += contribution
        scenario_pairs = selected_pairs[selected_pairs["scenario_id"] == scenario_id] if not selected_pairs.empty else pd.DataFrame()
        scenario_features = feature_plan[feature_plan["scenario_id"] == scenario_id] if not feature_plan.empty else pd.DataFrame()
        rows.append(
            {
                "scenario_id": scenario_id,
                "selected_assembly_pair_count": int(len(scenario_pairs)),
                "feature_assembled_units_total": float(scenario_features["feature_assembled_units"].sum()) if not scenario_features.empty else 0.0,
                "assembly_shortfall_units_total": float(scenario_features["assembly_shortfall_units"].sum()) if not scenario_features.empty else 0.0,
                "assembly_loss_unweighted": selected_loss,
                "assembly_pair_loss_unweighted": pair_loss,
                "assembly_shortfall_loss_unweighted": shortfall_loss,
                "assembly_tail_risk_loss_unweighted": tail_risk_loss,
            }
        )
    return pd.DataFrame(rows)


def _assembly_objective_breakdown(model_data: Stage6ModelData, x: np.ndarray) -> Dict[str, float]:
    name_to_value = dict(zip(model_data.variable_names, x))
    total = 0.0
    pair = 0.0
    shortfall = 0.0
    for term in model_data.assembly_loss_terms.itertuples(index=False):
        value = float(name_to_value.get(term.variable_name, 0.0))
        if value <= 1e-8:
            continue
        probability = _scenario_probability_from_terms(model_data, str(term.scenario_id))
        contribution = probability * float(term.loss_coefficient_unweighted) * value
        total += contribution
        if term.loss_source == "assembly_pair":
            pair += contribution
        elif term.loss_source == "assembly_shortfall":
            shortfall += contribution
    return {
        "selective_assembly_expected_cost_rmb": total,
        "assembly_pair_expected_penalty_rmb": pair,
        "assembly_shortfall_expected_penalty_rmb": shortfall,
    }


def _scenario_probability_from_terms(model_data: Stage6ModelData, scenario_id: str) -> float:
    terms = model_data.objective_terms
    if terms.empty:
        return 0.0
    values = terms[terms["scenario_id"] == scenario_id]["saa_probability"]
    return float(values.iloc[0]) if not values.empty else 0.0


def _expected_feature_value(feature_plan: pd.DataFrame, column: str) -> float:
    if feature_plan.empty:
        return 0.0
    by_scenario = feature_plan.groupby("scenario_id").agg(value=(column, "sum"), probability=("saa_probability", "first")).reset_index()
    return float((by_scenario["value"] * by_scenario["probability"]).sum())
