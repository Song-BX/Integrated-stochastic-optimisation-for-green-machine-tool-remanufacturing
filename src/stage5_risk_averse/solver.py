"""Solver interface and solution extraction for Stage 5."""

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

from .baseline import evaluate_baseline
from .checks import model_size_counts, run_solution_checks, summarize_checks
from .config import Stage5Config
from .io_utils import read_stage4_solution_summary
from .structures import Stage5Instance, Stage5ModelData, Stage5Solution


def solve_model(
    instance: Stage5Instance,
    model_data: Stage5ModelData,
    config: Stage5Config,
    tables: Dict[str, pd.DataFrame],
) -> Stage5Solution:
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
    objective_breakdown = _objective_breakdown(model_data, x)
    objective_breakdown.update(_risk_objective_breakdown(cvar_summary, config))
    objective_breakdown["total_with_cvar"] = float(result.fun) if result.fun is not None else None
    baseline = evaluate_baseline(instance, config, tables)
    summary_metrics = _summary_metrics(instance, model_data, x, first_stage_decisions, selected_routes, assembly_plan)
    summary_metrics.update(
        {
            "chance_pass_rate": instance.to_summary_dict().get("chance_pass_rate"),
            "cvar_value": cvar_summary.get("cvar_value"),
            "var_eta": cvar_summary.get("eta"),
            "worst_scenario_loss": cvar_summary.get("worst_scenario_loss"),
        }
    )
    stage4_comparison = read_stage4_solution_summary(config.stage4_results_dir)
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
        config.processing_window_periods,
    )
    summary_metrics["solution_check_summary"] = summarize_checks(checks)

    return Stage5Solution(
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
        stage4_comparison=stage4_comparison,
    )


def _scenario_risk_metrics(
    instance: Stage5Instance,
    model_data: Stage5ModelData,
    x: np.ndarray,
    config: Stage5Config,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    name_to_value = dict(zip(model_data.variable_names, x))
    eta_index = model_data.variable_groups["eta"]["cvar95"]
    eta = float(x[eta_index])
    rows = []
    total_weighted_tail = 0.0
    scenario_loss = {}
    for scenario_id in instance.scenario_ids:
        terms = model_data.scenario_loss_terms[model_data.scenario_loss_terms["scenario_id"] == scenario_id]
        loss = 0.0
        base_route_loss = 0.0
        route_tail_loss = 0.0
        for term in terms.itertuples(index=False):
            value = float(name_to_value.get(term.variable_name, 0.0))
            loss += float(term.loss_coefficient_unweighted) * value
            base_route_loss += float(term.base_route_loss_unweighted) * value
            route_tail_loss += float(term.route_tail_loss_rmb) * value
        tail_excess = float(x[model_data.variable_groups["tail_excess"][scenario_id]])
        probability = float(instance.scenario_sample.set_index("scenario_id").loc[scenario_id, "saa_probability"])
        residual = max(0.0, loss - eta - tail_excess)
        total_weighted_tail += probability * tail_excess
        scenario_loss[scenario_id] = loss
        rows.append(
            {
                "scenario_id": scenario_id,
                "saa_probability": probability,
                "scenario_loss": loss,
                "base_route_loss": base_route_loss,
                "route_tail_loss": route_tail_loss,
                "eta": eta,
                "tail_excess": tail_excess,
                "cvar_constraint_residual": residual,
                "weighted_tail_excess": probability * tail_excess,
            }
        )
    cvar_value = eta + total_weighted_tail / max(1e-9, 1.0 - float(config.cvar_confidence))
    summary = {
        "confidence": float(config.cvar_confidence),
        "lambda": float(config.cvar_lambda),
        "eta": eta,
        "weighted_tail_excess_sum": total_weighted_tail,
        "cvar_value": cvar_value,
        "cvar_objective_contribution": float(config.cvar_lambda) * cvar_value,
        "worst_scenario_loss": max(scenario_loss.values()) if scenario_loss else 0.0,
        "worst_scenario_id": max(scenario_loss, key=scenario_loss.get) if scenario_loss else None,
    }
    return pd.DataFrame(rows), summary


def _risk_objective_breakdown(cvar_summary: Dict[str, object], config: Stage5Config) -> Dict[str, float]:
    cvar_value = float(cvar_summary.get("cvar_value", 0.0))
    return {
        "cvar_value": cvar_value,
        "cvar_objective_contribution": float(config.cvar_lambda) * cvar_value,
    }

