"""Solver interface for the Stage 2 deterministic MILP."""

from __future__ import annotations

import time
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from .baseline import evaluate_baseline
from .checks import model_size_counts, run_solution_checks, summarize_checks
from .config import Stage2Config
from .structures import Stage2Instance, Stage2ModelData, Stage2Solution


def solve_model(instance: Stage2Instance, model_data: Stage2ModelData, config: Stage2Config, tables: Dict[str, pd.DataFrame]) -> Stage2Solution:
    """Solve the MILP and return a rich solution object."""

    start = time.perf_counter()
    result = milp(
        c=model_data.objective,
        integrality=model_data.integrality,
        bounds=Bounds(model_data.lower_bounds, model_data.upper_bounds),
        constraints=LinearConstraint(
            model_data.constraint_matrix,
            model_data.constraint_lhs,
            model_data.constraint_rhs,
        ),
        options={"time_limit": config.time_limit_seconds, "mip_rel_gap": config.mip_rel_gap},
    )
    solve_seconds = time.perf_counter() - start

    x = np.asarray(result.x if result.x is not None else np.zeros(len(model_data.variable_names)), dtype=float)
    variables = pd.DataFrame({"variable_name": model_data.variable_names, "value": x, "objective_coefficient": model_data.objective})
    selected_routes = _selected_routes(instance, model_data, x)
    objective_breakdown = _objective_breakdown(model_data, x)
    capacity_utilization = _capacity_utilization(instance, selected_routes, model_data, x)
    summary_metrics = _summary_metrics(instance, selected_routes, model_data, x, result)
    solution_checks = run_solution_checks(instance, model_data, x, selected_routes, capacity_utilization)
    summary_metrics["solution_check_summary"] = summarize_checks(solution_checks)
    baseline = evaluate_baseline(instance, config, tables)
    if baseline.get("objective_value") is not None and result.fun is not None and result.fun > 0:
        baseline["gap_vs_milp_pct"] = 100.0 * (float(baseline["objective_value"]) - float(result.fun)) / float(result.fun)

    return Stage2Solution(
        success=bool(result.success),
        status=int(result.status),
        status_message=str(result.message),
        objective_value=float(result.fun) if result.fun is not None else None,
        mip_gap=float(getattr(result, "mip_gap", np.nan)) if getattr(result, "mip_gap", None) is not None else None,
        solve_seconds=solve_seconds,
        variables=variables,
        selected_routes=selected_routes,
        objective_breakdown=objective_breakdown,
        summary_metrics=summary_metrics,
        capacity_utilization=capacity_utilization,
        baseline_comparison=baseline,
        solution_checks=solution_checks,
    )


def _selected_routes(instance: Stage2Instance, model_data: Stage2ModelData, x: np.ndarray) -> pd.DataFrame:
    route_indices = {
        tuple(key.split("|")): index
        for key, index in model_data.variable_groups["route"].items()
    }
    rows = []
    core_summary = instance.core_summary.set_index("core_id")
    for assignment in model_data.route_assignments.itertuples(index=False):
        value = x[route_indices[(assignment.core_id, assignment.route_class)]]
        if value > 0.5:
            fixed_cost = float(core_summary.loc[assignment.core_id, "fixed_accept_cost_rmb"])
            rows.append(
                {
                    "core_id": assignment.core_id,
                    "route_class": assignment.route_class,
                    "fixed_accept_cost_rmb": fixed_cost,
                    "economic_cost_rmb": float(assignment.economic_cost_rmb),
                    "environmental_score": float(assignment.environmental_score),
                    "expected_output_quality": float(assignment.expected_output_quality),
                    "expected_residual_life_h": float(assignment.expected_residual_life_h),
                    "risk_penalty": float(assignment.risk_penalty),
                    "route_success_proxy": float(assignment.route_success_proxy),
                }
            )
    return pd.DataFrame(rows)


def _objective_breakdown(model_data: Stage2ModelData, x: np.ndarray) -> Dict[str, float]:
    breakdown = {
        "accept_fixed_cost_rmb": 0.0,
        "route_economic_cost_rmb": 0.0,
        "environmental_cost_equiv": 0.0,
        "quality_penalty_equiv": 0.0,
        "reliability_penalty_equiv": 0.0,
        "procurement_cost_rmb": 0.0,
        "shortage_penalty_rmb": 0.0,
        "overtime_penalty_rmb": 0.0,
    }
    name_to_value = dict(zip(model_data.variable_names, x))
    for name, value in name_to_value.items():
        if value <= 1e-8:
            continue
        coef = float(model_data.objective[model_data.variable_names.index(name)])
        if name.startswith("accept["):
            breakdown["accept_fixed_cost_rmb"] += coef * value
        elif name == "procurement_units":
            breakdown["procurement_cost_rmb"] += coef * value
        elif name == "shortage_units":
            breakdown["shortage_penalty_rmb"] += coef * value
        elif name.startswith("overtime["):
            breakdown["overtime_penalty_rmb"] += coef * value

    if not model_data.objective_terms.empty:
        for term in model_data.objective_terms.itertuples(index=False):
            value = name_to_value.get(term.variable_name, 0.0)
            if value <= 1e-8:
                continue
            breakdown["route_economic_cost_rmb"] += float(term.economic_cost_rmb) * value
            breakdown["environmental_cost_equiv"] += float(term.environmental_cost_equiv) * value
            breakdown["quality_penalty_equiv"] += float(term.quality_penalty_equiv) * value
            breakdown["reliability_penalty_equiv"] += float(term.reliability_penalty_equiv) * value
    breakdown["total_reconstructed"] = float(sum(breakdown.values()))
    return breakdown


def _capacity_utilization(
    instance: Stage2Instance,
    selected_routes: pd.DataFrame,
    model_data: Stage2ModelData,
    x: np.ndarray,
) -> pd.DataFrame:
    rows = []
    route_table = model_data.route_assignments
    route_values = {
        tuple(key.split("|")): x[index]
        for key, index in model_data.variable_groups["route"].items()
    }
    procurement_units = x[model_data.variable_groups["procurement"]]
    for resource, overtime_idx in model_data.variable_groups["overtime"].items():
        used = 0.0
        resource_col = f"resource_h__{resource}"
        if resource_col in route_table.columns:
            for assignment in route_table.itertuples(index=False):
                used += float(getattr(assignment, resource_col)) * route_values.get((assignment.core_id, assignment.route_class), 0.0)
        used += procurement_units * _procurement_resource_hours(resource)
        available = float(instance.capacity_by_resource_h.get(resource, 0.0))
        overtime = float(x[overtime_idx])
        rows.append(
            {
                "resource_type": resource,
                "used_hours": used,
                "available_regular_hours": available,
                "overtime_hours": overtime,
                "utilization_rate_regular": used / available if available > 0 else None,
            }
        )
    return pd.DataFrame(rows).sort_values("resource_type").reset_index(drop=True)


def _summary_metrics(instance: Stage2Instance, selected_routes: pd.DataFrame, model_data: Stage2ModelData, x: np.ndarray, result: object) -> Dict[str, object]:
    procurement_units = float(x[model_data.variable_groups["procurement"]])
    shortage_units = float(x[model_data.variable_groups["shortage"]])
    accepted_units = int(len(selected_routes))
    productive_units = int((selected_routes["route_class"] != "scrap").sum()) if not selected_routes.empty else 0
    demand_fill = (productive_units + procurement_units) / instance.demand_units if instance.demand_units else None
    route_mix = selected_routes["route_class"].value_counts().to_dict() if not selected_routes.empty else {}
    counts = model_size_counts(model_data)
    return {
        "solver_status": int(result.status),
        "solver_success": bool(result.success),
        "candidate_core_count": instance.candidate_core_count,
        "accepted_core_count": accepted_units,
        "acceptance_rate": accepted_units / instance.candidate_core_count if instance.candidate_core_count else None,
        "productive_core_units": productive_units,
        "procurement_units": procurement_units,
        "shortage_units": shortage_units,
        "demand_units": instance.demand_units,
        "demand_fill_ratio": demand_fill,
        "route_mix": route_mix,
        "average_selected_quality": float(selected_routes["expected_output_quality"].mean()) if not selected_routes.empty else None,
        "average_selected_residual_life_h": float(selected_routes["expected_residual_life_h"].mean()) if not selected_routes.empty else None,
        "model_variable_count": counts["variable_count"],
        "model_constraint_count": counts["constraint_count"],
        "binary_variable_count": counts["binary_variable_count"],
        "general_integer_variable_count": counts["general_integer_variable_count"],
        "integer_variable_count": counts["integer_variable_count"],
    }


def _procurement_resource_hours(resource: str) -> float:
    if resource == "procurement":
        return 1.25
    if resource == "incoming_inspection":
        return 0.75
    if resource == "assembly":
        return 1.2
    if resource == "testing":
        return 1.0
    return 0.0
