"""Solver interface and solution extraction for Stage 4."""

from __future__ import annotations

import time
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from .baseline import evaluate_baseline
from .checks import model_size_counts, run_solution_checks, summarize_checks
from .config import Stage4Config
from .structures import Stage4Instance, Stage4ModelData, Stage4Solution


def solve_model(instance: Stage4Instance, model_data: Stage4ModelData, config: Stage4Config, tables: Dict[str, pd.DataFrame]) -> Stage4Solution:
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
    objective_breakdown = _objective_breakdown(model_data, x)
    baseline = evaluate_baseline(instance, config, tables)
    summary_metrics = _summary_metrics(instance, model_data, x, first_stage_decisions, selected_routes, assembly_plan)
    checks = run_solution_checks(
        instance,
        model_data,
        x,
        first_stage_decisions,
        selected_routes,
        assembly_plan,
        inventory_trajectory,
        capacity_utilization,
        config.processing_window_periods,
    )
    summary_metrics["solution_check_summary"] = summarize_checks(checks)

    return Stage4Solution(
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
    )


def _first_stage_decisions(instance: Stage4Instance, model_data: Stage4ModelData, x: np.ndarray) -> pd.DataFrame:
    core_map = instance.component_summary.groupby("core_id")["component_instance_id"].count().to_dict()
    accept = model_data.variable_groups["accept_core"]
    pre_procure = model_data.variable_groups["pre_procure"]
    rows = []
    for core in instance.core_summary.itertuples(index=False):
        rows.append(
            {
                "core_id": core.core_id,
                "accept_core": float(x[accept[core.core_id]]),
                "component_count": int(core_map.get(core.core_id, 0)),
                "fixed_accept_cost_rmb": float(core.fixed_accept_cost_rmb),
            }
        )
    for key, index in pre_procure.items():
        component_type, period = key.split("|")
        rows.append(
            {
                "core_id": None,
                "component_type": component_type,
                "period_id": period,
                "pre_procure_units": float(x[index]),
            }
        )
    return pd.DataFrame(rows)


def _selected_component_routes(instance: Stage4Instance, model_data: Stage4ModelData, x: np.ndarray) -> pd.DataFrame:
    route_table = instance.component_route_period_scenario_table.set_index(["scenario_id", "component_instance_id", "route_id", "period_id"])
    rows = []
    for key, index in model_data.variable_groups["x"].items():
        if x[index] <= 0.5:
            continue
        scenario_id, component_id, route_id, period = key.split("|")
        row = route_table.loc[(scenario_id, component_id, route_id, period)]
        rows.append(row.to_dict() | {"scenario_id": scenario_id, "component_instance_id": component_id, "route_id": route_id, "period_id": period})
    return pd.DataFrame(rows).sort_values(["scenario_id", "period_id", "component_instance_id", "route_id"]).reset_index(drop=True) if rows else pd.DataFrame()


def _scenario_assembly_plan(instance: Stage4Instance, model_data: Stage4ModelData, x: np.ndarray) -> pd.DataFrame:
    rows = []
    demand = instance.scenario_demand.set_index(["scenario_id", "period_id"])["demand_units"].to_dict()
    for scenario_id in instance.scenario_ids:
        for period in instance.periods:
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "period_id": period,
                    "saa_probability": float(instance.scenario_sample.set_index("scenario_id").loc[scenario_id, "saa_probability"]),
                    "demand_units": int(demand.get((scenario_id, period), 0)),
                    "assembled_units": float(x[model_data.variable_groups["assemble"][f"{scenario_id}|{period}"]]),
                    "backlog_units": float(x[model_data.variable_groups["backlog"][f"{scenario_id}|{period}"]]),
                }
            )
    return pd.DataFrame(rows)


def _scenario_inventory_trajectory(
    instance: Stage4Instance,
    model_data: Stage4ModelData,
    x: np.ndarray,
    selected_routes: pd.DataFrame,
    assembly_plan: pd.DataFrame,
) -> pd.DataFrame:
    bom = instance.bom_requirements.set_index("component_type")["required_quantity"].to_dict()
    rows = []
    for scenario_id in instance.scenario_ids:
        scenario_routes = selected_routes[selected_routes["scenario_id"] == scenario_id] if not selected_routes.empty else pd.DataFrame()
        for component_type in instance.component_types:
            produced_by_period = (
                scenario_routes[(scenario_routes["component_type"] == component_type) & (scenario_routes["productive_output"] == 1)]
                .groupby("period_id")
                .size()
                .to_dict()
                if not scenario_routes.empty
                else {}
            )
            for period in instance.periods:
                assembled = float(assembly_plan[(assembly_plan["scenario_id"] == scenario_id) & (assembly_plan["period_id"] == period)]["assembled_units"].iloc[0])
                procured = float(x[model_data.variable_groups["recourse_procure"][f"{scenario_id}|{component_type}|{period}"]])
                pre_procured = float(x[model_data.variable_groups["pre_procure"][f"{component_type}|{period}"]])
                ending = float(x[model_data.variable_groups["inventory"][f"{scenario_id}|{component_type}|{period}"]])
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "component_type": component_type,
                        "period_id": period,
                        "produced_components": float(produced_by_period.get(period, 0.0)),
                        "pre_procured_components": pre_procured,
                        "recourse_procured_components": procured,
                        "assembly_consumption": assembled * float(bom[component_type]),
                        "ending_inventory": ending,
                    }
                )
    return pd.DataFrame(rows)


def _scenario_capacity_utilization(instance: Stage4Instance, model_data: Stage4ModelData, x: np.ndarray) -> pd.DataFrame:
    route_table = instance.component_route_period_scenario_table
    selected_value = {tuple(key.split("|")): x[index] for key, index in model_data.variable_groups["x"].items()}
    capacity = instance.scenario_capacity_table.set_index(["scenario_id", "resource_type", "period_id"])["available_regular_hours"].to_dict()
    rows = []
    for key, overtime_idx in model_data.variable_groups["overtime"].items():
        scenario_id, resource, period = key.split("|")
        resource_col = f"resource_h__{resource}"
        used = 0.0
        if resource_col in route_table.columns:
            period_rows = route_table[(route_table["scenario_id"] == scenario_id) & (route_table["period_id"] == period)]
            for assignment in period_rows.itertuples(index=False):
                used += float(getattr(assignment, resource_col, 0.0)) * selected_value.get((assignment.scenario_id, assignment.component_instance_id, assignment.route_id, assignment.period_id), 0.0)
        available = float(capacity.get((scenario_id, resource, period), 0.0))
        overtime = float(x[overtime_idx])
        rows.append(
            {
                "scenario_id": scenario_id,
                "resource_type": resource,
                "period_id": period,
                "used_hours": used,
                "available_regular_hours": available,
                "overtime_hours": overtime,
                "utilization_rate_regular": used / available if available > 0 else None,
            }
        )
    return pd.DataFrame(rows).sort_values(["scenario_id", "period_id", "resource_type"]).reset_index(drop=True)


def _objective_breakdown(model_data: Stage4ModelData, x: np.ndarray) -> Dict[str, float]:
    breakdown = {
        "accept_fixed_cost_rmb": 0.0,
        "pre_procure_cost_rmb": 0.0,
        "route_economic_cost_rmb": 0.0,
        "environmental_cost_equiv": 0.0,
        "quality_penalty_equiv": 0.0,
        "reliability_penalty_equiv": 0.0,
        "warranty_risk_cost_rmb": 0.0,
        "recourse_procurement_cost_rmb": 0.0,
        "inventory_holding_cost_rmb": 0.0,
        "backlog_penalty_rmb": 0.0,
        "overtime_penalty_rmb": 0.0,
    }
    name_to_value = dict(zip(model_data.variable_names, x))
    objective_by_name = dict(zip(model_data.variable_names, model_data.objective))
    for name, value in name_to_value.items():
        if value <= 1e-8:
            continue
        contribution = float(objective_by_name[name]) * float(value)
        if name.startswith("accept_core["):
            breakdown["accept_fixed_cost_rmb"] += contribution
        elif name.startswith("pre_procure["):
            breakdown["pre_procure_cost_rmb"] += contribution
        elif name.startswith("recourse_procure["):
            breakdown["recourse_procurement_cost_rmb"] += contribution
        elif name.startswith("inventory["):
            breakdown["inventory_holding_cost_rmb"] += contribution
        elif name.startswith("backlog["):
            breakdown["backlog_penalty_rmb"] += contribution
        elif name.startswith("overtime["):
            breakdown["overtime_penalty_rmb"] += contribution
    if not model_data.objective_terms.empty:
        for term in model_data.objective_terms.itertuples(index=False):
            value = name_to_value.get(term.variable_name, 0.0)
            if value <= 1e-8:
                continue
            breakdown["route_economic_cost_rmb"] += float(term.economic_cost_rmb) * value * float(term.saa_probability)
            breakdown["environmental_cost_equiv"] += float(term.environmental_cost_equiv) * value * float(term.saa_probability)
            breakdown["quality_penalty_equiv"] += float(term.quality_penalty_equiv) * value * float(term.saa_probability)
            breakdown["reliability_penalty_equiv"] += float(term.reliability_penalty_equiv) * value * float(term.saa_probability)
            breakdown["warranty_risk_cost_rmb"] += float(term.warranty_risk_cost_rmb) * value * float(term.saa_probability)
    breakdown["total_reconstructed"] = float(sum(breakdown.values()))
    return breakdown


def _summary_metrics(
    instance: Stage4Instance,
    model_data: Stage4ModelData,
    x: np.ndarray,
    first_stage_decisions: pd.DataFrame,
    selected_routes: pd.DataFrame,
    assembly_plan: pd.DataFrame,
) -> Dict[str, object]:
    accepted = sum(1 for index in model_data.variable_groups["accept_core"].values() if x[index] > 0.5)
    counts = model_size_counts(model_data)
    route_mix = selected_routes["route_id"].value_counts().sort_index().to_dict() if not selected_routes.empty else {}
    scenario_summary = (
        assembly_plan.groupby("scenario_id")
        .agg(
            demand_units=("demand_units", "sum"),
            assembled_units=("assembled_units", "sum"),
            backlog_units=("backlog_units", "last"),
        )
        .reset_index()
    )
    return {
        "candidate_core_count": instance.candidate_core_count,
        "accepted_core_count": int(accepted),
        "acceptance_rate": accepted / instance.candidate_core_count if instance.candidate_core_count else None,
        "component_instance_count": instance.component_instance_count,
        "scenario_count": len(instance.scenario_ids),
        "selected_component_route_count": int(len(selected_routes)),
        "expected_demand_units": instance.demand_units_expected,
        "expected_assembled_units": float((assembly_plan["assembled_units"] * assembly_plan["saa_probability"]).sum()),
        "expected_final_backlog_units": float(
            assembly_plan.sort_values(["scenario_id", "period_id"]).groupby("scenario_id")["backlog_units"].last().mul(
                assembly_plan.groupby("scenario_id")["saa_probability"].first()
            ).sum()
        ),
        "route_mix": route_mix,
        "worst_scenario_backlog_units": float(scenario_summary["backlog_units"].max()) if not scenario_summary.empty else None,
        "worst_scenario_assembled_units": float(scenario_summary["assembled_units"].min()) if not scenario_summary.empty else None,
        **counts,
    }
