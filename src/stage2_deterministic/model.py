"""MILP matrix assembly for the deterministic Stage 2 base model."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix

from .config import Stage2Config
from .structures import Stage2Instance, Stage2ModelData


def build_model_data(instance: Stage2Instance, config: Stage2Config) -> Stage2ModelData:
    """Build scipy.optimize.milp vectors and sparse constraint matrix."""

    route_table = instance.core_route_table.copy().reset_index(drop=True)
    core_ids = list(instance.core_summary["core_id"])
    resources = _resources_used_by_model(route_table, instance.capacity_by_resource_h)

    variable_names: List[str] = []
    variable_groups: Dict[str, object] = {}

    accept_index = {}
    for core_id in core_ids:
        accept_index[core_id] = len(variable_names)
        variable_names.append(f"accept[{core_id}]")

    route_index = {}
    for row in route_table.itertuples(index=True):
        key = (row.core_id, row.route_class)
        route_index[key] = len(variable_names)
        variable_names.append(f"x[{row.core_id},{row.route_class}]")

    procurement_index = len(variable_names)
    variable_names.append("procurement_units")
    shortage_index = len(variable_names)
    variable_names.append("shortage_units")

    overtime_index = {}
    for resource in resources:
        overtime_index[resource] = len(variable_names)
        variable_names.append(f"overtime[{resource}]")

    n_variables = len(variable_names)
    objective = np.zeros(n_variables, dtype=float)
    integrality = np.zeros(n_variables, dtype=int)
    lower_bounds = np.zeros(n_variables, dtype=float)
    upper_bounds = np.full(n_variables, np.inf, dtype=float)

    for index in accept_index.values():
        integrality[index] = 1
        upper_bounds[index] = 1.0
    for index in route_index.values():
        integrality[index] = 1
        upper_bounds[index] = 1.0
    integrality[procurement_index] = 1
    integrality[shortage_index] = 1

    fixed_cost = instance.core_summary.set_index("core_id")["fixed_accept_cost_rmb"].to_dict()
    for core_id, index in accept_index.items():
        objective[index] = float(fixed_cost[core_id])

    objective_terms = []
    for row in route_table.itertuples(index=False):
        index = route_index[(row.core_id, row.route_class)]
        economic = float(row.economic_cost_rmb)
        environmental = config.env_weight * float(row.environmental_score)
        quality = config.quality_weight * max(0.0, instance.target_quality_score - float(row.expected_output_quality))
        reliability = config.quality_weight * config.reliability_weight * float(row.risk_penalty)
        objective[index] = economic + environmental + quality + reliability
        objective_terms.append(
            {
                "variable_name": variable_names[index],
                "core_id": row.core_id,
                "route_class": row.route_class,
                "economic_cost_rmb": economic,
                "environmental_cost_equiv": environmental,
                "quality_penalty_equiv": quality,
                "reliability_penalty_equiv": reliability,
                "objective_coefficient": objective[index],
            }
        )

    procurement_unit_cost = _procurement_unit_cost(instance)
    objective[procurement_index] = procurement_unit_cost
    objective[shortage_index] = _shortage_unit_penalty(instance, config)
    for resource, index in overtime_index.items():
        objective[index] = config.overtime_penalty_rmb_per_h * _resource_overtime_multiplier(resource)

    rows: List[Dict[int, float]] = []
    lhs: List[float] = []
    rhs: List[float] = []
    names: List[str] = []

    route_by_core = route_table.groupby("core_id")
    for core_id in core_ids:
        route_vars = [route_index[(core_id, route_class)] for route_class in route_by_core.get_group(core_id)["route_class"]]
        rows.append({**{index: 1.0 for index in route_vars}, accept_index[core_id]: -1.0})
        lhs.append(0.0)
        rhs.append(0.0)
        names.append(f"one_route_if_accepted[{core_id}]")

    # Demand balance: accepted productive cores plus procurement plus shortage must cover demand.
    productive_route_vars = {
        route_index[(row.core_id, row.route_class)]: 1.0
        for row in route_table.itertuples(index=False)
        if row.route_class != "scrap"
    }
    demand_row = dict(productive_route_vars)
    demand_row[procurement_index] = 1.0
    demand_row[shortage_index] = 1.0
    rows.append(demand_row)
    lhs.append(float(instance.demand_units))
    rhs.append(np.inf)
    names.append("aggregate_demand_balance")

    for resource in resources:
        resource_col = f"resource_h__{resource}"
        row = {}
        for assignment in route_table.itertuples(index=False):
            value = float(getattr(assignment, resource_col, 0.0)) if resource_col in route_table.columns else 0.0
            if abs(value) > 1e-12:
                row[route_index[(assignment.core_id, assignment.route_class)]] = value
        procurement_resource_h = _procurement_resource_hours(resource)
        if procurement_resource_h:
            row[procurement_index] = procurement_resource_h
        row[overtime_index[resource]] = -1.0
        rows.append(row)
        lhs.append(-np.inf)
        rhs.append(float(instance.capacity_by_resource_h.get(resource, 0.0)))
        names.append(f"capacity[{resource}]")

    # Accepted units must meet average residual life and quality floors.
    life_row = {}
    for assignment in route_table.itertuples(index=False):
        coef = float(assignment.expected_residual_life_h) - instance.min_required_life_h * config.life_constraint_ratio
        life_row[route_index[(assignment.core_id, assignment.route_class)]] = coef
    life_row[procurement_index] = max(0.0, 9000.0 - instance.min_required_life_h * config.life_constraint_ratio)
    rows.append(life_row)
    lhs.append(0.0)
    rhs.append(np.inf)
    names.append("average_residual_life_floor")

    quality_row = {}
    for assignment in route_table.itertuples(index=False):
        coef = float(assignment.expected_output_quality) - instance.target_quality_score
        quality_row[route_index[(assignment.core_id, assignment.route_class)]] = coef
    quality_row[procurement_index] = 0.96 - instance.target_quality_score
    rows.append(quality_row)
    lhs.append(0.0)
    rhs.append(np.inf)
    names.append("average_quality_floor")

    matrix = lil_matrix((len(rows), n_variables), dtype=float)
    for row_number, entries in enumerate(rows):
        for col_number, value in entries.items():
            matrix[row_number, col_number] = value

    variable_groups.update(
        {
            "accept": accept_index,
            "route": {f"{core_id}|{route_class}": index for (core_id, route_class), index in route_index.items()},
            "procurement": procurement_index,
            "shortage": shortage_index,
            "overtime": overtime_index,
        }
    )

    return Stage2ModelData(
        variable_names=variable_names,
        objective=objective,
        integrality=integrality,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        constraint_matrix=matrix.tocsr(),
        constraint_lhs=np.array(lhs, dtype=float),
        constraint_rhs=np.array(rhs, dtype=float),
        constraint_names=names,
        variable_groups=variable_groups,
        objective_terms=pd.DataFrame(objective_terms),
        route_assignments=route_table,
    )


def _resources_used_by_model(route_table: pd.DataFrame, capacity_by_resource: Dict[str, float]) -> List[str]:
    resources = set()
    for column in route_table.columns:
        if column.startswith("resource_h__") and route_table[column].abs().sum() > 1e-9:
            resources.add(column.removeprefix("resource_h__"))
    for resource in ["procurement", "incoming_inspection", "assembly", "testing"]:
        if resource in capacity_by_resource:
            resources.add(resource)
    return sorted(resources)


def _procurement_unit_cost(instance: Stage2Instance) -> float:
    machine_cost = float(instance.machine_summary.get("remanufacturing_cost_base_rmb", 0.0))
    selling_price = float(instance.machine_summary.get("selling_price", 0.0))
    return max(machine_cost * 1.45, selling_price * 0.92, 1.0)


def _shortage_unit_penalty(instance: Stage2Instance, config: Stage2Config) -> float:
    if config.shortage_penalty_rmb is not None:
        return float(config.shortage_penalty_rmb)
    selling_price = float(instance.machine_summary.get("selling_price", 0.0))
    machine_cost = float(instance.machine_summary.get("remanufacturing_cost_base_rmb", 0.0))
    return max(2.0 * selling_price, 2.8 * machine_cost, 1.0)


def _resource_overtime_multiplier(resource: str) -> float:
    if resource in {"laser", "quality_monitoring", "reliability_assessment"}:
        return 1.8
    if resource in {"assembly", "testing"}:
        return 1.3
    return 1.0


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
