"""MILP matrix assembly for Stage 3 multi-period component routing."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix

from .aggregation import PRODUCTIVE_ROUTES
from .config import Stage3Config
from .structures import Stage3Instance, Stage3ModelData


def build_model_data(instance: Stage3Instance, config: Stage3Config) -> Stage3ModelData:
    route_table = instance.component_route_period_table.reset_index(drop=True)
    periods = instance.periods
    component_types = instance.component_types
    resources = _resources_used(route_table, instance.capacity_table)
    core_ids = list(instance.core_summary["core_id"])

    variable_names: List[str] = []
    accept_index = {}
    for core_id in core_ids:
        accept_index[core_id] = len(variable_names)
        variable_names.append(f"accept_core[{core_id}]")

    x_index = {}
    for row in route_table.itertuples(index=True):
        key = (row.component_instance_id, row.route_id, row.period_id)
        x_index[key] = len(variable_names)
        variable_names.append(f"x[{row.component_instance_id},{row.route_id},{row.period_id}]")

    procure_index = {}
    inventory_index = {}
    for component_type in component_types:
        for period in periods:
            procure_index[(component_type, period)] = len(variable_names)
            variable_names.append(f"procure[{component_type},{period}]")
            inventory_index[(component_type, period)] = len(variable_names)
            variable_names.append(f"inventory[{component_type},{period}]")

    assemble_index = {}
    backlog_index = {}
    for period in periods:
        assemble_index[period] = len(variable_names)
        variable_names.append(f"assemble[{period}]")
        backlog_index[period] = len(variable_names)
        variable_names.append(f"backlog[{period}]")

    overtime_index = {}
    for resource in resources:
        for period in periods:
            overtime_index[(resource, period)] = len(variable_names)
            variable_names.append(f"overtime[{resource},{period}]")

    n_variables = len(variable_names)
    objective = np.zeros(n_variables, dtype=float)
    integrality = np.zeros(n_variables, dtype=int)
    lower_bounds = np.zeros(n_variables, dtype=float)
    upper_bounds = np.full(n_variables, np.inf, dtype=float)

    for index in accept_index.values():
        integrality[index] = 1
        upper_bounds[index] = 1.0
    for index in x_index.values():
        integrality[index] = 1
        upper_bounds[index] = 1.0
    for group in [procure_index, inventory_index, assemble_index, backlog_index]:
        for index in group.values():
            integrality[index] = 1

    objective_terms = []
    fixed_cost = instance.core_summary.set_index("core_id")["fixed_accept_cost_rmb"].to_dict()
    for core_id, index in accept_index.items():
        objective[index] = float(fixed_cost[core_id])

    for row in route_table.itertuples(index=False):
        index = x_index[(row.component_instance_id, row.route_id, row.period_id)]
        economic = float(row.economic_cost_rmb)
        environmental = config.env_weight * float(row.environmental_score)
        quality = config.quality_weight * max(0.0, instance.target_quality_score - float(row.expected_output_quality))
        reliability = config.quality_weight * config.reliability_weight * float(row.risk_penalty)
        objective[index] = economic + environmental + quality + reliability
        objective_terms.append(
            {
                "variable_name": variable_names[index],
                "component_instance_id": row.component_instance_id,
                "route_id": row.route_id,
                "period_id": row.period_id,
                "economic_cost_rmb": economic,
                "environmental_cost_equiv": environmental,
                "quality_penalty_equiv": quality,
                "reliability_penalty_equiv": reliability,
                "objective_coefficient": objective[index],
            }
        )

    procurement_cost = instance.procurement_costs.set_index("component_type")["unit_procurement_cost_rmb"].to_dict()
    inventory_value = instance.initial_inventory.set_index("component_type")["unit_value_rmb"].to_dict()
    for (component_type, _period), index in procure_index.items():
        objective[index] = float(procurement_cost.get(component_type, inventory_value.get(component_type, 1.0)))
    for (component_type, _period), index in inventory_index.items():
        objective[index] = config.inventory_holding_rate * float(inventory_value.get(component_type, 1.0))
    for period, index in backlog_index.items():
        objective[index] = _backlog_penalty(instance, config)
    for (resource, _period), index in overtime_index.items():
        objective[index] = config.overtime_penalty_rmb_per_h * _resource_overtime_multiplier(resource)

    rows: List[Dict[int, float]] = []
    lhs: List[float] = []
    rhs: List[float] = []
    names: List[str] = []

    route_by_component = route_table.groupby("component_instance_id")
    component_to_core = instance.component_summary.set_index("component_instance_id")["core_id"].to_dict()
    for component_id, group in route_by_component:
        row = {x_index[(r.component_instance_id, r.route_id, r.period_id)]: 1.0 for r in group.itertuples(index=False)}
        row[accept_index[component_to_core[component_id]]] = -1.0
        rows.append(row)
        lhs.append(-np.inf)
        rhs.append(0.0)
        names.append(f"component_processed_if_core_accepted[{component_id}]")

    capacity = instance.capacity_table.set_index(["resource_type", "period_id"])["available_regular_hours"].to_dict()
    for resource in resources:
        resource_col = f"resource_h__{resource}"
        for period in periods:
            row = {}
            if resource_col in route_table.columns:
                period_rows = route_table[route_table["period_id"] == period]
                for assignment in period_rows.itertuples(index=False):
                    value = float(getattr(assignment, resource_col, 0.0))
                    if abs(value) > 1e-12:
                        row[x_index[(assignment.component_instance_id, assignment.route_id, assignment.period_id)]] = value
            row[overtime_index[(resource, period)]] = -1.0
            rows.append(row)
            lhs.append(-np.inf)
            rhs.append(float(capacity.get((resource, period), 0.0)))
            names.append(f"capacity[{resource},{period}]")

    initial_inventory = instance.initial_inventory.set_index("component_type")["initial_quantity_available"].to_dict()
    bom_required = instance.bom_requirements.set_index("component_type")["required_quantity"].to_dict()
    productive_rows = route_table[route_table["route_id"].isin(PRODUCTIVE_ROUTES)]
    for component_type in component_types:
        produced_by_period = productive_rows[productive_rows["component_type"] == component_type].groupby("period_id")
        for position, period in enumerate(periods):
            row = {inventory_index[(component_type, period)]: 1.0}
            if position > 0:
                row[inventory_index[(component_type, periods[position - 1])]] = -1.0
            row[procure_index[(component_type, period)]] = -1.0
            if period in produced_by_period.groups:
                for assignment in produced_by_period.get_group(period).itertuples(index=False):
                    row[x_index[(assignment.component_instance_id, assignment.route_id, assignment.period_id)]] = -1.0
            row[assemble_index[period]] = float(bom_required[component_type])
            rhs_value = float(initial_inventory.get(component_type, 0.0)) if position == 0 else 0.0
            rows.append(row)
            lhs.append(rhs_value)
            rhs.append(rhs_value)
            names.append(f"inventory_balance[{component_type},{period}]")

    demand = instance.period_demand.set_index("period_id")["demand_units"].to_dict()
    for position, period in enumerate(periods):
        row = {backlog_index[period]: 1.0, assemble_index[period]: 1.0}
        if position > 0:
            row[backlog_index[periods[position - 1]]] = -1.0
        rows.append(row)
        lhs.append(float(demand.get(period, 0)))
        rhs.append(float(demand.get(period, 0)))
        names.append(f"demand_backlog_balance[{period}]")

    matrix = lil_matrix((len(rows), n_variables), dtype=float)
    for row_number, entries in enumerate(rows):
        for col_number, value in entries.items():
            matrix[row_number, col_number] = value

    return Stage3ModelData(
        variable_names=variable_names,
        objective=objective,
        integrality=integrality,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        constraint_matrix=matrix.tocsr(),
        constraint_lhs=np.array(lhs, dtype=float),
        constraint_rhs=np.array(rhs, dtype=float),
        constraint_names=names,
        variable_groups={
            "accept_core": accept_index,
            "x": {f"{component}|{route}|{period}": index for (component, route, period), index in x_index.items()},
            "procure": {f"{component}|{period}": index for (component, period), index in procure_index.items()},
            "inventory": {f"{component}|{period}": index for (component, period), index in inventory_index.items()},
            "assemble": assemble_index,
            "backlog": backlog_index,
            "overtime": {f"{resource}|{period}": index for (resource, period), index in overtime_index.items()},
        },
        objective_terms=pd.DataFrame(objective_terms),
    )


def _resources_used(route_table: pd.DataFrame, capacity_table: pd.DataFrame) -> List[str]:
    resources = set(capacity_table["resource_type"].unique())
    for column in route_table.columns:
        if column.startswith("resource_h__") and route_table[column].abs().sum() > 1e-9:
            resources.add(column.removeprefix("resource_h__"))
    return sorted(resources)


def _backlog_penalty(instance: Stage3Instance, config: Stage3Config) -> float:
    if config.backlog_penalty_rmb_per_unit_period is not None:
        return float(config.backlog_penalty_rmb_per_unit_period)
    selling_price = float(instance.machine_summary.get("selling_price", 0.0))
    return max(0.12 * selling_price, 1.0)


def _resource_overtime_multiplier(resource: str) -> float:
    if resource in {"laser", "quality_monitoring", "reliability_assessment"}:
        return 1.8
    if resource in {"assembly", "testing"}:
        return 1.3
    return 1.0
