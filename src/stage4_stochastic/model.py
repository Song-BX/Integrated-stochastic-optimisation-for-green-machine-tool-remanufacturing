"""MILP matrix assembly for the Stage 4 stochastic SAA model."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix

from stage3_multiperiod.aggregation import PRODUCTIVE_ROUTES

from .config import Stage4Config
from .structures import Stage4Instance, Stage4ModelData


def build_model_data(instance: Stage4Instance, config: Stage4Config) -> Stage4ModelData:
    route_table = instance.component_route_period_scenario_table.reset_index(drop=True)
    periods = instance.periods
    scenarios = instance.scenario_ids
    component_types = instance.component_types
    resources = _resources_used(route_table, instance.scenario_capacity_table)
    core_ids = list(instance.core_summary["core_id"])

    variable_names: List[str] = []
    accept_index = {}
    for core_id in core_ids:
        accept_index[core_id] = len(variable_names)
        variable_names.append(f"accept_core[{core_id}]")

    pre_procure_index = {}
    for component_type in component_types:
        for period in periods:
            pre_procure_index[(component_type, period)] = len(variable_names)
            variable_names.append(f"pre_procure[{component_type},{period}]")

    x_index = {}
    for row in route_table.itertuples(index=True):
        key = (row.scenario_id, row.component_instance_id, row.route_id, row.period_id)
        x_index[key] = len(variable_names)
        variable_names.append(f"x[{row.scenario_id},{row.component_instance_id},{row.route_id},{row.period_id}]")

    recourse_procure_index = {}
    inventory_index = {}
    for scenario_id in scenarios:
        for component_type in component_types:
            for period in periods:
                recourse_procure_index[(scenario_id, component_type, period)] = len(variable_names)
                variable_names.append(f"recourse_procure[{scenario_id},{component_type},{period}]")
                inventory_index[(scenario_id, component_type, period)] = len(variable_names)
                variable_names.append(f"inventory[{scenario_id},{component_type},{period}]")

    assemble_index = {}
    backlog_index = {}
    for scenario_id in scenarios:
        for period in periods:
            assemble_index[(scenario_id, period)] = len(variable_names)
            variable_names.append(f"assemble[{scenario_id},{period}]")
            backlog_index[(scenario_id, period)] = len(variable_names)
            variable_names.append(f"backlog[{scenario_id},{period}]")

    overtime_index = {}
    for scenario_id in scenarios:
        for resource in resources:
            for period in periods:
                overtime_index[(scenario_id, resource, period)] = len(variable_names)
                variable_names.append(f"overtime[{scenario_id},{resource},{period}]")

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
    for group in [pre_procure_index, recourse_procure_index, inventory_index, assemble_index, backlog_index]:
        for index in group.values():
            integrality[index] = 1

    objective_terms = []
    fixed_cost = instance.core_summary.set_index("core_id")["fixed_accept_cost_rmb"].to_dict()
    for core_id, index in accept_index.items():
        objective[index] = float(fixed_cost[core_id])

    procurement_cost = instance.procurement_costs.set_index("component_type")["unit_procurement_cost_rmb"].to_dict()
    inventory_value = instance.initial_inventory.set_index("component_type")["unit_value_rmb"].to_dict()
    scenario_probability = instance.scenario_sample.set_index("scenario_id")["saa_probability"].to_dict()
    scenario_multipliers = instance.scenario_sample.set_index("scenario_id").to_dict(orient="index")

    for (component_type, _period), index in pre_procure_index.items():
        objective[index] = float(procurement_cost.get(component_type, inventory_value.get(component_type, 1.0)))

    for row in route_table.itertuples(index=False):
        index = x_index[(row.scenario_id, row.component_instance_id, row.route_id, row.period_id)]
        probability = float(row.saa_probability)
        environmental_multiplier = float(scenario_multipliers[row.scenario_id].get("environmental_cost_multiplier", 1.0))
        warranty_multiplier = float(scenario_multipliers[row.scenario_id].get("warranty_failure_multiplier", 1.0))
        economic = float(row.economic_cost_rmb)
        environmental = config.env_weight * environmental_multiplier * float(row.environmental_score)
        quality = config.quality_weight * max(0.0, instance.target_quality_score - float(row.expected_output_quality))
        reliability = config.quality_weight * config.reliability_weight * warranty_multiplier * float(row.risk_penalty)
        warranty = warranty_multiplier * float(row.warranty_risk_cost_rmb)
        coefficient = probability * (economic + environmental + quality + reliability + warranty)
        objective[index] = coefficient
        objective_terms.append(
            {
                "variable_name": variable_names[index],
                "scenario_id": row.scenario_id,
                "saa_probability": probability,
                "component_instance_id": row.component_instance_id,
                "route_id": row.route_id,
                "period_id": row.period_id,
                "economic_cost_rmb": economic,
                "environmental_cost_equiv": environmental,
                "quality_penalty_equiv": quality,
                "reliability_penalty_equiv": reliability,
                "warranty_risk_cost_rmb": warranty,
                "weighted_objective_coefficient": coefficient,
            }
        )

    for (scenario_id, component_type, _period), index in recourse_procure_index.items():
        price_multiplier = float(scenario_multipliers[scenario_id].get("procurement_price_multiplier", 1.0))
        base_cost = float(procurement_cost.get(component_type, inventory_value.get(component_type, 1.0)))
        objective[index] = float(scenario_probability[scenario_id]) * base_cost * config.recourse_procurement_premium * price_multiplier
    for (scenario_id, component_type, _period), index in inventory_index.items():
        objective[index] = float(scenario_probability[scenario_id]) * config.inventory_holding_rate * float(inventory_value.get(component_type, 1.0))
    for (scenario_id, _period), index in backlog_index.items():
        objective[index] = float(scenario_probability[scenario_id]) * _backlog_penalty(instance, config)
    for (scenario_id, resource, _period), index in overtime_index.items():
        objective[index] = float(scenario_probability[scenario_id]) * config.overtime_penalty_rmb_per_h * _resource_overtime_multiplier(resource)

    rows: List[Dict[int, float]] = []
    lhs: List[float] = []
    rhs: List[float] = []
    names: List[str] = []

    component_to_core = instance.component_summary.set_index("component_instance_id")["core_id"].to_dict()
    for (scenario_id, component_id), group in route_table.groupby(["scenario_id", "component_instance_id"]):
        row = {
            x_index[(r.scenario_id, r.component_instance_id, r.route_id, r.period_id)]: 1.0
            for r in group.itertuples(index=False)
        }
        row[accept_index[component_to_core[component_id]]] = -1.0
        rows.append(row)
        lhs.append(-np.inf)
        rhs.append(0.0)
        names.append(f"component_processed_if_core_accepted[{scenario_id},{component_id}]")

    capacity = instance.scenario_capacity_table.set_index(["scenario_id", "resource_type", "period_id"])["available_regular_hours"].to_dict()
    for scenario_id in scenarios:
        scenario_rows = route_table[route_table["scenario_id"] == scenario_id]
        for resource in resources:
            resource_col = f"resource_h__{resource}"
            for period in periods:
                row = {}
                if resource_col in route_table.columns:
                    period_rows = scenario_rows[scenario_rows["period_id"] == period]
                    for assignment in period_rows.itertuples(index=False):
                        value = float(getattr(assignment, resource_col, 0.0))
                        if abs(value) > 1e-12:
                            row[x_index[(assignment.scenario_id, assignment.component_instance_id, assignment.route_id, assignment.period_id)]] = value
                row[overtime_index[(scenario_id, resource, period)]] = -1.0
                rows.append(row)
                lhs.append(-np.inf)
                rhs.append(float(capacity.get((scenario_id, resource, period), 0.0)))
                names.append(f"capacity[{scenario_id},{resource},{period}]")

    initial_inventory = instance.initial_inventory.set_index("component_type")["initial_quantity_available"].to_dict()
    bom_required = instance.bom_requirements.set_index("component_type")["required_quantity"].to_dict()
    productive_rows = route_table[route_table["route_id"].isin(PRODUCTIVE_ROUTES)]
    for scenario_id in scenarios:
        scenario_productive = productive_rows[productive_rows["scenario_id"] == scenario_id]
        for component_type in component_types:
            component_productive = scenario_productive[scenario_productive["component_type"] == component_type]
            produced_by_period = component_productive.groupby("period_id")
            for position, period in enumerate(periods):
                row = {inventory_index[(scenario_id, component_type, period)]: 1.0}
                if position > 0:
                    row[inventory_index[(scenario_id, component_type, periods[position - 1])]] = -1.0
                row[pre_procure_index[(component_type, period)]] = -1.0
                row[recourse_procure_index[(scenario_id, component_type, period)]] = -1.0
                if period in produced_by_period.groups:
                    for assignment in produced_by_period.get_group(period).itertuples(index=False):
                        row[x_index[(assignment.scenario_id, assignment.component_instance_id, assignment.route_id, assignment.period_id)]] = -1.0
                row[assemble_index[(scenario_id, period)]] = float(bom_required[component_type])
                rhs_value = float(initial_inventory.get(component_type, 0.0)) if position == 0 else 0.0
                rows.append(row)
                lhs.append(rhs_value)
                rhs.append(rhs_value)
                names.append(f"inventory_balance[{scenario_id},{component_type},{period}]")

    demand = instance.scenario_demand.set_index(["scenario_id", "period_id"])["demand_units"].to_dict()
    for scenario_id in scenarios:
        for position, period in enumerate(periods):
            row = {backlog_index[(scenario_id, period)]: 1.0, assemble_index[(scenario_id, period)]: 1.0}
            if position > 0:
                row[backlog_index[(scenario_id, periods[position - 1])]] = -1.0
            rows.append(row)
            lhs.append(float(demand.get((scenario_id, period), 0)))
            rhs.append(float(demand.get((scenario_id, period), 0)))
            names.append(f"demand_backlog_balance[{scenario_id},{period}]")

    matrix = lil_matrix((len(rows), n_variables), dtype=float)
    for row_number, entries in enumerate(rows):
        for col_number, value in entries.items():
            matrix[row_number, col_number] = value

    return Stage4ModelData(
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
            "pre_procure": {f"{component}|{period}": index for (component, period), index in pre_procure_index.items()},
            "x": {
                f"{scenario}|{component}|{route}|{period}": index
                for (scenario, component, route, period), index in x_index.items()
            },
            "recourse_procure": {
                f"{scenario}|{component}|{period}": index
                for (scenario, component, period), index in recourse_procure_index.items()
            },
            "inventory": {
                f"{scenario}|{component}|{period}": index
                for (scenario, component, period), index in inventory_index.items()
            },
            "assemble": {f"{scenario}|{period}": index for (scenario, period), index in assemble_index.items()},
            "backlog": {f"{scenario}|{period}": index for (scenario, period), index in backlog_index.items()},
            "overtime": {
                f"{scenario}|{resource}|{period}": index
                for (scenario, resource, period), index in overtime_index.items()
            },
        },
        objective_terms=pd.DataFrame(objective_terms),
    )


def _resources_used(route_table: pd.DataFrame, capacity_table: pd.DataFrame) -> List[str]:
    resources = set(capacity_table["resource_type"].unique())
    for column in route_table.columns:
        if column.startswith("resource_h__") and route_table[column].abs().sum() > 1e-9:
            resources.add(column.removeprefix("resource_h__"))
    return sorted(resources)


def _backlog_penalty(instance: Stage4Instance, config: Stage4Config) -> float:
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
