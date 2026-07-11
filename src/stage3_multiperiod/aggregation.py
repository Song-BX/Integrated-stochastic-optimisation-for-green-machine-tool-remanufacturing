"""Build deterministic multi-period component-routing instances."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .config import Stage3Config
from .structures import Stage3Instance


PRODUCTIVE_ROUTES = {"R1", "R2", "R3", "R4", "R5", "R6"}


def build_stage3_instance(tables: Dict[str, pd.DataFrame], config: Stage3Config) -> Stage3Instance:
    machine = _single_machine_row(tables["machine_types"], config.machine_type_id)
    periods = _period_window(tables["time_periods"], config.period_start, config.period_count)
    period_set = set(periods)
    machine_family = str(machine["machine_family"])
    min_required_life_h = float(machine["min_required_life_h"])
    target_quality = max(config.quality_floor, 0.70)

    orders = tables["orders"][
        (tables["orders"]["machine_type_id"] == config.machine_type_id)
        & (tables["orders"]["due_period"].isin(period_set))
    ].copy()
    cores = tables["returned_cores"][
        (tables["returned_cores"]["machine_type_id"] == config.machine_type_id)
        & (tables["returned_cores"]["arrival_period"].isin(period_set))
    ].copy()
    components = tables["component_inspection"][
        (tables["component_inspection"]["machine_type_id"] == config.machine_type_id)
        & (tables["component_inspection"]["core_id"].isin(set(cores["core_id"])))
    ].copy()
    if orders.empty:
        raise ValueError(f"No {config.machine_type_id} orders due inside {periods[0]}-{periods[-1]}.")
    if cores.empty or components.empty:
        raise ValueError(f"No {config.machine_type_id} returned cores/components inside {periods[0]}-{periods[-1]}.")

    bom = tables["bom"][tables["bom"]["machine_type_id"] == config.machine_type_id].copy()
    period_demand = _period_demand(orders, periods)
    route_coefficients = _route_coefficients(tables["processing_parameters"])
    route_feasibility = _route_feasibility(tables["route_feasibility"])
    component_summary = _component_summary(components, route_feasibility, route_coefficients)
    component_route_period = _component_route_period_table(component_summary, route_coefficients, periods)
    core_summary = _core_summary(cores, component_summary)
    initial_inventory = _initial_inventory(tables["initial_inventory"], config.machine_type_id, bom)
    procurement_costs = _procurement_costs(tables["procurement_parameters"], bom, config)
    capacity_table = _capacity_table(tables, config, periods)

    route_ids = sorted(component_route_period["route_id"].unique())
    component_types = sorted(bom["component_type"].unique())
    resource_types = sorted(capacity_table["resource_type"].unique())

    return Stage3Instance(
        machine_type_id=config.machine_type_id,
        machine_family=machine_family,
        periods=periods,
        component_types=component_types,
        route_ids=route_ids,
        resource_types=resource_types,
        demand_units=int(period_demand["demand_units"].sum()),
        candidate_core_count=int(cores["core_id"].nunique()),
        component_instance_count=int(component_summary["component_instance_id"].nunique()),
        min_required_life_h=min_required_life_h,
        target_quality_score=target_quality,
        machine_summary={
            "selling_price": float(machine["selling_price"]),
            "remanufacturing_cost_base_rmb": float(machine["remanufacturing_cost_base_rmb"]),
            "estimated_core_acquisition_cost_rmb": float(machine["estimated_core_acquisition_cost_rmb"]),
            "annual_demand_weight": float(machine.get("annual_demand_weight", 1.0)),
            "capacity_share": _capacity_share(tables["machine_types"], machine, config),
        },
        period_demand=period_demand,
        bom_requirements=bom[["component_type", "required_quantity", "replacement_cost_rmb"]].copy(),
        core_summary=core_summary,
        component_summary=component_summary,
        component_route_period_table=component_route_period,
        initial_inventory=initial_inventory,
        procurement_costs=procurement_costs,
        capacity_table=capacity_table,
    )


def _single_machine_row(machine_types: pd.DataFrame, machine_type_id: str) -> pd.Series:
    matches = machine_types[machine_types["machine_type_id"] == machine_type_id]
    if matches.empty:
        raise ValueError(f"Unknown machine_type_id: {machine_type_id}")
    return matches.iloc[0]


def _period_window(time_periods: pd.DataFrame, start: str, count: int) -> List[str]:
    ordered = time_periods.sort_values("period_index").reset_index(drop=True)
    matches = ordered.index[ordered["period_id"] == start].tolist()
    if not matches:
        raise ValueError(f"Unknown period_start: {start}")
    begin = matches[0]
    window = ordered.iloc[begin : begin + count]["period_id"].tolist()
    if len(window) != count:
        raise ValueError(f"Requested {count} periods from {start}, but only {len(window)} are available.")
    return window


def _period_demand(orders: pd.DataFrame, periods: List[str]) -> pd.DataFrame:
    orders = orders.copy()
    orders["quantity"] = pd.to_numeric(orders["quantity"], errors="coerce").fillna(0.0)
    by_period = orders.groupby("due_period")["quantity"].sum().to_dict()
    return pd.DataFrame(
        {
            "period_id": periods,
            "period_index_stage3": list(range(len(periods))),
            "demand_units": [int(by_period.get(period, 0)) for period in periods],
        }
    )


def _route_coefficients(processing_parameters: pd.DataFrame) -> pd.DataFrame:
    data = processing_parameters.copy()
    numeric_cols = [
        "base_processing_time_h",
        "variable_cost",
        "labor_cost_rmb",
        "material_cost_rmb",
        "setup_cost_rmb",
        "expected_energy_kwh",
        "expected_water_m3",
        "expected_pollutant_kg",
        "expected_carbon_kg",
        "quality_gain_contribution",
        "residual_life_gain_contribution_h",
        "route_success_probability_reference",
        "route_failure_probability_reference",
    ]
    for column in numeric_cols:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.0)
    data["operation_cost_rmb"] = data["variable_cost"] + data["labor_cost_rmb"] + data["material_cost_rmb"] + data["setup_cost_rmb"]
    grouped_resource = (
        data.groupby(["component_type", "quality_state", "route_id", "required_resource_type"], dropna=False)
        .agg(resource_hours=("base_processing_time_h", "sum"))
        .reset_index()
    )
    route_level = (
        data.groupby(["component_type", "quality_state", "route_id"], dropna=False)
        .agg(
            processing_time_h=("base_processing_time_h", "sum"),
            operation_cost_rmb=("operation_cost_rmb", "sum"),
            energy_kwh=("expected_energy_kwh", "sum"),
            water_m3=("expected_water_m3", "sum"),
            pollutant_kg=("expected_pollutant_kg", "sum"),
            carbon_kg=("expected_carbon_kg", "sum"),
            quality_gain=("quality_gain_contribution", "sum"),
            residual_life_gain_h=("residual_life_gain_contribution_h", "sum"),
            success_probability=("route_success_probability_reference", "mean"),
            failure_probability=("route_failure_probability_reference", "mean"),
        )
        .reset_index()
    )
    resource_pivot = grouped_resource.pivot_table(
        index=["component_type", "quality_state", "route_id"],
        columns="required_resource_type",
        values="resource_hours",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    resource_pivot.columns = [
        f"resource_h__{column}" if column not in {"component_type", "quality_state", "route_id"} else column
        for column in resource_pivot.columns
    ]
    return route_level.merge(resource_pivot, on=["component_type", "quality_state", "route_id"], how="left").fillna(0.0)


def _route_feasibility(route_feasibility: pd.DataFrame) -> pd.DataFrame:
    feasible = route_feasibility[route_feasibility["feasible"] == 1].copy()
    return feasible[["component_type", "quality_state", "route_id", "route_feasibility_score"]].copy()


def _component_summary(
    components: pd.DataFrame,
    route_feasibility: pd.DataFrame,
    route_coefficients: pd.DataFrame,
) -> pd.DataFrame:
    data = components.copy()
    numeric_cols = [
        "quality_score",
        "residual_life_mean_h",
        "failure_probability_prior",
        "inspection_cost",
        "replacement_cost_rmb",
        "is_key_component",
        "failure_consequence_index",
    ]
    for column in numeric_cols:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.0)
    available_routes = set(zip(route_coefficients["component_type"], route_coefficients["quality_state"], route_coefficients["route_id"]))
    feasible_routes = []
    for row in data.itertuples(index=False):
        routes = route_feasibility[
            (route_feasibility["component_type"] == row.component_type)
            & (route_feasibility["quality_state"] == row.observed_quality_state)
        ]["route_id"].tolist()
        routes = [route_id for route_id in routes if (row.component_type, row.observed_quality_state, route_id) in available_routes]
        feasible_routes.append(";".join(sorted(routes)))
    data["feasible_route_ids"] = feasible_routes
    data = data[data["feasible_route_ids"] != ""].copy()
    return data[
        [
            "component_instance_id",
            "core_id",
            "component_type",
            "observed_quality_state",
            "inspection_period",
            "quality_score",
            "residual_life_mean_h",
            "failure_probability_prior",
            "inspection_cost",
            "replacement_cost_rmb",
            "is_key_component",
            "failure_consequence_index",
            "feasible_route_ids",
        ]
    ].sort_values("component_instance_id").reset_index(drop=True)


def _component_route_period_table(
    component_summary: pd.DataFrame,
    route_coefficients: pd.DataFrame,
    periods: List[str],
) -> pd.DataFrame:
    period_rank = {period: index for index, period in enumerate(periods)}
    coeff_lookup = {
        (row.component_type, row.quality_state, row.route_id): row
        for row in route_coefficients.itertuples(index=False)
    }
    resource_cols = [column for column in route_coefficients.columns if column.startswith("resource_h__")]
    rows: List[Dict[str, object]] = []
    for component in component_summary.itertuples(index=False):
        first_period = _first_available_period(component.inspection_period, periods, period_rank)
        if first_period is None:
            continue
        for route_id in str(component.feasible_route_ids).split(";"):
            coeff = coeff_lookup[(component.component_type, component.observed_quality_state, route_id)]
            for period in periods[first_period:]:
                output_quality = _output_quality(component.quality_score, coeff.quality_gain, route_id)
                output_life = _output_life(component.residual_life_mean_h, coeff.residual_life_gain_h, route_id)
                row = {
                    "component_instance_id": component.component_instance_id,
                    "core_id": component.core_id,
                    "component_type": component.component_type,
                    "quality_state": component.observed_quality_state,
                    "inspection_period": component.inspection_period,
                    "period_id": period,
                    "route_id": route_id,
                    "economic_cost_rmb": float(coeff.operation_cost_rmb),
                    "environmental_score": float(coeff.energy_kwh) * 0.08
                    + float(coeff.water_m3) * 3.0
                    + float(coeff.pollutant_kg) * 18.0
                    + float(coeff.carbon_kg) * 0.25,
                    "energy_kwh": float(coeff.energy_kwh),
                    "water_m3": float(coeff.water_m3),
                    "pollutant_kg": float(coeff.pollutant_kg),
                    "carbon_kg": float(coeff.carbon_kg),
                    "expected_output_quality": output_quality,
                    "expected_residual_life_h": output_life,
                    "risk_penalty": max(0.0, float(component.failure_probability_prior) * (1.0 - min(output_quality, 0.98))),
                    "productive_output": 1 if route_id in PRODUCTIVE_ROUTES else 0,
                }
                for resource_col in resource_cols:
                    row[resource_col] = float(getattr(coeff, resource_col, 0.0))
                rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No feasible component-route-period rows generated for Stage 3.")
    return table.sort_values(["component_instance_id", "period_id", "route_id"]).reset_index(drop=True)


def _first_available_period(
    inspection_period: object,
    periods: List[str],
    period_rank: Dict[str, int],
) -> Optional[int]:
    if inspection_period in period_rank:
        return period_rank[str(inspection_period)]
    inspection_key = _period_sort_key(inspection_period)
    first_key = _period_sort_key(periods[0])
    last_key = _period_sort_key(periods[-1])
    if inspection_key is None or first_key is None or last_key is None:
        return None
    if inspection_key <= first_key:
        return 0
    if inspection_key > last_key:
        return None
    for index, period in enumerate(periods):
        period_key = _period_sort_key(period)
        if period_key is not None and period_key >= inspection_key:
            return index
    return None


def _period_sort_key(period_id: object) -> Optional[int]:
    text = str(period_id)
    digits = "".join(character for character in text if character.isdigit())
    if not digits:
        return None
    return int(digits)


def _output_quality(input_quality: float, quality_gain: float, route_id: str) -> float:
    if route_id == "R6":
        return 0.96
    if route_id == "R7":
        return 0.0
    return max(0.0, min(0.94, float(input_quality) + float(quality_gain)))


def _output_life(input_life: float, life_gain: float, route_id: str) -> float:
    if route_id == "R6":
        return max(9000.0, float(input_life) + float(life_gain))
    if route_id == "R7":
        return 0.0
    return max(0.0, float(input_life) + float(life_gain))


def _core_summary(cores: pd.DataFrame, component_summary: pd.DataFrame) -> pd.DataFrame:
    core_cols = [
        "core_id",
        "acceptability_score",
        "acquisition_cost",
        "transportation_cost_rmb",
        "disassembly_cost",
        "cleaning_cost_estimate_rmb",
    ]
    data = cores[core_cols].copy()
    for column in core_cols:
        if column != "core_id":
            data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.0)
    inspection_cost = component_summary.groupby("core_id")["inspection_cost"].sum().to_dict()
    data["component_inspection_cost_rmb"] = data["core_id"].map(inspection_cost).fillna(0.0)
    data["fixed_accept_cost_rmb"] = (
        data["acquisition_cost"]
        + data["transportation_cost_rmb"]
        + data["disassembly_cost"]
        + data["cleaning_cost_estimate_rmb"]
        + data["component_inspection_cost_rmb"]
    )
    return data.sort_values("core_id").reset_index(drop=True)


def _initial_inventory(initial_inventory: pd.DataFrame, machine_type_id: str, bom: pd.DataFrame) -> pd.DataFrame:
    inventory = initial_inventory[initial_inventory["machine_type_id"] == machine_type_id].copy()
    inventory["quantity_available"] = pd.to_numeric(inventory["quantity_available"], errors="coerce").fillna(0.0)
    by_component = inventory.groupby("component_type")["quantity_available"].sum().to_dict()
    rows = []
    for component_type in sorted(bom["component_type"].unique()):
        replacement_cost = float(bom[bom["component_type"] == component_type]["replacement_cost_rmb"].iloc[0])
        rows.append(
            {
                "component_type": component_type,
                "initial_quantity_available": float(by_component.get(component_type, 0.0)),
                "unit_value_rmb": replacement_cost,
            }
        )
    return pd.DataFrame(rows)


def _procurement_costs(procurement_parameters: pd.DataFrame, bom: pd.DataFrame, config: Stage3Config) -> pd.DataFrame:
    procurement = procurement_parameters[procurement_parameters["machine_type_id"] == config.machine_type_id].copy()
    procurement["base_unit_price_rmb"] = pd.to_numeric(procurement["base_unit_price_rmb"], errors="coerce")
    price_by_component = procurement.groupby("component_type")["base_unit_price_rmb"].median().to_dict()
    rows = []
    for bom_row in bom.itertuples(index=False):
        fallback = float(bom_row.replacement_cost_rmb) * config.procurement_cost_multiplier
        rows.append(
            {
                "component_type": bom_row.component_type,
                "unit_procurement_cost_rmb": float(price_by_component.get(bom_row.component_type, fallback)),
            }
        )
    return pd.DataFrame(rows).drop_duplicates("component_type").sort_values("component_type").reset_index(drop=True)


def _capacity_table(tables: Dict[str, pd.DataFrame], config: Stage3Config, periods: List[str]) -> pd.DataFrame:
    machine = _single_machine_row(tables["machine_types"], config.machine_type_id)
    share = _capacity_share(tables["machine_types"], machine, config)
    capacity = tables["capacity_calendar"][tables["capacity_calendar"]["period_id"].isin(periods)].copy()
    capacity["effective_capacity_h"] = pd.to_numeric(capacity["effective_capacity_h"], errors="coerce").fillna(0.0)
    grouped = capacity.groupby(["period_id", "resource_type"])["effective_capacity_h"].sum().reset_index()
    grouped["available_regular_hours"] = grouped["effective_capacity_h"] * share
    return grouped[["period_id", "resource_type", "available_regular_hours"]].sort_values(["period_id", "resource_type"]).reset_index(drop=True)


def _capacity_share(machine_types: pd.DataFrame, machine: pd.Series, config: Stage3Config) -> float:
    total_weight = float(pd.to_numeric(machine_types["annual_demand_weight"], errors="coerce").fillna(0.0).sum())
    machine_weight = float(machine.get("annual_demand_weight", 1.0))
    return max(config.capacity_share_floor, machine_weight / total_weight) * config.capacity_share_multiplier
