"""Aggregate raw data into a deterministic single-period Stage 2 instance."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .config import Stage2Config
from .structures import Stage2Instance


ROUTE_CLASS_TO_ROUTE_IDS: Dict[str, List[str]] = {
    "reuse": ["R1", "R2"],
    "repair": ["R3", "R5"],
    "laser": ["R4"],
    "replace": ["R6"],
    "scrap": ["R7"],
}

ROUTE_CLASS_ORDER = ["reuse", "repair", "laser", "replace", "scrap"]

ROUTE_CLASS_OUTPUT_QUALITY = {
    "reuse": 0.78,
    "repair": 0.80,
    "laser": 0.84,
    "replace": 0.96,
    "scrap": 0.0,
}


def build_stage2_instance(tables: Dict[str, pd.DataFrame], config: Stage2Config) -> Stage2Instance:
    """Build one deterministic aggregate instance for the configured machine type."""

    machine = _single_machine_row(tables["machine_types"], config.machine_type_id)
    machine_family = str(machine["machine_family"])
    min_required_life_h = float(machine["min_required_life_h"])
    machine_weight = float(machine.get("annual_demand_weight", 1.0))
    total_weight = float(tables["machine_types"]["annual_demand_weight"].sum())

    orders = _filter_periods(
        tables["orders"][tables["orders"]["machine_type_id"] == config.machine_type_id].copy(),
        "order_period",
        config,
    )
    cores = _filter_periods(
        tables["returned_cores"][tables["returned_cores"]["machine_type_id"] == config.machine_type_id].copy(),
        "arrival_period",
        config,
    )
    components = tables["component_inspection"][
        tables["component_inspection"]["machine_type_id"] == config.machine_type_id
    ].copy()
    if not cores.empty:
        components = components[components["core_id"].isin(set(cores["core_id"]))]

    if orders.empty:
        raise ValueError(f"No orders found for {config.machine_type_id} in the selected planning window.")
    if cores.empty or components.empty:
        raise ValueError(f"No returned cores or component inspections found for {config.machine_type_id}.")

    bom = tables["bom"][tables["bom"]["machine_type_id"] == config.machine_type_id].copy()
    bom_item_count = int(bom["bom_item_id"].nunique())
    average_required_component_count = float(bom["required_quantity"].sum()) if not bom.empty else 1.0

    route_coefficients = _build_route_coefficients(tables)
    core_summary = _build_core_summary(cores, components)
    core_route_table = _build_core_route_table(core_summary, components, tables["route_feasibility"], route_coefficients)

    route_classes = [route_class for route_class in ROUTE_CLASS_ORDER if route_class in set(core_route_table["route_class"])]
    if len(route_classes) > config.max_route_classes:
        route_classes = route_classes[: config.max_route_classes]
        core_route_table = core_route_table[core_route_table["route_class"].isin(route_classes)].copy()

    capacity_share = max(config.capacity_share_floor, machine_weight / total_weight) * config.capacity_share_multiplier
    capacity_by_resource_h = _capacity_by_resource(tables["capacity_calendar"], config, capacity_share)

    demand_units = int(pd.to_numeric(orders["quantity"], errors="coerce").fillna(0).sum())
    target_quality = max(float(config.quality_floor), _weighted_average(orders, "requested_min_system_reliability", default=0.92) - 0.22)

    return Stage2Instance(
        machine_type_id=config.machine_type_id,
        machine_family=machine_family,
        demand_units=demand_units,
        candidate_core_count=int(cores["core_id"].nunique()),
        bom_item_count=bom_item_count,
        average_required_component_count=average_required_component_count,
        min_required_life_h=min_required_life_h,
        target_quality_score=target_quality,
        capacity_by_resource_h=capacity_by_resource_h,
        route_classes=route_classes,
        route_class_to_route_ids={key: ROUTE_CLASS_TO_ROUTE_IDS[key] for key in route_classes},
        core_route_table=core_route_table,
        core_summary=core_summary,
        route_coefficients=route_coefficients,
        machine_summary={
            "selling_price": float(machine["selling_price"]),
            "remanufacturing_cost_base_rmb": float(machine["remanufacturing_cost_base_rmb"]),
            "estimated_core_acquisition_cost_rmb": float(machine["estimated_core_acquisition_cost_rmb"]),
            "annual_demand_weight": machine_weight,
            "capacity_share": capacity_share,
            "order_count": int(len(orders)),
            "order_value_rmb": float(pd.to_numeric(orders["order_value_rmb"], errors="coerce").fillna(0).sum()),
            "average_order_life_requirement_h": _weighted_average(orders, "requested_min_residual_life_h", default=min_required_life_h),
        },
        period_filter={
            "planning_period_start": config.planning_period_start,
            "planning_period_end": config.planning_period_end,
        },
    )


def _single_machine_row(machine_types: pd.DataFrame, machine_type_id: str) -> pd.Series:
    matches = machine_types[machine_types["machine_type_id"] == machine_type_id]
    if matches.empty:
        raise ValueError(f"Unknown machine_type_id: {machine_type_id}")
    return matches.iloc[0]


def _filter_periods(df: pd.DataFrame, period_column: str, config: Stage2Config) -> pd.DataFrame:
    if config.planning_period_start:
        df = df[df[period_column] >= config.planning_period_start]
    if config.planning_period_end:
        df = df[df[period_column] <= config.planning_period_end]
    return df


def _build_route_coefficients(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    processing = tables["processing_parameters"].copy()
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
        "resource_capacity_intensity",
    ]
    for column in numeric_cols:
        processing[column] = pd.to_numeric(processing[column], errors="coerce").fillna(0.0)

    processing["operation_cost_rmb"] = (
        processing["variable_cost"]
        + processing["labor_cost_rmb"]
        + processing["material_cost_rmb"]
        + processing["setup_cost_rmb"]
    )
    processing["route_class"] = processing["route_id"].map(_route_class_for_route_id)
    processing = processing.dropna(subset=["route_class"])

    grouped = (
        processing.groupby(["component_type", "quality_state", "route_class", "required_resource_type"], dropna=False)
        .agg(
            processing_time_h=("base_processing_time_h", "sum"),
            operation_cost_rmb=("operation_cost_rmb", "sum"),
            energy_kwh=("expected_energy_kwh", "sum"),
            water_m3=("expected_water_m3", "sum"),
            pollutant_kg=("expected_pollutant_kg", "sum"),
            carbon_kg=("expected_carbon_kg", "sum"),
            quality_gain=("quality_gain_contribution", "sum"),
            residual_life_gain_h=("residual_life_gain_contribution_h", "sum"),
            capacity_intensity=("resource_capacity_intensity", "sum"),
        )
        .reset_index()
    )
    route_level = (
        grouped.groupby(["component_type", "quality_state", "route_class"], dropna=False)
        .agg(
            processing_time_h=("processing_time_h", "sum"),
            operation_cost_rmb=("operation_cost_rmb", "sum"),
            energy_kwh=("energy_kwh", "sum"),
            water_m3=("water_m3", "sum"),
            pollutant_kg=("pollutant_kg", "sum"),
            carbon_kg=("carbon_kg", "sum"),
            quality_gain=("quality_gain", "sum"),
            residual_life_gain_h=("residual_life_gain_h", "sum"),
        )
        .reset_index()
    )
    resource_pivot = grouped.pivot_table(
        index=["component_type", "quality_state", "route_class"],
        columns="required_resource_type",
        values="processing_time_h",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    resource_pivot.columns = [
        f"resource_h__{column}" if column not in {"component_type", "quality_state", "route_class"} else column
        for column in resource_pivot.columns
    ]
    return route_level.merge(resource_pivot, on=["component_type", "quality_state", "route_class"], how="left")


def _build_core_summary(cores: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    components = components.copy()
    for column in [
        "quality_score",
        "residual_life_mean_h",
        "failure_probability_prior",
        "inspection_cost",
        "replacement_cost_rmb",
        "is_key_component",
        "missing_indicator",
        "dimension_out_of_tolerance_flag",
        "life_below_requirement_flag",
    ]:
        components[column] = pd.to_numeric(components[column], errors="coerce").fillna(0.0)

    critical = components["is_key_component"].clip(lower=0.0) + components["failure_consequence_index"].pipe(
        pd.to_numeric, errors="coerce"
    ).fillna(0.0)
    components["_critical_weight"] = critical.clip(lower=0.1)
    grouped = components.groupby("core_id").apply(_summarise_core_components, include_groups=False).reset_index()

    core_cols = [
        "core_id",
        "acceptability_score",
        "acquisition_cost",
        "transportation_cost_rmb",
        "disassembly_cost",
        "cleaning_cost_estimate_rmb",
        "expected_salvage_value_rmb",
        "first_stage_accept_decision",
    ]
    core_data = cores[core_cols].copy()
    for column in core_cols:
        if column != "core_id":
            core_data[column] = pd.to_numeric(core_data[column], errors="coerce").fillna(0.0)
    merged = core_data.merge(grouped, on="core_id", how="inner")
    merged["fixed_accept_cost_rmb"] = (
        merged["acquisition_cost"]
        + merged["transportation_cost_rmb"]
        + merged["disassembly_cost"]
        + merged["cleaning_cost_estimate_rmb"]
        + merged["component_inspection_cost_rmb"]
    )
    return merged.sort_values("core_id").reset_index(drop=True)


def _summarise_core_components(group: pd.DataFrame) -> pd.Series:
    weights = group["_critical_weight"].to_numpy(dtype=float)
    quality = group["quality_score"].to_numpy(dtype=float)
    life = group["residual_life_mean_h"].to_numpy(dtype=float)
    failure = group["failure_probability_prior"].to_numpy(dtype=float)
    return pd.Series(
        {
            "component_count": int(len(group)),
            "avg_quality_score": _safe_weighted_average(quality, weights),
            "avg_residual_life_h": _safe_weighted_average(life, weights),
            "avg_failure_probability": _safe_weighted_average(failure, weights),
            "component_inspection_cost_rmb": float(group["inspection_cost"].sum()),
            "component_replacement_cost_rmb": float(group["replacement_cost_rmb"].sum()),
            "missing_component_count": int(group["missing_indicator"].sum()),
            "out_of_tolerance_count": int(group["dimension_out_of_tolerance_flag"].sum()),
            "life_below_requirement_count": int(group["life_below_requirement_flag"].sum()),
        }
    )


def _build_core_route_table(
    core_summary: pd.DataFrame,
    components: pd.DataFrame,
    route_feasibility: pd.DataFrame,
    route_coefficients: pd.DataFrame,
) -> pd.DataFrame:
    feasible = route_feasibility[route_feasibility["feasible"] == 1].copy()
    feasible["route_class"] = feasible["route_id"].map(_route_class_for_route_id)
    feasible = feasible.dropna(subset=["route_class"])
    feasible_set = set(zip(feasible["component_type"], feasible["quality_state"], feasible["route_class"]))

    components = components.copy()
    components["_quality_state"] = components["observed_quality_state"].astype(str)
    rows: List[Dict[str, object]] = []
    coeff_lookup = {
        (row.component_type, row.quality_state, row.route_class): row
        for row in route_coefficients.itertuples(index=False)
    }
    resource_columns = [column for column in route_coefficients.columns if column.startswith("resource_h__")]

    for core in core_summary.itertuples(index=False):
        core_components = components[components["core_id"] == core.core_id]
        for route_class in ROUTE_CLASS_ORDER:
            if not _route_class_allowed(core_components, route_class, feasible_set):
                continue
            coeff = _aggregate_core_route_coefficients(core_components, route_class, coeff_lookup, resource_columns)
            if coeff is None:
                continue

            if route_class == "replace":
                route_cost = max(coeff["operation_cost_rmb"], 0.0) + float(core.component_replacement_cost_rmb)
                output_life = max(float(core.avg_residual_life_h), coeff["residual_life_gain_h"] + float(core.avg_residual_life_h), 9000.0)
            elif route_class == "scrap":
                route_cost = max(coeff["operation_cost_rmb"] - float(core.expected_salvage_value_rmb), 0.0)
                output_life = 0.0
            else:
                route_cost = coeff["operation_cost_rmb"]
                output_life = float(core.avg_residual_life_h) + coeff["residual_life_gain_h"]

            output_quality = max(
                min(float(core.avg_quality_score) + coeff["quality_gain"], ROUTE_CLASS_OUTPUT_QUALITY.get(route_class, 0.75)),
                0.0,
            )
            risk_penalty = max(0.0, float(core.avg_failure_probability) * (1.0 - min(output_quality, 0.98)))
            if route_class == "replace":
                risk_penalty *= 0.15
            if route_class == "scrap":
                risk_penalty = 0.0

            row = {
                "core_id": core.core_id,
                "route_class": route_class,
                "economic_cost_rmb": route_cost,
                "environmental_score": coeff["energy_kwh"] * 0.08
                + coeff["water_m3"] * 3.0
                + coeff["pollutant_kg"] * 18.0
                + coeff["carbon_kg"] * 0.25,
                "energy_kwh": coeff["energy_kwh"],
                "water_m3": coeff["water_m3"],
                "pollutant_kg": coeff["pollutant_kg"],
                "carbon_kg": coeff["carbon_kg"],
                "expected_output_quality": output_quality,
                "expected_residual_life_h": output_life,
                "risk_penalty": risk_penalty,
                "route_success_proxy": max(0.01, 1.0 - risk_penalty),
            }
            for resource_column, value in coeff["resource_h"].items():
                row[resource_column] = value
            rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No feasible core-route assignments were generated for Stage 2.")
    for resource_column in resource_columns:
        if resource_column not in table.columns:
            table[resource_column] = 0.0
    return table.fillna(0.0).sort_values(["core_id", "route_class"]).reset_index(drop=True)


def _aggregate_core_route_coefficients(
    core_components: pd.DataFrame,
    route_class: str,
    coeff_lookup: Dict[tuple, pd.Series],
    resource_columns: Iterable[str],
) -> Dict[str, object] | None:
    totals = {
        "processing_time_h": 0.0,
        "operation_cost_rmb": 0.0,
        "energy_kwh": 0.0,
        "water_m3": 0.0,
        "pollutant_kg": 0.0,
        "carbon_kg": 0.0,
        "quality_gain": 0.0,
        "residual_life_gain_h": 0.0,
    }
    resource_h = {column: 0.0 for column in resource_columns}
    matched = 0
    for component in core_components.itertuples(index=False):
        key = (component.component_type, component.observed_quality_state, route_class)
        coeff = coeff_lookup.get(key)
        if coeff is None:
            continue
        matched += 1
        for name in totals:
            totals[name] += float(getattr(coeff, name, 0.0))
        for resource_column in resource_columns:
            resource_h[resource_column] += float(getattr(coeff, resource_column, 0.0))
    if matched == 0:
        return None
    divisor = max(matched, 1)
    totals["quality_gain"] = totals["quality_gain"] / divisor
    totals["residual_life_gain_h"] = totals["residual_life_gain_h"] / divisor
    totals["resource_h"] = resource_h
    return totals


def _route_class_allowed(core_components: pd.DataFrame, route_class: str, feasible_set: set[tuple]) -> bool:
    if route_class == "scrap":
        return True
    present = 0
    feasible_count = 0
    for component in core_components.itertuples(index=False):
        present += 1
        if (component.component_type, component.observed_quality_state, route_class) in feasible_set:
            feasible_count += 1
    if present == 0:
        return False
    if route_class == "replace":
        return feasible_count > 0
    return feasible_count / present >= 0.35


def _capacity_by_resource(capacity_calendar: pd.DataFrame, config: Stage2Config, capacity_share: float) -> Dict[str, float]:
    calendar = _filter_periods(capacity_calendar.copy(), "period_id", config)
    calendar["effective_capacity_h"] = pd.to_numeric(calendar["effective_capacity_h"], errors="coerce").fillna(0.0)
    by_resource = calendar.groupby("resource_type")["effective_capacity_h"].sum().to_dict()
    return {resource: float(hours) * capacity_share for resource, hours in by_resource.items()}


def _route_class_for_route_id(route_id: str) -> str | None:
    for route_class, route_ids in ROUTE_CLASS_TO_ROUTE_IDS.items():
        if route_id in route_ids:
            return route_class
    return None


def _weighted_average(df: pd.DataFrame, column: str, default: float) -> float:
    if column not in df.columns or df.empty:
        return default
    values = pd.to_numeric(df[column], errors="coerce")
    weights = pd.to_numeric(df.get("quantity", pd.Series(1.0, index=df.index)), errors="coerce").fillna(1.0)
    valid = values.notna()
    if not valid.any() or weights[valid].sum() <= 0:
        return default
    return float(np.average(values[valid], weights=weights[valid]))


def _safe_weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights)
    if not valid.any() or weights[valid].sum() <= 0:
        return 0.0
    return float(np.average(values[valid], weights=weights[valid]))
