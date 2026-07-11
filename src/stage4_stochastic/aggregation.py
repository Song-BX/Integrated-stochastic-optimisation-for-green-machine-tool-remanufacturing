"""Build Stage 4 stochastic SAA instances."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from stage3_multiperiod.aggregation import PRODUCTIVE_ROUTES, build_stage3_instance

from .config import Stage4Config
from .structures import Stage4Instance


def build_stage4_instance(tables: Dict[str, pd.DataFrame], config: Stage4Config) -> Stage4Instance:
    """Build the SAA deterministic-equivalent input instance."""

    base = build_stage3_instance(tables, config)
    scenario_sample = _scenario_sample(tables["scenarios"], config)
    scenario_demand = _scenario_demand(tables["demand_scenarios"], scenario_sample, base.periods, config)
    scenario_capacity = _scenario_capacity_table(tables["capacity_calendar"], scenario_sample, base, config)
    route_table = _component_route_period_scenario_table(base, tables, scenario_sample, config)

    route_ids = sorted(route_table["route_id"].unique())
    resource_types = sorted(
        set(scenario_capacity["resource_type"].unique())
        | {column.removeprefix("resource_h__") for column in route_table.columns if column.startswith("resource_h__")}
    )
    expected_demand = float((scenario_demand["demand_units"] * scenario_demand["saa_probability"]).sum())

    return Stage4Instance(
        machine_type_id=base.machine_type_id,
        machine_family=base.machine_family,
        periods=base.periods,
        scenario_ids=scenario_sample["scenario_id"].tolist(),
        component_types=base.component_types,
        route_ids=route_ids,
        resource_types=resource_types,
        demand_units_expected=expected_demand,
        candidate_core_count=base.candidate_core_count,
        component_instance_count=base.component_instance_count,
        min_required_life_h=base.min_required_life_h,
        target_quality_score=base.target_quality_score,
        machine_summary=base.machine_summary,
        scenario_sample=scenario_sample,
        scenario_probability_summary=_scenario_probability_summary(scenario_sample),
        scenario_demand=scenario_demand,
        bom_requirements=base.bom_requirements,
        core_summary=base.core_summary,
        component_summary=base.component_summary,
        component_route_period_scenario_table=route_table,
        initial_inventory=base.initial_inventory,
        procurement_costs=base.procurement_costs,
        capacity_table=base.capacity_table,
        scenario_capacity_table=scenario_capacity,
    )


def _scenario_sample(scenarios: pd.DataFrame, config: Stage4Config) -> pd.DataFrame:
    data = scenarios.copy()
    numeric_cols = [
        "scenario_probability",
        "demand_multiplier",
        "capacity_availability_multiplier",
        "bottleneck_capacity_multiplier",
        "procurement_price_multiplier",
        "environmental_cost_multiplier",
        "warranty_failure_multiplier",
    ]
    for column in numeric_cols:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce").fillna(1.0)
    selected_ids = list(config.scenario_ids)
    selected = data[data["scenario_id"].isin(selected_ids)].copy()
    missing = sorted(set(selected_ids) - set(selected["scenario_id"]))
    if missing:
        raise ValueError(f"Stage 4 scenario sample is missing scenarios: {', '.join(missing)}")
    selected["scenario_order"] = selected["scenario_id"].map({scenario_id: index for index, scenario_id in enumerate(selected_ids)})
    selected = selected.sort_values("scenario_order").reset_index(drop=True)
    total_probability = float(selected["scenario_probability"].sum())
    if total_probability <= 0:
        raise ValueError("Selected Stage 4 scenarios have non-positive total probability.")
    selected["saa_probability"] = selected["scenario_probability"] / total_probability
    selected["selection_mode"] = config.scenario_mode
    selected["selection_reason"] = "highest-probability macro-group representative"
    keep_cols = [
        "scenario_id",
        "scenario_name",
        "macro_group",
        "quality_process_scenario",
        "environmental_scenario",
        "reliability_scenario",
        "scenario_probability",
        "saa_probability",
        "scenario_stress_index",
        "scenario_stress_level",
        "demand_multiplier",
        "capacity_availability_multiplier",
        "bottleneck_capacity_multiplier",
        "procurement_price_multiplier",
        "environmental_cost_multiplier",
        "warranty_failure_multiplier",
        "selection_mode",
        "selection_reason",
    ]
    return selected[[column for column in keep_cols if column in selected.columns]].copy()


def _scenario_probability_summary(sample: pd.DataFrame) -> Dict[str, object]:
    return {
        "scenario_count": int(sample["scenario_id"].nunique()),
        "raw_probability_sum": float(sample["scenario_probability"].sum()),
        "saa_probability_sum": float(sample["saa_probability"].sum()),
        "scenario_ids": sample["scenario_id"].tolist(),
    }


def _scenario_demand(
    demand_scenarios: pd.DataFrame,
    scenario_sample: pd.DataFrame,
    periods: List[str],
    config: Stage4Config,
) -> pd.DataFrame:
    data = demand_scenarios[
        (demand_scenarios["machine_type_id"] == config.machine_type_id)
        & (demand_scenarios["scenario_id"].isin(set(scenario_sample["scenario_id"])))
        & (demand_scenarios["period_id"].isin(set(periods)))
    ].copy()
    for column in ["scenario_realized_due_quantity", "scenario_expected_due_quantity"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["raw_demand"] = data["scenario_realized_due_quantity"].where(
        data["scenario_realized_due_quantity"].notna(), data["scenario_expected_due_quantity"]
    )
    grouped = data.groupby(["scenario_id", "period_id"], dropna=False)["raw_demand"].sum().to_dict()
    probability = scenario_sample.set_index("scenario_id")["saa_probability"].to_dict()
    rows = []
    for scenario_id in scenario_sample["scenario_id"]:
        for period_index, period in enumerate(periods):
            value = float(grouped.get((scenario_id, period), 0.0))
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "saa_probability": float(probability[scenario_id]),
                    "period_id": period,
                    "period_index_stage4": period_index,
                    "demand_units": int(round(value)),
                    "raw_demand_units": value,
                }
            )
    return pd.DataFrame(rows)


def _scenario_capacity_table(
    capacity_calendar: pd.DataFrame,
    scenario_sample: pd.DataFrame,
    base: object,
    config: Stage4Config,
) -> pd.DataFrame:
    capacity = capacity_calendar[capacity_calendar["period_id"].isin(base.periods)].copy()
    capacity["effective_capacity_h"] = pd.to_numeric(capacity["effective_capacity_h"], errors="coerce").fillna(0.0)
    capacity["bottleneck_flag_period"] = pd.to_numeric(capacity["bottleneck_flag_period"], errors="coerce").fillna(0.0)
    grouped = (
        capacity.groupby(["period_id", "resource_type"], dropna=False)
        .agg(effective_capacity_h=("effective_capacity_h", "sum"), bottleneck_flag=("bottleneck_flag_period", "max"))
        .reset_index()
    )
    share = float(base.machine_summary.get("capacity_share", 1.0))
    grouped["available_regular_hours_base"] = grouped["effective_capacity_h"] * share
    scenario_lookup = scenario_sample.set_index("scenario_id").to_dict(orient="index")
    rows = []
    for scenario_id in scenario_sample["scenario_id"]:
        multipliers = scenario_lookup[scenario_id]
        availability = float(multipliers.get("capacity_availability_multiplier", 1.0))
        bottleneck = float(multipliers.get("bottleneck_capacity_multiplier", 1.0))
        for capacity_row in grouped.itertuples(index=False):
            available = float(capacity_row.available_regular_hours_base) * availability
            if float(capacity_row.bottleneck_flag) > 0.5:
                available *= bottleneck
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "saa_probability": float(multipliers.get("saa_probability", 0.0)),
                    "period_id": capacity_row.period_id,
                    "resource_type": capacity_row.resource_type,
                    "available_regular_hours_base": float(capacity_row.available_regular_hours_base),
                    "available_regular_hours": max(0.0, available),
                    "bottleneck_flag": int(float(capacity_row.bottleneck_flag) > 0.5),
                }
            )
    return pd.DataFrame(rows).sort_values(["scenario_id", "period_id", "resource_type"]).reset_index(drop=True)


def _component_route_period_scenario_table(
    base: object,
    tables: Dict[str, pd.DataFrame],
    scenario_sample: pd.DataFrame,
    config: Stage4Config,
) -> pd.DataFrame:
    period_rank = {period: index for index, period in enumerate(base.periods)}
    quality_lookup = _component_quality_lookup(tables["component_quality_scenarios"], scenario_sample, config)
    outcome_lookup = _route_outcome_lookup(tables["route_outcome_scenarios"], scenario_sample, config)
    resource_lookup, resource_cols = _route_resource_lookup(tables["processing_parameters"])
    saa_probability = scenario_sample.set_index("scenario_id")["saa_probability"].to_dict()
    scenario_meta = scenario_sample.set_index("scenario_id").to_dict(orient="index")
    rows: List[Dict[str, object]] = []

    for scenario_id in scenario_sample["scenario_id"]:
        for component in base.component_summary.itertuples(index=False):
            first_period = _first_available_period(component.inspection_period, base.periods, period_rank)
            if first_period is None:
                continue
            last_period = min(len(base.periods), first_period + config.processing_window_periods)
            quality = quality_lookup.get((scenario_id, component.component_type), {})
            scenario_state = str(quality.get("expected_quality_state", component.observed_quality_state))
            quality_score = _clip01(float(component.quality_score) + float(quality.get("quality_score_delta", 0.0)))
            residual_life = max(0.0, float(component.residual_life_mean_h) + float(quality.get("residual_life_delta_h", 0.0)))
            failure_probability = _clip01(
                float(component.failure_probability_prior)
                + float(quality.get("failure_probability_delta", 0.0))
            )
            outcome_rows = outcome_lookup.get((scenario_id, component.component_type, scenario_state))
            if outcome_rows is None:
                outcome_rows = outcome_lookup.get((scenario_id, component.component_type, component.observed_quality_state))
            if not outcome_rows:
                continue
            base_feasible = set(str(component.feasible_route_ids).split(";"))
            for outcome in outcome_rows:
                route_id = str(outcome["route_id"])
                if route_id not in base_feasible:
                    continue
                resource_values = _scaled_resource_values(
                    resource_lookup,
                    resource_cols,
                    component.component_type,
                    scenario_state,
                    component.observed_quality_state,
                    route_id,
                    float(outcome.get("expected_total_route_time_h", 0.0)),
                )
                output_quality = _clip01(float(outcome.get("expected_quality_score_after", quality_score)))
                output_life = max(0.0, float(outcome.get("residual_life_after_mean_h", residual_life)))
                risk_penalty = _clip01(float(outcome.get("posterior_warranty_failure_probability", failure_probability)))
                economic_cost = float(outcome.get("expected_total_route_cost_rmb", 0.0))
                environmental_score = float(outcome.get("expected_environmental_impact_score", 0.0))
                warranty_cost = float(outcome.get("expected_failure_cost_rmb", 0.0)) + float(outcome.get("warranty_reserve_cost_rmb", 0.0))
                for period in base.periods[first_period:last_period]:
                    row = {
                        "scenario_id": scenario_id,
                        "saa_probability": float(saa_probability[scenario_id]),
                        "macro_group": scenario_meta[scenario_id].get("macro_group"),
                        "component_instance_id": component.component_instance_id,
                        "core_id": component.core_id,
                        "component_type": component.component_type,
                        "quality_state_baseline": component.observed_quality_state,
                        "quality_state_before_scenario": scenario_state,
                        "inspection_period": component.inspection_period,
                        "period_id": period,
                        "period_offset_from_inspection": int(period_rank[period] - first_period),
                        "route_id": route_id,
                        "route_category": outcome.get("route_category"),
                        "scenario_quality_score": quality_score,
                        "scenario_residual_life_h": residual_life,
                        "scenario_failure_probability": failure_probability,
                        "economic_cost_rmb": economic_cost,
                        "environmental_score": environmental_score,
                        "expected_output_quality": output_quality,
                        "expected_residual_life_h": output_life,
                        "risk_penalty": risk_penalty,
                        "warranty_risk_cost_rmb": warranty_cost,
                        "productive_output": 1 if route_id in PRODUCTIVE_ROUTES else 0,
                    }
                    row.update(resource_values)
                    rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No feasible component-route-period-scenario rows generated for Stage 4.")
    return table.sort_values(["scenario_id", "component_instance_id", "period_id", "route_id"]).reset_index(drop=True)


def _component_quality_lookup(
    component_quality_scenarios: pd.DataFrame,
    scenario_sample: pd.DataFrame,
    config: Stage4Config,
) -> Dict[tuple, Dict[str, object]]:
    data = component_quality_scenarios[
        (component_quality_scenarios["machine_type_id"] == config.machine_type_id)
        & (component_quality_scenarios["scenario_id"].isin(set(scenario_sample["scenario_id"])))
    ].copy()
    numeric_cols = [
        "quality_score_delta",
        "residual_life_delta_h",
        "failure_probability_mean_scenario",
        "failure_probability_baseline_mean",
        "out_of_tolerance_probability_scenario",
        "chance_quality_violation_probability",
        "cvar_quality_tail_index",
    ]
    for column in numeric_cols:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.0)
    data["failure_probability_delta"] = data["failure_probability_mean_scenario"] - data["failure_probability_baseline_mean"]
    lookup: Dict[tuple, Dict[str, object]] = {}
    for row in data.itertuples(index=False):
        lookup[(row.scenario_id, row.component_type)] = {
            "expected_quality_state": row.expected_quality_state,
            "quality_score_delta": float(row.quality_score_delta),
            "residual_life_delta_h": float(row.residual_life_delta_h),
            "failure_probability_delta": float(row.failure_probability_delta),
        }
    return lookup


def _route_outcome_lookup(
    route_outcome_scenarios: pd.DataFrame,
    scenario_sample: pd.DataFrame,
    config: Stage4Config,
) -> Dict[tuple, List[Dict[str, object]]]:
    data = route_outcome_scenarios[
        (route_outcome_scenarios["machine_type_id"] == config.machine_type_id)
        & (route_outcome_scenarios["scenario_id"].isin(set(scenario_sample["scenario_id"])))
    ].copy()
    data["route_feasible_under_scenario_flag"] = pd.to_numeric(data["route_feasible_under_scenario_flag"], errors="coerce").fillna(0)
    data = data[data["route_feasible_under_scenario_flag"] == 1].copy()
    numeric_cols = [
        "expected_quality_score_after",
        "residual_life_after_mean_h",
        "posterior_warranty_failure_probability",
        "expected_total_route_time_h",
        "expected_total_route_cost_rmb",
        "expected_environmental_impact_score",
        "expected_failure_cost_rmb",
        "warranty_reserve_cost_rmb",
    ]
    for column in numeric_cols:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.0)
    data["route_id"] = data["evaluated_route_id"]
    lookup: Dict[tuple, List[Dict[str, object]]] = {}
    keep_cols = [
        "scenario_id",
        "component_type",
        "quality_state_before_scenario",
        "route_id",
        "route_category",
        *numeric_cols,
    ]
    for row in data[keep_cols].itertuples(index=False):
        key = (row.scenario_id, row.component_type, row.quality_state_before_scenario)
        lookup.setdefault(key, []).append(row._asdict())
    return lookup


def _route_resource_lookup(processing_parameters: pd.DataFrame) -> tuple[Dict[tuple, Dict[str, float]], List[str]]:
    data = processing_parameters.copy()
    data["base_processing_time_h"] = pd.to_numeric(data["base_processing_time_h"], errors="coerce").fillna(0.0)
    grouped = (
        data.groupby(["component_type", "quality_state", "route_id", "required_resource_type"], dropna=False)
        .agg(resource_hours=("base_processing_time_h", "sum"))
        .reset_index()
    )
    pivot = grouped.pivot_table(
        index=["component_type", "quality_state", "route_id"],
        columns="required_resource_type",
        values="resource_hours",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    pivot.columns = [
        f"resource_h__{column}" if column not in {"component_type", "quality_state", "route_id"} else column
        for column in pivot.columns
    ]
    resource_cols = [column for column in pivot.columns if column.startswith("resource_h__")]
    lookup: Dict[tuple, Dict[str, float]] = {}
    for row in pivot.itertuples(index=False):
        values = {column: float(getattr(row, column, 0.0)) for column in resource_cols}
        lookup[(row.component_type, row.quality_state, row.route_id)] = values
    return lookup, resource_cols


def _scaled_resource_values(
    resource_lookup: Dict[tuple, Dict[str, float]],
    resource_cols: List[str],
    component_type: str,
    scenario_state: str,
    baseline_state: str,
    route_id: str,
    scenario_total_time: float,
) -> Dict[str, float]:
    resources = resource_lookup.get((component_type, scenario_state, route_id))
    if resources is None:
        resources = resource_lookup.get((component_type, baseline_state, route_id), {})
    base_total = sum(float(resources.get(column, 0.0)) for column in resource_cols)
    scale = scenario_total_time / base_total if base_total > 1e-9 and scenario_total_time > 0 else 1.0
    return {column: float(resources.get(column, 0.0)) * scale for column in resource_cols}


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


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
