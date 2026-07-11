"""Build Stage 5 risk-averse SAA instances."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from stage4_stochastic.aggregation import build_stage4_instance
from stage4_stochastic.config import Stage4Config

from .config import Stage5Config
from .structures import Stage5Instance


def build_stage5_instance(tables: Dict[str, pd.DataFrame], config: Stage5Config) -> Stage5Instance:
    """Build a Stage 4 SAA instance and add chance/CVaR data."""

    base = build_stage4_instance(tables, _stage4_config(config))
    min_reliability = _min_system_reliability(tables["machine_types"], config)
    reliability = _component_route_reliability(tables["reliability_parameters"], base.scenario_sample, min_reliability)
    risk = _component_route_risk(tables["risk_parameters"], base.scenario_sample)
    augmented = _augment_route_table(base.component_route_period_scenario_table, base.scenario_sample, reliability, risk)
    chance_report = _chance_constraint_report(augmented)
    filtered = augmented[
        (augmented["chance_constraint_satisfied_flag"].astype(float) >= 1.0)
        & (augmented["survival_probability_at_min_system_life"].astype(float) >= min_reliability)
    ].copy()
    if filtered.empty:
        raise ValueError(
            "Stage 5 chance constraints removed all component-route-period-scenario rows. "
            "Check reliability_parameters.csv or lower the reliability threshold for diagnostics."
        )

    route_ids = sorted(filtered["route_id"].unique())
    component_route_reliability = reliability.sort_values(
        ["scenario_id", "component_type", "quality_state_before", "route_id"]
    ).reset_index(drop=True)
    component_route_risk = risk.sort_values(["scenario_id", "component_type", "quality_state_before", "route_id"]).reset_index(drop=True)

    return Stage5Instance(
        machine_type_id=base.machine_type_id,
        machine_family=base.machine_family,
        periods=base.periods,
        scenario_ids=base.scenario_ids,
        component_types=base.component_types,
        route_ids=route_ids,
        resource_types=base.resource_types,
        demand_units_expected=base.demand_units_expected,
        candidate_core_count=base.candidate_core_count,
        component_instance_count=base.component_instance_count,
        min_required_life_h=base.min_required_life_h,
        target_quality_score=base.target_quality_score,
        machine_summary=base.machine_summary,
        scenario_sample=base.scenario_sample,
        scenario_probability_summary=base.scenario_probability_summary,
        scenario_demand=base.scenario_demand,
        bom_requirements=base.bom_requirements,
        core_summary=base.core_summary,
        component_summary=base.component_summary,
        component_route_period_scenario_table=filtered.sort_values(
            ["scenario_id", "component_instance_id", "period_id", "route_id"]
        ).reset_index(drop=True),
        initial_inventory=base.initial_inventory,
        procurement_costs=base.procurement_costs,
        capacity_table=base.capacity_table,
        scenario_capacity_table=base.scenario_capacity_table,
        min_system_reliability=min_reliability,
        chance_alpha=float(config.chance_alpha),
        stage4_route_candidate_count=int(len(base.component_route_period_scenario_table)),
        component_route_reliability=component_route_reliability,
        component_route_risk=component_route_risk,
        chance_constraint_report=chance_report,
    )


def _stage4_config(config: Stage5Config) -> Stage4Config:
    return Stage4Config(
        raw_dir=config.raw_dir,
        stage1_report=config.stage1_report,
        processed_dir=config.processed_dir,
        results_dir=config.results_dir,
        machine_type_id=config.machine_type_id,
        period_start=config.period_start,
        period_count=config.period_count,
        processing_window_periods=config.processing_window_periods,
        scenario_mode=config.scenario_mode,
        scenario_ids=config.scenario_ids,
        baseline_rule_id=config.baseline_rule_id,
        env_weight=config.env_weight,
        quality_weight=config.quality_weight,
        reliability_weight=config.reliability_weight,
        inventory_holding_rate=config.inventory_holding_rate,
        backlog_penalty_rmb_per_unit_period=config.backlog_penalty_rmb_per_unit_period,
        procurement_cost_multiplier=config.procurement_cost_multiplier,
        recourse_procurement_premium=config.recourse_procurement_premium,
        overtime_penalty_rmb_per_h=config.overtime_penalty_rmb_per_h,
        capacity_share_floor=config.capacity_share_floor,
        capacity_share_multiplier=config.capacity_share_multiplier,
        quality_floor=config.quality_floor,
        time_limit_seconds=config.time_limit_seconds,
        mip_rel_gap=config.mip_rel_gap,
    )


def _min_system_reliability(machine_types: pd.DataFrame, config: Stage5Config) -> float:
    if config.min_system_reliability is not None:
        return float(config.min_system_reliability)
    matches = machine_types[machine_types["machine_type_id"] == config.machine_type_id]
    if matches.empty:
        raise ValueError(f"Unknown machine_type_id for reliability threshold: {config.machine_type_id}")
    value = pd.to_numeric(matches.iloc[0].get("min_system_reliability"), errors="coerce")
    if pd.isna(value):
        raise ValueError(f"machine_types.csv has no min_system_reliability for {config.machine_type_id}.")
    return float(value)


def _component_route_reliability(
    reliability_parameters: pd.DataFrame,
    scenario_sample: pd.DataFrame,
    min_reliability: float,
) -> pd.DataFrame:
    data = reliability_parameters.copy()
    numeric_cols = [
        "scenario_probability",
        "expected_quality_score_after",
        "survival_probability_at_min_system_life",
        "survival_probability_at_warranty_life",
        "posterior_warranty_failure_probability",
        "chance_constraint_alpha",
        "min_system_reliability_reference",
        "chance_constraint_satisfied_flag",
        "reliability_risk_index",
        "cvar_reliability_risk_weight",
        "warranty_reserve_cost_rmb",
        "expected_failure_cost_rmb",
    ]
    for column in numeric_cols:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.0)
    scenario_map = scenario_sample[["scenario_id", "reliability_scenario"]].rename(columns={"reliability_scenario": "scenario_label"})
    data = data.merge(scenario_map, on="scenario_label", how="inner")
    keep_cols = [
        "scenario_id",
        "scenario_label",
        "component_type",
        "quality_state_before",
        "route_id",
        "expected_quality_score_after",
        "survival_probability_at_min_system_life",
        "survival_probability_at_warranty_life",
        "posterior_warranty_failure_probability",
        "chance_constraint_alpha",
        "min_system_reliability_reference",
        "chance_constraint_satisfied_flag",
        "reliability_risk_index",
        "cvar_reliability_risk_weight",
        "warranty_reserve_cost_rmb",
        "expected_failure_cost_rmb",
    ]
    data = data[[column for column in keep_cols if column in data.columns]].copy()
    data["min_system_reliability_threshold"] = float(min_reliability)
    data["stage5_chance_pass"] = (
        (data["chance_constraint_satisfied_flag"].astype(float) >= 1.0)
        & (data["survival_probability_at_min_system_life"].astype(float) >= float(min_reliability))
    )
    return data


def _component_route_risk(risk_parameters: pd.DataFrame, scenario_sample: pd.DataFrame) -> pd.DataFrame:
    data = risk_parameters[risk_parameters["risk_scope"] == "component_quality_route_scenario"].copy()
    numeric_cols = [
        "risk_probability",
        "risk_severity_score",
        "risk_index",
        "impact_cost_rmb_mean",
        "var95_loss_rmb",
        "cvar95_loss_rmb",
        "cvar_model_weight",
        "residual_cvar95_loss_rmb",
    ]
    for column in numeric_cols:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.0)
    scenario_map = scenario_sample[["scenario_id", "reliability_scenario"]].rename(columns={"reliability_scenario": "scenario_label"})
    data = data.merge(scenario_map, on="scenario_label", how="inner")
    data["weighted_residual_cvar95_loss_rmb"] = data["residual_cvar95_loss_rmb"] * data["cvar_model_weight"]
    data["weighted_cvar95_loss_rmb"] = data["cvar95_loss_rmb"] * data["cvar_model_weight"]
    data["weighted_risk_index"] = data["risk_index"] * data["cvar_model_weight"]
    grouped = (
        data.groupby(["scenario_id", "scenario_label", "component_type", "quality_state_before", "route_id"], dropna=False)
        .agg(
            risk_row_count=("risk_param_id", "count"),
            cvar_model_weight_sum=("cvar_model_weight", "sum"),
            residual_cvar95_weighted_sum=("weighted_residual_cvar95_loss_rmb", "sum"),
            cvar95_weighted_sum=("weighted_cvar95_loss_rmb", "sum"),
            risk_index_weighted_sum=("weighted_risk_index", "sum"),
            risk_index_max=("risk_index", "max"),
            impact_cost_mean_sum=("impact_cost_rmb_mean", "sum"),
            var95_loss_max=("var95_loss_rmb", "max"),
        )
        .reset_index()
    )
    weight = grouped["cvar_model_weight_sum"].replace(0.0, np.nan)
    grouped["route_tail_loss_rmb"] = (grouped["residual_cvar95_weighted_sum"] / weight).fillna(0.0)
    grouped["route_cvar95_loss_rmb"] = (grouped["cvar95_weighted_sum"] / weight).fillna(0.0)
    grouped["route_risk_index"] = (grouped["risk_index_weighted_sum"] / weight).fillna(0.0)
    return grouped


def _augment_route_table(
    route_table: pd.DataFrame,
    scenario_sample: pd.DataFrame,
    reliability: pd.DataFrame,
    risk: pd.DataFrame,
) -> pd.DataFrame:
    scenario_meta = scenario_sample[["scenario_id", "reliability_scenario"]].copy()
    augmented = route_table.merge(scenario_meta, on="scenario_id", how="left")
    reliability_cols = [
        "scenario_id",
        "scenario_label",
        "component_type",
        "quality_state_before",
        "route_id",
        "survival_probability_at_min_system_life",
        "survival_probability_at_warranty_life",
        "posterior_warranty_failure_probability",
        "chance_constraint_alpha",
        "min_system_reliability_reference",
        "chance_constraint_satisfied_flag",
        "reliability_risk_index",
        "cvar_reliability_risk_weight",
        "stage5_chance_pass",
    ]
    augmented = augmented.merge(
        reliability[[column for column in reliability_cols if column in reliability.columns]],
        left_on=["scenario_id", "reliability_scenario", "component_type", "quality_state_before_scenario", "route_id"],
        right_on=["scenario_id", "scenario_label", "component_type", "quality_state_before", "route_id"],
        how="left",
    )
    risk_cols = [
        "scenario_id",
        "scenario_label",
        "component_type",
        "quality_state_before",
        "route_id",
        "risk_row_count",
        "cvar_model_weight_sum",
        "route_tail_loss_rmb",
        "route_cvar95_loss_rmb",
        "route_risk_index",
        "risk_index_max",
        "impact_cost_mean_sum",
        "var95_loss_max",
    ]
    augmented = augmented.merge(
        risk[[column for column in risk_cols if column in risk.columns]],
        left_on=["scenario_id", "reliability_scenario", "component_type", "quality_state_before_scenario", "route_id"],
        right_on=["scenario_id", "scenario_label", "component_type", "quality_state_before", "route_id"],
        how="left",
        suffixes=("", "_risk"),
    )
    numeric_defaults = {
        "survival_probability_at_min_system_life": 0.0,
        "survival_probability_at_warranty_life": 0.0,
        "posterior_warranty_failure_probability": 1.0,
        "chance_constraint_alpha": 0.0,
        "min_system_reliability_reference": 0.0,
        "chance_constraint_satisfied_flag": 0.0,
        "reliability_risk_index": 1.0,
        "cvar_reliability_risk_weight": 0.0,
        "risk_row_count": 0.0,
        "cvar_model_weight_sum": 0.0,
        "route_tail_loss_rmb": 0.0,
        "route_cvar95_loss_rmb": 0.0,
        "route_risk_index": 0.0,
        "risk_index_max": 0.0,
        "impact_cost_mean_sum": 0.0,
        "var95_loss_max": 0.0,
    }
    for column, default in numeric_defaults.items():
        if column in augmented.columns:
            augmented[column] = pd.to_numeric(augmented[column], errors="coerce").fillna(default)
    augmented["stage5_chance_pass"] = augmented.get("stage5_chance_pass", False).fillna(False).astype(bool)
    augmented["stage5_reliability_joined"] = augmented["chance_constraint_alpha"].astype(float) > 0.0
    augmented["stage5_risk_joined"] = augmented["risk_row_count"].astype(float) > 0.0
    for column in ["scenario_label", "quality_state_before", "scenario_label_risk", "quality_state_before_risk"]:
        if column in augmented.columns:
            augmented = augmented.drop(columns=[column])
    return augmented


def _chance_constraint_report(augmented: pd.DataFrame) -> pd.DataFrame:
    if augmented.empty:
        return pd.DataFrame()
    data = augmented.copy()
    data["chance_pass"] = (
        data["stage5_chance_pass"].astype(bool)
        & data["stage5_reliability_joined"].astype(bool)
        & data["stage5_risk_joined"].astype(bool)
    )
    rows = (
        data.groupby(["scenario_id", "reliability_scenario", "component_type", "route_id"], dropna=False)
        .agg(
            candidate_rows=("component_instance_id", "count"),
            reliability_joined=("stage5_reliability_joined", "sum"),
            risk_joined=("stage5_risk_joined", "sum"),
            chance_pass_rows=("chance_pass", "sum"),
            min_survival_probability=("survival_probability_at_min_system_life", "min"),
            max_survival_probability=("survival_probability_at_min_system_life", "max"),
            mean_route_tail_loss_rmb=("route_tail_loss_rmb", "mean"),
        )
        .reset_index()
    )
    rows["excluded_rows"] = rows["candidate_rows"] - rows["chance_pass_rows"]
    return rows.sort_values(["scenario_id", "component_type", "route_id"]).reset_index(drop=True)

