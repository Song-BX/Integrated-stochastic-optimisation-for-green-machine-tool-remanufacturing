"""Build Stage 6 selective-assembly instances."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd

from stage5_risk_averse.aggregation import build_stage5_instance
from stage5_risk_averse.config import Stage5Config

from .config import Stage6Config
from .structures import Stage6Instance


CANDIDATE_SCREENING_FLAGS = [
    "in_tolerance_flag",
    "life_requirement_flag",
    "reliability_requirement_flag",
    "quality_state_requirement_flag",
    "source_compatibility_flag",
    "crack_ok_flag",
    "quality_loss_requirement_flag",
    "eligibility_flag",
]

PAIR_HARD_FLAGS = [
    "life_gap_constraint_flag",
    "reliability_constraint_flag",
    "quality_gap_constraint_flag",
    "source_mix_constraint_flag",
    "risk_constraint_flag",
    "individual_candidate_screening_flag",
    "compatibility_flag",
]

NEW_SOURCE_TYPES = {"new", "new_replacement"}
OLD_SOURCE_TYPES = {"reused", "remanufactured"}


def build_stage6_instance(tables: Dict[str, pd.DataFrame], config: Stage6Config) -> Stage6Instance:
    """Build a Stage 5 instance and add selective-assembly pools."""

    base = build_stage5_instance(tables, _stage5_config(config))
    requirements = _assembly_requirements(tables["assembly_requirements"], config)
    candidates = _assembly_candidate_pool(tables["assembly_candidates"], requirements, base, config)
    pairs = _assembly_pair_pool(tables["assembly_compatibility"], requirements, candidates, config)
    if candidates.empty:
        raise ValueError("Stage 6 candidate pool is empty after eligibility and Stage 5 consistency filtering.")
    if pairs.empty:
        raise ValueError("Stage 6 pair pool is empty after compatibility filtering.")

    payload = dict(base.__dict__)
    payload.update(
        {
            "assembly_time_granularity": config.assembly_time_granularity,
            "pairwise_mode": config.pairwise_mode,
            "candidate_pool_mode": config.candidate_pool_mode,
            "assembly_requirements": requirements,
            "assembly_candidate_pool": candidates,
            "assembly_pair_pool": pairs,
            "assembly_pool_summary": _pool_summary(requirements, candidates, pairs),
        }
    )
    return Stage6Instance(**payload)


def _stage5_config(config: Stage6Config) -> Stage5Config:
    return Stage5Config(
        raw_dir=config.raw_dir,
        stage1_report=config.stage1_report,
        processed_dir=config.processed_dir,
        results_dir=config.results_dir,
        stage4_results_dir=config.stage4_results_dir,
        machine_type_id=config.machine_type_id,
        period_start=config.period_start,
        period_count=config.period_count,
        processing_window_periods=config.processing_window_periods,
        scenario_mode=config.scenario_mode,
        scenario_ids=config.scenario_ids,
        baseline_rule_id=config.baseline_rule_id,
        risk_baseline_rule_id=config.risk_baseline_rule_id,
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
        cvar_confidence=config.cvar_confidence,
        cvar_lambda=config.cvar_lambda,
        chance_alpha=config.chance_alpha,
        min_system_reliability=config.min_system_reliability,
        time_limit_seconds=config.time_limit_seconds,
        mip_rel_gap=config.mip_rel_gap,
    )


def _assembly_requirements(requirements: pd.DataFrame, config: Stage6Config) -> pd.DataFrame:
    data = requirements[requirements["machine_type_id"] == config.machine_type_id].copy()
    if data.empty:
        raise ValueError(f"No assembly requirements found for {config.machine_type_id}.")
    numeric_columns = [
        "required_component_count",
        "tolerance_lower_mm",
        "tolerance_upper_mm",
        "max_dimension_chain_error_mm",
        "max_pairwise_life_gap_h",
        "min_system_reliability",
        "max_quality_loss",
        "criticality_weight",
        "precision_weight",
        "reliability_weight",
        "cvar_risk_weight",
    ]
    _coerce_numeric(data, numeric_columns)
    data["required_component_type_list"] = data["required_component_types"].map(_split_semicolon)
    data["required_component_count"] = data["required_component_count"].fillna(
        data["required_component_type_list"].map(len)
    )
    data["pair_count_divisor"] = data["required_component_count"].map(lambda value: max(1, int(round(float(value))) - 1))
    return data.sort_values("assembly_requirement_id").reset_index(drop=True)


def _assembly_candidate_pool(
    candidates: pd.DataFrame,
    requirements: pd.DataFrame,
    base: object,
    config: Stage6Config,
) -> pd.DataFrame:
    requirement_ids = set(requirements["assembly_requirement_id"])
    data = candidates[
        (candidates["machine_type_id"] == config.machine_type_id)
        & (candidates["assembly_requirement_id"].isin(requirement_ids))
    ].copy()
    if data.empty:
        return data
    numeric_columns = [
        *CANDIDATE_SCREENING_FLAGS,
        "cvar_tail_risk_index",
        "residual_life_mean_h_after",
        "reliability_estimate_after",
        "failure_probability_after",
        "signed_dimension_chain_contribution_mm",
        "absolute_dimension_chain_contribution_mm",
        "max_dimension_chain_error_mm",
        "quality_loss_value",
        "quality_loss_coefficient",
        "max_quality_loss",
        "expected_processing_time_h",
        "expected_processing_cost_rmb",
        "expected_energy_kwh",
        "expected_water_m3",
        "expected_carbon_kg",
        "precision_score",
        "life_score",
        "reliability_score",
        "quality_score_component",
        "overall_candidate_score",
    ]
    _coerce_numeric(data, numeric_columns)
    source = data["candidate_source_type"].astype(str).str.lower()
    data["new_backup_candidate_flag"] = source.isin(NEW_SOURCE_TYPES).astype(int)
    route_pairs = set(
        zip(
            base.component_route_period_scenario_table["component_instance_id"].astype(str),
            base.component_route_period_scenario_table["route_id"].astype(str),
        )
    )
    data["stage5_route_candidate_exists"] = [
        int((str(component_id), str(route_id)) in route_pairs)
        for component_id, route_id in zip(data["component_instance_id"], data["planned_route_id"])
    ]
    data["stage5_consistent_candidate_flag"] = (
        (data["new_backup_candidate_flag"] == 1) | (data["stage5_route_candidate_exists"] == 1)
    ).astype(int)
    screened = data[
        (data[CANDIDATE_SCREENING_FLAGS].min(axis=1) >= 1.0)
        & (data["stage5_consistent_candidate_flag"] == 1)
    ].copy()
    screened["old_candidate_flag"] = screened["candidate_source_type"].astype(str).str.lower().isin(OLD_SOURCE_TYPES).astype(int)
    return screened.sort_values(["assembly_requirement_id", "component_type", "assembly_candidate_id"]).reset_index(drop=True)


def _assembly_pair_pool(
    compatibility: pd.DataFrame,
    requirements: pd.DataFrame,
    candidates: pd.DataFrame,
    config: Stage6Config,
) -> pd.DataFrame:
    requirement_ids = set(requirements["assembly_requirement_id"])
    candidate_ids = set(candidates["assembly_candidate_id"])
    data = compatibility[
        (compatibility["machine_type_id"] == config.machine_type_id)
        & (compatibility["assembly_requirement_id"].isin(requirement_ids))
        & (compatibility["candidate_i_id"].isin(candidate_ids))
        & (compatibility["candidate_j_id"].isin(candidate_ids))
    ].copy()
    if data.empty:
        return data
    numeric_columns = [
        "dimension_constraint_flag",
        "dimension_soft_constraint_flag",
        *PAIR_HARD_FLAGS,
        "pair_dimension_error_mm",
        "max_dimension_chain_error_mm",
        "pair_quality_loss",
        "max_quality_loss",
        "pairwise_life_gap_h",
        "max_pairwise_life_gap_h",
        "feature_pair_reliability",
        "min_system_reliability",
        "quality_score_gap",
        "max_pairwise_quality_score_gap",
        "pair_cvar_tail_risk_index",
        "compatibility_score",
        "priority_weight",
        "cvar_risk_weight",
        "combined_processing_cost_rmb",
        "combined_energy_kwh",
        "combined_water_m3",
        "combined_carbon_kg",
    ]
    _coerce_numeric(data, numeric_columns)
    dimension_pass = (data["dimension_constraint_flag"] >= 1.0) | (data["dimension_soft_constraint_flag"] >= 1.0)
    status_pass = data["compatibility_status"].isin(["hard_feasible", "soft_feasible_with_penalty"])
    screened = data[(dimension_pass) & (data[PAIR_HARD_FLAGS].min(axis=1) >= 1.0) & status_pass].copy()
    screened["soft_pair_flag"] = screened["compatibility_status"].eq("soft_feasible_with_penalty").astype(int)
    candidate_lookup = candidates.set_index("assembly_candidate_id")[
        ["component_type", "candidate_source_type", "component_instance_id", "planned_route_id"]
    ].to_dict(orient="index")
    screened["candidate_i_component_type"] = screened["candidate_i_id"].map(
        lambda value: candidate_lookup.get(value, {}).get("component_type")
    )
    screened["candidate_j_component_type"] = screened["candidate_j_id"].map(
        lambda value: candidate_lookup.get(value, {}).get("component_type")
    )
    primary = requirements.set_index("assembly_requirement_id")["primary_selective_component_type"].to_dict()
    screened["touches_primary_component_flag"] = [
        int(row.candidate_i_component_type == primary.get(row.assembly_requirement_id) or row.candidate_j_component_type == primary.get(row.assembly_requirement_id))
        for row in screened.itertuples(index=False)
    ]
    divisor_by_requirement = requirements.set_index("assembly_requirement_id")["pair_count_divisor"].to_dict()
    screened["pair_count_divisor"] = screened["assembly_requirement_id"].map(divisor_by_requirement)
    screened = screened[
        (screened["pair_count_divisor"].astype(float) <= 1.0)
        | (screened["touches_primary_component_flag"].astype(int) == 1)
    ].copy()
    return screened.sort_values(["assembly_requirement_id", "compatibility_id"]).reset_index(drop=True)


def _pool_summary(requirements: pd.DataFrame, candidates: pd.DataFrame, pairs: pd.DataFrame) -> Dict[str, object]:
    by_requirement = {}
    for requirement_id in requirements["assembly_requirement_id"]:
        req_candidates = candidates[candidates["assembly_requirement_id"] == requirement_id]
        req_pairs = pairs[pairs["assembly_requirement_id"] == requirement_id]
        by_requirement[requirement_id] = {
            "candidate_count": int(len(req_candidates)),
            "pair_count": int(len(req_pairs)),
            "hard_pair_count": int((req_pairs["soft_pair_flag"] == 0).sum()) if "soft_pair_flag" in req_pairs else 0,
            "soft_pair_count": int((req_pairs["soft_pair_flag"] == 1).sum()) if "soft_pair_flag" in req_pairs else 0,
        }
    return {
        "requirement_count": int(len(requirements)),
        "candidate_count": int(len(candidates)),
        "pair_count": int(len(pairs)),
        "new_backup_candidate_count": int(candidates.get("new_backup_candidate_flag", pd.Series(dtype=int)).sum()),
        "old_stage5_candidate_count": int(candidates.get("old_candidate_flag", pd.Series(dtype=int)).sum()),
        "hard_pair_count": int((pairs["soft_pair_flag"] == 0).sum()) if "soft_pair_flag" in pairs else 0,
        "soft_pair_count": int((pairs["soft_pair_flag"] == 1).sum()) if "soft_pair_flag" in pairs else 0,
        "by_requirement": by_requirement,
    }


def _coerce_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _split_semicolon(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [token.strip() for token in str(value).split(";") if token.strip()]
