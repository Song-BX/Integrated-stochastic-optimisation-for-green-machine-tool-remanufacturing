"""Build Stage 8 instances from Stage 7."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from stage7_pareto.aggregation import build_stage7_instance
from stage7_pareto.config import Stage7Config

from .config import Stage8Config
from .structures import Stage8Instance


def build_stage8_instance(tables: Dict[str, pd.DataFrame], config: Stage8Config) -> Stage8Instance:
    """Build a Stage 7 instance and attach Stage 8 metadata."""

    base = build_stage7_instance(tables, stage7_config(config))
    payload = dict(base.__dict__)
    payload["heuristic_method"] = config.heuristic_method
    payload["pareto_mode"] = config.pareto_mode
    return Stage8Instance(**payload)


def stage7_config(config: Stage8Config) -> Stage7Config:
    """Map Stage 8 config fields to the Stage 7 builder/solver config."""

    return Stage7Config(
        raw_dir=config.raw_dir,
        stage1_report=config.stage1_report,
        processed_dir=config.processed_dir,
        results_dir=config.results_dir,
        stage4_results_dir=config.stage4_results_dir,
        stage5_results_dir=config.stage5_results_dir,
        stage6_results_dir=config.stage6_results_dir,
        machine_type_id=config.machine_type_id,
        period_start=config.period_start,
        period_count=config.period_count,
        processing_window_periods=config.processing_window_periods,
        scenario_mode=config.scenario_mode,
        scenario_ids=config.scenario_ids,
        epsilon_grid_size_env=config.epsilon_grid_size,
        epsilon_grid_size_assembly=config.epsilon_grid_size,
        payoff_cost_allowance=config.payoff_cost_allowance,
        augmentation_delta_factor=config.augmentation_delta_factor,
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
        dimension_penalty_weight=config.dimension_penalty_weight,
        assembly_quality_loss_weight=config.assembly_quality_loss_weight,
        life_gap_penalty_weight=config.life_gap_penalty_weight,
        compatibility_penalty_weight=config.compatibility_penalty_weight,
        soft_pair_penalty_rmb=config.soft_pair_penalty_rmb,
        assembly_shortfall_penalty_rmb=config.assembly_shortfall_penalty_rmb,
        assembly_risk_weight=config.assembly_risk_weight,
        risk_budget_reference_rmb=config.risk_budget_reference_rmb,
        time_limit_per_solve=config.repair_time_limit,
        mip_rel_gap=config.mip_rel_gap,
    )
