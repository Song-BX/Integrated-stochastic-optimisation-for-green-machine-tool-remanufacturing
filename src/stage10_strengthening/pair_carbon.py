"""Assembly pair-carbon objective audit for Stage 10."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from stage7_pareto.aggregation import build_stage7_instance
from stage7_pareto.config import Stage7Config
from stage7_pareto.model import assembly_pair_environmental_terms, build_model_data as build_stage7_model_data

from .config import Stage10Config
from .structures import PairCarbonSummary


def analyze_pair_carbon(tables: Dict[str, pd.DataFrame], config: Stage10Config) -> tuple[pd.DataFrame, pd.DataFrame, PairCarbonSummary, object]:
    """Build Stage 7 objective vectors and summarize pair-carbon coefficients."""

    stage7_config = _stage7_config(config, config.machine_types[0])
    instance = build_stage7_instance(tables, stage7_config)
    model_data = build_stage7_model_data(instance, stage7_config, tables)
    pair_terms = assembly_pair_environmental_terms(instance, model_data)
    environmental = np.asarray(model_data.objective_vectors["environmental_impact"], dtype=float)
    before_vector = environmental.copy()
    for row in pair_terms.itertuples(index=False):
        before_vector[int(row.variable_index)] -= float(row.weighted_coefficient)
    before_nonzero = int((np.abs(before_vector) > 1e-12).sum())
    after_nonzero = int((np.abs(environmental) > 1e-12).sum())
    pair_nonzero = int((pair_terms["weighted_coefficient"].abs() > 1e-12).sum()) if not pair_terms.empty else 0
    summary = PairCarbonSummary(
        machine_type_id=stage7_config.machine_type_id,
        period_start=stage7_config.period_start,
        period_count=stage7_config.period_count,
        pair_coefficient_count=int(len(pair_terms)),
        pair_nonzero_coefficient_count=pair_nonzero,
        environmental_nonzero_before=before_nonzero,
        environmental_nonzero_after=after_nonzero,
        total_weighted_pair_carbon=float(pair_terms["weighted_coefficient"].sum()) if not pair_terms.empty else 0.0,
        mean_pair_carbon_kg=float(pair_terms["pair_carbon_kg"].mean()) if not pair_terms.empty else 0.0,
        max_pair_carbon_kg=float(pair_terms["pair_carbon_kg"].max()) if not pair_terms.empty else 0.0,
        finite_objective_vector=bool(np.isfinite(environmental).all()),
    )
    breakdown = _environmental_breakdown(model_data, pair_terms, before_nonzero, after_nonzero)
    return pair_terms, breakdown, summary, model_data


def _stage7_config(config: Stage10Config, machine_type_id: str) -> Stage7Config:
    return Stage7Config(
        raw_dir=config.raw_dir,
        stage1_report=config.stage1_report,
        processed_dir=config.processed_dir,
        results_dir=config.results_dir,
        stage4_results_dir=config.stage4_results_dir,
        stage5_results_dir=config.stage5_results_dir,
        stage6_results_dir=config.stage6_results_dir,
        machine_type_id=machine_type_id,
        period_start="T0001",
        period_count=52,
        processing_window_periods=config.processing_window_periods,
        scenario_mode=config.scenario_mode,
        scenario_ids=config.scenario_ids,
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
        time_limit_per_solve=config.time_limit_seconds,
        mip_rel_gap=config.mip_rel_gap,
    )


def _environmental_breakdown(model_data: object, pair_terms: pd.DataFrame, before_nonzero: int, after_nonzero: int) -> pd.DataFrame:
    vector = np.asarray(model_data.objective_vectors["environmental_impact"], dtype=float)
    pair_indices = set(pair_terms["variable_index"].astype(int).tolist()) if not pair_terms.empty else set()
    route_indices = {
        model_data.variable_names.index(term.variable_name)
        for term in model_data.objective_terms.itertuples(index=False)
        if str(term.variable_name) in model_data.variable_names
    } if not model_data.objective_terms.empty else set()
    pre_indices = set(model_data.variable_groups.get("pre_procure", {}).values())
    recourse_indices = set(model_data.variable_groups.get("recourse_procure", {}).values())
    rows = [
        _breakdown_row("route_carbon", vector, route_indices),
        _breakdown_row("procurement_embedded_carbon", vector, pre_indices | recourse_indices),
        _breakdown_row("assembly_pair_carbon", vector, pair_indices),
    ]
    rows.append(
        {
            "component": "environmental_objective_total",
            "coefficient_count": int(len(vector)),
            "nonzero_count": int((np.abs(vector) > 1e-12).sum()),
            "coefficient_sum": float(vector.sum()),
            "coefficient_mean": float(vector.mean()) if len(vector) else 0.0,
            "coefficient_max": float(vector.max()) if len(vector) else 0.0,
            "nonzero_count_before_pair_carbon": before_nonzero,
            "nonzero_count_after_pair_carbon": after_nonzero,
        }
    )
    return pd.DataFrame(rows)


def _breakdown_row(component: str, vector: np.ndarray, indices: set[int]) -> Dict[str, object]:
    values = vector[list(indices)] if indices else np.array([], dtype=float)
    nonzero = np.abs(values) > 1e-12
    return {
        "component": component,
        "coefficient_count": int(len(values)),
        "nonzero_count": int(nonzero.sum()),
        "coefficient_sum": float(values.sum()) if len(values) else 0.0,
        "coefficient_mean": float(values.mean()) if len(values) else 0.0,
        "coefficient_max": float(values.max()) if len(values) else 0.0,
        "nonzero_count_before_pair_carbon": None,
        "nonzero_count_after_pair_carbon": None,
    }
