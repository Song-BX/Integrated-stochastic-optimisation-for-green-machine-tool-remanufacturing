"""Deterministic rule baseline evaluation for Stage 2."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from .aggregation import ROUTE_CLASS_ORDER
from .config import Stage2Config
from .structures import Stage2Instance


def evaluate_baseline(instance: Stage2Instance, config: Stage2Config, tables: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    """Evaluate a simple baseline rule from baseline_rules.csv on the same instance."""

    baseline_rules = tables["baseline_rules"]
    matches = baseline_rules[baseline_rules["baseline_rule_id"] == config.baseline_rule_id]
    if matches.empty:
        return {"baseline_rule_id": config.baseline_rule_id, "status": "missing_rule"}
    rule = matches.iloc[0]

    route_table = instance.core_route_table.copy()
    core_summary = instance.core_summary.set_index("core_id")
    accepted_quality_states = _split_states(rule.get("accepted_quality_states", "A;B;C"))
    min_quality = _optional_float(rule.get("min_quality_score_threshold"), default=0.0)
    min_life_ratio = _optional_float(rule.get("min_residual_life_ratio_threshold"), default=0.0)
    max_failure = _optional_float(rule.get("max_failure_probability_threshold"), default=1.0)

    candidates = instance.core_summary[
        (instance.core_summary["avg_quality_score"] >= min_quality)
        & (instance.core_summary["avg_residual_life_h"] >= min_life_ratio * instance.min_required_life_h)
        & (instance.core_summary["avg_failure_probability"] <= max_failure)
    ].copy()
    candidates = candidates.sort_values(["acceptability_score", "avg_quality_score"], ascending=False)

    selected_rows = []
    for core in candidates.itertuples(index=False):
        route_choice = _choose_route_for_rule(core.core_id, route_table, str(rule.get("route_selection_policy", "")), accepted_quality_states)
        if route_choice is not None:
            selected_rows.append(route_choice)
        if len(selected_rows) >= instance.demand_units:
            break

    selected = pd.DataFrame(selected_rows)
    productive_units = int((selected["route_class"] != "scrap").sum()) if not selected.empty else 0
    procurement_units = max(0, instance.demand_units - productive_units)
    fixed_cost = 0.0
    route_cost = 0.0
    environmental = 0.0
    quality_penalty = 0.0
    reliability_penalty = 0.0
    if not selected.empty:
        for row in selected.itertuples(index=False):
            fixed_cost += float(core_summary.loc[row.core_id, "fixed_accept_cost_rmb"])
            route_cost += float(row.economic_cost_rmb)
            environmental += config.env_weight * float(row.environmental_score)
            quality_penalty += config.quality_weight * max(0.0, instance.target_quality_score - float(row.expected_output_quality))
            reliability_penalty += config.quality_weight * config.reliability_weight * float(row.risk_penalty)

    procurement_cost = procurement_units * _procurement_unit_cost(instance)
    objective_value = fixed_cost + route_cost + environmental + quality_penalty + reliability_penalty + procurement_cost
    return {
        "baseline_rule_id": str(rule["baseline_rule_id"]),
        "baseline_rule_name": str(rule["baseline_rule_name"]),
        "status": "evaluated",
        "selected_core_count": int(len(selected)),
        "productive_units": productive_units,
        "procurement_units": procurement_units,
        "shortage_units": 0,
        "objective_value": float(objective_value),
        "route_mix": selected["route_class"].value_counts().to_dict() if not selected.empty else {},
        "objective_breakdown": {
            "accept_fixed_cost_rmb": fixed_cost,
            "route_economic_cost_rmb": route_cost,
            "environmental_cost_equiv": environmental,
            "quality_penalty_equiv": quality_penalty,
            "reliability_penalty_equiv": reliability_penalty,
            "procurement_cost_rmb": procurement_cost,
        },
    }


def _choose_route_for_rule(core_id: str, route_table: pd.DataFrame, policy: str, accepted_quality_states: set[str]) -> pd.Series | None:
    options = route_table[route_table["core_id"] == core_id].copy()
    if options.empty:
        return None
    if "A_to_R1_B_to_R2_C_to_R3_or_R5_D_to_R6_or_R7" in policy:
        priority = ["reuse", "repair", "laser", "replace", "scrap"]
    elif "argmin_expected_total_route_cost" in policy:
        return options.sort_values("economic_cost_rmb").iloc[0]
    elif "reuse" in policy.lower():
        priority = ["reuse", "repair", "laser", "replace", "scrap"]
    else:
        priority = ROUTE_CLASS_ORDER
    for route_class in priority:
        match = options[options["route_class"] == route_class]
        if not match.empty:
            return match.iloc[0]
    return options.iloc[0]


def _split_states(value: object) -> set[str]:
    text = "" if pd.isna(value) else str(value)
    return {part.strip() for part in text.split(";") if part.strip()}


def _optional_float(value: object, default: float) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _procurement_unit_cost(instance: Stage2Instance) -> float:
    machine_cost = float(instance.machine_summary.get("remanufacturing_cost_base_rmb", 0.0))
    selling_price = float(instance.machine_summary.get("selling_price", 0.0))
    return max(machine_cost * 1.45, selling_price * 0.92, 1.0)
