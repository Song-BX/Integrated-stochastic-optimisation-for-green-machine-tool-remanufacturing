"""Scenario-averaged baseline comparison for Stage 4."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from .config import Stage4Config
from .structures import Stage4Instance


def evaluate_baseline(instance: Stage4Instance, config: Stage4Config, tables: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    """Evaluate a BR02-like rule against the expected-value Stage 4 window."""

    baseline_rules = tables["baseline_rules"]
    matches = baseline_rules[baseline_rules["baseline_rule_id"] == config.baseline_rule_id]
    rule_name = str(matches.iloc[0]["baseline_rule_name"]) if not matches.empty else "quality_threshold_rule"

    route_priority = _route_priority(matches.iloc[0] if not matches.empty else None)
    min_quality = _float_value(matches.iloc[0], "min_quality_score_threshold", 0.35) if not matches.empty else 0.35
    min_life_ratio = _float_value(matches.iloc[0], "min_residual_life_ratio_threshold", 0.75) if not matches.empty else 0.75
    selected_rows = []
    for component in instance.component_summary.itertuples(index=False):
        if float(component.quality_score) < min_quality or float(component.residual_life_mean_h) < min_life_ratio * instance.min_required_life_h:
            continue
        feasible = str(component.feasible_route_ids).split(";")
        chosen = next((route_id for route_id in route_priority if route_id in feasible), None)
        if chosen:
            selected_rows.append((component.component_instance_id, component.component_type, chosen))

    selected_df = pd.DataFrame(selected_rows, columns=["component_instance_id", "component_type", "route_id"])
    demand = float((instance.scenario_demand["demand_units"] * instance.scenario_demand["saa_probability"]).sum()) if not instance.scenario_demand.empty else 0.0
    bom = instance.bom_requirements.set_index("component_type")["required_quantity"].to_dict()
    inventory = instance.initial_inventory.set_index("component_type")["initial_quantity_available"].to_dict()
    produced = selected_df[selected_df["route_id"] != "R7"].groupby("component_type").size().to_dict() if not selected_df.empty else {}
    max_assembly = int(round(demand))
    for component_type, required in bom.items():
        available = float(inventory.get(component_type, 0.0)) + float(produced.get(component_type, 0.0))
        max_assembly = min(max_assembly, int(available // float(required)))
    backlog = max(0, int(round(demand)) - max_assembly)
    route_mix = selected_df["route_id"].value_counts().sort_index().to_dict() if not selected_df.empty else {}
    return {
        "baseline_rule_id": config.baseline_rule_id,
        "baseline_rule_name": rule_name,
        "status": "evaluated",
        "selected_component_count": int(len(selected_df)),
        "estimated_assembly_units": int(max_assembly),
        "estimated_backlog_units": int(backlog),
        "route_mix": route_mix,
    }


def _route_priority(rule: pd.Series | None) -> list[str]:
    if rule is None:
        return ["R1", "R2", "R3", "R5", "R4", "R6", "R7"]
    raw = str(rule.get("route_priority_sequence", ""))
    parsed = [token.strip() for token in raw.replace(";", ",").split(",") if token.strip()]
    return parsed or ["R1", "R2", "R3", "R5", "R4", "R6", "R7"]


def _float_value(row: pd.Series, key: str, default: float) -> float:
    try:
        value = row.get(key, default)
        return float(value) if value is not None and value == value else default
    except Exception:
        return default
