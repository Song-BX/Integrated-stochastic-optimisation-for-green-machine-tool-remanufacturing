"""Simple deterministic baseline comparison for Stage 3."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from .config import Stage3Config
from .structures import Stage3Instance


def evaluate_baseline(instance: Stage3Instance, config: Stage3Config, tables: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    """Evaluate a BR02-like quality-threshold baseline on the same window."""

    baseline_rules = tables["baseline_rules"]
    matches = baseline_rules[baseline_rules["baseline_rule_id"] == config.baseline_rule_id]
    rule_name = str(matches.iloc[0]["baseline_rule_name"]) if not matches.empty else "quality_threshold_rule"

    components = instance.component_summary.copy()
    components = components[components["quality_score"] >= 0.35].copy()
    route_priority = ["R1", "R2", "R3", "R5", "R4", "R6", "R7"]
    selected = []
    for component in components.itertuples(index=False):
        feasible = str(component.feasible_route_ids).split(";")
        chosen = next((route_id for route_id in route_priority if route_id in feasible), None)
        if chosen:
            selected.append((component.component_instance_id, component.component_type, chosen))

    selected_df = pd.DataFrame(selected, columns=["component_instance_id", "component_type", "route_id"])
    demand = int(instance.period_demand["demand_units"].sum())
    bom = instance.bom_requirements.set_index("component_type")["required_quantity"].to_dict()
    inventory = instance.initial_inventory.set_index("component_type")["initial_quantity_available"].to_dict()
    produced = selected_df[selected_df["route_id"] != "R7"].groupby("component_type").size().to_dict() if not selected_df.empty else {}
    max_assembly = demand
    for component_type, required in bom.items():
        available = float(inventory.get(component_type, 0.0)) + float(produced.get(component_type, 0.0))
        max_assembly = min(max_assembly, int(available // float(required)))
    backlog = max(0, demand - max_assembly)
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
