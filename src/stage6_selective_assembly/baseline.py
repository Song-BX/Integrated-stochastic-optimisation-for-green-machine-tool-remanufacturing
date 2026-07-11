"""Baseline comparison helpers for Stage 6 selective assembly."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from stage5_risk_averse.baseline import evaluate_baseline as evaluate_stage5_baseline

from .config import Stage6Config
from .structures import Stage6Instance


def evaluate_baseline(instance: Stage6Instance, config: Stage6Config, tables: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    """Evaluate BR14 plus BR08/BR18 selective-assembly references."""

    baseline = evaluate_stage5_baseline(instance, config, tables)
    rules = tables["baseline_rules"]
    baseline["selective_assembly_rule"] = _rule_payload(rules, config.selective_assembly_baseline_rule_id)
    baseline["no_selective_assembly_ablation_rule"] = _rule_payload(rules, config.no_selective_assembly_ablation_rule_id)
    baseline["stage6_candidate_pool_size"] = int(len(instance.assembly_candidate_pool))
    baseline["stage6_pair_pool_size"] = int(len(instance.assembly_pair_pool))
    baseline["stage6_hard_pair_count"] = int((instance.assembly_pair_pool["soft_pair_flag"] == 0).sum())
    baseline["stage6_soft_pair_count"] = int((instance.assembly_pair_pool["soft_pair_flag"] == 1).sum())
    baseline["stage6_requirement_count"] = int(len(instance.assembly_requirements))
    return baseline


def _rule_payload(rules: pd.DataFrame, rule_id: str) -> Dict[str, object]:
    match = rules[rules["baseline_rule_id"] == rule_id]
    if match.empty:
        return {"baseline_rule_id": rule_id, "status": "missing"}
    rule = match.iloc[0]
    return {
        "baseline_rule_id": rule_id,
        "baseline_rule_name": rule.get("baseline_rule_name"),
        "assembly_policy": rule.get("assembly_policy"),
        "selective_assembly_enabled": _boolish(rule.get("selective_assembly_enabled_flag")),
        "chance_constraints_enabled": _boolish(rule.get("chance_constraint_enabled_flag")),
        "cvar_enabled": _boolish(rule.get("cvar_enabled_flag")),
        "scenario_aware": _boolish(rule.get("scenario_aware_flag")),
        "recourse_enabled": _boolish(rule.get("recourse_enabled_flag")),
        "risk_weight": _float_value(rule, "risk_weight", 0.0),
        "reliability_weight": _float_value(rule, "reliability_weight", 0.0),
        "assembly_weight": _float_value(rule, "assembly_weight", 0.0),
    }


def _float_value(row: pd.Series, key: str, default: float) -> float:
    try:
        value = row.get(key, default)
        return float(value) if value is not None and value == value else default
    except Exception:
        return default


def _boolish(value: object) -> bool | None:
    text = str(value).strip().lower()
    if not text or text == "nan":
        return None
    return text in {"1", "true", "yes", "y"}
