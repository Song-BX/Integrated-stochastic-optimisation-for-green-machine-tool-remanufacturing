"""Baseline comparison helpers for Stage 5."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from stage4_stochastic.baseline import evaluate_baseline as evaluate_stage4_baseline

from .config import Stage5Config
from .structures import Stage5Instance


def evaluate_baseline(instance: Stage5Instance, config: Stage5Config, tables: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    """Evaluate a BR14-style baseline with Stage 5 metadata."""

    baseline = evaluate_stage4_baseline(instance, config, tables)
    rules = tables["baseline_rules"]
    match = rules[rules["baseline_rule_id"] == config.risk_baseline_rule_id]
    if not match.empty:
        rule = match.iloc[0]
        baseline.update(
            {
                "risk_baseline_rule_id": config.risk_baseline_rule_id,
                "risk_weight": _float_value(rule, "risk_weight", config.cvar_lambda),
                "reliability_weight": _float_value(rule, "reliability_weight", config.reliability_weight),
                "uses_cvar": _boolish(rule.get("cvar_enabled_flag")),
                "uses_chance_constraints": _boolish(rule.get("chance_constraint_enabled_flag")),
                "uses_scenarios": _boolish(rule.get("scenario_aware_flag")),
                "allows_recourse": _boolish(rule.get("recourse_enabled_flag")),
            }
        )
    baseline["stage5_chance_pass_rate"] = instance.to_summary_dict().get("chance_pass_rate")
    return baseline


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
