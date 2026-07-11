"""Post-solve checks for Stage 5 risk-averse SAA solutions."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from stage4_stochastic.checks import (
    FAIL,
    PASS,
    model_size_counts,
    run_solution_checks as run_stage4_solution_checks,
    summarize_checks,
)

from .structures import Stage5Instance, Stage5ModelData


def run_solution_checks(
    instance: Stage5Instance,
    model_data: Stage5ModelData,
    x: np.ndarray,
    first_stage_decisions: pd.DataFrame,
    scenario_selected_routes: pd.DataFrame,
    scenario_assembly_plan: pd.DataFrame,
    scenario_inventory_trajectory: pd.DataFrame,
    scenario_capacity_utilization: pd.DataFrame,
    scenario_risk_metrics: pd.DataFrame,
    processing_window_periods: int,
    tolerance: float = 1e-6,
) -> List[Dict[str, object]]:
    checks = run_stage4_solution_checks(
        instance,
        model_data,
        x,
        first_stage_decisions,
        scenario_selected_routes,
        scenario_assembly_plan,
        scenario_inventory_trajectory,
        scenario_capacity_utilization,
        processing_window_periods,
        tolerance,
    )
    checks.extend(
        [
            _check_risk_reliability_join(instance),
            _check_selected_routes_chance(instance, scenario_selected_routes),
            _check_cvar_tail_constraints(scenario_risk_metrics, tolerance),
        ]
    )
    return checks


def _check_risk_reliability_join(instance: Stage5Instance) -> Dict[str, object]:
    table = instance.component_route_period_scenario_table
    reliability_joined = int(table["stage5_reliability_joined"].sum()) if "stage5_reliability_joined" in table else 0
    risk_joined = int(table["stage5_risk_joined"].sum()) if "stage5_risk_joined" in table else 0
    passed = reliability_joined == len(table) and risk_joined == len(table) and len(table) > 0
    return _check(
        "risk_reliability_joins_nonempty",
        passed,
        f"Stage 5 candidate rows={len(table)}, reliability_joined={reliability_joined}, risk_joined={risk_joined}.",
    )


def _check_selected_routes_chance(instance: Stage5Instance, selected_routes: pd.DataFrame) -> Dict[str, object]:
    if selected_routes.empty:
        return _check("selected_routes_satisfy_chance_constraints", True, "No component routes were selected.")
    violations = selected_routes[
        (pd.to_numeric(selected_routes["chance_constraint_satisfied_flag"], errors="coerce").fillna(0.0) < 1.0)
        | (
            pd.to_numeric(selected_routes["survival_probability_at_min_system_life"], errors="coerce").fillna(0.0)
            < float(instance.min_system_reliability)
        )
    ]
    return _check(
        "selected_routes_satisfy_chance_constraints",
        violations.empty,
        f"Chance-constraint route violations: {len(violations)}.",
        observed=violations.head(5).to_dict(orient="records"),
    )


def _check_cvar_tail_constraints(scenario_risk_metrics: pd.DataFrame, tolerance: float) -> Dict[str, object]:
    if scenario_risk_metrics.empty:
        return _check("cvar_tail_constraints", False, "No CVaR scenario risk metrics were generated.")
    residual = pd.to_numeric(scenario_risk_metrics["cvar_constraint_residual"], errors="coerce").fillna(0.0)
    max_violation = float(residual.max()) if not residual.empty else 0.0
    tail_min = float(pd.to_numeric(scenario_risk_metrics["tail_excess"], errors="coerce").fillna(0.0).min())
    passed = max_violation <= tolerance and tail_min >= -tolerance
    return _check(
        "cvar_tail_constraints",
        passed,
        f"Max CVaR residual={max_violation:.8g}; min tail_excess={tail_min:.8g}.",
    )


def _check(name: str, passed: bool, message: str, observed: object = None) -> Dict[str, object]:
    return {
        "check_name": name,
        "severity": PASS if passed else FAIL,
        "message": message,
        "observed": observed,
    }


__all__ = ["FAIL", "PASS", "model_size_counts", "run_solution_checks", "summarize_checks"]

