"""Post-solve checks for Stage 4 stochastic SAA solutions."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .structures import Stage4Instance, Stage4ModelData


PASS = "passed"
FAIL = "failed"


def model_size_counts(model_data: Stage4ModelData) -> Dict[str, int]:
    integer_mask = model_data.integrality > 0
    binary_mask = integer_mask & (model_data.lower_bounds == 0.0) & (model_data.upper_bounds == 1.0)
    return {
        "variable_count": len(model_data.variable_names),
        "constraint_count": len(model_data.constraint_names),
        "binary_variable_count": int(binary_mask.sum()),
        "general_integer_variable_count": int((integer_mask & ~binary_mask).sum()),
        "integer_variable_count": int(integer_mask.sum()),
    }


def run_solution_checks(
    instance: Stage4Instance,
    model_data: Stage4ModelData,
    x: np.ndarray,
    first_stage_decisions: pd.DataFrame,
    scenario_selected_routes: pd.DataFrame,
    scenario_assembly_plan: pd.DataFrame,
    scenario_inventory_trajectory: pd.DataFrame,
    scenario_capacity_utilization: pd.DataFrame,
    processing_window_periods: int,
    tolerance: float = 1e-6,
) -> List[Dict[str, object]]:
    checks = [
        _check_nonnegative(model_data, x, tolerance),
        _check_saa_probabilities(instance),
        _check_first_stage_constant(first_stage_decisions),
        _check_legal_routes(instance, scenario_selected_routes),
        _check_period_release(instance, scenario_selected_routes, processing_window_periods),
        _check_unaccepted_core_processing(instance, model_data, x, tolerance),
        _check_inventory_nonnegative(scenario_inventory_trajectory, tolerance),
        _check_capacity(scenario_capacity_utilization, tolerance),
        _check_matrix_residuals(model_data, x, tolerance),
        _check_final_backlog(scenario_assembly_plan),
    ]
    return checks


def summarize_checks(checks: List[Dict[str, object]]) -> Dict[str, int]:
    return {
        PASS: sum(1 for check in checks if check["severity"] == PASS),
        FAIL: sum(1 for check in checks if check["severity"] == FAIL),
    }


def _check_nonnegative(model_data: Stage4ModelData, x: np.ndarray, tolerance: float) -> Dict[str, object]:
    minimum = float(np.min(x)) if len(x) else 0.0
    return _check("nonnegative_variables", minimum >= -tolerance, f"Minimum variable value is {minimum:.8g}.")


def _check_saa_probabilities(instance: Stage4Instance) -> Dict[str, object]:
    total = float(instance.scenario_sample["saa_probability"].sum()) if not instance.scenario_sample.empty else 0.0
    passed = abs(total - 1.0) <= 1e-9
    return _check("scenario_probability_normalization", passed, f"Normalized scenario probability sum is {total:.10f}.")


def _check_first_stage_constant(first_stage_decisions: pd.DataFrame) -> Dict[str, object]:
    if first_stage_decisions.empty:
        return _check("first_stage_constant", False, "No first-stage decisions were generated.")
    repeated = first_stage_decisions.groupby("core_id")["accept_core"].nunique().max()
    passed = repeated <= 1
    return _check("first_stage_constant", passed, f"First-stage decision variability by core is {repeated}.")


def _check_legal_routes(instance: Stage4Instance, selected_routes: pd.DataFrame) -> Dict[str, object]:
    legal = set(
        zip(
            instance.component_route_period_scenario_table["scenario_id"],
            instance.component_route_period_scenario_table["component_instance_id"],
            instance.component_route_period_scenario_table["route_id"],
            instance.component_route_period_scenario_table["period_id"],
        )
    )
    chosen = set(
        zip(
            selected_routes.get("scenario_id", []),
            selected_routes.get("component_instance_id", []),
            selected_routes.get("route_id", []),
            selected_routes.get("period_id", []),
        )
    )
    illegal = sorted(chosen - legal)
    return _check(
        "legal_route_assignments",
        not illegal,
        f"Selected component-route-period-scenario assignments are legal; illegal_count={len(illegal)}.",
        observed=illegal[:5],
    )


def _check_period_release(instance: Stage4Instance, selected_routes: pd.DataFrame, processing_window_periods: int) -> Dict[str, object]:
    period_rank = {period: index for index, period in enumerate(instance.periods)}
    violations = []
    for row in selected_routes.itertuples(index=False):
        if period_rank[row.period_id] < period_rank.get(row.inspection_period, 0):
            violations.append((row.scenario_id, row.component_instance_id, row.route_id, row.period_id, row.inspection_period))
        if period_rank[row.period_id] - period_rank.get(row.inspection_period, 0) >= processing_window_periods:
            violations.append((row.scenario_id, row.component_instance_id, row.route_id, row.period_id, row.inspection_period))
    return _check(
        "period_release_window",
        not violations,
        f"Assignments outside inspection/release window: {len(violations)}.",
        observed=violations[:5],
    )


def _check_unaccepted_core_processing(instance: Stage4Instance, model_data: Stage4ModelData, x: np.ndarray, tolerance: float) -> Dict[str, object]:
    core_by_component = instance.component_summary.set_index("component_instance_id")["core_id"].to_dict()
    processed_by_scenario_core: Dict[tuple[str, str], float] = {}
    for key, index in model_data.variable_groups["x"].items():
        scenario_id, component_id, _route_id, _period = key.split("|")
        core_id = core_by_component[component_id]
        scenario_key = (scenario_id, core_id)
        processed_by_scenario_core[scenario_key] = processed_by_scenario_core.get(scenario_key, 0.0) + float(x[index])
    violations = []
    core_component_counts = instance.component_summary.groupby("core_id").size().to_dict()
    for (scenario_id, core_id), processed in processed_by_scenario_core.items():
        accept = float(x[model_data.variable_groups["accept_core"][core_id]])
        if processed > core_component_counts.get(core_id, 0) * accept + tolerance:
            violations.append((scenario_id, core_id, processed, accept))
    return _check("no_processing_without_core_acceptance", not violations, f"Core-processing violations: {len(violations)}.", observed=violations[:5])


def _check_inventory_nonnegative(inventory: pd.DataFrame, tolerance: float) -> Dict[str, object]:
    minimum = float(inventory["ending_inventory"].min()) if not inventory.empty else 0.0
    return _check("inventory_nonnegative", minimum >= -tolerance, f"Minimum ending inventory is {minimum:.8g}.")


def _check_capacity(capacity: pd.DataFrame, tolerance: float) -> Dict[str, object]:
    if capacity.empty:
        return _check("capacity_limits", True, "No capacity rows were generated.")
    violations = capacity[capacity["used_hours"] > capacity["available_regular_hours"] + capacity["overtime_hours"] + tolerance]
    return _check("capacity_limits", violations.empty, f"Capacity violations: {len(violations)}.", observed=violations.head(5).to_dict(orient="records"))


def _check_matrix_residuals(model_data: Stage4ModelData, x: np.ndarray, tolerance: float) -> Dict[str, object]:
    values = model_data.constraint_matrix @ x
    below = np.maximum(model_data.constraint_lhs - values, 0.0)
    above = np.maximum(values - model_data.constraint_rhs, 0.0)
    finite = np.concatenate([below[np.isfinite(below)], above[np.isfinite(above)]])
    max_violation = float(finite.max()) if finite.size else 0.0
    return _check("linear_constraint_residuals", max_violation <= tolerance, f"Maximum matrix residual violation is {max_violation:.8g}.")


def _check_final_backlog(assembly_plan: pd.DataFrame) -> Dict[str, object]:
    if assembly_plan.empty:
        return _check("final_backlog_reported", True, "No assembly rows were generated.")
    final_backlog = float(assembly_plan["backlog_units"].iloc[-1])
    return _check("final_backlog_reported", final_backlog >= 0.0, f"Final backlog is {final_backlog:.8g}.", observed=final_backlog)


def _check(name: str, passed: bool, message: str, observed: object = None) -> Dict[str, object]:
    return {
        "check_name": name,
        "severity": PASS if passed else FAIL,
        "message": message,
        "observed": observed,
    }
