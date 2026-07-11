"""Post-solve checks for Stage 3 multi-period solutions."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .structures import Stage3Instance, Stage3ModelData


PASS = "passed"
FAIL = "failed"


def model_size_counts(model_data: Stage3ModelData) -> Dict[str, int]:
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
    instance: Stage3Instance,
    model_data: Stage3ModelData,
    x: np.ndarray,
    selected_routes: pd.DataFrame,
    assembly_plan: pd.DataFrame,
    inventory_trajectory: pd.DataFrame,
    capacity_utilization: pd.DataFrame,
    tolerance: float = 1e-6,
) -> List[Dict[str, object]]:
    checks = [
        _check_nonnegative(model_data, x, tolerance),
        _check_legal_routes(instance, selected_routes),
        _check_period_release(instance, selected_routes),
        _check_unaccepted_core_processing(instance, model_data, x, tolerance),
        _check_inventory_nonnegative(inventory_trajectory, tolerance),
        _check_capacity(capacity_utilization, tolerance),
        _check_matrix_residuals(model_data, x, tolerance),
        _check_final_backlog(assembly_plan),
    ]
    return checks


def summarize_checks(checks: List[Dict[str, object]]) -> Dict[str, int]:
    return {
        PASS: sum(1 for check in checks if check["severity"] == PASS),
        FAIL: sum(1 for check in checks if check["severity"] == FAIL),
    }


def _check_nonnegative(model_data: Stage3ModelData, x: np.ndarray, tolerance: float) -> Dict[str, object]:
    minimum = float(np.min(x)) if len(x) else 0.0
    return _check("nonnegative_variables", minimum >= -tolerance, f"Minimum variable value is {minimum:.8g}.")


def _check_legal_routes(instance: Stage3Instance, selected_routes: pd.DataFrame) -> Dict[str, object]:
    legal = set(
        zip(
            instance.component_route_period_table["component_instance_id"],
            instance.component_route_period_table["route_id"],
            instance.component_route_period_table["period_id"],
        )
    )
    chosen = set(zip(selected_routes.get("component_instance_id", []), selected_routes.get("route_id", []), selected_routes.get("period_id", [])))
    illegal = sorted(chosen - legal)
    return _check(
        "legal_route_assignments",
        not illegal,
        f"Selected component-route-period assignments are legal; illegal_count={len(illegal)}.",
        observed=illegal[:5],
    )


def _check_period_release(instance: Stage3Instance, selected_routes: pd.DataFrame) -> Dict[str, object]:
    period_rank = {period: index for index, period in enumerate(instance.periods)}
    violations = []
    for row in selected_routes.itertuples(index=False):
        if period_rank[row.period_id] < period_rank.get(row.inspection_period, 0):
            violations.append((row.component_instance_id, row.route_id, row.period_id, row.inspection_period))
    return _check(
        "period_not_before_inspection",
        not violations,
        f"Assignments before inspection period: {len(violations)}.",
        observed=violations[:5],
    )


def _check_unaccepted_core_processing(instance: Stage3Instance, model_data: Stage3ModelData, x: np.ndarray, tolerance: float) -> Dict[str, object]:
    core_by_component = instance.component_summary.set_index("component_instance_id")["core_id"].to_dict()
    processed_by_core: Dict[str, float] = {}
    for key, index in model_data.variable_groups["x"].items():
        component_id, _route_id, _period = key.split("|")
        core_id = core_by_component[component_id]
        processed_by_core[core_id] = processed_by_core.get(core_id, 0.0) + float(x[index])
    violations = []
    for core_id, processed in processed_by_core.items():
        accept = float(x[model_data.variable_groups["accept_core"][core_id]])
        if processed > instance.component_summary[instance.component_summary["core_id"] == core_id].shape[0] * accept + tolerance:
            violations.append((core_id, processed, accept))
    return _check("no_processing_without_core_acceptance", not violations, f"Core-processing violations: {len(violations)}.", observed=violations[:5])


def _check_inventory_nonnegative(inventory: pd.DataFrame, tolerance: float) -> Dict[str, object]:
    minimum = float(inventory["ending_inventory"].min()) if not inventory.empty else 0.0
    return _check("inventory_nonnegative", minimum >= -tolerance, f"Minimum ending inventory is {minimum:.8g}.")


def _check_capacity(capacity: pd.DataFrame, tolerance: float) -> Dict[str, object]:
    if capacity.empty:
        return _check("capacity_limits", True, "No capacity rows were generated.")
    violations = capacity[capacity["used_hours"] > capacity["available_regular_hours"] + capacity["overtime_hours"] + tolerance]
    return _check("capacity_limits", violations.empty, f"Capacity violations: {len(violations)}.", observed=violations.head(5).to_dict(orient="records"))


def _check_matrix_residuals(model_data: Stage3ModelData, x: np.ndarray, tolerance: float) -> Dict[str, object]:
    values = model_data.constraint_matrix @ x
    below = np.maximum(model_data.constraint_lhs - values, 0.0)
    above = np.maximum(values - model_data.constraint_rhs, 0.0)
    finite = np.concatenate([below[np.isfinite(below)], above[np.isfinite(above)]])
    max_violation = float(finite.max()) if finite.size else 0.0
    return _check("linear_constraint_residuals", max_violation <= tolerance, f"Maximum matrix residual violation is {max_violation:.8g}.")


def _check_final_backlog(assembly_plan: pd.DataFrame) -> Dict[str, object]:
    final_backlog = float(assembly_plan["backlog_units"].iloc[-1]) if not assembly_plan.empty else 0.0
    return _check("final_backlog_reported", final_backlog >= 0.0, f"Final backlog is {final_backlog:.8g}.", observed=final_backlog)


def _check(name: str, passed: bool, message: str, observed: object = None) -> Dict[str, object]:
    return {
        "check_name": name,
        "severity": PASS if passed else FAIL,
        "message": message,
        "observed": observed,
    }
