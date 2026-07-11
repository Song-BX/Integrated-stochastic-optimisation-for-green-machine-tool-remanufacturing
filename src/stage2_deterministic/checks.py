"""Post-solve consistency checks for Stage 2 deterministic solutions."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .structures import Stage2Instance, Stage2ModelData


PASS = "passed"
FAIL = "failed"


def model_size_counts(model_data: Stage2ModelData) -> Dict[str, int]:
    """Return robust variable counts for scipy MILP data."""

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
    instance: Stage2Instance,
    model_data: Stage2ModelData,
    x: np.ndarray,
    selected_routes: pd.DataFrame,
    capacity_utilization: pd.DataFrame,
    tolerance: float = 1e-6,
) -> List[Dict[str, object]]:
    """Run lightweight feasibility and reporting checks on a solved solution."""

    checks: List[Dict[str, object]] = []
    checks.append(_check_nonnegative_variables(model_data, x, tolerance))
    checks.append(_check_route_assignments(instance, selected_routes))
    checks.append(_check_one_route_per_accepted(model_data, x, tolerance))
    checks.append(_check_demand_balance(instance, model_data, x, selected_routes, tolerance))
    checks.append(_check_capacity(capacity_utilization, tolerance))
    checks.append(_check_matrix_residuals(model_data, x, tolerance))
    return checks


def summarize_checks(checks: List[Dict[str, object]]) -> Dict[str, int]:
    return {
        PASS: sum(1 for check in checks if check["severity"] == PASS),
        FAIL: sum(1 for check in checks if check["severity"] == FAIL),
    }


def _check_nonnegative_variables(model_data: Stage2ModelData, x: np.ndarray, tolerance: float) -> Dict[str, object]:
    minimum = float(np.min(x)) if len(x) else 0.0
    return _check(
        "nonnegative_variables",
        minimum >= -tolerance,
        f"Minimum variable value is {minimum:.8g}.",
        expected=f">= {-tolerance}",
    )


def _check_route_assignments(instance: Stage2Instance, selected_routes: pd.DataFrame) -> Dict[str, object]:
    legal_pairs = set(zip(instance.core_route_table["core_id"], instance.core_route_table["route_class"]))
    selected_pairs = set(zip(selected_routes.get("core_id", []), selected_routes.get("route_class", [])))
    illegal = sorted(selected_pairs - legal_pairs)
    return _check(
        "legal_route_assignments",
        not illegal,
        f"Selected route assignments are present in the generated feasible route table; illegal_count={len(illegal)}.",
        observed=illegal[:5],
        expected="illegal_count=0",
    )


def _check_one_route_per_accepted(model_data: Stage2ModelData, x: np.ndarray, tolerance: float) -> Dict[str, object]:
    route_by_core: Dict[str, float] = {}
    for key, index in model_data.variable_groups["route"].items():
        core_id, _route_class = key.split("|")
        route_by_core[core_id] = route_by_core.get(core_id, 0.0) + float(x[index])
    deviations = []
    for core_id, accept_idx in model_data.variable_groups["accept"].items():
        deviations.append(abs(route_by_core.get(core_id, 0.0) - float(x[accept_idx])))
    max_deviation = max(deviations) if deviations else 0.0
    return _check(
        "one_route_per_accepted_core",
        max_deviation <= tolerance,
        f"Maximum |sum(route)-accept| deviation is {max_deviation:.8g}.",
        observed=max_deviation,
        expected=f"<= {tolerance}",
    )


def _check_demand_balance(
    instance: Stage2Instance,
    model_data: Stage2ModelData,
    x: np.ndarray,
    selected_routes: pd.DataFrame,
    tolerance: float,
) -> Dict[str, object]:
    procurement = float(x[model_data.variable_groups["procurement"]])
    shortage = float(x[model_data.variable_groups["shortage"]])
    productive = int((selected_routes["route_class"] != "scrap").sum()) if not selected_routes.empty else 0
    supplied = productive + procurement + shortage
    return _check(
        "aggregate_demand_balance",
        supplied + tolerance >= instance.demand_units,
        f"Productive cores + procurement + shortage = {supplied:.8g}; demand = {instance.demand_units}.",
        observed=supplied,
        expected=f">= {instance.demand_units}",
    )


def _check_capacity(capacity_utilization: pd.DataFrame, tolerance: float) -> Dict[str, object]:
    if capacity_utilization.empty:
        return _check("capacity_limits", True, "No capacity rows were generated.")
    lhs = capacity_utilization["used_hours"].astype(float)
    rhs = capacity_utilization["available_regular_hours"].astype(float) + capacity_utilization["overtime_hours"].astype(float)
    violations = capacity_utilization[lhs > rhs + tolerance]
    return _check(
        "capacity_limits",
        violations.empty,
        f"Capacity violations: {len(violations)}.",
        observed=violations[["resource_type", "used_hours", "available_regular_hours", "overtime_hours"]].head(5).to_dict(orient="records"),
        expected="used_hours <= available_regular_hours + overtime_hours",
    )


def _check_matrix_residuals(model_data: Stage2ModelData, x: np.ndarray, tolerance: float) -> Dict[str, object]:
    values = model_data.constraint_matrix @ x
    below = np.maximum(model_data.constraint_lhs - values, 0.0)
    above = np.maximum(values - model_data.constraint_rhs, 0.0)
    finite_below = below[np.isfinite(below)]
    finite_above = above[np.isfinite(above)]
    max_violation = 0.0
    if finite_below.size:
        max_violation = max(max_violation, float(finite_below.max()))
    if finite_above.size:
        max_violation = max(max_violation, float(finite_above.max()))
    return _check(
        "linear_constraint_residuals",
        max_violation <= tolerance,
        f"Maximum matrix residual violation is {max_violation:.8g}.",
        observed=max_violation,
        expected=f"<= {tolerance}",
    )


def _check(name: str, passed: bool, message: str, observed: object = None, expected: object = None) -> Dict[str, object]:
    return {
        "check_name": name,
        "severity": PASS if passed else FAIL,
        "message": message,
        "observed": observed,
        "expected": expected,
    }
