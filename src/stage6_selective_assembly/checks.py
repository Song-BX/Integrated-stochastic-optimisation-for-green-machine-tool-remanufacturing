"""Post-solve checks for Stage 6 selective-assembly solutions."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from stage5_risk_averse.checks import (
    FAIL,
    PASS,
    model_size_counts,
    run_solution_checks as run_stage5_solution_checks,
    summarize_checks,
)

from .aggregation import CANDIDATE_SCREENING_FLAGS, PAIR_HARD_FLAGS
from .structures import Stage6Instance, Stage6ModelData


def run_solution_checks(
    instance: Stage6Instance,
    model_data: Stage6ModelData,
    x: np.ndarray,
    first_stage_decisions: pd.DataFrame,
    scenario_selected_routes: pd.DataFrame,
    scenario_assembly_plan: pd.DataFrame,
    scenario_inventory_trajectory: pd.DataFrame,
    scenario_capacity_utilization: pd.DataFrame,
    scenario_risk_metrics: pd.DataFrame,
    selected_candidates: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    feature_plan: pd.DataFrame,
    processing_window_periods: int,
    tolerance: float = 1e-6,
) -> List[Dict[str, object]]:
    checks = run_stage5_solution_checks(
        instance,
        model_data,
        x,
        first_stage_decisions,
        scenario_selected_routes,
        scenario_assembly_plan,
        scenario_inventory_trajectory,
        scenario_capacity_utilization,
        scenario_risk_metrics,
        processing_window_periods,
        tolerance,
    )
    checks.extend(
        [
            _check_requirements(instance),
            _check_pools_nonempty(instance),
            _check_selected_candidate_eligibility(selected_candidates),
            _check_selected_pair_compatibility(selected_pairs),
            _check_feature_coverage(feature_plan, tolerance),
            _check_old_candidate_route_coupling(selected_candidates, scenario_selected_routes),
            _check_new_candidate_availability(instance, model_data, x, selected_candidates, tolerance),
        ]
    )
    return checks


def _check_requirements(instance: Stage6Instance) -> Dict[str, object]:
    count = int(len(instance.assembly_requirements))
    passed = count == 6 if instance.machine_type_id == "CK6150" else count > 0
    return _check("assembly_requirement_count", passed, f"Assembly requirement count for {instance.machine_type_id}: {count}.")


def _check_pools_nonempty(instance: Stage6Instance) -> Dict[str, object]:
    candidate_count = int(len(instance.assembly_candidate_pool))
    pair_count = int(len(instance.assembly_pair_pool))
    return _check(
        "assembly_candidate_and_pair_pools_nonempty",
        candidate_count > 0 and pair_count > 0,
        f"Candidate pool={candidate_count}; pair pool={pair_count}.",
    )


def _check_selected_candidate_eligibility(selected_candidates: pd.DataFrame) -> Dict[str, object]:
    if selected_candidates.empty:
        return _check("selected_candidates_eligible", True, "No assembly candidates were selected.")
    flags = [column for column in CANDIDATE_SCREENING_FLAGS if column in selected_candidates.columns]
    violations = selected_candidates[selected_candidates[flags].min(axis=1) < 1.0] if flags else selected_candidates.iloc[0:0]
    return _check(
        "selected_candidates_eligible",
        violations.empty,
        f"Selected candidate eligibility violations: {len(violations)}.",
        observed=violations.head(5).to_dict(orient="records"),
    )


def _check_selected_pair_compatibility(selected_pairs: pd.DataFrame) -> Dict[str, object]:
    if selected_pairs.empty:
        return _check("selected_pairs_compatible", True, "No assembly pairs were selected.")
    flags = [column for column in PAIR_HARD_FLAGS if column in selected_pairs.columns]
    dimension_pass = (
        (pd.to_numeric(selected_pairs.get("dimension_constraint_flag", 0), errors="coerce").fillna(0.0) >= 1.0)
        | (pd.to_numeric(selected_pairs.get("dimension_soft_constraint_flag", 0), errors="coerce").fillna(0.0) >= 1.0)
    )
    hard_flags_pass = selected_pairs[flags].min(axis=1) >= 1.0 if flags else pd.Series(True, index=selected_pairs.index)
    status_pass = selected_pairs["compatibility_status"].isin(["hard_feasible", "soft_feasible_with_penalty"])
    violations = selected_pairs[~(dimension_pass & hard_flags_pass & status_pass)]
    return _check(
        "selected_pairs_compatible",
        violations.empty,
        f"Selected pair compatibility violations: {len(violations)}.",
        observed=violations.head(5).to_dict(orient="records"),
    )


def _check_feature_coverage(feature_plan: pd.DataFrame, tolerance: float) -> Dict[str, object]:
    if feature_plan.empty:
        return _check("feature_assembly_coverage", False, "No feature assembly plan was generated.")
    residual = pd.to_numeric(feature_plan["coverage_residual"], errors="coerce").fillna(0.0).abs()
    max_residual = float(residual.max()) if not residual.empty else 0.0
    return _check("feature_assembly_coverage", max_residual <= tolerance, f"Max coverage residual={max_residual:.8g}.")


def _check_old_candidate_route_coupling(selected_candidates: pd.DataFrame, selected_routes: pd.DataFrame) -> Dict[str, object]:
    if selected_candidates.empty:
        return _check("old_candidates_have_processed_routes", True, "No assembly candidates were selected.")
    old = selected_candidates[pd.to_numeric(selected_candidates.get("old_candidate_flag", 0), errors="coerce").fillna(0).astype(int) == 1]
    if old.empty:
        return _check("old_candidates_have_processed_routes", True, "No old reused/remanufactured assembly candidates were selected.")
    route_set = set(
        zip(
            selected_routes.get("scenario_id", []),
            selected_routes.get("component_instance_id", []),
            selected_routes.get("route_id", []),
        )
    )
    violations = []
    for row in old.itertuples(index=False):
        key = (row.scenario_id, row.component_instance_id, row.planned_route_id)
        if key not in route_set:
            violations.append(key)
    return _check(
        "old_candidates_have_processed_routes",
        not violations,
        f"Old candidate route-coupling violations: {len(violations)}.",
        observed=violations[:5],
    )


def _check_new_candidate_availability(
    instance: Stage6Instance,
    model_data: Stage6ModelData,
    x: np.ndarray,
    selected_candidates: pd.DataFrame,
    tolerance: float,
) -> Dict[str, object]:
    if selected_candidates.empty:
        return _check("new_candidate_availability", True, "No assembly candidates were selected.")
    new_selected = selected_candidates[
        pd.to_numeric(selected_candidates.get("new_backup_candidate_flag", 0), errors="coerce").fillna(0).astype(int) == 1
    ]
    if new_selected.empty:
        return _check("new_candidate_availability", True, "No new/new_replacement candidates were selected.")
    initial = instance.initial_inventory.set_index("component_type")["initial_quantity_available"].to_dict()
    violations = []
    for (scenario_id, component_type), group in new_selected.groupby(["scenario_id", "component_type"], dropna=False):
        selected = float(len(group))
        available = float(initial.get(component_type, 0.0))
        for period in instance.periods:
            pre_key = f"{component_type}|{period}"
            recourse_key = f"{scenario_id}|{component_type}|{period}"
            if pre_key in model_data.variable_groups["pre_procure"]:
                available += float(x[model_data.variable_groups["pre_procure"][pre_key]])
            if recourse_key in model_data.variable_groups["recourse_procure"]:
                available += float(x[model_data.variable_groups["recourse_procure"][recourse_key]])
        if selected > available + tolerance:
            violations.append((scenario_id, component_type, selected, available))
    return _check(
        "new_candidate_availability",
        not violations,
        f"New candidate availability violations: {len(violations)}.",
        observed=violations[:5],
    )


def _check(name: str, passed: bool, message: str, observed: object = None) -> Dict[str, object]:
    return {
        "check_name": name,
        "severity": PASS if passed else FAIL,
        "message": message,
        "observed": observed,
    }


__all__ = ["FAIL", "PASS", "model_size_counts", "run_solution_checks", "summarize_checks"]
