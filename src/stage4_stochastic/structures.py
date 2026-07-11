"""Public data structures for Stage 4 stochastic optimisation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import pandas as pd


@dataclass
class Stage4Instance:
    """A two-stage stochastic rolling-window component-routing instance."""

    machine_type_id: str
    machine_family: str
    periods: List[str]
    scenario_ids: List[str]
    component_types: List[str]
    route_ids: List[str]
    resource_types: List[str]
    demand_units_expected: float
    candidate_core_count: int
    component_instance_count: int
    min_required_life_h: float
    target_quality_score: float
    machine_summary: Dict[str, Any]
    scenario_sample: pd.DataFrame
    scenario_probability_summary: Dict[str, Any]
    scenario_demand: pd.DataFrame
    bom_requirements: pd.DataFrame
    core_summary: pd.DataFrame
    component_summary: pd.DataFrame
    component_route_period_scenario_table: pd.DataFrame
    initial_inventory: pd.DataFrame
    procurement_costs: pd.DataFrame
    capacity_table: pd.DataFrame
    scenario_capacity_table: pd.DataFrame

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "machine_type_id": self.machine_type_id,
            "machine_family": self.machine_family,
            "period_start": self.periods[0] if self.periods else None,
            "period_end": self.periods[-1] if self.periods else None,
            "period_count": len(self.periods),
            "scenario_count": len(self.scenario_ids),
            "scenario_ids": self.scenario_ids,
            "component_types": self.component_types,
            "route_ids": self.route_ids,
            "resource_types": self.resource_types,
            "expected_demand_units": self.demand_units_expected,
            "candidate_core_count": self.candidate_core_count,
            "component_instance_count": self.component_instance_count,
            "min_required_life_h": self.min_required_life_h,
            "target_quality_score": self.target_quality_score,
            "machine_summary": self.machine_summary,
        }


@dataclass
class Stage4ModelData:
    """Matrix data passed to scipy.optimize.milp."""

    variable_names: List[str]
    objective: Any
    integrality: Any
    lower_bounds: Any
    upper_bounds: Any
    constraint_matrix: Any
    constraint_lhs: Any
    constraint_rhs: Any
    constraint_names: List[str]
    variable_groups: Dict[str, Any]
    objective_terms: pd.DataFrame


@dataclass
class Stage4Solution:
    """Solved Stage 4 SAA MILP solution and reporting tables."""

    success: bool
    status: int
    status_message: str
    objective_value: float | None
    mip_gap: float | None
    solve_seconds: float
    variables: pd.DataFrame
    first_stage_decisions: pd.DataFrame
    scenario_selected_component_routes: pd.DataFrame
    scenario_assembly_plan: pd.DataFrame
    scenario_inventory_trajectory: pd.DataFrame
    scenario_capacity_utilization: pd.DataFrame
    objective_breakdown: Dict[str, float]
    summary_metrics: Dict[str, Any]
    baseline_comparison: Dict[str, Any]
    solution_checks: List[Dict[str, Any]] = field(default_factory=list)

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["variables"] = self.variables.to_dict(orient="records")
        payload["first_stage_decisions"] = self.first_stage_decisions.to_dict(orient="records")
        payload["scenario_selected_component_routes"] = self.scenario_selected_component_routes.to_dict(orient="records")
        payload["scenario_assembly_plan"] = self.scenario_assembly_plan.to_dict(orient="records")
        payload["scenario_inventory_trajectory"] = self.scenario_inventory_trajectory.to_dict(orient="records")
        payload["scenario_capacity_utilization"] = self.scenario_capacity_utilization.to_dict(orient="records")
        return payload
