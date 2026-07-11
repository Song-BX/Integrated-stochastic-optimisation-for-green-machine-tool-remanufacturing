"""Public data structures for Stage 3 multi-period optimisation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import pandas as pd


@dataclass
class Stage3Instance:
    """A deterministic rolling-window component-routing instance."""

    machine_type_id: str
    machine_family: str
    periods: List[str]
    component_types: List[str]
    route_ids: List[str]
    resource_types: List[str]
    demand_units: int
    candidate_core_count: int
    component_instance_count: int
    min_required_life_h: float
    target_quality_score: float
    machine_summary: Dict[str, Any]
    period_demand: pd.DataFrame
    bom_requirements: pd.DataFrame
    core_summary: pd.DataFrame
    component_summary: pd.DataFrame
    component_route_period_table: pd.DataFrame
    initial_inventory: pd.DataFrame
    procurement_costs: pd.DataFrame
    capacity_table: pd.DataFrame

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "machine_type_id": self.machine_type_id,
            "machine_family": self.machine_family,
            "period_start": self.periods[0] if self.periods else None,
            "period_end": self.periods[-1] if self.periods else None,
            "period_count": len(self.periods),
            "component_types": self.component_types,
            "route_ids": self.route_ids,
            "resource_types": self.resource_types,
            "demand_units": self.demand_units,
            "candidate_core_count": self.candidate_core_count,
            "component_instance_count": self.component_instance_count,
            "min_required_life_h": self.min_required_life_h,
            "target_quality_score": self.target_quality_score,
            "machine_summary": self.machine_summary,
        }


@dataclass
class Stage3ModelData:
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
class Stage3Solution:
    """Solved Stage 3 MILP solution and derived reporting metrics."""

    success: bool
    status: int
    status_message: str
    objective_value: float | None
    mip_gap: float | None
    solve_seconds: float
    variables: pd.DataFrame
    selected_component_routes: pd.DataFrame
    assembly_plan: pd.DataFrame
    inventory_trajectory: pd.DataFrame
    capacity_utilization: pd.DataFrame
    objective_breakdown: Dict[str, float]
    summary_metrics: Dict[str, Any]
    baseline_comparison: Dict[str, Any]
    solution_checks: List[Dict[str, Any]] = field(default_factory=list)

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["variables"] = self.variables.to_dict(orient="records")
        payload["selected_component_routes"] = self.selected_component_routes.to_dict(orient="records")
        payload["assembly_plan"] = self.assembly_plan.to_dict(orient="records")
        payload["inventory_trajectory"] = self.inventory_trajectory.to_dict(orient="records")
        payload["capacity_utilization"] = self.capacity_utilization.to_dict(orient="records")
        return payload
