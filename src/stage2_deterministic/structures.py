"""Public data structures for Stage 2 deterministic optimisation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import pandas as pd


@dataclass
class Stage2Instance:
    """A deterministic aggregate planning snapshot for one machine family."""

    machine_type_id: str
    machine_family: str
    demand_units: int
    candidate_core_count: int
    bom_item_count: int
    average_required_component_count: float
    min_required_life_h: float
    target_quality_score: float
    capacity_by_resource_h: Dict[str, float]
    route_classes: List[str]
    route_class_to_route_ids: Dict[str, List[str]]
    core_route_table: pd.DataFrame
    core_summary: pd.DataFrame
    route_coefficients: pd.DataFrame
    machine_summary: Dict[str, Any] = field(default_factory=dict)
    period_filter: Dict[str, str | None] = field(default_factory=dict)

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "machine_type_id": self.machine_type_id,
            "machine_family": self.machine_family,
            "demand_units": self.demand_units,
            "candidate_core_count": self.candidate_core_count,
            "bom_item_count": self.bom_item_count,
            "average_required_component_count": self.average_required_component_count,
            "min_required_life_h": self.min_required_life_h,
            "target_quality_score": self.target_quality_score,
            "capacity_by_resource_h": self.capacity_by_resource_h,
            "route_classes": self.route_classes,
            "route_class_to_route_ids": self.route_class_to_route_ids,
            "machine_summary": self.machine_summary,
            "period_filter": self.period_filter,
        }


@dataclass
class Stage2ModelData:
    """Matrix model data passed to scipy.optimize.milp."""

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
    route_assignments: pd.DataFrame


@dataclass
class Stage2Solution:
    """Solved deterministic MILP solution and derived reporting metrics."""

    success: bool
    status: int
    status_message: str
    objective_value: float | None
    mip_gap: float | None
    solve_seconds: float
    variables: pd.DataFrame
    selected_routes: pd.DataFrame
    objective_breakdown: Dict[str, float]
    summary_metrics: Dict[str, Any]
    capacity_utilization: pd.DataFrame
    baseline_comparison: Dict[str, Any]
    solution_checks: List[Dict[str, Any]]

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["variables"] = self.variables.to_dict(orient="records")
        payload["selected_routes"] = self.selected_routes.to_dict(orient="records")
        payload["capacity_utilization"] = self.capacity_utilization.to_dict(orient="records")
        return payload
