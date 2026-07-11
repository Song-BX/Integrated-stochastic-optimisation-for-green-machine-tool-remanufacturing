"""Public data structures for Stage 5 risk-averse optimisation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict

import pandas as pd

from stage4_stochastic.structures import Stage4Instance, Stage4ModelData, Stage4Solution


@dataclass
class Stage5Instance(Stage4Instance):
    """A Stage 4 SAA instance augmented with reliability and risk data."""

    min_system_reliability: float = 0.0
    chance_alpha: float = 0.95
    stage4_route_candidate_count: int = 0
    component_route_reliability: pd.DataFrame = field(default_factory=pd.DataFrame)
    component_route_risk: pd.DataFrame = field(default_factory=pd.DataFrame)
    chance_constraint_report: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_summary_dict(self) -> Dict[str, Any]:
        payload = super().to_summary_dict()
        payload.update(
            {
                "min_system_reliability": self.min_system_reliability,
                "chance_alpha": self.chance_alpha,
                "stage4_route_candidate_count": self.stage4_route_candidate_count,
                "stage5_route_candidate_count": int(len(self.component_route_period_scenario_table)),
                "chance_pass_rate": (
                    float(len(self.component_route_period_scenario_table)) / float(self.stage4_route_candidate_count)
                    if self.stage4_route_candidate_count
                    else None
                ),
            }
        )
        return payload


@dataclass
class Stage5ModelData(Stage4ModelData):
    """Matrix data with CVaR loss-expression metadata."""

    scenario_loss_terms: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class Stage5Solution(Stage4Solution):
    """Solved Stage 5 CVaR/chance-constrained MILP solution."""

    scenario_risk_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    cvar_summary: Dict[str, Any] = field(default_factory=dict)
    chance_constraint_report: pd.DataFrame = field(default_factory=pd.DataFrame)
    stage4_comparison: Dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["variables"] = self.variables.to_dict(orient="records")
        payload["first_stage_decisions"] = self.first_stage_decisions.to_dict(orient="records")
        payload["scenario_selected_component_routes"] = self.scenario_selected_component_routes.to_dict(orient="records")
        payload["scenario_assembly_plan"] = self.scenario_assembly_plan.to_dict(orient="records")
        payload["scenario_inventory_trajectory"] = self.scenario_inventory_trajectory.to_dict(orient="records")
        payload["scenario_capacity_utilization"] = self.scenario_capacity_utilization.to_dict(orient="records")
        payload["scenario_risk_metrics"] = self.scenario_risk_metrics.to_dict(orient="records")
        payload["chance_constraint_report"] = self.chance_constraint_report.to_dict(orient="records")
        return payload

