"""Public data structures for Stage 6 selective assembly."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict

import pandas as pd

from stage5_risk_averse.structures import Stage5Instance, Stage5ModelData, Stage5Solution


@dataclass
class Stage6Instance(Stage5Instance):
    """A Stage 5 risk-averse instance augmented with selective-assembly pools."""

    assembly_time_granularity: str = "scenario_total"
    pairwise_mode: str = "sparse_exact"
    candidate_pool_mode: str = "stage5_consistent_with_new_backup"
    assembly_requirements: pd.DataFrame = field(default_factory=pd.DataFrame)
    assembly_candidate_pool: pd.DataFrame = field(default_factory=pd.DataFrame)
    assembly_pair_pool: pd.DataFrame = field(default_factory=pd.DataFrame)
    assembly_pool_summary: Dict[str, Any] = field(default_factory=dict)

    def to_summary_dict(self) -> Dict[str, Any]:
        payload = super().to_summary_dict()
        payload.update(
            {
                "assembly_time_granularity": self.assembly_time_granularity,
                "pairwise_mode": self.pairwise_mode,
                "candidate_pool_mode": self.candidate_pool_mode,
                "assembly_requirement_count": int(len(self.assembly_requirements)),
                "assembly_candidate_count": int(len(self.assembly_candidate_pool)),
                "assembly_pair_count": int(len(self.assembly_pair_pool)),
                "assembly_pool_summary": self.assembly_pool_summary,
            }
        )
        return payload


@dataclass
class Stage6ModelData(Stage5ModelData):
    """Matrix data with selective-assembly expression metadata."""

    assembly_loss_terms: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class Stage6Solution(Stage5Solution):
    """Solved Stage 6 selective-assembly MILP solution."""

    selected_assembly_candidates: pd.DataFrame = field(default_factory=pd.DataFrame)
    selected_assembly_pairs: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_assembly_plan: pd.DataFrame = field(default_factory=pd.DataFrame)
    dimension_chain_report: pd.DataFrame = field(default_factory=pd.DataFrame)
    assembly_quality_loss_report: pd.DataFrame = field(default_factory=pd.DataFrame)
    scenario_assembly_risk_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    stage5_comparison: Dict[str, Any] = field(default_factory=dict)
    assembly_baseline_comparison: Dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for name in [
            "variables",
            "first_stage_decisions",
            "scenario_selected_component_routes",
            "scenario_assembly_plan",
            "scenario_inventory_trajectory",
            "scenario_capacity_utilization",
            "scenario_risk_metrics",
            "chance_constraint_report",
            "selected_assembly_candidates",
            "selected_assembly_pairs",
            "feature_assembly_plan",
            "dimension_chain_report",
            "assembly_quality_loss_report",
            "scenario_assembly_risk_metrics",
        ]:
            value = getattr(self, name)
            payload[name] = value.to_dict(orient="records") if isinstance(value, pd.DataFrame) else value
        return payload
