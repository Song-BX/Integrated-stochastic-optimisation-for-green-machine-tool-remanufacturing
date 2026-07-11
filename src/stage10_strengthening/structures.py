"""Public data structures for Stage 10 strengthening outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import pandas as pd


@dataclass
class PairCarbonSummary:
    """Summary of assembly pair-carbon coefficients in the environmental objective."""

    machine_type_id: str
    period_start: str
    period_count: int
    pair_coefficient_count: int
    pair_nonzero_coefficient_count: int
    environmental_nonzero_before: int
    environmental_nonzero_after: int
    total_weighted_pair_carbon: float
    mean_pair_carbon_kg: float
    max_pair_carbon_kg: float
    finite_objective_vector: bool

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class SharedCapacityInstance:
    """Collection of single-machine Stage 4 instances used in the coupling experiment."""

    machine_types: Tuple[str, ...]
    periods: List[str]
    scenario_ids: List[str]
    shared_resources: Tuple[str, ...]
    instances: Dict[str, Any]
    model_data_by_machine: Dict[str, Any]

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "machine_types": list(self.machine_types),
            "period_start": self.periods[0] if self.periods else None,
            "period_end": self.periods[-1] if self.periods else None,
            "period_count": len(self.periods),
            "scenario_ids": self.scenario_ids,
            "scenario_count": len(self.scenario_ids),
            "shared_resources": list(self.shared_resources),
            "machine_summaries": {
                machine_type: instance.to_summary_dict()
                for machine_type, instance in self.instances.items()
            },
        }


@dataclass
class SharedCapacityModelData:
    """MILP matrix data for the shared-capacity coupled Stage 4 extension."""

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
    machine_offsets: Dict[str, int]
    machine_variable_slices: Dict[str, Tuple[int, int]]
    shared_overtime: Dict[str, int]
    shared_capacity_rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    shared_capacity_terms: Dict[str, Dict[int, float]] = field(default_factory=dict)

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "variable_count": int(len(self.variable_names)),
            "constraint_count": int(len(self.constraint_names)),
            "shared_capacity_row_count": int(len(self.shared_capacity_rows)),
            "shared_overtime_variable_count": int(len(self.shared_overtime)),
            "binary_variable_count": int((self.integrality == 1).sum()),
            "integer_variable_count": int((self.integrality == 1).sum()),
            "continuous_variable_count": int((self.integrality == 0).sum()),
            "machine_offsets": self.machine_offsets,
            "machine_variable_slices": self.machine_variable_slices,
        }


@dataclass
class Stage10Result:
    """All Stage 10 generated tables and summaries."""

    success: bool
    status_message: str
    pair_carbon_mapping: pd.DataFrame
    environmental_objective_breakdown: pd.DataFrame
    pair_carbon_summary: PairCarbonSummary
    shared_capacity_instance_summary: Dict[str, Any]
    shared_capacity_model_summary: Dict[str, Any]
    shared_capacity_solution_summary: Dict[str, Any]
    shared_capacity_comparison: pd.DataFrame
    shared_capacity_utilization: pd.DataFrame
    checks: List[Dict[str, Any]] = field(default_factory=list)
