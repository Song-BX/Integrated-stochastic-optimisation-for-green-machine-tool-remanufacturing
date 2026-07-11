"""Public data structures for Stage 7 Pareto analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import pandas as pd

from stage6_selective_assembly.structures import Stage6Instance, Stage6ModelData


@dataclass
class Stage7Instance(Stage6Instance):
    """Stage 6 instance with Stage 7 multi-objective metadata."""

    multiobjective_method: str = "augmented_epsilon_constraint"

    def to_summary_dict(self) -> Dict[str, Any]:
        payload = super().to_summary_dict()
        payload["multiobjective_method"] = self.multiobjective_method
        return payload


@dataclass
class Stage7ModelData(Stage6ModelData):
    """Stage 6 matrix with objective vectors for Pareto analysis."""

    objective_vectors: Dict[str, Any] = field(default_factory=dict)
    objective_vector_summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Stage7Solution:
    """Solved Stage 7 payoff/grid/Pareto results."""

    success: bool
    status_message: str
    solve_seconds: float
    payoff_table: pd.DataFrame
    epsilon_grid: pd.DataFrame
    grid_solution_summary: pd.DataFrame
    pareto_front: pd.DataFrame
    dominated_solutions: pd.DataFrame
    representative_solutions: Dict[str, Any]
    solution_checks: List[Dict[str, Any]]
    model_summary: Dict[str, Any]
    stage6_comparison: Dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for name in ["payoff_table", "epsilon_grid", "grid_solution_summary", "pareto_front", "dominated_solutions"]:
            value = getattr(self, name)
            payload[name] = value.to_dict(orient="records")
        return payload
