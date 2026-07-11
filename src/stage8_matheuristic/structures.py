"""Public data structures for Stage 8 matheuristic search."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd

from stage7_pareto.structures import Stage7Instance, Stage7ModelData


@dataclass
class Stage8Instance(Stage7Instance):
    """Stage 7 instance with Stage 8 heuristic metadata."""

    heuristic_method: str = "alns_milp_repair"
    pareto_mode: str = "approximate_augmented_epsilon"

    def to_summary_dict(self) -> Dict[str, Any]:
        payload = super().to_summary_dict()
        payload["heuristic_method"] = self.heuristic_method
        payload["pareto_mode"] = self.pareto_mode
        return payload


@dataclass
class Stage8ModelData(Stage7ModelData):
    """Stage 7 matrix with Stage 8 variable-pool metadata."""

    restriction_summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Stage8HeuristicState:
    """Current candidate pools used to restrict repeated MILP repairs."""

    allowed_route_keys: Set[str] = field(default_factory=set)
    allowed_candidate_keys: Set[str] = field(default_factory=set)
    allowed_pair_keys: Set[str] = field(default_factory=set)
    operator_scores: Dict[str, float] = field(default_factory=dict)
    operator_uses: Dict[str, int] = field(default_factory=dict)
    operator_successes: Dict[str, int] = field(default_factory=dict)

    def copy(self) -> "Stage8HeuristicState":
        return Stage8HeuristicState(
            allowed_route_keys=set(self.allowed_route_keys),
            allowed_candidate_keys=set(self.allowed_candidate_keys),
            allowed_pair_keys=set(self.allowed_pair_keys),
            operator_scores=dict(self.operator_scores),
            operator_uses=dict(self.operator_uses),
            operator_successes=dict(self.operator_successes),
        )


@dataclass
class Stage8Solution:
    """One restricted MILP repair result."""

    solution_id: str
    grid_id: str
    operator_name: str
    iteration: int
    success: bool
    status: int
    message: str
    feasible: bool
    solve_seconds: float
    objective_value: float | None
    economic_risk: float
    environmental_impact: float
    assembly_quality_loss: float
    slack_env: float
    slack_assembly: float
    stage6_constraint_max_residual: float
    env_constraint_residual: float
    assembly_constraint_residual: float
    allowed_route_count: int
    allowed_candidate_count: int
    allowed_pair_count: int
    x: np.ndarray | None = None

    def summary_row(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload.pop("x", None)
        return payload


@dataclass
class Stage8RunResult:
    """Complete Stage 8 heuristic run output."""

    success: bool
    status_message: str
    solve_seconds: float
    instance_summary: Dict[str, Any]
    heuristic_config: Dict[str, Any]
    initial_restriction_summary: pd.DataFrame
    operator_catalogue: pd.DataFrame
    benchmark_instances: pd.DataFrame
    model_summary: Dict[str, Any]
    iteration_log: pd.DataFrame
    repair_solve_log: pd.DataFrame
    incumbent_solution_summary: Dict[str, Any]
    approx_pareto_front: pd.DataFrame
    dominated_solutions: pd.DataFrame
    operator_scores: pd.DataFrame
    large_benchmark_summary: pd.DataFrame
    solution_checks: List[Dict[str, Any]]
    stage7_comparison: Dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for name in [
            "initial_restriction_summary",
            "operator_catalogue",
            "benchmark_instances",
            "iteration_log",
            "repair_solve_log",
            "approx_pareto_front",
            "dominated_solutions",
            "operator_scores",
            "large_benchmark_summary",
        ]:
            value = getattr(self, name)
            payload[name] = value.to_dict(orient="records") if isinstance(value, pd.DataFrame) else value
        return payload
