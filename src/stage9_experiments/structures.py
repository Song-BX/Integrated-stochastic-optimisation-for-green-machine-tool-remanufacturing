"""Public data structures for Stage 9 experiment suites."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import pandas as pd


@dataclass
class ExperimentSpec:
    """One planned or collected experiment."""

    experiment_id: str
    experiment_group: str
    model_stage: str
    description: str
    source_type: str = "existing_result"
    source_path: str | None = None
    command: str | None = None
    required: bool = False
    status: str = "planned"
    notes: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentResult:
    """Standardized metric row for one experiment."""

    experiment_id: str
    experiment_group: str
    model_stage: str
    description: str
    source_path: str | None
    status: str
    success: bool | None = None
    machine_type_id: str | None = None
    period_start: str | None = None
    period_count: int | None = None
    scenario_count: int | None = None
    variable_count: int | None = None
    constraint_count: int | None = None
    objective_value: float | None = None
    economic_risk: float | None = None
    environmental_impact: float | None = None
    assembly_quality_loss: float | None = None
    expected_assembled_units: float | None = None
    expected_final_backlog_units: float | None = None
    cvar_value: float | None = None
    eta: float | None = None
    worst_scenario_loss: float | None = None
    expected_assembly_shortfall_units: float | None = None
    route_mix_summary: str | None = None
    solve_seconds: float | None = None
    wall_seconds: float | None = None
    pareto_points: int | None = None
    feasible_repairs: int | None = None
    warning: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentSuiteResult:
    """Complete Stage 9 output tables and checks."""

    success: bool
    status_message: str
    manifest: pd.DataFrame
    metric_dictionary: Dict[str, Any]
    all_experiment_results: pd.DataFrame
    baseline_comparison: pd.DataFrame
    ablation_study: pd.DataFrame
    saa_stability: pd.DataFrame
    sensitivity_summary: pd.DataFrame
    exact_vs_matheuristic_gap: pd.DataFrame
    top5_benchmark_summary: pd.DataFrame
    experiment_checks: List[Dict[str, Any]] = field(default_factory=list)

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for name in [
            "manifest",
            "all_experiment_results",
            "baseline_comparison",
            "ablation_study",
            "saa_stability",
            "sensitivity_summary",
            "exact_vs_matheuristic_gap",
            "top5_benchmark_summary",
        ]:
            value = getattr(self, name)
            payload[name] = value.to_dict(orient="records") if isinstance(value, pd.DataFrame) else value
        return payload
