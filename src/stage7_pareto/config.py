"""Configuration for Stage 7 Pareto analysis runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

from stage4_stochastic.config import DEFAULT_SCENARIOS


@dataclass(frozen=True)
class Stage7Config:
    """Runtime configuration for the Stage 7 augmented epsilon-constraint model."""

    raw_dir: Path = Path("data/raw")
    stage1_report: Path = Path("data/processed/stage1/validation_report.json")
    processed_dir: Path = Path("data/processed/stage7")
    results_dir: Path = Path("data/results/stage7")
    stage4_results_dir: Path = Path("data/results/stage4")
    stage5_results_dir: Path = Path("data/results/stage5")
    stage6_results_dir: Path = Path("data/results/stage6")
    machine_type_id: str = "CK6150"
    period_start: str = "T0001"
    period_count: int = 52
    processing_window_periods: int = 8
    scenario_mode: str = "macro_representative_9"
    scenario_ids: Tuple[str, ...] = field(default_factory=lambda: DEFAULT_SCENARIOS)
    baseline_rule_id: str = "BR14"
    risk_baseline_rule_id: str = "BR14"
    selective_assembly_baseline_rule_id: str = "BR08"
    no_selective_assembly_ablation_rule_id: str = "BR18"
    multiobjective_method: str = "augmented_epsilon_constraint"
    primary_objective: str = "economic_risk"
    epsilon_objective_env: str = "environmental_impact"
    epsilon_objective_assembly: str = "assembly_quality_loss"
    epsilon_grid_size_env: int = 5
    epsilon_grid_size_assembly: int = 5
    payoff_cost_allowance: float = 0.30
    augmentation_delta_factor: float = 1e-4
    env_weight: float = 120.0
    quality_weight: float = 5000.0
    reliability_weight: float = 0.55
    inventory_holding_rate: float = 0.012
    backlog_penalty_rmb_per_unit_period: float | None = None
    procurement_cost_multiplier: float = 1.12
    recourse_procurement_premium: float = 1.18
    overtime_penalty_rmb_per_h: float = 220.0
    capacity_share_floor: float = 0.03
    capacity_share_multiplier: float = 1.0
    quality_floor: float = 0.62
    cvar_confidence: float = 0.95
    cvar_lambda: float = 0.22
    chance_alpha: float = 0.95
    min_system_reliability: float | None = None
    dimension_penalty_weight: float = 8000.0
    assembly_quality_loss_weight: float = 20000.0
    life_gap_penalty_weight: float = 0.08
    compatibility_penalty_weight: float = 12000.0
    soft_pair_penalty_rmb: float = 35000.0
    assembly_shortfall_penalty_rmb: float = 250000.0
    assembly_risk_weight: float = 0.22
    risk_budget_reference_rmb: float = 81805.27
    time_limit_per_solve: float = 120.0
    mip_rel_gap: float = 1e-4

    def resolved(self, root: Path) -> "Stage7Config":
        """Return a config with relative paths resolved below the project root."""

        return Stage7Config(
            raw_dir=_resolve(root, self.raw_dir),
            stage1_report=_resolve(root, self.stage1_report),
            processed_dir=_resolve(root, self.processed_dir),
            results_dir=_resolve(root, self.results_dir),
            stage4_results_dir=_resolve(root, self.stage4_results_dir),
            stage5_results_dir=_resolve(root, self.stage5_results_dir),
            stage6_results_dir=_resolve(root, self.stage6_results_dir),
            machine_type_id=self.machine_type_id,
            period_start=self.period_start,
            period_count=self.period_count,
            processing_window_periods=self.processing_window_periods,
            scenario_mode=self.scenario_mode,
            scenario_ids=tuple(self.scenario_ids),
            baseline_rule_id=self.baseline_rule_id,
            risk_baseline_rule_id=self.risk_baseline_rule_id,
            selective_assembly_baseline_rule_id=self.selective_assembly_baseline_rule_id,
            no_selective_assembly_ablation_rule_id=self.no_selective_assembly_ablation_rule_id,
            multiobjective_method=self.multiobjective_method,
            primary_objective=self.primary_objective,
            epsilon_objective_env=self.epsilon_objective_env,
            epsilon_objective_assembly=self.epsilon_objective_assembly,
            epsilon_grid_size_env=self.epsilon_grid_size_env,
            epsilon_grid_size_assembly=self.epsilon_grid_size_assembly,
            payoff_cost_allowance=self.payoff_cost_allowance,
            augmentation_delta_factor=self.augmentation_delta_factor,
            env_weight=self.env_weight,
            quality_weight=self.quality_weight,
            reliability_weight=self.reliability_weight,
            inventory_holding_rate=self.inventory_holding_rate,
            backlog_penalty_rmb_per_unit_period=self.backlog_penalty_rmb_per_unit_period,
            procurement_cost_multiplier=self.procurement_cost_multiplier,
            recourse_procurement_premium=self.recourse_procurement_premium,
            overtime_penalty_rmb_per_h=self.overtime_penalty_rmb_per_h,
            capacity_share_floor=self.capacity_share_floor,
            capacity_share_multiplier=self.capacity_share_multiplier,
            quality_floor=self.quality_floor,
            cvar_confidence=self.cvar_confidence,
            cvar_lambda=self.cvar_lambda,
            chance_alpha=self.chance_alpha,
            min_system_reliability=self.min_system_reliability,
            dimension_penalty_weight=self.dimension_penalty_weight,
            assembly_quality_loss_weight=self.assembly_quality_loss_weight,
            life_gap_penalty_weight=self.life_gap_penalty_weight,
            compatibility_penalty_weight=self.compatibility_penalty_weight,
            soft_pair_penalty_rmb=self.soft_pair_penalty_rmb,
            assembly_shortfall_penalty_rmb=self.assembly_shortfall_penalty_rmb,
            assembly_risk_weight=self.assembly_risk_weight,
            risk_budget_reference_rmb=self.risk_budget_reference_rmb,
            time_limit_per_solve=self.time_limit_per_solve,
            mip_rel_gap=self.mip_rel_gap,
        )


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (root / path).resolve()
