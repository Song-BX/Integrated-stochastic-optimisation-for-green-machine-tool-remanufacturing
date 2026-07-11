"""Configuration for Stage 8 ALNS + restricted MILP repair runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

from stage4_stochastic.config import DEFAULT_SCENARIOS


TOP5_52W_MACHINE_TYPES = ("CK6150", "CK6140", "VMC850", "VMC650", "CAK5085")


@dataclass(frozen=True)
class Stage8Config:
    """Runtime configuration for Stage 8 matheuristic runs."""

    raw_dir: Path = Path("data/raw")
    stage1_report: Path = Path("data/processed/stage1/validation_report.json")
    processed_dir: Path = Path("data/processed/stage8")
    results_dir: Path = Path("data/results/stage8")
    stage4_results_dir: Path = Path("data/results/stage4")
    stage5_results_dir: Path = Path("data/results/stage5")
    stage6_results_dir: Path = Path("data/results/stage6")
    stage7_results_dir: Path = Path("data/results/stage7")
    machine_type_id: str = "CK6150"
    period_start: str = "T0001"
    period_count: int = 52
    processing_window_periods: int = 8
    scenario_mode: str = "macro_representative_9"
    scenario_ids: Tuple[str, ...] = field(default_factory=lambda: DEFAULT_SCENARIOS)
    heuristic_method: str = "alns_milp_repair"
    pareto_mode: str = "approximate_augmented_epsilon"
    epsilon_grid_size: int = 3
    max_iterations: int = 24
    repair_time_limit: float = 20.0
    no_improve_limit: int = 8
    random_seed: int = 202607
    benchmark_suite: str | None = None
    benchmark_machine_types: Tuple[str, ...] = field(default_factory=lambda: TOP5_52W_MACHINE_TYPES)
    initial_routes_per_component: int = 1
    max_routes_per_component: int = 3
    route_expand_fraction: float = 0.18
    initial_pairs_per_requirement: int = 12
    pair_expand_count: int = 18
    pair_expand_fraction: float = 0.20
    initial_candidate_fraction: float = 0.55
    candidate_expand_fraction: float = 0.15
    payoff_cost_allowance: float = 0.30
    augmentation_delta_factor: float = 1e-4
    mip_rel_gap: float = 1e-4
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

    def resolved(self, root: Path) -> "Stage8Config":
        """Return a config with relative paths resolved below the project root."""

        return Stage8Config(
            raw_dir=_resolve(root, self.raw_dir),
            stage1_report=_resolve(root, self.stage1_report),
            processed_dir=_resolve(root, self.processed_dir),
            results_dir=_resolve(root, self.results_dir),
            stage4_results_dir=_resolve(root, self.stage4_results_dir),
            stage5_results_dir=_resolve(root, self.stage5_results_dir),
            stage6_results_dir=_resolve(root, self.stage6_results_dir),
            stage7_results_dir=_resolve(root, self.stage7_results_dir),
            machine_type_id=self.machine_type_id,
            period_start=self.period_start,
            period_count=self.period_count,
            processing_window_periods=self.processing_window_periods,
            scenario_mode=self.scenario_mode,
            scenario_ids=tuple(self.scenario_ids),
            heuristic_method=self.heuristic_method,
            pareto_mode=self.pareto_mode,
            epsilon_grid_size=self.epsilon_grid_size,
            max_iterations=self.max_iterations,
            repair_time_limit=self.repair_time_limit,
            no_improve_limit=self.no_improve_limit,
            random_seed=self.random_seed,
            benchmark_suite=self.benchmark_suite,
            benchmark_machine_types=tuple(self.benchmark_machine_types),
            initial_routes_per_component=self.initial_routes_per_component,
            max_routes_per_component=self.max_routes_per_component,
            route_expand_fraction=self.route_expand_fraction,
            initial_pairs_per_requirement=self.initial_pairs_per_requirement,
            pair_expand_count=self.pair_expand_count,
            pair_expand_fraction=self.pair_expand_fraction,
            initial_candidate_fraction=self.initial_candidate_fraction,
            candidate_expand_fraction=self.candidate_expand_fraction,
            payoff_cost_allowance=self.payoff_cost_allowance,
            augmentation_delta_factor=self.augmentation_delta_factor,
            mip_rel_gap=self.mip_rel_gap,
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
        )


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (root / path).resolve()
