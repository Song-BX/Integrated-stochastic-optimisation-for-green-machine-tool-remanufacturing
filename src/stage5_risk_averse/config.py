"""Configuration for Stage 5 risk-averse SAA MILP runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

from stage4_stochastic.config import DEFAULT_SCENARIOS


@dataclass(frozen=True)
class Stage5Config:
    """Runtime configuration for the Stage 5 CVaR/chance-constrained model."""

    raw_dir: Path = Path("data/raw")
    stage1_report: Path = Path("data/processed/stage1/validation_report.json")
    processed_dir: Path = Path("data/processed/stage5")
    results_dir: Path = Path("data/results/stage5")
    stage4_results_dir: Path = Path("data/results/stage4")
    machine_type_id: str = "CK6150"
    period_start: str = "T0001"
    period_count: int = 52
    processing_window_periods: int = 8
    scenario_mode: str = "macro_representative_9"
    scenario_ids: Tuple[str, ...] = field(default_factory=lambda: DEFAULT_SCENARIOS)
    baseline_rule_id: str = "BR14"
    risk_baseline_rule_id: str = "BR14"
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
    time_limit_seconds: float = 180.0
    mip_rel_gap: float = 1e-4

    def resolved(self, root: Path) -> "Stage5Config":
        """Return a config with relative paths resolved below the project root."""

        return Stage5Config(
            raw_dir=_resolve(root, self.raw_dir),
            stage1_report=_resolve(root, self.stage1_report),
            processed_dir=_resolve(root, self.processed_dir),
            results_dir=_resolve(root, self.results_dir),
            stage4_results_dir=_resolve(root, self.stage4_results_dir),
            machine_type_id=self.machine_type_id,
            period_start=self.period_start,
            period_count=self.period_count,
            processing_window_periods=self.processing_window_periods,
            scenario_mode=self.scenario_mode,
            scenario_ids=tuple(self.scenario_ids),
            baseline_rule_id=self.baseline_rule_id,
            risk_baseline_rule_id=self.risk_baseline_rule_id,
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
            time_limit_seconds=self.time_limit_seconds,
            mip_rel_gap=self.mip_rel_gap,
        )


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (root / path).resolve()

