"""Configuration for Stage 10 targeted model-strengthening runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

from stage4_stochastic.config import DEFAULT_SCENARIOS


@dataclass(frozen=True)
class Stage10Config:
    """Runtime configuration for pair-carbon and shared-capacity checks."""

    raw_dir: Path = Path("data/raw")
    stage1_report: Path = Path("data/processed/stage1/validation_report.json")
    processed_dir: Path = Path("data/processed/stage10")
    results_dir: Path = Path("data/results/stage10")
    stage4_results_dir: Path = Path("data/results/stage4")
    stage5_results_dir: Path = Path("data/results/stage5")
    stage6_results_dir: Path = Path("data/results/stage6")
    machine_types: Tuple[str, ...] = ("CK6150", "CK6140")
    period_start: str = "T0001"
    period_count: int = 26
    processing_window_periods: int = 8
    scenario_mode: str = "macro_representative_9"
    scenario_ids: Tuple[str, ...] = field(default_factory=lambda: DEFAULT_SCENARIOS)
    shared_resources: Tuple[str, ...] = ("machining", "grinding", "inspection", "assembly", "testing", "laser")
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
    time_limit_seconds: float = 120.0
    mip_rel_gap: float = 1e-4

    def resolved(self, root: Path) -> "Stage10Config":
        """Resolve relative paths below the project root."""

        return Stage10Config(
            raw_dir=_resolve(root, self.raw_dir),
            stage1_report=_resolve(root, self.stage1_report),
            processed_dir=_resolve(root, self.processed_dir),
            results_dir=_resolve(root, self.results_dir),
            stage4_results_dir=_resolve(root, self.stage4_results_dir),
            stage5_results_dir=_resolve(root, self.stage5_results_dir),
            stage6_results_dir=_resolve(root, self.stage6_results_dir),
            machine_types=tuple(self.machine_types),
            period_start=self.period_start,
            period_count=self.period_count,
            processing_window_periods=self.processing_window_periods,
            scenario_mode=self.scenario_mode,
            scenario_ids=tuple(self.scenario_ids),
            shared_resources=tuple(self.shared_resources),
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

