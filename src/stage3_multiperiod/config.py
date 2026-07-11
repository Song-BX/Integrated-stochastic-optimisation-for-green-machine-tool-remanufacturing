"""Configuration for the Stage 3 multi-period deterministic MILP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Stage3Config:
    """Runtime configuration for the Stage 3 rolling-window model."""

    raw_dir: Path = Path("data/raw")
    stage1_report: Path = Path("data/processed/stage1/validation_report.json")
    processed_dir: Path = Path("data/processed/stage3")
    results_dir: Path = Path("data/results/stage3")
    machine_type_id: str = "CK6150"
    period_start: str = "T0001"
    period_count: int = 52
    baseline_rule_id: str = "BR02"
    env_weight: float = 120.0
    quality_weight: float = 5000.0
    reliability_weight: float = 0.55
    inventory_holding_rate: float = 0.012
    backlog_penalty_rmb_per_unit_period: float | None = None
    procurement_cost_multiplier: float = 1.12
    overtime_penalty_rmb_per_h: float = 220.0
    capacity_share_floor: float = 0.03
    capacity_share_multiplier: float = 1.0
    quality_floor: float = 0.62
    life_constraint_ratio: float = 1.0
    time_limit_seconds: float = 60.0
    mip_rel_gap: float = 1e-4

    def resolved(self, root: Path) -> "Stage3Config":
        """Return a config with relative paths resolved below the project root."""

        return Stage3Config(
            raw_dir=_resolve(root, self.raw_dir),
            stage1_report=_resolve(root, self.stage1_report),
            processed_dir=_resolve(root, self.processed_dir),
            results_dir=_resolve(root, self.results_dir),
            machine_type_id=self.machine_type_id,
            period_start=self.period_start,
            period_count=self.period_count,
            baseline_rule_id=self.baseline_rule_id,
            env_weight=self.env_weight,
            quality_weight=self.quality_weight,
            reliability_weight=self.reliability_weight,
            inventory_holding_rate=self.inventory_holding_rate,
            backlog_penalty_rmb_per_unit_period=self.backlog_penalty_rmb_per_unit_period,
            procurement_cost_multiplier=self.procurement_cost_multiplier,
            overtime_penalty_rmb_per_h=self.overtime_penalty_rmb_per_h,
            capacity_share_floor=self.capacity_share_floor,
            capacity_share_multiplier=self.capacity_share_multiplier,
            quality_floor=self.quality_floor,
            life_constraint_ratio=self.life_constraint_ratio,
            time_limit_seconds=self.time_limit_seconds,
            mip_rel_gap=self.mip_rel_gap,
        )


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (root / path).resolve()
