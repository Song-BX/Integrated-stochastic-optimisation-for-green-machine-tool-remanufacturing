"""Configuration for the Stage 2 deterministic MILP run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Stage2Config:
    """Runtime configuration for a deterministic single-period model."""

    raw_dir: Path = Path("data/raw")
    stage1_report: Path = Path("data/processed/stage1/validation_report.json")
    processed_dir: Path = Path("data/processed/stage2")
    results_dir: Path = Path("data/results/stage2")
    machine_type_id: str = "CK6150"
    planning_period_start: str | None = None
    planning_period_end: str | None = None
    baseline_rule_id: str = "BR02"
    env_weight: float = 120.0
    quality_weight: float = 5000.0
    reliability_weight: float = 0.55
    shortage_penalty_rmb: float | None = None
    overtime_penalty_rmb_per_h: float = 180.0
    capacity_share_floor: float = 0.03
    capacity_share_multiplier: float = 1.0
    max_route_classes: int = 5
    life_constraint_ratio: float = 1.0
    quality_floor: float = 0.62
    time_limit_seconds: float = 60.0
    mip_rel_gap: float = 1e-4

    def resolved(self, root: Path) -> "Stage2Config":
        """Return a config with relative paths resolved under the project root."""

        return Stage2Config(
            raw_dir=_resolve(root, self.raw_dir),
            stage1_report=_resolve(root, self.stage1_report),
            processed_dir=_resolve(root, self.processed_dir),
            results_dir=_resolve(root, self.results_dir),
            machine_type_id=self.machine_type_id,
            planning_period_start=self.planning_period_start,
            planning_period_end=self.planning_period_end,
            baseline_rule_id=self.baseline_rule_id,
            env_weight=self.env_weight,
            quality_weight=self.quality_weight,
            reliability_weight=self.reliability_weight,
            shortage_penalty_rmb=self.shortage_penalty_rmb,
            overtime_penalty_rmb_per_h=self.overtime_penalty_rmb_per_h,
            capacity_share_floor=self.capacity_share_floor,
            capacity_share_multiplier=self.capacity_share_multiplier,
            max_route_classes=self.max_route_classes,
            life_constraint_ratio=self.life_constraint_ratio,
            quality_floor=self.quality_floor,
            time_limit_seconds=self.time_limit_seconds,
            mip_rel_gap=self.mip_rel_gap,
        )


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (root / path).resolve()
