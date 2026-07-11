"""Configuration for Stage 4 stochastic SAA MILP runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


DEFAULT_SCENARIOS: Tuple[str, ...] = (
    "S001",
    "S004",
    "S009",
    "S012",
    "S013",
    "S018",
    "S019",
    "S022",
    "S025",
)


def scenario_ids_for_count(scenarios: object, count: int) -> Tuple[str, ...]:
    """Select representative scenario ids for a requested SAA sample size."""

    if count == len(DEFAULT_SCENARIOS):
        return DEFAULT_SCENARIOS
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - pandas is already required by Stage 4.
        raise RuntimeError("pandas is required to select scenario-count profiles.") from exc
    if not isinstance(scenarios, pd.DataFrame):
        raise TypeError("scenario_ids_for_count expects a pandas DataFrame.")
    if count <= 0:
        raise ValueError("scenario count must be positive.")
    data = scenarios.copy()
    data["scenario_probability"] = pd.to_numeric(data["scenario_probability"], errors="coerce").fillna(0.0)
    if count >= len(data):
        selected = data.sort_values(["scenario_id"]).head(count)
    else:
        rows = []
        if "macro_group" in data.columns:
            for _group, group_frame in data.groupby("macro_group", dropna=False):
                rows.append(group_frame.sort_values(["scenario_probability", "scenario_id"], ascending=[False, True]).head(1))
        seeded = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=data.columns)
        remaining = data[~data["scenario_id"].isin(set(seeded["scenario_id"]))].copy()
        fill_count = max(0, count - len(seeded))
        fill = remaining.sort_values(["scenario_probability", "scenario_id"], ascending=[False, True]).head(fill_count)
        selected = pd.concat([seeded, fill], ignore_index=True)
    selected = selected.sort_values(["scenario_id"]).head(count)
    ids = tuple(str(value) for value in selected["scenario_id"].tolist())
    if len(ids) != count:
        raise ValueError(f"Unable to select {count} scenarios; selected {len(ids)}.")
    return ids


@dataclass(frozen=True)
class Stage4Config:
    """Runtime configuration for the Stage 4 SAA deterministic equivalent."""

    raw_dir: Path = Path("data/raw")
    stage1_report: Path = Path("data/processed/stage1/validation_report.json")
    processed_dir: Path = Path("data/processed/stage4")
    results_dir: Path = Path("data/results/stage4")
    machine_type_id: str = "CK6150"
    period_start: str = "T0001"
    period_count: int = 52
    processing_window_periods: int = 8
    scenario_mode: str = "macro_representative_9"
    scenario_ids: Tuple[str, ...] = field(default_factory=lambda: DEFAULT_SCENARIOS)
    baseline_rule_id: str = "BR02"
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
    time_limit_seconds: float = 120.0
    mip_rel_gap: float = 1e-4

    def resolved(self, root: Path) -> "Stage4Config":
        """Return a config with relative paths resolved below the project root."""

        return Stage4Config(
            raw_dir=_resolve(root, self.raw_dir),
            stage1_report=_resolve(root, self.stage1_report),
            processed_dir=_resolve(root, self.processed_dir),
            results_dir=_resolve(root, self.results_dir),
            machine_type_id=self.machine_type_id,
            period_start=self.period_start,
            period_count=self.period_count,
            processing_window_periods=self.processing_window_periods,
            scenario_mode=self.scenario_mode,
            scenario_ids=tuple(self.scenario_ids),
            baseline_rule_id=self.baseline_rule_id,
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
            time_limit_seconds=self.time_limit_seconds,
            mip_rel_gap=self.mip_rel_gap,
        )


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (root / path).resolve()
