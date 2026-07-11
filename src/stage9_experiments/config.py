"""Configuration for Stage 9 experiment-suite runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Stage9Config:
    """Runtime configuration for Stage 9 experiment collection/orchestration."""

    stage1_report: Path = Path("data/processed/stage1/validation_report.json")
    processed_dir: Path = Path("data/processed/stage9")
    results_dir: Path = Path("data/results/stage9")
    data_results_dir: Path = Path("data/results")
    data_processed_dir: Path = Path("data/processed")
    profile: str = "smoke"
    execution_mode: str = "collect-existing"
    machine_type_id: str = "CK6150"
    period_start: str = "T0001"
    period_count: int = 52
    processing_window_periods: int = 8
    run_epsilon_grid_size: int = 2
    run_max_iterations: int = 2
    run_repair_time_limit: float = 5.0

    def resolved(self, root: Path) -> "Stage9Config":
        """Resolve relative paths below the project root."""

        return Stage9Config(
            stage1_report=_resolve(root, self.stage1_report),
            processed_dir=_resolve(root, self.processed_dir),
            results_dir=_resolve(root, self.results_dir),
            data_results_dir=_resolve(root, self.data_results_dir),
            data_processed_dir=_resolve(root, self.data_processed_dir),
            profile=self.profile,
            execution_mode=self.execution_mode,
            machine_type_id=self.machine_type_id,
            period_start=self.period_start,
            period_count=self.period_count,
            processing_window_periods=self.processing_window_periods,
            run_epsilon_grid_size=self.run_epsilon_grid_size,
            run_max_iterations=self.run_max_iterations,
            run_repair_time_limit=self.run_repair_time_limit,
        )


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (root / path).resolve()
