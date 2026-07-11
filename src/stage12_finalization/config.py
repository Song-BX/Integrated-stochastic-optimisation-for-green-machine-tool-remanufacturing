"""Configuration for Stage 12 final experiment completion and figure audit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class Stage12Config:
    """Runtime configuration for Stage 12."""

    stage1_report: Path = Path("data/processed/stage1/validation_report.json")
    data_processed_dir: Path = Path("data/processed")
    data_results_dir: Path = Path("data/results")
    processed_dir: Path = Path("data/processed/stage12")
    results_dir: Path = Path("data/results/stage12")
    profile: str = "manuscript"
    execution_mode: str = "complete-and-audit"
    machine_type_id: str = "CK6150"
    period_start: str = "T0001"
    period_count: int = 52
    processing_window_periods: int = 8
    figure_backend: str = "matplotlib"
    figure_formats: Tuple[str, ...] = ("png", "svg", "pdf")
    table_formats: Tuple[str, ...] = ("csv", "md", "tex")
    dpi: int = 600
    quick_epsilon_grid_size: int = 2
    quick_max_iterations: int = 2
    quick_repair_time_limit: float = 5.0
    quick_saa_time_limit: float = 90.0
    quick_sensitivity_time_limit: float = 90.0
    completion_time_budget_seconds: float = 600.0

    def resolved(self, root: Path) -> "Stage12Config":
        """Resolve relative paths below the project root."""

        return Stage12Config(
            stage1_report=_resolve(root, self.stage1_report),
            data_processed_dir=_resolve(root, self.data_processed_dir),
            data_results_dir=_resolve(root, self.data_results_dir),
            processed_dir=_resolve(root, self.processed_dir),
            results_dir=_resolve(root, self.results_dir),
            profile=self.profile,
            execution_mode=self.execution_mode,
            machine_type_id=self.machine_type_id,
            period_start=self.period_start,
            period_count=self.period_count,
            processing_window_periods=self.processing_window_periods,
            figure_backend=self.figure_backend,
            figure_formats=tuple(self.figure_formats),
            table_formats=tuple(self.table_formats),
            dpi=self.dpi,
            quick_epsilon_grid_size=self.quick_epsilon_grid_size,
            quick_max_iterations=self.quick_max_iterations,
            quick_repair_time_limit=self.quick_repair_time_limit,
            quick_saa_time_limit=self.quick_saa_time_limit,
            quick_sensitivity_time_limit=self.quick_sensitivity_time_limit,
            completion_time_budget_seconds=self.completion_time_budget_seconds,
        )


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (root / path).resolve()
