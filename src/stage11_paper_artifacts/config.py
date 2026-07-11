"""Configuration for Stage 11 paper artifact generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class Stage11Config:
    """Runtime configuration for manuscript-ready table and figure artifacts."""

    stage1_report: Path = Path("data/processed/stage1/validation_report.json")
    data_processed_dir: Path = Path("data/processed")
    data_results_dir: Path = Path("data/results")
    processed_dir: Path = Path("data/processed/stage11")
    results_dir: Path = Path("data/results/stage11")
    profile: str = "manuscript"
    execution_mode: str = "collect-existing"
    figure_backend: str = "matplotlib"
    figure_formats: Tuple[str, ...] = ("png", "svg", "pdf")
    table_formats: Tuple[str, ...] = ("csv", "md", "tex")
    dpi: int = 600
    language: str = "en"
    stage1_gate_required: bool = True
    machine_type_id: str = "CK6150"
    currency_label: str = "RMB"
    random_seed: int = 202607
    core_tables: Tuple[str, ...] = field(
        default_factory=lambda: (
            "T1_stagewise_model_complexity",
            "T2_baseline_and_ablation",
            "T3_pareto_payoff_and_representatives",
            "T4_exact_vs_matheuristic_and_top5",
            "T5_risk_selective_assembly_metrics",
            "T6_stage10_strengthening",
            "T7_saa_sensitivity_manifest",
        )
    )
    core_figures: Tuple[str, ...] = field(
        default_factory=lambda: (
            "F1_model_architecture",
            "F2_data_to_model_pipeline",
            "F3_pareto_tradeoff_panels",
            "F4_baseline_ablation_comparison",
            "F5_matheuristic_convergence",
            "F6_exact_vs_matheuristic_top5",
            "F7_route_mix_and_operational_shift",
            "F8_stage10_strengthening",
        )
    )

    def resolved(self, root: Path) -> "Stage11Config":
        """Resolve relative paths below the project root."""

        return Stage11Config(
            stage1_report=_resolve(root, self.stage1_report),
            data_processed_dir=_resolve(root, self.data_processed_dir),
            data_results_dir=_resolve(root, self.data_results_dir),
            processed_dir=_resolve(root, self.processed_dir),
            results_dir=_resolve(root, self.results_dir),
            profile=self.profile,
            execution_mode=self.execution_mode,
            figure_backend=self.figure_backend,
            figure_formats=tuple(self.figure_formats),
            table_formats=tuple(self.table_formats),
            dpi=int(self.dpi),
            language=self.language,
            stage1_gate_required=bool(self.stage1_gate_required),
            machine_type_id=self.machine_type_id,
            currency_label=self.currency_label,
            random_seed=int(self.random_seed),
            core_tables=tuple(self.core_tables),
            core_figures=tuple(self.core_figures),
        )


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (root / path).resolve()

