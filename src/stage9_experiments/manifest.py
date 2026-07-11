"""Experiment manifest construction for Stage 9."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from .config import Stage9Config
from .structures import ExperimentSpec


def build_experiment_manifest(config: Stage9Config) -> pd.DataFrame:
    """Return the Stage 9 manifest as a DataFrame."""

    specs: List[ExperimentSpec] = []
    results = config.data_results_dir
    specs.extend(
        [
            ExperimentSpec(
                "baseline_stage3_deterministic",
                "baseline_comparison",
                "Stage3",
                "Deterministic multi-period, multi-component, multi-route MILP.",
                source_path=str(results / "stage3" / "solution_summary.json"),
                required=False,
            ),
            ExperimentSpec(
                "baseline_stage4_saa",
                "baseline_comparison",
                "Stage4",
                "Two-stage stochastic SAA without CVaR/selective assembly.",
                source_path=str(results / "stage4" / "solution_summary.json"),
                required=True,
            ),
            ExperimentSpec(
                "baseline_stage5_cvar",
                "baseline_comparison",
                "Stage5",
                "Risk-averse SAA with chance constraints and CVaR, without selective assembly.",
                source_path=str(results / "stage5" / "solution_summary.json"),
                required=True,
            ),
            ExperimentSpec(
                "baseline_stage6_selective_assembly",
                "baseline_comparison",
                "Stage6",
                "CVaR SAA with selective assembly and dimension-chain constraints.",
                source_path=str(results / "stage6" / "solution_summary.json"),
                required=True,
            ),
            ExperimentSpec(
                "baseline_stage7_pareto_anchor",
                "baseline_comparison",
                "Stage7",
                "Exact augmented epsilon-constraint Pareto analysis.",
                source_path=str(results / "stage7" / "pareto_front.csv"),
                required=True,
            ),
            ExperimentSpec(
                "baseline_stage8_matheuristic",
                "baseline_comparison",
                "Stage8",
                "ALNS + restricted MILP repair approximate Pareto analysis.",
                source_path=str(results / "stage8" / "approx_pareto_front.csv"),
                required=True,
            ),
        ]
    )
    specs.extend(_rule_baselines(results))
    specs.extend(_ablation_specs(results))
    specs.extend(_saa_specs(results, config))
    specs.extend(_sensitivity_specs(results, config))
    specs.extend(
        [
            ExperimentSpec(
                "exact_vs_matheuristic_stage7_stage8",
                "exact_vs_matheuristic",
                "Stage7_vs_Stage8",
                "Compare exact Stage 7 Pareto front and Stage 8 approximate Pareto front.",
                source_path=str(results / "stage8" / "approx_pareto_front.csv"),
                required=True,
            ),
            ExperimentSpec(
                "large_benchmark_top5_52w",
                "large_benchmark",
                "Stage8",
                "Top-five machine-type independent benchmark summary.",
                source_path=str(results / "stage8" / "large_benchmark_summary.csv"),
                required=True,
            ),
        ]
    )
    return pd.DataFrame([spec.to_dict() for spec in specs])


def metric_dictionary() -> dict[str, object]:
    """Return human-readable metric definitions."""

    return {
        "objective_value": "Original scalar objective reported by the source stage.",
        "economic_risk": "Economic/risk objective used by Stage 7/8 Pareto analysis.",
        "environmental_impact": "Carbon/environmental objective proxy used by Stage 7/8.",
        "assembly_quality_loss": "Selective-assembly and dimension-chain loss objective.",
        "expected_assembled_units": "Probability-weighted assembled units over the scenario set.",
        "expected_final_backlog_units": "Probability-weighted final backlog units.",
        "cvar_value": "CVaR value from Stage 5/6 risk metrics.",
        "eta": "VaR auxiliary variable in CVaR linearization.",
        "worst_scenario_loss": "Worst scenario loss among SAA scenarios.",
        "expected_assembly_shortfall_units": "Probability-weighted feature-level assembly shortfall.",
        "route_mix_summary": "Compact JSON-like route mix summary when available.",
        "pareto_points": "Number of nondominated Pareto points.",
        "feasible_repairs": "Number of feasible Stage 8 restricted MILP repairs.",
    }


def _rule_baselines(results: Path) -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            "rule_br14_risk_aware",
            "baseline_comparison",
            "BR14",
            "BR14 risk-aware rule reference extracted from Stage 6 if available.",
            source_path=str(results / "stage6" / "solution_summary.json"),
            required=False,
        ),
        ExperimentSpec(
            "rule_br08_selective_assembly",
            "baseline_comparison",
            "BR08",
            "BR08 selective-assembly-priority rule reference extracted from Stage 6 if available.",
            source_path=str(results / "stage6" / "solution_summary.json"),
            required=False,
        ),
        ExperimentSpec(
            "rule_br18_no_selective_assembly",
            "baseline_comparison",
            "BR18",
            "BR18 no-selective-assembly ablation rule reference extracted from Stage 6 if available.",
            source_path=str(results / "stage6" / "solution_summary.json"),
            required=False,
        ),
    ]


def _ablation_specs(results: Path) -> list[ExperimentSpec]:
    return [
        ExperimentSpec("ablation_no_stochasticity", "ablation_study", "Stage3_vs_Stage4", "Effect of adding SAA stochasticity.", source_path=str(results / "stage4" / "solution_summary.json")),
        ExperimentSpec("ablation_no_cvar", "ablation_study", "Stage4_vs_Stage5", "Effect of adding chance constraints and CVaR.", source_path=str(results / "stage5" / "solution_summary.json")),
        ExperimentSpec("ablation_no_selective_assembly", "ablation_study", "Stage5_vs_Stage6", "Effect of adding selective assembly.", source_path=str(results / "stage6" / "solution_summary.json")),
        ExperimentSpec("ablation_no_pareto", "ablation_study", "Stage6_vs_Stage7", "Effect of moving from single objective to Pareto analysis.", source_path=str(results / "stage7" / "pareto_front.csv")),
        ExperimentSpec("ablation_matheuristic_approximation", "ablation_study", "Stage7_vs_Stage8", "Effect of replacing exact Pareto solves with matheuristic repairs.", source_path=str(results / "stage8" / "approx_pareto_front.csv")),
    ]


def _saa_specs(results: Path, config: Stage9Config) -> list[ExperimentSpec]:
    specs = [
        ExperimentSpec(
            "saa_scenario_9_existing",
            "saa_stability",
            "Stage4",
            "Existing 9-scenario SAA result.",
            source_path=str(results / "stage4" / "solution_summary.json"),
            required=False,
        )
    ]
    for count in [18, 27]:
        specs.append(
            ExperimentSpec(
                f"saa_scenario_{count}_optional",
                "saa_stability",
                "Stage4",
                f"Optional {count}-scenario SAA run profile.",
                source_type="optional_run",
                source_path=str(config.results_dir / "runs" / f"saa_scenario_{count}" / "solution_summary.json"),
                command=f"run_stage4_stochastic with {count} scenarios",
                required=False,
                notes="Manifest only unless execution-mode run supports this profile.",
            )
        )
    return specs


def _sensitivity_specs(_results: Path, config: Stage9Config) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    values = {
        "cvar_lambda": [0.00, 0.11, 0.22, 0.44],
        "min_system_reliability": [0.90, 0.93, 0.95],
        "env_weight": [60, 120, 240],
        "assembly_shortfall_penalty_rmb": [125000, 250000, 500000],
    }
    for parameter, levels in values.items():
        for value in levels:
            specs.append(
                ExperimentSpec(
                    f"sensitivity_{parameter}_{value}",
                    "sensitivity_analysis",
                    "Stage6_or_Stage8",
                    f"Optional sensitivity run for {parameter}={value}.",
                    source_type="optional_run",
                    source_path=str(config.results_dir / "runs" / f"sensitivity_{parameter}_{value}" / "summary.json"),
                    command=f"run selected stage with {parameter}={value}",
                    required=False,
                    notes="Manifest row by default; run mode may execute selected quick cases.",
                )
            )
    return specs
