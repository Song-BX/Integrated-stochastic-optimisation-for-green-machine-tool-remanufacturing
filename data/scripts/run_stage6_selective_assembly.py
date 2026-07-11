"""Run the Stage 6 selective-assembly CVaR SAA MILP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and solve the Stage 6 selective-assembly MILP.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--stage1-report", type=Path, default=Path("data/processed/stage1/validation_report.json"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/stage6"))
    parser.add_argument("--results-dir", type=Path, default=Path("data/results/stage6"))
    parser.add_argument("--stage4-results-dir", type=Path, default=Path("data/results/stage4"))
    parser.add_argument("--stage5-results-dir", type=Path, default=Path("data/results/stage5"))
    parser.add_argument("--machine-type", default="CK6150")
    parser.add_argument("--period-start", default="T0001")
    parser.add_argument("--period-count", type=int, default=52)
    parser.add_argument("--processing-window-periods", type=int, default=8)
    parser.add_argument("--baseline-rule", default="BR14")
    parser.add_argument("--risk-baseline-rule", default="BR14")
    parser.add_argument("--selective-assembly-baseline-rule", default="BR08")
    parser.add_argument("--no-selective-assembly-ablation-rule", default="BR18")
    parser.add_argument("--cvar-confidence", type=float, default=0.95)
    parser.add_argument("--cvar-lambda", type=float, default=0.22)
    parser.add_argument("--chance-alpha", type=float, default=0.95)
    parser.add_argument("--min-system-reliability", type=float, default=None)
    parser.add_argument("--time-limit", type=float, default=240.0)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))

    from stage6_selective_assembly.aggregation import build_stage6_instance
    from stage6_selective_assembly.config import Stage6Config
    from stage6_selective_assembly.io_utils import read_stage6_tables, require_stage1_passed
    from stage6_selective_assembly.model import build_model_data
    from stage6_selective_assembly.reporting import write_stage6_reports
    from stage6_selective_assembly.solver import solve_model

    config = Stage6Config(
        raw_dir=args.raw_dir,
        stage1_report=args.stage1_report,
        processed_dir=args.processed_dir,
        results_dir=args.results_dir,
        stage4_results_dir=args.stage4_results_dir,
        stage5_results_dir=args.stage5_results_dir,
        machine_type_id=args.machine_type,
        period_start=args.period_start,
        period_count=args.period_count,
        processing_window_periods=args.processing_window_periods,
        baseline_rule_id=args.baseline_rule,
        risk_baseline_rule_id=args.risk_baseline_rule,
        selective_assembly_baseline_rule_id=args.selective_assembly_baseline_rule,
        no_selective_assembly_ablation_rule_id=args.no_selective_assembly_ablation_rule,
        cvar_confidence=args.cvar_confidence,
        cvar_lambda=args.cvar_lambda,
        chance_alpha=args.chance_alpha,
        min_system_reliability=args.min_system_reliability,
        time_limit_seconds=args.time_limit,
    ).resolved(root)

    stage1_payload = require_stage1_passed(config.stage1_report)
    tables = read_stage6_tables(config.raw_dir)
    instance = build_stage6_instance(tables, config)
    model_data = build_model_data(instance, config)
    solution = solve_model(instance, model_data, config, tables)
    paths = write_stage6_reports(instance, model_data, solution, config)

    print(
        json.dumps(
            {
                "stage1_summary": stage1_payload.get("summary", {}),
                "machine_type_id": config.machine_type_id,
                "period_start": config.period_start,
                "period_count": config.period_count,
                "processing_window_periods": config.processing_window_periods,
                "assembly_pool_summary": instance.assembly_pool_summary,
                "cvar_confidence": config.cvar_confidence,
                "cvar_lambda": config.cvar_lambda,
                "success": solution.success,
                "status": solution.status,
                "objective_value": solution.objective_value,
                "summary_metrics": solution.summary_metrics,
                "cvar_summary": solution.cvar_summary,
                "stage5_comparison": solution.stage5_comparison,
                "baseline_comparison": solution.baseline_comparison,
                "report_paths": paths,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if solution.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
