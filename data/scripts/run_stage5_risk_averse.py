"""Run the Stage 5 chance-constrained CVaR SAA MILP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and solve the Stage 5 risk-averse SAA MILP.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--stage1-report", type=Path, default=Path("data/processed/stage1/validation_report.json"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/stage5"))
    parser.add_argument("--results-dir", type=Path, default=Path("data/results/stage5"))
    parser.add_argument("--stage4-results-dir", type=Path, default=Path("data/results/stage4"))
    parser.add_argument("--machine-type", default="CK6150")
    parser.add_argument("--period-start", default="T0001")
    parser.add_argument("--period-count", type=int, default=52)
    parser.add_argument("--processing-window-periods", type=int, default=8)
    parser.add_argument("--baseline-rule", default="BR14")
    parser.add_argument("--risk-baseline-rule", default="BR14")
    parser.add_argument("--cvar-confidence", type=float, default=0.95)
    parser.add_argument("--cvar-lambda", type=float, default=0.22)
    parser.add_argument("--chance-alpha", type=float, default=0.95)
    parser.add_argument("--min-system-reliability", type=float, default=None)
    parser.add_argument("--time-limit", type=float, default=180.0)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))

    from stage5_risk_averse.aggregation import build_stage5_instance
    from stage5_risk_averse.config import Stage5Config
    from stage5_risk_averse.io_utils import read_stage5_tables, require_stage1_passed
    from stage5_risk_averse.model import build_model_data
    from stage5_risk_averse.reporting import write_stage5_reports
    from stage5_risk_averse.solver import solve_model

    config = Stage5Config(
        raw_dir=args.raw_dir,
        stage1_report=args.stage1_report,
        processed_dir=args.processed_dir,
        results_dir=args.results_dir,
        stage4_results_dir=args.stage4_results_dir,
        machine_type_id=args.machine_type,
        period_start=args.period_start,
        period_count=args.period_count,
        processing_window_periods=args.processing_window_periods,
        baseline_rule_id=args.baseline_rule,
        risk_baseline_rule_id=args.risk_baseline_rule,
        cvar_confidence=args.cvar_confidence,
        cvar_lambda=args.cvar_lambda,
        chance_alpha=args.chance_alpha,
        min_system_reliability=args.min_system_reliability,
        time_limit_seconds=args.time_limit,
    ).resolved(root)

    stage1_payload = require_stage1_passed(config.stage1_report)
    tables = read_stage5_tables(config.raw_dir)
    instance = build_stage5_instance(tables, config)
    model_data = build_model_data(instance, config)
    solution = solve_model(instance, model_data, config, tables)
    paths = write_stage5_reports(instance, model_data, solution, config)

    print(
        json.dumps(
            {
                "stage1_summary": stage1_payload.get("summary", {}),
                "machine_type_id": config.machine_type_id,
                "period_start": config.period_start,
                "period_count": config.period_count,
                "processing_window_periods": config.processing_window_periods,
                "cvar_confidence": config.cvar_confidence,
                "cvar_lambda": config.cvar_lambda,
                "success": solution.success,
                "status": solution.status,
                "objective_value": solution.objective_value,
                "summary_metrics": solution.summary_metrics,
                "cvar_summary": solution.cvar_summary,
                "baseline_comparison": solution.baseline_comparison,
                "stage4_comparison": solution.stage4_comparison,
                "report_paths": paths,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if solution.success else 2


if __name__ == "__main__":
    raise SystemExit(main())

