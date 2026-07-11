"""Run the Stage 4 stochastic SAA MILP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and solve the Stage 4 stochastic SAA MILP.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--stage1-report", type=Path, default=Path("data/processed/stage1/validation_report.json"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/stage4"))
    parser.add_argument("--results-dir", type=Path, default=Path("data/results/stage4"))
    parser.add_argument("--machine-type", default="CK6150")
    parser.add_argument("--period-start", default="T0001")
    parser.add_argument("--period-count", type=int, default=52)
    parser.add_argument("--processing-window-periods", type=int, default=8)
    parser.add_argument("--scenario-count", type=int, default=None)
    parser.add_argument("--scenario-ids", nargs="+", default=None)
    parser.add_argument("--baseline-rule", default="BR02")
    parser.add_argument("--time-limit", type=float, default=120.0)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))

    from stage4_stochastic.aggregation import build_stage4_instance
    from stage4_stochastic.config import DEFAULT_SCENARIOS, Stage4Config, scenario_ids_for_count
    from stage4_stochastic.io_utils import read_stage4_tables, require_stage1_passed
    from stage4_stochastic.model import build_model_data
    from stage4_stochastic.reporting import write_stage4_reports
    from stage4_stochastic.solver import solve_model

    raw_config = Stage4Config(raw_dir=args.raw_dir).resolved(root)
    scenario_ids = tuple(args.scenario_ids) if args.scenario_ids else DEFAULT_SCENARIOS
    scenario_mode = "macro_representative_9"
    if args.scenario_count is not None:
        scenario_tables = read_stage4_tables(raw_config.raw_dir)
        scenario_ids = scenario_ids_for_count(scenario_tables["scenarios"], args.scenario_count)
        scenario_mode = f"macro_probability_representative_{args.scenario_count}"

    config = Stage4Config(
        raw_dir=args.raw_dir,
        stage1_report=args.stage1_report,
        processed_dir=args.processed_dir,
        results_dir=args.results_dir,
        machine_type_id=args.machine_type,
        period_start=args.period_start,
        period_count=args.period_count,
        processing_window_periods=args.processing_window_periods,
        scenario_mode=scenario_mode,
        scenario_ids=scenario_ids,
        baseline_rule_id=args.baseline_rule,
        time_limit_seconds=args.time_limit,
    ).resolved(root)

    stage1_payload = require_stage1_passed(config.stage1_report)
    tables = read_stage4_tables(config.raw_dir)
    instance = build_stage4_instance(tables, config)
    model_data = build_model_data(instance, config)
    solution = solve_model(instance, model_data, config, tables)
    paths = write_stage4_reports(instance, model_data, solution, config)

    print(
        json.dumps(
            {
                "stage1_summary": stage1_payload.get("summary", {}),
                "machine_type_id": config.machine_type_id,
                "period_start": config.period_start,
                "period_count": config.period_count,
                "processing_window_periods": config.processing_window_periods,
                "scenario_mode": config.scenario_mode,
                "scenario_ids": config.scenario_ids,
                "success": solution.success,
                "status": solution.status,
                "objective_value": solution.objective_value,
                "summary_metrics": solution.summary_metrics,
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
