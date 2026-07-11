"""Run the Stage 2 deterministic MILP base model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and solve the Stage 2 deterministic MILP base model.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Directory containing raw CSV files.")
    parser.add_argument(
        "--stage1-report",
        type=Path,
        default=Path("data/processed/stage1/validation_report.json"),
        help="Stage 1 validation report JSON path.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/stage2"),
        help="Directory for Stage 2 processed model artefacts.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("data/results/stage2"),
        help="Directory for Stage 2 solution reports.",
    )
    parser.add_argument("--machine-type", default="CK6150", help="Machine type to solve as a smoke-test family.")
    parser.add_argument("--period-start", default=None, help="Optional starting period id, e.g. T0001.")
    parser.add_argument("--period-end", default=None, help="Optional ending period id, e.g. T0314.")
    parser.add_argument("--baseline-rule", default="BR02", help="baseline_rules.csv rule id for comparison.")
    parser.add_argument("--time-limit", type=float, default=60.0, help="MILP time limit in seconds.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))

    from stage2_deterministic.aggregation import build_stage2_instance
    from stage2_deterministic.config import Stage2Config
    from stage2_deterministic.io_utils import read_stage2_tables, require_stage1_passed
    from stage2_deterministic.model import build_model_data
    from stage2_deterministic.reporting import write_stage2_reports
    from stage2_deterministic.solver import solve_model

    config = Stage2Config(
        raw_dir=args.raw_dir,
        stage1_report=args.stage1_report,
        processed_dir=args.processed_dir,
        results_dir=args.results_dir,
        machine_type_id=args.machine_type,
        planning_period_start=args.period_start,
        planning_period_end=args.period_end,
        baseline_rule_id=args.baseline_rule,
        time_limit_seconds=args.time_limit,
    ).resolved(root)

    stage1_payload = require_stage1_passed(config.stage1_report)
    tables = read_stage2_tables(config.raw_dir)
    instance = build_stage2_instance(tables, config)
    model_data = build_model_data(instance, config)
    solution = solve_model(instance, model_data, config, tables)
    paths = write_stage2_reports(instance, model_data, solution, config)

    print(
        json.dumps(
            {
                "stage1_summary": stage1_payload.get("summary", {}),
                "machine_type_id": config.machine_type_id,
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
