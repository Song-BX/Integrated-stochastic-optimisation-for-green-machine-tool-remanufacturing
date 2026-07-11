"""Run the Stage 3 multi-period deterministic MILP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and solve the Stage 3 multi-period deterministic MILP.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--stage1-report", type=Path, default=Path("data/processed/stage1/validation_report.json"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/stage3"))
    parser.add_argument("--results-dir", type=Path, default=Path("data/results/stage3"))
    parser.add_argument("--machine-type", default="CK6150")
    parser.add_argument("--period-start", default="T0001")
    parser.add_argument("--period-count", type=int, default=52)
    parser.add_argument("--baseline-rule", default="BR02")
    parser.add_argument("--time-limit", type=float, default=60.0)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))

    from stage3_multiperiod.aggregation import build_stage3_instance
    from stage3_multiperiod.config import Stage3Config
    from stage3_multiperiod.io_utils import read_stage3_tables, require_stage1_passed
    from stage3_multiperiod.model import build_model_data
    from stage3_multiperiod.reporting import write_stage3_reports
    from stage3_multiperiod.solver import solve_model

    config = Stage3Config(
        raw_dir=args.raw_dir,
        stage1_report=args.stage1_report,
        processed_dir=args.processed_dir,
        results_dir=args.results_dir,
        machine_type_id=args.machine_type,
        period_start=args.period_start,
        period_count=args.period_count,
        baseline_rule_id=args.baseline_rule,
        time_limit_seconds=args.time_limit,
    ).resolved(root)

    stage1_payload = require_stage1_passed(config.stage1_report)
    tables = read_stage3_tables(config.raw_dir)
    instance = build_stage3_instance(tables, config)
    model_data = build_model_data(instance, config)
    solution = solve_model(instance, model_data, config, tables)
    paths = write_stage3_reports(instance, model_data, solution, config)

    print(
        json.dumps(
            {
                "stage1_summary": stage1_payload.get("summary", {}),
                "machine_type_id": config.machine_type_id,
                "period_start": config.period_start,
                "period_count": config.period_count,
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
