"""Run Stage 10 targeted model strengthening."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage 10 pair-carbon and shared-capacity strengthening.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--stage1-report", type=Path, default=Path("data/processed/stage1/validation_report.json"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/stage10"))
    parser.add_argument("--results-dir", type=Path, default=Path("data/results/stage10"))
    parser.add_argument("--stage4-results-dir", type=Path, default=Path("data/results/stage4"))
    parser.add_argument("--stage5-results-dir", type=Path, default=Path("data/results/stage5"))
    parser.add_argument("--stage6-results-dir", type=Path, default=Path("data/results/stage6"))
    parser.add_argument("--machine-types", nargs="+", default=["CK6150", "CK6140"])
    parser.add_argument("--period-start", default="T0001")
    parser.add_argument("--period-count", type=int, default=26)
    parser.add_argument("--processing-window-periods", type=int, default=8)
    parser.add_argument("--time-limit", type=float, default=120.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))

    from stage10_strengthening.config import Stage10Config
    from stage10_strengthening.runner import run_stage10

    config = Stage10Config(
        raw_dir=args.raw_dir,
        stage1_report=args.stage1_report,
        processed_dir=args.processed_dir,
        results_dir=args.results_dir,
        stage4_results_dir=args.stage4_results_dir,
        stage5_results_dir=args.stage5_results_dir,
        stage6_results_dir=args.stage6_results_dir,
        machine_types=tuple(args.machine_types),
        period_start=args.period_start,
        period_count=args.period_count,
        processing_window_periods=args.processing_window_periods,
        time_limit_seconds=args.time_limit,
    ).resolved(root)

    result, paths = run_stage10(config)
    print(
        json.dumps(
            {
                "success": result.success,
                "status_message": result.status_message,
                "machine_types": list(config.machine_types),
                "pair_carbon_summary": result.pair_carbon_summary.to_dict(),
                "shared_capacity_solution_summary": result.shared_capacity_solution_summary,
                "checks": result.checks,
                "report_paths": paths,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
