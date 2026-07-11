"""Run Stage 12 final experiment completion and figure audit."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage 12 finalization.")
    parser.add_argument("--stage1-report", type=Path, default=Path("data/processed/stage1/validation_report.json"))
    parser.add_argument("--data-processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--data-results-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/stage12"))
    parser.add_argument("--results-dir", type=Path, default=Path("data/results/stage12"))
    parser.add_argument("--profile", default="manuscript")
    parser.add_argument("--execution-mode", choices=["complete-and-audit", "audit-only", "complete-missing"], default="complete-and-audit")
    parser.add_argument("--machine-type", default="CK6150")
    parser.add_argument("--period-start", default="T0001")
    parser.add_argument("--period-count", type=int, default=52)
    parser.add_argument("--processing-window-periods", type=int, default=8)
    parser.add_argument("--figure-backend", default="matplotlib")
    parser.add_argument("--figure-formats", nargs="+", default=["png", "svg", "pdf"])
    parser.add_argument("--table-formats", nargs="+", default=["csv", "md", "tex"])
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--quick-epsilon-grid-size", type=int, default=2)
    parser.add_argument("--quick-max-iterations", type=int, default=2)
    parser.add_argument("--quick-repair-time-limit", type=float, default=5.0)
    parser.add_argument("--quick-saa-time-limit", type=float, default=90.0)
    parser.add_argument("--quick-sensitivity-time-limit", type=float, default=90.0)
    parser.add_argument("--completion-time-budget", type=float, default=600.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("MPLCONFIGDIR", str(root / "data" / "processed" / "stage12" / ".matplotlib"))
    sys.path.insert(0, str(root / "src"))

    from stage12_finalization.config import Stage12Config
    from stage12_finalization.runner import run_stage12

    config = Stage12Config(
        stage1_report=args.stage1_report,
        data_processed_dir=args.data_processed_dir,
        data_results_dir=args.data_results_dir,
        processed_dir=args.processed_dir,
        results_dir=args.results_dir,
        profile=args.profile,
        execution_mode=args.execution_mode,
        machine_type_id=args.machine_type,
        period_start=args.period_start,
        period_count=args.period_count,
        processing_window_periods=args.processing_window_periods,
        figure_backend=args.figure_backend,
        figure_formats=tuple(args.figure_formats),
        table_formats=tuple(args.table_formats),
        dpi=args.dpi,
        quick_epsilon_grid_size=args.quick_epsilon_grid_size,
        quick_max_iterations=args.quick_max_iterations,
        quick_repair_time_limit=args.quick_repair_time_limit,
        quick_saa_time_limit=args.quick_saa_time_limit,
        quick_sensitivity_time_limit=args.quick_sensitivity_time_limit,
        completion_time_budget_seconds=args.completion_time_budget,
    ).resolved(root)
    result, paths = run_stage12(config, root)
    print(
        json.dumps(
            {
                "success": result.success,
                "status_message": result.status_message,
                "summary": result.to_summary_dict(),
                "paths": paths,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
