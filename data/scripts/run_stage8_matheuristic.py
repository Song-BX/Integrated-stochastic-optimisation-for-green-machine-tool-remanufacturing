"""Run Stage 8 ALNS + restricted MILP repair matheuristic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and solve Stage 8 matheuristic.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--stage1-report", type=Path, default=Path("data/processed/stage1/validation_report.json"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/stage8"))
    parser.add_argument("--results-dir", type=Path, default=Path("data/results/stage8"))
    parser.add_argument("--stage4-results-dir", type=Path, default=Path("data/results/stage4"))
    parser.add_argument("--stage5-results-dir", type=Path, default=Path("data/results/stage5"))
    parser.add_argument("--stage6-results-dir", type=Path, default=Path("data/results/stage6"))
    parser.add_argument("--stage7-results-dir", type=Path, default=Path("data/results/stage7"))
    parser.add_argument("--machine-type", default="CK6150")
    parser.add_argument("--period-start", default="T0001")
    parser.add_argument("--period-count", type=int, default=52)
    parser.add_argument("--processing-window-periods", type=int, default=8)
    parser.add_argument("--epsilon-grid-size", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=24)
    parser.add_argument("--repair-time-limit", type=float, default=20.0)
    parser.add_argument("--no-improve-limit", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=202607)
    parser.add_argument("--benchmark-suite", choices=["top5_52w"], default=None)
    parser.add_argument("--mip-rel-gap", type=float, default=1e-4)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))

    from stage8_matheuristic.benchmark import run_benchmark_suite, run_single_instance
    from stage8_matheuristic.config import Stage8Config
    from stage8_matheuristic.io_utils import read_stage6_tables, require_stage1_passed

    config = Stage8Config(
        raw_dir=args.raw_dir,
        stage1_report=args.stage1_report,
        processed_dir=args.processed_dir,
        results_dir=args.results_dir,
        stage4_results_dir=args.stage4_results_dir,
        stage5_results_dir=args.stage5_results_dir,
        stage6_results_dir=args.stage6_results_dir,
        stage7_results_dir=args.stage7_results_dir,
        machine_type_id=args.machine_type,
        period_start=args.period_start,
        period_count=args.period_count,
        processing_window_periods=args.processing_window_periods,
        epsilon_grid_size=args.epsilon_grid_size,
        max_iterations=args.max_iterations,
        repair_time_limit=args.repair_time_limit,
        no_improve_limit=args.no_improve_limit,
        random_seed=args.random_seed,
        benchmark_suite=args.benchmark_suite,
        mip_rel_gap=args.mip_rel_gap,
    ).resolved(root)

    stage1_payload = require_stage1_passed(config.stage1_report)
    tables = read_stage6_tables(config.raw_dir)

    if config.benchmark_suite:
        summary, paths = run_benchmark_suite(tables, config)
        success = bool(not summary.empty and summary["success"].astype(bool).any())
        print(
            json.dumps(
                {
                    "stage1_summary": stage1_payload.get("summary", {}),
                    "benchmark_suite": config.benchmark_suite,
                    "benchmark_instances": len(summary),
                    "successful_instances": int(summary["success"].astype(bool).sum()) if not summary.empty else 0,
                    "large_benchmark_summary": summary.to_dict(orient="records"),
                    "report_paths": paths,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if success else 2

    result, paths = run_single_instance(tables, config)
    print(
        json.dumps(
            {
                "stage1_summary": stage1_payload.get("summary", {}),
                "machine_type_id": config.machine_type_id,
                "period_start": config.period_start,
                "period_count": config.period_count,
                "epsilon_grid_size": config.epsilon_grid_size,
                "max_iterations": config.max_iterations,
                "repair_solves": len(result.repair_solve_log),
                "feasible_repair_solves": int((result.repair_solve_log["feasible"] == True).sum()) if not result.repair_solve_log.empty else 0,  # noqa: E712
                "approx_pareto_points": len(result.approx_pareto_front),
                "success": result.success,
                "status_message": result.status_message,
                "solution_checks": result.solution_checks,
                "stage7_comparison": result.stage7_comparison,
                "report_paths": paths,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
