"""Run Stage 7 augmented epsilon-constraint Pareto analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and solve Stage 7 Pareto analysis.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--stage1-report", type=Path, default=Path("data/processed/stage1/validation_report.json"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/stage7"))
    parser.add_argument("--results-dir", type=Path, default=Path("data/results/stage7"))
    parser.add_argument("--stage4-results-dir", type=Path, default=Path("data/results/stage4"))
    parser.add_argument("--stage5-results-dir", type=Path, default=Path("data/results/stage5"))
    parser.add_argument("--stage6-results-dir", type=Path, default=Path("data/results/stage6"))
    parser.add_argument("--machine-type", default="CK6150")
    parser.add_argument("--period-start", default="T0001")
    parser.add_argument("--period-count", type=int, default=52)
    parser.add_argument("--processing-window-periods", type=int, default=8)
    parser.add_argument("--epsilon-grid-size", type=int, default=5)
    parser.add_argument("--time-limit", type=float, default=120.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))

    from stage7_pareto.aggregation import build_stage7_instance
    from stage7_pareto.config import Stage7Config
    from stage7_pareto.io_utils import read_stage6_tables, require_stage1_passed
    from stage7_pareto.model import build_model_data
    from stage7_pareto.reporting import write_stage7_reports
    from stage7_pareto.solver import solve_pareto

    config = Stage7Config(
        raw_dir=args.raw_dir,
        stage1_report=args.stage1_report,
        processed_dir=args.processed_dir,
        results_dir=args.results_dir,
        stage4_results_dir=args.stage4_results_dir,
        stage5_results_dir=args.stage5_results_dir,
        stage6_results_dir=args.stage6_results_dir,
        machine_type_id=args.machine_type,
        period_start=args.period_start,
        period_count=args.period_count,
        processing_window_periods=args.processing_window_periods,
        epsilon_grid_size_env=args.epsilon_grid_size,
        epsilon_grid_size_assembly=args.epsilon_grid_size,
        time_limit_per_solve=args.time_limit,
    ).resolved(root)

    stage1_payload = require_stage1_passed(config.stage1_report)
    tables = read_stage6_tables(config.raw_dir)
    instance = build_stage7_instance(tables, config)
    model_data = build_model_data(instance, config, tables)
    solution = solve_pareto(instance, model_data, config)
    paths = write_stage7_reports(instance, model_data, solution, config)
    print(
        json.dumps(
            {
                "stage1_summary": stage1_payload.get("summary", {}),
                "machine_type_id": config.machine_type_id,
                "period_start": config.period_start,
                "period_count": config.period_count,
                "epsilon_grid_rows": len(solution.epsilon_grid),
                "feasible_grid_points": int((solution.grid_solution_summary["feasible"] == 1).sum()),
                "pareto_points": len(solution.pareto_front),
                "success": solution.success,
                "status_message": solution.status_message,
                "solution_checks": solution.solution_checks,
                "report_paths": paths,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if solution.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
