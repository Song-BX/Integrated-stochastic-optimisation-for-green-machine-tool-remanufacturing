"""Run Stage 9 experiment-suite collection/orchestration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect and optionally run Stage 9 experiment-suite results.")
    parser.add_argument("--stage1-report", type=Path, default=Path("data/processed/stage1/validation_report.json"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/stage9"))
    parser.add_argument("--results-dir", type=Path, default=Path("data/results/stage9"))
    parser.add_argument("--data-results-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--data-processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--profile", default="smoke", choices=["smoke", "quick-run", "full-manifest"])
    parser.add_argument("--execution-mode", default="collect-existing", choices=["collect-existing", "run"])
    parser.add_argument("--machine-type", default="CK6150")
    parser.add_argument("--period-start", default="T0001")
    parser.add_argument("--period-count", type=int, default=52)
    parser.add_argument("--processing-window-periods", type=int, default=8)
    parser.add_argument("--run-epsilon-grid-size", type=int, default=2)
    parser.add_argument("--run-max-iterations", type=int, default=2)
    parser.add_argument("--run-repair-time-limit", type=float, default=5.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))

    from stage9_experiments.config import Stage9Config
    from stage9_experiments.suite import run_stage9_suite

    config = Stage9Config(
        stage1_report=args.stage1_report,
        processed_dir=args.processed_dir,
        results_dir=args.results_dir,
        data_results_dir=args.data_results_dir,
        data_processed_dir=args.data_processed_dir,
        profile=args.profile,
        execution_mode=args.execution_mode,
        machine_type_id=args.machine_type,
        period_start=args.period_start,
        period_count=args.period_count,
        processing_window_periods=args.processing_window_periods,
        run_epsilon_grid_size=args.run_epsilon_grid_size,
        run_max_iterations=args.run_max_iterations,
        run_repair_time_limit=args.run_repair_time_limit,
    ).resolved(root)
    result, paths, metadata = run_stage9_suite(config, root)
    print(
        json.dumps(
            {
                **metadata,
                "profile": config.profile,
                "execution_mode": config.execution_mode,
                "manifest_rows": len(result.manifest),
                "collected_rows": int((result.all_experiment_results["status"] == "collected").sum())
                if not result.all_experiment_results.empty
                else 0,
                "baseline_rows": len(result.baseline_comparison),
                "ablation_rows": len(result.ablation_study),
                "saa_rows": len(result.saa_stability),
                "sensitivity_rows": len(result.sensitivity_summary),
                "top5_rows": len(result.top5_benchmark_summary),
                "success": result.success,
                "status_message": result.status_message,
                "experiment_checks": result.experiment_checks,
                "report_paths": paths,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
