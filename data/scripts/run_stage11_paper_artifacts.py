"""Run Stage 11 manuscript table and figure artifact generation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Stage 11 manuscript-ready tables and figures.")
    parser.add_argument("--stage1-report", type=Path, default=Path("data/processed/stage1/validation_report.json"))
    parser.add_argument("--data-processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--data-results-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/stage11"))
    parser.add_argument("--results-dir", type=Path, default=Path("data/results/stage11"))
    parser.add_argument("--profile", default="manuscript")
    parser.add_argument("--execution-mode", default="collect-existing")
    parser.add_argument("--figure-backend", default="matplotlib")
    parser.add_argument("--figure-formats", nargs="+", default=["png", "svg", "pdf"])
    parser.add_argument("--table-formats", nargs="+", default=["csv", "md", "tex"])
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--language", default="en")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("MPLCONFIGDIR", str(root / "data" / "processed" / "stage11" / ".matplotlib"))
    sys.path.insert(0, str(root / "src"))

    from stage11_paper_artifacts.config import Stage11Config
    from stage11_paper_artifacts.runner import run_stage11

    config = Stage11Config(
        stage1_report=args.stage1_report,
        data_processed_dir=args.data_processed_dir,
        data_results_dir=args.data_results_dir,
        processed_dir=args.processed_dir,
        results_dir=args.results_dir,
        profile=args.profile,
        execution_mode=args.execution_mode,
        figure_backend=args.figure_backend,
        figure_formats=tuple(args.figure_formats),
        table_formats=tuple(args.table_formats),
        dpi=args.dpi,
        language=args.language,
    ).resolved(root)
    result, paths = run_stage11(config)
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
