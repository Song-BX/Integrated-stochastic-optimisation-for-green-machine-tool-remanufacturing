"""Stage 1 raw-data scan and validation CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan and validate Stage 1 remanufacturing datasets.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Directory containing raw CSV files.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/stage1"), help="Directory for generated reports.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    if not args.raw_dir.is_absolute():
        raw_dir = (root / args.raw_dir).resolve()
    else:
        raw_dir = args.raw_dir
    if not args.out_dir.is_absolute():
        out_dir = (root / args.out_dir).resolve()
    else:
        out_dir = args.out_dir

    sys.path.insert(0, str(root / "src"))

    from stage1_data.catalogue import scan_catalogue
    from stage1_data.reporting import write_all_reports
    from stage1_data.validators import validate_all, summarize_issues

    entries = scan_catalogue(raw_dir)
    issues = validate_all(raw_dir, entries)
    report_paths = write_all_reports(out_dir, entries, issues)
    summary = summarize_issues(issues)

    print(json.dumps(
        {
            "raw_dir": str(raw_dir),
            "out_dir": str(out_dir),
            "summary": summary,
            "report_paths": report_paths,
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

