"""Report writers for Stage 1 scan and validation outputs."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .catalogue import CatalogueEntry, catalogue_totals
from .io_utils import ensure_dir
from .schema_rules import EXPECTED_FILES, EXPECTED_ROW_COUNTS, EXPECTED_TOTAL_ROWS, EXPECTED_TOTAL_SIZE_MB, FOREIGN_KEYS, PRIMARY_KEYS
from .validators import FAIL, PASS, WARNING, ValidationIssue, summarize_issues


def write_catalogue_csv(entries: Sequence[CatalogueEntry], path: Path) -> None:
    ensure_dir(path.parent)
    fieldnames = list(entries[0].to_dict().keys()) if entries else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry.to_dict())


def write_schema_summary(entries: Sequence[CatalogueEntry], path: Path) -> None:
    ensure_dir(path.parent)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_total_rows": EXPECTED_TOTAL_ROWS,
        "expected_total_size_mb": EXPECTED_TOTAL_SIZE_MB,
        "catalogue_totals": catalogue_totals(entries),
        "expected_files": EXPECTED_FILES,
        "expected_row_counts": EXPECTED_ROW_COUNTS,
        "primary_keys": {file_name: list(columns) for file_name, columns in PRIMARY_KEYS.items()},
        "foreign_keys": [
            {
                "name": rule.name,
                "source_file": rule.source_file,
                "source_columns": list(rule.source_columns),
                "target_file": rule.target_file,
                "target_columns": list(rule.target_columns),
            }
            for rule in FOREIGN_KEYS
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_validation_json(issues: Sequence[ValidationIssue], path: Path) -> None:
    ensure_dir(path.parent)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summarize_issues(issues),
        "issues": [issue.to_dict() for issue in issues],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_failures_csv(issues: Sequence[ValidationIssue], path: Path) -> None:
    ensure_dir(path.parent)
    fieldnames = list(ValidationIssue("", "", "", "").to_dict().keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for issue in issues:
            if issue.severity in {FAIL, WARNING}:
                writer.writerow(issue.to_dict())


def write_validation_markdown(entries: Sequence[CatalogueEntry], issues: Sequence[ValidationIssue], path: Path) -> None:
    ensure_dir(path.parent)
    summary = summarize_issues(issues)
    totals = catalogue_totals(entries)
    lines: List[str] = [
        "# Stage 1 Validation Report",
        "",
        f"Generated at UTC: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Summary",
        "",
        f"- Files scanned: `{totals['existing_file_count']}` / `{totals['file_count']}`",
        f"- Total rows: `{totals['total_rows']}`",
        f"- Total size: `{totals['total_size_mb']} MB`",
        f"- Passed checks: `{summary.get(PASS, 0)}`",
        f"- Warnings: `{summary.get(WARNING, 0)}`",
        f"- Failed checks: `{summary.get(FAIL, 0)}`",
        "",
        "## Catalogue",
        "",
        "| File | Rows | Columns | Size MB | SHA256 prefix |",
        "|---|---:|---:|---:|---|",
    ]
    for entry in entries:
        sha_prefix = entry.sha256[:12] if entry.sha256 else ""
        lines.append(f"| `{entry.file_name}` | {entry.row_count} | {entry.column_count} | {entry.size_mb} | `{sha_prefix}` |")

    notable = [issue for issue in issues if issue.severity in {FAIL, WARNING}]
    lines.extend(["", "## Warnings And Failures", ""])
    if not notable:
        lines.append("No warnings or failures.")
    else:
        lines.append("| Severity | Check | File | Message | Observed | Expected |")
        lines.append("|---|---|---|---|---|---|")
        for issue in notable:
            lines.append(
                "| "
                + " | ".join(
                    [
                        issue.severity,
                        issue.check_name,
                        f"`{issue.file_name}`",
                        _escape_md(issue.message),
                        _escape_md(issue.observed_value or ""),
                        _escape_md(issue.expected_value or ""),
                    ]
                )
                + " |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _escape_md(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_all_reports(out_dir: Path, entries: Sequence[CatalogueEntry], issues: Sequence[ValidationIssue]) -> Dict[str, str]:
    ensure_dir(out_dir)
    paths = {
        "data_catalogue": out_dir / "data_catalogue.csv",
        "schema_summary": out_dir / "schema_summary.json",
        "validation_report_json": out_dir / "validation_report.json",
        "validation_report_md": out_dir / "validation_report.md",
        "validation_failures": out_dir / "validation_failures.csv",
    }
    write_catalogue_csv(entries, paths["data_catalogue"])
    write_schema_summary(entries, paths["schema_summary"])
    write_validation_json(issues, paths["validation_report_json"])
    write_validation_markdown(entries, issues, paths["validation_report_md"])
    write_failures_csv(issues, paths["validation_failures"])
    return {name: str(path) for name, path in paths.items()}

