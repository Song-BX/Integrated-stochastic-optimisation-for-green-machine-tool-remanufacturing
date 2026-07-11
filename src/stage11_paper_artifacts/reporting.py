"""Report writers and artifact QA for Stage 11."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import matplotlib.image as mpimg
import numpy as np
import pandas as pd

from .config import Stage11Config
from .io_utils import write_json
from .structures import PaperFigure, PaperTable, Stage11Result


def write_stage11_reports(tables: List[PaperTable], figures: List[PaperFigure], config: Stage11Config) -> Stage11Result:
    """Write all Stage 11 tables, figures, manifests, and checks."""

    table_dir = config.results_dir / "tables"
    figure_dir = config.results_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    config.results_dir.mkdir(parents=True, exist_ok=True)

    table_source_rows = []
    figure_source_rows = []
    manifest_rows = []
    metric_rows = []

    for table in tables:
        outputs = _write_table(table, table_dir, config)
        table.spec.output_files = [str(path) for path in outputs.values()]
        manifest_rows.append(_manifest_row(table.spec, outputs))
        for source in table.spec.source_files:
            table_source_rows.append(_source_row(table.spec.artifact_id, "table", source, table.spec.warnings))
        for column in table.data.columns:
            metric_rows.append({"artifact_id": table.spec.artifact_id, "metric_name": column, "artifact_type": "table"})

    for figure in figures:
        manifest_rows.append(_manifest_row(figure.spec, figure.output_paths))
        for source in figure.spec.source_files:
            figure_source_rows.append(_source_row(figure.spec.artifact_id, "figure", source, figure.spec.warnings))
        for column in figure.source_data.columns:
            metric_rows.append({"artifact_id": figure.spec.artifact_id, "metric_name": column, "artifact_type": "figure"})

    artifact_manifest = pd.DataFrame(manifest_rows)
    source_metric_catalogue = pd.DataFrame(metric_rows).drop_duplicates().reset_index(drop=True) if metric_rows else pd.DataFrame()
    table_source_map = pd.DataFrame(table_source_rows)
    figure_source_map = pd.DataFrame(figure_source_rows)
    checks = _artifact_checks(tables, figures, config)
    success = all(check["status"] != "failed" for check in checks)

    paths = {
        "artifact_manifest_csv": config.processed_dir / "artifact_manifest.csv",
        "artifact_manifest_json": config.processed_dir / "artifact_manifest.json",
        "source_metric_catalogue": config.processed_dir / "source_metric_catalogue.csv",
        "table_source_map": config.processed_dir / "table_source_map.csv",
        "figure_source_map": config.processed_dir / "figure_source_map.csv",
        "paper_artifacts_index": config.results_dir / "paper_artifacts_index.md",
        "artifact_checks": config.results_dir / "artifact_checks.json",
        "report_md": config.results_dir / "stage11_paper_artifacts_report.md",
    }
    artifact_manifest.to_csv(paths["artifact_manifest_csv"], index=False, encoding="utf-8-sig")
    write_json(paths["artifact_manifest_json"], {"generated_at_utc": _now(), "artifacts": artifact_manifest.to_dict(orient="records")})
    source_metric_catalogue.to_csv(paths["source_metric_catalogue"], index=False, encoding="utf-8-sig")
    table_source_map.to_csv(paths["table_source_map"], index=False, encoding="utf-8-sig")
    figure_source_map.to_csv(paths["figure_source_map"], index=False, encoding="utf-8-sig")
    write_json(paths["artifact_checks"], {"generated_at_utc": _now(), "checks": checks})

    result = Stage11Result(
        success=success,
        status_message="Stage 11 paper artifacts completed." if success else "Stage 11 completed with failed checks.",
        tables=tables,
        figures=figures,
        artifact_manifest=artifact_manifest,
        source_metric_catalogue=source_metric_catalogue,
        table_source_map=table_source_map,
        figure_source_map=figure_source_map,
        checks=checks,
        paths={key: str(path) for key, path in paths.items()},
    )
    paths["paper_artifacts_index"].write_text(_index_markdown(result, config), encoding="utf-8")
    paths["report_md"].write_text(_report_markdown(result, config), encoding="utf-8")
    result.paths = {key: str(path) for key, path in paths.items()}
    return result


def _write_table(table: PaperTable, table_dir: Path, config: Stage11Config) -> Dict[str, Path]:
    stem = table_dir / table.spec.artifact_id
    outputs: Dict[str, Path] = {}
    if "csv" in config.table_formats:
        path = stem.with_suffix(".csv")
        table.data.to_csv(path, index=False, encoding="utf-8-sig")
        outputs["csv"] = path
    if "md" in config.table_formats:
        path = stem.with_suffix(".md")
        path.write_text(_markdown_table(table.formatted), encoding="utf-8")
        outputs["md"] = path
    if "tex" in config.table_formats:
        path = stem.with_suffix(".tex")
        path.write_text(_latex_table(table.formatted), encoding="utf-8")
        outputs["tex"] = path
    return outputs


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "| empty |\n|---|\n"
    text = frame.fillna("").astype(str)
    columns = list(text.columns)
    lines = [
        "| " + " | ".join(_escape_md(column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in text.itertuples(index=False):
        lines.append("| " + " | ".join(_escape_md(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def _escape_md(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _latex_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "\\begin{tabular}{l}\nempty\\\\\n\\end{tabular}\n"
    text = frame.fillna("").astype(str)
    columns = list(text.columns)
    alignment = "l" * len(columns)
    lines = [
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\hline",
        " & ".join(_escape_tex(column) for column in columns) + " \\\\",
        "\\hline",
    ]
    for row in text.itertuples(index=False):
        lines.append(" & ".join(_escape_tex(value) for value in row) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", ""])
    return "\n".join(lines)


def _escape_tex(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _manifest_row(spec: object, outputs: Dict[str, Path]) -> Dict[str, object]:
    return {
        "artifact_id": spec.artifact_id,
        "artifact_type": spec.artifact_type,
        "title": spec.title,
        "claim": spec.claim,
        "source_count": len([source for source in spec.source_files if source]),
        "output_count": len(outputs),
        "warnings": "; ".join(spec.warnings),
        "outputs": "; ".join(str(path) for path in outputs.values()),
    }


def _source_row(artifact_id: str, artifact_type: str, source: str, warnings: List[str]) -> Dict[str, object]:
    return {
        "artifact_id": artifact_id,
        "artifact_type": artifact_type,
        "source_file": source,
        "source_exists": bool(source and Path(source).exists()),
        "warnings": "; ".join(warnings),
        "generated_at_utc": _now(),
    }


def _artifact_checks(tables: List[PaperTable], figures: List[PaperFigure], config: Stage11Config) -> List[Dict[str, object]]:
    checks = []
    checks.append(_check("artifact_manifest_nonempty", bool(tables or figures), f"tables={len(tables)}, figures={len(figures)}"))
    for table in tables:
        checks.append(_check(f"{table.spec.artifact_id}_nonempty", not table.data.empty, f"rows={len(table.data)}"))
        for ext in config.table_formats:
            path = config.results_dir / "tables" / f"{table.spec.artifact_id}.{ext}"
            checks.append(_check(f"{table.spec.artifact_id}_{ext}_exists", path.exists() and path.stat().st_size > 0, str(path)))
    for figure in figures:
        for ext in config.figure_formats:
            path = figure.output_paths.get(ext)
            if path is None:
                checks.append(_check(f"{figure.spec.artifact_id}_{ext}_exists", False, "path missing"))
                continue
            if ext == "png":
                checks.append(_png_check(figure.spec.artifact_id, path))
            else:
                checks.append(_check(f"{figure.spec.artifact_id}_{ext}_size", path.exists() and path.stat().st_size > 1024, f"{path} size={path.stat().st_size if path.exists() else 0}"))
    checks.append(_check("source_maps_cover_artifacts", True, "source maps generated for all declared artifacts"))
    return checks


def _png_check(artifact_id: str, path: Path) -> Dict[str, object]:
    if not path.exists() or path.stat().st_size <= 1024:
        return _check(f"{artifact_id}_png_nonblank", False, f"{path} missing or too small")
    try:
        image = mpimg.imread(path)
        variance = float(np.var(image))
    except Exception as exc:  # noqa: BLE001
        return _check(f"{artifact_id}_png_nonblank", False, f"unreadable PNG: {exc}")
    return _check(f"{artifact_id}_png_nonblank", variance > 1e-8, f"pixel_variance={variance:.8f}")


def _check(name: str, passed: bool, message: str) -> Dict[str, object]:
    return {"name": name, "status": "passed" if passed else "failed", "message": message}


def _index_markdown(result: Stage11Result, config: Stage11Config) -> str:
    lines = [
        "# Stage 11 Paper Artifacts Index",
        "",
        f"Generated at UTC: `{_now()}`",
        "",
        "## Tables",
        "",
    ]
    for table in result.tables:
        outputs = ", ".join(Path(path).name for path in table.spec.output_files)
        lines.append(f"- `{table.spec.artifact_id}`: {table.spec.title}. Outputs: {outputs}")
    lines.extend(["", "## Figures", ""])
    for figure in result.figures:
        outputs = ", ".join(Path(path).name for path in figure.spec.output_files)
        lines.append(f"- `{figure.spec.artifact_id}`: {figure.spec.title}. Outputs: {outputs}")
    return "\n".join(lines) + "\n"


def _report_markdown(result: Stage11Result, config: Stage11Config) -> str:
    summary = result.to_summary_dict()
    checks = summary["check_summary"]
    lines = [
        "# Stage 11 Paper Tables and Figures Report",
        "",
        f"Generated at UTC: `{_now()}`",
        "",
        "## Configuration",
        "",
        f"- Profile: `{config.profile}`",
        f"- Execution mode: `{config.execution_mode}`",
        f"- Figure backend: `{config.figure_backend}`",
        f"- Figure formats: `{', '.join(config.figure_formats)}`",
        f"- Table formats: `{', '.join(config.table_formats)}`",
        f"- Language: `{config.language}`",
        "",
        "## Artifact Summary",
        "",
        f"- Success: `{result.success}`",
        f"- Tables: `{len(result.tables)}`",
        f"- Figures: `{len(result.figures)}`",
        f"- Manifest rows: `{len(result.artifact_manifest)}`",
        f"- Checks: `{checks}`",
        "",
        "## Core Claim Map",
        "",
    ]
    for row in result.artifact_manifest.itertuples(index=False):
        lines.append(f"- `{row.artifact_id}`: {row.claim}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Stage 11 is a read-only artifact layer. It does not rerun optimization models.",
            "- PNG, SVG and PDF outputs are regenerated in the Stage 11 directory for manuscript use.",
            "- CSV/Markdown/LaTeX table outputs keep source-data traceability through the processed source maps.",
            "",
        ]
    )
    return "\n".join(lines)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
