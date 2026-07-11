"""Report writers for Stage 12 finalization."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Dict

import pandas as pd

from .config import Stage12Config
from .io_utils import write_json
from .structures import Stage12Result


def write_stage12_reports(result: Stage12Result, config: Stage12Config) -> Dict[str, str]:
    """Write Stage 12 CSV/JSON/Markdown outputs."""

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "final_experiment_manifest_csv": config.processed_dir / "final_experiment_manifest.csv",
        "final_experiment_manifest_json": config.processed_dir / "final_experiment_manifest.json",
        "completion_run_log": config.results_dir / "completion_run_log.csv",
        "blocking_gap_register": config.results_dir / "blocking_gap_register.csv",
        "figure_audit_catalogue": config.results_dir / "figure_audit_catalogue.csv",
        "claim_sentence_catalogue": config.results_dir / "claim_sentence_catalogue.csv",
        "claim_sentence_catalogue_md": config.results_dir / "claim_sentence_catalogue.md",
        "main_text_artifact_set": config.results_dir / "main_text_artifact_set.csv",
        "appendix_artifact_set": config.results_dir / "appendix_artifact_set.csv",
        "paper_evidence_pack": config.results_dir / "paper_evidence_pack.csv",
        "paper_evidence_pack_md": config.results_dir / "paper_evidence_pack.md",
        "result_reality_audit_csv": config.results_dir / "result_reality_audit.csv",
        "result_reality_audit_json": config.results_dir / "result_reality_audit.json",
        "result_reality_audit_md": config.results_dir / "result_reality_audit_report.md",
        "result_adjustment_log": config.results_dir / "result_adjustment_log.csv",
        "operational_reality_audit_csv": config.results_dir / "operational_reality_audit.csv",
        "operational_reality_audit_json": config.results_dir / "operational_reality_audit.json",
        "operational_reality_audit_md": config.results_dir / "operational_reality_audit_report.md",
        "operational_interpretation_adjustments": config.results_dir / "operational_interpretation_adjustments.csv",
        "paper_readiness_summary": config.results_dir / "paper_readiness_summary.json",
        "report_md": config.results_dir / "stage12_finalization_report.md",
    }
    result.final_experiment_manifest.to_csv(paths["final_experiment_manifest_csv"], index=False, encoding="utf-8-sig")
    write_json(
        paths["final_experiment_manifest_json"],
        {
            "generated_at_utc": _now(),
            "manifest": result.final_experiment_manifest.to_dict(orient="records"),
        },
    )
    _write_csv(result.completion_run_log, paths["completion_run_log"])
    _write_csv(result.blocking_gap_register, paths["blocking_gap_register"])
    _write_csv(result.figure_audit_catalogue, paths["figure_audit_catalogue"])
    _write_csv(result.claim_sentence_catalogue, paths["claim_sentence_catalogue"])
    _write_markdown_list(
        result.claim_sentence_catalogue,
        paths["claim_sentence_catalogue_md"],
        title="# Claim Sentence Catalogue",
        columns=("artifact_id", "artifact_type", "recommended_location", "claim_sentence", "possible_reviewer_question"),
    )
    _write_csv(result.main_text_artifact_set, paths["main_text_artifact_set"])
    _write_csv(result.appendix_artifact_set, paths["appendix_artifact_set"])
    _write_csv(result.paper_evidence_pack, paths["paper_evidence_pack"])
    _write_markdown_list(
        result.paper_evidence_pack,
        paths["paper_evidence_pack_md"],
        title="# Paper Evidence Pack",
        columns=("artifact_id", "evidence_source", "reader_takeaway", "caption_note"),
    )
    _write_csv(result.result_reality_audit, paths["result_reality_audit_csv"])
    write_json(
        paths["result_reality_audit_json"],
        {
            "generated_at_utc": _now(),
            "audit": result.result_reality_audit.to_dict(orient="records"),
            "adjustments": result.result_adjustment_log.to_dict(orient="records"),
        },
    )
    paths["result_reality_audit_md"].write_text(_reality_audit_markdown(result), encoding="utf-8")
    _write_csv(result.result_adjustment_log, paths["result_adjustment_log"])
    _write_csv(result.operational_reality_audit, paths["operational_reality_audit_csv"])
    write_json(
        paths["operational_reality_audit_json"],
        {
            "generated_at_utc": _now(),
            "audit": result.operational_reality_audit.to_dict(orient="records"),
            "adjustments": result.operational_interpretation_adjustments.to_dict(orient="records"),
        },
    )
    paths["operational_reality_audit_md"].write_text(_operational_audit_markdown(result), encoding="utf-8")
    _write_csv(result.operational_interpretation_adjustments, paths["operational_interpretation_adjustments"])
    write_json(paths["paper_readiness_summary"], result.paper_readiness_summary)
    paths["report_md"].write_text(_report_markdown(result, config), encoding="utf-8")
    result.paths = {key: str(path) for key, path in paths.items()}
    return result.paths


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        frame = pd.DataFrame()
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _write_markdown_list(frame: pd.DataFrame, path: Path, title: str, columns: tuple[str, ...]) -> None:
    lines = [title, ""]
    if frame.empty:
        lines.append("- No rows available.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return
    for row in frame.itertuples(index=False):
        parts = []
        for column in columns:
            value = getattr(row, column, "")
            if value is None:
                value = ""
            parts.append(f"`{column}`: {value}")
        lines.append("- " + "; ".join(parts))
    path.write_text("\n".join(lines), encoding="utf-8")


def _report_markdown(result: Stage12Result, config: Stage12Config) -> str:
    summary = result.paper_readiness_summary
    lines = [
        "# Stage 12 Final Experiment Completion and Figure Audit Report",
        "",
        f"Generated at UTC: `{_now()}`",
        "",
        "## Configuration",
        "",
        f"- Profile: `{config.profile}`",
        f"- Execution mode: `{config.execution_mode}`",
        f"- Machine type: `{config.machine_type_id}`",
        f"- Period window: `{config.period_start}` / `{config.period_count}` periods",
        f"- Figure backend: `{config.figure_backend}`",
        "",
        "## Readiness Summary",
        "",
        f"- Success: `{result.success}`",
        f"- Readiness level: `{summary.get('readiness_level')}`",
        f"- Completion items: `{len(result.final_experiment_manifest)}`",
        f"- Completed/available items: `{summary.get('completed_or_available_items')}`",
        f"- Blocking gaps: `{summary.get('blocking_gap_count')}`",
            f"- Audited artifacts: `{len(result.figure_audit_catalogue)}`",
            f"- Main-text artifacts: `{len(result.main_text_artifact_set)}`",
            f"- Appendix artifacts: `{len(result.appendix_artifact_set)}`",
            f"- Result reality audit rows: `{len(result.result_reality_audit)}`",
            f"- Operational reality audit rows: `{len(result.operational_reality_audit)}`",
            "",
            "## Blocking Gaps",
        "",
    ]
    if result.blocking_gap_register.empty:
        lines.append("- No blocking gaps were registered.")
    else:
        for row in result.blocking_gap_register.itertuples(index=False):
            lines.append(f"- `{row.source_experiment_id}`: {row.reason}")
    lines.extend(["", "## Main Text Artifact Set", ""])
    for row in result.main_text_artifact_set.itertuples(index=False):
        lines.append(f"- `{row.artifact_id}` ({row.artifact_type}): {row.claim_sentence}")
    lines.extend(["", "## Appendix / Supplementary Artifact Set", ""])
    for row in result.appendix_artifact_set.itertuples(index=False):
        lines.append(f"- `{row.artifact_id}` ({row.artifact_type}): {row.claim_sentence}")
    lines.extend(["", "## Evidence Pack", ""])
    for row in result.paper_evidence_pack.itertuples(index=False):
        lines.append(f"- `{row.artifact_id}`: {row.reader_takeaway}")
    lines.extend(["", "## Operational Reality Audit", ""])
    if result.operational_reality_audit.empty:
        lines.append("- No operational reality rows were generated.")
    else:
        for row in result.operational_reality_audit.itertuples(index=False):
            if row.status == "warning" or row.operational_reality_decision != "realistic_main_claim":
                lines.append(
                    f"- `{row.check_id}`: {row.operational_reality_decision} - {row.message} "
                    f"Recommended wording: {row.recommended_writing}"
                )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage 12 does not introduce a new optimization model.",
            "- Result reality audits distinguish blocking gaps from failed or flat-response quick runs without fabricating results.",
            "- Operational reality audits distinguish direct main claims from risk-averse behavior and appendix-only evidence.",
            "- Main-text artifacts are ready for manuscript drafting when their source maps, exports and claims pass the audit.",
            "- The evidence pack is the recommended starting point for Methods and Results drafting.",
            "",
        ]
    )
    return "\n".join(lines)


def _reality_audit_markdown(result: Stage12Result) -> str:
    lines = ["# Stage 12 Result Reality Audit", ""]
    if result.result_reality_audit.empty:
        return "\n".join(lines + ["- No audit rows available."])
    for row in result.result_reality_audit.itertuples(index=False):
        lines.append(f"- `{row.check_id}`: {row.status} ({row.evidence_decision}) - {row.message}")
    if not result.result_adjustment_log.empty:
        lines.extend(["", "## Adjustments", ""])
        for row in result.result_adjustment_log.itertuples(index=False):
            lines.append(f"- `{row.adjustment_type}` on `{row.target}`: {row.message}")
    return "\n".join(lines)


def _operational_audit_markdown(result: Stage12Result) -> str:
    lines = ["# Stage 12 Operational Reality Audit", ""]
    if result.operational_reality_audit.empty:
        return "\n".join(lines + ["- No audit rows available."])
    for row in result.operational_reality_audit.itertuples(index=False):
        lines.append(
            f"- `{row.check_id}` ({row.audit_dimension}, {row.model_stage}): "
            f"{row.status} / {row.operational_reality_decision} - {row.message} "
            f"Metric: {row.metric_value}. Writing: {row.recommended_writing}"
        )
    if not result.operational_interpretation_adjustments.empty:
        lines.extend(["", "## Interpretation Adjustments", ""])
        for row in result.operational_interpretation_adjustments.itertuples(index=False):
            lines.append(
                f"- `{row.adjustment_type}` on `{row.target_result}`: "
                f"{row.interpretation_adjustment} Evidence: {row.evidence}"
            )
    return "\n".join(lines)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
