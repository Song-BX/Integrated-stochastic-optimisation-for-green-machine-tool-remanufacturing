"""Reviewer-style paper artifact audit for Stage 12."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List

import matplotlib.image as mpimg
import numpy as np
import pandas as pd

from .config import Stage12Config
from .io_utils import existing_stage11_paths, read_csv, read_json
from .structures import FigureAuditFinding, FigureAuditSummary


MAIN_TEXT_ARTIFACTS = {
    "T1_stagewise_model_complexity",
    "T2_baseline_and_ablation",
    "T3_pareto_payoff_and_representatives",
    "T4_exact_vs_matheuristic_and_top5",
    "T5_risk_selective_assembly_metrics",
    "F1_model_architecture",
    "F2_data_to_model_pipeline",
    "F3_pareto_tradeoff_panels",
    "F4_baseline_ablation_comparison",
    "F5_matheuristic_convergence",
    "F6_exact_vs_matheuristic_top5",
    "F7_route_mix_and_operational_shift",
}

APPENDIX_BY_DEFAULT = {
    "T6_stage10_strengthening",
    "T7_saa_sensitivity_manifest",
    "F8_stage10_strengthening",
}


def audit_stage11_artifacts(config: Stage12Config) -> FigureAuditSummary:
    """Audit Stage 11 tables and figures for manuscript readiness."""

    paths = existing_stage11_paths(config)
    manifest = read_csv(paths["artifact_manifest"])
    stage11_checks = read_json(paths["artifact_checks"]) if paths["artifact_checks"].exists() else {"checks": []}
    table_sources = read_csv(paths["table_source_map"])
    figure_sources = read_csv(paths["figure_source_map"])
    check_lookup = {str(check.get("name")): check for check in stage11_checks.get("checks", [])}
    claim_counts = Counter(str(row.get("claim", "")).strip() for row in manifest.to_dict(orient="records"))

    findings: List[FigureAuditFinding] = []
    for row in manifest.to_dict(orient="records"):
        artifact_id = str(row["artifact_id"])
        artifact_type = str(row["artifact_type"])
        claim = str(row.get("claim", "")).strip()
        source_map = table_sources if artifact_type == "table" else figure_sources
        source_rows = source_map[source_map["artifact_id"] == artifact_id] if not source_map.empty else pd.DataFrame()
        source_complete = bool(not source_rows.empty and source_rows["source_exists"].astype(bool).all())
        if artifact_id == "F1_model_architecture" and artifact_type == "figure":
            source_complete = bool(
                any(
                    "stage6_selective_assembly_report.md" in str(value)
                    or "stage7_pareto_report.md" in str(value)
                    or "stage8_matheuristic_report.md" in str(value)
                    or "stage10_strengthening_report.md" in str(value)
                    for value in source_rows.get("source_file", pd.Series(dtype=object)).astype(str).tolist()
                )
            )
        output_paths = _parse_outputs(row.get("outputs", ""))
        export_complete = _exports_complete(output_paths, artifact_type, config)
        nonblank = _artifact_nonblank(artifact_id, artifact_type, output_paths, check_lookup)
        recommended = _recommended_location(artifact_id, claim, source_complete, export_complete, nonblank)
        risk, action = _risk_and_action(artifact_id, artifact_type, source_complete, export_complete, nonblank, recommended)
        findings.append(
            FigureAuditFinding(
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                title=str(row.get("title", "")),
                claim=claim,
                recommended_location=recommended,
                claim_unique=bool(claim and claim_counts[claim] == 1),
                source_map_complete=source_complete,
                export_complete=export_complete,
                nonblank_or_nonempty=nonblank,
                reviewer_risk=risk,
                action_required=action,
                rationale=_rationale(artifact_id, recommended),
            )
        )
    catalogue = pd.DataFrame([finding.to_dict() for finding in findings])
    main = catalogue[catalogue["recommended_location"] == "main_text"].reset_index(drop=True)
    appendix = catalogue[catalogue["recommended_location"].isin(["appendix", "supplementary"])].reset_index(drop=True)
    return FigureAuditSummary(findings=findings, main_text_artifacts=main, appendix_artifacts=appendix)


def _parse_outputs(value: object) -> List[Path]:
    if value is None or pd.isna(value):
        return []
    return [Path(part.strip()) for part in str(value).split(";") if part.strip()]


def _exports_complete(outputs: List[Path], artifact_type: str, config: Stage12Config) -> bool:
    if not outputs:
        return False
    suffixes = {path.suffix.lower().lstrip(".") for path in outputs}
    required = set(config.table_formats if artifact_type == "table" else config.figure_formats)
    return required.issubset(suffixes) and all(path.exists() and path.stat().st_size > 0 for path in outputs)


def _artifact_nonblank(artifact_id: str, artifact_type: str, outputs: List[Path], checks: Dict[str, dict]) -> bool:
    if artifact_type == "table":
        csv_paths = [path for path in outputs if path.suffix.lower() == ".csv"]
        if not csv_paths:
            return False
        try:
            frame = pd.read_csv(csv_paths[0], encoding="utf-8-sig")
            return not frame.empty
        except Exception:  # noqa: BLE001
            return False
    png_paths = [path for path in outputs if path.suffix.lower() == ".png"]
    check = checks.get(f"{artifact_id}_png_nonblank")
    if check and check.get("status") == "passed":
        return True
    if not png_paths:
        return False
    try:
        image = mpimg.imread(png_paths[0])
        return float(np.var(image)) > 1e-8
    except Exception:  # noqa: BLE001
        return False


def _recommended_location(artifact_id: str, claim: str, source_complete: bool, export_complete: bool, nonblank: bool) -> str:
    if not claim or not source_complete or not export_complete or not nonblank:
        return "revise_before_use"
    if artifact_id in APPENDIX_BY_DEFAULT:
        return "appendix"
    if artifact_id in MAIN_TEXT_ARTIFACTS:
        return "main_text"
    return "supplementary"


def _risk_and_action(
    artifact_id: str,
    artifact_type: str,
    source_complete: bool,
    export_complete: bool,
    nonblank: bool,
    recommended: str,
) -> tuple[str, str]:
    if not source_complete:
        return "high", "Repair source map before citing this artifact."
    if not export_complete:
        return "high", "Regenerate missing export formats."
    if not nonblank:
        return "high", "Regenerate or replace blank/unreadable artifact."
    if recommended == "appendix":
        return "medium", "Keep as appendix or use only to address reviewer concerns."
    if artifact_type == "figure" and artifact_id in {"F1_model_architecture", "F2_data_to_model_pipeline"}:
        return "low", "Use as schematic evidence; keep caption precise."
    return "low", "Ready for manuscript drafting."


def _rationale(artifact_id: str, recommended: str) -> str:
    if recommended == "main_text":
        return "High-value evidence for the main IJPR narrative."
    if recommended == "appendix":
        return "Important robustness or reviewer-facing evidence, but not central enough for main text by default."
    if recommended == "revise_before_use":
        return "Artifact fails at least one readiness condition."
    return "Useful supporting material after the main evidence chain is set."
