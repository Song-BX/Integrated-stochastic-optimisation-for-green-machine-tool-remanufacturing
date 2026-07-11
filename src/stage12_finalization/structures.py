"""Public data structures for Stage 12 finalization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import pandas as pd


@dataclass
class CompletionSpec:
    """One manuscript-critical completion item."""

    completion_id: str
    experiment_group: str
    source_experiment_id: str
    target_path: str
    action: str
    priority: str = "manuscript-critical"
    expected_runtime_class: str = "quick"
    status: str = "planned"
    reason: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CompletionResult:
    """Result of one completion attempt."""

    completion_id: str
    experiment_group: str
    source_experiment_id: str
    action: str
    status: str
    success: bool
    output_path: str | None = None
    seconds: float | None = None
    message: str | None = None
    affects_main_claim: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FigureAuditFinding:
    """Reviewer-style audit finding for one Stage 11 artifact."""

    artifact_id: str
    artifact_type: str
    title: str
    claim: str
    recommended_location: str
    claim_unique: bool
    source_map_complete: bool
    export_complete: bool
    nonblank_or_nonempty: bool
    reviewer_risk: str
    action_required: str
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FigureAuditSummary:
    """Complete figure/table audit output."""

    findings: List[FigureAuditFinding] = field(default_factory=list)
    main_text_artifacts: pd.DataFrame = field(default_factory=pd.DataFrame)
    appendix_artifacts: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "findings": [finding.to_dict() for finding in self.findings],
            "main_text_artifacts": self.main_text_artifacts.to_dict(orient="records"),
            "appendix_artifacts": self.appendix_artifacts.to_dict(orient="records"),
        }


@dataclass
class Stage12Result:
    """Complete Stage 12 finalization result."""

    success: bool
    status_message: str
    final_experiment_manifest: pd.DataFrame
    completion_run_log: pd.DataFrame
    blocking_gap_register: pd.DataFrame
    figure_audit_catalogue: pd.DataFrame
    claim_sentence_catalogue: pd.DataFrame
    main_text_artifact_set: pd.DataFrame
    appendix_artifact_set: pd.DataFrame
    paper_evidence_pack: pd.DataFrame
    paper_readiness_summary: Dict[str, Any]
    result_reality_audit: pd.DataFrame = field(default_factory=pd.DataFrame)
    result_adjustment_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    operational_reality_audit: pd.DataFrame = field(default_factory=pd.DataFrame)
    operational_interpretation_adjustments: pd.DataFrame = field(default_factory=pd.DataFrame)
    paths: Dict[str, str] = field(default_factory=dict)

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "status_message": self.status_message,
            "completion_items": int(len(self.final_experiment_manifest)),
            "completed_items": int((self.completion_run_log.get("success", pd.Series(dtype=bool)) == True).sum()) if not self.completion_run_log.empty else 0,  # noqa: E712
            "blocking_gaps": int(len(self.blocking_gap_register)),
            "audited_artifacts": int(len(self.figure_audit_catalogue)),
            "claim_sentences": int(len(self.claim_sentence_catalogue)),
            "main_text_artifacts": int(len(self.main_text_artifact_set)),
            "appendix_artifacts": int(len(self.appendix_artifact_set)),
            "paper_evidence_pack_rows": int(len(self.paper_evidence_pack)),
            "result_reality_audit_rows": int(len(self.result_reality_audit)),
            "result_adjustments": int(len(self.result_adjustment_log)),
            "operational_reality_audit_rows": int(len(self.operational_reality_audit)),
            "operational_interpretation_adjustments": int(len(self.operational_interpretation_adjustments)),
            "readiness": self.paper_readiness_summary,
        }
