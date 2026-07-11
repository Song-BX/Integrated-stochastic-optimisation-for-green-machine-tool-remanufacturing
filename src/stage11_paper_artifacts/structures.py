"""Public data structures for Stage 11 artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


@dataclass
class PaperArtifactSpec:
    """A declared manuscript artifact."""

    artifact_id: str
    artifact_type: str
    title: str
    claim: str
    source_files: List[str] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PaperTable:
    """A manuscript table plus provenance metadata."""

    spec: PaperArtifactSpec
    data: pd.DataFrame
    formatted: pd.DataFrame


@dataclass
class PaperFigure:
    """A manuscript figure plus provenance metadata."""

    spec: PaperArtifactSpec
    source_data: pd.DataFrame
    output_paths: Dict[str, Path] = field(default_factory=dict)


@dataclass
class Stage11Result:
    """Complete Stage 11 generation result."""

    success: bool
    status_message: str
    tables: List[PaperTable]
    figures: List[PaperFigure]
    artifact_manifest: pd.DataFrame
    source_metric_catalogue: pd.DataFrame
    table_source_map: pd.DataFrame
    figure_source_map: pd.DataFrame
    checks: List[Dict[str, Any]]
    paths: Dict[str, str] = field(default_factory=dict)

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "status_message": self.status_message,
            "table_count": len(self.tables),
            "figure_count": len(self.figures),
            "manifest_rows": int(len(self.artifact_manifest)),
            "check_summary": {
                status: sum(1 for check in self.checks if check.get("status") == status)
                for status in ["passed", "warning", "failed"]
            },
        }
