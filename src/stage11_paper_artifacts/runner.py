"""Orchestrate Stage 11 paper artifact generation."""

from __future__ import annotations

from typing import Dict

from .config import Stage11Config
from .figures import build_figures
from .io_utils import collect_existing_sources, ensure_output_dirs, require_stage1_gate
from .reporting import write_stage11_reports
from .tables import build_tables


def run_stage11(config: Stage11Config) -> tuple[object, Dict[str, str]]:
    """Run the read-only paper artifact generation workflow."""

    if config.execution_mode != "collect-existing":
        raise ValueError("Stage 11 currently supports execution_mode='collect-existing' only.")
    if config.figure_backend != "matplotlib":
        raise ValueError("Stage 11 currently supports figure_backend='matplotlib' only.")
    require_stage1_gate(config)
    dirs = ensure_output_dirs(config)
    snapshot = collect_existing_sources(config)
    tables = build_tables(snapshot, config)
    figures = build_figures(snapshot, config, dirs["figures"])
    result = write_stage11_reports(tables, figures, config)
    return result, result.paths
