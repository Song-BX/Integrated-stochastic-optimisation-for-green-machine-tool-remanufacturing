"""Orchestrate Stage 12 final experiment completion and figure audit."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from stage11_paper_artifacts.config import Stage11Config
from stage11_paper_artifacts.runner import run_stage11
from stage9_experiments.config import Stage9Config
from stage9_experiments.suite import run_stage9_suite

from .audit import audit_stage11_artifacts
from .completion import build_completion_manifest, complete_missing_experiments
from .config import Stage12Config
from .io_utils import ensure_dirs, require_stage1_gate
from .evidence_pack import build_claim_evidence_pack
from .operational_reality import build_operational_reality_audit
from .reality_audit import build_result_reality_audit
from .reporting import write_stage12_reports
from .structures import Stage12Result


def run_stage12(config: Stage12Config, root: Path) -> tuple[Stage12Result, Dict[str, str]]:
    """Run Stage 12 finalization."""

    if config.execution_mode not in {"complete-and-audit", "audit-only", "complete-missing"}:
        raise ValueError("Stage 12 execution_mode must be complete-and-audit, audit-only, or complete-missing.")
    if config.figure_backend != "matplotlib":
        raise ValueError("Stage 12 currently audits matplotlib Stage 11 artifacts only.")

    require_stage1_gate(config)
    ensure_dirs(config)

    manifest = build_completion_manifest(config)
    if config.execution_mode in {"complete-and-audit", "complete-missing"}:
        completion_log, gaps = complete_missing_experiments(config, root, manifest)
    else:
        completion_log = pd.DataFrame()
        gaps = pd.DataFrame()

    if config.execution_mode in {"complete-and-audit", "audit-only"}:
        _refresh_stage9(config, root)
        _refresh_stage11(config, root)
        audit_summary = audit_stage11_artifacts(config)
        catalogue = pd.DataFrame([finding.to_dict() for finding in audit_summary.findings])
        claim_catalogue, main, appendix, evidence_pack = build_claim_evidence_pack(config, catalogue)
    else:
        catalogue = pd.DataFrame()
        claim_catalogue = pd.DataFrame()
        main = pd.DataFrame()
        appendix = pd.DataFrame()
        evidence_pack = pd.DataFrame()

    readiness = _readiness_summary(manifest, completion_log, gaps, catalogue, main, appendix)
    reality_audit, adjustment_log = build_result_reality_audit(config, completion_log, gaps)
    operational_audit, operational_adjustments = build_operational_reality_audit(config)
    main_gap = bool(
        not gaps.empty
        and gaps.get("affects_main_claim", pd.Series(dtype=bool)).astype(bool).any()
    )
    failed_main_reality = _has_failed_main_reality(reality_audit)
    failed_main_operational = _has_failed_main_operational_reality(operational_audit)
    success = bool(
        not catalogue.empty
        and not main.empty
        and not appendix.empty
        and not _has_failed_main_artifacts(catalogue)
        and not failed_main_reality
        and not failed_main_operational
        and not main_gap
    )
    result = Stage12Result(
        success=success,
        status_message="Stage 12 finalization completed." if success else "Stage 12 completed with readiness gaps.",
        final_experiment_manifest=manifest,
        completion_run_log=completion_log,
        blocking_gap_register=gaps,
        figure_audit_catalogue=catalogue,
        claim_sentence_catalogue=claim_catalogue,
        main_text_artifact_set=main,
        appendix_artifact_set=appendix,
        paper_evidence_pack=evidence_pack,
        paper_readiness_summary=readiness,
        result_reality_audit=reality_audit,
        result_adjustment_log=adjustment_log,
        operational_reality_audit=operational_audit,
        operational_interpretation_adjustments=operational_adjustments,
    )
    paths = write_stage12_reports(result, config)
    return result, paths


def _refresh_stage9(config: Stage12Config, root: Path) -> None:
    """Regenerate Stage 9 collection tables after isolated completion runs."""

    stage9_config = Stage9Config(
        stage1_report=config.stage1_report,
        processed_dir=config.data_processed_dir / "stage9",
        results_dir=config.data_results_dir / "stage9",
        data_results_dir=config.data_results_dir,
        data_processed_dir=config.data_processed_dir,
        profile=config.profile,
        execution_mode="collect-existing",
        machine_type_id=config.machine_type_id,
        period_start=config.period_start,
        period_count=config.period_count,
        processing_window_periods=config.processing_window_periods,
        run_epsilon_grid_size=config.quick_epsilon_grid_size,
        run_max_iterations=config.quick_max_iterations,
        run_repair_time_limit=config.quick_repair_time_limit,
    ).resolved(root)
    run_stage9_suite(stage9_config, root)


def _refresh_stage11(config: Stage12Config, root: Path) -> None:
    """Regenerate Stage 11 artifacts after completion attempts."""

    stage11_config = Stage11Config(
        stage1_report=config.stage1_report,
        data_processed_dir=config.data_processed_dir,
        data_results_dir=config.data_results_dir,
        processed_dir=config.data_processed_dir / "stage11",
        results_dir=config.data_results_dir / "stage11",
        profile="manuscript",
        execution_mode="collect-existing",
        figure_backend=config.figure_backend,
        figure_formats=config.figure_formats,
        table_formats=config.table_formats,
        dpi=config.dpi,
        language="en",
    ).resolved(root)
    run_stage11(stage11_config)


def _readiness_summary(
    manifest: pd.DataFrame,
    completion_log: pd.DataFrame,
    gaps: pd.DataFrame,
    catalogue: pd.DataFrame,
    main: pd.DataFrame,
    appendix: pd.DataFrame,
) -> dict[str, object]:
    completed = int((completion_log.get("success", pd.Series(dtype=bool)) == True).sum()) if not completion_log.empty else 0  # noqa: E712
    blocking_count = int(len(gaps))
    main_ready = int(
        (
            (main.get("source_map_complete", pd.Series(dtype=bool)) == True)
            & (main.get("export_complete", pd.Series(dtype=bool)) == True)
            & (main.get("nonblank_or_nonempty", pd.Series(dtype=bool)) == True)
            & (main.get("claim_unique", pd.Series(dtype=bool)) == True)
        ).sum()
    ) if not main.empty else 0
    readiness_level = "ready_for_methods_results_draft"
    if blocking_count:
        readiness_level = "ready_with_explicit_robustness_gaps"
    if main.empty or main_ready != len(main):
        readiness_level = "not_ready_artifact_audit_failed"
    return {
        "readiness_level": readiness_level,
        "completion_items": int(len(manifest)),
        "completed_or_available_items": completed,
        "blocking_gap_count": blocking_count,
        "blocking_gaps_affect_main_claim": bool(not gaps.empty and gaps.get("affects_main_claim", pd.Series(dtype=bool)).astype(bool).any()),
        "audited_artifacts": int(len(catalogue)),
        "claim_sentence_artifacts": int(len(main)),
        "main_text_artifacts": int(len(main)),
        "main_text_artifacts_ready": main_ready,
        "appendix_artifacts": int(len(appendix)),
    }


def _has_failed_main_artifacts(catalogue: pd.DataFrame) -> bool:
    if catalogue.empty:
        return True
    main = catalogue[catalogue["recommended_location"] == "main_text"]
    if main.empty:
        return True
    readiness_cols = ["claim_unique", "source_map_complete", "export_complete", "nonblank_or_nonempty"]
    return not bool(main[readiness_cols].astype(bool).all(axis=None))


def _has_failed_main_reality(reality_audit: pd.DataFrame) -> bool:
    if reality_audit.empty:
        return False
    main = reality_audit[reality_audit.get("evidence_decision", pd.Series(dtype=object)).astype(str) == "main_text"]
    if main.empty:
        return False
    return bool(main.get("status", pd.Series(dtype=object)).astype(str).isin(["failed"]).any())


def _has_failed_main_operational_reality(operational_audit: pd.DataFrame) -> bool:
    if operational_audit.empty:
        return False
    main = operational_audit[
        operational_audit.get("evidence_decision", pd.Series(dtype=object)).astype(str).isin(
            ["main_text", "main_text_with_caution"]
        )
    ]
    if main.empty:
        return False
    return bool(
        (
            (main.get("status", pd.Series(dtype=object)).astype(str) == "failed")
            | (main.get("operational_reality_decision", pd.Series(dtype=object)).astype(str) == "do_not_use_as_main_claim")
        ).any()
    )
