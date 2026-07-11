"""Stage 9 orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from .analysis import build_suite_result
from .collectors import collect_experiment_results
from .config import Stage9Config
from .io_utils import require_stage1_passed
from .manifest import build_experiment_manifest
from .reporting import write_stage9_reports
from .runner import maybe_run_profile


def run_stage9_suite(config: Stage9Config, root: Path) -> tuple[object, Dict[str, str], Dict[str, object]]:
    """Run Stage 9 collection/orchestration and write reports."""

    stage1_payload = require_stage1_passed(config.stage1_report)
    run_summary = maybe_run_profile(config, root)
    manifest = build_experiment_manifest(config)
    all_results = collect_experiment_results(manifest, config)
    suite_result = build_suite_result(manifest, all_results, config)
    paths = write_stage9_reports(suite_result, config)
    return suite_result, paths, {"stage1_summary": stage1_payload.get("summary", {}), "run_summary": run_summary}
