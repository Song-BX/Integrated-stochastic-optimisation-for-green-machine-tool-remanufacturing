"""Reality checks for Stage 9-12 manuscript evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .config import Stage12Config
from .io_utils import read_csv


SENSITIVITY_PARAMETERS = {
    "cvar_lambda": 4,
    "min_system_reliability": 3,
    "env_weight": 3,
    "assembly_shortfall_penalty_rmb": 3,
}


def build_result_reality_audit(
    config: Stage12Config,
    completion_log: pd.DataFrame,
    gaps: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit whether collected manuscript results match their underlying files."""

    rows: List[Dict[str, Any]] = []
    adjustments: List[Dict[str, Any]] = []
    _audit_saa(config, rows, adjustments)
    _audit_sensitivity(config, rows, adjustments)
    _audit_stage11_t7(config, rows, adjustments)
    _audit_stage12_readiness(completion_log, gaps, rows, adjustments)
    _audit_core_stage9_tables(config, rows, adjustments)
    return pd.DataFrame(rows), pd.DataFrame(adjustments)


def _audit_saa(config: Stage12Config, rows: List[Dict[str, Any]], adjustments: List[Dict[str, Any]]) -> None:
    table = _read_stage9_csv(config, "saa_stability.csv")
    if table.empty:
        _row(rows, "stage9_saa_table", "failed", "missing", "Stage 9 SAA stability table is missing or empty.")
        return
    for count in (9, 18, 27):
        table_row = _find_saa_row(table, count)
        summary = _read_run_summary(config.data_results_dir / "stage9" / "runs" / f"saa_scenario_{count}" / "summary.json")
        if count == 9:
            passed = not table_row.empty
            _row(rows, f"saa_scenario_{count}", "passed" if passed else "failed", "main_text" if passed else "exclude", "Canonical 9-scenario SAA row is present.")
            continue
        if not summary:
            _row(rows, f"saa_scenario_{count}", "failed", "exclude", f"Isolated SAA summary for {count} scenarios is missing.")
            continue
        summary_success = _as_bool(summary.get("success"))
        table_success = _as_bool(table_row.iloc[0].get("success")) if not table_row.empty else None
        table_status = str(table_row.iloc[0].get("status", "")) if not table_row.empty else "missing"
        if summary_success is False:
            passed = table_success is False and "failed" in table_status
            _row(
                rows,
                f"saa_scenario_{count}",
                "passed" if passed else "warning",
                "appendix_or_audit",
                f"Run summary reports success=false/status={summary.get('status')}; table status={table_status}, success={table_success}.",
                source=str(config.data_results_dir / "stage9" / "runs" / f"saa_scenario_{count}" / "summary.json"),
            )
            if not passed:
                adjustments.append(_adjustment("stage9_saa_status", f"saa_scenario_{count}", "Propagated failed isolated-run status into Stage 9/11 audit tables."))
        else:
            passed = bool(table_success)
            _row(
                rows,
                f"saa_scenario_{count}",
                "passed" if passed else "warning",
                "main_text" if passed else "appendix_or_audit",
                f"Run summary success={summary_success}; table success={table_success}.",
                source=str(config.data_results_dir / "stage9" / "runs" / f"saa_scenario_{count}" / "summary.json"),
            )


def _audit_sensitivity(config: Stage12Config, rows: List[Dict[str, Any]], adjustments: List[Dict[str, Any]]) -> None:
    table = _read_stage9_csv(config, "sensitivity_summary.csv")
    if table.empty:
        _row(rows, "stage9_sensitivity_table", "failed", "exclude", "Stage 9 sensitivity table is missing or empty.")
        return
    for parameter, expected_count in SENSITIVITY_PARAMETERS.items():
        subset = table[table.get("parameter", pd.Series(dtype=object)).astype(str) == parameter].copy()
        if len(subset) < expected_count:
            _row(rows, f"sensitivity_{parameter}", "warning", "appendix_or_audit", f"Only {len(subset)}/{expected_count} sensitivity levels are present.")
            continue
        missing_summary = 0
        for row in subset.itertuples(index=False):
            path = config.data_results_dir / "stage9" / "runs" / f"sensitivity_{parameter}_{_format_level(getattr(row, 'level'))}" / "summary.json"
            if not path.exists():
                missing_summary += 1
        if missing_summary:
            _row(rows, f"sensitivity_{parameter}", "warning", "appendix_or_audit", f"{missing_summary} sensitivity rows lack isolated summary.json files.")
            continue
        metric_cols = ["objective_value", "expected_final_backlog_units", "cvar_value", "expected_assembly_shortfall_units"]
        flat_cols = [_is_flat(subset, column) for column in metric_cols if column in subset.columns]
        all_flat = bool(flat_cols and all(flat_cols))
        if all_flat:
            runtimes = pd.to_numeric(subset.get("solve_seconds", pd.Series(dtype=object)), errors="coerce").dropna()
            quick_proxy = bool(not runtimes.empty and runtimes.max() < 1.0)
            status = "warning" if quick_proxy else "passed"
            decision = "appendix_or_audit" if quick_proxy else "appendix"
            message = (
                "Flat response detected with sub-second summaries; classify as needs_rerun before using as robustness evidence."
                if quick_proxy
                else "Flat response detected; classify as true insensitive in this instance."
            )
            _row(rows, f"sensitivity_{parameter}", status, decision, message)
            if quick_proxy:
                adjustments.append(_adjustment("sensitivity_flat_response", parameter, "Flagged flat sub-second sensitivity response as needs_rerun/appendix-only."))
        else:
            _row(rows, f"sensitivity_{parameter}", "passed", "appendix", "Sensitivity axis has traceable isolated summaries and non-flat response.")


def _audit_stage11_t7(config: Stage12Config, rows: List[Dict[str, Any]], adjustments: List[Dict[str, Any]]) -> None:
    path = config.data_results_dir / "stage11" / "tables" / "T7_saa_sensitivity_manifest.csv"
    table = read_csv(path)
    if table.empty:
        _row(rows, "stage11_t7_manifest", "failed", "exclude", "Stage 11 T7 manifest is missing or empty.", source=str(path))
        return
    statuses = {str(row.source_key): str(row.status) for row in table.itertuples(index=False) if hasattr(row, "source_key")}
    saa_ok = statuses.get("stage9_saa", "").startswith("available")
    sensitivity_ok = statuses.get("stage9_sensitivity", "").startswith("available")
    passed = saa_ok and sensitivity_ok
    _row(
        rows,
        "stage11_t7_manifest",
        "passed" if passed else "warning",
        "appendix" if passed else "appendix_or_audit",
        f"T7 statuses: stage9_saa={statuses.get('stage9_saa')}, stage9_sensitivity={statuses.get('stage9_sensitivity')}.",
        source=str(path),
    )
    if passed:
        adjustments.append(_adjustment("stage11_t7_csv_reader", "T7_saa_sensitivity_manifest", "CSV-based availability replaces prior JSON-only missing classification."))


def _audit_stage12_readiness(
    completion_log: pd.DataFrame,
    gaps: pd.DataFrame,
    rows: List[Dict[str, Any]],
    adjustments: List[Dict[str, Any]],
) -> None:
    blocking_count = int(len(gaps))
    failed_nonblocking = 0
    if not completion_log.empty and "success" in completion_log.columns:
        success_mask = completion_log["success"].astype(str).str.lower().isin(["true", "1", "yes"])
        failed_nonblocking = int((~success_mask).sum())
    status = "passed" if blocking_count == 0 else "warning"
    decision = "main_text" if blocking_count == 0 else "appendix_or_audit"
    _row(rows, "stage12_readiness_text", status, decision, f"Blocking gaps={blocking_count}; failed/non-success completion rows={failed_nonblocking}.")
    adjustments.append(_adjustment("stage12_report_wording", "stage12_finalization_report", "Report interpretation now distinguishes blocking gaps from failed/non-blocking quick runs."))


def _audit_core_stage9_tables(config: Stage12Config, rows: List[Dict[str, Any]], _adjustments: List[Dict[str, Any]]) -> None:
    for name in ["baseline_comparison.csv", "ablation_study.csv", "exact_vs_matheuristic_gap.csv", "top5_benchmark_summary.csv"]:
        table = _read_stage9_csv(config, name)
        passed = not table.empty
        _row(rows, f"stage9_{Path(name).stem}", "passed" if passed else "failed", "main_text" if passed else "exclude", f"{name} rows={len(table)}.")


def _read_stage9_csv(config: Stage12Config, name: str) -> pd.DataFrame:
    return read_csv(config.data_results_dir / "stage9" / name)


def _read_run_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _find_saa_row(table: pd.DataFrame, count: int) -> pd.DataFrame:
    if "scenario_setting" not in table.columns:
        return table.iloc[0:0]
    return table[table["scenario_setting"].astype(str) == str(count)]


def _is_flat(frame: pd.DataFrame, column: str) -> bool:
    values = pd.to_numeric(frame[column], errors="coerce").dropna().round(8)
    return bool(not values.empty and values.nunique() <= 1)


def _format_level(value: Any) -> str:
    text = str(value)
    if "." in text:
        fractional = text.split(".", 1)[1]
        if fractional and set(fractional) != {"0"}:
            return text
        if text in {"0.0", "0.00"}:
            return text
    try:
        parsed = float(text)
    except ValueError:
        return text
    if parsed.is_integer():
        return str(int(parsed))
    return text


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _row(
    rows: List[Dict[str, Any]],
    check_id: str,
    status: str,
    evidence_decision: str,
    message: str,
    source: str = "",
) -> None:
    rows.append(
        {
            "check_id": check_id,
            "status": status,
            "evidence_decision": evidence_decision,
            "message": message,
            "source": source,
        }
    )


def _adjustment(adjustment_type: str, target: str, message: str) -> Dict[str, Any]:
    return {
        "adjustment_type": adjustment_type,
        "target": target,
        "message": message,
    }
