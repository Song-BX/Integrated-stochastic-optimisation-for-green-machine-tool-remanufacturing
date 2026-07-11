"""Validation routines for Stage 1 raw data checks."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .catalogue import CatalogueEntry, catalogue_totals
from .io_utils import iter_dict_rows, read_header, read_key_set, try_float
from .schema_rules import (
    BASELINE_WEIGHT_COLUMNS,
    EXPECTED_FILES,
    EXPECTED_ROW_COUNTS,
    EXPECTED_TOTAL_ROWS,
    FOREIGN_KEYS,
    NEGATIVE_ALLOWED_COLUMNS,
    NONNEGATIVE_COLUMN_KEYWORDS,
    PRIMARY_KEYS,
    PROBABILITY_COLUMNS,
)


PASS = "passed"
WARNING = "warning"
FAIL = "failed"


@dataclass
class ValidationIssue:
    check_name: str
    severity: str
    file_name: str
    message: str
    row_number: int | None = None
    column_name: str | None = None
    observed_value: str | None = None
    expected_value: str | None = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def passed_issue(check_name: str, file_name: str, message: str) -> ValidationIssue:
    return ValidationIssue(check_name=check_name, severity=PASS, file_name=file_name, message=message)


def failed_issue(check_name: str, file_name: str, message: str, **kwargs: object) -> ValidationIssue:
    return ValidationIssue(check_name=check_name, severity=FAIL, file_name=file_name, message=message, **kwargs)


def warning_issue(check_name: str, file_name: str, message: str, **kwargs: object) -> ValidationIssue:
    return ValidationIssue(check_name=check_name, severity=WARNING, file_name=file_name, message=message, **kwargs)


def summarize_issues(issues: Sequence[ValidationIssue]) -> Dict[str, int]:
    counts = {PASS: 0, WARNING: 0, FAIL: 0}
    for issue in issues:
        counts[issue.severity] = counts.get(issue.severity, 0) + 1
    return counts


def validate_file_existence(entries: Sequence[CatalogueEntry]) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    missing = [entry.file_name for entry in entries if not entry.exists]
    if not missing:
        issues.append(passed_issue("file_existence", "data/raw", f"All {len(EXPECTED_FILES)} expected CSV files exist."))
        return issues
    for file_name in missing:
        issues.append(failed_issue("file_existence", file_name, "Expected CSV file is missing."))
    return issues


def validate_catalogue_counts(entries: Sequence[CatalogueEntry]) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    entry_by_name = {entry.file_name: entry for entry in entries}
    totals = catalogue_totals(entries)
    if totals["total_rows"] == EXPECTED_TOTAL_ROWS:
        issues.append(passed_issue("total_row_count", "data/raw", f"Total row count matches expected {EXPECTED_TOTAL_ROWS}."))
    else:
        issues.append(
            failed_issue(
                "total_row_count",
                "data/raw",
                "Total row count does not match expected Stage 1 baseline.",
                observed_value=str(totals["total_rows"]),
                expected_value=str(EXPECTED_TOTAL_ROWS),
            )
        )

    for file_name, expected_rows in EXPECTED_ROW_COUNTS.items():
        entry = entry_by_name.get(file_name)
        if entry is None or not entry.exists:
            continue
        if entry.row_count == expected_rows:
            issues.append(passed_issue("file_row_count", file_name, f"Row count matches expected {expected_rows}."))
        else:
            issues.append(
                failed_issue(
                    "file_row_count",
                    file_name,
                    "Row count differs from data.md Stage 1 baseline.",
                    observed_value=str(entry.row_count),
                    expected_value=str(expected_rows),
                )
            )
    return issues


def validate_required_columns(raw_dir: Path) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    for file_name, key_columns in PRIMARY_KEYS.items():
        path = raw_dir / file_name
        if not path.exists():
            continue
        header = set(read_header(path))
        missing = [column for column in key_columns if column not in header]
        if not missing:
            issues.append(passed_issue("required_primary_key_columns", file_name, "Primary-key columns exist."))
        else:
            issues.append(
                failed_issue(
                    "required_primary_key_columns",
                    file_name,
                    "Primary-key columns are missing.",
                    observed_value=";".join(missing),
                    expected_value=";".join(key_columns),
                )
            )
    for rule in FOREIGN_KEYS:
        for file_name, columns in ((rule.source_file, rule.source_columns), (rule.target_file, rule.target_columns)):
            path = raw_dir / file_name
            if not path.exists():
                continue
            header = set(read_header(path))
            missing = [column for column in columns if column not in header]
            if missing:
                issues.append(
                    failed_issue(
                        "required_foreign_key_columns",
                        file_name,
                        f"Columns required by foreign-key rule {rule.name} are missing.",
                        observed_value=";".join(missing),
                        expected_value=";".join(columns),
                    )
                )
    if not any(issue.check_name == "required_foreign_key_columns" and issue.severity == FAIL for issue in issues):
        issues.append(passed_issue("required_foreign_key_columns", "data/raw", "All configured foreign-key columns exist."))
    return issues


def validate_primary_keys(raw_dir: Path) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    for file_name, key_columns in PRIMARY_KEYS.items():
        path = raw_dir / file_name
        if not path.exists():
            continue
        seen: set[Tuple[str, ...]] = set()
        duplicate_count = 0
        blank_count = 0
        examples: List[str] = []
        for row_number, row in enumerate(iter_dict_rows(path, key_columns), start=2):
            key = tuple(row.get(column, "") for column in key_columns)
            if any(value == "" for value in key):
                blank_count += 1
                if len(examples) < 5:
                    examples.append(f"row {row_number}: {key}")
            if key in seen:
                duplicate_count += 1
                if len(examples) < 5:
                    examples.append(f"row {row_number}: {key}")
            else:
                seen.add(key)
        if duplicate_count == 0 and blank_count == 0:
            issues.append(passed_issue("primary_key", file_name, f"Primary key {key_columns} is unique and nonblank."))
        else:
            issues.append(
                failed_issue(
                    "primary_key",
                    file_name,
                    "Primary key has duplicate or blank rows.",
                    observed_value=f"duplicates={duplicate_count}; blanks={blank_count}; examples={' | '.join(examples)}",
                    expected_value="duplicates=0; blanks=0",
                )
            )
    return issues


def validate_foreign_keys(raw_dir: Path) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    target_cache: Dict[Tuple[str, Tuple[str, ...]], set[Tuple[str, ...]]] = {}
    for rule in FOREIGN_KEYS:
        source_path = raw_dir / rule.source_file
        target_path = raw_dir / rule.target_file
        if not source_path.exists() or not target_path.exists():
            continue
        target_cache_key = (rule.target_file, rule.target_columns)
        if target_cache_key not in target_cache:
            target_keys, _, _ = read_key_set(target_path, rule.target_columns)
            target_cache[target_cache_key] = target_keys
        target_keys = target_cache[target_cache_key]

        missing_count = 0
        checked_count = 0
        examples: List[str] = []
        source_columns = tuple(rule.source_columns) + ((rule.skip_if_column,) if rule.skip_if_column else tuple())
        for row_number, row in enumerate(iter_dict_rows(source_path, source_columns), start=2):
            if rule.skip_if_column and row.get(rule.skip_if_column, "") in rule.skip_if_values:
                continue
            key = tuple(row.get(column, "") for column in rule.source_columns)
            if any(value == "" for value in key):
                continue
            checked_count += 1
            if key not in target_keys:
                missing_count += 1
                if len(examples) < 5:
                    examples.append(f"row {row_number}: {key}")
        if missing_count == 0:
            issues.append(passed_issue("foreign_key", rule.source_file, f"{rule.name}: {checked_count} references are valid."))
        else:
            issues.append(
                failed_issue(
                    "foreign_key",
                    rule.source_file,
                    f"{rule.name}: foreign-key references are missing in {rule.target_file}.",
                    observed_value=f"missing={missing_count}; examples={' | '.join(examples)}",
                    expected_value="missing=0",
                )
            )
    return issues


def validate_scenario_probability(raw_dir: Path, tolerance: float = 1e-6) -> List[ValidationIssue]:
    total = 0.0
    invalid = 0
    for row in iter_dict_rows(raw_dir / "scenarios.csv", ("scenario_probability",)):
        value = try_float(row["scenario_probability"])
        if value is None:
            invalid += 1
        else:
            total += value
    if invalid == 0 and abs(total - 1.0) <= tolerance:
        return [passed_issue("scenario_probability_sum", "scenarios.csv", f"Scenario probabilities sum to {total:.10f}.")]
    return [
        failed_issue(
            "scenario_probability_sum",
            "scenarios.csv",
            "Scenario probabilities must sum to 1.",
            observed_value=f"sum={total:.10f}; invalid={invalid}",
            expected_value=f"1 +/- {tolerance}",
        )
    ]


def validate_grid_counts(raw_dir: Path) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    scenarios = _unique_values(raw_dir / "scenarios.csv", "scenario_id")
    periods = _unique_values(raw_dir / "time_periods.csv", "period_id")
    machine_types = _unique_values(raw_dir / "machine_types.csv", "machine_type_id")
    baseline_rules = _unique_values(raw_dir / "baseline_rules.csv", "baseline_rule_id")

    demand_rows = _count_file_rows(raw_dir / "demand_scenarios.csv")
    expected_demand = len(scenarios) * len(periods) * len(machine_types)
    issues.append(_count_issue("demand_grid", "demand_scenarios.csv", demand_rows, expected_demand))

    quality_rows = _count_file_rows(raw_dir / "component_quality_scenarios.csv")
    quality_groups = {
        (row["machine_type_id"], row["component_type"])
        for row in iter_dict_rows(raw_dir / "component_quality_scenarios.csv", ("machine_type_id", "component_type"))
    }
    expected_quality = len(scenarios) * len(quality_groups)
    issues.append(_count_issue("component_quality_grid", "component_quality_scenarios.csv", quality_rows, expected_quality))

    perf_rows = _count_file_rows(raw_dir / "historical_performance.csv")
    months = _unique_values(raw_dir / "historical_performance.csv", "period_id")
    expected_perf = len(months) * len(machine_types) * len(baseline_rules)
    issues.append(_count_issue("historical_performance_grid", "historical_performance.csv", perf_rows, expected_perf))
    return issues


def _unique_values(path: Path, column: str) -> set[str]:
    return {row[column] for row in iter_dict_rows(path, (column,)) if row[column] != ""}


def _count_file_rows(path: Path) -> int:
    return sum(1 for _ in iter_dict_rows(path, ()))


def _count_issue(check_name: str, file_name: str, observed: int, expected: int) -> ValidationIssue:
    if observed == expected:
        return passed_issue(check_name, file_name, f"Observed row count {observed} matches expected grid.")
    return failed_issue(
        check_name,
        file_name,
        "Observed row count does not match expected grid.",
        observed_value=str(observed),
        expected_value=str(expected),
    )


def validate_quality_ranges(raw_dir: Path, tolerance: float = 1e-9) -> List[ValidationIssue]:
    ranges: List[Tuple[str, float, float]] = []
    for row in iter_dict_rows(raw_dir / "quality_states.csv", ("quality_state", "quality_score_min", "quality_score_max")):
        low = try_float(row["quality_score_min"])
        high = try_float(row["quality_score_max"])
        if low is None or high is None:
            return [
                failed_issue(
                    "quality_state_ranges",
                    "quality_states.csv",
                    "Quality range contains nonnumeric bounds.",
                    observed_value=str(row),
                )
            ]
        ranges.append((row["quality_state"], low, high))
    ranges.sort(key=lambda item: item[1])
    if not ranges:
        return [failed_issue("quality_state_ranges", "quality_states.csv", "No quality ranges found.")]
    if abs(ranges[0][1] - 0.0) > tolerance or abs(ranges[-1][2] - 1.0) > tolerance:
        return [
            failed_issue(
                "quality_state_ranges",
                "quality_states.csv",
                "Quality ranges must cover 0 to 1.",
                observed_value=str(ranges),
                expected_value="[0, 1]",
            )
        ]
    for previous, current in zip(ranges, ranges[1:]):
        if abs(previous[2] - current[1]) > tolerance:
            return [
                failed_issue(
                    "quality_state_ranges",
                    "quality_states.csv",
                    "Quality ranges have a gap or overlap.",
                    observed_value=str(ranges),
                    expected_value="contiguous non-overlapping intervals",
                )
            ]
    return [passed_issue("quality_state_ranges", "quality_states.csv", "A/B/C/D quality ranges cover 0-1 without gaps.")]


def validate_route_transition_probabilities(raw_dir: Path, tolerance: float = 1e-5, warning_tolerance: float = 1e-6) -> List[ValidationIssue]:
    file_name = "route_state_transition.csv"
    path = raw_dir / file_name
    columns = ("transition_prob_A", "transition_prob_B", "transition_prob_C", "transition_prob_D", "transition_prob_SCRAP")
    bad_count = 0
    warning_count = 0
    max_deviation = 0.0
    examples: List[str] = []
    for row_number, row in enumerate(iter_dict_rows(path, columns), start=2):
        values = [try_float(row[column]) or 0.0 for column in columns]
        deviation = abs(sum(values) - 1.0)
        max_deviation = max(max_deviation, deviation)
        if deviation > tolerance:
            bad_count += 1
            if len(examples) < 5:
                examples.append(f"row {row_number}: deviation={deviation}")
        elif deviation > warning_tolerance:
            warning_count += 1
    if bad_count == 0 and warning_count == 0:
        return [passed_issue("route_transition_probability_sum", file_name, f"All transition probability rows sum to 1 within {warning_tolerance}.")]
    if bad_count == 0:
        return [
            warning_issue(
                "route_transition_probability_sum",
                file_name,
                f"Transition probability sums are within failure tolerance {tolerance}, but {warning_count} rows exceed warning tolerance {warning_tolerance}.",
                observed_value=f"max_deviation={max_deviation}",
                expected_value=f"<= {tolerance}",
            )
        ]
    return [
        failed_issue(
            "route_transition_probability_sum",
            file_name,
            "Some transition probability rows exceed tolerance.",
            observed_value=f"bad_rows={bad_count}; max_deviation={max_deviation}; examples={' | '.join(examples)}",
            expected_value=f"<= {tolerance}",
        )
    ]


def validate_probability_ranges(raw_dir: Path) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    for file_name, columns in PROBABILITY_COLUMNS.items():
        path = raw_dir / file_name
        if not path.exists():
            continue
        header = set(read_header(path))
        present = [column for column in columns if column in header]
        invalid_count = 0
        examples: List[str] = []
        for row_number, row in enumerate(iter_dict_rows(path, present), start=2):
            for column in present:
                value = try_float(row[column])
                if value is None:
                    continue
                if value < -1e-12 or value > 1 + 1e-12:
                    invalid_count += 1
                    if len(examples) < 5:
                        examples.append(f"row {row_number} {column}={value}")
        if invalid_count == 0:
            issues.append(passed_issue("probability_range", file_name, f"Probability columns are within [0, 1]: {len(present)} columns checked."))
        else:
            issues.append(
                failed_issue(
                    "probability_range",
                    file_name,
                    "Probability columns contain values outside [0, 1].",
                    observed_value=f"invalid={invalid_count}; examples={' | '.join(examples)}",
                    expected_value="[0, 1]",
                )
            )
    return issues


def validate_nonnegative_ranges(raw_dir: Path) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    for file_name in EXPECTED_FILES:
        path = raw_dir / file_name
        if not path.exists():
            continue
        header = read_header(path)
        allowed_negative = set(NEGATIVE_ALLOWED_COLUMNS.get(file_name, ()))
        columns = [
            column
            for column in header
            if column not in allowed_negative
            and _looks_nonnegative_numeric(column)
            and not _looks_signed_measure(column)
        ]
        invalid_count = 0
        examples: List[str] = []
        for row_number, row in enumerate(iter_dict_rows(path, columns), start=2):
            for column in columns:
                value = try_float(row[column])
                if value is None:
                    continue
                if value < -1e-12:
                    invalid_count += 1
                    if len(examples) < 5:
                        examples.append(f"row {row_number} {column}={value}")
        if invalid_count == 0:
            issues.append(passed_issue("nonnegative_numeric_ranges", file_name, f"Nonnegative numeric keyword columns are valid: {len(columns)} columns checked."))
        else:
            issues.append(
                failed_issue(
                    "nonnegative_numeric_ranges",
                    file_name,
                    "Numeric columns expected to be nonnegative contain negative values.",
                    observed_value=f"invalid={invalid_count}; examples={' | '.join(examples)}",
                    expected_value=">= 0",
                )
            )
    return issues


def _looks_nonnegative_numeric(column: str) -> bool:
    lower = column.lower()
    return any(keyword in lower for keyword in NONNEGATIVE_COLUMN_KEYWORDS)


def _looks_signed_measure(column: str) -> bool:
    lower = column.lower()
    signed_markers = ("delta", "deviation", "shift", "error", "gap", "contribution")
    return any(marker in lower for marker in signed_markers)


def validate_baseline_weights(raw_dir: Path, tolerance: float = 1e-6) -> List[ValidationIssue]:
    invalid_count = 0
    max_deviation = 0.0
    examples: List[str] = []
    for row_number, row in enumerate(iter_dict_rows(raw_dir / "baseline_rules.csv", BASELINE_WEIGHT_COLUMNS), start=2):
        values = [try_float(row[column]) or 0.0 for column in BASELINE_WEIGHT_COLUMNS]
        deviation = abs(sum(values) - 1.0)
        max_deviation = max(max_deviation, deviation)
        if deviation > tolerance:
            invalid_count += 1
            if len(examples) < 5:
                examples.append(f"row {row_number}: sum={sum(values)}")
    if invalid_count == 0:
        return [passed_issue("baseline_weight_sum", "baseline_rules.csv", f"All baseline objective weights sum to 1 within {tolerance}.")]
    return [
        failed_issue(
            "baseline_weight_sum",
            "baseline_rules.csv",
            "Baseline objective weights do not sum to 1.",
            observed_value=f"invalid={invalid_count}; max_deviation={max_deviation}; examples={' | '.join(examples)}",
            expected_value=f"1 +/- {tolerance}",
        )
    ]


def validate_all(raw_dir: Path, entries: Sequence[CatalogueEntry]) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    issues.extend(validate_file_existence(entries))
    issues.extend(validate_catalogue_counts(entries))
    issues.extend(validate_required_columns(raw_dir))
    issues.extend(validate_primary_keys(raw_dir))
    issues.extend(validate_foreign_keys(raw_dir))
    issues.extend(validate_scenario_probability(raw_dir))
    issues.extend(validate_grid_counts(raw_dir))
    issues.extend(validate_quality_ranges(raw_dir))
    issues.extend(validate_route_transition_probabilities(raw_dir))
    issues.extend(validate_probability_ranges(raw_dir))
    issues.extend(validate_nonnegative_ranges(raw_dir))
    issues.extend(validate_baseline_weights(raw_dir))
    return issues
