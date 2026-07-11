"""Shared-capacity extension experiment for Stage 10."""

from __future__ import annotations

import time
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import block_diag, csr_matrix, hstack, lil_matrix, vstack

from stage4_stochastic.aggregation import build_stage4_instance
from stage4_stochastic.config import Stage4Config
from stage4_stochastic.model import _resource_overtime_multiplier, build_model_data as build_stage4_model_data

from .config import Stage10Config
from .structures import SharedCapacityInstance, SharedCapacityModelData


def build_shared_capacity_instance(tables: Dict[str, pd.DataFrame], config: Stage10Config) -> SharedCapacityInstance:
    """Build single-machine Stage 4 instances and matrices for capacity coupling."""

    instances = {}
    model_data = {}
    for machine_type in config.machine_types:
        stage4_config = _stage4_config(config, machine_type)
        instance = build_stage4_instance(tables, stage4_config)
        instances[machine_type] = instance
        model_data[machine_type] = build_stage4_model_data(instance, stage4_config)
    first = instances[config.machine_types[0]]
    _validate_common_index_sets(instances)
    resources = tuple(resource for resource in config.shared_resources if _resource_present(resource, instances.values()))
    return SharedCapacityInstance(
        machine_types=tuple(config.machine_types),
        periods=list(first.periods),
        scenario_ids=list(first.scenario_ids),
        shared_resources=resources,
        instances=instances,
        model_data_by_machine=model_data,
    )


def build_shared_capacity_model(instance: SharedCapacityInstance, config: Stage10Config) -> SharedCapacityModelData:
    """Create a coupled MILP by replacing selected single-machine capacity rows with shared rows."""

    variable_names: List[str] = []
    objective_parts: List[np.ndarray] = []
    integrality_parts: List[np.ndarray] = []
    lower_parts: List[np.ndarray] = []
    upper_parts: List[np.ndarray] = []
    matrices = []
    lhs_parts = []
    rhs_parts = []
    constraint_names: List[str] = []
    machine_offsets: Dict[str, int] = {}
    machine_slices: Dict[str, Tuple[int, int]] = {}
    variable_groups: Dict[str, Dict[str, int]] = {}

    offset = 0
    for machine_type in instance.machine_types:
        model_data = instance.model_data_by_machine[machine_type]
        machine_offsets[machine_type] = offset
        machine_slices[machine_type] = (offset, offset + len(model_data.variable_names))
        variable_names.extend([f"{machine_type}::{name}" for name in model_data.variable_names])
        objective_parts.append(np.asarray(model_data.objective, dtype=float))
        integrality_parts.append(np.asarray(model_data.integrality, dtype=int))
        lower_parts.append(np.asarray(model_data.lower_bounds, dtype=float))
        upper_parts.append(np.asarray(model_data.upper_bounds, dtype=float))
        keep_mask = _nonshared_capacity_mask(model_data.constraint_names, instance.shared_resources)
        matrices.append(model_data.constraint_matrix[keep_mask, :].tocsr())
        lhs_parts.append(np.asarray(model_data.constraint_lhs, dtype=float)[keep_mask])
        rhs_parts.append(np.asarray(model_data.constraint_rhs, dtype=float)[keep_mask])
        constraint_names.extend([f"{machine_type}::{name}" for name, keep in zip(model_data.constraint_names, keep_mask) if keep])
        for group_name, group in model_data.variable_groups.items():
            target = variable_groups.setdefault(group_name, {})
            for key, index in group.items():
                target[f"{machine_type}|{key}"] = offset + int(index)
        offset += len(model_data.variable_names)

    base_variable_count = len(variable_names)
    shared_overtime: Dict[str, int] = {}
    probabilities = instance.instances[instance.machine_types[0]].scenario_sample.set_index("scenario_id")["saa_probability"].to_dict()
    for scenario_id in instance.scenario_ids:
        for resource in instance.shared_resources:
            for period in instance.periods:
                key = f"{scenario_id}|{resource}|{period}"
                shared_overtime[key] = len(variable_names)
                variable_names.append(f"shared_overtime[{scenario_id},{resource},{period}]")
                probability = float(probabilities.get(scenario_id, 0.0))
                objective_parts.append(np.array([probability * config.overtime_penalty_rmb_per_h * _resource_overtime_multiplier(resource)]))
                integrality_parts.append(np.array([0], dtype=int))
                lower_parts.append(np.array([0.0], dtype=float))
                upper_parts.append(np.array([np.inf], dtype=float))
    variable_groups["shared_overtime"] = shared_overtime

    objective = np.concatenate(objective_parts) if objective_parts else np.array([], dtype=float)
    integrality = np.concatenate(integrality_parts) if integrality_parts else np.array([], dtype=int)
    lower_bounds = np.concatenate(lower_parts) if lower_parts else np.array([], dtype=float)
    upper_bounds = np.concatenate(upper_parts) if upper_parts else np.array([], dtype=float)

    base_matrix = block_diag(matrices, format="csr") if matrices else csr_matrix((0, base_variable_count), dtype=float)
    extra_cols = len(variable_names) - base_variable_count
    extended_base = hstack([base_matrix, csr_matrix((base_matrix.shape[0], extra_cols), dtype=float)], format="csr")
    shared_matrix, shared_lhs, shared_rhs, shared_names, shared_rows, shared_terms = _shared_capacity_rows(
        instance,
        variable_groups,
        shared_overtime,
        len(variable_names),
    )
    matrix = vstack([extended_base, shared_matrix], format="csr")
    lhs = np.concatenate(lhs_parts + [shared_lhs]) if lhs_parts else shared_lhs
    rhs = np.concatenate(rhs_parts + [shared_rhs]) if rhs_parts else shared_rhs
    names = constraint_names + shared_names
    return SharedCapacityModelData(
        variable_names=variable_names,
        objective=objective,
        integrality=integrality,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        constraint_matrix=matrix,
        constraint_lhs=lhs,
        constraint_rhs=rhs,
        constraint_names=names,
        variable_groups=variable_groups,
        machine_offsets=machine_offsets,
        machine_variable_slices=machine_slices,
        shared_overtime=shared_overtime,
        shared_capacity_rows=shared_rows,
        shared_capacity_terms=shared_terms,
    )


def solve_shared_capacity_experiment(
    shared_instance: SharedCapacityInstance,
    shared_model: SharedCapacityModelData,
    config: Stage10Config,
) -> tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Solve independent and shared-capacity models and return comparison artifacts."""

    independent_rows = []
    independent_utilization_frames = []
    for machine_type in shared_instance.machine_types:
        instance = shared_instance.instances[machine_type]
        model_data = shared_instance.model_data_by_machine[machine_type]
        result, x, seconds = _solve_matrix(model_data, config)
        util = _single_machine_capacity_utilization(instance, model_data, x, shared_instance.shared_resources)
        independent_utilization_frames.append(util.assign(capacity_mode="independent_capacity", machine_type_id=machine_type))
        independent_rows.append(
            _single_machine_summary(
                "independent_capacity",
                machine_type,
                instance,
                model_data,
                x,
                result,
                seconds,
                util,
            )
        )

    shared_result, shared_x, shared_seconds = _solve_matrix(shared_model, config)
    shared_utilization = _shared_capacity_utilization(shared_instance, shared_model, shared_x)
    shared_row = _shared_summary(shared_instance, shared_model, shared_x, shared_result, shared_seconds, shared_utilization)
    comparison = pd.DataFrame(independent_rows + [_aggregate_independent(independent_rows), shared_row])
    solution_summary = {
        "independent_capacity": {
            "machine_count": len(independent_rows),
            "aggregate_objective_value": float(sum(float(row.get("objective_value", 0.0) or 0.0) for row in independent_rows)),
            "aggregate_expected_final_backlog_units": float(sum(float(row.get("expected_final_backlog_units", 0.0) or 0.0) for row in independent_rows)),
            "aggregate_expected_overtime_hours": float(sum(float(row.get("expected_overtime_hours", 0.0) or 0.0) for row in independent_rows)),
        },
        "shared_capacity": {
            "status": shared_row["solver_status"],
            "success": bool(shared_row["success"]),
            "objective_value": shared_row["objective_value"],
            "expected_final_backlog_units": shared_row["expected_final_backlog_units"],
            "expected_overtime_hours": shared_row["expected_overtime_hours"],
            "solve_seconds": shared_seconds,
        },
        "shared_capacity_row_count": int(len(shared_model.shared_capacity_rows)),
        "shared_capacity_resources": list(shared_instance.shared_resources),
    }
    utilization = pd.concat(independent_utilization_frames + [shared_utilization.assign(capacity_mode="shared_capacity", machine_type_id="ALL")], ignore_index=True)
    return solution_summary, comparison, utilization


def _stage4_config(config: Stage10Config, machine_type_id: str) -> Stage4Config:
    return Stage4Config(
        raw_dir=config.raw_dir,
        stage1_report=config.stage1_report,
        processed_dir=config.processed_dir,
        results_dir=config.results_dir,
        machine_type_id=machine_type_id,
        period_start=config.period_start,
        period_count=config.period_count,
        processing_window_periods=config.processing_window_periods,
        scenario_mode=config.scenario_mode,
        scenario_ids=config.scenario_ids,
        env_weight=config.env_weight,
        quality_weight=config.quality_weight,
        reliability_weight=config.reliability_weight,
        inventory_holding_rate=config.inventory_holding_rate,
        backlog_penalty_rmb_per_unit_period=config.backlog_penalty_rmb_per_unit_period,
        procurement_cost_multiplier=config.procurement_cost_multiplier,
        recourse_procurement_premium=config.recourse_procurement_premium,
        overtime_penalty_rmb_per_h=config.overtime_penalty_rmb_per_h,
        capacity_share_floor=config.capacity_share_floor,
        capacity_share_multiplier=config.capacity_share_multiplier,
        quality_floor=config.quality_floor,
        time_limit_seconds=config.time_limit_seconds,
        mip_rel_gap=config.mip_rel_gap,
    )


def _validate_common_index_sets(instances: Dict[str, object]) -> None:
    iterator = iter(instances.values())
    first = next(iterator)
    for instance in iterator:
        if list(instance.periods) != list(first.periods):
            raise ValueError("Shared-capacity instances must use the same period window.")
        if list(instance.scenario_ids) != list(first.scenario_ids):
            raise ValueError("Shared-capacity instances must use the same scenario sample.")


def _resource_present(resource: str, instances: Iterable[object]) -> bool:
    for instance in instances:
        if resource in set(instance.resource_types):
            return True
        if f"resource_h__{resource}" in instance.component_route_period_scenario_table.columns:
            return True
    return False


def _nonshared_capacity_mask(names: List[str], shared_resources: Tuple[str, ...]) -> np.ndarray:
    shared = set(shared_resources)
    keep = []
    for name in names:
        parsed = _parse_capacity_name(name)
        keep.append(not parsed or parsed[1] not in shared)
    return np.array(keep, dtype=bool)


def _parse_capacity_name(name: str) -> tuple[str, str, str] | None:
    if not name.startswith("capacity[") or not name.endswith("]"):
        return None
    parts = name.removeprefix("capacity[").removesuffix("]").split(",")
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


def _shared_capacity_rows(
    instance: SharedCapacityInstance,
    variable_groups: Dict[str, Dict[str, int]],
    shared_overtime: Dict[str, int],
    n_variables: int,
) -> tuple[csr_matrix, np.ndarray, np.ndarray, List[str], pd.DataFrame, Dict[str, Dict[int, float]]]:
    row_entries: List[Dict[int, float]] = []
    lhs: List[float] = []
    rhs: List[float] = []
    names: List[str] = []
    rows = []
    terms: Dict[str, Dict[int, float]] = {}
    for scenario_id in instance.scenario_ids:
        for resource in instance.shared_resources:
            resource_col = f"resource_h__{resource}"
            for period in instance.periods:
                row: Dict[int, float] = {}
                available = 0.0
                for machine_type in instance.machine_types:
                    machine = instance.instances[machine_type]
                    model_data = instance.model_data_by_machine[machine_type]
                    capacity = machine.scenario_capacity_table.set_index(["scenario_id", "resource_type", "period_id"])["available_regular_hours"].to_dict()
                    available += float(capacity.get((scenario_id, resource, period), 0.0))
                    if resource_col not in machine.component_route_period_scenario_table.columns:
                        continue
                    route_rows = machine.component_route_period_scenario_table[
                        (machine.component_route_period_scenario_table["scenario_id"] == scenario_id)
                        & (machine.component_route_period_scenario_table["period_id"] == period)
                    ]
                    for assignment in route_rows.itertuples(index=False):
                        value = float(getattr(assignment, resource_col, 0.0))
                        if abs(value) <= 1e-12:
                            continue
                        key = f"{machine_type}|{assignment.scenario_id}|{assignment.component_instance_id}|{assignment.route_id}|{assignment.period_id}"
                        row[variable_groups["x"][key]] = row.get(variable_groups["x"][key], 0.0) + value
                overtime_key = f"{scenario_id}|{resource}|{period}"
                row[shared_overtime[overtime_key]] = -1.0
                name = f"shared_capacity[{scenario_id},{resource},{period}]"
                row_entries.append(row)
                lhs.append(-np.inf)
                rhs.append(float(available))
                names.append(name)
                terms[name] = dict(row)
                rows.append(
                    {
                        "constraint_name": name,
                        "scenario_id": scenario_id,
                        "resource_type": resource,
                        "period_id": period,
                        "available_regular_hours": float(available),
                        "shared_overtime_variable": f"shared_overtime[{scenario_id},{resource},{period}]",
                        "shared_overtime_variable_index": int(shared_overtime[overtime_key]),
                        "route_term_count": int(max(0, len(row) - 1)),
                    }
                )
    matrix = lil_matrix((len(row_entries), n_variables), dtype=float)
    for row_number, entries in enumerate(row_entries):
        for col_number, value in entries.items():
            matrix[row_number, col_number] = value
    return matrix.tocsr(), np.array(lhs, dtype=float), np.array(rhs, dtype=float), names, pd.DataFrame(rows), terms


def _solve_matrix(model_data: object, config: Stage10Config) -> tuple[object, np.ndarray, float]:
    start = time.perf_counter()
    result = milp(
        c=np.asarray(model_data.objective, dtype=float),
        integrality=np.asarray(model_data.integrality, dtype=int),
        bounds=Bounds(np.asarray(model_data.lower_bounds, dtype=float), np.asarray(model_data.upper_bounds, dtype=float)),
        constraints=LinearConstraint(model_data.constraint_matrix, model_data.constraint_lhs, model_data.constraint_rhs),
        options={"time_limit": config.time_limit_seconds, "mip_rel_gap": config.mip_rel_gap},
    )
    seconds = time.perf_counter() - start
    x = np.asarray(result.x if result.x is not None else np.zeros(len(model_data.variable_names)), dtype=float)
    return result, x, seconds


def _single_machine_summary(
    capacity_mode: str,
    machine_type: str,
    instance: object,
    model_data: object,
    x: np.ndarray,
    result: object,
    solve_seconds: float,
    utilization: pd.DataFrame,
) -> Dict[str, object]:
    return {
        "capacity_mode": capacity_mode,
        "machine_type_id": machine_type,
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "success": bool(result.success or result.x is not None),
        "objective_value": _objective_value(model_data, x, result),
        "expected_assembled_units": _expected_assembled(instance, model_data, x),
        "expected_final_backlog_units": _expected_final_backlog(instance, model_data, x),
        "expected_overtime_hours": _expected_overtime(instance, model_data, x),
        "mean_shared_resource_utilization": _mean(utilization, "utilization_rate_regular"),
        "max_shared_resource_utilization": _max(utilization, "utilization_rate_regular"),
        "solve_seconds": float(solve_seconds),
    }


def _shared_summary(
    instance: SharedCapacityInstance,
    model_data: SharedCapacityModelData,
    x: np.ndarray,
    result: object,
    solve_seconds: float,
    utilization: pd.DataFrame,
) -> Dict[str, object]:
    assembled = 0.0
    backlog = 0.0
    for machine_type in instance.machine_types:
        offset = model_data.machine_offsets[machine_type]
        machine = instance.instances[machine_type]
        single_model = instance.model_data_by_machine[machine_type]
        assembled += _expected_assembled(machine, single_model, x[offset : offset + len(single_model.variable_names)])
        backlog += _expected_final_backlog(machine, single_model, x[offset : offset + len(single_model.variable_names)])
    return {
        "capacity_mode": "shared_capacity",
        "machine_type_id": "+".join(instance.machine_types),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "success": bool(result.success or result.x is not None),
        "objective_value": _objective_value(model_data, x, result),
        "expected_assembled_units": float(assembled),
        "expected_final_backlog_units": float(backlog),
        "expected_overtime_hours": _expected_shared_overtime(instance, model_data, x),
        "mean_shared_resource_utilization": _mean(utilization, "utilization_rate_regular"),
        "max_shared_resource_utilization": _max(utilization, "utilization_rate_regular"),
        "solve_seconds": float(solve_seconds),
    }


def _aggregate_independent(rows: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "capacity_mode": "independent_capacity_total",
        "machine_type_id": "+".join(str(row["machine_type_id"]) for row in rows),
        "solver_status": "aggregate",
        "solver_message": "aggregate of independently solved single-machine Stage 4 models",
        "success": all(bool(row.get("success")) for row in rows),
        "objective_value": float(sum(float(row.get("objective_value", 0.0) or 0.0) for row in rows)),
        "expected_assembled_units": float(sum(float(row.get("expected_assembled_units", 0.0) or 0.0) for row in rows)),
        "expected_final_backlog_units": float(sum(float(row.get("expected_final_backlog_units", 0.0) or 0.0) for row in rows)),
        "expected_overtime_hours": float(sum(float(row.get("expected_overtime_hours", 0.0) or 0.0) for row in rows)),
        "mean_shared_resource_utilization": float(np.nanmean([row.get("mean_shared_resource_utilization", np.nan) for row in rows])),
        "max_shared_resource_utilization": float(np.nanmax([row.get("max_shared_resource_utilization", np.nan) for row in rows])),
        "solve_seconds": float(sum(float(row.get("solve_seconds", 0.0) or 0.0) for row in rows)),
    }


def _objective_value(model_data: object, x: np.ndarray, result: object) -> float:
    if result.fun is not None:
        return float(result.fun)
    return float(np.dot(np.asarray(model_data.objective, dtype=float), x))


def _expected_final_backlog(instance: object, model_data: object, x: np.ndarray) -> float:
    probabilities = instance.scenario_sample.set_index("scenario_id")["saa_probability"].to_dict()
    last_period = instance.periods[-1]
    total = 0.0
    for scenario_id, probability in probabilities.items():
        index = model_data.variable_groups["backlog"].get(f"{scenario_id}|{last_period}")
        if index is not None:
            total += float(probability) * float(x[index])
    return float(total)


def _expected_assembled(instance: object, model_data: object, x: np.ndarray) -> float:
    probabilities = instance.scenario_sample.set_index("scenario_id")["saa_probability"].to_dict()
    total = 0.0
    for key, index in model_data.variable_groups["assemble"].items():
        scenario_id, _period = key.split("|")
        total += float(probabilities.get(scenario_id, 0.0)) * float(x[index])
    return float(total)


def _expected_overtime(instance: object, model_data: object, x: np.ndarray) -> float:
    probabilities = instance.scenario_sample.set_index("scenario_id")["saa_probability"].to_dict()
    total = 0.0
    for key, index in model_data.variable_groups["overtime"].items():
        scenario_id, _resource, _period = key.split("|")
        total += float(probabilities.get(scenario_id, 0.0)) * float(x[index])
    return float(total)


def _expected_shared_overtime(instance: SharedCapacityInstance, model_data: SharedCapacityModelData, x: np.ndarray) -> float:
    probabilities = instance.instances[instance.machine_types[0]].scenario_sample.set_index("scenario_id")["saa_probability"].to_dict()
    total = 0.0
    for key, index in model_data.shared_overtime.items():
        scenario_id, _resource, _period = key.split("|")
        total += float(probabilities.get(scenario_id, 0.0)) * float(x[index])
    return float(total)


def _single_machine_capacity_utilization(instance: object, model_data: object, x: np.ndarray, shared_resources: Tuple[str, ...]) -> pd.DataFrame:
    selected_value = {tuple(key.split("|")): x[index] for key, index in model_data.variable_groups["x"].items()}
    capacity = instance.scenario_capacity_table.set_index(["scenario_id", "resource_type", "period_id"])["available_regular_hours"].to_dict()
    rows = []
    for key, overtime_idx in model_data.variable_groups["overtime"].items():
        scenario_id, resource, period = key.split("|")
        if resource not in set(shared_resources):
            continue
        used = 0.0
        resource_col = f"resource_h__{resource}"
        if resource_col in instance.component_route_period_scenario_table.columns:
            period_rows = instance.component_route_period_scenario_table[
                (instance.component_route_period_scenario_table["scenario_id"] == scenario_id)
                & (instance.component_route_period_scenario_table["period_id"] == period)
            ]
            for assignment in period_rows.itertuples(index=False):
                used += float(getattr(assignment, resource_col, 0.0)) * selected_value.get(
                    (assignment.scenario_id, assignment.component_instance_id, assignment.route_id, assignment.period_id),
                    0.0,
                )
        available = float(capacity.get((scenario_id, resource, period), 0.0))
        overtime = float(x[overtime_idx])
        rows.append(
            {
                "scenario_id": scenario_id,
                "resource_type": resource,
                "period_id": period,
                "used_hours": used,
                "available_regular_hours": available,
                "overtime_hours": overtime,
                "utilization_rate_regular": used / available if available > 1e-9 else np.nan,
                "utilization_rate_with_overtime": used / (available + overtime) if available + overtime > 1e-9 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _shared_capacity_utilization(instance: SharedCapacityInstance, model_data: SharedCapacityModelData, x: np.ndarray) -> pd.DataFrame:
    rows = []
    for meta in model_data.shared_capacity_rows.itertuples(index=False):
        entries = model_data.shared_capacity_terms[str(meta.constraint_name)]
        overtime_index = int(meta.shared_overtime_variable_index)
        used = sum(float(value) * float(x[index]) for index, value in entries.items() if index != overtime_index)
        overtime = float(x[overtime_index])
        available = float(meta.available_regular_hours)
        rows.append(
            {
                "constraint_name": meta.constraint_name,
                "scenario_id": meta.scenario_id,
                "resource_type": meta.resource_type,
                "period_id": meta.period_id,
                "used_hours": used,
                "available_regular_hours": available,
                "overtime_hours": overtime,
                "utilization_rate_regular": used / available if available > 1e-9 else np.nan,
                "utilization_rate_with_overtime": used / (available + overtime) if available + overtime > 1e-9 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _mean(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _max(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.max()) if not values.empty else None
