"""ALNS + restricted MILP repair solver for Stage 8."""

from __future__ import annotations

import random
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, hstack, lil_matrix, vstack

from stage4_stochastic.checks import model_size_counts
from stage7_pareto.model import objective_values

from .config import Stage8Config
from .io_utils import read_stage7_reference
from .restrictions import (
    choose_operator,
    initial_state,
    operator_catalogue,
    operator_scores_frame,
    restricted_upper_bounds,
    restriction_summary,
    score_operator,
    mutate_state,
)
from .structures import Stage8HeuristicState, Stage8Instance, Stage8ModelData, Stage8RunResult, Stage8Solution


FEASIBLE_STATUSES = {0, 1}


def solve_matheuristic(instance: Stage8Instance, model_data: Stage8ModelData, config: Stage8Config) -> Stage8RunResult:
    """Run Stage 8 approximate augmented epsilon matheuristic."""

    start = time.perf_counter()
    rng = random.Random(int(config.random_seed))
    state0 = initial_state(instance, model_data, config)
    state0.operator_uses["initial_repair"] = state0.operator_uses.get("initial_repair", 0) + 1

    anchor = _solve_linear_repair("ANCHOR", "initial_repair", 0, model_data, config, state0)
    score_operator(state0, "initial_repair", anchor.feasible, anchor.feasible, anchor.feasible)
    epsilon_grid = _epsilon_grid(anchor, config)

    states_by_grid = {row.grid_id: state0.copy() for row in epsilon_grid.itertuples(index=False)}
    incumbents: Dict[str, Stage8Solution | None] = {}
    all_solutions: List[Stage8Solution] = [anchor]
    iteration_rows: List[Dict[str, object]] = []

    for grid in epsilon_grid.itertuples(index=False):
        solution = _solve_augmented_repair(
            str(grid.grid_id),
            "initial_repair",
            0,
            model_data,
            config,
            states_by_grid[str(grid.grid_id)],
            float(grid.env_epsilon),
            float(grid.assembly_epsilon),
            float(grid.env_range),
            float(grid.assembly_range),
            float(grid.delta),
        )
        incumbents[str(grid.grid_id)] = solution if solution.feasible else None
        all_solutions.append(solution)
        iteration_rows.append(_iteration_row(solution, accepted=solution.feasible, improved=solution.feasible))

    no_improve = 0
    grid_rows = list(epsilon_grid.itertuples(index=False))
    for iteration in range(1, int(config.max_iterations) + 1):
        grid = grid_rows[(iteration - 1) % len(grid_rows)]
        grid_id = str(grid.grid_id)
        operator_name = choose_operator(state0, rng)
        state0.operator_uses[operator_name] = state0.operator_uses.get(operator_name, 0) + 1
        candidate_state = mutate_state(states_by_grid[grid_id], operator_name, instance, model_data, config, rng)
        solution = _solve_augmented_repair(
            grid_id,
            operator_name,
            iteration,
            model_data,
            config,
            candidate_state,
            float(grid.env_epsilon),
            float(grid.assembly_epsilon),
            float(grid.env_range),
            float(grid.assembly_range),
            float(grid.delta),
        )
        incumbent = incumbents.get(grid_id)
        improved = bool(solution.feasible and (incumbent is None or _solution_key(solution) < _solution_key(incumbent) - 1e-6))
        accepted = bool(solution.feasible and (incumbent is None or improved or _threshold_accept(solution, incumbent, iteration, config)))
        if accepted:
            states_by_grid[grid_id] = candidate_state
            incumbents[grid_id] = solution if incumbent is None or improved else incumbent
        score_operator(state0, operator_name, accepted=accepted, improved=improved, feasible=solution.feasible)
        all_solutions.append(solution)
        iteration_rows.append(_iteration_row(solution, accepted=accepted, improved=improved))
        no_improve = 0 if improved else no_improve + 1
        if no_improve >= int(config.no_improve_limit):
            no_improve = 0

    repair_log = pd.DataFrame([solution.summary_row() for solution in all_solutions])
    iteration_log = pd.DataFrame(iteration_rows)
    feasible = repair_log[repair_log["feasible"] == True].copy()  # noqa: E712
    pareto, dominated = _pareto_split(feasible, repair_log)
    best = _best_solution(all_solutions)
    stage7_reference = read_stage7_reference(config.stage7_results_dir)
    comparison = _stage7_comparison(pareto, stage7_reference)
    checks = _solution_checks(model_data, anchor, repair_log, pareto)
    solve_seconds = time.perf_counter() - start

    success = bool(anchor.feasible and not pareto.empty and any(check["severity"] == "failed" for check in checks) is False)
    return Stage8RunResult(
        success=success,
        status_message="Stage 8 matheuristic workflow completed." if success else "Stage 8 matheuristic did not meet all acceptance checks.",
        solve_seconds=solve_seconds,
        instance_summary=instance.to_summary_dict(),
        heuristic_config=_config_summary(config),
        initial_restriction_summary=restriction_summary(state0, model_data),
        operator_catalogue=operator_catalogue(),
        benchmark_instances=_benchmark_instances_frame(config),
        model_summary=model_size_counts(model_data)
        | {"objective_vector_summary": model_data.objective_vector_summary, "restriction_summary": model_data.restriction_summary},
        iteration_log=iteration_log,
        repair_solve_log=repair_log,
        incumbent_solution_summary=best.summary_row() if best is not None else {},
        approx_pareto_front=pareto,
        dominated_solutions=dominated,
        operator_scores=operator_scores_frame(state0),
        large_benchmark_summary=pd.DataFrame(),
        solution_checks=checks,
        stage7_comparison=comparison,
    )


def _solve_linear_repair(
    solution_id: str,
    operator_name: str,
    iteration: int,
    model_data: Stage8ModelData,
    config: Stage8Config,
    state: Stage8HeuristicState,
) -> Stage8Solution:
    start = time.perf_counter()
    upper = restricted_upper_bounds(model_data, state)
    result = milp(
        c=model_data.objective_vectors["economic_risk"],
        integrality=model_data.integrality,
        bounds=Bounds(model_data.lower_bounds, upper),
        constraints=LinearConstraint(model_data.constraint_matrix, model_data.constraint_lhs, model_data.constraint_rhs),
        options={"time_limit": config.repair_time_limit, "mip_rel_gap": config.mip_rel_gap},
    )
    seconds = time.perf_counter() - start
    x = np.asarray(result.x, dtype=float) if result.x is not None else None
    return _make_solution(
        solution_id=solution_id,
        grid_id="ANCHOR",
        operator_name=operator_name,
        iteration=iteration,
        model_data=model_data,
        state=state,
        result=result,
        x=x,
        solve_seconds=seconds,
        slack_env=np.nan,
        slack_assembly=np.nan,
        env_epsilon=np.nan,
        assembly_epsilon=np.nan,
    )


def _solve_augmented_repair(
    grid_id: str,
    operator_name: str,
    iteration: int,
    model_data: Stage8ModelData,
    config: Stage8Config,
    state: Stage8HeuristicState,
    env_epsilon: float,
    assembly_epsilon: float,
    env_range: float,
    assembly_range: float,
    delta: float,
) -> Stage8Solution:
    start = time.perf_counter()
    n = len(model_data.variable_names)
    objective = np.concatenate(
        [
            model_data.objective_vectors["economic_risk"],
            np.array([-delta / max(env_range, 1.0), -delta / max(assembly_range, 1.0)], dtype=float),
        ]
    )
    matrix = hstack([model_data.constraint_matrix, csr_matrix((model_data.constraint_matrix.shape[0], 2), dtype=float)], format="csr")
    extra = lil_matrix((2, n + 2), dtype=float)
    extra[0, :n] = model_data.objective_vectors["environmental_impact"]
    extra[0, n] = 1.0
    extra[1, :n] = model_data.objective_vectors["assembly_quality_loss"]
    extra[1, n + 1] = 1.0
    matrix = vstack([matrix, extra.tocsr()], format="csr")
    lhs = np.concatenate([model_data.constraint_lhs, np.array([env_epsilon, assembly_epsilon], dtype=float)])
    rhs = np.concatenate([model_data.constraint_rhs, np.array([env_epsilon, assembly_epsilon], dtype=float)])
    lower = np.concatenate([model_data.lower_bounds, np.zeros(2, dtype=float)])
    upper = np.concatenate([restricted_upper_bounds(model_data, state), np.full(2, np.inf, dtype=float)])
    integrality = np.concatenate([model_data.integrality, np.zeros(2, dtype=int)])
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(matrix, lhs, rhs),
        options={"time_limit": config.repair_time_limit, "mip_rel_gap": config.mip_rel_gap},
    )
    seconds = time.perf_counter() - start
    x_base = None
    slack_env = np.nan
    slack_assembly = np.nan
    if result.x is not None:
        full_x = np.asarray(result.x, dtype=float)
        x_base = full_x[:n]
        slack_env = float(full_x[n])
        slack_assembly = float(full_x[n + 1])
    return _make_solution(
        solution_id=f"{grid_id}_I{iteration:03d}_{operator_name}",
        grid_id=grid_id,
        operator_name=operator_name,
        iteration=iteration,
        model_data=model_data,
        state=state,
        result=result,
        x=x_base,
        solve_seconds=seconds,
        slack_env=slack_env,
        slack_assembly=slack_assembly,
        env_epsilon=env_epsilon,
        assembly_epsilon=assembly_epsilon,
    )


def _make_solution(
    solution_id: str,
    grid_id: str,
    operator_name: str,
    iteration: int,
    model_data: Stage8ModelData,
    state: Stage8HeuristicState,
    result: object,
    x: np.ndarray | None,
    solve_seconds: float,
    slack_env: float,
    slack_assembly: float,
    env_epsilon: float,
    assembly_epsilon: float,
) -> Stage8Solution:
    feasible = bool(x is not None and int(result.status) in FEASIBLE_STATUSES)
    values = objective_values(model_data, x) if x is not None else _empty_objectives()
    residual = _base_constraint_residual(model_data, x) if x is not None else np.nan
    env_residual = values["environmental_impact"] + slack_env - env_epsilon if x is not None and np.isfinite(env_epsilon) else np.nan
    assembly_residual = values["assembly_quality_loss"] + slack_assembly - assembly_epsilon if x is not None and np.isfinite(assembly_epsilon) else np.nan
    return Stage8Solution(
        solution_id=solution_id,
        grid_id=grid_id,
        operator_name=operator_name,
        iteration=iteration,
        success=bool(getattr(result, "success", False)),
        status=int(result.status),
        message=str(result.message),
        feasible=feasible,
        solve_seconds=float(solve_seconds),
        objective_value=float(result.fun) if result.fun is not None else None,
        economic_risk=float(values["economic_risk"]),
        environmental_impact=float(values["environmental_impact"]),
        assembly_quality_loss=float(values["assembly_quality_loss"]),
        slack_env=float(slack_env) if np.isfinite(slack_env) else np.nan,
        slack_assembly=float(slack_assembly) if np.isfinite(slack_assembly) else np.nan,
        stage6_constraint_max_residual=float(residual) if np.isfinite(residual) else np.nan,
        env_constraint_residual=float(env_residual) if np.isfinite(env_residual) else np.nan,
        assembly_constraint_residual=float(assembly_residual) if np.isfinite(assembly_residual) else np.nan,
        allowed_route_count=len(state.allowed_route_keys),
        allowed_candidate_count=len(state.allowed_candidate_keys),
        allowed_pair_count=len(state.allowed_pair_keys),
        x=x,
    )


def _epsilon_grid(anchor: Stage8Solution, config: Stage8Config) -> pd.DataFrame:
    f2_anchor = max(0.0, float(anchor.environmental_impact)) if np.isfinite(anchor.environmental_impact) else 1.0
    f3_anchor = max(0.0, float(anchor.assembly_quality_loss)) if np.isfinite(anchor.assembly_quality_loss) else 1.0
    env_values, env_degenerate = _grid_values(0.0, f2_anchor, int(config.epsilon_grid_size))
    assembly_values, assembly_degenerate = _grid_values(0.0, f3_anchor, int(config.epsilon_grid_size))
    env_range = _safe_range(env_values)
    assembly_range = _safe_range(assembly_values)
    delta = float(config.augmentation_delta_factor) * max(1.0, abs(float(anchor.economic_risk)) if np.isfinite(anchor.economic_risk) else 1.0)
    rows = []
    grid_id = 0
    for env_rank, env_epsilon in enumerate(env_values):
        for assembly_rank, assembly_epsilon in enumerate(assembly_values):
            grid_id += 1
            rows.append(
                {
                    "grid_id": f"G{grid_id:03d}",
                    "env_rank": env_rank,
                    "assembly_rank": assembly_rank,
                    "env_epsilon": float(env_epsilon),
                    "assembly_epsilon": float(assembly_epsilon),
                    "env_range": float(env_range),
                    "assembly_range": float(assembly_range),
                    "delta": float(delta),
                    "env_degenerate_range": bool(env_degenerate),
                    "assembly_degenerate_range": bool(assembly_degenerate),
                }
            )
    return pd.DataFrame(rows)


def _pareto_split(feasible: pd.DataFrame, all_solutions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feasible = feasible[feasible["grid_id"] != "ANCHOR"].copy()
    if feasible.empty:
        return pd.DataFrame(), all_solutions.copy().reset_index(drop=True)
    feasible = feasible.drop_duplicates(subset=["economic_risk", "environmental_impact", "assembly_quality_loss"])
    values = feasible[["economic_risk", "environmental_impact", "assembly_quality_loss"]].to_numpy(dtype=float)
    nondominated = []
    for i, point in enumerate(values):
        dominated = False
        for j, other in enumerate(values):
            if i == j:
                continue
            if np.all(other <= point + 1e-6) and np.any(other < point - 1e-6):
                dominated = True
                break
        nondominated.append(not dominated)
    pareto = feasible[nondominated].copy().sort_values(["economic_risk", "environmental_impact", "assembly_quality_loss"]).reset_index(drop=True)
    dominated_ids = set(feasible.loc[[not flag for flag in nondominated], "solution_id"])
    dominated = all_solutions[(all_solutions["solution_id"].isin(dominated_ids)) | (all_solutions["feasible"] != True)].copy()  # noqa: E712
    return pareto, dominated.reset_index(drop=True)


def _solution_checks(
    model_data: Stage8ModelData,
    anchor: Stage8Solution,
    repair_log: pd.DataFrame,
    pareto_front: pd.DataFrame,
) -> List[Dict[str, object]]:
    feasible = repair_log[repair_log["feasible"] == True].copy()  # noqa: E712
    return [
        _check("objective_vectors_finite", _vectors_finite(model_data), "All Stage 7 objective vectors are finite and match variable count."),
        _check("initial_restricted_milp_feasible", anchor.feasible, f"Initial restricted anchor status={anchor.status}."),
        _check("repair_has_feasible_solution", not feasible.empty, f"Feasible repair solutions={len(feasible)}."),
        _check("approx_pareto_front_nonempty", not pareto_front.empty, f"Approximate Pareto points={len(pareto_front)}."),
        _check("stage6_constraint_residuals", _residuals_ok(feasible, "stage6_constraint_max_residual"), "Stage 6 base constraints are satisfied."),
        _check("epsilon_residuals", _epsilon_residuals_ok(feasible), "Epsilon equality residuals are within tolerance."),
        _check("nondominated_front_valid", _nondominated_valid(pareto_front), "Approximate Pareto front is nondominated."),
    ]


def _stage7_comparison(pareto: pd.DataFrame, reference: Dict[str, object]) -> Dict[str, object]:
    payload = {"stage7_reference_status": reference.get("status")}
    if reference.get("status") != "loaded" or pareto.empty:
        payload["economic_anchor_gap_pct"] = None
        payload["exact_pareto_point_count"] = reference.get("pareto_point_count")
        return payload
    exact_min = reference.get("min_economic_risk")
    approx_min = _safe_min(pareto, "economic_risk")
    payload.update(
        {
            "exact_pareto_point_count": reference.get("pareto_point_count"),
            "approx_pareto_point_count": int(len(pareto)),
            "exact_min_economic_risk": exact_min,
            "approx_min_economic_risk": approx_min,
            "economic_anchor_gap_pct": ((approx_min - exact_min) / abs(exact_min) * 100.0) if exact_min not in (None, 0) and approx_min is not None else None,
            "approx_min_environmental_impact": _safe_min(pareto, "environmental_impact"),
            "approx_min_assembly_quality_loss": _safe_min(pareto, "assembly_quality_loss"),
            "exact_min_environmental_impact": reference.get("min_environmental_impact"),
            "exact_min_assembly_quality_loss": reference.get("min_assembly_quality_loss"),
        }
    )
    return payload


def _iteration_row(solution: Stage8Solution, accepted: bool, improved: bool) -> Dict[str, object]:
    row = solution.summary_row()
    row["accepted"] = bool(accepted)
    row["improved"] = bool(improved)
    return row


def _solution_key(solution: Stage8Solution) -> float:
    if solution.objective_value is None or not np.isfinite(solution.objective_value):
        return np.inf
    return float(solution.objective_value)


def _threshold_accept(solution: Stage8Solution, incumbent: Stage8Solution, iteration: int, config: Stage8Config) -> bool:
    current = _solution_key(incumbent)
    candidate = _solution_key(solution)
    if not np.isfinite(current) or not np.isfinite(candidate):
        return False
    temperature = max(0.001, 0.03 * (1.0 - min(iteration, config.max_iterations) / max(1.0, config.max_iterations)))
    return candidate <= current * (1.0 + temperature)


def _best_solution(solutions: List[Stage8Solution]) -> Stage8Solution | None:
    feasible = [solution for solution in solutions if solution.feasible]
    if not feasible:
        return None
    return min(feasible, key=lambda sol: (float(sol.economic_risk), float(sol.environmental_impact), float(sol.assembly_quality_loss)))


def _vectors_finite(model_data: Stage8ModelData) -> bool:
    n = len(model_data.variable_names)
    return all(len(vector) == n and np.isfinite(vector).all() for vector in model_data.objective_vectors.values())


def _residuals_ok(frame: pd.DataFrame, column: str, tolerance: float = 1e-6) -> bool:
    feasible = frame[frame["feasible"] == True] if "feasible" in frame else frame  # noqa: E712
    if feasible.empty or column not in feasible.columns:
        return False
    value = pd.to_numeric(feasible[column], errors="coerce").dropna().abs()
    return bool(not value.empty and float(value.max()) <= tolerance)


def _epsilon_residuals_ok(frame: pd.DataFrame, tolerance: float = 1e-6) -> bool:
    feasible = frame[(frame["feasible"] == True) & (frame["grid_id"] != "ANCHOR")].copy()  # noqa: E712
    if feasible.empty:
        return False
    env = pd.to_numeric(feasible["env_constraint_residual"], errors="coerce").dropna().abs()
    assembly = pd.to_numeric(feasible["assembly_constraint_residual"], errors="coerce").dropna().abs()
    return bool(not env.empty and not assembly.empty and float(max(env.max(), assembly.max())) <= tolerance)


def _nondominated_valid(pareto: pd.DataFrame) -> bool:
    if pareto.empty:
        return False
    values = pareto[["economic_risk", "environmental_impact", "assembly_quality_loss"]].to_numpy(dtype=float)
    for i, point in enumerate(values):
        for j, other in enumerate(values):
            if i == j:
                continue
            if np.all(other <= point + 1e-6) and np.any(other < point - 1e-6):
                return False
    return True


def _base_constraint_residual(model_data: Stage8ModelData, x: np.ndarray) -> float:
    values = model_data.constraint_matrix @ x
    below = np.maximum(model_data.constraint_lhs - values, 0.0)
    above = np.maximum(values - model_data.constraint_rhs, 0.0)
    finite = np.concatenate([below[np.isfinite(below)], above[np.isfinite(above)]])
    return float(finite.max()) if finite.size else 0.0


def _grid_values(ideal: float, anchor: float, size: int) -> Tuple[np.ndarray, bool]:
    if size <= 1 or abs(anchor - ideal) <= 1e-9:
        return np.full(max(1, size), float(anchor)), True
    low = min(float(ideal), float(anchor))
    high = max(float(ideal), float(anchor))
    return np.linspace(low, high, size), False


def _safe_range(values: np.ndarray) -> float:
    value = float(np.max(values) - np.min(values)) if len(values) else 0.0
    return value if value > 1e-9 else 1.0


def _empty_objectives() -> Dict[str, float]:
    return {"economic_risk": np.nan, "environmental_impact": np.nan, "assembly_quality_loss": np.nan}


def _check(name: str, passed: bool, message: str) -> Dict[str, object]:
    return {"check_name": name, "severity": "passed" if passed else "failed", "message": message}


def _safe_min(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns or frame.empty:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.min()) if not values.empty else None


def _config_summary(config: Stage8Config) -> Dict[str, object]:
    return {key: str(value) if hasattr(value, "as_posix") else value for key, value in config.__dict__.items()}


def _benchmark_instances_frame(config: Stage8Config) -> pd.DataFrame:
    if config.benchmark_suite == "top5_52w":
        machine_types = config.benchmark_machine_types
    else:
        machine_types = (config.machine_type_id,)
    return pd.DataFrame(
        [
            {
                "benchmark_suite": config.benchmark_suite or "single_instance",
                "machine_type_id": machine_type,
                "period_start": config.period_start,
                "period_count": config.period_count,
                "processing_window_periods": config.processing_window_periods,
                "epsilon_grid_size": config.epsilon_grid_size,
                "max_iterations": config.max_iterations,
                "repair_time_limit": config.repair_time_limit,
            }
            for machine_type in machine_types
        ]
    )
