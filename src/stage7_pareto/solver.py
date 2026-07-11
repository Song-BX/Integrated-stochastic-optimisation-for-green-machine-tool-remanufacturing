"""Payoff table, epsilon-grid solving, and Pareto extraction for Stage 7."""

from __future__ import annotations

import time
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, hstack, lil_matrix, vstack

from stage4_stochastic.checks import model_size_counts

from .config import Stage7Config
from .io_utils import read_stage6_solution_summary
from .model import objective_values
from .structures import Stage7Instance, Stage7ModelData, Stage7Solution


FEASIBLE_STATUSES = {0, 1}


def solve_pareto(
    instance: Stage7Instance,
    model_data: Stage7ModelData,
    config: Stage7Config,
) -> Stage7Solution:
    start = time.perf_counter()
    payoff, anchors = _payoff_table(model_data, config)
    epsilon_grid = _epsilon_grid(payoff, config)
    grid_summary = _solve_grid(model_data, config, epsilon_grid, anchors)
    pareto_front, dominated = _pareto_split(grid_summary)
    representatives = _representative_solutions(pareto_front)
    checks = _solution_checks(model_data, payoff, epsilon_grid, grid_summary, pareto_front)
    solve_seconds = time.perf_counter() - start
    success = bool((pd.to_numeric(grid_summary["feasible"], errors="coerce").fillna(0).astype(int) == 1).any() and not pareto_front.empty)
    return Stage7Solution(
        success=success,
        status_message="Stage 7 Pareto workflow completed." if success else "No feasible Pareto point found.",
        solve_seconds=solve_seconds,
        payoff_table=payoff,
        epsilon_grid=epsilon_grid,
        grid_solution_summary=grid_summary,
        pareto_front=pareto_front,
        dominated_solutions=dominated,
        representative_solutions=representatives,
        solution_checks=checks,
        model_summary=model_size_counts(model_data) | {"objective_vector_summary": model_data.objective_vector_summary},
        stage6_comparison=read_stage6_solution_summary(config.stage6_results_dir),
    )


def _payoff_table(model_data: Stage7ModelData, config: Stage7Config) -> tuple[pd.DataFrame, Dict[str, object]]:
    rows = []
    primary = "economic_risk"
    anchor = _solve_linear_objective(model_data, model_data.objective_vectors[primary], config)
    anchor_values = objective_values(model_data, anchor["x"]) if anchor["x"] is not None else _empty_objectives()
    rows.append(_payoff_row("economic_risk_anchor", anchor, anchor_values, fallback_used=False))
    f1_anchor = float(anchor_values[primary])
    allowance_rhs = f1_anchor * (1.0 + float(config.payoff_cost_allowance))
    anchors = {"economic_risk_anchor": anchor, "f1_anchor": f1_anchor}

    for objective_name, label in [("environmental_impact", "environmental_ideal"), ("assembly_quality_loss", "assembly_ideal")]:
        result = _solve_linear_objective(
            model_data,
            model_data.objective_vectors[objective_name],
            config,
            extra_upper_rows=[(model_data.objective_vectors[primary], allowance_rhs, "payoff_cost_allowance")],
        )
        fallback_used = result["x"] is None or not _is_feasible_status(result["status"])
        values = objective_values(model_data, result["x"]) if not fallback_used else anchor_values
        rows.append(_payoff_row(label, result if not fallback_used else anchor, values, fallback_used=fallback_used))
        anchors[label] = result if not fallback_used else anchor

    return pd.DataFrame(rows), anchors


def _epsilon_grid(payoff: pd.DataFrame, config: Stage7Config) -> pd.DataFrame:
    data = payoff.set_index("payoff_name")
    f2_anchor = float(data.loc["economic_risk_anchor", "environmental_impact"])
    f3_anchor = float(data.loc["economic_risk_anchor", "assembly_quality_loss"])
    f2_ideal = float(data.loc["environmental_ideal", "environmental_impact"])
    f3_ideal = float(data.loc["assembly_ideal", "assembly_quality_loss"])
    env_values, env_degenerate = _grid_values(f2_ideal, f2_anchor, int(config.epsilon_grid_size_env))
    assembly_values, assembly_degenerate = _grid_values(f3_ideal, f3_anchor, int(config.epsilon_grid_size_assembly))
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
                    "env_degenerate_range": bool(env_degenerate),
                    "assembly_degenerate_range": bool(assembly_degenerate),
                }
            )
    return pd.DataFrame(rows)


def _solve_grid(
    model_data: Stage7ModelData,
    config: Stage7Config,
    epsilon_grid: pd.DataFrame,
    anchors: Dict[str, object],
) -> pd.DataFrame:
    f1_anchor = float(anchors["f1_anchor"])
    env_range = _safe_range(epsilon_grid["env_epsilon"])
    assembly_range = _safe_range(epsilon_grid["assembly_epsilon"])
    delta = float(config.augmentation_delta_factor) * max(1.0, abs(f1_anchor))
    rows = []
    for grid in epsilon_grid.itertuples(index=False):
        result = _solve_augmented_grid_point(
            model_data,
            config,
            float(grid.env_epsilon),
            float(grid.assembly_epsilon),
            env_range,
            assembly_range,
            delta,
        )
        x = result["x_base"]
        values = objective_values(model_data, x) if x is not None else _empty_objectives()
        base_residual = _base_constraint_residual(model_data, x) if x is not None else np.nan
        env_residual = values["environmental_impact"] + float(result.get("slack_env", 0.0)) - float(grid.env_epsilon) if x is not None else np.nan
        assembly_residual = values["assembly_quality_loss"] + float(result.get("slack_assembly", 0.0)) - float(grid.assembly_epsilon) if x is not None else np.nan
        rows.append(
            {
                "grid_id": grid.grid_id,
                "env_rank": int(grid.env_rank),
                "assembly_rank": int(grid.assembly_rank),
                "env_epsilon": float(grid.env_epsilon),
                "assembly_epsilon": float(grid.assembly_epsilon),
                "success": bool(result["success"]),
                "status": int(result["status"]),
                "message": str(result["message"]),
                "feasible": int(result["x_base"] is not None and _is_feasible_status(result["status"])),
                "objective_augmented": float(result["fun"]) if result["fun"] is not None else np.nan,
                "economic_risk": values["economic_risk"],
                "environmental_impact": values["environmental_impact"],
                "assembly_quality_loss": values["assembly_quality_loss"],
                "slack_env": float(result.get("slack_env", np.nan)),
                "slack_assembly": float(result.get("slack_assembly", np.nan)),
                "stage6_constraint_max_residual": float(base_residual) if np.isfinite(base_residual) else np.nan,
                "env_constraint_residual": float(env_residual) if np.isfinite(env_residual) else np.nan,
                "assembly_constraint_residual": float(assembly_residual) if np.isfinite(assembly_residual) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _solve_augmented_grid_point(
    model_data: Stage7ModelData,
    config: Stage7Config,
    env_epsilon: float,
    assembly_epsilon: float,
    env_range: float,
    assembly_range: float,
    delta: float,
) -> Dict[str, object]:
    n = len(model_data.variable_names)
    objective = np.concatenate(
        [
            model_data.objective_vectors["economic_risk"],
            np.array([-delta / env_range, -delta / assembly_range], dtype=float),
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
    upper = np.concatenate([model_data.upper_bounds, np.full(2, np.inf, dtype=float)])
    integrality = np.concatenate([model_data.integrality, np.zeros(2, dtype=int)])
    result = _run_milp(objective, integrality, lower, upper, matrix, lhs, rhs, config)
    x = None
    slack_env = np.nan
    slack_assembly = np.nan
    if result.x is not None:
        full_x = np.asarray(result.x, dtype=float)
        x = full_x[:n]
        slack_env = float(full_x[n])
        slack_assembly = float(full_x[n + 1])
    return {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "fun": float(result.fun) if result.fun is not None else None,
        "x_base": x,
        "slack_env": slack_env,
        "slack_assembly": slack_assembly,
    }


def _solve_linear_objective(
    model_data: Stage7ModelData,
    objective: np.ndarray,
    config: Stage7Config,
    extra_upper_rows: list[tuple[np.ndarray, float, str]] | None = None,
) -> Dict[str, object]:
    matrix = model_data.constraint_matrix
    lhs = model_data.constraint_lhs
    rhs = model_data.constraint_rhs
    if extra_upper_rows:
        extra = lil_matrix((len(extra_upper_rows), len(model_data.variable_names)), dtype=float)
        extra_lhs = []
        extra_rhs = []
        for row_number, (row_vector, upper_bound, _name) in enumerate(extra_upper_rows):
            extra[row_number, :] = row_vector
            extra_lhs.append(-np.inf)
            extra_rhs.append(float(upper_bound))
        matrix = vstack([matrix, extra.tocsr()], format="csr")
        lhs = np.concatenate([lhs, np.asarray(extra_lhs, dtype=float)])
        rhs = np.concatenate([rhs, np.asarray(extra_rhs, dtype=float)])
    result = _run_milp(objective, model_data.integrality, model_data.lower_bounds, model_data.upper_bounds, matrix, lhs, rhs, config)
    return {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "fun": float(result.fun) if result.fun is not None else None,
        "x": np.asarray(result.x, dtype=float) if result.x is not None else None,
    }


def _run_milp(
    objective: np.ndarray,
    integrality: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    matrix: csr_matrix,
    lhs: np.ndarray,
    rhs: np.ndarray,
    config: Stage7Config,
):
    return milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(matrix, lhs, rhs),
        options={"time_limit": config.time_limit_per_solve, "mip_rel_gap": config.mip_rel_gap},
    )


def _payoff_row(name: str, result: Dict[str, object], values: Dict[str, float], fallback_used: bool) -> Dict[str, object]:
    return {
        "payoff_name": name,
        "success": bool(result["success"]),
        "status": int(result["status"]),
        "message": str(result["message"]),
        "fallback_used": bool(fallback_used),
        "economic_risk": float(values["economic_risk"]),
        "environmental_impact": float(values["environmental_impact"]),
        "assembly_quality_loss": float(values["assembly_quality_loss"]),
    }


def _pareto_split(grid_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feasible = grid_summary[grid_summary["feasible"] == 1].copy()
    if feasible.empty:
        return pd.DataFrame(), grid_summary.copy()
    feasible = feasible.drop_duplicates(subset=["economic_risk", "environmental_impact", "assembly_quality_loss"])
    nondominated_mask = []
    values = feasible[["economic_risk", "environmental_impact", "assembly_quality_loss"]].to_numpy(dtype=float)
    for i, point in enumerate(values):
        dominated = False
        for j, other in enumerate(values):
            if i == j:
                continue
            if np.all(other <= point + 1e-6) and np.any(other < point - 1e-6):
                dominated = True
                break
        nondominated_mask.append(not dominated)
    pareto = feasible[nondominated_mask].copy().sort_values(["economic_risk", "environmental_impact", "assembly_quality_loss"]).reset_index(drop=True)
    dominated_ids = set(feasible.loc[[not flag for flag in nondominated_mask], "grid_id"])
    dominated = grid_summary[grid_summary["grid_id"].isin(dominated_ids) | (grid_summary["feasible"] != 1)].copy().reset_index(drop=True)
    return pareto, dominated


def _representative_solutions(pareto: pd.DataFrame) -> Dict[str, object]:
    if pareto.empty:
        return {}
    reps = {}
    for objective in ["economic_risk", "environmental_impact", "assembly_quality_loss"]:
        idx = pareto[objective].astype(float).idxmin()
        reps[f"minimum_{objective}"] = pareto.loc[idx].to_dict()
    return reps


def _solution_checks(
    model_data: Stage7ModelData,
    payoff: pd.DataFrame,
    epsilon_grid: pd.DataFrame,
    grid_summary: pd.DataFrame,
    pareto_front: pd.DataFrame,
) -> list[Dict[str, object]]:
    checks = [
        _check("objective_vectors_finite", _vectors_finite(model_data), "All objective vectors have finite coefficients."),
        _check("payoff_anchor_feasible", bool(payoff.iloc[0]["status"] in FEASIBLE_STATUSES), "Economic-risk anchor payoff is feasible or optimal."),
        _check("epsilon_grid_size", len(epsilon_grid) == 25, f"Epsilon grid rows={len(epsilon_grid)}."),
        _check("grid_has_feasible_solution", bool((grid_summary["feasible"] == 1).any()), "At least one epsilon-grid point is feasible."),
        _check("pareto_front_nonempty", not pareto_front.empty, f"Pareto front points={len(pareto_front)}."),
        _check("stage6_constraint_residuals", _stage6_residuals_ok(grid_summary), "Stage 6 base constraints are satisfied for feasible grid points."),
        _check("epsilon_residuals", _epsilon_residuals_ok(grid_summary), "Epsilon equality residuals are within tolerance for feasible points."),
    ]
    return checks


def _vectors_finite(model_data: Stage7ModelData) -> bool:
    n = len(model_data.variable_names)
    return all(len(vector) == n and np.isfinite(vector).all() for vector in model_data.objective_vectors.values())


def _epsilon_residuals_ok(grid_summary: pd.DataFrame, tolerance: float = 1e-6) -> bool:
    feasible = grid_summary[grid_summary["feasible"] == 1]
    if feasible.empty:
        return False
    env = pd.to_numeric(feasible["env_constraint_residual"], errors="coerce").fillna(np.inf).abs().max()
    assembly = pd.to_numeric(feasible["assembly_constraint_residual"], errors="coerce").fillna(np.inf).abs().max()
    return float(max(env, assembly)) <= tolerance


def _stage6_residuals_ok(grid_summary: pd.DataFrame, tolerance: float = 1e-6) -> bool:
    feasible = grid_summary[grid_summary["feasible"] == 1]
    if feasible.empty:
        return False
    residual = pd.to_numeric(feasible["stage6_constraint_max_residual"], errors="coerce").fillna(np.inf).max()
    return float(residual) <= tolerance


def _base_constraint_residual(model_data: Stage7ModelData, x: np.ndarray) -> float:
    values = model_data.constraint_matrix @ x
    below = np.maximum(model_data.constraint_lhs - values, 0.0)
    above = np.maximum(values - model_data.constraint_rhs, 0.0)
    finite = np.concatenate([below[np.isfinite(below)], above[np.isfinite(above)]])
    return float(finite.max()) if finite.size else 0.0


def _check(name: str, passed: bool, message: str) -> Dict[str, object]:
    return {"check_name": name, "severity": "passed" if passed else "failed", "message": message}


def _grid_values(ideal: float, anchor: float, size: int) -> tuple[np.ndarray, bool]:
    if size <= 1 or abs(anchor - ideal) <= 1e-9:
        return np.full(max(1, size), float(anchor)), True
    low = min(float(ideal), float(anchor))
    high = max(float(ideal), float(anchor))
    return np.linspace(low, high, size), False


def _safe_range(values: Iterable[float]) -> float:
    series = pd.Series(values, dtype=float)
    value = float(series.max() - series.min())
    return value if value > 1e-9 else 1.0


def _empty_objectives() -> Dict[str, float]:
    return {"economic_risk": np.nan, "environmental_impact": np.nan, "assembly_quality_loss": np.nan}


def _is_feasible_status(status: int) -> bool:
    return int(status) in FEASIBLE_STATUSES
