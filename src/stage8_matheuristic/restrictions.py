"""Candidate-pool construction and adaptive restrictions for Stage 8."""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .config import Stage8Config
from .structures import Stage8HeuristicState, Stage8Instance, Stage8ModelData


OPERATORS = [
    "initial_repair",
    "route_expand",
    "route_swap",
    "pair_expand",
    "quality_repair",
    "carbon_focus",
    "risk_focus",
    "random_shake",
]


def operator_catalogue() -> pd.DataFrame:
    """Return the Stage 8 operator catalogue."""

    descriptions = {
        "initial_repair": "Solve the conservative initial restricted MILP.",
        "route_expand": "Open additional low-cost route-period choices for selected components.",
        "route_swap": "Replace part of the route pool with alternative feasible routes.",
        "pair_expand": "Open more compatible assembly pairs for shortfall-prone requirements.",
        "quality_repair": "Prefer pairs with low dimension, quality, and life-gap loss.",
        "carbon_focus": "Prefer low-carbon route and procurement alternatives.",
        "risk_focus": "Prefer routes with low warranty/tail-risk coefficients.",
        "random_shake": "Seeded random perturbation of route and pair pools.",
    }
    return pd.DataFrame([{"operator_name": name, "description": descriptions[name]} for name in OPERATORS])


def initial_state(instance: Stage8Instance, model_data: Stage8ModelData, config: Stage8Config) -> Stage8HeuristicState:
    """Build the conservative initial candidate pools."""

    state = Stage8HeuristicState(
        operator_scores={name: 0.0 for name in OPERATORS},
        operator_uses={name: 0 for name in OPERATORS},
        operator_successes={name: 0 for name in OPERATORS},
    )
    state.allowed_route_keys = _initial_route_keys(instance, model_data, config)
    state.allowed_candidate_keys = _initial_candidate_keys(instance, model_data, config)
    state.allowed_pair_keys = _initial_pair_keys(instance, model_data, config)
    return state


def mutate_state(
    state: Stage8HeuristicState,
    operator_name: str,
    instance: Stage8Instance,
    model_data: Stage8ModelData,
    config: Stage8Config,
    rng: random.Random,
) -> Stage8HeuristicState:
    """Return a mutated copy of the candidate pools for one ALNS iteration."""

    next_state = state.copy()
    next_state.operator_uses[operator_name] = next_state.operator_uses.get(operator_name, 0) + 1
    if operator_name == "route_expand":
        _expand_routes(next_state, instance, model_data, config, rng, mode="economic")
    elif operator_name == "route_swap":
        _swap_routes(next_state, instance, model_data, config, rng)
    elif operator_name == "pair_expand":
        _expand_pairs(next_state, instance, model_data, config, rng, mode="coverage")
    elif operator_name == "quality_repair":
        _expand_pairs(next_state, instance, model_data, config, rng, mode="quality")
    elif operator_name == "carbon_focus":
        _expand_routes(next_state, instance, model_data, config, rng, mode="carbon")
    elif operator_name == "risk_focus":
        _expand_routes(next_state, instance, model_data, config, rng, mode="risk")
    elif operator_name == "random_shake":
        _random_shake(next_state, instance, model_data, config, rng)
    return next_state


def restricted_upper_bounds(model_data: Stage8ModelData, state: Stage8HeuristicState) -> np.ndarray:
    """Return model upper bounds with unopened binary pools fixed to zero."""

    upper = np.asarray(model_data.upper_bounds, dtype=float).copy()
    _close_unallowed(upper, model_data.variable_groups.get("x", {}), state.allowed_route_keys)
    _close_unallowed(upper, model_data.variable_groups.get("select_candidate", {}), state.allowed_candidate_keys)
    _close_unallowed(upper, model_data.variable_groups.get("select_pair", {}), state.allowed_pair_keys)
    return upper


def restriction_summary(state: Stage8HeuristicState, model_data: Stage8ModelData) -> pd.DataFrame:
    """Summarize current allowed/unallowed variable counts."""

    rows = []
    for group_name, allowed in [
        ("x", state.allowed_route_keys),
        ("select_candidate", state.allowed_candidate_keys),
        ("select_pair", state.allowed_pair_keys),
    ]:
        total = len(model_data.variable_groups.get(group_name, {}))
        rows.append(
            {
                "variable_group": group_name,
                "allowed_count": int(len(allowed)),
                "total_count": int(total),
                "restricted_count": int(max(0, total - len(allowed))),
                "allowed_share": float(len(allowed) / total) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def score_operator(
    state: Stage8HeuristicState,
    operator_name: str,
    accepted: bool,
    improved: bool,
    feasible: bool,
) -> None:
    """Update adaptive operator scores."""

    reward = 0.0
    if feasible:
        reward += 1.0
    if accepted:
        reward += 2.0
    if improved:
        reward += 4.0
    state.operator_scores[operator_name] = state.operator_scores.get(operator_name, 0.0) + reward
    if feasible:
        state.operator_successes[operator_name] = state.operator_successes.get(operator_name, 0) + 1


def operator_scores_frame(state: Stage8HeuristicState) -> pd.DataFrame:
    """Return operator scores and use counts as a DataFrame."""

    rows = []
    for name in OPERATORS:
        uses = int(state.operator_uses.get(name, 0))
        successes = int(state.operator_successes.get(name, 0))
        rows.append(
            {
                "operator_name": name,
                "uses": uses,
                "successes": successes,
                "score": float(state.operator_scores.get(name, 0.0)),
                "success_rate": successes / uses if uses else 0.0,
            }
        )
    return pd.DataFrame(rows)


def choose_operator(state: Stage8HeuristicState, rng: random.Random) -> str:
    """Select an ALNS operator with score-biased roulette sampling."""

    candidates = [name for name in OPERATORS if name != "initial_repair"]
    weights = [max(1.0, state.operator_scores.get(name, 0.0) + 1.0) for name in candidates]
    return rng.choices(candidates, weights=weights, k=1)[0]


def _initial_route_keys(instance: Stage8Instance, model_data: Stage8ModelData, config: Stage8Config) -> set[str]:
    scores = _route_scores(instance, model_data)
    if scores.empty:
        return set(model_data.variable_groups.get("x", {}).keys())
    selected = []
    limit = max(1, int(config.initial_routes_per_component))
    for (_scenario_id, component_id), group in scores.groupby(["scenario_id", "component_instance_id"], dropna=False):
        selected.extend(group.sort_values("score").head(limit)["key"].tolist())
    return set(_existing_keys(selected, model_data.variable_groups.get("x", {})))


def _initial_candidate_keys(instance: Stage8Instance, model_data: Stage8ModelData, config: Stage8Config) -> set[str]:
    candidates = instance.assembly_candidate_pool.copy()
    if candidates.empty:
        return set(model_data.variable_groups.get("select_candidate", {}).keys())
    candidates["candidate_score"] = _candidate_score(candidates)
    selected_ids = set()
    new_ids = candidates[pd.to_numeric(candidates.get("new_backup_candidate_flag", 0), errors="coerce").fillna(0).astype(int) == 1][
        "assembly_candidate_id"
    ]
    selected_ids.update(str(value) for value in new_ids)
    target = max(len(selected_ids), int(np.ceil(len(candidates) * float(config.initial_candidate_fraction))))
    selected_ids.update(candidates.sort_values("candidate_score").head(target)["assembly_candidate_id"].astype(str).tolist())
    keys = [
        key
        for key in model_data.variable_groups.get("select_candidate", {})
        if _key_tail(key) in selected_ids
    ]
    return set(keys)


def _initial_pair_keys(instance: Stage8Instance, model_data: Stage8ModelData, config: Stage8Config) -> set[str]:
    pairs = _pair_scores(instance)
    if pairs.empty:
        return set(model_data.variable_groups.get("select_pair", {}).keys())
    selected_ids = set()
    limit = max(1, int(config.initial_pairs_per_requirement))
    for _requirement_id, group in pairs.groupby("assembly_requirement_id", dropna=False):
        selected_ids.update(group.sort_values("score").head(limit)["compatibility_id"].astype(str).tolist())
    keys = [key for key in model_data.variable_groups.get("select_pair", {}) if _key_tail(key) in selected_ids]
    return set(keys)


def _expand_routes(
    state: Stage8HeuristicState,
    instance: Stage8Instance,
    model_data: Stage8ModelData,
    config: Stage8Config,
    rng: random.Random,
    mode: str,
) -> None:
    scores = _route_scores(instance, model_data)
    if scores.empty:
        state.allowed_route_keys.update(model_data.variable_groups.get("x", {}).keys())
        return
    if mode == "carbon" and "carbon_score" in scores.columns:
        scores = scores.sort_values(["carbon_score", "score"])
    elif mode == "risk" and "risk_score" in scores.columns:
        scores = scores.sort_values(["risk_score", "score"])
    else:
        scores = scores.sort_values("score")
    unopened = [key for key in scores["key"].tolist() if key not in state.allowed_route_keys]
    count = _expand_count(len(unopened), config.route_expand_fraction, minimum=1)
    state.allowed_route_keys.update(_sample_front(unopened, count, rng))
    _cap_routes_per_component(state, scores, config)


def _swap_routes(
    state: Stage8HeuristicState,
    instance: Stage8Instance,
    model_data: Stage8ModelData,
    config: Stage8Config,
    rng: random.Random,
) -> None:
    if not state.allowed_route_keys:
        _expand_routes(state, instance, model_data, config, rng, mode="economic")
        return
    removable = list(state.allowed_route_keys)
    remove_count = min(len(removable), max(1, int(len(removable) * 0.08)))
    for key in rng.sample(removable, remove_count):
        state.allowed_route_keys.discard(key)
    _expand_routes(state, instance, model_data, config, rng, mode="economic")


def _expand_pairs(
    state: Stage8HeuristicState,
    instance: Stage8Instance,
    model_data: Stage8ModelData,
    config: Stage8Config,
    rng: random.Random,
    mode: str,
) -> None:
    pairs = _pair_scores(instance)
    if pairs.empty:
        state.allowed_pair_keys.update(model_data.variable_groups.get("select_pair", {}).keys())
        return
    if mode == "quality":
        pairs = pairs.sort_values(["quality_score", "score"])
    else:
        pairs = pairs.sort_values("score")
    unopened_ids = [str(row.compatibility_id) for row in pairs.itertuples(index=False) if _pair_key_open(row.compatibility_id, state, model_data) is False]
    add_ids = set(_sample_front(unopened_ids, min(config.pair_expand_count, len(unopened_ids)), rng))
    keys = [key for key in model_data.variable_groups.get("select_pair", {}) if _key_tail(key) in add_ids]
    state.allowed_pair_keys.update(keys)
    _open_pair_candidates(state, instance, model_data, add_ids)


def _random_shake(
    state: Stage8HeuristicState,
    instance: Stage8Instance,
    model_data: Stage8ModelData,
    config: Stage8Config,
    rng: random.Random,
) -> None:
    all_routes = list(model_data.variable_groups.get("x", {}).keys())
    all_pairs = list(model_data.variable_groups.get("select_pair", {}).keys())
    all_candidates = list(model_data.variable_groups.get("select_candidate", {}).keys())
    state.allowed_route_keys.update(rng.sample(all_routes, min(len(all_routes), _expand_count(len(all_routes), config.route_expand_fraction, minimum=1))))
    state.allowed_pair_keys.update(rng.sample(all_pairs, min(len(all_pairs), _expand_count(len(all_pairs), config.pair_expand_fraction, minimum=1))))
    state.allowed_candidate_keys.update(
        rng.sample(all_candidates, min(len(all_candidates), _expand_count(len(all_candidates), config.candidate_expand_fraction, minimum=1)))
    )


def _route_scores(instance: Stage8Instance, model_data: Stage8ModelData) -> pd.DataFrame:
    route_table = instance.component_route_period_scenario_table.set_index(["scenario_id", "component_instance_id", "route_id", "period_id"])
    objective_by_key = {}
    name_to_index = {name: index for index, name in enumerate(model_data.variable_names)}
    for term in model_data.objective_terms.itertuples(index=False):
        objective_by_key[str(term.variable_name)] = float(term.economic_cost_rmb) + float(term.quality_penalty_equiv) + float(term.reliability_penalty_equiv)
    rows = []
    for key, index in model_data.variable_groups.get("x", {}).items():
        scenario_id, component_id, route_id, period_id = key.split("|")
        table_key = (scenario_id, component_id, route_id, period_id)
        row = route_table.loc[table_key] if table_key in route_table.index else pd.Series(dtype=object)
        variable_name = model_data.variable_names[index]
        carbon = _first_available(row, ["expected_total_carbon_with_risk_kg", "expected_route_carbon_kg_total", "expected_carbon_kg_process"])
        risk = objective_by_key.get(variable_name, float(model_data.objective[index]))
        rows.append(
            {
                "key": key,
                "scenario_id": scenario_id,
                "component_instance_id": component_id,
                "route_id": route_id,
                "period_id": period_id,
                "score": float(model_data.objective[index]),
                "carbon_score": float(carbon),
                "risk_score": float(risk),
            }
        )
    return pd.DataFrame(rows)


def _pair_scores(instance: Stage8Instance) -> pd.DataFrame:
    pairs = instance.assembly_pair_pool.copy()
    if pairs.empty:
        return pd.DataFrame()
    pairs["quality_score"] = (
        _numeric_series(pairs, "pair_dimension_error_mm", 0.0).abs()
        + _numeric_series(pairs, "pair_quality_loss", 0.0)
        + _numeric_series(pairs, "pairwise_life_gap_h", 0.0) / 10000.0
    )
    pairs["score"] = (
        pairs["quality_score"]
        + (1.0 - _numeric_series(pairs, "compatibility_score", 0.0))
        + _numeric_series(pairs, "soft_pair_flag", 0.0) * 10.0
    )
    return pairs


def _candidate_score(candidates: pd.DataFrame) -> pd.Series:
    return (
        (1.0 - _numeric_series(candidates, "candidate_quality_score", 1.0))
        + (1.0 - _numeric_series(candidates, "candidate_reliability_score", 1.0))
        + _numeric_series(candidates, "old_candidate_flag", 0.0) * 0.2
    )


def _open_pair_candidates(
    state: Stage8HeuristicState,
    instance: Stage8Instance,
    model_data: Stage8ModelData,
    compatibility_ids: Iterable[str],
) -> None:
    pairs = instance.assembly_pair_pool[instance.assembly_pair_pool["compatibility_id"].astype(str).isin(set(compatibility_ids))]
    candidate_ids = set(pairs.get("candidate_i_id", pd.Series(dtype=object)).astype(str).tolist())
    candidate_ids.update(pairs.get("candidate_j_id", pd.Series(dtype=object)).astype(str).tolist())
    state.allowed_candidate_keys.update(
        key for key in model_data.variable_groups.get("select_candidate", {}) if _key_tail(key) in candidate_ids
    )


def _cap_routes_per_component(state: Stage8HeuristicState, scores: pd.DataFrame, config: Stage8Config) -> None:
    max_count = max(1, int(config.max_routes_per_component))
    allowed = scores[scores["key"].isin(state.allowed_route_keys)].copy()
    keep = []
    for (_scenario_id, component_id), group in allowed.groupby(["scenario_id", "component_instance_id"], dropna=False):
        keep.extend(group.sort_values("score").head(max_count)["key"].tolist())
    state.allowed_route_keys = set(keep)


def _close_unallowed(upper: np.ndarray, group: Dict[str, int], allowed: set[str]) -> None:
    for key, index in group.items():
        if key not in allowed:
            upper[index] = 0.0


def _existing_keys(keys: Iterable[str], group: Dict[str, int]) -> List[str]:
    existing = set(group)
    return [key for key in keys if key in existing]


def _key_tail(key: str) -> str:
    return key.split("|")[-1]


def _pair_key_open(compatibility_id: object, state: Stage8HeuristicState, model_data: Stage8ModelData) -> bool:
    value = str(compatibility_id)
    return any(_key_tail(key) == value for key in state.allowed_pair_keys if key in model_data.variable_groups.get("select_pair", {}))


def _expand_count(total: int, fraction: float, minimum: int) -> int:
    return max(minimum, int(np.ceil(max(0, total) * float(fraction))))


def _sample_front(values: List[str], count: int, rng: random.Random) -> List[str]:
    if count >= len(values):
        return list(values)
    front_count = max(count, int(count * 0.7))
    front = values[: min(len(values), front_count * 3)]
    if len(front) <= count:
        return front
    return rng.sample(front, count)


def _first_available(row: pd.Series, columns: list[str]) -> float:
    for column in columns:
        if column in row.index:
            value = pd.to_numeric(row.get(column), errors="coerce")
            if pd.notna(value):
                return float(value)
    return 0.0


def _numeric_series(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)
