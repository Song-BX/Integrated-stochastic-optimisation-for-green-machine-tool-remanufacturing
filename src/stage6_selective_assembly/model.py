"""MILP matrix assembly for the Stage 6 selective-assembly model."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, lil_matrix, vstack

from stage4_stochastic.model import build_model_data as build_stage4_model_data
from stage5_risk_averse.model import _route_objective_terms_with_risk, _scenario_loss_terms

from .config import Stage6Config
from .structures import Stage6Instance, Stage6ModelData


def build_model_data(instance: Stage6Instance, config: Stage6Config) -> Stage6ModelData:
    """Build Stage 6 by extending the Stage 4 matrix before adding CVaR rows."""

    base = build_stage4_model_data(instance, config)
    scenario_probabilities = instance.scenario_sample.set_index("scenario_id")["saa_probability"].to_dict()
    route_objective_terms = _route_objective_terms_with_risk(instance, base.objective_terms)
    variable_names = list(base.variable_names)
    objective = np.asarray(base.objective, dtype=float).copy()
    integrality = np.asarray(base.integrality, dtype=int).copy()
    lower_bounds = np.asarray(base.lower_bounds, dtype=float).copy()
    upper_bounds = np.asarray(base.upper_bounds, dtype=float).copy()
    variable_groups = {name: dict(group) for name, group in base.variable_groups.items()}

    assembly_vars = _append_assembly_variables(instance, config, scenario_probabilities, variable_names, objective, integrality, lower_bounds, upper_bounds)
    objective = assembly_vars["objective"]
    integrality = assembly_vars["integrality"]
    lower_bounds = assembly_vars["lower_bounds"]
    upper_bounds = assembly_vars["upper_bounds"]
    variable_groups.update(assembly_vars["variable_groups"])

    extended_base = hstack(
        [
            base.constraint_matrix,
            csr_matrix((base.constraint_matrix.shape[0], len(variable_names) - len(base.variable_names)), dtype=float),
        ],
        format="csr",
    )
    assembly_matrix, assembly_lhs, assembly_rhs, assembly_names = _assembly_constraints(instance, base, variable_groups, len(variable_names))

    matrix_without_cvar = vstack([extended_base, assembly_matrix], format="csr")
    lhs_without_cvar = np.concatenate([np.asarray(base.constraint_lhs, dtype=float), np.asarray(assembly_lhs, dtype=float)])
    rhs_without_cvar = np.concatenate([np.asarray(base.constraint_rhs, dtype=float), np.asarray(assembly_rhs, dtype=float)])
    names_without_cvar = list(base.constraint_names) + assembly_names

    interim = Stage6ModelData(
        variable_names=variable_names,
        objective=objective,
        integrality=integrality,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        constraint_matrix=matrix_without_cvar,
        constraint_lhs=lhs_without_cvar,
        constraint_rhs=rhs_without_cvar,
        constraint_names=names_without_cvar,
        variable_groups=variable_groups,
        objective_terms=route_objective_terms,
        scenario_loss_terms=pd.DataFrame(),
        assembly_loss_terms=assembly_vars["assembly_loss_terms"],
    )
    loss_terms, loss_coefficients = _stage6_scenario_loss_terms(interim, route_objective_terms, scenario_probabilities)
    return _add_cvar_rows(instance, config, interim, loss_terms, loss_coefficients, scenario_probabilities)


def _append_assembly_variables(
    instance: Stage6Instance,
    config: Stage6Config,
    scenario_probabilities: Dict[str, float],
    variable_names: List[str],
    objective: np.ndarray,
    integrality: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> Dict[str, object]:
    select_candidate = {}
    select_pair = {}
    feature_assemble = {}
    assembly_shortfall = {}
    assembly_loss_rows: List[Dict[str, object]] = []
    objective_values = objective
    integrality_values = integrality
    lower_values = lower_bounds
    upper_values = upper_bounds

    for scenario_id in instance.scenario_ids:
        probability = float(scenario_probabilities[scenario_id])
        for row in instance.assembly_candidate_pool.itertuples(index=False):
            key = f"{scenario_id}|{row.assembly_requirement_id}|{row.assembly_candidate_id}"
            select_candidate[key] = len(variable_names)
            variable_names.append(f"select_candidate[{scenario_id},{row.assembly_requirement_id},{row.assembly_candidate_id}]")
            objective_values = np.append(objective_values, 0.0)
            integrality_values = np.append(integrality_values, 1)
            lower_values = np.append(lower_values, 0.0)
            upper_values = np.append(upper_values, 1.0)

        for row in instance.assembly_pair_pool.itertuples(index=False):
            key = f"{scenario_id}|{row.assembly_requirement_id}|{row.compatibility_id}"
            select_pair[key] = len(variable_names)
            variable_name = f"select_pair[{scenario_id},{row.assembly_requirement_id},{row.compatibility_id}]"
            variable_names.append(variable_name)
            unweighted_loss = _pair_loss(row, config)
            objective_values = np.append(objective_values, probability * unweighted_loss)
            integrality_values = np.append(integrality_values, 1)
            lower_values = np.append(lower_values, 0.0)
            upper_values = np.append(upper_values, 1.0)
            assembly_loss_rows.append(
                {
                    "scenario_id": scenario_id,
                    "variable_name": variable_name,
                    "assembly_requirement_id": row.assembly_requirement_id,
                    "compatibility_id": row.compatibility_id,
                    "loss_source": "assembly_pair",
                    "loss_coefficient_unweighted": unweighted_loss,
                    "dimension_penalty_unweighted": _dimension_penalty(row, config),
                    "quality_loss_penalty_unweighted": _quality_penalty(row, config),
                    "life_gap_penalty_unweighted": _life_gap_penalty(row, config),
                    "compatibility_penalty_unweighted": _compatibility_penalty(row, config),
                    "soft_pair_penalty_unweighted": _soft_pair_penalty(row, config),
                    "assembly_tail_risk_loss_rmb": _assembly_tail_risk(row, config),
                }
            )

        for row in instance.assembly_requirements.itertuples(index=False):
            requirement_id = row.assembly_requirement_id
            feature_key = f"{scenario_id}|{requirement_id}"
            feature_assemble[feature_key] = len(variable_names)
            variable_names.append(f"feature_assemble[{scenario_id},{requirement_id}]")
            objective_values = np.append(objective_values, 0.0)
            integrality_values = np.append(integrality_values, 1)
            lower_values = np.append(lower_values, 0.0)
            upper_values = np.append(upper_values, np.inf)

            assembly_shortfall[feature_key] = len(variable_names)
            variable_name = f"assembly_shortfall[{scenario_id},{requirement_id}]"
            variable_names.append(variable_name)
            objective_values = np.append(objective_values, probability * float(config.assembly_shortfall_penalty_rmb))
            integrality_values = np.append(integrality_values, 1)
            lower_values = np.append(lower_values, 0.0)
            upper_values = np.append(upper_values, np.inf)
            assembly_loss_rows.append(
                {
                    "scenario_id": scenario_id,
                    "variable_name": variable_name,
                    "assembly_requirement_id": requirement_id,
                    "compatibility_id": None,
                    "loss_source": "assembly_shortfall",
                    "loss_coefficient_unweighted": float(config.assembly_shortfall_penalty_rmb),
                    "dimension_penalty_unweighted": 0.0,
                    "quality_loss_penalty_unweighted": 0.0,
                    "life_gap_penalty_unweighted": 0.0,
                    "compatibility_penalty_unweighted": 0.0,
                    "soft_pair_penalty_unweighted": 0.0,
                    "assembly_tail_risk_loss_rmb": 0.0,
                }
            )

    return {
        "objective": objective_values,
        "integrality": integrality_values,
        "lower_bounds": lower_values,
        "upper_bounds": upper_values,
        "variable_groups": {
            "select_candidate": select_candidate,
            "select_pair": select_pair,
            "feature_assemble": feature_assemble,
            "assembly_shortfall": assembly_shortfall,
        },
        "assembly_loss_terms": pd.DataFrame(assembly_loss_rows),
    }


def _assembly_constraints(
    instance: Stage6Instance,
    base_model: object,
    variable_groups: Dict[str, Dict[str, int]],
    n_variables: int,
) -> tuple[csr_matrix, np.ndarray, np.ndarray, List[str]]:
    rows: List[Dict[int, float]] = []
    lhs: List[float] = []
    rhs: List[float] = []
    names: List[str] = []
    candidate_by_id = instance.assembly_candidate_pool.set_index("assembly_candidate_id").to_dict(orient="index")
    pair_by_requirement = {
        requirement_id: group.copy()
        for requirement_id, group in instance.assembly_pair_pool.groupby("assembly_requirement_id")
    }
    requirement_divisor = instance.assembly_requirements.set_index("assembly_requirement_id")["pair_count_divisor"].to_dict()
    primary_component = instance.assembly_requirements.set_index("assembly_requirement_id")["primary_selective_component_type"].to_dict()

    for scenario_id in instance.scenario_ids:
        for pair in instance.assembly_pair_pool.itertuples(index=False):
            pair_key = f"{scenario_id}|{pair.assembly_requirement_id}|{pair.compatibility_id}"
            pair_index = variable_groups["select_pair"][pair_key]
            for side, candidate_id in [("i", pair.candidate_i_id), ("j", pair.candidate_j_id)]:
                candidate_key = f"{scenario_id}|{pair.assembly_requirement_id}|{candidate_id}"
                rows.append({pair_index: 1.0, variable_groups["select_candidate"][candidate_key]: -1.0})
                lhs.append(-np.inf)
                rhs.append(0.0)
                names.append(f"pair_requires_candidate_{side}[{scenario_id},{pair.compatibility_id}]")

        for requirement in instance.assembly_requirements.itertuples(index=False):
            requirement_id = requirement.assembly_requirement_id
            feature_key = f"{scenario_id}|{requirement_id}"
            coverage_row = {
                variable_groups["feature_assemble"][feature_key]: 1.0,
                variable_groups["assembly_shortfall"][feature_key]: 1.0,
            }
            for period in instance.periods:
                coverage_row[base_model.variable_groups["assemble"][f"{scenario_id}|{period}"]] = -1.0
            rows.append(coverage_row)
            lhs.append(0.0)
            rhs.append(0.0)
            names.append(f"assembly_feature_coverage[{scenario_id},{requirement_id}]")

            pair_rows = pair_by_requirement.get(requirement_id, pd.DataFrame())
            divisor = max(1.0, float(requirement_divisor.get(requirement_id, 1.0)))
            if pair_rows.empty:
                rows.append({variable_groups["feature_assemble"][feature_key]: 1.0})
            else:
                row = {variable_groups["feature_assemble"][feature_key]: divisor}
                for pair in pair_rows.itertuples(index=False):
                    row[variable_groups["select_pair"][f"{scenario_id}|{requirement_id}|{pair.compatibility_id}"]] = -1.0
                rows.append(row)
            lhs.append(-np.inf)
            rhs.append(0.0)
            names.append(f"feature_requires_star_pairs[{scenario_id},{requirement_id}]")

        for requirement_id, group in instance.assembly_candidate_pool.groupby("assembly_requirement_id", dropna=False):
            pair_rows = pair_by_requirement.get(requirement_id, pd.DataFrame())
            if pair_rows.empty:
                continue
            divisor = max(1.0, float(requirement_divisor.get(requirement_id, 1.0)))
            primary_type = primary_component.get(requirement_id)
            for candidate in group.itertuples(index=False):
                incident = pair_rows[
                    (pair_rows["candidate_i_id"] == candidate.assembly_candidate_id)
                    | (pair_rows["candidate_j_id"] == candidate.assembly_candidate_id)
                ]
                if incident.empty:
                    continue
                candidate_key = f"{scenario_id}|{requirement_id}|{candidate.assembly_candidate_id}"
                multiplier = divisor if str(candidate.component_type) == str(primary_type) and divisor > 1.0 else 1.0
                row = {variable_groups["select_candidate"][candidate_key]: -multiplier}
                for pair in incident.itertuples(index=False):
                    pair_key = f"{scenario_id}|{requirement_id}|{pair.compatibility_id}"
                    row[variable_groups["select_pair"][pair_key]] = 1.0
                rows.append(row)
                lhs.append(-np.inf)
                rhs.append(0.0)
                names.append(f"candidate_pair_incidence[{scenario_id},{requirement_id},{candidate.assembly_candidate_id}]")

        for (requirement_id, candidate_id), group in instance.assembly_candidate_pool.groupby(
            ["assembly_requirement_id", "assembly_candidate_id"], dropna=False
        ):
            key = f"{scenario_id}|{requirement_id}|{candidate_id}"
            row = {variable_groups["select_candidate"][key]: 1.0}
            candidate = candidate_by_id[candidate_id]
            if int(candidate.get("old_candidate_flag", 0)) == 1:
                component_id = str(candidate.get("component_instance_id"))
                route_id = str(candidate.get("planned_route_id"))
                linked_x = _route_indices_for_candidate(base_model.variable_groups["x"], scenario_id, component_id, route_id)
                for index in linked_x:
                    row[index] = row.get(index, 0.0) - 1.0
            rows.append(row)
            lhs.append(-np.inf)
            rhs.append(0.0 if int(candidate.get("old_candidate_flag", 0)) == 1 else 1.0)
            names.append(f"old_candidate_route_coupling[{scenario_id},{requirement_id},{candidate_id}]")

        for component_type, group in instance.assembly_candidate_pool[
            instance.assembly_candidate_pool["new_backup_candidate_flag"] == 1
        ].groupby("component_type", dropna=False):
            row = {}
            for candidate in group.itertuples(index=False):
                key = f"{scenario_id}|{candidate.assembly_requirement_id}|{candidate.assembly_candidate_id}"
                row[variable_groups["select_candidate"][key]] = row.get(variable_groups["select_candidate"][key], 0.0) + 1.0
            for period in instance.periods:
                pre_key = f"{component_type}|{period}"
                recourse_key = f"{scenario_id}|{component_type}|{period}"
                if pre_key in base_model.variable_groups["pre_procure"]:
                    row[base_model.variable_groups["pre_procure"][pre_key]] = row.get(base_model.variable_groups["pre_procure"][pre_key], 0.0) - 1.0
                if recourse_key in base_model.variable_groups["recourse_procure"]:
                    row[base_model.variable_groups["recourse_procure"][recourse_key]] = row.get(base_model.variable_groups["recourse_procure"][recourse_key], 0.0) - 1.0
            initial_available = _initial_available(instance, str(component_type))
            rows.append(row)
            lhs.append(-np.inf)
            rhs.append(initial_available)
            names.append(f"new_candidate_availability[{scenario_id},{component_type}]")

    matrix = lil_matrix((len(rows), n_variables), dtype=float)
    for row_number, entries in enumerate(rows):
        for col_number, value in entries.items():
            matrix[row_number, col_number] = float(value)
    return matrix.tocsr(), np.array(lhs, dtype=float), np.array(rhs, dtype=float), names


def _stage6_scenario_loss_terms(
    model_data: Stage6ModelData,
    route_objective_terms: pd.DataFrame,
    scenario_probabilities: Dict[str, float],
) -> tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    base_terms, coefficients = _scenario_loss_terms(model_data, route_objective_terms, scenario_probabilities)
    rows = [base_terms] if not base_terms.empty else []
    if not model_data.assembly_loss_terms.empty:
        assembly_terms = model_data.assembly_loss_terms.copy()
        for term in assembly_terms.itertuples(index=False):
            scenario_id = str(term.scenario_id)
            variable_name = str(term.variable_name)
            coefficient = float(term.loss_coefficient_unweighted)
            coefficients.setdefault(scenario_id, {})[variable_name] = coefficients.setdefault(scenario_id, {}).get(variable_name, 0.0) + coefficient
        rows.append(assembly_terms)
    loss_terms = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    return loss_terms, coefficients


def _add_cvar_rows(
    instance: Stage6Instance,
    config: Stage6Config,
    base: Stage6ModelData,
    loss_terms: pd.DataFrame,
    loss_coefficients: Dict[str, Dict[str, float]],
    scenario_probabilities: Dict[str, float],
) -> Stage6ModelData:
    variable_names = list(base.variable_names)
    objective = np.asarray(base.objective, dtype=float).copy()
    integrality = np.asarray(base.integrality, dtype=int).copy()
    lower_bounds = np.asarray(base.lower_bounds, dtype=float).copy()
    upper_bounds = np.asarray(base.upper_bounds, dtype=float).copy()
    variable_groups = {name: dict(group) for name, group in base.variable_groups.items()}

    eta_index = len(variable_names)
    variable_names.append("eta[cvar95]")
    objective = np.append(objective, float(config.cvar_lambda))
    integrality = np.append(integrality, 0)
    lower_bounds = np.append(lower_bounds, 0.0)
    upper_bounds = np.append(upper_bounds, np.inf)

    tail_excess_index: Dict[str, int] = {}
    for scenario_id in instance.scenario_ids:
        tail_excess_index[scenario_id] = len(variable_names)
        variable_names.append(f"tail_excess[{scenario_id}]")
        probability = float(scenario_probabilities[scenario_id])
        coefficient = float(config.cvar_lambda) * probability / max(1e-9, 1.0 - float(config.cvar_confidence))
        objective = np.append(objective, coefficient)
        integrality = np.append(integrality, 0)
        lower_bounds = np.append(lower_bounds, 0.0)
        upper_bounds = np.append(upper_bounds, np.inf)

    old_rows = base.constraint_matrix.shape[0]
    extended_base = hstack([base.constraint_matrix, csr_matrix((old_rows, len(variable_names) - len(base.variable_names)), dtype=float)], format="csr")
    cvar_rows = lil_matrix((len(instance.scenario_ids), len(variable_names)), dtype=float)
    cvar_lhs: List[float] = []
    cvar_rhs: List[float] = []
    cvar_names: List[str] = []
    name_to_index = {name: index for index, name in enumerate(variable_names)}
    for row_number, scenario_id in enumerate(instance.scenario_ids):
        for variable_name, coefficient in loss_coefficients.get(scenario_id, {}).items():
            cvar_rows[row_number, name_to_index[variable_name]] = float(coefficient)
        cvar_rows[row_number, eta_index] = -1.0
        cvar_rows[row_number, tail_excess_index[scenario_id]] = -1.0
        cvar_lhs.append(-np.inf)
        cvar_rhs.append(0.0)
        cvar_names.append(f"cvar_tail_excess[{scenario_id}]")

    variable_groups["eta"] = {"cvar95": eta_index}
    variable_groups["tail_excess"] = {scenario_id: index for scenario_id, index in tail_excess_index.items()}
    return Stage6ModelData(
        variable_names=variable_names,
        objective=objective,
        integrality=integrality,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        constraint_matrix=vstack([extended_base, cvar_rows.tocsr()], format="csr"),
        constraint_lhs=np.concatenate([np.asarray(base.constraint_lhs, dtype=float), np.asarray(cvar_lhs, dtype=float)]),
        constraint_rhs=np.concatenate([np.asarray(base.constraint_rhs, dtype=float), np.asarray(cvar_rhs, dtype=float)]),
        constraint_names=list(base.constraint_names) + cvar_names,
        variable_groups=variable_groups,
        objective_terms=base.objective_terms,
        scenario_loss_terms=loss_terms,
        assembly_loss_terms=base.assembly_loss_terms,
    )


def _route_indices_for_candidate(x_group: Dict[str, int], scenario_id: str, component_id: str, route_id: str) -> List[int]:
    prefix = f"{scenario_id}|{component_id}|{route_id}|"
    return [index for key, index in x_group.items() if key.startswith(prefix)]


def _initial_available(instance: Stage6Instance, component_type: str) -> float:
    data = instance.initial_inventory[instance.initial_inventory["component_type"] == component_type]
    if data.empty:
        return 0.0
    return float(pd.to_numeric(data.iloc[0].get("initial_quantity_available"), errors="coerce"))


def _pair_loss(row: object, config: Stage6Config) -> float:
    return (
        _dimension_penalty(row, config)
        + _quality_penalty(row, config)
        + _life_gap_penalty(row, config)
        + _compatibility_penalty(row, config)
        + _soft_pair_penalty(row, config)
        + _assembly_tail_risk(row, config)
    )


def _dimension_penalty(row: object, config: Stage6Config) -> float:
    max_error = max(abs(float(getattr(row, "max_dimension_chain_error_mm", 0.0))), 1e-9)
    ratio = abs(float(getattr(row, "pair_dimension_error_mm", 0.0))) / max_error
    return float(config.dimension_penalty_weight) * ratio


def _quality_penalty(row: object, config: Stage6Config) -> float:
    return float(config.assembly_quality_loss_weight) * max(0.0, float(getattr(row, "pair_quality_loss", 0.0)))


def _life_gap_penalty(row: object, config: Stage6Config) -> float:
    return float(config.life_gap_penalty_weight) * max(0.0, float(getattr(row, "pairwise_life_gap_h", 0.0)))


def _compatibility_penalty(row: object, config: Stage6Config) -> float:
    return float(config.compatibility_penalty_weight) * max(0.0, 1.0 - float(getattr(row, "compatibility_score", 0.0)))


def _soft_pair_penalty(row: object, config: Stage6Config) -> float:
    return float(config.soft_pair_penalty_rmb) if int(getattr(row, "soft_pair_flag", 0)) == 1 else 0.0


def _assembly_tail_risk(row: object, config: Stage6Config) -> float:
    risk_index = max(0.0, float(getattr(row, "pair_cvar_tail_risk_index", 0.0)))
    pair_weight = max(0.0, float(getattr(row, "cvar_risk_weight", 1.0)))
    return risk_index * pair_weight * float(config.assembly_risk_weight) * float(config.risk_budget_reference_rmb)
