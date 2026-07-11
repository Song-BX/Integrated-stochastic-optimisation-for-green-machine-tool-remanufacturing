"""MILP matrix assembly for the Stage 5 risk-averse SAA model."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, lil_matrix, vstack

from stage4_stochastic.model import build_model_data as build_stage4_model_data

from .config import Stage5Config
from .structures import Stage5Instance, Stage5ModelData


def build_model_data(instance: Stage5Instance, config: Stage5Config) -> Stage5ModelData:
    """Build the Stage 5 MILP by extending the Stage 4 matrix with CVaR rows."""

    base = build_stage4_model_data(instance, config)
    scenario_probabilities = instance.scenario_sample.set_index("scenario_id")["saa_probability"].to_dict()
    route_objective_terms = _route_objective_terms_with_risk(instance, base.objective_terms)
    loss_terms, loss_coefficients = _scenario_loss_terms(base, route_objective_terms, scenario_probabilities)

    variable_names = list(base.variable_names)
    objective = np.asarray(base.objective, dtype=float).copy()
    integrality = np.asarray(base.integrality, dtype=int).copy()
    lower_bounds = np.asarray(base.lower_bounds, dtype=float).copy()
    upper_bounds = np.asarray(base.upper_bounds, dtype=float).copy()
    variable_groups = dict(base.variable_groups)

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
    extra_cols = len(variable_names) - len(base.variable_names)
    extended_base = hstack([base.constraint_matrix, csr_matrix((old_rows, extra_cols), dtype=float)], format="csr")
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

    matrix = vstack([extended_base, cvar_rows.tocsr()], format="csr")
    constraint_lhs = np.concatenate([np.asarray(base.constraint_lhs, dtype=float), np.asarray(cvar_lhs, dtype=float)])
    constraint_rhs = np.concatenate([np.asarray(base.constraint_rhs, dtype=float), np.asarray(cvar_rhs, dtype=float)])
    constraint_names = list(base.constraint_names) + cvar_names
    variable_groups["eta"] = {"cvar95": eta_index}
    variable_groups["tail_excess"] = {scenario_id: index for scenario_id, index in tail_excess_index.items()}

    return Stage5ModelData(
        variable_names=variable_names,
        objective=objective,
        integrality=integrality,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        constraint_matrix=matrix,
        constraint_lhs=constraint_lhs,
        constraint_rhs=constraint_rhs,
        constraint_names=constraint_names,
        variable_groups=variable_groups,
        objective_terms=route_objective_terms,
        scenario_loss_terms=loss_terms,
    )


def _route_objective_terms_with_risk(instance: Stage5Instance, objective_terms: pd.DataFrame) -> pd.DataFrame:
    if objective_terms.empty:
        return objective_terms.copy()
    route_table = instance.component_route_period_scenario_table[
        [
            "scenario_id",
            "component_instance_id",
            "route_id",
            "period_id",
            "survival_probability_at_min_system_life",
            "route_tail_loss_rmb",
            "route_cvar95_loss_rmb",
            "route_risk_index",
        ]
    ].copy()
    data = objective_terms.merge(
        route_table,
        on=["scenario_id", "component_instance_id", "route_id", "period_id"],
        how="left",
    )
    for column in ["route_tail_loss_rmb", "route_cvar95_loss_rmb", "route_risk_index"]:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.0)
    base_loss_cols = [
        "economic_cost_rmb",
        "environmental_cost_equiv",
        "quality_penalty_equiv",
        "reliability_penalty_equiv",
        "warranty_risk_cost_rmb",
    ]
    data["route_expected_loss_unweighted"] = data[base_loss_cols].sum(axis=1)
    data["route_total_loss_unweighted"] = data["route_expected_loss_unweighted"] + data["route_tail_loss_rmb"]
    return data


def _scenario_loss_terms(
    model_data: object,
    route_objective_terms: pd.DataFrame,
    scenario_probabilities: Dict[str, float],
) -> tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    rows: List[Dict[str, object]] = []
    coefficients: Dict[str, Dict[str, float]] = {scenario_id: {} for scenario_id in scenario_probabilities}

    if not route_objective_terms.empty:
        for term in route_objective_terms.itertuples(index=False):
            variable_name = str(term.variable_name)
            scenario_id = str(term.scenario_id)
            coefficient = float(term.route_total_loss_unweighted)
            coefficients.setdefault(scenario_id, {})[variable_name] = coefficients.setdefault(scenario_id, {}).get(variable_name, 0.0) + coefficient
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "variable_name": variable_name,
                    "loss_source": "component_route_with_tail_risk",
                    "loss_coefficient_unweighted": coefficient,
                    "base_route_loss_unweighted": float(term.route_expected_loss_unweighted),
                    "route_tail_loss_rmb": float(term.route_tail_loss_rmb),
                }
            )

    name_by_index = {index: name for index, name in enumerate(model_data.variable_names)}
    objective = np.asarray(model_data.objective, dtype=float)
    for group_name in ["recourse_procure", "inventory", "backlog", "overtime"]:
        for key, index in model_data.variable_groups[group_name].items():
            scenario_id = key.split("|")[0]
            probability = float(scenario_probabilities[scenario_id])
            coefficient = float(objective[index]) / probability if probability > 1e-12 else 0.0
            if coefficient == 0.0:
                continue
            variable_name = name_by_index[index]
            coefficients.setdefault(scenario_id, {})[variable_name] = coefficients.setdefault(scenario_id, {}).get(variable_name, 0.0) + coefficient
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "variable_name": variable_name,
                    "loss_source": group_name,
                    "loss_coefficient_unweighted": coefficient,
                    "base_route_loss_unweighted": 0.0,
                    "route_tail_loss_rmb": 0.0,
                }
            )

    return pd.DataFrame(rows), coefficients

