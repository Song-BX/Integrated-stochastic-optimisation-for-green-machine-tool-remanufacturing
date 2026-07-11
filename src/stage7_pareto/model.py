"""Objective-vector construction for Stage 7 Pareto analysis."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from stage6_selective_assembly.model import build_model_data as build_stage6_model_data

from .config import Stage7Config
from .structures import Stage7Instance, Stage7ModelData


def build_model_data(instance: Stage7Instance, config: Stage7Config, tables: Dict[str, pd.DataFrame]) -> Stage7ModelData:
    """Build the Stage 6 matrix and attach Stage 7 objective vectors."""

    base = build_stage6_model_data(instance, config)
    vectors = _objective_vectors(instance, base, tables)
    summary = _objective_vector_summary(vectors)
    return Stage7ModelData(
        variable_names=base.variable_names,
        objective=base.objective,
        integrality=base.integrality,
        lower_bounds=base.lower_bounds,
        upper_bounds=base.upper_bounds,
        constraint_matrix=base.constraint_matrix,
        constraint_lhs=base.constraint_lhs,
        constraint_rhs=base.constraint_rhs,
        constraint_names=base.constraint_names,
        variable_groups=base.variable_groups,
        objective_terms=base.objective_terms,
        scenario_loss_terms=base.scenario_loss_terms,
        assembly_loss_terms=base.assembly_loss_terms,
        objective_vectors=vectors,
        objective_vector_summary=summary,
    )


def _objective_vectors(instance: Stage7Instance, model_data: object, tables: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
    n = len(model_data.variable_names)
    economic_risk = np.asarray(model_data.objective, dtype=float).copy()
    environmental = np.zeros(n, dtype=float)
    assembly_quality = np.zeros(n, dtype=float)
    _fill_route_environmental_vector(instance, model_data, environmental)
    _fill_procurement_environmental_vector(instance, model_data, tables, environmental)
    _fill_assembly_vectors(instance, model_data, environmental, assembly_quality)
    return {
        "economic_risk": economic_risk,
        "environmental_impact": environmental,
        "assembly_quality_loss": assembly_quality,
    }


def _fill_route_environmental_vector(instance: Stage7Instance, model_data: object, vector: np.ndarray) -> None:
    if model_data.objective_terms.empty:
        return
    route_table = instance.component_route_period_scenario_table.set_index(
        ["scenario_id", "component_instance_id", "route_id", "period_id"]
    )
    for term in model_data.objective_terms.itertuples(index=False):
        key = (term.scenario_id, term.component_instance_id, term.route_id, term.period_id)
        if key not in route_table.index:
            continue
        row = route_table.loc[key]
        coefficient = _first_available(
            row,
            [
                "expected_total_carbon_with_risk_kg",
                "expected_route_carbon_kg_total",
                "expected_carbon_kg_process",
                "expected_carbon_kg",
                "environmental_score",
            ],
        )
        variable_index = model_data.variable_names.index(term.variable_name)
        vector[variable_index] += float(term.saa_probability) * float(coefficient)


def _fill_procurement_environmental_vector(
    instance: Stage7Instance,
    model_data: object,
    tables: Dict[str, pd.DataFrame],
    vector: np.ndarray,
) -> None:
    procurement = tables["procurement_parameters"].copy()
    procurement = procurement[procurement["machine_type_id"] == instance.machine_type_id]
    procurement["embedded_carbon_kg_per_unit"] = pd.to_numeric(
        procurement.get("embedded_carbon_kg_per_unit", 0.0),
        errors="coerce",
    ).fillna(0.0)
    carbon = procurement.groupby("component_type")["embedded_carbon_kg_per_unit"].mean().to_dict()
    probabilities = instance.scenario_sample.set_index("scenario_id")["saa_probability"].to_dict()
    for key, index in model_data.variable_groups["pre_procure"].items():
        component_type, _period = key.split("|")
        vector[index] += float(carbon.get(component_type, 0.0))
    for key, index in model_data.variable_groups["recourse_procure"].items():
        scenario_id, component_type, _period = key.split("|")
        vector[index] += float(probabilities[scenario_id]) * float(carbon.get(component_type, 0.0))


def _fill_assembly_vectors(instance: Stage7Instance, model_data: object, environmental: np.ndarray, assembly_quality: np.ndarray) -> None:
    if model_data.assembly_loss_terms.empty:
        pair_terms = assembly_pair_environmental_terms(instance, model_data)
    else:
        pair_terms = assembly_pair_environmental_terms(instance, model_data)
    name_to_index = {name: index for index, name in enumerate(model_data.variable_names)}
    for term in model_data.assembly_loss_terms.itertuples(index=False):
        index = name_to_index.get(str(term.variable_name))
        if index is None:
            continue
        assembly_quality[index] += float(term.loss_coefficient_unweighted)
    for term in pair_terms.itertuples(index=False):
        index = int(term.variable_index)
        environmental[index] += float(term.weighted_coefficient)


def assembly_pair_environmental_terms(instance: Stage7Instance, model_data: object) -> pd.DataFrame:
    """Return selected-pair carbon coefficients aligned to Stage 6/7 select_pair variables."""

    select_pair = model_data.variable_groups.get("select_pair", {})
    if not select_pair or instance.assembly_pair_pool.empty:
        return pd.DataFrame(
            columns=[
                "scenario_id",
                "assembly_requirement_id",
                "compatibility_id",
                "variable_name",
                "variable_index",
                "saa_probability",
                "pair_carbon_kg",
                "carbon_source_column",
                "weighted_coefficient",
            ]
        )
    probabilities = instance.scenario_sample.set_index("scenario_id")["saa_probability"].to_dict()
    pair_lookup = instance.assembly_pair_pool.set_index("compatibility_id", drop=False)
    rows = []
    for key, index in select_pair.items():
        parts = key.split("|")
        if len(parts) != 3:
            continue
        scenario_id, requirement_id, compatibility_id = parts
        if compatibility_id not in pair_lookup.index:
            continue
        pair = pair_lookup.loc[compatibility_id]
        if isinstance(pair, pd.DataFrame):
            pair = pair.iloc[0]
        carbon, source_column = _pair_carbon_value(pair)
        probability = float(probabilities.get(scenario_id, 0.0))
        rows.append(
            {
                "scenario_id": scenario_id,
                "assembly_requirement_id": requirement_id,
                "compatibility_id": compatibility_id,
                "variable_name": model_data.variable_names[index],
                "variable_index": int(index),
                "saa_probability": probability,
                "pair_carbon_kg": float(carbon),
                "carbon_source_column": source_column,
                "weighted_coefficient": probability * float(carbon),
            }
        )
    return pd.DataFrame(rows)


def _objective_vector_summary(vectors: Dict[str, np.ndarray]) -> Dict[str, object]:
    rows = {}
    for name, vector in vectors.items():
        nonzero = np.abs(vector) > 1e-12
        rows[name] = {
            "length": int(len(vector)),
            "nonzero_count": int(nonzero.sum()),
            "min_coefficient": float(vector.min()) if len(vector) else 0.0,
            "max_coefficient": float(vector.max()) if len(vector) else 0.0,
            "finite": bool(np.isfinite(vector).all()),
        }
    return rows


def objective_value(model_data: Stage7ModelData, vector_name: str, x: np.ndarray) -> float:
    return float(np.dot(model_data.objective_vectors[vector_name], x))


def objective_values(model_data: Stage7ModelData, x: np.ndarray) -> Dict[str, float]:
    return {name: objective_value(model_data, name, x) for name in model_data.objective_vectors}


def _first_available(row: pd.Series, columns: list[str]) -> float:
    for column in columns:
        if column in row.index:
            value = pd.to_numeric(row.get(column), errors="coerce")
            if pd.notna(value):
                return float(value)
    return 0.0


def _pair_carbon_value(row: pd.Series) -> tuple[float, str]:
    for column in ["combined_carbon_kg", "pair_carbon_kg", "environmental_score"]:
        if column in row.index:
            value = pd.to_numeric(row.get(column), errors="coerce")
            if pd.notna(value):
                return float(value), column
    return 0.0, "none"
