"""Model-data construction and variable-pool utilities for Stage 8."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from stage7_pareto.model import build_model_data as build_stage7_model_data

from .aggregation import stage7_config
from .config import Stage8Config
from .structures import Stage8Instance, Stage8ModelData


def build_model_data(instance: Stage8Instance, config: Stage8Config, tables: Dict[str, pd.DataFrame]) -> Stage8ModelData:
    """Build the Stage 7 model matrix and attach Stage 8 restriction metadata."""

    base = build_stage7_model_data(instance, stage7_config(config), tables)
    return Stage8ModelData(
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
        objective_vectors=base.objective_vectors,
        objective_vector_summary=base.objective_vector_summary,
        restriction_summary=_restriction_summary(base.variable_groups),
    )


def _restriction_summary(variable_groups: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    return {
        "route_variable_count": len(variable_groups.get("x", {})),
        "candidate_variable_count": len(variable_groups.get("select_candidate", {})),
        "pair_variable_count": len(variable_groups.get("select_pair", {})),
        "unrestricted_variable_count": sum(
            len(group)
            for name, group in variable_groups.items()
            if name not in {"x", "select_candidate", "select_pair"}
        ),
    }
