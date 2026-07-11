"""Declarative Stage 1 schema and validation rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


EXPECTED_FILES: List[str] = [
    "assembly_candidates.csv",
    "assembly_compatibility.csv",
    "assembly_requirements.csv",
    "baseline_rules.csv",
    "bom.csv",
    "capacity_calendar.csv",
    "component_inspection.csv",
    "component_quality_scenarios.csv",
    "demand_scenarios.csv",
    "environmental_parameters.csv",
    "historical_performance.csv",
    "initial_inventory.csv",
    "machine_types.csv",
    "machines.csv",
    "orders.csv",
    "processing_parameters.csv",
    "procurement_parameters.csv",
    "quality_states.csv",
    "reliability_parameters.csv",
    "returned_cores.csv",
    "risk_parameters.csv",
    "route_feasibility.csv",
    "route_operations.csv",
    "route_outcome_scenarios.csv",
    "route_state_transition.csv",
    "routes.csv",
    "scenarios.csv",
    "time_periods.csv",
]


PRIMARY_KEYS: Dict[str, Tuple[str, ...]] = {
    "assembly_candidates.csv": ("assembly_candidate_id",),
    "assembly_compatibility.csv": ("compatibility_id",),
    "assembly_requirements.csv": ("assembly_requirement_id",),
    "baseline_rules.csv": ("baseline_rule_id",),
    "bom.csv": ("machine_type_id", "bom_item_id"),
    "capacity_calendar.csv": ("period_id", "machine_id"),
    "component_inspection.csv": ("component_instance_id",),
    "component_quality_scenarios.csv": ("scenario_id", "machine_type_id", "component_type"),
    "demand_scenarios.csv": ("demand_scenario_id",),
    "environmental_parameters.csv": ("environmental_param_id",),
    "historical_performance.csv": ("performance_record_id",),
    "initial_inventory.csv": ("inventory_item_id",),
    "machine_types.csv": ("machine_type_id",),
    "machines.csv": ("machine_id",),
    "orders.csv": ("order_id",),
    "processing_parameters.csv": ("component_type", "quality_state", "route_id", "operation_id"),
    "procurement_parameters.csv": ("procurement_param_id",),
    "quality_states.csv": ("quality_state",),
    "reliability_parameters.csv": ("reliability_param_id",),
    "returned_cores.csv": ("core_id",),
    "risk_parameters.csv": ("risk_param_id",),
    "route_feasibility.csv": ("component_type", "quality_state", "route_id"),
    "route_operations.csv": ("route_id", "operation_seq"),
    "route_outcome_scenarios.csv": ("route_outcome_scenario_id",),
    "route_state_transition.csv": ("component_type", "quality_state_before", "route_id", "scenario_label"),
    "routes.csv": ("route_id",),
    "scenarios.csv": ("scenario_id",),
    "time_periods.csv": ("period_id",),
}


EXPECTED_ROW_COUNTS: Dict[str, int] = {
    "time_periods.csv": 314,
    "machine_types.csv": 15,
    "bom.csv": 140,
    "returned_cores.csv": 1232,
    "component_inspection.csv": 8852,
    "quality_states.csv": 4,
    "routes.csv": 7,
    "route_feasibility.csv": 868,
    "route_operations.csv": 57,
    "processing_parameters.csv": 2908,
    "machines.csv": 47,
    "orders.csv": 607,
    "capacity_calendar.csv": 14758,
    "route_state_transition.csv": 1173,
    "assembly_requirements.csv": 84,
    "assembly_candidates.csv": 19876,
    "assembly_compatibility.csv": 14110,
    "procurement_parameters.csv": 474,
    "initial_inventory.csv": 263,
    "environmental_parameters.csv": 171,
    "reliability_parameters.csv": 1173,
    "risk_parameters.csv": 10557,
    "scenarios.csv": 27,
    "component_quality_scenarios.csv": 3564,
    "route_outcome_scenarios.csv": 40780,
    "demand_scenarios.csv": 127170,
    "baseline_rules.csv": 22,
    "historical_performance.csv": 23760,
}


EXPECTED_TOTAL_ROWS = 273013
EXPECTED_TOTAL_SIZE_MB = 353.13


@dataclass(frozen=True)
class ForeignKeyRule:
    name: str
    source_file: str
    source_columns: Tuple[str, ...]
    target_file: str
    target_columns: Tuple[str, ...]
    separator: str | None = None
    skip_if_column: str | None = None
    skip_if_values: Tuple[str, ...] = ()


FOREIGN_KEYS: List[ForeignKeyRule] = [
    ForeignKeyRule("bom.machine_type_id", "bom.csv", ("machine_type_id",), "machine_types.csv", ("machine_type_id",)),
    ForeignKeyRule("orders.machine_type_id", "orders.csv", ("machine_type_id",), "machine_types.csv", ("machine_type_id",)),
    ForeignKeyRule("returned_cores.machine_type_id", "returned_cores.csv", ("machine_type_id",), "machine_types.csv", ("machine_type_id",)),
    ForeignKeyRule("capacity_calendar.machine_id", "capacity_calendar.csv", ("machine_id",), "machines.csv", ("machine_id",)),
    ForeignKeyRule("capacity_calendar.period_id", "capacity_calendar.csv", ("period_id",), "time_periods.csv", ("period_id",)),
    ForeignKeyRule("component_inspection.core_id", "component_inspection.csv", ("core_id",), "returned_cores.csv", ("core_id",)),
    ForeignKeyRule("component_inspection.bom_item_id", "component_inspection.csv", ("machine_type_id", "bom_item_id"), "bom.csv", ("machine_type_id", "bom_item_id")),
    ForeignKeyRule("route_feasibility.route_id", "route_feasibility.csv", ("route_id",), "routes.csv", ("route_id",)),
    ForeignKeyRule("route_feasibility.quality_state", "route_feasibility.csv", ("quality_state",), "quality_states.csv", ("quality_state",)),
    ForeignKeyRule("route_operations.route_id", "route_operations.csv", ("route_id",), "routes.csv", ("route_id",)),
    ForeignKeyRule("processing_parameters.route_id", "processing_parameters.csv", ("route_id",), "routes.csv", ("route_id",)),
    ForeignKeyRule("processing_parameters.quality_state", "processing_parameters.csv", ("quality_state",), "quality_states.csv", ("quality_state",)),
    ForeignKeyRule("route_state_transition.route_id", "route_state_transition.csv", ("route_id",), "routes.csv", ("route_id",)),
    ForeignKeyRule("route_state_transition.quality_state_before", "route_state_transition.csv", ("quality_state_before",), "quality_states.csv", ("quality_state",)),
    ForeignKeyRule("assembly_requirements.machine_type_id", "assembly_requirements.csv", ("machine_type_id",), "machine_types.csv", ("machine_type_id",)),
    ForeignKeyRule("assembly_candidates.assembly_requirement_id", "assembly_candidates.csv", ("assembly_requirement_id",), "assembly_requirements.csv", ("assembly_requirement_id",)),
    ForeignKeyRule("assembly_candidates.machine_type_id", "assembly_candidates.csv", ("machine_type_id",), "machine_types.csv", ("machine_type_id",)),
    ForeignKeyRule(
        "assembly_candidates.component_instance_id",
        "assembly_candidates.csv",
        ("component_instance_id",),
        "component_inspection.csv",
        ("component_instance_id",),
        skip_if_column="candidate_source_type",
        skip_if_values=("new", "new_replacement"),
    ),
    ForeignKeyRule("assembly_compatibility.assembly_requirement_id", "assembly_compatibility.csv", ("assembly_requirement_id",), "assembly_requirements.csv", ("assembly_requirement_id",)),
    ForeignKeyRule("assembly_compatibility.candidate_i_id", "assembly_compatibility.csv", ("candidate_i_id",), "assembly_candidates.csv", ("assembly_candidate_id",)),
    ForeignKeyRule("assembly_compatibility.candidate_j_id", "assembly_compatibility.csv", ("candidate_j_id",), "assembly_candidates.csv", ("assembly_candidate_id",)),
    ForeignKeyRule("scenarios_to_component_quality", "component_quality_scenarios.csv", ("scenario_id",), "scenarios.csv", ("scenario_id",)),
    ForeignKeyRule("scenarios_to_route_outcome", "route_outcome_scenarios.csv", ("scenario_id",), "scenarios.csv", ("scenario_id",)),
    ForeignKeyRule("route_outcome.evaluated_route_id", "route_outcome_scenarios.csv", ("evaluated_route_id",), "routes.csv", ("route_id",)),
    ForeignKeyRule("scenarios_to_demand", "demand_scenarios.csv", ("scenario_id",), "scenarios.csv", ("scenario_id",)),
    ForeignKeyRule("demand.machine_type_id", "demand_scenarios.csv", ("machine_type_id",), "machine_types.csv", ("machine_type_id",)),
    ForeignKeyRule("demand.period_id", "demand_scenarios.csv", ("period_id",), "time_periods.csv", ("period_id",)),
    ForeignKeyRule("baseline_to_performance", "historical_performance.csv", ("baseline_rule_id",), "baseline_rules.csv", ("baseline_rule_id",)),
    ForeignKeyRule("performance.machine_type_id", "historical_performance.csv", ("machine_type_id",), "machine_types.csv", ("machine_type_id",)),
]


PROBABILITY_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "assembly_candidates.csv": (
        "transition_success_probability",
        "transition_rework_probability",
        "transition_scrap_probability",
        "failure_probability_after",
        "component_inspection_confidence",
    ),
    "component_inspection.csv": ("failure_probability_prior", "inspection_confidence"),
    "component_quality_scenarios.csv": (
        "baseline_state_probability_A",
        "baseline_state_probability_B",
        "baseline_state_probability_C",
        "baseline_state_probability_D",
        "adjusted_state_probability_A",
        "adjusted_state_probability_B",
        "adjusted_state_probability_C",
        "adjusted_state_probability_D",
        "failure_probability_mean_scenario",
        "failure_probability_p95_scenario",
        "missing_probability_scenario",
        "crack_probability_scenario",
        "out_of_tolerance_probability_scenario",
        "life_below_requirement_probability_scenario",
        "direct_reuse_probability",
        "repair_probability",
        "laser_repair_probability",
        "replacement_probability",
        "scrap_probability",
        "acceptance_probability_for_remanufacturing",
        "chance_quality_violation_probability",
    ),
    "environmental_parameters.csv": ("scenario_probability", "abatement_efficiency", "coolant_recycling_rate", "metal_chip_recycling_rate", "powder_recovery_rate"),
    "reliability_parameters.csv": (
        "scenario_probability",
        "probability_usable_A_or_B",
        "probability_serviceable_weighted",
        "transition_success_probability",
        "transition_rework_probability",
        "transition_scrap_probability",
        "survival_probability_at_min_component_life",
        "survival_probability_at_min_system_life",
        "survival_probability_at_warranty_life",
        "posterior_warranty_failure_probability",
        "reliability_estimate_at_warranty",
        "chance_constraint_alpha",
    ),
    "risk_parameters.csv": ("scenario_probability", "risk_probability", "residual_probability_after_mitigation"),
    "route_feasibility.csv": (
        "default_success_probability",
        "default_rework_probability",
        "default_failure_probability",
        "empirical_missing_rate",
        "empirical_crack_rate",
        "empirical_out_of_tolerance_rate",
        "empirical_life_below_requirement_rate",
    ),
    "route_operations.csv": ("optional_probability", "operation_yield_factor", "route_success_probability_reference"),
    "route_outcome_scenarios.csv": (
        "scenario_probability",
        "transition_prob_A_after",
        "transition_prob_B_after",
        "transition_prob_C_after",
        "transition_prob_D_after",
        "transition_prob_SCRAP_after",
        "probability_improve",
        "probability_same_state",
        "probability_degrade",
        "probability_usable_A_or_B",
        "route_success_probability",
        "route_rework_probability",
        "route_scrap_probability",
        "life_requirement_satisfied_probability",
        "posterior_warranty_failure_probability",
        "survival_probability_at_min_system_life",
    ),
    "route_state_transition.csv": (
        "scenario_probability",
        "transition_prob_A",
        "transition_prob_B",
        "transition_prob_C",
        "transition_prob_D",
        "transition_prob_SCRAP",
        "probability_improve",
        "probability_same_state",
        "probability_degrade",
        "probability_usable_A_or_B",
        "probability_serviceable_weighted",
        "default_success_probability",
        "default_rework_probability",
        "default_scrap_probability",
        "warranty_failure_probability_after",
        "conditional_reinspection_required_prob",
        "conditional_repair_retry_prob",
        "conditional_replacement_after_failure_prob",
    ),
    "scenarios.csv": (
        "scenario_probability",
        "macro_probability",
        "conditional_quality_process_probability",
        "saa_sample_weight",
        "scenario_cluster_weight",
        "baseline_acceptance_ratio_ref",
        "baseline_quality_A_share_ref",
        "baseline_quality_B_share_ref",
        "baseline_quality_C_share_ref",
        "baseline_quality_D_share_ref",
    ),
}


NONNEGATIVE_COLUMN_KEYWORDS: Tuple[str, ...] = (
    "cost",
    "time",
    "energy",
    "water",
    "pollutant",
    "carbon",
    "life",
    "quantity",
    "capacity",
    "hours",
    "_h",
    "kg",
    "m3",
    "price",
    "value",
    "penalty",
)


NEGATIVE_ALLOWED_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "environmental_parameters.csv": (
        "embedded_material_carbon_kg",
        "recycling_or_reuse_carbon_credit_kg",
        "carbon_kg",
        "green_benefit_credit_rmb",
        "net_environmental_cost_rmb",
    ),
}


BASELINE_WEIGHT_COLUMNS: Tuple[str, ...] = (
    "cost_weight",
    "time_weight",
    "environment_weight",
    "reliability_weight",
    "risk_weight",
    "assembly_weight",
    "reuse_weight",
)
