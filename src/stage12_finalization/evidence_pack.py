"""Claim-sentence and evidence-pack builders for Stage 12."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from .config import Stage12Config
from .io_utils import existing_stage11_paths, read_csv


CLAIM_SENTENCES: Dict[str, str] = {
    "T1_stagewise_model_complexity": (
        "The staged implementation builds from data validation to multiobjective matheuristics "
        "while preserving a traceable model-size record."
    ),
    "T2_baseline_and_ablation": (
        "The baseline and ablation evidence separates the effects of stochasticity, CVaR, "
        "selective assembly, Pareto analysis and matheuristic approximation."
    ),
    "T3_pareto_payoff_and_representatives": (
        "The payoff table and representative Pareto solutions define the cost, environmental "
        "and assembly-quality trade-offs used in the Results narrative."
    ),
    "T4_exact_vs_matheuristic_and_top5": (
        "The exact-versus-matheuristic and top-five benchmark evidence assesses whether "
        "restricted MILP repair preserves solution quality at larger scale."
    ),
    "T5_risk_selective_assembly_metrics": (
        "The risk and selective-assembly metrics link CVaR, backlog, assembly shortfall and "
        "route mix to the integrated Stage 6 model."
    ),
    "T6_stage10_strengthening": (
        "The Stage 10 strengthening evidence documents pair-carbon accounting and the "
        "shared-capacity extension as reviewer-facing robustness support."
    ),
    "T7_saa_sensitivity_manifest": (
        "The SAA and sensitivity manifest records which robustness experiments are complete "
        "and which remain bounded or unavailable."
    ),
    "F1_model_architecture": (
        "The model architecture figure presents the stochastic CVaR selective-assembly "
        "framework as a coupled production, risk and assembly-quality system."
    ),
    "F2_data_to_model_pipeline": (
        "The data pipeline figure shows how raw CSV evidence is transformed into validated "
        "instances, optimization outputs and manuscript artifacts."
    ),
    "F3_pareto_tradeoff_panels": (
        "The Pareto panels visualize the cost, environmental and assembly-quality trade-offs "
        "created by the augmented epsilon-constraint analysis."
    ),
    "F4_baseline_ablation_comparison": (
        "The baseline and ablation comparison figure summarizes how each modeling layer "
        "changes objective value, backlog, CVaR and assembly shortfall."
    ),
    "F5_matheuristic_convergence": (
        "The matheuristic convergence figure reports whether ALNS with restricted MILP repair "
        "stabilizes incumbents within the bounded search budget."
    ),
    "F6_exact_vs_matheuristic_top5": (
        "The exact-versus-matheuristic benchmark figure compares quality gaps, runtime and "
        "feasible repairs across exact and heuristic evidence."
    ),
    "F7_route_mix_and_operational_shift": (
        "The route-mix figure connects modeling additions to operational shifts in routing, "
        "procurement and backlog."
    ),
    "F8_stage10_strengthening": (
        "The Stage 10 strengthening figure decomposes environmental-accounting changes and "
        "shared-capacity effects."
    ),
}


REVIEWER_QUESTIONS: Dict[str, str] = {
    "T1_stagewise_model_complexity": "Does the paper document model growth without hiding infeasible or missing stages?",
    "T2_baseline_and_ablation": "Are the ablations comparable and tied to a single modeling change?",
    "T3_pareto_payoff_and_representatives": "Do the reported Pareto points support a real trade-off rather than a scalar-weight artifact?",
    "T4_exact_vs_matheuristic_and_top5": "How close is the heuristic evidence to exact Pareto references?",
    "T5_risk_selective_assembly_metrics": "Do risk and assembly metrics remain feasible under the same Stage 6 constraints?",
    "T6_stage10_strengthening": "Does the strengthened environmental objective or shared capacity alter the main claim?",
    "T7_saa_sensitivity_manifest": "Are SAA and sensitivity checks complete enough for manuscript robustness claims?",
    "F1_model_architecture": "Can every modeling block in the schematic be traced to implemented model reports?",
    "F2_data_to_model_pipeline": "Can the data-to-model chain be reproduced from the reported source files?",
    "F3_pareto_tradeoff_panels": "Are the Pareto points nondominated and sourced from Stage 7/8 outputs?",
    "F4_baseline_ablation_comparison": "Are missing metrics clearly separated from observed model differences?",
    "F5_matheuristic_convergence": "Does the convergence curve come from actual repair logs?",
    "F6_exact_vs_matheuristic_top5": "Are runtime and gap comparisons computed from the same benchmark scope?",
    "F7_route_mix_and_operational_shift": "Do route-mix changes have enough operational interpretation for IJPR readers?",
    "F8_stage10_strengthening": "Should this strengthening evidence stay in the appendix unless it becomes central?",
}


def build_claim_evidence_pack(
    config: Stage12Config,
    audit_catalogue: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create claim catalogue, curated main/appendix sets and an evidence pack."""

    if audit_catalogue.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    source_lookup = _source_lookup(config)
    rows: List[dict[str, object]] = []
    for audit_row in audit_catalogue.to_dict(orient="records"):
        artifact_id = str(audit_row.get("artifact_id", ""))
        claim_sentence = CLAIM_SENTENCES.get(artifact_id) or _fallback_claim(audit_row)
        source_files = source_lookup.get(artifact_id, [])
        evidence_source = "; ".join(source_files[:5])
        recommended_location = str(audit_row.get("recommended_location", "supplementary"))
        rows.append(
            {
                **audit_row,
                "claim_sentence": claim_sentence,
                "evidence_source": evidence_source,
                "reader_takeaway": _reader_takeaway(artifact_id, claim_sentence, recommended_location),
                "possible_reviewer_question": REVIEWER_QUESTIONS.get(
                    artifact_id,
                    "Is this artifact necessary for the main narrative and fully traceable?",
                ),
                "caption_note": _caption_note(artifact_id, recommended_location),
            }
        )

    catalogue = pd.DataFrame(rows)
    main = catalogue[catalogue["recommended_location"] == "main_text"].reset_index(drop=True)
    appendix = catalogue[catalogue["recommended_location"].isin(["appendix", "supplementary"])].reset_index(drop=True)
    evidence_pack = catalogue.sort_values(
        by=["recommended_location", "artifact_id"],
        key=lambda series: series.map({"main_text": "0", "appendix": "1", "supplementary": "2", "revise_before_use": "3"}).fillna("4")
        if series.name == "recommended_location"
        else series,
    ).reset_index(drop=True)
    return catalogue, main, appendix, evidence_pack


def _source_lookup(config: Stage12Config) -> Dict[str, List[str]]:
    paths = existing_stage11_paths(config)
    frames = []
    for key in ["table_source_map", "figure_source_map"]:
        path = paths[key]
        if path.exists():
            frames.append(read_csv(path))
    if not frames:
        return {}
    source_map = pd.concat(frames, ignore_index=True)
    if source_map.empty or "artifact_id" not in source_map.columns:
        return {}
    lookup: Dict[str, List[str]] = {}
    for artifact_id, group in source_map.groupby("artifact_id"):
        files = []
        for value in group.get("source_file", pd.Series(dtype=object)).astype(str).tolist():
            if value and value != "nan":
                files.append(value)
        lookup[str(artifact_id)] = _dedupe(files)
    return lookup


def _dedupe(values: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for value in values:
        normalized = str(Path(value)) if value else ""
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(normalized)
    return output


def _fallback_claim(row: dict[str, object]) -> str:
    claim = str(row.get("claim", "")).strip()
    if claim:
        return claim if claim.endswith(".") else f"{claim}."
    return "This artifact supports the manuscript evidence chain and requires source-traceable interpretation."


def _reader_takeaway(artifact_id: str, claim_sentence: str, location: str) -> str:
    if location == "main_text":
        return claim_sentence
    if artifact_id.startswith("T7"):
        return "Use this table to state which robustness checks are completed and which are bounded."
    if artifact_id in {"T6_stage10_strengthening", "F8_stage10_strengthening"}:
        return "Use this appendix evidence to answer reviewer concerns about pair carbon and shared capacity."
    return f"Use as supporting evidence: {claim_sentence}"


def _caption_note(artifact_id: str, location: str) -> str:
    if artifact_id.startswith("F"):
        base = "Caption should name the input stage outputs and avoid implying newly solved models."
    else:
        base = "Table note should state unavailable metrics as NA rather than estimated values."
    if location == "appendix":
        return f"{base} Position as appendix evidence unless it becomes central to the Results claim."
    return base
