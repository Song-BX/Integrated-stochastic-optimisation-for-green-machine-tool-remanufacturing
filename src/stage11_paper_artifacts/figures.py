"""Stage 11 manuscript figure builders."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from .config import Stage11Config
from .drawio_figures import build_drawio_data_pipeline, build_drawio_model_architecture
from .structures import PaperArtifactSpec, PaperFigure


PALETTE = {
    "neutral": "#667085",
    "light": "#d6dce5",
    "grid": "#e7eaf0",
    "blue": "#3f6f9f",
    "teal": "#4f9862",
    "orange": "#d9822b",
    "red": "#b94a48",
    "purple": "#8a6f8f",
    "grey": "#98a2b3",
    "ink": "#101828",
}


CMAP_BLUE = LinearSegmentedColormap.from_list("stage11_soft_blue", ["#f7fbff", "#d9e7f2", "#93b9d9", "#4f7da8", "#17365d"])
CMAP_ORANGE = LinearSegmentedColormap.from_list("stage11_soft_orange", ["#fffaf0", "#fee8b6", "#f6b95d", "#d97a22", "#7f3b08"])
CMAP_GREEN = LinearSegmentedColormap.from_list("stage11_soft_green", ["#fffde8", "#d9efb3", "#85c77f", "#348c55", "#14533d"])


def build_figures(snapshot: Dict[str, Any], config: Stage11Config, output_dir: Path) -> List[PaperFigure]:
    """Build all Stage 11 manuscript figures."""

    _set_style()
    builders = [
        build_drawio_model_architecture,
        build_drawio_data_pipeline,
        _figure_pareto_tradeoffs,
        _figure_baseline_ablation,
        _figure_matheuristic,
        _figure_exact_top5,
        _figure_route_shift,
        _figure_stage10,
    ]
    figures = []
    for builder in builders:
        figures.append(builder(snapshot, config, output_dir))
    return figures


def _figure_model_architecture(snapshot: Dict[str, Any], config: Stage11Config, output_dir: Path) -> PaperFigure:
    artifact_id = "F1_model_architecture"
    fig, ax = plt.subplots(figsize=(8.0, 4.25), dpi=config.dpi)
    ax.axis("off")
    blocks = [
        ("Condition-aware\nreturned cores\nquality states", 0.035, 0.62, 0.17, 0.20, PALETTE["blue"]),
        ("First-stage\nacceptance and\nbase procurement", 0.260, 0.62, 0.18, 0.20, PALETTE["teal"]),
        ("Scenario SAA\nquality, demand\nroute outcomes", 0.495, 0.62, 0.18, 0.20, PALETTE["orange"]),
        ("Second-stage\nrouting, inventory\nbacklog, overtime", 0.735, 0.62, 0.20, 0.20, PALETTE["purple"]),
        ("Reliability screen\nchance constraints\nCVaR tail loss", 0.230, 0.34, 0.22, 0.17, PALETTE["red"]),
        ("Selective assembly\ncompatibility pairs\ndimension chain", 0.500, 0.34, 0.22, 0.17, PALETTE["red"]),
        ("Multi-objective\nPareto analysis\nALNS-MILP repair", 0.760, 0.34, 0.18, 0.17, PALETTE["neutral"]),
    ]
    for label, x, y, w, h, color in blocks:
        _box(ax, x, y, w, h, label, color)
    for start, end in [(blocks[0], blocks[1]), (blocks[1], blocks[2]), (blocks[2], blocks[3])]:
        _arrow_between_boxes(ax, start, end, direction="right")
    _arrow_between_boxes(ax, blocks[2], blocks[4], direction="down")
    _arrow_between_boxes(ax, blocks[3], blocks[6], direction="down")
    _arrow_between_boxes(ax, blocks[4], blocks[5], direction="right")
    _arrow_between_boxes(ax, blocks[5], blocks[6], direction="right")
    _architecture_badges(ax)
    ax.text(0.035, 0.94, "(a) Integrated stochastic optimization architecture", fontsize=8, weight="bold")
    ax.text(
        0.035,
        0.06,
        "Claim: route choice, reliability risk, selective assembly and solution strategy form one coupled stochastic production system.",
        fontsize=7,
        color="#344054",
    )
    paths = _save_figure(fig, output_dir / artifact_id, config)
    spec = PaperArtifactSpec(
        artifact_id=artifact_id,
        artifact_type="figure",
        title="Integrated stochastic optimization architecture",
        claim="The proposed model integrates first-stage acceptance/procurement, scenario recourse, CVaR reliability risk, and selective assembly.",
        source_files=[
            str(config.data_results_dir / "stage6" / "stage6_selective_assembly_report.md"),
            str(config.data_results_dir / "stage7" / "stage7_pareto_report.md"),
            str(config.data_results_dir / "stage8" / "stage8_matheuristic_report.md"),
            str(config.data_results_dir / "stage10" / "stage10_strengthening_report.md"),
        ],
        output_files=[str(path) for path in paths.values()],
    )
    return PaperFigure(spec=spec, source_data=pd.DataFrame(blocks, columns=["label", "x", "y", "w", "h", "color"]), output_paths=paths)


def _figure_data_pipeline(snapshot: Dict[str, Any], config: Stage11Config, output_dir: Path) -> PaperFigure:
    artifact_id = "F2_data_to_model_pipeline"
    fig, ax = plt.subplots(figsize=(8.0, 3.9), dpi=config.dpi)
    ax.axis("off")
    catalogue = snapshot.get("stage1_catalogue", pd.DataFrame())
    total_files = int(len(catalogue)) if isinstance(catalogue, pd.DataFrame) and not catalogue.empty else 28
    total_rows = int(pd.to_numeric(catalogue.get("row_count"), errors="coerce").sum()) if isinstance(catalogue, pd.DataFrame) and "row_count" in catalogue.columns else 273013
    total_mb = float(pd.to_numeric(catalogue.get("size_mb"), errors="coerce").sum()) if isinstance(catalogue, pd.DataFrame) and "size_mb" in catalogue.columns else np.nan
    summary = snapshot.get("stage1_validation", {}).get("summary", {})
    baseline_rows = len(_csv(snapshot, "stage9_baseline"))
    sensitivity_rows = len(_csv(snapshot, "stage9_sensitivity"))
    nodes = [
        (f"Raw CSVs\n{total_files} files\n{_compact_number(total_rows)} rows", 0.02, 0.62, 0.14, 0.22),
        (f"Stage 1\nvalidation gate\nfailed={summary.get('failed', 'n/a')}", 0.195, 0.62, 0.15, 0.22),
        ("Stage 3-6\nMILP model chain\nrisk + assembly", 0.385, 0.62, 0.16, 0.22),
        ("Stage 7-8\nPareto + ALNS\nexact/repair", 0.590, 0.62, 0.15, 0.22),
        ("Stage 9-10\nexperiments +\nstrengthening", 0.785, 0.62, 0.17, 0.22),
        ("Stage 11-12\npaper artifacts\nand audits", 0.430, 0.10, 0.18, 0.22),
    ]
    for idx, (label, x, y, w, h) in enumerate(nodes):
        _box(ax, x, y, w, h, label, [PALETTE["blue"], PALETTE["teal"], PALETTE["orange"], PALETTE["purple"], PALETTE["red"], PALETTE["neutral"]][idx])
    for start, end in [(nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[3]), (nodes[3], nodes[4])]:
        _arrow_between_boxes(ax, start, end, direction="right")
    _elbow_arrow(ax, nodes[4], nodes[5])
    callouts = [
        ("Data size", f"{_compact_number(total_mb)} MB" if np.isfinite(total_mb) else "353 MB", 0.05, 0.47, PALETTE["blue"]),
        ("Gate checks", f"pass {summary.get('passed', 'n/a')} / warn {summary.get('warning', 'n/a')}", 0.22, 0.47, PALETTE["teal"]),
        ("Evidence rows", f"baseline {baseline_rows}, sensitivity {sensitivity_rows}", 0.42, 0.47, PALETTE["orange"]),
        ("Outputs", "7 tables + 8 figures", 0.62, 0.47, PALETTE["purple"]),
        ("Audit layer", "source + reality checks", 0.78, 0.47, PALETTE["red"]),
    ]
    for label, value, x, y, color in callouts:
        _metric_badge(ax, label, value, x, y, color)
    ax.text(0.03, 0.94, "(b) Data-to-model pipeline", fontsize=8, weight="bold")
    ax.text(0.03, 0.06, "Traceability is preserved from raw CSV validation to model runs, experiment collection, paper figures and reviewer-facing audits.", fontsize=7, color="#344054")
    paths = _save_figure(fig, output_dir / artifact_id, config)
    spec = PaperArtifactSpec(
        artifact_id=artifact_id,
        artifact_type="figure",
        title="Data-to-model pipeline",
        claim="The experiment evidence is traceable from raw CSV validation to manuscript-ready tables and figures.",
        source_files=[str(config.stage1_report)],
        output_files=[str(path) for path in paths.values()],
    )
    return PaperFigure(spec=spec, source_data=pd.DataFrame(nodes, columns=["label", "x", "y", "w", "h"]), output_paths=paths)


def _figure_pareto_tradeoffs(snapshot: Dict[str, Any], config: Stage11Config, output_dir: Path) -> PaperFigure:
    artifact_id = "F3_pareto_tradeoff_panels"
    grid = _csv(snapshot, "stage7_grid")
    pareto = _csv(snapshot, "stage7_pareto_front")
    payoff = _csv(snapshot, "stage7_payoff")
    feasible = grid[grid.get("feasible", pd.Series(dtype=int)) == 1] if not grid.empty else pd.DataFrame()
    fig, axes = plt.subplots(1, 2, figsize=(7.7, 3.35), dpi=config.dpi)
    _scatter_panel(axes[0], feasible, pareto, payoff, "economic_risk", "environmental_impact", "(a) Economic risk vs environmental impact")
    _scatter_panel(axes[1], feasible, pareto, payoff, "economic_risk", "assembly_quality_loss", "(b) Economic risk vs assembly quality loss")
    feasible_count = len(feasible)
    pareto_count = len(pareto)
    fig.text(
        0.50,
        0.015,
        f"Evidence encoded: {pareto_count} nondominated points from {feasible_count} feasible epsilon-grid solutions; stars mark representative objective anchors.",
        ha="center",
        va="bottom",
        fontsize=6.3,
        color="#344054",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    paths = _save_figure(fig, output_dir / artifact_id, config)
    data = pd.concat([feasible.assign(source="feasible_grid"), pareto.assign(source="pareto_front")], ignore_index=True, sort=False)
    spec = PaperArtifactSpec(
        artifact_id=artifact_id,
        artifact_type="figure",
        title="Pareto trade-off panels",
        claim="The augmented epsilon-constraint model quantifies economic-risk trade-offs against environmental and assembly-quality objectives.",
        source_files=[snapshot["source_paths"].get("stage7_grid", ""), snapshot["source_paths"].get("stage7_pareto_front", "")],
        output_files=[str(path) for path in paths.values()],
        warnings=[] if not data.empty else ["pareto_source_missing_or_empty"],
    )
    return PaperFigure(spec=spec, source_data=data, output_paths=paths)


def _figure_baseline_ablation(snapshot: Dict[str, Any], config: Stage11Config, output_dir: Path) -> PaperFigure:
    artifact_id = "F4_baseline_ablation_comparison"
    baseline = _csv(snapshot, "stage9_baseline")
    ablation = _csv(snapshot, "stage9_ablation")
    ablation_plot = _shorten_ablation_labels(ablation)
    fig = plt.figure(figsize=(7.9, 4.25), dpi=config.dpi)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 0.90], wspace=0.42, hspace=0.42)
    _metric_heatmap(fig.add_subplot(gs[:, 0]), baseline, "(a) Baseline metric dashboard")
    _lollipop_delta(fig.add_subplot(gs[0, 1]), ablation_plot, "display_id", "objective_delta", "(b) Objective delta")
    _ablation_multi_delta(fig.add_subplot(gs[1, 1]), ablation_plot, "(c) Operational and risk deltas")
    _figure_claim_text(
        fig,
        "Main reading: stochastic/CVaR layers alter delivery performance; selective assembly acts as a quality gate in the CK6150 instance.",
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.18)
    paths = _save_figure(fig, output_dir / artifact_id, config)
    data = pd.concat([baseline.assign(source="baseline"), ablation.assign(source="ablation")], ignore_index=True, sort=False)
    spec = PaperArtifactSpec(
        artifact_id=artifact_id,
        artifact_type="figure",
        title="Baseline and ablation comparison",
        claim="Baseline and ablation results isolate the contribution of stochasticity, CVaR, selective assembly, Pareto analysis, and matheuristic approximation.",
        source_files=[snapshot["source_paths"].get("stage9_baseline", ""), snapshot["source_paths"].get("stage9_ablation", "")],
        output_files=[str(path) for path in paths.values()],
        warnings=[] if not data.empty else ["baseline_or_ablation_missing"],
    )
    return PaperFigure(spec=spec, source_data=data, output_paths=paths)


def _figure_matheuristic(snapshot: Dict[str, Any], config: Stage11Config, output_dir: Path) -> PaperFigure:
    artifact_id = "F5_matheuristic_convergence"
    iteration = _csv(snapshot, "stage8_iteration_log")
    scores = _csv(snapshot, "stage8_operator_scores")
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.65), dpi=config.dpi, gridspec_kw={"width_ratios": [1.25, 1.0]})
    _convergence_panel(axes[0], iteration, "(a) ALNS incumbent convergence")
    _operator_bubble_panel(axes[1], scores, "(b) Operator use, score and reliability")
    _figure_claim_text(fig, "Repair solves stay feasible across the search; operator size encodes successful restricted-MILP repairs.")
    fig.tight_layout(w_pad=1.2, rect=[0, 0.06, 1, 1])
    paths = _save_figure(fig, output_dir / artifact_id, config)
    data = pd.concat([iteration.assign(source="iteration"), scores.assign(source="operator_scores")], ignore_index=True, sort=False)
    spec = PaperArtifactSpec(
        artifact_id=artifact_id,
        artifact_type="figure",
        title="Matheuristic convergence and operator scores",
        claim="ALNS-guided restricted MILP repair provides a reproducible search trajectory and interpretable operator contributions.",
        source_files=[snapshot["source_paths"].get("stage8_iteration_log", ""), snapshot["source_paths"].get("stage8_operator_scores", "")],
        output_files=[str(path) for path in paths.values()],
    )
    return PaperFigure(spec=spec, source_data=data, output_paths=paths)


def _figure_exact_top5(snapshot: Dict[str, Any], config: Stage11Config, output_dir: Path) -> PaperFigure:
    artifact_id = "F6_exact_vs_matheuristic_top5"
    gap = _csv(snapshot, "stage9_exact_gap")
    top5 = _csv(snapshot, "stage9_top5")
    fig = plt.figure(figsize=(7.9, 4.0), dpi=config.dpi)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 0.92], height_ratios=[0.75, 1.0], wspace=0.35, hspace=0.40)
    _benchmark_bubble_panel(fig.add_subplot(gs[:, 0]), top5, "(a) Runtime, Pareto coverage and repair feasibility")
    _benchmark_summary_panel(fig.add_subplot(gs[0, 1]), top5, gap, "(b) Benchmark summary")
    _benchmark_strip_panel(fig.add_subplot(gs[1, 1]), top5, "(c) Machine-level incumbent metrics")
    _figure_claim_text(fig, "Bubble position, size and colour jointly show runtime, Pareto set size, repair feasibility and economic-risk level.")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.18)
    paths = _save_figure(fig, output_dir / artifact_id, config)
    data = pd.concat([gap.assign(source="exact_gap"), top5.assign(source="top5")], ignore_index=True, sort=False)
    spec = PaperArtifactSpec(
        artifact_id=artifact_id,
        artifact_type="figure",
        title="Exact-vs-matheuristic and top5 benchmark",
        claim="The matheuristic preserves the exact economic anchor while scaling runtime and feasibility across machine types.",
        source_files=[snapshot["source_paths"].get("stage9_exact_gap", ""), snapshot["source_paths"].get("stage9_top5", "")],
        output_files=[str(path) for path in paths.values()],
    )
    return PaperFigure(spec=spec, source_data=data, output_paths=paths)


def _figure_route_shift(snapshot: Dict[str, Any], config: Stage11Config, output_dir: Path) -> PaperFigure:
    artifact_id = "F7_route_mix_and_operational_shift"
    route_mix = _route_mix(snapshot)
    baseline = _csv(snapshot, "stage9_baseline")
    audit = _csv(snapshot, "stage12_operational_audit")
    fig = plt.figure(figsize=(7.9, 4.05), dpi=config.dpi)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.18, 1.0], height_ratios=[1.0, 0.72], wspace=0.34, hspace=0.45)
    _route_heatmap(fig.add_subplot(gs[:, 0]), route_mix, "(a) Route-stage intensity")
    _operational_line_panel(fig.add_subplot(gs[0, 1]), baseline, "(b) Operational output shift")
    _audit_decision_panel(fig.add_subplot(gs[1, 1]), audit, "(c) Operational reality classification")
    _figure_claim_text(fig, "The operational shift is not a reuse-maximisation claim: strict reliability and assembly screens move the default solution toward fallback.")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.18)
    paths = _save_figure(fig, output_dir / artifact_id, config)
    data = pd.concat([route_mix.assign(source="route_mix"), baseline.assign(source="baseline"), audit.assign(source="operational_audit")], ignore_index=True, sort=False)
    spec = PaperArtifactSpec(
        artifact_id=artifact_id,
        artifact_type="figure",
        title="Route mix and operational shift",
        claim="Adding uncertainty, risk, and assembly constraints changes route use, backlog, and expected assembly output.",
        source_files=[
            snapshot["source_paths"].get("stage3_selected_routes", ""),
            snapshot["source_paths"].get("stage4_selected_routes", ""),
            snapshot["source_paths"].get("stage5_selected_routes", ""),
            snapshot["source_paths"].get("stage6_selected_routes", ""),
            snapshot["source_paths"].get("stage9_baseline", ""),
            snapshot["source_paths"].get("stage12_operational_audit", ""),
        ],
        output_files=[str(path) for path in paths.values()],
    )
    return PaperFigure(spec=spec, source_data=data, output_paths=paths)


def _figure_stage10(snapshot: Dict[str, Any], config: Stage11Config, output_dir: Path) -> PaperFigure:
    artifact_id = "F8_stage10_strengthening"
    env = _csv(snapshot, "stage10_env_breakdown")
    shared = _csv(snapshot, "stage10_shared_comparison")
    utilization = _csv(snapshot, "stage10_shared_utilization")
    fig = plt.figure(figsize=(7.9, 4.05), dpi=config.dpi)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.28], height_ratios=[1.0, 0.78], wspace=0.42, hspace=0.62)
    _environment_lollipop(fig.add_subplot(gs[:, 0]), env, "(a) Environmental coefficient contribution")
    _shared_capacity_dumbbell(fig.add_subplot(gs[0, 1]), shared, "(b) Shared-capacity mini experiment")
    _capacity_utilization_heatmap(fig.add_subplot(gs[1, 1]), utilization, "(c) Shared-resource utilization")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.16)
    paths = _save_figure(fig, output_dir / artifact_id, config)
    data = pd.concat([env.assign(source="env_breakdown"), shared.assign(source="shared_capacity"), utilization.assign(source="shared_capacity_utilization")], ignore_index=True, sort=False)
    spec = PaperArtifactSpec(
        artifact_id=artifact_id,
        artifact_type="figure",
        title="Stage 10 strengthening evidence",
        claim="Pair-carbon coefficients and shared-capacity coupling address two likely reviewer concerns without replacing the main model.",
        source_files=[snapshot["source_paths"].get("stage10_env_breakdown", ""), snapshot["source_paths"].get("stage10_shared_comparison", ""), snapshot["source_paths"].get("stage10_shared_utilization", "")],
        output_files=[str(path) for path in paths.values()],
    )
    return PaperFigure(spec=spec, source_data=data, output_paths=paths)


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.labelcolor": PALETTE["ink"],
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "legend.frameon": False,
        }
    )


def _architecture_badges(ax: plt.Axes) -> None:
    badges = [
        ("Decision variables", "accept, route, procure,\ninventory, assemble", 0.045, 0.16, PALETTE["teal"]),
        ("Uncertainty", "quality, demand,\nroute outcomes", 0.245, 0.16, PALETTE["orange"]),
        ("Constraints", "capacity, BOM,\nreliability, pairs", 0.445, 0.16, PALETTE["red"]),
        ("Objectives", "economic risk,\ncarbon, assembly loss", 0.645, 0.16, PALETTE["purple"]),
    ]
    for title, body, x, y, color in badges:
        _metric_badge(ax, title, body, x, y, color, width=0.16, height=0.118)


def _metric_badge(
    ax: plt.Axes,
    label: str,
    value: str,
    x: float,
    y: float,
    color: str,
    width: float = 0.15,
    height: float = 0.095,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=0.7,
        edgecolor=color,
        facecolor="#ffffff",
    )
    ax.add_patch(patch)
    ax.text(x + 0.012, y + height - 0.024, label, fontsize=5.8, color=color, weight="bold", va="top")
    ax.text(x + 0.012, y + height - 0.055, value, fontsize=5.6, color="#344054", va="top")


def _box(ax: plt.Axes, x: float, y: float, w: float, h: float, label: str, color: str) -> None:
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.01", linewidth=0.8, edgecolor=color, facecolor="white")
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=7, color="#101828")


def _arrow_between_boxes(ax: plt.Axes, start_box: tuple, end_box: tuple, direction: str) -> None:
    _, x0, y0, w0, h0, *_ = start_box
    _, x1, y1, w1, h1, *_ = end_box
    gap = 0.014
    if direction == "right":
        start = (x0 + w0 + gap, y0 + h0 / 2)
        end = (x1 - gap, y1 + h1 / 2)
    elif direction == "down":
        start = (x0 + w0 / 2, y0 - gap)
        end = (x1 + w1 / 2, y1 + h1 + gap)
    else:
        raise ValueError(f"Unsupported arrow direction: {direction}")
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, linewidth=0.9, color="#344054"))


def _elbow_arrow(ax: plt.Axes, start_box: tuple, end_box: tuple) -> None:
    _, x0, y0, w0, *_ = start_box
    _, x1, y1, w1, h1, *_ = end_box
    start = (x0 + w0 / 2, y0 - 0.012)
    lane_x = min(x0 + w0 + 0.035, 0.98)
    mid_y = y1 + h1 + 0.07
    mid_x = x1 + w1 / 2
    end = (mid_x, y1 + h1 + 0.012)
    ax.plot([start[0], lane_x, lane_x, mid_x], [start[1], start[1], mid_y, mid_y], color="#344054", linewidth=0.9)
    ax.add_patch(FancyArrowPatch((mid_x, mid_y), end, arrowstyle="-|>", mutation_scale=10, linewidth=0.9, color="#344054"))


def _save_figure(fig: plt.Figure, stem: Path, config: Stage11Config) -> Dict[str, Path]:
    paths = {}
    for ext in config.figure_formats:
        path = stem.with_suffix(f".{ext}")
        if ext == "png":
            fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")
        paths[ext] = path
    plt.close(fig)
    return paths


def _figure_claim_text(fig: plt.Figure, text: str) -> None:
    fig.text(
        0.5,
        0.025,
        text,
        ha="center",
        va="bottom",
        fontsize=6.2,
        color="#344054",
        bbox=dict(facecolor="#ffffff", edgecolor="#d0d5dd", boxstyle="round,pad=0.22", linewidth=0.6),
    )


def _scatter_panel(ax: plt.Axes, feasible: pd.DataFrame, pareto: pd.DataFrame, payoff: pd.DataFrame, x_col: str, y_col: str, title: str) -> None:
    if not feasible.empty and {x_col, y_col}.issubset(feasible.columns):
        feasible_plot = feasible.copy()
        feasible_plot[x_col] = pd.to_numeric(feasible_plot[x_col], errors="coerce")
        feasible_plot[y_col] = pd.to_numeric(feasible_plot[y_col], errors="coerce")
        feasible_plot = feasible_plot.dropna(subset=[x_col, y_col])
        dominated = feasible_plot.copy()
        if not pareto.empty and {x_col, y_col}.issubset(pareto.columns):
            pareto_pairs = set(
                zip(
                    pd.to_numeric(pareto[x_col], errors="coerce").round(6),
                    pd.to_numeric(pareto[y_col], errors="coerce").round(6),
                )
            )
            dominated = feasible_plot[
                ~pd.Series(list(zip(feasible_plot[x_col].round(6), feasible_plot[y_col].round(6))), index=feasible_plot.index).isin(pareto_pairs)
            ]
        ax.scatter(dominated[x_col], dominated[y_col], c=PALETTE["light"], s=16, label="Feasible/dominated", zorder=1)
    if not pareto.empty and {x_col, y_col}.issubset(pareto.columns):
        pareto_plot = pareto.copy()
        pareto_plot[x_col] = pd.to_numeric(pareto_plot[x_col], errors="coerce")
        pareto_plot[y_col] = pd.to_numeric(pareto_plot[y_col], errors="coerce")
        pareto_plot = pareto_plot.dropna(subset=[x_col, y_col])
        ax.plot(pareto_plot.sort_values(x_col)[x_col], pareto_plot.sort_values(x_col)[y_col], color=PALETTE["red"], linewidth=0.7, alpha=0.55, zorder=2)
        ax.scatter(pareto_plot[x_col], pareto_plot[y_col], facecolors="white", edgecolors=PALETTE["red"], linewidth=0.9, s=38, label="Pareto", zorder=3)
        if not pareto_plot.empty:
            min_risk = pareto_plot.loc[pareto_plot[x_col].idxmin()]
            min_obj = pareto_plot.loc[pareto_plot[y_col].idxmin()]
            ax.scatter([min_risk[x_col]], [min_risk[y_col]], marker="*", s=70, color=PALETTE["orange"], edgecolor="white", linewidth=0.4, label="Representative", zorder=4)
            _safe_annotation(ax, "economic\nanchor", min_risk[x_col], min_risk[y_col], xytext=(10, -22), color=PALETTE["orange"])
            if min_obj.name != min_risk.name:
                ax.scatter([min_obj[x_col]], [min_obj[y_col]], marker="*", s=70, color=PALETTE["teal"], edgecolor="white", linewidth=0.4, zorder=4)
                ideal_label = "environment\nideal" if y_col == "environmental_impact" else "assembly\nideal"
                _safe_annotation(ax, ideal_label, min_obj[x_col], min_obj[y_col], xytext=(-54, 14), color=PALETTE["teal"])
    if not payoff.empty and {"payoff_name", x_col, y_col}.issubset(payoff.columns):
        payoff_plot = payoff.copy()
        payoff_plot[x_col] = pd.to_numeric(payoff_plot[x_col], errors="coerce")
        payoff_plot[y_col] = pd.to_numeric(payoff_plot[y_col], errors="coerce")
        payoff_plot = payoff_plot.dropna(subset=[x_col, y_col])
        for _, row in payoff_plot.iterrows():
            marker = "P" if row["payoff_name"] == "economic_risk_anchor" else "X"
            ax.scatter([row[x_col]], [row[y_col]], marker=marker, s=38, color=PALETTE["ink"], edgecolor="white", linewidth=0.4, zorder=5)
        _objective_range_inset(ax, payoff_plot, x_col, y_col)
    ax.set_title(title, fontsize=8, loc="left")
    ax.set_xlabel(_label(x_col))
    ax.set_ylabel(_label(y_col))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=5.8, loc="lower left")


def _bar_metric(ax: plt.Axes, frame: pd.DataFrame, x_col: str, y_col: str, title: str, color: str) -> None:
    if frame.empty or x_col not in frame.columns or y_col not in frame.columns:
        _empty_axis(ax, "missing source")
        ax.set_title(title, fontsize=8, loc="left")
        return
    data = frame[[x_col, y_col]].copy()
    data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
    data = data.dropna(subset=[y_col])
    if data.empty:
        _empty_axis(ax, "no comparable values")
    else:
        ax.bar(data[x_col].astype(str), data[y_col], color=color)
        ax.tick_params(axis="x", rotation=30)
    ax.set_title(title, fontsize=8, loc="left")
    ax.set_ylabel(_label(y_col))
    ax.grid(True, axis="y", alpha=0.25)


def _metric_heatmap(ax: plt.Axes, frame: pd.DataFrame, title: str) -> None:
    metrics = [
        ("economic_risk", "Economic\nrisk"),
        ("expected_final_backlog_units", "Final\nbacklog"),
        ("cvar_value", "CVaR"),
        ("expected_assembled_units", "Assembled"),
        ("expected_assembly_shortfall_units", "Assembly\nshortfall"),
    ]
    if frame.empty or "model_stage" not in frame.columns:
        _empty_axis(ax, "baseline missing")
        ax.set_title(title, fontsize=8, loc="left")
        return
    stages = [stage for stage in ["Stage3", "Stage4", "Stage5", "Stage6", "Stage7", "Stage8"] if stage in set(frame["model_stage"].astype(str))]
    data = frame.set_index("model_stage")
    matrix = []
    labels = []
    for metric, _ in metrics:
        values = pd.to_numeric(data.reindex(stages)[metric], errors="coerce") if metric in data.columns else pd.Series(np.nan, index=stages)
        max_value = values.max(skipna=True)
        if pd.isna(max_value) or abs(max_value) < 1e-12:
            normalized = values * np.nan
        else:
            normalized = values / max_value
        matrix.append(normalized.to_numpy(dtype=float))
        labels.append(values.to_numpy(dtype=float))
    arr = np.array(matrix, dtype=float)
    im = ax.imshow(np.ma.masked_invalid(arr), aspect="auto", cmap=CMAP_BLUE, vmin=0, vmax=1)
    ax.set_title(title, fontsize=8, loc="left")
    ax.set_xticks(np.arange(len(stages)))
    ax.set_xticklabels(stages, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels([label for _, label in metrics])
    ax.tick_params(length=0)
    for row_idx in range(arr.shape[0]):
        for col_idx in range(arr.shape[1]):
            value = labels[row_idx][col_idx]
            if np.isfinite(value):
                prefix = ""
                if col_idx > 0:
                    prev = labels[row_idx][col_idx - 1]
                    if np.isfinite(prev) and abs(value - prev) > 1e-8:
                        prefix = "↑" if value > prev else "↓"
                text_color = "white" if np.isfinite(arr[row_idx, col_idx]) and arr[row_idx, col_idx] > 0.72 else PALETTE["ink"]
                ax.text(col_idx, row_idx, f"{prefix}{_compact_number(value)}", ha="center", va="center", fontsize=5.5, color=text_color)
            else:
                ax.text(col_idx, row_idx, "NA", ha="center", va="center", fontsize=5.5, color="#98a2b3")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Normalized", fontsize=6)
    cbar.ax.tick_params(labelsize=6, length=2)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _lollipop_delta(ax: plt.Axes, frame: pd.DataFrame, x_col: str, y_col: str, title: str) -> None:
    if frame.empty or x_col not in frame.columns or y_col not in frame.columns:
        _empty_axis(ax, "ablation missing")
        ax.set_title(title, fontsize=8, loc="left")
        return
    data = frame[[x_col, y_col]].copy()
    data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
    data = data.dropna(subset=[y_col])
    if data.empty:
        _empty_axis(ax, "no comparable values")
        ax.set_title(title, fontsize=8, loc="left")
        return
    order = np.arange(len(data))
    colors = [PALETTE["red"] if value > 1e-6 else PALETTE["teal"] if value < -1e-6 else PALETTE["neutral"] for value in data[y_col]]
    ax.axvline(0, color=PALETTE["light"], linewidth=0.8)
    for idx, value in enumerate(data[y_col]):
        ax.plot([0, value], [idx, idx], color="#d0d5dd", linewidth=1.1)
    ax.scatter(data[y_col], order, s=55, color=colors, edgecolor="white", linewidth=0.6, zorder=3)
    for idx, value in enumerate(data[y_col]):
        ax.text(value, idx + 0.16, _compact_number(value), ha="center", va="bottom", fontsize=5.8, color="#344054")
    ax.set_yticks(order)
    ax.set_yticklabels(data[x_col])
    ax.invert_yaxis()
    ax.set_title(title, fontsize=8, loc="left")
    ax.set_xlabel("Objective delta")
    _quiet_grid(ax, axis="x")


def _ablation_multi_delta(ax: plt.Axes, frame: pd.DataFrame, title: str) -> None:
    metrics = [
        ("expected_backlog_delta", "Backlog"),
        ("cvar_delta", "CVaR"),
        ("assembly_shortfall_delta", "Shortfall"),
        ("runtime_delta_seconds", "Runtime"),
    ]
    if frame.empty or "display_id" not in frame.columns:
        _empty_axis(ax, "ablation deltas missing")
        ax.set_title(title, fontsize=8, loc="left")
        return
    labels = frame["display_id"].astype(str).tolist()
    matrix = []
    raw = []
    for metric, _ in metrics:
        values = pd.to_numeric(frame.get(metric, pd.Series(np.nan, index=frame.index)), errors="coerce")
        scale = values.abs().max(skipna=True)
        normalized = values / scale if np.isfinite(scale) and scale > 1e-12 else values * np.nan
        matrix.append(normalized.to_numpy(dtype=float))
        raw.append(values.to_numpy(dtype=float))
    arr = np.array(matrix)
    im = ax.imshow(np.ma.masked_invalid(arr), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title(title, fontsize=8, loc="left")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=5.6)
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels([label for _, label in metrics])
    ax.tick_params(length=0)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = raw[i][j]
            if np.isfinite(value):
                ax.text(j, i, _compact_number(value), ha="center", va="center", fontsize=5.1, color="#101828")
            else:
                ax.text(j, i, "NA", ha="center", va="center", fontsize=5.1, color="#98a2b3")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Signed normalized delta", fontsize=5.7)
    cbar.ax.tick_params(labelsize=5.5, length=2)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _convergence_panel(ax: plt.Axes, iteration: pd.DataFrame, title: str) -> None:
    if iteration.empty or not {"iteration", "economic_risk", "feasible"}.issubset(iteration.columns):
        _empty_axis(ax, "iteration log missing")
        ax.set_title(title, fontsize=8, loc="left")
        return
    data = iteration.copy()
    data["iteration"] = pd.to_numeric(data["iteration"], errors="coerce")
    data["economic_risk"] = pd.to_numeric(data["economic_risk"], errors="coerce")
    data["objective_value"] = pd.to_numeric(data.get("objective_value"), errors="coerce")
    data["feasible_bool"] = data["feasible"].astype(str).str.lower().isin(["true", "1"])
    data["accepted_bool"] = data.get("accepted", pd.Series(False, index=data.index)).astype(str).str.lower().isin(["true", "1"])
    data["improved_bool"] = data.get("improved", pd.Series(False, index=data.index)).astype(str).str.lower().isin(["true", "1"])
    data = data.dropna(subset=["iteration", "economic_risk"]).sort_values(["iteration", "grid_id" if "grid_id" in data.columns else "iteration"])
    feasible = data[data["feasible_bool"]].copy()
    if feasible.empty:
        _empty_axis(ax, "no feasible repairs")
        ax.set_title(title, fontsize=8, loc="left")
        return
    incumbent = feasible.groupby("iteration", as_index=False)["economic_risk"].min().sort_values("iteration")
    incumbent["best_economic_risk"] = incumbent["economic_risk"].cummin()
    sizes = np.where(feasible["accepted_bool"], 34, 18)
    ax.scatter(feasible["iteration"], feasible["economic_risk"], s=sizes, color=PALETTE["grey"], edgecolor="white", linewidth=0.4, alpha=0.70, label="Repair solve", zorder=2)
    ax.plot(incumbent["iteration"], incumbent["economic_risk"], linewidth=0.8, color=PALETTE["grey"], linestyle="--", label="Incumbent", zorder=1)
    ax.plot(incumbent["iteration"], incumbent["best_economic_risk"], linewidth=1.4, color=PALETTE["blue"], label="Best-so-far", zorder=3)
    improved = feasible[feasible["improved_bool"] & (feasible["iteration"] > 0)]
    if not improved.empty:
        ax.scatter(improved["iteration"], improved["economic_risk"], marker="*", s=70, color=PALETTE["orange"], edgecolor="white", linewidth=0.4, label="Improved", zorder=4)
    best_row = incumbent.loc[incumbent["best_economic_risk"].idxmin()]
    _safe_annotation(
        ax,
        f"best\n{_compact_number(best_row['best_economic_risk'])}",
        best_row["iteration"],
        best_row["best_economic_risk"],
        xytext=(8, 18),
        color=PALETTE["blue"],
    )
    _candidate_pool_inset(ax, feasible)
    ax.text(
        0.98,
        0.06,
        f"feasible {int(feasible['feasible_bool'].sum())}/{len(data)}; accepted {int(feasible['accepted_bool'].sum())}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        color="#344054",
        bbox=dict(facecolor="white", edgecolor="#d0d5dd", boxstyle="round,pad=0.18"),
    )
    ax.set_title(title, fontsize=8, loc="left")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Economic risk")
    ax.legend(fontsize=5.4, loc="upper center", bbox_to_anchor=(0.58, 0.99), ncols=2, handlelength=1.2, columnspacing=0.9)
    _quiet_grid(ax)


def _operator_bubble_panel(ax: plt.Axes, scores: pd.DataFrame, title: str) -> None:
    required = {"operator_name", "uses", "successes", "score", "success_rate"}
    if scores.empty or not required.issubset(scores.columns):
        _empty_axis(ax, "operator scores missing")
        ax.set_title(title, fontsize=8, loc="left")
        return
    data = scores.copy()
    for column in ["uses", "successes", "score", "success_rate"]:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)
    size = 45 + 260 * data["successes"] / max(float(data["successes"].max()), 1.0)
    scatter = ax.scatter(data["uses"], data["score"], s=size, c=data["success_rate"], cmap=CMAP_GREEN, vmin=0, vmax=1, edgecolor="#344054", linewidth=0.4, alpha=0.9)
    top = data.sort_values(["score", "uses"], ascending=False).head(3)
    for _, row in top.iterrows():
        offset = (-30, -4) if row["score"] >= data["score"].max() else (4, 4)
        ha = "right" if offset[0] < 0 else "left"
        ax.annotate(str(row["operator_name"]).replace("_", "\n"), (row["uses"], row["score"]), xytext=offset, textcoords="offset points", fontsize=5.6, ha=ha)
    ax.set_title(title, fontsize=8, loc="left")
    ax.set_xlabel("Uses")
    ax.set_ylabel("Score")
    ax.margins(x=0.10, y=0.18)
    _quiet_grid(ax)
    cbar = ax.figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Success rate", fontsize=6)
    cbar.ax.tick_params(labelsize=6, length=2)


def _benchmark_bubble_panel(ax: plt.Axes, top5: pd.DataFrame, title: str) -> None:
    required = {"machine_type_id", "wall_seconds", "approx_pareto_points", "repair_solves", "feasible_repair_solves", "best_economic_risk"}
    if top5.empty or not required.issubset(top5.columns):
        _empty_axis(ax, "top5 benchmark missing")
        ax.set_title(title, fontsize=8, loc="left")
        return
    data = top5.copy()
    for column in ["wall_seconds", "approx_pareto_points", "repair_solves", "feasible_repair_solves", "best_economic_risk"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["feasible_repair_rate"] = data["feasible_repair_solves"] / data["repair_solves"].replace(0, np.nan)
    data = data.dropna(subset=["wall_seconds", "approx_pareto_points", "feasible_repair_rate", "best_economic_risk"])
    if data.empty:
        _empty_axis(ax, "no benchmark values")
        ax.set_title(title, fontsize=8, loc="left")
        return
    sizes = 90 + 360 * data["feasible_repair_rate"]
    scatter = ax.scatter(data["wall_seconds"], data["approx_pareto_points"], s=sizes, c=data["best_economic_risk"], cmap=CMAP_BLUE, edgecolor="#344054", linewidth=0.5, alpha=0.9)
    max_x = data["wall_seconds"].max()
    max_y = data["approx_pareto_points"].max()
    median_runtime = data["wall_seconds"].median()
    median_pareto = data["approx_pareto_points"].median()
    ax.axvline(median_runtime, color=PALETTE["grey"], linewidth=0.7, linestyle=":", zorder=0)
    ax.axhline(median_pareto, color=PALETTE["grey"], linewidth=0.7, linestyle=":", zorder=0)
    ax.text(
        0.02,
        0.96,
        f"median runtime {median_runtime:.0f}s\nmedian Pareto points {median_pareto:.0f}\nbubble size = repair rate",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.6,
        color="#344054",
        bbox=dict(facecolor="white", edgecolor="#d0d5dd", boxstyle="round,pad=0.18", linewidth=0.6),
    )
    for _, row in data.iterrows():
        offset = (-8, 4) if row["wall_seconds"] > max_x * 0.93 else (4, 3)
        ha = "right" if offset[0] < 0 else "left"
        va = "bottom" if row["approx_pareto_points"] < max_y * 0.96 else "top"
        ax.annotate(row["machine_type_id"], (row["wall_seconds"], row["approx_pareto_points"]), xytext=offset, textcoords="offset points", fontsize=6, ha=ha, va=va)
    ax.set_title(title, fontsize=8, loc="left")
    ax.set_xlabel("Wall seconds")
    ax.set_ylabel("Approx. Pareto points")
    ax.margins(x=0.08, y=0.18)
    _quiet_grid(ax)
    cbar = ax.figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Best economic risk", fontsize=6)
    cbar.ax.tick_params(labelsize=6, length=2)


def _benchmark_summary_panel(ax: plt.Axes, top5: pd.DataFrame, gap: pd.DataFrame, title: str) -> None:
    ax.axis("off")
    ax.set_title(title, fontsize=8, loc="left")
    success_count = int(top5["success"].astype(str).str.lower().isin(["true", "1"]).sum()) if "success" in top5.columns else 0
    total_count = len(top5)
    anchor_gap = "NA"
    if not gap.empty and "economic_risk_gap_pct" in gap.columns:
        anchor_gap = f"{pd.to_numeric(gap['economic_risk_gap_pct'], errors='coerce').iloc[0]:.3g}%"
    lines = [
        ("Success", f"{success_count}/{total_count}"),
        ("Anchor gap", anchor_gap),
        ("Median runtime", _compact_number(pd.to_numeric(top5.get("wall_seconds"), errors="coerce").median() if not top5.empty else np.nan)),
        ("Median repair rate", _format_rate(_safe_median_repair_rate(top5))),
    ]
    y = 0.82
    for label, value in lines:
        ax.text(0.04, y, label, transform=ax.transAxes, fontsize=6.2, color="#667085", ha="left")
        ax.text(0.96, y, value, transform=ax.transAxes, fontsize=7.2, color="#101828", ha="right", weight="bold")
        y -= 0.18


def _benchmark_strip_panel(ax: plt.Axes, top5: pd.DataFrame, title: str) -> None:
    required = {"machine_type_id", "feasible_repair_solves", "repair_solves", "incumbent_environmental_impact", "incumbent_assembly_quality_loss"}
    if top5.empty or not required.issubset(top5.columns):
        _empty_axis(ax, "benchmark strips missing")
        ax.set_title(title, fontsize=8, loc="left")
        return
    data = top5.copy()
    data["repair_rate"] = pd.to_numeric(data["feasible_repair_solves"], errors="coerce") / pd.to_numeric(data["repair_solves"], errors="coerce").replace(0, np.nan)
    metrics = [
        ("repair_rate", "Repair\nrate"),
        ("incumbent_environmental_impact", "Incumbent\nenv."),
        ("incumbent_assembly_quality_loss", "Incumbent\nassembly"),
    ]
    machines = data["machine_type_id"].astype(str).tolist()
    matrix = []
    raw = []
    for metric, _ in metrics:
        values = pd.to_numeric(data[metric], errors="coerce")
        scale = values.max(skipna=True)
        normalized = values / scale if np.isfinite(scale) and scale > 1e-12 else values * np.nan
        matrix.append(normalized.to_numpy(dtype=float))
        raw.append(values.to_numpy(dtype=float))
    arr = np.array(matrix)
    im = ax.imshow(np.ma.masked_invalid(arr), aspect="auto", cmap=CMAP_BLUE, vmin=0, vmax=1)
    ax.set_title(title, fontsize=8, loc="left")
    ax.set_xticks(np.arange(len(machines)))
    ax.set_xticklabels(machines, rotation=35, ha="right", fontsize=5.4)
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels([label for _, label in metrics])
    ax.tick_params(length=0)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = raw[i][j]
            text = _format_rate(value) if metrics[i][0] == "repair_rate" else _compact_number(value)
            ax.text(j, i, text, ha="center", va="center", fontsize=4.9, color="white" if np.isfinite(arr[i, j]) and arr[i, j] > 0.66 else PALETTE["ink"])
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Normalized", fontsize=5.7)
    cbar.ax.tick_params(labelsize=5.5, length=2)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _route_heatmap(ax: plt.Axes, route_mix: pd.DataFrame, title: str) -> None:
    stages = ["Stage3", "Stage4", "Stage5", "Stage6"]
    routes = [f"R{i}" for i in range(1, 8)]
    if route_mix.empty:
        pivot = pd.DataFrame(0, index=stages, columns=routes)
    else:
        pivot = route_mix.pivot_table(index="stage", columns="route_id", values="count", fill_value=0).reindex(index=stages, columns=routes, fill_value=0)
    values = pivot.to_numpy(dtype=float)
    row_sums = values.sum(axis=1, keepdims=True)
    shares = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums > 0)
    im = ax.imshow(shares, aspect="auto", cmap=CMAP_ORANGE, vmin=0, vmax=max(float(np.nanmax(shares)), 0.01))
    ax.set_title(title, fontsize=8, loc="left")
    ax.set_xticks(np.arange(len(routes)))
    ax.set_xticklabels(routes)
    ax.set_yticks(np.arange(len(stages)))
    ax.set_yticklabels(stages)
    ax.tick_params(length=0)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if values[i, j] > 0:
                text_color = "white" if shares[i, j] > 0.60 else PALETTE["ink"]
                ax.text(j, i, str(int(values[i, j])), ha="center", va="center", fontsize=6, color=text_color)
        if row_sums[i, 0] == 0:
            ax.text(3, i, "no old-part routes", ha="center", va="center", fontsize=5.8, color="#667085", style="italic")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Route share", fontsize=6)
    cbar.ax.tick_params(labelsize=6, length=2)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _operational_line_panel(ax: plt.Axes, baseline: pd.DataFrame, title: str) -> None:
    if baseline.empty or "model_stage" not in baseline.columns:
        _empty_axis(ax, "baseline missing")
        ax.set_title(title, fontsize=8, loc="left")
        return
    line = baseline[baseline["model_stage"].isin(["Stage3", "Stage4", "Stage5", "Stage6", "Stage7", "Stage8"])].copy()
    if line.empty:
        _empty_axis(ax, "no operational stages")
        ax.set_title(title, fontsize=8, loc="left")
        return
    x = np.arange(len(line))
    stage_labels = line["model_stage"].astype(str).tolist()
    if "Stage6" in stage_labels:
        stage6_idx = stage_labels.index("Stage6")
        ax.axvspan(stage6_idx - 0.28, stage6_idx + 0.28, color=PALETTE["orange"], alpha=0.10, zorder=0)
        ax.text(
            stage6_idx + 0.08,
            0.94,
            "quality gate\nfallback",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=5.7,
            color="#7a3e00",
            bbox=dict(facecolor="white", edgecolor="#f6b95d", boxstyle="round,pad=0.14", linewidth=0.5),
        )
    if "Stage3" in stage_labels:
        ax.text(
            0.12,
            0.08,
            "Stage3 route-only baseline;\noutput metrics start at Stage4",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=5.4,
            color="#667085",
            bbox=dict(facecolor="white", edgecolor="#d0d5dd", boxstyle="round,pad=0.12", linewidth=0.45),
        )
    if "expected_final_backlog_units" in line.columns:
        ax.plot(x, pd.to_numeric(line["expected_final_backlog_units"], errors="coerce"), marker="o", label="Backlog", color=PALETTE["orange"], linewidth=1.2)
    if "expected_assembled_units" in line.columns:
        ax.plot(x, pd.to_numeric(line["expected_assembled_units"], errors="coerce"), marker="s", label="Assembled", color=PALETTE["blue"], linewidth=1.2)
    if "expected_assembly_shortfall_units" in line.columns:
        shortfall = pd.to_numeric(line["expected_assembly_shortfall_units"], errors="coerce")
        if shortfall.notna().any():
            ax.plot(x, shortfall, marker="^", label="Assembly shortfall", color=PALETTE["red"], linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(line["model_stage"], rotation=30, ha="right")
    ax.set_title(title, fontsize=8, loc="left")
    ax.set_ylabel("Expected units")
    ax.legend(fontsize=6)
    _quiet_grid(ax)


def _audit_decision_panel(ax: plt.Axes, audit: pd.DataFrame, title: str) -> None:
    ax.set_title(title, fontsize=8, loc="left")
    ax.axis("off")
    if audit.empty or "operational_reality_decision" not in audit.columns:
        _empty_axis(ax, "operational audit unavailable")
        return
    main = audit[audit.get("evidence_decision", pd.Series(dtype=object)).astype(str).str.contains("main", case=False, na=False)].copy()
    if main.empty:
        main = audit.copy()
    counts = main["operational_reality_decision"].astype(str).value_counts()
    colors = {
        "realistic_main_claim": PALETTE["teal"],
        "explain_as_risk_averse_behavior": PALETTE["orange"],
        "do_not_use_as_main_claim": PALETTE["red"],
    }
    labels = [
        ("realistic_main_claim", "Realistic"),
        ("explain_as_risk_averse_behavior", "Risk-averse"),
        ("do_not_use_as_main_claim", "Do not use"),
    ]
    total = max(int(counts.sum()), 1)
    x0 = 0.04
    y0 = 0.55
    for key, label in labels:
        count = int(counts.get(key, 0))
        width = 0.88 * count / total
        if width > 0:
            ax.add_patch(FancyBboxPatch((x0, y0), width, 0.14, boxstyle="round,pad=0.01,rounding_size=0.01", linewidth=0, facecolor=colors[key], alpha=0.85, transform=ax.transAxes))
            ax.text(x0 + width / 2, y0 + 0.07, str(count), transform=ax.transAxes, ha="center", va="center", fontsize=6, color="white", weight="bold")
            x0 += width
    y = 0.34
    for key, label in labels:
        ax.scatter([0.08], [y], transform=ax.transAxes, s=34, color=colors[key])
        ax.text(0.15, y, label, transform=ax.transAxes, ha="left", va="center", fontsize=6.1, color="#344054")
        y -= 0.14
    warning_rows = main[main.get("status", pd.Series(dtype=object)).astype(str) == "warning"]
    ax.text(0.92, 0.05, f"warnings: {len(warning_rows)}", transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color="#667085")


def _environment_lollipop(ax: plt.Axes, env: pd.DataFrame, title: str) -> None:
    if env.empty or not {"component", "coefficient_sum"}.issubset(env.columns):
        _empty_axis(ax, "environment breakdown missing")
        ax.set_title(title, fontsize=8, loc="left")
        return
    labels = {
        "route_carbon": "Route\ncarbon",
        "procurement_embedded_carbon": "Procurement\nembedded",
        "assembly_pair_carbon": "Assembly\npair",
    }
    data = env[env["component"].isin(labels)].copy()
    data["display"] = data["component"].map(labels)
    data["coefficient_sum"] = pd.to_numeric(data["coefficient_sum"], errors="coerce")
    data = data.dropna(subset=["coefficient_sum"])
    data = data[data["coefficient_sum"] > 0].sort_values("coefficient_sum")
    if data.empty:
        _empty_axis(ax, "no environmental values")
        ax.set_title(title, fontsize=8, loc="left")
        return
    y = np.arange(len(data))
    x_start = max(data["coefficient_sum"].min() * 0.45, 1e-3)
    total = data["coefficient_sum"].sum()
    ax.hlines(y, x_start, data["coefficient_sum"], color="#d0d5dd", linewidth=1.2)
    ax.scatter(data["coefficient_sum"], y, s=70, color=[PALETTE["blue"], PALETTE["orange"], PALETTE["teal"]][: len(data)], edgecolor="white", linewidth=0.6, zorder=3)
    for idx, value in enumerate(data["coefficient_sum"]):
        dy = -0.18 if idx == len(data) - 1 else 0.12
        va = "top" if dy < 0 else "bottom"
        share = value / total if total > 0 else np.nan
        ax.text(value, idx + dy, f"{_compact_number(value)}\n({_format_nonzero_rate(share)})", ha="center", va=va, fontsize=5.5, color="#344054")
    ax.set_yticks(y)
    ax.set_yticklabels(data["display"])
    ax.set_title(title, fontsize=8, loc="left")
    ax.set_xlabel("Coefficient sum (log scale)")
    ax.set_xscale("log")
    ax.set_xlim(x_start, data["coefficient_sum"].max() * 1.8)
    _quiet_grid(ax, axis="x")


def _shared_capacity_dumbbell(ax: plt.Axes, shared: pd.DataFrame, title: str) -> None:
    required = {"capacity_mode", "objective_value", "expected_assembled_units", "expected_final_backlog_units", "solve_seconds"}
    if shared.empty or not required.issubset(shared.columns):
        _empty_axis(ax, "shared-capacity comparison missing")
        ax.set_title(title, fontsize=8, loc="left")
        return
    data = shared.set_index("capacity_mode")
    if "independent_capacity_total" not in data.index or "shared_capacity" not in data.index:
        _empty_axis(ax, "paired modes missing")
        ax.set_title(title, fontsize=8, loc="left")
        return
    metrics = [
        ("objective_value", "Objective"),
        ("expected_assembled_units", "Assembled"),
        ("expected_final_backlog_units", "Backlog"),
        ("solve_seconds", "Runtime"),
    ]
    rows = []
    for metric, label in metrics:
        independent = pd.to_numeric(pd.Series([data.loc["independent_capacity_total", metric]]), errors="coerce").iloc[0]
        coupled = pd.to_numeric(pd.Series([data.loc["shared_capacity", metric]]), errors="coerce").iloc[0]
        max_value = max(abs(independent), abs(coupled), 1e-9)
        rows.append((label, independent / max_value, coupled / max_value, independent, coupled))
    y = np.arange(len(rows))
    for idx, (label, independent_norm, coupled_norm, independent, coupled) in enumerate(rows):
        ax.plot([independent_norm, coupled_norm], [idx, idx], color="#98a2b3", linewidth=1.2, zorder=1)
    ax.scatter([row[1] for row in rows], y, color=PALETTE["blue"], s=45, label="Independent total", zorder=3)
    ax.scatter([row[2] for row in rows], y, color=PALETTE["red"], s=45, label="Shared capacity", marker="D", zorder=3)
    for idx, (_, _, _, independent, coupled) in enumerate(rows):
        label_x = 1.06 if idx != 0 else 1.08
        ax.text(label_x, idx, f"{_compact_number(independent)} -> {_compact_number(coupled)}", ha="left", va="center", fontsize=5.6, color="#344054")
    ax.set_yticks(y)
    ax.set_yticklabels([row[0] for row in rows])
    ax.set_xlim(-0.06, 1.68)
    ax.set_xlabel("Metric normalized within row")
    ax.set_title(title, fontsize=8, loc="left")
    _quiet_grid(ax, axis="x")
    ax.legend(fontsize=5.8, loc="upper left", bbox_to_anchor=(0.02, 0.88), ncols=1)
    if all(abs(row[3] - row[4]) < 1e-8 for row in rows[:3]):
        ax.text(
            0.97,
            0.88,
            "No material operational change\nin this mini instance",
            transform=ax.transAxes,
            fontsize=5.8,
            color="#667085",
            ha="right",
            va="top",
            bbox=dict(facecolor="white", edgecolor="#d0d5dd", boxstyle="round,pad=0.16"),
        )


def _capacity_utilization_heatmap(ax: plt.Axes, utilization: pd.DataFrame, title: str) -> None:
    required = {"capacity_mode", "resource_type", "utilization_rate_regular"}
    if utilization.empty or not required.issubset(utilization.columns):
        _empty_axis(ax, "capacity utilization missing")
        ax.set_title(title, fontsize=8, loc="left")
        return
    data = utilization.copy()
    data["utilization_rate_regular"] = pd.to_numeric(data["utilization_rate_regular"], errors="coerce")
    data["used_hours"] = pd.to_numeric(data.get("used_hours", np.nan), errors="coerce")
    data["available_regular_hours"] = pd.to_numeric(data.get("available_regular_hours", np.nan), errors="coerce")
    data["overtime_hours"] = pd.to_numeric(data.get("overtime_hours", np.nan), errors="coerce")
    finite_util = data["utilization_rate_regular"].dropna()
    if not finite_util.empty and float(finite_util.abs().max()) <= 1e-12:
        _capacity_zero_status_panel(ax, data, title)
        return
    pivot = data.pivot_table(index="capacity_mode", columns="resource_type", values="utilization_rate_regular", aggfunc="mean")
    wanted_rows = [row for row in ["independent_capacity", "independent_capacity_total", "shared_capacity"] if row in pivot.index]
    wanted_cols = [col for col in ["machining", "grinding", "inspection", "assembly", "testing", "laser"] if col in pivot.columns]
    pivot = pivot.reindex(index=wanted_rows, columns=wanted_cols)
    if pivot.empty:
        _empty_axis(ax, "no utilization values")
        ax.set_title(title, fontsize=8, loc="left")
        return
    arr = pivot.to_numpy(dtype=float)
    im = ax.imshow(np.ma.masked_invalid(arr), aspect="auto", cmap=CMAP_GREEN, vmin=0, vmax=max(float(np.nanmax(arr)), 0.01))
    ax.set_title(title, fontsize=8, loc="left")
    ax.set_xticks(np.arange(len(wanted_cols)))
    ax.set_xticklabels([col[:4] for col in wanted_cols], rotation=0, fontsize=5.6)
    row_labels = {
        "independent_capacity": "indep.",
        "independent_capacity_total": "indep.",
        "shared_capacity": "shared",
    }
    ax.set_yticks(np.arange(len(wanted_rows)))
    ax.set_yticklabels([row_labels.get(row, row) for row in wanted_rows])
    ax.tick_params(length=0)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i, j]):
                ax.text(j, i, _format_rate(arr[i, j]), ha="center", va="center", fontsize=5.1, color="white" if arr[i, j] > 0.65 else PALETTE["ink"])
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Utilization", fontsize=5.7)
    cbar.ax.tick_params(labelsize=5.5, length=2)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _capacity_zero_status_panel(ax: plt.Axes, data: pd.DataFrame, title: str) -> None:
    ax.set_title(title.replace("utilization", "status"), fontsize=8, loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    shared = data[data["capacity_mode"] == "shared_capacity"].copy()
    if shared.empty:
        shared = data.copy()
    row_count = int(len(shared))
    resource_count = int(shared["resource_type"].nunique()) if "resource_type" in shared.columns else 0
    used_hours = float(shared["used_hours"].sum(skipna=True)) if "used_hours" in shared.columns else 0.0
    available_hours = float(shared["available_regular_hours"].sum(skipna=True)) if "available_regular_hours" in shared.columns else 0.0
    overtime_hours = float(shared["overtime_hours"].sum(skipna=True)) if "overtime_hours" in shared.columns else 0.0
    stats = [
        ("Shared rows", f"{row_count:,}", PALETTE["blue"]),
        ("Resources", f"{resource_count}", PALETTE["teal"]),
        ("Route load", f"{used_hours:.1f} h", PALETTE["orange"]),
        ("Overtime", f"{overtime_hours:.1f} h", PALETTE["red"]),
    ]
    for idx, (label, value, color) in enumerate(stats):
        x0 = 0.02 + idx * 0.24
        rect = plt.Rectangle((x0, 0.56), 0.21, 0.30, facecolor="#ffffff", edgecolor="#d0d5dd", linewidth=0.8)
        ax.add_patch(rect)
        ax.scatter([x0 + 0.035], [0.76], s=34, color=color, zorder=3)
        ax.text(x0 + 0.065, 0.77, value, fontsize=7.3, fontweight="bold", color=PALETTE["ink"], va="center")
        ax.text(x0 + 0.025, 0.62, label, fontsize=5.8, color="#667085", va="center")
    note = (
        "Coupling rows are present, but this mini instance selects no\n"
        "shared-resource route load; shared capacity is non-binding."
    )
    ax.text(
        0.02,
        0.38,
        note,
        fontsize=6.1,
        color="#344054",
        va="top",
        bbox=dict(facecolor="#f8fafc", edgecolor="#d0d5dd", boxstyle="round,pad=0.22"),
    )
    ax.text(0.02, 0.11, f"Available regular capacity represented: {_compact_number(available_hours)} h", fontsize=5.9, color="#667085")


def _objective_range_inset(ax: plt.Axes, payoff: pd.DataFrame, x_col: str, y_col: str) -> None:
    if payoff.empty:
        return
    inset = ax.inset_axes([0.56, 0.62, 0.38, 0.26])
    values = [
        pd.to_numeric(payoff[x_col], errors="coerce").max() - pd.to_numeric(payoff[x_col], errors="coerce").min(),
        pd.to_numeric(payoff[y_col], errors="coerce").max() - pd.to_numeric(payoff[y_col], errors="coerce").min(),
    ]
    inset.barh([0, 1], values, color=[PALETTE["blue"], PALETTE["red"]], alpha=0.75)
    inset.set_yticks([0, 1])
    inset.set_yticklabels(["x range", "y range"], fontsize=4.8)
    inset.tick_params(axis="x", labelsize=4.8, length=2)
    inset.set_title("Payoff range", fontsize=5.2, loc="left")
    inset.grid(True, axis="x", color=PALETTE["grid"], linewidth=0.4)


def _candidate_pool_inset(ax: plt.Axes, feasible: pd.DataFrame) -> None:
    required = {"allowed_route_count", "allowed_pair_count", "iteration"}
    if feasible.empty or not required.issubset(feasible.columns):
        return
    inset = ax.inset_axes([0.06, 0.08, 0.34, 0.28])
    inset.patch.set_facecolor("white")
    inset.patch.set_alpha(0.94)
    pooled = feasible.copy()
    for column in ["allowed_route_count", "allowed_pair_count", "iteration"]:
        pooled[column] = pd.to_numeric(pooled[column], errors="coerce")
    pooled = pooled.dropna(subset=["iteration"])
    if pooled.empty:
        return
    summary = pooled.groupby("iteration", as_index=False)[["allowed_route_count", "allowed_pair_count"]].mean()
    inset.plot(summary["iteration"], summary["allowed_route_count"], color=PALETTE["teal"], linewidth=0.8, label="routes")
    inset.plot(summary["iteration"], summary["allowed_pair_count"], color=PALETTE["purple"], linewidth=0.8, label="pairs")
    inset.set_title("Open candidate pool", fontsize=5.2, loc="left")
    inset.tick_params(labelsize=4.8, length=2)
    inset.grid(True, color=PALETTE["grid"], linewidth=0.4)
    inset.legend(fontsize=4.6, loc="upper left")


def _shorten_ablation_labels(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "experiment_id" not in frame.columns:
        return frame.copy()
    mapping = {
        "ablation_no_stochasticity": "no stochasticity",
        "ablation_no_cvar": "no CVaR",
        "ablation_no_selective_assembly": "no selective\nassembly",
        "ablation_no_pareto": "no Pareto",
        "ablation_matheuristic_approximation": "matheuristic\napprox.",
    }
    data = frame.copy()
    data["display_id"] = data["experiment_id"].astype(str).map(mapping).fillna(data["experiment_id"].astype(str))
    return data


def _empty_axis(ax: plt.Axes, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color="#667085")
    ax.set_xticks([])
    ax.set_yticks([])


def _safe_annotation(ax: plt.Axes, text: str, x: float, y: float, xytext: tuple[int, int], color: str) -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=xytext,
        textcoords="offset points",
        fontsize=5.4,
        color=color,
        ha="left" if xytext[0] >= 0 else "right",
        va="bottom" if xytext[1] >= 0 else "top",
        arrowprops=dict(arrowstyle="-", color=color, linewidth=0.55, shrinkA=2.0, shrinkB=2.0),
        bbox=dict(facecolor="white", edgecolor=color, boxstyle="round,pad=0.12", linewidth=0.45, alpha=0.92),
        zorder=6,
    )


def _quiet_grid(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=PALETTE["grid"], linewidth=0.6, alpha=0.75)


def _route_mix(snapshot: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for stage, key in [("Stage3", "stage3_selected_routes"), ("Stage4", "stage4_selected_routes"), ("Stage5", "stage5_selected_routes"), ("Stage6", "stage6_selected_routes")]:
        frame = _csv(snapshot, key)
        if frame.empty or "route_id" not in frame.columns:
            continue
        counts = frame["route_id"].dropna().astype(str).value_counts()
        for route_id, count in counts.items():
            rows.append({"stage": stage, "route_id": route_id, "count": int(count)})
    return pd.DataFrame(rows)


def _csv(snapshot: Dict[str, Any], key: str) -> pd.DataFrame:
    value = snapshot.get("csv", {}).get(key)
    return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _label(value: str) -> str:
    return value.replace("_", " ").capitalize()


def _compact_number(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M".rstrip("0").rstrip(".")
    if abs_value >= 1_000:
        return f"{value / 1_000:.0f}k"
    if abs_value >= 10:
        return f"{value:.0f}"
    if abs_value >= 1:
        return f"{value:.2g}"
    if abs_value >= 1e-6:
        return f"{value:.2g}"
    return "0"


def _safe_median_repair_rate(top5: pd.DataFrame) -> float:
    if top5.empty or not {"repair_solves", "feasible_repair_solves"}.issubset(top5.columns):
        return np.nan
    repair = pd.to_numeric(top5["repair_solves"], errors="coerce").replace(0, np.nan)
    feasible = pd.to_numeric(top5["feasible_repair_solves"], errors="coerce")
    return float((feasible / repair).median(skipna=True))


def _format_rate(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{100 * value:.1f}%"


def _format_nonzero_rate(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    if value > 0 and 100 * value < 0.1:
        return "<0.1%"
    return _format_rate(value)
