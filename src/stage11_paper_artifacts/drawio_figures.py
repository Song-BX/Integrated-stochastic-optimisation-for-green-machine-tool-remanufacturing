"""Draw.io-based manuscript schematic builders for Stage 11."""

from __future__ import annotations

import html
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from .config import Stage11Config
from .structures import PaperArtifactSpec, PaperFigure


PALETTE = {
    "blue": "#3f6f9f",
    "teal": "#4f9862",
    "orange": "#d9822b",
    "red": "#b94a48",
    "purple": "#8a6f8f",
    "neutral": "#667085",
    "ink": "#101828",
    "grid": "#e7eaf0",
}


def build_drawio_model_architecture(snapshot: Dict[str, Any], config: Stage11Config, output_dir: Path) -> PaperFigure:
    """Build Figure 1 as an editable draw.io schematic and exported figure."""

    artifact_id = "F1_model_architecture"
    page = _DrawioPage(width=1500, height=760)
    page.text("(a) Integrated stochastic optimization architecture", 45, 35, 780, 32, size=20, bold=True)
    page.text("Production decision chain", 65, 100, 360, 26, size=12, color="#475467", bold=True)
    page.text("Risk, quality and solution coupling", 65, 300, 430, 26, size=12, color="#475467", bold=True)
    page.text("Model contents used by the manuscript evidence", 65, 525, 520, 26, size=12, color="#475467", bold=True)

    boxes = [
        ("condition", "Condition-aware<br>returned cores<br>quality states", 70, 135, 285, 120, PALETTE["blue"]),
        ("first", "First-stage<br>acceptance and<br>base procurement", 420, 135, 285, 120, PALETTE["teal"]),
        ("scenario", "Scenario SAA<br>quality, demand<br>route outcomes", 770, 135, 285, 120, PALETTE["orange"]),
        ("second", "Second-stage<br>routing, inventory<br>backlog, overtime", 1120, 135, 300, 120, PALETTE["purple"]),
        ("risk", "Reliability screen<br>chance constraints<br>CVaR tail loss", 420, 335, 285, 112, PALETTE["red"]),
        ("assembly", "Selective assembly<br>compatibility pairs<br>dimension chain", 770, 335, 285, 112, PALETTE["red"]),
        ("solver", "Pareto + ALNS<br>epsilon grid<br>MILP repair", 1120, 335, 300, 112, PALETTE["neutral"]),
    ]
    for cell_id, label, x, y, w, h, color in boxes:
        page.rounded_box(cell_id, label, x, y, w, h, color)

    for source, target in [("condition", "first"), ("first", "scenario"), ("scenario", "second"), ("risk", "assembly"), ("assembly", "solver")]:
        page.edge(source, target)
    page.elbow("scenario", "risk", [(912, 285), (562, 285)])
    page.elbow("second", "solver", [(1270, 285)])

    badges = [
        ("Decision variables", "accept, route, procure,<br>inventory, assemble", 90, 565, PALETTE["teal"]),
        ("Uncertainty", "quality, demand,<br>route outcomes", 425, 565, PALETTE["orange"]),
        ("Constraints", "capacity, BOM,<br>reliability, pairs", 760, 565, PALETTE["red"]),
        ("Objectives", "economic risk,<br>carbon, assembly loss", 1095, 565, PALETTE["purple"]),
    ]
    for idx, (title, body, x, y, color) in enumerate(badges):
        page.badge(f"badge{idx}", title, body, x, y, 250, 90, color)

    page.text(
        "Claim: route choice, reliability risk, selective assembly and solution strategy form one coupled stochastic production system.",
        60,
        700,
        1340,
        28,
        size=12,
        color="#344054",
    )

    source_data = pd.DataFrame(boxes, columns=["cell_id", "label", "x", "y", "w", "h", "color"])
    paths, warnings = _write_and_export(
        page,
        artifact_id,
        output_dir,
        config,
        {
            "kind": "architecture",
            "title": "(a) Integrated stochastic optimization architecture",
            "boxes": boxes,
            "badges": badges,
            "claim": "Claim: route choice, reliability risk, selective assembly and solution strategy form one coupled stochastic production system.",
        },
    )
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
        warnings=warnings,
    )
    return PaperFigure(spec=spec, source_data=source_data, output_paths=paths)


def build_drawio_data_pipeline(snapshot: Dict[str, Any], config: Stage11Config, output_dir: Path) -> PaperFigure:
    """Build Figure 2 as an editable draw.io schematic and exported figure."""

    artifact_id = "F2_data_to_model_pipeline"
    catalogue = snapshot.get("stage1_catalogue", pd.DataFrame())
    total_files = int(len(catalogue)) if isinstance(catalogue, pd.DataFrame) and not catalogue.empty else 28
    total_rows = int(pd.to_numeric(catalogue.get("row_count"), errors="coerce").sum()) if isinstance(catalogue, pd.DataFrame) and "row_count" in catalogue.columns else 273013
    total_mb = float(pd.to_numeric(catalogue.get("size_mb"), errors="coerce").sum()) if isinstance(catalogue, pd.DataFrame) and "size_mb" in catalogue.columns else np.nan
    summary = snapshot.get("stage1_validation", {}).get("summary", {})
    baseline_rows = len(_csv(snapshot, "stage9_baseline"))
    sensitivity_rows = len(_csv(snapshot, "stage9_sensitivity"))

    page = _DrawioPage(width=1500, height=620)
    page.text("(b) Data-to-model pipeline", 45, 30, 520, 34, size=20, bold=True)
    nodes = [
        ("raw_gate", f"Raw CSVs + Stage 1<br>{total_files} files / {_compact_number(total_rows)} rows<br>validation failed={summary.get('failed', 'n/a')}", 70, 125, 295, 140, PALETTE["blue"]),
        ("model", "Stage 3-6<br>deterministic, SAA,<br>CVaR + assembly MILPs", 430, 125, 295, 140, PALETTE["orange"]),
        ("experiments", "Stage 7-10<br>Pareto, ALNS,<br>experiments + strengthening", 790, 125, 295, 140, PALETTE["purple"]),
        ("artifacts", "Stage 11-12<br>paper artifacts,<br>audits and evidence pack", 1150, 125, 295, 140, PALETTE["neutral"]),
    ]
    for cell_id, label, x, y, w, h, color in nodes:
        page.rounded_box(cell_id, label, x, y, w, h, color)
    for source, target in [("raw_gate", "model"), ("model", "experiments"), ("experiments", "artifacts")]:
        page.edge(source, target)

    callouts = [
        ("Input scale", f"{_compact_number(total_mb)} MB<br>pass {summary.get('passed', 'n/a')} / warn {summary.get('warning', 'n/a')}", 85, 350, PALETTE["blue"]),
        ("Model scope", "multi-period, multi-route<br>chance + CVaR + pairs", 445, 350, PALETTE["orange"]),
        ("Evidence rows", f"baseline {baseline_rows}<br>sensitivity {sensitivity_rows}", 805, 350, PALETTE["purple"]),
        ("Paper package", "7 tables + 8 figures<br>source + reality audits", 1165, 350, PALETTE["neutral"]),
    ]
    for idx, (title, body, x, y, color) in enumerate(callouts):
        page.badge(f"callout{idx}", title, body, x, y, 260, 95, color)

    page.text(
        "Traceability is preserved from raw CSV validation to model runs, experiment collection, paper figures and reviewer-facing audits.",
        85,
        575,
        1320,
        26,
        size=12,
        color="#344054",
    )

    source_data = pd.DataFrame(nodes, columns=["cell_id", "label", "x", "y", "w", "h", "color"])
    paths, warnings = _write_and_export(
        page,
        artifact_id,
        output_dir,
        config,
        {
            "kind": "pipeline",
            "title": "(b) Data-to-model pipeline",
            "nodes": nodes,
            "callouts": callouts,
            "claim": "Traceability is preserved from raw CSV validation to model runs, experiment collection, paper figures and reviewer-facing audits.",
        },
    )
    spec = PaperArtifactSpec(
        artifact_id=artifact_id,
        artifact_type="figure",
        title="Data-to-model pipeline",
        claim="The experiment evidence is traceable from raw CSV validation to manuscript-ready tables and figures.",
        source_files=[str(config.stage1_report)],
        output_files=[str(path) for path in paths.values()],
        warnings=warnings,
    )
    return PaperFigure(spec=spec, source_data=source_data, output_paths=paths)


class _DrawioPage:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._next_id = 2
        self.root = ET.Element("root")
        ET.SubElement(self.root, "mxCell", id="0")
        ET.SubElement(self.root, "mxCell", id="1", parent="0")

    def rounded_box(self, cell_id: str, label: str, x: float, y: float, w: float, h: float, color: str) -> None:
        style = (
            "rounded=1;whiteSpace=wrap;html=1;arcSize=8;strokeWidth=2;"
            f"strokeColor={color};fillColor=#ffffff;fontColor={PALETTE['ink']};"
            "fontSize=15;spacing=8;"
        )
        cell = ET.SubElement(self.root, "mxCell", id=cell_id, value=label, style=style, vertex="1", parent="1")
        ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), **{"as": "geometry"})

    def badge(self, cell_id: str, title: str, body: str, x: float, y: float, w: float, h: float, color: str) -> None:
        label = f"<font color='{color}'><b>{html.escape(title)}</b></font><br><font color='#344054'>{body}</font>"
        style = (
            "rounded=1;whiteSpace=wrap;html=1;arcSize=8;strokeWidth=1.6;"
            f"strokeColor={color};fillColor=#ffffff;fontColor={PALETTE['ink']};"
            "fontSize=12;align=left;verticalAlign=middle;spacingLeft=12;spacingRight=10;"
        )
        cell = ET.SubElement(self.root, "mxCell", id=cell_id, value=label, style=style, vertex="1", parent="1")
        ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), **{"as": "geometry"})

    def text(self, label: str, x: float, y: float, w: float, h: float, size: int = 12, color: str = "#101828", bold: bool = False) -> None:
        cell_id = self._id("text")
        value = f"<b>{html.escape(label)}</b>" if bold else html.escape(label)
        style = f"text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontSize={size};fontColor={color};"
        cell = ET.SubElement(self.root, "mxCell", id=cell_id, value=value, style=style, vertex="1", parent="1")
        ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), **{"as": "geometry"})

    def edge(self, source: str, target: str) -> None:
        cell_id = self._id("edge")
        style = "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#344054;strokeWidth=2;"
        cell = ET.SubElement(self.root, "mxCell", id=cell_id, value="", style=style, edge="1", parent="1", source=source, target=target)
        ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})

    def elbow(self, source: str, target: str, points: Iterable[Tuple[float, float]]) -> None:
        cell_id = self._id("edge")
        style = "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor=#344054;strokeWidth=2;"
        cell = ET.SubElement(self.root, "mxCell", id=cell_id, value="", style=style, edge="1", parent="1", source=source, target=target)
        geometry = ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
        array = ET.SubElement(geometry, "Array", **{"as": "points"})
        for x, y in points:
            ET.SubElement(array, "mxPoint", x=str(x), y=str(y))

    def to_mxfile(self, name: str) -> str:
        graph = ET.Element(
            "mxGraphModel",
            dx="1500",
            dy="760",
            grid="1",
            gridSize="10",
            guides="1",
            tooltips="1",
            connect="1",
            arrows="1",
            fold="1",
            page="1",
            pageScale="1",
            pageWidth=str(self.width),
            pageHeight=str(self.height),
            math="0",
            shadow="0",
        )
        graph.append(self.root)
        mxfile = ET.Element(
            "mxfile",
            host="app.diagrams.net",
            modified="2026-06-23T00:00:00.000Z",
            agent="Codex",
            version="24.7.17",
            type="device",
            compressed="false",
        )
        diagram = ET.SubElement(mxfile, "diagram", id=name, name="Page-1")
        diagram.append(graph)
        return ET.tostring(mxfile, encoding="unicode")

    def _id(self, prefix: str) -> str:
        cell_id = f"{prefix}{self._next_id}"
        self._next_id += 1
        return cell_id


def _write_and_export(
    page: _DrawioPage,
    artifact_id: str,
    output_dir: Path,
    config: Stage11Config,
    render_spec: Dict[str, Any],
) -> Tuple[Dict[str, Path], List[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    drawio_path = output_dir / f"{artifact_id}.drawio"

    paths: Dict[str, Path] = {"drawio": drawio_path}
    warnings: List[str] = []
    if drawio_path.exists():
        warnings.append("existing_drawio_source_preserved")
        for ext in config.figure_formats:
            existing_export = output_dir / f"{artifact_id}.{ext}"
            if existing_export.exists():
                paths[ext] = existing_export
            else:
                warnings.append(f"existing_drawio_export_missing_{ext}")
        return paths, warnings
    else:
        drawio_path.write_text(page.to_mxfile(artifact_id), encoding="utf-8")
    _render_exports(render_spec, artifact_id, output_dir, config, paths, warnings)

    # The editable .drawio source is the canonical schematic source. Automated
    # draw.io/Electron export is opt-in because some Windows installations open
    # crash dialogs in headless CLI mode; the Matplotlib renderer above uses the
    # same layout spec and keeps the build non-interactive.
    if not _drawio_cli_enabled():
        return paths, warnings

    executable = _find_drawio_executable()
    if executable is None:
        warnings.append("drawio_executable_not_found_using_layout_renderer")
        return paths, warnings

    for ext in config.figure_formats:
        out_path = output_dir / f"{artifact_id}.{ext}"
        cmd = [
            str(executable),
            "--disable-gpu",
            f"--user-data-dir={output_dir.parent.parent / 'processed' / 'stage11' / 'drawio_user_data'}",
            "--export",
            "--format",
            ext,
            "--output",
            str(out_path),
            str(drawio_path),
        ]
        try:
            subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        except subprocess.TimeoutExpired:
            warnings.append(f"drawio_export_timeout_{ext}_kept_layout_renderer")
            continue
        if _wait_for_file(out_path):
            paths[ext] = out_path
    return paths, warnings


def _render_exports(
    render_spec: Dict[str, Any],
    artifact_id: str,
    output_dir: Path,
    config: Stage11Config,
    paths: Dict[str, Path],
    warnings: List[str],
) -> None:
    for ext in config.figure_formats:
        out_path = output_dir / f"{artifact_id}.{ext}"
        try:
            _render_layout(render_spec, out_path, ext, config.dpi)
            paths[ext] = out_path
        except Exception as exc:  # pragma: no cover - reported in Stage 11 checks.
            warnings.append(f"layout_renderer_failed_{ext}:{type(exc).__name__}")


def _render_layout(render_spec: Dict[str, Any], out_path: Path, ext: str, dpi: int) -> None:
    if render_spec["kind"] == "architecture":
        width, height = 1500, 760
        figsize = (8.3, 4.2)
    else:
        width, height = 1500, 620
        figsize = (8.3, 3.45)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")

    ax.text(45, 52, render_spec["title"], ha="left", va="center", fontsize=8.5, weight="bold", color=PALETTE["ink"])
    if render_spec["kind"] == "architecture":
        _render_architecture(ax, render_spec)
    else:
        _render_pipeline(ax, render_spec)
    fig.savefig(out_path, format=ext, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _render_architecture(ax: Any, render_spec: Dict[str, Any]) -> None:
    boxes = render_spec["boxes"]
    box_map = {cell_id: (x, y, w, h) for cell_id, _label, x, y, w, h, _color in boxes}
    _row_label(ax, 65, 110, "Production decision chain")
    _row_label(ax, 65, 310, "Risk, quality and solution coupling")
    _row_label(ax, 65, 535, "Model contents used by the manuscript evidence")
    for _cell_id, label, x, y, w, h, color in boxes:
        _box(ax, x, y, w, h, _plain(label), color, size=7.2)

    for source, target in [("condition", "first"), ("first", "scenario"), ("scenario", "second"), ("risk", "assembly"), ("assembly", "solver")]:
        _arrow_between(ax, box_map[source], box_map[target])
    _poly_arrow(ax, [_edge_point(box_map["scenario"], "bottom"), (912, 285), (562, 285), _edge_point(box_map["risk"], "top")])
    _poly_arrow(ax, [_edge_point(box_map["second"], "bottom"), (1270, 285), _edge_point(box_map["solver"], "top")])

    for title, body, x, y, color in render_spec["badges"]:
        _badge(ax, x, y, 250, 90, title, _plain(body), color)

    ax.text(60, 714, render_spec["claim"], ha="left", va="center", fontsize=5.8, color="#344054")


def _render_pipeline(ax: Any, render_spec: Dict[str, Any]) -> None:
    nodes = render_spec["nodes"]
    box_map = {cell_id: (x, y, w, h) for cell_id, _label, x, y, w, h, _color in nodes}
    for _cell_id, label, x, y, w, h, color in nodes:
        _box(ax, x, y, w, h, _plain(label), color, size=7.0)
    for source, target in [("raw_gate", "model"), ("model", "experiments"), ("experiments", "artifacts")]:
        _arrow_between(ax, box_map[source], box_map[target])

    for title, body, x, y, color in render_spec["callouts"]:
        _badge(ax, x, y, 260, 95, title, _plain(body), color)

    ax.text(85, 585, render_spec["claim"], ha="left", va="center", fontsize=5.8, color="#344054")


def _row_label(ax: Any, x: float, y: float, label: str) -> None:
    ax.text(x, y, label, ha="left", va="center", fontsize=5.8, weight="bold", color="#475467")
    ax.plot([x, x + 1365], [y + 20, y + 20], color="#e7eaf0", linewidth=0.6, zorder=0)


def _box(ax: Any, x: float, y: float, w: float, h: float, label: str, color: str, size: float) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=8",
        linewidth=1.2,
        edgecolor=color,
        facecolor="white",
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=size, color=PALETTE["ink"], linespacing=1.35, zorder=3)


def _badge(ax: Any, x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=8",
        linewidth=0.9,
        edgecolor=color,
        facecolor="white",
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + 18, y + 30, title, ha="left", va="center", fontsize=6.1, weight="bold", color=color, zorder=3)
    ax.text(x + 18, y + 58, body, ha="left", va="center", fontsize=5.8, color="#344054", linespacing=1.25, zorder=3)


def _arrow_between(ax: Any, source: Tuple[float, float, float, float], target: Tuple[float, float, float, float]) -> None:
    start = _edge_point(source, "right")
    end = _edge_point(target, "left")
    _arrow(ax, start, end)


def _poly_arrow(ax: Any, points: Sequence[Tuple[float, float]]) -> None:
    if len(points) < 2:
        return
    xs, ys = zip(*points[:-1])
    ax.plot(xs, ys, color="#344054", linewidth=0.9, zorder=1)
    for p0, p1 in zip(points[1:-2], points[2:-1]):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#344054", linewidth=0.9, zorder=1)
    _arrow(ax, points[-2], points[-1])


def _arrow(ax: Any, start: Tuple[float, float], end: Tuple[float, float]) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=0.9,
        color="#344054",
        shrinkA=4,
        shrinkB=6,
        zorder=1,
    )
    ax.add_patch(arrow)


def _edge_point(box: Tuple[float, float, float, float], side: str) -> Tuple[float, float]:
    x, y, w, h = box
    if side == "left":
        return x, y + h / 2
    if side == "right":
        return x + w, y + h / 2
    if side == "top":
        return x + w / 2, y
    return x + w / 2, y + h


def _plain(label: str) -> str:
    return html.unescape(label).replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")


def _drawio_cli_enabled() -> bool:
    return False


def _find_drawio_executable() -> Path | None:
    candidates = [
        shutil.which("draw.io"),
        shutil.which("drawio"),
        shutil.which("diagrams.net"),
        r"C:\Program Files\draw.io\draw.io.exe",
        r"C:\Users\Song\AppData\Local\Programs\draw.io\draw.io.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return None


def _wait_for_file(path: Path, timeout_s: float = 20.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists() and path.stat().st_size > 1024:
            return True
        time.sleep(0.25)
    return path.exists() and path.stat().st_size > 1024


def _csv(snapshot: Dict[str, Any], key: str) -> pd.DataFrame:
    value = snapshot.get("csv", {}).get(key)
    return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame()


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
