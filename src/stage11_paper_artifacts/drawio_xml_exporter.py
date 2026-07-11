"""Lightweight renderer for manually edited Stage 11 draw.io schematics."""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch


def export_drawio_xml(drawio_path: Path, formats: Iterable[str] = ("png", "svg", "pdf"), dpi: int = 600) -> Dict[str, Path]:
    """Export an uncompressed draw.io XML file using a Matplotlib subset renderer.

    The renderer covers the simple shapes used by the manuscript schematics:
    rounded boxes, text labels, ellipses, arrow-shape vertices, and straight
    mxCell edges with source/target points.
    """

    root = ET.fromstring(drawio_path.read_text(encoding="utf-8"))
    graph = root.find(".//mxGraphModel")
    if graph is None:
        raise ValueError(f"No mxGraphModel found in {drawio_path}")

    width = float(graph.get("pageWidth", "1500"))
    height = float(graph.get("pageHeight", "760"))
    fig_width = 8.3
    fig_height = max(2.5, fig_width * height / width)

    outputs: Dict[str, Path] = {}
    for fmt in formats:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis("off")

        cells = [cell for cell in root.findall(".//mxCell") if cell.get("id") not in {"0", "1"}]
        for z, cell in enumerate(cells):
            if cell.get("vertex") == "1":
                _draw_vertex(ax, cell, z)
        for z, cell in enumerate(cells):
            if cell.get("edge") == "1":
                _draw_edge(ax, cell, z)

        out_path = drawio_path.with_suffix(f".{fmt}")
        fig.savefig(out_path, format=fmt, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
        outputs[fmt] = out_path
    return outputs


def _draw_vertex(ax, cell: ET.Element, z: int) -> None:
    geom = cell.find("mxGeometry")
    if geom is None:
        return
    x, y, w, h = _geometry(geom)
    style = _style(cell.get("style", ""))
    value = _plain_text(cell.get("value", ""))

    if "mxgraph.arrows2.arrow" in style.get("shape", ""):
        _draw_arrow_shape(ax, x, y, w, h, style, z)
        return

    is_text = "text" in style or (style.get("strokeColor") == "none" and style.get("fillColor") == "none")
    if not is_text:
        _draw_shape(ax, x, y, w, h, style, z)
    if value:
        _draw_text(ax, x, y, w, h, value, style, z)


def _draw_shape(ax, x: float, y: float, w: float, h: float, style: Dict[str, str], z: int) -> None:
    fill = _color(style.get("fillColor"), "#FFFFFF")
    stroke = _color(style.get("strokeColor"), "#344054")
    lw = float(style.get("strokeWidth", "1.2") or 1.2)

    if style.get("shape") == "ellipse" or "ellipse" in style:
        patch = Ellipse((x + w / 2, y + h / 2), w, h, facecolor=fill, edgecolor=stroke, linewidth=lw, zorder=z)
    else:
        rounding = max(2.0, min(w, h) * 0.06) if style.get("rounded") == "1" else 0.01
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.015,rounding_size={rounding}",
            facecolor=fill,
            edgecolor=stroke,
            linewidth=lw,
            zorder=z,
        )
    ax.add_patch(patch)


def _draw_text(ax, x: float, y: float, w: float, h: float, value: str, style: Dict[str, str], z: int) -> None:
    align = style.get("align", "center")
    valign = style.get("verticalAlign", "middle")
    size = _font_size(style.get("fontSize"))
    color = _color(style.get("fontColor"), "#1F2933")
    weight = "bold" if style.get("fontStyle") == "1" or "<b>" in html.unescape(style.get("_raw_value", "")) else "normal"

    if align == "left":
        tx = x + min(18, max(6, w * 0.06))
        ha = "left"
    elif align == "right":
        tx = x + w - min(18, max(6, w * 0.06))
        ha = "right"
    else:
        tx = x + w / 2
        ha = "center"

    if valign == "top":
        ty = y + min(16, max(5, h * 0.12))
        va = "top"
    elif valign == "bottom":
        ty = y + h - min(16, max(5, h * 0.12))
        va = "bottom"
    else:
        ty = y + h / 2
        va = "center"

    ax.text(tx, ty, value, ha=ha, va=va, fontsize=size, color=color, weight=weight, linespacing=1.18, zorder=z + 0.5)


def _draw_edge(ax, cell: ET.Element, z: int) -> None:
    geom = cell.find("mxGeometry")
    if geom is None:
        return
    points = []
    for point in geom.findall(".//mxPoint"):
        if "x" in point.attrib and "y" in point.attrib:
            points.append((float(point.get("x", "0")), float(point.get("y", "0"))))
    if len(points) < 2:
        return

    style = _style(cell.get("style", ""))
    color = _color(style.get("strokeColor"), "#344054")
    lw = float(style.get("strokeWidth", "1.2") or 1.2)
    dashed = style.get("dashed") == "1"
    line_style = (0, (4, 3)) if dashed else "solid"
    for start, end in zip(points[:-1], points[1:]):
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=lw,
            linestyle=line_style,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=z + 50,
        )
        ax.add_patch(arrow)


def _draw_arrow_shape(ax, x: float, y: float, w: float, h: float, style: Dict[str, str], z: int) -> None:
    color = _color(style.get("fillColor") or style.get("strokeColor"), "#153A5B")
    if style.get("direction") == "south":
        start = (x + w / 2, y)
        end = (x + w / 2, y + h)
    else:
        start = (x, y + h / 2)
        end = (x + w, y + h / 2)
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=max(8, min(w, h) * 0.9),
        linewidth=max(1.2, min(w, h) * 0.18),
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=z + 25,
    )
    ax.add_patch(arrow)


def _geometry(geom: ET.Element) -> Tuple[float, float, float, float]:
    return (
        float(geom.get("x", "0")),
        float(geom.get("y", "0")),
        float(geom.get("width", "0")),
        float(geom.get("height", "0")),
    )


def _style(style: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for part in style.split(";"):
        if not part:
            continue
        if "=" in part:
            key, value = part.split("=", 1)
            result[key] = value
        else:
            result[part] = "1"
    return result


def _plain_text(value: str) -> str:
    value = html.unescape(value or "")
    value = re.sub(r"<br\s*/?>", "\n", value, flags=re.IGNORECASE)
    value = re.sub(r"</p>\s*<p[^>]*>", "\n", value, flags=re.IGNORECASE)
    value = re.sub(r"<[^>]+>", "", value)
    value = html.unescape(value)
    return value.replace("\xa0", " ").strip()


def _color(value: str | None, default: str) -> str:
    if not value or value in {"none", "transparent"}:
        return default
    if value.startswith("#"):
        return value
    rgb = re.search(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", value)
    if rgb:
        return "#%02x%02x%02x" % tuple(int(part) for part in rgb.groups())
    return default


def _font_size(value: str | None) -> float:
    try:
        size = float(value or 16)
    except ValueError:
        size = 16
    return max(4.8, size * 0.38)
