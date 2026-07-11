"""Export manually edited Stage 11 draw.io schematics to manuscript formats."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
mpl_config = ROOT / "data" / "processed" / "stage11" / "matplotlib_config"
mpl_config.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))

from src.stage11_paper_artifacts.drawio_xml_exporter import export_drawio_xml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("data/results/stage11/figures"),
        help="Directory containing Stage 11 draw.io figures.",
    )
    parser.add_argument(
        "--figure",
        action="append",
        default=["F1_model_architecture", "F2_data_to_model_pipeline"],
        help="Figure stem to export. Can be repeated.",
    )
    parser.add_argument("--formats", nargs="+", default=["png", "svg", "pdf"], help="Output formats.")
    parser.add_argument("--dpi", type=int, default=600, help="Raster export DPI.")
    args = parser.parse_args()

    exported = {}
    for figure in args.figure:
        drawio_path = args.figures_dir / f"{figure}.drawio"
        if not drawio_path.exists():
            raise FileNotFoundError(drawio_path)
        exported[figure] = {fmt: str(path) for fmt, path in export_drawio_xml(drawio_path, args.formats, args.dpi).items()}

    for figure, paths in exported.items():
        print(f"{figure}:")
        for fmt, path in paths.items():
            print(f"  {fmt}: {path}")


if __name__ == "__main__":
    main()
