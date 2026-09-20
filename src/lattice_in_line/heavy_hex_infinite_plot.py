"""Sketch of the infinite heavy-hex ladder as a few translated unit cells.

The infinite model in ``unit_cell/heavy_hex_ladder.py`` is a 9-column heavy-hex
strip that repeats every two qubit rows: one 23-site cell holds two connector
qubits entering from the previous cell, a 9-site rail, three connectors, and
a second 9-site rail; its last two edges leave into the next cell.  This module
draws ``--cells`` consecutive copies in the ``heavy_hex_generator`` sketch
style (rows at even lines, connectors between them), greys out the first and
last copies so the middle cell stands out as the representative one, and draws
the inter-cell bonds dashed so they are distinguishable from the intra-cell
bonds.  Dangling dashed stubs at both ends mark the continuation to infinity.

Usage::

    python -m lattice_in_line.heavy_hex_infinite_plot
    python -m lattice_in_line.heavy_hex_infinite_plot --cells 5 --ordering natural --horizontal

Plots follow the conventions of ``cluster_generator.py`` (optional matplotlib,
bare stems land in the repo-level ``plots/`` folder, PNG and PDF are written).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .cluster_ladder_plot import LAYER_COLORS, default_output_stem
from .unit_cell.heavy_hex_ladder import (
    NATURAL_EDGES,
    NATURAL_EDGE_LAYERS,
    UNIT_CELL_SITES,
    build,
)

# Geometry of one cell in natural site labels, (x = column, y = line within the
# cell); the cell period along y is CELL_PERIOD.  Connectors 0 and 1 sit at
# columns 2 and 6 above the first rail, connectors 11..13 at columns 0/4/8
# between the rails, and the rails occupy columns 0..8.
CELL_PERIOD = 4
NATURAL_COORDS: Tuple[Tuple[int, int], ...] = (
    (2, 0), (6, 0),                                   # 0, 1: connectors from the previous cell
    *((col, 1) for col in range(9)),                  # 2..10: first rail
    (0, 2), (4, 2), (8, 2),                           # 11..13: connectors between the rails
    *((col, 3) for col in range(9)),                  # 14..22: second rail
)
GREY = "#b8b8b8"
GREY_FACE = "#e6e6e6"
PLAIN_COLOR = "#607080"
STUB_LENGTH = 0.7


def site_position(cell: int, site: int) -> Tuple[float, float]:
    """Sketch coordinates of natural ``site`` in translated copy ``cell``."""
    x, y = NATURAL_COORDS[site]
    return float(x), float(y + CELL_PERIOD * cell)


def layer_of_edges() -> Dict[Tuple[int, int, int], int]:
    return {
        edge: number
        for number, layer in enumerate(NATURAL_EDGE_LAYERS)
        for edge in layer
    }


def plot_infinite_ladder(
    *,
    cells: int = 3,
    ordering: str = "optimized",
    show_layers: bool = True,
    horizontal: bool = False,
    output_stem: str | None = None,
    dpi: int = 300,
) -> Tuple[Path, Path]:
    """Draw ``cells`` copies of the unit cell; the outer copies are greyed."""
    if cells < 1:
        raise ValueError("cells must be positive")
    payload = build(ordering=ordering)
    labels = {int(site): position for site, position in payload["permutation"].items()}
    layer_lookup = layer_of_edges()
    middle = set(range(1, cells - 1)) if cells >= 3 else set(range(cells))

    stem = default_output_stem(
        output_stem or f"heavyHexInfinite{UNIT_CELL_SITES}_{cells}cells_{ordering}"
    )
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    try:
        import matplotlib
    except ImportError:
        print(
            "matplotlib not installed; skipping plot "
            "(pip install 'lattice_in_line[plot]')",
            file=sys.stderr,
        )
        return png_path, pdf_path
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    def place(cell: int, site: int) -> Tuple[float, float]:
        x, y = site_position(cell, site)
        # sketch convention: y grows downward on the page
        return (y, -x) if horizontal else (x, -y)

    def bond_color(edge: Tuple[int, int, int], cells_touched: Sequence[int]) -> str:
        if not any(cell in middle for cell in cells_touched):
            return GREY
        if show_layers:
            return LAYER_COLORS[layer_lookup[edge]]
        return PLAIN_COLOR

    height = CELL_PERIOD * cells
    width, tall = (0.62 * height + 2.4, 0.85 * 9 + 1.6) if horizontal else (0.85 * 9 + 1.4, 0.62 * height + 2.2)
    fig, ax = plt.subplots(figsize=(width, tall))

    for cell in range(cells):
        for edge in NATURAL_EDGES:
            i, j, offset = edge
            partner_cell = cell + offset
            start = place(cell, i)
            if partner_cell < cells:
                end = place(partner_cell, j)
                color = bond_color(edge, (cell, partner_cell))
            else:
                # bond leaving the last drawn cell: dashed stub towards +infinity
                full = place(partner_cell, j)
                end = tuple(s + STUB_LENGTH * (f - s) for s, f in zip(start, full))
                color = GREY
            ax.plot(
                [start[0], end[0]], [start[1], end[1]],
                color=color, linewidth=2.2 if offset else 2.0,
                linestyle=(0, (4, 3)) if offset else "-", zorder=1,
            )
    # bonds arriving from the cell before the first drawn one: stubs towards -infinity
    for i, j, offset in NATURAL_EDGES:
        if offset:
            target = place(0, j)
            source = place(-1, i)
            end = tuple(t + STUB_LENGTH * (s - t) for t, s in zip(target, source))
            ax.plot([target[0], end[0]], [target[1], end[1]], color=GREY,
                    linewidth=2.2, linestyle=(0, (4, 3)), zorder=1)

    for cell in range(cells):
        highlighted = cell in middle
        for site in range(UNIT_CELL_SITES):
            x, y = place(cell, site)
            connector = NATURAL_COORDS[site][1] in (0, 2)
            if highlighted:
                face = "#f2e8c9" if connector else "white"
                edge_color, text_color = "black", "black"
            else:
                face, edge_color, text_color = GREY_FACE, GREY, "#8a8a8a"
            ax.scatter([x], [y], s=470, facecolor=face, edgecolor=edge_color,
                       linewidth=1.0, zorder=2)
            ax.annotate(str(labels[site]), (x, y), ha="center", va="center",
                        fontsize=8, color=text_color, zorder=3)
        # cell tag relative to the representative cell
        tag = cell - (cells // 2)
        anchor = place(cell, 13)  # right-hand connector between the rails
        if horizontal:
            ax.annotate(f"cell {tag:+d}", (CELL_PERIOD * cell + 1.5, 0.85), ha="center",
                        va="bottom", fontsize=9, color="black" if highlighted else "#8a8a8a")
        else:
            ax.annotate(f"cell {tag:+d}", (anchor[0] + 0.9, anchor[1]), ha="left",
                        va="center", fontsize=9, color="black" if highlighted else "#8a8a8a")

    handles = [
        Line2D([], [], color=PLAIN_COLOR if not show_layers else "black", linewidth=2,
               label="intra-cell bond"),
        Line2D([], [], color=PLAIN_COLOR if not show_layers else "black", linewidth=2.2,
               linestyle=(0, (4, 3)), label="inter-cell bond"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=GREY_FACE,
               markeredgecolor=GREY, label="translated copy"),
    ]
    if show_layers:
        handles += [
            Line2D([], [], color=LAYER_COLORS[k], linewidth=2, label=f"layer {k + 1}")
            for k in range(3)
        ]
    ax.legend(handles=handles, loc="upper center", fontsize=8,
              bbox_to_anchor=(0.5, -0.03), ncol=3, frameon=False)
    ax.set_title(
        f"infinite heavy-hex ladder | {UNIT_CELL_SITES}-site unit cell x {cells} | "
        f"ordering={ordering} (labels = MPS positions)",
        fontsize=10,
    )
    margin = STUB_LENGTH + 0.8
    if horizontal:
        ax.set_xlim(-margin, height - 1 + margin)
        ax.set_ylim(-8.8, 1.7)
    else:
        ax.set_xlim(-1.4, 8 + 2.4)
        ax.set_ylim(-(height - 1) - margin, margin)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cells", type=int, default=3, help="unit cells to draw (default 3)")
    parser.add_argument("--ordering", choices=("optimized", "natural"), default="optimized",
                        help="site labels: MPS positions of this ordering")
    parser.add_argument("--no-layers", action="store_true",
                        help="single bond color instead of per-layer colors")
    parser.add_argument("--horizontal", action="store_true",
                        help="stack the cells left-to-right instead of top-to-bottom")
    parser.add_argument("--output-stem", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    png_path, pdf_path = plot_infinite_ladder(
        cells=args.cells,
        ordering=args.ordering,
        show_layers=not args.no_layers,
        horizontal=args.horizontal,
        output_stem=args.output_stem,
        dpi=args.dpi,
    )
    print(f"infinite ladder: wrote {png_path} and {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
