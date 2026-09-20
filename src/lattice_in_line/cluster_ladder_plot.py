"""Plot static clusters as a quasi-1D ladder in their MPS line ordering.

``cluster_generator.py`` builds and plots lattices from Bravais geometry; the
heavy-hex patches are static edge tables without generated coordinates, and
what one wants to see for them is not the physical layout but the *line
layout*: sites placed along the MPS ordering, wrapped into a few ladder legs
(four legs for the heavy-hex patches, whose optimized cutwidth is 4), with the
site enumeration and the edge structure (short arcs = good ordering) visible.

Usage::

    python -m lattice_in_line.cluster_ladder_plot                 # all heavy-hex
    python -m lattice_in_line.cluster_ladder_plot heavyHex35 \
        --ordering permutations_sat_cw.py --legs 4

Each node shows the MPS position (large, the enumeration used by the ordering
file) and the original patch label (small, top left).  Edges are colored by the
layers from ``CLUSTER_EDGE_LAYERS`` when the cluster defines them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .cluster_edges import CLUSTER_EDGES, CLUSTER_EDGE_LAYERS
from .cluster_graphs import _layout_stats, cluster_ordering_permutation

Edge = Tuple[int, int]

LAYER_COLORS = ("#d62728", "#1f77b4", "#2ca02c")  # bond-color layers 1 / 2 / 3
PLAIN_COLOR = "#607080"


def ladder_positions(
    n_sites: int, legs: int, *, snake: bool = False
) -> List[Tuple[float, float]]:
    """Map MPS position p to (x, y); leg 0 is the top row, columns advance in x."""
    positions = []
    for p in range(n_sites):
        column, leg = divmod(p, legs)
        if snake and column % 2:
            leg = legs - 1 - leg
        positions.append((float(column), float(-leg)))
    return positions


def edges_in_mps_order(
    name: str, permutation: Sequence[int]
) -> Tuple[List[Edge], List[int]]:
    """Return cluster edges relabeled to MPS positions plus a layer id per edge."""
    layer_of: Dict[Tuple[int, int], int] = {}
    for layer_index, layer in enumerate(CLUSTER_EDGE_LAYERS.get(name, [])):
        for i, j in layer:
            layer_of[tuple(sorted((i, j)))] = layer_index
    edges = []
    layers = []
    for i, j in CLUSTER_EDGES[name]:
        edges.append((permutation[i], permutation[j]))
        layers.append(layer_of.get(tuple(sorted((i, j))), -1))
    return edges, layers


def default_output_stem(stem: str) -> Path:
    """Bare stems go to the repo-level plots/ folder, like cluster_generator."""
    output_path = Path(stem)
    if output_path.is_absolute() or output_path.parent != Path("."):
        return output_path
    root = Path(__file__).resolve()
    for ancestor in root.parents:
        if (ancestor / "pyproject.toml").is_file():
            return ancestor / "plots" / output_path
    return Path.cwd() / output_path


def plot_cluster_ladder(
    name: str,
    *,
    ordering: str = "permutations_sat_cw.py",
    legs: int = 4,
    snake: bool = False,
    show_layers: bool = True,
    output_stem: str | None = None,
    dpi: int = 300,
) -> Tuple[Path, Path]:
    if name not in CLUSTER_EDGES:
        raise ValueError(
            f"Unknown cluster {name!r}. Available: {', '.join(sorted(CLUSTER_EDGES))}"
        )
    n_sites = max(max(edge) for edge in CLUSTER_EDGES[name]) + 1
    permutation = cluster_ordering_permutation(name, ordering=ordering)
    original_of = {permutation[i]: i for i in range(n_sites)}
    edges, edge_layers = edges_in_mps_order(name, permutation)
    bandwidth, avg_range, cutwidth = _layout_stats(edges)
    positions = ladder_positions(n_sites, legs, snake=snake)

    stem = default_output_stem(output_stem or f"{name}_ladder_{Path(ordering).stem}")
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
    from matplotlib.patches import FancyArrowPatch

    columns = (n_sites + legs - 1) // legs
    fig, ax = plt.subplots(figsize=(1.1 * columns + 1.6, 1.0 * legs + 1.8))

    has_layers = show_layers and name in CLUSTER_EDGE_LAYERS
    for (a, b), layer in zip(edges, edge_layers):
        (x1, y1), (x2, y2) = positions[a], positions[b]
        color = LAYER_COLORS[layer] if has_layers and layer >= 0 else PLAIN_COLOR
        manhattan = abs(x1 - x2) + abs(y1 - y2)
        # straight lines for ladder-adjacent sites, arcs for everything longer
        rad = 0.0 if manhattan <= 1.0 else (0.25 if y1 == y2 else 0.30)
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-",
                color=color,
                linewidth=1.6,
                alpha=0.9,
                zorder=1,
                shrinkA=9.5,
                shrinkB=9.5,
            )
        )

    xs = [x for x, _ in positions]
    ys = [y for _, y in positions]
    ax.scatter(
        xs, ys, s=560, facecolor="white", edgecolor="black", linewidth=1.1, zorder=2
    )
    for p, (x, y) in enumerate(positions):
        ax.annotate(
            str(p), (x, y), ha="center", va="center", fontsize=9, zorder=3
        )
        ax.annotate(
            str(original_of[p]),
            (x, y),
            xytext=(-11, 11),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=6.5,
            color="#996515",
            zorder=3,
        )

    handles = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor="black", label="MPS position (large) / original site (small)"),
    ]
    if has_layers:
        handles += [
            Line2D([], [], color=LAYER_COLORS[k], linewidth=2, label=f"layer {k + 1}")
            for k in range(3)
        ]
    ax.legend(handles=handles, loc="upper center", fontsize=8,
              bbox_to_anchor=(0.5, -0.06), ncol=len(handles), frameon=False)

    ax.set_title(
        f"{name} as {legs}-leg ladder | ordering={Path(ordering).name} | "
        f"bandwidth={bandwidth}, cutwidth={cutwidth}, avg range={avg_range:.2f}",
        fontsize=10,
    )
    ax.set_xlim(-0.8, columns - 0.2)
    ax.set_ylim(-(legs - 1) - 0.9, 0.9)
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
    heavy_hex = sorted(n for n in CLUSTER_EDGES if n.startswith("heavyHex"))
    parser.add_argument(
        "clusters", nargs="*", default=heavy_hex,
        help=f"clusters to plot (default: {' '.join(heavy_hex)})",
    )
    parser.add_argument("--ordering", default="permutations_sat_cw.py",
                        help="ordering file or method accepted by cluster_ordering_permutation")
    parser.add_argument("--legs", type=int, default=4, help="ladder legs (default: 4)")
    parser.add_argument("--snake", action="store_true",
                        help="boustrophedon columns instead of top-to-bottom")
    parser.add_argument("--no-layers", action="store_true",
                        help="single edge color instead of per-layer colors")
    parser.add_argument("--output-stem", default=None,
                        help="output stem for a single cluster (default: <name>_ladder_<ordering>)")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.legs < 1:
        raise ValueError("--legs must be positive")
    if args.output_stem is not None and len(args.clusters) != 1:
        raise ValueError("--output-stem requires exactly one cluster")
    for name in args.clusters:
        png_path, pdf_path = plot_cluster_ladder(
            name,
            ordering=args.ordering,
            legs=args.legs,
            snake=args.snake,
            show_layers=not args.no_layers,
            output_stem=args.output_stem,
            dpi=args.dpi,
        )
        print(f"{name}: wrote {png_path} and {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
