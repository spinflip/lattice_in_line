"""Parametric generator for finite heavy-hex patches.

``cluster_edges.py`` stores finite heavy-hex patches as explicit edge tables.
This module generates the same patches -- identical site enumeration and edge
sets -- from three parameters, and extends the family to arbitrary widths on
the way to the 2D limit:

* ``n_rows``   : number of horizontal qubit rows (top to bottom),
* ``cols``     : sites in a long row (must be ``4*k + 1``; the devices use 9),
* ``top_short``/``bottom_short`` : whether the boundary rows are the short
  ``cols - 4``-site rows spanning columns ``2 .. cols - 3``.

Sites are enumerated row by row (left to right), with each row followed by its
downward connector qubits; connector columns alternate ``{2, 6, ...}`` and
``{0, 4, 8, ...}`` down the lattice, starting with the former when the top row
is short.  This reproduces every entry of ``CLUSTER_EDGES`` exactly.

Edge layers: the four known patches return the stored (published)
colorings from ``CLUSTER_EDGE_LAYERS`` verbatim.  Those four tables are
mutually inconsistent (heavyHex74 recolors 37 of the 56 edges it shares with
heavyHex51), so no parametric rule can reproduce all of them; for new patches
this module instead produces a deterministic proper 3-edge-coloring (Kempe
chains on the bipartite, degree<=3 graph) and validates the matching/cover
invariants.

Usage::

    python -m lattice_in_line.heavy_hex_generator heavyHex51 --plot both
    python -m lattice_in_line.heavy_hex_generator --rows 9 --plot sketch --print-edges

Plots follow the conventions of ``cluster_generator.py``: matplotlib is
optional, bare output stems land in the repo-level ``plots/`` folder, and both
``.png`` and ``.pdf`` are written.  ``--plot ladder`` draws the quasi-1D view
of ``cluster_ladder_plot``; ``--plot sketch`` draws the traditional heavy-hex
lattice picture.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .cluster_edges import CLUSTER_EDGES, CLUSTER_EDGE_LAYERS
from .cluster_generator import (
    print_compact_coords,
    print_compact_edges,
    print_edge_list,
)
from .cluster_graphs import _layout_stats, _resolve_ordering_permutation
from .cluster_ladder_plot import (
    LAYER_COLORS,
    default_output_stem,
    ladder_positions,
)

Edge = Tuple[int, int]

# (n_rows, cols, top_short, bottom_short) for the stored edge tables.
KNOWN_PATCHES: Dict[str, Tuple[int, int, bool, bool]] = {
    "heavyHex28": (3, 9, False, True),
    "heavyHex35": (4, 9, True, True),
    "heavyHex51": (5, 9, True, False),
    "heavyHex74": (7, 9, True, False),
    "heavyHex75": (5, 13, True, False),
    "heavyHex99": (5, 17, True, False),
    "heavyHex108": (7, 13, True, False),
    "heavyHex142": (7, 17, True, False),
    "heavyHex185": (9, 17, True, False),
    "heavyHex399": (13, 25, True, False),
}


@dataclass(frozen=True)
class HeavyHexPatch:
    n_rows: int
    cols: int
    top_short: bool
    bottom_short: bool
    n_sites: int
    edges: Tuple[Edge, ...]
    coords: Tuple[Tuple[float, float], ...]  # (x, y) per site, y grows downward
    kinds: Tuple[str, ...]  # "row" or "conn" per site
    name: str
    horizontal_edges: Tuple[Tuple[int, Tuple[Edge, ...]], ...] = field(repr=False, default=())
    vertical_edges: Tuple[Tuple[int, Tuple[Edge, ...]], ...] = field(repr=False, default=())

    @property
    def coordination_two_sites(self) -> Tuple[int, ...]:
        degree = [0] * self.n_sites
        for a, b in self.edges:
            degree[a] += 1
            degree[b] += 1
        return tuple(site for site, value in enumerate(degree) if value == 2)


def heavy_hex_patch(
    n_rows: int,
    *,
    cols: int = 9,
    top_short: bool = True,
    bottom_short: bool = False,
    name: str | None = None,
) -> HeavyHexPatch:
    """Build a heavy-hex patch with the canonical table enumeration."""
    if n_rows < 2:
        raise ValueError("A heavy-hex patch needs at least two rows")
    if cols < 5 or cols % 4 != 1:
        raise ValueError("cols must be 4*k + 1 with k >= 1 (the devices use 9)")
    if n_rows == 2 and top_short and bottom_short:
        raise ValueError("A two-row patch cannot have two short rows")
    if bottom_short:
        # A short row only attaches to the inner {2, 6, ...} connector set, so
        # the alternation phase constrains the row count (cf. heavyHex28/35).
        last_connector_row = n_rows - 2
        inner_when_even = top_short
        if (last_connector_row % 2 == 0) != inner_when_even:
            raise ValueError(
                "bottom_short is inconsistent with this row count: the "
                "connector set above the bottom row must be the inner one "
                f"(use {'even' if top_short else 'odd'} n_rows, or a long bottom row)"
            )

    set_wide = tuple(range(0, cols, 4))       # {0, 4, 8, ...}
    set_inner = tuple(range(2, cols - 1, 4))  # {2, 6, ...}
    short = [
        (r == 0 and top_short) or (r == n_rows - 1 and bottom_short)
        for r in range(n_rows)
    ]

    site_index: Dict[Tuple[str, int, int], int] = {}
    coords: List[Tuple[float, float]] = []
    kinds: List[str] = []
    connector_columns: List[Tuple[int, ...]] = []
    index = 0
    for r in range(n_rows):
        row_cols = range(2, cols - 2) if short[r] else range(cols)
        for col in row_cols:
            site_index[("row", r, col)] = index
            coords.append((float(col), float(2 * r)))
            kinds.append("row")
            index += 1
        if r < n_rows - 1:
            first = set_inner if top_short else set_wide
            second = set_wide if top_short else set_inner
            columns = first if r % 2 == 0 else second
            connector_columns.append(columns)
            for col in columns:
                site_index[("conn", r, col)] = index
                coords.append((float(col), float(2 * r + 1)))
                kinds.append("conn")
                index += 1

    horizontal: List[Tuple[int, Tuple[Edge, ...]]] = []
    for r in range(n_rows):
        row_cols = list(range(2, cols - 2)) if short[r] else list(range(cols))
        row_edges = tuple(
            (site_index[("row", r, a)], site_index[("row", r, b)])
            for a, b in zip(row_cols[:-1], row_cols[1:])
        )
        horizontal.append((r, row_edges))
    vertical: List[Tuple[int, Tuple[Edge, ...]]] = []
    for r, columns in enumerate(connector_columns):
        bond_edges = []
        for col in columns:
            up = site_index[("row", r, col)]
            down = site_index[("row", r + 1, col)]
            mid = site_index[("conn", r, col)]
            bond_edges.append(tuple(sorted((up, mid))))
            bond_edges.append(tuple(sorted((mid, down))))
        vertical.append((r, tuple(bond_edges)))

    edges = tuple(edge for _, row in horizontal for edge in row) + tuple(
        edge for _, bond in vertical for edge in bond
    )
    resolved_name = name or _default_name(index, n_rows, cols, top_short, bottom_short)
    return HeavyHexPatch(
        n_rows=n_rows,
        cols=cols,
        top_short=top_short,
        bottom_short=bottom_short,
        n_sites=index,
        edges=edges,
        coords=tuple(coords),
        kinds=tuple(kinds),
        name=resolved_name,
        horizontal_edges=tuple(horizontal),
        vertical_edges=tuple(vertical),
    )


def _default_name(n_sites: int, n_rows: int, cols: int, top_short: bool, bottom_short: bool) -> str:
    for known, (rows, known_cols, ts, bs) in KNOWN_PATCHES.items():
        if (rows, known_cols, ts, bs) == (
            n_rows, cols, top_short, bottom_short
        ):
            return known
    return f"heavyHex{n_sites}_r{n_rows}c{cols}{'s' if top_short else 'l'}{'s' if bottom_short else 'l'}"


def patch_for_cluster(name: str) -> HeavyHexPatch:
    if name not in KNOWN_PATCHES:
        raise ValueError(
            f"Unknown heavy-hex cluster {name!r}. Known: {', '.join(KNOWN_PATCHES)}"
        )
    n_rows, cols, top_short, bottom_short = KNOWN_PATCHES[name]
    patch = heavy_hex_patch(
        n_rows,
        cols=cols,
        top_short=top_short,
        bottom_short=bottom_short,
        name=name,
    )
    reference = {tuple(sorted(edge)) for edge in CLUSTER_EDGES[name]}
    generated = {tuple(sorted(edge)) for edge in patch.edges}
    if generated != reference:
        raise AssertionError(f"Generated {name} does not reproduce cluster_edges.py")
    return patch


def validate_layers(patch: HeavyHexPatch, layers: Sequence[Sequence[Edge]]) -> None:
    """The three layers must be matchings that cover the edges exactly once."""
    expected = {tuple(sorted(edge)) for edge in patch.edges}
    seen: set = set()
    for number, layer in enumerate(layers, 1):
        sites: set = set()
        for edge in layer:
            normalized = tuple(sorted(edge))
            if normalized in seen:
                raise ValueError(f"Edge {normalized} colored twice (layer {number})")
            if normalized[0] in sites or normalized[1] in sites:
                raise ValueError(f"Layer {number} is not a matching at {normalized}")
            sites.update(normalized)
            seen.add(normalized)
    if seen != expected:
        raise ValueError("Layers do not cover the patch edges exactly")


def _kempe_three_coloring(n_sites: int, edges: Sequence[Edge]) -> List[List[Edge]]:
    """Deterministic proper 3-edge-coloring of a degree-<=3 bipartite graph."""
    color_at: List[Dict[int, int]] = [dict() for _ in range(n_sites)]  # site -> {color: neighbor}
    layers: List[List[Edge]] = [[], [], []]

    def free_colors(site: int) -> set:
        return {0, 1, 2} - set(color_at[site])

    for a, b in edges:
        shared = free_colors(a) & free_colors(b)
        if shared:
            color = min(shared)
        else:
            # Kempe chain: collect the maximal alpha/beta-alternating path from
            # b, then flip its colors atomically.  On a bipartite graph the
            # path cannot reach a, so alpha becomes free at both endpoints.
            alpha = min(free_colors(a))
            beta = min(free_colors(b))
            path = []
            site, current = b, alpha
            while current in color_at[site]:
                neighbor = color_at[site][current]
                path.append((site, neighbor, current))
                site = neighbor
                current = beta if current == alpha else alpha
            for u, v, old in path:
                new = beta if old == alpha else alpha
                del color_at[u][old]
                del color_at[v][old]
                color_at[u][new] = v
                color_at[v][new] = u
                edge = tuple(sorted((u, v)))
                layers[old].remove(edge)
                layers[new].append(edge)
            color = alpha
        color_at[a][color] = b
        color_at[b][color] = a
        layers[color].append(tuple(sorted((a, b))))
    return [sorted(layer) for layer in layers]


def edge_layers(patch: HeavyHexPatch) -> List[List[Edge]]:
    """Edge layers: stored tables for the known patches, generated otherwise."""
    if patch.name in CLUSTER_EDGE_LAYERS:
        layers = [
            [tuple(sorted(edge)) for edge in layer]
            for layer in CLUSTER_EDGE_LAYERS[patch.name]
        ]
    else:
        layers = _kempe_three_coloring(patch.n_sites, patch.edges)
    validate_layers(patch, layers)
    return layers


# ---------------------------------------------------------------------------
# Output helpers


def print_patch_tables(patch: HeavyHexPatch, layers: Sequence[Sequence[Edge]] | None) -> None:
    """Print edges (and layers) in the cluster_edges.py table style."""
    print(f'    "{patch.name}": [')
    for r, row_edges in patch.horizontal_edges:
        listing = ", ".join(str(edge) for edge in row_edges)
        print(f"        {listing},  # Row {r}")
    for r, bond_edges in patch.vertical_edges:
        listing = ", ".join(str(edge) for edge in bond_edges)
        print(f"        {listing},  # Vertical connections {r}-{r + 1}")
    print("    ],")
    if layers is not None:
        print(f'    "{patch.name}": [')
        for layer in layers:
            listing = ", ".join(str(tuple(edge)) for edge in layer)
            print(f"        [{listing}],")
    if layers is not None:
        print("    ],")


def plot_patch_sketch(
    patch: HeavyHexPatch,
    *,
    layers: Sequence[Sequence[Edge]] | None = None,
    output_stem: str | None = None,
    dpi: int = 300,
) -> Tuple[Path, Path]:
    """Draw the traditional heavy-hex lattice picture of the patch."""
    stem = default_output_stem(output_stem or f"{patch.name}_sketch")
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

    color_of: Dict[Edge, int] = {}
    for number, layer in enumerate(layers or ()):  # layer numbers 0..2
        for edge in layer:
            color_of[tuple(sorted(edge))] = number

    height = 2 * patch.n_rows - 1
    fig, ax = plt.subplots(
        figsize=(0.85 * patch.cols + 1.4, 0.62 * height + 1.8)
    )
    for a, b in patch.edges:
        (x1, y1), (x2, y2) = patch.coords[a], patch.coords[b]
        number = color_of.get(tuple(sorted((a, b))))
        color = LAYER_COLORS[number] if number is not None else "#607080"
        ax.plot([x1, x2], [-y1, -y2], color=color, linewidth=2.0, zorder=1)
    xs = [x for x, _ in patch.coords]
    ys = [-y for _, y in patch.coords]
    row_face = ["white" if kind == "row" else "#f2e8c9" for kind in patch.kinds]
    ax.scatter(xs, ys, s=470, facecolor=row_face, edgecolor="black", linewidth=1.0, zorder=2)
    for site, (x, y) in enumerate(patch.coords):
        ax.annotate(str(site), (x, -y), ha="center", va="center", fontsize=8, zorder=3)

    handles = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor="black", label="row qubit"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="#f2e8c9",
               markeredgecolor="black", label="connector qubit"),
    ]
    if layers is not None:
        handles += [
            Line2D([], [], color=LAYER_COLORS[k], linewidth=2, label=f"layer {k + 1}")
            for k in range(3)
        ]
    ax.legend(handles=handles, loc="upper center", fontsize=8,
              bbox_to_anchor=(0.5, -0.04), ncol=len(handles), frameon=False)
    ax.set_title(
        f"{patch.name} | {patch.n_rows} rows x {patch.cols} columns, "
        f"{patch.n_sites} sites",
        fontsize=10,
    )
    ax.set_xlim(-0.8, patch.cols - 0.2)
    ax.set_ylim(-(height - 1) - 0.8, 0.8)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_patch_ladder(
    patch: HeavyHexPatch,
    *,
    ordering: str = "permutations_sat_cw.py",
    legs: int = 4,
    snake: bool = False,
    layers: Sequence[Sequence[Edge]] | None = None,
    output_stem: str | None = None,
    dpi: int = 300,
) -> Tuple[Path, Path]:
    """Draw the quasi-1D ladder view (cluster_ladder_plot style) of the patch."""
    permutation, ordering_label = _resolve_ordering_permutation(
        patch.n_sites, list(patch.edges), ordering=ordering, key=patch.name
    )
    original_of = {permutation[i]: i for i in range(patch.n_sites)}
    color_of: Dict[Edge, int] = {}
    for number, layer in enumerate(layers or ()):
        for edge in layer:
            color_of[tuple(sorted(edge))] = number
    mps_edges = [(permutation[a], permutation[b]) for a, b in patch.edges]
    bandwidth, avg_range, cutwidth = _layout_stats(mps_edges)
    positions = ladder_positions(patch.n_sites, legs, snake=snake)

    stem = default_output_stem(output_stem or f"{patch.name}_ladder_{Path(ordering_label).stem}")
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

    columns = (patch.n_sites + legs - 1) // legs
    fig, ax = plt.subplots(figsize=(1.1 * columns + 1.6, 1.0 * legs + 1.8))
    for (a, b), original in zip(mps_edges, patch.edges):
        (x1, y1), (x2, y2) = positions[a], positions[b]
        number = color_of.get(tuple(sorted(original)))
        color = LAYER_COLORS[number] if number is not None else "#607080"
        manhattan = abs(x1 - x2) + abs(y1 - y2)
        rad = 0.0 if manhattan <= 1.0 else (0.25 if y1 == y2 else 0.30)
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2), connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-", color=color, linewidth=1.6, alpha=0.9,
                zorder=1, shrinkA=9.5, shrinkB=9.5,
            )
        )
    xs = [x for x, _ in positions]
    ys = [y for _, y in positions]
    ax.scatter(xs, ys, s=560, facecolor="white", edgecolor="black", linewidth=1.1, zorder=2)
    for p, (x, y) in enumerate(positions):
        ax.annotate(str(p), (x, y), ha="center", va="center", fontsize=9, zorder=3)
        ax.annotate(
            str(original_of[p]), (x, y), xytext=(-11, 11),
            textcoords="offset points", ha="center", va="center",
            fontsize=6.5, color="#996515", zorder=3,
        )
    handles = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor="black",
               label="MPS position (large) / original site (small)"),
    ]
    if layers is not None:
        handles += [
            Line2D([], [], color=LAYER_COLORS[k], linewidth=2, label=f"layer {k + 1}")
            for k in range(3)
        ]
    ax.legend(handles=handles, loc="upper center", fontsize=8,
              bbox_to_anchor=(0.5, -0.06), ncol=len(handles), frameon=False)
    ax.set_title(
        f"{patch.name} as {legs}-leg ladder | ordering={ordering_label} | "
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
    parser.add_argument(
        "cluster", nargs="?", default=None,
        help=f"known patch name ({', '.join(KNOWN_PATCHES)}); omit to use --rows",
    )
    parser.add_argument("--rows", type=int, default=None, help="number of qubit rows")
    parser.add_argument("--cols", type=int, default=9, help="long-row length, 4*k+1 (default 9)")
    parser.add_argument("--top-long", action="store_true",
                        help="make the top row a long row (default: short)")
    parser.add_argument("--bottom-short", action="store_true",
                        help="make the bottom row a short row (default: long)")
    parser.add_argument("--plot", choices=("sketch", "ladder", "both", "none"),
                        default="sketch")
    parser.add_argument("--ordering", default="permutations_sat_cw.py",
                        help="ladder ordering: file or method (rcm, king, amd, nd)")
    parser.add_argument("--legs", type=int, default=4, help="ladder legs (default 4)")
    parser.add_argument("--snake", action="store_true")
    parser.add_argument("--no-layers", action="store_true",
                        help="single edge color instead of per-layer colors")
    parser.add_argument("--print-edges", action="store_true",
                        help="print cluster_edges.py-style tables for the patch")
    parser.add_argument(
        "--format", choices=("python", "json", "edgelist", "none"), default="python",
        help="coordinate/edge output style, as in cluster_generator.py "
             "(python: coords/edges lists; edgelist: certifier-consumable 'u v' lines)",
    )
    parser.add_argument("--coords-per-line", type=int, default=1,
                        help="coordinate triples printed per line (python format)")
    parser.add_argument("--edges-per-line", type=int, default=8,
                        help="edge pairs printed per line (python format)")
    parser.add_argument("--output-stem", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if (args.cluster is None) == (args.rows is None):
        raise ValueError("Give either a known cluster name or --rows, not both")
    if args.cluster is not None:
        patch = patch_for_cluster(args.cluster)
    else:
        patch = heavy_hex_patch(
            args.rows,
            cols=args.cols,
            top_short=not args.top_long,
            bottom_short=args.bottom_short,
        )
    layers = None if args.no_layers else edge_layers(patch)
    source = "stored" if patch.name in CLUSTER_EDGE_LAYERS else "generated"
    print(
        f"{patch.name}: {patch.n_rows} rows x {patch.cols} columns, "
        f"{patch.n_sites} sites, {len(patch.edges)} edges, "
        f"coordination-2 sites: {len(patch.coordination_two_sites)}, "
        f"layers: {source}"
    )
    if args.print_edges:
        print_patch_tables(patch, layers)

    plot_paths = []
    if args.plot in ("sketch", "both"):
        png_path, pdf_path = plot_patch_sketch(
            patch, layers=layers, output_stem=args.output_stem, dpi=args.dpi
        )
        print(f"sketch: wrote {png_path} and {pdf_path}")
        plot_paths += [png_path, pdf_path]
    if args.plot in ("ladder", "both"):
        stem = f"{args.output_stem}_ladder" if args.output_stem else None
        png_path, pdf_path = plot_patch_ladder(
            patch, ordering=args.ordering, legs=args.legs, snake=args.snake,
            layers=layers, output_stem=stem, dpi=args.dpi,
        )
        print(f"ladder: wrote {png_path} and {pdf_path}")
        plot_paths += [png_path, pdf_path]

    # Cartesian coordinates matching the sketch (rows at even y, connectors
    # between them; y grows upward, z = 0), in cluster_generator.py's formats.
    cartesian = [(x, 0.0 - y, 0.0) for x, y in patch.coords]
    edges = list(patch.edges)
    header = [
        f"patch = {patch.name}",
        f"n_rows = {patch.n_rows}",
        f"cols = {patch.cols}",
        f"top_row = {'short' if patch.top_short else 'long'}",
        f"bottom_row = {'short' if patch.bottom_short else 'long'}",
        f"n_sites = {patch.n_sites}",
        f"n_edges = {len(edges)}",
        f"edge_layers = {source}",
    ] + [f"plot = {path}" for path in plot_paths]
    if args.format == "json":
        import json

        payload = {
            "patch": patch.name,
            "n_rows": patch.n_rows,
            "cols": patch.cols,
            "top_row": "short" if patch.top_short else "long",
            "bottom_row": "short" if patch.bottom_short else "long",
            "n_sites": patch.n_sites,
            "coords": [list(point) for point in cartesian],
            "edges": [list(edge) for edge in edges],
        }
        if layers is not None:
            payload["edge_layers"] = [[list(edge) for edge in layer] for layer in layers]
            payload["edge_layers_source"] = source
        print(json.dumps(payload, indent=2))
    elif args.format == "edgelist":
        print_edge_list(edges, header)
    elif args.format == "python":
        for line in header:
            print(f"# {line}")
        print()
        print_compact_coords(cartesian, per_line=args.coords_per_line)
        print()
        print_compact_edges(edges, per_line=args.edges_per_line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
