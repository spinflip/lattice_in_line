#!/usr/bin/env python3
"""Primitive colored unit cell for the infinite heavy-hex ladder."""
from __future__ import annotations

import argparse
import json


UNIT_CELL_SITES = 23

# Natural motif: two connectors, a 9-site rail, three connectors, and a
# second 9-site rail.  The final two edges enter the next translated cell.
NATURAL_EDGES = (
    (0, 4, 0), (1, 8, 0),
    (2, 3, 0), (3, 4, 0), (4, 5, 0), (5, 6, 0),
    (6, 7, 0), (7, 8, 0), (8, 9, 0), (9, 10, 0),
    (2, 11, 0), (11, 14, 0),
    (6, 12, 0), (12, 18, 0),
    (10, 13, 0), (13, 22, 0),
    (14, 15, 0), (15, 16, 0), (16, 17, 0), (17, 18, 0),
    (18, 19, 0), (19, 20, 0), (20, 21, 0), (21, 22, 0),
    (16, 0, 1), (20, 1, 1),
)

# Smallest translation-invariant three-matching decomposition.  The finite
# heavyHex51 and heavyHex74 patches use different valid colorings in their
# common bulk, so neither finite table can be repeated verbatim.  This balanced
# 8/9/9 coloring maximizes agreement with the two complete 23-site bulk motifs
# in the heavyHex74 table, with lexicographic tie breaking.
NATURAL_EDGE_LAYERS = (
    (
        (4, 5, 0), (7, 8, 0), (11, 14, 0), (6, 12, 0),
        (17, 18, 0), (19, 20, 0), (21, 22, 0), (16, 0, 1),
    ),
    (
        (0, 4, 0), (1, 8, 0), (2, 3, 0), (5, 6, 0),
        (9, 10, 0), (12, 18, 0), (13, 22, 0), (15, 16, 0),
        (20, 21, 0),
    ),
    (
        (3, 4, 0), (6, 7, 0), (8, 9, 0), (2, 11, 0),
        (10, 13, 0), (14, 15, 0), (16, 17, 0), (18, 19, 0),
        (20, 1, 1),
    ),
)

# Position -> natural site.  For the periodic graph this has bandwidth 4,
# average range 75/26 = 2.884615..., and cutwidth 4.
OPTIMIZED_ORDER = (
    0, 1, 8, 7, 4, 5, 9, 6, 3, 10, 12, 2,
    13, 22, 18, 11, 21, 19, 17, 14, 20, 15, 16,
)

# Fixed periodic cutwidth-oriented layouts (position -> natural site).
# Width 2: bandwidth 6, average range 144/38, cutwidth 5.
# Width 3: bandwidth 10, average range 226/50, cutwidth 6.
# These are stored layouts; selecting 'optimized' does not run an optimizer.
WIDER_OPTIMIZED_ORDERS = {
    2: (2, 1, 13, 12, 14, 0, 15, 9, 10, 11, 8, 5, 19, 4, 6, 18, 7,
        3, 32, 16, 20, 28, 17, 29, 31, 27, 21, 30, 24, 25, 26, 23, 22),
    3: (3, 2, 1, 18, 17, 19, 14, 15, 16, 0, 13, 20, 10, 11, 12, 25,
        42, 9, 24, 6, 7, 8, 5, 4, 23, 21, 41, 26, 38, 37, 39, 22, 27,
        40, 34, 35, 36, 33, 30, 31, 32, 29, 28),
}


def _wider_cell(width):
    """Two full rails, with alternating inner/wide connector rows.

    Each plaquette row contains width or width+1 hexagons. Horizontal bond
    colors repeat (0,1,0,2) on the first rail and (0,2,0,1) on the second;
    connector legs use the remaining colors at the rail endpoints. This is
    a periodic coloring, not a restriction of the finite-patch color tables.
    """
    inner = width + 1
    cols = 4 * inner + 1
    first = inner
    connectors = first + cols
    second = connectors + inner + 1
    layers = [[], [], []]
    for start, colors in ((first, (0, 1, 0, 2)), (second, (0, 2, 0, 1))):
        for x in range(cols - 1):
            layers[colors[x % 4]].append((start + x, start + x + 1, 0))
    for x in range(inner + 1):
        layers[1].append((first + 4 * x, connectors + x, 0))
        layers[2].append((connectors + x, second + 4 * x, 0))
    for x in range(inner):
        layers[2].append((x, first + 2 + 4 * x, 0))
        layers[1].append((second + 2 + 4 * x, x, 1))
    return second + cols, tuple(sorted(edge for layer in layers for edge in layer)), layers


def _permutation(ordering: str, length=UNIT_CELL_SITES, order=OPTIMIZED_ORDER):
    if ordering == "natural":
        return {site: site for site in range(length)}
    if ordering != "optimized":
        raise ValueError("ordering must be 'optimized' or 'natural'")
    return {site: position for position, site in enumerate(order)}


def _relabel_edge(edge, permutation):
    i, j, offset = edge
    i = permutation[i]
    j = permutation[j]
    if offset == 0 and i > j:
        i, j = j, i
    return i, j, offset


def _edge_sections(edges):
    intra_cell = []
    inter_cell = []
    for i, j, offset in edges:
        if offset == 0:
            intra_cell.append([i, j])
        elif offset == 1:
            inter_cell.append([i, j])
        else:
            raise ValueError(f"Unsupported heavy-hex cell offset {offset}")
    return {"intra_cell": intra_cell, "inter_cell": inter_cell}


def build(ordering="optimized", width=1):
    """Return the periodic geometry and its three edge layers."""
    if width not in (1, 2, 3):
        raise ValueError("width must be 1, 2, or 3 (alternating width,width+1 hexagons)")
    if width == 1:
        length, natural_edges, natural_layers = UNIT_CELL_SITES, NATURAL_EDGES, NATURAL_EDGE_LAYERS
        order = OPTIMIZED_ORDER
    else:
        length, natural_edges, natural_layers = _wider_cell(width)
        order = WIDER_OPTIMIZED_ORDERS[width]
    permutation = _permutation(str(ordering), length, order)
    edges = tuple(_relabel_edge(edge, permutation) for edge in natural_edges)
    layers = [
        _edge_sections(tuple(_relabel_edge(edge, permutation) for edge in layer))
        for layer in natural_layers
    ]
    sections = _edge_sections(edges)
    return {
        "unit_cell_sites": length,
        "intra_cell": {"1.0": sections["intra_cell"]},
        "inter_cell": {"1.0": sections["inter_cell"]},
        "edge_layers": layers,
        "ordering": str(ordering),
        "permutation": {str(site): position for site, position in permutation.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ordering", choices=("optimized", "natural"), default="optimized")
    parser.add_argument("--width", type=int, choices=(1, 2, 3), default=1,
                        help="alternating width,width+1 hexagons (default: 1,2)")
    print(json.dumps(build(**vars(parser.parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
