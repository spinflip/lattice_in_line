#!/usr/bin/env python3
"""Generate cluster_edges_NNN.py — next-nearest-neighbour (NNN) edge tables for
the lattice-derived clusters in cluster_edges.py, keyed by the SAME cluster
names and using the SAME site numbering.

NNN is the 2nd real-space distance shell, computed from the generated
coordinates (see cluster_generator.neighbor_shell_edges). The generator recipe
for each cluster is parsed from its name (e.g. ``kagomeYcyl288_16x12`` ->
``kagomeYcyl`` 16x12, ``pyrochlore128`` -> tilted 128). Molecular clusters
(fullerenes, icosa, ...) have no lattice recipe and are skipped. For every
cluster that IS generated, the script asserts that the generated nearest-
neighbour graph matches cluster_edges.py exactly, so the emitted NNN tables are
guaranteed index-compatible with the J1 tables.

Re-run after changing cluster_edges.py or the generator:
    python -m vmps_geometry.build_cluster_edges_NNN
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:  # works when run as a module: python -m vmps_geometry.build_cluster_edges_NNN
    from . import cluster_generator as cg
    from .cluster_edges import CLUSTER_EDGES as NN
except ImportError:  # also allow running as a plain script: python build_cluster_edges_NNN.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import cluster_generator as cg  # type: ignore
    from cluster_edges import CLUSTER_EDGES as NN  # type: ignore

Edge = Tuple[int, int]

PYRO_TILTED = {"pyrochlore48a", "pyrochlore48b", "pyrochlore48c",
               "pyrochlore48d", "pyrochlore64", "pyrochlore128"}
PYRO_DIAG = {"pyrochlore32": (2, 2, 2), "pyrochlore108": (3, 3, 3)}


def recipe(name: str):
    """Parse a cluster name into a generator recipe, or None if unmapped."""
    if name in PYRO_TILTED:
        return ("tilted", "pyrochlore", name.replace("pyrochlore", ""))
    if name in PYRO_DIAG:
        return ("diag", "pyrochlore", PYRO_DIAG[name])
    m = re.match(r"^([A-Za-z]+)\d+_(\d+)x(\d+)(?:x(\d+))?$", name)
    if m:
        nz = int(m.group(4)) if m.group(4) else 1
        return ("diag", m.group(1), (int(m.group(2)), int(m.group(3)), nz))
    return None


def coords_edges_periods(kind, lat, param):
    """Return (coords, nn_edges, periods) via the same code paths as the CLI."""
    if kind == "tilted":
        L = cg.get_lattice("pyrochlore", 0.138)
        sc = cg.PYROCHLORE_TILTED_SUPERCELLS[param]
        coords, edges = L.make_supercell(sc)
        return coords, edges, cg.supercell_periods(sc, L.bravais, 1.0)
    nx, ny, nz = param
    if lat in cg.KAGOME_STRIP_VARIANTS:
        knd, px = cg.kagome_variant_kind_and_periodicity(lat)
        coords, edges, pxv, pyv = cg.build_kagome_strip_edges(knd, nx, ny, px)
        return coords, edges, (pxv if px else None, pyv, None)
    if lat in cg.TRIANGULAR_STRIP_VARIANTS:
        knd, px = cg.triangular_variant_kind_and_periodicity(lat)
        coords, edges, pxv, pyv = cg.build_triangular_strip_edges(knd, nx, ny, px)
        return coords, edges, (pxv if px else None, pyv, None)
    L = cg.get_lattice(lat, 0.138)
    coords, edges = L.make_diagonal(nx, ny, nz)
    return coords, edges, cg.diagonal_periods(L, nx, ny, nz, 1.0)


def canon(edges) -> List[Edge]:
    return sorted((min(u, v), max(u, v)) for u, v in edges if u != v)


def build() -> Tuple[Dict[str, List[Edge]], List[Tuple[str, str]]]:
    tables: Dict[str, List[Edge]] = {}
    skipped: List[Tuple[str, str]] = []
    for name in sorted(NN):
        r = recipe(name)
        if r is None:
            skipped.append((name, "molecular / no lattice recipe"))
            continue
        try:
            coords, nn_edges, periods = coords_edges_periods(*r)
        except Exception as exc:  # noqa: BLE001 - report and skip
            skipped.append((name, f"generation failed: {exc!r}"))
            continue
        if canon(nn_edges) != canon(NN[name]):
            skipped.append((name, "NN numbering mismatch vs cluster_edges.py"))
            continue
        nnn, _ = cg.neighbor_shell_edges(coords, periods, 2)
        tables[name] = canon(nnn)
    return tables, skipped


def format_edges(edges: List[Edge], per_line: int = 8) -> str:
    lines = []
    for k in range(0, len(edges), per_line):
        chunk = edges[k:k + per_line]
        lines.append("        " + ", ".join(f"({i}, {j})" for i, j in chunk) + ",")
    return "\n".join(lines)


def render_module(tables: Dict[str, List[Edge]],
                  skipped: List[Tuple[str, str]]) -> str:
    out = [
        '"""Static next-nearest-neighbour (NNN) edge tables for the lattice',
        "clusters in cluster_edges.py.",
        "",
        "Keyed by the SAME cluster names and using the SAME site numbering as",
        "cluster_edges.py, so a CLUSTER_EDGES[name] (J1) and this module's",
        "CLUSTER_EDGES[name] (J2) form a consistent J1/J2 pair.",
        "",
        "GENERATED FILE — do not edit by hand. Regenerate with:",
        "    python -m vmps_geometry.build_cluster_edges_NNN",
        '"""',
        "from __future__ import annotations",
        "",
        "from typing import Dict, List, Tuple",
        "",
        "EdgeList = List[Tuple[int, int]]",
        "",
        "CLUSTER_EDGES: Dict[str, EdgeList] = {",
    ]
    for name in sorted(tables):
        out.append(f'    "{name}": [')
        out.append(format_edges(tables[name]))
        out.append("    ],")
    out.append("}")
    if skipped:
        out.append("")
        out.append("# Clusters without an NNN table (no lattice recipe / not generated):")
        for name, why in skipped:
            out.append(f"#   {name}: {why}")
    out.append("")
    return "\n".join(out)


def main() -> None:
    tables, skipped = build()
    target = Path(__file__).resolve().parent / "cluster_edges_NNN.py"
    target.write_text(render_module(tables, skipped), encoding="utf-8")
    print(f"wrote {target}")
    print(f"  NNN tables: {len(tables)} clusters")
    for name in sorted(tables):
        print(f"    {name:32s} {len(tables[name])} NNN edges")
    if skipped:
        print(f"  skipped: {len(skipped)}")
        for name, why in skipped:
            print(f"    {name:32s} {why}")


if __name__ == "__main__":
    main()
