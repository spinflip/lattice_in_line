"""
python make_cluster.py garnet --Nx 1 --Ny 1 --Nz 1
python make_cluster.py garnet --Nx 2 --Ny 1 --Nz 1
python make_cluster.py fcc --Nx 3 --Ny 3 --Nz 3
python make_cluster.py hyperkagome --Nx 3 --Ny 2 --Nz 2
python make_cluster.py pyrochlore --tilted 48a
python make_cluster.py pyrochlore --supercell "2,1,0;0,3,0;0,0,4"
python -m vmps_geometry.cluster_generator triangularYcyl --Nx 8 --Ny 4 --Nz 1
python -m vmps_geometry.cluster_generator triangularXtorus --Nx 8 --Ny 4 --Nz 1
python -m vmps_geometry.cluster_generator triangularBtorus --Nx 4 --Ny 4 --Nz 1
python -m vmps_geometry.cluster_generator kagomeBtorus --Nx 6 --Ny 6 --Nz 1 \
    --supersite-blocks path/to/kagomeBtorus108_6x6_ss2_assignment.txt
python -m vmps_geometry.cluster_generator C60 \
    --plot-permutation-file path/to/permutations_sat_bw.py
"""
from __future__ import annotations

import ast
import argparse
import importlib.util
import json
import sys
from math import atan2, cos, degrees, pi, sin, sqrt
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from fractions import Fraction
from typing import Dict, List, Sequence, Tuple

try:
    from .cluster_edges import CLUSTER_EDGES
except ImportError:  # Allow running this file directly.
    from cluster_edges import CLUSTER_EDGES


Vec3 = Tuple[float, float, float]
IVec3 = Tuple[int, int, int]
Bond = Tuple[int, int, IVec3]
Edge = Tuple[int, int]

MOLECULE_NAMES = (
    "icosa",
    "C12",
    "C20",
    "C24",
    "C26",
    "C28",
    "C30",
    "C36",
    "C40",
    "C60",
    "cubocta",
    "icosidodeca",
)


def resolve_geometry_path(path) -> Path:
    """Resolve plain paths plus legacy ``geometry/...`` paths into this package.

    Inlined (mirrors ``cluster_graphs.resolve_geometry_path``) so this module
    stays runnable as a plain script and free of the heavier ``cluster_graphs``
    import chain (which pulls in torch)."""
    raw = Path(path).expanduser()
    if raw.is_file():
        return raw
    parts = raw.parts
    if "geometry" in parts:
        idx = parts.index("geometry")
        candidate = Path(__file__).resolve().parent.joinpath(*parts[idx + 1:])
        if candidate.is_file():
            return candidate
    return raw


@dataclass(frozen=True)
class PlotSegment:
    start: Vec3
    end: Vec3
    wrap: IVec3
    sites: Edge | None = None


def det3(m: Sequence[Sequence[int]]) -> int:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def inv3_fraction(m: Sequence[Sequence[int]]) -> List[List[Fraction]]:
    d = det3(m)
    if d == 0:
        raise ValueError("Singular 3x3 matrix")

    cof = [
        [
            m[1][1] * m[2][2] - m[1][2] * m[2][1],
            -(m[1][0] * m[2][2] - m[1][2] * m[2][0]),
            m[1][0] * m[2][1] - m[1][1] * m[2][0],
        ],
        [
            -(m[0][1] * m[2][2] - m[0][2] * m[2][1]),
            m[0][0] * m[2][2] - m[0][2] * m[2][0],
            -(m[0][0] * m[2][1] - m[0][1] * m[2][0]),
        ],
        [
            m[0][1] * m[1][2] - m[0][2] * m[1][1],
            -(m[0][0] * m[1][2] - m[0][2] * m[1][0]),
            m[0][0] * m[1][1] - m[0][1] * m[1][0],
        ],
    ]

    return [[Fraction(cof[j][i], d) for j in range(3)] for i in range(3)]


def matvec_fraction(
    m: Sequence[Sequence[Fraction]],
    v: IVec3,
) -> Tuple[Fraction, Fraction, Fraction]:
    return tuple(sum(m[i][j] * v[j] for j in range(3)) for i in range(3))  # type: ignore


def parse_supercell(text: str) -> Tuple[IVec3, IVec3, IVec3]:
    rows = text.strip().split(";")
    if len(rows) != 3:
        raise ValueError(
            "Invalid --supercell. Expected exactly three ';'-separated vectors, "
            'for example: "2,1,0;0,3,0;0,0,4".'
        )

    vectors: List[IVec3] = []

    for row in rows:
        parts = row.strip().split(",")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid supercell vector {row!r}. "
                "Each vector must contain exactly three comma-separated integers."
            )

        try:
            vec = tuple(int(x.strip()) for x in parts)
        except ValueError as exc:
            raise ValueError(
                f"Invalid supercell vector {row!r}. "
                "All entries must be integers."
            ) from exc

        vectors.append(vec)  # type: ignore

    matrix = [list(v) for v in vectors]
    determinant = det3(matrix)

    if determinant == 0:
        raise ValueError(
            f"Invalid --supercell {text!r}. "
            "The three vectors are linearly dependent; determinant is zero."
        )

    return tuple(vectors)  # type: ignore


def cell_origin_from_bravais(cell: IVec3, bravais: Tuple[Vec3, Vec3, Vec3]) -> Vec3:
    x, y, z = cell
    a1, a2, a3 = bravais
    return (
        x * a1[0] + y * a2[0] + z * a3[0],
        x * a1[1] + y * a2[1] + z * a3[1],
        x * a1[2] + y * a2[2] + z * a3[2],
    )


def basis_position(
    cell: IVec3,
    sublattice: int,
    basis: Sequence[Vec3],
    bravais: Tuple[Vec3, Vec3, Vec3],
) -> Vec3:
    origin = cell_origin_from_bravais(cell, bravais)
    b = basis[sublattice]
    return (origin[0] + b[0], origin[1] + b[1], origin[2] + b[2])


def dist2(a: Vec3, b: Vec3) -> float:
    return sum((a[i] - b[i]) ** 2 for i in range(3))


def canonical_bond(s1: int, s2: int, d: IVec3) -> Bond:
    """
    Canonicalize the infinite-lattice undirected bond:

        (s1, cell 0) -- (s2, cell d)

    against its reversed representation:

        (s2, cell 0) -- (s1, cell -d)
    """
    current = (s1, s2, d)
    reverse = (s2, s1, (-d[0], -d[1], -d[2]))
    return min(current, reverse)


def build_nearest_neighbor_bonds(
    basis: Sequence[Vec3],
    bravais: Tuple[Vec3, Vec3, Vec3],
    search_range: int = 1,
    tol: float = 1.0e-10,
) -> Tuple[Bond, ...]:
    """
    Build the first-neighbor shell from basis coordinates and Bravais vectors.

    This is used for FCC and garnet. It searches neighboring cells in
    [-search_range, search_range]^3, finds the shortest nonzero distance,
    and returns a unique translated-bond table.
    """
    candidates: List[Tuple[float, int, int, IVec3]] = []
    n_basis = len(basis)

    for s1 in range(n_basis):
        r1 = basis_position((0, 0, 0), s1, basis, bravais)

        for s2 in range(n_basis):
            for dx in range(-search_range, search_range + 1):
                for dy in range(-search_range, search_range + 1):
                    for dz in range(-search_range, search_range + 1):
                        d = (dx, dy, dz)

                        if s1 == s2 and d == (0, 0, 0):
                            continue

                        r2 = basis_position(d, s2, basis, bravais)
                        r2_distance = dist2(r1, r2)

                        if r2_distance > tol:
                            candidates.append((r2_distance, s1, s2, d))

    if not candidates:
        raise RuntimeError("No candidate nearest-neighbor bonds found.")

    nearest = min(x[0] for x in candidates)

    bonds = set()
    for r2_distance, s1, s2, d in candidates:
        if abs(r2_distance - nearest) <= tol:
            bonds.add(canonical_bond(s1, s2, d))

    return tuple(sorted(bonds))


@dataclass(frozen=True)
class Lattice:
    name: str
    n_basis: int
    basis: Tuple[Vec3, ...]
    bravais: Tuple[Vec3, Vec3, Vec3]
    bonds: Tuple[Bond, ...]
    index_order: str = "xyz"
    expected_components: int = 1
    pbc: Tuple[bool, bool, bool] = (True, True, True)
    enforce_regular_degree: bool = True

    def wrap_cell(
        self,
        x: int,
        y: int,
        z: int,
        Nx: int,
        Ny: int,
        Nz: int,
    ) -> IVec3 | None:
        dims = (Nx, Ny, Nz)
        coords = [x, y, z]

        for axis, is_periodic in enumerate(self.pbc):
            if is_periodic:
                coords[axis] %= dims[axis]
            elif coords[axis] < 0 or coords[axis] >= dims[axis]:
                return None

        return (coords[0], coords[1], coords[2])

    def cell_index(self, x: int, y: int, z: int, Nx: int, Ny: int, Nz: int) -> int:
        wrapped = self.wrap_cell(x, y, z, Nx, Ny, Nz)
        if wrapped is None:
            raise ValueError(f"Cell {(x, y, z)} is outside the non-periodic cluster boundaries.")

        x, y, z = wrapped

        if self.index_order == "xyz":
            return x + Nx * y + Nx * Ny * z

        if self.index_order == "zyx":
            return z + Nz * y + Nz * Ny * x

        raise ValueError(f"Unknown index_order: {self.index_order}")

    def site_index(
        self,
        x: int,
        y: int,
        z: int,
        s: int,
        Nx: int,
        Ny: int,
        Nz: int,
    ) -> int:
        return self.n_basis * self.cell_index(x, y, z, Nx, Ny, Nz) + s

    def cell_origin(self, x: int, y: int, z: int, a: float) -> Vec3:
        origin = cell_origin_from_bravais((x, y, z), self.bravais)
        return (a * origin[0], a * origin[1], a * origin[2])

    def make_diagonal(
        self,
        Nx: int,
        Ny: int,
        Nz: int,
        a: float = 1.0,
    ) -> Tuple[List[Vec3], List[Edge]]:
        if min(Nx, Ny, Nz) <= 0:
            raise ValueError("Nx, Ny, and Nz must be positive integers.")

        n_sites = self.n_basis * Nx * Ny * Nz
        coords: List[Vec3] = [(0.0, 0.0, 0.0)] * n_sites
        edges: set[Edge] = set()

        for x in range(Nx):
            for y in range(Ny):
                for z in range(Nz):
                    origin = self.cell_origin(x, y, z, a)

                    for s, b in enumerate(self.basis):
                        i = self.site_index(x, y, z, s, Nx, Ny, Nz)
                        coords[i] = (
                            origin[0] + a * b[0],
                            origin[1] + a * b[1],
                            origin[2] + a * b[2],
                        )

                    for s1, s2, (dx, dy, dz) in self.bonds:
                        target = self.wrap_cell(x + dx, y + dy, z + dz, Nx, Ny, Nz)
                        if target is None:
                            continue

                        i = self.site_index(x, y, z, s1, Nx, Ny, Nz)
                        j = self.site_index(*target, s2, Nx, Ny, Nz)
                        edges.add(tuple(sorted((i, j))))

        return coords, sorted(edges)

    def make_supercell(
        self,
        vectors: Sequence[IVec3],
        a: float = 1.0,
    ) -> Tuple[List[Vec3], List[Edge]]:
        if len(vectors) != 3:
            raise ValueError("Need exactly three supercell vectors.")

        S = [list(v) for v in vectors]
        det_s = det3(S)
        n_cells = abs(det_s)

        if n_cells == 0:
            raise ValueError("Supercell vectors are linearly dependent.")

        ST = [[S[j][i] for j in range(3)] for i in range(3)]
        inv_ST = inv3_fraction(ST)

        def equivalent(a_cell: IVec3, b_cell: IVec3) -> bool:
            diff = (
                a_cell[0] - b_cell[0],
                a_cell[1] - b_cell[1],
                a_cell[2] - b_cell[2],
            )
            coeffs = matvec_fraction(inv_ST, diff)
            return all(c.denominator == 1 for c in coeffs)

        reps: List[IVec3] = [(0, 0, 0)]
        queue: List[IVec3] = [(0, 0, 0)]
        steps: List[IVec3] = [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ]

        def find_rep(cell: IVec3) -> int | None:
            for idx, rep in enumerate(reps):
                if equivalent(cell, rep):
                    return idx
            return None

        while queue and len(reps) < n_cells:
            r = queue.pop(0)
            for dx, dy, dz in steps:
                cand = (r[0] + dx, r[1] + dy, r[2] + dz)
                if find_rep(cand) is None:
                    reps.append(cand)
                    queue.append(cand)
                    if len(reps) == n_cells:
                        break

        if len(reps) != n_cells:
            raise RuntimeError(
                f"Failed to enumerate quotient lattice. "
                f"Found {len(reps)} representatives, expected {n_cells}."
            )

        for i in range(len(reps)):
            for j in range(i + 1, len(reps)):
                if equivalent(reps[i], reps[j]):
                    raise RuntimeError(
                        "Internal error: duplicate quotient representatives were generated."
                    )

        def wrap_index(cell: IVec3) -> int:
            idx = find_rep(cell)
            if idx is None:
                raise RuntimeError(f"Could not wrap cell {cell}.")
            return idx

        def site_index(cell: IVec3, s: int) -> int:
            return self.n_basis * wrap_index(cell) + s

        n_sites = self.n_basis * n_cells
        coords: List[Vec3] = [(0.0, 0.0, 0.0)] * n_sites
        edges: set[Edge] = set()

        for cell in reps:
            x, y, z = cell
            origin = self.cell_origin(x, y, z, a)

            for s, b in enumerate(self.basis):
                i = site_index(cell, s)
                coords[i] = (
                    origin[0] + a * b[0],
                    origin[1] + a * b[1],
                    origin[2] + a * b[2],
                )

            for s1, s2, (dx, dy, dz) in self.bonds:
                i = site_index(cell, s1)
                j = site_index((x + dx, y + dy, z + dz), s2)
                edges.add(tuple(sorted((i, j))))

        return coords, sorted(edges)

HYPERKAGOME = Lattice(
    name="hyperkagome",
    n_basis=12,
    bravais=(
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    basis=(
        (3 / 8, 7 / 8, 7 / 8),
        (5 / 8, 7 / 8, 5 / 8),
        (3 / 8, 7 / 8, 3 / 8),
        (5 / 8, 3 / 8, 1 / 8),
        (3 / 8, 3 / 8, 7 / 8),
        (5 / 8, 5 / 8, 7 / 8),
        (1 / 8, 5 / 8, 3 / 8),
        (7 / 8, 5 / 8, 5 / 8),
        (7 / 8, 3 / 8, 3 / 8),
        (1 / 8, 7 / 8, 1 / 8),
        (7 / 8, 1 / 8, 1 / 8),
        (1 / 8, 1 / 8, 7 / 8),
    ),
    bonds=(
        (0, 1, (0, 1, 0)),
        (0, 2, (0, 1, 0)),
        (0, 4, (0, 0, 0)),
        (0, 11, (0, 0, 0)),
        (1, 2, (0, 0, 0)),
        (1, 5, (0, 0, 0)),
        (1, 7, (0, 0, 0)),
        (2, 6, (0, 0, 0)),
        (2, 9, (0, 0, 0)),
        (3, 4, (0, 0, 1)),
        (3, 5, (0, 0, 1)),
        (3, 8, (0, 0, 0)),
        (3, 10, (0, 0, 0)),
        (4, 5, (0, 0, 0)),
        (4, 11, (0, 0, 0)),
        (5, 7, (0, 0, 0)),
        (6, 7, (-1, 0, 0)),
        (6, 8, (-1, 0, 0)),
        (6, 9, (0, 0, 0)),
        (7, 8, (0, 0, 0)),
        (8, 10, (0, 0, 0)),
        (9, 10, (-1, 1, 0)),
        (9, 11, (0, 1, 1)),
        (10, 11, (1, 0, 1)),
    ),
)


PYROCHLORE = Lattice(
    name="pyrochlore",
    n_basis=4,
    bravais=(
        (0.0, 0.5, 0.5),
        (0.5, 0.0, 0.5),
        (0.5, 0.5, 0.0),
    ),
    basis=(
        (0.0, 0.0, 0.0),
        (0.0, 0.25, 0.25),
        (0.25, 0.0, 0.25),
        (0.25, 0.25, 0.0),
    ),
    bonds=(
        (0, 1, (0, 0, 0)),
        (0, 1, (0, 0, -1)),
        (0, 2, (0, 0, 0)),
        (0, 2, (0, -1, 0)),
        (0, 3, (0, 0, 0)),
        (0, 3, (-1, 0, 0)),
        (1, 2, (0, 0, 0)),
        (1, 2, (0, -1, 1)),
        (1, 3, (0, 0, 0)),
        (1, 3, (-1, 0, 1)),
        (2, 3, (0, 0, 0)),
        (2, 3, (-1, 1, 0)),
    ),
)


def make_trillium(u: float) -> Lattice:
    return Lattice(
        name="trillium",
        n_basis=4,
        index_order="zyx",
        bravais=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        basis=(
            (u, u, u),
            (0.5 + u, 0.5 - u, 1.0 - u),
            (1.0 - u, 0.5 + u, 0.5 - u),
            (0.5 - u, 1.0 - u, 0.5 + u),
        ),
        bonds=(
            (0, 1, (-1, 0, -1)),
            (0, 1, (0, 0, -1)),
            (0, 2, (-1, 0, 0)),
            (0, 2, (-1, -1, 0)),
            (0, 3, (0, -1, 0)),
            (0, 3, (0, -1, -1)),
            (1, 2, (0, 0, 1)),
            (1, 2, (0, 0, 0)),
            (1, 3, (0, 0, 0)),
            (1, 3, (0, -1, 0)),
            (2, 3, (0, 0, 0)),
            (2, 3, (1, 0, 0)),
        ),
    )


FCC_BASIS: Tuple[Vec3, ...] = (
    (0.0, 0.0, 0.0),
)

FCC_BRAVAIS: Tuple[Vec3, Vec3, Vec3] = (
    (0.0, 0.5, 0.5),
    (0.5, 0.0, 0.5),
    (0.5, 0.5, 0.0),
)

FCC = Lattice(
    name="fcc",
    n_basis=1,
    bravais=FCC_BRAVAIS,
    basis=FCC_BASIS,
    bonds=build_nearest_neighbor_bonds(FCC_BASIS, FCC_BRAVAIS),
)


GARNET_12C: Tuple[Vec3, ...] = (
    (1 / 8, 0.0, 1 / 4),
    (3 / 8, 0.0, 3 / 4),
    (5 / 8, 0.0, 1 / 4),
    (7 / 8, 0.0, 3 / 4),
    (1 / 4, 1 / 8, 0.0),
    (3 / 4, 3 / 8, 0.0),
    (1 / 4, 5 / 8, 0.0),
    (3 / 4, 7 / 8, 0.0),
    (0.0, 1 / 4, 1 / 8),
    (0.0, 3 / 4, 3 / 8),
    (0.0, 1 / 4, 5 / 8),
    (0.0, 3 / 4, 7 / 8),
)

GARNET_BASIS: Tuple[Vec3, ...] = GARNET_12C + tuple(
    ((x + 0.5) % 1.0, (y + 0.5) % 1.0, (z + 0.5) % 1.0)
    for x, y, z in GARNET_12C
)

GARNET_BRAVAIS: Tuple[Vec3, Vec3, Vec3] = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)

GARNET = Lattice(
    name="garnet",
    n_basis=24,
    bravais=GARNET_BRAVAIS,
    basis=GARNET_BASIS,
    bonds=build_nearest_neighbor_bonds(GARNET_BASIS, GARNET_BRAVAIS),
    expected_components=2,
)


KAGOME_X_BASIS: Tuple[Vec3, ...] = (
    (0.0, 0.0, 0.0),
    (0.5, 0.0, 0.0),
    (0.25, sqrt(3.0) / 4.0, 0.0),
)

KAGOME_X_BRAVAIS: Tuple[Vec3, Vec3, Vec3] = (
    (1.0, 0.0, 0.0),
    (0.5, sqrt(3.0) / 2.0, 0.0),
    (0.0, 0.0, 1.0),
)

KAGOME_Y_BASIS: Tuple[Vec3, ...] = tuple(
    (y, x, z) for x, y, z in KAGOME_X_BASIS
)

KAGOME_Y_BRAVAIS: Tuple[Vec3, Vec3, Vec3] = tuple(
    (y, x, z) for x, y, z in KAGOME_X_BRAVAIS
)  # type: ignore

KAGOME_X_BONDS = build_nearest_neighbor_bonds(KAGOME_X_BASIS, KAGOME_X_BRAVAIS)
KAGOME_Y_BONDS = build_nearest_neighbor_bonds(KAGOME_Y_BASIS, KAGOME_Y_BRAVAIS)

TRIANGULAR_BASIS: Tuple[Vec3, ...] = ((0.0, 0.0, 0.0),)
TRIANGULAR_X_BRAVAIS: Tuple[Vec3, Vec3, Vec3] = (
    (1.0, 0.0, 0.0),
    (0.5, sqrt(3.0) / 2.0, 0.0),
    (0.0, 0.0, 1.0),
)
TRIANGULAR_X_BONDS: Tuple[Bond, ...] = (
    (0, 0, (1, 0, 0)),
    (0, 0, (0, 1, 0)),
    (0, 0, (1, -1, 0)),
)

KAGOME_STRIP_VARIANTS = {
    "kagomeXcyl",
    "kagomeYcyl",
    "kagomeXtorus",
    "kagomeYtorus",
}

TRIANGULAR_STRIP_VARIANTS = {
    "triangularXcyl",
    "triangularYcyl",
    "triangularXtorus",
    "triangularYtorus",
}


def _scale_vec(v: Vec3, a: float) -> Vec3:
    return (a * v[0], a * v[1], a * v[2])


def _add_vec(u: Vec3, v: Vec3) -> Vec3:
    return (u[0] + v[0], u[1] + v[1], u[2] + v[2])


def build_kagome_strip_edges(
    kind: str,
    Nx: int,
    Ny: int,
    periodic_x: bool,
    a: float = 1.0,
) -> Tuple[List[Vec3], List[Edge], Vec3, Vec3]:
    """
    Build the DMRG XC/YC kagome strip convention used by
    Depenbrock-McCulloch-Schollwoeck, PRL 109, 067201 (2012), and
    He-Zaletel-Oshikawa-Pollmann, PRB 90, 205116 (2014).

    Nx is the number of repeated 1D cells along the open/long direction.
    Ny is the paper cylinder suffix: YC Ny or XC Ny. The periodic short
    direction is always y. For the torus variants, the repeated 1D cells are
    also closed periodically in x.

    The edge tables intentionally match kagome_XC.py and kagome_YC.py.
    """
    if Nx <= 0:
        raise ValueError("Nx must be a positive integer for kagome strips.")
    if a <= 0:
        raise ValueError("a must be positive for kagome strips.")

    edges: set[Edge] = set()
    coords: List[Vec3] = []

    def add(i: int, j: int) -> None:
        if i == j:
            raise RuntimeError(f"Self-edge generated in kagome strip: {(i, j)}")
        edges.add(tuple(sorted((i, j))))

    if kind == "Y":
        if Ny < 2 or Ny % 2 != 0:
            raise ValueError(f"YC Ny must be an even integer at least 2, got {Ny}.")

        n_cell_sites = 3 * Ny // 2
        root3 = sqrt(3.0)

        def a_site(x: int, i: int) -> int:
            return x * n_cell_sites + i

        def b_site(x: int, i: int) -> int:
            return x * n_cell_sites + Ny // 2 + (i % Ny)

        # Embedding with a kagome axis along y. The x-cell translation is skew,
        # so the torus period along x is (sqrt(3)*Nx, -Nx).
        for x in range(Nx):
            for i in range(Ny // 2):
                coords.append(_scale_vec((root3 * x - root3 / 2.0, -x + 2.0 * i + 1.5, 0.0), a))
            for i in range(Ny):
                coords.append(_scale_vec((root3 * x, -x + float(i), 0.0), a))

        for x in range(Nx):
            for i in range(Ny):
                add(b_site(x, i), b_site(x, i + 1))

            for i in range(Ny // 2):
                add(a_site(x, i), b_site(x, 2 * i + 1))
                add(a_site(x, i), b_site(x, 2 * i + 2))

                xp = x + 1
                if xp < Nx:
                    add(b_site(x, 2 * i), a_site(xp, i))
                    add(b_site(x, 2 * i + 1), a_site(xp, i))
                elif periodic_x:
                    add(b_site(x, 2 * i), a_site(0, i))
                    add(b_site(x, 2 * i + 1), a_site(0, i))

        period_x = _scale_vec((root3 * Nx, -float(Nx), 0.0), a)
        period_y = _scale_vec((0.0, float(Ny), 0.0), a)

    elif kind == "X":
        if Ny < 4 or Ny % 4 != 0:
            raise ValueError(f"XC Ny must be a multiple of 4, got {Ny}.")

        n_y = Ny // 4
        n_cell_sites = 6 * n_y
        root3 = sqrt(3.0)
        block_height = 2.0 * root3

        def base(x: int, y: int) -> int:
            return x * n_cell_sites + 6 * y

        def a0(x: int, y: int) -> int:
            return base(x, y)

        def a1(x: int, y: int) -> int:
            return base(x, y) + 1

        def b_site(x: int, y: int) -> int:
            return base(x, y) + 2

        def c0(x: int, y: int) -> int:
            return base(x, y) + 3

        def c1(x: int, y: int) -> int:
            return base(x, y) + 4

        def d_site(x: int, y: int) -> int:
            return base(x, y) + 5

        # Embedding with a kagome axis along x and vertical period 2*a2-a1.
        # One y-block has height 2*sqrt(3), so Ny=4*n_y has circumference
        # Ny*sqrt(3)/2, as in the paper convention.
        for x in range(Nx):
            x0 = 2.0 * x
            for y in range(n_y):
                y0 = block_height * y
                coords.extend(
                    [
                        _scale_vec((x0 + 0.0, y0 + 0.0, 0.0), a),
                        _scale_vec((x0 + 1.0, y0 + 0.0, 0.0), a),
                        _scale_vec((x0 - 0.5, y0 + root3 / 2.0, 0.0), a),
                        _scale_vec((x0 + 0.0, y0 + root3, 0.0), a),
                        _scale_vec((x0 + 1.0, y0 + root3, 0.0), a),
                        _scale_vec((x0 + 0.5, y0 + 1.5 * root3, 0.0), a),
                    ]
                )

        for x in range(Nx):
            for y in range(n_y):
                yp = (y + 1) % n_y
                add(a0(x, y), a1(x, y))
                add(c0(x, y), c1(x, y))
                add(a0(x, y), b_site(x, y))
                add(b_site(x, y), c0(x, y))
                add(c0(x, y), d_site(x, y))
                add(c1(x, y), d_site(x, y))
                add(d_site(x, y), a0(x, yp))
                add(d_site(x, y), a1(x, yp))

                xp = x + 1
                if xp < Nx:
                    add(a1(x, y), a0(xp, y))
                    add(c1(x, y), c0(xp, y))
                    add(a1(x, y), b_site(xp, y))
                    add(c1(x, y), b_site(xp, y))
                elif periodic_x:
                    add(a1(x, y), a0(0, y))
                    add(c1(x, y), c0(0, y))
                    add(a1(x, y), b_site(0, y))
                    add(c1(x, y), b_site(0, y))

        period_x = _scale_vec((2.0 * Nx, 0.0, 0.0), a)
        period_y = _scale_vec((0.0, block_height * n_y, 0.0), a)

    else:
        raise ValueError(f"Unknown kagome strip kind: {kind!r}")

    return coords, sorted(edges), period_x, period_y


def build_triangular_strip_edges(
    kind: str,
    Nx: int,
    Ny: int,
    periodic_x: bool,
    a: float = 1.0,
) -> Tuple[List[Vec3], List[Edge], Vec3, Vec3]:
    """Build triangular XC/YC strips matching the unit-cell generators."""
    if Nx <= 0 or Ny < 2:
        raise ValueError("Triangular strips need Nx > 0 and Ny >= 2.")
    if a <= 0:
        raise ValueError("a must be positive for triangular strips.")
    if kind == "X" and not periodic_x and Nx < 2:
        raise ValueError("triangularXcyl needs Nx >= 2.")

    root3 = sqrt(3.0)
    if kind == "Y":
        longitudinal = (root3 / 2.0, -0.5, 0.0)
        transverse = (0.0, 1.0, 0.0)
    elif kind == "X":
        longitudinal = (1.0, 0.0, 0.0)
        transverse = (1.5, root3 / 2.0, 0.0)
    else:
        raise ValueError(f"Unknown triangular strip kind: {kind!r}")

    def site(x: int, y: int) -> int:
        return x * Ny + (y % Ny)

    coords = [
        _scale_vec(
            _add_vec(_scale_vec(longitudinal, float(x)), _scale_vec(transverse, float(y))),
            a,
        )
        for x in range(Nx)
        for y in range(Ny)
    ]
    edges: set[Edge] = set()

    def add(i: int, j: int) -> None:
        if i == j:
            raise RuntimeError(f"Self-edge generated in triangular strip: {(i, j)}")
        edges.add(tuple(sorted((i, j))))

    if kind == "Y":
        for x in range(Nx):
            for y in range(Ny):
                add(site(x, y), site(x, y + 1))
        for x in range(Nx if periodic_x else Nx - 1):
            xp = (x + 1) % Nx
            for y in range(Ny):
                add(site(x, y), site(xp, y))
                add(site(x, y), site(xp, y + 1))
    else:
        for offset in (1, 2):
            n_source = Nx if periodic_x else max(0, Nx - offset)
            for x in range(n_source):
                xp = (x + offset) % Nx
                for y in range(Ny):
                    if offset == 1:
                        add(site(x, y), site(xp, y))
                    add(site(x, y), site(xp, y - 1))

    period_x = _scale_vec(longitudinal, float(Nx) * a)
    period_y = _scale_vec(transverse, float(Ny) * a)
    return coords, sorted(edges), period_x, period_y

def kagome_variant_kind_and_periodicity(name: str) -> Tuple[str, bool]:
    if name == "kagomeXcyl":
        return "X", False
    if name == "kagomeYcyl":
        return "Y", False
    if name == "kagomeXtorus":
        return "X", True
    if name == "kagomeYtorus":
        return "Y", True
    raise ValueError(f"Unknown kagome strip variant: {name}")


def triangular_variant_kind_and_periodicity(name: str) -> Tuple[str, bool]:
    if name == "triangularXcyl":
        return "X", False
    if name == "triangularYcyl":
        return "Y", False
    if name == "triangularXtorus":
        return "X", True
    if name == "triangularYtorus":
        return "Y", True
    raise ValueError(f"Unknown triangular strip variant: {name}")


def validate_kagome_strip(name: str, coords: Sequence[Vec3], edges: Sequence[Edge], periodic_x: bool) -> None:
    n_sites = len(coords)
    if len(set(edges)) != len(edges):
        raise RuntimeError("Duplicate kagome strip edges found.")
    degree = [0] * n_sites
    for i, j in edges:
        if i == j:
            raise RuntimeError(f"Self-edge found: {(i, j)}")
        if not (0 <= i < n_sites and 0 <= j < n_sites):
            raise RuntimeError(f"Edge out of range: {(i, j)}")
        degree[i] += 1
        degree[j] += 1
    if periodic_x:
        if min(degree) != 4 or max(degree) != 4:
            raise RuntimeError(
                f"Unexpected torus degrees for {name}: min={min(degree)}, max={max(degree)}, expected=4. "
                "The requested torus is probably too short for a simple-edge representation."
            )
    else:
        if min(degree) < 2 or max(degree) > 4:
            raise RuntimeError(
                f"Unexpected cylinder degrees for {name}: min={min(degree)}, max={max(degree)}."
            )
    n_components = count_components(n_sites, edges)
    if n_components != 1:
        raise RuntimeError(f"Unexpected number of connected components: got {n_components}, expected 1.")


def validate_triangular_strip(
    name: str,
    coords: Sequence[Vec3],
    edges: Sequence[Edge],
    periodic_x: bool,
) -> None:
    n_sites = len(coords)
    if len(set(edges)) != len(edges):
        raise RuntimeError("Duplicate triangular strip edges found.")
    degree = [0] * n_sites
    for i, j in edges:
        if i == j:
            raise RuntimeError(f"Self-edge found: {(i, j)}")
        if not (0 <= i < n_sites and 0 <= j < n_sites):
            raise RuntimeError(f"Edge out of range: {(i, j)}")
        degree[i] += 1
        degree[j] += 1
    if periodic_x:
        if min(degree) != 6 or max(degree) != 6:
            raise RuntimeError(
                f"Unexpected torus degrees for {name}: min={min(degree)}, "
                f"max={max(degree)}, expected=6. Increase Nx/Ny to avoid "
                "collapsed edges in the simple graph."
            )
    elif min(degree) < 2 or max(degree) > 6:
        raise RuntimeError(
            f"Unexpected cylinder degrees for {name}: min={min(degree)}, max={max(degree)}."
        )
    if count_components(n_sites, edges) != 1:
        raise RuntimeError(f"{name} is disconnected.")


def validate_triangular_strip_geometry(
    coords: Sequence[Vec3],
    edges: Sequence[Edge],
    period_x: Vec3,
    period_y: Vec3,
    periodic_x: bool,
    expected_length: float,
    tol: float = 1.0e-10,
) -> None:
    px = period_x if periodic_x else None
    for edge in edges:
        _, _, d2 = nearest_periodic_delta(coords[edge[0]], coords[edge[1]], px, period_y)
        if abs(d2 - expected_length * expected_length) > tol:
            raise RuntimeError(
                f"Triangular strip edge {edge} has embedded length squared {d2}, "
                f"expected {expected_length * expected_length}."
            )


def build_edge_plot_segments(coords: Sequence[Vec3], edges: Sequence[Edge]) -> List[PlotSegment]:
    return [
        PlotSegment(start=coords[i], end=coords[j], wrap=(0, 0, 0), sites=(i, j))
        for i, j in edges
    ]


def build_schlegel_coordinates(
    n_sites: int,
    edges: Sequence[Edge],
    a: float = 1.0,
) -> List[Vec3]:
    """Return a radially expanded Schlegel-style embedding.

    A pentagonal face is preferred when present; otherwise the largest face is
    used. For the 3-connected molecular graphs here, fixing that face to a
    regular polygon and placing every other vertex at its neighbors' barycenter
    gives a crossing-free straight-line drawing. A radial power transform then
    opens space around the otherwise crowded center while fixing the exterior.
    """
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "Molecule plotting requires networkx; reinstall vmps_geometry."
        ) from exc

    import numpy as np

    graph = nx.Graph()
    graph.add_nodes_from(range(n_sites))
    graph.add_edges_from(edges)
    is_planar, embedding = nx.check_planarity(graph)
    if not is_planar:
        raise ValueError("A Schlegel projection requires a planar molecular graph.")

    seen_half_edges = set()
    faces = []
    for i in sorted(embedding):
        for j in embedding.neighbors_cw_order(i):
            if (i, j) not in seen_half_edges:
                faces.append(embedding.traverse_face(i, j, seen_half_edges))

    def canonical_face(face: Sequence[int]) -> Tuple[int, ...]:
        cycle = list(face)
        variants = []
        for order in (cycle, list(reversed(cycle))):
            variants.extend(
                tuple(order[offset:] + order[:offset])
                for offset in range(len(order))
            )
        return min(variants)

    face_sizes = {len(face) for face in faces}
    outer_size = 5 if 5 in face_sizes else max(face_sizes)
    outer = min(
        canonical_face(face) for face in faces if len(face) == outer_size
    )
    outer_positions = {
        site: (
            a * cos(0.5 * pi - 2.0 * pi * index / outer_size),
            a * sin(0.5 * pi - 2.0 * pi * index / outer_size),
        )
        for index, site in enumerate(outer)
    }

    interior = [site for site in range(n_sites) if site not in outer_positions]
    interior_index = {site: index for index, site in enumerate(interior)}
    matrix = np.zeros((len(interior), len(interior)), dtype=float)
    rhs = np.zeros((len(interior), 2), dtype=float)
    for site in interior:
        row = interior_index[site]
        matrix[row, row] = graph.degree(site)
        for neighbor in graph[site]:
            if neighbor in interior_index:
                matrix[row, interior_index[neighbor]] -= 1.0
            else:
                rhs[row] += outer_positions[neighbor]

    solved = np.linalg.solve(matrix, rhs) if interior else rhs
    positions = dict(outer_positions)
    positions.update(
        (site, tuple(solved[interior_index[site]])) for site in interior
    )

    radial_power = 0.55
    outer_radius = abs(a)
    coords = []
    for site in range(n_sites):
        x, y = positions[site]
        radius = sqrt(x * x + y * y)
        scale = (
            (radius / outer_radius) ** (radial_power - 1.0)
            if radius > 0.0 and outer_radius > 0.0
            else 1.0
        )
        coords.append((float(scale * x), float(scale * y), 0.0))
    return coords


def min_image_end(p: Vec3, q: Vec3, periods: "Periods") -> Tuple[Vec3, IVec3]:
    """Nearest periodic image of ``q`` relative to ``p`` and its integer wrap.
    Companion to :func:`min_image_dist2`, used to draw wrapped bonds."""
    px, py, pz = periods
    rx = (-1, 0, 1) if px is not None else (0,)
    ry = (-1, 0, 1) if py is not None else (0,)
    rz = (-1, 0, 1) if pz is not None else (0,)
    best_end = q
    best_wrap: IVec3 = (0, 0, 0)
    best_d2 = dist2(p, q)
    for mx in rx:
        for my in ry:
            for mz in rz:
                s = q
                if px is not None and mx:
                    s = _add_vec(s, _scale_vec(px, float(mx)))
                if py is not None and my:
                    s = _add_vec(s, _scale_vec(py, float(my)))
                if pz is not None and mz:
                    s = _add_vec(s, _scale_vec(pz, float(mz)))
                d = dist2(p, s)
                if d < best_d2 - 1.0e-12:
                    best_d2 = d
                    best_end = s
                    best_wrap = (mx, my, mz)
    return best_end, best_wrap


def build_periodic_plot_segments(
    coords: Sequence[Vec3], edges: Sequence[Edge], periods: "Periods"
) -> List[PlotSegment]:
    """Plot segments for an arbitrary edge list, drawing each bond to the
    nearest periodic image of its endpoint (so higher neighbour shells render
    with correct wrapping). Used for neighbour shells > 1."""
    segments: List[PlotSegment] = []
    for i, j in edges:
        end, wrap = min_image_end(coords[i], coords[j], periods)
        segments.append(PlotSegment(start=coords[i], end=end, wrap=wrap, sites=(i, j)))
    return segments


def nearest_periodic_delta(
    start: Vec3,
    end: Vec3,
    period_x: Vec3 | None,
    period_y: Vec3 | None,
) -> Tuple[Vec3, IVec3, float]:
    shifts_x = (-1, 0, 1) if period_x is not None else (0,)
    shifts_y = (-1, 0, 1) if period_y is not None else (0,)
    best_end = end
    best_wrap: IVec3 = (0, 0, 0)
    best_d2 = dist2(start, end)

    for mx in shifts_x:
        for my in shifts_y:
            shifted = end
            if period_x is not None:
                shifted = _add_vec(shifted, _scale_vec(period_x, float(mx)))
            if period_y is not None:
                shifted = _add_vec(shifted, _scale_vec(period_y, float(my)))
            d2 = dist2(start, shifted)
            if d2 < best_d2 - 1.0e-12:
                best_end = shifted
                best_wrap = (mx, my, 0)
                best_d2 = d2

    return best_end, best_wrap, best_d2


def build_strip_plot_segments(
    coords: Sequence[Vec3],
    edges: Sequence[Edge],
    period_x: Vec3,
    period_y: Vec3,
    periodic_x: bool,
) -> List[PlotSegment]:
    segments: List[PlotSegment] = []
    px = period_x if periodic_x else None
    py = period_y
    for i, j in edges:
        end, wrap, _ = nearest_periodic_delta(coords[i], coords[j], px, py)
        segments.append(PlotSegment(start=coords[i], end=end, wrap=wrap, sites=(i, j)))
    return segments


def align_strip_plot_axis(
    coords: Sequence[Vec3],
    segments: Sequence[PlotSegment],
    longitudinal: Vec3,
) -> Tuple[List[Vec3], List[PlotSegment], float]:
    """Rotate a 2D strip so its longitudinal lattice axis is horizontal."""
    angle = -atan2(longitudinal[1], longitudinal[0])
    cosine = cos(angle)
    sine = sin(angle)

    def rotate(point: Vec3) -> Vec3:
        return (
            cosine * point[0] - sine * point[1],
            sine * point[0] + cosine * point[1],
            point[2],
        )

    rotated_coords = [rotate(point) for point in coords]
    rotated_segments = [
        PlotSegment(
            start=rotate(segment.start),
            end=rotate(segment.end),
            wrap=segment.wrap,
            sites=segment.sites,
        )
        for segment in segments
    ]
    return rotated_coords, rotated_segments, angle


def build_kagome_strip_plot_segments(
    coords: Sequence[Vec3],
    edges: Sequence[Edge],
    period_x: Vec3,
    period_y: Vec3,
    periodic_x: bool,
) -> List[PlotSegment]:
    return build_strip_plot_segments(coords, edges, period_x, period_y, periodic_x)


def validate_kagome_strip_geometry(
    coords: Sequence[Vec3],
    edges: Sequence[Edge],
    period_x: Vec3,
    period_y: Vec3,
    periodic_x: bool,
    expected_length: float,
    tol: float = 1.0e-10,
) -> None:
    px = period_x if periodic_x else None
    py = period_y
    for edge in edges:
        i, j = edge
        _, _, d2 = nearest_periodic_delta(coords[i], coords[j], px, py)
        if abs(d2 - expected_length * expected_length) > tol:
            raise RuntimeError(
                f"Kagome strip edge {edge} has embedded length squared {d2}, "
                f"expected {expected_length * expected_length}."
            )

# Primitive kagome definitions retained for ordinary Bravais-cell checks.
# Paper-style DMRG XC/YC cylinders are generated by build_kagome_strip_edges
# above, because paper XC uses the vertical period 2*a2-a1 rather than a
# simple wrap along primitive vector a2.
KAGOME_VARIANTS: Dict[str, Lattice] = {
    "kagomeBcyl": Lattice(
        name="kagomeBcyl",
        n_basis=3,
        bravais=KAGOME_X_BRAVAIS,
        basis=KAGOME_X_BASIS,
        bonds=KAGOME_X_BONDS,
        pbc=(False, True, False),
        enforce_regular_degree=False,
    ),
    "kagomeBtorus": Lattice(
        name="kagomeBtorus",
        n_basis=3,
        bravais=KAGOME_X_BRAVAIS,
        basis=KAGOME_X_BASIS,
        bonds=KAGOME_X_BONDS,
        pbc=(True, True, False),
        enforce_regular_degree=True,
    ),
    "kagomeXcyl": Lattice(
        name="kagomeXcyl",
        n_basis=3,
        bravais=KAGOME_X_BRAVAIS,
        basis=KAGOME_X_BASIS,
        bonds=KAGOME_X_BONDS,
        pbc=(False, True, False),
        enforce_regular_degree=False,
    ),
    "kagomeYcyl": Lattice(
        name="kagomeYcyl",
        n_basis=3,
        bravais=KAGOME_Y_BRAVAIS,
        basis=KAGOME_Y_BASIS,
        bonds=KAGOME_Y_BONDS,
        pbc=(True, False, False),
        enforce_regular_degree=False,
    ),
    "kagomeXtorus": Lattice(
        name="kagomeXtorus",
        n_basis=3,
        bravais=KAGOME_X_BRAVAIS,
        basis=KAGOME_X_BASIS,
        bonds=KAGOME_X_BONDS,
        pbc=(True, True, False),
        enforce_regular_degree=False,
    ),
    "kagomeYtorus": Lattice(
        name="kagomeYtorus",
        n_basis=3,
        bravais=KAGOME_Y_BRAVAIS,
        basis=KAGOME_Y_BASIS,
        bonds=KAGOME_Y_BONDS,
        pbc=(True, True, False),
        enforce_regular_degree=False,
    ),
}

TRIANGULAR_BTORUS = Lattice(
    name="triangularBtorus",
    n_basis=1,
    bravais=TRIANGULAR_X_BRAVAIS,
    basis=TRIANGULAR_BASIS,
    bonds=TRIANGULAR_X_BONDS,
    pbc=(True, True, False),
    enforce_regular_degree=True,
)


SQUARE_BASIS: Tuple[Vec3, ...] = ((0.0, 0.0, 0.0),)
SQUARE_BRAVAIS: Tuple[Vec3, Vec3, Vec3] = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
SQUARE_BONDS: Tuple[Bond, ...] = (
    (0, 0, (1, 0, 0)),
    (0, 0, (0, 1, 0)),
)

# Plain square lattice as a cylinder (periodic y / circumference, open x) and a
# torus (periodic in both directions). Both are two-dimensional: use --Nz 1.
SQUARE_VARIANTS: Dict[str, Lattice] = {
    "squareCyl": Lattice(
        name="squareCyl",
        n_basis=1,
        bravais=SQUARE_BRAVAIS,
        basis=SQUARE_BASIS,
        bonds=SQUARE_BONDS,
        pbc=(False, True, False),
        enforce_regular_degree=False,
    ),
    "squareTorus": Lattice(
        name="squareTorus",
        n_basis=1,
        bravais=SQUARE_BRAVAIS,
        basis=SQUARE_BASIS,
        bonds=SQUARE_BONDS,
        pbc=(True, True, False),
        enforce_regular_degree=True,
    ),
}


PYROCHLORE_TILTED_SUPERCELLS: Dict[str, Tuple[IVec3, IVec3, IVec3]] = {
    "48a": ((-2, 0, 0), (-1, 1, 2), (0, -2, 2)),
    "48b": ((-1, 1, 2), (-2, 1, -1), (-2, -1, 1)),
    "48c": ((2, 1, 0), (0, 1, 2), (-2, 2, 0)),
    "48d": ((1, 1, 1), (-2, 2, 0), (-2, 0, 2)),
    "64": ((1, 1, 1), (3, -1, -1), (1, -3, 1)),
    "128": ((-2, 2, 2), (-2, 2, -2), (-2, -2, 2)),
}


def canonical_tilted_name(name: str) -> str:
    name = name.lower().strip()
    if name.startswith("pyrochlore"):
        name = name.removeprefix("pyrochlore")
    return name


def get_lattice(name: str, trillium_u: float) -> Lattice:
    if name in KAGOME_VARIANTS:
        return KAGOME_VARIANTS[name]
    if name == "triangularBtorus" or name in TRIANGULAR_STRIP_VARIANTS:
        return TRIANGULAR_BTORUS
    if name in SQUARE_VARIANTS:
        return SQUARE_VARIANTS[name]
    if name == "hyperkagome":
        return HYPERKAGOME
    if name == "pyrochlore":
        return PYROCHLORE
    if name == "trillium":
        return make_trillium(trillium_u)
    if name == "fcc":
        return FCC
    if name == "garnet":
        return GARNET

    raise ValueError(f"Unknown lattice: {name}")


def expected_degree(lattice: Lattice) -> int:
    return 2 * len(lattice.bonds) // lattice.n_basis


def count_components(n_sites: int, edges: Sequence[Edge]) -> int:
    if n_sites == 0:
        return 0

    adjacency: List[List[int]] = [[] for _ in range(n_sites)]
    for i, j in edges:
        adjacency[i].append(j)
        adjacency[j].append(i)

    seen = [False] * n_sites
    n_components = 0

    for start in range(n_sites):
        if seen[start]:
            continue

        n_components += 1
        q: deque[int] = deque([start])
        seen[start] = True

        while q:
            i = q.popleft()
            for j in adjacency[i]:
                if not seen[j]:
                    seen[j] = True
                    q.append(j)

    return n_components


def validate_graph(lattice: Lattice, coords: List[Vec3], edges: List[Edge]) -> None:
    n_sites = len(coords)

    if len(set(edges)) != len(edges):
        raise RuntimeError("Duplicate edges found.")

    degree = [0] * n_sites

    for i, j in edges:
        if i == j:
            raise RuntimeError(
                f"Self-edge found: {(i, j)}. "
                "This usually means the finite PBC cluster is too small "
                "for a simple-edge representation."
            )
        if not (0 <= i < n_sites and 0 <= j < n_sites):
            raise RuntimeError(f"Edge out of range: {(i, j)}")
        degree[i] += 1
        degree[j] += 1

    d = expected_degree(lattice)

    if lattice.enforce_regular_degree:
        if min(degree) != d or max(degree) != d:
            raise RuntimeError(
                f"Unexpected degrees: min={min(degree)}, max={max(degree)}, expected={d}. "
                "For FCC, this can happen when the cluster is too small and opposite "
                "nearest-neighbor bonds collapse onto the same simple edge."
            )

        expected_edges = n_sites * d // 2
        if len(edges) != expected_edges:
            raise RuntimeError(
                f"Unexpected edge count: got {len(edges)}, expected={expected_edges}. "
                "This usually indicates a too-small simple-graph PBC cluster or an invalid supercell."
            )

    n_components = count_components(n_sites, edges)
    if n_components != lattice.expected_components:
        raise RuntimeError(
            f"Unexpected number of connected components: got {n_components}, "
            f"expected {lattice.expected_components}."
        )


# ----------------------------------------------------------------------
# real-space neighbour shells beyond nearest neighbour (NNN and further)
# ----------------------------------------------------------------------

Periods = Tuple["Vec3 | None", "Vec3 | None", "Vec3 | None"]


def min_image_dist2(p: Vec3, q: Vec3, periods: Periods) -> float:
    """Minimum-image squared distance between ``p`` and ``q`` under up to three
    periodic directions. A ``None`` entry in ``periods`` means that axis is open
    (no wrapping). Generalizes :func:`nearest_periodic_delta` to three dims."""
    px, py, pz = periods
    rx = (-1, 0, 1) if px is not None else (0,)
    ry = (-1, 0, 1) if py is not None else (0,)
    rz = (-1, 0, 1) if pz is not None else (0,)
    best = dist2(p, q)
    for mx in rx:
        for my in ry:
            for mz in rz:
                if mx == 0 and my == 0 and mz == 0:
                    continue
                s = q
                if px is not None and mx:
                    s = _add_vec(s, _scale_vec(px, float(mx)))
                if py is not None and my:
                    s = _add_vec(s, _scale_vec(py, float(my)))
                if pz is not None and mz:
                    s = _add_vec(s, _scale_vec(pz, float(mz)))
                d = dist2(p, s)
                if d < best:
                    best = d
    return best


def diagonal_periods(
    lattice: "Lattice", Nx: int, Ny: int, Nz: int, a: float
) -> Periods:
    """Real-space period vectors of a diagonal ``Nx x Ny x Nz`` cluster, with
    ``None`` on non-periodic axes (per ``lattice.pbc``). Mirrors the geometry of
    :meth:`Lattice.make_diagonal` (period along axis i is ``Ni * a * bravais[i]``)."""
    counts = (Nx, Ny, Nz)
    periods: List["Vec3 | None"] = []
    for axis in range(3):
        if lattice.pbc[axis]:
            periods.append(_scale_vec(lattice.bravais[axis], a * counts[axis]))
        else:
            periods.append(None)
    return (periods[0], periods[1], periods[2])


def supercell_periods(
    vectors: Sequence[IVec3], bravais: Tuple[Vec3, Vec3, Vec3], a: float
) -> Periods:
    """Real-space period vectors of a :meth:`Lattice.make_supercell` cluster:
    each supercell vector is an integer combination of Bravais vectors, scaled
    by ``a``; all three axes are periodic."""
    out: List["Vec3 | None"] = []
    for v in vectors:
        acc: Vec3 = (0.0, 0.0, 0.0)
        for k in range(3):
            acc = _add_vec(acc, _scale_vec(bravais[k], float(v[k])))
        out.append(_scale_vec(acc, a))
    return (out[0], out[1], out[2])


def neighbor_shell_edges(
    coords: Sequence[Vec3],
    periods: Periods,
    shell: int,
    tol: float = 1.0e-6,
) -> Tuple[List[Edge], List[float]]:
    """Edges of the ``shell``-th real-space neighbour shell.

    For each site, the distinct minimum-image distances to all other sites are
    found; the site's ``shell``-th distinct distance defines its neighbours in
    that shell. An unordered pair ``{i, j}`` is a shell edge iff their distance
    matches the ``shell``-th distinct distance of ``i`` OR of ``j`` (union;
    reduces to the global shell for homogeneous clusters, and stays faithful
    near open boundaries where sites are inequivalent).

    Returns ``(sorted edges, sorted distinct shell distances realized)``.
    """
    if shell < 1:
        raise ValueError("shell must be a positive integer (1 = nearest neighbour).")
    n = len(coords)
    d2 = [[0.0] * n for _ in range(n)]
    for i in range(n):
        ci = coords[i]
        for j in range(i + 1, n):
            v = min_image_dist2(ci, coords[j], periods)
            d2[i][j] = v
            d2[j][i] = v

    shell_d2: List["float | None"] = [None] * n
    for i in range(n):
        distinct: List[float] = []
        for j in sorted(range(n), key=lambda jj: d2[i][jj]):
            if j == i:
                continue
            v = d2[i][j]
            if not distinct or v - distinct[-1] > tol:
                distinct.append(v)
            if len(distinct) >= shell:
                break
        if len(distinct) >= shell:
            shell_d2[i] = distinct[shell - 1]

    edges: set[Edge] = set()
    for i in range(n):
        target = shell_d2[i]
        if target is None:
            continue
        for j in range(n):
            if j != i and abs(d2[i][j] - target) <= tol:
                edges.add((min(i, j), max(i, j)))

    realized = sorted({round(v, 9) for v in shell_d2 if v is not None})
    return sorted(edges), [sqrt(r) for r in realized]


def validate_neighbor_shell_graph(n_sites: int, edges: Sequence[Edge]) -> Dict[str, float]:
    """Relaxed validation for higher neighbour shells (shell > 1).

    Rejects duplicate, self, and out-of-range edges (all signal a too-small PBC
    cluster), but does NOT assert the nearest-neighbour regular degree or
    component count, which generally do not hold for higher shells. Returns a
    stats dict and warns (to stderr) on irregular degree."""
    if len(set(edges)) != len(edges):
        raise RuntimeError("Duplicate edges found in neighbour-shell graph.")
    degree = [0] * n_sites
    for i, j in edges:
        if i == j:
            raise RuntimeError(
                f"Self-edge found: {(i, j)}. The cluster is too small for a "
                "simple-graph representation of this neighbour shell; "
                "increase --Nx/--Ny/--Nz."
            )
        if not (0 <= i < n_sites and 0 <= j < n_sites):
            raise RuntimeError(f"Edge out of range: {(i, j)}")
        degree[i] += 1
        degree[j] += 1
    dmin, dmax = (min(degree), max(degree)) if degree else (0, 0)
    if dmin != dmax:
        print(
            f"# warning: irregular neighbour-shell degree (min={dmin}, max={dmax}); "
            "expected near open boundaries, but can also indicate a too-small cluster",
            file=sys.stderr,
        )
    return {
        "n_components": float(count_components(n_sites, edges)),
        "degree_min": float(dmin),
        "degree_max": float(dmax),
        "degree_mean": (2.0 * len(edges) / n_sites) if n_sites else 0.0,
    }


def print_edge_list(edges: Sequence[Edge], header_lines: Sequence[str] = ()) -> None:
    """Whitespace ``u v`` edge list (0-indexed), directly consumable as a
    bandwidth-certifier ``--j2-file``. Header lines are emitted as ``#`` comments."""
    for line in header_lines:
        print(f"# {line}")
    for i, j in edges:
        print(f"{i} {j}")


def validate_supercell(
    lattice: Lattice,
    supercell: Tuple[IVec3, IVec3, IVec3],
    coords: List[Vec3],
    edges: List[Edge],
) -> None:
    matrix = [list(v) for v in supercell]
    determinant = det3(matrix)

    if determinant == 0:
        raise RuntimeError("Supercell determinant is zero.")

    n_cells = abs(determinant)
    expected_sites = lattice.n_basis * n_cells

    if len(coords) != expected_sites:
        raise RuntimeError(
            f"Unexpected site count for supercell: got {len(coords)}, "
            f"expected {expected_sites} = {lattice.n_basis} * abs(det)."
        )

    validate_graph(lattice, coords, edges)


def build_diagonal_plot_segments(
    lattice: Lattice,
    coords: Sequence[Vec3],
    Nx: int,
    Ny: int,
    Nz: int,
    a: float = 1.0,
) -> List[PlotSegment]:
    segments: Dict[Edge, PlotSegment] = {}

    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                for s1, s2, (dx, dy, dz) in lattice.bonds:
                    raw_target = (x + dx, y + dy, z + dz)
                    wrapped_target = lattice.wrap_cell(*raw_target, Nx, Ny, Nz)
                    if wrapped_target is None:
                        continue

                    i = lattice.site_index(x, y, z, s1, Nx, Ny, Nz)
                    j = lattice.site_index(*wrapped_target, s2, Nx, Ny, Nz)
                    key = tuple(sorted((i, j)))

                    wrap_values = []
                    for raw_value, wrapped_value, size, periodic in zip(
                        raw_target,
                        wrapped_target,
                        (Nx, Ny, Nz),
                        lattice.pbc,
                    ):
                        if periodic:
                            delta = raw_value - wrapped_value
                            wrap_values.append(0 if delta == 0 else int(delta // size))
                        else:
                            wrap_values.append(0)

                    wrap = (wrap_values[0], wrap_values[1], wrap_values[2])

                    if key in segments and segments[key].wrap == (0, 0, 0):
                        continue

                    if wrap != (0, 0, 0):
                        end_unscaled = basis_position(raw_target, s2, lattice.basis, lattice.bravais)
                        end = (a * end_unscaled[0], a * end_unscaled[1], a * end_unscaled[2])
                    else:
                        end = coords[j]

                    segments[key] = PlotSegment(start=coords[i], end=end, wrap=wrap, sites=(i, j))

    return list(segments.values())


def enumerate_supercell_representatives(vectors: Sequence[IVec3]) -> List[IVec3]:
    if len(vectors) != 3:
        raise ValueError("Need exactly three supercell vectors.")

    S = [list(v) for v in vectors]
    det_s = det3(S)
    n_cells = abs(det_s)

    if n_cells == 0:
        raise ValueError("Supercell vectors are linearly dependent.")

    ST = [[S[j][i] for j in range(3)] for i in range(3)]
    inv_ST = inv3_fraction(ST)

    def equivalent(a_cell: IVec3, b_cell: IVec3) -> bool:
        diff = (
            a_cell[0] - b_cell[0],
            a_cell[1] - b_cell[1],
            a_cell[2] - b_cell[2],
        )
        coeffs = matvec_fraction(inv_ST, diff)
        return all(c.denominator == 1 for c in coeffs)

    reps: List[IVec3] = [(0, 0, 0)]
    queue: List[IVec3] = [(0, 0, 0)]
    steps: List[IVec3] = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]

    def find_rep(cell: IVec3) -> int | None:
        for idx, rep in enumerate(reps):
            if equivalent(cell, rep):
                return idx
        return None

    while queue and len(reps) < n_cells:
        r = queue.pop(0)
        for dx, dy, dz in steps:
            cand = (r[0] + dx, r[1] + dy, r[2] + dz)
            if find_rep(cand) is None:
                reps.append(cand)
                queue.append(cand)
                if len(reps) == n_cells:
                    break

    if len(reps) != n_cells:
        raise RuntimeError(
            f"Failed to enumerate quotient lattice. "
            f"Found {len(reps)} representatives, expected {n_cells}."
        )

    return reps


def build_supercell_plot_segments(
    lattice: Lattice,
    coords: Sequence[Vec3],
    supercell: Sequence[IVec3],
    a: float = 1.0,
) -> List[PlotSegment]:
    reps = enumerate_supercell_representatives(supercell)

    S = [list(v) for v in supercell]
    ST = [[S[j][i] for j in range(3)] for i in range(3)]
    inv_ST = inv3_fraction(ST)

    def equivalent(a_cell: IVec3, b_cell: IVec3) -> bool:
        diff = (
            a_cell[0] - b_cell[0],
            a_cell[1] - b_cell[1],
            a_cell[2] - b_cell[2],
        )
        coeffs = matvec_fraction(inv_ST, diff)
        return all(c.denominator == 1 for c in coeffs)

    def find_rep(cell: IVec3) -> int:
        for idx, rep in enumerate(reps):
            if equivalent(cell, rep):
                return idx
        raise RuntimeError(f"Could not wrap cell {cell}.")

    def site_index(cell: IVec3, s: int) -> int:
        return lattice.n_basis * find_rep(cell) + s

    segments: Dict[Edge, PlotSegment] = {}

    for cell in reps:
        x, y, z = cell
        for s1, s2, (dx, dy, dz) in lattice.bonds:
            target_cell = (x + dx, y + dy, z + dz)
            rep_index = find_rep(target_cell)
            wrapped_rep = reps[rep_index]

            i = site_index(cell, s1)
            j = site_index(target_cell, s2)
            key = tuple(sorted((i, j)))

            wrap = (
                target_cell[0] - wrapped_rep[0],
                target_cell[1] - wrapped_rep[1],
                target_cell[2] - wrapped_rep[2],
            )

            if key in segments and segments[key].wrap == (0, 0, 0):
                continue

            if wrap != (0, 0, 0):
                end_unscaled = basis_position(target_cell, s2, lattice.basis, lattice.bravais)
                end = (a * end_unscaled[0], a * end_unscaled[1], a * end_unscaled[2])
            else:
                end = coords[j]

            segments[key] = PlotSegment(start=coords[i], end=end, wrap=wrap, sites=(i, j))

    return list(segments.values())


def plot_lattice(
    coords: Sequence[Vec3],
    segments: Sequence[PlotSegment],
    output_stem: str,
    dpi: int = 300,
    site_labels: Sequence[str] | None = None,
    supersite_blocks: Sequence[Sequence[int]] | None = None,
    site_marker_size: float = 20.0,
    legend_fontsize: float | None = None,
) -> Tuple[Path, Path]:
    # Plotting is optional: matplotlib lives in the "plot" extra. If it is not
    # installed, skip rendering (the CLI still prints coords/edges) instead of
    # crashing.
    try:
        import matplotlib
    except ImportError:
        print("matplotlib not installed; skipping plot "
              "(pip install 'vmps_geometry[plot]')", file=sys.stderr)
        stem = Path(output_stem)
        return stem.with_suffix(".png"), stem.with_suffix(".pdf")

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Bare stems go into the repo's single plots/ folder (the same one the
    # analysis tools use), located by walking up to the pyproject.toml so every
    # tool writes to one place regardless of the caller's cwd. Absolute stems or
    # stems that already name a directory are left as-is.
    output_path = Path(output_stem)
    if not output_path.is_absolute() and output_path.parent == Path("."):
        root = Path(__file__).resolve()
        for anc in root.parents:
            if (anc / "pyproject.toml").is_file():
                root = anc
                break
        else:
            root = Path.cwd()
        output_path = root / "plots" / output_path
    elif not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    png_path.parent.mkdir(parents=True, exist_ok=True)

    all_points = list(coords) + [segment.end for segment in segments]
    is_2d = all(abs(point[2]) < 1.0e-12 for point in all_points)

    def periodic_color(wrap: IVec3) -> str:
        if wrap[0] != 0:
            return "#55B6AD"
        if wrap[1] != 0:
            return "#E27767"
        if wrap[2] != 0:
            return "#C7BBFF"
        return "black"

    has_wrap_x = any(segment.wrap[0] != 0 for segment in segments)
    has_wrap_y = any(segment.wrap[1] != 0 for segment in segments)
    has_wrap_z = any(segment.wrap[2] != 0 for segment in segments)

    legend_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.5, label="internal bond"),
    ]
    if has_wrap_x:
        legend_handles.append(
            Line2D([0], [0], color="#55B6AD", linestyle="--", linewidth=1.5, label="periodic x-bond")
        )
    if has_wrap_y:
        legend_handles.append(
            Line2D([0], [0], color="#E27767", linestyle="--", linewidth=1.5, label="periodic y-bond")
        )
    if not is_2d and has_wrap_z:
        legend_handles.append(
            Line2D([0], [0], color="#C7BBFF", linestyle="--", linewidth=1.5, label="periodic z-bond")
        )

    if is_2d:
        figsize = (8.0, 8.0)
        if site_labels is not None or supersite_blocks is not None:
            xs = [point[0] for point in all_points]
            ys = [point[1] for point in all_points]
            spacing = _plot_site_spacing(coords)
            if spacing > 0.0:
                scale = 0.75 if supersite_blocks is not None else 0.55
                figsize = (
                    max(8.0, 2.0 + scale * (max(xs) - min(xs)) / spacing),
                    max(6.0, 2.0 + scale * (max(ys) - min(ys)) / spacing),
                )
        fig, ax = plt.subplots(figsize=figsize)

        for segment in segments:
            color = "black" if segment.wrap == (0, 0, 0) else periodic_color(segment.wrap)
            linestyle = "-" if segment.wrap == (0, 0, 0) else "--"
            ax.plot(
                [segment.start[0], segment.end[0]],
                [segment.start[1], segment.end[1]],
                color=color,
                linestyle=linestyle,
                linewidth=1.5,
                zorder=1,
            )

        ax.scatter(
            [r[0] for r in coords],
            [r[1] for r in coords],
            s=site_marker_size,
            c="black",
            zorder=2,
        )
        if supersite_blocks is not None:
            _draw_supersite_blocks_2d(ax, coords, segments, supersite_blocks)
        if site_labels is not None:
            spacing = _plot_site_spacing(coords)
            offset = 0.07 * spacing
            for label, point in zip(site_labels, coords):
                ax.text(
                    point[0] + offset,
                    point[1] + offset,
                    label,
                    fontsize=14,
                    color="black",
                    zorder=3,
                )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.legend(handles=legend_handles, loc="best", fontsize=legend_fontsize)
        fig.tight_layout()
    else:
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")

        for segment in segments:
            color = "black" if segment.wrap == (0, 0, 0) else periodic_color(segment.wrap)
            linestyle = "-" if segment.wrap == (0, 0, 0) else "--"
            ax.plot(
                [segment.start[0], segment.end[0]],
                [segment.start[1], segment.end[1]],
                [segment.start[2], segment.end[2]],
                color=color,
                linestyle=linestyle,
                linewidth=1.2,
            )

        ax.scatter(
            [r[0] for r in coords],
            [r[1] for r in coords],
            [r[2] for r in coords],
            s=site_marker_size,
            c="black",
            depthshade=False,
        )
        if supersite_blocks is not None:
            _draw_supersite_blocks_3d(ax, coords, segments, supersite_blocks)
        if site_labels is not None:
            for label, point in zip(site_labels, coords):
                ax.text(
                    point[0],
                    point[1],
                    point[2],
                    label,
                    fontsize=14,
                    color="black",
                )

        xs = [point[0] for point in all_points]
        ys = [point[1] for point in all_points]
        zs = [point[2] for point in all_points]
        xmid = 0.5 * (min(xs) + max(xs))
        ymid = 0.5 * (min(ys) + max(ys))
        zmid = 0.5 * (min(zs) + max(zs))
        span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
        half = 0.55 * span if span > 0 else 0.5
        ax.set_xlim(xmid - half, xmid + half)
        ax.set_ylim(ymid - half, ymid + half)
        ax.set_zlim(zmid - half, zmid + half)

        try:
            ax.set_box_aspect((1, 1, 1))
        except AttributeError:
            pass

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend(handles=legend_handles, loc="best", fontsize=legend_fontsize)
        fig.tight_layout()

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


def load_plot_supersite_blocks(path_text: str, n_sites: int) -> List[List[int]]:
    """Load and validate supersite blocks in their chain order for plotting."""
    path = resolve_geometry_path(path_text)
    if path.suffix.lower() == ".txt":
        marker = "blocks in chain order:"
        line = next(
            (
                line
                for line in path.read_text(encoding="utf-8").splitlines()
                if marker in line
            ),
            None,
        )
        if line is None:
            raise ValueError(f"Supersite assignment text {path} must contain {marker!r}")
        blocks = ast.literal_eval(line.split(marker, 1)[1].strip())
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        blocks = payload.get("chain_order")

    if not isinstance(blocks, list) or not blocks:
        raise ValueError(f"Supersite block file {path} must define a nonempty chain_order")

    normalized: List[List[int]] = []
    seen: Dict[int, int] = {}
    for supersite, block in enumerate(blocks):
        if not isinstance(block, list) or not block:
            raise ValueError(f"Supersite {supersite} must contain at least one site")
        normalized_block = [int(site) for site in block]
        for site in normalized_block:
            if site < 0 or site >= n_sites:
                raise ValueError(
                    f"Supersite {supersite} contains site {site}, outside 0..{n_sites - 1}"
                )
            if site in seen:
                raise ValueError(
                    f"Site {site} occurs in both supersite {seen[site]} and supersite {supersite}"
                )
            seen[site] = supersite
        normalized.append(normalized_block)

    missing = sorted(set(range(n_sites)) - set(seen))
    if missing:
        raise ValueError(f"Supersite block file {path} is missing sites: {missing[:8]}")
    return normalized


def supersite_display_points(
    coords: Sequence[Vec3],
    segments: Sequence[PlotSegment],
    block: Sequence[int],
) -> List[Vec3]:
    """Place a connected block in one periodic image for a compact outline."""
    block_set = set(block)
    displayed: Dict[int, Vec3] = {int(block[0]): coords[int(block[0])]}
    block_segments = [
        segment
        for segment in segments
        if segment.sites is not None
        and segment.sites[0] in block_set
        and segment.sites[1] in block_set
    ]

    changed = True
    while changed:
        changed = False
        for segment in block_segments:
            if segment.sites is None:
                continue
            i, j = segment.sites
            if i in displayed and j not in displayed:
                shift = tuple(displayed[i][axis] - segment.start[axis] for axis in range(3))
                displayed[j] = tuple(
                    segment.end[axis] + shift[axis] for axis in range(3)
                )  # type: ignore[assignment]
                changed = True
            elif j in displayed and i not in displayed:
                shift = tuple(displayed[j][axis] - segment.end[axis] for axis in range(3))
                displayed[i] = tuple(
                    segment.start[axis] + shift[axis] for axis in range(3)
                )  # type: ignore[assignment]
                changed = True

    return [displayed.get(int(site), coords[int(site)]) for site in block]


def _plot_site_spacing(coords: Sequence[Vec3]) -> float:
    nearest = []
    for i, point in enumerate(coords):
        distances = [sqrt(dist2(point, other)) for j, other in enumerate(coords) if i != j]
        if distances:
            nearest.append(min(distances))
    if not nearest:
        return 1.0
    nearest.sort()
    return nearest[len(nearest) // 2]


def _draw_supersite_blocks_2d(ax, coords, segments, blocks) -> None:
    from matplotlib.patches import Ellipse

    spacing = _plot_site_spacing(coords)
    padding = 0.28 * spacing
    color = "tab:blue"
    for supersite, block in enumerate(blocks):
        points = supersite_display_points(coords, segments, block)
        cx = sum(point[0] for point in points) / len(points)
        cy = sum(point[1] for point in points) / len(points)
        xx = sum((point[0] - cx) ** 2 for point in points)
        yy = sum((point[1] - cy) ** 2 for point in points)
        xy = sum((point[0] - cx) * (point[1] - cy) for point in points)
        angle = 0.5 * atan2(2.0 * xy, xx - yy) if len(points) > 1 else 0.0
        ca, sa = cos(angle), sin(angle)
        along = [(point[0] - cx) * ca + (point[1] - cy) * sa for point in points]
        across = [-(point[0] - cx) * sa + (point[1] - cy) * ca for point in points]
        radius_along = max((abs(value) for value in along), default=0.0) + padding
        radius_across = max((abs(value) for value in across), default=0.0) + padding
        ax.add_patch(
            Ellipse(
                (cx, cy),
                2.0 * radius_along,
                2.0 * radius_across,
                angle=degrees(angle),
                fill=False,
                edgecolor=color,
                linewidth=1.0,
                zorder=2.5,
            )
        )
        label_offset = radius_across + 0.08 * spacing
        ax.text(
            cx - label_offset * sa,
            cy + label_offset * ca,
            f"{supersite}",
            fontsize=18,
            color=color,
            ha="center",
            va="bottom",
            zorder=4,
        )


def _draw_supersite_blocks_3d(ax, coords, segments, blocks) -> None:
    spacing = _plot_site_spacing(coords)
    color = "#245B78"
    angles = [2.0 * pi * index / 48 for index in range(49)]
    for supersite, block in enumerate(blocks):
        points = supersite_display_points(coords, segments, block)
        center = tuple(sum(point[axis] for point in points) / len(points) for axis in range(3))
        radius = max(sqrt(dist2(center, point)) for point in points) + 0.28 * spacing
        for axes in ((0, 1), (0, 2), (1, 2)):
            ring = []
            for angle in angles:
                point = list(center)
                point[axes[0]] += radius * cos(angle)
                point[axes[1]] += radius * sin(angle)
                ring.append(point)
            ax.plot(
                [point[0] for point in ring],
                [point[1] for point in ring],
                [point[2] for point in ring],
                color=color,
                linewidth=0.7,
                alpha=0.65,
            )
        ax.text(
            center[0],
            center[1],
            center[2] + 1.1 * radius,
            f"{supersite}",
            fontsize=14,
            color=color,
        )


def _load_permutation_module(path: Path):
    spec = importlib.util.spec_from_file_location("_vmps_plot_permutations", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load permutation file: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _permutation_lookup_candidates(
    args: argparse.Namespace,
    n_sites: int,
    plot_stem: str,
    tilted_key: str | None = None,
) -> List[str]:
    candidates = [args.lattice, Path(plot_stem).name, f"{args.lattice}{n_sites}"]

    if args.lattice == "pyrochlore" and tilted_key is not None and tilted_key != "custom":
        candidates.insert(0, f"pyrochlore{tilted_key}")

    seen = set()
    unique = []
    for key in candidates:
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def load_plot_permutation_labels(
    path_text: str,
    args: argparse.Namespace,
    n_sites: int,
    plot_stem: str,
    tilted_key: str | None = None,
) -> Tuple[List[str], str]:
    path = Path(path_text)
    module = _load_permutation_module(path)
    permutations = getattr(module, "CUSTOM_PERMUTATIONS", None)
    if not isinstance(permutations, dict):
        raise ValueError(f"{path} does not define a CUSTOM_PERMUTATIONS dictionary.")

    candidates = _permutation_lookup_candidates(args, n_sites, plot_stem, tilted_key=tilted_key)
    key = next((candidate for candidate in candidates if candidate in permutations), None)
    if key is None:
        available = ", ".join(sorted(permutations))
        tried = ", ".join(candidates)
        raise ValueError(
            f"No permutation entry found in {path} for generated cluster. "
            f"Tried: {tried}. Available entries: {available}"
        )

    permutation = permutations[key]
    if isinstance(permutation, dict):
        missing = [i for i in range(n_sites) if i not in permutation]
        if missing:
            raise ValueError(f"Permutation {key!r} is missing site indices: {missing[:8]}")
        labels = [str(permutation[i]) for i in range(n_sites)]
    elif isinstance(permutation, (list, tuple)):
        if len(permutation) != n_sites:
            raise ValueError(
                f"Permutation {key!r} has length {len(permutation)}, expected {n_sites}."
            )
        labels = [str(value) for value in permutation]
    else:
        raise ValueError(f"Permutation {key!r} must be a dict, list, or tuple.")

    return labels, key


def print_compact_edges(edges: List[Edge], per_line: int = 8) -> None:
    print("edges = [")
    for k in range(0, len(edges), per_line):
        chunk = edges[k : k + per_line]
        line = ", ".join(f"({i}, {j})" for i, j in chunk)
        print(f"    {line},")
    print("]")


def print_compact_coords(coords: List[Vec3], per_line: int = 1) -> None:
    print("coords = [")
    for k in range(0, len(coords), per_line):
        chunk = coords[k : k + per_line]
        line = ", ".join(
            f"({r[0]:.16g}, {r[1]:.16g}, {r[2]:.16g})"
            for r in chunk
        )
        print(f"    {line},")
    print("]")


def automatic_plot_stem(
    cluster_name: str,
    n_sites: int,
    Nx: int | None,
    Ny: int | None,
    Nz: int | None,
    coords: Sequence[Vec3],
    tilted_key: str | None = None,
) -> str:
    if Nx is not None and Ny is not None and Nz is not None:
        is_2d = Nz == 1 and all(abs(r[2]) < 1.0e-12 for r in coords)
        if is_2d:
            return f"{cluster_name}{n_sites}_{Nx}x{Ny}"
        return f"{cluster_name}{n_sites}_{Nx}x{Ny}x{Nz}"

    if tilted_key is not None:
        return f"{cluster_name}{n_sites}_{tilted_key}"

    return f"{cluster_name}{n_sites}"


def shell_plot_stem(stem: str, shell: int) -> str:
    """Append a ``_shell<N>`` suffix to a plot stem for neighbour shells > 1."""
    return stem if shell == 1 else f"{stem}_shell{shell}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print coordinates and edge lists for finite lattices and molecules."
    )

    parser.add_argument(
        "lattice",
        choices=[
            "hyperkagome",
            "pyrochlore",
            "trillium",
            "fcc",
            "garnet",
            "kagomeBcyl",
            "kagomeBtorus",
            "kagomeYcyl",
            "kagomeXcyl",
            "kagomeYtorus",
            "kagomeXtorus",
            "triangularYcyl",
            "triangularXcyl",
            "triangularBtorus",
            "triangularYtorus",
            "triangularXtorus",
            "squareCyl",
            "squareTorus",
            *MOLECULE_NAMES,
        ],
        help="Lattice or molecule type.",
    )

    parser.add_argument("--Nx", type=int)
    parser.add_argument("--Ny", type=int)
    parser.add_argument("--Nz", type=int)

    parser.add_argument(
        "--tilted",
        type=str,
        default=None,
        help=(
            "Known tilted pyrochlore cluster: 48a, 48b, 48c, 48d, 64, 128, "
            "or pyrochlore48a, etc. Ignored for non-pyrochlore lattices."
        ),
    )

    parser.add_argument(
        "--supercell",
        type=str,
        default=None,
        help=(
            "Custom pyrochlore supercell as 'a,b,c;d,e,f;g,h,i'. "
            "Example: --supercell '2,1,0;0,3,0;0,0,4'. "
            "Only used for pyrochlore."
        ),
    )

    parser.add_argument(
        "--a",
        type=float,
        default=1.0,
        help="Overall lattice constant multiplying Cartesian coordinates.",
    )

    parser.add_argument(
        "--trillium-u",
        type=float,
        default=0.138,
        help="Internal coordinate u for the trillium/B20 basis.",
    )

    parser.add_argument(
        "--edges-per-line",
        type=int,
        default=8,
        help="Number of edge pairs printed per line.",
    )

    parser.add_argument(
        "--coords-per-line",
        type=int,
        default=1,
        help="Number of coordinate triples printed per line.",
    )

    parser.add_argument(
        "--format",
        choices=["python", "json", "edgelist"],
        default="python",
        help=(
            "Output format. 'edgelist' prints a whitespace 'u v' 0-indexed edge "
            "list (a bandwidth-certifier --j2-file)."
        ),
    )

    parser.add_argument(
        "--neighbor-shell",
        "--shell",
        dest="neighbor_shell",
        type=int,
        default=1,
        help=(
            "Which real-space neighbour shell to emit as edges: 1 = nearest "
            "neighbour (default), 2 = next-nearest (NNN), etc. Coordinates are "
            "unchanged, so shell 1 and shell 2 of the same cluster share site "
            "indexing and are usable as J1 and J2."
        ),
    )

    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip degree, edge-count, self-edge, and connected-component validation.",
    )

    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=300,
        help="DPI used when saving the PNG plot.",
    )

    parser.add_argument(
        "--plot-permutation-file",
        type=str,
        default=None,
        help=(
            "Optional path to a permutation file such as permutations_sat_bw.py. "
            "When set, the plot annotates each generated vertex with its "
            "optimized enumeration index from CUSTOM_PERMUTATIONS."
        ),
    )

    parser.add_argument(
        "--supersite-blocks",
        "--plot-supersite-blocks",
        dest="plot_supersite_blocks",
        type=str,
        default=None,
        help=(
            "Optional supersite assignment text or JSON layout. The plot outlines sites "
            "belonging to each block and labels supersites in chain order."
        ),
    )

    return parser.parse_args()


def resolve_output_edges(args, coords, periods, nn_edges):
    """Return ``(edges_to_emit, shell_meta)``. For ``--neighbor-shell 1`` this is
    the nearest-neighbour edges unchanged (``shell_meta`` is None). For higher
    shells it computes the real-space shell graph from ``coords``/``periods`` and
    (unless ``--no-validate``) runs the relaxed shell validator."""
    if args.neighbor_shell == 1:
        return nn_edges, None
    edges, shell_distances = neighbor_shell_edges(coords, periods, args.neighbor_shell)
    stats = None
    if not args.no_validate:
        stats = validate_neighbor_shell_graph(len(coords), edges)
    return edges, {
        "shell": args.neighbor_shell,
        "distances": shell_distances,
        "stats": stats,
    }


def shell_header_lines(shell_meta) -> List[str]:
    """Comment lines describing a higher neighbour shell (empty for shell 1)."""
    if shell_meta is None:
        return []
    dists = ", ".join(f"{d:.12g}" for d in shell_meta["distances"])
    lines = [
        f"neighbor_shell = {shell_meta['shell']}",
        f"shell_distances = [{dists}]",
    ]
    stats = shell_meta.get("stats")
    if stats is not None:
        lines.append(
            "shell_degree min/max/mean = "
            f"{int(stats['degree_min'])}/{int(stats['degree_max'])}/"
            f"{stats['degree_mean']:.6g}"
        )
        lines.append(f"shell_n_components = {int(stats['n_components'])}")
    return lines


def shell_json_fields(shell_meta) -> Dict:
    """JSON payload fields describing a higher neighbour shell (empty for shell 1)."""
    if shell_meta is None:
        return {}
    out: Dict = {
        "neighbor_shell": shell_meta["shell"],
        "shell_distances": shell_meta["distances"],
    }
    if shell_meta.get("stats") is not None:
        out["shell_stats"] = shell_meta["stats"]
    return out


def main() -> None:
    args = parse_args()

    if args.neighbor_shell < 1:
        raise ValueError("--neighbor-shell must be a positive integer (1 = nearest neighbour).")

    if args.edges_per_line <= 0:
        raise ValueError("--edges-per-line must be positive.")

    if args.coords_per_line <= 0:
        raise ValueError("--coords-per-line must be positive.")

    if args.plot_dpi <= 0:
        raise ValueError("--plot-dpi must be positive.")

    if args.lattice in MOLECULE_NAMES:
        if args.Nx is not None or args.Ny is not None or args.Nz is not None:
            raise ValueError("Molecules do not use --Nx, --Ny, or --Nz.")
        if args.supercell is not None or args.tilted is not None:
            raise ValueError("Molecules do not use --supercell or --tilted.")
        if args.neighbor_shell != 1:
            raise ValueError("Molecular Schlegel coordinates only define neighbor shell 1.")

        edges = list(CLUSTER_EDGES[args.lattice])
        n_sites = max(max(edge) for edge in edges) + 1
        coords = build_schlegel_coordinates(n_sites, edges, a=args.a)
        plot_segments = build_edge_plot_segments(coords, edges)
        plot_stem = args.lattice
        site_labels = None
        permutation_key = None
        if args.plot_permutation_file is not None:
            site_labels, permutation_key = load_plot_permutation_labels(
                args.plot_permutation_file,
                args,
                n_sites,
                plot_stem,
            )
        supersite_blocks = None
        if args.plot_supersite_blocks is not None:
            supersite_blocks = load_plot_supersite_blocks(
                args.plot_supersite_blocks, n_sites
            )
        png_path, pdf_path = plot_lattice(
            coords,
            plot_segments,
            plot_stem,
            dpi=args.plot_dpi,
            site_labels=site_labels,
            supersite_blocks=supersite_blocks,
        )

        if args.format == "edgelist":
            print_edge_list(edges, [
                f"molecule = {args.lattice}",
                f"n_sites = {n_sites}",
                f"n_edges = {len(edges)}",
            ])
            return

        if args.format == "json":
            payload = {
                "molecule": args.lattice,
                "projection": "Schlegel (Tutte embedding)",
                "n_sites": n_sites,
                "n_edges": len(edges),
                "coords": coords,
                "edges": edges,
                "plot_png": str(png_path),
                "plot_pdf": str(pdf_path),
            }
            if permutation_key is not None:
                payload["plot_permutation_file"] = args.plot_permutation_file
                payload["plot_permutation_key"] = permutation_key
            if supersite_blocks is not None:
                payload["plot_supersite_blocks"] = args.plot_supersite_blocks
                payload["num_supersites"] = len(supersite_blocks)
            print(json.dumps(payload, indent=2))
            return

        print(f"# molecule = {args.lattice}")
        print("# projection = Schlegel (Tutte embedding)")
        print(f"# n_sites = {n_sites}")
        print(f"# n_edges = {len(edges)}")
        if permutation_key is not None:
            print(f"# plot_permutation_file = {args.plot_permutation_file}")
            print(f"# plot_permutation_key = {permutation_key}")
        if supersite_blocks is not None:
            print(f"# plot_supersite_blocks = {args.plot_supersite_blocks}")
            print(f"# num_supersites = {len(supersite_blocks)}")
        print(f"# plot_png = {png_path}")
        print(f"# plot_pdf = {pdf_path}")
        print()
        print_compact_coords(coords, per_line=args.coords_per_line)
        print()
        print_compact_edges(edges, per_line=args.edges_per_line)
        return

    lattice = get_lattice(args.lattice, args.trillium_u)

    if args.lattice in KAGOME_STRIP_VARIANTS or args.lattice in TRIANGULAR_STRIP_VARIANTS:
        if args.Nx is None or args.Ny is None or args.Nz is None:
            raise ValueError("XC/YC strip variants need --Nx --Ny --Nz, with --Nz 1.")
        if args.Nz != 1:
            raise ValueError("Strip variants are two-dimensional; use --Nz 1.")
        if args.supercell is not None or args.tilted is not None:
            raise ValueError("Strip variants do not use --supercell or --tilted.")

        if args.lattice in KAGOME_STRIP_VARIANTS:
            kind, periodic_x = kagome_variant_kind_and_periodicity(args.lattice)
            coords, edges, period_x, period_y = build_kagome_strip_edges(
                kind, args.Nx, args.Ny, periodic_x, a=args.a
            )
            expected_degree_label = 4 if periodic_x else "2-4 with open x boundaries"
        else:
            kind, periodic_x = triangular_variant_kind_and_periodicity(args.lattice)
            coords, edges, period_x, period_y = build_triangular_strip_edges(
                kind, args.Nx, args.Ny, periodic_x, a=args.a
            )
            expected_degree_label = 6 if periodic_x else "2-6 with open x boundaries"

        if not args.no_validate:
            if args.lattice in KAGOME_STRIP_VARIANTS:
                validate_kagome_strip(args.lattice, coords, edges, periodic_x)
                validate_kagome_strip_geometry(coords, edges, period_x, period_y, periodic_x, args.a)
            else:
                validate_triangular_strip(args.lattice, coords, edges, periodic_x)
                validate_triangular_strip_geometry(
                    coords, edges, period_x, period_y, periodic_x, args.a
                )

        pbc = (periodic_x, True, False)
        periods = (period_x if periodic_x else None, period_y, None)
        out_edges, shell_meta = resolve_output_edges(args, coords, periods, edges)

        if args.neighbor_shell == 1:
            plot_segments = build_strip_plot_segments(coords, edges, period_x, period_y, periodic_x)
        else:
            plot_segments = build_periodic_plot_segments(coords, out_edges, periods)
        plot_coords, plot_segments, plot_rotation = align_strip_plot_axis(
            coords, plot_segments, period_x
        )
        plot_stem = shell_plot_stem(
            automatic_plot_stem(args.lattice, len(coords), args.Nx, args.Ny, args.Nz, coords),
            args.neighbor_shell,
        )
        site_labels = None
        permutation_key = None
        if args.plot_permutation_file is not None:
            site_labels, permutation_key = load_plot_permutation_labels(
                args.plot_permutation_file,
                args,
                len(coords),
                plot_stem,
            )
        supersite_blocks = None
        if args.plot_supersite_blocks is not None:
            supersite_blocks = load_plot_supersite_blocks(
                args.plot_supersite_blocks, len(coords)
            )
        png_path, pdf_path = plot_lattice(
            plot_coords,
            plot_segments,
            plot_stem,
            dpi=args.plot_dpi,
            site_labels=site_labels,
            supersite_blocks=supersite_blocks,
            site_marker_size=64.0,
            legend_fontsize=18.0,
        )

        if args.format == "edgelist":
            print_edge_list(out_edges, [
                f"lattice = {args.lattice}",
                f"Nx Ny Nz = {args.Nx} {args.Ny} {args.Nz}",
                f"n_sites = {len(coords)}",
                f"neighbor_shell = {args.neighbor_shell}",
                f"n_edges = {len(out_edges)}",
            ])
            return

        if args.format == "json":
            payload = {
                "lattice": args.lattice,
                "kind": kind,
                "n_sites": len(coords),
                "n_edges": len(out_edges),
                "expected_degree": expected_degree_label,
                "expected_components": 1,
                "pbc": pbc,
                "period_x": period_x,
                "period_y": period_y,
                "Nx": args.Nx,
                "Ny": args.Ny,
                "Nz": args.Nz,
                "coords": coords,
                "edges": out_edges,
                "plot_png": str(png_path),
                "plot_pdf": str(pdf_path),
                "plot_rotation_degrees": degrees(plot_rotation),
            }
            payload.update(shell_json_fields(shell_meta))
            if permutation_key is not None:
                payload["plot_permutation_file"] = args.plot_permutation_file
                payload["plot_permutation_key"] = permutation_key
            if supersite_blocks is not None:
                payload["plot_supersite_blocks"] = args.plot_supersite_blocks
                payload["num_supersites"] = len(supersite_blocks)
            print(json.dumps(payload, indent=2))
            return

        print(f"# lattice = {args.lattice}")
        print(f"# convention = paper DMRG {kind}C")
        print(f"# Nx Ny Nz = {args.Nx} {args.Ny} {args.Nz}")
        print(f"# unit_cell_sites = {len(coords) // args.Nx}")
        print(f"# degree = {expected_degree_label}")
        print("# expected_components = 1")
        print(f"# pbc = {pbc}")
        print(f"# period_x = {period_x}")
        print(f"# period_y = {period_y}")
        print(f"# n_sites = {len(coords)}")
        print(f"# n_edges = {len(out_edges)}")
        for line in shell_header_lines(shell_meta):
            print(f"# {line}")
        print(f"# plot_rotation_degrees = {degrees(plot_rotation):.12g}")
        if permutation_key is not None:
            print(f"# plot_permutation_file = {args.plot_permutation_file}")
            print(f"# plot_permutation_key = {permutation_key}")
        if supersite_blocks is not None:
            print(f"# plot_supersite_blocks = {args.plot_supersite_blocks}")
            print(f"# num_supersites = {len(supersite_blocks)}")
        print(f"# plot_png = {png_path}")
        print(f"# plot_pdf = {pdf_path}")
        print()
        print_compact_coords(coords, per_line=args.coords_per_line)
        print()
        print_compact_edges(out_edges, per_line=args.edges_per_line)
        return

    if args.supercell is not None and args.tilted is not None:
        raise ValueError("Use either --supercell or --tilted, not both.")

    if args.supercell is not None and args.lattice != "pyrochlore":
        raise ValueError("--supercell is currently only implemented for pyrochlore.")

    used_tilted = False
    tilted_key = None
    supercell = None

    if args.lattice == "pyrochlore" and args.supercell is not None:
        supercell = parse_supercell(args.supercell)
        coords, edges = lattice.make_supercell(supercell, a=args.a)
        used_tilted = True
        tilted_key = "custom"

        if not args.no_validate:
            validate_supercell(lattice, supercell, coords, edges)

    elif args.lattice == "pyrochlore" and args.tilted is not None:
        tilted_key = canonical_tilted_name(args.tilted)

        if tilted_key not in PYROCHLORE_TILTED_SUPERCELLS:
            allowed = ", ".join(sorted(PYROCHLORE_TILTED_SUPERCELLS))
            raise ValueError(
                f"Unknown pyrochlore tilted cluster {args.tilted!r}. "
                f"Allowed: {allowed}"
            )

        supercell = PYROCHLORE_TILTED_SUPERCELLS[tilted_key]
        coords, edges = lattice.make_supercell(supercell, a=args.a)
        used_tilted = True

        if not args.no_validate:
            validate_supercell(lattice, supercell, coords, edges)

    else:
        if args.Nx is None or args.Ny is None or args.Nz is None:
            raise ValueError("Need --Nx --Ny --Nz unless using pyrochlore --tilted or --supercell.")

        if min(args.Nx, args.Ny, args.Nz) <= 0:
            raise ValueError("--Nx, --Ny, and --Nz must be positive integers.")

        if (
            args.lattice in KAGOME_VARIANTS
            or args.lattice == "triangularBtorus"
            or args.lattice in SQUARE_VARIANTS
        ) and args.Nz != 1:
            raise ValueError("Two-dimensional lattice variants require --Nz 1.")

        coords, edges = lattice.make_diagonal(args.Nx, args.Ny, args.Nz, a=args.a)

        if not args.no_validate:
            validate_graph(lattice, coords, edges)

    if used_tilted:
        periods = supercell_periods(supercell, lattice.bravais, a=args.a)  # type: ignore[arg-type]
    else:
        periods = diagonal_periods(lattice, args.Nx, args.Ny, args.Nz, args.a)
    out_edges, shell_meta = resolve_output_edges(args, coords, periods, edges)

    if args.neighbor_shell != 1:
        plot_segments = build_periodic_plot_segments(coords, out_edges, periods)
    elif used_tilted:
        plot_segments = build_supercell_plot_segments(lattice, coords, supercell, a=args.a)  # type: ignore[arg-type]
    else:
        plot_segments = build_diagonal_plot_segments(
            lattice,
            coords,
            args.Nx,  # type: ignore[arg-type]
            args.Ny,  # type: ignore[arg-type]
            args.Nz,  # type: ignore[arg-type]
            a=args.a,
        )

    plot_stem = shell_plot_stem(
        automatic_plot_stem(
            args.lattice,
            len(coords),
            args.Nx,
            args.Ny,
            args.Nz,
            coords,
            tilted_key=tilted_key,
        ),
        args.neighbor_shell,
    )
    site_labels = None
    permutation_key = None
    if args.plot_permutation_file is not None:
        site_labels, permutation_key = load_plot_permutation_labels(
            args.plot_permutation_file,
            args,
            len(coords),
            plot_stem,
            tilted_key=tilted_key,
        )
    supersite_blocks = None
    if args.plot_supersite_blocks is not None:
        supersite_blocks = load_plot_supersite_blocks(
            args.plot_supersite_blocks, len(coords)
        )
    png_path, pdf_path = plot_lattice(
        coords,
        plot_segments,
        plot_stem,
        dpi=args.plot_dpi,
        site_labels=site_labels,
        supersite_blocks=supersite_blocks,
    )

    if args.format == "edgelist":
        header = [f"lattice = {args.lattice}"]
        if used_tilted:
            header.append(f"tilted = pyrochlore{tilted_key}")
        else:
            header.append(f"Nx Ny Nz = {args.Nx} {args.Ny} {args.Nz}")
        header += [
            f"n_sites = {len(coords)}",
            f"neighbor_shell = {args.neighbor_shell}",
            f"n_edges = {len(out_edges)}",
        ]
        print_edge_list(out_edges, header)
        return

    if args.format == "json":
        payload = {
            "lattice": args.lattice,
            "n_sites": len(coords),
            "n_edges": len(out_edges),
            "expected_degree": expected_degree(lattice),
            "expected_components": lattice.expected_components,
            "pbc": lattice.pbc,
            "coords": coords,
            "edges": out_edges,
            "plot_png": str(png_path),
            "plot_pdf": str(pdf_path),
        }
        payload.update(shell_json_fields(shell_meta))

        if used_tilted:
            payload["tilted"] = tilted_key
            payload["supercell"] = supercell
            payload["determinant"] = det3([list(v) for v in supercell])  # type: ignore
            payload["n_primitive_cells"] = abs(det3([list(v) for v in supercell]))  # type: ignore
        else:
            payload["Nx"] = args.Nx
            payload["Ny"] = args.Ny
            payload["Nz"] = args.Nz

        if permutation_key is not None:
            payload["plot_permutation_file"] = args.plot_permutation_file
            payload["plot_permutation_key"] = permutation_key
        if supersite_blocks is not None:
            payload["plot_supersite_blocks"] = args.plot_supersite_blocks
            payload["num_supersites"] = len(supersite_blocks)

        print(json.dumps(payload, indent=2))
        return

    print(f"# lattice = {args.lattice}")

    if used_tilted:
        determinant = det3([list(v) for v in supercell])  # type: ignore
        print(f"# tilted = pyrochlore{tilted_key}")
        print(f"# supercell = {supercell}")
        print(f"# determinant = {determinant}")
        print(f"# n_primitive_cells = {abs(determinant)}")
    else:
        print(f"# Nx Ny Nz = {args.Nx} {args.Ny} {args.Nz}")

    print(f"# n_basis = {lattice.n_basis}")
    print(f"# degree = {expected_degree(lattice)}")
    print(f"# expected_components = {lattice.expected_components}")
    print(f"# pbc = {lattice.pbc}")
    print(f"# n_sites = {len(coords)}")
    print(f"# n_edges = {len(out_edges)}")
    for line in shell_header_lines(shell_meta):
        print(f"# {line}")
    if permutation_key is not None:
        print(f"# plot_permutation_file = {args.plot_permutation_file}")
        print(f"# plot_permutation_key = {permutation_key}")
    if supersite_blocks is not None:
        print(f"# plot_supersite_blocks = {args.plot_supersite_blocks}")
        print(f"# num_supersites = {len(supersite_blocks)}")
    print(f"# plot_png = {png_path}")
    print(f"# plot_pdf = {pdf_path}")
    print()

    print_compact_coords(coords, per_line=args.coords_per_line)
    print()
    print_compact_edges(out_edges, per_line=args.edges_per_line)


if __name__ == "__main__":
    main()
