"""Cluster-generator regression tests: representative lattices generate valid
graphs, and the garnet NN-disconnected / NNN-connects invariant holds."""
from argparse import Namespace
from pathlib import Path

import lattice_in_line.cluster_generator as cg

# (lattice, Nx, Ny, Nz, expected n_sites) — all bulk (make_diagonal) lattices.
REPRESENTATIVE = [
    ("garnet", 2, 2, 2, 192),
    ("fcc", 3, 3, 3, 27),
    ("pyrochlore", 2, 2, 2, 32),
    ("hyperkagome", 2, 2, 2, 96),
    ("kagomeBtorus", 4, 4, 1, 48),
    ("triangularBtorus", 4, 4, 1, 16),
    ("squareTorus", 4, 4, 1, 16),
    ("squareCyl", 5, 4, 1, 20),
]


def test_representative_lattices_generate_valid_graphs():
    for name, nx, ny, nz, exp_n in REPRESENTATIVE:
        lattice = cg.get_lattice(name, 0.138)
        coords, edges = lattice.make_diagonal(nx, ny, nz)
        assert len(coords) == exp_n, name
        # validate_graph raises on self/dup/out-of-range edges, wrong degree, or
        # wrong component count — so a clean pass is a strong regression check.
        cg.validate_graph(lattice, coords, edges)


def test_garnet_nn_disconnected_and_nnn_connects():
    lattice = cg.get_lattice("garnet", 0.138)
    coords, edges = lattice.make_diagonal(2, 2, 2)
    n = len(coords)
    assert cg.count_components(n, edges) == 2  # two interpenetrating sublattices
    periods = cg.diagonal_periods(lattice, 2, 2, 2, 1.0)
    nnn, _ = cg.neighbor_shell_edges(coords, periods, 2)
    assert cg.count_components(n, list(edges) + nnn) == 1  # NNN bridges them


def test_triangular_nnn_is_sqrt3_shell_degree_6():
    lattice = cg.get_lattice("triangularBtorus", 0.138)
    coords, edges = lattice.make_diagonal(6, 6, 1)
    periods = cg.diagonal_periods(lattice, 6, 6, 1, 1.0)
    nnn, dists = cg.neighbor_shell_edges(coords, periods, 2)
    degree = [0] * len(coords)
    for i, j in nnn:
        degree[i] += 1
        degree[j] += 1
    assert min(degree) == max(degree) == 6
    assert abs(dists[0] - 3 ** 0.5) < 1e-9  # the sqrt(3) shell, not a z-diagonal


def test_c60_schlegel_coordinates_are_planar():
    edges = cg.CLUSTER_EDGES["C60"]
    coords = cg.build_schlegel_coordinates(60, edges)
    assert len(coords) == 60
    assert all(z == 0.0 for _, _, z in coords)
    assert sum(abs(x * x + y * y - 1.0) < 1.0e-12 for x, y, _ in coords) == 5
    assert min(
        cg.dist2(coords[i], coords[j]) ** 0.5
        for i in range(60)
        for j in range(i + 1, 60)
    ) > 0.08

    def orientation(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    for index, (i, j) in enumerate(edges):
        for k, l in edges[index + 1:]:
            if len({i, j, k, l}) < 4:
                continue
            turns = (
                orientation(coords[i], coords[j], coords[k]),
                orientation(coords[i], coords[j], coords[l]),
                orientation(coords[k], coords[l], coords[i]),
                orientation(coords[k], coords[l], coords[j]),
            )
            assert not (turns[0] * turns[1] < 0.0 and turns[2] * turns[3] < 0.0)


def test_c60_plot_permutation_labels_use_molecule_key():
    permutation_file = Path(cg.__file__).with_name("permutations_sat_bw.py")
    labels, key = cg.load_plot_permutation_labels(
        str(permutation_file),
        Namespace(lattice="C60"),
        60,
        "C60",
    )
    assert key == "C60"
    assert labels[5] == "8"
    assert sorted(map(int, labels)) == list(range(60))
