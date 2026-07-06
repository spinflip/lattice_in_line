"""Cluster-generator regression tests: representative lattices generate valid
graphs, and the garnet NN-disconnected / NNN-connects invariant holds."""
import vmps_geometry.cluster_generator as cg

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
