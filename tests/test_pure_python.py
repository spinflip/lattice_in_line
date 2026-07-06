"""Fast, solver-free unit tests for the core pure-Python routines."""
import numpy as np

import vmps_geometry.bandwidth_certifier as bc
import vmps_geometry.cluster_generator as cg
import vmps_geometry.fiedler_ordering as fo


def test_neighbor_shell_edges_square_shells():
    lattice = cg.get_lattice("squareTorus", 0.138)
    coords, edges = lattice.make_diagonal(4, 4, 1)
    periods = cg.diagonal_periods(lattice, 4, 4, 1, 1.0)

    nn, d1 = cg.neighbor_shell_edges(coords, periods, 1)
    nnn, d2 = cg.neighbor_shell_edges(coords, periods, 2)
    assert {(min(i, j), max(i, j)) for i, j in nn} == set(edges)  # shell 1 == NN
    assert abs(d1[0] - 1.0) < 1e-9
    assert abs(d2[0] - 2 ** 0.5) < 1e-9  # NNN = the diagonal (sqrt 2) shell


def test_sa_lex_stays_feasible_and_does_not_worsen_j2():
    n = 6
    e1 = [(i, i + 1) for i in range(5)]          # a path (dominant coupling)
    e2 = [(0, 3), (2, 5)]                          # secondary, longer-range edges
    init = list(range(1, n + 1))                  # identity, 1-based
    k1 = max(abs(init[u] - init[v]) for u, v in e1)
    start = max(abs(init[u] - init[v]) for u, v in e2)

    bw, lab = bc.sa_lex(n, e1, k1, e2, init, seed=0, t_budget=0.3)
    assert sorted(lab) == list(range(1, n + 1))               # still a permutation
    assert all(abs(lab[u] - lab[v]) <= k1 for u, v in e1)     # J1 cap respected
    assert bw <= start                                        # J2 bandwidth not worse


def test_fiedler_order_separates_weakly_linked_clusters():
    # Two triangles joined by one weak edge; the Fiedler order should keep each
    # triangle contiguous (i.e. all of one before all of the other).
    W = np.zeros((6, 6))
    for i, j in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]:
        W[i, j] = W[j, i] = 1.0
    W[2, 3] = W[3, 2] = 0.05

    order = fo.fiedler_order(W)
    assert sorted(order) == list(range(6))  # a valid permutation
    pos = {v: p for p, v in enumerate(order)}
    left = [pos[v] for v in (0, 1, 2)]
    right = [pos[v] for v in (3, 4, 5)]
    assert max(left) < min(right) or max(right) < min(left)


def test_make_weights_concurrence_zeroes_ferromagnetic_pairs():
    # For spin-1/2 <S_i.S_j>, concurrence = max(0, -2c - 1/2): zero for c >= -1/4.
    C = np.array([[0.0, 0.2, -0.4],
                  [0.2, 0.0, -0.1],
                  [-0.4, -0.1, 0.0]])
    W = fo.make_weights(C, "concurrence")
    assert W[0, 1] == 0.0                    # FM pair -> no entanglement weight
    assert W[1, 2] == 0.0                    # weak AFM (c=-0.1 >= -1/4) -> zero
    assert abs(W[0, 2] - (-2 * -0.4 - 0.5)) < 1e-12  # strong AFM -> 0.3
