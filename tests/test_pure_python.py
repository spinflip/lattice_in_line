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


def test_ss_intra_report_and_viol():
    # 8-cycle, q=2. Blocking (0,1)(2,3)(4,5)(6,7): every block is a cycle
    # edge -> 4 intra edges, 0 uncovered blocks.
    n = 8
    edges = [(i, (i + 1) % n) for i in range(n)]
    lab_edges = [1, 1, 2, 2, 3, 3, 4, 4]
    assert bc.ss_intra_report(lab_edges, edges, 2) == (4, 0)
    assert bc.ss_viol(lab_edges, edges, 2, 4, True) == 0
    # antipodal blocking (i, i+4): no block is an edge -> 0 intra, 4 uncovered
    lab_anti = [1, 2, 3, 4, 1, 2, 3, 4]
    assert bc.ss_intra_report(lab_anti, edges, 2) == (0, 4)
    assert bc.ss_viol(lab_anti, edges, 2, 0, True) == 4
    assert bc.ss_viol(lab_anti, edges, 2, 2, False) == 2


def test_ss_sa_chain_satisfies_intra_per_block():
    # From a random blocking, the SA (violation-first lexicographic cost)
    # must find a perfect matching along bonds of the 8-cycle.
    n = 8
    edges = [(i, (i + 1) % n) for i in range(n)]
    vi, bw, lab = bc._ss_sa_chain((n, edges, 2, 3, 2.0, None, 0, 0, True))
    assert vi == bc.ss_viol(lab, edges, 2, 0, True)   # reported viol is real
    assert vi == 0
    edge_set = {frozenset(e) for e in edges}
    blocks = {}
    for v, r in enumerate(lab):
        blocks.setdefault(r, []).append(v)
    for vs in blocks.values():                        # every block is a bond
        assert len(vs) == 2 and frozenset(vs) in edge_set


def test_ss_sa_chain_unconstrained_unchanged():
    # constraints off -> viol always 0 and a valid multiplicity-2 labeling
    n = 8
    edges = [(i, (i + 1) % n) for i in range(n)]
    vi, bw, lab = bc._ss_sa_chain((n, edges, 2, 1, 0.3, None, 0, 0, False))
    assert vi == 0
    assert sorted(lab) == sorted(list(range(1, 5)) * 2)
    assert bw == bc.ss_value(lab, edges)


def test_build_cnf_ss_hidden_bond_encodings():
    n = 8
    edges = [(i, (i + 1) % n) for i in range(n)]
    base_c, base_v, base_note = bc.build_cnf_ss(n, edges, 2, 3)
    con_c, con_v, con_note = bc.build_cnf_ss(n, edges, 2, 3,
                                             min_intra=2, intra_per_block=True)
    assert con_v > base_v and len(con_c) > len(base_c)
    assert "intra-block" in con_note and "internal edge" in con_note
    # all literals reference declared variables
    assert all(abs(x) <= con_v for cl in con_c for x in cl)
    # every clause of the coverage constraint is non-empty
    assert all(cl for cl in con_c)


def test_ss_accept_delta_never_overflows_exp():
    import math
    # The bug: a swap that WORSENS the violation (v2>vi) but greatly improves
    # bandwidth (m2<<mx) must anneal on the violation term (delta>0), not the
    # large-negative weighted sum that overflowed math.exp at small T.
    d = bc._ss_accept_delta(2, 5, 10, 1, 60, 5000)   # v2>vi, m2<<mx, c2>c
    assert d > 0
    math.exp(-d / 0.05)                              # must NOT raise OverflowError
    # unchanged-violation path keeps the original weighted (max, sum) form
    assert bc._ss_accept_delta(0, 5, 100, 0, 4, 90) == (5 - 4) * 4.0 + (100 - 90) * 0.001


def test_ss_sa_chain_bignode_intra_per_block_runs():
    # a 60-node cycle, constrained: exercises the constrained chain at low T on
    # a graph with large bandwidth (the regime that used to overflow); must
    # finish and report a real violation, without raising.
    n = 60
    edges = [(i, (i + 1) % n) for i in range(n)]
    vi, bw, lab = bc._ss_sa_chain((n, edges, 2, 7, 1.0, None, 0, 0, True))
    assert vi == bc.ss_viol(lab, edges, 2, 0, True)
    assert sorted(lab) == sorted(list(range(1, n // 2 + 1)) * 2)
