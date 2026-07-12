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


def test_cut_profile_vectorized_matches_bruteforce():
    rng = np.random.default_rng(3)
    n = 12
    W = rng.random((n, n))
    W = np.triu(W, 1)
    W[W < 0.5] = 0.0
    W = W + W.T
    pos = rng.permutation(n)
    m = fo.metrics(W, pos)
    cuts_bf = []
    for b in range(n - 1):
        left = {v for v in range(n) if pos[v] <= b}
        cuts_bf.append(sum(W[i, j] for i in range(n) for j in range(i + 1, n)
                           if (i in left) != (j in left)))
    assert np.allclose(m["cutwidth_max"], max(cuts_bf))
    iu, ju, w = fo._edge_arrays(W)
    assert np.allclose(fo._cut_profile(pos, iu, ju, w, n), cuts_bf)


def test_sa_order_chain_finds_weighted_path_cutwidth():
    # weighted path graph: the path order itself is cutwidth-optimal
    # (cutwidth = max edge weight); SA must find it from a random start.
    n = 10
    W = np.zeros((n, n))
    for i in range(n - 1):
        W[i, i + 1] = W[i + 1, i] = 1.0 + 0.1 * i
    p, s, perm = fo._sa_order_chain((W, "cut", 5, 1.5, None))
    assert sorted(perm) == list(range(n))                 # valid permutation
    assert abs(p - W.max()) < 1e-9                        # optimum reached


def test_anneal_order_never_worse_than_seed():
    rng = np.random.default_rng(1)
    n = 16
    W = rng.random((n, n)); W = np.triu(W, 1); W = W + W.T
    np.fill_diagonal(W, 0.0)
    seed_perm = list(range(n))
    pos0 = np.arange(n)
    before = fo.objective_value(W, pos0, "cut")
    perm = fo.anneal_order(W, "cut", 1.0, 1, 0, [seed_perm])  # procs=1 path
    pos = np.empty(n, dtype=int); pos[np.asarray(perm)] = np.arange(n)
    assert sorted(perm) == list(range(n))
    assert fo.objective_value(W, pos, "cut") <= before + 1e-12


def test_lex_heuristic_exits_early_when_cap_certified(tmp_path, monkeypatch):
    # An interrupted sweep re-runs lex-heuristic on finished caps; it must not
    # burn its full --time again once the cap's k2 is certified.
    import types
    j1 = tmp_path / "j1.txt"
    j1.write_text("0 1\n1 2\n2 3\n")
    j2 = tmp_path / "j2.txt"
    j2.write_text("0 2\n1 3\n")
    # pre-certify the cap k1=2: labeling with k2=2 recorded + k2=1 proven UNSAT
    e2 = [(0, 2), (1, 3)]
    st = bc.State(str(tmp_path), "t4__lex_j2__k1_2", 4, e2)
    st.record_labeling([1, 2, 3, 4], "test")
    st.record_unsat(1, "cpsat")
    lb, ub = bc.State.window(st.read())
    assert lb >= ub == 2                                  # certified
    monkeypatch.setattr(bc, "run_lex_heuristic",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("SA ran despite certified cap")))
    args = types.SimpleNamespace(
        cluster="t4", edges_module="cluster_edges.py", edge_file=str(j1),
        j2_file=str(j2), j2_cluster=None, j2_edges_module=None, k1=2,
        state_dir=str(tmp_path), time=999, procs=2, workers=1, seed=0,
        symmetry="reversal", fix_label1=None)
    bc.cmd_lex_heuristic(args)                            # must return, not run SA


def test_bandwidth_heuristics_valid_and_optimal_on_path():
    import vmps_geometry.bandwidth_heuristics_benchmark as bhb
    n = 20
    path = [(i, i + 1) for i in range(n - 1)]
    for name, fn in bhb.ALGORITHMS.items():
        order = fn(path)
        assert sorted(order) == list(range(n)), f"{name}: not a permutation"
        # every heuristic must recover the trivial optimum (bandwidth 1) on a path
        assert bhb.compute_bandwidth(path, order) == 1, f"{name}: bw != 1 on path"


def test_bandwidth_heuristics_handle_disconnected():
    import vmps_geometry.bandwidth_heuristics_benchmark as bhb
    # two disjoint triangles
    edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]
    for name, fn in bhb.ALGORITHMS.items():
        order = fn(edges)
        assert sorted(order) == list(range(6)), f"{name}: bad permutation"


def test_cm_rcm_reversal_invariance_and_profile():
    import vmps_geometry.bandwidth_heuristics_benchmark as bhb
    from vmps_geometry.cluster_edges import CLUSTER_EDGES
    _, edges = bhb.normalize_edges(CLUSTER_EDGES["pyrochlore64"])
    cm = bhb.cuthill_mckee_ordering(edges)
    rcm = bhb.reverse_cuthill_mckee_ordering(edges)
    # reversal leaves bandwidth and avg_range unchanged...
    assert bhb.compute_bandwidth(edges, cm) == bhb.compute_bandwidth(edges, rcm)
    assert abs(bhb.compute_envelope(edges, cm)
               - bhb.compute_envelope(edges, rcm)) < 1e-9
    # ...but rcm never has a worse matrix profile (its whole purpose)
    assert bhb.compute_profile(edges, rcm) <= bhb.compute_profile(edges, cm)


def test_compute_profile_known_value():
    import vmps_geometry.bandwidth_heuristics_benchmark as bhb
    path = [(0, 1), (1, 2), (2, 3)]
    assert bhb.compute_profile(path, [0, 1, 2, 3]) == 3      # each left-reach = 1
