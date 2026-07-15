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


def test_compute_cutwidth_path_cycle_and_bound():
    import vmps_geometry.bandwidth_heuristics_benchmark as bhb
    from vmps_geometry.cluster_edges import CLUSTER_EDGES
    path = [(i, i + 1) for i in range(9)]           # 10-node path
    assert bhb.compute_cutwidth(path, list(range(10))) == 1
    cyc = [(i, (i + 1) % 10) for i in range(10)]    # 10-cycle
    assert bhb.compute_cutwidth(cyc, list(range(10))) == 2
    # cut_max <= bandwidth * max_degree for any ordering (window bound)
    for g in ("C60", "pyrochlore64", "kagomeBtorus48_4x4"):
        _, edges = bhb.normalize_edges(CLUSTER_EDGES[g])
        adj = bhb.build_adjacency(edges)
        delta = max(len(a) for a in adj)
        order = bhb.cuthill_mckee_ordering(edges)
        B = bhb.compute_bandwidth(edges, order)
        assert bhb.compute_cutwidth(edges, order) <= B * delta


# ----- cutwidth certifier mode (MODE=cutwidth / cw-* commands) -----

def test_cutwidth_of_matches_bruteforce():
    import itertools
    # a small non-trivial graph: every permutation's cut_max via the sweep
    # must match an independent per-cut recomputation.
    edges = [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2)]
    n = 4

    def brute(lab):
        pos = [0] * n
        for v in range(n):
            pos[v] = lab[v] - 1
        best = 0
        for cut in range(1, n):                 # boundary after `cut` sites
            left = {v for v in range(n) if pos[v] < cut}
            c = sum(1 for u, v in edges
                    if (u in left) != (v in left))
            best = max(best, c)
        return best

    for p in itertools.permutations(range(1, n + 1)):
        assert bc.cutwidth_of(list(p), edges) == brute(list(p))


def test_cutwidth_of_path_and_cycle():
    path = [(i, i + 1) for i in range(9)]
    assert bc.cutwidth_of(list(range(1, 11)), path) == 1
    cyc = [(i, (i + 1) % 10) for i in range(10)]
    assert bc.cutwidth_of(list(range(1, 11)), cyc) == 2


def test_cutwidth_math_lb_is_valid_lower_bound():
    import itertools
    import random
    import vmps_geometry.bandwidth_heuristics_benchmark as bhb
    from vmps_geometry.cluster_edges import CLUSTER_EDGES
    # certified optima from the cutwidth campaigns: LB must never exceed them
    for g, copt in (("C12", 5), ("C20", 7), ("icosidodeca", 12)):
        _nodes, edges = bhb.normalize_edges(CLUSTER_EDGES[g])
        n = max(max(u, v) for u, v in edges) + 1
        lb = bc.cutwidth_math_lb(n, edges)
        # at least as strong as the old ceil(maxdeg/2) bound, never above c*
        assert lb >= max(1, -(-max(len(a) for a in bc.adjacency(n, edges)) // 2))
        assert lb <= copt
    # analytic tight cases: K6 (degree+spectral give 9 = c*), star (maxdeg/2),
    # path and cycle (spectral must not overshoot the trivial optima)
    K6 = [(i, j) for i in range(6) for j in range(i + 1, 6)]
    assert bc.cutwidth_math_lb(6, K6) == 9
    assert bc.cutwidth_math_lb(8, [(0, i) for i in range(1, 8)]) == 4
    assert bc.cutwidth_math_lb(8, [(i, i + 1) for i in range(7)]) == 1
    assert bc.cutwidth_math_lb(8, [(i, (i + 1) % 8) for i in range(8)]) == 2
    # brute-force validity on random small graphs
    rng = random.Random(7)
    for _ in range(15):
        n = rng.randint(4, 7)
        pool = [(i, j) for i in range(n) for j in range(i + 1, n)]
        E = rng.sample(pool, rng.randint(n - 1, len(pool)))
        true = min(bc.cutwidth_of(list(p), E)
                   for p in itertools.permutations(range(1, n + 1)))
        assert bc.cutwidth_math_lb(n, E) <= true


def test_cutwidth_state_objective_and_range(tmp_path):
    edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
    st = bc.CutwidthState(str(tmp_path), "toy__cw", 4, edges)
    lab = [1, 2, 3, 4]
    assert st.value_of(lab) == bc.cutwidth_of(lab, edges)
    assert st.range_of(lab) == bc.total_range(lab, edges)


def test_sa_cutwidth_returns_valid_permutation_and_optimal_path():
    path = [(i, i + 1) for i in range(7)]        # optimal cutwidth = 1
    cw, lab = bc.sa_cutwidth(8, path, None, 0, 1.5)
    assert sorted(lab) == list(range(1, 9))       # a genuine permutation
    assert cw == bc.cutwidth_of(lab, path) == 1


# ----- bandwidth<->cutwidth correlation script -----

def test_correlation_pearson_spearman_match_numpy():
    import vmps_geometry.bandwidth_cutwidth_correlation as bcc
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    y = 2.0 * x + rng.normal(scale=0.3, size=50)          # strong linear
    assert abs(bcc.pearson(x, y) - np.corrcoef(x, y)[0, 1]) < 1e-12
    # perfect monotone (nonlinear) -> Spearman 1, Pearson < 1
    z = np.arange(20, dtype=float)
    w = z ** 3
    assert abs(bcc.spearman(z, w) - 1.0) < 1e-12
    assert bcc.pearson(z, w) < 1.0
    # constant input -> nan, no crash
    assert np.isnan(bcc.pearson(np.ones(5), np.arange(5.0)))


def test_correlation_rankdata_handles_ties():
    import vmps_geometry.bandwidth_cutwidth_correlation as bcc
    r = bcc._rankdata(np.array([10.0, 10.0, 20.0, 5.0]))
    # two 10s share rank (0+1)/2 = 0.5; 5 is rank 0-> wait lowest gets 0
    assert list(r) == [1.5, 1.5, 3.0, 0.0]


def test_correlation_collect_points_shapes():
    import vmps_geometry.bandwidth_cutwidth_correlation as bcc
    names = ["C12", "C20"]
    algs = ["identity", "cm", "rcm"]
    # heuristics only: one point per (graph, algorithm)
    allpts = bcc.collect_points(names, algs, ["heuristics"], "all")
    assert len(allpts) == len(names) * len(algs)
    assert all(p["cutwidth"] >= 1 and p["bandwidth"] >= 1 for p in allpts)
    pg = bcc.collect_points(names, algs, ["heuristics"], "per-graph")
    assert len(pg) == len(names)
    for g in names:                     # best-of never exceeds any single ordering
        best_bw = min(p["bandwidth"] for p in allpts if p["graph"] == g)
        assert next(p for p in pg if p["graph"] == g)["bandwidth"] == best_bw


def test_correlation_includes_optimized_sources():
    import vmps_geometry.bandwidth_cutwidth_correlation as bcc
    from vmps_geometry.permutations_sat import CUSTOM_PERMUTATIONS as SAT
    # C12 has a SAT-certified ordering -> a 'sat' category point must appear,
    # and adding sources can only lower (never raise) the per-graph best-of.
    assert "C12" in SAT
    allpts = bcc.collect_points(["C12"], ["identity"], ["heuristics", "sat"], "all")
    cats = {p["category"] for p in allpts}
    assert "sat" in cats and "heuristic" in cats
    heur_only = bcc.collect_points(["C12"], ["identity"], ["heuristics"], "per-graph")
    with_sat = bcc.collect_points(["C12"], ["identity"], ["heuristics", "sat"], "per-graph")
    assert with_sat[0]["bandwidth"] <= heur_only[0]["bandwidth"]
    assert with_sat[0]["cutwidth"] <= heur_only[0]["cutwidth"]


def test_correlation_perm_to_ordering_roundtrip():
    import vmps_geometry.bandwidth_cutwidth_correlation as bcc
    verts = [0, 1, 2, 3]
    assert bcc._perm_to_ordering({0: 2, 1: 0, 2: 3, 3: 1}, verts) == [1, 3, 0, 2]
    assert bcc._perm_to_ordering({0: 0, 1: 1}, verts) is None          # wrong size
    assert bcc._perm_to_ordering({0: 0, 1: 1, 2: 1, 3: 3}, verts) is None  # dup pos


def test_cw_polish_holds_cutwidth_and_minimizes_range():
    import pytest
    pytest.importorskip("ortools")
    import vmps_geometry.bandwidth_heuristics_benchmark as bhb
    from vmps_geometry.cluster_edges import CLUSTER_EDGES
    _nodes, E = bhb.normalize_edges(CLUSTER_EDGES["C12"])
    n = 12
    ident = list(range(1, n + 1))
    c = bc.cutwidth_of(ident, E)
    status, lab, tr, lb, _note = bc.cpsat_polish_cutwidth(n, E, c, 30, 4, hint=ident)
    assert lab is not None
    assert bc.cutwidth_of(lab, E) <= c                 # bond-dim ceiling honored
    assert tr <= bc.total_range(ident, E)              # range never worsened
    assert tr == bc.total_range(lab, E)                # reported range consistent
    if status == "OPTIMAL":
        assert tr == lb                                # proven minimum range


def test_gather_permutations_builds_map(tmp_path):
    import json
    import vmps_geometry.gather_permutations as gp
    import vmps_geometry.bandwidth_heuristics_benchmark as bhb
    from vmps_geometry.cluster_edges import CLUSTER_EDGES
    _v, E = bhb.normalize_edges(CLUSTER_EDGES["C12"])
    n = 12
    lab = list(range(1, n + 1))                      # identity, 1-based
    run = tmp_path / "cw_run_C12"
    run.mkdir()
    (run / "C12__cw.json").write_text(json.dumps({
        "cluster": "C12__cw", "n": n, "num_edges": len(E),
        "best_labeling": lab, "sum_range": bc.total_range(lab, E),
        "ub": bc.cutwidth_of(lab, E), "math_lb": 2,
        "unsat": {"1": "math"}, "history": [],
    }))
    recs = gp.gather(str(tmp_path), "cw")
    assert len(recs) == 1 and recs[0]["cluster"] == "C12"
    assert recs[0]["perm0"] == {i: i for i in range(n)}       # identity -> 0-based
    assert recs[0]["stats"]["cutwidth"] == bc.cutwidth_of(lab, E)
    # the rendered module re-parses to the same permutation
    ns = {}
    exec(compile(gp.render(recs, "cw", str(tmp_path)), "<g>", "exec"), ns)
    assert ns["CUSTOM_PERMUTATIONS"]["C12"] == {i: i for i in range(n)}


def test_analyze_mpo_parse_and_correlate():
    import vmps_geometry.analyze_mpo_correlation as amc
    p = amc._default_file("permutations_sat.py")            # tracked, stable
    recs = amc.parse_file(p, "bw")
    assert len(recs) > 20
    by = {r["cluster"]: r for r in recs}
    # C12 comment stats are fixed in the tracked file
    assert by["C12"]["bandwidth"] == 4 and by["C12"]["cutwidth"] == 6
    assert by["C12"]["daux_max"] == 6
    assert all(r["source"] == "bw" for r in recs)
    out = amc.correlations(recs)                             # must not raise
    assert 0.0 <= out["cut_r"] <= 1.0 and out["n"] > 20
