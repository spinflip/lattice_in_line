"""Certifier state round-trip and lex-state independence (no solver needed)."""
import lattice_in_line.bandwidth_certifier as bc


def test_state_records_best_ub_and_lower_bound(tmp_path):
    n, edges = 4, [(0, 1), (1, 2), (2, 3)]
    st = bc.State(str(tmp_path), "demo", n, edges)
    assert bc.State.window(st.read())[1] is None  # no upper bound yet

    st.record_labeling([1, 2, 3, 4], "id")  # identity ordering, bandwidth 1
    assert st.read()["ub"] == bc.bandwidth_of([1, 2, 3, 4], edges) == 1

    st.record_labeling([1, 3, 2, 4], "worse")  # bandwidth 2 -> must NOT replace
    assert st.read()["ub"] == 1

    st.record_math_lb(1)
    st.record_unsat(0, "math")  # k=0 infeasible -> lb becomes 1
    lb, ub = bc.State.window(st.read())
    assert lb == 1 and ub == 1  # window closed at the certified optimum


def test_lex_state_is_keyed_by_k1(tmp_path, monkeypatch):
    """Different J1 caps (softenings) must land in independent state files."""
    n, e1, e2 = 4, [(0, 1), (1, 2), (2, 3)], [(0, 2), (1, 3)]
    monkeypatch.setattr(bc, "get_graph", lambda args: (n, e1))
    monkeypatch.setattr(bc, "load_j2", lambda args, nn: (e2, "j2demo"))

    class Args:
        pass

    paths = []
    for k1 in (8, 12):
        a = Args()
        a.cluster, a.state_dir, a.k1 = "clust", str(tmp_path), k1
        a.j2_cluster, a.j2_file, a.j2_edges_module, a.edges_module = "j2demo", None, None, None
        *_, k1_out, st = bc.lex_setup(a)
        assert k1_out == k1
        paths.append(st.path)

    assert paths[0] != paths[1]
    assert "k1_8" in paths[0] and "k1_12" in paths[1]
