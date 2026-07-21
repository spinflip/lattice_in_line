#!/usr/bin/env python3
"""
bandwidth_heuristics_benchmark.py — benchmark bandwidth- and envelope-reduction
HEURISTICS across every graph in cluster_edges.py.

Metrics reported per (graph, algorithm):
  bandwidth  max_e |pos_u - pos_v|            the certifier's objective
  avg_range  mean_e |pos_u - pos_v|           average interaction range R (was
                                              called "envelope" in older output)
  profile    sum_v ( pos_v - min_{u in N(v) u {v}} pos_u )
                                              the classic sparse-matrix
                                              envelope/profile (row leftward reach)
  cut_max    max_cut #{edges crossing the cut}   the UNWEIGHTED cutwidth = the
                                              MPO bond-dimension proxy (largest
                                              number of Hamiltonian terms in flight
                                              across any bond); for these lattices
                                              cut_max is ~1-2x the bandwidth

IMPORTANT — reversal invariance: reversing the whole ordering maps pos -> n-1-pos,
which leaves every |pos_u - pos_v| unchanged. So BOTH bandwidth and avg_range are
reversal-invariant, and Cuthill-McKee and Reverse Cuthill-McKee have IDENTICAL
bandwidth and avg_range. They differ ONLY in the profile — which is exactly the
quantity RCM was designed to reduce. That is why all three metrics are shown.

Heuristics (all dependency-light; spectral needs numpy, a core dependency):
  identity   baseline: the cluster's native site numbering
  cm         Cuthill-McKee
  rcm        Reverse Cuthill-McKee
  gps        Gibbs-Poole-Stockmeyer
  king       King's algorithm (minimum-front bandwidth greedy)
  sloan      Sloan (profile/wavefront reduction — the envelope specialist)
  spectral   Fiedler / spectral order (2nd eigenvector of the graph Laplacian)

The ordering routines and metric helpers are reused from
qubo/bandwidth_classical_compare.py where they already exist.

Usage:
  python -m lattice_in_line.bandwidth_heuristics_benchmark            # all graphs
  vmps-bandwidth-benchmark --graphs pyrochlore --summary-only
  vmps-bandwidth-benchmark --csv bench.csv --sort bandwidth
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from typing import Callable, Dict, List, Tuple

from .cluster_edges import CLUSTER_EDGES
from .qubo.bandwidth_classical_compare import (
    build_adjacency,
    compute_bandwidth,
    compute_avg_range,
    connected_components,
    cuthill_mckee_ordering,
    gps_ordering,
    normalize_edges,
    pseudo_peripheral_vertex,
    reverse_cuthill_mckee_ordering,
)

Edge = Tuple[int, int]
EdgeList = List[Edge]


# ----------------------------------------------------------------------
# extra metric: classic matrix profile / envelope (reversal-sensitive)

def compute_profile(edges: EdgeList, ordering: List[int]) -> int:
    adj = build_adjacency(edges)
    pos = {v: i for i, v in enumerate(ordering)}
    total = 0
    for v, p in pos.items():
        leftmost = p
        for u in adj[v]:
            if pos[u] < leftmost:
                leftmost = pos[u]
        total += p - leftmost
    return total


def compute_cutwidth(edges: EdgeList, ordering: List[int]) -> int:
    """cut_max: the largest number of edges crossing any of the n-1 chain cuts
    (unweighted cutwidth) -- the MPO bond-dimension proxy. Computed by treating
    each edge as a +1/-1 interval on the cut axis and prefix-summing."""
    _, normalized = normalize_edges(edges)
    pos = {v: i for i, v in enumerate(ordering)}
    n = len(ordering)
    inc = [0] * (n + 1)
    for u, v in normalized:
        a, b = sorted((pos[u], pos[v]))
        inc[a] += 1
        inc[b] -= 1
    cur = mx = 0
    for c in range(n - 1):
        cur += inc[c]
        if cur > mx:
            mx = cur
    return mx


def _bfs_dist(adj, root: int) -> Dict[int, int]:
    dist = {root: 0}
    q = deque([root])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


# ----------------------------------------------------------------------
# additional heuristics (CM / RCM / GPS come from bandwidth_classical_compare)

def king_ordering(edges: EdgeList) -> List[int]:
    """King's bandwidth-reduction heuristic: grow the ordering from a
    pseudo-peripheral start, and at each step number the frontier vertex whose
    numbering adds the FEWEST new vertices to the active front (ties: smaller
    degree). This directly damps the growth of the wavefront."""
    adj = build_adjacency(edges)
    n = len(adj)
    numbered = [False] * n
    in_front = [False] * n
    order: List[int] = []

    for comp in connected_components(adj):
        start = pseudo_peripheral_vertex(adj, min(comp, key=lambda v: len(adj[v])))
        front = {start}
        in_front[start] = True
        while front:
            def newly_added(v: int) -> Tuple[int, int, int]:
                add = sum(1 for u in adj[v] if not numbered[u] and not in_front[u])
                return (add, len(adj[v]), v)

            v = min(front, key=newly_added)
            front.discard(v)
            in_front[v] = False
            numbered[v] = True
            order.append(v)
            for u in adj[v]:
                if not numbered[u] and not in_front[u]:
                    front.add(u)
                    in_front[u] = True
    return order


# Sloan vertex statuses
_INACTIVE, _PREACTIVE, _ACTIVE, _POSTACTIVE = 0, 1, 2, 3


def sloan_ordering(edges: EdgeList, w1: int = 1, w2: int = 2) -> List[int]:
    """Sloan's profile/wavefront-reduction heuristic (priority-queue form, as in
    Sloan 1986 / Kumfert-Pothen). Priority P(v) = w1*dist(v, end) - w2*cdeg(v),
    where cdeg is the current (not-yet-numbered) degree, updated incrementally.
    Numbering runs from a pseudo-peripheral start toward the far endpoint; the
    global term pulls the front forward while the degree term keeps it thin."""
    adj = build_adjacency(edges)
    n = len(adj)
    status = [_INACTIVE] * n
    prio = [0] * n
    order: List[int] = []

    for comp in connected_components(adj):
        start = pseudo_peripheral_vertex(adj, min(comp, key=lambda v: len(adj[v])))
        end = pseudo_peripheral_vertex(adj, start)
        dist = _bfs_dist(adj, end)                 # distance to the far endpoint
        for v in comp:
            # initial current degree = static degree; the +1 counts v itself
            prio[v] = w1 * dist[v] - w2 * (len(adj[v]) + 1)
            status[v] = _INACTIVE
        status[start] = _PREACTIVE
        queue = {start}                            # holds pre/active vertices

        while queue:
            i = max(queue, key=lambda x: (prio[x], x))
            queue.discard(i)
            if status[i] == _PREACTIVE:
                for j in adj[i]:
                    prio[j] += w2                  # i leaves j's current degree
                    if status[j] == _INACTIVE:
                        status[j] = _PREACTIVE
                        queue.add(j)
            status[i] = _POSTACTIVE
            order.append(i)
            for j in adj[i]:
                if status[j] == _PREACTIVE:
                    status[j] = _ACTIVE
                    prio[j] += w2
                    for k in adj[j]:
                        if status[k] != _POSTACTIVE:
                            prio[k] += w2
                            if status[k] == _INACTIVE:
                                status[k] = _PREACTIVE
                                queue.add(k)
    return order


def spectral_ordering(edges: EdgeList) -> List[int]:
    """Fiedler / spectral ordering: sort each connected component by its Fiedler
    vector (eigenvector of the second-smallest Laplacian eigenvalue), the
    continuous relaxation of minimizing sum_e |pos_u - pos_v|^2."""
    import numpy as np
    adj = build_adjacency(edges)
    order: List[int] = []
    for comp in connected_components(adj):
        m = len(comp)
        if m == 1:
            order.append(comp[0])
            continue
        idx = {v: i for i, v in enumerate(comp)}
        L = np.zeros((m, m))
        for v in comp:
            L[idx[v], idx[v]] = len(adj[v])
            for u in adj[v]:
                L[idx[v], idx[u]] = -1.0
        vals, vecs = np.linalg.eigh(L)
        f = None
        for col in range(1, m):                    # first non-constant eigenvector
            if np.ptp(vecs[:, col]) > 1e-9:
                f = vecs[:, col]
                break
        if f is None:
            order.extend(comp)
            continue
        order.extend(sorted(comp, key=lambda v: f[idx[v]]))
    return order


# ----------------------------------------------------------------------

ALGORITHMS: Dict[str, Callable[[EdgeList], List[int]]] = {
    "identity": lambda edges: list(range(len(build_adjacency(edges)))),
    "cm": cuthill_mckee_ordering,
    "rcm": reverse_cuthill_mckee_ordering,
    "gps": gps_ordering,
    "king": king_ordering,
    "sloan": sloan_ordering,
    "spectral": spectral_ordering,
}
DEFAULT_ORDER = ["identity", "cm", "rcm", "gps", "king", "sloan", "spectral"]


def run_one(edges: EdgeList, alg: str) -> Dict:
    t0 = time.perf_counter()
    ordering = ALGORITHMS[alg](edges)
    dt = time.perf_counter() - t0
    return {
        "algorithm": alg,
        "bandwidth": compute_bandwidth(edges, ordering),
        "avg_range": compute_avg_range(edges, ordering),
        "profile": compute_profile(edges, ordering),
        "cut_max": compute_cutwidth(edges, ordering),
        "seconds": dt,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graphs", default=None,
                    help="comma-separated names or substrings to filter graphs "
                         "(default: all in cluster_edges.py)")
    ap.add_argument("--algorithms", default=",".join(DEFAULT_ORDER),
                    help=f"comma-separated subset of {list(ALGORITHMS)}")
    ap.add_argument("--sort",
                    choices=["name", "bandwidth", "cut_max", "avg_range", "profile"],
                    default="bandwidth", help="row sort within each per-graph table")
    ap.add_argument("--summary-only", action="store_true",
                    help="print only the aggregate summary, not per-graph tables")
    ap.add_argument("--csv", default=None, help="also write results to this CSV")
    args = ap.parse_args()

    algs = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    for a in algs:
        if a not in ALGORITHMS:
            ap.error(f"unknown algorithm {a!r}; choose from {list(ALGORITHMS)}")

    names = sorted(CLUSTER_EDGES)
    if args.graphs:
        want = [g.strip() for g in args.graphs.split(",") if g.strip()]
        names = [g for g in names if any(w == g or w in g for w in want)]
        if not names:
            ap.error("no graphs matched --graphs")

    rows: List[Dict] = []       # flat records for CSV + summary
    for g in names:
        vertices, edges = normalize_edges(CLUSTER_EDGES[g])
        n, m = len(vertices), len(edges)
        res = [run_one(edges, a) for a in algs]
        for r in res:
            r.update({"graph": g, "n": n, "m": m})
            rows.append(r)

        if not args.summary_only:
            print(f"\n=== {g}   (n={n}, edges={m}) ===")
            print(f"  {'algorithm':10s} {'bandwidth':>9s} {'cut_max':>8s} "
                  f"{'avg_range':>10s} {'profile':>9s} {'time_s':>10s}")
            keyf = {"name": lambda r: r["algorithm"],
                    "bandwidth": lambda r: (r["bandwidth"], r["cut_max"]),
                    "cut_max": lambda r: (r["cut_max"], r["bandwidth"]),
                    "avg_range": lambda r: (r["avg_range"], r["bandwidth"]),
                    "profile": lambda r: (r["profile"], r["bandwidth"])}[args.sort]
            bw_best = min(r["bandwidth"] for r in res)
            ct_best = min(r["cut_max"] for r in res)
            av_best = min(r["avg_range"] for r in res)
            pf_best = min(r["profile"] for r in res)
            for r in sorted(res, key=keyf):
                star = lambda val, best: "*" if val <= best + 1e-9 else " "
                print(f"  {r['algorithm']:10s} "
                      f"{r['bandwidth']:8d}{star(r['bandwidth'], bw_best)} "
                      f"{r['cut_max']:7d}{star(r['cut_max'], ct_best)} "
                      f"{r['avg_range']:9.4g}{star(r['avg_range'], av_best)} "
                      f"{r['profile']:8d}{star(r['profile'], pf_best)} "
                      f"{r['seconds']:10.4f}")

    # ---- aggregate summary ----
    print("\n" + "=" * 78)
    print(f"SUMMARY over {len(names)} graph(s): mean metrics and #best (ties count)")
    print("=" * 78)
    print(f"  {'algorithm':10s} {'mean_bw':>9s} {'mean_cut':>9s} {'mean_avgR':>10s} "
          f"{'mean_prof':>10s} {'#bw':>5s} {'#cut':>5s} {'#avgR':>6s} {'#prof':>6s} "
          f"{'tot_s':>9s}")
    by_alg: Dict[str, List[Dict]] = {a: [] for a in algs}
    for r in rows:
        by_alg[r["algorithm"]].append(r)
    # per-graph winners
    wins = {a: [0, 0, 0, 0] for a in algs}
    for g in names:
        gr = [r for r in rows if r["graph"] == g]
        for j, key in enumerate(("bandwidth", "cut_max", "avg_range", "profile")):
            best = min(r[key] for r in gr)
            for r in gr:
                if r[key] <= best + 1e-9:
                    wins[r["algorithm"]][j] += 1
    for a in algs:
        rs = by_alg[a]
        mb = sum(r["bandwidth"] for r in rs) / len(rs)
        mc = sum(r["cut_max"] for r in rs) / len(rs)
        ma = sum(r["avg_range"] for r in rs) / len(rs)
        mp = sum(r["profile"] for r in rs) / len(rs)
        ts = sum(r["seconds"] for r in rs)
        w = wins[a]
        print(f"  {a:10s} {mb:9.3f} {mc:9.3f} {ma:10.4g} {mp:10.1f} "
              f"{w[0]:5d} {w[1]:5d} {w[2]:6d} {w[3]:6d} {ts:9.3f}")
    print("\n  note: bandwidth and avg_range are reversal-invariant -> cm and rcm")
    print("  match on both (rcm only helps 'profile', the matrix envelope). king and")
    print("  sloan are wavefront/profile reducers: strong on avg_range/profile but they")
    print("  can give large bandwidth on dense graphs. cm/rcm and spectral win bandwidth.")

    if args.csv:
        import csv
        cols = ["graph", "n", "m", "algorithm", "bandwidth", "cut_max",
                "avg_range", "profile", "seconds"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r[c] for c in cols})
        print(f"\n[csv] wrote {len(rows)} rows to {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
