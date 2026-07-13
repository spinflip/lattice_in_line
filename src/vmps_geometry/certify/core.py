"""certify.core — shared foundation for the bandwidth certifier: graph
loading, elementary graph routines, combinatorial lower bounds, and the
resumable State file. No solver dependencies (stdlib only)."""
from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
import time
from collections import deque

__all__ = [
    "adjacency", "bfs_dist", "bandwidth_of", "ceil_div", "combinatorial_lb",
    "total_range", "load_cluster", "load_edge_file", "get_graph", "State",
    "print_status",
]


def load_cluster(name: str, module_path: str):
    spec = importlib.util.spec_from_file_location("cluster_edges", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    table = getattr(mod, "CLUSTER_EDGES")
    if name not in table:
        sys.exit(f"cluster '{name}' not found; available: {', '.join(sorted(table))}")
    edges = sorted({(min(u, v), max(u, v)) for u, v in table[name] if u != v})
    n = max(max(e) for e in edges) + 1
    return n, edges


def load_edge_file(path: str):
    idx, edges = {}, set()
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2 or line.lstrip().startswith(("#", "%")):
                continue
            u, v = parts[0], parts[1]
            if u == v:
                continue
            for w in (u, v):
                if w not in idx:
                    idx[w] = len(idx)
            a, b = idx[u], idx[v]
            edges.add((min(a, b), max(a, b)))
    return len(idx), sorted(edges)


def get_graph(args):
    if getattr(args, "edge_file", None):
        return load_edge_file(args.edge_file)
    return load_cluster(args.cluster, args.edges_module)


def adjacency(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def bfs_dist(adj, src):
    d = [-1] * len(adj)
    d[src] = 0
    q = deque([src])
    while q:
        u = q.popleft()
        for w in adj[u]:
            if d[w] < 0:
                d[w] = d[u] + 1
                q.append(w)
    return d


def bandwidth_of(lab, edges):
    return max(abs(lab[u] - lab[v]) for u, v in edges) if edges else 0


def ceil_div(a, b):
    return -(a // -b)


def combinatorial_lb(n, edges, verbose=False):
    adj = adjacency(n, edges)
    deg = [len(a) for a in adj]
    lb_deg = ceil_div(max(deg), 2) if edges else 0

    lb_diam, lb_ball = 1, 1
    visited = [False] * n
    for s in range(n):
        if visited[s]:
            continue
        d0 = bfs_dist(adj, s)
        comp = [v for v in range(n) if d0[v] >= 0]
        for v in comp:
            visited[v] = True
        if len(comp) <= 1:
            continue
        # diameter bound per component (double BFS gives a lower bound on the
        # true diameter; the bandwidth bound ceil((|C|-1)/diam) needs the true
        # diameter, so compute it exactly with all-sources BFS inside comp)
        diam = 0
        for v in comp:
            dv = bfs_dist(adj, v)
            diam = max(diam, max(dv[w] for w in comp))
            # ball bound: any connected subgraph H gives B >= ceil((|H|-1)/diam(H));
            # take H = ball of radius r around v, diam(H) <= 2r
            for r in range(1, diam + 1):
                cnt = sum(1 for w in comp if 0 <= dv[w] <= r)
                lb_ball = max(lb_ball, ceil_div(cnt - 1, 2 * r))
        lb_diam = max(lb_diam, ceil_div(len(comp) - 1, diam))
    lb = max(1, lb_deg, lb_diam, lb_ball)
    if verbose:
        print(f"  lower bounds: degree {lb_deg}, diameter {lb_diam}, "
              f"ball {lb_ball}  ->  LB = {lb}")
    return lb


def total_range(lab, edges):
    """Sum of interaction distances |phi(u)-phi(v)| over all edges."""
    return sum(abs(lab[u] - lab[v]) for u, v in edges)


class State:
    # objective naming for human-facing messages; subclasses override (e.g.
    # CutwidthState -> "cutwidth" / "c*"). The stored numbers are the same
    # objective-agnostic ub/unsat machinery either way.
    objective = "bandwidth"
    objective_abbrev = "B*"

    def __init__(self, state_dir, name, n, edges, weights=None, dist=None,
                 mult=1):
        os.makedirs(state_dir, exist_ok=True)
        self.path = os.path.join(state_dir, f"{name}.json")
        self.lock = self.path + ".lock"
        self.name, self.n, self.edges = name, n, edges
        self.weights = weights
        self.dist = dist            # optional host metric on 0-based positions
        self.mult = mult            # labels 1..n/mult, each used exactly mult times

    def value_of(self, lab):
        if self.dist is not None:
            return max(self.dist(lab[u] - 1, lab[v] - 1) for u, v in self.edges)
        if self.weights is None:
            return bandwidth_of(lab, self.edges)
        return max(w * abs(lab[u] - lab[v])
                   for (u, v), w in zip(self.edges, self.weights))

    def range_of(self, lab):
        """Secondary objective: total interaction range (sum of distances).
        Weighted modes sum the weighted distances; host modes sum host
        distances. Used to break ties among equal-bandwidth layouts."""
        if self.dist is not None:
            return sum(self.dist(lab[u] - 1, lab[v] - 1) for u, v in self.edges)
        if self.weights is None:
            return total_range(lab, self.edges)
        return sum(w * abs(lab[u] - lab[v])
                   for (u, v), w in zip(self.edges, self.weights))

    def _acquire(self, timeout=120):
        t0 = time.time()
        while True:
            try:
                fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return
            except FileExistsError:
                if time.time() - t0 > timeout:
                    sys.exit(f"could not acquire lock {self.lock} "
                             f"(stale? delete it if no job is running)")
                time.sleep(0.5 + random.random())

    def _release(self):
        try:
            os.remove(self.lock)
        except FileNotFoundError:
            pass

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                return json.load(f)
        return {"cluster": self.name, "n": self.n, "num_edges": len(self.edges),
                "best_labeling": None, "ub": None,
                "math_lb": None, "unsat": {}, "history": []}

    def read(self):
        return self._load()

    def update(self, fn):
        """fn(state_dict) -> state_dict, applied under the lock (merge-safe)."""
        self._acquire()
        try:
            st = self._load()
            st = fn(st)
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(st, f, indent=1)
            os.replace(tmp, self.path)
            return st
        finally:
            self._release()

    # -- convenience mutators ------------------------------------------
    def record_labeling(self, lab, source):
        bw = self.value_of(lab)
        rng_ = self.range_of(lab)
        if self.mult == 1:
            assert sorted(lab) == list(range(1, self.n + 1)), "not a permutation"
        else:
            m = self.n // self.mult
            expect = sorted(list(range(1, m + 1)) * self.mult)
            assert sorted(lab) == expect, \
                f"not a multiplicity-{self.mult} labeling onto 1..{m}"

        def fn(st):
            cur_bw = st["ub"]
            cur_rng = st.get("sum_range")
            better = (cur_bw is None or bw < cur_bw or
                      (bw == cur_bw and (cur_rng is None or rng_ < cur_rng)))
            if better:
                improved_bw = (cur_bw is None or bw < cur_bw)
                st["ub"], st["best_labeling"], st["sum_range"] = bw, lab, rng_
                ev = "new_ub" if improved_bw else "tiebreak_range"
                st["history"].append({"t": time.time(), "event": ev,
                                      "ub": bw, "sum_range": rng_,
                                      "source": source})
                if improved_bw:
                    print(f"[state] new {self.objective} upper bound {bw} "
                          f"(total range {rng_}) ({source})")
                else:
                    print(f"[state] same {self.objective} {bw}, smaller total "
                          f"range {rng_} < {cur_rng} ({source})")
            return st
        return self.update(fn)

    def record_unsat(self, k, proof):
        def fn(st):
            cur = st["unsat"].get(str(k))
            rank = {"math": 0, "cpsat": 1, "xsat": 2, "drat": 3}
            if cur is None or rank[proof] > rank.get(cur, -1):
                st["unsat"][str(k)] = proof
                st["history"].append({"t": time.time(), "event": "unsat",
                                      "k": k, "proof": proof})
                print(f"[state] k = {k} proven infeasible ({proof})")
            return st
        return self.update(fn)

    def record_math_lb(self, lb):
        def fn(st):
            if st["math_lb"] is None or lb > st["math_lb"]:
                st["math_lb"] = lb
            return st
        return self.update(fn)

    @staticmethod
    def window(st):
        """(lb, ub): lb = 1 + largest proven-infeasible k (any proof type)."""
        lb = st["math_lb"] or 1
        if st["unsat"]:
            lb = max(lb, 1 + max(int(k) for k in st["unsat"]))
        return lb, st["ub"]


def print_status(st, edges, objective="bandwidth", abbrev="B*", value_fn=None):
    """Human-readable status. `objective`/`abbrev` name the quantity (e.g.
    'cutwidth'/'c*'); `value_fn(lab, edges)` recomputes the incumbent's
    objective value for the verification line (defaults to bandwidth_of)."""
    if value_fn is None:
        value_fn = bandwidth_of
    lb, ub = State.window(st)
    print(f"cluster {st['cluster']}: n={st['n']}, |E|={st['num_edges']}")
    print(f"  certified window: {lb} <= {abbrev} <= {ub}")
    if st["unsat"]:
        strongest = {}
        for k, p in sorted(st["unsat"].items(), key=lambda x: int(x[0])):
            strongest[k] = p
        print(f"  infeasible k: " +
              ", ".join(f"{k} ({p})" for k, p in strongest.items()))
    print(f"  math lower bound: {st['math_lb']}")
    if ub is not None and lb == ub:
        decisive = st["unsat"].get(str(ub - 1), "math")
        print(f"  *** CERTIFIED OPTIMAL: {abbrev} = {ub} "
              f"(decisive infeasibility proof at k={ub - 1}: {decisive}) ***")
    if st["best_labeling"]:
        val = value_fn(st["best_labeling"], edges)
        sr = st.get("sum_range")
        extra = f", total interaction range {sr}" if sr is not None else ""
        print(f"  incumbent labeling verified: {objective} {val}{extra}")
