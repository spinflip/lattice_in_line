#!/usr/bin/env python3
"""
bandwidth_certifier.py — certified minimum-bandwidth computation for the
clusters in cluster_edges.py (or any edge list).

Certification model
-------------------
The minimum bandwidth k* is certified by:
  (i)  a feasible labeling with bandwidth k*        -> verifiable in O(|E|)
  (ii) an infeasibility proof for k*-1              -> covers all k < k* by
       monotonicity (if bandwidth k is achievable, so is k+1)
Infeasibility proofs come in three strengths, all tracked in the state file:
  - "math":   combinatorial bounds (degree, diameter, ball) — hand-checkable
  - "cpsat":  CP-SAT INFEASIBLE verdict / objective bound   — trust the solver
  - "drat":   external SAT solver UNSAT with DRAT proof     — machine-checkable
       (use `cnf` to export DIMACS, run kissat/cadical with proof logging,
        check with drat-trim, then record with `record-unsat --proof drat`)

State file
----------
One JSON file per cluster (default ./bw_state/<cluster>.json) accumulates the
best known labeling (upper bound) and all proven-infeasible k (lower bound).
All subcommands merge their results into it under a lock file, so you can run
many jobs in parallel on different k values / seeds across machines sharing a
filesystem.

Typical campaign on a big machine / cluster
-------------------------------------------
  # 1. structural info + combinatorial lower bounds
  python bandwidth_certifier.py info --cluster pyrochlore128

  # 2. hammer the upper bound (uses all cores given)
  python bandwidth_certifier.py heuristic --cluster pyrochlore128 \
         --time 3600 --procs 40

  # 3. optimization pass: improves UB and raises a proven LB
  python bandwidth_certifier.py optimize --cluster pyrochlore128 \
         --time 7200 --workers 16

  # 4. decision ladder, farmed: run many of these concurrently with
  #    different --k (SAT side: k = UB-1 descending; UNSAT side: k = LB asc.)
  python bandwidth_certifier.py decide --cluster pyrochlore128 --k 45 \
         --time 86400 --workers 16
  python bandwidth_certifier.py decide --cluster pyrochlore128 --k 29 \
         --time 86400 --workers 16

  #    or let one process walk the ladder from both ends:
  python bandwidth_certifier.py ladder --cluster pyrochlore128 \
         --time-per-k 14400 --workers 16

  # 5. final certificate for the decisive UNSAT:
  python bandwidth_certifier.py cnf --cluster pyrochlore128 --k 41 \
         --out pyro128_k41.cnf
  kissat pyro128_k41.cnf proof.drat        # exit 20 = UNSAT
  drat-trim pyro128_k41.cnf proof.drat     # verify proof
  python bandwidth_certifier.py record-unsat --cluster pyrochlore128 \
         --k 41 --proof drat

  # 6. anytime:
  python bandwidth_certifier.py status --cluster pyrochlore128

Symmetry breaking
-----------------
--symmetry reversal   (default) pin a max-degree vertex to the lower half;
                      always valid, factor ~2.
--symmetry orbit      requires pynauty (pip install pynauty). Computes vertex
                      orbits under Aut(G) and constrains the vertex receiving
                      label 1 to one orbit representative per orbit. For
                      vertex-transitive lattices this FIXES the label-1 vertex,
                      a reduction by a factor |orbit| = n. Do not combine with
                      reversal (orbit replaces it; completeness is preserved
                      because any labeling can be mapped by an automorphism so
                      that its label-1 vertex becomes a representative).
--fix-label1 V        manual version of the above if you know your graph is
                      vertex-transitive: forces label(V) = 1.

Requires: pip install ortools     (pynauty optional, for --symmetry orbit)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from collections import deque

# --------------------------------------------------------------------
# Foundation (State, graph loading, bounds) now lives in certify.core;
# re-exported so the CLI, ./bandwidth_certifier.py script runs, and any
# importers keep working unchanged.
# --------------------------------------------------------------------
if __package__ in (None, ""):  # bare-script run: make the package importable
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vmps_geometry.certify.core import *  # noqa: F401,F403
from vmps_geometry.certify.core import (  # noqa: F401  (explicit re-export)
    State, adjacency, bfs_dist, bandwidth_of, ceil_div, combinatorial_lb,
    total_range, load_cluster, load_edge_file, get_graph, print_status,
)

# ----------------------------------------------------------------------
# graph loading
# ----------------------------------------------------------------------







# ----------------------------------------------------------------------
# elementary graph routines
# ----------------------------------------------------------------------









# ----------------------------------------------------------------------
# combinatorial lower bounds (all hand-checkable "math" certificates)
# ----------------------------------------------------------------------



# ----------------------------------------------------------------------
# heuristics for the upper bound
# ----------------------------------------------------------------------

def rcm_labeling(n, edges):
    adj = adjacency(n, edges)
    deg = [len(a) for a in adj]
    visited = [False] * n
    order = []
    for sc in sorted(range(n), key=lambda v: deg[v]):
        if visited[sc]:
            continue
        d = bfs_dist(adj, sc)
        comp = [v for v in range(n) if d[v] >= 0 and not visited[v]]
        far = max(comp, key=lambda v: d[v])
        seen = {far}
        q = deque([far])
        comp_order = []
        while q:
            u = q.popleft()
            comp_order.append(u)
            for w in sorted(adj[u], key=lambda x: deg[x]):
                if w not in seen:
                    seen.add(w)
                    q.append(w)
        for v in comp_order:
            visited[v] = True
        order.extend(comp_order)
    order.reverse()
    lab = [0] * n
    for pos, v in enumerate(order):
        lab[v] = pos + 1
    return lab




def _eval_cost(lab, edges):
    """Lexicographic cost (max distance, sum of distances): among layouts of
    equal bandwidth, prefer the smaller TOTAL (hence average) interaction range."""
    m = s = 0
    for u, v in edges:
        d = abs(lab[u] - lab[v])
        s += d
        if d > m:
            m = d
    return m, s


def sa_worker(payload):
    """One independent annealing chain; returns (bw, labeling)."""
    n, edges, seed, t_budget, init, target = payload
    rng = random.Random(seed)
    adj = adjacency(n, edges)
    lab = init[:] if init else [i + 1 for i in range(n)]
    if not init:
        rng.shuffle(lab)
    m, c = _eval_cost(lab, edges)
    best, bm, bc = lab[:], m, c
    t_end = time.time() + t_budget
    T0 = max(2.0, m / 8)
    while time.time() < t_end:
        if bm <= target:
            return bm, best
        T = T0
        while T > 0.05 and time.time() < t_end:
            for _ in range(2000):
                # mixed move pool: random swap / critical-edge targeted swap
                if rng.random() < 0.5:
                    a, b = rng.randrange(n), rng.randrange(n)
                else:
                    crit = [(u, v) for u, v in edges if abs(lab[u] - lab[v]) == m]
                    u, v = crit[rng.randrange(len(crit))]
                    a = u if rng.random() < 0.5 else v
                    other = v if a == u else u
                    tgt = max(1, min(n, lab[other] + rng.randint(-m // 2, m // 2)))
                    b = lab.index(tgt)
                if a == b:
                    continue
                lab[a], lab[b] = lab[b], lab[a]
                m2, c2 = _eval_cost(lab, edges)
                if (m2, c2) <= (m, c) or \
                   rng.random() < math.exp(-((m2 - m) * 4 + (c2 - c) * 0.001) / T):
                    m, c = m2, c2
                    if (m, c) < (bm, bc):
                        best, bm, bc = lab[:], m, c
                else:
                    lab[a], lab[b] = lab[b], lab[a]
            T *= 0.93
        lab = best[:]
        m, c = bm, bc
        # perturb for next round
        for _ in range(n // 8):
            a, b = rng.randrange(n), rng.randrange(n)
            lab[a], lab[b] = lab[b], lab[a]
        m, c = _eval_cost(lab, edges)
    return bm, best


def run_heuristic(n, edges, total_time, procs, init=None, seed=0, target=0,
                  stall=None):
    """Parallel multi-start SA. Returns (bw, labeling)."""
    rcm = rcm_labeling(n, edges)
    best_lab, best_bw = rcm, bandwidth_of(rcm, edges)
    print(f"[heuristic] RCM upper bound: {best_bw}")
    if init is not None and bandwidth_of(init, edges) < best_bw:
        best_lab, best_bw = init[:], bandwidth_of(init, edges)
        print(f"[heuristic] state-file incumbent: {best_bw}")
    if best_bw <= target:
        print(f"[heuristic] upper bound {best_bw} already meets the proven "
              f"lower bound {target}; nothing to do")
        return best_bw, best_lab
    if stall is None or stall <= 0:
        stall = max(60.0, total_time / 10)
    t_end = time.time() + total_time
    last_improve = time.time()
    batch = max(15.0, min(stall / 2, total_time / 4))
    r = 0
    with mp.Pool(procs) as pool:
        while time.time() < t_end:
            if time.time() - last_improve > stall:
                print(f"[heuristic] no improvement for {stall:.0f}s "
                      f"({stall / 3600:.2f}h); stopping early at UB {best_bw}")
                break
            dur = min(batch, t_end - time.time())
            if dur <= 1:
                break
            jobs = []
            for p in range(procs):
                use_init = best_lab if (p % 2 == 0) else None  # half seeded, half fresh
                jobs.append((n, edges, seed + r * procs + p, dur,
                             use_init, target))
            r += 1
            for bw, lab in pool.imap_unordered(sa_worker, jobs):
                if bw < best_bw:
                    best_bw, best_lab = bw, lab
                    last_improve = time.time()
                    print(f"[heuristic] new upper bound: {best_bw}")
            if best_bw <= target:
                print(f"[heuristic] reached the proven lower bound {target}; "
                      f"stopping early")
                break
    return best_bw, best_lab

# ----------------------------------------------------------------------
# symmetry breaking
# ----------------------------------------------------------------------

def orbit_representatives(n, edges):
    """Vertex-orbit representatives under Aut(G) via pynauty (None if absent)."""
    try:
        import pynauty
    except ImportError:
        return None
    g = pynauty.Graph(n)
    adj = adjacency(n, edges)
    for v in range(n):
        g.connect_vertex(v, adj[v])
    # orbits: list of length n, orbits[v] = orbit id of v
    _, _, _, orbits, num_orbits = pynauty.autgrp(g)
    reps = sorted({min(v for v in range(n) if orbits[v] == o)
                   for o in set(orbits)})
    return reps, num_orbits


def apply_symmetry(model, lab, n, edges, mode, fix_label1):
    """Add symmetry-breaking constraints to a CP-SAT model. Returns a note."""
    if fix_label1 is not None:
        model.Add(lab[fix_label1] == 1)
        return f"fixed label 1 on vertex {fix_label1} (user asserts vertex-transitivity)"
    if mode == "orbit":
        res = orbit_representatives(n, edges)
        if res is None:
            print("[warn] pynauty not installed; falling back to reversal symmetry")
        else:
            reps, num_orbits = res
            if len(reps) == 1:
                model.Add(lab[reps[0]] == 1)
                return f"vertex-transitive: fixed label 1 on vertex {reps[0]} (factor n)"
            bools = []
            for r in reps:
                b = model.NewBoolVar(f"rep{r}")
                model.Add(lab[r] == 1).OnlyEnforceIf(b)
                bools.append(b)
            model.AddBoolOr(bools)
            return f"label-1 vertex restricted to {len(reps)} orbit representatives " \
                   f"({num_orbits} orbits)"
    # default: reversal
    adj = adjacency(n, edges)
    w = max(range(n), key=lambda v: len(adj[v]))
    model.Add(lab[w] <= (n + 1) // 2)
    return f"reversal: pinned vertex {w} to labels 1..{(n + 1) // 2}"


# ----------------------------------------------------------------------
# CP-SAT decision and optimization
# ----------------------------------------------------------------------

def cpsat_decide(n, edges, k, time_limit, workers, hint=None,
                 symmetry="reversal", fix_label1=None, log=False):
    from ortools.sat.python import cp_model
    model = cp_model.CpModel()
    lab = [model.NewIntVar(1, n, f"l{v}") for v in range(n)]
    model.AddAllDifferent(lab)
    for u, v in edges:
        d = model.NewIntVar(0, k, f"d{u}_{v}")
        model.AddAbsEquality(d, lab[u] - lab[v])
    note = apply_symmetry(model, lab, n, edges, symmetry, fix_label1)
    if hint is not None:
        for v in range(n):
            model.AddHint(lab[v], hint[v])
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    if log:
        solver.parameters.log_search_progress = True
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return "SAT", [solver.Value(lab[v]) for v in range(n)], note
    if status == cp_model.INFEASIBLE:
        return "UNSAT", None, note
    return "UNKNOWN", None, note


def cpsat_decide_caps(n, edges, caps, time_limit, workers, hint=None,
                      symmetry="reversal", fix_label1=None, log=False):
    """Decision problem with a per-edge distance cap: |l_u-l_v| <= caps[e]."""
    from ortools.sat.python import cp_model
    if any(c < 1 for c in caps):
        return "UNSAT", None, "some cap < 1 (labels are distinct, so d>=1)"
    model = cp_model.CpModel()
    lab = [model.NewIntVar(1, n, f"l{v}") for v in range(n)]
    model.AddAllDifferent(lab)
    for (u, v), c in zip(edges, caps):
        if c >= n - 1:
            continue                                  # non-binding
        d = model.NewIntVar(0, c, f"d{u}_{v}")
        model.AddAbsEquality(d, lab[u] - lab[v])
    note = apply_symmetry(model, lab, n, edges, symmetry, fix_label1)
    if hint is not None:
        for v in range(n):
            model.AddHint(lab[v], hint[v])
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    if log:
        solver.parameters.log_search_progress = True
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return "SAT", [solver.Value(lab[v]) for v in range(n)], note
    if status == cp_model.INFEASIBLE:
        return "UNSAT", None, note
    return "UNKNOWN", None, note


def cpsat_optimize(n, edges, lb, ub, time_limit, workers, hint=None,
                   symmetry="reversal", fix_label1=None, log=False):
    from ortools.sat.python import cp_model
    model = cp_model.CpModel()
    lab = [model.NewIntVar(1, n, f"l{v}") for v in range(n)]
    model.AddAllDifferent(lab)
    B = model.NewIntVar(lb, ub, "B")
    for u, v in edges:
        d = model.NewIntVar(0, n - 1, f"d{u}_{v}")
        model.AddAbsEquality(d, lab[u] - lab[v])
        model.Add(d <= B)
    note = apply_symmetry(model, lab, n, edges, symmetry, fix_label1)
    model.Minimize(B)
    if hint is not None:
        for v in range(n):
            model.AddHint(lab[v], hint[v])
        model.AddHint(B, bandwidth_of(hint, edges))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    if log:
        solver.parameters.log_search_progress = True
    status = solver.Solve(model)
    sol = None
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sol = [solver.Value(lab[v]) for v in range(n)]
    proven_lb = int(math.ceil(solver.BestObjectiveBound())) \
        if solver.BestObjectiveBound() > 0 else lb
    optimal = (status == cp_model.OPTIMAL)
    return sol, proven_lb, optimal, note


# ----------------------------------------------------------------------
# CNF export (for kissat/cadical + drat-trim machine-checkable certificates)
# ----------------------------------------------------------------------

def build_cnf(n, edges, k, fix_label1=None, symmetry="reversal", caps=None):
    """
    One-hot encoding: x_{v,i} <=> vertex v has label i (1-based labels).
    var(v,i) = v*n + i           (i in 1..n)
    Exactly-one per vertex and per label via Sinz sequential at-most-one
    (aux vars after the n^2 primary block) + at-least-one clauses.
    Forbidden pairs: for each edge (u,v) and |i-j| > k:  (~x_{u,i} | ~x_{v,j}).
    Returns (clauses, nvars, symmetry_note).
    """
    def var(v, i):
        return v * n + i

    clauses = []
    next_aux = n * n + 1

    def amo_sequential(lits):
        nonlocal next_aux
        m = len(lits)
        if m <= 1:
            return
        s = list(range(next_aux, next_aux + m - 1))
        next_aux += m - 1
        clauses.append([-lits[0], s[0]])
        for i in range(1, m - 1):
            clauses.append([-lits[i], s[i]])
            clauses.append([-s[i - 1], s[i]])
            clauses.append([-lits[i], -s[i - 1]])
        clauses.append([-lits[m - 1], -s[m - 2]])

    for v in range(n):                                   # rows: vertex v
        lits = [var(v, i) for i in range(1, n + 1)]
        clauses.append(lits[:])                          # at least one
        amo_sequential(lits)
    for i in range(1, n + 1):                            # cols: label i
        lits = [var(v, i) for v in range(n)]
        clauses.append(lits[:])
        amo_sequential(lits)

    for ei, (u, v) in enumerate(edges):                  # distance constraint
        cap = caps[ei] if caps is not None else k
        if cap < 1:                                       # distinct labels => d>=1
            clauses.append([var(u, 1)])
            clauses.append([-var(u, 1)])
            continue
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if abs(i - j) > cap:
                    clauses.append([-var(u, i), -var(v, j)])

    sym_note = "none"
    if fix_label1 is not None:
        clauses.append([var(fix_label1, 1)])
        sym_note = f"unit: vertex {fix_label1} has label 1"
    elif symmetry == "orbit":
        res = orbit_representatives(n, edges)
        if res and len(res[0]) < n:
            reps, _ = res
            clauses.append([var(r, 1) for r in reps])
            sym_note = f"label-1 vertex among {len(reps)} orbit representatives"
        else:
            symmetry = "reversal"
    if sym_note == "none" and symmetry == "reversal":
        adj = adjacency(n, edges)
        w = max(range(n), key=lambda v: len(adj[v]))
        for i in range((n + 1) // 2 + 1, n + 1):
            clauses.append([-var(w, i)])
        sym_note = f"reversal: vertex {w} restricted to labels 1..{(n + 1) // 2}"

    nvars = next_aux - 1
    return clauses, nvars, sym_note


def write_cnf(n, edges, k, path, fix_label1=None, symmetry="reversal", caps=None):
    clauses, nvars, sym_note = build_cnf(n, edges, k, fix_label1, symmetry, caps)
    with open(path, "w") as f:
        f.write(f"c bandwidth decision sat(G, k={k}), n={n}, |E|={len(edges)}\n")
        f.write(f"c var(v,i) = v*n+i for v in 0..n-1, i in 1..n; aux from {n * n + 1}\n")
        f.write(f"c symmetry: {sym_note}\n")
        f.write(f"p cnf {nvars} {len(clauses)}\n")
        for c in clauses:
            f.write(" ".join(map(str, c)) + " 0\n")
    return nvars, len(clauses), sym_note


def pysat_solve(clauses, solver_name, time_limit, want_proof=False):
    """Solve with a python-sat backend under a wall-clock limit.
    Returns (verdict, model_or_None, proof_lines_or_None);
    verdict in {"SAT", "UNSAT", "UNKNOWN"}."""
    import threading
    from pysat.solvers import Solver
    kwargs = {"bootstrap_with": clauses, "use_timer": True}
    if want_proof:
        kwargs["with_proof"] = True
    with Solver(name=solver_name, **kwargs) as s:
        timer = threading.Timer(time_limit, s.interrupt)
        timer.start()
        try:
            res = s.solve_limited(expect_interrupt=True)
        finally:
            timer.cancel()
        if res is True:
            return "SAT", s.get_model(), None
        if res is False:
            proof = s.get_proof() if want_proof else None
            return "UNSAT", None, proof
        return "UNKNOWN", None, None


def write_dimacs(path, clauses, nvars, header):
    with open(path, "w") as f:
        for h in header:
            f.write(f"c {h}\n")
        f.write(f"p cnf {nvars} {len(clauses)}\n")
        for c in clauses:
            f.write(" ".join(map(str, c)) + " 0\n")


def _pysat_proc(payload):
    clauses, name, time_limit, want_proof = payload
    return (name,) + pysat_solve(clauses, name, time_limit, want_proof)


def parallel_crosscheck(clauses, time_limit, want_proof):
    """Run CaDiCaL and Glucose concurrently (wall time = max, not sum).
    Returns (verdicts dict name->verdict, model_or_None, proof_or_None)."""
    jobs = [(clauses, "cadical153", time_limit, False),
            (clauses, "glucose42", time_limit, want_proof)]
    verdicts, model, proof = {}, None, None
    t0 = time.time()
    with mp.Pool(2) as pool:
        for name, v, mdl, prf in pool.imap_unordered(_pysat_proc, jobs):
            _el = time.time() - t0
            print(f"[crosscheck] {name}: {v}  ({_el:.1f}s = {_el / 3600:.2f}h)")
            verdicts[name] = v
            if v == "SAT" and mdl is not None:
                model = mdl
            if prf is not None:
                proof = prf
    return verdicts, model, proof


def crosscheck_unsat(n, edges, k, time_limit, fix_label1=None,
                     symmetry="reversal", proof_out=None, cnf_out=None,
                     caps=None, prebuilt=None):
    """Decide sat(G,k) with two independent SAT solver codebases.
    Returns (verdict, labeling_or_None, agree: bool, note).
    prebuilt = (clauses, nvars, note) bypasses the path-host encoding."""
    if prebuilt is not None:
        clauses, nvars, note = prebuilt
        if cnf_out:
            write_dimacs(cnf_out, clauses, nvars,
                         [f"host-mode decision, k={k}, n={n}, |E|={len(edges)}",
                          f"symmetry: {note}"])
    else:
        clauses, nvars, note = build_cnf(n, edges, k, fix_label1, symmetry, caps)
        if cnf_out:
            write_cnf(n, edges, k, cnf_out, fix_label1, symmetry, caps)
    vd, model, proof = parallel_crosscheck(clauses, time_limit,
                                           proof_out is not None)
    verdicts = list(vd.values())
    if proof_out and proof is not None:
        with open(proof_out, "w") as f:
            f.write("\n".join(proof) + "\n0\n" if proof and proof[-1] != "0"
                    else "\n".join(proof) + "\n")
        print(f"[crosscheck] DRAT proof written to {proof_out} "
              f"(verify later with: drat-trim {cnf_out or '<cnf>'} {proof_out})")
    if all(v == "UNSAT" for v in verdicts):
        return "UNSAT", None, True, note
    if "SAT" in verdicts:
        lab = None
        if model is not None:
            true_vars = {x for x in model if 0 < x <= n * n}
            lab = [next(i for i in range(1, n + 1) if v * n + i in true_vars)
                   for v in range(n)]
        return "SAT", lab, all(x in ("SAT", "UNKNOWN") for x in verdicts), note
    return "UNKNOWN", None, False, note


def decode_model(n, edges, k, model_file):
    """Decode a SAT solver 'v ...' model file back into a labeling and verify."""
    true_vars = set()
    with open(model_file) as f:
        for line in f:
            if line.startswith("v") or line[:1].lstrip("-").isdigit():
                for tok in line.replace("v", "").split():
                    x = int(tok)
                    if x > 0:
                        true_vars.add(x)
    lab = [None] * n
    for v in range(n):
        for i in range(1, n + 1):
            if v * n + i in true_vars:
                if lab[v] is not None:
                    sys.exit(f"vertex {v} has two labels — model invalid")
                lab[v] = i
    if None in lab or sorted(lab) != list(range(1, n + 1)):
        sys.exit("model does not encode a permutation")
    bw = bandwidth_of(lab, edges)
    if bw > k:
        sys.exit(f"model bandwidth {bw} exceeds k={k} — encoding/model mismatch")
    return lab, bw

# ----------------------------------------------------------------------
# shared state file (lock + merge, safe for many concurrent jobs)
# ----------------------------------------------------------------------





# ----------------------------------------------------------------------
# subcommands
# ----------------------------------------------------------------------

def cmd_info(args):
    n, edges = get_graph(args)
    adj = adjacency(n, edges)
    deg = [len(a) for a in adj]
    print(f"{args.cluster or args.edge_file}: n={n}, |E|={len(edges)}, "
          f"degree min/max {min(deg)}/{max(deg)}")
    lb = combinatorial_lb(n, edges, verbose=True)
    st = State(args.state_dir, args.cluster or "edgefile", n, edges)
    st.record_math_lb(lb)
    st.record_unsat(lb - 1, "math")
    if args.orbits:
        res = orbit_representatives(n, edges)
        if res is None:
            print("  (install pynauty for orbit information)")
        else:
            reps, num = res
            print(f"  automorphism orbits: {num} "
                  f"({'vertex-transitive' if num == 1 else 'reps: ' + str(reps[:10])})")
    print_status(st.read(), edges)


def cmd_heuristic(args):
    n, edges = get_graph(args)
    st = State(args.state_dir, args.cluster or "edgefile", n, edges)
    cur = st.read()
    init = cur["best_labeling"]
    lb, _ = State.window(cur)
    bw, lab = run_heuristic(n, edges, args.time, args.procs, init=init,
                            seed=args.seed, target=lb, stall=args.stall)
    st.record_labeling(lab, "heuristic")
    print_status(st.read(), edges)


def cmd_optimize(args):
    n, edges = get_graph(args)
    st = State(args.state_dir, args.cluster or "edgefile", n, edges)
    cur = st.read()
    lb, ub = State.window(cur)
    if cur["best_labeling"] is None:
        rcm = rcm_labeling(n, edges)
        st.record_labeling(rcm, "rcm")
        cur = st.read()
        lb, ub = State.window(cur)
    if ub is not None and lb >= ub:
        print("[optimize] window already closed; nothing to do")
        print_status(cur, edges)
        return
    sol, proven_lb, optimal, note = cpsat_optimize(
        n, edges, lb, ub, args.time, args.workers,
        hint=cur["best_labeling"], symmetry=args.symmetry,
        fix_label1=args.fix_label1, log=args.log)
    print(f"[optimize] symmetry: {note}")
    if sol:
        st.record_labeling(sol, "cpsat-optimize")
    if proven_lb > lb:
        # CP-SAT proved B >= proven_lb, i.e. all k <= proven_lb-1 infeasible.
        # Valid only if symmetry breaking was sound (reversal/orbit are).
        st.record_unsat(proven_lb - 1, "cpsat")
    if optimal:
        print("[optimize] CP-SAT closed the instance.")
    print_status(st.read(), edges)


def cmd_decide(args):
    n, edges = get_graph(args)
    st = State(args.state_dir, args.cluster or "edgefile", n, edges)
    cur = st.read()
    hint = cur["best_labeling"]
    if hint and bandwidth_of(hint, edges) > args.k + 10:
        hint = None
    res, lab, note = cpsat_decide(n, edges, args.k, args.time, args.workers,
                                  hint=hint, symmetry=args.symmetry,
                                  fix_label1=args.fix_label1, log=args.log)
    print(f"[decide k={args.k}] {res}   (symmetry: {note})")
    if res == "SAT":
        st.record_labeling(lab, f"cpsat-decide(k={args.k})")
    elif res == "UNSAT":
        st.record_unsat(args.k, "cpsat")
    print_status(st.read(), edges)


def cmd_ladder(args):
    n, edges = get_graph(args)
    st = State(args.state_dir, args.cluster or "edgefile", n, edges)
    lbm = combinatorial_lb(n, edges)
    st.record_math_lb(lbm)
    st.record_unsat(lbm - 1, "math")
    cur = st.read()
    if cur["best_labeling"] is None:
        bw, lab = run_heuristic(n, edges, min(120, args.time_per_k),
                                args.procs, seed=args.seed,
                                target=State.window(st.read())[0])
        st.record_labeling(lab, "heuristic")
        cur = st.read()
    side = 0  # alternate: 0 = SAT side (k = ub-1), 1 = UNSAT side (k = lb)
    while True:
        lb, ub = State.window(cur)
        if lb >= ub:
            break
        k = ub - 1 if side == 0 else lb
        print(f"[ladder] window [{lb},{ub}] -> trying k={k} "
              f"({'SAT' if side == 0 else 'UNSAT'} side)")
        hint = cur["best_labeling"] if side == 0 else None
        res, lab, _ = cpsat_decide(n, edges, k, args.time_per_k, args.workers,
                                   hint=hint, symmetry=args.symmetry,
                                   fix_label1=args.fix_label1)
        if res == "SAT":
            st.record_labeling(lab, f"ladder(k={k})")
        elif res == "UNSAT":
            st.record_unsat(k, "cpsat")
        else:
            print(f"[ladder] k={k} timed out; switching side")
            if side == 1:
                print("[ladder] both sides hard at current budget — stopping. "
                      "Increase --time-per-k or farm `decide` jobs in parallel.")
                break
            side = 1
            cur = st.read()
            continue
        side ^= 1
        cur = st.read()
    print_status(st.read(), edges)


def cmd_cnf(args):
    n, edges = get_graph(args)
    nv, nc, note = write_cnf(n, edges, args.k, args.out,
                             fix_label1=args.fix_label1, symmetry=args.symmetry)
    print(f"wrote {args.out}: {nv} vars, {nc} clauses (symmetry: {note})")
    print(f"run:    kissat {args.out} proof.drat       (exit 20 = UNSAT)")
    print(f"verify: drat-trim {args.out} proof.drat")
    print(f"then:   bandwidth_certifier.py record-unsat --cluster {args.cluster} "
          f"--k {args.k} --proof drat")
    print(f"if SAT: kissat prints a model; decode with the `decode` subcommand")


def cmd_decode(args):
    n, edges = get_graph(args)
    lab, bw = decode_model(n, edges, args.k, args.model)
    st = State(args.state_dir, args.cluster or "edgefile", n, edges)
    st.record_labeling(lab, f"sat-model(k={args.k})")
    print(f"decoded labeling, bandwidth {bw}")
    print_status(st.read(), edges)


def cmd_verify_unsat(args):
    """Decide sat(G,k) with two independent python-sat solvers (CaDiCaL +
    Glucose). Both UNSAT -> recorded as proof level "xsat". Optionally
    archives the CNF and a DRAT proof for later third-party verification."""
    n, edges = get_graph(args)
    st = State(args.state_dir, args.cluster or "edgefile", n, edges)
    res, lab, agree, note = crosscheck_unsat(
        n, edges, args.k, args.time, fix_label1=args.fix_label1,
        symmetry=args.symmetry, proof_out=args.proof_out, cnf_out=args.cnf_out)
    print(f"[verify-unsat k={args.k}] {res}   (symmetry: {note})")
    if res == "UNSAT" and agree:
        st.record_unsat(args.k, "xsat")
    elif res == "SAT":
        if lab is not None:
            st.record_labeling(lab, f"pysat(k={args.k})")
        print("[verify-unsat] instance is SAT — labeling recorded")
    else:
        print("[verify-unsat] inconclusive within the time limit")
    print_status(st.read(), edges)


def cmd_record_unsat(args):
    n, edges = get_graph(args)
    st = State(args.state_dir, args.cluster or "edgefile", n, edges)
    st.record_unsat(args.k, args.proof)
    print_status(st.read(), edges)


def cmd_import_labeling(args):
    """Seed the state with an existing labeling.
    Accepts: JSON dict {original: new} (0- or 1-based values), JSON list,
    or whitespace-separated labels (lab[v] on position v)."""
    n, edges = get_graph(args)
    text = open(args.file).read().strip()
    try:
        import ast as _ast
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = _ast.literal_eval(text)   # int-keyed dict literal
        if isinstance(data, dict):
            data = {str(k): v for k, v in data.items()}
        if isinstance(data, dict):
            vals = [int(data[str(v)] if str(v) in data else data[v])
                    for v in range(n)]
        else:
            vals = [int(x) for x in data]
    except (ValueError, SyntaxError):
        vals = [int(x) for x in text.split()]
    if len(vals) != n:
        sys.exit(f"expected {n} labels, got {len(vals)}")
    if sorted(vals) == list(range(n)):          # 0-based -> 1-based
        vals = [x + 1 for x in vals]
    if sorted(vals) != list(range(1, n + 1)):
        sys.exit("values are not a permutation of 0..n-1 or 1..n")
    st = State(args.state_dir, args.cluster or "edgefile", n, edges)
    st.record_labeling(vals, f"import({os.path.basename(args.file)})")
    print_status(st.read(), edges)


def cmd_polish(args):
    """Among layouts of the current (or --target) bandwidth B, find one
    minimizing the TOTAL interaction range sum_e |phi(u)-phi(v)|, via CP-SAT
    with the hard cap d_e <= B. If CP-SAT reaches OPTIMAL, the result is the
    provably minimum-range layout at that bandwidth. Works for both plain
    permutations and supersite labelings (--block q: each chain position holds
    exactly q sites). The improved labeling is recorded into the state."""
    n, edges = get_graph(args)
    q = getattr(args, "block", 1) or 1
    min_intra = per_block = 0
    if q > 1:
        if n % q:
            sys.exit(f"--block {q} does not divide n={n}")
        m = n // q
        min_intra, per_block = ss_constraints(args, edges)
        st = ss_state(args, n, edges)
    else:
        m = n
        st = State(args.state_dir, args.cluster or "edgefile", n, edges)
    cur = st.read()
    _, ub = State.window(cur)
    B = args.target if args.target is not None else ub
    if B is None:
        sys.exit("no bandwidth known yet; run a campaign first or pass --target")
    from ortools.sat.python import cp_model
    model = cp_model.CpModel()
    Bmat = {}
    if q == 1:
        lab = [model.NewIntVar(1, n, f"l{v}") for v in range(n)]
        model.AddAllDifferent(lab)
    else:
        # block index r(v) in 1..m, each block holding exactly q sites
        R = [model.NewIntVar(1, m, f"r{v}") for v in range(n)]
        for r in range(1, m + 1):
            bvars = []
            for v in range(n):
                b = model.NewBoolVar(f"b{v}_{r}")
                model.Add(R[v] == r).OnlyEnforceIf(b)
                model.Add(R[v] != r).OnlyEnforceIf(b.Not())
                bvars.append(b)
                Bmat[v, r] = b
            model.Add(sum(bvars) == q)
        lab = R
    dvars = []
    d_of = {}
    for u, v in edges:
        d = model.NewIntVar(0, B, f"d{u}_{v}")       # hard bandwidth cap
        model.AddAbsEquality(d, lab[u] - lab[v])
        dvars.append(d)
        d_of[u, v] = d
    model.Minimize(sum(dvars))                        # total interaction range
    # hidden-bond constraints (ss states tagged _ie<N>/_ipb): polishing must
    # not un-hide bonds the certified solution was required to hide
    if min_intra:
        bs = []
        for (u, v), d in d_of.items():
            b = model.NewBoolVar(f"in{u}_{v}")
            model.Add(d == 0).OnlyEnforceIf(b)
            model.Add(d != 0).OnlyEnforceIf(b.Not())
            bs.append(b)
        model.Add(sum(bs) >= min_intra)
    if per_block:
        for r in range(1, m + 1):
            lits = []
            for u, v in edges:
                e = model.NewBoolVar(f"e{u}_{v}_{r}")
                model.AddImplication(e, Bmat[u, r])
                model.AddImplication(e, Bmat[v, r])
                lits.append(e)
            model.AddBoolOr(lits)
    # symmetry: reversal of the chain (range is reversal-invariant -> sound)
    adj = adjacency(n, edges)
    w = max(range(n), key=lambda x: len(adj[x]))
    model.Add(lab[w] <= (m + 1) // 2)
    if cur["best_labeling"]:
        seed = cur["best_labeling"]
        if max(abs(seed[u] - seed[v]) for u, v in edges) <= B:
            for v in range(n):
                model.AddHint(lab[v], seed[v])
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = args.time
    solver.parameters.num_search_workers = args.workers
    if args.log:
        solver.parameters.log_search_progress = True
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("[polish] no solution within the time limit")
        return
    sol = [solver.Value(lab[v]) for v in range(n)]
    bw = max(abs(sol[u] - sol[v]) for u, v in edges)
    tr = sum(abs(sol[u] - sol[v]) for u, v in edges)
    proven = solver.BestObjectiveBound()
    tag = "OPTIMAL (minimum total range certified)" \
        if status == cp_model.OPTIMAL else "feasible (not closed)"
    print(f"[polish] bandwidth {bw}, total interaction range {tr} "
          f"(avg {tr / len(edges):.3f}); proven range lower bound "
          f"{proven:.0f}  [{tag}]")
    st.record_labeling(sol, f"polish(B={B})")
    print_status(st.read(), edges)


def cmd_status(args):
    n, edges = get_graph(args)
    st = State(args.state_dir, args.cluster or "edgefile", n, edges)
    print_status(st.read(), edges)


def cmd_export(args):
    n, edges = get_graph(args)
    st = State(args.state_dir, args.cluster or "edgefile", n, edges)
    cur = st.read()
    if not cur["best_labeling"]:
        sys.exit("no labeling in state yet")
    lab = cur["best_labeling"]
    perm0 = {v: lab[v] - 1 for v in range(n)}   # 0-based {original: new}
    if args.json:
        print(json.dumps({str(k): v for k, v in perm0.items()}))
    else:
        # integer-keyed map: valid Python literal, parse with ast.literal_eval
        print("{" + ", ".join(f"{v}: {perm0[v]}" for v in range(n)) + "}")




# ======================================================================
# J1-J2 LEXICOGRAPHIC MODE
# ======================================================================
# Stage 1: certify k1* = minimum bandwidth of the J1 graph (= the normal
#          unweighted campaign on --cluster).
# Stage 2: with the hard constraint d_e <= k1 on every J1 edge, minimize
#          k2 = max distance over J2 edges. Decision sat(k1, k2) is
#          monotone in k2, so the same ladder/certificate logic applies.
# Lex state file: <cluster>__lex_<j2name>.json; "ub"/"unsat" refer to k2.

def load_edge_file_indexed(path, n):
    """Edge list whose vertex ids are already 0..n-1 (no reindexing)."""
    edges = set()
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2 or line.lstrip().startswith(("#", "%")):
                continue
            u, v = int(parts[0]), int(parts[1])
            if u == v:
                continue
            if not (0 <= u < n and 0 <= v < n):
                sys.exit(f"vertex {max(u, v)} out of range in {path} (n={n})")
            edges.add((min(u, v), max(u, v)))
    return sorted(edges)


def load_j2(args, n):
    if getattr(args, "j2_cluster", None):
        # J2 table defaults to a sibling cluster_edges_NNN.py (next to the J1
        # --edges-module), or an explicit --j2-edges-module, or the J1 module.
        j2_module = getattr(args, "j2_edges_module", None)
        if not j2_module:
            sibling = os.path.join(os.path.dirname(args.edges_module) or ".",
                                   "cluster_edges_NNN.py")
            j2_module = sibling if os.path.exists(sibling) else args.edges_module
        spec = importlib.util.spec_from_file_location("ce2", j2_module)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        tab = getattr(mod, "CLUSTER_EDGES")
        if args.j2_cluster not in tab:
            sys.exit(f"J2 cluster '{args.j2_cluster}' not found in {j2_module}")
        edges = sorted({(min(u, v), max(u, v))
                        for u, v in tab[args.j2_cluster] if u != v})
        if max(max(e) for e in edges) >= n:
            sys.exit("J2 cluster has vertices beyond the J1 graph size")
        return edges, args.j2_cluster
    if getattr(args, "j2_file", None):
        name = os.path.splitext(os.path.basename(args.j2_file))[0]
        return load_edge_file_indexed(args.j2_file, n), name
    sys.exit("provide --j2-cluster NAME or --j2-file PATH")


def lex_setup(args):
    n, e1 = get_graph(args)
    e2, j2n = load_j2(args, n)
    # resolve k1 FIRST: explicit --k1, else the J1 state's bandwidth. We use the
    # best-known upper bound (a labeling achieving it exists, so the cap is
    # feasible) even when J1 is not yet certified — just warn in that case.
    k1 = args.k1
    if k1 is None:
        st1 = State(args.state_dir, args.cluster, n, e1).read()
        lb1, ub1 = State.window(st1)
        if ub1 is None:
            sys.exit("stage 1 (J1 bandwidth) has no result yet; run the normal "
                     "campaign on the J1 cluster first, or pass an explicit --k1 cap")
        if lb1 < ub1:
            print(f"[lex] WARNING: J1 bandwidth is not certified (window "
                  f"[{lb1}, {ub1}]); using the best-known upper bound k1={ub1} "
                  f"as the J1 cap")
        k1 = ub1
    # The J2 (k2) certification — the best labeling AND the proven-infeasible k2
    # values — is only valid for THIS J1 cap k1. Key the state file by k1 so
    # different caps (e.g. different SOFTEN) are independent and never mix.
    st2 = State(args.state_dir, f"{args.cluster}__lex_{j2n}__k1_{k1}", n, e2)
    return n, e1, e2, k1, st2


def lex_caps(e1, e2, k1, k2):
    return e1 + e2, [k1] * len(e1) + [k2] * len(e2)


def sa_lex(n, e1, k1, e2, init, seed, t_budget):
    """Minimize (max d over e2, count) subject to max d over e1 <= k1."""
    rng = random.Random(seed)
    lab = init[:]

    def feas(l):
        return all(abs(l[u] - l[v]) <= k1 for u, v in e1)

    def cost(l):
        m = s = 0
        for u, v in e2:
            d = abs(l[u] - l[v])
            s += d
            if d > m:
                m = d
        return m, s

    assert feas(lab), "initial labeling violates the J1 cap"
    m, c = cost(lab)
    best, bm, bc = lab[:], m, c
    t_end = time.time() + t_budget
    T = 2.0
    while time.time() < t_end:
        for _ in range(2000):
            a, b = rng.randrange(n), rng.randrange(n)
            if a == b:
                continue
            lab[a], lab[b] = lab[b], lab[a]
            if not feas(lab):
                lab[a], lab[b] = lab[b], lab[a]
                continue
            m2, c2 = cost(lab)
            if (m2, c2) <= (m, c) or \
               rng.random() < math.exp(-((m2 - m) * 4 + (c2 - c) * 0.001) / T):
                m, c = m2, c2
                if (m, c) < (bm, bc):
                    best, bm, bc = lab[:], m, c
            else:
                lab[a], lab[b] = lab[b], lab[a]
        T = max(0.05, T * 0.95)
    return bm, best


def _sa_lex_worker(payload):
    """One independent lex SA chain (picklable for mp.Pool)."""
    n, e1, k1, e2, seed, t_budget, init = payload
    return sa_lex(n, e1, k1, e2, init, seed, t_budget)


def run_lex_heuristic(n, e1, k1, e2, init, total_time, procs, seed):
    """Parallel multi-start SA for the lex (J1-capped) problem: minimize the J2
    bandwidth over labelings with J1 bandwidth <= k1. Half the chains warm-start
    from the running best, half from the supplied (J1-feasible) init. Returns
    (best_j2_bandwidth, best_labeling)."""
    best_lab = init[:]
    best_bw = max(abs(init[u] - init[v]) for u, v in e2) if e2 else 0
    print(f"[lex-heuristic] start J2 bandwidth: {best_bw} "
          f"(procs={procs}, time={total_time:.0f}s)")
    if total_time <= 0 or procs <= 1:
        bw, lab = sa_lex(n, e1, k1, e2, best_lab, seed, total_time)
        return (bw, lab) if bw < best_bw else (best_bw, best_lab)
    t_end = time.time() + total_time
    batch = max(15.0, total_time / 8)
    r = 0
    with mp.Pool(procs) as pool:
        while time.time() < t_end:
            dur = min(batch, t_end - time.time())
            if dur <= 1:
                break
            jobs = [(n, e1, k1, e2, seed + r * procs + p, dur,
                     best_lab if p % 2 == 0 else init) for p in range(procs)]
            r += 1
            for bw, lab in pool.imap_unordered(_sa_lex_worker, jobs):
                if bw < best_bw:
                    best_bw, best_lab = bw, lab
                    print(f"[lex-heuristic] new J2 bandwidth: {best_bw}")
    return best_bw, best_lab


def cmd_lex_heuristic(args):
    n, e1, e2, k1, st2 = lex_setup(args)
    print(f"[lex] heuristic with hard J1 cap k1={k1}; J2 has {len(e2)} edges")
    st2.record_math_lb(combinatorial_lb(n, e2))
    cur = st2.read()
    # resumability: an interrupted sweep re-runs this command. If this cap is
    # already certified there is nothing left to seed -- exit instead of
    # burning the full --time again.
    lb0, ub0 = State.window(cur)
    if ub0 is not None and lb0 >= ub0:
        print(f"[lex-heuristic] k2 already certified for this cap (k2*={ub0}); "
              f"nothing to seed — skipping")
        return
    # start from the current lex best if J1-feasible, else the J1 ordering,
    # else a CP-SAT feasible labeling under the J1 cap.
    init = cur["best_labeling"]
    if init is None or any(abs(init[u] - init[v]) > k1 for u, v in e1):
        init = State(args.state_dir, args.cluster, n, e1).read()["best_labeling"]
    if init is None or any(abs(init[u] - init[v]) > k1 for u, v in e1):
        res, init, _ = cpsat_decide_caps(
            n, e1, [k1] * len(e1), min(600, args.time), args.workers,
            symmetry=args.symmetry, fix_label1=args.fix_label1)
        if res != "SAT":
            sys.exit(f"could not find any labeling with J1 bandwidth <= {k1}")
    bm, lab = run_lex_heuristic(n, e1, k1, e2, init, args.time, args.procs, args.seed)
    st2.record_labeling(lab, "lex-heuristic")
    print_status(st2.read(), e2)


def cmd_lex_ladder(args):
    n, e1, e2, k1, st2 = lex_setup(args)
    print(f"[lex] stage-2 with hard J1 cap k1={k1}; J2 has {len(e2)} edges")
    lb2 = combinatorial_lb(n, e2)          # unconstrained LB on the J2 graph is valid
    st2.record_math_lb(lb2)
    st2.record_unsat(lb2 - 1, "math")
    cur = st2.read()
    if cur["best_labeling"] is None:
        st1 = State(args.state_dir, args.cluster, n, e1).read()
        seedlab = st1["best_labeling"]
        if seedlab is None or bandwidth_of(seedlab, e1) > k1:
            res, seedlab, _ = cpsat_decide_caps(
                n, e1, [k1] * len(e1), args.time_per_k, args.workers,
                symmetry=args.symmetry, fix_label1=args.fix_label1)
            if res != "SAT":
                sys.exit(f"could not find any labeling with J1 bandwidth <= {k1}")
        bm, lab = sa_lex(n, e1, k1, e2, seedlab, args.seed,
                         min(120, args.time_per_k))
        st2.record_labeling(lab, "lex-heuristic")
        cur = st2.read()
    side = 0
    while True:
        lb, ub = State.window(cur)
        if lb >= ub:
            break
        k2 = ub - 1 if side == 0 else lb
        print(f"[lex-ladder] k2 window [{lb},{ub}] -> trying k2={k2}")
        E, caps = lex_caps(e1, e2, k1, k2)
        hint = cur["best_labeling"] if side == 0 else None
        res, lab, _ = cpsat_decide_caps(n, E, caps, args.time_per_k,
                                        args.workers, hint=hint,
                                        symmetry=args.symmetry,
                                        fix_label1=args.fix_label1)
        if res == "SAT":
            st2.record_labeling(lab, f"lex-ladder(k2={k2})")
        elif res == "UNSAT":
            st2.record_unsat(k2, "cpsat")
        else:
            if side == 1:
                print("[lex-ladder] both sides hard at this budget; stopping")
                break
            side = 1
            cur = st2.read()
            continue
        side ^= 1
        cur = st2.read()
    print_status(st2.read(), e2)
    lb, ub = State.window(st2.read())
    if lb >= ub:
        print(f"[lex] CERTIFIED lexicographic optimum: (k1, k2) = ({k1}, {ub})")


def cmd_lex_decide(args):
    n, e1, e2, k1, st2 = lex_setup(args)
    E, caps = lex_caps(e1, e2, k1, args.k)
    hint = st2.read()["best_labeling"]
    res, lab, note = cpsat_decide_caps(n, E, caps, args.time, args.workers,
                                       hint=hint, symmetry=args.symmetry,
                                       fix_label1=args.fix_label1)
    print(f"[lex-decide k1={k1} k2={args.k}] {res}  (symmetry: {note})")
    if res == "SAT":
        st2.record_labeling(lab, f"lex-decide(k2={args.k})")
    elif res == "UNSAT":
        st2.record_unsat(args.k, "cpsat")
    print_status(st2.read(), e2)


def cmd_lex_cnf(args):
    n, e1, e2, k1, _ = lex_setup(args)
    E, caps = lex_caps(e1, e2, k1, args.k)
    nv, nc, note = write_cnf(n, E, args.k, args.out,
                             fix_label1=args.fix_label1,
                             symmetry=args.symmetry, caps=caps)
    print(f"wrote {args.out}: {nv} vars, {nc} clauses "
          f"(k1={k1} on J1, k2={args.k} on J2; symmetry: {note})")


def cmd_lex_verify(args):
    n, e1, e2, k1, st2 = lex_setup(args)
    E, caps = lex_caps(e1, e2, k1, args.k)
    res, lab, agree, note = crosscheck_unsat(
        n, E, args.k, args.time, fix_label1=args.fix_label1,
        symmetry=args.symmetry, proof_out=args.proof_out,
        cnf_out=args.cnf_out, caps=caps)
    print(f"[lex-verify k1={k1} k2={args.k}] {res}")
    if res == "UNSAT" and agree:
        st2.record_unsat(args.k, "xsat")
    elif res == "SAT" and lab is not None:
        st2.record_labeling(lab, f"lex-pysat(k2={args.k})")
    print_status(st2.read(), e2)


def cmd_lex_export(args):
    n, e1, e2, k1, st2 = lex_setup(args)
    cur = st2.read()
    if not cur["best_labeling"]:
        sys.exit("no labeling in the lex state yet")
    lab = cur["best_labeling"]
    print("{" + ", ".join(f"{v}: {lab[v] - 1}" for v in range(n)) + "}")


# ======================================================================
# SPIN-GLASS / DENSE WEIGHTED MODE  (bottleneck objective max w_e * d_e)
# ======================================================================
# Input: dense float64 J matrix (.npy or whitespace text). Weights are
# |J| symmetrized, entries <= cutoff * max|J| dropped, then scaled to
# integers w_int = round(|J|/max * 10**digits). Objective values live on
# the finite grid {w_int * d}; the ladder walks that grid. State stores
# scaled-integer objective values; reports also show original units.

def load_jmatrix(path, cutoff, digits):
    import numpy as np
    M = np.load(path) if path.endswith(".npy") else np.loadtxt(path)
    M = np.abs(np.asarray(M, dtype=float))
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        sys.exit("J matrix must be square")
    n = M.shape[0]
    M = (M + M.T) / 2.0
    np.fill_diagonal(M, 0.0)
    mx = float(M.max())
    if mx <= 0:
        sys.exit("J matrix has no nonzero off-diagonal entries")
    scale = (10 ** digits) / mx
    edges, weights = [], []
    for u in range(n):
        for v in range(u + 1, n):
            if M[u, v] > cutoff * mx:
                w = int(round(M[u, v] * scale))
                if w > 0:
                    edges.append((u, v))
                    weights.append(w)
    return n, edges, weights, mx


def w_setup(args):
    n, edges, weights, mx = load_jmatrix(args.jmatrix, args.cutoff, args.digits)
    name = args.name or os.path.splitext(os.path.basename(args.jmatrix))[0]
    st = State(args.state_dir, f"{name}__wb", n, edges, weights=weights)
    to_orig = mx / (10 ** args.digits)
    return n, edges, weights, st, to_orig


def w_grid(n, weights):
    return sorted({w * d for w in set(weights) for d in range(1, n)})


def w_caps(weights, K):
    return [K // w for w in weights]


def w_math_lb(n, edges, weights):
    """Weighted degree bound: among the j heaviest edges at v, one endpoint
    sits at label distance >= ceil(j/2), and its weight >= w_(j)."""
    inc = [[] for _ in range(n)]
    for (u, v), w in zip(edges, weights):
        inc[u].append(w)
        inc[v].append(w)
    lb = 1
    for v in range(n):
        ws = sorted(inc[v], reverse=True)
        for j, w in enumerate(ws, start=1):
            lb = max(lb, w * ((j + 1) // 2))
    return lb


def w_window(st_dict, grid):
    """(largest proven-infeasible K, current ub, certified?)."""
    X = max((int(k) for k in st_dict["unsat"]), default=0)
    if st_dict.get("math_lb"):
        X = max(X, st_dict["math_lb"] - 1)
    ub = st_dict["ub"]
    nxt = next((g for g in grid if g > X), None)
    return X, ub, (ub is not None and nxt is not None and ub <= nxt)


def sa_weighted(n, edges, weights, init, seed, t_budget):
    rng = random.Random(seed)
    lab = init[:] if init else [i + 1 for i in range(n)]
    if not init:
        rng.shuffle(lab)

    def cost(l):
        m = s = 0
        for (u, v), w in zip(edges, weights):
            x = w * abs(l[u] - l[v])
            s += x
            if x > m:
                m = x
        return m, s

    m, c = cost(lab)
    best, bm, bc = lab[:], m, c
    t_end = time.time() + t_budget
    T = max(2.0, bm / 8)
    while time.time() < t_end:
        for _ in range(1000):
            a, b = rng.randrange(n), rng.randrange(n)
            if a == b:
                continue
            lab[a], lab[b] = lab[b], lab[a]
            m2, c2 = cost(lab)
            if (m2, c2) <= (m, c) or \
               rng.random() < math.exp(-((m2 - m) / max(1, bm / 20)
                                         + (c2 - c) * 0.001) / T):
                m, c = m2, c2
                if (m, c) < (bm, bc):
                    best, bm, bc = lab[:], m, c
            else:
                lab[a], lab[b] = lab[b], lab[a]
        T = max(0.05, T * 0.97)
    return bm, best


def cmd_w_ladder(args):
    n, edges, weights, st, to_orig = w_setup(args)
    grid = w_grid(n, weights)
    print(f"[w] n={n}, kept {len(edges)} weighted edges, "
          f"{len(set(weights))} distinct weights, grid size {len(grid)}")
    lbm = w_math_lb(n, edges, weights)
    st.record_math_lb(lbm)
    print(f"[w] weighted degree lower bound: {lbm} "
          f"(= {lbm * to_orig:.6g} in original units)")
    cur = st.read()
    if cur["best_labeling"] is None:
        bm, lab = sa_weighted(n, edges, weights, None, args.seed,
                              args.heur_time)
        st.record_labeling(lab, "w-heuristic")
        cur = st.read()
    while True:
        X, ub, closed = w_window(cur, grid)
        print(f"[w-ladder] proven > {X} ({X * to_orig:.6g}), "
              f"ub = {ub} ({(ub or 0) * to_orig:.6g})")
        if closed:
            break
        sat_cand = max((g for g in grid if g < ub), default=None)
        unsat_cand = next((g for g in grid if g > X), None)
        K = sat_cand if (args.side == "sat" or unsat_cand is None) \
            else unsat_cand if args.side == "unsat" \
            else (sat_cand if (ub - sat_cand) <= (unsat_cand - X) else unsat_cand)
        print(f"[w-ladder] deciding K = {K} ({K * to_orig:.6g})")
        hint = cur["best_labeling"]
        res, lab, _ = cpsat_decide_caps(n, edges, w_caps(weights, K),
                                        args.time_per_k, args.workers,
                                        hint=hint, symmetry=args.symmetry,
                                        fix_label1=args.fix_label1)
        if res == "SAT":
            st.record_labeling(lab, f"w-ladder(K={K})")
        elif res == "UNSAT":
            st.record_unsat(K, "cpsat")
        else:
            print("[w-ladder] timed out; raise --time-per-k or farm w-decide jobs")
            break
        cur = st.read()
    X, ub, closed = w_window(st.read(), grid)
    if closed:
        print(f"[w] CERTIFIED optimum: K* = {ub} scaled "
              f"= {ub * to_orig:.8g} in original units "
              f"(decisive: nothing on the grid in ({X}, {ub}))")
    else:
        print(f"[w] open window: proven > {X}, best {ub}")


def cmd_w_decide(args):
    n, edges, weights, st, to_orig = w_setup(args)
    K = int(round(args.kval / to_orig)) if args.original_units else int(args.kval)
    res, lab, note = cpsat_decide_caps(n, edges, w_caps(weights, K),
                                       args.time, args.workers,
                                       hint=st.read()["best_labeling"],
                                       symmetry=args.symmetry,
                                       fix_label1=args.fix_label1)
    print(f"[w-decide K={K} ({K * to_orig:.6g})] {res}  (symmetry: {note})")
    if res == "SAT":
        st.record_labeling(lab, f"w-decide(K={K})")
    elif res == "UNSAT":
        st.record_unsat(K, "cpsat")


def cmd_w_cnf(args):
    n, edges, weights, st, to_orig = w_setup(args)
    K = int(round(args.kval / to_orig)) if args.original_units else int(args.kval)
    nv, nc, note = write_cnf(n, edges, K, args.out,
                             fix_label1=args.fix_label1,
                             symmetry=args.symmetry,
                             caps=w_caps(weights, K))
    print(f"wrote {args.out}: {nv} vars, {nc} clauses (K={K}; symmetry: {note})")


def cmd_w_verify(args):
    n, edges, weights, st, to_orig = w_setup(args)
    K = int(round(args.kval / to_orig)) if args.original_units else int(args.kval)
    res, lab, agree, note = crosscheck_unsat(
        n, edges, K, args.time, fix_label1=args.fix_label1,
        symmetry=args.symmetry, proof_out=args.proof_out,
        cnf_out=args.cnf_out, caps=w_caps(weights, K))
    print(f"[w-verify K={K}] {res}")
    if res == "UNSAT" and agree:
        st.record_unsat(K, "xsat")
    elif res == "SAT" and lab is not None:
        st.record_labeling(lab, f"w-pysat(K={K})")


def cmd_w_export(args):
    n, edges, weights, st, to_orig = w_setup(args)
    cur = st.read()
    if not cur["best_labeling"]:
        sys.exit("no labeling in the weighted state yet")
    lab = cur["best_labeling"]
    print("{" + ", ".join(f"{v}: {lab[v] - 1}" for v in range(n)) + "}")




# ======================================================================
# 2-LEG LADDER HOST MODE  (open ends)
# ======================================================================
# Place the n vertices bijectively on a 2 x (n/2) open ladder: positions
# p = 2x + y with x in 0..m-1 (m = n/2), y in {0, 1}; host distance is
# d((x1,y1),(x2,y2)) = |x1-x2| + |y1-y2| (= graph distance on the ladder).
# Objective: minimize k = max over edges of the host distance. The decision
# problem sat(G, k) is monotone in k, so the standard certification ladder,
# state file, and SAT cross-check all apply. State name: <cluster>__h2.
# Labelings stored in the state are 1-based positions lab[v] = p + 1.

def h2_m(n):
    if n % 2:
        sys.exit("2-leg ladder host requires an even number of vertices")
    return n // 2


def h2_dist(p, q):
    return abs(p // 2 - q // 2) + abs(p % 2 - q % 2)


def h2_value(lab, edges):
    return max(h2_dist(lab[u] - 1, lab[v] - 1) for u, v in edges)


def h2_ballmax(n, R):
    """max over host nodes p of |B(p, R)| on the 2 x (n/2) ladder."""
    return max(sum(1 for q in range(n) if h2_dist(p, q) <= R)
               for p in range(n))


def h2_math_lb(n, edges):
    m = h2_m(n)
    adj = adjacency(n, edges)
    Delta = max(len(a) for a in adj)
    # degree/ball bound: some node must hold Delta neighbors within radius k
    k_deg = 1
    while h2_ballmax(n, k_deg) - 1 < Delta:
        k_deg += 1
    # diameter bound: host diameter m must be covered by <= diam(G) edges
    diam_g = 0
    for v in range(n):
        d = bfs_dist(adj, v)
        if min(d) < 0:
            diam_g = None                    # disconnected: skip this bound
            break
        diam_g = max(diam_g, max(d))
    k_diam = ceil_div(m, diam_g) if diam_g else 1
    # subgraph ball bound: |B_G(v, r)| <= |B_H(p, r*k)| for some p
    ballmax = [h2_ballmax(n, R) for R in range(0, m + 3)]

    def need_k(cnt, r):
        k = 1
        while ballmax[min(r * k, m + 2)] < cnt:
            k += 1
        return k

    k_ball = 1
    for v in range(n):
        d = bfs_dist(adj, v)
        reach = max(x for x in d if x >= 0)
        for r in range(1, reach + 1):
            cnt = sum(1 for x in d if 0 <= x <= r)
            k_ball = max(k_ball, need_k(cnt, r))
    return max(1, k_deg, k_diam, k_ball)


def h2_symmetry(model, X, Y, n, edges, fix_label1, reps=None):
    """Host automorphisms of the open ladder: x-reversal and leg flip.
    With fix_label1 (vertex-transitive G) or orbit reps, pin to corner (0,0),
    which simultaneously breaks both host reflections."""
    m = h2_m(n)
    if fix_label1 is not None:
        model.Add(X[fix_label1] == 0)
        model.Add(Y[fix_label1] == 0)
        return f"vertex {fix_label1} fixed to corner (0,0)"
    if reps:
        if len(reps) == 1:
            model.Add(X[reps[0]] == 0)
            model.Add(Y[reps[0]] == 0)
            return f"vertex-transitive: vertex {reps[0]} fixed to corner (0,0)"
        bools = []
        for r in reps:
            b = model.NewBoolVar(f"rep{r}")
            model.Add(X[r] == 0).OnlyEnforceIf(b)
            model.Add(Y[r] == 0).OnlyEnforceIf(b)
            bools.append(b)
        model.AddBoolOr(bools)
        return f"corner (0,0) occupied by one of {len(reps)} orbit reps"
    adj = adjacency(n, edges)
    w = max(range(n), key=lambda v: len(adj[v]))
    model.Add(X[w] <= (m - 1) // 2)
    model.Add(Y[w] == 0)
    return f"host reflections: vertex {w} pinned to x<= {(m - 1) // 2}, leg 0"


def cpsat_decide_h2(n, edges, k, time_limit, workers, hint=None,
                    symmetry="reversal", fix_label1=None, log=False):
    from ortools.sat.python import cp_model
    m = h2_m(n)
    model = cp_model.CpModel()
    X = [model.NewIntVar(0, m - 1, f"x{v}") for v in range(n)]
    Y = [model.NewIntVar(0, 1, f"y{v}") for v in range(n)]
    P = [model.NewIntVar(0, n - 1, f"p{v}") for v in range(n)]
    for v in range(n):
        model.Add(P[v] == 2 * X[v] + Y[v])
    model.AddAllDifferent(P)
    for u, v in edges:
        dx = model.NewIntVar(0, k, f"dx{u}_{v}")
        dy = model.NewIntVar(0, min(1, k), f"dy{u}_{v}")
        model.AddAbsEquality(dx, X[u] - X[v])
        model.AddAbsEquality(dy, Y[u] - Y[v])
        model.Add(dx + dy <= k)
    reps = None
    if symmetry == "orbit" and fix_label1 is None:
        res = orbit_representatives(n, edges)
        if res is not None:
            reps = res[0]
    note = h2_symmetry(model, X, Y, n, edges, fix_label1, reps)
    if hint is not None:
        for v in range(n):
            p = hint[v] - 1
            model.AddHint(X[v], p // 2)
            model.AddHint(Y[v], p % 2)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    if log:
        solver.parameters.log_search_progress = True
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        lab = [2 * solver.Value(X[v]) + solver.Value(Y[v]) + 1 for v in range(n)]
        return "SAT", lab, note
    if status == cp_model.INFEASIBLE:
        return "UNSAT", None, note
    return "UNKNOWN", None, note


def build_cnf_h2(n, edges, k, fix_label1=None, symmetry="reversal"):
    """One-hot CNF over ladder positions; x_{v,i} <=> v at position i-1.
    Same variable convention as the path encoding, so decode_model works."""
    m = h2_m(n)

    def var(v, i):
        return v * n + i

    clauses = []
    next_aux = n * n + 1

    def amo_sequential(lits):
        nonlocal next_aux
        L = len(lits)
        if L <= 1:
            return
        s = list(range(next_aux, next_aux + L - 1))
        next_aux += L - 1
        clauses.append([-lits[0], s[0]])
        for i in range(1, L - 1):
            clauses.append([-lits[i], s[i]])
            clauses.append([-s[i - 1], s[i]])
            clauses.append([-lits[i], -s[i - 1]])
        clauses.append([-lits[L - 1], -s[L - 2]])

    for v in range(n):
        lits = [var(v, i) for i in range(1, n + 1)]
        clauses.append(lits[:])
        amo_sequential(lits)
    for i in range(1, n + 1):
        lits = [var(v, i) for v in range(n)]
        clauses.append(lits[:])
        amo_sequential(lits)

    for u, v in edges:
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i != j and h2_dist(i - 1, j - 1) > k:
                    clauses.append([-var(u, i), -var(v, j)])

    reps = None
    if symmetry == "orbit" and fix_label1 is None:
        res = orbit_representatives(n, edges)
        if res is not None:
            reps = res[0]
    if fix_label1 is not None:
        clauses.append([var(fix_label1, 1)])           # position 0 = corner (0,0)
        note = f"vertex {fix_label1} fixed to corner (0,0)"
    elif reps:
        clauses.append([var(r, 1) for r in reps])
        note = f"corner (0,0) occupied by one of {len(reps)} orbit reps"
    else:
        adj = adjacency(n, edges)
        w = max(range(n), key=lambda v: len(adj[v]))
        for i in range(1, n + 1):
            p = i - 1
            if p // 2 > (m - 1) // 2 or p % 2 == 1:
                clauses.append([-var(w, i)])
        note = f"host reflections: vertex {w} pinned to x<= {(m - 1) // 2}, leg 0"
    return clauses, next_aux - 1, note


def sa_h2(n, edges, init, seed, t_budget):
    rng = random.Random(seed)
    lab = init[:] if init else [i + 1 for i in range(n)]
    if not init:
        rng.shuffle(lab)

    def cost(l):
        mx = s = 0
        for u, v in edges:
            d = h2_dist(l[u] - 1, l[v] - 1)
            s += d
            if d > mx:
                mx = d
        return mx, s

    mx, c = cost(lab)
    best, bm, bc = lab[:], mx, c
    t_end = time.time() + t_budget
    T = max(2.0, bm / 4)
    while time.time() < t_end:
        for _ in range(2000):
            a, b = rng.randrange(n), rng.randrange(n)
            if a == b:
                continue
            lab[a], lab[b] = lab[b], lab[a]
            m2, c2 = cost(lab)
            if (m2, c2) <= (mx, c) or \
               rng.random() < math.exp(-((m2 - mx) * 4 + (c2 - c) * 0.001) / T):
                mx, c = m2, c2
                if (mx, c) < (bm, bc):
                    best, bm, bc = lab[:], mx, c
            else:
                lab[a], lab[b] = lab[b], lab[a]
        T = max(0.05, T * 0.95)
    return bm, best


def h2_state(args, n, edges):
    return State(args.state_dir, f"{args.cluster or 'edgefile'}__h2", n, edges,
                 dist=h2_dist)


def cmd_h2_run(args):
    n, edges = get_graph(args)
    m = h2_m(n)
    st = h2_state(args, n, edges)
    lbm = h2_math_lb(n, edges)
    st.record_math_lb(lbm)
    st.record_unsat(lbm - 1, "math")
    print(f"[h2] 2 x {m} open ladder host; math lower bound {lbm}")
    cur = st.read()
    if cur["best_labeling"] is None:
        bm, lab = sa_h2(n, edges, None, args.seed, args.heur_time)
        st.record_labeling(lab, "h2-heuristic")
        cur = st.read()
    side = 0
    while True:
        lb, ub = State.window(cur)
        if lb >= ub:
            break
        k = ub - 1 if side == 0 else lb
        print(f"[h2-run] window [{lb},{ub}] -> deciding k={k}")
        hint = cur["best_labeling"] if side == 0 else None
        res, lab, _ = cpsat_decide_h2(n, edges, k, args.time_per_k,
                                      args.workers, hint=hint,
                                      symmetry=args.symmetry,
                                      fix_label1=args.fix_label1)
        if res == "SAT":
            st.record_labeling(lab, f"h2-run(k={k})")
        elif res == "UNSAT":
            st.record_unsat(k, "cpsat")
        else:
            if side == 1:
                print("[h2-run] both sides hard at this budget; stopping")
                break
            side = 1
            cur = st.read()
            continue
        side ^= 1
        cur = st.read()
    cur = st.read()
    lb, ub = State.window(cur)
    print(f"[h2] certified window: {lb} <= k* <= {ub}")
    if ub is not None and lb >= ub:
        print(f"[h2] *** CERTIFIED OPTIMAL ladder dilation: k* = {ub} ***")


def cmd_h2_decide(args):
    n, edges = get_graph(args)
    st = h2_state(args, n, edges)
    hint = st.read()["best_labeling"]
    res, lab, note = cpsat_decide_h2(n, edges, args.k, args.time, args.workers,
                                     hint=hint, symmetry=args.symmetry,
                                     fix_label1=args.fix_label1)
    print(f"[h2-decide k={args.k}] {res}  (symmetry: {note})")
    if res == "SAT":
        st.record_labeling(lab, f"h2-decide(k={args.k})")
    elif res == "UNSAT":
        st.record_unsat(args.k, "cpsat")
    lb, ub = State.window(st.read())
    print(f"[h2] window now [{lb},{ub}]")


def cmd_h2_cnf(args):
    n, edges = get_graph(args)
    clauses, nvars, note = build_cnf_h2(n, edges, args.k,
                                        fix_label1=args.fix_label1,
                                        symmetry=args.symmetry)
    write_dimacs(args.out, clauses, nvars,
                 [f"2-leg ladder host decision, k={args.k}, n={n}, "
                  f"|E|={len(edges)}",
                  f"var(v,i)=v*n+i, position p=i-1=(2x+y)",
                  f"symmetry: {note}"])
    print(f"wrote {args.out}: {nvars} vars, {len(clauses)} clauses "
          f"(symmetry: {note})")


def cmd_h2_verify(args):
    n, edges = get_graph(args)
    st = h2_state(args, n, edges)
    pre = build_cnf_h2(n, edges, args.k, fix_label1=args.fix_label1,
                       symmetry=args.symmetry)
    res, lab, agree, note = crosscheck_unsat(
        n, edges, args.k, args.time, proof_out=args.proof_out,
        cnf_out=args.cnf_out, prebuilt=pre)
    print(f"[h2-verify k={args.k}] {res}")
    if res == "UNSAT" and agree:
        st.record_unsat(args.k, "xsat")
    elif res == "SAT" and lab is not None:
        if h2_value(lab, edges) <= args.k:
            st.record_labeling(lab, f"h2-pysat(k={args.k})")
    lb, ub = State.window(st.read())
    print(f"[h2] window now [{lb},{ub}]")


def cmd_h2_export(args):
    n, edges = get_graph(args)
    st = h2_state(args, n, edges)
    cur = st.read()
    if not cur["best_labeling"]:
        sys.exit("no ladder embedding in the state yet")
    lab = cur["best_labeling"]
    print("ladder dilation of this embedding:", h2_value(lab, edges))
    print("{" + ", ".join(f"{v}: ({(lab[v] - 1) // 2}, {(lab[v] - 1) % 2})"
                          for v in range(n)) + "}")




# ======================================================================
# SUPERSITE MODE: block q spins per chain position (default q = 2: rungs)
# ======================================================================
# Partition the n vertices into m = n/q blocks (supersites), order the
# blocks on an open chain, and minimize  k = max_{(u,v) in E} |r(u)-r(v)|
# where r(v) is the block index. The partition is part of the optimization
# (equivalently: a load-q embedding into the path P_m, minimizing dilation).
# Intra-block edges cost 0. Decision sat(G,k) is monotone in k -> the same
# ladder/state/certificate machinery applies. Labelings in the state are
# 1-based block indices lab[v] = r + 1, each value used exactly q times.
# State name: <cluster>__ss<q>.
#
# Hidden-bond constraints (optional): --min-intra-edges N requires >= N edges
# with both endpoints in one block; --intra-per-block requires EVERY block to
# contain at least one edge (for q=2 the blocking is then a perfect matching
# of G along bonds). Neither constraint involves k, so sat(G,k) stays monotone
# and the certificates remain valid FOR THE CONSTRAINED PROBLEM. Constrained
# runs use their own state file (suffix _ie<N> / _ipb): their UNSAT proofs do
# not transfer to the unconstrained problem, and vice versa.

def ss_m(n, q):
    if n % q:
        sys.exit(f"supersite mode with block size {q} requires q | n")
    return n // q


def ss_value(lab, edges):
    return max(abs(lab[u] - lab[v]) for u, v in edges) if edges else 0


def ss_intra_report(lab, edges, q):
    """(n_intra, uncovered): edges hidden inside blocks, and blocks with no
    internal edge."""
    m = len(lab) // q
    covered = set()
    n_intra = 0
    for u, v in edges:
        if lab[u] == lab[v]:
            n_intra += 1
            covered.add(lab[u])
    return n_intra, m - len(covered)


def ss_viol(lab, edges, q, min_intra, per_block):
    """Total hidden-bond constraint violation of a labeling (0 = satisfied)."""
    n_intra, uncovered = ss_intra_report(lab, edges, q)
    v = max(0, min_intra - n_intra)
    if per_block:
        v += uncovered
    return v


def ss_constraints(args, edges):
    """Validated (min_intra, per_block) from the CLI flags."""
    mi = getattr(args, "min_intra_edges", 0) or 0
    pb = bool(getattr(args, "intra_per_block", False))
    if mi < 0:
        sys.exit("--min-intra-edges must be >= 0")
    if mi > len(edges):
        sys.exit(f"--min-intra-edges {mi} exceeds |E| = {len(edges)}")
    return mi, pb


def ss_math_lb(n, edges, q):
    m = ss_m(n, q)
    adj = adjacency(n, edges)
    Delta = max(len(a) for a in adj)
    # ball bound: blocks within distance k hold q*(2k+1) spins incl. v itself
    k_deg = 1
    while q * (2 * k_deg + 1) - 1 < Delta:
        k_deg += 1
    # diameter bound: block distance m-1 must be covered by <= diam(G) edges
    diam_g = 0
    for v in range(n):
        d = bfs_dist(adj, v)
        if min(d) < 0:
            diam_g = None
            break
        diam_g = max(diam_g, max(d))
    k_diam = ceil_div(m - 1, diam_g) if diam_g else 1
    # subgraph ball bound: |B_G(v,r)| <= q*(2rk+1)
    k_ball = 1
    for v in range(n):
        d = bfs_dist(adj, v)
        reach = max(x for x in d if x >= 0)
        for r in range(1, reach + 1):
            cnt = sum(1 for x in d if 0 <= x <= r)
            while q * (2 * r * k_ball + 1) < cnt:
                k_ball += 1
    return max(1, k_deg, k_diam, k_ball)


def cpsat_decide_ss(n, edges, q, k, time_limit, workers, hint=None,
                    symmetry="reversal", fix_label1=None, log=False,
                    min_intra=0, intra_per_block=False):
    from ortools.sat.python import cp_model
    m = ss_m(n, q)
    model = cp_model.CpModel()
    R = [model.NewIntVar(0, m - 1, f"r{v}") for v in range(n)]
    # exactly q vertices per block, via channeling booleans
    B = {}
    for v in range(n):
        for r in range(m):
            b = model.NewBoolVar(f"b{v}_{r}")
            model.Add(R[v] == r).OnlyEnforceIf(b)
            model.Add(R[v] != r).OnlyEnforceIf(b.Not())
            B[v, r] = b
    for r in range(m):
        model.Add(sum(B[v, r] for v in range(n)) == q)
    d_of = {}
    for u, v in edges:
        d = model.NewIntVar(0, k, f"d{u}_{v}")
        model.AddAbsEquality(d, R[u] - R[v])
        d_of[u, v] = d
    # hidden-bond constraints: force bonds inside supersites (see section
    # header). Neither involves k, so the decision stays monotone in k.
    cnotes = []
    if min_intra:
        bs = []
        for (u, v), d in d_of.items():
            b = model.NewBoolVar(f"in{u}_{v}")
            model.Add(d == 0).OnlyEnforceIf(b)
            model.Add(d != 0).OnlyEnforceIf(b.Not())
            bs.append(b)
        model.Add(sum(bs) >= min_intra)
        cnotes.append(f">={min_intra} intra-block edges")
    if intra_per_block:
        # e => both endpoints in block r; one true e per block suffices
        for r in range(m):
            lits = []
            for u, v in edges:
                e = model.NewBoolVar(f"e{u}_{v}_{r}")
                model.AddImplication(e, B[u, r])
                model.AddImplication(e, B[v, r])
                lits.append(e)
            model.AddBoolOr(lits)
        cnotes.append("every block has an internal edge")
    # symmetry: block-order reversal; with vertex-transitivity, pin to block 0
    reps = None
    if symmetry == "orbit" and fix_label1 is None:
        res = orbit_representatives(n, edges)
        if res is not None:
            reps = res[0]
    if fix_label1 is not None:
        model.Add(R[fix_label1] == 0)
        note = f"vertex {fix_label1} fixed to block 0"
    elif reps and len(reps) == 1:
        model.Add(R[reps[0]] == 0)
        note = f"vertex-transitive: vertex {reps[0]} fixed to block 0"
    elif reps:
        bools = []
        for rr in reps:
            bb = model.NewBoolVar(f"rep{rr}")
            model.Add(R[rr] == 0).OnlyEnforceIf(bb)
            bools.append(bb)
        model.AddBoolOr(bools)
        note = f"block 0 contains one of {len(reps)} orbit reps"
    else:
        adj = adjacency(n, edges)
        w = max(range(n), key=lambda x: len(adj[x]))
        model.Add(R[w] <= (m - 1) // 2)
        note = f"reversal: vertex {w} pinned to blocks 0..{(m - 1) // 2}"
    if cnotes:
        note += "; " + "; ".join(cnotes)
    if hint is not None:
        for v in range(n):
            model.AddHint(R[v], hint[v] - 1)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    if log:
        solver.parameters.log_search_progress = True
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return "SAT", [solver.Value(R[v]) + 1 for v in range(n)], note
    if status == cp_model.INFEASIBLE:
        return "UNSAT", None, note
    return "UNKNOWN", None, note


def _sinz_at_most(clauses, next_aux, lits, bound):
    """Sinz sequential counter: append clauses enforcing "at most `bound` of
    `lits` are true". Auxiliary vars are numbered from `next_aux`; returns the
    updated next_aux. Shared by build_cnf_ss and build_cnf_cw."""
    L = len(lits)
    if bound >= L:
        return next_aux
    if bound == 0:
        for x in lits:
            clauses.append([-x])
        return next_aux
    s = [[0] * (bound + 1) for _ in range(L)]
    for i in range(L - 1):
        for j in range(1, bound + 1):
            s[i][j] = next_aux
            next_aux += 1
    for i in range(L - 1):
        clauses.append([-lits[i], s[i][1]])
        if i > 0:
            for j in range(1, bound + 1):
                clauses.append([-s[i - 1][j], s[i][j]])
            for j in range(2, bound + 1):
                clauses.append([-lits[i], -s[i - 1][j - 1], s[i][j]])
        if i > 0:
            clauses.append([-lits[i], -s[i - 1][bound]])
    if L >= 2:
        clauses.append([-lits[L - 1], -s[L - 2][bound]])
    return next_aux


def build_cnf_ss(n, edges, q, k, fix_label1=None, symmetry="reversal",
                 min_intra=0, intra_per_block=False):
    """One-hot CNF over blocks: x_{v,r} <=> vertex v in block r (r = 1..m).
    Exactly-one per vertex; exactly-q per block (sequential cardinality);
    forbidden pairs |r - r'| > k per edge; optional hidden-bond constraints
    (min_intra / intra_per_block, matching cpsat_decide_ss).
    Variable convention var(v,r) = v*m + r, so decoding differs from the
    path mode (positions run 1..m, repeated)."""
    m = ss_m(n, q)

    def var(v, r):
        return v * m + r

    clauses = []
    next_aux = n * m + 1

    def at_most(lits, bound):
        nonlocal next_aux
        next_aux = _sinz_at_most(clauses, next_aux, lits, bound)

    def exactly(lits, c):
        at_most(lits, c)
        at_most([-x for x in lits], len(lits) - c)   # at least c

    for v in range(n):
        lits = [var(v, r) for r in range(1, m + 1)]
        clauses.append(lits[:])                       # at least one block
        at_most(lits, 1)
    for r in range(1, m + 1):
        exactly([var(v, r) for v in range(n)], q)

    for u, v in edges:
        for r1 in range(1, m + 1):
            for r2 in range(1, m + 1):
                if abs(r1 - r2) > k:
                    clauses.append([-var(u, r1), -var(v, r2)])

    # hidden-bond constraints (mirror cpsat_decide_ss, so ss-verify checks the
    # SAME decision). e(u,v,r) => both endpoints in block r: one implication
    # direction suffices for "at least" requirements.
    cnotes = []
    if min_intra or intra_per_block:
        e_of = {}
        for (u, v) in edges:
            for r in range(1, m + 1):
                e = next_aux
                next_aux += 1
                clauses.append([-e, var(u, r)])
                clauses.append([-e, var(v, r)])
                e_of[u, v, r] = e
        if intra_per_block:
            for r in range(1, m + 1):
                clauses.append([e_of[u, v, r] for (u, v) in edges])
            cnotes.append("every block has an internal edge")
        if min_intra:
            ts = []
            for (u, v) in edges:
                t = next_aux
                next_aux += 1
                clauses.append([-t] + [e_of[u, v, r] for r in range(1, m + 1)])
                ts.append(t)
            # at least min_intra of ts true <=> at most |ts|-min_intra false
            at_most([-t for t in ts], len(ts) - min_intra)
            cnotes.append(f">={min_intra} intra-block edges")

    reps = None
    if symmetry == "orbit" and fix_label1 is None:
        res = orbit_representatives(n, edges)
        if res is not None:
            reps = res[0]
    if fix_label1 is not None:
        clauses.append([var(fix_label1, 1)])
        note = f"vertex {fix_label1} fixed to block 0"
    elif reps:
        clauses.append([var(rr, 1) for rr in reps])
        note = f"block 0 contains one of {len(reps)} orbit reps"
    else:
        adj = adjacency(n, edges)
        w = max(range(n), key=lambda x: len(adj[x]))
        for r in range((m - 1) // 2 + 2, m + 1):
            clauses.append([-var(w, r)])
        note = f"reversal: vertex {w} pinned to blocks 0..{(m - 1) // 2}"
    if cnotes:
        note += "; " + "; ".join(cnotes)
    return clauses, next_aux - 1, note


def ss_decode(n, q, model_lits):
    m = n // q
    true_vars = {x for x in model_lits if 0 < x <= n * m}
    lab = []
    for v in range(n):
        rs = [r for r in range(1, m + 1) if v * m + r in true_vars]
        if len(rs) != 1:
            sys.exit(f"vertex {v} has {len(rs)} blocks - model invalid")
        lab.append(rs[0])
    return lab


def _ss_accept_delta(v2, m2, c2, vi, mx, c):
    """SA acceptance cost delta for a supersite move that WORSENED the
    lexicographic objective (violation, then max distance, then sum). Anneal on
    the leading worse term only: a large bandwidth gain must not buy a violation
    increase (violation-first), and folding the terms into one signed sum can
    make -delta/T large-positive and overflow math.exp. Returns delta >= 0."""
    if v2 != vi:
        return (v2 - vi) * 8.0          # violation changed -> it dominates
    return (m2 - mx) * 4.0 + (c2 - c) * 0.001


def _ss_sa_chain(payload):
    """One independent supersite annealing chain (picklable for mp.Pool).
    Cost is lexicographic (constraint violation, max distance, sum): with
    hidden-bond constraints active the chain first drives the violation to 0,
    then minimizes bandwidth among feasible blockings."""
    n, edges, q, seed, t_budget, init, target, min_intra, per_block = payload
    rng = random.Random(seed)
    lab = init[:] if init else [(i // q) + 1 for i in range(n)]
    if not init:
        rng.shuffle(lab)
    m = n // q

    def cost(l):
        mx = s = 0
        for u, v in edges:
            d = abs(l[u] - l[v])
            s += d
            if d > mx:
                mx = d
        viol = 0
        if min_intra or per_block:
            covered = set()
            n_in = 0
            for u, v in edges:
                if l[u] == l[v]:
                    n_in += 1
                    covered.add(l[u])
            viol = max(0, min_intra - n_in)
            if per_block:
                viol += m - len(covered)
        return viol, mx, s

    vi, mx, c = cost(lab)
    best, bv, bm, bc = lab[:], vi, mx, c
    t_end = time.time() + t_budget
    T = max(2.0, bm / 4)
    while time.time() < t_end and (bm > target or bv > 0):
        for _ in range(2000):
            a, b = rng.randrange(n), rng.randrange(n)
            if a == b or lab[a] == lab[b]:
                continue                       # swap across blocks only
            lab[a], lab[b] = lab[b], lab[a]
            v2, m2, c2 = cost(lab)
            if (v2, m2, c2) <= (vi, mx, c):
                accept = True
            else:
                delta = _ss_accept_delta(v2, m2, c2, vi, mx, c)
                accept = delta <= 0 or rng.random() < math.exp(-delta / T)
            if accept:
                vi, mx, c = v2, m2, c2
                if (vi, mx, c) < (bv, bm, bc):
                    best, bv, bm, bc = lab[:], vi, mx, c
            else:
                lab[a], lab[b] = lab[b], lab[a]
        T = max(0.05, T * 0.95)
    return bv, bm, best


def sa_ss(n, edges, q, init, seed, t_budget, procs=1, target=0, stall=None,
          min_intra=0, per_block=False):
    """Parallel multi-start supersite SA. Returns (viol, bandwidth, labeling);
    viol > 0 means the hidden-bond constraints could not be satisfied and the
    labeling must NOT be recorded as an upper bound for the constrained run."""
    best_lab = init[:] if init else [(i // q) + 1 for i in range(n)]
    best_bw = max(abs(best_lab[u] - best_lab[v]) for u, v in edges)
    best_vi = ss_viol(best_lab, edges, q, min_intra, per_block)
    if best_vi == 0 and best_bw <= target:
        return 0, best_bw, best_lab
    if procs <= 1:
        # single chain (small graphs / explicit serial request)
        return _ss_sa_chain((n, edges, q, seed, t_budget, init, target,
                             min_intra, per_block))
    if stall is None or stall <= 0:
        stall = max(60.0, t_budget / 10)
    t_end = time.time() + t_budget
    last_improve = time.time()
    batch = max(15.0, min(stall / 2, t_budget / 4))
    r = 0
    with mp.Pool(procs) as pool:
        while time.time() < t_end and (best_bw > target or best_vi > 0):
            if time.time() - last_improve > stall:
                print(f"[ss-heuristic] no improvement for {stall:.0f}s "
                      f"({stall / 3600:.2f}h); stopping early at UB {best_bw}")
                break
            dur = min(batch, t_end - time.time())
            if dur <= 1:
                break
            jobs = [(n, edges, q, seed + r * procs + p, dur,
                     best_lab if p % 2 == 0 else None, target,
                     min_intra, per_block)
                    for p in range(procs)]
            r += 1
            for vi, bw, lab in pool.imap_unordered(_ss_sa_chain, jobs):
                if (vi, bw) < (best_vi, best_bw):
                    best_vi, best_bw, best_lab = vi, bw, lab
                    last_improve = time.time()
                    print(f"[ss-heuristic] new upper bound: {best_bw}"
                          + (f" (constraint violation {best_vi})"
                             if best_vi else ""))
    return best_vi, best_bw, best_lab


def ss_state(args, n, edges):
    # Constrained runs get their own state file: hidden-bond UNSAT proofs are
    # conditional on the constraint and must never widen/narrow the window of
    # the unconstrained problem (or of a differently-constrained one).
    name = f"{args.cluster or 'edgefile'}__ss{args.block}"
    if getattr(args, "min_intra_edges", 0):
        name += f"_ie{args.min_intra_edges}"
    if getattr(args, "intra_per_block", False):
        name += "_ipb"
    return State(args.state_dir, name, n, edges, mult=args.block)


def cmd_ss_run(args):
    n, edges = get_graph(args)
    q = args.block
    m = ss_m(n, q)
    min_intra, per_block = ss_constraints(args, edges)
    st = ss_state(args, n, edges)
    lbm = ss_math_lb(n, edges, q)     # constraint shrinks the feasible set,
    st.record_math_lb(lbm)            # so unconstrained LBs remain valid
    st.record_unsat(lbm - 1, "math")
    print(f"[ss] {m} supersites of {q} spins; math lower bound {lbm}")
    if min_intra or per_block:
        print(f"[ss] hidden-bond constraints: "
              + ", ".join(([f">={min_intra} intra edges"] if min_intra else [])
                          + (["every block internally bonded"] if per_block
                             else [])))
    cur = st.read()
    if cur["best_labeling"] is None:
        vi, bm, lab = sa_ss(n, edges, q, None, args.seed, args.heur_time,
                            procs=args.procs, target=lbm, stall=args.stall,
                            min_intra=min_intra, per_block=per_block)
        if vi == 0:
            st.record_labeling(lab, "ss-heuristic")
        else:
            print(f"[ss] heuristic ended with constraint violation {vi}; "
                  f"no upper bound recorded (CP-SAT will search from scratch)")
        cur = st.read()
    side = 0 if cur["best_labeling"] is not None else 1
    while True:
        lb, ub = State.window(cur)
        if ub is not None and lb >= ub:
            break
        if lb > m - 1:
            print("[ss-run] lower bound exceeds m-1: the hidden-bond "
                  "constraints are infeasible on this graph (e.g. no perfect "
                  "matching along bonds for q=2)")
            break
        k = ub - 1 if (side == 0 and ub is not None) else lb
        print(f"[ss-run] window [{lb},{ub}] -> deciding k={k}")
        hint = cur["best_labeling"] if side == 0 else None
        res, lab, _ = cpsat_decide_ss(n, edges, q, k, args.time_per_k,
                                      args.workers, hint=hint,
                                      symmetry=args.symmetry,
                                      fix_label1=args.fix_label1,
                                      min_intra=min_intra,
                                      intra_per_block=per_block)
        if res == "SAT":
            st.record_labeling(lab, f"ss-run(k={k})")
        elif res == "UNSAT":
            st.record_unsat(k, "cpsat")
        else:
            if side == 1:
                print("[ss-run] both sides hard at this budget; stopping")
                break
            side = 1
            cur = st.read()
            continue
        side ^= 1
        cur = st.read()
    lb, ub = State.window(st.read())
    print(f"[ss] certified window: {lb} <= k* <= {ub}")
    if ub is not None and lb >= ub:
        print(f"[ss] *** CERTIFIED OPTIMAL supersite bandwidth: k* = {ub} ***")


def cmd_ss_decide(args):
    n, edges = get_graph(args)
    min_intra, per_block = ss_constraints(args, edges)
    st = ss_state(args, n, edges)
    hint = st.read()["best_labeling"]
    res, lab, note = cpsat_decide_ss(n, edges, args.block, args.k, args.time,
                                     args.workers, hint=hint,
                                     symmetry=args.symmetry,
                                     fix_label1=args.fix_label1,
                                     min_intra=min_intra,
                                     intra_per_block=per_block)
    print(f"[ss-decide k={args.k}] {res}  (symmetry: {note})")
    if res == "SAT":
        st.record_labeling(lab, f"ss-decide(k={args.k})")
    elif res == "UNSAT":
        st.record_unsat(args.k, "cpsat")
    lb, ub = State.window(st.read())
    print(f"[ss] window now [{lb},{ub}]")


def cmd_ss_cnf(args):
    n, edges = get_graph(args)
    min_intra, per_block = ss_constraints(args, edges)
    clauses, nvars, note = build_cnf_ss(n, edges, args.block, args.k,
                                        fix_label1=args.fix_label1,
                                        symmetry=args.symmetry,
                                        min_intra=min_intra,
                                        intra_per_block=per_block)
    write_dimacs(args.out, clauses, nvars,
                 [f"supersite (q={args.block}) decision, k={args.k}, n={n}, "
                  f"|E|={len(edges)}",
                  f"var(v,r)=v*m+r, m={n // args.block}, block index r-1",
                  f"symmetry: {note}"])
    print(f"wrote {args.out}: {nvars} vars, {len(clauses)} clauses "
          f"(symmetry: {note})")


def cmd_ss_verify(args):
    n, edges = get_graph(args)
    min_intra, per_block = ss_constraints(args, edges)
    st = ss_state(args, n, edges)
    pre = build_cnf_ss(n, edges, args.block, args.k,
                       fix_label1=args.fix_label1, symmetry=args.symmetry,
                       min_intra=min_intra, intra_per_block=per_block)
    clauses, nvars, note = pre
    vd, mdl, proof = parallel_crosscheck(clauses, args.time,
                                         args.proof_out is not None)
    verdicts = list(vd.values())
    if args.cnf_out:
        write_dimacs(args.cnf_out, clauses, nvars,
                     [f"supersite q={args.block} k={args.k}", f"symmetry: {note}"])
    if args.proof_out and proof is not None:
        with open(args.proof_out, "w") as f:
            f.write("\n".join(proof) + "\n")
        print(f"[ss-verify] DRAT proof written to {args.proof_out}")
    if all(v == "UNSAT" for v in verdicts):
        st.record_unsat(args.k, "xsat")
        print(f"[ss-verify] both solvers UNSAT -> recorded (xsat)")
    elif "SAT" in verdicts and mdl is not None:
        lab = ss_decode(n, args.block, [x for x in mdl if x > 0])
        if ss_value(lab, edges) <= args.k and \
           ss_viol(lab, edges, args.block, min_intra, per_block) == 0:
            st.record_labeling(lab, f"ss-pysat(k={args.k})")
    lb, ub = State.window(st.read())
    print(f"[ss] window now [{lb},{ub}]")


def cmd_ss_blocks(args):
    """Emit the COMPLETE physical structure of a certified supersite solution,
    i.e. every term of the full Hamiltonian, partitioned by the chain layout:
      - intra_supersite: edges of G with both endpoints in one block (the
        on-site terms of each d^q-dimensional supersite);
      - inter_supersite: ALL remaining edges, grouped by the (ordered) pair of
        supersites they connect, annotated with their chain distance.
    No edge is ever filtered out: sum of all listed edges == |E(G)|.
    Optionally annotates every edge with its coupling from --jmatrix.
    Output is JSON to stdout or --out."""
    import json
    n, edges = get_graph(args)
    q = args.block
    m = n // q
    st = ss_state(args, n, edges)
    cur = st.read()
    if not cur["best_labeling"]:
        sys.exit("no supersite assignment in the state yet")
    lab = cur["best_labeling"]                       # lab[v] = block index + 1
    blk_of = [lab[v] - 1 for v in range(n)]
    blocks = [[v for v in range(n) if blk_of[v] == r] for r in range(m)]

    Jw = None
    if args.jmatrix:
        import numpy as np
        M = np.load(args.jmatrix) if args.jmatrix.endswith(".npy") \
            else np.loadtxt(args.jmatrix)
        M = (np.asarray(M, float) + np.asarray(M, float).T) / 2.0
        Jw = M

    def edge_rec(u, v):
        rec = {"u": u, "v": v}
        if Jw is not None:
            rec["J"] = float(Jw[u, v])
        return rec

    # partition EVERY edge into intra vs inter (no range filter)
    intra_by_block = {r: [] for r in range(m)}
    inter_by_pair = {}
    n_intra = n_inter = 0
    for (u, v) in edges:
        ru, rv = blk_of[u], blk_of[v]
        if ru == rv:
            intra_by_block[ru].append(edge_rec(u, v))
            n_intra += 1
        else:
            key = (min(ru, rv), max(ru, rv))
            inter_by_pair.setdefault(key, []).append(edge_rec(u, v))
            n_inter += 1
    assert n_intra + n_inter == len(edges), "edge partition lost terms!"

    intra = [{"supersite": r, "sites": blocks[r],
              "internal_edges": intra_by_block[r]} for r in range(m)]
    inter = [{"supersite_a": ra, "supersite_b": rb,
              "chain_distance": rb - ra, "couplings": es}
             for (ra, rb), es in sorted(inter_by_pair.items())]

    max_d = max((it["chain_distance"] for it in inter), default=0)
    n_bonded_blocks = sum(1 for r in range(m) if intra_by_block[r])
    out = {
        "cluster": args.cluster or "edgefile",
        "block_size": q, "num_supersites": m,
        "num_sites": n, "num_edges_total": len(edges),
        "num_intra_edges": n_intra, "num_inter_edges": n_inter,
        "supersites_with_internal_edges": n_bonded_blocks,
        "supersite_bandwidth": ss_value(lab, edges),
        "max_chain_distance": max_d,
        "chain_order": blocks,
        "intra_supersite": intra,
        "inter_supersite": inter,
    }
    text = json.dumps(out, indent=1)
    if args.out:
        open(args.out, "w").write(text)
        from collections import Counter
        bydist = Counter()
        for it in inter:
            bydist[it["chain_distance"]] += len(it["couplings"])
        print(f"wrote {args.out}")
        print(f"  {m} supersites of {q}; supersite bandwidth "
              f"{out['supersite_bandwidth']}")
        print(f"  edges: {len(edges)} total = {n_intra} intra + {n_inter} inter "
              f"(all terms included)")
        print(f"  supersites with internal edges: {n_bonded_blocks}/{m}")
        print(f"  inter couplings by chain distance: "
              + ", ".join(f"d={d}: {bydist[d]}" for d in sorted(bydist)))
    else:
        print(text)


def cmd_ss_export(args):
    n, edges = get_graph(args)
    st = ss_state(args, n, edges)
    cur = st.read()
    if not cur["best_labeling"]:
        sys.exit("no supersite assignment in the state yet")
    lab = cur["best_labeling"]
    print("supersite bandwidth of this assignment:", ss_value(lab, edges))
    print("{" + ", ".join(f"{v}: {lab[v] - 1}" for v in range(n)) + "}")
    m = n // args.block
    blocks = [[v for v in range(n) if lab[v] == r + 1] for r in range(m)]
    print("blocks in chain order:", blocks)




# ======================================================================
# EQUIVARIANT-ANSATZ MODE (lattice-structure-exploiting exact bound)
# ======================================================================
# Restrict layouts to lattice-respecting families and certify the optimum
# *within the family* exactly (tiny search space -> CP-SAT closes it fast).
# Two ansatz families, both parameterized by a cell ordering C (a
# permutation of the m_c unit cells) and ONE shared basis permutation S:
#   cell-major  : pos(cell c, site s) = nb * C[c] + S[s]
#                 (each cell occupies a contiguous block, same internal
#                  pattern in every cell)
#   basis-major : pos(cell c, site s) = m_c * S[s] + C[c]
#                 (sublattice-striped: all copies of basis site s are
#                  contiguous, cells interleaved)
# The certified ansatz optimum is a rigorous upper bound for the true
# bandwidth; the gap to the campaign incumbent measures how strongly the
# true optimum breaks the lattice symmetry. The resulting labeling is
# recorded into the cluster's standard state file as a seed.
# Requires cluster_generator.py (and pynauty if the generator numbering
# differs from cluster_edges.py).

def load_generator_instance(args):
    import importlib.util as ilu
    spec = ilu.spec_from_file_location("cluster_generator", args.generator)
    gen = ilu.module_from_spec(spec)
    sys.modules["cluster_generator"] = gen
    spec.loader.exec_module(gen)
    lattice = gen.get_lattice(args.lattice, args.trillium_u)
    nb = lattice.n_basis
    if args.supercell or args.tilted:
        if args.tilted:
            key = gen.canonical_tilted_name(args.tilted)
            vectors = gen.PYROCHLORE_TILTED_SUPERCELLS[key]
        else:
            vectors = gen.parse_supercell(args.supercell)
        coords, edges_gen = lattice.make_supercell(vectors)
    else:
        if None in (args.Nx, args.Ny, args.Nz):
            sys.exit("need --Nx --Ny --Nz, or --supercell / --tilted")
        coords, edges_gen = lattice.make_diagonal(args.Nx, args.Ny, args.Nz)
    n = len(coords)
    edges_gen = sorted({(min(u, v), max(u, v)) for u, v in edges_gen})
    return n, edges_gen, nb


def iso_map_to_cluster(n, edges_gen, edges_tgt):
    """gen vertex -> target vertex; identity if edge sets already equal."""
    if edges_gen == edges_tgt:
        return list(range(n))
    try:
        import pynauty
    except ImportError:
        sys.exit("generator and cluster numberings differ; "
                 "pip install pynauty for the isomorphism mapping")

    def canon(edges):
        d = {v: [] for v in range(n)}
        for u, v in edges:
            d[u].append(v)
            d[v].append(u)
        return pynauty.canon_label(pynauty.Graph(n, adjacency_dict=d))

    lab_g, lab_t = canon(edges_gen), canon(edges_tgt)
    pos_in_g = {v: i for i, v in enumerate(lab_g)}
    mapping = [lab_t[pos_in_g[v]] for v in range(n)]
    mapped = sorted((min(mapping[u], mapping[v]), max(mapping[u], mapping[v]))
                    for u, v in edges_gen)
    if mapped != edges_tgt:
        sys.exit("graphs are not isomorphic — wrong lattice/shape for this "
                 "cluster?")
    return mapping


def eq_optimize(n, edges_gen, nb, ansatz, time_limit, workers):
    """Certified minimum bandwidth within one ansatz family.
    Returns (status_str, best_B or None, labeling 1-based or None)."""
    from ortools.sat.python import cp_model
    mc = n // nb
    model = cp_model.CpModel()
    C = [model.NewIntVar(0, mc - 1, f"C{c}") for c in range(mc)]
    S = [model.NewIntVar(0, nb - 1, f"S{s}") for s in range(nb)]
    model.AddAllDifferent(C)
    model.AddAllDifferent(S)
    B = model.NewIntVar(1, n - 1, "B")
    P = []
    for v in range(n):
        c, s = divmod(v, nb)
        p = model.NewIntVar(0, n - 1, f"P{v}")
        if ansatz == "cell":
            model.Add(p == nb * C[c] + S[s])
        else:                                   # basis-major
            model.Add(p == mc * S[s] + C[c])
        P.append(p)
    for u, v in edges_gen:
        d = model.NewIntVar(0, n - 1, f"d{u}_{v}")
        model.AddAbsEquality(d, P[u] - P[v])
        model.Add(d <= B)
    model.Add(C[0] <= (mc - 1) // 2)            # cell-order reversal
    model.Minimize(B)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        lab = [solver.Value(P[v]) + 1 for v in range(n)]
        tag = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"
        return tag, solver.Value(B), lab
    return "UNKNOWN", None, None


def cmd_eq_run(args):
    n_g, edges_gen, nb = load_generator_instance(args)
    n, edges_tgt = get_graph(args)
    if n_g != n or len(edges_gen) != len(edges_tgt):
        sys.exit(f"size mismatch: generator n={n_g}/|E|={len(edges_gen)} vs "
                 f"cluster n={n}/|E|={len(edges_tgt)}")
    mapping = iso_map_to_cluster(n, edges_gen, sorted(edges_tgt))
    st = State(args.state_dir, args.cluster or "edgefile", n, edges_tgt)
    cur = st.read()
    _, incumbent = State.window(cur)
    families = ["cell", "basis"] if args.ansatz == "both" else [args.ansatz]
    for fam in families:
        tag, Bopt, lab_gen = eq_optimize(n, edges_gen, nb, fam,
                                         args.time, args.workers)
        if Bopt is None:
            print(f"[eq:{fam}] no solution found within the time limit")
            continue
        cert = "CERTIFIED ansatz optimum" if tag == "OPTIMAL" \
            else "best found (ansatz not closed; raise --time)"
        print(f"[eq:{fam}-major] bandwidth {Bopt}  ({cert})")
        lab_tgt = [0] * n
        for v in range(n):
            lab_tgt[mapping[v]] = lab_gen[v]
        assert bandwidth_of(lab_tgt, edges_tgt) == Bopt, "mapping changed value"
        st.record_labeling(lab_tgt, f"equivariant-{fam}({tag.lower()})")
    cur = st.read()
    lb, ub = State.window(cur)
    print(f"[eq] cluster window now [{lb},{ub}]"
          + (f"  (incumbent before eq: {incumbent})" if incumbent else ""))
    print("[eq] gap between ansatz optimum and incumbent measures how "
          "strongly the true optimum breaks the lattice cell structure")




# ======================================================================
# ss-seed : translation-blocking heuristic for the supersite search
# ======================================================================
# For each candidate lattice translation t (axis-aligned AND tilted), pair
# every site r with its translate r+t (mod the cluster periods). When this is
# a perfect involution (every site paired uniquely, q=2), it defines a fixed
# blocking. We then contract each pair to a block, build the block graph, and
# run bandwidth minimization on it (heuristic + a short CP-SAT optimize) to get
# the inter-block bandwidth of that blocking. The best translation gives an
# early, structurally-motivated upper bound, recorded into the supersite state.
# Requires cluster_generator.py (for coordinates) and numpy.

def _periods_realspace(lat, dims, coords):
    """Real-space period vectors of the cluster along each periodic Bravais
    axis: L_a * a-th Bravais vector, for axes with pbc True."""
    import numpy as np
    bravais = np.array(lat.bravais, float)
    periods = []
    for ax in range(3):
        if lat.pbc[ax]:
            periods.append(dims[ax] * bravais[ax])
    return periods


def _canon_factory(periods, tol=1e-4):
    """Return a canonicalizer that reduces a 3D point into the cluster cell by
    expressing the periodic components in the (possibly oblique) period basis
    and taking fractional coordinates modulo 1. Non-periodic directions pass
    through unchanged."""
    import numpy as np
    per = [p for p in periods if float(np.linalg.norm(p)) > 1e-9]
    if not per:
        return lambda v: tuple(np.round(np.asarray(v, float) / tol).astype(np.int64))
    M = np.array(per).T                       # 3 x d, d = #periodic axes
    # least-squares fractional coordinates; works for d <= 3 oblique axes
    Mpinv = np.linalg.pinv(M)

    def canon(v):
        v = np.asarray(v, float)
        f = Mpinv @ v                          # fractional along period axes
        f = f - np.round(f)                    # into [-0.5, 0.5)
        red = v - M @ (Mpinv @ v - f)          # remove integer period multiples
        return tuple(np.round(red / tol).astype(np.int64))
    return canon


def _match_under_translation(coords, t, periods, tol=1e-4):
    """match[i] = j if site j sits at coords[i]+t modulo the (oblique) periods,
    else None. A valid q=2 blocking is a perfect involution with no fixed
    points."""
    import numpy as np
    P = np.array(coords, float)
    n = len(P)
    canon = _canon_factory(periods, tol)
    lut = {canon(P[i]): i for i in range(n)}
    if len(lut) != n:
        return [None] * n                      # degenerate reduction; reject
    return [lut.get(canon(P[i] + np.asarray(t, float))) for i in range(n)]


def _candidate_translations(lat, dims=None, extra=2):
    """Lattice-translation vectors to try, in real space. A blocking pairing
    must be a TRANSLATION SYMMETRY of the infinite lattice (an integer
    combination of Bravais vectors) AND a fixed-point-free involution on the
    finite cluster. On a periodic axis of length L, the natural q=2 involution
    is translation by L/2 cells; we therefore range each periodic coefficient
    up to L_a/2 + extra, and non-periodic / unknown axes over [-extra, extra].
    Covers axis-aligned (single nonzero coeff) and tilted (mixed) directions."""
    import numpy as np
    import itertools
    bravais = np.array(lat.bravais, float)
    axes = [a for a in range(3) if any(abs(x) > 1e-9 for x in bravais[a])]
    # per-axis coefficient ranges
    ranges = []
    for a in axes:
        if dims is not None and a < len(dims) and getattr(lat, "pbc", (1,1,1))[a]:
            hi = dims[a] // 2 + extra
        else:
            hi = extra
        ranges.append(range(-hi, hi + 1))
    cands = {}
    for coeffs in itertools.product(*ranges):
        if all(c == 0 for c in coeffs):
            continue
        first_nz = next(c for c in coeffs if c != 0)
        if first_nz < 0:
            continue                              # dedupe +/-
        v = np.zeros(3)
        for c, a in zip(coeffs, axes):
            v = v + c * bravais[a]
        key = tuple(np.round(v / 1e-4).astype(np.int64))
        cands.setdefault(key, v)
    return sorted(cands.values(), key=lambda v: float(np.linalg.norm(v)))


def _match_nn_along_direction(coords, t, periods, tol=1e-4):
    """Pair each site with its nearest neighbor in the +/- t direction, then
    extract a perfect matching from those candidate pairs.

    Unlike the translation-involution matcher, t need NOT be a lattice
    symmetry: for every site we look at the geometrically closest other site
    whose displacement (minimum-image under the periods) is most parallel to t
    and within a reasonable distance, propose that undirected pair, and then
    greedily select a set of disjoint pairs covering every site. Returns a
    match list (match[i]=j, involution) or None if no perfect matching of the
    proposed pairs exists."""
    import numpy as np
    P = np.array(coords, float)
    n = len(P)
    th = np.asarray(t, float)
    tn = th / (np.linalg.norm(th) + 1e-30)
    pervecs = [p for p in periods if float(np.linalg.norm(p)) > 1e-9]

    def min_image(d):
        """shortest displacement equivalent to d under the periodic lattice."""
        if not pervecs:
            return d
        best = d.copy()
        # search small integer combinations of period vectors
        import itertools
        rngs = [range(-1, 2)] * len(pervecs)
        for combo in itertools.product(*rngs):
            shift = sum((c * pv for c, pv in zip(combo, pervecs)), np.zeros(3))
            cand = d - shift
            if cand @ cand < best @ best:
                best = cand
        return best

    # candidate pair for each site: the other site whose min-image displacement
    # is closest in direction to +/- t, weighted by distance.
    cand_pairs = []
    for i in range(n):
        best_j, best_score = None, -1e30
        for j in range(n):
            if j == i:
                continue
            d = min_image(P[j] - P[i])
            dist = float(np.linalg.norm(d))
            if dist < 1e-9:
                continue
            align = abs((d / dist) @ tn)              # 1 = perfectly along t
            if align < 0.9:                           # must be ~parallel to t
                continue
            score = align - 0.05 * dist               # prefer aligned & near
            if score > best_score:
                best_score, best_j = score, j
        if best_j is not None:
            cand_pairs.append((i, best_j))

    # build an undirected candidate graph and find a perfect matching
    from collections import defaultdict
    adj = defaultdict(set)
    for i, j in cand_pairs:
        adj[i].add(j)
        adj[j].add(i)
    match = [None] * n

    def try_augment(u, visited):
        for w in adj[u]:
            if w in visited:
                continue
            visited.add(w)
            if match[w] is None or try_augment(match[w], visited):
                match[w] = u
                match[u] = w
                return True
        return False

    for u in range(n):
        if match[u] is None:
            if not try_augment(u, {u}):
                return None                            # no perfect matching
    if any(match[i] is None or match[match[i]] != i or match[i] == i
           for i in range(n)):
        return None
    return match


def _strip_candidates(steps, Nx, Ny, extra=2):
    """Candidate translations for a strip: integer combinations c0*step_trans +
    c1*step_long, covering nearest-neighbor steps and half-period shifts along
    each strip direction. steps = [transverse_step, longitudinal_step]."""
    import numpy as np
    import itertools
    st = np.array(steps, float)
    hi0 = Ny // 2 + extra
    hi1 = Nx // 2 + extra
    cands = {}
    for c0 in range(-hi0, hi0 + 1):
        for c1 in range(-hi1, hi1 + 1):
            if c0 == 0 and c1 == 0:
                continue
            first = c0 if c0 != 0 else c1
            if first < 0:
                continue
            v = c0 * st[0] + c1 * st[1]
            key = tuple(np.round(v / 1e-4).astype(np.int64))
            cands.setdefault(key, v)
    return sorted(cands.values(), key=lambda v: float(np.linalg.norm(v)))


def _block_graph(n, edges, match):
    """Contract the q=2 blocking given by the involution `match` into a block
    graph. Returns (m, block_edges, blocks) where blocks[r] = [site,site]."""
    blk = [-1] * n
    blocks = []
    for i in range(n):
        if blk[i] == -1:
            j = match[i]
            r = len(blocks)
            blk[i] = blk[j] = r
            blocks.append(sorted((i, j)))
    m = len(blocks)
    be = set()
    for (u, v) in edges:
        a, b = blk[u], blk[v]
        if a != b:
            be.add((min(a, b), max(a, b)))
    return m, sorted(be), blocks


def cmd_ss_seed(args):
    import numpy as np
    import importlib.util as ilu
    if args.block != 2:
        sys.exit("ss-seed pairs sites under a translation, so it is q=2 only")
    spec = ilu.spec_from_file_location("cluster_generator", args.generator)
    gen = ilu.module_from_spec(spec)
    sys.modules["cluster_generator"] = gen
    spec.loader.exec_module(gen)
    lat = gen.get_lattice(args.lattice, args.trillium_u)

    strip_variants = set()
    for setname in ("KAGOME_STRIP_VARIANTS", "TRIANGULAR_STRIP_VARIANTS"):
        strip_variants |= set(getattr(gen, setname, set()))

    if args.lattice in strip_variants:
        # Cylinders/tori built by the dedicated strip builders. This is the
        # ONLY path that yields open boundaries correctly; get_lattice()/
        # make_diagonal() would return a full torus for these names.
        if None in (args.Nx, args.Ny):
            sys.exit(f"{args.lattice} needs --Nx and --Ny")
        if args.lattice in getattr(gen, "KAGOME_STRIP_VARIANTS", set()):
            kind, per_x = gen.kagome_variant_kind_and_periodicity(args.lattice)
            coords, edges_gen, px, py = gen.build_kagome_strip_edges(
                kind, args.Nx, args.Ny, per_x, a=1.0)
        else:
            kind, per_x = gen.triangular_variant_kind_and_periodicity(args.lattice)
            coords, edges_gen, px, py = gen.build_triangular_strip_edges(
                kind, args.Nx, args.Ny, per_x, a=1.0)
        # periods: the transverse direction always wraps; the longitudinal
        # (x) direction wraps only for the torus variants (per_x True).
        periods = [np.array(py, float)]
        if per_x:
            periods.append(np.array(px, float))
        dims = None
    elif args.supercell or args.tilted:
        if args.tilted:
            vectors = gen.PYROCHLORE_TILTED_SUPERCELLS[
                gen.canonical_tilted_name(args.tilted)]
        else:
            vectors = gen.parse_supercell(args.supercell)
        coords, edges_gen = lat.make_supercell(vectors)
        bravais = np.array(lat.bravais, float)
        periods = [np.array(v, float) @ bravais for v in vectors]
        dims = None
    else:
        if None in (args.Nx, args.Ny, args.Nz):
            sys.exit("need --Nx --Ny --Nz, or --supercell / --tilted")
        coords, edges_gen = lat.make_diagonal(args.Nx, args.Ny, args.Nz)
        dims = (args.Nx, args.Ny, args.Nz)
        periods = _periods_realspace(lat, dims, coords)
    n = len(coords)
    edges_gen = sorted({(min(u, v), max(u, v)) for u, v in edges_gen})

    # map generator numbering -> cluster_edges numbering (identity or iso)
    n_t, edges_tgt = get_graph(args)
    edges_tgt = sorted(edges_tgt)
    if n_t != n or len(edges_tgt) != len(edges_gen):
        sys.exit(f"size mismatch: generator {n}/{len(edges_gen)} vs "
                 f"cluster {n_t}/{len(edges_tgt)}")
    mapping = iso_map_to_cluster(n, edges_gen, edges_tgt)  # gen v -> tgt v

    if n % 2:
        sys.exit("translation blocking needs q=2, but n is odd")

    st = ss_state(args, n, edges_tgt)

    if args.lattice in strip_variants:
        # strip translations: integer multiples of the two period directions,
        # plus the nearest-neighbor bond directions inferred from coords.
        gens = []
        for base in periods:
            gens.append(np.array(base, float))
        # also include the primitive longitudinal/transverse steps (period / N)
        if dims is None:
            steps = []
            steps.append(np.array(py, float) / max(1, args.Ny))
            steps.append(np.array(px, float) / max(1, args.Nx))
            cands = _strip_candidates(steps, args.Nx, args.Ny)
        else:
            cands = _candidate_translations(lat, dims)
    else:
        cands = _candidate_translations(lat, dims)
    print(f"[ss-seed] {len(cands)} candidate translations (axis-aligned + "
          f"tilted); testing which give a valid q=2 pairing...")
    best = None
    best_t = None
    seen_blockings = set()
    n_valid = 0
    n_invalid = 0
    results = []                                       # (bw, t, |t|)
    for t in cands:
        if args.mode == "nn":
            match = _match_nn_along_direction(coords, t, periods)
            if match is None:
                n_invalid += 1
                continue                               # no perfect NN matching
        else:
            match = _match_under_translation(coords, t, periods)
            if any(j is None for j in match) \
               or any(match[match[i]] != i for i in range(n)) \
               or any(match[i] == i for i in range(n)):
                n_invalid += 1
                continue                               # not a valid involution
        m, be_gen, blocks_gen = _block_graph(n, edges_gen, match)
        sig = tuple(sorted(tuple(b) for b in blocks_gen))
        if sig in seen_blockings:
            continue                                   # same blocking as a
        seen_blockings.add(sig)                        # shorter translation
        n_valid += 1
        bw_h, blab = run_heuristic(m, be_gen, args.heur_time,
                                   args.procs, seed=args.seed)
        sol, plb, opt, _ = cpsat_optimize(
            m, be_gen, max(1, combinatorial_lb(m, be_gen)), bw_h,
            args.opt_time, args.workers, hint=blab, symmetry="reversal")
        block_order = sol if sol else blab
        bw = bandwidth_of(block_order, be_gen)
        tnorm = float(np.linalg.norm(t))
        cstr = f"({t[0]:+.2f},{t[1]:+.2f},{t[2]:+.2f})"
        print(f"[ss-seed]   valid pairing: t={cstr} |t|={tnorm:.3f} "
              f"-> {m} blocks, inter-block bandwidth {bw}")
        results.append((bw, cstr, tnorm))
        if best is None or bw < best[0]:
            best = (bw, blocks_gen, block_order)
            best_t = cstr
    reason = ("no perfect nearest-neighbor matching" if args.mode == "nn"
              else "not a clean involution")
    print(f"[ss-seed] tested {len(cands)} directions: {n_valid} distinct valid "
          f"blockings, {n_invalid} invalid ({reason})")
    if best is None:
        print("[ss-seed] RESULT: no translation gives a valid q=2 pairing on "
              "this cluster; nothing seeded (the campaign will start from SA).")
        return
    results.sort()
    ranked = ", ".join(f"{c}:bw{b}" for b, c, _ in results[:5])
    print(f"[ss-seed] best translations by bandwidth: {ranked}"
          + (" ..." if len(results) > 5 else ""))
    bw, blocks_gen, block_order = best
    print(f"[ss-seed] RESULT: best translation blocking is t={best_t}, "
          f"giving supersite bandwidth {bw} (seeded as the upper bound).")
    # build the supersite labeling on the TARGET numbering:
    # block_order[r] = chain position (1..m) of block r; each site in block r
    # inherits that position.
    lab = [0] * n
    for r, pair in enumerate(blocks_gen):
        pos = block_order[r]
        for v in pair:
            lab[mapping[v]] = pos
    assert sorted(lab) == sorted(list(range(1, n // 2 + 1)) * 2)
    print(f"[ss-seed] best translation blocking: supersite bandwidth {bw}")
    mi, pb = ss_constraints(args, edges_tgt)
    vi = ss_viol(lab, edges_tgt, 2, mi, pb)
    if vi == 0:
        st.record_labeling(lab, "ss-seed(translation)")
    else:
        print(f"[ss-seed] blocking violates the hidden-bond constraints "
              f"(violation {vi}, e.g. pairs not along bonds); NOT seeded")
    print_status(st.read(), edges_tgt)


# ======================================================================
# CUTWIDTH MODE: minimize cut_max = max over the n-1 chain cuts of the number
# ======================================================================
# of edges crossing that cut (the MPO bond-dimension proxy). Plain permutation
# (lab[v] = 1-based position). Decision sat(G,c) is monotone in c, so the same
# ladder/state/certificate machinery applies. State name: <cluster>__cw.
#
# NOTE: cutwidth lower bounds are weak (only ceil(maxdeg/2) here), so CP-SAT
# closes the window only for SMALL clusters (~n <= 25-30). Large clusters get
# the SA upper bound with an open window -- same as bandwidth on hard tori.

def cutwidth_of(lab, edges):
    """cut_max for a 1-based vertex->position labeling: largest number of edges
    crossing any of the n-1 chain cuts (via a +1/-1 interval sweep)."""
    n = len(lab)
    inc = [0] * (n + 2)
    for u, v in edges:
        a, b = (lab[u], lab[v]) if lab[u] < lab[v] else (lab[v], lab[u])
        inc[a] += 1
        inc[b] -= 1                       # edge crosses cuts a..b-1
    cur = mx = 0
    for p in range(1, n):                 # cut positions 1..n-1
        cur += inc[p]
        if cur > mx:
            mx = cur
    return mx


def cutwidth_math_lb(n, edges, q=1):
    """Cheap valid lower bound on the cutwidth: the best of three.

    - ceil(maxdeg/2): the max-degree vertex splits its edges left/right of its
      own position, so the busier side carries >= ceil(deg/2). For blocked
      layouts (q > 1) up to q-1 of its edges can hide inside its own block, so
      the term weakens to ceil((maxdeg-(q-1))/2).
    - degree-sum: the cut after the first k positions is crossed by at least
      sum(deg of those k vertices) - 2*(edges among them), which is at least
      (sum of the k smallest degrees) - 2*min(k(k-1)/2, |E|); maximize over k.
    - spectral (Fiedler): for any vertex subset S, e(S, ~S) >= lambda2*|S||~S|/n,
      so every ordering's middle cut gives cutwidth >= lambda2*floor(n/2)*
      ceil(n/2)/n. One Laplacian eigenvalue; on dense graphs this is often far
      stronger than what the CP-SAT UNSAT ladder can prove in hours.

    With q > 1 the bound applies to the BLOCKED cutwidth (supersites of q
    vertices): block-cuts split the vertex set at multiples of q only, so the
    degree-sum and spectral terms are evaluated at k = q, 2q, ..."""
    if not edges:
        return 0
    adj = adjacency(n, edges)
    degs = sorted(len(a) for a in adj)
    lb = max(1, ceil_div(degs[-1] - (q - 1), 2)) if degs[-1] > q - 1 else 1
    m = len(edges)
    acc = 0
    for k in range(1, n + 1):
        acc += degs[k - 1]
        if k % q == 0:                        # block-cuts sit at multiples of q
            lb = max(lb, acc - 2 * min(k * (k - 1) // 2, m))
    try:
        import numpy as np
        L = np.zeros((n, n))
        for u, v in edges:
            L[u, u] += 1.0
            L[v, v] += 1.0
            L[u, v] -= 1.0
            L[v, u] -= 1.0
        lam2 = float(np.linalg.eigvalsh(L)[1])
        if lam2 > 1e-9:
            k = ((n // q) // 2) * q           # block-cut closest to the middle
            if k > 0:
                # small slack guards the ceil against float noise in lambda2
                lb = max(lb, math.ceil(lam2 * k * (n - k) / n - 1e-6))
    except ImportError:                       # numpy absent: skip spectral term
        pass
    return lb


class CutwidthState(State):
    """State whose objective is cutwidth (not bandwidth). Plain permutation."""
    objective = "cutwidth"
    objective_abbrev = "c*"

    def value_of(self, lab):
        return cutwidth_of(lab, self.edges)

    def range_of(self, lab):
        return total_range(lab, self.edges)   # tie-break: total interaction range


def cw_state(args, n, edges):
    """State for cutwidth runs. --block q > 1 minimizes the BLOCKED cutwidth
    (supersites of q vertices) and gets its own multiplicity-q state file, plus
    hidden-bond tags -- constrained certificates are conditional on the
    constraints and must never mix with the plain (or differently-constrained)
    window. q = 1 keeps the original plain name for backward compatibility."""
    q = getattr(args, "block", 1) or 1
    name = f"{args.cluster or 'edgefile'}__cw"
    if q > 1:
        name += str(q)
        if getattr(args, "min_intra_edges", 0):
            name += f"_ie{args.min_intra_edges}"
        if getattr(args, "intra_per_block", False):
            name += "_ipb"
    return CutwidthState(args.state_dir, name, n, edges, mult=q)


def _cw_sa_chain(payload):
    """One cutwidth annealing chain (picklable). Cost is lexicographic
    (cut_max, total_range) minimized directly; moves = pair swap / segment
    reversal / relocation. Returns (cut_max, total_range, 1-based labeling)."""
    n, edges, seed, t_budget, init = payload
    rng = random.Random(seed)
    if init is not None:                       # init is 1-based vertex->position
        perm = [0] * n
        for v in range(n):
            perm[init[v] - 1] = v              # perm[pos] = vertex (0-based)
    else:
        perm = list(range(n))
        rng.shuffle(perm)

    def cost(pm):
        pos = [0] * n
        for i, v in enumerate(pm):
            pos[v] = i
        inc = [0] * (n + 1)
        s = 0
        for u, v in edges:
            a, b = pos[u], pos[v]
            if a > b:
                a, b = b, a
            inc[a] += 1
            inc[b] -= 1
            s += b - a
        cur = mx = 0
        for p in range(n - 1):
            cur += inc[p]
            if cur > mx:
                mx = cur
        return mx, s

    cmax, srange = cost(perm)
    best, bm, bs = perm[:], cmax, srange
    T = 0.05
    t_end = time.time() + t_budget
    while time.time() < t_end and bm > 0:
        for _ in range(200):
            new = perm[:]
            r = rng.random()
            if r < 0.4:                                    # pair swap
                a, b = rng.randrange(n), rng.randrange(n)
                new[a], new[b] = new[b], new[a]
            elif r < 0.7:                                  # segment reversal
                a, b = sorted((rng.randrange(n), rng.randrange(n)))
                new[a:b + 1] = new[a:b + 1][::-1]
            else:                                          # relocation
                x = new.pop(rng.randrange(n))
                new.insert(rng.randrange(n), x)
            m2, s2 = cost(new)
            if (m2, s2) <= (cmax, srange):
                accept = True
            else:
                if m2 != cmax:
                    delta = (m2 - cmax) / max(cmax, 1e-12)
                else:
                    delta = 0.1 * (s2 - srange) / max(srange, 1e-12)
                accept = delta <= 0 or (delta / T < 700
                                        and rng.random() < math.exp(-delta / T))
            if accept:
                perm, cmax, srange = new, m2, s2
                if (cmax, srange) < (bm, bs):
                    best, bm, bs = perm[:], cmax, srange
        T = max(1e-4, T * 0.97)
    lab = [0] * n
    for i, v in enumerate(best):
        lab[v] = i + 1
    return bm, bs, lab


def sa_cutwidth(n, edges, init, seed, t_budget, procs=1, stall=None):
    """Parallel multi-start cutwidth SA. Returns (cut_max, 1-based labeling)."""
    best_lab = init[:] if init else list(range(1, n + 1))
    best_cw = cutwidth_of(best_lab, edges)
    if procs <= 1:
        cmax, _s, lab = _cw_sa_chain((n, edges, seed, t_budget, init))
        return (cmax, lab) if cmax < best_cw else (best_cw, best_lab)
    if stall is None or stall <= 0:
        stall = max(60.0, t_budget / 10)
    t_end = time.time() + t_budget
    last_improve = time.time()
    batch = max(15.0, min(stall / 2, t_budget / 4))
    r = 0
    with mp.Pool(procs) as pool:
        while time.time() < t_end and best_cw > 0:
            if time.time() - last_improve > stall:
                print(f"[cw-heuristic] no improvement for {stall:.0f}s "
                      f"({stall / 3600:.2f}h); stopping early at UB {best_cw}")
                break
            dur = min(batch, t_end - time.time())
            if dur <= 1:
                break
            jobs = [(n, edges, seed + r * procs + p, dur,
                     best_lab if p % 2 == 0 else None)
                    for p in range(procs)]
            r += 1
            for cmax, _s, lab in pool.imap_unordered(_cw_sa_chain, jobs):
                if cmax < best_cw:
                    best_cw, best_lab = cmax, lab
                    last_improve = time.time()
                    print(f"[cw-heuristic] new upper bound: {best_cw}")
    return best_cw, best_lab


def _sscw_sa_chain(payload):
    """One blocked-cutwidth annealing chain (picklable). lab[v] = block index
    1..m, exactly q per block; moves are cross-block vertex swaps. Cost is
    lexicographic (hidden-bond violation, cut_max over block-cuts, total block
    range) with violation-first annealing (reuses _ss_accept_delta). Returns
    (viol, cut_max, labeling)."""
    n, edges, q, seed, t_budget, init, min_intra, per_block = payload
    rng = random.Random(seed)
    lab = init[:] if init else [(i // q) + 1 for i in range(n)]
    if not init:
        rng.shuffle(lab)
    m = n // q

    def cost(l):
        inc = [0] * (m + 2)
        s = 0
        for u, v in edges:
            a, b = (l[u], l[v]) if l[u] < l[v] else (l[v], l[u])
            inc[a] += 1
            inc[b] -= 1
            s += b - a
        cur = mx = 0
        for p in range(1, m):
            cur += inc[p]
            if cur > mx:
                mx = cur
        viol = 0
        if min_intra or per_block:
            covered = set()
            n_in = 0
            for u, v in edges:
                if l[u] == l[v]:
                    n_in += 1
                    covered.add(l[u])
            viol = max(0, min_intra - n_in)
            if per_block:
                viol += m - len(covered)
        return viol, mx, s

    vi, mx, c = cost(lab)
    best, bv, bm, bc = lab[:], vi, mx, c
    t_end = time.time() + t_budget
    T = max(2.0, bm / 4)
    while time.time() < t_end and (bm > 0 or bv > 0):
        for _ in range(2000):
            a, b = rng.randrange(n), rng.randrange(n)
            if a == b or lab[a] == lab[b]:
                continue                       # swap across blocks only
            lab[a], lab[b] = lab[b], lab[a]
            v2, m2, c2 = cost(lab)
            if (v2, m2, c2) <= (vi, mx, c):
                accept = True
            else:
                delta = _ss_accept_delta(v2, m2, c2, vi, mx, c)
                accept = delta <= 0 or (delta / T < 700
                                        and rng.random() < math.exp(-delta / T))
            if accept:
                vi, mx, c = v2, m2, c2
                if (vi, mx, c) < (bv, bm, bc):
                    best, bv, bm, bc = lab[:], vi, mx, c
            else:
                lab[a], lab[b] = lab[b], lab[a]
        T = max(0.05, T * 0.95)
    return bv, bm, best


def sa_sscw(n, edges, q, init, seed, t_budget, procs=1, target=0, stall=None,
            min_intra=0, per_block=False):
    """Parallel multi-start blocked-cutwidth SA. Returns (viol, cut, labeling);
    viol > 0 means the hidden-bond constraints could not be satisfied and the
    labeling must NOT be recorded as an upper bound for the constrained run."""
    best_lab = init[:] if init else [(i // q) + 1 for i in range(n)]
    best_cw = cutwidth_of(best_lab, edges)
    best_vi = ss_viol(best_lab, edges, q, min_intra, per_block)
    if best_vi == 0 and best_cw <= target:
        return 0, best_cw, best_lab
    if procs <= 1:
        return _sscw_sa_chain((n, edges, q, seed, t_budget, init,
                               min_intra, per_block))
    if stall is None or stall <= 0:
        stall = max(60.0, t_budget / 10)
    t_end = time.time() + t_budget
    last_improve = time.time()
    batch = max(15.0, min(stall / 2, t_budget / 4))
    r = 0
    with mp.Pool(procs) as pool:
        while time.time() < t_end and (best_cw > target or best_vi > 0):
            if time.time() - last_improve > stall:
                print(f"[cw-heuristic] no improvement for {stall:.0f}s "
                      f"({stall / 3600:.2f}h); stopping early at UB {best_cw}")
                break
            dur = min(batch, t_end - time.time())
            if dur <= 1:
                break
            jobs = [(n, edges, q, seed + r * procs + p, dur,
                     best_lab if p % 2 == 0 else None, min_intra, per_block)
                    for p in range(procs)]
            r += 1
            for vi, cw, lab in pool.imap_unordered(_sscw_sa_chain, jobs):
                if (vi, cw) < (best_vi, best_cw):
                    best_vi, best_cw, best_lab = vi, cw, lab
                    last_improve = time.time()
                    print(f"[cw-heuristic] new blocked upper bound: {best_cw}"
                          + (f" (constraint violation {best_vi})"
                             if best_vi else ""))
    return best_vi, best_cw, best_lab


def _cutwidth_cpsat_model(model, n, edges, c, symmetry="reversal",
                          fix_label1=None):
    """Build the shared cutwidth CP-SAT model into `model`: pos in 0..n-1
    AllDifferent; left[v,b] <=> pos[v] <= b; cross[e,b] = left[u,b] XOR
    left[v,b]; at most c crossings per cut; plus symmetry breaking. Returns
    (pos vars, note). Used by both the decision and the range polish."""
    pos = [model.NewIntVar(0, n - 1, f"p{v}") for v in range(n)]
    model.AddAllDifferent(pos)
    left = {}
    for v in range(n):
        for b in range(n - 1):
            lv = model.NewBoolVar(f"l{v}_{b}")
            model.Add(pos[v] <= b).OnlyEnforceIf(lv)
            model.Add(pos[v] >= b + 1).OnlyEnforceIf(lv.Not())
            left[v, b] = lv
    for b in range(n - 1):
        crs = []
        for (u, v) in edges:
            x = model.NewBoolVar(f"x{u}_{v}_{b}")
            a, d = left[u, b], left[v, b]
            model.Add(x <= a + d)
            model.Add(x >= a - d)
            model.Add(x >= d - a)
            model.Add(x <= 2 - a - d)
            crs.append(x)
        model.Add(sum(crs) <= c)
    # symmetry breaking (cutwidth is reversal-invariant)
    reps = None
    if symmetry == "orbit" and fix_label1 is None:
        res = orbit_representatives(n, edges)
        if res is not None:
            reps = res[0]
    if fix_label1 is not None:
        model.Add(pos[fix_label1] == 0)
        note = f"vertex {fix_label1} fixed to position 0"
    elif reps and len(reps) == 1:
        model.Add(pos[reps[0]] == 0)
        note = f"vertex-transitive: vertex {reps[0]} at position 0"
    elif reps:
        bools = []
        for rr in reps:
            bb = model.NewBoolVar(f"rep{rr}")
            model.Add(pos[rr] == 0).OnlyEnforceIf(bb)
            bools.append(bb)
        model.AddBoolOr(bools)
        note = f"position 0 is one of {len(reps)} orbit reps"
    else:
        adj = adjacency(n, edges)
        w = max(range(n), key=lambda x: len(adj[x]))
        model.Add(pos[w] <= (n - 1) // 2)
        note = f"reversal: vertex {w} pinned to positions 0..{(n - 1) // 2}"
    return pos, note


def cpsat_decide_cutwidth(n, edges, c, time_limit, workers, hint=None,
                          symmetry="reversal", fix_label1=None, log=False):
    """CP-SAT decision sat(G, cutwidth <= c). Returns (verdict, 1-based
    labeling_or_None, note)."""
    from ortools.sat.python import cp_model
    model = cp_model.CpModel()
    pos, note = _cutwidth_cpsat_model(model, n, edges, c, symmetry, fix_label1)
    if hint is not None:
        for v in range(n):
            model.AddHint(pos[v], hint[v] - 1)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    if log:
        solver.parameters.log_search_progress = True
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return "SAT", [solver.Value(pos[v]) + 1 for v in range(n)], note
    if status == cp_model.INFEASIBLE:
        return "UNSAT", None, note
    return "UNKNOWN", None, note


def cpsat_polish_cutwidth(n, edges, c, time_limit, workers, hint=None,
                          symmetry="reversal", fix_label1=None, log=False,
                          on_improve=None):
    """Among layouts with cutwidth <= c, minimize the TOTAL interaction range
    sum_e |pos_u - pos_v| via CP-SAT. If it reaches OPTIMAL the result is the
    provably minimum-range layout at that bond dimension. Returns
    (status_str, 1-based labeling_or_None, total_range, proven_lb, note).

    `on_improve(lab, tr)`, if given, is called for EVERY improving incumbent the
    solver finds during the search (not just the final one). This makes the
    polish resumable: each better layout is persisted immediately, so an
    interrupted run keeps its progress and a re-run continues from it."""
    from ortools.sat.python import cp_model
    model = cp_model.CpModel()
    pos, note = _cutwidth_cpsat_model(model, n, edges, c, symmetry, fix_label1)
    dvars = []
    for u, v in edges:
        d = model.NewIntVar(0, n - 1, f"d{u}_{v}")
        model.AddAbsEquality(d, pos[u] - pos[v])
        dvars.append(d)
    model.Minimize(sum(dvars))                         # total interaction range
    if hint is not None:
        for v in range(n):
            model.AddHint(pos[v], hint[v] - 1)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    if log:
        solver.parameters.log_search_progress = True

    cb = None
    if on_improve is not None:
        class _Recorder(cp_model.CpSolverSolutionCallback):
            def on_solution_callback(self):
                lab = [int(self.Value(pos[v])) + 1 for v in range(n)]
                tr = sum(abs(lab[u] - lab[v]) for u, v in edges)
                on_improve(lab, tr)
        cb = _Recorder()

    status = solver.Solve(model, cb) if cb else solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return "NONE", None, None, None, note
    lab = [solver.Value(pos[v]) + 1 for v in range(n)]
    tr = sum(abs(lab[u] - lab[v]) for u, v in edges)
    tag = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"
    return tag, lab, tr, solver.BestObjectiveBound(), note


def _sscw_cpsat_model(model, n, edges, q, c, symmetry="reversal",
                      fix_label1=None, min_intra=0, intra_per_block=False):
    """Blocked-cutwidth CP-SAT core: R[v] in 0..m-1 with exactly q vertices per
    block (cpsat_decide_ss's cardinality channeling); left/cross channeling
    caps every block-cut at c crossing edges (_cutwidth_cpsat_model's
    structure); optional hidden-bond constraints -- an intra-block edge crosses
    no cut at all, so hiding bonds directly relieves every cut it would have
    spanned. Returns (R vars, note)."""
    m = ss_m(n, q)
    R = [model.NewIntVar(0, m - 1, f"r{v}") for v in range(n)]
    B = {}
    for v in range(n):
        for r in range(m):
            b = model.NewBoolVar(f"b{v}_{r}")
            model.Add(R[v] == r).OnlyEnforceIf(b)
            model.Add(R[v] != r).OnlyEnforceIf(b.Not())
            B[v, r] = b
    for r in range(m):
        model.Add(sum(B[v, r] for v in range(n)) == q)
    left = {}
    for v in range(n):
        for b in range(m - 1):
            lv = model.NewBoolVar(f"l{v}_{b}")
            model.Add(R[v] <= b).OnlyEnforceIf(lv)
            model.Add(R[v] >= b + 1).OnlyEnforceIf(lv.Not())
            left[v, b] = lv
    for b in range(m - 1):
        crs = []
        for (u, v) in edges:
            x = model.NewBoolVar(f"x{u}_{v}_{b}")
            a, d = left[u, b], left[v, b]
            model.Add(x <= a + d)
            model.Add(x >= a - d)
            model.Add(x >= d - a)
            model.Add(x <= 2 - a - d)
            crs.append(x)
        model.Add(sum(crs) <= c)
    cnotes = []
    if min_intra:
        bs = []
        for u, v in edges:
            d = model.NewIntVar(0, m - 1, f"d{u}_{v}")
            model.AddAbsEquality(d, R[u] - R[v])
            b = model.NewBoolVar(f"in{u}_{v}")
            model.Add(d == 0).OnlyEnforceIf(b)
            model.Add(d != 0).OnlyEnforceIf(b.Not())
            bs.append(b)
        model.Add(sum(bs) >= min_intra)
        cnotes.append(f">={min_intra} intra-block edges")
    if intra_per_block:
        for r in range(m):
            lits = []
            for u, v in edges:
                e = model.NewBoolVar(f"e{u}_{v}_{r}")
                model.AddImplication(e, B[u, r])
                model.AddImplication(e, B[v, r])
                lits.append(e)
            model.AddBoolOr(lits)
        cnotes.append("every block has an internal edge")
    # symmetry breaking (blocked cutwidth is block-order-reversal invariant)
    reps = None
    if symmetry == "orbit" and fix_label1 is None:
        res = orbit_representatives(n, edges)
        if res is not None:
            reps = res[0]
    if fix_label1 is not None:
        model.Add(R[fix_label1] == 0)
        note = f"vertex {fix_label1} fixed to block 0"
    elif reps and len(reps) == 1:
        model.Add(R[reps[0]] == 0)
        note = f"vertex-transitive: vertex {reps[0]} fixed to block 0"
    elif reps:
        bools = []
        for rr in reps:
            bb = model.NewBoolVar(f"rep{rr}")
            model.Add(R[rr] == 0).OnlyEnforceIf(bb)
            bools.append(bb)
        model.AddBoolOr(bools)
        note = f"block 0 contains one of {len(reps)} orbit reps"
    else:
        adj = adjacency(n, edges)
        w = max(range(n), key=lambda x: len(adj[x]))
        model.Add(R[w] <= (m - 1) // 2)
        note = f"reversal: vertex {w} pinned to blocks 0..{(m - 1) // 2}"
    if cnotes:
        note += "; " + "; ".join(cnotes)
    return R, note


def cpsat_decide_sscw(n, edges, q, c, time_limit, workers, hint=None,
                      symmetry="reversal", fix_label1=None, log=False,
                      min_intra=0, intra_per_block=False):
    """CP-SAT decision sat(G, blocked cutwidth <= c) for supersites of q.
    Returns (verdict, 1-based block labeling_or_None, note)."""
    from ortools.sat.python import cp_model
    model = cp_model.CpModel()
    R, note = _sscw_cpsat_model(model, n, edges, q, c, symmetry, fix_label1,
                                min_intra, intra_per_block)
    if hint is not None:
        for v in range(n):
            model.AddHint(R[v], hint[v] - 1)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    if log:
        solver.parameters.log_search_progress = True
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return "SAT", [solver.Value(R[v]) + 1 for v in range(n)], note
    if status == cp_model.INFEASIBLE:
        return "UNSAT", None, note
    return "UNKNOWN", None, note


def cpsat_polish_sscw(n, edges, q, c, time_limit, workers, hint=None,
                      symmetry="reversal", fix_label1=None, log=False,
                      min_intra=0, intra_per_block=False, on_improve=None):
    """Among blockings with blocked cutwidth <= c (and the hidden-bond
    constraints satisfied), minimize the total block-interaction range. Mirrors
    cpsat_polish_cutwidth, including the resumable on_improve callback."""
    from ortools.sat.python import cp_model
    model = cp_model.CpModel()
    R, note = _sscw_cpsat_model(model, n, edges, q, c, symmetry, fix_label1,
                                min_intra, intra_per_block)
    m = ss_m(n, q)
    dvars = []
    for u, v in edges:
        d = model.NewIntVar(0, m - 1, f"pd{u}_{v}")
        model.AddAbsEquality(d, R[u] - R[v])
        dvars.append(d)
    model.Minimize(sum(dvars))
    if hint is not None:
        for v in range(n):
            model.AddHint(R[v], hint[v] - 1)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    if log:
        solver.parameters.log_search_progress = True
    cb = None
    if on_improve is not None:
        class _Recorder(cp_model.CpSolverSolutionCallback):
            def on_solution_callback(self):
                lab = [int(self.Value(R[v])) + 1 for v in range(n)]
                on_improve(lab, sum(abs(lab[u] - lab[v]) for u, v in edges))
        cb = _Recorder()
    status = solver.Solve(model, cb) if cb else solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return "NONE", None, None, None, note
    lab = [solver.Value(R[v]) + 1 for v in range(n)]
    tr = sum(abs(lab[u] - lab[v]) for u, v in edges)
    tag = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"
    return tag, lab, tr, solver.BestObjectiveBound(), note


def build_cnf_sscw(n, edges, q, c, fix_label1=None, symmetry="reversal",
                   min_intra=0, intra_per_block=False):
    """CNF for 'blocked cutwidth <= c'. One-hot x[v,r] = var(v,r) = v*m + r
    (r = 1..m, build_cnf_ss's convention, so ss_decode reads the model);
    exactly-one block per vertex, exactly-q vertices per block; left[v,b] via
    reified OR over r <= b; cross[e,b] XOR; at-most-c crossings per block-cut
    (Sinz); optional hidden-bond constraints matching cpsat_decide_sscw so
    cw-verify cross-checks the SAME decision."""
    m = ss_m(n, q)

    def var(v, r):
        return v * m + r

    clauses = []
    next_aux = n * m + 1

    def at_most(lits, bound):
        nonlocal next_aux
        next_aux = _sinz_at_most(clauses, next_aux, lits, bound)

    def exactly(lits, cc):
        at_most(lits, cc)
        at_most([-x for x in lits], len(lits) - cc)   # at least cc

    for v in range(n):
        lits = [var(v, r) for r in range(1, m + 1)]
        clauses.append(lits[:])                       # at least one block
        at_most(lits, 1)
    for r in range(1, m + 1):
        exactly([var(v, r) for v in range(n)], q)

    left = {}
    for v in range(n):
        for b in range(1, m):
            lvar = next_aux
            next_aux += 1
            left[v, b] = lvar
            ors = [var(v, r) for r in range(1, b + 1)]
            clauses.append([-lvar] + ors)             # L -> (block <= b)
            for x in ors:
                clauses.append([-x, lvar])            # (v in r<=b) -> L
    for b in range(1, m):
        cross_lits = []
        for (u, v) in edges:
            xb = next_aux
            next_aux += 1
            lu, lv = left[u, b], left[v, b]
            clauses.append([-xb, lu, lv])
            clauses.append([-xb, -lu, -lv])
            clauses.append([xb, -lu, lv])
            clauses.append([xb, lu, -lv])
            cross_lits.append(xb)
        next_aux = _sinz_at_most(clauses, next_aux, cross_lits, c)

    # hidden-bond constraints (mirror build_cnf_ss)
    cnotes = []
    if min_intra or intra_per_block:
        e_of = {}
        for (u, v) in edges:
            for r in range(1, m + 1):
                e = next_aux
                next_aux += 1
                clauses.append([-e, var(u, r)])
                clauses.append([-e, var(v, r)])
                e_of[u, v, r] = e
        if intra_per_block:
            for r in range(1, m + 1):
                clauses.append([e_of[u, v, r] for (u, v) in edges])
            cnotes.append("every block has an internal edge")
        if min_intra:
            ts = []
            for (u, v) in edges:
                t = next_aux
                next_aux += 1
                clauses.append([-t] + [e_of[u, v, r] for r in range(1, m + 1)])
                ts.append(t)
            at_most([-t for t in ts], len(ts) - min_intra)
            cnotes.append(f">={min_intra} intra-block edges")

    reps = None
    if symmetry == "orbit" and fix_label1 is None:
        res = orbit_representatives(n, edges)
        if res is not None:
            reps = res[0]
    if fix_label1 is not None:
        clauses.append([var(fix_label1, 1)])
        note = f"vertex {fix_label1} fixed to block 0"
    elif reps:
        clauses.append([var(rr, 1) for rr in reps])
        note = f"block 0 contains one of {len(reps)} orbit reps"
    else:
        adj = adjacency(n, edges)
        w = max(range(n), key=lambda x: len(adj[x]))
        for r in range((m - 1) // 2 + 2, m + 1):
            clauses.append([-var(w, r)])
        note = f"reversal: vertex {w} pinned to blocks 0..{(m - 1) // 2}"
    if cnotes:
        note += "; " + "; ".join(cnotes)
    return clauses, next_aux - 1, note


def build_cnf_cw(n, edges, c, fix_label1=None, symmetry="reversal"):
    """CNF for 'cutwidth <= c'. var(v,p)=v*n+p, position p in 1..n; exactly-one
    per vertex and per position; left[v,b] via reified OR; cross[e,b] XOR;
    at-most-c crossings per cut (Sinz counter)."""
    def var(v, p):
        return v * n + p

    clauses = []
    next_aux = n * n + 1
    for v in range(n):
        lits = [var(v, p) for p in range(1, n + 1)]
        clauses.append(lits[:])
        next_aux = _sinz_at_most(clauses, next_aux, lits, 1)
    for p in range(1, n + 1):
        lits = [var(v, p) for v in range(n)]
        clauses.append(lits[:])
        next_aux = _sinz_at_most(clauses, next_aux, lits, 1)
    left = {}
    for v in range(n):
        for b in range(1, n):
            lvar = next_aux
            next_aux += 1
            left[v, b] = lvar
            ors = [var(v, p) for p in range(1, b + 1)]
            clauses.append([-lvar] + ors)             # L -> (position <= b)
            for x in ors:
                clauses.append([-x, lvar])            # (v at p<=b) -> L
    for b in range(1, n):
        cross_lits = []
        for (u, v) in edges:
            xb = next_aux
            next_aux += 1
            lu, lv = left[u, b], left[v, b]
            clauses.append([-xb, lu, lv])             # xb -> lu | lv
            clauses.append([-xb, -lu, -lv])           # xb -> !(lu & lv)
            clauses.append([xb, -lu, lv])             # (lu & !lv) -> xb
            clauses.append([xb, lu, -lv])             # (!lu & lv) -> xb
            cross_lits.append(xb)
        next_aux = _sinz_at_most(clauses, next_aux, cross_lits, c)
    reps = None
    if symmetry == "orbit" and fix_label1 is None:
        res = orbit_representatives(n, edges)
        if res is not None:
            reps = res[0]
    if fix_label1 is not None:
        clauses.append([var(fix_label1, 1)])
        note = f"vertex {fix_label1} at position 1"
    elif reps:
        clauses.append([var(rr, 1) for rr in reps])
        note = f"position 1 is one of {len(reps)} orbit reps"
    else:
        adj = adjacency(n, edges)
        w = max(range(n), key=lambda x: len(adj[x]))
        for p in range((n - 1) // 2 + 2, n + 1):
            clauses.append([-var(w, p)])              # w in the left half
        note = f"reversal: vertex {w} pinned to positions 1..{(n - 1) // 2 + 1}"
    return clauses, next_aux - 1, note


def cw_decode(n, model_lits):
    true_vars = {x for x in model_lits if 0 < x <= n * n}
    lab = []
    for v in range(n):
        ps = [p for p in range(1, n + 1) if v * n + p in true_vars]
        if len(ps) != 1:
            sys.exit(f"vertex {v} has {len(ps)} positions - model invalid")
        lab.append(ps[0])
    return lab


def _cw_q(args, edges):
    """(q, min_intra, per_block) from the cw CLI flags. q = 1 is the plain
    permutation mode; q > 1 minimizes the BLOCKED cutwidth over supersites of
    q vertices, optionally under the ss hidden-bond constraints."""
    q = getattr(args, "block", 1) or 1
    if q > 1:
        return q, *ss_constraints(args, edges)
    if getattr(args, "min_intra_edges", 0) or getattr(args, "intra_per_block",
                                                      False):
        sys.exit("hidden-bond constraints require --block > 1")
    return 1, 0, False


def cmd_cw_run(args):
    n, edges = get_graph(args)
    q, min_intra, per_block = _cw_q(args, edges)
    if q > 1:
        ss_m(n, q)                           # validates q | n
    st = cw_state(args, n, edges)
    lbm = cutwidth_math_lb(n, edges, q)
    st.record_math_lb(lbm)
    if lbm > 1:
        st.record_unsat(lbm - 1, "math")
    blocked = f" (supersites of {q})" if q > 1 else ""
    print(f"[cw] cutwidth campaign{blocked}: {n} sites, {len(edges)} edges; "
          f"math lower bound {lbm}")
    if min_intra or per_block:
        print("[cw] hidden-bond constraints: "
              + ", ".join(([f">={min_intra} intra edges"] if min_intra else [])
                          + (["every block internally bonded"] if per_block
                             else [])))
    cur = st.read()
    if cur["best_labeling"] is None:
        if q == 1:
            cw, lab = sa_cutwidth(n, edges, None, args.seed, args.heur_time,
                                  procs=args.procs, stall=args.stall)
            st.record_labeling(lab, "cw-heuristic")
        else:
            vi, cw, lab = sa_sscw(n, edges, q, None, args.seed, args.heur_time,
                                  procs=args.procs, target=lbm,
                                  stall=args.stall, min_intra=min_intra,
                                  per_block=per_block)
            if vi == 0:
                st.record_labeling(lab, "cw-heuristic")
            else:
                print(f"[cw] heuristic ended with constraint violation {vi}; "
                      f"no upper bound recorded (CP-SAT searches from scratch)")
        cur = st.read()
    # Overall wall-clock budget for the whole ladder. Without it, a wide window
    # (dense graph) marches through dozens of decisions at up to time_per_k each
    # -> tens of hours. ladder_time <= 0 means unlimited (old behaviour).
    ladder_time = getattr(args, "ladder_time", 0) or 0
    t0 = time.time()
    # Retire a side once its decision times out: for a wide window the SAT side
    # (lower the UB) only gets harder as ub falls, so retrying k=ub-1 after every
    # UNSAT success just burns time (observed: 15 decisions, UB never moved).
    # Work the SAT side until it succeeds/gives up, then let the UNSAT side raise
    # the LB with the remaining budget -> a tighter certified interval.
    sat_dead = unsat_dead = False
    while True:
        if ladder_time > 0 and time.time() - t0 > ladder_time:
            print(f"[cw-run] ladder time budget ({ladder_time / 3600:.2f}h) "
                  f"exhausted; stopping with the best known layout")
            break
        lb, ub = State.window(cur)
        if ub is not None and lb >= ub:
            break
        if lb > len(edges):
            print("[cw-run] lower bound exceeds |E|: the hidden-bond "
                  "constraints are infeasible on this graph (e.g. no perfect "
                  "matching along bonds for q=2)")
            break
        if not sat_dead and ub is not None:
            side = 0                         # try to lower the upper bound
        elif not unsat_dead:
            side = 1                         # try to raise the lower bound
        else:
            print("[cw-run] both sides exhausted at this budget; stopping")
            break
        k = ub - 1 if side == 0 else lb
        print(f"[cw-run] window [{lb},{ub}] -> deciding cutwidth<={k} "
              f"({(time.time() - t0) / 3600:.2f}h into ladder)")
        hint = cur["best_labeling"] if side == 0 else None
        # never let a single decision overrun the remaining ladder budget
        dt = args.time_per_k
        if ladder_time > 0:
            dt = min(dt, max(1.0, ladder_time - (time.time() - t0)))
        if q == 1:
            res, lab, _ = cpsat_decide_cutwidth(n, edges, k, dt,
                                                args.workers, hint=hint,
                                                symmetry=args.symmetry,
                                                fix_label1=args.fix_label1)
        else:
            res, lab, _ = cpsat_decide_sscw(n, edges, q, k, dt,
                                            args.workers, hint=hint,
                                            symmetry=args.symmetry,
                                            fix_label1=args.fix_label1,
                                            min_intra=min_intra,
                                            intra_per_block=per_block)
        if res == "SAT":
            st.record_labeling(lab, f"cw-run(k={k})")
        elif res == "UNSAT":
            st.record_unsat(k, "cpsat")
        elif side == 0:
            sat_dead = True
            print(f"[cw-run] SAT side gave up at cutwidth<={k}; "
                  f"upper bound stays {ub}")
        else:
            unsat_dead = True
            print(f"[cw-run] UNSAT side gave up at cutwidth<={k}; "
                  f"lower bound stays {lb}")
        cur = st.read()
    lb, ub = State.window(st.read())
    print(f"[cw] certified window: {lb} <= cutwidth* <= {ub}")
    if ub is not None and lb >= ub:
        print(f"[cw] *** CERTIFIED OPTIMAL cutwidth: c* = {ub} ***")


def cmd_cw_decide(args):
    n, edges = get_graph(args)
    q, min_intra, per_block = _cw_q(args, edges)
    st = cw_state(args, n, edges)
    hint = st.read()["best_labeling"]
    if q == 1:
        res, lab, note = cpsat_decide_cutwidth(n, edges, args.k, args.time,
                                               args.workers, hint=hint,
                                               symmetry=args.symmetry,
                                               fix_label1=args.fix_label1)
    else:
        res, lab, note = cpsat_decide_sscw(n, edges, q, args.k, args.time,
                                           args.workers, hint=hint,
                                           symmetry=args.symmetry,
                                           fix_label1=args.fix_label1,
                                           min_intra=min_intra,
                                           intra_per_block=per_block)
    print(f"[cw-decide c={args.k}] {res}  (symmetry: {note})")
    if res == "SAT":
        st.record_labeling(lab, f"cw-decide(c={args.k})")
    elif res == "UNSAT":
        st.record_unsat(args.k, "cpsat")
    lb, ub = State.window(st.read())
    print(f"[cw] window now [{lb},{ub}]")


def _cw_cnf(args, n, edges):
    """Build the CNF for the (possibly blocked) cutwidth decision at --k."""
    q, min_intra, per_block = _cw_q(args, edges)
    if q == 1:
        clauses, nvars, note = build_cnf_cw(n, edges, args.k,
                                            fix_label1=args.fix_label1,
                                            symmetry=args.symmetry)
        conv = "var(v,p) = v*n + p, position p in 1..n"
    else:
        clauses, nvars, note = build_cnf_sscw(n, edges, q, args.k,
                                              fix_label1=args.fix_label1,
                                              symmetry=args.symmetry,
                                              min_intra=min_intra,
                                              intra_per_block=per_block)
        conv = f"var(v,r) = v*m + r, m = {n // q}, block index r-1"
    return q, clauses, nvars, note, conv


def cmd_cw_cnf(args):
    n, edges = get_graph(args)
    q, clauses, nvars, note, conv = _cw_cnf(args, n, edges)
    blocked = f" blocked (q={q})" if q > 1 else ""
    write_dimacs(args.out, clauses, nvars,
                 [f"cutwidth{blocked} decision, c={args.k}, n={n}, "
                  f"|E|={len(edges)}", conv, f"symmetry: {note}"])
    print(f"wrote {args.out}: {nvars} vars, {len(clauses)} clauses "
          f"(symmetry: {note})")


def cmd_cw_verify(args):
    n, edges = get_graph(args)
    st = cw_state(args, n, edges)
    q, clauses, nvars, note, _conv = _cw_cnf(args, n, edges)
    vd, mdl, proof = parallel_crosscheck(clauses, args.time,
                                         args.proof_out is not None)
    verdicts = list(vd.values())
    if args.cnf_out:
        write_dimacs(args.cnf_out, clauses, nvars,
                     [f"cutwidth c={args.k}" + (f" q={q}" if q > 1 else ""),
                      f"symmetry: {note}"])
    if args.proof_out and proof is not None:
        with open(args.proof_out, "w") as f:
            f.write("\n".join(proof) + "\n")
        print(f"[cw-verify] DRAT proof written to {args.proof_out}")
    if all(v == "UNSAT" for v in verdicts):
        st.record_unsat(args.k, "xsat")
        print("[cw-verify] both solvers UNSAT -> recorded (xsat)")
    elif "SAT" in verdicts and mdl is not None:
        lits = [x for x in mdl if x > 0]
        lab = cw_decode(n, lits) if q == 1 else ss_decode(n, q, lits)
        if cutwidth_of(lab, edges) <= args.k:
            st.record_labeling(lab, f"cw-pysat(k={args.k})")
    lb, ub = State.window(st.read())
    print(f"[cw] window now [{lb},{ub}]")


def cmd_cw_polish(args):
    """Among layouts with cutwidth <= c* (the current UB, or --target), find one
    minimizing the total interaction range (avg range x |E|) via CP-SAT, holding the
    bond dimension fixed. The refined layout is recorded into the state; the
    range tie-break in record_labeling keeps it only if it does not raise
    cutwidth and lowers the range."""
    n, edges = get_graph(args)
    q, min_intra, per_block = _cw_q(args, edges)
    st = cw_state(args, n, edges)
    cur = st.read()
    _, ub = State.window(cur)
    c = args.target if args.target is not None else ub
    if c is None:
        sys.exit("no cutwidth known yet; run cw-run first or pass --target")
    seed = cur["best_labeling"]
    hint = seed if (seed and cutwidth_of(seed, edges) <= c) else None
    r0 = total_range(seed, edges) if seed else None
    print(f"[cw-polish] minimizing total range at cutwidth <= {c} "
          f"(current range {r0})")
    # persist every improving incumbent immediately -> the polish is resumable:
    # if interrupted, the best so far is already saved; a re-run continues.
    def _record(lab, tr):
        st.record_labeling(lab, f"cw-polish(c={c})")
    if q == 1:
        status, lab, tr, lb, note = cpsat_polish_cutwidth(
            n, edges, c, args.time, args.workers, hint=hint,
            symmetry=args.symmetry, fix_label1=args.fix_label1, log=args.log,
            on_improve=_record)
    else:
        status, lab, tr, lb, note = cpsat_polish_sscw(
            n, edges, q, c, args.time, args.workers, hint=hint,
            symmetry=args.symmetry, fix_label1=args.fix_label1, log=args.log,
            min_intra=min_intra, intra_per_block=per_block,
            on_improve=_record)
    if lab is None:
        print("[cw-polish] no solution within the time limit")
        return
    cw = cutwidth_of(lab, edges)
    tag = "minimum total range certified" if status == "OPTIMAL" \
        else "feasible (not proven minimal)"
    print(f"[cw-polish] cutwidth {cw}, total range {tr} "
          f"(avg {tr / len(edges):.3f}); proven range lower bound {lb:.0f} "
          f"[{tag}]")
    st.record_labeling(lab, f"cw-polish(c={c})")
    print_status(st.read(), edges, objective="cutwidth", abbrev="c*",
                 value_fn=cutwidth_of)


def cmd_cw_export(args):
    n, edges = get_graph(args)
    q, min_intra, per_block = _cw_q(args, edges)
    st = cw_state(args, n, edges)
    cur = st.read()
    if not cur["best_labeling"]:
        sys.exit("no cutwidth assignment in the state yet")
    lab = cur["best_labeling"]
    print("cutwidth of this assignment:", cutwidth_of(lab, edges))
    if q > 1:
        n_in, uncovered = ss_intra_report(lab, edges, q)
        print(f"blocked (supersites of {q}): {n_in} edges hidden inside "
              f"blocks, {uncovered} blocks without an internal edge")
        print("(map is {site: 0-based block index})")
    print("{" + ", ".join(f"{v}: {lab[v] - 1}" for v in range(n)) + "}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, needs_k=False):
        p.add_argument("--cluster", default=None)
        p.add_argument("--edges-module", default="cluster_edges.py",
                       help="path to cluster_edges.py")
        p.add_argument("--edge-file", default=None,
                       help="alternative: plain edge-list file")
        p.add_argument("--state-dir", default="bw_state")
        p.add_argument("--symmetry", choices=["reversal", "orbit"],
                       default="reversal")
        p.add_argument("--fix-label1", type=int, default=None,
                       help="force label 1 onto this vertex (only valid if "
                            "the graph is vertex-transitive)")
        p.add_argument("--log", action="store_true",
                       help="CP-SAT search log")
        if needs_k:
            p.add_argument("--k", type=int, required=True)

    p = sub.add_parser("info");       common(p)
    p.add_argument("--orbits", action="store_true")
    p.set_defaults(fn=cmd_info)

    p = sub.add_parser("heuristic");  common(p)
    p.add_argument("--time", type=float, default=600)
    p.add_argument("--procs", type=int, default=os.cpu_count())
    p.add_argument("--seed", type=int, default=0,
                   help="master RNG seed (chain i in round r uses seed + r*procs + i)")
    p.add_argument("--stall", type=float, default=None,
                   help="stop after this many seconds without UB improvement "
                        "(default: max(60, time/10))")
    p.set_defaults(fn=cmd_heuristic)

    p = sub.add_parser("optimize");   common(p)
    p.add_argument("--time", type=float, default=3600)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.set_defaults(fn=cmd_optimize)

    p = sub.add_parser("decide");     common(p, needs_k=True)
    p.add_argument("--time", type=float, default=3600)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.set_defaults(fn=cmd_decide)

    p = sub.add_parser("ladder");     common(p)
    p.add_argument("--time-per-k", type=float, default=3600)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--procs", type=int, default=os.cpu_count())
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_ladder)

    p = sub.add_parser("cnf");        common(p, needs_k=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_cnf)

    p = sub.add_parser("decode");     common(p, needs_k=True)
    p.add_argument("--model", required=True, help="SAT solver model output file")
    p.set_defaults(fn=cmd_decode)

    p = sub.add_parser("verify-unsat"); common(p, needs_k=True)
    p.add_argument("--time", type=float, default=86400)
    p.add_argument("--proof-out", default=None,
                   help="write a DRAT proof here (Glucose backend)")
    p.add_argument("--cnf-out", default=None,
                   help="also archive the DIMACS CNF here")
    p.set_defaults(fn=cmd_verify_unsat)

    p = sub.add_parser("record-unsat"); common(p, needs_k=True)
    p.add_argument("--proof", choices=["cpsat", "xsat", "drat"], required=True)
    p.set_defaults(fn=cmd_record_unsat)

    p = sub.add_parser("import-labeling"); common(p)
    p.add_argument("--file", required=True,
                   help="labeling as JSON dict/list or whitespace labels")
    p.set_defaults(fn=cmd_import_labeling)

    p = sub.add_parser("polish");     common(p)
    p.add_argument("--block", type=int, default=1,
                   help="supersite size (1 = plain permutation); polishes the "
                        "matching ss state when >1")
    p.add_argument("--min-intra-edges", type=int, default=0,
                   help="ss only: keep >= N edges hidden inside supersites "
                        "(matches the _ie<N> state)")
    p.add_argument("--intra-per-block", action="store_true",
                   help="ss only: keep every supersite internally bonded "
                        "(matches the _ipb state)")
    p.add_argument("--target", type=int, default=None,
                   help="bandwidth to hold (default: current certified UB)")
    p.add_argument("--time", type=float, default=3600)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.set_defaults(fn=cmd_polish)

    p = sub.add_parser("status");     common(p)
    p.set_defaults(fn=cmd_status)

    # ---- lexicographic J1-J2 ----
    def lexcommon(p, needs_k=False):
        common(p, needs_k=needs_k)
        p.add_argument("--j2-cluster", default=None)
        p.add_argument("--j2-file", default=None)
        p.add_argument("--j2-edges-module", default=None,
                       help="module with the J2 CLUSTER_EDGES table for "
                            "--j2-cluster (default: a sibling cluster_edges_NNN.py "
                            "next to --edges-module, else --edges-module itself)")
        p.add_argument("--k1", type=int, default=None,
                       help="J1 cap (default: certified k1* from the cluster state)")

    p = sub.add_parser("lex-heuristic"); lexcommon(p)
    p.add_argument("--time", type=float, default=600)
    p.add_argument("--procs", type=int, default=os.cpu_count())
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_lex_heuristic)

    p = sub.add_parser("lex-ladder"); lexcommon(p)
    p.add_argument("--time-per-k", type=float, default=3600)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_lex_ladder)

    p = sub.add_parser("lex-decide"); lexcommon(p, needs_k=True)
    p.add_argument("--time", type=float, default=3600)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.set_defaults(fn=cmd_lex_decide)

    p = sub.add_parser("lex-cnf"); lexcommon(p, needs_k=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_lex_cnf)

    p = sub.add_parser("lex-verify"); lexcommon(p, needs_k=True)
    p.add_argument("--time", type=float, default=86400)
    p.add_argument("--proof-out", default=None)
    p.add_argument("--cnf-out", default=None)
    p.set_defaults(fn=cmd_lex_verify)

    p = sub.add_parser("lex-export"); lexcommon(p)
    p.set_defaults(fn=cmd_lex_export)

    # ---- spin-glass / dense weighted ----
    def wcommon(p, needs_kval=False):
        common(p)
        p.add_argument("--jmatrix", required=True,
                       help="dense |J| matrix: .npy or whitespace text")
        p.add_argument("--name", default=None,
                       help="state name (default: jmatrix filename stem)")
        p.add_argument("--cutoff", type=float, default=0.0,
                       help="drop entries <= cutoff * max|J| (relative)")
        p.add_argument("--digits", type=int, default=6,
                       help="integer scaling precision: w = round(|J|/max*10^digits)")
        if needs_kval:
            p.add_argument("--kval", type=float, required=True)
            p.add_argument("--original-units", action="store_true",
                           help="interpret --kval in original |J| units "
                                "instead of scaled integers")

    p = sub.add_parser("w-ladder"); wcommon(p)
    p.add_argument("--heur-time", type=float, default=300)
    p.add_argument("--time-per-k", type=float, default=3600)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--side", choices=["auto", "sat", "unsat"], default="auto")
    p.set_defaults(fn=cmd_w_ladder)

    p = sub.add_parser("w-decide"); wcommon(p, needs_kval=True)
    p.add_argument("--time", type=float, default=3600)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.set_defaults(fn=cmd_w_decide)

    p = sub.add_parser("w-cnf"); wcommon(p, needs_kval=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_w_cnf)

    p = sub.add_parser("w-verify"); wcommon(p, needs_kval=True)
    p.add_argument("--time", type=float, default=86400)
    p.add_argument("--proof-out", default=None)
    p.add_argument("--cnf-out", default=None)
    p.set_defaults(fn=cmd_w_verify)

    p = sub.add_parser("w-export"); wcommon(p)
    p.set_defaults(fn=cmd_w_export)

    # ---- 2-leg ladder host ----
    p = sub.add_parser("h2-run"); common(p)
    p.add_argument("--heur-time", type=float, default=300)
    p.add_argument("--time-per-k", type=float, default=3600)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_h2_run)

    p = sub.add_parser("h2-decide"); common(p, needs_k=True)
    p.add_argument("--time", type=float, default=3600)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.set_defaults(fn=cmd_h2_decide)

    p = sub.add_parser("h2-cnf"); common(p, needs_k=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_h2_cnf)

    p = sub.add_parser("h2-verify"); common(p, needs_k=True)
    p.add_argument("--time", type=float, default=86400)
    p.add_argument("--proof-out", default=None)
    p.add_argument("--cnf-out", default=None)
    p.set_defaults(fn=cmd_h2_verify)

    p = sub.add_parser("h2-export"); common(p)
    p.set_defaults(fn=cmd_h2_export)

    # ---- supersites (blocked rungs) ----
    def sscommon(p, needs_k=False):
        common(p, needs_k=needs_k)
        p.add_argument("--block", type=int, default=2,
                       help="spins per supersite (default 2 = ladder rungs)")
        p.add_argument("--min-intra-edges", type=int, default=0,
                       help="require at least this many edges hidden inside "
                            "supersites (0 = off). Uses a separate state file "
                            "tagged _ie<N>; certificates are conditional on "
                            "the constraint.")
        p.add_argument("--intra-per-block", action="store_true",
                       help="require EVERY supersite to contain at least one "
                            "edge (q=2: the blocking is a perfect matching "
                            "along bonds). Separate state tagged _ipb.")

    p = sub.add_parser("ss-run"); sscommon(p)
    p.add_argument("--heur-time", type=float, default=300)
    p.add_argument("--time-per-k", type=float, default=3600)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--procs", type=int, default=os.cpu_count(),
                   help="parallel SA chains in the heuristic phase")
    p.add_argument("--stall", type=float, default=None,
                   help="stop heuristic after this many seconds without "
                        "improvement (default max(60, heur-time/10))")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_ss_run)

    p = sub.add_parser("ss-decide"); sscommon(p, needs_k=True)
    p.add_argument("--time", type=float, default=3600)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.set_defaults(fn=cmd_ss_decide)

    p = sub.add_parser("ss-cnf"); sscommon(p, needs_k=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_ss_cnf)

    p = sub.add_parser("ss-verify"); sscommon(p, needs_k=True)
    p.add_argument("--time", type=float, default=86400)
    p.add_argument("--proof-out", default=None)
    p.add_argument("--cnf-out", default=None)
    p.set_defaults(fn=cmd_ss_verify)

    p = sub.add_parser("ss-export"); sscommon(p)
    p.set_defaults(fn=cmd_ss_export)

    p = sub.add_parser("ss-seed"); sscommon(p)
    p.add_argument("--generator", default="cluster_generator.py")
    p.add_argument("--lattice", required=True)
    p.add_argument("--mode", choices=["involution", "nn"], default="nn",
                   help="involution: pair r with r+t only when t is a true "
                        "half-period symmetry (strict). nn (default): pair each "
                        "site with its nearest neighbor along t and extract a "
                        "perfect matching, for ANY direction t.")
    p.add_argument("--Nx", type=int, default=None)
    p.add_argument("--Ny", type=int, default=None)
    p.add_argument("--Nz", type=int, default=None)
    p.add_argument("--supercell", default=None,
                   help='e.g. --supercell="-2,2,2;2,-2,2;2,2,-2"')
    p.add_argument("--tilted", default=None)
    p.add_argument("--trillium-u", type=float, default=0.25)
    p.add_argument("--heur-time", type=float, default=30)
    p.add_argument("--opt-time", type=float, default=60)
    p.add_argument("--procs", type=int, default=os.cpu_count())
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_ss_seed)

    p = sub.add_parser("ss-blocks"); sscommon(p)
    p.add_argument("--jmatrix", default=None,
                   help="optional coupling matrix (.npy/text) to annotate "
                        "each edge with its J value")
    p.add_argument("--out", default=None, help="write JSON here")
    p.set_defaults(fn=cmd_ss_blocks)

    # ---- cutwidth mode (minimize MPO bond dimension) ----
    def cwblock(p):
        p.add_argument("--block", type=int, default=1,
                       help="supersite size q (default 1 = plain permutation); "
                            "q > 1 minimizes the BLOCKED cutwidth, i.e. the "
                            "bond dimension of the blocked MPO")
        p.add_argument("--min-intra-edges", type=int, default=0,
                       help="(with --block) require >= N edges hidden inside "
                            "supersites")
        p.add_argument("--intra-per-block", action="store_true",
                       help="(with --block) require every supersite to contain "
                            "an interaction edge")

    p = sub.add_parser("cw-run"); common(p); cwblock(p)
    p.add_argument("--heur-time", type=float, default=120.0)
    p.add_argument("--time-per-k", type=float, default=300.0)
    p.add_argument("--ladder-time", type=float, default=14400.0,
                   help="overall wall-clock budget for the CP-SAT decision "
                        "ladder (phase 4); <=0 means unlimited. Prevents a wide "
                        "cutwidth window from running for tens of hours.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--procs", type=int, default=1)
    p.add_argument("--stall", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_cw_run)
    p = sub.add_parser("cw-decide"); common(p, needs_k=True); cwblock(p)
    p.add_argument("--time", type=float, default=300.0)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(fn=cmd_cw_decide)
    p = sub.add_parser("cw-cnf"); common(p, needs_k=True); cwblock(p)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_cw_cnf)
    p = sub.add_parser("cw-verify"); common(p, needs_k=True); cwblock(p)
    p.add_argument("--time", type=float, default=300.0)
    p.add_argument("--cnf-out", default=None)
    p.add_argument("--proof-out", default=None)
    p.set_defaults(fn=cmd_cw_verify)
    p = sub.add_parser("cw-polish"); common(p); cwblock(p)
    p.add_argument("--target", type=int, default=None,
                   help="hold cutwidth <= this (default: current UB)")
    p.add_argument("--time", type=float, default=1800.0)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(fn=cmd_cw_polish)
    p = sub.add_parser("cw-export"); common(p); cwblock(p)
    p.set_defaults(fn=cmd_cw_export)

    # ---- equivariant ansatz (needs cluster_generator.py) ----
    p = sub.add_parser("eq-run"); common(p)
    p.add_argument("--generator", default="cluster_generator.py")
    p.add_argument("--lattice", required=True)
    p.add_argument("--Nx", type=int, default=None)
    p.add_argument("--Ny", type=int, default=None)
    p.add_argument("--Nz", type=int, default=None)
    p.add_argument("--supercell", default=None,
                   help='e.g. --supercell="-2,2,2;2,-2,2;2,2,-2"')
    p.add_argument("--tilted", default=None)
    p.add_argument("--trillium-u", type=float, default=0.25)
    p.add_argument("--ansatz", choices=["cell", "basis", "both"],
                   default="both")
    p.add_argument("--time", type=float, default=600)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.set_defaults(fn=cmd_eq_run)

    p = sub.add_parser("export");     common(p)
    p.add_argument("--json", action="store_true",
                   help="emit JSON (string keys) instead of the default "
                        "integer-keyed Python dict literal")
    p.set_defaults(fn=cmd_export)

    args = ap.parse_args()
    if args.cluster is None and args.edge_file is None \
            and getattr(args, "jmatrix", None) is None:
        sys.exit("provide --cluster NAME, --edge-file PATH, or --jmatrix PATH")
    args.fn(args)


if __name__ == "__main__":
    main()