#!/usr/bin/env python3
"""
fiedler_ordering.py — spectral (Fiedler) site ordering for finite-size DMRG,
from spin-spin correlation data <S_i . S_j> (or mutual information).

Why and what
------------
For an MPS/DMRG calculation, the cost driver is the entanglement the state
must carry across each bond of the 1D ordering. The standard, cheap, and
effective heuristic (Barcza-Legeza-Marti-Reiher orbital ordering in quantum
chemistry DMRG) is to order sites by the Fiedler vector — the eigenvector of
the second-smallest eigenvalue of the weighted graph Laplacian L = D - W —
which is the continuous relaxation of minimizing sum_ij W_ij (pos_i - pos_j)^2.

FM/AFM asymmetry (--weights modes)
----------------------------------
Raw |<S_i.S_j>| measures TOTAL correlation, not entanglement. For SU(2)-
symmetric spin-1/2 states the two-site reduced density matrix is fixed by
c = <S_i.S_j> in [-3/4, +1/4], and its pair entanglement (concurrence) is

    C(c) = max(0, -2c - 1/2),

which is ZERO for all c >= -1/4: strongly ferromagnetically correlated pairs
(c -> +1/4, locally aligned, near-product) cost almost no bond dimension when
separated, while singlet-like AFM pairs (c -> -3/4) are maximally expensive.
Weighting by |c| therefore OVERWEIGHTS ferromagnetic bonds. Modes:

  --weights abs          W_ij = |C_ij|                  (sign-blind; use only
                         if you know FM correlations are weak/absent)
  --weights concurrence  W_ij = max(0, -2*C_ij - 1/2)   (recommended when the
                         input is <S_i.S_j> of a spin-1/2 SU(2)-symmetric
                         state; FM pairs correctly get weight ~ 0)
  --weights mi           input matrix IS mutual information I_ij; used as-is
                         (the best option if your DMRG code can export
                         two-site mutual information — it captures the
                         FM/AFM asymmetry automatically and is the
                         community-standard weight)

Caveats: pair concurrence ignores multipartite entanglement (an FM domain
still carries some collective entanglement in the symmetric sector, typically
small/logarithmic), and the formula assumes SU(2) symmetry and spin-1/2; with
fields, anisotropy, or higher spin, prefer --weights mi.

Optimization and reporting
--------------------------
Objectives (--objective):
  quad  : sum W_ij d_ij^2   (the Fiedler objective; default)
  cut   : max over chain cuts of total W crossing  (the proxy for the worst
          MPS bond dimension — usually what you want for DMRG)
--refine runs a windowed local search on top of the Fiedler order.
--anneal SECS runs a multi-start parallel simulated annealing (--procs chains;
  swap / segment-reversal / relocation moves) that optimizes the objective
  DIRECTLY, using the Fiedler order only as one seed among random restarts.
  Use this when the plain Fiedler order underperforms — typically on dense
  correlation matrices, where the sea of weak long-range weights dominates
  the spectral (quadratic) relaxation and washes out the strong bonds:
      ... --weights concurrence --objective cut --anneal 600
The report prints, for each order produced: weighted cutwidth (max and
profile quantiles), sum W*d, sum W*d^2, and max W*d; the best order under
--objective is the one written out.

MPO note: the bond dimension of the MPO itself counts INTERACTION TERMS
crossing each cut and is independent of coupling sign and of correlations;
if you also care about MPO compactness, order/check against the Hamiltonian
graph (unweighted cutwidth/bandwidth — your certifier handles that side).

Usage
-----
  python fiedler_ordering.py CORR.npy --weights concurrence --refine
  python fiedler_ordering.py corr.txt --weights mi --objective cut --refine
  # direct SA optimization of the DMRG bond proxy (recommended when the
  # Fiedler order gives poor DMRG energies):
  python fiedler_ordering.py CORR.npy --objective cut --anneal 600 --out order.txt
  # results JSON: extract correlations of the lowest-energy run automatically
  python fiedler_ordering.py results.json --refine --out order.txt

Input: EITHER a square matrix (.npy or whitespace text), OR a results .json
(e.g. model=Heis_sys=pyrochlore64_Mlimit=8000.json). For a .json the script
finds the run with the lowest energy that carries correlations and builds the
signed <S_i.S_j> matrix from its 'correlations.values' (fields i, j, and the
value field, default 'SdagS'). Output: the ordering as an integer-keyed map
{original: position} (0-based) to stdout (and to --out if given), plus a report.
Requires: numpy.
"""

import argparse
import json
import math
import os
import random
import sys

import numpy as np


# ----------------------------------------------------------------------

def load_matrix(path):
    M = np.load(path) if path.endswith(".npy") else np.loadtxt(path)
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        sys.exit("input must be a square matrix")
    M = (M + M.T) / 2.0
    np.fill_diagonal(M, 0.0)
    return M


def _run_energy(run):
    """Total ground-state energy of a run (fall back to per-site), or None."""
    gs = run.get("ground_state", {}) if isinstance(run, dict) else {}
    if isinstance(gs, dict):
        for k in ("energy", "energy_per_site"):
            if gs.get(k) is not None:
                return float(gs[k])
    if isinstance(run, dict) and run.get("energy") is not None:
        return float(run["energy"])
    return None


def _run_corr_values(run):
    """List of pairwise-correlation entries for a run, or None."""
    if not isinstance(run, dict):
        return None
    c = run.get("correlations")
    if isinstance(c, dict):
        return c.get("values")
    if isinstance(c, list):
        return c
    return None


def load_results_json(path, corr_field="SdagS"):
    """Build the signed <S_i.S_j> matrix from the lowest-energy run that has
    correlations in a results JSON. Returns the n x n matrix."""
    with open(path) as f:
        d = json.load(f)
    runs = d.get("runs") if isinstance(d, dict) else None
    if not runs:
        sys.exit(f"{path}: no 'runs' found (not a results JSON?)")
    items = runs.items() if isinstance(runs, dict) else enumerate(runs)

    global_min = None
    candidates = []   # (energy, name, values) for runs that have correlations
    for name, run in items:
        e = _run_energy(run)
        if e is not None:
            global_min = e if global_min is None else min(global_min, e)
        vals = _run_corr_values(run)
        if vals:
            candidates.append((e, str(name), vals))
    if not candidates:
        sys.exit(f"{path}: no run contains correlations")

    # lowest energy among runs with correlations (unknown energies sort last)
    candidates.sort(key=lambda t: (t[0] is None, t[0] if t[0] is not None else 0.0))
    e_sel, name_sel, vals = candidates[0]

    n = 1 + max(max(int(v["i"]), int(v["j"])) for v in vals)
    C = np.zeros((n, n), dtype=float)
    for v in vals:
        if corr_field not in v:
            sys.exit(f"correlation entry has no field '{corr_field}': {v} "
                     f"(use --corr-field to name the value field)")
        i, j, w = int(v["i"]), int(v["j"]), float(v[corr_field])
        C[i, j] = w
        C[j, i] = w
    np.fill_diagonal(C, 0.0)

    print(f"[json] {len(candidates)} run(s) with correlations; using lowest-energy "
          f"'{name_sel}' (E={e_sel}); built {n}x{n} signed <S_i.S_j> from "
          f"{len(vals)} pairs (field '{corr_field}')", file=sys.stderr)
    if global_min is not None and e_sel is not None and e_sel > global_min + 1e-9:
        print(f"[json] note: a lower-energy run (E={global_min}) has no "
              f"correlations; ordering uses the lowest-energy run that does",
              file=sys.stderr)
    return C


def make_weights(C, mode):
    if mode == "abs":
        W = np.abs(C)
    elif mode == "concurrence":
        W = np.maximum(0.0, -2.0 * C - 0.5)
    elif mode == "mi":
        if (C < -1e-12).any():
            sys.exit("--weights mi expects a nonnegative mutual-information "
                     "matrix; got negative entries (did you pass <S_i.S_j>?)")
        W = np.maximum(C, 0.0)
    else:
        sys.exit(f"unknown weight mode {mode}")
    np.fill_diagonal(W, 0.0)
    if W.max() <= 0:
        sys.exit("weight matrix is identically zero under this mode "
                 "(e.g. all-FM correlations with --weights concurrence); "
                 "nothing to order — any ordering is equivalent for this proxy")
    return W


# ----------------------------------------------------------------------

def fiedler_order(W):
    """Order = argsort of the Fiedler vector of L = D - W."""
    n = W.shape[0]
    L = np.diag(W.sum(axis=1)) - W
    vals, vecs = np.linalg.eigh(L)
    # eigenvalues ascending; index 0 ~ constant vector (or per-component);
    # take the first eigenvector with non-trivial variation
    for idx in range(1, n):
        f = vecs[:, idx]
        if np.ptp(f) > 1e-12:
            break
    order = list(np.argsort(f, kind="stable"))
    return order  # order[pos] = original site


# ----------------------------------------------------------------------
# objectives (pos[v] = 0-based position of site v)

def _edge_arrays(W):
    """(iu, ju, w) of the nonzero upper-triangle weights."""
    iu, ju = np.triu_indices(W.shape[0], 1)
    w = W[iu, ju]
    nz = w > 0
    return iu[nz], ju[nz], w[nz]


def _cut_profile(pos, iu, ju, w, n):
    """Weight crossing each of the n-1 chain bonds, vectorized (O(E + n)):
    an edge spanning positions [lo, hi] crosses bonds lo..hi-1; accumulate as
    interval increments and prefix-sum."""
    lo = np.minimum(pos[iu], pos[ju])
    hi = np.maximum(pos[iu], pos[ju])
    diff = np.zeros(n, dtype=float)
    np.add.at(diff, lo, w)
    np.add.at(diff, hi, -w)
    return np.cumsum(diff)[:n - 1]


def metrics(W, pos):
    n = W.shape[0]
    iu, ju, w = _edge_arrays(W)
    d = np.abs(pos[iu] - pos[ju]).astype(float)
    cuts = _cut_profile(pos, iu, ju, w, n)
    return {
        "cutwidth_max": float(cuts.max()),
        "cut_profile_q": [float(np.quantile(cuts, q)) for q in (0.5, 0.9, 1.0)],
        "sum_wd": float((w * d).sum()),
        "sum_wd2": float((w * d * d).sum()),
        "max_wd": float((w * d).max()),
    }


def objective_value(W, pos, objective):
    m = metrics(W, pos)
    return m["sum_wd2"] if objective == "quad" else m["cutwidth_max"]


def refine(W, order, objective, time_budget, seed):
    """Windowed local search: adjacent swaps + random window-relocations."""
    import time as _t
    rng = random.Random(seed)
    n = len(order)
    pos = np.empty(n, dtype=int)
    for p, v in enumerate(order):
        pos[v] = p
    best = objective_value(W, pos, objective)
    t_end = _t.time() + time_budget
    cur = pos.copy()
    curval = best
    bestpos = pos.copy()
    # inv[p] = the site currently at position p (inverse of the site->position
    # map `cur`), kept in sync so the swaps/relocations below are O(1) lookups
    # instead of O(n) np.where scans.
    inv = np.empty(n, dtype=int)
    inv[cur] = np.arange(n)
    while _t.time() < t_end:
        improved = False
        # pass 1: all adjacent transpositions
        for p in range(n - 1):
            a = int(inv[p])
            b = int(inv[p + 1])
            cur[a], cur[b] = cur[b], cur[a]
            inv[p], inv[p + 1] = b, a
            val = objective_value(W, cur, objective)
            if val < curval - 1e-12:
                curval = val
                improved = True
            else:
                cur[a], cur[b] = cur[b], cur[a]
                inv[p], inv[p + 1] = a, b
        # pass 2: random relocations of a site to a nearby position
        for _ in range(2 * n):
            v = rng.randrange(n)
            shift = rng.randint(-4, 4)
            p_old = int(cur[v])
            p_new = min(n - 1, max(0, p_old + shift))
            if p_new == p_old:
                continue
            newpos = cur.copy()
            step = 1 if p_new > p_old else -1
            for q in range(p_old, p_new, step):
                w_at = int(inv[q + step])
                newpos[w_at] = q
            newpos[v] = p_new
            val = objective_value(W, newpos, objective)
            if val < curval - 1e-12:
                cur, curval, improved = newpos, val, True
                inv[cur] = np.arange(n)   # accepted -> rebuild the inverse
        if curval < best - 1e-12:
            best, bestpos = curval, cur.copy()
        if not improved:
            break
    return bestpos


# ----------------------------------------------------------------------
# annealed ordering (--anneal): multi-start SA that OPTIMIZES the objective
# directly, using Fiedler only as one seed. This is the tool to reach for when
# the spectral order is poor -- e.g. a dense correlation matrix whose sea of
# weak long-range weights dominates the quadratic relaxation.

def _sa_cost(pos, iu, ju, w, n, objective):
    """(primary, secondary): for 'cut' the max bond weight with the total
    range sum_wd as a tie-break gradient (the max alone is a flat landscape);
    for 'quad' the smooth sum_wd2 alone."""
    d = np.abs(pos[iu] - pos[ju]).astype(float)
    if objective == "quad":
        return float((w * d * d).sum()), 0.0
    cuts = _cut_profile(pos, iu, ju, w, n)
    return float(cuts.max()), float((w * d).sum())


def _sa_order_chain(payload):
    """One independent annealing chain over permutations (picklable for
    mp.Pool). Moves: pair swap, segment reversal (2-opt), site relocation.
    Metropolis anneals on the LEADING worsened cost term only, relative-scaled
    and exp-clamped (cf. the certifier's supersite SA)."""
    import time as _t
    W, objective, seed, t_budget, init_perm = payload
    rng = random.Random(seed)
    n = W.shape[0]
    iu, ju, w = _edge_arrays(W)
    if init_perm is not None:
        perm = np.asarray(init_perm, dtype=int).copy()
    else:
        perm = np.arange(n)
        rng.shuffle(perm)
    pos = np.empty(n, dtype=int)
    pos[perm] = np.arange(n)
    p1, s1 = _sa_cost(pos, iu, ju, w, n, objective)
    bestperm, bp, bs = perm.copy(), p1, s1
    T = 0.05                              # relative-delta temperature
    t_end = _t.time() + t_budget
    while _t.time() < t_end:
        for _ in range(200):
            new = perm.copy()
            r = rng.random()
            if r < 0.4:                                    # pair swap
                a, b = rng.randrange(n), rng.randrange(n)
                new[a], new[b] = new[b], new[a]
            elif r < 0.7:                                  # segment reversal
                a, b = sorted((rng.randrange(n), rng.randrange(n)))
                new[a:b + 1] = new[a:b + 1][::-1]
            else:                                          # site relocation
                a, b = rng.randrange(n), rng.randrange(n)
                site = new[a]
                new = np.delete(new, a)
                new = np.insert(new, b, site)
            pos[new] = np.arange(n)
            p2, s2 = _sa_cost(pos, iu, ju, w, n, objective)
            if (p2, s2) <= (p1, s1):
                accept = True
            else:
                if p2 != p1:
                    delta = (p2 - p1) / max(abs(p1), 1e-12)
                else:
                    delta = 0.1 * (s2 - s1) / max(abs(s1), 1e-12)
                accept = delta / T < 700 and rng.random() < math.exp(-delta / T)
            if accept:
                perm, p1, s1 = new, p2, s2
                if (p1, s1) < (bp, bs):
                    bestperm, bp, bs = perm.copy(), p1, s1
        T = max(1e-4, T * 0.97)
    return bp, bs, [int(v) for v in bestperm]


def anneal_order(W, objective, time_budget, procs, seed, seed_perms):
    """Parallel multi-start SA. seed_perms: initial permutations (e.g. the
    Fiedler order); half the chains restart from the incumbent, half randomly.
    Returns the best perm (perm[pos] = site)."""
    import time as _t
    n = W.shape[0]
    iu, ju, w = _edge_arrays(W)

    def cost_of(perm):
        pos = np.empty(n, dtype=int)
        pos[np.asarray(perm)] = np.arange(n)
        return _sa_cost(pos, iu, ju, w, n, objective)

    best_perm = min(seed_perms, key=cost_of)
    bp, bs = cost_of(best_perm)
    print(f"[anneal] seed {objective}: {bp:.6g}", file=sys.stderr)
    if procs <= 1:
        p, s, perm = _sa_order_chain((W, objective, seed, time_budget, best_perm))
        return perm if (p, s) < (bp, bs) else best_perm
    import multiprocessing as mp
    t_end = _t.time() + time_budget
    rnd = 0
    with mp.Pool(procs) as pool:
        while True:
            dur = min(20.0, t_end - _t.time())
            if dur < 1:
                break
            jobs = [(W, objective, seed + rnd * procs + p, dur,
                     best_perm if p % 2 == 0 else None)
                    for p in range(procs)]
            rnd += 1
            for p, s, perm in pool.imap_unordered(_sa_order_chain, jobs):
                if (p, s) < (bp, bs):
                    bp, bs, best_perm = p, s, perm
                    print(f"[anneal] new best {objective}: {bp:.6g}",
                          file=sys.stderr)
    return best_perm


# ----------------------------------------------------------------------

def report(tag, W, pos):
    m = metrics(W, pos)
    print(f"  {tag:10s} cutwidth(max) {m['cutwidth_max']:.6g}   "
          f"cut q50/q90/max {m['cut_profile_q'][0]:.4g}/"
          f"{m['cut_profile_q'][1]:.4g}/{m['cut_profile_q'][2]:.4g}   "
          f"sum w*d {m['sum_wd']:.6g}   sum w*d^2 {m['sum_wd2']:.6g}   "
          f"max w*d {m['max_wd']:.6g}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("matrix", help="correlation/MI matrix (.npy or text), "
                                   "OR a results .json (correlations of the "
                                   "lowest-energy run are extracted)")
    ap.add_argument("--corr-field", default="SdagS",
                    help="value field of each correlation entry in a results "
                         "JSON (default: SdagS)")
    ap.add_argument("--out", default=None,
                    help="also write the permutation map to this file "
                         "(import-labeling format)")
    ap.add_argument("--weights", choices=["abs", "concurrence", "mi"],
                    default="concurrence")
    ap.add_argument("--refine", action="store_true",
                    help="local search on top of the Fiedler order")
    ap.add_argument("--objective", choices=["quad", "cut"], default="quad",
                    help="optimization objective: quad = sum w*d^2 (Fiedler "
                         "objective), cut = weighted cutwidth (DMRG bond proxy)")
    ap.add_argument("--refine-time", type=float, default=30.0)
    ap.add_argument("--anneal", type=float, default=0.0, metavar="SECS",
                    help="multi-start parallel SA that optimizes --objective "
                         "directly, seeded by the Fiedler order plus random "
                         "restarts. The recommended optimizer when the plain "
                         "Fiedler order is poor (e.g. dense correlation "
                         "matrices); combine with --objective cut for DMRG.")
    ap.add_argument("--procs", type=int, default=os.cpu_count(),
                    help="parallel SA chains for --anneal")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.matrix.endswith(".json"):
        C = load_results_json(args.matrix, args.corr_field)
    else:
        C = load_matrix(args.matrix)
    n = C.shape[0]
    if args.weights != "mi" and (C.max() > 0.26 or C.min() < -0.76):
        print("[warn] entries outside [-3/4, 1/4]: input may not be spin-1/2 "
              "<S_i.S_j>; check normalization or use --weights mi",
              file=sys.stderr)
    W = make_weights(C, args.weights)

    order = fiedler_order(W)
    pos_f = np.empty(n, dtype=int)
    for p, v in enumerate(order):
        pos_f[v] = p

    print(f"n = {n}, weight mode = {args.weights}, "
          f"nonzero weights = {int((np.triu(W, 1) > 0).sum())}")
    report("identity", W, np.arange(n))
    report("fiedler", W, pos_f)
    candidates = [pos_f]
    if args.refine:
        pos_r = refine(W, order, args.objective, args.refine_time, args.seed)
        report(f"refined({args.objective})", W, pos_r)
        candidates.append(pos_r)
    if args.anneal > 0:
        # seed the SA with every order we have so far (as perms: perm[pos]=site)
        seed_perms = [list(np.argsort(pos)) for pos in candidates]
        perm_a = anneal_order(W, args.objective, args.anneal, args.procs,
                              args.seed, seed_perms)
        pos_a = np.empty(n, dtype=int)
        pos_a[np.asarray(perm_a)] = np.arange(n)
        report(f"anneal({args.objective})", W, pos_a)
        candidates.append(pos_a)
    pos_best = min(candidates,
                   key=lambda p: objective_value(W, p, args.objective))

    order_map = "{" + ", ".join(f"{v}: {int(pos_best[v])}" for v in range(n)) + "}"
    print("\nordering map {original_site: position}:")
    print(order_map)
    if args.out:
        with open(args.out, "w") as f:
            f.write(order_map + "\n")
        print(f"[out] wrote permutation to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
