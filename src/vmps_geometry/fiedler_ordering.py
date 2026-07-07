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

Refinement and reporting
------------------------
--refine runs a windowed local search on top of the Fiedler order, minimizing
  --objective quad  : sum W_ij d_ij^2   (the Fiedler objective; default)
  --objective cut   : max over chain cuts of total W crossing  (the proxy for
                      the worst MPS bond dimension — usually what you want
                      for DMRG; slightly slower to evaluate)
The report prints, for the identity, Fiedler, and refined orders: weighted
cutwidth (max and profile quartiles), sum W*d, sum W*d^2, and max W*d.

MPO note: the bond dimension of the MPO itself counts INTERACTION TERMS
crossing each cut and is independent of coupling sign and of correlations;
if you also care about MPO compactness, order/check against the Hamiltonian
graph (unweighted cutwidth/bandwidth — your certifier handles that side).

Usage
-----
  python fiedler_ordering.py CORR.npy --weights concurrence --refine
  python fiedler_ordering.py corr.txt --weights mi --objective cut --refine
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

def metrics(W, pos):
    n = W.shape[0]
    iu, ju = np.triu_indices(n, 1)
    w = W[iu, ju]
    nz = w > 0
    iu, ju, w = iu[nz], ju[nz], w[nz]
    d = np.abs(pos[iu] - pos[ju]).astype(float)
    cuts = np.zeros(n - 1)
    lo = np.minimum(pos[iu], pos[ju])
    hi = np.maximum(pos[iu], pos[ju])
    for a, b, wt in zip(lo, hi, w):
        cuts[a:b] += wt
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
                    help="refinement objective: quad = sum w*d^2 (Fiedler "
                         "objective), cut = weighted cutwidth (DMRG bond proxy)")
    ap.add_argument("--refine-time", type=float, default=30.0)
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
    pos_best = pos_f
    if args.refine:
        pos_r = refine(W, order, args.objective, args.refine_time, args.seed)
        report(f"refined({args.objective})", W, pos_r)
        if objective_value(W, pos_r, args.objective) \
                <= objective_value(W, pos_f, args.objective):
            pos_best = pos_r

    order_map = "{" + ", ".join(f"{v}: {int(pos_best[v])}" for v in range(n)) + "}"
    print("\nordering map {original_site: position}:")
    print(order_map)
    if args.out:
        with open(args.out, "w") as f:
            f.write(order_map + "\n")
        print(f"[out] wrote permutation to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
