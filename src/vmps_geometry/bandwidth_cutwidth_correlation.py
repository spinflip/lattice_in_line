#!/usr/bin/env python3
"""
bandwidth_cutwidth_correlation.py — quantify how well BANDWIDTH predicts
CUTWIDTH (the MPO bond-dimension proxy) across every graph in cluster_edges.py.

The certifier minimizes bandwidth, but the quantity that actually sets the DMRG
MPO/MPS bond dimension is the cutwidth (cut_max = max number of Hamiltonian edges
crossing any chain cut). This script measures the bandwidth<->cutwidth correlation
so you know how good a proxy bandwidth is: it collects (bandwidth, cutwidth) pairs
per graph, then reports the Pearson (linear) and Spearman (rank) correlation and
draws a scatter plot with the least-squares fit.

Ordering sources (--sources, default all three):
  heuristics   the bandwidth/envelope heuristics (identity, cm, rcm, gps, king,
               sloan, spectral) run live on each graph
  sat          the SAT-certified orderings shipped in permutations_sat.py
  qubo         the QUBO-optimized orderings shipped in permutations_qubo.py
The sat/qubo solutions are the real optimized layouts (at or near the true
optimum), so including them anchors the correlation where it physically matters --
in --mode per-graph the best-of point uses them wherever a graph has one.

Two datasets (choose with --mode):
  per-graph  one point per graph = (best bandwidth found, best cutwidth found),
             each minimized independently over the heuristics as an estimate of
             the true optimum (default). Answers: "across our graphs, is a
             graph's bandwidth predictive of its cutwidth?"  (r ~ 0.95)
  all        every (graph, heuristic) ordering is a point. Answers: "across
             achievable orderings, does bandwidth track cutwidth?" This cloud
             includes deliberately-bad orderings (king/sloan spike cutwidth on
             dense graphs), so its linear r is lower while the rank rho stays high.

Usage:
  python -m vmps_geometry.bandwidth_cutwidth_correlation   # -> plots/bandwidth_cutwidth_correlation.png
  vmps-bandwidth-cutwidth-correlation --mode per-graph --out corr.png
  vmps-bandwidth-cutwidth-correlation --graphs pyrochlore,kagome --csv pts.csv

Plotting needs matplotlib (the [plot] extra); without it the correlations are
still printed and a CSV can still be written.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple


def _plots_dir() -> str:
    """Repo-root plots/ folder (gitignored scratch output), resolved from this
    module's location so it works regardless of the caller's cwd."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    d = os.path.join(root, "plots")
    os.makedirs(d, exist_ok=True)
    return d

import numpy as np

from .cluster_edges import CLUSTER_EDGES
from .bandwidth_heuristics_benchmark import (
    ALGORITHMS,
    DEFAULT_ORDER,
    compute_bandwidth,
    compute_cutwidth,
    normalize_edges,
    run_one,
)

# Optimized/certified orderings shipped in the repo, as {graph: {site: position}}.
# These are the real solutions (SAT-certified or QUBO-optimized), typically at or
# near the true optimum -- far better than the bandwidth/envelope heuristics.
OPT_SOURCES = ("sat", "qubo")


def _load_opt_source(name: str) -> Dict[str, Dict[int, int]]:
    if name == "sat":
        from .permutations_sat import CUSTOM_PERMUTATIONS
    elif name == "qubo":
        from .permutations_qubo import CUSTOM_PERMUTATIONS
    else:
        raise ValueError(name)
    return CUSTOM_PERMUTATIONS


def _perm_to_ordering(perm: Dict[int, int], vertices: List[int]):
    """Invert a {site: position} table into a pos->site ordering list (the
    convention compute_bandwidth/compute_cutwidth expect). Returns None if the
    table is not a valid permutation of this graph's vertices."""
    if set(perm) != set(vertices):
        return None
    if sorted(perm.values()) != list(range(len(vertices))):
        return None
    order = [None] * len(vertices)
    for site, pos in perm.items():
        order[pos] = site
    return order


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson linear correlation; nan if either side has zero variance."""
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank) — Spearman without scipy."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    # average tied ranks
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation = Pearson of the ranks."""
    if len(x) < 2:
        return float("nan")
    return pearson(_rankdata(x), _rankdata(y))


def _linfit(x: np.ndarray, y: np.ndarray):
    """Least-squares slope, intercept; None if x has no spread."""
    if len(x) >= 2 and np.std(x) > 0:
        a, b = np.polyfit(x, y, 1)
        return float(a), float(b)
    return None


def stats_of(pts: List[Dict]) -> Dict:
    """Correlation + linear fit summary for a set of points."""
    x = np.array([p["bandwidth"] for p in pts], dtype=float)
    y = np.array([p["cutwidth"] for p in pts], dtype=float)
    fit = _linfit(x, y)
    return {
        "n": len(pts),
        "r": pearson(x, y),
        "rho": spearman(x, y),
        "slope": fit[0] if fit else float("nan"),
        "intercept": fit[1] if fit else float("nan"),
    }


# marker + fit-line colour per data category
STYLES = {
    "heuristic": {"marker": "o", "label": "heuristics", "color": "tab:red"},
    "sat":       {"marker": "s", "label": "SAT-certified", "color": "tab:blue"},
    "qubo":      {"marker": "^", "label": "QUBO-optimized", "color": "tab:green"},
    "best":      {"marker": "o", "label": "best per graph", "color": "tab:red"},
}


def collect_points(names: List[str], algs: List[str], sources: List[str],
                   mode: str) -> List[Dict]:
    """Return records with keys graph, n, category, algorithm, bandwidth,
    cutwidth. `sources` is any subset of {'heuristics', 'sat', 'qubo'}."""
    opt = {s: _load_opt_source(s) for s in sources if s in OPT_SOURCES}
    pts: List[Dict] = []
    for g in names:
        vertices, edges = normalize_edges(CLUSTER_EDGES[g])
        n = len(vertices)
        recs: List[Tuple[str, str, int, int]] = []   # (category, label, bw, cut)
        if "heuristics" in sources:
            for a in algs:
                r = run_one(edges, a)
                recs.append(("heuristic", a, r["bandwidth"], r["cut_max"]))
        for sname, table in opt.items():
            if g not in table:
                continue
            order = _perm_to_ordering(table[g], vertices)
            if order is None:
                print(f"[warn] {sname} permutation for {g} is not a valid "
                      f"permutation of its {n} vertices; skipping",
                      file=sys.stderr)
                continue
            recs.append((sname, sname, compute_bandwidth(edges, order),
                         compute_cutwidth(edges, order)))
        if not recs:
            continue
        if mode == "per-graph":
            pts.append({
                "graph": g, "n": n, "category": "best", "algorithm": "best",
                "bandwidth": min(r[2] for r in recs),
                "cutwidth": min(r[3] for r in recs),
            })
        else:  # all
            for cat, label, bw, cut in recs:
                pts.append({
                    "graph": g, "n": n, "category": cat, "algorithm": label,
                    "bandwidth": bw, "cutwidth": cut,
                })
    return pts


def plot(pts: List[Dict], fit: str, out: str, show: bool) -> bool:
    """Scatter of cutwidth vs bandwidth. `fit` is 'per-category' (a separate
    regression line + r for each of heuristics/SAT/QUBO) or 'pooled' (one line
    over all points) or 'none'."""
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[plot] matplotlib unavailable ({e}); skipping plot. "
              f"Install the [plot] extra to enable it.", file=sys.stderr)
        return False

    x = np.array([p["bandwidth"] for p in pts], dtype=float)
    y = np.array([p["cutwidth"] for p in pts], dtype=float)
    n = np.array([p["n"] for p in pts], dtype=float)
    cats_present = [c for c in STYLES if any(p["category"] == c for p in pts)]

    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    norm = Normalize(vmin=n.min(), vmax=n.max())
    sc = None
    for cat in cats_present:                 # marker per category, colour = n
        idx = [i for i, p in enumerate(pts) if p["category"] == cat]
        sc = ax.scatter(x[idx], y[idx], c=n[idx], cmap="viridis", norm=norm,
                        marker=STYLES[cat]["marker"], s=46, alpha=0.85,
                        edgecolors="black", linewidths=0.4, zorder=3)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("cluster size  n (sites)")

    legend_handles = []
    if fit == "per-category" and len(cats_present) > 1:
        # one regression line + one r per category (the honest view: these are
        # different populations, so a single pooled slope would be misleading)
        for cat in cats_present:
            idx = [i for i, p in enumerate(pts) if p["category"] == cat]
            cx, cy = x[idx], y[idx]
            f = _linfit(cx, cy)
            col = STYLES[cat]["color"]
            if f is not None:
                a, b = f
                xs = np.array([cx.min(), cx.max()])
                ax.plot(xs, a * xs + b, color=col, lw=1.8, zorder=2)
                lab = (f"{STYLES[cat]['label']}: r={pearson(cx, cy):.2f}, "
                       f"slope={a:.2f}")
            else:
                lab = STYLES[cat]["label"]
            legend_handles.append(Line2D([], [], color=col,
                                         marker=STYLES[cat]["marker"],
                                         markeredgecolor="black", label=lab))
    elif fit != "none":                      # pooled single line
        f = _linfit(x, y)
        if f is not None:
            a, b = f
            xs = np.array([x.min(), x.max()])
            ax.plot(xs, a * xs + b, "r-", lw=1.8, zorder=2,
                    label=fr"fit: $C = {a:.2f}\,B + {b:.2f}$  (r={pearson(x, y):.2f})")
        for cat in cats_present:             # still label the markers
            if len(cats_present) > 1:
                legend_handles.append(Line2D(
                    [], [], marker=STYLES[cat]["marker"], linestyle="none",
                    color="0.4", markeredgecolor="black",
                    label=STYLES[cat]["label"]))

    # y = x reference (cutwidth == bandwidth)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, alpha=0.5, zorder=1,
            label="cutwidth = bandwidth")

    ax.set_xlabel(r"bandwidth $B$")
    ax.set_ylabel(r"cutwidth $C$")
    st = stats_of(pts)
    ax.set_title(f"Pearson r = {st['r']:.3f}    Spearman ρ = {st['rho']:.3f}")
    auto, _ = ax.get_legend_handles_labels()
    ax.legend(handles=auto + legend_handles, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    root, _ = os.path.splitext(out)
    fig.savefig(out, dpi=300)
    pdf = root + ".pdf"
    fig.savefig(pdf)                       # vector PDF (dpi-independent)
    print(f"[plot] wrote {out} (300 dpi) and {pdf}")
    if show:
        plt.show()
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["all", "per-graph"], default="per-graph",
                    help="'per-graph' = one best-of point per graph (default); "
                         "'all' = every graph x heuristic point")
    ap.add_argument("--graphs", default=None,
                    help="comma-separated names/substrings to filter graphs")
    ap.add_argument("--algorithms", default=",".join(DEFAULT_ORDER),
                    help=f"comma-separated subset of {list(ALGORITHMS)}")
    ap.add_argument("--sources", default="heuristics,sat,qubo",
                    help="comma-separated subset of {heuristics, sat, qubo}: "
                         "which orderings to include. sat/qubo are the "
                         "certified/optimized solutions shipped in the repo "
                         "(default: all three)")
    ap.add_argument("--fit", choices=["per-category", "pooled", "none"],
                    default="per-category",
                    help="regression drawn on the scatter: 'per-category' fits "
                         "heuristics/SAT/QUBO separately (default -- they are "
                         "different populations; a pooled slope is misleading and "
                         "leverage-dominated); 'pooled' = one line over all points")
    ap.add_argument("--out", default=None,
                    help="output image path (default: "
                         "plots/bandwidth_cutwidth_correlation.png)")
    ap.add_argument("--csv", default=None, help="also write the points to CSV")
    ap.add_argument("--no-plot", action="store_true", help="skip the plot")
    ap.add_argument("--show", action="store_true",
                    help="open an interactive window as well as saving")
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(_plots_dir(), "bandwidth_cutwidth_correlation.png")

    algs = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    for a in algs:
        if a not in ALGORITHMS:
            ap.error(f"unknown algorithm {a!r}; choose from {list(ALGORITHMS)}")

    valid_sources = ("heuristics",) + OPT_SOURCES
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    for s in sources:
        if s not in valid_sources:
            ap.error(f"unknown source {s!r}; choose from {list(valid_sources)}")

    names = sorted(CLUSTER_EDGES)
    if args.graphs:
        want = [g.strip() for g in args.graphs.split(",") if g.strip()]
        names = [g for g in names if any(w == g or w in g for w in want)]
        if not names:
            ap.error("no graphs matched --graphs")

    pts = collect_points(names, algs, sources, args.mode)
    if not pts:
        ap.error("no data points collected (no graphs matched the chosen "
                 "sources). Check --sources / --graphs.")
    x = np.array([p["bandwidth"] for p in pts], dtype=float)
    y = np.array([p["cutwidth"] for p in pts], dtype=float)
    overall = stats_of(pts)

    tag = "per-graph best-of" if args.mode == "per-graph" else "per ordering"
    print("=" * 70)
    print(f"bandwidth vs cutwidth over {len(names)} graph(s), mode={args.mode} "
          f"({tag})")
    print(f"  sources         : {', '.join(sources)}")
    print(f"  points          : {len(pts)}")
    print(f"  bandwidth range : {int(x.min())} .. {int(x.max())}")
    print(f"  cutwidth  range : {int(y.min())} .. {int(y.max())}")
    print(f"  cut/bw ratio    : mean {np.mean(y / x):.3f}, "
          f"median {np.median(y / x):.3f}, max {np.max(y / x):.3f}")
    # per-category breakdown: these are different populations, so report each
    cats = [c for c in STYLES if any(p["category"] == c for p in pts)]
    if len(cats) > 1:
        print("  --- per category (fit each separately) ---")
        print(f"  {'category':14s} {'#pts':>5s} {'Pearson':>8s} "
              f"{'Spearman':>9s} {'slope':>7s} {'intcpt':>7s}")
        for c in cats:
            s = stats_of([p for p in pts if p["category"] == c])
            print(f"  {STYLES[c]['label']:14s} {s['n']:5d} {s['r']:8.4f} "
                  f"{s['rho']:9.4f} {s['slope']:7.2f} {s['intercept']:7.2f}")
        print("  --- pooled (all points together; leverage-dominated) ---")
    print(f"  Pearson  r      : {overall['r']:.4f}   (linear association)")
    print(f"  Spearman rho    : {overall['rho']:.4f}   "
          f"(rank / monotonic association)")
    print("=" * 70)

    if args.csv:
        import csv
        cols = (["graph", "n", "category", "algorithm", "bandwidth", "cutwidth"]
                if args.mode == "all" else
                ["graph", "n", "bandwidth", "cutwidth"])
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(pts)
        print(f"[csv] wrote {args.csv} ({len(pts)} rows)")

    if not args.no_plot:
        plot(pts, args.fit, args.out, args.show)


if __name__ == "__main__":
    main()
