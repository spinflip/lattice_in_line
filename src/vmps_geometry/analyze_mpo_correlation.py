#!/usr/bin/env python3
"""
analyze_mpo_correlation.py — does the DMRG MPO bond dimension track bandwidth or
cutwidth, and when do the two objectives diverge?

It reads the per-entry comment stats written in the permutation tables
  # bandwidth=B, envelope=E, cutwidth=C
  # MPO dAux_avg=.., dAux_max=D[, dAux_sum=..]
from permutations_sat_bw.py (bandwidth-optimized layouts) and
permutations_sat_cw.py (cutwidth-optimized layouts), then:

  1. correlates the MPO bond dimension d_aux^max against bandwidth and against
     cutwidth (pooled over both files) -> which is the better predictor;
  2. for every cluster present in BOTH files, measures the bandwidth<->cutwidth
     divergence (how much the bandwidth-optimal ordering costs in cutwidth, and
     how much bandwidth the cutwidth-optimal ordering wastes).

Three figures are written (PNG at 300 dpi + vector PDF) under the repo's plots/
folder by default:
  plots/mpo_bonddim.png     the two matched pairings: peak d_aux^max vs cutwidth
                            (the DMRG cost driver) | mean d_aux^avg vs envelope
  plots/mpo_bandwidth.png   peak d_aux^max vs bandwidth (the proxy-of-a-proxy)
  plots/mpo_divergence.png  arrows from the bandwidth-opt to the cutwidth-opt
                            layout of each shared cluster in (bandwidth, cutwidth)

Usage:
  vmps-analyze-mpo-correlation
  vmps-analyze-mpo-correlation --out-prefix figs/mpo   # custom prefix/location
  vmps-analyze-mpo-correlation --no-plot                # numbers only
  vmps-analyze-mpo-correlation --bw-file a.py --cw-file b.py

Plotting needs matplotlib (the [plot] extra); without it the numbers still print.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List, Optional

import numpy as np

from .bandwidth_cutwidth_correlation import _linfit, _plots_dir, pearson, spearman

_MARK = {"bw": "o", "cw": "s"}
_COLOR = {"bw": "tab:blue", "cw": "tab:orange"}
_LABEL = {"bw": "bandwidth-opt", "cw": "cutwidth-opt"}

_RE_GEO = re.compile(
    r"#\s*bandwidth=([0-9.]+),\s*envelope=([0-9.]+),\s*cutwidth=([0-9]+)")
_RE_MPO = re.compile(
    r"#\s*MPO\s+dAux_avg=([0-9.]+),\s*dAux_max=([0-9]+)"
    r"(?:,\s*dAux_sum=([0-9]+))?")
_RE_KEY = re.compile(r'"([A-Za-z0-9_]+)"\s*:')


def parse_file(path: str, source: str) -> List[Dict]:
    """Parse the '# bandwidth=.. / # MPO dAux_..' comment stats preceding each
    "cluster": {...} entry into records. dAux fields are optional."""
    recs: List[Dict] = []
    cur: Dict = {}
    with open(path) as f:
        for line in f:
            s = line.strip()
            m = _RE_GEO.match(s)
            if m:
                cur.update(bandwidth=float(m.group(1)),
                           envelope=float(m.group(2)),
                           cutwidth=int(m.group(3)))
                continue
            m = _RE_MPO.match(s)
            if m:
                cur.update(daux_avg=float(m.group(1)),
                           daux_max=int(m.group(2)))
                if m.group(3):
                    cur["daux_sum"] = int(m.group(3))
                continue
            m = _RE_KEY.match(s)
            if m:
                if "cutwidth" in cur:
                    cur["cluster"] = m.group(1)
                    cur["source"] = source
                    recs.append(cur)
                cur = {}
    return recs


def _default_file(name: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def correlations(recs: List[Dict]) -> Dict:
    """Correlate d_aux^max with bandwidth and cutwidth over records that carry a
    dAux value. Returns a summary dict and prints it."""
    have = [r for r in recs if "daux_max" in r]
    bw = np.array([r["bandwidth"] for r in have], dtype=float)
    cut = np.array([r["cutwidth"] for r in have], dtype=float)
    dam = np.array([r["daux_max"] for r in have], dtype=float)
    out = {
        "n": len(have),
        "bw_r": pearson(bw, dam), "bw_rho": spearman(bw, dam),
        "cut_r": pearson(cut, dam), "cut_rho": spearman(cut, dam),
    }
    print("=" * 68)
    print(f"peak MPO bond dim d_aux^max over {len(have)} DMRG-measured layouts")
    print(f"  vs cutwidth  : Pearson {out['cut_r']:.3f}   Spearman {out['cut_rho']:.3f}")
    print(f"  vs bandwidth : Pearson {out['bw_r']:.3f}   Spearman {out['bw_rho']:.3f}")
    winner = "cutwidth" if out["cut_r"] > out["bw_r"] else "bandwidth"
    print(f"  -> {winner} is the better predictor of the peak bond dimension")
    have_avg = [r for r in have if "daux_avg" in r]
    if have_avg:
        env = np.array([r["envelope"] for r in have_avg], dtype=float)
        daa = np.array([r["daux_avg"] for r in have_avg], dtype=float)
        out["env_r"] = pearson(env, daa)
        print(f"mean MPO bond dim d_aux^avg over {len(have_avg)} layouts")
        print(f"  vs envelope  : Pearson {out['env_r']:.3f}   "
              f"Spearman {spearman(env, daa):.3f}  "
              f"(near-identity: avg cut-load = R*|E|/(n-1))")
    print("=" * 68)
    return out


# Quasi-1D hyperkagome variants (thin Nx1x1 / 2x2x1 supercells) excluded from
# the divergence figure by default -- they crowd the low corner and don't
# diverge; override with --exclude (empty string keeps everything).
DEFAULT_DIVERGENCE_EXCLUDE = ("hyperkagome48_4x1x1", "hyperkagome72_6x1x1",
                              "hyperkagome48_2x2x1", "hyperkagome60_5x1x1")


def divergence(recs: List[Dict], exclude=()) -> List[Dict]:
    """For clusters present in BOTH sources (minus `exclude`), quantify the
    bandwidth<->cutwidth objective divergence."""
    exclude = set(exclude)
    bwm = {r["cluster"]: r for r in recs if r["source"] == "bw"}
    cwm = {r["cluster"]: r for r in recs if r["source"] == "cw"}
    rows: List[Dict] = []
    for c in sorted(set(bwm) & set(cwm) - exclude):
        b, w = bwm[c], cwm[c]
        rows.append({
            "cluster": c,
            "cut_bwopt": b["cutwidth"], "cut_cwopt": w["cutwidth"],
            "bw_bwopt": b["bandwidth"], "bw_cwopt": w["bandwidth"],
            "cut_ratio": b["cutwidth"] / w["cutwidth"] if w["cutwidth"] else 1.0,
        })
    rows.sort(key=lambda r: r["cut_ratio"], reverse=True)
    skipped = sorted((set(bwm) & set(cwm)) & exclude)
    if skipped:
        print(f"\n(excluded {len(skipped)} cluster(s) from the divergence "
              f"figure: {', '.join(skipped)})")
    print(f"bandwidth<->cutwidth divergence over {len(rows)} shared cluster(s):")
    print(f"  {'cluster':26s}{'cut(bw-opt)':>12s}{'cut(cw-opt)':>12s}"
          f"{'ratio':>7s}{'bw(bw-opt)':>11s}{'bw(cw-opt)':>11s}")
    for r in rows:
        flag = "  <- diverges" if r["cut_bwopt"] - r["cut_cwopt"] > 2 else ""
        print(f"  {r['cluster']:26s}{r['cut_bwopt']:12d}{r['cut_cwopt']:12d}"
              f"{r['cut_ratio']:7.2f}{r['bw_bwopt']:11.0f}{r['bw_cwopt']:11.0f}"
              f"{flag}")
    return rows


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 14})
        return plt
    except Exception as e:  # noqa: BLE001
        print(f"[plot] matplotlib unavailable ({e}); skipping figures "
              f"(install the [plot] extra).", file=sys.stderr)
        return None


def _save(fig, prefix: str, name: str) -> None:
    fig.savefig(f"{prefix}_{name}.png", dpi=300)
    fig.savefig(f"{prefix}_{name}.pdf")
    print(f"[plot] wrote {prefix}_{name}.png (300 dpi) and {prefix}_{name}.pdf")


def _panel(ax, have: List[Dict], xkey: str, ykey: str, xlabel: str,
           ylabel: str, xsym: str, ysym: str) -> None:
    """Scatter ykey vs xkey coloured/marked by source, with a separate
    least-squares fit per layout family whose full linear law (ysym = a*xsym + b,
    r) is printed in the legend. `xsym`/`ysym` are mathtext symbols (no $)."""
    from matplotlib.lines import Line2D
    x = np.array([r[xkey] for r in have], dtype=float)
    y = np.array([r[ykey] for r in have], dtype=float)
    handles = []
    for s in [s for s in ("bw", "cw") if any(r["source"] == s for r in have)]:
        idx = [i for i, r in enumerate(have) if r["source"] == s]
        xs_s, ys_s = x[idx], y[idx]
        ax.scatter(xs_s, ys_s, c=_COLOR[s], marker=_MARK[s], s=46, alpha=0.85,
                   edgecolors="black", linewidths=0.4, zorder=3)
        f = _linfit(xs_s, ys_s)
        if f is not None:
            a, b = f
            xr = np.array([xs_s.min(), xs_s.max()])
            ax.plot(xr, a * xr + b, color=_COLOR[s], lw=1.8, zorder=2)
            lab = (f"{_LABEL[s]}: ${ysym} = {a:.2f}\\,{xsym} {b:+.2f}$"
                   f"  (r={pearson(xs_s, ys_s):.2f})")
        else:
            lab = _LABEL[s]
        handles.append(Line2D([], [], color=_COLOR[s], marker=_MARK[s],
                              markeredgecolor="black", label=lab))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(handles=handles, loc="upper left", framealpha=0.9)


def plot_bonddim(recs: List[Dict], prefix: str) -> None:
    """Main figure: the two matched pairings -- the peak MPO bond dimension vs
    cutwidth (what sets DMRG cost), and the mean bond dimension vs envelope."""
    plt = _plt()
    if plt is None:
        return
    have = [r for r in recs if "daux_max" in r and "daux_avg" in r]
    fig, axs = plt.subplots(1, 2, figsize=(13.4, 5.8))
    _panel(axs[0], have, "cutwidth", "daux_max", r"cutwidth $C$",
           r"peak MPO bond dim  $\chi_{\mathrm{max}}$",
           "C", r"\chi_{\mathrm{max}}")
    _panel(axs[1], have, "envelope", "daux_avg",
           r"average interaction range $R$",
           r"mean MPO bond dim  $\chi_{\mathrm{avg}}$",
           "R", r"\chi_{\mathrm{avg}}")
    fig.tight_layout()
    _save(fig, prefix, "bonddim")


def plot_bonddim_bandwidth(recs: List[Dict], prefix: str) -> None:
    """Second figure: the peak MPO bond dimension against bandwidth -- the
    proxy-of-a-proxy, weaker and family-dependent than the cutwidth pairing."""
    plt = _plt()
    if plt is None:
        return
    have = [r for r in recs if "daux_max" in r]
    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    _panel(ax, have, "bandwidth", "daux_max", r"bandwidth $B$",
           r"peak MPO bond dim  $\chi_{\mathrm{max}}$",
           "B", r"\chi_{\mathrm{max}}")
    fig.tight_layout()
    _save(fig, prefix, "bandwidth")


def plot_divergence(recs: List[Dict], rows: List[Dict], prefix: str) -> None:
    plt = _plt()
    if plt is None:
        return
    from matplotlib.lines import Line2D
    if not rows:
        print("[plot] no shared clusters; skipping divergence figure",
              file=sys.stderr)
        return
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    vals = [v for r in recs for v in (r["bandwidth"], r["cutwidth"])]
    lo, hi = min(vals) * 0.8, max(vals) * 1.25
    # log-log: bandwidth spans 4..>200 while most clusters sit at 5..30, so a
    # linear scale crushes them into the corner; log spreads them out and the
    # C = B reference stays a straight line.
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.4)
    for r in rows:
        diff = r["cut_bwopt"] - r["cut_cwopt"]      # cutwidth the bw-opt loses
        if diff > 2:                                # real divergence
            col, lw, al = "red", 2.0, 0.95
        elif diff == 2:                             # borderline
            col, lw, al = "darkorange", 1.8, 0.9
        elif diff == 1:                             # marginal
            col, lw, al = "gold", 1.6, 0.8
        else:                                       # objectives agree
            col, lw, al = "0.6", 1.0, 0.5
        ax.annotate("", xy=(r["bw_cwopt"], r["cut_cwopt"]),
                    xytext=(r["bw_bwopt"], r["cut_bwopt"]),
                    arrowprops=dict(arrowstyle="->", color=col, lw=lw, alpha=al))
        ax.scatter([r["bw_bwopt"]], [r["cut_bwopt"]], c="tab:blue", s=42,
                   edgecolors="black", linewidths=0.4, zorder=4)
        ax.scatter([r["bw_cwopt"]], [r["cut_cwopt"]], c="tab:orange",
                   marker="s", s=42, edgecolors="black", linewidths=0.4,
                   zorder=4)
        if diff > 2:                                # name only the divergers
            ax.annotate(r["cluster"], (r["bw_cwopt"], r["cut_cwopt"]),
                        color="red", xytext=(4, -2), textcoords="offset points",
                        fontsize=8)
    marker_legend = ax.legend(handles=[
        Line2D([], [], marker="o", color="w", markerfacecolor="tab:blue",
               markeredgecolor="k", label="bandwidth-optimal"),
        Line2D([], [], marker="s", color="w", markerfacecolor="tab:orange",
               markeredgecolor="k", label="cutwidth-optimal"),
    ], loc="upper left", framealpha=0.95)
    ax.add_artist(marker_legend)
    ax.legend(handles=[
        Line2D([], [], color="red", lw=2, label="diverge (cutwidth drops > 2)"),
        Line2D([], [], color="darkorange", lw=2, label="cutwidth drops 2"),
        Line2D([], [], color="gold", lw=2, label="cutwidth drops 1"),
        Line2D([], [], color="0.6", lw=1, label="agree (cutwidth unchanged)"),
        Line2D([], [], ls="--", color="k", alpha=0.4, label="C = B"),
    ], loc="lower right", framealpha=0.95)
    ax.set_xlabel(r"bandwidth $B$")
    ax.set_ylabel(r"cutwidth $C$")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, prefix, "divergence")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bw-file", default=_default_file("permutations_sat_bw.py"),
                    help="bandwidth-optimized permutation table")
    ap.add_argument("--cw-file",
                    default=_default_file("permutations_sat_cw.py"),
                    help="cutwidth-optimized permutation table")
    ap.add_argument("--out-prefix", default=None,
                    help="output figure path prefix (default: plots/mpo)")
    ap.add_argument("--no-plot", action="store_true", help="skip the figures")
    ap.add_argument("--exclude", default=",".join(DEFAULT_DIVERGENCE_EXCLUDE),
                    help="comma-separated clusters to drop from the divergence "
                         "figure (default: the quasi-1D hyperkagome variants; "
                         "pass an empty string to keep all)")
    args = ap.parse_args()
    if args.out_prefix is None:
        args.out_prefix = os.path.join(_plots_dir(), "mpo")

    recs: List[Dict] = []
    for path, src in [(args.bw_file, "bw"), (args.cw_file, "cw")]:
        if not os.path.isfile(path):
            print(f"[warn] {path} not found; skipping", file=sys.stderr)
            continue
        recs += parse_file(path, src)
    if not recs:
        ap.error("no permutation entries parsed from the given files")

    correlations(recs)
    rows = divergence(recs, [c.strip() for c in args.exclude.split(",")
                             if c.strip()])
    n_daux = sum(1 for r in recs if "daux_max" in r)
    if not args.no_plot:
        # up-front, on stdout, so a skip is never silent
        if _plt() is None:
            print("[plot] NO FIGURES WRITTEN: matplotlib is not available in "
                  "this Python interpreter. Install it with\n"
                  "         pip install matplotlib\n"
                  "       (or `pip install -e .[plot]` from the repo), then re-run.")
        elif n_daux == 0:
            print("[plot] NO FIGURES WRITTEN: none of the parsed entries carry "
                  "MPO dAux comment stats (# MPO dAux_avg=.., dAux_max=..).\n"
                  "       The plots need those; the permutation tables must "
                  "include them.")
        else:
            print(f"[plot] writing figures under {os.path.dirname(args.out_prefix)}/ ...")
            plot_bonddim(recs, args.out_prefix)
            plot_bonddim_bandwidth(recs, args.out_prefix)
            plot_divergence(recs, rows, args.out_prefix)


if __name__ == "__main__":
    main()
