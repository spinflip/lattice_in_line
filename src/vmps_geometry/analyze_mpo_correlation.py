#!/usr/bin/env python3
"""
analyze_mpo_correlation.py — does the DMRG MPO bond dimension track bandwidth or
cutwidth, and when do the two objectives diverge?

It reads the per-entry comment stats written in the permutation tables
  # bandwidth=B, envelope=E, cutwidth=C
  # MPO dAux_avg=.., dAux_max=D[, dAux_sum=..]
from permutations_sat.py (bandwidth-optimized layouts) and
permutations_sat_cutwidth.py (cutwidth-optimized layouts), then:

  1. correlates the MPO bond dimension d_aux^max against bandwidth and against
     cutwidth (pooled over both files) -> which is the better predictor;
  2. for every cluster present in BOTH files, measures the bandwidth<->cutwidth
     divergence (how much the bandwidth-optimal ordering costs in cutwidth, and
     how much bandwidth the cutwidth-optimal ordering wastes).

Two figures are written (PNG at 300 dpi + vector PDF):
  <prefix>_bonddim.png     d_aux^max vs bandwidth | vs cutwidth, with r / rho
  <prefix>_divergence.png  arrows from the bandwidth-opt to the cutwidth-opt
                           layout of each shared cluster in (bandwidth, cutwidth)

Usage:
  vmps-analyze-mpo-correlation
  vmps-analyze-mpo-correlation --out-prefix figs/mpo --no-plot   # numbers only
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

from .bandwidth_cutwidth_correlation import _linfit, pearson, spearman

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
    print(f"MPO bond dimension d_aux^max over {len(have)} DMRG-measured layouts")
    print(f"  vs bandwidth : Pearson {out['bw_r']:.3f}   Spearman {out['bw_rho']:.3f}")
    print(f"  vs cutwidth  : Pearson {out['cut_r']:.3f}   Spearman {out['cut_rho']:.3f}")
    winner = "cutwidth" if out["cut_r"] > out["bw_r"] else "bandwidth"
    print(f"  -> {winner} is the better predictor of the MPO bond dimension")
    print("=" * 68)
    return out


def divergence(recs: List[Dict]) -> List[Dict]:
    """For clusters present in BOTH sources, quantify the objective divergence."""
    bwm = {r["cluster"]: r for r in recs if r["source"] == "bw"}
    cwm = {r["cluster"]: r for r in recs if r["source"] == "cw"}
    rows: List[Dict] = []
    for c in sorted(set(bwm) & set(cwm)):
        b, w = bwm[c], cwm[c]
        rows.append({
            "cluster": c,
            "cut_bwopt": b["cutwidth"], "cut_cwopt": w["cutwidth"],
            "bw_bwopt": b["bandwidth"], "bw_cwopt": w["bandwidth"],
            "cut_ratio": b["cutwidth"] / w["cutwidth"] if w["cutwidth"] else 1.0,
        })
    rows.sort(key=lambda r: r["cut_ratio"], reverse=True)
    print(f"\nbandwidth<->cutwidth divergence over {len(rows)} shared cluster(s):")
    print(f"  {'cluster':26s}{'cut(bw-opt)':>12s}{'cut(cw-opt)':>12s}"
          f"{'ratio':>7s}{'bw(bw-opt)':>11s}{'bw(cw-opt)':>11s}")
    for r in rows:
        flag = "  <- diverges" if r["cut_ratio"] > 1.05 else ""
        print(f"  {r['cluster']:26s}{r['cut_bwopt']:12d}{r['cut_cwopt']:12d}"
              f"{r['cut_ratio']:7.2f}{r['bw_bwopt']:11.0f}{r['bw_cwopt']:11.0f}"
              f"{flag}")
    return rows


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:  # noqa: BLE001
        print(f"[plot] matplotlib unavailable ({e}); skipping figures "
              f"(install the [plot] extra).", file=sys.stderr)
        return None


def _save(fig, prefix: str, name: str) -> None:
    fig.savefig(f"{prefix}_{name}.png", dpi=300)
    fig.savefig(f"{prefix}_{name}.pdf")
    print(f"[plot] wrote {prefix}_{name}.png (300 dpi) and {prefix}_{name}.pdf")


def plot_bonddim(recs: List[Dict], prefix: str) -> None:
    plt = _plt()
    if plt is None:
        return
    from matplotlib.lines import Line2D
    have = [r for r in recs if "daux_max" in r]
    bw = np.array([r["bandwidth"] for r in have])
    cut = np.array([r["cutwidth"] for r in have])
    dam = np.array([r["daux_max"] for r in have])
    srcs = [s for s in ("bw", "cw") if any(r["source"] == s for r in have)]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    for ax, x, xlab in [(axs[0], bw, r"bandwidth $B$"),
                        (axs[1], cut, r"cutwidth $C$")]:
        handles = []
        for s in srcs:                          # fit each population separately
            idx = [i for i, r in enumerate(have) if r["source"] == s]
            xs_s, ys_s = x[idx], dam[idx]
            ax.scatter(xs_s, ys_s, c=_COLOR[s], marker=_MARK[s], s=46,
                       alpha=0.85, edgecolors="black", linewidths=0.4, zorder=3)
            f = _linfit(xs_s, ys_s)
            if f is not None:
                a, b = f
                xr = np.array([xs_s.min(), xs_s.max()])
                ax.plot(xr, a * xr + b, color=_COLOR[s], lw=1.8, zorder=2)
                lab = (f"{_LABEL[s]}: r={pearson(xs_s, ys_s):.2f}, "
                       f"slope={a:.2f}")
            else:
                lab = _LABEL[s]
            handles.append(Line2D([], [], color=_COLOR[s], marker=_MARK[s],
                                  markeredgecolor="black", label=lab))
        ax.set_xlabel(xlab)
        ax.grid(alpha=0.25)
        ax.legend(handles=handles, loc="upper left", framealpha=0.9, fontsize=9)
        ax.set_title(f"$d_{{aux}}^{{max}}$ vs {xlab.split()[0]}   "
                     f"(pooled r={pearson(x, dam):.3f})")
    axs[0].set_ylabel(r"MPO bond dimension  $d_{aux}^{max}$")
    fig.suptitle(f"Which objective predicts the MPO bond dimension? "
                 f"({len(have)} DMRG-measured layouts, fitted per family)",
                 fontsize=13)
    fig.tight_layout()
    _save(fig, prefix, "bonddim")


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
        div = r["cut_ratio"] > 1.05
        col = "crimson" if div else "0.6"
        ax.annotate("", xy=(r["bw_cwopt"], r["cut_cwopt"]),
                    xytext=(r["bw_bwopt"], r["cut_bwopt"]),
                    arrowprops=dict(arrowstyle="->", color=col,
                                    lw=1.8 if div else 1.0,
                                    alpha=0.9 if div else 0.5))
        ax.scatter([r["bw_bwopt"]], [r["cut_bwopt"]], c="tab:blue", s=42,
                   edgecolors="black", linewidths=0.4, zorder=4)
        ax.scatter([r["bw_cwopt"]], [r["cut_cwopt"]], c="tab:orange",
                   marker="s", s=42, edgecolors="black", linewidths=0.4,
                   zorder=4)
        if div:
            ax.annotate(f"{r['cluster']} (×{r['cut_ratio']:.2f})",
                        (r["bw_cwopt"], r["cut_cwopt"]), fontsize=8,
                        color="crimson", xytext=(4, -2),
                        textcoords="offset points")
    ax.legend(handles=[
        Line2D([], [], marker="o", color="w", markerfacecolor="tab:blue",
               markeredgecolor="k", label="bandwidth-opt"),
        Line2D([], [], marker="s", color="w", markerfacecolor="tab:orange",
               markeredgecolor="k", label="cutwidth-opt"),
        Line2D([], [], color="crimson", lw=2, label="diverge (cutwidth drops >5%)"),
        Line2D([], [], color="0.6", lw=1, label="agree (cutwidth ~unchanged)"),
        Line2D([], [], ls="--", color="k", alpha=0.4, label="C = B"),
    ], loc="lower right", framealpha=0.95, fontsize=9)
    ax.set_xlabel(r"bandwidth $B$")
    ax.set_ylabel(r"cutwidth $C$  (MPO bond-dim proxy)")
    ax.set_title("When do the objectives diverge?\n"
                 "arrow = bandwidth-opt → cutwidth-opt layout, same cluster")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, prefix, "divergence")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bw-file", default=_default_file("permutations_sat.py"),
                    help="bandwidth-optimized permutation table")
    ap.add_argument("--cw-file",
                    default=_default_file("permutations_sat_cutwidth.py"),
                    help="cutwidth-optimized permutation table")
    ap.add_argument("--out-prefix", default="mpo",
                    help="output figure path prefix (default: mpo)")
    ap.add_argument("--no-plot", action="store_true", help="skip the figures")
    args = ap.parse_args()

    recs: List[Dict] = []
    for path, src in [(args.bw_file, "bw"), (args.cw_file, "cw")]:
        if not os.path.isfile(path):
            print(f"[warn] {path} not found; skipping", file=sys.stderr)
            continue
        recs += parse_file(path, src)
    if not recs:
        ap.error("no permutation entries parsed from the given files")

    correlations(recs)
    rows = divergence(recs)
    if not args.no_plot:
        plot_bonddim(recs, args.out_prefix)
        plot_divergence(recs, rows, args.out_prefix)


if __name__ == "__main__":
    main()
