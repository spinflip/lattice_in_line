#!/usr/bin/env python3
"""
gather_permutations.py — collect the layouts a certification campaign produced in
a lattice_in_line_data folder into a CUSTOM_PERMUTATIONS map, in the same format as
permutations_sat_bw.py / permutations_sat_cw.py.

It scans the per-cluster run directories a campaign writes:
  cutwidth (MODE=cutwidth):  <data>/cw_run_<cluster>/<cluster>__cw.json
  bandwidth (plain):         <data>/bw_run_<cluster>/<cluster>.json
reads each state's best_labeling, converts it to a {site: position} map (0-based),
and annotates every entry with the layout's bandwidth / avg_range / cutwidth plus
whether the objective is certified optimal or just a heuristic upper bound.

Usage:
  lil-gather-permutations                        # cutwidth, ~/lattice_in_line_data -> stdout
  lil-gather-permutations --mode bw --out permutations_bw_gathered.py
  lil-gather-permutations --data-dir /path/to/lattice_in_line_data --mode cw \
      --out src/lattice_in_line/permutations_sat_cw.py

Stats are computed from cluster_edges.py, so a run whose cluster is not in
CLUSTER_EDGES is still emitted (permutation only, stats noted as unavailable).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional

from .cluster_edges import CLUSTER_EDGES
from .bandwidth_heuristics_benchmark import normalize_edges
from .bandwidth_certifier import bandwidth_of, cutwidth_of, total_range

MODES = {
    "cw": {"prefix": "cw_run_", "suffix": "__cw", "objective": "cutwidth"},
    "bw": {"prefix": "bw_run_", "suffix": "", "objective": "bandwidth"},
}


def _window(st: Dict) -> tuple:
    """(lb, ub) mirroring State.window: lb = 1 + largest proven-infeasible k."""
    lb = st.get("math_lb") or 1
    if st.get("unsat"):
        lb = max(lb, 1 + max(int(k) for k in st["unsat"]))
    return lb, st.get("ub")


def _state_path(run_dir: str, cluster: str, suffix: str) -> Optional[str]:
    exact = os.path.join(run_dir, f"{cluster}{suffix}.json")
    if os.path.isfile(exact):
        return exact
    cands = sorted(glob.glob(os.path.join(run_dir, "*.json")))
    return cands[0] if cands else None


def gather(data_dir: str, mode: str) -> List[Dict]:
    m = MODES[mode]
    out: List[Dict] = []
    for run_dir in sorted(glob.glob(os.path.join(data_dir, m["prefix"] + "*"))):
        if not os.path.isdir(run_dir):
            continue
        cluster = os.path.basename(run_dir)[len(m["prefix"]):]
        jf = _state_path(run_dir, cluster, m["suffix"])
        if jf is None:
            print(f"[skip] {cluster}: no state JSON in {run_dir}", file=sys.stderr)
            continue
        with open(jf) as f:
            st = json.load(f)
        lab = st.get("best_labeling")
        if not lab:
            print(f"[skip] {cluster}: state has no best_labeling yet",
                  file=sys.stderr)
            continue
        n = len(lab)
        perm0 = {v: lab[v] - 1 for v in range(n)}          # site -> 0-based pos
        rec = {"cluster": cluster, "n": n, "perm0": perm0}
        if cluster in CLUSTER_EDGES:
            _verts, edges = normalize_edges(CLUSTER_EDGES[cluster])
            rec["stats"] = {
                "bandwidth": bandwidth_of(lab, edges),
                "cutwidth": cutwidth_of(lab, edges),
                # avg_range = average interaction range R = mean edge length
                # (MinLA cost / |E|). Serialized as "avg_range=" in the comment
                # stats; parsers (analyze_mpo_correlation, vmps_torch) also
                # accept the pre-rename "envelope=" key from old files.
                "avg_range": total_range(lab, edges) / len(edges),
            }
        else:
            rec["stats"] = None
        lb, ub = _window(st)
        rec["lb"], rec["ub"] = lb, ub
        rec["certified"] = ub is not None and lb is not None and lb >= ub
        out.append(rec)
    out.sort(key=lambda r: (r["n"], r["cluster"]))
    return out


def render(records: List[Dict], mode: str, data_dir: str) -> str:
    obj = MODES[mode]["objective"]
    lines = [
        f"# orderings optimized for the {obj}",
        f"# gathered by gather_permutations.py from {data_dir} (mode={mode})",
        "# stats are geometry-only (bandwidth/avg_range/cutwidth of the layout);",
        "# MPO bond-dimension stats come from DMRG and are not filled in here.",
        "CUSTOM_PERMUTATIONS = {",
    ]
    for i, r in enumerate(records):
        if i:
            lines.append("")
        s = r["stats"]
        if s:
            lines.append(f"\t# bandwidth={s['bandwidth']}, "
                         f"avg_range={s['avg_range']:.2f}, "
                         f"cutwidth={s['cutwidth']}")
        else:
            lines.append(f"\t# ({r['cluster']} not in cluster_edges.py; "
                         f"stats unavailable)")
        if r["certified"]:
            lines.append(f"\t# certified optimal {obj} = {r['ub']}")
        else:
            lines.append(f"\t# heuristic upper bound; certified interval "
                         f"[{r['lb']}, {r['ub']}] (not proven optimal)")
        body = ", ".join(f"{v}: {r['perm0'][v]}" for v in range(r["n"]))
        lines.append(f'\t"{r["cluster"]}": {{{body}}},')
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["cw", "bw"], default="cw",
                    help="cw = cutwidth runs (cw_run_*), bw = bandwidth runs "
                         "(bw_run_*)   (default: cw)")
    ap.add_argument("--data-dir",
                    default=os.path.join(os.path.expanduser("~"),
                                         "lattice_in_line_data"),
                    help="the lattice_in_line_data folder to scan")
    ap.add_argument("--out", default=None,
                    help="output .py file (default: stdout)")
    args = ap.parse_args()

    if not os.path.isdir(args.data_dir):
        ap.error(f"data dir not found: {args.data_dir}")
    records = gather(args.data_dir, args.mode)
    if not records:
        ap.error(f"no {args.mode} runs with a saved layout found under "
                 f"{args.data_dir}")
    text = render(records, args.mode, args.data_dir)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"[gather] wrote {len(records)} entrie(s) to {args.out}",
              file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
