#!/usr/bin/env python3
"""dmrg_log_to_json.py — convert a DMRG run .log into the results-JSON schema
read by fiedler_ordering.py (the same shape as the model=...json files).

It extracts, per sweep, the bond dimension (Mmax), the energy E and E/L, and the
spin-spin correlation block — lines of the form

    i=0, j=1, d=1, SdagS=-0.2849518941

(optionally grouped by 'correlations at distance d=N' headers). The correlation
block is attached to the LOWEST-energy sweep, since correlations are measured on
the final, best-converged state.

Output JSON:
  {"runs": {"run_0001": {"ground_state": {"energy":.., "energy_per_site":..,
     "bond_space": {"Mmax":..}}, "correlations": {"values":
     [{"i":.., "j":.., "distance":.., "SdagS":..}, ...]}}, ...},
   "metadata": {"source": "<logfile>"}}

Usage:
  python dmrg_log_to_json.py RUN.log [-o OUT.json]   # default OUT = RUN.json
then:
  python fiedler_ordering.py OUT.json --refine --out order.txt
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

# An energy line: "E     = -31.131..." or "E = -31.131..." (not "E/L =").
E_RE = re.compile(r"^\s*E\s*=\s*([-\d.eE+]+)")
EL_RE = re.compile(r"^\s*E/L\s*=\s*([-\d.eE+]+)")
# A standalone Mmax line: "Mmax  = 5000" (NOT the "Bond space : Mmax = .." lines).
MMAX_RE = re.compile(r"^\s*Mmax\s*=\s*(\d+)\b")
# A correlation entry, tolerant of spacing.
CORR_RE = re.compile(
    r"i\s*=\s*(\d+)\s*,\s*j\s*=\s*(\d+)\s*,\s*d\s*=\s*(-?\d+)\s*,\s*SdagS\s*=\s*([-\d.eE+]+)"
)


def parse_log(path):
    """Return (sweeps, corr): sweeps = [(Mmax, E, E/L)] in file order;
    corr = {(i,j): (distance, SdagS)} over unordered pairs."""
    e = el = None
    sweeps = []
    corr = {}
    with open(path) as f:
        for line in f:
            m = E_RE.match(line)
            if m:
                e = float(m.group(1))
                continue
            m = EL_RE.match(line)
            if m:
                el = float(m.group(1))
                continue
            m = MMAX_RE.match(line)
            if m and e is not None:
                sweeps.append((int(m.group(1)), e, el))
                continue
            m = CORR_RE.search(line)
            if m:
                i, j = int(m.group(1)), int(m.group(2))
                corr[(min(i, j), max(i, j))] = (int(m.group(3)), float(m.group(4)))
    return sweeps, corr


def build_payload(sweeps, corr, source):
    # de-duplicate sweeps (same Mmax + energy can be reported twice), keep order
    seen, runs = set(), []
    for mmax, e, el in sweeps:
        key = (mmax, round(e, 9))
        if key not in seen:
            seen.add(key)
            runs.append((mmax, e, el))
    if not runs:
        # no Mmax-paired energy found; still emit one run so the correlations
        # are usable (energy unknown -> null)
        runs = [(None, None, None)]

    values = [
        {"i": i, "j": j, "distance": d, "SdagS": v}
        for (i, j), (d, v) in sorted(corr.items())
    ]
    # correlations belong to the lowest-energy sweep (the final converged state)
    energies = [e for _, e, _ in runs]
    imin = min(
        range(len(runs)),
        key=lambda k: (energies[k] is None, energies[k] if energies[k] is not None else 0.0),
    )

    out = {"runs": {}, "metadata": {"source": os.path.basename(source)}}
    for k, (mmax, e, el) in enumerate(runs, start=1):
        gs = {"energy": e, "energy_per_site": el}
        if mmax is not None:
            gs["bond_space"] = {"Mmax": mmax}
        run = {"ground_state": gs}
        if k - 1 == imin and values:
            run["correlations"] = {"values": values}
        out["runs"][f"run_{k:04d}"] = run
    return out, runs, imin, len(values)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("log", help="DMRG run .log file")
    ap.add_argument("-o", "--out", default=None,
                    help="output JSON (default: the log path with .json)")
    args = ap.parse_args()

    sweeps, corr = parse_log(args.log)
    if not corr:
        sys.exit(f"{args.log}: found no 'i=.., j=.., d=.., SdagS=..' correlation "
                 "lines — nothing for fiedler_ordering.py to use")
    payload, runs, imin, nvals = build_payload(sweeps, corr, args.log)

    out = args.out or (os.path.splitext(args.log)[0] + ".json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=1)

    n = 1 + max(max(p) for p in corr)
    print(f"parsed {len(runs)} sweep(s); correlations: {nvals} pairs on {n} sites")
    for k, (mmax, e, el) in enumerate(runs):
        tag = "  <- correlations (lowest energy)" if k == imin else ""
        print(f"  run_{k+1:04d}: Mmax={mmax} E={e} E/L={el}{tag}")
    print(f"wrote {out}")
    print(f"next: python3 fiedler_ordering.py {out} --refine --out order.txt")


if __name__ == "__main__":
    main()
