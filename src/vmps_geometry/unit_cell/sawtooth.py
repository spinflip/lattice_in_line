#!/usr/bin/env python3
"""Sawtooth-chain unit-cell geometry."""
from __future__ import annotations

import argparse
import json

aliases = {
    "tAB": "JAB",
    "tBB": "JBB",
}


def _append_edge_map(target, coupling, edge):
    key = str(float(coupling))
    target.setdefault(key, []).append([int(edge[0]), int(edge[1])])


def _apply_aliases(kwargs, params):
    for alias, canonical in aliases.items():
        if alias not in kwargs:
            continue
        if canonical in kwargs:
            raise TypeError(f"Received both {canonical} and its alias {alias}")
        params[canonical] = kwargs.pop(alias)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected parameter(s): {unexpected}")


def build(JAB=1.0, JBB=0.5, **kwargs):
    params = {"JAB": JAB, "JBB": JBB}
    _apply_aliases(kwargs, params)
    JAB = params["JAB"]
    JBB = params["JBB"]
    payload = {
        "unit_cell_sites": 2,
        "intra_cell": {},
        "inter_cell": {},
    }
    _append_edge_map(payload["intra_cell"], JAB, (0, 1))
    _append_edge_map(payload["inter_cell"], JAB, (0, 1))
    _append_edge_map(payload["inter_cell"], JBB, (0, 0))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute the sawtooth-chain unit-cell geometry.")
    parser.add_argument("--JAB", "--tAB", dest="JAB", type=float, default=1.0)
    parser.add_argument("--JBB", "--tBB", dest="JBB", type=float, default=0.5)
    args = parser.parse_args()
    print(json.dumps(build(JAB=args.JAB, JBB=args.JBB), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
