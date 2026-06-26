#!/usr/bin/env python3
"""Triangular XC4 strip unit-cell geometry."""
from __future__ import annotations

import argparse
import json

aliases = {
    "t": "J",
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


def build(Ly=4, J=1.0, **kwargs):
    params = {"Ly": Ly, "J": J}
    _apply_aliases(kwargs, params)
    Ly = params["Ly"]
    J = params["J"]
    Ly = int(Ly)
    if Ly < 2:
        raise ValueError(f"Ly must be at least 2, got {Ly}")
    payload = {
        "unit_cell_sites": Ly,
        "intra_cell": {},
        "inter_cell": {},
        "inter_cell_2": {},
    }
    for a in range(Ly):
        prev = (a - 1) % Ly
        _append_edge_map(payload["inter_cell"], J, (a, a))
        _append_edge_map(payload["inter_cell"], J, (a, prev))
        _append_edge_map(payload["inter_cell_2"], J, (a, prev))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute the triangular XC4 unit-cell geometry.")
    parser.add_argument("--Ly", type=int, default=4)
    parser.add_argument("--J", "--t", dest="J", type=float, default=1.0)
    args = parser.parse_args()
    print(json.dumps(build(Ly=args.Ly, J=args.J), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
