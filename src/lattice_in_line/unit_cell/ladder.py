#!/usr/bin/env python3
"""Ladder unit-cell geometry."""
from __future__ import annotations

import argparse
import json

aliases = {
    "tperp": "Jperp",
    "tpara": "Jpara",
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


def build(Jperp=1.0, Jpara=1.0, Ly=2, **kwargs):
    params = {"Jperp": Jperp, "Jpara": Jpara, "Ly": Ly}
    _apply_aliases(kwargs, params)
    Jperp = params["Jperp"]
    Jpara = params["Jpara"]
    Ly = params["Ly"]
    Ly = int(Ly)
    if Ly < 2:
        raise ValueError(f"Ly must be at least 2, got {Ly}")
    payload = {
        "unit_cell_sites": Ly,
        "intra_cell": {},
        "inter_cell": {},
    }
    for y in range(Ly - 1):
        _append_edge_map(payload["intra_cell"], Jperp, (y, y + 1))
    for y in range(Ly):
        _append_edge_map(payload["inter_cell"], Jpara, (y, y))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute the ladder unit-cell geometry.")
    parser.add_argument("--Jperp", "--tperp", dest="Jperp", type=float, default=1.0)
    parser.add_argument("--Jpara", "--tpara", dest="Jpara", type=float, default=1.0)
    parser.add_argument("--Ly", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(build(Jperp=args.Jperp, Jpara=args.Jpara, Ly=args.Ly), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
