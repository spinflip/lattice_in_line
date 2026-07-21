#!/usr/bin/env python3
"""Kagome XC strip unit-cell geometry."""
from __future__ import annotations

import argparse
import json

aliases = {
    "t": "J",
}


def _append_edge_map(target, coupling, edge):
    key = str(float(coupling))
    target.setdefault(key, []).append([int(edge[0]), int(edge[1])])


def _base(y):
    return 6 * y


def _a0(y):
    return _base(y)


def _a1(y):
    return _base(y) + 1


def _b(y):
    return _base(y) + 2


def _c0(y):
    return _base(y) + 3


def _c1(y):
    return _base(y) + 4


def _d(y):
    return _base(y) + 5


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
    if Ly < 4 or Ly % 4 != 0:
        raise ValueError(f"Ly must be a multiple of 4, got {Ly}")
    n_y = Ly // 4
    payload = {
        "unit_cell_sites": 6 * n_y,
        "intra_cell": {},
        "inter_cell": {},
    }

    for y in range(n_y):
        yp = (y + 1) % n_y
        _append_edge_map(payload["intra_cell"], J, (_a0(y), _a1(y)))
        _append_edge_map(payload["intra_cell"], J, (_c0(y), _c1(y)))
        _append_edge_map(payload["intra_cell"], J, (_a0(y), _b(y)))
        _append_edge_map(payload["intra_cell"], J, (_b(y), _c0(y)))
        _append_edge_map(payload["intra_cell"], J, (_c0(y), _d(y)))
        _append_edge_map(payload["intra_cell"], J, (_c1(y), _d(y)))
        _append_edge_map(payload["intra_cell"], J, (_d(y), _a0(yp)))
        _append_edge_map(payload["intra_cell"], J, (_d(y), _a1(yp)))

        _append_edge_map(payload["inter_cell"], J, (_a1(y), _a0(y)))
        _append_edge_map(payload["inter_cell"], J, (_c1(y), _c0(y)))
        _append_edge_map(payload["inter_cell"], J, (_a1(y), _b(y)))
        _append_edge_map(payload["inter_cell"], J, (_c1(y), _b(y)))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute the kagome XC unit-cell geometry.")
    parser.add_argument("--Ly", type=int, default=4)
    parser.add_argument("--J", "--t", dest="J", type=float, default=1.0)
    args = parser.parse_args()
    print(json.dumps(build(Ly=args.Ly, J=args.J), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
