#!/usr/bin/env python3
"""Heptagon-pentagon (5/7 skewed) ladder unit-cell geometry.

The geometry follows the 5/7 skewed ladder studied in
Jana, Dey, Kumar, Ramasesha, and Raghunathan, arXiv:2507.09502
("Electronic and magnetic ground states of {112} grain boundary in
graphene in the extended Hubbard model").

The unit cell has eight sites.  Using 0-indexed labels (paper labels
are off by +1), the lower leg is formed by sites 0, 2, 4, 6 (paper
1, 3, 5, 7) and the upper leg by sites 1, 3, 5, 7 (paper 2, 4, 6, 8).
Two rung bonds, (0, 1) and (5, 6) (paper 1-2 and 6-7), connect the
legs.  Together with the leg bonds these produce the alternating
five- and seven-membered rings that give the ladder its name.

Hamiltonian terms (cf. eq. (1) of the paper):

    H = -t  Sum_{<i,j>}  (c^dag_i c_j + h.c.)
        + U Sum_i n_{i,up} n_{i,down}
        + V Sum_{<i,j>}  n_i n_j

where <i,j> runs over all leg and rung bonds.  Here we expose the two
hopping amplitudes separately as ``Jperp`` (rungs) and ``Jpara``
(legs) -- matching the convention of ``ladder.py`` -- so that the
caller can break the equal-hopping assumption when desired.
"""
from __future__ import annotations

import argparse
import json


aliases = {
    "tperp": "Jperp",
    "tpara": "Jpara",
    "trung": "Jperp",
    "tleg": "Jpara",
    "t_rung": "Jperp",
    "t_leg": "Jpara",
}


# Number of sites in a single unit cell of the 5/7 skewed ladder.
UNIT_CELL_SITES = 8

# Intra-cell rung bonds (paper labels 1-2 and 6-7).
INTRA_RUNG_BONDS = (
    (0, 1),  # paper 1-2
    (5, 6),  # paper 6-7
)

# Intra-cell leg bonds: pairs within a leg differing by 2 in paper
# numbering.  Lower leg first, then upper leg.
INTRA_LEG_BONDS = (
    (0, 2),  # lower leg, paper 1-3
    (2, 4),  # lower leg, paper 3-5
    (4, 6),  # lower leg, paper 5-7
    (1, 3),  # upper leg, paper 2-4
    (3, 5),  # upper leg, paper 4-6
    (5, 7),  # upper leg, paper 6-8
)

# Inter-cell leg bonds.  Each entry (src, dst) means: site ``src`` in
# unit cell i is connected to site ``dst`` in unit cell i+1.  These
# correspond to paper sites 7 -> 9 (= 1 of next cell) and 8 -> 10
# (= 2 of next cell).
INTER_LEG_BONDS = (
    (6, 0),  # lower leg, paper 7 -> 1 of next cell
    (7, 1),  # upper leg, paper 8 -> 2 of next cell
)


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


def build(Jperp=1.0, Jpara=1.0, **kwargs):
    """Return the unit-cell geometry of the 5/7 skewed ladder.

    Parameters
    ----------
    Jperp : float
        Coupling on the two rung bonds (paper labels 1-2 and 6-7).
        Aliases: ``tperp``, ``trung``, ``t_rung``.
    Jpara : float
        Coupling on the leg bonds (all pairs differing by 2 in paper
        labelling, both intra- and inter-cell).
        Aliases: ``tpara``, ``tleg``, ``t_leg``.

    Returns
    -------
    dict
        ``unit_cell_sites`` -- number of sites in the unit cell (8).
        ``intra_cell`` -- maps each coupling value (as a string key)
        to the list of [i, j] site pairs inside the unit cell.
        ``inter_cell`` -- maps each coupling value to the list of
        [src, dst] pairs connecting cell i (site ``src``) to cell
        i+1 (site ``dst``).
    """
    params = {"Jperp": Jperp, "Jpara": Jpara}
    _apply_aliases(kwargs, params)
    Jperp = params["Jperp"]
    Jpara = params["Jpara"]
    payload = {
        "unit_cell_sites": UNIT_CELL_SITES,
        "intra_cell": {},
        "inter_cell": {},
    }
    for edge in INTRA_RUNG_BONDS:
        _append_edge_map(payload["intra_cell"], Jperp, edge)
    for edge in INTRA_LEG_BONDS:
        _append_edge_map(payload["intra_cell"], Jpara, edge)
    for edge in INTER_LEG_BONDS:
        _append_edge_map(payload["inter_cell"], Jpara, edge)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the heptagon-pentagon (5/7 skewed) ladder "
            "unit-cell geometry."
        )
    )
    parser.add_argument(
        "--Jperp", "--tperp", "--trung", "--t_rung",
        dest="Jperp", type=float, default=1.0,
        help="Hopping/coupling on the rung bonds (paper 1-2 and 6-7).",
    )
    parser.add_argument(
        "--Jpara", "--tpara", "--tleg", "--t_leg",
        dest="Jpara", type=float, default=1.0,
        help="Hopping/coupling on the leg bonds (paper sites differing by 2).",
    )
    args = parser.parse_args()
    print(json.dumps(
        build(Jperp=args.Jperp, Jpara=args.Jpara),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()