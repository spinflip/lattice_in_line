import pytest

from lattice_in_line.unit_cell.heavy_hex_ladder import (
    NATURAL_EDGES,
    NATURAL_EDGE_LAYERS,
    UNIT_CELL_SITES,
    build,
)
from lattice_in_line.cluster_edges import CLUSTER_EDGES, CLUSTER_EDGE_LAYERS


def _triples(payload):
    edges = [(i, j, 0) for i, j in payload["intra_cell"]["1.0"]]
    edges += [(i, j, 1) for i, j in payload["inter_cell"]["1.0"]]
    return edges


def _layer_triples(payload, layer):
    edges = [(i, j, 0) for i, j in layer["intra_cell"]]
    edges += [(i, j, 1) for i, j in layer["inter_cell"]]
    return edges


def _metrics(edges, length=UNIT_CELL_SITES):
    ranges = [j + offset * length - i for i, j, offset in edges]
    cuts = []
    for cut in range(length):
        cuts.append(sum(
            1
            for i, j, offset in edges
            for shift in (-length, 0, length)
            if i + shift < cut + 0.5 < j + offset * length + shift
        ))
    return max(ranges), sum(ranges) / len(ranges), max(cuts)


def test_heavy_hex_infinite_geometry_and_layers():
    payload = build(ordering="natural")
    assert payload["unit_cell_sites"] == 23
    assert len(NATURAL_EDGES) == 26
    assert sorted(edge for layer in NATURAL_EDGE_LAYERS for edge in layer) == sorted(NATURAL_EDGES)
    assert [len(layer) for layer in NATURAL_EDGE_LAYERS] == [8, 9, 9]

    degree = [0] * UNIT_CELL_SITES
    for layer in NATURAL_EDGE_LAYERS:
        matched = set()
        for i, j, _ in layer:
            assert i not in matched
            assert j not in matched
            matched.update((i, j))
    for i, j, _ in NATURAL_EDGES:
        degree[i] += 1
        degree[j] += 1
    assert degree.count(2) == 17
    assert degree.count(3) == 6


def test_heavy_hex_optimized_periodic_metrics_and_bulk_repeat():
    payload = build(ordering="optimized")
    edges = _triples(payload)
    assert _metrics(edges) == (4, 75 / 26, 4)
    assert sorted(edge for layer in payload["edge_layers"] for edge in _layer_triples(payload, layer)) == sorted(edges)

    # The 51- and 74-site patches consist of a five-site cap followed by two
    # and three copies of the natural 23-site motif, respectively.
    natural = set(NATURAL_EDGES)
    for copies in (2, 3):
        expanded = set()
        for cell in range(copies):
            for i, j, offset in natural:
                if cell + offset < copies:
                    expanded.add((5 + cell * 23 + i, 5 + (cell + offset) * 23 + j))
        assert len(expanded) == copies * 24 + (copies - 1) * 2
        cluster_name = "heavyHex51" if copies == 2 else "heavyHex74"
        finite_bulk = {
            tuple(sorted(edge))
            for edge in CLUSTER_EDGES[cluster_name]
            if min(edge) >= 5
        }
        assert {tuple(sorted(edge)) for edge in expanded} == finite_bulk


def test_periodic_coloring_has_documented_heavy_hex74_bulk_agreement():
    finite_colors = {
        tuple(sorted(edge)): color
        for color, layer in enumerate(CLUSTER_EDGE_LAYERS["heavyHex74"])
        for edge in layer
    }
    agreement = 0
    for color, layer in enumerate(NATURAL_EDGE_LAYERS):
        for i, j, offset in layer:
            for base in (28, 51):
                actual = (
                    (base + i, base + j)
                    if offset == 0
                    else (base - UNIT_CELL_SITES + i, base + j)
                )
                agreement += finite_colors[tuple(sorted(actual))] == color
    assert agreement == 35


@pytest.mark.parametrize("width,clusters,metrics", [
    (2, ("heavyHex75", "heavyHex108"), (6, 144 / 38, 5)),
    (3, ("heavyHex99", "heavyHex142"), (10, 226 / 50, 6)),
])
def test_wider_periodic_cells_reconstruct_finite_bulk(width, clusters, metrics):
    natural = build(ordering="natural", width=width)
    optimized = build(width=width)
    length = 10 * width + 13
    assert natural["unit_cell_sites"] == optimized["unit_cell_sites"] == length
    edges = _triples(natural)
    assert len(edges) == 12 * width + 14
    assert len(natural["inter_cell"]["1.0"]) == width + 1
    degree = [sum(site in (i, j) for i, j, _ in edges) for site in range(length)]
    assert degree.count(2) == 6 * width + 11
    assert degree.count(3) == 4 * width + 2
    assert set(degree) == {2, 3}
    assert _metrics(_triples(optimized), length) == metrics
    permutation = {int(k): v for k, v in optimized["permutation"].items()}
    assert sorted(permutation.values()) == list(range(length))
    relabeled = [(min(permutation[i], permutation[j]), max(permutation[i], permutation[j]), 0)
                 if offset == 0 else (permutation[i], permutation[j], offset)
                 for i, j, offset in edges]
    assert sorted(relabeled) == sorted(_triples(optimized))

    for payload in (natural, optimized):
        layers = [_layer_triples(payload, layer) for layer in payload["edge_layers"]]
        assert [len(layer) for layer in layers] == [4 * width + 4, 4 * width + 5, 4 * width + 5]
        assert sorted(edge for layer in layers for edge in layer) == sorted(_triples(payload))
        for layer in layers:
            endpoints = [site for cell in range(-1, 2) for i, j, offset in layer
                         for site in (cell * length + i, (cell + offset) * length + j)]
            assert len(endpoints) == len(set(endpoints))
    cap = 4 * width + 1
    for copies, cluster in zip((2, 3), clusters):
        expanded = {tuple(sorted((cap + cell * length + i, cap + (cell + offset) * length + j)))
                    for cell in range(copies) for i, j, offset in edges if cell + offset < copies}
        bulk = {tuple(sorted(edge)) for edge in CLUSTER_EDGES[cluster] if min(edge) >= cap}
        assert expanded == bulk


def test_invalid_periodic_width_is_rejected():
    with pytest.raises(ValueError, match="width"):
        build(width=0)
