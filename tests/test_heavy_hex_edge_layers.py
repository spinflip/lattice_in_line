from lattice_in_line.cluster_edges import CLUSTER_EDGES
from lattice_in_line.cluster_graphs import cluster_edge_layers, cluster_ordering_permutation


EXPECTED_LAYER_SIZES = {
    "heavyHex28": (9, 11, 10),
    "heavyHex35": (13, 12, 13),
    "heavyHex51": (19, 19, 18),
    "heavyHex74": (28, 27, 27),
    "heavyHex75": (31, 31, 22),
    "heavyHex99": (41, 41, 30),
    "heavyHex108": (44, 44, 34),
    "heavyHex142": (58, 58, 46),
    "heavyHex185": (75, 75, 62),
    "heavyHex399": (161, 161, 142),
}


def _normalized(edges):
    return {tuple(sorted(edge)) for edge in edges}


def test_heavy_hex_edge_layers_cover_each_graph_as_matchings():
    for cluster, sizes in EXPECTED_LAYER_SIZES.items():
        layers = cluster_edge_layers(cluster, ordering="identity")
        assert tuple(map(len, layers)) == sizes
        assert sum(map(len, layers)) == len(CLUSTER_EDGES[cluster])
        assert _normalized(edge for layer in layers for edge in layer) == _normalized(
            CLUSTER_EDGES[cluster]
        )
        for layer in layers:
            flattened = [site for edge in layer for site in edge]
            assert len(flattened) == len(set(flattened))


def test_heavy_hex_edge_layers_use_original_to_mps_permutation():
    for cluster in EXPECTED_LAYER_SIZES:
        raw = cluster_edge_layers(cluster, ordering="identity")
        permutation = cluster_ordering_permutation(
            cluster, ordering="permutations_sat_cw.py"
        )
        expected = [
            sorted(tuple(sorted((permutation[i], permutation[j]))) for i, j in layer)
            for layer in raw
        ]
        assert cluster_edge_layers(
            cluster, ordering="permutations_sat_cw.py"
        ) == expected
