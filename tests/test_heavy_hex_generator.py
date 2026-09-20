import pytest

from lattice_in_line.cluster_edges import CLUSTER_EDGES, CLUSTER_EDGE_LAYERS
from lattice_in_line.heavy_hex_generator import (
    KNOWN_PATCHES,
    edge_layers,
    heavy_hex_patch,
    main,
    patch_for_cluster,
    validate_layers,
)


@pytest.mark.parametrize("name", sorted(KNOWN_PATCHES))
def test_known_patches_reproduce_cluster_edges_exactly(name):
    patch = patch_for_cluster(name)
    reference = {tuple(sorted(edge)) for edge in CLUSTER_EDGES[name]}
    generated = {tuple(sorted(edge)) for edge in patch.edges}
    assert generated == reference
    assert patch.n_sites == max(max(edge) for edge in reference) + 1
    assert patch.name == name


@pytest.mark.parametrize("name", sorted(KNOWN_PATCHES))
def test_known_patches_return_valid_edge_layers(name):
    patch = patch_for_cluster(name)
    layers = edge_layers(patch)
    validate_layers(patch, layers)
    if name in CLUSTER_EDGE_LAYERS:
        stored = [
            sorted(tuple(sorted(edge)) for edge in layer)
            for layer in CLUSTER_EDGE_LAYERS[name]
        ]
        assert [sorted(layer) for layer in layers] == stored


@pytest.mark.parametrize(
    "rows, cols, top_short, bottom_short",
    [(9, 9, True, False), (6, 13, True, True), (5, 9, False, True), (12, 9, True, False)],
)
def test_generated_layers_are_valid_edge_colorings(rows, cols, top_short, bottom_short):
    patch = heavy_hex_patch(rows, cols=cols, top_short=top_short, bottom_short=bottom_short)
    layers = edge_layers(patch)
    assert len(layers) == 3
    validate_layers(patch, layers)


def test_generated_patch_grows_by_full_row_period():
    five = patch_for_cluster("heavyHex51")
    seven = patch_for_cluster("heavyHex74")
    nine = heavy_hex_patch(9, top_short=True, bottom_short=False)
    assert seven.n_sites - five.n_sites == nine.n_sites - seven.n_sites == 23


def test_row_and_connector_coordinates_form_heavy_hex_pattern():
    patch = patch_for_cluster("heavyHex35")
    # connectors sit on odd y between the even-y rows they join
    for (a, b) in patch.edges:
        ya, yb = patch.coords[a][1], patch.coords[b][1]
        if patch.kinds[a] == "conn" or patch.kinds[b] == "conn":
            assert abs(ya - yb) == 1.0
        else:
            assert ya == yb


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError, match="two short rows"):
        heavy_hex_patch(2, top_short=True, bottom_short=True)
    with pytest.raises(ValueError, match="4\\*k"):
        heavy_hex_patch(3, cols=8)
    with pytest.raises(ValueError, match="at least two rows"):
        heavy_hex_patch(1)
    with pytest.raises(ValueError, match="bottom_short is inconsistent"):
        heavy_hex_patch(4, top_short=False, bottom_short=True)
    with pytest.raises(ValueError, match="bottom_short is inconsistent"):
        heavy_hex_patch(5, top_short=True, bottom_short=True)


def test_cli_smoke_known_and_generated(capsys):
    assert main(["heavyHex28", "--plot", "none", "--print-edges"]) == 0
    output = capsys.readouterr().out
    assert "heavyHex28" in output and "layers: stored" in output
    assert main(["--rows", "5", "--plot", "none"]) == 0
    assert "layers: stored" in capsys.readouterr().out  # 5 rows, cols 9 == heavyHex51
    assert main(["--rows", "9", "--plot", "none"]) == 0
    assert "layers: generated" in capsys.readouterr().out


def test_cli_prints_coords_and_edges_like_cluster_generator(capsys):
    assert main(["heavyHex28", "--plot", "none"]) == 0
    output = capsys.readouterr().out
    assert "# n_sites = 28" in output
    assert "coords = [" in output and "edges = [" in output
    assert "(0, 0, 0)" in output  # top-left row qubit, no negative zero
    assert "(0, -1, 0)" in output  # first connector qubit

    assert main(["heavyHex28", "--plot", "none", "--format", "edgelist"]) == 0
    lines = capsys.readouterr().out.splitlines()
    pair_lines = [line for line in lines if line and not line.startswith(("#", "heavyHex"))]
    assert pair_lines[0] == "0 1" and len(pair_lines) == 30

    assert main(["heavyHex28", "--plot", "none", "--format", "json"]) == 0
    import json

    payload = json.loads(capsys.readouterr().out.split("\n", 1)[1])
    assert payload["n_sites"] == 28
    assert len(payload["edges"]) == 30
    assert len(payload["edge_layers"]) == 3
