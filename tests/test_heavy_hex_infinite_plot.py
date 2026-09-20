import pytest

from lattice_in_line.heavy_hex_infinite_plot import (
    CELL_PERIOD,
    NATURAL_COORDS,
    main,
    plot_infinite_ladder,
    site_position,
)
from lattice_in_line.unit_cell.heavy_hex_ladder import NATURAL_EDGES, UNIT_CELL_SITES


def test_every_natural_site_has_a_coordinate():
    assert len(NATURAL_COORDS) == UNIT_CELL_SITES
    assert len(set(NATURAL_COORDS)) == UNIT_CELL_SITES


def test_coordinates_realize_every_bond_as_a_unit_step():
    """Rail bonds join neighbouring columns on one line; connector bonds join
    the same column on neighbouring lines -- also across the cell boundary."""
    for i, j, offset in NATURAL_EDGES:
        (xi, yi), (xj, yj) = site_position(0, i), site_position(offset, j)
        dx, dy = abs(xi - xj), abs(yi - yj)
        assert (dx, dy) in {(1.0, 0.0), (0.0, 1.0)}, (i, j, offset)
        if offset:
            assert dy == 1.0 and yj == CELL_PERIOD  # enters the next copy's top line


def test_connector_columns_alternate_between_rails():
    top = sorted(NATURAL_COORDS[s][0] for s in (0, 1))
    middle = sorted(NATURAL_COORDS[s][0] for s in (11, 12, 13))
    assert top == [2, 6] and middle == [0, 4, 8]


@pytest.mark.parametrize("cells, horizontal", [(3, False), (1, False), (4, True)])
def test_plot_smoke(tmp_path, cells, horizontal):
    pytest.importorskip("matplotlib")
    png, pdf = plot_infinite_ladder(
        cells=cells, horizontal=horizontal, output_stem=str(tmp_path / "ladder")
    )
    assert png.is_file() and pdf.is_file()


def test_cli(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    assert main(["--cells", "3", "--ordering", "natural", "--no-layers",
                 "--output-stem", str(tmp_path / "cli")]) == 0
    assert "wrote" in capsys.readouterr().out
