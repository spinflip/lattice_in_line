"""Edge-file format contract: the --j2-file loader and the generator's
edgelist output round-trip consistently."""
import pytest

import lattice_in_line.bandwidth_certifier as bc
import lattice_in_line.cluster_generator as cg


def test_load_edge_file_indexed_parsing(tmp_path):
    p = tmp_path / "edges.txt"
    p.write_text("# a comment\n0 1\n1 2\n2 1\n3 3\n% another\n")
    edges = bc.load_edge_file_indexed(str(p), 4)
    # comments skipped, self-loop (3,3) dropped, (2,1) canonicalised & deduped
    assert edges == [(0, 1), (1, 2)]


def test_load_edge_file_indexed_rejects_out_of_range(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("0 5\n")  # vertex 5 >= n=4
    with pytest.raises(SystemExit):
        bc.load_edge_file_indexed(str(p), 4)


def test_edgelist_roundtrips_to_generator_edges(tmp_path):
    lattice = cg.get_lattice("pyrochlore", 0.138)
    coords, edges = lattice.make_diagonal(2, 2, 2)
    n = len(coords)
    p = tmp_path / "nn.txt"
    p.write_text("# header\n" + "\n".join(f"{i} {j}" for i, j in edges) + "\n")
    parsed = bc.load_edge_file_indexed(str(p), n)
    assert set(parsed) == {(min(i, j), max(i, j)) for i, j in edges}
