"""cluster_edges_NNN.py invariants: every NNN table shares numbering with its
J1 (cluster_edges.py) entry, stays in range, and molecular clusters are absent.

The NNN table is a generated artifact (vmps-build-nnn-tables) and is not
shipped; these invariant checks run only when a generated table is present."""
import pytest

import lattice_in_line.build_cluster_edges_NNN as builder
from lattice_in_line.cluster_edges import CLUSTER_EDGES as NN

NNN = pytest.importorskip(
    "lattice_in_line.cluster_edges_NNN",
    reason="cluster_edges_NNN.py not generated (run vmps-build-nnn-tables)",
).CLUSTER_EDGES


def _canon(edges):
    return sorted((min(u, v), max(u, v)) for u, v in edges if u != v)


def test_nnn_tables_nonempty():
    assert len(NNN) > 0


def test_nnn_numbering_matches_j1_and_in_range():
    for name in NNN:
        assert name in NN, f"{name} has no J1 table"
        recipe = builder.recipe(name)
        assert recipe is not None, f"{name} unexpectedly has no generator recipe"
        # Regenerate the NN graph the same way the NNN table was built; the
        # numbering MUST equal cluster_edges.py or J1/J2 wouldn't be a valid pair.
        _, nn_edges, _ = builder.coords_edges_periods(*recipe)
        assert builder.canon(nn_edges) == _canon(NN[name]), name
        # NNN vertices stay within the J1 site count.
        n = 1 + max(max(e) for e in NN[name])
        assert all(0 <= i < n and 0 <= j < n for i, j in NNN[name]), name


def test_molecular_clusters_have_no_nnn():
    for mol in ("C12", "C60", "icosa", "cubocta"):
        assert mol in NN
        assert mol not in NNN
