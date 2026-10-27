"""Exports of --edge-file graphs must use the file's own vertex labels.

load_edge_file re-indexes vertices by order of first appearance. If an export
prints those internal indices, applying the map to the caller's own sites
silently scrambles the lattice: on a path written in shuffled line order the
certified cutwidth 1 layout degrades to 7.

The fixture labels sites 100..111 so that the internal indices (0..11) differ
from the labels as a *set*, not merely as a permutation -- otherwise an export
keyed by internal index passes a set comparison by accident.
"""
import ast
import subprocess
import sys

from lattice_in_line.bandwidth_heuristics_benchmark import compute_cutwidth
from lattice_in_line.certify.core import edge_file_labels, load_edge_file

OFFSET = 100
# A 12-site path 100-101-...-111 whose line order scrambles first appearance.
SHUFFLED_PATH = [(1, 2), (10, 11), (3, 4), (9, 10), (8, 9), (4, 5),
                 (7, 8), (0, 1), (6, 7), (2, 3), (5, 6)]
SHUFFLED_PATH = [(u + OFFSET, v + OFFSET) for u, v in SHUFFLED_PATH]
SITES = {u for e in SHUFFLED_PATH for u in e}


def _write(tmp_path):
    p = tmp_path / "edges.txt"
    p.write_text("".join(f"{u} {v}\n" for u, v in SHUFFLED_PATH))
    return p


def _run(args, state_dir, path):
    return subprocess.run(
        [sys.executable, "-m", "lattice_in_line.bandwidth_certifier", *args,
         "--edge-file", str(path), "--state-dir", str(state_dir)],
        capture_output=True, text=True, check=True).stdout


def _mapping(out):
    return ast.literal_eval(next(ln for ln in out.splitlines() if ln.startswith("{")))


def test_labels_match_the_internal_indices(tmp_path):
    path = _write(tmp_path)
    n, edges = load_edge_file(str(path))
    labels = edge_file_labels(str(path))
    assert len(labels) == n
    pos = {w: i for i, w in enumerate(labels)}
    rebuilt = sorted({(min(pos[str(u)], pos[str(v)]), max(pos[str(u)], pos[str(v)]))
                      for u, v in SHUFFLED_PATH})
    assert rebuilt == edges


def test_cw_export_is_usable_on_the_original_labels(tmp_path):
    path = _write(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    _run(["cw-run", "--heur-time", "2", "--time-per-k", "5", "--ladder-time", "8",
          "--procs", "2", "--workers", "2"], state, path)
    perm = _mapping(_run(["cw-export"], state, path))

    assert set(perm) == SITES, "keys must be the file's labels, not internal indices"
    # the layout must achieve the certified optimum on the real lattice
    order = [v - OFFSET for v in sorted(perm, key=lambda v: perm[v])]
    assert compute_cutwidth([(i, i + 1) for i in range(11)], order) == 1


def test_every_affected_export_keys_the_layout_by_the_files_labels(tmp_path):
    """export, h2-export and ss-export take their graph from get_graph, so they
    share the defect. w-export is excluded: its graph comes from --jmatrix,
    which is indexed 0..n-1 already."""
    path = _write(tmp_path)
    state = tmp_path / "state"
    state.mkdir()

    _run(["heuristic", "--time", "2", "--procs", "2"], state, path)
    assert set(_mapping(_run(["export"], state, path))) == SITES, "export"

    _run(["h2-run", "--heur-time", "2", "--time-per-k", "5", "--workers", "2"],
         state, path)
    _run(["ss-run", "--block", "2", "--heur-time", "2", "--time-per-k", "5",
          "--workers", "2", "--procs", "2"], state, path)
    assert set(_mapping(_run(["h2-export"], state, path))) == SITES, "h2-export"

    out = _run(["ss-export", "--block", "2"], state, path)
    assert set(_mapping(out)) == SITES, "ss-export"
    blocks = ast.literal_eval(
        next(ln for ln in out.splitlines() if ln.startswith("blocks in chain order:"))
        .split(":", 1)[1].strip())
    assert {v for b in blocks for v in b} == SITES, "ss-export block listing"
