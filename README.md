# vmps_geometry

Geometry, ordering, and bandwidth-certification toolkit for VMPS/DMRG frustrated-magnet
lattice models: cluster generation, nearest- and further-neighbour edge tables,
certified minimum-bandwidth campaigns (including a lexicographic J1→J2 mode),
correlation-aware (Fiedler) orderings, and a QUBO experiment track.

The import package is `vmps_geometry`. Only `numpy` is required at the core; every
heavier subsystem is an optional extra and is imported lazily.

## Installation

```bash
cd /path/to/vmps_geometry
pip install -e .            # core (numpy only)
pip install -e '.[all]'     # everything
```

Extras (declared in `pyproject.toml`):

| Extra | Pulls in | Enables |
|---|---|---|
| `plot` | matplotlib | lattice plots (otherwise skipped, coords/edges still emitted) |
| `certify` | ortools, python-sat | CP-SAT decisions + independent SAT cross-check |
| `orbits` | pynauty | orbit (Aut(G)) symmetry breaking (falls back to reversal if absent) |
| `qubo` | scipy | the experimental QUBO track (reverse Cuthill–McKee) |
| `dev` | pytest | test suite |

## Command-line tools

| Command | What it does |
|---|---|
| `vmps-cluster-generator` | generate a cluster's coordinates + neighbour graph |
| `vmps-bandwidth-certifier` | certified minimum-bandwidth engine (many subcommands) |
| `vmps-fiedler-ordering` | correlation-aware ordering from a DMRG correlation matrix / results JSON |
| `vmps-dmrg-log-to-json` | convert a DMRG `.log` into the results-JSON schema |
| `vmps-build-nnn-tables` | regenerate `cluster_edges_NNN.py` from the generator |
| `vmps-qubo-*` | experimental QUBO bandwidth track (see below) |

## Cluster generation

```bash
# nearest-neighbour (J1) graph, human-readable python output
vmps-cluster-generator kagomeBtorus --Nx 6 --Ny 6 --Nz 1

# next-nearest-neighbour (J2): the 2nd real-space distance shell.
# --neighbor-shell N (alias --shell) selects shell N; N=1 is the default NN graph.
vmps-cluster-generator garnet --Nx 2 --Ny 2 --Nz 2 --neighbor-shell 2 --format edgelist > garnet_nnn.txt
```

- `--format {python,json,edgelist}` — `edgelist` prints whitespace `u v` pairs
  (0-indexed), directly consumable as the certifier's `--j2-file`.
- Site indexing is **shell-independent**, so generating a cluster once at `--shell 1`
  and once at `--shell 2` yields matching numbering — the two are usable as J1 and J2.
- Plots (with the `plot` extra) are written to `plots/` (gitignored); `_shell<N>` is
  added to the filename for shells > 1.

## Bandwidth certification

`vmps-bandwidth-certifier` certifies the minimum bandwidth k\* of a cluster with
(i) a feasible labeling of bandwidth k\* and (ii) an infeasibility proof for k\*−1.
Proofs come at three strengths, all tracked in the state file: `math` (combinatorial
bounds), `cpsat` (CP-SAT verdict), `drat` (external SAT solver UNSAT + DRAT proof).

State lives under **`~/vmps_geometry_data/`** by convention (one JSON per cluster in
`bw_run_<cluster>/`, campaign logs alongside). All subcommands merge into the state
file under a lock, so many jobs can run in parallel across seeds / k-values / machines.

The subcommands (`info`, `heuristic`, `optimize`, `ladder`, `decide`, `cnf`,
`verify-unsat`, … plus the `lex-*`, `w-*`, `h2-*`, `ss-*` families) can be driven by
hand, but the campaign scripts wrap the whole workflow. They live in the package
directory (`src/vmps_geometry/`) and append to their logs:

```bash
cd src/vmps_geometry
./certify_cluster.sh pyrochlore128          # full campaign for one cluster
./certify_small.sh                          # batch of small clusters
./certify_large.sh pyrochlore128            # one big cluster (nohup, more resources)
```

## Lexicographic J1 → J2 bandwidth

Given a J1 (nearest-neighbour) graph and a J2 (next-nearest-neighbour) graph on the
**same** sites, minimize the J2 bandwidth k2 subject to the J1 bandwidth staying at a
cap k1 — the lexicographic (k1, then k2) objective. J2 defaults to the same-named entry
in `cluster_edges_NNN.py`, or pass an explicit edge file.

```bash
cd src/vmps_geometry
# prerequisite: J1 bandwidth already in the shared data dir
# (run ./certify_small.sh / ./certify_large.sh), or pass K1=<cap> explicitly.
./certify_lex_small.sh  kagomeYcyl288_16x12                 # small batch / one cluster
./certify_lex_large.sh  kagomeYcyl288_16x12                 # big: ~10h per relaxed k1, ≤160 CPUs
./certify_lex_large.sh  garnet2x2x2 garnet_nnn.txt          # explicit J2 edge file
```

Key knobs (env vars on the master driver `certify_lex_cluster.sh`):

- **`K1`** — explicit J1 bandwidth cap. If unset, the base cap is the best-known J1
  bandwidth from the plain campaign's state (works even if J1 isn't fully certified).
- **`SOFTEN` / `SOFTEN_MIN` / `SOFTEN_MAX`** — sweep the J1 cap over `k1 = base + s` for
  `s = SOFTEN_MIN..SOFTEN_MAX`. Loosening J1 (higher s) buys a smaller k2 — the J1↔J2
  trade-off. Each s is independent; `SOFTEN=n` is a shortcut for a single cap.
- **`LADDER_TIME`** — per-softening wall-clock budget, split into a parallel-SA seed
  phase (`HEUR_TIME`, default half) and the CP-SAT ladder; enforced by coreutils
  `timeout` or a portable bash fallback. `0` = no cap.
- **`TIME_PER_K`**, **`WORKERS`**, **`PROCS`**, **`SEED`**, **`SYMMETRY`** — per-decision
  time, CP-SAT threads, SA chains, seed, and `orbit|reversal` navigation.

Each softening's state/labeling is tagged by effective cap
(`<cluster>__lex_<j2>__k1_<eff>.json`), so caps never share files and the sweep is
resumable.

## Correlation-aware (Fiedler) ordering

Order sites by the second-smallest eigenvector of the **weighted** graph Laplacian
`L = D − W`, where `W` comes from DMRG two-site correlations — a good DMRG site order
puts strongly-correlated sites near each other. Pipeline:

```bash
# 1. DMRG run log -> results JSON
vmps-dmrg-log-to-json model=Heis_sys=pyrochlore64.log -o results.json

# 2. results JSON (correlations of the lowest-energy run are picked automatically)
#    -> permutation. Also accepts a raw .npy / whitespace-text correlation matrix.
vmps-fiedler-ordering results.json --weights concurrence --refine --out order.txt
```

`--weights`: `abs` (|C|, sign-blind), `concurrence` (`max(0, −2C − 1/2)`, recommended
for spin-1/2 `<S_i·S_j>` — FM pairs correctly get ~0 weight), or `mi` (input is already
mutual information). `--refine` adds a local-search polish; `--objective` picks weighted
bandwidth vs. cutwidth.

## Regenerating the NNN tables

`cluster_edges_NNN.py` is generated from `cluster_edges.py` + the generator and kept in
the repo. After changing either, regenerate and commit:

```bash
vmps-build-nnn-tables         # rewrites src/vmps_geometry/cluster_edges_NNN.py
```

CI regenerates and diffs it, so the committed table can never drift from the generator.

## QUBO track (experimental)

`vmps_geometry.qubo` (`vmps-qubo-bandwidth`, `vmps-qubo-classical-compare`,
`vmps-qubo-sweep`) is an **experimental** bandwidth-via-QUBO track (needs the `qubo`
extra). It carries none of the certifier's guarantees and is not maintained in lockstep
with it — prefer `vmps-bandwidth-certifier` for real results.

The heavy lifting is done by two **C++ solvers** the Python front-ends shell out to.
Their source is vendored under [`src/vmps_geometry/qubo/cpp/`](src/vmps_geometry/qubo/cpp);
build them with `make` (the QUBO solver needs [Eigen](https://eigen.tuxfamily.org);
the Makefile auto-detects a Homebrew/system `eigen3`):

```bash
cd src/vmps_geometry/qubo/cpp
make                                   # -> bandwidth_qubo_tsq_cpp, bandwidth_classical_compare_cpp
```

Point the Python driver at the built binary with `--cpp-binary`
(default: `bandwidth_qubo_tsq_cpp`, looked up from the working directory):

```bash
vmps-qubo-bandwidth ... --cpp-binary src/vmps_geometry/qubo/cpp/bandwidth_qubo_tsq_cpp
```

The dense NumPy tabu-search solver `qubo/tsqubo.py` is a Python translation of the C++
header-only library **[libtsqubo](https://github.com/rliang/libtsqubo)** by rliang
(MIT, © 2021 rliang); that header is vendored at `qubo/cpp/libtsqubo/tsqubo.h` with its
original license.

**Reference.** The QUBO formulations used here follow Qinyu Guo and Michael J. Dinneen,
*"Quantum Annealing for Computing Bandwidth: QUBO Formulations and Solution Strategies"*
(School of Computer Science, University of Auckland). Its three modeling strategies —
decision-based with external binary search, optimization-based with auxiliary/slack
variables, and exponential-penalty — correspond to this track's solver modes.

## Development

```bash
pip install -e '.[dev]'
pytest -q
```

CI runs the test suite, `bash -n` on every campaign script, advisory `shellcheck`, and
the NNN-table regeneration guard.

## License

MIT — see [LICENSE](LICENSE).
