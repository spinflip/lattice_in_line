# Lattice in Line

This repo has two goals:

1. Generate lattice graphs for density-matrix renormalization group (DMRG) computations (including visualization).
2. Perform "lattice compilation", i.e. compute an optimal enumeration to best fit the 1D lattice geometry of DMRG.

The corresponding paper reference is: TBA

The code for this project was made with heavy use of Fable 5.

The entry point of 1. is `lil-cluster-generator` (directs to `cluster_generator.py`).
The entry point of 2. is `lil-bandwidth-certifier`, with additional wrappers around it:

- `certify_cluster.sh`: the most general script
- `certify_small.sh`: allocates small resources for a small cluster
- `certify_large.sh`: allocates larger resources for a large cluster.

The import package is `lattice_in_line`.

Lattice compilation runs a multi-stage CP-SAT campaign as described in the paper, attempting to certify the result. It can be run to optimize for the cutwidth (`MODE=cutwidth`, the default) or for the bandwidth (`MODE=plain`). In both cases, the secondary objective of the average interaction range is optimized as well. Cutwidth is the default and gives much better downstream DMRG performance for complex clusters.

The file `cluster_edges.py` is a datafile that holds the graph edges for various lattices from frustrated magnetism discussed in the paper.
Similarly, the files `permutations_sat_cw.py` and `permutations_sat_bw.py` hold pre-computed site permutations (cutwidth-optimal and bandwidth-optimal, respectively).

My workflow is:

- Run `lil-cluster-generator`, which gives real-space coordinates of the lattice, as well as the graph as a list of edges.
- Add the result to `cluster_edges.py`.
- Run the CP-SAT campaign using `lil-bandwidth-certifier` or the shell scripts listed above.
- Add the resulting optimized permutation to `permutations_sat_cw.py`.
- These results can then be imported in a DMRG code.

The code remains hackable and extensible. If you build on it and find it useful, please consider citing the project and dropping me a message.

There is also an experimental QUBO module coupled with tabu search for the same problem, but this approach gave me much worse results than CP-SAT (results in `permutations_qubo_bw.py`). Other experimental modules only described in the outlook of the paper are Fiedler ordering, lexicographic J1-J2 bandwidth optimization, and optimization with simultaneous supersite creation.

## Installation

Requires Python ≥ 3.10; the core install pulls in only NumPy.

```bash
cd /path/to/lattice_in_line
pip install -e .            # core (numpy only)
pip install -e '.[all]'     # everything
```

On a distribution- or Homebrew-managed Python, `pip install` refuses to touch the
system environment (PEP 668). Use a virtual environment:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[all]'
```

Extras (declared in `pyproject.toml`):

| Extra | Pulls in | Enables |
|---|---|---|
| `plot` | matplotlib | lattice, ladder and analysis plots (otherwise skipped, coords/edges still emitted) |
| `certify` | ortools, python-sat | CP-SAT decisions + independent SAT cross-check |
| `orbits` | pynauty | orbit (Aut(G)) symmetry breaking (falls back to reversal if absent); increases performance |
| `qubo` | scipy | SciPy's `reverse_cuthill_mckee`: the reference RCM used as the QUBO track's warm start and as the RCM baseline in the benchmarks. Without it both fall back to a pure-python Cuthill–McKee whose tie-breaking gives worse orderings (C60: cutwidth 16 instead of 13) |
| `dev` | pytest | test suite |

## Command-line tools

*Graphs*

| Command | What it does |
|---|---|
| `lil-cluster-generator` | generate a cluster's coordinates + neighbour graph |
| `lil-build-nnn-tables` | regenerate `cluster_edges_NNN.py` from the generator |

*Ordering*

| Command | What it does |
|---|---|
| `lil-bandwidth-certifier` | the certification engine: cutwidth (`cw-*`) and bandwidth subcommands |
| `lil-fiedler-ordering` | correlation-aware ordering from a DMRG correlation matrix / results JSON |
| `lil-bandwidth-benchmark` | run every ordering heuristic (RCM, GPS, Sloan, King, spectral) over the cluster set |
| `lil-gather-permutations` | collect campaign results into a `permutations_*.py` table |

*Analysis and plots*

| Command | What it does |
|---|---|
| `lil-analyze-mpo-correlation` | correlate MPO bond dimension against cutwidth / bandwidth / average range |
| `lil-bandwidth-cutwidth-correlation` | correlate the two objectives across clusters |
| `lil-cluster-ladder-plot` | draw a cluster as a quasi-1D ladder in its MPS ordering |
| `lil-qubo-*` | experimental QUBO bandwidth track (see below) |

## Cluster generation

```bash
# nearest-neighbour (J1) graph, human-readable python output
lil-cluster-generator triangularBtorus --Nx 6 --Ny 6 --Nz 1

# next-nearest-neighbour (J2): the 2nd real-space distance shell.
# --neighbor-shell N (alias --shell) selects shell N; N=1 is the default NN graph.
lil-cluster-generator triangularYcyl --Nx 8 --Ny 4 --Nz 1 \
    --neighbor-shell 2 --format edgelist > triangular_nnn.txt
```

Lattices: `triangular{Y,X,B}{cyl,torus}`, `kagome{Y,X,B}{cyl,torus}`,
`square{Cyl,Torus}`, `hyperkagome`, `pyrochlore` (incl. `--tilted`), `trillium`,
`fcc`, `garnet`, and the molecules `icosa`, `cubocta`, `icosidodeca`, `C12`–`C60`.
Cell counts come from `--Nx/--Ny/--Nz`, or an arbitrary supercell from
`--supercell "a,b,c;d,e,f;g,h,i"`.

- `--format {python,json,edgelist}` — `edgelist` prints whitespace `u v` pairs
  (0-indexed), directly consumable as the certifier's `--j2-file`.
- Site indexing is **shell-independent**, so generating a cluster once at `--shell 1`
  and once at `--shell 2` yields matching numbering — the two are usable as J1 and J2.
- Plots (with the `plot` extra) are written to `plots/` (gitignored); `_shell<N>` is
  added to the filename for shells > 1. `--plot-permutation-file` draws the lattice
  in the ordering stored in a `permutations_*.py` table.

## Cutwidth certification

The cutwidth `c*` — the largest number of Hamiltonian edges crossing any cut of the
MPS chain — is the quantity that sets the MPO bond dimension, so it is the default
objective. A campaign aims to close the window `[LB, UB]` with (i) a layout of
cutwidth `UB` and (ii) an infeasibility proof for `UB − 1`. Proofs come at three
strengths, all tracked in the state file:

| Level | Meaning |
|---|---|
| `math` | combinatorial bounds (max degree, degree-sum prefix, spectral/Fiedler) — hand-checkable |
| `cpsat` | CP-SAT INFEASIBLE verdict — trust the solver |
| `xsat` | two independent SAT solvers (CaDiCaL + Glucose) agree on UNSAT; the DRAT proof is archived |

Cutwidth lower bounds are much weaker than bandwidth ones, so the window closes only
for small clusters; for larger ones the campaign still returns the best layout it
found, with the window reported honestly.

The campaign scripts wrap the whole workflow. They live in the package directory and
append to their logs:

```bash
cd src/lattice_in_line
./certify_cluster.sh pyrochlore128          # full campaign for one cluster
./certify_small.sh                          # batch of small clusters (8 CPUs)
./certify_large.sh pyrochlore128            # one big cluster (nohup, 32 CPUs)
MODE=plain ./certify_large.sh pyrochlore128 # bandwidth campaign instead
```

Phases: structural lower bounds → simulated-annealing upper bound → alternating CP-SAT
decision ladder → SAT cross-check of the decisive UNSAT (only once the window closes)
→ polish, which minimizes the total interaction range while holding the cutwidth
fixed. Each run ends by printing the layout's `bandwidth`, `cutwidth` and `avg_range`.

Useful knobs (environment variables): `CPUS` (peak cores), `MODE`, `SEED`,
`TIME_HEUR`, `TIME_PER_K`, `LADDER_TIME`, `POLISH_TIME`, `SAT_TIME=0` to skip the
cross-check, `PLAN_ONLY=1` to print the plan and exit, `YES=1` to skip the prompt.

State lives under **`~/lattice_in_line_data/`** by convention: one JSON per cluster in
`cw_run_<cluster>/` (`bw_run_<cluster>/` for bandwidth), with the campaign log as
`certify_<cluster>_cw.log` / `_bw.log`. All subcommands merge into the state file
under a lock, so runs are resumable and many jobs can proceed in parallel across
seeds / k-values / machines.

Driving it by hand is possible too: `cw-run`, `cw-decide`, `cw-cnf`, `cw-verify`,
`cw-polish`, `cw-export` for cutwidth; `info`, `heuristic`, `optimize`, `ladder`,
`decide`, `cnf`, `verify-unsat`, `polish`, `status`, `export` for bandwidth; plus the
`lex-*` (lexicographic J1→J2), `w-*` (weighted), `h2-*` and `ss-*` (supersite)
families.

## Regenerating the NNN tables

`cluster_edges_NNN.py` is generated from `cluster_edges.py` + the generator and kept in
the repo (the lexicographic J1→J2 mode reads it as its default J2 source). After changing
either input, regenerate and commit:

```bash
lil-build-nnn-tables         # rewrites src/lattice_in_line/cluster_edges_NNN.py
```

CI regenerates and diffs it, so the committed table can never drift from the generator.

## QUBO track (experimental)

`lattice_in_line.qubo` (`lil-qubo-bandwidth`, `lil-qubo-classical-compare`,
`lil-qubo-sweep`, needs the `qubo` extra) formulates the bandwidth problem as a QUBO
and attacks it with tabu search. It carries none of the certifier's guarantees and
performed clearly worse in the paper's benchmarks — prefer `lil-bandwidth-certifier`
for real results. The formulations follow Guo and Dinneen, *"Quantum Annealing for
Computing Bandwidth: QUBO Formulations and Solution Strategies"* (University of
Auckland), whose three strategies correspond to this track's solver modes.

The heavy lifting is done by two C++ solvers the Python front-ends shell out to.
Build them first (needs [Eigen](https://eigen.tuxfamily.org); the Makefile auto-detects
a Homebrew/system `eigen3`), then point the driver at the binary:

```bash
cd src/lattice_in_line/qubo/cpp && make
lil-qubo-bandwidth ... --cpp-binary src/lattice_in_line/qubo/cpp/bandwidth_qubo_tsq_cpp
```

The tabu solver is a translation of **[libtsqubo](https://github.com/rliang/libtsqubo)**
by rliang (MIT, © 2021), vendored at `qubo/cpp/libtsqubo/tsqubo.h`.

## Paper figure reproduction

- lil-analyze-mpo-correlation
- lil-bandwidth-benchmark
- lil-cluster-ladder-plot

## Development

```bash
pip install -e '.[dev]'
pytest -q
```

CI runs the test suite, `bash -n` on every campaign script, advisory `shellcheck`, and
the NNN-table regeneration guard.

## License

MIT — see [LICENSE](LICENSE).
