from __future__ import annotations

import argparse
import subprocess
import time
from dataclasses import dataclass
from itertools import combinations
from math import ceil, log2
from pathlib import Path
from typing import Dict, List, Tuple, Hashable, Optional, Iterable, Sequence

import numpy as np

from .tsqubo import TSQUBO
from ..cluster_edges import CLUSTER_EDGES


Edge = Tuple[int, int]
EdgeList = List[Edge]
QuboDict = Dict[Tuple[Hashable, Hashable], float]


@dataclass
class QuboModel:
    Q: QuboDict
    variables: List[Hashable]
    index: Dict[Hashable, int]

    def to_matrix(self) -> np.ndarray:
        """
        Dense matrix for TSQUBO.

        TSQUBO uses:
            f(x) = x.T @ Q @ x

        For off-diagonal QUBO terms c*x_i*x_j, put c in one triangle only.
        """
        n = len(self.variables)
        mat = np.zeros((n, n), dtype=np.float64)

        for (u, v), coeff in self.Q.items():
            i = self.index[u]
            j = self.index[v]

            if i == j:
                mat[i, i] += coeff
            else:
                if i > j:
                    i, j = j, i
                mat[i, j] += coeff

        return mat


@dataclass
class BandwidthResult:
    method: str
    ordering: List[int]              # position -> vertex
    position: Dict[int, int]         # vertex -> position
    bandwidth: int
    energy: float
    feasible_permutation: bool
    raw_bits: np.ndarray
    model: QuboModel
    elapsed_s: float = 0.0
    model_build_s: float = 0.0
    matrix_assembly_s: float = 0.0
    solver_core_s: float = 0.0
    scan_count: int = 0
    solver_runs: int = 0


def _require_gurobi():
    try:
        import gurobipy as gp
    except ImportError as exc:
        raise RuntimeError(
            "Gurobi backend requested, but gurobipy is not installed. "
            "Install gurobipy and ensure your Gurobi license is configured."
        ) from exc
    return gp


class QuboBuilder:
    def __init__(self) -> None:
        self.Q: QuboDict = {}
        self.variables_seen: set[Hashable] = set()

    def _register(self, *variables: Hashable) -> None:
        for var in variables:
            self.variables_seen.add(var)

    def add(self, u: Hashable, v: Hashable, coeff: float) -> None:
        if coeff == 0:
            return

        self._register(u, v)

        if repr(u) > repr(v):
            u, v = v, u

        self.Q[(u, v)] = self.Q.get((u, v), 0.0) + float(coeff)

    def add_linear(self, u: Hashable, coeff: float) -> None:
        self.add(u, u, coeff)

    def add_quadratic(self, u: Hashable, v: Hashable, coeff: float) -> None:
        if u == v:
            self.add_linear(u, coeff)
        else:
            self.add(u, v, coeff)

    def add_square_of_linear(
        self,
        terms: Iterable[Tuple[Hashable, float]],
        weight: float,
    ) -> None:
        """
        Adds:
            weight * (sum_i c_i x_i)^2

        Constant terms are omitted.
        """
        terms = list(terms)

        for var, coeff in terms:
            self.add_linear(var, weight * coeff * coeff)

        for (v1, c1), (v2, c2) in combinations(terms, 2):
            self.add_quadratic(v1, v2, weight * 2.0 * c1 * c2)

    def add_one_hot(self, variables: List[Hashable], weight: float) -> None:
        """
        Adds:
            weight * (sum_i x_i - 1)^2

        Dropping the constant gives:
            -weight * sum_i x_i
            +2*weight * sum_{i<j} x_i x_j
        """
        for var in variables:
            self.add_linear(var, -weight)

        for u, v in combinations(variables, 2):
            self.add_quadratic(u, v, 2.0 * weight)

    def add_and_constraint(
        self,
        z: Hashable,
        x: Hashable,
        y: Hashable,
        weight: float,
    ) -> None:
        """
        Enforces:
            z = x AND y

        Penalty:
            weight * (x*y - 2*x*z - 2*y*z + 3*z)
        """
        self.add_quadratic(x, y, weight)
        self.add_quadratic(x, z, -2.0 * weight)
        self.add_quadratic(y, z, -2.0 * weight)
        self.add_linear(z, 3.0 * weight)

    def build(self) -> QuboModel:
        variables = sorted(self.variables_seen, key=repr)
        index = {var: i for i, var in enumerate(variables)}
        return QuboModel(Q=self.Q, variables=variables, index=index)


def normalize_edges(edges: EdgeList) -> Tuple[List[int], EdgeList]:
    vertices = sorted(set(v for e in edges for v in e))
    remap = {v: i for i, v in enumerate(vertices)}

    normalized = []
    for u, v in edges:
        if u == v:
            continue

        a = remap[u]
        b = remap[v]

        if a > b:
            a, b = b, a

        normalized.append((a, b))

    return list(range(len(vertices))), sorted(set(normalized))


def original_to_normalized_vertex_map(edges: EdgeList) -> Dict[int, int]:
    vertices = sorted(set(v for e in edges for v in e))
    return {old_vertex: new_vertex for new_vertex, old_vertex in enumerate(vertices)}


def x_var(v: int, pos: int) -> Tuple[str, int, int]:
    return ("x", v, pos)


def k_var(bit: int) -> Tuple[str, int]:
    return ("K", bit)


def s_var(edge_id: int, bit: int) -> Tuple[str, int, int]:
    return ("s", edge_id, bit)


def z_var(edge_id: int, i: int, j: int) -> Tuple[str, int, int, int]:
    return ("z", edge_id, i, j)


def add_permutation_constraints(qb: QuboBuilder, n: int, weight: float) -> None:
    """
    Enforce:
        each vertex gets exactly one position,
        each position gets exactly one vertex.
    """
    for v in range(n):
        qb.add_one_hot([x_var(v, i) for i in range(n)], weight)

    for i in range(n):
        qb.add_one_hot([x_var(v, i) for v in range(n)], weight)


def build_decision_model(
    edges: EdgeList,
    k: int,
    permutation_weight: float = 500.0,
    edge_weight: float = 100.0,
) -> QuboModel:
    """
    Decision QUBO.

    Checks whether there is a permutation with bandwidth <= k.
    Penalizes edge placements with |position(u)-position(v)| > k.
    """
    vertices, edges = normalize_edges(edges)
    n = len(vertices)

    qb = QuboBuilder()
    add_permutation_constraints(qb, n, permutation_weight)

    for u, v in edges:
        for i in range(n):
            for j in range(n):
                if abs(i - j) > k:
                    qb.add_quadratic(x_var(u, i), x_var(v, j), edge_weight)

    return qb.build()


def build_exponential_model(
    edges: EdgeList,
    permutation_weight: float = 5000.0,
    base: Optional[float] = None,
    max_distance: Optional[int] = None,
) -> QuboModel:
    """
    Exponential-penalty QUBO.

    Objective:
        sum_edges sum_{i,j} base^{|i-j|} x_{u,i} x_{v,j}

    plus permutation penalties.
    """
    vertices, edges = normalize_edges(edges)
    n = len(vertices)

    if base is None:
        base = n * (n - 1) / 2 + 1

    if max_distance is not None and max_distance < 0:
        raise ValueError("max_distance must be non-negative")

    qb = QuboBuilder()
    add_permutation_constraints(qb, n, permutation_weight)

    for u, v in edges:
        for i in range(n):
            for j in range(n):
                dist = abs(i - j)
                if max_distance is not None and dist > max_distance:
                    continue
                qb.add_quadratic(x_var(u, i), x_var(v, j), base ** dist)

    return qb.build()


def build_optimization_model(
    edges: EdgeList,
    permutation_weight: float = 200.0,
    and_weight: float = 200.0,
    bandwidth_constraint_weight: float = 100.0,
    minimize_k_weight: float = 1.0,
) -> QuboModel:
    """
    One-shot optimization QUBO.

    Variables:
        x[v,i]      = vertex-position assignment
        z[e,i,j]    = x[u,i] AND x[v,j] for edge e=(u,v)
        K bits      = encoded bandwidth
        s[e] bits   = edge-specific slack

    For each edge:
        d_e = sum_{i,j} |i-j| z[e,i,j]
        d_e <= K

    Encoded as:
        d_e - K + s_e = 0

    This formulation is expensive with dense tabu search.
    """
    vertices, edges = normalize_edges(edges)
    n = len(vertices)

    max_bandwidth = n - 1
    num_bits = max(1, ceil(log2(max_bandwidth + 1)))

    qb = QuboBuilder()
    add_permutation_constraints(qb, n, permutation_weight)

    for b in range(num_bits):
        qb.add_linear(k_var(b), minimize_k_weight * (2 ** b))

    for edge_id, (u, v) in enumerate(edges):
        for i in range(n):
            for j in range(n):
                qb.add_and_constraint(
                    z=z_var(edge_id, i, j),
                    x=x_var(u, i),
                    y=x_var(v, j),
                    weight=and_weight,
                )

        distance_terms = []
        for i in range(n):
            for j in range(n):
                dist = abs(i - j)
                if dist:
                    distance_terms.append((z_var(edge_id, i, j), dist))

        k_terms = [(k_var(b), -(2 ** b)) for b in range(num_bits)]
        slack_terms = [(s_var(edge_id, b), 2 ** b) for b in range(num_bits)]

        qb.add_square_of_linear(
            distance_terms + k_terms + slack_terms,
            weight=bandwidth_constraint_weight,
        )

    return qb.build()


def compute_bandwidth(edges: EdgeList, position: Dict[int, int]) -> int:
    _, normalized_edges = normalize_edges(edges)

    if not normalized_edges:
        return 0

    return max(abs(position[u] - position[v]) for u, v in normalized_edges)


def compute_avg_range(edges: EdgeList, position: Dict[int, int]) -> float:
    """Mean edge length (MinLA cost / |E|), i.e. the average interaction range.

    Historically called "envelope" here -- not to be confused with the classic
    sparse-matrix envelope/profile (sum of per-row bandwidths).
    """
    _, normalized_edges = normalize_edges(edges)

    if not normalized_edges:
        return 0.0

    total = 0
    for u, v in normalized_edges:
        total += abs(position[u] - position[v])

    return total / len(normalized_edges)


# Deprecated alias (pre-rename name).
compute_envelope = compute_avg_range


def bandwidth_lower_bound(edges: EdgeList) -> int:
    """
    Simple lower bound:
        bandwidth >= ceil(max_degree / 2)
    """
    vertices, normalized_edges = normalize_edges(edges)
    n = len(vertices)

    deg = [0] * n
    for u, v in normalized_edges:
        deg[u] += 1
        deg[v] += 1

    if not deg:
        return 0

    return (max(deg) + 1) // 2


def decode_ordering(
    bits: np.ndarray,
    model: QuboModel,
    n: int,
) -> Tuple[List[int], Dict[int, int], bool]:
    """
    Decode x[v,i] variables.

    If the solver output is not a valid permutation, return a repaired
    permutation and feasible_permutation=False.
    """
    active_positions: Dict[int, List[int]] = {v: [] for v in range(n)}
    active_vertices: Dict[int, List[int]] = {i: [] for i in range(n)}

    for v in range(n):
        for i in range(n):
            idx = model.index.get(x_var(v, i))
            if idx is not None and int(bits[idx]) == 1:
                active_positions[v].append(i)
                active_vertices[i].append(v)

    feasible = all(len(active_positions[v]) == 1 for v in range(n))
    feasible = feasible and all(len(active_vertices[i]) == 1 for i in range(n))

    if feasible:
        position = {v: active_positions[v][0] for v in range(n)}
        ordering = [active_vertices[i][0] for i in range(n)]
        return ordering, position, True

    # Greedy repair.
    position: Dict[int, int] = {}
    used_positions: set[int] = set()

    for v in range(n):
        chosen = None

        for i in active_positions[v]:
            if i not in used_positions:
                chosen = i
                break

        if chosen is None:
            for i in range(n):
                if i not in used_positions:
                    chosen = i
                    break

        if chosen is None:
            raise RuntimeError("failed to repair invalid permutation")

        position[v] = chosen
        used_positions.add(chosen)

    ordering: List[int] = [-1] * n
    for v, i in position.items():
        ordering[i] = v

    return ordering, position, False


def degree_ordering(edges: EdgeList) -> List[int]:
    vertices, normalized_edges = normalize_edges(edges)
    n = len(vertices)

    deg = [0] * n
    for u, v in normalized_edges:
        deg[u] += 1
        deg[v] += 1

    return sorted(range(n), key=lambda v: deg[v])


def reverse_cuthill_mckee_ordering(edges: EdgeList) -> List[int]:
    """
    Optional RCM baseline.
    Requires scipy.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import reverse_cuthill_mckee

    vertices, normalized_edges = normalize_edges(edges)
    n = len(vertices)

    rows = []
    cols = []
    data = []

    for u, v in normalized_edges:
        rows.extend([u, v])
        cols.extend([v, u])
        data.extend([1, 1])

    A = csr_matrix((data, (rows, cols)), shape=(n, n))
    return list(map(int, reverse_cuthill_mckee(A)))


def candidate_initial_orderings(
    edges: EdgeList,
    max_random: int,
    seed: Optional[int],
    rcm_start: bool = False,
) -> List[List[int]]:
    """
    Small set of useful starts:
        identity,
        degree ordering,
        RCM if scipy is available,
        reversed versions,
        random orderings.
    """
    vertices, normalized_edges = normalize_edges(edges)
    n = len(vertices)
    rng = np.random.default_rng(seed)

    orderings: List[List[int]] = []

    rcm_ordering: Optional[List[int]] = None
    try:
        rcm_ordering = reverse_cuthill_mckee_ordering(normalized_edges)
    except Exception:
        rcm_ordering = None

    if rcm_start and rcm_ordering is not None:
        orderings.append(rcm_ordering)

    orderings.append(list(range(n)))
    orderings.append(degree_ordering(normalized_edges))

    if rcm_ordering is not None:
        orderings.append(rcm_ordering)

    base = list(orderings)
    for ordering in base:
        orderings.append(list(reversed(ordering)))

    for _ in range(max_random):
        orderings.append(list(map(int, rng.permutation(n))))

    seen = set()
    unique: List[List[int]] = []

    for ordering in orderings:
        key = tuple(ordering)
        if key not in seen:
            unique.append(ordering)
            seen.add(key)

    return unique


def make_initial_bits_from_ordering(
    model: QuboModel,
    edges: EdgeList,
    ordering: List[int],
    include_auxiliary: bool,
) -> np.ndarray:
    """
    Build an initial bitstring from a given ordering.

    For decision/exponential:
        only x variables are relevant.

    For optimization:
        also initialize z, K, and slack consistently.
    """
    vertices, normalized_edges = normalize_edges(edges)
    n = len(vertices)

    bits = np.zeros(len(model.variables), dtype=np.int8)
    position = {v: i for i, v in enumerate(ordering)}

    for v in range(n):
        idx = model.index.get(x_var(v, position[v]))
        if idx is not None:
            bits[idx] = 1

    if not include_auxiliary:
        return bits

    bandwidth = compute_bandwidth(normalized_edges, position)
    num_bits = max(1, ceil(log2(n)))

    for b in range(num_bits):
        idx = model.index.get(k_var(b))
        if idx is not None:
            bits[idx] = (bandwidth >> b) & 1

    for edge_id, (u, v) in enumerate(normalized_edges):
        pu = position[u]
        pv = position[v]
        d = abs(pu - pv)

        idx = model.index.get(z_var(edge_id, pu, pv))
        if idx is not None:
            bits[idx] = 1

        slack = bandwidth - d
        for b in range(num_bits):
            idx = model.index.get(s_var(edge_id, b))
            if idx is not None:
                bits[idx] = (slack >> b) & 1

    return bits


def solve_model_with_tsq_fast(
    model: QuboModel,
    edges: EdgeList,
    method: str,
    initial_orderings: List[List[int]],
    tabu_tenure: int = 12,
    cutoff: int = 150,
    fixed_weight: Optional[int] = None,
    include_auxiliary_initialization: bool = False,
    stop_at_bandwidth: Optional[int] = None,
) -> BandwidthResult:
    """
    Fast TSQUBO solve wrapper.

    For decision/exponential, fixed_weight=n is used because exactly n
    assignment bits should be active.
    """
    vertices, normalized_edges = normalize_edges(edges)
    n = len(vertices)

    best: Optional[BandwidthResult] = None
    t0 = time.perf_counter()
    t_matrix0 = time.perf_counter()
    matrix = model.to_matrix()
    matrix_assembly_s = time.perf_counter() - t_matrix0
    solver_core_s = 0.0
    solver_runs = 0

    for ordering0 in initial_orderings:
        initial_bits = make_initial_bits_from_ordering(
            model=model,
            edges=normalized_edges,
            ordering=ordering0,
            include_auxiliary=include_auxiliary_initialization,
        )

        t_solver0 = time.perf_counter()
        solver = TSQUBO.from_matrix(matrix, fixed_weight=fixed_weight)
        solution = solver.solve(
            tabu_tenure=tabu_tenure,
            cutoff=cutoff,
            initial_x=initial_bits,
        )
        solver_core_s += time.perf_counter() - t_solver0
        solver_runs += 1

        bits = solution.x.astype(np.int8, copy=True)
        ordering, position, feasible_perm = decode_ordering(bits, model, n)
        bw = compute_bandwidth(normalized_edges, position)

        result = BandwidthResult(
            method=method,
            ordering=ordering,
            position=position,
            bandwidth=bw,
            energy=float(solution.fx),
            feasible_permutation=feasible_perm,
            raw_bits=bits,
            model=model,
        )

        if best is None:
            best = result
        else:
            key = (
                not result.feasible_permutation,
                result.bandwidth,
                result.energy,
            )
            best_key = (
                not best.feasible_permutation,
                best.bandwidth,
                best.energy,
            )

            if key < best_key:
                best = result

        if (
            stop_at_bandwidth is not None
            and best.feasible_permutation
            and best.bandwidth <= stop_at_bandwidth
        ):
            break

    assert best is not None
    best.elapsed_s = time.perf_counter() - t0
    best.matrix_assembly_s = matrix_assembly_s
    best.solver_core_s = solver_core_s
    best.solver_runs = solver_runs
    return best


def solve_model_with_gurobi(
    model: QuboModel,
    edges: EdgeList,
    method: str,
    initial_orderings: List[List[int]],
    include_auxiliary_initialization: bool = False,
    time_limit_s: Optional[float] = None,
    mip_gap: Optional[float] = None,
) -> BandwidthResult:
    """
    Exact/branch-and-bound solve wrapper using Gurobi.

    This solves the QUBO directly as a binary quadratic model. If an initial
    ordering is supplied, it is passed as a warm start.
    """
    t0 = time.perf_counter()
    gp = _require_gurobi()

    vertices, normalized_edges = normalize_edges(edges)
    n = len(vertices)

    grb = gp.GRB
    m = gp.Model(f"bandwidth_qubo_{method}")
    m.Params.OutputFlag = 0
    m.Params.NonConvex = 2
    if time_limit_s is not None:
        m.Params.TimeLimit = float(time_limit_s)
    if mip_gap is not None:
        m.Params.MIPGap = float(mip_gap)

    xvars = m.addVars(len(model.variables), vtype=grb.BINARY, name="x")

    objective = gp.QuadExpr()
    for (u, v), coeff in model.Q.items():
        i = model.index[u]
        j = model.index[v]
        if i == j:
            objective += float(coeff) * xvars[i]
        else:
            objective += float(coeff) * xvars[i] * xvars[j]
    m.setObjective(objective, grb.MINIMIZE)

    if initial_orderings:
        start_bits = make_initial_bits_from_ordering(
            model=model,
            edges=normalized_edges,
            ordering=initial_orderings[0],
            include_auxiliary=include_auxiliary_initialization,
        )
        for i, bit in enumerate(start_bits):
            xvars[i].Start = int(bit)

    m.optimize()

    if m.SolCount <= 0:
        raise RuntimeError(f"Gurobi did not return a solution (status {m.Status}).")

    bits = np.array([int(round(xvars[i].X)) for i in range(len(model.variables))], dtype=np.int8)
    ordering, position, feasible_perm = decode_ordering(bits, model, n)
    bw = compute_bandwidth(normalized_edges, position)

    return BandwidthResult(
        method=method,
        ordering=ordering,
        position=position,
        bandwidth=bw,
        energy=float(m.ObjVal),
        feasible_permutation=feasible_perm,
        raw_bits=bits,
        model=model,
        elapsed_s=time.perf_counter() - t0,
        solver_core_s=time.perf_counter() - t0,
        scan_count=1,
        solver_runs=1,
    )


def run_cpp_solver(
    *,
    cpp_binary: Path,
    cluster: str,
    method: str,
    random_starts: int,
    tabu_tenure: int,
    cutoff: int,
    seed: int,
    rcm_start: bool,
    rcm: bool,
    exponential_base: Optional[float],
    exponential_max_distance: Optional[int],
    matrix_backend: str,
) -> str:
    cpp_binary = cpp_binary.expanduser().resolve()
    if not cpp_binary.exists():
        raise FileNotFoundError(f"C++ solver binary not found: {cpp_binary}")

    cmd = [
        str(cpp_binary),
        "--cluster",
        cluster,
        "--method",
        method,
        "--random-starts",
        str(random_starts),
        "--tabu-tenure",
        str(tabu_tenure),
        "--cutoff",
        str(cutoff),
        "--seed",
        str(seed),
        "--matrix-backend",
        matrix_backend,
    ]
    if rcm_start:
        cmd.append("--rcm-start")
    if rcm:
        cmd.append("--rcm")
    if exponential_base is not None:
        cmd.extend(["--exponential-base", str(exponential_base)])
    if exponential_max_distance is not None:
        cmd.extend(["--exponential-max-distance", str(exponential_max_distance)])

    completed = subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout


def solve_decision_fast(
    edges: EdgeList,
    backend: str = "tabu",
    max_random_starts: int = 4,
    tabu_tenure: int = 12,
    cutoff: int = 150,
    seed: Optional[int] = 123,
    rcm_start: bool = False,
    permutation_weight: float = 500.0,
    edge_weight: float = 100.0,
    gurobi_time_limit_s: Optional[float] = None,
    gurobi_mip_gap: Optional[float] = None,
) -> BandwidthResult:
    """
    Fast decision-QUBO solver.

    Uses an exact binary search over k only for exact backends.

    For heuristic tabu search, scanning is safer than binary search because a
    failed solve for k is not a proof of infeasibility.
    """
    vertices, normalized_edges = normalize_edges(edges)
    n = len(vertices)

    initial_orderings = candidate_initial_orderings(
        normalized_edges,
        max_random=max_random_starts,
        seed=seed,
        rcm_start=rcm_start,
    )

    def solve_for_k(k: int) -> BandwidthResult:
        t_build0 = time.perf_counter()
        model = build_decision_model(
            normalized_edges,
            k=k,
            permutation_weight=permutation_weight,
            edge_weight=edge_weight,
        )
        model_build_s = time.perf_counter() - t_build0
        if backend == "tabu":
            result = solve_model_with_tsq_fast(
                model=model,
                edges=normalized_edges,
                method=f"decision(k={k})",
                initial_orderings=initial_orderings,
                tabu_tenure=tabu_tenure,
                cutoff=cutoff,
                fixed_weight=n,
                include_auxiliary_initialization=False,
                stop_at_bandwidth=k,
            )
            result.model_build_s = model_build_s
            return result
        if backend == "gurobi":
            result = solve_model_with_gurobi(
                model=model,
                edges=normalized_edges,
                method=f"decision(k={k})",
                initial_orderings=initial_orderings,
                include_auxiliary_initialization=False,
                time_limit_s=gurobi_time_limit_s,
                mip_gap=gurobi_mip_gap,
            )
            result.model_build_s = model_build_s
            return result
        raise ValueError(f"Unknown backend {backend!r}. Use 'tabu' or 'gurobi'.")

    def result_key(result: BandwidthResult) -> Tuple[bool, int, float]:
        return (
            not result.feasible_permutation,
            result.bandwidth,
            result.energy,
        )

    lo = bandwidth_lower_bound(normalized_edges)
    hi = n - 1
    best: Optional[BandwidthResult] = None
    t_total0 = time.perf_counter()
    total_model_build_s = 0.0
    total_matrix_assembly_s = 0.0
    total_solver_core_s = 0.0
    scan_count = 0
    solver_runs = 0

    if backend == "gurobi":
        while lo <= hi:
            k = (lo + hi) // 2
            result = solve_for_k(k)
            scan_count += 1
            total_model_build_s += result.model_build_s
            total_matrix_assembly_s += result.matrix_assembly_s
            total_solver_core_s += result.solver_core_s
            solver_runs += result.solver_runs
            best_bw = result.bandwidth if best is None else min(best.bandwidth, result.bandwidth)
            print(
                f"decision progress: scan={scan_count} k={k} "
                f"solver_runs_so_far={solver_runs} best_bw_so_far={best_bw} "
                f"elapsed_s={time.perf_counter() - t_total0:.3f}"
            )

            feasible_for_k = result.feasible_permutation and result.bandwidth <= k

            if feasible_for_k:
                best = result
                hi = k - 1
            else:
                lo = k + 1
    else:
        # Heuristic tabu search cannot safely drive a binary search. We scan all
        # k values and keep the best actual bandwidth that was found.
        for k in range(lo, hi + 1):
            result = solve_for_k(k)
            scan_count += 1
            total_model_build_s += result.model_build_s
            total_matrix_assembly_s += result.matrix_assembly_s
            total_solver_core_s += result.solver_core_s
            solver_runs += result.solver_runs
            if best is None or result_key(result) < result_key(best):
                best = result
            assert best is not None
            print(
                f"decision progress: scan={scan_count}/{hi - lo + 1} k={k} "
                f"solver_runs_so_far={solver_runs} best_bw_so_far={best.bandwidth} "
                f"elapsed_s={time.perf_counter() - t_total0:.3f}"
            )

    if best is None:
        best = solve_for_k(n - 1)
        scan_count += 1
        total_model_build_s += best.model_build_s
        total_matrix_assembly_s += best.matrix_assembly_s
        total_solver_core_s += best.solver_core_s
        solver_runs += best.solver_runs

    best.method = "decision"
    best.model_build_s = total_model_build_s
    best.matrix_assembly_s = total_matrix_assembly_s
    best.solver_core_s = total_solver_core_s
    best.elapsed_s = time.perf_counter() - t_total0
    best.scan_count = scan_count
    best.solver_runs = solver_runs
    return best


def solve_exponential_fast(
    edges: EdgeList,
    backend: str = "tabu",
    max_random_starts: int = 4,
    tabu_tenure: int = 12,
    cutoff: int = 150,
    seed: Optional[int] = 123,
    rcm_start: bool = False,
    permutation_weight: float = 5000.0,
    exponential_base: Optional[float] = None,
    exponential_max_distance: Optional[int] = None,
    gurobi_time_limit_s: Optional[float] = None,
    gurobi_mip_gap: Optional[float] = None,
) -> BandwidthResult:
    """
    Fast exponential formulation.

    This is one QUBO solve with n^2 variables.
    """
    vertices, normalized_edges = normalize_edges(edges)
    n = len(vertices)

    initial_orderings = candidate_initial_orderings(
        normalized_edges,
        max_random=max_random_starts,
        seed=seed,
        rcm_start=rcm_start,
    )

    t_build0 = time.perf_counter()
    model = build_exponential_model(
        normalized_edges,
        permutation_weight=permutation_weight,
        base=exponential_base,
        max_distance=exponential_max_distance,
    )
    model_build_s = time.perf_counter() - t_build0

    if backend == "tabu":
        result = solve_model_with_tsq_fast(
            model=model,
            edges=normalized_edges,
            method="exponential",
            initial_orderings=initial_orderings,
            tabu_tenure=tabu_tenure,
            cutoff=cutoff,
            fixed_weight=n,
            include_auxiliary_initialization=False,
        )
        result.model_build_s = model_build_s
        result.elapsed_s = result.model_build_s + result.matrix_assembly_s + result.solver_core_s
        result.scan_count = 1
        return result

    if backend == "gurobi":
        result = solve_model_with_gurobi(
            model=model,
            edges=normalized_edges,
            method="exponential",
            initial_orderings=initial_orderings,
            include_auxiliary_initialization=False,
            time_limit_s=gurobi_time_limit_s,
            mip_gap=gurobi_mip_gap,
        )
        result.model_build_s = model_build_s
        result.elapsed_s += model_build_s
        result.scan_count = 1
        return result

    raise ValueError(f"Unknown backend {backend!r}. Use 'tabu' or 'gurobi'.")


def solve_optimization_guarded(
    edges: EdgeList,
    backend: str = "tabu",
    max_random_starts: int = 1,
    tabu_tenure: int = 8,
    cutoff: int = 50,
    seed: Optional[int] = 123,
    rcm_start: bool = False,
    permutation_weight: float = 200.0,
    and_weight: float = 200.0,
    bandwidth_constraint_weight: float = 100.0,
    minimize_k_weight: float = 1.0,
    max_variables: int = 900,
    gurobi_time_limit_s: Optional[float] = None,
    gurobi_mip_gap: Optional[float] = None,
) -> BandwidthResult:
    """
    Guarded one-shot optimization formulation.

    This formulation is usually too large for dense tabu search once n and m
    are moderate, because it introduces O(m*n^2) z variables.
    """
    vertices, normalized_edges = normalize_edges(edges)
    n = len(vertices)
    m = len(normalized_edges)

    bits = max(1, ceil(log2(n)))
    estimated_total = n * n + m * n * n + m * bits + bits

    if estimated_total > max_variables:
        raise RuntimeError(
            "The optimization formulation is too large for dense TSQUBO. "
            f"Estimated variables: {estimated_total}. "
            f"Limit: {max_variables}. "
            "Use method='decision' or method='exponential'."
        )

    initial_orderings = candidate_initial_orderings(
        normalized_edges,
        max_random=max_random_starts,
        seed=seed,
        rcm_start=rcm_start,
    )

    t_build0 = time.perf_counter()
    model = build_optimization_model(
        normalized_edges,
        permutation_weight=permutation_weight,
        and_weight=and_weight,
        bandwidth_constraint_weight=bandwidth_constraint_weight,
        minimize_k_weight=minimize_k_weight,
    )
    model_build_s = time.perf_counter() - t_build0

    if backend == "tabu":
        result = solve_model_with_tsq_fast(
            model=model,
            edges=normalized_edges,
            method="optimization",
            initial_orderings=initial_orderings,
            tabu_tenure=tabu_tenure,
            cutoff=cutoff,
            fixed_weight=None,
            include_auxiliary_initialization=True,
        )
        result.model_build_s = model_build_s
        result.elapsed_s = result.model_build_s + result.matrix_assembly_s + result.solver_core_s
        result.scan_count = 1
        return result

    if backend == "gurobi":
        result = solve_model_with_gurobi(
            model=model,
            edges=normalized_edges,
            method="optimization",
            initial_orderings=initial_orderings,
            include_auxiliary_initialization=True,
            time_limit_s=gurobi_time_limit_s,
            mip_gap=gurobi_mip_gap,
        )
        result.model_build_s = model_build_s
        result.elapsed_s += model_build_s
        result.scan_count = 1
        return result

    raise ValueError(f"Unknown backend {backend!r}. Use 'tabu' or 'gurobi'.")


def solve_bandwidth_qubo_tsq_fast(
    edges: EdgeList,
    method: str = "decision",
    backend: str = "tabu",
    max_random_starts: int = 4,
    tabu_tenure: int = 12,
    cutoff: int = 150,
    seed: Optional[int] = 123,
    rcm_start: bool = False,
    exponential_base: Optional[float] = None,
    exponential_max_distance: Optional[int] = None,
    gurobi_time_limit_s: Optional[float] = None,
    gurobi_mip_gap: Optional[float] = None,
) -> BandwidthResult:
    """
    Recommended entry point.

    method:
        decision      practical default
        exponential   one-shot practical alternative
        optimization  guarded; only for very small graphs
    """
    if method == "decision":
        return solve_decision_fast(
            edges,
            backend=backend,
            max_random_starts=max_random_starts,
            tabu_tenure=tabu_tenure,
            cutoff=cutoff,
            seed=seed,
            rcm_start=rcm_start,
            gurobi_time_limit_s=gurobi_time_limit_s,
            gurobi_mip_gap=gurobi_mip_gap,
        )

    if method == "exponential":
        return solve_exponential_fast(
            edges,
            backend=backend,
            max_random_starts=max_random_starts,
            tabu_tenure=tabu_tenure,
            cutoff=cutoff,
            seed=seed,
            rcm_start=rcm_start,
            exponential_base=exponential_base,
            exponential_max_distance=exponential_max_distance,
            gurobi_time_limit_s=gurobi_time_limit_s,
            gurobi_mip_gap=gurobi_mip_gap,
        )

    if method == "optimization":
        return solve_optimization_guarded(
            edges,
            backend=backend,
            max_random_starts=min(max_random_starts, 1),
            tabu_tenure=min(tabu_tenure, 8),
            cutoff=min(cutoff, 50),
            seed=seed,
            rcm_start=rcm_start,
            gurobi_time_limit_s=gurobi_time_limit_s,
            gurobi_mip_gap=gurobi_mip_gap,
        )

    raise ValueError(
        f"Unknown method {method!r}. "
        "Use 'decision', 'exponential', or 'optimization'."
    )


def ordering_to_position(ordering: List[int]) -> Dict[int, int]:
    return {v: i for i, v in enumerate(ordering)}


def format_qubo_size_plot(num_variables: int, unit: int = 10, max_width: int = 60) -> str:
    if num_variables < 0:
        raise ValueError("num_variables must be non-negative")
    if unit <= 0:
        raise ValueError("unit must be positive")

    bar_len = min(max_width, max(1, ceil(num_variables / unit))) if num_variables else 0
    bar = "#" * bar_len
    return f"[{bar}] {num_variables} vars (1 # ~= {unit} vars)"


def print_result(
    result: BandwidthResult,
    edges: EdgeList,
) -> None:
    vertices, normalized_edges = normalize_edges(edges)
    original_position = {v: v for v in vertices}
    original_bandwidth = compute_bandwidth(normalized_edges, original_position)
    original_avg_range = compute_avg_range(normalized_edges, original_position)
    recomputed_bandwidth = compute_bandwidth(edges, result.position)
    recomputed_avg_range = compute_avg_range(edges, result.position)
    print(f"method: {result.method}")
    print(f"old vertex -> assigned position: {result.position}")
    print(f"bandwidth: {recomputed_bandwidth} (avg_range: {recomputed_avg_range:.6g})")
    print(
        "bandwidth change vs original ordering: "
        f"{recomputed_bandwidth - original_bandwidth:+d} "
        f"(original: {original_bandwidth}, avg_range: {original_avg_range:.6g})"
    )
    print(f"energy: {result.energy:.6g}")
    print(f"valid permutation from solver: {result.feasible_permutation}")
    print(f"number of QUBO variables: {len(result.model.variables)}")
    print(f"QUBO size plot: {format_qubo_size_plot(len(result.model.variables))}")
    print(f"elapsed_s: {result.elapsed_s:.6f}")
    print(
        "timing breakdown (s): "
        f"build={result.model_build_s:.6f}, "
        f"matrix={result.matrix_assembly_s:.6f}, "
        f"solver={result.solver_core_s:.6f}"
    )
    print(f"search effort: scan_count={result.scan_count}, solver_runs={result.solver_runs}")
    print("matrix multiplications: n/a for tabu backend (incremental QUBO updates, not dense matmul)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=str, default="C12")
    parser.add_argument(
        "--method",
        type=str,
        default="decision",
        choices=["decision", "exponential", "optimization", "all"],
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="tabu",
        choices=["tabu", "gurobi"],
    )
    parser.add_argument(
        "--implementation",
        type=str,
        default="python",
        choices=["python", "cpp"],
    )
    parser.add_argument("--cpp-binary", type=Path, default=Path("bandwidth_qubo_tsq_cpp"))
    parser.add_argument(
        "--matrix-backend",
        type=str,
        default="map",
        choices=["map", "eigen-sparse"],
    )
    parser.add_argument("--random-starts", type=int, default=4)
    parser.add_argument("--tabu-tenure", type=int, default=12)
    parser.add_argument("--cutoff", type=int, default=150)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--rcm-start", action="store_true")
    parser.add_argument("--exponential-base", type=float, default=None)
    parser.add_argument("--exponential-max-distance", type=int, default=None)
    parser.add_argument("--gurobi-time-limit", type=float, default=None)
    parser.add_argument("--gurobi-mip-gap", type=float, default=None)
    parser.add_argument("--rcm", action="store_true")
    args = parser.parse_args()

    if args.cluster not in CLUSTER_EDGES:
        available = ", ".join(sorted(CLUSTER_EDGES))
        raise KeyError(
            f"Unknown cluster {args.cluster!r}. "
            f"Available clusters: {available}"
        )

    edges = CLUSTER_EDGES[args.cluster]
    methods = ["decision", "exponential", "optimization"] if args.method == "all" else [args.method]

    vertices, normalized_edges = normalize_edges(edges)
    if args.implementation == "cpp":
        if args.backend != "tabu":
            raise ValueError("The C++ implementation currently supports only backend='tabu'.")
    else:
        print(f"cluster: {args.cluster}")
        print(f"number of edges: {len(edges)}")
        print(f"number of vertices: {len(vertices)}")
        print(f"simple bandwidth lower bound: {bandwidth_lower_bound(normalized_edges)}")
        original_position = {v: v for v in vertices}
        print(
            "original ordering bandwidth: "
            f"{compute_bandwidth(normalized_edges, original_position)} "
            f"(avg_range: {compute_avg_range(normalized_edges, original_position):.6g})"
        )
        print(f"implementation: python")
        print(f"backend: {args.backend}")

    if args.rcm and args.implementation != "cpp":
        print()
        print("=" * 80)
        print("RCM baseline")

        try:
            rcm_order = reverse_cuthill_mckee_ordering(normalized_edges)
            rcm_pos = ordering_to_position(rcm_order)
            rcm_bw = compute_bandwidth(normalized_edges, rcm_pos)

            print(f"ordering, position -> vertex: {rcm_order}")
            print(f"position, vertex -> position: {rcm_pos}")
            print(f"bandwidth: {rcm_bw} (avg_range: {compute_avg_range(normalized_edges, rcm_pos):.6g})")

        except ImportError:
            print("scipy not installed; skipping RCM baseline")

    for method in methods:
        try:
            if args.implementation == "cpp":
                output = run_cpp_solver(
                    cpp_binary=args.cpp_binary,
                    cluster=args.cluster,
                    method=method,
                    random_starts=args.random_starts,
                    tabu_tenure=args.tabu_tenure,
                    cutoff=args.cutoff,
                    seed=args.seed,
                    rcm_start=args.rcm_start,
                    rcm=args.rcm,
                    exponential_base=args.exponential_base,
                    exponential_max_distance=args.exponential_max_distance,
                    matrix_backend=args.matrix_backend,
                )
                print(output.strip())
            else:
                print()
                print("=" * 80)
                print(f"QUBO method: {method}")
                result = solve_bandwidth_qubo_tsq_fast(
                    normalized_edges,
                    method=method,
                    backend=args.backend,
                    max_random_starts=args.random_starts,
                    tabu_tenure=args.tabu_tenure,
                    cutoff=args.cutoff,
                    seed=args.seed,
                    rcm_start=args.rcm_start,
                    exponential_base=args.exponential_base,
                    exponential_max_distance=args.exponential_max_distance,
                    gurobi_time_limit_s=args.gurobi_time_limit,
                    gurobi_mip_gap=args.gurobi_mip_gap,
                )
                print_result(result, normalized_edges)

        except RuntimeError as exc:
            print(f"skipped: {exc}")

if __name__ == "__main__":
    main()
