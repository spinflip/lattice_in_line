from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from math import ceil
from typing import Dict, List, Tuple, Set, Optional, Callable, Iterable

from ..cluster_edges import CLUSTER_EDGES


Edge = Tuple[int, int]
EdgeList = List[Edge]
Adjacency = List[Set[int]]


@dataclass
class BandwidthAlgorithmResult:
    name: str
    ordering: List[int]              # position -> vertex
    position: Dict[int, int]         # vertex -> position
    bandwidth: int
    optimal: bool
    elapsed_s: float
    nodes_searched: int = 0
    timed_out: bool = False
    note: str = ""


@dataclass
class RecognitionResult:
    feasible: bool
    ordering: Optional[List[int]]
    nodes_searched: int
    timed_out: bool


@dataclass
class ExactSearchConfig:
    name: str
    position_strategy: str           # "left_to_right" or "perimeter"
    candidate_strategy: str          # "degree", "tight", "domain"
    node_limit: int = 2_000_000
    time_limit_s: float = 30.0
    use_forward_check: bool = True


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


def build_adjacency(edges: EdgeList) -> Adjacency:
    vertices, normalized = normalize_edges(edges)
    n = len(vertices)

    adj: Adjacency = [set() for _ in range(n)]
    for u, v in normalized:
        adj[u].add(v)
        adj[v].add(u)

    return adj


def compute_bandwidth(edges: EdgeList, ordering: List[int]) -> int:
    _, normalized = normalize_edges(edges)
    position = {v: i for i, v in enumerate(ordering)}

    if not normalized:
        return 0

    return max(abs(position[u] - position[v]) for u, v in normalized)


def compute_envelope(edges: EdgeList, ordering: List[int]) -> float:
    _, normalized = normalize_edges(edges)
    position = {v: i for i, v in enumerate(ordering)}

    if not normalized:
        return 0.0

    total = 0
    for u, v in normalized:
        total += abs(position[u] - position[v])

    return total / len(normalized)


def ordering_to_position(ordering: List[int]) -> Dict[int, int]:
    return {v: i for i, v in enumerate(ordering)}


def bandwidth_lower_bound(edges: EdgeList) -> int:
    """
    Simple lower bound:
        bandwidth >= ceil(max_degree / 2)

    This is weak, but cheap and safe.
    """
    adj = build_adjacency(edges)

    if not adj:
        return 0

    max_degree = max(len(nei) for nei in adj)
    return ceil(max_degree / 2)


def connected_components(adj: Adjacency) -> List[List[int]]:
    n = len(adj)
    seen = [False] * n
    comps: List[List[int]] = []

    for start in range(n):
        if seen[start]:
            continue

        q = deque([start])
        seen[start] = True
        comp = []

        while q:
            u = q.popleft()
            comp.append(u)

            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)

        comps.append(comp)

    return comps


def bfs_levels(adj: Adjacency, root: int) -> List[List[int]]:
    n = len(adj)
    dist = [-1] * n
    dist[root] = 0

    q = deque([root])
    while q:
        u = q.popleft()

        for v in sorted(adj[u], key=lambda x: len(adj[x])):
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                q.append(v)

    max_dist = max(dist)
    levels: List[List[int]] = [[] for _ in range(max_dist + 1)]

    for v, d in enumerate(dist):
        if d >= 0:
            levels[d].append(v)

    return levels


def pseudo_peripheral_vertex(adj: Adjacency, start: Optional[int] = None) -> int:
    """
    George-Liu style pseudo-peripheral search.
    """
    n = len(adj)

    if n == 0:
        raise ValueError("empty graph")

    if start is None:
        start = min(range(n), key=lambda v: len(adj[v]))

    root = start
    old_depth = -1

    while True:
        levels = bfs_levels(adj, root)
        depth = len(levels)

        last_level = levels[-1]
        candidate = min(last_level, key=lambda v: len(adj[v]))

        if depth <= old_depth:
            return root

        root = candidate
        old_depth = depth


def cuthill_mckee_ordering(edges: EdgeList) -> List[int]:
    """
    Cuthill-McKee heuristic.

    Handles disconnected graphs by restarting from a low-degree unvisited vertex.
    """
    adj = build_adjacency(edges)
    n = len(adj)
    visited = [False] * n
    order: List[int] = []

    while len(order) < n:
        start = min(
            (v for v in range(n) if not visited[v]),
            key=lambda v: len(adj[v]),
        )

        q = deque([start])
        visited[start] = True

        while q:
            u = q.popleft()
            order.append(u)

            nbrs = [v for v in adj[u] if not visited[v]]
            nbrs.sort(key=lambda v: len(adj[v]))

            for v in nbrs:
                visited[v] = True
                q.append(v)

    return order


def reverse_cuthill_mckee_ordering(edges: EdgeList) -> List[int]:
    """
    Reverse Cuthill-McKee.

    This implementation avoids scipy so the comparison file has no dependency
    beyond CLUSTER_EDGES.
    """
    return list(reversed(cuthill_mckee_ordering(edges)))


def gps_ordering(edges: EdgeList) -> List[int]:
    """
    Gibbs-Poole-Stockmeyer-style bandwidth heuristic.

    This is a practical GPS-style implementation:
      1. find pseudo-peripheral endpoints,
      2. build level structures from both endpoints,
      3. order level by level with low-degree tie-breaking,
      4. keep the better of both endpoint orientations and reversals.

    It is intentionally dependency-free.
    """
    adj = build_adjacency(edges)
    n = len(adj)

    if n == 0:
        return []

    orderings: List[List[int]] = []

    for comp in connected_components(adj):
        if not comp:
            continue

        start = min(comp, key=lambda v: len(adj[v]))
        u = pseudo_peripheral_vertex(adj, start)
        v = pseudo_peripheral_vertex(adj, u)

        for root in [u, v]:
            levels = bfs_levels(adj, root)
            order = []

            for level in levels:
                level_sorted = sorted(
                    level,
                    key=lambda x: (
                        len(adj[x]),
                        sum(1 for y in adj[x] if y in order),
                        x,
                    ),
                )
                order.extend(level_sorted)

            if len(order) == n:
                orderings.append(order)
                orderings.append(list(reversed(order)))

    if not orderings:
        return reverse_cuthill_mckee_ordering(edges)

    return min(orderings, key=lambda o: compute_bandwidth(edges, o))


def initial_upper_bound_orderings(edges: EdgeList) -> List[List[int]]:
    n = len(build_adjacency(edges))
    identity = list(range(n))
    rcm = reverse_cuthill_mckee_ordering(edges)
    gps = gps_ordering(edges)

    orderings = [
        identity,
        list(reversed(identity)),
        rcm,
        list(reversed(rcm)),
        gps,
        list(reversed(gps)),
    ]

    seen = set()
    unique = []

    for o in orderings:
        key = tuple(o)
        if key not in seen:
            unique.append(o)
            seen.add(key)

    return unique


def best_heuristic_ordering(edges: EdgeList) -> List[int]:
    candidates = initial_upper_bound_orderings(edges)
    return min(candidates, key=lambda o: compute_bandwidth(edges, o))


def position_order(n: int, strategy: str) -> List[int]:
    if strategy == "left_to_right":
        return list(range(n))

    if strategy == "perimeter":
        order = []
        lo = 0
        hi = n - 1

        while lo <= hi:
            order.append(lo)
            if lo != hi:
                order.append(hi)
            lo += 1
            hi -= 1

        return order

    raise ValueError(f"unknown position strategy: {strategy!r}")


def is_compatible(
    v: int,
    pos: int,
    assigned_pos: Dict[int, int],
    adj: Adjacency,
    k: int,
) -> bool:
    for u in adj[v]:
        pu = assigned_pos.get(u)
        if pu is not None and abs(pos - pu) > k:
            return False

    return True


def forward_check(
    unplaced: Set[int],
    remaining_positions: Set[int],
    assigned_pos: Dict[int, int],
    adj: Adjacency,
    k: int,
) -> bool:
    """
    Safe feasibility check:
    every unplaced vertex must have at least one remaining position compatible
    with all already placed neighbors.
    """
    for v in unplaced:
        ok = False

        for pos in remaining_positions:
            if is_compatible(v, pos, assigned_pos, adj, k):
                ok = True
                break

        if not ok:
            return False

    return True


def candidate_vertices_for_position(
    pos: int,
    unplaced: Set[int],
    assigned_pos: Dict[int, int],
    remaining_positions: Set[int],
    adj: Adjacency,
    k: int,
    strategy: str,
) -> List[int]:
    candidates = [
        v for v in unplaced
        if is_compatible(v, pos, assigned_pos, adj, k)
    ]

    if strategy == "degree":
        candidates.sort(key=lambda v: (-len(adj[v]), v))
        return candidates

    if strategy == "tight":
        def tight_key(v: int) -> Tuple[int, int, int]:
            placed_neighbors = sum(1 for u in adj[v] if u in assigned_pos)
            unplaced_neighbors = sum(1 for u in adj[v] if u in unplaced)
            return (-placed_neighbors, -len(adj[v]), -unplaced_neighbors)

        candidates.sort(key=tight_key)
        return candidates

    if strategy == "domain":
        def domain_size(v: int) -> Tuple[int, int, int]:
            count = 0
            for p in remaining_positions:
                if is_compatible(v, p, assigned_pos, adj, k):
                    count += 1

            return (count, -len(adj[v]), v)

        candidates.sort(key=domain_size)
        return candidates

    raise ValueError(f"unknown candidate strategy: {strategy!r}")


def recognize_bandwidth_exact(
    edges: EdgeList,
    k: int,
    config: ExactSearchConfig,
) -> RecognitionResult:
    """
    Exact recognition via branch-and-bound/backtracking.

    Returns feasible=True iff it found an ordering with bandwidth <= k.
    If timed_out=True, feasible=False means only "not found within limits".
    """
    adj = build_adjacency(edges)
    n = len(adj)

    positions = position_order(n, config.position_strategy)

    assigned_at_pos: Dict[int, int] = {}
    assigned_pos: Dict[int, int] = {}
    unplaced: Set[int] = set(range(n))
    remaining_positions: Set[int] = set(range(n))

    start_time = time.perf_counter()
    nodes = 0
    timed_out = False

    def dfs(depth: int) -> bool:
        nonlocal nodes, timed_out

        nodes += 1

        if nodes >= config.node_limit:
            timed_out = True
            return False

        if time.perf_counter() - start_time >= config.time_limit_s:
            timed_out = True
            return False

        if depth == n:
            return True

        pos = positions[depth]

        candidates = candidate_vertices_for_position(
            pos=pos,
            unplaced=unplaced,
            assigned_pos=assigned_pos,
            remaining_positions=remaining_positions,
            adj=adj,
            k=k,
            strategy=config.candidate_strategy,
        )

        for v in candidates:
            assigned_at_pos[pos] = v
            assigned_pos[v] = pos
            unplaced.remove(v)
            remaining_positions.remove(pos)

            ok = True
            if config.use_forward_check:
                ok = forward_check(
                    unplaced=unplaced,
                    remaining_positions=remaining_positions,
                    assigned_pos=assigned_pos,
                    adj=adj,
                    k=k,
                )

            if ok and dfs(depth + 1):
                return True

            remaining_positions.add(pos)
            unplaced.add(v)
            del assigned_pos[v]
            del assigned_at_pos[pos]

            if timed_out:
                return False

        return False

    feasible = dfs(0)

    if feasible:
        ordering = [assigned_at_pos[i] for i in range(n)]
    else:
        ordering = None

    return RecognitionResult(
        feasible=feasible,
        ordering=ordering,
        nodes_searched=nodes,
        timed_out=timed_out,
    )


def exact_minimize_bandwidth(
    edges: EdgeList,
    config: ExactSearchConfig,
) -> BandwidthAlgorithmResult:
    """
    Minimize by recognition from lower bound upward.

    This is safe but can be expensive.
    """
    t0 = time.perf_counter()

    lb = bandwidth_lower_bound(edges)
    ub_order = best_heuristic_ordering(edges)
    ub = compute_bandwidth(edges, ub_order)

    total_nodes = 0
    any_timeout = False

    best_order = ub_order
    best_bw = ub

    for k in range(lb, ub + 1):
        rec = recognize_bandwidth_exact(edges, k, config)
        total_nodes += rec.nodes_searched

        if rec.timed_out:
            any_timeout = True
            break

        if rec.feasible and rec.ordering is not None:
            best_order = rec.ordering
            best_bw = compute_bandwidth(edges, best_order)

            return BandwidthAlgorithmResult(
                name=config.name,
                ordering=best_order,
                position=ordering_to_position(best_order),
                bandwidth=best_bw,
                optimal=True,
                elapsed_s=time.perf_counter() - t0,
                nodes_searched=total_nodes,
                timed_out=False,
                note=f"proved optimum by recognition at k={k}",
            )

    return BandwidthAlgorithmResult(
        name=config.name,
        ordering=best_order,
        position=ordering_to_position(best_order),
        bandwidth=best_bw,
        optimal=False,
        elapsed_s=time.perf_counter() - t0,
        nodes_searched=total_nodes,
        timed_out=any_timeout,
        note=(
            "returned best heuristic upper bound because exact recognition "
            "hit a time/node limit"
            if any_timeout
            else "returned best known ordering"
        ),
    )


def caprara_salazar_gonzalez(
    edges: EdgeList,
    node_limit: int = 2_000_000,
    time_limit_s: float = 30.0,
) -> BandwidthAlgorithmResult:
    """
    Practical exact-recognition/minimization variant inspired by the
    Caprara-Salazar-Gonzalez recognition family.

    Uses:
      - lower/upper bound bracketing,
      - forward-checking,
      - domain-size candidate ordering.
    """
    config = ExactSearchConfig(
        name="Caprara-Salazar-Gonzalez",
        position_strategy="left_to_right",
        candidate_strategy="domain",
        node_limit=node_limit,
        time_limit_s=time_limit_s,
        use_forward_check=True,
    )

    return exact_minimize_bandwidth(edges, config)


def del_corso_manzini(
    edges: EdgeList,
    node_limit: int = 2_000_000,
    time_limit_s: float = 30.0,
) -> BandwidthAlgorithmResult:
    """
    Practical Del Corso-Manzini-style exact depth-first search.
    """
    config = ExactSearchConfig(
        name="Del Corso-Manzini",
        position_strategy="left_to_right",
        candidate_strategy="tight",
        node_limit=node_limit,
        time_limit_s=time_limit_s,
        use_forward_check=True,
    )

    return exact_minimize_bandwidth(edges, config)


def del_corso_manzini_perimeter(
    edges: EdgeList,
    node_limit: int = 2_000_000,
    time_limit_s: float = 30.0,
) -> BandwidthAlgorithmResult:
    """
    Practical Del Corso-Manzini perimeter-search variant.

    The search assigns positions from the perimeter inward:
        0, n-1, 1, n-2, ...

    This preserves exactness because it only changes the branching order.
    """
    config = ExactSearchConfig(
        name="Del Corso-Manzini perimeter",
        position_strategy="perimeter",
        candidate_strategy="tight",
        node_limit=node_limit,
        time_limit_s=time_limit_s,
        use_forward_check=True,
    )

    return exact_minimize_bandwidth(edges, config)


def saxe_gurari_sudborough(
    edges: EdgeList,
    node_limit: int = 2_000_000,
    time_limit_s: float = 30.0,
) -> BandwidthAlgorithmResult:
    """
    Practical Saxe-Gurari-Sudborough-style recognition/minimization.

    Uses exact recognition with a minimum-domain branching rule.
    This is suitable for small/medium comparison cases.
    """
    config = ExactSearchConfig(
        name="Saxe-Gurari-Sudborough",
        position_strategy="left_to_right",
        candidate_strategy="domain",
        node_limit=node_limit,
        time_limit_s=time_limit_s,
        use_forward_check=True,
    )

    return exact_minimize_bandwidth(edges, config)


def gibbs_poole_stockmeyer(edges: EdgeList) -> BandwidthAlgorithmResult:
    t0 = time.perf_counter()
    ordering = gps_ordering(edges)
    bw = compute_bandwidth(edges, ordering)

    return BandwidthAlgorithmResult(
        name="Gibbs-Poole-Stockmeyer",
        ordering=ordering,
        position=ordering_to_position(ordering),
        bandwidth=bw,
        optimal=False,
        elapsed_s=time.perf_counter() - t0,
        note="heuristic",
    )


def reverse_cuthill_mckee(edges: EdgeList) -> BandwidthAlgorithmResult:
    t0 = time.perf_counter()
    ordering = reverse_cuthill_mckee_ordering(edges)
    bw = compute_bandwidth(edges, ordering)

    return BandwidthAlgorithmResult(
        name="Reverse Cuthill-McKee",
        ordering=ordering,
        position=ordering_to_position(ordering),
        bandwidth=bw,
        optimal=False,
        elapsed_s=time.perf_counter() - t0,
        note="heuristic",
    )


def run_algorithm(
    name: str,
    edges: EdgeList,
    node_limit: int,
    time_limit_s: float,
) -> BandwidthAlgorithmResult:
    key = name.lower().replace("_", "-")

    if key in {"csg", "caprara", "caprara-salazar-gonzalez"}:
        return caprara_salazar_gonzalez(
            edges,
            node_limit=node_limit,
            time_limit_s=time_limit_s,
        )

    if key in {"dcm", "del-corso-manzini"}:
        return del_corso_manzini(
            edges,
            node_limit=node_limit,
            time_limit_s=time_limit_s,
        )

    if key in {"dcm-perimeter", "del-corso-manzini-perimeter", "perimeter"}:
        return del_corso_manzini_perimeter(
            edges,
            node_limit=node_limit,
            time_limit_s=time_limit_s,
        )

    if key in {"sgs", "saxe-gurari-sudborough", "saxe"}:
        return saxe_gurari_sudborough(
            edges,
            node_limit=node_limit,
            time_limit_s=time_limit_s,
        )

    if key in {"gps", "gibbs-poole-stockmeyer"}:
        return gibbs_poole_stockmeyer(edges)

    if key in {"rcm", "reverse-cuthill-mckee"}:
        return reverse_cuthill_mckee(edges)

    raise ValueError(f"unknown algorithm: {name!r}")


def print_result(result: BandwidthAlgorithmResult, edges: EdgeList) -> None:
    print(f"algorithm: {result.name}")
    print(f"ordering, position -> vertex: {result.ordering}")
    print(f"position, vertex -> position: {result.position}")
    print(f"bandwidth: {result.bandwidth} (envelope: {compute_envelope(edges, result.ordering):.6g})")
    print(f"optimal: {result.optimal}")
    print(f"elapsed_s: {result.elapsed_s:.6f}")
    print(f"nodes_searched: {result.nodes_searched}")
    print(f"timed_out: {result.timed_out}")
    if result.note:
        print(f"note: {result.note}")


def print_summary_table(results: List[BandwidthAlgorithmResult], edges: EdgeList) -> None:
    print()
    print("=" * 100)
    print("summary")
    print("=" * 100)
    print(f"{'algorithm':36s} {'bw':>5s} {'env':>9s} {'optimal':>8s} {'time_s':>12s} {'nodes':>12s} note")
    print("-" * 100)

    for r in sorted(results, key=lambda x: (x.bandwidth, x.elapsed_s, x.name)):
        env = compute_envelope(edges, r.ordering)
        print(
            f"{r.name:36s} "
            f"{r.bandwidth:5d} "
            f"{env:9.6g} "
            f"{str(r.optimal):>8s} "
            f"{r.elapsed_s:12.6f} "
            f"{r.nodes_searched:12d} "
            f"{r.note}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=str, default="C12")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="all",
        choices=[
            "all",
            "csg",
            "dcm",
            "dcm-perimeter",
            "sgs",
            "gps",
            "rcm",
        ],
    )
    parser.add_argument("--node-limit", type=int, default=2_000_000)
    parser.add_argument("--time-limit", type=float, default=30.0)
    args = parser.parse_args()

    if args.cluster not in CLUSTER_EDGES:
        available = ", ".join(sorted(CLUSTER_EDGES))
        raise KeyError(
            f"unknown cluster {args.cluster!r}; available clusters: {available}"
        )

    edges = CLUSTER_EDGES[args.cluster]
    vertices, normalized_edges = normalize_edges(edges)

    print(f"cluster: {args.cluster}")
    print(f"vertices: {len(vertices)}")
    print(f"edges: {len(normalized_edges)}")
    print(f"simple lower bound: {bandwidth_lower_bound(normalized_edges)}")
    original_order = list(range(len(vertices)))
    print(
        "original ordering bandwidth: "
        f"{compute_bandwidth(normalized_edges, original_order)} "
        f"(envelope: {compute_envelope(normalized_edges, original_order):.6g})"
    )
    best_heur = best_heuristic_ordering(normalized_edges)
    print(
        "best heuristic upper bound: "
        f"{compute_bandwidth(normalized_edges, best_heur)} "
        f"(envelope: {compute_envelope(normalized_edges, best_heur):.6g})"
    )

    if args.algorithm == "all":
        algorithms = [
            "rcm",
            "gps",
            "dcm",
            "dcm-perimeter",
            "sgs",
            "csg",
        ]
    else:
        algorithms = [args.algorithm]

    results: List[BandwidthAlgorithmResult] = []

    for alg in algorithms:
        print()
        print("=" * 100)

        result = run_algorithm(
            name=alg,
            edges=normalized_edges,
            node_limit=args.node_limit,
            time_limit_s=args.time_limit,
        )

        print_result(result, normalized_edges)
        results.append(result)

    if len(results) > 1:
        print_summary_table(results, normalized_edges)


if __name__ == "__main__":
    main()
