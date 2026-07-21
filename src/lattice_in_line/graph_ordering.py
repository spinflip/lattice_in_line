"""Graph ordering helpers used by molecule model builders."""

from collections import deque
from typing import Dict, Iterable, List, Sequence, Set, Tuple


Edge = Tuple[int, int]
GraphOrdering = str
GRAPH_ORDERING_CHOICES = ("rcm", "king", "amd", "nd")


def _build_adjacency(n_sites: int, edges: Sequence[Edge]) -> List[set]:
    adjacency = [set() for _ in range(n_sites)]
    for i, j in edges:
        adjacency[i].add(j)
        adjacency[j].add(i)
    return adjacency


def _transform_from_order(order: Iterable[int], n_sites: int) -> List[int]:
    transform = [0] * n_sites
    for new_pos, old_pos in enumerate(order):
        transform[old_pos] = new_pos
    return transform


def _connected_components(
    adjacency: List[set],
    nodes: Set[int],
) -> List[List[int]]:
    remaining = set(nodes)
    components: List[List[int]] = []
    while remaining:
        root = min(remaining)
        queue = deque([root])
        remaining.remove(root)
        component = [root]
        while queue:
            site = queue.popleft()
            for nbr in sorted(adjacency[site]):
                if nbr in remaining:
                    remaining.remove(nbr)
                    queue.append(nbr)
                    component.append(nbr)
        components.append(component)
    components.sort(key=lambda comp: (len(comp), comp))
    return components


def _bfs_within(
    adjacency: List[set],
    allowed: Set[int],
    source: int,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    queue = deque([source])
    distance = {source: 0}
    parent = {source: source}
    while queue:
        site = queue.popleft()
        for nbr in sorted(adjacency[site]):
            if nbr not in allowed or nbr in distance:
                continue
            distance[nbr] = distance[site] + 1
            parent[nbr] = site
            queue.append(nbr)
    return distance, parent


def reverse_cuthill_mckee_permutation(
    n_sites: int,
    edges: Sequence[Edge],
    start: int = 0,
) -> List[int]:
    """Mirror the C++ ``compress_CuthillMcKee(..., true)`` default ordering.

    The C++ helper starts the Cuthill-McKee traversal from vertex 0 and then
    reverses the resulting order. For the connected fullerene graphs we use
    here, the simple BFS implementation below reproduces the same permutation.
    """
    adjacency = _build_adjacency(n_sites, edges)
    degree = [len(nbrs) for nbrs in adjacency]
    visited = [False] * n_sites
    ordering: List[int] = []

    starts = [start] + [site for site in range(n_sites) if site != start]
    for root in starts:
        if visited[root]:
            continue
        queue = deque([root])
        visited[root] = True
        while queue:
            site = queue.popleft()
            ordering.append(site)
            nbrs = sorted(
                (nbr for nbr in adjacency[site] if not visited[nbr]),
                    key=lambda nbr: (degree[nbr], nbr),
            )
            for nbr in nbrs:
                visited[nbr] = True
                queue.append(nbr)

    return _transform_from_order(reversed(ordering), n_sites)


def king_permutation(
    n_sites: int,
    edges: Sequence[Edge],
    start: int = 0,
) -> List[int]:
    """Return a King-style profile-reducing ordering.

    This follows the same low-degree level-set spirit as Cuthill-McKee, but
    reorders each frontier greedily so vertices most connected to the already
    ordered set are placed first while keeping the remaining degree small.
    """
    adjacency = _build_adjacency(n_sites, edges)
    degree = [len(nbrs) for nbrs in adjacency]
    visited = [False] * n_sites
    ordering: List[int] = []

    starts = [start] + [site for site in range(n_sites) if site != start]
    for root in starts:
        if visited[root]:
            continue

        frontier = [root]
        visited[root] = True
        ordering.append(root)

        while frontier:
            level = {
                nbr
                for site in frontier
                for nbr in adjacency[site]
                if not visited[nbr]
            }
            if not level:
                break

            level_order: List[int] = []
            while level:
                best = max(
                    level,
                    key=lambda node: (
                        sum(1 for nbr in adjacency[node] if nbr in frontier or nbr in level_order),
                        -sum(1 for nbr in adjacency[node] if not visited[nbr] and nbr not in level_order),
                        -degree[node],
                        -node,
                    ),
                )
                level.remove(best)
                visited[best] = True
                level_order.append(best)

            frontier = level_order
            ordering.extend(level_order)

    return _transform_from_order(ordering, n_sites)


def approximate_minimum_degree_permutation(
    n_sites: int,
    edges: Sequence[Edge],
    start: int = 0,
) -> List[int]:
    """Return a minimum-degree-style elimination ordering.

    For the small fullerene graphs used here, we maintain the filled graph
    explicitly and repeatedly eliminate the currently smallest-degree vertex.
    This mirrors the intent of AMD while staying compact and dependency-free.
    """
    adjacency = _build_adjacency(n_sites, edges)
    active = [True] * n_sites
    order: List[int] = []

    for _ in range(n_sites):
        remaining = [site for site in range(n_sites) if active[site]]
        if not remaining:
            break

        chosen = min(
            remaining,
            key=lambda site: (
                sum(1 for nbr in adjacency[site] if active[nbr]),
                0 if site == start else 1,
                site,
            ),
        )

        neighbors = [nbr for nbr in adjacency[chosen] if active[nbr]]
        for i, left in enumerate(neighbors):
            for right in neighbors[i + 1:]:
                adjacency[left].add(right)
                adjacency[right].add(left)

        for nbr in neighbors:
            adjacency[nbr].discard(chosen)
        adjacency[chosen].clear()
        active[chosen] = False
        order.append(chosen)

    return _transform_from_order(order, n_sites)


def nested_dissection_permutation(
    n_sites: int,
    edges: Sequence[Edge],
    start: int = 0,
) -> List[int]:
    """Return a nested-dissection-style ordering.

    We recursively choose a small separator from a BFS level near the middle
    of a pseudo-diameter, order the disconnected components first, and place
    the separator last.
    """
    adjacency = _build_adjacency(n_sites, edges)

    def choose_separator(nodes: Set[int]) -> List[int]:
        seed = start if start in nodes else min(nodes)
        dist_seed, _ = _bfs_within(adjacency, nodes, seed)
        left = max(dist_seed, key=lambda site: (dist_seed[site], -site))
        dist_left, _ = _bfs_within(adjacency, nodes, left)
        if len(dist_left) != len(nodes):
            # Graph subset is unexpectedly disconnected; let the caller split it.
            return []

        levels: Dict[int, List[int]] = {}
        for site, dist in dist_left.items():
            levels.setdefault(dist, []).append(site)
        for level_nodes in levels.values():
            level_nodes.sort()

        total = len(nodes)
        prefix = 0
        best_level = None
        best_score = None
        for level in sorted(levels):
            width = len(levels[level])
            left_size = prefix
            right_size = total - prefix - width
            score = (abs(left_size - right_size), width, level)
            if best_score is None or score < best_score:
                best_score = score
                best_level = level
            prefix += width

        separator = list(levels.get(best_level, []))
        if not separator:
            return [min(nodes)]
        return separator

    def recurse(component_nodes: Set[int]) -> List[int]:
        if not component_nodes:
            return []
        if len(component_nodes) <= 4:
            return sorted(component_nodes)

        components = _connected_components(adjacency, component_nodes)
        if len(components) > 1:
            ordered: List[int] = []
            for comp in components:
                ordered.extend(recurse(set(comp)))
            return ordered

        separator = choose_separator(component_nodes)
        if not separator:
            return sorted(component_nodes)

        remainder = set(component_nodes) - set(separator)
        split_components = _connected_components(adjacency, remainder)
        if len(split_components) <= 1:
            # Fallback: choose one low-degree vertex as a minimal separator.
            fallback = min(
                component_nodes,
                key=lambda site: (len(adjacency[site] & component_nodes), site),
            )
            remainder = set(component_nodes)
            remainder.remove(fallback)
            split_components = _connected_components(adjacency, remainder)
            separator = [fallback]

        ordered = []
        for comp in split_components:
            ordered.extend(recurse(set(comp)))
        ordered.extend(sorted(separator))
        return ordered

    return _transform_from_order(recurse(set(range(n_sites))), n_sites)


def graph_ordering_permutation(
    n_sites: int,
    edges: Sequence[Edge],
    *,
    method: GraphOrdering = "rcm",
    start: int = 0,
) -> List[int]:
    """Dispatch graph ordering by method name."""
    method_norm = method.lower()
    if method_norm == "rcm":
        return reverse_cuthill_mckee_permutation(n_sites, edges, start=start)
    if method_norm == "king":
        return king_permutation(n_sites, edges, start=start)
    if method_norm == "amd":
        return approximate_minimum_degree_permutation(n_sites, edges, start=start)
    if method_norm == "nd":
        return nested_dissection_permutation(n_sites, edges, start=start)
    raise ValueError(
        f"Unsupported graph ordering {method!r}. Supported: {', '.join(GRAPH_ORDERING_CHOICES)}"
    )
