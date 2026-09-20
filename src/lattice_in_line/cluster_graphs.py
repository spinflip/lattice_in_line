"""Cluster graph tables and helpers.

At the moment this module ships fullerene clusters extracted from the C++
model definitions, but the naming is intentionally general so it can be
extended to other finite clusters such as rings or tori.
"""
from __future__ import annotations

import ast
import inspect
import importlib.util
import json
import math
import re
import warnings
from collections import deque
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

try:
    import tomllib
except ImportError:  # pragma: no cover - Python < 3.11 fallback.
    tomllib = None

import numpy as np

from .cluster_edges import CLUSTER_EDGES, CLUSTER_EDGE_LAYERS
from .graph_ordering import GRAPH_ORDERING_CHOICES, graph_ordering_permutation


EdgeList = List[Tuple[int, int]]
WeightedEdgeList = List[Tuple[int, int, float]]
_INTER_CELL_SECTION_RE = re.compile(r"^inter_cell(?:_(\d+))?$")


class MissingPermutationEntry(ValueError):
    """Raised when an optional permutation table has no entry for a graph."""


# Old permutation-table filenames -> their stratified successors, so callers
# using the pre-rename names keep resolving to the renamed files.
_PERMUTATION_FILE_ALIASES = {
    "permutations_sat.py": "permutations_sat_bw.py",
    "permutations_sat_cutwidth.py": "permutations_sat_cw.py",
    "permutations_qubo.py": "permutations_qubo_bw.py",
}


def resolve_geometry_path(path) -> Path:
    """Resolve plain paths plus legacy ``geometry/...`` paths into this package."""
    raw = Path(path).expanduser()
    if raw.is_file():
        return raw
    base = Path(__file__).resolve().parent

    def _match(candidate: Path) -> Path | None:
        if candidate.is_file():
            return candidate
        if candidate.suffix == ".py":
            aliased = candidate.with_name(candidate.name.replace("-", "_"))
            if aliased.is_file():
                return aliased
            # The permutation tables were stratified/renamed (bandwidth vs
            # cutwidth); accept the old names as aliases for their successors.
            renamed = _PERMUTATION_FILE_ALIASES.get(candidate.name)
            if renamed is not None and (candidate.with_name(renamed)).is_file():
                return candidate.with_name(renamed)
        return None

    if len(raw.parts) == 1:
        matched = _match(base / raw.name)
        if matched is not None:
            return matched
    parts = raw.parts
    # Accept a package-dir prefix and resolve the tail against this package.
    for prefix in ("lattice_in_line", "geometry"):
        if prefix in parts:
            idx = parts.index(prefix)
            matched = _match(base.joinpath(*parts[idx + 1:]))
            if matched is not None:
                return matched
    return raw


def cluster_names() -> List[str]:
    return list(CLUSTER_EDGES.keys())


def cluster_size(name: str) -> int:
    if name not in CLUSTER_EDGES:
        raise ValueError(f"Unsupported cluster {name!r}")
    max_index = max(max(i, j) for i, j in CLUSTER_EDGES[name])
    return max_index + 1


def _load_python_permutation_payload(path: Path) -> object:
    spec = importlib.util.spec_from_file_location(f"vmps_permutation_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not import permutation file {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for key in ("CUSTOM_PERMUTATIONS", "PERMUTATIONS", "permutations"):
        payload = getattr(module, key, None)
        if payload is not None:
            return payload
    build = getattr(module, "build", None)
    if callable(build):
        return build()
    raise ValueError(
        f"Python permutation file {path} must define CUSTOM_PERMUTATIONS, "
        "PERMUTATIONS, permutations, or a callable build()"
    )


def _normalize_permutation(raw, n_sites: int, *, source: str) -> List[int]:
    if isinstance(raw, Mapping):
        try:
            perm = [int(raw[idx]) for idx in range(n_sites)]
        except KeyError as exc:
            raise ValueError(f"{source} is missing a mapping for vertex {exc.args[0]}") from exc
    elif isinstance(raw, (list, tuple)):
        perm = [int(val) for val in raw]
    else:
        raise ValueError(f"{source} must be a list/tuple or a map from old->new indices")

    if len(perm) != n_sites:
        raise ValueError(f"{source} has length {len(perm)}, expected {n_sites}")
    if sorted(perm) != list(range(n_sites)):
        raise ValueError(f"{source} is not a valid permutation of 0..{n_sites - 1}")
    return perm


def _load_custom_ordering_permutation(
    ordering_path: Path,
    n_sites: int,
    *,
    key: str,
) -> List[int]:
    payload = _load_python_permutation_payload(ordering_path)

    if isinstance(payload, Mapping):
        if key in payload:
            return _normalize_permutation(payload[key], n_sites, source=f"{ordering_path.name}:{key}")
        if str(key) in payload:
            return _normalize_permutation(payload[str(key)], n_sites, source=f"{ordering_path.name}:{key}")
        int_keys = all(isinstance(item, int) for item in payload.keys())
        if int_keys and len(payload) == n_sites:
            return _normalize_permutation(payload, n_sites, source=ordering_path.name)
        if len(payload) == 1:
            only_key = next(iter(payload))
            return _normalize_permutation(payload[only_key], n_sites, source=f"{ordering_path.name}:{only_key}")
        available = ", ".join(str(name) for name in sorted(payload))
        raise MissingPermutationEntry(
            f"Permutation file {ordering_path} does not define an entry for {key!r}. "
            f"Available entries: {available}"
        )

    return _normalize_permutation(payload, n_sites, source=ordering_path.name)


def _resolve_ordering_permutation(
    n_sites: int,
    edges: EdgeList,
    *,
    ordering: str,
    key: str,
) -> Tuple[List[int], str]:
    ordering_text = str(ordering).strip()
    if ordering_text.lower() in {"none", "identity", "raw"}:
        return list(range(n_sites)), ordering_text.lower()
    ordering_path = resolve_geometry_path(ordering_text)
    if ordering_path.is_file():
        try:
            return (
                _load_custom_ordering_permutation(ordering_path, n_sites, key=key),
                ordering_path.name,
            )
        except MissingPermutationEntry as exc:
            fallback = "rcm"
            print(f"WARNING: {exc}; falling back to {fallback} ordering")
            return (
                graph_ordering_permutation(n_sites, edges, method=fallback, start=0),
                f"{fallback} (fallback from {ordering_path.name})",
            )
    if ordering_path.suffix and ordering_text.lower() not in GRAPH_ORDERING_CHOICES:
        raise ValueError(f"Ordering file {ordering_path} does not exist")
    return (
        graph_ordering_permutation(n_sites, edges, method=ordering_text, start=0),
        ordering_text,
    )


def _layout_stats(edges: Sequence[Tuple[int, int]]) -> Tuple[int, float, int]:
    spans = [abs(int(j) - int(i)) for i, j in edges]
    max_site = max((max(int(i), int(j)) for i, j in edges), default=0)
    cut_width = max(
        (
            sum(1 for i, j in edges if min(int(i), int(j)) <= cut < max(int(i), int(j)))
            for cut in range(max_site)
        ),
        default=0,
    )
    bandwidth = max(spans, default=0)
    # Mean edge length = average interaction range R (formerly "envelope").
    avg_range = (sum(spans) / len(spans)) if spans else 0.0
    return bandwidth, avg_range, cut_width


def cluster_ordering_permutation(name: str, *, ordering: str = "rcm") -> List[int]:
    """Return the site permutation mapping original indices to reordered indices."""
    if name not in CLUSTER_EDGES:
        raise ValueError(
            f"Unsupported cluster {name!r}. Supported: {', '.join(cluster_names())}"
        )
    perm, _ = _resolve_ordering_permutation(
        cluster_size(name),
        CLUSTER_EDGES[name],
        ordering=ordering,
        key=name,
    )
    return perm


def cluster_edge_layers(name: str, *, ordering: str = "rcm") -> List[EdgeList]:
    """Return the three disjoint heavy-hex edge-color layers in MPS order."""
    if name in CLUSTER_EDGE_LAYERS:
        raw_layers = CLUSTER_EDGE_LAYERS[name]
    elif name.startswith("heavyHex"):
        from .heavy_hex_generator import edge_layers, patch_for_cluster

        try:
            raw_layers = edge_layers(patch_for_cluster(name))
        except ValueError as error:
            supported = ", ".join(CLUSTER_EDGE_LAYERS)
            raise ValueError(
                f"No edge layers for cluster {name!r}. Supported: {supported}"
            ) from error
    else:
        supported = ", ".join(CLUSTER_EDGE_LAYERS)
        raise ValueError(f"No edge layers for cluster {name!r}. Supported: {supported}")
    if len(raw_layers) != 3:
        raise ValueError(f"Expected three edge layers for {name}, got {len(raw_layers)}")
    expected = {tuple(sorted(edge)) for edge in CLUSTER_EDGES[name]}
    colored = []
    for layer_index, layer in enumerate(raw_layers, start=1):
        sites = set()
        for edge in layer:
            normalized = tuple(sorted(edge))
            if normalized in colored:
                raise ValueError(f"Duplicate edge {normalized} in {name} layer {layer_index}")
            if normalized[0] in sites or normalized[1] in sites:
                raise ValueError(f"Layer {layer_index} of {name} is not a matching")
            sites.update(normalized)
            colored.append(normalized)
    if set(colored) != expected or len(colored) != len(expected):
        raise ValueError(f"Edge layers for {name} do not exactly cover its cluster edges")

    permutation = cluster_ordering_permutation(name, ordering=ordering)
    relabeled_layers = []
    for layer in raw_layers:
        relabeled = [tuple(sorted((permutation[i], permutation[j]))) for i, j in layer]
        relabeled_layers.append(sorted(relabeled))
    return relabeled_layers


def _ordered_cluster_edges(
    name: str,
    *,
    ordering: str = "rcm",
    verbose: bool = False,
) -> Tuple[int, EdgeList]:
    if name not in CLUSTER_EDGES:
        raise ValueError(
            f"Unsupported cluster {name!r}. Supported: {', '.join(cluster_names())}"
        )

    L = cluster_size(name)
    edges = CLUSTER_EDGES[name]
    perm, ordering_label = _resolve_ordering_permutation(L, edges, ordering=ordering, key=name)
    inverse = [0] * L
    for old, new in enumerate(perm):
        inverse[old] = new

    relabeled = []
    for i, j in edges:
        ii, jj = inverse[i], inverse[j]
        if ii > jj:
            ii, jj = jj, ii
        relabeled.append((ii, jj))
    relabeled.sort()

    if verbose:
        perm_str = ", ".join(f"{i}\u2192{j}" for i, j in enumerate(perm))
        print(f"{name} ordering ({ordering_label}): {perm_str}")
        bandwidth, avg_range, cut_width = _layout_stats(relabeled)
        print(f"{name} bandwidth={bandwidth}, avg_range={avg_range:.2f}, cut_width={cut_width}")

    return L, relabeled


def cluster_coupling_matrix(
    name: str,
    coupling: float = 1.0,
    *,
    ordering: str = "rcm",
) -> np.ndarray:
    """Return the upper-triangular J_ij matrix for a supported cluster."""
    L, edges = _ordered_cluster_edges(name, ordering=ordering, verbose=True)

    out = np.zeros((L, L), dtype=np.float64)
    for i, j in edges:
        out[i, j] = coupling
    return out


def parse_unit_cell_params(raw_params) -> Dict[str, object]:
    params: Dict[str, object] = {}
    if raw_params is None:
        return params
    for raw_entry in raw_params:
        entry = str(raw_entry).strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(
                f"Invalid unit-cell parameter {raw_entry!r}; expected key=value"
            )
        key, value_text = entry.split("=", 1)
        key = key.strip()
        value_text = value_text.strip()
        if not key:
            raise ValueError(f"Invalid unit-cell parameter {raw_entry!r}; empty key")
        try:
            value = ast.literal_eval(value_text)
        except (SyntaxError, ValueError):
            value = value_text
        params[key] = value
    return params


def _load_python_geometry_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"lattice_in_line_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not import Python geometry file {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_python_geometry_payload(path: Path, params: Mapping[str, object] | None) -> object:
    module = _load_python_geometry_module(path)
    build = getattr(module, "build", None)
    if not callable(build):
        raise ValueError(
            f"Python geometry file {path} must define a callable build(**params)"
        )
    return build(**dict(params or {}))


def _unit_cell_density_params_from_inputs(
    *,
    build_params,
    aliases,
    hopping_params,
    density_inputs,
    prefix: str,
):
    if not density_inputs:
        return None
    coupling_params = {name for name in build_params if name.startswith(("J", "t"))}
    coupling_params.update(aliases)
    coupling_params.update(aliases.values())

    density_params = {
        key: value
        for key, value in hopping_params.items()
        if key not in coupling_params
    }
    for name in build_params:
        if name.startswith(("J", "t")):
            density_params[name] = 0.0

    for key, value in density_inputs.items():
        suffix = key[len(prefix):]
        if not suffix:
            raise ValueError(f"Unit-cell density coupling key {key!r} is ambiguous; use e.g. {prefix}perp")
        matched = False
        for candidate in (f"t{suffix}", f"J{suffix}"):
            canonical = aliases.get(candidate, candidate)
            if canonical in build_params:
                density_params[canonical] = value
                matched = True
                break
        if not matched:
            raise ValueError(
                f"Could not map density coupling {key!r} to a unit-cell hopping parameter. "
                f"Known aliases: {aliases}"
            )
    return density_params


def split_unit_cell_hopping_density_ph_params(path, params: Mapping[str, object] | None):
    """Split unit-cell parameters into hopping, V, and Vph inputs.

    Density parameters are named like the matching hopping parameter with a
    leading ``V`` or ``Vph`` instead of ``t``/``J``; aliases from the unit-cell
    file are followed, e.g. ``Vperp -> tperp -> Jperp`` and
    ``Vphperp -> tperp -> Jperp``.
    """
    params = dict(params or {})
    hopping_params = {}
    density_inputs = {}
    ph_density_inputs = {}
    for key, value in params.items():
        skey = str(key)
        if skey.startswith("Vph"):
            ph_density_inputs[skey] = value
        elif skey.startswith("V"):
            density_inputs[skey] = value
        else:
            hopping_params[key] = value
    if not density_inputs and not ph_density_inputs:
        return hopping_params, None, None

    unit_path = resolve_geometry_path(path)
    if unit_path.suffix.lower() != ".py":
        raise ValueError("Unit-cell V*/Vph* aliases are only supported for Python unit-cell files")
    module = _load_python_geometry_module(unit_path)
    build = getattr(module, "build", None)
    if not callable(build):
        raise ValueError(f"Python geometry file {unit_path} must define a callable build(**params)")

    aliases = getattr(module, "aliases", {})
    if aliases is None:
        aliases = {}
    aliases = {str(key): str(value) for key, value in dict(aliases).items()}
    build_params = set(inspect.signature(build).parameters)

    density_params = _unit_cell_density_params_from_inputs(
        build_params=build_params,
        aliases=aliases,
        hopping_params=hopping_params,
        density_inputs=density_inputs,
        prefix="V",
    )
    ph_density_params = _unit_cell_density_params_from_inputs(
        build_params=build_params,
        aliases=aliases,
        hopping_params=hopping_params,
        density_inputs=ph_density_inputs,
        prefix="Vph",
    )
    return hopping_params, density_params, ph_density_params


def split_unit_cell_hopping_density_params(path, params: Mapping[str, object] | None):
    """Split unit-cell parameters into hopping and density-coupling inputs.

    Density parameters are named like the matching hopping parameter with a
    leading ``V`` instead of ``t``/``J``; aliases from the unit-cell file are
    followed, e.g. ``Vperp -> tperp -> Jperp``.
    """
    hopping_params, density_params, _ = split_unit_cell_hopping_density_ph_params(path, params)
    return hopping_params, density_params


def _load_cluster_file_payload(path, *, params: Mapping[str, object] | None = None) -> object:
    cluster_path = resolve_geometry_path(path)
    text = cluster_path.read_text(encoding="utf-8")
    suffix = cluster_path.suffix.lower()
    if suffix == ".py" and (params or "def build" in text):
        return _load_python_geometry_payload(cluster_path, params)
    if suffix == ".json":
        return json.loads(text)
    if suffix == ".toml":
        if tomllib is None:
            raise ValueError("TOML cluster files require Python 3.11 or newer")
        try:
            return tomllib.loads(text)
        except tomllib.TOMLDecodeError:
            if any(token in text for token in ("[intra_cell]", "[inter_cell]", "[inter_cell_", "[couplings]")):
                return _load_repeated_key_coupling_toml(text)
            raise
    if tomllib is not None and any(token in text for token in ("[intra_cell]", "[inter_cell]", "[inter_cell_", "[couplings]")):
        try:
            return tomllib.loads(text)
        except tomllib.TOMLDecodeError:
            return _load_repeated_key_coupling_toml(text)

    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        tree = ast.parse(text, filename=str(cluster_path))
        assignments = {}
        for stmt in tree.body:
            if not isinstance(stmt, ast.Assign):
                continue
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = ast.literal_eval(stmt.value)
        for key in ("couplings", "J", "j", "j_edges", "edges_by_j"):
            if key in assignments:
                return assignments[key]
        if any(
            key == "unit_cell_sites" or key == "intra_cell" or _INTER_CELL_SECTION_RE.match(key)
            for key in assignments
        ):
            return assignments
        if len(assignments) == 1:
            return next(iter(assignments.values()))
        raise ValueError(
            f"Could not parse cluster file {cluster_path}. Expected a J-to-edge map "
            "or an assignment such as couplings = {1.0: [(0, 1)]}."
        )


def _strip_inline_comment(line: str) -> str:
    in_string = False
    escaped = False
    for idx, char in enumerate(line):
        if char == "\\" and in_string and not escaped:
            escaped = True
            continue
        if char == '"' and not escaped:
            in_string = not in_string
        if char == "#" and not in_string:
            return line[:idx]
        escaped = False
    return line


def _load_repeated_key_coupling_toml(text: str) -> dict:
    """Parse the small coupling-file TOML subset while merging repeated J keys."""
    payload = {}
    section = None
    for raw_line in text.splitlines():
        line = _strip_inline_comment(raw_line).strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            if section in ("intra_cell", "couplings") or _INTER_CELL_SECTION_RE.match(section):
                payload.setdefault(section, {})
            continue
        if "=" not in line:
            continue
        key_text, value_text = line.split("=", 1)
        key_text = key_text.strip()
        key = ast.literal_eval(key_text) if key_text.startswith('"') else key_text
        value = ast.literal_eval(value_text.strip().rstrip(","))
        if section in ("intra_cell", "couplings") or (section is not None and _INTER_CELL_SECTION_RE.match(section)):
            target = payload.setdefault(section, {})
            target.setdefault(key, []).extend(value)
        else:
            payload[key] = value
    return payload


def _extract_cluster_file_map(payload) -> Tuple[int, object]:
    if not isinstance(payload, dict):
        raise ValueError("Cluster file must contain a map from coupling J to edge lists")
    n_sites = None
    if "couplings" in payload:
        coupling_map = payload["couplings"]
        n_sites = payload.get("L", payload.get("n_sites", payload.get("sites")))
    else:
        reserved = {"L", "n_sites", "sites"}
        coupling_map = {key: val for key, val in payload.items() if key not in reserved}
        for key in reserved:
            if key in payload:
                n_sites = payload[key]
                break
    if not isinstance(coupling_map, dict) or not coupling_map:
        raise ValueError("Cluster file must contain at least one coupling-to-edges entry")
    return (-1 if n_sites is None else int(n_sites)), coupling_map


def load_cluster_file_edges(path) -> Tuple[int, WeightedEdgeList]:
    """Load a user cluster file as ``(L, [(i, j, J), ...])``.

    Accepted forms include JSON/TOML maps such as ``{"1.0": [[0, 1]]}``
    and Python-literal/assignment files such as ``couplings = {1.0: [(0, 1)]}``.
    Coupling values from the file are used directly.
    """
    n_sites, coupling_map = _extract_cluster_file_map(_load_cluster_file_payload(path))
    seen = {}
    weighted: WeightedEdgeList = []
    max_site = -1

    for raw_coupling, raw_edges in coupling_map.items():
        coupling = float(raw_coupling)
        if not math.isfinite(coupling):
            raise ValueError(f"Invalid non-finite coupling {raw_coupling!r}")
        if not isinstance(raw_edges, (list, tuple)):
            raise ValueError(f"Edges for coupling {raw_coupling!r} must be a list")
        for raw_edge in raw_edges:
            if not isinstance(raw_edge, (list, tuple)) or len(raw_edge) != 2:
                raise ValueError(f"Invalid edge {raw_edge!r} for coupling {raw_coupling!r}")
            i, j = int(raw_edge[0]), int(raw_edge[1])
            if i < 0 or j < 0:
                raise ValueError(f"Cluster edge indices must be non-negative, got {(i, j)}")
            if i == j:
                raise ValueError(f"Self-edge {(i, j)} is not allowed")
            if i > j:
                i, j = j, i
            edge = (i, j)
            if edge in seen:
                raise ValueError(
                    f"Cluster edge {edge} appears more than once "
                    f"(J={seen[edge]:g} and J={coupling:g})"
                )
            seen[edge] = coupling
            weighted.append((i, j, coupling))
            max_site = max(max_site, i, j)

    if not weighted:
        raise ValueError("Cluster file does not contain any edges")
    inferred_sites = max_site + 1
    if n_sites < 0:
        n_sites = inferred_sites
    elif n_sites < inferred_sites:
        raise ValueError(
            f"Cluster file declares L={n_sites}, but edge indices require at least L={inferred_sites}"
        )
    weighted.sort(key=lambda item: (item[0], item[1], item[2]))
    return n_sites, weighted


def cluster_file_ordering_permutation(path, *, ordering: str = "rcm") -> List[int]:
    L, weighted_edges = load_cluster_file_edges(path)
    edges = [(i, j) for i, j, _ in weighted_edges]
    perm, _ = _resolve_ordering_permutation(L, edges, ordering=ordering, key=resolve_geometry_path(path).stem)
    return perm


def _ordered_cluster_file_edges(
    path,
    *,
    ordering: str = "rcm",
    verbose: bool = False,
) -> Tuple[int, WeightedEdgeList]:
    L, weighted_edges = load_cluster_file_edges(path)
    perm, ordering_label = _resolve_ordering_permutation(
        L,
        [(i, j) for i, j, _ in weighted_edges],
        ordering=ordering,
        key=resolve_geometry_path(path).stem,
    )
    relabeled = []
    for i, j, coupling in weighted_edges:
        ii, jj = perm[i], perm[j]
        if ii > jj:
            ii, jj = jj, ii
        relabeled.append((ii, jj, coupling))
    relabeled.sort(key=lambda item: (item[0], item[1], item[2]))

    if verbose:
        cluster_path = resolve_geometry_path(path)
        perm_str = ", ".join(f"{i}\u2192{j}" for i, j in enumerate(perm))
        print(f"{cluster_path.name} ordering ({ordering_label}): {perm_str}")
        bandwidth, avg_range, cut_width = _layout_stats([(i, j) for i, j, _ in relabeled])
        counts = {}
        for _, _, coupling in weighted_edges:
            counts[coupling] = counts.get(coupling, 0) + 1
        coupling_summary = ", ".join(f"J={coupling:g}: {count}" for coupling, count in sorted(counts.items()))
        print(f"{cluster_path.name}: L={L}, edges={len(weighted_edges)}, {coupling_summary}")
        print(f"{cluster_path.name} bandwidth={bandwidth}, avg_range={avg_range:.2f}, cut_width={cut_width}")

    return L, relabeled


def cluster_file_coupling_matrix(
    path,
    *,
    ordering: str = "rcm",
) -> np.ndarray:
    """Return an upper-triangular J_ij matrix from a user-supplied cluster file."""
    L, weighted_edges = _ordered_cluster_file_edges(path, ordering=ordering, verbose=True)
    out = np.zeros((L, L), dtype=np.float64)
    for i, j, coupling in weighted_edges:
        out[i, j] = coupling
    return out


def _normalize_boundary(boundary: str) -> str:
    boundary = str(boundary).strip().lower()
    if boundary not in ("open", "periodic"):
        raise ValueError(f"Unsupported boundary {boundary!r}; expected 'open' or 'periodic'")
    return boundary


def _extract_unit_cell_file_payload(payload) -> Tuple[int, object, Dict[int, object]]:
    if not isinstance(payload, dict):
        raise ValueError("Unit-cell file must contain unit_cell_sites, intra_cell, and/or inter_cell")
    if "unit_cell_sites" not in payload:
        raise ValueError("Unit-cell file must define unit_cell_sites")
    unit_cell_sites = int(payload["unit_cell_sites"])
    if unit_cell_sites <= 0:
        raise ValueError(f"unit_cell_sites must be positive, got {unit_cell_sites}")
    intra = payload.get("intra_cell", {})
    inter_maps: Dict[int, object] = {}
    for key, value in payload.items():
        match = _INTER_CELL_SECTION_RE.match(str(key))
        if not match:
            continue
        offset = 1 if match.group(1) is None else int(match.group(1))
        if offset <= 0:
            raise ValueError(f"{key} must use a positive cell offset")
        inter_maps[offset] = value
    if not isinstance(intra, dict) or any(not isinstance(edge_map, dict) for edge_map in inter_maps.values()):
        raise ValueError("intra_cell and inter_cell[_N] must be maps from coupling J to edge lists")
    if not intra and not inter_maps:
        raise ValueError("Unit-cell file must contain at least one intra_cell or inter_cell edge")
    return unit_cell_sites, intra, inter_maps


def _add_weighted_edge(
    weighted: WeightedEdgeList,
    seen: dict,
    i: int,
    j: int,
    coupling: float,
    *,
    source: str,
) -> None:
    if i == j:
        raise ValueError(f"Self-edge {(i, j)} from {source} is not allowed")
    if i > j:
        i, j = j, i
    edge = (i, j)
    if edge in seen:
        raise ValueError(
            f"Unit-cell edge {edge} appears more than once "
            f"(J={seen[edge]:g} and J={coupling:g})"
        )
    seen[edge] = coupling
    weighted.append((i, j, coupling))


def _iter_unit_cell_edge_map(edge_map, unit_cell_sites: int, label: str):
    for raw_coupling, raw_edges in edge_map.items():
        coupling = float(raw_coupling)
        if not math.isfinite(coupling):
            raise ValueError(f"Invalid non-finite coupling {raw_coupling!r}")
        if not isinstance(raw_edges, (list, tuple)):
            raise ValueError(f"{label} edges for coupling {raw_coupling!r} must be a list")
        for raw_edge in raw_edges:
            if not isinstance(raw_edge, (list, tuple)) or len(raw_edge) != 2:
                raise ValueError(f"Invalid {label} edge {raw_edge!r} for coupling {raw_coupling!r}")
            i, j = int(raw_edge[0]), int(raw_edge[1])
            if not (0 <= i < unit_cell_sites and 0 <= j < unit_cell_sites):
                raise ValueError(
                    f"{label} edge {(i, j)} is outside the unit cell with "
                    f"{unit_cell_sites} sites"
                )
            yield coupling, i, j


def load_unit_cell_file_edges(path, L: int, *, boundary: str = "open",
                              params: Mapping[str, object] | None = None) -> Tuple[int, WeightedEdgeList]:
    """Expand a repeated unit-cell file into ``(L, [(i, j, J), ...])``."""
    L = int(L)
    if L <= 0:
        raise ValueError(f"L must be positive, got {L}")
    boundary = _normalize_boundary(boundary)
    unit_cell_sites, intra, inter_maps = _extract_unit_cell_file_payload(
        _load_cluster_file_payload(path, params=params)
    )
    remainder = L % unit_cell_sites
    if remainder and boundary == "periodic":
        raise ValueError(
            f"Periodic unit-cell repetition requires L to be a multiple of "
            f"unit_cell_sites={unit_cell_sites}, got L={L}"
        )
    if remainder:
        warnings.warn(
            f"L={L} contains {L // unit_cell_sites} complete unit cell(s) plus "
            f"the first {remainder} site(s) of a final cell with "
            f"unit_cell_sites={unit_cell_sites}; dropping edges to omitted sites",
            UserWarning,
        )
    n_cells = (L + unit_cell_sites - 1) // unit_cell_sites
    weighted: WeightedEdgeList = []
    seen = {}

    for coupling, i, j in _iter_unit_cell_edge_map(intra, unit_cell_sites, "intra_cell"):
        for cell in range(n_cells):
            base = cell * unit_cell_sites
            if base + i >= L or base + j >= L:
                continue
            _add_weighted_edge(
                weighted,
                seen,
                base + i,
                base + j,
                coupling,
                source=f"intra_cell cell={cell}",
            )

    for offset, inter in sorted(inter_maps.items()):
        inter_cells = range(n_cells if boundary == "periodic" else max(0, n_cells - offset))
        label = "inter_cell" if offset == 1 else f"inter_cell_{offset}"
        for coupling, i, j in _iter_unit_cell_edge_map(inter, unit_cell_sites, label):
            for cell in inter_cells:
                next_cell = (cell + offset) % n_cells
                src = cell * unit_cell_sites + i
                dst = next_cell * unit_cell_sites + j
                if src >= L or dst >= L:
                    continue
                _add_weighted_edge(
                    weighted,
                    seen,
                    src,
                    dst,
                    coupling,
                    source=f"{label} cell={cell}",
                )

    if not weighted:
        raise ValueError("Expanded unit-cell file does not contain any edges")
    weighted.sort(key=lambda item: (item[0], item[1], item[2]))
    return L, weighted


def unit_cell_file_ordering_permutation(path, L: int, *, boundary: str = "open",
                                        ordering: str = "rcm",
                                        params: Mapping[str, object] | None = None) -> List[int]:
    L, weighted_edges = load_unit_cell_file_edges(path, L, boundary=boundary, params=params)
    edges = [(i, j) for i, j, _ in weighted_edges]
    perm, _ = _resolve_ordering_permutation(L, edges, ordering=ordering, key=resolve_geometry_path(path).stem)
    return perm


def infer_bipartite_sublattice(L: int, edges: Sequence[Tuple[int, int]]) -> List[int]:
    """Return a +/-1 bipartition for an undirected graph or raise if impossible."""
    L = int(L)
    neighbors: List[List[int]] = [[] for _ in range(L)]
    for i, j in edges:
        i, j = int(i), int(j)
        if i == j:
            raise ValueError(f"Self edge {(i, j)} is not bipartite")
        if not (0 <= i < L and 0 <= j < L):
            raise ValueError(f"Edge {(i, j)} is outside graph with L={L}")
        neighbors[i].append(j)
        neighbors[j].append(i)

    sublattice = [0] * L
    for root in range(L):
        if sublattice[root] != 0:
            continue
        sublattice[root] = 1
        queue = deque([root])
        while queue:
            cur = queue.popleft()
            for nxt in neighbors[cur]:
                want = -sublattice[cur]
                if sublattice[nxt] == 0:
                    sublattice[nxt] = want
                    queue.append(nxt)
                elif sublattice[nxt] != want:
                    raise ValueError("Hopping graph is not bipartite")
    return sublattice


def unit_cell_file_hopping_matrix_and_sublattice(
    path,
    L: int,
    *,
    boundary: str = "open",
    ordering: str = "rcm",
    params: Mapping[str, object] | None = None,
) -> Tuple[np.ndarray, List[int]]:
    """Return ordered upper-triangular hopping matrix and ordered bipartition."""
    L, weighted_edges = load_unit_cell_file_edges(path, L, boundary=boundary, params=params)
    edges = [(i, j) for i, j, _ in weighted_edges]
    sublattice = infer_bipartite_sublattice(L, edges)
    perm, ordering_label = _resolve_ordering_permutation(
        L, edges, ordering=ordering, key=resolve_geometry_path(path).stem
    )
    ordered_sublattice = [0] * L
    for old, new in enumerate(perm):
        ordered_sublattice[new] = sublattice[old]

    relabeled = []
    for i, j, hopping in weighted_edges:
        ii, jj = perm[i], perm[j]
        if ii > jj:
            ii, jj = jj, ii
        relabeled.append((ii, jj, hopping))
    relabeled.sort(key=lambda item: (item[0], item[1], item[2]))

    out = np.zeros((L, L), dtype=np.float64)
    for i, j, hopping in relabeled:
        out[i, j] = hopping

    unit_cell_path = resolve_geometry_path(path)
    perm_str = ", ".join(f"{i}\u2192{j}" for i, j in enumerate(perm))
    print(f"{unit_cell_path.name} ordering ({ordering_label}): {perm_str}")
    bandwidth, avg_range, cut_width = _layout_stats([(i, j) for i, j, _ in relabeled])
    counts = {}
    for _, _, hopping in weighted_edges:
        counts[hopping] = counts.get(hopping, 0) + 1
    hopping_summary = ", ".join(f"t={hopping:g}: {count}" for hopping, count in sorted(counts.items()))
    pattern = "".join("A" if g > 0 else "B" for g in ordered_sublattice)
    print(
        f"{unit_cell_path.name}: L={L}, boundary={_normalize_boundary(boundary)}, "
        f"edges={len(weighted_edges)}, {hopping_summary}"
    )
    print(f"{unit_cell_path.name} bandwidth={bandwidth}, avg_range={avg_range:.2f}, cut_width={cut_width}")
    print(f"{unit_cell_path.name} sublattice={pattern}")
    return out, ordered_sublattice


def _ordered_unit_cell_file_edges(
    path,
    L: int,
    *,
    boundary: str = "open",
    ordering: str = "rcm",
    verbose: bool = False,
    params: Mapping[str, object] | None = None,
    ordering_params: Mapping[str, object] | None = None,
) -> Tuple[int, WeightedEdgeList]:
    L, weighted_edges = load_unit_cell_file_edges(path, L, boundary=boundary, params=params)
    _, ordering_edges = load_unit_cell_file_edges(
        path,
        L,
        boundary=boundary,
        params=params if ordering_params is None else ordering_params,
    )
    perm, ordering_label = _resolve_ordering_permutation(
        L,
        [(i, j) for i, j, _ in ordering_edges],
        ordering=ordering,
        key=resolve_geometry_path(path).stem,
    )
    relabeled = []
    for i, j, coupling in weighted_edges:
        ii, jj = perm[i], perm[j]
        if ii > jj:
            ii, jj = jj, ii
        relabeled.append((ii, jj, coupling))
    relabeled.sort(key=lambda item: (item[0], item[1], item[2]))

    if verbose:
        unit_cell_path = resolve_geometry_path(path)
        perm_str = ", ".join(f"{i}\u2192{j}" for i, j in enumerate(perm))
        print(f"{unit_cell_path.name} ordering ({ordering_label}): {perm_str}")
        bandwidth, avg_range, cut_width = _layout_stats([(i, j) for i, j, _ in relabeled])
        counts = {}
        for _, _, coupling in weighted_edges:
            counts[coupling] = counts.get(coupling, 0) + 1
        coupling_summary = ", ".join(f"J={coupling:g}: {count}" for coupling, count in sorted(counts.items()))
        print(
            f"{unit_cell_path.name}: L={L}, boundary={_normalize_boundary(boundary)}, "
            f"edges={len(weighted_edges)}, {coupling_summary}"
        )
        print(f"{unit_cell_path.name} bandwidth={bandwidth}, avg_range={avg_range:.2f}, cut_width={cut_width}")

    return L, relabeled


def unit_cell_file_coupling_matrix(
    path,
    L: int,
    *,
    boundary: str = "open",
    ordering: str = "rcm",
    params: Mapping[str, object] | None = None,
    ordering_params: Mapping[str, object] | None = None,
) -> np.ndarray:
    """Return an ordered upper-triangular J_ij matrix from a repeated unit-cell file."""
    L, weighted_edges = _ordered_unit_cell_file_edges(
        path,
        L,
        boundary=boundary,
        ordering=ordering,
        verbose=True,
        params=params,
        ordering_params=ordering_params,
    )
    out = np.zeros((L, L), dtype=np.float64)
    for i, j, coupling in weighted_edges:
        out[i, j] = coupling
    return out


def _distance_matrix_from_edges(L: int, edges: EdgeList) -> List[List[int]]:
    neighbors: List[List[int]] = [[] for _ in range(L)]
    for i, j in edges:
        neighbors[i].append(j)
        neighbors[j].append(i)

    distance = [[0 if i == j else -1 for j in range(L)] for i in range(L)]
    for src in range(L):
        queue = deque([src])
        while queue:
            cur = queue.popleft()
            base = distance[src][cur]
            for nxt in neighbors[cur]:
                if distance[src][nxt] != -1:
                    continue
                distance[src][nxt] = base + 1
                queue.append(nxt)
    return distance


def cluster_distance_matrix(
    name: str,
    *,
    ordering: str = "rcm",
    reordered: bool = False,
) -> List[List[int]]:
    """Return the graph distance matrix for a supported cluster."""
    if reordered:
        L, edges = _ordered_cluster_edges(name, ordering=ordering, verbose=False)
    else:
        L = cluster_size(name)
        edges = CLUSTER_EDGES[name]
    return _distance_matrix_from_edges(L, edges)


def cluster_file_distance_matrix(
    path,
    *,
    ordering: str = "rcm",
    reordered: bool = False,
) -> List[List[int]]:
    """Return the graph distance matrix for a user-supplied cluster file."""
    if reordered:
        L, weighted_edges = _ordered_cluster_file_edges(path, ordering=ordering, verbose=False)
    else:
        L, weighted_edges = load_cluster_file_edges(path)
    edges = [(i, j) for i, j, _ in weighted_edges]
    return _distance_matrix_from_edges(L, edges)


def unit_cell_file_distance_matrix(
    path,
    L: int,
    *,
    boundary: str = "open",
    ordering: str = "rcm",
    reordered: bool = False,
    params: Mapping[str, object] | None = None,
) -> List[List[int]]:
    """Return the graph distance matrix for an expanded repeated unit-cell file."""
    if reordered:
        L, weighted_edges = _ordered_unit_cell_file_edges(
            path, L, boundary=boundary, ordering=ordering, verbose=False, params=params
        )
    else:
        L, weighted_edges = load_unit_cell_file_edges(path, L, boundary=boundary, params=params)
    edges = [(i, j) for i, j, _ in weighted_edges]
    return _distance_matrix_from_edges(L, edges)


def cluster_pairs_at_distance(distance_matrix: List[List[int]], dist: int) -> List[Tuple[int, int]]:
    """Return all site pairs with graph distance ``dist``."""
    pairs = []
    L = len(distance_matrix)
    for j in range(L):
        for i in range(j):
            if distance_matrix[i][j] == dist:
                pairs.append((i, j))
    return pairs
