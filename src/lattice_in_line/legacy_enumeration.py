"""Helpers for converting legacy cluster vertex enumerations."""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple

Edge = Tuple[int, int]

# Maps old CLUSTER_EDGES labels to the current cluster_generator.py labels.
PYROCHLORE_LEGACY_TO_GENERATOR: Dict[str, Dict[int, int]] = {
    "pyrochlore48a": {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
        8: 40, 9: 41, 10: 42, 11: 43, 12: 44, 13: 45, 14: 46, 15: 47,
        16: 16, 17: 17, 18: 18, 19: 19, 20: 32, 21: 33, 22: 34, 23: 35,
        24: 8, 25: 9, 26: 10, 27: 11, 28: 24, 29: 25, 30: 26, 31: 27,
        32: 28, 33: 29, 34: 30, 35: 31, 36: 12, 37: 13, 38: 14, 39: 15,
        40: 36, 41: 37, 42: 38, 43: 39, 44: 20, 45: 21, 46: 22, 47: 23,
    },
    "pyrochlore48b": {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
        8: 32, 9: 33, 10: 34, 11: 35, 12: 28, 13: 29, 14: 30, 15: 31,
        16: 8, 17: 9, 18: 10, 19: 11, 20: 20, 21: 21, 22: 22, 23: 23,
        24: 40, 25: 41, 26: 42, 27: 43, 28: 44, 29: 45, 30: 46, 31: 47,
        32: 16, 33: 17, 34: 18, 35: 19, 36: 12, 37: 13, 38: 14, 39: 15,
        40: 36, 41: 37, 42: 38, 43: 39, 44: 24, 45: 25, 46: 26, 47: 27,
    },
    "pyrochlore48c": {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 28, 5: 29, 6: 30, 7: 31,
        8: 44, 9: 45, 10: 46, 11: 47, 12: 36, 13: 37, 14: 38, 15: 39,
        16: 20, 17: 21, 18: 22, 19: 23, 20: 40, 21: 41, 22: 42, 23: 43,
        24: 12, 25: 13, 26: 14, 27: 15, 28: 4, 29: 5, 30: 6, 31: 7,
        32: 16, 33: 17, 34: 18, 35: 19, 36: 8, 37: 9, 38: 10, 39: 11,
        40: 24, 41: 25, 42: 26, 43: 27, 44: 32, 45: 33, 46: 34, 47: 35,
    },
    "pyrochlore48d": {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 32, 5: 33, 6: 34, 7: 35,
        8: 36, 9: 37, 10: 38, 11: 39, 12: 44, 13: 45, 14: 46, 15: 47,
        16: 40, 17: 41, 18: 42, 19: 43, 20: 8, 21: 9, 22: 10, 23: 11,
        24: 20, 25: 21, 26: 22, 27: 23, 28: 16, 29: 17, 30: 18, 31: 19,
        32: 12, 33: 13, 34: 14, 35: 15, 36: 24, 37: 25, 38: 26, 39: 27,
        40: 4, 41: 5, 42: 6, 43: 7, 44: 28, 45: 29, 46: 30, 47: 31,
    },
    "pyrochlore64": {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 5, 5: 4, 6: 7, 7: 6,
        8: 15, 9: 14, 10: 13, 11: 12, 12: 53, 13: 52, 14: 55, 15: 54,
        16: 19, 17: 18, 18: 17, 19: 16, 20: 57, 21: 56, 22: 59, 23: 58,
        24: 47, 25: 46, 26: 45, 27: 44, 28: 34, 29: 35, 30: 32, 31: 33,
        32: 48, 33: 49, 34: 50, 35: 51, 36: 26, 37: 27, 38: 24, 39: 25,
        40: 60, 41: 61, 42: 62, 43: 63, 44: 42, 45: 43, 46: 40, 47: 41,
        48: 28, 49: 29, 50: 30, 51: 31, 52: 9, 53: 8, 54: 11, 55: 10,
        56: 39, 57: 38, 58: 37, 59: 36, 60: 22, 61: 23, 62: 20, 63: 21,
    },
    "pyrochlore128": {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
        8: 12, 9: 13, 10: 14, 11: 15, 12: 112, 13: 113, 14: 114, 15: 115,
        16: 76, 17: 77, 18: 78, 19: 79, 20: 116, 21: 117, 22: 118, 23: 119,
        24: 72, 25: 73, 26: 74, 27: 75, 28: 56, 29: 57, 30: 58, 31: 59,
        32: 20, 33: 21, 34: 22, 35: 23, 36: 40, 37: 41, 38: 42, 39: 43,
        40: 120, 41: 121, 42: 122, 43: 123, 44: 68, 45: 69, 46: 70, 47: 71,
        48: 44, 49: 45, 50: 46, 51: 47, 52: 28, 53: 29, 54: 30, 55: 31,
        56: 8, 57: 9, 58: 10, 59: 11, 60: 32, 61: 33, 62: 34, 63: 35,
        64: 88, 65: 89, 66: 90, 67: 91, 68: 48, 69: 49, 70: 50, 71: 51,
        72: 64, 73: 65, 74: 66, 75: 67, 76: 104, 77: 105, 78: 106, 79: 107,
        80: 84, 81: 85, 82: 86, 83: 87, 84: 124, 85: 125, 86: 126, 87: 127,
        88: 16, 89: 17, 90: 18, 91: 19, 92: 36, 93: 37, 94: 38, 95: 39,
        96: 92, 97: 93, 98: 94, 99: 95, 100: 96, 101: 97, 102: 98, 103: 99,
        104: 108, 105: 109, 106: 110, 107: 111, 108: 80, 109: 81, 110: 82, 111: 83,
        112: 100, 113: 101, 114: 102, 115: 103, 116: 60, 117: 61, 118: 62, 119: 63,
        120: 24, 121: 25, 122: 26, 123: 27, 124: 52, 125: 53, 126: 54, 127: 55,
    },
}

PYROCHLORE_GENERATOR_TO_LEGACY: Dict[str, Dict[int, int]] = {
    name: {new: old for old, new in mapping.items()}
    for name, mapping in PYROCHLORE_LEGACY_TO_GENERATOR.items()
}


def legacy_to_generator_permutation(cluster: str) -> Dict[int, int]:
    """Return a copy of the old-label -> generator-label map for ``cluster``."""
    return dict(PYROCHLORE_LEGACY_TO_GENERATOR[cluster])


def generator_to_legacy_permutation(cluster: str) -> Dict[int, int]:
    """Return a copy of the generator-label -> old-label map for ``cluster``."""
    return dict(PYROCHLORE_GENERATOR_TO_LEGACY[cluster])


def convert_legacy_vertices(cluster: str, vertices: Iterable[int]) -> Tuple[int, ...]:
    """Convert a sequence of legacy vertex labels to generator labels."""
    mapping = PYROCHLORE_LEGACY_TO_GENERATOR[cluster]
    return tuple(mapping[int(vertex)] for vertex in vertices)


def convert_legacy_edges(cluster: str, edges: Iterable[Edge]) -> Tuple[Edge, ...]:
    """Convert legacy-labeled undirected edges to generator labels."""
    mapping = PYROCHLORE_LEGACY_TO_GENERATOR[cluster]
    return tuple(sorted(tuple(sorted((mapping[int(i)], mapping[int(j)]))) for i, j in edges))


def convert_legacy_site_permutation(cluster: str, permutation: Mapping[int, int]) -> Dict[int, int]:
    """Convert a site->position permutation from legacy labels to generator labels."""
    mapping = PYROCHLORE_LEGACY_TO_GENERATOR[cluster]
    return {mapping[int(site)]: int(position) for site, position in permutation.items()}


def convert_legacy_site_permutations(permutations: Mapping[str, Mapping[int, int]]) -> Dict[str, Dict[int, int]]:
    """Convert every known legacy pyrochlore site permutation in a table."""
    converted: Dict[str, Dict[int, int]] = {
        str(name): {int(site): int(position) for site, position in permutation.items()}
        for name, permutation in permutations.items()
    }
    for name in PYROCHLORE_LEGACY_TO_GENERATOR:
        if name in converted:
            converted[name] = convert_legacy_site_permutation(name, converted[name])
    return converted


__all__ = [
    "PYROCHLORE_GENERATOR_TO_LEGACY",
    "PYROCHLORE_LEGACY_TO_GENERATOR",
    "convert_legacy_edges",
    "convert_legacy_site_permutation",
    "convert_legacy_site_permutations",
    "convert_legacy_vertices",
    "generator_to_legacy_permutation",
    "legacy_to_generator_permutation",
]
