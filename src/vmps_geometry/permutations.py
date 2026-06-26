"""Default optimized cluster permutations.

The source tables are split into ``permutations_qubo`` and ``permutations_sat``.
This compatibility table preserves the active choices from the original mixed
file while keeping every source-specific alternative available explicitly.
"""

from .permutations_qubo import CUSTOM_PERMUTATIONS as _QUBO_PERMUTATIONS
from .permutations_sat import CUSTOM_PERMUTATIONS as _SAT_PERMUTATIONS


CUSTOM_PERMUTATIONS = dict(_SAT_PERMUTATIONS)
for _name in (
    "icosidodeca",
    "hyperkagome72_3x2x1",
    "hyperkagome96_2x2x2",
    "kagomeYcyl192_16x8",
):
    CUSTOM_PERMUTATIONS[_name] = _QUBO_PERMUTATIONS[_name]

__all__ = ["CUSTOM_PERMUTATIONS"]
