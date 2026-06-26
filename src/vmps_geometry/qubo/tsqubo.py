"""Dense NumPy tabu search for QUBO, with optional fixed-cardinality search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatVector = NDArray[np.float64]
IntVector = NDArray[np.int8]


@dataclass
class TSQUBOSolution:
    """Holds a QUBO solution, its value, and flip reevaluation vector."""

    fx: float = 0.0
    x: IntVector = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    dx: FloatVector = field(default_factory=lambda: np.zeros(0, dtype=np.float64))

    def copy(self) -> "TSQUBOSolution":
        return TSQUBOSolution(float(self.fx), self.x.copy(), self.dx.copy())


class TSQUBOInstance:
    """Dense QUBO instance backed by NumPy arrays.

    Users can create an instance incrementally with ``add_component()`` or load
    a full objective matrix with ``from_matrix()`` / ``set_matrix()``.

    The matrix API follows standard QUBO semantics:

        f(x) = x.T @ Q @ x
    """

    def __init__(self, n: int = 0, matrix: Optional[ArrayLike] = None) -> None:
        if matrix is not None and n:
            raise ValueError("pass either n or matrix, not both")
        if matrix is not None:
            self.set_matrix(matrix)
            return
        if n < 0:
            raise ValueError("n must be non-negative")
        self.n = n
        self.qubo_matrix = np.zeros((n, n), dtype=np.float64)
        self.interaction_matrix = np.zeros((n, n), dtype=np.float64)
        self.diagonal = np.zeros(n, dtype=np.float64)

    @classmethod
    def from_matrix(cls, matrix: ArrayLike) -> "TSQUBOInstance":
        return cls(matrix=matrix)

    def set_matrix(self, matrix: ArrayLike) -> None:
        qubo = np.asarray(matrix, dtype=np.float64)
        if qubo.ndim != 2 or qubo.shape[0] != qubo.shape[1]:
            raise ValueError("matrix must be square")
        self.n = int(qubo.shape[0])
        self.qubo_matrix = np.array(qubo, dtype=np.float64, copy=True)
        self.interaction_matrix = self.qubo_matrix + self.qubo_matrix.T
        self.diagonal = np.diag(self.qubo_matrix).copy()
        np.fill_diagonal(self.interaction_matrix, self.diagonal)

    def _ensure_size(self, index: int) -> None:
        if index < 0:
            raise ValueError("indices must be non-negative")
        if index < self.n:
            return

        new_n = index + 1
        qubo = np.zeros((new_n, new_n), dtype=np.float64)
        interaction = np.zeros((new_n, new_n), dtype=np.float64)
        diagonal = np.zeros(new_n, dtype=np.float64)

        if self.n:
            qubo[: self.n, : self.n] = self.qubo_matrix
            interaction[: self.n, : self.n] = self.interaction_matrix
            diagonal[: self.n] = self.diagonal

        self.n = new_n
        self.qubo_matrix = qubo
        self.interaction_matrix = interaction
        self.diagonal = diagonal

    def add_component(self, i: int, j: int, q: float) -> None:
        """Adds a component using the original symmetric component API."""

        self._ensure_size(max(i, j))
        value = float(q)
        if i == j:
            self.qubo_matrix[i, i] += value
            self.interaction_matrix[i, i] += value
            self.diagonal[i] += value
            return

        self.qubo_matrix[i, j] += value
        self.qubo_matrix[j, i] += value
        interaction = 2.0 * value
        self.interaction_matrix[i, j] += interaction
        self.interaction_matrix[j, i] += interaction

    def coefficient(self, i: int, j: int) -> float:
        if not (0 <= i < self.n and 0 <= j < self.n):
            raise IndexError("coefficient index out of range")
        return float(self.interaction_matrix[i, j])

    def evaluate(self, x: Sequence[int]) -> float:
        bits = self._normalize_solution(x).astype(np.float64, copy=False)
        return float(bits @ self.qubo_matrix @ bits)

    def _normalize_solution(self, x: Sequence[int]) -> IntVector:
        bits = np.asarray(x)
        size = int(bits.shape[0]) if bits.ndim >= 1 else 0
        if bits.ndim != 1 or size != self.n:
            raise ValueError(f"expected solution of length {self.n}, got {size}")
        if not np.all((bits == 0) | (bits == 1)):
            raise ValueError("solutions must be binary")
        return bits.astype(np.int8, copy=True)


class TSQUBO:
    """Dense tabu search for QUBO instances using NumPy.

    If ``fixed_weight`` is provided, search is restricted to solutions whose
    binary variables sum to that exact value. Feasible moves then become pair
    swaps between one active and one inactive variable.
    """

    def __init__(
        self,
        inst_or_matrix: Union[TSQUBOInstance, ArrayLike],
        fixed_weight: Optional[int] = None,
    ) -> None:
        self.inst = (
            inst_or_matrix
            if isinstance(inst_or_matrix, TSQUBOInstance)
            else TSQUBOInstance.from_matrix(inst_or_matrix)
        )
        if fixed_weight is not None and not (0 <= fixed_weight <= self.inst.n):
            raise ValueError("fixed_weight must be between 0 and the number of variables")
        self.fixed_weight = fixed_weight
        self.inc = TSQUBOSolution(
            0.0,
            np.zeros(self.inst.n, dtype=np.int8),
            np.zeros(self.inst.n, dtype=np.float64),
        )
        self.cur = TSQUBOSolution(
            0.0,
            np.zeros(self.inst.n, dtype=np.int8),
            np.zeros(self.inst.n, dtype=np.float64),
        )
        self.tabulist = np.zeros(self.inst.n, dtype=np.int64)
        self.iteration = 0

    @classmethod
    def from_matrix(
        cls, matrix: ArrayLike, fixed_weight: Optional[int] = None
    ) -> "TSQUBO":
        return cls(matrix, fixed_weight=fixed_weight)

    def commit_incumbent(self) -> None:
        self.inc = self.cur.copy()

    def reset_solutions(self, initial_x: Optional[Sequence[int]] = None) -> None:
        if initial_x is None:
            x = np.zeros(self.inst.n, dtype=np.int8)
            dx = self.inst.diagonal.copy()
            self.cur = TSQUBOSolution(0.0, x, dx)
            if self.fixed_weight is not None:
                self._greedy_fill_to_weight(self.fixed_weight)
        else:
            x = self.inst._normalize_solution(initial_x)
            if self.fixed_weight is not None and int(x.sum()) != self.fixed_weight:
                raise ValueError(
                    f"expected exactly {self.fixed_weight} active variables, got {int(x.sum())}"
                )
            self.cur = TSQUBOSolution(
                self.inst.evaluate(x),
                x,
                self._compute_flip_deltas(x),
            )
        self.commit_incumbent()

    def reset_tabu(self) -> None:
        self.tabulist = np.zeros(self.inst.n, dtype=np.int64)
        self.iteration = 0

    def flip_current(self, i: int) -> None:
        if self.fixed_weight is not None:
            raise RuntimeError("flip_current() would violate the fixed-cardinality constraint")
        self._flip_current(i)

    def swap_current(self, out_index: int, in_index: int) -> None:
        if self.fixed_weight is None:
            raise RuntimeError("swap_current() is only available in fixed-cardinality mode")
        if out_index == in_index:
            raise ValueError("swap indices must be different")
        if self.cur.x[out_index] != 1 or self.cur.x[in_index] != 0:
            raise ValueError("swap_current() expects a 1->0 move and a 0->1 move")
        self._flip_current(out_index)
        self._flip_current(in_index)

    def local_search(self) -> None:
        while True:
            if self.fixed_weight is None:
                move = self._best_single_flip(ignore_tabu=True)
                if move is None or move[1] >= 0.0:
                    return
                self._flip_current(move[0])
            else:
                move = self._best_swap(ignore_tabu=True)
                if move is None or move[2] >= 0.0:
                    return
                self.swap_current(move[0], move[1])

    def iterate(self, tabu_tenure: int) -> bool:
        if tabu_tenure < 0:
            raise ValueError("tabu_tenure must be non-negative")

        if self.fixed_weight is None:
            move = self._best_single_flip(ignore_tabu=False)
            if move is None:
                return False
            index = move[0]
            self.iteration += 1
            self.tabulist[index] = self.iteration + tabu_tenure
            self._flip_current(index)
        else:
            move = self._best_swap(ignore_tabu=False)
            if move is None:
                return False
            out_index, in_index = move[0], move[1]
            self.iteration += 1
            self.tabulist[out_index] = self.iteration + tabu_tenure
            self.tabulist[in_index] = self.iteration + tabu_tenure
            self.swap_current(out_index, in_index)

        return self.cur.fx < self.inc.fx

    def iterate_cutoff(self, tabu_tenure: int, cutoff: int) -> None:
        if cutoff < 0:
            raise ValueError("cutoff must be non-negative")
        while True:
            improved = False
            for _ in range(cutoff):
                if self.iterate(tabu_tenure):
                    self.local_search()
                    self.commit_incumbent()
                    improved = True
                    break
            if not improved:
                return

    def solve(
        self,
        tabu_tenure: int,
        cutoff: int,
        initial_x: Optional[Sequence[int]] = None,
    ) -> TSQUBOSolution:
        self.reset_solutions(initial_x=initial_x)
        self.reset_tabu()
        self.local_search()
        self.commit_incumbent()
        self.iterate_cutoff(tabu_tenure, cutoff)
        return self.inc.copy()

    def _compute_flip_deltas(self, x: IntVector) -> FloatVector:
        x_float = x.astype(np.float64, copy=False)
        field = self.inst.interaction_matrix @ x_float + self.inst.diagonal * (1.0 - x_float)
        return (1.0 - 2.0 * x_float) * field

    def _greedy_fill_to_weight(self, weight: int) -> None:
        for _ in range(weight):
            candidate_dx = np.where(self.cur.x == 0, self.cur.dx, np.inf)
            best = int(np.argmin(candidate_dx))
            self._flip_current(best)

    def _flip_current(self, i: int) -> None:
        delta = float(self.cur.dx[i])
        self.cur.fx += delta
        self.cur.x[i] = 1 - self.cur.x[i]
        self.cur.dx[i] = -delta

        spins = 1.0 - 2.0 * self.cur.x.astype(np.float64, copy=False)
        update = (1.0 - 2.0 * float(self.cur.x[i])) * spins * self.inst.interaction_matrix[i]
        update[i] = 0.0
        self.cur.dx -= update

    def _best_single_flip(self, ignore_tabu: bool) -> Optional[Tuple[int, float]]:
        candidate_dx = self.cur.dx.copy()
        if not ignore_tabu:
            blocked = (self.iteration < self.tabulist) & (
                self.cur.fx + candidate_dx >= self.inc.fx
            )
            candidate_dx[blocked] = np.inf

        index = int(np.argmin(candidate_dx))
        if not np.isfinite(candidate_dx[index]):
            return None
        return index, float(candidate_dx[index])

    def _best_swap(self, ignore_tabu: bool) -> Optional[Tuple[int, int, float]]:
        ones = np.flatnonzero(self.cur.x == 1)
        zeros = np.flatnonzero(self.cur.x == 0)
        if ones.size == 0 or zeros.size == 0:
            return None

        delta_matrix = (
            self.cur.dx[ones][:, None]
            + self.cur.dx[zeros][None, :]
            - self.inst.interaction_matrix[np.ix_(ones, zeros)]
        )

        if not ignore_tabu:
            blocked = (self.iteration < self.tabulist[ones])[:, None] | (
                self.iteration < self.tabulist[zeros]
            )[None, :]
            improving = self.cur.fx + delta_matrix < self.inc.fx
            delta_matrix = np.where(blocked & ~improving, np.inf, delta_matrix)

        flat_index = int(np.argmin(delta_matrix))
        best_delta = float(delta_matrix.flat[flat_index])
        if not np.isfinite(best_delta):
            return None
        row, col = np.unravel_index(flat_index, delta_matrix.shape)
        return int(ones[row]), int(zeros[col]), best_delta

    def _swap_delta(self, out_index: int, in_index: int) -> float:
        return float(
            self.cur.dx[out_index]
            + self.cur.dx[in_index]
            - self.inst.interaction_matrix[out_index, in_index]
        )


__all__ = ["TSQUBO", "TSQUBOInstance", "TSQUBOSolution"]
