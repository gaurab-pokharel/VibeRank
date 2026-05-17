"""
models.py — Generative models for pairwise comparison tournaments.

Each model defines:
    - A pairwise probability matrix P where P[i,j] = Pr(i beats j)
    - A method to sample tournaments (win matrices) from P
    - The true ranking sigma_0

Reference:
    Bradley, Terry (1952). "Rank Analysis of Incomplete Block Designs."
    Negahban, Oh, Shah (2017). "Rank Centrality: Ranking from Pairwise Comparisons."
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Tournament:
    """
    A single sampled tournament with metadata.
    
    The win_matrix follows the convention used in the RC class:
        win_matrix[i, j] = 1 if item i beat item j, 0 otherwise.
    For k=1 comparisons per pair, this is a binary matrix with
    win_matrix[i,j] + win_matrix[j,i] = 1 for all i != j.
    """
    win_matrix: np.ndarray
    n: int = field(init=False)

    def __post_init__(self):
        self.n = self.win_matrix.shape[0]

    @property
    def out_degrees(self) -> np.ndarray:
        return self.win_matrix.sum(axis=1).astype(int)

    @property
    def circular_triad_count(self) -> int:
        """T_n via out-degree formula: T = C(n,3) - sum_i C(s_i, 2)."""
        s = self.out_degrees  # now int array
        total_triples = self.n * (self.n - 1) * (self.n - 2) // 6
        transitive_triples = np.sum(s * (s - 1) // 2)
        return total_triples - transitive_triples

    @property
    def t_max(self) -> int:
        n = self.n
        if n % 2 == 0:
            return n * (n * n - 4) // 24
        else:
            return n * (n * n - 1) // 24

    @property
    def normalized_cycle_rate(self) -> float:
        """zeta = T_n / T_max in [0, 1]. 0 = perfectly transitive, 1 = maximally cyclic."""
        tm = self.t_max
        return self.circular_triad_count / tm if tm > 0 else 0.0


class ComparisonModel(ABC):
    """Abstract base class for pairwise comparison generative models."""

    def __init__(self, n: int):
        self.n = n
        self._P: Optional[np.ndarray] = None

    @property
    def P(self) -> np.ndarray:
        """Pairwise probability matrix: P[i,j] = Pr(i beats j)."""
        if self._P is None:
            self._P = self._build_probability_matrix()
        return self._P

    @abstractmethod
    def _build_probability_matrix(self) -> np.ndarray:
        ...

    @property
    def true_ranking(self) -> np.ndarray:
        """
        True ranking: true_ranking[i] = rank of item i.
        Rank 1 = best. Items indexed so item 0 is best, item n-1 is worst.
        """
        return np.arange(1, self.n + 1)

    def sample_tournament(self, rng: np.random.Generator) -> Tournament:
        """
        Sample a single tournament (one comparison per unordered pair).
        Returns a Tournament whose win_matrix is compatible with RC.rank_centrality().
        """
        P = self.P
        n = self.n
        U = rng.random((n, n))
        result = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                if U[i, j] < P[i, j]:
                    result[i, j] = 1.0
                else:
                    result[j, i] = 1.0
        return Tournament(win_matrix=result)

    def expected_cycle_count(self) -> float:
        """
        Exact E[T_n] = sum_{i<j<k} c_{ijk}.
        c_{ijk} = P[i,j]*P[j,k]*P[k,i] + P[j,i]*P[k,j]*P[i,k].
        """
        P = self.P
        n = self.n
        E_T = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    E_T += (P[i, j] * P[j, k] * P[k, i] +
                            P[j, i] * P[k, j] * P[i, k])
        return E_T


class EquallySpacedBTL(ComparisonModel):
    """
    BTL model with equally-spaced log-scores.

    P(i beats j) = sigma(beta * (j - i))

    Equivalent to the geometric-score parameterization w_i = e^{(n-i)*beta}
    used in Section 3.3 of Negahban, Oh, Shah (2017), under the identification
    beta = log(b) / n.

    Parameters
    ----------
    n : int
        Number of items.
    beta : float >= 0
        Per-position discriminability. beta=0 is uniform random,
        beta->inf is deterministic.
    """

    def __init__(self, n: int, beta: float):
        super().__init__(n)
        self.beta = beta

    def _build_probability_matrix(self) -> np.ndarray:
        n = self.n
        P = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    P[i, j] = 0.5
                else:
                    gap = j - i
                    P[i, j] = 1.0 / (1.0 + np.exp(-self.beta * gap))
        return P

    @property
    def adjacent_flip_probability(self) -> float:
        """Probability that adjacent items are misordered."""
        return 1.0 / (1.0 + np.exp(self.beta))

    @property
    def dynamic_range(self) -> float:
        """b = w_max / w_min = exp(beta * (n-1))."""
        return np.exp(self.beta * (self.n - 1))

    def __repr__(self) -> str:
        return (f"EquallySpacedBTL(n={self.n}, beta={self.beta:.4f}, "
                f"adj_flip={self.adjacent_flip_probability:.3f})")