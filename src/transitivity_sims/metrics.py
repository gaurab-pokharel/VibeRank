"""
metrics.py — Evaluation metrics for ranking quality.

Measures accuracy (vs ground truth) and self-consistency (across samples).
"""

import numpy as np
from scipy.stats import kendalltau, spearmanr
from dataclasses import dataclass


@dataclass
class TrialMetrics:
    """Metrics from a single trial."""
    cycle_count: int
    cycle_rate: float              # zeta = T_n / T_max
    kendall_tau: float             # RC ranking vs true ranking
    spearman_rho: float            # RC ranking vs true ranking
    top_k_accuracy: float          # fraction of true top-k recovered
    pairwise_accuracy: float       # fraction of pairs correctly ordered
    adjacent_accuracy: float       # fraction of adjacent pairs correct


@dataclass
class ConsistencyMetrics:
    """Metrics from a self-consistency trial (two independent samples)."""
    cycle_rate_avg: float
    kendall_tau: float             # RC(sample1) vs RC(sample2)
    spearman_rho: float            # RC(sample1) vs RC(sample2)


def kendall_tau_correlation(ranking_a: np.ndarray, ranking_b: np.ndarray) -> float:
    tau, _ = kendalltau(ranking_a, ranking_b)
    return tau


def spearman_correlation(ranking_a: np.ndarray, ranking_b: np.ndarray) -> float:
    rho, _ = spearmanr(ranking_a, ranking_b)
    return rho


def pairwise_accuracy(estimated_ranking: np.ndarray,
                      true_ranking: np.ndarray) -> float:
    """
    Fraction of pairs (i,j) where the estimated ranking agrees
    with the true ranking on which is better.
    """
    n = len(true_ranking)
    correct = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            true_order = true_ranking[i] < true_ranking[j]
            est_order = estimated_ranking[i] < estimated_ranking[j]
            if true_order == est_order:
                correct += 1
            total += 1
    return correct / total


def adjacent_pair_accuracy(estimated_ranking: np.ndarray,
                           true_ranking: np.ndarray) -> float:
    """
    Fraction of truly-adjacent pairs that are correctly ordered.
    These are the hardest pairs under rank-gap-dependent noise.
    """
    n = len(true_ranking)
    true_order = np.argsort(true_ranking)
    correct = 0
    total = n - 1
    for k in range(n - 1):
        item_a = true_order[k]
        item_b = true_order[k + 1]
        if estimated_ranking[item_a] < estimated_ranking[item_b]:
            correct += 1
    return correct / total


def top_k_accuracy(estimated_ranking: np.ndarray,
                   true_ranking: np.ndarray,
                   k: int = 5) -> float:
    """Fraction of the true top-k that appear in the estimated top-k."""
    k = min(k, len(true_ranking))
    true_top_k = set(np.where(true_ranking <= k)[0])
    est_top_k = set(np.where(estimated_ranking <= k)[0])
    return len(true_top_k & est_top_k) / k