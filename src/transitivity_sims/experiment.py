"""
experiment.py — Experiment orchestration.

Uses the existing RC implementation from src.viberank.rankers.rc
to run Rank Centrality on synthetic BTL tournaments.
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any

from transitivity_sims.models import ComparisonModel, EquallySpacedBTL, Tournament
from transitivity_sims.metrics import (
    TrialMetrics, ConsistencyMetrics,
    kendall_tau_correlation, spearman_correlation,
    pairwise_accuracy, adjacent_pair_accuracy, top_k_accuracy
)

# Import the existing RC implementation
from viberank.rankers.rank_centrality import RC



# Try to import the existing RC implementation.
# If it fails (e.g., base class init crashes with comparator=None),
# fall back to a standalone implementation of the same algorithm.
_USE_VIBERANK_RC = False
# try:
from viberank.rankers.rank_centrality import RC as _RC
#     # Test that we can instantiate with comparator=None
#     _test = _RC(items=[0, 1], comparator=None, folder_name=None)
#     _test.rank_centrality(np.array([[0, 1], [0, 0]], dtype=float))
#     _USE_VIBERANK_RC = True
#     del _test
# except Exception:
#     pass
 
 
def _standalone_rank_centrality(win_matrix, tol=1e-8, num_iterations=1000):
    """
    Standalone RC implementation matching the logic in src.viberank.rankers.rc.
    Used as fallback if the RC class can't be instantiated with comparator=None.
    """
    N = win_matrix.shape[0]
    A = np.zeros((N, N))
 
    for i in range(N):
        for j in range(N):
            if i != j:
                total = win_matrix[i, j] + win_matrix[j, i]
                if total > 0:
                    A[i, j] = win_matrix[j, i] / total
 
    d = np.sum(A, axis=1)
    d_max = np.max(d)
    if d_max == 0:
        return np.ones(N) / N
 
    P = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                P[i, j] = A[i, j] / d_max
        P[i, i] = 1 - np.sum(A[i, :]) / d_max
 
    p = np.ones(N) / N
    for _ in range(num_iterations):
        new_p = p.dot(P)
        if np.linalg.norm(new_p - p, 1) < tol:
            p = new_p
            break
        p = new_p
    p = p / np.sum(p)
    return p
 
 
def rc_scores_from_tournament(tournament: Tournament) -> np.ndarray:
    """
    Run Rank Centrality on a tournament.
 
    Tries the existing RC class from src.viberank first; if that fails
    (e.g., Ranker base class doesn't accept comparator=None), falls back
    to a standalone implementation of the same algorithm.
    """
    if _USE_VIBERANK_RC:
        rc = _RC(
            items=list(range(tournament.n)),
            comparator=None,
            compare_probability=1.0,
            folder_name=None
        )
        scores, _ = rc.rank_centrality(tournament.win_matrix)
        return scores
    else:
        return _standalone_rank_centrality(tournament.win_matrix)
 
 
def scores_to_ranking(scores: np.ndarray) -> np.ndarray:
    """Convert score vector to ranking (1 = best, ties broken arbitrarily)."""
    order = np.argsort(-scores)
    rank = np.empty_like(order)
    rank[order] = np.arange(1, len(scores) + 1)
    return rank
 
 
# ──────────────────────────────────────────────
# Config and result containers
# ──────────────────────────────────────────────
 
@dataclass
class ExperimentConfig:
    """Configuration for a single experiment (one model parameterization)."""
    n: int
    beta: float
    n_trials: int = 500
    n_consistency_trials: int = 200
    top_k: int = 5
    seed: int = 42
 
 
@dataclass
class ExperimentResult:
    """Results from running all trials for one configuration."""
    config: ExperimentConfig
    model_info: Dict[str, Any]
    trial_metrics: List[TrialMetrics]
    consistency_metrics: List[ConsistencyMetrics]
    expected_cycles: float
    elapsed_seconds: float
 
    @property
    def n(self) -> int:
        return self.config.n
 
    @property
    def beta(self) -> float:
        return self.config.beta
 
    def _gather(self, attr: str) -> np.ndarray:
        return np.array([getattr(m, attr) for m in self.trial_metrics])
 
    @property
    def cycle_rates(self) -> np.ndarray:
        return self._gather('cycle_rate')
 
    @property
    def kendall_taus(self) -> np.ndarray:
        return self._gather('kendall_tau')
 
    @property
    def spearman_rhos(self) -> np.ndarray:
        return self._gather('spearman_rho')
 
    @property
    def pairwise_accuracies(self) -> np.ndarray:
        return self._gather('pairwise_accuracy')
 
    @property
    def adjacent_accuracies(self) -> np.ndarray:
        return self._gather('adjacent_accuracy')
 
    @property
    def top_k_accuracies(self) -> np.ndarray:
        return self._gather('top_k_accuracy')
 
    @property
    def consistency_taus(self) -> np.ndarray:
        return np.array([m.kendall_tau for m in self.consistency_metrics])
 
    @property
    def consistency_rhos(self) -> np.ndarray:
        return np.array([m.spearman_rho for m in self.consistency_metrics])
 
    @property
    def consistency_cycle_rates(self) -> np.ndarray:
        return np.array([m.cycle_rate_avg for m in self.consistency_metrics])
 
    def summary(self) -> Dict[str, float]:
        return {
            'n': self.n,
            'beta': self.beta,
            'adj_flip_prob': self.model_info['adj_flip_prob'],
            'E[T_n]': self.expected_cycles,
            'mean_cycle_rate': float(np.mean(self.cycle_rates)),
            'mean_kendall_tau': float(np.mean(self.kendall_taus)),
            'std_kendall_tau': float(np.std(self.kendall_taus)),
            'mean_spearman_rho': float(np.mean(self.spearman_rhos)),
            'mean_pairwise_acc': float(np.mean(self.pairwise_accuracies)),
            'mean_adjacent_acc': float(np.mean(self.adjacent_accuracies)),
            'mean_top_k_acc': float(np.mean(self.top_k_accuracies)),
            'mean_consistency_tau': float(np.mean(self.consistency_taus)),
            'mean_consistency_rho': float(np.mean(self.consistency_rhos)),
        }
 
 
# ──────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────
 
class ExperimentRunner:
    """
    Runs accuracy and self-consistency experiments for the
    equally-spaced BTL model using the existing RC implementation.
    """
 
    def run(self, config: ExperimentConfig, verbose: bool = True) -> ExperimentResult:
        t0 = time.time()
 
        model = EquallySpacedBTL(n=config.n, beta=config.beta)
        true_ranking = model.true_ranking
        rng = np.random.default_rng(config.seed)
 
        if verbose:
            print(f"  {model} ... ", end='', flush=True)
 
        # Precompute expected cycle count
        E_T = model.expected_cycle_count() if config.n <= 60 else np.nan
 
        # ── Accuracy trials ──
        trial_metrics = []
        for _ in range(config.n_trials):
            tourn = model.sample_tournament(rng)
            scores = rc_scores_from_tournament(tourn)
            est_ranking = scores_to_ranking(scores)
 
            trial_metrics.append(TrialMetrics(
                cycle_count=tourn.circular_triad_count,
                cycle_rate=tourn.normalized_cycle_rate,
                kendall_tau=kendall_tau_correlation(est_ranking, true_ranking),
                spearman_rho=spearman_correlation(est_ranking, true_ranking),
                top_k_accuracy=top_k_accuracy(est_ranking, true_ranking, config.top_k),
                pairwise_accuracy=pairwise_accuracy(est_ranking, true_ranking),
                adjacent_accuracy=adjacent_pair_accuracy(est_ranking, true_ranking),
            ))
 
        # ── Self-consistency trials ──
        consistency_metrics = []
        for _ in range(config.n_consistency_trials):
            tourn1 = model.sample_tournament(rng)
            tourn2 = model.sample_tournament(rng)
 
            scores1 = rc_scores_from_tournament(tourn1)
            scores2 = rc_scores_from_tournament(tourn2)
            rank1 = scores_to_ranking(scores1)
            rank2 = scores_to_ranking(scores2)
 
            cycle_rate_avg = (tourn1.normalized_cycle_rate +
                              tourn2.normalized_cycle_rate) / 2
 
            consistency_metrics.append(ConsistencyMetrics(
                cycle_rate_avg=cycle_rate_avg,
                kendall_tau=kendall_tau_correlation(rank1, rank2),
                spearman_rho=spearman_correlation(rank1, rank2),
            ))
 
        elapsed = time.time() - t0
 
        result = ExperimentResult(
            config=config,
            model_info={
                'adj_flip_prob': model.adjacent_flip_probability,
                'dynamic_range': model.dynamic_range,
            },
            trial_metrics=trial_metrics,
            consistency_metrics=consistency_metrics,
            expected_cycles=E_T,
            elapsed_seconds=elapsed,
        )
 
        if verbose:
            s = result.summary()
            print(f"done ({elapsed:.1f}s) | "
                  f"zeta={s['mean_cycle_rate']:.3f}, "
                  f"tau={s['mean_kendall_tau']:.3f}, "
                  f"rho_s={s['mean_spearman_rho']:.3f}, "
                  f"adj_acc={s['mean_adjacent_acc']:.3f}")
 
        return result
 
 
def run_experiment_grid(
    n_values: List[int],
    beta_values: List[float],
    n_trials: int = 500,
    n_consistency_trials: int = 200,
    seed: int = 42,
    verbose: bool = True,
) -> List[ExperimentResult]:
    """Run experiments across a grid of (n, beta) values."""
 
    runner = ExperimentRunner()
    results = []
 
    for n in n_values:
        if verbose:
            print(f"\n{'='*60}")
            print(f"  n = {n}")
            print(f"{'='*60}")
 
        for beta in beta_values:
            config = ExperimentConfig(
                n=n,
                beta=beta,
                n_trials=n_trials,
                n_consistency_trials=n_consistency_trials,
                top_k=min(5, n // 4),
                seed=seed,
            )
            result = runner.run(config, verbose=verbose)
            results.append(result)
 
    return results
 
