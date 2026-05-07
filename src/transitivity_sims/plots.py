"""
plots.py — Visualization of experiment results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List
from transitivity_sims.experiment import ExperimentResult


COLORS_N = {20: '#2176AE', 30: '#57B8FF', 50: '#B66D0D', 100: '#FBB13C'}
MARKERS_N = {20: 'o', 30: 's', 50: '^', 100: 'D'}


def _style(n):
    return COLORS_N.get(n, '#333333'), MARKERS_N.get(n, 'v')

def _by_n(results, n):
    return sorted([r for r in results if r.n == n], key=lambda r: r.beta)

def _n_vals(results):
    return sorted(set(r.n for r in results))


def plot_beta_vs_accuracy(results: List[ExperimentResult], output_path: str):
    """beta on x-axis, Kendall tau / adjacent acc / pairwise acc on y-axis."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for n_val in _n_vals(results):
        subset = _by_n(results, n_val)
        color, marker = _style(n_val)

        betas = [r.beta for r in subset]

        axes[0].errorbar(betas, [np.mean(r.kendall_taus) for r in subset],
                         yerr=[np.std(r.kendall_taus) for r in subset],
                         marker=marker, color=color, label=f'n={n_val}',
                         capsize=3, linewidth=1.5, markersize=5)
        axes[1].errorbar(betas, [np.mean(r.adjacent_accuracies) for r in subset],
                         yerr=[np.std(r.adjacent_accuracies) for r in subset],
                         marker=marker, color=color, label=f'n={n_val}',
                         capsize=3, linewidth=1.5, markersize=5)
        axes[2].errorbar(betas, [np.mean(r.pairwise_accuracies) for r in subset],
                         yerr=[np.std(r.pairwise_accuracies) for r in subset],
                         marker=marker, color=color, label=f'n={n_val}',
                         capsize=3, linewidth=1.5, markersize=5)

    for ax in axes:
        ax.set_xlabel(r'$\beta$', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(r'Kendall $\tau$', fontsize=11)
    axes[0].set_title('Overall ranking accuracy', fontsize=12)
    axes[1].set_ylabel('Adjacent pair accuracy', fontsize=11)
    axes[1].set_title('Hardest pairs: adjacent items', fontsize=12)
    axes[2].set_ylabel('Pairwise accuracy', fontsize=11)
    axes[2].set_title('All-pairs accuracy', fontsize=12)
    axes[0].axhline(y=0, color='gray', linestyle=':', alpha=0.5)

    fig.suptitle(r'RC accuracy vs discriminability $\beta$', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_cycle_rate_vs_accuracy(results: List[ExperimentResult], output_path: str):
    """Normalized cycle rate on x-axis, accuracy + consistency on y-axis."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for n_val in _n_vals(results):
        subset = _by_n(results, n_val)
        color, marker = _style(n_val)

        zetas = [np.mean(r.cycle_rates) for r in subset]
        taus = [np.mean(r.kendall_taus) for r in subset]
        tau_stds = [np.std(r.kendall_taus) for r in subset]
        cons = [np.mean(r.consistency_taus) for r in subset]
        cons_stds = [np.std(r.consistency_taus) for r in subset]

        axes[0].errorbar(zetas, taus, yerr=tau_stds, marker=marker,
                         color=color, label=f'n={n_val}', capsize=3,
                         linewidth=1.5, markersize=5)
        axes[1].errorbar(zetas, cons, yerr=cons_stds, marker=marker,
                         color=color, label=f'n={n_val}', capsize=3,
                         linewidth=1.5, markersize=5)

        for i, r in enumerate(subset):
            lbl = f'{r.beta:.2f}' if r.beta < 1 else f'{r.beta:.1f}'
            axes[0].annotate(rf'$\beta$={lbl}', (zetas[i], taus[i]),
                             textcoords="offset points", xytext=(5, 5),
                             fontsize=6, alpha=0.7, color=color)

    axes[0].set_xlabel(r'Normalized cycle rate $\zeta$', fontsize=11)
    axes[0].set_ylabel(r'Kendall $\tau$ (RC vs true)', fontsize=11)
    axes[0].set_title('Accuracy vs observed cyclicity', fontsize=12)
    axes[0].axhline(y=0, color='gray', linestyle=':', alpha=0.5)

    axes[1].set_xlabel(r'Normalized cycle rate $\zeta$', fontsize=11)
    axes[1].set_ylabel(r'Kendall $\tau$ (RC$_1$ vs RC$_2$)', fontsize=11)
    axes[1].set_title('Self-consistency vs observed cyclicity', fontsize=12)

    for ax in axes:
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.02, 1.02)

    fig.suptitle(r'Observed cycle rate $\zeta$ as diagnostic for RC reliability',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_scatter_trials(results: List[ExperimentResult], output_path: str):
    """Trial-level scatter: each dot = one tournament."""
    n_values = _n_vals(results)
    fig, axes = plt.subplots(1, len(n_values), figsize=(7 * len(n_values), 5.5))
    if len(n_values) == 1:
        axes = [axes]

    for ax, n_val in zip(axes, n_values):
        for r in _by_n(results, n_val):
            lbl = rf'$\beta$={r.beta:.2f}' if r.beta < 1 else rf'$\beta$={r.beta:.1f}'
            ax.scatter(r.cycle_rates, r.kendall_taus, alpha=0.1, s=6, label=lbl)
        ax.set_xlabel(r'$\zeta$', fontsize=11)
        ax.set_ylabel(r'Kendall $\tau$', fontsize=11)
        ax.set_title(f'n = {n_val}', fontsize=12)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.15, 1.05)
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize=7, ncol=2,
                  markerscale=3, loc='upper right', framealpha=0.8)

    fig.suptitle('Trial-level: each dot = one sampled tournament', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_self_consistency(results: List[ExperimentResult], output_path: str):
    """Accuracy vs self-consistency scatter."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    for n_val in _n_vals(results):
        subset = _by_n(results, n_val)
        color, marker = _style(n_val)

        acc = [np.mean(r.kendall_taus) for r in subset]
        cons = [np.mean(r.consistency_taus) for r in subset]

        ax.scatter(acc, cons, marker=marker, color=color,
                   s=60, label=f'n={n_val}', zorder=5)

        for i, r in enumerate(subset):
            lbl = f'{r.beta:.2f}' if r.beta < 1 else f'{r.beta:.1f}'
            ax.annotate(rf'$\beta$={lbl}', (acc[i], cons[i]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7, alpha=0.7, color=color)

    x = np.linspace(0, 1, 100)
    ax.plot(x, x**2, 'k--', alpha=0.3, label=r'$\tau_{cons} = \tau_{acc}^2$')
    ax.plot(x, x, 'k:', alpha=0.2, label=r'$y = x$')

    ax.set_xlabel(r'Accuracy: Kendall $\tau$ (RC vs true)', fontsize=11)
    ax.set_ylabel(r'Self-consistency: Kendall $\tau$ (RC$_1$ vs RC$_2$)', fontsize=11)
    ax.set_title('Accuracy predicts self-consistency', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_top_k_recovery(results: List[ExperimentResult], output_path: str):
    """beta vs top-k accuracy."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))

    for n_val in _n_vals(results):
        subset = _by_n(results, n_val)
        color, marker = _style(n_val)
        k = subset[0].config.top_k

        ax.errorbar([r.beta for r in subset],
                    [np.mean(r.top_k_accuracies) for r in subset],
                    yerr=[np.std(r.top_k_accuracies) for r in subset],
                    marker=marker, color=color, label=f'n={n_val} (k={k})',
                    capsize=3, linewidth=1.5, markersize=5)

    ax.set_xlabel(r'$\beta$', fontsize=11)
    ax.set_ylabel('Top-k recovery accuracy', fontsize=11)
    ax.set_title('Can RC identify the best items?', fontsize=12)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_all_plots(results: List[ExperimentResult], output_dir: str = './figures'):
    """Generate all diagnostic plots."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("\nGenerating plots...")
    plot_beta_vs_accuracy(results, f'{output_dir}/beta_vs_accuracy.png')
    plot_cycle_rate_vs_accuracy(results, f'{output_dir}/cycle_rate_vs_accuracy.png')
    plot_scatter_trials(results, f'{output_dir}/scatter_trials.png')
    plot_self_consistency(results, f'{output_dir}/self_consistency.png')
    plot_top_k_recovery(results, f'{output_dir}/top_k_recovery.png')
    print("  All plots generated.")