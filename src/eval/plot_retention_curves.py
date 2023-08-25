"""Utilities for plotting retention curves.

Script adapted from Uncertainty Baselines project,
    code due to Neil Band and Angelos Filos.
    (see https://github.com/google/uncertainty-baselines for full citation).
"""

import matplotlib.pyplot as plt
import numpy as np

from sdc.assessment import calc_uncertainty_regection_curve


BASELINE_TO_COLOR_HEX = {
    'Oracle': '#148f41',  # green
    'Model': '#7d1323',  # red
    'Random': '#36337a'  # blue
}


def get_sparsification_factor(arr_len):
    """Determine with which multiple we
    subsample the array (for easier plotting)."""
    sparsification_factor = None
    if arr_len > 100000:
        sparsification_factor = 1000
    elif arr_len > 10000:
        sparsification_factor = 100
    elif arr_len > 1000:
        sparsification_factor = 10

    return sparsification_factor


def plot_retention_curve_with_baselines(
    uncertainty: np.ndarray,
    error: np.ndarray,
    metric_name: str = 'weightedADE',
    group_by_uncertainty: bool = True,
):
    """
    Plot a retention curve with Random and Oracle baselines.

    Assumes that `uncertainty` convey uncertainty
    for a particular point.

    Args:
        uncertainty: uncertainty score.
        error: array of error values, e.g., weightedADE.
        metric_name: retention metric, displayed on y-axis.
    """
    M = len(error)

    methods = ['Random', 'Model', 'Oracle']
    ls = [":", "-", "--"]
    aucs_retention_curves = []

    # Random results
    random_indices = np.arange(M)
    np.random.shuffle(random_indices)
    retention_curve = calc_uncertainty_regection_curve(
        errors=error, uncertainty=random_indices, group_by_uncertainty=group_by_uncertainty)
    aucs_retention_curves.append((retention_curve.mean(), retention_curve))

    # Get baseline results
    retention_curve = calc_uncertainty_regection_curve(
        errors=error, uncertainty=uncertainty, group_by_uncertainty=group_by_uncertainty)
    aucs_retention_curves.append((retention_curve.mean(), retention_curve))

    # Optimal results
    retention_curve = calc_uncertainty_regection_curve(
        errors=error, uncertainty=error, group_by_uncertainty=group_by_uncertainty)
    aucs_retention_curves.append((retention_curve.mean(), retention_curve))

    plt.rcParams.update({'font.size': 12})

    fig, ax = plt.subplots()
    for b, (baseline, (auc, retention_values)) in enumerate(
            zip(methods, aucs_retention_curves)):
        color = BASELINE_TO_COLOR_HEX[baseline]

        # Subsample the retention value,
        # as there are likely many of them
        # (on any of the dataset splits)
        sparsification_factor = get_sparsification_factor(
            retention_values.shape[0])
        retention_values = retention_values[::sparsification_factor][::-1]
        retention_thresholds = np.arange(
            len(retention_values)) / len(retention_values)

        ax.plot(
            retention_thresholds,
            retention_values,
            linestyle=ls[b],
            label=f'{baseline} [AUC: {auc:.3f}]',
            color=color)

    ax.grid()
    ax.set(xlabel='Retention Fraction [-]', ylabel=metric_name + " [-]")
    ax.legend()
    fig.tight_layout()
    plt.show()
    return fig
