"""Charts comparing the two training methods."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_target_vs_achieved(results: dict, path: str) -> None:
    """Scatter of target vs simulated value; points on the diagonal are perfect designs."""
    fig, axes = plt.subplots(2, len(results), figsize=(5.5 * len(results), 9), squeeze=False)
    for col, (name, df) in enumerate(results.items()):
        for row, (target, achieved, label) in enumerate([
            ("target_gain", "achieved_gain", "Gain (Vout / Vin)"),
            ("target_rin_ohms", "achieved_rin_ohms", "Input resistance (ohms)"),
        ]):
            ax = axes[row][col]
            ax.scatter(df[target], df[achieved], s=6, alpha=0.4)
            lo = min(df[target].min(), df[achieved].min())
            hi = max(df[target].max(), df[achieved].max())
            ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(f"Target {label.lower()}")
            ax.set_ylabel(f"Simulated {label.lower()}")
            ax.set_title(f"{name}: {label}")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_error_distribution(results: dict, path: str) -> None:
    """Histogram of the worst of the two errors for each test specification."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.logspace(-3, 3, 60)
    for name, df in results.items():
        worst = np.maximum(df["gain_error_pct"], df["rin_error_pct"])
        ax.hist(np.clip(worst, bins[0], bins[-1]), bins=bins, alpha=0.6, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Largest specification error (%)")
    ax.set_ylabel("Number of test specifications")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
