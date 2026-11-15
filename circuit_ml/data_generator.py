"""Synthetic tabular data: random ladder circuits and how they behave.

Each row is one random circuit (its resistor values) together with the gain and
input resistance measured by the simulator. Because every target comes from a
real circuit, every specification in the dataset is achievable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT_STAGES, LOAD_OHMS, R_MAX, R_MIN, VIN
from .ladder import measure_ladder


def column_names(stages: int) -> tuple[list[str], list[str]]:
    return [f"Rs{k + 1}" for k in range(stages)], [f"Rp{k + 1}" for k in range(stages)]


def generate_dataset(n_samples: int, stages: int = DEFAULT_STAGES, load: float = LOAD_OHMS,
                     seed: int = 0) -> pd.DataFrame:
    """Create `n_samples` random circuits. Resistances are log-uniform between R_MIN and R_MAX."""
    rng = np.random.default_rng(seed)
    log_r = rng.uniform(np.log10(R_MIN), np.log10(R_MAX), size=(n_samples, 2 * stages))
    resistors = 10.0 ** log_r

    gains = np.empty(n_samples)
    rins = np.empty(n_samples)
    for i in range(n_samples):
        gains[i], rins[i] = measure_ladder(resistors[i, :stages], resistors[i, stages:], load, VIN)

    series_cols, shunt_cols = column_names(stages)
    df = pd.DataFrame(resistors, columns=series_cols + shunt_cols)
    df["gain"] = gains
    df["rin_ohms"] = rins
    return df


def split_dataset(df: pd.DataFrame, seed: int = 0, train: float = 0.7, val: float = 0.15):
    """Shuffle and split into train / validation / test frames."""
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_train = int(len(shuffled) * train)
    n_val = int(len(shuffled) * val)
    return (shuffled.iloc[:n_train], shuffled.iloc[n_train:n_train + n_val],
            shuffled.iloc[n_train + n_val:])
