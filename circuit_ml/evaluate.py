"""Judge a trained designer by simulating every circuit it proposes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LOAD_OHMS, VIN
from .data_generator import column_names
from .ladder import measure_ladder
from .model import spec_features


def evaluate_design(design_model, df: pd.DataFrame, stages: int, load: float = LOAD_OHMS) -> pd.DataFrame:
    """For each specification in `df`, ask the model for a circuit and simulate it."""
    log_r = design_model.predict(spec_features(df), verbose=0)
    resistors = 10.0 ** log_r
    series_cols, shunt_cols = column_names(stages)

    achieved_gain = np.empty(len(df))
    achieved_rin = np.empty(len(df))
    for i in range(len(df)):
        achieved_gain[i], achieved_rin[i] = measure_ladder(
            resistors[i, :stages], resistors[i, stages:], load, VIN)

    out = pd.DataFrame(resistors, columns=series_cols + shunt_cols)
    out["target_gain"] = df["gain"].to_numpy()
    out["target_rin_ohms"] = df["rin_ohms"].to_numpy()
    out["achieved_gain"] = achieved_gain
    out["achieved_rin_ohms"] = achieved_rin
    out["gain_error_pct"] = 100 * np.abs(achieved_gain - out["target_gain"]) / out["target_gain"]
    out["rin_error_pct"] = 100 * np.abs(achieved_rin - out["target_rin_ohms"]) / out["target_rin_ohms"]
    return out


def summarize(results: pd.DataFrame, true_df: pd.DataFrame | None = None, stages: int | None = None) -> dict:
    """Headline numbers for one model."""
    both = np.maximum(results["gain_error_pct"], results["rin_error_pct"])
    summary = {
        "median_gain_error_pct": float(results["gain_error_pct"].median()),
        "median_rin_error_pct": float(results["rin_error_pct"].median()),
        "mean_gain_error_pct": float(results["gain_error_pct"].mean()),
        "mean_rin_error_pct": float(results["rin_error_pct"].mean()),
        "share_within_5pct": float((both <= 5).mean()),
        "share_within_10pct": float((both <= 10).mean()),
    }
    if true_df is not None and stages is not None:
        series_cols, shunt_cols = column_names(stages)
        cols = series_cols + shunt_cols
        log_gap = np.abs(np.log10(results[cols].to_numpy()) - np.log10(true_df[cols].to_numpy()))
        summary["mean_resistor_gap_decades"] = float(log_gap.mean())
    return summary
