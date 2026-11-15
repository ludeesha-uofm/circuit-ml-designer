"""Generate data, train both designers, and compare them using the simulator.

Example:
    python train.py --samples 20000 --epochs 200 --out results
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from circuit_ml.config import DEFAULT_STAGES, LOAD_OHMS
from circuit_ml.data_generator import generate_dataset, split_dataset
from circuit_ml.evaluate import evaluate_design, summarize
from circuit_ml.model import build_design_model, spec_features, train_physics, train_supervised
from circuit_ml.plots import plot_error_distribution, plot_target_vs_achieved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--samples", type=int, default=20000, help="number of synthetic circuits")
    parser.add_argument("--stages", type=int, default=DEFAULT_STAGES, help="ladder stages")
    parser.add_argument("--epochs", type=int, default=200, help="maximum training epochs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results", help="output folder")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(args.seed)

    print(f"Generating {args.samples} random {args.stages}-stage circuits with the simulator...")
    data = generate_dataset(args.samples, args.stages, LOAD_OHMS, seed=args.seed)
    data.to_csv(os.path.join(args.out, "circuit_data.csv"), index=False)
    train_df, val_df, test_df = split_dataset(data, seed=args.seed)
    print(f"Train {len(train_df)} | validation {len(val_df)} | test {len(test_df)}")

    models, results, summaries = {}, {}, {}
    for name, trainer in [("supervised", train_supervised), ("physics-informed", train_physics)]:
        print(f"\nTraining the {name} model...")
        model = build_design_model(args.stages, spec_features(train_df))
        kwargs = {"load": LOAD_OHMS} if name == "physics-informed" else {}
        history = trainer(model, train_df, val_df, args.stages, epochs=args.epochs, **kwargs)
        print(f"  stopped after {len(history.history['loss'])} epochs, "
              f"best validation loss {min(history.history['val_loss']):.5f}")
        model.save(os.path.join(args.out, f"design_{name.replace('-', '_')}.keras"))
        results[name] = evaluate_design(model, test_df, args.stages, LOAD_OHMS)
        results[name].to_csv(os.path.join(args.out, f"test_results_{name.replace('-', '_')}.csv"), index=False)
        summaries[name] = summarize(results[name], test_df.reset_index(drop=True), args.stages)
        models[name] = model

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(summaries, f, indent=2)
    plot_target_vs_achieved(results, os.path.join(args.out, "target_vs_achieved.png"))
    plot_error_distribution(results, os.path.join(args.out, "error_distribution.png"))

    print("\nResults on unseen test specifications (each proposed circuit simulated):")
    for name, s in summaries.items():
        print(f"  {name:17s} median gain error {s['median_gain_error_pct']:7.2f}% | "
              f"median input-resistance error {s['median_rin_error_pct']:7.2f}% | "
              f"both within 5%: {100 * s['share_within_5pct']:5.1f}% | "
              f"mean resistor gap {s['mean_resistor_gap_decades']:.2f} decades")
    print(f"\nSaved models, tables and charts to '{args.out}/'.")


if __name__ == "__main__":
    main()
