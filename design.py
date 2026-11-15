"""Ask the trained model to design a circuit, then verify it with the simulator.

Example:
    python design.py --gain 0.25 --rin 3000
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from circuit_ml.config import LOAD_OHMS
from circuit_ml.ladder import measure_ladder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--gain", type=float, required=True, help="wanted Vout/Vin, between 0 and 1")
    parser.add_argument("--rin", type=float, required=True, help="wanted input resistance in ohms")
    parser.add_argument("--model", default="results/design_physics_informed.keras")
    args = parser.parse_args()

    if not 0 < args.gain < 1:
        parser.error("--gain must be between 0 and 1 for a passive resistor ladder")
    if args.rin <= 0:
        parser.error("--rin must be positive")

    from tensorflow import keras
    model = keras.models.load_model(args.model, compile=False)
    stages = model.output_shape[-1] // 2

    spec = np.array([[np.log10(args.gain), np.log10(args.rin)]], dtype="float32")
    resistors = 10.0 ** model.predict(spec, verbose=0)[0]
    series, shunt = resistors[:stages], resistors[stages:]
    gain, rin = measure_ladder(series, shunt, LOAD_OHMS)

    print(f"Proposed {stages}-stage ladder (load = {LOAD_OHMS:.0f} ohms):")
    for k in range(stages):
        print(f"  Stage {k + 1}: series {series[k]:10.1f} ohms | shunt {shunt[k]:10.1f} ohms")
    print("\nChecked with the circuit simulator:")
    print(f"  gain  wanted {args.gain:.4f}  got {gain:.4f}  ({100 * abs(gain - args.gain) / args.gain:.2f}% off)")
    print(f"  Rin   wanted {args.rin:.1f}  got {rin:.1f} ohms  ({100 * abs(rin - args.rin) / args.rin:.2f}% off)")
    print("\nNote: some specifications cannot be built from a passive ladder with this load.")


if __name__ == "__main__":
    main()
