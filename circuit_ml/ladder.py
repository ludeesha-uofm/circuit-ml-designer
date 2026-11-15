"""The circuit family the model designs: an R-ladder network.

    Vin --Rs1--+--Rs2--+--Rs3--+
               |       |       |
              Rp1     Rp2     Rp3 (all to ground), plus the fixed load across the last node

Each stage has one series resistor (Rs) and one shunt resistor (Rp).
The two numbers that describe how the circuit behaves are:
  * gain             = Vout / Vin
  * input resistance = Vin / I_in
"""
from __future__ import annotations

import numpy as np

from .config import LOAD_OHMS, VIN
from .simulator import Circuit


def build_ladder(series, shunt, load: float = LOAD_OHMS, vin: float = VIN) -> Circuit:
    """Build a ladder circuit object from lists of series and shunt resistances (ohms)."""
    series, shunt = list(series), list(shunt)
    if len(series) != len(shunt) or not series:
        raise ValueError("series and shunt must be non-empty lists of the same length")
    stages = len(series)
    circuit = Circuit()
    circuit.add_voltage_source("Vin", 1, 0, vin)
    for k in range(stages):
        circuit.add_resistor(f"Rs{k + 1}", k + 1, k + 2, series[k])
        circuit.add_resistor(f"Rp{k + 1}", k + 2, 0, shunt[k])
    circuit.add_resistor("Rload", stages + 1, 0, load)
    return circuit


def measure_ladder(series, shunt, load: float = LOAD_OHMS, vin: float = VIN) -> tuple[float, float]:
    """Simulate a ladder with the MNA simulator. Returns (gain, input_resistance_ohms)."""
    stages = len(series)
    solution = build_ladder(series, shunt, load, vin).solve()
    gain = solution.voltage(stages + 1) / vin
    rin = vin / solution.source_currents["Vin"]
    return gain, rin


def analytic_ladder(series, shunt, load: float = LOAD_OHMS) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form ladder response, vectorised over a batch.

    Used to cross-check the simulator and as the basis of the differentiable
    version in model.py. Arrays have shape (..., stages).
    """
    series, shunt = np.asarray(series, dtype=float), np.asarray(shunt, dtype=float)
    stages = series.shape[-1]

    def parallel(x, y):
        return x * y / (x + y)

    z = parallel(shunt[..., -1], load)  # resistance seen at the last node
    zs = [z]
    for k in range(stages - 2, -1, -1):
        z = parallel(shunt[..., k], series[..., k + 1] + z)
        zs.append(z)
    zs.reverse()  # zs[k] = resistance seen at node k+1 looking towards ground

    gain = np.ones_like(zs[0])
    for k in range(stages):
        gain = gain * zs[k] / (series[..., k] + zs[k])
    rin = series[..., 0] + zs[0]
    return gain, rin
