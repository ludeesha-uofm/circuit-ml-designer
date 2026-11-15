# Circuit Design Machine Learning Prototype

A Python prototype that teaches a neural network to **design linear electrical circuits**, and a circuit simulator that **checks every circuit the model proposes**.

You give it a target behaviour (how much the output voltage should be, and how much current the circuit should draw). The model proposes resistor values for a 3-stage resistor ladder network that should behave that way. The simulator then solves the real circuit equations to see how well the proposal works.

```
Vin --Rs1--+--Rs2--+--Rs3--+
           |       |       |
          Rp1     Rp2     Rp3      (all to ground), plus a fixed 1 kOhm load on the last node
```

## What is in the project

| File | What it does |
| --- | --- |
| `circuit_ml/simulator.py` | DC circuit simulator for resistor networks using modified nodal analysis (the method behind SPICE-style tools) |
| `circuit_ml/ladder.py` | Builds the ladder circuit and measures its gain and input resistance, plus a closed-form version used for cross-checking |
| `circuit_ml/data_generator.py` | Creates the synthetic tabular dataset: random circuits and their simulated behaviour |
| `circuit_ml/model.py` | The neural network (TensorFlow/Keras) and the two training methods |
| `circuit_ml/evaluate.py` | Simulates every circuit the model proposes and measures the error |
| `train.py` | Runs the whole experiment and saves models, tables and charts |
| `design.py` | Ask the trained model for a circuit with a chosen gain and input resistance |
| `tests/` | Unit tests for the simulator and ladder maths |

## Run it

```bash
pip install -r requirements.txt
python -m pytest -q tests        # check the simulator
python train.py                  # about 2 minutes on a laptop CPU
python design.py --gain 0.05 --rin 2000
```

`train.py` writes everything to `results/`: the dataset, trained models, per-circuit results, `metrics.json`, and two charts.

## The experiment: two ways to train the same network

The network takes `log10(gain)` and `log10(input resistance)` and outputs the log10 of the six resistor values.

1. **Supervised.** Copy the resistor values stored in the dataset. This is ambiguous: many different circuits give the same gain and input resistance, so the network learns an average that is often not a working circuit.
2. **Physics-informed.** Use no resistor labels. The loss measures the gain and input resistance of the *proposed* circuit, computed with a differentiable version of the ladder equations, so any valid design is rewarded.

### Results (20,000 synthetic circuits, 3,000 unseen test specifications)

Every proposed circuit was simulated with the MNA simulator.

| Model | Median gain error | Median input-resistance error | Both within 5% |
| --- | --- | --- | --- |
| Supervised | 1066% | 33.3% | 0.0% |
| Physics-informed | 1.26% | 2.36% | 84.1% |

Your numbers may differ slightly between runs and machines.

The supervised model's resistor values are not far off in log terms (about 0.78 decades on average), but small mistakes in a resistor ladder multiply through the stages, so the simulated gain is far from the target. The physics-informed model's resistor values differ from the dataset's by a similar amount (0.87 decades), yet its circuits work, because many different circuits meet the same specification.

## Limitations

- Only DC resistor circuits are supported (no capacitors, inductors or transistors).
- The topology is fixed (a ladder); the model chooses the resistor values, not the layout.
- Specifications outside what a passive ladder with this load can achieve will not be met.
- The differentiable ladder equations and the simulator are two independent implementations of the same circuit. A test checks that they agree.

## Ideas for next steps

- Let the model choose between several circuit layouts.
- Add AC analysis with capacitors and inductors.
- Search for designs with several valid answers and pick the cheapest.
