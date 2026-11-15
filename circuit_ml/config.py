"""Shared settings for the circuit design prototype."""

# Resistor values the model is allowed to choose, in ohms.
R_MIN = 10.0
R_MAX = 100_000.0

# Fixed load resistor at the output of the ladder network, in ohms.
LOAD_OHMS = 1_000.0

# Input voltage applied to the circuit, in volts.
VIN = 1.0

# Number of stages in the ladder network (each stage = 1 series + 1 shunt resistor).
DEFAULT_STAGES = 3
