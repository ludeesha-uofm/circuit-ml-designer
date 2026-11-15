import numpy as np
import pytest

from circuit_ml.ladder import analytic_ladder, measure_ladder
from circuit_ml.simulator import Circuit, CircuitError


def test_voltage_divider():
    c = Circuit()
    c.add_voltage_source("V", 1, 0, 10.0)
    c.add_resistor("R1", 1, 2, 1000.0)
    c.add_resistor("R2", 2, 0, 3000.0)
    s = c.solve()
    assert s.voltage(2) == pytest.approx(7.5)
    assert s.source_currents["V"] == pytest.approx(10.0 / 4000.0)


def test_parallel_resistors_current():
    c = Circuit()
    c.add_voltage_source("V", 1, 0, 12.0)
    c.add_resistor("R1", 1, 0, 100.0)
    c.add_resistor("R2", 1, 0, 300.0)
    s = c.solve()
    assert s.source_currents["V"] == pytest.approx(12 / 100 + 12 / 300)


def test_kirchhoff_current_law_at_internal_node():
    c = Circuit()
    c.add_voltage_source("V", 1, 0, 5.0)
    c.add_resistor("R1", 1, 2, 220.0)
    c.add_resistor("R2", 2, 0, 470.0)
    c.add_resistor("R3", 2, 0, 1000.0)
    s = c.solve()
    v2 = s.voltage(2)
    into_node = (5.0 - v2) / 220.0
    out_of_node = v2 / 470.0 + v2 / 1000.0
    assert into_node == pytest.approx(out_of_node)


def test_floating_node_is_rejected():
    c = Circuit()
    c.add_voltage_source("V", 1, 0, 5.0)
    c.add_resistor("R1", 1, 0, 100.0)
    c.add_resistor("R2", 2, 3, 100.0)  # not connected to the rest
    with pytest.raises(CircuitError):
        c.solve()


def test_invalid_resistance_is_rejected():
    c = Circuit()
    with pytest.raises(CircuitError):
        c.add_resistor("R", 1, 0, -5.0)
    with pytest.raises(CircuitError):
        c.add_resistor("R", 1, 0, 0.0)


def test_loop_of_voltage_sources_is_rejected():
    c = Circuit()
    c.add_voltage_source("V1", 1, 0, 5.0)
    c.add_voltage_source("V2", 1, 0, 3.0)
    with pytest.raises(CircuitError):
        c.solve()


def test_simulator_matches_closed_form_ladder():
    rng = np.random.default_rng(1)
    for _ in range(50):
        r = 10 ** rng.uniform(1, 5, size=6)
        series, shunt = r[:3], r[3:]
        gain_sim, rin_sim = measure_ladder(series, shunt)
        gain_exact, rin_exact = analytic_ladder(series, shunt)
        assert gain_sim == pytest.approx(float(gain_exact), rel=1e-9)
        assert rin_sim == pytest.approx(float(rin_exact), rel=1e-9)
