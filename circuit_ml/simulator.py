"""A small DC circuit simulator for linear resistor networks.

It uses modified nodal analysis (MNA), the same method used by SPICE-style
simulators. Node 0 is ground. Every other node is numbered 1, 2, 3, ...

The simulator is the "judge" in this project: whatever circuit the machine
learning model proposes is evaluated here, not by the model's own maths.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GROUND = 0


class CircuitError(ValueError):
    """Raised when a circuit is invalid or cannot be solved."""


@dataclass(frozen=True)
class Resistor:
    name: str
    node_a: int
    node_b: int
    ohms: float


@dataclass(frozen=True)
class VoltageSource:
    name: str
    plus: int
    minus: int
    volts: float


@dataclass
class Solution:
    """Result of a DC solve."""

    node_voltages: np.ndarray  # index = node number; node 0 (ground) is 0 V
    source_currents: dict[str, float] = field(default_factory=dict)  # amps delivered

    def voltage(self, node: int) -> float:
        return float(self.node_voltages[node])


class Circuit:
    """A collection of resistors and ideal DC voltage sources."""

    def __init__(self) -> None:
        self.resistors: list[Resistor] = []
        self.sources: list[VoltageSource] = []

    def add_resistor(self, name: str, node_a: int, node_b: int, ohms: float) -> None:
        if not np.isfinite(ohms) or ohms <= 0:
            raise CircuitError(f"Resistor {name} must have a positive resistance, got {ohms}")
        if node_a == node_b:
            raise CircuitError(f"Resistor {name} connects node {node_a} to itself")
        self._check_nodes(name, node_a, node_b)
        self.resistors.append(Resistor(name, node_a, node_b, float(ohms)))

    def add_voltage_source(self, name: str, plus: int, minus: int, volts: float) -> None:
        if plus == minus:
            raise CircuitError(f"Source {name} connects node {plus} to itself")
        self._check_nodes(name, plus, minus)
        self.sources.append(VoltageSource(name, plus, minus, float(volts)))

    @staticmethod
    def _check_nodes(name: str, *nodes: int) -> None:
        for node in nodes:
            if not isinstance(node, (int, np.integer)) or node < 0:
                raise CircuitError(f"{name}: node numbers must be integers >= 0, got {node}")

    # ------------------------------------------------------------------ solving
    def _node_count(self) -> int:
        nodes = [0]
        for r in self.resistors:
            nodes += [r.node_a, r.node_b]
        for s in self.sources:
            nodes += [s.plus, s.minus]
        return max(nodes)

    def _check_connected(self, n: int) -> None:
        """Every node must be linked to ground, otherwise the maths has no unique answer."""
        neighbours: dict[int, set[int]] = {i: set() for i in range(n + 1)}
        for r in self.resistors:
            neighbours[r.node_a].add(r.node_b)
            neighbours[r.node_b].add(r.node_a)
        for s in self.sources:
            neighbours[s.plus].add(s.minus)
            neighbours[s.minus].add(s.plus)
        seen, stack = {GROUND}, [GROUND]
        while stack:
            for nxt in neighbours[stack.pop()]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        used = {GROUND}
        for r in self.resistors:
            used |= {r.node_a, r.node_b}
        for s in self.sources:
            used |= {s.plus, s.minus}
        floating = sorted(used - seen)
        if floating:
            raise CircuitError(f"Nodes {floating} are not connected to ground")

    def solve(self) -> Solution:
        """Solve for all node voltages and the current delivered by each source."""
        if not self.resistors and not self.sources:
            raise CircuitError("The circuit is empty")
        n = self._node_count()
        m = len(self.sources)
        self._check_connected(n)

        a = np.zeros((n + m, n + m))
        z = np.zeros(n + m)

        for r in self.resistors:
            g = 1.0 / r.ohms
            i, j = r.node_a - 1, r.node_b - 1  # row/column index; -1 means ground
            if i >= 0:
                a[i, i] += g
            if j >= 0:
                a[j, j] += g
            if i >= 0 and j >= 0:
                a[i, j] -= g
                a[j, i] -= g

        for k, s in enumerate(self.sources):
            row = n + k
            if s.plus > 0:
                a[s.plus - 1, row] += 1.0
                a[row, s.plus - 1] += 1.0
            if s.minus > 0:
                a[s.minus - 1, row] -= 1.0
                a[row, s.minus - 1] -= 1.0
            z[row] = s.volts

        try:
            x = np.linalg.solve(a, z)
        except np.linalg.LinAlgError as exc:
            raise CircuitError("The circuit equations are singular (for example, a loop of sources)") from exc

        voltages = np.concatenate(([0.0], x[:n]))
        # x[n + k] is the current flowing into the + terminal, so the source delivers its negative.
        currents = {s.name: float(-x[n + k]) for k, s in enumerate(self.sources)}
        return Solution(node_voltages=voltages, source_currents=currents)
