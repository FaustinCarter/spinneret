"""Device configuration: everything the operator guidance needs to know about a device.

Operators edit this on the Device page. Values are in volts. Nothing here is required for
analysis; it makes the next-scan guidance safer and more specific.
"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, Field


class VirtualGates(BaseModel):
    """Mapping from virtual plunger voltages to physical gate voltages: physical = M @ virtual.

    If the operator scans in virtual coordinates, the recommendation is computed in those
    coordinates and translated to physical gate voltages with this matrix.
    """
    virtual: list[str] = Field(default_factory=list)      # e.g. ["vP1","vP2","vP3"]
    physical: list[str] = Field(default_factory=list)     # e.g. ["P1","P2","P3","M1"]
    matrix: list[list[float]] = Field(default_factory=list)  # len(physical) x len(virtual)

    def to_physical(self, virtual_deltas: dict[str, float]) -> dict[str, float]:
        if not self.matrix:
            return {}
        m = np.asarray(self.matrix, float)
        v = np.array([virtual_deltas.get(g, 0.0) for g in self.virtual])
        out = m @ v
        return {g: float(d) for g, d in zip(self.physical, out)}


class DeviceConfig(BaseModel):
    name: str = "default"
    description: str = ""
    carrier: Literal["electron", "hole"] = "electron"  # electron: more plunger voltage -> more electrons
    plungers: list[str] = Field(default_factory=lambda: ["P1", "P2", "P3"])
    sensor_gate: str = "M1"
    # barrier gate between two plungers, keyed "P1-P2" (order-insensitive)
    barriers: dict[str, str] = Field(default_factory=lambda: {"P1-P2": "X1", "P2-P3": "X2"})
    # safe operating range per gate, [min, max] volts
    safe_limits: dict[str, list[float]] = Field(default_factory=lambda: {
        "P1": [0.0, 1.2], "P2": [0.0, 1.2], "P3": [0.0, 1.2]})
    max_step: float = 0.050              # largest DC move recommended in one go (V)
    barrier_step: float = 0.010          # suggested barrier change for merged dots (V)
    # typical addition voltage per plunger (V); refined automatically from analysed scans
    addition_voltage: dict[str, float] = Field(default_factory=lambda: {
        "P1": 0.030, "P2": 0.030, "P3": 0.030})
    points_per_addition: int = 12        # resolution target for recommended scans
    min_points: int = 48
    max_points: int = 256
    virtual_gates: Optional[VirtualGates] = None
    notes: str = ""

    def barrier_between(self, g1: str, g2: str) -> Optional[str]:
        return self.barriers.get(f"{g1}-{g2}") or self.barriers.get(f"{g2}-{g1}")

    def limits(self, gate: str) -> tuple[float, float]:
        lo, hi = self.safe_limits.get(gate, [-np.inf, np.inf])
        return float(lo), float(hi)

    def spacing(self, gate: str) -> float:
        return float(self.addition_voltage.get(gate, 0.030))
