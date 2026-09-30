"""Device configuration: everything the operator guidance needs to know about a device.

Operators edit this on the Device page. Values are in volts. Nothing here is required for
analysis, and no voltage scale is assumed: unset voltage settings are replaced by spacings
measured in the scans or by fractions of the current window. Setting them makes the next-scan
guidance safer and more specific.
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
    """Per-device settings. Nothing here is required, and nothing assumes a voltage scale:
    voltages vary widely between devices and technologies, so every voltage-valued setting is
    optional. Unset values are replaced by quantities measured in the scans themselves (line
    spacings) or by fractions of the current scan window."""
    name: str = "default"
    description: str = ""
    carrier: Literal["electron", "hole"] = "electron"  # electron: more plunger voltage -> more electrons
    # gate names (HRL convention: P plungers, X exchange gates, T reservoir tunnel gates, M sensor)
    plungers: list[str] = Field(default_factory=lambda: ["P1", "P2", "P3"])
    sensor_gate: str = "M1"
    # exchange (barrier) gate between two plungers, keyed "P1-P2" (order-insensitive)
    barriers: dict[str, str] = Field(default_factory=lambda: {"P1-P2": "X1", "P2-P3": "X2"})
    # reservoir tunnel gate next to each edge plunger
    tunnel_gates: dict[str, str] = Field(default_factory=lambda: {"P1": "T1", "P3": "T2"})
    # safe operating range per gate, [min, max] volts; gates without an entry are not checked
    safe_limits: dict[str, list[float]] = Field(default_factory=dict)
    # largest DC move recommended in one go (V); unset: at most one window width per move
    max_step: Optional[float] = Field(None, gt=0)
    # suggested exchange-gate change for merged dots or tie-bar coupling (V); unset: a quarter of
    # the plunger's electron spacing, or a tenth of the window if the spacing is unknown
    barrier_step: Optional[float] = Field(None, gt=0)
    # typical addition voltage per plunger (V), an optional starting guess; spacings measured on
    # this device take over as soon as scans are analysed
    addition_voltage: dict[str, float] = Field(default_factory=dict)
    points_per_addition: int = 12        # resolution target for recommended scans
    min_points: int = 48
    max_points: int = 256
    virtual_gates: Optional[VirtualGates] = None
    # optional calibrations, used only to express tie-bar results in energy units
    lever_arm: dict[str, float] = Field(default_factory=dict)   # eV per V, per plunger
    electron_temperature: Optional[float] = Field(None, gt=0)   # K
    # target band for the tie-bar coupling ratio (interdot width / tie-bar length); unset: report only
    tiebar_coupling_target: Optional[list[float]] = None
    notes: str = ""

    def barrier_between(self, g1: str, g2: str) -> Optional[str]:
        return self.barriers.get(f"{g1}-{g2}") or self.barriers.get(f"{g2}-{g1}")

    def tunnel_gate_for(self, plunger: str) -> Optional[str]:
        return self.tunnel_gates.get(plunger)

    def plunger_for_tunnel_gate(self, gate: str) -> Optional[str]:
        return next((p for p, t in self.tunnel_gates.items() if t == gate), None)

    def limits(self, gate: str) -> tuple[float, float]:
        lo, hi = self.safe_limits.get(gate, [-np.inf, np.inf])
        return float(lo), float(hi)

    def has_limits(self, gate: str) -> bool:
        return gate in self.safe_limits

    def spacing(self, gate: str) -> Optional[float]:
        """Configured typical addition voltage (V), or None if not set."""
        v = self.addition_voltage.get(gate)
        return float(v) if v else None

    def step_limit(self, window_span: float) -> float:
        """Largest move for one gate (V): the device's max_step, else one window width."""
        return float(self.max_step) if self.max_step else float(abs(window_span))

    def barrier_step_for(self, spacing_v: Optional[float], window_span: float) -> float:
        if self.barrier_step:
            return float(self.barrier_step)
        return 0.25 * spacing_v if spacing_v else 0.1 * abs(window_span)
