"""Core data model shared by every part of ChargeCell.

Conventions (read these once, they matter everywhere):

* A scan is a 2D charge-stability diagram: ``signal[iy, ix]`` measured while sweeping
  ``x_gate`` over ``x`` and ``y_gate`` over ``y`` (both in volts, ascending order after import).
* "Canonical orientation" means increasing voltage on either axis adds electrons to the dot
  under that plunger. Importers flip axes (per device config) so that this always holds.
* Dot "a" is the dot under ``x_gate``; dot "b" is the dot under ``y_gate``. Every other dot in
  the device is a "spectator" for that scan.
"""
from __future__ import annotations

import datetime as _dt
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------------------------
# Outcome vocabulary
# ---------------------------------------------------------------------------------------------
FOUND = "FOUND"
NOT_IN_WINDOW = "NOT_IN_WINDOW"
UNINTERPRETABLE = "UNINTERPRETABLE"
STATUSES = [FOUND, NOT_IN_WINDOW, UNINTERPRETABLE]

# Reason codes. "none" is used with FOUND. The order is the model's class order: never reorder,
# only append (trained models depend on the indices).
REASONS = [
    "none",
    # NOT_IN_WINDOW reasons
    "no_transitions",        # no charge transitions visible at all
    "occupancy_too_low",     # empty region visible, but window stops before (1,1)
    "no_reference",          # transitions visible but no empty (0) region -> can't count electrons
    "partially_visible",     # (1,1) cut off by the window edge
    # UNINTERPRETABLE reasons
    "low_snr",
    "sensor_insensitive",    # charge sensor off its flank / lost contrast over much of the scan
    "dots_merged",           # single-dot-like pattern (interdot coupling too strong)
    "charge_instability",    # repeated charge jumps / switching between sweeps
    "resolution_too_coarse",
]
NOT_IN_WINDOW_REASONS = REASONS[1:5]
UNINTERPRETABLE_REASONS = REASONS[5:]

REASON_TEXT = {
    "none": "The (1,1) cell is in the window and the electron count is anchored.",
    "no_transitions": "No charge transitions are visible in this window.",
    "occupancy_too_low": "The empty region is visible, but the window stops before one "
                         "electron is loaded in each dot.",
    "no_reference": "Transitions are visible, but there is no empty region to count "
                    "electrons from.",
    "partially_visible": "The (1,1) cell is only partly inside the window.",
    "low_snr": "The noise is too high relative to the transition contrast.",
    "sensor_insensitive": "The charge sensor has lost sensitivity over much of the scan.",
    "dots_merged": "The pattern looks like a single dot: the two dots are too strongly coupled.",
    "charge_instability": "The pattern jumps between sweeps (charge switching).",
    "resolution_too_coarse": "Too few points per electron to resolve the charge cells.",
}

# Line families used for annotation and for the model's line channels.
#   a         : boundary where only dot a's electron number changes (reservoir transition)
#   b         : boundary where only dot b's electron number changes
#   interdot  : an electron moves between dot a and dot b (total number conserved)
#   spectator : a transition of a dot that is not being swept
#   sensor    : a feature caused by the charge sensor itself (e.g. crossing a sensor peak)
LINE_FAMILIES = ["a", "b", "interdot", "spectator", "sensor"]

N_OCC_CLASSES = 5          # occupancy classes 0,1,2,3,4+ (per dot)
OCC_IGNORE = -1            # label value meaning "unknown / not anchored"


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def new_id(prefix: str = "scan") -> str:
    """Ids sort in creation order (to the millisecond): the analysis history relies on it."""
    now = _dt.datetime.now()
    stamp = now.strftime("%Y%m%d-%H%M%S") + f"{now.microsecond // 1000:03d}"
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:6]}"


# ---------------------------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------------------------
@dataclass
class Scan:
    signal: np.ndarray                      # (ny, nx) float32
    x: np.ndarray                           # (nx,) volts, ascending
    y: np.ndarray                           # (ny,) volts, ascending
    x_gate: str
    y_gate: str
    id: str = field(default_factory=new_id)
    device: str = "default"
    cooldown: str = ""
    kind: str = "PvP"                       # PvP is what the (1,1) finder handles
    voltage_state: dict[str, float] = field(default_factory=dict)
    fast_axis: str = "y"                    # which gate is swept in the inner loop
    source: str = "file"                    # file | api | synthetic | virtual_device
    created: str = field(default_factory=now_iso)
    units: str = "a.u."
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.signal = np.asarray(self.signal, dtype=np.float32)
        self.x = np.asarray(self.x, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=np.float64)
        if self.signal.shape != (len(self.y), len(self.x)):
            raise ValueError(
                f"signal shape {self.signal.shape} does not match (len(y), len(x)) = "
                f"({len(self.y)}, {len(self.x)})"
            )
        # enforce ascending axes
        if len(self.x) > 1 and self.x[0] > self.x[-1]:
            self.x = self.x[::-1].copy()
            self.signal = self.signal[:, ::-1].copy()
        if len(self.y) > 1 and self.y[0] > self.y[-1]:
            self.y = self.y[::-1].copy()
            self.signal = self.signal[::-1, :].copy()

    @property
    def shape(self) -> tuple[int, int]:
        return self.signal.shape  # type: ignore[return-value]

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return float(self.x[0]), float(self.x[-1]), float(self.y[0]), float(self.y[-1])

    @property
    def pitch(self) -> tuple[float, float]:
        dx = float(np.median(np.diff(self.x))) if len(self.x) > 1 else 0.0
        dy = float(np.median(np.diff(self.y))) if len(self.y) > 1 else 0.0
        return dx, dy

    def meta(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "x_gate": self.x_gate,
            "y_gate": self.y_gate,
            "device": self.device,
            "cooldown": self.cooldown,
            "kind": self.kind,
            "voltage_state": self.voltage_state,
            "fast_axis": self.fast_axis,
            "source": self.source,
            "created": self.created,
            "units": self.units,
            "notes": self.notes,
            "extra": self.extra,
            "shape": list(self.shape),
            "extent": list(self.extent),
        }

    @classmethod
    def from_meta(cls, meta: dict[str, Any], signal, x, y) -> "Scan":
        keep = {k: meta[k] for k in (
            "id", "x_gate", "y_gate", "device", "cooldown", "kind", "voltage_state",
            "fast_axis", "source", "created", "units", "notes", "extra") if k in meta}
        return cls(signal=signal, x=x, y=y, **keep)
