"""Voltage formatting for operator-facing text.

Devices and technologies use very different voltage scales, so nothing assumes millivolts: the
unit and number of decimals follow the size of the value.
"""
from __future__ import annotations

import math


def fmt_dv(v: float, signed: bool = True) -> str:
    """A voltage difference, e.g. '+12.5 mV', '-0.240 V', '+35 µV'."""
    a = abs(v)
    sign = "+" if signed else ""
    if a >= 1.0:
        return f"{v:{sign}.3f} V" if signed else f"{a:.3f} V"
    if a >= 1e-3:
        return f"{v * 1e3:{sign}.1f} mV" if signed else f"{a * 1e3:.1f} mV"
    return f"{v * 1e6:{sign}.0f} µV" if signed else f"{a * 1e6:.0f} µV"


def fmt_mag(v: float) -> str:
    """An unsigned voltage difference, e.g. '12.5 mV'."""
    return fmt_dv(v, signed=False)


def decimals_for(span: float) -> int:
    """Decimals (in volts) that resolve about a thousandth of ``span``."""
    if not span or not math.isfinite(span):
        return 4
    return int(min(9, max(3, math.ceil(-math.log10(abs(span) / 1000.0)))))


def fmt_v(v: float, span: float | None = None) -> str:
    """An absolute voltage in volts, precise enough for a window of width ``span``."""
    return f"{v:.{decimals_for(span) if span else 4}f} V"
