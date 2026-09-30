"""Classical quality checks. They run before the model and catch data that should never reach it
(broken files, constant images, tiny scans) and produce numbers the guidance uses (noise level,
clipping). Charge switching is left to the model: classically it is indistinguishable from a
transition line running parallel to the fast axis."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .schema import Scan


def quality_metrics(scan: Scan) -> dict:
    sig = scan.signal.astype(float)
    ny, nx = sig.shape
    out: dict = {"shape": [ny, nx], "warnings": [], "hard_fail": None}
    finite = np.isfinite(sig)
    out["finite_fraction"] = float(finite.mean())
    if finite.mean() < 0.95:
        out["hard_fail"] = "invalid_values"
        out["warnings"].append("More than 5% of the points are NaN or infinite.")
        return out
    sig = np.where(finite, sig, np.nanmedian(sig))
    if min(ny, nx) < 16:
        out["hard_fail"] = "too_small"
        out["warnings"].append(f"The scan is only {nx} x {ny} points; use at least 32 per axis.")
        return out
    if np.ptp(sig) <= 1e-12 * (abs(sig).max() + 1e-30):
        out["hard_fail"] = "constant"
        out["warnings"].append("Every point has the same value: the sensor or readout is not "
                               "responding.")
        return out

    lo, hi = sig.min(), sig.max()
    tol = 1e-6 * (hi - lo)
    clipped = float(((sig <= lo + tol) | (sig >= hi - tol)).mean())
    out["clipped_fraction"] = clipped
    if clipped > 0.05:
        out["warnings"].append(f"{clipped:.0%} of points sit at the minimum or maximum value; the "
                               "amplifier or digitiser may be saturating.")

    q10, q90 = np.percentile(sig, [10, 90])
    spread = max(q90 - q10, 1e-30)
    resid = sig - ndimage.median_filter(sig, size=3, mode="nearest")
    noise = 1.4826 * np.median(np.abs(resid - np.median(resid))) * 1.25
    out["noise_rel"] = float(noise / spread)

    out["points"] = [nx, ny]
    return out
