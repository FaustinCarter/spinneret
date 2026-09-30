"""Resampling and feature extraction shared by training and inference.

The network always sees a fixed-size square image (``size`` x ``size``) with three channels:
robust-normalised signal and its smoothed x/y derivatives. Charge transitions are steps in the
signal, so the derivatives make them local and polarity-consistent features.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import schema


def resample(arr: np.ndarray, size: int, order: int) -> np.ndarray:
    """Resample (ny,nx[,...]) to (size,size[,...]). order=0 for labels, 1 for signals."""
    ny, nx = arr.shape[:2]
    if (ny, nx) == (size, size):
        return arr.copy()
    zy, zx = size / ny, size / nx
    if order == 0:
        iy = np.clip(((np.arange(size) + 0.5) / zy - 0.5).round().astype(int), 0, ny - 1)
        ix = np.clip(((np.arange(size) + 0.5) / zx - 0.5).round().astype(int), 0, nx - 1)
        return arr[iy][:, ix]
    src = arr.astype(np.float32)
    # light anti-aliasing when shrinking
    sig = [max(0.0, (1 / zy - 1) / 2), max(0.0, (1 / zx - 1) / 2)]
    if max(sig) > 0:
        src = ndimage.gaussian_filter(src, sigma=sig, mode="nearest")
    yy = (np.arange(size) + 0.5) / zy - 0.5
    xx = (np.arange(size) + 0.5) / zx - 0.5
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    return ndimage.map_coordinates(src, [Y, X], order=1, mode="nearest").astype(np.float32)


def robust_scale(a: np.ndarray) -> tuple[float, float]:
    med = float(np.median(a))
    q1, q3 = np.percentile(a, [10, 90])
    scale = float(q3 - q1) / 2.56 if q3 > q1 else float(np.std(a)) + 1e-9
    return med, max(scale, 1e-9)


def features(sig: np.ndarray) -> np.ndarray:
    """(size,size) signal -> (3,size,size) float32 network input."""
    sig = np.nan_to_num(sig.astype(np.float32))
    med, sc = robust_scale(sig)
    z = np.clip((sig - med) / sc, -6, 6)
    sm = ndimage.gaussian_filter(z, 0.8, mode="nearest")
    gy, gx = np.gradient(sm)
    g_scale = float(np.percentile(np.abs(np.concatenate([gx.ravel(), gy.ravel()])), 99)) + 1e-6
    gx = np.clip(gx / g_scale, -3, 3)
    gy = np.clip(gy / g_scale, -3, 3)
    return np.stack([z, gx, gy]).astype(np.float32)


def pack_lines(masks: dict[str, np.ndarray], families=None) -> np.ndarray:
    """Line masks -> uint8 bitmask, one bit per family in the kind's order (default PvP)."""
    out = np.zeros(next(iter(masks.values())).shape, np.uint8)
    for bit, fam in enumerate(families or schema.LINE_FAMILIES):
        if fam in masks:
            out |= (masks[fam].astype(np.uint8) << bit)
    return out


def unpack_lines(packed: np.ndarray, n_families: int | None = None) -> np.ndarray:
    """uint8 bitmask (...,H,W) -> float (...,F,H,W)."""
    bits = [(packed >> b) & 1 for b in range(n_families or len(schema.LINE_FAMILIES))]
    return np.stack(bits, axis=-3).astype(np.float32)


def dilate(mask: np.ndarray, it: int = 1) -> np.ndarray:
    return ndimage.binary_dilation(mask, iterations=it) if it > 0 else mask
