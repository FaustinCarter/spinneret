"""Transition-line lattice from model outputs, in index space.

For dot a the lines are near-vertical: i = m * j + c (i: column index, j: row index).
For dot b they are near-horizontal: j = m * i + c. Both are handled by one routine that works
on arrays where the counting direction is along axis 1 (dot b uses transposed arrays).

When dot a is anchored (empty region visible), boundaries come from the occupancy map and each
line gets its electron index k (the k -> k+1 transition). Otherwise lines come from the line
channel and their indices are unknown.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def _fit(i: np.ndarray, j: np.ndarray, S: int) -> tuple[float, float, np.ndarray]:
    """Least squares i = m j + c with one round of outlier trimming. Returns (m, c_center, keep)."""
    keep = np.ones(len(i), bool)
    m, c = 0.0, float(np.median(i))
    for _ in range(2):
        if keep.sum() < 3 or np.ptp(j[keep]) < 2:
            m, c = 0.0, float(np.median(i[keep]))
        else:
            m, c = np.polyfit(j[keep], i[keep], 1)
        r = np.abs(i - (m * j + c))
        keep = r <= max(2.0, 2.5 * np.median(r[keep]) + 1e-6)
    jc = (S - 1) / 2
    return float(m), float(m * jc + c), keep


def _from_occupancy(occ: np.ndarray, S: int) -> list[dict]:
    lines = []
    for k in range(int(occ.min()), min(int(occ.max()), 4)):
        pts_i, pts_j = [], []
        for j in range(S):
            row = occ[j]
            hit = np.nonzero(row > k)[0]
            if len(hit) and hit[0] > 0 and (row[:hit[0]] <= k).all():
                pts_i.append(hit[0] - 0.5)
                pts_j.append(j)
        if len(pts_j) >= max(4, S // 5):
            m, c, keep = _fit(np.array(pts_i, float), np.array(pts_j, float), S)
            if abs(m) < 2.0:
                lines.append(dict(c=c, m=m, index=k, n=int(keep.sum())))
    return lines


def _from_line_map(p: np.ndarray, p_inter: np.ndarray, S: int) -> list[dict]:
    """Lines of one family from the line channels.

    A boundary between k and k+1 electrons is a staircase: reservoir segments joined by interdot
    segments. Each interdot segment belongs to exactly one boundary of each dot, so joining the
    family's pixels through the interdot channel recovers whole staircases. Fitting segments
    separately would count each jog as a new line and underestimate the spacing.
    """
    fam = p > 0.5
    union = fam | (p_inter > 0.5)
    lab, n = ndimage.label(ndimage.binary_dilation(union), structure=np.ones((3, 3)))
    segs = []
    for k in range(1, n + 1):
        comp = (lab == k) & union
        if not (comp & fam).any():
            continue
        jj, ii = np.nonzero(comp)
        if np.ptp(jj) < max(3, S // 12):
            continue
        m, c, keep = _fit(ii.astype(float), jj.astype(float), S)
        if abs(m) < 1.5:
            segs.append(dict(c=c, m=m, ii=ii, jj=jj))
    if not segs:
        return []
    segs.sort(key=lambda s: s["c"])
    clusters = [[segs[0]]]
    for s in segs[1:]:
        if s["c"] - clusters[-1][-1]["c"] < max(3.0, 0.12 * S):
            clusters[-1].append(s)
        else:
            clusters.append([s])
    lines = []
    for cl in clusters:
        ii = np.concatenate([s["ii"] for s in cl]).astype(float)
        jj = np.concatenate([s["jj"] for s in cl]).astype(float)
        if np.ptp(jj) < 0.25 * S:
            continue
        m, c, keep = _fit(ii, jj, S)
        lines.append(dict(c=c, m=m, index=None, n=int(keep.sum())))
    return lines


def _summarise(lines: list[dict], source: str) -> dict:
    cs = sorted(l["c"] for l in lines)
    spacing = float(np.median(np.diff(cs))) if len(cs) >= 2 else None
    slope = float(np.median([l["m"] for l in lines])) if lines else None
    return dict(lines=sorted(lines, key=lambda l: l["c"]), spacing=spacing, slope=slope,
                source=source)


def extract_lattice(occ_a: np.ndarray, occ_b: np.ndarray, lines_p: np.ndarray,
                    ref_a: bool, ref_b: bool) -> dict:
    """occ_*: (S,S) argmax occupancy; lines_p: (5,S,S) probabilities (a, b, interdot, ...)."""
    S = occ_a.shape[0]
    a = _from_occupancy(occ_a, S) if ref_a else []
    a_src = "occupancy" if a else "lines"
    if not a:
        a = _from_line_map(lines_p[0], lines_p[2], S)
    b = _from_occupancy(occ_b.T, S) if ref_b else []
    b_src = "occupancy" if b else "lines"
    if not b:
        b = _from_line_map(lines_p[1].T, lines_p[2].T, S)
    return {"a": _summarise(a, a_src), "b": _summarise(b, b_src), "size": S}
