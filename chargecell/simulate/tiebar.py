"""Tie-bar scans: plunger vs plunger, zoomed on the (1,1)-(2,0) charge transition.

HRL evaluates a "tiebar" PvP with a CNN to get a rough measure of the interdot tunnel coupling
and an initial guess of the spin-to-charge readout coordinates (QPU paper, Fig. S20e). The tie
bar is the interdot transition segment joining the two triple points where (1,1) meets (2,0):

    TP_low  = (1,0) | (1,1) | (2,0)          TP_high = (1,1) | (2,0) | (2,1)

Dot a is on x, dot b on y, and "(2,0)" means dot a holds two electrons. Rendering reuses the PvP
physics (interdot tunnel coupling, charge sensor, noise); only the window and the labels differ.

Coupling. Across the tie bar the charge moves between the dots over a detuning width set by the
tunnel coupling t_c and the temperature. For the sensor derivative, FWHM ~ sqrt((3.07 t_c)^2 +
(3.53 kT)^2) in detuning energy (t_c >> kT: 2 t_c sqrt(2^(2/3) - 1) * 2; t_c << kT: 4 kT
arccosh(sqrt 2)). ChargeCell reports it without assuming a voltage or energy scale, as a ratio:
FWHM measured across the tie bar divided by the tie-bar length (both in volts). The length is
set by the mutual charging energy, so the ratio is roughly 3 t_c / E_m.
"""
from __future__ import annotations

import numpy as np

from .. import kinds, schema
from ..preprocess import dilate, pack_lines, resample
from .generator import (Artifacts, Window, _loguniform, oracle as pvp_oracle, render,
                        sample_artifacts, sample_device, sensor_peak_mask, voltage_grid)
from .physics import DeviceParams, classical_ground_state, sensor_current, sensor_slope

STRONG_RATIO = 0.6   # coupling ratio above which the triple points are no longer distinct
MIN_TIEBAR_SNR = 2.0  # contrast of the interdot step itself against the noise

FWHM_T = 3.07     # FWHM of the charge-transfer derivative in units of t_c (t_c >> kT)
FWHM_K = 3.53     # ... in units of kT (t_c << kT)


def static_energy(p: DeviceParams, n) -> float:
    n = np.asarray(n, float)
    e = 0.5 * float((p.U * n * (n - 1)).sum())
    for i in range(3):
        for j in range(i + 1, 3):
            e += p.Um[i, j] * n[i] * n[j]
    return e + float((p.Ev * np.maximum(0.0, n - 2)).sum())


def _state(pair, c, na, nb, nc):
    n = np.zeros(3)
    n[pair[0]], n[pair[1]], n[c] = na, nb, nc
    return n


def _meet(p: DeviceParams, states, pair, c, v_spec) -> np.ndarray:
    """(x, y) in mV where three charge states are degenerate (a triple point)."""
    rows, rhs = [], []
    for n, m in ((states[0], states[1]), (states[1], states[2])):
        d = n - m
        r = d @ p.lever                     # E(n) - E(m) = const - r . V
        rows.append([r[pair[0]], r[pair[1]]])
        rhs.append(d @ p.offset + static_energy(p, n) - static_energy(p, m) - r[c] * v_spec)
    return np.linalg.solve(np.array(rows), np.array(rhs))


def _raster_tps(p: DeviceParams, pair, v_spec: float, centre, lo0, hi0, n: int = 320):
    """Triple points where the (n_a, n_b) regions (1,0)|(1,1)|(2,0) and (1,1)|(2,0)|(2,1) meet on
    a fine classical ground-state map (spectator free). Falls back to the given points."""
    from scipy import ndimage
    a, b = pair
    c = ({0, 1, 2} - {a, b}).pop()
    half = np.array([p.addition_voltage(a), p.addition_voltage(b)]) * 0.9
    xs = np.linspace(centre[0] - half[0], centre[0] + half[0], n)
    ys = np.linspace(centre[1] - half[1], centre[1] + half[1], n)
    X, Y = np.meshgrid(xs, ys)
    V = np.zeros((n * n, 3))
    V[:, a], V[:, b], V[:, c] = X.ravel(), Y.ravel(), v_spec
    g = classical_ground_state(p, V).reshape(n, n, 3)
    na, nb = g[..., a], g[..., b]
    near = lambda i, j: ndimage.maximum_filter(((na == i) & (nb == j)).astype(np.uint8), 3) > 0
    r11, r20 = near(1, 1), near(2, 0)
    out = []
    for other, fallback in (((1, 0), lo0), ((2, 1), hi0)):
        jj, ii = np.nonzero(r11 & r20 & near(*other))
        out.append(np.array([xs[ii].mean(), ys[jj].mean()]) if len(ii) else np.asarray(fallback))
    return out[0], out[1]


def tiebar_geometry(p: DeviceParams, pair, v_spec: float) -> dict:
    """Classical triple points of the (1,1)-(2,0) tie bar, its direction and normal."""
    a, b = pair
    c = ({0, 1, 2} - {a, b}).pop()

    def tps(nc):
        lo = _meet(p, [_state(pair, c, 1, 0, nc), _state(pair, c, 1, 1, nc),
                       _state(pair, c, 2, 0, nc)], pair, c, v_spec)
        hi = _meet(p, [_state(pair, c, 1, 1, nc), _state(pair, c, 2, 0, nc),
                       _state(pair, c, 2, 1, nc)], pair, c, v_spec)
        return lo, hi

    def gs_at(pt):
        V = np.zeros((1, 3))
        V[0, a], V[0, b], V[0, c] = pt[0], pt[1], v_spec
        return classical_ground_state(p, V)[0]

    # the spectator's occupancy must be the one it really has at the tie bar (self-consistent)
    nc, (lo, hi) = 1, tps(1)
    for cand in (1, 0, 2, 3, 4):
        lo_c, hi_c = tps(cand)
        g = gs_at((lo_c + hi_c) / 2)
        if g[c] == cand and (g[a], g[b]) in ((1, 1), (2, 0)):
            nc, lo, hi = cand, lo_c, hi_c
            break
    # refine on a fine ground-state map: the spectator may change its occupancy right at a
    # triple point (strong mutual charging), which the fixed-spectator solution cannot capture
    lo, hi = _raster_tps(p, pair, v_spec, (lo + hi) / 2, lo, hi)
    vec = hi - lo
    L = float(np.hypot(*vec))
    u = vec / max(L, 1e-12)
    n = np.array([u[1], -u[0]])             # perpendicular; points to +x/-y, i.e. into (2,0)
    if n[0] - n[1] < 0:
        n = -n
    # detuning energy per mV along the normal: eps = E(1,1) - E(2,0)
    dlt = _state(pair, c, 1, 1, nc) - _state(pair, c, 2, 0, nc)
    r = dlt @ p.lever
    deps_du = abs(r[a] * n[0] + r[b] * n[1])
    return dict(tp_low=lo, tp_high=hi, mid=(lo + hi) / 2, length=L, u=u, normal=n,
                spectator_occ=nc, deps_du=float(deps_du))


def coupling_truth(p: DeviceParams, pair, geo: dict) -> dict:
    a, b = pair
    tc = float(p.t0[a, b])
    fwhm_e = float(np.hypot(FWHM_T * tc, FWHM_K * p.kT))
    fwhm_v = fwhm_e / max(geo["deps_du"], 1e-12)
    return dict(tc_meV=tc, kT_meV=float(p.kT), Um_meV=float(p.Um[a, b]), fwhm_mV=fwhm_v,
                coupling_ratio=fwhm_v / max(geo["length"], 1e-12))


# ---------------------------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------------------------
TIEBAR_NIW = ("no_tiebar", "partially_visible")


def sample_tiebar_window(rng, p: DeviceParams, pair, intent: str, coarse: bool = False,
                         v_spec: float | None = None) -> tuple[Window, dict]:
    a, b = pair
    c = ({0, 1, 2} - {a, b}).pop()
    if v_spec is None:
        v_spec = float(p.v11[c] + rng.normal(0, 0.15) * p.addition_voltage(c))
    geo = tiebar_geometry(p, pair, v_spec)
    dva, dvb = p.addition_voltage(a), p.addition_voltage(b)
    L = geo["length"]
    ext = np.abs(geo["u"]) * L                      # tie-bar extent along x and y
    for _ in range(50):
        wx = max(rng.uniform(0.35, 1.0) * dva, 1.4 * ext[0] + 0.1 * dva)
        wy = max(rng.uniform(0.35, 1.0) * dvb, 1.4 * ext[1] + 0.1 * dvb)
        if rng.random() < 0.5:                      # square in volts, as often measured
            wx = wy = max(wx, wy)
        W = np.array([wx, wy])
        if intent == schema.FOUND:
            centre = geo["mid"] + rng.normal(0, 0.08, 2) * W
        elif intent == "partially_visible":
            sgn = rng.choice([-1.0, 1.0])
            centre = (geo["mid"] + sgn * geo["u"] * (0.5 * L + rng.uniform(0.02, 0.35) * W.min())
                      + rng.normal(0, 0.05, 2) * W)
        else:                                        # no_tiebar
            ang = rng.uniform(0, 2 * np.pi)
            centre = geo["mid"] + np.array([np.cos(ang), np.sin(ang)]) * rng.uniform(0.75, 1.6) * W
        lo, hi = centre - W / 2, centre + W / 2
        inside = lambda pt, m=0.04: bool(np.all(pt > lo + m * W) and np.all(pt < hi - m * W))
        both = inside(geo["tp_low"]) and inside(geo["tp_high"])
        if (intent == schema.FOUND) == both or intent == "no_tiebar":
            break
    if coarse:
        px_per_L = rng.uniform(1.0, 3.5)
    else:
        px_per_L = rng.uniform(8.0, 50.0)
    pitch = max(L / px_per_L, 1e-9)
    nx = int(np.clip(round(wx / pitch), 12 if coarse else 40, 200))
    ny = int(np.clip(round(wy / pitch), 12 if coarse else 40, 200))
    w = Window(pair=(a, b), x0=float(lo[0]), x1=float(hi[0]), y0=float(lo[1]), y1=float(hi[1]),
               nx=nx, ny=ny, v_spectator=float(v_spec), fast_axis=str(rng.choice(["x", "y"])))
    return w, geo


def retune_sensor(p: DeviceParams, w: Window) -> None:
    """Reference the charge sensor at the window centre, as an operator retunes it before a
    zoomed scan (in place)."""
    V = voltage_grid(p, w)[2]
    c = (w.ny // 2) * w.nx + w.nx // 2
    p.v_ref = V[c].copy()
    p.n_ref = classical_ground_state(p, V[c:c + 1])[0].astype(float)


def readout_offset(fwhm: float, length: float) -> float:
    """How far past the tie bar, into (2,0), the first readout guess sits: clear of the
    transition (1.5 widths) but near it (a quarter to three quarters of the tie-bar length)."""
    return float(min(max(1.5 * fwhm, 0.25 * length), 0.75 * length))


# ---------------------------------------------------------------------------------------------
# ground truth
# ---------------------------------------------------------------------------------------------
def region_map(na: np.ndarray, nb: np.ndarray) -> np.ndarray:
    reg = np.full(na.shape, 4, np.int8)
    for k, (i, j) in enumerate(((1, 1), (2, 0), (1, 0), (2, 1))):
        reg[(na == i) & (nb == j)] = k
    return reg


def tiebar_masks(na: np.ndarray, nb: np.ndarray) -> dict:
    """Line masks (both sides of each boundary) for the tie-bar line families."""
    ny, nx = na.shape
    reg = region_map(na, nb)
    masks = {f: np.zeros((ny, nx), bool) for f in ("a", "b", "tiebar", "interdot")}
    for s0, s1 in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                   ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
        da = na[s1] != na[s0]
        db = nb[s1] != nb[s0]
        tot = (na[s1] + nb[s1]) != (na[s0] + nb[s0])
        inter = da & db & ~tot
        tb = inter & (((reg[s0] == 0) & (reg[s1] == 1)) | ((reg[s0] == 1) & (reg[s1] == 0)))
        for m, cond in ((masks["a"], da & ~db), (masks["b"], db & ~da),
                        (masks["tiebar"], tb), (masks["interdot"], inter & ~tb),
                        (masks["a"], da & db & tot), (masks["b"], da & db & tot)):
            m[s0] |= cond
            m[s1] |= cond
    return masks


def tiebar_oracle(p: DeviceParams, w: Window, art: Artifacts, rend: dict, geo: dict) -> dict:
    a, b = w.pair
    V = voltage_grid(p, w)[2]
    cg = classical_ground_state(p, V).reshape(w.ny, w.nx, 3)
    na, nb = cg[..., a], cg[..., b]
    masks = tiebar_masks(na, nb)
    edges = masks["a"] | masks["b"] | masks["tiebar"] | masks["interdot"]
    masks["sensor"] = sensor_peak_mask(rend["mu"], p.s_ES, edges)
    lo = np.array([w.x0, w.y0])
    hi = np.array([w.x1, w.y1])
    W = hi - lo
    inside = lambda pt: bool(np.all(pt > lo + 0.03 * W) and np.all(pt < hi - 0.03 * W))
    coupling = coupling_truth(p, w.pair, geo)
    px = W[0] / max(w.nx - 1, 1)
    py = W[1] / max(w.ny - 1, 1)
    px_across = geo["length"] / max(px, py)

    # interpretability from the PvP oracle (noise, sensor, jumps); resolution judged on the bar
    base = pvp_oracle(p, w, art, rend)
    # contrast of the tie-bar step itself: moving one electron from dot b to dot a shifts the
    # sensor by kappa_a - kappa_b, which is often much smaller than a reservoir step
    tb = masks["tiebar"]
    snr_tb = np.nan
    if tb.any():
        mu = rend["mu"]
        sigma_eff = np.sqrt(rend["sigma_white"] ** 2
                            + (np.median(np.abs(sensor_slope(p, mu))) * rend["sigma_mu"]) ** 2)
        dk = p.s_kappa[a] - p.s_kappa[b]
        contrast = np.abs(sensor_current(p, mu[tb] + dk / 2) - sensor_current(p, mu[tb] - dk / 2))
        snr_tb = float(np.percentile(contrast, 75) / sigma_eff)
    reason = None
    merged = (p.merged_pair is not None and set(p.merged_pair) == {a, b}) \
        or coupling["coupling_ratio"] > STRONG_RATIO
    if merged:
        reason = "dots_merged"
    elif px_across < 5.0:
        reason = "resolution_too_coarse"
    elif base["status"] == schema.UNINTERPRETABLE and base["reason"] in (
            "low_snr", "sensor_insensitive", "charge_instability"):
        reason = base["reason"]
    elif not np.isnan(snr_tb) and snr_tb < MIN_TIEBAR_SNR:
        reason = "low_snr"
    tb_visible = masks["tiebar"].sum() >= 3
    if reason is not None:
        status = schema.UNINTERPRETABLE
    elif tb_visible and inside(geo["tp_low"]) and inside(geo["tp_high"]):
        status, reason = schema.FOUND, "none"
    else:
        status = schema.NOT_IN_WINDOW
        reason = "partially_visible" if tb_visible else "no_tiebar"
    readout = geo["mid"] + geo["normal"] * readout_offset(coupling["fwhm_mV"], geo["length"])
    return dict(status=status, reason=reason, masks=masks, region=region_map(na, nb),
                tp_low=geo["tp_low"].tolist(), tp_high=geo["tp_high"].tolist(),
                mid=geo["mid"].tolist(), length=geo["length"], normal=geo["normal"].tolist(),
                readout=readout.tolist(), px_across=float(px_across), snr=base["snr"],
                snr_tiebar=None if np.isnan(snr_tb) else snr_tb,
                **coupling)


# ---------------------------------------------------------------------------------------------
# one-call sample and training arrays
# ---------------------------------------------------------------------------------------------
TIEBAR_UNINTERP = ("low_snr", "sensor_insensitive", "dots_merged", "charge_instability",
                   "resolution_too_coarse")


def generate_tiebar_sample(rng: np.random.Generator, preset: str = "mixed",
                           mix=(0.40, 0.35, 0.25)) -> dict:
    intent = rng.choice(schema.STATUSES, p=list(mix))
    art_kind, coarse, merged = "normal", False, False
    if intent == schema.UNINTERPRETABLE:
        art_kind = str(rng.choice(TIEBAR_UNINTERP))
        coarse = art_kind == "resolution_too_coarse"
        merged = art_kind == "dots_merged"
        w_intent = schema.FOUND if rng.random() < 0.7 else "partially_visible"
    elif intent == schema.NOT_IN_WINDOW:
        w_intent = str(rng.choice(TIEBAR_NIW))
    else:
        w_intent = schema.FOUND
    pair = tuple(int(v) for v in rng.permutation(3)[:2])
    p = sample_device(rng, preset, merged_pair=pair if merged else None)
    # coupled neighbours (as in a real tie-bar scan), from thermally limited to strong
    a, b = pair
    if not merged:
        um = p.Um[a, b] if p.Um[a, b] > 0 else 0.15 * float(np.mean(p.U))
        p.Um[a, b] = p.Um[b, a] = max(um, 0.08 * float(np.mean(p.U)))
        p.t0[a, b] = p.t0[b, a] = float(_loguniform(rng, 0.002, 0.3 * p.Um[a, b]))
    if rng.random() < 0.85:                 # the reservoirs were tuned before zooming in
        p.latch = np.zeros(3)
    # the sensor was positioned to see charge move between the two dots: one of them couples
    # clearly more strongly than the other
    near, far = (a, b) if p.s_kappa[a] >= p.s_kappa[b] else (b, a)
    p.s_kappa[far] = min(p.s_kappa[far], p.s_kappa[near] * rng.uniform(0.15, 0.65))
    w, geo = sample_tiebar_window(rng, p, pair, w_intent, coarse=coarse)
    if rng.random() < 0.85:                 # the sensor was retuned for the zoomed window
        retune_sensor(p, w)
    art = sample_artifacts(rng, "normal" if art_kind in ("dots_merged", "resolution_too_coarse")
                           else art_kind)
    rend = render(p, w, art, rng)
    truth = tiebar_oracle(p, w, art, rend, geo)
    return dict(params=p, window=w, artifacts=art, render=rend, truth=truth, geometry=geo,
                intent=str(intent))


def tiebar_to_arrays(s: dict, size: int) -> tuple[dict, dict]:
    spec = kinds.TIEBAR
    r, t, w, p = s["render"], s["truth"], s["window"], s["params"]
    a, b = w.pair
    sig = resample(r["signal"], size, 1)
    V = voltage_grid(p, w)[2]
    cg = classical_ground_state(p, V).reshape(w.ny, w.nx, 3)
    cg_r = resample(cg, size, 0)
    na, nb = cg_r[..., a], cg_r[..., b]
    masks = tiebar_masks(na, nb)
    masks["sensor"] = resample(dilate(t["masks"]["sensor"], 1).astype(np.uint8), size, 0) > 0
    arrays = dict(
        signal=sig.astype(np.float16),
        occ=region_map(na, nb)[None].astype(np.int8),
        lines=pack_lines(masks, spec.line_families),
        status=schema.STATUSES.index(t["status"]),
        reason=spec.reasons.index(t["reason"]),
        ref=np.zeros(0, np.int8),
    )
    meta = dict(
        kind="tiebar", status=t["status"], reason=t["reason"], pair=list(w.pair), snr=t["snr"],
        tp_low_mV=t["tp_low"], tp_high_mV=t["tp_high"], length_mV=t["length"],
        readout_mV=t["readout"], tc_meV=t["tc_meV"], kT_meV=t["kT_meV"], Um_meV=t["Um_meV"],
        fwhm_mV=t["fwhm_mV"], coupling_ratio=t["coupling_ratio"],
        window_mV=[w.x0, w.x1, w.y0, w.y1], native_shape=[w.ny, w.nx], fast_axis=w.fast_axis,
        artifact=s["artifacts"].kind,
    )
    return arrays, meta
