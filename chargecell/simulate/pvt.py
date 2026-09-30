"""Plunger-vs-tunnel-gate (PvT) scans of an edge dot next to its reservoir.

HRL's tune-up loads one electron into each plunger with PvT scans: the dot's plunger P is swept
against the tunnel gate T between the dot and its reservoir, and the "loading lines" (the dot's
0->1, 1->2, ... transitions) are read off (HRL QPU paper, Supplement S5 and Fig. S20c).

Model, on top of the triple-dot physics in ``physics.py``:

* T shifts every dot's energy through its own lever arm (strongest on the edge dot), so the
  loading lines are tilted in the (P, T) plane: dP/dT = -r_T, the ratio of the lever arms.
* The reservoir tunnel rate grows exponentially with T. With tau the time per pixel,
  log10(Gamma tau) = (V_T - T_open)/beta10 - gamma (V_P - P_ref)/beta10, so Gamma tau = 1 at
  T_open. Below T_open electrons cannot follow the sweep: transitions latch, shift along the
  sweep direction and disappear (a stochastic process per sweep line).
* Far above T_open, hbar Gamma exceeds kT and lifetime broadening smears the lines. This starts
  D_b = log10(kT tau / hbar) decades above T_open: 5-7 decades for microsecond-to-millisecond
  pixels at 50-300 mK.

All scales are drawn relative to the dot's addition voltage, so no voltage scale is assumed.
Units follow physics.py: mV and meV.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .. import kinds, schema
from ..preprocess import dilate, pack_lines, resample
from .generator import (Artifacts, _loguniform, _slow_index, measure_sensor, sample_artifacts,
                        sample_device, sensor_peak_mask)
from .physics import DeviceParams, classical_ground_state, occupations, sensor_slope

SLOW_GT = 0.3          # Gamma*tau below this: electrons do not follow the sweep ("slow")
OPEN_BROAD = 1.5       # hbar*Gamma / kT above this: lines visibly lifetime-broadened ("open")
FAST_AVERAGE_GT = 20.0  # Gamma*tau above this: the sensor averages many tunnelling events


@dataclass
class PvTParams:
    dot: int                  # the edge dot next to the reservoir
    t_lever: np.ndarray       # (3,) meV/mV: tunnel-gate lever arms on the three dots
    T_open: float             # mV: tunnel gate where Gamma*tau = 1 (at P = P_ref)
    beta10: float             # mV of tunnel gate per decade of tunnel rate
    gamma: float              # effect of the plunger on log(rate), relative to the tunnel gate
    P_ref: float              # mV
    D_b: float                # decades from Gamma*tau = 1 to hbar*Gamma = kT
    s_lever_T: float          # sensor crosstalk from T after compensation, widths/mV

    def log_gt(self, P, T):
        return (np.asarray(T) - self.T_open) / self.beta10 \
            - self.gamma * (np.asarray(P) - self.P_ref) / self.beta10

    def T_where(self, log_gt: float, P: float) -> float:
        """Tunnel-gate voltage where log10(Gamma tau) = log_gt at plunger voltage P."""
        return float(self.T_open + self.beta10 * log_gt + self.gamma * (P - self.P_ref))

    @property
    def T_broad(self) -> float:
        return self.T_open + self.D_b * self.beta10

    def to_dict(self) -> dict:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict) -> "PvTParams":
        return cls(**{k: (np.asarray(v, float) if k == "t_lever" else v) for k, v in d.items()})


@dataclass
class PvTWindow:
    dot: int
    x0: float                 # plunger (mV)
    x1: float
    y0: float                 # tunnel gate (mV)
    y1: float
    nx: int
    ny: int
    v_plungers: list          # (3,) mV: DC value of every plunger (entry `dot` is swept)
    fast_axis: str = "x"


def neighbour(dot: int) -> int:
    return 1


def sample_pvt_params(rng: np.random.Generator, p: DeviceParams, dot: int = 0) -> PvTParams:
    dva = p.addition_voltage(dot)
    r_T = rng.uniform(0.2, 0.7)
    beta10 = float(rng.uniform(0.2, 1.2) * dva)
    t_lever = np.zeros(3)
    t_lever[dot] = r_T * p.lever[dot, dot]
    nb = neighbour(dot)
    t_lever[nb] = r_T * p.lever[nb, nb] * rng.uniform(0.05, 0.3)
    return PvTParams(
        dot=dot, t_lever=t_lever,
        T_open=float(p.v11[dot] * rng.uniform(0.5, 1.5)),
        beta10=beta10,
        # the plunger changes the rate by at most half a decade per electron
        gamma=float(rng.uniform(-0.5, 0.5) * beta10 / dva),
        P_ref=float(p.v11[dot]),
        D_b=float(rng.uniform(4.5, 7.0)),
        s_lever_T=float(p.s_lever[dot] * r_T * rng.normal(0.0, 0.08)),
    )


def pvt_device(p: DeviceParams, q: PvTParams) -> DeviceParams:
    """The triple dot with offsets shifted so that its (1,1,1) cell sits at the usual plunger
    voltages when the tunnel gate is at T_open (the tunnel gate enters through the offsets)."""
    d = DeviceParams.from_dict(p.to_dict())
    d.offset = p.offset + q.t_lever * q.T_open
    return d


def line_position(p: DeviceParams, q: PvTParams, k: int, T: float, v_plungers) -> float:
    """Plunger voltage (mV) of the dot's k->k+1 loading line at tunnel gate T (classical)."""
    d = q.dot
    dva = p.addition_voltage(d)
    P = np.linspace(q.P_ref - 6 * dva, q.P_ref + 6 * dva, 1201)
    V = np.tile(np.asarray(v_plungers, float), (len(P), 1))
    V[:, d] = P
    off = pvt_device(p, q).offset[None, :] - q.t_lever[None, :] * T
    n = classical_ground_state(p, V, offset=np.repeat(off, len(P), 0))[:, d]
    idx = np.nonzero((n[:-1] <= k) & (n[1:] > k))[0]
    return float(P[idx[0]] + (P[1] - P[0]) / 2) if len(idx) else float("nan")


# ---------------------------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------------------------
PVT_NIW = ("no_transitions", "occupancy_too_low", "no_reference", "tunnel_rate_too_low",
           "reservoir_too_open")


def sample_pvt_window(rng, p: DeviceParams, q: PvTParams, intent: str, coarse: bool = False,
                      v_plungers=None) -> PvTWindow:
    d = q.dot
    dva = p.addition_voltage(d)
    r_T = q.t_lever[d] / p.lever[d, d]
    if v_plungers is None:
        v_plungers = p.v11.copy()
        for j in range(3):
            if j == d:
                continue
            dvj = p.addition_voltage(j)
            v_plungers[j] = (p.v11[j] - rng.uniform(0.8, 2.0) * dvj if rng.random() < 0.7
                             else p.v11[j] + rng.normal(0, 0.2) * dvj)
    # tunnel-gate range, in decades of tunnel rate
    h_dec = rng.uniform(2.0, 8.0)
    H = h_dec * q.beta10
    if intent == "tunnel_rate_too_low":
        y1 = q.T_open - rng.uniform(0.2, 2.0) * q.beta10
        y0 = y1 - H
    elif intent == "reservoir_too_open":
        y0 = q.T_broad + rng.uniform(0.3, 2.0) * q.beta10
        y1 = y0 + H
    else:
        # the clean band [T_open, T_broad] covers a good part of the window
        band_lo, band_hi = q.T_open, q.T_broad
        centre = rng.uniform(band_lo + 0.15 * (band_hi - band_lo), band_hi - 0.15 * (band_hi - band_lo))
        y0 = centre - rng.uniform(0.3, 0.7) * H
        y1 = y0 + H
    Tm = 0.5 * (y0 + y1)
    P0 = line_position(p, q, 0, Tm, v_plungers)
    if not np.isfinite(P0):
        P0 = q.P_ref - 0.5 * dva
    tilt = r_T * H                         # how far the lines move in P across the T range
    p0_lo, p0_hi = P0 - tilt / 2, P0 + tilt / 2
    if intent in (schema.FOUND, "tunnel_rate_too_low", "reservoir_too_open"):
        x0 = p0_lo - rng.uniform(0.5, 1.3) * dva
        x1 = p0_hi + dva + rng.uniform(0.4, 1.2) * dva
    elif intent == "no_transitions":
        x1 = p0_lo - rng.uniform(0.15, 1.0) * dva
        x0 = x1 - rng.uniform(1.5, 4.0) * dva
    elif intent == "occupancy_too_low":
        # the 1->2 line (at >= p0_lo + dva on every row) stays outside the window
        x1 = p0_lo + rng.uniform(0.3, 0.85) * dva
        x0 = min(p0_lo - rng.uniform(0.5, 1.5) * dva, x1 - 1.2 * dva)
    else:                                   # no_reference
        x0 = p0_hi + rng.uniform(0.15, 1.0) * dva
        x1 = x0 + rng.uniform(1.5, 4.0) * dva
    wx = x1 - x0
    ppa = rng.uniform(1.5, 4.0) if coarse else rng.uniform(7.0, 30.0)
    nx = int(np.clip(round(wx / dva * ppa), 16 if coarse else 40, 220))
    ny = int(np.clip(round(h_dec * rng.uniform(4.0, 20.0)), 24, 160))
    return PvTWindow(dot=d, x0=float(x0), x1=float(x1), y0=float(y0), y1=float(y1), nx=nx,
                     ny=ny, v_plungers=[float(v) for v in v_plungers],
                     fast_axis=str(rng.choice(["x", "y"], p=[0.7, 0.3])))


# ---------------------------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------------------------
def pvt_grid(p: DeviceParams, q: PvTParams, w: PvTWindow):
    xs = np.linspace(w.x0, w.x1, w.nx)
    ys = np.linspace(w.y0, w.y1, w.ny)
    X, Y = np.meshgrid(xs, ys)
    V = np.tile(np.asarray(w.v_plungers, float), (w.nx * w.ny, 1))
    V[:, w.dot] = X.ravel()
    off = pvt_device(p, q).offset[None, :] - q.t_lever[None, :] * Y.ravel()[:, None]
    return xs, ys, X, Y, V, off


def _latch(mean: np.ndarray, gt: np.ndarray, w: PvTWindow, dots, rng) -> np.ndarray:
    """Observed occupancy when the reservoir is slow. Each pixel relaxes toward a sample of the
    equilibrium occupancy with probability 1 - exp(-Gamma tau); a line starts from the state its
    predecessor ended in (the fly-back is too fast to relax)."""
    ny, nx = w.ny, w.nx
    m = mean.reshape(ny, nx, 3)
    g = gt.reshape(ny, nx)
    if w.fast_axis == "x":                 # lines are rows, time runs along x
        m_l, g_l = m, g
    else:                                  # lines are columns
        m_l, g_l = m.transpose(1, 0, 2), g.T
    n_lines, n_fast = g_l.shape
    obs = m_l.copy()
    p_relax = 1.0 - np.exp(-np.minimum(g_l, 50.0))
    for d in dots:
        md = m_l[:, :, d]
        sample = lambda v: np.floor(v) + (rng.random(v.shape) < (v - np.floor(v)))
        end_eq = sample(md[:, -1])
        state = np.concatenate([end_eq[:1], end_eq[:-1]])    # previous line's end state
        out = np.empty_like(md)
        for t in range(n_fast):
            target = sample(md[:, t])
            relax = rng.random(n_lines) < p_relax[:, t]
            state = np.where(relax, target, state)
            out[:, t] = np.where(g_l[:, t] > FAST_AVERAGE_GT, md[:, t], state)
        obs[:, :, d] = out
    obs = obs if w.fast_axis == "x" else obs.transpose(1, 0, 2)
    return obs.reshape(-1, 3)


def render_pvt(p: DeviceParams, q: PvTParams, w: PvTWindow, art: Artifacts,
               rng: np.random.Generator) -> dict:
    xs, ys, X, Y, V, off = pvt_grid(p, q, w)
    lg = q.log_gt(X.ravel(), Y.ravel())
    hg_kt = 10.0 ** (lg - q.D_b)                         # hbar Gamma / kT
    kT_eff = p.kT * np.sqrt(1.0 + (0.5 * hg_kt) ** 2)    # lifetime broadening
    # charge jumps between sweeps shift the offsets
    slow = _slow_index(w).ravel()
    n_slow = w.nx if w.fast_axis == "y" else w.ny
    off = off.copy()
    for frac, d_off in art.jumps:
        off[slow >= int(frac * n_slow)] += np.asarray(d_off) * p.U
    mean, gs = occupations(p, V, offset=off, kT=kT_eff)
    obs = _latch(mean, 10.0 ** lg, w, [q.dot, neighbour(q.dot)], rng)
    # the operator tunes the sensor for this scan: reference it at the window centre
    ps = DeviceParams.from_dict(p.to_dict())
    c = (w.ny // 2) * w.nx + w.nx // 2
    ps.v_ref, ps.n_ref = V[c].copy(), gs[c].astype(float)
    mu_extra = -q.s_lever_T * (Y.ravel() - Y.ravel()[c])
    meas = measure_sensor(ps, V, obs, w, art, rng, mu_extra=mu_extra)
    return dict(x=xs, y=ys, occ=gs.reshape(w.ny, w.nx, 3), log_gt=lg.reshape(w.ny, w.nx),
                **meas)


# ---------------------------------------------------------------------------------------------
# ground truth
# ---------------------------------------------------------------------------------------------
def regime_map(q: PvTParams, lg: np.ndarray) -> np.ndarray:
    """0 slow (electrons do not follow), 1 good, 2 open (lifetime-broadened)."""
    reg = np.ones(lg.shape, np.int8)
    reg[10.0 ** lg < SLOW_GT] = 0
    reg[10.0 ** (lg - q.D_b) > OPEN_BROAD] = 2
    return reg


def _edges(o: np.ndarray) -> np.ndarray:
    m = np.zeros(o.shape, bool)
    dx = o[:, 1:] != o[:, :-1]
    dy = o[1:, :] != o[:-1, :]
    m[:, 1:] |= dx
    m[:, :-1] |= dx
    m[1:, :] |= dy
    m[:-1, :] |= dy
    return m


def pvt_oracle(p: DeviceParams, q: PvTParams, w: PvTWindow, art: Artifacts, rend: dict) -> dict:
    d, nb = q.dot, neighbour(q.dot)
    occ = rend["occ"]
    od = occ[..., d]
    ny, nx = od.shape
    dva = p.addition_voltage(d)
    px = (w.x1 - w.x0) / max(nx - 1, 1)
    reg = regime_map(q, rend["log_gt"])
    nonslow = reg != 0
    load = _edges(od) & nonslow
    others = [j for j in range(3) if j != d]
    spect = np.zeros_like(load)
    for j in others:
        spect |= _edges(occ[..., j]) & nonslow
    masks = {"load": load, "spectator": spect & ~load}
    masks["sensor"] = sensor_peak_mask(rend["mu"], p.s_ES, _edges(od) | spect)

    # anchoring: an empty dot next to a visible transition, where electrons can follow
    has0 = ((od == 0) & nonslow).mean() > 0.03
    has1 = ((od >= 1) & nonslow).mean() > 0.03
    rows = nonslow.any(1)
    width0 = ((od == 0) & nonslow).sum(1) * px
    wide = (width0[rows] >= 0.25 * dva).mean() > 0.2 if rows.any() else False
    ref = bool(has0 and has1 and wide)

    # rows (tunnel-gate values) where the k->k+1 line is crossed with electrons following
    # cleanly (good) or at least visibly (not slow)
    def rows_with(k, ok):
        n = 0
        for j in range(ny):
            cross = np.nonzero((od[j, :-1] <= k) & (od[j, 1:] > k))[0]
            if len(cross) and ok(reg[j, cross[0]]):
                n += 1
        return n

    need = max(3, int(0.15 * ny))
    g0, g1 = rows_with(0, lambda r: r == 1), rows_with(1, lambda r: r == 1)
    v0, v1 = rows_with(0, lambda r: r != 0), rows_with(1, lambda r: r != 0)
    any_true = _edges(od).any()
    frac_slow = float((reg == 0).mean())
    frac_open = float((reg == 2).mean())

    # interpretability (as for PvP): contrast of the loading steps against the noise
    mu = rend["mu"]
    sigma_eff = np.sqrt(rend["sigma_white"] ** 2
                        + (np.median(np.abs(sensor_slope(p, mu))) * rend["sigma_mu"]) ** 2)
    snr, frac_weak = np.nan, 0.0
    if load.sum() >= 5:
        from .physics import sensor_current
        contrast = np.abs(sensor_current(p, mu[load]) - sensor_current(p, mu[load] - p.s_kappa[d]))
        c95 = np.percentile(contrast, 95)
        snr = float(np.percentile(contrast, 75) / sigma_eff)
        frac_weak = float((contrast < max(1.0 * sigma_eff, 0.1 * c95)).mean())
    frac_flat = float((np.abs(sensor_slope(p, mu)) < 0.2 * 0.77 * p.s_amp).mean())
    big_jumps = sum(1 for _, dd in art.jumps if np.max(np.abs(dd)) >= 0.15)
    ppa = dva / px

    reason = None
    if ppa < 5.0:
        reason = "resolution_too_coarse"
    elif frac_flat > 0.5 and (np.isnan(snr) or snr < 3.0 or frac_weak > 0.5):
        reason = "sensor_insensitive"
    elif not np.isnan(snr) and snr < 1.5:
        reason = "low_snr"
    elif not np.isnan(snr) and frac_weak > 0.6:
        reason = "sensor_insensitive"
    elif big_jumps >= 2:
        reason = "charge_instability"

    if reason is not None:
        status = schema.UNINTERPRETABLE
    elif ref and g0 >= need and g1 >= need:
        status, reason = schema.FOUND, "none"
    else:
        status = schema.NOT_IN_WINDOW
        if not load.any():
            if any_true and frac_slow >= frac_open and frac_slow > 0.5:
                reason = "tunnel_rate_too_low"
            elif any_true and frac_open > 0.5:
                reason = "reservoir_too_open"
            elif any_true and frac_slow > 0.3:
                reason = "tunnel_rate_too_low"
            else:
                reason = "no_transitions"
        elif frac_slow > 0.6:
            reason = "tunnel_rate_too_low"      # fix the tunnel gate first: few rows show lines
        elif frac_open > 0.6:
            reason = "reservoir_too_open"
        elif not ref:
            reason = "no_reference"
        elif g0 >= need and v1 < need:
            reason = "occupancy_too_low"        # the tunnel gate is fine; the 1->2 line is not in view
        elif (g0 < need or g1 < need) and max(v0, g0) >= need and frac_slow < 0.2 and frac_open < 0.2:
            reason = "occupancy_too_low"
        elif frac_slow >= frac_open:
            reason = "tunnel_rate_too_low"
        else:
            reason = "reservoir_too_open"

    # truth for guidance checks: clean tunnel band and the one-electron operating point
    T_op = q.T_open + 0.35 * (q.T_broad - q.T_open)
    P0 = line_position(p, q, 0, T_op, w.v_plungers)
    P1 = line_position(p, q, 1, T_op, w.v_plungers)
    return dict(
        status=status, reason=reason, ref=ref, masks=masks, regime=reg,
        snr=None if np.isnan(snr) else snr, rows_good=[g0, g1],
        T_open=q.T_open, T_broad=q.T_broad,
        operating_point=[0.5 * (P0 + P1), T_op] if np.isfinite(P0 + P1) else None,
        spacing=dva, pixels_per_addition=float(ppa),
    )


# ---------------------------------------------------------------------------------------------
# one-call sample and training arrays
# ---------------------------------------------------------------------------------------------
PVT_UNINTERP = ("low_snr", "sensor_insensitive", "charge_instability", "resolution_too_coarse")


def generate_pvt_sample(rng: np.random.Generator, preset: str = "mixed",
                        mix=(0.40, 0.35, 0.25)) -> dict:
    intent = rng.choice(schema.STATUSES, p=list(mix))
    art_kind, coarse = "normal", False
    if intent == schema.UNINTERPRETABLE:
        art_kind = str(rng.choice(PVT_UNINTERP))
        coarse = art_kind == "resolution_too_coarse"
        w_intent = schema.FOUND if rng.random() < 0.6 else str(rng.choice(PVT_NIW))
    elif intent == schema.NOT_IN_WINDOW:
        w_intent = str(rng.choice(PVT_NIW))
    else:
        w_intent = schema.FOUND
    p = sample_device(rng, preset)
    dot = int(rng.choice([0, 2]))
    q = sample_pvt_params(rng, p, dot)
    w = sample_pvt_window(rng, p, q, w_intent, coarse=coarse)
    art = sample_artifacts(rng, art_kind)
    rend = render_pvt(p, q, w, art, rng)
    truth = pvt_oracle(p, q, w, art, rend)
    return dict(params=p, pvt=q, window=w, artifacts=art, render=rend, truth=truth,
                intent=str(intent))


def pvt_to_arrays(s: dict, size: int) -> tuple[dict, dict]:
    spec = kinds.PVT
    r, t, w = s["render"], s["truth"], s["window"]
    d = s["pvt"].dot
    sig = resample(r["signal"], size, 1)
    occ_all = resample(r["occ"], size, 0)
    reg_r = resample(t["regime"], size, 0)
    od = occ_all[..., d]
    nonslow = reg_r != 0
    load = _edges(od) & nonslow
    spect = np.zeros_like(load)
    for j in range(3):
        if j != d:
            spect |= _edges(occ_all[..., j]) & nonslow
    masks = {"load": load, "spectator": spect & ~load,
             "sensor": resample(dilate(t["masks"]["sensor"], 1).astype(np.uint8), size, 0) > 0}
    occ_r = np.where(nonslow, np.minimum(od, 4), schema.OCC_IGNORE)   # unobservable if latched
    # the regime is supervised on rows that contain a transition of the dot
    rows = _edges(od).any(1)
    reg_lab = np.where(rows[:, None], reg_r, schema.OCC_IGNORE)
    arrays = dict(
        signal=sig.astype(np.float16),
        occ=np.stack([occ_r, reg_lab]).astype(np.int8),
        lines=pack_lines(masks, spec.line_families),
        status=schema.STATUSES.index(t["status"]),
        reason=spec.reasons.index(t["reason"]),
        ref=np.array([t["ref"]], np.int8),
    )
    meta = dict(
        kind="PvT", status=t["status"], reason=t["reason"], ref=t["ref"], snr=t["snr"],
        dot=d, T_open_mV=t["T_open"], T_broad_mV=t["T_broad"],
        operating_point_mV=t["operating_point"], spacing_mV=t["spacing"],
        window_mV=[w.x0, w.x1, w.y0, w.y1], native_shape=[w.ny, w.nx], fast_axis=w.fast_axis,
        artifact=s["artifacts"].kind, rows_good=t["rows_good"],
    )
    return arrays, meta
