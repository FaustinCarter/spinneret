"""Synthetic scan generation with ground-truth labels.

A synthetic sample is produced in three steps:
  1. sample a device (``sample_device``) - lever arms, charging energies, tunnel couplings,
     sensor coupling, compensation quality...
  2. sample a scan window and measurement artifacts (``sample_window``/``sample_artifacts``),
     aiming at a requested outcome so the dataset stays balanced
  3. render the scan (``render``) and compute the ground truth with the oracle (``oracle``).
     The oracle decides the final label from the rendered physics, not from the intent in
     step 2, so labels are always consistent with the data.

The same ``render`` function powers the Virtual Device in the GUI, where an operator can practise
the scan -> analyse -> move loop without hardware.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .. import schema
from .physics import (DeviceParams, centre_offsets, classical_ground_state, occupations,
                      sensor_current, sensor_mu, sensor_slope)

PLUNGERS = ("P1", "P2", "P3")
ANCHOR_WIDTH = 1.3   # empty region needed to count electrons, in addition voltages


# ---------------------------------------------------------------------------------------------
# Device sampling
# ---------------------------------------------------------------------------------------------
PRESETS = {
    # Loosely matched to published HRL SLEDGE data (addition voltages of a few tens of mV at
    # plunger biases near 0.8 V, strong interdot coupling, one sensor dot near one end).
    "hrl_linear": dict(geometry="linear"),
    "hrl_triangle": dict(geometry="triangle"),
    "mixed": dict(geometry=None),
}


def _loguniform(rng, lo, hi, size=None):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size))


def sample_device(rng: np.random.Generator, preset: str = "mixed",
                  merged_pair: tuple[int, int] | None = None) -> DeviceParams:
    geometry = PRESETS.get(preset, {}).get("geometry") or rng.choice(["linear", "triangle"],
                                                                      p=[0.7, 0.3])
    neighbours = {(0, 1), (1, 2)} if geometry == "linear" else {(0, 1), (1, 2), (0, 2)}
    alpha = rng.uniform(0.06, 0.14, 3)                          # meV/mV
    lever = np.diag(alpha)
    Ubar = rng.uniform(2.5, 5.5)
    U = Ubar * rng.uniform(0.85, 1.15, 3)
    Um = np.zeros((3, 3))
    t0 = np.zeros((3, 3))
    for i in range(3):
        for j in range(i + 1, 3):
            near = (i, j) in neighbours
            r_ij, r_ji = (rng.uniform(0.08, 0.35, 2) if near else rng.uniform(0.02, 0.10, 2))
            lever[i, j] = alpha[i] * r_ij
            lever[j, i] = alpha[j] * r_ji
            Um[i, j] = Um[j, i] = Ubar * (rng.uniform(0.08, 0.30) if near
                                          else rng.uniform(0.02, 0.08))
            if near:
                t0[i, j] = t0[j, i] = _loguniform(rng, 0.003, 0.25)
    if merged_pair is not None:
        i, j = merged_pair
        Um[i, j] = Um[j, i] = Ubar * rng.uniform(0.80, 0.93)
        t0[i, j] = t0[j, i] = rng.uniform(0.3, 1.0)
    Ev = rng.uniform(0.0, 0.6, 3)
    v11 = rng.uniform(650.0, 950.0, 3)
    offset = centre_offsets(lever, U, Um, v11)

    # sensor near dot 0; kappa falls off with distance
    pos = {"linear": np.array([1.0, 1.7, 2.5]), "triangle": np.array([1.0, 1.4, 1.4])}[geometry]
    power = rng.uniform(0.3, 1.5)
    kappa = rng.uniform(0.25, 0.7) * pos ** (-power) * rng.uniform(0.7, 1.3, 3)
    s_lever = rng.uniform(0.05, 0.15) * pos ** (-1.0)
    comp_sigma = rng.uniform(0.01, 0.12)
    comp_residual = rng.normal(0.0, comp_sigma, 3)
    if rng.random() < 0.05:                                    # compensation switched off
        comp_residual = np.ones(3)
    side = rng.choice([-1.0, 1.0])
    latch = np.where(rng.random(3) < 0.2, rng.uniform(0.3, 0.9, 3), 0.0)

    return DeviceParams(
        lever=lever, U=U, Um=Um, Ev=Ev, t0=t0, tgrow=float(rng.uniform(0.0, 1.5)),
        kT=float(0.013 * rng.uniform(1.0, 3.0)), offset=offset, v11=v11,
        s_lever=s_lever, comp_residual=comp_residual, s_kappa=kappa,
        s_ES=float(rng.uniform(4.0, 10.0)), s_mu_ref=float(side * rng.uniform(0.4, 1.0)),
        v_ref=v11.copy(), n_ref=np.ones(3), s_amp=1.0, s_base=float(rng.uniform(0.0, 0.2)),
        latch=latch, geometry=str(geometry),
        merged_pair=tuple(merged_pair) if merged_pair is not None else None,
    )


# ---------------------------------------------------------------------------------------------
# Window + artifacts
# ---------------------------------------------------------------------------------------------
@dataclass
class Window:
    pair: tuple[int, int]            # (dot a on x, dot b on y)
    x0: float
    x1: float
    y0: float
    y1: float
    nx: int
    ny: int
    v_spectator: float               # mV on the spectator plunger
    fast_axis: str = "y"

    @property
    def spectator(self) -> int:
        return ({0, 1, 2} - set(self.pair)).pop()


@dataclass
class Artifacts:
    snr_target: float = 10.0          # contrast of a typical step / white-noise sigma
    pink_sigma: float = 0.0           # widths, sensor 1/f noise
    telegraph_amp: float = 0.0        # widths
    telegraph_rate: float = 0.0       # switches per pixel
    jumps: list = field(default_factory=list)   # [(slow_index_fraction, [d_off0,d_off1,d_off2])]
    gain_drift: float = 0.0
    sensor_detune: float = 0.0        # extra widths added to the sensor operating point
    kind: str = "normal"              # which failure mode was injected, if any


def sample_window(rng, p: DeviceParams, intent: str, pair=None, coarse: bool = False) -> Window:
    if pair is None:
        pair = tuple(int(v) for v in rng.permutation(3)[:2])
    a, b = pair
    c = ({0, 1, 2} - {a, b}).pop()
    dva, dvb, dvc = p.addition_voltage(a), p.addition_voltage(b), p.addition_voltage(c)
    ca, cb = p.v11[a], p.v11[b]
    if intent == schema.FOUND:
        # keep enough of the empty region inside to count from (ANCHOR_WIDTH and a margin),
        # then the (1,1) cell and part of its neighbours
        ea, eb = rng.uniform(1.4, 2.2), rng.uniform(1.4, 2.2)
        wx, wy = (ea + rng.uniform(1.25, 2.6)) * dva, (eb + rng.uniform(1.25, 2.6)) * dvb
        if rng.random() < 0.5:
            wy = max(wx * dvb / dva, (eb + 1.25) * dvb)
        cx = ca - 0.5 * dva - ea * dva + wx / 2 + rng.normal(0, 0.08) * dva
        cy = cb - 0.5 * dvb - eb * dvb + wy / 2 + rng.normal(0, 0.08) * dvb
    else:
        wx, wy = rng.uniform(0.8, 4.5) * dva, rng.uniform(0.8, 4.5) * dvb
        cx = ca + rng.uniform(-4.0, 5.0) * dva
        cy = cb + rng.uniform(-4.0, 5.0) * dvb
    # spectator mostly holds one electron; sometimes 0 or 2 (the "unverified spectator" case)
    r = rng.random()
    spec_shift = rng.normal(0, 0.2) if r < 0.8 else rng.choice([-1.0, 1.0]) * rng.uniform(0.7, 1.2)
    v_spec = p.v11[c] + spec_shift * dvc
    if coarse:
        ppa = rng.uniform(1.5, 4.0)
    else:
        ppa = rng.uniform(7.0, 35.0)                       # pixels per addition voltage
    nx = int(np.clip(round(wx / dva * ppa), 16 if coarse else 32, 220))
    ny = int(np.clip(round(wy / dvb * ppa), 16 if coarse else 32, 220))
    return Window(pair=(a, b), x0=cx - wx / 2, x1=cx + wx / 2, y0=cy - wy / 2, y1=cy + wy / 2,
                  nx=nx, ny=ny, v_spectator=float(v_spec),
                  fast_axis=str(rng.choice(["x", "y"])))


def sample_artifacts(rng, kind: str = "normal") -> Artifacts:
    art = Artifacts(
        snr_target=float(_loguniform(rng, 2.0, 40.0)),
        pink_sigma=float(rng.uniform(0.0, 0.12)),
        gain_drift=float(rng.uniform(0.0, 0.08)),
        kind=kind,
    )
    if rng.random() < 0.3:
        art.telegraph_amp = float(rng.uniform(0.1, 0.5))
        art.telegraph_rate = float(_loguniform(rng, 1e-4, 2e-2))
    if rng.random() < 0.12:
        art.jumps = [(float(rng.uniform(0.1, 0.9)), rng.normal(0, 0.08, 3).tolist())]
    if kind == "low_snr":
        art.snr_target = float(rng.uniform(0.15, 0.8))
    elif kind == "sensor_insensitive":
        art.sensor_detune = float(rng.choice([-1, 1]) * rng.uniform(2.2, 3.5))
    elif kind == "charge_instability":
        art.jumps = [(float(f), rng.normal(0, rng.uniform(0.3, 0.9), 3).tolist())
                     for f in np.sort(rng.uniform(0.05, 0.95, rng.integers(3, 8)))]
    return art


# ---------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------
def _pink(rng, n: int) -> np.ndarray:
    f = np.fft.rfftfreq(n)
    spec = (rng.normal(size=len(f)) + 1j * rng.normal(size=len(f)))
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0
    x = np.fft.irfft(spec, n)
    return x / (x.std() + 1e-12)


def _telegraph(rng, n: int, rate: float) -> np.ndarray:
    if rate <= 0:
        return np.zeros(n)
    flips = rng.random(n) < rate
    return (np.cumsum(flips) % 2).astype(float)


def _raster_index(w: Window) -> np.ndarray:
    """time index of each pixel (ny,nx) given the fast axis."""
    iy, ix = np.meshgrid(np.arange(w.ny), np.arange(w.nx), indexing="ij")
    return ix * w.ny + iy if w.fast_axis == "y" else iy * w.nx + ix


def _slow_index(w: Window) -> np.ndarray:
    iy, ix = np.meshgrid(np.arange(w.ny), np.arange(w.nx), indexing="ij")
    return ix if w.fast_axis == "y" else iy


def voltage_grid(p: DeviceParams, w: Window) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(w.x0, w.x1, w.nx)
    ys = np.linspace(w.y0, w.y1, w.ny)
    X, Y = np.meshgrid(xs, ys)
    V = np.zeros((w.ny * w.nx, 3))
    V[:, w.pair[0]] = X.ravel()
    V[:, w.pair[1]] = Y.ravel()
    V[:, w.spectator] = w.v_spectator
    return xs, ys, V


def _apply_latching(gs: np.ndarray, mean: np.ndarray, latch: np.ndarray, w: Window, rng):
    """Delay reservoir transitions along the fast axis for dots with slow tunnel rates."""
    if not np.any(latch > 0):
        return mean
    m = mean.reshape(w.ny, w.nx, 3).copy()
    g = gs.reshape(w.ny, w.nx, 3)
    lines = range(w.nx) if w.fast_axis == "y" else range(w.ny)
    for k in lines:
        seq_m = m[:, k, :] if w.fast_axis == "y" else m[k, :, :]
        seq_g = g[:, k, :] if w.fast_axis == "y" else g[k, :, :]
        for d in np.nonzero(latch > 0)[0]:
            change = np.nonzero(np.diff(seq_g[:, d]) != 0)[0]
            for j in change:
                total_change = seq_g[j + 1].sum() != seq_g[j].sum()
                if not total_change:
                    continue
                delay = rng.geometric(1.0 - latch[d]) - 1
                end = min(j + 1 + delay, len(seq_m))
                seq_m[j + 1:end, d] = seq_m[j, d]
    return m.reshape(-1, 3)


def render(p: DeviceParams, w: Window, art: Artifacts, rng: np.random.Generator) -> dict:
    """Render a scan. Returns signal (ny,nx), axes, ground-truth occupancy and sensor info."""
    xs, ys, V = voltage_grid(p, w)
    slow = _slow_index(w).ravel()
    n_slow = w.nx if w.fast_axis == "y" else w.ny

    # charge jumps: offsets change between fast sweeps
    offsets = np.tile(p.offset, (len(V), 1))
    for frac, d_off in art.jumps:
        k = int(frac * n_slow)
        offsets[slow >= k] += np.asarray(d_off) * p.U
    mean = np.empty((len(V), 3))
    gs = np.empty((len(V), 3), dtype=np.int64)
    for off in np.unique(offsets, axis=0):
        sel = np.all(offsets == off, axis=1)
        mean[sel], gs[sel] = occupations(p, V[sel], offset=off)

    mean_obs = _apply_latching(gs, mean, p.latch, w, rng)
    meas = measure_sensor(p, V, mean_obs, w, art, rng)
    return dict(x=xs, y=ys, occ=gs.reshape(w.ny, w.nx, 3), **meas)


def measure_sensor(p: DeviceParams, V: np.ndarray, occ_obs: np.ndarray, w, art: Artifacts,
                   rng: np.random.Generator, mu_extra: np.ndarray | float = 0.0) -> dict:
    """Charge-sensor current for pixels V (P,3) with observed occupations (P,3): sensor
    crosstalk, 1/f and telegraph noise on its chemical potential, gain drift, white noise.
    ``w`` needs nx, ny and fast_axis; ``mu_extra`` adds crosstalk from other swept gates."""
    slow = _slow_index(w).ravel()
    n_slow = w.nx if w.fast_axis == "y" else w.ny
    mu_clean = sensor_mu(p, V, occ_obs) + art.sensor_detune + mu_extra
    t_idx = _raster_index(w).ravel()
    order = np.argsort(t_idx)
    noise_mu = np.zeros(len(V))
    if art.pink_sigma > 0:
        noise_mu[order] += art.pink_sigma * _pink(rng, len(V))
    if art.telegraph_amp > 0:
        noise_mu[order] += art.telegraph_amp * _telegraph(rng, len(V), art.telegraph_rate)
    clean = sensor_current(p, mu_clean)
    sig = sensor_current(p, mu_clean + noise_mu)
    if art.gain_drift > 0:
        drift = 1.0 + art.gain_drift * _pink(rng, n_slow) / 3.0
        sig *= drift[slow]
    # white noise sized relative to a typical charge step at the operating point
    typical_step = np.abs(sensor_slope(p, np.array([p.s_mu_ref])))[0] * float(np.mean(p.s_kappa))
    sigma_w = max(typical_step, 0.02 * p.s_amp) / art.snr_target
    sig = sig + rng.normal(0, sigma_w, len(V))
    return dict(
        signal=sig.reshape(w.ny, w.nx).astype(np.float32),
        clean=clean.reshape(w.ny, w.nx),
        mu=mu_clean.reshape(w.ny, w.nx),
        sigma_white=float(sigma_w),
        sigma_mu=float(np.sqrt(art.pink_sigma ** 2 + (art.telegraph_amp / 2) ** 2)),
    )


# ---------------------------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------------------------
def boundary_masks(occ_a: np.ndarray, occ_b: np.ndarray, occ_c: np.ndarray | None = None):
    """Pixel masks (ny,nx) for each line family from occupancy maps (both sides of a boundary)."""
    ny, nx = occ_a.shape
    masks = {f: np.zeros((ny, nx), bool) for f in schema.LINE_FAMILIES}

    def mark(sl0, sl1, cond, fam):
        masks[fam][sl0] |= cond
        masks[fam][sl1] |= cond

    for axis in (0, 1):
        if axis == 1:
            s0, s1 = (slice(None), slice(None, -1)), (slice(None), slice(1, None))
        else:
            s0, s1 = (slice(None, -1), slice(None)), (slice(1, None), slice(None))
        da = occ_a[s1] != occ_a[s0]
        db = occ_b[s1] != occ_b[s0]
        tot = (occ_a[s1] + occ_b[s1]) != (occ_a[s0] + occ_b[s0])
        mark(s0, s1, da & ~db, "a")
        mark(s0, s1, db & ~da, "b")
        mark(s0, s1, da & db & ~tot, "interdot")
        mark(s0, s1, da & db & tot, "a")      # rare diagonal double step: count for both
        mark(s0, s1, da & db & tot, "b")
        if occ_c is not None:
            mark(s0, s1, occ_c[s1] != occ_c[s0], "spectator")
    return masks


def sensor_peak_mask(mu: np.ndarray, ES: float, charge_edges: np.ndarray) -> np.ndarray:
    """Pixels where the sensor crosses one of its own Coulomb peaks (not caused by a charge
    transition). These show up as ridges that are easy to mistake for dot transitions."""
    k = np.round(mu / ES)
    r = mu - k * ES
    mask = np.zeros(mu.shape, bool)
    for axis in (0, 1):
        s0 = [slice(None)] * 2
        s1 = [slice(None)] * 2
        s0[axis] = slice(None, -1)
        s1[axis] = slice(1, None)
        s0, s1 = tuple(s0), tuple(s1)
        cross = (np.sign(r[s0]) != np.sign(r[s1])) & (k[s0] == k[s1])
        mask[s0] |= cross
        mask[s1] |= cross
    return mask & ~charge_edges


def oracle(p: DeviceParams, w: Window, art: Artifacts, rend: dict) -> dict:
    """Decide the true status, reason, anchoring and the correct next move for a rendered scan."""
    a, b = w.pair
    c = w.spectator
    occ = rend["occ"]
    oa, ob, oc = occ[..., a], occ[..., b], occ[..., c]
    ny, nx = oa.shape
    dva, dvb = p.addition_voltage(a), p.addition_voltage(b)
    px = (w.x1 - w.x0) / max(nx - 1, 1)
    py = (w.y1 - w.y0) / max(ny - 1, 1)
    masks = boundary_masks(oa, ob, oc)
    charge_edges = masks["a"] | masks["b"] | masks["interdot"] | masks["spectator"]
    masks["sensor"] = sensor_peak_mask(rend["mu"], p.s_ES, charge_edges)

    # --- anchoring ("reference") per dot: an empty region wide enough to trust, plus a transition
    def anchored(o, pitch, dv, axis):
        has0 = (o == 0).mean() > 0.03
        has1 = (o >= 1).mean() > 0.03
        width0 = (o == 0).sum(axis=axis) * pitch        # per row (axis=1) or column (axis=0)
        # only a line-free stretch wider than any occupied cell proves the dot is empty: an
        # occupied cell is at most ~1.25 addition voltages wide (the third electron also pays the
        # valley/orbital energy), and a narrower strip at the edge could be one
        wide = (width0 >= ANCHOR_WIDTH * dv).mean() > 0.2
        return bool(has0 and has1 and wide)

    ref_a = anchored(oa, px, dva, 1)
    ref_b = anchored(ob, py, dvb, 0)

    # --- (1,1) visibility, relative to the full cell (from a large classical simulation)
    full = full_cell(p, w)
    V = voltage_grid(p, w)[2]
    cg = classical_ground_state(p, V).reshape(ny, nx, 3)
    in11 = (cg[..., a] == 1) & (cg[..., b] == 1)
    vis_frac = float(in11.sum() * px * py / full["area"]) if full["area"] > 0 else 0.0

    # --- interpretability: contrast of the charge steps themselves (not the sensor background)
    mu = rend["mu"]
    sigma_eff = np.sqrt(rend["sigma_white"] ** 2
                        + (np.median(np.abs(sensor_slope(p, mu))) * rend["sigma_mu"]) ** 2)
    steps = []
    for fam, kap in (("a", p.s_kappa[a]), ("b", p.s_kappa[b])):
        m = masks[fam]
        if m.sum():
            steps.append(np.abs(sensor_current(p, mu[m]) - sensor_current(p, mu[m] - kap)))
    snr = np.nan
    frac_weak = 0.0
    if steps and sum(len(s_) for s_ in steps) >= 5:
        contrast = np.concatenate(steps)
        c95 = np.percentile(contrast, 95)
        snr = float(np.percentile(contrast, 75) / sigma_eff)
        frac_weak = float((contrast < max(1.0 * sigma_eff, 0.1 * c95)).mean())
    max_slope = 0.77 * p.s_amp
    frac_flat = float((np.abs(sensor_slope(p, mu)) < 0.2 * max_slope).mean())
    big_jumps = sum(1 for _, d in art.jumps if np.max(np.abs(d)) >= 0.15)
    ppa = min(dva / px, dvb / py)

    reason = None
    if p.merged_pair is not None and set(p.merged_pair) == {a, b}:
        reason = "dots_merged"
    elif ppa < 5.0:
        reason = "resolution_too_coarse"
    elif frac_flat > 0.5 and ((not np.isnan(snr) and (snr < 3.0 or frac_weak > 0.5))
                              or np.isnan(snr)):
        reason = "sensor_insensitive"
    elif not np.isnan(snr) and snr < 1.5:
        reason = "low_snr"
    elif not np.isnan(snr) and frac_weak > 0.6:
        reason = "sensor_insensitive"
    elif big_jumps >= 2:
        reason = "charge_instability"

    any_lines = bool((masks["a"] | masks["b"] | masks["interdot"]).sum() > 3)
    if reason is not None:
        status = schema.UNINTERPRETABLE
    elif ref_a and ref_b and vis_frac >= 0.6:
        status, reason = schema.FOUND, "none"
    else:
        status = schema.NOT_IN_WINDOW
        a_has_lines = masks["a"].any() or masks["interdot"].any()
        b_has_lines = masks["b"].any() or masks["interdot"].any()
        a_missing_ref = (oa >= 1).all() and a_has_lines
        b_missing_ref = (ob >= 1).all() and b_has_lines
        if not any_lines:
            reason = "no_transitions"
        elif (not ref_a and (oa >= 1).mean() > 0.97) or (not ref_b and (ob >= 1).mean() > 0.97) \
                or a_missing_ref or b_missing_ref:
            reason = "no_reference"
        elif ref_a and ref_b:
            reason = "partially_visible"
        elif vis_frac > 0.05:
            reason = "partially_visible"
        else:
            reason = "occupancy_too_low"

    return dict(
        status=status, reason=reason, ref_a=ref_a, ref_b=ref_b, vis_frac=vis_frac,
        snr=None if np.isnan(snr) else snr, frac_weak=frac_weak,
        spectator_occ=int(np.bincount(oc.ravel()).argmax()),
        target=full["centroid"], cell_area=full["area"],
        spacing=(dva, dvb), masks=masks,
        pixels_per_addition=float(ppa),
    )


def full_cell(p: DeviceParams, w: Window, n: int = 120) -> dict:
    """Centroid (mV) and area (mV^2) of the complete (1,1) cell for this pair/spectator bias."""
    a, b = w.pair
    dva, dvb = p.addition_voltage(a), p.addition_voltage(b)
    big = Window(pair=w.pair, x0=p.v11[a] - 3.5 * dva, x1=p.v11[a] + 3.5 * dva,
                 y0=p.v11[b] - 3.5 * dvb, y1=p.v11[b] + 3.5 * dvb, nx=n, ny=n,
                 v_spectator=w.v_spectator)
    xs, ys, V = voltage_grid(p, big)
    cg = classical_ground_state(p, V).reshape(n, n, 3)
    m = (cg[..., a] == 1) & (cg[..., b] == 1)
    if not m.any():
        return dict(area=0.0, centroid=None)
    X, Y = np.meshgrid(xs, ys)
    dA = (xs[1] - xs[0]) * (ys[1] - ys[0])
    return dict(area=float(m.sum() * dA), centroid=[float(X[m].mean()), float(Y[m].mean())])


# ---------------------------------------------------------------------------------------------
# One-call sample
# ---------------------------------------------------------------------------------------------
UNINTERP_KINDS = ["low_snr", "sensor_insensitive", "dots_merged", "charge_instability",
                  "resolution_too_coarse"]


def generate_sample(rng: np.random.Generator, preset: str = "mixed",
                    mix=(0.40, 0.35, 0.25)) -> dict:
    """One labelled synthetic scan at native resolution."""
    intent = rng.choice(schema.STATUSES, p=list(mix))
    kind = "normal"
    merged_pair = None
    coarse = False
    if intent == schema.UNINTERPRETABLE:
        kind = str(rng.choice(UNINTERP_KINDS))
        coarse = kind == "resolution_too_coarse"
    pair = tuple(int(v) for v in rng.permutation(3)[:2])
    if kind == "dots_merged":
        merged_pair = pair
    p = sample_device(rng, preset, merged_pair=merged_pair)
    w_intent = intent
    if intent == schema.UNINTERPRETABLE:
        w_intent = schema.FOUND if rng.random() < 0.5 else schema.NOT_IN_WINDOW
    w = sample_window(rng, p, w_intent, pair=pair, coarse=coarse)
    art = sample_artifacts(rng, kind)
    rend = render(p, w, art, rng)
    truth = oracle(p, w, art, rend)
    return dict(params=p, window=w, artifacts=art, render=rend, truth=truth, intent=str(intent))


def window_to_dict(w: Window) -> dict:
    return asdict(w)


def artifacts_to_dict(a: Artifacts) -> dict:
    return asdict(a)
