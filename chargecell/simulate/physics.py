"""Physics model for synthetic charge-stability diagrams of Si/SiGe triple-dot EO devices.

Model
-----
Charge states n = (n0, n1, n2) of three dots, n_i in {0..NMAX}. Classical (constant-interaction)
energy, measured from a grounded reservoir (grand canonical, reservoir chemical potential = 0):

    E(n) = sum_i U_i n_i (n_i - 1) / 2 + sum_{i<j} Um_ij n_i n_j
           + sum_i Ev_i max(0, n_i - 2)                # valley/orbital cost for the 3rd+ electron
           + sum_i n_i eps_i(V),     eps_i = offset_i - sum_j lever_ij V_j

Interdot tunnelling t_ij couples states that differ by one electron hopping i <-> j. Its strength
grows with the number of electrons in the pair (tgrow), which reproduces the curved interdot
transitions HRL reports at higher occupancy. For each pixel we keep the K lowest classical
states, diagonalise the small Hamiltonian, and take a thermal average over its eigenstates.

The charge sensor is a dot in Coulomb blockade. Its chemical potential shifts with each electron
in the array (kappa_i, falling off with distance) and with any gate crosstalk left over after
compensation. Current is a sum of thermally broadened Coulomb peaks, so the sensor can walk off
its flank or cross its own peaks, which are the realistic ways a scan becomes uninterpretable.

Units: voltages in mV, dot energies in meV, sensor energies in units of the sensor peak width.
"""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass

import numpy as np

NMAX = 5
K_LOW = 6  # classical states kept per pixel for the tunnelling Hamiltonian

STATES = np.array(list(itertools.product(range(NMAX + 1), repeat=3)), dtype=np.int64)  # (S,3)
_STATE_INDEX = {tuple(s): k for k, s in enumerate(STATES)}


@dataclass
class DeviceParams:
    lever: np.ndarray          # (3,3) meV/mV
    U: np.ndarray              # (3,)  meV
    Um: np.ndarray             # (3,3) meV, symmetric, zero diagonal
    Ev: np.ndarray             # (3,)  meV
    t0: np.ndarray             # (3,3) meV, symmetric, zero diagonal
    tgrow: float
    kT: float                  # meV
    offset: np.ndarray         # (3,)  meV
    v11: np.ndarray            # (3,)  mV: plunger voltages at the centre of (1,1,1)
    # sensor (energies in units of the sensor peak width)
    s_lever: np.ndarray        # (3,)  width/mV, plunger -> sensor crosstalk before compensation
    comp_residual: np.ndarray  # (3,)  fraction of that crosstalk left after compensation
    s_kappa: np.ndarray        # (3,)  width per electron
    s_ES: float                # sensor peak spacing, widths
    s_mu_ref: float            # sensor chemical potential at the tuning reference, widths
    v_ref: np.ndarray          # (3,)  mV, where the sensor was tuned
    n_ref: np.ndarray          # (3,)  occupancy at the tuning reference
    s_amp: float
    s_base: float
    latch: np.ndarray          # (3,)  per-pixel probability of *not yet* tunnelling (0 = fast)
    geometry: str = "linear"
    merged_pair: tuple[int, int] | None = None

    # ------------------------------------------------------------------ helpers
    def addition_voltage(self, i: int) -> float:
        """Width of a charge cell of dot i along its own plunger, in mV."""
        return float(self.U[i] / self.lever[i, i])

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "DeviceParams":
        arr_keys = {"lever", "U", "Um", "Ev", "t0", "offset", "v11", "s_lever", "comp_residual",
                    "s_kappa", "v_ref", "n_ref", "latch"}
        kw = {k: (np.asarray(v, dtype=float) if k in arr_keys else v) for k, v in d.items()}
        if kw.get("merged_pair") is not None:
            kw["merged_pair"] = tuple(kw["merged_pair"])
        return cls(**kw)


def centre_offsets(lever: np.ndarray, U: np.ndarray, Um: np.ndarray, v11: np.ndarray) -> np.ndarray:
    """Offsets that put the centre of the (1,1,1) cell at plunger voltages ``v11``."""
    eps_star = -U / 2.0 - Um.sum(axis=1)  # each neighbour holds one electron
    return eps_star + lever @ v11


# ---------------------------------------------------------------------------------------------
# Energies
# ---------------------------------------------------------------------------------------------
def _static_energy(p: DeviceParams) -> np.ndarray:
    n = STATES.astype(float)
    e = 0.5 * (p.U[None, :] * n * (n - 1)).sum(1)
    for i in range(3):
        for j in range(i + 1, 3):
            e += p.Um[i, j] * n[:, i] * n[:, j]
    e += (p.Ev[None, :] * np.maximum(0.0, n - 2)).sum(1)
    return e


def _tunnel_matrix(p: DeviceParams) -> np.ndarray:
    S = len(STATES)
    T = np.zeros((S, S))
    for s, n in enumerate(STATES):
        for i in range(3):
            for j in range(3):
                if i == j or p.t0[i, j] <= 0 or n[i] == 0 or n[j] == NMAX:
                    continue
                m = n.copy()
                m[i] -= 1
                m[j] += 1
                s2 = _STATE_INDEX[tuple(m)]
                pair = n[i] + n[j]
                T[s, s2] = p.t0[i, j] * (1.0 + p.tgrow * max(0, pair - 2))
    return np.maximum(T, T.T)


_cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}


def _precompute(p: DeviceParams) -> tuple[np.ndarray, np.ndarray]:
    # keyed on the parameter values (not object identity, which Python may reuse)
    key = np.concatenate([p.U, p.Um.ravel(), p.Ev, p.t0.ravel(), [p.tgrow]]).tobytes()
    if key not in _cache:
        if len(_cache) > 64:
            _cache.clear()
        _cache[key] = (_static_energy(p), _tunnel_matrix(p))
    return _cache[key]


def detuning(p: DeviceParams, V: np.ndarray, offset: np.ndarray | None = None) -> np.ndarray:
    """eps (P,3) for plunger voltages V (P,3) in mV."""
    off = p.offset if offset is None else offset
    return off[None, :] - V @ p.lever.T


def classical_ground_state(p: DeviceParams, V: np.ndarray, offset=None) -> np.ndarray:
    """T=0, t=0 ground state occupations (P,3). Cheap; used for geometry/oracle labels."""
    E0, _ = _precompute(p)
    eps = detuning(p, V, offset)
    E = E0[None, :] + eps @ STATES.T.astype(float)
    return STATES[np.argmin(E, axis=1)]


def occupations(p: DeviceParams, V: np.ndarray, offset=None, chunk: int = 8192):
    """Thermal expectation <n_i> (P,3) and ground-state charge configuration (P,3)."""
    E0, T = _precompute(p)
    NS = STATES.astype(float)
    out_mean = np.empty((len(V), 3))
    out_gs = np.empty((len(V), 3), dtype=np.int64)
    for start in range(0, len(V), chunk):
        v = V[start:start + chunk]
        eps = detuning(p, v, offset)
        E = E0[None, :] + eps @ NS.T                              # (P,S)
        idx = np.argpartition(E, K_LOW, axis=1)[:, :K_LOW]        # (P,K)
        Ek = np.take_along_axis(E, idx, axis=1)
        H = T[idx[:, :, None], idx[:, None, :]]                   # (P,K,K)
        H[:, np.arange(K_LOW), np.arange(K_LOW)] = Ek
        H -= Ek.min(1)[:, None, None] * np.eye(K_LOW)[None]       # numerical conditioning
        w, vec = np.linalg.eigh(H)
        prob = vec ** 2                                           # (P,K_state,K_eig)
        boltz = np.exp(-(w - w[:, :1]) / max(p.kT, 1e-6))
        boltz /= boltz.sum(1, keepdims=True)
        nk = NS[idx]                                              # (P,K_state,3)
        per_eig = np.einsum("pse,psi->pei", prob, nk)             # (P,K_eig,3)
        out_mean[start:start + len(v)] = np.einsum("pe,pei->pi", boltz, per_eig)
        gs_state = np.take_along_axis(idx, prob[:, :, 0].argmax(1)[:, None], axis=1)[:, 0]
        out_gs[start:start + len(v)] = STATES[gs_state]
    return out_mean, out_gs


# ---------------------------------------------------------------------------------------------
# Sensor
# ---------------------------------------------------------------------------------------------
def sensor_mu(p: DeviceParams, V: np.ndarray, occ: np.ndarray) -> np.ndarray:
    """Sensor chemical potential (widths) for plunger voltages V (P,3) and occupations (P,3)."""
    residual_slope = p.comp_residual * p.s_lever
    return (p.s_mu_ref - (V - p.v_ref[None, :]) @ residual_slope
            + (occ - p.n_ref[None, :]) @ p.s_kappa)


def sensor_current(p: DeviceParams, mu: np.ndarray) -> np.ndarray:
    total = np.zeros_like(mu)
    for m in range(-4, 5):
        total += 1.0 / np.cosh(np.clip(mu - m * p.s_ES, -40, 40)) ** 2
    return p.s_base + p.s_amp * total


def sensor_slope(p: DeviceParams, mu: np.ndarray, h: float = 1e-3) -> np.ndarray:
    return (sensor_current(p, mu + h) - sensor_current(p, mu - h)) / (2 * h)
