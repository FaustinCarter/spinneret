"""Import charge-stability scans saved by spinQICK (``SpinqickData.save_data``, netCDF4).

Layout written by spinQICK (checked against its source):

    root attributes : timestamp (unix s), experiment_name ('_gvg_dc', '_gvg_baseband', ...),
                      cfg (JSON), cfg_type, spinqick_version, voltage_state (JSON gate -> V)
    swept_variables/x, swept_variables/y : one variable per swept gate (V), attrs ax_dim, loop_no
    raw_data_0      : dims (reps, triggers, [avgs], x, y, IQ)   raw ADC
    analyzed_data/analyzed_0 : same without IQ (conductance / transconductance)

The data are indexed [..., x, y]; ChargeCell stores (y, x). gvg_dc sweeps y fast (inner loop).
Axes with several gates (e.g. a virtual sweep of P1 and P2 together) are imported using the
first gate's voltages; the others are recorded in ``extra``.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

from ..schema import Scan


def _kind(x_gate: str, y_gate: str, experiment: str) -> str:
    if x_gate.startswith("P") and y_gate.startswith("P"):
        return "PvP"
    if {x_gate[0], y_gate[0]} == {"P", "T"}:
        return "PvT"
    return experiment.strip("_") or "2D"


def load_spinqick_nc(path: str | Path, device: str = "default", cooldown: str = "",
                     analyzed_index: int = 0) -> Scan:
    import netCDF4  # optional dependency

    path = Path(path)
    with netCDF4.Dataset(path, "r") as nc:
        exp = str(getattr(nc, "experiment_name", ""))
        if "swept_variables" not in nc.groups:
            raise ValueError("This netCDF file has no 'swept_variables' group; it does not look "
                             "like a spinQICK 2D scan.")
        axes = {}
        for name, grp in nc["swept_variables"].groups.items():
            gates = list(grp.variables.keys())
            axes[name] = dict(gates=gates, data={g: np.asarray(grp[g][:], float) for g in gates},
                              loop_no=int(getattr(grp, "loop_no", 0)))
        if "x" not in axes or "y" not in axes:
            raise ValueError(f"Expected swept axes 'x' and 'y', found {sorted(axes)}. Only 2D "
                             "gate-vs-gate scans can be imported.")
        var = None
        iq = False
        if "analyzed_data" in nc.groups:
            key = f"analyzed_{analyzed_index}"
            if key in nc["analyzed_data"].variables:
                var = nc["analyzed_data"][key]
        if var is None:
            var = nc[f"raw_data_{analyzed_index}"]
            iq = True
        arr = np.asarray(var[:], float)
        dims = [d[:-4] if d.endswith("_dim") else d for d in var.dimensions]
        units = str(getattr(var, "units", "a.u."))
        if iq:
            k = dims.index("IQ")
            arr = np.moveaxis(arr, k, -1)
            arr = np.hypot(arr[..., 0], arr[..., 1])
            dims = [d for d in dims if d != "IQ"]
            units = "adc_raw |IQ|"
        reduce = tuple(i for i, d in enumerate(dims) if d not in ("x", "y"))
        if reduce:
            arr = arr.mean(axis=reduce)
            dims = [d for d in dims if d in ("x", "y")]
        if dims != ["x", "y"]:
            arr = arr.T if dims == ["y", "x"] else arr
        signal = arr.T  # (y, x)
        vstate = {}
        if hasattr(nc, "voltage_state"):
            try:
                vstate = {k: float(v) for k, v in json.loads(nc.voltage_state).items()
                          if isinstance(v, (int, float))}
            except (ValueError, TypeError):
                vstate = {}
        ts = getattr(nc, "timestamp", None)
        created = (dt.datetime.fromtimestamp(int(ts)).astimezone().isoformat(timespec="seconds")
                   if ts is not None else None)
        version = str(getattr(nc, "spinqick_version", ""))

    xg, yg = axes["x"]["gates"][0], axes["y"]["gates"][0]
    fast = "y" if axes["y"]["loop_no"] > axes["x"]["loop_no"] else "x"
    extra = dict(spinqick_version=version, experiment_name=exp, file=path.name)
    for ax in ("x", "y"):
        if len(axes[ax]["gates"]) > 1:
            extra[f"{ax}_all_gates"] = {g: [float(v[0]), float(v[-1])]
                                        for g, v in axes[ax]["data"].items()}
    kw = dict(signal=signal, x=axes["x"]["data"][xg], y=axes["y"]["data"][yg], x_gate=xg,
              y_gate=yg, device=device, cooldown=cooldown, kind=_kind(xg, yg, exp),
              voltage_state=vstate, fast_axis=fast, source="spinqick", units=units,
              notes=f"imported from {path.name}", extra=extra)
    if created:
        kw["created"] = created
    return Scan(**kw)


def write_mock_spinqick_nc(path: str | Path, signal_yx: np.ndarray, x: np.ndarray, y: np.ndarray,
                           x_gate: str = "P1", y_gate: str = "P2",
                           voltage_state: dict | None = None, raw_only: bool = False) -> None:
    """Write a file with spinQICK's layout (used by the tests; handy for trying the importer)."""
    import netCDF4

    nx, ny = len(x), len(y)
    with netCDF4.Dataset(path, "w", format="NETCDF4") as nc:
        nc.timestamp = 1790000000
        nc.cfg_type = "GvgDcConfig"
        nc.experiment_name = "_gvg_dc"
        nc.spinqick_version = "mock"
        sw = nc.createGroup("swept_variables")
        for name, gate, vals, loop in (("x", x_gate, x, 0), ("y", y_gate, y, 1)):
            g = sw.createGroup(name)
            g.ax_dim = len(vals)
            g.loop_no = loop
            nc.createDimension(f"{name}_dim", len(vals))
            v = g.createVariable(gate, "f4", (f"{name}_dim",))
            v[:] = vals
            v.units = "V"
        nc.createDimension("reps_dim", 1)
        nc.createDimension("triggers_dim", 1)
        nc.createDimension("IQ_dim", 2)
        data_xy = signal_yx.T
        raw = nc.createVariable("raw_data_0", "f4",
                                ("reps_dim", "triggers_dim", "x_dim", "y_dim", "IQ_dim"))
        raw[:] = np.stack([data_xy, np.zeros_like(data_xy)], -1)[None, None]
        if not raw_only:
            ana = nc.createGroup("analyzed_data")
            a = ana.createVariable("analyzed_0", "f4", ("reps_dim", "triggers_dim", "x_dim",
                                                        "y_dim"))
            a[:] = data_xy[None, None]
            a.units = "conductance"
        nc.cfg = json.dumps({"mock": True})
        nc.voltage_state = json.dumps(voltage_state or {})
