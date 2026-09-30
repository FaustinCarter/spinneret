# The chargecell/1 exchange protocol

ChargeCell does not talk to instruments. Any measurement backend (QCoDeS, QICK, Labber, a
home-grown DAQ script, ...) sends it a scan as JSON and gets a JSON answer back. The same format
submits scans, with or without an expert label, as training data.

- HTTP: run `chargecell serve` and `POST` to the endpoints below.
- Files: `chargecell analyze request.json --json` (or `--out DIR` to write
  `<name>.response.json`). Request files can also be imported on the GUI's Scans page.
- Python: `chargecell.client` (standard library plus numpy; copy it into your environment).
- Schema: `chargecell schema` or `GET /api/v1/schema` gives the JSON Schema of both messages.
  The models are defined in `chargecell/protocol.py`, which is the source of truth.

ChargeCell is **advisory**. It never moves a gate. The backend (or a person) decides whether to
act on a response, and should ramp DC voltages in steps no larger than `max_step`.

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/api/v1/analyze` | request | response |
| POST | `/api/v1/scans` | request, optional `label` | `{protocol, request_id, scan_id, kind, labelled}` |
| GET | `/api/v1/scans/{scan_id}/response` | – | the saved result of an analysed scan, as a response |
| GET | `/api/v1/schema` | – | `{protocol, request, response}` JSON Schemas |

Errors: `422` means the request is invalid or unsupported. `detail` is either a list of field
errors or a sentence. `409` means no trained model is available.

## Request

```json
{
  "protocol": "chargecell/1",
  "request_id": "run42-step3",
  "scan": {
    "kind": "PvP",
    "x": {"gate": "P1", "start": 0.800, "stop": 0.890, "points": 90},
    "y": {"gate": "P2", "values": [0.810, 0.811, "..."]},
    "signal": [[0.12, 0.13, "..."], "..."],
    "voltage_unit": "V",
    "voltage_state": {"P3": 0.845, "X1": 0.310, "X2": 0.305, "M1": 0.920},
    "fast_axis": "y",
    "units": "nA",
    "device": "devA",
    "cooldown": "CD7",
    "id": "devA-20260930-101500",
    "created": "2026-09-30T10:15:00+00:00",
    "notes": "",
    "metadata": {"run": 42}
  },
  "options": {"save": true, "model_id": null}
}
```

| Field | Meaning |
|---|---|
| `scan.kind` | `PvP` (plunger vs plunger) is the only kind analysed today. Other kinds, such as `PvT` or `tiebar`, can be submitted as training data for future models. |
| `scan.x`, `scan.y` | The swept gates. `x` is horizontal (dot A), `y` vertical (dot B). Give either `values` or `start`/`stop`/`points`. Either sweep direction is fine. |
| `scan.signal` | `signal[iy][ix]`: `ny` rows of `nx` values. Use `null` for missing points. Or send it binary: `{"dtype": "float32", "shape": [ny, nx], "data": "<base64 of little-endian bytes, C order>"}` (dtypes: float32, float64, int16, int32, uint16, uint32). |
| `scan.voltage_unit` | `V` or `mV`. Applies to `x`, `y`, `voltage_state` and label coordinates. Responses are always in volts. |
| `scan.voltage_state` | DC voltages of the other gates during the scan. Spectator plungers matter most: they let ChargeCell check spectator occupancy against earlier scans. |
| `scan.device`, `scan.cooldown` | Device settings (safe limits, step size, gate names) are looked up by `device`. Set them on the GUI's Device page or with `PUT /api/devices/{name}`. `cooldown` keeps train/test splits honest. |
| `scan.id` | Optional; letters, digits, `.`, `_`, `-`. Sending the same id again overwrites that scan. |
| `options.save` | Store the scan and result in the workspace (default `true`). Stored results appear in the GUI, can be labelled, and let later guidance on the same device use them as history. |
| `options.model_id` | Use a specific model version instead of the active one. |
| `label` | Only for `POST /api/v1/scans`. See below. |

Unknown fields are rejected, so typos surface as errors instead of being silently ignored.

## Response

```json
{
  "protocol": "chargecell/1",
  "request_id": "run42-step3",
  "scan_id": "devA-20260930-101500",
  "outcome": "next_scan",
  "status": "NOT_IN_WINDOW",
  "reason": "occupancy_too_low",
  "reason_text": "The empty region is visible, but the window stops before one electron is loaded in each dot.",
  "confidence": 0.91,
  "needs_review": false,
  "features": [],
  "next_scan": {
    "purpose": "locate",
    "window": {"x": {"gate": "P1", "start": 0.861, "stop": 0.957, "points": 48},
               "y": {"gate": "P2", "start": 0.842, "stop": 0.938, "points": 48}},
    "move": {"P1": 0.064, "P2": 0.045},
    "gate_changes": {},
    "physical_moves": null,
    "max_step": 0.05,
    "confidence": "high",
    "basis": "lattice fitted to anchored transitions"
  },
  "suggestion": null,
  "headline": "(1,1) is outside this window. To centre it, raise P1 by 64.0 mV and raise P2 by 45.0 mV.",
  "steps": ["Next scan: P1 0.8610 to 0.9570 V (48 points), P2 0.8420 to 0.9380 V (48 points)."],
  "warnings": [],
  "model_id": "model-20260930-...",
  "created": "2026-09-30T10:15:02+00:00"
}
```

### `outcome`: what the backend should do

| outcome | status | Meaning | Where to look |
|---|---|---|---|
| `found` | `FOUND` | (1,1) is in the window, and the empty region is visible for both dots, so the electron count is certain. | `features`. `next_scan` may hold an optional `readout_zoom` window on the (1,1)-(2,0) boundary. |
| `next_scan` | `NOT_IN_WINDOW` | ChargeCell is confident where to look next (guidance confidence high or medium). | `next_scan` |
| `no_confident_step` | `UNINTERPRETABLE`, or `NOT_IN_WINDOW` with nothing to navigate by | ChargeCell could not turn this scan into a next step with confidence. `reason` says why. | `suggestion` (best guess, for a person to review), `headline`, `steps` |

For `UNINTERPRETABLE` scans, `suggestion.purpose` is `rescan_after_fix`: fix the named problem
(retune the sensor, lower a barrier, average longer, ...), then rescan the same window. For
low-confidence exploration, `suggestion.purpose` is `locate`.

`needs_review: true` means the model was unsure or one of its checks disagreed. It can accompany
any outcome. A cautious automated loop should hand these scans to a person.

### `features` (outcome `found`)

| type | label | Fields |
|---|---|---|
| `charge_cell` | `(1,1)` | `point` (centre, gate → V), `polygon` (outline as (x, y) in V), `extent` (gate → [min, max] V) |
| `transition_point` | `(1,1)-(2,0)`, `(1,1)-(0,2)` | `point`: midpoint of the boundary, useful for setting up readout |

Future scan kinds (PvT, tiebar) will add feature types. Clients should ignore types they do not know.

### `next_scan` / `suggestion` (a `ScanStep`)

| Field | Meaning |
|---|---|
| `purpose` | `locate` (find (1,1)), `rescan_after_fix`, or `readout_zoom` |
| `window` | Absolute start/stop in volts and the number of points for both swept gates. Always inside the device's safe limits. |
| `move` | Change of the window centre for each swept gate (V). If the full move is larger than `max_step`, ChargeCell gives only the first step and says so in `warnings`. The move can exceed `max_step` only when the window had to be shifted inside the safe limits, which `warnings` also reports. |
| `gate_changes` | Changes to other gates to make before scanning (V), e.g. `{"X1": -0.010}` to separate merged dots. |
| `physical_moves` | The move in physical gates, when the scan was in virtual gates and the device has a virtual-gate matrix. |
| `max_step` | Largest DC step the device allows (V). Ramp in steps no larger than this. |
| `confidence`, `basis` | How the step was computed (`high`: from anchored transitions in this scan). |

## Submitting training data

`POST /api/v1/scans` takes the same request. With no `label`, the scan is stored and can be
labelled later in the GUI (the label queue puts the most informative scans first). With a label:

```json
"label": {
  "status": "NOT_IN_WINDOW",
  "reason": "no_reference",
  "a_boundaries": [[[0.830, 0.810], [0.834, 0.850]]],
  "b_boundaries": [],
  "a_offset": 0,
  "b_offset": null,
  "spectator_lines": [],
  "sensor_lines": [],
  "annotator": "jdoe",
  "reviewed": true,
  "notes": ""
}
```

Label fields mean the same as in the GUI's labeller (`docs/CODEMAP.md` §2, "Annotation"):

- Each `a_boundaries` entry is a polyline of (x, y) points, drawn bottom to top, separating k
  from k+1 electrons in dot A.
- `b_boundaries` are the same for dot B, drawn left to right.
- `a_offset` is the number of electrons in dot A left of every boundary: `0` means that region
  is empty, and `null` means unknown. `b_offset` is the same for dot B, below every boundary.
- `reason` must fit `status`: `none` for FOUND; one of `no_transitions`, `occupancy_too_low`,
  `no_reference`, `partially_visible` for NOT_IN_WINDOW; one of `low_snr`, `sensor_insensitive`,
  `dots_merged`, `charge_instability`, `resolution_too_coarse` for UNINTERPRETABLE.

A status alone is accepted, but the boundaries are what teach the network where the cells are.
Only `PvP` scans are used when training the current model.

## Python client

```python
from chargecell.client import ChargeCellClient, make_request

cc = ChargeCellClient("http://127.0.0.1:8765")
r = cc.analyze(signal, x_volts, y_volts, x_gate="P1", y_gate="P2", device="devA",
               cooldown="CD7", voltage_state={"P3": 0.845, "M1": 0.920})
if r["outcome"] == "found":
    cell = r["features"][0]["point"]                 # {"P1": ..., "P2": ...}
elif r["outcome"] == "next_scan":
    step = r["next_scan"]                            # ramp by <= step["max_step"], then scan
else:
    print(r["reason_text"], r["headline"])           # a person should look

cc.submit(signal, x_volts, y_volts, "P1", "P2", device="devA", cooldown="CD7")   # training data
make_request(...)                                    # same JSON, for file exchange
```

## Versioning

The `protocol` field is `chargecell/1`. Adding optional request fields, response fields, feature
types or enum values (for example new `kind`s or `purpose`s) keeps version 1. Clients should
ignore response fields they do not know. Renaming or removing fields, or changing their meaning,
requires `chargecell/2`.
