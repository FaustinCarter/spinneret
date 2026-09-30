# The chargecell/1 exchange protocol

ChargeCell does not talk to instruments. Any measurement backend (QCoDeS, QICK, Labber, a
home-grown DAQ script, ...) sends it a scan as JSON and gets a JSON answer back. The same format
submits scans, with or without an expert label, as training data.

Three kinds of charge-stability scan are analysed, one model each, following HRL's tune-up:

| `scan.kind` | Axes | Goal | Found means |
|---|---|---|---|
| `PvP` | plunger vs plunger (dot A on x, dot B on y) | find the (1,1) cell | (1,1) centre, outline, readout boundaries |
| `PvT` | plunger vs the tunnel gate to its reservoir (either way round) | load one electron with a clean tunnel rate | loading lines, the clean tunnel-gate range, the one-electron operating point |
| `tiebar` | plunger vs plunger zoomed on the (1,1)-(2,0) transition | set up readout | triple points, tie bar with a coupling measurement, first readout point |

**No voltage scale is assumed anywhere.** Devices and technologies differ widely, so every
number in a response is measured from the scan itself or derived from your device settings;
device settings are optional, and missing ones are replaced by fractions of the scan window.

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
| GET | `/api/v1/schema` | – | JSON Schemas of the request, the response and the run messages |
| POST | `/api/v1/runs` | `{device, title, cooldown, run_id?}` | the new run's summary (see [Runs](#runs-the-automation-tree)) |
| GET | `/api/v1/runs?device=&source=` | – | run summaries, most recent first |
| GET | `/api/v1/runs/{run_id}` | – | the run's graded tree (`?format=text` for one line per action) |
| POST | `/api/v1/runs/{run_id}/events` | `{text, gate_changes, voltage_unit, by, note}` | the recorded node |
| POST | `/api/v1/runs/{run_id}/close` | `{result, note}` | the run's summary |
| GET | `/api/v1/run_stats?device=&source=` | – | success rates and failure modes across runs |

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
  "options": {"save": true, "model_id": null, "run_id": "q1-tuneup", "stage": null}
}
```

| Field | Meaning |
|---|---|
| `scan.kind` | `PvP`, `PvT` or `tiebar` (see the table above). Other kinds can be submitted as training data for future models; analysing them returns 422. |
| `scan.x`, `scan.y` | The swept gates. PvP and tiebar: `x` is dot A, `y` dot B, and the tie bar is the (1,1)-(2,0) transition where dot A holds two (swap the axes to study (1,1)-(0,2)). PvT: one axis is the plunger, the other its reservoir tunnel gate (ChargeCell recognises tunnel gates from the device settings or a `T` name). Give either `values` or `start`/`stop`/`points`. Either sweep direction is fine. |
| `scan.signal` | `signal[iy][ix]`: `ny` rows of `nx` values. Use `null` for missing points. Or send it binary: `{"dtype": "float32", "shape": [ny, nx], "data": "<base64 of little-endian bytes, C order>"}` (dtypes: float32, float64, int16, int32, uint16, uint32). |
| `scan.voltage_unit` | `V` or `mV`. Applies to `x`, `y`, `voltage_state` and label coordinates. Responses are always in volts. |
| `scan.voltage_state` | DC voltages of the other gates during the scan. Spectator plungers matter most: they let ChargeCell check spectator occupancy against earlier scans. |
| `scan.device`, `scan.cooldown` | Device settings (safe limits, step size, gate names) are looked up by `device`. Set them on the GUI's Device page or with `PUT /api/devices/{name}`. `cooldown` keeps train/test splits honest. |
| `scan.id` | Optional; letters, digits, `.`, `_`, `-`. Sending the same id again overwrites that scan. |
| `options.save` | Store the scan and result in the workspace (default `true`). Stored results appear in the GUI, can be labelled, and let later guidance on the same device use them as history. |
| `options.model_id` | Use a specific model version instead of the active one. |
| `options.run_id` | The tune-up run to record this scan in (created if new). Default: the device's open run, or a new one after 4 hours without scans. See [Runs](#runs-the-automation-tree). |
| `options.stage` | Optional name of the tune-up stage, e.g. `"Q1 loading"`. Default: one stage per kind of scan and pair of gates. |
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
  "created": "2026-09-30T10:15:02+00:00",
  "run": {"run_id": "q1-tuneup", "stage_id": "n1", "node_id": "n7"}
}
```

### `outcome`: what the backend should do

| outcome | status | Meaning | Where to look |
|---|---|---|---|
| `found` | `FOUND` | PvP: (1,1) is in the window and the empty region (more than an electron spacing of it) is visible for both dots, so the count is certain. PvT: the empty dot and its first two loading lines are traced where electrons load cleanly. Tiebar: the tie bar and both triple points are in the window. | `features`. PvP: `next_scan` may hold a `readout_zoom` window to send as a `tiebar` scan. Tiebar: `next_scan` may be `retune_coupling` (change an exchange gate, rescan). |
| `next_scan` | `NOT_IN_WINDOW` | ChargeCell is confident where to look next (guidance confidence high or medium). `next_scan.purpose` `confirm`: the goal appears to be in this window and every check passed except the confidence threshold, so rescan the same window once with `next_scan.averaging` times longer integration. | `next_scan` |
| `no_confident_step` | `UNINTERPRETABLE`, or `NOT_IN_WINDOW` with nothing to navigate by | ChargeCell could not turn this scan into a next step with confidence. `reason` says why. | `suggestion` (best guess, for a person to review), `headline`, `steps` |

For `UNINTERPRETABLE` scans, `suggestion.purpose` is `rescan_after_fix`: fix the named problem
(retune the sensor, lower a barrier, average longer, ...), then rescan the same window. For
low-confidence exploration, `suggestion.purpose` is `locate`.

`needs_review: true` means the model was unsure or one of its checks disagreed. It can accompany
any outcome. A cautious automated loop should hand these scans to a person.

A typical automated tune-up of one qubit pair: PvT on each edge plunger until `found` (sets its
tunnel gate and loads one electron), PvP on the pair until `found`, then the `readout_zoom` window
as a `tiebar` scan until `found`, applying `retune_coupling` steps if you set a coupling target.

### `features` (outcome `found`)

| kind | type | label | Fields |
|---|---|---|---|
| PvP | `charge_cell` | `(1,1)` | `point` (centre, gate → V), `polygon` (outline as (x, y) in V), `extent` (gate → [min, max] V) |
| PvP | `transition_point` | `(1,1)-(2,0)`, `(1,1)-(0,2)` | `point`: midpoint of the boundary, useful for setting up readout |
| PvT | `loading_line` | `0->1`, `1->2`, ... | `polyline` along the line where electrons load visibly; `properties.electrons_before` |
| PvT | `operating_point` | `one electron` | `point` (plunger and tunnel gate); `properties.tunnel_gate_clean_from` / `_to`: the tunnel-gate range where electrons load cleanly (null: beyond the window) |
| tiebar | `triple_point` | `(1,0)-(1,1)-(2,0)`, `(1,1)-(2,0)-(2,1)` | `point` |
| tiebar | `tie_bar` | `(1,1)-(2,0)` | `polyline` (the two triple points), `point` (midpoint), `properties` below |
| tiebar | `readout_point` | `spin-to-charge readout (first guess)` | `point`: on the (2,0) side, just past the transition |

Tie-bar `properties`: `length_V`; `interdot_width_V` (FWHM of the charge transfer across the tie
bar, fitted on the raw signal); `coupling_ratio` = width / length, a dimensionless, voltage-free
measure of the interdot tunnel coupling (about 3 t_c / E_m, with E_m the mutual charging energy;
it cannot go below the thermal width, roughly 0.05-0.15 depending on the electron temperature
and E_m); `assessment` (`weak` / `in target` / `strong` against the device's
`tiebar_coupling_target`, or `strong` when the triple points are close to merging); and, only
when the device settings give plunger lever arms and the electron temperature,
`tunnel_coupling_ueV`, `tunnel_coupling_GHz` and `thermal_limited`.

Clients should ignore feature types and properties they do not know.

### `next_scan` / `suggestion` (a `ScanStep`)

| Field | Meaning |
|---|---|
| `purpose` | `locate` (move toward the goal), `confirm` (rescan the same window with longer averaging; asked at most once per window), `rescan_after_fix`, `readout_zoom` (PvP → take a tie-bar scan), or `retune_coupling` (tiebar: change the exchange gate in `gate_changes`, rescan) |
| `scan_kind` | The kind to send the next scan as (`tiebar` for a readout zoom, else the same kind). |
| `window` | Absolute start/stop in volts and the number of points for both swept gates. Always inside the device's safe limits. |
| `move` | Change of the window centre for each swept gate (V). If the full move is larger than `max_step`, ChargeCell gives only the first step and says so in `warnings`. The move can exceed `max_step` only when the window had to be shifted inside the safe limits, which `warnings` also reports. |
| `gate_changes` | Changes to other gates to make before scanning (V), e.g. `{"X1": -0.010}` to separate merged dots. |
| `physical_moves` | The move in physical gates, when the scan was in virtual gates and the device has a virtual-gate matrix. |
| `max_step` | Largest DC step the device allows (V), from the device settings. Ramp in steps no larger than this. `null`: no limit is set; moves were then limited to one window width. |
| `confidence`, `basis` | How the step was computed (`high`: from anchored transitions in this scan). |
| `averaging` | `confirm` only: integrate this many times longer per point than the scan just sent (4: noise halves). Otherwise `null`. |

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

Label fields mean the same as in the GUI's labeller (`docs/CODEMAP.md` §2, "Annotation"), for
every scan kind:

- Each `a_boundaries` entry is a polyline of (x, y) points, drawn bottom to top, separating k
  from k+1 electrons in dot A (the dot swept along x).
- `b_boundaries` are the same for dot B (along y), drawn left to right.
- `a_offset` is the number of electrons in dot A left of every boundary: `0` means that region
  is empty (at least an electron spacing of it is visible), and `null` means unknown.
  `b_offset` is the same for dot B, below every boundary.
- **Tie-bar scans:** `a_boundaries` holds dot A's 1 -> 2 boundary, which runs along the tie bar
  between the two triple points, and `b_boundaries` dot B's 0 -> 1 boundary. Leave the offsets
  out: they default to the tie-bar zoom's counts, 1 and 0.
- **PvT scans:** the loading lines of the plunger's dot go in `a_boundaries` when the plunger is
  on x (in `b_boundaries` when it is on y), with the count left of (below) the first line as
  its offset. `clean_T: [low, high]` gives the tunnel-gate values, in `voltage_unit`, between
  which electrons load cleanly; use `null` for a side where the clean range continues beyond
  the window.
- `reason` must fit `status` and the scan kind: `none` for FOUND; the kind's NOT_IN_WINDOW and
  UNINTERPRETABLE reasons otherwise (`chargecell schema` lists them; PvP: `no_transitions`,
  `occupancy_too_low`, `no_reference`, `partially_visible` / `low_snr`, `sensor_insensitive`,
  `dots_merged`, `charge_instability`, `resolution_too_coarse`; PvT adds
  `tunnel_rate_too_low` and `reservoir_too_open`; tie bar: `no_tiebar`, `partially_visible`).

A status alone is accepted, but the boundaries are what teach the network where things are.
Labelled scans of every kind are used when that kind's model is retrained.

## Runs: the automation tree

HRL records every action of an automated tune-up as a node of a tree, grades each node when it
happens, and mines the trees of many runs for success rates and failure modes (QPU paper,
S5.3). ChargeCell keeps the same record for everything it sees. It stays advisory, so it records
what it was sent, what it answered, and what the backend tells it; it never drives the device.

```
run            one tune-up session on one device
  stage        one goal pursued with one kind of scan and pair of gates ("Find the (1,1) cell of P1-P2")
    measure    a scan: window, gate changes since the last scan, whether it followed the last advice
      analysis status, reason, confidence (a re-analysis with a new model adds another)
      advice   found / next scan / no confident step, with the window
      review   a person's label for the scan: confirms or corrects the analysis
    action     something the backend or operator reports doing, or a note
```

Every saved analysis (`options.save`, the default) is recorded. The response's `run` block says
where. Pass `options.run_id` to group a tune-up's scans; without it a scan joins its device's
open run, or starts one after 4 hours without scans on that device. A scan starts a new stage
when its kind or gate pair differs from the previous scan's (or when `options.stage` names a new
stage); returning to an earlier goal later starts another stage, so the tree stays in time order.
A depth-first walk of the tree is the order in which things happened.

**Grades** are set when a node is recorded: `pass`; `warn` (a person should look: the model
asked for review, no confident next step, or a reviewer's label disagreed); `fail` (the scan
could not be interpreted, or, on practice devices, a FOUND that the ground truth shows is
wrong); `info` (actions and notes); `open` (stages and runs in progress). A stage is `pass` once
its goal is found, `fail` if it stalled (3 scans in a row without a confident next step) or its
FOUND was wrong, and `warn` if the run moved on or ended without reaching the goal. A closed run
is `pass` if every stage reached its goal, `fail` if it was aborted or a stage failed, else `warn`.

Start a run (optional; choose the id or let ChargeCell generate one):

```json
POST /api/v1/runs
{"device": "devA", "title": "Q1 tune-up, cooldown 7", "cooldown": "CD7", "run_id": "q1-tuneup"}
```

Report what the backend did between scans, so the tree is complete (`gate_changes` in
`voltage_unit`, default volts; `note: true` for a remark):

```json
POST /api/v1/runs/q1-tuneup/events
{"text": "retuned the sensor", "gate_changes": {"M1": 0.002}, "by": "tuneup.py"}
```

Close it (`result`: `done`, `stopped` or `aborted`):

```json
POST /api/v1/runs/q1-tuneup/close
{"result": "done", "note": "Q1 ready for readout calibration"}
```

`GET /api/v1/runs/{run_id}` returns the run with its nodes in depth-first order, each with
`{id, parent, type, time, grade, title, summary, data, depth}`. `data` holds the details: for a
measure the window, `voltage_state`, `gate_changes`, `followed` (`yes`/`no`/`null`) and
`deviation`; for an analysis `status`, `reason`, `confidence`, `needs_review`, `model_id`, the
checks that held back a FOUND, and `truth` on practice devices; for advice `outcome`, `headline`,
`steps`, `window`, `gate_changes` and the found `features`. `?format=text` gives one line per
node. `GET /api/v1/run_stats` summarises all runs per scan kind: goals reached, median and 90th
percentile scans to the goal, stalled and unfinished goals, wrong FOUNDs on practice devices,
the most common failed or flagged analyses, where unfinished goals stopped, how often the next
scan followed the advice, and how often reviewers' labels agreed.

The GUI shows runs on the **Runs** page; `chargecell runs [run_id] [--stats]` prints them.

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

run = cc.start_run(device="devA", title="Q1 tune-up")["id"]      # optional: group a tune-up
r = cc.analyze(..., run_id=run)
cc.log(run, "retuned the sensor", gate_changes={"M1": 0.002})
cc.close_run(run, "done")
print(cc.run_text(run))                              # the graded tree, one line per action
```

## Versioning

The `protocol` field is `chargecell/1`. Adding optional request fields, response fields, feature
types or enum values (for example new `kind`s or `purpose`s) keeps version 1. Clients should
ignore response fields they do not know. Renaming or removing fields, or changing their meaning,
requires `chargecell/2`.
