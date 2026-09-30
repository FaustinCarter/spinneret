# ChargeCell

ChargeCell reads the charge-stability scans of an exchange-only spin-qubit tune-up and tells you
what they show and where to scan next. It follows HRL's pipeline, with one model per kind of scan:

| Scan | Question | When found, you get |
|---|---|---|
| **Plunger vs tunnel gate (PvT)** | Where does the edge dot hold one electron, and at which tunnel-gate setting do electrons load cleanly? | Loading lines, the clean tunnel-gate range, the one-electron operating point. |
| **Plunger vs plunger (PvP)** | Is the (1,1) charge cell in this window? | Cell centre and outline, readout boundaries, a window for the tie-bar scan. |
| **Tie bar** (PvP zoomed on (1,1)-(2,0)) | Where are the triple points, how strongly are the dots coupled, where to read out? | Triple points, a coupling measurement, a first spin-to-charge readout point. |

Every scan gets one of three outcomes:

| Outcome | Meaning | What ChargeCell gives you |
|---|---|---|
| **Found** | The target is in view and certain (for (1,1): the empty region is visible for both dots, so electrons can be counted). | The keypoints above. |
| **Not in this window** | The scan is readable but the target is elsewhere or cannot be counted to. | Which gates to move, which way, how far, and the next scan window (start, stop, points). |
| **Can't interpret this scan** | Too noisy, sensor off its flank, dots merged, charge jumps, too few points. | What to fix (retune the sensor, lower an exchange gate, average longer, more points) before rescanning. |

ChargeCell assumes **no voltage scale**: devices and technologies differ widely, so every number
is measured from your scans or taken from your (optional) device settings.

It comes with a GUI for people who do not write software: review results, label scans,
generate synthetic training data, train new model versions, and practise on simulated devices.
Measurement software talks to it through a small, backend-neutral JSON protocol
(`docs/PROTOCOL.md`), so it works with any control stack.

Every analysed scan is also recorded in an **automation tree**, as in HRL's tune-up software: each
run of a tune-up is a tree of scans, conclusions, advice and reported actions, graded as they
happen, so a long unattended run can be audited afterwards and many runs can be compared (scans
needed per goal, where runs stall, which problems come up most).

## Install

Python 3.10 or newer.

```bash
git lfs install                        # once per machine: model weights are stored with Git LFS
git lfs pull                           # in your clone, if it was made before git-lfs was set up
pip install .                          # from a clone of this repository
chargecell serve                       # opens http://127.0.0.1:8765
```

The workspace (scans, labels, models) lives in `~/chargecell-workspace` unless you pass
`-w /path/to/workspace`. The starter models bundled in `chargecell/assets/models/` (one per scan
kind, trained on simulated scans) are installed on first start. To rebuild them:
`python scripts/train_starter.py --kind <PvP|PvT|tiebar> ... --bundle` (see the script).

## Quick start

1. **Try it without hardware.** On the Scans page choose a scan kind next to **New practice
   device**. ChargeCell simulates a triple dot and takes a first scan. Click **Analyse**, read
   the outcome, then **Measure it on the practice device** to follow the advice. Repeat until it
   says found. **Reveal the true answer** shows whether the call was right.
2. **Set up your device** (optional). On the Device page enter gate names (P plungers, X
   exchange gates, T tunnel gates, M sensor), safe limits, the largest step you allow, and
   optionally a rough electron spacing per plunger.
3. **Bring in real scans.** Either have your measurement code send them (see below), or use
   Scans page, **Import files**: `.json` (chargecell/1 requests or plain arrays), `.npz` and `.csv`.
   Always fill in the cooldown and the scan kind.
4. **Review and act.** Each scan gets an outcome and next steps. Copy the settings, or download
   the result as JSON for your measurement code.
5. **Label.** On the Label page, draw the dot boundaries, set the electron counts, choose the
   outcome (plunger-vs-plunger scans for now). The queue shows the scans the model is least
   sure about first.
6. **Audit.** The Runs page shows each tune-up as a graded tree: every scan, what ChargeCell
   concluded and advised, whether the next scan followed the advice, and where it got stuck.
7. **Retrain.** On the Train page, pick a scan kind and combine synthetic datasets with your
   labels. Each version is scored on held-out cooldowns it never saw, and its FOUND threshold is
   calibrated so that FOUND calls are right at least 97% of the time (configurable). Make a
   version active when its real-data scores beat the previous one.

## Connecting your measurement setup

ChargeCell never drives instruments. Your measurement code sends each scan as JSON and gets back
one of three answers: the keypoints (`found`), the next window to scan (`next_scan`), or
`no_confident_step` with the reason and a suggestion for a person to review.

```python
from chargecell.client import ChargeCellClient

cc = ChargeCellClient("http://127.0.0.1:8765")          # a running `chargecell serve`
r = cc.analyze(signal, x_volts, y_volts, x_gate="P1", y_gate="P2", device="devA",
               cooldown="CD7", voltage_state={"P3": 0.845, "M1": 0.920})
print(r["outcome"], r["headline"])
r = cc.analyze(signal, p1_volts, t1_volts, x_gate="P1", y_gate="T1", kind="PvT", device="devA")
```

The same JSON works over HTTP from any language (`POST /api/v1/analyze`), or as files
(`chargecell analyze request.json --json`). Labelled scans can be submitted for training with
`POST /api/v1/scans`. To group a tune-up into one run of the automation tree, start a run and
pass its id (`cc.start_run(...)`, `analyze(..., run_id=...)`), report what your code does between
scans (`cc.log(...)`) and close it (`cc.close_run(...)`); without a run id, scans join their
device's open run. Full specification: `docs/PROTOCOL.md`.

Synthetic training data come from ChargeCell's own physics-based simulator: a constant-interaction
triple dot with tunnel coupling, a realistic charge sensor, reservoir tunnel rates set by the
tunnel gates, and measurement artefacts.

## Command line

```bash
chargecell serve                                                # GUI
chargecell simulate --name sim1 -n 3000 --size 96               # synthetic dataset (--kind PvT|tiebar)
chargecell train --synthetic sim1 --activate                    # train and activate (--kind ...)
chargecell analyze scan.csv --x-gate P1 --y-gate P2             # print outcome and next steps
chargecell analyze request.json --json                          # chargecell/1 request -> response
chargecell schema                                               # JSON Schema of the protocol
chargecell navigate --devices 20                                # test guidance on simulated devices
chargecell runs                                                 # tune-up runs; `runs <id>` prints one, --stats
```

## What is validated, and what is not

(Pending: filled in once the starter models are trained and evaluated.)

More: `docs/OPERATOR_GUIDE.md` (plain-language guide), `docs/DESIGN.md` (how it works),
`docs/PROTOCOL.md` (integration), `docs/CODEMAP.md` (developer reference),
`docs/HANDOFF.md` (status and next steps).

## Repository layout

```
chargecell/        the package (simulators, models, analysis per scan kind, protocol, automation tree, server + GUI, CLI)
tests/             pytest suite (oracle-based guidance tests, API workflow, training smoke tests)
scripts/           train_starter, eval_model, nav_trace, gui_check
docs/              operator guide, design, protocol, code map, research notes, decisions, handoff
```
