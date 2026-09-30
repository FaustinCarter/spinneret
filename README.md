# ChargeCell

ChargeCell looks at a double-quantum-dot charge stability diagram and answers one question:
**is the (1,1) charge cell in this window, and if not, where should I scan next?**

Every scan gets one of three outcomes:

| Outcome | Meaning | What ChargeCell gives you |
|---|---|---|
| **(1,1) found** | The (1,1) cell is visible *and* the empty (0,0) region is visible for both dots, so the electron count is certain. | Cell centre, outline, readout boundary, a zoom window for readout setup. |
| **(1,1) not in this window** | The scan is readable but (1,1) is elsewhere or cannot be counted to. | Which gates to move, which way, how far, and the next scan window (start, stop, points). |
| **Can't interpret this scan** | Too noisy, sensor off its flank, dots merged, charge jumps, too few points. | What to fix (retune sensor, lower barrier, average longer, more points) before rescanning. |

It comes with a GUI for people who do not write software: review results, label scans,
generate synthetic training data, train new model versions, and practise on simulated devices.
Measurement software talks to it through a small, backend-neutral JSON protocol
(`docs/PROTOCOL.md`), so it works with any control stack.

## Install

Python 3.10 or newer.

```bash
git lfs install                        # once per machine: model weights are stored with Git LFS
pip install .                          # from a clone of this repository
chargecell serve                       # opens http://127.0.0.1:8765
```

The workspace (scans, labels, models) lives in `~/chargecell-workspace` unless you pass
`-w /path/to/workspace`. If a starter model is bundled in `chargecell/assets/models/`, it is
installed on first start. **This first version does not bundle one yet**: generate a synthetic
dataset and train a model first (Synthetic data page, then Train page), or run
`python scripts/train_starter.py --bundle` once (about 40 minutes on one CPU core).

## Quick start

1. **Try it without hardware.** On the Review page click **New practice device**. ChargeCell
   simulates a triple dot and takes a first scan. Click **Analyse**, read the outcome, then click
   **Measure it on the practice device** to follow the advice. Repeat until (1,1) is found.
   **Reveal the true answer** shows whether the call was right.
2. **Set up your device.** On the Device page enter plunger, sensor and barrier gate names,
   safe voltage limits, the largest step you allow, and a rough addition voltage per plunger.
3. **Bring in real scans.** Either have your measurement code send them (see below), or use
   Scans page, **Import files**: `.json` (chargecell/1 requests or plain arrays), `.npz` and `.csv`.
   Always fill in the cooldown.
4. **Review and act.** Each scan gets an outcome and next steps. Copy the settings, or download
   the result as JSON for your measurement code.
5. **Label.** On the Label page, draw the dot boundaries, set the electron counts, choose the
   outcome. The queue shows the scans the model is least sure about first.
6. **Retrain.** On the Train page, combine synthetic datasets with your labels. Each version is
   scored on held-out cooldowns it never saw, and its FOUND threshold is calibrated so that FOUND
   calls are right at least 97% of the time (configurable). Make a version active when its real-data
   scores beat the previous one.

## Connecting your measurement setup

ChargeCell never drives instruments. Your measurement code sends each scan as JSON and gets back
one of three answers: the (1,1) cell's coordinates (`found`), the next window to scan
(`next_scan`), or `no_confident_step` with the reason and a suggestion for a person to review.

```python
from chargecell.client import ChargeCellClient

cc = ChargeCellClient("http://127.0.0.1:8765")          # a running `chargecell serve`
r = cc.analyze(signal, x_volts, y_volts, x_gate="P1", y_gate="P2", device="devA",
               cooldown="CD7", voltage_state={"P3": 0.845, "M1": 0.920})
print(r["outcome"], r["headline"])
```

The same JSON works over HTTP from any language (`POST /api/v1/analyze`), or as files
(`chargecell analyze request.json --json`). Labelled scans can be submitted for training with
`POST /api/v1/scans`. Full specification: `docs/PROTOCOL.md`.

Synthetic training data come from ChargeCell's own physics-based simulator: a constant-interaction
triple dot with tunnel coupling, a realistic charge sensor, and measurement artefacts.

## Command line

```bash
chargecell serve                                   # GUI
chargecell simulate --name sim1 -n 3000 --size 96  # synthetic dataset
chargecell train --synthetic sim1 --activate       # train and activate a model version
chargecell analyze scan.csv --x-gate P1 --y-gate P2  # print outcome and next steps
chargecell analyze request.json --json            # chargecell/1 request -> response
chargecell schema                                  # JSON Schema of the protocol
chargecell navigate --devices 20                   # test guidance on simulated devices
```

## What is validated, and what is not

- **Guidance logic** (tested with perfect perception on 30 random simulated devices): reaches the
  correct (1,1) cell on 30/30, median 2.5 scans, 90th percentile 4. Recommended moves always stay
  within safe limits and are split into steps no larger than the configured maximum.
- **Model**: no trained model ships yet. A first run on 4000 simulated 64x64 scans reached about
  72% outcome accuracy and 77% per-pixel electron-count accuracy on simulated validation data
  before it was interrupted. The conservative FOUND rule and review flags compensate for this.
  Accuracy on your device is unknown until you label real scans from at least two cooldowns and
  retrain; the Train page then reports held-out real-data scores.
- **Not yet covered:** plunger-vs-barrier and readout ("tiebar") scans, hole devices in the
  labelling tool, and verification of spectator dots beyond consistency with earlier scans.

More: `docs/OPERATOR_GUIDE.md` (plain-language guide), `docs/DESIGN.md` (how it works),
`docs/PROTOCOL.md` (integration), `docs/CODEMAP.md` (developer reference),
`docs/HANDOFF.md` (status and next steps).

## Repository layout

```
chargecell/        the package (simulator, model, analysis, protocol, server + GUI, CLI)
tests/             pytest suite (oracle-based guidance tests, API workflow, training smoke test)
scripts/           train_starter, eval_model, nav_trace, gui_check
docs/              operator guide, design, code map, research notes, decisions, handoff
```
