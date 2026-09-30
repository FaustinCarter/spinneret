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

## Install

Python 3.10 or newer.

```bash
pip install ./chargecell[spinqick]     # netCDF4 for spinQICK files
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
3. **Import real scans.** Scans page, **Import files**: spinQICK `.nc` files, or `.npz`, `.json`
   and `.csv` from elsewhere. Always fill in the cooldown.
4. **Review and act.** Each scan gets an outcome and next steps. Copy the settings, or download a
   spinQICK script that steps the DC point safely and runs the next `gvg_dc` scan.
5. **Label.** On the Label page, draw the dot boundaries, set the electron counts, choose the
   outcome. The queue shows the scans the model is least sure about first.
6. **Retrain.** On the Train page, combine synthetic datasets with your labels. Each version is
   scored on held-out cooldowns it never saw, and its FOUND threshold is calibrated so that FOUND
   calls are right at least 97% of the time (configurable). Make a version active when its real-data
   scores beat the previous one.

## spinQICK integration

spinQICK (HRL Laboratories' open-source QICK-based control software) is used where it fits:
**reading your measurements and running the next scan.**

- **Import:** files written by `SpinqickData.save_data` (e.g. from
  `TuneElectrostatics.gvg_dc` or `gvg_baseband`) are read directly, including gate names, swept
  voltages, analysed or raw IQ data, and the recorded DC voltage state.
- **Export:** the recommended next scan becomes a short script that reads the live DC point, moves
  in steps no larger than your limit with `vdc.set_dc_voltage_compensate` (sensor compensation),
  and calls `gvg_dc` with ranges relative to the new DC point. Read it before running it.

spinQICK does not contain a device simulator, so it cannot generate synthetic training data.
ChargeCell ships its own physics-based simulator for that (constant-interaction triple dot with
tunnel coupling, a realistic charge sensor, and measurement artefacts). The importer was written
against spinQICK's source and tested on files with the same layout; please check it on your own
files before relying on it.

## Command line

```bash
chargecell serve                                   # GUI
chargecell simulate --name sim1 -n 3000 --size 96  # synthetic dataset
chargecell train --synthetic sim1 --activate       # train and activate a model version
chargecell analyze scan.nc                         # print outcome and next steps
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
`docs/CODEMAP.md` (developer reference), `docs/HANDOFF.md` (status and next steps).

## Repository layout

```
chargecell/        the package (simulator, model, analysis, importers, server + GUI, CLI)
tests/             pytest suite (oracle-based guidance tests, API workflow, training smoke test)
scripts/           train_starter, eval_model, nav_trace, gui_check
docs/              operator guide, design, code map, research notes, decisions, handoff
```
