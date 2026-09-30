# Handoff: state of ChargeCell

Updated 2026-09-30 by the second Claude session (Claude Code, cloud container with 4 CPU cores,
15 GB RAM, no GPU). The first build (2026-09-29, claude.ai sandbox) is summarised in section 1.
Read this first, then `docs/CODEMAP.md` before changing code.

## 1. What the user asked for

The user works with Si/SiGe exchange-only (EO) spin-qubit devices in HRL Laboratories' style.

**First session (2026-09-29):** a system that looks at a double-quantum-dot charge stability
diagram and either finds the (1,1) cell, says it is not in the window, or says the window is not
interpretable; a reading of HRL's EO papers; a full build (simulated training data, easy expert
annotation, training pipeline, a GUI for non-programmers, next-scan guidance, open-source models
only where they clearly help); packaging for GitHub and Claude Code.

**Second session (2026-09-30), in the user's words or close to them:**
- Unpack the project into the empty `spinneret` repository and continue (the package keeps the
  name ChargeCell).
- "Remove spinQICK entirely": spinQICK has no simulator. Instead, "a generic format for any
  backend to send in data for analysis or training and receive a response that includes either
  coordinates for (1,1) cell, tie-bar, etc. or instructions for where to scan next, or an
  indicator that the image wasn't able to generate a next step with any confidence."
  → the chargecell/1 protocol (`docs/PROTOCOL.md`).
- "Generate some simulated data for now" (no real data yet); read the literature for
  representative EO data.
- "The protocols should be agnostic to actual voltages as those will vary widely based on device
  and technology." → no voltage scale anywhere (DECISIONS 16).
- Gate names follow the HRL convention: P1, P2, ... plungers; X1, X2, ... exchange gates (T
  tunnel gates and M sensors as in HRL's papers).
- "Let's do the PvT tie-bar models next." → done (DESIGN sections 8, 9).

Answers to the first session's open questions: keep the name ChargeCell; train here on CPU
(PvP at 96 px, 3 members); bundled weights in **Git LFS**; **no license** for now; acceptance bar
as proposed (FOUND precision ≥ 0.95 on held-out synthetic data; closed loop finds the goal on
≥ 70% of practice devices with no wrong FOUND); **advisory only** (ChargeCell never moves gates);
priorities after PvT/tie-bar: the **automation tree / audit log**. Spectator verification and
hole-device labelling were not chosen.

**Foundation models are still deliberately not used** (DESIGN section 5, DECISIONS 6).

## 2. Status summary

| Area | State |
|---|---|
| chargecell/1 protocol (HTTP, files, Python client), spinQICK removed | Done, tested |
| Voltage-agnostic settings and guidance | Done, tested (random-scale practice devices) |
| PvP simulator, labels, model, decision, guidance | Done; anchoring and demotion fixed after evaluating the first model |
| PvT simulator, model, analysis, guidance | Done; see section 3 for the trained model |
| Tie-bar simulator, model, analysis (measured coupling), guidance | Done; see section 3 |
| Practice devices with P, X, T gates; closed-loop evaluation per kind | Done |
| GUI: kinds, practice per kind, keypoint display, Runs page | Done, browser-checked (`scripts/gui_check.py`) |
| Labelling for all three kinds (labeller, model drafts, protocol, training) | Done, tested against simulator truth |
| Starter models bundled (Git LFS) | In progress (PvP retraining; PvT and tie-bar next) |
| CI (GitHub Actions) | Green (one red run, the label round trip, fixed in the next push) |
| Automation tree / audit log (`runs.py`, Runs page, protocol run endpoints, CLI) | Done, tested |

## 3. Verified results (reproducible)

- `pytest -q`: 33 tests (about 3 minutes on 4 cores; `test_kinds.py` trains tiny models).
- Guidance with perfect perception, 30 practice devices each (`python scripts/nav_trace.py
  --oracle --bench --kind <kind>`), every FOUND correct: PvP 30/30 (median 3 scans; 30/30 also
  with `--no-prior`), PvT 30/30 (median 3 since the stricter anchoring; half the devices start
  with the tunnel gate too closed; 30/30 with `--no-prior`), tie bar 30/30 (median 1).
- Tie-bar coupling measurement: the fitted interdot width is within 35% of the physics
  (`test_tiebar_width_is_measured_from_the_signal`).
- Trained starter models (`scripts/eval_model.py`, fresh held-out scans, seed 2027; closed loop
  on 20 practice devices):

(Pending: filled in once the starter models are trained and evaluated.)

- Earlier model versions of this session, for the record (each failure was traced with the
  closed-loop records and fixed in the simulator, the harness or the decision logic):
  - PvP v0 (96 px, 3 x 12 epochs, 4000 scans, old labels): FOUND precision 0.97, recall 0.27,
    closed loop 5/20 with one wrong FOUND → DECISIONS 17, 18.
  - PvP v1 (96 px, 3 x 14 epochs, 6000 scans): held-out precision 0.976, recall 0.70; closed
    loop 12/20 found, 9 right (the practice operator ignored "retune the sensor") → DECISIONS
    25, 26 (practice devices follow the fixes, hidden-line check, harder training windows,
    zoom out instead of rescanning the same window).
  - PvT v1 (64 px, 2 x 10 epochs, 5000 scans): held-out precision 0.96, recall 0.65; closed
    loop 6/20, one wrong. Practice windows (the operator's first guess, up to ~15 decades of
    tunnel rate, 90 x 90 points) lay outside the training windows (2-8 decades, often ~24
    rows): the model read latching as charge instability. Training windows widened.
  - Tie bar v1 (64 px, 2 x 10 epochs, 5000 scans): held-out precision 0.96, recall 0.27;
    closed loop 2/20. Practice sensors could not see the interdot step (truly too noisy in half
    the scans); practice devices now have a readout-capable sensor, and the model gets twice
    the data and longer training.

## 4. Unfinished work, in priority order

### 4.1 Done this session: automation tree / audit log

Built as proposed (DESIGN §12, PROTOCOL "Runs", DECISIONS 23-24): `runs/<id>.json` trees of
run → stage → measure → analysis/advice/review, plus actions and notes; graded when recorded;
`options.run_id` / `options.stage` and `/api/v1/runs*` endpoints for backends; automatic grouping
per device (4-hour gap); practice loops and `virtual.evaluate` record runs with truth-checked
FOUNDs; Runs page and `chargecell runs`. Possible follow-ups, if the user wants them: ChargeCell
proposing the next *stage* (PvT → PvP → tie bar per qubit, open question 1), and comparing
statistics between model versions (runs already store `model_id` per analysis).

### 4.2 Done this session: labels for PvT and tie-bar scans

The labeller, drafts from the model, validation, the protocol (`clean_T`) and training
(`real_arrays`) handle all three kinds (CODEMAP §2 "Annotation"). Checked against simulator
truth only; the first real PvT and tie-bar scans should be labelled by an expert and compared
with the drafts before relying on them.

### 4.3 P1: Real data

No real data yet (the user asked to work on simulations for now). When scans arrive: label from
at least two cooldowns, retrain with `use_real` on, compare real-data metrics; tune the simulator
presets (line widths, sensor drift, noise, tunnel-rate ranges) against real scans; check the
tie-bar coupling ratio against independent tunnel-coupling measurements.

### 4.4 P2: Extensions

- Readout calibration after the tie bar (HRL's spin-to-charge histogram step, Fig. S20f).
- PvT for interior dots (loaded through neighbours, no own reservoir) and a tie bar for
  (1,1)-(0,2) without swapping axes.
- Spectator verification as a status, cross-pair consistency (not chosen by the user yet).
- Hole-device labelling conventions (analysis already handles holes).
- Simulator realism: compare with QDarts / qarray; reservoir-starved interior dots.

### 4.5 P3: Housekeeping

- Starlette's TestClient warns that httpx will be replaced by httpx2.
- CI does not fetch LFS weights (tests do not need them; saves LFS bandwidth).

## 5. Known issues and sharp edges

- **One heavy job at a time.** On 4 cores, running data generation or the test suite during
  training slowed epochs from ~90 s to ~13 min (load average 10). Train in the background and do
  light work meanwhile.
- **Oracle tests re-render without noise**, so the oracle can call FOUND on a scan whose noisy
  truth is low_snr; correctness is judged geometrically per kind (`virtual.*_is_right`).
- **Tie-bar triple points** are located on a fine ground-state map: the spectator can change
  occupancy right at a triple point when mutual charging is strong, which fixed-spectator
  formulas get wrong.
- **PvT orientation**: analysis transposes a scan with the tunnel gate on x; results and overlays
  carry the analysis's gate order (`analysis.scan`, `overlays.x_gate`), and the GUI maps by gate
  name.
- **Practice devices shift with T and X gates** (`virtual.device_state` also moves `v11`); the
  PvP ground truth uses the shifted device.
- **Practice devices follow the advice** since this session: the sensor is tuned at the start
  voltages and retuned when ChargeCell reports sensor_insensitive or low_snr (which also averages
  longer), and advised gate changes are applied (`virtual.apply_advice`). Closed-loop numbers
  from before this change are not comparable (the loop used to stall on sensor problems nobody
  fixed).
- **The automation tree records every saved analysis**, including those in tests and in
  `chargecell navigate`/`eval_model.py` (practice runs, source `practice`). Each practice device
  has its own device name, so each gets its own run. Batch re-analysis does not record.
- **Label drafts from occupancy maps** keep boundaries that only clip a window corner (2+
  points); dropping them used to shift every count in the window by one. Maps with charge jumps
  (non-monotonic occupancy) still cannot be represented by boundary polylines.
- Earlier items still true: the line-map lattice joins reservoir segments through interdot
  segments, rejects spacings far from the prior/history, and never shrinks a window on an
  unmeasured spacing (don't undo these together); only measured spacings are stored in
  `lattice_v`; `plot.js` stays an IIFE; `[hidden] { display: none !important }` is required;
  `JobManager` runs one job at a time.

## 6. Open questions for the user

1. **Automation tree**: it records and grades what the backend does (4.1). Should ChargeCell
   also plan the sequence of stages per qubit (PvT → PvP → tie bar), or leave that to the
   backend?
2. **Tie-bar coupling target**: what range of coupling ratio (or t_c in µeV, with lever arms and
   electron temperature) do you want for readout? Without it ChargeCell only reports.
3. **Real data**: when available, a few scans of each kind (any format) from two cooldowns.
4. **Device facts** that are still unknown to ChargeCell: which T gate serves which plunger on
   your devices, and whether interior dots need their own loading procedure.

## 7. How to resume

```bash
git lfs install && git lfs pull
pip install -e ".[dev]"
pytest -q                                              # expect 24 passed
python scripts/nav_trace.py --oracle --bench --kind PvP   # and PvT, tiebar: expect 30/30
chargecell -w /tmp/demo serve --no-browser             # GUI; practice devices on the Scans page
```

Then continue with section 4 in order.
