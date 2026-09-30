# Handoff: state of ChargeCell

Updated 2026-09-30 by the second Claude session (Claude Code, cloud container with 4 CPU cores,
15 GB RAM, no GPU), last after the model-switching, remote-training and interface work (section
4.2). The first build (2026-09-29, claude.ai sandbox) is summarised in section 1.
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
- Later in the session: "focus less on getting the model training right, and more on: making
  sure we can easily switch models; making sure it's easy to train on gpu that might not [be] on
  the same machine as the web ui; making sure the user interface is really easy to use and
  intuitive; making sure that any text/instructions/labels are jargon free and use simple plain
  technical English. For technical terms that are unavoidable, add a glossary to the readme."
  → done (section 4.2; DESIGN 13; DECISIONS 33-35).

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
| Starter models bundled (Git LFS) | **Not bundled**: none meets the acceptance bar yet (section 3); candidates kept in `models/candidates/` (Git LFS, verified upload) |
| CI (GitHub Actions) | Green (one red run, the label round trip, fixed in the next push) |
| Automation tree / audit log (`runs.py`, Runs page, protocol run endpoints, CLI) | Done, tested |
| FOUND threshold calibrated on the final decision; test-time augmentation; `chargecell calibrate` | Done, tested (DECISIONS 27, 28) |
| Confirmation rescan for a FOUND held back only by confidence (all kinds, protocol purpose `confirm`) | Done, tested (DECISIONS 30) |
| Simulator "too noisy" labels use the noise visible in the image | Done; all three kinds retrained on it (DECISIONS 31) |
| Model versions: names, plain test summaries, switching, model files (zip/folder), Models page, `chargecell models` | Done, tested (`test_models.py`) |
| Training on another computer: training-job files, `chargecell worker` / `train-job`, GPU selection, Train page | Done, tested (`test_remote.py`); checked over real HTTP on CPU only (no GPU here) |
| Web page: Home, Models, four-step Train page, plain-language labels and advice, README glossary | Done, browser-checked (`scripts/gui_check.py`) |

## 3. Verified results (reproducible)

- `pytest -q`: 43 tests (about 1.5-3 minutes on 4 cores; `test_kinds.py` and `test_remote.py`
  train tiny models).
- Guidance with perfect perception, 30 practice devices each (`python scripts/nav_trace.py
  --oracle --bench --kind <kind>`), every FOUND correct: PvP 30/30 (median 3 scans; 30/30 also
  with `--no-prior`), PvT 30/30 (median 3 since the stricter anchoring; half the devices start
  with the tunnel gate too closed; 30/30 with `--no-prior`), tie bar 30/30 (median 1).
- Tie-bar coupling measurement: the fitted interdot width is within 35% of the physics
  (`test_tiebar_width_is_measured_from_the_signal`).
- **Starter models: none meets the acceptance bar yet, so none is bundled.** All three kinds
  were retrained on the corrected labels (DECISIONS 31) and calibrated at target precision 0.99
  on held-out scans that share no seeds with training (`chargecell calibrate`). Final numbers on
  seed 2029, which no decision looked at (`scripts/eval_model.py --n 300 --nav 30 --seed 2029`;
  closed loop: 30 practice devices, at most 8 scans each). The models are kept, not installed,
  in `models/candidates/` (Git LFS).

  | Kind | Model | Training | Threshold | Held-out FOUND precision / recall | Closed loop: found, wrong, median scans | Bar |
  |---|---|---|---|---|---|---|
  | PvP | `model-20260930-073322754-6eac19` | 96 px, 3 x 14 epochs, 10000 scans, 2 h | 0.995 | 0.973 / 0.41 | 15/30, 0 wrong, 5 | precision and "no wrong FOUND" met; 21/30 needed |
  | PvT | `model-20260930-095542034-6aca7b` | 64 px, 2 x 12 epochs, 8000 scans, 22 min | 0.995 | 0.958 / 0.41 | 6/30, 1 wrong, 4.5 | not met |
  | Tie bar | `model-20260930-110759674-24455b` | 64 px, 2 x 14 epochs, 10000 scans, 33 min | 0.995 | 1.0 / 0.17 (seed 2028) | 9/30, 0 wrong, 3 (seed 2028) | "no wrong FOUND" met; 21/30 needed |

  Why the threshold is 0.995 (the calibration's cap): at the usual target 0.97 on their own
  validation split the thresholds would be 0.795 (PvP; precision 0.972, recall 0.66), 0.83
  (PvT; 0.971, 0.77) and 0.975 (tie bar; 0.970, 0.59), but closed loops visit mostly near-miss windows, and there the PvP network
  gave geometrically wrong FOUNDs at 0.96-0.99 (far-dot miscounts by one; DECISIONS 32). At
  0.995 no PvP closed loop (seeds 2028, 2029) had a wrong FOUND. What limits PvP now is
  discrimination: in the final loop, 30 scans with (1,1) in view were held back by confidence
  alone (median 0.94). PvT's loop fails on perception: latching read as charge instability, the
  tunnel regime misjudged ("reservoir too open" on clean windows), and an empty dot "seen" when
  the first loading line sat at the window edge; 20-25% of its confident FOUND candidates in the
  loop were wrong at every confidence level. Section 4.1 lists the options.

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
  - PvP v2 (96 px, 3 x 14 epochs, 8000 scans): threshold 0.995, recall 0.19, closed loop 4/30.
    Calibrated on the final decision (DECISIONS 27): 0.985, recall 0.35, closed loop 12/30 with
    2 wrong; with test-time augmentation (DECISIONS 28): 9/30, none wrong; with the confirm
    step (DECISIONS 30): 8/30, none wrong. On the practice scans that held (1,1) the network
    mostly said FOUND at 0.8-0.98, below the threshold. The threshold was forced up by scans the
    simulator wrongly labelled too noisy (DECISIONS 31). Recalibrated against corrected labels
    on 2500 held-out scans: threshold 0.585, precision 0.97, recall 0.66. (A calibration on
    scans from the training seeds gave 0.5 and a closed loop of 23/30 with 4 wrong FOUNDs at
    confidence 0.57-0.82: a sensor peak read as two transitions, a strongly tilted pattern with
    the dots mixed up, a faint far-dot line hallucinated, a cell sliver at the window edge.)
  - PvT v2 (64 px, 2 x 12 epochs, 6000 scans): held-out 0.967 / 0.586, closed loop 7/20 with 3
    wrong. PvT v3 (7000 scans, stricter anchoring labels, old noise labels; final-decision
    calibration and TTA): threshold 0.985, held-out precision 1.0, recall 0.50, closed loop
    9/20 with 2 wrong.
  - All three kinds were then retrained on data with the corrected noise labels (below).

## 4. Unfinished work, in priority order

### 4.1 P0: Starter models that meet the acceptance bar

Section 3 has the numbers. What was learned, in the order it matters:

1. **Labels first.** The simulator's "too noisy" labels were wrong for streaky but readable scans
   (DECISIONS 31); fixing them did more than any training change. Check new label rules against
   images before training on them.
2. **Closed loops, not the synthetic mix, set the threshold.** Near-miss windows dominate a loop.
   Calibrate at 0.99 on held-out data, then check a development loop (seeds 2027/2028 were used
   for development here; 2029 is the final check) before trusting a model.
3. **PvP needs better discrimination near (1,1).** The wrong calls were far-dot miscounts by
   one (faint lines), and in the final loop 30 scans that held (1,1) were stopped by confidence
   alone. The next step is training aimed at that failure: more weak far-dot coupling and
   drifted-sensor windows near (1,1) in the synthetic data, and more epochs (validation accuracy
   was still creeping up at epoch 14). About 2 h per PvP run on 4 cores; nothing else may run
   meanwhile (section 5). Not worth doing: accepting a FOUND when the confirmation rescan agrees
   with the first scan at a lower threshold. Estimated from the closed-loop records (runs of
   seeds 2028 and 2029): 2-3 of every 6-10 agreeing pairs were wrong, because a rescan of the
   same window repeats the same miscount, and only 1-2 devices would have been gained.
4. **PvT needs perception work** before threshold tuning can help: latching still read as charge
   instability (widen the latching range in training windows), the tunnel-gate regime
   misjudged on clean windows, and a line-free stretch left of the first traced line taken as the
   empty dot when the real first line sat at the window edge (train with first lines at the edge;
   consider requiring the empty stretch to exceed the widest occupied cell seen in the scan).
5. **Tie bar**: safe (no wrong FOUND) but it stops early: 8 of its 21 stalled loops ended on
   "dots merged" for tie bars that were strongly coupled but still two dots. Compare the
   practice devices' coupling range with the training windows, and add strongly coupled,
   still-separate tie bars to training. (Its final numbers are on seed 2028, which was fresh for
   this kind; no decision looked at it.)
6. **Bundle the conservative PvP model now?** It gave no wrong FOUND in 60 closed loops and
   0.973 held-out precision, but finds (1,1) on half the devices (bar: 70%). The user decides
   (open question 1). To bundle: `bundle()` in `scripts/train_starter.py`, then check README and
   OPERATOR_GUIDE ("installed on first start").

### 4.2 Done this session: switching models, training elsewhere, a plainer interface

What the user asked for last (section 1), and what exists now:

- **Switching models.** Models page (per scan kind: the model in use with plain test results,
  other versions with Use / Rename / Download / Delete, Add a model file from a zip or a folder,
  an offer to re-analyse after switching); `chargecell models list|use|add|export|rename|delete`;
  HTTP endpoints in CODEMAP §4. Model files are validated before anything changes.
- **Training on a GPU elsewhere.** Train page → "Another computer": the page shows the
  `chargecell worker --server ... --token ...` command (or, when ChargeCell listens only on
  127.0.0.1, the two ways to connect: `--host 0.0.0.0` on a trusted network, or an SSH tunnel),
  follows the remote progress, and can cancel or retry. No network: download the training-job
  file, `chargecell train-job` on the GPU computer, add the model file. `--device auto|cuda|mps|
  cpu`. **Not yet run on a real GPU** (this container has none): the CUDA path is the standard
  PyTorch one (model and batches moved to the device, pinned memory, loader workers), but the
  first real GPU run should be watched (open question 7).
- **Interface.** Home page (tasks, getting-started checklist, models in use), a four-step Train
  page, renamed navigation (History, Simulated scans, Device settings), plain words in the GUI,
  the advice texts (`recommend.py`, `pvt.py`, `tiebar.py`), the reason texts and the CLI help;
  README rewritten with sections on switching models and training elsewhere, and a glossary;
  OPERATOR_GUIDE updated.

Possible follow-ups: a model comparison view (two versions side by side on the same scans);
showing on the Train page when a worker last asked for work (workers are only visible once they
claim a job); a first-run wizard for Device settings.

### 4.2b Done this session: automation tree / audit log

Built as proposed (DESIGN §12, PROTOCOL "Runs", DECISIONS 23-24): `runs/<id>.json` trees of
run → stage → measure → analysis/advice/review, plus actions and notes; graded when recorded;
`options.run_id` / `options.stage` and `/api/v1/runs*` endpoints for backends; automatic grouping
per device (4-hour gap); practice loops and `virtual.evaluate` record runs with truth-checked
FOUNDs; Runs page and `chargecell runs`. Possible follow-ups, if the user wants them: ChargeCell
proposing the next *stage* (PvT → PvP → tie bar per qubit, open question 1), and comparing
statistics between model versions (runs already store `model_id` per analysis).

### 4.3 Done this session: labels for PvT and tie-bar scans

The labeller, drafts from the model, validation, the protocol (`clean_T`) and training
(`real_arrays`) handle all three kinds (CODEMAP §2 "Annotation"). Checked against simulator
truth only; the first real PvT and tie-bar scans should be labelled by an expert and compared
with the drafts before relying on them.

### 4.4 P1: Real data

No real data yet (the user asked to work on simulations for now). When scans arrive: label from
at least two cooldowns, retrain with `use_real` on, compare real-data metrics; tune the simulator
presets (line widths, sensor drift, noise, tunnel-rate ranges) against real scans; check the
tie-bar coupling ratio against independent tunnel-coupling measurements.

### 4.5 P2: Extensions

- Readout calibration after the tie bar (HRL's spin-to-charge histogram step, Fig. S20f).
- PvT for interior dots (loaded through neighbours, no own reservoir) and a tie bar for
  (1,1)-(0,2) without swapping axes.
- Spectator verification as a status, cross-pair consistency (not chosen by the user yet).
- Hole-device labelling conventions (analysis already handles holes).
- Simulator realism: compare with QDarts / qarray; reservoir-starved interior dots.

### 4.6 P3: Housekeeping

- Starlette's TestClient warns that httpx will be replaced by httpx2.
- `<workspace>/tmp/` collects uploads and model-file exports; nothing cleans it yet.
- CI does not fetch LFS weights (tests do not need them; saves LFS bandwidth).

## 5. Known issues and sharp edges

- **One heavy job at a time.** On 4 cores, running data generation or the test suite during
  training slowed epochs from ~90 s to ~13 min (load average 10); even a single-worker
  generation of 1250 tie-bar scans made one PvP epoch take 17.6 min instead of 2.4. Train in
  the background and do light work meanwhile.
- **Calibration data must be held out by seed.** Synthetic datasets are deterministic in their
  seed: a "new" set generated with a training seed re-renders the training images (with the
  labels of the current code), and a threshold calibrated on it is too low (DECISIONS 31).
  `recalibrate(ws, model_id, held_out=[...])` takes held-out sets explicitly.
- **Thresholds and practice loops.** The synthetic calibration mix has far fewer near-miss
  windows than a closed loop visits (the guidance steers toward (1,1), so most scans in a loop
  are close calls). A threshold that gives precision 0.97 on the mix can still give a wrong
  FOUND in a loop of 30 devices; check the closed loop before trusting a calibration.
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
- **Remote training.** The worker token is in `<workspace>/worker_token` (delete it to make a
  new one; running workers then need the new command). A job file is deleted when its model
  comes back or the job is cancelled; a failed job keeps it for "Try again". The web page has no
  login: never start it with `--host 0.0.0.0` on an untrusted network. The worker bypasses web
  proxies only for local and private addresses (`worker._is_local`).
- **Words on screen.** Page ids in URLs and code are unchanged (`runs`, `synthetic`, `device`)
  while the navigation says History, Simulated scans, Device settings. `scripts/gui_check.py`
  clicks some buttons by their text; update it when renaming buttons.
- Earlier items still true: the line-map lattice joins reservoir segments through interdot
  segments, rejects spacings far from the prior/history, and never shrinks a window on an
  unmeasured spacing (don't undo these together); only measured spacings are stored in
  `lattice_v`; `plot.js` stays an IIFE; `[hidden] { display: none !important }` is required;
  `JobManager` runs one job at a time.

## 6. Open questions for the user

1. **Starter models** (sections 3 and 4.1): none meets the acceptance bar yet. Bundle the
   conservative PvP model now (no wrong FOUND in 60 closed loops, finds (1,1) on half the
   devices), or wait until a retrained one reaches 70%?
2. **Automation tree**: it records and grades what the backend does (4.2b). Should ChargeCell
   also plan the sequence of stages per qubit (PvT → PvP → tie bar), or leave that to the
   backend?
3. **Tie-bar coupling target**: what range of coupling ratio (or t_c in µeV, with lever arms and
   electron temperature) do you want for readout? Without it ChargeCell only reports.
4. **Real data**: when available, a few scans of each kind (any format) from two cooldowns.
5. **Device facts** that are still unknown to ChargeCell: which T gate serves which plunger on
   your devices, and whether interior dots need their own loading procedure.
6. **The interface**: is the four-step Train page and the Home checklist what you had in mind?
   Are there words in the GUI or the glossary that your operators still find unclear?
7. **GPU training**: which computer and GPU will train (NVIDIA with CUDA, or Apple)? Can it reach
   the GUI computer over the network, or should the SSH tunnel / job-file route be the default
   in the instructions?

## 7. How to resume

```bash
git lfs install && git lfs pull
pip install -e ".[dev]"
pytest -q                                              # expect 43 passed
python scripts/nav_trace.py --oracle --bench --kind PvP   # and PvT, tiebar: expect 30/30
chargecell -w /tmp/demo serve --no-browser             # GUI; practice devices on the Scans page
```

Then continue with section 4 in order.
