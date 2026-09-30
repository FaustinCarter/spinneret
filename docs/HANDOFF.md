# Handoff: state of ChargeCell at the end of the first build

Written 2026-09-29 by the Claude session that built ChargeCell (claude.ai, sandboxed Linux with one
CPU and 3 GB RAM). It is for the next Claude session working in Claude Code without the original
conversation. Read this first, then `docs/CODEMAP.md` before changing code.

## 1. What the user asked for

The user (works with Si/SiGe exchange-only (EO) spin-qubit devices in HRL Laboratories' style)
asked for, in order:

1. A system that looks at a double-quantum-dot charge stability diagram and either **detects
   the (1,1) charge cell**, or says **(1,1) is not in the data window**, or says **the window is not
   interpretable**.
2. A reading of HRL's exchange-only qubit papers (summarised in `docs/RESEARCH_NOTES.md`).
3. A full build. The request in the user's words, condensed:
   - use spinQICK "as suggested" to generate synthetic training data;
   - an easy annotation pipeline so human experts can label real data;
   - a training pipeline;
   - state-of-the-art open-source AI models only where they add clear, obvious value;
   - a GUI usable by someone who is not a software engineer (label data, generate synthetic data,
     run the pipeline, review outputs, etc.);
   - feedback that tells the operator how to move the voltage knobs and rescan when the
     interesting region is not in the window;
   - anything else an expert would consider obvious.
4. Package everything for a GitHub repo and Claude Code, with instructions for the next session
   (this document).

**Correction already given to the user:** spinQICK (github.com/HRL-Laboratories/spinqick) has no
device simulator. It was cloned and checked; its only "simulation" is MESA forecasting. The
earlier research summary had said spinQICK could be used "to generate real training data in
HRL-style formats", meaning real measurements, and the user read that as synthetic generation.
ChargeCell therefore uses spinQICK for **real-data import** (netCDF) and **next-scan export**
(a script calling its API). Synthetic data comes from ChargeCell's own simulator. Keep this
framing consistent in docs and replies.

**Foundation models were deliberately not used** (no SAM, pretrained backbones, or VLMs). The
reasoning is in `docs/DESIGN.md` §5 and `docs/DECISIONS.md`. Don't add one without a measured gain.

## 2. Status summary

| Area | State |
|---|---|
| Physics simulator + oracle labels | Done, tested |
| Annotation format, dense labels, round trip | Done, tested (~99% pixel agreement) |
| U-Net ensemble, training, calibration, model cards | Done; smoke-tested; **no finished starter model** |
| Inference, decision gates, lattice, guidance | Done; guidance validated with oracle perception |
| spinQICK importer / generic importers / exporter | Done; tested on mock files only |
| Practice (virtual) devices, navigation evaluation | Done |
| FastAPI server (all GUI endpoints), job runner | Done, API-tested |
| GUI (6 pages) | Done; browser-tested except the pages that need a model (see §4.2) |
| CLI | Done (`serve`, `simulate`, `train`, `analyze`, `navigate`) |
| Tests | 13 passing (`pytest -q`, ~40 s on 1 CPU) |
| Docs | README, OPERATOR_GUIDE, DESIGN, CODEMAP, RESEARCH_NOTES, DECISIONS, this file |
| Starter model bundled in the package | **Not done** (§4.1) |
| CI workflow | Written, **never run** (§4.6) |

## 3. Verified results (reproducible)

- `pytest -q`: 13 passed. Covers simulator consistency, label round trip, unknown-offset
  handling, spinQICK netCDF import (analysed and raw-IQ), spinQICK upload through the HTTP import endpoint, generic importers (matrix CSV in mV,
  long CSV, npz, json), spinQICK script generation (step splitting, compiles), the full HTTP
  workflow the GUI uses, and a tiny end-to-end training run.
- Guidance with perfect perception (`python scripts/nav_trace.py --oracle --bench`): 30/30
  random practice devices reach the correct (1,1) cell, median 2.5 scans, 90th percentile 4.
- `tests/test_guidance.py` thresholds: status agreement > 90%; anchored target error median
  < 0.3 cell spacings; correct move direction > 80% when unanchored; navigation ≥ 5/6.
- GUI: `python scripts/gui_check.py` against a running server. It screenshots all pages, drives
  the labeller with real mouse and keyboard input, and checks the annotation round trip. Last
  run: OK, no console errors.
- Starter model, partial (the run was killed by the sandbox before finishing). Settings: 4000
  synthetic scans at 64 px, base 16, 8 epochs, about 108 s/epoch on 1 CPU. Member 0 validation
  status accuracy by epoch: 0.55, 0.52, 0.63, –, 0.67, 0.69, 0.72, 0.72. Per-pixel occupancy
  accuracy rose from 0.63 to 0.77. Member 1 reached 0.65 at epoch 4 when killed. **These are
  modest numbers.** The FOUND threshold calibration and review flags exist because of this.

## 4. Unfinished work, in priority order

### 4.1 P0: Train, evaluate, and bundle the starter model

`cli.py serve` installs any model found in `chargecell/assets/models/<id>/` into an empty
workspace (`_install_bundled_model`). Nothing is there yet, so a new user must train first.
The README and OPERATOR_GUIDE were written as if a starter model ships; see step 5.

1. `python scripts/train_starter.py --workers <cores-1> --bundle`. Consider `--size 96
   --epochs 12 --ensemble 3` if hardware allows; the defaults reproduce the interrupted run.
   Run it in the foreground or under `nohup`. In the claude.ai sandbox, background processes
   were killed whenever the conversation paused, twice. Check how Claude Code behaves before
   relying on long background jobs.
2. `python scripts/eval_model.py --model-dir starter_ws/models/<id>`. It reports held-out
   decision accuracy, FOUND precision and recall, the review rate, and closed-loop navigation
   with the real model.
3. Acceptance bar (proposed; confirm with the user): FOUND precision ≥ 0.95 on held-out
   synthetic data (the calibration targets 0.97); navigation finds (1,1) on ≥ 70% of devices
   with no wrong FOUND. If precision fails, raise `target_precision`, train longer, or use 96 px.
   If navigation stalls, trace it with `scripts/nav_trace.py --model-dir ...`; perception errors
   in the anchor heads (`ref`) and noisy occupancy maps are the likely causes (see the first
   item in §5, which should be fixed before judging the navigation numbers).
4. Fill `docs/DESIGN.md` §9 (placeholder text `STARTER_MODEL_RESULTS`) and the "What is
   validated" section of `README.md` with the measured numbers.
5. Check the README's claims ("A starter model trained on simulated data is installed on first
   start"; quick-start step 1 analyses without training) against reality after bundling.
6. Model files are about 8 MB per member at base 16. Fine for git at this size. See open
   question Q5 about LFS or Releases.

### 4.2 P0: Visually verify the GUI with a model active

Never seen in a browser because no model existed:

- the Review page with an analysis: occupancy overlay, (n,m) labels, lines layer, FOUND polygon,
  centre cross, readout diamond, dashed next-window box, arrow, target ring, decision panel,
  "Model details";
- the Train page with a running job (progress, loss chart) and model cards (metrics,
  confusion matrix);
- Synthetic data previews (thumbnails);
- "Measure it on the practice device" chaining to the next scan;
- "Accept as label" and "Correct in labeller".

After 4.1: start the server on a demo workspace, run `scripts/gui_check.py`, open the PNGs,
then click through the practice loop manually (Playwright scripts are fine). Things to check:
overlays align with the data for hole devices (`drawGridImage` with a negative dx), the arrow
direction, and that the view zooms out to include the next window.

### 4.3 P1: Validate spinQICK integration on real files and the installed version

- The importer follows `SpinqickData.save_data` as read from spinQICK's source (details in
  RESEARCH_NOTES §3). It was only tested on files written by `write_mock_spinqick_nc`. Ask the
  user for a few real `.nc` files (gvg_dc and gvg_baseband) and their spinQICK version.
- The generated script uses `te.vdc.get_dc_voltage`, `te.vdc.set_dc_voltage_compensate(volts,
  gates, iso_gates)` (absolute volts; requires a cross-coupling matrix in the hardware config)
  and `te.gvg_dc(g_gates=..., g_range=..., measure_buffer=..., compensate=...)` (ranges relative
  to the DC point). Verify these signatures against the user's installed spinQICK.
- `MEASURE_BUFFER_US` is left as a placeholder the user must fill. Consider reading it from the
  scan's saved `cfg` attribute instead (stored as JSON in the netCDF root).

### 4.4 P1: Model quality on real data

Real performance is unknown until the user labels scans from ≥ 2 cooldowns. Real metrics appear
in model cards only when a held-out real split exists (`group_split` by device|cooldown). When
labels arrive:

- retrain with `use_real` on and compare real-data metrics between versions;
- look at `needs_review` rates and the label queue ordering;
- if simulations look unlike the user's data, tune `simulate/generator.py` PRESETS and artefact
  ranges against real scans (line widths, sensor drift, noise spectra).

### 4.5 P2: Extensions discussed but not built

- **Spectator verification as a status.** The research notes proposed
  `FOUND_SPECTATOR_UNVERIFIED`. Today spectators are only reported as advice in `spectator_check`.
  A cross-pair consistency layer (P1–P2, P2–P3, P1–P3 scans) would make (1,1,1) a real verdict.
- **PvT (plunger-vs-tunnel-gate) and "tiebar" models**, as in HRL's own pipeline (three CNNs:
  PvT, PvP, tiebar). Reuse the labelling, training, and job infrastructure; add a scan `kind`
  and per-kind heads or models.
- **Keypoint-graph output and an automation tree** (HRL-style action log with a grade per
  node) so tune-up runs can be audited.
- **Hole-device labelling.** Analysis handles holes (`Grid.for_scan` flips), but the labeller
  assumes occupancy grows with voltage. `prediction_for_annotation` flips indices for holes, so
  drafts are consistent. Manual drawing conventions and `labels.relative_occupancy` need a
  carrier-aware direction.
- **Simulator realism.** Consider comparing against or adding QDarts / qarray backends.
  Curved interdot lines at strong coupling are modelled via tunnel coupling. Reservoir-starved
  interior dots (slow or missing reservoir lines) are not modelled.

### 4.6 P3: Housekeeping

- `.github/workflows/tests.yml` was written but never run. Fix it on the first push if it fails
  (torch CPU wheel index, Python version).
- No LICENSE file (open question Q1).
- `httpx` deprecation warning from Starlette's TestClient; a netCDF4/numpy binary warning in the
  sandbox (harmless).

## 5. Known issues and sharp edges

- **Guidance confidence ignores lattice plausibility (open, P1).** Found in the final check with
  a deliberately untrained model (`train_starter.py -n 48 --size 32 --epochs 1 --ensemble 1
  --base 8`, then `nav_trace.py --model-dir ...`). Both dots were "anchored", but the noisy
  occupancy map yielded four stacked indexed lines within about 3 px
  (`a=[15.9, 17.8, 17.8, 19.3]`), and the recommendation still said `confidence=high`, basis
  "lattice fitted to anchored transitions". Indexed lines from `_from_occupancy` are not
  clustered or checked. Proposed fix in `recommend.py`: when both dots are anchored but the
  indexed-line spacing fails `plausible()` (or lines are closer than ~0.4 of the prior spacing),
  downgrade to `medium`, compute the target from the prior spacing, and add a warning. Then add
  an oracle-style unit test that injects a corrupted occupancy map. The FOUND gates were not
  affected (no false FOUND).

- **Line-map lattice and interdot jogs.** Fitting reservoir segments separately counted each
  interdot jog as a line, so the spacing came out 3–6x too small and navigation looped with
  shrinking windows. Fixed by joining segments through the interdot channel
  (`lattice._from_line_map`), by rejecting spacings more than 2.5x away from the device prior,
  and by never shrinking a window on an unmeasured spacing. Don't undo these three together.
- **History pollution.** `lattice_v.spacing` is stored in analyses only when the guidance
  accepted it as "measured in this scan". `history_spacing` takes the median of those.
- **Oracle tests re-render without noise.** `OracleAnalyzer` can therefore call FOUND on a scan
  whose noisy ground truth says `low_snr`. Navigation correctness is judged geometrically
  (`virtual._found_is_right`), not by comparing status strings.
- `evaluate_navigation` writes practice devices and device configs into the workspace you pass.
  Use a scratch workspace (the scripts do).
- Server: `GET /api/scans/{id}/analysis` returns 404 when not analysed. The Review page avoids
  the call by checking `scan_summary.analysis` first.
- Front end: `plot.js` must stay wrapped in an IIFE. Classic scripts share one global lexical
  scope, so `class Plot` collided with `const { Plot } = window.ChargePlot` in app.js.
- CSS: `[hidden] { display: none !important }` is required because `.stack`, `.grid-2`, etc.
  set `display` and would otherwise override the `hidden` attribute.
- `JobManager` runs one job at a time on a thread. Synthetic generation can still use worker
  processes inside that job (`workers`).
- Decision on hard quality failures uses a fixed S = 64 grid for guidance (no model involved).
- `quality.py` once had a sweep-jump detector; it was removed because it flagged clean scans.
  Charge jumps are left to the model's `charge_instability` reason.

## 6. Open questions for the user

Ask these rather than guessing:

1. **License**: which one, if any? None was added.
2. **Real data**: can they share a few spinQICK `.nc` files (with and without (1,1)) and their
   spinQICK version? Are their scans PvP via `gvg_dc`, or also `gvg_baseband`?
3. **Device facts**: gate names (P/X/T/M convention?), safe voltage limits, typical addition
   voltages, electrons or holes, which gate sits between each plunger pair.
4. **Compute**: where will training run (CPU cores, GPU)? That decides 64 vs 96 px and the
   ensemble size.
5. **Model weights in git**: commit bundled weights (~8–25 MB), use Git LFS, or attach them to
   GitHub Releases?
6. **Autonomy**: should ChargeCell stay advisory (operator runs the exported script), or
   eventually run scans in a closed loop through spinQICK? This affects safety design.
7. **Priorities** among spectator verification, PvT/tiebar models, and hole-device labelling.
8. **Acceptance criteria** for the starter model (§4.1 step 3 is a proposal).

## 7. How to resume

```bash
pip install -e ".[dev,spinqick]"          # torch CPU wheel is fine
pytest -q                                   # expect 13 passed
python scripts/nav_trace.py --oracle --bench   # expect 30/30, median ~2.5
python scripts/train_starter.py --bundle    # §4.1
```

Then fix the first item in §5 (small, testable), and work through §4 in order. Update this file as items close. Once the starter model is
bundled and verified, move finished items to a short "Done" list and keep §5 and §6 current.
