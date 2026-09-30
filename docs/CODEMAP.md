# Code map

Where things live, what they promise, and the invariants that are easy to break.
The high-level design and rationale are in `DESIGN.md`; this file is the developer's reference.

```
chargecell/
  schema.py            core types and vocabularies (Scan, statuses, PvP reasons, line families)
  kinds.py             scan kinds (PvP, PvT, tiebar): reasons, line families, network heads
  config.py            DeviceConfig (all optional: gate roles, safe limits, steps, priors, calibrations)
  units.py             voltage formatting that follows the size of the value
  storage.py           Workspace: plain-folder persistence (JSON + npz)
  preprocess.py        resampling, network input features, line-mask packing
  quality.py           classical pre-model checks (hard fails, noise, clipping)
  labels.py            annotation format <-> dense labels; status suggestion; model drafts
  simulate/physics.py  triple-dot constant-interaction + tunnel-coupling model, sensor dot
  simulate/generator.py device/window/artefact sampling, rendering, oracle labels (PvP)
  simulate/pvt.py      plunger-vs-tunnel-gate scans: reservoir rate, latching, broadening, labels
  simulate/tiebar.py   (1,1)-(2,0) zooms: windows, triple points, coupling truth, labels
  model/unet.py        ChargeCellNet (multi-head U-Net)
  model/dataset.py     synthetic shard building, real-label arrays, group split, torch Dataset
  model/train.py       TrainConfig, ensemble training, calibration, metrics, model cards
  model/infer.py       Grid (index <-> volts), canonical input, Analyzer (cached ensemble)
  analysis/decide.py   quality gate + ensemble + FOUND gates -> decision, keypoints, overlays
  analysis/lattice.py  transition-line lattice from occupancy / line maps
  analysis/recommend.py next-scan guidance, safety, device history, spectator check (+ shared
                       helpers: finalize_window, widen_if_revisited, common_fix)
  analysis/pvt.py      PvT decision, loading lines, clean tunnel band, operating point, guidance
  analysis/tiebar.py   tie-bar decision, triple points, measured coupling, readout point, guidance
  importers/generic.py load_any(): .json (protocol request or plain)/.npz/.csv dispatch
  protocol.py          chargecell/1 request/response models, analysis -> response, handlers
  client.py            stdlib-only HTTP client and request builder for measurement code
  virtual.py           practice devices (P, X, T gates; any kind); closed-loop evaluation per kind
  jobs.py              background JobManager (one worker thread)
  server/app.py        FastAPI app (create_app) behind the GUI
  server/static/       index.html, style.css, plot.js, app.js (no build step)
  cli.py, __main__.py  command line
tests/                 conftest (OracleAnalyzer), test_core, test_guidance, test_api,
                       test_protocol, test_kinds
scripts/               train_starter, eval_model, nav_trace, gui_check
```

## 1. Conventions that everything depends on

- **Units.** Scans and all public APIs use **volts**. The simulator works internally in
  **mV** (`simulate/*`, `Window.x0` etc.). `virtual.py` converts at the boundary. The GUI
  displays mV; operator text uses `units.fmt_*` (the unit follows the size of the value).
- **No voltage scale.** Nothing may assume a device's voltage scale. `DeviceConfig` voltage
  settings are optional; guidance falls back to spacings measured in the scans (or earlier scans)
  and to fractions of the window (`step_limit`, `barrier_step_for`). Practice devices have a
  random overall scale, so the oracle benchmarks catch accidental absolute constants.
- **Scan kinds** (`kinds.py`): `PvP`, `PvT`, `tiebar`, one model each (active model per kind).
  PvT: x = plunger, y = its tunnel gate (analysis transposes otherwise; index j increases toward
  faster tunnelling). Tie bar: PvP axes, the (1,1)-(2,0) transition with two electrons in dot A.
- **Scan orientation.** `Scan.signal` is `(ny, nx)`; `x` and `y` are ascending (the
  constructor flips axes and signal if needed). `x_gate` is horizontal, "dot A";
  `y_gate` is vertical, "dot B". The spectator is the third plunger.
- **Canonical index space.** The model sees an `S x S` grid (S = model size, 64 or 96) oriented
  so occupancy increases with index i (x) and j (y). `model/infer.py: Grid.for_scan(scan, S,
  carrier)` maps index to volts. For holes `dx, dy < 0` (flipped). **All analysis geometry
  (lattice, targets, windows) is computed in index space and converted with `Grid.to_volts` /
  `to_index` at the end.** This is why hole devices need no special cases downstream.
- **Line families** (PvP: `schema.LINE_FAMILIES`; other kinds: `kinds.py`): `a` (dot A reservoir lines, near-vertical),
  `b` (dot B, near-horizontal), `interdot`, `spectator`, `sensor` (the sensor's own Coulomb
  peaks: artefacts, not charge transitions).
- **Occupancy classes**: 0..4 where 4 means ≥ 4 (`N_OCC_CLASSES = 5`); `OCC_IGNORE = -1`
  marks pixels excluded from the loss (unknown offsets).
- **Statuses**: `FOUND`, `NOT_IN_WINDOW`, `UNINTERPRETABLE` (all kinds). **Reasons** are per
  kind (`kinds.KINDS[k].reasons`) and in a fixed order (the network head depends on it; never
  reorder, only append and retrain). PvP:
  `none | no_transitions, occupancy_too_low, no_reference, partially_visible |
  low_snr, sensor_insensitive, dots_merged, charge_instability, resolution_too_coarse`.
  `NOT_IN_WINDOW_REASONS = REASONS[1:5]`, `UNINTERPRETABLE_REASONS = REASONS[5:]`.
- **Anchoring ("ref")**: a dot is anchored when its empty (n = 0) region is visible over more
  than an occupied cell could span (`ANCHOR_WIDTH` = 1.3 addition voltages), so absolute electron
  numbers are known. FOUND requires both dots anchored (PvP) or the dot anchored (PvT).

## 2. Data contracts

### Scan (`schema.Scan`, dataclass)
`signal, x, y, x_gate, y_gate, id, device, cooldown, kind (PvP | PvT | tiebar), voltage_state (gate -> V),
fast_axis (inner sweep loop), source (file|api|synthetic|virtual_device), created,
units, notes, extra`. `meta()` / `from_meta()` round-trip through `meta.json`.
Practice-device scans keep ground truth in `extra["truth"]` (status, reason, and per kind:
target_V and spectator_occupancy; T_open_V, T_broad_V, operating_point_V; tp_low_V, tp_high_V,
readout_V, length_V, coupling_ratio, tc_meV). The server strips it from `GET /api/scans/{id}` and serves it only via
`/truth`.

### Workspace layout (`storage.Workspace`)
```
devices/<name>.json                  DeviceConfig
scans/<id>/scan.npz                  signal, x, y
scans/<id>/meta.json                 Scan.meta()
scans/<id>/annotation.json           current label
scans/<id>/annotation_history/<stamp>_<annotator>.json   every save (audit trail)
scans/<id>/analysis.json             latest analysis result
synthetic/<name>/manifest.json + shard_###.npz + shard_###.json (per-sample meta)
models/<id>/model.json + member_<k>.pt   (weights of bundled models are in Git LFS)
models/ACTIVE, models/ACTIVE_<kind>  active model id per scan kind (ACTIVE = PvP)
virtual_devices/<id>.json            practice device parameters and state
jobs/<id>.json                       job records
```
`scan_ids()` is sorted descending, so the newest id comes first (ids embed a timestamp).

### Annotation (`labels.empty_annotation`)
```json
{"scan_id", "annotator", "created", "updated",
 "status": "FOUND|NOT_IN_WINDOW|UNINTERPRETABLE|null", "reason": "<REASONS>|null",
 "a_boundaries": [[[x_V, y_V], ...], ...],   // each: a k->k+1 staircase of dot A, drawn bottom->top
 "b_boundaries": [...],                      // dot B, drawn left->right
 "a_offset": 0|1|2|3|null, "b_offset": ...,  // electrons left of / below all boundaries; null = unknown
 "spectator_lines": [...], "sensor_lines": [...],
 "notes", "origin": "manual|model:<id>", "reviewed": bool}
```
`dense_from_annotation(ann, xs, ys)` gives per-pixel `occ_a`, `occ_b` (with OCC_IGNORE where
the offset is unknown), line masks, and `ref_a`/`ref_b`. `relative_occupancy` counts boundaries
crossed. `suggest_status` derives a status and reason from the drawn geometry (edge threshold
0.25). `annotation_from_prediction` turns a model prediction into an editable draft. The
round trip truth -> polylines -> dense agrees on ~99% of pixels.

### Synthetic shards (`model/dataset.py: sample_to_arrays`, `simulate/pvt.pvt_to_arrays`, `simulate/tiebar.tiebar_to_arrays`)
Per sample, at `size x size`: `signal` float16; `occ` int8 (H, S, S), the kind's class heads in
order (PvP: electrons in a, b clipped to 4; PvT: electrons (-1 where electrons cannot follow),
tunnel regime 0 slow / 1 good / 2 open (-1 on rows without a transition); tiebar: region 0 (1,1)
/ 1 (2,0) / 2 (1,0) / 3 (2,1) / 4 other); `lines` uint8 bitmask (one bit per family of the kind,
`preprocess.pack_lines(masks, families)`); `status`, `reason` int8 indices (reason into the
kind's list); `ref` int8 (n_ref,). The manifest records `kind`. Per-sample meta (json) includes status, reason, ref flags, vis_frac,
snr, spectator_occ, target_mV, spacing_mV, pair, window_mV, native_shape, fast_axis, artifact,
geometry.

### Network input and outputs
`preprocess.features(sig)` gives (3, S, S): robust z-scored signal (clipped ±6), then x- and
y-gradients of a σ = 0.8 Gaussian-smoothed copy (scaled by the 99th percentile, clipped ±3).
`ChargeCellNet(kind=...)`: dense head (the kind's class-head logits, then one logit per line
family) and global head (3 status + the kind's reasons + n_ref), built from bottleneck and
decoder pooled features. PvP: 2·5 + 5 dense, 3 + 10 + 2 global (the original layout, so old
PvP weights load unchanged). Heads are defined in `kinds.py`; **never reorder a kind's reasons or
line families** (append, then retrain).

### Prediction dict (`Analyzer.predict`)
`status_p (3,)` ensemble mean; `status_members (M,3)`; `reason_p (10,)`; `ref_p (2,)`;
`<head>_p` per class head (PvP `occ_a_p, occ_b_p (5,S,S)`; PvT `occ_p (5,S,S), regime_p
(3,S,S)`; tiebar `region_p (5,S,S)`); `lines_p (F,S,S)`; `mutual_info` (ensemble disagreement on
status); `occ_entropy`; `signal_canonical (S,S)`. `tests/conftest.py: truth_to_prediction`,
`pvt_truth_to_prediction` and `tiebar_truth_to_prediction` build the same dicts from simulator
truth. Keep them in sync.

### Model card (`models/<id>/model.json`)
`id, created, config (TrainConfig), architecture (+params), found_threshold, calibrated_on
("real" if ≥ 20 held-out real scans else "synthetic"), split_note, data (synthetic names,
n_train, n_real_train, n_val), metrics {synthetic|real: status_accuracy, found_precision,
found_recall, occupancy_pixel_accuracy, confusion, confusion_labels, line_f1, n, ...},
history (per member per epoch), uncertainty_threshold` (90th percentile of validation mutual
information).

### Analysis result (`analysis/decide.analyze`, saved as analysis.json)
`analyze()` dispatches on `scan.kind`. PvT and tiebar results share the top-level keys below
(kind, status, reason, reason_text, demotion, confidence, needs_review, model_id,
found_threshold, probabilities, uncertainty, recommendation, overlays{size, x_gate, y_gate,
extent}) and add `features` (protocol-ready keypoints), `keypoints` (PvT: operating_point,
tunnel_gate_clean_from/_to; tiebar: tp_low, tp_high, tiebar_mid, readout) and, for tiebar,
`coupling`. PvT results also carry `transposed` (the scan had the tunnel gate on x). PvP:
```
scan_id, created, kind, quality, scan{x_gate,y_gate,device},
status, reason, reason_text, ref[2], cell{area_fraction, edge_fraction, centroid_idx,
centroid_v{gate:V}, polygon_v[[x,y]], range_v{gate:[lo,hi]}} | null,
keypoints{cell_centre, readout_20, readout_02 : {gate:V}},
spectators[{gate, voltage, status: verified|unverified|unknown voltage, by_scan}],
demotion[str], confidence, needs_review, model_id, found_threshold,
probabilities{status{}, reason{}, ref[2], members[[3]]},
uncertainty{score, mutual_info, occupancy_entropy},
lattice{a|b: {lines[{c, m, index|null, n}], spacing, slope, source}, size},
lattice_v{spacing{gate: V|null}},           # only spacings accepted as measured
recommendation{...see below}, overlays{size, extent[x_i0,x_iS,y_j0,y_jS], occ_code b64 u8
(a*8+b, 7 = unknown), lines{family: b64 u8 prob*255}}
```

### Recommendation (`analysis/recommend.recommend`)
`kind: found | move | explore | fix_then_rescan`, `headline`, `steps[]`,
`confidence: high|medium|low`, `basis`, `target{gate:V}`,
`next_window{x_gate, y_gate, x:[start,stop,n], y:[...]}` (absolute V; for a split move this is
the first step), `move{gate:ΔV}` (window-centre change, or a barrier change for merged dots),
`physical_moves` (virtual gates), `tiebar_window` (PvP FOUND: take a tie-bar scan there),
`retune_window` (tiebar FOUND with a coupling outside the target; `move` then holds the
exchange-gate change), `warnings[]`,
`spacing_v{gate}`, `spacing_source{gate: "measured in this scan" | "from earlier scans on this
device" | "device prior" | "unknown"}`.

## 3. Modules

### simulate/physics.py
Triple dot, states n ∈ {0..5}³ (`STATES`). `DeviceParams` holds the lever arms (gate -> dot),
U (charging), Um (mutual), tunnel couplings (growing with occupancy via `tgrow`), temperature,
sensor dot parameters, and `v11` (plunger voltages of the (1,1,1) centre). `centre_offsets` puts
the (1,1,1) centre at `v11`. `classical_ground_state` (T = 0, t = 0) is cheap and used for
geometry and oracle labels. `occupations` diagonalises the K_LOW = 6 lowest configurations
(batched `eigh`) with a thermal average; results are cached by parameter bytes.
`sensor_mu/current/slope`: the sensor dot with sech² Coulomb peaks, coupled to dots and gates,
plus compensation residuals.

### simulate/generator.py
`PRESETS` (hrl_linear, hrl_triangle, mixed). `sample_device` samples cross-capacitance
decaying with distance (nearest U(0.25, 0.7) × distance^-power), compensation residual σ in
U(0.01, 0.12), compensation off in 5% of devices. `sample_window(rng, p, intent, pair, coarse)`
places the window to aim for an outcome. `sample_artifacts` / `Artifacts`: jumps, latching,
pink and telegraph noise, gain drift, white noise at a target SNR. `render` produces the
signal, axes, occupancy, and sensor info. `oracle` produces the true status, reason, ref flags,
vis_frac, SNR (charge-step contrast / noise), masks, target, spacing, and spectator occupancy.
UNINTERPRETABLE precedence: merged > coarse (< 5 px per addition) > sensor_insensitive >
low_snr (< 1.5) > weak sensor > charge_instability (≥ 2 big jumps). FOUND needs both refs and
vis_frac ≥ 0.6. `full_cell` gives the complete (1,1) cell geometry. `generate_sample(rng,
preset, mix)` takes about 55 ms and yields ~41% FOUND, 27% NOT, 32% UNINTERP by default.

### model/dataset.py, model/train.py
`build_synthetic` writes shards (multiprocess via `workers`) and a manifest with counts.
`real_arrays` turns labelled scans into arrays (dense labels resampled to S).
`group_split` splits by `device|cooldown`. `CSDDataset` augments with polarity flips and
transposes (a transpose swaps A and B labels). `train`: trains the ensemble members
(OneCycle LR, optional `max_minutes`), then `full_metrics` on validation sets,
`calibrate_found_threshold` (smallest threshold reaching `target_precision`), writes the model
card, and activates the model if none is active. Progress callback: `(fraction, message,
record|None)`.

### model/infer.py
`Grid` (index <-> volts, including holes). `canonical_signal` resamples and orients.
`Analyzer.get(ws, model_id)` loads and caches an ensemble. **Clear `Analyzer._cache` after
training** (the server does). Tests monkeypatch `Analyzer.get` to return the OracleAnalyzer.

### analysis/decide.py
`analyze(ws, scan, model_id=None, cfg=None)`: quality hard fail -> UNINTERPRETABLE without the
model. Otherwise it predicts and applies the FOUND gates: argmax FOUND, p ≥ threshold, both refs,
(1,1) area ≥ 0.5%, edge fraction ≤ 0.4. On failure it demotes to NOT_IN_WINDOW or UNINTERPRETABLE
(whichever of p[1] and p[2] is larger), with reasons restricted to the status. Then it builds
cell geometry, keypoints, the lattice, spectator check, recommendation, and overlays.
`needs_review` is set if the scan was demoted, mutual information exceeds the model's
threshold, or max p < 0.6. `prediction_for_annotation` gives a prediction on the scan's own
grid (for drafts).

### analysis/lattice.py
`extract_lattice(occ_a, occ_b, lines_p, ref_a, ref_b)`. An anchored dot uses the per-row first
crossing of each k -> k+1 in the occupancy map (indexed lines). Otherwise `_from_line_map(p_fam,
p_interdot, S)` joins family pixels through the interdot channel into whole staircases,
labels connected components after one dilation, fits `i = m·j + c` (c taken at the centre row),
clusters by c (< max(3, 0.12·S)), and keeps lines spanning ≥ 0.25·S rows. Dot B uses transposed
arrays. Output per dot: lines, spacing (median gap), slope, source.

### analysis/recommend.py
See DESIGN §7 for the logic. Key points: spacing comes from measured-if-plausible (0.4–2.5x
the prior), else history median, else prior. The anchored target solves the tilted mid-lines of
both dots. Exploration is 0.75 of a window. The history prior (last FOUND on the same gate pair)
beats exploration. The window is 3.2 cells (never shrunk on an unmeasured spacing). Points are
`points_per_addition` per addition voltage, clipped to [min, max]. Large moves are split to
`max_step` (reported window = first step, target = final). Windows are kept inside safe limits.
Reason-specific fixes apply for UNINTERPRETABLE.

### importers
`load_any(path, content=None, meta=None)` accepts bytes for uploads; `meta.axis_units = "mV"`
scales axes to V. A `.json` file with a `scan` object is parsed as a chargecell/1 request.

### protocol.py (chargecell/1, specified in docs/PROTOCOL.md)
Pydantic models `Request` (`ScanIn`, `Axis`, `EncodedArray`, `Label`, `Options`) and
`Response` (`Feature`, `ScanStep`, `Window`). Request models forbid unknown fields; ids,
device and cooldown names are restricted to `[A-Za-z0-9._-]` because they become paths.
`response_from_analysis(analysis, cfg)` maps statuses to outcomes: FOUND -> `found`;
NOT_IN_WINDOW with guidance confidence high/medium -> `next_scan`; everything else ->
`no_confident_step` with a `suggestion`. `handle_analyze` and `handle_submit` are shared by
the HTTP API and the CLI. Only `ANALYSABLE_KINDS` (PvP) are analysed; other kinds are stored,
and `model/dataset.real_arrays` skips them. **Change the protocol additively** (new optional
fields, enum values) or bump to chargecell/2.

### virtual.py
Practice devices use plungers P1, P2, P3 (dots 0, 1, 2), exchange gates X1 (P1-P2) and X2 (P2-P3)
and tunnel gates T1 (dot 0) and T2 (dot 2). `create` scales the whole device by a random factor
(addition voltages from a few mV to a few hundred mV), starts the plungers −2.5…+3 addition
voltages from (1,1,1), the tunnel gates in (``tuned``) or below the clean band, and writes a
DeviceConfig with a ±30% spacing prior and safe limits (both optional, ``prior``/``limits``).
`device_state(vd, voltage_state)` applies the X gates (interdot coupling, exponential in X, and
small shifts) and the T gates (dot shifts, and latching of the edge dots in plunger scans) and
moves `v11` with them. `measure(..., kind)` renders PvP, tiebar (two plungers) or PvT (a plunger
and its tunnel gate, either way round) at SNR U(4, 25), updates the DC point and stores the
truth; `simulate_clean` re-renders it noise-free for the oracle. `evaluate(ws, kind, ...)` runs
the closed loop per kind (`evaluate_navigation` = PvP) and judges each FOUND against the truth
(`_found_is_right`, `pvt_found_is_right`, `tiebar_found_is_right`). Start windows are three
typical spacings (`start_window`); tie-bar runs start near the true tie bar (`_tiebar_start`).

## 4. HTTP API (`server/app.py`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/status` | workspace, active model (PvP), active_models (per kind), counts, devices, running jobs |
| GET | `/api/scans?device=&labelled=yes|no&source=` | scan summaries (newest first) |
| POST | `/api/scans/import` | multipart `files[]` + x_gate, y_gate, device, cooldown, axis_units, notes, kind |
| GET/PATCH/DELETE | `/api/scans/{id}` | scan (signal as base64 float32 LE) / edit meta / delete |
| GET | `/api/scans/{id}/truth` | practice-device ground truth |
| GET/PUT/DELETE | `/api/scans/{id}/annotation` | label (+ preview occ_code on the scan grid and suggestion) |
| POST | `/api/scans/{id}/annotation/preview` | preview without saving |
| POST | `/api/scans/{id}/draft_from_model` | annotation draft from the model |
| GET | `/api/label_queue` | unlabelled PvP scans, most uncertain first |
| POST/GET | `/api/scans/{id}/analyze`, `/analysis` | run / fetch analysis (404 if none) |
| GET | `/api/v1/scans/{id}/response?download=` | saved analysis as a chargecell/1 response |
| POST | `/api/v1/analyze` | chargecell/1 request -> response (422 invalid, 409 no model) |
| POST | `/api/v1/scans` | store a scan (+ optional label) as training data |
| GET | `/api/v1/schema` | JSON Schemas of request and response |
| POST | `/api/analyze_all?only_new=true` | job |
| GET / PUT | `/api/devices`, `/api/devices/{name}` | device configs (validated by pydantic) |
| GET/POST | `/api/synthetic` | list / generate (job; body includes `kind`) |
| GET/DELETE | `/api/synthetic/{name}/preview?n=&offset=`, `/api/synthetic/{name}` | thumbnails / delete |
| GET | `/api/models` | model cards (+kind, +active flag per kind) |
| POST | `/api/models/{id}/activate`, `/api/train` | activate (for its kind) / train (job; body = TrainConfig fields incl. `kind`; datasets must match kind and size) |
| GET/POST | `/api/jobs`, `/api/jobs/{id}`, `/api/jobs/{id}/cancel` | job status / cancel |
| GET/POST | `/api/virtual`, `/api/virtual/{vd}/measure` | practice devices (create takes the first scan of body `kind`; measure takes `kind`) |
| POST | `/api/scans/{id}/run_next?window=next_window|tiebar_window|retune_window` | practice mode: apply gate changes, measure the recommended window (as a tiebar scan for the last two) and analyse |
| GET | `/`, `/static/*`, `/api/docs` | GUI, assets, OpenAPI docs |

## 5. GUI (`server/static`)

No framework and no build step; it works offline. `index.html` has the rail nav and page
sections. `style.css` defines tokens: bg #E9EDF1, ink #17202B, accent #2E5AAC, found #1D7A4C,
move #A86400, fail #A8322A, plus line colours. It uses the system font stack with tabular
numerals.

- `plot.js` (IIFE, exports `window.ChargePlot`): `Plot` class for the heatmap (grey or viridis,
  gradient-magnitude view, percentile contrast, invert), wheel zoom, shift/middle-drag pan (or
  left-drag when `panOnDrag`), axes in mV, cursor readout, and `layers` (functions drawn after
  the heatmap). `drawGridImage(img, extent)` draws an index-grid image at data coordinates,
  handling reversed axes. `hooks: down/drag/up/hover/dblclick/context` are for editing tools.
- `app.js`: helpers (`h()` element builder, `api()` fetch wrapper with error toasts),
  `scanPicker`, `plotBar`, overlay builders (`codeImage`, `codeLabels`, `linesImage`,
  `dashedRect`, `arrow`), and `pages.{review,label,scans,synthetic,train,device}`, each with
  `build()` (once) and `enter(arg)` (on every route). Routing is by hash `#/<page>/<scanId>`.
  `pollStatus` runs every 1.5 s (job indicator, model indicator, refresh on job completion).
  `S.pendingDraft` makes "Correct in labeller" open the Label page with a model draft loaded.
  The annotator name is kept in localStorage (`cc-annotator`). Scan kinds: `KIND_LABEL`,
  `statusLabel(kind, status)`; `drawFeatures` draws PvT/tie-bar keypoints (points are gate -> V,
  polylines are on the analysis axes and swapped when a PvT analysis was transposed); `onAxes`
  maps a recommended window onto the scan's own axes by gate name. The Label page and label queue
  handle PvP scans only; `practiceButtons()` creates a practice device for a chosen kind.
- Labeller: tools V/A/B/S/E; click adds points; double-click or Enter finishes (trailing
  near-duplicate points are dropped); Alt-click inserts; right-click deletes a point; Backspace;
  Ctrl+Z/Shift+Z undo stack of JSON snapshots; live preview is debounced 180 ms. Saving FOUND
  when the geometry does not support it asks for confirmation.

## 6. Tests

- `conftest.py`: `OracleAnalyzer` (perfect predictions from simulator truth; re-renders
  practice-device scans noise-free), fixtures `ws` and `oracle_analyzer` (monkeypatches
  `Analyzer.get`).
- `test_core.py`: simulator, label round trip, unknown offsets, generic importers.
- `test_guidance.py`: decisions and targets with perfect perception, safety limits and step
  splitting, closed-loop navigation, confidence downgrade for stacked indexed lines.
- `test_protocol.py`: request forms and validation, a practice device driven to (1,1) purely
  through `/api/v1/analyze` (via `chargecell.client`), training submission, errors, CLI file
  mode.
- `test_api.py`: the whole GUI workflow over HTTP; training smoke test (48 scans, 32 px,
  1 epoch).
- `test_kinds.py`: PvT and tie-bar simulators and labels, coupling truth grows with t_c, the
  fitted interdot width matches the physics, oracle closed loops for PvT (no priors or limits)
  and tie bar, PvT axes either way round, tiny training runs per kind (per-kind active model),
  and protocol responses with the new feature types.
- Guidance benchmarks outside pytest: `scripts/nav_trace.py --oracle --bench --kind PvP|PvT|tiebar
  [--no-prior] [-v]` (30 devices).
