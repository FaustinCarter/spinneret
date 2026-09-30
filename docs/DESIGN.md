# ChargeCell design

## 1. The problem

ChargeCell follows HRL's tune-up of exchange-only qubits, which uses three kinds of
charge-stability scan, each read by its own model (HRL QPU paper, Supplement S5):

1. **PvT** (plunger vs the tunnel gate to its reservoir): load one electron under an edge
   plunger, with the tunnel gate where electrons load cleanly (section 8).
2. **PvP** (plunger vs plunger): find the (1,1) cell of a pair (this section and sections 3-7).
3. **Tie bar** (PvP zoomed on the (1,1)-(2,0) transition): triple points, a rough interdot
   tunnel coupling, a first spin-to-charge readout point (section 9).

**No voltage scale is assumed.** Voltages vary widely between devices and technologies. The
networks see normalised images; guidance measures spacings from the scans themselves, uses the
operator's settings only when given, and otherwise sizes moves as fractions of the window.

A plunger-vs-plunger charge stability diagram of dots A and B is a honeycomb of cells, each with a
fixed electron number (n_A, n_B). The goal is the (1,1) cell. Two facts shape the design:

- **Counting needs an anchor.** Cells look alike, so (1,1) can only be named with certainty when
  the empty region (n = 0) is visible for both dots. Without it, the best possible answer is "not
  in this window; move toward fewer electrons". ChargeCell treats this as a hard rule. The
  empty region must be wider than any occupied cell (1.3 electron spacings): a narrower line-free
  strip at the edge could be an occupied cell cut off by the window, and the first trained model
  learned to guess there (section 11).
- **Guidance needs geometry, not just a label.** To tell the operator where to go, the tool must
  measure the transition lines (position, spacing, tilt) and extrapolate.

So the model predicts per-pixel electron numbers and line maps (dense, interpretable, checkable),
and deterministic code turns those into a decision and a next scan.

## 2. Pipeline

```
scan file ─► importer ─► quality gate ─► ensemble U-Net ─► consistency checks ─► decision
  (API, .json, .csv, …)   (NaN, flat,      (occupancy A/B,   (threshold, anchors,   │
                           too small)       lines, status,    cell size, edge)      ▼
                                            reason, anchors)                lattice fit ─► guidance
                                                                            (lines, spacing, tilt)   (move, window,
                                                                                                      safety, history)
```

## 3. Synthetic data (`chargecell/simulate`)

ChargeCell generates its own training data; no external simulator is needed.

- **Physics.** Constant-interaction triple dot with a gate lever-arm matrix (cross-capacitance
  decays with distance), charging and mutual energies, and interdot tunnel coupling that grows with
  occupancy (curved interdot transitions, as reported by HRL for strongly coupled SLEDGE devices).
  The ground state is found by diagonalising the few lowest charge configurations; the thermal
  average gives smooth transitions.
- **Sensor.** A sensor dot with Coulomb peaks (sech² line shape) capacitively coupled to all dots
  and gates. Its working point drifts as the plungers sweep, so contrast fades and can invert
  across a large window, just as on real devices. Compensation residuals are sampled.
- **Artefacts.** White noise at a target SNR, 1/f drift and random telegraph noise along the slow
  axis, gain drift, charge jumps between sweeps, and latching (delayed transitions along the fast
  axis at low tunnel rates).
- **Labels come from the physics.** Status and reason are computed from ground truth (is (1,1)
  in the window, are the empty regions visible, what is the charge-step contrast relative to
  noise), not from the generator's intent. The requested outcome mix only steers where windows
  are placed. The noise in "too noisy" (`low_snr`, and the weak-contrast part of
  `sensor_insensitive`) is the noise a step has to stand out from in the image: the
  pixel-to-pixel noise along the sweep (`visible_noise`). Slow sensor drift and rare telegraph
  switches show as streaks between sweeps but leave each step sharp, so they do not make a scan
  unreadable; an earlier definition counted them like white noise and labelled about half of
  its `low_snr` scans unreadable although an expert (and the network) reads them easily.
- **Presets.** Linear and triangular triple-dot geometries with parameter ranges loosely matched to
  published HRL SLEDGE data. Every device is randomised (domain randomisation), so the model sees
  far more variety than one real device offers.

## 4. Labels (`chargecell/labels.py`)

An annotation is a set of polylines, not a painted mask:

- **dot A boundaries** (each k → k+1 transition of dot A, drawn as a staircase through the
  interdot jogs), **dot B boundaries**, optional **spectator** and **sensor artefact** lines;
- **offsets**: the electron number left of all A boundaries and below all B boundaries
  (0 when the empty region is visible, or Unknown);
- **outcome** and **reason**, annotator, notes, second-check flag. Every save is versioned.

From these, dense per-pixel occupancy maps and line masks are derived exactly as for synthetic
data. Unknown offsets make the occupancy loss ignore that dot, so a scan without an empty region
still teaches line detection and outcome classification. Round-trip (truth → polylines → dense
labels) agrees with ground truth on ~99% of pixels.

## 5. Model (`chargecell/model`)

- **Network.** A compact U-Net (about 2M parameters) with a shared encoder and six heads:
  per-pixel occupancy for dot A and dot B (0–4+), line maps for five families (A, B, interdot,
  spectator, sensor), and image-level outcome, reason and anchor (empty region visible for A / B).
- **Input.** The scan is resampled to a fixed square grid in canonical orientation (occupancy
  increasing to the right and up; hole devices are flipped) and robustly normalised.
- **Training.** Synthetic scans plus real labelled scans (weighted), with polarity and transpose
  augmentation (a transpose swaps the roles of A and B, labels included). Real scans are split by
  device and cooldown so held-out scores reflect a new cooldown, not a near-duplicate scan.
- **Ensemble.** Several members trained with different seeds. Their disagreement (mutual
  information) flags scans for review and ranks the labelling queue (active learning).
- **Calibration.** The FOUND threshold is chosen on held-out data as the lowest threshold at which
  FOUND precision reaches the target (default 97%). Below it, the tool says "not in window" or
  "can't interpret" and asks for review. A missed FOUND costs one extra scan; a wrong FOUND can
  cost an experiment.

**Other kinds.** PvT and tie-bar models use the same network with their own heads, defined in
`chargecell/kinds.py` (PvT: electrons, tunnel regime, three line families, one anchor flag;
tie bar: a five-class region map and five line families). Each kind has its own model versions,
calibrated threshold and active model.

**Why no foundation model.** Pretrained vision models (e.g. SAM, large pretrained backbones or
vision-language models) were considered. Charge stability diagrams are small, single-channel,
and governed by precise geometry; the hard part is counting from an anchor and extrapolating the
lattice, which none of those models do. A small network trained on physics-based simulations runs
in milliseconds on a CPU, can be retrained on site, and its outputs can be checked against
physics. A large pretrained model would add weight and opacity without a clear gain here.

## 6. Decision (`chargecell/analysis/decide.py`)

FOUND requires **all** of:

1. outcome head's top class is FOUND, with probability ≥ the calibrated threshold;
2. the anchor heads say the empty region is visible for both dots;
3. the occupancy maps contain a (1,1) region larger than 0.5% of the window;
4. at most 40% of that region's border is the window edge (not cut off);
5. the empty region is at least 1.3 electron spacings wide for both dots, measured on the
   occupancy map against the 0→1 / 1→2 spacing in the same map (a geometric check that does not
   trust the anchor heads);
6. the network's own line map agrees that the empty region is empty: no column (for dot A; row
   for dot B) of its interior, away from the edge, has a mean line probability of that dot's
   family above 0.22. A faint first transition that the occupancy map missed would otherwise
   make every count one too high.

Otherwise the result is demoted and flagged for review. A demoted scan becomes "can't
interpret" only if the network gives that more than 50%: FOUND and "not in window" both mean
"readable", so a confident-but-unverified FOUND becomes "not in window" with the reason taken
from the failed check. `held_back_by` records why: `checks` (one of checks 2-6 failed) or
`confidence` (only the threshold did; the guidance then asks for a confirmation scan, §7). Hard
quality failures (non-finite data, constant signal, tiny scans) skip the model.

The threshold is calibrated on held-out synthetic scans for the **final** decision: the smallest
threshold at which FOUND calls that also pass checks 2-6 reach the target precision (0.97).
Predictions are averaged over test-time augmentations (the polarity flip and, for PvP, the axis
swap), both at calibration and in analysis; the model card records `tta` so older models keep
the plain ensemble their threshold was calibrated with. PvT and tie-bar scans use the same
scheme with their own checks (`found_gates(kind)`).

## 7. Guidance (`chargecell/analysis/recommend.py`)

Everything is computed in the model's index grid and converted to volts at the end.

- **Lattice.** When a dot is anchored, its boundaries are traced from the occupancy map, each with
  a known electron index. Otherwise they come from the line maps: reservoir segments are joined
  through interdot segments into whole staircases before fitting (fitting segments separately
  counts each jog as a line and halves the apparent spacing). Each boundary is fitted as a tilted
  line; spacing is the median gap.
- **Target.** With both dots anchored, the (1,1) centre is the intersection of the mid-lines of
  the A and B (1)-bands, solved with both tilts. With one dot anchored, that dot's target is used and
  the other dot explores. With neither, the tool explores: toward fewer electrons if lines but no
  empty region are visible, toward more if no transitions are visible, by 3/4 of a window so scans
  overlap.
- **Device memory.** The last FOUND position for the same gate pair on the device beats blind
  exploration (with a warning that other gates may have moved since).
- **Spacing.** Measured in this scan, else the median from earlier scans on the device, else the
  optional configured prior. Measurements more than 2.5x away from the prior (or, without one,
  from the median of at least two earlier measurements) are rejected with a warning. With only
  one transition per dot in view and no known spacing, the next window keeps that transition in
  view and is 1.5x wider, rather than guessing a voltage. Target windows are about 3.4 cells,
  shifted toward the empty region so that more than a full cell of it is in view, and are never
  shrunk on an unmeasured spacing. Points are set to about 12 per addition voltage.
- **Retune after large moves.** A move of more than one electron spacing (or 30% of the window
  without a known spacing) changes the electron numbers, which shifts the charge sensor along its
  Coulomb peak; the far dot's lines fade first, and a missed faint first line shifts every
  count. Such moves ask for a sensor retune at the new window centre (`retune_sensor`, also
  in the protocol), as HRL's tune-up retunes its sensor dot between steps.
- **Confirm once.** A FOUND held back only by the confidence threshold (every geometric check
  passed) is most often the right window scanned a little too noisily. The guidance asks for one
  rescan of the same window with 4x longer averaging (`kind` `confirm`; protocol purpose
  `confirm` with `averaging` 4), unless the same window was already scanned without success in
  the last three scans of the pair. Then the rules below apply. The same holds for PvT and tie
  bar.
- **No circles.** A window that was already scanned without success is not proposed again: the
  next scan covers both it and the present window. If that is the present window itself (the
  guidance would rescan what it just saw, typically after a FOUND was held back), the next
  window is 1.5x wider around the expected cell instead, and the headline says the call was
  held back. A window that could not be interpreted twice is widened 1.5x. A large line-free
  region below the lowest transition is treated as the probable empty region and kept in view.
- **Safety.** Windows are kept inside the device's safe limits, when set (a warning says when
  they are not). Moves larger than the step limit (the device's max step, else one window width)
  are cut to the limit, and the operator is told to rescan and re-analyse after each step.
- **Fixes.** For "can't interpret", reason-specific instructions (sensor retune, barrier
  reduction by a configurable step, longer averaging, at least twice the points).
- **Spectators.** A pairwise scan cannot show the third dot's occupancy. ChargeCell checks
  whether an earlier FOUND scan that swept the spectator's plunger contains its present voltage,
  and says so. This ignores cross-talk from gates moved since, so it is a consistency check only.

## 8. PvT scans (`simulate/pvt.py`, `analysis/pvt.py`)

- **Physics.** The tunnel gate T shifts the dots through its own lever arms (strongest on the
  edge dot), so loading lines are tilted: dP/dT = -(T lever / P lever), 0.2-0.7. The reservoir
  rate is exponential in T: log10(Γτ) = (V_T - T_open)/β10 - γ(V_P - P_ref)/β10, with τ the time
  per pixel. Below T_open electrons cannot follow the sweep: each pixel relaxes with probability
  1 - exp(-Γτ) toward a sample of the equilibrium, and a sweep line starts where the previous one
  ended, so transitions latch, move along the sweep and vanish. log10(kTτ/ħ) = 4.5-7 decades
  above T_open, ħΓ exceeds kT and lifetime broadening smears the lines.
- **Labels.** Electrons in the swept dot (ignored where electrons cannot follow), a per-pixel
  tunnel regime (slow / good / open, on rows that contain a transition), loading, spectator and
  sensor lines, one anchoring flag. Reasons: `no_transitions`, `occupancy_too_low`,
  `no_reference`, `tunnel_rate_too_low`, `reservoir_too_open`, and the usual uninterpretable ones.
  If more than 60% of the window is too slow (or too open), the tunnel gate is fixed first.
- **Decision.** FOUND needs the calibrated threshold, the empty dot, and the 0→1 and 1→2 lines
  traced over enough rows where electrons load cleanly.
- **Keypoints.** Loading lines (fitted per electron on clean rows), the tunnel-gate range where
  electrons load cleanly, and the one-electron operating point: 35% into that range, midway
  between the first two lines.
- **Guidance.** Too slow / too open: centre the tunnel gate on the clean band if one is visible,
  else step by 0.6 / 0.5 of the window. No empty dot: move the plunger so the lowest visible line
  sits 60% into the window. First line only: put it 25% into a 1.25x wider window. Nothing
  visible: explore toward more electrons and a more open tunnel gate. Plunger moves follow the
  measured tilt of the lines when the tunnel gate moves.

## 9. Tie-bar scans (`simulate/tiebar.py`, `analysis/tiebar.py`)

- **Physics.** The PvP simulator, zoomed on the (1,1)-(2,0) transition (dot A on x holds two),
  with interdot coupling from 2 µeV (thermally limited) to 0.3 of the mutual charging energy,
  the sensor retuned for the zoom and positioned to see charge move between the dots. Triple
  points are located on a fine ground-state map (the spectator may change occupancy at a triple
  point when mutual charging is strong).
- **Labels.** A per-pixel region map (the four cells around the tie bar: (1,1), (2,0), (1,0),
  (2,1), and "other") and a, b, tiebar, interdot and sensor lines. Reasons: `no_tiebar`,
  `partially_visible`, and uninterpretable ones including `dots_merged` (coupling ratio above
  0.6: the triple points are no longer distinct) and `low_snr` when the interdot step itself is
  too faint.
- **Decision.** FOUND needs the calibrated threshold, the (1,1)|(2,0) boundary, and both triple
  points found where three regions meet, inside the window.
- **Coupling is measured, not predicted.** The raw signal is sampled across the tie bar on its
  central part and a step s(t) = a + b t + c tanh((t - t0)/w) is fitted; FWHM = 1.763 w. The
  coupling ratio FWHM / tie-bar length is voltage-free (about 3 t_c / E_m). With lever arms and
  electron temperature in the device settings, t_c ≈ sqrt(FWHM_E² - (3.53 kT)²) / 3.07 is also
  given in µeV and GHz. Tested against the physics: within 35% of the true FWHM when resolved.
- **Guidance.** Readout point: the tie-bar midpoint moved into (2,0) by max(1.5 FWHM, 0.25
  length), at most 0.75 length. With a coupling target set, weak/strong coupling proposes an
  exchange-gate step and a rescan. Tie bar cut off: follow it toward the missing triple point.
  No tie bar: return to the (1,1)-(2,0) boundary of the last PvP FOUND on the pair, else zoom out.

## 10. Evaluation

- **Unit and integration tests** (`pytest`): simulator consistency, label round trip, file
  importers, the chargecell/1 protocol (parsing, HTTP, file mode, a practice device driven to
  (1,1) through the protocol alone), the full HTTP workflow the GUI uses, and a training smoke
  test.
- **Guidance with perfect perception** (`tests/test_guidance.py`, `tests/test_kinds.py`,
  `scripts/nav_trace.py --oracle --bench --kind ...`, ground truth in place of the network) on
  practice devices with random voltage scales, 30 devices per kind: PvP 30/30 (median 3 scans,
  also 30/30 with no priors or limits), PvT 30/30 (median 3, half the devices start with the
  tunnel gate too closed), tie bar 30/30 (median 1); every FOUND matches the ground truth.
- **Practice devices behave like a tuned-up operator's device**: the sensor is tuned at the
  starting voltages and is sensitive to the P1-P2 interdot step (as a readout sensor must be);
  the simulated operator follows every piece of advice, including the fixes (retune the sensor,
  average longer, change an exchange gate) and the sensor retune before a tie-bar zoom. Each
  closed loop is recorded as a run of the automation tree, which is how the failures of earlier
  model versions were traced (HANDOFF section 3).
- **Starter models**: see section 11.

## 11. Starter models

(Pending: filled in once the starter models are trained and evaluated.)

## 12. Automation tree (`chargecell/runs.py`)

HRL records every action of an automated tune-up as a node of a tree, grades each node when it
happens, and analyses the trees of many runs for success rates and failure modes (QPU paper,
S5.3). ChargeCell keeps the same record, within its advisory role: it cannot see what the
backend does unless told, so the tree holds what ChargeCell was sent, what it answered, and
what the backend or operator reports (the protocol's run events).

- **Shape.** run → stage (one goal: a kind of scan on a pair of gates) → measure (a scan) →
  analysis + advice (+ review, a person's label), with action and note nodes for what the
  backend reports. Children are kept in time order, so a depth-first walk is the order of
  actions. HRL inserts data collection as children of the action that asked for it; ChargeCell
  keeps a stage's scans as siblings, which reads more easily and carries the same order.
- **Grouping.** A backend groups a tune-up with `options.run_id` (or starts a run explicitly).
  Without it, scans join their device's open run, and a new run starts after 4 hours without
  scans on that device. A scan starts a new stage when its kind or gate pair changes, or when the
  backend names a new stage; returning to an earlier goal later is a new stage.
- **Grades at the time.** Scans: data quality. Analyses: pass, check (the model asked for
  review), fail (can't interpret; on practice devices also a FOUND the truth shows is wrong).
  Advice: pass for found or a confident next scan, check for "no confident step". Stages and runs
  are graded from their children: reached, stalled (3 scans in a row without a confident step),
  left without reaching the goal, or wrong.
- **Following the advice.** Each scan records whether its window matches the last advice (both
  ends within 10% of the advised span on both axes) and which other gates changed since the last
  scan. This separates "the guidance failed" from "the guidance was not followed".
- **Statistics across runs.** Per kind: goals reached, median and 90th-percentile scans to the
  goal, stalled and unfinished goals, wrong FOUNDs on practice devices; the most common failed or
  flagged analyses; the last analysis of goals that were not reached (where runs stall); advice
  followed; reviewers' agreement.
- **Practice devices** record their closed loops as runs, including the fixes the simulated
  operator applies (gate changes, sensor retune, longer averaging), so the evaluation of a model
  can be read node by node. Re-analysing a scan with the same model and the same result adds
  nothing to the tree.

## 13. Limitations and next steps

- The starter models have only seen simulations. Label real scans from at least two cooldowns and
  retrain before trusting them on a new device.
- PvT and tie-bar labels are drawn as boundaries (loading lines and the clean tunnel-gate range;
  dot A's 1 -> 2 and dot B's 0 -> 1 boundaries around the tie bar) and converted to the dense
  heads of their models; they have only been checked against simulator truth so far.
- The labelling tool assumes electron orientation (occupancy grows with voltage). Analysis supports
  hole devices; labelling them needs a flipped drawing convention.
- The chargecell/1 protocol has only been exercised by the tests and practice devices, not yet by
  a real measurement backend.
- Spectator verification is a consistency check against earlier scans, not a measurement.
- The automation tree only knows what it is told: a backend that does not report its own
  actions (gate changes between scans are inferred from `voltage_state`) leaves gaps.
