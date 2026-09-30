# ChargeCell design

## 1. The problem

A plunger-vs-plunger charge stability diagram of dots A and B is a honeycomb of cells, each with a
fixed electron number (n_A, n_B). The goal is the (1,1) cell. Two facts shape the design:

- **Counting needs an anchor.** Cells look alike, so (1,1) can only be named with certainty when
  the empty region (n = 0) is visible for both dots. Without it, the best possible answer is "not
  in this window; move toward fewer electrons". ChargeCell treats this as a hard rule.
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
  are placed.
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
4. at most 40% of that region's border is the window edge (not cut off).

Otherwise the result is demoted to "not in window" or "can't interpret" and flagged for review.
Hard quality failures (non-finite data, constant signal, tiny scans) skip the model.

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
  configured prior. Measurements more than 2.5x away from the prior are rejected with a warning.
  The window is sized to about 3.2 cells (so the empty region and the (2,0)/(0,2) neighbours are in
  view) and never shrunk on an unmeasured spacing. Points are set to about 12 per addition voltage.
- **Safety.** Windows are kept inside the device's safe limits. Moves larger than the step limit
  are cut to the limit, and the operator is told to rescan and re-analyse after each step.
- **Fixes.** For "can't interpret", reason-specific instructions (sensor retune, barrier
  reduction by a configurable step, longer averaging, more points).
- **Spectators.** A pairwise scan cannot show the third dot's occupancy. ChargeCell checks
  whether an earlier FOUND scan that swept the spectator's plunger contains its present voltage,
  and says so. This ignores cross-talk from gates moved since, so it is a consistency check only.

## 8. Evaluation

- **Unit and integration tests** (`pytest`): simulator consistency, label round trip, file
  importers, the chargecell/1 protocol (parsing, HTTP, file mode, a practice device driven to
  (1,1) through the protocol alone), the full HTTP workflow the GUI uses, and a training smoke
  test.
- **Guidance with perfect perception** (`tests/test_guidance.py`, using ground truth in place of
  the network): >90% outcome agreement, anchored targets within 0.3 cell spacings (median),
  >80% correct move direction when unanchored, and closed-loop navigation on 30 random devices:
  30/30 reach the correct (1,1) cell, median 2.5 scans, 90th percentile 4.
- **Starter model**: see section 9.

## 9. Starter model

Not finished yet (see `docs/HANDOFF.md` §4.1). The first training run (4000 simulated scans at
64x64, base 16, 2 members x 8 epochs, about 108 s per epoch on one CPU core) was interrupted
during the second member. The first member reached 0.72 validation outcome accuracy and 0.77
per-pixel occupancy accuracy after 8 epochs, still improving. `scripts/train_starter.py`
reproduces the run and `scripts/eval_model.py` measures the full decision pipeline and closed-loop
navigation. Record the results here.

## 10. Limitations and next steps

- The starter model has only seen simulations. Label real scans from at least two cooldowns and
  retrain before trusting it on a new device.
- Only plunger-vs-plunger scans are modelled. HRL's tune-up also uses plunger-vs-barrier
  (tunnel-rate) and readout "tiebar" scans; each would get its own head or model, following the
  same labelling and training pipeline.
- The labelling tool assumes electron orientation (occupancy grows with voltage). Analysis supports
  hole devices; labelling them needs a flipped drawing convention.
- The chargecell/1 protocol has only been exercised by the tests and practice devices, not yet by
  a real measurement backend.
- Spectator verification is a consistency check against earlier scans, not a measurement.
