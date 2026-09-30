# Decision log

Each entry: the decision, why, and what would change it. Newest last.

1. **Three outcomes plus a reason code, not a bare classifier.** Operators need to know what to
   do next, so every non-FOUND result names a cause from a fixed vocabulary
   (`schema.REASONS`) and gets matching guidance. *Revisit if* users need finer-grained reasons;
   append to REASONS (never reorder) and retrain.

2. **FOUND requires both dots anchored (empty region visible).** Cells look alike, so
   identifying (1,1) is a counting problem. A (2,2) that looks like (1,1) is the costliest error.
   *Revisit if* the user has an independent anchor (e.g. PvT scans establishing occupancy),
   which could be passed in as known offsets.

3. **Precision-first calibration.** The FOUND threshold is the lowest reaching 97% precision on
   held-out data; everything below is demoted and flagged. A missed FOUND costs one scan.

4. **Dense per-pixel outputs plus deterministic geometry**, instead of end-to-end regression of
   "where to go". Outputs can be checked against physics (lattice fits, cell size, edge contact),
   and guidance logic can be tested on its own with ground truth (OracleAnalyzer).

5. **Own physics simulator; a backend-neutral protocol for real I/O.** spinQICK was first
   planned for synthetic data, but it has no simulator. It was then used for netCDF import and a
   generated next-scan script, and was later removed at the user's request (2026-09-30) in favour
   of the chargecell/1 JSON protocol (`docs/PROTOCOL.md`), which any control stack can speak.
   ChargeCell stays free of instrument code. The simulator models what the HRL papers show matters: strong tunnel coupling, a sensor dot that drifts off
   its flank, spectator lines, and 1/f and telegraph noise.

6. **No foundation models.** SAM, large pretrained backbones, and VLMs were considered. The
   hard parts (anchored counting, lattice extrapolation) are not what they do; a 2M-parameter
   U-Net runs in milliseconds on CPU and retrains on site. *Revisit if* a pretrained
   encoder measurably improves real-data metrics at equal compute. That would be the only
   acceptable evidence.

7. **Polyline annotation (boundaries + offsets), not painted masks.** It is fast for experts,
   exact, matches how physicists read the diagrams, and "unknown offset" lets unanchored scans
   still teach line detection.

8. **Group split by device and cooldown.** Scans from one cooldown are near-duplicates;
   splitting randomly would inflate real-data scores.

9. **Ensemble disagreement drives review flags and the labelling queue** (active learning), so
   labelling effort goes where the model is weakest.

10. **Advisory by default, with safety in the guidance itself.** Recommendations never leave
    the safe limits, and moves larger than `max_step` are split. Protocol responses carry
    `max_step` so a backend can ramp safely. ChargeCell never moves gates itself. The user chose
    to keep it advisory (2026-09-30): an automated loop lives in the backend, which should hand
    `needs_review` and `no_confident_step` results to a person.

11. **All guidance geometry in the model's index space**, converted to volts at the end, so
    hole devices and reversed sweeps need no special cases.

12. **Staircase-aware lattice fitting** (join reservoir segments through interdot segments),
    **spacing plausibility vs. the device prior (0.4–2.5x)**, and **no window shrinking on
    unmeasured spacing**. Together these fixed navigation loops found by the oracle tests.

13. **GUI: no framework, no build step, works offline** (lab PCs are often air-gapped). One
    FastAPI process serves both API and static files; the workspace is a plain folder of JSON and
    npz files that can be backed up or versioned.

14. **Practice devices** (simulated) in the GUI, for operator training and as the end-to-end
    test harness for guidance (`evaluate_navigation`).
