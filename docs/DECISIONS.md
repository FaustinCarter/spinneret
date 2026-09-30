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

## Decisions of 2026-09-30 (second session)

15. **One model per scan kind, one architecture.** HRL runs separate CNNs for PvT, PvP and tie-bar
    scans; ChargeCell does the same with one U-Net whose heads come from `kinds.py`. Each kind
    has its own datasets, model versions, calibrated threshold and active model. PvP keeps its
    original layout, so older PvP weights still load.

16. **No voltage scale anywhere** (user's instruction). Device settings with voltages are all
    optional and empty by default; guidance falls back to spacings measured in the scans, the
    device's history, and fractions of the window (one window per move, a quarter spacing for an
    exchange-gate step). Practice devices use a random overall voltage scale so the oracle
    benchmarks catch any absolute constant. Operator text picks units from the value's size.

17. **Anchoring needs more than a cell of empty region.** The first trained PvP model made wrong
    FOUND calls exactly two spacings off: the training labels had called a dot anchored when a
    line-free strip of 0.25 spacings was visible, which is indistinguishable from an occupied
    cell cut off by the window. Labels now need 1.3 spacings (the widest occupied cell is about
    1.25), target windows include that much, and a geometric FOUND gate checks it at analysis
    time independently of the network.

18. **A demoted FOUND stays readable.** FOUND and NOT_IN_WINDOW both mean the scan is readable, so
    a FOUND that fails a check becomes UNINTERPRETABLE only when the network gives that more
    than 50%. The old rule compared the two leftovers (0.06 vs 0.02 for a 0.92 FOUND) and sent
    readable scans to "fix and rescan" loops.

19. **The tie-bar coupling is measured, not predicted.** The network only locates the tie bar and
    triple points; the width of the charge transfer is fitted on the raw signal and reported as
    a voltage-free ratio (width / tie-bar length), and in µeV only when the user supplies lever
    arms and electron temperature. It is checked against the simulator's physics in the tests.

20. **PvT: tunnel regime per pixel, tunnel gate first.** The PvT network predicts slow / good /
    open tunnelling per pixel (supervised on rows that contain a transition) besides electrons
    and lines. When more than 60% of the window is too slow or too open, fixing the tunnel gate
    takes priority over plunger moves, because loading lines seen in a few rows cannot be
    trusted to steer the plunger.

21. **Practice devices model the T and X gates.** Tunnel gates set the edge dots' reservoir
    rates (and latching in plunger scans), exchange gates set the interdot coupling. This makes
    the whole PvT → PvP → tie-bar sequence practisable and testable in closed loop, with each
    FOUND judged against the ground truth for its kind.

22. **User choices recorded:** keep the name ChargeCell inside the spinneret repository; train
    here on CPU (PvP at 96 px, 3 members); bundled weights in Git LFS; no license for now;
    advisory only (ChargeCell never moves gates; automation lives in the backend); spinQICK
    removed; next priorities after this: PvT/tie-bar models (done), then the automation tree.

23. **The automation tree records; it does not drive.** HRL's tree logs actions its own software
    takes. ChargeCell is advisory (decision 22), so its tree holds what it was sent, what it
    answered, whether the next scan followed that answer, and what the backend reports doing
    (run events). Gate changes between scans are also inferred from `voltage_state`. A stage's
    scans are siblings rather than HRL's chain of inserted children: the depth-first order is
    the same and the tree stays readable.

24. **Every saved analysis is recorded, grouped automatically.** Backends need not change
    anything: scans join their device's open run, and a run ends after 4 hours without scans.
    Backends that want exact grouping pass `options.run_id`. Batch re-analysis after training is
    not a tune-up action and is not recorded; re-analysing a scan with the same result adds
    nothing.

25. **Never rescan the window just seen.** The closed-loop record of the first model showed the
    PvP guidance proposing the present window again and again after a FOUND was held back (the
    "cover both windows" rule gave the same window when the suggestion lay inside it). An
    identical window can only help against noise or sensor trouble, which UNINTERPRETABLE
    handles; for everything else the next window is now 1.5x wider around the expected cell.

26. **Two more checks against miscounting.** The closed loop's wrong FOUNDs came from a faint
    first transition missed by the occupancy map (counts one too high) and sensor Coulomb-peak
    features mistaken for dot lines. FOUND now also needs the network's own line map to agree
    that the empty region is empty (`hidden_line_in_empty`); training data put a third of the
    FOUND-intent windows at the anchoring boundary and 30% of the others next to (1,1), and the
    line loss weighs sensor, spectator and interdot pixels twice as much. Practice devices now
    follow the fixes ChargeCell asks for (retune the sensor, average longer), so closed-loop
    numbers measure the guidance rather than an operator who ignores it.
