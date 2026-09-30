# ChargeCell: instructions for Claude

ChargeCell reads the charge-stability scans of an exchange-only qubit tune-up, one model per scan
kind as in HRL's pipeline: PvT (load one electron with a clean reservoir tunnel rate), PvP (find
the (1,1) cell) and tiebar (triple points, coupling, readout point). It tells the operator what a
scan shows and where to scan next. It has simulators, a labelling GUI, a training pipeline, and a
backend-neutral JSON protocol (chargecell/1) for measurement software. Users are physicists and lab technicians working on Si/SiGe exchange-only
spin qubits (HRL style), many of them not software engineers.

**Start here:** @docs/HANDOFF.md has the current status, unfinished work in priority order,
known sharp edges, and open questions for the user. Read `docs/CODEMAP.md` before changing code.
Background: `docs/DESIGN.md`, `docs/DECISIONS.md`, `docs/RESEARCH_NOTES.md`. Integration:
`docs/PROTOCOL.md`.

## Commands

```bash
git lfs install && git lfs pull                 # bundled model weights are in Git LFS
pip install -e ".[dev]"                          # Python >= 3.10; CPU torch is fine
pytest -q                                        # 24 tests, ~3 min on 4 cores
python scripts/nav_trace.py --oracle --bench --kind PvP   # guidance benchmark, expect 30/30
python scripts/nav_trace.py --oracle --bench --kind PvT   # (also tiebar; add --no-prior)
chargecell -w /tmp/demo_ws serve --no-browser    # GUI at http://127.0.0.1:8765
chargecell schema                                # chargecell/1 JSON Schema
python scripts/gui_check.py --out /tmp/shots     # browser check (needs playwright + chromium)
python scripts/train_starter.py --kind PvP --size 96 --epochs 14 --ensemble 3 -n 6000 --workers 4 --bundle
python scripts/eval_model.py --model-dir <ws>/models/<id>   # any kind; held-out + closed loop
```

## Rules for this codebase

- **Units:** volts in scans and public APIs; mV inside `simulate/`. Convert at boundaries.
- **Geometry in index space:** analysis works on the model's S x S canonical grid
  (`model/infer.py: Grid`) and converts to volts at the end. Don't add hole-device special
  cases downstream of `Grid`.
- **Never reorder a kind's reasons or line families** (`schema.REASONS`, `LINE_FAMILIES`, and
  the per-kind lists in `kinds.py`): network heads depend on the order. Append only, then retrain.
- **No voltage scale.** Never add an absolute voltage constant or default: device settings are
  optional, and fallbacks are measured spacings or fractions of the window. Practice devices have
  random voltage scales so the oracle benchmarks catch mistakes.
- **One heavy job at a time on this 4-core box.** Training uses all cores; running data
  generation or the test suite alongside it slowed epochs 5-10x.
- **FOUND stays conservative:** calibrated threshold + anchoring (an empty region wider than 1.3
  spacings, checked geometrically too) + target not cut off; PvT and tiebar have their own gates.
  Don't loosen these gates to improve recall without the user's agreement.
- **Guidance must stay safe:** windows inside `DeviceConfig.safe_limits`, moves split to
  `max_step`. Tests in `tests/test_guidance.py` guard this.
- **Keep `tests/conftest.py: truth_to_prediction` (and the PvT/tiebar versions) in sync** with
  `Analyzer.predict`'s output keys. Oracle tests check guidance logic separately from model
  quality; run the oracle benchmarks for all three kinds after changing guidance.
- **Front end:** plain JS, no build step, must work offline. `plot.js` stays wrapped in an IIFE.
  Keep the design tokens in `style.css`. After UI changes, run `scripts/gui_check.py` and look
  at the screenshots.
- **Operator-facing text** (GUI, OPERATOR_GUIDE, guidance headlines/steps) is plain language
  with concrete numbers: which knob, which direction, how far.
- **No instrument code.** spinQICK was removed at the user's request. Measurement backends
  talk to ChargeCell only through the chargecell/1 protocol (`protocol.py`, `docs/PROTOCOL.md`).
  ChargeCell stays advisory and never moves gates. Change the protocol additively (new optional
  fields or enum values) or bump the version; keep `docs/PROTOCOL.md` and `tests/test_protocol.py`
  in step with `protocol.py`.
- **Model weights** in `chargecell/assets/models/` are tracked with Git LFS (`.gitattributes`).
  Replace the bundled model only with one that meets the acceptance bar in HANDOFF.
- Run `pytest -q` before committing. Update `docs/HANDOFF.md` when you finish or discover
  work items. Remove the `@docs/HANDOFF.md` import above once the handoff items are all closed.
