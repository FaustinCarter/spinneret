# ChargeCell: instructions for Claude

ChargeCell finds the (1,1) charge cell in double-quantum-dot charge stability diagrams and tells
the operator where to scan next. It has a simulator, labelling GUI, training pipeline, and a
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
pytest -q                                        # 15 tests, ~50 s on 4 cores
python scripts/nav_trace.py --oracle --bench     # guidance logic benchmark (expect 30/30)
chargecell -w /tmp/demo_ws serve --no-browser    # GUI at http://127.0.0.1:8765
chargecell schema                                # chargecell/1 JSON Schema
python scripts/gui_check.py --out /tmp/shots     # browser check (needs playwright + chromium)
python scripts/train_starter.py --size 96 --epochs 12 --ensemble 3 --workers 4 --bundle
python scripts/eval_model.py --model-dir <ws>/models/<id>
```

## Rules for this codebase

- **Units:** volts in scans and public APIs; mV inside `simulate/`. Convert at boundaries.
- **Geometry in index space:** analysis works on the model's S x S canonical grid
  (`model/infer.py: Grid`) and converts to volts at the end. Don't add hole-device special
  cases downstream of `Grid`.
- **Never reorder `schema.REASONS` or `LINE_FAMILIES`**: network heads depend on the order.
  Append only, then retrain.
- **FOUND stays conservative:** both dots anchored + calibrated threshold + cell not cut off.
  Don't loosen these gates to improve recall without the user's agreement.
- **Guidance must stay safe:** windows inside `DeviceConfig.safe_limits`, moves split to
  `max_step`. Tests in `tests/test_guidance.py` guard this.
- **Keep `tests/conftest.py: truth_to_prediction` in sync** with `Analyzer.predict`'s output
  keys. Oracle tests check guidance logic separately from model quality.
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
