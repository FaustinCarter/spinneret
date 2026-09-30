# ChargeCell: instructions for Claude

ChargeCell finds the (1,1) charge cell in double-quantum-dot charge stability diagrams and tells
the operator where to scan next. It has a simulator, labelling GUI, training pipeline, and
spinQICK integration. Users are physicists and lab technicians working on Si/SiGe exchange-only
spin qubits (HRL style), many of them not software engineers.

**Start here:** @docs/HANDOFF.md has the current status, unfinished work in priority order,
known sharp edges, and open questions for the user. Read `docs/CODEMAP.md` before changing code.
Background: `docs/DESIGN.md`, `docs/DECISIONS.md`, `docs/RESEARCH_NOTES.md`.

## Commands

```bash
pip install -e ".[dev,spinqick]"               # Python >= 3.10; CPU torch is fine
pytest -q                                        # 13 tests, ~40 s on one core
python scripts/nav_trace.py --oracle --bench     # guidance logic benchmark (expect 30/30)
chargecell -w /tmp/demo_ws serve --no-browser    # GUI at http://127.0.0.1:8765
python scripts/gui_check.py --out /tmp/shots     # browser check (needs playwright + chromium)
python scripts/train_starter.py --bundle         # train + bundle the starter model
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
- **spinQICK:** it has no simulator. It is used for importing real `.nc` files and for the
  exported next-scan script. API facts are in `docs/RESEARCH_NOTES.md` §3; verify against the
  user's installed version before changing `export.py` or `importers/spinqick_nc.py`.
- Run `pytest -q` before committing. Update `docs/HANDOFF.md` when you finish or discover
  work items. Remove the `@docs/HANDOFF.md` import above once the handoff items are all closed.
