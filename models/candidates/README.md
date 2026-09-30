# Candidate models (not installed)

Models trained on simulated scans that did **not** meet the acceptance bar for bundling
(FOUND precision ≥ 0.95 on held-out synthetic scans, and a closed loop that reaches the goal on
≥ 70% of practice devices with no wrong FOUND). They are kept here, in Git LFS, so that the
next round of work does not have to retrain them, and so that you can try them. ChargeCell does
not install them on first start; only `chargecell/assets/models/` is installed.

| Kind | Model | Threshold | Held-out FOUND precision / recall | Closed loop (30 practice devices) |
|---|---|---|---|---|
| PvP | `model-20260930-073322754-6eac19` | 0.995 | 0.973 / 0.41 | 15/30 found, none wrong |
| PvT | `model-20260930-095542034-6aca7b` | 0.995 | 0.958 / 0.41 | 6/30 found, 1 wrong |
| Tie bar | `model-20260930-110759674-24455b` | 0.995 | 1.0 / 0.17 | 9/30 found, none wrong |

Numbers from `scripts/eval_model.py --n 300 --nav 30 --seed 2029` (tie bar: `--seed 2028`);
details and the reasons they fall short are in `docs/HANDOFF.md` section 3. The PvP and tie-bar
models never gave a wrong FOUND in any closed loop at this threshold; they are conservative, not
wrong.

To try one, add its folder on the Models page (**Add a model file**, then choose the folder), or
run `chargecell models add models/candidates/<folder> --use`. To recalibrate it on your own
held-out data: `chargecell calibrate --model <id> ...`.
