# ChargeCell

ChargeCell reads the charge-stability scans you take while tuning up an exchange-only spin-qubit
device (Si/SiGe, HRL style). For each scan it tells you what the scan shows and, if the goal is
not in view, which gate to change, in which direction and by how much. It only gives advice: it
never changes a gate voltage itself. Words in *italics* on first use are explained in the
[glossary](#glossary) at the end.

It has one *model* per kind of scan, following HRL's tune-up steps:

| Scan kind | Question it answers | When it finds the goal, you get |
|---|---|---|
| **Plunger vs tunnel gate** (PvT) | Where does the edge dot hold exactly one electron, and at which *tunnel-gate* setting do electrons load cleanly? | The *loading lines*, the clean tunnel-gate range, and the one-electron point. |
| **Plunger vs plunger** (PvP) | Is the *(1,1) cell* in this scan? | The cell's centre and outline, and where to take the tie-bar scan. |
| **Tie bar** (a plunger vs plunger zoom on the (1,1)-(2,0) line) | Where are the *triple points*, how strongly are the two dots coupled, and where should readout start? | The triple points, a coupling measurement, and a first readout point. |

Every scan gets one of three answers:

| Answer | Meaning | What ChargeCell gives you |
|---|---|---|
| **Found** | The goal is in view, and ChargeCell is sure (for (1,1): an empty region is visible for both dots, so the electrons can be counted). | The points listed above. |
| **Not in this scan** | The scan is readable, but the goal is somewhere else or cannot be counted to. | Which gates to change, which way, how far, and the next scan range (start, stop, number of points). |
| **Can't read this scan** | Too noisy, the *sensor* is not sensitive, the dots have merged, charges jump, or there are too few points. | What to fix first (retune the sensor, lower an exchange gate, average longer, take more points). |

ChargeCell assumes **nothing about your voltages**: devices differ widely, so every number comes
from your scans or from your (optional) device settings.

Everything happens on a web page that ChargeCell serves on your computer: look at results, label
scans, switch models, train new ones (here or on another computer with a GPU), and practise on
simulated devices. Measurement software talks to ChargeCell through a small JSON message format
(`docs/PROTOCOL.md`), so it works with any control software. Every analysed scan is also kept in
a **History**: each tune-up session is recorded step by step (scans, answers, advice, what was
done), so a long unattended session can be checked afterwards.

## Install

Python 3.10 or newer.

```bash
git lfs install                        # once per computer: model files are stored with Git LFS
git lfs pull                           # in your copy of this repository
pip install .                          # from your copy of this repository
chargecell serve                       # opens http://127.0.0.1:8765 in your browser
```

ChargeCell keeps scans, labels and models in one *workspace* folder, `~/chargecell-workspace`,
unless you pass `-w /path/to/folder`.

**Models.** No model comes installed yet: the models trained so far do not meet the agreed
quality bar (see "What has been tested"). To try ChargeCell now, add the trial models kept in
`models/candidates/`: on the **Models** page choose **Add a model file** and pick a model folder,
or run `chargecell models add models/candidates/<folder> --use`. You can also train your own
(**Simulated scans**, then **Train**).

## Quick start

The **Home** page lists the main tasks and a short getting-started checklist.

1. **Try it without hardware.** Choose a scan kind next to **New practice device** (Scans or
   Review page). ChargeCell simulates a three-dot device and takes a first scan. Click
   **Analyse**, read the answer, then **Measure it on the practice device** to follow the advice.
   Repeat until it says found. **Reveal the true answer** shows whether it was right.
2. **Enter your device** (optional). On **Device settings**: gate names (P plungers, X exchange
   gates, T tunnel gates, M sensor), safe voltage limits, the largest single step you allow, and,
   if you know it, a rough spacing between electrons for each plunger.
3. **Bring in real scans.** Let your measurement software send them (see below), or use
   **Scans → Import scan files**: `.json`, `.npz`, `.csv`. Always fill in the cooldown and the
   scan kind.
4. **Review and act.** Each scan gets an answer and next steps. Copy the settings of the next
   scan, or download the result for your measurement software.
5. **Label.** Where ChargeCell is wrong, correct it on the **Label** page: draw the charge lines
   (loading lines and the clean tunnel-gate range for PvT scans), set the electron counts,
   choose the answer. The scans the model is least sure about come first.
6. **Check the History.** Each tune-up session, step by step: every scan, what ChargeCell
   concluded and advised, whether the next scan followed the advice, and where it got stuck.
7. **Train and switch models.** See the next two sections.

## Switching models

Each scan kind has its own model, and one model per kind is **in use**: it reads every new scan
of that kind. The **Models** page shows, for each kind, the model in use with plain test results
("when it said found, it was right 97% of the time"), and the other versions you have.

- **Use this one** switches at once. Scans already analysed keep their old answer; ChargeCell
  offers to analyse them again with the new model.
- **Download** gives a *model file* (`.zip`). **Add a model file** on another ChargeCell adds it
  there (a model folder with `model.json` and `member_0.pt` works too). Adding the same model
  twice does nothing.
- **Rename** gives a model a name you will recognise. **Delete** removes a version that is not
  in use.

The same from the command line:

```bash
chargecell models                               # list; * marks the model in use for each kind
chargecell models use "PvP retrained"           # by name, id, or the end of the id
chargecell models add model-file.zip --use      # add a model file or folder, and use it
chargecell models export "PvP retrained" --out models/   # write a model file
chargecell models rename <id> --name "PvP retrained"
chargecell models delete <name>                 # not the one in use
```

## Training on another computer (GPU)

Training on a processor takes 20 minutes to 2 hours per model and slows the computer down. A
computer with an NVIDIA GPU (or an Apple M-series chip) is much faster, and it does not have to
be the one that runs ChargeCell's web page.

On the **Train** page, choose the scan kind and the data, then **Another computer** under "Where
should it train?". ChargeCell packs the training data into one *training-job file* and waits for
a training computer. The page shows the exact commands. There are three ways to connect:

1. **Over the lab network** (simplest). Start ChargeCell so other computers can reach it:

   ```bash
   chargecell -w <workspace> serve --host 0.0.0.0
   ```

   On the training computer, install ChargeCell (the same version) and a PyTorch build with GPU
   support (see pytorch.org), then run the command shown on the Train page:

   ```bash
   chargecell worker --server http://<gui-computer>:8765 --token <token from the Train page>
   ```

   The *worker* keeps running and trains every job you send, one at a time. Progress shows on the
   Train page; the finished model comes back by itself and appears on the Models page (and is
   put in use if you ticked "Use the new model as soon as it is ready").
2. **Through SSH**, if ChargeCell should stay private to its computer (the default). On the
   training computer, open a tunnel, then run the worker through it:

   ```bash
   ssh -N -L 8765:localhost:8765 <you>@<gui-computer>        # keep this running
   chargecell worker --server http://localhost:8765 --token <token>
   ```
3. **By hand**, with no network between the two. Download the training-job file from the Train
   page (or run `chargecell train ... --job-file job.zip`), carry it over, and run
   `chargecell train-job job.zip` there. That writes a model file; add it on the Models page.

`--device` chooses the hardware: `auto` (the default: a GPU if there is one), `cuda`, `mps` or
`cpu`. A failed job can be tried again from the Train page, and a job waiting for a training
computer survives a restart of ChargeCell.

**Security.** ChargeCell's web page has no login. With `--host 0.0.0.0` anyone on the network
can open it, so do this only on a lab network you trust (or use SSH, route 2). The worker
connections need the token shown on the Train page; it is kept in `<workspace>/worker_token`
(delete that file to make a new token).

## Connecting your measurement setup

ChargeCell never drives instruments. Your measurement software sends each scan as JSON and gets
back one of three answers: the points found (`found`), the next scan to take (`next_scan`), or
`no_confident_step` with the reason and a suggestion for a person to look at.

```python
from chargecell.client import ChargeCellClient

cc = ChargeCellClient("http://127.0.0.1:8765")          # a running `chargecell serve`
r = cc.analyze(signal, x_volts, y_volts, x_gate="P1", y_gate="P2", device="devA",
               cooldown="CD7", voltage_state={"P3": 0.845, "M1": 0.920})
print(r["outcome"], r["headline"])
r = cc.analyze(signal, p1_volts, t1_volts, x_gate="P1", y_gate="T1", kind="PvT", device="devA")
```

The same JSON works over HTTP from any language (`POST /api/v1/analyze`), or as files
(`chargecell analyze request.json --json`). Labelled scans can be sent for training with
`POST /api/v1/scans`. To group a tune-up session in the History, start a run and pass its id
(`cc.start_run(...)`, `analyze(..., run_id=...)`), report what your software does between scans
(`cc.log(...)`) and close it (`cc.close_run(...)`); without a run id, scans join their device's
open run. Full description: `docs/PROTOCOL.md`.

Simulated training scans come from ChargeCell's own physics-based simulator: three dots with
tunnel coupling, a realistic charge sensor, tunnel rates set by the tunnel gates, and typical
measurement problems (noise, sensor drift, charge jumps).

## Command line

```bash
chargecell serve                                                # the web page
chargecell models                                               # list and switch models (see above)
chargecell simulate --name sim1 -n 3000 --size 96               # simulated scans (--kind PvT|tiebar)
chargecell train --synthetic sim1 --activate                    # train here and use the model
chargecell train --synthetic sim1 --job-file job.zip            # ... or pack it for another computer
chargecell train-job job.zip                                    # train from a job file (GPU computer)
chargecell worker --server http://lab-pc:8765 --token <token>   # train jobs sent from the Train page
chargecell analyze scan.csv --x-gate P1 --y-gate P2             # print the answer and next steps
chargecell analyze request.json --json                          # chargecell/1 request -> response
chargecell runs                                                 # the History; `runs <id>` shows one, --stats
chargecell calibrate --model <id> --target 0.99 --held-out sim2 # how sure a model must be to say found
chargecell navigate --devices 20                                # try the advice on simulated devices
chargecell schema                                               # the format of the chargecell/1 messages
```

`chargecell <command> --help` explains each option.

## What has been tested, and what has not

- **The advice**, with perfect perception (the simulator's true answer in place of the model):
  on 30 simulated devices per scan kind, with random voltage scales, ChargeCell reaches the goal
  on 30 of 30 for each kind, and every "found" is right (typically after 3, 3 and 1 scans for
  PvP, PvT and tie bar). The automated tests (`pytest`) check this and much more.
- **The trained models, on simulated scans only.** No real scan has been seen yet. The agreed
  bar for installing a model with ChargeCell: on simulated scans kept out of training, at least
  95% of its "found" answers are right; and on 30 practice devices it reaches the goal on at
  least 70% within 8 scans, with no wrong "found". Results so far
  (`scripts/eval_model.py --n 300 --nav 30 --seed 2029`):

  | Kind | "Found" right / goals recognised (simulated scans) | Practice devices: goal reached, wrong "found" | Bar |
  |---|---|---|---|
  | PvP | 97% / 41% | 15 of 30, none wrong | not met (needs 21 of 30) |
  | PvT | 96% / 41% | 6 of 30, 1 wrong | not met |
  | Tie bar | 100% / 17% | 9 of 30, none wrong | not met |

  The PvP model is careful rather than wrong: when it is not sure, it says so and asks for a
  confirmation scan or a wider scan. Next steps are in `docs/HANDOFF.md` (section 4.1).
- **Not tested:** anything on real devices. Label real scans from at least two cooldowns before
  trusting any model; the Models page then shows results on your own scans.

More: `docs/OPERATOR_GUIDE.md` (guide for operators), `docs/DESIGN.md` (how it works),
`docs/PROTOCOL.md` (for measurement software), `docs/CODEMAP.md` (for developers),
`docs/HANDOFF.md` (status and next steps).

## Glossary

**Device and physics**

- **Charge-stability scan** (charge-stability diagram): a 2D scan of the sensor signal while two
  gate voltages are swept. Lines in it mark where a dot gains or loses an electron.
- **Quantum dot, dot**: a small region of the device that holds a countable number of
  electrons. The devices here have three dots in a row or triangle.
- **Plunger gate** (P1, P2, ...): the gate above a dot; raising it adds electrons to that dot.
- **Exchange gate** (X1, X2, ...): the gate between two dots; it sets how strongly they are
  coupled. On these devices a lower voltage means weaker coupling.
- **Tunnel gate** (T1, T2, ...): the gate between an edge dot and its reservoir; it sets how
  fast electrons tunnel in and out. Too closed: electrons do not load during the sweep. Too open:
  the lines smear out.
- **Reservoir**: the electron supply next to an edge dot.
- **Sensor** (M1, ...): a nearby dot whose current changes when the electron number of the
  device dots changes. It works best on the steep side (**flank**) of one of its **Coulomb
  peaks**: the current peaks you see when sweeping the sensor gate alone. Retuning the sensor
  means moving it back onto that flank.
- **Charge line** (transition): a line in a scan where one dot gains or loses an electron.
- **Loading line**: in a plunger vs tunnel gate scan, the line where the dot gains its next
  electron from the reservoir.
- **Interdot line**: a short line where an electron moves from one dot to the other (the total
  stays the same).
- **(1,1) cell**: the region of a plunger vs plunger scan where each of the two dots holds
  exactly one electron. (2,0) means two electrons in the first dot and none in the second.
- **Empty region**: the region with no electrons in a dot. ChargeCell counts electrons from it,
  so it must be in view before ChargeCell says "found".
- **Electron spacing**: how far a plunger must move to add one more electron.
- **Tie bar**: the short interdot line between the (1,1) and (2,0) cells, used to read out the
  qubit. Its two ends are the **triple points**, where three charge states meet.
- **Coupling ratio**: width of the interdot line divided by the tie-bar length; a measure of how
  strongly the two dots are coupled.
- **Spectator dot**: a dot that is not swept in a scan but whose electron number still matters
  (for (1,1,1), each of the three dots must hold one electron).
- **Latching**: when electrons tunnel so slowly that lines shift or break up along the sweep.
- **Charge jumps** (charge switching): the pattern shifts suddenly between sweep lines because
  a charge nearby moves.
- **Lever arm**: how much a gate voltage shifts a dot's energy (in eV per V). Optional; only used
  to give tie-bar results in energy units.
- **Virtual gates**: combinations of physical gates that each move one dot only. If you scan in
  virtual gates, give the matrix on the Device settings page.
- **Cooldown**: one cooling of the device to its operating temperature. Scans from different
  cooldowns can look different, so ChargeCell tests models on cooldowns they were not trained on.

**Scans and advice**

- **Scan window** (window): the voltage range of the two swept gates, and the number of points.
- **Safe limits**: the lowest and highest voltage you allow on a gate. Suggested scans always stay
  inside them.
- **Practice device**: a simulated device on which you can try ChargeCell and follow its advice
  without hardware.
- **Run** (History): one tune-up session on one device, recorded step by step. A new run starts
  after 4 hours without scans, or when your measurement software starts one.
- **Confirmation scan**: the same window scanned again with longer averaging, asked for when
  everything looks right but the model is not sure enough.

**Models and training**

- **Model**: the trained program that reads scans of one kind. A model is several **networks**
  (neural networks) trained the same way whose answers are averaged, which makes the answers
  steadier.
- **In use**: the model that reads new scans of its kind. One per scan kind.
- **Label**: your answer for a scan (its charge lines, electron counts and outcome). Labels are
  used to test models and to train new ones.
- **Simulated scans**: scans made by ChargeCell's simulator, where the right answer is known
  exactly. Models learn mostly from them until enough real scans are labelled.
- **Training**: letting a model learn from simulated and labelled scans. A **pass** (epoch) is one
  run through all the training data.
- **Kept out of training** (held out): scans used only to test a model, never to train it.
- **Confidence**: how sure the model is of its answer. ChargeCell says "found" only above a
  set confidence (the **"found" threshold**), chosen after training so that "found" is right at
  least 97% of the time on scans kept out of training (you can choose another share).
- **Precision** ("right when it said found") and **recall** ("recognised"): the share of
  "found" answers that were right, and the share of scans showing the goal that the model
  recognised.
- **GPU**: a graphics processor; it trains models much faster than a normal processor (CPU).
- **Model file**: a `.zip` holding one model, for moving it between computers.
- **Training-job file**: a `.zip` holding everything needed to train a model on another computer
  (the settings and the training data).
- **Worker**: `chargecell worker`, run on a training computer; it collects training jobs from the
  Train page, trains them, and sends the models back.
- **Token**: a password-like code that lets a worker connect. It is shown on the Train page.
- **Workspace**: the folder where ChargeCell keeps scans, labels, models and the History.
- **chargecell/1**: the JSON message format measurement software uses to talk to ChargeCell.

## Repository layout

```
chargecell/        the package (simulators, models, analysis per scan kind, protocol, History, web page, command line)
tests/             automated tests (advice checked against the simulator's truth, web API, training)
scripts/           train_starter, eval_model, nav_trace, gui_check
docs/              operator guide, design, protocol, code map, research notes, decisions, handoff
models/candidates/ trained models below the quality bar (Git LFS; not installed)
```
