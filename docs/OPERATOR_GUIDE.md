# ChargeCell operator guide

This guide is for the person at the fridge. No programming needed. Words that may be new are
explained in the glossary at the end of the README.

## Before the first use

Start ChargeCell (`chargecell serve`) and open the **Home** page. It lists the main tasks and a
short checklist of what is still missing.

**Models.** Each kind of scan needs its own model. The bottom of the sidebar shows which kinds
have one (a tick) and which do not (a dash); click it to open the **Models** page. No model comes
installed yet: the models trained so far on simulated scans are careful but do not find the goal
often enough (README, "What has been tested"). To get one:

- someone gives you a model file (`.zip`): **Models → Add a model file**, tick "Use it now";
- or add a trial model from `models/candidates/` in the repository the same way (choose its
  folder);
- or make one: **Simulated scans** (about 3000 scans of that kind), then **Train** (see "Making a
  better model" below).

**Your device.** On **Device settings**, fill in what you know: gate names (P plungers, X
exchange gates, T tunnel gates, M sensor), safe voltage limits, the largest single step you
allow. Everything is optional and ChargeCell assumes nothing about your voltages: without safe
limits it warns you, and without a step limit it moves at most one scan width at a time.

## Three kinds of scan

ChargeCell reads the three scans of an exchange-only tune-up, in this order:

1. **Plunger vs tunnel gate (PvT)**, for each edge dot: sweep the plunger against the tunnel gate
   between the dot and its reservoir. ChargeCell finds where electrons load cleanly and where the
   dot holds exactly one electron.
2. **Plunger vs plunger (PvP)**, for each pair of dots: find the (1,1) cell (the main loop below).
3. **Tie bar**: a plunger vs plunger zoom on the (1,1)-(2,0) line, which ChargeCell suggests
   once (1,1) is found. It marks the two triple points, measures how strongly the dots are
   coupled, and suggests a first readout point.

When you import a file, pick its kind (or put `"kind"` in the request from your software).

## The daily loop

1. Take a plunger vs plunger scan (for example P1 vs P2) the way you normally do.
2. Get it into ChargeCell. If your measurement software is connected (docs/PROTOCOL.md), it
   arrives on its own. Otherwise use **Scans → Import scan files**. ChargeCell `.json` files need
   nothing else. For other files, type the two gate names, the device and the cooldown.
3. Open it on **Review** and click **Analyse**.
4. Read the big coloured words, then the "What to do next" box.
5. Do what it says, take the new scan, and repeat.

Under the answer, "Analysed with ..." names the model that read the scan; **Change model** opens
the Models page.

## Reading the result

**Green: (1,1) found.** The outline marks the (1,1) cell; the cross marks its centre; the green
diamond marks the (1,1)-(2,0) line used for readout. ChargeCell only says "found" if it can
also see the empty region for both dots, because that is the only way to be sure the cell is
really (1,1) and not (2,2) or (3,1).

**Amber: (1,1) is not in this scan.** The dashed box on the plot is the next scan, and the arrow
shows the move. The panel lists exact voltages. "How sure this advice is: high" means the move
was worked out from charge lines counted in this scan. "Medium" or "low" means ChargeCell is
searching: it moves by three quarters of a scan width so the new scan overlaps the old one.

**Green (plunger vs tunnel gate): one-electron point found.** Blue lines mark the loading lines
(first electron, second electron, ...); the green cross is the suggested plunger and tunnel-gate
setting for one electron. The panel gives the tunnel-gate range where electrons load cleanly.
Below it the lines fade and jump sideways (the tunnel gate is too closed for electrons to follow
the sweep); above it they smear out (the dot is barely separated from its reservoir).

**Green (tie bar): tie bar found.** The teal dots are the two triple points, the green bar joins
them, and the green cross is a first guess for the readout point, just past the line on the
(2,0) side. The coupling ratio (width of the interdot line divided by the length of the tie bar)
measures how strongly the two dots are coupled without any calibration. If you give lever arms
and the electron temperature on Device settings, it is also shown in µeV and GHz. If you set a
target range for the ratio, ChargeCell tells you which way to move the exchange gate.

**Red: can't read this scan.** Fix the problem named in the panel, then rescan the same window.

| Problem | Usual fix |
|---|---|
| Too noisy | Retune the sensor to the steepest flank of its Coulomb peak; average longer (4 times the time per point halves the noise). |
| Sensor not sensitive | Sweep the sensor gate across its Coulomb peak and park it on the steepest flank. Check sensor compensation is on. |
| Dots merged into one | Lower the exchange gate between the two dots a little (the panel says how much) and rescan. |
| Charges jump during the scan | Wait a few minutes after big moves; rescan; avoid large steps. |
| Too few points | Rescan with the number of points shown (at least twice as many). |

**"Please check by eye"** means the model was unsure or one of its safety checks failed. Look at
the scan yourself before acting. These scans are also the most useful ones to label.

**"(1,1) may be in this window, but the electron count is not certain"** means the model thinks
it sees (1,1) but a safety check held it back (for example, not enough of the empty region is
in view to count electrons from). The next window is usually wider, so that the empty region and
the neighbouring cells are in view. The panel says why under "Not called found because ...".

**"(1,1) appears to be in this window ... Rescan the same window, averaging 4 times longer, to
confirm"** means every check passed, but the model was not sure enough to say "found". This
usually means the scan was a little noisy. Check that the sensor sits on the steepest flank of
its peak, average 4 times longer per point, and scan the same window again. ChargeCell asks for
this at most once per window; if the rescan still does not settle it, it suggests a wider
window. The same applies to PvT and tie-bar scans.

## Doing the move safely

- **Copy settings** gives you the next window as text.
- **Download result (JSON)** saves the result in the format measurement software reads
  (docs/PROTOCOL.md), so a script can set up the next scan for you. Ramp the gates in steps no
  larger than the device's step limit.
- Moves are never larger than the device's step limit and never leave its safe limits (set on
  Device settings). If a move is bigger, ChargeCell tells you to take the first step, rescan,
  and analyse again.
- **Retune the sensor when asked.** After a move of more than about one electron, the steps say
  "Before this scan, retune the sensor ...": park the sensor gate on the steepest flank of its
  Coulomb peak at the centre of the new window. The dot farther from the sensor gives the
  weakest lines, and a missed faint line makes every electron count wrong.
- If you moved other gates since the last "found" scan, targets taken from earlier scans may be
  off.

## History: the record of a tune-up

Every scan ChargeCell analyses is recorded on the **History** page, grouped into runs (one
tune-up session on one device) and steps (one goal each, for example "Find the (1,1) cell of
P1-P2"). For each scan you see what was measured, what ChargeCell concluded, what it advised,
and whether the next scan followed that advice. Notes you add and actions your measurement
software reports appear in the same list, in the order they happened.

Each step gets a mark when it happens: a green tick (fine), an amber **!** (a person should
look), a red cross (failed: the scan could not be read, or a goal got stuck after three scans in
a row without a confident next step), or a blue dot (still in progress). Use it to:

- see where a long unattended tune-up got stuck, and why;
- check whether a bad result came from the advice or from a scan that did not follow it;
- compare over many runs how many scans each goal takes and which problems come up most often
  (the table under the run).

Scans from the same device join the same run until nothing happens for 4 hours; your
measurement software can also start and name runs itself (docs/PROTOCOL.md). Click **Mark run
finished** when a tune-up is done. **Download** saves a run as JSON or as text.

## Labelling a scan

Your labels teach the model about your devices. A label takes about a minute.

1. Open **Label**. The next scan offered is the one the model is least sure about.
2. Press **A** and click along each dot A boundary (the line where dot A gains an electron),
   from bottom to top. Follow the jogs at the interdot lines. Double-click or press Enter to
   finish.
3. Press **B** and do the same for dot B boundaries, left to right.
4. Set how many electrons sit to the left of all A boundaries, and below all B boundaries.
   Choose 0 if you can see the empty region; choose **Unknown** if you can't tell.
5. The labels on the plot show the electron numbers your lines imply. Check that (1,1) is where
   you think it is.
6. Choose the outcome. "From your lines" suggests one; click **Use this** if it agrees with you.
7. Type your name and click **Save and next**.

Shortcuts: **V** select and drag points, **Alt-click** inserts a point, **right-click** deletes a
point, **Backspace** removes the last point or the selected line, **Ctrl+Z** undo, **G** edge
view (makes faint lines stand out). Scroll to zoom, Shift-drag to move.

**Start from the model's reading** fills in the model's answer so you only correct it. On
Review, **It's right: save as label** saves the model's answer directly when it is clearly right.

Tick **Checked by a second person** when someone else has looked at the label. You can train on
double-checked labels only.

**Tie-bar scans** are labelled the same way, with two lines: dot A's boundary where it goes from
1 to 2 electrons (it runs along the tie bar between the two triple points) and dot B's boundary
from 0 to 1. The counts are already set to those of a tie-bar zoom (1 and 0).

**Plunger vs tunnel gate scans**: press **L** and draw each loading line of the dot, following
it across the tunnel-gate range. Say how many electrons sit left of the first line (0 if the
empty dot is visible). Then mark the **clean tunnel-gate range**: where the lines are sharp and
continuous. Type the two values or use **Pick on plot**; leave a field empty if the clean range
continues beyond the window. The plot shades where the tunnel gate is too closed or too open.

## Switching models

The **Models** page shows, for each scan kind, the model in use and its test results in plain
words: how often its "found" answers were right, and how many of the scans that showed the goal
it recognised. Trust results on **your labelled scans** most: they come from cooldowns the model
never saw. "Right when it said found" should stay high (97% or more).

- **Use this one** switches straight away. ChargeCell then offers to analyse the scans of that
  kind again with the new model.
- **Download** saves a model file, to add to ChargeCell on another computer.
- **Rename** gives a model a name you will recognise; **Delete** removes a version you no longer
  need (not the one in use).

## Making a better model

1. **Simulated scans**: make a few thousand simulated scans of the kind you need (about 3000 is a
   good start). The image size must match the model: 96 pixels for plunger vs plunger, 64 for the
   others.
2. **Train**, in four steps on one page: the scan kind; what it learns from (tick the sets of
   simulated scans, and keep your labelled scans ticked); where it trains; a name.
3. **Where it trains.** "This computer" works but takes 20 minutes to 2 hours and slows the
   computer down. "Another computer" (for example one with a GPU) is faster: the page shows the
   command to run there (`chargecell worker ...`), and the progress appears on the Train page as
   if the training ran here. The README ("Training on another computer") explains the choices,
   including carrying the training data over by hand when there is no network.
4. When it finishes, compare the new model with the one in use on the **Models** page, and click
   **Use this one** if it is better (or tick "Use the new model as soon as it is ready" before
   starting). Old versions are kept, so you can switch back.

Results on real scans appear once you have labelled scans from at least two cooldowns.

## Practising

**New practice device** creates a simulated three-dot device with its own quirks, for the scan
kind you choose next to the button. Its gates respond like a real device: the tunnel gates T1
and T2 decide whether P1 and P3 load electrons, and the exchange gates X1 and X2 set how strongly
the dots couple. Scan it, follow the advice with **Measure it on the practice device** (or **Take
the tie-bar scan** once (1,1) is found), and use **Reveal the true answer** to check. It is a safe
way to learn the workflow and to see how the advice behaves. Practice devices come with very
different voltage scales on purpose. Practice runs appear in the History too, where every "found"
is checked against the true answer.
