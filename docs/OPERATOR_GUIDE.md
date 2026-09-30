# ChargeCell operator guide

This guide is for the person at the fridge. No programming needed.

## Before the first use

ChargeCell ships starter models trained on simulated scans, installed on first start. The
sidebar lists which scan kinds have a model. If one is missing, go to **Synthetic data**,
generate about 3000 scans of that kind, then **Train** a model on them (see "Making a better
model" below).

On the **Device** page, fill in what you know: gate names (P plungers, X exchange gates,
T tunnel gates, M sensor), safe limits, the largest step you allow. Everything is optional and
ChargeCell assumes no voltage scale: without limits it warns you, without a step limit it moves
at most one window at a time.

## Three kinds of scan

ChargeCell reads the three scans of an exchange-only tune-up, in this order:

1. **Plunger vs tunnel gate (PvT)**, for each edge dot: sweep the plunger against the tunnel gate
   between the dot and its reservoir. ChargeCell finds where electrons load cleanly and where the
   dot holds exactly one electron.
2. **Plunger vs plunger (PvP)**, for each pair: find the (1,1) cell (the main loop below).
3. **Tie bar**: a plunger-plunger zoom on the (1,1)-(2,0) transition, which ChargeCell suggests
   once (1,1) is found. It marks the two triple points, measures the coupling between the dots
   and suggests a first readout point.

When you import a file, pick its kind (or put `"kind"` in the request from your software).

## The daily loop

1. Take a plunger-vs-plunger scan (for example P1 vs P2) the way you normally do.
2. Get it into ChargeCell. If your measurement software is connected (docs/PROTOCOL.md), it
   arrives on its own. Otherwise use **Scans → Import files**. ChargeCell `.json` request files
   need nothing else. For other files, type the two gate names, the device and the cooldown.
3. Open it on **Review** and click **Analyse**.
4. Read the big coloured word, then the "What to do next" box.
5. Do what it says, take the new scan, and repeat.

## Reading the result

**Green: (1,1) found.** The outline marks the (1,1) cell; the cross marks its centre; the green
diamond marks the (1,1)-(2,0) boundary used for readout. ChargeCell only says "found" if it can
also see the empty region for both dots, because that is the only way to be sure the cell is
really (1,1) and not (2,2) or (3,1).

**Amber: (1,1) not in this window.** The dashed box on the plot is the next scan, and the arrow
shows the move. The panel lists exact voltages. "Confidence: high" means the move was calculated
from lines visible in this scan. "Medium" or "low" means ChargeCell is exploring: it moves by
three quarters of a window so the new scan overlaps the old one.

**Green (plunger vs tunnel gate): one-electron point found.** Blue lines mark the loading lines
(0->1, 1->2, ...); the green cross is the suggested plunger and tunnel-gate setting for one
electron. The panel gives the tunnel-gate range where electrons load cleanly. Below it the lines
fade and jump sideways (the tunnel gate is too closed for electrons to follow the sweep); above
it they smear out (the dot is barely separated from its reservoir).

**Green (tie bar): tie bar found.** The teal dots are the two triple points, the green bar joins
them, and the green cross is a first guess for the spin-to-charge readout point, just past the
transition on the (2,0) side. The coupling ratio (width of the transition divided by the length
of the tie bar) measures how strongly the two dots are coupled without needing any calibration.
If you give lever arms and the electron temperature on the Device page, it is also shown in µeV
and GHz. If you set a target range for the ratio, ChargeCell tells you which way to move the
exchange gate.

**Red: can't interpret this scan.** Fix the problem named in the panel, then rescan the same window.

| Problem | Usual fix |
|---|---|
| Too noisy | Retune the sensor to its steepest flank; average longer (4x the time halves the noise). |
| Sensor lost sensitivity | Sweep the sensor gate across its Coulomb peak and park it on the steepest flank. Check sensor compensation is on. |
| Dots merged | Lower the exchange gate between the two dots a little (the panel says how much) and rescan. |
| Charge jumps | Wait a few minutes after big moves; rescan; avoid large steps. |
| Too few points | Rescan with the number of points shown (at least twice as many). |

**"Needs review"** means the model was unsure or one of its safety checks failed. Look at the scan
yourself before acting. These scans are also the most useful ones to label.

**"(1,1) may be in this window, but the electron count is not certain"** means the model thinks it
sees (1,1) but a safety check held it back (for example, not enough of the empty region is in
view to count electrons from). The next window is usually wider, so that the empty region and the
neighbouring cells are in view.

## Doing the move safely

- **Copy settings** gives you the next window as text.
- **Download result (JSON)** saves the result in the format measurement software reads
  (docs/PROTOCOL.md), so a script can set up the next scan for you. Ramp the gates in steps no
  larger than the device's step limit.
- Moves are never larger than the device's step limit and never leave its safe limits (set on the
  Device page). If a move is bigger, ChargeCell tells you to take the first step, rescan, and
  analyse again.
- If you moved other gates since the last "found" scan, targets taken from device history may be off.

## Runs: the record of a tune-up

Every scan ChargeCell analyses is recorded on the **Runs** page, grouped into runs (one tune-up
session on one device) and stages (one goal, for example "Find the (1,1) cell of P1-P2"). For
each scan you see what was measured, what ChargeCell concluded, what it advised, and whether the
next scan followed that advice. Notes you add and actions your measurement software reports
appear in the same list, in the order they happened.

Each step gets a mark when it happens: a green tick (fine), an amber **!** (a person should
look), a red cross (failed: the scan could not be read, or a goal stalled after three scans in a
row without a confident next step), or a blue dot (still in progress). Use it to:

- see where a long unattended tune-up got stuck, and why;
- check whether a bad result came from the guidance or from a scan that did not follow it;
- compare over many runs how many scans each goal takes and which problems come up most often
  (the table under the run).

Scans from the same device join the same run until nothing happens for 4 hours; your
measurement software can also start and name runs itself (docs/PROTOCOL.md). Click **Mark run
finished** when a tune-up is done. **Download** saves a run as JSON or as text.

## Labelling a scan

Your labels teach the model about your devices. A label takes about a minute.

1. Open **Label**. The next scan in the queue is the one the model is least sure about.
2. Press **A** and click along each dot A boundary (the line where dot A gains an electron),
   from bottom to top. Follow the jogs at the interdot lines. Double-click or press Enter to finish.
3. Press **B** and do the same for dot B boundaries, left to right.
4. Set how many electrons sit to the left of all A boundaries, and below all B boundaries.
   Choose 0 if you can see the empty region; choose **Unknown** if you can't tell.
5. The shaded labels on the plot show the electron numbers your lines imply. Check that (1,1) is
   where you think it is.
6. Choose the outcome. "From your lines" suggests one; use it if it agrees with you.
7. Type your name and click **Save and next**.

Shortcuts: **V** select and drag points, **Alt-click** inserts a point, **right-click** deletes a
point, **Backspace** removes the last point or the selected line, **Ctrl+Z** undo, **G** gradient
view (makes faint lines visible). Scroll to zoom, Shift-drag to pan.

**Draft from model** fills in the model's guess so you only correct it. On Review,
**Accept as label** saves the model's answer directly when it is clearly right.

Mark **Second check done** when a second person has looked at the label. You can train on only
double-checked labels.

## Making a better model

1. **Synthetic data**: generate a few thousand simulated scans (about 3000 is a good start). Use
   the same image size you plan to train with.
2. **Train**: tick the synthetic datasets, keep "Include labelled real scans" on, and start.
   Training runs in the background; the sidebar shows progress.
3. When it finishes, compare versions. Trust the **held-out real scans** numbers most: they come
   from cooldowns the model never saw. "FOUND calls that are right" should stay high (97% or more).
4. Click **Make active** on the better version. Old versions are kept, so you can switch back.

Real-data scores appear once you have labelled scans from at least two cooldowns.

## Practising

**New practice device** creates a simulated triple dot with its own quirks, for the scan kind you
choose next to the button. Its gates respond like a real device: the tunnel gates T1 and T2
decide whether P1 and P3 load electrons, and the exchange gates X1 and X2 set how strongly the
dots couple. Scan it, follow the advice with **Measure it on the practice device** (or **Take the
tie-bar scan** once (1,1) is found), and use **Reveal the true answer** to check. It is a safe way
to learn the workflow and to see how the guidance behaves. Practice devices come in very
different voltage scales on purpose. Practice runs appear on the Runs page too, where every
"found" is checked against the true answer.
