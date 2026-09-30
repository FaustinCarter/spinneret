# Research notes

Facts gathered during the original session that shaped the design. Web sources were read in
September 2026; re-check anything time-sensitive.

## 1. HRL exchange-only (EO) qubit papers

| Paper | ID | Relevance |
|---|---|---|
| Blumoff et al., "Fast and high-fidelity state preparation and measurement in triple-quantum-dot spin qubits", PRX Quantum 3, 010352 (2022) | arXiv:2112.09801 | Readout by detuning from an idle point in (1,1) toward (2,0) (PSB with a dot charge sensor, "M" gate). Initialization uses the (2,0)-(3,0) boundary (often >10x faster than (1,0)-(2,0)). Sensor is a dot in Coulomb blockade; 1/f charge noise limits SNR. Gate naming: P plungers, X exchange barriers, T tunnel gates to reservoirs, M sensor. |
| Acuna et al., "Coherent control of a triangular exchange-only spin qubit", Phys. Rev. Applied 22, 044057 (2024) | arXiv:2406.03705 | Pairwise diagrams sweep two plungers with others fixed; the (1,1,1) point sits at the centre, with neighbours (0,1,1), (1,1,0), etc. Curved interdot transitions at higher occupancy from strong tunnel coupling (>10 GHz extracted). Linear compensation on M1 during scans keeps sensitivity. Readout at (3,1,1)-(4,1,0); initialization via (4,1,0)-(5,1,0). |
| S. Ha et al., "Two-dimensional Si spin qubit arrays with multilevel interconnects", PRX Quantum 6, 030327 (2025) | arXiv:2502.08861 | Sensor sensitivity falls roughly as 1/r³ with distance, so dots far from the sensor give faint lines. Regular charging transitions indicate low disorder, which justifies per-device priors on spacing and slope. |
| HRL Quantum Team, "A digitally controlled silicon quantum processing unit" (Nature, July 2026) | arXiv:2604.16216 | Supplement: automated tune-up with **separate CNNs for PvT, PvP and "tiebar" scans** (tiebar = PvP zoomed on the (1,1)-(2,0) transition), each trained on thousands of human-labelled datasets, outputting a graph of **keypoints** used to set gate voltages. Tune-up actions are recorded in an **automation tree**; each node is graded as it runs, and trees are analysed for success rates and failure modes. 54-dot three-rail chip, up to 18 EO qubits; dot loading through reservoirs accumulated by 8 B gates coupled via 12 T gates. |

Context: IBM announced an agreement to acquire HRL Laboratories on 23 July 2026, so HRL's
publication and open-source plans may change.

### HRL's tune-up pipeline in detail (QPU paper, Supplement S5 and Fig. S20; read 2026-09-30)

- Flow per qubit: sensor (Z-gate pinch-off, tune-DCS on M), electron loading, spin-to-charge
  conversion, exchange-axis calibration.
- **PvT** (plunger versus tunnel gate) "displays electron loading lines as identified by a CNN".
  T gates couple the reservoirs (accumulated by B gates) to the dots; PvT scans are how one
  electron is loaded under each plunger and how reservoir tunnel rates are set.
- **PvP** "reveals charge stability cells as identified by a CNN. The idle operating voltage
  setpoint is automatically updated to the center of the (1,1) charge cell."
- **Tiebar**: "a PvP experiment zoomed-in on the (1,1)-(2,0) charge transition", evaluated with
  a CNN that "provides a rough measure of interdot tunnel coupling and an initial guess of the
  spin-to-charge measurement coordinates". The same paper calls the charge transition traversed
  by slow initialization ramps "the tie bar".
- Each CNN is trained on thousands of human-labelled data sets and outputs a graph of keypoints
  (relational network, attention decoder, MLP + RGCN heads for keypoints and their links).
- Tune-up actions form an automation tree (depth-first order, each node graded when it runs;
  trees from many runs are mined for success rates and failure modes).
- Gate families on the 54-dot chip: P (plungers), X and Y (exchange), T (reservoir tunnel),
  B (reservoir bath), M and Z (sensors), S (SPAM). All DC biases were below 1.0 V there, a limit
  of the cryo-controller; ChargeCell does not assume it.

### Parameters used for the PvT and tie-bar simulators (scale-free)

- Interdot tunnel coupling is exponential in the barrier (exchange) gate voltage; Mills et al.,
  "Computer-automated tuning procedures for semiconductor quantum dot arrays" (arXiv:1907.10775)
  report e-folding voltages of 25-33 mV in Si/SiGe, target t_c ~ 12 µeV, >= 20 µeV for shuttling,
  electron temperature ~55 mK, lever arm ~0.2. Simulated t_c spans 2 µeV to 0.3 E_m.
- Charge-transfer width across the interdot transition (DiCarlo et al. 2004 form, polarisation
  eps/Omega tanh(Omega/2kT), Omega = sqrt(eps^2 + 4 t_c^2)): FWHM of the derivative is 3.07 t_c
  for t_c >> kT and 3.53 kT for t_c << kT. ChargeCell reports FWHM / tie-bar length (both in
  volts), which is voltage-scale free, and converts to µeV only with user-supplied lever arms.
- Reservoir tunnel rates depend exponentially on the tunnel gate. The simulator draws the gate
  voltage per decade of rate as 0.2-1.2 plunger addition voltages, the T-to-plunger lever-arm
  ratio as 0.2-0.7 (tilted loading lines), and the distance from "electrons follow the sweep"
  (Gamma tau = 1) to lifetime broadening (hbar Gamma = kT) as log10(kT tau / hbar) = 4.5-7
  decades (microsecond-to-millisecond pixels at 50-300 mK).
- Everything is drawn relative to each dot's addition voltage, and practice devices get a random
  overall voltage scale (addition voltages from a few mV to a few hundred mV), so nothing depends
  on one technology's voltages.

### Implications that were adopted
- **Counting needs an anchor.** A pairwise (1,1) is really (1,1,N₃), and without the empty
  region the count is unknown. FOUND requires both dots anchored.
- **Strong tunnel coupling** means curved interdot lines, so the simulator includes tunnel
  coupling growing with occupancy, and the lattice fit tolerates staircases.
- **The sensor is a dot.** Losing the flank, crossing its own Coulomb peak, and compensation
  drift cause the main uninterpretable failures, so there are explicit `sensor_insensitive`
  reasons and a `sensor` line family.
- **Spectator lines** are a third line family, not noise.
- **Low disorder** justifies device priors and history-based spacing.
- The outcome + reason + keypoints output is in the spirit of HRL's keypoint graphs and node
  grades.

- PvT and tie-bar models (built 2026-09-30, see DESIGN §8 and §9): separate models per scan kind as
  in HRL's pipeline, with keypoints (loading lines, operating point, triple points, tie bar,
  readout point) in the protocol response.
- An automation tree (built 2026-09-30, DESIGN §12, `chargecell/runs.py`): every scan, analysis,
  advice and reported action of a tune-up recorded as a graded tree, with statistics across runs.

### Proposed but not built (see HANDOFF)
- A `FOUND_SPECTATOR_UNVERIFIED` status and cross-pair consistency (P1–P2, P2–P3, P1–P3).
- A simulator backend comparison with QDarts (finite tunnel coupling, non-constant charging
  energies, sensor dots) or qarray.
- Reservoir-starved interior dots (missing or slow reservoir lines) in simulation.

## 2. Earlier design discussion (first turn of the original session)

- Outputs: status + reason code + (1,1) polygon + suggested next-window shift.
- Identifying (1,1) is a **counting problem**: it requires an anchoring empty region.
- **Precision first for FOUND**: a false FOUND is worse than a missed one.
- Per-pixel segmentation (U-Net) trained on simulations, with ensemble disagreement for
  abstention.

## 3. spinQICK (not used)

spinQICK (github.com/HRL-Laboratories/spinqick), checked in September 2026, has **no device
simulator**; its only "simulation" is MESA forecasting. It therefore cannot generate synthetic
training data. An early version of ChargeCell imported spinQICK netCDF files and exported
spinQICK scripts; both were removed in favour of the backend-neutral chargecell/1 protocol
(`docs/PROTOCOL.md`). A spinQICK user can send its scans through the protocol like any other
backend.
