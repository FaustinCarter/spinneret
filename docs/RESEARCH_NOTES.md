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

### Proposed but not built (see HANDOFF §4.5)
- A `FOUND_SPECTATOR_UNVERIFIED` status and cross-pair consistency (P1–P2, P2–P3, P1–P3).
- PvT and tiebar models; keypoints for triple points and interdot endpoints.
- An automation tree / action log.
- A simulator backend comparison with QDarts (finite tunnel coupling, non-constant charging
  energies, sensor dots) or qarray.
- Reservoir-starved interior dots (missing or slow reservoir lines) in simulation.

## 2. Earlier design discussion (first turn of the original session)

- Outputs: status + reason code + (1,1) polygon + suggested next-window shift.
- Identifying (1,1) is a **counting problem**: it requires an anchoring empty region.
- **Precision first for FOUND**: a false FOUND is worse than a missed one.
- Per-pixel segmentation (U-Net) trained on simulations, with ensemble disagreement for
  abstention.

## 3. spinQICK facts (from its source, github.com/HRL-Laboratories/spinqick)

Checked by cloning the repository in September 2026. **Re-verify against the user's installed
version.**

- **No device simulator.** The only "simulation" is MESA forecasting. Synthetic training data
  cannot come from spinQICK.
- `TuneElectrostatics.gvg_dc(g_gates=(['P1'], ['P2']), g_range=((x0, x1, nx), (y0, y1, ny)),
  measure_buffer, compensate='M1')`. Ranges are **relative to the present DC point**; **y is
  the fast (inner) axis**. `gvg_baseband` takes absolute ranges.
- `vdc.set_dc_voltage(volts, gate)` sets a DC voltage (no ramp; checks `max_v`).
- `vdc.set_dc_voltage_compensate(volts, gates, iso_gates)` takes **absolute** target volts. It
  computes deltas from `get_dc_voltage`, solves the compensation with the cross-coupling matrix
  from the hardware config (raises if that matrix is missing), and sets all gates.
- `retune_dcs(m_dot, m_range, measure_buffer, set_v)` retunes the sensor; `tune_mz` also exists.
- **netCDF layout** (`SpinqickData.save_data`, filename `<unix timestamp><experiment_name>.nc`):
  - root attributes: `timestamp` (unix seconds, int), `experiment_name` (e.g. `_gvg_dc`),
    `cfg` (JSON), `cfg_type`, `spinqick_version`, `voltage_state` (JSON gate -> V);
  - group `swept_variables/x`, `swept_variables/y`: one variable per swept gate (volts) with
    attributes `ax_dim` and `loop_no` (the higher loop_no is the inner/fast loop); dimensions
    `x_dim`, `y_dim` are created at the **root**;
  - `raw_data_0`: dims (reps, triggers, [avgs], x, y, IQ);
  - `analyzed_data/analyzed_0`: dims (reps, triggers, x, y), with a `units` attribute;
  - data are indexed **[..., x, y]**. ChargeCell stores (y, x).
- `load_spinqick_nc` prefers analysed data, falls back to |IQ| of raw data, averages all
  non-x/y dimensions, and imports multi-gate (virtual) axes using the first gate's voltages
  (the others go into `extra`).
