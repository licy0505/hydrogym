# L1A-2j: CHNS nonstationarity forensic audit

- **Profile:** `forensic`
- **Status:** `complete`
- **Git SHA:** `f4b10e20a868960d075b5975080e9f821ee171e8`
- **Solver contract:** `11` (required 11)
- **Scope:** Diagnostic/forensic only. Production solver semantics, defaults, contract, thresholds, and acceptance conditions are unchanged.

## Frozen production configuration

| Case | Settings |
|---|---|
| authority_60 | N=128, eps=0.09375 (2dx), dt=0.004, M=0.002, target=60° |
| control_150 | N=128, eps=0.09375 (2dx), dt=0.004, M=0.002, target=150° |
| control_90 | N=128, eps=0.09375 (2dx), dt=0.004, M=0.002, target=90° |

## Classification

- **60° late-time phenomenology:** `MULTIPLE_PHENOMENA` — `ASSIGNED`
- **Mechanism:** `CAPILLARY_PRESSURE_IMBALANCE` — `ASSIGNED`

### Phenomenology evidence labels

| Label | Status | Evidence |
|---|---|---|
| `DECAYING_BUT_SLOW` | `SUSPECTED` | rolling 40k-50k kinetic-energy and phase-rate trends are used; this is not inferred from a single endpoint |
| `ANGLE_LIMIT_CYCLE` | `SUSPECTED` | dominant angle period=39.865264892578125; peak-to-peak=0.20864744043329608; estimated cycles=0.999000999000999; spectral power fraction=0.8900836953626682 |
| `CONTACT_LINE_STICK_SLIP` | `NOT_TESTED` | every-step burst detector: {'status': 'measured', 'episodes': 0, 'dwell_intervals': 1999, 'jump_intervals': 0, 'left_displacement_cells_per_sample_max': 6.451648549917384e-06, 'right_displacement_cells_per_sample_max': 6.45164919887975e-06, 'sample_step_intervals': 1, 'diagnostic_pattern_thresholds': {'dwell_cell_displacement': 0.01, 'jump_cell_displacement': 0.25, 'lookback_samples': 20}, 'rule': 'reported as contact-line stick-slip only when repeated bilateral dwell-to-jump episodes appear in contiguous every-step burst'}; No stick-slip episode was detected; the finite burst does not falsify the broader phenomenology. |
| `PHASE_ONLY_NONSTATIONARITY` | `SUPPORTED` | M_ref freeze-u/CH-only branch: angle delta=-0.004479546532088818, free-energy delta=-0.0005903216113408161, diagnostic convergence=False |
| `HYDRODYNAMIC_NONSTATIONARITY` | `SUPPORTED` | freeze-phi 5k-step kinetic-energy retention=0.9963736174425827; phase field is checked bitwise fixed |
| `STEADY_RECIRCULATION` | `NOT_TESTED` | velocity-reset path starts at u=v=0; max speed 0.0 -> 0.00012431858750156814; production stationarity gate=False; see kinetic-energy trend and controls |
| `NOISY_STATIONARY_PLATEAU` | `NOT_TESTED` | production 200-step gate=False; angle high-cadence peak-to-peak=0.20864744043329608 |
| `CLASSIFIER_FALSE_NEGATIVE` | `NOT_TESTED` | production-cadence gate=False; 10-step sensitivity gate=False; dense gate is not production acceptance |
| `MULTI_FREQUENCY_OSCILLATION` | `NOT_TESTED` | 10-step angle spectrum has 1 peaks at or above 5% power; a single dominant peak is insufficient for a multi-frequency assignment. |

### Mechanism evidence matrix

| Mechanism | Status | Evidence |
|---|---|---|
| `PHASE_KINETICS_LIMITED` | `SUSPECTED` | Matched freeze-u M_ref branch angle drift=0.004479546532088818 deg; diagnostic stationarity=False. A nonconverged phase-only branch with small angle drift is treated as suspected, not falsified. |
| `CAPILLARY_PRESSURE_IMBALANCE` | `SUPPORTED` | same-state, same-step freeze-phi A/B: {'freeze_phi_normal_final_over_initial_kinetic': 0.9963736174425827, 'freeze_phi_capillary_off_final_over_initial_kinetic': 0.00023418578056694827, 'matched_steps': 500, 'diagnostic_only': True}; exact capillary term/work history is available |
| `PRESSURE_PROJECTION_LIMITED` | `FALSIFIED` | projection reduction median=9.310750580000951e-07; pressure-Poisson residual L2 max=3.831301423196701e-07; classifier thresholds unchanged |
| `BRINKMAN_WALL_COUPLING` | `SUSPECTED` | Brinkman damping remains the exact existing implicit factor; term/work, chi>0.5 velocity, wall shells, and contact-line neighborhoods are measured; no parameter change is made; max chi>0.5 speed=0.00012409011633652737; max |Brinkman work proxy|=6.822560568049592e-07; median Brinkman L2=0.008079678908854375 |
| `CONTACT_LINE_PINNING` | `NOT_TESTED` | Every-step burst diagnostic: {'diagnostic_pattern_thresholds': {'dwell_cell_displacement': 0.01, 'jump_cell_displacement': 0.25, 'lookback_samples': 20}, 'dwell_intervals': 1999, 'episodes': 0, 'jump_intervals': 0, 'left_displacement_cells_per_sample_max': 6.451648549917384e-06, 'right_displacement_cells_per_sample_max': 6.45164919887975e-06, 'rule': 'reported as contact-line stick-slip only when repeated bilateral dwell-to-jump episodes appear in contiguous every-step burst', 'sample_step_intervals': 1, 'status': 'measured'}. No stick-slip episode does not rule out static pinning; no separate unpinning-force/threshold test was run. |
| `Y_PERIODIC_TOPOLOGY_COUPLING` | `SUSPECTED` | y-periodic seam v-jump/rms median=0.3031109371160533; periodic-x liquid component counts=[1.0]; seam is not a wall |
| `TIME_SPLITTING_OR_DT_SENSITIVITY` | `NOT_TESTED` | conditional matched-physical-time dt/2 diagnostic was not triggered or requested |
| `MOMENTUM_VISCOSITY_DISCRETIZATION` | `NOT_TESTED` | no viscosity-operator ablation; exact applied nu*lap(u) terms and localization are measured, but no production changes or unsupported inference |
| `VARIABLE_DENSITY_COUPLING` | `NOT_TESTED` | rho(phi) range and overlap are measured; no density-model ablation is permitted in this stage |
| `CAPILLARY_DENSITY_SCALING` | `NOT_TESTED` | production capillary denominator rho_l is preserved; no alternate denominator ablation is run |
| `CLASSIFIER_LOGIC_ONLY` | `NOT_TESTED` | formal and high-cadence production criterion evaluations are both retained; cadence sensitivity cannot close production acceptance |

## Production stationarity decomposition

- Production-cadence 200-step gate: `False`; dense every-10-step calculation is sensitivity only.

| Criterion | Raw | Threshold | Raw / threshold | Pass | Window |
|---|---:|---:|---:|---|---|
| angle_stationarity | 0.10482108682566604 | 0.1 | 1.0482108682566604 | False | 32 samples / 0.05 M·t |
| free_energy_stability | 1.9257243953352265e-05 | 0.0001 | 0.19257243953352265 | True | 32 samples / 0.05 M·t |
| maximum_speed | 0.00012409011633652737 | 0.0005 | 0.24818023267305472 | True | 32 samples / 0.05 M·t |
| phase_rate | 0.05337776457647862 | 0.001 | 53.37776457647862 | False | 32 samples / 0.05 M·t |

## Matched controls

| Target | Steps | Converged | Angle | Max speed |
|---:|---:|---|---:|---:|
| 150° | 100000 | True | 147.7536522851593 | 0.00033555281355144233 |
| 90° | 27200 | True | 89.41024267586671 | 0.000268017597547329 |

## Diagnostic branches

- `freeze_phi`: `measured`; diagnostic only; production acceptance evidence = `False`
- `freeze_phi_capillary_off`: `measured`; diagnostic only; production acceptance evidence = `False`
- `freeze_u_phase_only_M4_diagnostic`: `measured`; diagnostic only; production acceptance evidence = `False`
- `freeze_u_phase_only_Mref`: `measured`; diagnostic only; production acceptance evidence = `False`
- `velocity_reset`: `measured`; diagnostic only; production acceptance evidence = `False`

## Mass, pressure, and frozen blockers

- Formal mass is `sum_i(V_i*phi_i)`: max drift 2.078020484504438e-15 vs diagnostic limit 0.001; pass = `True`.
- `dx*dy*sum(phi)` and legacy hard-mask mass are diagnostic-only and never acceptance gates.
- `N-CH-MASS-PRECISION`: `resolved_in_contract_v11; preserve`
- `N-WALL-ALIGNMENT-TRANSPORT-DOMAIN`: `resolved_in_contract_v9; regression confirmed; preserve`
- `W-CONTACT-ANGLE`: `open through L1A-2j absent dedicated closure`

## Unmeasured sections

- `dt_half_diagnostic`: **unmeasured_conditional_not_triggered**

## Artifacts

- Raw runs, snapshots, manifests, and resumable checkpoints: `artifacts/l1a2j`
- Machine-readable report: `chns_nonstationarity_report.json`
