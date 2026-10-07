# L1A-2m stationarity metric-domain forensics

- **Status:** `complete`
- **Profile:** `forensic`
- **Final verdict:** `INACTIVE_STATE_COUPLING`
- **Solver contract:** `11` (unchanged)
- **Production semantics changed:** `False`
- **Diagnostic-only:** yes; the production phase-rate gate, thresholds and semantics are untouched.

## Provenance states

### authority_060

- step: `50000`; acceptance: `accepted_exact_rehydration`; per-field hash match to frozen L1A-2k: `True`
- config fingerprint: `2563f20daad7bcdcbb11150fc54daa6a66ef3ff04326b7434d01c0560d5c495f`

### control_090

- step: `27200`; acceptance: `accepted_exact_rehydration`; per-field hash match to frozen L1A-2k: `True`
- config fingerprint: `c7857ca99258032c79ba00bfb9de8d24e5e5ef905fd551c854625a81d9ecdea0`

### control_150

- step: `100000`; acceptance: `accepted_exact_rehydration`; per-field hash match to frozen L1A-2k: `True`
- config fingerprint: `82a53cc0e10d6ec8000660b6f4e4632bf69c69c61c566ee43eedb8f15677c730`

### ch_only_equilibrium_060

- step: `180000`; acceptance: `accepted_exact_rehydration`; per-field hash match to frozen L1A-2k: `True`
- config fingerprint: `e17af41512ac417b8e91d41c896c5478ca31c0e62544c71863e054447065e4a9`

## Support-class contribution ledger (60° authority endpoint)

| Class | E | f | cells | nonzero-rate cells | max abs r | RMS r |
|---|---:|---:|---:|---:|---:|---:|
| ZERO_VOLUME | 0.000000e+00 | 0.000000e+00 | 640 | 0 | 0.000000e+00 | 0.000000e+00 |
| PARTIAL_VOLUME | 1.474712e-09 | 2.076892e-02 | 128 | 128 | 2.769163e-04 | 7.241147e-05 |
| FULL_VOLUME | 6.953102e-08 | 9.792311e-01 | 15616 | 15616 | 3.725848e-04 | 4.501561e-05 |

- E_all: `7.100573e-08`; R_prod: `2.664690e-04`

## Production vs shadow physical-domain metric

| Window | R_prod mean/median | R_V mean/median | ZERO f mean | PARTIAL f mean | FULL f mean |
|---|---|---|---|---|---|
| authority_060 | 2.901865e-04 / 2.882534e-04 | 2.895921e-04 / 2.876473e-04 | 0.000e+00 | 1.258e-02 | 9.874e-01 |
| control_090 | 7.290469e-04 / 7.254069e-04 | 7.260448e-04 / 7.224073e-04 | 0.000e+00 | 2.471e-02 | 9.753e-01 |
| control_150 | 4.158291e-04 / 4.155903e-04 | 4.155051e-04 / 4.152665e-04 | 0.000e+00 | 4.673e-03 | 9.953e-01 |
| ch_only_equilibrium_060 | 8.547648e-05 / 8.535867e-05 | 8.528485e-05 / 8.516784e-05 | 0.000e+00 | 1.340e-02 | 9.866e-01 |

## Matched-control calibration (R_V, report-only)

- authority/control median ratio: `0.692681282215336`
- effect size d: `-0.4853625162690682`; control-like: `True`
- No R_V threshold is proposed or applied in this stage.

## Final verdict

- `INACTIVE_STATE_COUPLING`
- rule: section 25: counterfactual changes confined to V=0 phi measurably alter physical operators or the next physical state

### Decision gates

- zero_cells_materially_inflate_R_prod: `False`
- inactive_cells_causally_inactive: `False`
- physical_activity_control_like: `True`
- remaining_criteria_show_no_true_nonstationarity: `True`
- robust_across_late_window: `True`

## Mechanism matrix

| Candidate | Status | Scope |
|---|---|---|
| `CLASSIFIER_DOMAIN_MISMATCH` | `SUPPORTED` | production full-grid dxdy norm vs exact control-volume support; the metric measures a different domain than the conserved state |
| `ZERO_VOLUME_RATE_DOMINANCE` | `FALSIFIED` | max ZERO_VOLUME energy fraction over the late window = 0.0 |
| `PARTIAL_CELL_WEIGHTING_MISMATCH` | `SUPPORTED` | partial cells carry a full dx*dy weight in R_prod but only V_i < dx*dy in R_V |
| `INACTIVE_STATE_COUPLING` | `SUPPORTED` | sections 13/14 counterfactual operator/one-step evidence on physical cells and faces |
| `TRUE_PHYSICAL_PHASE_NONSTATIONARITY` | `FALSIFIED` | matched 60/90/150/CH-only comparison restricted to V_i > 0 (sections 8/26) |
| `ANGLE_WINDOW_ALIASING` | `FALSIFIED` | angle window spread under the production cadence vs denser available sampling |
| `ANGLE_TRUE_RESIDUAL_MOTION` | `SUSPECTED` | late-window angle slope and left/right contact-line drift; no gate change |
| `CLASSIFIER_FALSE_NEGATIVE` | `FALSIFIED` | section 24 decision rule |
| `MIXED_CLASSIFIER_AND_PHYSICS` | `FALSIFIED` | section 27 decision rule |

## Unmeasured sections

- `M_4x_closure`: outside L1A-2m scope (frozen M = M_ref)
- `dt_half`: outside L1A-2m scope (frozen dt = 0.004)

## Blockers

- `N-STATIONARITY-METRIC-DOMAIN` = {'entered_as': 'suspected_problem', 'exit_status': 'confirmed_inactive_state_coupling', 'resolved_in_this_stage': False}
- `N-CH-MASS-PRECISION` = resolved_in_contract_v11
- `N-WALL-ALIGNMENT-TRANSPORT-DOMAIN` = resolved_in_contract_v9
- `N-CAPILLARY-PRESSURE-BALANCE` = structural background
- `W-CONTACT-ANGLE` = open
