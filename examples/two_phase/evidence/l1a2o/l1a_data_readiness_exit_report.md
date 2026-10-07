# L1A-2o L1A data-readiness exit audit

- **Profile:** `forensic`
- **Headline verdict:** `L1B_DATA_NOT_READY`
- **SOLVER_SURROGATE_DATA_READY:** `FAIL`
- **PHYSICAL_PUBLICATION_DATA_READY:** `FAIL`
- **L1A_STATUS:** `BLOCKED`
- **Solver contract:** `11` (unchanged)
- **git SHA:** `d4adee6de874`

## Category statuses

| Category | Status |
---|---
| `SOLVER_NUMERICAL_STABILITY` | `FAIL` |
| `FORMAL_MASS_CONSERVATION` | `PASS` |
| `SPATIAL_REFINEMENT` | `FAIL` |
| `TEMPORAL_REFINEMENT` | `FAIL` |
| `IMPACT_CONTACT_GAP` | `PASS` |
| `WETTING_RESIDUAL_RELEVANCE` | `PASS` |
| `PHI_SAMPLE_FIDELITY` | `PASS` |
| `VELOCITY_SAMPLE_FIDELITY` | `PASS` |
| `GEOMETRY_SAMPLE_FIDELITY` | `PASS` |
| `FRAME_CADENCE` | `PASS` |
| `GENERATOR_ACCEPTANCE` | `FAIL` |
| `DATASET_LINEAGE_PER_FILE` | `PASS` |
| `FRESH_TRAINING_AGGREGATE_LINEAGE` | `UNMEASURED` |
| `SIMPLE_SURFACE_COVERAGE` | `FAIL` |
| `COMPLEX_SURFACE_CANARY` | `PASS` |
| `EXTERNAL_DYNAMIC_VALIDATION` | `UNMEASURED` |

## Canary acceptance

- `flat_impact_canary__flat_We=100.0_cos_theta=0.5`: accepted=False dt=0.004 horizon=8.0 frames=100
- `flat_impact_canary__flat_We=100.0_cos_theta=0.0`: accepted=False dt=0.004 horizon=8.0 frames=100
- `flat_impact_canary__flat_We=100.0_cos_theta=-0.5`: accepted=False dt=0.004 horizon=8.0 frames=100
- `flat_impact_canary__flat_We=200.0_cos_theta=0.5`: accepted=False dt=0.004 horizon=8.0 frames=100
- `flat_impact_canary__flat_We=200.0_cos_theta=0.0`: accepted=False dt=0.004 horizon=8.0 frames=100
- `flat_impact_canary__flat_We=200.0_cos_theta=-0.5`: accepted=False dt=0.004 horizon=8.0 frames=100
- `pillar_training_canary__pillars_We=100.0_cos_theta=-0.5_height=0.4_n_pillars=4_width=0.3`: accepted=True dt=0.004 horizon=8.0 frames=100
- `complex_heldout_canary__random_pillars_We=100.0_cos_theta=-0.5_n_pillars=7_width_range=(0.2, 0.4)`: accepted=False dt=0.004 horizon=8.0 frames=100

## Frozen 60 deg residual vs the L1B horizon

- frozen window t in [160.0, 200.0], nominal L1B horizon t <= 8.0
- horizon/window-start ratio: 0.05
- residual (saved grid, frozen window) vs impact frame signal:
  - `phi`: residual=1.920e-03 signal=8.019e-04 ratio=2.3945096546409212
  - `u`: residual=1.103e-05 signal=5.338e-05 ratio=0.2066900015826551
  - `v`: residual=1.386e-05 signal=4.752e-05 ratio=0.29164325344786957

## Impact contact gap (I-CONTACT-GAP)

- classification: `CONTACT_ESTABLISHED`
- per-resolution: `{"production": "CONTACT_ESTABLISHED", "spatial_refined": "CONTACT_ESTABLISHED", "temporal_refined": "CONTACT_ESTABLISHED"}`

## Refinement (existing provisional 3% key-observable target)

- spatial (N=144 vs N=192): at_target=['formal_mass', 'contact_line_left', 'contact_line_right'] above_target=['spread_width', 'beta', 'drop_vertical_extent', 'centroid_y', 'max_speed']
- temporal (dt vs dt/2): at_target=['formal_mass', 'contact_line_right'] above_target=['spread_width', 'beta', 'drop_vertical_extent', 'centroid_y', 'contact_line_left', 'max_speed']

## Sample representation

- `phi`: worst ratio class `SUBDOMINANT` ratio_max=3.4401217660808606e-06
- `u`: worst ratio class `SUBDOMINANT` ratio_max=0.0008061881455877122
- `v`: worst ratio class `SUBDOMINANT` ratio_max=0.0006713666799863075

## Blockers

- `IMPACT-PHI-OVERSHOOT`: new_target_critical_blocker_created_in_L1A2o -> TARGET_CRITICAL
- `W-CONTACT-ANGLE`: open -> BOUNDED_CAVEAT
- `I-CONTACT-GAP`: open -> BOUNDED_CAVEAT
- `P-VARDENS-PROJ`: open -> NOT_MATERIAL_TO_CURRENT_L1B_TASK
- `P-CAP-RHO`: open -> NOT_MATERIAL_TO_CURRENT_L1B_TASK
- `P-VARVISC`: open -> NOT_MATERIAL_TO_CURRENT_L1B_TASK
- `N-DT`: open_until_measured_here -> TARGET_CRITICAL
- `BC-Y-PERIODIC`: open -> NOT_MATERIAL_TO_CURRENT_L1B_TASK
- `N-INACTIVE-PHASE-STATE-COUPLING`: confirmed_problem_in_contract_v11 -> BOUNDED_CAVEAT
- `D-FRESH-TRAIN-CONTRACT`: open_l1b_blocker -> UNMEASURED

## L1B envelope

- generation_allowed: False
- caveats: ['FRESH_TRAINING_AGGREGATE_LINEAGE', 'EXTERNAL_DYNAMIC_VALIDATION']

> This stage is a decision stage: no solver, threshold, schema, or cadence change was made.
