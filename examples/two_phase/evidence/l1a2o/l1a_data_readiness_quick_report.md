# L1A-2o L1A data-readiness exit audit

- **Profile:** `quick`
- **Headline verdict:** `L1B_DATA_NOT_READY`
- **SOLVER_SURROGATE_DATA_READY:** `FAIL`
- **PHYSICAL_PUBLICATION_DATA_READY:** `FAIL`
- **L1A_STATUS:** `BLOCKED`
- **Solver contract:** `11` (unchanged)
- **git SHA:** `710aa0ed9a69`

## Category statuses

| Category | Status |
---|---
| `SOLVER_NUMERICAL_STABILITY` | `PASS` |
| `FORMAL_MASS_CONSERVATION` | `PASS` |
| `SPATIAL_REFINEMENT` | `FAIL` |
| `TEMPORAL_REFINEMENT` | `FAIL` |
| `IMPACT_CONTACT_GAP` | `PASS_WITH_CAVEAT` |
| `WETTING_RESIDUAL_RELEVANCE` | `UNMEASURED` |
| `PHI_SAMPLE_FIDELITY` | `PASS` |
| `VELOCITY_SAMPLE_FIDELITY` | `PASS` |
| `GEOMETRY_SAMPLE_FIDELITY` | `PASS` |
| `FRAME_CADENCE` | `PASS` |
| `GENERATOR_ACCEPTANCE` | `PASS` |
| `DATASET_LINEAGE_PER_FILE` | `PASS` |
| `FRESH_TRAINING_AGGREGATE_LINEAGE` | `UNMEASURED` |
| `SIMPLE_SURFACE_COVERAGE` | `PASS` |
| `COMPLEX_SURFACE_CANARY` | `PASS` |
| `EXTERNAL_DYNAMIC_VALIDATION` | `UNMEASURED` |

## Canary acceptance

- `flat_quick__flat_We=100.0_cos_theta=0.5`: accepted=True dt=0.002 horizon=0.08 frames=2
- `pillar_quick__pillars_R=0.65_Re=120.0_We=120.0_cos_theta=-0.25_dt=0.002_eps_factor=2.0_height=0.45_impact_gap_eps=0.75_n_pillars=4_u_impact=1.0_velocity_mode=uniform_width=0.45`: accepted=True dt=0.002 horizon=0.08 frames=2
- `complex_quick__random_pillars_R=0.65_Re=120.0_We=120.0_cos_theta=0.0_dt=0.002_eps_factor=2.0_height_range=(0.3, 0.55)_impact_gap_eps=0.75_n_pillars=5_u_impact=1.0_velocity_mode=uniform_width_range=(0.4, 0.55)`: accepted=True dt=0.002 horizon=0.08 frames=2

## Impact contact gap (I-CONTACT-GAP)

- classification: `DIFFUSE_CONTACT_ONLY`
- per-resolution: `{"production": "DIFFUSE_CONTACT_ONLY", "spatial_refined": "DIFFUSE_CONTACT_ONLY", "temporal_refined": "DIFFUSE_CONTACT_ONLY"}`

## Refinement (existing provisional 3% key-observable target)

- spatial (N=144 vs N=192): at_target=['drop_vertical_extent', 'contact_line_right'] above_target=['formal_mass', 'spread_width', 'beta', 'centroid_y', 'contact_line_left', 'max_speed']
- temporal (dt vs dt/2): at_target=['formal_mass', 'spread_width', 'beta', 'drop_vertical_extent', 'centroid_y', 'contact_line_left', 'contact_line_right'] above_target=['max_speed']

## Sample representation

- `phi`: worst ratio class `SUBDOMINANT` ratio_max=8.818660912954379e-07
- `u`: worst ratio class `SUBDOMINANT` ratio_max=0.0004660392032103827
- `v`: worst ratio class `SUBDOMINANT` ratio_max=0.00045961778626656233

## Blockers

- `IMPACT-PHI-OVERSHOOT`: new_target_critical_blocker_created_in_L1A2o -> TARGET_CRITICAL
- `W-CONTACT-ANGLE`: open -> BOUNDED_CAVEAT
- `I-CONTACT-GAP`: open -> BOUNDED_CAVEAT
- `P-VARDENS-PROJ`: open -> NOT_MATERIAL_TO_CURRENT_L1B_TASK
- `P-CAP-RHO`: open -> NOT_MATERIAL_TO_CURRENT_L1B_TASK
- `P-VARVISC`: open -> NOT_MATERIAL_TO_CURRENT_L1B_TASK
- `N-DT`: open_until_measured_here -> BOUNDED_CAVEAT
- `BC-Y-PERIODIC`: open -> NOT_MATERIAL_TO_CURRENT_L1B_TASK
- `N-INACTIVE-PHASE-STATE-COUPLING`: confirmed_problem_in_contract_v11 -> BOUNDED_CAVEAT
- `D-FRESH-TRAIN-CONTRACT`: open_l1b_blocker -> UNMEASURED

## L1B envelope

- generation_allowed: False
- caveats: ['IMPACT_CONTACT_GAP', 'WETTING_RESIDUAL_RELEVANCE', 'FRESH_TRAINING_AGGREGATE_LINEAGE', 'EXTERNAL_DYNAMIC_VALIDATION']

> This stage is a decision stage: no solver, threshold, schema, or cadence change was made.
