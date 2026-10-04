# L1A-2g mass-precision audit

- solver contract = 10, implicit phase solver = `weighted_spd_nullspace_preserving_v1`, mass invariant = `componentwise_cutcell_volume`
- conserved quantity: `sum_i V_i phi_i`
- profile = forensic, generated 2026-10-03T08:59:36Z
- **passed: True**

## Checks

| check | passed | detail |
| --- | --- | --- |
| `S_c_is_zero` | True | max|S c| = 0.000e+00 |
| `A_c_equals_c` | True | max|A 1 - 1|_V = 0.000e+00 |
| `flux_telescoping_machine_zero` | True | accumulated face sum: advective = 3.662e-17, CH = 7.802e-19 (of the flux scale); the f32 divergence field reproduces it to 1.167e-09 / 8.564e-09 |
| `pinned_v9_cTy_equals_cTb_on_this_rhs` | True | recorded (the identity holds only to the truncation of the run): cTy - cTb = -2.334e-08 |
| `single_fluid_component` | True | components = 1 (block Q required if > 1) |
| `no_post_step_mass_projection` | True | scanned 10 functions; findings []; legacy-only ['_phase_update: _project_phase_outside_solid (legacy branch only)'] |
| `pinned_v9_M1_at_roundoff_floor` | True | mean = +0.0000 E_round (frac+ 0.000, frac- 0.000, exact-zero 1.000) |
| `pinned_v9_M2_at_roundoff_floor` | True | mean = +0.0021 E_round (frac+ 0.683, frac- 0.317, exact-zero 0.000) |
| `pinned_v9_M3_at_roundoff_floor` | True | mean = +0.0000 E_round (frac+ 0.000, frac- 0.000, exact-zero 1.000) |
| `pinned_v9_M4_at_roundoff_floor` | True | mean = +0.0178 E_round (frac+ 0.983, frac- 0.017, exact-zero 0.000) |
| `production_M1_at_roundoff_floor` | True | mean = +0.0000 E_round (frac+ 0.000, frac- 0.000, exact-zero 1.000) |
| `production_M2_at_roundoff_floor` | True | mean = +0.0040 E_round (frac+ 0.717, frac- 0.283, exact-zero 0.000) |
| `production_M3_at_roundoff_floor` | True | mean = +0.0000 E_round (frac+ 0.000, frac- 0.000, exact-zero 1.000) |
| `production_M4_at_roundoff_floor` | True | mean = +0.0000 E_round (frac+ 0.000, frac- 0.000, exact-zero 1.000) |
| `pinned_v9_first_stage_above_roundoff_floor` | True | first stage above the 0.05 E_round floor = M5 (KRYLOV_NULL_MODE expected before the transform stages) |
| `pinned_v9_inverse_transform_is_systematic` | True | M6 (y->phi): mean = +0.1613 E_round, frac+ 1.000, frac- 0.000 |
| `production_transform_stages_are_identity` | True | contract v10 solves in phi: M4 == M3 and M6 == M5 exactly |
| `production_solve_keeps_the_conserved_mode` | True | per-substep M5 mean = -0.0009 E_round, frac+ = 0.450 (no sign preference: in float32 the mode is kept to the round-off floor, in float64 to 2.3e-16 relative -- see drift_matrix_float64) |
| `drift_matrix_ran` | True | every matrix row produced a trajectory |
| `float64_reference_is_reduction_independent` | True | the two independent references (host math.fsum and the float64 device reduction) of M agree to 9.7e-10 E_round (one float64 ulp of M); the working-dtype reduction differs by -0.278 E_round on one state and is reported, never used as the reference |
| `working_dtype_reduction_is_the_reported_risk` | True | the working-dtype reduction of M differs from the float64 reference by -0.278 E_round on one state; the audit therefore reports the float64 value |

## M0-M8 ledger (per-substep mean, units of `E_round`)

| case | solver | M1 adv | M2 CH | M3 rhs | M4 transform | M5 solve | M6 inverse | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N48_pinned_v9 | 2.9 it | +0.0000 | -0.0007 | +0.0000 | +0.0000 | -0.0448 | +0.0000 | -0.0456 |
| N48_production | 1.9 it | +0.0000 | -0.0012 | +0.0000 | +0.0000 | +0.0031 | +0.0000 | +0.0019 |
| N128_pinned_v9 | 8.8 it | +0.0000 | +0.0021 | +0.0000 | +0.0178 | -0.1496 | +0.1613 | +0.0317 |
| N128_production | 7.8 it | +0.0000 | +0.0040 | +0.0000 | +0.0000 | -0.0009 | +0.0000 | +0.0030 |

## Drift matrix

| N | dtype | rtol | solver | steps | relative drift | per substep (E_round) | frac+ | 150k extrapolation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 48 | float32 | 1e-04 | pinned_v9 | 1000 | -7.643e-04 | -6.1513 | 0.000 | -1.100e-01 |
| 48 | float32 | 1e-04 | production | 1000 | -4.328e-06 | -0.0363 | 0.349 | -6.498e-04 |
| 48 | float32 | 1e-06 | pinned_v9 | 1000 | -1.613e-05 | -0.1348 | 0.096 | -2.411e-03 |
| 48 | float32 | 1e-06 | production | 1000 | -4.309e-07 | -0.0037 | 0.505 | -6.632e-05 |
| 48 | float32 | 1e-08 | pinned_v9 | 1000 | -9.211e-06 | -0.0774 | 0.218 | -1.384e-03 |
| 48 | float32 | 1e-08 | production | 1000 | -3.341e-06 | -0.0281 | 0.372 | -5.026e-04 |
| 128 | float32 | 1e-04 | pinned_v9 | 1000 | -9.438e-03 | -79.2174 | 0.000 | -1.417e+00 |
| 128 | float32 | 1e-04 | production | 1000 | -4.491e-07 | -0.0038 | 0.449 | -6.780e-05 |
| 128 | float32 | 1e-06 | pinned_v9 | 1000 | +1.528e-05 | +0.1283 | 0.956 | +2.294e-03 |
| 128 | float32 | 1e-06 | production | 1000 | -1.382e-06 | -0.0116 | 0.363 | -2.071e-04 |
| 128 | float32 | 1e-08 | pinned_v9 | 1000 | +7.733e-05 | +0.6490 | 1.000 | +1.161e-02 |
| 128 | float32 | 1e-08 | production | 1000 | -1.265e-06 | -0.0106 | 0.383 | -1.899e-04 |
| 128 | float64 | 1e-06 | pinned_v9 | 1000 | -8.067e-05 | -363393557.1602 | 0.000 | -1.210e-02 |
| 128 | float64 | 1e-06 | production | 1000 | +2.309e-16 | +0.0005 | 0.184 | +1.733e-14 |
| 128 | float64 | 1e-08 | pinned_v9 | 1000 | -4.587e-07 | -2065143.8651 | 0.000 | -6.878e-05 |
| 128 | float64 | 1e-08 | production | 1000 | +2.309e-16 | +0.0010 | 0.166 | +3.467e-14 |

## First-loss verdict

**MULTIPLE_CONTRIBUTORS** -- first systematic stage M4->M5 (KRYLOV_NULL_MODE) -- confirmed by the tolerance sweep; the transform stages are the largest tolerance-independent terms

- `KRYLOV_NULL_MODE` at M4->M5: -0.1496 E_round (frac+ 0.000, tolerance controlled: True) -- the truncated conserved mode c^T y != c^T b; the sign and size follow the stopping tolerance, and it is the first stage above the round-off floor
- `Y_TO_PHI_TRANSFORM` at M5->M6: +0.1613 E_round (frac+ 1.000, tolerance controlled: False) -- y * fl(1/sqrt(V)) is a double rounding with a one-sided, data-independent bias (+1.8e-8 uniform random, +2.0e-8 on the field, +2.2e-8 near 1): it does not respond to the tolerance and it partially cancels the Krylov term
- `PHI_TO_Y_TRANSFORM` at M3->M4: +0.0178 E_round (frac+ 0.983, tolerance controlled: False) -- fl(sqrt(V))^2 != V on the cut cells only (128 of 15744 live cells at N=128), so the y-space mass weight is not the physical weight: measured as a +2.84e-9 relative weight-definition mismatch; the ledger's M3->M4 stage stays at the round-off floor because the product rounding partially cancels

Repair taken: remove the transform from the mass-carrying path (contract v10).
Repair rejected: nullspace-preserving c-perp Krylov recurrence alone (it addresses the second mechanism).
