# L1A-2q — temporal convergence and time-integrator closure

- git sha: `1be9e031a24af902c2a5d645efcb12e350cbfc68` · solver contract: **12** · default policy: `impact_phase_cap_dx2_v1`
- frozen gates: refinement 0.03, overshoot 0.02 (never relaxed) · frozen peak window 0.16–0.48

## Verdict (B27/B28)

- temporal verdict: **NONASYMPTOTIC_OR_MULTIPLE_SCALES**
- selected action: **ADDITIONAL_TARGETED_DIAGNOSTIC**
- L1B exit verdict: **L1B_DATA_NOT_READY**
- note: refinement differences do not decrease consistently (B27); named next diagnostic: B17 predictor/pressure/projected-velocity decomposition of the start-up transient (t<0.2), where the projection destroys the uniform impact impulse — the max_speed key observable never enters a dt-convergent regime (mechanism matrix)

## Observed orders (B9)

| quantity | window order | status |
|---|---|---|
| flat_we100_ct050:spread_width |  | UNDETERMINED |
| flat_we100_ct050:drop_vertical_extent |  | NONMONOTONE |
| flat_we100_ct050:centroid_y | 0.86 | FIRST_ORDER_LIKE |
| flat_we100_ct050:max_speed |  | NONMONOTONE |
| flat_we200_ct000:spread_width |  | UNDETERMINED |
| flat_we200_ct000:drop_vertical_extent |  | NONMONOTONE |
| flat_we200_ct000:centroid_y | 0.86 | FIRST_ORDER_LIKE |
| flat_we200_ct000:max_speed |  | NONMONOTONE |
| pillar_training:spread_width |  | UNDETERMINED |
| pillar_training:drop_vertical_extent |  | NONMONOTONE |
| pillar_training:centroid_y | 0.88 | FIRST_ORDER_LIKE |
| pillar_training:max_speed |  | NONMONOTONE |
| complex_heldout:spread_width |  | UNDETERMINED |
| complex_heldout:drop_vertical_extent | 1.47 | NOT_IN_ASYMPTOTIC_REGIME |
| complex_heldout:centroid_y | 0.87 | FIRST_ORDER_LIKE |
| complex_heldout:max_speed |  | NONMONOTONE |

## Peak attribution (B10/B11)

- global max_speed over all horizon frames (start-up transient): dt=0.002: 0.2484 @ t=0.08, dt=0.001: 0.1218 @ t=0.08, dt=0.0005: 0.0302 @ t=0.08
- global peak amplitude E1=1.039 E2=3.034 p_obs=-1.55
- flat_we100_ct050 E1_dt0.002_vs_dt0.001: peak 0.1235 vs 0.0342 (rel 2.6128), t_peak 0.1620 vs 0.1610 (shift +0.00100), pointwise 0.0413, aligned 0.0412
- flat_we100_ct050 E2_dt0.001_vs_dt0.0005: peak 0.0342 vs 0.0114 (rel 2.0103), t_peak 0.1610 vs 0.4800 (shift -0.31900), pointwise 0.0097, aligned 0.0067
- flat_we200_ct000 E1_dt0.002_vs_dt0.001: peak 0.1184 vs 0.0290 (rel 3.0790), t_peak 0.1620 vs 0.1610 (shift +0.00100), pointwise 0.0413, aligned 0.0412
- flat_we200_ct000 E2_dt0.001_vs_dt0.0005: peak 0.0290 vs 0.0025 (rel 10.7261), t_peak 0.1610 vs 0.1605 (shift +0.00050), pointwise 0.0086, aligned 0.0086
- pillar_training E1_dt0.002_vs_dt0.001: peak 0.1970 vs 0.0682 (rel 1.8887), t_peak 0.1620 vs 0.1610 (shift +0.00100), pointwise 0.0607, aligned 0.0605
- pillar_training E2_dt0.001_vs_dt0.0005: peak 0.0682 vs 0.0282 (rel 1.4190), t_peak 0.1610 vs 0.1605 (shift +0.00050), pointwise 0.0146, aligned 0.0146
- complex_heldout E1_dt0.002_vs_dt0.001: peak 0.3228 vs 0.1310 (rel 1.4639), t_peak 0.1620 vs 0.1610 (shift +0.00100), pointwise 0.1013, aligned 0.1008
- complex_heldout E2_dt0.001_vs_dt0.0005: peak 0.1310 vs 0.0398 (rel 2.2899), t_peak 0.1610 vs 0.1605 (shift +0.00050), pointwise 0.0333, aligned 0.0332

## Mechanism matrix (B26)

- `FIRST_ORDER_TEMPORAL_TRUNCATION`: **SUSPECTED**
- `TIME_SPLITTING_OR_COUPLING_LIMITATION`: **SUSPECTED**
- `PHASE_ADVECTION_TEMPORAL_ERROR`: **NOT_TESTED**
- `CH_TEMPORAL_ERROR`: **NOT_TESTED**
- `CAPILLARY_MOMENTUM_TEMPORAL_ERROR`: **NOT_TESTED**
- `PROJECTION_TEMPORAL_ERROR`: **SUSPECTED**
- `BRINKMAN_TIME_RESPONSE`: **FALSIFIED**
- `PEAK_TIMING_MISALIGNMENT`: **FALSIFIED**
- `PEAK_AMPLITUDE_NONCONVERGENCE`: **SUPPORTED**
- `SPATIAL_TEMPORAL_ERROR_CONFOUNDING`: **SUPPORTED**
- `MULTIPLE_TIME_SCALES`: **SUPPORTED**
- `NEAR_ZERO_NORMALIZATION_AMPLIFICATION`: **SUPPORTED**

## Spatial recheck (B21)

- spread_width: rel RMS 0.0303 (gate 0.03)
- beta: rel RMS 0.0303 (gate 0.03)
- drop_vertical_extent: rel RMS 0.0151 (gate 0.03)
- centroid_y: rel RMS 0.0270 (gate 0.03)
- centroid_x: rel RMS 0.0000 (gate 0.03)
- max_speed: rel RMS 0.0522 (gate 0.03)
- formal_mass: rel RMS 0.0052 (gate 0.03)

## Cost (B19)

- dt=0.002: projected T=8 90s (1.00x coarsest)
- dt=0.001: projected T=8 161s (1.80x coarsest)
- dt=0.0005: projected T=8 199s (2.22x coarsest)
