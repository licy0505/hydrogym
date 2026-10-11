# L1A-2u final report - radial-envelope incident seed and Brinkman-dt deconfounding

- stage `L1A-2u` version `l1a2u_v1`, profile `forensic`, diagnostic only, no promotion
- status `L1A2U_DIAGNOSTIC_COMPLETE_NO_PROMOTION`; mechanism `INTERIOR_SPEED_INADMISSIBLE`

## A. Scope, authority and provenance
- verified main `09e2993b8da478d79e768054194201af57121784`, branch `arena/02b6c71b-hydrogym`, head`fecf7ecd6b33a53b03d6d1b0d55372b9d9a0d6e5`
- solver contract `12`, policy `impact_phase_cap_dx2_v1`
- L1A-2t re-publication commit `fecf7ecd6b33a53b03d6d1b0d55372b9d9a0d6e5`; reported stage commits `c52f049, 2dd0f16, 2649031` are absent from the object store and the remote and remain unverifiable
- L1A-2t evidence re-verified here: `True` with 14 bound files; audited module sha256 `f8c374892fc87a873b7ee2de8457b97b8418fe5d9520c3a2aadff8777dd01e56`

## B. Frozen contract and prohibitions
- frozen and untouched: `phasefield.py`, `generate_dataset.py`, `cases.py`, `timestep_policy.py`, the CHNS/
  Brinkman/projection/cut-cell operators, every threshold, every historical evidence file and the quality baseline
- quality baseline sha256 `52b41ab68b4d6b86aaa254389eb5398e65ad8c5cbdcd12626c0466248dd3d70b`; no contract 13, no L1B generation, no ML
- frozen adjacent red light: `test_l1a2r_merged_base_verified_or_fail_closed` - `FROZEN_LINEAGE_DRIFT_CONFIRMED`
- measured failure: AuditValidationError: L1A-2s prerequisite/source freeze failed: ['origin_main_is_verified_merge_base']; latest_main=09e2993b8da478d79e768054194201af57121784

## C. Frozen C2 replay
- verdict `BASELINE_REPLAY_OK` against the published L1A-2t t=0 hashes
- `flat_we100_ct050`: MEASURED, matched={'u': True, 'v': True, 'phi': True, 'geometry': True}
- `flat_we200_ct000`: MEASURED, matched={'u': True, 'v': True, 'phi': True, 'geometry': True}
- `pillar_training`: MEASURED, matched={'u': True, 'v': True, 'phi': True, 'geometry': True}
- `complex_heldout`: MEASURED, matched={'u': True, 'v': True, 'phi': True, 'geometry': True}

## D. t=0 feasibility of a wall-confined incident seed
- worst-case `chi >= 0.01` gate-band edge:3.4456 dx
- `flat_we100_ct050`: gate band to 2.5 dx, open cells below the gate start at 3.5 dx, phi=0.5 support at 2.5 dx withchi=0.034445, inside band=True
- `flat_we200_ct000`: gate band to 2.5 dx, open cells below the gate start at 3.5 dx, phi=0.5 support at 2.5 dx withchi=0.034445, inside band=True
- `pillar_training`: gate band to 3.191 dx, open cells below the gate start at 3.5 dx, phi=0.5 support at 2.7 dx withchi=0.026597, inside band=True
- `complex_heldout`: gate band to 3.446 dx, open cells below the gate start at 3.448 dx, phi=0.5 support at 5.682 dx withchi=0.00051239, inside band=False
- structural finding: every case places the frozen phi=0.5 contour support cell inside the chi >= 0.01 gate band, so the frozen gate caps the initial normal speed *at the contour itself* at 0.02 u_impact
- consequence: no gate-compliant initializer of this family can place incident momentum at the t=0 contact footprint; incident momentum can only be placed above the band and advected downward
- verdict `T0_FEASIBILITY_NO_GO_FOR_T0_CONTACT_MOMENTUM`; `FEASIBLE_WINDOW_EXISTS_ABOVE_THE_GATE_BAND`
- probe `flat_we100_ct050/P_DELTA_1.0`: MEASURED, accepted=True, failing=[], core coverage=1
- probe `flat_we100_ct050/P_DELTA_2.5`: MEASURED, accepted=False, failing=['central_D'], core coverage=0.9796
- probe `flat_we100_ct050/P_DELTA_2.75`: MEASURED, accepted=False, failing=['central_D'], core coverage=0.9545
- probe `flat_we100_ct050/P_DELTA_3.0`: MEASURED, accepted=False, failing=['central_D', 'open_face_FV_divergence'], core coverage=0.9246
- probe `flat_we100_ct050/P_DELTA_4.0`: NO_CORE_MOTION
- probe `flat_we200_ct000/P_DELTA_1.0`: MEASURED, accepted=True, failing=[], core coverage=1
- probe `flat_we200_ct000/P_DELTA_2.5`: MEASURED, accepted=False, failing=['central_D'], core coverage=0.9796
- probe `flat_we200_ct000/P_DELTA_2.75`: MEASURED, accepted=False, failing=['central_D'], core coverage=0.9545
- probe `flat_we200_ct000/P_DELTA_3.0`: MEASURED, accepted=False, failing=['central_D', 'open_face_FV_divergence'], core coverage=0.9246
- probe `flat_we200_ct000/P_DELTA_4.0`: NO_CORE_MOTION
- probe `pillar_training/P_DELTA_1.0`: MEASURED, accepted=True, failing=[], core coverage=1
- probe `pillar_training/P_DELTA_2.5`: MEASURED, accepted=False, failing=['central_D'], core coverage=0.9796
- probe `pillar_training/P_DELTA_2.75`: MEASURED, accepted=False, failing=['central_D'], core coverage=0.9545
- probe `pillar_training/P_DELTA_3.0`: MEASURED, accepted=False, failing=['central_D', 'open_face_FV_divergence'], core coverage=0.9246
- probe `pillar_training/P_DELTA_4.0`: NO_CORE_MOTION
- probe `complex_heldout/P_DELTA_1.0`: MEASURED, accepted=True, failing=[], core coverage=1
- probe `complex_heldout/P_DELTA_2.5`: MEASURED, accepted=False, failing=['central_D'], core coverage=0.9796
- probe `complex_heldout/P_DELTA_2.75`: MEASURED, accepted=False, failing=['central_D'], core coverage=0.9545
- probe `complex_heldout/P_DELTA_3.0`: MEASURED, accepted=False, failing=['central_D', 'open_face_FV_divergence'], core coverage=0.9246
- probe `complex_heldout/P_DELTA_4.0`: NO_CORE_MOTION

## E. Prespecified radial-envelope family
- `psi = A * sx * smooth5((R_in - r)/w) * smooth5((dist_y_seam - 3 dy)/(3 dy))`, `u = D_y psi`, `v = -D_x psi`
- no SDF wall taper, no post-derivative mask, one scalar matching the frozen `phi>=0.9` coremean to `-u_impact`
- `R0_WEAK`: delta=1.5 dx, transition width=2.0 dx - weak roll-off: minimum geometrically admissible clearance, widest transition
- `R1_MEDIUM`: delta=2.0 dx, transition width=2.0 dx - medium roll-off: mid-window clearance at the prespecified transition width
- `R2_SHARP`: delta=2.0 dx, transition width=1.0 dx - strong roll-off: mid-window clearance with the sharpest transition

## F. t=0 frozen-gate constraint matrix
- `flat_we100_ct050` `C2_BASE`: COMPATIBLE, failing=[], chi>=0.01 speed/U=0, wall sample/U=0, deep solid/U=0, core v=-0.5, interior peak/U=4.32 (ADMISSIBLE)
- `flat_we100_ct050` `R0_WEAK`: COMPATIBLE, failing=[], chi>=0.01 speed/U=0, wall sample/U=0, deep solid/U=0, core v=-0.5, interior peak/U=31 (DISCRETIZATION_INADMISSIBLE)
- `flat_we100_ct050` `R1_MEDIUM`: COMPATIBLE, failing=[], chi>=0.01 speed/U=0, wall sample/U=0, deep solid/U=0, core v=-0.5, interior peak/U=89.8 (DISCRETIZATION_INADMISSIBLE)
- `flat_we100_ct050` `R2_SHARP`: COMPATIBLE, failing=[], chi>=0.01 speed/U=0, wall sample/U=0, deep solid/U=0, core v=-0.5, interior peak/U=36.3 (DISCRETIZATION_INADMISSIBLE)
- `flat_we100_ct050` control `C0_UNIFORM`: INITIALIZER_INCOMPATIBLE_CASE, chi>=0.01speed/U=1, interior peak/U=1
- `flat_we100_ct050` control `C1_STREAMFUNCTION_V0`: INITIALIZER_INCOMPATIBLE_CASE, chi>=0.01speed/U=3.795, interior peak/U=4.91
- `flat_we200_ct000` `C2_BASE`: COMPATIBLE, failing=[], chi>=0.01 speed/U=0, wall sample/U=0, deep solid/U=0, core v=-0.5, interior peak/U=4.32 (ADMISSIBLE)
- `flat_we200_ct000` `R0_WEAK`: COMPATIBLE, failing=[], chi>=0.01 speed/U=0, wall sample/U=0, deep solid/U=0, core v=-0.5, interior peak/U=31 (DISCRETIZATION_INADMISSIBLE)
- `flat_we200_ct000` `R1_MEDIUM`: COMPATIBLE, failing=[], chi>=0.01 speed/U=0, wall sample/U=0, deep solid/U=0, core v=-0.5, interior peak/U=89.8 (DISCRETIZATION_INADMISSIBLE)
- `flat_we200_ct000` `R2_SHARP`: COMPATIBLE, failing=[], chi>=0.01 speed/U=0, wall sample/U=0, deep solid/U=0, core v=-0.5, interior peak/U=36.3 (DISCRETIZATION_INADMISSIBLE)
- `flat_we200_ct000` control `C0_UNIFORM`: INITIALIZER_INCOMPATIBLE_CASE, chi>=0.01speed/U=1, interior peak/U=1
- `flat_we200_ct000` control `C1_STREAMFUNCTION_V0`: INITIALIZER_INCOMPATIBLE_CASE, chi>=0.01speed/U=3.795, interior peak/U=4.91
- `pillar_training` `C2_BASE`: COMPATIBLE, failing=[], chi>=0.01 speed/U=1.561e-05, wall sample/U=0, deep solid/U=0, core v=-0.5, interior peak/U=4.22 (ADMISSIBLE)
- `pillar_training` `R0_WEAK`: COMPATIBLE, failing=[], chi>=0.01 speed/U=0, wall sample/U=0, deep solid/U=0, core v=-0.5, interior peak/U=31 (DISCRETIZATION_INADMISSIBLE)
- `pillar_training` `R1_MEDIUM`: COMPATIBLE, failing=[], chi>=0.01 speed/U=0, wall sample/U=0, deep solid/U=0, core v=-0.5, interior peak/U=89.8 (DISCRETIZATION_INADMISSIBLE)
- `pillar_training` `R2_SHARP`: COMPATIBLE, failing=[], chi>=0.01 speed/U=0, wall sample/U=0, deep solid/U=0, core v=-0.5, interior peak/U=36.3 (DISCRETIZATION_INADMISSIBLE)
- `pillar_training` control `C0_UNIFORM`: INITIALIZER_INCOMPATIBLE_CASE, chi>=0.01speed/U=1, interior peak/U=1
- `pillar_training` control `C1_STREAMFUNCTION_V0`: INITIALIZER_INCOMPATIBLE_CASE, chi>=0.01speed/U=1.195, interior peak/U=4.91
- `complex_heldout` `C2_BASE`: COMPATIBLE, failing=[], chi>=0.01 speed/U=0.001059, wall sample/U=0, deep solid/U=0, core v=-0.5, interior peak/U=4.21 (ADMISSIBLE)
- `complex_heldout` `R0_WEAK`: COMPATIBLE, failing=[], chi>=0.01 speed/U=0, wall sample/U=0, deep solid/U=0, core v=-0.5, interior peak/U=31 (DISCRETIZATION_INADMISSIBLE)
- `complex_heldout` `R1_MEDIUM`: COMPATIBLE, failing=[], chi>=0.01 speed/U=0, wall sample/U=0, deep solid/U=0, core v=-0.5, interior peak/U=89.8 (DISCRETIZATION_INADMISSIBLE)
- `complex_heldout` `R2_SHARP`: COMPATIBLE, failing=[], chi>=0.01 speed/U=0, wall sample/U=0, deep solid/U=0, core v=-0.5, interior peak/U=36.3 (DISCRETIZATION_INADMISSIBLE)
- `complex_heldout` control `C0_UNIFORM`: INITIALIZER_INCOMPATIBLE_CASE, chi>=0.01speed/U=1, interior peak/U=1
- `complex_heldout` control `C1_STREAMFUNCTION_V0`: INITIALIZER_INCOMPATIBLE_CASE, chi>=0.01speed/U=3.512, interior peak/U=4.91
- interior-speed ceiling/U=4.91 measured from the frozen reference family in the same cases: ['complex_heldout/C1_STREAMFUNCTION_V0', 'flat_we100_ct050/C1_STREAMFUNCTION_V0', 'flat_we200_ct000/C1_STREAMFUNCTION_V0', 'pillar_training/C1_STREAMFUNCTION_V0']
- candidates admissible in every case: []
- candidates inadmissible in every case: ['R0_WEAK', 'R1_MEDIUM', 'R2_SHARP']
- screen rule: a candidate is admissible only while its peak open-cell seed speed stays within the maximum the frozen reference family (C0/C1/C2) itself produces in the same cases; the ceiling is measured here and is not adjustable, and an inadmissible seed invalidates any flux, approach or event comparison derived from its flow field
- bounded pre-contact stability probe on `complex_heldout` over t<=0.064 (precontact only: True)
    - `C2_BASE`: STABLE_WITHIN_PROBE, peak/U t0=4.21 final=4.18, first non-finite macro step=UNMEASURED
    - `R0_WEAK`: STABLE_WITHIN_PROBE, peak/U t0=31 final=14.8, first non-finite macro step=UNMEASURED
    - `R1_MEDIUM`: DIVERGED_WITHIN_PROBE, peak/U t0=89.8 final=UNMEASURED, first non-finite macro step=20
    - `R2_SHARP`: STABLE_WITHIN_PROBE, peak/U t0=36.3 final=15.1, first non-finite macro step=UNMEASURED
- compatible in every case: ['R0_WEAK', 'R1_MEDIUM', 'R2_SHARP']
- selection rule: prespecified median-design candidate R1_MEDIUM is compatible in every case

## G. Near-wall and incident flux budget (t=0)
- `flat_we100_ct050` `C2_BASE`: downward phase flux above the band -0.00026218, mass-weighted downward momentum -0.0050238, gate-band max speed/U 0
- `flat_we100_ct050` `R0_WEAK`: downward phase flux above the band -0.00026683, mass-weighted downward momentum -0.0035018, gate-band max speed/U 0
- `flat_we100_ct050` `R1_MEDIUM`: downward phase flux above the band -0.00042076, mass-weighted downward momentum -0.0048022, gate-band max speed/U 0
- `flat_we100_ct050` `R2_SHARP`: downward phase flux above the band -0.00025251, mass-weighted downward momentum -0.003164, gate-band max speed/U 0
- `flat_we100_ct050` `R0_WEAK` confinement: psi support from 4.5 dx, velocity support from 3.5 dx, cut-cell flux support from 3.5 dx
- `flat_we100_ct050` `R1_MEDIUM` confinement: psi support from 5.5 dx, velocity support from 4.5 dx, cut-cell flux support from 4.5 dx
- `flat_we100_ct050` `R2_SHARP` confinement: psi support from 5.5 dx, velocity support from 4.5 dx, cut-cell flux support from 4.5 dx
- `flat_we200_ct000` `C2_BASE`: downward phase flux above the band -0.00026218, mass-weighted downward momentum -0.0050238, gate-band max speed/U 0
- `flat_we200_ct000` `R0_WEAK`: downward phase flux above the band -0.00026683, mass-weighted downward momentum -0.0035018, gate-band max speed/U 0
- `flat_we200_ct000` `R1_MEDIUM`: downward phase flux above the band -0.00042076, mass-weighted downward momentum -0.0048022, gate-band max speed/U 0
- `flat_we200_ct000` `R2_SHARP`: downward phase flux above the band -0.00025251, mass-weighted downward momentum -0.003164, gate-band max speed/U 0
- `flat_we200_ct000` `R0_WEAK` confinement: psi support from 4.5 dx, velocity support from 3.5 dx, cut-cell flux support from 3.5 dx
- `flat_we200_ct000` `R1_MEDIUM` confinement: psi support from 5.5 dx, velocity support from 4.5 dx, cut-cell flux support from 4.5 dx
- `flat_we200_ct000` `R2_SHARP` confinement: psi support from 5.5 dx, velocity support from 4.5 dx, cut-cell flux support from 4.5 dx
- `pillar_training` `C2_BASE`: downward phase flux above the band -0.00060121, mass-weighted downward momentum -0.012717, gate-band max speed/U 1.561e-05
- `pillar_training` `R0_WEAK`: downward phase flux above the band -0.0010415, mass-weighted downward momentum -0.023263, gate-band max speed/U 0
- `pillar_training` `R1_MEDIUM`: downward phase flux above the band -0.0023279, mass-weighted downward momentum -0.056201, gate-band max speed/U 0
- `pillar_training` `R2_SHARP`: downward phase flux above the band -0.0010318, mass-weighted downward momentum -0.021267, gate-band max speed/U 0
- `pillar_training` `R0_WEAK` confinement: psi support from 4.7 dx, velocity support from 3.7 dx, cut-cell flux support from 3.7 dx
- `pillar_training` `R1_MEDIUM` confinement: psi support from 5.7 dx, velocity support from 4.7 dx, cut-cell flux support from 4.7 dx
- `pillar_training` `R2_SHARP` confinement: psi support from 5.7 dx, velocity support from 4.7 dx, cut-cell flux support from 4.7 dx
- `complex_heldout` `C2_BASE`: downward phase flux above the band 0.00016236, mass-weighted downward momentum 0.0046744, gate-band max speed/U 0.001059
- `complex_heldout` `R0_WEAK`: downward phase flux above the band 0.00058053, mass-weighted downward momentum 0.012088, gate-band max speed/U 0
- `complex_heldout` `R1_MEDIUM`: downward phase flux above the band 0.0010878, mass-weighted downward momentum 0.013148, gate-band max speed/U 0
- `complex_heldout` `R2_SHARP`: downward phase flux above the band 0.00059245, mass-weighted downward momentum 0.01209, gate-band max speed/U 0
- `complex_heldout` `R0_WEAK` confinement: psi support from 7.025 dx, velocity support from 6.165 dx, cut-cell flux support from 5.682 dx
- `complex_heldout` `R1_MEDIUM` confinement: psi support from 7.547 dx, velocity support from 6.605 dx, cut-cell flux support from 6.165 dx
- `complex_heldout` `R2_SHARP` confinement: psi support from 7.547 dx, velocity support from 6.605 dx, cut-cell flux support from 6.165 dx

## H. eta_pen / dt factorial
- verdict `DT_ETA_CONFOUNDED` on admissible seeds only (confounded=3, residual=1); all-rows diagnostic including inadmissible seeds: `DT_ETA_CONFOUNDED` ({'confounded': 3, 'residual': 3})
- rows excluded from the verdict: ['flat_we100_ct050/R1_MEDIUM']
- A coupled historical: dt [0.002, 0.001, 0.0005] with eta_pen = 2 dt (`COUPLED_DT_ETA_BASELINE`)
- B pure dt: same dt levels with the diagnostic override eta_pen = 0.004 (`FIXED_ETA_DT_DIAGNOSTIC`)
- C fixed dt: dt = 0.002 with eta_pen in [0.004, 0.002, 0.001] (`FIXED_DT_ETA_DIAGNOSTIC`)
- window t <= 0.24; sampling cadence matched at 0.002 physical time across dt levels
- verdict `DT_ETA_CONFOUNDED` (confounded observables 3, residual observables 1)
- `flat_we100_ct050` `C2_BASE` coupled_dt_eta_baseline dt_0.002: eta=0.004, dt/eta=0.5, front speed -0.27349, gate-band peak speed/U 0.7026, contact 0.21
- `flat_we100_ct050` `C2_BASE` coupled_dt_eta_baseline dt_0.001: eta=0.002, dt/eta=0.5, front speed -0.263, gate-band peak speed/U 0.5493, contact 0.232
- `flat_we100_ct050` `C2_BASE` coupled_dt_eta_baseline dt_0.0005: eta=0.001, dt/eta=0.5, front speed -0.11301, gate-band peak speed/U 0.38, contact UNMEASURED
- `flat_we100_ct050` `C2_BASE` fixed_eta_dt_diagnostic dt_0.002: eta=0.004, dt/eta=0.5, front speed -0.27349, gate-band peak speed/U 0.7026, contact 0.21
- `flat_we100_ct050` `C2_BASE` fixed_eta_dt_diagnostic dt_0.001: eta=0.004, dt/eta=0.25, front speed -0.27304, gate-band peak speed/U 0.7022, contact 0.212
- `flat_we100_ct050` `C2_BASE` fixed_eta_dt_diagnostic dt_0.0005: eta=0.004, dt/eta=0.125, front speed -0.27279, gate-band peak speed/U 0.7019, contact 0.212
- `flat_we100_ct050` `C2_BASE` fixed_dt_eta_diagnostic eta_0.004: eta=0.004, dt/eta=0.5, front speed -0.27349, gate-band peak speed/U 0.7026, contact 0.21
- `flat_we100_ct050` `C2_BASE` fixed_dt_eta_diagnostic eta_0.002: eta=0.002, dt/eta=1, front speed -0.26378, gate-band peak speed/U 0.5501, contact 0.23
- `flat_we100_ct050` `C2_BASE` fixed_dt_eta_diagnostic eta_0.001: eta=0.001, dt/eta=2, front speed -0.11823, gate-band peak speed/U 0.3819, contact UNMEASURED
- `flat_we100_ct050` `R1_MEDIUM` coupled_dt_eta_baseline dt_0.002: eta=0.004, dt/eta=0.5, front speed -0.25057, gate-band peak speed/U 4.607e+04, contact UNMEASURED
- `flat_we100_ct050` `R1_MEDIUM` coupled_dt_eta_baseline dt_0.001: eta=0.002, dt/eta=0.5, front speed -0.33324, gate-band peak speed/U 27.14, contact 0.09
- `flat_we100_ct050` `R1_MEDIUM` coupled_dt_eta_baseline dt_0.0005: eta=0.001, dt/eta=0.5, front speed -0.32992, gate-band peak speed/U 22.92, contact 0.1
- `flat_we100_ct050` `R1_MEDIUM` fixed_eta_dt_diagnostic dt_0.002: eta=0.004, dt/eta=0.5, front speed -0.25057, gate-band peak speed/U 4.607e+04, contact UNMEASURED
- `flat_we100_ct050` `R1_MEDIUM` fixed_eta_dt_diagnostic dt_0.001: eta=0.004, dt/eta=0.25, front speed -0.33889, gate-band peak speed/U 28.14, contact 0.082
- `flat_we100_ct050` `R1_MEDIUM` fixed_eta_dt_diagnostic dt_0.0005: eta=0.004, dt/eta=0.125, front speed -0.3387, gate-band peak speed/U 27.25, contact 0.082
- `flat_we100_ct050` `R1_MEDIUM` fixed_dt_eta_diagnostic eta_0.004: eta=0.004, dt/eta=0.5, front speed -0.25057, gate-band peak speed/U 4.607e+04, contact UNMEASURED
- `flat_we100_ct050` `R1_MEDIUM` fixed_dt_eta_diagnostic eta_0.002: eta=0.002, dt/eta=1, front speed -0.20027, gate-band peak speed/U 3.859e+04, contact UNMEASURED
- `flat_we100_ct050` `R1_MEDIUM` fixed_dt_eta_diagnostic eta_0.001: eta=0.001, dt/eta=2, front speed -0.15404, gate-band peak speed/U 3.124e+04, contact UNMEASURED
- `pillar_training` `C2_BASE` coupled_dt_eta_baseline dt_0.002: eta=0.004, dt/eta=0.5, front speed -0.079076, gate-band peak speed/U 0.9437, contact UNMEASURED
- `pillar_training` `C2_BASE` coupled_dt_eta_baseline dt_0.001: eta=0.002, dt/eta=0.5, front speed -0.062265, gate-band peak speed/U 0.8584, contact UNMEASURED
- `pillar_training` `C2_BASE` coupled_dt_eta_baseline dt_0.0005: eta=0.001, dt/eta=0.5, front speed -0.046539, gate-band peak speed/U 0.7158, contact UNMEASURED
- `pillar_training` `C2_BASE` fixed_eta_dt_diagnostic dt_0.002: eta=0.004, dt/eta=0.5, front speed -0.079076, gate-band peak speed/U 0.9437, contact UNMEASURED
- `pillar_training` `C2_BASE` fixed_eta_dt_diagnostic dt_0.001: eta=0.004, dt/eta=0.25, front speed -0.078571, gate-band peak speed/U 0.9436, contact UNMEASURED
- `pillar_training` `C2_BASE` fixed_eta_dt_diagnostic dt_0.0005: eta=0.004, dt/eta=0.125, front speed -0.078325, gate-band peak speed/U 0.9436, contact UNMEASURED
- `pillar_training` `C2_BASE` fixed_dt_eta_diagnostic eta_0.004: eta=0.004, dt/eta=0.5, front speed -0.079076, gate-band peak speed/U 0.9437, contact UNMEASURED
- `pillar_training` `C2_BASE` fixed_dt_eta_diagnostic eta_0.002: eta=0.002, dt/eta=1, front speed -0.063034, gate-band peak speed/U 0.8585, contact UNMEASURED
- `pillar_training` `C2_BASE` fixed_dt_eta_diagnostic eta_0.001: eta=0.001, dt/eta=2, front speed -0.048126, gate-band peak speed/U 0.7167, contact UNMEASURED
- rule: the coupled and the fixed-eta dt trends are compared on the same observables at matched physical times; a material difference (>25% of the larger ratio) marks the observable as confounded by the Brinkman penalty timescale, otherwise as a time-discretization residual; rows from interior-inadmissible seeds that were actually run here are labelled and excluded

## I. Interface versus fluid velocity
- `flat_we100_ct050` `C2_BASE`: front speed d(gap)/dt -0.27349, mean local normal fluid speed 0.044701, max 0.059809, contact 0.21
  - t=0 split at the support window: adv 1.381e-07, CH explicit 1.274e-05, CH implicit -1.138e-05, total1.495e-06, closure residual -1.14e-19
- `flat_we100_ct050` `R1_MEDIUM`: front speed d(gap)/dt -0.25057, mean local normal fluid speed -96.011, max 0.21958, contact UNMEASURED
  - t=0 split at the support window: adv 6.274e-06, CH explicit 1.274e-05, CH implicit -1.102e-05, total7.988e-06, closure residual 3.88e-19
- `flat_we200_ct000` `C2_BASE`: front speed d(gap)/dt -0.093783, mean local normal fluid speed 0.04766, max 0.059848, contact UNMEASURED
  - t=0 split at the support window: adv 1.381e-07, CH explicit 1.228e-05, CH implicit -1.136e-05, total1.057e-06, closure residual -3.92e-20
- `flat_we200_ct000` `R1_MEDIUM`: front speed d(gap)/dt -0.29782, mean local normal fluid speed -70.383, max 0.21554, contact UNMEASURED
  - t=0 split at the support window: adv 6.274e-06, CH explicit 1.228e-05, CH implicit -1.1e-05, total7.55e-06, closure residual 4.62e-19
- `pillar_training` `C2_BASE`: front speed d(gap)/dt -0.079076, mean local normal fluid speed 0.061309, max 0.088322, contact UNMEASURED
  - t=0 split at the support window: adv 2.049e-07, CH explicit 5.846e-05, CH implicit -9.435e-05, total-3.568e-05, closure residual 1.46e-19
- `pillar_training` `R1_MEDIUM`: front speed d(gap)/dt -0.3342, mean local normal fluid speed -62.97, max 0.22728, contact UNMEASURED
  - t=0 split at the support window: adv 6.274e-06, CH explicit 5.846e-05, CH implicit -9.399e-05, total-2.926e-05, closure residual 4.81e-19
- `complex_heldout` `C2_BASE`: front speed d(gap)/dt -0.096722, mean local normal fluid speed 0.048491, max 0.5077, contact UNMEASURED
  - t=0 split at the support window: adv 9.35e-06, CH explicit 6.044e-06, CH implicit -8.18e-06, total7.214e-06, closure residual -2.42e-20
- `complex_heldout` `R1_MEDIUM`: front speed d(gap)/dt -0.39778, mean local normal fluid speed 427.17, max 8540.2, contact UNMEASURED
  - t=0 split at the support window: adv 3.176e-05, CH explicit 6.044e-06, CH implicit -8.778e-06, total2.903e-05, closure residual 9.9e-21

## J. Counterfactual ledger (eta axis only)
- 9 matched entries; counterfactual over the eta_pen axis only, holding the seed, the geometry, the physics and the matched physical sampling times fixed; every quantity is compared at the same position and time
- `flat_we100_ct050` `C2_BASE` dt=0.002: eta 0.004 -> 0.004, gate-band peak speed delta 0, front-speed delta 0
- `flat_we100_ct050` `C2_BASE` dt=0.001: eta 0.002 -> 0.004, gate-band peak speed delta 0.1529, front-speed delta -0.010042
- `flat_we100_ct050` `C2_BASE` dt=0.0005: eta 0.001 -> 0.004, gate-band peak speed delta 0.3219, front-speed delta -0.15978
- `flat_we100_ct050` `R1_MEDIUM` dt=0.002: eta 0.004 -> 0.004, gate-band peak speed delta 0, front-speed delta 0
- `flat_we100_ct050` `R1_MEDIUM` dt=0.001: eta 0.002 -> 0.004, gate-band peak speed delta 1.007, front-speed delta -0.0056532
- `flat_we100_ct050` `R1_MEDIUM` dt=0.0005: eta 0.001 -> 0.004, gate-band peak speed delta 4.331, front-speed delta -0.0087782
- `pillar_training` `C2_BASE` dt=0.002: eta 0.004 -> 0.004, gate-band peak speed delta 0, front-speed delta 0
- `pillar_training` `C2_BASE` dt=0.001: eta 0.002 -> 0.004, gate-band peak speed delta 0.08529, front-speed delta -0.016306
- `pillar_training` `C2_BASE` dt=0.0005: eta 0.001 -> 0.004, gate-band peak speed delta 0.2278, front-speed delta -0.031786

## K. Failure injection and adversarial checks
- eta_override_does_not_change_production_default: {"default_eta_pen": 0.004, "expected": 0.004, "holds": true}
- eta_override_is_local_and_reversible: {"bundle_eta_before": 0.004, "bundle_eta_after": 0.004, "cloned_eta": 0.001, "holds": true}
- c2_t0_hash_reproduces_l1a2t_record_at_N48_is_not_comparable: {"note": "N=48 workspace hash is not comparable to the N=192 record; the N=192 replay is the binding check", "record_u_hash_present": false, "workspace_u_hash": "eec51fcf982d4c9afc795f33043f568a63223b2c09bcf374b23f50572730a4cf", "full_hash_comparison_deferred_to": "c2_reproduction_check"}
- radial_family_uses_no_sdf_taper: {"envelope_above_zero_sdf": 0.0, "holds": true, "envelope_is_radial_only": true, "psi_support_min_sdf_over_dx": 5.5, "radius_support_over_R": 0.6341172456741333}
- l1a2t_evidence_still_verifies: {"verified": true, "module_sha_matches_stage_record": true}
- missing_l1a2t_commits_are_recorded: {"commits": ["c52f049", "2dd0f16", "2649031"], "resolved": {"c52f049": null, "2dd0f16": null, "2649031": null}, "all_unresolved": true}
- injected_failures_are_rejected: {"negative_eta_rejected": true, "zero_width_rejected": true, "empty_plateau_rejected": true, "unknown_candidate_rejected": true}
- all self checks hold: True

## L. Decision
- stage status `L1A2U_DIAGNOSTIC_COMPLETE_NO_PROMOTION`
- t=0 feasibility `T0_FEASIBILITY_NO_GO_FOR_T0_CONTACT_MOMENTUM` / `FEASIBLE_WINDOW_EXISTS_ABOVE_THE_GATE_BAND`
- radial candidate outcome `RADIAL_SEED_DISCRETIZATION_INADMISSIBLE_NO_GAIN_CLAIM`
- dt/eta `DT_ETA_CONFOUNDED`; mechanism `INTERIOR_SPEED_INADMISSIBLE`
- cases with material flux and approach gain:0 of 4 compared (0 of 4 case readouts valid)
- gain claim status: WITHHELD: the chosen seed is interior-speed inadmissible, so its flux and approach materiality rows are recorded but cannot support a speed-gain claim
- inadmissible-readout cases: ['complex_heldout', 'flat_we100_ct050', 'flat_we200_ct000', 'pillar_training']; interior-speed screen: {"ceiling_over_u_impact": 4.905526491646905, "ceiling_sources": ["complex_heldout/C1_STREAMFUNCTION_V0", "flat_we100_ct050/C1_STREAMFUNCTION_V0", "flat_we200_ct000/C1_STREAMFUNCTION_V0", "pillar_training/C1_STREAMFUNCTION_V0"], "admissible_in_every_case": [], "inadmissible_in_every_case": ["R0_WEAK", "R1_MEDIUM", "R2_SHARP"], "rule": "a candidate is admissible only while its peak open-cell seed speed stays within the maximum the frozen reference family (C0/C1/C2) itself produces in the same cases; the ceiling is measured here and is not adjustable, and an inadmissible seed invalidates any flux
- provenance: commit SHAs verifiable=False, content recovered=True, remote provenance blocked=False
- frozen lineage gate blocked: True

## M. Quality, tests and hosted CI
- quality baseline `52b41ab68b4d6b86aaa254389eb5398e65ad8c5cbdcd12626c0466248dd3d70b`, declared new diagnostics 0, ratchet expectation `PASS_BASELINE_ONLY`
- local validation bound to this revision: True
- local validation payload: {"final_head_adjacent_recheck": {"command": "JAX_ENABLE_X64=1 PYTHONPATH=. ../../.venv/bin/pytest -q tests/test_radial_incident_seed_eta_dt_audit.py tests/test_nearwall_taper_interface_transport_audit.py tests/test_impact_initialization_compatibility_audit.py tests/test_impact_impulse_projection_audit.py", "duration_seconds": 143.67, "failures": {"test_l1a2r_merged_base_verified_or_fail_closed": "FROZEN_LINEAGE_DRIFT_CONFIRMED (pre-existing; reproduced identically on a pristine worktree at the verified main 09e2993 with this stage absent)", "test_nearwall_taper_interface_transport_audit.py::test_manifest_mismatch_reference_is_recorded_in_preflight": "SUPERSEDED_BY_NEXT_STAGE_BY_CONSTRUCTION - the frozen L1A-2t preflight asserts the worktree holds no untracked changes outside L1A-2t deliverables and that every commit on top of the verified main touches only L1A-2t paths; that assertion ca
- hosted CI for this head: `success` (observed after push)
- base main CI (run 38120327651, head 865317d4):success 8/8 - engineering CI of the verified main this branch is based on; not evidence forthis head

## N. Exactly one next action
- Re-screen the radial family at t=0 only, with the measured interior-speed ceiling added as a prespecified design constraint and the frozen six gates unchanged, and report whether any member of the prespecified family can hold the incident flux gain while keeping the peak open-cell seed speed inside the frozen reference family range; no trajectory, no event and no refinement claim until that screen passes.
- bounded scope: one t=0 screening pass over the prespecified family; no trajectory, no generator, no ML
- must not be started in this stage: True

