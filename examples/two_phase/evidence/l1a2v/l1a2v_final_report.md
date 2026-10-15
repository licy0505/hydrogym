# L1A-2v final report - interior-speed-constrained t=0 admissibility screen

## A. Scope, authority and baseline
- sole study baseline: PR #24 head `11f299d02179169ab6d8427c7b7b8b8b686f00aa` (verified main `09e2993b8da478d79e768054194201af57121784`)
- branch `arena/02b6c71b-hydrogym`, contract 12, policy `impact_phase_cap_dx2_v1`
- screen profile: N=192, dt=0.002, cases ['flat_we100_ct050', 'flat_we200_ct000', 'pillar_training', 'complex_heldout']
- trajectories run: 0; contact events: NOT_RUN_IN_THIS_STAGE

## B. Frozen contract and prohibitions
- six frozen gates: {"central_D_dimensionless_Linf_max": 7.62939453125e-06, "core_mean_speed_relative_error_max": 0.01, "deep_solid_speed_over_u_impact_max": 3.814697265625e-06, "embedded_wall_bilinear_normal_speed_over_u_impact_max": 0.02, "nearwall_chi_ge_0p01_speed_over_u_impact_max": 0.02, "open_face_FV_dimensionless_Linf_max": 0.01, "region_definitions": {"deep_solid": "sdf < -2*dx", "nearwall_sdf": "abs(sdf) <= 3*dx", "nearwall_chi": "chi >= 0.01", "partial_cells": "0 < V_i < dx*dy", "wall_samples": "wall_measure > 0, bilinear cell-velocity sample at wall centroid", "liquid_mean": "sum(V_i*phi_i*v_i)/sum(V_i*phi_i), raw phi as specified", "core_mean": "V_i-weighted mean over phi >= 0.9"}}
- gate source: inherited verbatim from production.impact_initialization_compatibility_audit.INITIAL_ACCEPTANCE; not re-tuned, not relaxed, not re-ordered
- core normalisation target: -1.0 (unchanged)
- prohibitions: {"trajectories": "none; the frozen substep kernel is never constructed by this module", "contact_events": "none", "temporal_refinement": "none", "spatial_refinement": "none", "geometry_change": "none", "re_tuning_after_results": "forbidden; a non-admissible family is reported, never adjusted", "historical_evidence_edits": "none", "production_solver_edits": "none", "contract_13": false, "l1b_data_generation": false, "ml_training": false}

## C. Sealed evidence verification
- L1A-2u frozen verifier: true, bound files 13
- L1A-2t frozen verifier: true, bound files 14
- filename inventory across every sealed directory: 17 verified, 1 superseded
    - superseded record `examples/two_phase/evidence/quality1/check_invariants.py`: last changed by `ea00347c8a6cebf6cfbff36f744d39ccd60784d1 QUALITY-2: Option-R SHA-pinned quality-debt ratchet`
    - superseded record `examples/two_phase/evidence/quality1/codespell.txt`: last changed by `ea00347c8a6cebf6cfbff36f744d39ccd60784d1 QUALITY-2: Option-R SHA-pinned quality-debt ratchet`
    - superseded record `examples/two_phase/evidence/quality1/quality_baseline.json`: last changed by `ea00347c8a6cebf6cfbff36f744d39ccd60784d1 QUALITY-2: Option-R SHA-pinned quality-debt ratchet`
    - superseded record `scripts/quality_debt_gate.py`: last changed by `32c1cbf4237189408cc8b9a8d8f5ec659bf57ba6 QUALITY-3: reconcile PR22 quality baseline and record integration evidence`

## D. Prespecified screen contract
- envelope limit (unmodifiable): 4.905526491646905
- envelope provenance: {"recorded_in": "examples/two_phase/evidence/l1a2u/mechanism_decision.json", "recorded_value": 4.905526491646905, "prespecified_value": 4.905526491646905, "matches": true, "measured_from": ["complex_heldout/C1_STREAMFUNCTION_V0", "flat_we100_ct050/C1_STREAMFUNCTION_V0", "flat_we200_ct000/C1_STREAMFUNCTION_V0", "pillar_training/C1_STREAMFUNCTION_V0"], "measurement_rule": "a candidate is admissible only while its peak open-cell seed speed stays within the maximum the frozen reference family (C0/C1/C2) itself produces in the same cases; the ceiling is measured here and is not adjustable, and an inadmissible seed invalidates any flux, approach or event comparison derived from its flow field"}
- implied shape-ratio limit: 4.905526491646905
- checks per candidate: ['six_frozen_initial_constraints', 'global_speed_envelope', 'core_normalisation_target']

## E. t=0 per-candidate measurements
- `flat_we100_ct050` `C2_BASE`: ADMISSIBLE, peak/U=4.316 at cell [95, 32] (sdf/dx=24.5, chi=6.44e-15, in band=false, in solid=false), CFL peak=0.138 domain max=0.138, div D=1.83e-07, div FV=5.74e-07, deep solid/U=0, wall normal/U=0, band speed/U=0, core v/U=-1, six gates=true, envelope=true
- `flat_we100_ct050` `R0_WEAK`: NOT_ADMISSIBLE, peak/U=30.97 at cell [78, 23] (sdf/dx=15.5, chi=1.06e-09, in band=false, in solid=false), CFL peak=1.39 domain max=1.39, div D=9.54e-07, div FV=2.21e-06, deep solid/U=0, wall normal/U=0, band speed/U=0, core v/U=-1, six gates=true, envelope=false
- `flat_we100_ct050` `R1_MEDIUM`: NOT_ADMISSIBLE, peak/U=89.77 at cell [78, 24] (sdf/dx=16.5, chi=2.79e-10, in band=false, in solid=false), CFL peak=4.02 domain max=4.02, div D=3.46e-06, div FV=7.63e-06, deep solid/U=0, wall normal/U=0, band speed/U=0, core v/U=-1, six gates=true, envelope=false
- `flat_we100_ct050` `R2_SHARP`: NOT_ADMISSIBLE, peak/U=36.33 at cell [77, 25] (sdf/dx=17.5, chi=7.35e-11, in band=false, in solid=false), CFL peak=1.64 domain max=1.64, div D=9.54e-07, div FV=1.91e-06, deep solid/U=0, wall normal/U=0, band speed/U=0, core v/U=-1, six gates=true, envelope=false
- `flat_we200_ct000` `C2_BASE`: ADMISSIBLE, peak/U=4.316 at cell [95, 32] (sdf/dx=24.5, chi=6.44e-15, in band=false, in solid=false), CFL peak=0.138 domain max=0.138, div D=1.83e-07, div FV=5.74e-07, deep solid/U=0, wall normal/U=0, band speed/U=0, core v/U=-1, six gates=true, envelope=true
- `flat_we200_ct000` `R0_WEAK`: NOT_ADMISSIBLE, peak/U=30.97 at cell [78, 23] (sdf/dx=15.5, chi=1.06e-09, in band=false, in solid=false), CFL peak=1.39 domain max=1.39, div D=9.54e-07, div FV=2.21e-06, deep solid/U=0, wall normal/U=0, band speed/U=0, core v/U=-1, six gates=true, envelope=false
- `flat_we200_ct000` `R1_MEDIUM`: NOT_ADMISSIBLE, peak/U=89.77 at cell [78, 24] (sdf/dx=16.5, chi=2.79e-10, in band=false, in solid=false), CFL peak=4.02 domain max=4.02, div D=3.46e-06, div FV=7.63e-06, deep solid/U=0, wall normal/U=0, band speed/U=0, core v/U=-1, six gates=true, envelope=false
- `flat_we200_ct000` `R2_SHARP`: NOT_ADMISSIBLE, peak/U=36.33 at cell [77, 25] (sdf/dx=17.5, chi=7.35e-11, in band=false, in solid=false), CFL peak=1.64 domain max=1.64, div D=9.54e-07, div FV=1.91e-06, deep solid/U=0, wall normal/U=0, band speed/U=0, core v/U=-1, six gates=true, envelope=false
- `pillar_training` `C2_BASE`: ADMISSIBLE, peak/U=4.225 at cell [95, 45] (sdf/dx=24.7, chi=5.05e-15, in band=false, in solid=false), CFL peak=0.135 domain max=0.135, div D=1.79e-07, div FV=3.58e-07, deep solid/U=0, wall normal/U=0, band speed/U=1.561e-05, core v/U=-1, six gates=true, envelope=true
- `pillar_training` `R0_WEAK`: NOT_ADMISSIBLE, peak/U=30.97 at cell [78, 36] (sdf/dx=20.2, chi=2.03e-12, in band=false, in solid=false), CFL peak=1.39 domain max=1.39, div D=9.54e-07, div FV=2.21e-06, deep solid/U=0, wall normal/U=0, band speed/U=0, core v/U=-1, six gates=true, envelope=false
- `pillar_training` `R1_MEDIUM`: NOT_ADMISSIBLE, peak/U=89.77 at cell [78, 37] (sdf/dx=21, chi=7.1e-13, in band=false, in solid=false), CFL peak=4.02 domain max=4.02, div D=3.46e-06, div FV=7.63e-06, deep solid/U=0, wall normal/U=0, band speed/U=0, core v/U=-1, six gates=true, envelope=false
- `pillar_training` `R2_SHARP`: NOT_ADMISSIBLE, peak/U=36.33 at cell [77, 38] (sdf/dx=22.4, chi=1.09e-13, in band=false, in solid=false), CFL peak=1.64 domain max=1.64, div D=9.54e-07, div FV=1.91e-06, deep solid/U=0, wall normal/U=0, band speed/U=0, core v/U=-1, six gates=true, envelope=false
- `complex_heldout` `C2_BASE`: ADMISSIBLE, peak/U=4.213 at cell [95, 54] (sdf/dx=27.6, chi=3.89e-16, in band=false, in solid=false), CFL peak=0.135 domain max=0.135, div D=2.09e-07, div FV=5.51e-07, deep solid/U=0, wall normal/U=0, band speed/U=0.001059, core v/U=-1, six gates=true, envelope=true
- `complex_heldout` `R0_WEAK`: NOT_ADMISSIBLE, peak/U=30.97 at cell [78, 45] (sdf/dx=26.3, chi=6.66e-16, in band=false, in solid=false), CFL peak=1.39 domain max=1.39, div D=9.54e-07, div FV=2.21e-06, deep solid/U=0, wall normal/U=0, band speed/U=0, core v/U=-1, six gates=true, envelope=false
- `complex_heldout` `R1_MEDIUM`: NOT_ADMISSIBLE, peak/U=89.77 at cell [78, 46] (sdf/dx=27.2, chi=3.89e-16, in band=false, in solid=false), CFL peak=4.02 domain max=4.02, div D=3.46e-06, div FV=7.63e-06, deep solid/U=0, wall normal/U=0, band speed/U=0, core v/U=-1, six gates=true, envelope=false
- `complex_heldout` `R2_SHARP`: NOT_ADMISSIBLE, peak/U=36.33 at cell [77, 47] (sdf/dx=27.6, chi=3.89e-16, in band=false, in solid=false), CFL peak=1.64 domain max=1.64, div D=9.54e-07, div FV=1.91e-06, deep solid/U=0, wall normal/U=0, band speed/U=0, core v/U=-1, six gates=true, envelope=false

## F. Near-wall band classification and the refuted L1A-2u universal claim
- `flat_we100_ct050`: phi=0.5 support sdf/dx=2.5, chi=0.03445, inside gate band=true, band max sdf/dx=2.5
- `flat_we200_ct000`: phi=0.5 support sdf/dx=2.5, chi=0.03445, inside gate band=true, band max sdf/dx=2.5
- `pillar_training`: phi=0.5 support sdf/dx=2.7, chi=0.0266, inside gate band=true, band max sdf/dx=3.19
- `complex_heldout`: phi=0.5 support sdf/dx=5.68, chi=0.0005124, inside gate band=false, band max sdf/dx=3.45
- cases with support inside the band: ['flat_we100_ct050', 'flat_we200_ct000', 'pillar_training']
- cases with support outside the band: ['complex_heldout']
- L1A-2u statement (verbatim, sealed): "every case places the frozen phi=0.5 contour support cell inside the chi >= 0.01 gate band, so the frozen gate caps the initial normal speed *at the contour itself* at 0.02 u_impact"
- that statement is universal: true; refuted: true
- correction: the L1A-2u statement that every frozen canary places the phi=0.5 contour support cell inside the chi >= 0.01 gate band does not hold: complex_heldout places it outside (measured here). The statement is recorded as refuted in this independent evidence rather than edited in place; the L1A-2u evidence remains sealed and byte-identical.

## G. Shape invariance of the envelope and core requirements
- derived limit: peak/core <= 4.905526491646905 = envelope / |core target|
- `C2_BASE`: peak/U=4.316, core/U=-1, shape ratio=4.316 (limit 4.906), passes=true, core reached if rescaled to the envelope=1.1367
- `R0_WEAK`: peak/U=30.97, core/U=-1, shape ratio=30.97 (limit 4.906), passes=false, core reached if rescaled to the envelope=0.15838
- `R1_MEDIUM`: peak/U=89.77, core/U=-1, shape ratio=89.77 (limit 4.906), passes=false, core reached if rescaled to the envelope=0.054646
- `R2_SHARP`: peak/U=36.33, core/U=-1, shape ratio=36.33 (limit 4.906), passes=false, core reached if rescaled to the envelope=0.13503
- rule: the envelope (4.905526491646905) and the core target (core-mean v = -u_impact, dimensionless magnitude 1.0) together require a shape ratio peak/core <= 4.905526491646905; the ratio is invariant under amplitude rescaling, so a violation cannot be repaired by re-scaling within the family

## H. Admissibility decision
- verdict `NO_ADMISSIBLE_RADIAL_SEED_UNDER_PRESPECIFIED_FAMILY`
- admissible radial members: []
- family stopped: true; re-tuning performed: false
- stop rule: when no prespecified member is admissible the stage stops and reports; it does not re-tune envelope width, radial support radius or amplitude to manufacture a passing member
- `R0_WEAK`: admissible=false, primary failure=global_speed_envelope, peak/envelope worst-case ratio=6.314, shape ratio=30.97
- `R1_MEDIUM`: admissible=false, primary failure=global_speed_envelope, peak/envelope worst-case ratio=18.3, shape ratio=89.77
- `R2_SHARP`: admissible=false, primary failure=global_speed_envelope, peak/envelope worst-case ratio=7.406, shape ratio=36.33
- reference baseline `C2_BASE`: frozen reference baseline, not a member of the radial family under test, admissible=true
- refuted universal claim recorded in this independent evidence: true

## I. Failure injection and adversarial checks
- probe `envelope_must_equal_the_sealed_l1a2u_value`: raised=false
- probe `unknown_candidate_is_rejected`: raised=true - unknown reference candidate 'NOT_A_CANDIDATE'
- probe `envelope_comparison_is_fail_closed_at_the_boundary`: raised=false
- probe `relaxed_envelope_must_be_refused`: raised=true - envelope relaxation refused by construction: a relaxed limit 46.46067278703233 would admit R0_WEAK (peak 30.973781858021553) and is not permitted by this stage
- adversarial: {"envelope_is_a_single_case_independent_constant": true, "core_target_unchanged_from_l1a2u": true, "candidate_set_is_the_frozen_one": true, "family_parameters_not_modified": {"R0_WEAK": {"delta_dx": 1.5, "w_dx": 2.0, "role": "weak roll-off: minimum geometrically admissible clearance, widest transition"}, "R1_MEDIUM": {"delta_dx": 2.0, "w_dx": 2.0, "role": "medium roll-off: mid-window clearance at the prespecified transition width"}, "R2_SHARP": {"delta_dx": 2.0, "w_dx": 1.0, "role": "strong roll-off: mid-window clearance with the sharpest transition"}}, "no_kernel_constructed_by_this_module": true, "verdict_vocabulary_fixed": true}

## J. Quality, tests and hosted CI
- quality baseline `52b41ab68b4d6b86aaa254389eb5398e65ad8c5cbdcd12626c0466248dd3d70b`, declared new diagnostics 0, expectation `PASS_BASELINE_ONLY`
- frozen adjacent failure `test_l1a2r_merged_base_verified_or_fail_closed`: FROZEN_LINEAGE_DRIFT_CONFIRMED - pre-existing; reproduced identically on a pristine worktree at the verified main with this stage absent; not edited
- frozen adjacent failure `tests/test_nearwall_taper_interface_transport_audit.py::test_manifest_mismatch_reference_is_recorded_in_preflight`: SUPERSEDED_BY_NEXT_STAGE_BY_CONSTRUCTION - the frozen L1A-2t preflight asserts that every commit above the verified main touches only L1A-2t paths and that no other stage's files are present; that assertion cannot hold on a branch that carries L1A-2u and L1A-2v; not edited
- not run: ['trajectories', 'contact events', 'temporal refinement', 'spatial refinement', 'T=8 recompute', 'L1B data generation', 'ML training', 'contract 13']

## K. Blockers and inherited statuses
- inherited: {"SOLVER_CONTRACT_VERSION": 12, "L1A_STATUS": "BLOCKED", "L1B_DATA": "L1B_DATA_NOT_READY", "unchanged_by_this_stage": true}
- blocker for any contact study: no admissible radial member exists under the prespecified family

- hosted CI observed after push: `success` for head `aeac0a74400757e1ebe1900103fbd360df0523a1` (previous record: `UNMEASURED`)

## L. Exactly one next action
- Do not re-tune the radial family. At t=0 only, measure the *reachable* interaction window that the frozen six gates themselves define: for each canary case, compute the maximum incident normal momentum that can be placed inside the admissible region (sdf above the gate band and open-cell speed under the 4.905526491646905 envelope) using a support whose interior peak respects the envelope by construction, and report whether that reachable window can deliver any incident momentum to the frozen phi=0.5 contour at all; no trajectory, event or refinement work until that t=0 bound is established.
- bounded scope: one t=0 reachability bound per canary case; no trajectory, no event, no refinement, no new initializer promotion
- must not be started in this stage: true

## M. Status and provenance
- module sha256 `1890e6557755dfdc68d6386434fd70dada65db4f268eeeb55c28a7f9aed82f42`
- sealed dependency hashes: {"evidence/l1a2u/manifest.json": "457d9d27c109fd3deef0c943a7a81e517ce6706bc7498254af3054d764c109b5", "evidence/l1a2t/manifest.json": "c7f90a4aa4b48c795966ba41778e42e7335a1c57add1ce02d17c7a7d63356a9f", "production/radial_incident_seed_eta_dt_audit.py": "e005d24f9e3c5f5ccd21fcff5957a523860dfc23918f576a1c87e47738fdc1c3"}
- status `NO_ADMISSIBLE_RADIAL_SEED_UNDER_PRESPECIFIED_FAMILY`; no promotion; no trajectory; overlay evidence only

