# L1A-2x - frozen CHNS early-window interface transport attribution

## A. Scope, authority and baseline
- baseline: PR #24 head `45749bc603568bdac8629a90e529ae1573796d51`; verified main `09e2993b8da478d79e768054194201af57121784`; branch `arena/02b6c71b-hydrogym`; solver contract 12; boundary model `impermeable_flux`
- one case `pillar_training`, N=192, dt=0.002, macro steps 40 of at most 40 (120 frozen substeps, t <= 0.08)
- trajectory count 1, control arms 0, contact events scored 0, refinements 0

## B. Frozen contract and prohibitions
- the frozen solver is executed, never edited: `pf.rhs`, `phase_advective_fluxes`,
  `chemical_potential_fluxes`, `control_volume_divergence`, `solve_ch_implicit`, the frozen Brinkman
  damping and the frozen pressure projection, with the frozen face apertures and control volumes
- frozen source hashes: `{"examples/two_phase/phasefield.py": "024665742dce67bc970052e9760fa375ccda096f534cae5faaf2b446c11fe27c", "examples/two_phase/production/impact_initialization_compatibility_audit.py": "788791df2b0bb3e737fe68cba6e5cd3e98d100285184e0b727ce02363309a3b0", "examples/two_phase/production/radial_incident_seed_eta_dt_audit.py": "e005d24f9e3c5f5ccd21fcff5957a523860dfc23918f576a1c87e47738fdc1c3", "examples/two_phase/production/timestep_policy.py": "fa1da721609b274ffea1752ff4e074653d03ff24793ae418c109bbcc97bbb223"}`
- prohibitions: {"control_arm_runs": 0, "refinement_runs": 0, "contact_events_scored": 0, "t8_recomputed": false, "l1b_data_generated": false, "ml_training_runs": 0, "solver_source_edited": false, "initializer_promoted": false}
- contract flags unchanged: `SOLVER_CONTRACT_VERSION=12`, `L1A_STATUS=BLOCKED`, `L1B_DATA_NOT_READY`

## C. Sealed evidence verification
- L1A-2t: VERIFIED, bound files 14, manifest sha256 `c7f90a4aa4b48c79...`
- L1A-2u: VERIFIED, bound files 13, manifest sha256 `457d9d27c109fd3d...`
- L1A-2v: VERIFIED, bound files 14, manifest sha256 `3f30a76a338638c7...`
- L1A-2w: VERIFIED, bound files 16, manifest sha256 `8b58544e5467e93e...`
- all verified: True; failures: []

## D. Initial state identity (frozen L1A-2s C2, never a ramp or a uniform seed)
- state source: frozen audit.build_candidate SDF_TAPERED_STREAMFUNCTION_V1; candidate key `C2_BASE`
- phi/u/v hashes match the sealed L1A-2w evidence: {"phi_hash": true, "u_hash": true, "v_hash": true, "contour_cell": true, "contour_cell_chi": true, "contour_cell_normal_velocity": true}
- support cell [95, 23] at gap 2.70598 dx, |grad phi| 7.02306 per length unit, interface normal [0.023917, 0.999714], wall normal [0.0, -1.0]
- geometry hash `ea3159bf7e107f42...`, parameter hash `c11634c275a29f1e...`

## E. The window and its ledger
- 120 substep rows, each carrying the advective increment, the explicit chemical-potential increment, the implicit CG increment, the complete update, the mass ledger, the closed-face flux, the CG diagnostics and the fixed-support probe
- contact guard: {"triggered": false, "macro_step": null, "gap_over_dx": null} (threshold 1.5 dx, never scored)

## F. Ingredient closure, conservation and the implicit solve
- mass: drift 0 absolute / 0 relative over the window; worst substep drift 4.47044e-16
- ingredient telescoping: worst |sum V div| advective 3.29791e-17, chemical 9.15398e-17
- impermeable wall: 1968 + 2112 closed faces carry a worst flux of 0
- recombination: worst cell-wise |d_adv + d_ch + d_imp - d_full| = 1.11022e-16 against a largest increment of 0.0390936
- implicit solve: 120 solves, iterations 18..18, worst relative residual 5.7189e-07, all converged True

## G. Capture equivalence with the frozen public step
- worst |dphi| versus the eager `pf.step_with_diagnostics`: 1.41392e-08; versus the jitted `pf.step`: 1.41392e-08 (threshold 1e-06)
- worst gap difference over the window: 8.9429e-09 dx (threshold 0.0001)
- meaning: the frozen public step advances its three substeps inside a jitted lax.scan, so XLA fusion changes the last-bit rounding of the elementwise update; the observed differences are reported as measured and are compared against thresholds that are two orders above them and at least three orders below the smallest physical displacement the stage reports

## H. Fixed-support attribution
- linearisation: dn = -(n_phi . n_w)/|grad phi| * dphi on the fixed t=0 support point and normal; coefficient 4.55512 dx per unit phi
- cumulative over the window (dx, positive toward the solid): advective 0.104318, chemical explicit 0.144568 (the near-cancelling half), implicit -0.184466, net chemical -0.0398977, complete 0.0644206
- Cahn-Hilliard cancellation: |net CH| / (|explicit| + |implicit|) = 0.121257 (qualifier below 0.25); per macro step: [0.4713, 0.5916, 0.6542, 0.7375, 0.8798, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7936, 0.808, 0.1908, 0.087, 0.0943, 0.0802, 0.0643, 0.0499, 0.0376, 0.0273, 0.0187, 0.0114, 0.0054, 0.0003, 0.0039, 0.0074, 0.0103, 0.0127, 0.0147, 0.0162, 0.0175, 0.0185, 0.0192, 0.0197, 0.0201, 0.0203, 0.0204, 0.0203, 0.0202, 0.02]
- the frozen global minimum-gap probe moves 0.0626487 dx over the same window; the fixed-support linearisation moves 0.0644206 dx
- fixed support edge {"axis": 1, "i": 95, "j": 23, "t0_fraction": 0.00598143513462881, "t0_x_over_dx": 95.5, "t0_y_over_dx": 23.50598143513463, "t0_radius_over_dx": 0.0}; branch lost at macro step 25 (None means the tracked frozen edge carries the crossing for the whole window)
- prespecified attribution rules: [["PRIMARY_POINT_T0_COEFFICIENT", "point sample at the fixed support point with the t=0 coefficient"], ["POINT_LOCAL_COEFFICIENT", "point sample at the fixed support point with the per-substep coefficient"], ["SUPPORT_CELL_T0_COEFFICIENT", "support-cell value with the t=0 coefficient"]]
    - PRIMARY_POINT_T0_COEFFICIENT: cumulative dn {"adv": 0.104318, "ch": 0.144568, "imp": -0.184466, "full": 0.064421} dx, margin 0.0644206 dx, ranking ADVECTIVE_LARGER
    - POINT_LOCAL_COEFFICIENT: cumulative dn {"adv": 0.102027, "ch": 0.141164, "imp": -0.180294, "full": 0.062898} dx, margin 0.0628977 dx, ranking ADVECTIVE_LARGER
    - SUPPORT_CELL_T0_COEFFICIENT: cumulative dn {"adv": 0.10423, "ch": 0.12623, "imp": -0.166221, "full": 0.064239} dx, margin 0.0642392 dx, ranking ADVECTIVE_LARGER
- margin spread across the prespecified rules: 1.38778e-17 dx; every rule agrees on the ranking: True
    - t=0.002: gap 2.71278 dx (cell [96, 23]), phi(O) 0.498484, v_n(O) 0.00175514, |grad phi| 7.13354, uniqueness NON_UNIQUE
    - t=0.018: gap 2.73504 dx (cell [96, 23]), phi(O) 0.493386, v_n(O) 0.0144101, |grad phi| 7.14727, uniqueness NON_UNIQUE
    - t=0.034: gap 2.7218 dx (cell [96, 23]), phi(O) 0.496404, v_n(O) 0.0255845, |grad phi| 7.14283, uniqueness NON_UNIQUE
    - t=0.05: gap 2.69972 dx (cell [95, 23]), phi(O) 0.501421, v_n(O) 0.0360109, |grad phi| 7.1663, uniqueness NON_UNIQUE
    - t=0.066: gap 2.67169 dx (cell [96, 23]), phi(O) 0.507679, v_n(O) 0.0459383, |grad phi| 7.19986, uniqueness NON_UNIQUE
- per-macro-step fixed-support displacement (dx, positive toward the solid): step, t, advective, CH explicit, CH implicit, net CH, complete
    -  1 t=0.002: adv 5.05034e-05, ch_expl 0.00390192, ch_impl -0.0108579, net_ch -0.00695599, full -0.00690549
    -  2 t=0.004: adv 0.000198244, ch_expl 0.00295737, ch_impl -0.0115268, net_ch -0.00856941, full -0.00837117
    -  3 t=0.006: adv 0.000344843, ch_expl 0.00174493, ch_impl -0.00834761, net_ch -0.00660268, full -0.00625784
    -  4 t=0.008: adv 0.000489813, ch_expl 0.000833457, ch_impl -0.00551749, net_ch -0.00468404, full -0.00419422
    -  5 t=0.01: adv 0.000632804, ch_expl 0.000223591, ch_impl -0.00349582, net_ch -0.00327223, full -0.00263943
    -  6 t=0.012: adv 0.000773828, ch_expl -0.000159001, ch_impl -0.00212308, net_ch -0.00228208, full -0.00150825
    -  7 t=0.014: adv 0.000913021, ch_expl -0.000377375, ch_impl -0.00120815, net_ch -0.00158552, full -0.000672503
    -  8 t=0.016: adv 0.00105054, ch_expl -0.000475614, ch_impl -0.00061249, net_ch -0.0010881, full -3.75649e-05
    -  9 t=0.018: adv 0.00118653, ch_expl -0.000483614, ch_impl -0.00024401, net_ch -0.000727624, full 0.000458907
    - 10 t=0.02: adv 0.00132112, ch_expl -0.000421911, ch_impl -4.18287e-05, net_ch -0.00046374, full 0.000857383
    - 11 t=0.022: adv 0.00145442, ch_expl -0.000304989, ch_impl 3.50932e-05, net_ch -0.000269896, full 0.00118453
    - 12 t=0.024: adv 0.00158652, ch_expl -0.000143367, ch_impl 1.52228e-05, net_ch -0.000128144, full 0.00145838
    - 13 t=0.026: adv 0.0017175, ch_expl 5.50992e-05, ch_impl -8.10877e-05, net_ch -2.59885e-05, full 0.00169151
    - 14 t=0.028: adv 0.00184741, ch_expl 0.000284402, ch_impl -0.000238874, net_ch 4.55281e-05, full 0.00189294
    - 15 t=0.03: adv 0.00197631, ch_expl 0.000539855, ch_impl -0.000446857, net_ch 9.29984e-05, full 0.00206931
    - 16 t=0.032: adv 0.00210425, ch_expl 0.000817737, ch_impl -0.00069631, net_ch 0.000121427, full 0.00222567
    - 17 t=0.034: adv 0.00223125, ch_expl 0.00111506, ch_impl -0.000980351, net_ch 0.000134711, full 0.00236597
    - 18 t=0.036: adv 0.00235736, ch_expl 0.00142942, ch_impl -0.00129347, net_ch 0.000135945, full 0.00249331
    - 19 t=0.038: adv 0.0024826, ch_expl 0.00175886, ch_impl -0.00163125, net_ch 0.000127608, full 0.0026102
    - 20 t=0.04: adv 0.00260698, ch_expl 0.0021018, ch_impl -0.00199007, net_ch 0.000111731, full 0.00271871
    - 21 t=0.042: adv 0.00273052, ch_expl 0.00245698, ch_impl -0.00236699, net_ch 8.99859e-05, full 0.00282051
    - 22 t=0.044: adv 0.00285323, ch_expl 0.00282336, ch_impl -0.0027596, net_ch 6.37584e-05, full 0.00291699
    - 23 t=0.046: adv 0.00297513, ch_expl 0.00320014, ch_impl -0.00316593, net_ch 3.42107e-05, full 0.00300934
    - 24 t=0.048: adv 0.00309622, ch_expl 0.00358668, ch_impl -0.00358436, net_ch 2.31941e-06, full 0.00309854
    - 25 t=0.05: adv 0.0032165, ch_expl 0.00398251, ch_impl -0.00401362, net_ch -3.11079e-05, full 0.00318539
    - 26 t=0.052: adv 0.00333597, ch_expl 0.00438727, ch_impl -0.00445267, net_ch -6.53918e-05, full 0.00327058
    - 27 t=0.054: adv 0.00345463, ch_expl 0.00480072, ch_impl -0.00490068, net_ch -9.99603e-05, full 0.00335467
    - 28 t=0.056: adv 0.00357249, ch_expl 0.00522269, ch_impl -0.00535703, net_ch -0.000134342, full 0.00343815
    - 29 t=0.058: adv 0.00368953, ch_expl 0.0056531, ch_impl -0.00582125, net_ch -0.00016815, full 0.00352138
    - 30 t=0.06: adv 0.00380575, ch_expl 0.00609194, ch_impl -0.00629301, net_ch -0.000201073, full 0.00360468
    - 31 t=0.062: adv 0.00392114, ch_expl 0.00653922, ch_impl -0.00677208, net_ch -0.000232858, full 0.00368828
    - 32 t=0.064: adv 0.00403569, ch_expl 0.00699502, ch_impl -0.00725833, net_ch -0.000263303, full 0.00377239
    - 33 t=0.066: adv 0.0041494, ch_expl 0.00745943, ch_impl -0.00775169, net_ch -0.000292255, full 0.00385715
    - 34 t=0.068: adv 0.00426226, ch_expl 0.00793258, ch_impl -0.00825217, net_ch -0.000319596, full 0.00394266
    - 35 t=0.07: adv 0.00437424, ch_expl 0.00841459, ch_impl -0.00875984, net_ch -0.000345246, full 0.004029
    - 36 t=0.072: adv 0.00448535, ch_expl 0.00890563, ch_impl -0.00927478, net_ch -0.00036915, full 0.0041162
    - 37 t=0.074: adv 0.00459556, ch_expl 0.00940584, ch_impl -0.00979711, net_ch -0.000391277, full 0.00420428
    - 38 t=0.076: adv 0.00470487, ch_expl 0.00991537, ch_impl -0.010327, net_ch -0.000411617, full 0.00429325
    - 39 t=0.078: adv 0.00481326, ch_expl 0.0104344, ch_impl -0.0108646, net_ch -0.000430177, full 0.00438308
    - 40 t=0.08: adv 0.00492071, ch_expl 0.0109631, ch_impl -0.01141, net_ch -0.000446978, full 0.00447374

## I. Residual decomposition
- R_nonlinearity: accumulated |.| = 0.00403141 dx (signed -0.00185116)
- R_interpolation: accumulated |.| = 9.84019e-06 dx (signed 3.73478e-06)
- R_support_migration: accumulated |.| = 1.12354e-09 dx (signed 0)
- R_operator_splitting: accumulated |.| = 6.39621e-15 dx (signed -2.82587e-17)
- R_linearisation_coefficient_drift: accumulated |.| = 0.00265114 dx (signed 0.00265114)
- total residual band: 0.00669238 dx
- the whole-window displacement (dx, positive toward the solid): frozen global gap 0.0626487, bilinear line root 0.0625695, linearised point estimate 0.0644206
- the tracked frozen edge carries the crossing over macro steps 1..24 of 40; from macro step 25 on the frozen linear crossing has moved off it (recorded as a support change of the frozen reconstruction)
- over that common span (24 macro steps) all four measurements agree: frozen global gap 0.00317475, frozen tracked edge 0.00317475, bilinear line root 0.00317101, linearised point estimate 0.00328573 dx; the linearisation overshoots the geometric reconstruction by 0.00011098 dx (3.5 % of the geometric motion) and its total band over that span is 0.00346224 dx
- physical closure between the frozen global probe and the fixed-support linearisation: 0.00177193 dx; between the tracked frozen edge and the linearisation: 0.00185116 dx; frozen support edge alive through the window: False
- algebraic identity residuals (implementation check): {"identity_ingredients_plus_splitting": 1.734723475976807e-18}
- definitions: R_support_migration = d1 - d2 (the frozen global minimum-gap rule against the frozen rule restricted to the tracked support edge); R_interpolation = d2 - d3 (the frozen linear edge crossing against the bilinear root along the fixed interface line); R_nonlinearity = d3 - d4 (the bilinear line root against the linearised point estimate); R_operator_splitting = the algebraic recombination of the three ingredients against the complete update; R_linearisation_coefficient_drift = the change of the local coefficient kappa across the window, which is the assumption behind a fixed-coefficient linearisation. All quantities are vertical displacements in dx units, positive toward the solid.

## J. Validity screen
- criteria: {"fixed_point_probe_band": 0.1, "local_gradient_floor_ratio": 0.5, "max_substep_displacement_over_eps": 0.25, "contour_uniqueness_margin_dx": 0.01, "equivalence_max_phi_diff": 1e-06, "equivalence_max_gap_diff_over_dx": 0.0001}
- failures per guard: {"gradient_resolved": 0, "probe_band": 0, "ray_root_exists": 0, "substep_locality": 0, "frozen_support_edge_alive": 16, "contour_unique": 40}; blocking: {"frozen_support_edge_alive": 16}
- observed: {"min_gradient_ratio_to_t0": 1.0074368611017006, "max_abs_probe_offset": 0.01414246723386614, "max_substep_displacement_over_eps": 0.0019702171834202566, "contour_uniqueness_margins_over_dx": [2.2637891561316792e-11, 7.313261107810831e-11, 1.284412576296745e-10, 1.1532108601386426e-10, 1.4145484783512075e-10, 8.543921126147325e-11, 2.163318413295201e-10, 3.156586103614245e-10, 4.2860648363785003e-10, 4.845475132242427e-10, 4.411369047829794e-10, 4.107967299660231e-10, 4.1888270629897306e-10, 3.375077994860476e-10, 3.3156766221509315e-10, 1.4762591149519722e-10, 2.490452288839151e-11, 3.8252068179644994e-11, 2.6408386588627764e-10, 3.0968294595368207e-10, 1.7926282680491568e-10, 2.986908498314733e-10, 2.6688695697885123e-10, 2.8206770252836577e-10, 3.5423042277216155e-10, 9.305622938882152e-11, 3.248281643664086e-10, 8.846257060213247e-11, 1.2008527505713573e-10, 2.0450130477911443e-10, 5.244125134140631e-10, 7.510259081300319e-10, 7.016645042767777e-10, 8.84941897538738e-10, 1.1776428721077536e-09, 1.3145964317118342e-09, 1.5039489653645433e-09, 1.3270096133055631e-09, 1.5186785162768501e-09, 1.803201143957267e-09], "contour_uniqueness_ok": false, "support_cells_seen": 
- direct frozen probe: UNRESOLVED_NON_UNIQUE_CONTOUR_MATCH_ON_THIS_SURFACE; fixed-support attribution: UNRESOLVED

## K. Decision
- verdict `EARLY_WINDOW_ATTRIBUTION_UNRESOLVED`
- reason: at least one prespecified validity guard failed ({"frozen_support_edge_alive": 16}), so no mechanism is claimed; the per-substep ledger, the attribution table and the residual decomposition are still reported as measured quantities. The frozen linear reconstruction stopped carrying the contact-side crossing on the tracked support edge at macro step 25: the reconstructed curve passed a shared grid node, so its support moved to the neighbouring segment. Every other guard passed, and the reconstructed crossing position itself changes smoothly across that node, but a support change of the frozen reconstruction is a prespecified unresolved condition, so the dominance comparison is not entered
- magnitudes: advective 0.104318 dx, net chemical 0.0398977 dx, margin 0.0644206 dx against a dominance floor of 1.92303e-14 dx (3.0 x the comparison band)
- comparison band (spread of the margin across the prespecified attribution rules plus the algebraic recombination term): 6.41009e-15 dx, margin/comparison band 1.00499e+13; total residual-map band 0.00669238 dx, margin/total map band 9.62596
- recorded, not claimed: the dominance rule returns `ADVECTIVE_TERM_CARRIES_THE_EARLY_WINDOW_CONTOUR_MOTION` with the comparison band and `ADVECTIVE_TERM_CARRIES_THE_EARLY_WINDOW_CONTOUR_MOTION` with the larger total residual-map band; rankings agree across the prespecified rules: True; the sign of the net motion matches the larger term: True
- the verdict is withheld because:
    - blocking validity guards failed: {"frozen_support_edge_alive": 16}
    - the frozen global minimum-gap probe is not unique on this surface: UNRESOLVED_NON_UNIQUE_CONTOUR_MATCH_ON_THIS_SURFACE
- direct frozen probe: UNRESOLVED_NON_UNIQUE_CONTOUR_MATCH_ON_THIS_SURFACE (recorded, not a verdict); fixed-support attribution: UNRESOLVED; capture equivalence holds: True; blocking guards failed: {"frozen_support_edge_alive": 16}; the tracked support edge carried the crossing for 24 of 40 macro steps
- band semantics: the decision compares two contributions of the SAME sampled increments through the SAME positive coefficient, so the band that matters for the comparison is the spread of the margin across the prespecified attribution rules (plus the machine-level algebraic recombination term); the larger total-map band is reported separately and used only to bound the total-displacement closure, it is not the uncertainty of a comparison
- Cahn-Hilliard near-cancellation qualifier: True (ratio 0.121257)
- scope: {"case": "pillar_training", "t_window": [0.0, 0.08000000000000002], "macro_steps_run": 40, "dt": 0.002, "window_limited_to_early_response": true, "contact_threshold_reached": false, "contact_events_scored": 0}
- forbidden claims respected: {"no_contact_mechanism_claim": true, "no_impact_claim_from_early_displacement": true, "no_extrapolation_to_late_contact_time": true, "no_refinement_or_convergence_claim": true, "no_production_initializer_promotion": true}
- the early-window displacement reported here is a pre-contact response of the frozen initial state; it is not a real impact and it is not extrapolated to the late contact time

## L. Adversarial and falsification checks
- `linearisation_recovers_a_known_translation`: {"rows": [{"offset_over_dx": -0.5, "wall_normal": [0.0, 1.0], "expected_displacement_toward_wall_over_dx": -0.5, "recovered_displacement_toward_wall_over_dx": -0.5125689235396458, "absolute_error_over_dx": 0.012568923539645782, "linearised_estimate_holds": true}, {"offset_over_dx": -0.1, "wall_normal": [0.0, 1.0], "expected_displacement_toward_wall_over_dx": -0.1, "recovered_displacement_toward_wall_over_dx": -0.1033491422270814, "absolute_error_over_dx": 0.0033491422270813948, "linearised_estimate_holds": true}, {"offset_over_dx": 0.1, "wall_nor
- `linearisation_sign_flips_with_the_wall_normal`: {"rows": [{"offset_over_dx": -0.5, "wall_normal": [0.0, -1.0], "expected_displacement_toward_wall_over_dx": 0.5, "recovered_displacement_toward_wall_over_dx": 0.5125689235396458, "absolute_error_over_dx": 0.012568923539645782, "linearised_estimate_holds": true}, {"offset_over_dx": 0.5, "wall_normal": [0.0, -1.0], "expected_displacement_toward_wall_over_dx": -0.5, "recovered_displacement_toward_wall_over_dx": -0.5125689235396458, "absolute_error_over_dx": 0.012568923539645782, "linearised_estimate_holds": true}], "holds": true}
- `gradient_guard_refuses_a_flat_profile`: {"guards": {"gradient_resolved": false, "probe_band": true, "ray_root_exists": true, "substep_locality": true}, "refused": true}
- `probe_band_guard_refuses_a_drifting_probe`: {"guards": {"gradient_resolved": true, "probe_band": false, "ray_root_exists": true, "substep_locality": true}, "refused": true}
- `ingredient_identity_catches_a_fabricated_increment`: {"healthy_identity_over_dx": 4.92227786308419e-17, "broken_identity_over_dx": 0.004555118236463979, "discriminates": true}
- `verdict_rule_degrades_to_mixed_on_a_near_tie`: {"near_tie": "MIXED_TERMS_WITHIN_THE_RESIDUAL_BAND", "clear_win": "ADVECTIVE_TERM_CARRIES_THE_EARLY_WINDOW_CONTOUR_MOTION", "unresolved_win": "EARLY_WINDOW_ATTRIBUTION_UNRESOLVED", "degrades": true, "grants_on_a_clear_win": true, "withholds_when_unresolved": true}
- `contour_uniqueness_record_is_self_consistent`: {"status_consistent": true, "margins_over_dx": [2.2637891561316792e-11, 7.313261107810831e-11, 1.284412576296745e-10, 1.1532108601386426e-10, 1.4145484783512075e-10, 8.543921126147325e-11, 2.163318413295201e-10, 3.156586103614245e-10, 4.2860648363785003e-10, 4.845475132242427e-10, 4.411369047829794e-10, 4.107967299660231e-10, 4.1888270629897306e-10, 3.375077994860476e-10, 3.3156766221509315e-10, 1.4762591149519722e-10, 2.490452288839151e-11, 3.8252068179644994e-11, 2.6408386588627764e-10, 3.0968294595368207e-10, 1.7926282680491568e-10, 2.986908
- `frozen_sources_are_unmodified`: {"hashes_hold": true, "note": "the frozen solver, the frozen audit source and the L1A-2u/2s helpers re-hash to the values recorded at generation time"}
- `window_budget_is_capped`: {"refused_over_cap_request": true, "cap": 40}
- `ledger_is_not_vacuous`: {"max_abs_mass_adv": 1.845726874880784e-20, "max_abs_mass_ch": 7.000265481357666e-20, "closed_faces_exist": true}
- `no_forbidden_claim_in_the_decision_payload`: {"forbidden_phrases": ["real impact", "impact mechanism", "contact event occurred", "confirmed contact"], "found": [], "holds": true}
- `all_hold`: true
- all checks hold: True

## M. Quality, tests and hosted CI
- quality baseline sha256 `52b41ab68b4d6b86aaa254389eb5398e65ad8c5cbdcd12626c0466248dd3d70b`
- frozen adjacent suite: SUPERSEDED_BY_NEXT_STAGE_BY_CONSTRUCTION, suite failures {"l1a2t": 1, "l1a2u": 0, "l1a2v": 5, "l1a2w": 0}
    - failing: tests/test_nearwall_taper_interface_transport_audit.py::test_manifest_mismatch_reference_is_recorded_in_preflight
    - failing: tests/test_interior_speed_admissibility_screen_audit.py::test_required_evidence_files_are_declared_and_complete
    - failing: tests/test_interior_speed_admissibility_screen_audit.py::test_manifest_self_verification_and_tamper_detection
    - failing: tests/test_interior_speed_admissibility_screen_audit.py::test_evidence_verification_rejects_a_moved_envelope
    - failing: tests/test_interior_speed_admissibility_screen_audit.py::test_reseal_records_the_observed_ci_status_without_touching_physics
    - failing: tests/test_interior_speed_admissibility_screen_audit.py::test_report_carries_every_required_section
- old L1A-2s lineage failure, reported separately: ['tests/test_impact_initialization_compatibility_audit.py::test_l1a2r_merged_base_verified_or_fail_closed'] (FROZEN_LINEAGE_DRIFT_CONFIRMED)
- not run: ["control trajectories", "contact events", "temporal refinement", "spatial refinement", "T=8 recomputation", "L1B data generation", "ML training", "contract 13"]
- declared new diagnostics 0, expectation `PASS_BASELINE_ONLY`
- hosted CI observed after push: `UNMEASURED` (a re-seal records the observed result)

## N. Exactly one next action
- Prespecify, without editing the frozen event rule, a displacement readout whose support is the reconstructed contour arc itself (the bilinear line root along the frozen interface normal, already recorded in this stage's ledger on every macro step) and then re-run this stage's identical single-case window under that readout, so that the blocking condition becomes the continuity of the reconstructed position instead of the identity of the grid edge that carries it. Diagnose the frozen global probe's non-uniqueness on this surface in the same run and report it separately. No mechanism may be claimed until every blocking guard passes.
- bounded scope: {"cases": [], "dt": 0.002, "macro_step_cap": 40, "window": "t <= 0.08", "solver": "frozen, unmodified", "criteria": "this stage's, unchanged", "no_extension_toward_contact": true, "no_control_arms": true, "no_refinement": true, "no_generator_or_ml": true}
- must not be started in this stage: True
