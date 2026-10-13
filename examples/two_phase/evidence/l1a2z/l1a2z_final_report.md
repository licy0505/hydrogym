# L1A-2z - prespecified own-base attribution and unique-support measurement

## A. Scope, authority and baseline
- baseline: PR #24 head `cd20a7838fb616f3857092c580f3c24802c6115a` (L1A-2y); verified main `09e2993b8da478d79e768054194201af57121784`; branch `arena/02b6c71b-hydrogym` at head `0abf257c53528561a1c4b22a1defb04c840c492a`; solver contract 12
- one case `pillar_training`, N=192, dt=0.002, 40 macro steps of at most 40 (120 frozen substeps, t <= 0.08)
- trajectory count 1, control arms 0, contact events scored 0, refinements 0, T=8 recomputed False, L1B data generated False
- prespecification: `examples/two_phase/l1a2z_prespecification.md` sha256 `488568ab96221883...`, commit `7a9fdf0eea48`, ancestor of head True, measurement-code hash-block syncs declared 2 / observed 2
- frozen inputs re-verified: L1A-2y verified True over 22 bound files; the whole frozen chain L1A-2t/u/v/w/x verified True over 4 manifests, 57 bound files

## B. Why the L1A-2y primary rule could not close
- the frozen L1A-2x substep is `adv`, then `ch_explicit` (both folded into `phi_hat`), then `ch_implicit`; writing it any other way is not what the solver executed
- L1A-2y applied each ingredient to the same substep base and then chained each ingredient's own increments across the window, so its three totals measured displacements of partial fields that telescope to nothing in particular
- the sealed consequence: its primary rule summed to -0.12231966 dx against a total of 0.062648682 dx, a closure of -0.12231966 dx - 12.2 times its own 0.01 dx band
- the frozen event threshold `CONTOUR_UNIQUENESS_MARGIN_DX = 0.01` dx is unchanged in the frozen module, which this stage imports rather than edits

## C. The corrected rule and why it closes
- measure `Q` on the consecutive actual states of each frozen substep: `D_adv = sum_k [Q(S1_k) - Q(S0_k)]`, `D_ch = sum_k [Q(S2_k) - Q(S1_k)]`, `D_imp = sum_k [Q(S3_k) - Q(S2_k)]`
- the sum telescopes to `Q(phi_final) - Q(phi_initial)` by construction; the run reports the floating-point residual of that identity rather than assuming it
- closure in this window: total 0 dx, worst substep 0 dx, band 0.01 dx
- the identity is reported as a closure check and is never presented as evidence that a physical operator caused the motion

## D. The frozen L1A-2x uniqueness diagnostic, retained and explained
- the frozen diagnostic was recomputed on every intermediate state: statuses ["NON_UNIQUE"], margin range [0, 1.85041e-09] dx
- the reason is structural, not numerical: the two lowest-gap `phi = 0.5` crossings at t = 0 are the mirror pair `x = 2.984375` and `x = 3.015625` at the same height, and the frozen SDF is exactly linear in the normal coordinate there, so both sit at the same wall distance to machine precision
- the frozen rule is retained verbatim as a historical comparator. Its verdict is not the verdict of this stage and no frozen file was edited to change it

## E. The prespecified branch-uniqueness rule and its t = 0 degeneracy
- candidates: every reconstructed crossing (176 at t = 0); components: connected components of the same reconstruction built from the frozen per-cell segments and validated cell by cell against them (176 cells, matches True)
- minimum: `g*` = 2.7059815 dx; tie tolerance 1e-06 dx selects 2 tied candidates; competition tolerance 0.01 dx admits 2
- the tied candidates lie on 1 component(s): the point degeneracy is inside one branch, so the branch is unique and the point is not. The point degeneracy is recorded, both tied supports are measured, and their difference is reported as 1.8032011e-09 dx
- deterministic tie-break: the smallest-x tied candidate (PRIMARY) is the reporting support; it is never used as evidence of uniqueness and no region was shrunk to leave one candidate
- the frozen minimum-gap rule reproduced exactly: {"point_matches": true, "gradient_norm_difference": 0.0, "wall_normal_matches": true, "interface_normal_matches": true, "gap_matches": true, "holds": true}

## F. The corrected attribution in dx
- primary convention `SEQUENTIAL_OWN_BASE` on support `PRIMARY`, along the frozen wall normal, positive toward the solid, raw dx, nothing normalised:
    - adv: 0.101044289
    - ch_explicit: 0.131055775
    - ch_implicit: -0.169451383
    - ch_pair: -0.0383956072
    - full: 0.0626486819
- the two Cahn-Hilliard halves are one update split by the implementation into an explicit flux and an implicit solve; they are never presented as two independent mechanisms
- the other prespecified conventions, for the record (dx): {"SEQUENTIAL_OWN_BASE": {"adv": 0.101044289, "ch_pair": -0.038395607, "full": 0.062648682}, "MARGINAL_OWN_BASE": {"adv": 0.101044289, "ch_pair": -0.046332665}, "LINEARISED_POINT_T0": {"adv": 0.104318318, "ch_pair": -0.039897708, "full": 0.06442061}, "LINEARISED_SUPPORT_CELL_T0": {"adv": 0.104229717, "ch_pair": -0.039990542, "full": 0.064239175}}
- per macro step (dx): [-0.00680327, -0.00809743, -0.00598449, -0.00399232, -0.0025097, -0.00143475, -0.00063996, -3.454e-05, 0.00044044, 0.00082295, 0.00113796, 0.00140235, 0.00162792, 0.00182314, 0.00199428, 0.00214612, 0.00228241, 0.00240613, 0.0025197, 0.00262508, 0.00272393, 0.00281761, 0.00290728, 0.0029939, 0.00308592, 0.00324559, 0.00332098, 0.00339508, 0.00346831, 0.00354099, 0.0036134, 0.00368574, 0.00375816, 0.0038308, 0.00390372, 0.00397699, 0.00405064, 0.00412468, 0.00419909, 0.00427385]

## G. Raw closure and residual decomposition
- every term, raw, in dx:
    - R_sequential_telescoping_total: 0
    - R_sequential_telescoping_max_substep: 0
    - R_marginal_interaction: -0.007937057755
    - R_ch_pair_sequential_vs_marginal: 0.007937057755
    - R_support_tie_margin: 1.803201144e-09
    - R_root_matching_noise: 0
    - R_nonlinearity_point_vs_geometric: -0.001771928513
    - R_nonlinearity_cell_vs_geometric: -0.001590493041
    - R_convention_spread: 0.007937057755
    - R_order_swap_pair_total: 4.480670448e-07
    - R_phi_hat_identity: 2.220446049e-16
    - R_operator_recombination: 1.110223025e-16
    - R_mass_residual: 4.470435647e-16
- comparison error 0.0079370578 dx made of {"convention_spread": 0.007937058, "support_tie_spread": 2e-09, "root_matching_noise": 0.0, "prespecified_floor": 0.001}
- dominance margin 0.062648682 dx against the frozen floor of 3.0 x the comparison error = 0.023811173 dx

## H. Comparison with L1A-2y and L1A-2x
- L1A-2y primary closure -0.12231966 dx versus this stage's 0 dx on the identical window, with the frozen band unchanged at 0.01 dx
- L1A-2y own-base comparator closure -0.0079370578 dx is the closest thing that stage recorded to this rule; the difference between the two is the base anchoring, not a threshold
- identical-window check against the sealed ledger: roots differ by at most 0 dx over 40 macro steps; per-substep increment differences {"max_abs_difference_adv": 0.0, "max_abs_difference_ch": 0.0, "max_abs_difference_imp": 0.0, "max_abs_difference_full": 0.0}
- L1A-2x carry-over unchanged: its four-way closure and its halted macro-25 support crossing are read from its sealed payload and are not recomputed here as new evidence

## I. Support uniqueness over the window
- intermediate states checked: 360, rule satisfied 360, non-unique 0, exact-threshold 0, enumeration mismatches 0, multi-component competition 0, branch reassignments 0
- the t = 0 tie is inside one branch and stays inside it for the whole window; the recorded segment change at macro 25 / substep 74 is a within-branch event and is reported rather than resolved away

## J. Identical-window reconstruction check
- initial and final state hashes match the sealed L1A-2y window: True and True; tolerance 0.05 dx; view holds True
- a trajectory difference would have been treated as a separate regression and would have blocked the verdict; the check is recorded either way

## K. Conservation, solver convergence and capture equivalence
- mass drift 0 (relative 0), worst substep mass residual 4.47044e-16, worst closed-face flux 0, CG iterations 18..18, all converged True
- capture versus the frozen public step: phi 1.41392e-08 eager / 1.41392e-08 jitted, gap 8.9429e-09 dx, within the frozen thresholds True

## L. Adversarial and falsification results
- 19 checks, all hold True, failed []
    - exact_telescoping_identity_on_a_synthetic_operator_sequence: True
    - a_wrong_base_or_wrong_pairing_is_detected: True
    - a_mixed_family_sum_is_not_admissible_even_when_it_lands_inside_the_band: True
    - operator_order_is_recorded_and_the_swap_is_measured: True
    - nonlinear_contour_response_is_measured_not_assumed: True
    - the_frozen_closure_gate_can_fail: True
    - a_rule_that_improves_the_number_by_rescaling_one_leg_is_rejected: True
    - the_frozen_l1a2x_diagnostic_is_retained_and_not_redefined: True
    - the_prior_evidence_is_unmodified_by_this_run: True
    - candidate_permutation_and_enumeration_order_do_not_move_the_answer: True
    - the_tied_second_candidate_lies_on_the_tracked_arc: True
    - a_broken_or_absent_observable_is_caught_by_the_gate: True
    - two_distinct_branches_near_the_wall_are_reported_as_non_unique: True
    - an_exact_threshold_grid_node_is_flagged: True
    - the_recorded_support_switch_stays_on_one_branch: True
    - the_branch_identity_relation_table_is_discriminating: True
    - the_tie_break_is_never_used_as_evidence_of_uniqueness: True
    - the_identical_window_tolerance_can_reject: True
    - evidence_mutation_is_rejected_by_the_independent_verifier: True
- the gate is demonstrably able to fail: the sealed L1A-2y primary rule misses the same 0.01 dx band that this stage uses, and a mixed-family sum is shown to be able to land inside that band, which is why the prespecified rule and not the band is the contract

## M. Quality ratchet, stage tests and frozen adjacent suites
- legacy debt ratchet: expected `PASS_BASELINE_ONLY` against {"ruff": 171, "formatter": 27, "isort": 28, "codespell": 3}, new-code diagnostics declared 0
- stage suite: 70 passed in 2.58s - `JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest tests/test_l1a2z_sequential_ownbase_attribution_audit.py -q`
- predecessor stage suite: 71 passed in 11.88s - declared structural failures observed [], declared but not observed []
- frozen adjacent suites: 7 failed, 222 passed in 74.68s (0:01:14) - declared structural set {"l1a2t": 1, "l1a2u": 1, "l1a2v": 5, "l1a2w": 1, "l1a2x": 0}; legacy lineage suite 1 failed, 48 passed in 39.11s, pre-existing
- observations complete True; failures outside the declared classes []
- hosted CI: no result is claimed here; the observed post-push result is recorded by the re-seal step and this line is replaced with it

## N. Validity screen and the verdict
- verdict: `EARLY_WINDOW_MEASUREMENT_ATTRIBUTION_RESOLVED`
- A measurement validity True; B mathematical attribution closure True; C physical mechanism False - this stage never answers C
- dominance reading `ADVECTIVE_LEG_LARGER` with margin 0.062648682 dx against floor 0.023811173 dx
- withheld because: []
- measurement attribution resolved under the specified convention True
- validity screen blocking failures: {}

## O. Limitations and what would be needed for a mechanism claim
- no causal physical mechanism is claimed
- the two Cahn-Hilliard halves are never presented as two independent mechanisms
- no contact-time or late-time inference is made from this early window
- no claim is made that the measurement convention is the only admissible one
- scope: one case, one dt, one trajectory, t <= 0.08, zero contact events. Nothing here is a statement about the impact at t ~ 0.354, about a real contact mechanism, about a grid or time-step refinement, or about any parameter outside the frozen set
- the two Cahn-Hilliard halves, the four attribution conventions and the support tie are all measurement choices; a resolved attribution under this convention is not a statement that this convention is the only admissible one

## P. Exactly one next action
- Design, with the owner, one controlled comparison that could turn the resolved measurement attribution into a testable statement about a physical mechanism - for example a prespecified second trajectory whose advective or Cahn-Hilliard increment is suppressed on the same initial state - and freeze it before any such trajectory is run; no trajectory, no control arm and no physics change is started in this stage
- not started in this stage and requiring owner approval before it is started
- bounded scope of any approved follow-up: {"cases": [], "criteria": "this stage's, unchanged", "dt": 0.002, "macro_step_cap": 40, "no_control_arms": true, "no_extension_toward_contact": true, "no_generator_or_ml": true, "no_refinement": true, "solver": "frozen, unmodified", "window": "t <= 0.08"}
