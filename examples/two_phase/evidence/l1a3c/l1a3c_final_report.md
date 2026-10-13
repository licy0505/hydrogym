# L1A-3c — Zero-Mobility Admissibility and Single-Control Early-Window Experiment

**Verdict: `CONTROL_OBSERVABLE_INVALID`. `PHYSICAL_MECHANISM_NOT_ESTABLISHED`.**

Exactly one baseline replay and exactly one zero-mobility control were executed. No solver, earlier evidence, frozen root gate or quality baseline was changed. No experiment was repeated.

## 1. Published head and PR status

```json
{
  "branch": "arena/02b6c71b-hydrogym",
  "final_published_head_and_CI": "To be observed after the report/evidence commit, recorded in PR #24 publication appendix; not inferred here",
  "head_at_report_generation": "bd7a8fc7ddcb43c2e93ad78eac45b60238417690",
  "isDraft": true,
  "pull_request": 24,
  "state": "OPEN"
}
```
The scientific report cannot contain the SHA of the commit containing itself. Its final-head publication and observed hosted checks will be appended to PR #24, preserving all earlier sections. PR remains OPEN/Draft.

## 2. Mathematical and discrete zero-mobility admissibility

For finite fields, shared-face stiffness K and L=V_safe^-1 K, the executed phase substep is b=phi+h*(a-div J_CH), J_CH=-M*w*delta(mu_exp), followed by (I+h*M*eps*L^2)*phi_new=b. At M=0, CH flux is zero and the implicit matrix is I, condition number 1. The existing exchange CG sees zero correction rhs and exits with zero iterations/residual; no inverse-M denominator or unsupported branch is introduced. The phase solver is still called. Chemical potential, wall energy, capillary forcing, momentum, Brinkman damping and pressure projection remain. Suppressing diffusive mobility is not deleting capillary coupling. Conservative shared-face advection remains and telescopes in weighted mass. This does not identify numerical interface motion with exact material velocity.

## 3. Exact mobility-dependent solver code paths

```json
{
  "all_M_attribute_occurrences": [
    {
      "expression": "return -p.M * weight_x * grad_x, -p.M * weight_y * grad_y",
      "function": "chemical_potential_fluxes",
      "line": 2143
    },
    {
      "expression": "return -p.M * weight_x * grad_x, -p.M * weight_y * grad_y",
      "function": "chemical_potential_fluxes",
      "line": 2143
    },
    {
      "expression": "alpha = jnp.asarray(float(dt) * float(p.M) * float(p.eps), dtype=rhs_field.dtype)",
      "function": "solve_ch_implicit",
      "line": 3065
    },
    {
      "expression": "\"implicit_alpha\": float(dt) * float(p.M) * float(p.eps),",
      "function": "implicit_solve_diagnostics",
      "line": 3108
    },
    {
      "expression": "\"spectral_bound_estimate\": float(dt) * float(p.M) * float(p.eps) * float(jnp.max(diagonal)) ** 2,",
      "function": "implicit_solve_diagnostics",
      "line": 3109
    },
    {
      "expression": "source_hat = jnp.fft.rfft2(phi_rhs) - p.M * m2 * jnp.fft.rfft2(mu_expl)",
      "function": "_phase_update",
      "line": 3143
    },
    {
      "expression": "denominator = 1.0 + dt * p.M * p.eps * m2**2",
      "function": "_phase_update",
      "line": 3142
    }
  ],
  "boundary": "shared apertures close wall/seam flux; no separate mobility wall flux",
  "chemical_potential": "mu=fprime(phi)/eps+wall_derivative-eps*fluid_laplacian(phi); independent of M",
  "conditioning": "A=I has condition number 1; no inverse mobility or zero-mobility preconditioner",
  "conservation": "sum(V*div(face_flux)) telescopes; implicit identity adds no exchange at M=0",
  "explicit": "J_CH,f=-M*w_f*(mu_exp,j-mu_exp,i); b=phi+h*(a-div(J_CH))",
  "implicit": "A=I+h*M*eps*L^2; L=V_safe^-1*K; solve A*phi_new=b in the V inner product",
  "legacy_unexecuted": "projection_legacy FFT denominator=1+h*M*eps*m2^2; at zero it is 1; not selected",
  "limitations": "Finite fields/geometry only; discrete advection persists; no material-velocity claim.",
  "momentum": "rhs retains +(SIGMA_NORM/We)*mu*grad(phi)/rho_l, viscosity, advection, gravity flag",
  "normalization": "CG floors/denominators depend on rhs/residual, not 1/M; zero residual skips body",
  "pressure_and_Brinkman": "unchanged damping then frozen central-D/G periodic pressure projection",
  "source_sha256": "024665742dce67bc970052e9760fa375ccda096f534cae5faaf2b446c11fe27c",
  "zero_limit": "J_CH=0, A=I, correction_rhs=b-A*b=0; zero CG iterations, phi_new=b"
}
```
The selected path is impermeable_flux / sdf_cutcell_fv_v1 / phase_only_float64_v1, contract 12. The unselected legacy FFT branch is audited but not used. No matrix/preconditioner is modified.

## 4. Prespecification boundary and source hashes

```json
{
  "holds": true,
  "prespecification_commit": "7eb3898b974ee2cc3b4aa8a44d6348b79b587273",
  "source_hashes": {
    "examples/two_phase/evidence/l1a2z/manifest.json": "65d42c5fc11c244975d8cf40a06d8a4fdd26884830677c36c8697eb4e63f94e4",
    "examples/two_phase/evidence/l1a3a/manifest.json": "c5c32bcd9ee56f7ac9188368bf78192a2ff076ef51ea3ac9c3b6cfb5fd84dfb8",
    "examples/two_phase/evidence/l1a3a/phase_a_audit.json": "890cbfc96f48bfc4695fbb55f9c8ac41bf135d4b6b221e6de2b4bf72f367ae80",
    "examples/two_phase/evidence/l1a3b/manifest.json": "29157b647adf6ce1701bee60b5d0fb77322fd77f5b1aeb73c801d909d5d11845",
    "examples/two_phase/l1a2z_prespecification.md": "488568ab96221883bda7233b586f645f1ebc64694e8b6bc876774473ba3449eb",
    "examples/two_phase/l1a3b_prespecification.md": "d280a12919ff04a8b5a0705185d0909bafb56c5b836091e8c8df75bfea1025a7",
    "examples/two_phase/l1a3c_prespecification.md": "5bc54b90edf4bbed4cf735442f1c780930979d3e3989693b19eec173858afee0",
    "examples/two_phase/phasefield.py": "024665742dce67bc970052e9760fa375ccda096f534cae5faaf2b446c11fe27c",
    "examples/two_phase/production/frozen_chns_early_window_transport_attribution_audit.py": "f0fddeabfdd4498123c84f8cd9125470056e589ef6e2241f2448a12eb3d0214e",
    "examples/two_phase/production/frozen_reconstruction_branch_attribution_audit.py": "d6c9e05a42384b227faa6d86b47284984d14e0255d7ba17345fd359ad7f51131",
    "examples/two_phase/production/impact_initialization_compatibility_audit.py": "788791df2b0bb3e737fe68cba6e5cd3e98d100285184e0b727ce02363309a3b0",
    "examples/two_phase/production/l1a2y_contour_branch_readout.py": "563553a398f2eea53bf0cc73d53047ea483c37d6686d2572f92ddf3fad540e97",
    "examples/two_phase/production/l1a2z_sequential_ownbase_attribution_audit.py": "1c1815991e929860d377855a2d4c46d890040da0d0ee1733ff9f675784f5e858",
    "examples/two_phase/production/l1a3b_support_indexed_observable_audit.py": "64ea61b912630cd35b8ed3dd1bfcf51a733b12373325340e4983b9814625a104",
    "examples/two_phase/production/l1a3c_zero_mobility_control_audit.py": "e56c3a2e1bf9a8699bb4096374d7e5003734a37c9f27e5d02398fd55b72c1345",
    "examples/two_phase/tests/test_l1a2z_sequential_ownbase_attribution_audit.py": "0f329a0fd0b62f9de411e5f3de544b53e60378e964887b49878cb99037ec4a70",
    "examples/two_phase/tests/test_l1a3b_support_indexed_observable_audit.py": "932ce4f2d5fc7ab37d05eb71e962833a9e75708786a728e1d955db2c9993b686",
    "examples/two_phase/tests/test_l1a3c_zero_mobility_control_audit.py": "8d68d4fdad3153e247e2326dd53530d9a385f94a2c52723f1cc5dbb5afc52f5c"
  }
}
```
The protocol/source/tests were committed before either target arm. No postmeasurement source/test/protocol change occurred. Protocol boundary 7eb3898b974ee2cc3b4aa8a44d6348b79b587273 precedes measurement; actual execution heads are recorded in exclusive receipt files.

## 5. Baseline reproducibility and replay

```json
{
  "full_contour_replay": true,
  "historical_endpoints_dx": {
    "RAY_95": 0.0626486818930445,
    "RAY_96": 0.0626486836962457
  },
  "holds": true,
  "max_root_difference_dx": 0.0,
  "state_hashes_match_every_intermediate": true
}
```
Sealed historical records lacked intermediate phi/candidate arrays, so one full baseline replay was prespecified. Initial physical state, geometry, parameters and 360 intermediate phi hashes plus final phi/u/v hashes reproduced the sealed trajectory exactly. Root difference was zero; independent replay verification preceded control. All 360 contours and candidate lists were newly reconstructed without modifying historical evidence. Frozen baseline endpoints: d95=+0.0626486818930445 dx, d96=+0.0626486836962457 dx.

## 6. Control execution status

One M=0 fully coupled pillar_training arm: N=192, dx=dy=0.03125, dt=0.002, 40 macro steps / 120 substeps, t<=0.08. Initial-state hashes were identical; every unrelated dataclass parameter was identical, and only M changed. Exclusive STARTED_NO_RETRY_PERMITTED receipts preceded evolution. Complete initial/final state hashes and all readouts are in baseline.json/control.json. Both arms finished with no contact-guard trigger or single-arm numerical failure.

```json
{
  "baseline_final": {
    "phi": "74e1d3f94c7a063f4f78bc69cd2b54bb7fe3481e8a4b8154a7dce1404545e5cf",
    "u": "8283f068cb571d5022393e897bcccffafdceeed457895bf0fdc9e5b345b4f728",
    "v": "d2a4fad0deb00144816324f6fd0b9f39110fc23bafffb894a8308b63c5e7a0cf"
  },
  "baseline_initial": {
    "phi": "3022be21ea6a46ebe2e74b3a4903ae96b8e4527cbafadf1ba0ce0f0f7c14245c",
    "u": "2b921fbcd185c4fe36173e804b91ddefb4840c580a3410548948f27057c455e9",
    "v": "3c0963be8ef830f47db9b613ee3a7da64e1fd8694a04cd72be2d95d31a7adf37"
  },
  "control_final": {
    "phi": "1f0d238ee2d05cb911fb78f01e2379194596b563168a958321b56733aaa51f05",
    "u": "35892070b4aabc116467a45390bc38b8c88e409aedecf4ddf90ea42515a26fef",
    "v": "88772ac175a60347fc4bc02dda370ea5426bcd91568d863ad2abd5097eeb398f"
  },
  "control_initial": {
    "phi": "3022be21ea6a46ebe2e74b3a4903ae96b8e4527cbafadf1ba0ce0f0f7c14245c",
    "u": "2b921fbcd185c4fe36173e804b91ddefb4840c580a3410548948f27057c455e9",
    "v": "3c0963be8ef830f47db9b613ee3a7da64e1fd8694a04cd72be2d95d31a7adf37"
  }
}
```
## 7. Both support-specific displacements

These are individually valid single-arm displacements, not a valid final intervention-effect measurement. Positive displacement is toward the solid.

| Arm | d95 (dx) | d96 (dx) |
|---|---:|---:|
| baseline | +0.062648681893044511 | +0.062648683696245655 |
| control | +0.099010366113294168 | +0.099010358939125354 |

Complete Q and displacement time series, all 360 intermediate records per arm and 40 macro boundary vectors are bound in support_displacements.json. No primary ray or scalar displacement is reported.

## 8. Complete paired effect vector

E_j=d_j(control)-d_j(baseline), positive if zero mobility advances farther toward solid. The complete final E=[E95,E96] is **unavailable/INVALID**, not zero. 311/360 pointwise comparisons pass; 49 fail. The entire required early-window paired protocol and final pair fail. The full time-indexed validity/effect record is in paired_response.json. No final component is inferred from the two individually valid trajectories or extrapolated from earlier passing states.

```json
{
  "E_dx": null,
  "absolute_support_to_support_effect_disagreement_dx": null,
  "no_partial_support_effect_reported": true,
  "raw_signed_effect_range_dx": null,
  "sign_discordant": null,
  "status": "INVALID",
  "validity_failures": [
    "control:PAIRED_GEOMETRIC_BRANCH_CORRESPONDENCE_LOST"
  ]
}
```
## 9. Raw support disagreement and signs

```json
{
  "blocker": "PAIRED_GEOMETRIC_BRANCH_CORRESPONDENCE_LOST",
  "final_absolute_support_effect_disagreement_dx": null,
  "final_displacement_support_difference_dx": {
    "baseline": -1.803201143957267e-09,
    "control": 7.174168814572113e-09
  },
  "final_failed_pair": {
    "effect": {
      "E_dx": null,
      "absolute_support_to_support_effect_disagreement_dx": null,
      "no_partial_support_effect_reported": true,
      "raw_signed_effect_range_dx": null,
      "sign_discordant": null,
      "status": "INVALID",
      "validity_failures": [
        "control:PAIRED_GEOMETRIC_BRANCH_CORRESPONDENCE_LOST"
      ]
    },
    "paired_correspondence": {
      "holds": false,
      "relation": {
        "crossings_after": 178,
        "crossings_before": 176,
        "max_min_distance_after_to_before_over_dx": 0.4275161900580997,
        "max_min_distance_before_to_after_over_dx": 0.4610161732986212,
        "reassignment": true,
        "relation": "DISJOINT_REASSIGNMENT",
        "tolerance_over_dx": 0.375
      },
      "status": "CORRESPONDENCE_LOST",
      "topology_event": "DISJOINT_REASSIGNMENT"
    },
    "state_name": "s3",
    "substep_index": 119
  },
  "final_paired_E_dx": null,
  "final_paired_sign_discordance": null,
  "final_signed_effect_range_dx": null,
  "final_signed_support_effect_disagreement_dx": null,
  "first_failed_pair": {
    "effect": {
      "E_dx": null,
      "absolute_support_to_support_effect_disagreement_dx": null,
      "no_partial_support_effect_reported": true,
      "raw_signed_effect_range_dx": null,
      "sign_discordant": null,
      "status": "INVALID",
      "validity_failures": [
        "control:PAIRED_GEOMETRIC_BRANCH_CORRESPONDENCE_LOST"
      ]
    },
    "paired_correspondence": {
      "holds": false,
      "relation": {
        "crossings_after": 178,
        "crossings_before": 178,
        "max_min_distance_after_to_before_over_dx": 0.37576366060551897,
        "max_min_distance_before_to_after_over_dx": 0.5026621620898484,
        "reassignment": true,
        "relation": "DISJOINT_REASSIGNMENT",
        "tolerance_over_dx": 0.375
      },
      "status": "CORRESPONDENCE_LOST",
      "topology_event": "DISJOINT_REASSIGNMENT"
    },
    "state_name": "s1",
    "substep_index": 89
  },
  "interpretation": "Single-arm roots stay valid. The frozen same-time global-branch relation fails; this is not proof that the physical branch changed identity. The complete paired protocol fails closed.",
  "large_disagreement_is_not_automatically_invalid": true,
  "no_partial_or_extrapolated_final_effect": true,
  "noncontiguous_pointwise_valid_pairs": "Some later states pass independently; none repairs a failed complete-window comparison or authorizes extrapolation.",
  "pair_status_counts": {
    "INVALID": 49,
    "VALID": 311
  },
  "pointwise_valid_sign_counts": {
    "{'RAY_95': -1, 'RAY_96': -1}": 1,
    "{'RAY_95': 0, 'RAY_96': 0}": 1,
    "{'RAY_95': 1, 'RAY_96': 1}": 309
  }
}
```
The first rejected pair is S1 at substep index 89 (macro step 30): cross-trajectory geometric distances exceed the frozen 0.375 dx threshold. The final distances are 0.4610161732986212 and 0.4275161900580997 dx. This relation label is a fail-closed geometric diagnostic, not proof of physical branch reassignment. Single-arm tracking and component counts remain valid. Final paired signed range, effect signs and support disagreement are null because the required pair is invalid; raw single-arm support differences above are distinct diagnostics, not substitute effects. All valid pointwise effects retain both signs, signed range and disagreement. No averaging, preferred component, favorable subset or support-independent sensitivity claim.

## 10. Numerical uncertainty and unresolved limitations

Total solver-state/discretization uncertainty remains null and effect resolution unresolved. Decimal reconstruction uses the same stored field and measures interpolation arithmetic only. Max measured arithmetic differences: baseline 3.552713678800501e-15 dx, control 0.0 dx. Neither zero baseline replay discrepancy nor zero measured control arithmetic discrepancy establishes a total uncertainty bound. Historical approximately 1.8e-9 dx agreement is not a noise floor or intervention tolerance. Binary support invariance remains UNRESOLVED_NO_JUSTIFIED_BINARY_THRESHOLD. No mobility-sensitivity-resolved verdict is permitted by this freeze.

## 11. Conservation, reconstruction and solver convergence

```json
{
  "baseline": {
    "cg_all_converged": true,
    "cg_iterations_max": 18,
    "cg_iterations_min": 18,
    "cg_relative_residual_max": 5.718904648233452e-07,
    "closed_face_flux_max": 0.0,
    "component_counts": {
      "1": 360
    },
    "contact_gap_min_dx": 2.643332799122099,
    "contact_guard": {
      "triggered": false
    },
    "failed_checks": [],
    "global_min_gap_status_counts": {
      "NON_UNIQUE": 360
    },
    "mass_final": 1.5500927792312642,
    "mass_initial": 1.5500927792312642,
    "operator_recombination_linf": 1.1102230246251565e-16,
    "raw_mass_drift": 0.0,
    "readout_status_counts": {
      "VALID": 360
    },
    "reconstruction_arithmetic_max_difference_dx": 3.552713678800501e-15,
    "source_capture_equivalence": "synthetic public-step check passed before target; baseline all intermediate hashes matched sealed capture; no duplicate target control",
    "worst_balance_residual": 4.470435646983889e-16,
    "zero_limiting_checks_all_pass": true
  },
  "control": {
    "cg_all_converged": true,
    "cg_iterations_max": 0,
    "cg_iterations_min": 0,
    "cg_relative_residual_max": 0.0,
    "closed_face_flux_max": 0.0,
    "component_counts": {
      "1": 360
    },
    "contact_gap_min_dx": 2.6069711167050507,
    "contact_guard": {
      "triggered": false
    },
    "failed_checks": [],
    "global_min_gap_status_counts": {
      "NON_UNIQUE": 360
    },
    "mass_final": 1.5500927792312642,
    "mass_initial": 1.5500927792312642,
    "operator_recombination_linf": 1.1102230246251565e-16,
    "raw_mass_drift": 0.0,
    "readout_status_counts": {
      "VALID": 360
    },
    "reconstruction_arithmetic_max_difference_dx": 0.0,
    "source_capture_equivalence": "synthetic public-step check passed before target; baseline all intermediate hashes matched sealed capture; no duplicate target control",
    "worst_balance_residual": 4.462288560404621e-16,
    "zero_limiting_checks_all_pass": true
  }
}
```
Full per-substep mass balance, face flux, solver residuals and diagnostics are in the arm files. Existing residual/recombination gates remain 1e-12, closed-face flux exactly zero and CG fail-closed. All zero-arm limiting checks pass. Synthetic public/capture equivalence passed before measurement; target baseline exactly matched the sealed capture. No additional public-step shadow target control was executed. Global minimum-gap NON_UNIQUE remains visible for 360 states in each arm and is not used to choose a support.

## 12. Adversarial tests

47 focused tests passed before target measurement and again afterward. They discriminate wrong explicit/implicit mobility, mass/CG failure, deleted capillary/momentum coupling, altered states/geometry/unrelated parameters, missing/multiple/ambiguous roots, correspondence loss, ray identity swaps, enumeration/order permutations, inverted signs, opposite responses, large valid support disagreement, artificial averaging, unsupported uncertainty/invariance, baseline initial/final/intermediate hash failures, source/evidence tampering, invalid promotion and duplicate receipts. Target trajectories were not executed by the tests.

## 13. Evidence verification

The manifest binds every scientific payload and all 18 source/test/protocol hashes. Consumer command: `JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=examples/two_phase .venv/bin/python -m production.l1a3c_zero_mobility_control_audit verify`. It independently recomputes the paired results, validates execution count, decisions, sources, prerequisite seals and committed boundary. Independent stdlib file/source hash verification and mutation rejection are also recorded in verification.json. No prior-stage seal is rewritten.

## 14. Adjacent suites and quality ratchet

```json
{
  "added_diagnostics": 0,
  "adjacent_failures": [
    "test_boundary_compatible_incident_momentum_bound_audit.py::test_preflight_is_clean_on_the_repository_head",
    "test_impact_initialization_compatibility_audit.py::test_l1a2r_merged_base_verified_or_fail_closed",
    "test_interior_speed_admissibility_screen_audit.py::test_evidence_verification_rejects_a_moved_envelope",
    "test_interior_speed_admissibility_screen_audit.py::test_manifest_self_verification_and_tamper_detection",
    "test_interior_speed_admissibility_screen_audit.py::test_report_carries_every_required_section",
    "test_interior_speed_admissibility_screen_audit.py::test_required_evidence_files_are_declared_and_complete",
    "test_interior_speed_admissibility_screen_audit.py::test_reseal_records_the_observed_ci_status_without_touching_physics",
    "test_l1a3b_support_indexed_observable_audit.py::test_full_stage_evidence_verifier_rejects_a_mutated_sealed_historical_vector",
    "test_nearwall_taper_interface_transport_audit.py::test_manifest_mismatch_reference_is_recorded_in_preflight"
  ],
  "adjacent_passed": 435,
  "command": "JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=. ../../.venv/bin/python -m pytest tests/test_l1a3c_zero_mobility_control_audit.py tests/test_l1a3b_support_indexed_observable_audit.py tests/test_l1a2z_sequential_ownbase_attribution_audit.py tests/test_frozen_reconstruction_branch_attribution_audit.py tests/test_nearwall_taper_interface_transport_audit.py tests/test_radial_incident_seed_eta_dt_audit.py tests/test_interior_speed_admissibility_screen_audit.py tests/test_boundary_compatible_incident_momentum_bound_audit.py tests/test_frozen_chns_early_window_transport_attribution_audit.py tests/test_impact_initialization_compatibility_audit.py -q",
  "environment": {
    "codespell": "2.4.3",
    "isort": "6.1.0",
    "jax": "0.10.2",
    "jaxlib": "0.10.2",
    "numpy": "2.4.6",
    "python": "3.11.2",
    "ruff": "0.15.22",
    "scipy": "1.17.1"
  },
  "focused_failures": [],
  "focused_passed": 47,
  "frozen_source_changes_since_measurement": [],
  "head": "bd7a8fc7ddcb43c2e93ad78eac45b60238417690",
  "historical_counts": {
    "codespell": 3,
    "format": 27,
    "isort": 28,
    "ruff": 171
  },
  "lineage_restore": "sandbox reset; published blobs verified bytewise, full history fetched, branch restored; no target receipt existed",
  "passed": true,
  "quality_command": "PATH=.venv/bin:$PATH .venv/bin/python scripts/quality_debt_gate.py --tool all",
  "ratchet": "PASS_BASELINE_ONLY",
  "run_result": "9 failed, 482 passed in 158.38s",
  "source_hashes": {
    "examples/two_phase/evidence/l1a2z/manifest.json": "65d42c5fc11c244975d8cf40a06d8a4fdd26884830677c36c8697eb4e63f94e4",
    "examples/two_phase/evidence/l1a3a/manifest.json": "c5c32bcd9ee56f7ac9188368bf78192a2ff076ef51ea3ac9c3b6cfb5fd84dfb8",
    "examples/two_phase/evidence/l1a3a/phase_a_audit.json": "890cbfc96f48bfc4695fbb55f9c8ac41bf135d4b6b221e6de2b4bf72f367ae80",
    "examples/two_phase/evidence/l1a3b/manifest.json": "29157b647adf6ce1701bee60b5d0fb77322fd77f5b1aeb73c801d909d5d11845",
    "examples/two_phase/l1a2z_prespecification.md": "488568ab96221883bda7233b586f645f1ebc64694e8b6bc876774473ba3449eb",
    "examples/two_phase/l1a3b_prespecification.md": "d280a12919ff04a8b5a0705185d0909bafb56c5b836091e8c8df75bfea1025a7",
    "examples/two_phase/l1a3c_prespecification.md": "5bc54b90edf4bbed4cf735442f1c780930979d3e3989693b19eec173858afee0",
    "examples/two_phase/phasefield.py": "024665742dce67bc970052e9760fa375ccda096f534cae5faaf2b446c11fe27c",
    "examples/two_phase/production/frozen_chns_early_window_transport_attribution_audit.py": "f0fddeabfdd4498123c84f8cd9125470056e589ef6e2241f2448a12eb3d0214e",
    "examples/two_phase/production/frozen_reconstruction_branch_attribution_audit.py": "d6c9e05a42384b227faa6d86b47284984d14e0255d7ba17345fd359ad7f51131",
    "examples/two_phase/production/impact_initialization_compatibility_audit.py": "788791df2b0bb3e737fe68cba6e5cd3e98d100285184e0b727ce02363309a3b0",
    "examples/two_phase/production/l1a2y_contour_branch_readout.py": "563553a398f2eea53bf0cc73d53047ea483c37d6686d2572f92ddf3fad540e97",
    "examples/two_phase/production/l1a2z_sequential_ownbase_attribution_audit.py": "1c1815991e929860d377855a2d4c46d890040da0d0ee1733ff9f675784f5e858",
    "examples/two_phase/production/l1a3b_support_indexed_observable_audit.py": "64ea61b912630cd35b8ed3dd1bfcf51a733b12373325340e4983b9814625a104",
    "examples/two_phase/production/l1a3c_zero_mobility_control_audit.py": "e56c3a2e1bf9a8699bb4096374d7e5003734a37c9f27e5d02398fd55b72c1345",
    "examples/two_phase/tests/test_l1a2z_sequential_ownbase_attribution_audit.py": "0f329a0fd0b62f9de411e5f3de544b53e60378e964887b49878cb99037ec4a70",
    "examples/two_phase/tests/test_l1a3b_support_indexed_observable_audit.py": "932ce4f2d5fc7ab37d05eb71e962833a9e75708786a728e1d955db2c9993b686",
    "examples/two_phase/tests/test_l1a3c_zero_mobility_control_audit.py": "8d68d4fdad3153e247e2326dd53530d9a385f94a2c52723f1cc5dbb5afc52f5c"
  },
  "suites": {
    "test_boundary_compatible_incident_momentum_bound_audit": {
      "failed": [
        "test_boundary_compatible_incident_momentum_bound_audit.py::test_preflight_is_clean_on_the_repository_head"
      ],
      "passed": 44
    },
    "test_frozen_chns_early_window_transport_attribution_audit": {
      "failed": [],
      "passed": 58
    },
    "test_frozen_reconstruction_branch_attribution_audit": {
      "failed": [],
      "passed": 71
    },
    "test_impact_initialization_compatibility_audit": {
      "failed": [
        "test_impact_initialization_compatibility_audit.py::test_l1a2r_merged_base_verified_or_fail_closed"
      ],
      "passed": 48
    },
    "test_interior_speed_admissibility_screen_audit": {
      "failed": [
        "test_interior_speed_admissibility_screen_audit.py::test_required_evidence_files_are_declared_and_complete",
        "test_interior_speed_admissibility_screen_audit.py::test_manifest_self_verification_and_tamper_detection",
        "test_interior_speed_admissibility_screen_audit.py::test_evidence_verification_rejects_a_moved_envelope",
        "test_interior_speed_admissibility_screen_audit.py::test_reseal_records_the_observed_ci_status_without_touching_physics",
        "test_interior_speed_admissibility_screen_audit.py::test_report_carries_every_required_section"
      ],
      "passed": 24
    },
    "test_l1a2z_sequential_ownbase_attribution_audit": {
      "failed": [],
      "passed": 70
    },
    "test_l1a3b_support_indexed_observable_audit": {
      "failed": [
        "test_l1a3b_support_indexed_observable_audit.py::test_full_stage_evidence_verifier_rejects_a_mutated_sealed_historical_vector"
      ],
      "passed": 24
    },
    "test_nearwall_taper_interface_transport_audit": {
      "failed": [
        "test_nearwall_taper_interface_transport_audit.py::test_manifest_mismatch_reference_is_recorded_in_preflight"
      ],
      "passed": 46
    },
    "test_radial_incident_seed_eta_dt_audit": {
      "failed": [],
      "passed": 50
    }
  },
  "uncommitted_preflight_observation": "10 failed, 434 passed in 133.22s; extra radial case failed because stage files were uncommitted; neither masked nor counted as a pass"
}
```
Results are NOT all green: 47/47 focused, 24/25 L1A-3b, 70/70 L1A-2z, 71/71 L1A-2y; frozen adjacent sweep 222 passed/7 failed; legacy lineage 48 passed/1 failed. The immutable L1A-3b consumer-preflight test adds one successor-stage failure, recorded above, not edited or greened. Its content verification separately passes. The uncommitted preflight observation (10 failed/434 passed) is retained and not counted as success; the extra radial test passes on the committed clean boundary. Ratchet PASS_BASELINE_ONLY: 171/27/28/3 historical findings, zero added. No historical failure inventory or quality baseline was changed.

## 15. Final-head hosted CI

NOT_OBSERVED at report generation. Local tests and L1A-3b CI are not CI evidence for L1A-3c. After normal push of this report/evidence commit, exact head, hosted run/check IDs and observed outcomes will be recorded in the publication appendix of PR #24 and the delivered publication record. No automatic merge or repeated scientific reseal to chase a self-referential commit SHA.

## 16. Scientific verdict

```json
{
  "decision_inputs": {
    "admissible": true,
    "executed": true,
    "protocol": true,
    "valid": false
  },
  "physical_status": "PHYSICAL_MECHANISM_NOT_ESTABLISHED",
  "verdict": "CONTROL_OBSERVABLE_INVALID"
}
```
A is demonstrated for the implemented finite discrete zero-mobility limit; B fails the complete paired correspondence protocol despite valid single-arm readouts; C has no valid complete final E vector and no resolved sensitivity verdict; D remains unanswered. This is an early-time model diagnostic, not a contact/impact mechanism experiment. A failed pair is recorded rather than repaired after measurement.

## 17. Next action and stop boundary

NOT_STARTED; separate owner approval is required. No rerun, alternate mobility, altered root/branch gate, preferred support, contact-stage extension, solver patch, refinement, T=8 regeneration, L1B data or training is authorized. Any proposed cross-trajectory correspondence revision requires a separately approved freeze cycle; the current invalid verdict and evidence remain immutable.
