# L1A-3b — Tie-break-free support-indexed observable contract

**Verdict:** `SUPPORT_INDEXED_OBSERVABLE_FROZEN`

## 1. Verdict and scope

This stage freezes the support-indexed observable only. It does not authorize or run an `M=0` trajectory,
alter solver or physics code, promote an initializer, or make a mobility-sensitivity or physical-mechanism
claim.

## 2. Mathematical observable and signs

`Q(t) = [q95(t), q96(t)]`, with immutable identifiers `RAY_95` and `RAY_96`. Each component is the frozen
linear `phi=0.5` root coordinate `y/dx` on its fixed vertical wall-normal ray. For each ray,
`d_j(t;M)=q_j(0;M)-q_j(t;M)`, positive toward the solid. For a future paired experiment with identical
initial-state hashes, `E_j=d_j(t;M_control)-d_j(t;M_baseline)=q_j(t;M_baseline)-q_j(t;M_control)`. No
component is averaged, preferred, maximized, minimized, or discarded.

## 3. Ray geometry and fixed identity

Grid `[192, 192]`, `dx=0.03125`, `dy=0.03125`. Identity is the cell-centred column, independent of a
selected root or component-label integer.

| Identity | Column | x | Frozen support y | Initial root (`dx`) |
|---|---:|---:|---:|---:|
| `RAY_95` | 95 | 2.984375 | 0.7345619198479572 | 23.5059814351346 |
| `RAY_96` | 96 | 3.015625 | 0.7345619198479572 | 23.5059814351346 |

## 4. Readout, branch correspondence, thresholds, and failure states

Each state enumerates every vertical-ray crossing on the frozen globally reconstructed `phi=0.5` contour,
retains all candidates, validates the frozen edge/cell reconstruction, and maps each candidate to its
current contour component. Exactly one root must lie in the frozen `±1.5 dx` window; its step is at most
`0.25 dx`; no competing root may be closer than `0.5 dx`; and the two selected roots must belong to one
current component. Cross-state correspondence is geometric with the frozen `0.375 dx` component tolerance;
raw labels are never compared across trajectories. Candidate or ray order cannot resolve a tie.
Missing roots, multiple admissible roots, near-degenerate roots, excessive root steps, exact threshold
nodes, frozen enumeration/cell mismatches, split support components, and disjoint or ambiguous geometric
correspondence all produce explicit validity failures. The global `NON_UNIQUE` minimum-gap diagnostic is not
used as a substitute for either fixed-ray root.

## 5. Frozen global contour diagnostic and topology

The L1A-2x global minimum-gap diagnostic remains unchanged: `{'NON_UNIQUE': 360}` over the historical
intermediate states. L1A-2z branch uniqueness is reported separately. Split/merge relations are retained as
topology events; they never silently change ray identity or select a support.

## 6. Symmetric vector and paired-effect reporting

A future valid paired report must include both `d_95`, `d_96`, `E_95`, and `E_96`; the raw signed effect
range; signed difference; absolute support disagreement; sign by ray and discordance; initial hash equality;
root-reconstruction differences; total solver-state uncertainty if defensibly available; and every validity
failure. Invalid pairs produce no partial effect vector. The `paired_effect_vector` implementation also
verifies the algebraic root-coordinate identity.

## 7. Support agreement and numerical uncertainty

Historical maximum support-root spread over 360 stored states is `1.85040605061e-09 dx`; the endpoint spread
is `1.80320114396e-09 dx`. These are observed baseline differences, not a noise floor. A per-root Decimal
check reports floating-point interpolation discrepancy for future stored fields; it is not solver/model
uncertainty. The total solver-state uncertainty bound and binary support-invariance threshold are unset, so
support agreement remains `UNRESOLVED_NO_JUSTIFIED_BINARY_THRESHOLD`.

## 8. Adversarial and falsification validation

Checks: `17/17` passed; failed `[]`.
Frozen cases: `two_co_minimal_supports_same_branch, genuinely_competing_branches_rejected,
candidate_enumeration_permutation_invariant, ray_order_permutation_equivariant,
preferential_single_support_selection_rejected, distinct_valid_support_responses_retained,
opposite_sign_effects_reported, equal_support_effects_reported, missing_root_fails_closed,
multiple_roots_on_one_ray_fail_closed, branch_fragmentation_or_correspondence_loss_detected,
historical_segment_switch_keeps_ray_identity, near_limit_perturbations_are_not_tie_broken,
signed_displacement_convention_checked, swapped_fixed_ray_identity_rejected,
scalar_mean_cannot_hide_disagreement, evidence_mutation_rejected`.

## 9. Historical L1A-2z Phase-C comparison

Verified `360` stored S1/S2/S3 vectors and `120` substep readbacks. q1/q2/q3 statuses: `{'OK': 720}`;
q0/direct statuses: `{'OK': 480}`; branch status: `{'UNIQUE_BRANCH': 360}`; topology relations:
`{'PERSISTENT': 716, 'FRAGMENTED_OR_SPLIT': 2, 'GREW_OR_MERGED': 2}`. Both rays shared one component in all
stored states. Segment changes at the two fixed columns were recorded without changing ray IDs.
- `RAY_95`: q(0) `23.5059814351346 dx`; q(T) `23.4433327532416 dx`; d(T) `+0.0626486818930445 dx`.
- `RAY_96`: q(0) `23.5059814351346 dx`; q(T) `23.4433327514384 dx`; d(T) `+0.0626486836962457 dx`.
Limitation: the sealed L1A-2z ledger lacks intermediate phi arrays and full per-ray candidate lists. Every
stored root/status/segment/component record and uniqueness row was checked, but no solver replay or
independent full-candidate reconstruction is claimed.

## 10. Sealed evidence and source hashes

Freeze boundary: `6849cc0dd77383a728ffdaf8c8c23df10c9a8e15`; post-boundary source/test/doc changes: `[]`; holds `True`.
L1A-2z manifest SHA-256: `65d42c5fc11c244975d8cf40a06d8a4fdd26884830677c36c8697eb4e63f94e4`.
L1A-3a blocker manifest SHA-256: `c5c32bcd9ee56f7ac9188368bf78192a2ff076ef51ea3ac9c3b6cfb5fd84dfb8`.
Source hash map:
```json
{
  "l1a2x_module": "f0fddeabfdd4498123c84f8cd9125470056e589ef6e2241f2448a12eb3d0214e",
  "l1a2y_readout_module": "563553a398f2eea53bf0cc73d53047ea483c37d6686d2572f92ddf3fad540e97",
  "l1a2z_manifest": "65d42c5fc11c244975d8cf40a06d8a4fdd26884830677c36c8697eb4e63f94e4",
  "l1a2z_module": "1c1815991e929860d377855a2d4c46d890040da0d0ee1733ff9f675784f5e858",
  "l1a2z_test": "0f329a0fd0b62f9de411e5f3de544b53e60378e964887b49878cb99037ec4a70",
  "l1a3a_manifest": "c5c32bcd9ee56f7ac9188368bf78192a2ff076ef51ea3ac9c3b6cfb5fd84dfb8",
  "l1a3a_phase_a_record": "890cbfc96f48bfc4695fbb55f9c8ac41bf135d4b6b221e6de2b4bf72f367ae80",
  "prespecification": "d280a12919ff04a8b5a0705185d0909bafb56c5b836091e8c8df75bfea1025a7",
  "stage_module": "64ea61b912630cd35b8ed3dd1bfcf51a733b12373325340e4983b9814625a104",
  "stage_test": "932ce4f2d5fc7ab37d05eb71e962833a9e75708786a728e1d955db2c9993b686"
}
```

## 11. Focused tests, adjacent suites, and quality ratchet

Observed local results:
```json
{
  "commands": {
    "frozen_adjacent_sweep": "JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=. ../../.venv/bin/python -m pytest tests/test_nearwall_taper_interface_transport_audit.py tests/test_radial_incident_seed_eta_dt_audit.py tests/test_interior_speed_admissibility_screen_audit.py tests/test_boundary_compatible_incident_momentum_bound_audit.py tests/test_frozen_chns_early_window_transport_attribution_audit.py -q",
    "l1a2y_suite": "JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=. ../../.venv/bin/python -m pytest tests/test_frozen_reconstruction_branch_attribution_audit.py -q",
    "l1a2z_suite": "JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=. ../../.venv/bin/python -m pytest tests/test_l1a2z_sequential_ownbase_attribution_audit.py -q",
    "legacy_lineage_suite": "JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=. ../../.venv/bin/python -m pytest tests/test_impact_initialization_compatibility_audit.py -q",
    "quality_ratchet": "PATH=.venv/bin:/home/user/uvtools/bin:$PATH .venv/bin/python scripts/quality_debt_gate.py --tool all",
    "stage_suite": "JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=. ../../.venv/bin/python -m pytest tests/test_l1a3b_support_indexed_observable_audit.py -q"
  },
  "control_trajectories_run": 0,
  "declared_frozen_adjacent_failures_from_immutable_l1a2z_record": [
    "test_boundary_compatible_incident_momentum_bound_audit.py::test_preflight_is_clean_on_the_repository_head",
    "test_interior_speed_admissibility_screen_audit.py::test_evidence_verification_rejects_a_moved_envelope",
    "test_interior_speed_admissibility_screen_audit.py::test_manifest_self_verification_and_tamper_detection",
    "test_interior_speed_admissibility_screen_audit.py::test_report_carries_every_required_section",
    "test_interior_speed_admissibility_screen_audit.py::test_required_evidence_files_are_declared_and_complete",
    "test_interior_speed_admissibility_screen_audit.py::test_reseal_records_the_observed_ci_status_without_touching_physics",
    "test_nearwall_taper_interface_transport_audit.py::test_manifest_mismatch_reference_is_recorded_in_preflight",
    "test_radial_incident_seed_eta_dt_audit.py::test_report_contains_every_required_section"
  ],
  "declared_legacy_lineage_failures_from_immutable_l1a2z_record": [
    "test_impact_initialization_compatibility_audit.py::test_l1a2r_merged_base_verified_or_fail_closed"
  ],
  "frozen_adjacent_sweep": {
    "failing": [
      "tests/test_nearwall_taper_interface_transport_audit.py::test_manifest_mismatch_reference_is_recorded_in_preflight",
      "tests/test_interior_speed_admissibility_screen_audit.py::test_required_evidence_files_are_declared_and_complete",
      "tests/test_interior_speed_admissibility_screen_audit.py::test_manifest_self_verification_and_tamper_detection",
      "tests/test_interior_speed_admissibility_screen_audit.py::test_evidence_verification_rejects_a_moved_envelope",
      "tests/test_interior_speed_admissibility_screen_audit.py::test_reseal_records_the_observed_ci_status_without_touching_physics",
      "tests/test_interior_speed_admissibility_screen_audit.py::test_report_carries_every_required_section",
      "tests/test_boundary_compatible_incident_momentum_bound_audit.py::test_preflight_is_clean_on_the_repository_head"
    ],
    "result": "7 failed, 222 passed in 106.39s"
  },
  "holds": true,
  "l1a2y_suite": {
    "failing": [],
    "result": "71 passed in 23.81s"
  },
  "legacy_lineage_suite": {
    "failing": [
      "tests/test_impact_initialization_compatibility_audit.py::test_l1a2r_merged_base_verified_or_fail_closed"
    ],
    "result": "1 failed, 48 passed in 61.16s"
  },
  "observations": {
    "frozen_adjacent_sweep": {
      "failing": [
        "tests/test_nearwall_taper_interface_transport_audit.py::test_manifest_mismatch_reference_is_recorded_in_preflight",
        "tests/test_interior_speed_admissibility_screen_audit.py::test_required_evidence_files_are_declared_and_complete",
        "tests/test_interior_speed_admissibility_screen_audit.py::test_manifest_self_verification_and_tamper_detection",
        "tests/test_interior_speed_admissibility_screen_audit.py::test_evidence_verification_rejects_a_moved_envelope",
        "tests/test_interior_speed_admissibility_screen_audit.py::test_reseal_records_the_observed_ci_status_without_touching_physics",
        "tests/test_interior_speed_admissibility_screen_audit.py::test_report_carries_every_required_section",
        "tests/test_boundary_compatible_incident_momentum_bound_audit.py::test_preflight_is_clean_on_the_repository_head"
      ],
      "result": "7 failed, 222 passed in 106.39s"
    },
    "github": {
      "hosted_ci": {
        "reason": "L1A-3b has not yet been pushed to the existing Draft PR; prior checks refer to the L1A-3a head.",
        "status": "NOT_OBSERVED_FOR_NEW_HEAD"
      },
      "pr_head_sha": "29dd981dd0b162af05ca9bba00aa90f7dc5e8fea",
      "pr_is_draft": true,
      "pr_state": "OPEN"
    },
    "l1a2y_suite": {
      "failing": [],
      "result": "71 passed in 23.81s"
    },
    "legacy_lineage_suite": {
      "failing": [
        "tests/test_impact_initialization_compatibility_audit.py::test_l1a2r_merged_base_verified_or_fail_closed"
      ],
      "result": "1 failed, 48 passed in 61.16s"
    },
    "observed_at": "2026-10-13T20:14:47.236640+08:00",
    "predecessor_suite": {
      "failing": [],
      "result": "70 passed in 7.17s"
    },
    "quality_ratchet": {
      "added_diagnostics": 0,
      "historical_counts": {
        "codespell": 3,
        "format": 27,
        "isort": 28,
        "ruff": 171
      },
      "result": "PASS_BASELINE_ONLY"
    },
    "stage_suite": {
      "failing": [],
      "result": "25 passed in 1.47s"
    },
    "unexpected_regressions": []
  },
  "observed_at": "2026-10-13T20:14:47.236640+08:00",
  "predecessor_suite": {
    "failing": [],
    "result": "70 passed in 7.17s"
  },
  "quality_gate_problems": [],
  "quality_ratchet": {
    "baseline_path": "quality/legacy_two_phase_debt_v1.json",
    "baseline_sha256": "52b41ab68b4d6b86aaa254389eb5398e65ad8c5cbdcd12626c0466248dd3d70b",
    "expected": "PASS_BASELINE_ONLY",
    "expected_historical_counts": {
      "codespell": 3,
      "format": 27,
      "isort": 28,
      "ruff": 171
    },
    "new_diagnostics_allowed": 0,
    "new_python_files": 2,
    "standalone_ratchet_results": {
      "added_diagnostics": 0,
      "historical_counts": {
        "codespell": 3,
        "format": 27,
        "isort": 28,
        "ruff": 171
      },
      "result": "PASS_BASELINE_ONLY"
    }
  },
  "stage": "L1A-3b",
  "stage_suite": {
    "failing": [],
    "result": "25 passed in 1.47s"
  },
  "unexpected_regressions": []
}
```
The legacy-debt ratchet must remain `PASS_BASELINE_ONLY` with zero added diagnostics; declared frozen-adjacent and legacy-lineage failures remain visible and are not counted as passes.

## 12. Branch, pull request, and hosted CI

Publication record:
```json
{
  "branch": "arena/02b6c71b-hydrogym",
  "head_at_report_generation": "6849cc0dd77383a728ffdaf8c8c23df10c9a8e15",
  "hosted_ci": {
    "reason": "L1A-3b has not yet been pushed to the existing Draft PR; prior checks refer to the L1A-3a head.",
    "status": "NOT_OBSERVED_FOR_NEW_HEAD"
  },
  "prespecification_boundary_commit": "6849cc0dd77383a728ffdaf8c8c23df10c9a8e15",
  "pull_request_head_sha": "29dd981dd0b162af05ca9bba00aa90f7dc5e8fea",
  "pull_request_is_draft": true,
  "pull_request_number": 24,
  "pull_request_state": "OPEN"
}
```
The authorized branch is `arena/02b6c71b-hydrogym`; PR #24 remains Draft. Hosted CI is claimed only for a pushed head explicitly named in the publication record.

## 13. Next action and stop boundary

No control trajectory, M=0 measurement, contact event, refinement, or physical-mechanism study was run. Next action remains NOT_STARTED and requires separate owner approval.
