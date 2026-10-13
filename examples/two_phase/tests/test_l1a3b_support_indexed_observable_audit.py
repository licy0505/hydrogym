"""Falsification and evidence tests for the L1A-3b two-ray observable contract."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from production import l1a3b_support_indexed_observable_audit as audit

TWO_PHASE = Path(__file__).resolve().parents[1]


def _historical_inputs():
    ledger = audit._load_json(audit.EVIDENCE_2Z / "sequential_ownbase_ledger.json")
    design = audit._load_json(audit.EVIDENCE_2Z / "support_uniqueness_design.json")
    initial = audit._load_json(audit.EVIDENCE_2Z / "initial_state_identity.json")
    blocker = audit._load_json(audit.EVIDENCE_3A / "phase_a_audit.json")
    return ledger, design, initial, blocker


def test_frozen_ray_geometry_matches_sealed_l1a2z_and_l1a3a_records():
    ledger, design, initial, blocker = _historical_inputs()
    rays, grid, provenance = audit.frozen_ray_identities(ledger, design, initial, blocker)
    assert [(ray.ray_id, ray.column_index, ray.x) for ray in rays] == [
        ("RAY_95", 95, 2.984375),
        ("RAY_96", 96, 3.015625),
    ]
    assert grid.dx == grid.dy == 0.03125
    assert [ray.initial_root_over_dx for ray in rays] == [23.50598143513463] * 2
    assert provenance["same_initial_component"] is True
    assert provenance["geometry"]["holds"] is True
    assert audit.verify_l1a3a_blocker()["control_run"] is False
    assert audit.verify_l1a2z_evidence()["verdict"] == "EARLY_WINDOW_MEASUREMENT_ATTRIBUTION_RESOLVED"


def test_all_available_two_ray_intermediate_records_and_endpoints_are_checked():
    ledger, design, initial, blocker = _historical_inputs()
    history, rays, grid = audit.historical_validation(ledger, design, initial, blocker)
    assert history["stored_record_gate_holds"] is True
    assert history["intermediate_state_count"] == 360
    assert history["substep_count"] == 120
    assert history["readout_status_counts_q1_q2_q3"] == {"OK": 720}
    assert history["q0_remeasure_and_q3_direct_status_counts"] == {"OK": 480}
    assert history["branch_uniqueness_status_counts"] == {"UNIQUE_BRANCH": 360}
    assert history["frozen_l1a2x_global_diagnostic_status_counts_retained_verbatim"] == {"NON_UNIQUE": 360}
    assert history["historical_competition_set_size_counts"] == {"2": 333, "4": 27}
    assert history["same_current_component_for_both_rays_in_all_states"] is True
    assert history["geometric_component_labels_compared_across_states"] is False
    assert history["geometry_validation"]["grid_shape"] == [192, 192]
    assert history["geometry_validation"]["dx"] == history["geometry_validation"]["dy"] == 0.03125
    assert len(history["all_intermediate_vectors"]) == 360
    assert len(history["all_per_substep_readbacks"]) == 120
    assert all(record["units"] == "dx" for record in history["all_per_substep_readbacks"])
    assert history["full_independent_candidate_replay"]["available_from_sealed_l1a2z"] is False
    assert history["problems"] == []
    assert history["endpoints"]["RAY_95"]["toward_solid_displacement_dx"] == pytest.approx(0.06264868189304451)
    assert history["endpoints"]["RAY_96"]["toward_solid_displacement_dx"] == pytest.approx(0.06264868369624566)
    assert rays[0].ray_id == "RAY_95" and rays[1].ray_id == "RAY_96"
    assert grid.Nx == grid.Ny == 192


def test_historical_support_spread_is_reported_but_not_used_as_tolerance():
    ledger, design, initial, blocker = _historical_inputs()
    history, _, _ = audit.historical_validation(ledger, design, initial, blocker)
    agreement = audit.support_agreement_assessment(history)
    assert agreement["original_baseline_final_support_root_spread_dx"] == pytest.approx(1.803201143957267e-09)
    assert agreement["historical_baseline_difference_is_a_noise_floor"] is False
    assert agreement["binary_support_invariance_threshold_dx"] is None
    assert agreement["support_agreement_conclusion"] == "UNRESOLVED_NO_JUSTIFIED_BINARY_THRESHOLD"
    assert agreement["solver_state_uncertainty_bound_dx"] is None


def test_historical_grid_segment_switches_do_not_switch_ray_identity():
    ledger, design, initial, blocker = _historical_inputs()
    history, _, _ = audit.historical_validation(ledger, design, initial, blocker)
    events = history["ray_segment_switches"]
    assert {event["ray_id"] for event in events} == {"RAY_95", "RAY_96"}
    assert all(event["ray_identity_changed"] is False for event in events)
    assert {event["from_segment_index"] for event in events} == {22, 23}
    assert {event["to_segment_index"] for event in events} == {22, 23}


def test_frozen_production_entry_recomputes_global_diagnostic_without_using_it_as_a_tie_break():
    grid = type("Grid", (), {"Nx": 192, "Ny": 192, "dx": 0.03125, "dy": 0.03125, "Lx": 6.0, "Ly": 6.0})()
    rays = [
        audit.RayIdentity("RAY_95", 95, 2.984375, 0.7345619198479572, 23.5),
        audit.RayIdentity("RAY_96", 96, 3.015625, 0.7345619198479572, 23.5),
    ]
    x = (np.arange(grid.Nx, dtype=np.float64) + 0.5)[:, None] * grid.dx
    y = (np.arange(grid.Ny, dtype=np.float64) + 0.5)[None, :] * grid.dy
    distance_squared = (x - 3.0) ** 2 + (y - 1.0) ** 2
    field = 0.5 + 0.25 * np.tanh(20.0 * (0.27**2 - distance_squared))
    bundle = type("Bundle", (), {"p": grid, "solid": type("Solid", (), {"sdf": np.ones_like(field)})()})()
    result = audit.frozen_support_indexed_readout(field, grid, rays, {"RAY_95": 23.5, "RAY_96": 23.5}, bundle=bundle)
    assert result["status"] == "VALID"
    assert result["same_current_component"] is True
    assert result["global_contour"]["original_l1a2x_diagnostic"]["status"] == "NON_UNIQUE"
    assert result["global_contour"]["original_diagnostic_redefined"] is False


def test_frozen_production_entry_requires_fixed_geometry_and_original_global_diagnostic():
    ledger, design, initial, blocker = _historical_inputs()
    rays, grid, _ = audit.frozen_ray_identities(ledger, design, initial, blocker)
    previous = {ray.ray_id: ray.initial_root_over_dx for ray in rays}
    missing_bundle = audit.frozen_support_indexed_readout(
        np.ones((grid.Nx, grid.Ny)), grid, rays, previous, bundle=None
    )
    assert missing_bundle["status"] == "INVALID"
    assert missing_bundle["Q_dx"] == {"RAY_95": None, "RAY_96": None}
    assert missing_bundle["blocking_failures"] == ["ORIGINAL_GLOBAL_CONTOUR_DIAGNOSTIC_BUNDLE_REQUIRED"]
    nonsquare = type("Grid", (), {"Nx": 192, "Ny": 192, "dx": 0.03125, "dy": 0.0625, "Lx": 6.0, "Ly": 12.0})()
    assert audit.validate_ray_geometry(rays, nonsquare)["holds"] is False


def test_frozen_contour_readout_enumerates_roots_and_maps_both_to_same_global_component():
    grid = audit._synthetic_grid()
    rays = audit._synthetic_rays(grid)
    field = audit._circle_phi(grid)
    result = audit.support_indexed_readout(field, grid, rays, {"RAY_A": 10.25, "RAY_B": 10.25})
    assert result["status"] == "VALID"
    assert result["same_current_component"] is True
    assert set(result["Q_dx"]) == {"RAY_A", "RAY_B"}
    assert result["ray_records"]["RAY_A"]["candidate_count_all_intersections"] == 2
    assert result["ray_records"]["RAY_B"]["candidate_count_all_intersections"] == 2
    assert result["global_contour"]["frozen_enumeration"]["matches"] is True
    assert result["global_contour"]["cell_validation"]["matches"] is True
    assert result["global_contour"]["original_diagnostic_redefined"] is False
    assert result["no_primary_support_or_scalar_aggregation"] is True


def test_ray_order_permutation_is_equivariant_and_keeps_named_components():
    grid = audit._synthetic_grid()
    rays = audit._synthetic_rays(grid)
    field = audit._circle_phi(grid)
    previous = {"RAY_A": 10.25, "RAY_B": 10.25}
    forward = audit.support_indexed_readout(field, grid, rays, previous)
    reverse = audit.support_indexed_readout(field, grid, list(reversed(rays)), previous)
    assert forward["status"] == reverse["status"] == "VALID"
    assert forward["Q_dx"] == reverse["Q_dx"]
    assert {ray_id: item["selected_root_over_dx"] for ray_id, item in forward["ray_records"].items()} == {
        ray_id: item["selected_root_over_dx"] for ray_id, item in reverse["ray_records"].items()
    }


def test_candidate_enumeration_permutation_does_not_tie_break():
    candidates = [
        {"root_over_dx": 10.25, "component_label_within_this_state": 4, "segment_index": 10},
        {"root_over_dx": 25.25, "component_label_within_this_state": 9, "segment_index": 25},
    ]
    first = audit.select_unique_admissible_root(candidates, 10.25)
    reversed_order = audit.select_unique_admissible_root(list(reversed(candidates)), 10.25)
    assert first["status"] == reversed_order["status"] == "OK"
    assert first["selected"] == reversed_order["selected"]


def test_genuinely_competing_nested_branches_fail_as_multiple_roots_on_each_ray():
    grid = audit._synthetic_grid()
    rays = audit._synthetic_rays(grid)
    result = audit.support_indexed_readout(audit._annulus_phi(grid), grid, rays, {"RAY_A": 10.75, "RAY_B": 10.75})
    assert result["status"] == "INVALID"
    assert "RAY_A:MULTIPLE_ADMISSIBLE_ROOTS" in result["blocking_failures"]
    assert "RAY_B:MULTIPLE_ADMISSIBLE_ROOTS" in result["blocking_failures"]
    assert result["Q_dx"] == {"RAY_A": None, "RAY_B": None}


def test_missing_roots_fail_closed_without_a_fallback_support():
    grid = audit._synthetic_grid()
    rays = audit._synthetic_rays(grid)
    result = audit.support_indexed_readout(np.ones((grid.Nx, grid.Ny)), grid, rays, {"RAY_A": 10.25, "RAY_B": 10.25})
    assert result["status"] == "INVALID"
    assert "NO_GLOBAL_CONTOUR" in result["blocking_failures"]
    assert all(value is None for value in result["Q_dx"].values())
    with pytest.raises(audit.AuditBlocked, match="exactly"):
        audit.validate_complete_vector({"RAY_95": 1.0})


def test_exact_threshold_nodes_and_bad_ray_ordering_are_explicit_failures():
    grid = audit._synthetic_grid()
    rays = audit._synthetic_rays(grid)
    field = audit._circle_phi(grid)
    field[10, 9] = 0.5
    result = audit.support_indexed_readout(field, grid, rays, {"RAY_A": 10.25, "RAY_B": 10.25})
    assert result["status"] == "INVALID"
    assert "EXACT_THRESHOLD_NODE_DEGENERACY" in result["blocking_failures"]
    assert audit.validate_ray_geometry(list(reversed(rays)), grid)["holds"] is True


def test_close_roots_and_step_limit_fail_without_favorable_selection():
    tied = [
        {"root_over_dx": 10.25, "component_label_within_this_state": 2, "segment_index": 10},
        {"root_over_dx": 10.49, "component_label_within_this_state": 7, "segment_index": 11},
    ]
    close = audit.select_unique_admissible_root(tied, 10.25)
    assert close["status"] == "MULTIPLE_ADMISSIBLE_ROOTS"
    boundary = [{"root_over_dx": 10.25, "component_label_within_this_state": 2, "segment_index": 10}]
    assert audit.select_unique_admissible_root(boundary, 10.5)["status"] == "OK"
    just_over = audit.select_unique_admissible_root(boundary, 10.5000000001)
    assert just_over["status"] == "BRANCH_STEP_TOO_LARGE"
    near = audit.select_unique_admissible_root(
        [
            {"root_over_dx": 1.49, "component_label_within_this_state": 2, "segment_index": 1},
            {"root_over_dx": 1.989, "component_label_within_this_state": 3, "segment_index": 2},
        ],
        0.0,
    )
    assert near["status"] == "ROOT_SEPARATION_BELOW_FROZEN_AMBIGUITY_LIMIT"


def test_support_branch_fragmentation_across_the_two_rays_is_invalid():
    grid = audit._synthetic_grid()
    rays = [
        audit.RayIdentity("RAY_A", 10, 1.05, 0.0, 17.15),
        audit.RayIdentity("RAY_B", 15, 1.55, 0.0, 17.15),
    ]
    x = (np.arange(grid.Nx, dtype=np.float64) + 0.5)[:, None] * grid.dx
    y = (np.arange(grid.Ny, dtype=np.float64) + 0.5)[None, :] * grid.dy
    distance_a = (x - 1.05) ** 2 + (y - 1.825) ** 2
    distance_b = (x - 1.55) ** 2 + (y - 1.825) ** 2
    a = 0.5 + 0.25 * np.tanh(100.0 * (0.11**2 - distance_a))
    b = 0.5 + 0.25 * np.tanh(100.0 * (0.11**2 - distance_b))
    field = np.maximum(a, b)
    result = audit.support_indexed_readout(field, grid, rays, {"RAY_A": 17.15, "RAY_B": 17.15})
    assert result["status"] == "INVALID"
    assert "THE_TWO_FIXED_RAYS_NO_LONGER_BELONG_TO_ONE_CURRENT_COMPONENT" in result["blocking_failures"]


def test_geometric_branch_correspondence_does_not_compare_component_integers():
    grid = audit._synthetic_grid()
    before = np.array([[1.0, 1.0], [1.1, 1.0], [1.2, 1.0]])
    after = before + np.array([0.01, 0.0])
    related = audit.geometric_branch_correspondence(before, after, grid)
    lost = audit.geometric_branch_correspondence(before, before + 1.0, grid)
    assert related["holds"] is True
    assert related["status"] == "GEOMETRICALLY_CORRESPONDENT"
    assert related["labels_compared"] is False
    assert lost["holds"] is False
    assert lost["status"] == "CORRESPONDENCE_LOST"


def test_frozen_ray_ids_reject_a_column_swap_even_if_the_x_values_are_swapped_too():
    grid = type("Grid", (), {"Nx": 192, "Ny": 192, "dx": 0.1, "dy": 0.1, "Lx": 19.2, "Ly": 19.2})()
    rays = [
        audit.RayIdentity("RAY_95", 96, 9.65, 1.0, 10.0),
        audit.RayIdentity("RAY_96", 95, 9.55, 1.0, 10.0),
    ]
    assert audit.validate_ray_geometry(rays, grid, require_frozen_ids=True)["holds"] is False


def test_signed_displacements_and_paired_effects_keep_both_signs_and_raw_range():
    initial = {"RAY_95": 10.0, "RAY_96": 10.0}
    baseline = {"RAY_95": 9.8, "RAY_96": 9.7}
    control = {"RAY_95": 9.6, "RAY_96": 9.8}
    assert audit.signed_displacement_vector(initial, baseline) == pytest.approx({"RAY_95": 0.2, "RAY_96": 0.3})
    effect = audit.paired_effect_vector(
        initial,
        baseline,
        control,
        baseline_initial_hashes={"phi": "common"},
        control_initial_hashes={"phi": "common"},
    )
    assert effect["E_dx"] == pytest.approx({"RAY_95": 0.2, "RAY_96": -0.1})
    assert effect["raw_signed_effect_range_dx"] == pytest.approx([-0.1, 0.2])
    assert effect["absolute_support_to_support_effect_disagreement_dx"] == pytest.approx(0.3)
    assert effect["effects_have_opposite_signs"] is True
    assert effect["coordinate_identity_baseline_minus_control_dx"] == pytest.approx(effect["E_dx"])
    assert effect["scalar_mean_or_preferred_support_reported"] is False


def test_equal_support_effects_and_wrong_initial_hashes_are_handled_symmetrically():
    equal = audit.paired_effect_vector(
        {"RAY_95": 10.0, "RAY_96": 10.0},
        {"RAY_95": 9.8, "RAY_96": 9.8},
        {"RAY_95": 9.6, "RAY_96": 9.6},
        baseline_initial_hashes={"phi": "same"},
        control_initial_hashes={"phi": "same"},
    )
    assert equal["E_dx"]["RAY_95"] == pytest.approx(equal["E_dx"]["RAY_96"])
    assert equal["absolute_support_to_support_effect_disagreement_dx"] == 0.0
    with pytest.raises(audit.AuditBlocked, match="initial-state hashes differ"):
        audit.paired_effect_vector(
            {"RAY_95": 10.0, "RAY_96": 10.0},
            {"RAY_95": 9.8, "RAY_96": 9.8},
            {"RAY_95": 9.6, "RAY_96": 9.6},
            baseline_initial_hashes={"phi": "base"},
            control_initial_hashes={"phi": "control"},
        )


def test_paired_effect_audit_preserves_invalidity_failures_without_partial_effects():
    invalid = audit.paired_effect_audit(
        {"RAY_95": 10.0, "RAY_96": 10.0},
        {"status": "INVALID", "blocking_failures": ["RAY_96:MULTIPLE_ADMISSIBLE_ROOTS"]},
        {"status": "VALID", "Q_dx": {"RAY_95": 9.7, "RAY_96": 9.8}},
        baseline_initial_hashes={"phi": "same"},
        control_initial_hashes={"phi": "same"},
    )
    assert invalid["status"] == "INVALID"
    assert invalid["E_dx"] is None
    assert invalid["raw_signed_effect_range_dx"] is None
    assert invalid["validity_failures"] == ["baseline:RAY_96:MULTIPLE_ADMISSIBLE_ROOTS"]
    assert invalid["no_partial_support_effect_reported"] is True


def test_response_report_never_emits_a_mean_that_could_hide_support_disagreement():
    report = audit.response_report({"RAY_95": 10.0, "RAY_96": 10.0}, {"RAY_95": 9.9, "RAY_96": 9.2})
    assert report["displacement_toward_solid_dx"] == pytest.approx({"RAY_95": 0.1, "RAY_96": 0.8})
    assert report["absolute_support_to_support_disagreement_dx"] == pytest.approx(0.7)
    assert "mean" not in report
    assert "primary" not in report
    assert report["scalar_aggregation_performed"] is False


def test_interpolation_uncertainty_is_only_the_decimal_reconstruction_difference():
    grid = audit._synthetic_grid()
    ray = audit._synthetic_rays(grid)[0]
    field = audit._circle_phi(grid)
    roots = audit.readout.frozen_column_crossings(field, ray.column_index, grid)
    uncertainty = audit.root_reconstruction_uncertainty(
        field, ray, roots[0]["segment_index"], roots[0]["y_over_dx"], grid
    )
    assert uncertainty["observed_float_interpolation_difference_dx"] >= 0.0
    assert uncertainty["solver_or_model_uncertainty_bounded"] is False


def test_evidence_hash_verifier_rejects_payload_mutation(tmp_path):
    payload = tmp_path / "readout.json"
    payload.write_text(json.dumps({"Q_dx": {"RAY_95": 1.0, "RAY_96": 2.0}}))
    manifest = {"files": {"readout.json": audit.sha256_file(payload)}}
    assert audit.verify_bound_files(tmp_path, manifest) == []
    payload.write_text(json.dumps({"Q_dx": {"RAY_95": 1.0, "RAY_96": 200.0}}))
    assert audit.verify_bound_files(tmp_path, manifest) == ["sha mismatch readout.json"]


def test_full_stage_evidence_verifier_rejects_a_mutated_sealed_historical_vector(tmp_path):
    source = audit.EVIDENCE_OWN
    if not (source / "manifest.json").is_file():
        pytest.skip("the stage evidence is assembled after the frozen implementation tests")
    shutil.copytree(source, tmp_path / "l1a3b")
    copy = tmp_path / "l1a3b"
    verified = audit.verify_evidence_manifest(copy)
    report = json.loads((copy / "l1a3b_final_report.json").read_text())
    assert verified["verified"] is True
    assert report["required_report_items_count"] == 13
    assert len(report["report_items"]) == 13
    history_path = copy / "historical_validation.json"
    history = json.loads(history_path.read_text())
    history["all_intermediate_vectors"][0]["Q_dx"]["RAY_95"] += 1.0
    history_path.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")
    with pytest.raises(audit.AuditBlocked, match="sha mismatch historical_validation.json"):
        audit.verify_evidence_manifest(copy)


def test_adversarial_check_inventory_is_complete_and_every_probe_holds():
    result = audit.adversarial_and_falsification()
    assert result["checks_run"] == 17
    assert result["passed_checks"] == 17
    assert result["failed_checks"] == []
    assert set(result["checks"]) == set(audit.ADVERSARIAL_CHECK_NAMES)
    assert all(check["holds"] for check in result["checks"].values())


def test_quality_gate_requires_zero_stage_regressions_and_only_declared_frozen_failures():
    frozen_quality = audit._load_json(audit.EVIDENCE_2Z / "quality_and_blocker_matrix.json")
    observations = {
        "observed_at": "2026-10-13T12:00:00+08:00",
        "stage_suite": {"result": "24 passed in 1.2s", "failing": []},
        "predecessor_suite": {"result": "70 passed in 6.6s", "failing": []},
        "l1a2y_suite": {"result": "71 passed in 23.4s", "failing": []},
        "frozen_adjacent_sweep": {
            "result": "8 failed, 221 passed in 103s",
            "failing": [f"tests/{name}" for name in frozen_quality["frozen_adjacent_suite"]["declared"]],
        },
        "legacy_lineage_suite": {
            "result": "1 failed, 48 passed in 60s",
            "failing": [f"tests/{name}" for name in frozen_quality["legacy_lineage_suite"]["declared"]],
        },
        "quality_ratchet": {
            "result": "PASS_BASELINE_ONLY",
            "added_diagnostics": 0,
            "historical_counts": {"ruff": 171, "format": 27, "isort": 28, "codespell": 3},
        },
        "unexpected_regressions": [],
    }
    result = audit.quality_and_blockers(observations)
    assert result["holds"] is True
    assert result["l1a2y_suite"]["result"].startswith("71 passed")
    assert result["quality_ratchet"]["baseline_sha256"] == audit.sha256_file(
        audit.REPO_ROOT / "quality" / "legacy_two_phase_debt_v1.json"
    )

    changed = json.loads(json.dumps(observations))
    changed["frozen_adjacent_sweep"]["failing"].append("test_unapproved_regression")
    changed["frozen_adjacent_sweep"]["result"] = "9 failed, 221 passed in 103s"
    rejected = audit.quality_and_blockers(changed)
    assert rejected["holds"] is False
    assert any("outside the sealed declared set" in issue for issue in rejected["quality_gate_problems"])
