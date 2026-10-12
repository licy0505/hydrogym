"""L1A-2w boundary-compatible incident-momentum bound: identities, provenance and fail-closed gates.

Every test verifies either a prespecified property of the t=0 bound, an exact identity of the
necessary conditions, a provenance rule, or a fail-closed failure mode. No test edits a frozen
production source, a threshold or any historical evidence, and no test runs a trajectory: the stage
is a t=0 study by construction.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from production import boundary_compatible_incident_momentum_bound_audit as stage  # noqa: E402
from production import impact_initialization_compatibility_audit as audit  # noqa: E402
from production import radial_incident_seed_eta_dt_audit as stage_2u  # noqa: E402

TWO_PHASE = Path(__file__).resolve().parents[1]
EVIDENCE = TWO_PHASE / "evidence" / "l1a2w"


@pytest.fixture(scope="module")
def flat_bundle():
    return stage.bundle_for("flat_we100_ct050")


@pytest.fixture(scope="module")
def evidence() -> dict:
    payload = {}
    for name in ("mechanism_decision.json", "feasibility_constructions.json", "incident_momentum_bound.json"):
        payload[name] = json.loads((EVIDENCE / name).read_text())
    payload["manifest.json"] = json.loads((EVIDENCE / "manifest.json").read_text())
    payload["discrete_flux_and_divergence_verification.json"] = json.loads(
        (EVIDENCE / "discrete_flux_and_divergence_verification.json").read_text()
    )
    payload["constraint_geometry_matrix.json"] = json.loads((EVIDENCE / "constraint_geometry_matrix.json").read_text())
    payload["report.md"] = (EVIDENCE / "l1a2w_final_report.md").read_text()
    return payload


def _synthetic_geometry(in_band: bool = True, point_cap: float | None = None) -> dict:
    row = {
        "u_impact": 0.5,
        "contour_support_inside_gate_band": in_band,
        "contour_point_cap_over_u_impact": point_cap
        if point_cap is not None
        else (0.02 if in_band else 4.905526491646905),
        "liquid_volume_admissible": 1.0,
        "core_volume": 1.0,
        "in_band_normal_distance_from_contour_to_band_top": 0.0 if in_band else None,
    }
    return {"cases": {name: dict(row) for name in stage.CASES}}


def _synthetic_bounds(geometry: dict) -> dict:
    return stage.analytical_bounds(
        {
            "cases": {
                name: {
                    **row,
                    "v_band": stage.BAND_SPEED_OVER_U_IMPACT * row["u_impact"],
                }
                for name, row in geometry["cases"].items()
            }
        }
    )


# ---------------------------------------------------------------------------
# prespecification and provenance
# ---------------------------------------------------------------------------
def test_frozen_caps_are_inherited_not_redefined():
    assert stage.BAND_SPEED_OVER_U_IMPACT == float(
        audit.INITIAL_ACCEPTANCE["nearwall_chi_ge_0p01_speed_over_u_impact_max"]
    )
    assert stage.ENVELOPE_OVER_U_IMPACT == 4.905526491646905


def test_radial_family_is_untouched():
    contract = stage.bound_contract()
    assert contract["radial_family_untouched"]["re_tuned"] is False
    assert contract["radial_family_untouched"]["parameters"] == {
        key: dict(stage_2u.RADIAL_CANDIDATES[key]) for key in stage_2u.RADIAL_ORDER
    }
    assert stage.RADIAL_ORDER == tuple(stage_2u.RADIAL_ORDER)


def test_contract_declares_every_prohibition():
    prohibitions = stage.bound_contract()["prohibitions"]
    assert prohibitions["trajectories"].startswith("none")
    assert prohibitions["contact_events"] == "none"
    assert prohibitions["temporal_refinement"] == "none"
    assert prohibitions["spatial_refinement"] == "none"
    assert prohibitions["t8_recomputation"] == "none"
    assert prohibitions["l1b_data_generation"] is False
    assert prohibitions["ml_training"] is False
    assert prohibitions["contract_13"] is False
    assert prohibitions["historical_evidence_edits"] == "none"
    assert prohibitions["production_initializer_promotion"] == "none"


def test_bound_contract_lists_every_derived_bound():
    derived = stage.bound_contract()["bounds_derived"]
    for key in (
        "B1_contour_cell_normal_speed",
        "B1b_contour_point_normal_speed",
        "B2_placeable_downward_momentum",
        "B3_in_band_phase_flux",
        "B4_in_band_transport_time",
        "B5_shape_ratio_criterion",
    ):
        assert key in derived


def test_ramp_ladders_are_disjoint_and_carry_the_arm_labels():
    primary = set(stage.RAMP_LADDER)
    exploratory = set(stage.RAMP_LADDER_EXPLORATORY)
    assert len(primary) == len(stage.RAMP_LADDER)
    assert len(exploratory) == len(stage.RAMP_LADDER_EXPLORATORY)
    arms = {arm for arm, _ladder in stage.RAMP_ARMS}
    assert arms == {"PRESPECIFIED_PRIMARY", "EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS"}


def test_exploratory_arm_has_no_promotion_rights(evidence):
    payload = evidence["feasibility_constructions.json"]
    assert payload["promoted_to_production_initializer"] is False
    assert payload["arms_declared_before_measurement"] == ["PRESPECIFIED_PRIMARY"]
    assert payload["arms_added_after_seeing_the_primary_arm"] == ["EXPLORATORY_POST_HOC_NO_PROMOTION_RIGHTS"]
    for case in payload["cases"].values():
        for record in case["constructions"].values():
            assert "NEVER_PROMOTED" in record["labels"]
            assert "DIAGNOSTIC_CONSTRUCTION" in record["labels"]


# ---------------------------------------------------------------------------
# preflight (fail-closed) and provenance
# ---------------------------------------------------------------------------
def test_preflight_is_clean_on_the_repository_head():
    state = stage.preflight_state()
    assert state["baseline_head_sha"] == stage.BASELINE_HEAD_SHA
    assert state["trajectories_run"] == 0
    assert state["contact_events_measured"] == 0
    assert state["refinement_studies_run"] == 0
    assert state["solver_contract_version"] == 12
    assert all(state["checks"].values())


def test_preflight_fails_closed_on_a_foreign_head(monkeypatch):
    real_git = stage._git
    baseline = stage.BASELINE_HEAD_SHA

    def fake_git(args, *, required=True):
        if args[:2] == ["merge-base", "--is-ancestor"] and args[2] == baseline:
            return None
        return real_git(args, required=required)

    monkeypatch.setattr(stage, "_git", fake_git)
    with pytest.raises(stage.AuditBlocked):
        stage.preflight_state()


def test_preflight_fails_closed_on_a_worktree_dirty_outside_stage_paths(monkeypatch):
    real_git = stage._git

    def fake_git(args, *, required=True):
        if args[:2] == ["status", "--porcelain"]:
            return " M examples/two_phase/phasefield.py"
        return real_git(args, required=required)

    monkeypatch.setattr(stage, "_git", fake_git)
    with pytest.raises(stage.AuditBlocked):
        stage.preflight_state()


def test_porcelain_parsing_survives_a_stripped_first_line():
    """`_git` strips stdout, which eats the leading space of the first status line."""
    assert stage._porcelain_path(" D examples/two_phase/evidence/l1a2w/manifest.json") == (
        "examples/two_phase/evidence/l1a2w/manifest.json"
    )
    assert stage._porcelain_path("D examples/two_phase/evidence/l1a2w/manifest.json") == (
        "examples/two_phase/evidence/l1a2w/manifest.json"
    )
    assert stage._porcelain_path("?? examples/two_phase/evidence/l1a2w/") == ("examples/two_phase/evidence/l1a2w/")
    assert stage._porcelain_path(" M examples/two_phase/phasefield.py") == "examples/two_phase/phasefield.py"
    assert stage._porcelain_path('R  "old name" -> "new name"') == "new name"


def test_preflight_detects_a_foreign_path_on_the_first_status_line(monkeypatch):
    real_git = stage._git

    def fake_git(args, *, required=True):
        if args[:2] == ["status", "--porcelain"]:
            return " M examples/two_phase/production/phasefield.py\n?? keep/this"
        return real_git(args, required=required)

    monkeypatch.setattr(stage, "_git", fake_git)
    with pytest.raises(stage.AuditBlocked):
        stage.preflight_state()


def test_stage_owned_paths_cover_the_stage_deliverables():
    owned = stage.STAGE_OWNED_PATHS
    assert "examples/two_phase/production/boundary_compatible_incident_momentum_bound_audit.py" in owned
    assert "examples/two_phase/tests/test_boundary_compatible_incident_momentum_bound_audit.py" in owned
    assert "examples/two_phase/evidence/l1a2w/" in owned
    assert stage._stage_owned("examples/two_phase/evidence/l1a2w/manifest.json")
    assert not stage._stage_owned("examples/two_phase/production/impact_initialization_compatibility_audit.py")


def test_sealed_evidence_of_the_three_prerequisite_stages_verifies():
    sealed = stage.verify_sealed_evidence()
    for key in ("l1a2t", "l1a2u", "l1a2v"):
        assert sealed[key]["verified"] is True
        assert sealed[key]["bound_files"] >= 13
    assert sealed["envelope_readback"]["matches"] is True
    assert sealed["read_only"] is True


# ---------------------------------------------------------------------------
# necessary conditions: identities and units
# ---------------------------------------------------------------------------
def test_cell_cap_switches_on_band_membership():
    assert stage._cell_cap_over_u_impact(True) == stage.BAND_SPEED_OVER_U_IMPACT
    assert stage._cell_cap_over_u_impact(False) == stage.ENVELOPE_OVER_U_IMPACT


def test_b1_uses_the_band_cap_inside_and_the_envelope_outside():
    inside = _synthetic_bounds(_synthetic_geometry(in_band=True))
    assert inside["per_case"]["flat_we100_ct050"]["B1_source"] == "band cap"
    assert inside["per_case"]["flat_we100_ct050"]["B1_contour_normal_speed_cap_over_u_impact"] == pytest.approx(
        stage.BAND_SPEED_OVER_U_IMPACT
    )
    outside = _synthetic_bounds(_synthetic_geometry(in_band=False))
    assert outside["per_case"]["flat_we100_ct050"]["B1_source"] == "reference envelope"
    assert outside["per_case"]["flat_we100_ct050"]["B1_contour_normal_speed_cap_over_u_impact"] == pytest.approx(
        stage.ENVELOPE_OVER_U_IMPACT
    )


def test_b2_is_the_pointwise_cap_times_the_admissible_liquid_volume():
    bounds = _synthetic_bounds(_synthetic_geometry(in_band=True))
    row = bounds["per_case"]["flat_we100_ct050"]
    assert row["B2_placeable_downward_momentum_max"] == pytest.approx(stage.ENVELOPE_OVER_U_IMPACT * 0.5 * 1.0)
    assert row["B2_core_mandated_downward_momentum"] == pytest.approx(0.5)


def test_b4_reports_a_zero_in_band_leg_and_an_unmeasured_outside_case():
    inside = _synthetic_bounds(_synthetic_geometry(in_band=True))
    row = inside["per_case"]["flat_we100_ct050"]
    assert row["B4_in_band_transport_time_lower_bound"] == 0.0
    assert "exactly at the band top" in row["B4_in_band_transport_time_note"]
    outside = _synthetic_bounds(_synthetic_geometry(in_band=False))
    assert outside["per_case"]["flat_we100_ct050"]["B4_in_band_transport_time_lower_bound"] is None
    assert outside["per_case"]["flat_we100_ct050"]["B4_in_band_transport_time_note"] == "UNMEASURED_NO_IN_BAND_LEG"


def test_face_budget_bound_is_the_band_cap_times_the_face_capacity():
    """The B3 attachment must be the band speed times the measured in-band face area*phi."""
    bounds = stage.attach_face_budget(
        _synthetic_bounds(_synthetic_geometry(in_band=True)),
        {"face_budget": {name: {"in_band_face_area_phi_upwind": 0.5} for name in stage.CASES}},
    )
    row = bounds["per_case"]["flat_we100_ct050"]
    assert row["B3_in_band_face_area_phi_upwind"] == 0.5
    assert row["B3_in_band_flux_bound"] == pytest.approx(stage.BAND_SPEED_OVER_U_IMPACT * 0.5 * 0.5)


# ---------------------------------------------------------------------------
# verification machinery: fail-closed and discriminating
# ---------------------------------------------------------------------------
def _row(**overrides):
    row = {
        "candidate": "SYNTHETIC",
        "gates_pass": True,
        "peak_open_cell_speed_over_u_impact": 1.0,
        "band_max_speed_over_u_impact": 0.0,
        "contour_cell_normal_velocity_over_u_impact": 0.0,
        "contour_point_normal_velocity_over_u_impact": 0.0,
        "placeable_downward_momentum": 0.0,
        "in_band_downward_phase_flux": 0.0,
        "in_band_downward_face_max_speed_over_u_impact": 0.0,
    }
    row.update(overrides)
    return row


def _bounds_for_check(cap: float = 0.02, point_cap: float = 2.0, momentum: float = 1.0, flux: float = 1e-3):
    return {
        "per_case": {
            "flat_we100_ct050": {
                "B1_contour_normal_speed_cap_over_u_impact": cap,
                "B1b_contour_point_cap_over_u_impact": point_cap,
                "B2_placeable_downward_momentum_max": momentum,
                "B3_in_band_flux_bound": flux,
            }
        }
    }


def test_verify_bound_rows_accepts_a_row_at_the_bound():
    report = stage.verify_bound_rows(
        _bounds_for_check(), [("flat_we100_ct050", "tag", _row(contour_cell_normal_velocity_over_u_impact=0.02))]
    )
    assert report["violations"] == []
    assert report["bound_holds_on_every_admissible_state"] is True
    assert report["checks"]["B1_contour_cell_normal_speed[tag]"]["holds"] is True


def test_verify_bound_rows_flags_a_violation_above_the_bound():
    report = stage.verify_bound_rows(
        _bounds_for_check(), [("flat_we100_ct050", "tag", _row(contour_cell_normal_velocity_over_u_impact=0.03))]
    )
    assert report["violations"] == ["B1_contour_cell_normal_speed[tag]"]
    assert report["bound_holds_on_every_admissible_state"] is False


def test_verify_bound_rows_exempts_an_inadmissible_state_without_hiding_it():
    row = _row(gates_pass=False, contour_cell_normal_velocity_over_u_impact=99.0)
    report = stage.verify_bound_rows(_bounds_for_check(), [("flat_we100_ct050", "tag", row)])
    assert report["violations"] == []
    assert report["bound_holds_on_every_admissible_state"] is True
    assert report["inadmissible_rows"] == {"tag": "frozen six gates failed"}
    assert report["checks"]["B1_contour_cell_normal_speed[tag]"]["holds"] is None
    assert report["checks"]["B1_contour_cell_normal_speed[tag]"]["measured"] == 99.0


def test_is_admissible_requires_both_the_gates_and_the_envelope():
    assert stage.is_admissible(_row())[0] is True
    assert stage.is_admissible(_row(peak_open_cell_speed_over_u_impact=stage.ENVELOPE_OVER_U_IMPACT * 2))[0] is False
    assert stage.is_admissible(_row(gates_pass=False))[0] is False


def test_discrete_identity_of_the_frozen_face_rule_matches_the_frozen_operator(flat_bundle):
    """The B3 measurement must reproduce the frozen `phase_advective_fluxes` on the same in-band faces."""
    import phasefield as pf

    state, _details = stage_2u.build_any_state(flat_bundle, stage.C2_KEY)
    u = np.asarray(state.u, dtype=np.float64)
    v = np.asarray(state.v, dtype=np.float64)
    _fx, frozen_fy = pf.phase_advective_fluxes(u, v, state.phi, flat_bundle.solid, flat_bundle.p)
    frozen_fy = np.asarray(frozen_fy, dtype=np.float64)
    chi, _sdf, volume = stage._band_and_volume(flat_bundle)
    _aperture_x, aperture_y = pf.phase_face_apertures(flat_bundle.solid, flat_bundle.p)
    aperture_y = np.asarray(aperture_y, dtype=np.float64)
    open_cells = volume > 0.0
    band_open = (chi >= 0.01) & open_cells
    both = band_open & np.roll(band_open, -1, axis=1)
    v_face = 0.5 * (v + np.roll(v, -1, axis=1))
    mine = np.where(both, aperture_y * v_face * np.roll(np.asarray(state.phi, dtype=np.float64), -1, axis=1), 0.0)
    assert np.allclose(mine[both], frozen_fy[both], rtol=0, atol=1e-14)


# ---------------------------------------------------------------------------
# the measured physics statement
# ---------------------------------------------------------------------------
def test_frozen_family_places_exactly_zero_at_the_contour_cell(flat_bundle):
    for key in stage.SCREEN_CANDIDATES:
        state, details = stage_2u.build_any_state(flat_bundle, key)
        measured = stage._state_metrics(flat_bundle, key, state, details)
        assert measured["contour_cell_inside_band"] is True
        assert measured["contour_cell_normal_velocity_over_u_impact"] == 0.0
        assert measured["band_max_speed_over_u_impact"] == 0.0
        assert measured["in_band_downward_phase_flux"] == 0.0
        assert measured["gates_pass"] is True


def test_c2_ramp_reproduces_the_frozen_c2_field(flat_bundle):
    """The `(onset 4 dx, width 8 dx)` ramp must be the frozen C2 taper, not a new shape."""
    psi_ramp = stage.ramp_streamfunction(flat_bundle, 4.0, 8.0)
    _u, _v, psi_c2, _details = audit.build_c2_velocity(flat_bundle)
    psi_c2 = np.asarray(psi_c2, dtype=np.float64)
    ratios = np.divide(psi_ramp, psi_c2, out=np.zeros_like(psi_ramp), where=np.abs(psi_c2) > 1e-300)
    core = np.abs(psi_c2) > 1e-6 * np.max(np.abs(psi_c2))
    deviation = float(np.max(np.abs(ratios[core] - 1.0)))
    assert deviation < 1e-3, deviation


def test_unit_shape_ratios_are_amplitude_independent(flat_bundle):
    psi = stage.ramp_streamfunction(flat_bundle, 2.0, 4.0)
    base = stage.unit_shape_ratios(flat_bundle, psi)
    scaled = stage.unit_shape_ratios(flat_bundle, 7.5 * psi)
    for key in ("band_core_ratio", "peak_core_ratio"):
        assert scaled[key] == pytest.approx(base[key], rel=1e-12)
    assert scaled["streamfunction_scale_for_core_target"] == pytest.approx(
        base["streamfunction_scale_for_core_target"] / 7.5, rel=1e-12
    )


def test_shape_criterion_rejects_a_ramp_that_cannot_meet_the_band_gate(flat_bundle):
    thin = stage.unit_shape_ratios(flat_bundle, stage.ramp_streamfunction(flat_bundle, 0.0, 2.0))
    assert thin["band_core_ratio"] > stage.BAND_SPEED_OVER_U_IMPACT
    assert thin["band_core_ratio_exceeds_the_gate"] is True


def test_no_shape_is_admissible_while_delivering_flow_to_an_in_band_contour(evidence):
    payload = evidence["feasibility_constructions.json"]
    classification = evidence["constraint_geometry_matrix.json"]["classification"]
    in_band = set(classification["cases_inside_band"])
    assert in_band
    for case_name in in_band:
        case = payload["cases"][case_name]
        for key in case["admissible_constructions"]:
            record = case["constructions"][key]
            assert record["achieved_contour_cell_normal_over_u_impact"] == 0.0
        for key, record in case["constructions"].items():
            if record["status"] == "SHAPE_REJECTED_ANALYTICALLY":
                assert record["shape_invariants"]["band_core_ratio"] > stage.BAND_SPEED_OVER_U_IMPACT


def test_band_gate_ratio_floor_is_above_the_gate_for_every_rejected_shape(evidence):
    payload = evidence["feasibility_constructions.json"]
    ratios = [
        record["shape_invariants"]["band_core_ratio"]
        for case in payload["cases"].values()
        for record in case["constructions"].values()
        if record["status"] == "SHAPE_REJECTED_ANALYTICALLY"
    ]
    assert ratios
    assert min(ratios) > stage.BAND_SPEED_OVER_U_IMPACT


# ---------------------------------------------------------------------------
# evidence, verdict semantics and the prohibition on infeasibility claims
# ---------------------------------------------------------------------------
def test_manifest_binds_the_module_and_claims_no_unpushed_ci(evidence):
    manifest = evidence["manifest.json"]
    assert manifest["stage"] == stage.STAGE
    assert manifest["baseline_head_sha"] == stage.BASELINE_HEAD_SHA
    assert manifest["baseline_pull_request"] == 24
    assert manifest["trajectories_run"] == 0
    assert manifest["contact_events_measured"] == 0
    assert manifest["refinement_studies_run"] == 0
    assert manifest["constructions_promoted"] is False
    assert manifest["no_promotion"] is True
    assert manifest["no_hosted_ci_claim_before_push"] is True
    assert manifest["required_files"] == list(stage.EVIDENCE_FILES)
    assert manifest["module_sha256"] == stage.sha256_file(Path(stage.__file__))


def test_evidence_verification_accepts_the_committed_directory():
    report = stage.verify_evidence_manifest(EVIDENCE)
    assert report["verified"] is True
    assert report["problems"] == []
    assert report["missing_required"] == []


def test_evidence_verification_detects_a_tampered_bound_file(tmp_path):
    target = tmp_path / "l1a2w"
    shutil.copytree(EVIDENCE, target)
    path = target / "mechanism_decision.json"
    payload = json.loads(path.read_text())
    payload["verdict"] = "BOUND_REFUTED"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with pytest.raises(stage.AuditBlocked):
        stage.verify_evidence_manifest(target)


def test_evidence_verification_rejects_a_promoted_construction(tmp_path):
    target = tmp_path / "l1a2w"
    shutil.copytree(EVIDENCE, target)
    path = target / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["constructions_promoted"] = True
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(stage.AuditBlocked):
        stage.verify_evidence_manifest(target)


def test_evidence_verification_rejects_a_missing_required_artifact(tmp_path):
    target = tmp_path / "l1a2w"
    shutil.copytree(EVIDENCE, target)
    (target / "l1a2w_final_report.md").unlink()
    with pytest.raises(stage.AuditBlocked):
        stage.verify_evidence_manifest(target)


def test_reseal_refuses_bad_inputs(tmp_path):
    target = tmp_path / "l1a2w"
    shutil.copytree(EVIDENCE, target)
    with pytest.raises(stage.AuditBlocked):
        stage.reseal_ci_status("looks-fine", "0" * 40, target)
    with pytest.raises(stage.AuditBlocked):
        stage.reseal_ci_status("success", "not-a-sha", target)


def test_reseal_records_only_an_observed_status_on_a_copy(tmp_path):
    target = tmp_path / "l1a2w"
    shutil.copytree(EVIDENCE, target)
    head = "a" * 40
    report = stage.reseal_ci_status("success", head, target)
    assert report["verified"] is True
    quality = json.loads((target / "quality_and_blocker_matrix.json").read_text())
    assert quality["hosted_ci"]["status_for_this_head"] == "success"
    assert quality["hosted_ci"]["observed_head_sha"] == head
    assert "hosted CI observed after push" in (target / "l1a2w_final_report.md").read_text()


def test_decision_never_claims_infeasibility(evidence):
    decision = evidence["mechanism_decision.json"]
    applied = decision["constructibility_rule_applied"]
    assert applied["proven_infeasible"] == []
    assert "NOT_FOUND_NOT_PROVEN_INFEASIBLE" in applied["statement"]
    assert "not a proof of global infeasibility" in decision["headline"]
    assert "PROVEN_INFEASIBLE_BY_BOUND" in stage.CONSTRUCTIBILITY_STATES


def test_decision_separates_instantaneous_from_finite_time(evidence):
    decision = evidence["mechanism_decision.json"]
    split = decision["instantaneous_vs_finite_time"]
    assert "t=0 state" in split["instantaneous"]
    assert "not a bound on the evolved solver" in split["finite_time"]


def test_failed_searches_are_labelled_as_not_found_yet(evidence):
    payload = evidence["feasibility_constructions.json"]
    for case in payload["cases"].values():
        if not case["admissible_constructions"]:
            assert case["constructibility_summary"] == "NOT_FOUND_NOT_PROVEN_INFEASIBLE"


def test_verification_covered_every_frozen_candidate_and_construction(evidence):
    verification = evidence["discrete_flux_and_divergence_verification.json"]
    assert verification["bound_holds_on_every_admissible_state"] is True
    assert verification["violations"] == []
    for case_name in stage.CASES:
        for key in stage.SCREEN_CANDIDATES:
            assert f"B1_contour_cell_normal_speed[{case_name}/{key}]" in verification["checks"]
    assert verification["bound_rows_checked"] >= len(stage.CASES) * len(stage.SCREEN_CANDIDATES)
    assert verification["inadmissible_rows"]


def test_report_records_the_headline_and_the_frozen_adjacent_failures(evidence):
    report = evidence["report.md"]
    assert "BOUND_PROVEN_NECESSARY_ONLY_CONSTRUCTION_BELOW_THE_CAP" in report
    assert "exactly zero normal fluid velocity" in report
    assert "FROZEN_LINEAGE_DRIFT_CONFIRMED" in report
    assert "SUPERSEDED_BY_NEXT_STAGE_BY_CONSTRUCTION" in report
    assert "6 failed, 163 passed" in report
    assert "not a proof of global infeasibility" in report


def test_next_action_claims_no_unmeasured_attainability(evidence):
    action = evidence["mechanism_decision.json"]["next_action"]["action"]
    assert "attains it" not in action
    assert "none of the screened same-language shapes is admissible" in action
    assert "no refinement, no new initializer, no generator" in action


def test_stage_never_constructs_the_frozen_substep_kernel():
    source = Path(stage.__file__).read_text()
    needle = "make_fast" + "_substep"
    assert needle not in source
    assert "run_trajectory" not in source
    assert "run_short_operator_comparison" not in source


def test_point_cap_is_the_weighted_sum_of_the_cell_caps(evidence):
    geometry = evidence["constraint_geometry_matrix.json"]
    for case in geometry["cases"].values():
        weights = case["contour_interpolation_support"]
        expected = sum(row["weight"] * row["cell_cap_over_u_impact"] for row in weights.values())
        assert case["contour_point_cap_over_u_impact"] == pytest.approx(expected, rel=1e-12)
