"""L1A-2v interior-speed admissibility screen: identities, provenance and fail-closed gates.

Every test verifies either a prespecified property of the t=0 screen, an exact reproduction of
the sealed L1A-2u baseline, a provenance rule, or a fail-closed failure mode. No test edits a
frozen production source, a threshold, or any historical evidence, and no test runs a
trajectory: the stage is a t=0 screen by construction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from production import impact_initialization_compatibility_audit as audit  # noqa: E402
from production import interior_speed_admissibility_screen_audit as stage  # noqa: E402
from production import radial_incident_seed_eta_dt_audit as stage_2u  # noqa: E402

TWO_PHASE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def metrics():
    return stage.per_candidate_metrics()


@pytest.fixture(scope="module")
def bands():
    return stage.band_classification()


@pytest.fixture(scope="module")
def shapes(metrics):
    return stage.shape_invariance(metrics)


@pytest.fixture(scope="module")
def decision(metrics, bands, shapes):
    return stage.screen_decision(metrics, bands, shapes)


# ---------------------------------------------------------------------------
# prespecification and provenance
# ---------------------------------------------------------------------------
def test_envelope_equals_the_sealed_l1a2u_measurement():
    sealed = json.loads((TWO_PHASE / "evidence" / "l1a2u" / "mechanism_decision.json").read_text())
    recorded = float(sealed["interior_speed_screen"]["ceiling_over_u_impact"])
    assert recorded == stage.ENVELOPE_MAX_OPEN_CELL_SPEED_OVER_U_IMPACT
    observed = stage.read_sealed_envelope()
    assert observed["matches"] is True
    assert observed["prespecified_value"] == stage.ENVELOPE_MAX_OPEN_CELL_SPEED_OVER_U_IMPACT
    assert len(observed["measured_from"]) == len(stage.CASES)


def test_envelope_cannot_silently_move(monkeypatch):
    monkeypatch.setattr(stage, "ENVELOPE_MAX_OPEN_CELL_SPEED_OVER_U_IMPACT", 6.0)
    with pytest.raises(stage.AuditBlocked):
        stage.read_sealed_envelope()


def test_six_frozen_gates_are_inherited_verbatim():
    contract = stage.screen_contract()
    for name, limit in contract["frozen_six_gates"].items():
        assert audit.INITIAL_ACCEPTANCE[name] == limit
    assert contract["core_normalisation_target"]["target_over_u_impact"] == -1.0


def test_family_parameters_are_the_frozen_ones():
    contract = stage.screen_contract()
    assert tuple(stage.SCREEN_CANDIDATES) == (stage.C2_KEY, *stage_2u.RADIAL_ORDER)
    for key in stage.RADIAL_ORDER:
        assert stage_2u.RADIAL_CANDIDATES[key]["delta_dx"] == stage_2u.RADIAL_CANDIDATES[key]["delta_dx"]
    assert [stage_2u.RADIAL_CANDIDATES[k]["delta_dx"] for k in stage.RADIAL_ORDER] == [1.5, 2.0, 2.0]
    assert [stage_2u.RADIAL_CANDIDATES[k]["w_dx"] for k in stage.RADIAL_ORDER] == [2.0, 2.0, 1.0]
    assert contract["global_speed_envelope"]["modifiable_after_results"] is False


def test_core_target_is_minus_u_impact_in_both_units(metrics):
    for case_name in stage.CASES:
        for key in stage.SCREEN_CANDIDATES:
            row = metrics["cases"][case_name]["candidates"][key]
            assert row["u_impact"] == 0.5
            assert abs(row["core_mean_normal_speed_over_u_impact"] + 1.0) <= 1e-6
            assert abs(row["core_mean_normal_speed_raw"] + row["u_impact"]) <= 1e-6


def test_screen_reproduces_the_sealed_l1a2u_matrix_exactly(metrics):
    record = stage.reproduction_check_against_sealed_l1a2u(metrics)
    assert record["verdict"] == "BASELINE_REPLAY_OK"
    assert record["compared_rows"] == len(stage.CASES) * len(stage.SCREEN_CANDIDATES)
    assert record["worst_peak_abs_delta"] == 0.0
    assert record["worst_core_abs_delta"] == 0.0
    assert all(row["six_gate_status_match"] for row in record["rows"].values())


def test_reproduction_failure_is_fail_closed(monkeypatch, metrics):
    tampered = json.loads(json.dumps(metrics))
    tampered["cases"][stage.CASES[0]]["candidates"]["R1_MEDIUM"]["peak"]["peak_over_u_impact"] *= 1.5
    record = stage.reproduction_check_against_sealed_l1a2u(tampered)
    assert record["verdict"] == "BASELINE_REPLAY_MISMATCH"


# ---------------------------------------------------------------------------
# screen outcome
# ---------------------------------------------------------------------------
def test_every_radial_member_fails_the_envelope_in_every_case(metrics, decision):
    for key in stage.RADIAL_ORDER:
        for case_name in stage.CASES:
            row = metrics["cases"][case_name]["candidates"][key]
            assert row["six_frozen_initial_constraints"] is True
            assert row["global_speed_envelope_pass"] is False
            assert row["status"] == "NOT_ADMISSIBLE"
            assert row["peak"]["peak_over_u_impact"] > stage.ENVELOPE_MAX_OPEN_CELL_SPEED_OVER_U_IMPACT
        assert decision["radial_family_rows"][key]["primary_failure"] == "global_speed_envelope"


def test_reference_baseline_is_admissible_in_every_case(metrics):
    for case_name in stage.CASES:
        row = metrics["cases"][case_name]["candidates"][stage.C2_KEY]
        assert row["six_frozen_initial_constraints"] is True
        assert row["global_speed_envelope_pass"] is True
        assert row["status"] == "ADMISSIBLE"


def test_peak_speed_sits_in_the_open_interior_not_at_the_wall(metrics):
    for case_name in stage.CASES:
        for key in stage.SCREEN_CANDIDATES:
            peak = metrics["cases"][case_name]["candidates"][key]["peak"]
            assert peak["peak_inside_gate_band"] is False
            assert peak["peak_inside_solid"] is False
            assert peak["peak_sdf_over_dx"] > 3.0


def test_local_cfl_separates_the_frozen_baseline_from_the_rejected_seeds(metrics):
    for case_name in stage.CASES:
        candidates = metrics["cases"][case_name]["candidates"]
        assert candidates[stage.C2_KEY]["peak"]["local_cfl_at_peak_sum"] <= 1.0
        for key in stage.RADIAL_ORDER:
            assert candidates[key]["peak"]["local_cfl_at_peak_sum"] > 1.0


def test_decision_stops_without_re_tuning(decision):
    assert decision["verdict"] == "NO_ADMISSIBLE_RADIAL_SEED_UNDER_PRESPECIFIED_FAMILY"
    assert decision["admissible_radial_members"] == []
    assert decision["family_stopped"] is True
    assert decision["re_tuning_performed"] is False
    assert decision["trajectories_run"] == 0
    assert decision["contact_events_measured"] == 0
    assert decision["refinement_studies_run"] == 0
    for row in decision["radial_family_rows"].values():
        assert row["failure_reason"] and "envelope" in row["failure_reason"]
        assert row["peak_over_envelope_ratio"] > 1.0


def test_next_action_is_single_and_not_started(decision):
    action = decision["next_action"]
    assert action["must_not_be_started_in_this_stage"] is True
    assert isinstance(action["bounded_scope"], str) and action["bounded_scope"]
    assert "no trajectory" in action["bounded_scope"]


# ---------------------------------------------------------------------------
# shape invariance
# ---------------------------------------------------------------------------
def test_shape_limit_is_envelope_over_the_core_target(shapes):
    assert shapes["shape_ratio_limit"] == stage.ENVELOPE_MAX_OPEN_CELL_SPEED_OVER_U_IMPACT
    assert shapes["rows"][stage.C2_KEY]["shape_passes"] is True
    for key in stage.RADIAL_ORDER:
        assert shapes["rows"][key]["shape_passes"] is False
        assert shapes["rows"][key]["shape_ratio"] > shapes["shape_ratio_limit"]


def test_rescaling_to_the_envelope_destroys_the_incident_speed(shapes):
    for key in stage.RADIAL_ORDER:
        row = shapes["rows"][key]
        assert 0.0 < row["rescaled_core_target_if_envelope_satisfied"] < 0.2
    assert shapes["rows"][stage.C2_KEY]["rescaled_core_target_if_envelope_satisfied"] == pytest.approx(1.0, rel=0.2)


# ---------------------------------------------------------------------------
# near-wall band classification and the refuted L1A-2u claim
# ---------------------------------------------------------------------------
def test_band_classification_splits_three_cases_from_complex_heldout(bands):
    assert bands["cases_with_support_inside_gate_band"] == [
        "flat_we100_ct050",
        "flat_we200_ct000",
        "pillar_training",
    ]
    assert bands["cases_with_support_outside_gate_band"] == ["complex_heldout"]
    for name in ("flat_we100_ct050", "flat_we200_ct000"):
        assert bands["per_case"][name]["phi05_support_sdf_over_dx"] == pytest.approx(2.50, abs=1e-6)
    assert bands["per_case"]["pillar_training"]["phi05_support_sdf_over_dx"] == pytest.approx(2.70, abs=1e-6)
    assert bands["per_case"]["complex_heldout"]["phi05_support_sdf_over_dx"] == pytest.approx(5.6819, abs=1e-3)


def test_l1a2u_universal_claim_is_refuted_from_the_sealed_text(bands):
    assert bands["l1a2u_statement_is_universal"] is True
    verbatim = bands["l1a2u_sealed_statement_verbatim"]
    assert verbatim.startswith("every case places the frozen phi=0.5 contour support cell")
    assert bands["l1a2u_universal_claim_refuted"] is True
    assert bands["l1a2u_statement_holds_for"] == bands["cases_with_support_inside_gate_band"]
    assert bands["l1a2u_statement_refuted_for"] == ["complex_heldout"]
    assert "does not hold" in bands["correction"]


def test_sealed_l1a2u_evidence_is_unmodified_by_this_stage():
    manifest = json.loads((TWO_PHASE / "evidence" / "l1a2u" / "manifest.json").read_text())
    for name, want in manifest["evidence_files"].items():
        assert stage.sha256_file(TWO_PHASE / "evidence" / "l1a2u" / name) == want


# ---------------------------------------------------------------------------
# failure injection and adversarial self-checks
# ---------------------------------------------------------------------------
def test_adversarial_self_checks_hold():
    record = stage.failure_and_adversarial_tests(stage.screen_contract())
    assert all(value for key, value in record["adversarial"].items() if isinstance(value, bool))
    probes = record["injected_failures"]
    assert probes["unknown_candidate_is_rejected"]["raised"] is True
    assert probes["relaxed_envelope_must_be_refused"]["raised"] is True
    assert probes["envelope_comparison_is_fail_closed_at_the_boundary"]["raised"] is False
    assert probes["envelope_must_equal_the_sealed_l1a2u_value"]["raised"] is False


def test_envelope_comparison_rejects_a_hairline_excess():
    limit = stage.ENVELOPE_MAX_OPEN_CELL_SPEED_OVER_U_IMPACT
    assert limit * (1.0 - 1e-9) <= limit
    assert not (limit * (1.0 + 1e-12) <= limit)
    stage._assert_envelope_comparison_is_fail_closed()


def test_preflight_fails_closed_on_a_foreign_head(monkeypatch):
    real_git = stage._git

    def fake_git(args, *, required=True):
        if args[:2] == ["rev-parse", "HEAD"]:
            return "0" * 40
        return real_git(args, required=required)

    monkeypatch.setattr(stage, "_git", fake_git)
    with pytest.raises(stage.AuditBlocked):
        stage.preflight_state()


def test_preflight_fails_closed_on_a_foreign_commit_above_the_baseline(monkeypatch):
    real_git = stage._git

    def fake_git(args, *, required=True):
        if args[:2] == ["log", "--oneline"] and args[2].startswith(stage.BASELINE_HEAD_SHA):
            return "deadbeef feat: touches a frozen solver"
        if args[:2] == ["show", "--pretty=format:"]:
            return "examples/two_phase/phasefield.py\n"
        return real_git(args, required=required)

    monkeypatch.setattr(stage, "_git", fake_git)
    with pytest.raises(stage.AuditBlocked):
        stage.preflight_state()


def test_no_trajectory_kernel_is_referenced_by_this_module():
    body = stage._source_without_docstring()
    assert stage._kernel_needle() not in body
    assert "run_factorial_trajectory" not in body


# ---------------------------------------------------------------------------
# evidence assembly and verification
# ---------------------------------------------------------------------------
def test_required_evidence_files_are_declared_and_complete(tmp_path):
    result = stage.run(out=tmp_path / "evidence", stages="all")
    out_dir = tmp_path / "evidence"
    for name in stage.EVIDENCE_FILES:
        assert (out_dir / name).is_file(), name
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["trajectories_run"] == 0
    assert manifest["contact_events_measured"] == 0
    assert manifest["refinement_studies_run"] == 0
    assert manifest["no_hosted_ci_claim_before_push"] is True
    assert manifest["no_promotion"] is True
    assert manifest["baseline_head_sha"] == stage.BASELINE_HEAD_SHA
    assert manifest["envelope_max_open_cell_speed_over_u_impact"] == (stage.ENVELOPE_MAX_OPEN_CELL_SPEED_OVER_U_IMPACT)
    assert result["admissibility_decision.json"]["verdict"] == ("NO_ADMISSIBLE_RADIAL_SEED_UNDER_PRESPECIFIED_FAMILY")


def test_manifest_self_verification_and_tamper_detection(tmp_path):
    out_dir = tmp_path / "evidence"
    stage.run(out=out_dir, stages="all")
    assert stage.verify_evidence_manifest(out_dir)["verified"] is True
    report = out_dir / "l1a2v_final_report.md"
    report.write_text(report.read_text() + "\ntampered\n")
    with pytest.raises(stage.AuditBlocked):
        stage.verify_evidence_manifest(out_dir)


def test_evidence_verification_rejects_a_moved_envelope(tmp_path):
    out_dir = tmp_path / "evidence"
    stage.run(out=out_dir, stages="all")
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["envelope_max_open_cell_speed_over_u_impact"] = 6.0
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(stage.AuditBlocked):
        stage.verify_evidence_manifest(out_dir)


def test_reseal_records_the_observed_ci_status_without_touching_physics(tmp_path):
    out_dir = tmp_path / "evidence"
    stage.run(out=out_dir, stages="all")
    before = {
        name: stage.sha256_file(out_dir / name)
        for name in stage.EVIDENCE_FILES
        if name not in ("quality_and_blocker_matrix.json", "l1a2v_final_report.md", "manifest.json")
    }
    verdict_before = json.loads((out_dir / "admissibility_decision.json").read_text())["verdict"]
    report = stage.reseal_ci_status("success", "1" * 40, out_dir)
    assert report["verified"] is True
    quality = json.loads((out_dir / "quality_and_blocker_matrix.json").read_text())
    assert quality["hosted_ci"]["status_for_this_head"] == "success"
    assert quality["hosted_ci"]["observed_head_sha"] == "1" * 40
    assert quality["hosted_ci"]["resealed_after_push"] is True
    for name, want in before.items():
        assert stage.sha256_file(out_dir / name) == want, name
    assert json.loads((out_dir / "admissibility_decision.json").read_text())["verdict"] == verdict_before
    assert "hosted CI observed after push" in (out_dir / "l1a2v_final_report.md").read_text()
    with pytest.raises(stage.AuditBlocked):
        stage.reseal_ci_status("definitely-a-status", "1" * 40, out_dir)
    with pytest.raises(stage.AuditBlocked):
        stage.reseal_ci_status("success", "not-a-sha", out_dir)


def test_report_carries_every_required_section(tmp_path):
    result = stage.run(out=tmp_path / "evidence", stages="all")
    report = (tmp_path / "evidence" / "l1a2v_final_report.md").read_text()
    for heading in (
        "## A. Scope, authority and baseline",
        "## B. Frozen contract and prohibitions",
        "## C. Sealed evidence verification",
        "## D. Prespecified screen contract",
        "## E. t=0 per-candidate measurements",
        "## F. Near-wall band classification",
        "## G. Shape invariance",
        "## H. Admissibility decision",
        "## I. Failure injection and adversarial checks",
        "## J. Quality, tests and hosted CI",
        "## K. Blockers and inherited statuses",
        "## L. Exactly one next action",
        "## M. Status and provenance",
    ):
        assert heading in report, heading
    assert result["preflight.json"]["trajectories_run"] == 0
    assert stage.BASELINE_HEAD_SHA in report
    assert "NO_ADMISSIBLE_RADIAL_SEED_UNDER_PRESPECIFIED_FAMILY" in report


def test_unmeasured_values_render_as_labels_never_as_zero():
    assert stage._fmt(None) == "UNMEASURED"
    assert stage._fmt(np.int64(7)) == "7"
    assert stage._fmt(float("nan")) == "nan"
    assert stage._fmt("global_speed_envelope") == "global_speed_envelope"
