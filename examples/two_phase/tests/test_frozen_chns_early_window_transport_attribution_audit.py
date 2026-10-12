"""L1A-2x frozen CHNS early-window interface transport attribution: identities and fail-closed gates.

Every test either verifies a prespecified property of the single sanctioned window, an exact identity
of the recorded discrete increments, a provenance/seal rule, or a fail-closed failure mode. No test
edits a frozen production source, a threshold, a historical evidence file or the quality baseline, and
no test runs a trajectory: the window is executed once by the stage itself and every physical assertion
here is made against that sealed evidence plus cheap synthetic fixtures.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from production import frozen_chns_early_window_transport_attribution_audit as stage  # noqa: E402

TWO_PHASE = Path(__file__).resolve().parents[1]
EVIDENCE = TWO_PHASE / "evidence" / "l1a2x"
MODULE_PATH = TWO_PHASE / "production" / "frozen_chns_early_window_transport_attribution_audit.py"

EVIDENCE_PAYLOADS = (
    "mechanism_decision.json",
    "substep_ledger.json",
    "interface_attribution.json",
    "residual_decomposition.json",
    "conservation_and_convergence.json",
    "capture_equivalence.json",
    "validity_screen.json",
    "initial_state_identity.json",
    "source_and_provenance.json",
    "frozen_evidence_verification.json",
    "trajectory_specification.json",
    "adversarial_and_falsification.json",
    "quality_and_blocker_matrix.json",
    "next_action.json",
    "preflight.json",
)


@pytest.fixture(scope="module")
def evidence() -> dict:
    payload = {name: json.loads((EVIDENCE / name).read_text()) for name in EVIDENCE_PAYLOADS}
    payload["manifest.json"] = json.loads((EVIDENCE / "manifest.json").read_text())
    payload["report.md"] = (EVIDENCE / "l1a2x_final_report.md").read_text()
    payload["report.json"] = json.loads((EVIDENCE / "l1a2x_final_report.json").read_text())
    return payload


@pytest.fixture(scope="module")
def bundle():
    bundle, _state = stage._bundle_and_state()
    return bundle


@pytest.fixture(scope="module")
def support(bundle):
    _bundle, state = stage._bundle_and_state()
    return stage.fixed_support(bundle, state)


def _copy_evidence(tmp_path: Path) -> Path:
    target = tmp_path / "l1a2x"
    shutil.copytree(EVIDENCE, target)
    return target


# ---------------------------------------------------------------------------
# prespecified constants, scope and prohibitions
# ---------------------------------------------------------------------------
def test_prespecified_constants_are_the_owner_mandated_ones():
    assert stage.CASE_NAME == "pillar_training"
    assert stage.PROFILE_N == 192
    assert stage.REQUESTED_DT == 0.002
    assert stage.MAX_MACRO_STEPS == 40
    assert stage.SUBSTEPS_PER_MACRO == 3
    assert stage.CONTACT_GAP_THRESHOLD_DX == 1.5
    assert stage.FIXED_POINT_PROBE_BAND == 0.10
    assert stage.LOCAL_GRADIENT_FLOOR_RATIO == 0.5
    assert stage.MAX_SUBSTEP_DISPLACEMENT_OVER_EPS == 0.25
    assert stage.CONTOUR_UNIQUENESS_MARGIN_DX == 0.01
    assert stage.DOMINANCE_MARGIN_FACTOR == 3.0
    assert stage.CH_CANCELLATION_QUALIFIER == 0.25
    assert stage.EQUIVALENCE_MAX_PHI_DIFF == 1e-6
    assert stage.EQUIVALENCE_MAX_GAP_DIFF_DX == 1e-4
    assert stage.RAY_HALF_WINDOW_DX == 2.0
    assert stage.RAY_SCAN_STEP_DX == 0.05
    assert stage.RAY_ROOT_TOLERANCE_DX == 1e-6
    assert stage.SUPPORT_CROSSING_RADIUS_DX == 1.5


def test_baseline_branch_and_contract_are_the_owner_mandated_ones():
    assert stage.BASELINE_HEAD_SHA == "45749bc603568bdac8629a90e529ae1573796d51"
    assert stage.VERIFIED_MAIN_SHA == "09e2993b8da478d79e768054194201af57121784"
    assert stage.BRANCH == "arena/02b6c71b-hydrogym"
    assert stage.STAGE == "L1A-2x"
    assert stage.STAGE_VERSION == "l1a2x_v1"
    assert len(stage.BASELINE_HEAD_SHA) == 40


def test_provenance_records_contract_twelve_and_the_frozen_status_flags(evidence):
    provenance = evidence["source_and_provenance.json"]
    assert provenance["solver_contract_version"] == 12
    assert provenance["frozen_status_flags"] == {
        "contract": 12,
        "L1A_STATUS": "BLOCKED",
        "L1B_DATA_NOT_READY": "unchanged",
    }
    assert provenance["branch"] == stage.BRANCH
    assert provenance["baseline_head_sha"] == stage.BASELINE_HEAD_SHA
    assert provenance["trajectories_run"] == 1


def test_exactly_one_trajectory_and_no_control_arm_is_run(evidence):
    ledger = evidence["substep_ledger.json"]
    prohibitions = ledger["prohibitions"]
    assert ledger["macro_steps_run"] == 40
    assert ledger["substeps_recorded"] == 120
    assert prohibitions == {
        "control_arm_runs": 0,
        "refinement_runs": 0,
        "contact_events_scored": 0,
        "t8_recomputed": False,
        "l1b_data_generated": False,
        "ml_training_runs": 0,
        "solver_source_edited": False,
        "initializer_promoted": False,
    }
    manifest = evidence["manifest.json"]
    assert manifest["trajectories_run"] == 1
    assert manifest["control_arms_run"] == 0
    assert manifest["contact_events_scored"] == 0
    assert manifest["refinement_studies_run"] == 0
    assert manifest["t8_recomputed"] is False
    assert manifest["l1b_data_generated"] is False
    assert manifest["ml_training_runs"] == 0


def test_initial_state_is_the_frozen_c2_and_never_a_forbidden_seed(evidence):
    identity = evidence["initial_state_identity.json"]
    assert identity["candidate_key"] == stage.stage_2u.C2_KEY == "C2_BASE"
    assert identity["case_name"] == stage.CASE_NAME
    assert "SDF_TAPERED_STREAMFUNCTION_V1" in identity["state_source"]
    assert "frozen" in identity["state_source"]
    forbiddens = " ".join(identity["prohibited_alternatives_not_used"])
    assert "UNIFORM_ALL_DOMAIN" in forbiddens
    assert "STREAMFUNCTION_LOCALIZED_V0" in forbiddens
    assert "ramp" in forbiddens
    assert "radial" in forbiddens
    assert all(identity["identity_checks"].values())


def test_module_runs_the_frozen_c2_construction_and_never_a_forbidden_initializer():
    source = MODULE_PATH.read_text()
    assert "build_any_state(bundle, stage_2u.C2_KEY)" in source
    assert "prohibited_alternatives_not_used" in source
    assert source.count("UNIFORM_ALL_DOMAIN") == 1
    assert source.count("STREAMFUNCTION_LOCALIZED_V0") == 1
    assert "0.354" not in source
    assert "t_contact" not in source


def test_stage_declares_only_its_own_twelve_stages_and_eighteen_evidence_files():
    assert stage.STAGES == (
        "preflight",
        "sealed",
        "identity",
        "window",
        "attribution",
        "conservation",
        "equivalence",
        "validity",
        "decision",
        "adversarial",
        "quality",
        "assemble",
    )
    assert len(stage.EVIDENCE_FILES) == 18
    assert len(set(stage.EVIDENCE_FILES)) == 18
    assert "manifest.json" in stage.EVIDENCE_FILES
    assert tuple(stage.EVIDENCE_FILES).count("manifest.json") == 1


# ---------------------------------------------------------------------------
# prerequisite seals and source hashes
# ---------------------------------------------------------------------------
def test_prerequisite_stages_verify_with_real_seals_and_source_hashes():
    report = stage.verify_prerequisites()
    assert report["all_verified"] is True
    assert report["failures"] == []
    assert report["verified_main_sha"] == stage.VERIFIED_MAIN_SHA
    assert {name: entry["bound_files"] for name, entry in report["stages"].items()} == {
        "L1A-2t": 14,
        "L1A-2u": 13,
        "L1A-2v": 14,
        "L1A-2w": 16,
    }
    assert all(entry["status"] == "VERIFIED" for entry in report["stages"].values())
    assert all(entry["problems"] == [] for entry in report["stages"].values())
    assert all(entry["source_hashes_unverifiable"] == [] for entry in report["stages"].values())
    assert sum(entry["source_hashes_recorded"] for entry in report["stages"].values()) == 28


def test_seal_reader_supports_every_frozen_manifest_convention():
    sha = "a" * 64
    assert stage._recorded_evidence_seal({"files": {"x.json": sha}}) == {"x.json": sha}
    assert stage._recorded_evidence_seal({"evidence_files": {"x.json": {"sha256": sha}}}) == {"x.json": sha}
    assert stage._recorded_evidence_seal({"evidence_files": {"x.json": sha}}) == {"x.json": sha}
    assert stage._recorded_evidence_seal({"files": {}, "evidence_files": {}}) == {}
    assert stage._recorded_evidence_seal({}) == {}


def test_an_empty_seal_is_never_counted_as_verified(tmp_path):
    target = _copy_evidence(tmp_path)
    manifest = json.loads((target / "manifest.json").read_text())
    del manifest["files"]
    (target / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(stage.AuditBlocked):
        stage.verify_evidence_manifest(target)


def test_an_unverifiable_recorded_source_hash_fails_closed():
    report = stage._verify_recorded_sources({"ghost_module": "0" * 64})
    assert report["checked"] == 1
    assert report["unverifiable"] == ["ghost_module"]
    assert report["problems"]
    assert stage._verify_recorded_sources(None)["problems"] == []


def test_preflight_fails_closed_on_a_commit_above_the_baseline_that_is_not_stage_owned(monkeypatch):
    real_git = stage._git

    def fake_git(args, *, required=True):
        if args[:2] == ["rev-parse", "HEAD"] and len(args) == 2:
            return "a" * 40
        if args[:3] == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return stage.BRANCH
        if args[:2] == ["status", "--porcelain"]:
            return ""
        if args[:2] == ["log", "--format=%H"] and args[2].endswith(stage.BASELINE_HEAD_SHA):
            return stage.BASELINE_HEAD_SHA
        if args[:2] == ["log", "--format=%H"]:
            return "b" * 40
        if args[:3] == ["rev-parse", "--is-shallow-repository"]:
            return "false"
        return real_git(args, required=required)

    monkeypatch.setattr(stage, "_git", fake_git)
    monkeypatch.setattr(stage, "_commit_touches_only_stage_paths", lambda commit: False)
    with pytest.raises(stage.AuditBlocked):
        stage.preflight_state()


def test_prerequisite_verification_fails_closed_on_a_tampered_or_missing_bound_file(monkeypatch, tmp_path):
    copied = tmp_path / "evidence" / "l1a2v"
    shutil.copytree(TWO_PHASE / "evidence" / "l1a2v", copied)
    monkeypatch.setattr(stage, "TWO_PHASE", tmp_path)
    monkeypatch.setattr(stage, "FROZEN_STAGES", (("L1A-2v", "l1a2v"),))
    assert stage.verify_prerequisites()["stages"]["L1A-2v"]["status"] == "VERIFIED"
    (copied / "admissibility_decision.json").write_text("{}\n")
    with pytest.raises(stage.AuditBlocked):
        stage.verify_prerequisites()
    shutil.copyfile(
        TWO_PHASE / "evidence" / "l1a2v" / "admissibility_decision.json", copied / "admissibility_decision.json"
    )
    (copied / "failure_and_adversarial_tests.json").unlink()
    with pytest.raises(stage.AuditBlocked):
        stage.verify_prerequisites()


def test_prerequisite_verification_fails_closed_on_an_unverifiable_source_hash(monkeypatch, tmp_path):
    copied = tmp_path / "evidence" / "l1a2v"
    shutil.copytree(TWO_PHASE / "evidence" / "l1a2v", copied)
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_hashes"] = {"ghost_module": "0" * 64}
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(stage, "TWO_PHASE", tmp_path)
    monkeypatch.setattr(stage, "FROZEN_STAGES", (("L1A-2v", "l1a2v"),))
    with pytest.raises(stage.AuditBlocked):
        stage.verify_prerequisites()


def test_recorded_source_hashes_are_verified_by_content_including_history():
    report = stage._verify_recorded_sources({"phasefield": stage.sha256_file(TWO_PHASE / "phasefield.py")})
    assert report["problems"] == []
    assert report["verified_from_the_worktree"] == {"phasefield": ["examples/two_phase/phasefield.py"]}


# ---------------------------------------------------------------------------
# evidence integrity, sealing and the hosted CI record
# ---------------------------------------------------------------------------
def test_evidence_directory_seals_and_verifies_every_required_file():
    report = stage.verify_evidence_manifest(EVIDENCE)
    assert report["verified"] is True
    assert report["problems"] == []
    assert report["missing_required"] == []
    assert report["bound_files"] == len(stage.EVIDENCE_FILES) - 1
    manifest = json.loads((EVIDENCE / "manifest.json").read_text())
    assert set(manifest["files"]) == set(stage.EVIDENCE_FILES) - {"manifest.json"}
    assert set(manifest["required_files"]) == set(stage.EVIDENCE_FILES)
    for name in stage.EVIDENCE_FILES:
        assert (EVIDENCE / name).is_file(), name


def test_evidence_verification_detects_a_tampered_bound_file(tmp_path):
    target = _copy_evidence(tmp_path)
    ledger = json.loads((target / "substep_ledger.json").read_text())
    ledger["macro_steps_run"] = 41
    (target / "substep_ledger.json").write_text(json.dumps(ledger))
    with pytest.raises(stage.AuditBlocked):
        stage.verify_evidence_manifest(target)


@pytest.mark.parametrize(
    "key, value",
    [
        ("initializer_promoted", True),
        ("contact_events_scored", 1),
        ("trajectories_run", 2),
        ("t8_recomputed", True),
        ("l1b_data_generated", True),
        ("no_promotion", False),
        ("verdict", "CONTACT_MECHANISM_CONFIRMED"),
    ],
)
def test_evidence_verification_rejects_a_forbidden_or_unprespecified_claim(tmp_path, key, value):
    target = _copy_evidence(tmp_path)
    manifest = json.loads((target / "manifest.json").read_text())
    manifest[key] = value
    (target / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(stage.AuditBlocked):
        stage.verify_evidence_manifest(target)


def test_evidence_verification_rejects_a_missing_required_artifact(tmp_path):
    target = _copy_evidence(tmp_path)
    (target / "interface_attribution.json").unlink()
    with pytest.raises(stage.AuditBlocked):
        stage.verify_evidence_manifest(target)


def test_reseal_refuses_unobserved_inputs(tmp_path):
    target = _copy_evidence(tmp_path)
    with pytest.raises(stage.AuditBlocked):
        stage.reseal_ci_status("green", "45749bc603568bdac8629a90e529ae1573796d51", target)
    with pytest.raises(stage.AuditBlocked):
        stage.reseal_ci_status("success", "45749bc", target)
    with pytest.raises(stage.AuditBlocked):
        stage.reseal_ci_status("success", "z" * 40, target)


def test_reseal_records_only_the_observed_status_and_touches_no_physics(tmp_path):
    target = _copy_evidence(tmp_path)
    before = {
        name: stage.sha256_file(target / name)
        for name in stage.EVIDENCE_FILES
        if name not in {"quality_and_blocker_matrix.json", "l1a2x_final_report.md", "manifest.json"}
    }
    head = "45749bc603568bdac8629a90e529ae1573796d51"
    report = stage.reseal_ci_status("success", head, target)
    assert report["verified"] is True
    quality = json.loads((target / "quality_and_blocker_matrix.json").read_text())
    assert quality["hosted_ci"]["status_for_this_head"] == "success"
    assert quality["hosted_ci"]["observed_head_sha"] == head
    assert quality["hosted_ci"]["resealed_after_push"] is True
    report_text = (target / "l1a2x_final_report.md").read_text()
    assert f"- hosted CI observed after push: `success` for head `{head}`" in report_text
    assert report_text.count("- hosted CI observed after push") == 1
    for name, want in before.items():
        assert stage.sha256_file(target / name) == want, name


def test_manifest_binds_the_module_and_never_claims_an_unpushed_ci_result(evidence):
    manifest = evidence["manifest.json"]
    assert manifest["module_sha256"] == stage.sha256_file(MODULE_PATH)
    assert manifest["module_sha256"] == evidence["source_and_provenance.json"]["module_sha256"]
    assert manifest["no_hosted_ci_claim_before_push"] is True
    assert manifest["no_promotion"] is True
    assert manifest["solver_contract_version"] == 12
    assert manifest["baseline_head_sha"] == stage.BASELINE_HEAD_SHA
    assert manifest["verified_main_sha"] == stage.VERIFIED_MAIN_SHA
    assert manifest["evidence_class"] == "FROZEN_CHNS_ATTRIBUTION"
    reseal = manifest.get("hosted_ci_reseal")
    if reseal is not None:
        assert reseal["performed"] is True
        assert len(reseal["observed_head_sha"]) == 40
        assert reseal["physics_evidence_untouched"] is True


# ---------------------------------------------------------------------------
# the window, the initial state and the fixed support
# ---------------------------------------------------------------------------
def test_window_stays_inside_the_early_response_and_never_reaches_contact(evidence):
    ledger = evidence["substep_ledger.json"]
    assert ledger["requested_dt"] == 0.002
    assert ledger["substeps_per_macro"] == 3
    assert ledger["t_final"] <= 0.08 + 1e-12
    assert ledger["contact_guard"]["triggered"] is False
    assert ledger["contact_guard"]["macro_step"] is None
    assert ledger["contact_threshold_over_dx"] == 1.5
    gaps = [row["gap_global_over_dx"] for row in ledger["macro_rows"]]
    assert min(gaps) > 1.5
    assert len(ledger["rows"]) == 3 * len(ledger["macro_rows"]) == 120
    assert [row["substep"] for row in ledger["rows"]] == list(range(120))


def test_run_budget_is_capped_and_refuses_an_over_cap_request():
    with pytest.raises(stage.AuditBlocked):
        stage.run_ledger(stage.MAX_MACRO_STEPS + 1)
    with pytest.raises(stage.AuditBlocked):
        stage.run_ledger(0)


def test_initial_state_identity_matches_the_sealed_hashes_of_the_upstream_stage(evidence):
    identity = evidence["initial_state_identity.json"]
    assert identity["hashes"] == identity["sealed_hashes"]
    assert identity["hashes"]["phi"].startswith("3022be21")
    assert identity["hashes"]["u"].startswith("2b921fbc")
    assert identity["hashes"]["v"].startswith("3c0963be")


def test_initial_state_identity_is_reproducible_from_the_frozen_construction(evidence):
    recomputed = stage.initial_state_identity()
    committed = evidence["initial_state_identity.json"]
    assert recomputed["hashes"] == committed["hashes"]
    assert recomputed["support"]["cell"] == committed["support"]["cell"]
    assert recomputed["parameter_hash"] == committed["parameter_hash"]
    assert recomputed["geometry_hash"] == committed["geometry_hash"]


def test_fixed_support_and_linearisation_coefficient_reproduce_from_the_frozen_state(support, evidence):
    assert support["cell"] == [95, 23]
    assert support["gap_over_dx"] == pytest.approx(2.70598, abs=1e-5)
    assert support["gradient_norm"] == pytest.approx(7.023055, rel=1e-6)
    assert support["wall_normal_into_solid"] == [0.0, -1.0]
    assert support["interface_normal"][0] == pytest.approx(0.023917, abs=1e-6)
    assert support["interface_normal"][1] == pytest.approx(0.999714, abs=1e-6)
    recorded = evidence["substep_ledger.json"]["linearisation"]
    assert recorded["coefficient0_over_dx"] == pytest.approx(
        -recorded["interface_normal_dot_wall_normal"]
        / recorded["t0_gradient_norm_at_support"]
        / evidence["substep_ledger.json"]["parameters"]["dx"],
        rel=1e-12,
    )
    assert recorded["coefficient0_over_dx"] == pytest.approx(4.555118, rel=1e-6)


def test_the_frozen_support_edge_carries_the_t0_crossing_through_the_support_point(evidence):
    edge = evidence["substep_ledger.json"]["frozen_support_edge"]
    assert edge["identity"]["axis"] == 1
    assert [edge["identity"]["i"], edge["identity"]["j"]] == [95, 23]
    assert edge["identity"]["t0_radius_over_dx"] == 0.0
    assert edge["t0_tracking"]["y_over_dx"] == pytest.approx(23.505981, abs=1e-6)
    assert edge["t0_tracking"]["branch_lost"] is False


# ---------------------------------------------------------------------------
# the per-substep discrete ledger
# ---------------------------------------------------------------------------
def test_every_substep_recombines_its_three_recorded_increments(evidence):
    ledger = evidence["substep_ledger.json"]
    assert max(row["recombination_linf"] for row in ledger["rows"]) <= 1e-14
    assert max(abs(row["mass_residual"]) for row in ledger["rows"]) <= 1e-14
    worst_increment = max(row["dphi_norm_linf"] for row in ledger["rows"])
    assert worst_increment > 0.0
    for row in ledger["rows"]:
        assert row["t_after"] > row["t_before"]
        assert row["advective_flux_linf"] >= 0.0
        assert row["chemical_flux_linf"] >= 0.0


def test_the_recorded_increments_telescope_to_the_reported_cumulative_values(evidence):
    ledger = evidence["substep_ledger.json"]
    attribution = evidence["interface_attribution.json"]
    coefficient = attribution["coefficient_over_dx"]
    for key, ingredient in (
        ("adv", "dphi_adv_at_support"),
        ("ch", "dphi_ch_at_support"),
        ("imp", "dphi_imp_at_support"),
        ("full", "dphi_full_at_support"),
    ):
        total = sum(coefficient * row[ingredient] for row in ledger["rows"])
        assert total == pytest.approx(attribution["cumulative_dn_dx"][key], rel=1e-12)
    net = attribution["cumulative_dn_dx"]["ch"] + attribution["cumulative_dn_dx"]["imp"]
    assert net == pytest.approx(attribution["cumulative_dn_dx"]["net_ch"], rel=1e-12)
    closure = attribution["cumulative_dn_dx"]["full"] - attribution["cumulative_dn_dx"]["adv"] - net
    assert abs(closure) <= 1e-9


def test_mass_and_impermeable_walls_hold_over_every_substep(evidence):
    conservation = evidence["conservation_and_convergence.json"]
    mass = conservation["mass"]
    assert mass["mass_drift_absolute"] == 0.0
    assert mass["mass_drift_relative"] == 0.0
    assert abs(mass["max_abs_telescoping_adv"]) <= 1e-15
    assert abs(mass["max_abs_telescoping_ch"]) <= 1e-15
    wall = conservation["wall"]
    assert wall["max_abs_flux_on_closed_faces"] == 0.0
    assert wall["closed_face_count_x"] > 0
    assert wall["closed_face_count_y"] > 0
    assert conservation["recombination"]["max_recombination_linf"] <= 1e-14
    assert conservation["recombination"]["max_abs_substep_increment_linf"] > 0.0


def test_every_implicit_solve_converged_under_the_frozen_tolerance(evidence):
    implicit = evidence["conservation_and_convergence.json"]["implicit_solve"]
    assert implicit["substeps"] == 120
    assert implicit["all_converged"] is True
    assert implicit["max_relative_residual"] <= implicit["rtol"]
    assert implicit["max_iterations"] <= implicit["iteration_cap"]
    ledger = evidence["substep_ledger.json"]
    assert implicit["max_iterations"] == max(row["cg_iterations"] for row in ledger["rows"])
    assert max(row["cg_relative_residual"] for row in ledger["rows"]) <= implicit["rtol"]


def test_the_capture_reproduces_the_frozen_public_step_within_the_thresholds(evidence):
    equivalence = evidence["capture_equivalence.json"]
    assert equivalence["macro_steps_compared"] == 40
    assert equivalence["max_phi_difference_vs_eager_public_step"] <= stage.EQUIVALENCE_MAX_PHI_DIFF
    assert equivalence["max_phi_difference_vs_jitted_public_step"] <= stage.EQUIVALENCE_MAX_PHI_DIFF
    assert equivalence["max_gap_difference_over_dx"] <= stage.EQUIVALENCE_MAX_GAP_DIFF_DX
    assert equivalence["max_velocity_difference_vs_eager_public_step"] <= 1e-5
    assert equivalence["max_phi_difference_vs_eager_public_step"] > 0.0
    assert equivalence["thresholds"] == {
        "max_phi_difference": stage.EQUIVALENCE_MAX_PHI_DIFF,
        "max_gap_difference_over_dx": stage.EQUIVALENCE_MAX_GAP_DIFF_DX,
    }
    validity = evidence["validity_screen.json"]
    assert validity["equivalence_ok"] is True


# ---------------------------------------------------------------------------
# attribution, residuals and the frozen contour reconstructions
# ---------------------------------------------------------------------------
def test_the_frozen_reconstructions_and_the_linearisation_measure_the_same_span(evidence):
    residuals = evidence["residual_decomposition.json"]
    coverage = residuals["tracked_edge_coverage"]
    assert coverage["macros_available"] >= 1
    assert coverage["first_macro_step"] == 1
    common = residuals["window_where_every_measurement_exists"]["cumulative_dn_dx"]
    assert common["d1_frozen_global_gap"] == pytest.approx(common["d2_frozen_tracked_edge"], abs=1e-12)
    assert common["d1_frozen_global_gap"] == pytest.approx(common["d3_bilinear_line_root"], abs=1e-4)
    linearised = residuals["window_where_every_measurement_exists"]["cumulative_dn_dx"]["d4_linearised_point_estimate"]
    assert abs(linearised - common["d1_frozen_global_gap"]) <= 0.05 * abs(common["d1_frozen_global_gap"])
    assert residuals["window_where_every_measurement_exists"]["total_band_dx"] > 0.0


def test_the_residual_terms_are_separated_and_algebraically_consistent(evidence):
    residuals = evidence["residual_decomposition.json"]
    for key in residuals["cumulative_abs_band"]:
        assert np.isfinite(residuals["cumulative_abs_band"][key])
    assert abs(residuals["cumulative_signed"]["R_operator_splitting"]) <= 1e-12
    assert residuals["cumulative_abs_band"]["R_support_migration"] <= 1e-4
    assert residuals["cumulative_abs_band"]["R_interpolation"] <= 1e-3
    assert abs(residuals["algebraic_identity_residuals"]["identity_ingredients_plus_splitting"]) <= 1e-14
    definition = residuals["definition"]
    for term in ("R_support_migration", "R_interpolation", "R_nonlinearity", "R_operator_splitting"):
        assert term in definition
    assert residuals["total_residual_band_dx"] == pytest.approx(
        sum(residuals["cumulative_abs_band"].values()), rel=1e-12
    )


def test_the_prespecified_attribution_rules_are_recorded_and_agree(evidence):
    attribution = evidence["interface_attribution.json"]
    rules = [entry[0] for entry in attribution["rules"]]
    assert rules == [
        "PRIMARY_POINT_T0_COEFFICIENT",
        "POINT_LOCAL_COEFFICIENT",
        "SUPPORT_CELL_T0_COEFFICIENT",
    ]
    assert set(attribution["family"]) == set(rules)
    assert attribution["rankings_agree_across_rules"] is True
    assert all(entry["ranking"] in {"ADVECTIVE_LARGER", "NET_CH_LARGER"} for entry in attribution["family"].values())
    assert attribution["margin_spread_across_rules_dx"] <= 1e-3
    for rule, entry in attribution["family"].items():
        ingredients = entry["cumulative_dn_dx"]
        assert ingredients["full"] == pytest.approx(
            ingredients["adv"] + ingredients["ch"] + ingredients["imp"], abs=1e-9
        ), rule
        assert entry["margin_dx"] == pytest.approx(
            abs(ingredients["adv"]) - abs(ingredients["ch"] + ingredients["imp"]), rel=1e-12
        ), rule
    assert len(attribution["per_macro"]) == 40
    assert len(attribution["per_substep"]["full"]) == 120


def test_the_cahn_hilliard_halves_are_reported_as_a_pair_not_as_two_terms(evidence):
    attribution = evidence["interface_attribution.json"]
    cancellation = attribution["cahn_hilliard_cancellation"]
    assert len(cancellation["per_macro"]) == 40
    assert 0.0 <= cancellation["cumulative_ratio"] <= 1.0
    assert cancellation["cumulative_ratio"] == pytest.approx(
        abs(attribution["cumulative_dn_dx"]["net_ch"])
        / (abs(attribution["cumulative_dn_dx"]["ch"]) + abs(attribution["cumulative_dn_dx"]["imp"])),
        rel=1e-12,
    )
    decision = evidence["mechanism_decision.json"]
    assert decision["cahn_hilliard_cancellation_qualifier"] is (
        cancellation["cumulative_ratio"] < stage.CH_CANCELLATION_QUALIFIER
    )


def test_advection_accumulates_toward_the_solid_while_the_early_recoil_is_chemical(evidence):
    attribution = evidence["interface_attribution.json"]
    increments = [row["dn_adv"] for row in attribution["per_macro"]]
    assert all(value >= 0.0 for value in increments)
    assert all(increments[index] <= increments[index + 1] for index in range(len(increments) - 1))
    assert attribution["cumulative_dn_dx"]["adv"] == pytest.approx(sum(increments), rel=1e-12)
    net_chemical = [row["dn_net_ch"] for row in attribution["per_macro"]]
    assert net_chemical[0] < 0.0
    assert sum(net_chemical[:8]) < 0.0
    ledger = evidence["substep_ledger.json"]
    velocities = [row["normal_velocity_at_support"] for row in ledger["macro_rows"]]
    assert all(value >= 0.0 for value in velocities)
    assert velocities[-1] > velocities[0]
    gaps = [row["gap_global_over_dx"] for row in ledger["macro_rows"]]
    assert max(gaps) > gaps[0] > gaps[-1]


# ---------------------------------------------------------------------------
# validity guards, the withheld verdict and the single next action
# ---------------------------------------------------------------------------
def test_every_blocking_guard_passed_and_only_the_support_change_withheld_the_verdict(evidence):
    validity = evidence["validity_screen.json"]
    assert validity["failures"]["gradient_resolved"] == 0
    assert validity["failures"]["probe_band"] == 0
    assert validity["failures"]["ray_root_exists"] == 0
    assert validity["failures"]["substep_locality"] == 0
    assert validity["failures"]["contour_unique"] == 40
    assert validity["blocking_failures"] == {"frozen_support_edge_alive": 16}
    assert validity["unresolved"] is True
    assert validity["direct_probe_unique"] is False
    assert validity["direct_probe_status"] == "UNRESOLVED_NON_UNIQUE_CONTOUR_MATCH_ON_THIS_SURFACE"
    assert validity["fixed_support_attribution_status"] == "UNRESOLVED"
    assert validity["observed"]["frozen_support_edge_branch_lost_at_macro_step"] == 25
    assert validity["observed"]["tracked_edge_continuity_macros_ok"] == 24
    assert validity["substeps"] == 120
    assert validity["observed"]["max_abs_probe_offset"] <= stage.FIXED_POINT_PROBE_BAND
    assert validity["observed"]["min_gradient_ratio_to_t0"] >= stage.LOCAL_GRADIENT_FLOOR_RATIO
    assert validity["observed"]["max_substep_displacement_over_eps"] <= stage.MAX_SUBSTEP_DISPLACEMENT_OVER_EPS


def test_the_verdict_is_unresolved_and_every_dominance_claim_is_withheld(evidence):
    decision = evidence["mechanism_decision.json"]
    assert decision["verdict"] in stage.DECISION_STATES
    assert decision["verdict"] == "EARLY_WINDOW_ATTRIBUTION_UNRESOLVED"
    assert decision["verdict_withheld_because"]
    reasons = " ".join(decision["verdict_withheld_because"])
    assert "frozen_support_edge_alive" in reasons
    assert "not unique" in reasons
    recorded = decision["dominance_rule_outcome_recorded_not_claimed"]
    assert set(recorded) == {
        "with_the_comparison_band",
        "with_the_total_residual_map_band",
        "note",
    }
    assert all(value in stage.DECISION_STATES for key, value in recorded.items() if key.startswith("with_the_"))
    assert decision["capture_equivalence_ok"] is True
    assert decision["scope"]["contact_events_scored"] == 0
    assert decision["scope"]["contact_threshold_reached"] is False
    assert decision["scope"]["window_limited_to_early_response"] is True
    assert all(decision["forbidden_claims_respected"].values())
    assert decision["next_action_is_not_started_in_this_stage"] is True


def test_the_decision_records_magnitudes_and_bands_without_a_claim(evidence):
    decision = evidence["mechanism_decision.json"]
    attribution = evidence["interface_attribution.json"]
    assert decision["cumulative_dn_dx"] == attribution["cumulative_dn_dx"]
    assert decision["advective_magnitude_dx"] == pytest.approx(abs(attribution["cumulative_dn_dx"]["adv"]), rel=1e-12)
    assert decision["net_chemical_hilliard_magnitude_dx"] == pytest.approx(
        abs(attribution["cumulative_dn_dx"]["net_ch"]), rel=1e-12
    )
    assert decision["dominance_margin_dx"] == pytest.approx(
        decision["advective_magnitude_dx"] - decision["net_chemical_hilliard_magnitude_dx"], rel=1e-12
    )
    assert decision["dominance_floor_dx"] == pytest.approx(
        stage.DOMINANCE_MARGIN_FACTOR * decision["comparison_band_dx"], rel=1e-12
    )
    assert decision["comparison_band_dx"] > 0.0
    assert decision["total_map_band_dx"] >= decision["comparison_band_dx"]
    assert "band" in decision["band_semantics"]


def test_the_single_next_action_is_bounded_and_deliberately_not_started(evidence):
    action = evidence["next_action.json"]
    assert action["must_not_be_started_in_this_stage"] is True
    assert action["requires_owner_approval"] is True
    assert action["bounded_scope"]["no_extension_toward_contact"] is True
    assert action["bounded_scope"]["no_control_arms"] is True
    assert action["bounded_scope"]["no_refinement"] is True
    assert action["bounded_scope"]["no_generator_or_ml"] is True
    assert action["bounded_scope"]["macro_step_cap"] == stage.MAX_MACRO_STEPS
    assert "contour arc" in action["action"]
    assert "No mechanism may be claimed" in action["action"]


def test_substep_guards_refuse_flat_drifting_and_oversized_rows(evidence):
    ledger = evidence["substep_ledger.json"]
    gradient0 = ledger["linearisation"]["t0_gradient_norm_at_support"]
    eps_over_dx = ledger["parameters"]["eps"] / ledger["parameters"]["dx"]
    coefficient = ledger["linearisation"]["coefficient0_over_dx"]
    healthy = ledger["rows"][0]
    assert all(stage.substep_guards(healthy, gradient0, eps_over_dx, coefficient).values())
    flat = dict(healthy, gradient_norm_at_support=0.0)
    assert stage.substep_guards(flat, gradient0, eps_over_dx, coefficient)["gradient_resolved"] is False
    drifting = dict(healthy, phi_at_support=0.5 + 1.5 * stage.FIXED_POINT_PROBE_BAND)
    assert stage.substep_guards(drifting, gradient0, eps_over_dx, coefficient)["probe_band"] is False
    lost = dict(healthy, line_root_over_dx=None)
    assert stage.substep_guards(lost, gradient0, eps_over_dx, coefficient)["ray_root_exists"] is False
    far = dict(healthy, dphi_full_at_support=1.0)
    assert stage.substep_guards(far, gradient0, eps_over_dx, coefficient)["substep_locality"] is False


def test_the_verdict_rule_withholds_on_a_near_tie_and_on_an_unresolved_guard():
    assert (
        stage._verdict_from_magnitudes(advective=1.0, chemical=1.0, band=0.1, unresolved=False)
        == "MIXED_TERMS_WITHIN_THE_RESIDUAL_BAND"
    )
    assert (
        stage._verdict_from_magnitudes(advective=1.0, chemical=0.1, band=0.1, unresolved=False)
        == "ADVECTIVE_TERM_CARRIES_THE_EARLY_WINDOW_CONTOUR_MOTION"
    )
    assert (
        stage._verdict_from_magnitudes(advective=0.1, chemical=1.0, band=0.1, unresolved=False)
        == "CAHN_HILLIARD_TERM_CARRIES_THE_EARLY_WINDOW_CONTOUR_MOTION"
    )
    assert (
        stage._verdict_from_magnitudes(advective=9.0, chemical=0.0, band=0.1, unresolved=True)
        == "EARLY_WINDOW_ATTRIBUTION_UNRESOLVED"
    )


# ---------------------------------------------------------------------------
# synthetic reconstruction checks (cheap, no trajectory)
# ---------------------------------------------------------------------------
def test_the_linearisation_recovers_a_known_translation_and_flips_with_the_wall_normal(bundle):
    for offset in (-0.5, -0.1, 0.1, 0.5):
        forward = stage._synthetic_translation_probe(bundle, offset)
        assert forward["linearised_estimate_holds"] is True
        mirror = stage._synthetic_translation_probe(bundle, offset, (0.0, -1.0))
        assert mirror["linearised_estimate_holds"] is True
        assert np.sign(mirror["recovered_displacement_toward_wall_over_dx"]) == np.sign(
            mirror["expected_displacement_toward_wall_over_dx"]
        )


def test_the_line_root_recovers_an_analytic_crossing_and_reports_a_missing_one(bundle, support):
    import phasefield as pf  # noqa: E402

    p = bundle.p
    origin = np.array(support["point"], dtype=np.float64)
    direction = np.array(support["interface_normal"], dtype=np.float64)
    grid_y = np.asarray(pf.grids(p)[1], dtype=np.float64)
    target = origin[1] + 0.25 * float(p.dx)
    field = 0.5 + 8.0 * (grid_y - target) / float(p.eps)
    _x, y_over_dx, info = stage._line_root(field, origin, direction, bundle)
    assert info["roots_found"] >= 1
    assert y_over_dx == pytest.approx(target / float(p.dx), abs=1e-5)
    flat = np.full_like(grid_y, 0.6)
    s, y, flat_info = stage._line_root(flat, origin, direction, bundle)
    assert s is None and y is None and flat_info["roots_found"] == 0


def test_the_tracked_edge_reports_a_lost_branch_when_it_no_longer_straddles(bundle, support):
    import phasefield as pf  # noqa: E402

    p = bundle.p
    origin = np.array(support["point"], dtype=np.float64)
    edge = stage._support_edge_identity(np.asarray(stage._bundle_and_state()[1].phi), origin, bundle)
    grid_y = np.asarray(pf.grids(p)[1], dtype=np.float64)
    inside = 0.5 + 8.0 * (grid_y - (origin[1] + 0.25 * float(p.dx))) / float(p.eps)
    moved = stage._tracked_edge_crossing(inside, edge, bundle)
    assert moved["branch_lost"] is False
    assert moved["y_over_dx"] == pytest.approx((origin[1] + 0.25 * float(p.dx)) / float(p.dx), abs=1e-9)
    far = 0.5 + 8.0 * (grid_y - (origin[1] + 4.0 * float(p.dx))) / float(p.eps)
    lost = stage._tracked_edge_crossing(far, edge, bundle)
    assert lost["branch_lost"] is True
    assert lost["y_over_dx"] is None


# ---------------------------------------------------------------------------
# report, quality matrix and adversarial record
# ---------------------------------------------------------------------------
def test_report_carries_every_required_section_and_the_headline(evidence):
    text = evidence["report.md"]
    for section in (
        "## A. Scope, authority and baseline",
        "## B. Frozen contract and prohibitions",
        "## C. Sealed evidence verification",
        "## D. Initial state identity",
        "## E. The window and its ledger",
        "## F. Ingredient closure, conservation and the implicit solve",
        "## G. Capture equivalence with the frozen public step",
        "## H. Fixed-support attribution",
        "## I. Residual decomposition",
        "## J. Validity screen",
        "## K. Decision",
        "## L. Adversarial and falsification checks",
        "## M. Quality, tests and hosted CI",
        "## N. Exactly one next action",
    ):
        assert section in text, section
    assert "recorded, not claimed" in text
    assert "per-macro-step fixed-support displacement" in text
    assert "the verdict is withheld because" in text
    assert "EARLY_WINDOW_ATTRIBUTION_UNRESOLVED" in text
    payload = evidence["report.json"]
    assert payload["stage"] == stage.STAGE
    assert payload["verdict"] == "EARLY_WINDOW_ATTRIBUTION_UNRESOLVED"
    assert len(payload["reference_times"]) == 40


def test_no_evidence_or_report_claims_an_impact_or_a_contact_mechanism(evidence):
    forbidden = ("contact event occurred", "confirmed contact", "impact mechanism", "postcontact mechanism")
    for name in ("mechanism_decision.json", "next_action.json", "report.json"):
        text = json.dumps(evidence[name])
        assert not any(phrase in text for phrase in forbidden), name
    report = evidence["report.md"]
    assert "it is not a real impact" in report
    assert "not extrapolated to the late contact time" in report


def test_adversarial_checks_all_hold_and_exercise_every_guard(evidence):
    adversarial = evidence["adversarial_and_falsification.json"]
    assert adversarial["all_hold"] is True
    checks = adversarial["checks"]
    assert checks["linearisation_recovers_a_known_translation"]["holds"] is True
    assert checks["linearisation_sign_flips_with_the_wall_normal"]["holds"] is True
    assert checks["gradient_guard_refuses_a_flat_profile"]["refused"] is True
    assert checks["probe_band_guard_refuses_a_drifting_probe"]["refused"] is True
    assert checks["ingredient_identity_catches_a_fabricated_increment"]["discriminates"] is True
    assert checks["frozen_sources_are_unmodified"]["hashes_hold"] is True
    assert checks["window_budget_is_capped"]["refused_over_cap_request"] is True
    assert checks["ledger_is_not_vacuous"]["max_abs_mass_adv"] >= 0.0
    assert checks["ledger_is_not_vacuous"]["closed_faces_exist"] is True


def test_quality_matrix_keeps_the_structural_and_the_lineage_failures_separate(evidence):
    quality = evidence["quality_and_blocker_matrix.json"]
    assert quality["declared_new_code_diagnostics"] == 0
    assert quality["ratchet_expectation"] == "PASS_BASELINE_ONLY"
    baseline = Path(stage.__file__).resolve().parents[3] / "quality" / "legacy_two_phase_debt_v1.json"
    assert quality["quality_baseline_sha256"] == stage.sha256_file(baseline)
    assert quality["quality_baseline_sha256"] != "MISSING"
    structural = quality["frozen_adjacent_suite"]
    assert structural["classification"] == "SUPERSEDED_BY_NEXT_STAGE_BY_CONSTRUCTION"
    assert structural["suite_failures"] == {"l1a2t": 1, "l1a2u": 0, "l1a2v": 5, "l1a2w": 0}
    assert len(structural["failing"]) == sum(structural["suite_failures"].values()) == 6
    lineage = quality["legacy_lineage_suite"]
    assert lineage["count"] == 1
    assert lineage["classification"] == "FROZEN_LINEAGE_DRIFT_CONFIRMED"
    assert lineage["failing"] == [
        "tests/test_impact_initialization_compatibility_audit.py::test_l1a2r_merged_base_verified_or_fail_closed"
    ]
    assert structural["failing"] != lineage["failing"]
    observed = quality["frozen_sweep_observed_at_the_generating_tree_state"]
    assert observed["passed"] == 163 and observed["failed"] == 8
    assert len(observed["failing"]) == 8
    assert set(observed["failing"]) >= set(structural["failing"])
    assert observed["caused_by_the_uncommitted_stage_deliverables"] == [
        "tests/test_radial_incident_seed_eta_dt_audit.py::test_report_contains_every_required_section",
        "tests/test_boundary_compatible_incident_momentum_bound_audit.py::test_preflight_is_clean_on_the_repository_head",
    ]
    assert set(observed["failing"]) - set(structural["failing"]) == set(
        observed["caused_by_the_uncommitted_stage_deliverables"]
    )
    for item in ("control trajectories", "contact events", "T=8 recomputation", "L1B data generation", "ML training"):
        assert item in quality["not_run"]


def test_the_module_hash_recorded_in_the_evidence_is_the_module_on_disk(evidence):
    provenance = evidence["source_and_provenance.json"]
    digest = hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest()
    assert provenance["module_sha256"] == digest
    assert (
        provenance["module_path"]
        == "examples/two_phase/production/frozen_chns_early_window_transport_attribution_audit.py"
    )
    for name, want in provenance["frozen_source_sha256"].items():
        target = TWO_PHASE / name.replace("examples/two_phase/", "")
        assert stage.sha256_file(target) == want, name
