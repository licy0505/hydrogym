"""L1A-2y reconstruction-stable early-window interface attribution: identities and fail-closed gates.

Every test either verifies a prespecified property of the single sanctioned window, an exact identity of
the recorded discrete increments, a rule of the prespecified contour-branch readout, a provenance/seal
rule, or a fail-closed failure mode. No test edits a frozen production source, a threshold, a historical
evidence file or the quality baseline, and no test runs a trajectory: the one window is executed by the
stage itself and every physical assertion here is made against that sealed evidence plus cheap synthetic
fixtures built by the frozen readout module.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from production import frozen_reconstruction_branch_attribution_audit as stage  # noqa: E402

TWO_PHASE = Path(__file__).resolve().parents[1]
EVIDENCE = TWO_PHASE / "evidence" / "l1a2y"
MODULE_PATH = TWO_PHASE / "production" / "frozen_reconstruction_branch_attribution_audit.py"
READOUT_PATH = TWO_PHASE / "production" / "l1a2y_contour_branch_readout.py"

EVIDENCE_PAYLOADS = tuple(
    name for name in stage.EVIDENCE_FILES if name not in {"manifest.json", "l1a2y_final_report.md"}
)

FORBIDDEN_PHRASES = (
    "real impact",
    "impact mechanism",
    "confirmed contact",
    "contact event occurred",
    "two independent mechanisms",
)


@pytest.fixture(scope="module")
def evidence() -> dict:
    payload = {name: json.loads((EVIDENCE / name).read_text()) for name in EVIDENCE_PAYLOADS}
    payload["manifest.json"] = json.loads((EVIDENCE / "manifest.json").read_text())
    payload["report.md"] = (EVIDENCE / "l1a2y_final_report.md").read_text()
    payload["report.json"] = json.loads((EVIDENCE / "l1a2y_final_report.json").read_text())
    return payload


@pytest.fixture(scope="module")
def ledger(evidence) -> dict:
    return evidence["substep_ledger.json"]


def _copy_evidence(tmp_path: Path) -> Path:
    target = tmp_path / "l1a2y"
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
    assert stage.DOMINANCE_FACTOR == 3.0
    assert stage.ATTRIBUTION_CLOSURE_BAND_DX == 0.01
    assert stage.TOTAL_DISPLACEMENT_CLOSURE_BAND_DX == 0.01
    assert stage.WALK_NOISE_FLOOR_DX == 1.0e-3
    assert stage.CROSS_CHECK_TOLERANCE_DX == 0.05
    assert stage.STATIC_RECONSTRUCTION_TOLERANCE_DX == 0.01
    assert stage.EQUIVALENCE_MAX_PHI_DIFF == 1e-6
    assert stage.EQUIVALENCE_MAX_GAP_DIFF_DX == 1e-4
    assert stage.DECISION_STATES == (
        "ADVECTIVE_TERM_CARRIES_THE_EARLY_WINDOW_CONTOUR_MOTION",
        "CAHN_HILLIARD_TERM_CARRIES_THE_EARLY_WINDOW_CONTOUR_MOTION",
        "MIXED_TERMS_WITHIN_THE_COMPARISON_ERROR",
        "EARLY_WINDOW_ATTRIBUTION_UNRESOLVED",
    )


def test_readout_constants_are_the_prespecified_ones():
    readout = stage.readout
    assert readout.ROOT_MATCH_MAX_STEP_DX == 0.25
    assert readout.RAY_SEARCH_HALF_WINDOW_DX == 1.5
    assert readout.ROOT_AMBIGUITY_SEPARATION_DX == 0.5
    assert readout.ARC_ROOT_MATCH_TOLERANCE_DX == 0.05
    assert readout.ARC_WINDOW_RADIUS_DX == 3.0
    assert readout.CROSS_RAY_TOLERANCE_DX == 0.005
    assert readout.GRADIENT_FLOOR_RATIO == 0.5
    assert readout.PROBE_BAND == 0.10
    assert readout.MAX_SUBSTEP_DISPLACEMENT_OVER_EPS == 0.25
    assert readout.SADDLE_AMBIGUITY_TOLERANCE == 1e-3
    assert readout.RAY_SCAN_STEP_DX == 0.05
    assert readout.READOUT_VERSION == "contour_branch_v1"


def test_baseline_branch_and_contract_are_the_owner_mandated_ones():
    assert stage.BASELINE_HEAD_SHA == "3695282e1de6c990514636b28d185ba3f4427a5b"
    assert stage.VERIFIED_MAIN_SHA == "09e2993b8da478d79e768054194201af57121784"
    assert stage.BRANCH == "arena/02b6c71b-hydrogym"
    assert stage.STAGE == "L1A-2y"
    assert stage.STAGE_VERSION == "l1a2y_v1"
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
    assert provenance["phase_boundary_model"] == "impermeable_flux"
    assert provenance["phase_storage_model"] == "phase_only_float64_v1"
    assert provenance["phase_transport_geometry"] == "sdf_cutcell_fv_v1"


def test_exactly_one_trajectory_and_no_control_arm_is_run(evidence, ledger):
    assert ledger["macro_steps_run"] == 40
    assert ledger["substeps_recorded"] == 120
    assert ledger["prohibitions"] == {
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
    assert manifest["solver_source_edited"] is False
    assert manifest["initializer_promoted"] is False


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
    assert all(identity["identity_checks"].values())


def test_module_runs_the_frozen_c2_construction_and_never_a_forbidden_initializer():
    source = MODULE_PATH.read_text()
    assert "build_any_state(bundle, stage_2u.C2_KEY)" in source
    assert "prohibited_alternatives_not_used" in source
    assert source.count("UNIFORM_ALL_DOMAIN") == 1
    assert source.count("STREAMFUNCTION_LOCALIZED_V0") == 1
    assert "0.354" not in source
    assert "t_contact" not in source


def test_stage_declares_its_own_sixteen_stages_and_the_required_evidence_set():
    assert stage.STAGES == (
        "preflight",
        "sealed",
        "identity",
        "design",
        "reader",
        "equivalence",
        "reaudit",
        "window",
        "static",
        "attribution",
        "closure",
        "validity",
        "decision",
        "adversarial",
        "quality",
        "assemble",
    )
    assert len(stage.EVIDENCE_FILES) == 23
    assert len(set(stage.EVIDENCE_FILES)) == 23
    assert tuple(stage.EVIDENCE_FILES).count("manifest.json") == 1
    assert tuple(stage.EVIDENCE_FILES).count("l1a2y_final_report.md") == 1
    assert stage.plan_stages("design") == ["preflight", "sealed", "identity", "design"]
    assert stage.plan_stages("validity") == [
        "preflight",
        "window",
        "static",
        "attribution",
        "closure",
        "validity",
    ]
    with pytest.raises(stage.AuditBlocked):
        stage.plan_stages("nonsense")


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
        "L1A-2x": 17,
    }
    assert all(entry["status"] == "VERIFIED" for entry in report["stages"].values())
    assert all(entry["problems"] == [] for entry in report["stages"].values())
    assert all(entry["source_hashes_unverifiable"] == [] for entry in report["stages"].values())
    assert sum(entry["source_hashes_recorded"] for entry in report["stages"].values()) >= 26


def test_seal_reader_supports_every_frozen_manifest_convention():
    seal_reader = stage.stage_2x._recorded_evidence_seal
    sha = "a" * 64
    assert seal_reader({"files": {"x.json": sha}}) == {"x.json": sha}
    assert seal_reader({"evidence_files": {"x.json": {"sha256": sha}}}) == {"x.json": sha}
    assert seal_reader({"evidence_files": {"x.json": sha}}) == {"x.json": sha}
    assert seal_reader({"files": {}, "evidence_files": {}}) == {}
    assert seal_reader({}) == {}


def test_an_empty_seal_is_never_counted_as_verified(tmp_path):
    target = _copy_evidence(tmp_path)
    manifest = json.loads((target / "manifest.json").read_text())
    del manifest["files"]
    (target / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(stage.AuditBlocked):
        stage.verify_evidence_manifest(target)


def test_an_unverifiable_recorded_source_hash_fails_closed():
    report = stage.stage_2x._verify_recorded_sources({"ghost_module": "0" * 64})
    assert report["checked"] == 1
    assert report["unverifiable"] == ["ghost_module"]
    assert report["problems"]
    assert stage.stage_2x._verify_recorded_sources(None)["problems"] == []


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


def test_preflight_refuses_a_worktree_dirty_outside_the_stage_deliverables(monkeypatch):
    real_git = stage._git

    def fake_git(args, *, required=True):
        if args[:3] == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return stage.BRANCH
        if args[:2] == ["status", "--porcelain"]:
            return " M examples/two_phase/phasefield.py"
        if args[:3] == ["rev-parse", "--is-shallow-repository"]:
            return "false"
        return real_git(args, required=required)

    monkeypatch.setattr(stage, "_git", fake_git)
    with pytest.raises(stage.AuditBlocked):
        stage.preflight_state()


def test_prerequisite_verification_fails_closed_on_a_tampered_or_missing_bound_file(monkeypatch, tmp_path):
    copied = tmp_path / "evidence" / "l1a2x"
    shutil.copytree(TWO_PHASE / "evidence" / "l1a2x", copied)
    monkeypatch.setattr(stage, "TWO_PHASE", tmp_path)
    monkeypatch.setattr(stage, "FROZEN_STAGES", (("L1A-2x", "l1a2x"),))
    assert stage.verify_prerequisites()["stages"]["L1A-2x"]["status"] == "VERIFIED"
    (copied / "mechanism_decision.json").write_text("{}\n")
    with pytest.raises(stage.AuditBlocked):
        stage.verify_prerequisites()
    shutil.copyfile(TWO_PHASE / "evidence" / "l1a2x" / "mechanism_decision.json", copied / "mechanism_decision.json")
    (copied / "validity_screen.json").unlink()
    with pytest.raises(stage.AuditBlocked):
        stage.verify_prerequisites()


def test_prerequisite_verification_fails_closed_on_an_unverifiable_source_hash(monkeypatch, tmp_path):
    copied = tmp_path / "evidence" / "l1a2x"
    shutil.copytree(TWO_PHASE / "evidence" / "l1a2x", copied)
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_hashes"] = {"ghost_module": "0" * 64}
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(stage, "TWO_PHASE", tmp_path)
    monkeypatch.setattr(stage, "FROZEN_STAGES", (("L1A-2x", "l1a2x"),))
    with pytest.raises(stage.AuditBlocked):
        stage.verify_prerequisites()


def test_recorded_source_hashes_are_verified_by_content_including_history():
    report = stage.stage_2x._verify_recorded_sources({"phasefield": stage.sha256_file(TWO_PHASE / "phasefield.py")})
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
    assert report["recorded_source_hashes"] == 4
    assert report["recorded_source_hashes_verified_from_the_worktree"] == 4
    assert report["verdict"] in stage.DECISION_STATES


def test_evidence_verification_detects_a_tampered_bound_file(tmp_path):
    target = _copy_evidence(tmp_path)
    assert stage.verify_evidence_manifest(target)["verified"] is True
    payload = json.loads((target / "residual_decomposition.json").read_text())
    payload["terms_dx"]["R_attribution_closure"] = 0.0
    (target / "residual_decomposition.json").write_text(json.dumps(payload))
    with pytest.raises(stage.AuditBlocked):
        stage.verify_evidence_manifest(target)


def test_evidence_verification_rejects_a_missing_required_artifact(tmp_path):
    target = _copy_evidence(tmp_path)
    (target / "static_reconstruction_study.json").unlink()
    with pytest.raises(stage.AuditBlocked):
        stage.verify_evidence_manifest(target)


def test_evidence_verification_rejects_a_moved_verdict_or_contract(tmp_path):
    target = _copy_evidence(tmp_path)
    manifest = json.loads((target / "manifest.json").read_text())
    manifest["verdict"] = "PRODUCTION_READY_CONFIRMED_CONTACT"
    (target / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(stage.AuditBlocked):
        stage.verify_evidence_manifest(target)
    manifest["verdict"] = "EARLY_WINDOW_ATTRIBUTION_UNRESOLVED"
    manifest["solver_contract_version"] = 13
    (target / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(stage.AuditBlocked):
        stage.verify_evidence_manifest(target)


def test_reseal_refuses_unobserved_inputs(tmp_path):
    target = _copy_evidence(tmp_path)
    for status, head in (("green", "a" * 40), ("success", "short"), ("success", "z" * 40)):
        with pytest.raises(stage.AuditBlocked):
            stage.reseal_ci_status(status, head, target)


def test_reseal_records_only_the_observed_status_and_touches_no_physics(tmp_path):
    target = _copy_evidence(tmp_path)
    before = {
        name: (target / name).read_bytes()
        for name in (
            "substep_ledger.json",
            "interface_attribution.json",
            "residual_decomposition.json",
            "static_reconstruction_study.json",
            "mechanism_decision.json",
            "validity_screen.json",
        )
    }
    report = stage.reseal_ci_status("success", "c" * 40, target)
    assert report["verified"] is True
    assert report["hosted_ci_reseal"] == {
        "performed": True,
        "status": "success",
        "observed_head_sha": "c" * 40,
        "physics_evidence_untouched": True,
    }
    quality = json.loads((target / "quality_and_blocker_matrix.json").read_text())
    assert quality["hosted_ci"]["status_for_this_head"] == "success"
    assert quality["hosted_ci"]["observed_head_sha"] == "c" * 40
    assert "success" in (target / "l1a2y_final_report.md").read_text()
    for name, payload in before.items():
        assert (target / name).read_bytes() == payload
    assert (target / "manifest.json").read_bytes() != (EVIDENCE / "manifest.json").read_bytes()
    assert stage.verify_evidence_manifest(EVIDENCE)["verified"] is True


def test_manifest_binds_the_module_and_never_claims_an_unpushed_ci_result(evidence):
    manifest = evidence["manifest.json"]
    assert manifest["stage"] == stage.STAGE
    assert manifest["stage_version"] == stage.STAGE_VERSION
    assert manifest["solver_contract_version"] == 12
    assert manifest["no_promotion"] is True
    assert manifest["no_hosted_ci_claim_before_push"] is True
    assert manifest["required_files"] == list(stage.EVIDENCE_FILES)
    assert manifest["module_sha256"] == stage.sha256_file(MODULE_PATH)
    assert manifest["readout_module_sha256"] == stage.sha256_file(READOUT_PATH)
    assert manifest["readout_version"] == stage.readout.READOUT_VERSION
    assert manifest["readout_fingerprint_sha256"] == evidence["readout_design.json"]["fingerprint_sha256"]
    assert set(manifest["files"]) == set(stage.EVIDENCE_FILES) - {"manifest.json"}


# ---------------------------------------------------------------------------
# phase 1: the prespecified readout design
# ---------------------------------------------------------------------------
def test_readout_design_is_frozen_with_the_prespecified_rules(evidence):
    design = evidence["readout_design.json"]
    assert design["readout_version"] == "contour_branch_v1"
    assert design["fingerprint_sha256"] == "95e42ac169016fd46ea868085795a9f5583a7f65ec9bbcca41700b723b34e232"
    rules = design["rules"]
    assert set(rules) == {
        "branch_definition",
        "continuity",
        "root_matching",
        "ambiguity",
        "failure",
        "no_favourable_selection",
        "cahn_hilliard_pairing",
        "why_branch_following",
    }
    assert "0.25 dx" in rules["continuity"]
    assert "+/- 1.5 dx" in rules["root_matching"]
    assert "0.5 dx" in rules["ambiguity"]
    assert "0.05 dx" in rules["failure"]
    assert "pinned at t = 0" in rules["branch_definition"]
    assert "never resolved by preferring a candidate" in rules["failure"]
    assert design["verdict_rule"]["dominance_factor"] == stage.DOMINANCE_FACTOR
    assert design["verdict_rule"]["attribution_closure_band_dx"] == stage.ATTRIBUTION_CLOSURE_BAND_DX
    assert "No term is normalised" in design["verdict_rule"]["statement"]
    level = design["t0_pinned_point_phi_level"]
    assert 0.0 < level < 1.0
    assert abs(level - 0.5) <= stage.readout.PROBE_BAND


def test_the_pinned_branch_is_the_frozen_support_column_and_crossing(evidence):
    design = evidence["readout_design.json"]
    pinned = design["pinned_branch"]
    assert pinned["column_index"] == 95
    assert pinned["column_x_over_dx"] == pytest.approx(95.5, abs=1e-12)
    assert pinned["t0_root_over_dx"] == pytest.approx(23.50598143513463, abs=1e-12)
    assert pinned["t0_root_segment_index"] == 23
    assert pinned["wall_normal_into_solid"] == [0.0, -1.0]
    assert pinned["t0_root_at_shared_node"] is False
    assert design["readout_version"] == "contour_branch_v1"
    model = design["pinned_branch_model"]
    assert "tanh" in model["field"]
    assert "dy = 0.1 dx" in model["field"]
    assert model["matched_t0_root_over_dx"] == pytest.approx(design["pinned_branch"]["t0_root_over_dx"])
    assert len(model["crossings_at_t0_over_dx"]) == 3
    assert design["t0_readout"]["matched"]["s_over_dx"] == pytest.approx(pinned["t0_root_over_dx"], abs=1e-12)
    assert design["t0_readout"]["status"] == "OK"
    assert design["t0_readout"]["arc"]["status"] == "OPEN_PATH"


def test_synthetic_validation_holds_on_every_prespecified_case(evidence):
    synthetic = evidence["synthetic_reader_validation.json"]
    assert synthetic["readout_version"] == "contour_branch_v1"
    assert synthetic["all_hold"] is True
    assert synthetic["checks_pass"] is True
    assert set(synthetic["cases"]) == {
        "branch_disappearance_is_a_failure",
        "column_root_is_a_frozen_segment_crossing",
        "continuity_across_a_shared_grid_node",
        "double_roots_are_refused_or_separated",
        "linear_translation_is_recovered",
        "no_favourable_selection",
        "reconstruction_bias_is_measured_not_assumed",
        "saddle_cells_are_detected",
        "symmetric_contour_mirrors_exactly",
    }
    assert all(bool(case["holds"]) for case in synthetic["cases"].values())


def test_the_shared_grid_node_case_is_continuous_and_reports_one_root(evidence):
    case = evidence["synthetic_reader_validation.json"]["cases"]["continuity_across_a_shared_grid_node"]
    assert case["support_switch_observed"] is True
    assert case["holds"] is True
    assert case["segment_indices"] == [15, 15, 16, 16, 16, 16]
    assert case["max_step_over_dx"] <= stage.readout.ROOT_MATCH_MAX_STEP_DX
    rows = case["rows"]
    assert all(row["status"] == "OK" for row in rows)
    node_rows = [row for row in rows if row["at_shared_node"]]
    assert len(node_rows) == 1
    assert node_rows[0]["crossing_over_dx"] == pytest.approx(node_rows[0]["matched_y_over_dx"], abs=1e-9)
    assert {row["segment_index"] for row in rows} == {15, 16}


def test_ambiguity_is_tested_before_the_continuity_bound(evidence):
    case = evidence["synthetic_reader_validation.json"]["cases"]["double_roots_are_refused_or_separated"]
    assert case["refuses_when_close"] is True
    assert case["separates_when_far"] is True
    assert case["close_status"] == "AMBIGUOUS_MULTIPLE_ROOTS"
    assert case["far_status"] == "OK"
    assert case["close_separation_over_dx"] < stage.readout.ROOT_AMBIGUITY_SEPARATION_DX
    assert case["far_separation_over_dx"] > stage.readout.ROOT_AMBIGUITY_SEPARATION_DX


def test_a_vanished_branch_is_a_failure_and_never_a_step(evidence):
    case = evidence["synthetic_reader_validation.json"]["cases"]["branch_disappearance_is_a_failure"]
    assert case["accepts_a_root_within_the_continuity_bound"] is True
    assert case["refuses_a_vanished_branch"] is True
    assert case["refuses_a_root_outside_the_search_window"] is True
    assert case["refuses_a_root_beyond_the_continuity_bound"] is True
    assert case["monotone_status"] == "NO_ROOT_IN_WINDOW"
    assert case["outside_status"] == "NO_ROOT_IN_WINDOW"
    assert case["jump_status"] == "BRANCH_STEP_TOO_LARGE"
    assert case["candidates_outside_the_window"] == 0


def test_the_reader_is_direction_blind_and_cannot_pick_a_favourable_root(evidence):
    cases = evidence["synthetic_reader_validation.json"]["cases"]
    favourable = cases["no_favourable_selection"]
    assert favourable["prefers_the_nearer_root"] is True
    assert favourable["nearer_root_moves_away_from_the_solid"] is False
    assert favourable["mirrored_wall_normal_gives_the_same_root"] is True
    assert favourable["matched_root_over_dx"] == pytest.approx(favourable["mirrored_root_over_dx"], abs=1e-12)
    assert favourable["candidate_roots_over_dx"][0] == pytest.approx(15.3, abs=1e-9)
    mirror = cases["symmetric_contour_mirrors_exactly"]
    assert mirror["root_on_the_axis"] is True
    assert mirror["mirror_identical"] is True
    assert mirror["arc_status"] == "OPEN_PATH"
    assert mirror["root_over_dx"] == pytest.approx(mirror["expected_root_over_dx"], abs=1e-9)


def test_the_reconstruction_bias_is_measured_not_assumed(evidence):
    case = evidence["synthetic_reader_validation.json"]["cases"]["reconstruction_bias_is_measured_not_assumed"]
    assert case["diffuseness_eps_over_dx"] == pytest.approx(1.5)
    assert case["max_abs_bias_over_dx"] > 0.0
    assert case["max_abs_bias_over_dx"] <= stage.readout.ROOT_MATCH_MAX_STEP_DX / 10.0
    assert case["bias_variation_over_the_sweep_over_dx"] > 0.0
    assert case["sweep_over_dx"] == pytest.approx(0.08, abs=1e-12)
    assert len(case["rows"]) == 41
    on_the_node = [row for row in case["rows"] if abs(row["applied_over_dx"] - 16.0) <= 1e-12]
    assert on_the_node and on_the_node[0]["bias_over_dx"] == 0.0


def test_the_linear_translation_and_the_saddle_case_hold(evidence):
    cases = evidence["synthetic_reader_validation.json"]["cases"]
    translation = cases["linear_translation_is_recovered"]
    assert translation["max_abs_error_over_dx"] <= stage.readout.ROOT_MATCH_MAX_STEP_DX / 10.0
    assert translation["max_walk_error_over_dx"] <= stage.readout.ROOT_MATCH_MAX_STEP_DX
    saddle = cases["saddle_cells_are_detected"]
    assert saddle["saddle"] is True
    assert saddle["crossing_count"] == 4
    assert saddle["segment_count"] == 2
    assert saddle["saddle_value"] == 1.0
    column = cases["column_root_is_a_frozen_segment_crossing"]
    assert column["segment_indices"] == [15]
    assert column["roots_in_window_over_dx"] == [16.368266046]
    assert column["applied_over_dx"] == pytest.approx(16.37, abs=1e-12)


def test_the_readout_reproduces_the_frozen_sampler_and_contour_on_the_t0_state(evidence):
    equivalence = evidence["frozen_reconstruction_equivalence.json"]
    assert equivalence["all_hold"] is True
    assert equivalence["checks"] == {
        "sampler_matches_the_frozen_sampler": True,
        "column_crossings_match_the_frozen_reconstruction": True,
        "pinned_root_is_the_frozen_crossing": True,
        "pinned_segment_is_the_one_l1a2x_tracked": True,
        "pinned_column_is_the_frozen_support_column": True,
    }
    assert equivalence["sampler_max_abs_difference"] <= 1e-12
    assert equivalence["frozen_contour"] == (
        "impact_initialization_compatibility_audit._threshold_contour_edge_crossings"
    )
    pinned = equivalence["pinned_branch"]
    design_pinned = evidence["readout_design.json"]["pinned_branch"]
    assert pinned == {key: design_pinned[key] for key in pinned}
    assert "only used if" in equivalence["rule"]
    assert equivalence["support_gap_over_dx_reference"] == pytest.approx(2.705981482818345, abs=1e-12)


# ---------------------------------------------------------------------------
# the re-audit of the L1A-2x comparison error
# ---------------------------------------------------------------------------
def test_the_raw_rule_margins_and_spread_of_l1a2x_are_recomputed_from_its_sealed_payload(evidence):
    reaudit = evidence["comparison_error_reaudit.json"]
    assert reaudit["raw_margins_dx"] == {
        "POINT_LOCAL_COEFFICIENT": pytest.approx(0.06289765049975118, abs=1e-15),
        "PRIMARY_POINT_T0_COEFFICIENT": pytest.approx(0.06442061040557828, abs=1e-15),
        "SUPPORT_CELL_T0_COEFFICIENT": pytest.approx(0.06423917493446041, abs=1e-15),
    }
    assert reaudit["raw_margin_spread_dx"] == pytest.approx(0.0015229599058270982, abs=1e-15)
    sealed = json.loads((TWO_PHASE / "evidence" / "l1a2x" / "interface_attribution.json").read_text())
    sealed_margins = {name: entry["margin_dx"] for name, entry in sealed["family"].items()}
    recomputed = max(
        abs(float(value) - float(sealed_margins["PRIMARY_POINT_T0_COEFFICIENT"]))
        for name, value in sealed_margins.items()
        if name != "PRIMARY_POINT_T0_COEFFICIENT"
    )
    assert recomputed == pytest.approx(reaudit["raw_margin_spread_dx"], abs=1e-15)


def test_the_reaudit_does_not_normalise_the_spread(evidence):
    reaudit = evidence["comparison_error_reaudit.json"]
    stages = {entry["rule"]: entry for entry in reaudit["per_rule"]}
    assert set(stages) == {
        "POINT_LOCAL_COEFFICIENT",
        "PRIMARY_POINT_T0_COEFFICIENT",
        "SUPPORT_CELL_T0_COEFFICIENT",
    }
    for entry in reaudit["per_rule"]:
        assert entry["margin_dx"] == pytest.approx(entry["full_displacement_dx"], abs=1e-12)
    recorded = reaudit["how_l1a2x_reported_it"]["recorded_rescaling"]
    assert {entry["rule"] for entry in recorded} == {"POINT_LOCAL_COEFFICIENT", "SUPPORT_CELL_T0_COEFFICIENT"}
    assert all(entry["rescaling_to_the_primary_full_displacement"] < 1.0 for entry in recorded)
    assert reaudit["raw_margin_spread_dx"] > 1e-6
    assert reaudit["how_l1a2x_reported_it"]["recorded_margin_spread_across_rules_dx"] < 1e-15


def test_the_rescaling_criticism_is_recorded_with_the_arithmetic_note(evidence):
    how = evidence["comparison_error_reaudit.json"]["how_l1a2x_reported_it"]
    assert "rescaled" in how["criticism"]
    assert "not a correct quotient" in how["arithmetic_note"]
    assert "recomputed here" in how["arithmetic_note"]
    assert "sealed" in evidence["comparison_error_reaudit.json"]["source"]
    assert "unedited" in evidence["comparison_error_reaudit.json"]["source"]


def test_the_reconstruction_nonlinearity_and_rule_terms_are_separated(evidence):
    reaudit = evidence["comparison_error_reaudit.json"]
    spread = reaudit["what_the_spread_is_made_of"]
    assert spread["coefficient_leg_over_dx"] == pytest.approx(-0.0001814354711178745, abs=1e-15)
    assert spread["sampling_point_leg_over_dx"] == pytest.approx(-0.0015229599058270982, abs=1e-15)
    residuals = reaudit["l1a2x_measured_reconstruction_residuals_dx"]
    for key in (
        "interpolation",
        "support_migration",
        "nonlinearity",
        "operator_splitting",
        "linearisation_coefficient_drift",
        "common_window_total_band_dx",
    ):
        assert key in residuals
    assert residuals["interpolation"] < residuals["nonlinearity"]
    assert residuals["support_migration"] < residuals["nonlinearity"]
    assert residuals["operator_splitting"] < 1e-14
    tracked = residuals["tracked_edge_coverage"]
    assert tracked["first_macro_step"] == 1
    assert tracked["last_macro_step"] == 24
    assert tracked["first_macro_step_without_it"] == 25
    assert tracked["macros_total"] == 40
    assert tracked["macros_available"] == 24


def test_the_stage_comparison_error_is_raw_and_made_of_measured_terms(evidence):
    rule = evidence["comparison_error_reaudit.json"]["prespecified_comparison_error_of_this_stage"]
    assert rule["dominance_factor"] == stage.DOMINANCE_FACTOR
    assert "max(raw rule spread, measured walk noise floor" in rule["rule"]
    validity = evidence["validity_screen.json"]
    terms = validity["comparison_error_terms_dx"]
    assert set(terms) == {"prespecified_floor", "rule_spread", "walk_noise"}
    assert validity["comparison_error_dx"] == pytest.approx(max(float(value) for value in terms.values()), abs=1e-15)
    assert float(terms["rule_spread"]) > float(terms["prespecified_floor"])
    assert validity["dominance_floor_dx"] == pytest.approx(
        stage.DOMINANCE_FACTOR * validity["comparison_error_dx"], rel=1e-12
    )


# ---------------------------------------------------------------------------
# the window, its ledger and the frozen identities
# ---------------------------------------------------------------------------
def test_window_stays_inside_the_early_response_and_never_reaches_contact(ledger):
    assert ledger["case_name"] == "pillar_training"
    assert ledger["profile_N"] == stage.PROFILE_N
    assert ledger["requested_dt"] == stage.REQUESTED_DT
    assert ledger["macro_steps_run"] == 40
    assert ledger["macro_steps_requested"] == 40
    assert ledger["substeps_per_macro"] == stage.SUBSTEPS_PER_MACRO
    assert ledger["substeps_recorded"] == 120
    assert ledger["t_final"] == pytest.approx(40 * stage.REQUESTED_DT, abs=1e-12)
    assert ledger["contact_threshold_over_dx"] == stage.CONTACT_GAP_THRESHOLD_DX
    assert ledger["contact_guard"] == {"triggered": False, "macro_step": None, "gap_over_dx": None}
    macro_rows = ledger["macro_rows"]
    assert len(macro_rows) == 40
    assert min(row["gap_global_over_dx"] for row in macro_rows) > stage.CONTACT_GAP_THRESHOLD_DX
    assert [row["macro_step"] for row in macro_rows] == list(range(1, 41))


def test_run_budget_is_capped_and_refuses_an_over_cap_request():
    with pytest.raises(stage.AuditBlocked):
        stage.run_window(stage.MAX_MACRO_STEPS + 1)
    with pytest.raises(stage.AuditBlocked):
        stage.run_window(0)


def test_initial_state_identity_matches_the_sealed_hashes_of_the_upstream_stages(evidence):
    identity = evidence["initial_state_identity.json"]
    assert identity["hashes"] == identity["sealed_hashes"]
    assert identity["hashes"] == identity["sealed_l1a2w_candidate_hashes"]
    assert identity["hashes"]["phi"] == "3022be21ea6a46ebe2e74b3a4903ae96b8e4527cbafadf1ba0ce0f0f7c14245c"
    assert identity["hashes"]["u"] == "2b921fbcd185c4fe36173e804b91ddefb4840c580a3410548948f27057c455e9"
    assert identity["hashes"]["v"] == "3c0963be8ef830f47db9b613ee3a7da64e1fd8694a04cd72be2d95d31a7adf37"
    assert identity["support"]["cell"] == [95, 23]
    assert identity["support"]["wall_normal_into_solid"] == [0.0, -1.0]
    assert identity["support"]["gap_over_dx"] == pytest.approx(2.705981482818345, abs=1e-12)
    assert identity["parameters"]["substep_dt"] == pytest.approx(stage.REQUESTED_DT / 3, abs=1e-15)


def test_initial_state_identity_is_reproducible_from_the_frozen_construction(evidence):
    rebuilt = stage._initial_state_identity()
    sealed = evidence["initial_state_identity.json"]
    assert rebuilt["hashes"] == sealed["hashes"]
    assert rebuilt["parameter_hash"] == sealed["parameter_hash"]
    assert rebuilt["geometry_hash"] == sealed["geometry_hash"]
    assert rebuilt["support"] == sealed["support"]
    assert all(rebuilt["identity_checks"].values())


def test_every_substep_recombines_its_three_recorded_increments(ledger):
    rows = ledger["rows"]
    assert len(rows) == 120
    assert [row["substep"] for row in rows[:3]] == [0, 1, 2]
    assert max(row["recombination_linf"] for row in rows) <= 1e-15
    for row in rows:
        assert row["root_ingredient_statuses"] == {"adv": "OK", "ch": "OK", "chpair": "OK", "imp": "OK"}
        assert row["root_full_status"] == "OK"
        assert row["arc_status"] == "OPEN_PATH"
        assert row["tracked_segment_index"] in (22, 23)
    assert max(row["arc_segments_traced"] for row in rows) < stage.readout.ARC_MAX_SEGMENTS


def test_the_recorded_increments_telescope_to_the_reported_cumulative_values(evidence, ledger):
    attribution = evidence["interface_attribution.json"]
    cumulative = attribution["cumulative_dn_dx"]
    for key, report_key in (
        ("adv", "dn_adv"),
        ("ch_explicit", "dn_ch_explicit"),
        ("ch_implicit", "dn_ch_implicit"),
        ("ch_pair", "dn_ch_pair"),
        ("full", "dn_full"),
    ):
        total = sum(float(row[report_key]) for row in attribution["per_macro"])
        assert total == pytest.approx(float(cumulative[key]), abs=1e-12)
    assert len(attribution["per_macro"]) == 40
    last = attribution["per_macro"][-1]
    assert last["root_full_over_dx"] == pytest.approx(ledger["macro_rows"][-1]["root_full_over_dx"], abs=1e-12)
    assert last["root_status"] == "OK"
    assert last["tracked_segment_index"] == 22


def test_mass_and_impermeable_walls_hold_over_every_substep(evidence):
    conservation = evidence["conservation_and_convergence.json"]
    assert conservation["mass"]["mass_drift_absolute"] == 0.0
    assert conservation["mass"]["mass_drift_relative"] == 0.0
    assert conservation["mass"]["max_abs_substep_drift"] <= 1e-15
    assert conservation["mass"]["max_abs_telescoping_adv"] <= 1e-16
    assert conservation["mass"]["max_abs_telescoping_ch"] <= 1e-16
    assert conservation["wall"]["max_abs_flux_on_closed_faces"] == 0.0
    assert "exactly zero" in conservation["wall"]["rule"]
    assert conservation["recombination"]["max_recombination_linf"] <= 1e-15
    rows = evidence["substep_ledger.json"]["rows"]
    assert max(row["closed_face_flux_max"] for row in rows) == 0.0
    assert max(row["mass_residual"] for row in rows) <= 1e-15


def test_every_implicit_solve_converged_under_the_frozen_tolerance(evidence):
    implicit = evidence["conservation_and_convergence.json"]["implicit_solve"]
    assert implicit["substeps"] == 120
    assert implicit["all_converged"] is True
    assert implicit["rtol"] == 1e-6
    assert implicit["max_relative_residual"] <= implicit["rtol"]
    assert implicit["max_iterations"] < implicit["iteration_cap"]
    rows = evidence["substep_ledger.json"]["rows"]
    assert all(row["cg_converged"] for row in rows)
    assert max(row["cg_relative_residual"] for row in rows) <= implicit["rtol"]


def test_the_capture_reproduces_the_frozen_public_step_within_the_thresholds(evidence):
    capture = evidence["capture_equivalence.json"]
    assert capture["macro_steps_compared"] == 40
    assert capture["max_phi_difference_vs_eager_public_step"] <= capture["thresholds"]["max_phi_difference"]
    assert capture["max_phi_difference_vs_jitted_public_step"] <= capture["thresholds"]["max_phi_difference"]
    assert capture["max_gap_difference_over_dx"] <= capture["thresholds"]["max_gap_difference_over_dx"]
    assert capture["max_velocity_difference_vs_eager_public_step"] <= 1e-6
    assert capture["thresholds"]["max_phi_difference"] == stage.EQUIVALENCE_MAX_PHI_DIFF
    assert capture["thresholds"]["max_gap_difference_over_dx"] == stage.EQUIVALENCE_MAX_GAP_DIFF_DX


# ---------------------------------------------------------------------------
# attribution, the static reconstruction and the residual decomposition
# ---------------------------------------------------------------------------
def test_the_ingredient_legs_and_the_primary_rule_are_recorded(evidence):
    attribution = evidence["interface_attribution.json"]
    assert attribution["primary_rule"] == "RULE_A_GEOMETRIC_TRACKED_BRANCH"
    assert attribution["rules"] == [
        "RULE_A_GEOMETRIC_TRACKED_BRANCH",
        "RULE_B_GEOMETRIC_OWN_BASE",
        "RULE_D_LINEARISED_POINT_T0_COEFFICIENT",
        "RULE_E_LINEARISED_SUPPORT_CELL_T0_COEFFICIENT",
    ]
    cumulative = attribution["cumulative_dn_dx"]
    assert set(cumulative) == {
        "adv",
        "ch_explicit",
        "ch_implicit",
        "ch_pair",
        "ch_pair_direct_over_dx",
        "full",
        "normal_ray_full_dx",
    }
    assert attribution["walk_noise_floor_dx"] >= 0.0
    assert set(attribution["rule_margins_dx"]) == set(attribution["rules"])
    assert set(attribution["comparison_terms_dx"]) == {
        "RULE_B_GEOMETRIC_OWN_BASE",
        "RULE_D_LINEARISED_POINT_T0_COEFFICIENT",
        "RULE_E_LINEARISED_SUPPORT_CELL_T0_COEFFICIENT",
    }


def test_the_cahn_hilliard_halves_are_reported_as_one_update_not_two_mechanisms(evidence):
    attribution = evidence["interface_attribution.json"]
    pairing = attribution["cahn_hilliard_pairing"]
    assert pairing["cancellation_ratio"] == 1.0
    assert pairing["pair_dx"] == pytest.approx(pairing["explicit_dx"] + pairing["implicit_dx"], abs=1e-12)
    assert pairing["pair_directly_measured_dx"] != pytest.approx(pairing["pair_dx"], abs=1e-3)
    assert "one Cahn-Hilliard update" in pairing["reported_as"]
    assert "never presented as two independent physical mechanisms" in pairing["forbidden_claim"]
    decision = evidence["mechanism_decision.json"]
    assert decision["cahn_hilliard_halves_are_one_update"] is True
    assert decision["cahn_hilliard_cancellation_ratio"] == 1.0
    assert decision["forbidden_claims_respected"]["no_separate_mechanism_for_the_two_cahn_hilliard_halves"] is True


def test_the_arithmetic_rules_reproduce_the_sealed_l1a2x_arithmetic(evidence):
    check = evidence["adversarial_and_falsification.json"]["checks"][
        "the_arithmetic_rules_reproduce_the_sealed_l1a2x_window"
    ]
    assert check["holds"] is True
    for table in check["max_differences_over_dx"].values():
        assert set(table) == {"adv", "ch", "imp", "full"}
        assert max(float(value) for value in table.values()) <= 1e-12
    sealed = json.loads((TWO_PHASE / "evidence" / "l1a2x" / "interface_attribution.json").read_text())
    assert evidence["interface_attribution.json"]["linearised_dn_dx"]["full"] == pytest.approx(
        sealed["family"]["PRIMARY_POINT_T0_COEFFICIENT"]["cumulative_dn_dx"]["full"], abs=1e-12
    )


def test_the_static_reconstruction_holds_and_is_not_a_rescaled_level_fraction(evidence):
    study = evidence["static_reconstruction_study.json"]
    assert study["holds"] is True
    assert study["macros_total"] == 40
    assert study["macros_with_a_resolved_support_cell"] == 40
    assert study["support_cell_resolved"] is True
    assert study["tolerance_over_dx"] == stage.STATIC_RECONSTRUCTION_TOLERANCE_DX
    assert study["max_static_reconstruction_forward_over_dx"] <= study["tolerance_over_dx"]
    assert study["max_static_reconstruction_backward_over_dx"] <= study["tolerance_over_dx"]
    assert study["max_operator_recombination_over_dx"] <= study["tolerance_over_dx"]
    assert study["max_order_symmetry_over_dx"] <= study["tolerance_over_dx"]
    assert study["max_half_increment_nonlinearity_over_dx"] <= study["tolerance_over_dx"]
    assert study["level_based_fraction_max_abs"] > 1.0
    assert study["level_based_fraction_inside_the_unit_interval"] < study["level_based_fraction_total"]
    assert "never rescales the increment" in study["level_based_fraction_note"]
    assert "not a usable interpolation weight" in study["level_based_fraction_note"]
    assert set(study["readout_statuses_seen"]) == {"OK"}
    assert len(study["anchors_over_dx"]) == 40
    assert "backward" in study["nonlinearity_attribution"]


def test_the_static_reconstruction_separates_every_order_term(evidence):
    rows = evidence["static_reconstruction_study.json"]["rows"]
    assert len(rows) == 40
    for row in rows:
        assert set(row["statuses"]) == {
            "base",
            "full",
            "ingredient_sum",
            "half",
            "adv",
            "ch",
            "imp",
            "without_adv",
            "without_ch",
            "without_imp",
        }
        assert all(status == "OK" for status in row["statuses"].values())
        assert row["R_static_forward_over_dx"] <= stage.STATIC_RECONSTRUCTION_TOLERANCE_DX
        assert row["R_static_backward_over_dx"] <= stage.STATIC_RECONSTRUCTION_TOLERANCE_DX
        assert row["R_operator_recombination_over_dx"] <= 1e-12
        assert len(row["increment_at_cells"]) == len(row["base_levels_at_cells"]) == 2
        assert "isolates the nonlinearity" in row["resolution"]


def test_the_residual_decomposition_separates_every_term_in_raw_dx(evidence):
    residuals = evidence["residual_decomposition.json"]
    terms = residuals["terms_dx"]
    assert set(terms) == {
        "R_attribution_closure",
        "R_attribution_closure_own_base",
        "R_ch_pair_closure",
        "R_ch_pair_closure_own_base",
        "R_nonlinearity_point_linearisation",
        "R_nonlinearity_support_cell_linearisation",
        "R_operator_recombination",
        "R_static_reconstruction_forward",
        "R_static_reconstruction_ch_pair",
        "R_order_symmetry",
        "R_half_increment_nonlinearity",
        "R_rule_spread",
        "R_walk_noise",
        "R_cross_ray_full_displacement",
        "R_operator_splitting",
    }
    attribution = evidence["interface_attribution.json"]
    cumulative = attribution["cumulative_dn_dx"]
    assert terms["R_attribution_closure"] == pytest.approx(
        float(cumulative["full"]) - float(cumulative["adv"]) - float(cumulative["ch_pair"]), abs=1e-12
    )
    assert terms["R_operator_splitting"] <= 1e-15
    assert terms["R_rule_spread"] == pytest.approx(attribution["raw_rule_spread_dx"], abs=1e-15)
    assert terms["R_walk_noise"] == pytest.approx(attribution["walk_noise_floor_dx"], abs=1e-15)
    assert "none is normalised" in residuals["definition"]
    assert "post_window_diagnosis" in residuals


def test_the_cross_ray_displacement_is_an_independent_ray_of_the_same_total(evidence):
    residuals = evidence["residual_decomposition.json"]
    attribution = evidence["interface_attribution.json"]
    cumulative = attribution["cumulative_dn_dx"]
    assert residuals["cross_ray_total_displacement_over_dx"] == pytest.approx(
        float(cumulative["normal_ray_full_dx"]), abs=1e-12
    )
    assert residuals["measured_total_displacement_over_dx"] == pytest.approx(float(cumulative["full"]), abs=1e-12)
    assert residuals["terms_dx"]["R_cross_ray_full_displacement"] == pytest.approx(
        float(cumulative["full"]) - float(cumulative["normal_ray_full_dx"]), abs=1e-12
    )
    assert residuals["linearised_total_displacement_over_dx"] == pytest.approx(
        float(attribution["linearised_dn_dx"]["full"]), abs=1e-12
    )
    caveat = residuals["post_window_diagnosis"]["reading_caveats"]["fixed_ray_vs_tracked_branch"]
    assert "fixed point" in caveat
    assert "never about the interface as a whole" in caveat


def test_the_post_window_diagnosis_is_exploratory_and_promotes_nothing(evidence):
    diagnosis = evidence["residual_decomposition.json"]["post_window_diagnosis"]
    assert diagnosis["status"] == "EXPLORATORY_DIAGNOSIS_NO_PROMOTION_RIGHTS"
    assert diagnosis["primary_rule"] == "RULE_A_GEOMETRIC_TRACKED_BRANCH"
    assert "telescopes" in diagnosis["why_the_primary_rule_fails"]
    assert set(diagnosis["numbers"]["own_base_legs_dx"]) == {"adv", "ch_pair", "complete_update"}
    assert set(diagnosis["coherent_families"]) == {
        "own_base",
        "point_linearised",
        "support_cell_linearised",
        "agreement",
    }
    assert "no promotion rights" in diagnosis["consequence"]
    assert diagnosis["coherent_families"]["own_base"].startswith("the per-substep displacement")
    structure = diagnosis["measured_structure_of_the_failure"]
    assert structure["closure_band_over_dx"] == stage.ATTRIBUTION_CLOSURE_BAND_DX
    assert structure["primary_rule_closure_over_dx"] > 10.0 * structure["closure_band_over_dx"]
    assert structure["own_base_rule_closure_over_dx"] <= structure["closure_band_over_dx"]
    assert "no promotion rights" in structure["note"]
    assert "dominated by the difference" in structure["comparison_error_note"]
    assert diagnosis["numbers"]["sealed_l1a2x_primary_rule_dx"]["full"] == pytest.approx(
        evidence["interface_attribution.json"]["linearised_dn_dx"]["full"], abs=1e-12
    )


# ---------------------------------------------------------------------------
# validity screen, decision and the single next action
# ---------------------------------------------------------------------------
def test_the_frozen_global_contour_match_is_non_unique_and_is_a_guard(evidence):
    uniqueness = evidence["validity_screen.json"]["macro_guards"]["frozen_global_contour_uniqueness"]
    assert uniqueness["holds"] is False
    assert uniqueness["macros_total"] == 40
    assert uniqueness["macros_non_unique"] == 40
    assert uniqueness["non_unique_macro_steps"] == list(range(1, 41))
    assert uniqueness["statuses_seen"] == ["NON_UNIQUE"]
    assert uniqueness["min_margin_over_dx"] > 0.0
    assert uniqueness["max_margin_over_dx"] < 1e-8
    assert uniqueness["second_candidate_distance_over_dx"] == [1.0]
    assert "non-unique root" in uniqueness["rule"]
    assert "reported not re-defined" in uniqueness["source"]
    rows = evidence["substep_ledger.json"]["macro_rows"]
    assert all(row["contour_uniqueness"]["status"] == "NON_UNIQUE" for row in rows)
    assert all(row["contour_uniqueness"]["candidates_within_uniqueness_tolerance"] >= 2 for row in rows)
    decision = evidence["mechanism_decision.json"]
    assert any("not unique" in reason for reason in decision["verdict_withheld_because"])
    diagnosis = evidence["residual_decomposition.json"]["post_window_diagnosis"]
    assert "non-unique on every macro step" in diagnosis["what_failed"]
    assert "not uniquely attached to the frozen global contour" in diagnosis["consequence"]


def test_the_tracked_branch_root_itself_is_unique_on_the_ray(evidence):
    design = evidence["readout_design.json"]
    assert design["t0_readout"]["candidates_over_dx"] == [design["pinned_branch"]["t0_root_over_dx"]]
    assert design["t0_readout"]["separation_over_dx"] is None
    statuses = {row["root_full_status"] for row in evidence["substep_ledger.json"]["rows"]}
    assert statuses == {"OK"}
    ingredient_statuses = {
        status for row in evidence["substep_ledger.json"]["rows"] for status in row["root_ingredient_statuses"].values()
    }
    assert ingredient_statuses == {"OK"}
    assert evidence["validity_screen.json"]["failures"]["ambiguity_free"] == 0
    assert evidence["validity_screen.json"]["substeps"] == 120


def test_only_the_attribution_closure_gate_fails_and_it_is_the_only_withheld_reason(evidence):
    validity = evidence["validity_screen.json"]
    assert validity["unresolved"] is True
    assert validity["attribution_closure_ok"] is False
    assert validity["linearisation_closure_ok"] is True
    assert validity["static_reconstruction_ok"] is True
    assert validity["equivalence_ok"] is True
    assert set(validity["failures"]) == {
        "branch_continuity",
        "ingredient_continuity",
        "step_within_continuity_bound",
        "gradient_resolved",
        "probe_band",
        "arc_open_path",
        "ambiguity_free",
    }
    assert all(count == 0 for count in validity["failures"].values())
    assert validity["blocking_failures"] == {"frozen_global_contour_uniqueness": 40}
    assert abs(float(evidence["residual_decomposition.json"]["terms_dx"]["R_attribution_closure"])) > (
        stage.ATTRIBUTION_CLOSURE_BAND_DX
    )
    decision = evidence["mechanism_decision.json"]
    assert decision["verdict_withheld_because"] == [
        "the attribution closure exceeds its band",
        "the frozen global contour match at the tracked support point is not unique on 40 of 40 macro steps",
    ]
    assert "no mechanism is claimed" in decision["reason"]
    assert decision["capture_equivalence_ok"] is True


def test_the_verdict_is_unresolved_and_every_dominance_claim_is_withheld(evidence):
    decision = evidence["mechanism_decision.json"]
    assert decision["verdict"] == "EARLY_WINDOW_ATTRIBUTION_UNRESOLVED"
    assert decision["verdict"] in stage.DECISION_STATES
    assert decision["dominance_margin_dx"] <= 0.0
    assert decision["dominance_floor_dx"] > 0.0
    assert decision["margin_over_comparison_error"] < 1.0
    assert decision["dominance_margin_dx"] == pytest.approx(
        float(decision["rule_margins_dx"]["RULE_A_GEOMETRIC_TRACKED_BRANCH"]), abs=1e-12
    )
    assert decision["dominance_floor_dx"] == pytest.approx(
        decision["dominance_factor"] * decision["comparison_error_dx"], rel=1e-12
    )


def test_the_decision_records_magnitudes_and_bands_without_a_claim(evidence):
    decision = evidence["mechanism_decision.json"]
    cumulative = evidence["interface_attribution.json"]["cumulative_dn_dx"]
    assert decision["advective_magnitude_dx"] == pytest.approx(abs(float(cumulative["adv"])), abs=1e-12)
    assert decision["net_chemical_hilliard_magnitude_dx"] == pytest.approx(abs(float(cumulative["ch_pair"])), abs=1e-12)
    assert decision["comparison_error_terms_dx"]
    assert all(value is True for value in decision["forbidden_claims_respected"].values())
    assert decision["scope"]["contact_threshold_reached"] is False
    assert decision["scope"]["contact_events_scored"] == 0
    assert decision["scope"]["window_limited_to_early_response"] is True
    assert decision["scope"]["macro_steps_run"] == 40
    assert decision["case_name"] == stage.CASE_NAME
    assert decision["segment_binding_switch"]["found"] is True
    assert decision["segment_binding_switch"]["macro_step"] == 25


def test_the_verdict_rule_withholds_on_a_near_tie_and_on_an_unresolved_guard(evidence):
    ledger = evidence["substep_ledger.json"]
    attribution = evidence["interface_attribution.json"]
    residuals = evidence["residual_decomposition.json"]
    study = evidence["static_reconstruction_study.json"]
    equivalence = evidence["capture_equivalence.json"]
    validity = evidence["validity_screen.json"]
    floor = float(validity["dominance_floor_dx"])

    def verdict(**updates):
        return stage.decide(
            ledger,
            attribution,
            residuals,
            study,
            stage._fabricated_validity(validity, **updates),
            equivalence,
        )["verdict"]

    assert verdict(blocking_failures={"branch_continuity": 1}, unresolved=True) == (
        "EARLY_WINDOW_ATTRIBUTION_UNRESOLVED"
    )
    assert (
        verdict(blocking_failures={"frozen_global_contour_uniqueness": 40}, unresolved=True, uniqueness_holds=False)
        == "EARLY_WINDOW_ATTRIBUTION_UNRESOLVED"
    )
    assert verdict(blocking_failures={}, unresolved=False, dominance_margin_dx=0.999 * floor) == (
        "MIXED_TERMS_WITHIN_THE_COMPARISON_ERROR"
    )
    assert verdict(blocking_failures={}, unresolved=False, dominance_margin_dx=floor) == (
        "ADVECTIVE_TERM_CARRIES_THE_EARLY_WINDOW_CONTOUR_MOTION"
    )
    assert verdict(blocking_failures={}, unresolved=False, dominance_margin_dx=-floor) == (
        "CAHN_HILLIARD_TERM_CARRIES_THE_EARLY_WINDOW_CONTOUR_MOTION"
    )


def test_the_single_next_action_is_bounded_and_deliberately_not_started(evidence):
    action = evidence["next_action.json"]
    assert "corrected primary decomposition rule" in action["action"]
    assert "the attribution closure exceeds its band" in action["action"]
    assert action["requires_owner_approval"] is True
    assert action["must_not_be_started_in_this_stage"] is True
    scope = action["bounded_scope"]
    assert scope["cases"] == []
    assert scope["dt"] == stage.REQUESTED_DT
    assert scope["macro_step_cap"] == stage.MAX_MACRO_STEPS
    assert scope["window"] == "t <= 0.08"
    assert scope["no_extension_toward_contact"] is True
    assert scope["no_control_arms"] is True
    assert scope["no_refinement"] is True
    assert scope["no_generator_or_ml"] is True
    assert evidence["mechanism_decision.json"]["next_action_is_not_started_in_this_stage"] is True


# ---------------------------------------------------------------------------
# the cross-check against the identical sealed L1A-2x window
# ---------------------------------------------------------------------------
def test_the_identical_window_cross_check_holds_and_identifies_the_l1a2x_stop(evidence):
    cross = evidence["adversarial_and_falsification.json"]["identical_window_cross_check"]
    assert cross["all_hold"] is True
    assert all(cross["checks"].values())
    assert set(cross["checks"]) == {
        "the_window_is_the_identical_one",
        "final_state_hashes_match_the_sealed_window",
        "the_support_change_of_l1a2x_is_identified_here",
        "the_support_change_is_reported_as_a_continuous_event",
        "root_agrees_where_l1a2x_could_measure_it",
        "this_stage_measures_the_window_l1a2x_could_not",
        "the_independent_line_root_agrees_with_the_tracked_root",
    }
    assert cross["first_lost_macro_step"] == 25
    assert cross["max_gap_difference_over_dx"] == 0.0
    assert cross["max_phi_at_point_difference"] == 0.0
    assert cross["macros_where_l1a2x_could_measure_its_pinned_edge"] == 24
    assert cross["macros_where_l1a2x_lost_its_pinned_edge"] == list(range(25, 41))
    assert cross["macros_measured_by_each_stage"] == {"l1a2x": 40, "this_stage": 40}
    assert cross["max_root_difference_over_dx"] <= stage.CROSS_CHECK_TOLERANCE_DX
    assert cross["max_line_root_difference_over_dx"] <= stage.CROSS_CHECK_TOLERANCE_DX
    assert cross["final_state_hash_matches"] == {"phi": True, "u": True, "v": True}
    assert cross["tolerance_over_dx"] == stage.CROSS_CHECK_TOLERANCE_DX
    assert cross["support_change_this_stage"] == evidence["substep_ledger.json"]["segment_binding_switch"]
    sealed = json.loads((TWO_PHASE / "evidence" / "l1a2x" / "initial_state_identity.json").read_text())
    assert evidence["substep_ledger.json"]["initial_state"]["phi_hash"] == sealed["hashes"]["phi"]


def test_the_support_change_is_reported_as_a_continuous_event(evidence, ledger):
    switch = ledger["segment_binding_switch"]
    assert switch["found"] is True
    assert switch["macro_step"] == 25
    assert switch["substep"] == 74
    assert switch["from_segment_index"] == 23
    assert switch["to_segment_index"] == 22
    assert "ordinary recorded event" in switch["note"]
    rows = ledger["macro_rows"]
    assert {row["macro_step"] for row in rows if row["tracked_segment_index"] == 22} == set(range(25, 41))
    assert {row["macro_step"] for row in rows if row["tracked_segment_index"] == 23} == set(range(1, 25))
    roots = [float(row["root_full_over_dx"]) for row in rows]
    max_step = max(abs(right - left) for left, right in zip(roots[:-1], roots[1:]))
    assert max_step <= stage.readout.ROOT_MATCH_MAX_STEP_DX
    turn = roots.index(max(roots))
    assert roots[turn] == max(roots)
    assert roots[turn:] == sorted(roots[turn:], reverse=True)
    assert roots[turn:] == sorted(roots[turn:], reverse=True)
    assert evidence["validity_screen.json"]["observed"]["segment_binding_switch"]["substep"] == 74
    assert evidence["validity_screen.json"]["observed"]["tracked_segment_indices"] == [22, 23]


def test_the_closure_gate_bites_on_a_fabricated_violation_and_reports_the_observed_closure(evidence):
    check = evidence["adversarial_and_falsification.json"]["checks"]["attribution_closure_gate_bites"]
    terms = evidence["residual_decomposition.json"]["terms_dx"]
    assert check["holds"] is True
    assert check["band_over_dx"] == stage.ATTRIBUTION_CLOSURE_BAND_DX
    assert check["gate_accepts_the_fabricated_inside_band_closure"] is True
    assert check["gate_rejects_the_fabricated_beyond_band_closure"] is True
    assert check["fabricated_closure_inside_the_band"] == pytest.approx(0.5 * check["band_over_dx"])
    assert check["fabricated_closure_beyond_the_band"] == pytest.approx(10.0 * check["band_over_dx"])
    assert check["observed_closure_within_band"] is False
    assert check["observed_closure_over_dx"] == pytest.approx(float(terms["R_attribution_closure"]), abs=1e-15)
    assert abs(float(check["observed_closure_over_dx"])) > check["band_over_dx"]
    assert check["observed_closure_over_band_ratio"] == pytest.approx(
        abs(float(terms["R_attribution_closure"])) / check["band_over_dx"], rel=1e-12
    )
    assert check["observed_own_base_closure_within_band"] is True
    assert check["observed_own_base_closure_over_dx"] == pytest.approx(
        float(terms["R_attribution_closure_own_base"]), abs=1e-15
    )
    assert "never forced inside the band" in check["rule"]


# ---------------------------------------------------------------------------
# adversarial suite, report and quality matrix
# ---------------------------------------------------------------------------
def test_adversarial_checks_all_hold_and_exercise_every_guard(evidence):
    adversarial = evidence["adversarial_and_falsification.json"]
    assert adversarial["all_hold"] is True
    assert adversarial["failed_checks"] == []
    checks = adversarial["checks"]
    assert set(checks) == {
        "reader_tracks_the_support_change_on_the_real_grid",
        "a_crossing_exactly_on_a_cell_centre_is_reported_once",
        "a_jump_beyond_the_continuity_bound_is_refused_on_the_real_grid",
        "reader_is_direction_blind_on_the_real_field",
        "the_reader_cannot_pick_a_favourable_root",
        "the_decision_rule_follows_its_prespecified_thresholds",
        "attribution_closure_gate_bites",
        "comparison_error_is_not_normalised",
        "l1a2x_spread_is_reproduced_from_its_sealed_payload",
        "the_arithmetic_rules_reproduce_the_sealed_l1a2x_window",
        "frozen_sources_and_the_readout_are_unmodified",
        "the_sealed_stages_are_untouched_on_disk",
        "the_window_budget_is_capped",
        "the_ledger_is_not_vacuous",
        "identical_window_cross_check_holds",
        "the_static_reconstruction_gate_bites_when_it_is_violated",
        "the_attribution_is_a_pure_function_of_the_recorded_ledger",
        "the_withheld_verdict_cannot_be_flipped_by_choosing_a_nicer_rule",
        "no_forbidden_claim_in_the_reported_payload",
    }
    walk = checks["reader_tracks_the_support_change_on_the_real_grid"]
    assert walk["segment_indices"][0] == 23 and walk["segment_indices"][-1] == 24
    assert len(walk["segment_switches_at"]) == 1
    assert walk["max_step_over_dx"] <= stage.readout.ROOT_MATCH_MAX_STEP_DX
    assert checks["a_jump_beyond_the_continuity_bound_is_refused_on_the_real_grid"]["status"] == (
        "BRANCH_STEP_TOO_LARGE"
    )
    blind = checks["reader_is_direction_blind_on_the_real_field"]
    assert blind["displacement_forward_over_dx"] == pytest.approx(-blind["displacement_mirrored_over_dx"], abs=1e-15)
    thresholds = checks["the_decision_rule_follows_its_prespecified_thresholds"]["fabricated_screens"]
    assert thresholds == {
        "blocking_guard_failed": "EARLY_WINDOW_ATTRIBUTION_UNRESOLVED",
        "non_unique_global_contour": "EARLY_WINDOW_ATTRIBUTION_UNRESOLVED",
        "margin_below_the_floor": "MIXED_TERMS_WITHIN_THE_COMPARISON_ERROR",
        "margin_at_the_floor": "ADVECTIVE_TERM_CARRIES_THE_EARLY_WINDOW_CONTOUR_MOTION",
        "margin_at_minus_the_floor": "CAHN_HILLIARD_TERM_CARRIES_THE_EARLY_WINDOW_CONTOUR_MOTION",
    }
    assert checks["the_window_budget_is_capped"]["refused_a_request_above_the_cap"] is True
    assert checks["the_ledger_is_not_vacuous"]["substeps"] == 120
    assert checks["no_forbidden_claim_in_the_reported_payload"]["found"] == []
    assert checks["the_sealed_stages_are_untouched_on_disk"]["holds"] is True
    assert checks["frozen_sources_and_the_readout_are_unmodified"]["holds"] is True
    assert checks["comparison_error_is_not_normalised"]["holds"] is True
    static_gate = checks["the_static_reconstruction_gate_bites_when_it_is_violated"]
    assert static_gate["observed_holds"] is True
    assert static_gate["gate_rejects_the_fabricated_study"] is True
    assert checks["the_attribution_is_a_pure_function_of_the_recorded_ledger"]["holds"] is True
    nicer = checks["the_withheld_verdict_cannot_be_flipped_by_choosing_a_nicer_rule"]
    assert nicer["fabricated_primary_rule"] == "RULE_B_GEOMETRIC_OWN_BASE"
    assert nicer["resulting_verdict"] == "EARLY_WINDOW_ATTRIBUTION_UNRESOLVED"
    assert nicer["resulting_withheld_reasons"]
    assert checks["l1a2x_spread_is_reproduced_from_its_sealed_payload"]["holds"] is True


def test_report_carries_every_required_section_and_the_headline(evidence):
    report = evidence["report.md"]
    sections = evidence["report.json"]["sections"]
    assert sections == [
        "## A. Scope, authority and baseline",
        "## B. Phase 1: the prespecified readout",
        "## C. Re-audit of the L1A-2x comparison error (the old evidence is untouched)",
        "## D. Verification of the sealed prerequisite evidence",
        "## E. The window and its substep ledger",
        "## F. Ingredient closure, conservation and the implicit solve",
        "## G. Reconstruction-stable attribution",
        "## H. Static component reconstruction and the order decomposition",
        "## I. Residual decomposition (raw dx, nothing normalised)",
        "## J. Validity screen",
        "## K. Decision",
        "## L. Cross-check of the identical window",
        "## M. Adversarial and falsification checks",
        "## N. Quality, focused tests and hosted CI",
        "## O. Post-window diagnosis of the failed gates (exploratory, no promotion rights)",
        "## P. Exactly one next action",
    ]
    for section in sections:
        assert section in report
    assert evidence["report.json"]["verdict"] == "EARLY_WINDOW_ATTRIBUTION_UNRESOLVED"
    assert evidence["report.json"]["stage"] == stage.STAGE
    assert (
        evidence["report.json"]["readout_fingerprint_sha256"] == (evidence["readout_design.json"]["fingerprint_sha256"])
    )
    headline = evidence["report.json"]["headline"]
    assert "EARLY_WINDOW_ATTRIBUTION_UNRESOLVED" in headline
    assert "40" in headline
    assert evidence["report.json"]["reference_times"][0] == pytest.approx(0.002, abs=1e-12)


def test_no_evidence_or_report_claims_an_impact_or_a_contact_mechanism(evidence):
    decision = evidence["mechanism_decision.json"]
    assert all(value is True for value in decision["forbidden_claims_respected"].values())
    scanned = json.dumps(
        {
            "decision": decision,
            "study": evidence["static_reconstruction_study.json"],
            "attribution": evidence["interface_attribution.json"],
        }
    )
    assert [phrase for phrase in FORBIDDEN_PHRASES if phrase in scanned] == []
    report = evidence["report.md"]
    assert "it is not a real impact" in report
    assert "contact events scored 0" in report
    assert "contact_threshold_reached" in report
    assert "no mechanism is claimed" in report
    assert "is not a real impact" in report


def test_quality_matrix_keeps_the_structural_and_the_lineage_failures_separate(evidence):
    quality = evidence["quality_and_blocker_matrix.json"]
    assert quality["declared_new_code_diagnostics"] == 0
    assert quality["ratchet_expectation"] == "PASS_BASELINE_ONLY"
    assert quality["legacy_debt_baseline"] == {"ruff": 171, "formatter": 27, "isort": 28, "codespell": 3}
    assert quality["quality_baseline_path"] == "quality/legacy_two_phase_debt_v1.json"
    assert quality["stage_test_file"] == "tests/test_frozen_reconstruction_branch_attribution_audit.py"
    structural = quality["frozen_adjacent_suite"]
    lineage = quality["legacy_lineage_suite"]
    assert structural["classification"] == "SUPERSEDED_BY_THE_NEXT_STAGE_BY_CONSTRUCTION"
    assert structural["suite_failures"] == {"l1a2t": 1, "l1a2u": 1, "l1a2v": 5, "l1a2w": 1, "l1a2x": 0}
    assert structural["suite_failure_count"] == 8
    assert len(structural["failing"]) == 8
    assert all("::" in name for name in structural["failing"])
    assert structural["sweep_result"] == "8 failed, 221 passed"
    assert "test_frozen_chns_early_window_transport_attribution_audit.py" in structural["sweep_command"]
    assert "test_impact_initialization_compatibility_audit.py" not in structural["sweep_command"]
    assert not any("test_impact_initialization" in name for name in structural["failing"])
    assert not any("test_frozen_chns" in name for name in structural["failing"])
    assert any("test_radial_incident" in name for name in structural["failing"])
    assert evidence["preflight.json"]["head_sha"] in structural["measured_at"]
    assert "present in the worktree but uncommitted" in structural["measured_at"]
    assert structural["measured_at"].startswith("head ")
    combined = structural["combined_sweep_observation"]
    assert combined["result"] == "9 failed, 269 passed"
    assert "test_impact_initialization_compatibility_audit.py" in combined["command"]
    assert "test_frozen_chns_early_window_transport_attribution_audit.py" in combined["command"]
    assert "one invocation of the combined command returned 9 failed / 269 passed" in combined["derivation"]
    assert "dirty outside its own paths" in structural["state_dependence"]
    assert "the state the stage ran in" in structural["state_dependence"]
    assert "structural" in structural["cause"]
    assert "recorded as observed rather than repaired" in structural["cause"]
    assert lineage["classification"] == "FROZEN_LINEAGE_DRIFT_CONFIRMED"
    assert "test_impact_initialization_compatibility_audit.py" in lineage["sweep_command"]
    assert lineage["sweep_command"].count("tests/") == 1
    assert lineage["sweep_result"] == "1 failed, 48 passed"
    assert lineage["failing"] == [
        "test_impact_initialization_compatibility_audit.py::test_l1a2r_merged_base_verified_or_fail_closed"
    ]
    assert lineage["count"] == 1
    assert not set(structural["failing"]) & set(lineage["failing"])
    assert "not caused by this stage" in lineage["cause"]
    assert "no frozen test was edited" in structural["action"]
    assert quality["verdict"] == "EARLY_WINDOW_ATTRIBUTION_UNRESOLVED"
    assert quality["not_run"]
    assert quality["hosted_ci"]["rule"] == "no hosted result is claimed for a head that was not pushed and observed"


def test_the_module_hash_recorded_in_the_evidence_is_the_module_and_the_readout_on_disk(evidence):
    provenance = evidence["source_and_provenance.json"]
    assert provenance["module_sha256"] == stage.sha256_file(MODULE_PATH)
    assert provenance["readout_module_sha256"] == stage.sha256_file(READOUT_PATH)
    assert (
        provenance["module_path"] == "examples/two_phase/production/frozen_reconstruction_branch_attribution_audit.py"
    )
    assert provenance["readout_module_path"] == "examples/two_phase/production/l1a2y_contour_branch_readout.py"
    frozen = provenance["frozen_source_sha256"]
    assert frozen["examples/two_phase/phasefield.py"] == stage.sha256_file(TWO_PHASE / "phasefield.py")
    for path in frozen:
        assert (TWO_PHASE.parents[1] / path).is_file()
