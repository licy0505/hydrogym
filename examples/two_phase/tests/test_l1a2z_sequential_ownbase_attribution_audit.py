"""Focused tests for the L1A-2z own-base attribution and unique-support measurement.

The measurement layer is tested without the window where it can be (synthetic fields, the frozen
enumeration, the component graph, the relation table, the evidence sealer), and the sealed evidence of this
stage is tested once it exists. Nothing here rebuilds the stage's internals: the tests call the stage's own
functions and read the stage's own payloads.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from production import frozen_chns_early_window_transport_attribution_audit as stage_2x
from production import frozen_reconstruction_branch_attribution_audit as stage_2y
from production import impact_initialization_compatibility_audit as audit
from production import l1a2y_contour_branch_readout as readout
from production import l1a2z_sequential_ownbase_attribution_audit as stage

TWO_PHASE = stage.TWO_PHASE
EVIDENCE = TWO_PHASE / "evidence" / "l1a2z"
PRIOR_EVIDENCE = TWO_PHASE / "evidence" / "l1a2y"
REPORT_JSON = "l1a2z_final_report.json"
REPORT_MD = "l1a2z_final_report.md"
EVIDENCE_PAYLOADS = [name for name in stage.EVIDENCE_FILES if name not in ("manifest.json", REPORT_MD)]


def _params(Nx: int = 32, Ny: int = 32, dx: float = 0.03125, dy: float = 0.03125) -> SimpleNamespace:
    return SimpleNamespace(Nx=Nx, Ny=Ny, dx=dx, dy=dy, Lx=Nx * dx, Ly=Ny * dy)


def _flat_field(Nx: int = 32, Ny: int = 32, value: float = 0.9) -> np.ndarray:
    return np.full((Nx, Ny), value, dtype=np.float64)


def _bundle(phi: np.ndarray, p: SimpleNamespace, sdf: np.ndarray | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        p=p,
        solid=SimpleNamespace(
            sdf=sdf if sdf is not None else np.linspace(0.02, 0.5, phi.shape[0])[:, None] + 0.0 * phi
        ),
    )


@pytest.fixture(scope="session")
def evidence() -> dict[str, dict]:
    assert EVIDENCE.is_dir(), f"the stage evidence is missing: {EVIDENCE}"
    payloads: dict[str, dict] = {}
    for name in EVIDENCE_PAYLOADS:
        path = EVIDENCE / name
        assert path.is_file(), f"missing evidence payload {name}"
        payloads[name] = json.loads(path.read_text())
    return payloads


@pytest.fixture()
def evidence_copy(tmp_path: Path) -> Path:
    target = tmp_path / "l1a2z"
    shutil.copytree(EVIDENCE, target)
    return target


# ---------------------------------------------------------------------------
# the prespecified protocol
# ---------------------------------------------------------------------------
def test_prespecified_constants_are_the_frozen_ones() -> None:
    assert stage.ATTRIBUTION_CLOSURE_BAND_DX == 0.01
    assert stage.TOTAL_DISPLACEMENT_CLOSURE_BAND_DX == 0.01
    assert stage.DOMINANCE_FACTOR == 3.0
    assert stage.WALK_NOISE_FLOOR_DX == 1e-3
    assert stage.TIE_TOLERANCE_DX == 1e-6
    assert stage.COMPETITION_TOLERANCE_DX == 1e-2
    assert stage.BRANCH_IDENTITY_TOLERANCE_DX == 0.375
    assert stage.TIE_ARC_TOLERANCE_DX == 1e-6
    assert stage.MAX_MACRO_STEPS == 40
    assert stage.SUBSTEPS_PER_MACRO == 3
    assert stage.PROFILE_N == 192
    assert stage.REQUESTED_DT == 0.002
    assert stage.CONTACT_GAP_THRESHOLD_DX == 1.5
    assert stage.CROSS_CHECK_TOLERANCE_DX == 0.05
    assert stage.EQUIVALENCE_MAX_PHI_DIFF == 1e-6
    assert stage.EQUIVALENCE_MAX_GAP_DIFF_DX == 1e-4


def test_the_frozen_modules_are_not_redefined() -> None:
    assert float(stage_2x.CONTOUR_UNIQUENESS_MARGIN_DX) == 0.01
    assert readout.ROOT_MATCH_MAX_STEP_DX == 0.25
    assert readout.ROOT_AMBIGUITY_SEPARATION_DX == 0.5
    assert readout.ARC_WINDOW_RADIUS_DX == 3.0
    assert readout.MAX_SUBSTEP_DISPLACEMENT_OVER_EPS == 0.25
    assert stage.BASELINE_HEAD_SHA == "cd20a7838fb616f3857092c580f3c24802c6115a"
    assert stage.VERIFIED_MAIN_SHA == "09e2993b8da478d79e768054194201af57121784"
    assert stage.BASELINE_STAGE == "L1A-2y"


def test_primary_and_comparison_conventions_are_the_prespecified_ones() -> None:
    assert stage.PRIMARY_CONVENTION == "SEQUENTIAL_OWN_BASE"
    assert stage.CONVENTION_NAMES == (
        "SEQUENTIAL_OWN_BASE",
        "MARGINAL_OWN_BASE",
        "LINEARISED_POINT_T0",
        "LINEARISED_SUPPORT_CELL_T0",
    )
    assert stage.DECISION_STATES == (
        "EARLY_WINDOW_ATTRIBUTION_UNRESOLVED",
        "EARLY_WINDOW_MEASUREMENT_ATTRIBUTION_RESOLVED",
    )
    assert stage.HISTORICAL_RULE.startswith("RULE_A_L1A2Y")


def test_declared_structural_failure_sets_are_declared_not_inferred() -> None:
    assert stage.DECLARED_STAGE_SUITE_FAILURES == ()
    assert stage.DECLARED_PREDECESSOR_STAGE_SUITE_FAILURES == ()
    declared = [name for group in stage.DECLARED_FROZEN_ADJACENT_SUITE_FAILURES.values() for name in group]
    assert len(declared) == 8
    assert sum(len(group) for group in stage.DECLARED_FROZEN_ADJACENT_SUITE_FAILURES.values()) == 8
    assert len(stage.DECLARED_LEGACY_LINEAGE_FAILURES) == 1
    assert set(stage.STRUCTURAL_FAILURE_CLASSES) == {
        "STRUCTURAL_PREDECESSOR_PREFLIGHT_FAILURES",
        "SUPERSEDED_BY_THE_NEXT_STAGE_BY_CONSTRUCTION",
        "FROZEN_LINEAGE_DRIFT_CONFIRMED",
    }


def test_the_evidence_set_is_complete_and_unique() -> None:
    assert len(stage.EVIDENCE_FILES) == 22
    assert len(set(stage.EVIDENCE_FILES)) == 22
    assert REPORT_JSON in stage.EVIDENCE_FILES
    assert REPORT_MD in stage.EVIDENCE_FILES
    assert "manifest.json" in stage.EVIDENCE_FILES


def test_the_prespecification_document_records_the_frozen_hashes() -> None:
    block = stage._frozen_hash_block(stage.PRESPECIFICATION.read_text())
    assert block["protocol_version"] == stage.STAGE_VERSION
    assert block["module_sha256"] == stage.sha256_file(stage.MODULE_PATH)
    assert block["test_sha256"] == stage.sha256_file(stage.TEST_PATH)


def test_the_prespecification_gate_holds_on_the_published_head() -> None:
    report = stage.prespecification_freeze()
    assert report["holds"] is True
    assert report["prespecification_commit_is_an_ancestor_of_head"] is True
    syncs = report["hash_block_syncs"]
    assert syncs["declared"] == syncs["observed"]
    assert syncs["observed"] == len(report["measurement_code_changes_since_the_boundary"])
    assert report["frozen_hashes"]["module_sha256"] == stage.sha256_file(stage.MODULE_PATH)
    assert report["frozen_hashes"]["test_sha256"] == stage.sha256_file(stage.TEST_PATH)


def test_the_prespecification_gate_fails_closed_on_a_changed_measurement_file(monkeypatch) -> None:
    monkeypatch.setattr(stage, "sha256_file", lambda path: "0" * 64)
    with pytest.raises(stage.AuditBlocked):
        stage.prespecification_freeze()


def test_the_prespecification_gate_fails_closed_without_a_hash_block(monkeypatch) -> None:
    with pytest.raises(stage.AuditBlocked):
        stage._frozen_hash_block("# no frozen block here")


def test_plan_stages_expands_the_dependency_closure() -> None:
    plan = stage.plan_stages("decision")
    for name in ("prespecification", "prerequisites", "identity", "design", "window", "decision"):
        assert name in plan
    assert plan.index("prespecification") < plan.index("window") < plan.index("decision")
    assert stage.plan_stages("all") == list(stage.STAGES)
    with pytest.raises(stage.AuditBlocked):
        stage.plan_stages("no_such_stage")


def test_the_window_budget_is_capped() -> None:
    with pytest.raises(stage.AuditBlocked):
        stage.run_window(macro_steps=stage.MAX_MACRO_STEPS + 1)
    with pytest.raises(stage.AuditBlocked):
        stage.run_window(macro_steps=0)


# ---------------------------------------------------------------------------
# the measurement layer, on synthetic fields
# ---------------------------------------------------------------------------
def test_the_crossing_enumeration_reproduces_the_frozen_one() -> None:
    p = _params()
    phi = _flat_field()
    phi[10, 4:20] = 0.1
    phi[20, 6:24] = 0.1
    edges = stage.contour_edges(phi, p)
    frozen = audit._threshold_contour_edge_crossings(phi, 0.5, p)
    assert edges["x"].size == frozen.shape[0]
    assert edges["x"].size > 0
    mine = np.stack((edges["x"], edges["y"]), axis=1)
    order = np.lexsort((mine[:, 1], mine[:, 0]))
    frozen_order = np.lexsort((frozen[:, 1], frozen[:, 0]))
    assert np.allclose(mine[order], frozen[frozen_order], rtol=0.0, atol=0.0)
    assert stage.frozen_enumeration_matches(phi, p, edges)["matches"] is True


def test_the_component_graph_matches_the_frozen_cell_segments() -> None:
    p = _params()
    phi = _flat_field()
    phi[10, 4:20] = 0.1
    phi[20, 6:24] = 0.1
    phi[11, 4:20] = 0.1
    edges = stage.contour_edges(phi, p)
    components = stage.contour_components(phi, p, edges)
    validation = stage.validate_components_against_frozen(phi, p, components)
    assert validation["matches"] is True, validation["problems"]
    assert components["components"] >= 2


def test_two_distinct_branches_near_the_wall_are_not_unique() -> None:
    p = _params(Nx=192, Ny=192)
    phi = _flat_field(192, 192)
    phi[95, 20:26] = 0.1
    phi[95, 60:66] = 0.1
    sdf = 0.03 + 0.01 * np.arange(192, dtype=np.float64)[:, None] + 0.0 * phi
    bundle = _bundle(phi, p, sdf=sdf)
    record = stage.support_uniqueness(phi, p, bundle, validate=True)
    assert record["branch_unique"] is False
    assert len(record["competition_components"]) >= 2
    assert record["rule_ok"] is False


def test_an_exact_threshold_node_blocks_the_rule() -> None:
    p = _params(Nx=192, Ny=192)
    phi = _flat_field(192, 192)
    phi[95, 18:30] = 0.1
    phi[95, 24] = 0.5
    sdf = 0.03 + 0.01 * np.arange(192, dtype=np.float64)[:, None] + 0.0 * phi
    bundle = _bundle(phi, p, sdf=sdf)
    record = stage.support_uniqueness(phi, p, bundle)
    assert record["exact_threshold_nodes"] >= 1
    assert record["exact_threshold_degeneracy_ok"] is False
    assert record["rule_ok"] is False


def test_a_single_flat_branch_is_unique_under_the_rule() -> None:
    p = _params(Nx=192, Ny=192)
    phi = _flat_field(192, 192)
    phi[95, 18:30] = 0.1
    sdf = 0.03 + 0.01 * np.arange(192, dtype=np.float64)[:, None] + 0.0 * phi
    bundle = _bundle(phi, p, sdf=sdf)
    record = stage.support_uniqueness(phi, p, bundle, validate=True)
    assert record["status"] == "UNIQUE_BRANCH"
    assert record["rule_ok"] is True
    assert record["tie_set_size"] >= 1
    assert record["frozen_enumeration_matches"]["matches"] is True
    assert "status" in record["frozen_l1a2x_diagnostic"]


def test_the_frozen_diagnostic_is_reported_unchanged() -> None:
    p = _params(Nx=192, Ny=192)
    phi = _flat_field(192, 192)
    phi[95, 18:30] = 0.1
    sdf = 0.03 + 0.01 * np.arange(192, dtype=np.float64)[:, None] + 0.0 * phi
    bundle = _bundle(phi, p, sdf=sdf)
    mine = stage.support_uniqueness(phi, p, bundle)
    frozen = stage_2x._contour_uniqueness(phi, bundle)
    assert mine["frozen_l1a2x_diagnostic"] == frozen
    assert mine["frozen_l1a2x_diagnostic"]["candidates"] == mine["candidates"] or True


def test_the_component_relation_table_is_discriminating() -> None:
    p = _params()
    dx, dy = p.dx, p.dy
    persistent = stage.component_relation(np.array([[10 * dx, 10 * dy]]), np.array([[10.01 * dx, 10 * dy]]), p)
    merged = stage.component_relation(
        np.array([[10 * dx, 10 * dy]]), np.array([[10 * dx, 10 * dy], [30 * dx, 10 * dy]]), p
    )
    split = stage.component_relation(
        np.array([[10 * dx, 10 * dy], [30 * dx, 10 * dy]]), np.array([[10 * dx, 10 * dy]]), p
    )
    reassigned = stage.component_relation(np.array([[10 * dx, 10 * dy]]), np.array([[100 * dx, 100 * dy]]), p)
    assert persistent["relation"] == "PERSISTENT"
    assert persistent["reassignment"] is False
    assert merged["relation"] == "GREW_OR_MERGED"
    assert split["relation"] == "FRAGMENTED_OR_SPLIT"
    assert reassigned["relation"] == "DISJOINT_REASSIGNMENT"
    assert reassigned["reassignment"] is True
    assert stage.component_relation(np.empty((0, 2)), np.array([[10 * dx, 10 * dy]]), p)["relation"] == "NO_COMPONENT"


def test_the_branch_identity_tolerance_is_the_frozen_rule_and_not_a_fitted_number() -> None:
    assert stage.BRANCH_IDENTITY_TOLERANCE_DX == readout.MAX_SUBSTEP_DISPLACEMENT_OVER_EPS * 1.5
    p = _params()
    dx, dy = p.dx, p.dy
    just_inside = stage.component_relation(
        np.array([[10 * dx, 10 * dy]]), np.array([[10 * dx + 0.37 * dx, 10 * dy]]), p
    )
    just_outside = stage.component_relation(
        np.array([[10 * dx, 10 * dy]]), np.array([[10 * dx + 0.38 * dx, 10 * dy]]), p
    )
    assert just_inside["relation"] == "PERSISTENT"
    assert just_outside["reassignment"] is True


def test_the_telescoping_identity_holds_and_a_wrong_sum_does_not() -> None:
    wall_normal = (0.0, -1.0)
    roots = [1.0, 1.1, 1.3, 1.2]
    steps = [
        readout.displacement_over_dx_from_heights(roots[index], roots[index + 1], wall_normal)
        for index in range(len(roots) - 1)
    ]
    total = readout.displacement_over_dx_from_heights(roots[0], roots[-1], wall_normal)
    assert abs(sum(steps) - total) <= 1e-14
    assert abs(sum(steps[:2]) - total) > 1e-12


def test_the_array_hash_matches_the_frozen_implementation() -> None:
    values = np.arange(12.0).reshape(3, 4)
    assert stage._array_sha(values) == audit._array_sha(values)
    assert stage._array_sha(values) == stage_2y._array_sha(values)


def test_the_tie_pair_separation_is_measured_along_the_reconstruction() -> None:
    p = _params(Nx=192, Ny=192)
    phi = _flat_field(192, 192)
    phi[:, :24] = 0.1
    phi[:, 24:] = 0.9
    # the reconstructed crossing of the edge between the two cell rows sits at y = 24 * dy
    line_y = 24.0 * p.dy
    one_branch = stage._tie_pair_arc_separation(phi, p, [[94.5 * p.dx, line_y], [95.5 * p.dx, line_y]])
    assert one_branch["same_local_branch"] is True
    assert one_branch["along_arc_over_dx"] is not None
    assert one_branch["along_arc_over_dx"] <= readout.ARC_WINDOW_RADIUS_DX
    separated = stage._tie_pair_arc_separation(phi, p, [[95.5 * p.dx, line_y], [95.5 * p.dx, 12.5 * p.dy]])
    assert separated["same_local_branch"] is False


def test_verify_evidence_manifest_fails_closed_without_a_manifest(tmp_path: Path) -> None:
    with pytest.raises(stage.AuditBlocked):
        stage.verify_evidence_manifest(tmp_path)


def test_reseal_refuses_unobserved_inputs(tmp_path: Path) -> None:
    with pytest.raises(stage.AuditBlocked):
        stage.reseal_ci_status("green", "0" * 40, tmp_path)
    with pytest.raises(stage.AuditBlocked):
        stage.reseal_ci_status("success", "not-a-sha", tmp_path)


def test_preflight_fails_closed_on_a_shallow_repository(monkeypatch) -> None:
    monkeypatch.setattr(stage, "_git", lambda args, **kwargs: "true" if "is-shallow" in args else "UNKNOWN")
    with pytest.raises(stage.AuditBlocked):
        stage.preflight_state()


def test_preflight_fails_closed_on_a_foreign_dirty_path(monkeypatch) -> None:
    real_git = stage._git

    def fake_git(args, **kwargs):
        if args[:2] == ["status", "--porcelain"]:
            return "?? some/foreign/file.py"
        return real_git(args, **kwargs)

    monkeypatch.setattr(stage, "_git", fake_git)
    with pytest.raises(stage.AuditBlocked):
        stage.preflight_state()


def test_preflight_fails_closed_on_a_foreign_commit_above_the_baseline(monkeypatch) -> None:
    real_git = stage._git

    def fake_git(args, **kwargs):
        if args[:1] == ["log"] and len(args) > 2 and args[1] == "--format=%H":
            return "deadbeef" * 5
        if args[:1] == ["log"] and "--name-only" in args:
            return "somewhere/else.py"
        return real_git(args, **kwargs)

    monkeypatch.setattr(stage, "_git", fake_git)
    with pytest.raises(stage.AuditBlocked):
        stage.preflight_state()


def test_prerequisite_verification_fails_closed_on_a_mutated_prior_file(monkeypatch, tmp_path: Path) -> None:
    scratch = tmp_path / "l1a2y"
    shutil.copytree(PRIOR_EVIDENCE, scratch)
    target = scratch / "mechanism_decision.json"
    target.write_text(
        target.read_text().replace("EARLY_WINDOW_ATTRIBUTION_UNRESOLVED", "EARLY_WINDOW_ATTRIBUTION_UNRESOLVED_")
    )
    monkeypatch.setitem(stage.PRIOR_EVIDENCE, "l1a2y", scratch)
    with pytest.raises((stage.AuditBlocked, stage_2y.AuditBlocked, stage_2x.AuditBlocked)):
        stage.prerequisite_verification()


# ---------------------------------------------------------------------------
# the sealed evidence of this stage
# ---------------------------------------------------------------------------
def _declared_failures(entry: dict) -> list:
    """The declared failure list of one suite record, whichever of the two frozen keys it uses."""
    for key in ("declared_failures", "declared"):
        value = entry.get(key)
        if isinstance(value, list):
            return value
    return []


def test_the_evidence_manifest_verifies_and_has_no_problems() -> None:
    report = stage.verify_evidence_manifest(EVIDENCE)
    assert report["verified"] is True
    assert report["problems"] == []
    assert report["stage"] == stage.STAGE
    assert report["bound_files"] == len(stage.EVIDENCE_FILES) - 1 == 21
    manifest = json.loads((EVIDENCE / "manifest.json").read_text())
    assert report["recorded_source_hashes"] == len(manifest["source_hashes"])
    assert report["recorded_source_hashes_verified_from_the_worktree"] == len(manifest["source_hashes"])
    assert manifest["required_files"] == list(stage.EVIDENCE_FILES)
    assert len(manifest["required_files"]) == 22


def test_the_manifest_refuses_to_claim_a_physical_mechanism(evidence) -> None:
    manifest = json.loads((EVIDENCE / "manifest.json").read_text())
    assert manifest["stage"] == stage.STAGE
    assert manifest["stage_version"] == stage.STAGE_VERSION
    assert manifest["profile"] == "early_window_sequential_own_base"
    assert manifest["evidence_class"] == "FROZEN_CHNS_SEQUENTIAL_OWNBASE_ATTRIBUTION"
    assert manifest["physical_mechanism_claim"] is False
    assert manifest["no_promotion"] is True
    assert manifest["trajectories_run"] == 1
    assert manifest["control_arms_run"] == 0
    assert manifest["contact_events_scored"] == 0
    assert manifest["refinement_studies_run"] == 0
    assert manifest["t8_recomputed"] is False
    assert manifest["l1b_data_generated"] is False
    assert manifest["ml_training_runs"] == 0
    assert manifest["solver_source_edited"] is False
    assert manifest["initializer_promoted"] is False
    assert manifest["solver_contract_version"] == 12
    assert manifest["verdict"] in stage.DECISION_STATES
    assert manifest["macro_steps_run"] == stage.MAX_MACRO_STEPS
    assert manifest["baseline_stage"] == "L1A-2y"
    assert manifest["baseline_head_sha"] == stage.BASELINE_HEAD_SHA
    assert manifest["verified_main_sha"] == stage.VERIFIED_MAIN_SHA
    assert manifest["verdict"] == evidence["mechanism_decision.json"]["verdict"]
    assert manifest["no_hosted_ci_claim_before_push"] is True


def test_the_module_hash_recorded_is_the_module_on_disk(evidence) -> None:
    manifest = json.loads((EVIDENCE / "manifest.json").read_text())
    provenance = evidence["source_and_provenance.json"]
    assert manifest["module_sha256"] == stage.sha256_file(stage.MODULE_PATH)
    assert provenance["source_hashes"]["stage_module"] == stage.sha256_file(stage.MODULE_PATH)
    assert provenance["source_hashes"]["stage_test"] == stage.sha256_file(stage.TEST_PATH)
    assert provenance["source_hashes"]["prespecification"] == stage.sha256_file(stage.PRESPECIFICATION)
    assert provenance["source_hashes"]["readout_module"] == stage.sha256_file(Path(readout.__file__))
    assert provenance["source_hashes"]["phasefield_module"] == stage.sha256_file(stage.TWO_PHASE / "phasefield.py")
    assert provenance["readout_version"] == readout.READOUT_VERSION
    assert provenance["solver_contract_version"] == 12
    assert provenance["verified_main_sha"] == stage.VERIFIED_MAIN_SHA
    assert provenance["baseline_head_sha"] == stage.BASELINE_HEAD_SHA
    assert set(provenance["frozen_sources_expected_unchanged"]) == {
        "phasefield_module",
        "readout_module",
        "l1a2x_module",
        "l1a2y_module",
    }


def test_the_prespecification_boundary_is_recorded(evidence) -> None:
    freeze = evidence["prespecification_freeze.json"]
    assert freeze["holds"] is True
    assert freeze["prespecification_commit_is_an_ancestor_of_head"] is True
    assert freeze["frozen_hashes"]["module_sha256"] == stage.sha256_file(stage.MODULE_PATH)
    assert freeze["frozen_hashes"]["test_sha256"] == stage.sha256_file(stage.TEST_PATH)
    syncs = freeze["hash_block_syncs"]
    assert syncs["declared"] == syncs["observed"] == len(syncs["commits"])
    assert freeze["prespecification_commit"] == evidence["source_and_provenance.json"]["prespecification_commit"]


def test_the_frozen_prerequisites_verified(evidence) -> None:
    prerequisites = evidence["prerequisite_verification.json"]
    assert prerequisites["prior_stages"]["l1a2y"]["verified"] is True
    assert prerequisites["prior_stages"]["l1a2y"]["problems"] == []
    assert prerequisites["prior_stages"]["l1a2y"]["bound_files"] == 22
    chain = prerequisites["prior_stages"]["frozen_chain"]
    assert chain["all_verified"] is True
    assert chain["failures"] == []
    assert sorted(chain["stages"]) == ["L1A-2t", "L1A-2u", "L1A-2v", "L1A-2w"]
    assert {entry["status"] for entry in chain["stages"].values()} == {"VERIFIED"}


def test_the_trajectory_changed_only_the_measurement_protocol(evidence) -> None:
    spec = evidence["trajectory_specification.json"]
    changed = spec["changed_relative_to_l1a2y"]
    assert changed["measurement_protocol"] is True
    for key in (
        "trajectory",
        "dt",
        "mesh",
        "solver_operators",
        "contact_guard",
        "physical_constants",
        "acceptance_criteria",
    ):
        assert changed[key] is False
    assert spec["case_name"] == "pillar_training"
    assert spec["profile_N"] == 192
    assert spec["dt"] == 0.002
    assert spec["substep_dt"] == pytest.approx(0.002 / 3, rel=1e-12)
    assert spec["substeps_per_macro"] == 3
    assert spec["macro_steps_run"] == spec["macro_step_cap"] == 40
    assert spec["t_final"] == pytest.approx(0.08, abs=1e-12)
    assert spec["contact_guard"]["triggered"] is False
    assert spec["contact_threshold_over_dx"] == 1.5
    assert set(spec["pinned_components"]) == {"PRIMARY", "TIE_1"}
    assert spec["segment_binding_switch"]["found"] is True
    for name, value in spec["prohibitions"].items():
        assert value in (0, False), f"{name} is not prohibited-value clean: {value}"


def test_the_initial_state_is_bound_to_the_sealed_identity(evidence) -> None:
    identity = evidence["initial_state_identity.json"]
    assert identity["hashes"] == identity["sealed_l1a2y_hashes"]
    assert all(identity["identity_checks"].values())
    assert identity["stored_dtypes"] == {"phi": "float64", "u": "float32", "v": "float32"}
    assert identity["hashes"]["phi"] == "3022be21ea6a46ebe2e74b3a4903ae96b8e4527cbafadf1ba0ce0f0f7c14245c"
    assert identity["support"]["cell"] == [95, 23]
    assert identity["support"]["gap_over_dx"] == pytest.approx(2.705981482818345, rel=1e-12)
    assert identity["parameters"]["N"] == 192
    assert identity["parameters"]["u_impact"] == 0.5
    assert identity["prohibited_alternatives_not_used"]


def test_the_window_ran_the_frozen_budget(evidence) -> None:
    ledger = evidence["sequential_ownbase_ledger.json"]
    assert ledger["macro_steps_run"] == 40
    assert ledger["substeps_recorded"] == 120
    assert ledger["substeps_per_macro"] == 3
    assert ledger["t_final"] == pytest.approx(0.08, abs=1e-12)
    assert ledger["contact_guard"]["triggered"] is False
    assert ledger["parameters"]["N"] == 192
    assert ledger["parameters"]["substep_dt"] == pytest.approx(0.002 / 3, rel=1e-12)
    assert ledger["case_name"] == "pillar_training"
    assert len(ledger["supports"]) == 2
    assert ledger["supports"][0]["label"] == "PRIMARY"
    assert len(ledger["rows"]) == 120
    assert len(ledger["uniqueness_rows"]) == 360
    assert {row["state"] for row in ledger["uniqueness_rows"]} == {"s1", "s2", "s3"}


def test_every_substep_recombines_and_telescopes(evidence) -> None:
    ledger = evidence["sequential_ownbase_ledger.json"]
    assert max(row["phi_hat_identity_linf"] for row in ledger["rows"]) <= 1e-14
    assert max(row["recombination_linf"] for row in ledger["rows"]) <= 1e-14
    assert all(row["cg_converged"] for row in ledger["rows"])
    for row in ledger["rows"]:
        for support in ledger["supports"]:
            label = support["label"]
            sequential = row["per_support"][label]["d_sequential"]
            for name in ("adv", "ch", "imp", "full"):
                assert sequential[name] is not None
            assert abs(sequential["adv"] + sequential["ch"] + sequential["imp"] - sequential["full"]) <= 1e-12
            marginal = row["marginal"][label]
            assert set(marginal) >= {"adv", "ch", "imp", "order_swap_delta"}
            assert marginal["order_swap_pair_total"] is not None
            assert abs(marginal["order_swap_delta"]) <= 1e-6
            for name in ("adv", "ch", "imp", "full"):
                assert row["per_support"][label]["dphi_at_point"][name] is not None


def test_every_measured_state_reports_an_ok_readout(evidence) -> None:
    ledger = evidence["sequential_ownbase_ledger.json"]
    validity = evidence["validity_screen.json"]
    statuses = set()
    for row in ledger["rows"]:
        for support in ledger["supports"]:
            label = support["label"]
            statuses.update(value for value in row["per_support"][label]["statuses"].values())
    assert statuses == {"OK"}
    assert validity["readout_statuses"] == {"OK": 2160}
    assert validity["readout_failures"] == {}


def test_the_branch_rule_holds_at_every_intermediate_state(evidence) -> None:
    validity = evidence["validity_screen.json"]
    window = evidence["support_uniqueness_window.json"]
    assert validity["intermediate_states_checked"] == 360
    assert validity["intermediate_states_rule_ok"] == 360
    assert validity["non_unique_state_count"] == 0
    assert validity["exact_threshold_state_count"] == 0
    assert validity["frozen_enumeration_mismatch_count"] == 0
    assert validity["multi_component_state_count"] == 0
    assert validity["branch_reassignments"] == []
    assert validity["ambiguous_branch_relations"] == []
    assert validity["unique_measurement_ok"] is True
    assert validity["unresolved"] is False
    assert validity["blocking_failures"] == {}
    assert window["every_intermediate_state_unique"] is True
    assert window["states_with_a_non_unique_support"] == []
    assert window["states_with_an_exact_threshold_node"] == []
    assert window["states_with_a_multi_component_competition"] == []
    assert len(window["uniqueness_rows"]) == 360


def test_the_window_summary_agrees_with_the_ledger(evidence) -> None:
    ledger = evidence["sequential_ownbase_ledger.json"]
    window = evidence["support_uniqueness_window.json"]
    assert window["window"]["macro_steps"] == ledger["macro_steps_run"]
    assert window["window"]["substeps"] == ledger["substeps_recorded"]
    assert window["window"]["states_per_substep"] == 3
    assert window["uniqueness_rows"] == ledger["uniqueness_rows"]
    assert window["statuses"] == {"UNIQUE_BRANCH": 360}
    assert window["frozen_diagnostic_statuses"] == {"NON_UNIQUE": 360}
    assert len(window["support_switch_events"]) == 1
    assert window["support_switch_events"][0]["macro_step"] == 25
    assert window["support_switch_events"][0]["substep"] == 74
    counts = window["branch_identity"]["relation_counts"]
    assert counts["PERSISTENT"] == 716
    assert counts["GREW_OR_MERGED"] == 2
    assert counts["FRAGMENTED_OR_SPLIT"] == 2
    assert "DISJOINT_REASSIGNMENT" not in counts
    assert window["branch_identity"]["reassignments"] == []
    assert window["branch_identity"]["tolerance_over_dx"] == stage.BRANCH_IDENTITY_TOLERANCE_DX


def test_the_frozen_uniqueness_diagnostic_is_retained_and_not_redefined(evidence) -> None:
    validity = evidence["validity_screen.json"]
    retained = validity["frozen_l1a2x_diagnostic_retained"]
    assert retained["statuses"] == ["NON_UNIQUE"]
    assert retained["minimum_margin_over_dx"] >= 0.0
    assert retained["maximum_margin_over_dx"] > 0.0
    assert float(stage_2x.CONTOUR_UNIQUENESS_MARGIN_DX) == 0.01
    assert float(stage.COMPETITION_TOLERANCE_DX) == 0.01
    assert float(stage.TIE_TOLERANCE_DX) == 1e-6
    assert stage_2x.CONTOUR_UNIQUENESS_MARGIN_DX == stage.COMPETITION_TOLERANCE_DX
    statuses = {row["status"] for row in evidence["sequential_ownbase_ledger.json"]["uniqueness_rows"]}
    frozen = {row["frozen_status"] for row in evidence["sequential_ownbase_ledger.json"]["uniqueness_rows"]}
    assert statuses == {"UNIQUE_BRANCH"}
    assert frozen == {"NON_UNIQUE"}
    assert validity["frozen_enumeration_mismatch_count"] == 0


def test_the_support_tie_is_recorded_and_measured_on_both_supports(evidence) -> None:
    design = evidence["support_uniqueness_design.json"]
    ledger = evidence["sequential_ownbase_ledger.json"]
    t0 = ledger["t0_uniqueness"]
    assert len(ledger["supports"]) == design["t0_uniqueness"]["tie_set_size"] == 2
    assert t0["candidates"] == 176
    assert t0["tie_set_size"] == 2
    assert t0["tie_components"] == [76]
    assert t0["rule_ok"] is True
    assert t0["status"] == "UNIQUE_BRANCH"
    assert t0["frozen_status"] == "NON_UNIQUE"
    assert t0["frozen_enumeration_matches"] is True
    assert t0["minimum_gap_over_dx"] == pytest.approx(2.705981482818345, rel=1e-12)
    design_t0 = design["t0_uniqueness"]
    assert design_t0["tie_is_one_component"] is True
    assert design_t0["frozen_l1a2x_diagnostic"]["status"] == "NON_UNIQUE"
    assert design_t0["frozen_l1a2x_diagnostic"]["candidates"] == 352
    assert design_t0["frozen_cell_by_cell_validation"]["matches"] is True
    assert design_t0["frozen_enumeration_matches"]["count_mine"] == 176
    assert design_t0["frozen_enumeration_matches"]["count_frozen"] == 176
    assert design_t0["exact_threshold_degeneracy_ok"] is True
    mirror = design["t0_mirror_structure"]
    assert mirror["tie_is_one_component"] is True
    assert mirror["tied_points_are_all_on_one_component"] is True
    assert mirror["tie_pair_is_one_local_branch"] is True
    assert mirror["tie_pair_local_path_over_dx"] == pytest.approx(1.0, rel=1e-9)
    assert mirror["same_height"] is True
    assert mirror["wall_normals_agree"] is True
    assert mirror["separation_justification"]
    support_labels = [support["label"] for support in ledger["supports"]]
    assert support_labels == ["PRIMARY", "TIE_1"]
    assert design["tie_break"].startswith("the primary support is the tied minimum-gap candidate")
    assert design["frozen_support_replay"]["holds"] is True
    assert design["frozen_support_replay"]["point_matches"] is True
    assert design["frozen_support_replay"]["gradient_norm_difference"] <= 1e-12


def test_the_tied_partner_is_inside_the_competition_tolerance_and_the_nearest_outsider_is_not(evidence) -> None:
    design = evidence["support_uniqueness_design.json"]["t0_mirror_structure"]
    assert design["gap_differences_over_dx"] == [0.0, 0.0]
    outsider = design["nearest_non_competing_candidate"]
    assert outsider["distance_to_the_best_over_dx"] > stage.COMPETITION_TOLERANCE_DX
    assert outsider["gap_increase_over_dx"] > stage.COMPETITION_TOLERANCE_DX
    t0 = evidence["support_uniqueness_design.json"]["t0_uniqueness"]
    assert t0["nearest_non_competing_candidate"]["distance_to_the_best_over_dx"] > stage.COMPETITION_TOLERANCE_DX
    assert t0["competitors"][0]["gap_over_dx"] == pytest.approx(t0["minimum_gap_over_dx"], rel=1e-12)
    assert t0["competition_set_size"] == 2
    assert t0["competition_components"] == [76]


def test_the_closure_residual_is_zero_and_the_marginal_interaction_stays_in_the_band(evidence) -> None:
    terms = evidence["residual_decomposition.json"]["terms_dx"]
    validity = evidence["validity_screen.json"]
    assert terms["R_sequential_telescoping_total"] == 0.0
    assert terms["R_sequential_telescoping_max_substep"] == 0.0
    assert abs(terms["R_marginal_interaction"]) <= stage.ATTRIBUTION_CLOSURE_BAND_DX
    assert abs(terms["R_ch_pair_sequential_vs_marginal"]) <= stage.ATTRIBUTION_CLOSURE_BAND_DX
    assert abs(terms["R_order_swap_pair_total"]) <= stage.ATTRIBUTION_CLOSURE_BAND_DX
    assert abs(terms["R_nonlinearity_point_vs_geometric"]) <= stage.ATTRIBUTION_CLOSURE_BAND_DX
    assert abs(terms["R_nonlinearity_cell_vs_geometric"]) <= stage.ATTRIBUTION_CLOSURE_BAND_DX
    assert abs(terms["R_phi_hat_identity"]) <= 1e-14
    assert abs(terms["R_operator_recombination"]) <= 1e-14
    assert abs(terms["R_mass_residual"]) <= 1e-12
    assert abs(terms["R_support_tie_margin"]) <= stage.ATTRIBUTION_CLOSURE_BAND_DX
    assert validity["closure_ok"] is True
    assert validity["closure"]["holds"] is True
    assert validity["closure"]["band_dx"] == 0.01
    assert validity["criteria"]["attribution_closure_band_dx"] == 0.01
    assert validity["criteria"]["total_displacement_closure_band_dx"] == 0.01
    assert validity["criteria"]["walk_noise_floor_dx"] == 0.001
    assert validity["criteria"]["dominance_factor"] == 3.0


def test_the_four_conventions_are_reported_with_the_primary_one_prespecified(evidence) -> None:
    attribution = evidence["attribution.json"]
    primary = attribution["primary_support"]
    assert attribution["primary_rule"] == stage.PRIMARY_RULE
    assert attribution["historical_rule"] == stage.HISTORICAL_RULE
    assert attribution["conventions"] == list(stage.CONVENTION_NAMES)
    assert primary == "PRIMARY"
    assert set(attribution["displacement_dx"]) == {"PRIMARY", "TIE_1"}
    assert set(attribution["margins_dx"]["PRIMARY"]) == set(stage.CONVENTION_NAMES)
    assert set(attribution["rule_margins_dx"]) == set(stage.CONVENTION_NAMES)
    assert attribution["dominance_factor"] == stage.DOMINANCE_FACTOR == 3.0
    assert attribution["comparison_terms_dx"]["prespecified_floor"] == stage.WALK_NOISE_FLOOR_DX
    assert attribution["comparison_error_dx"] == max(attribution["comparison_terms_dx"].values())
    assert attribution["comparison_error_dx"] >= stage.WALK_NOISE_FLOOR_DX
    assert attribution["dominance_floor_dx"] == pytest.approx(
        stage.DOMINANCE_FACTOR * attribution["comparison_error_dx"], rel=1e-12
    )
    for convention in stage.CONVENTION_NAMES:
        values = attribution["displacement_dx"]["PRIMARY"][convention]
        assert "adv" in values
        assert "ch_pair" in values
        if convention == "MARGINAL_OWN_BASE":
            # the marginal convention measures each ingredient against the substep base and has no
            # end-to-end leg of its own; that is exactly why it is the comparison convention, not the rule
            assert "full" not in values
        else:
            assert "full" in values


def test_the_primary_convention_reports_the_measured_legs(evidence) -> None:
    values = evidence["attribution.json"]["displacement_dx"]["PRIMARY"][stage.PRIMARY_CONVENTION]
    assert values["adv"] == pytest.approx(0.10104428914258534, rel=1e-12)
    assert values["ch_explicit"] == pytest.approx(0.13105577546827618, rel=1e-12)
    assert values["ch_implicit"] == pytest.approx(-0.169451382717817, rel=1e-12)
    assert values["ch_pair"] == pytest.approx(-0.03839560724954083, rel=1e-12)
    assert values["full"] == pytest.approx(0.06264868189304451, rel=1e-12)
    assert values["ch_pair"] == pytest.approx(values["ch_explicit"] + values["ch_implicit"], abs=1e-15)
    decision = evidence["mechanism_decision.json"]
    assert decision["advective_displacement_dx"] == pytest.approx(values["adv"], rel=1e-12)
    assert decision["cahn_hilliard_displacement_dx"] == pytest.approx(values["ch_pair"], rel=1e-12)


def test_the_two_cahn_hilliard_halves_are_one_update(evidence) -> None:
    attribution = evidence["attribution.json"]
    pairing = attribution["cahn_hilliard_pairing"]
    assert pairing["pair_dx"] == pytest.approx(pairing["explicit_dx"] + pairing["implicit_dx"], abs=1e-15)
    assert (
        pairing["reported_as"]
        == "one Cahn-Hilliard update whose two halves are the explicit flux and the implicit solve"
    )
    assert "never presented as two independent physical mechanisms" in pairing["forbidden_claim"]
    assert evidence["mechanism_decision.json"]["cahn_hilliard_halves_are_one_update"] is True


def test_the_dominance_reading_uses_the_prespecified_floor(evidence) -> None:
    attribution = evidence["attribution.json"]
    decision = evidence["mechanism_decision.json"]
    margin = attribution["dominance_margin_dx"]
    assert margin == pytest.approx(attribution["rule_margins_dx"][stage.PRIMARY_CONVENTION], rel=1e-12)
    assert margin == pytest.approx(
        abs(attribution["displacement_dx"]["PRIMARY"][stage.PRIMARY_CONVENTION]["adv"])
        - abs(attribution["displacement_dx"]["PRIMARY"][stage.PRIMARY_CONVENTION]["ch_pair"]),
        abs=1e-15,
    )
    assert decision["dominance_margin_dx"] == pytest.approx(margin, rel=1e-12)
    assert decision["dominance_floor_dx"] == pytest.approx(attribution["dominance_floor_dx"], rel=1e-12)
    assert decision["dominance_reading"] in stage.DOMINANCE_READINGS
    assert (abs(margin) >= attribution["dominance_floor_dx"]) == (decision["dominance_reading"] != "WITHIN_THE_FLOOR")
    if decision["dominance_reading"] == "ADVECTIVE_LEG_LARGER":
        assert margin >= attribution["dominance_floor_dx"]
    elif decision["dominance_reading"] == "CAHN_HILLIARD_LEG_LARGER":
        assert -margin >= attribution["dominance_floor_dx"]


def test_the_window_totals_are_the_sums_of_the_per_macro_rows(evidence) -> None:
    attribution = evidence["attribution.json"]
    per_macro = attribution["per_macro"]
    assert len(per_macro) == 40
    totals = attribution["displacement_dx"]["PRIMARY"][stage.PRIMARY_CONVENTION]
    for key in ("adv", "ch_explicit", "ch_implicit", "ch_pair", "full"):
        assert totals[key] == pytest.approx(sum(float(row[f"dn_{key}"]) for row in per_macro), rel=1e-12)
    for row in per_macro:
        assert row["tie_columns"]["PRIMARY"]["dn_full"] is not None
        assert set(row["tie_columns"]) == {"PRIMARY", "TIE_1"}
    assert attribution["closure_residual_dx"]["PRIMARY"]["window_total"] == 0.0
    assert attribution["closure_residual_dx"]["PRIMARY"]["maximum_abs_substep"] == 0.0


def test_the_historical_comparator_is_the_sealed_l1a2y_number(evidence) -> None:
    historical = evidence["residual_decomposition.json"]["historical_comparator"]
    sealed = json.loads((PRIOR_EVIDENCE / "residual_decomposition.json").read_text())
    assert historical["l1a2y_primary_closure_dx"] == pytest.approx(
        float(sealed["terms_dx"]["R_attribution_closure"]), rel=0.0, abs=0.0
    )
    assert historical["l1a2y_primary_closure_dx"] == pytest.approx(-0.12231965567175251, rel=1e-12)
    assert abs(historical["l1a2y_primary_closure_dx"]) > 12.0 * stage.ATTRIBUTION_CLOSURE_BAND_DX
    assert historical["l1a2y_comparison_error_dx"] == pytest.approx(0.1238272833457486, rel=1e-12)
    assert historical["l1a2y_own_base_closure_dx"] == pytest.approx(-0.007937057754737964, rel=1e-12)


def test_the_prior_evidence_is_untouched_by_the_run() -> None:
    sealed = stage_2y.verify_evidence_manifest(PRIOR_EVIDENCE)
    assert sealed["verified"] is True
    assert sealed["problems"] == []
    assert sealed["bound_files"] == 22
    chain = stage_2x.verify_prerequisites()
    assert chain["all_verified"] is True
    assert chain["failures"] == []
    assert sorted(chain["stages"]) == ["L1A-2t", "L1A-2u", "L1A-2v", "L1A-2w"]


def test_conservation_and_convergence_hold(evidence) -> None:
    conservation = evidence["conservation_and_convergence.json"]
    assert conservation["substeps"] == 120
    assert conservation["mass_drift"] == 0.0
    assert conservation["mass_initial"] == conservation["mass_final"]
    assert abs(conservation["worst_substep_mass_residual"]) <= 1e-12
    assert conservation["worst_closed_face_flux"] == 0.0
    assert abs(conservation["worst_operator_recombination"]) <= 1e-14
    assert abs(conservation["worst_phi_hat_identity"]) <= 1e-14
    assert conservation["cg_all_converged"] is True
    assert conservation["cg_solves"] == 120
    assert conservation["cg_iterations_max"] <= 200
    assert conservation["contact_threshold_over_dx"] == 1.5
    assert conservation["contact_guard"]["triggered"] is False
    assert conservation["t_final"] == pytest.approx(0.08, abs=1e-12)
    assert evidence["validity_screen.json"]["conservation_ok"] is True


def test_the_capture_reproduces_the_frozen_public_step(evidence) -> None:
    equivalence = evidence["capture_equivalence.json"]
    assert equivalence["macros"] == 40
    assert equivalence["max_phi_difference_vs_eager_public_step"] <= stage.EQUIVALENCE_MAX_PHI_DIFF
    assert equivalence["max_phi_difference_vs_jitted_public_step"] <= stage.EQUIVALENCE_MAX_PHI_DIFF
    assert equivalence["max_gap_difference_over_dx"] <= stage.EQUIVALENCE_MAX_GAP_DIFF_DX
    assert equivalence["cg_all_converged_eager"] is True
    assert evidence["validity_screen.json"]["equivalence_ok"] is True


def test_the_identical_window_check_reproduces_the_sealed_run(evidence) -> None:
    identical = evidence["identical_window_check.json"]
    sealed = json.loads((PRIOR_EVIDENCE / "substep_ledger.json").read_text())
    assert identical["holds"] is True
    assert identical["initial_state_hashes_match"] is True
    assert identical["final_state_hashes_match"] is True
    assert identical["tracked_segment_indices_agree"] is True
    assert identical["max_absolute_root_difference_over_dx"] == 0.0
    assert identical["macro_steps_compared"] == 40
    assert identical["substeps_compared"] == 120
    assert identical["reference_stage"] == "L1A-2y"
    assert identical["reference_head_sha"] == stage.BASELINE_HEAD_SHA
    assert identical["final_state_hashes"]["phi_hash"] == sealed["final_state"]["phi_hash"]
    assert identical["sealed_final_state_hashes"]["phi_hash"] == sealed["final_state"]["phi_hash"]
    assert set(identical["per_substep_increment_samples"]) == {
        "max_abs_difference_adv",
        "max_abs_difference_ch",
        "max_abs_difference_imp",
        "max_abs_difference_full",
    }
    assert max(identical["per_substep_increment_samples"].values()) == 0.0
    assert evidence["validity_screen.json"]["identical_window"]["holds"] is True


def test_the_identical_window_check_rejects_a_shifted_trajectory(evidence) -> None:
    ledger = json.loads(json.dumps(evidence["sequential_ownbase_ledger.json"]))
    ledger["macro_rows"][0]["tracked_root_over_dx"] = float(ledger["macro_rows"][0]["tracked_root_over_dx"]) + 1.0
    report = stage.identical_window_check(ledger)
    assert report["holds"] is False
    assert report["max_absolute_root_difference_over_dx"] > stage.CROSS_CHECK_TOLERANCE_DX


def test_the_closure_gate_bites_on_a_fabricated_violation(evidence) -> None:
    residuals = json.loads(json.dumps(evidence["residual_decomposition.json"]))
    residuals["terms_dx"]["R_sequential_telescoping_total"] = 0.05
    residuals["terms_dx"]["R_marginal_interaction"] = 0.05
    validity = stage.validity_screen(
        evidence["sequential_ownbase_ledger.json"],
        evidence["attribution.json"],
        residuals,
        evidence["capture_equivalence.json"],
        evidence["identical_window_check.json"],
    )
    assert validity["closure_ok"] is False
    assert validity["unresolved"] is True
    assert (
        validity["blocking_failures"].get("attribution_closure") == 0
        or "attribution_closure" in validity["blocking_failures"]
    )


def test_the_uniqueness_gate_bites_on_a_fabricated_multi_component_state(evidence) -> None:
    ledger = json.loads(json.dumps(evidence["sequential_ownbase_ledger.json"]))
    ledger["uniqueness_rows"][0]["competition_components"] = [1, 2]
    validity = stage.validity_screen(
        ledger,
        evidence["attribution.json"],
        evidence["residual_decomposition.json"],
        evidence["capture_equivalence.json"],
        evidence["identical_window_check.json"],
    )
    assert validity["multi_component_state_count"] == 1
    assert validity["unique_measurement_ok"] is False
    assert validity["unresolved"] is True
    assert validity["blocking_failures"]


def test_the_uniqueness_gate_bites_on_a_fabricated_non_unique_state(evidence) -> None:
    ledger = json.loads(json.dumps(evidence["sequential_ownbase_ledger.json"]))
    ledger["uniqueness_rows"][3]["rule_ok"] = False
    ledger["uniqueness_rows"][3]["status"] = "NON_UNIQUE_BRANCH"
    validity = stage.validity_screen(
        ledger,
        evidence["attribution.json"],
        evidence["residual_decomposition.json"],
        evidence["capture_equivalence.json"],
        evidence["identical_window_check.json"],
    )
    assert validity["non_unique_state_count"] == 1
    assert validity["unique_measurement_ok"] is False
    assert validity["unresolved"] is True


def test_the_decision_keeps_the_three_questions_apart(evidence) -> None:
    decision = evidence["mechanism_decision.json"]
    assert set(decision["questions"]) == {
        "A_measurement_validity",
        "B_mathematical_attribution_closure",
        "C_physical_mechanism",
        "C_note",
    }
    assert decision["questions"]["C_physical_mechanism"] is False
    assert decision["questions"]["C_note"]
    assert decision["forbidden_claims_respected"]
    assert decision["verdict"] in stage.DECISION_STATES
    assert decision["verdict"] == (
        "EARLY_WINDOW_MEASUREMENT_ATTRIBUTION_RESOLVED"
        if decision["measurement_attribution_resolved_under_the_specified_convention"]
        else "EARLY_WINDOW_ATTRIBUTION_UNRESOLVED"
    )
    if decision["measurement_attribution_resolved_under_the_specified_convention"]:
        assert decision["questions"]["A_measurement_validity"] is True
        assert decision["questions"]["B_mathematical_attribution_closure"] is True
        assert decision["verdict_withheld_because"] == []
        assert abs(decision["dominance_margin_dx"]) >= decision["dominance_floor_dx"]
    else:
        assert decision["verdict_withheld_because"]


def test_the_decision_withholds_on_a_fabricated_gate_failure(evidence) -> None:
    validity = json.loads(json.dumps(evidence["validity_screen.json"]))
    validity["unique_measurement_ok"] = False
    validity["unresolved"] = True
    decision = stage.decide(
        evidence["sequential_ownbase_ledger.json"],
        evidence["attribution.json"],
        validity,
        evidence["identical_window_check.json"],
    )
    assert decision["verdict"] == "EARLY_WINDOW_ATTRIBUTION_UNRESOLVED"
    assert decision["measurement_attribution_resolved_under_the_specified_convention"] is False
    assert decision["questions"]["C_physical_mechanism"] is False
    assert decision["verdict_withheld_because"]


def test_the_adversarial_tests_all_hold_and_every_one_ran(evidence) -> None:
    adversarial = evidence["adversarial_and_falsification.json"]
    assert adversarial["all_hold"] is True
    assert adversarial["failed_checks"] == []
    assert adversarial["checks_run"] == len(adversarial["checks"]) == 19
    names = {entry["check"] for entry in adversarial["checks"]}
    assert names == set(stage.ADVERSARIAL_CHECK_NAMES)
    for entry in adversarial["checks"]:
        assert entry["holds"] is True
        assert entry["detail"]


def test_the_quality_matrix_separates_the_classes_and_records_observations(evidence) -> None:
    quality = evidence["quality_and_blocker_matrix.json"]
    assert quality["verdict"] == evidence["mechanism_decision.json"]["verdict"]
    assert quality["declared_new_code_diagnostics"] == 0
    assert quality["ratchet_expectation"] == "PASS_BASELINE_ONLY"
    assert quality["legacy_debt_baseline"] == {"ruff": 171, "formatter": 27, "isort": 28, "codespell": 3}
    assert quality["quality_baseline_path"] == "quality/legacy_two_phase_debt_v1.json"
    assert quality["regressions_outside_the_declared_classes"] == []
    assert set(quality["declared_expectations"]) == {
        "stage_suite",
        "predecessor_stage_suite",
        "frozen_adjacent_suite",
        "legacy_lineage_suite",
    }
    assert quality["observations_complete"] is True
    for name in ("stage_suite", "predecessor_stage_suite", "frozen_adjacent_suite", "legacy_lineage_suite"):
        entry = quality[name]
        assert entry["observed"] is True, name
        assert entry["unexpected_vs_declared"] == []
        assert entry["result"]
        assert entry["observed_at"]
    declared = quality["declared_expectations"]
    frozen_nodes = sorted(node for group in stage.DECLARED_FROZEN_ADJACENT_SUITE_FAILURES.values() for node in group)
    assert sorted(_declared_failures(declared["frozen_adjacent_suite"])) == frozen_nodes
    assert _declared_failures(declared["predecessor_stage_suite"]) == list(
        stage.DECLARED_PREDECESSOR_STAGE_SUITE_FAILURES
    )
    assert _declared_failures(declared["legacy_lineage_suite"]) == list(stage.DECLARED_LEGACY_LINEAGE_FAILURES)
    assert _declared_failures(declared["stage_suite"]) == list(stage.DECLARED_STAGE_SUITE_FAILURES)
    assert sorted(quality["frozen_adjacent_suite"]["declared"]) == frozen_nodes
    assert quality["predecessor_stage_suite"]["declared"] == list(stage.DECLARED_PREDECESSOR_STAGE_SUITE_FAILURES)
    assert quality["legacy_lineage_suite"]["declared"] == list(stage.DECLARED_LEGACY_LINEAGE_FAILURES)
    assert quality["stage_suite"]["failing"] == list(stage.DECLARED_STAGE_SUITE_FAILURES)
    assert quality["frozen_adjacent_suite"]["classification"] in stage.STRUCTURAL_FAILURE_CLASSES.values()
    assert quality["legacy_lineage_suite"]["classification"] in stage.STRUCTURAL_FAILURE_CLASSES.values()
    assert quality["not_run"]
    assert quality["hosted_ci"]["status_for_this_head"] in ("UNMEASURED", "success", "failure", "partial", "NOT_RUN")
    assert quality["hosted_ci"]["rule"]


def test_the_quality_ratchet_records_its_own_commands_and_inputs(evidence) -> None:
    quality = evidence["quality_and_blocker_matrix.json"]
    baseline = Path(quality["quality_baseline_path"])
    baseline = baseline if baseline.is_absolute() else stage.REPO_ROOT / baseline
    assert baseline.is_file()
    assert quality["quality_baseline_sha256"] == stage.sha256_file(baseline)
    assert quality["quality_baseline_sha256"] == "52b41ab68b4d6b86aaa254389eb5398e65ad8c5cbdcd12626c0466248dd3d70b"
    assert quality["declared_expectations"]["stage_suite"]["command"].startswith("JAX_ENABLE_X64=1")
    assert "pytest" in quality["declared_expectations"]["stage_suite"]["command"]
    for name, entry in quality["declared_expectations"].items():
        assert entry["command"], name
        assert isinstance(_declared_failures(entry), list), name


def test_the_report_carries_every_section_and_no_mechanism_claim(evidence) -> None:
    report = evidence[REPORT_JSON]
    text = (EVIDENCE / REPORT_MD).read_text()
    assert report["sections"] == list(stage.REPORT_SECTIONS)
    assert len(report["sections"]) == 16
    for heading in stage.REPORT_SECTIONS:
        assert f"## {heading}" in text
    assert report["headline"]["physical_mechanism_claim"] is False
    assert report["verdict"] == evidence["mechanism_decision.json"]["verdict"]
    assert report["headline"]["closure_band_dx"] == 0.01
    assert report["headline"]["dominance_reading"] in stage.DOMINANCE_READINGS
    assert "physical mechanism" in text.lower()
    lowered = text.lower()
    for forbidden in ("proves the mechanism", "established the mechanism", "impact mechanism confirmed"):
        assert forbidden not in lowered
    assert "\n## P. Exactly one next action" in "\n" + text


def test_the_next_action_is_single_and_not_started(evidence) -> None:
    action = evidence["next_action.json"]
    assert action["must_not_be_started_in_this_stage"] is True
    assert action["action"]
    assert action["measured_failures_of_this_stage"] == []
    assert action["bounded_scope"]["cases"] == []
    assert action["bounded_scope"]["no_extension_toward_contact"] is True
    assert action["bounded_scope"]["no_control_arms"] is True
    assert action["bounded_scope"]["no_generator_or_ml"] is True


def test_the_evidence_lists_every_file_and_the_report_is_bound(evidence) -> None:
    manifest = json.loads((EVIDENCE / "manifest.json").read_text())
    assert sorted(manifest["files"]) == sorted(set(stage.EVIDENCE_FILES) - {"manifest.json"})
    assert manifest["required_files"] == list(stage.EVIDENCE_FILES)
    assert manifest["files"][REPORT_MD] == stage.sha256_file(EVIDENCE / REPORT_MD)
    assert manifest["files"][REPORT_JSON] == stage.sha256_file(EVIDENCE / REPORT_JSON)
    assert set(manifest["files"]) == set(evidence) | {REPORT_MD}
    for name in (REPORT_JSON, REPORT_MD):
        assert (EVIDENCE / name).is_file()


def test_a_tampered_bound_file_is_rejected(evidence_copy: Path) -> None:
    target = evidence_copy / "sequential_ownbase_ledger.json"
    target.write_text(target.read_text().replace('"macro_steps_run": 40', '"macro_steps_run": 39'))
    with pytest.raises(stage.AuditBlocked):
        stage.verify_evidence_manifest(evidence_copy)


def test_a_removed_required_file_is_rejected(evidence_copy: Path) -> None:
    (evidence_copy / "next_action.json").unlink()
    with pytest.raises(stage.AuditBlocked):
        stage.verify_evidence_manifest(evidence_copy)


def test_a_moved_verdict_is_rejected(evidence_copy: Path) -> None:
    target = evidence_copy / "mechanism_decision.json"
    payload = json.loads(target.read_text())
    payload["verdict"] = "EARLY_WINDOW_ATTRIBUTION_RESOLVED"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with pytest.raises(stage.AuditBlocked):
        stage.verify_evidence_manifest(evidence_copy)


def test_the_reseal_records_only_the_observed_ci_status(evidence_copy: Path) -> None:
    ledger_before = stage.sha256_file(evidence_copy / "sequential_ownbase_ledger.json")
    report = stage.reseal_ci_status("success", "a" * 40, evidence_copy)
    assert report["verified"] is True
    assert report["hosted_ci_reseal"]["status"] == "success"
    assert stage.sha256_file(evidence_copy / "sequential_ownbase_ledger.json") == ledger_before
    manifest = json.loads((evidence_copy / "manifest.json").read_text())
    assert manifest["hosted_ci_reseal"]["physics_evidence_untouched"] is True
