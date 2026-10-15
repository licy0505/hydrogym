"""Pre-measurement falsification for L1A-3c; no target trajectory is executed here."""

from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from production import l1a3c_zero_mobility_control_audit as audit


@pytest.fixture(scope="module")
def synthetic():
    bundle, state = audit.synthetic_bundle()
    result, ingredient = audit.capture.substep_capture(*state, bundle)
    return bundle, state, audit.pf.State(*result), ingredient


def test_zero_limit_consistency_and_conservation(synthetic):
    bundle, state, new, ingredient = synthetic
    assert audit.zero_checks(bundle, state, ingredient, new)["holds"]
    assert np.array_equal(np.asarray(new.phi), np.asarray(ingredient["phi_hat"]))
    assert int(ingredient["solve_info"].iterations) == 0


def test_full_admissibility_on_synthetic_not_target():
    result = audit.admissibility()
    assert result["holds"]
    assert result["target_trajectories_run"] == 0
    assert result["checks"]["momentum_and_capillary_retained"]


@pytest.mark.parametrize("mutation", ["explicit", "implicit", "mass", "solver"])
def test_wrong_zero_limit_rejected(synthetic, mutation):
    bundle, state, new, original = synthetic
    ingredient = dict(original)
    if mutation == "explicit":
        ingredient["ch_x"] = np.asarray(original["ch_x"]) + 1e-4
    elif mutation == "implicit":
        ingredient["phi_hat"] = np.asarray(original["phi_hat"]) + 1e-4
    elif mutation == "mass":
        new = new._replace(phi=new.phi + 1e-4)
    else:
        ingredient["solve_info"] = audit.pf.ImplicitSolveInfo(1, 0.01, False)
    assert not audit.zero_checks(bundle, state, ingredient, new)["holds"]


@pytest.mark.parametrize("delete", ["capillary", "momentum"])
def test_deleted_coupling_is_discriminated(monkeypatch, delete):
    original = audit.pf.rhs

    def wrong(state, solid, p):
        values = list(original(state, solid, p))
        if delete == "capillary":
            cap = (audit.pf.SIGMA_NORM / p.We) * values[3] * audit.pf._ddx(state.phi, p.dx) / p.rho_l
            values[1] = (values[1] - cap).astype(p.dtype)
        else:
            values[1] = values[1] * 0
        return tuple(values)

    monkeypatch.setattr(audit.pf, "rhs", wrong)
    assert not audit.admissibility()["checks"]["momentum_and_capillary_retained"]


@pytest.fixture
def intervention(synthetic):
    bundle, state, _, _ = synthetic
    base = audit.configuration(replace(bundle.p, M=0.002))
    control = audit.configuration(bundle.p)
    hashes = audit.hashes(state)
    return base, control, hashes


def test_only_mobility_changes(intervention):
    b, c, h = intervention
    audit.intervention_gate(b, c, h, h, "geometry", "geometry")


@pytest.mark.parametrize("mutation", ["initial", "geometry", "We", "dt", "mobility", "wetting", "pressure_dtype"])
def test_changed_unrelated_configuration_rejected(intervention, mutation):
    b, c, h = intervention
    other = dict(h)
    geometry = "geometry"
    if mutation == "initial":
        other["phi"] = "different"
    elif mutation == "geometry":
        geometry = "different"
    elif mutation == "We":
        c["We"] = 200
    elif mutation == "dt":
        c["dt"] = 0.001
    elif mutation == "mobility":
        c["M"] = 0.0001
    elif mutation == "wetting":
        c["wetting_model"] = "none"
    else:
        c["dtype"] = "float64"
    with pytest.raises(audit.Blocked):
        audit.intervention_gate(b, c, h, other, "geometry", geometry)


def effect(b=None, c=None):
    return audit.observable.paired_effect_vector(
        {"RAY_95": 10.0, "RAY_96": 10.0},
        b or {"RAY_95": 9.8, "RAY_96": 9.8},
        c or {"RAY_95": 9.6, "RAY_96": 9.9},
        baseline_initial_hashes={"phi": "same"},
        control_initial_hashes={"phi": "same"},
    )


def test_sign_identity_opposite_signs_and_large_disagreement():
    e = effect(c={"RAY_95": 8.0, "RAY_96": 11.0})
    assert e["E_dx"]["RAY_95"] == pytest.approx(1.8)
    assert e["E_dx"]["RAY_96"] == pytest.approx(-1.2)
    assert e["effects_have_opposite_signs"]
    assert e["absolute_support_to_support_effect_disagreement_dx"] == pytest.approx(3)
    assert e["raw_signed_effect_range_dx"] == pytest.approx([-1.2, 1.8])
    assert not e["scalar_mean_or_preferred_support_reported"]
    assert e["solver_state_uncertainty_bound_dx"] is None


def test_invalid_readout_does_not_drop_ray():
    r = {
        "status": "INVALID",
        "blocking_failures": ["RAY_95:NO_ROOT_IN_WINDOW"],
        "Q_dx": {"RAY_95": None, "RAY_96": 9.9},
    }
    result = audit.observable.paired_effect_audit(
        {"RAY_95": 10.0, "RAY_96": 10.0}, r, r, baseline_initial_hashes={}, control_initial_hashes={}
    )
    assert result["E_dx"] is None
    assert result["validity_failures"]


@pytest.mark.parametrize(
    "case",
    [
        "missing_root_fails_closed",
        "multiple_roots_on_one_ray_fail_closed",
        "branch_fragmentation_or_correspondence_loss_detected",
        "swapped_fixed_ray_identity_rejected",
        "candidate_enumeration_permutation_invariant",
        "ray_order_permutation_equivariant",
        "signed_displacement_convention_checked",
        "scalar_mean_cannot_hide_disagreement",
    ],
)
def test_frozen_observable_falsification_classes(case):
    assert audit.observable._synthetic_adversarial_checks()[case]["holds"]


def test_cross_trajectory_geometric_loss_not_label_comparison():
    p = SimpleNamespace(dx=0.03125, Lx=6.0, Ly=6.0)
    before = np.asarray([[3.0, 0.7], [3.01, 0.71]])
    assert audit.observable.geometric_branch_correspondence(before, before[::-1], p)["holds"]
    assert not audit.observable.geometric_branch_correspondence(before, before + 1.0, p)["holds"]


def valid_report():
    return {
        "physical_status": audit.PHYSICAL_STATUS,
        "verdict": "CONTROL_PAIRED_RESPONSE_RECORDED",
        "decision_inputs": {"admissible": True, "protocol": True, "executed": True, "valid": True},
        "total_uncertainty": None,
        "support_independence": "UNRESOLVED_NO_JUSTIFIED_BINARY_THRESHOLD",
        "scalar_effect": None,
    }


@pytest.mark.parametrize("mutation", ["average", "uncertainty", "invariance", "failed_gate", "physical", "resolved"])
def test_invalid_result_promotion_rejected(mutation):
    report = valid_report()
    if mutation == "average":
        report["scalar_effect"] = 0.15
    elif mutation == "uncertainty":
        report["total_uncertainty"] = 1.8e-9
    elif mutation == "invariance":
        report["support_independence"] = "AGREES"
    elif mutation == "failed_gate":
        report["decision_inputs"]["protocol"] = False
    elif mutation == "physical":
        report["physical_status"] = "ESTABLISHED"
    else:
        report["verdict"] = "EARLY_WINDOW_MOBILITY_SENSITIVITY_RESOLVED"
    with pytest.raises(audit.Blocked):
        audit.validate_report(report)


@pytest.mark.parametrize(
    "a,p,e,v,expected",
    [
        (False, True, False, False, "ZERO_MOBILITY_ADMISSIBILITY_BLOCKED"),
        (True, False, False, False, "CONTROL_PROTOCOL_OR_BASELINE_BLOCKED"),
        (True, True, True, False, "CONTROL_OBSERVABLE_INVALID"),
        (True, True, True, True, "CONTROL_PAIRED_RESPONSE_RECORDED"),
    ],
)
def test_verdict_logic(a, p, e, v, expected):
    assert audit.verdict(a, p, e, v) == expected


def test_baseline_reconstruction_failures_rejected():
    with pytest.raises(audit.Blocked, match="incomplete"):
        audit.baseline_compare([], {}, {})
    with pytest.raises(audit.Blocked, match="initial identity"):
        audit.baseline_compare([{}] * 360, {}, {})
    h = audit.load(audit.TP / "evidence/l1a2z/initial_state_identity.json")["hashes"]
    with pytest.raises(audit.Blocked, match="final-state"):
        audit.baseline_compare([{}] * 360, h, {})


def test_sealed_baseline_endpoints_and_prerequisites():
    assert audit.prerequisites()["l1a3b_content_verified"]
    assert {r.column_index for r in audit.rays()} == {95, 96}


def test_tampered_evidence_and_source_mismatch_fail_closed(tmp_path, monkeypatch):
    report = valid_report()
    audit.write(tmp_path / "report.json", report)
    manifest = {
        "stage": "L1A-3c",
        "source_hashes": {"not": "matching"},
        "files": {"report.json": audit.sha(tmp_path / "report.json")},
    }
    audit.write(tmp_path / "manifest.json", manifest)
    with pytest.raises(audit.Blocked, match="source hash"):
        audit.verify(tmp_path)
    monkeypatch.setattr(audit, "sources", lambda: manifest["source_hashes"])
    audit.write(tmp_path / "report.json", {"tampered": True})
    with pytest.raises(audit.Blocked, match="tampered evidence"):
        audit.verify(tmp_path)


def test_receipt_is_exclusive(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "EVIDENCE", tmp_path)
    audit.execution_receipt("control")
    with pytest.raises(FileExistsError):
        audit.execution_receipt("control")


def test_unfrozen_source_cannot_execute(monkeypatch):
    def denied():
        raise audit.Blocked("unfrozen source")

    monkeypatch.setattr(audit, "freeze", denied)
    with pytest.raises(audit.Blocked):
        audit.execute("control")


def test_paired_initial_mismatch_and_sign_inversion_fail():
    with pytest.raises(audit.observable.AuditBlocked):
        audit.observable.paired_effect_vector(
            {"RAY_95": 1.0, "RAY_96": 1.0},
            {"RAY_95": 0.5, "RAY_96": 0.5},
            {"RAY_95": 0.4, "RAY_96": 0.4},
            baseline_initial_hashes={"phi": "a"},
            control_initial_hashes={"phi": "b"},
        )
    result = effect()
    inverted = {k: -v for k, v in result["E_dx"].items()}
    assert inverted != result["coordinate_identity_baseline_minus_control_dx"]


def test_no_target_experiment_from_test_runner(monkeypatch):
    # Receipt calls are the last mandatory operation before target evolution.
    monkeypatch.setattr(audit, "execution_receipt", lambda arm: pytest.fail(f"unexpected trajectory {arm}"))
    assert audit.admissibility()["target_trajectories_run"] == 0


def test_signed_effect_or_averaging_mutation_is_rejected():
    e = effect()
    audit.validate_effect(e)
    for mutation in ("inversion", "average", "range", "difference"):
        wrong = copy.deepcopy(e)
        if mutation == "inversion":
            wrong["E_dx"] = {k: -v for k, v in e["E_dx"].items()}
        elif mutation == "average":
            mean = sum(e["E_dx"].values()) / 2
            wrong["E_dx"] = {k: mean for k in e["E_dx"]}
        elif mutation == "range":
            wrong["raw_signed_effect_range_dx"] = [0, 0]
        else:
            wrong["absolute_support_to_support_effect_disagreement_dx"] = 0
        with pytest.raises(audit.Blocked):
            audit.validate_effect(wrong)


def test_bad_quality_gate_cannot_authorize_target():
    with pytest.raises(audit.Blocked, match="focused"):
        audit.quality_gate({})
    with pytest.raises(audit.Blocked, match="ratchet"):
        audit.quality_gate({"focused_passed": 44, "focused_failures": []})


def test_baseline_intermediate_hash_mutation_is_rejected():
    sealed = audit.load(audit.TP / "evidence/l1a2z/sequential_ownbase_ledger.json")
    initial = audit.load(audit.TP / "evidence/l1a2z/initial_state_identity.json")["hashes"]
    final = {k: sealed["final_state"][f"{k}_hash"] for k in ("phi", "u", "v")}
    records = [{"substep_index": 0, "state_name": "s1", "phi_hash": "tampered"}] * 360
    with pytest.raises(audit.Blocked, match="reconstruction hash"):
        audit.baseline_compare(records, initial, final)
