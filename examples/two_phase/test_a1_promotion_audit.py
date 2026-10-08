"""Contract tests for Stage-2 evidence aggregation and readiness separation."""

import json

from production.a1_promotion_audit import (
    PHYSICS_REQUIRED_GATES,
    build_stage2_report,
    load_chns_closure,
    load_physics_evidence,
    _merge_stage2_mass_gates,
)


def test_unmeasured_physics_and_scale_stay_unverified():
    report = build_stage2_report(
        quick={"status": "PASS"},
        dtype={"status": "PASS"},
        dataset={"status": "PASS"},
        restart={"status": "PASS"},
        chns={"status": "NOT_RUN", "blocker": "no report"},
        physics={"status": "NOT_RUN", "missing_gates": list(PHYSICS_REQUIRED_GATES)},
        performance={"status": "MEASURED"},
        gpu={"status": "UNAVAILABLE", "reason": "no GPU"},
    )
    assert report["readiness"]["CONTRACT11_LINEAGE_READY"] == "PASS"
    assert report["readiness"]["PHYSICS_NUMERICS_READY"] == "NOT_RUN"
    assert report["readiness"]["WETTING_CLOSURE_READY"] == "NOT_RUN"
    assert report["readiness"]["GPU_PRODUCTION_SCALE_READY"] == "UNAVAILABLE"


def test_fresh_quick_and_medium_measurements_are_hashed_into_physics_ledger(tmp_path):
    quick_path = tmp_path / "quick.json"
    medium_path = tmp_path / "medium.json"
    quick_path.write_text('{"status":"PASS"}', encoding="utf-8")
    medium_path.write_text('{"status":"PASS"}', encoding="utf-8")
    initial = {"status": "NOT_RUN", "gates": {}, "artifact_hashes": {}}
    merged = _merge_stage2_mass_gates(
        initial,
        {"status": "PASS", "rows": {}},
        {"status": "PASS", "rows": {}},
        quick_path,
        medium_path,
    )
    assert merged["gates"]["ch_only_quick"]["status"] == "PASS"
    assert merged["gates"]["ch_only_medium"]["status"] == "PASS"
    assert merged["gates"]["ch_only_quick"]["artifacts"][0]["sha256"]
    assert merged["gates"]["ch_only_medium"]["artifacts"][0]["sha256"]
    assert "ch_only_quick" not in merged["missing_gates"]
    assert "ch_only_medium" not in merged["missing_gates"]
    assert merged["status"] == "NOT_RUN"  # all other required physics gates are still absent


def test_physics_gate_requires_every_measured_gate_and_artifact(tmp_path):
    evidence_file = tmp_path / "measurement.json"
    evidence_file.write_text("measured artifact", encoding="utf-8")
    report_path = tmp_path / "physics.json"
    gates = {
        name: {"status": "PASS", "artifacts": [evidence_file.name], "measurement": {"value": 0.0}}
        for name in PHYSICS_REQUIRED_GATES
    }
    report_path.write_text(json.dumps({"gates": gates}), encoding="utf-8")

    report = load_physics_evidence(str(report_path), root=tmp_path)
    assert report["status"] == "PASS"
    assert set(report["gates"]) == set(PHYSICS_REQUIRED_GATES)
    assert report["artifact_hashes"][evidence_file.name]

    gates.pop(PHYSICS_REQUIRED_GATES[-1])
    report_path.write_text(json.dumps({"gates": gates}), encoding="utf-8")
    incomplete = load_physics_evidence(str(report_path), root=tmp_path)
    assert incomplete["status"] == "NOT_RUN"
    assert incomplete["missing_gates"] == [PHYSICS_REQUIRED_GATES[-1]]


def test_chns_closure_requires_baseline_profile_and_current_contract_lineage(tmp_path):
    from production.run_validation import _contact_angle_acceptance

    def _cases(solver_contract_version):
        return [
            {
                "target_deg": target,
                "measured_deg": target,
                "finite": True,
                "converged": True,
                "mass_relative_drift": 0.0,
                "total_mass_relative_drift": 0.0,
                "max_solid_liquid_fraction": 0.0,
                "phase_boundary_model": "impermeable_flux",
                "enforce_solid_phi": False,
                "wall_measure_method": "sdf_cutcell_v1",
                "runtime": {
                    "solver_contract_version": solver_contract_version,
                    "phase_storage_model": "phase_only_float64_v1",
                    "phase_state_dtype": "float64",
                    "velocity_state_dtype": "float32",
                    "N": 128,
                    "steps": 600 if target == 90.0 else 3000,
                    "max_steps": 8000,
                },
                "samples": [{"max_speed": 0.0}],
            }
            for target in (60.0, 90.0, 120.0, 150.0)
        ]

    import phasefield as pf

    current_contract = int(pf.SOLVER_CONTRACT_VERSION)
    cases = _cases(current_contract)
    benchmarks = {"contact_angle": {"cases": cases}}
    accepted, evidence = _contact_angle_acceptance(benchmarks)
    assert accepted, evidence

    # Explicit negative control: evidence generated under the previous contract is
    # NOT silently accepted as current-contract production evidence.
    stale_benchmarks = {"contact_angle": {"cases": _cases(current_contract - 1)}}
    stale_accepted, stale_evidence = _contact_angle_acceptance(stale_benchmarks)
    assert not stale_accepted
    assert stale_evidence["storage_lineage_ok"] is not True

    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"config": {"profile": "baseline"}, "benchmarks": benchmarks}))
    closure = load_chns_closure(str(baseline), root=tmp_path)
    assert closure["status"] == "PASS"
    assert closure["acceptance"] is True

    stale_baseline = tmp_path / "baseline-stale-contract.json"
    stale_baseline.write_text(json.dumps({"config": {"profile": "baseline"}, "benchmarks": stale_benchmarks}))
    stale_closure = load_chns_closure(str(stale_baseline), root=tmp_path)
    assert stale_closure["status"] == "FAIL"
    assert stale_closure["acceptance"] is False

    ci = tmp_path / "ci.json"
    ci.write_text(json.dumps({"config": {"profile": "ci"}, "benchmarks": benchmarks}))
    not_baseline = load_chns_closure(str(ci), root=tmp_path)
    assert not_baseline["status"] == "FAIL"
    assert not_baseline["acceptance"] is True
