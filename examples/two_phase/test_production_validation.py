"""Contract and observable tests for the L1A-1 validation framework."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from production.config import (
    ValidationConfig,
    canonical_config_json,
    config_sha256,
    load_config,
    validate_config_data,
)
from production.convergence import (
    adjacent_relative_changes,
    run_grid_refinement_study,
    run_interface_thickness_study,
)
from production.observables import (
    beta_from_width,
    bottom_gap,
    center_of_mass,
    contact_signal,
    fluid_phase_mass,
    periodic_spreading_width,
    total_phase_mass,
)
from production.report import REPORT_SCHEMA_VERSION, validate_report_schema, write_report_atomic

import phasefield as pf  # noqa: E402

HERE = Path(__file__).resolve().parent


def _minimal_report() -> dict:
    return {
        "validation_report_schema_version": REPORT_SCHEMA_VERSION,
        "physics_status": "BASELINE_ONLY",
        "contract_status": "PASS",
        "repository": {
            "git_sha": "a" * 40,
            "solver_contract_version": 5,
            "phasefield_sha256": "b" * 64,
            "validation_code_sha256": "c" * 64,
        },
        "runtime": {
            "python_version": "3.12.0",
            "jax_version": "0.0.test",
            "jax_backend": "cpu",
            "devices": ["TFRT_CPU_0"],
            "platform": "test",
        },
        "config": {"profile": "ci", "sha256": "d" * 64, "path": "ci.json"},
        "benchmarks": {
            "static_droplet": {},
            "contact_angle": {},
            "impact": {},
            "convergence": {},
        },
        "known_solver_blockers": [],
        "provisional_readiness_targets": {},
        "notes": [],
    }


def test_periodic_spreading_width_across_seam_is_four_cells():
    nx, ny, dx = 64, 12, 0.1
    phi = np.zeros((nx, ny), dtype=np.float32)
    phi[[62, 63, 0, 1], :] = 1.0

    width = periodic_spreading_width(phi, threshold=0.5, dx=dx, Lx=nx * dx)

    assert width == pytest.approx(4.0 * dx)
    assert width < (nx * dx) / 10.0


def test_periodic_spreading_width_single_and_full_domain_edges():
    phi = np.zeros((64, 8), dtype=np.float32)
    phi[9, :] = 1.0
    assert periodic_spreading_width(phi, 0.5, 0.25, 16.0) == pytest.approx(0.25)
    phi[:, 0] = 0.5
    assert periodic_spreading_width(phi, 0.5, 0.25, 16.0) == pytest.approx(16.0)
    assert periodic_spreading_width(np.zeros_like(phi), 0.5, 0.25, 16.0) == 0.0


def test_periodic_center_of_mass_stays_at_domain_seam():
    nx, ny, dx, dy = 64, 20, 0.1, 0.2
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    phi = np.zeros((nx, ny), dtype=np.float64)
    phi[[0, 1, nx - 2, nx - 1], 7:13] = 1.0
    sdf = np.ones_like(phi)

    x_cm, y_cm = center_of_mass(phi, sdf, x, y)

    Lx = nx * dx
    assert min(x_cm, Lx - x_cm) < 0.5 * dx
    assert y_cm == pytest.approx(float(y[7:13].mean()))
    assert abs(x_cm - Lx / 2.0) > Lx / 3.0


def test_contact_gap_and_contact_signal_use_sdf():
    phi = np.zeros((5, 6), dtype=np.float32)
    sdf = np.full_like(phi, 1.0)
    phi[2, 1] = 0.8
    sdf[2, 1] = 0.1
    phi[2, 3] = 0.6
    sdf[2, 3] = 0.4

    assert bottom_gap(phi, sdf, threshold=0.5) == pytest.approx(0.1)
    assert contact_signal(phi, sdf, dx=0.1, phi_threshold=0.5, gap_cells=1.5)
    sdf[2, 1] = 0.2
    assert bottom_gap(phi, sdf, threshold=0.5) == pytest.approx(0.2)
    assert not contact_signal(phi, sdf, dx=0.1, phi_threshold=0.5, gap_cells=1.5)


def test_mass_helpers_distinguish_total_from_fluid_region():
    phi = np.ones((4, 3), dtype=np.float32)
    sdf = np.ones_like(phi)
    sdf[:2, :] = -1.0
    dx, dy = 0.2, 0.5
    assert total_phase_mass(phi, dx, dy) == pytest.approx(12 * dx * dy)
    assert fluid_phase_mass(phi, sdf, dx, dy) == pytest.approx(6 * dx * dy)


def test_beta_is_width_over_initial_diameter():
    assert beta_from_width(width=1.2, R=0.3) == pytest.approx(2.0)


def test_report_schema_valid_missing_field_and_nan():
    report = _minimal_report()
    assert validate_report_schema(report) == []

    missing = dict(report)
    del missing["runtime"]
    assert any("runtime" in error for error in validate_report_schema(missing))

    nonfinite = _minimal_report()
    nonfinite["benchmarks"]["static_droplet"]["bad_metric"] = float("nan")
    assert any("non-finite" in error for error in validate_report_schema(nonfinite))


@pytest.mark.parametrize(
    "version, valid",
    [
        (4, True),
        (5, True),
        (6, True),
        (7, True),
        (8, True),
        (9, True),
        (10, True),
        (3, False),
        (11, False),
        (True, False),
        ("5", False),
        (5.0, False),
        (None, False),
    ],
)
def test_report_schema_solver_contract_lineage_fails_closed(version, valid):
    """Historical (v4-v9) and current (v10) reports validate; unknown contracts fail closed."""
    report = _minimal_report()
    report["repository"]["solver_contract_version"] = version
    contract_errors = [e for e in validate_report_schema(report) if "solver_contract_version" in e]
    assert (not contract_errors) is valid


def test_atomic_report_writer_emits_strict_json(tmp_path):
    path = tmp_path / "nested" / "report.json"
    write_report_atomic(_minimal_report(), path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["physics_status"] == "BASELINE_ONLY"
    assert path.parent.is_dir()


def test_config_hash_is_independent_of_dict_insertion_order():
    first = {
        "profile": "ci",
        "dtype": "float32",
        "benchmarks": {"static_droplet": {"N": 48, "steps": 2}},
        "output": {"directory": "out"},
    }
    second = {
        "output": {"directory": "out"},
        "benchmarks": {"static_droplet": {"steps": 2, "N": 48}},
        "dtype": "float32",
        "profile": "ci",
    }
    assert canonical_config_json(first) == canonical_config_json(second)
    assert config_sha256(first) == config_sha256(second)


def test_all_shipped_profiles_validate():
    for name in ("ci.json", "baseline.json", "convergence.example.json"):
        config = load_config(HERE / "production" / "configs" / name)
        assert isinstance(config, ValidationConfig)
        assert len(config_sha256(config)) == 64


def test_config_rejects_unknown_keys_and_unreasonable_values():
    data = {
        "profile": "ci",
        "dtype": "float32",
        "benchmarks": {"static_droplet": {"radii": [0.5], "N": 48, "steps": 2}},
        "output": {"directory": "out", "surprise": True},
    }
    assert any("Unknown key" in error for error in validate_config_data(data))
    data["output"] = {"directory": "out"}
    data["benchmarks"]["static_droplet"]["N"] = -1
    assert any("positive" in error for error in validate_config_data(data))


def test_adjacent_convergence_relative_change():
    assert adjacent_relative_changes([1.0, 0.9, 0.89]) == pytest.approx([1 / 9, 0.01 / 0.89])


def test_convergence_runners_separate_coupled_and_fixed_physical_modes():
    measure = lambda case: {"observable": float(case["N"])}
    coupled = run_grid_refinement_study(
        {"R": 0.7, "eps_factor": 2.0},
        [96, 128],
        measure,
        eps_factor=2.0,
    )
    assert coupled.study_type == "coupled_grid_interface_refinement"
    assert coupled.points[0].parameters["eps_over_dx"] == pytest.approx(2.0)
    assert coupled.points[0].parameters["Cn"] > coupled.points[1].parameters["Cn"]

    fixed = run_grid_refinement_study(
        {"R": 0.7},
        [96, 128],
        measure,
        eps_mode="fixed_physical",
        fixed_eps=0.1,
    )
    assert fixed.study_type == "fixed_physical_eps_grid_refinement"
    assert fixed.points[0].parameters["eps"] == fixed.points[1].parameters["eps"] == pytest.approx(0.1)
    assert fixed.points[1].parameters["eps_over_dx"] > fixed.points[0].parameters["eps_over_dx"]

    thickness = run_interface_thickness_study(
        {"R": 0.7},
        [1.5, 2.5],
        measure,
        N=128,
    )
    assert thickness.study_type == "interface_thickness_sensitivity"
    assert thickness.points[0].parameters["dx"] == thickness.points[1].parameters["dx"]
    assert thickness.points[1].parameters["eps"] > thickness.points[0].parameters["eps"]


def test_tiny_static_droplet_execution_has_finite_metrics_and_positive_mass():
    pytest.importorskip("jax")
    from production.validation import run_static_droplet_case

    result = run_static_droplet_case(
        R=0.6,
        N=32,
        steps=1,
        We=100.0,
        Re=200.0,
        eps_factor=2.0,
        save_every=1,
        dt=0.001,
        dtype="float32",
    )
    assert result.finite
    assert result.mass_initial > 0.0
    assert result.mass_final > 0.0
    assert np.isfinite([result.mass_relative_drift, result.max_speed_peak, result.delta_p, result.laplace_ratio]).all()
    assert result.pressure_diagnostic_method == "projection_reconstructed"
    assert len(result.time_series["time"]) == 2


# ---------------------------------------------------------------------------
#  L1A-2a: capillary sign consistency, evidence-gated blocker, before/after guards
# ---------------------------------------------------------------------------


def test_laplace_pressure_increases_with_curvature():
    """Smaller drops (larger curvature 1/R) have a larger *positive* projection-pressure jump."""
    pytest.importorskip("jax")
    from production.validation import run_static_droplet_case

    cases = {
        radius: run_static_droplet_case(
            R=radius,
            N=64,
            steps=60,
            We=100.0,
            Re=200.0,
            eps_factor=2.0,
            save_every=60,
            dt=0.002,
            dtype="float32",
        )
        for radius in (0.7, 1.0)
    }
    small, large = cases[0.7], cases[1.0]
    assert small.laplace_ratio > 0.0 and large.laplace_ratio > 0.0  # signed: never abs()
    assert small.delta_p > large.delta_p > 0.0  # smaller R -> larger positive delta_p
    assert 1.2 < small.delta_p / large.delta_p < 1.7  # ideal 1/0.7 = 1.43; coarse-grid tolerance
    for case in cases.values():
        assert 0.7 < case.laplace_ratio < 1.3  # scale sanity on a coarse grid, not an accuracy gate


def test_capillary_audit_reports_consistent_conventions():
    pytest.importorskip("jax")
    import phasefield as pf
    from production.capillary_audit import CONVENTIONS, run_audit

    assert {
        "phase_orientation",
        "outward_normal",
        "chemical_potential",
        "korteweg_force",
        "pressure_gradient",
        "laplace_jump",
    } <= set(CONVENTIONS)
    result = run_audit(N=64)
    assert [check.name for check in result.checks if not check.passed] == []
    assert result.diagnosis["three_conventions_consistent"] is True
    assert result.to_dict()["settings"]["solver_contract_version"] == pf.SOLVER_CONTRACT_VERSION == 10


def test_capillary_audit_detects_a_flipped_force_sign(monkeypatch):
    """Mutation check: the audit is not vacuous -- a sign regression is caught and blamed on the force."""
    pytest.importorskip("jax")
    import phasefield as pf
    from production.capillary_audit import run_audit

    real_rhs = pf.rhs

    def flipped(state, solid, params):
        phi_rhs, u_rhs, v_rhs, mu, mu_expl = real_rhs(state, solid, params)
        return phi_rhs, -u_rhs, -v_rhs, mu, mu_expl  # at rest: exactly the contract-v4 force

    monkeypatch.setattr(pf, "rhs", flipped)
    result = run_audit(N=64)
    failed = {check.name for check in result.checks if not check.passed}
    assert {"korteweg_force_direction", "laplace_jump_sign_and_scale", "free_energy_consistency"} <= failed
    assert result.diagnosis["force_convention_consistent"] is False
    assert result.diagnosis["projection_convention_consistent"] is True
    assert result.diagnosis["diagnostic_pressure_convention_consistent"] is True


def _static_benchmarks(ratios, *, r_squared=0.9996, slope=0.0109, extra_failed_record=False):
    radii = [0.6, 0.8, 1.0, 1.2][: len(ratios)]
    cases = [{"R": radius, "finite": True, "laplace_ratio": ratio} for radius, ratio in zip(radii, ratios)]
    if extra_failed_record:
        cases.append({"R": 1.4, "finite": False, "error": "FloatingPointError: injected"})
    return {
        "static_droplet": {"cases": cases, "summary": {"r_squared": r_squared, "slope_delta_p_vs_inv_R": slope}},
        "contact_angle": {"summary": None, "cases": []},
        "impact": {"cases": []},
        "convergence": {},
    }


def _blockers(benchmarks):
    from production.run_validation import _assessed_blockers

    return {item["id"]: item for item in _assessed_blockers(benchmarks)}


def test_laplace_sign_blocker_is_closed_only_with_evidence():
    good = _blockers(_static_benchmarks([1.0459, 1.0194, 1.0083, 1.0032]))
    assert good["P-LAPLACE-SIGN"]["status"] == "resolved_in_contract_v5"
    assert good["P-LAPLACE-SIGN"]["evidence"]["all_ratios_positive"] is True
    assert good["P-LAPLACE-SIGN"]["evidence"]["scaling_correct"] is True
    assert good["P-LAPLACE-SIGN"]["evidence"]["provisional_laplace_target_met"] is True
    # the closed sign blocker must not close anything else
    for name in ("P-VARDENS-PROJ", "P-CAP-RHO", "N-DT", "P-VARVISC", "BC-Y-PERIODIC"):
        assert good[name]["status"] == "open"

    wrong_sign = _blockers(_static_benchmarks([-1.0459, -1.0194, -1.0083, -1.0032], slope=-0.0109))
    assert wrong_sign["P-LAPLACE-SIGN"]["status"] == "confirmed_problem"
    one_bad = _blockers(_static_benchmarks([1.0459, 1.0194, -0.01, 1.0032]))
    assert one_bad["P-LAPLACE-SIGN"]["status"] == "confirmed_problem"

    too_few = _blockers(_static_benchmarks([1.0459, 1.0194], r_squared=1.0))
    assert too_few["P-LAPLACE-SIGN"]["status"] == "measurement_required"  # 2 radii cannot show 1/R scaling
    missing_case = _blockers(_static_benchmarks([1.0459, 1.0194, 1.0083], extra_failed_record=True))
    assert missing_case["P-LAPLACE-SIGN"]["status"] == "measurement_required"

    bad_scaling = _blockers(_static_benchmarks([1.0459, 1.0194, 1.0083, 1.0032], r_squared=0.9))
    assert bad_scaling["P-LAPLACE-SIGN"]["status"] == "confirmed_problem"
    negative_slope = _blockers(_static_benchmarks([1.0459, 1.0194, 1.0083, 1.0032], slope=-0.0109))
    assert negative_slope["P-LAPLACE-SIGN"]["status"] == "confirmed_problem"

    # sign fixed and 1/R holds, but the provisional 5 % magnitude goal is missed: closed, and honestly labelled
    coarse = _blockers(_static_benchmarks([1.2, 1.1, 1.05, 1.02]))
    assert coarse["P-LAPLACE-SIGN"]["status"] == "resolved_in_contract_v5"
    assert coarse["P-LAPLACE-SIGN"]["evidence"]["provisional_laplace_target_met"] is False


def test_report_schema_allows_resolved_blocker_status_only_when_known():
    report = _minimal_report()
    blocker = {"id": "P-LAPLACE-SIGN", "severity": "high", "description": "d", "status": "resolved_in_contract_v5"}
    report["known_solver_blockers"] = [blocker]
    assert validate_report_schema(report) == []
    blocker["status"] = "resolved_in_contract_v6"
    assert validate_report_schema(report) == []  # historical evidence-gated closure
    blocker["status"] = "resolved_in_contract_v7"
    assert validate_report_schema(report) == []  # v7 natural-boundary evidence is supported
    blocker["status"] = "resolved_in_contract_v8"
    assert validate_report_schema(report) == []  # v8 embedded wall-measure evidence is supported
    blocker["status"] = "resolved_in_contract_v9"
    assert validate_report_schema(report) == []  # v9 cut-cell transport evidence is supported
    blocker["status"] = "thermodynamic_equilibrium_validated_v9"
    assert validate_report_schema(report) == []  # the W-CONTACT-ANGLE closure wording
    blocker["status"] = "resolved_in_contract_v10"
    assert any("status is invalid" in error for error in validate_report_schema(report))


def test_ci_profile_report_records_contract_v9_lineage_and_stays_baseline_only(tmp_path, monkeypatch):
    pytest.importorskip("jax")
    import phasefield as pf
    from production.config import compute_file_sha256
    from production.run_validation import run_validation

    monkeypatch.setenv("JAX_PLATFORMS", "cpu")  # run_validation pins the CPU for the ci profile
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    config_path = HERE / "production" / "configs" / "ci.json"
    assert run_validation(str(config_path), str(tmp_path / "ci")) == 0
    report = json.loads((tmp_path / "ci" / "report.json").read_text(encoding="utf-8"))
    assert validate_report_schema(report) == []
    assert report["repository"]["solver_contract_version"] == pf.SOLVER_CONTRACT_VERSION == 10
    assert report["repository"]["wall_measure_method"] == pf.WALL_MEASURE_METHOD == "sdf_cutcell_v1"
    assert report["repository"]["wall_measure_contract_version"] == pf.WALL_MEASURE_CONTRACT_VERSION == 1
    assert report["repository"]["phasefield_sha256"] == compute_file_sha256(HERE / "phasefield.py")
    assert len(report["repository"]["validation_code_sha256"]) == 64
    assert len(report["config"]["sha256"]) == 64
    assert report["physics_status"] == "BASELINE_ONLY"  # never VALIDATED / PRODUCTION_READY here
    blockers = {item["id"]: item["status"] for item in report["known_solver_blockers"]}
    assert blockers["P-LAPLACE-SIGN"] == "measurement_required"  # one radius cannot establish 1/R scaling
    for name in ("P-VARDENS-PROJ", "P-CAP-RHO", "N-DT", "P-VARVISC", "BC-Y-PERIODIC"):
        assert blockers[name] == "open"
    static = report["benchmarks"]["static_droplet"]["cases"][0]
    assert static["delta_p"] > 0.0 and static["laplace_ratio"] > 0.0


def _angle_benchmarks(*, boundary="impermeable_flux", converged=True, theta_90=90.0, wall_measure=None):
    pytest.importorskip("jax")
    import phasefield as pf

    cases = []
    for target in (60.0, 90.0, 120.0, 150.0):
        angle = theta_90 if target == 90.0 else target
        cases.append(
            {
                "target_deg": target,
                "measured_deg": angle if converged else None,
                "absolute_error_deg": abs(angle - target) if converged else None,
                "finite": True,
                "converged": bool(converged),
                "mass_relative_drift": 5e-4,
                "total_mass_relative_drift": 5e-4,
                "max_solid_liquid_fraction": 0.0,
                "phase_boundary_model": boundary,
                "wall_measure_method": (pf.WALL_MEASURE_METHOD if wall_measure is None else wall_measure),
                "enforce_solid_phi": boundary == "projection_legacy",
                "samples": [{"max_speed": 1e-4}],
            }
        )
    return {
        "static_droplet": {"cases": [], "summary": None},
        "contact_angle": {"cases": cases, "summary": {}},
        "impact": {"cases": []},
        "convergence": {},
    }


def test_contact_angle_blockers_close_only_after_full_acceptance():
    """The label tracks the *live* contract, and never closes on partial or legacy evidence."""
    good = _blockers(_angle_benchmarks())
    assert good["W-CONTACT-ANGLE"]["status"] == f"resolved_in_contract_v{pf.SOLVER_CONTRACT_VERSION}"
    assert good["W-CONTACT-ANGLE"]["evidence"]["wall_measure_ok"] is True
    assert good["P-SOLID-PIN"]["status"] == f"resolved_in_contract_v{pf.SOLVER_CONTRACT_VERSION}"

    partial = _blockers(_angle_benchmarks(converged=False))
    assert partial["W-CONTACT-ANGLE"]["status"] == "measurement_required"
    assert partial["P-SOLID-PIN"]["status"] == "confirmed_problem"

    legacy = _blockers(_angle_benchmarks(boundary="projection_legacy"))
    assert legacy["W-CONTACT-ANGLE"]["status"] == "confirmed_problem"
    assert legacy["P-SOLID-PIN"]["status"] == "confirmed_problem"

    # Contract v8: a four-angle matrix reproduced with the pinned legacy wall measure is not
    # evidence about the production wetting model, so it can never close the blocker.
    legacy_measure = _blockers(_angle_benchmarks(wall_measure="diffuse_sdf_v7"))
    assert legacy_measure["W-CONTACT-ANGLE"]["status"] == "confirmed_problem"
    assert legacy_measure["W-CONTACT-ANGLE"]["evidence"]["wall_measure_ok"] is False
    assert legacy_measure["W-CONTACT-ANGLE"]["evidence"]["accepted"] is False

    bad_neutral = _blockers(_angle_benchmarks(theta_90=94.0))
    assert bad_neutral["W-CONTACT-ANGLE"]["status"] == "confirmed_problem"
    assert bad_neutral["P-SOLID-PIN"]["status"] == "confirmed_problem"


def _comparison_report(
    *,
    contract,
    ratios,
    slope,
    peak=2.0e-4,
    static_mass=1.0e-7,
    impact_mass=1.0e-3,
    impact_peak=2.3,
    r_squared=0.99964,
    config_sha="d" * 64,
    status="BASELINE_ONLY",
    laplace_status="confirmed_problem",
    cap_rho_status="open",
):
    report = _minimal_report()
    report["physics_status"] = status
    report["repository"]["solver_contract_version"] = contract
    report["config"]["sha256"] = config_sha
    radii = [0.6, 0.8, 1.0, 1.2]
    report["benchmarks"]["static_droplet"] = {
        "cases": [
            {
                "R": radius,
                "finite": True,
                "laplace_ratio": ratio,
                "delta_p": ratio / (radius * 100.0),
                "max_speed_peak": peak,
                "max_speed_final": peak,
                "kinetic_energy_final": 1.0e-8,
                "mass_relative_drift": static_mass,
            }
            for radius, ratio in zip(radii, ratios)
        ],
        "summary": {"slope_delta_p_vs_inv_R": slope, "r_squared": r_squared},
    }
    report["benchmarks"]["contact_angle"] = {
        "cases": [{"target_deg": 90.0, "measured_deg": 117.0, "signed_error_deg": 27.0}],
        "summary": {"mae_deg": 27.0, "max_absolute_error_deg": 27.0},
    }
    report["benchmarks"]["impact"] = {
        "cases": [
            {
                "case_name": "flat",
                "finite": True,
                "beta_max": 2.2,
                "final_y_cm": 0.9,
                "mass_drift": impact_mass,
                "first_contact_time": None,
                "time_series": {"g_0.5": [0.30, 0.1484], "g_0.1": [0.2, 0.0547], "max_speed": [impact_peak, 0.5]},
            }
        ]
    }
    statuses = {
        "P-VARDENS-PROJ": "open",
        "P-CAP-RHO": cap_rho_status,
        "N-DT": "open",
        "W-CONTACT-ANGLE": "confirmed_problem",
        "P-LAPLACE-SIGN": laplace_status,
        "P-VARVISC": "open",
        "BC-Y-PERIODIC": "open",
    }
    report["known_solver_blockers"] = [
        {"id": name, "severity": "high", "status": status_, "description": name} for name, status_ in statuses.items()
    ]
    return report


def _guard_failures(before, after):
    from production.compare_reports import compare_reports

    result = compare_reports(before, after)
    return {g["name"] for g in result["guards"] if not g["passed"] and g["kind"] == "guard"}, result


def _healthy_pair(**after_overrides):
    before = _comparison_report(contract=4, ratios=[-1.0459, -1.0193, -1.0083, -1.0031], slope=-0.0109)
    after_kwargs = dict(
        contract=5,
        ratios=[1.0459, 1.0194, 1.0083, 1.0032],
        slope=0.0109,
        peak=1.5e-4,
        laplace_status="resolved_in_contract_v5",
    )
    after_kwargs.update(after_overrides)
    return before, _comparison_report(**after_kwargs)


def test_compare_reports_healthy_l1a2a_pair_passes_every_guard():
    from production.compare_reports import format_markdown

    before, after = _healthy_pair()
    failures, result = _guard_failures(before, after)
    assert failures == set()
    assert result["guards_passed"] is True
    assert all(g["passed"] for g in result["guards"])  # P1 targets also met for this pair
    assert result["laplace"]["after_max_abs_error"] == pytest.approx(0.0459)
    markdown = format_markdown(result)
    assert "| 0.6 | -1.0459 | +1.0459 | 0.0459 |" in markdown
    assert "\\|After−1\\|" in markdown


@pytest.mark.parametrize(
    "override, guard",
    [
        (dict(ratios=[1.0459, 1.0194, -0.02, 1.0032]), "P0_all_laplace_ratios_positive"),
        (dict(r_squared=0.9), "P0_delta_p_vs_inv_R_scaling"),
        (dict(slope=-0.0109), "P0_delta_p_vs_inv_R_scaling"),
        (dict(peak=1.5e-3), "spurious_current_peak_guard"),  # > max(5 x 2e-4, 1e-3)
        (dict(static_mass=2.0e-4), "static_mass_drift_guard"),
        (dict(impact_mass=6.0e-3), "impact_mass_drift_guard"),
        (dict(impact_peak=9.0), "impact_no_explosive_acceleration_guard"),
        (dict(config_sha="e" * 64), "baseline_config_unchanged"),
        (dict(status="VALIDATED"), "physics_status_not_promoted"),
        (dict(status="PRODUCTION_READY"), "physics_status_not_promoted"),
        (dict(cap_rho_status="resolved_in_contract_v5"), "untouched_blockers_remain_open"),
        (dict(contract=4), "solver_contract_bumped"),
    ],
)
def test_compare_reports_guards_fail_closed(override, guard):
    before, after = _healthy_pair(**override)
    failures, result = _guard_failures(before, after)
    assert guard in failures
    assert result["guards_passed"] is False


def test_compare_reports_blocker_closed_without_evidence_is_flagged():
    before, after = _healthy_pair(ratios=[1.0459, 1.0194, -0.02, 1.0032])
    failures, _ = _guard_failures(before, after)
    assert "laplace_blocker_closed_only_with_evidence" in failures


def test_compare_reports_cli_exit_status(tmp_path):
    from production.compare_reports import main

    before, after = _healthy_pair()
    paths = {}
    for name, report in (("before", before), ("after", after)):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(report), encoding="utf-8")
    md = tmp_path / "compare.md"
    assert main(["--before", str(paths["before"]), "--after", str(paths["after"]), "--markdown", str(md)]) == 0
    assert "L1A-2a before / after" in md.read_text(encoding="utf-8")

    _, bad_after = _healthy_pair(ratios=[1.0459, 1.0194, -0.02, 1.0032])
    paths["bad"] = tmp_path / "bad.json"
    paths["bad"].write_text(json.dumps(bad_after), encoding="utf-8")
    assert main(["--before", str(paths["before"]), "--after", str(paths["bad"])]) == 1  # guard failed: STOP
    assert main(["--before", str(paths["before"]), "--after", str(tmp_path / "missing.json")]) == 2


def test_compare_reports_target_mirrors_provisional_targets():
    pytest.importorskip("jax")
    from production import compare_reports
    from production.validation import LAPLACE_SIGN_CLOSURE_CRITERIA, PROVISIONAL_READINESS_TARGETS

    assert compare_reports.LAPLACE_TARGET == PROVISIONAL_READINESS_TARGETS["laplace_relative_error"]
    assert compare_reports.LAPLACE_MIN_R_SQUARED == LAPLACE_SIGN_CLOSURE_CRITERIA["min_r_squared"]


# ---------------------------------------------------------------------------
#  L1A-2b: solid / gas-film ablation harness (diagnostic only)
# ---------------------------------------------------------------------------


def test_phase_boundary_audit_covers_v8_flux_and_thermodynamics():
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    from production.phase_boundary_audit import run_phase_boundary_audit

    previous_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        audit = run_phase_boundary_audit(
            N=32, sessile_steps=8, energy_steps=4, ab_steps=8, sample_every=4, dtype=jnp.float64
        )
    finally:
        jax.config.update("jax_enable_x64", previous_x64)
    failed = [check.name for check in audit.checks if not check.passed]
    assert failed == []
    assert audit.settings["solver_contract_version"] == 10
    assert audit.settings["wall_measure_method"] == "sdf_cutcell_v1"
    variational = audit.numbers["v8_variational_audit"]
    assert variational["relative_error"] <= 1e-6  # wall-measure gate: mu == d(F_bulk + F_wall^h)/dphi
    assert variational["worst_amplitude_relative_error"] <= 1e-4
    assert variational["separate_wetting_mu_max_abs"] == 0.0  # exactly one wall contribution
    assert variational["wall_measure_total_length"] == pytest.approx(6.0, rel=1e-9)
    ab = audit.numbers["neutral_90_projection_ab"]["impermeable_v8"]
    assert ab["max_solid_phase_fraction"] <= 1e-6


def test_solid_gas_film_audit_variates_one_factor_at_a_time():
    pytest.importorskip("jax")
    from production.solid_gas_film_audit import QUICK_FACTORS, run_solid_gas_film_audit

    result = run_solid_gas_film_audit(N=32, steps=30, save_every=10, factors=QUICK_FACTORS)
    assert result["settings"]["diagnostic_only"] is True
    assert result["settings"]["production_defaults_changed"] is False
    factors = {run["factor"] for run in result["runs"]}
    assert factors == {"wetting_model", "eta_pen_over_dt", "nu_g_over_nu_l", "phase_boundary_model"}
    for run in result["runs"]:
        assert run["finite"] is True
        metrics = run["metrics"]
        for key in ("minimum_gap_0.5", "minimum_gap_0.1", "beta_max", "final_y_cm", "peak_speed", "mass_drift"):
            assert metrics[key] is not None and np.isfinite(metrics[key])
        others = [k for k in result["settings"]["baseline_case"]]
        assert others  # the baseline case is recorded verbatim
    # every quick grid contains exactly the baseline default of its factor
    assert QUICK_FACTORS["eta_pen_over_dt"] == [2.0] and QUICK_FACTORS["nu_g_over_nu_l"] == [10.0]


# ---------------------------------------------------------------------------
#  L1A-2b: clean sessile seed, equilibrium gating and the A/B/C matrix
# ---------------------------------------------------------------------------


def test_contact_angle_case_uses_the_clean_seed_and_gates_on_convergence():
    from production import validation as V

    pytest.importorskip("jax")
    case = V.run_contact_angle_case(90.0, N=48, max_steps=20, dt=4e-3, eps_factor=2.0, R=0.8, dtype="float32")
    assert case.converged is False  # 20 steps cannot satisfy 3 windows of 5e-4 speed
    assert case.measured_deg is None and case.absolute_error_deg is None
    assert case.signed_error_deg is None
    assert isinstance(case.final_sampled_deg, float) and 0.0 <= case.final_sampled_deg <= 180.0
    assert case.initial_solid_liquid_fraction == 0.0
    assert case.wetting_model == "surface_energy"
    assert case.phase_boundary_model == "impermeable_flux" and case.enforce_solid_phi is False
    assert case.final_solid_liquid_fraction <= 1e-6
    assert case.implicit_relative_residual_max <= 1e-6
    assert case.relaxation_steps == 20
    sample_fields = {"step", "time", "measured_angle_deg", "max_speed", "total_mass", "fluid_mass"}
    assert case.samples and all(sample_fields <= set(row) for row in case.samples)
    assert case.total_mass_relative_drift <= 1e-3


def test_contact_angle_case_reports_the_angle_once_the_windows_converge():
    from production import validation as V

    pytest.importorskip("jax")
    case = V.run_contact_angle_case(
        90.0,
        N=64,
        max_steps=5,
        dt=4e-3,
        eps_factor=2.0,
        R=0.6,
        dtype="float32",
        sample_every=5,
        angle_tol_deg=180.0,
        speed_tol=1e9,
        windows=1,
    )
    assert case.converged is True
    assert case.converged_step == 5
    assert case.converged_time == pytest.approx(case.samples[-1]["time"])
    assert case.measured_deg is not None and case.absolute_error_deg is not None
    summary = V.summarize_contact_angles([case])
    assert summary["converged_case_count"] == 1
    assert summary["mae_deg"] == pytest.approx(case.absolute_error_deg)


def test_contact_angle_summary_does_not_call_a_partial_subset_matrix_mae():
    from types import SimpleNamespace

    from production.validation import summarize_contact_angles

    cases = [
        SimpleNamespace(
            target_deg=float(target),
            finite=True,
            converged=(target == 90),
            measured_deg=float(target) + 0.5 if target == 90 else None,
            absolute_error_deg=0.5 if target == 90 else None,
            final_sampled_deg=float(target) + 0.5,
            mass_relative_drift=1e-4,
            total_mass_relative_drift=1e-4,
            max_solid_liquid_fraction=0.0,
        )
        for target in (60, 90, 120, 150)
    ]
    summary = summarize_contact_angles(cases)
    assert summary["converged_case_count"] == 1
    assert summary["mae_deg"] is None
    assert summary["converged_subset_mae_deg"] == pytest.approx(0.5)
    assert summary["all_targets_converged"] is False


def test_contact_angle_summary_excludes_drifting_runs_from_the_error_statistics():
    from production import validation as V

    pytest.importorskip("jax")
    cases = [
        V.run_contact_angle_case(target, N=48, max_steps=10, dt=4e-3, eps_factor=2.0, R=0.8, dtype="float32")
        for target in (60.0, 120.0)
    ]
    summary = V.summarize_contact_angles(cases)
    assert summary["converged_case_count"] == 0
    assert summary["mae_deg"] is None and summary["rmse_deg"] is None and summary["max_absolute_error_deg"] is None
    assert summary["monotonic_target_to_measured"] is False
    assert summary["non_converged_targets"] == [60.0, 120.0]
    assert len(summary["final_angles_deg"]) == 2


def test_contact_angle_matrix_reports_history_and_convergence_accounting():
    from production import contact_angle_matrix as M

    pytest.importorskip("jax")
    matrix = M.build_matrix(
        [60.0, 120.0],
        ["surface_energy"],
        N=32,
        max_steps=20,
        dt=4e-3,
        R=0.6,
        sample_every=20,
        angle_tol_deg=0.25,
        speed_tol=5e-4,
        windows=3,
    )
    assert "history_recorded" in matrix["profiles"]
    history = matrix["profiles"]["history_recorded"]["summary"]
    assert history["mae_deg"] == pytest.approx(30.635, abs=1e-3)
    assert history["converged_subset_mae_deg"] is None
    assert history["monotonic_target_to_measured"] is False
    summary = matrix["profiles"]["surface_energy"]["summary"]
    assert summary["converged_case_count"] == 0 and summary["mae_deg"] is None
    assert summary["converged_subset_mae_deg"] is None
    table = M.format_matrix(matrix)
    assert "| surface_energy | 60 |" in table and "| history_recorded | 120 |" in table
    assert "180" not in table.splitlines()[0]  # header only

    partial_rows = [
        {"target_deg": 60.0, "measured_deg": None, "absolute_error_deg": None, "converged": False},
        {"target_deg": 90.0, "measured_deg": 91.0, "absolute_error_deg": 1.0, "converged": True},
        {"target_deg": 120.0, "measured_deg": None, "absolute_error_deg": None, "converged": False},
    ]
    partial = M.summarize_rows(partial_rows)
    assert partial["mae_deg"] is None
    assert partial["rmse_deg"] is None and partial["max_absolute_error_deg"] is None
    assert partial["converged_subset_mae_deg"] == pytest.approx(1.0)
