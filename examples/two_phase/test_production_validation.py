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


HERE = Path(__file__).resolve().parent


def _minimal_report() -> dict:
    return {
        "validation_report_schema_version": REPORT_SCHEMA_VERSION,
        "physics_status": "BASELINE_ONLY",
        "contract_status": "PASS",
        "repository": {
            "git_sha": "a" * 40,
            "solver_contract_version": 4,
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
