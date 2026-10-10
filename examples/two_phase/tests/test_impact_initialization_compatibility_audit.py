"""L1A-2s invariant, initialization, causality, and fail-closed protocol tests."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import phasefield as pf  # noqa: E402
from production import impact_impulse_projection_audit as impulse_audit  # noqa: E402
from production import impact_initialization_compatibility_audit as audit  # noqa: E402
from production import timestep_policy  # noqa: E402
from production import validation  # noqa: E402


@pytest.fixture(scope="module")
def primary_bundle():
    case = audit._canary_cases()["flat_we100_ct050"]
    return audit._derive_bundle("flat_we100_ct050", case, 48, 0.004)


@pytest.fixture(scope="module")
def pillar_bundle():
    case = audit._canary_cases()["pillar_training"]
    return audit._derive_bundle("pillar_training", case, 48, 0.004)


@pytest.fixture(scope="module")
def pillar_bundle_192():
    case = audit._canary_cases()["pillar_training"]
    return audit._derive_bundle("pillar_training", case, 192, 0.004)


def _synthetic_event_rows(bundle, *, approach=0.2, post=True):
    rows = []
    gaps = (0.52, 0.41, 0.30, 0.22, 0.08)
    times = (0.0, 0.002, 0.004, 0.006, 0.008)
    for index, (t, gap) in enumerate(zip(times, gaps)):
        is_contact = index == len(times) - 1
        rows.append(
            {
                "t": t,
                "contact_phi05": is_contact,
                "gap_phi05": gap,
                "gap_phi01": gap + 0.1,
                "gap_phi05_over_dx": gap / bundle.p.dx,
                "gap_phi01_over_dx": (gap + 0.1) / bundle.p.dx,
                "gap_phi05_over_eps": gap / bundle.p.eps,
                "gap_phi01_over_eps": (gap + 0.1) / bundle.p.eps,
                "local_approach_speed_into_solid": approach,
                "local_support_phi05": {
                    "gap": gap,
                    "support_xy": [3.0, 0.5],
                    "wall_xy": [3.0, 0.25],
                    "local_wall_normal_into_solid": [0.0, -1.0],
                },
                "local_support_phi01": {"gap": gap + 0.1},
                "v_liquid": -0.2,
                "v_core": -0.2,
                "momentum": {"liquid_momentum_y": -0.1, "liquid_physical_kinetic_energy": 0.02},
                "beta": 1.0 + (0.4 if is_contact else 0.0),
                "drop_height": 1.0,
                "contact_line_x_center": 3.0,
            }
        )
    if post:
        rows.extend(
            [
                {**rows[-1], "t": 0.02, "contact_phi05": True, "beta": 1.5, "v_liquid": -0.1, "gap_phi05": 0.05},
                {**rows[-1], "t": 0.04, "contact_phi05": True, "beta": 1.6, "v_liquid": -0.05, "gap_phi05": 0.04},
            ]
        )
    return {"rows": rows}


def test_l1a2r_merged_base_verified_or_fail_closed():
    result = audit._frozen_source_gate()
    assert all(result["checks"].values())
    assert audit.PR21["state"] == "MERGED"
    assert audit.PR21["merge_sha"] == audit.BASE_MAIN_SHA
    assert result["latest_main_sha"] in audit.VERIFIED_MAIN_SHAS
    assert result["integration_lineage"]["original_base_main_sha"] == audit.BASE_MAIN_SHA
    assert result["integration_lineage"]["main_lineage_verified"] is True


def test_contract12_and_policy_pinned():
    assert pf.SOLVER_CONTRACT_VERSION == 12
    assert pf.SOLVER_CONTRACT_12_TRAJECTORY_POLICY == "impact_phase_cap_dx2_v1"
    assert timestep_policy.DEFAULT_POLICY_NAME == "impact_phase_cap_dx2_v1"
    schedule = timestep_policy.effective_dt_for_case(audit._canary_cases()["flat_we100_ct050"], 192, 0.004)
    assert schedule["effective_dt"] == pytest.approx(0.002)


def test_uniform_build_case_bitwise_unchanged():
    case = audit._canary_cases()["flat_we100_ct050"]
    a = pf.build_case(case, N=48, dt=0.004)
    b = pf.build_case(case, N=48, dt=0.004)
    assert [audit._array_sha(np.asarray(x)) for x in (a[2].phi, a[2].u, a[2].v)] == [
        audit._array_sha(np.asarray(x)) for x in (b[2].phi, b[2].u, b[2].v)
    ]
    assert audit._geometry_hash(a[1]) == audit._geometry_hash(b[1])
    assert np.array_equal(np.asarray(a[2].v), -0.5 * np.ones((48, 48), dtype=np.float32))


def test_phasefield_and_generator_sources_unchanged():
    hashes = audit._source_hashes()
    baseline = json.loads((audit.TWO_PHASE / "evidence/l1a2r/manifest.json").read_text())["binding"]["source_hashes"]
    assert hashes["phasefield"] == baseline["phasefield"]
    assert hashes["generate_dataset"] == baseline["generate_dataset"]
    assert hashes["cases"] == baseline["cases"]
    assert hashes["timestep_policy"] == baseline["timestep_policy"]


def test_all_old_thresholds_unchanged():
    assert audit._generator_thresholds_unchanged() == {
        "max_phi_overshoot": 0.02,
        "max_solid_leak": 5e-4,
        "min_total_mass_ratio": 0.995,
        "max_total_mass_ratio": 1.005,
        "max_speed": 5.0,
        "key_observable_refinement_change": 0.03,
    }
    assert validation.PROVISIONAL_READINESS_TARGETS["key_observable_refinement_change"] == 0.03


def test_no_new_default_velocity_mode():
    assert inspect.signature(pf.droplet_initial_state).parameters["velocity_mode"].default == "uniform"
    assert inspect.signature(pf.build_case).parameters["dt"].default == 4e-3
    assert "velocity_mode" not in audit._canary_cases()["flat_we100_ct050"]


def test_diagnostic_cases_have_same_phi_and_geometry(primary_bundle):
    states = [audit.build_candidate(primary_bundle, name)[0] for name in audit.CANDIDATES]
    phi_hashes = [audit._array_sha(np.asarray(state.phi)) for state in states]
    assert len(set(phi_hashes)) == 1
    assert len({audit._geometry_hash(primary_bundle.solid)}) == 1
    assert all(np.array_equal(np.asarray(state.phi), np.asarray(primary_bundle.base.phi)) for state in states)
    assert (primary_bundle.p.M, primary_bundle.p.eps, primary_bundle.p.We, primary_bundle.p.Re) == pytest.approx(
        (0.002, 1.5 * primary_bundle.p.dx, 100.0, 200.0)
    )


def test_streamfunction_discrete_D_divergence_is_measured(primary_bundle):
    state, _ = audit.build_candidate(primary_bundle, "STREAMFUNCTION_LOCALIZED_V0")
    metrics = audit._divergence_metrics(state.u, state.v, primary_bundle.solid, primary_bundle.p)
    assert metrics["initial_D_div_Linf"] >= 0.0
    assert metrics["initial_D_div_L2_grid_rms"] >= 0.0
    assert metrics["cutcell_flux_divergence_status"] == "MEASURED_OPEN_FACE_RECONSTRUCTION"


def test_streamfunction_v0_is_not_misclassified_as_solid_compatible(primary_bundle):
    state, details = audit.build_candidate(primary_bundle, "STREAMFUNCTION_LOCALIZED_V0")
    metrics = audit.initial_metrics(primary_bundle, "STREAMFUNCTION_LOCALIZED_V0", state, details)
    assert metrics["initial_constraints_pass"] is False
    assert details["large_solid_velocity_is_not_relabelled_compatible"] is True
    assert any(
        not metrics["acceptance_checks"][name]
        for name in (
            "open_face_FV_divergence",
            "deep_solid_velocity",
            "nearwall_chi_velocity",
            "embedded_wall_normal_velocity",
        )
    )


def test_masked_uniform_large_divergence_is_not_misclassified(primary_bundle):
    mask = jnp.asarray(np.asarray(primary_bundle.base.phi) >= 0.5, dtype=primary_bundle.p.dtype)
    u = jnp.zeros_like(mask)
    v = -primary_bundle.u_impact * mask
    metrics = audit._divergence_metrics(u, v, primary_bundle.solid, primary_bundle.p)
    assert metrics["initial_D_div_Linf"] * primary_bundle.p.dx / primary_bundle.u_impact > 0.1


def test_candidate_C2_uses_case_specific_sdf_and_y0(primary_bundle, pillar_bundle):
    flat = audit.build_candidate(primary_bundle, "SDF_TAPERED_STREAMFUNCTION_V1")[1]
    pillar = audit.build_candidate(pillar_bundle, "SDF_TAPERED_STREAMFUNCTION_V1")[1]
    assert flat["geometry_sdf_hash"] == audit._array_sha(np.asarray(primary_bundle.solid.sdf))
    assert pillar["geometry_sdf_hash"] == audit._array_sha(np.asarray(pillar_bundle.solid.sdf))
    assert flat["y0"] == primary_bundle.y0
    assert pillar["y0"] == pillar_bundle.y0
    assert flat["y0"] != pillar["y0"]


def test_candidate_C2_periodic_x_seam_continuity():
    base = audit._canary_cases()["flat_we100_ct050"]
    case = {**base, "x0": 0.05}
    bundle = audit._derive_bundle("flat_we100_ct050_seam", case, 48, 0.004)
    _u, _v, psi, _details = audit.build_c2_velocity(bundle)
    psi = np.asarray(psi, dtype=np.float64)
    periodic_diffs = np.abs(np.roll(psi, -1, axis=0) - psi)
    seam = periodic_diffs[-1]
    adjacent = np.maximum(periodic_diffs[0], periodic_diffs[-2])
    assert np.isfinite(psi).all()
    assert float(np.max(seam)) <= 2.0 * float(np.max(adjacent)) + 1e-10


def test_candidate_C2_records_deep_solid_and_cutcell_wall_metrics(pillar_bundle_192):
    state, details = audit.build_candidate(pillar_bundle_192, "SDF_TAPERED_STREAMFUNCTION_V1")
    metrics = audit.initial_metrics(pillar_bundle_192, "SDF_TAPERED_STREAMFUNCTION_V1", state, details)
    assert metrics["deep_solid_cell_count"] > 0
    assert metrics["partial_cutcell_count"] > 0
    assert metrics["embedded_wall_flux_status"].startswith("MEASURED_")
    assert metrics["cutcell_flux_divergence_status"] == "MEASURED_OPEN_FACE_RECONSTRUCTION"


def test_velocity_rescaling_preserves_solenoidality(primary_bundle):
    state, details = audit.build_candidate(primary_bundle, "SDF_TAPERED_STREAMFUNCTION_V1")
    _u, _v, psi, _details = audit.build_c2_velocity(primary_bundle)
    unit_u = pf._ddy(psi, primary_bundle.p.dy)
    unit_v = -pf._ddx(psi, primary_bundle.p.dx)
    div = pf._ddx(state.u, primary_bundle.p.dx) + pf._ddy(state.v, primary_bundle.p.dy)
    scaled_div = np.max(np.abs(np.asarray(div))) * primary_bundle.p.dx / primary_bundle.u_impact
    assert np.isfinite(details["amplitude_scale"])
    assert scaled_div <= audit.INITIAL_ACCEPTANCE["central_D_dimensionless_Linf_max"]
    assert np.max(np.abs(np.asarray(state.u) - details["amplitude_scale"] * np.asarray(unit_u))) < 2e-6
    assert np.max(np.abs(np.asarray(state.v) - details["amplitude_scale"] * np.asarray(unit_v))) < 2e-6


def test_post_derivative_masking_is_not_silently_applied(primary_bundle):
    state, details = audit.build_candidate(primary_bundle, "SDF_TAPERED_STREAMFUNCTION_V1")
    u, v, psi, _ = audit.build_c2_velocity(primary_bundle)
    assert details["post_derivative_masking"] is False
    assert np.allclose(np.asarray(state.u), np.asarray(u), rtol=0.0, atol=0.0)
    assert np.allclose(np.asarray(state.v), np.asarray(v), rtol=0.0, atol=0.0)
    assert np.array_equal(
        np.asarray(u), np.asarray(details["amplitude_scale"] * pf._ddy(psi, primary_bundle.p.dy), dtype=np.float32)
    )


def test_candidate_C2_never_relabels_We_or_Re(primary_bundle):
    before = (primary_bundle.p.We, primary_bundle.p.Re, primary_bundle.p.M, primary_bundle.p.eps)
    state, details = audit.build_candidate(primary_bundle, "SDF_TAPERED_STREAMFUNCTION_V1")
    after = (primary_bundle.p.We, primary_bundle.p.Re, primary_bundle.p.M, primary_bundle.p.eps)
    assert before == after
    assert details["candidate"] == "SDF_TAPERED_STREAMFUNCTION_V1"
    assert "contact" not in details
    assert np.array_equal(np.asarray(state.phi), np.asarray(primary_bundle.base.phi))


def test_no_success_if_initial_constraints_are_unmeasured():
    row = {"measurements_complete": False, "initial_constraints_pass": True}
    matrix = {case: {"SDF_TAPERED_STREAMFUNCTION_V1": row} for case in audit.CASE_NAMES}
    assert audit._initialization_verdict(matrix) == "INCONCLUSIVE"


def test_jit_ledger_matches_pf_step_within_dtype_tolerance():
    case = audit._canary_cases()["flat_we100_ct050"]
    bundle = audit._derive_bundle("flat_we100_ct050", case, 24, 0.004)
    # The N=24 diffuse drop has no phi>=0.9 core, so validate the unchanged production seed.
    result = impulse_audit.validate_ledger(bundle.base, bundle.solid, bundle.p)
    assert all(result["within_dtype_tolerance_public_step"].values())
    assert all(result["fast_kernel_within_dtype_tolerance"].values())
    assert result["rhs_recomposition_bitwise"] == [True, True, True]


def test_projection_only_uniform_is_noop(primary_bundle):
    u = jnp.zeros((primary_bundle.N, primary_bundle.N), dtype=primary_bundle.p.dtype)
    v = -0.5 * jnp.ones_like(u)
    div = pf._ddx(u, primary_bundle.p.dx) + pf._ddy(v, primary_bundle.p.dy)
    pr = pf.poisson_solve(div / primary_bundle.p.dt, primary_bundle.p.m2_proj)
    projected_u = u - primary_bundle.p.dt * pf._ddx(pr, primary_bundle.p.dx)
    projected_v = v - primary_bundle.p.dt * pf._ddy(pr, primary_bundle.p.dy)
    assert np.array_equal(np.asarray(projected_u), np.asarray(u))
    assert np.array_equal(np.asarray(projected_v), np.asarray(v))


def test_periodic_projection_plain_grid_mean_correction_is_zero(primary_bundle):
    rng = np.random.default_rng(17)
    pressure = jnp.asarray(rng.normal(size=(primary_bundle.N, primary_bundle.N)), dtype=primary_bundle.p.dtype)
    gx = pf._ddx(pressure, primary_bundle.p.dx)
    gy = pf._ddy(pressure, primary_bundle.p.dy)
    assert float(np.mean(np.asarray(gx))) == pytest.approx(0.0, abs=2e-7)
    assert float(np.mean(np.asarray(gy))) == pytest.approx(0.0, abs=2e-7)


def test_brinkman_mean_change_not_confused_with_projection(primary_bundle):
    _new, ledger = impulse_audit.substep_ledger(
        primary_bundle.base,
        primary_bundle.solid,
        primary_bundle.p,
        float(primary_bundle.p.dt) / 3.0,
        capture_fields=True,
    )
    means = ledger["full_grid_mean_v"]
    brinkman_mean = means["after_brinkman"] - means["after_explicit"]
    projection_mean = means["after_projection"] - means["after_brinkman"]
    assert abs(brinkman_mean) > 1e-5
    assert abs(projection_mean) < 1e-7
    assert abs(means["mean_of_proj_correction"]) < 1e-7


def test_no_solid_uniform_retains_impulse():
    case = audit._canary_cases()["flat_we100_ct050"]
    bundle = audit._derive_bundle("flat_we100_ct050", case, 24, 0.002)
    # A very distant SDF gives an exactly empty chi/embedded-wall field; SDF=1
    # still has a small smooth-indicator tail that legitimately Brinkman-damps.
    sdf_empty = 100.0 * jnp.ones((24, 24), dtype=bundle.p.dtype)
    solid_empty = pf.make_solid(sdf_empty, bundle.p, cos_theta=0.5)
    assert float(np.max(np.asarray(solid_empty.chi))) == 0.0
    state = pf.State(
        phi=jnp.zeros((24, 24), dtype=pf.phase_state_dtype(bundle.p)),
        u=jnp.zeros((24, 24), dtype=bundle.p.dtype),
        v=-0.5 * jnp.ones((24, 24), dtype=bundle.p.dtype),
        t=0.0,
    )
    kernel = impulse_audit.make_fast_substep(solid_empty, bundle.p, float(bundle.p.dt) / 3.0)
    for _ in range(120 * 3):
        state, _ = kernel(state)
    assert float(state.t) == pytest.approx(0.24, abs=1e-12)
    assert np.max(np.abs(np.asarray(state.v) + 0.5)) < 1e-5


def test_approach_sign_is_local_wall_normal_aware():
    support = {"gap_cell": [0, 0], "local_wall_normal_into_solid": [0.0, -1.0]}
    u = np.zeros((2, 2))
    down = -0.4 * np.ones((2, 2))
    up = 0.4 * np.ones((2, 2))

    class Grid:
        Nx = Ny = 2
        dx = dy = 1.0

    assert audit._approach_speed(u, down, support, Grid()) == pytest.approx(0.4)
    assert audit._approach_speed(u, up, support, Grid()) == pytest.approx(-0.4)


def test_local_gap_moves_subcell_with_reconstructed_phase_contour(primary_bundle):
    p = primary_bundle.p
    x = (np.arange(p.Nx, dtype=np.float64) + 0.5) * float(p.dx)
    y = (np.arange(p.Ny, dtype=np.float64) + 0.5) * float(p.dy)
    X, Y = np.meshgrid(x, y, indexing="ij")
    sx = (X - primary_bundle.x0 + 0.5 * p.Lx) % p.Lx - 0.5 * p.Lx

    def circle(center_y):
        sy = (Y - center_y + 0.5 * p.Ly) % p.Ly - 0.5 * p.Ly
        radius = np.hypot(sx, sy)
        return 0.5 * (1.0 - np.tanh((radius - primary_bundle.R) / (np.sqrt(2.0) * p.eps)))

    # A synthetic contour translation only tests subcell localization; no canary input,
    # geometry, or trajectory is moved or used as audit evidence.
    upper = audit._local_gap(circle(primary_bundle.y0), 0.5, primary_bundle.solid, p)
    lower = audit._local_gap(circle(primary_bundle.y0 - 0.04), 0.5, primary_bundle.solid, p)
    assert upper["gap_method"] == "BILINEAR_SDF_ON_LINEAR_PERIODIC_PHI_THRESHOLD_CONTOUR"
    assert lower["gap"] < upper["gap"]
    assert upper["gap"] - lower["gap"] > 0.02
    assert upper["threshold_contour_crossing_count"] > 0
    assert np.isclose(np.linalg.norm(upper["local_wall_normal_into_solid"]), 1.0)


def test_approach_speed_bilinearly_samples_reconstructed_contour_point():
    class Grid:
        Nx = Ny = 2
        dx = dy = 1.0
        Lx = Ly = 2.0

    u = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    v = np.zeros((2, 2))
    support = {
        "gap_cell": [0, 0],
        "support_xy": [1.0, 0.5],
        "local_wall_normal_into_solid": [1.0, 0.0],
    }
    assert audit._approach_speed(u, v, support, Grid()) == pytest.approx(0.5)


def test_phi_wall_contact_without_approach_is_not_authenticated(primary_bundle):
    result = audit.authenticate_impact(
        _synthetic_event_rows(primary_bundle, approach=0.0, post=True), primary_bundle, observed_horizon=0.48
    )
    assert result["verdict"] == "CONTACT_WITHOUT_MEANINGFUL_IMPACT"
    assert result["meaningful_local_normal_approach"] is False


def test_post_contact_dynamics_required(primary_bundle):
    result = audit.authenticate_impact(
        _synthetic_event_rows(primary_bundle, approach=0.2, post=False), primary_bundle, observed_horizon=0.48
    )
    assert result["verdict"] == "CONTACT_WITHOUT_MEANINGFUL_IMPACT"
    assert result["meaningful_local_normal_approach"] is True
    assert result["postcontact_dynamics_verified"] is False


def test_no_contact_0p48_is_not_called_no_contact_T8(primary_bundle):
    rows = {"rows": [{"t": t, "contact_phi05": False, "gap_phi05": 1.0} for t in (0.0, 0.24, 0.48)]}
    result = audit.authenticate_impact(rows, primary_bundle, observed_horizon=0.48)
    assert result["verdict"] == "NO_CONTACT_WITHIN_OBSERVED_WINDOW"


def test_full_horizon_no_contact_uses_correct_status(primary_bundle):
    rows = {"rows": [{"t": t, "contact_phi05": False, "gap_phi05": 1.0} for t in (0.0, 4.0, 8.0)]}
    result = audit.authenticate_impact(rows, primary_bundle, observed_horizon=8.0)
    assert result["verdict"] == "NO_CONTACT_OR_HOVER_AT_DATASET_HORIZON"


def test_retention_temporal_gate_is_not_u_contact_over_u_impact_gate():
    fine, coarse = 0.31, 0.309
    temporal = audit._relative_difference(fine, coarse, scale=0.2)
    assert temporal["relative"] < 0.03
    assert "contact_speed_matches_u_impact" not in audit.INITIAL_ACCEPTANCE
    assert audit.APPROACH_FLOOR_FRACTION == 0.2


def test_equal_physical_time_dt_comparisons(primary_bundle):
    rows = [{"t": 0.0}, {"t": 0.08}, {"t": 0.16}, {"t": 0.24}]
    assert audit._match_row({"rows": rows}, 0.16) == {"t": 0.16}
    assert audit._match_row({"rows": rows}, 0.1600001) is None
    for dt in audit.DT_LEVELS:
        assert round(audit.FRAME_CADENCE / dt) * dt == pytest.approx(audit.FRAME_CADENCE)


def test_frame_cadence_and_T8_are_kept():
    assert audit.FRAME_CADENCE == pytest.approx(0.08)
    assert audit.DATASET_HORIZON == pytest.approx(8.0)
    assert int(audit.DATASET_HORIZON / audit.FRAME_CADENCE) == 100


def test_saved_frame_schedule_uses_integer_steps_without_interpolation():
    actual = np.asarray([index * 0.002 * 40 for index in range(101)])
    result = audit._validate_frame_times(actual, t_end=8.0, cadence=0.08)
    assert result["status"] == "MEASURED"
    assert result["actual_frame_count"] == 101
    assert result["interpolation_used"] is False
    with pytest.raises(audit.AuditValidationError):
        audit._validate_frame_times(actual[:-1], t_end=8.0, cadence=0.08)


def test_spatial_refinement_reports_eps_and_eps_over_dx():
    p144 = pf.PhaseFieldParams(Nx=144, Ny=144, Lx=6.0, Ly=6.0)
    p192 = pf.PhaseFieldParams(Nx=192, Ny=192, Lx=6.0, Ly=6.0)
    assert p144.eps == pytest.approx(0.0625)
    assert p192.eps == pytest.approx(0.046875)
    assert p144.eps / p144.dx == pytest.approx(1.5)
    assert p192.eps / p192.dx == pytest.approx(1.5)


def test_common_grid_field_comparison_is_explicit():
    field = np.ones((12, 12), dtype=np.float32)
    mapped = audit._map_common_grid(field, (18, 18))
    assert mapped.shape == (18, 18)
    assert np.allclose(mapped, 1.0)


def test_near_zero_velocity_normalization_is_fail_closed():
    assert audit._relative_difference(0.0, 0.0, scale=0.0)["status"] == "UNMEASURED"
    field = audit._field_error(np.zeros((4, 4)), np.zeros((4, 4)), physical_scale=0.0)
    assert field["status"] == "UNMEASURED"
    assert field["relative_L2_with_physical_floor"] is None
    assert field["near_zero_fail_closed"] is True


def test_3pct_refinement_gate_is_unchanged():
    assert validation.PROVISIONAL_READINESS_TARGETS["key_observable_refinement_change"] == 0.03
    assert audit._generator_thresholds_unchanged()["key_observable_refinement_change"] == 0.03


def test_sample_phi_u_v_dtypes_unchanged():
    source = (audit.TWO_PHASE / "generate_dataset.py").read_text()
    assert "phi_d = _downsample_history(phi, ds).astype(np.float32)" in source
    assert "u_d = _downsample_history(u, ds).astype(np.float16)" in source
    assert "v_d = _downsample_history(v, ds).astype(np.float16)" in source


def test_diagnostic_samples_cannot_pass_as_production_lineage(tmp_path: Path):
    path = tmp_path / "diagnostic.npz"
    metadata = {"diagnostic_only": True, "production_lineage_eligible": False, "solver_contract_version": 12}
    np.savez_compressed(
        path,
        dataset_schema_version=np.array(3, dtype=np.int32),
        dataset_fingerprint=np.array("diagnostic-fingerprint"),
        case=np.array(json.dumps(metadata)),
        phi=np.zeros((1, 4, 4), dtype=np.float32),
    )
    assert audit.generator._saved_case_is_current(path, "diagnostic-fingerprint") is False


def test_no_contract13_promotion_in_2s():
    assert pf.SOLVER_CONTRACT_VERSION == 12
    assert audit.STAGE_VERSION == "l1a2s_v1"
    assert audit.CANDIDATES == ("UNIFORM_ALL_DOMAIN", "STREAMFUNCTION_LOCALIZED_V0", "SDF_TAPERED_STREAMFUNCTION_V1")


def test_negative_or_missing_case_stages_cannot_assemble_PASS():
    result = audit._assemble_stage_gate({"initial": {"status": "MEASURED", "pass": True}}, ("initial", "temporal"))
    assert result["status"] == "UNMEASURED"
    assert result["missing_or_unmeasured_stages"] == ["temporal"]


def test_missing_cutcell_measurement_cannot_pass_initializer(primary_bundle):
    row = audit.initial_metrics(
        primary_bundle,
        "SDF_TAPERED_STREAMFUNCTION_V1",
        audit.build_candidate(primary_bundle, "SDF_TAPERED_STREAMFUNCTION_V1")[0],
        {},
    )
    row["measurements_complete"] = False
    matrix = {case: {"SDF_TAPERED_STREAMFUNCTION_V1": row} for case in audit.CASE_NAMES}
    assert audit._initialization_verdict(matrix) == "INCONCLUSIVE"


def test_c2_distinct_surfaces_have_case_specific_sdf_hashes(primary_bundle, pillar_bundle):
    _, flat_details = audit.build_candidate(primary_bundle, "SDF_TAPERED_STREAMFUNCTION_V1")
    _, pillar_details = audit.build_candidate(pillar_bundle, "SDF_TAPERED_STREAMFUNCTION_V1")
    assert flat_details["geometry_sdf_hash"] != pillar_details["geometry_sdf_hash"]
    assert flat_details["x0"] == 3.0 and pillar_details["x0"] == 3.0
    assert flat_details["y0"] != pillar_details["y0"]


def test_c2_taper_is_zero_on_wall_stencils(primary_bundle):
    state, details = audit.build_candidate(primary_bundle, "SDF_TAPERED_STREAMFUNCTION_V1")
    sdf = np.asarray(primary_bundle.solid.sdf)
    speed = np.hypot(np.asarray(state.u), np.asarray(state.v))
    assert details["post_derivative_masking"] is False
    assert np.max(speed[np.abs(sdf) <= 3.0 * primary_bundle.p.dx]) == 0.0


def test_c2_cutcell_flux_reconstruction_uses_open_apertures(primary_bundle):
    state, _ = audit.build_candidate(primary_bundle, "SDF_TAPERED_STREAMFUNCTION_V1")
    metrics = audit._divergence_metrics(state.u, state.v, primary_bundle.solid, primary_bundle.p)
    assert "A_f" in metrics["cutcell_flux_reconstruction"]
    assert "V_i" in metrics["cutcell_flux_reconstruction"]
    assert metrics["cutcell_flux_divergence_status"] != "UNMEASURED"


def test_case_specific_y0_is_not_shared_mutable_state():
    cases = audit._canary_cases()
    bundles = [audit._derive_bundle(name, cases[name], 48, 0.004) for name in audit.CASE_NAMES]
    assert [bundle.case_name for bundle in bundles] == list(audit.CASE_NAMES)
    assert len({bundle.y0 for bundle in bundles}) > 1


def test_contract_and_source_anchors_are_frozen():
    mapped = audit._source_operator_map()
    assert "actual contract-12 periodic central operators" in mapped["central_difference_D"]
    assert "EmbeddedFluidGeometry" in mapped["phasefield_geometry_authority"]
    assert mapped["embedded_wall_flux_status"] == "MEASURED_WITH_DECLARED_BILINEAR_RECONSTRUCTION"


def test_event_authentication_keeps_phi01_and_phi05_gaps_separate(primary_bundle):
    result = audit.authenticate_impact(
        _synthetic_event_rows(primary_bundle, approach=0.2, post=True), primary_bundle, observed_horizon=0.48
    )
    assert "contact_gap_phi05_over_dx" in result
    assert "contact_gap_phi01_over_dx" in result
    assert result["contact_gap_criterion"].startswith("local phi=0.5")
    assert result["verdict"] == "IMPACT_AUTHENTICATED"
    assert result["meaningful_local_normal_approach"] is True
    assert result["postcontact_dynamics_verified"] is True


def test_uniform_and_streamfunction_controls_remain_negative_controls(primary_bundle):
    c0, _ = audit.build_candidate(primary_bundle, "UNIFORM_ALL_DOMAIN")
    c1, _ = audit.build_candidate(primary_bundle, "STREAMFUNCTION_LOCALIZED_V0")
    c2, _ = audit.build_candidate(primary_bundle, "SDF_TAPERED_STREAMFUNCTION_V1")
    m0 = audit.initial_metrics(primary_bundle, "UNIFORM_ALL_DOMAIN", c0, {})
    m1 = audit.initial_metrics(primary_bundle, "STREAMFUNCTION_LOCALIZED_V0", c1, {})
    m2 = audit.initial_metrics(primary_bundle, "SDF_TAPERED_STREAMFUNCTION_V1", c2, {})
    assert not m0["initial_constraints_pass"]
    assert not m1["initial_constraints_pass"]
    assert m2["initial_constraints_pass"]
