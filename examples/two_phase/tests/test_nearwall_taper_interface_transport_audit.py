"""L1A-2t taper/interface-transport audit: identities, invariants and fail-closed gates.

Every test either verifies a discrete identity of the frozen operator sequence, a
declared taper property, a provenance/fail-closed rule, or an explicitly labelled
diagnostic observable. No test edits a frozen production source or threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import phasefield as pf  # noqa: E402
from production import impact_impulse_projection_audit as impulse  # noqa: E402
from production import impact_initialization_compatibility_audit as audit  # noqa: E402
from production import nearwall_taper_interface_transport_audit as stage  # noqa: E402
from production import timestep_policy  # noqa: E402

QUICK_N = 48


@pytest.fixture(scope="module")
def flat_bundle():
    case = audit._canary_cases()["flat_we100_ct050"]
    return audit._derive_bundle("flat_we100_ct050", case, QUICK_N, 0.004)


@pytest.fixture(scope="module")
def pillar_bundle():
    case = audit._canary_cases()["pillar_training"]
    return audit._derive_bundle("pillar_training", case, QUICK_N, 0.004)


def _state(bundle, key):
    return stage.build_candidate_state(bundle, key)[0]


def _phase_substep(bundle, state):
    p = bundle.p
    h = float(p.dt) / 3.0
    phi_rhs, _u, _v, _mu, mu_expl = pf.rhs(state, bundle.solid, p)
    return pf._phase_update(state.phi, state.u, state.v, bundle.solid, p, h, phi_rhs, mu_expl)[0]


# ---------------------------------------------------------------------------
# taper definition
# ---------------------------------------------------------------------------
def test_smooth5_endpoints_and_derivatives():
    z = jnp.asarray([-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5], dtype=jnp.float64)
    values = np.asarray(stage.smooth5(z), dtype=np.float64)
    assert values[0] == 0.0 and values[1] == 0.0 and values[-1] == 1.0 and values[-2] == 1.0
    assert values[2] == pytest.approx(6 * 0.25**5 - 15 * 0.25**4 + 10 * 0.25**3)
    zz = jnp.linspace(0.0, 1.0, 2001, dtype=jnp.float64)
    derivative = np.gradient(np.asarray(stage.smooth5(zz), dtype=np.float64), np.asarray(zz, dtype=np.float64))
    # C2 smoothness: the taper is flat (zero slope) at both ends of its transition.
    assert abs(derivative[0]) < 1e-2
    assert abs(derivative[-1]) < 1e-2
    assert np.max(np.abs(derivative)) == pytest.approx(1.875, rel=1e-4)


def test_wall_taper_zero_set_matches_declared_onset():
    sdf = jnp.asarray([-1.0, 0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0], dtype=jnp.float64) * 0.03125
    taper = np.asarray(stage.wall_taper(sdf, 0.03125, 4.0, 8.0), dtype=np.float64)
    # sdf/dx = { -1, 0, 1, 2, 4, 8, 12, 20 }: zero up to and including the onset, one from onset+width
    assert taper[0] == 0.0 and taper[4] == 0.0
    assert taper[6] == 1.0 and taper[7] == 1.0
    assert 0.0 < taper[5] < 1.0
    assert np.all(np.diff(taper) >= 0.0)
    control = np.asarray(stage.wall_taper(sdf, 0.03125, None, None), dtype=np.float64)
    assert np.all(control == 1.0)


def test_taper_family_reproduces_c2_bitwise(flat_bundle, pillar_bundle):
    for bundle in (flat_bundle, pillar_bundle):
        reference_state, _details = audit.build_candidate(bundle, "SDF_TAPERED_STREAMFUNCTION_V1")
        u, v, psi, taper, family = stage.build_tapered_velocity(bundle, 4.0, 8.0)
        assert np.array_equal(np.asarray(reference_state.u), np.asarray(u))
        assert np.array_equal(np.asarray(reference_state.v), np.asarray(v))
        assert np.array_equal(np.asarray(audit.build_c2_velocity(bundle)[2]), np.asarray(psi))
        assert float(family["amplitude_scale"]) == pytest.approx(
            float(audit.build_c2_velocity(bundle)[3]["amplitude_scale"]), rel=0.0, abs=0.0
        )
        assert np.asarray(taper).max() == 1.0


def test_no_post_differentiation_mask_and_zero_streamfunction_near_wall(flat_bundle):
    u, v, psi, taper, details = stage.build_tapered_velocity(flat_bundle, 4.0, 8.0)
    assert details["post_derivative_masking"] is False
    assert details["diagnostic_pressure_projection_applied_to_seed"] is False
    sdf = np.asarray(flat_bundle.solid.sdf, dtype=np.float64)
    inside = sdf <= 4.0 * flat_bundle.p.dx
    assert np.all(np.asarray(psi)[inside] == 0.0)
    assert np.all(np.asarray(taper)[inside] == 0.0)
    del u, v


def test_core_amplitude_and_gas_return_normalization():
    for key, expected_sign in (("C2_BASE", -1.0), ("T1_NEAR_NARROW", -1.0), ("T3_FAR_NARROW", -1.0)):
        case = audit._canary_cases()["flat_we100_ct050"]
        bundle = audit._derive_bundle("flat_we100_ct050", case, QUICK_N, 0.004)
        state = _state(bundle, key)
        volume = np.asarray(pf.phase_control_volumes(bundle.solid, bundle.p), dtype=np.float64)
        phi = np.asarray(state.phi, dtype=np.float64)
        weight = volume * (phi >= 0.9)
        core_v = float(np.sum(weight * np.asarray(state.v, dtype=np.float64)) / np.sum(weight))
        assert core_v == pytest.approx(expected_sign * bundle.u_impact, rel=1e-6)


def test_gas_return_flow_stays_bounded(flat_bundle):
    for key in stage.CANDIDATE_RUN_ORDER:
        state = _state(flat_bundle, key)
        metrics = audit.initial_metrics(flat_bundle, "SDF_TAPERED_STREAMFUNCTION_V1", state, {"candidate": key})
        # Loose declared bounds for the coarse method-smoke grid; the forensic evidence
        # reports the exact values instead of relying on this assertion.
        assert metrics["return_flow_peak_speed"] / flat_bundle.u_impact < 0.5
        assert metrics["max_speed_global"] / flat_bundle.u_impact < 3.0


# ---------------------------------------------------------------------------
# discrete operators
# ---------------------------------------------------------------------------
def test_periodic_central_operators_commute(flat_bundle):
    p = flat_bundle.p
    rng = np.random.default_rng(3)
    field = jnp.asarray(rng.standard_normal((QUICK_N, QUICK_N)), dtype=p.dtype)
    left = pf._ddy(pf._ddx(field, p.dx), p.dy)
    right = pf._ddx(pf._ddy(field, p.dy), p.dx)
    scale = float(np.max(np.abs(np.asarray(left, dtype=np.float64))))
    assert float(np.max(np.abs(np.asarray(left - right, dtype=np.float64)))) <= 1e-5 * scale


def test_independent_central_D_versus_open_face_fv(flat_bundle):
    state = _state(flat_bundle, "C2_BASE")
    metrics = audit._divergence_metrics(state.u, state.v, flat_bundle.solid, flat_bundle.p)
    assert metrics["initial_D_div_Linf"] < metrics["cutcell_flux_div_Linf"] + 1e-3
    assert metrics["cutcell_flux_divergence_status"] == "MEASURED_OPEN_FACE_RECONSTRUCTION"
    # The two reconstructions are different objects and must never be silently substituted.
    assert metrics["cutcell_flux_reconstruction"]
    assert metrics["D_div_L2_fluid_volume_rms"] is not None


def test_zero_blocked_face_flux_and_telescoping(flat_bundle):
    p = flat_bundle.p
    solid = flat_bundle.solid
    state = _state(flat_bundle, "C2_BASE")
    operator = pf.phase_transport_operator(solid, p)
    flux_x, flux_y = pf.phase_advective_fluxes(state.u, state.v, state.phi, solid, p)
    aperture_x = np.asarray(operator.aperture_x, dtype=np.float64)
    aperture_y = np.asarray(operator.aperture_y, dtype=np.float64)
    if (aperture_x <= 0.0).any():
        assert np.all(np.asarray(flux_x, dtype=np.float64)[aperture_x <= 0.0] == 0.0)
    if (aperture_y <= 0.0).any():
        assert np.all(np.asarray(flux_y, dtype=np.float64)[aperture_y <= 0.0] == 0.0)
    divergence = pf.control_volume_divergence(flux_x, flux_y, operator.volume_safe)
    volume = np.asarray(operator.volume, dtype=np.float64)
    scale = float(np.sum(volume * np.abs(np.asarray(divergence, dtype=np.float64))))
    assert abs(float(np.sum(volume * np.asarray(divergence, dtype=np.float64)))) <= 1e-12 * max(scale, 1.0)


def test_wall_and_solid_constraints_are_measured_not_assumed(flat_bundle, pillar_bundle):
    for bundle in (flat_bundle, pillar_bundle):
        state = _state(bundle, "C2_BASE")
        metrics = audit._wall_metrics(state.u, state.v, bundle.solid, bundle.p)
        # The coarse grid may contain no cell deeper than -2 dx; the measurement must still be
        # reported with its count instead of being silently replaced by a synthetic zero.
        assert metrics["deep_solid_cell_count"] >= 0
        assert metrics["wall_area_cell_count"] > 0
        assert metrics["max_abs_velocity_deep_solid"] >= 0.0
        assert metrics["embedded_wall_flux_status"].startswith("MEASURED")


def test_contour_distance_sign_and_normal():
    case = audit._canary_cases()["flat_we100_ct050"]
    bundle = audit._derive_bundle("flat_we100_ct050", case, QUICK_N, 0.004)
    support = audit._local_gap(np.asarray(bundle.base.phi), 0.5, bundle.solid, bundle.p)
    assert support["gap"] > 0.0
    normal = support["local_wall_normal_into_solid"]
    assert normal is not None
    assert np.linalg.norm(normal) == pytest.approx(1.0, rel=1e-12)
    # The frozen convention is positive into the solid: moving along the normal must
    # decrease the SDF (the flat wall normal points downward).
    sdf = np.asarray(bundle.solid.sdf, dtype=np.float64)
    point = support["support_xy"]
    step = 0.25 * float(bundle.p.dx)
    before = audit._periodic_bilinear_sample(sdf, np.asarray([point[0]]), np.asarray([point[1]]), bundle.p)[0]
    after = audit._periodic_bilinear_sample(
        sdf,
        np.asarray([point[0] + step * normal[0]]),
        np.asarray([point[1] + step * normal[1]]),
        bundle.p,
    )[0]
    assert after < before


def test_rigid_contour_translation_is_recovered(flat_bundle):
    """A synthetic rigid shift must be returned by the matched-contour estimator."""
    p = flat_bundle.p
    x_axis = (np.arange(p.Nx, dtype=np.float64) + 0.5) * float(p.dx)
    y_axis = (np.arange(p.Ny, dtype=np.float64) + 0.5) * float(p.dy)
    X, Y = np.meshgrid(x_axis, y_axis, indexing="ij")
    center_x, center_y, radius = 3.0, 2.0, 0.8
    shift = 0.4 * float(p.dx)
    before = 0.5 * (1.0 - np.tanh((np.hypot(X - center_x, Y - center_y) - radius) / (1.5 * p.dx)))
    after = 0.5 * (1.0 - np.tanh((np.hypot(X - center_x, Y - center_y + shift) - radius) / (1.5 * p.dx)))
    support = audit._local_gap(before, 0.5, flat_bundle.solid, p)
    support_after = audit._local_gap(after, 0.5, flat_bundle.solid, p)
    motion = stage.contour_motion_audit(flat_bundle, "SYNTHETIC", before, after, support, support_after, 1.0)
    matched = motion["matched_contour_motion"]
    assert matched["usable_match_fraction"] > 0.9
    # The rigid shift moves the contour by exactly `shift`; the median displacement must follow.
    assert matched["median_displacement_over_dx"] == pytest.approx(shift / p.dx, rel=0.15)


def test_independent_contour_reconstruction_agrees(flat_bundle):
    phi = np.asarray(flat_bundle.base.phi, dtype=np.float64)
    official = audit._threshold_contour_edge_crossings(phi, 0.5, flat_bundle.p)
    independent = stage.independent_contour_points(phi, 0.5, flat_bundle.p)
    assert official.shape[0] > 0 and independent.shape[0] > 0
    index, distance = stage._match_points(independent, official, flat_bundle.p.Lx, flat_bundle.p.Ly)
    assert np.mean(distance <= 0.5 * flat_bundle.p.dx) > 0.95
    del index


def test_support_switch_and_small_gradient_guards(flat_bundle):
    p = flat_bundle.p
    flat_field = np.full((QUICK_N, QUICK_N), 0.5, dtype=np.float64)
    assert stage.independent_contour_points(flat_field, 0.5, p).size == 0
    support = audit._local_gap(np.asarray(flat_bundle.base.phi), 0.5, flat_bundle.solid, p)
    guard = stage._linearized_normal_advance(flat_bundle, flat_field, flat_field + 1e-3, support, 1.0)
    assert guard["status"] == "UNDEFINED_POOR_CONDITIONING"
    assert guard["refused_division"] is True
    motion = stage.contour_motion_audit(
        flat_bundle,
        "NO_CONTOUR",
        flat_field,
        flat_field + 1e-3,
        support,
        audit._local_gap(flat_field, 0.5, flat_bundle.solid, p),
        1.0,
    )
    assert motion["matched_contour_motion"]["status"] == "UNDEFINED_NO_CONTOUR"


def test_bilinear_velocity_interpolation_is_periodic(flat_bundle):
    p = flat_bundle.p
    u = np.asarray(_state(flat_bundle, "C2_BASE").u, dtype=np.float64)
    left = audit._periodic_bilinear_sample(u, np.asarray([0.0]), np.asarray([1.0]), p)[0]
    right = audit._periodic_bilinear_sample(u, np.asarray([p.Lx]), np.asarray([1.0]), p)[0]
    assert left == pytest.approx(right, rel=1e-12)
    x_axis = (np.arange(p.Nx, dtype=np.float64) + 0.5) * float(p.dx)
    exact = audit._periodic_bilinear_sample(u, np.asarray([x_axis[3]]), np.asarray([x_axis[4]]), p)[0]
    assert exact == pytest.approx(u[3, 4], rel=1e-12)


# ---------------------------------------------------------------------------
# phase budget identity
# ---------------------------------------------------------------------------
def test_phase_component_ledger_closes_and_reproduces_phase_update(flat_bundle, pillar_bundle):
    for bundle in (flat_bundle, pillar_bundle):
        ledger = stage.phase_component_ledger(bundle, "C2_BASE", _state(bundle, "C2_BASE"), n_substeps=2)
        assert ledger["active_branch"]["subcycling_active"] is False
        for row in ledger["substeps"]:
            assert row["phase_update_bitwise_reproduced"] is True
            assert row["advective_rate_equals_rhs_phi_rhs"] is True
            assert row["closure_linf_active_cells"] <= 1e-14
            assert abs(row["weighted_mass_budget"]["closure_residual"]) <= 1e-12
            assert row["blocked_face_flux_max_abs"]["advective"] == 0.0
            assert row["blocked_face_flux_max_abs"]["ch"] == 0.0


def test_ch_flux_is_zero_on_blocked_faces_and_stiffness_is_symmetric(flat_bundle):
    p = flat_bundle.p
    solid = flat_bundle.solid
    state = _state(flat_bundle, "C2_BASE")
    _mu, _u, _v, _mu_full, mu_expl = pf.rhs(state, solid, p)
    ch_x, ch_y = pf.chemical_potential_fluxes(mu_expl, solid, p)
    weight_x, weight_y = pf.phase_face_weights(solid, p)
    blocked_x = np.asarray(weight_x, dtype=np.float64) == 0.0
    blocked_y = np.asarray(weight_y, dtype=np.float64) == 0.0
    assert np.all(np.asarray(ch_x, dtype=np.float64)[blocked_x] == 0.0)
    assert np.all(np.asarray(ch_y, dtype=np.float64)[blocked_y] == 0.0)
    # K is symmetric: <K a, b> == <a, K b> on the shared face weights.
    rng = np.random.default_rng(11)
    a = jnp.asarray(rng.standard_normal((QUICK_N, QUICK_N)), dtype=p.dtype)
    b = jnp.asarray(rng.standard_normal((QUICK_N, QUICK_N)), dtype=p.dtype)
    left = float(
        np.sum(
            np.asarray(pf.graph_stiffness_apply(a, weight_x, weight_y), dtype=np.float64)
            * np.asarray(b, dtype=np.float64)
        )
    )
    right = float(
        np.sum(
            np.asarray(a, dtype=np.float64)
            * np.asarray(pf.graph_stiffness_apply(b, weight_x, weight_y), dtype=np.float64)
        )
    )
    # Algebraic symmetry of the shared-face assembly; the residual (~3e-8 relative) is
    # float32 accumulation of the 48*48 products, not a structural asymmetry.
    assert left == pytest.approx(right, rel=1e-5)
    assert abs(left - right) / max(abs(left), 1.0) < 1e-7


def test_advection_subcycling_branch_flag_is_explicit(flat_bundle):
    p = flat_bundle.p
    active = pf.phase_advection_subcycles(p)
    assert active is bool(getattr(p, "phase_advection_subcycling", pf.PHASE_ADVECTION_SUBCYCLING) != "disabled")
    state = _state(flat_bundle, "C2_BASE")
    h = float(p.dt) / 3.0
    source = pf.advective_phase_source(state.phi, state.u, state.v, flat_bundle.solid, p, h)
    flux_x, flux_y = pf.phase_advective_fluxes(state.u, state.v, state.phi, flat_bundle.solid, p)
    divergence = pf.control_volume_divergence(
        flux_x, flux_y, pf.phase_transport_operator(flat_bundle.solid, p).volume_safe
    )
    if not active:
        assert np.array_equal(np.asarray(source), np.asarray(-divergence))


def test_zero_fluid_ch_only_evolution_changes_phase_only(flat_bundle):
    p = flat_bundle.p
    h = float(p.dt) / 3.0
    state = _state(flat_bundle, "C2_BASE")
    substep = stage.make_ch_only_phase_substep(flat_bundle.solid, p, h, tuple(np.shape(state.phi)))
    updated = substep(state.phi)
    assert not np.array_equal(np.asarray(updated), np.asarray(state.phi))
    volume = np.asarray(pf.phase_control_volumes(flat_bundle.solid, p), dtype=np.float64)
    before = float(np.sum(volume * np.asarray(state.phi, dtype=np.float64)))
    after = float(np.sum(volume * np.asarray(updated, dtype=np.float64)))
    assert abs(after - before) <= 1e-9 * max(abs(before), 1.0)


def test_frozen_velocity_phase_substep_uses_frozen_velocity(flat_bundle):
    p = flat_bundle.p
    h = float(p.dt) / 3.0
    state = _state(flat_bundle, "C2_BASE")
    substep = stage.make_frozen_velocity_phase_substep(flat_bundle.solid, p, h, state.u, state.v)
    updated = substep(state.phi)
    phi_rhs, _u, _v, _mu, mu_expl = pf.rhs(state, flat_bundle.solid, p)
    reference = pf._phase_update(state.phi, state.u, state.v, flat_bundle.solid, p, h, phi_rhs, mu_expl)[0]
    # The jitted frozen-velocity kernel fuses differently from the eager reference, so the
    # comparison uses a tight float64 tolerance rather than bitwise equality.
    assert np.allclose(np.asarray(updated), np.asarray(reference), rtol=0.0, atol=1e-13)


# ---------------------------------------------------------------------------
# momentum ledger and projection identity
# ---------------------------------------------------------------------------
def test_fast_kernel_reproduces_production_step(flat_bundle):
    state = _state(flat_bundle, "C2_BASE")
    result = impulse.validate_ledger(state, flat_bundle.solid, flat_bundle.p)
    assert result["fast_kernel_within_dtype_tolerance"] == {
        "phi": True,
        "u": True,
        "v": True,
    }
    assert all(result["rhs_recomposition_bitwise"])


def test_same_state_momentum_stage_ledger_matches_production(flat_bundle):
    state = _state(flat_bundle, "C2_BASE")
    ledger = stage.momentum_stage_ledger(flat_bundle, "C2_BASE", state, n_substeps=1)
    row = ledger["substeps"][0]
    assert row["cg_converged"] is True
    assert max(row["fast_kernel_vs_capture_state_max_abs_delta"].values()) <= 1e-6
    stages = row["approach_speed_by_velocity_stage_into_solid"]
    assert stages["before"] is not None
    for name in ("explicit_rhs", "brinkman", "periodic_pressure_projection"):
        assert row["approach_delta_by_operator"][name] is not None


def test_projection_plain_grid_mean_identity_and_local_redistribution(flat_bundle):
    state = _state(flat_bundle, "C2_BASE")
    ledger = stage.momentum_stage_ledger(flat_bundle, "C2_BASE", state, n_substeps=1)
    identity = ledger["substeps"][0]["projection_plain_grid_mean_identity"]
    assert abs(identity["mean_of_proj_correction"]) <= 1e-9
    assert abs(identity["mean_ddx_pressure"]) <= 1e-9
    assert abs(identity["mean_ddy_pressure"]) <= 1e-9


# ---------------------------------------------------------------------------
# initializer compatibility, quick vs saved evidence
# ---------------------------------------------------------------------------
def test_t0_phase_and_geometry_are_shared_across_candidates(flat_bundle):
    hashes = {}
    for key in (*stage.CANDIDATE_RUN_ORDER, "NC_NO_WALL_TAPER"):
        state = _state(flat_bundle, key)
        hashes[key] = (
            audit._array_sha(np.asarray(state.phi)),
            audit._geometry_hash(flat_bundle.solid),
        )
    assert len(set(hashes.values())) == 1


def test_saved_evidence_comparison_fails_closed_on_the_wrong_grid(flat_bundle):
    """The saved L1A-2s hashes are N=192 quantities: the quick grid must refuse them."""
    saved = json.loads((stage.L1A2S_DIR / "initializer_constraint_matrix.json").read_text())
    with pytest.raises(stage.AuditBlocked):
        stage.c2_exact_reproduction(flat_bundle, saved, compare_saved_evidence=True)
    record = stage.c2_exact_reproduction(flat_bundle, saved, compare_saved_evidence=False)
    assert record["saved_evidence_compared"] is False
    assert record["hash_checks"]["status"] is False
    assert record["u_bitwise_equal_to_l1a2s_builder"] is True


def test_ineligible_candidate_is_excluded_from_event_runs(flat_bundle):
    """T1/T2 break the frozen near-wall gate and may only appear as quarantined controls."""
    state = _state(flat_bundle, "T1_NEAR_NARROW")
    metrics = audit.initial_metrics(flat_bundle, "SDF_TAPERED_STREAMFUNCTION_V1", state, {})
    assert metrics["initial_constraints_pass"] is False
    assert metrics["acceptance_checks"]["nearwall_chi_velocity"] is False
    assert metrics["acceptance_checks"]["deep_solid_velocity"] is True


def test_compatible_candidate_passes_every_frozen_gate(flat_bundle, pillar_bundle):
    for bundle in (flat_bundle, pillar_bundle):
        for key in ("C2_BASE", "T3_FAR_NARROW"):
            state = _state(bundle, key)
            metrics = audit.initial_metrics(bundle, "SDF_TAPERED_STREAMFUNCTION_V1", state, {})
            assert metrics["initial_constraints_pass"] is True, (bundle.case_name, key)


def test_contact_authentication_cannot_borrow_another_variant(flat_bundle):
    """A run without its own contact/postcontact rows must not authenticate."""
    rows = [
        {
            "t": 0.0,
            "step": 0,
            "contact_phi05": False,
            "gap_phi05": 0.4,
            "gap_phi01": 0.5,
            "gap_phi05_over_dx": 3.0,
            "local_approach_speed_into_solid": 0.2,
            "v_liquid": -0.3,
            "beta": 1.0,
            "drop_height": 1.4,
            "contact_line_x_center": 3.0,
        }
    ]
    verdict = audit.authenticate_impact({"rows": rows}, flat_bundle, observed_horizon=0.8)
    assert verdict["verdict"] == "NO_CONTACT_WITHIN_OBSERVED_WINDOW"
    assert verdict["contact_time"] is None


def test_event_matrix_marks_missing_stages_without_promotion():
    decision = stage.decide(profile="forensic", initial_matrix=None, tier3=None, temporal=None, budget=None)
    assert decision["stage_decision"] == "PREREQUISITE_BLOCKED"
    assert decision["taper_mechanism_verdict"] == "INCONCLUSIVE"


def test_quick_profile_can_never_produce_a_physical_verdict():
    decision = stage.decide(profile="quick", initial_matrix=None, tier3=None, temporal=None, budget=None)
    assert decision["quick_only"] is True
    assert decision["physical_verdict_valid"] is False
    assert decision["taper_mechanism_verdict"] == "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# provenance, thresholds and freeze rules
# ---------------------------------------------------------------------------
def test_frozen_thresholds_and_contract_unchanged():
    assert int(pf.SOLVER_CONTRACT_VERSION) == 12
    assert timestep_policy.DEFAULT_POLICY_NAME == "impact_phase_cap_dx2_v1"
    assert audit.CONTACT_GAP_CELLS == 1.5
    assert audit.APPROACH_FLOOR_FRACTION == 0.2
    assert audit.POSTCONTACT_WINDOW == 0.24
    assert audit.DATASET_HORIZON == 8.0
    assert stage.PRIMARY_EVENT_HORIZON == 0.8
    assert stage.TARGETED_EXTENSION_HORIZON == 1.2
    assert stage.DT_LEVELS == (0.002, 0.001, 0.0005)


def test_declared_candidate_set_is_exactly_the_prespecified_one():
    assert stage.CANDIDATE_RUN_ORDER == ("C2_BASE", "T1_NEAR_NARROW", "T2_NEAR_WIDE", "T3_FAR_NARROW")
    assert set(stage.TAPER_CANDIDATES) == set(stage.CANDIDATE_RUN_ORDER)
    assert stage.TAPER_CANDIDATES["C2_BASE"]["onset_a_dx"] == 4.0
    assert stage.TAPER_CANDIDATES["C2_BASE"]["width_b_dx"] == 8.0
    assert stage.TAPER_CANDIDATES["T1_NEAR_NARROW"]["onset_a_dx"] == 1.0
    assert stage.TAPER_CANDIDATES["T2_NEAR_WIDE"]["width_b_dx"] == 8.0
    assert stage.TAPER_CANDIDATES["T3_FAR_NARROW"]["onset_a_dx"] == 4.0
    assert stage.QUARANTINED_CONTROLS["NC_NO_WALL_TAPER"]["eligible"] is False
    assert Path(stage.__file__).name == "nearwall_taper_interface_transport_audit.py"


def test_unknown_candidate_is_rejected(flat_bundle):
    """No extra tuned candidate may be constructed outside the prespecified set."""
    with pytest.raises(stage.AuditBlocked):
        stage.build_candidate_state(flat_bundle, "C2_TUNED_EXTRA")
    with pytest.raises(stage.AuditBlocked):
        stage.build_candidate_state(flat_bundle, "T4_EXTRA_TUNED")


def test_frozen_sources_are_hash_bound_to_the_l1a2s_manifest():
    manifest = json.loads((stage.L1A2S_DIR / "manifest.json").read_text())
    current = audit._source_hashes()
    for key in ("phasefield", "cases", "generate_dataset", "timestep_policy"):
        assert current[key] == manifest["source_hashes_current"][key], key


def test_worktree_guard_rejects_foreign_changes():
    assert stage._unexpected_worktree_changes("") == []
    allowed = "?? examples/two_phase/evidence/l1a2t/manifest.json"
    assert stage._unexpected_worktree_changes(allowed) == []
    # Same entry with its leading status space already stripped (git output is trimmed upstream).
    assert stage._unexpected_worktree_changes(allowed.lstrip()) == []
    assert (
        stage._unexpected_worktree_changes(
            "M examples/two_phase/tests/test_nearwall_taper_interface_transport_audit.py"
        )
        == []
    )
    foreign = " M examples/two_phase/phasefield.py"
    assert stage._unexpected_worktree_changes(foreign) == [foreign.strip()]
    assert stage._unexpected_worktree_changes("M quality/legacy_two_phase_debt_v1.json") == [
        "M quality/legacy_two_phase_debt_v1.json"
    ]


def test_manifest_mismatch_reference_is_recorded_in_preflight():
    preflight = stage.preflight_state()
    assert preflight["checks"]["stage_has_not_touched_frozen_sources"] is True
    assert preflight["frozen_evidence_input_sha256"]
    assert preflight["prerequisite"]["verified_merge_sha"] == stage.VERIFIED_MAIN_SHA
    assert preflight["source_freeze_checks"]["phasefield"] is True


def test_dt_invariant_initial_fields():
    cases = audit._canary_cases()
    hashes = []
    for requested_dt in stage.DT_LEVELS:
        bundle = audit._derive_bundle("flat_we100_ct050", cases["flat_we100_ct050"], QUICK_N, requested_dt)
        state = _state(bundle, "C2_BASE")
        hashes.append(
            (
                audit._array_sha(np.asarray(state.phi)),
                audit._array_sha(np.asarray(state.u)),
                audit._array_sha(np.asarray(state.v)),
            )
        )
    assert len(set(hashes)) == 1


def _band_record(bundle, key):
    spec = stage.TAPER_CANDIDATES[key]
    u, v, psi, taper, _ = stage.build_tapered_velocity(bundle, spec["onset_a_dx"], spec["width_b_dx"])
    support = audit._local_gap(np.asarray(bundle.base.phi), 0.5, bundle.solid, bundle.p)
    record = stage.distance_band_matrix(bundle, pf.State(phi=bundle.base.phi, u=u, v=v, t=0.0), psi, taper, support)
    return support, record


def test_front_window_is_a_periodic_x_band(flat_bundle):
    """Regression: a 1-D mask broadcast against ``[x, y]`` fields selected a y band."""
    support = audit._local_gap(np.asarray(flat_bundle.base.phi), 0.5, flat_bundle.solid, flat_bundle.p)
    window = stage._front_window(support, flat_bundle.p)
    assert window.shape == (flat_bundle.p.Nx, 1)
    columns = np.flatnonzero(window[:, 0])
    assert columns.size == 2 * stage.FRONT_WINDOW_CELLS + 1
    i0 = int(support["gap_cell"][0])
    expected = sorted((np.arange(i0 - stage.FRONT_WINDOW_CELLS, i0 + stage.FRONT_WINDOW_CELLS + 1)) % flat_bundle.p.Nx)
    assert columns.tolist() == expected


def test_distance_band_matrix_measures_real_bands_and_the_taper(flat_bundle):
    support, bands = _band_record(flat_bundle, "C2_BASE")
    assert set(bands["bands"]) == {"0_1", "1_2", "2_3", "3_4", "4_6", "6_12"}
    assert all(entry["status"] == "MEASURED" for entry in bands["bands"].values())
    assert bands["at_frozen_support"]["wall_taper"] == 0.0
    assert bands["at_frozen_support"]["approach_speed_over_u_impact"] == 0.0
    support_band = bands["bands"]["2_3"]
    assert support_band["mean_taper"] == 0.0
    assert support_band["max_abs_speed_over_u_impact"] == 0.0
    _, t1 = _band_record(flat_bundle, "T1_NEAR_NARROW")
    assert t1["bands"]["2_3"]["mean_taper"] > 0.1
    assert t1["bands"]["2_3"]["max_abs_speed_over_u_impact"] > 0.1
    assert t1["bands"]["0_1"]["mean_taper"] == 0.0


def test_frozen_gate_location_is_an_open_diffuse_fluid_cell(flat_bundle):
    """The frozen chi>=0.01 gate maximum sits in an open cell of the diffuse band."""
    _, t1 = _band_record(flat_bundle, "T1_NEAR_NARROW")
    gate = t1["frozen_gate_location_diagnostic"]
    assert gate["status"] == "MEASURED"
    assert 0.0 < gate["argmax_chi"] < 0.5
    assert gate["argmax_control_volume_fraction"] == 1.0
    assert 1.5 <= gate["argmax_sdf_over_dx"] <= 3.5
    assert gate["argmax_wall_taper"] > 0.0
    assert gate["argmax_speed_over_u_impact"] > 0.1


def test_compact_row_supports_authentication_end_to_end(flat_bundle):
    """Regression: every key the frozen authentication reads must survive compaction."""
    p = flat_bundle.p
    state = _state(flat_bundle, "C2_BASE")
    row = audit._trajectory_row(
        state.phi, state.u, state.v, flat_bundle.solid, p, flat_bundle, 0, 0.0, include_constraints=False
    )
    compact = stage._compact_row(row)
    assert "local_support_phi05" in compact
    assert "gap_phi01_over_eps" in compact
    verdict = audit.authenticate_impact(
        {"rows": stage._authentication_rows([compact])}, flat_bundle, observed_horizon=0.0
    )
    assert verdict["verdict"] == "NO_CONTACT_WITHIN_OBSERVED_WINDOW"


def test_extension_plumbing_stays_inside_the_declared_horizon(flat_bundle):
    """A run without contact must not claim an extended event label; the extension stays bounded."""
    run = stage.run_tapered_trajectory(bundle=flat_bundle, key="C2_BASE", t_end=0.008, profile="quick")
    assert run["stop_reason"] == "horizon_reached"
    assert "primary_horizon_authentication" not in run
    assert "impact_authentication_scope" not in run
    final_state = run["_final_state"]
    final_step = round(run["t_end_actual"] / float(flat_bundle.p.dt))
    extension = stage._extend_trajectory(
        flat_bundle, "C2_BASE", final_state, final_step, 0.016, float(audit.POSTCONTACT_WINDOW)
    )
    assert extension["extension_reason"]
    assert extension["contact_step"] is None
    assert len(extension["rows"]) > 0
    assert extension["extension_horizon"] == 0.016


def test_required_evidence_files_are_the_declared_deliverables():
    assert set(stage.REQUIRED_EVIDENCE_FILES) == {
        "prerequisite_and_source_map.json",
        "taper_candidate_design_contract.json",
        "candidate_initial_compatibility_matrix.json",
        "front_velocity_distance_band_matrix.json",
        "phase_advective_CH_implicit_budget.json",
        "interface_contour_motion_audit.json",
        "first_substep_operator_ledger.json",
        "bounded_candidate_event_matrix.json",
        "temporal_sensitivity_matrix.json",
        "counterfactual_and_falsification_matrix.json",
        "quality_and_blocker_matrix.json",
        "mechanism_decision.json",
        "l1a2t_final_report.json",
        "l1a2t_final_report.md",
    }
    assert "manifest.json" not in stage.REQUIRED_EVIDENCE_FILES
    assert len(stage.REQUIRED_EVIDENCE_FILES) == 14


def test_verify_evidence_manifest_fails_closed(tmp_path):
    payload = {"stage": 1}
    for name in stage.REQUIRED_EVIDENCE_FILES:
        (tmp_path / name).write_text(json.dumps(payload))
    manifest_entries = {name: {"sha256": stage.sha256_file(tmp_path / name)} for name in stage.REQUIRED_EVIDENCE_FILES}
    manifest = {
        "stage": stage.STAGE,
        "stage_version": stage.STAGE_VERSION,
        "evidence_class": "SYNTHETIC_TEST_FIXTURE",
        "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
        "timestep_policy": stage.EXPECTED_POLICY,
        "no_promotion": True,
        "evidence_files": manifest_entries,
    }
    stage.write_json(tmp_path / "manifest.json", manifest)
    report = stage.verify_evidence_manifest(tmp_path)
    assert report["verified"] is True
    assert report["bound_files"] == 14

    (tmp_path / "bounded_candidate_event_matrix.json").write_text(json.dumps({"tampered": True}))
    with pytest.raises(stage.AuditBlocked, match="SHA256_MISMATCH"):
        stage.verify_evidence_manifest(tmp_path)
    (tmp_path / "bounded_candidate_event_matrix.json").write_text(json.dumps(payload))
    (tmp_path / "l1a2t_final_report.md").unlink()
    with pytest.raises(stage.AuditBlocked, match="MISSING"):
        stage.verify_evidence_manifest(tmp_path)
    (tmp_path / "l1a2t_final_report.md").write_text(json.dumps(payload))
    (tmp_path / "unbound.json").write_text("{}")
    with pytest.raises(stage.AuditBlocked, match="UNBOUND_FILE_PRESENT"):
        stage.verify_evidence_manifest(tmp_path)
    (tmp_path / "unbound.json").unlink()
    manifest["no_promotion"] = False
    stage.write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(stage.AuditBlocked, match="no_promotion"):
        stage.verify_evidence_manifest(tmp_path)


def test_validation_payload_is_source_bound(tmp_path):
    path = tmp_path / "validation.json"
    path.write_text(json.dumps({"stage_source_sha256": "0" * 64}))
    with pytest.raises(stage.AuditBlocked, match="does not match the running module"):
        stage.load_validation_payload(path)
    path.write_text(json.dumps({"stage_source_sha256": stage.sha256_file(Path(stage.__file__)), "stage": stage.STAGE}))
    loaded = stage.load_validation_payload(path)
    assert loaded["stage"] == stage.STAGE
    path.write_text(json.dumps({"stage": stage.STAGE}))
    with pytest.raises(stage.AuditBlocked, match="stage_source_sha256"):
        stage.load_validation_payload(path)
    assert stage.load_validation_payload(None) is None


def test_render_report_sections_a_to_r_in_order():
    initial = {}
    for case in stage.CASE_NAMES:
        candidates = {}
        for key in stage.CANDIDATE_RUN_ORDER:
            candidates[key] = {
                "candidate": key,
                "frozen_initial_acceptance": {
                    "checks": {"nearwall_chi_velocity": True},
                    "dimensionless_residuals": {"nearwall_chi_speed_over_u_impact": 0.0},
                    "pass": True,
                },
                "initial_front_contour": {
                    "local_normal_approach_speed_into_solid": 0.0,
                    "exceeds_frozen_approach_floor": False,
                },
            }
        initial[case] = {
            "candidates": candidates,
            "quarantined_controls": {
                "NC_NO_WALL_TAPER": {
                    "initial_front_contour": {"local_normal_approach_speed_into_solid": 0.3},
                    "frozen_initial_acceptance": {
                        "pass": False,
                        "dimensionless_residuals": {"nearwall_chi_speed_over_u_impact": 0.9},
                    },
                }
            },
        }
    decision = {
        "taper_mechanism_verdict": "TAPER_COMPATIBILITY_TRADEOFF",
        "candidate_outcome": "COMPATIBLE_IMPACT_BUT_QUALITY_OR_REFINEMENT_FAIL",
        "stage_decision": "DIAGNOSTIC_COMPLETE_NO_PROMOTION",
        "next_action": {"action": "one bounded action", "bounded_scope": "initializer design only"},
    }
    preflight = {
        "prerequisite": {
            "verified_merge_sha": stage.VERIFIED_MAIN_SHA,
            "verified_sha_parents": ["a", "b"],
            "branch": "arena/test",
            "merge_base_sha": stage.VERIFIED_MAIN_SHA,
            "worktree_clean_within_stage_deliverables": True,
            "commits_since_verified_main": [],
            "pr22_hosted_python_ci_run": 1,
            "pr22_hosted_python_ci_status": "recorded",
        },
        "source_freeze_checks": {"phasefield": True},
    }
    markdown = stage.render_report(
        profile="forensic",
        decision=decision,
        preflight=preflight,
        run_meta={"N": 192},
        initial_matrix=initial,
        band_matrix=None,
        tier2=None,
        tier3=None,
        temporal=None,
        falsification=None,
        reproduction={},
        validation={"status": "SYNTHETIC"},
    )
    headings = [line for line in markdown.splitlines() if line.startswith("## ")]
    letters = [f"{chr(ord('A') + index)}." for index in range(18)]
    assert len(headings) == 18
    assert all(heading.startswith(f"## {letter}") for heading, letter in zip(headings, letters, strict=True))
    assert "SYNTHETIC" in markdown
    assert "CANDIDATE_PROMOTED=false" in markdown
    assert "L1B_DATA=L1B_DATA_NOT_READY" in markdown


@pytest.mark.slow
def test_forensic_gate_reproduces_saved_c2_hashes():
    cases = audit._canary_cases()
    saved = json.loads((stage.L1A2S_DIR / "initializer_constraint_matrix.json").read_text())
    for name in stage.CASE_NAMES:
        bundle = audit._derive_bundle(name, cases[name], 192, 0.004)
        record = stage.c2_exact_reproduction(bundle, saved, compare_saved_evidence=True)
        assert all(record["hash_checks"].values()), name
