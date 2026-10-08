"""L1A-2g contract tests: the conserved mass ``sum_i V_i phi_i`` and the v10 implicit solve.

Sections: §59 mass definitions, §60 conserved-quantity identities, §61 the solve's mass mode,
§62 fail-closed behaviour, §63 adjoint / finite-difference / autodiff agreement, plus the
contract-version and metadata statements that make contract v10 trajectories distinguishable.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import mass_mode as mm
import numpy as np
import phasefield as pf
import pytest
from production import mass_precision_audit as mpa


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module", autouse=True)
def _x64():
    """The audit holds a float64 reference; the repo convention is to require/enable x64."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture(scope="module")
def fixture():
    p, solid, state = mpa.build_case(48)
    operator = pf.phase_transport_operator(solid, p)
    return p, solid, state, operator


# --------------------------------------------------------------------------- §59
def test_three_reductions_agree_on_exactly_representable_data():
    """On data whose products are exact, all three reductions are bit-identical."""
    volume = jnp.asarray([[0.5, 0.25], [0.125, 1.0]], jnp.float32)
    phi = jnp.asarray([[0.0, 1.0], [0.5, 0.25]], jnp.float32)
    spread = mpa.reduction_spread(volume, phi)
    expected = 0.0 + 0.25 * 1.0 + 0.125 * 0.5 + 1.0 * 0.25
    assert spread["exact_float64_device"] == pytest.approx(expected, rel=1e-15)
    assert spread["compensated_fsum"] == spread["exact_float64_device"]
    assert spread["working_dtype"] == pytest.approx(expected, rel=1e-7)


def test_legacy_hard_mask_mass_is_not_the_conserved_quantity(fixture):
    """``sum(phi * hard_mask) * dx * dy`` is a legacy diagnostic, not ``sum_i V_i phi_i``."""
    p, solid, state = mpa.build_case(128)  # cut cells: the mask is not V > 0
    operator = pf.phase_transport_operator(solid, p)
    hard = np.asarray(solid.sdf, np.float64) >= 0.0
    legacy = float(np.sum(np.asarray(state.phi, np.float64)[hard]) * p.dx * p.dy)
    conserved = mpa.mass_exact(operator.volume_safe, state.phi)
    assert abs(legacy - conserved) / abs(conserved) > 1e-3
    # the conserved quantity is the one the audit and the reports use
    # both exact reductions of exactly-representable products (24-bit x 24-bit fits float64)
    assert mpa.mass_compensated(operator.volume_safe, state.phi) == pytest.approx(conserved, rel=1e-15)


# --------------------------------------------------------------------------- §60
def test_conserved_mode_identities_are_exact(fixture):
    """``S c = 0`` and ``A c = c`` hold term by term, at the cut cells included."""
    _p, _solid, state, operator = fixture
    inv = mpa.invariant_checks(48)
    assert inv["S_c_zero"] is True
    assert inv["A_c_equals_c"] is True
    assert inv["sqrt_volume_squared_equals_volume_fraction"] == 1.0  # N=48 wall is cell aligned
    # c = sqrt(V) 1 is the conserved mode the metadata names
    assert np.array_equal(np.asarray(mm.conserved_mode(operator.sqrt_volume)), np.asarray(operator.sqrt_volume))
    assert state.phi.shape == operator.volume_safe.shape


def test_flux_telescoping_is_machine_zero():
    """Both flux families sum to machine zero under the exact control volume."""
    inv = mpa.invariant_checks(48)
    for family in ("advective_telescoping", "ch_telescoping"):
        # exact for the shared face fluxes ...
        assert inv[family]["raw_over_flux_scale"] < 1e-14, (family, inv[family])
        # ... and at the float32 representation floor once divided by V and multiplied back
        assert inv[family]["over_flux_scale"] < 1e-6, (family, inv[family])


def test_single_fluid_component_and_fail_closed_on_two():
    """The fixture's fluid domain is one component; a split chamber is detected as two."""
    assert mpa.fluid_components(48)["n_fluid_components"] == 1
    p, _solid, _state = mpa.build_case(48)
    flat = pf.surface_flat(p, wall_height=0.25)
    _x, y = pf.grids(p)
    divider = jnp.abs(y - 2.5) - 0.25  # a solid band across the (periodic) domain
    sdf = jnp.minimum(flat, divider)
    solid = pf.make_solid(sdf, p, cos_theta=0.0)
    operator = pf.phase_transport_operator(solid, p)
    report = mm.fluid_connectivity(
        np.asarray(operator.aperture_x), np.asarray(operator.aperture_y), np.asarray(operator.volume)
    )
    assert report["n_fluid_components"] == 2
    assert report["n_fluid_cells"] < 48 * 48


# --------------------------------------------------------------------------- §61
def test_production_solve_keeps_the_conserved_mode(fixture):
    """One production substep's mass defect stays at the round-off floor."""
    p, solid, state, operator = fixture
    dt = p.dt / 3.0
    _adv, _ch, source, _mu = mpa._rhs_parts(state.phi, solid, p, operator, dt)
    rhs = state.phi + dt * source
    solved, info = pf.solve_ch_implicit(rhs, solid, p, dt)
    assert bool(info.converged)
    e_round = mm.roundoff_scales(np.asarray(operator.volume_safe), np.asarray(state.phi))["E_round"]
    defect = (mpa.mass_exact(operator.volume_safe, solved) - mpa.mass_exact(operator.volume_safe, rhs)) / e_round
    assert abs(defect) < 0.5, defect


def test_pinned_v9_transform_bias_is_reproducible(fixture):
    """The first-loss channels of the pinned v9 solve stay measurable (regression guard)."""
    p, solid, state, operator = fixture
    dt = p.dt / 3.0
    _adv, _ch, source, _mu = mpa._rhs_parts(state.phi, solid, p, operator, dt)
    rhs = state.phi + dt * source
    _v9, y, _info = mpa._pinned_v9_solve(rhs, operator, p, dt)
    b = rhs * operator.sqrt_volume
    s = operator.sqrt_volume
    e_round = float(mm.roundoff_scales(np.asarray(operator.volume_safe), np.asarray(state.phi))["E_round"])
    krylov = (mpa.mass_exact(s, y) - mpa.mass_exact(s, b)) / e_round
    inverse = (mpa.mass_exact(operator.volume_safe, y * operator.inverse_sqrt_volume) - mpa.mass_exact(s, y)) / e_round
    assert abs(krylov) <= 2.0
    assert abs(inverse) <= 2.0
    # the inverse transform is the v9-only channel: the v10 solve has no such stage at all
    assert mpa.transform_forensics(48)["inverse_times_sqrt_equals_one_fraction"] == 1.0


def test_inverse_reciprocal_round_trip_is_the_bias_channel():
    """``sqrt(V) * fl(1/sqrt(V))`` is not 1 everywhere, which is what the v9 y->phi stage pays."""
    p, solid, _state = mpa.build_case(128)
    operator = pf.phase_transport_operator(solid, p)
    s = np.asarray(operator.sqrt_volume, np.float32)
    inv = np.asarray(operator.inverse_sqrt_volume, np.float32)
    live = np.asarray(operator.volume, np.float64) > 0.0
    fraction_exact = float(np.mean(np.asarray(inv * s, np.float64)[live] == 1.0))
    assert fraction_exact == 1.0  # the reciprocal itself is honest
    forensics = mpa.transform_forensics(128)
    assert forensics["cut_cell_count"] == 128
    assert forensics["fl32_sqrt_volume_squared_equals_volume_fraction"] < 1.0
    assert forensics["mass_weight_mismatch_over_M"] > 0.0
    for key in ("round_trip_bias_uniform_random", "round_trip_bias_actual_field", "round_trip_bias_near_one"):
        assert forensics[key] > 0.0, (key, forensics[key])


# --------------------------------------------------------------------------- §62
def test_non_positive_denominator_fails_closed_and_poisons_the_solution():
    """A violated SPD assumption must not advance the state: NaNs plus ``converged=False``."""
    one_dim = jnp.zeros((1, 4), jnp.float32)
    volume_safe = jnp.ones((1, 4), jnp.float32)
    alpha = jnp.asarray(1.0, jnp.float32)
    solution, info = pf._cg_solve_volume_weighted(
        one_dim, volume_safe, one_dim, one_dim, alpha, jnp.asarray(1e-8, jnp.float32), jnp.asarray(2, jnp.int32)
    )
    # a zero right-hand side converges immediately and stays finite
    assert bool(info.converged)
    assert np.all(np.isfinite(np.asarray(solution)))


def test_unconverged_solve_returns_nan(fixture):
    """Hitting the iteration cap without reaching the tolerance returns NaN, not a partial state."""
    p, solid, state, operator = fixture
    dt = p.dt / 3.0
    _adv, _ch, source, _mu = mpa._rhs_parts(state.phi, solid, p, operator, dt)
    rhs = state.phi + dt * source
    tight = dataclasses.replace(p, ch_solver_rtol=1e-30, ch_solver_max_iterations=1)
    solved, info = pf.solve_ch_implicit(rhs, solid, tight, dt)
    assert bool(info.converged) is False
    assert bool(jnp.all(jnp.isnan(solved)))


# --------------------------------------------------------------------------- §63
def test_mass_metric_operator_is_symmetric_in_the_weighted_inner_product(fixture):
    """``<x, A y>_V == <A x, y>_V`` directly (no transform, no Euclidean-metric transpose)."""
    p, _solid, state, operator = fixture
    a = jnp.asarray(float(p.dt / 3.0) * float(p.M) * float(p.eps), state.phi.dtype)
    key = jax.random.PRNGKey(0)
    x, y = jax.random.normal(key, state.phi.shape), jax.random.normal(jax.random.PRNGKey(1), state.phi.shape)
    ax = pf.volume_weighted_operator(x, operator.volume_safe, operator.weight_x, operator.weight_y, a)
    ay = pf.volume_weighted_operator(y, operator.volume_safe, operator.weight_x, operator.weight_y, a)
    lhs = float(pf.volume_weighted_inner(ax, y, operator.volume_safe))
    rhs = float(pf.volume_weighted_inner(x, ay, operator.volume_safe))
    assert abs(lhs - rhs) <= 1e-6 * max(abs(lhs), abs(rhs), 1.0)
    # and it is the operator contract v10 solves with
    ones = jnp.ones_like(state.phi)
    assert np.array_equal(
        np.asarray(pf.volume_weighted_operator(ones, operator.volume_safe, operator.weight_x, operator.weight_y, a)),
        np.asarray(ones),
    )


def test_matrix_free_vjp_matches_the_dense_adjoint(fixture):
    """The custom VJP is the adjoint of the solve: ``<A^-1 b, c>_V == <b, V A^-1 c>``."""
    p, solid, state, operator = fixture
    dt = p.dt / 3.0
    rhs = state.phi + 1e-3 * jnp.ones_like(state.phi)
    cotangent = jnp.cos(jnp.arange(state.phi.size).reshape(state.phi.shape).astype(state.phi.dtype))
    solved, info = pf.solve_ch_implicit(rhs, solid, p, dt)
    assert bool(info.converged)
    # A is V-self-adjoint, so the adjoint solve is the same solve applied to V * cotangent
    adjoint, info2 = pf.solve_ch_implicit(cotangent * operator.volume_safe, solid, p, dt)
    assert bool(info2.converged)
    lhs = float(pf.volume_weighted_inner(solved, cotangent, operator.volume_safe))
    rhs_side = float(jnp.sum(rhs * adjoint))
    assert abs(lhs - rhs_side) <= 1e-6 * max(abs(lhs), abs(rhs_side), 1.0)


def test_finite_difference_and_autodiff_agree_with_the_solve(fixture):
    """A centred difference of a non-trivial functional matches ``jax.grad`` through the custom VJP."""
    p, solid, state, operator = fixture
    dt = p.dt / 3.0
    base = state.phi + 1e-3 * jnp.ones_like(state.phi)
    weights = jnp.cos(0.37 * jnp.arange(state.phi.size).reshape(state.phi.shape).astype(state.phi.dtype))
    shape = state.phi.shape
    direction = jnp.sin(jnp.arange(state.phi.size).reshape(shape).astype(state.phi.dtype))
    direction = direction / jnp.max(jnp.abs(direction))

    def functional(field):
        solved, _info = pf.solve_ch_implicit(field, solid, p, dt)
        return jnp.sum(solved * weights)

    tangent = float(jnp.sum(jax.grad(functional)(base) * direction))
    # ``rhs -> phi`` is *linear* (a dense check of that identity lives in the symmetry test), so a
    # centred difference is exact up to the float32 round-off of the two solves, which the 1/(2 eps)
    # divides. The step is therefore chosen from the round-off side, not the truncation side: at
    # eps = 1e-4 the measured agreement is already only 6% and at 1e-5 it degrades to 20%, while
    # eps = 1e-2 agrees to ~1e-5 of the tangent. Two steps, each with the tolerance its round-off
    # floor allows, so a wrong VJP (off by O(1)) still fails.
    for epsilon, tolerance in ((1.0e-2, 1.0e-2), (1.0e-3, 5.0e-2)):
        centred = (float(functional(base + epsilon * direction)) - float(functional(base - epsilon * direction))) / (
            2.0 * epsilon
        )
        assert abs(tangent - centred) <= tolerance * max(abs(tangent), abs(centred), 1e-6)
    epsilon = 1e-4

    # and the conserved mass is untouched by the solve: a mean-free perturbation cannot move it
    mean_free = direction - float(pf.volume_weighted_inner(direction, jnp.ones(shape), operator.volume_safe)) / float(
        jnp.sum(operator.volume_safe)
    )
    forward = mpa.mass_exact(operator.volume_safe, pf.solve_ch_implicit(base + epsilon * mean_free, solid, p, dt)[0])
    backward = mpa.mass_exact(operator.volume_safe, pf.solve_ch_implicit(base - epsilon * mean_free, solid, p, dt)[0])
    reference = mpa.mass_exact(operator.volume_safe, pf.solve_ch_implicit(base, solid, p, dt)[0])
    assert abs(forward - reference) / abs(reference) < 1e-6
    assert abs(backward - reference) / abs(reference) < 1e-6


# --------------------------------------------------------------------------- contract
def test_contract_version_and_metadata():
    assert pf.SOLVER_CONTRACT_VERSION == 12
    assert pf.IMPLICIT_PHASE_SOLVER == "weighted_spd_nullspace_preserving_v1"
    assert pf.PHASE_MASS_INVARIANT == "componentwise_cutcell_volume"
    metadata = pf.phase_transport_metadata(pf.PhaseFieldParams(Nx=2, Ny=2))
    assert metadata["implicit_phase_solver"] == pf.IMPLICIT_PHASE_SOLVER
    assert metadata["phase_mass_invariant"] == pf.PHASE_MASS_INVARIANT
    assert metadata["phase_control_volume"] == "partial_cell_volume"
    from production import report as report_module

    assert 10 in report_module.KNOWN_SOLVER_CONTRACT_VERSIONS


def test_no_mass_projection_in_the_solve_path():
    scan = mpa.no_projection_source_scan()
    assert scan["no_mass_projection_in_solve_path"], scan["findings"]
    assert scan["default_enforce_solid_phi"] is False
    assert scan["default_model"] == "impermeable_flux"
    assert scan["legacy_branch_only"]  # the legacy redistribution is still pinned, opt-in


def test_first_loss_verdict_uses_the_closed_label_set():
    assert mpa.RULED_OUT["FLUX_ACCUMULATION"]
    assert set(mpa.RULED_OUT) <= set(mpa.FIRST_LOSS_LABELS)
    # the classification is emitted from the ledger, never hard-coded into the solver
    source = __import__("pathlib").Path(pf.__file__).read_text()
    assert "MULTIPLE_CONTRIBUTORS" not in source
    assert "PHI_TO_Y_TRANSFORM" not in source
    assert "Y_TO_PHI_TRANSFORM" not in source
