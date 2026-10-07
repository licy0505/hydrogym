"""Focused L1A-2k tests: production-operator decomposition only, no long trajectories."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import phasefield as pf  # noqa: E402
from production import capillary_pressure_balance_audit as audit  # noqa: E402
from production import chns_nonstationarity_audit as chns  # noqa: E402


def _tiny_case():
    return chns._make_case(60.0, N_value=24, dt=chns.DT, M=chns.M_REF)


def test_manufactured_gradient_is_recovered_and_diagnostic_curl_is_small():
    p, _solid, _state, _config = _tiny_case()
    nx, ny = p.Nx, p.Ny
    x = (jnp.arange(nx, dtype=jnp.float64) + 0.5)[:, None] * p.dx
    y = (jnp.arange(ny, dtype=jnp.float64) + 0.5)[None, :] * p.dy
    q = (jnp.sin(2.0 * jnp.pi * 2.0 * x / p.Lx) * jnp.cos(2.0 * jnp.pi * y / p.Ly)).astype(p.dtype)
    ax = pf._ddx(q, p.dx)
    ay = pf._ddy(q, p.dy)

    decomposition = audit._production_operator_arrays(ax, ay, p)
    manufactured = audit._manufactured_operator_tests(p)

    assert decomposition["metrics"]["passed"]
    assert decomposition["metrics"]["residual_fraction_of_source_l2"] < 5.0e-5
    assert decomposition["metrics"]["residual_divergence_l2"] <= decomposition["metrics"]["solve_tolerance_l2"]
    assert manufactured["manufactured_gradient_curl_passed"]
    assert manufactured["adjoint_passed"]
    assert manufactured["curl_operator"].startswith("diagnostic_curl_v1")


def test_random_vector_decomposition_fails_closed_checks_and_reconstructs():
    p, _solid, _state, _config = _tiny_case()
    rng = np.random.default_rng(142)
    ax = rng.normal(size=(p.Nx, p.Ny)).astype(np.float32)
    ay = rng.normal(size=(p.Nx, p.Ny)).astype(np.float32)

    decomposition = audit._production_operator_arrays(ax, ay, p)
    metrics = decomposition["metrics"]

    assert metrics["passed"]
    assert all(metrics["checks"].values())
    assert metrics["residual_divergence_l2"] <= metrics["solve_tolerance_l2"]
    assert metrics["poisson_solve_residual_l2"] <= metrics["solve_tolerance_l2"]
    assert metrics["reconstruction_linf"] <= metrics["reconstruction_tolerance_linf"]
    assert np.isfinite(decomposition["potential"]).all()


def test_force_density_acceleration_and_production_rhs_reconstruction():
    p, solid, state, _config = _tiny_case()
    components = audit._capillary_components(state, solid, p)
    force = audit._force_decomposition(components, p)

    assert components["production_rhs_reconstruction_exact"]
    assert force["production_work_dtype_acceleration"]["passed"]
    scaling = force["force_density_vs_acceleration"]
    assert scaling["rho_l_constant"] == p.rho_l
    assert scaling["residual_fraction_difference_force_vs_acceleration"] < 1.0e-6
    assert np.isfinite(
        force["discrete_form_identity"]["production_residual_equals_alternative_minus_defect_residual_l2_error"]
    )


def test_discrete_product_identity_reconstructs_and_equal_density_ab_is_isolated():
    p, solid, state, _config = _tiny_case()
    components = audit._capillary_components(state, solid, p)
    force = audit._force_decomposition(components, p)
    ab = audit._equal_density_one_step_ab(state, solid, p)

    assert components["identity_reconstruction_linf"] < 1.0e-10
    assert force["discrete_form_identity"]["alternative_form_diagnostic_only"]
    assert force["discrete_form_identity"]["not_a_force_ablation_or_production_validation"]
    assert ab["diagnostic_only"]
    assert ab["pressure_symbol_max_abs_difference"] == 0.0
    assert ab["capillary_acceleration_max_abs_difference"] == 0.0
    assert ab["state_differences_one_production_step"]["phi"]["bitwise_equal"]
    assert ab["state_differences_one_production_step"]["u"]["bitwise_equal"]
    assert ab["state_differences_one_production_step"]["v"]["bitwise_equal"]


def test_masks_include_fixed_contact_exclusion_radii_seam_and_wall_bins():
    p, solid, state, _config = _tiny_case()
    positions = {
        "contact_line_exists": True,
        "left_contact_x_wrapped": 0.75,
        "right_contact_x_wrapped": 2.25,
        "wall_y": 0.25,
    }
    masks = audit._region_masks(state.phi, solid, p, positions)

    required = {
        "interface_fluid_phi_005_095",
        "near_wall_fluid_0_2dx",
        "near_wall_fluid_0_4dx",
        "wall_distance_fluid_0_1dx",
        "wall_distance_solid_0_1dx",
        "y_periodic_seam_2cells",
        "y_periodic_seam_4cells",
        "contact_line_left_within_2dx",
        "contact_line_left_within_4dx",
        "contact_line_right_within_2dx",
        "contact_line_right_within_4dx",
        "outside_contact_line_union_2dx",
        "outside_contact_line_union_4dx",
        "liquid_phi_ge_095_fluid",
        "gas_phi_le_005_fluid",
        "brinkman_chi_gt_050",
    }
    assert required <= set(masks)
    assert np.all(masks["contact_line_left_within_2dx"] <= masks["contact_line_left_within_4dx"])
    assert np.all(masks["contact_line_union_within_2dx"] <= masks["contact_line_union_within_4dx"])
    assert np.array_equal(masks["outside_contact_line_union_4dx"], ~masks["contact_line_union_within_4dx"])


def test_projected_one_step_impulse_and_frozen_phi_branch_are_diagnostic():
    p, solid, state, _config = _tiny_case()
    impulse = audit._projected_one_step_impulse(state, solid, p)
    frozen = audit._frozen_phi_forcing_ab(state, solid, p, steps=2)

    assert impulse["diagnostic_only"]
    assert impulse["phi_bitwise_fixed"]
    assert impulse["capillary_on_final_velocity_l2"] >= impulse["capillary_off_final_velocity_l2"]
    assert frozen["capillary_on"]["phi_bitwise_fixed"]
    assert frozen["capillary_off"]["phi_bitwise_fixed"]
    assert frozen["matched_difference"]["only_capillary_scale_changed"]
    assert frozen["matched_difference"]["production_acceptance_evidence"] is False
    assert len(frozen["capillary_on"]["step_metric_final"]) > 0


def test_nan_source_is_rejected_before_pressure_solve():
    p, _solid, _state, _config = _tiny_case()
    ax = np.zeros((p.Nx, p.Ny), dtype=np.float32)
    ay = np.zeros_like(ax)
    ax[0, 0] = np.nan

    try:
        audit._production_operator_arrays(ax, ay, p)
    except audit.AuditValidationError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("non-finite production source was silently accepted")
