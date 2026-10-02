"""Conservative phase-boundary and natural-wetting audit (L1A-2c).

The audit is deliberately diagnostic: it does not tune the solver. It checks
face-level impermeability, telescoping fluid-mass conservation, the embedded
Young normal-condition sign for flat/inclined SDF walls, matrix-free implicit
residuals, solid leakage, pure-CH free-energy relaxation, and a projection-vs-v7
neutral sessile trace.

Run from ``examples/two_phase``::

    python -m production.phase_boundary_audit --json artifacts/production_validation/phase_boundary_audit.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import phasefield as pf

#: Contract-v8 variational gates (L1A-2e): the production wall operator must be the exact
#: variational derivative of the discrete bulk + wall energy in float64 (specification gate
#: 1e-6, ideal 1e-8). The worst single centred amplitude is bounded separately because
#: float64 cancellation inflates the smallest one.
VARIATIONAL_RELATIVE_TOLERANCE = 1.0e-6
VARIATIONAL_WORST_AMPLITUDE_TOLERANCE = 1.0e-4
#: The exact nonlinear natural-BC profile is differentiated with second-order central stencils,
#: so at eps/dx = 2 the residual is pure truncation (~9e-3 flat, ~7e-3 inclined) and falls to
#: ~2.5e-3 / ~1.9e-3 at eps/dx = 4. The bound is that truncation level, not a fitted tolerance.
NATURAL_BC_EXACT_PROFILE_TOLERANCE = 2.0e-2


@dataclass
class AuditCheck:
    name: str
    passed: bool
    expectation: str
    observed: dict[str, Any]


@dataclass
class PhaseBoundaryAudit:
    settings: dict[str, Any]
    checks: list[AuditCheck]
    numbers: dict[str, Any]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["audit"] = "phase_boundary"
        result["audit_schema_version"] = 1
        result["passed"] = self.passed
        return result


def _natural_bc_profile(sdf, p, theta_deg: float, profile: str):
    """Phase fields that satisfy ``eps dphi/dn + g_w'(phi) = 0`` in a known way.

    ``exact`` is the analytic solution of the *nonlinear* natural condition along the wall
    normal, ``phi = 0.5 (1 - tanh(cos(theta) sdf / (sqrt(2) eps)))``: substituting
    ``dphi/ds = -2 phi (1-phi) cos(theta)/(sqrt(2) eps)`` and
    ``g_w'(phi) = -sqrt(2) cos(theta) phi (1-phi)`` (sigma_0 = sqrt(2)/6, h' = 6 phi (1-phi))
    satisfies it identically at *every* distance from the wall, so the only error left is the
    finite-difference stencil.

    ``linear`` is the tangent linearization at ``phi = 0.5``, ``phi = 0.5 + (g_w'(0.5)/eps) sdf``.
    It satisfies the condition only at the wall plane itself: because ``g_w'`` is nonlinear, a
    control cell whose centre is ``d`` away from the wall sees a prescribed value that differs
    by the analytic factor ``6 phi (1-phi) / 1.5``. That factor is the O(d/eps) wall-placement
    error of any cell-centred embedded BC, and it is reported explicitly rather than hidden
    inside a loose tolerance.
    """
    cos_theta = math.cos(math.radians(theta_deg))
    if profile == "exact":
        return 0.5 * (1.0 - jnp.tanh(cos_theta * sdf / (math.sqrt(2.0) * p.eps)))
    if profile == "linear":
        q_sdf = float(pf.wall_energy_derivative(jnp.asarray(0.5, dtype=p.dtype), cos_theta)) / float(p.eps)
        return 0.5 + q_sdf * sdf
    raise ValueError(f"unknown natural-BC profile {profile!r}")


def _normal_bc_relative_error(
    sdf, p, theta_deg: float, *, region: np.ndarray | None = None, profile: str = "exact"
) -> float:
    """Relative error of the natural BC on the cells the production operator actually forces.

    The active set is the contract-v8 embedded wall measure (``A_wall,i > 0``): exactly the
    fluid-side control cells that receive the Robin flux, with the measure-weighted cell normal
    the operator uses. ``profile='exact'`` leaves only the finite-difference truncation error;
    ``profile='linear'`` is checked against the linearized wall value by
    :func:`_normal_bc_linearization_deviation`.
    """
    cos_theta = math.cos(math.radians(theta_deg))
    phi = _natural_bc_profile(sdf, p, theta_deg, profile)
    gx = (jnp.roll(phi, -1, axis=0) - jnp.roll(phi, 1, axis=0)) / (2.0 * p.dx)
    gy = pf._ddy_nonperiodic(phi, p.dy)
    solid = pf.make_solid(sdf, p, cos_theta=cos_theta)
    nx, ny = pf.wall_measure_normal(solid, p)
    measured = nx * gx + ny * gy
    prescribed = pf.natural_wall_normal_derivative(phi, solid, p)
    measure = np.asarray(pf.wall_measure_density(solid, p), dtype=np.float64)
    peak = float(measure[region].max()) if region is not None else float(measure.max())
    active = measure > 0.9 * peak
    if region is not None:
        active &= region
    measured_np = np.asarray(measured, dtype=np.float64)
    prescribed_np = np.asarray(prescribed, dtype=np.float64)
    active &= np.isfinite(measured_np) & np.isfinite(prescribed_np)
    if not active.any():
        return math.inf
    denominator = max(float(np.max(np.abs(prescribed_np[active]))), 1e-30)
    return float(np.max(np.abs(measured_np[active] - prescribed_np[active])) / denominator)


def _normal_bc_linearization_deviation(sdf, p, theta_deg: float, *, region: np.ndarray | None = None) -> dict:
    """Split the linear-probe error into stencil error and analytic wall-placement error.

    Returns the linear probe's error against the *wall-plane* value ``-g_w'(0.5)/eps`` (which a
    linear field satisfies exactly, so this is pure stencil/normal error) together with the
    analytic deviation of ``-g_w'(phi_k)/eps`` from that wall value at the control-cell centres.
    """
    cos_theta = math.cos(math.radians(theta_deg))
    phi = _natural_bc_profile(sdf, p, theta_deg, "linear")
    gx = (jnp.roll(phi, -1, axis=0) - jnp.roll(phi, 1, axis=0)) / (2.0 * p.dx)
    gy = pf._ddy_nonperiodic(phi, p.dy)
    solid = pf.make_solid(sdf, p, cos_theta=cos_theta)
    nx, ny = pf.wall_measure_normal(solid, p)
    measured = np.asarray(nx * gx + ny * gy, dtype=np.float64)
    wall_value = -float(pf.wall_energy_derivative(jnp.asarray(0.5, dtype=p.dtype), cos_theta)) / float(p.eps)
    prescribed = np.asarray(pf.natural_wall_normal_derivative(phi, solid, p), dtype=np.float64)
    measure = np.asarray(pf.wall_measure_density(solid, p), dtype=np.float64)
    peak = float(measure[region].max()) if region is not None else float(measure.max())
    active = measure > 0.9 * peak
    if region is not None:
        active &= region
    if not active.any():
        return {"stencil_relative_error": math.inf, "placement_relative_deviation": math.inf, "n_active": 0}
    scale = max(abs(wall_value), 1e-30)
    return {
        "stencil_relative_error": float(np.max(np.abs(measured[active] - wall_value)) / scale),
        "placement_relative_deviation": float(np.max(np.abs(prescribed[active] - wall_value)) / scale),
        "n_active": int(np.count_nonzero(active)),
        "max_abs_sdf_over_dx_on_active": float(np.max(np.abs(np.asarray(sdf, dtype=np.float64)[active])) / p.dx),
    }


def _solid_fraction(phi, solid) -> float:
    positive = jnp.maximum(phi, 0.0)
    return float(jnp.sum(positive * solid.chi_hard) / jnp.maximum(jnp.sum(positive), 1e-30))


def _case_params(N: int, *, boundary: str, wetting: str, projection: bool, dt: float, dtype):
    return pf.PhaseFieldParams(
        Nx=N,
        Ny=N,
        Lx=6.0,
        Ly=6.0,
        dt=dt,
        eps=2.0 * 6.0 / N,
        dtype=dtype,
        wetting_model=wetting,
        phase_boundary_model=boundary,
        enforce_solid_phi=projection,
        ch_solver_rtol=1e-8 if dtype == jnp.float64 else 1e-6,
        ch_solver_max_iterations=200,
    )


def _neutral_trace(N: int, boundary: str, steps: int, sample_every: int, dtype) -> dict[str, Any]:
    legacy = boundary == "projection_legacy"
    p = _case_params(
        N,
        boundary=boundary,
        wetting="surface_energy_volume_v6" if legacy else "surface_energy",
        projection=legacy,
        dt=4.0e-3,
        dtype=dtype,
    )
    solid = pf.make_solid(pf.surface_flat(p, wall_height=0.25), p, cos_theta=0.0)
    state = pf.sessile_initial_state(p, solid, R=1.1, wall_height=0.25)
    mass0 = float(pf.liquid_mass(state.phi, solid, p))
    total0 = float(jnp.sum(state.phi) * p.dx * p.dy)
    step_fn = jax.jit(pf.step_with_diagnostics, static_argnums=(2,))
    trace = {
        name: []
        for name in (
            "step",
            "time",
            "angle_deg",
            "max_speed",
            "fluid_mass",
            "total_mass",
            "solid_phase_fraction",
            "free_energy",
            "implicit_iterations",
            "implicit_relative_residual",
        )
    }
    solver_max_iterations = 0
    solver_max_residual = 0.0

    def sample(step: int, info=None):
        nonlocal solver_max_iterations, solver_max_residual
        if info is not None:
            solver_max_iterations = max(solver_max_iterations, int(jnp.max(info.implicit_iterations)))
            solver_max_residual = max(solver_max_residual, float(jnp.max(info.implicit_relative_residuals)))
        trace["step"].append(int(step))
        trace["time"].append(float(state.t))
        trace["angle_deg"].append(float(pf.measure_contact_angle(state.phi, solid, p)))
        trace["max_speed"].append(float(jnp.sqrt(jnp.max(state.u**2 + state.v**2))))
        trace["fluid_mass"].append(float(pf.liquid_mass(state.phi, solid, p)))
        trace["total_mass"].append(float(jnp.sum(state.phi) * p.dx * p.dy))
        trace["solid_phase_fraction"].append(_solid_fraction(state.phi, solid))
        trace["free_energy"].append(float(pf.phase_free_energy(state.phi, solid, p)))
        trace["implicit_iterations"].append(int(solver_max_iterations))
        trace["implicit_relative_residual"].append(float(solver_max_residual))

    sample(0)
    for index in range(steps):
        state, info = step_fn(state, solid, p)
        if (index + 1) % sample_every == 0 or index + 1 == steps:
            sample(index + 1, info)
    mass_final = float(pf.liquid_mass(state.phi, solid, p))
    total_final = float(jnp.sum(state.phi) * p.dx * p.dy)
    return {
        "phase_boundary_model": boundary,
        "wetting_model": p.wetting_model,
        "enforce_solid_phi": bool(p.enforce_solid_phi),
        "projection_path_expected": bool(p.phase_boundary_model == "projection_legacy" and p.enforce_solid_phi),
        "mass_relative_drift": abs(mass_final - mass0) / max(abs(mass0), 1e-30),
        "total_mass_relative_drift": abs(total_final - total0) / max(abs(total0), 1e-30),
        "final_max_speed": trace["max_speed"][-1],
        "max_solid_phase_fraction": max(trace["solid_phase_fraction"]),
        "implicit_iterations_max": int(solver_max_iterations),
        "implicit_relative_residual_max": float(solver_max_residual),
        "angle_trace_deg": trace["angle_deg"],
        "energy_trace": trace["free_energy"],
        "trace": trace,
    }


def run_phase_boundary_audit(
    N: int = 64,
    *,
    sessile_steps: int = 120,
    energy_steps: int = 40,
    ab_steps: int = 120,
    sample_every: int = 20,
    dtype=jnp.float64,
) -> PhaseBoundaryAudit:
    """Run face-flux, BC, solver, energy, leak, and neutral A/B checks."""
    if N < 16 or min(sessile_steps, energy_steps, ab_steps, sample_every) < 1:
        raise ValueError("N must be >=16 and all step/sample counts must be positive")
    p = _case_params(
        N,
        boundary="impermeable_flux",
        wetting="surface_energy",
        projection=False,
        dt=4.0e-3,
        dtype=dtype,
    )
    sdf = pf.surface_flat(p, wall_height=0.25)
    solid = pf.make_solid(sdf, p, cos_theta=0.0)
    fluid = np.asarray(solid.sdf >= 0.0)
    crossing_x = fluid ^ np.roll(fluid, -1, axis=0)
    crossing_y = fluid ^ np.roll(fluid, -1, axis=1)
    aperture_x, aperture_y = (np.asarray(value) for value in pf.fluid_face_apertures(solid, p))

    rng = np.random.default_rng(2026)
    phi = jnp.asarray(rng.uniform(0.0, 1.0, size=(N, N)), dtype=p.dtype)
    u = jnp.asarray(rng.normal(size=(N, N)), dtype=p.dtype)
    v = jnp.asarray(rng.normal(size=(N, N)), dtype=p.dtype)
    mu = (pf.grids(p)[1] - 0.25).astype(p.dtype)
    adv_x, adv_y = (np.asarray(value) for value in pf.phase_advective_fluxes(u, v, phi, solid, p))
    ch_x, ch_y = (np.asarray(value) for value in pf.chemical_potential_fluxes(mu, solid, p))
    div_adv = pf.divergence_from_face_fluxes(*pf.phase_advective_fluxes(u, v, phi, solid, p), p)
    div_ch = pf.divergence_from_face_fluxes(*pf.chemical_potential_fluxes(mu, solid, p), p)
    mass_sum = float(jnp.sum(jnp.where(solid.sdf >= 0.0, div_adv + div_ch, 0.0)))

    flat_bc_error = _normal_bc_relative_error(sdf, p, 60.0)
    flat_bc_linear = _normal_bc_linearization_deviation(sdf, p, 60.0)
    X, Y = pf.grids(p)
    slope = 0.45
    inclined_sdf = (Y - slope * (X - 3.0) - 0.25) / math.sqrt(1.0 + slope**2)
    x_region = (np.asarray(X) > 2.5) & (np.asarray(X) < 3.5)
    inclined_bc_error = _normal_bc_relative_error(inclined_sdf, p, 120.0, region=x_region)
    inclined_bc_linear = _normal_bc_linearization_deviation(inclined_sdf, p, 120.0, region=x_region)
    reversed_sdf = (Y + slope * (X - 3.0) - 0.25) / math.sqrt(1.0 + slope**2)
    reversed_bc_error = _normal_bc_relative_error(reversed_sdf, p, 120.0, region=x_region)
    reversed_bc_linear = _normal_bc_linearization_deviation(reversed_sdf, p, 120.0, region=x_region)

    rhs = jnp.where(solid.sdf >= 0.0, phi, 0.0)
    implicit_solution, implicit_info = pf.solve_ch_implicit(rhs, solid, p, p.dt / 3.0)
    implicit_alpha = (p.dt / 3.0) * p.M * p.eps
    implicit_lap = pf.fluid_laplacian(implicit_solution, solid, p)
    implicit_residual_field = implicit_solution + implicit_alpha * pf.fluid_laplacian(implicit_lap, solid, p) - rhs
    implicit_rel = float(jnp.linalg.norm(implicit_residual_field) / jnp.maximum(jnp.linalg.norm(rhs), 1e-30))

    # Full v7 discrete variational audit: differentiate F_bulk+F_wall along a
    # fluid-supported direction and compare with the production natural-BC mu.
    p_var = _case_params(
        N,
        boundary="impermeable_flux",
        wetting="surface_energy",
        projection=False,
        dt=4.0e-3,
        dtype=dtype,
    )
    solid_var = pf.make_solid(sdf, p_var, cos_theta=math.cos(math.radians(60.0)))
    X_var, Y_var = pf.grids(p_var)
    active_var = solid_var.sdf >= 0.0
    phi_var = jnp.where(
        active_var,
        0.5 + 0.15 * jnp.cos(2.0 * jnp.pi * X_var / p_var.Lx) * jnp.cos(2.0 * jnp.pi * Y_var / p_var.Ly),
        0.0,
    )
    direction_var = active_var * (
        0.2 + 0.1 * jnp.sin(2.0 * jnp.pi * X_var / p_var.Lx) * jnp.cos(jnp.pi * Y_var / p_var.Ly)
    )
    mu_var = pf.chemical_potential(phi_var, solid_var, p_var)
    predicted_directional = float(jnp.sum(mu_var * direction_var) * p_var.dx * p_var.dy)
    variational_rows = []
    for amplitude in (1.0e-6, 1.0e-5, 1.0e-4):
        fd = float(
            (
                pf.phase_free_energy(phi_var + amplitude * direction_var, solid_var, p_var)
                - pf.phase_free_energy(phi_var - amplitude * direction_var, solid_var, p_var)
            )
            / (2.0 * amplitude)
        )
        variational_rows.append(
            {
                "amplitude": float(amplitude),
                "finite_difference": fd,
                "mu_inner_product": predicted_directional,
                "relative_error": abs(fd - predicted_directional) / max(abs(predicted_directional), 1e-30),
            }
        )
    # Amplitude-optimized: float64 cancellation dominates the small amplitudes, FD truncation
    # the large ones. Every amplitude is reported.
    best = min(variational_rows, key=lambda row: row["relative_error"])
    fd_directional = best["finite_difference"]
    variational_relative_error = best["relative_error"]
    variational_worst_amplitude = max(row["relative_error"] for row in variational_rows)
    separate_wall_mu_max = float(jnp.max(jnp.abs(pf.wetting_mu(phi_var, solid_var, p_var))))
    wall_flux_total = float(np.sum(np.asarray(pf.wall_measure_density(solid_var, p_var), dtype=np.float64))) * (
        p_var.dx * p_var.dy
    )

    # Isolated u=0 neutral Cahn--Hilliard relaxation: test the actual discrete
    # bulk+wall energy without momentum exchanging kinetic and surface energy.
    p_energy = _case_params(
        N,
        boundary="impermeable_flux",
        wetting="surface_energy",
        projection=False,
        dt=2.0e-4,
        dtype=dtype,
    )
    solid_energy = pf.make_solid(pf.surface_flat(p_energy, wall_height=0.25), p_energy, cos_theta=0.0)
    phase = pf.sessile_initial_state(p_energy, solid_energy, R=1.1, wall_height=0.25).phi
    zero = jnp.zeros_like(phase)
    energy_trace = [float(pf.phase_free_energy(phase, solid_energy, p_energy))]
    phase_info = None
    for _ in range(energy_steps):
        for _stage in range(3):
            phase, phase_info = pf.phase_transport_step(phase, zero, zero, solid_energy, p_energy)
            if not bool(phase_info.converged):
                break
        energy_trace.append(float(pf.phase_free_energy(phase, solid_energy, p_energy)))
    energy_scale = max(1.0, abs(energy_trace[0]))
    positive_increases = [max(0.0, b - a) for a, b in zip(energy_trace, energy_trace[1:])]
    energy_increase_count = sum(increase > 1e-10 * energy_scale for increase in positive_increases)
    energy_max_rel_increase = max(positive_increases, default=0.0) / energy_scale
    energy_total_increase = sum(positive_increases) / energy_scale
    energy_ok = (
        energy_trace[-1] <= energy_trace[0] + 1e-6 * energy_scale
        and energy_total_increase <= 1e-6
        and energy_increase_count == 0
    )

    sessile = pf.sessile_initial_state(p, solid, R=1.1, wall_height=0.25)
    sessile_fn = jax.jit(pf.step_with_diagnostics, static_argnums=(2,))
    for _ in range(sessile_steps):
        sessile, _ = sessile_fn(sessile, solid, p)
    sessile_solid_fraction = _solid_fraction(sessile.phi, solid)
    sessile_mass0 = float(pf.liquid_mass(pf.sessile_initial_state(p, solid, R=1.1, wall_height=0.25).phi, solid, p))
    sessile_mass = float(pf.liquid_mass(sessile.phi, solid, p))
    sessile_mass_drift = abs(sessile_mass - sessile_mass0) / max(abs(sessile_mass0), 1e-30)

    ab = {
        "projection_v6": _neutral_trace(N, "projection_legacy", ab_steps, sample_every, dtype),
        "impermeable_v8": _neutral_trace(N, "impermeable_flux", ab_steps, sample_every, dtype),
    }
    checks = [
        AuditCheck(
            "advective_cross_wall_flux_zero",
            bool(
                np.max(np.abs(adv_x[crossing_x]), initial=0.0) == 0.0
                and np.max(np.abs(adv_y[crossing_y]), initial=0.0) == 0.0
            ),
            "all fluid-solid advective face fluxes are exactly zero",
            {
                "crossing_x_max_abs": float(np.max(np.abs(adv_x[crossing_x]), initial=0.0)),
                "crossing_y_max_abs": float(np.max(np.abs(adv_y[crossing_y]), initial=0.0)),
            },
        ),
        AuditCheck(
            "ch_cross_wall_flux_zero",
            bool(
                np.max(np.abs(ch_x[crossing_x]), initial=0.0) == 0.0
                and np.max(np.abs(ch_y[crossing_y]), initial=0.0) == 0.0
            ),
            "all fluid-solid Cahn-Hilliard fluxes are exactly zero",
            {
                "crossing_x_max_abs": float(np.max(np.abs(ch_x[crossing_x]), initial=0.0)),
                "crossing_y_max_abs": float(np.max(np.abs(ch_y[crossing_y]), initial=0.0)),
            },
        ),
        AuditCheck(
            "fluid_mass_flux_telescoping",
            abs(mass_sum) <= (1e-10 if dtype == jnp.float64 else 1e-5),
            "sum of advective+CH phase RHS over hard fluid cells is zero to round-off",
            {"sum_rhs_fluid": mass_sum},
        ),
        AuditCheck(
            "periodic_y_seam_is_blocked",
            bool(not aperture_y[:, -1].any() and fluid[:, -1].all() and not fluid[:, 0].any() and aperture_x.any()),
            "bottom wall blocks top-to-bottom periodic y transport; x fluid faces remain periodic",
            {"y_seam_open_faces": int(aperture_y[:, -1].sum()), "x_open_faces": int(aperture_x.sum())},
        ),
        AuditCheck(
            "flat_wall_natural_bc_sign",
            flat_bc_linear["stencil_relative_error"] <= 1.0e-9 and flat_bc_error <= NATURAL_BC_EXACT_PROFILE_TOLERANCE,
            "on the measure-carrying control cells the linearized natural-BC profile reproduces "
            "dphi/dn = -g_w'(0.5)/eps to round-off (sign and normal orientation), and the *exact* nonlinear "
            f"natural-BC profile tanh(cos(theta) sdf/(sqrt(2) eps)) reproduces it to <= "
            f"{NATURAL_BC_EXACT_PROFILE_TOLERANCE:g} (second-order stencil truncation at eps/dx = 2; it drops "
            "~3.6x at eps/dx = 4)",
            {
                "linear_stencil_relative_error": flat_bc_linear["stencil_relative_error"],
                "exact_profile_relative_error": flat_bc_error,
                "linear_wall_placement_deviation": flat_bc_linear["placement_relative_deviation"],
                "max_abs_sdf_over_dx_on_active": flat_bc_linear["max_abs_sdf_over_dx_on_active"],
                "target_deg": 60.0,
            },
        ),
        AuditCheck(
            "inclined_wall_natural_bc_orientation",
            max(inclined_bc_linear["stencil_relative_error"], reversed_bc_linear["stencil_relative_error"]) <= 1.0e-9
            and max(inclined_bc_error, reversed_bc_error) <= NATURAL_BC_EXACT_PROFILE_TOLERANCE,
            "the SDF normal BC has the same sign and magnitude for positive and negative wall inclination "
            "(linearized stencil error at round-off for both slopes, exact profile within truncation)",
            {
                "positive_slope_exact_error": inclined_bc_error,
                "negative_slope_exact_error": reversed_bc_error,
                "positive_slope_stencil_error": inclined_bc_linear["stencil_relative_error"],
                "negative_slope_stencil_error": reversed_bc_linear["stencil_relative_error"],
                "positive_slope_placement_deviation": inclined_bc_linear["placement_relative_deviation"],
                "negative_slope_placement_deviation": reversed_bc_linear["placement_relative_deviation"],
            },
        ),
        AuditCheck(
            "neutral_90_wall_has_zero_bc_source",
            float(jnp.max(jnp.abs(pf.natural_wall_normal_derivative(sessile.phi, solid, p)))) <= 1e-12,
            "cos(90 deg)=0 gives exactly neutral wall-normal derivative to round-off",
            {
                "max_abs_normal_derivative": float(
                    jnp.max(jnp.abs(pf.natural_wall_normal_derivative(sessile.phi, solid, p)))
                )
            },
        ),
        AuditCheck(
            "matrix_free_implicit_solver_residual",
            bool(implicit_info.converged) and implicit_rel <= p.ch_solver_rtol,
            f"CG residual <= {p.ch_solver_rtol:g}; nonconvergence poisons the solution with NaN",
            {
                "iterations": int(implicit_info.iterations),
                "reported_relative_residual": float(implicit_info.relative_residual),
                "recomputed_relative_residual": implicit_rel,
                "converged": bool(implicit_info.converged),
            },
        ),
        AuditCheck(
            "v8_energy_variational_derivative_no_double_count",
            variational_relative_error <= VARIATIONAL_RELATIVE_TOLERANCE
            and variational_worst_amplitude <= VARIATIONAL_WORST_AMPLITUDE_TOLERANCE
            and separate_wall_mu_max == 0.0,
            f"centred FD derivative of F_bulk + F_wall^h equals the production mu inner product within "
            f"{VARIATIONAL_RELATIVE_TOLERANCE:g} (float64, amplitude-optimized; worst single amplitude "
            f"{variational_worst_amplitude:.3e}), and the wall energy is not also added via wetting_mu",
            {
                "finite_difference": fd_directional,
                "mu_inner_product": predicted_directional,
                "relative_error": variational_relative_error,
                "worst_amplitude_relative_error": variational_worst_amplitude,
                "amplitudes": variational_rows,
                "separate_wetting_mu_max_abs": separate_wall_mu_max,
                "wall_measure_total_length": wall_flux_total,
                "expected_wall_length": float(p_var.Lx),
            },
        ),
        AuditCheck(
            "phase_free_energy_nonincreasing_u_zero",
            energy_ok,
            "neutral u=0 CH relaxation has no systematic F_bulk+F_wall increase",
            {
                "initial": energy_trace[0],
                "final": energy_trace[-1],
                "increase_samples": int(energy_increase_count),
                "max_relative_increase": energy_max_rel_increase,
                "cumulative_relative_increase": energy_total_increase,
                "trace": energy_trace,
            },
        ),
        AuditCheck(
            "solid_phase_leak_projection_free",
            sessile_solid_fraction <= 1e-6 and sessile_mass_drift <= 1e-3,
            "v7 neutral sessile run does not use projection, solid phase fraction <=1e-6 and fluid-mass drift <=1e-3",
            {
                "solid_phase_fraction": sessile_solid_fraction,
                "fluid_mass_relative_drift": sessile_mass_drift,
                "steps": int(sessile_steps),
            },
        ),
    ]
    return PhaseBoundaryAudit(
        settings={
            "N": int(N),
            "dtype": str(np.dtype(dtype)),
            "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
            "phase_boundary_model": "impermeable_flux",
            "wetting_model": "surface_energy",
            "wall_measure_method": str(p.wall_measure),
            "wall_measure_contract_version": int(pf.WALL_MEASURE_CONTRACT_VERSION),
            "ch_solver_rtol": float(p.ch_solver_rtol),
            "ch_solver_max_iterations": int(p.ch_solver_max_iterations),
            "sessile_steps": int(sessile_steps),
            "energy_steps": int(energy_steps),
            "ablation_steps": int(ab_steps),
            "sample_every": int(sample_every),
            "projection_legacy_is_diagnostic_only": True,
        },
        checks=checks,
        numbers={
            "wall_bc_relative_errors": {
                "flat_exact_profile": flat_bc_error,
                "flat_linear_stencil": flat_bc_linear["stencil_relative_error"],
                "flat_linear_placement_deviation": flat_bc_linear["placement_relative_deviation"],
                "inclined_positive_slope": inclined_bc_error,
                "inclined_negative_slope": reversed_bc_error,
            },
            "implicit_solver": {
                "iterations": int(implicit_info.iterations),
                "relative_residual": implicit_rel,
                "converged": bool(implicit_info.converged),
            },
            "v8_variational_audit": {
                "finite_difference": fd_directional,
                "mu_inner_product": predicted_directional,
                "relative_error": variational_relative_error,
                "worst_amplitude_relative_error": variational_worst_amplitude,
                "amplitudes": variational_rows,
                "separate_wetting_mu_max_abs": separate_wall_mu_max,
                "wall_measure_total_length": wall_flux_total,
            },
            "phase_energy": {
                "trace": energy_trace,
                "increase_samples": int(energy_increase_count),
                "max_relative_increase": energy_max_rel_increase,
                "cumulative_relative_increase": energy_total_increase,
            },
            "neutral_90_projection_ab": ab,
        },
    )


def format_markdown(audit: PhaseBoundaryAudit) -> str:
    lines = ["# Conservative phase-boundary audit (L1A-2c; wall measure updated by L1A-2e)", ""]
    lines.append(
        f"- contract={audit.settings['solver_contract_version']} N={audit.settings['N']} "
        f"boundary={audit.settings['phase_boundary_model']} wetting={audit.settings['wetting_model']}"
    )
    lines += ["", "| check | result | expectation | observed |", "| --- | --- | --- | --- |"]
    for check in audit.checks:
        values = ", ".join(f"{key}={value}" for key, value in check.observed.items() if key != "trace")
        lines.append(f"| {check.name} | {'PASS' if check.passed else 'FAIL'} | {check.expectation} | {values} |")
    ab = audit.numbers["neutral_90_projection_ab"]
    lines += ["", "## Neutral 90-degree projection-removal A/B", ""]
    lines += [
        "| path | angle trace (deg) | max speed final | fluid mass drift | solid fraction | energy trace |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for name, row in ab.items():
        lines.append(
            f"| {name} | {row['angle_trace_deg']} | {row['final_max_speed']:.3e} | "
            f"{row['mass_relative_drift']:.3e} | {row['max_solid_phase_fraction']:.3e} | {row['energy_trace']} |"
        )
    lines += ["", f"**Audit: {'PASS' if audit.passed else 'FAIL'}**", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="L1A-2c conservative phase-boundary audit")
    parser.add_argument("--json", default=None, help="write strict JSON audit results")
    parser.add_argument("--markdown", default=None, help="write Markdown audit results")
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--sessile-steps", type=int, default=120)
    parser.add_argument("--energy-steps", type=int, default=40)
    parser.add_argument("--ab-steps", type=int, default=120)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--quick", action="store_true", help="small CI audit while exercising the v7 path")
    args = parser.parse_args(argv)
    if args.quick:
        args.N = min(args.N, 32)
        args.sessile_steps = min(args.sessile_steps, 12)
        args.energy_steps = min(args.energy_steps, 8)
        args.ab_steps = min(args.ab_steps, 12)
        args.sample_every = min(args.sample_every, 4)
    jax.config.update("jax_enable_x64", True)
    audit = run_phase_boundary_audit(
        N=args.N,
        sessile_steps=args.sessile_steps,
        energy_steps=args.energy_steps,
        ab_steps=args.ab_steps,
        sample_every=args.sample_every,
        dtype=jnp.float64,
    )
    payload = audit.to_dict()
    report = format_markdown(audit)
    print(report)
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.markdown:
        path = Path(args.markdown)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
    return 0 if audit.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
