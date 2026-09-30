"""Analytic capillary / pressure sign audit for the JAX two-phase solver (L1A-2a).

Run from ``examples/two_phase``::

    python -m production.capillary_audit [--N 128] [--R 0.8] [--json out.json]

The audit performs **no time stepping and no benchmark post-processing**.  It
evaluates the solver's own operators (``chemical_potential``, ``rhs``,
``pressure_field``, ``poisson_solve`` ...) on analytic interfaces and compares
them with sign conventions that are *derived* from the free energy, not fitted
to the Laplace benchmark.  Exit status is 0 only if every check is consistent.

Conventions (derivation)
------------------------
* **Phase orientation.**  ``phi = 1`` liquid, ``phi = 0`` gas, with the drop profile
  ``phi(r) = 0.5 (1 - tanh((r - R) / (sqrt(2) eps)))``.
* **Interface normal.**  ``dphi/dr < 0`` across the interface, so ``grad(phi)``
  points gas -> liquid (outside -> inside).  The outward normal (liquid -> gas) is
  ``n_out = -grad(phi) / |grad(phi)| = +r_hat`` for a drop.
* **Chemical potential.**  ``F[phi] = int f(phi)/eps + eps/2 |grad phi|^2 dV`` with
  ``f = phi^2 (1 - phi)^2`` and ``mu = dF/dphi = f'(phi)/eps - eps lap(phi)``.
  On the tanh profile the planar part ``f'/eps - eps phi_rr`` cancels and only the curvature term
  ``-eps (dphi/dr) / r = +eps |dphi/dr| / r > 0`` survives at the interface of a convex liquid drop
  (Gibbs-Thomson; equilibrium ``mu = sigma_e / R``, ``sigma_e = sqrt(2) / 6``).
* **Korteweg force.**  Cahn-Hilliard is the gradient flow ``d(phi)/dt = -u.grad(phi) +
  M lap(mu)``, so advection changes the free energy at the rate
  ``dF/dt = int mu d(phi)/dt = - int mu u.grad(phi) dV``.  The kinetic energy changes by
  ``int u.F dV``.  Total-energy consistency (no spurious source of energy) requires
  ``F_cap = + mu grad(phi)``; the equivalent form is ``-phi grad(mu)`` because
  ``mu grad(phi) + phi grad(mu) = grad(mu phi)`` is a pure gradient that the pressure
  absorbs.  The pairs ``{+mu grad(phi), -phi grad(mu)}`` (consistent) and
  ``{-mu grad(phi), +phi grad(mu)}`` (anti-capillary, i.e. negative surface tension)
  are the two sign classes; the choice between them is fixed by energetics.
  With ``mu > 0`` and ``dphi/dr < 0`` the consistent force is ``F . n_out < 0``:
  it points toward the liquid, i.e. toward the centre of curvature.
* **Pressure gradient.**  ``du/dt = ... - grad(P) + F``; the projection is
  ``u+ = u* - dt grad(P)``, ``lap(P) = div(u*) / dt``.  A static drop therefore has
  ``grad(P) = F`` (irrotational part) and ``P_liquid - P_gas = - int_in^out F_r dr``.
* **Laplace jump.**  ``delta_p = P_liquid - P_gas = + sigma / R = + 1 / (We R)`` for a
  2-D circle, i.e. ``laplace_ratio = delta_p R We -> +1``.
* **Why Form A (``+mu grad(phi)``) rather than Form B (``-phi grad(mu)``).**  Both sit in the
  consistent class, but for Form B the projection pressure is the *reduced* pressure
  ``P_B = P_A - (SIGMA_NORM/We) mu phi``.  Its jump is the Laplace value minus the bulk
  ``mu`` and vanishes for a fully relaxed drop (``mu`` uniform), so it does not measure
  surface tension.  With Form A the jump is ``(SIGMA_NORM/We) int mu dphi``, which is
  ``1 / (We R)`` for both the tanh initial profile and the relaxed profile.  The choice
  is *not* made from the benchmark sign.  (For a variable density ``rho(phi)`` the two
  forms stop being equivalent; that is revisited in L1A-2c, not here.)

The checks isolate the three conventions that must agree: the force convention,
the projection convention and the diagnostic-pressure convention.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import phasefield as pf

CONVENTIONS: dict[str, str] = {
    "phase_orientation": "phi = 1 liquid, phi = 0 gas; phi(r) = 0.5 (1 - tanh((r - R) / (sqrt(2) eps)))",
    "grad_phi_direction": "dphi/dr < 0 across the interface: grad(phi) points gas -> liquid (outside -> inside)",
    "outward_normal": "n_out (liquid -> gas) = -grad(phi) / |grad(phi)| = +r_hat for a drop",
    "chemical_potential": (
        "mu = dF/dphi = f'(phi)/eps - eps lap(phi); mu = +eps |dphi/dr| / r > 0 at a convex liquid interface"
    ),
    "korteweg_force": (
        "F_cap = +(SIGMA_NORM/We) mu grad(phi) / rho_l  ~  -(SIGMA_NORM/We) phi grad(mu) / rho_l "
        "(up to a pure gradient); F . n_out < 0 (toward the liquid / centre of curvature)"
    ),
    "pressure_gradient": (
        "du/dt = ... - grad(P) + F; u+ = u* - dt grad(P), lap(P) = div(u*)/dt; static balance grad(P) = F"
    ),
    "laplace_jump": "delta_p = P_liquid - P_gas = +sigma/R = +1/(We R); laplace_ratio = delta_p R We -> +1 (2-D)",
}

# Tolerances.  They gate *sign and internal consistency*; none is tuned to a benchmark value.
_SCALE_BAND = (0.85, 1.15)  # laplace ratio sanity band at the audit resolution
_AGREEMENT_TOL = 0.05  # force-integral vs projection jump, relative to max(|ratio|, 1)
_COSINE_MIN = 0.9  # solenoidal parts of equivalent forms; equivalence is O((dx/eps)^2)
_IDENTITY_TOL = 1.0e-3  # float32 FFT identities (a sign error gives ~1 or ~2)
# Phase identity of the probes ("inside" is liquid, "outside" is gas).  Only the labelling is checked,
# so the tolerance allows a diffuse interface on coarse grids (eps/R ~ 0.25 gives phi ~ 0.97 at the core).
_LIQUID_MIN = 0.9
_GAS_MAX = 0.1


@dataclass
class AuditCheck:
    name: str
    passed: bool
    expectation: str
    observed: dict[str, float]
    note: str = ""


@dataclass
class AuditResult:
    conventions: dict[str, str]
    settings: dict[str, Any]
    checks: list[AuditCheck] = field(default_factory=list)
    diagnosis: dict[str, Any] = field(default_factory=dict)

    @property
    def all_passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["all_passed"] = self.all_passed
        return out


# ---------------------------------------------------------------------------------------
#  helpers (solver operators are used as-is; reductions are done in float64)
# ---------------------------------------------------------------------------------------


def _np(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def _f(x) -> float:
    return float(x)


def _cos(ax, ay, bx, by) -> float:
    dot = float(np.sum(ax * bx + ay * by))
    norm = math.sqrt(float(np.sum(ax * ax + ay * ay))) * math.sqrt(float(np.sum(bx * bx + by * by)))
    return dot / max(norm, 1.0e-300)


def _make_params(N: int, We: float, eps_factor: float) -> pf.PhaseFieldParams:
    p = pf.PhaseFieldParams(Nx=N, Ny=N, Lx=6.0, Ly=6.0, Re=200.0, We=We, dt=2.0e-3)
    p.eps = float(eps_factor) * p.dx
    return p


def _geometry(p: pf.PhaseFieldParams, R: float) -> dict[str, np.ndarray]:
    X, Y = pf.grids(p)
    rx, ry = _np(X) - p.Lx / 2.0, _np(Y) - p.Ly / 2.0
    r = np.hypot(rx, ry)
    rs = np.maximum(r, 1.0e-12)
    return {
        "r": r,
        "ur_x": rx / rs,
        "ur_y": ry / rs,
        "inside": r < 0.3 * R,
        "outside": r > 2.5 * R,
        "band": np.abs(r - R) < 5.0 * p.eps,
    }


def _rest_state(phi) -> pf.State:
    phi = jnp.asarray(phi)
    return pf.State(phi=phi, u=jnp.zeros_like(phi), v=jnp.zeros_like(phi), t=0.0)


def _elliptical_state(p: pf.PhaseFieldParams, R: float, aspect: float = 1.3) -> pf.State:
    """Non-equilibrium elliptical drop at rest: curvature varies along the interface."""
    X, Y = pf.grids(p)
    rx, ry = X - p.Lx / 2.0, Y - p.Ly / 2.0
    rho = jnp.sqrt(aspect * rx**2 + ry**2 / aspect)
    phi = 0.5 * (1.0 - jnp.tanh((rho - R) / (jnp.sqrt(2.0) * p.eps)))
    return _rest_state(phi.astype(p.dtype))


def _solenoidal(fx, fy, p: pf.PhaseFieldParams):
    """Split ``F`` with the solver's own projection (unit dt): return (F - grad P, P)."""
    fx, fy = jnp.asarray(fx, p.dtype), jnp.asarray(fy, p.dtype)
    div = pf._ddx(fx, p.dx) + pf._ddy(fy, p.dy)
    pressure = pf.poisson_solve(div, p.m2_proj)
    return fx - pf._ddx(pressure, p.dx), fy - pf._ddy(pressure, p.dy), pressure


def _laplace_ratio_from_pressure(pressure, geo: dict[str, np.ndarray], p: pf.PhaseFieldParams) -> float:
    pr = _np(pressure)
    return (float(pr[geo["inside"]].mean()) - float(pr[geo["outside"]].mean())) * p.We


# ---------------------------------------------------------------------------------------
#  checks
# ---------------------------------------------------------------------------------------


def _check_phi_orientation(p, solid, state, geo, R) -> AuditCheck:
    phi = _np(state.phi)
    px, py = _np(pf._ddx(state.phi, p.dx)), _np(pf._ddy(state.phi, p.dy))
    band = geo["band"]
    dphi_dr = px * geo["ur_x"] + py * geo["ur_y"]
    gnorm = np.maximum(np.hypot(px, py), 1.0e-30)
    n_out_dot_r = (-px * geo["ur_x"] - py * geo["ur_y"]) / gnorm
    observed = {
        "phi_inside_min": _f(phi[geo["inside"]].min()),
        "phi_outside_max": _f(phi[geo["outside"]].max()),
        "dphi_dr_band_max": _f(dphi_dr[band].max()),
        "n_out_dot_r_hat_band_min": _f(n_out_dot_r[band].min()),
    }
    passed = (
        observed["phi_inside_min"] > _LIQUID_MIN
        and observed["phi_outside_max"] < _GAS_MAX
        and observed["dphi_dr_band_max"] < 0.0
        and observed["n_out_dot_r_hat_band_min"] > 0.9
    )
    return AuditCheck(
        "phi_orientation_and_normal",
        bool(passed),
        "phi=1 inside (liquid), phi=0 outside (gas); dphi/dr<0; n_out=-grad(phi)/|grad(phi)| = +r_hat",
        observed,
    )


def _discrete_free_energy(f: np.ndarray, p: pf.PhaseFieldParams) -> float:
    """F_h[phi] = sum f(phi)/eps + eps/2 |D+ phi|^2; its gradient is f'/eps - eps * (5-point laplacian)."""
    gx, gy = (np.roll(f, -1, 0) - f) / p.dx, (np.roll(f, -1, 1) - f) / p.dy
    return float(np.sum(f**2 * (1.0 - f) ** 2 / p.eps + 0.5 * p.eps * (gx**2 + gy**2)) * p.dx * p.dy)


def _check_chemical_potential(p, solid, state, geo, R) -> AuditCheck:
    mu = _np(pf.chemical_potential(state.phi, solid, p))
    phi = _np(state.phi)
    # (a) mu is +dF/dphi (directional derivative of the discrete energy along random psi).
    psi = np.random.default_rng(0).standard_normal(phi.shape)
    tau = 1.0e-3
    directional = (_discrete_free_energy(phi + tau * psi, p) - _discrete_free_energy(phi - tau * psi, p)) / (2.0 * tau)
    predicted = float(np.sum(mu * psi) * p.dx * p.dy)
    scale = float(np.sum(np.abs(mu * psi)) * p.dx * p.dy)
    variational_error = abs(directional - predicted) / max(scale, 1.0e-30)
    # (b) sign and size of mu at a convex liquid interface (Gibbs-Thomson).
    band, r = geo["band"], geo["r"]
    phi_r = _np(pf._ddx(state.phi, p.dx)) * geo["ur_x"] + _np(pf._ddy(state.phi, p.dy)) * geo["ur_y"]
    gibbs_thomson = float(np.sum((mu * (-phi_r))[band] / (2.0 * np.pi * r[band])) * p.dx * p.dy)
    sigma_e = math.sqrt(2.0) / 6.0
    observed = {
        "variational_relative_error": variational_error,
        "mu_max": _f(mu.max()),
        "mu_max_analytic_sqrt2_over_4R": math.sqrt(2.0) / (4.0 * R),
        "int_mu_dphi_times_R_over_sigma_e": gibbs_thomson * R / sigma_e,
    }
    passed = (
        variational_error < 1.0e-4
        and observed["mu_max"] > 0.0
        and 0.85 < observed["int_mu_dphi_times_R_over_sigma_e"] < 1.15
    )
    return AuditCheck(
        "chemical_potential_sign",
        bool(passed),
        "mu = +dF/dphi (directional derivative); mu>0 at a convex liquid interface; int mu dphi = sigma_e/R",
        observed,
    )


def _check_force_direction(p, solid, state, geo, R) -> tuple[AuditCheck, float]:
    _, u_rhs, v_rhs, _, _ = pf.rhs(state, solid, p)  # state at rest: rhs == capillary acceleration
    fr = _np(u_rhs) * geo["ur_x"] + _np(v_rhs) * geo["ur_y"]
    band, r = geo["band"], geo["r"]
    delta_p_force = -float(np.sum(fr[band] / (2.0 * np.pi * r[band])) * p.dx * p.dy)
    ratio_force = delta_p_force * R * p.We
    observed = {
        "net_radial_force_integral_F_dot_n_out": -delta_p_force,
        "laplace_ratio_from_force_integral": ratio_force,
    }
    return (
        AuditCheck(
            "korteweg_force_direction",
            bool(delta_p_force > 0.0),
            "net capillary force at a convex liquid interface points toward the liquid: int F.n_out dr < 0",
            observed,
        ),
        ratio_force,
    )


def _check_projection_convention(p) -> AuditCheck:
    X, Y = pf.grids(p)
    psi = 1.0e-2 * jnp.sin(2.0 * jnp.pi * X / p.Lx) * jnp.cos(4.0 * jnp.pi * Y / p.Ly)
    psi = psi.astype(p.dtype)
    fx, fy = pf._ddx(psi, p.dx), pf._ddy(psi, p.dy)  # F = +grad(psi)
    div = pf._ddx(fx, p.dx) + pf._ddy(fy, p.dy)
    pressure = pf.poisson_solve(div, p.m2_proj)
    target = _np(psi) - float(_np(psi).mean())
    rel_error = float(np.linalg.norm(_np(pressure) - target) / np.linalg.norm(target))
    residual = float(
        np.linalg.norm(_np(fx - pf._ddx(pressure, p.dx)) + _np(fy - pf._ddy(pressure, p.dy)))
        / np.linalg.norm(_np(fx) + _np(fy))
    )
    observed = {"pressure_vs_psi_relative_error": rel_error, "gradient_force_residual_after_projection": residual}
    return AuditCheck(
        "pressure_gradient_convention",
        bool(rel_error < _IDENTITY_TOL and residual < _IDENTITY_TOL),
        "F = +grad(psi) is balanced by grad(P) with P = +psi (momentum carries -grad(P))",
        observed,
    )


def _check_diagnostic_equals_projection(p, solid, ellipse) -> AuditCheck:
    _, u_rhs, v_rhs, _, _ = pf.rhs(ellipse, solid, p)
    pressure = pf.pressure_field(ellipse, solid, p)
    dt = p.dt / 3.0
    damp = 1.0 / (1.0 + dt * solid.chi / p.eta_pen)
    u_star, v_star = (ellipse.u + dt * u_rhs) * damp, (ellipse.v + dt * v_rhs) * damp
    div_before = pf._ddx(u_star, p.dx) + pf._ddy(v_star, p.dy)
    u_plus = u_star - dt * pf._ddx(pressure, p.dx)
    v_plus = v_star - dt * pf._ddy(pressure, p.dy)
    div_after = pf._ddx(u_plus, p.dx) + pf._ddy(v_plus, p.dy)
    ratio = float(np.linalg.norm(_np(div_after)) / max(np.linalg.norm(_np(div_before)), 1.0e-30))
    return AuditCheck(
        "diagnostic_equals_projection",
        bool(ratio < _IDENTITY_TOL),
        "pressure_field() is the projection pressure of step(): u* - dt grad(P) is divergence free",
        {"div_after_over_div_before": ratio},
    )


def _check_laplace_jump(p, solid, state, geo, R) -> tuple[AuditCheck, float]:
    pressure = pf.pressure_field(state, solid, p)
    phi = _np(state.phi)
    ratio = _laplace_ratio_from_pressure(pressure, geo, p) * R
    observed = {
        "laplace_ratio": ratio,
        "phi_mean_inside_probe": float(phi[geo["inside"]].mean()),
        "phi_mean_outside_probe": float(phi[geo["outside"]].mean()),
    }
    passed = (
        ratio > 0.0
        and _SCALE_BAND[0] < ratio < _SCALE_BAND[1]
        and observed["phi_mean_inside_probe"] > _LIQUID_MIN
        and observed["phi_mean_outside_probe"] < _GAS_MAX
    )
    return (
        AuditCheck(
            "laplace_jump_sign_and_scale",
            bool(passed),
            "delta_p = P_liquid - P_gas > 0 and delta_p R We ~ +1 (probes: liquid core, far gas)",
            observed,
        ),
        ratio,
    )


def _check_force_matches_projection(ratio_force: float, ratio_projection: float) -> AuditCheck:
    difference = abs(ratio_force - ratio_projection)
    passed = difference <= _AGREEMENT_TOL * max(abs(ratio_projection), 1.0)
    return AuditCheck(
        "force_integral_matches_projection_jump",
        bool(passed),
        "the force integral -int F_r dr and the projection-pressure jump are the same quantity "
        "(catches any compensating sign in the diagnostic)",
        {
            "ratio_from_force_integral": ratio_force,
            "ratio_from_projection_pressure": ratio_projection,
            "abs_difference": difference,
        },
    )


def _check_free_energy(p, solid, ellipse) -> AuditCheck:
    _, u_rhs, v_rhs, mu, _ = pf.rhs(ellipse, solid, p)
    pressure = pf.pressure_field(ellipse, solid, p)
    dt = p.dt / 3.0
    u_sol = dt * (u_rhs - pf._ddx(pressure, p.dx))  # velocity driven by the solenoidal part of F
    v_sol = dt * (v_rhs - pf._ddy(pressure, p.dy))
    px, py = pf._ddx(ellipse.phi, p.dx), pf._ddy(ellipse.phi, p.dy)
    # dF/dt(advection) = -int mu u.grad(phi); the flow must *decrease* the interfacial free energy.
    power = float(np.sum(_np(mu) * (_np(u_sol) * _np(px) + _np(v_sol) * _np(py))) * p.dx * p.dy)
    cosine = _cos(_np(mu) * _np(px), _np(mu) * _np(py), _np(u_sol), _np(v_sol))
    return AuditCheck(
        "free_energy_consistency",
        bool(power > 0.0 and cosine > 0.0),
        "flow driven by the capillary force lowers F: -dF/dt = int mu u.grad(phi) > 0 (cos(mu grad phi, u) > 0)",
        {"minus_dF_dt_per_unit": power / dt, "cosine_mu_grad_phi_vs_driven_velocity": cosine},
    )


def _check_form_equivalence(p, solid, ellipse) -> AuditCheck:
    _, u_rhs, v_rhs, mu, _ = pf.rhs(ellipse, solid, p)
    coeff = float(pf.SIGMA_NORM) / float(p.We) / float(p.rho_l)
    phi = ellipse.phi
    a_x, a_y = coeff * mu * pf._ddx(phi, p.dx), coeff * mu * pf._ddy(phi, p.dy)  # Form A: + mu grad(phi)
    b_x, b_y = -coeff * phi * pf._ddx(mu, p.dx), -coeff * phi * pf._ddy(mu, p.dy)  # Form B: - phi grad(mu)
    sa_x, sa_y, _ = _solenoidal(a_x, a_y, p)
    sb_x, sb_y, _ = _solenoidal(b_x, b_y, p)
    ss_x, ss_y, _ = _solenoidal(u_rhs, v_rhs, p)
    diff_x, diff_y, _ = _solenoidal(a_x - b_x, a_y - b_y, p)
    sa_norm = math.sqrt(float(np.sum(_np(sa_x) ** 2 + _np(sa_y) ** 2)))
    diff_norm = math.sqrt(float(np.sum(_np(diff_x) ** 2 + _np(diff_y) ** 2)))
    cos_ab = _cos(_np(sa_x), _np(sa_y), _np(sb_x), _np(sb_y))
    cos_solver_a = _cos(_np(ss_x), _np(ss_y), _np(sa_x), _np(sa_y))
    cos_solver_b = _cos(_np(ss_x), _np(ss_y), _np(sb_x), _np(sb_y))
    observed = {
        "cos_solenoidal_FormA_vs_FormB": cos_ab,
        "nongradient_part_of_A_minus_B_over_solenoidal_A": diff_norm / max(sa_norm, 1.0e-30),
        "cos_solver_vs_FormA": cos_solver_a,
        "cos_solver_vs_FormB": cos_solver_b,
    }
    passed = cos_ab > _COSINE_MIN and cos_solver_a > _COSINE_MIN and cos_solver_b > _COSINE_MIN
    return AuditCheck(
        "force_form_equivalence_class",
        bool(passed),
        "Form A (+mu grad phi) and Form B (-phi grad mu) drive the same flow (cos ~ +1); "
        "the solver force belongs to that class, not to its negative",
        observed,
        note="Form A and Form B differ by grad(mu phi) only in the continuum; on the grid the "
        "difference is O((dx/eps)^2).",
    )


def _check_relaxed_mu_form_choice(p, geo, R) -> AuditCheck:
    """Why Form A: a relaxed drop (mu uniform) keeps its Laplace jump only in Form A."""
    mu0 = math.sqrt(2.0) / 6.0 / R  # equilibrium chemical potential of a circle (sigma_e / R)
    X, Y = pf.grids(p)
    rx, ry = X - p.Lx / 2.0, Y - p.Ly / 2.0
    r = jnp.sqrt(rx**2 + ry**2)
    phi = (0.5 * (1.0 - jnp.tanh((r - R) / (jnp.sqrt(2.0) * p.eps)))).astype(p.dtype)
    coeff = float(pf.SIGMA_NORM) / float(p.We)
    ones = jnp.ones_like(phi)
    fa_x, fa_y = coeff * mu0 * ones * pf._ddx(phi, p.dx), coeff * mu0 * ones * pf._ddy(phi, p.dy)
    fb_x, fb_y = -coeff * phi * pf._ddx(mu0 * ones, p.dx), -coeff * phi * pf._ddy(mu0 * ones, p.dy)
    _, _, pa = _solenoidal(fa_x, fa_y, p)
    _, _, pb = _solenoidal(fb_x, fb_y, p)
    ratio_a = _laplace_ratio_from_pressure(pa, geo, p) * R
    ratio_b = _laplace_ratio_from_pressure(pb, geo, p) * R
    return AuditCheck(
        "relaxed_mu_form_choice",
        bool(0.9 < ratio_a < 1.1 and abs(ratio_b) < 1.0e-3),
        "informational: for uniform (relaxed) mu Form A keeps delta_p R We = +1, Form B loses it (reduced pressure)",
        {"laplace_ratio_FormA": ratio_a, "laplace_ratio_FormB": ratio_b},
        note="Form A is retained: its projection pressure is the mechanical pressure whose jump is the "
        "Young-Laplace value for any admissible mu; this is a design argument, not a benchmark fit.",
    )


# ---------------------------------------------------------------------------------------
#  driver
# ---------------------------------------------------------------------------------------


def run_audit(N: int = 128, R: float = 0.8, We: float = 100.0, eps_factor: float = 2.0) -> AuditResult:
    """Evaluate every convention check on the solver currently imported as ``phasefield``."""
    if N < 32 or R <= 0.0 or We <= 0.0 or eps_factor <= 0.0:
        raise ValueError("N must be >= 32 and R, We, eps_factor must be positive")
    p = _make_params(N, We, eps_factor)
    solid = pf.empty_solid(p)
    state = pf.droplet_initial_state(p, x0=p.Lx / 2.0, y0=p.Ly / 2.0, R=R, u_impact=0.0)
    geo = _geometry(p, R)
    ellipse = _elliptical_state(p, R)

    result = AuditResult(
        conventions=dict(CONVENTIONS),
        settings={
            "N": int(N),
            "R": float(R),
            "We": float(We),
            "eps": float(p.eps),
            "eps_over_dx": float(p.eps / p.dx),
            "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
            "sigma_norm": float(pf.SIGMA_NORM),
        },
    )
    orientation = _check_phi_orientation(p, solid, state, geo, R)
    potential = _check_chemical_potential(p, solid, state, geo, R)
    direction, ratio_force = _check_force_direction(p, solid, state, geo, R)
    projection = _check_projection_convention(p)
    diagnostic = _check_diagnostic_equals_projection(p, solid, ellipse)
    laplace, ratio_projection = _check_laplace_jump(p, solid, state, geo, R)
    agreement = _check_force_matches_projection(ratio_force, ratio_projection)
    energy = _check_free_energy(p, solid, ellipse)
    forms = _check_form_equivalence(p, solid, ellipse)
    relaxed = _check_relaxed_mu_form_choice(p, geo, R)
    result.checks = [
        orientation,
        potential,
        direction,
        projection,
        diagnostic,
        laplace,
        agreement,
        energy,
        forms,
        relaxed,
    ]
    force_ok = direction.passed and energy.passed and forms.passed
    projection_ok = projection.passed and diagnostic.passed
    diagnostic_ok = agreement.passed
    result.diagnosis = {
        "force_convention_consistent": bool(force_ok),
        "projection_convention_consistent": bool(projection_ok),
        "diagnostic_pressure_convention_consistent": bool(diagnostic_ok),
        "three_conventions_consistent": bool(force_ok and projection_ok and diagnostic_ok and laplace.passed),
    }
    return result


def format_report(result: AuditResult) -> str:
    lines = ["Capillary / pressure sign audit (no time stepping, solver operators on analytic interfaces)", ""]
    settings = result.settings
    lines.append(
        "settings: N={N} R={R} We={We} eps/dx={eps_over_dx:g} SOLVER_CONTRACT_VERSION={solver_contract_version}".format(
            **settings
        )
    )
    lines += ["", "Derived conventions:"]
    lines += [f"  - {key}: {text}" for key, text in result.conventions.items()]
    lines += ["", "Checks:"]
    for check in result.checks:
        lines.append(f"  [{'PASS' if check.passed else 'FAIL'}] {check.name}")
        lines.append(f"         expect: {check.expectation}")
        lines.append("         observed: " + ", ".join(f"{k}={v:.6g}" for k, v in check.observed.items()))
    lines += ["", "Diagnosis:"]
    lines += [f"  {key}: {value}" for key, value in result.diagnosis.items()]
    lines.append("")
    lines.append(
        "VERDICT: CONSISTENT"
        if result.all_passed
        else "VERDICT: INCONSISTENT (a sign convention disagrees with the free-energy derivation)"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analytic capillary / pressure sign audit for the two-phase solver")
    parser.add_argument("--N", type=int, default=128, help="grid size (default: baseline N=128)")
    parser.add_argument("--R", type=float, default=0.8, help="drop radius (default 0.8)")
    parser.add_argument("--We", type=float, default=100.0)
    parser.add_argument("--eps-factor", type=float, default=2.0, help="eps / dx (default: baseline 2.0)")
    parser.add_argument("--json", default=None, help="optional path for a strict-JSON audit record")
    args = parser.parse_args(argv)
    try:
        result = run_audit(N=args.N, R=args.R, We=args.We, eps_factor=args.eps_factor)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(format_report(result))
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0 if result.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
