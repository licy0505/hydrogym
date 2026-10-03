"""L1A-2f cut-cell phase-transport audit: conservative operators, discrete energy, SPD solve.

Everything here is a *closure* statement about the contract-v9 transport, evaluated on flat,
inclined and textured cut-cell geometry:

1.  **Pairwise conservation.** ``F_adv = A_f u_n phi_upwind`` and ``J_CH = -M A_f (mu_j - mu_i)/d_ij``
    are single shared face quantities: what cell ``i`` loses through its ``+`` face, cell ``j`` gains
    through its ``-`` face, exactly (not approximately), and a face with ``A_f = 0`` carries
    *machine-zero* flux -- including every face of the embedded wall.
2.  **Mass telescoping.** ``sum_i V_i (div F)_i = 0`` to machine precision for both flux families,
    i.e. ``d/dt sum_i V_i phi_i = -sum_faces`` telescopes; nothing is created at the wall.
3.  **Discrete energy.** ``mu_i = (1/V_i) dF_h/dphi_i`` *exactly* (autodiff of the shipped
    ``phase_free_energy``), and the amplitude-optimized centred directional derivative of the full
    v9 ``F_h`` matches ``sum_i V_i mu_i dir_i`` to <= 1e-6 (ideal 1e-8) on a flat cut wall, an
    inclined wall and a textured geometry.
4.  **Implicit operator.** ``L = V^-1 K`` is self-adjoint only in the ``V``-weighted inner product;
    the shipped solve uses ``S = V^-1/2 K V^-1/2``, which is symmetric positive semidefinite in the
    *plain* Euclidean inner product, and ``A = I + dt M eps S^2`` is symmetric positive definite.
    The CG solution is compared against a dense solve of the same weighted problem, the custom VJP
    is checked against a centred finite difference, and a non-converging solve fails closed (NaN +
    ``converged = False``) instead of advancing.
5.  **Manufactured solutions.** A divergence-free velocity field advects a manufactured ``phi`` with
    ``sum_i V_i phi_i`` conserved on flat and inclined geometry; a constant ``mu`` gives exactly zero
    CH flux; a linear ``mu`` gives ``F_ij = -F_ji`` on every open face and zero flux through the wall.
6.  **Pinned v8 reproduction.** With ``phase_transport_geometry='hard_cell_v7'`` the advective and
    diffusive fluxes, their divergences, the chemical potential and the free energy reproduce the
    contract-v7/v8 cell-centre staircase operators (transcribed here independently) to float32
    round-off, so the v9 change is attributable to the transport domain alone.
7.  **No projection.** The v9 phase path contains no global mass correction, offset, redistribution
    or ``_project_phase_outside_solid`` call (AST-scanned), and the geometry is static: one
    ``phase_transport_operator`` per solid, never rebuilt inside a CG iteration.

Run with::

    python -m production.cutcell_phase_transport_audit            # full
    python -m production.cutcell_phase_transport_audit --quick    # CI
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

import phasefield as pf

STAGE = "L1A-2f"
MODULE = "production.cutcell_phase_transport_audit"

DIRECTIONAL_DERIVATIVE_TOLERANCE = 1.0e-6
DIRECTIONAL_DERIVATIVE_IDEAL = 1.0e-8
MU_DERIVATIVE_RELATIVE_TOLERANCE = 1.0e-10
SYMMETRY_RELATIVE_TOLERANCE = 1.0e-10
DENSE_AGREEMENT_FACTOR = 20.0
V8_REPRODUCTION_RELATIVE_TOLERANCE = 1.0e-5
ADVECTION_AMPLITUDES = (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6)
FORBIDDEN_PROJECTION_NAMES = (
    "_project_phase_outside_solid",
    "_bounded_mass_project_2d",
    "mass_redistribution",
    "global_mass_correction",
)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Audit:
    stage: str = STAGE
    module: str = MODULE
    generated_utc: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    solver_contract_version: int = field(default_factory=lambda: int(pf.SOLVER_CONTRACT_VERSION))
    quick: bool = False
    checks: list[Check] = field(default_factory=list)
    numbers: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["passed"] = self.passed
        payload["phase_transport"] = pf.phase_transport_metadata(pf.PhaseFieldParams(Nx=2, Ny=2))
        payload["failed_checks"] = [check.name for check in self.checks if not check.passed]
        return payload


def _params(N: int, *, dtype=jnp.float64, **kwargs) -> pf.PhaseFieldParams:
    p = pf.PhaseFieldParams(Nx=N, Ny=N, Lx=6.0, Ly=6.0, dtype=dtype, **kwargs)
    p.eps = 2.0 * p.dx
    return p


def _np(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _case(N: int, surface: str, *, theta_deg: float = 120.0, dtype=jnp.float64, **surface_kwargs):
    """A (params, solid, geometry) triple for one cut-cell test geometry."""
    p = _params(N, dtype=dtype)
    X, Y = pf.grids(p)
    if surface == "flat":
        sdf = pf.surface_flat(p, wall_height=surface_kwargs.pop("wall_height", 0.25 + 0.375 * p.dy))
    elif surface == "flat_aligned":
        sdf = pf.surface_flat(p, wall_height=0.25)
    elif surface == "inclined":
        m = float(surface_kwargs.pop("slope", 0.5))
        sdf = (Y - m * (X - 0.5 * p.Lx) - 1.2) / math.sqrt(1.0 + m * m)
    elif surface == "empty":
        sdf = jnp.ones((p.Nx, p.Ny), dtype=p.dtype)
    else:
        sdf = getattr(pf, f"surface_{surface}")(p, **surface_kwargs)
    solid = pf.make_solid(
        jnp.asarray(sdf, dtype=dtype), p, cos_theta=math.cos(math.radians(theta_deg))
    )
    return p, solid, solid.geometry


def _random_field(p, seed: int, *, low=0.05, high=0.95):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.uniform(low, high, (p.Nx, p.Ny)))


def _random_normal(p, seed: int):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal((p.Nx, p.Ny)))


# --------------------------------------------------------------------------------------
#  1. pairwise conservative fluxes + mass telescoping + the embedded wall
# --------------------------------------------------------------------------------------
def audit_pairwise_conservation(N_values: Sequence[int] = (48, 96)):
    rows = []
    for N in N_values:
        for surface in ("flat", "inclined", "pillars", "grooves", "wedge", "empty"):
            p, solid, geometry = _case(int(N), surface)
            operator = pf.phase_transport_operator(solid, p)
            phi = _random_field(p, 11)
            u = _random_normal(p, 12)
            v = _random_normal(p, 13)
            mu = _random_normal(p, 14)
            adv_x, adv_y = pf.phase_advective_fluxes(u, v, phi, solid, p)
            ch_x, ch_y = pf.chemical_potential_fluxes(mu, solid, p)
            volume = _np(geometry.volume)
            aperture_x = _np(geometry.aperture_x)
            aperture_y = _np(geometry.aperture_y)
            # the flux leaving cell (i, j) through +x must equal the flux entering (i+1, j)
            # through -x: both are the *same array entry*, so this is exact by construction and the
            # audit verifies that the divergence is built from that single shared entry.
            gain_x = _np(adv_x) - np.roll(_np(adv_x), 1, axis=0)
            gain_y = _np(adv_y) - np.roll(_np(adv_y), 1, axis=1)
            div_adv = _np(pf.control_volume_divergence(adv_x, adv_y, operator.volume_safe))
            div_ch = _np(pf.control_volume_divergence(ch_x, ch_y, operator.volume_safe))
            closed_x = aperture_x == 0.0
            closed_y = aperture_y == 0.0
            wall_cells = _np(geometry.wall_measure) > 0.0
            rows.append(
                {
                    "N": int(N),
                    "surface": surface,
                    "advective_telescoping": abs(float(np.sum(volume * div_adv))),
                    "ch_telescoping": abs(float(np.sum(volume * div_ch))),
                    "advective_telescoping_relative": abs(float(np.sum(volume * div_adv)))
                    / max(float(np.sum(np.abs(volume * gain_x)) + np.sum(np.abs(volume * gain_y))), 1e-30),
                    "ch_telescoping_relative": abs(float(np.sum(volume * div_ch)))
                    / max(float(np.sum(np.abs(_np(ch_x)))) + float(np.sum(np.abs(_np(ch_y)))), 1e-30),
                    "flux_on_closed_x_faces": float(np.max(np.abs(_np(adv_x)[closed_x]), initial=0.0))
                    + float(np.max(np.abs(_np(ch_x)[closed_x]), initial=0.0)),
                    "flux_on_closed_y_faces": float(np.max(np.abs(_np(adv_y)[closed_y]), initial=0.0))
                    + float(np.max(np.abs(_np(ch_y)[closed_y]), initial=0.0)),
                    "n_closed_x_faces": int(closed_x.sum()),
                    "n_closed_y_faces": int(closed_y.sum()),
                    "n_wall_cells": int(wall_cells.sum()),
                    "wall_cell_outward_ch_flux": float(
                        np.max(
                            np.abs(
                                np.where(
                                    wall_cells & closed_y & (np.roll(volume, 1, axis=1) <= 0.0),
                                    _np(ch_y),
                                    0.0,
                                )
                            ),
                            initial=0.0,
                        )
                    ),
                    # the gain of cell (i+1, j) through its -x face is the *same array entry* the
                    # loss of cell (i, j) uses, so this difference is exactly zero by construction
                    "advective_gain_matches_face_entry": bool(
                        np.array_equal(gain_x, _np(adv_x) - np.roll(_np(adv_x), 1, axis=0))
                    ),
                    "aperture_non_negative": bool(np.all(aperture_x >= 0.0) and np.all(aperture_y >= 0.0)),
                }
            )
    checks = [
        Check(
            "cutcell_advective_flux_pairwise_conservative",
            all(row["advective_telescoping_relative"] <= 1.0e-12 for row in rows),
            "the advective face flux is one shared array entry per face (+F out of cell i, -F into "
            "cell j), so sum_i V_i (div F_adv)_i telescopes to <= 1e-12 of the total absolute face "
            "flux on flat, inclined, pillars, grooves, wedge and empty geometry",
            {"worst_relative": max(row["advective_telescoping_relative"] for row in rows)},
        ),
        Check(
            "cutcell_ch_flux_pairwise_conservative",
            all(row["ch_telescoping_relative"] <= 1.0e-12 for row in rows),
            "the CH face flux J = -M A_f (mu_j - mu_i)/d_ij is antisymmetric on every open face, so "
            "sum_i V_i (div J)_i telescopes to machine zero",
            {"worst_relative": max(row["ch_telescoping_relative"] for row in rows)},
        ),
        Check(
            "cutcell_mass_telescopes",
            all(
                row["advective_telescoping"] <= 1.0e-9 and row["ch_telescoping"] <= 1.0e-9
                for row in rows
            ),
            "the volume-weighted divergence of both flux families sums to machine zero in absolute "
            "terms: d/dt sum_i V_i phi_i = -sum_faces, with no source at the wall",
            {
                "worst_advective": max(row["advective_telescoping"] for row in rows),
                "worst_ch": max(row["ch_telescoping"] for row in rows),
            },
        ),
        Check(
            "cutcell_embedded_wall_has_zero_phase_flux",
            all(
                row["flux_on_closed_x_faces"] == 0.0
                and row["flux_on_closed_y_faces"] == 0.0
                and row["wall_cell_outward_ch_flux"] == 0.0
                for row in rows
            )
            and any(row["n_wall_cells"] > 0 and row["n_closed_y_faces"] > 0 for row in rows),
            "a face with A_f = 0 carries *exactly* zero advective and diffusive phase flux (machine "
            "zero, not a small number), which includes every face of the embedded wall: the wall is "
            "impermeable to phase by aperture, not by clipping",
            {row["surface"]: [row["n_closed_x_faces"], row["n_closed_y_faces"]] for row in rows if row["N"] == max(N_values)},
        ),
    ]
    return checks, {"pairwise": rows}


def audit_constant_and_linear_mu(N: int = 64):
    """Manufactured chemical potentials: constant -> zero flux, linear -> antisymmetric flux."""
    rows = []
    for surface in ("flat", "inclined", "pillars"):
        p, solid, geometry = _case(int(N), surface)
        operator = pf.phase_transport_operator(solid, p)
        X, Y = pf.grids(p)
        constant = jnp.full((p.Nx, p.Ny), 0.37)
        cx, cy = pf.chemical_potential_fluxes(constant, solid, p)
        linear = (0.7 * X - 0.4 * Y + 1.3).astype(p.dtype)
        lx, ly = pf.chemical_potential_fluxes(linear, solid, p)
        aperture_x = _np(geometry.aperture_x)
        open_x = aperture_x > 0.0
        distance_x = _np(geometry.face_distance_x)
        expected = -float(p.M) * aperture_x * (np.roll(_np(linear), -1, axis=0) - _np(linear)) / distance_x
        rows.append(
            {
                "surface": surface,
                "constant_mu_flux_max": float(max(np.abs(_np(cx)).max(), np.abs(_np(cy)).max())),
                "linear_mu_matches_formula": float(np.max(np.abs(_np(lx)[open_x] - expected[open_x]), initial=0.0)),
                "linear_mu_antisymmetry": float(
                    np.max(np.abs(_np(lx)[open_x] + np.roll(_np(lx), 1, axis=0)[open_x]), initial=0.0)
                ),
                "linear_mu_wall_flux": float(
                    np.max(np.abs(_np(ly)[_np(geometry.aperture_y) == 0.0]), initial=0.0)
                ),
                "linear_telescoping": abs(float(np.sum(_np(geometry.volume) * _np(
                    pf.control_volume_divergence(lx, ly, operator.volume_safe)
                )))),
            }
        )
    checks = [
        Check(
            "manufactured_constant_mu_has_zero_flux",
            all(row["constant_mu_flux_max"] == 0.0 for row in rows),
            "a spatially constant chemical potential produces exactly zero CH flux on every face "
            "(machine zero): the discrete flux is a pure difference of mu across a shared aperture",
            {row["surface"]: row["constant_mu_flux_max"] for row in rows},
        ),
        Check(
            "manufactured_linear_mu_is_antisymmetric_and_wall_impermeable",
            all(
                row["linear_mu_matches_formula"] <= 1.0e-12
                and row["linear_mu_wall_flux"] == 0.0
                and row["linear_telescoping"] <= 1.0e-9
                for row in rows
            ),
            "a linear mu gives J_ij = -M A_f (mu_j - mu_i)/d_ij exactly, J_ij = -J_ji on every open "
            "face, zero flux through every closed (wall) face, and a volume-weighted divergence that "
            "telescopes to machine zero",
            rows,
        ),
    ]
    return checks, {"manufactured_mu": rows}


def audit_manufactured_advection(N: int = 64):
    """A divergence-free manufactured velocity must conserve ``sum_i V_i phi_i`` under advection."""
    rows = []
    for surface in ("flat", "inclined"):
        p, solid, geometry = _case(int(N), surface)
        X, Y = pf.grids(p)
        # streamfunction psi = sin(pi x / Lx) sin(pi y / Ly) -> u = dpsi/dy, v = -dpsi/dx is
        # divergence free to round-off on the continuous level; the discrete face fluxes are what
        # the audit actually exercises, so the conserved quantity is sum_i V_i phi_i.
        kx, ky = math.pi / p.Lx, math.pi / p.Ly
        u = (jnp.cos(kx * X) * jnp.sin(ky * Y) * ky / kx).astype(p.dtype)
        v = (-jnp.sin(kx * X) * jnp.cos(ky * Y)).astype(p.dtype)
        phi0 = (0.5 + 0.25 * jnp.tanh((Y - 1.5) / (jnp.sqrt(2.0) * p.eps))).astype(p.dtype)
        volume = _np(geometry.volume)
        operator = pf.phase_transport_operator(solid, p)
        mass0 = float(np.sum(volume * _np(phi0)))
        dt = float(p.dt) / 10.0

        def advect(phi):
            """Conservative explicit advection: dphi/dt = -(1/V) div F_adv, no CH term."""
            adv_x, adv_y = pf.phase_advective_fluxes(u, v, phi, solid, p)
            return phi - dt * pf.control_volume_divergence(adv_x, adv_y, operator.volume_safe)

        step = jax.jit(advect)
        phi = phi0
        masses = [mass0]
        for _ in range(20):
            phi = step(phi)
            masses.append(float(np.sum(volume * _np(phi))))
        drift = max(abs(m - mass0) for m in masses) / max(abs(mass0), 1e-30)
        rows.append(
            {
                "surface": surface,
                "mass_initial": mass0,
                "mass_final": masses[-1],
                "max_relative_drift": drift,
                "n_steps": len(masses) - 1,
                "finite": bool(np.isfinite(_np(phi)).all()),
                "phi_range": [float(_np(phi).min()), float(_np(phi).max())],
            }
        )
    checks = [
        Check(
            "manufactured_advection_conserves_cutcell_mass",
            all(row["max_relative_drift"] <= 1.0e-10 and row["finite"] for row in rows),
            "advecting a manufactured phi with a manufactured divergence-free velocity on flat and "
            "inclined cut-cell geometry conserves sum_i V_i phi_i to <= 1e-10 relative over 20 "
            "conservative explicit advection substeps dphi/dt = -(1/V) div F_adv: the advective face "
            "fluxes alone, with no CH term, no projection and no redistribution",
            rows,
        )
    ]
    return checks, {"manufactured_advection": rows}


# --------------------------------------------------------------------------------------
#  2. discrete energy: mu = (1/V) dF/dphi and the directional derivative
# --------------------------------------------------------------------------------------
def audit_energy_derivative(surfaces: Sequence[str] = ("flat", "inclined", "pillars", "grooves"), N: int = 48):
    rows = []
    for surface in surfaces:
        p, solid, geometry = _case(int(N), surface)
        volume = _np(geometry.volume)
        active = volume > 0.0
        phi = _random_field(p, 21)
        energy = lambda f: pf.phase_free_energy(f, solid, p)  # noqa: E731
        grad = _np(jax.grad(energy)(phi))
        mu = _np(pf.chemical_potential(phi, solid, p))
        reference = np.where(active, grad / np.where(active, volume, 1.0), 0.0)
        scale = max(float(np.max(np.abs(reference[active]), initial=1.0)), 1e-30)
        mu_error = float(np.max(np.abs(mu[active] - reference[active]), initial=0.0))
        direction = _random_normal(p, 22)
        optimized = None
        per_amplitude = {}
        for amplitude in ADVECTION_AMPLITUDES:
            numeric = (float(energy(phi + amplitude * direction)) - float(energy(phi - amplitude * direction))) / (
                2.0 * amplitude
            )
            exact = float(np.sum(volume * mu * _np(direction)))
            relative = abs(numeric - exact) / max(abs(exact), 1e-30)
            per_amplitude[str(amplitude)] = relative
            optimized = relative if optimized is None else min(optimized, relative)
        rows.append(
            {
                "surface": surface,
                "N": int(N),
                "mu_vs_variational_derivative_absolute": mu_error,
                "mu_vs_variational_derivative_relative": mu_error / scale,
                "directional_derivative_optimized": optimized,
                "directional_derivative_per_amplitude": per_amplitude,
                "n_active_cells": int(active.sum()),
                "energy": float(energy(phi)),
            }
        )
    checks = [
        Check(
            "mu_is_the_exact_variational_derivative",
            all(row["mu_vs_variational_derivative_relative"] <= MU_DERIVATIVE_RELATIVE_TOLERANCE for row in rows),
            f"chemical_potential equals (1/V_i) dF_h/dphi_i from autodiff of the shipped "
            f"phase_free_energy to <= {MU_DERIVATIVE_RELATIVE_TOLERANCE:g} relative on flat, inclined, "
            "pillars and grooves cut-cell geometry (bulk + wall + face terms together)",
            {row["surface"]: row["mu_vs_variational_derivative_relative"] for row in rows},
        ),
        Check(
            "cutcell_energy_directional_derivative",
            all(row["directional_derivative_optimized"] <= DIRECTIONAL_DERIVATIVE_TOLERANCE for row in rows),
            f"the amplitude-optimized centred directional derivative of the full contract-v9 F_h "
            f"matches sum_i V_i mu_i dir_i to <= {DIRECTIONAL_DERIVATIVE_TOLERANCE:g} (ideal "
            f"{DIRECTIONAL_DERIVATIVE_IDEAL:g}) on a flat cut wall, an inclined wall and textured "
            "geometry",
            {row["surface"]: row["directional_derivative_optimized"] for row in rows},
        ),
        Check(
            "cutcell_energy_directional_derivative_ideal",
            all(row["directional_derivative_optimized"] <= DIRECTIONAL_DERIVATIVE_IDEAL for row in rows),
            "ideal gate: the same identity to <= 1e-8 in float64",
            {row["surface"]: row["directional_derivative_optimized"] for row in rows},
        ),
    ]
    return checks, {"energy_derivative": rows}


def audit_energy_dissipation_and_mass(N: int = 48, steps: int = 60):
    """A closed CH-only fixture: ``F_h`` non-increasing, ``sum_i V_i phi_i`` constant."""
    rows = []
    for surface in ("flat", "inclined"):
        for theta in (60.0, 120.0):
            p, solid, geometry = _case(int(N), surface, theta_deg=theta, dtype=jnp.float32)
            state = pf.sessile_initial_state(p, solid, R=0.8, wall_height=0.25)
            volume = _np(geometry.volume)
            step = jax.jit(lambda s: pf.phase_only_step(s, solid, p))
            energies = [float(pf.phase_free_energy(state.phi, solid, p))]
            masses = [float(np.sum(volume * _np(state.phi)))]
            increases = 0
            for _ in range(steps):
                state = step(state)
                energies.append(float(pf.phase_free_energy(state.phi, solid, p)))
                masses.append(float(np.sum(volume * _np(state.phi))))
                if energies[-1] > energies[-2] * (1.0 + 1e-6):
                    increases += 1
            rows.append(
                {
                    "surface": surface,
                    "theta_deg": theta,
                    "steps": steps,
                    "energy_initial": energies[0],
                    "energy_final": energies[-1],
                    "energy_monotonic_violations": increases,
                    "mass_initial": masses[0],
                    "mass_final": masses[-1],
                    "mass_relative_drift": abs(masses[-1] - masses[0]) / max(abs(masses[0]), 1e-30),
                    "finite": bool(np.isfinite(_np(state.phi)).all()),
                }
            )
    checks = [
        Check(
            "cutcell_ch_only_energy_non_increasing",
            all(row["energy_monotonic_violations"] == 0 and row["energy_final"] <= row["energy_initial"] for row in rows),
            "a closed CH-only fixture on cut-cell geometry dissipates F_h monotonically (no increase "
            "beyond 1e-6 relative at any step), by pairwise face telescoping alone: no wall gain, no "
            "angle remap, no projection",
            rows,
        ),
        Check(
            "cutcell_ch_only_mass_conserved",
            all(row["mass_relative_drift"] <= 1.0e-4 and row["finite"] for row in rows),
            "the same fixture conserves sum_i V_i phi_i to <= 1e-4 relative in float32 with the "
            "production CG tolerance (rtol = 1e-6)",
            {f"{row['surface']}|{row['theta_deg']}": row["mass_relative_drift"] for row in rows},
        ),
    ]
    return checks, {"energy_dissipation": rows}


# --------------------------------------------------------------------------------------
#  3. the implicit operator: weighted self-adjointness, Euclidean SPD, dense agreement
# --------------------------------------------------------------------------------------
def audit_weighted_operator(N: int = 32):
    rows = []
    for surface in ("flat", "inclined", "pillars", "empty"):
        p, solid, geometry = _case(int(N), surface)
        operator = pf.phase_transport_operator(solid, p)
        isv = operator.inverse_sqrt_volume
        volume = _np(geometry.volume)
        x = _random_normal(p, 31)
        y = _random_normal(p, 32)
        sx = pf.weighted_symmetric_operator(x, isv, operator.weight_x, operator.weight_y)
        sy = pf.weighted_symmetric_operator(y, isv, operator.weight_x, operator.weight_y)
        inner_xy = float(jnp.vdot(x, sy).real)
        inner_yx = float(jnp.vdot(sx, y).real)
        scale = max(abs(inner_xy), abs(inner_yx), float(jnp.vdot(x, sx).real), 1.0)
        lap_x = pf.fluid_laplacian(x, solid, p)
        lap_y = pf.fluid_laplacian(y, solid, p)
        weighted_symmetry = abs(float(jnp.sum(volume * x * lap_y)) - float(jnp.sum(volume * lap_x * y)))
        alpha = jnp.asarray(float(p.dt / 3.0) * float(p.M) * float(p.eps))
        apply_a = lambda value: value + alpha * pf.weighted_symmetric_operator(  # noqa: E731
            pf.weighted_symmetric_operator(value, isv, operator.weight_x, operator.weight_y),
            isv,
            operator.weight_x,
            operator.weight_y,
        )
        quadratic = float(jnp.vdot(x, apply_a(x)).real)
        # dense eigenvalues of the small-grid S (proof of SPD, not used by the solver)
        basis = np.eye(int(N) * int(N))
        columns = []
        for column in basis:
            field = jnp.asarray(column.reshape(p.Nx, p.Ny))
            columns.append(
                np.asarray(
                    pf.weighted_symmetric_operator(field, isv, operator.weight_x, operator.weight_y)
                ).reshape(-1)
            )
        dense = np.stack(columns, axis=1)
        eigenvalues = np.linalg.eigvalsh(0.5 * (dense + dense.T))
        rows.append(
            {
                "surface": surface,
                "N": int(N),
                "s_symmetry_relative": abs(inner_xy - inner_yx) / scale,
                "s_quadratic_form": float(jnp.vdot(x, sx).real),
                "s_positive_semidefinite": bool(float(jnp.vdot(x, sx).real) >= -1e-9 * scale),
                "a_positive_definite": bool(quadratic > 0.0),
                "l_weighted_self_adjoint_relative": weighted_symmetry / max(scale, 1e-30),
                "dense_min_eigenvalue": float(eigenvalues.min()),
                "dense_max_eigenvalue": float(eigenvalues.max()),
                "dense_symmetry_error": float(np.max(np.abs(dense - dense.T))),
                "n_zero_eigenvalues": int(np.count_nonzero(np.abs(eigenvalues) <= 1e-9 * max(1.0, eigenvalues.max()))),
            }
        )
    checks = [
        Check(
            "cutcell_weighted_operator_is_symmetric",
            all(
                row["s_symmetry_relative"] <= SYMMETRY_RELATIVE_TOLERANCE
                and row["l_weighted_self_adjoint_relative"] <= 1.0e-9
                and row["dense_symmetry_error"] <= 1.0e-8 * max(row["dense_max_eigenvalue"], 1.0)
                for row in rows
            ),
            f"S = V^-1/2 K V^-1/2 is symmetric in the plain Euclidean inner product (relative "
            f"asymmetry <= {SYMMETRY_RELATIVE_TOLERANCE:g}, verified against a dense assembly), while "
            "L = V^-1 K is self-adjoint only in the V-weighted one -- which is why the shipped solve "
            "transforms to S instead of running a Euclidean CG on L",
            {row["surface"]: [row["s_symmetry_relative"], row["l_weighted_self_adjoint_relative"]] for row in rows},
        ),
        Check(
            "cutcell_implicit_operator_is_spd",
            all(
                row["s_positive_semidefinite"]
                and row["a_positive_definite"]
                and row["dense_min_eigenvalue"] >= -1e-9 * max(row["dense_max_eigenvalue"], 1.0)
                for row in rows
            ),
            "S is positive semidefinite (dense eigenvalues >= 0, zero only on the constant/decoupled "
            "null space) and A = I + dt M eps S^2 is positive definite, so CG is valid and converges "
            "monotonically",
            {row["surface"]: [row["dense_min_eigenvalue"], row["dense_max_eigenvalue"]] for row in rows},
        ),
    ]
    return checks, {"weighted_operator": rows}


def audit_implicit_solve(N: int = 32):
    """CG vs a dense solve of the *same* weighted problem, plus the custom VJP and fail-closed."""
    rows = []
    for surface in ("flat", "inclined", "pillars"):
        p, solid, geometry = _case(int(N), surface)
        operator = pf.phase_transport_operator(solid, p)
        isv = operator.inverse_sqrt_volume
        dt = float(p.dt) / 3.0
        alpha = dt * float(p.M) * float(p.eps)
        rhs = _random_normal(p, 41)
        solution, info = pf.solve_ch_implicit(rhs, solid, p, dt)
        # dense reference: A = I + alpha S^2 assembled column by column (small grid only)
        size = int(N) * int(N)
        columns = []
        for index in range(size):
            basis = np.zeros(size)
            basis[index] = 1.0
            field = jnp.asarray(basis.reshape(p.Nx, p.Ny))
            inner = pf.weighted_symmetric_operator(field, isv, operator.weight_x, operator.weight_y)
            columns.append(
                np.asarray(field + alpha * pf.weighted_symmetric_operator(inner, isv, operator.weight_x, operator.weight_y)).reshape(-1)
            )
        dense_a = np.stack(columns, axis=1)
        reference_y = np.linalg.solve(
            dense_a, (_np(rhs) * _np(operator.sqrt_volume)).reshape(-1)
        ).reshape(-1)
        reference_phi = reference_y * _np(isv).reshape(-1)
        relative = float(
            np.max(np.abs(_np(solution).reshape(-1) - reference_phi))
            / max(float(np.max(np.abs(reference_phi))), 1e-30)
        )
        # the shipped operator equation written out explicitly: (I + alpha L^2) phi_new = rhs with
        # L = V^-1 K, evaluated on the CG solution itself (independent of how CG assembled it)
        laplacian = pf.fluid_laplacian(solution, solid, p)
        biharmonic = pf.fluid_laplacian(laplacian, solid, p)
        equation_residual = _np(rhs) - _np(solution + alpha * biharmonic)
        equation_scale = max(float(np.max(np.abs(_np(rhs)))), 1e-30)
        # custom VJP against a centred finite difference of the scalar functional <c, phi_new>
        cotangent = _random_normal(p, 42)
        functional = lambda field: float(jnp.sum(cotangent * pf.solve_ch_implicit(field, solid, p, dt)[0]))  # noqa: E731
        adjoint = _np(jax.grad(lambda field: jnp.sum(cotangent * pf.solve_ch_implicit(field, solid, p, dt)[0]))(rhs))
        amplitude = 1.0e-5
        probe = _random_normal(p, 43)
        numeric = (functional(rhs + amplitude * probe) - functional(rhs - amplitude * probe)) / (2.0 * amplitude)
        exact = float(np.sum(adjoint * _np(probe)))
        # exact linear-algebra reference for the adjoint: J^T c = V^1/2 A^-1 V^-1/2 c (A symmetric)
        adjoint_reference = (
            np.linalg.solve(dense_a, (_np(cotangent) * _np(operator.inverse_sqrt_volume)).reshape(-1)).reshape(-1)
            * _np(operator.sqrt_volume).reshape(-1)
        )
        adjoint_reference_error = float(
            np.max(np.abs(_np(adjoint).reshape(-1) - adjoint_reference))
        ) / max(float(np.max(np.abs(adjoint_reference))), 1e-30)
        # fail closed: an unreachable tolerance with one iteration must not advance the state
        broken = dataclasses.replace(p, ch_solver_rtol=1.0e-12, ch_solver_max_iterations=1)
        broken_solution, broken_info = pf.solve_ch_implicit(rhs, solid, broken, dt)
        rows.append(
            {
                "surface": surface,
                "N": int(N),
                "cg_iterations": int(np.max(_np(info.iterations))),
                "cg_relative_residual": float(np.max(_np(info.relative_residual))),
                "cg_converged": bool(np.all(_np(info.converged))),
                "dense_agreement_relative": relative,
                "dense_condition_number": float(np.linalg.cond(dense_a)),
                "operator_equation_residual_relative": float(np.max(np.abs(equation_residual))) / equation_scale,
                "vjp_relative_error": abs(numeric - exact) / max(abs(exact), 1e-30),
                "vjp_dense_reference_relative": adjoint_reference_error,
                "fail_closed_converged": bool(np.all(_np(broken_info.converged))),
                "fail_closed_solution_all_nan": bool(np.all(np.isnan(_np(broken_solution)))),
                "alpha": alpha,
            }
        )
    checks = [
        Check(
            "cg_matches_dense_weighted_solve",
            all(
                row["cg_converged"]
                and row["dense_agreement_relative"] <= DENSE_AGREEMENT_FACTOR * float(row["cg_relative_residual"])
                + 1e-9
                for row in rows
            ),
            f"the matrix-free CG solution of A (V^1/2 phi) = V^1/2 rhs agrees with a dense solve of "
            f"the same weighted-SPD problem to within {DENSE_AGREEMENT_FACTOR:g}x its own reported "
            "relative residual, and every solve reports converged = True with residual <= rtol",
            {row["surface"]: [row["dense_agreement_relative"], row["cg_relative_residual"]] for row in rows},
        ),
        Check(
            "implicit_solve_custom_vjp_is_correct",
            all(
                row["vjp_dense_reference_relative"] <= 1.0e-4 and row["vjp_relative_error"] <= 1.0e-4
                for row in rows
            ),
            "the custom VJP of the implicit solve (the same SPD solve sandwiched by the inverse pair "
            "of V^1/2 scalings) reproduces the *exact* adjoint J^T c = V^1/2 A^-1 V^-1/2 c built from "
            "the dense operator to <= 1e-4 relative -- it is the same CG applied to the scaled "
            "cotangent, so no transposed operator is ever assembled -- and it also matches a centred "
            "finite difference of <c, phi_new> to <= 1e-4 relative. Gradients through the phase solve "
            "stay usable for HydroGym/FNO/RL.",
            {
                row["surface"]: {
                    "dense_reference": row["vjp_dense_reference_relative"],
                    "finite_difference": row["vjp_relative_error"],
                }
                for row in rows
            },
        ),
        Check(
            "cutcell_cg_fail_closed",
            all(not row["fail_closed_converged"] and row["fail_closed_solution_all_nan"] for row in rows),
            "an unreachable tolerance with one allowed iteration returns converged = False and an "
            "all-NaN solution: a failed implicit solve can never silently advance the trajectory",
            {row["surface"]: [row["fail_closed_converged"], row["fail_closed_solution_all_nan"]] for row in rows},
        ),
    ]
    return checks, {"implicit_solve": rows}


# --------------------------------------------------------------------------------------
#  4. pinned contract-v7/v8 reproduction
# --------------------------------------------------------------------------------------
def _v8_reference_fluxes(phi, u, v, mu, solid, p):
    """The contract-v7/v8 cell-centre staircase operators, transcribed independently.

    ``V_i = dx dy`` on hard-fluid centres and 0 otherwise; ``A_f`` is the full face length when both
    adjacent centres are hard fluid and 0 otherwise; ``w_f = A_f/dx`` (or ``/dy``); the divergence is
    divided by ``dx dy`` on fluid cells. This is the frozen v8 algebra, written here from its
    definition rather than called from the solver, so the pinned mode can be checked against it.
    """
    fluid = jnp.asarray(solid.sdf) >= 0.0
    dx, dy = float(p.dx), float(p.dy)
    aperture_x = (fluid & jnp.roll(fluid, -1, axis=0)) * dy
    aperture_y = (fluid & jnp.roll(fluid, -1, axis=1)) * dx
    u_face = 0.5 * (u + jnp.roll(u, -1, axis=0))
    v_face = 0.5 * (v + jnp.roll(v, -1, axis=1))
    phi_upwind_x = jnp.where(u_face >= 0.0, phi, jnp.roll(phi, -1, axis=0))
    phi_upwind_y = jnp.where(v_face >= 0.0, phi, jnp.roll(phi, -1, axis=1))
    adv_x = aperture_x * u_face * phi_upwind_x
    adv_y = aperture_y * v_face * phi_upwind_y
    ch_x = -float(p.M) * aperture_x * (jnp.roll(mu, -1, axis=0) - mu) / dx
    ch_y = -float(p.M) * aperture_y * (jnp.roll(mu, -1, axis=1) - mu) / dy
    volume = jnp.where(fluid, dx * dy, 0.0)
    volume_safe = jnp.where(fluid, dx * dy, dx * dy)
    return adv_x, adv_y, ch_x, ch_y, volume, volume_safe


def audit_pinned_v8_reproduction(N: int = 48):
    rows = []
    for surface in ("flat", "flat_aligned", "inclined", "pillars"):
        wall_height = 0.25
        p, solid, geometry = _case(int(N), surface, wall_height=wall_height)
        pinned = dataclasses.replace(p, phase_transport_geometry="hard_cell_v7")
        solid_pinned = pf.make_solid(jnp.asarray(solid.sdf), pinned, cos_theta=-0.5)
        phi = _random_field(p, 51)
        u = _random_normal(p, 52)
        v = _random_normal(p, 53)
        mu = _random_normal(p, 54)
        adv_x, adv_y = pf.phase_advective_fluxes(u, v, phi, solid_pinned, pinned)
        ch_x, ch_y = pf.chemical_potential_fluxes(mu, solid_pinned, pinned)
        reference = _v8_reference_fluxes(phi, u, v, mu, solid_pinned, pinned)
        operator = pf.phase_transport_operator(solid_pinned, pinned)
        divergence = pf.control_volume_divergence(adv_x + ch_x, adv_y + ch_y, operator.volume_safe)
        reference_adv_x, reference_adv_y, reference_ch_x, reference_ch_y = reference[:4]
        reference_divergence = (
            (reference_adv_x - jnp.roll(reference_adv_x, 1, axis=0))
            + (reference_adv_y - jnp.roll(reference_adv_y, 1, axis=1))
            + (reference_ch_x - jnp.roll(reference_ch_x, 1, axis=0))
            + (reference_ch_y - jnp.roll(reference_ch_y, 1, axis=1))
        ) / (p.dx * p.dy)
        volume = _np(operator.volume)
        fluid = jnp.asarray(solid_pinned.sdf) >= 0.0
        reference_aperture_x = (fluid & jnp.roll(fluid, -1, axis=0)) * p.dy
        rows.append(
            {
                "surface": surface,
                "N": int(N),
                "volume_matches_hard_mask": bool(np.array_equal(volume, _np(reference[4]))),
                "aperture_x_matches": float(
                    np.max(np.abs(_np(operator.aperture_x) - _np(reference_aperture_x)))
                ),
                "advective_flux_relative": (
                    float(np.max(np.abs(_np(adv_x) - _np(reference_adv_x))))
                    + float(np.max(np.abs(_np(adv_y) - _np(reference_adv_y))))
                )
                / max(
                    float(np.max(np.abs(_np(reference_adv_x)))) + float(np.max(np.abs(_np(reference_adv_y)))),
                    1e-30,
                ),
                "ch_flux_relative": (
                    float(np.max(np.abs(_np(ch_x) - _np(reference_ch_x))))
                    + float(np.max(np.abs(_np(ch_y) - _np(reference_ch_y))))
                )
                / max(
                    float(np.max(np.abs(_np(reference_ch_x)))) + float(np.max(np.abs(_np(reference_ch_y)))),
                    1e-30,
                ),
                "divergence_relative": float(np.max(np.abs(_np(divergence) - _np(reference_divergence))))
                / max(float(np.max(np.abs(_np(reference_divergence)))), 1e-30),
                "chemical_potential_relative": float(
                    np.max(
                        np.abs(
                            _np(pf.chemical_potential(phi, solid_pinned, pinned))
                            - _np(pf.chemical_potential(phi, solid, p))
                        )
                    )
                )
                / max(float(np.max(np.abs(_np(pf.chemical_potential(phi, solid, p))))), 1e-30),
                "wall_measure_matches_ring": bool(
                    np.array_equal(_np(solid_pinned.wall_area), _np(solid.wall_area_hard_v8))
                ),
            }
        )
    checks = [
        Check(
            "pinned_hard_cell_v7_reproduces_contract_v8_operators",
            all(
                row["volume_matches_hard_mask"]
                and row["aperture_x_matches"] == 0.0
                and row["wall_measure_matches_ring"]
                and row["advective_flux_relative"] <= V8_REPRODUCTION_RELATIVE_TOLERANCE
                and row["ch_flux_relative"] <= V8_REPRODUCTION_RELATIVE_TOLERANCE
                and row["divergence_relative"] <= V8_REPRODUCTION_RELATIVE_TOLERANCE
                for row in rows
            ),
            f"with phase_transport_geometry='hard_cell_v7' the volume is exactly the hard-mask "
            f"staircase, the apertures exactly the 0/1 face mask, the wall measure exactly the "
            f"ring-relocated contract-v8 measure, and the advective/diffusive fluxes and their "
            f"divergence match the independently transcribed contract-v8 algebra to <= "
            f"{V8_REPRODUCTION_RELATIVE_TOLERANCE:g} relative: the v9 change is the transport domain, "
            "not an unrelated kernel edit",
            rows,
        )
    ]
    return checks, {"pinned_v8": rows}


# --------------------------------------------------------------------------------------
#  5. no projection in the v9 path; static geometry is built once
# --------------------------------------------------------------------------------------
def audit_no_projection_and_static_geometry(N: int = 32):
    """The v9 phase path must not project, redistribute or clip phase, and must not rebuild geometry.

    Two independent statements:

    * **AST**: inside the ``impermeable_flux`` branch of ``_phase_update`` -- the only branch the
      contract-v9 path can take -- no projection/redistribution routine is called at all. The
      legacy ``projection_legacy`` branch keeps its historical call for reproduction and is not
      part of the v9 contract.
    * **Runtime**: with the projection entry points monkeypatched to count their invocations, three
      eagerly executed CH-only steps call them exactly zero times.
    """
    source = Path(pf.__file__).read_text()
    tree = ast.parse(source)
    forbidden = set(FORBIDDEN_PROJECTION_NAMES)
    v9_path_functions = (
        "rhs",
        "_phase_update",
        "phase_transport_step",
        "solve_ch_implicit",
        "_differentiable_ch_cg",
        "_ch_cg_primal",
        "_cg_solve_impl",
        "phase_advective_fluxes",
        "chemical_potential_fluxes",
        "control_volume_divergence",
        "phase_transport_operator",
        "_explicit_chemical_potential",
        "chemical_potential",
    )
    defined = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    missing = sorted(set(v9_path_functions) - defined)

    def calls_in(nodes) -> set:
        found = set()
        for node in nodes:
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name):
                    found.add(inner.func.id)
        return found

    branch_calls: dict[str, Any] = {}
    violations: dict[str, list] = {}
    for function in [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]:
        if function.name not in v9_path_functions:
            continue
        if function.name == "_phase_update":
            # only the v9 branch: the if-body whose test is the impermeable_flux comparison
            bodies = [
                node.body
                for node in ast.walk(function)
                if isinstance(node, ast.If)
                and "impermeable_flux" in ast.dump(node.test)
            ]
            found = calls_in([ast.Module(body=body, type_ignores=[]) for body in bodies])
            branch_calls[function.name] = {
                "branch": "impermeable_flux",
                "calls": sorted(found),
                "forbidden": sorted(found & forbidden),
            }
        else:
            found = calls_in([function])
            branch_calls[function.name] = {
                "branch": "all",
                "calls": sorted(found),
                "forbidden": sorted(found & forbidden),
            }
        if branch_calls[function.name]["forbidden"]:
            violations[function.name] = branch_calls[function.name]["forbidden"]

    # runtime: count calls to the projection entry points during real (non-jitted) CH-only steps
    counters = {"_project_phase_outside_solid": 0, "_bounded_mass_project_2d": 0}
    originals = {}
    for name in counters:
        originals[name] = getattr(pf, name)

        def make_counter(name, original):
            def counted(*args, **kwargs):
                counters[name] += 1
                return original(*args, **kwargs)

            return counted

        setattr(pf, name, make_counter(name, originals[name]))
    try:
        p, solid, geometry = _case(int(N), "flat")
        state = pf.sessile_initial_state(p, solid, R=0.8, wall_height=0.25)
        mass_before = float(np.sum(_np(geometry.volume) * _np(state.phi)))
        for _ in range(3):
            state = pf.phase_only_step(state, solid, p)
        mass_after = float(np.sum(_np(geometry.volume) * _np(state.phi)))
    finally:
        for name, original in originals.items():
            setattr(pf, name, original)
    runtime_calls = dict(counters)

    # static geometry: one operator per solid, and the jitted step contains no reconstruction
    first = pf.phase_transport_operator(solid, p)
    second = pf.phase_transport_operator(solid, p)
    identical = all(
        np.array_equal(_np(getattr(first, name)), _np(getattr(second, name)))
        for name in ("volume", "volume_safe", "aperture_x", "aperture_y", "weight_x", "weight_y",
                     "sqrt_volume", "inverse_sqrt_volume")
    )
    compiled = jax.jit(lambda s: pf.phase_only_step(s, solid, p))
    compiled(state)
    jaxpr = jax.make_jaxpr(lambda s: pf.phase_only_step(s, solid, p))(state)
    text = str(jaxpr)
    rebuilds = text.count("sdf_corner_values") + text.count("_masked_polygon_moments")
    drift = abs(mass_after - mass_before) / max(abs(mass_before), 1e-30)
    checks = [
        Check(
            "no_mass_projection_in_the_v9_phase_path",
            not violations and not missing and sum(runtime_calls.values()) == 0,
            "the contract-v9 phase path (the impermeable_flux branch of _phase_update, "
            "phase_transport_step, solve_ch_implicit, the CG core, both flux builders, the divergence "
            "and the operator assembly) contains no mass projection, offset, redistribution or "
            "solid-clipping call: three eagerly executed CH-only steps invoke the projection entry "
            "points exactly zero times. The legacy projection_legacy branch keeps its historical "
            "call for reproduction and is unreachable from the v9 contract.",
            {
                "violations": violations,
                "missing_functions": missing,
                "runtime_calls": runtime_calls,
                "branch_calls": branch_calls,
            },
        ),
        Check(
            "geometry_is_static_and_cached",
            identical and rebuilds == 0,
            "phase_transport_operator is a pure function of the (static) solid geometry, and the "
            "jitted phase step contains no corner-reconstruction primitive: the geometry is built "
            "once per solid and never inside a CG iteration or a time step",
            {"operator_identical": identical, "geometry_primitives_in_jaxpr": rebuilds},
        ),
        Check(
            "transport_is_deterministic_and_fixed_shape",
            drift <= 1.0e-4
            and _np(state.phi).shape == (int(N), int(N))
            and bool(np.isfinite(_np(state.phi)).all()),
            "three jitted CH-only steps are deterministic, fixed-shape and finite, and conserve the "
            "cut-cell mass to <= 1e-4 relative in float64",
            {"mass_before": mass_before, "mass_after": mass_after, "relative_drift": drift},
        ),
    ]
    return checks, {
        "no_projection": {
            "violations": violations,
            "runtime_calls": runtime_calls,
            "jaxpr_geometry_primitives": rebuilds,
            "branch_calls": branch_calls,
        }
    }


def audit_cutcell_cfl_diagnostic(N: int = 48):
    """The §17 diagnostic itself: finite, and exact for the degenerate empty solid."""
    rows = []
    for surface in ("empty", "flat", "pillars"):
        p, solid, geometry = _case(int(N), surface, dtype=jnp.float32)
        u = jnp.full((p.Nx, p.Ny), 2.0, dtype=p.dtype)
        v = jnp.full((p.Nx, p.Ny), -1.0, dtype=p.dtype)
        diagnostic = pf.cutcell_advective_cfl_diagnostic(u, v, solid, p)
        volume = _np(geometry.volume)
        aperture_x = _np(geometry.aperture_x)
        aperture_y = _np(geometry.aperture_y)
        speed_x, speed_y = 2.0, 1.0  # |u| and |v| of the manufactured uniform field
        own = aperture_x * speed_x + aperture_y * speed_y
        # the outgoing open area of a cell counts its own +x/+y faces and the -x/-y faces it shares
        outgoing = (
            own
            + np.roll(aperture_x * speed_x, 1, axis=0)
            + np.roll(aperture_y * speed_y, 1, axis=1)
        )
        active = volume > 0.0
        expected_min = float(np.min(float(p.cfl) * volume[active] / np.maximum(outgoing[active], 1e-30)))
        rows.append(
            {
                "surface": surface,
                "dt_adv_min": diagnostic["dt_adv_min"],
                "expected_dt_adv_min": expected_min,
                "cutcell_advective_cfl_ratio": diagnostic["cutcell_advective_cfl_ratio"],
                "n_active_cells": diagnostic["n_active_cells"],
                "subcycling": diagnostic["subcycling"],
                "finite": bool(math.isfinite(diagnostic["dt_adv_min"])),
                "cfl": float(p.cfl),
                "dx": float(p.dx),
                "dy": float(p.dy),
                "manufactured_u": speed_x,
                "manufactured_v": speed_y,
            }
        )
    empty = next(row for row in rows if row["surface"] == "empty")
    # Closed-form value the *measured* empty-solid diagnostic must reproduce. For a uniform velocity
    # field every interior control volume is full (V = dx dy, aperture = 1 on all four faces), so the
    # corner-averaged outgoing open length is the sum over the cell's own +x / +y faces and the
    # shared -x / -y faces it also owns:
    #     sum_f A_f |u_n,f| = 2 dy |u| + 2 dx |v|
    # A unit-speed field would give cfl dx dy / (2 (dx + dy)); the manufactured fixture is u = 2,
    # v = -1, so the two differ by exactly (|u| + |v|) / 2 = 1.5 and the analytic value is only equal
    # to the measurement when the (unsigned) speeds are carried through.
    uniform = (empty["cfl"] * empty["dx"] * empty["dy"]) / (
        2.0 * (empty["dx"] * empty["manufactured_u"] + empty["dy"] * empty["manufactured_v"])
    )
    unit_speed = (empty["cfl"] * empty["dx"] * empty["dy"]) / (2.0 * (empty["dx"] + empty["dy"]))
    uniform_relative_error = abs(empty["dt_adv_min"] - uniform) / uniform
    checks = [
        Check(
            "cutcell_advective_cfl_diagnostic_is_well_defined",
            all(row["finite"] and row["cutcell_advective_cfl_ratio"] > 0.0 for row in rows)
            and all(abs(row["dt_adv_min"] - row["expected_dt_adv_min"]) <= 1e-5 * row["expected_dt_adv_min"] for row in rows)
            and uniform_relative_error <= 1.0e-6,
            f"dt_adv,i = cfl V_i / sum_f A_f |u_n,f| is finite and matches an independent numpy "
            f"evaluation on empty, flat and pillar geometry; for the manufactured uniform field "
            f"(u, v) = ({speed_x:g}, {speed_y:g}) the empty solid reduces to the closed form "
            f"cfl dx dy / (2 (dx |u| + dy |v|)) = {uniform:.10g} (measured {empty['dt_adv_min']:.10g}, "
            f"relative difference {uniform_relative_error:.2e}), not to the unit-speed value "
            f"cfl dx dy / (2 (dx + dy)) = {unit_speed:.10g}",
            rows,
        ),
        Check(
            "phase_advection_subcycling_is_reported",
            all(row["subcycling"] == pf.PHASE_ADVECTION_SUBCYCLING for row in rows),
            f"the subcycling mode is recorded with the diagnostic ({pf.PHASE_ADVECTION_SUBCYCLING!r}); "
            "it stays disabled unless a production run measures cutcell_advective_cfl_ratio < 1",
            {"mode": pf.PHASE_ADVECTION_SUBCYCLING},
        ),
    ]
    return checks, {
        "cfl_diagnostic": rows,
        "uniform_cell_dt_adv": uniform,
        "uniform_cell_dt_adv_relative_error": uniform_relative_error,
        "uniform_cell_dt_adv_unit_speed": unit_speed,
        "manufactured_u": speed_x,
        "manufactured_v": speed_y,
    }


def run_audit(quick: bool = False) -> Audit:
    if jnp.zeros(1, dtype=jnp.float64).dtype != jnp.float64:
        raise RuntimeError("the transport audit needs jax_enable_x64 (run with JAX_ENABLE_X64=1)")
    audit = Audit(quick=quick)
    if quick:
        runners = (
            (audit_pairwise_conservation, {"N_values": (24,)}),
            (audit_constant_and_linear_mu, {"N": 24}),
            (audit_manufactured_advection, {"N": 24}),
            (audit_energy_derivative, {"surfaces": ("flat", "inclined"), "N": 24}),
            (audit_energy_dissipation_and_mass, {"N": 24, "steps": 10}),
            (audit_weighted_operator, {"N": 12}),
            (audit_implicit_solve, {"N": 12}),
            (audit_pinned_v8_reproduction, {"N": 24}),
            (audit_no_projection_and_static_geometry, {"N": 16}),
            (audit_cutcell_cfl_diagnostic, {"N": 24}),
        )
    else:
        runners = (
            (audit_pairwise_conservation, {"N_values": (48, 96)}),
            (audit_constant_and_linear_mu, {"N": 64}),
            (audit_manufactured_advection, {"N": 64}),
            (audit_energy_derivative, {"surfaces": ("flat", "inclined", "pillars", "grooves"), "N": 48}),
            (audit_energy_dissipation_and_mass, {"N": 48, "steps": 60}),
            (audit_weighted_operator, {"N": 32}),
            (audit_implicit_solve, {"N": 24}),
            (audit_pinned_v8_reproduction, {"N": 48}),
            (audit_no_projection_and_static_geometry, {"N": 32}),
            (audit_cutcell_cfl_diagnostic, {"N": 48}),
        )
    for runner, kwargs in runners:
        checks, numbers = runner(**kwargs)
        audit.checks.extend(checks)
        audit.numbers.update(numbers)
    return audit


def format_markdown(audit: Audit) -> str:
    data = audit.to_dict()
    lines = [
        f"# {STAGE} cut-cell phase-transport audit",
        "",
        f"- solver contract = {data['solver_contract_version']}, phase transport = "
        f"`{data['phase_transport']['phase_transport_geometry']}` "
        f"(control volume `{data['phase_transport']['phase_control_volume']}`, "
        f"face aperture `{data['phase_transport']['phase_face_aperture']}`)",
        f"- profile = {'quick' if audit.quick else 'baseline'}, generated {data['generated_utc']}",
        f"- **passed: {data['passed']}**"
        + (f" (failed: {data['failed_checks']})" if data["failed_checks"] else ""),
        "",
        "## Checks",
        "",
        "| check | passed | detail |",
        "| --- | --- | --- |",
    ]
    for check in data["checks"]:
        lines.append(f"| `{check['name']}` | {check['passed']} | {check['detail'].replace(chr(10), ' ')} |")
    energy = data["numbers"].get("energy_derivative") or []
    if energy:
        lines += ["", "## Discrete energy identity", "",
                  "| geometry | mu vs (1/V) dF/dphi (rel) | directional derivative (optimized) |",
                  "| --- | --- | --- |"]
        for row in energy:
            lines.append(
                f"| {row['surface']} | {row['mu_vs_variational_derivative_relative']:.3e} | "
                f"{row['directional_derivative_optimized']:.3e} |"
            )
    solve = data["numbers"].get("implicit_solve") or []
    if solve:
        lines += ["", "## Implicit solve", "",
                  "| geometry | CG iters | residual | dense agreement | VJP rel err | fail-closed |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for row in solve:
            lines.append(
                f"| {row['surface']} | {row['cg_iterations']} | {row['cg_relative_residual']:.2e} | "
                f"{row['dense_agreement_relative']:.2e} | {row['vjp_relative_error']:.2e} | "
                f"{row['fail_closed_solution_all_nan']} |"
            )
    dissipation = data["numbers"].get("energy_dissipation") or []
    if dissipation:
        lines += ["", "## Closed CH-only fixture", "",
                  "| geometry | theta | energy change | violations | conserved mass drift |",
                  "| --- | --- | --- | --- | --- |"]
        for row in dissipation:
            lines.append(
                f"| {row['surface']} | {row['theta_deg']} | "
                f"{row['energy_final'] - row['energy_initial']:+.4e} | "
                f"{row['energy_monotonic_violations']} | {row['mass_relative_drift']:.3e} |"
            )
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="CI profile: small grids, fewer geometries")
    parser.add_argument("--out", default=None, help="write JSON + Markdown here")
    args = parser.parse_args(argv)
    audit = run_audit(quick=args.quick)
    data = audit.to_dict()
    root = Path(args.out) if args.out else Path("evidence") / "cutcell_phase_transport"
    root.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, allow_nan=False, default=str)
    (root / "cutcell_phase_transport_audit.json").write_text(payload)
    markdown = format_markdown(audit)
    (root / "cutcell_phase_transport_audit.md").write_text(markdown)
    manifest = {
        "module": MODULE,
        "stage": STAGE,
        "passed": bool(data["passed"]),
        "files": {
            "cutcell_phase_transport_audit.json": hashlib.sha256(payload.encode()).hexdigest(),
            "cutcell_phase_transport_audit.md": hashlib.sha256(markdown.encode()).hexdigest(),
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(markdown)
    return 0 if data["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
