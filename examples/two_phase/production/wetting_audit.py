"""Thermodynamic audits for the diffuse wall wetting model (L1A-2b).

Four independent checks, all runnable without a trajectory:

1. **measurement** -- the contract-v6 contact-angle measurement is validated on
   synthetic circular caps of known geometric angle (and the legacy area/width
   measurement is quantified on the same caps, so its bias is on the record).
2. **wall delta** -- the normalized diffuse surface delta integrates to one across
   the wall at N = 64 / 96 / 128 and carries no periodic-y ghost.  The mutation
   check re-runs the *same* construction with a periodic y-derivative to prove the
   check is not vacuous (that variant does produce a top ghost).
3. **wall energy** -- Young endpoint difference, exact 90 deg neutrality,
   hydrophilic/hydrophobic signs, and a float64 directional-derivative audit
   proving ``mu_wall = dF_wall/dphi`` (central difference vs ``sum(mu_wall dphi)``).
4. **dispatch** -- an unknown ``wetting_model`` fails closed in the solver.

    python -m production.wetting_audit --json artifacts/wetting_audit.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The model as documented in ``phasefield.py``; transcribed here on purpose so the
#: audit fails if the solver implementation drifts from the documented physics.
WALL_SIGMA0 = math.sqrt(2.0) / 6.0
MEASUREMENT_TOLERANCES = {"mae_deg": 2.0, "max_error_deg": 3.0}
WALL_DELTA_NORMAL_INTEGRAL_TOLERANCE = 0.03
GHOST_TO_PHYSICAL_LIMIT = 1.0e-3
YOUNG_RELATIVE_TOLERANCE = 1.0e-10
NEUTRAL_ABSOLUTE_TOLERANCE = 1.0e-14
DIRECTIONAL_DERIVATIVE_TOLERANCE = 1.0e-4

CONVENTIONS = {
    "surface_tension": "sigma_0 = sqrt(2)/6 from f(phi) = phi^2 (1-phi)^2; Korteweg force scaled by 1/sigma_0",
    "wall_energy": "g_w(phi, theta) = -sigma_0 cos(theta) h(phi), h(phi) = phi^2 (3 - 2 phi)",
    "young_relation": "gamma_SG - gamma_SL = g_w(0) - g_w(1) = sigma_0 cos(theta_e)",
    "wall_delta": "delta_wall(sdf) = (1/2a) sech^2(sdf/a) |grad sdf|, a = 1.5 dx (legacy/diagnostic since v8)",
    "production_wall_measure": "A_wall,i = marching-squares length of sdf = 0 assigned to fluid-side control cells",
    "production_wall_energy": "F_wall^h = sum_i A_wall,i g_w(phi_i, theta_i); dA/dV = A_wall,i/(dx dy) in the flux",
    "chemical_potential": "mu_wall = dg_w/dphi * delta_wall = -sigma_0 cos(theta) 6 phi (1-phi) delta_wall",
    "neutral_angle": "theta = 90 deg gives g_w == 0 and mu_wall == 0 identically",
}


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": bool(self.passed), "detail": self.detail, "value": self.value}


@dataclass
class WettingAudit:
    checks: list[Check] = field(default_factory=list)
    numbers: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_schema_version": 1,
            "audit": "wetting",
            "settings": self.settings,
            "conventions": dict(CONVENTIONS),
            "checks": [check.to_dict() for check in self.checks],
            "numbers": self.numbers,
            "passed": all(check.passed for check in self.checks),
        }


def _wall_energy_density_reference(phi, cos_theta):
    """Independent transcription of g_w(phi, theta) (float64 numpy)."""
    import numpy as np

    return -WALL_SIGMA0 * np.asarray(cos_theta) * (np.asarray(phi) ** 2 * (3.0 - 2.0 * np.asarray(phi)))


def _wall_delta_periodic_y(sdf, dx, dy, a):
    """Mutation variant: periodic y-derivative -- documented to create a seam ghost."""
    import numpy as np

    gx = (np.roll(sdf, -1, axis=0) - np.roll(sdf, 1, axis=0)) / (2.0 * dx)
    gy = (np.roll(sdf, -1, axis=1) - np.roll(sdf, 1, axis=1)) / (2.0 * dy)
    grad = np.sqrt(gx**2 + gy**2 + 1e-30)
    return (1.0 / (2.0 * a)) * (1.0 - np.tanh(np.clip(sdf / a, -60.0, 60.0)) ** 2) * grad


def audit_wall_delta(N_values=(64, 96, 128), wall_height: float = 0.25, width_factor: float = 1.5):
    """Normalized normal integral, localization and periodic-y ghost check of the wall delta.

    Three numbers per grid:

    * ``normal_integral`` -- integral of delta across the wall; must be 1 (this is the
      quantity the periodic-y seam ghost used to inflate by 13 % at N = 128);
    * ``far_field_to_peak`` -- largest |delta| further than six kernel widths from the
      wall, relative to the peak: the "no material top ghost" localization test;
    * ``periodic_y_variant_*`` -- the same construction with a *periodic* y-derivative,
      kept as a mutation check so the ghost test cannot pass vacuously.
    """
    import numpy as np
    import phasefield as pf

    rows: dict[int, dict[str, float]] = {}
    for N in N_values:
        p = pf.PhaseFieldParams(Nx=N, Ny=N, Lx=6.0, Ly=6.0, dtype=pf.jnp.float64)
        sdf = pf.surface_flat(p, wall_height=wall_height)
        a = width_factor * p.dx
        delta = np.asarray(pf.wall_delta(sdf, p), dtype=np.float64)
        column = delta[N // 2, :]
        integral = float(column.sum() * p.dy)
        peak = float(np.abs(delta).max())
        distance = np.asarray(sdf, dtype=np.float64)
        far = np.abs(distance) > 6.0 * a
        far_ratio = float(np.abs(delta[far]).max() / max(peak, 1e-300)) if far.any() else 0.0
        mutated = np.asarray(_wall_delta_periodic_y(np.asarray(sdf), p.dx, p.dy, a), dtype=np.float64)
        mutated_integral = float(mutated[N // 2, :].sum() * p.dy)
        rows[int(N)] = {
            "normal_integral": integral,
            "normal_integral_relative_error": abs(integral - 1.0),
            "far_field_to_peak": far_ratio,
            "periodic_y_variant_normal_integral": mutated_integral,
            "periodic_y_variant_relative_error": abs(mutated_integral - 1.0),
            "periodic_y_variant_seam_excess_to_peak": float(np.abs(mutated - delta).max() / max(peak, 1e-300)),
        }
    integrals = [rows[N]["normal_integral_relative_error"] for N in sorted(rows)]
    converged = all(b <= a for a, b in zip(integrals, integrals[1:]))
    mutation_detected = any(
        rows[N]["periodic_y_variant_relative_error"] > WALL_DELTA_NORMAL_INTEGRAL_TOLERANCE for N in sorted(rows)
    )
    checks = [
        Check(
            "wall_delta_normal_integral",
            all(error <= WALL_DELTA_NORMAL_INTEGRAL_TOLERANCE for error in integrals) and converged,
            f"normal integral of delta_wall is 1 within {100 * WALL_DELTA_NORMAL_INTEGRAL_TOLERANCE:.0f}% at "
            f"N={sorted(rows)} and the error decreases with N",
            rows,
        ),
        Check(
            "wall_delta_no_periodic_y_ghost",
            all(rows[N]["far_field_to_peak"] < GHOST_TO_PHYSICAL_LIMIT for N in sorted(rows)) and mutation_detected,
            "no material far-field/top ghost, and the periodic-y mutation is detected by the integral test",
            rows,
        ),
    ]
    return checks, {"wall_delta": rows}


def audit_measurement(
    N: int = 128, targets=(60.0, 90.0, 120.0, 150.0), R: float = 1.1, wall_height: float = 0.25, eps_factor: float = 2.0
):
    """Synthetic circular-cap audit of the contact-angle measurement."""
    import numpy as np
    import phasefield as pf

    p = pf.PhaseFieldParams(Nx=N, Ny=N, Lx=6.0, Ly=6.0, dtype=pf.jnp.float32)
    p.eps = float(eps_factor) * p.dx
    sdf = pf.surface_flat(p, wall_height=wall_height)
    solid = pf.make_solid(sdf, p, cos_theta=0.0)
    X, Y = pf.grids(p)
    rows = []
    for target in targets:
        theta = math.radians(float(target))
        y_c = wall_height - R * math.cos(theta)
        r = pf.jnp.sqrt((X - 0.5 * p.Lx) ** 2 + (Y - y_c) ** 2)
        phi = 0.5 * (1.0 - pf.jnp.tanh((r - R) / (math.sqrt(2.0) * p.eps)))
        phi = pf.jnp.where(sdf >= 0.0, phi, 0.0)
        rows.append(
            {
                "target_deg": float(target),
                "measured_deg": float(pf.measure_contact_angle(phi, solid, p)),
                "measured_deg_legacy": float(pf.measure_contact_angle_area_width(phi, solid, p)),
            }
        )
    for row in rows:
        row["error_deg"] = row["measured_deg"] - row["target_deg"]
        row["error_deg_legacy"] = row["measured_deg_legacy"] - row["target_deg"]
    errors = np.abs([row["error_deg"] for row in rows])
    legacy_errors = np.abs([row["error_deg_legacy"] for row in rows])
    measured = [row["measured_deg"] for row in rows]
    monotonic = all(b >= a for a, b in zip(measured, measured[1:]))
    finite = all(math.isfinite(value) for value in measured)
    numbers = {
        "cases": rows,
        "mae_deg": float(errors.mean()),
        "max_error_deg": float(errors.max()),
        "monotonic": bool(monotonic),
        "all_finite": bool(finite),
        "legacy_mae_deg": float(legacy_errors.mean()),
        "legacy_max_error_deg": float(legacy_errors.max()),
        "N": int(N),
    }
    checks = [
        Check(
            "synthetic_measurement_accuracy",
            bool(
                finite
                and monotonic
                and errors.mean() <= MEASUREMENT_TOLERANCES["mae_deg"]
                and errors.max() <= MEASUREMENT_TOLERANCES["max_error_deg"]
            ),
            f"synthetic cap MAE <= {MEASUREMENT_TOLERANCES['mae_deg']} deg, max <= "
            f"{MEASUREMENT_TOLERANCES['max_error_deg']} deg, monotonic, finite",
            numbers,
        )
    ]
    return checks, numbers


def audit_wall_energy(
    N: int = 128,
    targets=(60.0, 90.0, 120.0, 150.0),
    wall_height: float = 0.25,
    eps_factor: float = 2.0,
    amplitudes=(1.0e-4, 3.0e-4),
):
    """Young endpoints, 90 deg neutrality, signs and the variational derivative of F_wall."""
    import numpy as np
    import phasefield as pf

    p = pf.PhaseFieldParams(
        Nx=N,
        Ny=N,
        Lx=6.0,
        Ly=6.0,
        dtype=pf.jnp.float64,
        wetting_model="surface_energy_volume_v6",
    )
    p.eps = float(eps_factor) * p.dx
    sdf = pf.surface_flat(p, wall_height=wall_height)

    endpoint_rows = []
    for target in targets:
        cos_theta = float(math.cos(math.radians(float(target))))
        g0 = float(pf.wall_energy_density(pf.jnp.asarray(0.0, dtype=p.dtype), cos_theta))
        g1 = float(pf.wall_energy_density(pf.jnp.asarray(1.0, dtype=p.dtype), cos_theta))
        endpoint_rows.append(
            {
                "target_deg": float(target),
                "g_wall_0": g0,
                "g_wall_1": g1,
                "gamma_SG_minus_SL": g0 - g1,
                "sigma0_cos_theta": WALL_SIGMA0 * cos_theta,
            }
        )
    endpoint_error = max(
        abs(row["gamma_SG_minus_SL"] - row["sigma0_cos_theta"]) / max(abs(row["sigma0_cos_theta"]), 1e-30)
        for row in endpoint_rows
    )

    # 90 deg neutrality on a realistic diffuse field, not only at the endpoints.
    state = pf.sessile_initial_state(p, pf.make_solid(sdf, p, cos_theta=0.0), R=1.1, wall_height=wall_height)
    neutral_solid = pf.make_solid(sdf, p, cos_theta=math.cos(math.radians(90.0)))
    neutral_g = float(pf.jnp.max(pf.jnp.abs(pf.wall_energy_density(state.phi, neutral_solid.cos_theta))))
    neutral_mu = float(pf.jnp.max(pf.jnp.abs(pf.wetting_mu(state.phi, neutral_solid, p))))
    neutral_endpoint = float(
        pf.jnp.max(
            pf.jnp.abs(
                pf.wall_energy_density(pf.jnp.linspace(0.0, 1.0, 33), pf.jnp.asarray(math.cos(math.radians(90.0))))
            )
        )
    )

    sign_rows = []
    phi_half = pf.jnp.full((N, N), 0.5, dtype=p.dtype)
    wall_cell = np.unravel_index(int(np.argmax(np.asarray(pf.wall_delta(sdf, p), dtype=np.float64))), (N, N))
    for target in (60.0, 90.0, 120.0, 150.0):
        cos_theta = float(math.cos(math.radians(target)))
        solid = pf.make_solid(sdf, p, cos_theta=cos_theta)
        mu_mid = float(pf.wetting_mu(phi_half, solid, p)[wall_cell])
        sign_rows.append({"target_deg": float(target), "cos_theta": cos_theta, "mu_wall_at_half": mu_mid})

    # Directional derivative of F_wall with an independent transcription of g_w.
    x = np.linspace(0.0, 6.0, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    phi0 = np.clip(0.5 + 0.35 * np.cos(2.0 * np.pi * X / 6.0) * np.cos(np.pi * Y / 6.0), 0.05, 0.95)
    direction = np.cos(3.0 * np.pi * X / 6.0) * np.sin(2.0 * np.pi * Y / 6.0)
    delta_direction = np.asarray(pf.wall_delta(sdf, p), dtype=np.float64)
    variational = []
    for target in (60.0, 120.0):
        cos_theta = math.cos(math.radians(target))
        solid = pf.make_solid(sdf, p, cos_theta=cos_theta)
        delta = np.asarray(pf.wall_delta(solid.sdf, p), dtype=np.float64)

        def free_energy(field: np.ndarray) -> float:
            return float(np.sum(_wall_energy_density_reference(field, cos_theta) * delta) * p.dx * p.dy)

        mu_wall = np.asarray(pf.wetting_mu(pf.jnp.asarray(phi0), solid, p), dtype=np.float64)
        for label, probe in (("wall_delta_aligned", delta_direction), ("smooth_mixed_mode", direction)):
            inner = float(np.sum(mu_wall * probe) * p.dx * p.dy)
            for amplitude in amplitudes:
                fd = (free_energy(phi0 + amplitude * probe) - free_energy(phi0 - amplitude * probe)) / (2.0 * amplitude)
                variational.append(
                    {
                        "target_deg": float(target),
                        "probe": label,
                        "amplitude": float(amplitude),
                        "finite_difference": float(fd),
                        "mu_inner_product": float(inner),
                        "relative_error": float(abs(fd - inner) / max(abs(inner), 1e-300)),
                    }
                )
    variational_error = max(row["relative_error"] for row in variational)

    checks = [
        Check(
            "wall_energy_young_endpoint_difference",
            endpoint_error <= YOUNG_RELATIVE_TOLERANCE,
            "g_w(0) - g_w(1) == sigma_0 cos(theta) for every target",
            {"max_relative_error": endpoint_error, "cases": endpoint_rows},
        ),
        Check(
            "wall_energy_90deg_is_neutral",
            max(neutral_g, neutral_mu, neutral_endpoint) <= NEUTRAL_ABSOLUTE_TOLERANCE,
            "cos(90 deg) = 0 gives g_w == 0 and mu_wall == 0 to round-off",
            {"max_abs_g_wall": neutral_g, "max_abs_mu_wall": neutral_mu, "max_abs_endpoint_g": neutral_endpoint},
        ),
        Check(
            "wall_energy_hydrophilic_hydrophobic_sign",
            all(
                (row["mu_wall_at_half"] < 0.0) if row["cos_theta"] > 1e-12 else (row["mu_wall_at_half"] > 0.0)
                for row in sign_rows
                if abs(row["cos_theta"]) > 1e-12
            ),
            "cos(theta) > 0 favours liquid at the wall (mu_wall < 0 at phi = 1/2) and cos(theta) < 0 the reverse",
            sign_rows,
        ),
        Check(
            "wall_energy_directional_derivative_matches_mu",
            variational_error <= DIRECTIONAL_DERIVATIVE_TOLERANCE,
            (
                "central difference of F_wall matches sum(mu_wall dphi) within "
                f"{DIRECTIONAL_DERIVATIVE_TOLERANCE:g} (float64; the aligned probe is FD-truncation limited, "
                "2e-9 to 6e-7 at the fine amplitude)"
            ),
            {"max_relative_error": variational_error, "cases": variational},
        ),
    ]
    return checks, {"wall_energy": {"endpoints": endpoint_rows, "variational": variational, "signs": sign_rows}}


def audit_production_wall_energy(
    N: int = 128, targets=(60.0, 90.0, 120.0, 150.0), wall_height: float = 0.25, eps_factor: float = 2.0
):
    """Contract-v8 production path: one wall contribution, on the embedded geometric measure.

    The ``wall_delta`` checks above audit the *legacy diffuse kernel*, which contract v8 keeps only
    for the pinned reproduction modes. This check audits what the production solver does now:

    * ``phase_free_energy`` equals an independently transcribed ``F_bulk + sum_i A_wall,i g_w(phi_i)``;
    * ``wetting_mu`` is exactly zero in production mode, so the Young energy is not added twice;
    * ``sum_i A_wall,i`` is the geometric wall length (no global gain, no fluid-share factor);
    * the wall part of ``chemical_potential`` equals ``A_wall,i g_w'(phi_i)/(dx dy)`` = dF_wall^h/dphi / V.

    The deeper geometry/translation/variational matrix lives in ``production/wall_measure_audit.py``.
    """
    import numpy as np
    import phasefield as pf

    p = pf.PhaseFieldParams(Nx=N, Ny=N, Lx=6.0, Ly=6.0, dtype=pf.jnp.float64)
    p.eps = float(eps_factor) * p.dx
    sdf = np.asarray(pf.surface_flat(p, wall_height=wall_height), dtype=np.float64)
    X, Y = pf.grids(p)
    fluid = sdf >= 0.0
    phi = np.where(fluid, np.clip(0.5 - 0.45 * np.tanh((sdf - 0.5) / (math.sqrt(2.0) * p.eps)), 0.0, 1.0), 0.0)
    rows: list[dict[str, Any]] = []
    for target in targets:
        cos_theta = math.cos(math.radians(float(target)))
        solid = pf.make_solid(pf.jnp.asarray(sdf), p, cos_theta=cos_theta)
        area = np.asarray(solid.wall_area, dtype=np.float64)
        phase = pf.jnp.asarray(phi)
        # independent transcription of F_bulk + F_wall^h
        aperture_x, aperture_y = pf.fluid_face_apertures(solid, p)
        grad_x = (np.roll(phi, -1, axis=0) - phi) / p.dx
        grad_y = (np.roll(phi, -1, axis=1) - phi) / p.dy
        bulk_density = np.sum(np.where(fluid, phi**2 * (1.0 - phi) ** 2 / p.eps, 0.0)) + 0.5 * p.eps * np.sum(
            np.asarray(aperture_x) * grad_x**2 + np.asarray(aperture_y) * grad_y**2
        )
        wall = float(np.sum(_wall_energy_density_reference(phi, cos_theta) * area))
        reference_energy = bulk_density * p.dx * p.dy + wall
        solver_energy = float(pf.phase_free_energy(phase, solid, p))
        mu = np.asarray(pf.chemical_potential(phase, solid, p), dtype=np.float64)
        bulk_mu = np.asarray(pf.fprime(phase) / p.eps - p.eps * pf.fluid_laplacian(phase, solid, p), dtype=np.float64)
        wall_mu_reference = _wall_energy_density_reference_derivative(phi, cos_theta) * area / (p.dx * p.dy)
        rows.append(
            {
                "target_deg": float(target),
                "energy_relative_error": abs(solver_energy - reference_energy) / max(abs(reference_energy), 1e-30),
                "wall_operator_max_absolute_error": float(np.max(np.abs(mu - bulk_mu - wall_mu_reference))),
                "wall_operator_scale": float(np.max(np.abs(wall_mu_reference))),
                "separate_wetting_mu_max_abs": float(np.max(np.abs(pf.wetting_mu(phase, solid, p)))),
                "measure_total": float(area.sum()),
                "measure_relative_error": abs(float(area.sum()) - p.Lx) / p.Lx,
                "n_wall_cells": int(np.count_nonzero(area > 0.0)),
            }
        )
    checks = [
        Check(
            "production_wall_energy_is_single_embedded_contribution",
            all(row["energy_relative_error"] <= 1.0e-12 for row in rows)
            and all(
                row["wall_operator_max_absolute_error"] <= 1.0e-12 * max(row["wall_operator_scale"], 1.0)
                for row in rows
            )
            and all(row["separate_wetting_mu_max_abs"] == 0.0 for row in rows)
            and all(row["measure_relative_error"] <= 1.0e-12 for row in rows),
            "phase_free_energy == transcribed F_bulk + sum_i A_wall,i g_w(phi_i); the wall part of "
            "chemical_potential == A_wall,i g_w'(phi_i)/(dx dy); wetting_mu is exactly zero (no double "
            "counting); sum_i A_wall,i == Lx exactly (no global gain or fluid-share factor)",
            rows,
        )
    ]
    return checks, {"production_wall_energy": rows}


def _wall_energy_density_reference_derivative(phi, cos_theta):
    """Independent transcription of g_w'(phi, theta) (float64 numpy)."""
    import numpy as np

    phase = np.asarray(phi, dtype=np.float64)
    return -WALL_SIGMA0 * np.asarray(cos_theta, dtype=np.float64) * 6.0 * phase * (1.0 - phase)


def audit_model_dispatch(N: int = 32):
    """``wetting_model`` must fail closed on unknown names and honour ``none``."""
    import numpy as np
    import phasefield as pf

    results: dict[str, Any] = {}
    failed_closed = False
    try:
        pf.PhaseFieldParams(Nx=N, Ny=N, Lx=6.0, Ly=6.0, wetting_model="definitely_not_a_model")
    except ValueError:
        failed_closed = True
    results["construction_failed_closed"] = failed_closed
    p = pf.PhaseFieldParams(Nx=N, Ny=N, Lx=6.0, Ly=6.0, wetting_model="none")
    solid = pf.make_solid(pf.surface_flat(p, wall_height=0.25), p, cos_theta=0.7)
    phi = pf.jnp.full((N, N), 0.4, dtype=p.dtype)
    results["none_mu_max_abs"] = float(pf.jnp.max(pf.jnp.abs(pf.wetting_mu(phi, solid, p))))
    results["none_is_zero"] = results["none_mu_max_abs"] == 0.0
    p_bad = pf.PhaseFieldParams(Nx=N, Ny=N, Lx=6.0, Ly=6.0)
    object.__setattr__(p_bad, "wetting_model", "shadow_model")
    runtime_failed_closed = False
    try:
        pf.wetting_mu(phi, solid, p_bad)
    except ValueError:
        runtime_failed_closed = True
    results["runtime_failed_closed"] = runtime_failed_closed
    checks = [
        Check(
            "wetting_model_fails_closed",
            failed_closed and runtime_failed_closed and results["none_is_zero"],
            "unknown wetting_model raises at construction and in wetting_mu; 'none' returns exact zeros",
            results,
        )
    ]
    return checks, {"dispatch": results}


def run_audit(N: int = 128, dtype_independent: bool = True) -> WettingAudit:
    """Run every wetting audit; ``N`` sets the resolution of the measurement/delta checks."""
    import phasefield as pf

    audit = WettingAudit()
    audit.settings = {
        "N": int(N),
        "wall_delta_width_factor": 1.5,
        "sigma_0": WALL_SIGMA0,
        "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
        "wall_measure_method": str(pf.WALL_MEASURE_METHOD),
        "wall_measure_contract_version": int(pf.WALL_MEASURE_CONTRACT_VERSION),
        "wall_delta_status": (
            "legacy/diagnostic kernel since contract v8; the production measure is the exact cut-cell wall area "
            "(see production/wall_measure_audit.py)"
        ),
        "tolerances": {
            "measurement": MEASUREMENT_TOLERANCES,
            "wall_delta_normal_integral": WALL_DELTA_NORMAL_INTEGRAL_TOLERANCE,
            "ghost_to_physical": GHOST_TO_PHYSICAL_LIMIT,
            "young_relative": YOUNG_RELATIVE_TOLERANCE,
            "neutral_absolute": NEUTRAL_ABSOLUTE_TOLERANCE,
            "directional_derivative": DIRECTIONAL_DERIVATIVE_TOLERANCE,
        },
    }
    measurement_checks, measurement_numbers = audit_measurement(N=N)
    delta_checks, delta_numbers = audit_wall_delta()
    energy_checks, energy_numbers = audit_wall_energy(N=N)
    dispatch_checks, dispatch_numbers = audit_model_dispatch()
    production_checks, production_numbers = audit_production_wall_energy(N=N)
    audit.checks = [*measurement_checks, *delta_checks, *energy_checks, *production_checks, *dispatch_checks]
    audit.numbers = {
        **measurement_numbers,
        **delta_numbers,
        **energy_numbers,
        **production_numbers,
        **dispatch_numbers,
    }
    layout = audit.numbers.pop("cases", None)
    if layout is not None:
        audit.numbers["synthetic_cases"] = layout
    return audit


def format_markdown(audit: WettingAudit) -> str:
    lines = ["# L1A-2b wetting audit (wall measure updated by L1A-2e)", ""]
    lines.append(f"- N = {audit.settings['N']}, sigma_0 = {audit.settings['sigma_0']:.12f}")
    lines.append("")
    lines.append("| check | result | detail |")
    lines.append("| --- | --- | --- |")
    for check in audit.checks:
        lines.append(f"| {check.name} | {'PASS' if check.passed else 'FAIL'} | {check.detail} |")
    lines.append("")
    cases = audit.numbers.get("synthetic_cases", [])
    if cases:
        lines.append("| synthetic target | measured | error | legacy measured | legacy error |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in cases:
            lines.append(
                f"| {row['target_deg']:.0f} | {row['measured_deg']:.3f} | {row['error_deg']:+.3f} | "
                f"{row['measured_deg_legacy']:.3f} | {row['error_deg_legacy']:+.3f} |"
            )
        lines.append("")
        lines.append(
            f"measurement MAE = {audit.numbers['mae_deg']:.4f} deg (legacy {audit.numbers['legacy_mae_deg']:.4f} deg)"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diffuse wall-wetting thermodynamic audit (L1A-2b)")
    parser.add_argument("--json", type=str, default=None, help="write the machine-readable audit here")
    parser.add_argument("--markdown", type=str, default=None, help="write a markdown summary here")
    parser.add_argument("--N", type=int, default=128)
    args = parser.parse_args(argv)
    import jax

    jax.config.update("jax_enable_x64", True)
    audit = run_audit(N=args.N)
    payload = audit.to_dict()
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    markdown = format_markdown(audit)
    if args.markdown:
        path = Path(args.markdown)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    print(f"wetting audit: {'PASS' if payload['passed'] else 'FAIL'}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
