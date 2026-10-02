"""Contact-line kinetics, Young boundary residuals, and equilibration classification (L1A-2d).

This module provides solver-independent, NumPy-first observables and analysis
routines used to isolate why non-neutral sessile droplets (60/120/150 deg) do
not equilibrate to their Young target angles within 10,000 steps under solver
contract v7:

* ``young_boundary_residual`` -- evaluates ``R_Y = eps * dphi/dn + g_w'(phi)``
  strictly in the contact-line neighborhood (``|sdf| <= 2 dx``,
  ``0.05 <= phi <= 0.95``) using fluid-domain second-order finite differences;
* ``manufactured_young_boundary_field`` / ``audit_young_boundary_residual`` --
  analytic diffuse contact-line fields satisfying the natural Young BC to verify
  that the boundary-residual diagnostic converges at ``O((dx/eps)^2)``;
* ``contact_line_positions`` / ``contact_line_velocity`` -- periodic-safe
  extraction of left/right contact-line positions, contact width, speeds, and
  detachment / topology-change flags;
* ``late_time_linear_trend`` / ``fit_relaxation_asymptote`` -- late-time drift
  slopes and diagnostic-only exponential relaxation fits (never treated as an
  equilibrium angle);
* ``evaluate_mobility_collapse`` / ``classify_equilibration`` -- falsifiable
  classification into the frozen L1A-2d mechanism categories.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from production import observables as obs

WALL_SIGMA0 = math.sqrt(2.0) / 6.0

ALLOWED_CLASSIFICATIONS = (
    "KINETICS_LIMITED",
    "THERMODYNAMIC_EQUILIBRIUM_BIASED",
    "HYDRODYNAMIC_COUPLING_LIMITED",
    "BOUNDARY_DISCRETIZATION_LIMITED",
    "RESOLUTION_LIMITED",
    "TIME_STEP_LIMITED",
    "TOPOLOGY_CHANGE_OR_DETACHMENT",
    "INCONCLUSIVE",
)


def wall_switch_derivative_np(phi: Any) -> np.ndarray:
    """Derivative ``h'(phi) = 6 phi (1 - phi)`` of ``h(phi) = phi^2 (3 - 2 phi)``."""
    arr = np.asarray(phi, dtype=np.float64)
    return 6.0 * arr * (1.0 - arr)


def wall_energy_derivative_np(phi: Any, cos_theta: Any) -> np.ndarray:
    """Young wall free-energy derivative ``g_w'(phi) = -sigma_0 cos(theta) h'(phi)``."""
    return -WALL_SIGMA0 * np.asarray(cos_theta, dtype=np.float64) * wall_switch_derivative_np(phi)


def _ddy_nonperiodic_np(field: np.ndarray, dy: float) -> np.ndarray:
    """Second-order interior central y-derivative with one-sided domain edges."""
    out = np.empty_like(field, dtype=np.float64)
    out[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2.0 * dy)
    out[:, 0] = (field[:, 1] - field[:, 0]) / dy
    out[:, -1] = (field[:, -1] - field[:, -2]) / dy
    return out


def fluid_outward_normal_np(sdf: Any, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Unit normal pointing from fluid (``sdf >= 0``) into solid (``sdf < 0``)."""
    distance = np.asarray(sdf, dtype=np.float64)
    gx = (np.roll(distance, -1, axis=0) - np.roll(distance, 1, axis=0)) / (2.0 * dx)
    gy = _ddy_nonperiodic_np(distance, dy)
    norm = np.sqrt(gx * gx + gy * gy + 1e-30)
    return -gx / norm, -gy / norm


def fluid_wall_delta_integral(
    sdf: Any,
    dx: float,
    dy: float,
    width: float | None = None,
) -> dict[str, float]:
    """Quantify the full-domain vs fluid-half-space normal integral of ``wall_delta``.

    In contract v7 (``phase_boundary_model='impermeable_flux'``), ``phi`` is
    evolved only on ``sdf >= 0`` while ``wall_delta(sdf)`` is a symmetric
    two-sided kernel centered on ``sdf = 0``. Consequently only the ``sdf >= 0``
    half of the kernel acts on the fluid phase field.
    """
    distance = np.asarray(sdf, dtype=np.float64)
    if distance.ndim != 2:
        raise ValueError("sdf must be a 2-D array")
    a = float(1.5 * dx if width is None else width)
    if not (math.isfinite(dx) and math.isfinite(dy) and math.isfinite(a) and dx > 0 and dy > 0 and a > 0):
        raise ValueError("dx, dy, and width must be finite and positive")
    gx = (np.roll(distance, -1, axis=0) - np.roll(distance, 1, axis=0)) / (2.0 * dx)
    gy = _ddy_nonperiodic_np(distance, dy)
    grad = np.sqrt(gx * gx + gy * gy + 1e-30)
    scaled = np.clip(distance / a, -60.0, 60.0)
    delta = (1.0 / (2.0 * a)) * (1.0 - np.tanh(scaled) ** 2) * grad

    col_idx = distance.shape[0] // 2
    col_delta = delta[col_idx, :]
    col_sdf = distance[col_idx, :]
    total_integral = float(np.sum(col_delta) * dy)
    fluid_integral = float(np.sum(col_delta[col_sdf >= 0.0]) * dy)
    solid_integral = float(np.sum(col_delta[col_sdf < 0.0]) * dy)
    return {
        "total_normal_integral": total_integral,
        "fluid_side_normal_integral": fluid_integral,
        "solid_side_normal_integral": solid_integral,
        "fluid_fraction_of_wall_kernel": fluid_integral / max(total_integral, 1e-30),
    }


def _fluid_aware_gradients(phi: np.ndarray, fluid: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Second-order spatial derivatives of ``phi`` using only fluid cells (``sdf >= 0``).

    When ``phi`` is clipped to zero inside the solid (``sdf < 0``), a standard
    central difference at the first fluid cell above the wall would differentiate
    across the artificial clip. Where a neighbor lies in ``sdf < 0``, this
    stencil switches to the three-point second-order one-sided difference into
    the fluid domain, preserving ``O(dx^2, dy^2)`` accuracy up to the wall.
    """
    nx, ny = phi.shape
    # x-gradient (periodic domain)
    phi_xp1 = np.roll(phi, -1, axis=0)
    phi_xp2 = np.roll(phi, -2, axis=0)
    phi_xm1 = np.roll(phi, 1, axis=0)
    phi_xm2 = np.roll(phi, 2, axis=0)
    fl_xp1 = np.roll(fluid, -1, axis=0)
    fl_xp2 = np.roll(fluid, -2, axis=0)
    fl_xm1 = np.roll(fluid, 1, axis=0)
    fl_xm2 = np.roll(fluid, 2, axis=0)

    gx = (phi_xp1 - phi_xm1) / (2.0 * dx)
    fwd_x = fluid & (~fl_xm1) & fl_xp1 & fl_xp2
    bwd_x = fluid & (~fl_xp1) & fl_xm1 & fl_xm2
    gx = np.where(fwd_x, (-3.0 * phi + 4.0 * phi_xp1 - phi_xp2) / (2.0 * dx), gx)
    gx = np.where(bwd_x, (3.0 * phi - 4.0 * phi_xm1 + phi_xm2) / (2.0 * dx), gx)

    # y-gradient (non-periodic slab domain)
    gy = _ddy_nonperiodic_np(phi, dy)
    for j in range(ny):
        if j + 2 < ny:
            left_solid = np.ones(nx, dtype=bool) if j == 0 else (~fluid[:, j - 1])
            mask_fwd = fluid[:, j] & left_solid & fluid[:, j + 1] & fluid[:, j + 2]
            gy[:, j] = np.where(
                mask_fwd,
                (-3.0 * phi[:, j] + 4.0 * phi[:, j + 1] - phi[:, j + 2]) / (2.0 * dy),
                gy[:, j],
            )
        if j - 2 >= 0:
            right_solid = np.ones(nx, dtype=bool) if j == ny - 1 else (~fluid[:, j + 1])
            mask_bwd = fluid[:, j] & right_solid & fluid[:, j - 1] & fluid[:, j - 2]
            gy[:, j] = np.where(
                mask_bwd,
                (3.0 * phi[:, j] - 4.0 * phi[:, j - 1] + phi[:, j - 2]) / (2.0 * dy),
                gy[:, j],
            )
    return gx, gy


def young_boundary_residual(
    phi: Any,
    sdf: Any,
    dx: float,
    dy: float,
    eps: float,
    cos_theta: Any,
    *,
    sdf_band_cells: float = 2.0,
    phi_min: float = 0.05,
    phi_max: float = 0.95,
    tiny: float = 1.0e-10,
) -> dict[str, Any]:
    """Evaluate the Young boundary residual ``R_Y = eps * dphi/dn + g_w'(phi)``.

    Evaluated strictly in the contact-line neighborhood:
    ``0 <= sdf <= sdf_band_cells * dx`` (within ``|sdf| <= 2 dx`` on the fluid
    side) and ``phi_min <= phi <= phi_max``. Never averaged over the bulk domain.
    """
    phase = np.asarray(phi, dtype=np.float64)
    distance = np.asarray(sdf, dtype=np.float64)
    if phase.ndim != 2 or phase.shape != distance.shape:
        raise ValueError("phi and sdf must be matching 2-D arrays")
    if not (math.isfinite(dx) and math.isfinite(dy) and math.isfinite(eps) and dx > 0 and dy > 0 and eps > 0):
        raise ValueError("dx, dy, and eps must be finite and positive")
    if not (0.0 < phi_min < phi_max < 1.0 and sdf_band_cells > 0 and tiny > 0):
        raise ValueError("invalid contact-line neighborhood bounds or tiny")

    fluid = distance >= 0.0
    gx, gy = _fluid_aware_gradients(phase, fluid, dx, dy)
    nx, ny = fluid_outward_normal_np(distance, dx, dy)
    dphi_dn = nx * gx + ny * gy
    eps_dphi_dn = float(eps) * dphi_dn
    gw_prime = wall_energy_derivative_np(phase, cos_theta)
    residual = eps_dphi_dn + gw_prime

    band_width = float(sdf_band_cells) * float(dx)
    mask = fluid & (np.abs(distance) <= band_width) & (phase >= float(phi_min)) & (phase <= float(phi_max))
    n_points = int(np.sum(mask))
    if n_points == 0:
        return {
            "RY_l2": 0.0,
            "RY_linf": 0.0,
            "RY_normalized_l2": 0.0,
            "RY_normalized_linf": 0.0,
            "RY_over_sigma0_l2": 0.0,
            "n_points": 0,
            "mean_eps_dphi_dn": 0.0,
            "mean_gw_prime": 0.0,
        }

    res_pts = residual[mask]
    term_normal = eps_dphi_dn[mask]
    term_wall = gw_prime[mask]

    ry_l2 = float(np.sqrt(np.mean(res_pts**2)))
    ry_linf = float(np.max(np.abs(res_pts)))
    denom_l2 = float(np.sqrt(np.mean(term_normal**2)) + np.sqrt(np.mean(term_wall**2)) + float(tiny))
    denom_linf = float(np.max(np.abs(term_normal)) + np.max(np.abs(term_wall)) + float(tiny))
    ry_norm_l2 = float(ry_l2 / denom_l2)
    ry_norm_linf = float(ry_linf / denom_linf)

    return {
        "RY_l2": ry_l2,
        "RY_linf": ry_linf,
        "RY_normalized_l2": ry_norm_l2,
        "RY_normalized_linf": ry_norm_linf,
        "RY_over_sigma0_l2": float(ry_l2 / WALL_SIGMA0),
        "n_points": n_points,
        "mean_eps_dphi_dn": float(np.mean(term_normal)),
        "mean_gw_prime": float(np.mean(term_wall)),
    }


def young_boundary_residual_first_layer(
    phi: Any,
    sdf: Any,
    dx: float,
    dy: float,
    eps: float,
    cos_theta: Any,
    *,
    wall_area: Any,
    wall_normal_x: Any,
    wall_normal_y: Any,
    phi_min: float = 0.05,
    phi_max: float = 0.95,
    area_tolerance: float = 0.0,
    tiny: float = 1.0e-10,
) -> dict[str, Any]:
    """First-fluid-layer Young residual, weighted by the embedded wall measure (L1A-2e).

    ``R_Y = eps * dphi/dn + g_w'(phi)`` evaluated *only* on the wall-adjacent control
    cells that carry the production wall flux (``A_wall,i > area_tolerance``), with
    ``n`` the area-weighted cut-cell normal and ``dphi/dn`` from the fluid-aware
    second-order stencils of :func:`_fluid_aware_gradients`. The primary norms are
    restricted to the contact-line band ``phi_min <= phi <= phi_max``; the
    measure-weighted mean over *all* wall cells is reported separately, because away
    from the contact line both terms of ``R_Y`` must vanish individually.

    Unlike the two-cell-band residual of :func:`young_boundary_residual` (kept for
    comparison), this metric samples exactly the cells where the operator imposes the
    condition, so a manufactured field satisfying the natural BC drives it to the
    finite-difference truncation error instead of to an O(eps)-wide band average.
    """
    phase = np.asarray(phi, dtype=np.float64)
    distance = np.asarray(sdf, dtype=np.float64)
    area = np.asarray(wall_area, dtype=np.float64)
    normal_x = np.asarray(wall_normal_x, dtype=np.float64)
    normal_y = np.asarray(wall_normal_y, dtype=np.float64)
    for name, array in (("phi", phase), ("sdf", distance), ("wall_area", area)):
        if array.ndim != 2 or array.shape != phase.shape:
            raise ValueError(f"{name} must match phi shape {phase.shape}; got {array.shape}")
    if normal_x.shape != phase.shape or normal_y.shape != phase.shape:
        raise ValueError("wall normals must match the phi shape")
    if not (math.isfinite(dx) and math.isfinite(dy) and math.isfinite(eps) and dx > 0 and dy > 0 and eps > 0):
        raise ValueError("dx, dy, and eps must be finite and positive")
    if not (0.0 < phi_min < phi_max < 1.0 and tiny > 0 and area_tolerance >= 0.0):
        raise ValueError("invalid contact-line band, area tolerance, or tiny")

    wall_cells = area > float(area_tolerance)
    band = wall_cells & (phase >= float(phi_min)) & (phase <= float(phi_max))
    n_wall_cells = int(np.count_nonzero(wall_cells))
    n_band = int(np.count_nonzero(band))
    empty = {
        "RY_first_l2": 0.0,
        "RY_first_linf": 0.0,
        "RY_first_normalized_l2": 0.0,
        "RY_first_normalized_linf": 0.0,
        "RY_first_over_sigma0_l2": 0.0,
        "wall_measure_weighted_RY": 0.0,
        "wall_measure_weighted_RY_all_wall": 0.0,
        "n_wall_cells": n_wall_cells,
        "n_wall_cells_band": n_band,
        "n_points": n_band,
        "n_wall_cells_nonfluid": int(np.count_nonzero(wall_cells & (distance < 0.0))),
        "wall_area_total": float(np.sum(area)),
        "wall_area_in_band": float(np.sum(area[band])),
        "mean_eps_dphi_dn": 0.0,
        "mean_gw_prime": 0.0,
    }
    if n_wall_cells == 0:
        return empty

    fluid = distance >= 0.0
    grad_x, grad_y = _fluid_aware_gradients(phase, fluid, dx, dy)
    dphi_dn = normal_x * grad_x + normal_y * grad_y
    term_normal = float(eps) * dphi_dn
    term_wall = wall_energy_derivative_np(phase, cos_theta)
    residual = term_normal + term_wall

    area_all = float(np.sum(area[wall_cells]))
    weighted_all = float(np.sum(area[wall_cells] * np.abs(residual[wall_cells])) / max(area_all, tiny))
    if n_band == 0:
        out = dict(empty)
        out["wall_measure_weighted_RY_all_wall"] = weighted_all
        return out

    res = residual[band]
    weights = area[band]
    normal_band = term_normal[band]
    wall_band = term_wall[band]
    ry_l2 = float(np.sqrt(np.mean(res**2)))
    ry_linf = float(np.max(np.abs(res)))
    denom_l2 = float(np.sqrt(np.mean(normal_band**2)) + np.sqrt(np.mean(wall_band**2)) + tiny)
    denom_linf = float(np.max(np.abs(normal_band)) + np.max(np.abs(wall_band)) + tiny)
    return {
        "RY_first_l2": ry_l2,
        "RY_first_linf": ry_linf,
        "RY_first_normalized_l2": float(ry_l2 / denom_l2),
        "RY_first_normalized_linf": float(ry_linf / denom_linf),
        "RY_first_over_sigma0_l2": float(ry_l2 / WALL_SIGMA0),
        "wall_measure_weighted_RY": float(np.sum(weights * np.abs(res)) / max(float(np.sum(weights)), tiny)),
        "wall_measure_weighted_RY_all_wall": weighted_all,
        "n_wall_cells": n_wall_cells,
        "n_wall_cells_band": n_band,
        "n_points": n_band,
        "n_wall_cells_nonfluid": int(np.count_nonzero(wall_cells & (distance < 0.0))),
        "wall_area_total": area_all,
        "wall_area_in_band": float(np.sum(weights)),
        "mean_eps_dphi_dn": float(np.mean(normal_band)),
        "mean_gw_prime": float(np.mean(wall_band)),
    }


def manufactured_young_boundary_field(
    x: Any,
    y: Any,
    sdf: Any,
    eps: float,
    theta_deg: float,
    *,
    x0: float = 3.0,
    half_width: float = 1.1,
    Lx: float = 6.0,
    mask_solid: bool = True,
) -> np.ndarray:
    """Construct a 2-D diffuse contact-line field satisfying ``eps dphi/dn + g_w'(phi) == 0``.

    Because ``h'(phi) = 6 phi (1 - phi)`` and ``sigma_0 = sqrt(2)/6``, any profile
    of the form ``phi(x, s) = 0.5 * (1 - tanh((d_x(x) + s * cos(theta)) / (sqrt(2) * eps)))``
    with ``s = sdf`` satisfies ``eps * (-dphi/ds) + g_w'(phi) = 0`` identically at
    every point ``(x, s)``.
    """
    distance = np.asarray(sdf, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim == 1:
        x_grid = np.broadcast_to(x_arr[:, None], distance.shape)
    else:
        x_grid = x_arr
    del y  # normal coordinate enters through sdf
    cos_theta = math.cos(math.radians(float(theta_deg)))
    rel_x = ((x_grid - float(x0) + 0.5 * float(Lx)) % float(Lx)) - 0.5 * float(Lx)
    # Smooth periodic footprint with two contact lines at x0 +- half_width
    dx_profile = np.sqrt(rel_x**2 + (0.25 * float(half_width)) ** 2) - float(half_width)
    xi = (dx_profile + distance * cos_theta) / (math.sqrt(2.0) * float(eps))
    phi = 0.5 * (1.0 - np.tanh(xi))
    if mask_solid:
        phi = np.where(distance >= 0.0, phi, 0.0)
    return phi


def audit_young_boundary_residual(
    N_values: Sequence[int] = (64, 128, 256),
    targets: Sequence[float] = (60.0, 90.0, 120.0, 150.0),
    *,
    Lx: float = 6.0,
    Ly: float = 6.0,
    eps_factor: float = 2.0,
    wall_height: float = 0.25,
) -> dict[str, Any]:
    """Validate ``young_boundary_residual`` on manufactured fields before trajectory use."""
    rows: list[dict[str, Any]] = []
    for N in N_values:
        dx = float(Lx) / int(N)
        dy = float(Ly) / int(N)
        eps = float(eps_factor) * dx
        x = (np.arange(N, dtype=np.float64) + 0.5) * dx
        y = (np.arange(N, dtype=np.float64) + 0.5) * dy
        X, Y = np.meshgrid(x, y, indexing="ij")
        sdf = Y - float(wall_height)
        for target in targets:
            cos_theta = math.cos(math.radians(float(target)))
            phi = manufactured_young_boundary_field(X, Y, sdf, eps, float(target), Lx=Lx, mask_solid=True)
            res = young_boundary_residual(phi, sdf, dx, dy, eps, cos_theta)
            # Also evaluate against a deliberately mismatched wall angle to prove non-vacuity
            wrong_theta = 150.0 if float(target) <= 90.0 else 60.0
            wrong_res = young_boundary_residual(phi, sdf, dx, dy, eps, math.cos(math.radians(wrong_theta)))
            rows.append(
                {
                    "N": int(N),
                    "dx": dx,
                    "eps": eps,
                    "target_deg": float(target),
                    "RY_l2": res["RY_l2"],
                    "RY_linf": res["RY_linf"],
                    "RY_normalized_l2": res["RY_normalized_l2"],
                    "RY_normalized_linf": res["RY_normalized_linf"],
                    "n_points": res["n_points"],
                    "mismatched_wall_theta_deg": wrong_theta,
                    "mismatched_RY_normalized_l2": wrong_res["RY_normalized_l2"],
                }
            )

    baseline_rows = [r for r in rows if r["N"] == 128] if any(r["N"] == 128 for r in rows) else rows
    max_norm_l2_at_baseline = max((r["RY_normalized_l2"] for r in baseline_rows), default=math.inf)
    min_mismatch_norm_l2 = min((r["mismatched_RY_normalized_l2"] for r in baseline_rows), default=0.0)
    all_have_points = all(r["n_points"] >= 8 for r in rows)
    passed = bool(all_have_points and max_norm_l2_at_baseline <= 0.03 and min_mismatch_norm_l2 >= 0.40)
    return {
        "passed": passed,
        "max_normalized_l2_at_baseline": float(max_norm_l2_at_baseline),
        "min_mismatched_normalized_l2": float(min_mismatch_norm_l2),
        "cases": rows,
    }


def contact_line_positions(
    phi: Any,
    sdf: Any,
    dx: float,
    dy: float,
    *,
    level: float = 0.5,
    eps: float | None = None,
    x0: float | None = None,
    Lx: float | None = None,
) -> dict[str, Any]:
    """Periodic-safe contact-line positions, contact width, and detachment detector.

    Uses both direct near-wall contour crossings and the circle-fit + wall-plane
    intersection around the periodic drop centroid. When the droplet detaches
    (``bottom_gap > max(2*eps, 2.5*dx)`` or no wall contact exists),
    ``contact_line_exists`` is ``False``, ``detachment_observed`` is ``True``,
    and ``sessile_angle_deg`` is ``None`` (the circle fit is no longer a sessile
    contact angle).
    """
    import phasefield as pf

    phase = np.asarray(phi, dtype=np.float64)
    distance = np.asarray(sdf, dtype=np.float64)
    if phase.ndim != 2 or phase.shape != distance.shape:
        raise ValueError("phi and sdf must be matching 2-D arrays")
    nx, ny = phase.shape
    if not (math.isfinite(dx) and math.isfinite(dy) and dx > 0 and dy > 0):
        raise ValueError("dx and dy must be finite and positive")
    period = float(nx * dx if Lx is None else Lx)
    eps_val = float(2.0 * dx if eps is None else eps)

    x_axis = (np.arange(nx, dtype=np.float64) + 0.5) * dx
    y_axis = (np.arange(ny, dtype=np.float64) + 0.5) * dy

    fluid_liquid = (phase >= float(level)) & (distance >= 0.0)
    if not np.any(fluid_liquid):
        return {
            "contact_line_exists": False,
            "detachment_observed": True,
            "contour_wall_intersection_count": 0,
            "bottom_gap": None,
            "left_contact_x": None,
            "right_contact_x": None,
            "left_contact_x_wrapped": None,
            "right_contact_x_wrapped": None,
            "contact_width": 0.0,
            "x_cm": None,
            "y_cm": None,
            "top_height": None,
            "sessile_angle_deg": None,
            "raw_circle_fit_angle_deg": None,
        }

    _, y_cm = obs.center_of_mass(phase, distance, x_axis, y_axis)
    # Periodic circular mean in x so droplets straddling x=0 / x=Lx unwrap around their true center
    fluid_weights = np.where(distance >= 0.0, np.clip(phase, 0.0, 1.0), 0.0)
    theta_x = (2.0 * math.pi / period) * x_axis[:, None]
    sin_sum = float(np.sum(fluid_weights * np.sin(theta_x)))
    cos_sum = float(np.sum(fluid_weights * np.cos(theta_x)))
    x_cm = float(((math.atan2(sin_sum, cos_sum) * period / (2.0 * math.pi)) + period) % period)
    ref_x = float(x_cm if x0 is None else x0)
    gap = obs.bottom_gap(phase, distance, threshold=float(level))
    top_h = obs.top_height(phase, distance, y_axis, threshold=float(level))

    # Extract all contour points in fluid (cutoff=0) and bulk contour points (cutoff=max(eps, dx))
    all_pts = pf.contact_angle_contour_points(
        phase, distance, dx, dy, level=float(level), cutoff=0.0, x0=ref_x, period=period
    )
    near_wall_band = max(2.0 * eps_val, 2.5 * dy)
    near_wall_pts = all_pts[all_pts[:, 2] <= near_wall_band] if len(all_pts) else np.zeros((0, 3))
    intersection_count = int(len(near_wall_pts))

    # Wall plane height at ref_x
    col_index = int(np.clip(round((ref_x - 0.5 * dx) / dx), 0, nx - 1))
    col_sdf = distance[col_index, :]
    wall_crossings = []
    for j in range(ny - 1):
        a, b = col_sdf[j], col_sdf[j + 1]
        if a == 0.0:
            wall_crossings.append(float(y_axis[j]))
        elif (a < 0.0) != (b < 0.0):
            t_w = -a / (b - a)
            wall_crossings.append(float(y_axis[j] + t_w * (y_axis[j + 1] - y_axis[j])))
    wall_y = float(max(wall_crossings)) if wall_crossings else float("nan")

    cutoff = max(eps_val, dx)
    bulk_pts = all_pts[all_pts[:, 2] >= cutoff] if len(all_pts) else np.zeros((0, 3))
    raw_angle_deg: float | None = None
    circle_left_unwrapped: float | None = None
    circle_right_unwrapped: float | None = None
    if len(bulk_pts) >= 8 and math.isfinite(wall_y):
        try:
            xc_rel, yc, radius = pf.fit_circle(bulk_pts[:, 0], bulk_pts[:, 1])
            if radius > 0.0 and math.isfinite(radius):
                cosine = (wall_y - yc) / radius
                raw_angle_deg = float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))))
                dy_wall = wall_y - yc
                if radius * radius > dy_wall * dy_wall:
                    half_chord = math.sqrt(max(0.0, radius * radius - dy_wall * dy_wall))
                    circle_left_unwrapped = float(ref_x + xc_rel - half_chord)
                    circle_right_unwrapped = float(ref_x + xc_rel + half_chord)
        except (ValueError, np.linalg.LinAlgError):
            pass

    attached = bool(gap <= near_wall_band and intersection_count >= 2)
    if attached and circle_left_unwrapped is not None and circle_right_unwrapped is not None:
        left_unwrapped = circle_left_unwrapped
        right_unwrapped = circle_right_unwrapped
    elif attached and len(near_wall_pts) >= 2:
        left_unwrapped = float(ref_x + np.min(near_wall_pts[:, 0]))
        right_unwrapped = float(ref_x + np.max(near_wall_pts[:, 0]))
    else:
        attached = False
        left_unwrapped = None
        right_unwrapped = None

    if not attached or left_unwrapped is None or right_unwrapped is None:
        return {
            "contact_line_exists": False,
            "detachment_observed": True,
            "contour_wall_intersection_count": intersection_count,
            "bottom_gap": float(gap),
            "left_contact_x": None,
            "right_contact_x": None,
            "left_contact_x_wrapped": None,
            "right_contact_x_wrapped": None,
            "contact_width": 0.0,
            "x_cm": float(x_cm),
            "y_cm": float(y_cm),
            "top_height": float(top_h),
            "sessile_angle_deg": None,
            "raw_circle_fit_angle_deg": raw_angle_deg,
        }

    width = float(max(0.0, right_unwrapped - left_unwrapped))
    return {
        "contact_line_exists": True,
        "detachment_observed": False,
        "contour_wall_intersection_count": intersection_count,
        "bottom_gap": float(gap),
        "left_contact_x": float(left_unwrapped),
        "right_contact_x": float(right_unwrapped),
        "left_contact_x_wrapped": float(left_unwrapped % period),
        "right_contact_x_wrapped": float(right_unwrapped % period),
        "contact_width": width,
        "x_cm": float(x_cm),
        "y_cm": float(y_cm),
        "top_height": float(top_h),
        "sessile_angle_deg": raw_angle_deg,
        "raw_circle_fit_angle_deg": raw_angle_deg,
    }


def _unwrap_periodic_series(values: Sequence[float | None], period: float | None) -> np.ndarray:
    """Unwrap a 1-D coordinate series across a periodic domain of length ``period``."""
    arr = np.asarray([np.nan if v is None else float(v) for v in values], dtype=np.float64)
    if period is None or not math.isfinite(period) or period <= 0:
        return arr
    valid = np.isfinite(arr)
    if np.sum(valid) < 2:
        return arr
    out = arr.copy()
    angles = 2.0 * math.pi * out[valid] / float(period)
    out[valid] = np.unwrap(angles) * float(period) / (2.0 * math.pi)
    return out


def contact_line_velocity(
    times: Sequence[float],
    left_x: Sequence[float | None],
    right_x: Sequence[float | None],
    *,
    Lx: float | None = None,
) -> dict[str, Any]:
    """Compute periodic-safe contact-line velocities and speeds from sampled positions."""
    t_arr = np.asarray(times, dtype=np.float64)
    if t_arr.ndim != 1 or len(t_arr) != len(left_x) or len(t_arr) != len(right_x):
        raise ValueError("times, left_x, and right_x must be 1-D sequences of matching length")
    if len(t_arr) == 0:
        raise ValueError("times must not be empty")
    if len(t_arr) > 1 and np.any(np.diff(t_arr) <= 0):
        raise ValueError("times must be strictly increasing")

    left_u = _unwrap_periodic_series(left_x, Lx)
    right_u = _unwrap_periodic_series(right_x, Lx)
    n = len(t_arr)
    v_left = np.full(n, np.nan, dtype=np.float64)
    v_right = np.full(n, np.nan, dtype=np.float64)

    if n >= 2:
        for i in range(n):
            if i == 0:
                dt = t_arr[1] - t_arr[0]
                v_left[i] = (left_u[1] - left_u[0]) / dt
                v_right[i] = (right_u[1] - right_u[0]) / dt
            elif i == n - 1 or not (np.isfinite(left_u[i + 1]) and np.isfinite(left_u[i - 1])):
                dt = t_arr[i] - t_arr[i - 1]
                v_left[i] = (left_u[i] - left_u[i - 1]) / dt
                v_right[i] = (right_u[i] - right_u[i - 1]) / dt
            else:
                dt = t_arr[i + 1] - t_arr[i - 1]
                v_left[i] = (left_u[i + 1] - left_u[i - 1]) / dt
                v_right[i] = (right_u[i + 1] - right_u[i - 1]) / dt

    speed_left = np.abs(v_left)
    speed_right = np.abs(v_right)
    mean_speed = 0.5 * (speed_left + speed_right)
    spreading_rate = 0.5 * (v_right - v_left)
    translation_velocity = 0.5 * (v_left + v_right)

    def _clean_list(arr: np.ndarray) -> list[float | None]:
        return [float(v) if math.isfinite(float(v)) else None for v in arr]

    finite_mean = mean_speed[np.isfinite(mean_speed)]
    finite_trans = translation_velocity[np.isfinite(translation_velocity)]
    return {
        "left_velocity": _clean_list(v_left),
        "right_velocity": _clean_list(v_right),
        "left_speed": _clean_list(speed_left),
        "right_speed": _clean_list(speed_right),
        "mean_speed": _clean_list(mean_speed),
        "spreading_rate": _clean_list(spreading_rate),
        "translation_velocity": _clean_list(translation_velocity),
        "final_left_speed": float(speed_left[-1]) if np.isfinite(speed_left[-1]) else None,
        "final_right_speed": float(speed_right[-1]) if np.isfinite(speed_right[-1]) else None,
        "final_mean_speed": float(mean_speed[-1]) if np.isfinite(mean_speed[-1]) else None,
        "mean_translation_velocity": float(np.mean(finite_trans)) if finite_trans.size else None,
        "max_mean_speed": float(np.max(finite_mean)) if finite_mean.size else None,
    }


def late_time_linear_trend(
    times: Sequence[float],
    values: Sequence[float | None],
    *,
    fraction: float = 0.2,
    min_points: int = 5,
    zero_slope_tol: float = 1.0e-12,
) -> dict[str, Any]:
    """Compute linear slope, R^2, and sign over the last ``fraction`` of samples."""
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must lie in (0, 1]")
    t_all = np.asarray(times, dtype=np.float64)
    v_all = np.asarray([np.nan if v is None else float(v) for v in values], dtype=np.float64)
    if t_all.ndim != 1 or t_all.shape != v_all.shape or len(t_all) < 2:
        raise ValueError("times and values must be 1-D sequences with at least 2 elements")

    count = max(int(min_points), int(math.ceil(len(t_all) * float(fraction))))
    count = min(len(t_all), max(2, count))
    t_win = t_all[-count:]
    v_win = v_all[-count:]
    valid = np.isfinite(t_win) & np.isfinite(v_win)
    t_win = t_win[valid]
    v_win = v_win[valid]
    if len(t_win) < 2 or float(np.ptp(t_win)) <= 0.0:
        return {
            "slope": 0.0,
            "intercept": 0.0,
            "r_squared": 0.0,
            "sign": 0,
            "n_samples": int(len(t_win)),
            "t_start": None,
            "t_end": None,
            "delta_value": 0.0,
        }

    t_mean = float(np.mean(t_win))
    v_mean = float(np.mean(v_win))
    dt = t_win - t_mean
    dv = v_win - v_mean
    denom = float(np.sum(dt * dt))
    slope = float(np.sum(dt * dv) / denom) if denom > 1e-30 else 0.0
    intercept = float(v_mean - slope * t_mean)
    pred = slope * t_win + intercept
    ss_res = float(np.sum((v_win - pred) ** 2))
    ss_tot = float(np.sum(dv * dv))
    r_squared = float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0)) if ss_tot > 1e-24 else 0.0
    sign = 0 if abs(slope) <= float(zero_slope_tol) else (1 if slope > 0.0 else -1)
    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "sign": int(sign),
        "n_samples": int(len(t_win)),
        "t_start": float(t_win[0]),
        "t_end": float(t_win[-1]),
        "delta_value": float(v_win[-1] - v_win[0]),
    }


def fit_relaxation_asymptote(
    times: Sequence[float],
    angles: Sequence[float | None],
    *,
    target_deg: float | None = None,
) -> dict[str, Any]:
    """Fit ``theta(t) = theta_inf + A * exp(-t / tau)`` as a diagnostic only.

    The extrapolated ``theta_inf_fit`` is NEVER reported as an equilibrium contact
    angle (``diagnostic_only=True``, ``is_equilibrium_angle=False``).
    """
    t_all = np.asarray(times, dtype=np.float64)
    a_all = np.asarray([np.nan if a is None else float(a) for a in angles], dtype=np.float64)
    valid = np.isfinite(t_all) & np.isfinite(a_all)
    t = t_all[valid]
    y = a_all[valid]
    if len(t) < 4 or float(np.ptp(t)) <= 0.0 or float(np.ptp(y)) <= 1e-9:
        return {
            "theta_inf_fit": None,
            "amplitude_fit": None,
            "tau_fit": None,
            "fit_r_squared": None,
            "diagnostic_only": True,
            "is_equilibrium_angle": False,
            "used_for_convergence_gate": False,
        }

    t_shift = t - t[0]
    horizon = float(t_shift[-1])
    tau_grid = np.geomspace(max(horizon * 0.02, 1e-3), horizon * 20.0, 240)
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    best_sse = math.inf
    best_params = (float(y[-1]), 0.0, float(horizon))
    for tau in tau_grid:
        basis = np.exp(-t_shift / float(tau))
        design = np.column_stack([np.ones_like(basis), basis])
        coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
        theta_inf, amp = float(coeffs[0]), float(coeffs[1])
        if not (0.0 <= theta_inf <= 180.0):
            continue
        pred = theta_inf + amp * basis
        sse = float(np.sum((y - pred) ** 2))
        if sse < best_sse:
            best_sse = sse
            best_params = (theta_inf, amp, float(tau))

    theta_inf, amp, tau = best_params
    r2 = float(np.clip(1.0 - best_sse / ss_tot, 0.0, 1.0)) if ss_tot > 1e-24 and math.isfinite(best_sse) else 0.0
    return {
        "theta_inf_fit": float(theta_inf),
        "amplitude_fit": float(amp),
        "tau_fit": float(tau),
        "fit_r_squared": float(r2),
        "target_deg": None if target_deg is None else float(target_deg),
        "diagnostic_only": True,
        "is_equilibrium_angle": False,
        "used_for_convergence_gate": False,
    }


def evaluate_mobility_collapse(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compare ``theta(M*t)``, ``F(M*t)``, and ``contact_width(M*t)`` across mobilities."""
    if len(runs) < 2:
        return {
            "run_count": len(runs),
            "common_mt_max": 0.0,
            "angle_collapse_max_spread_deg": None,
            "angle_collapse_rms_deg": None,
            "energy_collapse_max_rel_spread": None,
            "width_collapse_max_spread": None,
            "converged_count": sum(1 for r in runs if r.get("converged") is True),
            "converged_equilibrium_spread_deg": None,
            "mt_curves_collapse": False,
        }

    series_list = []
    for run in runs:
        samples = run.get("samples", [])
        mt = np.asarray([float(s["mobility_scaled_time"]) for s in samples], dtype=np.float64)
        ang = np.asarray(
            [np.nan if s.get("measured_angle_deg") is None else float(s["measured_angle_deg"]) for s in samples],
            dtype=np.float64,
        )
        en = np.asarray([float(s["free_energy"]) for s in samples], dtype=np.float64)
        wid = np.asarray(
            [np.nan if s.get("contact_width") is None else float(s["contact_width"]) for s in samples],
            dtype=np.float64,
        )
        valid = np.isfinite(mt) & np.isfinite(ang) & np.isfinite(en)
        if np.sum(valid) >= 2:
            series_list.append((mt[valid], ang[valid], en[valid], wid[valid]))

    if len(series_list) < 2:
        return {
            "run_count": len(runs),
            "common_mt_max": 0.0,
            "angle_collapse_max_spread_deg": None,
            "angle_collapse_rms_deg": None,
            "energy_collapse_max_rel_spread": None,
            "width_collapse_max_spread": None,
            "converged_count": 0,
            "converged_equilibrium_spread_deg": None,
            "mt_curves_collapse": False,
        }

    mt_min = max(float(s[0][0]) for s in series_list)
    mt_max = min(float(s[0][-1]) for s in series_list)
    if mt_max <= mt_min:
        return {
            "run_count": len(runs),
            "common_mt_max": float(max(0.0, mt_max)),
            "angle_collapse_max_spread_deg": None,
            "angle_collapse_rms_deg": None,
            "energy_collapse_max_rel_spread": None,
            "width_collapse_max_spread": None,
            "converged_count": 0,
            "converged_equilibrium_spread_deg": None,
            "mt_curves_collapse": False,
        }

    grid = np.linspace(mt_min, mt_max, 25)
    interp_angles = np.vstack([np.interp(grid, s[0], s[1]) for s in series_list])
    interp_energies = np.vstack([np.interp(grid, s[0], s[2]) for s in series_list])
    interp_widths = np.vstack([np.interp(grid, s[0], s[3]) for s in series_list])

    angle_spreads = np.ptp(interp_angles, axis=0)
    angle_max_spread = float(np.max(angle_spreads))
    angle_rms_spread = float(np.sqrt(np.mean(angle_spreads**2)))

    energy_scale = max(1.0, float(np.mean(np.abs(interp_energies))))
    energy_max_rel = float(np.max(np.ptp(interp_energies, axis=0)) / energy_scale)
    width_max_spread = float(np.max(np.ptp(interp_widths, axis=0)))

    converged_angles = [
        float(r["equilibrium_angle_deg"])
        for r in runs
        if r.get("converged") is True and isinstance(r.get("equilibrium_angle_deg"), (int, float))
    ]
    eq_spread = float(np.ptp(converged_angles)) if len(converged_angles) >= 2 else None
    collapse_ok = bool(angle_max_spread <= 1.0 and (eq_spread is None or eq_spread <= 2.0))
    return {
        "run_count": len(runs),
        "common_mt_min": float(mt_min),
        "common_mt_max": float(mt_max),
        "angle_collapse_max_spread_deg": angle_max_spread,
        "angle_collapse_rms_deg": angle_rms_spread,
        "energy_collapse_max_rel_spread": energy_max_rel,
        "width_collapse_max_spread": width_max_spread,
        "converged_count": len(converged_angles),
        "converged_equilibrium_angles_deg": converged_angles,
        "converged_equilibrium_spread_deg": eq_spread,
        "mt_curves_collapse": collapse_ok,
    }


def classify_equilibration(
    evidence: dict[str, Any],
    *,
    acceptance_tol_deg: float = 5.0,
    ry_small_tol: float = 0.15,
    dt_spread_tol_deg: float = 2.0,
) -> tuple[str, dict[str, Any]]:
    """Classify a single run or multi-run target audit into the frozen L1A-2d categories.

    Returns ``(label, details)`` where ``label`` is guaranteed to belong to
    :data:`ALLOWED_CLASSIFICATIONS`.
    """
    detachment = bool(evidence.get("detachment_observed", False))
    contact_exists = bool(evidence.get("contact_line_exists", True))
    if detachment or not contact_exists:
        return "TOPOLOGY_CHANGE_OR_DETACHMENT", {
            "rule": "detachment_or_missing_contact_line",
            "detachment_observed": detachment,
            "contact_line_exists": contact_exists,
        }

    dt_spread = evidence.get("dt_angle_spread_deg")
    if isinstance(dt_spread, (int, float)) and math.isfinite(float(dt_spread)) and float(dt_spread) > dt_spread_tol_deg:
        return "TIME_STEP_LIMITED", {
            "rule": "material_dt_sensitivity",
            "dt_angle_spread_deg": float(dt_spread),
            "dt_spread_tol_deg": float(dt_spread_tol_deg),
        }

    res_improves = bool(evidence.get("resolution_error_systematically_decreases", False))
    if res_improves:
        return "RESOLUTION_LIMITED", {
            "rule": "systematic_refinement_error_reduction",
            "resolution_error_systematically_decreases": True,
        }

    ch_only_conv_near = evidence.get("ch_only_converged_near_target")
    chns_conv_near = evidence.get("chns_converged_near_target")
    if ch_only_conv_near is True and chns_conv_near is False:
        return "HYDRODYNAMIC_COUPLING_LIMITED", {
            "rule": "ch_only_reaches_target_but_chns_fails",
            "ch_only_converged_near_target": True,
            "chns_converged_near_target": False,
        }

    # One-factor wall-gain ablation (diagnostic only): if multiplying the applied wall energy by the
    # inverse of the measured fluid-side wall-kernel fraction moves the *same* CH-only run to the Young
    # target while the unmodified run stays biased, the bias is the wall-flux assembly itself.
    if evidence.get("gain_ablation_restores_target") is True:
        return "BOUNDARY_DISCRETIZATION_LIMITED", {
            "rule": "wall_gain_ablation_restores_target",
            "wall_kernel_fluid_fraction": evidence.get("wall_kernel_fluid_fraction"),
            "ablation_equilibrium_error_deg": evidence.get("ablation_equilibrium_error_deg"),
            "unmodified_equilibrium_error_deg": evidence.get("unmodified_equilibrium_error_deg"),
            "secondary_mechanism": "KINETICS_LIMITED" if evidence.get("transient_kinetics_limited") else None,
        }

    converged = bool(evidence.get("converged", False))
    eq_error = evidence.get("equilibrium_error_deg")
    ry_norm = float(evidence.get("RY_normalized_l2", 0.0))
    mt_collapse = bool(evidence.get("mt_curves_collapse", False))
    higher_m_converges_near_target = evidence.get("higher_m_converges_near_target")
    higher_m_converges_biased = evidence.get("higher_m_converges_biased")
    relaxing_toward_target = bool(evidence.get("relaxing_toward_target", False))

    if converged and isinstance(eq_error, (int, float)) and math.isfinite(float(eq_error)):
        abs_err = abs(float(eq_error))
        if abs_err <= float(acceptance_tol_deg):
            return "KINETICS_LIMITED", {
                "rule": "converged_within_acceptance_tolerance",
                "equilibrium_error_deg": float(eq_error),
                "acceptance_tol_deg": float(acceptance_tol_deg),
                "RY_normalized_l2": ry_norm,
            }
        # Converged outside acceptance tolerance -> wrong equilibrium!
        if (
            ry_norm > float(ry_small_tol)
            and bool(evidence.get("ry_plateau", False))
            and bool(evidence.get("prefer_boundary_discretization", False))
        ):
            return "BOUNDARY_DISCRETIZATION_LIMITED", {
                "rule": "converged_outside_acceptance_with_large_wall_residual",
                "equilibrium_error_deg": float(eq_error),
                "RY_normalized_l2": ry_norm,
                "ry_small_tol": float(ry_small_tol),
                "secondary_mechanism": "THERMODYNAMIC_EQUILIBRIUM_BIASED",
            }
        return "THERMODYNAMIC_EQUILIBRIUM_BIASED", {
            "rule": "converged_to_wrong_equilibrium_angle",
            "equilibrium_error_deg": float(eq_error),
            "acceptance_tol_deg": float(acceptance_tol_deg),
            "RY_normalized_l2": ry_norm,
            "boundary_discretization_residual_large": bool(ry_norm > float(ry_small_tol)),
            "secondary_mechanism": "BOUNDARY_DISCRETIZATION_LIMITED" if ry_norm > float(ry_small_tol) else None,
        }

    # Not yet converged within the run's step budget:
    if higher_m_converges_near_target is True and mt_collapse and ry_norm <= float(ry_small_tol):
        return "KINETICS_LIMITED", {
            "rule": "slow_relaxation_with_mt_collapse_to_valid_equilibrium",
            "mt_curves_collapse": True,
            "RY_normalized_l2": ry_norm,
        }

    if higher_m_converges_biased is True:
        if ry_norm > float(ry_small_tol):
            return "BOUNDARY_DISCRETIZATION_LIMITED", {
                "rule": "transient_kinetics_plus_large_wall_residual_and_biased_high_m_equilibrium",
                "RY_normalized_l2": ry_norm,
                "mt_curves_collapse": mt_collapse,
                "secondary_mechanism": "THERMODYNAMIC_EQUILIBRIUM_BIASED",
                "kinetics_at_10k": "KINETICS_LIMITED",
            }
        return "THERMODYNAMIC_EQUILIBRIUM_BIASED", {
            "rule": "high_m_converges_to_wrong_equilibrium",
            "RY_normalized_l2": ry_norm,
            "mt_curves_collapse": mt_collapse,
            "kinetics_at_10k": "KINETICS_LIMITED",
        }

    if relaxing_toward_target and (mt_collapse or ry_norm <= float(ry_small_tol)):
        return "KINETICS_LIMITED", {
            "rule": "monotonically_relaxing_toward_target",
            "relaxing_toward_target": True,
            "mt_curves_collapse": mt_collapse,
            "RY_normalized_l2": ry_norm,
        }

    if ry_norm > float(ry_small_tol) and bool(evidence.get("ry_plateau", False)):
        return "BOUNDARY_DISCRETIZATION_LIMITED", {
            "rule": "large_plateaued_wall_boundary_residual",
            "RY_normalized_l2": ry_norm,
            "ry_small_tol": float(ry_small_tol),
        }

    return "INCONCLUSIVE", {
        "rule": "insufficient_convergence_or_trend_evidence",
        "converged": converged,
        "RY_normalized_l2": ry_norm,
    }
