"""L1A-2n diagnostic ghost / domain-closure prototypes (repair DESIGN only).

Nothing in this module is reachable from the production path: ``phasefield.py`` never
imports it, and the production step semantics are untouched. Each candidate is a
diagnostic evaluation used to measure how much of the inactive-storage coupling survives
when the arbitrary raw ``phi[V=0]`` dependence is removed.

Candidates (stage spec sections 27-29):

* ``raw_storage_v0`` -- the production reference (no closure; reads raw ``phi[I0]``).
* ``boundary_consistent_ghost_v1`` -- candidate A: ghost values for the inactive cells
  derived from the *existing* validated Young wall boundary condition
  ``eps dphi/dn + g_w'(phi) = 0`` (``phasefield.natural_wall_normal_derivative``), linearly
  extended from the nearest physical (wall-measure) cell. Deterministic, independent of the
  raw inactive storage, formal-mass neutral (support has ``V_i = 0``), not a new
  contact-angle model.
* ``nearest_physical_extension_v1`` -- candidate B (falsification/control): copy the
  nearest physical cell value; labelled a numerical control because it is not derived from
  the governing boundary condition.
* ``one_sided_cap_closure_v1`` -- candidate C: operator-side closure; the capillary
  y-gradient at wall-adjacent physical cells is evaluated one-sided from physical cells
  only (the cut-cell wall face is closed, so the wall-side neighbour is not part of the
  physical stencil). Never multiplies a contaminated output by ``V>0`` after the stencil.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

import phasefield as pf

GHOST_CLOSURES = (
    "raw_storage_v0",
    "boundary_consistent_ghost_v1",
    "nearest_physical_extension_v1",
)


def _nearest_physical_cell(physical: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index map of the nearest physical cell in the same column (4-neighbour y direction).

    Returns ``(index, distance_rows)``: for every cell the same-column ``V_i > 0`` cell with
    minimal row distance (ties resolved toward larger y, i.e. away from the wall), or ``-1``
    when the column has no physical cell. The production geometry makes the physical domain a
    single connected slab above the wall slab, so the nearest physical cell of an inactive
    cell is the first physical row above it.
    """
    nx, ny = physical.shape
    nearest = np.full((nx, ny), -1, dtype=np.int32)
    distance = np.full((nx, ny), np.iinfo(np.int32).max, dtype=np.int32)
    # scan upward (increasing j): nearest physical at or above
    current = np.full(nx, -1, dtype=np.int32)
    for j in range(ny):
        current = np.where(physical[:, j], j, current)
        candidate = current
        dist = np.where(candidate >= 0, np.abs(j - candidate), np.iinfo(np.int32).max)
        take = dist < distance
        nearest = np.where(take, candidate, nearest)
        distance = np.where(take, dist, distance)
    # scan downward (decreasing j): nearest physical at or below
    current = np.full(nx, -1, dtype=np.int32)
    for j in range(ny - 1, -1, -1):
        current = np.where(physical[:, j], j, current)
        candidate = current
        dist = np.where(candidate >= 0, np.abs(j - candidate), np.iinfo(np.int32).max)
        # strict tie-break toward the upward scan (away from the wall)
        take = dist < distance
        nearest = np.where(take, candidate, nearest)
        distance = np.where(take, dist, distance)
    return nearest, distance


def ghost_values(
    phi: np.ndarray,
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
    closure: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return ``(phi_with_ghost_inactive_values, metadata)`` for the chosen closure.

    Physical cells are bitwise untouched in every closure; only the storage behind
    ``V_i == 0`` is replaced. ``raw_storage_v0`` returns the input unchanged (the
    production reference).
    """
    phi = np.asarray(phi, dtype=np.float64)
    volume = np.asarray(pf.phase_transport_operator(solid, p).volume, dtype=np.float64)
    physical = volume > 0.0
    inactive = ~physical
    if closure == "raw_storage_v0":
        return phi.copy(), {"closure": closure, "changed_inactive_cells": 0}
    ghost = phi.copy()
    if closure == "nearest_physical_extension_v1":
        nearest, _rows = _nearest_physical_cell(physical)
        i_idx = np.arange(phi.shape[0])[:, None]
        safe = np.clip(nearest, 0, None)
        source = phi[i_idx, safe]
        has_source = nearest >= 0
        ghost[inactive] = np.where(has_source, source, phi)[inactive]
        meta = {
            "closure": closure,
            "changed_inactive_cells": int((ghost[inactive] != phi[inactive]).sum()),
            "derivation": "nearest physical cell value (numerical control, not BC-derived)",
        }
        return ghost, meta
    if closure == "boundary_consistent_ghost_v1":
        # Young-consistent linear extension: at every wall-measure (physical) cell the
        # production natural BC fixes eps*dphi/dn = -g_w'(phi); extend that slope from the
        # nearest physical cell through the wall plane into the inactive cell centres.
        sdf = np.asarray(solid.sdf, dtype=np.float64)
        wall_area = np.asarray(solid.wall_area, dtype=np.float64)
        derivative = np.asarray(pf.natural_wall_normal_derivative(jnp.asarray(phi), solid, p), dtype=np.float64)
        nearest, _rows = _nearest_physical_cell(physical)
        i_idx = np.arange(phi.shape[0])[:, None]
        safe_nearest = np.clip(nearest, 0, None)
        # nearest holds a y-index (j') for every cell (i, j): sample phi[i, j'] etc.
        phi_nearest = phi[i_idx, safe_nearest]
        slope = derivative[i_idx, safe_nearest]
        # normal distance from the nearest physical cell centre, through the wall plane, to
        # the ghost cell centre: wall_distance of the nearest cell when it carries the wall
        # measure, its own |sdf| as the wall-plane distance otherwise, plus the ghost
        # cell's |sdf| (the wall lies between the two centres by construction).
        wall_distance_nearest = np.asarray(solid.wall_distance, dtype=np.float64)[i_idx, safe_nearest]
        sdf_nearest = np.abs(sdf)[i_idx, safe_nearest]
        wall_plane_distance = np.where(wall_area > 0.0, wall_distance_nearest, sdf_nearest)
        distance_into_solid = wall_plane_distance + np.abs(sdf)
        has_source = nearest >= 0
        values = np.where(has_source, phi_nearest + slope * distance_into_solid, phi)
        ghost[inactive] = np.clip(values, 0.0, 1.0)[inactive]
        meta = {
            "closure": closure,
            "changed_inactive_cells": int((ghost[inactive] != phi[inactive]).sum()),
            "derivation": (
                "linear extension of the production natural wall BC eps*dphi/dn = -g_w'(phi)"
                " from the nearest wall-measure physical cell; clipped to the phase range"
            ),
            "uses_raw_inactive_storage": False,
            "uses_only_production_boundary_semantics": True,
        }
        return ghost, meta
    raise KeyError(f"unknown ghost closure {closure!r}")


def capillary_acceleration_diagnostic(
    phi: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
    closure: str,
) -> dict[str, np.ndarray]:
    """The production Korteweg evaluation under a ghost closure (candidate A/B path).

    Identical arithmetic to ``phasefield.rhs`` except that ``grad(phi)`` consumes the
    closure-constructed field instead of the raw storage on ``V_i == 0`` cells.
    """
    ghost, meta = ghost_values(phi, solid, p, closure)
    phi_g = jnp.asarray(ghost)
    mu = pf.chemical_potential(jnp.asarray(phi, dtype=phi_g.dtype), solid, p)
    phi_x = pf._ddx(phi_g, p.dx)
    phi_y = pf._ddy(phi_g, p.dy)
    cap_x = (pf.SIGMA_NORM / p.We) * mu * phi_x / p.rho_l
    cap_y = (pf.SIGMA_NORM / p.We) * mu * phi_y / p.rho_l
    del u, v
    return {"cap_x": np.asarray(cap_x, dtype=np.float64), "cap_y": np.asarray(cap_y, dtype=np.float64), "meta": meta}


def one_sided_cap_closure_v1(
    phi: np.ndarray,
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
) -> dict[str, np.ndarray]:
    """Candidate C: operator-side one-sided closure for the capillary y-gradient.

    At physical cells whose wall-side y-neighbour is inactive (``V_i == 0``), the closed
    wall face means the central difference ``_ddy`` is not a physical stencil anyway; the
    diagnostic closure uses the one-sided physical-side difference ``(phi_j - phi_jp1)/dy``.
    Physical cells with two physical y-neighbours keep the production central difference
    bitwise. Nothing is masked after the stencil: the contaminated input is never read.
    """
    phi_j = np.asarray(phi, dtype=np.float64)
    volume = np.asarray(pf.phase_transport_operator(solid, p).volume, dtype=np.float64)
    inactive = volume == 0.0
    central = np.asarray(pf._ddy(jnp.asarray(phi_j), p.dy))
    # one-sided differences that read PHYSICAL cells only:
    # away-from-wall form (phi_j - phi_{j+1})/dy, toward-wall form (phi_j - phi_{j-1})/dy
    away = np.empty_like(phi_j)
    away[:, :-1] = (phi_j[:, :-1] - phi_j[:, 1:]) / float(p.dy)
    away[:, -1] = central[:, -1]
    toward = np.empty_like(phi_j)
    toward[:, 1:] = (phi_j[:, 1:] - phi_j[:, :-1]) / float(p.dy)
    toward[:, 0] = central[:, 0]
    # a physical cell loses its central stencil when its -y neighbour is inactive, or when it
    # is the top row and the +y WRAP neighbour (raw jnp.roll) is the inactive bottom storage
    closure_mask = np.zeros(phi_j.shape, dtype=bool)
    closure_mask[:, 1:] |= inactive[:, :-1]
    top_row_wraps_into_inactive = inactive[:, 0].copy()
    phi_y = np.where(closure_mask, away, central)
    # axis 0 is x: the top y-row is the LAST index along axis 1
    phi_y[:, -1] = np.where(top_row_wraps_into_inactive, toward[:, -1], central[:, -1])
    mu = pf.chemical_potential(jnp.asarray(phi_j), solid, p)
    cap_x = (pf.SIGMA_NORM / p.We) * mu * pf._ddx(jnp.asarray(phi_j), p.dx) / p.rho_l
    cap_y = (pf.SIGMA_NORM / p.We) * mu * phi_y / p.rho_l
    return {
        "cap_x": np.asarray(cap_x, dtype=np.float64),
        "cap_y": np.asarray(cap_y, dtype=np.float64),
        "meta": {
            "closure": "one_sided_cap_closure_v1",
            "operator_side": True,
            "wall_adjacent_physical_cells": int((closure_mask & (volume > 0.0)).sum()),
        },
    }


def diagnostic_one_substep(
    phi: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
    closure: str,
) -> dict[str, np.ndarray]:
    """One production-arithmetic substep with the capillary closure applied (diagnostic).

    Replicates the exact production substep expressions (predictor, Brinkman damping,
    periodic projection) with ``u_rhs``/``v_rhs`` recomputed from the closure capillary.
    This is a diagnostic recomputation, never the production step.
    """
    cap = capillary_acceleration_diagnostic(phi, u, v, solid, p, closure)
    mu_expl = np.asarray(pf._explicit_chemical_potential(jnp.asarray(phi), solid, p), dtype=np.float64)
    adv_u = pf.div_upwind(jnp.asarray(u), jnp.asarray(v), jnp.asarray(u), p.dx, p.dy)
    adv_v = pf.div_upwind(jnp.asarray(u), jnp.asarray(v), jnp.asarray(v), p.dx, p.dy)
    lap_u = pf._lap(jnp.asarray(u), p.dx, p.dy)
    lap_v = pf._lap(jnp.asarray(v), p.dx, p.dy)
    nu = pf.nu_of(jnp.asarray(phi), p)
    u_rhs = (-adv_u + nu * lap_u).astype(np.float64) + cap["cap_x"]
    v_rhs = (-adv_v + nu * lap_v).astype(np.float64) + cap["cap_y"]
    dt = p.dt / 3.0
    chi = np.asarray(solid.chi, dtype=np.float64)
    damp = 1.0 / (1.0 + dt * chi / float(p.eta_pen))
    u_new = (np.asarray(u, dtype=np.float64) + dt * u_rhs) * damp
    v_new = (np.asarray(v, dtype=np.float64) + dt * v_rhs) * damp
    div = pf._ddx(jnp.asarray(u_new), p.dx) + pf._ddy(jnp.asarray(v_new), p.dy)
    pressure = pf.poisson_solve(jnp.asarray(div) / dt, p.m2_proj)
    u_proj = u_new - dt * np.asarray(pf._ddx(pressure, p.dx))
    v_proj = v_new - dt * np.asarray(pf._ddy(pressure, p.dy))
    return {
        "u_projected": u_proj,
        "v_projected": v_proj,
        "cap_x": cap["cap_x"],
        "cap_y": cap["cap_y"],
        "mu_expl": mu_expl,
        "ghost_meta": cap["meta"],
        "closure": closure,
    }
