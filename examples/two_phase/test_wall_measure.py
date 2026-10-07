"""L1A-2e tests: the embedded Young wall measure and boundary-flux assembly (contract v8).

Everything here is fast (N <= 96, <= 600 steps) so it can gate CI. The long equilibrium,
translation, refinement, precision and CHNS matrices live in
``production/embedded_young_audit.py`` and are never run in CI.

Coverage required by the L1A-2e specification:

* flat-wall measure equals the geometric length and is invariant under sub-cell translation;
* inclined-wall measure is the Euclidean cut length, with no Manhattan/staircase gain;
* no periodic-seam ghost (y extrapolated, x roll-equivariant);
* the production wall operator is the variational derivative of the discrete bulk + wall energy;
* exactly one wall-energy contribution (no double counting with the legacy kernel);
* the first-fluid-layer Young residual on manufactured fields, weighted by the new measure;
* the measure is angle-independent and carries no global share/gain correction;
* contract 8 with contract-7 data automatically stale;
* a small CH-only run moves 60/120/150 deg in the correct direction;
* no empirical contact-angle correction exists anywhere in the production path.
"""

from __future__ import annotations

import ast
import json
import dataclasses
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import phasefield as pf
import pytest
from production import contact_line_kinetics as clk
from production import wall_measure_audit as wma

HERE = Path(__file__).resolve().parent

WALL_SIGMA0 = math.sqrt(2.0) / 6.0
TRANSLATION_OFFSETS = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)
TARGETS = (60.0, 90.0, 120.0, 150.0)


@pytest.fixture
def x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _params(N=64, *, eps_factor=2.0, dtype=jnp.float32, **kwargs) -> pf.PhaseFieldParams:
    p = pf.PhaseFieldParams(Nx=N, Ny=N, Lx=6.0, Ly=6.0, dtype=dtype, **kwargs)
    p.eps = float(eps_factor) * p.dx
    return p


def _flat(N=64, *, offset=0.0, theta=90.0, dtype=jnp.float32, eps_factor=2.0):
    p = _params(N, eps_factor=eps_factor, dtype=dtype)
    wall_height = 0.25 + float(offset) * p.dy
    sdf = pf.surface_flat(p, wall_height=wall_height)
    solid = pf.make_solid(sdf, p, cos_theta=math.cos(math.radians(theta)))
    return p, solid, wall_height


def _flat_pinned_v8(N=64, *, offset=0.0, theta=90.0, dtype=jnp.float32, eps_factor=2.0):
    """The same wall with the pinned contract-v7/v8 transport geometry (hard-fluid ring)."""
    p, solid, wall_height = _flat(N, offset=offset, theta=theta, dtype=dtype, eps_factor=eps_factor)
    pinned = dataclasses.replace(p, phase_transport_geometry="hard_cell_v7")
    return pinned, pf.make_solid(pf.surface_flat(pinned, wall_height=wall_height), pinned), wall_height


# ---------------------------------------------------------------------------
#  flat wall: exact length and grid-alignment independence
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("N", [32, 64, 96, 128])
def test_flat_wall_measure_equals_geometric_length(N, x64):
    """sum_i A_wall,i is the wall length itself, not a diffuse-kernel fraction of it."""
    p, solid, _ = _flat(N, dtype=jnp.float64)
    area = np.asarray(solid.wall_area, dtype=np.float64)
    assert area.sum() == pytest.approx(p.Lx, rel=1e-12, abs=0.0)
    positive = area > 0.0
    assert positive.sum() == N  # exactly one control cell per column
    assert np.allclose(area[positive], p.dx, rtol=1e-12, atol=0.0)  # each carries one cut of length dx
    # one cell layer only, and always on a cell that *owns* a control volume
    rows = np.unique(np.argwhere(positive)[:, 1])
    assert rows.size == 1
    # Contract v9 hosts the measure on the cut cell itself, which may have a solid centre but
    # always has V_i > 0 (a positive-length contour implies a positive fluid polygon).
    assert np.all(np.asarray(solid.geometry.volume)[positive] > 0.0)
    assert np.allclose(np.asarray(solid.wall_normal_x)[positive], 0.0, atol=1e-14)
    assert np.allclose(np.asarray(solid.wall_normal_y)[positive], -1.0, rtol=0.0, atol=1e-12)
    # the pinned contract-v7/v8 mode still relocates the measure to a hard-fluid centre
    _p8, solid8, _ = _flat_pinned_v8(N, dtype=jnp.float64)
    area8 = np.asarray(solid8.wall_area, dtype=np.float64)
    positive8 = area8 > 0.0
    assert area8.sum() == pytest.approx(p.Lx, rel=1e-12, abs=0.0)
    assert np.all(np.asarray(solid8.sdf)[positive8] >= 0.0)


def test_flat_wall_measure_is_translation_invariant(x64):
    """The primary merge gate: translating the wall inside a cell cannot change the measure.

    The contract-v7 diffuse kernel changed its *effective fluid-side* measure by tens of percent
    over the same offsets (the L1A-2d root cause); the v8 cut-cell measure is invariant to
    round-off, so both the total and every per-cell value are offset independent.
    """
    totals, per_cell, rows, legacy = [], [], [], []
    for offset in TRANSLATION_OFFSETS:
        p, solid, _ = _flat(96, offset=offset, dtype=jnp.float64)
        area = np.asarray(solid.wall_area, dtype=np.float64)
        totals.append(area.sum())
        per_cell.append(area[area > 0.0])
        rows.append(np.unique(np.argwhere(area > 0.0)[:, 1]).tolist())
        legacy.append(
            clk.fluid_wall_delta_integral(np.asarray(solid.sdf, dtype=np.float64), p.dx, p.dy)[
                "fluid_fraction_of_wall_kernel"
            ]
        )
    totals = np.asarray(totals)
    spread = (totals.max() - totals.min()) / totals.mean()
    assert spread <= 0.01  # specification gate
    assert spread <= 0.005  # ideal gate
    assert spread == 0.0  # measured: exactly invariant
    for values in per_cell:
        assert np.allclose(values, per_cell[0], rtol=1e-12, atol=0.0)
    # the control-cell row may move with the offset; the measure does not
    assert len({tuple(row) for row in rows}) >= 1
    # falsification control: the pinned v7 kernel is strongly alignment dependent
    legacy = np.asarray(legacy)
    assert (legacy.max() - legacy.min()) / legacy.mean() > 0.1


def test_flat_wall_subcell_offset_gives_the_same_short_time_angle():
    """Same physics, not just same measure: two sub-cell offsets relax identically (CI-sized)."""
    angles = []
    for offset in (0.0, 0.5):
        p = pf.PhaseFieldParams(Nx=64, Ny=64, Lx=6.0, Ly=6.0, dt=4e-3, M=8e-3, eps=2.0 * 6.0 / 64)
        wall_height = 0.25 + offset * p.dy
        solid = pf.make_solid(pf.surface_flat(p, wall_height=wall_height), p, cos_theta=math.cos(math.radians(60.0)))
        state = pf.sessile_initial_state(p, solid, R=1.1, wall_height=wall_height)
        step = jax.jit(pf.phase_only_step, static_argnums=(2,))
        for _ in range(600):
            state = step(state, solid, p)
        angles.append(float(pf.measure_contact_angle(state.phi, solid, p)))
    # 600 steps of an accelerated CH-only relaxation is a proxy, not an equilibrium: the gate here
    # is that two sub-cell wall placements relax indistinguishably (the converged gate is 2 deg).
    assert abs(angles[0] - angles[1]) <= 2.0
    assert angles[0] < 90.0 and angles[1] < 90.0


# ---------------------------------------------------------------------------
#  inclined walls: Euclidean cut length, no Manhattan gain
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("slope", [-0.5, -0.25, 0.25, 0.5])
def test_inclined_wall_measure_is_euclidean_length(slope, x64):
    p = _params(128, dtype=jnp.float64)
    X, Y = pf.grids(p)
    dx = p.dx
    sdf = (Y - slope * (X - 3.0) - 0.25) / math.sqrt(1.0 + slope * slope)
    area, normal_x, normal_y, distance, _cx, _cy, info = pf.wall_cut_measure(sdf, p)
    area = np.asarray(area, dtype=np.float64)
    x_axis = (np.arange(p.Nx) + 0.5) * dx
    height = 0.25 + slope * (x_axis - 3.0)
    interior = (height > 6.0 * dx) & (height < p.Ly - 6.0 * dx)
    interior[:8] = False  # the inclined *plane* is not periodic in x: stay off the seam
    interior[-8:] = False
    first, last = int(np.argmax(interior)), int(p.Nx - np.argmax(interior[::-1]))
    measured = float(area[first:last].sum())
    euclidean = (last - first) * dx * math.sqrt(1.0 + slope * slope)
    manhattan = (last - first) * dx * (1.0 + abs(slope))
    assert measured == pytest.approx(euclidean, rel=2.0e-2)
    assert abs(measured - euclidean) < 0.5 * abs(manhattan - euclidean)  # no staircase gain
    expected_normal = np.asarray([slope, -1.0]) / math.sqrt(1.0 + slope * slope)
    positive = area[first:last] > 0.0
    normals = np.stack([np.asarray(normal_x)[first:last][positive], np.asarray(normal_y)[first:last][positive]], -1)
    assert np.allclose(normals, expected_normal, rtol=0.0, atol=1e-9)
    assert float(info["measure_conservation_error"]) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
#  seam ghosts
# ---------------------------------------------------------------------------
def test_no_periodic_seam_ghost_in_the_wall_measure(x64):
    """No wall measure at the periodic-y seam, and the x seam is not distinguished."""
    p, solid, _ = _flat(64, dtype=jnp.float64)
    area = np.asarray(solid.wall_area, dtype=np.float64)
    assert area[:, int(0.85 * p.Ny) :].max() == 0.0  # nothing near the material top boundary
    assert np.count_nonzero(area) == p.Nx  # one wall cell per column, no duplicates

    # x-seam equivariance: rolling a grid-periodic geometry rolls the measure identically
    sdf = np.asarray(pf.surface_flat(p, wall_height=0.25), dtype=np.float64)
    reference = np.asarray(pf.wall_cut_measure(jnp.asarray(sdf), p)[0], dtype=np.float64)
    for roll in (1, 5, 32):
        rolled = np.asarray(pf.wall_cut_measure(jnp.asarray(np.roll(sdf, roll, axis=0)), p)[0], dtype=np.float64)
        assert np.allclose(rolled, np.roll(reference, roll, axis=0), rtol=0.0, atol=1e-12)

    # Mutation control: a *periodically wrapped* y corner construction would invent a contour at
    # the seam (the top row of the domain is far from the wall, the bottom row is inside it), while
    # the shipped linear extrapolation does not. This keeps the ghost test from passing vacuously.
    centre = np.asarray(solid.sdf, dtype=np.float64)
    wrapped = np.concatenate([centre[:, -1:], centre, centre[:, :1]], axis=1)
    wrapped_corners = 0.25 * (
        wrapped[:, :-1] + np.roll(wrapped, 1, axis=0)[:, :-1] + wrapped[:, 1:] + np.roll(wrapped, 1, axis=0)[:, 1:]
    )
    extrapolated = np.asarray(pf.sdf_corner_values(jnp.asarray(centre), p), dtype=np.float64)
    wrapped_crossing = np.count_nonzero((wrapped_corners[:, :-1] >= 0.0) != (wrapped_corners[:, 1:] >= 0.0))
    extrapolated_crossing = np.count_nonzero((extrapolated[:, :-1] >= 0.0) != (extrapolated[:, 1:] >= 0.0))
    assert wrapped_crossing > extrapolated_crossing  # the wrap invents seam contours
    assert extrapolated_crossing == p.Nx  # exactly the one real wall crossing per column
    assert extrapolated[:, 0].max() < 0.0 < extrapolated[:, -1].min()  # extrapolated, not wrapped


def test_textured_geometry_measure_is_finite_positive_and_conserved(x64):
    """Pillars/grooves/wedge: finite, positive, localized, conserved, fluid-side, unit normals."""
    p = _params(96, dtype=jnp.float64)
    for name, kwargs in (
        ("pillars", {"wall_height": 0.25, "n_pillars": 4, "width": 0.3, "height": 0.4}),
        ("grooves", {"wall_height": 0.25, "n_grooves": 6, "width": 0.25, "depth": 0.35}),
        ("wedge", {"wall_height": 1.5, "slope": 0.5}),
    ):
        sdf = np.asarray(getattr(pf, f"surface_{name}")(p, **kwargs), dtype=np.float64)
        geometry, _geometry_info = pf.embedded_fluid_geometry(jnp.asarray(sdf), p)
        area, normal_x, normal_y, distance, _cx, _cy, info = pf.wall_cut_measure(
            jnp.asarray(sdf), p, volume=geometry.volume
        )
        area = np.asarray(area, dtype=np.float64)
        positive = area > 0.0
        assert positive.any() and np.isfinite(area).all()
        assert area[positive].min() > 0.0
        assert float(info["measure_conservation_error"]) == pytest.approx(0.0, abs=1e-12)
        # contract v9: every control cell that hosts wall measure owns a positive control volume.
        # A solid *centre* is allowed (that is exactly what a cut cell is), a solid volume is not.
        assert float(info["length_on_zero_volume_cells"]) == 0.0
        assert float(info["length_on_positive_volume_cells"]) == pytest.approx(float(area.sum()), rel=1e-12)
        # the pinned contract-v7/v8 ring relocation keeps the hard-fluid host statement
        _a8, _nx8, _ny8, _d8, _cx8, _cy8, info8 = pf.wall_cut_measure(
            jnp.asarray(sdf), p, control_cell="hard_fluid_ring", volume=geometry.volume
        )
        assert float(info8["length_on_solid_cells"]) == 0.0  # every control cell is hard fluid
        assert float(np.asarray(_a8, dtype=np.float64).sum()) == pytest.approx(float(area.sum()), rel=1e-12)
        magnitude = np.sqrt(np.asarray(normal_x) ** 2 + np.asarray(normal_y) ** 2)
        counts = np.asarray(info["segment_count_field"], dtype=np.float64)
        single = positive & (counts <= 1.0 + 1e-9)
        assert np.allclose(magnitude[single], 1.0, rtol=0.0, atol=1e-9)
        assert (magnitude[positive & ~single] <= 1.0 + 1e-9).all()  # weighted mean of two faces
        assert area[:, int(0.85 * p.Ny) :].max() == 0.0 or name == "hierarchical"
        # the curved periodic wedge measures its analytic arc length to well under a percent
        if name == "wedge":
            exact = wma._analytic_boundary_length(name, p, kwargs)
            assert area.sum() == pytest.approx(exact, rel=1e-3)


# ---------------------------------------------------------------------------
#  discrete energy and its variational derivative
# ---------------------------------------------------------------------------
def _reference_wall_energy(phi, area, cos_theta):
    """Independent transcription of F_wall^h = sum_i A_wall,i g_w(phi_i, theta)."""
    phase = np.asarray(phi, dtype=np.float64)
    h = phase**2 * (3.0 - 2.0 * phase)
    return float(np.sum(np.asarray(area, dtype=np.float64) * (-WALL_SIGMA0 * cos_theta * h)))


@pytest.mark.parametrize("theta", TARGETS)
def test_energy_directional_derivative_matches_wall_operator(theta, x64):
    """mu == d(F_bulk + F_wall^h)/dphi in float64 (gate 1e-6, ideal 1e-8)."""
    p = _params(64, dtype=jnp.float64)
    sdf = np.asarray(pf.surface_flat(p, wall_height=0.25), dtype=np.float64)
    X, Y = pf.grids(p)
    fluid = sdf >= 0.0
    # The base field carries two x wavenumbers on purpose: with a single cos(2 pi X/Lx) mode the
    # bulk directional derivative cancels exactly by symmetry at 90 deg (0/0), which would make the
    # neutral case a vacuous check.
    phi0 = np.where(
        fluid,
        np.clip(
            0.5
            + 0.3 * np.cos(2.0 * np.pi * X / p.Lx) * np.cos(np.pi * Y / p.Ly)
            + 0.1 * np.cos(4.0 * np.pi * X / p.Lx),
            0.02,
            0.98,
        ),
        0.0,
    )
    direction = np.where(fluid, 0.2 * np.sin(2.0 * np.pi * X / p.Lx) + 0.1 * np.cos(np.pi * Y / p.Ly), 0.0)
    cos_theta = math.cos(math.radians(theta))
    solid = pf.make_solid(jnp.asarray(sdf), p, cos_theta=cos_theta)
    phi = jnp.asarray(phi0)
    probe = jnp.asarray(direction)
    mu = np.asarray(pf.chemical_potential(phi, solid, p), dtype=np.float64)
    predicted = float(np.sum(mu * direction) * p.dx * p.dy)
    errors = []
    for amplitude in (1.0e-6, 1.0e-5, 1.0e-4):
        centred = (
            float(pf.phase_free_energy(phi + amplitude * probe, solid, p))
            - float(pf.phase_free_energy(phi - amplitude * probe, solid, p))
        ) / (2.0 * amplitude)
        errors.append(abs(centred - predicted) / max(abs(centred), abs(predicted), 1e-10))
    assert abs(predicted) > 1e-9  # non-degenerate for every target, including the neutral wall
    assert min(errors) <= 1e-6  # specification gate
    assert min(errors) <= 1e-8  # ideal gate


@pytest.mark.parametrize("theta", [60.0, 120.0])
def test_phase_free_energy_uses_the_same_measure_once(theta, x64):
    """``phase_free_energy`` wall term == sum_i A_wall,i g_w(phi_i), and only that."""
    p = _params(64, dtype=jnp.float64)
    sdf = np.asarray(pf.surface_flat(p, wall_height=0.25), dtype=np.float64)
    cos_theta = math.cos(math.radians(theta))
    solid = pf.make_solid(jnp.asarray(sdf), p, cos_theta=cos_theta)
    phi = jnp.asarray(np.where(sdf >= 0.0, 0.5 - 0.45 * np.tanh((sdf - 0.4) / (math.sqrt(2.0) * p.eps)), 0.0))
    area = np.asarray(solid.wall_area, dtype=np.float64)
    total = float(pf.phase_free_energy(phi, solid, p))
    wall = float(pf.wall_free_energy(phi, solid, p))
    assert wall == pytest.approx(_reference_wall_energy(phi, area, cos_theta), rel=1e-12, abs=1e-15)
    no_wall_params = _params(64, dtype=jnp.float64, wetting_model="none")
    no_wall_solid = pf.make_solid(jnp.asarray(sdf), no_wall_params, cos_theta=cos_theta)
    bulk_only = float(pf.phase_free_energy(phi, no_wall_solid, no_wall_params))
    assert total == pytest.approx(bulk_only + wall, rel=1e-12, abs=1e-15)
    # the legacy diffuse kernel would give a different (smaller, alignment-dependent) wall energy
    legacy_params = _params(64, dtype=jnp.float64, wall_measure="diffuse_sdf_v7")
    legacy_solid = pf.make_solid(jnp.asarray(sdf), legacy_params, cos_theta=cos_theta)
    legacy_wall = float(pf.wall_free_energy(phi, legacy_solid, legacy_params))
    assert legacy_wall != pytest.approx(wall, rel=1e-3)


def test_production_wall_energy_is_not_double_counted(x64):
    """One contribution: wetting_mu is exactly zero and the flux equals A g_w'/(dx dy)."""
    p = _params(64, dtype=jnp.float64)
    sdf = np.asarray(pf.surface_flat(p, wall_height=0.25), dtype=np.float64)
    solid = pf.make_solid(jnp.asarray(sdf), p, cos_theta=math.cos(math.radians(60.0)))
    phi = jnp.asarray(np.where(sdf >= 0.0, 0.5 - 0.45 * np.tanh((sdf - 0.4) / (math.sqrt(2.0) * p.eps)), 0.0))
    assert float(jnp.max(jnp.abs(pf.wetting_mu(phi, solid, p)))) == 0.0
    explicit = np.asarray(pf._explicit_chemical_potential(phi, solid, p), dtype=np.float64)
    bulk = np.asarray(pf.fprime(phi) / p.eps, dtype=np.float64)
    phase = np.asarray(phi, dtype=np.float64)
    expected = (
        (-WALL_SIGMA0 * math.cos(math.radians(60.0)) * 6.0 * phase * (1.0 - phase))
        * np.asarray(solid.wall_area, dtype=np.float64)
        / (p.dx * p.dy)
    )
    assert np.max(np.abs(explicit - bulk - expected)) <= 1e-12 * max(np.max(np.abs(expected)), 1.0)
    # mutation: adding the legacy kernel on top must be visible (the check is not vacuous)
    double = explicit + np.asarray(
        pf.wall_energy_derivative(phi, solid.cos_theta) * pf.wall_delta(solid.sdf, p), dtype=np.float64
    )
    assert np.max(np.abs(double - explicit)) > 1e-3 * np.max(np.abs(expected))


# ---------------------------------------------------------------------------
#  first-fluid-layer Young residual
# ---------------------------------------------------------------------------
def _first_layer(N, theta, *, eps_factor=2.0, wall_theta=None, offset=0.0, slope=0.0):
    p = _params(N, eps_factor=eps_factor, dtype=jnp.float64)
    dx = p.dx
    x_axis = (np.arange(N) + 0.5) * dx
    X, Y = np.meshgrid(x_axis, x_axis, indexing="ij")
    if slope:
        sdf = (Y - slope * (X - 3.0) - 0.25) / math.sqrt(1.0 + slope * slope)
    else:
        sdf = Y - (0.25 + offset * dx)
    phi = clk.manufactured_young_boundary_field(X, Y, sdf, p.eps, theta)
    solid = pf.make_solid(jnp.asarray(sdf, dtype=np.float64), p, cos_theta=math.cos(math.radians(theta)))
    return clk.young_boundary_residual_first_layer(
        phi,
        sdf,
        dx,
        dx,
        p.eps,
        math.cos(math.radians(theta if wall_theta is None else wall_theta)),
        wall_area=np.asarray(solid.wall_area, dtype=np.float64),
        wall_normal_x=np.asarray(solid.wall_normal_x, dtype=np.float64),
        wall_normal_y=np.asarray(solid.wall_normal_y, dtype=np.float64),
    )


def test_first_layer_residual_manufactured_flat_wall(x64):
    """90 deg: both terms of R_Y vanish identically, so the residual is round-off."""
    result = _first_layer(96, 90.0)
    assert result["n_wall_cells"] == 96 and result["n_wall_cells_nonfluid"] == 0
    assert result["RY_first_linf"] <= 1e-12
    assert result["wall_measure_weighted_RY"] <= 1e-12


@pytest.mark.parametrize("theta", [60.0, 120.0, 150.0])
def test_first_layer_residual_manufactured_non_neutral(theta, x64):
    result = _first_layer(96, theta)
    assert result["n_wall_cells_band"] > 0
    assert result["RY_first_normalized_l2"] <= 5.0e-2  # second-order stencil truncation at eps/dx = 2
    assert result["wall_measure_weighted_RY"] <= 5.0e-2 * WALL_SIGMA0 * 6.0
    # it is a discretization error, not an inconsistency: a thicker interface reduces it
    thick = _first_layer(96, theta, eps_factor=4.0)
    assert thick["RY_first_normalized_l2"] < result["RY_first_normalized_l2"]


@pytest.mark.parametrize("theta", TARGETS)
def test_first_layer_residual_is_shift_invariant(theta, x64):
    """The residual must not depend on where the wall falls inside a cell."""
    values = [_first_layer(96, theta, offset=offset)["RY_first_normalized_l2"] for offset in (0.0, 0.5)]
    assert max(values) <= 5.0e-2
    if abs(theta - 90.0) > 1e-9:
        assert abs(values[0] - values[1]) <= 2.0e-2


@pytest.mark.parametrize("slope", [-0.45, 0.45])
def test_first_layer_residual_manufactured_inclined_wall(slope, x64):
    """Both inclination signs: the measure-weighted normal makes the residual small."""
    p = _params(96, dtype=jnp.float64)
    sdf = np.asarray(
        (pf.grids(p)[1] - slope * (pf.grids(p)[0] - 3.0) - 0.25) / math.sqrt(1.0 + slope * slope), dtype=np.float64
    )
    X = np.asarray(pf.grids(p)[0])
    # Tangentially uniform manufactured layer: phi = 0.5 (1 - tanh(sdf cos(theta)/(sqrt(2) eps)))
    # has grad(phi) parallel to grad(sdf), so n.grad(phi) = -dphi/ds and the *nonlinear* natural
    # condition holds identically for any wall inclination. The footprint variant used for flat
    # walls also varies along x, which an inclined normal would pick up as a spurious residual.
    phi = 0.5 * (1.0 - np.tanh(sdf * math.cos(math.radians(120.0)) / (math.sqrt(2.0) * p.eps)))
    phi = np.where(sdf >= 0.0, phi, 0.0)
    solid = pf.make_solid(jnp.asarray(sdf), p, cos_theta=math.cos(math.radians(120.0)))
    area = np.asarray(solid.wall_area, dtype=np.float64)
    # The manufactured footprint puts its contact lines at x = 3.0 +/- 1.1; pick the one whose wall
    # is inside the domain for this slope (the inclined plane leaves the domain on the other side).
    centre = 1.9 if slope < 0 else 4.1
    region = (np.asarray(X) > centre - 0.4) & (np.asarray(X) < centre + 0.4)
    local = np.zeros_like(area)
    local[region] = area[region]
    result = clk.young_boundary_residual_first_layer(
        phi,
        sdf,
        p.dx,
        p.dy,
        p.eps,
        math.cos(math.radians(120.0)),
        wall_area=local,
        wall_normal_x=np.asarray(solid.wall_normal_x, dtype=np.float64),
        wall_normal_y=np.asarray(solid.wall_normal_y, dtype=np.float64),
    )
    assert result["n_wall_cells_band"] > 0
    assert result["RY_first_normalized_l2"] <= 5.0e-2  # second-order stencil truncation only


def test_first_layer_residual_detects_a_wrong_wall_angle(x64):
    """Non-vacuity: feeding the wrong Young angle must be loud."""
    assert _first_layer(96, 60.0, wall_theta=60.0)["RY_first_normalized_l2"] < 0.05
    assert _first_layer(96, 60.0, wall_theta=90.0)["RY_first_normalized_l2"] > 0.2
    assert _first_layer(96, 150.0, wall_theta=60.0)["RY_first_normalized_l2"] > 0.2


def test_first_layer_residual_uses_the_new_measure_not_the_old_band(x64):
    """The retained two-cell-band residual is kept for comparison and is a different metric."""
    N, theta = 96, 120.0
    p = _params(N, dtype=jnp.float64)
    dx = p.dx
    x_axis = (np.arange(N) + 0.5) * dx
    X, Y = np.meshgrid(x_axis, x_axis, indexing="ij")
    sdf = Y - 0.25
    phi = clk.manufactured_young_boundary_field(X, Y, sdf, p.eps, theta)
    solid = pf.make_solid(jnp.asarray(sdf, dtype=np.float64), p, cos_theta=math.cos(math.radians(theta)))
    first = clk.young_boundary_residual_first_layer(
        phi,
        sdf,
        dx,
        dx,
        p.eps,
        math.cos(math.radians(theta)),
        wall_area=np.asarray(solid.wall_area, dtype=np.float64),
        wall_normal_x=np.asarray(solid.wall_normal_x, dtype=np.float64),
        wall_normal_y=np.asarray(solid.wall_normal_y, dtype=np.float64),
    )
    band = clk.young_boundary_residual(phi, sdf, dx, dx, p.eps, math.cos(math.radians(theta)))
    assert first["n_wall_cells"] == N  # one control cell per column
    assert band["n_points"] > first["n_wall_cells_band"]  # the band is wider than the first layer
    for key in (
        "RY_first_l2",
        "RY_first_linf",
        "RY_first_normalized_l2",
        "RY_first_normalized_linf",
        "wall_measure_weighted_RY",
        "n_wall_cells",
    ):
        assert key in first and np.isfinite(first[key])
    for key in ("RY_l2", "RY_normalized_l2", "n_points"):
        assert key in band


# ---------------------------------------------------------------------------
#  no empirical correction: geometry only, angle independent
# ---------------------------------------------------------------------------
def test_wall_measure_is_angle_independent_and_has_no_global_share_gain(x64):
    p = _params(64, dtype=jnp.float64)
    sdf = np.asarray(pf.surface_flat(p, wall_height=0.25), dtype=np.float64)
    reference = None
    fields = {}
    for theta in TARGETS:
        cos_theta = math.cos(math.radians(theta))
        solid = pf.make_solid(jnp.asarray(sdf), p, cos_theta=cos_theta)
        area = np.asarray(solid.wall_area, dtype=np.float64)
        if reference is None:
            reference = area
        assert np.array_equal(area, reference)  # geometry does not know the angle
        assert area.sum() == pytest.approx(p.Lx, rel=1e-12)  # no 1/f gain, no multiplier
        phi = jnp.asarray(np.where(sdf >= 0.0, 0.5 - 0.4 * np.tanh((sdf - 0.4) / (math.sqrt(2.0) * p.eps)), 0.0))
        fields[theta] = np.asarray(
            pf.wall_energy_derivative(phi, solid.cos_theta) * area / (p.dx * p.dy), dtype=np.float64
        )
    # the only angle dependence is the physical Young cosine: mu_wall/cos(theta) is one field
    base = fields[60.0] / math.cos(math.radians(60.0))
    for theta in (120.0, 150.0):
        assert np.allclose(fields[theta] / math.cos(math.radians(theta)), base, rtol=1e-12, atol=1e-15)
    assert float(np.max(np.abs(fields[90.0]))) <= 1.0e-15  # cos(90 deg) is round-off, not exact zero


def test_g_w_and_sigma0_are_unchanged(x64):
    """The wall energy itself must not be touched to fix the contact angle."""
    assert pf.WALL_SIGMA0 == pytest.approx(math.sqrt(2.0) / 6.0, rel=0.0, abs=0.0)
    for phi in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert float(pf.wall_switch(jnp.asarray(phi))) == pytest.approx(phi**2 * (3.0 - 2.0 * phi), rel=1e-15)
        assert float(pf.wall_switch_derivative(jnp.asarray(phi))) == pytest.approx(6.0 * phi * (1.0 - phi), rel=1e-15)
    for theta in TARGETS:
        cos_theta = math.cos(math.radians(theta))
        g0 = float(pf.wall_energy_density(jnp.asarray(0.0), cos_theta))
        g1 = float(pf.wall_energy_density(jnp.asarray(1.0), cos_theta))
        assert g0 == pytest.approx(0.0, abs=1e-16)
        assert g0 - g1 == pytest.approx(WALL_SIGMA0 * cos_theta, rel=1e-14)


FORBIDDEN_IDENTIFIERS = {
    "wall_gain",
    "contact_angle_gain",
    "theta_scale",
    "cos_theta_scale",
    "angle_lookup",
    "cos_theta_over_f",
    "wall_measure_gain",
    "fitted_contact_angle",
    "contact_angle_correction",
    "young_calibration",
}


def _code_identifiers(path: Path) -> set[str]:
    """Every identifier in the *code* of a module (docstrings and comments are not code)."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.keyword) and node.arg is not None:
            names.add(node.arg)
    return names


def test_no_empirical_contact_angle_correction_in_the_production_path():
    """No gain, remap, lookup or fitted factor may exist in the solver or its audits."""
    for name in ("phasefield.py", "production/wall_measure_audit.py", "production/embedded_young_audit.py"):
        identifiers = _code_identifiers(HERE / name)
        assert not (identifiers & FORBIDDEN_IDENTIFIERS), (name, sorted(identifiers & FORBIDDEN_IDENTIFIERS))
    # the production parameter set has no angle-calibration knob at all
    fields = {f.name for f in pf.PhaseFieldParams.__dataclass_fields__.values()}
    assert not (fields & FORBIDDEN_IDENTIFIERS)
    assert "wall_measure" in fields and "wall_delta_width" in fields
    # the diagnostic-only ablation knob lives in the audit runner, never in the solver
    assert "wall_gain" not in fields
    runner = (HERE / "production" / "nonneutral_wetting_audit.py").read_text()
    assert "wall_gain" in runner and "diagnostic ablation knob" in runner


def test_wall_measure_option_fails_closed_and_legacy_is_pinned(x64):
    with pytest.raises(ValueError, match="wall_measure"):
        pf.PhaseFieldParams(Nx=16, Ny=16, wall_measure="cos_theta_over_f")
    p = _params(64, dtype=jnp.float64, wall_measure="diffuse_sdf_v7")
    sdf = np.asarray(pf.surface_flat(p, wall_height=0.25), dtype=np.float64)
    solid = pf.make_solid(jnp.asarray(sdf), p, cos_theta=math.cos(math.radians(60.0)))
    legacy_density = np.asarray(pf.wall_measure_density(solid, p), dtype=np.float64)
    assert np.allclose(legacy_density, np.asarray(pf.wall_delta(solid.sdf, p), dtype=np.float64), rtol=0.0, atol=0.0)
    production = _params(64, dtype=jnp.float64)
    production_solid = pf.make_solid(jnp.asarray(sdf), production, cos_theta=math.cos(math.radians(60.0)))
    cut_density = np.asarray(pf.wall_measure_density(production_solid, production), dtype=np.float64)
    # contract v9: the density is wall length per unit *control volume* (A_wall,i / V_i), so the
    # volume-weighted integral reproduces the geometric wall length exactly -- including on the
    # cut row, where V_i < dx dy and the density is correspondingly larger than 1/dy.
    volume = np.asarray(production_solid.geometry.volume, dtype=np.float64)
    assert float((cut_density * volume).sum()) == pytest.approx(production.Lx, rel=1e-12)
    assert float(np.asarray(production_solid.wall_area, dtype=np.float64).sum()) == pytest.approx(
        production.Lx, rel=1e-12
    )
    legacy_total = float(legacy_density.sum()) * p.dx * p.dy
    legacy_fluid = float(np.sum(legacy_density[sdf >= 0.0])) * p.dx * p.dy
    # The falsified root cause: only the fluid half of the two-sided kernel acts on the transported
    # phase field, so the effective wall measure is f * (wall length) with f far from 1 and grid
    # dependent, while the production measure is the wall length exactly.
    assert legacy_total == pytest.approx(production.Lx, rel=0.05)  # the kernel is normalized overall
    assert legacy_fluid < 0.75 * production.Lx
    assert abs(legacy_fluid - production.Lx) / production.Lx > 0.2
    # legacy reproduction modes keep their historical measure pinned
    v6 = _params(64, dtype=jnp.float64, wetting_model="surface_energy_volume_v6")
    pf.make_solid(jnp.asarray(sdf), v6, cos_theta=math.cos(math.radians(60.0)))
    assert pf.wall_measure_is_cutcell(v6) is False
    assert pf.wall_measure_is_cutcell(production) is True


# ---------------------------------------------------------------------------
#  contract v8 / v7 staleness
# ---------------------------------------------------------------------------
def test_contract11_metadata_and_contract10_staleness():
    """Contract 11 retains v9 geometry/v10 solve semantics and adds A1 storage lineage."""
    assert pf.SOLVER_CONTRACT_VERSION == 11
    assert pf.WALL_MEASURE_METHOD == "sdf_cutcell_v1"
    assert pf.WALL_MEASURE_CONTRACT_VERSION == 1
    assert pf.PHASE_TRANSPORT_GEOMETRY == "sdf_cutcell_fv_v1"
    import generate_dataset as G

    payload_defaults = {
        "wall_measure": pf.WALL_MEASURE_METHOD,
        "wall_measure_contract_version": pf.WALL_MEASURE_CONTRACT_VERSION,
        "solver_contract": int(pf.SOLVER_CONTRACT_VERSION),
    }
    assert payload_defaults["solver_contract"] == 11
    source = (HERE / "generate_dataset.py").read_text()
    assert 'wall_measure=str(case.get("wall_measure", pf.WALL_MEASURE_METHOD))' in source
    assert '"wall_measure_method": str(params.wall_measure)' in source
    # contract v9: the fingerprints/manifests additionally carry the phase-transport metadata, so a
    # contract-8 staircase dataset is stale by version and by geometry semantics at once
    assert "phase_transport_metadata" in source
    assert G.DATASET_SCHEMA_VERSION == 3  # the .npz schema itself is unchanged
    assert "DATASET_SCHEMA_VERSION" in source


def test_validation_report_records_the_wall_measure_lineage():
    from production.report import KNOWN_RESOLVED_STATUSES, KNOWN_SOLVER_CONTRACT_VERSIONS

    assert 8 in KNOWN_SOLVER_CONTRACT_VERSIONS
    assert "resolved_in_contract_v8" in KNOWN_RESOLVED_STATUSES
    from production.validation import KNOWN_SOLVER_BLOCKERS

    ids = {item["id"] for item in KNOWN_SOLVER_BLOCKERS}
    assert "N-CH-MASS-PRECISION" in ids  # precision blocker stays on the record
    status = {item["id"]: item["status"] for item in KNOWN_SOLVER_BLOCKERS}
    # Contract 11 selects A1 to remove the contract-10 storage loss and solve defect; the Stage-2
    # quick/medium/long and production-default CHNS evidence is still a measurement requirement.
    assert status["N-CH-MASS-PRECISION"] == "measurement_required"
    assert status["I-CONTACT-GAP"] == "measurement_required"  # kept independent of wetting
    assert status["W-CONTACT-ANGLE"] == "measurement_required"  # not self-declared resolved


# ---------------------------------------------------------------------------
#  small CH-only relaxation moves in the correct direction
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("theta", [60.0, 120.0, 150.0])
def test_ch_only_nonneutral_motion_is_toward_the_target(theta):
    """CI-sized CH-only check: the sign of the motion is the Young sign (not a magnitude gate)."""
    p = pf.PhaseFieldParams(Nx=64, Ny=64, Lx=6.0, Ly=6.0, dt=4e-3, M=8e-3, eps=2.0 * 6.0 / 64)
    solid = pf.make_solid(pf.surface_flat(p, wall_height=0.25), p, cos_theta=math.cos(math.radians(theta)))
    state = pf.sessile_initial_state(p, solid, R=1.1, wall_height=0.25)
    assert float(jnp.sum(solid.wall_area)) == pytest.approx(p.Lx, rel=1e-5)
    initial_angle = float(pf.measure_contact_angle(state.phi, solid, p))
    step = jax.jit(pf.phase_only_step, static_argnums=(2,))
    widths = []
    energies = [float(pf.phase_free_energy(state.phi, solid, p))]
    for index in range(600):
        state = step(state, solid, p)
        if index + 1 == 200:
            widths.append(float(pf.spreading_width(state.phi, p)))
        energies.append(float(pf.phase_free_energy(state.phi, solid, p)))
    final_angle = float(pf.measure_contact_angle(state.phi, solid, p))
    widths.append(float(pf.spreading_width(state.phi, p)))
    assert abs(initial_angle - 90.0) < 1.0  # the seed is the target-independent 90 deg cap
    if theta < 90.0:
        assert final_angle < initial_angle  # hydrophilic: the angle decreases
        assert widths[-1] >= widths[0]  # and the drop spreads
    else:
        assert final_angle > initial_angle  # hydrophobic: the angle increases
        assert widths[-1] <= widths[0]  # and the contact width retracts
    scale = max(1.0, abs(energies[0]))
    assert max((b - a) / scale for a, b in zip(energies, energies[1:])) <= 1.0e-5  # CH energy nonincreasing
    assert energies[-1] < energies[0]


def test_l1a2d_root_cause_diagnostic_is_still_available(x64):
    """The falsification diagnostic f stays in the tree, is diagnostic-only, and is not 1.0."""
    shares = {}
    for N in (64, 96, 128, 192):
        p = pf.PhaseFieldParams(Nx=N, Ny=N, Lx=6.0, Ly=6.0)
        sdf = np.asarray(pf.surface_flat(p, wall_height=0.25), dtype=np.float64)
        info = clk.fluid_wall_delta_integral(sdf, p.dx, p.dy)
        shares[N] = info["fluid_fraction_of_wall_kernel"]
    assert shares[128] == pytest.approx(0.614, abs=2e-3)
    assert shares[96] == pytest.approx(0.500, abs=5e-3)
    assert shares[64] < 0.45 and shares[192] == pytest.approx(0.500, abs=5e-3)
    # production never reads f: the measure is the geometric length, not the kernel share
    p = _params(128, dtype=jnp.float64)
    solid = pf.make_solid(jnp.asarray(pf.surface_flat(p, wall_height=0.25), dtype=np.float64), p, cos_theta=0.5)
    assert float(np.sum(solid.wall_area)) == pytest.approx(p.Lx, rel=1e-12)


def test_wall_measure_audit_module_passes_its_own_gates(x64):
    """The CI-sized wall-measure audit (geometry + variational + first layer) passes end to end."""
    audit = wma.run_audit(quick=True)
    failed = [check.name for check in audit.checks if not check.passed]
    assert failed == []
    payload = audit.to_dict()
    assert payload["solver_contract_version"] == 11
    assert payload["wall_measure_method"] == "sdf_cutcell_v1"
    json.dumps(payload, allow_nan=False)  # strict JSON


def test_example_config_documents_the_l1a2e_profile():
    """The shipped example config must match the runner's frozen baseline profile and gates."""
    from production import embedded_young_audit as eya

    cfg = json.loads((HERE / "production" / "configs" / "embedded_young.example.json").read_text())
    base = eya.PROFILES["baseline"]
    assert cfg["stage"] == "L1A-2e"
    assert cfg["solver_contract_version"] == 8  # frozen L1A-2e profile document
    assert cfg["wall_measure_method"] == pf.WALL_MEASURE_METHOD == "sdf_cutcell_v1"
    assert cfg["wall_measure_contract_version"] == pf.WALL_MEASURE_CONTRACT_VERSION == 1
    assert cfg["trajectory_semantics_changed"] is True
    assert cfg["base_commit"] == "d53015849160ebfd51a77fcc108635dc763bca62"
    assert cfg["targets_deg"] == base["targets"] and cfg["stage_step_budgets"] == base["budgets"]
    assert cfg["base_case"]["N"] == base["N"] and cfg["base_case"]["dt"] == base["dt"]
    assert cfg["sections"]["translation"]["offsets_over_dy"] == base["translation"]["offsets"]
    assert cfg["sections"]["resolution"]["measure_N_values"] == base["resolution"]["N_values"]
    assert cfg["gates"] == eya.GATES
    assert cfg["gates"]["wall_measure_translation_spread_limit"] == 0.01
    assert cfg["gates"]["translation_angle_spread_deg"] == 2.0
    assert cfg["gates"]["ch_only_mae_deg"] == 5.0 and cfg["gates"]["ch_only_max_error_deg"] == 10.0
    assert cfg["gates"]["strong_60_120_error_deg"] == 5.0 and cfg["gates"]["strong_150_error_deg"] == 8.0
    assert cfg["gates"]["formal_mass_drift"] == 1.0e-3
    assert cfg["geometry_gates"]["directional_derivative_tolerance"] == 1.0e-6
    assert cfg["geometry_gates"]["directional_derivative_ideal"] == 1.0e-8
    # the forbidden-correction list is part of the contract document, not just prose
    assert "cos(theta)/f" in cfg["forbidden_production_corrections"]
    assert "global wall gain" in cfg["forbidden_production_corrections"]
    exceptions = cfg["preflight_exceptions"]
    assert exceptions["PREFLIGHT_EXCEPTION_PREVIOUS_REVIEW"] == "AUTHORIZED"
    assert exceptions["PREFLIGHT_EXCEPTION_DEPENDENCY_AUDIT"] == "AUTHORIZED_FOR_DEVELOPMENT"
    assert "not a physics acceptance waiver" in exceptions["scope"]


def test_cutcell_alignment_config_documents_the_l1a2f_profile():
    """The L1A-2f example config must match the runner's frozen baseline profile and gates."""
    from production import cutcell_alignment_audit as caa

    cfg = json.loads((HERE / "production" / "configs" / "cutcell_alignment.example.json").read_text())
    base = caa.PROFILES["baseline"]
    assert cfg["stage"] == "L1A-2f"
    assert cfg["solver_contract_version"] == pf.SOLVER_CONTRACT_VERSION == 11
    assert cfg["phase_transport_geometry"] == pf.PHASE_TRANSPORT_GEOMETRY == "sdf_cutcell_fv_v1"
    assert cfg["phase_control_volume"] == pf.PHASE_CONTROL_VOLUME == "partial_cell_volume"
    assert cfg["phase_face_aperture"] == pf.PHASE_FACE_APERTURE == "partial_open_length"
    assert cfg["phase_advection_subcycling"] == pf.PHASE_ADVECTION_SUBCYCLING == "disabled"
    assert cfg["trajectory_semantics_changed"] is True
    assert cfg["base_case"]["N"] == base["N"] and cfg["base_case"]["dt"] == base["dt"]
    assert cfg["base_case"]["R"] == base["R"] and cfg["base_case"]["eps_factor"] == base["eps_factor"]
    assert cfg["base_case"]["M_ref"] == 2.0e-3
    assert cfg["targets_deg"] == base["targets"]
    assert cfg["sections"]["translation"]["offsets_over_dy"] == base["translation"]["offsets"]
    assert cfg["sections"]["translation"]["targets_deg"] == base["translation"]["targets"]
    assert cfg["sections"]["translation"]["budgets"] == base["budgets"]
    assert cfg["sections"]["translation"]["neutral_budget"] == base["neutral_budget"]
    assert cfg["sections"]["resolution"]["equilibrium_N_values"] == base["resolution"]["N_values"]
    assert cfg["sections"]["resolution"]["targets_deg"] == base["resolution"]["targets"]
    assert cfg["sections"]["translation_v8"]["offsets_over_dy"] == base["translation_v8"]["offsets"]
    assert cfg["sections"]["translation_v8"]["targets_deg"] == base["translation_v8"]["targets"]
    assert cfg["sections"]["primary_float64"]["targets_deg"] == base["float64"]["targets"]
    assert cfg["sections"]["primary_float64"]["ch_solver_rtol"] == base["float64"]["rtol"]
    assert cfg["sections"]["precision"]["target_deg"] == base["precision"]["target"]
    assert cfg["sections"]["precision"]["fixed_steps"] == base["precision"]["fixed_steps"]
    assert cfg["sections"]["precision"]["mobility_factor"] == base["precision"]["mobility_factor"]
    assert cfg["sections"]["laplace"]["N"] == base["laplace"]["N"]
    assert cfg["sections"]["laplace"]["radii"] == base["laplace"]["radii"]
    assert [case["name"] for case in cfg["sections"]["impact"]["cases"]] == [
        case["name"] for case in base["impact"]["cases"]
    ]
    assert cfg["evidence_runner"] == "production.cutcell_alignment_audit"
    assert cfg["evidence_sections"] == list(caa.SECTIONS)
    for name in ("production.cutcell_geometry_audit", "production.cutcell_phase_transport_audit"):
        assert name in cfg["closure_audits"]
    for key, value in caa.GATES.items():
        if key in cfg["gates"]:
            assert cfg["gates"][key] == pytest.approx(value) if isinstance(value, float) else True
    assert cfg["gates"]["translation_angle_spread_deg"] == 2.0
    assert cfg["gates"]["translation_angle_spread_strong_deg"] == 1.0
    assert cfg["gates"]["resolution_angle_spread_deg"] == 2.0
    assert cfg["gates"]["falsification_max_abs_slope_deg_per_cell"] == 0.5
    assert cfg["gates"]["cg_relative_residual_float32"] == 1.0e-6
    assert cfg["gates"]["cg_relative_residual_float64"] == 1.0e-8
    assert cfg["gates"]["cutcell_advective_cfl_ratio_limit"] == 1.0
    assert cfg["geometry_gates"]["cutcell_flat_wall_area_relative_error"] == 1.0e-12
    assert cfg["geometry_gates"]["empty_solid_degeneracy_exact"] is True
    # the same forbidden-correction statement as the L1A-2e config
    assert "cos(theta)/f" in cfg["forbidden_production_corrections"]
    assert "global wall gain" in cfg["forbidden_production_corrections"]


def test_embedded_young_formal_mass_gate_uses_cutcell_invariant_and_full_convergence():
    """A drifting hard-mask diagnostic is not a phase-mass failure or a fake equilibrium angle."""
    from production import embedded_young_audit as eya

    cases = [
        {
            "target_deg": target,
            "converged": True,
            "steps": 50_000,
            "stop_reason": "converged",
            "equilibrium_angle_deg": target,
            "final_sampled_angle_deg": target,
            "mass_drift": 0.25,  # hard-mask diagnostic, deliberately large
            "mass_drift_final": 0.25,
            "conserved_mass_drift": 1.0e-5,
            "conserved_mass_drift_final": 1.0e-5,
            "solid_phase_fraction_max": 0.0,
            "free_energy_monotonic_violations": 0,
            "free_energy_initial": 1.0,
            "free_energy_final": 1.0,
            "RY_first_normalized_l2": 0.0,
            "RY_normalized_l2": 0.0,
            "wall_measure_weighted_RY": 0.0,
            "implicit_iterations_max": 10,
            "implicit_residual_max": 1.0e-8,
            "wall_area_total": 6.0,
            "wall_area_relative_error": 0.0,
        }
        for target in eya.FORMAL_ANGLE_TARGETS
    ]
    gates, derived = eya.evaluate_gates(
        {"primary_float64": cases},
        {"translation_measure": [], "inclined": []},
        dict(eya.PROFILES["baseline"]),
    )
    by_name = {gate["gate"]: gate for gate in gates}
    assert by_name["ch_only_four_targets_float64"]["passed"] is True
    assert by_name["formal_mass_drift_float64"]["passed"] is True
    assert derived["primary_float64"]["max_conserved_mass_drift"] == pytest.approx(1.0e-5)
    assert derived["primary_float64"]["max_hard_mask_mass_drift_diagnostic"] == pytest.approx(0.25)

    # The four-angle equilibrium gate must reject even a neutral case with a plausible final sample
    # if it never passed the stationarity window.
    unconverged = [dict(case) for case in cases]
    unconverged[1].update(converged=False, equilibrium_angle_deg=None, stop_reason="budget_exhausted")
    failed_gates, _ = eya.evaluate_gates(
        {"primary_float64": unconverged},
        {"translation_measure": [], "inclined": []},
        dict(eya.PROFILES["baseline"]),
    )
    failed = {gate["gate"]: gate for gate in failed_gates}
    assert failed["ch_only_four_targets_float64"]["passed"] is False


def test_embedded_young_quick_profile_runs_end_to_end(tmp_path, x64):
    """The evidence runner's quick profile produces a schema-valid report and evaluates its gates."""
    from production import embedded_young_audit as eya

    report = eya.run_audit("quick", tmp_path / "l1a2e_quick", overwrite=True)
    assert eya.validate_report(report) == []
    assert report["stage"] == "L1A-2e"
    # the L1A-2e runner now runs under contract 9: its sections are re-measured on cut-cell control
    # volumes, and the report carries the v9 transport metadata
    assert report["solver_contract_version"] == 11
    assert report["phase_transport_geometry"] == "sdf_cutcell_fv_v1"
    assert report["phase_control_volume"] == "partial_cell_volume"
    assert report["trajectory_semantics_changed"] is True
    assert report["preflight_exceptions"]["PREFLIGHT_EXCEPTION_PREVIOUS_REVIEW"] == "AUTHORIZED"
    assert report["preflight_exceptions"]["PREFLIGHT_EXCEPTION_DEPENDENCY_AUDIT"] == "AUTHORIZED_FOR_DEVELOPMENT"
    gates = {gate["gate"]: gate for gate in report["gates"]}
    assert gates["wall_measure_translation_spread"]["passed"] is True
    assert gates["inclined_wall_measure_error"]["passed"] is True
    assert (tmp_path / "l1a2e_quick" / "embedded_young_report.json").exists()
    assert (tmp_path / "l1a2e_quick" / "wall_measure_audit.json").exists()
    # fail-closed: an equilibrium angle without convergence is rejected
    broken = json.loads(json.dumps(report))
    for cases in broken["cases"].values():
        for case in cases:
            case["converged"] = False
            case["equilibrium_angle_deg"] = 70.0  # an equilibrium angle claimed without converging
    assert any("without converging" in error for error in eya.validate_report(broken))
    broken = json.loads(json.dumps(report))
    for cases in broken["cases"].values():
        for case in cases:
            case["wall_measure_method"] = "cos_theta_over_f"
    assert any("wall_measure_method" in error for error in eya.validate_report(broken))
    broken = json.loads(json.dumps(report))
    broken["solver_contract_version"] = 7
    assert any("contract" in error for error in eya.validate_report(broken))
    # a contract-8 report may not claim the v9 transport geometry
    broken = json.loads(json.dumps(report))
    broken["solver_contract_version"] = 8
    assert any("contract" in error for error in eya.validate_report(broken))
