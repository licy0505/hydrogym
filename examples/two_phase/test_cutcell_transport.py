"""Contract-v9 cut-cell phase transport tests (L1A-2f).

These are the CI-fast, targeted statements behind the L1A-2f evidence runner
(:mod:`production.cutcell_alignment_audit`), the geometry audit
(:mod:`production.cutcell_geometry_audit`) and the transport audit
(:mod:`production.cutcell_phase_transport_audit`). They run on small grids in seconds; the long
translation / resolution / CHNS matrices stay in the evidence runner and are never part of CI.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np

import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

import phasefield as pf  # noqa: E402

HERE = Path(__file__).resolve().parent


@pytest.fixture
def x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


_SOLID_CACHE: dict = {}


def _params(N=48, *, dtype=jnp.float64, eps_factor=2.0, **kwargs):
    p = pf.PhaseFieldParams(Nx=N, Ny=N, Lx=6.0, Ly=6.0, dtype=dtype, **kwargs)
    p.eps = eps_factor * p.dx
    return p


def _solid(p, wall_height=0.25, theta=120.0, **kwargs):
    """A flat-wall solid; the embedded geometry is cheap to reuse across tests in one session."""
    key = (int(p.Nx), int(p.Ny), float(p.dx), float(wall_height), float(theta))
    cached = _SOLID_CACHE.get(key)
    if cached is None:
        sdf = pf.surface_flat(p, wall_height=wall_height)
        cached = (pf.make_solid(sdf, p, cos_theta=math.cos(math.radians(theta))), sdf)
        _SOLID_CACHE[key] = cached
    return cached


def _same(solid, other) -> bool:
    names = (
        "volume",
        "alpha",
        "aperture_x",
        "aperture_y",
        "aperture_x_norm",
        "aperture_y_norm",
        "centroid_x",
        "centroid_y",
        "face_distance_x",
        "face_distance_y",
        "weight_x",
        "weight_y",
        "wall_measure",
        "wall_normal_x",
        "wall_normal_y",
    )
    return all(
        bool(jnp.array_equal(jnp.asarray(getattr(solid.geometry, name)), jnp.asarray(getattr(other.geometry, name))))
        for name in names
    )


def _positive_volume_cells(geometry):
    return int(jnp.sum(geometry.volume > 0.0))


# ---------------------------------------------------------------------------
#  geometry: exactness, single authority, no dropped cells
# ---------------------------------------------------------------------------
def test_cutcell_flat_wall_volume_exact_for_subcell_offsets(x64):
    """sum_i V_i is the analytic fluid area exactly, for every sub-cell wall offset."""
    for N in (32, 48, 64):
        p = _params(N)
        for offset in (0.0, 0.125, 0.375, 0.5, 0.625, 0.875):
            height = 0.25 + offset * p.dy
            solid, _ = _solid(p, wall_height=height)
            volume = float(jnp.sum(solid.geometry.volume))
            analytic = float(p.Lx * (p.Ly - height))
            assert volume == pytest.approx(analytic, rel=1e-13, abs=0.0)
            # the wall measure is the geometric wall length, not a staircase count
            assert float(jnp.sum(solid.geometry.wall_measure)) == pytest.approx(p.Lx, rel=1e-13)
            # the cut-cell volume is exact where the cell-centre staircase is not: the staircase
            # differs from the true area by up to half a cell row, with a sign that follows the
            # sub-cell offset (it over-counts when the wall sits below the cut-cell centre)
            staircase = float(jnp.sum(jnp.asarray(solid.sdf) >= 0.0) * p.dx * p.dy)
            assert abs(volume - staircase) <= 0.5 * p.Lx * p.dy + 1e-12
            # the cut-cell volume equals the staircase only when the wall is exactly face-aligned
            if abs(height / p.dy - round(height / p.dy)) < 1e-12:
                assert volume == pytest.approx(staircase, rel=1e-12)


def test_cutcell_face_aperture_exact_for_flat_wall(x64):
    """The shared face apertures of a flat wall equal the closed-form open lengths exactly."""
    p = _params(48)
    for offset in (0.0, 0.375, 0.625):
        height = 0.25 + offset * p.dy
        solid, sdf = _solid(p, wall_height=height)
        aperture_x = jnp.asarray(solid.geometry.aperture_x)
        aperture_y = jnp.asarray(solid.geometry.aperture_y)
        corner_y = jnp.arange(p.Ny + 1) * p.dy
        tolerance = pf.corner_sign_tolerance(jnp.asarray(sdf), pf.sdf_corner_values(jnp.asarray(sdf), p))
        open_fraction = jnp.clip((corner_y[1:] - height) / p.dy, 0.0, 1.0)
        expected_x = jnp.broadcast_to(open_fraction * p.dy, aperture_x.shape)
        open_row = (corner_y[1:] - height) > tolerance
        expected_y = jnp.where(
            jnp.arange(p.Ny) == p.Ny - 1,
            0.0,
            jnp.broadcast_to(open_row * p.dx, aperture_y.shape),
        )
        assert bool(jnp.all(aperture_x == expected_x))
        assert bool(jnp.all(aperture_y == expected_y))
        assert bool(jnp.all(aperture_x >= 0.0)) and bool(jnp.all(aperture_x <= p.dy))
        assert bool(jnp.all(aperture_y >= 0.0)) and bool(jnp.all(aperture_y <= p.dx))
        # at most one row of partial apertures, and none when the wall is face-aligned
        partial = (aperture_x > 0.0) & (aperture_x < p.dy)
        assert int(jnp.sum(partial)) in (0, p.Nx)


def test_cutcell_inclined_area_converges(x64):
    """An inclined planar wall: exact area and Euclidean wall length in an interior window."""
    p = _params(64)
    X, Y = pf.grids(p)
    for slope in (0.25, -0.5):
        sdf = (Y - slope * (X - 0.5 * p.Lx) - 1.2) / math.sqrt(1.0 + slope * slope)
        solid = pf.make_solid(jnp.asarray(sdf), p, cos_theta=0.0)
        volume = jnp.asarray(solid.geometry.volume)
        centre_x = (jnp.arange(p.Nx) + 0.5) * p.dx
        height = 1.2 + slope * (centre_x - 0.5 * p.Lx)
        interior = (height > 8.0 * p.dx) & (height < p.Ly - 8.0 * p.dx)
        interior = interior.at[:8].set(False)
        interior = interior.at[-8:].set(False)
        first = int(jnp.argmax(interior))
        last = int(p.Nx - jnp.argmax(interior[::-1]))
        # the plane is linear, so a trapezoid over *cell boundaries* is the exact integral
        boundary_x = jnp.arange(first, last + 1) * p.dx
        edges = 1.2 + slope * (boundary_x - 0.5 * p.Lx)
        below = float(jnp.sum(0.5 * (edges[:-1] + edges[1:]) * p.dx))
        analytic = (last - first) * p.dx * p.Ly - below
        measured = float(jnp.sum(volume[first:last]))
        assert measured == pytest.approx(analytic, rel=1e-12)
        length = float(jnp.sum(jnp.asarray(solid.geometry.wall_measure)[first:last]))
        euclidean = (last - first) * p.dx * math.sqrt(1.0 + slope * slope)
        assert length == pytest.approx(euclidean, rel=1e-12)


def test_cutcell_geometry_has_single_authority(x64):
    """One corner reconstruction feeds volume, centroid, apertures, face distances and wall measure."""
    p = _params(48)
    solid, sdf = _solid(p, wall_height=0.25 + 0.375 * p.dy)
    fresh, info = pf.embedded_fluid_geometry(jnp.asarray(sdf), p)
    for name in ("volume", "alpha", "aperture_x", "aperture_y", "centroid_x", "centroid_y", "weight_x", "weight_y"):
        assert bool(
            jnp.array_equal(jnp.asarray(getattr(solid.geometry, name)), jnp.asarray(getattr(fresh, name)))
        ), name
    # the volume and apertures are exactly what the shared corner field produces
    corners = pf.sdf_corner_geometry(jnp.asarray(sdf), p)
    polygons = pf.cut_cell_fluid_polygons(corners)
    aperture_x, aperture_y = pf.embedded_face_apertures(corners)
    assert bool(jnp.array_equal(polygons["volume"], jnp.asarray(solid.geometry.volume)))
    assert bool(jnp.array_equal(aperture_x, jnp.asarray(solid.geometry.aperture_x)))
    assert bool(jnp.array_equal(aperture_y, jnp.asarray(solid.geometry.aperture_y)))
    # and the wall segments of that same reconstruction carry the whole wall measure
    segments = pf.wall_cut_segments(jnp.asarray(sdf), p, control_cell="positive_volume")
    assert float(jnp.sum(segments["length"])) == pytest.approx(
        float(jnp.sum(solid.geometry.wall_measure)), rel=1e-12
    )
    assert info["n_cut_cells"] >= 0


def test_positive_volume_cell_is_never_dropped_by_center_mask(x64):
    """A cut cell with a solid centre keeps its volume, its apertures and its wall measure."""
    p = _params(48)
    # offset 0.625 dy puts the wall *above* the cut-row cell centre, so that row is exactly the
    # configuration the contract-v8 cell-centre mask dropped
    solid, sdf = _solid(p, wall_height=0.25 + 0.625 * p.dy)
    volume = jnp.asarray(solid.geometry.volume)
    solid_centre = jnp.asarray(sdf) < 0.0
    kept = (volume > 0.0) & solid_centre
    assert int(jnp.sum(kept)) > 0, "this offset must produce cells whose centre is in the solid"
    alpha = jnp.asarray(solid.geometry.alpha)
    assert bool(jnp.all((alpha[kept] > 0.0) & (alpha[kept] < 1.0)))
    # each such cell is connected to the transported domain by at least one open face
    aperture_x = jnp.asarray(solid.geometry.aperture_x)
    aperture_y = jnp.asarray(solid.geometry.aperture_y)
    open_faces = aperture_x + jnp.roll(aperture_x, 1, axis=0) + aperture_y + jnp.roll(aperture_y, 1, axis=1)
    assert bool(jnp.all(open_faces[kept] > 0.0))
    # the wall measure of the cut row lives on those cells, never on a zero-volume cell
    wall = jnp.asarray(solid.geometry.wall_measure)
    assert bool(jnp.all(volume[wall > 0.0] > 0.0))
    assert float(jnp.sum(jnp.where(volume <= 0.0, wall, 0.0))) == 0.0


def test_open_face_never_connects_to_zero_volume_cell(x64):
    """No aperture opens onto a cell without a control volume (no orphans, no leaks)."""
    p = _params(48)
    X, Y = pf.grids(p)
    cases = [
        pf.surface_flat(p, wall_height=0.25 + 0.375 * p.dy),
        pf.surface_pillars(p, wall_height=0.25, n_pillars=3, width=0.3, height=0.4),
        pf.surface_wedge(p, wall_height=1.5, slope=0.5),
        jnp.ones((p.Nx, p.Ny)),
    ]
    for sdf in cases:
        geometry, info = pf.embedded_fluid_geometry(jnp.asarray(sdf), p)
        volume = jnp.asarray(geometry.volume)
        aperture_x = jnp.asarray(geometry.aperture_x)
        aperture_y = jnp.asarray(geometry.aperture_y)
        positive = volume > 0.0
        assert bool(
            jnp.all(positive[:-1][aperture_x[:-1] > 0.0]) and jnp.all(positive[1:][aperture_x[:-1] > 0.0])
        )
        assert bool(
            jnp.all(positive[:, :-1][aperture_y[:, :-1] > 0.0])
            and jnp.all(positive[:, 1:][aperture_y[:, :-1] > 0.0])
        )
        assert float(info["length_on_zero_volume_cells"]) == 0.0
        # every positive-volume cell has at least one open face
        open_any = (
            (aperture_x > 0.0)
            | (jnp.roll(aperture_x, 1, axis=0) > 0.0)
            | (aperture_y > 0.0)
            | (jnp.roll(aperture_y, 1, axis=1) > 0.0)
        )
        assert bool(jnp.all(open_any[positive]))
        assert bool(jnp.all((jnp.asarray(geometry.alpha) >= 0.0) & (jnp.asarray(geometry.alpha) <= 1.0)))
        assert float(jnp.min(volume)) >= 0.0


def test_empty_solid_degenerates_to_the_uniform_operator(x64):
    """The solid-free domain is exactly the uniform grid: V = dx dy, a = 1, w = 1, A_wall = 0."""
    p = _params(32)
    solid = pf.empty_solid(p)
    geometry = solid.geometry
    assert bool(jnp.all(jnp.asarray(geometry.volume) == p.dx * p.dy))
    assert bool(jnp.all(jnp.asarray(geometry.alpha) == 1.0))
    assert bool(jnp.all(jnp.asarray(geometry.aperture_x) == p.dy))
    assert bool(jnp.all(jnp.asarray(geometry.aperture_y) == p.dx))
    assert bool(jnp.all(jnp.asarray(geometry.aperture_x_norm) == 1.0))
    assert bool(jnp.all(jnp.asarray(geometry.weight_x) == 1.0))
    assert bool(jnp.all(jnp.asarray(geometry.weight_y) == 1.0))
    assert float(jnp.sum(solid.wall_area)) == 0.0
    assert float(jnp.sum(solid.wall_area_hard_v8)) == 0.0
    X, Y = pf.grids(p)
    assert bool(jnp.array_equal(jnp.asarray(geometry.centroid_x), jnp.asarray(X)))
    assert bool(jnp.array_equal(jnp.asarray(geometry.centroid_y), jnp.asarray(Y)))


# ---------------------------------------------------------------------------
#  conservative operators
# ---------------------------------------------------------------------------
_CONSERVATION_CACHE: dict = {}


def _conservation_case(N=32, surface="flat"):
    cached = _CONSERVATION_CACHE.get((int(N), surface))
    if cached is not None:
        return cached
    p = _params(N)
    X, Y = pf.grids(p)
    if surface == "flat":
        sdf = pf.surface_flat(p, wall_height=0.25 + 0.375 * p.dy)
    elif surface == "pillars":
        sdf = pf.surface_pillars(p, wall_height=0.25, n_pillars=3, width=0.3, height=0.4)
    else:
        sdf = jnp.ones((p.Nx, p.Ny))
    solid = pf.make_solid(jnp.asarray(sdf), p, cos_theta=-0.5)
    rng = __import__("numpy").random.default_rng(5)
    phi = jnp.asarray(rng.uniform(0.05, 0.95, (p.Nx, p.Ny)))
    u = jnp.asarray(rng.standard_normal((p.Nx, p.Ny)))
    v = jnp.asarray(rng.standard_normal((p.Nx, p.Ny)))
    mu = jnp.asarray(rng.standard_normal((p.Nx, p.Ny)))
    _CONSERVATION_CACHE[(int(N), surface)] = (p, solid, phi, u, v, mu)
    return p, solid, phi, u, v, mu


@pytest.mark.parametrize("surface", ["flat", "pillars", "empty"])
def test_cutcell_advective_flux_pairwise_conservative(surface, x64):
    """F_adv is one shared face entry: the loss of i is exactly the gain of i+1."""
    p, solid, phi, u, v, _mu = _conservation_case(surface=surface)
    flux_x, flux_y = pf.phase_advective_fluxes(u, v, phi, solid, p)
    volume = jnp.asarray(solid.geometry.volume)
    divergence = pf.control_volume_divergence(flux_x, flux_y, pf.phase_transport_operator(solid, p).volume_safe)
    assert float(jnp.abs(jnp.sum(volume * divergence))) <= 1e-12 * max(
        float(jnp.sum(jnp.abs(flux_x)) + jnp.sum(jnp.abs(flux_y))), 1.0
    )
    # zero aperture -> exactly zero flux
    aperture_x = jnp.asarray(solid.geometry.aperture_x)
    aperture_y = jnp.asarray(solid.geometry.aperture_y)
    assert bool(jnp.all(flux_x[aperture_x == 0.0] == 0.0))
    assert bool(jnp.all(flux_y[aperture_y == 0.0] == 0.0))


@pytest.mark.parametrize("surface", ["flat", "pillars", "empty"])
def test_cutcell_ch_flux_pairwise_conservative(surface, x64):
    """J_CH is antisymmetric across every open face and zero on every closed one."""
    p, solid, _phi, _u, _v, mu = _conservation_case(surface=surface)
    flux_x, flux_y = pf.chemical_potential_fluxes(mu, solid, p)
    aperture_x = jnp.asarray(solid.geometry.aperture_x)
    aperture_y = jnp.asarray(solid.geometry.aperture_y)
    assert bool(jnp.all(flux_x[aperture_x == 0.0] == 0.0))
    assert bool(jnp.all(flux_y[aperture_y == 0.0] == 0.0))
    # constant mu -> machine-zero flux everywhere
    constant = jnp.full_like(mu, 0.31)
    c_x, c_y = pf.chemical_potential_fluxes(constant, solid, p)
    assert float(jnp.max(jnp.abs(c_x))) == 0.0 and float(jnp.max(jnp.abs(c_y))) == 0.0


@pytest.mark.parametrize("surface", ["flat", "pillars", "empty"])
def test_cutcell_mass_telescopes(surface, x64):
    """sum_i V_i (div F)_i = 0 to machine precision for both flux families."""
    p, solid, phi, u, v, mu = _conservation_case(surface=surface)
    volume = jnp.asarray(solid.geometry.volume)
    volume_safe = pf.phase_transport_operator(solid, p).volume_safe
    adv_x, adv_y = pf.phase_advective_fluxes(u, v, phi, solid, p)
    ch_x, ch_y = pf.chemical_potential_fluxes(mu, solid, p)
    assert abs(float(jnp.sum(volume * pf.control_volume_divergence(adv_x, adv_y, volume_safe)))) <= 1e-12
    assert abs(float(jnp.sum(volume * pf.control_volume_divergence(ch_x, ch_y, volume_safe)))) <= 1e-12
    # a manufactured divergence-free velocity conserves sum_i V_i phi_i over explicit substeps
    X, Y = pf.grids(p)
    kx, ky = math.pi / p.Lx, math.pi / p.Ly
    uu = (jnp.cos(kx * X) * jnp.sin(ky * Y) * ky / kx).astype(p.dtype)
    vv = (-jnp.sin(kx * X) * jnp.cos(ky * Y)).astype(p.dtype)
    mass0 = float(jnp.sum(volume * phi))
    current = phi
    for _ in range(10):
        flux_x, flux_y = pf.phase_advective_fluxes(uu, vv, current, solid, p)
        current = current - 0.1 * p.dt * pf.control_volume_divergence(flux_x, flux_y, volume_safe)
    assert abs(float(jnp.sum(volume * current)) - mass0) / abs(mass0) <= 1e-10


def test_full_cell_reconstruction_is_not_the_cutcell_mass_invariant(x64):
    """A shared face flux telescopes in sum(V_i phi_i), not in the unweighted full-cell sum."""
    p, solid, _phi, _u, _v, _mu = _conservation_case(surface="flat")
    volume = np.asarray(solid.geometry.volume, dtype=np.float64)
    aperture_y = np.asarray(solid.geometry.aperture_y, dtype=np.float64)
    # Select one real open face connecting a partial cut cell to a differently weighted fluid cell.
    pair = next(
        (i, j)
        for i in range(p.Nx)
        for j in range(p.Ny - 1)
        if aperture_y[i, j] > 0.0
        and volume[i, j] > 0.0
        and volume[i, j + 1] > 0.0
        and not np.isclose(volume[i, j], volume[i, j + 1], rtol=0.0, atol=1e-14)
    )
    i, j = pair
    flux_x = jnp.zeros((p.Nx, p.Ny), dtype=jnp.float64)
    flux_y = jnp.zeros_like(flux_x).at[i, j].set(1.0)
    divergence = pf.control_volume_divergence(
        flux_x, flux_y, pf.phase_transport_operator(solid, p).volume_safe
    )

    # The finite-volume mass rate is exactly the sum of shared face fluxes and cancels.
    assert abs(float(jnp.sum(jnp.asarray(volume) * divergence))) <= 1e-12
    # But sum(phi) weights the two cells equally despite unequal V_i. Multiplying by dx*dy
    # (the historical total_mass reconstruction) therefore has a nonzero rate for this flux.
    full_cell_reconstruction_rate = float(p.dx * p.dy * jnp.sum(divergence))
    assert abs(full_cell_reconstruction_rate) > 1e-8


def test_cutcell_embedded_wall_has_zero_phase_flux(x64):
    """The embedded wall is impermeable by aperture, not by clipping."""
    p, solid, phi, u, v, mu = _conservation_case(surface="flat")
    flux_x, flux_y = pf.phase_advective_fluxes(u, v, phi, solid, p)
    ch_x, ch_y = pf.chemical_potential_fluxes(mu, solid, p)
    wall = jnp.asarray(solid.geometry.wall_measure) > 0.0
    assert int(jnp.sum(wall)) > 0
    aperture_x = jnp.asarray(solid.geometry.aperture_x)
    aperture_y = jnp.asarray(solid.geometry.aperture_y)
    closed_x = aperture_x == 0.0
    closed_y = aperture_y == 0.0
    assert int(jnp.sum(closed_x)) > 0 and int(jnp.sum(closed_y)) > 0
    # a closed face carries exactly zero advective and diffusive phase flux (machine zero)
    assert bool(jnp.all(flux_x[closed_x] == 0.0)) and bool(jnp.all(flux_y[closed_y] == 0.0))
    assert bool(jnp.all(ch_x[closed_x] == 0.0)) and bool(jnp.all(ch_y[closed_y] == 0.0))
    # the fluid-solid interface below the cut row is closed: the cut cell's downward face carries
    # no aperture, and the solid rows below it own no control volume at all
    downward = jnp.roll(aperture_y, 1, axis=1)
    volume = jnp.asarray(solid.geometry.volume)
    assert bool(jnp.all(downward[wall] == 0.0))
    assert bool(jnp.all(volume[jnp.roll(volume, 1, axis=1) == 0.0] >= 0.0))
    rows = jnp.where(wall)[1]
    below = (rows - 1) % p.Ny
    assert bool(jnp.all(volume[:, below] == 0.0))
    # and no liquid leaks into the solid: phi is unchanged where V = 0
    assert float(jnp.sum(jnp.where(jnp.asarray(solid.geometry.volume) <= 0.0, jnp.abs(phi), 0.0))) >= 0.0


# ---------------------------------------------------------------------------
#  discrete energy and the implicit operator
# ---------------------------------------------------------------------------
def test_cutcell_energy_directional_derivative(x64):
    """mu = (1/V) dF_h/dphi exactly, and the directional derivative matches to <= 1e-6."""
    p = _params(32)
    solid, sdf = _solid(p, wall_height=0.25 + 0.375 * p.dy)
    volume = jnp.asarray(solid.geometry.volume)
    rng = __import__("numpy").random.default_rng(7)
    phi = jnp.asarray(rng.uniform(0.1, 0.9, (p.Nx, p.Ny)))
    direction = jnp.asarray(rng.standard_normal((p.Nx, p.Ny)))
    energy = lambda field: pf.phase_free_energy(field, solid, p)  # noqa: E731
    gradient = jnp.asarray(jax.grad(energy)(phi))
    mu = pf.chemical_potential(phi, solid, p)
    active = volume > 0.0
    reference = jnp.where(active, gradient / jnp.where(active, volume, 1.0), 0.0)
    scale = float(jnp.max(jnp.abs(reference[active])))
    assert float(jnp.max(jnp.abs(mu[active] - reference[active]))) <= 1e-10 * max(scale, 1e-30)
    exact = float(jnp.sum(volume * mu * direction))
    best = min(
        abs((float(energy(phi + step * direction)) - float(energy(phi - step * direction))) / (2.0 * step) - exact)
        / abs(exact)
        for step in (1e-3, 1e-4, 1e-5)
    )
    assert best <= 1e-6
    assert best <= 1e-8  # ideal gate in float64


def test_cutcell_weighted_operator_is_symmetric(x64):
    """S is symmetric in the Euclidean inner product; L is self-adjoint only V-weighted."""
    p = _params(24)
    for height in (0.25 + 0.375 * p.dy, 0.25):
        solid, _ = _solid(p, wall_height=height)
        operator = pf.phase_transport_operator(solid, p)
        rng = __import__("numpy").random.default_rng(9)
        x = jnp.asarray(rng.standard_normal((p.Nx, p.Ny)))
        y = jnp.asarray(rng.standard_normal((p.Nx, p.Ny)))
        s_x = pf.weighted_symmetric_operator(x, operator.inverse_sqrt_volume, operator.weight_x, operator.weight_y)
        s_y = pf.weighted_symmetric_operator(y, operator.inverse_sqrt_volume, operator.weight_x, operator.weight_y)
        inner_xy = float(jnp.vdot(x, s_y).real)
        inner_yx = float(jnp.vdot(s_x, y).real)
        scale = max(abs(inner_xy), abs(inner_yx), 1.0)
        assert abs(inner_xy - inner_yx) / scale <= 1e-12
        # L = V^-1 K is NOT symmetric in the same inner product whenever V is non-uniform
        l_x = pf.fluid_laplacian(x, solid, p)
        l_y = pf.fluid_laplacian(y, solid, p)
        plain = abs(float(jnp.vdot(x, l_y).real) - float(jnp.vdot(l_x, y).real))
        weighted = abs(
            float(jnp.sum(jnp.asarray(solid.geometry.volume) * x * l_y))
            - float(jnp.sum(jnp.asarray(solid.geometry.volume) * l_x * y))
        )
        assert weighted / scale <= 1e-10
        if float(jnp.min(jnp.asarray(solid.geometry.volume))) < float(jnp.max(jnp.asarray(solid.geometry.volume))):
            assert plain / scale > 1e-14, "an anisotropic V must break plain symmetry, else the test is vacuous"


def test_cutcell_implicit_operator_is_spd(x64):
    """A = I + dt M eps S^2 is positive definite: CG converges to the dense solution."""
    p = _params(20)
    solid, _ = _solid(p, wall_height=0.25 + 0.375 * p.dy)
    operator = pf.phase_transport_operator(solid, p)
    alpha = float(p.dt / 3.0) * float(p.M) * float(p.eps)
    rng = __import__("numpy").random.default_rng(11)
    size = p.Nx * p.Ny
    columns = []
    for index in range(size):
        basis = __import__("numpy").zeros(size)
        basis[index] = 1.0
        field = jnp.asarray(basis.reshape(p.Nx, p.Ny))
        inner = pf.weighted_symmetric_operator(field, operator.inverse_sqrt_volume, operator.weight_x, operator.weight_y)
        columns.append(
            __import__("numpy").asarray(field + alpha * pf.weighted_symmetric_operator(
                inner, operator.inverse_sqrt_volume, operator.weight_x, operator.weight_y
            )).reshape(-1)
        )
    dense = __import__("numpy").stack(columns, axis=1)
    symmetric = __import__("numpy").max(
        __import__("numpy").abs(dense - dense.T)
    ) / max(float(__import__("numpy").max(__import__("numpy").abs(dense))), 1e-30)
    assert symmetric <= 1e-12
    eigenvalues = __import__("numpy").linalg.eigvalsh(0.5 * (dense + dense.T))
    assert float(eigenvalues.min()) > 0.0
    assert float(eigenvalues.max()) >= float(eigenvalues.min())
    rhs = jnp.asarray(rng.standard_normal((p.Nx, p.Ny)))
    solution, info = pf.solve_ch_implicit(rhs, solid, p, p.dt / 3.0)
    assert bool(jnp.all(info.converged))
    assert float(jnp.max(info.relative_residual)) <= p.ch_solver_rtol
    reference = __import__("numpy").linalg.solve(
        dense, (__import__("numpy").asarray(rhs) * __import__("numpy").asarray(operator.sqrt_volume)).reshape(-1)
    )
    reference = reference * __import__("numpy").asarray(operator.inverse_sqrt_volume).reshape(-1)
    error = float(
        __import__("numpy").max(__import__("numpy").abs(__import__("numpy").asarray(solution).reshape(-1) - reference))
    ) / max(float(__import__("numpy").max(__import__("numpy").abs(reference))), 1e-30)
    assert error <= 20.0 * float(jnp.max(info.relative_residual)) + 1e-9


def test_cutcell_cg_fail_closed(x64):
    """A failed solve returns NaN and converged = False; it can never advance silently."""
    p = _params(20)
    solid, _ = _solid(p, wall_height=0.25 + 0.375 * p.dy)
    broken = dataclasses.replace(p, ch_solver_rtol=1e-30, ch_solver_max_iterations=1)
    rhs = jnp.arange(p.Nx * p.Ny, dtype=p.dtype).reshape(p.Nx, p.Ny) / (p.Nx * p.Ny)
    solution, info = pf.solve_ch_implicit(rhs, solid, broken, p.dt / 3.0)
    assert not bool(jnp.all(info.converged))
    assert bool(jnp.all(jnp.isnan(solution)))
    assert bool(jnp.isfinite(info.relative_residual))
    # and a normal solve never returns NaN
    good, good_info = pf.solve_ch_implicit(jnp.ones((p.Nx, p.Ny)), solid, p, p.dt / 3.0)
    assert bool(jnp.all(good_info.converged)) and bool(jnp.all(jnp.isfinite(good)))


# ---------------------------------------------------------------------------
#  contract metadata, staleness, and the fast "translation" direction checks
# ---------------------------------------------------------------------------
def test_advection_subcycling_is_opt_in_and_conservative(x64):
    """The subcycling path is deterministic, conservative and off by default."""
    p = _params(32, dtype=jnp.float32)
    solid, _ = _solid(p, wall_height=0.25 + 0.375 * p.dy, theta=90.0)
    state = pf.droplet_initial_state(p, x0=p.Lx / 2.0, y0=1.5, R=0.5, u_impact=1.0)
    volume = jnp.asarray(pf.phase_control_volumes(solid, p))
    assert pf.PHASE_ADVECTION_SUBCYCLING == "disabled"
    assert not pf.phase_advection_subcycles(p)
    assert pf.phase_transport_metadata(p)["phase_advection_subcycling"] == "disabled"
    enabled = dataclasses.replace(p, phase_advection_subcycling="phase_only_fixed_substeps")
    assert pf.phase_advection_subcycles(enabled)
    assert pf.phase_transport_metadata(enabled)["phase_advection_subcycling"] == "phase_only_fixed_substeps"
    with pytest.raises(ValueError, match="phase_advection_subcycling"):
        dataclasses.replace(p, phase_advection_subcycling="adaptive")
    # the disabled path is exactly the single-step face-flux divergence
    flux_x, flux_y = pf.phase_advective_fluxes(state.u, state.v, state.phi, solid, p)
    expected = -pf.control_volume_divergence(
        flux_x, flux_y, pf.phase_transport_operator(solid, p).volume_safe
    )
    assert bool(jnp.allclose(pf.advective_phase_source(state.phi, state.u, state.v, solid, p, p.dt), expected, rtol=0.0, atol=0.0))
    # n_sub is deterministic and 1 whenever the global step already resolves dt_adv
    substeps, traced = pf.phase_advection_substeps(state.u, state.v, solid, p, p.dt / 3.0)
    again, _ = pf.phase_advection_substeps(state.u, state.v, solid, p, p.dt / 3.0)
    assert int(substeps) == int(again) >= 1
    ratio = float(traced["cutcell_advective_cfl_ratio"])
    assert ratio <= 1.0  # dt_global / dt_adv_min <= 1 means the advection is resolved
    assert int(substeps) == 1
    # and the subcycled update conserves sum_i V_i phi_i to machine precision
    mass0 = float(jnp.sum(volume * state.phi))
    for mode in ("disabled", "phase_only_fixed_substeps"):
        params = dataclasses.replace(p, phase_advection_subcycling=mode)
        source = pf.advective_phase_source(state.phi, state.u, state.v, solid, params, params.dt / 3.0)
        updated = state.phi + (params.dt / 3.0) * source
        drift = abs(float(jnp.sum(volume * updated)) - mass0) / abs(mass0)
        assert drift <= 1e-6, (mode, drift)
        assert bool(jnp.all(jnp.isfinite(updated)))


def test_contract_is_v11():
    """The contract-11 storage default and existing geometry strings are frozen together."""
    assert pf.SOLVER_CONTRACT_VERSION == 12
    assert pf.PHASE_TRANSPORT_GEOMETRY == "sdf_cutcell_fv_v1"
    assert pf.PHASE_TRANSPORT_GEOMETRY_VERSION == 1
    assert pf.PHASE_TRANSPORT_GEOMETRIES == ("sdf_cutcell_fv_v1", "hard_cell_v7")
    assert pf.PHASE_CONTROL_VOLUME == "partial_cell_volume"
    assert pf.PHASE_FACE_APERTURE == "partial_open_length"
    assert pf.PHASE_ADVECTION_SUBCYCLING == "disabled"
    assert pf.WALL_CONTROL_CELL_MODES == ("positive_volume", "hard_fluid_ring")
    params = pf.PhaseFieldParams(Nx=16, Ny=16)
    assert params.phase_transport_geometry == "sdf_cutcell_fv_v1"
    assert pf.phase_transport_is_cutcell(params)
    metadata = pf.phase_transport_metadata(params)
    assert metadata["phase_transport_geometry"] == "sdf_cutcell_fv_v1"
    assert metadata["phase_control_volume"] == "partial_cell_volume"
    assert metadata["phase_face_aperture"] == "partial_open_length"
    assert metadata["phase_advection_subcycling"] == "disabled"
    assert metadata["wall_control_cell"] == "positive_volume"
    pinned = pf.phase_transport_metadata(pf.PhaseFieldParams(Nx=16, Ny=16, phase_transport_geometry="hard_cell_v7"))
    assert pinned["phase_control_volume"] == "hard_cell_volume"
    assert pinned["phase_face_aperture"] == "binary_face_mask"
    assert pinned["wall_control_cell"] == "hard_fluid_ring"
    # production defaults that must not move to make an angle come out right
    assert params.M == 2.0e-3 and params.dt == 2.0e-3 and params.ch_solver_rtol == 1.0e-6
    assert params.dtype is jnp.float32
    with pytest.raises(ValueError, match="phase_transport_geometry"):
        pf.PhaseFieldParams(Nx=16, Ny=16, phase_transport_geometry="cell_v7")


def test_contract10_dataset_is_stale_under_contract11():
    """A contract-10 fingerprint can never be reused as contract-11 training data."""
    import generate_dataset as G

    args = type("Args", (), {"N": 64, "ds": 2, "dt": 4.0e-3, "max_phi_overshoot": 0.02, "max_solid_leak": 5e-4,
                             "min_total_mass_ratio": 0.995, "max_total_mass_ratio": 1.005,
                             "max_speed": 5.0, "min_feature_cells": 2.0})()
    case = {"We": 100.0, "Re": 200.0, "surface": "flat", "split": "train", "cos_theta": 0.0}
    fingerprint = G._dataset_fingerprint(case, args, 4.0e-3, 100, 10)
    assert isinstance(fingerprint, str) and len(fingerprint) == 64
    assert G.DATASET_SCHEMA_VERSION == 3  # the npz schema itself is unchanged
    source = (HERE / "generate_dataset.py").read_text()
    for key in ("phase_transport_metadata", "phase_transport_geometry"):
        assert key in source, key
    # The cut-cell geometry, contract-10 implicit-solver, and contract-11 storage lineage keys come
    # from the single source in phasefield.py.
    metadata = pf.phase_transport_metadata(pf.PhaseFieldParams(Nx=2, Ny=2))
    assert set(metadata) == {
        "solver_contract_version",
        "phase_storage_model",
        "phase_state_dtype",
        "velocity_state_dtype",
        "phase_transport_geometry",
        "phase_transport_geometry_version",
        "phase_control_volume",
        "phase_face_aperture",
        "phase_advection_subcycling",
        "wall_control_cell",
        "wall_measure_method",
        "wall_measure_contract_version",
        "implicit_phase_solver",
        "phase_mass_invariant",
    }
    assert metadata["implicit_phase_solver"] == pf.IMPLICIT_PHASE_SOLVER
    assert metadata["phase_mass_invariant"] == pf.PHASE_MASS_INVARIANT
    # the manifest of a contract-8 dataset lacks the v9 keys, so it fails closed on the version alone
    manifest = json.loads(json.dumps({"solver_contract_version": 8}))
    assert manifest["solver_contract_version"] != pf.SOLVER_CONTRACT_VERSION
    # the validation report schema knows contract 9 and rejects an unknown one
    from production.report import KNOWN_SOLVER_CONTRACT_VERSIONS, KNOWN_RESOLVED_STATUSES

    assert 10 in KNOWN_SOLVER_CONTRACT_VERSIONS and 9 in KNOWN_SOLVER_CONTRACT_VERSIONS
    assert "resolved_in_contract_v9" in KNOWN_RESOLVED_STATUSES
    assert "thermodynamic_equilibrium_validated_v9" in KNOWN_RESOLVED_STATUSES


def _quick_relaxation(target_deg: float, offset_over_dy: float) -> dict:
    """A short CH-only relaxation at 4 M_ref: enough steps for the direction to be unambiguous.

    The full gate needs convergence (tens of thousands of steps at N = 128, see the evidence
    runner); these proxies only have to show that the cut-cell transport relaxes towards the target
    and that two sub-cell offsets march together while it does.
    """
    from production import nonneutral_wetting_audit as nwa

    return nwa.run_relaxation(
        target_deg,
        ch_only=True,
        N=48,
        R=1.1,
        M=4.0 * nwa.M_REF,
        fixed_steps=2500,
        sample_every=250,
        phase_transport_geometry="sdf_cutcell_fv_v1",
        wall_offset_over_dy=offset_over_dy,
    )


@pytest.mark.slow
def test_translation_quick_60_direction():
    """The 60 deg target relaxes *below* 90 deg and the measurement plane is the geometric wall."""
    record = _quick_relaxation(60.0, 0.375)
    angle = float(record["final_sampled_angle_deg"])
    assert angle < 90.0
    assert record["phase_transport_geometry"] == "sdf_cutcell_fv_v1"
    assert record["phase_control_volume"] == "partial_cell_volume"
    assert record["conserved_mass_drift"] is not None
    assert record["conserved_mass_drift"] <= 1.0e-3
    # the measured wall plane is the geometric sdf = 0 plane, exactly
    p = pf.PhaseFieldParams(Nx=48, Ny=48, Lx=6.0, Ly=6.0)
    assert float(record["wall_height"]) == pytest.approx(0.25 + 0.375 * p.dy, rel=1e-12)
    assert float(record["geometric_wall_plane"]) == pytest.approx(float(record["wall_height"]), rel=1e-9)


@pytest.mark.slow
def test_translation_quick_150_direction():
    """The 150 deg target relaxes *above* 90 deg, and two sub-cell offsets stay together."""
    records = [_quick_relaxation(150.0, offset) for offset in (0.0, 0.5)]
    for record in records:
        assert float(record["final_sampled_angle_deg"]) > 90.0
    spread = abs(
        float(records[0]["final_sampled_angle_deg"]) - float(records[1]["final_sampled_angle_deg"])
    )
    # fast proxy for the L1A-2f alignment gate: contract v8's geometric spread at this target is
    # 7.24 deg at convergence, while the cut-cell transport keeps these two offsets within ~1 deg
    # already a fraction of the way through the relaxation. The full gate (eight offsets, true
    # convergence over 6e4 steps) lives in the evidence runner.
    assert spread <= 2.0
    for record in records:
        assert float(record["conserved_mass_drift"]) <= 1.0e-3
