"""
Lightweight physics tests for the two-phase droplet-impact solver.

These are fast enough for CI and skip cleanly when JAX is not installed (JAX is an
optional extra in HydroGym).  They verify the two properties that the whole
learning pipeline relies on:

1. liquid mass is (nearly) conserved by the Cahn--Hilliard + advection update;
2. the pressure projection leaves the velocity divergence-free to round-off
   (i.e. the discrete grad/div/Laplacian symbols are mutually consistent);
3. the capillary sign convention is physical (L1A-2a, solver contract v5): the
   Korteweg force points toward the liquid, never creates interfacial free
   energy, and gives a *positive* Laplace jump ``P_liquid - P_gas`` for a convex
   drop.  Assertions on ``delta_p`` must stay signed -- never ``abs(delta_p)``.

Run with::

    pytest examples/two_phase/test_two_phase.py
"""

import ast
import inspect

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

import phasefield as pf  # noqa: E402


def test_mass_conservation_no_wall():
    p = pf.PhaseFieldParams(Nx=96, Ny=96, Lx=6.0, Ly=6.0, Re=200.0, We=100.0, dt=4e-3)
    solid = pf.empty_solid(p)
    st = pf.droplet_initial_state(p, x0=3.0, y0=3.0, R=0.8, u_impact=0.2)
    m0 = float(pf.liquid_mass(st.phi, solid, p))
    step = jax.jit(pf.step, static_argnums=(2,))
    for _ in range(300):
        st = step(st, solid, p)
    m1 = float(pf.liquid_mass(st.phi, solid, p))
    assert abs(m1 - m0) / m0 < 1e-2


def test_projection_is_divergence_free():
    # With no solid the Brinkman damping is the identity, so after the projection
    # the discrete divergence must vanish to round-off.  (Near a wall the damping
    # intentionally re-introduces divergence to enforce no-slip.)
    p = pf.PhaseFieldParams(Nx=96, Ny=96, Lx=6.0, Ly=6.0, Re=200.0, We=100.0, dt=4e-3)
    solid = pf.empty_solid(p)
    st = pf.droplet_initial_state(p, x0=3.0, y0=3.0, R=0.8, u_impact=0.5)
    step = jax.jit(pf.step, static_argnums=(2,))
    for _ in range(20):
        st = step(st, solid, p)
    div = pf._ddx(st.u, p.dx) + pf._ddy(st.v, p.dy)
    umax = float(jnp.abs(jnp.stack([st.u, st.v])).max()) + 1e-8
    assert float(jnp.abs(div).max()) / (umax / p.dx) < 1e-3


def test_solid_geometry_sign_convention():
    # fluid must NOT be marked solid and vice-versa
    p = pf.PhaseFieldParams(Nx=64, Ny=64, Lx=6.0, Ly=6.0)
    sdf = pf.surface_flat(p, wall_height=0.25)
    solid = pf.make_solid(sdf, p, cos_theta=0.0)
    chi = np.asarray(solid.chi)
    _, Y = [np.asarray(a) for a in pf.grids(p)]
    # deep in the fluid (top) chi ~ 0 ; deep in the wall (bottom) chi ~ 1
    assert chi[Y > 1.0].mean() < 0.01
    assert chi[Y < 0.1].mean() > 0.9


def test_step_advances_one_requested_dt():
    """The three internal substeps must represent exactly one public dt."""
    p = pf.PhaseFieldParams(Nx=64, Ny=64, Lx=6.0, Ly=6.0, dt=2e-3)
    solid = pf.empty_solid(p)
    st = pf.droplet_initial_state(p, x0=3.0, y0=3.0, R=0.8, u_impact=0.0)
    out = pf.step(st, solid, p)
    assert float(out.t) == pytest.approx(p.dt, rel=0.0, abs=1e-8)


def test_generated_case_starts_outside_solid():
    """The default case builder must not seed liquid inside the wall."""
    p, solid, st = pf.build_case(
        dict(surface="flat", We=12.0, Re=70.0, R=0.65, u_impact=0.2, eps_factor=3.0),
        N=96,
        dt=1e-3,
    )
    overlap = float(jnp.sum(st.phi * solid.chi) / jnp.maximum(jnp.sum(st.phi), 1e-12))
    assert overlap < 0.05


def test_remote_tall_obstacle_does_not_raise_local_support():
    """Initial clearance is based on the drop footprint, not a global max height."""
    p = pf.PhaseFieldParams(Nx=128, Ny=128, Lx=6.0, Ly=6.0)
    X, Y = pf.grids(p)
    wall = pf.surface_flat(p)
    remote = pf.sdf_box(X, Y, 4.35, 4.85, 0.25, 2.0)
    sdf = pf.sdf_union(wall, remote)
    local = float(pf._local_surface_top(sdf, p, x0=1.0, radius=0.65, margin=0.1))
    global_top = float(jnp.max(jnp.where(sdf < 0.0, Y, -jnp.inf)))
    assert global_top > local + 1.0
    assert local == pytest.approx(float(pf._local_surface_top(wall, p, x0=1.0, radius=0.65, margin=0.1)))


def test_smoke_flat_drop_moves_downward_and_reaches_wall_signal():
    """A representative L0 case should exhibit basic kinematics, not a static field."""
    p, solid, initial = pf.build_case(
        dict(
            surface="flat",
            We=120.0,
            Re=120.0,
            R=0.65,
            u_impact=1.0,
            eps_factor=2.0,
            impact_gap_eps=0.75,
            velocity_mode="uniform",
            dt=2e-3,
        ),
        N=64,
        dt=2e-3,
    )
    step = jax.jit(pf.step, static_argnums=(2,))
    states = [initial]
    for _ in range(40):
        states.append(step(states[-1], solid, p))
    y = jnp.arange(p.Ny) * p.dy + 0.5 * p.dy
    centroids = [float(jnp.sum(s.phi * y[None, :]) / jnp.maximum(jnp.sum(s.phi), 1e-8)) for s in states]
    assert centroids[-1] < centroids[0]
    # The lower fluid cells must eventually carry a nonzero liquid signal.
    near_wall = (y > 0.25) & (y < 0.75)
    assert float(jnp.max(states[-1].phi[:, near_wall])) > 1e-4


def test_wetting_band_is_fluid_side_only():
    p = pf.PhaseFieldParams(Nx=96, Ny=96, Lx=6.0, Ly=6.0, wet_band=0.1)
    solid = pf.make_solid(pf.surface_flat(p), p, cos_theta=0.5)
    band = np.asarray(pf.wet_band(solid, p))
    sdf = np.asarray(solid.sdf)
    assert np.max(np.abs(band[sdf < 0.0])) == pytest.approx(0.0, abs=1e-7)
    assert float(np.max(band[(sdf >= 0.0) & (sdf < p.wet_band)])) > 0.5
    assert float(np.max(band[sdf > 4.0 * p.wet_band])) < 1e-5


def test_solid_projection_is_bounded_and_mass_conserving():
    p = pf.PhaseFieldParams(Nx=64, Ny=64, Lx=6.0, Ly=6.0, enforce_solid_phi=True)
    solid = pf.make_solid(pf.surface_flat(p), p, cos_theta=0.0)
    phi = np.asarray(pf.droplet_initial_state(p, x0=3.0, y0=0.4, R=0.5, u_impact=0.0).phi)
    before = float(phi.sum())
    projected = np.asarray(pf._project_phase_outside_solid(jnp.asarray(phi), solid, p))
    assert projected.min() >= -1e-7
    assert projected.max() <= 1.0 + 1e-7
    assert np.max(np.abs(projected[np.asarray(solid.sdf) < 0.0])) < 1e-6
    assert projected.sum() == pytest.approx(before, rel=2e-5, abs=2e-5)


def test_lowwe_clearance_scales_with_eps():
    p, solid, st = pf.build_case(
        dict(
            surface="flat",
            We=12.0,
            Re=70.0,
            R=0.65,
            u_impact=0.2,
            eps_factor=3.0,
            impact_gap=0.03,
            impact_gap_eps=2.5,
        ),
        N=96,
        dt=1e-3,
    )
    solid_mask = np.asarray(solid.sdf) < 0.0
    assert float(np.max(np.asarray(st.phi)[solid_mask])) < 0.05


def test_poisson_masks_centered_difference_nyquist_null_modes():
    p = pf.PhaseFieldParams(Nx=64, Ny=64, Lx=6.0, Ly=6.0)
    i = jnp.arange(p.Nx)[:, None]
    checker = jnp.where((i % 2) == 0, 1.0, -1.0) * jnp.ones((1, p.Ny))
    sol = pf.poisson_solve(checker, p.m2_proj)
    assert np.isfinite(np.asarray(sol)).all()
    assert float(jnp.max(jnp.abs(sol))) < 1e-6


def test_periodic_wedge_has_no_edge_cliff_and_normalised_levelset():
    p = pf.PhaseFieldParams(Nx=192, Ny=192, Lx=6.0, Ly=6.0)
    sdf = np.asarray(pf.surface_wedge(p, slope=0.5))
    assert float(np.max(np.abs(sdf[0] - sdf[-1]))) < 2.0 * p.dx
    gx = np.asarray(pf._ddx(jnp.asarray(sdf), p.dx))
    gy = np.asarray(pf._ddy(jnp.asarray(sdf), p.dy))
    near = np.abs(sdf) < 2.0 * p.dx
    grad_norm = np.sqrt(gx**2 + gy**2)
    assert float(np.median(grad_norm[near])) == pytest.approx(1.0, rel=0.12)


def test_liquid_mass_uses_geometric_fluid():
    p = pf.PhaseFieldParams(Nx=64, Ny=64, Lx=6.0, Ly=6.0)
    solid = pf.make_solid(pf.surface_flat(p), p)
    phi = jnp.ones((p.Nx, p.Ny))
    expected = float(jnp.sum(solid.sdf >= 0.0) * p.dx * p.dy)
    assert float(pf.liquid_mass(phi, solid, p)) == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
#  Capillary / pressure sign consistency (L1A-2a, SOLVER_CONTRACT_VERSION 5)
# ---------------------------------------------------------------------------


def _drop_setup(N, R, We=100.0, eps_factor=2.0):
    """Circular liquid drop (phi = 1 inside) at rest in a periodic gas box, no solid."""
    p = pf.PhaseFieldParams(Nx=N, Ny=N, Lx=6.0, Ly=6.0, Re=200.0, We=We, dt=2e-3)
    p.eps = eps_factor * p.dx
    solid = pf.empty_solid(p)
    state = pf.droplet_initial_state(p, x0=p.Lx / 2.0, y0=p.Ly / 2.0, R=R, u_impact=0.0)
    return p, solid, state


def _radial_offsets(p):
    X, Y = [np.asarray(a, dtype=np.float64) for a in pf.grids(p)]
    return X - p.Lx / 2.0, Y - p.Ly / 2.0


def _elliptical_drop(p, R, aspect=1.3):
    """Non-equilibrium elliptical drop at rest: curvature varies, so mu grad(phi) is not a gradient."""
    X, Y = pf.grids(p)
    rx, ry = X - p.Lx / 2.0, Y - p.Ly / 2.0
    rho = jnp.sqrt(aspect * rx**2 + ry**2 / aspect)
    phi = (0.5 * (1.0 - jnp.tanh((rho - R) / (jnp.sqrt(2.0) * p.eps)))).astype(p.dtype)
    return pf.State(phi=phi, u=jnp.zeros_like(phi), v=jnp.zeros_like(phi), t=0.0)


def test_solver_contract_is_v5():
    """The capillary sign fix changes trajectories, so the solver contract must be 5 (v4 data is stale)."""
    assert pf.SOLVER_CONTRACT_VERSION == 5


def test_static_drop_pressure_jump_has_correct_sign():
    """A convex liquid drop (phi = 1) has P_liquid > P_gas: the Laplace jump is +sigma/R.

    The assertion is *signed* on purpose.  The coarse band on ``delta_p * R * We`` (Laplace: ~1)
    only guards against a gross scaling change on this small grid; it is not an accuracy gate.
    """
    R = 0.8
    p, solid, state = _drop_setup(N=64, R=R)
    step = jax.jit(pf.step, static_argnums=(2,))
    for _ in range(60):
        state = step(state, solid, p)
    pressure = np.asarray(pf.pressure_field(state, solid, p), dtype=np.float64)
    rx, ry = _radial_offsets(p)
    r = np.hypot(rx, ry)
    liquid_core, far_gas = r < 0.3 * R, r > 2.5 * R
    assert float(np.asarray(state.phi)[liquid_core].mean()) > 0.9  # "inside" really is liquid
    assert float(np.asarray(state.phi)[far_gas].mean()) < 0.1  # "outside" really is gas
    delta_p = float(pressure[liquid_core].mean() - pressure[far_gas].mean())  # P_liquid - P_gas
    assert delta_p > 0.0
    assert 0.7 < delta_p * R * p.We < 1.3


def test_korteweg_force_points_toward_liquid():
    """F . n_out < 0 with n_out = -grad(phi)/|grad(phi)| (liquid -> gas), and mu > 0 at the interface."""
    p, solid, state = _drop_setup(N=64, R=0.8)
    _, u_rhs, v_rhs, mu, _ = pf.rhs(state, solid, p)  # at rest: rhs is the capillary acceleration only
    px, py = np.asarray(pf._ddx(state.phi, p.dx), np.float64), np.asarray(pf._ddy(state.phi, p.dy), np.float64)
    gnorm = np.maximum(np.hypot(px, py), 1e-30)
    n_out_x, n_out_y = -px / gnorm, -py / gnorm
    net = float(np.sum((np.asarray(u_rhs, np.float64) * n_out_x + np.asarray(v_rhs, np.float64) * n_out_y) * gnorm))
    assert net < 0.0
    assert float(np.max(np.asarray(mu))) > 0.0
    # n_out really is the outward radial direction of the drop
    rx, ry = _radial_offsets(p)
    r = np.maximum(np.hypot(rx, ry), 1e-12)
    band = np.abs(r - 0.8) < 2.0 * p.eps
    assert float(np.min((n_out_x * rx + n_out_y * ry)[band] / r[band])) > 0.9


def test_capillary_flow_does_not_create_interface_free_energy():
    """Energy consistency, independent of any pressure diagnostic.

    Cahn-Hilliard advection changes the free energy by dF/dt = -int mu u.grad(phi).  The velocity
    driven by the solenoidal part of the capillary force must therefore have int mu u.grad(phi) > 0.
    """
    p, solid, _ = _drop_setup(N=64, R=0.8)
    state = _elliptical_drop(p, R=0.8)
    _, u_rhs, v_rhs, mu, _ = pf.rhs(state, solid, p)
    pressure = pf.pressure_field(state, solid, p)
    dt = p.dt / 3.0
    u = dt * (u_rhs - pf._ddx(pressure, p.dx))
    v = dt * (v_rhs - pf._ddy(pressure, p.dy))
    px, py = pf._ddx(state.phi, p.dx), pf._ddy(state.phi, p.dy)
    minus_dF_dt = float(np.sum(np.asarray(mu, np.float64) * (np.asarray(u * px + v * py, np.float64))))
    assert minus_dF_dt > 0.0


def test_pressure_field_reuses_rhs_forces(monkeypatch):
    """pressure_field() and step() take *all* forces from rhs(): no second copy of the capillary formula."""
    p, solid, state = _drop_setup(N=32, R=0.8)
    reference = np.asarray(pf.pressure_field(state, solid, p), dtype=np.float64)
    assert np.abs(reference).max() > 1e-4  # a non-trivial Laplace pressure

    real_rhs = pf.rhs

    def negated(st, so, params):
        phi_rhs, u_rhs, v_rhs, mu, mu_expl = real_rhs(st, so, params)
        return phi_rhs, -u_rhs, -v_rhs, mu, mu_expl

    monkeypatch.setattr(pf, "rhs", negated)
    flipped = np.asarray(pf.pressure_field(state, solid, p), dtype=np.float64)
    np.testing.assert_allclose(flipped, -reference, rtol=1e-5, atol=1e-9)

    def force_free(st, so, params):
        phi_rhs, u_rhs, v_rhs, mu, mu_expl = real_rhs(st, so, params)
        return phi_rhs, jnp.zeros_like(u_rhs), jnp.zeros_like(v_rhs), mu, mu_expl

    monkeypatch.setattr(pf, "rhs", force_free)
    assert float(np.abs(np.asarray(pf.pressure_field(state, solid, p))).max()) == 0.0
    out = pf.step(state, solid, p)
    assert float(jnp.abs(out.u).max()) == 0.0 and float(jnp.abs(out.v).max()) == 0.0


def test_capillary_formula_lives_only_in_rhs():
    """Static guard against a stale duplicated capillary formula (e.g. in a diagnostic)."""
    tree = ast.parse(inspect.getsource(pf))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    def names(fn):
        return {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}

    def calls(fn, target):
        return any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == target for n in ast.walk(fn)
        )

    assert {name for name, fn in functions.items() if "SIGMA_NORM" in names(fn)} == {"rhs"}
    for name in ("step", "pressure_field"):
        assert calls(functions[name], "rhs"), f"{name}() must obtain its forces from rhs()"
    assert not names(functions["pressure_field"]) & {"SIGMA_NORM", "chemical_potential", "fprime", "_lap"}
