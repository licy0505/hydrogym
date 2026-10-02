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
   drop.  Assertions on ``delta_p`` must stay signed -- never ``abs(delta_p)``;
4. the contact-angle measurement is accurate on synthetic caps of known
   geometric angle and the sessile initial state contains no liquid in the
   solid (L1A-2b, commit 1).

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


@pytest.fixture
def x64():
    """Run a test with JAX float64 enabled (restored afterwards)."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


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
    p = pf.PhaseFieldParams(
        Nx=64, Ny=64, Lx=6.0, Ly=6.0, phase_boundary_model="projection_legacy", enforce_solid_phi=True
    )
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


def test_solver_contract_is_v8():
    """The embedded cut-cell Young wall measure changes trajectory semantics: contract 7 -> 8."""
    assert pf.SOLVER_CONTRACT_VERSION == 8
    assert pf.WETTING_MODELS == ("surface_energy", "surface_energy_volume_v6", "legacy_affinity", "none")
    assert pf.WALL_MEASURE_METHODS == ("sdf_cutcell_v1", "diffuse_sdf_v7")
    assert pf.WALL_MEASURE_METHOD == "sdf_cutcell_v1"
    assert pf.WALL_MEASURE_CONTRACT_VERSION == 1
    params = pf.PhaseFieldParams(Nx=32, Ny=32)
    assert params.wetting_model == "surface_energy"
    assert params.phase_boundary_model == "impermeable_flux"
    assert params.wall_measure == "sdf_cutcell_v1"
    assert params.enforce_solid_phi is False
    # production defaults that must not move to make the contact angle come out right
    assert params.M == 2.0e-3 and params.ch_solver_rtol == 1.0e-6 and params.dtype is jnp.float32
    with pytest.raises(ValueError, match="wall_measure"):
        pf.PhaseFieldParams(Nx=32, Ny=32, wall_measure="cos_theta_over_f")


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
    for name in ("step_with_diagnostics", "pressure_field"):
        assert calls(functions[name], "rhs"), f"{name}() must obtain its forces from rhs()"
    assert calls(functions["step"], "step_with_diagnostics")
    assert not names(functions["pressure_field"]) & {"SIGMA_NORM", "chemical_potential", "fprime", "_lap"}


#######################################################################################
#           L1A-2b: audited contact-angle measurement and clean sessile setup          #
#######################################################################################


def _synthetic_cap(p, solid_theta_deg: float, R: float = 1.1, wall_height: float = 0.25):
    """Diffuse 2-D cap of known geometric contact angle on a flat wall (diagnostic only)."""
    X, Y = pf.grids(p)
    theta = np.deg2rad(solid_theta_deg)
    y_c = wall_height - R * np.cos(theta)
    r = jnp.sqrt((X - 0.5 * p.Lx) ** 2 + (Y - y_c) ** 2)
    return (0.5 * (1.0 - jnp.tanh((r - R) / (jnp.sqrt(2.0) * p.eps)))).astype(p.dtype)


def _flat_wall(p, theta_deg: float = 90.0, wall_height: float = 0.25):
    sdf = pf.surface_flat(p, wall_height=wall_height)
    solid = pf.make_solid(sdf, p, cos_theta=jnp.cos(jnp.deg2rad(theta_deg)))
    return sdf, solid


def test_synthetic_contact_angle_measurement():
    """The measurement must recover known geometric angles on synthetic caps (no solver)."""
    p = pf.PhaseFieldParams(Nx=128, Ny=128, Lx=6.0, Ly=6.0, dtype=jnp.float32)
    p.eps = 2.0 * p.dx
    sdf, solid = _flat_wall(p)
    targets = (60.0, 90.0, 120.0, 150.0)
    measured = []
    for target in targets:
        phi = _synthetic_cap(p, target)
        phi = jnp.where(sdf >= 0.0, phi, 0.0)
        measured.append(pf.measure_contact_angle(phi, solid, p))
    assert all(np.isfinite(measured))
    errors = np.abs(np.asarray(measured) - np.asarray(targets))
    assert errors.mean() <= 2.0
    assert errors.max() <= 3.0
    assert all(b >= a for a, b in zip(measured, measured[1:])), measured


def test_legacy_area_width_measurement_is_biased_for_obtuse_caps():
    """Legacy measurement is retained, but documented as unusable for theta > 90 deg."""
    p = pf.PhaseFieldParams(Nx=128, Ny=128, Lx=6.0, Ly=6.0, dtype=jnp.float32)
    p.eps = 2.0 * p.dx
    sdf, solid = _flat_wall(p)
    errors = []
    for target in (60.0, 90.0, 120.0, 150.0):
        phi = jnp.where(sdf >= 0.0, _synthetic_cap(p, target), 0.0)
        errors.append(abs(pf.measure_contact_angle_area_width(phi, solid, p) - target))
    assert np.mean(errors) > 5.0  # contract-v5 measurement defect, in numbers
    assert max(errors) > 10.0


def test_clean_sessile_initial_state_has_no_solid_liquid():
    """Target-independent clean start: no liquid in the geometric solid, same shape for all targets."""
    p = pf.PhaseFieldParams(Nx=96, Ny=96, Lx=6.0, Ly=6.0, dtype=jnp.float32)
    p.eps = 2.0 * p.dx
    _, solid = _flat_wall(p)
    states = []
    for target in (60.0, 90.0, 120.0, 150.0):
        solid_t = pf.make_solid(pf.surface_flat(p, wall_height=0.25), p, cos_theta=jnp.cos(np.deg2rad(target)))
        state = pf.sessile_initial_state(p, solid_t, R=1.1, wall_height=0.25)
        phi = np.asarray(state.phi, dtype=np.float64)
        distance = np.asarray(solid_t.sdf, dtype=np.float64)
        total = phi.sum() * p.dx * p.dy
        fluid = phi[distance >= 0.0].sum() * p.dx * p.dy
        assert total > 0.0
        assert abs(phi[distance < 0.0]).sum() * p.dx * p.dy / total < 1e-6
        assert fluid / total == pytest.approx(1.0, rel=1e-12)
        states.append(phi)
    for other in states[1:]:
        np.testing.assert_allclose(states[0], other, rtol=0, atol=0)


def test_clean_sessile_initial_state_is_a_90deg_cap():
    p = pf.PhaseFieldParams(Nx=128, Ny=128, Lx=6.0, Ly=6.0, dtype=jnp.float32)
    p.eps = 2.0 * p.dx
    _, solid = _flat_wall(p)
    state = pf.sessile_initial_state(p, solid, R=1.1, wall_height=0.25)
    assert float(jnp.max(jnp.abs(state.u))) == 0.0
    assert float(jnp.max(jnp.abs(state.v))) == 0.0
    assert pf.measure_contact_angle(state.phi, solid, p) == pytest.approx(90.0, abs=0.5)


def test_contact_angle_measurement_is_rotation_invariant_across_the_x_seam():
    """A drop crossing the periodic x seam must measure the same angle."""
    p = pf.PhaseFieldParams(Nx=128, Ny=128, Lx=6.0, Ly=6.0, dtype=jnp.float32)
    p.eps = 2.0 * p.dx
    X, Y = pf.grids(p)
    sdf, solid = _flat_wall(p)
    theta = np.deg2rad(120.0)
    y_c = 0.25 - 1.1 * np.cos(theta)
    angles = []
    for X0 in (3.0, 0.0, 5.9):
        sx = (X - X0 + 0.5 * p.Lx) % p.Lx - 0.5 * p.Lx
        r = jnp.sqrt(sx**2 + (Y - y_c) ** 2)
        phi = jnp.where(sdf >= 0.0, 0.5 * (1.0 - jnp.tanh((r - 1.1) / (jnp.sqrt(2.0) * p.eps))), 0.0)
        angles.append(pf.measure_contact_angle(phi, solid, p))
    assert all(np.isfinite(angles))
    assert max(angles) - min(angles) < 0.5
    assert all(abs(a - 120.0) < 1.0 for a in angles)


#######################################################################################
#        L1A-2b: Young-consistent wall surface energy (solver contract v6)             #
#######################################################################################


def test_surface_delta_normal_integral(x64):
    """Integral of delta_wall across the wall must be 1 and converge with N."""
    errors = {}
    for N in (64, 96, 128):
        p = pf.PhaseFieldParams(Nx=N, Ny=N, Lx=6.0, Ly=6.0, dtype=jnp.float64)
        sdf = pf.surface_flat(p, wall_height=0.25)
        delta = np.asarray(pf.wall_delta(sdf, p), dtype=np.float64)
        errors[N] = abs(float(delta[N // 2, :].sum() * p.dy) - 1.0)
    assert all(error <= 0.03 for error in errors.values()), errors
    assert errors[128] < errors[96] < errors[64], errors  # a converging quadrature, not a renormalization


def test_surface_delta_has_no_periodic_y_ghost(x64):
    """The shipped delta is localized; a periodic y-derivative would inject a seam ghost."""
    p = pf.PhaseFieldParams(Nx=128, Ny=128, Lx=6.0, Ly=6.0, dtype=jnp.float64)
    sdf = pf.surface_flat(p, wall_height=0.25)
    delta = np.asarray(pf.wall_delta(sdf, p), dtype=np.float64)
    distance = np.asarray(sdf, dtype=np.float64)
    peak = float(np.abs(delta).max())
    far = np.abs(distance) > 6.0 * p.wall_delta_width
    assert float(np.abs(delta[far]).max() / peak) < 1e-3  # no material top/far-field ghost

    gy = (np.roll(distance, -1, axis=1) - np.roll(distance, 1, axis=1)) / (2.0 * p.dy)
    gx = (np.roll(distance, -1, axis=0) - np.roll(distance, 1, axis=0)) / (2.0 * p.dx)
    periodic = (
        (1.0 / (2.0 * p.wall_delta_width))
        * (1.0 - np.tanh(np.clip(distance / p.wall_delta_width, -60.0, 60.0)) ** 2)
        * np.sqrt(gx**2 + gy**2 + 1e-30)
    )
    periodic_integral = float(periodic[64, :].sum() * p.dy)
    assert abs(periodic_integral - 1.0) > 0.03  # the ghost is real, so the test is not vacuous
    assert float(np.abs(periodic - delta).max() / peak) > 1e-2


def test_wall_energy_young_endpoint_difference(x64):
    for target in (60.0, 90.0, 120.0, 150.0):
        cos_theta = float(np.cos(np.deg2rad(target)))
        g0 = float(pf.wall_energy_density(jnp.asarray(0.0, dtype=jnp.float64), cos_theta))
        g1 = float(pf.wall_energy_density(jnp.asarray(1.0, dtype=jnp.float64), cos_theta))
        assert g0 - g1 == pytest.approx(pf.WALL_SIGMA0 * cos_theta, rel=1e-10, abs=1e-12)
    assert float(pf.WALL_SIGMA0) == pytest.approx(float(np.sqrt(2.0) / 6.0), rel=1e-12)


def test_wall_energy_90deg_is_neutral(x64):
    """theta = 90 deg is exactly neutral: g_w == 0 and mu_wall == 0 to round-off."""
    p = pf.PhaseFieldParams(Nx=64, Ny=64, Lx=6.0, Ly=6.0, dtype=jnp.float64)
    sdf = pf.surface_flat(p, wall_height=0.25)
    # cos(90 deg) is 6.1e-17 in float64 (4.4e-8 in float32): the model must be neutral
    # to that accuracy, not "small but finite".
    solid = pf.make_solid(sdf, p, cos_theta=float(np.cos(np.deg2rad(90.0))))
    phi = jnp.linspace(0.0, 1.0, 64, dtype=p.dtype)
    assert float(jnp.max(jnp.abs(pf.wall_energy_density(phi, solid.cos_theta)))) <= 1e-15
    assert float(jnp.max(jnp.abs(pf.wetting_mu(jnp.broadcast_to(phi[None, :], (64, 64)), solid, p)))) <= 1e-15
    p32 = pf.PhaseFieldParams(Nx=32, Ny=32, Lx=6.0, Ly=6.0, dtype=jnp.float32, wall_energy_amp=5.0)
    solid32 = pf.make_solid(pf.surface_flat(p32, wall_height=0.25), p32, cos_theta=float(np.cos(np.deg2rad(90.0))))
    mu32 = float(jnp.max(jnp.abs(pf.wetting_mu(jnp.full((32, 32), 0.5, dtype=p32.dtype), solid32, p32))))
    assert mu32 <= 1e-5  # float32 round-off only


def test_wall_energy_hydrophilic_hydrophobic_sign(x64):
    """cos(theta) > 0 pulls phi up at the wall (liquid favoured); cos(theta) < 0 pushes it down."""
    p = pf.PhaseFieldParams(Nx=64, Ny=64, Lx=6.0, Ly=6.0, dtype=jnp.float64, wetting_model="surface_energy_volume_v6")
    sdf = pf.surface_flat(p, wall_height=0.25)
    phi = jnp.full((64, 64), 0.5, dtype=p.dtype)
    delta = np.asarray(pf.wall_delta(sdf, p), dtype=np.float64)
    cell = np.unravel_index(int(np.argmax(delta)), delta.shape)
    for target, expected_sign in ((60.0, -1.0), (120.0, +1.0)):
        cos_theta = float(np.cos(np.deg2rad(target)))
        solid = pf.make_solid(sdf, p, cos_theta=cos_theta)
        mu_wall = float(pf.wetting_mu(phi, solid, p)[cell])
        assert np.sign(mu_wall) == expected_sign, (target, mu_wall)
        gamma_sl = float(pf.wall_energy_density(jnp.asarray(1.0, dtype=jnp.float64), cos_theta))
        gamma_sg = float(pf.wall_energy_density(jnp.asarray(0.0, dtype=jnp.float64), cos_theta))
        if target < 90.0:
            assert gamma_sl < gamma_sg  # liquid-covered wall is the lower-energy state
        else:
            assert gamma_sl > gamma_sg  # gas is favoured at the wall
    assert np.sign(float(pf.wetting_mu(phi, solid, p)[cell])) == np.sign(-float(np.cos(np.deg2rad(120.0))))


def test_wall_energy_directional_derivative_matches_mu(x64):
    """mu_wall must be the variational derivative of F_wall = int g_w(phi) delta_wall dV (float64)."""
    p = pf.PhaseFieldParams(Nx=32, Ny=32, Lx=6.0, Ly=6.0, dtype=jnp.float64, wetting_model="surface_energy_volume_v6")
    sdf = pf.surface_flat(p, wall_height=0.25)
    x = np.linspace(0.0, 6.0, 32, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    phi = np.clip(0.5 + 0.35 * np.cos(2.0 * np.pi * X / 6.0) * np.cos(np.pi * Y / 6.0), 0.05, 0.95)
    for target in (60.0, 90.0, 120.0):
        cos_theta = float(np.cos(np.deg2rad(target)))
        solid = pf.make_solid(sdf, p, cos_theta=cos_theta)
        delta = np.asarray(pf.wall_delta(solid.sdf, p), dtype=np.float64)
        direction = delta  # aligned with the only place the wall term acts

        def free_energy(field):
            return float(np.sum(np.asarray(pf.wall_energy_density(field, cos_theta)) * delta) * p.dx * p.dy)

        mu_wall = np.asarray(pf.wetting_mu(jnp.asarray(phi), solid, p), dtype=np.float64)
        inner = float(np.sum(mu_wall * direction) * p.dx * p.dy)
        if target == 90.0:
            assert abs(inner) <= 1e-12  # neutral wall: no wall work for any direction
            continue
        for amplitude in (1e-5, 1e-4):
            fd = (free_energy(phi + amplitude * direction) - free_energy(phi - amplitude * direction)) / (2 * amplitude)
            assert abs(fd - inner) / abs(inner) <= 1e-4, (target, amplitude, fd, inner)


def test_wetting_mu_dispatch_fails_closed_and_none_is_zero():
    p = pf.PhaseFieldParams(Nx=32, Ny=32, Lx=6.0, Ly=6.0, wetting_model="none")
    solid = pf.make_solid(pf.surface_flat(p), p, cos_theta=0.7)
    phi = jnp.full((32, 32), 0.4, dtype=p.dtype)
    assert float(jnp.max(jnp.abs(pf.wetting_mu(phi, solid, p)))) == 0.0
    with pytest.raises(ValueError):
        pf.PhaseFieldParams(Nx=32, Ny=32, wetting_model="guess_an_angle")
    p_bad = pf.PhaseFieldParams(Nx=32, Ny=32)
    object.__setattr__(p_bad, "wetting_model", "not_a_model")
    with pytest.raises(ValueError):
        pf.wetting_mu(phi, solid, p_bad)


def test_surface_energy_mode_ignores_legacy_wetting_parameters():
    """wall_energy_amp / wet_band must never act as hidden calibration in surface_energy mode."""
    base = dict(Nx=32, Ny=32, Lx=6.0, Ly=6.0, wetting_model="surface_energy")
    phi = jnp.full((32, 32), 0.5, dtype=jnp.float32)
    results = []
    for amp, band in ((5.0, 0.15), (0.5, 0.08), (50.0, 0.4)):
        p = pf.PhaseFieldParams(**base, wall_energy_amp=amp, wet_band=band)
        solid = pf.make_solid(pf.surface_flat(p), p, cos_theta=0.5)
        results.append(np.asarray(pf.wetting_mu(phi, solid, p), dtype=np.float64))
    np.testing.assert_array_equal(results[0], results[1])
    np.testing.assert_array_equal(results[0], results[2])


def test_legacy_affinity_mode_still_available_for_reproducibility():
    p = pf.PhaseFieldParams(
        Nx=32, Ny=32, Lx=6.0, Ly=6.0, wetting_model="legacy_affinity", wall_energy_amp=5.0, wet_band=0.15
    )
    solid = pf.make_solid(pf.surface_flat(p), p, cos_theta=0.5)
    phi = jnp.full((32, 32), 0.5, dtype=p.dtype)
    mu_legacy = np.asarray(pf.wetting_mu(phi, solid, p), dtype=np.float64)
    p_new = pf.PhaseFieldParams(Nx=32, Ny=32, Lx=6.0, Ly=6.0, wetting_model="surface_energy")
    mu_new = np.asarray(pf.wetting_mu(phi, pf.make_solid(pf.surface_flat(p_new), p_new, cos_theta=0.5), p_new))
    assert not np.allclose(mu_legacy, mu_new)  # the semantics really did change
    phi_w = float(pf.phi_wet_of(jnp.asarray(0.5)))
    assert phi_w == pytest.approx(0.75)


def test_contact_angle_targets_have_correct_direction_ci():
    """CI-sized direction check: a hydrophilic target spreads, a hydrophobic one retracts.

    Deliberately small (N=48, R=0.6 -> the drop radius is only ~5 cells) so it asserts
    *direction* and contract behaviour, never a quantitative angle; the matrices live in
    ``production/wall_measure_audit.py`` and ``production/embedded_young_audit.py``.

    Two signals are checked, at two horizons, because the seeded 90 deg cap first relaxes its
    *shape*: at this resolution the circle fit of a 300-step transient is dominated by that
    relaxation (with the contract-v8 measure both targets briefly move the wrong way before the
    wetting signal takes over at ~500 steps). The contact width is a direct wetting-direction
    observable and is correct from the start; the fitted angle ordering is asserted at 900 steps,
    where the pinned contract-v7 measure and the v8 measure agree.
    """
    import math

    from production import observables as obs

    wall_height = 0.25
    measured = []
    widths = []
    for target in (60.0, 120.0):
        p = pf.PhaseFieldParams(Nx=48, Ny=48, Lx=6.0, Ly=6.0, dt=4e-3)
        p.eps = 2.0 * p.dx
        p.dt = min(float(p.dt), float(pf.stable_dt(p, u_max=2.0)))
        sdf = pf.surface_flat(p, wall_height=wall_height)
        solid = pf.make_solid(sdf, p, cos_theta=math.cos(math.radians(target)))
        state = pf.sessile_initial_state(p, solid, R=0.6, wall_height=wall_height)
        mass0 = obs.liquid_mass(state.phi, solid.sdf, p.dx, p.dy)
        assert float(jnp.min(state.phi[solid.sdf < 0.0])) == 0.0
        assert float(jnp.sum(solid.wall_area)) == pytest.approx(p.Lx, rel=1e-5)  # v8 measure is the wall length
        step_fn = jax.jit(pf.step, static_argnums=(2,))
        width_early = None
        for index in range(900):
            state = step_fn(state, solid, p)
            if index + 1 == 300:
                width_early = float(pf.spreading_width(state.phi, p))
        mass1 = obs.liquid_mass(state.phi, solid.sdf, p.dx, p.dy)
        angle = float(pf.measure_contact_angle(state.phi, solid, p))
        assert math.isfinite(angle) and 0.0 < angle < 180.0
        assert abs(mass1 - mass0) / mass0 <= 1e-3
        measured.append(angle)
        widths.append(width_early)
    assert widths[0] > widths[1], widths  # hydrophilic spreads wider than hydrophobic by step 300
    assert measured[0] < measured[1], measured  # and the fitted angle ordering follows by step 900


# ---------------------------------------------------------------------------
#  L1A-2c: conservative impermeable phase boundary and natural Young BC
# ---------------------------------------------------------------------------


def _phase_boundary_setup(N=64, *, dtype=jnp.float32, theta=90.0, rtol=1e-6):
    import math

    p = pf.PhaseFieldParams(
        Nx=N,
        Ny=N,
        Lx=6.0,
        Ly=6.0,
        dt=4.0e-3,
        dtype=dtype,
        eps=2.0 * 6.0 / N,
        ch_solver_rtol=rtol,
    )
    solid = pf.make_solid(pf.surface_flat(p, wall_height=0.25), p, cos_theta=math.cos(math.radians(theta)))
    return p, solid


def test_phase_advective_flux_zero_across_solid_faces():
    p, _flat_solid = _phase_boundary_setup()
    X, Y = pf.grids(p)
    pillar = pf.sdf_box(X, Y, 2.5, 3.5, 0.25, 0.8)
    solid = pf.make_solid(pf.sdf_union(pf.surface_flat(p, 0.25), pillar), p, cos_theta=0.0)
    rng = np.random.default_rng(17)
    phi = jnp.asarray(rng.uniform(0.0, 1.0, (p.Nx, p.Ny)), dtype=p.dtype)
    u = jnp.asarray(rng.normal(size=phi.shape), dtype=p.dtype)
    v = jnp.asarray(rng.normal(size=phi.shape), dtype=p.dtype)
    fluid = np.asarray(solid.sdf >= 0.0)
    ax, ay = (np.asarray(item) for item in pf.fluid_face_apertures(solid, p))
    fx, fy = (np.asarray(item) for item in pf.phase_advective_fluxes(u, v, phi, solid, p))
    crossing_x = fluid ^ np.roll(fluid, -1, axis=0)
    crossing_y = fluid ^ np.roll(fluid, -1, axis=1)
    assert np.any(crossing_x) and np.any(crossing_y)
    assert not ax[crossing_x].any() and not ay[crossing_y].any()
    assert np.max(np.abs(fx[crossing_x])) == 0.0
    assert np.max(np.abs(fy[crossing_y])) == 0.0


def test_phase_ch_flux_zero_across_solid_faces():
    p, _flat_solid = _phase_boundary_setup()
    X, Y = pf.grids(p)
    pillar = pf.sdf_box(X, Y, 2.5, 3.5, 0.25, 0.8)
    solid = pf.make_solid(pf.sdf_union(pf.surface_flat(p, 0.25), pillar), p, cos_theta=0.0)
    mu = (Y - 0.25).astype(p.dtype)  # a chemical-potential gradient points into the bottom wall
    fluid = np.asarray(solid.sdf >= 0.0)
    crossing_x = fluid ^ np.roll(fluid, -1, axis=0)
    crossing_y = fluid ^ np.roll(fluid, -1, axis=1)
    jx, jy = (np.asarray(item) for item in pf.chemical_potential_fluxes(mu, solid, p))
    assert np.max(np.abs(jx[crossing_x]), initial=0.0) == 0.0
    assert np.max(np.abs(jy[crossing_y]), initial=0.0) == 0.0


def test_face_flux_divergence_conserves_fluid_mass(x64):
    p, solid = _phase_boundary_setup(N=48, dtype=jnp.float64, rtol=1e-8)
    rng = np.random.default_rng(23)
    phi = jnp.asarray(rng.uniform(0.0, 1.0, (p.Nx, p.Ny)), dtype=p.dtype)
    mu = jnp.asarray(rng.normal(size=phi.shape), dtype=p.dtype)
    u = jnp.asarray(rng.normal(size=phi.shape), dtype=p.dtype)
    v = jnp.asarray(rng.normal(size=phi.shape), dtype=p.dtype)
    ax, ay = pf.phase_advective_fluxes(u, v, phi, solid, p)
    cx, cy = pf.chemical_potential_fluxes(mu, solid, p)
    divergence = pf.divergence_from_face_fluxes(ax + cx, ay + cy, p)
    fluid = jnp.asarray(solid.sdf >= 0.0)
    assert abs(float(jnp.sum(jnp.where(fluid, divergence, 0.0)))) <= 1e-10


def test_phase_boundary_blocks_periodic_y_wrap_at_bottom_wall():
    p, solid = _phase_boundary_setup(N=48)
    fluid = np.asarray(solid.sdf >= 0.0)
    aperture_x, aperture_y = pf.fluid_face_apertures(solid, p)
    aperture_y = np.asarray(aperture_y)
    assert fluid[:, -1].all() and not fluid[:, 0].any()
    assert np.asarray(aperture_x).any()  # x-periodic fluid-fluid faces remain available
    assert not aperture_y[:, -1].any()
    phi = jnp.ones((p.Nx, p.Ny), dtype=p.dtype)
    velocity = jnp.ones_like(phi)
    _, y_flux = pf.phase_advective_fluxes(velocity, velocity, phi, solid, p)
    assert float(jnp.max(jnp.abs(y_flux[:, -1]))) == 0.0


def _natural_bc_probe(sdf, p, theta_deg, *, region=None, profile="exact"):
    """Relative error of the natural Young BC on the cells the production measure forces.

    ``profile='exact'`` uses the analytic solution of the *nonlinear* condition,
    ``phi = 0.5 (1 - tanh(cos(theta) sdf / (sqrt(2) eps)))``, which satisfies
    ``eps dphi/dn + g_w'(phi) = 0`` at every distance from the wall, so only the second-order
    stencil error remains. ``profile='linear'`` uses the tangent linearization at ``phi = 0.5``
    and is compared against the wall-plane value ``-g_w'(0.5)/eps``: a linear field is
    differentiated exactly, so that error is round-off and it pins the sign/normal convention.
    """
    import math

    cos_theta = math.cos(math.radians(theta_deg))
    solid = pf.make_solid(sdf, p, cos_theta=cos_theta)
    if profile == "exact":
        phi = 0.5 * (1.0 - jnp.tanh(cos_theta * sdf / (math.sqrt(2.0) * p.eps)))
    elif profile == "linear":
        q_sdf = float(pf.wall_energy_derivative(jnp.asarray(0.5, dtype=p.dtype), cos_theta)) / p.eps
        phi = 0.5 + q_sdf * sdf
    else:
        raise ValueError(profile)
    gx = (jnp.roll(phi, -1, axis=0) - jnp.roll(phi, 1, axis=0)) / (2.0 * p.dx)
    gy = pf._ddy_nonperiodic(phi, p.dy)
    nx, ny = pf.wall_measure_normal(solid, p)
    measured = nx * gx + ny * gy
    if profile == "exact":
        prescribed = pf.natural_wall_normal_derivative(phi, solid, p)
    else:
        prescribed = jnp.full_like(
            phi, -float(pf.wall_energy_derivative(jnp.asarray(0.5, dtype=p.dtype), cos_theta)) / p.eps
        )
    weight = np.asarray(pf.wall_measure_density(solid, p), dtype=np.float64)
    peak = float(weight[region].max()) if region is not None else float(weight.max())
    valid = weight > 0.9 * peak
    if region is not None:
        valid &= region
    assert valid.any(), "the wall measure selected no active cell"
    error = np.asarray(measured - prescribed, dtype=np.float64)[valid]
    scale = max(float(np.max(np.abs(np.asarray(prescribed)[valid]))), 1e-30)
    return float(np.max(np.abs(error)) / scale)


def test_flat_wall_natural_contact_bc(x64):
    p, solid = _phase_boundary_setup(N=64, dtype=jnp.float64, theta=60.0)
    # linearized probe: exact stencil, so this pins the sign and normal convention
    assert _natural_bc_probe(solid.sdf, p, 60.0, profile="linear") < 1.0e-9
    # exact nonlinear natural-BC profile: only second-order truncation remains at eps/dx = 2
    assert _natural_bc_probe(solid.sdf, p, 60.0) < 0.02


@pytest.mark.parametrize("slope", [-0.45, 0.45])
def test_inclined_wall_natural_contact_bc_orientation(slope, x64):
    p = pf.PhaseFieldParams(Nx=96, Ny=96, Lx=6.0, Ly=6.0, eps=2.0 * 6.0 / 96, dtype=jnp.float64)
    X, Y = pf.grids(p)
    sdf = (Y - slope * (X - 3.0) - 0.25) / np.sqrt(1.0 + slope * slope)
    region = (np.asarray(X) > 2.5) & (np.asarray(X) < 3.5)
    assert _natural_bc_probe(sdf, p, 120.0, region=region, profile="linear") < 1.0e-9
    assert _natural_bc_probe(sdf, p, 120.0, region=region) < 0.02


def test_neutral_90_wall_has_zero_natural_contact_bc_source(x64):
    p, solid = _phase_boundary_setup(N=48, dtype=jnp.float64, theta=90.0, rtol=1e-8)
    phi = jnp.full((p.Nx, p.Ny), 0.5, dtype=p.dtype)
    assert float(jnp.max(jnp.abs(pf.natural_wall_normal_derivative(phi, solid, p)))) <= 1e-14
    assert float(jnp.max(jnp.abs(pf._natural_wall_laplacian_flux(phi, solid, p)))) <= 1e-14
    assert float(jnp.max(jnp.abs(pf.wetting_mu(phi, solid, p)))) == 0.0


def test_v7_path_does_not_call_mass_redistribution_projection(monkeypatch):
    p, solid = _phase_boundary_setup(N=32)
    state = pf.sessile_initial_state(p, solid, R=0.6, wall_height=0.25)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("v7 impermeable-flux path called the legacy mass projection")

    monkeypatch.setattr(pf, "_project_phase_outside_solid", forbidden)
    updated = pf.step(state, solid, p)
    assert np.isfinite(np.asarray(updated.phi)).all()
    with pytest.raises(ValueError, match="legacy-only"):
        pf.PhaseFieldParams(Nx=32, Ny=32, enforce_solid_phi=True)


def test_solid_phase_leak_remains_near_zero_without_projection():
    p, solid = _phase_boundary_setup(N=48)
    state = pf.sessile_initial_state(p, solid, R=0.8, wall_height=0.25)
    initial_mass = float(pf.liquid_mass(state.phi, solid, p))
    step_fn = jax.jit(pf.step, static_argnums=(2,))
    for _ in range(150):
        state = step_fn(state, solid, p)
    solid_mass = float(jnp.sum(state.phi * solid.chi_hard) * p.dx * p.dy)
    fluid_mass = float(pf.liquid_mass(state.phi, solid, p))
    assert solid_mass / initial_mass <= 1e-6
    assert abs(fluid_mass - initial_mass) / initial_mass <= 1e-3


def test_ch_implicit_solver_converges_and_reports_residual(x64):
    p, solid = _phase_boundary_setup(N=48, dtype=jnp.float64, rtol=1e-8)
    rng = np.random.default_rng(31)
    right = jnp.asarray(rng.normal(size=(p.Nx, p.Ny)), dtype=p.dtype)
    right = jnp.where(solid.sdf >= 0.0, right, 0.0)
    dt = p.dt / 3.0
    solution, info = pf.solve_ch_implicit(right, solid, p, dt)
    alpha = dt * p.M * p.eps
    lap = pf.fluid_laplacian(solution, solid, p)
    residual = solution + alpha * pf.fluid_laplacian(lap, solid, p) - right
    relative = float(jnp.linalg.norm(residual) / jnp.linalg.norm(right))
    assert bool(info.converged)
    assert int(info.iterations) <= p.ch_solver_max_iterations
    assert relative <= 1e-8
    assert float(info.relative_residual) <= 1e-8


def test_ch_implicit_solver_fail_closed_on_nonconvergence():
    p, solid = _phase_boundary_setup(N=48)
    p.ch_solver_max_iterations = 1
    rng = np.random.default_rng(33)
    right = jnp.asarray(rng.normal(size=(p.Nx, p.Ny)), dtype=p.dtype)
    solution, info = pf.solve_ch_implicit(right, solid, p, p.dt / 3.0)
    assert not bool(info.converged)
    assert not np.isfinite(np.asarray(solution)).all()


def test_v7_implicit_phase_solve_has_a_finite_custom_adjoint():
    p, solid = _phase_boundary_setup(N=16)
    rng = np.random.default_rng(37)
    phi = jnp.asarray(rng.uniform(0.1, 0.9, size=(p.Nx, p.Ny)), dtype=p.dtype)
    u = jnp.zeros_like(phi)
    v = jnp.zeros_like(phi)

    def objective(field):
        updated, _info = pf.phase_transport_step(field, u, v, solid, p)
        return jnp.sum(updated**2)

    gradient = jax.grad(objective)(phi)
    assert np.isfinite(np.asarray(gradient)).all()
    assert float(jnp.linalg.norm(gradient)) > 0.0
