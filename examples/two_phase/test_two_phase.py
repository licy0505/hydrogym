"""
Lightweight physics tests for the two-phase droplet-impact solver.

These are fast enough for CI and skip cleanly when JAX is not installed (JAX is an
optional extra in HydroGym).  They verify the two properties that the whole
learning pipeline relies on:

1. liquid mass is (nearly) conserved by the Cahn--Hilliard + advection update;
2. the pressure projection leaves the velocity divergence-free to round-off
   (i.e. the discrete grad/div/Laplacian symbols are mutually consistent).

Run with::

    pytest examples/two_phase/test_two_phase.py
"""

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
