"""Physics validation of the phase-field droplet solver (run before any ML)."""

import time

import jax
import jax.numpy as jnp
import numpy as np

import phasefield as pf

jax.config.update("jax_platform_name", "cpu")


def run(state, solid, p, n, chunk=100):
    for i in range(0, n, chunk):
        state = pf.advance(state, solid, p, min(chunk, n - i))
    return state


print("=" * 70)
print("TEST 1  static droplet: mass conservation + Laplace law  dp = sigma / R")
print("=" * 70)
for R in (0.8, 1.1):
    p = pf.PhaseFieldParams(Nx=192, Ny=192, Lx=6.0, Ly=6.0, Re=200.0, We=100.0, dt=4e-3)
    solid = pf.empty_solid(p)
    st = pf.droplet_initial_state(p, x0=3.0, y0=3.0, R=R, u_impact=0.0)
    m0 = float(pf.liquid_mass(st.phi, solid, p))
    t0 = time.time()
    st = run(st, solid, p, 1000)
    wall = time.time() - t0
    m1 = float(pf.liquid_mass(st.phi, solid, p))
    ke = float(jnp.sum(0.5 * pf.rho_of(st.phi, p) * (st.u**2 + st.v**2)) * p.dx * p.dy)
    pr = pf.pressure_field(st, solid, p)
    # pressure inside (average over a small disc at the centre) vs outside (ring)
    X, Y = pf.grids(p)
    r = jnp.sqrt((X - 3.0) ** 2 + (Y - 3.0) ** 2)
    p_in = float(jnp.sum(pr * (r < 0.3 * R)) / jnp.sum(r < 0.3 * R))
    p_out = float(jnp.sum(pr * (r > 2.5 * R)) / jnp.sum(r > 2.5 * R))
    dp = p_in - p_out
    print(f"  R={R}:  mass {m0:.5f} -> {m1:.5f}  (drift {100 * (m1 - m0) / m0:+.3f} %)")
    print(f"         dp={dp:.4e}   dp*R*We={dp * R * p.We:.4f}   (Laplace: should be ~1)")
    print(f"         residual KE={ke:.3e}   1000 steps in {wall:.1f} s ({1000 / wall:.0f} steps/s)")

print()
print("=" * 70)
print("TEST 2  sessile drop: apparent contact angle vs. target")
print("=" * 70)
for target in (30.0, 60.0, 90.0, 120.0, 150.0):
    p = pf.PhaseFieldParams(Nx=192, Ny=192, Lx=6.0, Ly=6.0, Re=200.0, We=100.0, dt=4e-3)
    sdf = pf.surface_flat(p, wall_height=0.25)
    solid = pf.make_solid(sdf, p, cos_theta=np.cos(np.deg2rad(target)))
    X, Y = pf.grids(p)
    R = 1.1
    r = jnp.sqrt((X - 3.0) ** 2 + (Y - 1.2) ** 2)
    phi = 0.5 * (1.0 - jnp.tanh((r - R) / (jnp.sqrt(2.0) * p.eps)))
    st = pf.State(phi=phi.astype(p.dtype), u=jnp.zeros_like(phi), v=jnp.zeros_like(phi), t=0.0)
    st = run(st, solid, p, 3000)
    ang = pf.measure_contact_angle(st.phi, solid, p)
    print(f"  target {target:5.1f} deg  ->  measured {ang:6.2f} deg   (err {ang - target:+.2f})")
