"""
Two-phase (liquid/gas) Cahn-Hilliard-Navier-Stokes solver for droplet impact
============================================================================

A small, self-contained, fully differentiable 2-D solver for *phase distribution*
of a liquid droplet impacting a solid surface, written in the style of HydroGym's
JAX backend (``hydrogym.jax``): a config dataclass, a pure ``step`` function, and
a ``lax.scan`` rollout.

Model
-----
Conservative phase-field (Cahn-Hilliard) coupled to incompressible Navier-Stokes
with one-fluid properties, Korteweg/CSF capillary force and Brinkman volume
penalization for the solid (wall + micro-structures):

    d(phi)/dt + div(u phi) = M * lap(mu) + wall_energy_term
    mu = f'(phi)/eps - eps * lap(phi)
    du/dt + div(u u) = -grad(p)/rho + div(nu grad u)
                       - (1/We) mu grad(phi) / rho - (1/Fr^2) (rho-<rho>)/rho yhat
                       - (chi/eta) u
    div(u) = 0

* phi = 1 liquid, phi = 0 gas, f(phi) = phi^2 (1-phi)^2, so the equilibrium
  interface profile is phi = 0.5 (1 - tanh((r-R)/(sqrt(2) eps))) and the surface
  tension of that profile is sigma = sqrt(2)/6 per unit energy prefactor.  The
  Korteweg force therefore carries a factor 6/sqrt(2) = 3 sqrt(2) so that the
  non-dimensional surface tension is exactly 1/We (verified by the Laplace test
  in ``validate_physics.py``).
* rho(phi) = rho_g + (rho_l - rho_g) phi, likewise for nu
* solid geometry enters through the indicator chi in [0, 1] (1 = solid) and the
  surface delta ds = |grad chi|, which carries the wetting (contact angle) energy
* contact angle may vary in space, so mixed-wettability surfaces are supported

The solver is periodic in both directions; the wall is a solid slab inside the
domain, so no special boundary treatment is needed for the FFT Poisson solve.

Intended use inside HydroGym
----------------------------
``PhaseFieldFlow``/``step``/``rollout`` are the pieces a ``hydrogym.jax``
environment needs: a flow config with ``initialize_state``, a differentiable
stepper, and a ``lax.scan`` rollout.  See ``README.md`` in this folder for how to
wrap it as a Gymnasium/Gymnax environment and how to swap in m-AIA level-set or
JAX-Fluids two-phase data for 3-D.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import lax


# f(phi) = phi^2 (1-phi)^2  ->  sigma = sqrt(2)/6, so the Korteweg force is
# rescaled by SIGMA_NORM = 6/sqrt(2) to make the non-dimensional sigma = 1/We.
SIGMA_NORM = 6.0 / jnp.sqrt(2.0)


#######################################################################################
#                                                                                     #
#                             STATE / PARAMETERS                                      #
#                                                                                     #
#######################################################################################


class State(NamedTuple):
    """Solver state: phase field + velocity."""

    phi: jnp.ndarray  # (Nx, Ny) in [0, 1], 1 = liquid
    u: jnp.ndarray  # (Nx, Ny) x-velocity
    v: jnp.ndarray  # (Nx, Ny) y-velocity
    t: float


class Solid(NamedTuple):
    """Immutable description of the solid surface (wall + micro-structure)."""

    chi: jnp.ndarray  # (Nx, Ny) indicator, 1 = solid
    ds: jnp.ndarray  # (Nx, Ny) surface delta |grad chi|
    cos_theta: jnp.ndarray  # (Nx, Ny) cos(contact angle) evaluated on the solid surface
    sdf: jnp.ndarray  # (Nx, Ny) signed distance to the solid, <0 inside solid
    chi_hard: jnp.ndarray  # (Nx, Ny) 0/1 mask of the solid interior (impermeable)


@dataclass(eq=False)  # eq=False keeps identity hashing, so params can be a jit static arg
class PhaseFieldParams:
    """Non-dimensional parameters of the two-phase solver."""

    # grid / domain
    Nx: int = 192
    Ny: int = 192
    Lx: float = 6.0
    Ly: float = 6.0

    # physics (non-dimensionalised with droplet diameter D=1, rho_l, impact speed U)
    Re: float = 200.0  # rho_l U D / mu_l
    We: float = 100.0  # rho_l U^2 D / sigma
    Fr: float = 1e6  # U / sqrt(g D)  (1e6 ~ no gravity)
    rho_l: float = 1.0
    rho_g: float = 0.1
    nu_l: float = None  # defaults to 1/Re
    nu_g: float = None  # defaults to nu_l * 10
    M: float = 2.0e-3  # Cahn-Hilliard mobility
    eps: float = None  # interface thickness, defaults to 1.5 * dx

    # numerics
    dt: float = 2.0e-3
    eta_pen: float = None  # penalization timescale, defaults to 2*dt
    wall_energy_amp: float = 5.0  # amplitude of the (conservative) wetting energy
    wet_band: float = 0.15  # half-width of the near-wall wetting band
    cfl: float = 0.4  # used by ``stable_dt``
    use_gravity: bool = False
    dtype: type = jnp.float32

    # extras that the RL / surrogate layer likes to have around
    extras: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.eps is None:
            self.eps = 1.5 * self.Lx / self.Nx
        if self.nu_l is None:
            self.nu_l = 1.0 / self.Re
        if self.nu_g is None:
            self.nu_g = 10.0 * self.nu_l
        if self.eta_pen is None:
            self.eta_pen = 2.0 * self.dt

    @property
    def dx(self) -> float:
        return self.Lx / self.Nx

    @property
    def dy(self) -> float:
        return self.Ly / self.Ny

    @property
    def m2(self):
        """Fourier symbol of the 5-point Laplacian used by the finite-difference
        stencils: m2 = (2-2cos(kx dx))/dx^2 + (2-2cos(ky dy))/dy^2.

        Inverting *this* symbol (rather than |k|^2) in the pressure Poisson solve
        makes the projection exactly consistent with the central-difference
        divergence/gradient, so the projected velocity is divergence-free to
        machine precision and no odd-even (checkerboard) mode is excited.  The
        same symbol is used for the implicit Cahn-Hilliard biharmonic operator.
        """
        kx = jnp.fft.fftfreq(self.Nx, d=self.dx) * 2.0 * jnp.pi
        ky = jnp.fft.rfftfreq(self.Ny, d=self.dy) * 2.0 * jnp.pi
        m2 = (2.0 - 2.0 * jnp.cos(kx * self.dx))[:, None] / self.dx**2 + (2.0 - 2.0 * jnp.cos(ky * self.dy))[
            None, :
        ] / self.dy**2
        return m2.astype(jnp.float64 if self.dtype == jnp.float64 else jnp.float32)

    @property
    def m2_proj(self):
        """Fourier symbol of grad_c . grad_c, the operator that the pressure
        projection must invert.

        The projection removes the divergence built with the *same* central
        differences that appear in ``_ddx``/``_ddy``, whose symbol is
        sin(k dx)/dx -- not the 2 sin(k dx/2)/dx of the 5-point Laplacian.  Using
        the 5-point symbol here makes ``div(u)`` grow by a factor up to 2 at every
        projection instead of vanishing, which is a fatal odd-even instability.
        With this symbol the projected field is divergence-free (in the
        finite-difference sense used by the advection fluxes) to round-off; the
        Nyquist mode lies in the null space of the discrete divergence and is
        damped by viscosity.
        """
        kx = jnp.fft.fftfreq(self.Nx, d=self.dx) * 2.0 * jnp.pi
        ky = jnp.fft.rfftfreq(self.Ny, d=self.dy) * 2.0 * jnp.pi
        m2 = (jnp.sin(kx * self.dx) / self.dx)[:, None] ** 2 + (jnp.sin(ky * self.dy) / self.dy)[None, :] ** 2
        return m2.astype(jnp.float64 if self.dtype == jnp.float64 else jnp.float32)


#######################################################################################
#                                                                                     #
#                             GEOMETRY (micro-structures)                             #
#                                                                                     #
#######################################################################################


def smooth_indicator(sdf: jnp.ndarray, dx: float) -> jnp.ndarray:
    """Smoothed Heaviside of a signed distance field (1 inside the solid)."""
    return 0.5 * (1.0 - jnp.tanh(sdf / (1.5 * dx)))


def surface_delta(chi: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """|grad chi| -- a diffuse surface measure used to apply wall energies."""
    gx = (jnp.roll(chi, -1, axis=0) - jnp.roll(chi, 1, axis=0)) / (2.0 * dx)
    gy = (jnp.roll(chi, -1, axis=1) - jnp.roll(chi, 1, axis=1)) / (2.0 * dy)
    return jnp.sqrt(gx**2 + gy**2 + 1e-12)


def make_solid(
    sdf: jnp.ndarray,
    params: PhaseFieldParams,
    cos_theta: float | jnp.ndarray = jnp.cos(jnp.deg2rad(90.0)),
) -> Solid:
    """Build a :class:`Solid` from a signed-distance field (negative inside solid).

    ``cos_theta`` may be a scalar (uniform wettability) or a full (Nx, Ny) array
    (spatially patterned wettability).
    """
    chi = smooth_indicator(sdf, params.dx).astype(params.dtype)
    ds = surface_delta(chi, params.dx, params.dy).astype(params.dtype)
    if jnp.ndim(cos_theta) == 0:
        cos_theta = jnp.full_like(chi, float(cos_theta))
    return Solid(
        chi=chi,
        ds=ds,
        cos_theta=cos_theta.astype(params.dtype),
        sdf=sdf.astype(params.dtype),
        chi_hard=(chi > 0.5).astype(params.dtype),
    )


def grids(params: PhaseFieldParams):
    x = (jnp.arange(params.Nx) + 0.5) * params.dx
    y = (jnp.arange(params.Ny) + 0.5) * params.dy
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    return X, Y


def sdf_box(X, Y, x0, x1, y0, y1):
    """Signed distance to an axis-aligned box (negative inside)."""
    dx_ = jnp.maximum(jnp.maximum(x0 - X, X - x1), 0.0)
    dy_ = jnp.maximum(jnp.maximum(y0 - Y, Y - y1), 0.0)
    outside = jnp.sqrt(dx_**2 + dy_**2)
    inside = jnp.minimum(jnp.maximum(x0 - X, jnp.maximum(X - x1, jnp.maximum(y0 - Y, Y - y1))), 0.0)
    return outside + inside


def sdf_union(*sdfs):
    return jnp.min(jnp.stack(sdfs, axis=0), axis=0)


# -------------------------------------------------------------------------------------
#  Surface generators: each returns a signed-distance field (negative inside solid)
# -------------------------------------------------------------------------------------


def surface_flat(params: PhaseFieldParams, wall_height: float = 0.25, **_) -> jnp.ndarray:
    """Plain flat wall at the bottom of the domain."""
    _, Y = grids(params)
    return Y - wall_height


def surface_pillars(
    params: PhaseFieldParams,
    wall_height: float = 0.25,
    n_pillars: int = 4,
    width: float = 0.3,
    height: float = 0.4,
    center: float | None = None,
    **_,
) -> jnp.ndarray:
    """Periodic rectangular pillar array sitting on a flat wall."""
    X, Y = grids(params)
    if center is None:
        center = params.Lx / 2.0
    pitch = params.Lx / n_pillars
    # distance to the nearest pillar centre-line (periodic in x)
    rel = (X - center + 0.5 * pitch) % pitch - 0.5 * pitch
    pillar = sdf_box(rel, Y, -width / 2, width / 2, wall_height, wall_height + height)
    return sdf_union(pillar, Y - wall_height)


def surface_random_pillars(
    params: PhaseFieldParams,
    wall_height: float = 0.25,
    n_pillars: int = 7,
    width_range: Tuple[float, float] = (0.15, 0.4),
    height_range: Tuple[float, float] = (0.2, 0.7),
    seed: int = 0,
    **_,
) -> jnp.ndarray:
    """Randomly placed/sized pillars -- an *unseen* complex surface."""
    import numpy as np

    rng = np.random.default_rng(seed)
    X, Y = grids(params)
    xs = np.sort(rng.uniform(0.3, params.Lx - 0.3, size=n_pillars))
    # enforce a minimum gap so pillars do not merge into a wall
    for i in range(1, n_pillars):
        if xs[i] - xs[i - 1] < 0.5:
            xs[i] = xs[i - 1] + 0.5
    ws = rng.uniform(*width_range, size=n_pillars)
    hs = rng.uniform(*height_range, size=n_pillars)
    sdfs = [Y - wall_height]
    for xc, w, h in zip(xs, ws, hs):
        if xc + w / 2 > params.Lx - 0.05:
            continue
        sdfs.append(sdf_box(X, Y, xc - w / 2, xc + w / 2, wall_height, wall_height + h))
    return sdf_union(*sdfs)


def surface_grooves(
    params: PhaseFieldParams,
    wall_height: float = 0.25,
    n_grooves: int = 6,
    width: float = 0.25,
    depth: float = 0.35,
    **_,
) -> jnp.ndarray:
    """Flat wall with rectangular grooves cut into it (negative structures)."""
    _, Y = grids(params)
    X, _ = grids(params)
    pitch = params.Lx / n_grooves
    rel = (X + 0.5 * pitch) % pitch - 0.5 * pitch
    wall = Y - wall_height
    groove = sdf_box(rel, Y, -width / 2, width / 2, wall_height - depth, wall_height)
    # solid = wall minus the groove boxes  ->  sdf = max(sdf_wall, -sdf_groove)
    return jnp.maximum(wall, -groove)


def surface_hierarchical(
    params: PhaseFieldParams,
    wall_height: float = 0.25,
    n_pillars: int = 3,
    width: float = 0.6,
    height: float = 0.5,
    n_sub: int = 3,
    sub_width: float = 0.1,
    sub_height: float = 0.15,
    **_,
) -> jnp.ndarray:
    """Two-scale (micro/nano) pillars: big pillars carrying small pillars."""
    X, Y = grids(params)
    pitch = params.Lx / n_pillars
    rel = (X + 0.5 * pitch) % pitch - 0.5 * pitch
    big = sdf_box(rel, Y, -width / 2, width / 2, wall_height, wall_height + height)
    sub_pitch = width / n_sub
    rel_sub = (rel + 0.5 * sub_pitch) % sub_pitch - 0.5 * sub_pitch
    top = wall_height + height
    small = sdf_box(rel_sub, Y, -sub_width / 2, sub_width / 2, top, top + sub_height)
    small = jnp.where(jnp.abs(rel) < width / 2, small, 1e3 * jnp.ones_like(small))
    return sdf_union(big, small, Y - wall_height)


def surface_wedge(params: PhaseFieldParams, wall_height: float = 0.25, slope: float = 0.5, **_) -> jnp.ndarray:
    """Inclined wall (asymmetric spreading test)."""
    X, Y = grids(params)
    return Y - (wall_height + slope * (X - params.Lx / 2))


SURFACE_REGISTRY = {
    "flat": surface_flat,
    "pillars": surface_pillars,
    "random_pillars": surface_random_pillars,
    "grooves": surface_grooves,
    "hierarchical": surface_hierarchical,
    "wedge": surface_wedge,
}


#######################################################################################
#                                                                                     #
#                             SOLVER                                                  #
#                                                                                     #
#######################################################################################


def _ddx(f, dx):
    return (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2.0 * dx)


def _ddy(f, dy):
    return (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2.0 * dy)


def _lap(f, dx, dy):
    return (jnp.roll(f, -1, axis=0) - 2 * f + jnp.roll(f, 1, axis=0)) / dx**2 + (
        jnp.roll(f, -1, axis=1) - 2 * f + jnp.roll(f, 1, axis=1)
    ) / dy**2


def _advect_flux(u, q, dx, axis):
    """3rd-order upwind flux of ``u*q`` across faces normal to ``axis``."""
    if axis == 0:
        u_f = 0.5 * (u + jnp.roll(u, -1, axis=0))
        q_m1, q_0, q_p1, q_p2 = (
            jnp.roll(q, 1, axis=0),
            q,
            jnp.roll(q, -1, axis=0),
            jnp.roll(q, -2, axis=0),
        )
    else:
        u_f = 0.5 * (u + jnp.roll(u, -1, axis=1))
        q_m1, q_0, q_p1, q_p2 = (
            jnp.roll(q, 1, axis=1),
            q,
            jnp.roll(q, -1, axis=1),
            jnp.roll(q, -2, axis=1),
        )
    left = -q_m1 / 6.0 + 5.0 * q_0 / 6.0 + q_p1 / 3.0
    right = q_0 / 3.0 + 5.0 * q_p1 / 6.0 - q_p2 / 6.0
    q_f = jnp.where(u_f > 0, left, right)
    return u_f * q_f / dx


def div_upwind(u, v, q, dx, dy):
    """div(u q) with 3rd-order upwinding."""
    fx = _advect_flux(u, q, dx, 0)
    fy = _advect_flux(v, q, dy, 1)
    return (fx - jnp.roll(fx, 1, axis=0)) + (fy - jnp.roll(fy, 1, axis=1))


def fprime(phi):
    """f'(phi) for f(phi) = phi^2 (1-phi)^2."""
    return 2.0 * phi * (1.0 - phi) * (1.0 - 2.0 * phi)


def phi_wet_of(cos_theta):
    """Map a (possibly patterned) contact-angle cosine to a wall target value in [0,1].

    phi_w = 1 is strongly solvophilic (liquid-loving -> small apparent angle),
    phi_w = 0 is solvophobic (beading -> large angle), phi_w = 0.5 ~ neutral.
    The linear map is a convenient control; the *achieved* apparent angle is
    measured/calibrated by ``measure_contact_angle`` (see validation).
    """
    return 0.5 + 0.5 * jnp.clip(cos_theta, -1.0, 1.0)


def wet_band(solid: Solid, p: PhaseFieldParams):
    return 0.5 * (1.0 - jnp.tanh(solid.sdf / p.wet_band))


def wetting_mu(phi, solid: Solid, p: PhaseFieldParams):
    """Conservative surface-affinity contribution to the chemical potential.

    Enters mu (hence the Cahn-Hilliard flux M*grad(mu)) so it is in divergence form
    and conserves liquid mass, while still biasing the contact line toward phi_w.
    """
    phi_w = phi_wet_of(solid.cos_theta)
    band = wet_band(solid, p)
    return -p.wall_energy_amp * band * (phi_w - phi)


def chemical_potential(phi, solid: Solid, p: PhaseFieldParams):
    """mu = f'(phi)/eps - eps lap(phi) + conservative wetting affinity."""
    return fprime(phi) / p.eps - p.eps * _lap(phi, p.dx, p.dy) + wetting_mu(phi, solid, p)


def poisson_solve(rhs, m2):
    """Periodic solve of the *discrete* Poisson problem lap(x) = rhs by FFT.

    ``m2`` is the Fourier symbol of the same 5-point Laplacian that the
    finite-difference stencils realise (see :attr:`PhaseFieldParams.m2`).
    """
    rhs_hat = jnp.fft.rfft2(rhs)
    m2_safe = m2.at[0, 0].set(1.0)
    # the Laplacian has symbol -m2, hence the minus sign
    x_hat = (-rhs_hat / m2_safe).at[0, 0].set(0.0)
    return jnp.fft.irfft2(x_hat, s=rhs.shape)


def rho_of(phi, p: PhaseFieldParams):
    return p.rho_g + (p.rho_l - p.rho_g) * phi


def nu_of(phi, p: PhaseFieldParams):
    return p.nu_g + (p.nu_l - p.nu_g) * phi


def rhs(state: State, solid: Solid, p: PhaseFieldParams):
    """Explicit right-hand sides and the explicitly-treated chemical potential.

    ``mu_expl`` contains every part of the chemical potential except the
    stabilising ``-eps*lap(phi)`` term, which is integrated implicitly.
    """
    phi, u, v = state.phi, state.u, state.v
    dx, dy = p.dx, p.dy

    mu = chemical_potential(phi, solid, p)
    mu_x, mu_y = _ddx(mu, dx), _ddy(mu, dy)

    # mobility is damped inside the solid so that the order parameter there is
    # essentially slaved to the wetting boundary condition
    mob = 1.0 - 0.9 * solid.chi
    mu_expl = fprime(phi) / p.eps + wetting_mu(phi, solid, p)

    # --- phase field: advection + wetting reaction, biharmonic diffusion implicit ---
    phi_rhs = -div_upwind(u, v, phi, dx, dy)

    # --- momentum ---
    rho = rho_of(phi, p)
    nu = nu_of(phi, p)
    adv_u = div_upwind(u, v, u, dx, dy)
    adv_v = div_upwind(u, v, v, dx, dy)
    lap_u, lap_v = _lap(u, dx, dy), _lap(v, dx, dy)

    # capillary (Korteweg) force  -(1/We) mu grad(phi)
    cap_x = -(SIGMA_NORM / p.We) * mu * mu_x / rho
    cap_y = -(SIGMA_NORM / p.We) * mu * mu_y / rho

    if p.use_gravity:
        g_y = -(1.0 / p.Fr**2) * (rho - jnp.mean(rho)) / rho
    else:
        g_y = jnp.zeros_like(phi)

    u_rhs = -adv_u + nu * lap_u + cap_x
    v_rhs = -adv_v + nu * lap_v + cap_y + g_y

    return phi_rhs, u_rhs, v_rhs, mu, mob * mu_expl


def step(state: State, solid: Solid, p: PhaseFieldParams) -> State:
    """One semi-implicit SSP-RK3 step.

    The 4th-order Cahn-Hilliard diffusion is integrated implicitly in Fourier
    space inside every stage, which removes the O(eps^4/M) stability restriction;
    the Brinkman penalization of the solid is implicit as well.
    """
    dt = p.dt
    m2 = p.m2
    denom = 1.0 + dt * p.M * p.eps * m2**2

    def substep(carry, _):
        phi, u, v, t = carry
        phi_rhs, u_rhs, v_rhs, mu, mu_expl = rhs(State(phi, u, v, t), solid, p)

        # implicit CH diffusion:  (I + dt M eps^2 lap^2) phi+ = phi + dt * explicit
        # lap(mu_expl) == -k2 * rfft2(mu_expl)
        source_hat = jnp.fft.rfft2(phi_rhs) - p.M * m2 * jnp.fft.rfft2(mu_expl)
        phi_hat = (jnp.fft.rfft2(phi) + dt * source_hat) / denom
        phi_new = jnp.fft.irfft2(phi_hat, s=phi.shape)

        # Brinkman penalization, implicit -> unconditionally stable
        damp = 1.0 / (1.0 + dt * solid.chi / p.eta_pen)
        u_new = (u + dt * u_rhs) * damp
        v_new = (v + dt * v_rhs) * damp

        # pressure projection (inverts grad_c . grad_c, see PhaseFieldParams.m2_proj)
        div = _ddx(u_new, p.dx) + _ddy(v_new, p.dy)
        pr = poisson_solve(div / dt, p.m2_proj)
        u_new = u_new - dt * _ddx(pr, p.dx)
        v_new = v_new - dt * _ddy(pr, p.dy)
        return (phi_new, u_new, v_new, t + dt), None

    (phi, u, v, t), _ = lax.scan(substep, (state.phi, state.u, state.v, state.t), None, length=3)
    return State(phi=phi, u=u, v=v, t=t)


def stable_dt(p: PhaseFieldParams, u_max: float = 2.0) -> float:
    dx = min(p.dx, p.dy)
    return p.cfl * dx / u_max


def droplet_initial_state(
    p: PhaseFieldParams,
    x0: float = 3.0,
    y0: float = 3.0,
    R: float = 0.5,
    u_impact: float = 1.0,
) -> State:
    """Circular droplet of radius R centred at (x0, y0) moving downwards at u_impact."""
    X, Y = grids(p)
    r = jnp.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
    phi = 0.5 * (1.0 - jnp.tanh((r - R) / (jnp.sqrt(2.0) * p.eps)))
    # momentum-consistent translation of the droplet only: v = -U0 rho_l phi / rho(phi)
    rho = rho_of(phi, p)
    v = -u_impact * p.rho_l * phi / rho
    u = jnp.zeros_like(phi)
    return State(phi=phi.astype(p.dtype), u=u.astype(p.dtype), v=v.astype(p.dtype), t=0.0)


def advance(state: State, solid: Solid, p: PhaseFieldParams, n_steps: int) -> State:
    """Integrate ``n_steps`` steps without saving anything."""
    return lax.fori_loop(0, n_steps, lambda _i, s: step(s, solid, p), state)


def rollout(
    state: State,
    solid: Solid,
    p: PhaseFieldParams,
    n_steps: int,
    save_every: int = 1,
):
    """Integrate ``n_steps`` steps with ``lax.scan``; return (final, saved_states).

    Saved states are stacked as arrays of shape (n_saved, Nx, Ny).
    """

    def body(carry, _):
        s, buf_phi, buf_u, buf_v, i = carry
        s = step(s, solid, p)
        save = (i % save_every) == 0
        buf_phi = jnp.where(save, buf_phi.at[i // save_every].set(s.phi), buf_phi)
        buf_u = jnp.where(save, buf_u.at[i // save_every].set(s.u), buf_u)
        buf_v = jnp.where(save, buf_v.at[i // save_every].set(s.v), buf_v)
        return (s, buf_phi, buf_u, buf_v, i + 1), None

    save_every = max(1, min(save_every, n_steps))
    n_saved = n_steps // save_every
    buf_phi = jnp.zeros((n_saved, p.Nx, p.Ny), dtype=p.dtype)
    buf_u = jnp.zeros_like(buf_phi)
    buf_v = jnp.zeros_like(buf_phi)
    (final, phi_hist, u_hist, v_hist, _), _ = lax.scan(body, (state, buf_phi, buf_u, buf_v, 0), None, length=n_steps)
    return final, phi_hist, u_hist, v_hist


#######################################################################################
#                                                                                     #
#                             DIAGNOSTICS / OBSERVABLES                               #
#                                                                                     #
#######################################################################################


def liquid_mass(phi, solid: Solid, p: PhaseFieldParams):
    """Liquid area (2-D 'mass'), excluding whatever sits inside the solid."""
    return jnp.sum(phi * (1.0 - solid.chi)) * p.dx * p.dy


def spreading_width(phi, p: PhaseFieldParams, thresh: float = 0.5):
    """Horizontal extent of the liquid, D(t) -- the classic impact observable."""
    mask = phi > thresh
    x = jnp.arange(p.Nx) * p.dx
    xs = jnp.where(mask.any(axis=1), x, jnp.nan)
    return jnp.nanmax(xs) - jnp.nanmin(xs)


def contact_area(phi, solid: Solid, p: PhaseFieldParams, band: float = 0.15):
    """Liquid area within ``band`` of the solid surface (how far it wets/penetrates)."""
    near = (solid.sdf > -band) & (solid.sdf < band)
    return jnp.sum(phi * near) * p.dx * p.dy


def penetration_depth(phi, solid: Solid, p: PhaseFieldParams, y_wall: float = 0.25, thresh: float = 0.5):
    """Mean liquid volume fraction between the pillar tips and the wall."""
    _, Y = grids(p)
    zone = (Y < y_wall + 0.5) & (Y > y_wall)
    return jnp.sum(phi * zone) / jnp.maximum(jnp.sum(zone), 1.0)


def measure_contact_angle(phi, solid: Solid, p: PhaseFieldParams, level: float = 0.5):
    """Apparent contact angle of a (near-static) drop from its area and wetted width.

    Uses the 2-D circular-cap relation: with wetted width w and area A,
    solve A = R^2 (theta - sin theta cos theta), w = 2 R sin theta for theta.
    """
    import numpy as np
    from scipy.optimize import brentq

    A = float(liquid_mass(phi, solid, p))
    w = float(spreading_width(phi, p, level))
    if w <= 0 or A <= 0:
        return float("nan")

    def resid(theta):
        R = w / (2.0 * np.sin(theta))
        return R**2 * (theta - np.sin(theta) * np.cos(theta)) - A

    lo, hi = 1e-3, np.pi - 1e-3
    try:
        theta = brentq(resid, lo, hi)
    except ValueError:
        return float("nan")
    return float(np.rad2deg(theta))


def pressure_field(state: State, solid: Solid, p: PhaseFieldParams):
    """Projection pressure of the current step (used for diagnostics, e.g. Laplace law)."""
    _, u_rhs, v_rhs, _, _ = rhs(state, solid, p)
    damp = 1.0 / (1.0 + p.dt * solid.chi / p.eta_pen)
    u_new = (state.u + p.dt * u_rhs) * damp
    v_new = (state.v + p.dt * v_rhs) * damp
    div = _ddx(u_new, p.dx) + _ddy(v_new, p.dy)
    return poisson_solve(div / p.dt, p.m2_proj)


def empty_solid(p: PhaseFieldParams) -> Solid:
    """A solid-free domain (useful for validation cases)."""
    z = jnp.zeros((p.Nx, p.Ny), dtype=p.dtype)
    return Solid(chi=z, ds=z, cos_theta=z, sdf=jnp.ones_like(z), chi_hard=jnp.zeros_like(z))


def build_case(case: dict, N: int = 192, dt: float = 4e-3):
    """Materialise (params, solid, initial_state) from a case dict (see cases.py)."""
    p = PhaseFieldParams(Nx=N, Ny=N, Lx=6.0, Ly=6.0, Re=case.get("Re", 200.0), We=case.get("We", 100.0), dt=dt)
    gen = SURFACE_REGISTRY[case.get("surface", "flat")]
    surf_kwargs = {
        k: v
        for k, v in case.items()
        if k
        in (
            "wall_height",
            "n_pillars",
            "width",
            "height",
            "n_grooves",
            "depth",
            "n_sub",
            "sub_width",
            "sub_height",
            "slope",
            "width_range",
            "height_range",
            "seed",
            "center",
        )
    }
    sdf = gen(p, **surf_kwargs)
    solid = make_solid(sdf, p, cos_theta=float(case.get("cos_theta", 0.0)))
    R = case.get("R", 0.7)
    y0 = case.get("y0", 0.25 + R - 0.1)
    state = droplet_initial_state(p, x0=3.0, y0=y0, R=R, u_impact=case.get("u_impact", 0.5))
    return p, solid, state


def downsample(field, f):
    """Average-pool by factor f on both axes."""
    Nx, Ny = field.shape
    return field.reshape(Nx // f, f, Ny // f, f).mean(axis=(1, 3))
