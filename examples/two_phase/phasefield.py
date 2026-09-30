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

    d(phi)/dt + div(u phi) = M * lap(mu)                      (conservative; no source)
    mu = f'(phi)/eps - eps * lap(phi) + mu_wall               (mu = dF/dphi)
    du/dt + div(u u) = -grad(P) + div(nu grad u)
                       + (SIGMA_NORM/We) mu grad(phi) / rho_l - (1/Fr^2) (rho-<rho>)/rho yhat
                       - (chi/eta) u
    div(u) = 0

  P is the projection pressure of a constant-coefficient projection and the
  capillary acceleration divides by the constant ``rho_l``.  Neither is the
  variable-density form (known blockers P-VARDENS-PROJ and P-CAP-RHO, see
  ``production/validation.py``); they are documented here as implemented.

* phi = 1 liquid, phi = 0 gas, f(phi) = phi^2 (1-phi)^2, so the equilibrium
  interface profile is phi = 0.5 (1 - tanh((r-R)/(sqrt(2) eps))) and the surface
  tension of that profile is sigma = sqrt(2)/6 per unit energy prefactor.  The
  Korteweg force therefore carries a factor 6/sqrt(2) = 3 sqrt(2) so that the
  non-dimensional surface tension is exactly 1/We (checked by the static-droplet
  Laplace benchmark in ``production/`` and analytically by
  ``production/capillary_audit.py``).

Contact-angle measurement (SOLVER_CONTRACT_VERSION >= 6)
--------------------------------------------------------
``measure_contact_angle`` fits a circle to the ``phi = 0.5`` contour and evaluates
``theta = acos((y_w - y_c) / Rc)``.  The contract-v5 area/full-width inversion is
retained as ``measure_contact_angle_area_width`` but is *diagnostic only*: it is
only consistent for caps of at most half a circle, and on synthetic caps of known
angle (N = 128) it shows MAE 11.9 deg / max 31.7 deg, while the circle fit
recovers them with MAE 0.007 deg / max 0.016 deg.

Capillary sign convention (SOLVER_CONTRACT_VERSION >= 5)
--------------------------------------------------------
* Orientation.  phi = 1 in the liquid, phi = 0 in the gas.  Across the interface
  dphi/dr < 0, so grad(phi) points gas -> liquid (outside -> inside of a drop).
  The outward normal (liquid -> gas) is n_out = -grad(phi) / |grad(phi)|.
* Chemical potential.  mu = dF/dphi for F = int f(phi)/eps + eps/2 |grad phi|^2.
  At the interface of a convex liquid drop mu = +eps |dphi/dr| / r > 0
  (Gibbs-Thomson; the relaxed value is sigma/R).
* Force.  Cahn-Hilliard advection changes the free energy at the rate
  dF/dt = -int mu u.grad(phi), so a force that does not create free energy is
  + mu grad(phi), equivalently - phi grad(mu): the two differ by the pure
  gradient grad(mu phi), which the pressure absorbs.  The implemented form is
  F = +(SIGMA_NORM/We) mu grad(phi) / rho_l.  It points toward the liquid
  (F . n_out < 0), i.e. toward the centre of curvature.  Contract v4 used the
  opposite sign, a negative surface tension: the capillary-driven flow raised the
  interfacial free energy and the projection pressure of a convex drop was lower
  in the liquid than in the gas.
* Pressure.  The projection removes the irrotational part of the force,
  u+ = u* - dt grad(P), lap(P) = div(u*)/dt, so a static drop has grad(P) = F and
  P_liquid - P_gas = -int F_r dr.  With phi = 1 in liquid and phi = 0 in gas, this
  sign yields positive p_liquid - p_gas for a convex liquid droplet under the
  current mu convention: delta_p = +1/(We R) for a 2-D circle.
  ``pressure_field`` re-evaluates ``rhs`` and the same projection, so it inherits
  this convention; the capillary formula must not be duplicated anywhere else.
* rho(phi) = rho_g + (rho_l - rho_g) phi, likewise for nu
* solid geometry enters through the indicator chi in [0, 1] (1 = solid, used by
  the Brinkman penalization) and the analytic signed distance sdf, which carries
  the wetting (contact angle) energy
* contact angle may vary in space, so mixed-wettability surfaces are supported

Wall wetting (SOLVER_CONTRACT_VERSION >= 6)
-------------------------------------------
The bulk free energy is F_bulk = int [ f(phi)/eps + eps/2 |grad phi|^2 ] dV with
f(phi) = phi^2 (1-phi)^2, whose equilibrium tanh profile carries the surface
tension sigma_0 = sqrt(2)/6 (the Korteweg force is scaled by SIGMA_NORM = 1/sigma_0
so the non-dimensional tension is exactly 1/We).  The wall free energy uses the
same normalization:

    F_wall  = int g_w(phi, theta) delta_wall(sdf) dV
    g_w     = -sigma_0 cos(theta) h(phi),      h(phi) = phi^2 (3 - 2 phi)
    mu_wall = dg_w/dphi * delta_wall = -sigma_0 cos(theta) h'(phi) delta_wall
    delta_wall(sdf) = (1/2a) sech^2(sdf/a) |grad sdf|,   a = 1.5 dx

so that g_w(0) = 0, g_w(1) = -sigma_0 cos(theta) and

    gamma_SG - gamma_SL = g_w(0) - g_w(1) = sigma_0 cos(theta_e)   (Young),

with theta = 90 deg exactly neutral (g_w == 0) and h'(0) = h'(1) = 0 so no wall
force leaks into either bulk phase.  There is no empirical amplitude, no gain and
no theta -> theta mapping: ``mu_wall`` is the exact variational derivative of
``F_wall`` (checked by ``production/wetting_audit.py``).

``wetting_model`` selects the semantics: ``surface_energy`` (default),
``legacy_affinity`` (the contract-v5 volumetric band affinity,
``wall_energy_amp``/``wet_band``; reproducibility only) or ``none`` (ablation).
Unknown values fail closed.  The legacy parameters are ignored in
``surface_energy`` mode and must never be used to fit the apparent angle.

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

import math
from dataclasses import dataclass, field
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import lax

# f(phi) = phi^2 (1-phi)^2  ->  sigma = sqrt(2)/6, so the Korteweg force is
# rescaled by SIGMA_NORM = 6/sqrt(2) to make the non-dimensional sigma = 1/We.
SIGMA_NORM = 6.0 / jnp.sqrt(2.0)

# Bump this whenever the solver/data contract changes in a trajectory-changing way.
#   5: L1A-2a -- Korteweg capillary force sign corrected to +mu grad(phi) (see the module
#      docstring); the static-drop projection pressure now has p_liquid > p_gas.
#   6: L1A-2b -- the default wall wetting semantics are the Young-consistent diffuse wall
#      surface energy ``g_w(phi, theta) = -sigma_0 cos(theta) h(phi)`` with
#      ``mu_wall = dg_w/dphi * delta_wall``; the contract-v5 volumetric affinity is kept
#      behind ``wetting_model='legacy_affinity'`` for reproducibility only.  Every v5
#      trajectory is trajectory-stale under v6 (dataset fingerprints include this number).
SOLVER_CONTRACT_VERSION = 6


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
    # Wall wetting model.  ``surface_energy`` (default, contract v6) uses the
    # Young-consistent diffuse wall energy defined in the module docstring;
    # ``legacy_affinity`` reproduces the contract-v5 volumetric band affinity for
    # reproducibility only, and ``none`` switches wetting off (ablation).
    wetting_model: str = "surface_energy"
    # Width of the normalized diffuse wall delta.  Defaults to the solid smoothing
    # width ``1.5 dx`` used by ``smooth_indicator``; it is a discretization width,
    # never a contact-angle calibration knob.
    wall_delta_width: float = None
    # LEGACY ONLY (wetting_model='legacy_affinity'): amplitude and half-width of the
    # contract-v5 near-wall affinity.  Both are ignored by ``surface_energy`` and must
    # never be used as a hidden calibration factor for the Young angle.
    wall_energy_amp: float = 5.0
    wet_band: float = 0.15
    cfl: float = 0.4  # used by ``stable_dt``
    use_gravity: bool = False
    dtype: type = jnp.float32
    # If enabled, each sub-step uses a bounded, mass-conserving projection that
    # removes phi from the geometric solid without deleting liquid mass.
    enforce_solid_phi: bool = False

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
        if self.wall_delta_width is None:
            # Match ``smooth_indicator``: the same width that smears the solid
            # indicator is the width of the wall surface delta.
            self.wall_delta_width = 1.5 * self.Lx / self.Nx
        if self.wetting_model not in WETTING_MODELS:
            raise ValueError(f"unknown wetting_model {self.wetting_model!r}; expected one of {sorted(WETTING_MODELS)}")

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
        # Exact geometric solid mask.  When ``enforce_solid_phi`` is enabled
        # the phase projection below keeps phi out of these cells while
        # preserving total liquid mass.
        chi_hard=(sdf < 0.0).astype(params.dtype),
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


def _local_surface_top(
    sdf: jnp.ndarray,
    p: PhaseFieldParams,
    x0: float,
    radius: float,
    margin: float | None = None,
) -> jnp.ndarray:
    """Return the highest solid cell beneath a local, periodically wrapped footprint.

    A global ``max(Y[sdf < 0])`` is unsafe for textured surfaces: a tall pillar
    on the other side of the domain can move a drop that is nowhere near it.
    The support search is therefore restricted to the periodic x-distance of the
    drop footprint plus a small diffuse-interface margin.  The zero fallback is
    finite and is only used for a degenerate SDF with no solid cell in that
    local window.
    """
    X, Y = grids(p)
    if margin is None:
        margin = max(2.0 * float(p.eps), float(p.dx))
    periodic_dx = (X - float(x0) + 0.5 * p.Lx) % p.Lx - 0.5 * p.Lx
    support = jnp.abs(periodic_dx) <= float(radius) + float(margin)
    valid = support & (sdf < 0.0) & jnp.isfinite(sdf) & jnp.isfinite(Y)
    local_top = jnp.max(jnp.where(valid, Y, jnp.asarray(0.0, dtype=Y.dtype)))
    return jnp.where(jnp.any(valid), local_top, jnp.asarray(0.0, dtype=Y.dtype))


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


def surface_wedge(
    params: PhaseFieldParams,
    wall_height: float = 0.25,
    slope: float = 0.5,
    center: float | None = None,
    **_,
) -> jnp.ndarray:
    """Periodic locally-inclined wall used as the asymmetric-spreading test."""
    X, Y = grids(params)
    if center is None:
        center = params.Lx / 2.0
    wave_number = 6.0 * jnp.pi / params.Lx
    phase = wave_number * (X - center)
    amplitude = slope / wave_number
    height = wall_height + amplitude * jnp.sin(phase)
    dh_dx = slope * jnp.cos(phase)
    return (Y - height) / jnp.sqrt(1.0 + dh_dx**2)


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


#: Surface tension carried by the equilibrium tanh profile with
#: f(phi) = phi^2 (1-phi)^2: sigma_0 = sqrt(2)/6.  The Korteweg force is scaled by
#: SIGMA_NORM = 1/sigma_0 (see the module docstring), so ``sigma_0`` is the *actual*
#: non-dimensional surface tension and is the correct prefactor for wall energies.
#: Evaluated in pure Python: the constant must not depend on ``jax_enable_x64``.
WALL_SIGMA0 = math.sqrt(2.0) / 6.0

#: Wall wetting models accepted by :func:`wetting_mu`.
WETTING_MODELS = ("surface_energy", "legacy_affinity", "none")


def phi_wet_of(cos_theta):
    """LEGACY (``wetting_model='legacy_affinity'``) target wall value in [0, 1].

    phi_w = 1 is strongly solvophilic, phi_w = 0 solvophobic, phi_w = 0.5 neutral.
    This linear map is a *control*, not a thermodynamic relation, and the achieved
    angle was measured to be non-monotonic in it; it is retained only so the
    contract-v5 trajectories can be reproduced.  Do not use it in
    ``surface_energy`` mode.
    """
    return 0.5 + 0.5 * jnp.clip(cos_theta, -1.0, 1.0)


def wall_switch(phi):
    """Smooth wall interpolation h(phi) = phi^2 (3 - 2 phi), h(0) = 0, h(1) = 1."""
    return phi**2 * (3.0 - 2.0 * phi)


def wall_switch_derivative(phi):
    """h'(phi) = 6 phi (1 - phi); identically zero at both endpoints."""
    return 6.0 * phi * (1.0 - phi)


def wall_energy_density(phi, cos_theta):
    """Young-consistent wall free-energy density g_w(phi, theta), per unit wall area.

        g_w(phi, theta) = -sigma_0 cos(theta) h(phi)

    with g_w(0) = 0 and g_w(1) = -sigma_0 cos(theta), so that the endpoint
    difference obeys Young's relation

        gamma_SG - gamma_SL = g_w(0) - g_w(1) = sigma_0 cos(theta_e).

    ``theta = 90 deg`` is exactly neutral (g_w identically zero), which also makes
    the model symmetric between liquid and gas.  There is no empirical amplitude.
    """
    return -WALL_SIGMA0 * jnp.asarray(cos_theta) * wall_switch(phi)


def _ddy_nonperiodic(field, dy):
    """y-derivative with one-sided edges: the solid slab is not periodic in y.

    A periodic roll would see the sdf jump from the top of the domain to the solid
    and inject a spurious |grad sdf| (and hence a fake wall surface) at the y seam.
    """
    interior = (field[:, 2:] - field[:, :-2]) / (2.0 * dy)
    left = ((field[:, 1] - field[:, 0]) / dy)[:, None]
    right = ((field[:, -1] - field[:, -2]) / dy)[:, None]
    return jnp.concatenate([left, interior, right], axis=1)


def wall_delta(sdf, p: PhaseFieldParams, width: float | None = None):
    """Normalized, ghost-free diffuse wall surface measure.

        delta_wall(sdf) = (1 / 2a) sech^2(sdf / a) |grad sdf|,   a = p.wall_delta_width

    * localized at sdf = 0 and normalized: the normal integral is 1 (0.07 % on the
      N = 128 baseline grid; the audit in ``production/wetting_audit.py`` reports it
      for N = 64 / 96 / 128);
    * evaluated from the analytic sdf, *never* from |grad chi| of the periodic solid
      indicator, which carries a fake peak at the y seam;
    * |grad sdf| uses one-sided differences at the y domain edges
      (:func:`_ddy_nonperiodic`), so no material top ghost;
    * geometry-safe on textured sdf's (it uses the local normal).

    ``a`` is a discretization width tied to the solid smoothing, not a calibration
    parameter of the contact angle.
    """
    import math

    a = float(p.wall_delta_width if width is None else width)
    # Plain Python check: jnp.isfinite() of a Python float becomes a tracer under jit.
    if not math.isfinite(a) or a <= 0.0:
        raise ValueError("wall_delta_width must be finite and positive")
    sdf = jnp.asarray(sdf)
    dx, dy = float(p.dx), float(p.dy)
    gx = (jnp.roll(sdf, -1, axis=0) - jnp.roll(sdf, 1, axis=0)) / (2.0 * dx)
    gy = _ddy_nonperiodic(sdf, dy)
    grad = jnp.sqrt(gx**2 + gy**2 + 1e-30)
    scaled = jnp.clip(sdf / a, -60.0, 60.0)
    return (1.0 / (2.0 * a)) * (1.0 - jnp.tanh(scaled) ** 2) * grad


def wet_band(solid: Solid, p: PhaseFieldParams):
    """LEGACY ONLY (``wetting_model='legacy_affinity'``) fluid-side wetting envelope.

    The previous ``0.5 * (1 - tanh(sdf / width))`` tends to one throughout
    the solid volume, so the wall free-energy acted as a bulk source inside
    the obstacle.  Wetting is a boundary effect: keep it on the fluid side
    and decay it away from the wall.
    """
    width = max(float(p.wet_band), float(p.dx))
    d = jnp.maximum(solid.sdf, 0.0)
    band = jnp.exp(-((d / width) ** 2))
    return jnp.where(solid.sdf >= 0.0, band, 0.0).astype(p.dtype)


def wetting_mu(phi, solid: Solid, p: PhaseFieldParams):
    """Wall contribution to the chemical potential, ``mu_wall = dF_wall/dphi``.

    ``surface_energy`` (default, contract v6): the wall free energy

        F_wall = int g_w(phi, theta) delta_wall(sdf) dV,
        g_w(phi, theta) = -sigma_0 cos(theta) h(phi),  h(phi) = phi^2 (3 - 2 phi)

    gives the variationally consistent chemical potential

        mu_wall = dg_w/dphi * delta_wall = -sigma_0 cos(theta) h'(phi) delta_wall

    with h'(phi) = 6 phi (1 - phi).  It enters ``mu`` and therefore the conservative
    Cahn-Hilliard flux ``M grad(mu)``: liquid mass is unchanged, the wall free energy
    is the only thing that moves, and ``theta`` is the Young equilibrium contact
    angle (no amplitude parameter, no calibration factor).

    ``legacy_affinity``: the contract-v5 volumetric band affinity, for reproducing
    old trajectories only.  ``none``: no wall term (ablation).  Unknown names fail
    closed here as well as in ``PhaseFieldParams.__post_init__``.
    """
    if p.wetting_model == "surface_energy":
        return -WALL_SIGMA0 * solid.cos_theta * wall_switch_derivative(phi) * wall_delta(solid.sdf, p)
    if p.wetting_model == "legacy_affinity":
        phi_w = phi_wet_of(solid.cos_theta)
        band = wet_band(solid, p)
        return -p.wall_energy_amp * band * (phi_w - phi)
    if p.wetting_model == "none":
        return jnp.zeros_like(phi)
    raise ValueError(f"unknown wetting_model {p.wetting_model!r}; expected one of {sorted(WETTING_MODELS)}")


def chemical_potential(phi, solid: Solid, p: PhaseFieldParams):
    """mu = f'(phi)/eps - eps lap(phi) + conservative wetting affinity."""
    return fprime(phi) / p.eps - p.eps * _lap(phi, p.dx, p.dy) + wetting_mu(phi, solid, p)


def poisson_solve(rhs, m2):
    """Periodic solve of ``lap(x)=rhs`` with every discrete null mode removed."""
    rhs_hat = jnp.fft.rfft2(rhs)
    scale = jnp.maximum(jnp.max(m2), jnp.asarray(1.0, dtype=m2.dtype))
    tol = 64.0 * jnp.finfo(m2.dtype).eps * scale
    null = m2 <= tol
    denom = jnp.where(null, 1.0, m2)
    x_hat = jnp.where(null, 0.0, -rhs_hat / denom)
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

    # Keep the spectral CH mobility spatially constant.  The old code multiplied
    # ``mu_expl`` by a cell-centred mobility and then took a Laplacian, which
    # computes Δ(mob*mu), not the conservative operator ∇·(mob∇mu).
    # Solid impermeability is enforced after the semi-implicit CH update by the
    # bounded, mass-conserving geometric projection below.
    mu_expl = fprime(phi) / p.eps + wetting_mu(phi, solid, p)

    # --- phase field: conservative advection, biharmonic diffusion implicit ---
    phi_rhs = -div_upwind(u, v, phi, dx, dy)

    # --- momentum ---
    rho = rho_of(phi, p)
    nu = nu_of(phi, p)
    adv_u = div_upwind(u, v, u, dx, dy)
    adv_v = div_upwind(u, v, v, dx, dy)
    lap_u, lap_v = _lap(u, dx, dy), _lap(v, dx, dy)

    # Capillary (Korteweg/CSF) acceleration  F = +(SIGMA_NORM/We) mu grad(phi) / rho_l.
    # Sign (module docstring, "Capillary sign convention"): phi = 1 in the liquid, so
    # grad(phi) points gas -> liquid and mu > 0 at a convex liquid interface; F is then
    # directed toward the liquid (F . n_out < 0).  The free energy fixes the sign:
    # advection changes it by -int mu u.grad(phi), so +mu grad(phi) (== -phi grad(mu) up to
    # a gradient absorbed by the pressure) never creates free energy.  Contract v4 had
    # the opposite sign.  This is the only place the force is evaluated: ``pressure_field``
    # calls ``rhs``.  The denominator rho_l (not rho(phi)) is the open blocker P-CAP-RHO.
    phi_x, phi_y = _ddx(phi, dx), _ddy(phi, dy)
    cap_x = (SIGMA_NORM / p.We) * mu * phi_x / p.rho_l
    cap_y = (SIGMA_NORM / p.We) * mu * phi_y / p.rho_l

    if p.use_gravity:
        g_y = -(1.0 / p.Fr**2) * (rho - jnp.mean(rho)) / rho
    else:
        g_y = jnp.zeros_like(phi)

    u_rhs = -adv_u + nu * lap_u + cap_x
    v_rhs = -adv_v + nu * lap_v + cap_y + g_y

    return phi_rhs, u_rhs, v_rhs, mu, mu_expl


def _bounded_mass_project_2d(phi, active, target_mass, weight):
    """Project ``phi`` to [0,1] on ``active`` cells at fixed total mass.

    A scalar Lagrange multiplier is found by bisection for

        sum(active * clip(phi + lambda * weight, 0, 1)) = target_mass.

    This avoids the old hard solid clip, which deleted liquid every time the
    diffuse interface touched the wall.
    """
    active = active.astype(phi.dtype)
    base = jnp.clip(phi, 0.0, 1.0) * active
    capacity = jnp.sum(active)
    target = jnp.clip(target_mass, 0.0, capacity)
    weight = jnp.maximum(weight, 1.0e-4) * active

    lo = jnp.asarray(-1.0e4, dtype=phi.dtype)
    hi = jnp.asarray(1.0e4, dtype=phi.dtype)

    def body(_i, bounds):
        lo_, hi_ = bounds
        mid = 0.5 * (lo_ + hi_)
        candidate = jnp.clip(base + mid * weight, 0.0, 1.0) * active
        mass = jnp.sum(candidate)
        lo_ = jnp.where(mass < target, mid, lo_)
        hi_ = jnp.where(mass < target, hi_, mid)
        return lo_, hi_

    lo, hi = lax.fori_loop(0, 40, body, (lo, hi))
    lam = 0.5 * (lo + hi)
    return jnp.clip(base + lam * weight, 0.0, 1.0) * active


def _project_phase_outside_solid(phi, solid: Solid, p: PhaseFieldParams):
    """Remove phase from the geometric solid without changing total mass."""
    active = 1.0 - solid.chi_hard
    base = jnp.clip(phi, 0.0, 1.0)
    wall_scale = max(2.0 * float(p.eps), float(p.dx))
    d = jnp.maximum(solid.sdf, 0.0)
    wall_weight = jnp.exp(-((d / wall_scale) ** 2))
    interface_weight = 4.0 * base * (1.0 - base)
    weight = active * (wall_weight + 0.25 * interface_weight + 1.0e-3)
    return _bounded_mass_project_2d(phi, active, jnp.sum(phi), weight)


def step(state: State, solid: Solid, p: PhaseFieldParams) -> State:
    """One step made of three semi-implicit Euler substeps.

    The 4th-order Cahn-Hilliard diffusion is integrated implicitly in Fourier
    space inside every stage, which removes the O(eps^4/M) stability restriction;
    the Brinkman penalization of the solid is implicit as well.
    """
    # Three Euler substeps must sum to one requested step.  Previously each
    # substep used p.dt and advanced time by 3*p.dt, while files and plots
    # reported p.dt.  We do not claim SSP-RK3 accuracy for this integrator.
    dt = p.dt / 3.0
    m2 = p.m2
    denom = 1.0 + dt * p.M * p.eps * m2**2

    def substep(carry, _):
        phi, u, v, t = carry
        phi_rhs, u_rhs, v_rhs, mu, mu_expl = rhs(State(phi, u, v, t), solid, p)

        # implicit CH diffusion: (I + dt M eps lap^2) phi+ = phi + dt * explicit
        # lap(mu_expl) == -k2 * rfft2(mu_expl)
        source_hat = jnp.fft.rfft2(phi_rhs) - p.M * m2 * jnp.fft.rfft2(mu_expl)
        phi_hat = (jnp.fft.rfft2(phi) + dt * source_hat) / denom
        phi_new = jnp.fft.irfft2(phi_hat, s=phi.shape)
        if p.enforce_solid_phi:
            phi_new = _project_phase_outside_solid(phi_new, solid, p)

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
    velocity_mode: str = "uniform",
) -> State:
    """Circular droplet of radius R centred at (x0, y0) moving downwards at u_impact."""
    X, Y = grids(p)
    r = jnp.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
    phi = 0.5 * (1.0 - jnp.tanh((r - R) / (jnp.sqrt(2.0) * p.eps)))
    if velocity_mode == "uniform":
        # A uniform translation is exactly divergence-free on the periodic
        # grid, so the projection preserves the requested impact speed.  The
        # wall's Brinkman term then supplies the relative no-slip condition.
        u = jnp.zeros_like(phi)
        v = -u_impact * jnp.ones_like(phi)
    elif velocity_mode == "streamfunction":
        # Optional localized alternative.  A local downward-only velocity is
        # compressible; this streamfunction construction makes it divergence
        # free, with a weak return flow outside the drop.
        sx = X - x0
        sx = (sx + 0.5 * p.Lx) % p.Lx - 0.5 * p.Lx
        sy = Y - y0
        sy = (sy + 0.5 * p.Ly) % p.Ly - 0.5 * p.Ly
        radial = jnp.sqrt(sx * sx + sy * sy + 1e-12)
        envelope = 0.5 * (1.0 - jnp.tanh((radial - (R + 0.35)) / max(2.0 * p.eps, 0.08)))
        psi = u_impact * sx * envelope
        u = _ddy(psi, p.dy)
        v = -_ddx(psi, p.dx)
        divergence = _ddx(u, p.dx) + _ddy(v, p.dy)
        pr = poisson_solve(divergence, p.m2_proj)
        u = u - _ddx(pr, p.dx)
        v = v - _ddy(pr, p.dy)
    else:
        raise ValueError(f"unknown velocity_mode={velocity_mode!r}")
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
        save = ((i + 1) % save_every) == 0
        buf_phi = jnp.where(save, buf_phi.at[(i + 1) // save_every - 1].set(s.phi), buf_phi)
        buf_u = jnp.where(save, buf_u.at[(i + 1) // save_every - 1].set(s.u), buf_u)
        buf_v = jnp.where(save, buf_v.at[(i + 1) // save_every - 1].set(s.v), buf_v)
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
    """Liquid area (2-D 'mass') in the geometric fluid region."""
    fluid = (solid.sdf >= 0.0).astype(phi.dtype)
    return jnp.sum(phi * fluid) * p.dx * p.dy


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


#######################################################################################
#                       CONTACT-ANGLE MEASUREMENT (L1A-2b)                            #
#######################################################################################
#
# Contract v5 measured the apparent angle from the liquid area and the *full*
# horizontal width of the thresholded field, inverted through the circular-cap
# relation A = R^2 (theta - sin theta cos theta), w = 2 R sin theta.  That
# inversion is only consistent for a cap that is at most half a circle: once
# theta > 90 deg the widest point of the drop is its equator (w = 2R), not the
# contact line, so the same formula is fed an inconsistent (A, w) pair.  On
# synthetic circular caps at N=128 the legacy measurement shows MAE = 11.86 deg
# and a maximum error of 31.7 deg (a true 150 deg cap is read as 118 deg).
#
# ``measure_contact_angle`` (contract v6) instead extracts the level-set contour
# of phi and least-squares fits a circle to it, which is exact for the
# equilibrium shape of a 2-D drop (constant mean curvature).  The same synthetic
# caps are recovered with MAE = 0.007 deg / max error 0.016 deg; the audit lives
# in ``production/wetting_audit.py`` and is run by the test suite.  The angle is
# extracted from the fitted centre and radius through the cap relation
# ``y_c = y_w - Rc cos(theta)`` (the circle centre sits on the wall plane for
# theta = 90 deg).


def _periodic_centroid_x(phi, p: PhaseFieldParams, x0: float | None = None) -> float:
    """Phase-weighted periodic x centroid (used to unwrap the contour around the drop)."""
    import numpy as np

    phase = np.asarray(phi, dtype=np.float64)
    x_axis = (np.arange(p.Nx) + 0.5) * p.dx
    weights = phase.sum(axis=1)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        return float(x0 if x0 is not None else 0.5 * p.Lx)
    angles = 2.0 * np.pi * x_axis / p.Lx
    cos_mean = float((weights * np.cos(angles)).sum() / total)
    sin_mean = float((weights * np.sin(angles)).sum() / total)
    if abs(cos_mean) < 1e-12 and abs(sin_mean) < 1e-12:
        return float(x0 if x0 is not None else 0.5 * p.Lx)
    mean_angle = np.arctan2(sin_mean, cos_mean) % (2.0 * np.pi)
    return float((mean_angle / (2.0 * np.pi)) * p.Lx)


def contact_angle_contour_points(
    phi,
    sdf,
    dx: float,
    dy: float,
    level: float = 0.5,
    cutoff: float = 0.0,
    x0: float | None = None,
    period: float | None = None,
):
    """Level-set crossings of ``phi == level`` on grid edges, in physical coordinates.

    The x direction is periodic (edges wrap), the y direction is not (the wall is
    a slab, so an edge that wraps through the y seam would be a fake crossing).
    Coordinates are cell centres, ``(i + t + 0.5) * dx``; the ``+0.5`` is not
    cosmetic -- omitting it shifts the fitted circle by half a cell and the
    measured angle by O(dx / R).  Points closer to the solid than ``cutoff`` are
    dropped so the diffuse contact region cannot bias the fit.  The x coordinates
    are unwrapped about the periodic centroid of the drop and returned in a
    window centred on it.
    """
    import numpy as np

    phase = np.asarray(phi, dtype=np.float64)
    distance = np.asarray(sdf, dtype=np.float64)
    if phase.ndim != 2 or phase.shape != distance.shape:
        raise ValueError("phi and sdf must be matching two-dimensional fields")
    nx, ny = phase.shape
    if nx < 2 or ny < 2:
        raise ValueError("contact-angle measurement needs at least a 2x2 grid")
    if not all(np.isfinite(v) for v in (dx, dy, level, cutoff)) or dx <= 0 or dy <= 0:
        raise ValueError("dx/dy must be finite and positive")

    measured_x = (phase - level) * (np.roll(phase, -1, axis=0) - level) < 0.0
    denom_x = np.roll(phase, -1, axis=0) - phase
    t_x = np.where(np.abs(denom_x) > 1e-300, (level - phase) / np.where(denom_x == 0.0, 1.0, denom_x), 0.0)
    sdf_x = distance + t_x * (np.roll(distance, -1, axis=0) - distance)
    i_grid = np.broadcast_to(np.arange(nx)[:, None], phase.shape).astype(np.float64)
    j_grid = np.broadcast_to(np.arange(ny)[None, :], phase.shape).astype(np.float64)
    points_x = np.stack(
        [
            (i_grid + t_x + 0.5) * dx,
            (j_grid + 0.5) * dy,
            sdf_x,
        ],
        axis=-1,
    )[measured_x]

    measured_y = (phase[:, :-1] - level) * (phase[:, 1:] - level) < 0.0
    denom_y = phase[:, 1:] - phase[:, :-1]
    t_y = np.where(np.abs(denom_y) > 1e-300, (level - phase[:, :-1]) / np.where(denom_y == 0.0, 1.0, denom_y), 0.0)
    sdf_y = distance[:, :-1] + t_y * (distance[:, 1:] - distance[:, :-1])
    i_grid_y = np.broadcast_to(np.arange(nx)[:, None], t_y.shape).astype(np.float64)
    j_grid_y = np.broadcast_to(np.arange(ny - 1)[None, :], t_y.shape).astype(np.float64)
    points_y = np.stack(
        [
            (i_grid_y + 0.5) * dx,
            (j_grid_y + t_y + 0.5) * dy,
            sdf_y,
        ],
        axis=-1,
    )[measured_y]

    if points_x.size and points_y.size:
        points = np.concatenate([points_x, points_y], axis=0)
    else:
        points = points_x if points_x.size else points_y
    points = points[points[:, 2] >= float(cutoff)]
    if points.size == 0:
        return np.zeros((0, 3))
    length = float(period) if period is not None else nx * dx
    reference = 0.5 * length if x0 is None else float(x0)
    points[:, 0] = ((points[:, 0] - reference + 0.5 * length) % length) - 0.5 * length
    return points


def fit_circle(x, y):
    """Algebraic (Kasa) least-squares circle fit; returns ``(x_c, y_c, R)``."""
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3:
        raise ValueError("a circle fit needs at least three points")
    design = np.stack([x, y, np.ones_like(x)], axis=1)
    rhs = x**2 + y**2
    solution, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    x_c = 0.5 * solution[0]
    y_c = 0.5 * solution[1]
    radius_squared = solution[2] + x_c**2 + y_c**2
    if not np.isfinite(radius_squared) or radius_squared <= 0.0:
        raise ValueError("degenerate circle fit")
    return float(x_c), float(y_c), float(np.sqrt(radius_squared))


def wall_plane_height(solid: Solid, p: PhaseFieldParams, x0: float | None = None) -> float:
    """Height of the solid surface beneath ``x0`` (sdf = 0 crossing of that column).

    Valid for locally planar walls (the sessile benchmark); a strongly textured
    surface has no single contact plane and the apparent-angle concept itself
    changes meaning.
    """
    import numpy as np

    distance = np.asarray(solid.sdf, dtype=np.float64)
    if x0 is None:
        x0 = 0.5 * p.Lx
    index = int(np.clip(round((float(x0) - 0.5 * p.dx) / p.dx), 0, p.Nx - 1))
    column = distance[index, :]
    y_axis = (np.arange(p.Ny) + 0.5) * p.dy
    crossings = []
    for j in range(p.Ny - 1):
        a, b = column[j], column[j + 1]
        if a == 0.0:
            crossings.append(y_axis[j])
        elif (a < 0.0) != (b < 0.0) and np.isfinite(a) and np.isfinite(b):
            t = -a / (b - a)
            crossings.append(float(y_axis[j] + t * (y_axis[j + 1] - y_axis[j])))
    if not crossings:
        return float("nan")
    return float(np.max(crossings))


def measure_contact_angle(
    phi,
    solid: Solid,
    p: PhaseFieldParams,
    level: float = 0.5,
    cutoff_factor: float = 1.0,
) -> float:
    """Apparent contact angle (deg) from a circle fitted to the interface contour.

    ``theta = acos((y_w - y_c) / Rc)`` with the fitted centre ``(x_c, y_c)`` and
    radius ``Rc`` and the wall plane ``y_w``.  Contour points closer to the solid
    than ``max(cutoff_factor * eps, dx)`` are excluded; the defaults are the ones
    validated on synthetic circular caps (N = 128: MAE 0.007 deg, max 0.016 deg).
    Returns ``nan`` when the contour is too short or degenerate to fit.
    """
    import numpy as np

    cutoff = max(float(cutoff_factor) * float(p.eps), float(p.dx))
    reference = _periodic_centroid_x(np.asarray(phi), p)
    try:
        points = contact_angle_contour_points(
            phi, solid.sdf, p.dx, p.dy, level=level, cutoff=cutoff, x0=reference, period=p.Lx
        )
    except (ValueError, FloatingPointError):
        return float("nan")
    if len(points) < 8:
        return float("nan")
    try:
        _x_c, y_c, radius = fit_circle(points[:, 0], points[:, 1])
    except (ValueError, np.linalg.LinAlgError):
        return float("nan")
    wall = wall_plane_height(solid, p, x0=reference)
    if not np.isfinite(wall) or radius <= 0.0:
        return float("nan")
    cosine = (wall - y_c) / radius
    if not np.isfinite(cosine):
        return float("nan")
    return float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))))


def measure_contact_angle_area_width(phi, solid: Solid, p: PhaseFieldParams, level: float = 0.5):
    """Legacy (contract <= v5) area + full-width measurement; retained for comparison only.

    Documented as biased for ``theta > 90`` deg (see the section comment above);
    never use it to accept or reject a wetting model.
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


def sessile_initial_state(
    p: PhaseFieldParams,
    solid: Solid,
    R: float = 1.1,
    wall_height: float | None = None,
    theta0_deg: float = 90.0,
    x0: float | None = None,
) -> State:
    """Clean, target-independent sessile initial state (L1A-2b).

    A diffuse cap of geometric angle ``theta0_deg`` whose circle centre lies at
    ``y_c = y_w - R cos(theta0)``, masked to the geometric fluid region so that
    the solid holds no liquid at all (``theta0_deg = 90`` puts the centre on the
    wall plane).  The same state is used for every target angle: no target knows
    its own angle at t = 0.  ``u = v = 0``.  See ``production/README.md`` for why
    this replaces the legacy overlapping seed.
    """
    import numpy as np

    if wall_height is None:
        wall_height = wall_plane_height(solid, p, x0=x0)
    if not np.isfinite(wall_height):
        raise ValueError("cannot determine the wall plane for the sessile initial state")
    theta0 = np.deg2rad(float(theta0_deg))
    X, Y = grids(p)
    centre_x = 0.5 * p.Lx if x0 is None else float(x0)
    y_c = float(wall_height) - float(R) * np.cos(theta0)
    r = jnp.sqrt((X - centre_x) ** 2 + (Y - y_c) ** 2)
    phi = 0.5 * (1.0 - jnp.tanh((r - float(R)) / (jnp.sqrt(2.0) * p.eps)))
    phi = jnp.where(solid.sdf >= 0.0, phi, 0.0).astype(p.dtype)
    zero = jnp.zeros_like(phi)
    return State(phi=phi, u=zero, v=zero, t=0.0)


def pressure_field(state: State, solid: Solid, p: PhaseFieldParams):
    """Projection pressure of the current step (used for diagnostics, e.g. Laplace law).

    This is the pressure of the first substep of ``step``: momentum ``du/dt = rhs - grad(P)``
    with the same constant-coefficient projection ``lap(P) = div(u*)/dt``.  The forces come
    from ``rhs`` alone, so the capillary sign convention is inherited and must never be
    re-implemented here.  For a convex liquid drop (phi = 1 liquid) P_liquid > P_gas.
    """
    _, u_rhs, v_rhs, _, _ = rhs(state, solid, p)
    dt = p.dt / 3.0
    damp = 1.0 / (1.0 + dt * solid.chi / p.eta_pen)
    u_new = (state.u + dt * u_rhs) * damp
    v_new = (state.v + dt * v_rhs) * damp
    div = _ddx(u_new, p.dx) + _ddy(v_new, p.dy)
    return poisson_solve(div / dt, p.m2_proj)


def empty_solid(p: PhaseFieldParams) -> Solid:
    """A solid-free domain (useful for validation cases)."""
    z = jnp.zeros((p.Nx, p.Ny), dtype=p.dtype)
    return Solid(chi=z, ds=z, cos_theta=z, sdf=jnp.ones_like(z), chi_hard=jnp.zeros_like(z))


def build_case(case: dict, N: int = 192, dt: float | None = 4e-3):
    """Materialise (params, solid, initial_state) from a case dict (see cases.py).

    ``dt`` is clipped to the CFL-stable value for the chosen ``N``.  A case may
    set ``eps_factor`` (interface thickness in grid spacings), ``impact_gap``,
    ``Re``, ``wall_energy_amp`` and the solid-mask options.  The generated drop
    is placed above the local support under its footprint, so a remote obstacle
    cannot change its initial height.
    """
    if dt is None:
        dt = 2e-3
    p0 = PhaseFieldParams(Nx=N, Ny=N, Lx=6.0, Ly=6.0, dt=dt)
    dt = min(float(dt), float(stable_dt(p0, u_max=2.0)))
    p = PhaseFieldParams(
        Nx=N,
        Ny=N,
        Lx=6.0,
        Ly=6.0,
        Re=case.get("Re", 200.0),
        We=case.get("We", 100.0),
        dt=dt,
        eps=(float(case["eps_factor"]) * 6.0 / N) if "eps_factor" in case else case.get("eps"),
        # Generated trajectories use the mass-conserving solid projection.
        enforce_solid_phi=bool(case.get("enforce_solid_phi", True)),
        wall_energy_amp=float(case.get("wall_energy_amp", 5.0)),
        wet_band=float(case.get("wet_band", 0.15)),
    )
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
    # A geometric non-overlap is not enough for a diffuse interface.  Require
    # clearance in units of eps so the initial tanh tail is not already inside
    # the wall.  Explicit ``impact_gap`` may enlarge, but never shrink, this.
    x0 = float(case.get("x0", 3.0))
    local_surface_top = _local_surface_top(
        sdf,
        p,
        x0=x0,
        radius=float(R),
        margin=case.get("surface_margin"),
    )
    surface_top = float(local_surface_top)
    min_gap = max(float(case.get("impact_gap_eps", 2.0)) * float(p.eps), 0.05)
    requested_gap = float(case.get("impact_gap", min_gap))
    gap = max(requested_gap, min_gap)
    y0_default = surface_top + float(R) + gap
    y0 = float(case.get("y0", y0_default))
    clearance = y0 - float(R) - surface_top
    if clearance < min_gap - 1.0e-12:
        raise ValueError(
            f"initial diffuse interface is too close to the solid: clearance={clearance:.6g}, "
            f"required>={min_gap:.6g} ({case.get('impact_gap_eps', 2.0):g} eps)"
        )
    state = droplet_initial_state(
        p,
        x0=x0,
        y0=y0,
        R=R,
        u_impact=case.get("u_impact", 0.5),
        velocity_mode=case.get("velocity_mode", "uniform"),
    )
    return p, solid, state


def downsample(field, f):
    """Average-pool by factor f on both axes."""
    Nx, Ny = field.shape
    return field.reshape(Nx // f, f, Ny // f, f).mean(axis=(1, 3))
