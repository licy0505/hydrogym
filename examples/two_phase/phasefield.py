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

Wall wetting and conservative phase boundary (SOLVER_CONTRACT_VERSION >= 7)
-----------------------------------------------------------------------------
The bulk free energy is F_bulk = int [ f(phi)/eps + eps/2 |grad phi|^2 ] dV,
with f(phi) = phi^2 (1-phi)^2 and sigma_0 = sqrt(2)/6. The Young wall energy is

    F_wall = int g_w(phi, theta) dA
    g_w    = -sigma_0 cos(theta) h(phi),  h(phi) = phi^2 (3 - 2 phi)

so gamma_SG - gamma_SL = g_w(0) - g_w(1) = sigma_0 cos(theta_e). The production
``wetting_model='surface_energy'`` imposes this energy through the single natural
condition

    eps * d(phi)/dn + g_w'(phi) = 0,      d(mu)/dn = 0,

where n points from fluid into solid. The first condition is assembled into the
SDF-based embedded-boundary Laplacian (the Robin normal derivative is supplied
as its wall-face flux); it is not also added as a second ``mu_wall`` source.
Since contract v8 the surface measure multiplying that flux is the exact
geometric cut-cell wall area ``A_wall,i / (dx dy)`` (see the L1A-2e section
below); ``wall_delta`` remains only for the pinned legacy reproduction modes and
for diagnostics. The normal comes from -grad(sdf). For a bottom wall, positive y
points into fluid, therefore the y-slope of a compatible linear profile is
``+g_w'/eps``; the outward-normal derivative is ``-g_w'/eps``. This sign follows
the stated normal convention.
At 90 degrees g_w and the prescribed normal derivative are exactly zero.

The phase equation is a face-flux finite-volume update. Fluid-solid faces have
exactly zero advective and Cahn-Hilliard flux, x remains periodic, and the y seam
is blocked by the same hard aperture when it connects fluid to solid. The stiff
fourth-order term uses a deterministic matrix-free CG solve of
``(I + dt M eps L^T L) phi_new = rhs`` with L the masked face Laplacian. Its
relative residual is reported and a failed solve poisons the state with NaNs so
that callers fail closed. The v7 production step does not call the old
mass-redistribution projection.

``phase_boundary_model='projection_legacy'`` retains the contract-v6 FFT plus
post-step redistribution only for reproduction. ``wetting_model`` accepts
``surface_energy`` (v7 natural BC), ``surface_energy_volume_v6`` (v6 diffuse
chemical-potential form, diagnostic/reproduction only), ``legacy_affinity``
(contract-v5 reproduction only), and ``none`` (ablation). ``wall_energy_amp`` and
``wet_band`` are legacy-only; they are never calibration knobs.

The momentum and pressure projection remain periodic in both directions. The
fluid-cell face apertures specifically govern phase transport; this stage does
not change the Brinkman or momentum formulations.

Diagnostic CH-only mode (L1A-2d)
--------------------------------
``phase_only_step_with_diagnostics`` advances the phase equation with ``u = v = 0`` by calling the exact
production :func:`phase_transport_step` (same face apertures, implicit CG solve and natural Young BC) and
never touches momentum. It is an experiment mode for equilibration audits; it changes no default.

Embedded Young wall measure (SOLVER_CONTRACT_VERSION >= 8, L1A-2e)
------------------------------------------------------------------
Contract v7 applied the Young natural condition through the *diffuse* SDF kernel ``wall_delta``, i.e. the
wall surface measure was ``delta_wall(sdf) dV`` spread symmetrically about ``sdf = 0``. Only the ``sdf >= 0``
half of that kernel acts on the transported phase field, and that fluid-side share ``f`` depends on where the
wall falls inside a cell: at N = 128 it is 0.614, at N = 96/192 it is 0.500, at N = 64 it is 0.386 (L1A-2d,
``production/contact_line_kinetics.fluid_wall_delta_integral``). The effective wall forcing was therefore
``f`` times the Young value, and the resulting equilibria followed ``acos(f cos(theta))`` rather than
``theta``. Contract v8 replaces that kernel in the production path with an exact geometric measure:

    F_wall^h = sum_i A_wall,i * g_w(phi_i, theta_i)

where ``A_wall,i`` is the length of the ``sdf = 0`` contour assigned to the fluid-side control cell ``i``.
The contour is built deterministically by marching squares on corner-sampled SDF values
(:func:`wall_cut_segments`); each cut segment keeps its true length, centroid and unit normal
(``n = -grad sdf``, fluid -> solid) and is assigned to the nearest control cell whose centre lies in the hard
fluid (``sdf >= 0``), so the forcing always lands on a cell that participates in the open-face CH transport.
Consequences, all audited in ``production/wall_measure_audit.py``:

* ``sum_i A_wall,i`` equals the geometric wall length (exact for flat and inclined walls, chord-limited for
  curved ones) and is independent of sub-cell wall placement -- a flat wall translated by ``k/8 dy`` changes
  the total measure by round-off only, whereas the v7 fluid-side share ``f`` swings by tens of percent;
* the measure density ``A_wall,i / (dx dy)`` replaces ``wall_delta`` in exactly one place, the embedded
  boundary flux of the face Laplacian, so the production wall operator is the variational derivative of
  ``F_bulk + F_wall^h`` (float64 centred directional derivative, relative error <= 1e-6);
* ``g_w``, ``h``, ``sigma_0``, ``M``, ``dt``, ``eps``, the Brinkman/gas models and the CG tolerance are
  unchanged. There is no fitted factor, no ``cos(theta)/f``, no angle remap and no global wall gain: the
  measure is geometry only and is identical for every contact angle.

What contract v8 does *not* remove: ``g_w`` is evaluated at the control-cell value, so the condition is
imposed up to one cell away from the wall, and the phase-transport domain is still the cell-centre hard-fluid
mask ``sdf >= 0`` (contract v7), whose boundary jumps by a full cell as the wall crosses a cell centre. Both
are O(dx) sub-cell alignment effects of the *transport domain*, not of the measure; the L1A-2e translation
sweep measures them and ``production/README.md`` section I records which one dominates.

``wall_measure='diffuse_sdf_v7'`` retains the contract-v7 kernel for falsification/reproduction, and the
legacy modes (``surface_energy_volume_v6``, ``projection_legacy``, ``legacy_affinity``) keep their historical
measure pinned so their trajectories remain reproducible. The production default is ``sdf_cutcell_v1``
(``WALL_MEASURE_METHOD``), reported in dataset and validation metadata together with
``WALL_MEASURE_CONTRACT_VERSION``.

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
#   5: L1A-2a -- capillary sign corrected to +mu grad(phi).
#   6: L1A-2b -- Young-consistent diffuse wall surface energy became the default.
#   7: L1A-2c -- conservative phase transport on hard fluid-cell faces, zero
#      advective/CH cross-solid flux, matrix-free implicit CH solve, and one
#      natural Young wall condition. The v6 FFT/projection path is retained only
#      as ``phase_boundary_model='projection_legacy'``. Older trajectories are
#      stale (fingerprints include this version and phase_boundary_model).
#   8: L1A-2e -- the production Young wall condition is assembled with the exact
#      embedded cut-cell wall measure ``A_wall,i`` (marching squares on the SDF
#      zero contour, assigned to fluid-side control cells) instead of the
#      grid-alignment-dependent fluid share of the diffuse ``wall_delta`` kernel.
#      ``g_w``, ``h``, ``sigma_0`` and every transport default are unchanged; the
#      wall measure is geometry only (``WALL_MEASURE_METHOD='sdf_cutcell_v1'``,
#      ``WALL_MEASURE_CONTRACT_VERSION=1``). v7 trajectories are stale.
SOLVER_CONTRACT_VERSION = 8

#: Production embedded wall-measure construction (L1A-2e). ``sdf_cutcell_v1`` is the
#: deterministic marching-squares cut-cell measure; ``diffuse_sdf_v7`` is the pinned
#: contract-v7 kernel, retained for falsification/reproduction only. Neither option
#: may carry a fitted factor: the measure is geometry, never a function of theta.
WALL_MEASURE_METHOD = "sdf_cutcell_v1"
WALL_MEASURE_CONTRACT_VERSION = 1
WALL_MEASURE_METHODS = ("sdf_cutcell_v1", "diffuse_sdf_v7")


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
    """Immutable description of the solid surface (wall + micro-structure).

    ``wall_area``/``wall_normal_*``/``wall_distance`` are the contract-v8 embedded wall
    measure: the geometric length of the ``sdf = 0`` contour assigned to each fluid-side
    control cell, the area-weighted unit normal (fluid -> solid) of that cell's wall, and
    the area-weighted normal distance from the cell centre to that wall. They are pure
    geometry -- identical for every contact angle -- and are built once per solid by
    :func:`make_solid`. ``wall_area`` is what the production operator uses; the normal and the
    distance are the geometric diagnostics that record *where* the condition is imposed relative
    to the wall (see :func:`wall_plane_phi` and ``production/README.md`` section I).
    """

    chi: jnp.ndarray  # (Nx, Ny) indicator, 1 = solid
    ds: jnp.ndarray  # (Nx, Ny) surface delta |grad chi|
    cos_theta: jnp.ndarray  # (Nx, Ny) cos(contact angle) evaluated on the solid surface
    sdf: jnp.ndarray  # (Nx, Ny) signed distance to the solid, <0 inside solid
    chi_hard: jnp.ndarray  # (Nx, Ny) 0/1 mask of the solid interior (impermeable)
    wall_area: jnp.ndarray  # (Nx, Ny) cut-cell wall length assigned to this control cell
    wall_normal_x: jnp.ndarray  # (Nx, Ny) area-weighted wall normal, x component
    wall_normal_y: jnp.ndarray  # (Nx, Ny) area-weighted wall normal, y component
    wall_distance: jnp.ndarray  # (Nx, Ny) area-weighted normal distance centre -> wall, >= 0


class ImplicitSolveInfo(NamedTuple):
    """CG iteration count, relative residual, and convergence flag."""

    iterations: jnp.ndarray
    relative_residual: jnp.ndarray
    converged: jnp.ndarray


class StepDiagnostics(NamedTuple):
    """Per-internal-stage CH solver diagnostics returned by ``step_with_diagnostics``."""

    implicit_iterations: jnp.ndarray
    implicit_relative_residuals: jnp.ndarray
    implicit_converged: jnp.ndarray


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
    # Young-consistent wall energy. ``surface_energy`` (default) is the v7
    # natural boundary condition; ``surface_energy_volume_v6`` and
    # ``legacy_affinity`` are diagnostic/reproduction modes; ``none`` is an ablation.
    wetting_model: str = "surface_energy"
    phase_boundary_model: str = "impermeable_flux"
    # Width of the normalized SDF wall measure; fixed by solid smoothing and never
    # a contact-angle fit parameter. LEGACY/diagnostic since contract v8: the
    # production measure is the exact cut-cell wall area.
    wall_delta_width: float = None
    # Embedded wall-measure construction (contract v8). ``sdf_cutcell_v1`` is the
    # production geometry measure; ``diffuse_sdf_v7`` reproduces contract v7.
    wall_measure: str = WALL_MEASURE_METHOD
    # LEGACY ONLY: contract-v5 volumetric affinity amplitude and band width.
    wall_energy_amp: float = 5.0
    wet_band: float = 0.15
    # Matrix-free CH implicit solve. Production float32 default is <=1e-6; audits
    # may select 1e-8 or tighter in float64. Failure is fail-closed.
    ch_solver_rtol: float = 1.0e-6
    ch_solver_max_iterations: int = 200
    cfl: float = 0.4  # used by ``stable_dt``
    use_gravity: bool = False
    dtype: type = jnp.float32
    # LEGACY ONLY: mass-redistribution projection is valid only with
    # phase_boundary_model='projection_legacy'.
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
        if self.phase_boundary_model not in PHASE_BOUNDARY_MODELS:
            raise ValueError(
                f"unknown phase_boundary_model {self.phase_boundary_model!r}; "
                f"expected one of {sorted(PHASE_BOUNDARY_MODELS)}"
            )
        if self.wall_measure not in WALL_MEASURE_METHODS:
            raise ValueError(
                f"unknown wall_measure {self.wall_measure!r}; expected one of {sorted(WALL_MEASURE_METHODS)}"
            )
        if self.enforce_solid_phi and self.phase_boundary_model != "projection_legacy":
            raise ValueError(
                "enforce_solid_phi is legacy-only; use phase_boundary_model='projection_legacy' "
                "to enable post-step mass redistribution"
            )
        if not math.isfinite(float(self.ch_solver_rtol)) or float(self.ch_solver_rtol) <= 0.0:
            raise ValueError("ch_solver_rtol must be finite and positive")
        if (
            isinstance(self.ch_solver_max_iterations, bool)
            or not isinstance(self.ch_solver_max_iterations, int)
            or self.ch_solver_max_iterations < 1
        ):
            raise ValueError("ch_solver_max_iterations must be a positive integer")

    @property
    def dx(self) -> float:
        return self.Lx / self.Nx

    @property
    def dy(self) -> float:
        return self.Ly / self.Ny

    @property
    def m2(self):
        """Fourier symbol of the 5-point Laplacian used by the periodic stencils.

        Inverting *this* symbol (rather than |k|^2) in the pressure Poisson solve
        makes the projection consistent with its central-difference divergence/
        gradient. The same symbol is retained only by the contract-v6
        ``projection_legacy`` CH reproduction path; v7 uses the face-aperture
        matrix-free operator instead.
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


def sdf_corner_values(sdf: jnp.ndarray, p: PhaseFieldParams) -> jnp.ndarray:
    """Corner samples of a cell-centre signed distance field, shape ``(Nx, Ny+1)``.

    ``q[i, j]`` is the SDF at the grid corner ``(i*dx, j*dy)``. ``i`` is periodic
    (``q[Nx, j] == q[0, j]``, so the x seam carries a continuous contour), and
    ``j`` enumerates the ``Ny+1`` physical y-faces. Each corner is the average of
    the four surrounding cell centres: exact for a locally linear SDF (flat walls,
    inclined walls, box faces) and second-order accurate otherwise.

    The two y-boundary rows are *linearly extrapolated* from the first/last two
    cell rows, never periodically wrapped, so the y seam cannot create a ghost
    contour (the v7 ``wall_delta`` needed the same one-sided treatment).
    """
    c = jnp.asarray(sdf)
    if c.ndim != 2 or c.shape != (p.Nx, p.Ny):
        raise ValueError(f"sdf must have shape ({p.Nx}, {p.Ny}); got {c.shape}")
    bottom_ghost = 2.0 * c[:, :1] - c[:, 1:2]
    top_ghost = 2.0 * c[:, -1:] - c[:, -2:-1]
    extended = jnp.concatenate([bottom_ghost, c, top_ghost], axis=1)
    extended_left = jnp.roll(extended, 1, axis=0)
    return 0.25 * (extended_left[:, :-1] + extended[:, :-1] + extended_left[:, 1:] + extended[:, 1:])


def _shift_cells(array: jnp.ndarray, di: int, dj: int) -> jnp.ndarray:
    """Return ``out[i, j] = array[(i+di) % Nx, j+dj]`` with zeros outside the y range."""
    out = jnp.roll(array, -di, axis=0)
    if dj > 0:
        out = jnp.concatenate([out[:, dj:], jnp.zeros_like(out[:, :dj])], axis=1)
    elif dj < 0:
        out = jnp.concatenate([jnp.zeros_like(out[:, :-dj]), out[:, :dj]], axis=1)
    return out


#: Marching-squares edge pairing. Edge 0 = bottom (a-b), 1 = right (b-c),
#: 2 = top (d-c), 3 = left (a-d); corner bits are a=1, b=2, c=4, d=8 with
#: "fluid" = ``sdf >= 0``. ``-1`` marks an unused slot. Cases 5 and 10 are the
#: saddles: the table holds the ``centre >= 0`` variant and :func:`wall_cut_segments`
#: flips the pairing deterministically when the bilinear centre value is negative.
_MS_FIRST_A = (-1, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 1, 1, 0, 0, -1)
_MS_SECOND_A = (-1, 3, 1, 3, 2, 1, 2, 3, 3, 2, 3, 2, 3, 1, 3, -1)
_MS_FIRST_B = (-1, -1, -1, -1, -1, 2, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1)
_MS_SECOND_B = (-1, -1, -1, -1, -1, 3, -1, -1, -1, -1, 2, -1, -1, -1, -1, -1)


#: Deterministic control-cell search rings (offsets in cells), nearest first. Only
#: hard-fluid cell centres (``sdf >= 0``) are eligible, so the wall forcing always
#: lands on a cell that the open-face CH transport actually updates.
#: Offsets covered when aggregating segments onto their control cells. It must span the
#: widest search ring, otherwise a segment assigned further away would be dropped and the
#: total measure would no longer equal the contour length.
_CONTROL_CELL_GATHER_OFFSETS = (-2, -1, 0, 1, 2)

_CONTROL_CELL_RINGS = (
    ((0, 0),),
    ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)),
    (
        (2, 0),
        (-2, 0),
        (0, 2),
        (0, -2),
        (2, 1),
        (2, -1),
        (-2, 1),
        (-2, -1),
        (1, 2),
        (1, -2),
        (-1, 2),
        (-1, -2),
        (2, 2),
        (2, -2),
        (-2, 2),
        (-2, -2),
    ),
)


def wall_cut_segments(sdf: jnp.ndarray, p: PhaseFieldParams) -> dict:
    """Deterministic marching-squares segments of the ``sdf = 0`` contour.

    Two fixed slots per cell (only the saddle cases use both), so every array has
    the static shape ``(Nx, Ny, 2)`` and the whole construction is JAX-traceable
    with no data-dependent shapes. Per segment:

    * ``length`` -- the true cut length inside the cell (exact for a linear SDF);
    * ``mid_x``/``mid_y`` -- the segment centroid;
    * ``normal_x``/``normal_y`` -- the unit normal ``-grad(sdf)`` at the centroid
      (fluid -> solid, the same convention as :func:`fluid_outward_normal`), from
      the bilinear corner interpolation, so it follows curved and inclined walls;
    * ``target_i``/``target_j`` -- the fluid-side control cell the segment is
      assigned to: the cut cell when its centre is in the hard fluid, otherwise the
      hard-fluid cell whose centre is nearest the segment centroid, searched over
      deterministic rings of radius <= 2 (nearest ring first, ties broken by a static
      priority). The assignment is unique, so no contour length is counted twice and
      none is dropped.

    Because the control cell is always a hard-fluid cell (``sdf >= 0``), the wall
    forcing lands on a cell that participates in the open-face CH transport; a
    solid-centred cut cell is isolated by the face apertures and would silently
    swallow it.
    """
    tiny = 1.0e-30
    q = sdf_corner_values(sdf, p)
    nx_cells, ny_cells = int(p.Nx), int(p.Ny)
    dx, dy = float(p.dx), float(p.dy)
    left = jnp.roll(q, -1, axis=0)
    a, b, d, c = q[:, :-1], left[:, :-1], q[:, 1:], left[:, 1:]  # BL, BR, TL, TR

    def _crossing(v0, v1):
        """Linear zero-crossing parameter in [0, 1] along an edge from v0 to v1."""
        denominator = v0 - v1
        safe = jnp.where(jnp.abs(denominator) > tiny, denominator, 1.0)
        t = jnp.where(jnp.abs(denominator) > tiny, v0 / safe, 0.5)
        return jnp.clip(t, 0.0, 1.0)

    t0, t1, t2, t3 = _crossing(a, b), _crossing(b, c), _crossing(d, c), _crossing(a, d)
    x = jnp.broadcast_to((jnp.arange(nx_cells) * dx)[:, None], (nx_cells, ny_cells))
    y = jnp.broadcast_to((jnp.arange(ny_cells) * dy)[None, :], (nx_cells, ny_cells))
    point_x = jnp.stack([x + t0 * dx, x + dx, x + t2 * dx, x], axis=-1)
    point_y = jnp.stack([y, y + t1 * dy, y + dy, y + t3 * dy], axis=-1)

    sa, sb, sc, sd = (a >= 0.0), (b >= 0.0), (c >= 0.0), (d >= 0.0)
    index = sa.astype(jnp.int32) + 2 * sb.astype(jnp.int32) + 4 * sc.astype(jnp.int32) + 8 * sd.astype(jnp.int32)
    centre = 0.25 * (a + b + c + d)
    table = jnp.asarray
    first_a = table(_MS_FIRST_A, dtype=jnp.int32)[index]
    second_a = table(_MS_SECOND_A, dtype=jnp.int32)[index]
    first_b = table(_MS_FIRST_B, dtype=jnp.int32)[index]
    second_b = table(_MS_SECOND_B, dtype=jnp.int32)[index]
    saddle_negative = ((index == 5) | (index == 10)) & (centre < 0.0)
    second_a = jnp.where(saddle_negative & (index == 5), 3, jnp.where(saddle_negative & (index == 10), 1, second_a))
    first_b = jnp.where(saddle_negative & (index == 5), 1, jnp.where(saddle_negative & (index == 10), 2, first_b))
    second_b = jnp.where(saddle_negative & (index == 5), 2, jnp.where(saddle_negative & (index == 10), 3, second_b))

    def _endpoints(first, second):
        px = jnp.take_along_axis(point_x, first[..., None], axis=-1)[..., 0]
        py = jnp.take_along_axis(point_y, first[..., None], axis=-1)[..., 0]
        qx = jnp.take_along_axis(point_x, second[..., None], axis=-1)[..., 0]
        qy = jnp.take_along_axis(point_y, second[..., None], axis=-1)[..., 0]
        return px, py, qx, qy

    ax_, ay, bx, by = _endpoints(first_a, second_a)
    cx_, cy, ddx, ddy = _endpoints(first_b, second_b)
    mid_x = jnp.stack([0.5 * (ax_ + bx), 0.5 * (cx_ + ddx)], axis=-1)
    mid_y = jnp.stack([0.5 * (ay + by), 0.5 * (cy + ddy)], axis=-1)
    length = jnp.stack(
        [
            jnp.sqrt((bx - ax_) ** 2 + (by - ay) ** 2),
            jnp.sqrt((ddx - cx_) ** 2 + (ddy - cy) ** 2),
        ],
        axis=-1,
    )
    valid = jnp.stack([first_a >= 0, first_b >= 0], axis=-1)
    length = jnp.where(valid, length, 0.0)

    # Bilinear gradient of the SDF at each segment centroid -> unit normal.
    u = (mid_x - x[..., None]) / dx
    v = (mid_y - y[..., None]) / dy
    du = ((b - a)[..., None] * (1.0 - v) + (c - d)[..., None] * v) / dx
    dv = ((d - a)[..., None] * (1.0 - u) + (c - b)[..., None] * u) / dy
    gradient = jnp.sqrt(du * du + dv * dv + tiny)
    normal_x = jnp.where(valid, -du / gradient, 0.0)
    normal_y = jnp.where(valid, -dv / gradient, 0.0)
    fluid_x = du / gradient
    fluid_y = dv / gradient

    # Fluid-side control cell for each segment.
    cell_sdf = jnp.reshape(jnp.asarray(sdf), (-1,))
    i_index = jnp.broadcast_to(jnp.arange(nx_cells, dtype=jnp.int32)[:, None], (nx_cells, ny_cells))[..., None]
    j_index = jnp.broadcast_to(jnp.arange(ny_cells, dtype=jnp.int32)[None, :], (nx_cells, ny_cells))[..., None]

    def _probe(di, dj):
        target_i = (i_index + di.astype(jnp.int32)) % nx_cells
        target_j = j_index + dj.astype(jnp.int32)
        inside = (target_j >= 0) & (target_j < ny_cells)
        target_j = jnp.clip(target_j, 0, ny_cells - 1)
        value = jnp.take(cell_sdf, target_i * ny_cells + target_j)
        return (value >= 0.0) & inside, target_i, target_j

    # Deterministic ring search for the hard-fluid control cell nearest to each segment
    # centroid. Ring 0 keeps the cut cell itself whenever its centre is fluid (the
    # flat/inclined-wall case), so the assignment never moves further from the contour
    # than the geometry requires. Concave corners of textured SDFs can have every
    # immediate neighbour solid-centred; the wider rings keep their measure on a
    # transported cell instead of dropping it. Centroid offset relative to the cut-cell
    # centre, in cell units (|offset| <= 0.5 per component).
    cell_centre_x = (i_index + 0.5) * dx
    cell_centre_y = (j_index + 0.5) * dy
    centroid_offset_x = (mid_x - cell_centre_x) / dx
    centroid_offset_y = (mid_y - cell_centre_y) / dy
    target_i, target_j = _probe(jnp.round(fluid_x).astype(jnp.int32), jnp.round(fluid_y).astype(jnp.int32))[1:]
    assigned_fluid = jnp.zeros(valid.shape, dtype=jnp.bool_)
    for ring in _CONTROL_CELL_RINGS:
        keys, fluid_flags, candidates_i, candidates_j = [], [], [], []
        for priority, (di, dj) in enumerate(ring):
            is_fluid, candidate_i, candidate_j = _probe(
                jnp.full(valid.shape, di, dtype=jnp.int32), jnp.full(valid.shape, dj, dtype=jnp.int32)
            )
            # Rank fluid candidates by the distance from their centre to this segment's
            # centroid, in cell units: the wall condition belongs on the hard-fluid cell
            # that is geometrically nearest to the piece of wall it represents. Ranking by
            # normal alignment instead prefers sideways neighbours (a cell above a vertical
            # pillar face has zero normal offset) and ranking by ring order alone moves
            # measure along an inclined wall. Fluid eligibility is carried *separately* from
            # the rank: folding it into the key magnitude would let a large distance penalty
            # mask a valid fluid cell and push the whole measure one cell off the wall.
            distance_squared = (di - centroid_offset_x) ** 2 + (dj - centroid_offset_y) ** 2
            keys.append(
                is_fluid.astype(distance_squared.dtype) * 8.0 - distance_squared + (len(ring) - priority) * 1.0e-3
            )
            fluid_flags.append(is_fluid)
            candidates_i.append(candidate_i)
            candidates_j.append(candidate_j)
        stacked_keys = jnp.stack(keys, axis=-1)
        best = jnp.argmax(stacked_keys, axis=-1)[..., None]
        ring_fluid = jnp.take_along_axis(jnp.stack(fluid_flags, axis=-1), best, axis=-1)[..., 0]
        ring_i = jnp.take_along_axis(jnp.stack(candidates_i, axis=-1), best, axis=-1)[..., 0]
        ring_j = jnp.take_along_axis(jnp.stack(candidates_j, axis=-1), best, axis=-1)[..., 0]
        adopt = ring_fluid & (~assigned_fluid)
        target_i = jnp.where(adopt, ring_i, target_i)
        target_j = jnp.where(adopt, ring_j, target_j)
        assigned_fluid = assigned_fluid | adopt
    target_i = jnp.where(valid, target_i, 0)
    target_j = jnp.where(valid, target_j, 0)
    return {
        "length": length,
        "mid_x": mid_x,
        "mid_y": mid_y,
        "normal_x": normal_x,
        "normal_y": normal_y,
        "target_i": target_i,
        "target_j": target_j,
        "valid": valid,
        "assigned_to_fluid_cell": assigned_fluid & valid,
    }


def wall_cut_measure(sdf: jnp.ndarray, p: PhaseFieldParams) -> tuple:
    """Exact embedded wall measure: cut-contour length assigned to fluid cells.

    Returns ``(wall_area, wall_normal_x, wall_normal_y, wall_distance, info)`` where
    ``wall_area`` is the per-cell geometric wall length ``A_wall,i`` (units of length, so
    ``sum(A_wall,i) == wall length``, not 1), the normals are the area-weighted unit
    normals of the wall carried by each cell, and ``wall_distance`` is the area-weighted
    normal distance from the control-cell centre to that wall (``>= 0``, fluid side).
    The aggregation is a fixed 5x5 neighbourhood gather keyed on the segment target index
    -- no scatter with duplicate indices -- so it is deterministic in every backend.
    """
    segments = wall_cut_segments(sdf, p)
    length = segments["length"]
    target_i, target_j = segments["target_i"], segments["target_j"]
    nx_cells, ny_cells = int(p.Nx), int(p.Ny)
    i_index = jnp.broadcast_to(jnp.arange(nx_cells, dtype=jnp.int32)[:, None], (nx_cells, ny_cells))
    j_index = jnp.broadcast_to(jnp.arange(ny_cells, dtype=jnp.int32)[None, :], (nx_cells, ny_cells))
    dtype = length.dtype
    area = jnp.zeros((nx_cells, ny_cells), dtype=dtype)
    normal_x_sum = jnp.zeros_like(area)
    normal_y_sum = jnp.zeros_like(area)
    centroid_x_sum = jnp.zeros_like(area)
    centroid_y_sum = jnp.zeros_like(area)
    distance_sum = jnp.zeros_like(area)
    segment_count = jnp.zeros_like(area)
    centre_x = (jnp.arange(nx_cells, dtype=dtype) + 0.5)[:, None, None] * jnp.asarray(p.dx, dtype=dtype)
    centre_y = (jnp.arange(ny_cells, dtype=dtype) + 0.5)[None, :, None] * jnp.asarray(p.dy, dtype=dtype)
    for di in _CONTROL_CELL_GATHER_OFFSETS:
        for dj in _CONTROL_CELL_GATHER_OFFSETS:
            contribution = _shift_cells(length, di, dj)
            match = (_shift_cells(target_i, di, dj) == i_index[..., None]) & (
                _shift_cells(target_j, di, dj) == j_index[..., None]
            )
            contribution = jnp.where(match, contribution, 0.0)
            segment_normal_x = _shift_cells(segments["normal_x"], di, dj)
            segment_normal_y = _shift_cells(segments["normal_y"], di, dj)
            segment_mid_x = _shift_cells(segments["mid_x"], di, dj)
            segment_mid_y = _shift_cells(segments["mid_y"], di, dj)
            area += jnp.sum(contribution, axis=-1)
            normal_x_sum += jnp.sum(contribution * segment_normal_x, axis=-1)
            normal_y_sum += jnp.sum(contribution * segment_normal_y, axis=-1)
            centroid_x_sum += jnp.sum(contribution * segment_mid_x, axis=-1)
            centroid_y_sum += jnp.sum(contribution * segment_mid_y, axis=-1)
            # Signed normal offset from this control-cell centre to the segment, area weighted.
            distance_sum += jnp.sum(
                contribution
                * ((segment_mid_x - centre_x) * segment_normal_x + (segment_mid_y - centre_y) * segment_normal_y),
                axis=-1,
            )
            segment_count += jnp.sum(jnp.where(match & (contribution > 0.0), 1.0, 0.0), axis=-1)
    safe_area = jnp.maximum(area, jnp.asarray(1.0e-30, dtype=dtype))
    has_wall = area > 0.0
    normal_x = jnp.where(has_wall, normal_x_sum / safe_area, 0.0)
    normal_y = jnp.where(has_wall, normal_y_sum / safe_area, 0.0)
    distance = jnp.maximum(jnp.where(has_wall, distance_sum / safe_area, 0.0), 0.0)
    cell_sdf = jnp.asarray(sdf)
    fluid_cells = cell_sdf >= 0.0
    total_length = jnp.sum(jnp.where(segments["valid"], segments["length"], 0.0))
    info = {
        "total_contour_length": total_length,
        "total_assigned_length": jnp.sum(area),
        "measure_conservation_error": jnp.sum(area) - total_length,
        "n_wall_cells": jnp.sum(has_wall),
        "n_segments": jnp.sum(segments["valid"]),
        # Degenerate slots: the contour passes exactly through a grid corner, so the two
        # crossing points coincide. They carry no measure and are not assignable; counting
        # them separately keeps "every segment is assigned exactly once" an exact statement.
        "n_positive_length_segments": jnp.sum(segments["valid"] & (segments["length"] > 0.0)),
        "n_segments_multi_per_cell": jnp.sum(jnp.sum(segments["valid"].astype(dtype), axis=-1) > 1.0),
        "length_on_fluid_cells": jnp.sum(jnp.where(fluid_cells, area, 0.0)),
        "length_on_solid_cells": jnp.sum(jnp.where(fluid_cells, 0.0, area)),
        "segments_assigned_to_fluid_cell": jnp.sum(segments["assigned_to_fluid_cell"]),
        "max_cell_length": jnp.max(area),
        "min_positive_cell_length": jnp.where(
            jnp.any(has_wall),
            jnp.min(jnp.where(has_wall, area, jnp.maximum(jnp.max(area), 1.0))),
            jnp.asarray(0.0, dtype=dtype),
        ),
        "wall_centroid_x": jnp.sum(centroid_x_sum),
        "wall_centroid_y": jnp.sum(centroid_y_sum) / jnp.maximum(jnp.sum(area), 1.0e-30),
        "mean_segments_per_wall_cell": jnp.sum(segment_count) / jnp.maximum(jnp.sum(has_wall), 1.0),
        "segment_count_field": segment_count,
        "max_wall_distance_over_dx": jnp.max(distance) / jnp.asarray(p.dx, dtype=dtype),
        "mean_wall_distance_over_dx": jnp.sum(distance_sum)
        / jnp.maximum(jnp.sum(area), jnp.asarray(1.0e-30, dtype=dtype))
        / jnp.asarray(p.dx, dtype=dtype),
    }
    return area, normal_x, normal_y, distance, info


def make_solid(
    sdf: jnp.ndarray,
    params: PhaseFieldParams,
    cos_theta: float | jnp.ndarray = jnp.cos(jnp.deg2rad(90.0)),
) -> Solid:
    """Build a :class:`Solid` from a signed-distance field (negative inside solid).

    ``cos_theta`` may be a scalar (uniform wettability) or a full (Nx, Ny) array
    (spatially patterned wettability).

    The contract-v8 embedded wall measure (cut-contour length per fluid-side
    control cell and its area-weighted normal) is built here, once per geometry:
    it is pure geometry, so it is identical for every contact angle and never
    depends on ``phi``.
    """
    chi = smooth_indicator(sdf, params.dx).astype(params.dtype)
    ds = surface_delta(chi, params.dx, params.dy).astype(params.dtype)
    if jnp.ndim(cos_theta) == 0:
        cos_theta = jnp.full_like(chi, float(cos_theta))
    wall_area, wall_normal_x, wall_normal_y, wall_distance, _ = wall_cut_measure(sdf.astype(params.dtype), params)
    return Solid(
        chi=chi,
        ds=ds,
        cos_theta=cos_theta.astype(params.dtype),
        sdf=sdf.astype(params.dtype),
        # Exact geometric solid mask. The v7 phase-face apertures use this hard
        # geometry to close every fluid-solid phase-flux face.
        chi_hard=(sdf < 0.0).astype(params.dtype),
        wall_area=wall_area.astype(params.dtype),
        wall_normal_x=wall_normal_x.astype(params.dtype),
        wall_normal_y=wall_normal_y.astype(params.dtype),
        wall_distance=wall_distance.astype(params.dtype),
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


def fluid_face_apertures(solid: Solid, p: PhaseFieldParams):
    """Hard 0/1 apertures for +x and +y phase faces.

    Arrays are face-aligned with the left/lower cell: ``ax[i,j]`` connects
    ``(i,j)`` to ``(i+1,j)`` and ``ay[i,j]`` connects ``(i,j)`` to
    ``(i,j+1)``. Both axes are represented periodically, but an aperture is open
    only when both adjacent cell centres are in the hard fluid (``sdf >= 0``).
    In particular, the periodic-y seam is closed when it joins top fluid to the
    bottom solid slab.
    """
    del p  # geometry is already sampled on the parameter grid
    fluid = solid.sdf >= 0.0
    return fluid & jnp.roll(fluid, -1, axis=0), fluid & jnp.roll(fluid, -1, axis=1)


def divergence_from_face_fluxes(flux_x, flux_y, p: PhaseFieldParams):
    """Finite-volume divergence of +axis face fluxes (flux density, not divided)."""
    return (flux_x - jnp.roll(flux_x, 1, axis=0)) / p.dx + (flux_y - jnp.roll(flux_y, 1, axis=1)) / p.dy


def phase_advective_fluxes(u, v, phi, solid: Solid, p: PhaseFieldParams):
    """Conservative upwind ``u*phi`` face fluxes with exact solid-face blocking."""
    aperture_x, aperture_y = fluid_face_apertures(solid, p)
    u_face = 0.5 * (u + jnp.roll(u, -1, axis=0))
    v_face = 0.5 * (v + jnp.roll(v, -1, axis=1))
    phi_x = jnp.where(u_face >= 0.0, phi, jnp.roll(phi, -1, axis=0))
    phi_y = jnp.where(v_face >= 0.0, phi, jnp.roll(phi, -1, axis=1))
    return aperture_x * u_face * phi_x, aperture_y * v_face * phi_y


def chemical_potential_fluxes(mu, solid: Solid, p: PhaseFieldParams):
    """Conservative Cahn--Hilliard flux ``J_CH=-M grad(mu)`` on open fluid faces."""
    aperture_x, aperture_y = fluid_face_apertures(solid, p)
    grad_x = (jnp.roll(mu, -1, axis=0) - mu) / p.dx
    grad_y = (jnp.roll(mu, -1, axis=1) - mu) / p.dy
    return -p.M * aperture_x * grad_x, -p.M * aperture_y * grad_y


def fluid_laplacian(phi, solid: Solid, p: PhaseFieldParams):
    """Symmetric negative-semidefinite face Laplacian with zero solid-face flux."""
    aperture_x, aperture_y = fluid_face_apertures(solid, p)
    grad_x = aperture_x * (jnp.roll(phi, -1, axis=0) - phi) / p.dx
    grad_y = aperture_y * (jnp.roll(phi, -1, axis=1) - phi) / p.dy
    return divergence_from_face_fluxes(grad_x, grad_y, p)


#: Surface tension carried by the equilibrium tanh profile with
#: f(phi) = phi^2 (1-phi)^2: sigma_0 = sqrt(2)/6.  The Korteweg force is scaled by
#: SIGMA_NORM = 1/sigma_0 (see the module docstring), so ``sigma_0`` is the *actual*
#: non-dimensional surface tension and is the correct prefactor for wall energies.
#: Evaluated in pure Python: the constant must not depend on ``jax_enable_x64``.
WALL_SIGMA0 = math.sqrt(2.0) / 6.0

#: Wetting and phase-boundary semantics accepted by the v7 solver.
WETTING_MODELS = ("surface_energy", "surface_energy_volume_v6", "legacy_affinity", "none")
PHASE_BOUNDARY_MODELS = ("impermeable_flux", "projection_legacy")


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


def fluid_outward_normal(sdf, p: PhaseFieldParams):
    """Return n pointing from fluid to solid, computed from the signed distance.

    ``sdf`` is negative in solid and positive in fluid, so ``grad(sdf)`` points
    from solid into fluid and the fluid-domain outward normal is its negative.
    x is periodic; y uses one-sided physical-edge differences to avoid a seam ghost.
    """
    gx = (jnp.roll(sdf, -1, axis=0) - jnp.roll(sdf, 1, axis=0)) / (2.0 * p.dx)
    gy = _ddy_nonperiodic(sdf, p.dy)
    norm = jnp.sqrt(gx * gx + gy * gy + 1e-30)
    return -gx / norm, -gy / norm


def wall_energy_derivative(phi, cos_theta):
    """g_w'(phi) for the Young wall free energy, per unit wall area."""
    return -WALL_SIGMA0 * jnp.asarray(cos_theta) * wall_switch_derivative(phi)


def natural_wall_normal_derivative(phi, solid: Solid, p: PhaseFieldParams):
    """Prescribed fluid-outward derivative from ``eps*dphi/dn + g_w'(phi)=0``."""
    return -wall_energy_derivative(phi, solid.cos_theta) / p.eps


def wall_measure_is_cutcell(p: PhaseFieldParams) -> bool:
    """True when the production path uses the contract-v8 geometric cut-cell measure.

    The legacy reproduction modes keep their historical measure *pinned* so their
    trajectories stay reproducible: ``surface_energy_volume_v6`` (the v6 diffuse
    volume form) and ``projection_legacy`` (the v6/v7 FFT path) always use
    ``wall_delta``. Only ``impermeable_flux`` + ``surface_energy`` follows
    ``p.wall_measure``.
    """
    return bool(
        p.wall_measure == "sdf_cutcell_v1"
        and p.wetting_model == "surface_energy"
        and p.phase_boundary_model == "impermeable_flux"
    )


def wall_measure_density(solid: Solid, p: PhaseFieldParams):
    """Wall surface measure per unit cell volume, ``dA/dV`` (units 1/length).

    Contract v8 production: the exact cut-cell wall length divided by the cell
    volume, ``A_wall,i / (dx dy)``. Its cell-volume integral is the geometric wall
    length and it is independent of where the wall falls inside a cell. The pinned
    legacy modes and ``wall_measure='diffuse_sdf_v7'`` return the v7 normalized
    diffuse kernel ``wall_delta``, whose fluid-side share is grid-alignment
    dependent (L1A-2d root cause).
    """
    if wall_measure_is_cutcell(p):
        return solid.wall_area / (p.dx * p.dy)
    return wall_delta(solid.sdf, p)


def _pad_y(field: jnp.ndarray, count: int) -> jnp.ndarray:
    """Replicate the first/last y rows so y-shifts never wrap into the other boundary."""
    bottom = jnp.repeat(field[:, :1], count, axis=1)
    top = jnp.repeat(field[:, -1:], count, axis=1)
    return jnp.concatenate([bottom, field, top], axis=1)


def fluid_aware_gradient(phi, solid: Solid, p: PhaseFieldParams):
    """Second-order gradients of ``phi`` that never reach across the impermeable wall.

    A centred difference at the first fluid cell above a wall would differentiate into a
    hard-solid cell whose ``phi`` is frozen by the face apertures, which biases the wall
    normal derivative. Where the neighbour in a direction is hard solid, this switches to the
    three-point second-order one-sided stencil into the fluid, so the gradient stays
    ``O(dx^2, dy^2)`` up to the wall. x remains periodic; y is padded by edge replication,
    never wrapped. This is the JAX counterpart of the diagnostic stencil used by
    ``production.contact_line_kinetics._fluid_aware_gradients``.
    """
    fluid = solid.sdf >= 0.0
    dx, dy = p.dx, p.dy
    phi_xp1 = jnp.roll(phi, -1, axis=0)
    phi_xm1 = jnp.roll(phi, 1, axis=0)
    phi_xp2 = jnp.roll(phi, -2, axis=0)
    phi_xm2 = jnp.roll(phi, 2, axis=0)
    fluid_xp1 = jnp.roll(fluid, -1, axis=0)
    fluid_xm1 = jnp.roll(fluid, 1, axis=0)
    fluid_xp2 = jnp.roll(fluid, -2, axis=0)
    fluid_xm2 = jnp.roll(fluid, 2, axis=0)
    grad_x = (phi_xp1 - phi_xm1) / (2.0 * dx)
    grad_x = jnp.where(
        fluid & (~fluid_xm1) & fluid_xp1 & fluid_xp2, (-3.0 * phi + 4.0 * phi_xp1 - phi_xp2) / (2.0 * dx), grad_x
    )
    grad_x = jnp.where(
        fluid & (~fluid_xp1) & fluid_xm1 & fluid_xm2, (3.0 * phi - 4.0 * phi_xm1 + phi_xm2) / (2.0 * dx), grad_x
    )

    pad = 2
    padded_phi = _pad_y(phi, pad)
    padded_fluid = _pad_y(fluid.astype(phi.dtype), pad)

    def shift_y(array, offset):
        return array[:, pad + offset : pad + offset + array.shape[1] - 2 * pad]

    phi_yp1, phi_ym1 = shift_y(padded_phi, 1), shift_y(padded_phi, -1)
    phi_yp2, phi_ym2 = shift_y(padded_phi, 2), shift_y(padded_phi, -2)
    fluid_yp1, fluid_ym1 = shift_y(padded_fluid, 1) > 0.5, shift_y(padded_fluid, -1) > 0.5
    fluid_yp2, fluid_ym2 = shift_y(padded_fluid, 2) > 0.5, shift_y(padded_fluid, -2) > 0.5
    grad_y = _ddy_nonperiodic(phi, dy)
    grad_y = jnp.where(
        fluid & (~fluid_ym1) & fluid_yp1 & fluid_yp2, (-3.0 * phi + 4.0 * phi_yp1 - phi_yp2) / (2.0 * dy), grad_y
    )
    grad_y = jnp.where(
        fluid & (~fluid_yp1) & fluid_ym1 & fluid_ym2, (3.0 * phi - 4.0 * phi_ym1 + phi_ym2) / (2.0 * dy), grad_y
    )
    return grad_x, grad_y


def wall_plane_phi(phi, solid: Solid, p: PhaseFieldParams):
    """DIAGNOSTIC: ``phi`` extrapolated from each control-cell centre to the wall it represents.

        phi_wall,i = phi_i + d_i * (n_i . grad phi_i)

    with the *geometric* distance ``d_i = solid.wall_distance`` and the area-weighted wall
    normal ``n_i`` (fluid -> solid). This is a geometry-only construction: ``d_i`` and ``n_i``
    come from the cut contour, never from ``theta``, ``phi`` or any fitted quantity, and the
    extrapolation is exact for a locally linear profile. It removes the ``O(d/eps)`` error of
    imposing ``eps dphi/dn + g_w'(phi) = 0`` at a cell centre that sits up to one cell away
    from the wall -- the error that made the equilibrium depend on where the wall falls inside
    a cell (measured with cell-centre evaluation: about -1.8 deg/dx at 60 deg and -4.3 deg/dx
    at 120 deg, i.e. a 3.7 deg spread across the eight sub-cell offsets).

    The result is clamped to the physical volume-fraction range ``[0, 1]``. This is not a
    limiter on the physics: ``h(phi) = phi^2 (3 - 2 phi)`` is an interpolation function on
    ``[0, 1]``, and evaluating its cubic extension at an extrapolated ``phi > 1`` makes
    ``F_wall`` unbounded below, which destabilizes the explicit wall term (without the clamp a
    120 deg run at ``d = 0.92 dx`` overshoots to ``phi = 1.018`` in 10 steps and fails closed
    with NaNs by step 50). The clamp has no angle dependence and no fitted parameter, and it is
    inactive whenever the extrapolated wall value stays physical.
    """
    grad_x, grad_y = fluid_aware_gradient(phi, solid, p)
    normal_derivative = solid.wall_normal_x * grad_x + solid.wall_normal_y * grad_y
    return jnp.clip(phi + solid.wall_distance * normal_derivative, 0.0, 1.0)


def wall_chemical_potential(phi, solid: Solid, p: PhaseFieldParams):
    """``dF_wall^h/dphi`` per unit cell volume: the production wall operator.

    Exactly variational by construction (the gradient of :func:`wall_free_energy`), which for the
    shipped local energy is the analytic ``A_wall,i g_w'(phi_i)/(dx dy)``. The gradient form is
    kept so that any future wall-energy expression -- including the diagnostic wall-plane
    extrapolation of :func:`wall_plane_phi` -- cannot drift from the energy it is derived from.
    """
    return jax.grad(lambda field: wall_free_energy(field, solid, p))(phi) / (p.dx * p.dy)


def wall_measure_normal(solid: Solid, p: PhaseFieldParams):
    """Unit wall normal (fluid -> solid) attached to the active measure.

    The cut-cell measure carries the area-weighted contour normal of each control
    cell; the legacy diffuse kernel falls back to ``-grad(sdf)``.
    """
    if wall_measure_is_cutcell(p):
        return solid.wall_normal_x, solid.wall_normal_y
    return fluid_outward_normal(solid.sdf, p)


def wall_free_energy(phi, solid: Solid, p: PhaseFieldParams):
    """Discrete Young wall free energy ``F_wall^h`` (absolute, not a density).

    Production (contract v8)::

        F_wall^h = sum_i A_wall,i * g_w(phi_i, theta_i)

    with the geometric cut-cell length ``A_wall,i``, so the total measure equals the wall length
    and the variational derivative is exactly ``dF_wall^h/dphi_i = A_wall,i g_w'(phi_i)``, which
    divided by the cell volume is the wall term of :func:`chemical_potential`. The pinned legacy
    modes keep the diffuse volume form ``int g_w delta_wall dV``.

    ``g_w`` is evaluated at the control-cell value ``phi_i``, so the condition is imposed at a
    point ``d_i <= dx`` from the wall (a first-order wall-placement error, measured and reported
    by ``production/phase_boundary_audit._normal_bc_linearization_deviation``). The geometric
    distance ``solid.wall_distance`` and the extrapolated value :func:`wall_plane_phi` are
    shipped as *diagnostics*: imposing the condition at the wall plane instead was implemented
    and measured (L1A-2e), and it reduces the placement error but not the sub-cell alignment
    spread of the equilibrium angle, whose dominant term is the cell-centre hard-fluid mask that
    fixes the discrete transport domain (see ``production/README.md`` section I). Keeping the
    production operator local also keeps it unconditionally stable: the extrapolated cubic
    ``h`` needs a clamp to ``[0, 1]`` to avoid a runaway.
    """
    if p.wetting_model == "surface_energy" and wall_measure_is_cutcell(p):
        return jnp.sum(wall_energy_density(phi, solid.cos_theta) * solid.wall_area)
    if p.wetting_model in ("surface_energy", "surface_energy_volume_v6"):
        return jnp.sum(wall_energy_density(phi, solid.cos_theta) * wall_delta(solid.sdf, p)) * p.dx * p.dy
    if p.wetting_model == "legacy_affinity":
        delta_phi = phi - phi_wet_of(solid.cos_theta)
        return 0.5 * p.wall_energy_amp * jnp.sum(wet_band(solid, p) * delta_phi**2) * p.dx * p.dy
    return jnp.asarray(0.0, dtype=jnp.asarray(phi).dtype)


def natural_wall_gradient(phi, solid: Solid, p: PhaseFieldParams):
    """Cartesian gradient target ``(dphi/dn) n`` on the embedded wall."""
    nx, ny = wall_measure_normal(solid, p)
    derivative = natural_wall_normal_derivative(phi, solid, p)
    return derivative * nx, derivative * ny


def _natural_wall_laplacian_flux(phi, solid: Solid, p: PhaseFieldParams):
    """Embedded boundary flux contribution to ``lap(phi)``.

    The divergence theorem adds ``(dphi/dn) dA / dV`` to the cell Laplacian;
    :func:`wall_measure_density` supplies ``dA/dV``. Since contract v8 that measure
    is the exact geometric cut-cell wall area assigned to fluid-side control cells,
    so the total wall flux is proportional to the true wall length for every
    sub-cell wall position. This is the single natural-BC representation:
    ``chemical_potential`` consumes the returned Laplacian and does not separately
    add a diffuse ``mu_wall`` term.
    """
    return natural_wall_normal_derivative(phi, solid, p) * wall_measure_density(solid, p)


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


def _surface_energy_volume_mu(phi, solid: Solid, p: PhaseFieldParams):
    """Contract-v6 diffuse wall derivative; diagnostic/reproduction only."""
    return wall_energy_derivative(phi, solid.cos_theta) * wall_delta(solid.sdf, p)


def wetting_mu(phi, solid: Solid, p: PhaseFieldParams):
    """Standalone diffuse wall chemical-potential term for legacy/audit modes.

    The v7 production mode ``surface_energy`` returns exact zeros here because
    the Young energy is imposed once, through ``_natural_wall_laplacian_flux`` in
    :func:`chemical_potential`. To reproduce the contract-v6 diffuse-domain form,
    select ``surface_energy_volume_v6`` explicitly. ``legacy_affinity`` is the
    contract-v5 volume control; ``none`` switches it off. Unknown modes fail closed.
    """
    if p.wetting_model == "surface_energy":
        return jnp.zeros_like(phi)
    if p.wetting_model == "surface_energy_volume_v6":
        return _surface_energy_volume_mu(phi, solid, p)
    if p.wetting_model == "legacy_affinity":
        phi_w = phi_wet_of(solid.cos_theta)
        band = wet_band(solid, p)
        return -p.wall_energy_amp * band * (phi_w - phi)
    if p.wetting_model == "none":
        return jnp.zeros_like(phi)
    raise ValueError(f"unknown wetting_model {p.wetting_model!r}; expected one of {sorted(WETTING_MODELS)}")


def _explicit_chemical_potential(phi, solid: Solid, p: PhaseFieldParams):
    """All explicit terms, including exactly one selected wall-energy representation."""
    mu = fprime(phi) / p.eps
    if p.phase_boundary_model == "impermeable_flux" and p.wetting_model == "surface_energy":
        if wall_measure_is_cutcell(p):
            # Contract v8: the Young energy is imposed once, as the exact variational derivative
            # of F_wall^h = sum_i A_wall,i g_w(phi_i) -- the discrete form of the embedded Robin
            # boundary flux, not an additional mu_wall source on top of it.
            return mu + wall_energy_derivative(phi, solid.cos_theta) * wall_measure_density(solid, p)
        # Pinned contract-v7 reproduction: Robin boundary flux with the diffuse SDF kernel.
        return mu - p.eps * _natural_wall_laplacian_flux(phi, solid, p)
    if p.phase_boundary_model == "projection_legacy" and p.wetting_model == "surface_energy":
        # A historical contract-v6 trajectory used the diffuse-domain derivative.
        return mu + _surface_energy_volume_mu(phi, solid, p)
    return mu + wetting_mu(phi, solid, p)


def chemical_potential(phi, solid: Solid, p: PhaseFieldParams):
    """Return the bulk chemical potential with the selected wall treatment once.

    v7 uses the symmetric face Laplacian and embeds ``eps*dphi/dn=-g_w'`` into
    its boundary flux. The chemical-potential no-flux condition is imposed by the
    face apertures used in the CH transport, not by modifying this scalar field.
    The projection_legacy branch retains the fully-periodic contract-v6 operator.
    """
    mu_explicit = _explicit_chemical_potential(phi, solid, p)
    if p.phase_boundary_model == "impermeable_flux":
        return mu_explicit - p.eps * fluid_laplacian(phi, solid, p)
    return mu_explicit - p.eps * _lap(phi, p.dx, p.dy)


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
    """Momentum right-hand sides and explicit phase terms.

    For the v7 ``impermeable_flux`` model, ``phi_rhs`` is the conservative
    face-flux advection divergence and ``mu_expl`` contains f'(phi)/eps plus the
    one selected wall representation. The stiff ``-eps*L(phi)`` term is handled
    by :func:`solve_ch_implicit`. ``projection_legacy`` retains the v6 periodic
    advection and FFT-compatible chemical-potential split for reproduction.
    """
    phi, u, v = state.phi, state.u, state.v
    dx, dy = p.dx, p.dy

    mu = chemical_potential(phi, solid, p)
    mu_expl = _explicit_chemical_potential(phi, solid, p)

    if p.phase_boundary_model == "impermeable_flux":
        adv_x, adv_y = phase_advective_fluxes(u, v, phi, solid, p)
        phi_rhs = -divergence_from_face_fluxes(adv_x, adv_y, p)
    else:
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


def _cg_solve_impl(rhs_field, aperture_x, aperture_y, dx, dy, alpha, rtol, max_iterations):
    """CG core for ``(I + alpha L^T L)x = rhs``; L is the masked face Laplacian."""

    def laplacian(value):
        grad_x = aperture_x * (jnp.roll(value, -1, axis=0) - value) / dx
        grad_y = aperture_y * (jnp.roll(value, -1, axis=1) - value) / dy
        return (grad_x - jnp.roll(grad_x, 1, axis=0)) / dx + (grad_y - jnp.roll(grad_y, 1, axis=1)) / dy

    def operator(value):
        lap = laplacian(value)
        return value + alpha * laplacian(lap)

    x0 = jnp.zeros_like(rhs_field)
    residual0 = rhs_field - operator(x0)
    direction0 = residual0
    residual_sq0 = jnp.vdot(residual0, residual0).real
    rhs_norm = jnp.sqrt(jnp.vdot(rhs_field, rhs_field).real)
    scale = jnp.maximum(rhs_norm, jnp.asarray(1e-30, dtype=rhs_field.dtype))
    rel0 = jnp.sqrt(residual_sq0) / scale

    def condition(carry):
        _x, _r, _d, _rr, rel, iteration = carry
        return (iteration < max_iterations) & jnp.isfinite(rel) & (rel > rtol)

    def body(carry):
        x, residual, direction, residual_sq, _rel, iteration = carry
        image = operator(direction)
        denominator = jnp.vdot(direction, image).real
        valid_denominator = jnp.isfinite(denominator) & (denominator > 0.0)
        safe_denominator = jnp.where(valid_denominator, denominator, 1.0)
        step_length = residual_sq / safe_denominator
        x_new = x + step_length * direction
        r_candidate = residual - step_length * image
        r_new = jnp.where(valid_denominator, r_candidate, jnp.full_like(r_candidate, jnp.nan))
        residual_sq_new = jnp.vdot(r_new, r_new).real
        rel_new = jnp.sqrt(residual_sq_new) / scale
        safe_rr = jnp.maximum(residual_sq, jnp.asarray(1e-30, dtype=rhs_field.dtype))
        beta = residual_sq_new / safe_rr
        direction_new = r_new + beta * direction
        return x_new, r_new, direction_new, residual_sq_new, rel_new, iteration + 1

    x, _residual, _direction, _residual_sq, relative_residual, iterations = lax.while_loop(
        condition,
        body,
        (x0, residual0, direction0, residual_sq0, rel0, jnp.asarray(0, dtype=jnp.int32)),
    )
    converged = jnp.isfinite(relative_residual) & (relative_residual <= rtol)
    solution = jnp.where(converged, x, jnp.full_like(x, jnp.nan))
    return solution, ImplicitSolveInfo(iterations, relative_residual, converged)


@jax.custom_vjp
def _differentiable_ch_cg(rhs_field, aperture_x, aperture_y, dx, dy, alpha, rtol, max_iterations):
    """Implicitly differentiated matrix-free CG solve (adjoint uses the same SPD solve)."""
    return _cg_solve_impl(rhs_field, aperture_x, aperture_y, dx, dy, alpha, rtol, max_iterations)


def _differentiable_ch_cg_fwd(rhs_field, aperture_x, aperture_y, dx, dy, alpha, rtol, max_iterations):
    output = _cg_solve_impl(rhs_field, aperture_x, aperture_y, dx, dy, alpha, rtol, max_iterations)
    return output, (aperture_x, aperture_y, dx, dy, alpha, rtol, max_iterations, output[1].converged)


def _differentiable_ch_cg_bwd(residual, cotangents):
    aperture_x, aperture_y, dx, dy, alpha, rtol, max_iterations, forward_converged = residual
    solution_cotangent = cotangents[0]
    adjoint, adjoint_info = _cg_solve_impl(
        solution_cotangent, aperture_x, aperture_y, dx, dy, alpha, rtol, max_iterations
    )
    valid = forward_converged & adjoint_info.converged
    rhs_cotangent = jnp.where(valid, adjoint, jnp.full_like(adjoint, jnp.nan))
    # Geometry and physical coefficients are fixed/static for this solver path.
    return rhs_cotangent, None, None, None, None, None, None, None


_differentiable_ch_cg.defvjp(_differentiable_ch_cg_fwd, _differentiable_ch_cg_bwd)


def solve_ch_implicit(rhs_field, solid: Solid, p: PhaseFieldParams, dt: float):
    """Solve ``(I + dt*M*eps*L^T L) phi = rhs`` by matrix-free CG.

    ``L`` is the symmetric face-aperture Laplacian. No dense matrix or FFT
    assumption is used. The operator is SPD because ``L`` is symmetric and
    negative semidefinite. The custom VJP differentiates the implicit equation
    with a matching adjoint CG solve, so the v7 phase path remains usable in
    gradient-based HydroGym/FNO/RL workflows. A failed solve returns NaNs and an
    explicit ``converged=False`` diagnostic; it can never silently advance.
    """
    aperture_x, aperture_y = fluid_face_apertures(solid, p)
    alpha = jnp.asarray(float(dt) * float(p.M) * float(p.eps), dtype=rhs_field.dtype)
    rtol = jnp.asarray(p.ch_solver_rtol, dtype=rhs_field.dtype)
    max_iterations = jnp.asarray(p.ch_solver_max_iterations, dtype=jnp.int32)
    return _differentiable_ch_cg(
        rhs_field,
        aperture_x,
        aperture_y,
        jnp.asarray(p.dx, dtype=rhs_field.dtype),
        jnp.asarray(p.dy, dtype=rhs_field.dtype),
        alpha,
        rtol,
        max_iterations,
    )


def _phase_update(phi, u, v, solid: Solid, p: PhaseFieldParams, dt: float, phi_rhs, mu_expl):
    """One phase substep; v7 uses only conservative face fluxes and matrix-free CG."""
    if p.phase_boundary_model == "impermeable_flux":
        ch_x, ch_y = chemical_potential_fluxes(mu_expl, solid, p)
        source = phi_rhs - divergence_from_face_fluxes(ch_x, ch_y, p)
        return solve_ch_implicit(phi + dt * source, solid, p, dt)

    # Reproduction-only contract-v6 path: constant-M periodic FFT biharmonic
    # update followed, optionally, by the historical mass redistribution.
    m2 = p.m2
    denominator = 1.0 + dt * p.M * p.eps * m2**2
    source_hat = jnp.fft.rfft2(phi_rhs) - p.M * m2 * jnp.fft.rfft2(mu_expl)
    phi_hat = (jnp.fft.rfft2(phi) + dt * source_hat) / denominator
    phi_new = jnp.fft.irfft2(phi_hat, s=phi.shape)
    if p.enforce_solid_phi:
        phi_new = _project_phase_outside_solid(phi_new, solid, p)
    zero = jnp.asarray(0, dtype=jnp.int32)
    one = jnp.asarray(0.0, dtype=phi.dtype)
    return phi_new, ImplicitSolveInfo(zero, one, jnp.asarray(True))


def phase_transport_step(phi, u, v, solid: Solid, p: PhaseFieldParams, dt: float | None = None):
    """Advance only the phase equation (useful for isolated CH-energy audits)."""
    dt = p.dt / 3.0 if dt is None else float(dt)
    if p.phase_boundary_model == "impermeable_flux":
        adv_x, adv_y = phase_advective_fluxes(u, v, phi, solid, p)
        ch_mu = _explicit_chemical_potential(phi, solid, p)
        ch_x, ch_y = chemical_potential_fluxes(ch_mu, solid, p)
        source = -divergence_from_face_fluxes(adv_x, adv_y, p) - divergence_from_face_fluxes(ch_x, ch_y, p)
        return solve_ch_implicit(phi + dt * source, solid, p, dt)
    advective_rhs = -div_upwind(u, v, phi, p.dx, p.dy)
    mu_exp = _explicit_chemical_potential(phi, solid, p)
    return _phase_update(phi, u, v, solid, p, dt, advective_rhs, mu_exp)


def step_with_diagnostics(state: State, solid: Solid, p: PhaseFieldParams):
    """One public step and its per-substep CG residual/iteration diagnostics."""
    dt = p.dt / 3.0

    def substep(carry, _):
        phi, u, v, t = carry
        phi_rhs, u_rhs, v_rhs, _mu, mu_expl = rhs(State(phi, u, v, t), solid, p)
        phi_new, solve_info = _phase_update(phi, u, v, solid, p, dt, phi_rhs, mu_expl)

        # Brinkman penalization remains the existing implicit velocity damping.
        damp = 1.0 / (1.0 + dt * solid.chi / p.eta_pen)
        u_new = (u + dt * u_rhs) * damp
        v_new = (v + dt * v_rhs) * damp

        # Momentum/pressure projection is intentionally unchanged by L1A-2c.
        div = _ddx(u_new, p.dx) + _ddy(v_new, p.dy)
        pr = poisson_solve(div / dt, p.m2_proj)
        u_new = u_new - dt * _ddx(pr, p.dx)
        v_new = v_new - dt * _ddy(pr, p.dy)
        return (phi_new, u_new, v_new, t + dt), solve_info

    (phi, u, v, t), info = lax.scan(substep, (state.phi, state.u, state.v, state.t), None, length=3)
    diagnostics = StepDiagnostics(info.iterations, info.relative_residual, info.converged)
    return State(phi=phi, u=u, v=v, t=t), diagnostics


def step(state: State, solid: Solid, p: PhaseFieldParams) -> State:
    """Advance one requested dt using three semi-implicit Euler substeps.

    In the default v7 path the CH stiffness is treated by a fail-closed,
    matrix-free iterative solve and solid boundaries are enforced in the face
    fluxes. The legacy FFT/projection mechanism is unreachable unless explicitly
    selected through ``phase_boundary_model='projection_legacy'``.
    """
    return step_with_diagnostics(state, solid, p)[0]


def phase_only_step_with_diagnostics(state: State, solid: Solid, p: PhaseFieldParams):
    """Advance one public dt with u = v = 0 using the exact v7 phase operator (L1A-2d).

    Performs the same three ``dt = p.dt / 3.0`` substeps as :func:`step_with_diagnostics`,
    calling :func:`phase_transport_step` with identically zero velocity fields so
    that Cahn-Hilliard thermodynamic relaxation is isolated from Navier-Stokes,
    Brinkman, capillary-momentum, and pressure-projection coupling.
    """
    dt = p.dt / 3.0
    zero_u = jnp.zeros_like(state.phi)
    zero_v = jnp.zeros_like(state.phi)

    def substep(carry, _):
        phi, t = carry
        phi_new, solve_info = phase_transport_step(phi, zero_u, zero_v, solid, p, dt=dt)
        return (phi_new, t + dt), solve_info

    (phi, t), info = lax.scan(substep, (state.phi, state.t), None, length=3)
    diagnostics = StepDiagnostics(info.iterations, info.relative_residual, info.converged)
    return State(phi=phi, u=zero_u, v=zero_v, t=t), diagnostics


def phase_only_step(state: State, solid: Solid, p: PhaseFieldParams) -> State:
    """Advance one public dt in CH-only mode (u = v = 0) using the v7 phase operator."""
    return phase_only_step_with_diagnostics(state, solid, p)[0]


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


def phase_free_energy(phi, solid: Solid, p: PhaseFieldParams):
    """Discrete bulk + wall free energy consistent with the production operator.

    Each open +axis face is counted once. The wall contribution uses *the same*
    measure as the operator (:func:`wall_free_energy`): the exact cut-cell wall
    length ``sum_i A_wall,i g_w(phi_i)`` in the contract-v8 production path, the
    pinned diffuse ``int g_w delta_wall dV`` for the legacy reproduction modes. Its
    derivative enters the chemical potential solely through the embedded natural
    boundary flux, so there is exactly one wall-energy contribution.
    """
    fluid = (solid.sdf >= 0.0).astype(phi.dtype)
    aperture_x, aperture_y = fluid_face_apertures(solid, p)
    grad_x = (jnp.roll(phi, -1, axis=0) - phi) / p.dx
    grad_y = (jnp.roll(phi, -1, axis=1) - phi) / p.dy
    bulk = jnp.sum(fluid * phi**2 * (1.0 - phi) ** 2 / p.eps)
    bulk += 0.5 * p.eps * jnp.sum(aperture_x * grad_x**2 + aperture_y * grad_y**2)
    # ``wall_free_energy`` is the single definition of the wall term (absolute
    # energy) shared with the operator, so the two can never drift apart.
    return bulk * p.dx * p.dy + wall_free_energy(phi, solid, p)


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


def discrete_fluid_boundary_height(solid: Solid, p: PhaseFieldParams):
    """DIAGNOSTIC: the wall plane the cell-centre fluid mask actually gives the phase field.

    The phase transport domain is ``sdf >= 0`` at cell *centres* (contract v7), so the lowest
    hard-fluid row starts at its bottom face ``j_first * dy`` -- up to one cell away from the
    geometric ``sdf = 0`` wall, and jumping by a full cell as the wall crosses a centre. The L1A-2e
    translation study measures the same interface against both planes to separate the wall-measure
    alignment error (zero by construction) from this transport-domain alignment error.
    """
    fluid = solid.sdf >= 0.0
    rows = jnp.any(fluid, axis=0)
    first_fluid_row = jnp.argmax(rows.astype(jnp.int32))
    return first_fluid_row * p.dy


def measure_contact_angle(
    phi,
    solid: Solid,
    p: PhaseFieldParams,
    level: float = 0.5,
    cutoff_factor: float = 1.0,
    wall_plane: float | None = None,
) -> float:
    """Apparent contact angle (deg) from a circle fitted to the interface contour.

    ``theta = acos((y_w - y_c) / Rc)`` with the fitted centre ``(x_c, y_c)`` and
    radius ``Rc`` and the wall plane ``y_w``.  Contour points closer to the solid
    than ``max(cutoff_factor * eps, dx)`` are excluded; the defaults are the ones
    validated on synthetic circular caps (N = 128: MAE 0.007 deg, max 0.016 deg).
    Returns ``nan`` when the contour is too short or degenerate to fit.

    ``y_w`` is the geometric ``sdf = 0`` crossing of the drop's column
    (:func:`wall_plane_height`). ``wall_plane`` overrides it for diagnostics only -- the L1A-2e
    alignment study uses it to measure the same interface against the *discrete* transport
    boundary (the bottom face of the lowest hard-fluid row), which is what the cell-centre fluid
    mask actually gives the phase field.
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
    wall = wall_plane_height(solid, p, x0=reference) if wall_plane is None else float(wall_plane)
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
    return Solid(
        chi=z,
        ds=z,
        cos_theta=z,
        sdf=jnp.ones_like(z),
        chi_hard=jnp.zeros_like(z),
        wall_area=z,
        wall_normal_x=z,
        wall_normal_y=z,
        wall_distance=z,
    )


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
        # New production cases use impermeable face fluxes, not redistribution.
        phase_boundary_model=str(case.get("phase_boundary_model", "impermeable_flux")),
        enforce_solid_phi=bool(case.get("enforce_solid_phi", False)),
        wetting_model=str(case.get("wetting_model", "surface_energy")),
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
    if p.phase_boundary_model == "impermeable_flux":
        # Initial diffuse tails are clipped only at initialization (without
        # redistribution); subsequent updates cannot transport phase through a
        # fluid-solid face. Clearance validation keeps this mass correction tiny.
        state = State(
            phi=jnp.where(solid.sdf >= 0.0, state.phi, 0.0),
            u=state.u,
            v=state.v,
            t=state.t,
        )
    return p, solid, state


def downsample(field, f):
    """Average-pool by factor f on both axes."""
    Nx, Ny = field.shape
    return field.reshape(Nx // f, f, Ny // f, f).mean(axis=(1, 3))
