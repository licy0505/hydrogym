"""L1A-2i: phase-state storage precision and the drift-clean CHNS closure.

Frozen upstream diagnosis (L1A-2h): the residual contract-v10 phase-mass drift is a systematic
float32 **phase-state storage** bias -- the update is rounded onto the float32 grid and the lost
increment is not remembered -- not a Krylov defect. This module compares three state
representations under *identical* physics:

``A0_float32``            contract-10 baseline: ``phi`` in ``p.dtype`` (float32).
``A1_phase_float64``      ``phi`` stored in float64; the phase transport, chemical potential and
                          implicit solve follow the field; ``u``/``v``, the geometry arrays, the
                          momentum terms and the pressure projection stay float32.
``B1_compensated``        ``phi = phi_hi + phi_lo`` (both float32), a TwoSum-style compensated
                          local accumulation. Operators read ``phi_hi``.
``C1_residual_feedback``  ``phi`` float32 plus a per-cell residual memory ``r_store`` fed back into
                          the next local update. Operators read ``phi``.
``F_full_float64``        full float64 reference (phase *and* velocity).

Rules this module obeys (spec 19, 32, 33):

* ``B1``/``C1`` are strictly cell-local: no global mass is read, no neighbour is touched, nothing is
  redistributed, no contact angle is calibrated;
* the formal conservation gate is always evaluated on the field the *operators see*
  (``phi_phys``), never on a hidden bookkeeping sum. The compensated sum is reported separately and
  labelled as bookkeeping.

Nothing in this module changes a production default; the phase-only float64 model is the opt-in
``phase_storage_model`` of :mod:`phasefield` and the default stays ``float32_contract_10``.

Profiles: ``quick`` (small N, short horizons), ``baseline`` (contract-10 reproduction), ``forensic``
(the full matrix, plus the long-horizon and restart matrices).
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import phasefield as pf
from production import krylov_roundoff_audit as kra
from production import mass_precision_audit as mpa

STAGE = "L1A-2i"
MODULE = "production.phase_storage_precision_audit"

#: The state representations compared by this stage.
CANDIDATES = (
    "A0_float32",
    "A1_phase_float64",
    "A2_float64_storage_f32_krylov",
    "B1_compensated",
    "C1_residual_feedback",
    "F_full_float64",
)
#: Candidates whose persistent phase field is float64.
PHASE_FLOAT64_CANDIDATES = ("A1_phase_float64", "A2_float64_storage_f32_krylov", "F_full_float64")
#: The Krylov layer stays float32 for A2 (mixed precision: storage float64, solve float32).
FLOAT32_KRYLOV_CANDIDATES = ("A0_float32", "A2_float64_storage_f32_krylov", "B1_compensated", "C1_residual_feedback")
CANDIDATE_LABELS = {
    "A0_float32": "contract-10 float32 phase state",
    "A1_phase_float64": "phase-only float64 storage",
    "A2_float64_storage_f32_krylov": "phase-only float64 storage with the float32 Krylov recurrence",
    "B1_compensated": "compensated local accumulation (phi_hi + phi_lo, operators read phi_hi)",
    "C1_residual_feedback": "local residual feedback (phi + r_store, operators read phi)",
    "F_full_float64": "full float64 reference",
}
#: Candidates that carry a second persistent phase field (hidden numerical state).
HIDDEN_STATE_CANDIDATES = ("B1_compensated", "C1_residual_feedback")
#: The storage model each candidate would become in production (spec 50).
PHASE_STORAGE_MODEL_OF = {
    "A0_float32": "float32_contract_10",
    "A1_phase_float64": "phase_only_float64_v1",
    "A2_float64_storage_f32_krylov": "phase_only_float64_v1 + float32_krylov_v1",
    "B1_compensated": "compensated_local_accumulator_v1",
    "C1_residual_feedback": "residual_feedback_v1",
    "F_full_float64": "full_float64_reference",
}
#: Machine-readable rejection reasons (spec 49).
REJECTION_REASONS = (
    "MASS_GATE_FAIL",
    "LINEAR_BIAS_REMAINS",
    "PHYSICS_DRIFT",
    "ENERGY_REGRESSION",
    "RESTART_NOT_REPRODUCIBLE",
    "AD_FAILURE",
    "GPU_COST_TOO_HIGH",
    "RUNTIME_COST_TOO_HIGH",
    "MEMORY_COST_TOO_HIGH",
    "DATASET_STATE_AMBIGUOUS",
    "IMPLEMENTATION_TOO_INVASIVE",
    "INCONCLUSIVE",
    "NOT_LEAST_INVASIVE",
)

# ------------------------------------------------------------------ production-faithful CG
def production_correction(rhs_field, volume_safe, weight_x, weight_y, alpha, rtol, max_iterations):
    """The exchange correction ``x`` of contract v10's shipped solve, returned instead of ``rhs + x``.

    This is a deliberate line-by-line copy of :func:`phasefield._cg_solve_volume_weighted` -- same
    operations, same order, same NaN gating, same convergence test -- with the single difference that
    it hands back ``x`` rather than the sum. The audit needs ``x`` because every B/C storage rule is
    defined on the *last addition* ``fl(rhs + x)`` and its exact residual; recovering ``x`` from
    ``solved - rhs`` is a second rounding of the quantity under study. The copy is pinned by
    ``test_phase_storage_precision.py::test_production_correction_matches_shipped_solve``, which
    asserts ``rhs + x`` is bit-identical to ``phasefield.solve_ch_implicit(rhs)`` across fixtures and
    iteration counts -- so *every* candidate is compared on the same solver and A0 is production.
    """
    def operator(value):
        return pf.volume_weighted_operator(value, volume_safe, weight_x, weight_y, alpha)

    correction_rhs = rhs_field - operator(rhs_field)
    x0 = jnp.zeros_like(rhs_field)
    residual0 = correction_rhs
    direction0 = residual0
    residual_sq0 = pf.volume_weighted_inner(residual0, residual0, volume_safe)
    rhs_norm = jnp.sqrt(pf.volume_weighted_inner(rhs_field, rhs_field, volume_safe))
    scale = jnp.maximum(rhs_norm, jnp.asarray(1.0e-30, dtype=rhs_field.dtype))
    rel0 = jnp.sqrt(residual_sq0) / scale

    def condition(carry):
        _x, _r, _d, _rr, rel, iteration = carry
        return (iteration < max_iterations) & jnp.isfinite(rel) & (rel > rtol)

    def body(carry):
        x, residual, direction, residual_sq, _rel, iteration = carry
        image = operator(direction)
        denominator = pf.volume_weighted_inner(direction, image, volume_safe)
        valid_denominator = jnp.isfinite(denominator) & (denominator > 0.0)
        safe_denominator = jnp.where(valid_denominator, denominator, 1.0)
        step_length = residual_sq / safe_denominator
        x_new = x + step_length * direction
        r_candidate = residual - step_length * image
        r_new = jnp.where(valid_denominator, r_candidate, jnp.full_like(r_candidate, jnp.nan))
        residual_sq_new = pf.volume_weighted_inner(r_new, r_new, volume_safe)
        rel_new = jnp.sqrt(residual_sq_new) / scale
        safe_rr = jnp.maximum(residual_sq, jnp.asarray(1.0e-30, dtype=rhs_field.dtype))
        beta = residual_sq_new / safe_rr
        direction_new = r_new + beta * direction
        return x_new, r_new, direction_new, residual_sq_new, rel_new, iteration + 1

    x, _residual, _direction, _residual_sq, relative_residual, iterations = jax.lax.while_loop(
        condition,
        body,
        (x0, residual0, direction0, residual_sq0, rel0, jnp.asarray(0, dtype=jnp.int32)),
    )
    converged = jnp.isfinite(relative_residual) & (relative_residual <= rtol)
    return x, converged, iterations


# ------------------------------------------------------------------ TwoSum / compensation
def two_sum(a, b):
    """Knuth's error-free ``a + b``: returns ``(s, e)`` with ``s + e == a + b`` exactly.

    ``s = fl(a + b)`` and ``e`` is the exact rounding residual of that single addition (no overflow
    assumed). In JAX this is a handful of elementwise ops; its gradient flows through ``s`` as for a
    plain add plus the ``e`` path, which the AD audit measures rather than assumes.
    """
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def renormalize_pair(hi, lo):
    """Renormalize ``(hi, lo)`` so ``|lo| <= 0.5 ulp(hi)`` (an error-free expansion)."""
    return two_sum(hi, lo)


def compensated_update(rhs, correction, lo):
    """B1: fold production's final state-update rounding into a compensated pair.

    ``rhs`` and ``correction`` are exactly what the float32 production path computes; the pair
    ``(hi, lo)`` is the compensated representation of their sum. Returns the new pair and the exact
    residual of the update.
    """
    s, e = two_sum(rhs, correction)
    lo_new = lo + e
    hi_new, lo_fold = renormalize_pair(s, lo_new)
    return hi_new, lo_fold, e


def residual_feedback_update(rhs, correction, r_store):
    """C1: carry the local update residual in ``r_store`` and feed it into the next update."""
    increment = correction + r_store
    phi_new, e = two_sum(rhs, increment)
    return phi_new, e, increment


def operator_field(state, candidate):
    """The phase field the production operators read (``phi_phys``) for a candidate."""
    if candidate == "B1_compensated":
        return state["phi"]  # phi_hi
    return state["phi"]


def bookkeeping_mass(state, volume, candidate):
    """The compensated bookkeeping sum, reported separately from the physical mass (spec 32/33)."""
    if candidate in HIDDEN_STATE_CANDIDATES:
        total = state["phi"].astype(jnp.float64) + state["aux"].astype(jnp.float64)
    else:
        total = state["phi"].astype(jnp.float64)
    return jnp.sum(volume * total)


def physical_mass(state, volume, candidate):
    """``M = sum_i V_i phi_phys_i`` with ``phi_phys`` the field the operators actually use."""
    field = operator_field(state, candidate)
    return jnp.sum(volume * field.astype(jnp.float64))


def volume_of(operator):
    return jnp.asarray(operator.volume_safe, jnp.float64)


# ------------------------------------------------------------------ fixtures
def storage_dtype(p, candidate):
    return jnp.float64 if candidate in PHASE_FLOAT64_CANDIDATES else p.dtype


def build_case(
    N: int,
    *,
    target_deg: float = 150.0,
    M_factor: float = 4.0,
    rtol: float = 1.0e-6,
    candidate: str = "A0_float32",
    wall_offset_over_dy: float = 0.0,
    R: float = 1.1,
):
    """The frozen L1A-2h fixture, built for one storage candidate.

    Geometry, parameters and initial profile are identical across candidates; only the persistent
    phase representation changes. ``F_full_float64`` additionally promotes the velocity state.
    """
    if candidate not in CANDIDATES:
        raise KeyError(f"unknown candidate {candidate!r}; known: {list(CANDIDATES)}")
    wall_height = 0.25 + float(wall_offset_over_dy) * (6.0 / int(N))
    p, solid, state32 = mpa.build_case(
        N, M=M_factor * mpa.M_REF, rtol=rtol, target_deg=target_deg, wall_height=wall_height, R=R
    )
    if candidate in PHASE_FLOAT64_CANDIDATES:
        p = dataclasses.replace(p, phase_storage_model="phase_only_float64_v1")
        solid = pf.make_solid(
            pf.surface_flat(p, wall_height=wall_height), p, cos_theta=math.cos(math.radians(target_deg))
        )
        base = pf.sessile_initial_state(p, solid, R=R, wall_height=wall_height)
        phi64 = base.phi
        if candidate == "F_full_float64":
            aux = jnp.zeros(phi64.shape, jnp.float64)
        else:
            aux = jnp.zeros(phi64.shape, p.dtype)
        u = base.u.astype(jnp.float64) if candidate == "F_full_float64" else base.u
        v = base.v.astype(jnp.float64) if candidate == "F_full_float64" else base.v
    else:
        phi64 = state32.phi
        aux = jnp.zeros(phi64.shape, p.dtype)
        u, v = state32.u, state32.v
    operator = pf.phase_transport_operator(solid, p)
    state = {
        "phi": phi64,
        "u": u,
        "v": v,
        "t": jnp.asarray(0.0, jnp.float64 if candidate == "F_full_float64" else p.dtype),
        "aux": aux,
    }
    volume = volume_of(operator)
    m0 = float(physical_mass(state, volume, candidate))
    e_round = float(mpa.reduction_spread(operator.volume_safe, state32.phi)["E_round"])
    return p, solid, state, operator, m0, e_round


def to_state(state):
    """The ``phasefield.State`` of the operator-visible fields (aux is not a physical field)."""
    return pf.State(phi=state["phi"], u=state["u"], v=state["v"], t=state["t"])


# ------------------------------------------------------------------ the candidate substep
def exchange_increment(state, solid, p, operator, dt, candidate, *, frozen_correction=None):
    """``(rhs, correction, converged, iterations)`` of one candidate's exchange solve.

    Thin wrapper over :func:`production_fields` for the sections that do not need the momentum
    right-hand sides (the fixed-state causal test works at a frozen state).
    """
    rhs, correction, converged, iterations, _u_rhs, _v_rhs = production_fields(
        state, solid, p, operator, dt, candidate, frozen_correction=frozen_correction
    )
    return rhs, correction, converged, iterations


def production_fields(state, solid, p, operator, dt, candidate, *, frozen_correction=None):
    """``(rhs, correction, converged, iterations, u_rhs, v_rhs)`` -- the whole contract-10 update.

    The physics is contract 10 untouched; only the *working* dtype and, for A2, the dtype of the
    Krylov recurrence differ. ``frozen_correction`` is an AD-probe escape hatch (see
    :func:`gradient_check`) and is never used by a conservation measurement.
    """
    working = jnp.float64 if candidate in PHASE_FLOAT64_CANDIDATES else p.dtype
    nxt = to_state(state)
    _phi_rhs, u_rhs, v_rhs, _mu, mu_expl = pf.rhs(nxt, solid, p)
    ch = -pf.control_volume_divergence(
        *pf.chemical_potential_fluxes(mu_expl, solid, p), operator.volume_safe
    )
    advective = kra.production_advective_rate(
        state["phi"], state["u"], state["v"], solid, p, dt, _phi_rhs
    )
    phi = state["phi"].astype(working)
    rhs = phi + dt * (advective.astype(working) + ch.astype(working))

    alpha = jnp.asarray(dt * float(p.M) * float(p.eps), working)
    rtol = jnp.asarray(float(p.ch_solver_rtol), working)
    max_iterations = jnp.asarray(int(p.ch_solver_max_iterations), jnp.int32)
    if frozen_correction is not None:
        return rhs, frozen_correction, jnp.asarray(True), jnp.asarray(0, jnp.int32), u_rhs, v_rhs
    if candidate in FLOAT32_KRYLOV_CANDIDATES and working == jnp.float64:
        # A2 (mixed precision): the *state* is float64 and the update is added in float64, but the
        # exchange solve runs its recurrence in float32. The downcast touches only the solve's
        # right-hand side; the correction itself is mass-neutral by the telescoping of the cut-cell
        # divergence, so nothing is rounded back onto the float32 state grid.
        correction, converged, iterations = production_correction(
            rhs.astype(p.dtype),
            operator.volume_safe,
            operator.weight_x,
            operator.weight_y,
            alpha.astype(p.dtype),
            rtol.astype(p.dtype),
            max_iterations,
        )
        return rhs, correction, converged, iterations, u_rhs, v_rhs
    correction, converged, iterations = production_correction(
        rhs, operator.volume_safe, operator.weight_x, operator.weight_y, alpha, rtol, max_iterations
    )
    return rhs, correction, converged, iterations, u_rhs, v_rhs


def apply_storage_rule(rhs, correction, converged, aux, candidate):
    """``(stored, aux_new, residual)``: how one candidate persists ``rhs + correction``.

    ``residual`` is the exact difference between what was stored and the float64 evaluation of the
    same two terms -- the quantity the formal mass gate is *not* allowed to hide (spec 19/32/33).
    """
    production_value = jnp.where(converged, rhs + correction, jnp.full_like(rhs, jnp.nan))
    if candidate in PHASE_FLOAT64_CANDIDATES:
        production_value = production_value.astype(jnp.float64)
    if candidate == "B1_compensated":
        stored, aux_new, residual = compensated_update(rhs, correction, aux)
        return stored, aux_new, residual
    if candidate == "C1_residual_feedback":
        stored, aux_new, _increment = residual_feedback_update(rhs, correction, aux)
        exact = rhs.astype(jnp.float64) + correction.astype(jnp.float64)
        return stored, aux_new, stored.astype(jnp.float64) - exact
    exact = rhs.astype(jnp.float64) + correction.astype(jnp.float64)
    return production_value, aux, production_value.astype(jnp.float64) - exact


def storage_substep(state, solid, p, operator, dt, candidate, *, momentum: bool = True,
                    frozen_correction=None, diagnostics: bool = False):
    """One production substep with the candidate's phase-state representation.

    The *physics* is contract 10 untouched: the same fluxes, the same chemical potential, the same
    exchange solve, the same momentum damping and projection. Only the way the updated phase field
    is *stored* differs.
    """
    rhs, correction, converged, iterations, u_rhs, v_rhs = production_fields(
        state, solid, p, operator, dt, candidate, frozen_correction=frozen_correction
    )
    info = pf.ImplicitSolveInfo(iterations, jnp.asarray(float(p.ch_solver_rtol), rhs.dtype), converged)
    phi_new, aux_new, _residual = apply_storage_rule(rhs, correction, converged, state["aux"], candidate)

    if momentum:
        damp = 1.0 / (1.0 + dt * solid.chi / p.eta_pen)
        u_new = (state["u"] + dt * u_rhs) * damp
        v_new = (state["v"] + dt * v_rhs) * damp
        divergence = pf._ddx(u_new, p.dx) + pf._ddy(v_new, p.dy)
        pressure = pf.poisson_solve(divergence / dt, p.m2_proj)
        u_new = u_new - dt * pf._ddx(pressure, p.dx)
        v_new = v_new - dt * pf._ddy(pressure, p.dy)
        if candidate == "F_full_float64":
            u_new = u_new.astype(jnp.float64)
            v_new = v_new.astype(jnp.float64)
    else:
        u_new, v_new = state["u"], state["v"]

    out = {
        "phi": phi_new,
        "u": u_new,
        "v": v_new,
        "t": state["t"] + jnp.asarray(dt, state["t"].dtype),
        "aux": aux_new,
    }
    if diagnostics:
        # the two mass terms of one update, in mass units: the mass of the exchange correction the
        # solve returned, and the mass the storage rule dropped relative to the float64 sum of its
        # own two terms. Their running sums explain the drift series (spec 26/27).
        exact = rhs.astype(jnp.float64) + correction.astype(jnp.float64)
        out["solve_defect"] = jnp.sum(volume_of(operator) * correction.astype(jnp.float64))
        out["storage_loss"] = jnp.sum(volume_of(operator) * (phi_new.astype(jnp.float64) - exact))
    return out, info


def storage_step(state, solid, p, operator, candidate, *, substeps: int = 3):
    """One public step = ``substeps`` phase substeps of ``p.dt / substeps`` with the momentum path."""
    dt = p.dt / float(substeps)
    for _ in range(int(substeps)):
        state, info = storage_substep(state, solid, p, operator, dt, candidate)
    return state, info


# ------------------------------------------------------------------ how the pairs behave
def pair_normalization(phi, aux):
    """Fraction of cells violating ``|phi_lo| <= 0.5 ulp(phi)`` and the worst ratio."""
    aux64 = jnp.asarray(aux, jnp.float64)
    spacing = jnp.abs(jnp.spacing(jnp.asarray(phi, jnp.float32)).astype(jnp.float64))
    bound = 0.5 * spacing
    ratio = jnp.abs(aux64) / jnp.where(bound > 0, bound, 1e-45)
    return {
        "violating_fraction": float(jnp.mean(ratio > 1.0)),
        "max_ratio": float(jnp.max(ratio)),
        "max_abs_aux": float(jnp.max(jnp.abs(aux64))),
        "nonzero_fraction": float(jnp.mean(aux64 != 0.0)),
    }


def cell_populations(phi, operator):
    """Stratify cells by phase value and by cut/full fraction (spec 27)."""
    volume = np.asarray(operator.volume_safe, np.float64)
    cell = np.asarray(operator.aperture_x, np.float64) * np.asarray(operator.aperture_y, np.float64)
    dx_dy = (6.0 / phi.shape[0]) * (6.0 / phi.shape[1])
    values = np.asarray(phi, np.float64)
    abs_phi = np.abs(values)
    return {
        "gas": abs_phi < 0.01,
        "near_gas": (abs_phi >= 0.01) & (abs_phi < 0.1),
        "interface": (abs_phi >= 0.1) & (abs_phi <= 0.9),
        "near_liquid": (abs_phi > 0.9) & (abs_phi <= 0.99),
        "liquid": abs_phi > 0.99,
        "cut": cell < 0.999 * dx_dy,
        "full": cell >= 0.999 * dx_dy,
        "fluid": volume > 0.0,
    }


def population_table(weights, population, volume, e_round):
    """``E_round``-normalised mass of a local quantity restricted to a cell population."""
    mask = np.asarray(population, bool)
    if not mask.any():
        return {"cells": 0, "mass_over_E_round": 0.0}
    w = np.asarray(weights, np.float64)[mask]
    v = np.asarray(volume, np.float64)[mask]
    return {
        "cells": int(mask.sum()),
        "mass_over_E_round": float(np.sum(v * w) / e_round),
    }


# ------------------------------------------------------------------ fixed-state causal test
def fixed_state_causal(
    N: int = 128,
    *,
    target_deg: float = 150.0,
    M_factor: float = 4.0,
    warmup: int = 2000,
    rtol: float = 1.0e-6,
):
    """Apply only the candidate storage rule to *identical* production inputs (spec 26/27).

    At one late-time state, every candidate assembles its own update from the same host state -- the
    operators see the same fields, so the only thing that differs is the dtype of the update and how
    it is stored. Two numbers are reported per candidate, and they are deliberately not mixed:

    ``storage_loss_over_E_round``
        ``sum_i V_i (stored_i - (rhs_i + x_i))`` evaluated in float64 -- the mass the *storage rule*
        throws away on this single update, in units of the mass-reduction rounding ``E_round``;
    ``input_shift_over_E_round``
        the same sum applied to the difference between the candidate's *exact* update and the
        float32 baseline's exact update. This is what the candidate changed *before* storage (the
        float64 right-hand side of A1/A2, the mixed Krylov of A2). It is reported separately because
        it is a change of the equation, not a rounding of the stored field.

    The population decomposition (spec 27) is reported for the baseline's storage residual, i.e. for
    the field the production operators actually read.
    """
    p, solid, state, operator, _m0, e_round = build_case(
        N, target_deg=target_deg, M_factor=M_factor, rtol=rtol, candidate="A0_float32"
    )
    volume = volume_of(operator)
    dt = p.dt / 3.0

    def warm(carry, _):
        out, _info = storage_substep(carry, solid, p, operator, dt, "A0_float32")
        return out, None

    if warmup:
        state, _ = jax.lax.scan(warm, state, None, length=int(warmup))

    populations = cell_populations(state["phi"], operator)
    working_ulp = {
        "float32": float(np.spacing(np.float32(1.0))),
        "float64": float(np.spacing(np.float64(1.0))),
    }
    rows: dict[str, Any] = {}
    base_exact = None
    production_residual = None
    for candidate in CANDIDATES:
        rhs, correction, converged, iterations = exchange_increment(
            state, solid, p, operator, dt, candidate
        )
        stored, aux_new, _residual = apply_storage_rule(
            rhs, correction, converged, state["aux"], candidate
        )
        exact = rhs.astype(jnp.float64) + correction.astype(jnp.float64)
        loss = np.asarray(stored.astype(jnp.float64) - exact, np.float64)
        if base_exact is None:
            base_exact = exact
        shift = np.asarray(exact - base_exact, np.float64)
        scale = float(np.max(np.abs(np.asarray(rhs, np.float64))))
        solve_defect = np.asarray(
            jnp.asarray(volume) * correction.astype(jnp.float64), np.float64
        )
        record = {
            "working_dtype": str(rhs.dtype),
            "krylov_dtype": str(correction.dtype),
            "solve_mass_defect_over_E_round": float(solve_defect.sum() / e_round),
            "solve_mass_defect_abs_total_over_E_round": float(np.abs(solve_defect).sum() / e_round),
            "cg_iterations": int(iterations),
            "cg_converged": bool(converged),
            "rhs_max_abs": scale,
            "storage_loss_over_E_round": float(np.sum(np.asarray(volume) * loss) / e_round),
            "storage_loss_abs_total_over_E_round": float(np.sum(np.abs(np.asarray(volume) * loss)) / e_round),
            "storage_loss_nonzero_fraction": float(np.mean(loss != 0.0)),
            "storage_loss_max_abs": float(np.max(np.abs(loss))),
            "storage_loss_mean_ulps_of_working_dtype": float(
                np.mean(np.abs(loss)) / (working_ulp[str(rhs.dtype)] * max(abs(scale), 1e-30))
            ),
            "input_shift_over_E_round": float(np.sum(np.asarray(volume) * shift) / e_round),
            "input_shift_max_abs": float(np.max(np.abs(shift))),
            "populations": {
                name: population_table(loss, mask, volume, e_round)
                for name, mask in populations.items()
            },
        }
        if candidate == "A0_float32":
            production_residual = record
        rows[candidate] = record

    return {
        "N": int(N),
        "target_deg": float(target_deg),
        "M_factor": float(M_factor),
        "warmup_substeps": int(warmup),
        "E_round": float(e_round),
        #: the incumbent's storage rounding, decomposed by cell population (spec 27)
        "production_storage_residual": {
            "mass_over_E_round": production_residual["storage_loss_over_E_round"],
            "nonzero_fraction": production_residual["storage_loss_nonzero_fraction"],
            "population_decomposition_over_E_round": {
                name: entry["mass_over_E_round"]
                for name, entry in production_residual["populations"].items()
            },
        },
        "candidates": rows,
        "note": "storage_loss is measured against the float64 evaluation of the candidate's own two "
        "terms; input_shift is the difference of that exact update from the float32 baseline's, so a "
        "candidate cannot hide a change of the equation inside the storage metric.",
    }


def population_series(N: int = 128, *, target_deg: float = 150.0, M_factor: float = 4.0, warmup: int = 600):
    """The per-population *first-loss* ledger of the baseline over consecutive updates (spec 27).

    One fixed set of cell populations is taken from the warmed-up state; the storage residual of
    ``warmup`` further baseline updates is then accumulated per population, so the question
    "where does the mass actually go on the float32 grid" is answered on the same cells throughout.
    """
    p, solid, state, operator, _m0, e_round = build_case(
        N, target_deg=target_deg, M_factor=M_factor, candidate="A0_float32"
    )
    volume = volume_of(operator)
    dt = p.dt / 3.0

    def warm(carry, _):
        out, _info = storage_substep(carry, solid, p, operator, dt, "A0_float32")
        return out, None

    if warmup:
        state, _ = jax.lax.scan(warm, state, None, length=int(warmup))
    populations = cell_populations(state["phi"], operator)

    def body(carry, _):
        rhs, correction, converged, iterations, _u, _v = production_fields(
            carry, solid, p, operator, dt, "A0_float32"
        )
        stored, aux_new, _residual = apply_storage_rule(
            rhs, correction, converged, carry["aux"], "A0_float32"
        )
        exact = rhs.astype(jnp.float64) + correction.astype(jnp.float64)
        loss = jnp.asarray(volume) * (stored.astype(jnp.float64) - exact)
        momentum_damp = 1.0 / (1.0 + dt * solid.chi / p.eta_pen)
        u_new = (carry["u"] + dt * _u) * momentum_damp
        v_new = (carry["v"] + dt * _v) * momentum_damp
        divergence = pf._ddx(u_new, p.dx) + pf._ddy(v_new, p.dy)
        pressure = pf.poisson_solve(divergence / dt, p.m2_proj)
        out = {
            "phi": stored,
            "u": u_new - dt * pf._ddx(pressure, p.dx),
            "v": v_new - dt * pf._ddy(pressure, p.dy),
            "t": carry["t"] + jnp.asarray(dt, carry["t"].dtype),
            "aux": aux_new,
        }
        return out, loss

    _final, losses = jax.lax.scan(body, state, None, length=int(warmup))
    accumulated = np.asarray(jnp.sum(losses, axis=0), np.float64)
    return {
        "N": int(N),
        "target_deg": float(target_deg),
        "updates": int(warmup),
        "E_round": float(e_round),
        "total_over_E_round": float(accumulated.sum() / e_round),
        "per_cell_total_max_abs": float(np.max(np.abs(accumulated))),
        "populations": {
            name: {
                "cells": int(np.asarray(mask, bool).sum()),
                "total_over_E_round": float(accumulated[np.asarray(mask, bool)].sum() / e_round),
                "mean_per_cell_over_E_round": float(
                    accumulated[np.asarray(mask, bool)].mean() / e_round
                )
                if np.asarray(mask, bool).any()
                else 0.0,
            }
            for name, mask in populations.items()
        },
        "cell_fraction": {name: float(np.asarray(mask, bool).mean()) for name, mask in populations.items()},
        "note": "accumulated storage residual (V_i times the float32 rounding of one update) over "
        "consecutive baseline updates at one fixed population assignment",
    }


# ------------------------------------------------------------------ drift matrix
def drift_series(
    N: int = 48,
    *,
    target_deg: float = 150.0,
    M_factor: float = 4.0,
    public_steps: int = 2500,
    sample_every: int = 50,
    substeps_per_step: int = 3,
    candidate: str = "A0_float32",
    rtol: float = 1.0e-6,
    wall_offset_over_dy: float = 0.0,
):
    """Sampled physical-mass series of one candidate over ``public_steps`` public steps.

    One public step is ``substeps_per_step`` phase substeps of ``p.dt / substeps_per_step`` -- the
    production stepping -- and ``sample_every`` counts *public* steps, so the horizons are directly
    comparable with the L1A-2h closure ledger (whose "2500 steps" was 2500 public steps = 7500
    substeps). The mass is always the field the *operators see*; the compensated bookkeeping sum and
    the auxiliary field's magnitude are sampled alongside it and labelled as bookkeeping.
    """
    p, solid, state, operator, m0, e_round = build_case(
        N,
        target_deg=target_deg,
        M_factor=M_factor,
        rtol=rtol,
        candidate=candidate,
        wall_offset_over_dy=wall_offset_over_dy,
    )
    volume = volume_of(operator)
    dt = p.dt / float(substeps_per_step)
    total = int(public_steps) * int(substeps_per_step)
    every = int(sample_every) * int(substeps_per_step)
    samples = (total - 1) // every + 1

    def substep(carry, index):
        inner = {key: carry[key] for key in ("phi", "u", "v", "t", "aux")}
        out, _info = storage_substep(inner, solid, p, operator, dt, candidate, diagnostics=True)
        solve_total = carry["solve_total"] + out.pop("solve_defect")
        storage_total = carry["storage_total"] + out.pop("storage_loss")
        out["solve_total"] = solve_total
        out["storage_total"] = storage_total
        sampled = index % every == 0
        slot = index // every
        out["masses"] = jnp.where(
            sampled, carry["masses"].at[slot].set(physical_mass(out, volume, candidate)), carry["masses"]
        )
        out["bookkeeping"] = jnp.where(
            sampled,
            carry["bookkeeping"].at[slot].set(bookkeeping_mass(out, volume, candidate)),
            carry["bookkeeping"],
        )
        out["aux_max"] = jnp.where(
            sampled, carry["aux_max"].at[slot].set(jnp.max(jnp.abs(out["aux"]))), carry["aux_max"]
        )
        return out, None

    carry = dict(state)
    carry["masses"] = jnp.zeros(samples, jnp.float64)
    carry["bookkeeping"] = jnp.zeros(samples, jnp.float64)
    carry["aux_max"] = jnp.zeros(samples, jnp.float64)
    carry["solve_total"] = jnp.asarray(0.0, jnp.float64)
    carry["storage_total"] = jnp.asarray(0.0, jnp.float64)
    final, _ = jax.jit(lambda c: jax.lax.scan(substep, c, jnp.arange(int(total))))(carry)
    masses = np.array(np.asarray(final["masses"], np.float64), copy=True)
    book = np.array(np.asarray(final["bookkeeping"], np.float64), copy=True)
    masses[0] = m0
    book[0] = m0
    return {
        "candidate": candidate,
        "public_steps": int(public_steps),
        "substeps": int(total),
        "sample_every": int(sample_every),
        "E_round": float(e_round),
        "m0": float(m0),
        "masses": masses.tolist(),
        "bookkeeping_masses": book.tolist(),
        "aux_max_abs": np.asarray(final["aux_max"], np.float64).tolist(),
        "final_phi_max": float(jnp.max(jnp.abs(final["phi"]))),
        "angle_deg": float(pf.measure_contact_angle(np.asarray(final["phi"]), solid, p)),
        "ledger": {
            "solve_defect_mass": float(final["solve_total"]),
            "solve_defect_over_E_round": float(final["solve_total"] / e_round),
            "storage_loss_mass": float(final["storage_total"]),
            "storage_loss_over_E_round": float(final["storage_total"] / e_round),
            "explained_relative_drift": float((final["solve_total"] + final["storage_total"]) / m0),
        },
        "final": final,
    }


def series_classification(masses, e_round: float, sample_every: int) -> dict[str, Any]:
    """L1A-2h's classifier on a sampled mass series (kept *bit-identical* on purpose).

    ``numpy.polyfit`` warns about the conditioning of its x axis (the step counter, up to 1e5)
    whenever the normalised mass series is small; the fitted coefficients are the ones L1A-2h used, so
    only that specific ``RankWarning`` is silenced here and the fact is recorded in the report notes.
    """
    import warnings

    values = np.asarray(masses, np.float64)
    if values.size < 3:
        return {
            "steps": int((values.size - 1) * sample_every),
            "sample_every": int(sample_every),
            "final_relative_drift": float((values[-1] - values[0]) / values[0]),
            "slope_over_E_round_per_step": None,
            "slope_t_statistic": None,
            "increment_autocorrelation": None,
            "random_walk_ratio": None,
            "verdict": "INSUFFICIENT_SAMPLES",
        }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.RankWarning)
        return kra.series_classification(values, e_round, sample_every)


def drift_matrix(profile: str) -> dict[str, Any]:
    """Quick and medium drift for every candidate (spec 34/35)."""
    quick = profile == "quick"
    rows: dict[str, Any] = {}
    fixtures = [
        ("quick_offset0.0", dict(N=48, target_deg=150.0, M_factor=4.0, steps=2500, sample_every=50, offset=0.0)),
        ("quick_offset0.5", dict(N=48, target_deg=150.0, M_factor=4.0, steps=2500, sample_every=50, offset=0.5)),
    ]
    if not quick:
        fixtures.append(
            ("medium_150deg", dict(N=128, target_deg=150.0, M_factor=4.0, steps=5000, sample_every=100, offset=0.0))
        )
        fixtures.append(
            ("medium_60deg", dict(N=128, target_deg=60.0, M_factor=4.0, steps=5000, sample_every=100, offset=0.0))
        )
    for candidate in CANDIDATES:
        per_candidate: dict[str, Any] = {}
        for label, spec in fixtures:
            payload = drift_series(
                spec["N"],
                target_deg=spec["target_deg"],
                M_factor=spec["M_factor"],
                public_steps=spec["steps"],
                sample_every=spec["sample_every"],
                candidate=candidate,
                wall_offset_over_dy=spec["offset"],
            )
            stats = series_classification(payload["masses"], payload["E_round"], payload["sample_every"])
            stats["bookkeeping_final_relative_drift"] = float(
                (payload["bookkeeping_masses"][-1] - payload["m0"]) / payload["m0"]
            )
            stats["aux_max_abs_final"] = float(payload["aux_max_abs"][-1])
            per_candidate[label] = stats
        rows[candidate] = per_candidate
    return rows


# ------------------------------------------------------------------ restart equivalence (spec 24)
def restart_equivalence(
    N: int = 48,
    *,
    target_deg: float = 150.0,
    M_factor: float = 4.0,
    first: int = 200,
    second: int = 200,
    candidate: str = "A0_float32",
):
    """Interrupted vs uninterrupted continuation, and the cost of dropping hidden state."""
    p, solid, state, operator, m0, e_round = build_case(
        N, target_deg=target_deg, M_factor=M_factor, candidate=candidate
    )
    volume = volume_of(operator)
    dt = p.dt / 3.0

    def advance(state_in, steps):
        def body(carry, _):
            out, _info = storage_substep(carry, solid, p, operator, dt, candidate)
            return out, physical_mass(out, volume, candidate)

        return jax.lax.scan(body, state_in, None, length=int(steps))

    run_first = jax.jit(lambda s: advance(s, first))
    run_second = jax.jit(lambda s: advance(s, second))
    checkpoint, masses_a = run_first(state)
    continued, masses_b = run_second(checkpoint)
    uninterrupted, masses_ab = jax.jit(lambda s: advance(s, first + second))(state)

    def diff(a, b):
        return float(jnp.max(jnp.abs(a.astype(jnp.float64) - b.astype(jnp.float64))))

    formal = {
        "phi_max_abs_difference": diff(continued["phi"], uninterrupted["phi"]),
        "u_max_abs_difference": diff(continued["u"], uninterrupted["u"]),
        "v_max_abs_difference": diff(continued["v"], uninterrupted["v"]),
        "aux_max_abs_difference": diff(continued["aux"], uninterrupted["aux"]),
        "mass_drift_interrupted": float((np.asarray(masses_b)[-1] - m0) / m0),
        "mass_drift_uninterrupted": float((np.asarray(masses_ab)[-1] - m0) / m0),
    }
    # a real checkpoint round trip through np.savez: shape and dtype of every persistent array must
    # survive, including the candidate's hidden state (spec 23/24 -- no silent loss on restart).
    import io

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{key: np.asarray(checkpoint[key]) for key in ("phi", "u", "v", "t", "aux")})
    buffer.seek(0)
    with np.load(buffer) as loaded:
        restored = {
            key: jnp.asarray(loaded[key], dtype=checkpoint[key].dtype)
            for key in ("phi", "u", "v", "t", "aux")
        }
    round_trip, _m = run_second(restored)
    formal["disk_round_trip_phi_max_abs_difference"] = diff(round_trip["phi"], uninterrupted["phi"])
    formal["disk_round_trip_dtypes"] = {key: str(np.asarray(restored[key]).dtype) for key in restored}
    dropped = None
    if candidate in HIDDEN_STATE_CANDIDATES:
        zeroed = dict(checkpoint)
        zeroed["aux"] = jnp.zeros_like(checkpoint["aux"])
        dropped_state, _m = run_second(zeroed)
        dropped = {
            "phi_max_abs_difference_vs_uninterrupted": diff(dropped_state["phi"], uninterrupted["phi"]),
            "note": "restarting with aux = 0 -- NOT an allowed formal path for a candidate whose "
            "hidden state influences the trajectory",
        }
    return {"candidate": candidate, "formal_restart": formal, "aux_dropped_restart": dropped}


# ------------------------------------------------------------------ AD (spec 44)
def gradient_check(
    N: int = 24,
    *,
    target_deg: float = 150.0,
    M_factor: float = 4.0,
    steps: int = 3,
    candidate: str = "A0_float32",
):
    """AD audit (spec 44) of the storage rule, separated from a pre-existing solver property.

    The contract-10 implicit solve runs its CG recurrence in ``jax.lax.while_loop``, so reverse-mode
    differentiation through the *full* step is unavailable for **every** candidate including the
    shipped A0 -- that is a property of contract 10, not of a storage model, and the audit reports it
    rather than hiding it. What a storage model can change is

    * whether the local storage rule itself is differentiable (its ``jnp.where``/``atan2``-free
      TwoSum and residual arithmetic must be), and
    * whether the trajectory functional still matches a finite difference in forward mode.

    Both are measured with the exchange increment of each substep *frozen* at its value from the
    initial state (``frozen_correction``), which removes the CG from the differentiated path without
    changing the storage arithmetic.
    """
    p, solid, state, operator, _m0, _e_round = build_case(
        N, target_deg=target_deg, M_factor=M_factor, candidate=candidate
    )
    volume = volume_of(operator)
    dt = p.dt / 3.0
    frozen = _frozen_correction(state, solid, p, operator, candidate, dt)

    def functional(phi, aux):
        carry = dict(state)
        carry["phi"] = phi
        carry["aux"] = aux

        def body(c, _):
            out, _info = storage_substep(
                c, solid, p, operator, dt, candidate, frozen_correction=frozen
            )
            return out, None

        out, _ = jax.lax.scan(body, carry, None, length=int(steps))
        return physical_mass(out, volume, candidate)

    phi0 = state["phi"] + jnp.asarray(1.0e-4, state["phi"].dtype)
    aux0 = state["aux"]
    value = functional(phi0, aux0)
    direction = jnp.cos(jnp.arange(phi0.size).reshape(phi0.shape).astype(phi0.dtype))
    eps = 1.0e-3
    finite_difference = float(
        (functional(phi0 + eps * direction, aux0) - functional(phi0 - eps * direction, aux0)) / (2 * eps)
    )
    reverse = jax.grad(lambda phi: functional(phi, aux0))(phi0)
    reverse_directional = float(jnp.sum(reverse * direction))
    _value, tangent = jax.jvp(lambda phi: functional(phi, aux0), (phi0,), (direction,))
    forward_directional = float(tangent)

    def relative(a, b):
        return float(abs(a - b) / max(abs(b), 1e-30))

    full_step_gradient = {"available": None, "error": None}
    try:
        jax.grad(lambda phi: _full_step_functional(phi, state, solid, p, operator, dt, candidate, steps))(phi0)
        full_step_gradient["available"] = True
    except Exception as exc:  # pragma: no cover - depends on the shipped solver
        full_step_gradient["available"] = False
        full_step_gradient["error"] = f"{type(exc).__name__}: {exc}".split("\n")[0][:200]
        full_step_gradient["note"] = (
            "the shipped contract-10 CG uses lax.while_loop; this is identical for the float32 "
            "baseline and every candidate, so it is not a regression of any storage model"
        )
    return {
        "candidate": candidate,
        "frozen_correction_max_abs": float(jnp.max(jnp.abs(frozen))),
        "value": float(value),
        "storage_rule_reverse_mode_finite": bool(jnp.all(jnp.isfinite(reverse))),
        "storage_rule_gradient_abs_max": float(jnp.max(jnp.abs(reverse))),
        "reverse_directional": reverse_directional,
        "forward_directional": forward_directional,
        "finite_difference": finite_difference,
        "reverse_vs_finite_difference": relative(reverse_directional, finite_difference),
        "forward_vs_finite_difference": relative(forward_directional, finite_difference),
        "storage_rule_ad_ok": bool(jnp.all(jnp.isfinite(reverse)))
        and relative(forward_directional, finite_difference) < 0.05,
        "full_step_reverse_mode": full_step_gradient,
    }


def _frozen_correction(state, solid, p, operator, candidate, dt):
    """The exchange correction of the initial state, computed once (AD probe only)."""
    working = jnp.float64 if candidate in PHASE_FLOAT64_CANDIDATES else p.dtype
    nxt = to_state(state)
    phi_rhs, _u_rhs, _v_rhs, _mu, mu_expl = pf.rhs(nxt, solid, p)
    ch = -pf.control_volume_divergence(
        *pf.chemical_potential_fluxes(mu_expl, solid, p), operator.volume_safe
    )
    advective = kra.production_advective_rate(state["phi"], state["u"], state["v"], solid, p, dt, phi_rhs)
    rhs = state["phi"].astype(working) + dt * (advective.astype(working) + ch.astype(working))
    alpha = jnp.asarray(dt * float(p.M) * float(p.eps), working)
    if candidate in FLOAT32_KRYLOV_CANDIDATES and working == jnp.float64:
        rhs_solve, alpha_solve, rtol_solve = rhs.astype(p.dtype), alpha.astype(p.dtype), jnp.asarray(
            float(p.ch_solver_rtol), p.dtype
        )
    else:
        rhs_solve, alpha_solve, rtol_solve = rhs, alpha, jnp.asarray(float(p.ch_solver_rtol), working)
    correction, _converged, _iterations = production_correction(
        rhs_solve,
        operator.volume_safe,
        operator.weight_x,
        operator.weight_y,
        alpha_solve,
        rtol_solve,
        jnp.asarray(int(p.ch_solver_max_iterations), jnp.int32),
    )
    return jax.lax.stop_gradient(correction.astype(working))


def _full_step_functional(phi, state, solid, p, operator, dt, candidate, steps):
    carry = dict(state)
    carry["phi"] = phi

    def body(c, _):
        out, _info = storage_substep(c, solid, p, operator, dt, candidate)
        return out, None

    out, _ = jax.lax.scan(body, carry, None, length=int(steps))
    return physical_mass(out, volume_of(operator), candidate)


# ------------------------------------------------------------------ performance (spec 39/41/42)
def persistent_bytes(state, candidate):
    """Bytes of the persistent state arrays (the storage model's own cost).

    ``aux`` is counted only for the candidates whose hidden field is part of the persistent
    checkpoint; A1/A2/F carry an all-zero float32 ``aux`` inside this audit's dict carry for code
    uniformity, which is *not* part of their production state.
    """
    total = 0
    for key in ("phi", "u", "v", "aux"):
        value = state[key]
        if candidate not in HIDDEN_STATE_CANDIDATES and key == "aux":
            continue
        total += int(np.prod(value.shape)) * int(np.dtype(value.dtype).itemsize)
    return total


def performance(
    N: int = 128,
    *,
    target_deg: float = 150.0,
    M_factor: float = 4.0,
    steps: int = 40,
    repeats: int = 3,
    candidates: Sequence[str] = CANDIDATES,
):
    """Per-substep wall clock plus persistent memory, per candidate (CPU-only).

    ``repeats`` timed repetitions are taken after a warm compile and the **minimum** is reported
    (micro-benchmark convention), together with the spread across repeats: a single shot on a shared
    CPU sandbox is not repeatable enough to rank candidates whose measured ratio is near 1. The spread
    is part of the artifact so a ratio inside it cannot be over-read.
    """
    rows = []
    reference = None
    for candidate in candidates:
        p, solid, state, operator, _m0, _e_round = build_case(
            N, target_deg=target_deg, M_factor=M_factor, candidate=candidate
        )
        dt = p.dt / 3.0

        def substeps(state_in, n):
            def body(carry, _):
                out, _info = storage_substep(carry, solid, p, operator, dt, candidate)
                return out, jnp.asarray(0.0)

            final, _ys = jax.lax.scan(body, state_in, None, length=int(n))
            return final

        compiled = jax.jit(lambda s: substeps(s, steps))
        jax.block_until_ready(compiled(state)["phi"])
        samples = []
        for _ in range(max(int(repeats), 1)):
            started = time.perf_counter()
            jax.block_until_ready(compiled(state)["phi"])
            samples.append((time.perf_counter() - started) / max(int(steps), 1))
        per_substep = min(samples)
        if reference is None:
            reference = per_substep
        lowered = None
        try:
            lowered = jax.jit(lambda s: substeps(s, 1)).lower(state).compile().memory_analysis()
        except Exception:  # pragma: no cover - memory analysis is best effort
            lowered = None
        rows.append(
            {
                "candidate": candidate,
                "N": int(N),
                "substeps": int(steps),
                "seconds_per_substep": float(per_substep),
                "seconds_per_substep_samples": [float(value) for value in samples],
                "seconds_per_substep_spread": float(max(samples) - min(samples)),
                "runtime_ratio": float(per_substep / reference),
                "repeats": int(max(int(repeats), 1)),
                "persistent_bytes": int(persistent_bytes(state, candidate)),
                "memory_analysis": None
                if lowered is None
                else {
                    "argument_bytes": int(getattr(lowered, "argument_size_in_bytes", 0)),
                    "temporary_bytes": int(getattr(lowered, "temporary_size_in_bytes", 0)),
                    "alias_bytes": int(getattr(lowered, "alias_size_in_bytes", 0)),
                },
            }
        )
    return rows


def ledger_cross_check(report: dict[str, Any]) -> dict[str, Any]:
    """Read the frozen L1A-2h gate rows and the contract-10 closure ledger back from disk.

    Nothing here is re-derived: the L1A-2h quick gate is the row the incumbent failed, so reproducing
    it is the check that this audit's fixture and mass metric are the same ones (spec 34), and the
    50k CHNS closure rows are quoted as the "before" series the staged closure must improve on.
    """
    root = Path(__file__).resolve().parents[1] / "evidence"
    out: dict[str, Any] = {"quick_gate_reference": None, "quick_gate_reproduction": None, "closure_ledger": None}
    krylov = root / "l1a2h" / "krylov_roundoff_report.json"
    if krylov.exists():
        payload = json.loads(krylov.read_text())
        rows = (payload.get("numbers") or {}).get("quick_gate") or {}
        out["quick_gate_reference"] = {
            key: {
                "final_relative_drift": row.get("final_relative_drift"),
                "passed": row.get("passed"),
                "verdict": row.get("verdict"),
                "increment_autocorrelation": row.get("increment_autocorrelation"),
                "slope_t_statistic": row.get("slope_t_statistic"),
            }
            for key, row in rows.items()
        }
        mine = (report.get("quick_matrix") or {}).get("A0_float32") or {}
        pairs = {
            "offset_0.0": "quick_offset0.0",
            "offset_0.5": "quick_offset0.5",
        }
        differences = {}
        for reference_key, audit_key in pairs.items():
            reference_row = rows.get(reference_key)
            audit_row = mine.get(audit_key)
            if not reference_row or not audit_row:
                continue
            differences[reference_key] = {
                "ledger": reference_row.get("final_relative_drift"),
                "audit": audit_row.get("final_relative_drift"),
                "absolute_difference": abs(
                    float(reference_row.get("final_relative_drift") or 0.0)
                    - float(audit_row.get("final_relative_drift") or 0.0)
                ),
            }
        out["quick_gate_reproduction"] = {
            "rows": differences,
            "note": "the audit's A0 runs the full CHNS substep chain (momentum on) while the L1A-2h "
            "quick gate runs the phase-only substep; the two agree to ~3e-07 in relative drift, which "
            "is the advective contribution the momentum path adds. The fixture (N, angle, dt, M, "
            "wall offsets) and the mass metric are the same.",
        }
    closure = root / "mass_precision" / "closure.json"
    if closure.exists():
        payload = json.loads(closure.read_text())
        out["closure_ledger"] = {
            "source": str(closure.relative_to(root.parent)),
            "solver_contract_version": payload.get("solver_contract_version"),
            "rows": payload.get("closure") or payload.get("rows") or payload,
        }
    return out


# ------------------------------------------------------------------ gates
#: Acceptance gates of spec 34-36, 39-42, 44, 48, 57 (as (name, metric, bound, kind)).
QUICK_GATE = 2.0e-6
QUICK_GATE_STRONG = 5.0e-7
MEDIUM_GATE = 2.0e-4
MEDIUM_GATE_STRONG = 1.0e-4
ANGLE_DELTA_GATE = 0.2
LONG_HORIZON_PROJECTION_BOUND = 1.0e-3
RUNTIME_PREFERRED = 0.15
MEMORY_PREFERRED = 0.20
#: The horizon the long-run slope is projected to when asking "is the drift clean to the CHNS cap".
CLOSURE_HORIZON = 200_000

#: Profiles (spec 3): what each one runs.
PROFILES = {
    "quick": {
        "causal_N": 48,
        "causal_warmup": 300,
        "medium": False,
        "long_horizons": (),
        "long_candidates": (
            "A0_float32",
            "A1_phase_float64",
            "A2_float64_storage_f32_krylov",
            "B1_compensated",
            "C1_residual_feedback",
        ),
        "restart_candidates": ("A0_float32", "A1_phase_float64", "A2_float64_storage_f32_krylov", "B1_compensated"),
        "performance_N": 48,
        "performance_steps": 20,
        "gradient_N": 24,
    },
    "baseline": {
        "causal_N": 128,
        "causal_warmup": 600,
        "medium": True,
        "long_horizons": (5_000, 10_000, 25_000, 50_000),
        "long_candidates": (
            "A0_float32",
            "A1_phase_float64",
            "A2_float64_storage_f32_krylov",
            "B1_compensated",
            "C1_residual_feedback",
        ),
        "restart_candidates": (
            "A0_float32",
            "A1_phase_float64",
            "A2_float64_storage_f32_krylov",
            "B1_compensated",
            "C1_residual_feedback",
        ),
        "performance_N": 128,
        "performance_steps": 30,
        "gradient_N": 32,
    },
    "forensic": {
        "causal_N": 128,
        "causal_warmup": 2000,
        "medium": True,
        "long_horizons": (5_000, 10_000, 25_000, 50_000, 100_000),
        "long_candidates": CANDIDATES,
        "restart_candidates": CANDIDATES,
        "performance_N": 128,
        "performance_steps": 50,
        "gradient_N": 32,
    },
}

#: How invasive each candidate is in production terms (spec 48), from the smallest change up.
INVASIVENESS = {
    "A1_phase_float64": {
        "rank": 2,
        "production_files": ["phasefield.py"],
        "new_state_fields": [],
        "persistent_state_dtype_only": True,
        "notes": "phi (and its update path) stored in float64; pf.State layout, restarts and datasets "
        "keep their shape and their field names.",
    },
    "A2_float64_storage_f32_krylov": {
        "rank": 1,
        "production_files": ["phasefield.py"],
        "new_state_fields": [],
        "persistent_state_dtype_only": True,
        "notes": "identical persistent layout to A1 (phi in float64) but the Krylov recurrence stays "
        "float32: the exchange correction is mass-neutral by the telescoping of the cut-cell "
        "divergence, so the state is never rounded back onto the float32 grid. Chosen over A1 only if "
        "it clears the same gates -- the measurement decides.",
    },
    "B1_compensated": {
        "rank": 3,
        "production_files": ["phasefield.py", "generate_dataset.py", "restart/checkpoint schema"],
        "new_state_fields": ["phase_lo (compensation pair member)"],
        "persistent_state_dtype_only": False,
        "notes": "second persistent phase array; operators must read phi_hi; checkpoints and datasets "
        "must serialize the pair and the restart version must reject old data.",
    },
    "C1_residual_feedback": {
        "rank": 4,
        "production_files": ["phasefield.py", "generate_dataset.py", "restart/checkpoint schema"],
        "new_state_fields": ["phase_residual (local feedback memory)"],
        "persistent_state_dtype_only": False,
        "notes": "one extra persistent array AND a changed local update rule; restart state version "
        "must be bumped or a resumed run silently loses the memory.",
    },
    "A0_float32": {
        "rank": 0,
        "production_files": [],
        "new_state_fields": [],
        "persistent_state_dtype_only": False,
        "notes": "the incumbent; kept as the reference row, not a candidate for selection.",
    },
    "F_full_float64": {
        "rank": 5,
        "production_files": ["phasefield.py (global promotion)"],
        "new_state_fields": [],
        "persistent_state_dtype_only": False,
        "notes": "global promotion is forbidden without cost evidence (spec 76) and is used here only "
        "as the float64 reference ceiling.",
    },
}


def _gate(value, bound, *, available=True, strong_bound=None):
    if not available or value is None or not math.isfinite(float(value)):
        return {"measured": None, "bound": bound, "passed": None, "status": "NOT_MEASURED"}
    passed = abs(float(value)) <= bound
    record = {"measured": float(value), "bound": float(bound), "passed": bool(passed)}
    if strong_bound is not None:
        record["strong_bound"] = float(strong_bound)
        record["strong_passed"] = bool(abs(float(value)) <= strong_bound)
    return record


# ------------------------------------------------------------------ dataset lineage (spec 52)
def dataset_lineage(candidate: str) -> dict[str, Any]:
    """What a candidate would do to the dataset/restart lineage, read from the shipped sources."""
    import ast

    root = Path(__file__).resolve().parent.parent
    generator = root / "generate_dataset.py"
    payload_keys: list[str] = []
    schema_version = None
    current_check = ""
    if generator.exists():
        tree = ast.parse(generator.read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "DATASET_SCHEMA_VERSION" for t in node.targets
            ):
                schema_version = int(node.value.value)
            if isinstance(node, ast.FunctionDef) and node.name == "_dataset_fingerprint":
                for sub in ast.walk(node):
                    if isinstance(sub, ast.keyword) and sub.arg:
                        payload_keys.append(sub.arg)
            if isinstance(node, ast.FunctionDef) and node.name == "_saved_case_is_current":
                current_check = ast.get_docstring(node) or ""
    hidden = candidate in HIDDEN_STATE_CANDIDATES
    return {
        "candidate": candidate,
        "requires_new_persistent_field": hidden,
        "dataset_schema_version_now": schema_version,
        "fingerprint_payload_keys": sorted(set(payload_keys)),
        "fingerprint_has_solver_sha256": any(k.endswith("sha256") for k in payload_keys),
        "fingerprint_has_solver_contract": any("contract" in k for k in payload_keys),
        "fingerprint_has_phase_storage_model": any("phase_storage" in k for k in payload_keys),
        "stale_dataset_guard": current_check.strip(),
        "decision": (
            "storage model must be added to the trajectory fingerprint and to "
            "phase_transport_metadata; the restart state version must be bumped "
            "(old checkpoints cannot be resumed)"
            if hidden
            else "no new dataset field; the solver source hash and the contract-11 metadata already "
            "invalidate every contract-10 dataset, and the checkpoint layout is unchanged"
        ),
        "silent_resume_possible": False if hidden else False,
        "operator_read_semantics": {
            "A0_float32": "operators read phi (float32)",
            "A1_phase_float64": "operators read phi (float64)",
            "A2_float64_storage_f32_krylov": "operators read phi (float64); only the exchange "
            "correction comes from the float32 Krylov recurrence and it is added in float64",
            "B1_compensated": "operators read phi_hi only; phi_lo never enters a flux, a chemical "
            "potential, a divergence or the solve",
            "C1_residual_feedback": "operators read phi only; the residual memory enters the next "
            "local update and nothing else",
            "F_full_float64": "operators read phi (float64)",
        }[candidate],
    }


# ------------------------------------------------------------------ long horizon (spec 36)
def long_horizon(profile: str) -> dict[str, Any]:
    spec = PROFILES[profile]
    rows: dict[str, Any] = {}
    for candidate in spec["long_candidates"]:
        per_candidate = []
        for horizon in spec["long_horizons"]:
            payload = drift_series(
                48, target_deg=150.0, public_steps=int(horizon), sample_every=max(int(horizon) // 60, 1),
                candidate=candidate,
            )
            stats = series_classification(payload["masses"], payload["E_round"], payload["sample_every"])
            slope_rel_per_step = float(stats["slope_over_E_round_per_step"] or 0.0) * payload["E_round"]
            stats["projected_relative_drift_at_200k"] = abs(slope_rel_per_step) * CLOSURE_HORIZON
            stats["projected_clean"] = abs(slope_rel_per_step) * CLOSURE_HORIZON < LONG_HORIZON_PROJECTION_BOUND
            stats["bookkeeping_relative_drift"] = float(
                (payload["bookkeeping_masses"][-1] - payload["m0"]) / payload["m0"]
            )
            stats["aux_max_abs"] = float(payload["aux_max_abs"][-1])
            stats["_masses"] = payload["masses"]
            per_candidate.append({"horizon": int(horizon), **stats})
        rows[candidate] = per_candidate
    return rows


def fit_summary(per_candidate: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Least-squares fit of |final drift| against horizon: the linear/sqrt-N classification."""
    horizons = np.asarray([row["horizon"] for row in per_candidate], np.float64)
    drift = np.asarray([abs(row["final_relative_drift"]) for row in per_candidate], np.float64)
    if len(horizons) < 2:
        return {"slope_per_step": None, "slope_t": None, "model": "INSUFFICIENT_HORIZONS"}
    slope, intercept = np.polyfit(horizons, drift, 1)
    residual = drift - (slope * horizons + intercept)
    dof = max(len(horizons) - 2, 1)
    error = float(np.sqrt((residual**2).sum() / dof / ((horizons - horizons.mean()) ** 2).sum()))
    linear = error == 0.0 or abs(slope / error) > 3.0
    sqrt_fit = np.polyfit(np.sqrt(horizons), drift, 1)
    sqrt_residual = drift - (sqrt_fit[0] * np.sqrt(horizons) + sqrt_fit[1])
    return {
        "model": "LINEAR" if linear else "SUB_LINEAR_OR_RANDOM_WALK",
        "slope_per_step": float(slope),
        "slope_t": float(slope / error) if error > 0 else None,
        "linear_residual_rms": float(np.sqrt(np.mean(residual**2))),
        "sqrt_fit_residual_rms": float(np.sqrt(np.mean(sqrt_residual**2))),
    }


# ------------------------------------------------------------------ candidate rows (spec 66)
def candidate_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    quick = report.get("quick_matrix") or {}
    medium = report.get("medium_matrix") or {}
    causal = (report.get("fixed_state_causal") or {}).get("candidates") or {}
    long_run = report.get("long_horizon") or {}
    restart = report.get("restart_equivalence") or {}
    gradients = report.get("gradients") or {}
    perf = {row["candidate"]: row for row in (report.get("performance") or [])}
    a0_medium = (medium.get("A0_float32") or {}).get("medium_150deg") or {}
    a0_angle = a0_medium.get("angle_deg")

    rows = []
    for candidate in CANDIDATES:
        quick_rows = quick.get(candidate) or {}
        quick_values = {key: row.get("final_relative_drift") for key, row in quick_rows.items()}
        quick_ok = all(
            row.get("passed") for row in quick_rows.values()
        ) and bool(quick_rows.get("quick_offset0.0", {}).get("in_band", True))
        quick_ledger = {
            key: row.get("ledger") for key, row in quick_rows.items()
        }
        medium_row = (medium.get(candidate) or {}).get("medium_150deg") or {}
        medium_value = medium_row.get("final_relative_drift")
        angle_delta = None
        if a0_angle is not None and medium_row.get("angle_deg") is not None:
            angle_delta = abs(float(medium_row["angle_deg"]) - float(a0_angle))
        long_rows = long_run.get(candidate) or []
        long_ok = bool(long_rows) and all(
            row["verdict"] != "SYSTEMATIC_BIAS_DOMINANT" and row["projected_clean"] for row in long_rows
        )
        restart_row = restart.get(candidate) or {}
        formal = restart_row.get("formal_restart") or {}
        restart_ok = all(
            formal.get(key) == 0.0
            for key in (
                "phi_max_abs_difference",
                "u_max_abs_difference",
                "v_max_abs_difference",
                "aux_max_abs_difference",
            )
        ) if formal else None
        grad = gradients.get(candidate) or {}
        ad_ok = None if not grad else bool(grad.get("storage_rule_ad_ok"))
        performance_row = perf.get(candidate) or {}
        runtime_ratio = performance_row.get("runtime_ratio")
        memory_ratio = performance_row.get("persistent_bytes_ratio")

        reasons: list[str] = []
        if quick_ok is False:
            reasons.append("MASS_GATE_FAIL")
        if medium_value is not None and abs(float(medium_value)) > MEDIUM_GATE:
            reasons.append("MASS_GATE_FAIL")
        if angle_delta is not None and angle_delta > ANGLE_DELTA_GATE:
            reasons.append("PHYSICS_DRIFT")
        if long_rows and not long_ok:
            reasons.append("LINEAR_BIAS_REMAINS")
        if restart_ok is False:
            reasons.append("RESTART_NOT_REPRODUCIBLE")
        if ad_ok is False:
            reasons.append("AD_FAILURE")
        cost_over_budget = (
            None
            if runtime_ratio is None or memory_ratio is None
            else bool(runtime_ratio - 1.0 > RUNTIME_PREFERRED or memory_ratio - 1.0 > MEMORY_PREFERRED)
        )
        if (report.get("dataset_lineage") or {}).get(candidate, {}).get("requires_new_persistent_field"):
            reasons.append("DATASET_STATE_AMBIGUOUS")
        rows.append(
            {
                "candidate": candidate,
                "label": CANDIDATE_LABELS[candidate],
                "storage_model": PHASE_STORAGE_MODEL_OF[candidate],
                "phase_dtype": "float64" if candidate in PHASE_FLOAT64_CANDIDATES else "float32",
                "hidden_state": candidate in HIDDEN_STATE_CANDIDATES,
                "hidden_state_serialized": candidate in HIDDEN_STATE_CANDIDATES,
                "quick": quick_values,
                "quick_ledger": quick_ledger,
                "causal": {
                    "solve_mass_defect_over_E_round": (causal.get(candidate) or {}).get(
                        "solve_mass_defect_over_E_round"
                    ),
                    "storage_loss_over_E_round": (causal.get(candidate) or {}).get(
                        "storage_loss_over_E_round"
                    ),
                    "input_shift_over_E_round": (causal.get(candidate) or {}).get(
                        "input_shift_over_E_round"
                    ),
                    "working_dtype": (causal.get(candidate) or {}).get("working_dtype"),
                    "krylov_dtype": (causal.get(candidate) or {}).get("krylov_dtype"),
                },
                "quick_passed": quick_ok if quick_values else None,
                "medium_final_relative_drift": medium_value,
                "medium_passed": None
                if medium_value is None
                else abs(float(medium_value)) <= MEDIUM_GATE,
                "angle_delta_deg": angle_delta,
                "long_horizon": [
                    {
                        "horizon": row["horizon"],
                        "final_relative_drift": row["final_relative_drift"],
                        "verdict": row["verdict"],
                        "slope_t_statistic": row["slope_t_statistic"],
                        "autocorrelation": row["increment_autocorrelation"],
                        "random_walk_ratio": row["random_walk_ratio"],
                        "projected_relative_drift_at_200k": row["projected_relative_drift_at_200k"],
                        "projected_clean": row["projected_clean"],
                    }
                    for row in long_rows
                ],
                "long_horizon_clean": long_ok if long_rows else None,
                "restart_reproducible": restart_ok,
                "ad_ok": ad_ok,
                "runtime_ratio_vs_A0": runtime_ratio,
                "persistent_bytes_ratio_vs_A0": memory_ratio,
                "cost_within_preferred_budget": None if cost_over_budget is None else not cost_over_budget,
                "cost_note": (
                    None
                    if cost_over_budget is None
                    else "above the preferred envelope (spec 39-42): reported, not silently accepted"
                    if cost_over_budget
                    else "within the preferred envelope on the measured (CPU) device"
                ),
                "dataset_impact": "new persistent field; restart version bump required"
                if candidate in HIDDEN_STATE_CANDIDATES
                else "no new field",
                "invasiveness": INVASIVENESS[candidate],
                "rejection_reasons": sorted(set(reasons)),
                "status": "REFERENCE" if candidate in ("A0_float32", "F_full_float64") else "CANDIDATE",
            }
        )
    return rows


def select_candidate(rows: Sequence[dict[str, Any]], report: dict[str, Any]) -> dict[str, Any]:
    """The least invasive candidate that clears every *measured* gate (spec 48).

    Every candidate is labelled: ``REJECTED`` when a hard gate failed (with the machine-readable
    reasons), ``SELECTED`` for the least invasive survivor, ``ALTERNATIVE`` for a survivor that is
    more invasive than the chosen one, and ``CANDIDATE`` while a gate is still unmeasured -- a profile
    that has not measured the medium or long-horizon gates must never look like a decision.
    """
    order = sorted(
        (row for row in rows if row["status"] == "CANDIDATE"),
        key=lambda row: INVASIVENESS[row["candidate"]]["rank"],
    )
    passing: list[dict[str, Any]] = []
    unmeasured_any: list[str] = []
    for row in order:
        gates = {
            "quick": row["quick_passed"],
            "medium": row["medium_passed"],
            "long_horizon": row["long_horizon_clean"],
            "restart": row["restart_reproducible"],
            "ad": row["ad_ok"],
        }
        hard = [name for name, value in gates.items() if value is False]
        if hard:
            row["status"] = "REJECTED"
            row["rejection_reasons"] = sorted(set(row["rejection_reasons"]))
            continue
        missing = [name for name, value in gates.items() if value is None]
        if missing:
            unmeasured_any.extend(missing)
            row["status"] = "CANDIDATE"
            row["rejection_reasons"] = sorted(set(row["rejection_reasons"] + ["INCONCLUSIVE"]))
            continue
        passing.append(row)
    if not passing:
        return {
            "selected": None,
            "reason": (
                "gates not yet measured: " + ", ".join(sorted(set(unmeasured_any)))
                if unmeasured_any
                else "no candidate cleared every measured gate"
            ),
            "rows_considered": [r["candidate"] for r in order],
        }
    chosen = passing[0]
    chosen["status"] = "SELECTED"
    for row in passing[1:]:
        row["status"] = "ALTERNATIVE"
        row["rejection_reasons"] = sorted(set(row["rejection_reasons"] + ["NOT_LEAST_INVASIVE"]))
    return {
        "selected": chosen["candidate"],
        "storage_model": chosen.get("storage_model"),
        "reason": "least invasive candidate clearing the quick, medium, long-horizon, restart and "
        "AD gates",
        "rows_considered": [r["candidate"] for r in order],
        "quick_relative_drift": chosen["quick"],
        "medium_relative_drift": chosen["medium_final_relative_drift"],
    }


# ------------------------------------------------------------------ reports
def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_reports(report: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Any] = {}

    def dump(name: str, payload: Any) -> None:
        path = out_dir / name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        written[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}

    matrix = {
        "stage": STAGE,
        "profile": report["profile"],
        "gates": report["gates"],
        "candidate_rows": report["candidate_rows"],
        "selection": report["selection"],
        "cost": report.get("cost"),
        "production_readiness": report.get("production_readiness"),
        "verdict": report["verdict"],
    }
    dump("candidate_matrix.json", matrix)
    dump("performance.json", report.get("performance"))
    dump("restart_equivalence.json", report.get("restart_equivalence"))
    long_series = report.pop("_long_series", None)
    if long_series:
        dump("long_run_series.json", long_series)

    serialisable = dict(report)
    serialisable.pop("_final_state", None)
    dump("phase_storage_precision_report.json", serialisable)
    (out_dir / "phase_storage_precision_report.md").write_text(render_markdown(report))
    written["phase_storage_precision_report.md"] = {
        "bytes": (out_dir / "phase_storage_precision_report.md").stat().st_size,
        "sha256": _sha256(out_dir / "phase_storage_precision_report.md"),
    }
    manifest = {
        "stage": STAGE,
        "module": MODULE,
        "profile": report["profile"],
        "spec": "TWO_PHASE_L1A2I_PHASE_STATE_STORAGE_PRECISION_AGENT_SPEC.md",
        "contract_version": report["contract_version"],
        "artifacts": written,
        "environment": report["environment"],
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verdict": report["verdict"],
    }
    dump("manifest.json", manifest)
    return written


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    add(f"# L1A-2i phase-state storage precision report ({report['profile']} profile)")
    add("")
    add(f"- stage: `{STAGE}` (spec `TWO_PHASE_L1A2I_PHASE_STATE_STORAGE_PRECISION_AGENT_SPEC.md`)")
    add(f"- contract version: `{report['contract_version']}` (unchanged by this audit)")
    add(f"- verdict: **{report['verdict']}**")
    add(f"- selection: `{report['selection'].get('selected')}` -- {report['selection'].get('reason')}")
    add(f"- production readiness: {report.get('production_readiness', {}).get('statement')}")
    cost = report.get("cost") or {}
    add(f"- cost: runtime ratio {cost.get('runtime_ratio_vs_A0')} (preferred +"
        f"{cost.get('runtime_preferred_fraction')}), memory ratio {cost.get('persistent_bytes_ratio_vs_A0')} "
        f"-- {cost.get('verdict')}")
    add("")
    add("## Gates")
    add("")
    add("| gate | value |")
    add("|---|---|")
    for key, value in report["gates"].items():
        add(f"| {key} | `{value}` |")
    add("")
    add("## Drift ledger (accumulated over the quick gate)")
    add("")
    add("| candidate | solve defect / E_round | storage loss / E_round | explained drift | measured drift |")
    add("|---|---|---|---|---|")
    for row in report["candidate_rows"]:
        ledger = (row.get("quick_ledger") or {}).get("quick_offset0.0") or {}
        if not ledger:
            continue
        measured = row["quick"].get("quick_offset0.0")
        measured_text = "n/a" if measured is None else f"{measured:+.3e}"
        add(f"| `{row['candidate']}` | {ledger['solve_defect_over_E_round']:+.1f} | "
            f"{ledger['storage_loss_over_E_round']:+.1f} | {ledger['explained_relative_drift']:+.3e} | "
            f"{measured_text} |")
    add("")
    add("## Candidate rows")
    add("")
    add(
        "| candidate | quick (off 0.0 / 0.5) | medium | angle delta | long horizon | restart | AD | "
        "runtime ratio | status |"
    )
    add("|---|---|---|---|---|---|---|---|---|")
    for row in report["candidate_rows"]:
        quick = row["quick"]
        quick_text = " / ".join(
            "n/a" if quick.get(key) is None else f"{quick[key]:+.3e}"
            for key in ("quick_offset0.0", "quick_offset0.5")
        )
        medium_value = row["medium_final_relative_drift"]
        medium_text = "n/a" if medium_value is None else f"{medium_value:+.3e}"
        angle_text = "n/a" if row["angle_delta_deg"] is None else f"{row['angle_delta_deg']:.3f}"
        long_text = "n/a" if row["long_horizon_clean"] is None else ("clean" if row["long_horizon_clean"] else "BIAS")
        runtime = "n/a" if row["runtime_ratio_vs_A0"] is None else f"{row['runtime_ratio_vs_A0']:.3f}"
        add(
            f"| `{row['candidate']}` | {quick_text} | {medium_text} | {angle_text} | {long_text} | "
            f"{row['restart_reproducible']} | {row['ad_ok']} | {runtime} | {row['status']} |"
        )
    add("")
    if report.get("fixed_state_causal"):
        add("## Fixed-state causal test (spec 26)")
        add("")
        causal = report["fixed_state_causal"]
        add(f"- N = {causal['N']}, target = {causal['target_deg']:.0f} deg, M = {causal['M_factor']}x M_ref, "
            f"warmup = {causal['warmup_substeps']} substeps, `E_round` = {causal['E_round']:.4e}")
        add(f"- baseline storage residual "
            f"{causal['production_storage_residual']['mass_over_E_round']:+.4f} E_round per update, "
            f"{causal['production_storage_residual']['nonzero_fraction']:.3f} of the updates rounded")
        add("")
        add(
            "| candidate | working dtype | Krylov dtype | solve defect / E_round | "
            "storage loss / E_round | input shift / E_round | nonzero |"
        )
        add("|---|---|---|---|---|---|---|")
        for name, row in causal["candidates"].items():
            add(f"| `{name}` | {row['working_dtype']} | {row['krylov_dtype']} | "
                f"{row['solve_mass_defect_over_E_round']:+.4f} | {row['storage_loss_over_E_round']:+.4f} | "
                f"{row['input_shift_over_E_round']:+.4f} | {row['storage_loss_nonzero_fraction']:.3f} |")
        add("")
        add("Baseline storage residual by cell population (E_round):")
        add("")
        for name, value in causal["production_storage_residual"]["population_decomposition_over_E_round"].items():
            add(f"- `{name}`: {value:+.4f}")
        add("")
    if report.get("long_horizon"):
        add("## Long horizon (spec 36)")
        add("")
        for candidate, rows in report["long_horizon"].items():
            fits = fit_summary([row for row in rows])
            parts = ", ".join(
                f"{row['horizon']//1000}k: {row['final_relative_drift']:+.3e} ({row['verdict'].lower()})"
                for row in rows
            )
            add(f"- `{candidate}`: {parts} -- fit {fits['model']} "
                f"(slope t = {fits['slope_t']})")
        add("")
    if report.get("dataset_lineage"):
        add("## Dataset and restart lineage (spec 52)")
        add("")
        for candidate, row in report["dataset_lineage"].items():
            add(f"- `{candidate}`: {row['decision']}")
        add("")
    if report.get("notes"):
        add("## Notes")
        add("")
        for note in report["notes"]:
            add(f"- {note}")
        add("")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ driver
def _environment() -> dict[str, Any]:
    import os

    devices = jax.devices()
    return {
        "jax": jax.__version__,
        "x64": bool(jax.config.jax_enable_x64),
        "devices": [f"{device.platform}:{device.device_kind}" for device in devices],
        "gpu_available": any(device.platform == "gpu" for device in devices),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "phasefield_sha256": _sha256(Path(pf.__file__)),
        "reduction_dtype": str(jnp.zeros(1, jnp.float32).sum().dtype),
    }


def run_audit(
    profile: str = "quick",
    out_dir: str | Path | None = None,
    *,
    quick_steps: int = 2500,
    medium_steps: int = 5000,
    write: bool = True,
) -> dict[str, Any]:
    """Run one audit profile and (optionally) write the evidence files."""
    if profile not in PROFILES:
        raise KeyError(f"unknown profile {profile!r}; known: {sorted(PROFILES)}")
    spec = PROFILES[profile]
    started = time.time()
    report: dict[str, Any] = {
        "stage": STAGE,
        "module": MODULE,
        "profile": profile,
        "contract_version": int(pf.SOLVER_CONTRACT_VERSION),
        "implicit_phase_solver": pf.IMPLICIT_PHASE_SOLVER,
        "environment": _environment(),
        "notes": [],
    }

    # 1. quick drift matrix (both offsets, every candidate) -- the primary gate
    quick_rows: dict[str, Any] = {}
    for candidate in CANDIDATES:
        per_candidate: dict[str, Any] = {}
        for label, offset in (("quick_offset0.0", 0.0), ("quick_offset0.5", 0.5)):
            payload = drift_series(
                48, target_deg=150.0, M_factor=1.0, public_steps=int(quick_steps), sample_every=50,
                candidate=candidate, wall_offset_over_dy=offset,
            )
            stats = series_classification(payload["masses"], payload["E_round"], payload["sample_every"])
            stats["bookkeeping_final_relative_drift"] = float(
                (payload["bookkeeping_masses"][-1] - payload["m0"]) / payload["m0"]
            )
            stats["aux_max_abs_final"] = float(payload["aux_max_abs"][-1])
            stats["ledger"] = payload["ledger"]
            stats["passed"] = abs(stats["final_relative_drift"]) <= QUICK_GATE
            stats["strong"] = abs(stats["final_relative_drift"]) <= QUICK_GATE_STRONG
            stats["in_band"] = True
            per_candidate[label] = stats
        quick_rows[candidate] = per_candidate
    report["quick_matrix"] = quick_rows

    # 2. medium matrix (spec 35), only in the larger profiles
    medium_rows: dict[str, Any] = {}
    if spec["medium"]:
        for candidate in CANDIDATES:
            per_candidate = {}
            for label, degrees in (("medium_150deg", 150.0), ("medium_60deg", 60.0)):
                payload = drift_series(
                    128, target_deg=degrees, public_steps=int(medium_steps), sample_every=100,
                    candidate=candidate,
                )
                stats = series_classification(payload["masses"], payload["E_round"], payload["sample_every"])
                stats["bookkeeping_final_relative_drift"] = float(
                    (payload["bookkeeping_masses"][-1] - payload["m0"]) / payload["m0"]
                )
                stats["angle_deg"] = float(payload["angle_deg"])
                stats["passed"] = abs(stats["final_relative_drift"]) <= MEDIUM_GATE
                stats["strong_passed"] = abs(stats["final_relative_drift"]) <= MEDIUM_GATE_STRONG
                stats["ledger"] = payload["ledger"]
                per_candidate[label] = stats
            medium_rows[candidate] = per_candidate
    report["medium_matrix"] = medium_rows

    # 2b. lineage cross-check against the frozen L1A-2h evidence (read back, never re-derived)
    report["ledger_cross_check"] = ledger_cross_check(report)

    # 3. fixed-state causal test (spec 26) + the per-population first-loss ledger (spec 27)
    report["fixed_state_causal"] = fixed_state_causal(
        spec["causal_N"], warmup=spec["causal_warmup"]
    )
    report["population_series"] = population_series(
        spec["causal_N"], warmup=max(spec["causal_warmup"] // 2, 100)
    )

    # 4. long horizon (spec 36)
    if spec["long_horizons"]:
        long_rows = long_horizon(profile)
        report["_long_series"] = {
            candidate: [
                {"horizon": row["horizon"], "masses": row["_masses"]} for row in rows
            ]
            for candidate, rows in long_rows.items()
        }
        report["long_horizon"] = {
            candidate: [
                {key: value for key, value in row.items() if key != "_masses"} for row in rows
            ]
            for candidate, rows in long_rows.items()
        }

    # 5. restart equivalence (spec 24)
    report["restart_equivalence"] = {
        candidate: restart_equivalence(48, candidate=candidate)
        for candidate in spec["restart_candidates"]
    }

    # 6. gradients (spec 44)
    report["gradients"] = {
        candidate: gradient_check(spec["gradient_N"], candidate=candidate)
        for candidate in CANDIDATES
    }

    # 7. performance (spec 39-42)
    report["performance"] = performance(
        spec["performance_N"], steps=spec["performance_steps"]
    )
    by_candidate = {row["candidate"]: row for row in report["performance"]}
    base_bytes = by_candidate["A0_float32"]["persistent_bytes"]
    for row in report["performance"]:
        row["persistent_bytes_ratio"] = (
            row["persistent_bytes"] / base_bytes if base_bytes else None
        )
        row["gpu_measured"] = False

    # 8. dataset lineage (spec 52)
    report["dataset_lineage"] = {candidate: dataset_lineage(candidate) for candidate in CANDIDATES}

    # 9. gates + candidate rows + selection
    report["gates"] = {
        "quick_bound": QUICK_GATE,
        "quick_strong_bound": QUICK_GATE_STRONG,
        "medium_bound": MEDIUM_GATE,
        "medium_strong_bound": MEDIUM_GATE_STRONG,
        "angle_delta_bound_deg": ANGLE_DELTA_GATE,
        "long_horizon_projection_bound": LONG_HORIZON_PROJECTION_BOUND,
        "runtime_preferred_fraction": RUNTIME_PREFERRED,
        "memory_preferred_fraction": MEMORY_PREFERRED,
        "closure_horizon": CLOSURE_HORIZON,
    }
    report["candidate_rows"] = candidate_rows(report)
    report["selection"] = select_candidate(report["candidate_rows"], report)
    report["cost"] = cost_assessment(report)
    report["production_readiness"] = production_readiness(report)
    report["verdict"] = _verdict(report)
    report["elapsed_seconds"] = time.time() - started
    report["profiles"] = {name: {key: str(value) for key, value in body.items()} for name, body in PROFILES.items()}
    if not report["environment"]["gpu_available"]:
        report["notes"].append(
            "CPU-only environment: GPU cost is not verified (no GPU device present). The performance "
            "section is a CPU measurement and is reported as such."
        )
    report["notes"].append(
        "numpy.polyfit's RankWarning about its step-axis conditioning is silenced in "
        "series_classification (the coefficients are unchanged, the warning is about the x scale)."
    )
    report["notes"].append(
        "The audit's A0 row uses the full CHNS substep chain (momentum, Brinkmann damping and "
        "pressure projection on, as in step_with_diagnostics); the L1A-2h closure ledger's CH-only "
        "fixture is reproduced by the same chain to within 3.1e-07 relative drift at 2500 public "
        "steps, the difference being the phase advection the momentum path adds."
    )
    if write and out_dir is not None:
        report["artifacts"] = write_reports(report, Path(out_dir))
    return report



def cost_assessment(report: dict[str, Any]) -> dict[str, Any]:
    """The cost of the selected candidate, against the *preferred* envelope of spec 39-42."""
    selected = (report.get("selection") or {}).get("selected")
    rows = {row["candidate"]: row for row in (report.get("performance") or [])}
    row = rows.get(selected or "", {})
    runtime_ratio = row.get("runtime_ratio")
    memory_ratio = row.get("persistent_bytes_ratio")
    within_runtime = None if runtime_ratio is None else (runtime_ratio - 1.0) <= RUNTIME_PREFERRED
    within_memory = None if memory_ratio is None else (memory_ratio - 1.0) <= MEMORY_PREFERRED
    return {
        "selected": selected,
        "measured_on": "CPU",
        "N": row.get("N"),
        "seconds_per_substep": row.get("seconds_per_substep"),
        "runtime_ratio_vs_A0": runtime_ratio,
        "persistent_bytes_ratio_vs_A0": memory_ratio,
        "runtime_preferred_fraction": RUNTIME_PREFERRED,
        "memory_preferred_fraction": MEMORY_PREFERRED,
        "runtime_within_preferred_budget": within_runtime,
        "memory_within_preferred_budget": within_memory,
        "gpu_measured": False,
        "gpu_note": "no GPU device in this environment; float64 throughput on the production GPU is "
        "unverified, so the cost envelope is *not* accepted silently (spec 42).",
        "verdict": (
            "NOT_MEASURED"
            if within_runtime is None
            else "WITHIN_PREFERRED_BUDGET"
            if (within_runtime and within_memory)
            else "ABOVE_PREFERRED_BUDGET"
        ),
    }


def production_readiness(report: dict[str, Any]) -> dict[str, Any]:
    """Whether the evidence supports going to production scale, stated without euphemism."""
    selection = report.get("selection") or {}
    selected = selection.get("selected")
    cost = report.get("cost") or {}
    rows = {row["candidate"]: row for row in report.get("candidate_rows") or []}
    row = rows.get(selected or "", {})
    blockers: list[str] = []
    if selected is None:
        blockers.append("no candidate cleared every measured gate")
    if row.get("restart_reproducible") is False:
        blockers.append("restart is not reproducible for the selected candidate")
    if row.get("ad_ok") is False:
        blockers.append("the storage rule breaks forward-mode AD agreement")
    if cost.get("verdict") == "ABOVE_PREFERRED_BUDGET":
        blockers.append("runtime/memory above the preferred envelope on the measured device")
    if not cost.get("gpu_measured", True):
        blockers.append("GPU cost unverified (no GPU in this environment)")
    quick_ok = row.get("quick_passed")
    medium_ok = row.get("medium_passed")
    if quick_ok is False or medium_ok is False:
        blockers.append("a mass gate failed")
    return {
        "selected": selected,
        "ready_for_l1b_production_scale": not blockers,
        "blockers": blockers,
        "statement": (
            "READY FOR L1B PRODUCTION SCALE"
            if not blockers
            else "NOT READY FOR L1B PRODUCTION SCALE ("
            + "; ".join(blockers)
            + ")"
        ),
    }


def _verdict(report: dict[str, Any]) -> str:
    selection = report["selection"]
    if not selection.get("selected"):
        if report["profile"] == "quick":
            return "NOT_READY_FOR_SELECTION"
        return "NO_CANDIDATE_READY_FOR_L1B_PRODUCTION_SCALE"
    cost = report.get("cost") or {}
    if cost.get("verdict") == "ABOVE_PREFERRED_BUDGET":
        return "PHASE_STORAGE_MODEL_SELECTED_COST_ABOVE_PREFERRED_BUDGET"
    if cost.get("verdict") == "NOT_MEASURED":
        return "PHASE_STORAGE_MODEL_SELECTED_COST_UNMEASURED"
    return "PHASE_STORAGE_MODEL_SELECTED"



def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", default="quick", choices=sorted(PROFILES))
    parser.add_argument("--out", default=None, help="evidence directory (default: none)")
    parser.add_argument("--quick-steps", type=int, default=2500)
    parser.add_argument("--medium-steps", type=int, default=5000)
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args(argv)

    report = run_audit(
        args.profile,
        args.out,
        quick_steps=args.quick_steps,
        medium_steps=args.medium_steps,
        write=args.out is not None,
    )
    if args.print_json:
        json.dump(
            {key: value for key, value in report.items() if not key.startswith("_")},
            sys.stdout,
            indent=2,
            sort_keys=True,
        )
        sys.stdout.write("\n")
    print(
        f"[{STAGE}] profile={args.profile} verdict={report['verdict']} "
        f"selected={report['selection'].get('selected')} elapsed={report['elapsed_seconds']:.1f}s"
    )
    if args.out:
        print(f"[{STAGE}] evidence written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
