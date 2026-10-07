"""Non-neutral sessile equilibration and contact-line kinetics audit (L1A-2d).

Diagnostic / falsifiable physics audit. It never changes a production default and never feeds a result
back into the solver (``trajectory_semantics_changed`` stays false for this stage). It reports the *live*
solver contract: contract 7 for the frozen L1A-2d evidence, contract 8 once the L1A-2e embedded wall
measure is in the tree, in which case the same frozen runner diagnoses the v8 operator and the pinned
``wall_measure='diffuse_sdf_v7'`` option reproduces the v7 root cause with identical code.

Question: why do the 60/120/150 deg sessile drops not reach equilibrium within 10,000 steps?  The audit
separates mechanisms by running

* **CH-only** (``u = v = 0``; :func:`phasefield.phase_only_step_with_diagnostics`, which calls the exact
  v7 phase operator -- same face apertures, matrix-free implicit CH solve and natural Young BC) and
  **full CHNS** (:func:`phasefield.step_with_diagnostics`) from the same clean 90-degree cap;
* a mobility sweep (``M/M_ref = 0.5, 1, 2, 4``) compared on ``M*t`` and on *converged / staged* states, never
  on equal step counts;
* a dt sweep at fixed physical time, a fixed-N interface-thickness sweep (A) and a *coupled grid/interface
  refinement* (B, physical eps changes with dx);
* a Young boundary residual ``R_Y = eps dphi/dn + g_w'`` in the contact-line neighbourhood, validated on
  manufactured fields before it is used on a trajectory;
* a one-factor, diagnostic-only wall-gain ablation (the cosine handed to the unchanged solver is scaled by
  the inverse of the measured fluid-side wall-kernel fraction).  It is an experiment, not a calibration.

Run from ``examples/two_phase``::

    python -m production.nonneutral_wetting_audit --profile quick --out artifacts/production_validation/l1a2d_quick
    python -m production.nonneutral_wetting_audit --profile baseline --out artifacts/production_validation/l1a2d
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import phasefield as pf
from production import contact_line_kinetics as clk
from production import observables as obs
from production.config import get_git_sha

STAGE = "L1A-2d"
M_REF = 2.0e-3
SECTIONS = ("primary", "mobility", "dt", "resolution", "gain_ablation")

#: Historical v7 evidence (N=128, 10,000 steps, old 3-window angle+speed gate). Never overwritten.
HISTORICAL_V7_BASELINE = [
    {
        "target_deg": 60.0,
        "equilibrium_angle_deg": None,
        "final_sampled_angle_deg": 75.829,
        "steps": 10000,
        "converged": False,
        "final_max_speed": 1.887e-3,
        "fluid_mass_drift": 6.42e-4,
    },
    {
        "target_deg": 90.0,
        "equilibrium_angle_deg": 89.961,
        "final_sampled_angle_deg": 89.961,
        "steps": 600,
        "converged": True,
        "final_max_speed": 3.995e-4,
        "fluid_mass_drift": 3.51e-5,
    },
    {
        "target_deg": 120.0,
        "equilibrium_angle_deg": None,
        "final_sampled_angle_deg": 102.659,
        "steps": 10000,
        "converged": False,
        "final_max_speed": 1.619e-3,
        "fluid_mass_drift": 5.22e-4,
    },
    {
        "target_deg": 150.0,
        "equilibrium_angle_deg": None,
        "final_sampled_angle_deg": 113.256,
        "steps": 10000,
        "converged": False,
        "final_max_speed": 2.628e-3,
        "fluid_mass_drift": 5.67e-4,
    },
]

CRITERIA = {
    "window_samples": 5,  # K (minimum number of samples inside the window)
    "window_mobility_time": 0.05,  # stationarity window, in units of M*t
    "angle_tol_deg": 0.10,  # max pairwise |dtheta| in the window
    # |dF|/max(1,|F|) between consecutive samples. The strict spec value 1e-7 is reported separately
    # (``energy_stationary_strict``) but cannot gate: the *neutral 90 deg control* keeps a bulk relaxation of
    # dF/F ~ 4e-5 per 200 steps at 10k steps with the angle fixed to 0.02 deg, so a 1e-7 gate would never fire.
    "energy_rel_tol": 1.0e-4,
    "strict_energy_rel_tol": 1.0e-7,
    "phase_rate_l2_tol": 1.0e-3,  # ||phi_{n+1}-phi_n||_2 / dt
    "chns_speed_tol": 5.0e-4,  # extra gate for full CHNS only
    "acceptance_error_deg": 5.0,  # provisional engineering target (frozen, not tuned)
    "relaxing_rate_deg_per_Mt": 3.0,  # staged runner: continue while |dtheta/d(M t)| >= this (and F decreasing)
}

PROFILES: dict[str, dict[str, Any]] = {
    "baseline": {
        "N": 128,
        "R": 1.1,
        "dt": 4.0e-3,
        "eps_factor": 2.0,
        "sample_every": 200,
        "targets": [60.0, 90.0, 120.0, 150.0],
        "budgets": [10000, 25000, 50000],
        "mobility": {"targets": [60.0, 120.0, 150.0], "factors": [0.5, 1.0, 2.0, 4.0]},
        "dt_sweep": {"targets": [150.0], "dts": [4.0e-3, 2.0e-3, 1.0e-3], "horizon": 40.0},
        "resolution_a": {"targets": [60.0, 150.0], "eps_factors": [1.5, 2.0, 2.5], "budget": 10000},
        "resolution_b": {"targets": [60.0, 150.0], "N_values": [96, 128, 192], "budget": 10000},
        "gain_ablation": {
            "targets": [60.0, 120.0, 150.0],
            "mobility_factor": 4.0,
            "N_values": [96, 192],
            "grid_targets": [60.0, 150.0],
        },
    },
    "quick": {
        "N": 48,
        "R": 0.6,
        "dt": 4.0e-3,
        "eps_factor": 2.0,
        "sample_every": 40,
        "targets": [60.0, 150.0],
        "budgets": [200, 400],
        "mobility": {"targets": [150.0], "factors": [1.0, 2.0]},
        "dt_sweep": {"targets": [150.0], "dts": [4.0e-3, 2.0e-3], "horizon": 1.6},
        "resolution_a": {"targets": [150.0], "eps_factors": [1.5, 2.5], "budget": 200},
        "resolution_b": {"targets": [150.0], "N_values": [32, 48], "budget": 200},
        "gain_ablation": {"targets": [150.0], "mobility_factor": 2.0, "N_values": [], "grid_targets": []},
    },
}


# --------------------------------------------------------------------------------------
#  stepping
# --------------------------------------------------------------------------------------
@functools.partial(jax.jit, static_argnums=(2, 3, 4))
def _advance(state, solid, p, n_steps: int, ch_only: bool):
    """Advance ``n_steps`` public steps; also return phi before the last step and CG diagnostics."""
    step_fn = pf.phase_only_step_with_diagnostics if ch_only else pf.step_with_diagnostics

    def body(carry, _):
        s, max_it, max_res = carry
        s_next, info = step_fn(s, solid, p)
        return (
            s_next,
            jnp.maximum(max_it, jnp.max(info.implicit_iterations)),
            jnp.maximum(max_res, jnp.max(info.implicit_relative_residuals)),
        ), None

    init = (state, jnp.int32(0), jnp.asarray(0.0, dtype=state.phi.dtype))
    (s_prev, max_it, max_res), _ = jax.lax.scan(body, init, None, length=n_steps - 1)
    s_final, info = step_fn(s_prev, solid, p)
    max_it = jnp.maximum(max_it, jnp.max(info.implicit_iterations))
    max_res = jnp.maximum(max_res, jnp.max(info.implicit_relative_residuals))
    return s_prev.phi, s_final, max_it, max_res, jnp.all(info.implicit_converged)


def _clean(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _sample(
    state,
    prev_phi,
    solid,
    p,
    *,
    ch_only: bool,
    steps: int,
    cos_eff: float,
    phi_ref_mass: float,
    M: float,
    volume=None,
    phi_ref_conserved_mass: float | None = None,
):
    phi = np.asarray(state.phi, dtype=np.float64)
    if not np.isfinite(phi).all():
        raise FloatingPointError("non-finite phase field (implicit solve failed closed)")
    sdf = np.asarray(solid.sdf, dtype=np.float64)
    prev = np.asarray(prev_phi, dtype=np.float64)
    rate = (phi - prev) / float(p.dt)
    theta = float(pf.measure_contact_angle(state.phi, solid, p))
    # Diagnostic reference plane: the bottom face of the lowest hard-fluid row, i.e. the wall the
    # cell-centre fluid mask actually gives the phase field (see pf.discrete_fluid_boundary_height).
    discrete_wall = float(pf.discrete_fluid_boundary_height(solid, p))
    theta_discrete = float(pf.measure_contact_angle(state.phi, solid, p, wall_plane=discrete_wall))
    pos = clk.contact_line_positions(phi, sdf, p.dx, p.dy, eps=p.eps, Lx=p.Lx)
    ry = clk.young_boundary_residual(phi, sdf, p.dx, p.dy, p.eps, cos_eff)
    ry_first = clk.young_boundary_residual_first_layer(
        phi,
        sdf,
        p.dx,
        p.dy,
        p.eps,
        cos_eff,
        wall_area=np.asarray(solid.wall_area, dtype=np.float64),
        wall_normal_x=np.asarray(solid.wall_normal_x, dtype=np.float64),
        wall_normal_y=np.asarray(solid.wall_normal_y, dtype=np.float64),
    )
    free_energy = float(pf.phase_free_energy(state.phi, solid, p))
    fluid_mass = obs.liquid_mass(phi, sdf, p.dx, p.dy)
    # Contract v9 conserved quantity: Q = sum_i V_i phi_i over the cut-cell control volumes.
    # ``fluid_mass`` above stays as the hard-mask diagnostic so the two can be compared directly.
    conserved_mass = None if volume is None else float(np.sum(phi * np.asarray(volume, dtype=np.float64)))
    positive = max(float(np.maximum(phi, 0.0).sum()), 1e-30)
    solid_fraction = float(np.maximum(phi[sdf < 0.0], 0.0).sum() / positive)
    u = np.asarray(state.u, dtype=np.float64)
    v = np.asarray(state.v, dtype=np.float64)
    # §17 cut-cell advective CFL diagnostic: only meaningful when the velocity is live (the CH-only
    # runs hold u = v = 0, where every dt_adv is infinite and the ratio is not a constraint).
    cfl = None
    if not ch_only and float(np.max(np.abs(u))) + float(np.max(np.abs(v))) > 0.0:
        cfl = pf.cutcell_advective_cfl_diagnostic(state.u, state.v, solid, p)
    speed2 = u * u + v * v
    rho = np.asarray(pf.rho_of(state.phi, p), dtype=np.float64)
    chi = np.asarray(solid.chi, dtype=np.float64)
    dA = p.dx * p.dy
    kinetic = float(np.sum(0.5 * rho * speed2) * dA)
    brinkman = float(np.sum(rho * chi * speed2) / p.eta_pen * dA)
    scaled_interface = float(pf.SIGMA_NORM) / float(p.We) * free_energy
    return {
        "step": int(steps),
        "time": float(state.t),
        "mobility_scaled_time": float(M) * float(state.t),
        "measured_angle_deg": theta if math.isfinite(theta) and pos["contact_line_exists"] else None,
        "raw_angle_deg": theta if math.isfinite(theta) else None,
        "discrete_wall_plane": discrete_wall,
        "geometric_wall_plane": float(pf.wall_plane_height(solid, p)),
        "measured_angle_deg_discrete_wall": theta_discrete if math.isfinite(theta_discrete) else None,
        "contact_line_exists": bool(pos["contact_line_exists"]),
        "detachment_observed": bool(pos["detachment_observed"]),
        "contour_wall_intersection_count": int(pos["contour_wall_intersection_count"]),
        "bottom_gap": pos["bottom_gap"],
        "left_contact_x": pos["left_contact_x"],
        "right_contact_x": pos["right_contact_x"],
        "left_contact_x_wrapped": pos["left_contact_x_wrapped"],
        "right_contact_x_wrapped": pos["right_contact_x_wrapped"],
        "contact_width": pos["contact_width"] if pos["contact_line_exists"] else None,
        "y_cm": pos["y_cm"],
        "top_height": pos["top_height"],
        "fluid_mass": float(fluid_mass),
        "mass_drift": abs(float(fluid_mass) - phi_ref_mass) / max(abs(phi_ref_mass), 1e-30),
        "cutcell_advective_cfl_ratio": None if cfl is None else cfl["cutcell_advective_cfl_ratio"],
        "dt_adv_min": None if cfl is None else cfl["dt_adv_min"],
        "conserved_liquid_mass": conserved_mass,
        "conserved_mass_drift": (
            None
            if conserved_mass is None or phi_ref_conserved_mass is None
            else abs(conserved_mass - phi_ref_conserved_mass) / max(abs(phi_ref_conserved_mass), 1e-30)
        ),
        "solid_phase_fraction": solid_fraction,
        "free_energy": free_energy,
        "RY_l2": ry["RY_l2"],
        "RY_linf": ry["RY_linf"],
        "RY_normalized_l2": ry["RY_normalized_l2"],
        "RY_normalized_linf": ry["RY_normalized_linf"],
        "RY_n_points": ry["n_points"],
        # L1A-2e: first-fluid-layer residual on the cells that carry the wall measure.
        "RY_first_l2": ry_first["RY_first_l2"],
        "RY_first_linf": ry_first["RY_first_linf"],
        "RY_first_normalized_l2": ry_first["RY_first_normalized_l2"],
        "RY_first_normalized_linf": ry_first["RY_first_normalized_linf"],
        "wall_measure_weighted_RY": ry_first["wall_measure_weighted_RY"],
        "wall_measure_weighted_RY_all_wall": ry_first["wall_measure_weighted_RY_all_wall"],
        "n_wall_cells": ry_first["n_wall_cells"],
        "n_wall_cells_band": ry_first["n_wall_cells_band"],
        "phase_rate_l2": float(np.sqrt(np.sum(rate * rate) * dA)),
        "phase_rate_linf": float(np.max(np.abs(rate))),
        "max_speed": 0.0 if ch_only else float(np.sqrt(speed2.max())),
        "E_kin": kinetic,
        "brinkman_dissipation_proxy": brinkman,
        "E_interface_wall_scaled": scaled_interface,
        "E_total": kinetic + scaled_interface,
    }


def _window_converged(
    samples: list[dict[str, Any]], ch_only: bool, crit: dict[str, Any], M: float = M_REF
) -> dict[str, Any]:
    """Stationarity gate on the last ``window_mobility_time`` of mobility-scaled time (>= K samples).

    The window is fixed in ``M*t`` (not in steps or physical time): a fixed step window let a run that still
    drifted at ~15 deg per unit ``M*t`` look "stationary" (0.06 deg per window) and converge ~5 deg early.
    ``angle_tol_deg`` over ``window_mobility_time`` therefore means ``|dtheta/d(M t)| <~ 2 deg``.
    """
    k = int(crit["window_samples"])
    fail = {"converged": False, "angle_ok": False, "energy_ok": False, "rate_ok": False, "speed_ok": False}
    if len(samples) < k:
        return fail
    last_mt = samples[-1]["mobility_scaled_time"]
    span = float(crit["window_mobility_time"])
    win = [r for r in samples if r["mobility_scaled_time"] >= last_mt - span - 1e-12]
    covered = samples[0]["mobility_scaled_time"] <= last_mt - span + 1e-12
    if len(win) < k or not covered:
        return fail
    angles = [r["measured_angle_deg"] for r in win]
    if any(a is None for a in angles):
        return fail
    angle_spread = max(angles) - min(angles)  # max pairwise |dtheta|
    energies = [r["free_energy"] for r in win]
    dF = max(abs(b - a) / max(1.0, abs(b)) for a, b in zip(energies, energies[1:]))
    angle_ok = angle_spread <= float(crit["angle_tol_deg"])
    energy_ok = dF <= float(crit["energy_rel_tol"])
    # phase rate scales with M: compare it in mobility-scaled units so slow runs cannot pass trivially
    rate_ok = win[-1]["phase_rate_l2"] * (M_REF / float(M)) <= float(crit["phase_rate_l2_tol"])
    speed_ok = True if ch_only else win[-1]["max_speed"] <= float(crit["chns_speed_tol"])
    return {
        "converged": bool(angle_ok and energy_ok and rate_ok and speed_ok),
        "angle_ok": bool(angle_ok),
        "energy_ok": bool(energy_ok),
        "rate_ok": bool(rate_ok),
        "speed_ok": bool(speed_ok),
        "angle_window_spread_deg": float(angle_spread),
        "window_samples_used": len(win),
        "window_mobility_time": span,
        "energy_window_max_rel_change": float(dF),
        "energy_stationary_strict": bool(dF <= float(crit["strict_energy_rel_tol"])),
    }


def _still_relaxing(samples: list[dict[str, Any]], crit: dict[str, Any], M: float, verdict: dict[str, Any]) -> bool:
    """Staged-runner rule: continue while F still decreases and the run is not yet stationary.

    "Not yet stationary" means the late angle drifts at a clear *mobility-scaled* rate (a fixed degrees-per-window
    threshold would stop slow runs early) or the energy/phase-rate gate is the only one still failing.
    """
    pairs = [(r["time"], r["measured_angle_deg"]) for r in samples if r["measured_angle_deg"] is not None]
    if len(pairs) < 6:
        return True
    trend = clk.late_time_linear_trend([t for t, _ in pairs], [a for _, a in pairs])
    energy = clk.late_time_linear_trend([r["time"] for r in samples], [r["free_energy"] for r in samples])
    rate_per_mt = abs(trend["slope"]) / float(M)
    unfinished = rate_per_mt >= float(crit["relaxing_rate_deg_per_Mt"]) or not (
        verdict.get("energy_ok") and verdict.get("rate_ok")
    )
    return bool(unfinished and energy["slope"] < 0.0)


def _thin(samples: list[dict[str, Any]], keep: int) -> list[dict[str, Any]]:
    if len(samples) <= keep:
        return samples
    idx = np.unique(np.linspace(0, len(samples) - 1, keep).round().astype(int))
    return [samples[i] for i in idx]


def run_relaxation(
    target_deg: float,
    *,
    ch_only: bool,
    N: int = 128,
    eps_factor: float = 2.0,
    M: float = M_REF,
    dt: float = 4.0e-3,
    R: float = 1.1,
    wall_height: float = 0.25,
    budgets: Sequence[int] = (10000, 25000, 50000),
    sample_every: int = 200,
    criteria: dict[str, Any] | None = None,
    wall_gain: float = 1.0,
    fixed_steps: int | None = None,
    keep_samples: int = 80,
    label: str = "",
    group: str = "primary",
    dtype: str = "float32",
    ch_solver_rtol: float | None = None,
    ch_solver_max_iterations: int | None = None,
    wall_measure: str | None = None,
    budgets_extra: Sequence[int] = (),
    phase_transport_geometry: str | None = None,
    wall_offset_over_dy: float = 0.0,
    checkpoint_in: str | Path | None = None,
    checkpoint_out: str | Path | None = None,
    start_step: int = 0,
    mass_reference: float | None = None,
    conserved_mass_reference: float | None = None,
    prior_samples: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """One staged relaxation. ``fixed_steps`` disables the staged/convergence stop (dt sweep).

    ``wall_gain`` multiplies cos(theta) handed to the *unchanged* solver and is a diagnostic ablation knob
    only; production runs always use 1.0. ``wall_measure`` selects the embedded wall measure
    (``sdf_cutcell_v1`` production default, ``diffuse_sdf_v7`` = the pinned contract-v7 kernel) and exists so
    the v7 root cause can be reproduced with identical code; it is a geometry choice, never a fitted factor.
    ``dtype``/``ch_solver_*`` are the precision-matrix knobs (L1A-2e): the production default stays
    float32 with ``rtol = 1e-6``. ``checkpoint_in``/``checkpoint_out`` and the reference/sample arguments
    resume the exact fluid and velocity state for staged audits; they do not alter the production step.
    """
    crit = dict(CRITERIA if criteria is None else criteria)
    budgets = sorted({int(b) for b in budgets} | {int(b) for b in budgets_extra})
    started = time.perf_counter()
    kwargs: dict[str, Any] = {}
    if ch_solver_rtol is not None:
        kwargs["ch_solver_rtol"] = float(ch_solver_rtol)
    if ch_solver_max_iterations is not None:
        kwargs["ch_solver_max_iterations"] = int(ch_solver_max_iterations)
    if wall_measure is not None:
        kwargs["wall_measure"] = str(wall_measure)
    if phase_transport_geometry is not None:
        # "sdf_cutcell_fv_v1" is the contract-v9 production transport; "hard_cell_v7" pins the
        # contract-v7/v8 cell-centre staircase domain for the A/B falsification evidence.
        kwargs["phase_transport_geometry"] = str(phase_transport_geometry)
    p = pf.PhaseFieldParams(
        Nx=N,
        Ny=N,
        Lx=6.0,
        Ly=6.0,
        Re=200.0,
        We=100.0,
        dt=dt,
        M=M,
        eps=eps_factor * 6.0 / N,
        dtype=jnp.float64 if str(dtype) == "float64" else jnp.float32,
        **kwargs,
    )
    cos_target = math.cos(math.radians(float(target_deg)))
    cos_eff = float(wall_gain) * cos_target
    # sub-cell wall translation: the geometric wall sits at ``wall_height + offset * dy`` so the
    # same physical wall can be placed anywhere inside a cell (the L1A-2f translation matrix).
    height = float(wall_height) + float(wall_offset_over_dy) * float(p.dy)
    sdf = pf.surface_flat(p, wall_height=height)
    solid = pf.make_solid(sdf, p, cos_theta=cos_eff)
    seed_state = pf.sessile_initial_state(p, solid, R=R, wall_height=height)
    step_offset = int(start_step)
    if step_offset < 0:
        raise ValueError("start_step must be non-negative")

    checkpoint_signature = {
        "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
        "N": int(N),
        "target_deg": float(target_deg),
        "dynamics_mode": "ch_only" if ch_only else "chns",
        "M": float(M),
        "dt": float(p.dt),
        "eps": float(p.eps),
        "R": float(R),
        "wall_height": float(height),
        "wall_gain": float(wall_gain),
        "phase_boundary_model": str(p.phase_boundary_model),
        "phase_transport_geometry": str(p.phase_transport_geometry),
        "phase_storage_model": str(p.phase_storage_model),
        "wall_measure_method": str(p.wall_measure),
    }
    if checkpoint_in is not None:
        with np.load(Path(checkpoint_in), allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
            for key, expected in checkpoint_signature.items():
                actual = metadata.get(key)
                if isinstance(expected, float):
                    matches = isinstance(actual, (int, float)) and math.isclose(
                        float(actual), expected, rel_tol=1e-12, abs_tol=1e-14
                    )
                else:
                    matches = actual == expected
                if not matches:
                    raise ValueError(
                        f"checkpoint {checkpoint_in} metadata mismatch for {key}: {actual!r} != {expected!r}"
                    )
            checkpoint_step = int(metadata["steps"])
            if step_offset not in (0, checkpoint_step):
                raise ValueError(f"start_step {step_offset} does not match checkpoint step {checkpoint_step}")
            step_offset = checkpoint_step
            state = pf.State(
                phi=jnp.asarray(data["phi"]),
                u=jnp.asarray(data["u"]),
                v=jnp.asarray(data["v"]),
                t=jnp.asarray(data["time"]),
            )
        for name in ("phi", "u", "v"):
            values = np.asarray(getattr(state, name))
            if values.shape != (N, N) or not np.isfinite(values).all():
                raise ValueError(f"checkpoint {checkpoint_in} has invalid state.{name}")
    else:
        if step_offset != 0 or prior_samples:
            raise ValueError("a nonzero start_step or prior_samples requires checkpoint_in")
        state = seed_state

    cutcell = pf.phase_transport_is_cutcell(p)
    volume = np.asarray(solid.geometry.volume, dtype=np.float64) if cutcell else None
    computed_mass_reference = float(obs.liquid_mass(np.asarray(seed_state.phi), np.asarray(solid.sdf), p.dx, p.dy))
    computed_conserved_reference = (
        float(np.sum(np.asarray(seed_state.phi, dtype=np.float64) * volume)) if cutcell else None
    )
    if checkpoint_in is not None and (mass_reference is None or (cutcell and conserved_mass_reference is None)):
        raise ValueError("resumed runs require both original fluid-mass and conserved-mass references")
    mass0 = computed_mass_reference if mass_reference is None else float(mass_reference)
    mass0_conserved = (
        computed_conserved_reference if conserved_mass_reference is None else float(conserved_mass_reference)
    )
    sample_extra = {"volume": volume, "phi_ref_conserved_mass": mass0_conserved}
    kernel = clk.fluid_wall_delta_integral(np.asarray(solid.sdf), p.dx, p.dy)
    initial = _sample(
        state,
        state.phi,
        solid,
        p,
        ch_only=ch_only,
        steps=step_offset,
        cos_eff=cos_eff,
        phi_ref_mass=mass0,
        M=M,
        **sample_extra,
    )
    samples = list(prior_samples) if prior_samples else [initial]
    if prior_samples:
        if not checkpoint_in:
            raise ValueError("prior_samples require checkpoint_in")
        if int(samples[-1].get("step", -1)) != step_offset:
            raise ValueError("the last prior sample step must match the checkpoint step")
        if not math.isclose(float(samples[-1].get("time", math.nan)), float(state.t), rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError("the last prior sample time must match the checkpoint state time")
    applied = step_offset
    it_max, res_max = 0, 0.0
    verdict: dict[str, Any] = {"converged": False}
    stop_reason = "budget_exhausted"
    stage_log = []
    cg_failed = False
    if fixed_steps is not None and int(fixed_steps) < 0:
        raise ValueError("fixed_steps must be non-negative")
    total_budget = step_offset + int(fixed_steps) if fixed_steps is not None else budgets[-1]
    if total_budget < step_offset:
        raise ValueError("final step budget must be at least the continuation start step")
    next_budget_idx = 0
    # The stationarity window must span the same *mobility-scaled* time at every M: with a fixed step cadence a
    # slow (small-M) run drifts < 0.1 deg per window while still tens of degrees from equilibrium (observed:
    # M = 0.5 M_ref "converged" at 82.6 deg for a 60 deg target with a ~71 deg equilibrium).
    every = max(1, int(round(float(sample_every) * M_REF / float(M))))
    while applied < total_budget:
        n = min(every, total_budget - applied)
        prev_phi, state, max_it, max_res, ok = _advance(state, solid, p, n, ch_only)
        applied += n
        it_max = max(it_max, int(max_it))
        res_max = max(res_max, float(max_res))
        if not bool(ok):
            cg_failed = True
            stop_reason = "implicit_solve_failed_closed"
            break
        try:
            row = _sample(
                state,
                prev_phi,
                solid,
                p,
                ch_only=ch_only,
                steps=applied,
                cos_eff=cos_eff,
                phi_ref_mass=mass0,
                M=M,
                **sample_extra,
            )
        except FloatingPointError:
            cg_failed = True
            stop_reason = "non_finite_state"
            break
        samples.append(row)
        if row["detachment_observed"] or not row["contact_line_exists"]:
            stop_reason = "detachment_or_topology_change"
            break
        verdict = _window_converged(samples[1:], ch_only, crit, M)
        if fixed_steps is None:
            if verdict["converged"]:
                stop_reason = "converged"
                break
            if next_budget_idx < len(budgets) - 1 and applied >= budgets[next_budget_idx]:
                relaxing = _still_relaxing(samples[1:], crit, M, verdict)
                stage_log.append({"at_step": int(applied), "continue": bool(relaxing)})
                next_budget_idx += 1
                if not relaxing:
                    stop_reason = "plateau_without_convergence"
                    break
    elapsed = time.perf_counter() - started
    verdict = _window_converged(samples[1:], ch_only, crit, M) if len(samples) > 1 else verdict
    final = samples[-1]
    detached = any(r["detachment_observed"] for r in samples)

    valid = [r for r in samples if r["measured_angle_deg"] is not None]
    times = [r["time"] for r in valid]
    late = {}
    for name, key in (
        ("angle", "measured_angle_deg"),
        ("energy", "free_energy"),
        ("contact_width", "contact_width"),
        ("RY_l2", "RY_l2"),
    ):
        pts = [(r["time"], r[key]) for r in valid if r.get(key) is not None]
        if len(pts) >= 3:
            late[name] = clk.late_time_linear_trend([t for t, _ in pts], [v for _, v in pts])
        else:
            late[name] = None
    fit = clk.fit_relaxation_asymptote(times, [r["measured_angle_deg"] for r in valid], target_deg=target_deg)

    vel = None
    wrapped = [r for r in samples if r["left_contact_x_wrapped"] is not None]
    if len(wrapped) >= 3:
        vel = clk.contact_line_velocity(
            [r["time"] for r in wrapped],
            [r["left_contact_x_wrapped"] for r in wrapped],
            [r["right_contact_x_wrapped"] for r in wrapped],
            Lx=p.Lx,
        )
        for r, ls, rs, ms in zip(wrapped, vel["left_speed"], vel["right_speed"], vel["mean_speed"]):
            r["left_contact_line_speed"], r["right_contact_line_speed"], r["mean_contact_line_speed"] = ls, rs, ms
    energies = [r["free_energy"] for r in samples]
    scale = max(1.0, abs(energies[0]))
    violations = int(sum((b - a) / scale > 1.0e-6 for a, b in zip(energies, energies[1:])))
    converged = bool(verdict["converged"]) and stop_reason == "converged"
    eq_angle = final["measured_angle_deg"] if converged else None
    record = {
        "group": group,
        "label": label,
        "target_deg": float(target_deg),
        "dynamics_mode": "ch_only" if ch_only else "chns",
        "N": int(N),
        "dx": float(p.dx),
        "eps": float(p.eps),
        "eps_over_dx": float(p.eps / p.dx),
        "Cn": float(p.eps / (2.0 * R)),
        "M": float(M),
        "M_over_M_ref": float(M / M_REF),
        "dt": float(p.dt),
        "R": float(R),
        "continuation_start_step": int(step_offset),
        "steps": int(applied),
        "mass_reference_initial": float(mass0),
        "physical_time": float(final["time"]),
        "mobility_scaled_time": float(M * final["time"]),
        "stop_reason": stop_reason,
        "stage_log": stage_log,
        "sample_every_steps": int(every),
        "wall_gain_diagnostic": float(wall_gain),
        "applied_cos_theta": cos_eff,
        "dtype": str(dtype),
        "ch_solver_rtol": float(p.ch_solver_rtol),
        "ch_solver_max_iterations": int(p.ch_solver_max_iterations),
        "wall_measure_method": str(p.wall_measure),
        "wall_offset_cells": float((wall_height - 0.25) / p.dy),
        "final_angle_deg_discrete_wall": final["measured_angle_deg_discrete_wall"],
        "discrete_wall_plane": final["discrete_wall_plane"],
        "geometric_wall_plane": final["geometric_wall_plane"],
        "wall_area_total": float(np.sum(np.asarray(solid.wall_area, dtype=np.float64))),
        "wall_area_expected_length": float(p.Lx),
        "wall_area_relative_error": abs(float(np.sum(np.asarray(solid.wall_area, dtype=np.float64))) - p.Lx) / p.Lx,
        "n_wall_cells": int(np.count_nonzero(np.asarray(solid.wall_area) > 0.0)),
        "converged": converged,
        "convergence_window": {k: v for k, v in verdict.items() if k != "converged"},
        "equilibrium_angle_deg": eq_angle,
        "equilibrium_error_deg": (eq_angle - target_deg) if eq_angle is not None else None,
        "final_sampled_angle_deg": final["measured_angle_deg"],
        "final_sampled_error_deg": (final["measured_angle_deg"] - target_deg)
        if final["measured_angle_deg"] is not None
        else None,
        "detachment_observed": bool(detached),
        "contact_line_exists": bool(final["contact_line_exists"]),
        "free_energy_initial": float(energies[0]),
        "free_energy_final": float(energies[-1]),
        "free_energy_monotonic_violations": violations,
        "phase_rate_l2": final["phase_rate_l2"],
        "phase_rate_linf": final["phase_rate_linf"],
        "RY_l2": final["RY_l2"],
        "RY_linf": final["RY_linf"],
        "RY_normalized_l2": final["RY_normalized_l2"],
        "RY_normalized_linf": final["RY_normalized_linf"],
        "RY_n_points": final["RY_n_points"],
        "RY_first_l2": final["RY_first_l2"],
        "RY_first_linf": final["RY_first_linf"],
        "RY_first_normalized_l2": final["RY_first_normalized_l2"],
        "RY_first_normalized_linf": final["RY_first_normalized_linf"],
        "wall_measure_weighted_RY": final["wall_measure_weighted_RY"],
        "wall_measure_weighted_RY_all_wall": final["wall_measure_weighted_RY_all_wall"],
        "n_wall_cells_band": final["n_wall_cells_band"],
        "contact_width": final["contact_width"],
        "contact_line_speed": final.get("mean_contact_line_speed"),
        "y_cm": final["y_cm"],
        "top_height": final["top_height"],
        "final_max_speed": final["max_speed"],
        "E_kin_final": final["E_kin"],
        "E_total_final": final["E_total"],
        "brinkman_dissipation_proxy_final": final["brinkman_dissipation_proxy"],
        "late_trends": late,
        "asymptotic_fit": fit,
        "wall_kernel": kernel,
        "mass_drift": float(max(r["mass_drift"] for r in samples)),
        "mass_drift_final": final["mass_drift"],
        # the contract-v9 conserved quantity: sum_i V_i phi_i on the cut-cell control volumes
        "conserved_mass_drift": (
            None
            if not cutcell
            else float(max(r["conserved_mass_drift"] for r in samples if r["conserved_mass_drift"] is not None))
        ),
        "conserved_mass_drift_final": None if not cutcell else final["conserved_mass_drift"],
        "conserved_mass_reference_initial": mass0_conserved,
        "conserved_liquid_mass_initial": mass0_conserved,
        "conserved_liquid_mass_final": final["conserved_liquid_mass"],
        "wall_height": height,
        "wall_offset_over_dy": float(wall_offset_over_dy),
        # §17: the measured cut-cell advective CFL ratio (dt_global / min_i dt_adv,i) of the run
        "cutcell_advective_cfl_ratio": (
            None
            if not any(r["cutcell_advective_cfl_ratio"] is not None for r in samples)
            else max(r["cutcell_advective_cfl_ratio"] for r in samples if r["cutcell_advective_cfl_ratio"] is not None)
        ),
        "dt_adv_min": (
            None
            if not any(r["dt_adv_min"] is not None for r in samples)
            else min(r["dt_adv_min"] for r in samples if r["dt_adv_min"] is not None)
        ),
        **pf.phase_transport_metadata(p),
        "solid_phase_fraction_max": float(max(r["solid_phase_fraction"] for r in samples)),
        "implicit_iterations_max": int(it_max),
        "implicit_residual_max": float(res_max),
        "implicit_solve_failed": bool(cg_failed),
        "wall_seconds": float(elapsed),
        "samples": _thin(samples, keep_samples),
        "n_samples_total": len(samples),
        "classification": None,
        "classification_evidence": {},
    }
    if checkpoint_out is not None:
        checkpoint_path = Path(checkpoint_out)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_metadata = {
            **checkpoint_signature,
            "steps": int(applied),
            "phase_state_dtype": np.asarray(state.phi).dtype.name,
            "velocity_state_dtype": np.asarray(state.u).dtype.name,
        }
        temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                phi=np.asarray(state.phi),
                u=np.asarray(state.u),
                v=np.asarray(state.v),
                time=np.asarray(state.t),
                metadata_json=np.asarray(json.dumps(checkpoint_metadata, sort_keys=True)),
            )
        temporary.replace(checkpoint_path)
        record["checkpoint_out"] = str(checkpoint_path)
    else:
        record["checkpoint_out"] = None
    return _clean(record)


# --------------------------------------------------------------------------------------
#  sections
# --------------------------------------------------------------------------------------
def _base_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(N=cfg["N"], eps_factor=cfg["eps_factor"], dt=cfg["dt"], R=cfg["R"], sample_every=cfg["sample_every"])


def _log(msg: str) -> None:
    print(f"[l1a2d {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _announce(rec: dict[str, Any]) -> dict[str, Any]:
    _log(
        f"{rec['group']:<13s} {rec['dynamics_mode']:<7s} target={rec['target_deg']:5.1f} N={rec['N']} "
        f"M/Mref={rec['M_over_M_ref']:.1f} dt={rec['dt']:.0e} steps={rec['steps']:>6d} conv={rec['converged']} "
        f"theta={rec['final_sampled_angle_deg']} F={rec['free_energy_final']:.5f} stop={rec['stop_reason']}"
    )
    return rec


def run_section(
    name: str,
    cfg: dict[str, Any],
    *,
    dynamics: str,
    targets: Sequence[float] | None,
    mobility_factors: Sequence[float] | None,
    max_steps: int | None,
) -> list[dict[str, Any]]:
    base = _base_kwargs(cfg)
    budgets = [b for b in cfg["budgets"] if max_steps is None or b <= max_steps] or [
        int(max_steps or cfg["budgets"][0])
    ]
    if max_steps is not None and max_steps not in budgets and max_steps < cfg["budgets"][-1]:
        budgets = sorted(set(budgets + [int(max_steps)]))
    modes = {"ch_only": [True], "chns": [False], "both": [True, False]}[dynamics]
    out: list[dict[str, Any]] = []
    if name == "primary":
        for target in targets if targets is not None else cfg["targets"]:
            for ch_only in modes:
                out.append(
                    _announce(
                        run_relaxation(
                            target, ch_only=ch_only, budgets=budgets, group="primary", label="primary", **base
                        )
                    )
                )
    elif name == "mobility":
        spec = cfg["mobility"]
        factors = mobility_factors if mobility_factors is not None else spec["factors"]
        for target in targets if targets is not None else spec["targets"]:
            for f in factors:
                if abs(float(f) - 1.0) < 1e-12:
                    continue  # M_ref is the primary CH-only case; reused at assembly
                out.append(
                    _announce(
                        run_relaxation(
                            target,
                            ch_only=True,
                            M=M_REF * float(f),
                            budgets=budgets,
                            group="mobility",
                            label=f"M_x{f:g}",
                            **base,
                        )
                    )
                )
    elif name == "dt":
        spec = cfg["dt_sweep"]
        for target in targets if targets is not None else spec["targets"]:
            for ch_only in modes:
                for dt in spec["dts"]:
                    steps = int(round(spec["horizon"] / dt))
                    kw = dict(
                        base,
                        dt=dt,
                        sample_every=max(1, int(round(0.8 / dt)) if cfg["N"] >= 128 else max(1, steps // 8)),
                    )
                    out.append(
                        _announce(
                            run_relaxation(
                                target,
                                ch_only=ch_only,
                                fixed_steps=steps,
                                budgets=[steps],
                                group="dt_sensitivity",
                                label=f"dt_{dt:g}",
                                **kw,
                            )
                        )
                    )
    elif name == "resolution":
        a, b = cfg["resolution_a"], cfg["resolution_b"]
        for target in targets if targets is not None else a["targets"]:
            for ch_only in modes:
                for ef in a["eps_factors"]:
                    kw = dict(base, eps_factor=ef)
                    out.append(
                        _announce(
                            run_relaxation(
                                target,
                                ch_only=ch_only,
                                budgets=[a["budget"]],
                                fixed_steps=a["budget"],
                                group="resolution_A_fixed_N_eps_sweep",
                                label=f"eps_factor_{ef:g}",
                                **kw,
                            )
                        )
                    )
        for target in targets if targets is not None else b["targets"]:
            for ch_only in modes:
                for N in b["N_values"]:
                    kw = dict(base, N=N, eps_factor=b.get("eps_factor", cfg["eps_factor"]))
                    out.append(
                        _announce(
                            run_relaxation(
                                target,
                                ch_only=ch_only,
                                budgets=[b["budget"]],
                                fixed_steps=b["budget"],
                                group="resolution_B_coupled_grid_interface_refinement",
                                label=f"N_{N}",
                                **kw,
                            )
                        )
                    )
    elif name == "gain_ablation":
        spec = cfg["gain_ablation"]
        M = M_REF * float(spec["mobility_factor"])
        for target in targets if targets is not None else spec["targets"]:
            probe = pf.PhaseFieldParams(Nx=base["N"], Ny=base["N"], Lx=6.0, Ly=6.0)
            frac = clk.fluid_wall_delta_integral(np.asarray(pf.surface_flat(probe, 0.25)), probe.dx, probe.dy)
            gain = 1.0 / frac["fluid_side_normal_integral"]
            out.append(
                _announce(
                    run_relaxation(
                        target,
                        ch_only=True,
                        M=M,
                        budgets=budgets,
                        wall_gain=gain,
                        group="wall_gain_ablation",
                        label="gain_1_over_fluid_fraction",
                        **base,
                    )
                )
            )
            for N in spec["N_values"] if target in spec.get("grid_targets", spec["targets"]) else []:
                kw = dict(base, N=N)
                out.append(
                    _announce(
                        run_relaxation(
                            target,
                            ch_only=True,
                            M=M,
                            budgets=budgets,
                            group="quasi_equilibrium_by_grid",
                            label=f"N_{N}_unmodified",
                            **kw,
                        )
                    )
                )
    else:
        raise ValueError(f"unknown section {name!r}")
    return out


# --------------------------------------------------------------------------------------
#  classification / report
# --------------------------------------------------------------------------------------
def _find(cases, **match):
    for c in cases:
        if all(c.get(k) == v for k, v in match.items()):
            return c
    return None


def _mobility_runs(cases: list[dict[str, Any]], target: float) -> list[dict[str, Any]]:
    runs = [
        c
        for c in cases
        if c["target_deg"] == target
        and c["dynamics_mode"] == "ch_only"
        and c["group"] in ("primary", "mobility")
        and c["N"] == cases_N(cases)
        and abs(c["eps_over_dx"] - 2.0) < 1e-9
        and abs(c["dt"] - next(x["dt"] for x in cases if x["group"] == "primary")) < 1e-12
    ]
    return sorted(runs, key=lambda c: c["M"])


def cases_N(cases) -> int:
    return int(next(c["N"] for c in cases if c["group"] == "primary"))


def classify_targets(cases: list[dict[str, Any]], tol: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fill per-case classification (primary/chns, 60/120/150 targets) and return summary + collapse info."""
    summary: dict[str, Any] = {}
    collapse: dict[str, Any] = {}
    for target in sorted({c["target_deg"] for c in cases if c["group"] == "primary"}):
        ch = _find(cases, group="primary", target_deg=target, dynamics_mode="ch_only")
        ns = _find(cases, group="primary", target_deg=target, dynamics_mode="chns")
        runs = _mobility_runs(cases, target)
        coll = clk.evaluate_mobility_collapse(runs) if len(runs) >= 2 else None
        collapse[str(target)] = coll
        ref = ch or ns
        if ref is None:
            continue
        fast = max(runs, key=lambda c: c["M"]) if runs else ref
        fast_err = fast.get("equilibrium_error_deg")
        ablation = _find(cases, group="wall_gain_ablation", target_deg=target)
        unmodified = fast if (ablation is not None and fast["M"] == ablation["M"]) else None
        if unmodified is None and ablation is not None:
            unmodified = next((c for c in runs if c["M"] == ablation["M"]), None)

        def best_error(run):
            if run is None:
                return None
            return run["equilibrium_error_deg"] if run["converged"] else run["final_sampled_error_deg"]

        abl_err = best_error(ablation)
        abl_delta = ((ablation or {}).get("late_trends", {}).get("angle") or {}).get("delta_value")
        abl_stationary = bool(
            ablation is not None and (ablation["converged"] or (abl_delta is not None and abs(abl_delta) <= 0.5))
        )
        # A non-converged ablation that is still approaching the target *from the unmodified side* and already
        # sits inside the acceptance band is reported as such (``ablation_converged`` stays False): it falsifies
        # "the wall term is irrelevant" but is never counted as an equilibrium angle.
        abl_toward = bool(
            ablation is not None and abl_delta is not None and abl_err is not None and abl_delta * (-abl_err) > 0.0
        )
        abl_reaches_band = bool(abl_stationary or abl_toward)
        unm_err = best_error(unmodified)
        dt_runs = [
            c
            for c in cases
            if c["group"] == "dt_sensitivity" and c["target_deg"] == target and c["dynamics_mode"] == "ch_only"
        ]
        dt_spread = None
        if len(dt_runs) >= 2 and all(c["final_sampled_angle_deg"] is not None for c in dt_runs):
            vals = [c["final_sampled_angle_deg"] for c in dt_runs]
            dt_spread = float(max(vals) - min(vals))
        ry_trend = (fast.get("late_trends") or {}).get("RY_l2")
        ry_plateau = bool(
            ry_trend is not None and fast["RY_l2"] and abs(ry_trend["delta_value"]) <= 0.1 * max(fast["RY_l2"], 1e-30)
        )
        wall = ref["wall_kernel"]
        frac = wall["fluid_side_normal_integral"]
        slope = (ref["late_trends"].get("angle") or {}).get("slope", 0.0) or 0.0
        toward = (
            (target - ref["final_sampled_angle_deg"]) * slope > 0
            if ref["final_sampled_angle_deg"] is not None
            else False
        )
        near_ch = bool(fast["converged"] and abs(fast["equilibrium_error_deg"]) <= tol)
        near_chns = bool(ns is not None and ns["converged"] and abs(ns["equilibrium_error_deg"]) <= tol)
        evidence = {
            "detachment_observed": any(c["detachment_observed"] for c in (ch, ns) if c),
            "contact_line_exists": all(c["contact_line_exists"] for c in (ch, ns) if c),
            "converged": bool(fast["converged"]),
            "equilibrium_error_deg": fast_err,
            "RY_normalized_l2": fast["RY_normalized_l2"] or 0.0,
            "mt_curves_collapse": bool(coll and coll["mt_curves_collapse"]),
            "dt_angle_spread_deg": dt_spread,
            "ch_only_converged_near_target": near_ch if ns is not None else None,
            "chns_converged_near_target": near_chns if ns is not None else None,
            "relaxing_toward_target": bool(toward),
            "wall_kernel_fluid_fraction": frac,
            "gain_ablation_restores_target": bool(
                abl_err is not None
                and unm_err is not None
                and abl_reaches_band
                and abs(abl_err) <= tol
                and abs(unm_err) > tol
            ),
            "ablation_stationary": abl_stationary,
            "ablation_converged": bool(ablation is not None and ablation["converged"]),
            "ablation_still_approaching_target": abl_toward,
            "ablation_equilibrium_error_deg": abl_err,
            "unmodified_equilibrium_error_deg": unm_err,
            "transient_kinetics_limited": bool(coll and coll["mt_curves_collapse"]),
            "ry_plateau": ry_plateau,
            "chns_dt_note": "CHNS dt-dependence is reported separately; eta_pen = 2 dt couples Brinkman to dt",
        }
        if abs(target - 90.0) < 1e-9:
            summary[str(target)] = {"classification": None, "note": "neutral control", "evidence": evidence}
            continue
        label, details = clk.classify_equilibration(evidence, acceptance_tol_deg=tol)
        summary[str(target)] = {"classification": label, "classification_details": details, "evidence": evidence}
        for c in (ch, ns):
            if c is not None:
                c["classification"] = label
                c["classification_evidence"] = {
                    "target_level_rule": details.get("rule"),
                    **{
                        k: evidence[k]
                        for k in (
                            "mt_curves_collapse",
                            "RY_normalized_l2",
                            "wall_kernel_fluid_fraction",
                            "gain_ablation_restores_target",
                            "dt_angle_spread_deg",
                        )
                    },
                }
    return summary, collapse


def recommend_next_stage(summary: dict[str, Any]) -> dict[str, Any]:
    labels = {k: v["classification"] for k, v in summary.items() if v.get("classification")}
    order = {
        "BOUNDARY_DISCRETIZATION_LIMITED": (
            "L1A-2e: embedded-boundary Young flux assembly (wall-kernel weighting)",
            "Outcome B (discrete Young equilibrium / diffuse wall formulation)",
        ),
        "THERMODYNAMIC_EQUILIBRIUM_BIASED": (
            "L1A-2e: discrete Young-equilibrium / diffuse-interface wall formulation",
            "Outcome B",
        ),
        "HYDRODYNAMIC_COUPLING_LIMITED": ("L1A-2e: contact-line hydrodynamics / Brinkman coupling audit", "Outcome A"),
        "KINETICS_LIMITED": ("mobility / accelerated-equilibrium strategy", "Outcome C"),
        "RESOLUTION_LIMITED": ("grid / Cahn-number convergence", "Outcome D"),
        "TIME_STEP_LIMITED": ("temporal discretization / adaptive dt", "Outcome E"),
    }
    chosen = None
    for key in order:
        if key in labels.values():
            chosen = key
            break
    if chosen is None:
        return {
            "stage": "L1A-2e",
            "decision": "INCONCLUSIVE",
            "decision_tree_outcome": None,
            "rationale": "no mechanism isolated; extend budgets or instrumentation before changing a subsystem",
            "per_target_labels": labels,
            "single_subsystem": None,
        }
    title, outcome = order[chosen]
    return {
        "stage": "L1A-2e",
        "decision": title,
        "decision_tree_outcome": outcome,
        "driving_label": chosen,
        "per_target_labels": labels,
        "single_subsystem": title,
        "note": "evidence-based; production defaults (M, Brinkman, wall energy gain) are NOT changed by L1A-2d",
    }


def _unresolved_blockers() -> list[dict[str, Any]]:
    from production.validation import KNOWN_SOLVER_BLOCKERS

    wanted = {
        "W-CONTACT-ANGLE",
        "P-SOLID-PIN",
        "I-CONTACT-GAP",
        "N-DT",
        "P-VARDENS-PROJ",
        "P-CAP-RHO",
        "P-VARVISC",
        "BC-Y-PERIODIC",
        "D-FRESH-TRAIN-CONTRACT",
    }
    rows = [
        {"id": b["id"], "status": b["status"], "closed_by_l1a2d": False}
        for b in KNOWN_SOLVER_BLOCKERS
        if b["id"] in wanted
    ]
    rows.append(
        {
            "id": "N-CG-MASS-DRIFT (new observation, not in KNOWN_SOLVER_BLOCKERS)",
            "status": "open",
            "closed_by_l1a2d": False,
            "description": "fluid-region mass drifts linearly with steps at the float32 CG rtol "
            "(> 1e-3 beyond ~18k steps)",
        }
    )
    return rows


def _spread(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return float(max(values) - min(values)) if len(values) >= 2 else None


def sweep_summaries(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Machine-readable answers to the dt / eps / grid / kernel-prediction questions."""
    out: dict[str, Any] = {}

    def angle(c):
        return c["equilibrium_angle_deg"] if c["converged"] else c["final_sampled_angle_deg"]

    dt = [c for c in cases if c["group"] == "dt_sensitivity"]
    out["dt_sensitivity_fixed_physical_time"] = {
        f"{t:g}/{m}": {
            "dts": [c["dt"] for c in dt if c["target_deg"] == t and c["dynamics_mode"] == m],
            "final_angle_deg": [angle(c) for c in dt if c["target_deg"] == t and c["dynamics_mode"] == m],
            "free_energy": [c["free_energy_final"] for c in dt if c["target_deg"] == t and c["dynamics_mode"] == m],
            "RY_normalized_l2": [c["RY_normalized_l2"] for c in dt if c["target_deg"] == t and c["dynamics_mode"] == m],
            "late_angle_slope": [
                c["late_trends"]["angle"]["slope"] for c in dt if c["target_deg"] == t and c["dynamics_mode"] == m
            ],
            "angle_spread_deg": _spread([angle(c) for c in dt if c["target_deg"] == t and c["dynamics_mode"] == m]),
        }
        for t in sorted({c["target_deg"] for c in dt})
        for m in ("ch_only", "chns")
        if any(c["target_deg"] == t and c["dynamics_mode"] == m for c in dt)
    }
    for key, group in (
        ("resolution_A_fixed_N_eps_sweep", "resolution_A_fixed_N_eps_sweep"),
        ("resolution_B_coupled_grid_interface_refinement", "resolution_B_coupled_grid_interface_refinement"),
    ):
        rows = [c for c in cases if c["group"] == group]
        out[key] = {
            f"{t:g}/{m}": [
                {
                    "N": c["N"],
                    "eps": c["eps"],
                    "eps_over_dx": c["eps_over_dx"],
                    "Cn": c["Cn"],
                    "angle_deg_at_budget": angle(c),
                    "steps": c["steps"],
                    "free_energy": c["free_energy_final"],
                    "RY_normalized_l2": c["RY_normalized_l2"],
                    "wall_kernel_fluid_fraction": c["wall_kernel"]["fluid_side_normal_integral"],
                }
                for c in rows
                if c["target_deg"] == t and c["dynamics_mode"] == m
            ]
            for t in sorted({c["target_deg"] for c in rows})
            for m in ("ch_only", "chns")
            if any(c["target_deg"] == t and c["dynamics_mode"] == m for c in rows)
        }
    # Converged CH-only states at the fastest mobility, per grid: observed vs fluid-fraction prediction.
    pred = []
    for c in cases:
        fast = c["group"] in ("mobility", "quasi_equilibrium_by_grid") and c["dynamics_mode"] == "ch_only"
        if fast and c["converged"] and c["wall_gain_diagnostic"] == 1.0 and c["M_over_M_ref"] >= 4.0:
            f = c["wall_kernel"]["fluid_side_normal_integral"]
            cos_pred = f * math.cos(math.radians(c["target_deg"]))
            pred.append(
                {
                    "target_deg": c["target_deg"],
                    "N": c["N"],
                    "wall_kernel_fluid_fraction": f,
                    "observed_equilibrium_deg": c["equilibrium_angle_deg"],
                    "kernel_prediction_deg": float(math.degrees(math.acos(max(-1.0, min(1.0, cos_pred))))),
                    "prediction_note": "diagnostic only: theta = acos(f cos theta_Young); "
                    "never used as an equilibrium angle",
                }
            )
    out["fluid_fraction_prediction_vs_converged_ch_only"] = sorted(pred, key=lambda r: (r["target_deg"], r["N"]))
    return out


def compare_ch_only_vs_chns(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Per target: CH-only (M_ref, final / fastest converged) against full CHNS at identical settings."""
    out = {}
    for target in sorted({c["target_deg"] for c in cases if c["group"] == "primary"}):
        ch = _find(cases, group="primary", target_deg=target, dynamics_mode="ch_only")
        ns = _find(cases, group="primary", target_deg=target, dynamics_mode="chns")
        fast = [c for c in cases if c["group"] == "mobility" and c["target_deg"] == target and c["converged"]]
        fast = max(fast, key=lambda c: c["M"]) if fast else None

        def row(c):
            if c is None:
                return None
            return {
                k: c.get(k)
                for k in (
                    "steps",
                    "mobility_scaled_time",
                    "converged",
                    "equilibrium_angle_deg",
                    "final_sampled_angle_deg",
                    "contact_width",
                    "free_energy_final",
                    "RY_l2",
                    "RY_normalized_l2",
                    "phase_rate_l2",
                    "final_max_speed",
                    "E_kin_final",
                    "brinkman_dissipation_proxy_final",
                    "E_total_final",
                )
            }

        out[f"{target:g}"] = {"ch_only_M_ref": row(ch), "ch_only_fastest_converged": row(fast), "chns_M_ref": row(ns)}
    return out


def _observations(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Cross-case facts that are not mechanism labels but constrain how the evidence may be used."""
    gate = 1.0e-3
    over = [c for c in cases if c["mass_drift"] > gate]
    rates = [c["mass_drift_final"] / max(c["steps"], 1) * 1000.0 for c in cases if c["steps"] > 0]
    kernels = {}
    for c in cases:
        kernels[str(c["N"])] = c["wall_kernel"]["fluid_side_normal_integral"]
    neutral = [c for c in cases if c["target_deg"] == 90.0 and c["converged"]]
    return {
        "fluid_mass_drift_gate": gate,
        "cases_exceeding_mass_drift_gate": len(over),
        "max_fluid_mass_drift": max((c["mass_drift"] for c in cases), default=None),
        "fluid_mass_drift_per_1000_steps_max": max(rates, default=None),
        "mass_drift_interpretation": (
            "systematic, linear-in-steps loss at the float32 CG rtol (1e-6); a one-off float64 / rtol=1e-8 probe "
            "(production/README.md section H) gave a ~100x smaller drift, so it tracks the implicit-solve "
            "tolerance and is not a conservation-form defect"
        ),
        "fluid_side_wall_kernel_fraction_by_N": kernels,
        "neutral_control_angle_offset_deg": [c["equilibrium_angle_deg"] - 90.0 for c in neutral],
        # L1A-2e: the first-fluid-layer residual replaces the band diagnostic as the
        # discriminating Young-BC metric; both are kept so the two can be compared.
        "wall_measure_method_by_case": sorted({str(c.get("wall_measure_method")) for c in cases}),
        "first_layer_residual_by_measure": {
            str(measure): [c.get("RY_first_normalized_l2") for c in cases if c.get("wall_measure_method") == measure]
            for measure in sorted({str(c.get("wall_measure_method")) for c in cases})
        },
        "max_first_layer_residual_normalized": max(
            (c.get("RY_first_normalized_l2") or 0.0 for c in cases), default=None
        ),
        "max_wall_area_relative_error": max((c.get("wall_area_relative_error") or 0.0 for c in cases), default=None),
    }


def assemble_report(
    sections: dict[str, list[dict[str, Any]]], profile: str, cfg: dict[str, Any], manufactured: dict[str, Any]
) -> dict[str, Any]:
    cases = [c for name in SECTIONS for c in sections.get(name, [])]
    tol = CRITERIA["acceptance_error_deg"]
    summary: dict[str, Any] = {}
    collapse: dict[str, Any] = {}
    if any(c["group"] == "primary" for c in cases):
        summary, collapse = classify_targets(cases, tol)
        if profile == "quick":
            # 10^2-step budgets cannot establish any mechanism: the smoke profile exercises the pipeline only.
            for row in summary.values():
                if row.get("classification"):
                    row["smoke_profile_computed_label"] = row["classification"]
                    row["classification"] = "INCONCLUSIVE"
                    row["classification_details"] = {"rule": "smoke_profile_budgets_too_short"}
            for case in cases:
                if case["classification"] is not None:
                    case["classification"] = "INCONCLUSIVE"
    return {
        "stage": STAGE,
        "profile": profile,
        "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
        **pf.phase_transport_metadata(pf.PhaseFieldParams(Nx=2, Ny=2)),
        "trajectory_semantics_changed": False,
        "git_sha": get_git_sha(),
        "criteria": dict(CRITERIA),
        "settings": {k: v for k, v in cfg.items()},
        "historical_v7_baseline": HISTORICAL_V7_BASELINE,
        "manufactured_young_residual_audit": manufactured,
        "cases": cases,
        "mobility_collapse": collapse,
        "classification_summary": summary,
        "recommended_next_stage": recommend_next_stage(summary),
        "unresolved_blockers": _unresolved_blockers(),
        "observations": _observations(cases),
        "sweep_summaries": sweep_summaries(cases),
        "ch_only_vs_chns": compare_ch_only_vs_chns(cases),
        "notes": [
            "Non-converged sampled angles are never equilibrium angles; extrapolated theta_inf is diagnostic only.",
            "A run that converges to a wrong angle is converged=true with a signed equilibrium_error_deg.",
            "wall_gain_ablation scales cos(theta) handed to the unchanged solver; it is a falsification "
            "experiment, not a calibration.",
        ],
    }


REQUIRED_CASE_FIELDS = (
    "target_deg",
    "dynamics_mode",
    "N",
    "dx",
    "eps",
    "eps_over_dx",
    "Cn",
    "M",
    "dt",
    "steps",
    "physical_time",
    "mobility_scaled_time",
    "converged",
    "equilibrium_angle_deg",
    "final_sampled_angle_deg",
    "equilibrium_error_deg",
    "detachment_observed",
    "contact_line_exists",
    "free_energy_initial",
    "free_energy_final",
    "free_energy_monotonic_violations",
    "phase_rate_l2",
    "phase_rate_linf",
    "RY_l2",
    "RY_linf",
    "RY_normalized_l2",
    "contact_width",
    "contact_line_speed",
    "late_trends",
    "asymptotic_fit",
    "classification",
    "classification_evidence",
    "mass_drift",
    "solid_phase_fraction_max",
    "implicit_iterations_max",
    "implicit_residual_max",
    # L1A-2e additions: precision / wall-measure provenance and the first-fluid-layer residual
    "dtype",
    "ch_solver_rtol",
    "wall_measure_method",
    "wall_offset_cells",
    "wall_area_total",
    "wall_area_relative_error",
    "n_wall_cells",
    "RY_first_l2",
    "RY_first_normalized_l2",
    "wall_measure_weighted_RY",
)
REQUIRED_TOP = (
    "stage",
    "solver_contract_version",
    "trajectory_semantics_changed",
    "git_sha",
    "cases",
    "classification_summary",
    "recommended_next_stage",
    "unresolved_blockers",
)


def validate_report(report: dict[str, Any]) -> list[str]:
    """Schema + honesty checks; returns all errors."""
    errors = [f"missing top-level field {k}" for k in REQUIRED_TOP if k not in report]
    if errors:
        return errors
    if report["stage"] != STAGE:
        errors.append("stage must be L1A-2d")
    # The diagnostic audit records the live contract, never silently labels a current trajectory
    # with a historical solver version. Contracts 7-10 remain recognized for frozen evidence.
    if report["solver_contract_version"] not in (7, 8, 9, 10, 11):
        errors.append(f"solver_contract_version must be 7, 8, 9, 10 or 11; got {report['solver_contract_version']!r}")
    if report["solver_contract_version"] == 11:
        expected_storage = {
            "phase_storage_model": pf.PHASE_ONLY_FLOAT64_STORAGE_MODEL,
            "phase_state_dtype": "float64",
            "velocity_state_dtype": "float32",
        }
        for key, value in expected_storage.items():
            if report.get(key) != value:
                errors.append(f"contract-11 report {key} must be {value!r}")
    if report["solver_contract_version"] == 8:
        # a contract-8 report cannot claim the v9 transport geometry
        for key in ("phase_transport_geometry", "phase_control_volume", "phase_face_aperture"):
            if key in report and report[key] not in (None, "hard_cell_v7", "hard_cell_volume", "binary_face_mask"):
                errors.append(f"a contract-8 report cannot claim {key}={report[key]!r}")
    if report["solver_contract_version"] != int(pf.SOLVER_CONTRACT_VERSION):
        errors.append("solver_contract_version must match the live solver contract")
    if report["trajectory_semantics_changed"] is not False:
        errors.append("trajectory_semantics_changed must be false for the L1A-2d diagnostic stage")
    for i, case in enumerate(report["cases"]):
        for key in REQUIRED_CASE_FIELDS:
            if key not in case:
                errors.append(f"cases[{i}] missing {key}")
        if case.get("converged") is False and case.get("equilibrium_angle_deg") is not None:
            errors.append(f"cases[{i}] reports an equilibrium angle without converging")
        if case.get("classification") not in (None, *clk.ALLOWED_CLASSIFICATIONS):
            errors.append(f"cases[{i}] has an unknown classification")
        if case.get("detachment_observed") and case.get("equilibrium_angle_deg") is not None:
            errors.append(f"cases[{i}] reports a sessile angle after detachment")
        if case.get("wall_measure_method") not in pf.WALL_MEASURE_METHODS:
            errors.append(f"cases[{i}] has an unknown wall_measure_method")
        if case.get("wall_area_relative_error") is not None and case["wall_area_relative_error"] > 1.0e-6:
            errors.append(f"cases[{i}] wall measure does not equal the geometric wall length")
    try:
        json.dumps(report, allow_nan=False)
    except ValueError as exc:
        errors.append(f"not strict JSON: {exc}")
    return errors


def format_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['stage']} non-neutral equilibration audit (profile={report['profile']})",
        "",
        f"- solver contract = {report['solver_contract_version']}, "
        f"trajectory semantics changed = {report['trajectory_semantics_changed']}, git = {report['git_sha'][:12]}",
        "",
    ]
    lines += [
        "| group | mode | target | N | M/Mref | dt | steps | converged | final/eq angle | err | F | RY_norm | stop |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for c in report["cases"]:
        ang = c["equilibrium_angle_deg"] if c["converged"] else c["final_sampled_angle_deg"]
        err = c["equilibrium_error_deg"] if c["converged"] else c["final_sampled_error_deg"]
        f = lambda v, n=3: "n/a" if v is None else f"{v:.{n}f}"
        lines.append(
            f"| {c['group']}/{c['label']} | {c['dynamics_mode']} | {c['target_deg']:.0f} | {c['N']} | "
            f"{c['M_over_M_ref']:.1f} | {c['dt']:.0e} | {c['steps']} | {c['converged']} | {f(ang)} | {f(err)} | "
            f"{f(c['free_energy_final'], 5)} | {f(c['RY_normalized_l2'])} | {c['stop_reason']} |"
        )
    lines += ["", "## Classification", ""]
    for t, row in report["classification_summary"].items():
        lines.append(
            f"- {t} deg: **{row.get('classification')}** "
            f"({row.get('classification_details', {}).get('rule', row.get('note'))})"
        )
    lines += ["", f"## Next stage: {report['recommended_next_stage']['decision']}", ""]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------
#  CLI
# --------------------------------------------------------------------------------------
def run_audit(
    profile: str,
    out: Path,
    *,
    dynamics: str = "both",
    targets: Sequence[float] | None = None,
    mobility_factors: Sequence[float] | None = None,
    max_steps: int | None = None,
    sections: Sequence[str] = SECTIONS,
    overwrite: bool = False,
) -> dict[str, Any]:
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {sorted(PROFILES)}")
    cfg = PROFILES[profile]
    if out.exists() and overwrite and set(sections) == set(SECTIONS):
        shutil.rmtree(out)
    cache = out / "sections"
    cache.mkdir(parents=True, exist_ok=True)
    manufactured = clk.audit_young_boundary_residual()
    if not manufactured["passed"]:
        raise RuntimeError("manufactured Young-residual audit failed; fix the diagnostic before using trajectories")
    loaded: dict[str, list[dict[str, Any]]] = {}
    for name in SECTIONS:
        path = cache / f"{name}.json"
        if name in sections:
            if path.exists() and not overwrite:
                _log(f"section {name}: cached")
                loaded[name] = json.loads(path.read_text())
                continue
            _log(f"section {name}: running")
            loaded[name] = run_section(
                name, cfg, dynamics=dynamics, targets=targets, mobility_factors=mobility_factors, max_steps=max_steps
            )
            path.write_text(json.dumps(loaded[name], allow_nan=False))
        elif path.exists():
            loaded[name] = json.loads(path.read_text())
    report = assemble_report(loaded, profile, cfg, manufactured)
    errors = validate_report(report)
    if errors:
        raise RuntimeError("invalid L1A-2d report: " + "; ".join(errors[:10]))
    (out / "nonneutral_equilibration_report.json").write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    (out / "nonneutral_equilibration_report.md").write_text(format_markdown(report))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="L1A-2d non-neutral equilibration / contact-line kinetics audit")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="quick")
    parser.add_argument("--targets", type=float, nargs="+", default=None)
    parser.add_argument("--dynamics", choices=("ch_only", "chns", "both"), default="both")
    parser.add_argument("--mobility-factors", type=float, nargs="+", default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="cap the staged budget")
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=SECTIONS,
        default=list(SECTIONS),
        help="run only these sections (cached sections are merged into the report)",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    report = run_audit(
        args.profile,
        args.out,
        dynamics=args.dynamics,
        targets=args.targets,
        mobility_factors=args.mobility_factors,
        max_steps=args.max_steps,
        sections=args.sections,
        overwrite=args.overwrite,
    )
    print(format_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
