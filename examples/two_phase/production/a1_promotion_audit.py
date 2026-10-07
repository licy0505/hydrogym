"""Stage-2 production verification for the frozen L1A-2i A1 phase-storage choice.

This audit does not select among candidates. It reproduces the selected A1 quick gate, audits the
production default's dtype boundary, checks strict restart and schema-3 sample lineage, and measures
CPU/GPU execution separately. Physics and wetting closure remain independent evidence tracks: the
report never upgrades them merely because the A1 quick gate passes.

Run from ``examples/two_phase`` with x64 enabled before importing JAX::

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python -m production.a1_promotion_audit \\
        --out artifacts/l1a2i_stage2/promotion --overwrite

A completed four-angle production-default CHNS report from ``production.run_validation`` may be
attached with ``--chns-report``. The full physics readiness gate requires an explicit, artifact-backed
``--physics-evidence`` JSON; unprovided measurements remain NOT_RUN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import phasefield as pf
from production import mass_precision_audit as mpa
from production import phase_storage_precision_audit as psa
from production.dataset_lineage import (
    DATASET_SAMPLE_CAST_POLICY,
    DATASET_SAMPLE_REPRESENTATION,
    DATASET_SCHEMA_VERSION,
    sample_lineage_metadata,
    validate_training_sample_lineage,
)
from production.restart import load_checkpoint, save_checkpoint

HERE = Path(__file__).resolve().parent.parent
STAGE1_MATRIX = HERE / "evidence" / "l1a2i" / "candidate_matrix.json"
QUICK_OFFSETS = (0.0, 0.5)
QUICK_STEPS = 2500
QUICK_SAMPLE_EVERY = 50
PHYSICS_REQUIRED_GATES = (
    "ch_only_quick",
    "ch_only_medium",
    "four_angle_ch_only_equilibrium",
    "energy_dissipation",
    "alignment_and_grid_sensitivity",
    "impact_regression",
    "laplace_scaling",
    "capillary_sign_and_energy_alignment",
)


def _jsonable(value: Any) -> Any:
    """Convert NumPy/JAX values to strict JSON primitives."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.ndarray, jax.Array)):
        array = np.asarray(value)
        return _jsonable(array.tolist() if array.ndim else array.item())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _jsonable(float(value))
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_info() -> dict[str, str]:
    def git(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=HERE, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    return {"branch": git("branch", "--show-current"), "git_sha": git("rev-parse", "HEAD")}


def _stage1_quick_reference() -> dict[str, float | None]:
    """Read the frozen Stage-1 A1 values, without making them a substitute for this run."""
    if not STAGE1_MATRIX.is_file():
        return {f"quick_offset{offset}": None for offset in QUICK_OFFSETS}
    try:
        report = json.loads(STAGE1_MATRIX.read_text(encoding="utf-8"))
        row = next(
            item for item in report.get("candidate_rows", []) if item.get("candidate") == "A1_phase_float64"
        )
        quick = row.get("quick") or {}
        return {
            "quick_offset0.0": quick.get("quick_offset0.0"),
            "quick_offset0.5": quick.get("quick_offset0.5"),
        }
    except (OSError, ValueError, StopIteration, TypeError):
        return {f"quick_offset{offset}": None for offset in QUICK_OFFSETS}


def _run_quick_reproduction(public_steps: int = QUICK_STEPS) -> dict[str, Any]:
    """Re-run the frozen N=48 A1 quick mass gate and precision ledger at both wall offsets."""
    if int(public_steps) != QUICK_STEPS:
        raise ValueError(f"the promotion quick gate is frozen at {QUICK_STEPS} public steps")
    try:
        cpu = jax.devices("cpu")[0]
    except (IndexError, RuntimeError) as exc:
        raise RuntimeError("the frozen Stage-1 quick reproduction is a CPU gate, but no JAX CPU device is available") from exc
    rows: dict[str, Any] = {}
    reference = _stage1_quick_reference()
    for offset in QUICK_OFFSETS:
        with jax.default_device(cpu):
            payload = psa.drift_series(
                N=48,
                target_deg=150.0,
                M_factor=4.0,
                public_steps=QUICK_STEPS,
                sample_every=QUICK_SAMPLE_EVERY,
                candidate="A1_phase_float64",
                wall_offset_over_dy=offset,
            )
        classification = psa.series_classification(
            payload["masses"], payload["E_round"], payload["sample_every"]
        )
        measured = float(classification["final_relative_drift"])
        passed = abs(measured) <= float(psa.QUICK_GATE)
        strong = abs(measured) <= float(psa.QUICK_GATE_STRONG)
        label = f"quick_offset{offset:.1f}"
        reference_value = reference.get(label)
        rows[label] = {
            "N": 48,
            "target_contact_angle_deg": 150.0,
            "public_steps_requested": QUICK_STEPS,
            "public_steps_executed": int(payload["public_steps"]),
            "substeps": int(payload["substeps"]),
            "sample_every_public_steps": QUICK_SAMPLE_EVERY,
            "last_sampled_public_step": int(classification["steps"]),
            "wall_offset_over_dy": offset,
            "storage_model": pf.PHASE_ONLY_FLOAT64_STORAGE_MODEL,
            "phase_dtype": "float64",
            "velocity_dtype": "float32",
            "initial_fluid_mass": float(payload["m0"]),
            "final_relative_mass_drift": measured,
            "quick_bound": float(psa.QUICK_GATE),
            "quick_passed": bool(passed),
            "strong_bound": float(psa.QUICK_GATE_STRONG),
            "strong_passed": bool(strong),
            "stage1_reference_final_relative_drift": reference_value,
            "absolute_delta_from_stage1_reference": (
                None if reference_value is None else abs(measured - float(reference_value))
            ),
            "series_classification": classification,
            "precision_ledger": payload["ledger"],
            "angle_deg": float(payload["angle_deg"]),
            "max_abs_phi": float(payload["final_phi_max"]),
        }
    return {
        "status": "PASS" if all(row["quick_passed"] for row in rows.values()) else "FAIL",
        "gate": {
            "N": 48,
            "target_contact_angle_deg": 150.0,
            "public_steps": QUICK_STEPS,
            "wall_offsets_over_dy": list(QUICK_OFFSETS),
            "quick_bound": float(psa.QUICK_GATE),
            "strong_bound": float(psa.QUICK_GATE_STRONG),
            "M_factor_vs_mass_precision_reference": 4.0,
            "sample_every_public_steps": QUICK_SAMPLE_EVERY,
        },
        "rows": rows,
        "stage1_reference_artifact": str(STAGE1_MATRIX.relative_to(HERE)) if STAGE1_MATRIX.exists() else None,
        "interpretation": (
            "The final reported drift is the last frozen sampled mass point, matching the Stage-1 "
            "classification convention; the run still executes all 2500 public steps."
        ),
    }


def _run_medium_reproduction() -> dict[str, Any]:
    """Reproduce A1's frozen N=128, 60/150-degree, 5000-step CH-only medium gate."""
    try:
        cpu = jax.devices("cpu")[0]
    except (IndexError, RuntimeError) as exc:
        raise RuntimeError("the A1 medium reproduction requires a JAX CPU device") from exc
    rows: dict[str, Any] = {}
    for target in (60.0, 150.0):
        with jax.default_device(cpu):
            payload = psa.drift_series(
                N=128,
                target_deg=target,
                M_factor=4.0,
                public_steps=5000,
                sample_every=100,
                candidate="A1_phase_float64",
                wall_offset_over_dy=0.0,
            )
        classification = psa.series_classification(
            payload["masses"], payload["E_round"], payload["sample_every"]
        )
        measured = float(classification["final_relative_drift"])
        rows[f"medium_{int(target)}deg"] = {
            "N": 128,
            "target_contact_angle_deg": target,
            "public_steps_requested": 5000,
            "public_steps_executed": int(payload["public_steps"]),
            "substeps": int(payload["substeps"]),
            "last_sampled_public_step": int(classification["steps"]),
            "M_factor": 4.0,
            "wall_offset_over_dy": 0.0,
            "final_relative_mass_drift": measured,
            "medium_bound": float(psa.MEDIUM_GATE),
            "medium_passed": abs(measured) <= float(psa.MEDIUM_GATE),
            "strong_bound": float(psa.MEDIUM_GATE_STRONG),
            "strong_passed": abs(measured) <= float(psa.MEDIUM_GATE_STRONG),
            "series_classification": classification,
            "precision_ledger": payload["ledger"],
            "angle_deg": float(payload["angle_deg"]),
        }
    return {
        "status": "PASS" if all(row["medium_passed"] for row in rows.values()) else "FAIL",
        "gate": {
            "N": 128,
            "targets_deg": [60.0, 150.0],
            "public_steps": 5000,
            "sample_every_public_steps": 100,
            "medium_bound": float(psa.MEDIUM_GATE),
            "strong_bound": float(psa.MEDIUM_GATE_STRONG),
        },
        "rows": rows,
    }


def _dtype_name(value: Any) -> str:
    return np.asarray(value).dtype.name


def run_dtype_audit() -> dict[str, Any]:
    """Trace representative CH, geometry, momentum and projection intermediates at runtime."""
    if not bool(jax.config.x64_enabled):
        raise RuntimeError("JAX x64 is disabled; contract-11 A1 must fail closed, not fall back")
    p = pf.PhaseFieldParams(Nx=32, Ny=32, Lx=6.0, Ly=6.0, dt=4.0e-3, eps=2.0 * 6.0 / 32)
    wall_height = 0.25
    solid = pf.make_solid(pf.surface_flat(p, wall_height=wall_height), p, cos_theta=math.cos(math.radians(150.0)))
    state = pf.sessile_initial_state(p, solid, R=1.1, wall_height=wall_height, theta0_deg=150.0)
    dt = p.dt / 3.0
    operator = pf.phase_transport_operator(solid, p)
    phi_rhs, u_rhs, v_rhs, mu, mu_exp = pf.rhs(state, solid, p)
    adv_x, adv_y = pf.phase_advective_fluxes(state.u, state.v, state.phi, solid, p)
    ch_x, ch_y = pf.chemical_potential_fluxes(mu_exp, solid, p)
    adv_div = pf.control_volume_divergence(adv_x, adv_y, operator.volume_safe)
    ch_div = pf.control_volume_divergence(ch_x, ch_y, operator.volume_safe)
    implicit_rhs = state.phi + dt * (phi_rhs - ch_div)
    solved, solve_info = pf.solve_ch_implicit(implicit_rhs, solid, p, dt)
    correction = solved - implicit_rhs

    # Representative initial CG vectors and one matrix-free operator application. The recurrence
    # follows these arrays' dtype for every iteration; production does not allocate a lower-precision
    # Krylov copy under A1.
    cg_x0 = jnp.zeros_like(implicit_rhs)
    cg_r0 = implicit_rhs - cg_x0
    cg_p0 = cg_r0
    lap_p = pf.graph_stiffness_apply(cg_p0, operator.weight_x, operator.weight_y) / operator.volume_safe
    lap2_p = pf.graph_stiffness_apply(lap_p, operator.weight_x, operator.weight_y) / operator.volume_safe
    alpha = jnp.asarray(dt * p.M * p.eps, dtype=implicit_rhs.dtype)
    cg_Ap0 = cg_p0 + alpha * lap2_p

    bulk_mu = pf.fprime(state.phi) / p.eps
    wall_mu = pf.wall_energy_derivative(state.phi, solid.cos_theta) * pf.wall_measure_density(solid, p)
    rho = pf.rho_of(state.phi, p)
    viscosity = pf.nu_of(state.phi, p)
    grad_x = pf._ddx(state.phi, p.dx)
    grad_y = pf._ddy(state.phi, p.dy)
    capillary_x = (pf.SIGMA_NORM / p.We) * mu * grad_x / p.rho_l
    capillary_y = (pf.SIGMA_NORM / p.We) * mu * grad_y / p.rho_l

    momentum_dt = jnp.asarray(dt, dtype=p.dtype)
    damping = 1.0 / (1.0 + momentum_dt * solid.chi / p.eta_pen)
    u_predictor = (state.u + momentum_dt * u_rhs) * damping
    v_predictor = (state.v + momentum_dt * v_rhs) * damping
    pressure = pf.poisson_solve(
        (pf._ddx(u_predictor, p.dx) + pf._ddy(v_predictor, p.dy)) / momentum_dt,
        p.m2_proj,
    )
    u_projected = u_predictor - momentum_dt * pf._ddx(pressure, p.dx)
    v_projected = v_predictor - momentum_dt * pf._ddy(pressure, p.dy)
    stepped, diagnostics = pf.step_with_diagnostics(state, solid, p)

    observed: dict[str, Any] = {
        "persistent_phase_state": state.phi,
        "persistent_velocity_u": state.u,
        "persistent_velocity_v": state.v,
        "persistent_time": state.t,
        "explicit_phase_rhs": phi_rhs,
        "advective_phase_flux_x": adv_x,
        "advective_phase_flux_y": adv_y,
        "chemical_potential_flux_x": ch_x,
        "chemical_potential_flux_y": ch_y,
        "cutcell_advective_divergence": adv_div,
        "cutcell_ch_divergence": ch_div,
        "bulk_chemical_potential_derivative": bulk_mu,
        "wall_chemical_potential_derivative": wall_mu,
        "chemical_potential": mu,
        "explicit_chemical_potential": mu_exp,
        "implicit_rhs": implicit_rhs,
        "cg_initial_solution": cg_x0,
        "cg_initial_residual": cg_r0,
        "cg_initial_search_direction": cg_p0,
        "cg_operator_application": cg_Ap0,
        "implicit_solution": solved,
        "implicit_correction": correction,
        "density": rho,
        "kinematic_viscosity": viscosity,
        "phase_gradient_x": grad_x,
        "phase_gradient_y": grad_y,
        "capillary_force_x": capillary_x,
        "capillary_force_y": capillary_y,
        "momentum_rhs_u": u_rhs,
        "momentum_rhs_v": v_rhs,
        "momentum_predictor_u": u_predictor,
        "momentum_predictor_v": v_predictor,
        "pressure_projection": pressure,
        "projected_velocity_u": u_projected,
        "projected_velocity_v": v_projected,
        "cutcell_volume": operator.volume,
        "cutcell_volume_safe": operator.volume_safe,
        "cutcell_aperture_x": operator.aperture_x,
        "cutcell_aperture_y": operator.aperture_y,
        "cutcell_centroid_x": solid.geometry.centroid_x,
        "cutcell_centroid_y": solid.geometry.centroid_y,
        "cutcell_wall_measure": solid.wall_area,
        "cutcell_wall_normal_x": solid.wall_normal_x,
        "cutcell_wall_normal_y": solid.wall_normal_y,
        "production_step_phase_output": stepped.phi,
        "production_step_velocity_u_output": stepped.u,
        "production_step_velocity_v_output": stepped.v,
        "production_step_time_output": stepped.t,
    }
    float64_names = {
        "persistent_phase_state",
        "explicit_phase_rhs",
        "advective_phase_flux_x",
        "advective_phase_flux_y",
        "chemical_potential_flux_x",
        "chemical_potential_flux_y",
        "cutcell_advective_divergence",
        "cutcell_ch_divergence",
        "bulk_chemical_potential_derivative",
        "wall_chemical_potential_derivative",
        "chemical_potential",
        "explicit_chemical_potential",
        "implicit_rhs",
        "cg_initial_solution",
        "cg_initial_residual",
        "cg_initial_search_direction",
        "cg_operator_application",
        "implicit_solution",
        "implicit_correction",
        "density",
        "kinematic_viscosity",
        "phase_gradient_x",
        "phase_gradient_y",
        "capillary_force_x",
        "capillary_force_y",
        "production_step_phase_output",
    }
    float32_names = {
        "persistent_velocity_u",
        "persistent_velocity_v",
        "persistent_time",
        "momentum_rhs_u",
        "momentum_rhs_v",
        "momentum_predictor_u",
        "momentum_predictor_v",
        "pressure_projection",
        "projected_velocity_u",
        "projected_velocity_v",
        "production_step_velocity_u_output",
        "production_step_velocity_v_output",
        "production_step_time_output",
        "cutcell_volume",
        "cutcell_volume_safe",
        "cutcell_aperture_x",
        "cutcell_aperture_y",
        "cutcell_centroid_x",
        "cutcell_centroid_y",
        "cutcell_wall_measure",
        "cutcell_wall_normal_x",
        "cutcell_wall_normal_y",
    }
    dtypes = {name: _dtype_name(value) for name, value in observed.items()}
    expected = {**{name: "float64" for name in float64_names}, **{name: "float32" for name in float32_names}}
    mismatches = {
        name: {"actual": dtypes[name], "expected": dtype}
        for name, dtype in expected.items()
        if dtypes[name] != dtype
    }
    solve_converged = bool(np.asarray(solve_info.converged))
    step_converged = bool(np.all(np.asarray(diagnostics.implicit_converged)))
    return {
        "status": "PASS" if not mismatches and solve_converged and step_converged else "FAIL",
        "solver_contract_version": pf.SOLVER_CONTRACT_VERSION,
        "phase_storage_model": p.phase_storage_model,
        "phase_state_dtype": dtypes["persistent_phase_state"],
        "velocity_state_dtype": dtypes["persistent_velocity_u"],
        "dtypes": dtypes,
        "expected_dtypes": expected,
        "mismatches": mismatches,
        "implicit_solve": {
            "converged": solve_converged,
            "iterations": int(np.asarray(solve_info.iterations)),
            "relative_residual": float(np.asarray(solve_info.relative_residual)),
        },
        "production_step": {
            "all_implicit_solves_converged": step_converged,
            "iterations": np.asarray(diagnostics.implicit_iterations).tolist(),
            "relative_residuals": np.asarray(diagnostics.implicit_relative_residuals).tolist(),
        },
        "scope": "runtime arrays; geometry and velocities remain float32 while all phase/CH/CG/capillary arrays remain float64",
    }


def run_dataset_lineage_audit() -> dict[str, Any]:
    """Exercise current/stale schema-3 lineage through both generator and surrogate guards."""
    from generate_dataset import _saved_case_is_current
    import surrogate as surrogate_module

    params = pf.PhaseFieldParams(Nx=2, Ny=2)
    metadata = {**pf.phase_transport_metadata(params), **sample_lineage_metadata(params)}
    validate_training_sample_lineage(metadata, phi_dtype="float32")

    with tempfile.TemporaryDirectory(prefix="hydrogym-contract11-dataset-") as temp:
        current_path = Path(temp) / "current.npz"
        np.savez(
            current_path,
            dataset_schema_version=np.asarray(DATASET_SCHEMA_VERSION, dtype=np.int32),
            dataset_fingerprint=np.asarray("current-fingerprint"),
            case=np.asarray(json.dumps(metadata, sort_keys=True)),
            phi=np.zeros((1, 2, 2), dtype=np.float32),
        )
        current_generator_accepts = _saved_case_is_current(current_path, "current-fingerprint")
        with np.load(current_path, allow_pickle=False) as archive:
            surrogate_module._require_current_dataset(archive, str(current_path))

        stale_path = Path(temp) / "stale-v10.npz"
        stale_metadata = dict(metadata)
        stale_metadata.update(solver_contract_version=10, phase_storage_model=pf.LEGACY_FLOAT32_STORAGE_MODEL)
        np.savez(
            stale_path,
            dataset_schema_version=np.asarray(DATASET_SCHEMA_VERSION, dtype=np.int32),
            dataset_fingerprint=np.asarray("stale-fingerprint"),
            case=np.asarray(json.dumps(stale_metadata, sort_keys=True)),
            phi=np.zeros((1, 2, 2), dtype=np.float32),
        )
        stale_generator_rejects = not _saved_case_is_current(stale_path, "stale-fingerprint")
        with np.load(stale_path, allow_pickle=False) as archive:
            try:
                surrogate_module._require_current_dataset(archive, str(stale_path))
            except RuntimeError:
                stale_surrogate_rejects = True
            else:
                stale_surrogate_rejects = False

        wrong_cast_path = Path(temp) / "wrong-cast.npz"
        np.savez(
            wrong_cast_path,
            dataset_schema_version=np.asarray(DATASET_SCHEMA_VERSION, dtype=np.int32),
            dataset_fingerprint=np.asarray("wrong-cast"),
            case=np.asarray(json.dumps(metadata, sort_keys=True)),
            phi=np.zeros((1, 2, 2), dtype=np.float64),
        )
        wrong_cast_rejected = not _saved_case_is_current(wrong_cast_path, "wrong-cast")

    passed = current_generator_accepts and stale_generator_rejects and stale_surrogate_rejects and wrong_cast_rejected
    return {
        "status": "PASS" if passed else "FAIL",
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "dataset_schema_decision": {
            "selected_version": 3,
            "layout_changed": False,
            "rationale": (
                "Existing NPZ fields and tensor shapes are unchanged. The phi array is a derived "
                "training observable, downsample-mean then cast once to float32; it is not restart-authoritative."
            ),
        },
        "solver_phase_storage_model": pf.PHASE_ONLY_FLOAT64_STORAGE_MODEL,
        "solver_phase_dtype": "float64",
        "stored_phi_dtype": "float32",
        "sample_cast_policy": DATASET_SAMPLE_CAST_POLICY,
        "sample_representation": DATASET_SAMPLE_REPRESENTATION,
        "current_generator_accepts": bool(current_generator_accepts),
        "current_surrogate_reader_accepts": True,
        "contract10_generator_rejects": bool(stale_generator_rejects),
        "contract10_surrogate_reader_rejects": bool(stale_surrogate_rejects),
        "wrong_sample_dtype_rejects": bool(wrong_cast_rejected),
        "case_metadata": metadata,
        "restart_authoritative": False,
    }


def _advance_rollout(state: pf.State, solid: pf.Solid, p: pf.PhaseFieldParams, steps: int):
    run = jax.jit(lambda initial: pf.rollout(initial, solid, p, int(steps), save_every=1))
    result = run(state)
    jax.block_until_ready(result[0].phi)
    return result


def run_restart_smoke(*, N: int = 48, first: int = 2, second: int = 2) -> dict[str, Any]:
    """Round-trip a strict contract-11 checkpoint and compare a split run with an uninterrupted run."""
    p = pf.PhaseFieldParams(Nx=N, Ny=N, Lx=6.0, Ly=6.0, dt=2.0e-3, eps=2.0 * 6.0 / N)
    wall_height = 0.25
    solid = pf.make_solid(pf.surface_flat(p, wall_height=wall_height), p, cos_theta=0.0)
    initial = pf.sessile_initial_state(p, solid, R=1.1, wall_height=wall_height, theta0_deg=90.0)
    uninterrupted, full_phi, full_u, full_v = _advance_rollout(initial, solid, p, first + second)
    first_state, first_phi, first_u, first_v = _advance_rollout(initial, solid, p, first)

    with tempfile.TemporaryDirectory(prefix="hydrogym-contract11-restart-") as temp:
        checkpoint_path = Path(temp) / "state.npz"
        saved_metadata = save_checkpoint(checkpoint_path, first_state, p, step_index=first)
        restored, loaded_metadata = load_checkpoint(checkpoint_path, expected_params=p)
        restarted, second_phi, second_u, second_v = _advance_rollout(restored, solid, p, second)

        # Mutate only the contract marker. Strict v10 -> v11 restart must reject, not cast/promote.
        stale_path = Path(temp) / "stale-v10.npz"
        with np.load(checkpoint_path, allow_pickle=False) as archive:
            stale_arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
        stale_meta = json.loads(str(stale_arrays["metadata"].item()))
        stale_meta["solver_contract_version"] = 10
        stale_arrays["metadata"] = np.asarray(json.dumps(stale_meta, sort_keys=True, separators=(",", ":")))
        np.savez_compressed(stale_path, **stale_arrays)
        try:
            load_checkpoint(stale_path, expected_params=p)
        except RuntimeError as exc:
            stale_v10_rejected = "stale solver checkpoint contract" in str(exc)
        else:
            stale_v10_rejected = False

    phi_difference = float(np.max(np.abs(np.asarray(restarted.phi) - np.asarray(uninterrupted.phi))))
    u_difference = float(np.max(np.abs(np.asarray(restarted.u) - np.asarray(uninterrupted.u))))
    v_difference = float(np.max(np.abs(np.asarray(restarted.v) - np.asarray(uninterrupted.v))))
    hist_phi = np.concatenate([np.asarray(first_phi), np.asarray(second_phi)], axis=0)
    hist_u = np.concatenate([np.asarray(first_u), np.asarray(second_u)], axis=0)
    hist_v = np.concatenate([np.asarray(first_v), np.asarray(second_v)], axis=0)
    mass_hist_split = np.asarray(
        jnp.sum(jnp.asarray(hist_phi) * pf.phase_control_volumes(solid, p)[None, :, :], axis=(1, 2))
    )
    mass_hist_full = np.asarray(
        jnp.sum(full_phi * pf.phase_control_volumes(solid, p)[None, :, :], axis=(1, 2))
    )
    angle_split: list[float | None] = []
    angle_full: list[float | None] = []
    angle_error: str | None = None
    try:
        for a, b in zip(hist_phi, np.asarray(full_phi)):
            angle_split.append(float(pf.measure_contact_angle(a, solid, p)))
            angle_full.append(float(pf.measure_contact_angle(b, solid, p)))
    except Exception as exc:  # keep the failure visible; it blocks the restart equivalence claim
        angle_error = f"{type(exc).__name__}: {exc}"
        angle_split = []
        angle_full = []
    angle_equal = bool(
        len(angle_split) == len(angle_full) == first + second
        and np.array_equal(np.asarray(angle_split), np.asarray(angle_full))
    )
    result_checks = {
        "phase_state_exact": phi_difference == 0.0,
        "velocity_u_exact": u_difference == 0.0,
        "velocity_v_exact": v_difference == 0.0,
        "phase_history_exact": np.array_equal(hist_phi, np.asarray(full_phi)),
        "velocity_history_exact": np.array_equal(hist_u, np.asarray(full_u))
        and np.array_equal(hist_v, np.asarray(full_v)),
        "mass_history_exact": np.array_equal(mass_hist_split, mass_hist_full),
        "angle_history_exact": angle_equal,
        "stale_v10_checkpoint_rejected": bool(stale_v10_rejected),
        "phase_dtype_float64": np.asarray(restored.phi).dtype.name == "float64",
        "velocity_dtype_float32": np.asarray(restored.u).dtype.name == "float32"
        and np.asarray(restored.v).dtype.name == "float32",
    }
    passed = all(result_checks.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "checkpoint_schema_version": saved_metadata["checkpoint_schema_version"],
        "restart_state_version": saved_metadata["restart_state_version"],
        "solver_contract_version": loaded_metadata["solver_contract_version"],
        "phase_storage_model": loaded_metadata["phase_storage_model"],
        "solver_phase_dtype": np.asarray(restored.phi).dtype.name,
        "velocity_dtype": np.asarray(restored.u).dtype.name,
        "grid": {"Nx": N, "Ny": N},
        "step_split": {"first": first, "second": second, "total": first + second},
        "max_abs_differences": {
            "phi": phi_difference,
            "u": u_difference,
            "v": v_difference,
        },
        "mass_history_relative_drift": (
            0.0
            if mass_hist_full.size == 0
            else float((mass_hist_full[-1] - mass_hist_full[0]) / max(abs(mass_hist_full[0]), 1e-30))
        ),
        "mass_history_split": mass_hist_split.tolist(),
        "mass_history_uninterrupted": mass_hist_full.tolist(),
        "angle_history_split_deg": angle_split,
        "angle_history_uninterrupted_deg": angle_full,
        "angle_measurement_error": angle_error,
        "checks": result_checks,
    }


def run_cpu_benchmark(*, N: int = 128, steps: int = 20, repeats: int = 5) -> dict[str, Any]:
    """Benchmark the actual public CHNS step under legacy float32 and production A1 storage on CPU."""
    try:
        devices = jax.devices("cpu")
    except RuntimeError:
        devices = []
    if not devices:
        return {"status": "UNAVAILABLE", "reason": "no JAX CPU device"}
    device = devices[0]
    rows: dict[str, Any] = {}
    with jax.default_device(device):
        for model in (pf.LEGACY_FLOAT32_STORAGE_MODEL, pf.PHASE_ONLY_FLOAT64_STORAGE_MODEL):
            p = pf.PhaseFieldParams(
                Nx=N,
                Ny=N,
                Lx=6.0,
                Ly=6.0,
                phase_storage_model=model,
            )
            solid = pf.make_solid(
                pf.surface_flat(p, wall_height=0.25), p, cos_theta=math.cos(math.radians(150.0))
            )
            initial = pf.sessile_initial_state(p, solid, R=1.1, wall_height=0.25, theta0_deg=150.0)

            def rollout(state: pf.State):
                def body(carry, _):
                    nxt, diag = pf.step_with_diagnostics(carry, solid, p)
                    return nxt, (jnp.all(diag.implicit_converged), jnp.max(diag.implicit_relative_residuals))

                return jax.lax.scan(body, state, None, length=int(steps))

            compiled = jax.jit(rollout)
            warm, warm_diag = compiled(initial)
            jax.block_until_ready(warm.phi)
            if not bool(np.all(np.asarray(warm_diag[0]))):
                raise RuntimeError(f"CPU benchmark solve did not converge for {model}")
            timings: list[float] = []
            for _ in range(int(repeats)):
                start = time.perf_counter()
                result, diagnostics = compiled(initial)
                jax.block_until_ready(result.phi)
                timings.append(time.perf_counter() - start)
            if not bool(np.all(np.asarray(diagnostics[0]))):
                raise RuntimeError(f"CPU benchmark solve did not converge for {model}")
            persistent_bytes = sum(np.asarray(getattr(initial, name)).nbytes for name in ("phi", "u", "v", "t"))
            rows[model] = {
                "device": str(device),
                "steps_per_repeat": int(steps),
                "repeats": int(repeats),
                "elapsed_seconds_samples": timings,
                "median_seconds": float(statistics.median(timings)),
                "median_seconds_per_public_step": float(statistics.median(timings) / steps),
                "public_steps_per_second": float(steps / statistics.median(timings)),
                "persistent_state_bytes": int(persistent_bytes),
                "phase_dtype": np.asarray(initial.phi).dtype.name,
                "velocity_dtype": np.asarray(initial.u).dtype.name,
                "all_implicit_solves_converged": bool(np.all(np.asarray(diagnostics[0]))),
            }
    legacy = rows[pf.LEGACY_FLOAT32_STORAGE_MODEL]
    a1 = rows[pf.PHASE_ONLY_FLOAT64_STORAGE_MODEL]
    return {
        "status": "MEASURED",
        "benchmark": {
            "grid_N": int(N),
            "public_steps": int(steps),
            "warmup_repeats": 1,
            "timed_repeats": int(repeats),
            "model": "actual jitted pf.step_with_diagnostics CHNS public-step path",
            "physical_defaults": "PhaseFieldParams defaults except N, Lx=Ly=6, target=150 deg",
        },
        "rows": rows,
        "a1_over_legacy_runtime_ratio": a1["median_seconds"] / legacy["median_seconds"],
        "a1_over_legacy_persistent_state_bytes_ratio": (
            a1["persistent_state_bytes"] / legacy["persistent_state_bytes"]
        ),
        "temporary_peak_memory": "not measured by the portable CPU benchmark",
    }


def _device_record(device: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "platform": str(device.platform),
        "device_kind": str(getattr(device, "device_kind", "unknown")),
        "id": int(getattr(device, "id", -1)),
        "process_index": int(getattr(device, "process_index", lambda: 0)())
        if callable(getattr(device, "process_index", None))
        else int(getattr(device, "process_index", 0)),
    }
    try:
        stats = device.memory_stats()
        if stats:
            data["memory_stats"] = {str(key): int(value) for key, value in stats.items() if isinstance(value, (int, np.integer))}
    except Exception:
        data["memory_stats"] = None
    return data


def run_gpu_qualification(*, N: int = 192, steps: int = 2000, repeats: int = 1) -> dict[str, Any]:
    """If present, qualify an A1 N=192/2000-step production-default CHNS rollout on each GPU."""
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        devices = []
    if not devices:
        return {
            "status": "UNAVAILABLE",
            "reason": "JAX reports no GPU device; no GPU performance or scale claim is made",
            "device_count": 0,
            "target": {"N": int(N), "public_steps": int(steps), "storage_model": pf.PHASE_ONLY_FLOAT64_STORAGE_MODEL},
        }
    rows: list[dict[str, Any]] = []
    for device in devices:
        try:
            with jax.default_device(device):
                p = pf.PhaseFieldParams(Nx=N, Ny=N, Lx=6.0, Ly=6.0)
                solid = pf.make_solid(
                    pf.surface_flat(p, wall_height=0.25), p, cos_theta=math.cos(math.radians(150.0))
                )
                initial = pf.sessile_initial_state(p, solid, R=1.1, wall_height=0.25, theta0_deg=150.0)

                def rollout(state: pf.State):
                    def body(carry, _):
                        nxt, diag = pf.step_with_diagnostics(carry, solid, p)
                        return nxt, (
                            jnp.all(diag.implicit_converged),
                            jnp.max(diag.implicit_relative_residuals),
                        )

                    return jax.lax.scan(body, state, None, length=int(steps))

                compiled = jax.jit(rollout)
                # Warm compile and verify before timing production steps.
                warm_state, warm_diag = compiled(initial)
                jax.block_until_ready(warm_state.phi)
                if not bool(np.all(np.asarray(warm_diag[0]))):
                    raise RuntimeError("warmup contains a nonconverged implicit solve")
                times: list[float] = []
                result = warm_state
                diagnostics = warm_diag
                for _ in range(int(repeats)):
                    start = time.perf_counter()
                    result, diagnostics = compiled(initial)
                    jax.block_until_ready(result.phi)
                    times.append(time.perf_counter() - start)
                all_converged = bool(np.all(np.asarray(diagnostics[0])))
                finite = all(
                    bool(np.isfinite(np.asarray(getattr(result, name))).all())
                    for name in ("phi", "u", "v", "t")
                )
                mass_initial = float(pf.liquid_mass(initial.phi, solid, p))
                mass_final = float(pf.liquid_mass(result.phi, solid, p))
                mass_drift = (mass_final - mass_initial) / max(abs(mass_initial), 1e-30)
                row_passed = all_converged and finite and abs(mass_drift) <= 1.0e-3
                rows.append(
                    {
                        "device": _device_record(device),
                        "status": "PASS" if row_passed else "FAIL",
                        "elapsed_seconds_samples": times,
                        "median_seconds": float(statistics.median(times)),
                        "public_steps_per_second": float(steps / statistics.median(times)),
                        "all_implicit_solves_converged": all_converged,
                        "finite_final_state": finite,
                        "fluid_mass_relative_drift": mass_drift,
                        "max_implicit_relative_residual": float(np.max(np.asarray(diagnostics[1]))),
                        "phase_dtype": np.asarray(result.phi).dtype.name,
                        "velocity_dtype": np.asarray(result.u).dtype.name,
                        "state_shape": list(result.phi.shape),
                    }
                )
        except Exception as exc:
            rows.append(
                {
                    "device": _device_record(device),
                    "status": "FAIL",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    all_passed = bool(rows) and all(row.get("status") == "PASS" for row in rows)
    return {
        "status": "PASS" if all_passed else "FAIL",
        "target": {
            "N": int(N),
            "public_steps": int(steps),
            "repeats": int(repeats),
            "storage_model": pf.PHASE_ONLY_FLOAT64_STORAGE_MODEL,
            "velocity_dtype": "float32",
            "dataset_resolution_reference": 192,
            "acceptance": "all CH solves converged, final fields finite, abs fluid-mass drift <= 1e-3",
        },
        "devices": rows,
    }


def _validated_evidence_file(path_text: str, root: Path) -> tuple[bool, str | None]:
    path = Path(path_text)
    if not path.is_absolute():
        path = root / path
    try:
        resolved = path.resolve(strict=True)
        return True, _sha256(resolved)
    except (OSError, ValueError):
        return False, None


def load_physics_evidence(path: str | None, *, root: Path) -> dict[str, Any]:
    """Verify that every required physics gate has a PASS row backed by readable artifacts."""
    if not path:
        return {
            "status": "NOT_RUN",
            "required_gates": list(PHYSICS_REQUIRED_GATES),
            "missing_gates": list(PHYSICS_REQUIRED_GATES),
            "artifact_hashes": {},
        }
    source = Path(path)
    if not source.is_absolute():
        source = root / source
    report = json.loads(source.read_text(encoding="utf-8"))
    gates = report.get("gates") if isinstance(report, dict) else None
    if not isinstance(gates, dict):
        return {"status": "BLOCKED", "reason": "physics evidence JSON must contain a gates object"}
    evidence: dict[str, Any] = {}
    missing: list[str] = []
    failed: list[str] = []
    artifact_hashes: dict[str, str] = {}
    for gate in PHYSICS_REQUIRED_GATES:
        row = gates.get(gate)
        if not isinstance(row, dict) or row.get("status") not in {"PASS", "FAIL"}:
            missing.append(gate)
            continue
        artifacts = row.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            missing.append(gate)
            continue
        artifact_rows: list[dict[str, str]] = []
        artifacts_valid = True
        for item in artifacts:
            ok, digest = _validated_evidence_file(str(item), source.parent)
            if not ok or digest is None:
                artifacts_valid = False
                break
            artifact_rows.append({"path": str(item), "sha256": digest})
        if not artifacts_valid:
            missing.append(gate)
            continue
        evidence[gate] = {"status": row["status"], "artifacts": artifact_rows, "measurement": row.get("measurement")}
        artifact_hashes.update({item["path"]: item["sha256"] for item in artifact_rows})
        if row["status"] != "PASS":
            failed.append(gate)
    status = "PASS" if not missing and not failed else "FAIL" if failed else "NOT_RUN"
    return {
        "status": status,
        "evidence_file": str(source),
        "evidence_file_sha256": _sha256(source),
        "required_gates": list(PHYSICS_REQUIRED_GATES),
        "missing_gates": missing,
        "failed_gates": failed,
        "gates": evidence,
        "artifact_hashes": artifact_hashes,
    }


def _merge_stage2_mass_gates(
    physics: dict[str, Any],
    quick: dict[str, Any],
    medium: dict[str, Any],
    quick_artifact_path: Path,
    medium_artifact_path: Path,
) -> dict[str, Any]:
    """Fold fresh Stage-2 quick and medium measurements into the physics evidence ledger."""
    result = dict(physics)
    gates = dict(result.get("gates") or {})
    local_hashes: dict[str, str] = {}
    for name, measurement, path in (
        ("ch_only_quick", quick, quick_artifact_path),
        ("ch_only_medium", medium, medium_artifact_path),
    ):
        artifact = {"path": str(path), "sha256": _sha256(path)}
        has_evidence = measurement.get("status") in {"PASS", "FAIL"}
        gates[name] = {
            "status": measurement.get("status", "NOT_RUN"),
            "artifacts": [artifact] if has_evidence else [],
            "measurement": measurement,
        }
        if has_evidence:
            local_hashes[str(path)] = artifact["sha256"]
    missing = [
        gate
        for gate in PHYSICS_REQUIRED_GATES
        if gate not in gates or not gates[gate].get("artifacts")
    ]
    failed = [
        gate for gate in PHYSICS_REQUIRED_GATES
        if gate in gates and gates[gate].get("status") == "FAIL" and gates[gate].get("artifacts")
    ]
    result.update(
        {
            "status": "PASS" if not missing and not failed else "FAIL" if failed else "NOT_RUN",
            "required_gates": list(PHYSICS_REQUIRED_GATES),
            "missing_gates": missing,
            "failed_gates": failed,
            "gates": gates,
            "artifact_hashes": {
                **(result.get("artifact_hashes") or {}),
                **local_hashes,
            },
        }
    )
    return result


def load_chns_closure(path: str | None, *, root: Path) -> dict[str, Any]:
    """Assess staged four-angle CHNS acceptance from an emitted production validation report."""
    if not path:
        return {
            "status": "NOT_RUN",
            "profile": None,
            "required_targets_deg": [60.0, 90.0, 120.0, 150.0],
            "acceptance": None,
            "evidence": None,
        }
    source = Path(path)
    if not source.is_absolute():
        source = root / source
    report = json.loads(source.read_text(encoding="utf-8"))
    from production.run_validation import _contact_angle_acceptance

    benchmarks = report.get("benchmarks", {})
    accepted, evidence = _contact_angle_acceptance(benchmarks)
    profile = (report.get("config") or {}).get("profile")
    contact_cases = benchmarks.get("contact_angle", {}).get("cases", [])
    production_scale_rows = [
        {"runtime": row.get("runtime", {}), "converged": row.get("converged") is True}
        for row in contact_cases
        if isinstance(row, dict) and isinstance(row.get("runtime"), dict)
    ]
    scale_ok = len(production_scale_rows) == 4 and all(
        int(row["runtime"].get("N", 0)) >= 128
        and int(row["runtime"].get("max_steps", 0)) >= 3000
        and (int(row["runtime"].get("steps", 0)) >= 3000 or row["converged"])
        for row in production_scale_rows
    )
    # Closure is a baseline-profile staged CHNS result, not a CI one-angle smoke or a tiny test fixture.
    ready = bool(accepted and profile == "baseline" and scale_ok)
    if ready:
        blocker = None
    elif not evidence.get("complete_target_set"):
        blocker = "a complete 60/90/120/150-degree CHNS target set is required"
    elif profile != "baseline" or not scale_ok:
        blocker = "a baseline-profile report with four N>=128 A1 CHNS runs of at least 3000 steps is required"
    else:
        failed_checks = [
            name
            for name in ("all_finite", "all_converged", "monotonic", "mass_ok", "v7_boundary_model_ok", "wall_measure_ok", "storage_lineage_ok")
            if evidence.get(name) is not True
        ]
        blocker = "measured four-angle CHNS acceptance failed: " + ", ".join(failed_checks or ["angle tolerances"])
    return {
        "status": "PASS" if ready else "FAIL" if evidence.get("complete_target_set") else "NOT_RUN",
        "profile": profile,
        "source_report": str(source),
        "source_report_sha256": _sha256(source),
        "required_targets_deg": [60.0, 90.0, 120.0, 150.0],
        "acceptance": bool(accepted),
        "production_scale_rows_ok": bool(scale_ok),
        "evidence": evidence,
        "blocker": blocker,
    }


def _status_label(passed: bool) -> str:
    return "PASS" if passed else "BLOCKED"


def build_stage2_report(
    *,
    quick: dict[str, Any],
    dtype: dict[str, Any],
    dataset: dict[str, Any],
    restart: dict[str, Any],
    chns: dict[str, Any],
    physics: dict[str, Any],
    performance: dict[str, Any],
    gpu: dict[str, Any],
    medium: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract_passed = all(
        section.get("status") == "PASS" for section in (quick, dtype, dataset, restart)
    )
    wetting_status = chns.get("status", "NOT_RUN")
    physics_status = physics.get("status", "NOT_RUN")
    medium = medium or {"status": "NOT_RUN", "reason": "medium gate was not run"}
    if medium.get("status") == "FAIL":
        physics_status = "FAIL"
    elif medium.get("status") != "PASS" and physics_status == "PASS":
        physics_status = "NOT_RUN"
    gpu_status = "PASS" if gpu.get("status") == "PASS" else (
        "UNAVAILABLE" if gpu.get("status") == "UNAVAILABLE" else "BLOCKED"
    )
    return {
        "stage": "L1A-2i Stage 2",
        "solver_contract_version": pf.SOLVER_CONTRACT_VERSION,
        "production_storage_model": pf.PHASE_STORAGE_MODEL,
        "readiness": {
            "CONTRACT11_LINEAGE_READY": _status_label(contract_passed),
            "PHYSICS_NUMERICS_READY": physics_status,
            "WETTING_CLOSURE_READY": wetting_status,
            "GPU_PRODUCTION_SCALE_READY": gpu_status,
        },
        "evidence_status": {
            "quick_reproduction": quick.get("status"),
            "dtype_audit": dtype.get("status"),
            "dataset_lineage": dataset.get("status"),
            "restart_equivalence": restart.get("status"),
            "ch_only_medium": medium.get("status"),
            "chns_closure": chns.get("status"),
            "cpu_performance": performance.get("status"),
            "gpu_qualification": gpu.get("status"),
            "physics_numerics": physics.get("status"),
        },
        "blockers": {
            "physics_numerics": sorted(
                set(physics.get("missing_gates", [])) | set(physics.get("failed_gates", []))
            ),
            "wetting_closure": chns.get("blocker"),
            "gpu_scale": gpu.get("reason") if gpu.get("status") == "UNAVAILABLE" else None,
        },
        "quick_reproduction": quick,
        "dtype_audit": dtype,
        "dataset_lineage": dataset,
        "restart_equivalence": restart,
        "medium_reproduction": medium,
        "chns_closure": chns,
        "physics_numerics": physics,
        "performance": performance,
        "gpu_qualification": gpu,
        "notes": [
            "Candidate selection is frozen at A1; this report does not reconsider A2/B1/C1.",
            "Passing contract-11 lineage, dtype, restart and quick checks is not physics readiness.",
            "Dataset schema 3 is retained only because the NPZ field layout is unchanged; the stored phase sample is a derived float32 observable, not restart state.",
            "CPU performance and GPU production-scale qualification are reported separately from physics/numerics readiness.",
        ],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _markdown_summary(report: dict[str, Any]) -> str:
    lines = [
        "# L1A-2i Stage 2 A1 promotion report",
        "",
        f"- Solver contract: **{report['solver_contract_version']}**",
        f"- Production storage model: **{report['production_storage_model']}**",
        "",
        "## Independent readiness statuses",
        "",
        "| Status | Result |",
        "|---|---|",
    ]
    for name, status in report["readiness"].items():
        lines.append(f"| `{name}` | **{status}** |")
    lines.extend(["", "## Evidence summary", "", "| Gate | Result |", "|---|---|"])
    for name, status in report["evidence_status"].items():
        lines.append(f"| {name.replace('_', ' ')} | {status} |")
    quick = report["quick_reproduction"]
    lines.extend(["", "## N=48 / 150° / 2500-step A1 quick reproduction", ""])
    lines.extend(
        [
            "| Wall offset (dy) | Public steps executed | Last sampled step | Relative fluid-mass drift | Quick gate | Strong gate | Stage-1 reference |",
            "|---:|---:|---:|---:|---|---|---:|",
        ]
    )
    for row in quick.get("rows", {}).values():
        ref = row.get("stage1_reference_final_relative_drift")
        lines.append(
            f"| {row['wall_offset_over_dy']:.1f} | {row['public_steps_executed']} | "
            f"{row['last_sampled_public_step']} | {row['final_relative_mass_drift']:.9g} | "
            f"{row['quick_passed']} | {row['strong_passed']} | {ref if ref is not None else 'not found'} |"
        )
    medium = report.get("medium_reproduction", {})
    lines.extend(["", "## N=128 CH-only medium reproduction", ""])
    lines.extend(
        [
            "| Target | Public steps executed | Last sampled step | Relative mass drift | Bound | Gate |",
            "|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in medium.get("rows", {}).values():
        lines.append(
            f"| {row['target_contact_angle_deg']:.0f}° | {row['public_steps_executed']} | "
            f"{row['last_sampled_public_step']} | {row['final_relative_mass_drift']:.9g} | "
            f"{row['medium_bound']:.3g} | {row['medium_passed']} |"
        )
    lines.extend(["", "## Blockers", ""])
    physics_missing = report["blockers"].get("physics_numerics") or []
    if physics_missing:
        lines.append("Physics/numerics missing or failed gates: " + ", ".join(physics_missing) + ".")
    chns_blocker = report["blockers"].get("wetting_closure")
    if chns_blocker:
        lines.append(f"CHNS wetting closure: {chns_blocker}.")
    gpu_blocker = report["blockers"].get("gpu_scale")
    if gpu_blocker:
        lines.append(f"GPU scale: {gpu_blocker}.")
    lines.extend(["", "## Precision ledger", ""])
    for label, row in quick.get("rows", {}).items():
        ledger = row.get("precision_ledger", {})
        lines.append(
            f"- `{label}`: solve defect mass = `{ledger.get('solve_defect_mass')}`, "
            f"storage loss = `{ledger.get('storage_loss_mass')}`, "
            f"explained relative drift = `{ledger.get('explained_relative_drift')}`."
        )
    lines.append("")
    return "\n".join(lines)


def run_stage2_audit(
    output_dir: str | Path,
    *,
    chns_report_path: str | None = None,
    physics_evidence_path: str | None = None,
    run_benchmark: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    out = Path(output_dir)
    if not out.is_absolute():
        out = HERE / out
    if out.exists() and not overwrite:
        raise FileExistsError(f"output directory already exists: {out}; pass --overwrite")
    out.mkdir(parents=True, exist_ok=True)

    errors: dict[str, str] = {}
    sections: dict[str, dict[str, Any]] = {}
    operations = {
        "quick": lambda: _run_quick_reproduction(),
        "dtype": run_dtype_audit,
        "dataset": run_dataset_lineage_audit,
        "restart": run_restart_smoke,
        "medium": _run_medium_reproduction,
        "chns": lambda: load_chns_closure(chns_report_path, root=HERE),
        "physics": lambda: load_physics_evidence(physics_evidence_path, root=HERE),
        "performance": run_cpu_benchmark if run_benchmark else lambda: {"status": "NOT_RUN", "reason": "disabled"},
        "gpu": run_gpu_qualification,
    }
    for name, operation in operations.items():
        try:
            sections[name] = operation()
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
            sections[name] = {"status": "ERROR", "error": errors[name]}

    quick_path = out / "quick_reproduction_report.json"
    medium_path = out / "medium_reproduction_report.json"
    _write_json(quick_path, sections["quick"])
    _write_json(medium_path, sections["medium"])
    sections["physics"] = _merge_stage2_mass_gates(
        sections["physics"], sections["quick"], sections["medium"], quick_path, medium_path
    )
    report = build_stage2_report(
        quick=sections["quick"],
        dtype=sections["dtype"],
        dataset=sections["dataset"],
        restart=sections["restart"],
        chns=sections["chns"],
        physics=sections["physics"],
        performance=sections["performance"],
        gpu=sections["gpu"],
        medium=sections["medium"],
    )
    if errors:
        report["errors"] = errors
    artifacts = {
        "a1_promotion_report.json": report,
        "dataset_lineage_report.json": sections["dataset"],
        "restart_report.json": sections["restart"],
        "quick_reproduction_report.json": sections["quick"],
        "medium_reproduction_report.json": sections["medium"],
        "chns_closure_report.json": sections["chns"],
        "physics_numerics_report.json": sections["physics"],
        "performance_report.json": sections["performance"],
        "gpu_qualification_report.json": sections["gpu"],
    }
    for filename, payload in artifacts.items():
        _write_json(out / filename, payload)
    (out / "a1_promotion_report.md").write_text(_markdown_summary(report), encoding="utf-8")

    source_files = (
        "phasefield.py",
        "generate_dataset.py",
        "surrogate.py",
        "production/a1_promotion_audit.py",
        "production/capillary_audit.py",
        "production/cutcell_alignment_audit.py",
        "production/dataset_lineage.py",
        "production/embedded_young_audit.py",
        "production/mass_precision_audit.py",
        "production/nonneutral_wetting_audit.py",
        "production/phase_boundary_audit.py",
        "production/phase_storage_precision_audit.py",
        "production/restart.py",
        "production/run_validation.py",
        "production/validation.py",
        "production/wall_measure_audit.py",
    )

    def _input_record(path_text: str | None) -> dict[str, str | None] | None:
        if not path_text:
            return None
        path = Path(path_text)
        if not path.is_absolute():
            path = HERE / path
        try:
            resolved = path.resolve(strict=True)
            return {"path": str(resolved), "sha256": _sha256(resolved)}
        except (OSError, ValueError):
            return {"path": str(path), "sha256": None}

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "L1A-2i Stage 2",
        "command": " ".join(sys.argv),
        "git": _git_info(),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "jax_version": getattr(jax, "__version__", "unknown"),
            "jax_enable_x64": bool(jax.config.x64_enabled),
            "jax_devices": [_device_record(device) for device in jax.devices()],
        },
        "input_reports": {
            "stage1_candidate_matrix": {
                "path": str(STAGE1_MATRIX),
                "sha256": _sha256(STAGE1_MATRIX) if STAGE1_MATRIX.is_file() else None,
            },
            "chns_report": _input_record(chns_report_path),
            "physics_evidence": _input_record(physics_evidence_path),
            "baseline_config": {
                "path": str(HERE / "production/configs/baseline.json"),
                "sha256": _sha256(HERE / "production/configs/baseline.json")
                if (HERE / "production/configs/baseline.json").is_file()
                else None,
            },
        },
        "source_hashes": {
            relative: _sha256(HERE / relative)
            for relative in source_files
            if (HERE / relative).is_file()
        },
        "arguments": {
            "chns_report": chns_report_path,
            "physics_evidence": physics_evidence_path,
            "run_cpu_benchmark": bool(run_benchmark),
            "overwrite": bool(overwrite),
        },
        "artifacts": {
            filename: {"sha256": _sha256(out / filename), "bytes": (out / filename).stat().st_size}
            for filename in [*artifacts, "a1_promotion_report.md"]
        },
        "readiness": report["readiness"],
    }
    _write_json(out / "manifest.json", manifest)
    print(f"Stage-2 report: {out / 'a1_promotion_report.json'}")
    print(json.dumps(report["readiness"], indent=2, sort_keys=True))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="artifacts/l1a2i_stage2/promotion")
    parser.add_argument("--chns-report", default=None, help="completed production.run_validation baseline report.json")
    parser.add_argument("--physics-evidence", default=None, help="artifact-backed JSON summary of required physics gates")
    parser.add_argument("--skip-cpu-benchmark", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    report = run_stage2_audit(
        args.out,
        chns_report_path=args.chns_report,
        physics_evidence_path=args.physics_evidence,
        run_benchmark=not args.skip_cpu_benchmark,
        overwrite=args.overwrite,
    )
    return 0 if report["readiness"]["CONTRACT11_LINEAGE_READY"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
