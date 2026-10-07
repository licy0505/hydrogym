"""L1A-2j diagnostic-only forensics for the contract-11 60-degree CHNS late-time state.

This module observes the production solver without changing its numerical semantics or defaults.
It reuses the frozen L1A-2i/L1A-2j configuration, the existing contact-angle estimator and the
production stationarity criteria. Diagnostic branches (freeze-phi, freeze-u, velocity reset,
capillary-off, and optional half-dt) are explicitly labeled and can never be used as production
acceptance evidence.

Run from ``examples/two_phase``::

    JAX_ENABLE_X64=1 python -m production.chns_nonstationarity_audit \\
        --profile quick --out artifacts/l1a2j_quick
    JAX_ENABLE_X64=1 python -m production.chns_nonstationarity_audit \\
        --profile forensic --target 60 --out artifacts/l1a2j

Large snapshots and time series are written only below the selected artifact directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy import ndimage, signal, stats

import phasefield as pf
from production import contact_line_kinetics as clk
from production import nonneutral_wetting_audit as nwa
from production import observables as obs
from production.config import get_git_sha

STAGE = "L1A-2j"
SOLVER_CONTRACT = 11
M_REF = float(nwa.M_REF)
TARGET = 60.0
CONTROL_TARGETS = (90.0, 150.0)
N = 128
LENGTH = 6.0
RADIUS = 1.1
WALL_HEIGHT = 0.25
DT = 4.0e-3
EPS_FACTOR = 2.0
AUTHORITY_START = 40_000
AUTHORITY_END = 50_000
DENSE_CADENCE = 10
BURST_START = 48_000
BURST_STEPS = 2_000
FIELD_CADENCE = 250
MASS_DRIFT_LIMIT = 1.0e-3

PHENOMENOLOGY_LABELS = (
    "DECAYING_BUT_SLOW",
    "NOISY_STATIONARY_PLATEAU",
    "ANGLE_LIMIT_CYCLE",
    "CONTACT_LINE_STICK_SLIP",
    "STEADY_RECIRCULATION",
    "PHASE_ONLY_NONSTATIONARITY",
    "HYDRODYNAMIC_NONSTATIONARITY",
    "MULTI_FREQUENCY_OSCILLATION",
    "CLASSIFIER_FALSE_NEGATIVE",
    "MULTIPLE_PHENOMENA",
    "INCONCLUSIVE",
)
MECHANISM_LABELS = (
    "PHASE_KINETICS_LIMITED",
    "CAPILLARY_PRESSURE_IMBALANCE",
    "PRESSURE_PROJECTION_LIMITED",
    "BRINKMAN_WALL_COUPLING",
    "CONTACT_LINE_PINNING",
    "Y_PERIODIC_TOPOLOGY_COUPLING",
    "TIME_SPLITTING_OR_DT_SENSITIVITY",
    "MOMENTUM_VISCOSITY_DISCRETIZATION",
    "VARIABLE_DENSITY_COUPLING",
    "CAPILLARY_DENSITY_SCALING",
    "CLASSIFIER_LOGIC_ONLY",
    "MULTIPLE_CONTRIBUTORS",
    "INCONCLUSIVE",
)
MECHANISM_STATUSES = ("SUPPORTED", "SUSPECTED", "FALSIFIED", "NOT_TESTED")

FORMAL_CRITERION_KEYS = (
    "angle_stationarity",
    "free_energy_stability",
    "phase_rate",
    "maximum_speed",
)
FROZEN_PRODUCTION_CRITERIA = {
    "window_samples": 5,
    "window_mobility_time": 0.05,
    "angle_tol_deg": 0.10,
    "energy_rel_tol": 1.0e-4,
    "strict_energy_rel_tol": 1.0e-7,
    "phase_rate_l2_tol": 1.0e-3,
    "chns_speed_tol": 5.0e-4,
    "acceptance_error_deg": 5.0,
    "relaxing_rate_deg_per_Mt": 3.0,
}

SUBSTEP_METRIC_NAMES = (
    "divergence_before_l2",
    "divergence_before_linf",
    "divergence_after_l2",
    "divergence_after_linf",
    "projection_reduction_l2",
    "projection_reduction_linf",
    "pressure_correction_l2",
    "pressure_correction_linf",
    "pressure_gradient_l2",
    "pressure_gradient_linf",
    "poisson_residual_l2",
    "poisson_residual_linf",
    "pressure_mean",
    "pressure_mean_abs",
    "term_reconstruction_u_linf",
    "term_reconstruction_v_linf",
    "advection_l2",
    "advection_linf",
    "advection_work_proxy",
    "viscous_l2",
    "viscous_linf",
    "viscous_work_proxy",
    "capillary_l2",
    "capillary_linf",
    "capillary_work_proxy",
    "gravity_l2",
    "gravity_linf",
    "gravity_work_proxy",
    "brinkman_l2",
    "brinkman_linf",
    "brinkman_work_proxy",
    "pressure_projection_l2",
    "pressure_projection_linf",
    "pressure_projection_work_proxy",
    "substep_time",
)

STEP_METRIC_NAMES = (
    "phase_rate_l2",
    "phase_rate_linf",
    "formal_phase_mass",
    "phase_free_energy",
    "bulk_energy",
    "gradient_energy",
    "wall_energy",
    "kinetic_energy",
    "max_speed",
    "rms_speed_full_grid",
    "cg_iterations_max",
    "cg_relative_residual_max",
    "cg_converged_all",
    "state_time",
)


class ForensicFields(NamedTuple):
    """Fields from the exact final internal substep; force fields are the arrays actually applied."""

    term_phi: jnp.ndarray
    term_u: jnp.ndarray
    term_v: jnp.ndarray
    pressure: jnp.ndarray
    mu: jnp.ndarray
    rho: jnp.ndarray
    nu: jnp.ndarray
    advection_u: jnp.ndarray
    advection_v: jnp.ndarray
    viscous_u: jnp.ndarray
    viscous_v: jnp.ndarray
    capillary_u: jnp.ndarray
    capillary_v: jnp.ndarray
    gravity_u: jnp.ndarray
    gravity_v: jnp.ndarray
    brinkman_u: jnp.ndarray
    brinkman_v: jnp.ndarray
    projection_u: jnp.ndarray
    projection_v: jnp.ndarray
    divergence_before: jnp.ndarray
    divergence_after: jnp.ndarray
    poisson_residual: jnp.ndarray
    phase_rate: jnp.ndarray


class ForensicStep(NamedTuple):
    state: pf.State
    step_metrics: jnp.ndarray
    substep_metrics: jnp.ndarray
    iterations: jnp.ndarray
    residuals: jnp.ndarray
    converged: jnp.ndarray
    fields: ForensicFields


class CheckpointError(RuntimeError):
    """A forensic checkpoint failed its solver/config/state lineage checks."""


def _log(message: str) -> None:
    print(f"[l1a2j {time.strftime('%H:%M:%S')}] {message}", flush=True)


def _json_clean(value: Any) -> Any:
    """Copy nested values into strict-JSON primitives; non-finite diagnostics become null."""
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_clean(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _canonical_hash(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _array_hash(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _state_hashes(state: pf.State) -> dict[str, str]:
    return {name: _array_hash(getattr(state, name)) for name in ("phi", "u", "v", "t")}


def production_config(*, target_deg: float = TARGET, N_value: int = N, dt: float = DT, M: float = M_REF) -> dict[str, Any]:
    """The frozen contract-11 primary/control config; no production parameter is inferred."""
    return {
        "target_deg": float(target_deg),
        "N": int(N_value),
        "Nx": int(N_value),
        "Ny": int(N_value),
        "Lx": float(LENGTH),
        "Ly": float(LENGTH),
        "Re": 200.0,
        "We": 100.0,
        "Fr": 1.0e6,
        "rho_l": 1.0,
        "rho_g": 0.1,
        "nu_l": 1.0 / 200.0,
        "nu_g": 10.0 / 200.0,
        "viscosity_model": "production_nu_of_phi",
        "capillary_denominator": "rho_l",
        "pressure_projection": "unchanged_constant_coefficient_m2_proj",
        "R": float(RADIUS),
        "wall_height": float(WALL_HEIGHT),
        "eps_factor": float(EPS_FACTOR),
        "eps": float(EPS_FACTOR * LENGTH / N_value),
        "wall_delta_width": float(1.5 * LENGTH / N_value),
        "dt": float(dt),
        "M": float(M),
        "phase_storage_model": pf.PHASE_ONLY_FLOAT64_STORAGE_MODEL,
        "phase_state_dtype": "float64",
        "velocity_state_dtype": "float32",
        "phase_boundary_model": "impermeable_flux",
        "wetting_model": "surface_energy",
        "wall_measure": pf.WALL_MEASURE_METHOD,
        "phase_transport_geometry": pf.PHASE_TRANSPORT_GEOMETRY,
        "phase_advection_subcycling": pf.PHASE_ADVECTION_SUBCYCLING,
        "eta_pen": float(2.0 * dt),
        "ch_solver_rtol": 1.0e-6,
        "ch_solver_max_iterations": 200,
        "enforce_solid_phi": False,
        "use_gravity": False,
        "solid_geometry": "surface_flat_wall_height_0.25",
        "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
    }


def _make_case(target_deg: float = TARGET, *, N_value: int = N, dt: float = DT, M: float = M_REF):
    config = production_config(target_deg=target_deg, N_value=N_value, dt=dt, M=M)
    p = pf.PhaseFieldParams(
        Nx=N_value,
        Ny=N_value,
        Lx=LENGTH,
        Ly=LENGTH,
        Re=200.0,
        We=100.0,
        dt=dt,
        M=M,
        eps=EPS_FACTOR * LENGTH / N_value,
        dtype=jnp.float32,
        phase_storage_model=pf.PHASE_ONLY_FLOAT64_STORAGE_MODEL,
        phase_boundary_model="impermeable_flux",
        wetting_model="surface_energy",
        wall_measure=pf.WALL_MEASURE_METHOD,
        phase_transport_geometry=pf.PHASE_TRANSPORT_GEOMETRY,
        phase_advection_subcycling=pf.PHASE_ADVECTION_SUBCYCLING,
        use_gravity=False,
    )
    solid = pf.make_solid(
        pf.surface_flat(p, wall_height=WALL_HEIGHT),
        p,
        cos_theta=math.cos(math.radians(float(target_deg))),
    )
    state = pf.sessile_initial_state(p, solid, R=RADIUS, wall_height=WALL_HEIGHT)
    return p, solid, state, config


def _source_hashes() -> dict[str, str]:
    here = Path(__file__).resolve().parent.parent
    paths = {
        "phasefield": here / "phasefield.py",
        "nonneutral_wetting_audit": here / "production" / "nonneutral_wetting_audit.py",
        "contact_line_kinetics": here / "production" / "contact_line_kinetics.py",
        "audit_runner": Path(__file__).resolve(),
    }
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def _checkpoint_metadata(
    state: pf.State,
    config: dict[str, Any],
    *,
    step_index: int,
    kind: str,
    parent_state_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    if pf.SOLVER_CONTRACT_VERSION != SOLVER_CONTRACT:
        raise RuntimeError(f"L1A-2j requires contract 11, found {pf.SOLVER_CONTRACT_VERSION}")
    if config.get("solver_contract_version") != SOLVER_CONTRACT:
        raise RuntimeError("forensic checkpoint config is not contract 11")
    p, _, _, _ = _make_case(
        float(config["target_deg"]), N_value=int(config["N"]), dt=float(config["dt"]), M=float(config["M"])
    )
    return {
        "stage": STAGE,
        "checkpoint_schema_version": 1,
        "checkpoint_kind": str(kind),
        "step_index": int(step_index),
        "git_sha": get_git_sha(),
        "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
        "production_semantics_changed": False,
        "diagnostic_only": str(kind).startswith(("freeze_", "velocity_reset", "dt_ablation", "quick_test")),
        "diagnostic_branch": str(kind) if str(kind).startswith(("freeze_", "velocity_reset", "dt_ablation", "quick_test")) else None,
        "config": config,
        "config_fingerprint": _canonical_hash(config),
        "source_hashes": _source_hashes(),
        "state_hashes": _state_hashes(state),
        "parent_state_hashes": parent_state_hashes,
        **pf.phase_transport_metadata(p),
    }


def save_forensic_checkpoint(
    path: str | Path,
    state: pf.State,
    config: dict[str, Any],
    *,
    step_index: int,
    kind: str,
    parent_state_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Atomically save phi/u/v/time with SHA, contract, config fingerprint, and per-array hashes."""
    metadata = _checkpoint_metadata(
        state, config, step_index=step_index, kind=kind, parent_state_hashes=parent_state_hashes
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            phi=np.asarray(state.phi),
            u=np.asarray(state.u),
            v=np.asarray(state.v),
            t=np.asarray(state.t),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)),
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    return metadata


def load_forensic_checkpoint(
    path: str | Path,
    *,
    expected_config: dict[str, Any],
    expected_step: int | None = None,
    expected_kind: str | None = None,
    require_same_sha: bool = True,
) -> tuple[pf.State, dict[str, Any]]:
    """Load only an exact config/contract/lineage match; never migrate or silently recast."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        required = {"phi", "u", "v", "t", "metadata_json"}
        if required - set(archive.files):
            raise CheckpointError(f"checkpoint missing {sorted(required - set(archive.files))}")
        arrays = {key: np.array(archive[key], copy=True) for key in ("phi", "u", "v", "t")}
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
    if metadata.get("stage") != STAGE or metadata.get("solver_contract_version") != SOLVER_CONTRACT:
        raise CheckpointError("checkpoint is not an L1A-2j contract-11 state")
    if metadata.get("production_semantics_changed") is not False:
        raise CheckpointError("checkpoint claims production semantics changed")
    if metadata.get("config_fingerprint") != _canonical_hash(expected_config):
        raise CheckpointError("checkpoint config fingerprint mismatch")
    if metadata.get("config") != expected_config:
        raise CheckpointError("checkpoint config metadata mismatch")
    if expected_step is not None and metadata.get("step_index") != int(expected_step):
        raise CheckpointError("checkpoint step index mismatch")
    if expected_kind is not None and metadata.get("checkpoint_kind") != expected_kind:
        raise CheckpointError("checkpoint kind mismatch")
    if require_same_sha and metadata.get("git_sha") != get_git_sha():
        raise CheckpointError("checkpoint Git SHA mismatch")
    if metadata.get("source_hashes") != _source_hashes():
        raise CheckpointError("checkpoint source hashes mismatch")
    hashes = {key: _array_hash(value) for key, value in arrays.items()}
    if metadata.get("state_hashes") != hashes:
        raise CheckpointError("checkpoint state-array hash mismatch")
    expected_dtypes = {"phi": "float64", "u": "float32", "v": "float32", "t": "float32"}
    if {key: value.dtype.name for key, value in arrays.items()} != expected_dtypes:
        raise CheckpointError("checkpoint array dtype mismatch")
    if arrays["phi"].shape != (int(expected_config["Nx"]), int(expected_config["Ny"])):
        raise CheckpointError("checkpoint phase shape mismatch")
    state = pf.State(
        phi=jnp.asarray(arrays["phi"], dtype=jnp.float64),
        u=jnp.asarray(arrays["u"], dtype=jnp.float32),
        v=jnp.asarray(arrays["v"], dtype=jnp.float32),
        t=jnp.asarray(arrays["t"].item(), dtype=jnp.float32),
    )
    return state, metadata


def _masked_l2(value: jnp.ndarray, mask: jnp.ndarray, cell_area: float) -> jnp.ndarray:
    value64 = jnp.asarray(value, dtype=jnp.float64)
    mask64 = jnp.asarray(mask, dtype=jnp.float64)
    return jnp.sqrt(jnp.sum(value64 * value64 * mask64) * float(cell_area))


def _masked_linf(value: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    return jnp.max(jnp.abs(jnp.where(mask, value, jnp.zeros_like(value))))


def _vector_l2(first: jnp.ndarray, second: jnp.ndarray, mask: jnp.ndarray, area: float) -> jnp.ndarray:
    a = jnp.asarray(first, dtype=jnp.float64)
    b = jnp.asarray(second, dtype=jnp.float64)
    weight = jnp.asarray(mask, dtype=jnp.float64)
    return jnp.sqrt(jnp.sum((a * a + b * b) * weight) * float(area))


def _vector_linf(first: jnp.ndarray, second: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    return jnp.max(jnp.where(mask, jnp.sqrt(first * first + second * second), 0.0))


def _vector_work(
    velocity_u: jnp.ndarray,
    velocity_v: jnp.ndarray,
    accel_u: jnp.ndarray,
    accel_v: jnp.ndarray,
    area: float,
) -> jnp.ndarray:
    return jnp.sum(
        jnp.asarray(velocity_u, dtype=jnp.float64) * jnp.asarray(accel_u, dtype=jnp.float64)
        + jnp.asarray(velocity_v, dtype=jnp.float64) * jnp.asarray(accel_v, dtype=jnp.float64)
    ) * float(area)


def _energy_parts(phi: jnp.ndarray, solid: pf.Solid, p: pf.PhaseFieldParams) -> tuple[jnp.ndarray, ...]:
    """Return the three exact pieces of the existing discrete phase free energy."""
    operator = pf.phase_transport_operator(solid, p)
    bulk = jnp.sum(operator.volume * phi**2 * (1.0 - phi) ** 2 / p.eps)
    dx_phi = jnp.roll(phi, -1, axis=0) - phi
    dy_phi = jnp.roll(phi, -1, axis=1) - phi
    gradient = 0.5 * p.eps * jnp.sum(operator.weight_x * dx_phi**2 + operator.weight_y * dy_phi**2)
    wall = pf.wall_free_energy(phi, solid, p)
    return bulk, gradient, wall, bulk + gradient + wall


def _momentum_components(
    state: pf.State,
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
    mu: jnp.ndarray,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], ...]:
    """Re-evaluate the exact discrete addends in ``pf.rhs`` for audit-only decomposition."""
    phi, u, v = state.phi, state.u, state.v
    adv_u = -pf.div_upwind(u, v, u, p.dx, p.dy)
    adv_v = -pf.div_upwind(u, v, v, p.dx, p.dy)
    nu = pf.nu_of(phi, p)
    visc_u = nu * pf._lap(u, p.dx, p.dy)
    visc_v = nu * pf._lap(v, p.dx, p.dy)
    phi_x, phi_y = pf._ddx(phi, p.dx), pf._ddy(phi, p.dy)
    cap_u = (pf.SIGMA_NORM / p.We) * mu * phi_x / p.rho_l
    cap_v = (pf.SIGMA_NORM / p.We) * mu * phi_y / p.rho_l
    if p.use_gravity:
        rho = pf.rho_of(phi, p)
        gravity_v = -(1.0 / p.Fr**2) * (rho - jnp.mean(rho)) / rho
    else:
        gravity_v = jnp.zeros_like(phi)
    gravity_u = jnp.zeros_like(phi)
    return (adv_u, adv_v), (visc_u, visc_v), (cap_u, cap_v), (gravity_u, gravity_v)


def _metric_pair(first, second, velocity_u, velocity_v, area):
    magnitude = jnp.sqrt(first * first + second * second)
    return (
        _vector_l2(first, second, jnp.ones_like(first, dtype=jnp.bool_), area),
        jnp.max(magnitude),
        _vector_work(velocity_u, velocity_v, first, second, area),
    )


def _forensic_substep(
    carry: tuple[jnp.ndarray, ...],
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
    *,
    freeze_phi: bool,
    capillary_scale: float,
) -> tuple[tuple[jnp.ndarray, ...], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, ForensicFields]]:
    """One exact production internal substep plus observational diagnostics."""
    phi, u, v, t = carry
    dt = p.dt / 3.0
    state = pf.State(phi, u, v, t)
    phi_rhs, u_rhs, v_rhs, mu, mu_expl = pf.rhs(state, solid, p)
    (adv_u, adv_v), (visc_u, visc_v), (cap_u, cap_v), (grav_u, grav_v) = _momentum_components(
        state, solid, p, mu
    )
    # Ordered addend sum/cast follows production rhs. Only the capillary-off branch forms an
    # alternate diagnostic RHS; it is never part of the authority/control trajectory.
    u_reconstructed = (adv_u + visc_u + capillary_scale * cap_u).astype(p.dtype)
    v_reconstructed = (adv_v + visc_v + capillary_scale * cap_v + grav_v).astype(p.dtype)
    if capillary_scale == 1.0:
        momentum_u, momentum_v = u_rhs, v_rhs
    else:
        momentum_u, momentum_v = u_reconstructed, v_reconstructed
    rho = pf.rho_of(phi, p)
    nu = pf.nu_of(phi, p)
    damp = 1.0 / (1.0 + dt * solid.chi / p.eta_pen)
    undamped_u = u + dt * momentum_u
    undamped_v = v + dt * momentum_v
    predictor_u = undamped_u * damp
    predictor_v = undamped_v * damp
    brinkman_u = (predictor_u - undamped_u) / dt
    brinkman_v = (predictor_v - undamped_v) / dt
    divergence_before = pf._ddx(predictor_u, p.dx) + pf._ddy(predictor_v, p.dy)
    pressure = pf.poisson_solve(divergence_before / dt, p.m2_proj)
    grad_pressure_u = pf._ddx(pressure, p.dx)
    grad_pressure_v = pf._ddy(pressure, p.dy)
    projection_u = -grad_pressure_u
    projection_v = -grad_pressure_v
    projected_u = predictor_u - dt * grad_pressure_u
    projected_v = predictor_v - dt * grad_pressure_v
    divergence_after = pf._ddx(projected_u, p.dx) + pf._ddy(projected_v, p.dy)
    lap_pressure = pf._ddx(grad_pressure_u, p.dx) + pf._ddy(grad_pressure_v, p.dy)
    poisson_residual = divergence_before / dt - lap_pressure
    if freeze_phi:
        phi_new = phi
        solve_info = pf.ImplicitSolveInfo(
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=phi.dtype),
            jnp.asarray(True),
        )
    else:
        phi_new, solve_info = pf._phase_update(phi, u, v, solid, p, dt, phi_rhs, mu_expl)

    area = p.dx * p.dy
    ones = jnp.ones_like(phi, dtype=jnp.bool_)
    term_values = []
    for first, second in (
        (adv_u, adv_v),
        (visc_u, visc_v),
        (capillary_scale * cap_u, capillary_scale * cap_v),
        (grav_u, grav_v),
        (brinkman_u, brinkman_v),
        (projection_u, projection_v),
    ):
        term_values.extend(_metric_pair(first, second, u, v, area))
    grad_norm_u = _masked_l2(grad_pressure_u, ones, area)
    grad_norm_v = _masked_l2(grad_pressure_v, ones, area)
    pressure_gradient_l2 = jnp.sqrt(grad_norm_u * grad_norm_u + grad_norm_v * grad_norm_v)
    pressure_gradient_linf = jnp.max(jnp.sqrt(grad_pressure_u**2 + grad_pressure_v**2))
    before_l2 = _masked_l2(divergence_before, ones, area)
    before_linf = _masked_linf(divergence_before, ones)
    after_l2 = _masked_l2(divergence_after, ones, area)
    after_linf = _masked_linf(divergence_after, ones)
    residual_l2 = _masked_l2(poisson_residual, ones, area)
    residual_linf = _masked_linf(poisson_residual, ones)
    metrics = jnp.asarray(
        [
            before_l2,
            before_linf,
            after_l2,
            after_linf,
            after_l2 / jnp.maximum(before_l2, 1.0e-30),
            after_linf / jnp.maximum(before_linf, 1.0e-30),
            _masked_l2(pressure, ones, area),
            _masked_linf(pressure, ones),
            pressure_gradient_l2,
            pressure_gradient_linf,
            residual_l2,
            residual_linf,
            jnp.mean(pressure.astype(jnp.float64)),
            jnp.abs(jnp.mean(pressure.astype(jnp.float64))),
            jnp.max(jnp.abs(u_rhs - u_reconstructed)),
            jnp.max(jnp.abs(v_rhs - v_reconstructed)),
            *term_values,
            (t + dt).astype(jnp.float64),
        ],
        dtype=jnp.float64,
    )
    fields = ForensicFields(
        phi,
        u,
        v,
        pressure,
        mu,
        rho,
        nu,
        adv_u,
        adv_v,
        visc_u,
        visc_v,
        cap_u * capillary_scale,
        cap_v * capillary_scale,
        grav_u,
        grav_v,
        brinkman_u,
        brinkman_v,
        projection_u,
        projection_v,
        divergence_before,
        divergence_after,
        poisson_residual,
        jnp.zeros_like(phi),
    )
    return (phi_new, projected_u, projected_v, t + dt), (
        metrics,
        solve_info.iterations,
        solve_info.relative_residual,
        solve_info.converged,
        fields,
    )


def _forensic_step_impl(
    state: pf.State,
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
    *,
    freeze_phi: bool = False,
    capillary_scale: float = 1.0,
) -> ForensicStep:
    """Diagnostic-only replica used for force decomposition; authority states advance via production."""

    def body(carry, _):
        return _forensic_substep(carry, solid, p, freeze_phi=freeze_phi, capillary_scale=capillary_scale)

    (phi, u, v, t), (sub_metrics, iterations, residuals, converged, fields_each) = jax.lax.scan(
        body, (state.phi, state.u, state.v, state.t), None, length=3
    )
    state_out = pf.State(
        phi=phi.astype(pf.phase_state_dtype(p)),
        u=u.astype(p.dtype),
        v=v.astype(p.dtype),
        t=t.astype(p.dtype),
    )
    rate = (state_out.phi - state.phi) / float(p.dt)
    area = p.dx * p.dy
    rate_l2 = _masked_l2(rate, jnp.ones_like(rate, dtype=jnp.bool_), area)
    rate_linf = _masked_linf(rate, jnp.ones_like(rate, dtype=jnp.bool_))
    mass = pf.liquid_mass(state_out.phi, solid, p)
    bulk, gradient, wall, free_energy = _energy_parts(state_out.phi, solid, p)
    speed2 = state_out.u**2 + state_out.v**2
    kinetic = jnp.sum(0.5 * pf.rho_of(state_out.phi, p) * speed2) * area
    max_speed = jnp.max(jnp.sqrt(speed2))
    rms_speed = jnp.sqrt(jnp.mean(speed2.astype(jnp.float64)))
    step_metrics = jnp.asarray(
        [
            rate_l2,
            rate_linf,
            mass,
            free_energy,
            bulk,
            gradient,
            wall,
            kinetic,
            max_speed,
            rms_speed,
            jnp.max(iterations),
            jnp.max(residuals),
            jnp.all(converged).astype(jnp.float64),
            state_out.t.astype(jnp.float64),
        ],
        dtype=jnp.float64,
    )
    fields = jax.tree_util.tree_map(lambda item: item[-1], fields_each)
    fields = fields._replace(phase_rate=rate)
    return ForensicStep(state_out, step_metrics, sub_metrics, iterations, residuals, converged, fields)


def _production_step_metrics(state_in, state_out, solid, p, diagnostics):
    """Scalar observations for a state returned by the unchanged production step."""
    rate = (state_out.phi - state_in.phi) / float(p.dt)
    area = p.dx * p.dy
    rate_l2 = _masked_l2(rate, jnp.ones_like(rate, dtype=jnp.bool_), area)
    rate_linf = _masked_linf(rate, jnp.ones_like(rate, dtype=jnp.bool_))
    mass = pf.liquid_mass(state_out.phi, solid, p)
    bulk, gradient, wall, free_energy = _energy_parts(state_out.phi, solid, p)
    speed2 = state_out.u**2 + state_out.v**2
    kinetic = jnp.sum(0.5 * pf.rho_of(state_out.phi, p) * speed2) * area
    max_speed = jnp.max(jnp.sqrt(speed2))
    rms_speed = jnp.sqrt(jnp.mean(speed2.astype(jnp.float64)))
    iterations = diagnostics.implicit_iterations
    residuals = diagnostics.implicit_relative_residuals
    converged = diagnostics.implicit_converged
    return jnp.asarray(
        [
            rate_l2,
            rate_linf,
            mass,
            free_energy,
            bulk,
            gradient,
            wall,
            kinetic,
            max_speed,
            rms_speed,
            jnp.max(iterations),
            jnp.max(residuals),
            jnp.all(converged).astype(jnp.float64),
            state_out.t.astype(jnp.float64),
        ],
        dtype=jnp.float64,
    )


@partial(jax.jit, static_argnums=(2, 3, 4))
def forensic_step_jit(
    state: pf.State,
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
    freeze_phi: bool = False,
    capillary_scale: float = 1.0,
) -> ForensicStep:
    return _forensic_step_impl(state, solid, p, freeze_phi=freeze_phi, capillary_scale=capillary_scale)


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def advance_forensic_block(
    state: pf.State,
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
    freeze_phi: bool,
    capillary_scale: float,
    steps_per_sample: int,
    num_samples: int,
):
    """Advance diagnostic steps in a compiled scan, returning one exact state every sample group."""
    template = _forensic_step_impl(
        state, solid, p, freeze_phi=freeze_phi, capillary_scale=capillary_scale
    ).fields
    zero_fields = jax.tree_util.tree_map(jnp.zeros_like, template)

    def one_step(carry, _):
        local_state, _last_fields = carry
        result = _forensic_step_impl(
            local_state, solid, p, freeze_phi=freeze_phi, capillary_scale=capillary_scale
        )
        return (result.state, result.fields), (
            result.step_metrics,
            result.substep_metrics,
            result.iterations,
            result.residuals,
            result.converged,
        )

    def sample_group(carry, _):
        carry_out, observations = jax.lax.scan(one_step, carry, None, length=steps_per_sample)
        state_out, fields_out = carry_out
        return carry_out, (state_out, fields_out, observations)

    initial = (state, zero_fields)
    (final_state, final_fields), (sample_states, sample_fields, observations) = jax.lax.scan(
        sample_group, initial, None, length=num_samples
    )
    return final_state, sample_states, sample_fields, observations, final_fields


@partial(jax.jit, static_argnums=(2, 3, 4))
def advance_production_observed_block(
    state: pf.State,
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
    steps_per_sample: int,
    num_samples: int,
):
    """Observe blocks while production ``pf.step_with_diagnostics`` alone advances the state."""
    template = _forensic_step_impl(state, solid, p, freeze_phi=False, capillary_scale=1.0).fields
    zero_fields = jax.tree_util.tree_map(jnp.zeros_like, template)

    def one_step(carry, _):
        local_state, _last_fields = carry
        audit = _forensic_step_impl(local_state, solid, p, freeze_phi=False, capillary_scale=1.0)
        production_state, production_info = pf.step_with_diagnostics(local_state, solid, p)
        production_rate = (production_state.phi - local_state.phi) / float(p.dt)
        fields = audit.fields._replace(phase_rate=production_rate)
        metrics = _production_step_metrics(local_state, production_state, solid, p, production_info)
        return (production_state, fields), (
            metrics,
            audit.substep_metrics,
            production_info.implicit_iterations,
            production_info.implicit_relative_residuals,
            production_info.implicit_converged,
        )

    def sample_group(carry, _):
        carry_out, observations = jax.lax.scan(one_step, carry, None, length=steps_per_sample)
        state_out, fields_out = carry_out
        return carry_out, (state_out, fields_out, observations)

    initial = (state, zero_fields)
    (final_state, final_fields), (sample_states, sample_fields, observations) = jax.lax.scan(
        sample_group, initial, None, length=num_samples
    )
    return final_state, sample_states, sample_fields, observations, final_fields


@partial(jax.jit, static_argnums=(2, 3))
def _advance_standard_jit(state, solid, p, steps):
    def body(current, _):
        next_state, _ = pf.step_with_diagnostics(current, solid, p)
        return next_state, None

    return jax.lax.scan(body, state, None, length=steps)[0]


def _advance_standard(state: pf.State, solid: pf.Solid, p: pf.PhaseFieldParams, steps: int) -> pf.State:
    """Fast exact production-only advance used before the dense forensic window."""
    if steps < 1:
        return state
    return _advance_standard_jit(state, solid, p, int(steps))


@partial(jax.jit, static_argnums=(2,))
def _observe_current_state_jit(state: pf.State, solid: pf.Solid, p: pf.PhaseFieldParams) -> ForensicFields:
    """Observe the *first* production substep from a checkpoint without changing its state."""
    _phi_rhs, u_rhs, v_rhs, mu, _mu_explicit = pf.rhs(state, solid, p)
    (adv_u, adv_v), (visc_u, visc_v), (cap_u, cap_v), (grav_u, grav_v) = _momentum_components(state, solid, p, mu)
    dt = p.dt / 3.0
    damp = 1.0 / (1.0 + dt * solid.chi / p.eta_pen)
    undamped_u = state.u + dt * u_rhs
    undamped_v = state.v + dt * v_rhs
    predictor_u = undamped_u * damp
    predictor_v = undamped_v * damp
    brinkman_u = (predictor_u - undamped_u) / dt
    brinkman_v = (predictor_v - undamped_v) / dt
    divergence_before = pf._ddx(predictor_u, p.dx) + pf._ddy(predictor_v, p.dy)
    pressure = pf.poisson_solve(divergence_before / dt, p.m2_proj)
    grad_u, grad_v = pf._ddx(pressure, p.dx), pf._ddy(pressure, p.dy)
    projected_u = predictor_u - dt * grad_u
    projected_v = predictor_v - dt * grad_v
    divergence_after = pf._ddx(projected_u, p.dx) + pf._ddy(projected_v, p.dy)
    residual = divergence_before / dt - pf._ddx(grad_u, p.dx) - pf._ddy(grad_v, p.dy)
    return ForensicFields(
        state.phi,
        state.u,
        state.v,
        pressure,
        mu,
        pf.rho_of(state.phi, p),
        pf.nu_of(state.phi, p),
        adv_u,
        adv_v,
        visc_u,
        visc_v,
        cap_u,
        cap_v,
        grav_u,
        grav_v,
        brinkman_u,
        brinkman_v,
        -grad_u,
        -grad_v,
        divergence_before,
        divergence_after,
        residual,
        jnp.zeros_like(state.phi),
    )


def _field_components_from_checkpoint(state: pf.State, solid: pf.Solid, p: pf.PhaseFieldParams) -> ForensicFields:
    """One observational first-substep projection/term evaluation; the checkpoint state is untouched."""
    return _observe_current_state_jit(state, solid, p)



def _as_float(value: Any) -> float | None:
    try:
        result = float(np.asarray(value))
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _side_circle_angle(points: np.ndarray, wall_y: float, side: str) -> float | None:
    if not math.isfinite(wall_y) or points.ndim != 2 or points.shape[1] < 2:
        return None
    selected = points[points[:, 0] < 0.0] if side == "left" else points[points[:, 0] > 0.0]
    if len(selected) < 6:
        return None
    try:
        xc, yc, radius = pf.fit_circle(selected[:, 0], selected[:, 1])
    except (ValueError, np.linalg.LinAlgError):
        return None
    if radius <= 0.0 or not math.isfinite(radius):
        return None
    cosine = (wall_y - yc) / radius
    if abs(cosine) > 1.0:
        return None
    return float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _contact_angle_sides(phi: np.ndarray, solid: pf.Solid, p: pf.PhaseFieldParams, positions: dict[str, Any]):
    if not positions.get("contact_line_exists") or positions.get("x_cm") is None:
        return {"left_angle_deg": None, "right_angle_deg": None, "side_angle_method": "unmeasured_no_contact"}
    xcm = float(positions["x_cm"])
    points = pf.contact_angle_contour_points(
        phi,
        np.asarray(solid.sdf),
        p.dx,
        p.dy,
        level=0.5,
        cutoff=max(float(p.eps), float(p.dx)),
        x0=xcm,
        period=p.Lx,
    )
    wall_y = pf.wall_plane_height(solid, p, x0=xcm)
    return {
        "left_angle_deg": _side_circle_angle(points, wall_y, "left"),
        "right_angle_deg": _side_circle_angle(points, wall_y, "right"),
        "side_angle_method": "independent_bulk_contour_half_circle_fits_diagnostic",
    }


def _periodic_component_count(mask: np.ndarray) -> int:
    """Four-connected components with the production x-periodicity closed across the seam."""
    labels, count = ndimage.label(mask.astype(bool), structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    if count == 0:
        return 0
    parent = np.arange(count + 1, dtype=np.int64)

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    def union(left: int, right: int) -> None:
        if left and right:
            a, b = find(left), find(right)
            if a != b:
                parent[b] = a

    for j in range(mask.shape[1]):
        union(int(labels[0, j]), int(labels[-1, j]))
    roots = {find(int(label)) for label in np.unique(labels) if label > 0}
    return len(roots)


def _mirror_x(field: np.ndarray, center_x: float, dx: float, period: float) -> np.ndarray:
    nx = field.shape[0]
    x = (np.arange(nx, dtype=np.float64) + 0.5) * dx
    mirrored = (2.0 * center_x - x) % period
    index = mirrored / dx - 0.5
    left = np.floor(index).astype(np.int64)
    fraction = index - left
    left %= nx
    right = (left + 1) % nx
    return (1.0 - fraction[:, None]) * field[left, :] + fraction[:, None] * field[right, :]


def _relative_residual(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    numerator = float(np.sqrt(np.sum((a[mask] - b[mask]) ** 2)))
    denominator = float(np.sqrt(np.sum(a[mask] ** 2)))
    return numerator / max(denominator, 1.0e-30)


_TERM_FIELD_NAMES = {
    "advection": ("advection_u", "advection_v"),
    "viscous": ("viscous_u", "viscous_v"),
    "capillary": ("capillary_u", "capillary_v"),
    "gravity": ("gravity_u", "gravity_v"),
    "brinkman": ("brinkman_u", "brinkman_v"),
    "pressure_projection": ("projection_u", "projection_v"),
}


def _spatial_metrics(
    state: pf.State,
    fields: ForensicFields,
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
    positions: dict[str, Any],
) -> dict[str, float | None]:
    """Grid-integrated global/interface/wall/contact-line norms of exact applied term arrays."""
    phi_term = np.asarray(fields.term_phi, dtype=np.float64)
    u_term = np.asarray(fields.term_u, dtype=np.float64)
    v_term = np.asarray(fields.term_v, dtype=np.float64)
    phi = np.asarray(state.phi, dtype=np.float64)
    u = np.asarray(state.u, dtype=np.float64)
    v = np.asarray(state.v, dtype=np.float64)
    sdf = np.asarray(solid.sdf, dtype=np.float64)
    dx, dy, area = float(p.dx), float(p.dy), float(p.dx * p.dy)
    fluid = sdf >= 0.0
    interface = fluid & (phi_term >= 0.05) & (phi_term <= 0.95)
    dist = np.maximum(sdf, 0.0)
    wall_shells = {
        "wall_0_1dx": fluid & (dist <= dx),
        "wall_1_2dx": fluid & (dist > dx) & (dist <= 2.0 * dx),
        "wall_2_4dx": fluid & (dist > 2.0 * dx) & (dist <= 4.0 * dx),
        "wall_gt_4dx": fluid & (dist > 4.0 * dx),
    }
    x_axis = (np.arange(p.Nx, dtype=np.float64) + 0.5) * dx
    nearwall = fluid & (dist <= max(4.0 * dx, 2.0 * p.eps))
    contacts: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        center = positions.get(f"{side}_contact_x_wrapped")
        if center is None:
            contacts[side] = np.zeros_like(fluid)
        else:
            delta = np.abs((x_axis - float(center) + 0.5 * p.Lx) % p.Lx - 0.5 * p.Lx)
            contacts[side] = nearwall & (delta[:, None] <= max(2.0 * p.eps, dx))

    out: dict[str, float | None] = {}

    def vec_metrics(name: str, ax: np.ndarray, ay: np.ndarray, mask: np.ndarray, suffix: str) -> None:
        if not np.any(mask):
            out[f"{name}_{suffix}_l2"] = None
            out[f"{name}_{suffix}_linf"] = None
            out[f"{name}_{suffix}_work_proxy"] = None
            return
        magnitude2 = ax**2 + ay**2
        out[f"{name}_{suffix}_l2"] = float(np.sqrt(np.sum(magnitude2[mask]) * area))
        out[f"{name}_{suffix}_linf"] = float(np.sqrt(np.max(magnitude2[mask])))
        out[f"{name}_{suffix}_work_proxy"] = float(np.sum((u_term[mask] * ax[mask] + v_term[mask] * ay[mask]) * area))

    for term, (x_name, y_name) in _TERM_FIELD_NAMES.items():
        ax = np.asarray(getattr(fields, x_name), dtype=np.float64)
        ay = np.asarray(getattr(fields, y_name), dtype=np.float64)
        vec_metrics(term, ax, ay, np.ones_like(fluid), "global")
        vec_metrics(term, ax, ay, interface, "interface")
        for region, mask in wall_shells.items():
            vec_metrics(term, ax, ay, mask, region)
        for side, mask in contacts.items():
            vec_metrics(term, ax, ay, mask, f"contact_{side}")

    speed2 = u**2 + v**2
    rho = np.asarray(pf.rho_of(state.phi, p), dtype=np.float64)
    phase_rate = np.asarray(fields.phase_rate, dtype=np.float64)
    out["fluid_rms_speed"] = float(np.sqrt(np.mean(speed2[fluid]))) if np.any(fluid) else None
    out["fluid_max_speed"] = float(np.sqrt(np.max(speed2[fluid]))) if np.any(fluid) else None
    out["fluid_kinetic_energy"] = float(np.sum(0.5 * rho[fluid] * speed2[fluid]) * area)
    chi = np.asarray(solid.chi, dtype=np.float64)
    penalized = chi > 0.5
    out["chi_weighted_speed_rms"] = float(np.sqrt(np.sum(chi * speed2) / max(np.sum(chi), 1.0e-30)))
    out["penalized_region_speed_rms"] = float(np.sqrt(np.mean(speed2[penalized]))) if np.any(penalized) else None
    out["penalized_region_speed_max"] = float(np.sqrt(np.max(speed2[penalized]))) if np.any(penalized) else None
    out["penalized_region_kinetic_energy"] = float(np.sum(0.5 * rho[penalized] * speed2[penalized]) * area)
    out["interface_phase_rate_l2"] = float(np.sqrt(np.sum(phase_rate[interface] ** 2) * area)) if np.any(interface) else None
    out["interface_phase_rate_linf"] = float(np.max(np.abs(phase_rate[interface]))) if np.any(interface) else None
    out["wall_near_phase_rate_l2"] = float(np.sqrt(np.sum(phase_rate[nearwall] ** 2) * area)) if np.any(nearwall) else None
    for side, mask in contacts.items():
        key = f"contact_{side}"
        out[f"{key}_kinetic_energy"] = float(np.sum(0.5 * rho[mask] * speed2[mask]) * area)
        out[f"{key}_max_speed"] = float(np.sqrt(np.max(speed2[mask]))) if np.any(mask) else None
        out[f"{key}_phase_rate_l2"] = float(np.sqrt(np.sum(phase_rate[mask] ** 2) * area)) if np.any(mask) else None
        chi = np.asarray(solid.chi, dtype=np.float64)
        out[f"{key}_brinkman_dissipation_proxy"] = float(
            np.sum(rho[mask] * chi[mask] * speed2[mask]) / float(p.eta_pen) * area
        )

    # Periodic-y seam: report the discrete cross-seam jump and gradient mismatch rather than
    # demanding equality of neighboring cell-centre values (which are one dy apart).
    for name, field in (("u", u), ("v", v), ("pressure", np.asarray(fields.pressure, dtype=np.float64))):
        seam_jump = field[:, 0] - field[:, -1]
        grad_first = (field[:, 1] - field[:, -1]) / (2.0 * dy)
        grad_last = (field[:, 0] - field[:, -2]) / (2.0 * dy)
        scale = max(float(np.sqrt(np.mean(field**2))), 1.0e-30)
        out[f"y_seam_{name}_jump_rms"] = float(np.sqrt(np.mean(seam_jump**2)))
        out[f"y_seam_{name}_jump_over_rms"] = out[f"y_seam_{name}_jump_rms"] / scale
        out[f"y_seam_{name}_gradient_mismatch_rms"] = float(np.sqrt(np.mean((grad_first - grad_last) ** 2)))

    if positions.get("x_cm") is not None:
        center = float(positions["x_cm"])
        phi_mirror = _mirror_x(phi, center, dx, float(p.Lx))
        u_mirror = _mirror_x(u, center, dx, float(p.Lx))
        v_mirror = _mirror_x(v, center, dx, float(p.Lx))
        active = fluid
        out["symmetry_phi_even_relative_l2"] = _relative_residual(phi, phi_mirror, active)
        out["symmetry_u_odd_relative_l2"] = _relative_residual(u, -u_mirror, active)
        out["symmetry_v_even_relative_l2"] = _relative_residual(v, v_mirror, active)
    else:
        out["symmetry_phi_even_relative_l2"] = None
        out["symmetry_u_odd_relative_l2"] = None
        out["symmetry_v_even_relative_l2"] = None

    out["liquid_components_periodic_x"] = float(_periodic_component_count((phi >= 0.5) & fluid))
    out["minimum_liquid_wall_gap"] = _as_float(obs.bottom_gap(phi, sdf, threshold=0.5))
    return out


def _sample_row(
    state: pf.State,
    fields: ForensicFields,
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
    *,
    step: int,
    mobility: float,
    formal_mass_reference: float,
    diagnostic_horizon: str,
    step_metrics: np.ndarray | None = None,
    phase_rate_l2_dense: float | None = None,
    phase_rate_l2_formal: float | None = None,
    pressure_provenance: str | None = None,
) -> dict[str, Any]:
    phi = np.asarray(state.phi, dtype=np.float64)
    u = np.asarray(state.u, dtype=np.float64)
    v = np.asarray(state.v, dtype=np.float64)
    sdf = np.asarray(solid.sdf, dtype=np.float64)
    time_value = float(np.asarray(state.t))
    positions = clk.contact_line_positions(phi, sdf, p.dx, p.dy, eps=p.eps, Lx=p.Lx)
    theta_raw = _as_float(pf.measure_contact_angle(state.phi, solid, p))
    theta = theta_raw if positions.get("contact_line_exists") else None
    side_angles = _contact_angle_sides(phi, solid, p, positions)
    formal_mass = float(pf.liquid_mass(state.phi, solid, p))
    free_energy = float(pf.phase_free_energy(state.phi, solid, p))
    bulk, gradient, wall, _ = (float(np.asarray(value)) for value in _energy_parts(state.phi, solid, p))
    rho = np.asarray(pf.rho_of(state.phi, p), dtype=np.float64)
    area = float(p.dx * p.dy)
    speed2 = u * u + v * v
    kinetic = float(np.sum(0.5 * rho * speed2) * area)
    mass_dxdy = float(np.sum(phi) * area)
    mass_hard_mask = float(obs.liquid_mass(phi, sdf, p.dx, p.dy))
    metrics = np.full(len(STEP_METRIC_NAMES), np.nan) if step_metrics is None else np.asarray(step_metrics)
    row: dict[str, Any] = {
        "step": int(step),
        "time": time_value,
        "mobility_scaled_time": float(mobility) * time_value,
        "measured_angle_deg": theta,
        "raw_production_angle_deg": theta_raw,
        "left_angle_deg": side_angles["left_angle_deg"],
        "right_angle_deg": side_angles["right_angle_deg"],
        "side_angle_method": side_angles["side_angle_method"],
        "contact_line_exists": bool(positions.get("contact_line_exists")),
        "detachment_observed": bool(positions.get("detachment_observed")),
        "contour_wall_intersection_count": int(positions.get("contour_wall_intersection_count", 0)),
        "bottom_gap": positions.get("bottom_gap"),
        "left_contact_x": positions.get("left_contact_x"),
        "right_contact_x": positions.get("right_contact_x"),
        "left_contact_x_wrapped": positions.get("left_contact_x_wrapped"),
        "right_contact_x_wrapped": positions.get("right_contact_x_wrapped"),
        "contact_width": positions.get("contact_width"),
        "x_cm": positions.get("x_cm"),
        "y_cm": positions.get("y_cm"),
        "top_height": positions.get("top_height"),
        "formal_phase_mass_sum_V_phi": formal_mass,
        "formal_phase_mass_drift_from_reference": abs(formal_mass - formal_mass_reference) / max(abs(formal_mass_reference), 1.0e-30),
        "mass_dxdy_sum_phi_diagnostic_nonconserved": mass_dxdy,
        "hard_mask_mass_diagnostic_nonconserved": mass_hard_mask,
        "free_energy": free_energy,
        "bulk_energy": bulk,
        "gradient_energy": gradient,
        "wall_energy": wall,
        "kinetic_energy": kinetic,
        "max_speed": float(np.sqrt(np.max(speed2))),
        "rms_speed_full_grid": float(np.sqrt(np.mean(speed2))),
        "phase_rate_l2_instantaneous_per_dt": _as_float(metrics[0]) if metrics.size else None,
        "phase_rate_linf_instantaneous_per_dt": _as_float(metrics[1]) if metrics.size else None,
        "phase_rate_l2_dense_sample_interval_over_dt": phase_rate_l2_dense,
        # At step multiples of 200 this reproduces nwa._sample's (phi-phi_prev)/p.dt definition.
        "phase_rate_l2_production_200_step_sample_over_dt": phase_rate_l2_formal,
        "cg_iterations_max_public_step": _as_float(metrics[10]) if metrics.size else None,
        "cg_relative_residual_max_public_step": _as_float(metrics[11]) if metrics.size else None,
        "cg_converged_all_substeps_public_step": bool(metrics[12] >= 1.0) if metrics.size else None,
        "diagnostic_horizon": diagnostic_horizon,
        "pressure_provenance": pressure_provenance or (
            "diagnostic_last_substep_projection_reconstruction_from_preceding_state"
            if step_metrics is not None
            else "current_state_next_step_pressure_field_diagnostic_not_stored_in_State"
        ),
    }
    row.update(positions)
    row.update(_spatial_metrics(state, fields, solid, p, positions))
    return _json_clean(row)


def _attach_contact_line_velocities(rows: list[dict[str, Any]], *, Lx: float) -> None:
    if not rows:
        return
    velocity = clk.contact_line_velocity(
        [row["time"] for row in rows],
        [row.get("left_contact_x_wrapped") for row in rows],
        [row.get("right_contact_x_wrapped") for row in rows],
        Lx=Lx,
    )
    for index, row in enumerate(rows):
        for key, values_key in (
            ("left_contact_velocity", "left_velocity"),
            ("right_contact_velocity", "right_velocity"),
            ("left_contact_speed", "left_speed"),
            ("right_contact_speed", "right_speed"),
            ("contact_line_mean_speed", "mean_speed"),
            ("contact_line_spreading_rate", "spreading_rate"),
            ("contact_line_translation_velocity", "translation_velocity"),
        ):
            row[key] = velocity[values_key][index]
        row["contact_line_displacement_per_dt_dx"] = (
            None
            if row["contact_line_mean_speed"] is None
            else float(row["contact_line_mean_speed"] * DT / (LENGTH / N))
        )


def _records_to_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float, bool)) or value is None})
    out: dict[str, np.ndarray] = {}
    for key in keys:
        values = [np.nan if row.get(key) is None else float(row[key]) for row in rows]
        out[key] = np.asarray(values, dtype=np.float64)
    return out


def _signal_summary(times: Sequence[float], values: Sequence[float | None]) -> dict[str, Any]:
    t = np.asarray(times, dtype=np.float64)
    v = np.asarray([np.nan if value is None else float(value) for value in values], dtype=np.float64)
    valid = np.isfinite(t) & np.isfinite(v)
    t, v = t[valid], v[valid]
    if len(t) < 5:
        return {"status": "unmeasured", "n_samples": int(len(t))}
    order = np.argsort(t)
    t, v = t[order], v[order]
    slope = stats.linregress(t, v)
    try:
        robust = stats.theilslopes(v, t, alpha=0.95)
        robust_slope = float(robust.slope)
        robust_low, robust_high = float(robust.low_slope), float(robust.high_slope)
    except (ValueError, FloatingPointError):
        robust_slope, robust_low, robust_high = float(slope.slope), None, None
    detrended = signal.detrend(v, type="linear")
    dt = float(np.median(np.diff(t)))
    frequencies, powers = signal.periodogram(detrended, fs=1.0 / dt, window="hann", detrend=False, scaling="spectrum")
    nonzero = frequencies > 0.0
    if np.any(nonzero):
        candidates = np.flatnonzero(nonzero)
        peak_i = int(candidates[np.argmax(powers[nonzero])])
        total_power = float(np.sum(powers[nonzero]))
        dom_power = float(powers[peak_i])
        dominant = {
            "frequency_per_time": float(frequencies[peak_i]),
            "period_time": float(1.0 / frequencies[peak_i]) if frequencies[peak_i] > 0 else None,
            "power_fraction": dom_power / max(total_power, 1.0e-300),
            "power": dom_power,
        }
    else:
        dominant = {"frequency_per_time": None, "period_time": None, "power_fraction": 0.0, "power": 0.0}
    centered = detrended - np.mean(detrended)
    variance = float(np.dot(centered, centered))
    if len(centered) > 1 and variance > 1.0e-30:
        autocorr1 = float(np.dot(centered[:-1], centered[1:]) / variance)
    else:
        autocorr1 = None
    peaks, properties = signal.find_peaks(np.abs(detrended), prominence=max(float(np.std(detrended)) * 0.5, 1.0e-15))
    return _json_clean(
        {
            "status": "measured",
            "n_samples": len(t),
            "time_start": t[0],
            "time_end": t[-1],
            "mean": np.mean(v),
            "std": np.std(v),
            "minimum": np.min(v),
            "maximum": np.max(v),
            "peak_to_peak": np.ptp(v),
            "ols_slope_per_time": slope.slope,
            "ols_slope_stderr": slope.stderr,
            "ols_slope_pvalue": slope.pvalue,
            "ols_r_squared": slope.rvalue**2,
            "theil_sen_slope_per_time": robust_slope,
            "theil_sen_slope_95pct_low": robust_low,
            "theil_sen_slope_95pct_high": robust_high,
            "lag1_autocorrelation_detrended": autocorr1,
            "dominant_spectral_peak": dominant,
            "prominent_abs_deviation_peaks": int(len(peaks)),
            "median_sample_interval": dt,
            "nyquist_frequency_per_time": 0.5 / dt,
        }
    )


def _series_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    time_values = [row.get("time") for row in rows]
    fields = (
        "measured_angle_deg",
        "left_angle_deg",
        "right_angle_deg",
        "left_contact_x",
        "right_contact_x",
        "contact_width",
        "left_contact_speed",
        "right_contact_speed",
        "contact_line_spreading_rate",
        "contact_line_translation_velocity",
        "x_cm",
        "y_cm",
        "top_height",
        "free_energy",
        "kinetic_energy",
        "phase_rate_l2_instantaneous_per_dt",
        "phase_rate_l2_dense_sample_interval_over_dt",
        "max_speed",
        "rms_speed_full_grid",
        "formal_phase_mass_sum_V_phi",
        "formal_phase_mass_drift_from_reference",
        "capillary_contact_left_l2",
        "capillary_contact_right_l2",
        "pressure_projection_contact_left_l2",
        "pressure_projection_contact_right_l2",
        "brinkman_contact_left_l2",
        "brinkman_contact_right_l2",
        "y_seam_v_jump_over_rms",
        "y_seam_pressure_jump_over_rms",
        "symmetry_phi_even_relative_l2",
        "symmetry_u_odd_relative_l2",
        "symmetry_v_even_relative_l2",
        "liquid_components_periodic_x",
        "minimum_liquid_wall_gap",
    )
    return {field: _signal_summary(time_values, [row.get(field) for row in rows]) for field in fields}


def _formal_sample_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": row["step"],
        "time": row["time"],
        "mobility_scaled_time": row["mobility_scaled_time"],
        "measured_angle_deg": row["measured_angle_deg"],
        "free_energy": row["free_energy"],
        "phase_rate_l2": row.get("phase_rate_l2_production_200_step_sample_over_dt"),
        "max_speed": row["max_speed"],
    }


def _criterion_window(samples: list[dict[str, Any]], ch_only: bool, M: float = M_REF) -> dict[str, Any]:
    """Exact ``nwa._window_converged`` output plus decomposed raw/normalized criteria."""
    result = nwa._window_converged(samples, ch_only, dict(nwa.CRITERIA), M=M)
    if not samples:
        return {"status": "unmeasured", "criteria": {}, "production_gate": result}
    crit = dict(nwa.CRITERIA)
    last_mt = float(samples[-1]["mobility_scaled_time"])
    span = float(crit["window_mobility_time"])
    window = [row for row in samples if float(row["mobility_scaled_time"]) >= last_mt - span - 1.0e-12]
    covered = bool(samples[0]["mobility_scaled_time"] <= last_mt - span + 1.0e-12)
    enough = len(window) >= int(crit["window_samples"])
    valid_angles = enough and covered and all(row.get("measured_angle_deg") is not None for row in window)
    energy_values = [float(row["free_energy"]) for row in window]
    angle_raw = float(np.ptp([row["measured_angle_deg"] for row in window])) if valid_angles else None
    energy_raw = (
        max(abs(b - a) / max(1.0, abs(b)) for a, b in zip(energy_values, energy_values[1:]))
        if enough and covered and len(energy_values) >= 2
        else None
    )
    rate_raw = (
        float(window[-1]["phase_rate_l2"]) * (M_REF / float(M))
        if enough and covered and window[-1].get("phase_rate_l2") is not None
        else None
    )
    speed_raw = float(window[-1]["max_speed"]) if enough and covered and not ch_only else (0.0 if ch_only and enough and covered else None)
    criterion_specs = (
        ("angle_stationarity", angle_raw, float(crit["angle_tol_deg"]), result.get("angle_ok")),
        ("free_energy_stability", energy_raw, float(crit["energy_rel_tol"]), result.get("energy_ok")),
        ("phase_rate", rate_raw, float(crit["phase_rate_l2_tol"]), result.get("rate_ok")),
        ("maximum_speed", speed_raw, float(crit["chns_speed_tol"]), True if ch_only else result.get("speed_ok")),
    )
    details = {}
    for key, raw, threshold, passed in criterion_specs:
        details[key] = {
            "threshold": threshold,
            "window_mobility_time": span,
            "minimum_window_samples": int(crit["window_samples"]),
            "window_samples_used": len(window),
            "window_covered": covered,
            "raw_value": raw,
            "normalized_value_raw_over_threshold": None if raw is None else raw / threshold,
            "passed": None if raw is None or not enough or not covered else bool(passed),
        }
    details["strict_energy_observation_non_gating"] = {
        "threshold": float(crit["strict_energy_rel_tol"]),
        "raw_value": energy_raw,
        "normalized_value_raw_over_threshold": None if energy_raw is None else energy_raw / float(crit["strict_energy_rel_tol"]),
        "passed": None if energy_raw is None else bool(energy_raw <= float(crit["strict_energy_rel_tol"])),
        "acceptance_gate": False,
    }
    return _json_clean(
        {
            "status": "measured" if enough and covered else "unmeasured_window_not_covered",
            "sample_cadence_steps": None,
            "window_start_mobility_scaled_time": float(window[0]["mobility_scaled_time"]) if window else None,
            "window_end_mobility_scaled_time": last_mt,
            "window_physical_time": float(window[-1]["time"] - window[0]["time"]) if window else None,
            "criteria": details,
            "production_gate": result,
        }
    )


def _criterion_persistence(
    samples: list[dict[str, Any]], *, ch_only: bool, cadence_steps: int, M: float = M_REF
) -> dict[str, Any]:
    rows = []
    thresholds = {
        "angle_stationarity": float(nwa.CRITERIA["angle_tol_deg"]),
        "free_energy_stability": float(nwa.CRITERIA["energy_rel_tol"]),
        "phase_rate": float(nwa.CRITERIA["phase_rate_l2_tol"]),
        "maximum_speed": float(nwa.CRITERIA["chns_speed_tol"]),
    }
    for end in range(1, len(samples) + 1):
        prefix = samples[:end]
        result = nwa._window_converged(prefix, ch_only, dict(nwa.CRITERIA), M=M)
        if "window_samples_used" not in result:
            result = {"converged": None, "angle_ok": None, "energy_ok": None, "rate_ok": None, "speed_ok": None}
            raw_values = {label: None for label in thresholds}
        else:
            last_mt = float(prefix[-1]["mobility_scaled_time"])
            window = [row for row in prefix if float(row["mobility_scaled_time"]) >= last_mt - float(nwa.CRITERIA["window_mobility_time"]) - 1.0e-12]
            raw_values = {
                "angle_stationarity": result.get("angle_window_spread_deg"),
                "free_energy_stability": result.get("energy_window_max_rel_change"),
                "phase_rate": float(window[-1]["phase_rate_l2"]) * (M_REF / float(M)),
                "maximum_speed": 0.0 if ch_only else float(window[-1]["max_speed"]),
            }
        detailed = {}
        for label, threshold in thresholds.items():
            raw = raw_values[label]
            normalized = None if raw is None else float(raw) / threshold
            passed_key = {
                "angle_stationarity": "angle_ok",
                "free_energy_stability": "energy_ok",
                "phase_rate": "rate_ok",
                "maximum_speed": "speed_ok",
            }[label]
            passed = result.get(passed_key)
            if label == "maximum_speed" and ch_only and raw is not None:
                passed = True
            detailed[label] = {
                "raw_value": raw,
                "threshold": threshold,
                "normalized_value_raw_over_threshold": normalized,
                "margin_threshold_minus_raw": None if raw is None else threshold - float(raw),
                "passed": passed,
            }
        rows.append({"step": int(prefix[-1]["step"]), **result, "criteria": detailed})
    summary = {}
    for label, threshold in thresholds.items():
        passed_key = {
            "angle_stationarity": "angle_ok",
            "free_energy_stability": "energy_ok",
            "phase_rate": "rate_ok",
            "maximum_speed": "speed_ok",
        }[label]
        values = [row.get(passed_key) for row in rows]
        valid = [value for value in values if value is not None]
        normalized = [
            row["criteria"][label]["normalized_value_raw_over_threshold"]
            for row in rows
            if row["criteria"][label]["normalized_value_raw_over_threshold"] is not None
        ]
        longest = current = 0
        for value in values:
            if value is False:
                current += 1
                longest = max(longest, current)
            elif value is True or value is None:
                current = 0
        summary[label] = {
            "windows_evaluated": len(valid),
            "failed_windows": sum(value is False for value in valid),
            "passed_windows": sum(value is True for value in valid),
            "fraction_failed": None if not valid else sum(value is False for value in valid) / len(valid),
            "longest_consecutive_failed_windows": longest,
            "longest_consecutive_failure_time": longest * cadence_steps * float(nwa.PROFILES["baseline"]["dt"]),
            "latest_window_passed": valid[-1] if valid else None,
            "threshold": threshold,
            "maximum_normalized_value": max(normalized) if normalized else None,
            "maximum_threshold_excess_normalized": max((value - 1.0 for value in normalized), default=None),
            "normalized_distance_to_threshold_latest": None if not normalized else 1.0 - normalized[-1],
        }
    return _json_clean({"window_end_rows": rows, "criteria": summary})



def _write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    text = json.dumps(_json_clean(value), indent=2, sort_keys=True, allow_nan=False)
    temporary.write_text(text + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def _write_npz(path: str | Path, **arrays: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **{key: np.asarray(value) for key, value in arrays.items()})
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


def _read_json(path: str | Path, default: Any):
    source = Path(path)
    if not source.exists():
        return default
    return json.loads(source.read_text(encoding="utf-8"))


def _checkpoint_candidates(out: Path) -> list[Path]:
    repo = Path(__file__).resolve().parents[3]
    return [
        out / "checkpoints" / "authority_060_step_050000.npz",
        out / "checkpoints" / "authority_resume_latest.npz",
        repo / "examples" / "two_phase" / "artifacts" / "contract11_followup" / "chns" / "theta_060_step050000.npz",
        repo / "examples" / "two_phase" / "evidence" / "l1a2i" / "theta_060_step050000.npz",
    ]


def _reusable_endpoint(out: Path, config: dict[str, Any]) -> tuple[pf.State | None, dict[str, Any]]:
    evidence = []
    endpoint_path = out / "checkpoints" / "authority_060_step_050000.npz"
    for candidate in _checkpoint_candidates(out):
        if not candidate.exists():
            evidence.append({"path": str(candidate), "status": "not_found"})
            continue
        try:
            state, metadata = load_forensic_checkpoint(
                candidate,
                expected_config=config,
                expected_step=AUTHORITY_END if candidate.name.endswith("050000.npz") else None,
                expected_kind="authority_50k" if candidate == endpoint_path else None,
            )
            evidence.append({"path": str(candidate), "status": "compatible", "state_hashes": metadata["state_hashes"]})
            if candidate == endpoint_path:
                return state, {"candidates": evidence, "selected": str(candidate), "metadata": metadata}
        except (OSError, ValueError, KeyError, CheckpointError, json.JSONDecodeError) as exc:
            evidence.append({"path": str(candidate), "status": "rejected", "reason": str(exc)})
    return None, {"candidates": evidence, "selected": None}


def _save_field_snapshot(
    directory: Path,
    step: int,
    state: pf.State,
    fields: ForensicFields,
    config: dict[str, Any],
    *,
    pressure_provenance: str,
) -> Path:
    arrays = {f"state_{name}": np.asarray(getattr(state, name)) for name in ("phi", "u", "v", "t")}
    arrays.update({f"field_{name}": np.asarray(getattr(fields, name)) for name in ForensicFields._fields})
    metadata = {
        "stage": STAGE,
        "step": int(step),
        "config": config,
        "config_fingerprint": _canonical_hash(config),
        "git_sha": get_git_sha(),
        "source_hashes": _source_hashes(),
        "state_hashes": _state_hashes(state),
        "pressure_provenance": pressure_provenance,
        "force_field_time": "last_internal_substep_input_state; exact terms applied in that substep",
        "diagnostic_only": True,
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    path = directory / f"step_{step:06d}.npz"
    _write_npz(path, **arrays)
    return path


def _append_history(history: dict[str, list[np.ndarray]], observations: Any) -> None:
    step_metrics, substep_metrics, iterations, residuals, converged = observations
    history["step_metrics"].append(np.asarray(step_metrics, dtype=np.float64).reshape(-1, len(STEP_METRIC_NAMES)))
    history["substep_metrics"].append(
        np.asarray(substep_metrics, dtype=np.float64).reshape(-1, 3, len(SUBSTEP_METRIC_NAMES))
    )
    history["iterations"].append(np.asarray(iterations, dtype=np.int32).reshape(-1, 3))
    history["residuals"].append(np.asarray(residuals, dtype=np.float64).reshape(-1, 3))
    history["converged"].append(np.asarray(converged, dtype=bool).reshape(-1, 3))


def _flatten_history(history: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    flattened = {}
    for key, chunks in history.items():
        flattened[key] = np.concatenate(chunks, axis=0) if chunks else np.zeros((0,), dtype=np.float64)
    return flattened


def _save_partial_authority(out: Path, rows, burst_rows, history, state, step, config, parent_hashes, formal_previous_phi) -> None:
    partial = out / "partial"
    _write_json(partial / "late_samples.json", rows)
    _write_json(partial / "anti_alias_samples.json", burst_rows)
    _write_npz(partial / "production_step_diagnostics.npz", **_flatten_history(history))
    formal_path = partial / "formal_previous_phi.npy"
    formal_tmp = formal_path.with_suffix(".npy.tmp")
    with formal_tmp.open("wb") as stream:
        np.save(stream, np.asarray(formal_previous_phi, dtype=np.float64), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(formal_tmp, formal_path)
    save_forensic_checkpoint(
        out / "checkpoints" / "authority_resume_latest.npz",
        state,
        config,
        step_index=step,
        kind="authority_resume",
        parent_state_hashes=parent_hashes,
    )
    _write_json(
        partial / "progress.json",
        {
            "stage": STAGE,
            "profile": "forensic",
            "status": "running",
            "next_step": step,
            "config_fingerprint": _canonical_hash(config),
            "git_sha": get_git_sha(),
            "source_hashes": _source_hashes(),
            "state_hashes": _state_hashes(state),
            "latest_checkpoint": "checkpoints/authority_resume_latest.npz",
            "completed_late_samples": len(rows),
            "completed_anti_alias_samples": len(burst_rows),
            "unmeasured_until_step_50000": True,
        },
    )


def _load_partial_history(path: Path) -> dict[str, list[np.ndarray]]:
    history = {key: [] for key in ("step_metrics", "substep_metrics", "iterations", "residuals", "converged")}
    source = path / "production_step_diagnostics.npz"
    if source.exists():
        with np.load(source, allow_pickle=False) as archive:
            for key in history:
                history[key] = [np.array(archive[key], copy=True)]
    return history


def _initial_or_resume_authority(out: Path, p, solid, seed, config):
    checkpoint_dir = out / "checkpoints"
    latest = checkpoint_dir / "authority_resume_latest.npz"
    partial_dir = out / "partial"
    if latest.exists():
        try:
            state, metadata = load_forensic_checkpoint(
                latest, expected_config=config, expected_kind="authority_resume"
            )
            step = int(metadata["step_index"])
            if not AUTHORITY_START <= step < AUTHORITY_END or step % FIELD_CADENCE:
                raise CheckpointError(f"resume step {step} is outside the allowed sparse boundary schedule")
            rows = _read_json(partial_dir / "late_samples.json", [])
            burst_rows = _read_json(partial_dir / "anti_alias_samples.json", [])
            rows = [row for row in rows if int(row["step"]) <= step]
            burst_rows = [row for row in burst_rows if int(row["step"]) <= step]
            if not rows or int(rows[-1]["step"]) != step:
                raise CheckpointError("partial sample manifest does not reach the hashed resume state")
            history = _load_partial_history(partial_dir)
            _log(f"resuming authority at step {step} from exact SHA/config/state-hash checkpoint")
            return state, step, rows, burst_rows, history, metadata["state_hashes"], True
        except (OSError, ValueError, KeyError, CheckpointError, json.JSONDecodeError) as exc:
            _log(f"rejecting stale/incompatible resume checkpoint: {exc}")

    step40_path = checkpoint_dir / "authority_060_step_040000.npz"
    if step40_path.exists():
        try:
            state, metadata = load_forensic_checkpoint(
                step40_path,
                expected_config=config,
                expected_step=AUTHORITY_START,
                expected_kind="authority_40k",
            )
            return state, AUTHORITY_START, [], [], {key: [] for key in ("step_metrics", "substep_metrics", "iterations", "residuals", "converged")}, metadata["state_hashes"], False
        except (OSError, ValueError, KeyError, CheckpointError, json.JSONDecodeError) as exc:
            _log(f"rejecting step-40k checkpoint: {exc}")

    _log("no matching <=40k authority restart; rerunning exact production trajectory from step 0")
    state = seed
    step = 0
    while step < AUTHORITY_START:
        n = min(1000, AUTHORITY_START - step)
        state = _advance_standard(state, solid, p, n)
        state.t.block_until_ready()
        step += n
        if step % 5000 == 0:
            _log(f"production authority advance {step}/{AUTHORITY_START}")
    save_forensic_checkpoint(
        step40_path,
        state,
        config,
        step_index=AUTHORITY_START,
        kind="authority_40k",
    )
    return state, AUTHORITY_START, [], [], {key: [] for key in ("step_metrics", "substep_metrics", "iterations", "residuals", "converged")}, _state_hashes(state), False


def _host_phi_rate(phi_new: np.ndarray, phi_old: np.ndarray, p: pf.PhaseFieldParams) -> float:
    rate = (np.asarray(phi_new, dtype=np.float64) - np.asarray(phi_old, dtype=np.float64)) / float(p.dt)
    return float(np.sqrt(np.sum(rate * rate) * float(p.dx * p.dy)))


def _run_authority(out: Path, endpoint_candidate: pf.State | None, candidate_manifest: dict[str, Any]):
    p, solid, seed, config = _make_case(TARGET, N_value=N, dt=DT, M=M_REF)
    if int(pf.SOLVER_CONTRACT_VERSION) != SOLVER_CONTRACT or not jax.config.x64_enabled:
        raise RuntimeError("the 60-degree production authority requires x64 phase storage and contract 11")
    checkpoint_dir = out / "checkpoints"
    snapshot_dir = out / "fields" / "authority_060"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    formal_mass_reference = float(pf.liquid_mass(seed.phi, solid, p))
    completed = _load_completed_authority(out, config, formal_mass_reference)
    if completed is not None:
        _log("reusing a completed authority record after SHA/config/state-hash validation")
        completed["candidate_comparison"] = candidate_manifest
        return completed
    resume_state, resume_step, rows, burst_rows, history, parent_hashes, resumed = _initial_or_resume_authority(
        out, p, solid, seed, config
    )
    production_next, production_info = pf.step_with_diagnostics(resume_state, solid, p)
    forensic_next = forensic_step_jit(resume_state, solid, p)
    state_tolerances = {"phi": 1.0e-10, "u": 1.0e-7, "v": 1.0e-7, "t": 0.0}
    state_errors = {}
    for key in ("phi", "u", "v", "t"):
        production_value = np.asarray(getattr(production_next, key), dtype=np.float64)
        forensic_value = np.asarray(getattr(forensic_next.state, key), dtype=np.float64)
        state_errors[key] = float(np.max(np.abs(production_value - forensic_value)))
    state_transition_equal = all(state_errors[key] <= state_tolerances[key] for key in state_tolerances)
    production_pressure = pf.pressure_field(resume_state, solid, p)
    observed_pressure = _field_components_from_checkpoint(resume_state, solid, p).pressure
    pressure_error = float(np.max(np.abs(np.asarray(production_pressure, dtype=np.float64) - np.asarray(observed_pressure, dtype=np.float64))))
    pressure_tolerance = 1.0e-7
    pressure_field_equal = pressure_error <= pressure_tolerance
    term_error = float(np.max(np.asarray(forensic_next.substep_metrics)[:, 14:16]))
    iteration_equal = bool(np.array_equal(np.asarray(production_info.implicit_iterations), np.asarray(forensic_next.iterations)))
    residual_error = float(np.max(np.abs(np.asarray(production_info.implicit_relative_residuals, dtype=np.float64) - np.asarray(forensic_next.residuals, dtype=np.float64))))
    residual_tolerance = 1.0e-12
    residual_equal = residual_error <= residual_tolerance
    converged_equal = bool(np.array_equal(np.asarray(production_info.implicit_converged), np.asarray(forensic_next.converged)))
    if not state_transition_equal or not pressure_field_equal or term_error != 0.0 or not iteration_equal or not residual_equal or not converged_equal:
        raise RuntimeError(
            "forensic instrumentation failed closed at the authority restart: "
            f"state_errors={state_errors}, pressure_error={pressure_error}, term_linf={term_error}, "
            f"iterations_equal={iteration_equal}, residual_error={residual_error}, converged_equal={converged_equal}"
        )
    update_verification = {
        "restart_step": int(resume_step),
        "production_step_state_within_roundoff_tolerance": state_transition_equal,
        "state_max_abs_error_by_field": state_errors,
        "state_tolerance_by_field": state_tolerances,
        "first_substep_pressure_field_within_tolerance_of_pf_pressure_field": pressure_field_equal,
        "pressure_field_max_abs_error": pressure_error,
        "pressure_field_tolerance": pressure_tolerance,
        "momentum_addend_reconstruction_linf_max": term_error,
        "implicit_iterations_equal": iteration_equal,
        "implicit_relative_residuals_max_abs_error": residual_error,
        "implicit_relative_residuals_tolerance": residual_tolerance,
        "implicit_converged_flags_equal": converged_equal,
        "production_step_function": "pf.step_with_diagnostics",
        "forensic_step_function": "_forensic_step_impl; diagnostic fields do not feed back into the state",
        "status": "passed",
    }
    _write_json(out / "production_transition_verification.json", update_verification)
    field_manifest = []
    existing_manifest = _read_json(snapshot_dir / "manifest.json", [])
    field_manifest.extend(existing_manifest)
    all_steps_before_resume = resume_step
    if not rows:
        boundary_fields = _field_components_from_checkpoint(resume_state, solid, p)
        initial = _sample_row(
            resume_state,
            boundary_fields,
            solid,
            p,
            step=resume_step,
            mobility=M_REF,
            formal_mass_reference=formal_mass_reference,
            diagnostic_horizon="authority_060_40k_to_50k",
        )
        initial["phase_rate_l2_dense_sample_interval_over_dt"] = None
        initial["phase_rate_l2_production_200_step_sample_over_dt"] = None
        rows = [initial]
        _save_field_snapshot(
            snapshot_dir,
            resume_step,
            resume_state,
            boundary_fields,
            config,
            pressure_provenance="current_state_next_step_pressure_field_diagnostic_not_stored_in_State",
        )
        field_manifest.append({"step": resume_step, "path": f"step_{resume_step:06d}.npz", "pressure_provenance": initial["pressure_provenance"]})
    if resume_step == AUTHORITY_START and not (checkpoint_dir / "authority_060_step_040000.npz").exists():
        save_forensic_checkpoint(
            checkpoint_dir / "authority_060_step_040000.npz",
            resume_state,
            config,
            step_index=AUTHORITY_START,
            kind="authority_40k",
        )

    dense_previous_phi = np.asarray(resume_state.phi, dtype=np.float64)
    formal_previous_phi = dense_previous_phi.copy()
    formal_reference_path = out / "partial" / "formal_previous_phi.npy"
    if resumed and formal_reference_path.exists():
        saved_formal = np.load(formal_reference_path, allow_pickle=False)
        if saved_formal.shape == dense_previous_phi.shape and np.isfinite(saved_formal).all():
            formal_previous_phi = np.asarray(saved_formal, dtype=np.float64)
        else:
            raise CheckpointError("formal-cadence phase reference failed shape/finite validation")
    step = resume_step
    checkpoint_parent_hashes = parent_hashes
    progress_path = out / "partial" / "progress.json"
    while step < AUTHORITY_END:
        if step < BURST_START:
            stride, groups = DENSE_CADENCE, FIELD_CADENCE // DENSE_CADENCE
            count = min(groups, (BURST_START - step) // stride)
            count = min(count, (AUTHORITY_END - step) // stride)
        else:
            stride, groups = 1, 50
            count = min(groups, AUTHORITY_END - step)
        if count <= 0:
            break
        _log(f"instrumented authority window {step}->{step + stride * count} (stride={stride})")
        final_state, sampled_states, sampled_fields, observations, final_fields = advance_production_observed_block(
            resume_state,
            solid,
            p,
            stride,
            count,
        )
        sampled_step_metrics, sampled_substep_metrics, sampled_iterations, sampled_residuals, sampled_converged = observations
        sampled_step_metrics = np.asarray(sampled_step_metrics, dtype=np.float64)
        sampled_substep_metrics = np.asarray(sampled_substep_metrics, dtype=np.float64)
        sampled_iterations = np.asarray(sampled_iterations, dtype=np.int32)
        sampled_residuals = np.asarray(sampled_residuals, dtype=np.float64)
        sampled_converged = np.asarray(sampled_converged, dtype=bool)
        for frame in range(count):
            current_step = step + stride * (frame + 1)
            current_state = pf.State(
                phi=sampled_states.phi[frame],
                u=sampled_states.u[frame],
                v=sampled_states.v[frame],
                t=sampled_states.t[frame],
            )
            current_fields = jax.tree_util.tree_map(lambda item: item[frame], sampled_fields)
            metric_history = sampled_step_metrics[frame]
            current_phi = np.asarray(current_state.phi, dtype=np.float64)
            dense_rate = _host_phi_rate(current_phi, dense_previous_phi, p)
            formal_rate = None
            if current_step % 200 == 0:
                formal_rate = _host_phi_rate(current_phi, formal_previous_phi, p)
                formal_previous_phi = current_phi.copy()
            row = _sample_row(
                current_state,
                current_fields,
                solid,
                p,
                step=current_step,
                mobility=M_REF,
                formal_mass_reference=formal_mass_reference,
                diagnostic_horizon="authority_060_40k_to_50k",
                step_metrics=metric_history[-1],
                phase_rate_l2_dense=dense_rate,
                phase_rate_l2_formal=formal_rate,
                pressure_provenance="last_substep_projection_reconstructed_from_production_trajectory_state; transition advanced by pf.step_with_diagnostics",
            )
            dense_previous_phi = current_phi.copy()
            if current_step <= BURST_START or current_step % DENSE_CADENCE == 0:
                rows.append(row)
            if current_step > BURST_START:
                row_burst = dict(row)
                row_burst["phase_rate_l2_one_step_sample_over_dt"] = dense_rate
                burst_rows.append(row_burst)
            if current_step % FIELD_CADENCE == 0:
                saved = _save_field_snapshot(
                    snapshot_dir,
                    current_step,
                    current_state,
                    current_fields,
                    config,
                    pressure_provenance="last_substep_projection_reconstructed_from_production_trajectory_state; transition advanced by pf.step_with_diagnostics",
                )
                field_manifest.append(
                    {
                        "step": current_step,
                        "path": str(saved.relative_to(out)),
                        "pressure_provenance": "last_substep_projection_reconstructed_from_production_trajectory_state; transition advanced by pf.step_with_diagnostics",
                    }
                )
            if current_step in (48_000, 50_000):
                save_forensic_checkpoint(
                    checkpoint_dir / f"authority_060_step_{current_step:06d}.npz",
                    current_state,
                    config,
                    step_index=current_step,
                    kind="authority_48k" if current_step == 48_000 else "authority_50k",
                    parent_state_hashes=checkpoint_parent_hashes,
                )
                checkpoint_parent_hashes = _state_hashes(current_state)
        _append_history(
            history,
            (
                sampled_step_metrics,
                sampled_substep_metrics,
                sampled_iterations,
                sampled_residuals,
                sampled_converged,
            ),
        )
        step += stride * count
        resume_state = final_state
        _save_partial_authority(
            out, rows, burst_rows, history, resume_state, step, config, checkpoint_parent_hashes, formal_previous_phi
        )
        _write_json(snapshot_dir / "manifest.json", field_manifest)
        _log(f"authority forensic progress {step}/{AUTHORITY_END}; samples={len(rows)}, anti_alias={len(burst_rows)}")

    if step != AUTHORITY_END:
        _write_json(
            out / "partial" / "progress.json",
            {"stage": STAGE, "profile": "forensic", "status": "incomplete", "next_step": step, "unmeasured_until_step_50000": True},
        )
        raise RuntimeError(f"forensic authority stopped at {step}, expected {AUTHORITY_END}")
    _attach_contact_line_velocities(rows, Lx=p.Lx)
    _attach_contact_line_velocities(burst_rows, Lx=p.Lx)
    _write_json(out / "late_samples.json", rows)
    _write_json(out / "anti_alias_burst.json", burst_rows)
    flattened = _flatten_history(history)
    flattened.update({
        "substep_metric_names": np.asarray(SUBSTEP_METRIC_NAMES),
        "step_metric_names": np.asarray(STEP_METRIC_NAMES),
        "production_step_indices": np.arange(AUTHORITY_START + 1, AUTHORITY_END + 1, dtype=np.int64),
        "production_step_indices_for_substeps": np.repeat(np.arange(AUTHORITY_START + 1, AUTHORITY_END + 1, dtype=np.int64), 3),
        "internal_substep_indices": np.tile(np.arange(3, dtype=np.int8), AUTHORITY_END - AUTHORITY_START),
        "production_step_times": flattened["step_metrics"][:, STEP_METRIC_NAMES.index("state_time")],
        "internal_substep_times": flattened["substep_metrics"][:, :, SUBSTEP_METRIC_NAMES.index("substep_time")].reshape(-1),
    })
    _write_npz(out / "authority_production_step_diagnostics.npz", **flattened)
    _write_npz(out / "late_timeseries.npz", **_records_to_arrays(rows))
    _write_npz(out / "anti_alias_burst.npz", **_records_to_arrays(burst_rows))
    _write_json(snapshot_dir / "manifest.json", field_manifest)
    final_checkpoint_state, final_checkpoint_meta = load_forensic_checkpoint(
        checkpoint_dir / "authority_060_step_050000.npz",
        expected_config=config,
        expected_step=AUTHORITY_END,
        expected_kind="authority_50k",
    )
    candidate_comparison = {
        "previously_discovered_endpoint_candidate": candidate_manifest,
        "rerun_endpoint_state_hashes": _state_hashes(final_checkpoint_state),
        "trajectory_endpoint_matches_reused_candidate": (
            None if endpoint_candidate is None else _state_hashes(endpoint_candidate) == _state_hashes(final_checkpoint_state)
        ),
        "reused_endpoint_for_dense_window": False,
        "reason": "dense 40k-50k evidence was generated from a <=40k checkpoint; endpoint candidates are lineage-checked and only reused if exactly identical",
    }
    _write_json(out / "endpoint_checkpoint_comparison.json", candidate_comparison)
    _write_json(
        out / "partial" / "progress.json",
        {
            "stage": STAGE,
            "profile": "forensic",
            "status": "authority_50k_complete",
            "next_step": AUTHORITY_END,
            "config_fingerprint": _canonical_hash(config),
            "git_sha": get_git_sha(),
            "state_hashes": final_checkpoint_meta["state_hashes"],
            "completed_late_samples": len(rows),
            "completed_anti_alias_samples": len(burst_rows),
            "unmeasured_until_step_50000": False,
        },
    )
    return {
        "p": p,
        "solid": solid,
        "seed": seed,
        "config": config,
        "final_state": final_checkpoint_state,
        "rows": rows,
        "burst_rows": burst_rows,
        "history": flattened,
        "field_manifest": field_manifest,
        "candidate_comparison": candidate_comparison,
        "update_verification": update_verification,
        "resumed": resumed,
        "formal_mass_reference": formal_mass_reference,
    }


def _control_case(out: Path, target: float) -> dict[str, Any]:
    case_dir = out / "controls" / f"theta_{int(target):03d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    p, solid, seed, config = _make_case(target, N_value=N, dt=DT, M=M_REF)
    kind = f"control_{int(target):03d}"
    checkpoint_path = case_dir / "endpoint.npz"
    record_path = case_dir / "relaxation_record.json"
    raw_checkpoint = case_dir / "nonneutral_wetting_checkpoint.npz"
    record = None
    state = None
    if checkpoint_path.exists() and record_path.exists():
        try:
            state, metadata = load_forensic_checkpoint(
                checkpoint_path, expected_config=config, expected_kind=kind
            )
            cached = _read_json(record_path, {})
            if (
                cached.get("git_sha") == get_git_sha()
                and cached.get("config_fingerprint") == _canonical_hash(config)
                and cached.get("state_hashes") == metadata["state_hashes"]
            ):
                record = cached["record"]
                _log(f"reusing compatible converged-control evidence at theta={target:.0f}")
            else:
                raise CheckpointError("control sidecar SHA/config/state hashes do not match its checkpoint")
        except (OSError, ValueError, KeyError, CheckpointError, json.JSONDecodeError) as exc:
            _log(f"rejecting cached theta={target:.0f} control: {exc}")
            state, record = None, None
    if state is None or record is None:
        budgets = (10_000, 25_000, 50_000) if target == 90.0 else (50_000, 100_000)
        _log(f"running production CHNS matched control theta={target:.0f} through staged budgets {budgets}")
        run_kwargs = dict(
            ch_only=False,
            N=N,
            eps_factor=EPS_FACTOR,
            M=M_REF,
            dt=DT,
            R=RADIUS,
            wall_height=WALL_HEIGHT,
            sample_every=200,
            keep_samples=2_000,
            label=f"L1A-2j production control {target:.0f} deg",
            group="L1A-2j-control",
            dtype="float32",
        )
        record = nwa.run_relaxation(
            target,
            budgets=budgets,
            checkpoint_out=raw_checkpoint,
            **run_kwargs,
        )
        if target == 150.0 and int(record.get("steps", 0)) < 100_000:
            # The 150-degree comparison is explicitly staged to 100k. If the upstream staged
            # heuristic elects to stop at 50k, continue this control from its exact contract-11
            # checkpoint for the remaining fixed horizon; this never extends the 60-degree authority.
            remaining = 100_000 - int(record["steps"])
            if not raw_checkpoint.exists():
                raise RuntimeError("150-degree staged control did not save its 50k checkpoint")
            _log(f"150-degree control stage stopped at {record['steps']}; forcing the required diagnostic continuation to 100000")
            record = nwa.run_relaxation(
                target,
                budgets=budgets,
                fixed_steps=remaining,
                checkpoint_in=raw_checkpoint,
                checkpoint_out=raw_checkpoint,
                start_step=int(record["steps"]),
                mass_reference=float(record["mass_reference_initial"]),
                conserved_mass_reference=float(record["conserved_mass_reference_initial"]),
                prior_samples=list(record["samples"]),
                **run_kwargs,
            )
        if not raw_checkpoint.exists():
            raise RuntimeError(f"control run did not emit its checkpoint: {raw_checkpoint}")
        with np.load(raw_checkpoint, allow_pickle=False) as archive:
            raw_metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
            if raw_metadata.get("solver_contract_version") != SOLVER_CONTRACT:
                raise RuntimeError(f"theta={target:.0f} raw control checkpoint has a different contract")
            if raw_metadata.get("phase_storage_model") != pf.PHASE_ONLY_FLOAT64_STORAGE_MODEL:
                raise RuntimeError(f"theta={target:.0f} control checkpoint has a different phase storage model")
            state = pf.State(
                phi=jnp.asarray(archive["phi"], dtype=jnp.float64),
                u=jnp.asarray(archive["u"], dtype=jnp.float32),
                v=jnp.asarray(archive["v"], dtype=jnp.float32),
                t=jnp.asarray(archive["time"], dtype=jnp.float32),
            )
        control_step = int(record["steps"])
        metadata = save_forensic_checkpoint(
            checkpoint_path,
            state,
            config,
            step_index=control_step,
            kind=kind,
        )
        _write_json(
            record_path,
            {
                "stage": STAGE,
                "git_sha": get_git_sha(),
                "config_fingerprint": _canonical_hash(config),
                "state_hashes": metadata["state_hashes"],
                "record": record,
            },
        )
        _write_json(case_dir / "nwa_checkpoint_lineage.json", {"raw_checkpoint_metadata": raw_metadata, "forensic_metadata": metadata})
    samples = list(record.get("samples", []))
    exact_gate_samples = samples[1:] if len(samples) > 1 else samples
    gate = nwa._window_converged(exact_gate_samples, False, dict(nwa.CRITERIA), M=M_REF) if exact_gate_samples else {"converged": False}
    formal = [_formal_sample_from_control_sample(row) for row in exact_gate_samples]
    criterion = _criterion_window(formal, False, M=M_REF)
    endpoint_diagnostics = forensic_step_jit(state, solid, p)
    endpoint_fields = _field_components_from_checkpoint(state, solid, p)
    formal_mass_reference = float(pf.liquid_mass(seed.phi, solid, p))
    endpoint_row = _sample_row(
        state,
        endpoint_fields,
        solid,
        p,
        step=int(record["steps"]),
        mobility=M_REF,
        formal_mass_reference=formal_mass_reference,
        diagnostic_horizon=f"converged_control_theta_{int(target):03d}",
        step_metrics=np.asarray(endpoint_diagnostics.step_metrics),
        phase_rate_l2_formal=_as_float(record.get("phase_rate_l2")),
    )
    field_dir = out / "fields" / f"control_{int(target):03d}"
    field_dir.mkdir(parents=True, exist_ok=True)
    _save_field_snapshot(
        field_dir,
        int(record["steps"]),
        state,
        endpoint_fields,
        config,
        pressure_provenance="current_state_first_substep_pressure_field_diagnostic_not_stored_in_State",
    )
    return {
        "target_deg": float(target),
        "profile": "production CHNS control, unchanged contract-11 config",
        "run_record": record,
        "recorded_production_gate": gate,
        "stationarity_decomposition": criterion,
        "endpoint_state_hashes": _state_hashes(state),
        "endpoint_diagnostics": endpoint_row,
        "converged": bool(gate.get("converged", False)),
        "status": "measured" if record.get("steps") else "unmeasured",
    }


def _formal_sample_from_control_sample(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": row.get("step"),
        "time": row.get("time"),
        "mobility_scaled_time": row.get("mobility_scaled_time"),
        "measured_angle_deg": row.get("measured_angle_deg"),
        "free_energy": row.get("free_energy"),
        "phase_rate_l2": row.get("phase_rate_l2"),
        "max_speed": row.get("max_speed"),
    }


def _run_controls(out: Path) -> dict[str, Any]:
    controls = {}
    for target in CONTROL_TARGETS:
        controls[str(int(target))] = _control_case(out, target)
        _write_json(out / "controls" / f"theta_{int(target):03d}" / "control_analysis.json", controls[str(int(target))])
    return controls


@partial(jax.jit, static_argnums=(2, 3, 4))
def _advance_phase_only_samples(state, solid, p, steps_per_sample, num_samples):
    def one_step(current, _):
        next_state, info = pf.phase_only_step_with_diagnostics(current, solid, p)
        return next_state, (
            jnp.max(info.implicit_iterations),
            jnp.max(info.implicit_relative_residuals),
            jnp.all(info.implicit_converged),
        )

    def group(current, _):
        end_state, infos = jax.lax.scan(one_step, current, None, length=steps_per_sample)
        return end_state, (end_state, jnp.max(infos[0]), jnp.max(infos[1]), jnp.all(infos[2]))

    final_state, (sample_states, iterations, residuals, converged) = jax.lax.scan(
        group, state, None, length=num_samples
    )
    return final_state, sample_states, iterations, residuals, converged


def _run_phase_only_branch(
    out: Path,
    endpoint_state: pf.State,
    *,
    mobility: float,
    steps: int = 7_600,
    sample_every: int = 200,
) -> dict[str, Any]:
    label = "freeze_u_phase_only_Mref" if math.isclose(mobility, M_REF) else "freeze_u_phase_only_M4_diagnostic"
    p, solid, _seed, config = _make_case(TARGET, N_value=N, dt=DT, M=mobility)
    zero = jnp.zeros_like(endpoint_state.u)
    state = pf.State(endpoint_state.phi, zero, zero, endpoint_state.t)
    start_phi_hash = _array_hash(state.phi)
    directory = out / "diagnostics" / label
    cached_analysis = directory / "analysis.json"
    cached_checkpoint = directory / "endpoint.npz"
    if cached_analysis.exists() and cached_checkpoint.exists():
        try:
            cached_state, cached_meta = load_forensic_checkpoint(
                cached_checkpoint,
                expected_config=config,
                expected_step=AUTHORITY_END + steps,
                expected_kind=label,
            )
            cached = _read_json(cached_analysis, {})
            if (
                cached_meta.get("parent_state_hashes") == _state_hashes(endpoint_state)
                and cached.get("start_phi_hash") == start_phi_hash
                and cached.get("config_fingerprint") == _canonical_hash(config)
                and cached.get("final_phi_hash") == _array_hash(cached_state.phi)
            ):
                _log(f"reusing compatible diagnostic phase-only branch {label}")
                return cached
        except (OSError, ValueError, KeyError, CheckpointError, json.JSONDecodeError) as exc:
            _log(f"rejecting cached phase-only diagnostic {label}: {exc}")
    volume = np.asarray(solid.geometry.volume, dtype=np.float64)
    mass_ref_formal = float(pf.liquid_mass(state.phi, solid, p))
    mass_ref_hard = float(obs.liquid_mass(np.asarray(state.phi), np.asarray(solid.sdf), p.dx, p.dy))
    samples = [
        nwa._sample(
            state,
            state.phi,
            solid,
            p,
            ch_only=True,
            steps=AUTHORITY_END,
            cos_eff=math.cos(math.radians(TARGET)),
            phi_ref_mass=mass_ref_hard,
            M=mobility,
            volume=volume,
            phi_ref_conserved_mass=mass_ref_formal,
        )
    ]
    history = {"iterations": [], "residuals": [], "converged": []}
    groups = int(steps // sample_every)
    if groups * sample_every != steps:
        raise ValueError("phase-only diagnostic horizon must be divisible by its sample cadence")
    _log(f"running diagnostic freeze-u/CH-only branch {label}: {steps} steps, sample every {sample_every}")
    final_state, sampled_states, iterations, residuals, converged = _advance_phase_only_samples(
        state, solid, p, sample_every, groups
    )
    previous = np.asarray(state.phi, dtype=np.float64)
    phase_rows = []
    for index in range(groups):
        current_state = pf.State(
            sampled_states.phi[index], sampled_states.u[index], sampled_states.v[index], sampled_states.t[index]
        )
        sample_step = AUTHORITY_END + (index + 1) * sample_every
        row = nwa._sample(
            current_state,
            previous,
            solid,
            p,
            ch_only=True,
            steps=sample_step,
            cos_eff=math.cos(math.radians(TARGET)),
            phi_ref_mass=mass_ref_hard,
            M=mobility,
            volume=volume,
            phi_ref_conserved_mass=mass_ref_formal,
        )
        previous = np.asarray(current_state.phi, dtype=np.float64)
        row["diagnostic_only"] = True
        row["diagnostic_branch"] = label
        row["production_acceptance_evidence"] = False
        row["transport_advection"] = "disabled; phase_only_step_with_diagnostics supplies exactly zero u and v"
        row["pressure_policy"] = "not_applicable_phase_only_no_momentum_or_pressure_solve"
        phase_rows.append(row)
    final_hash = _array_hash(final_state.phi)
    velocities_zero = bool(np.array_equal(np.asarray(final_state.u), np.zeros_like(np.asarray(final_state.u))) and np.array_equal(np.asarray(final_state.v), np.zeros_like(np.asarray(final_state.v))))
    if not velocities_zero:
        raise RuntimeError("freeze-u diagnostic failed: velocity did not remain bitwise zero")
    exact_samples = samples + phase_rows
    gate = nwa._window_converged(exact_samples[1:], True, dict(nwa.CRITERIA), M=mobility)
    analysis = {
        "label": label,
        "diagnostic_only": True,
        "production_acceptance_evidence": False,
        "config": config,
        "config_fingerprint": _canonical_hash(config),
        "start_from_authority_step": AUTHORITY_END,
        "steps": int(steps),
        "sample_every_steps": int(sample_every),
        "formal_phase_mass_start_sum_V_phi": mass_ref_formal,
        "formal_phase_mass_final_sum_V_phi": float(pf.liquid_mass(final_state.phi, solid, p)),
        "formal_mass_drift": abs(float(pf.liquid_mass(final_state.phi, solid, p)) - mass_ref_formal) / max(abs(mass_ref_formal), 1e-30),
        "phase_state_bitwise_changed": start_phi_hash != final_hash,
        "start_phi_hash": start_phi_hash,
        "final_phi_hash": final_hash,
        "velocity_zero_bitwise_all_steps_endpoint": velocities_zero,
        "production_window_gate_diagnostic_only": gate,
        "stationarity_decomposition_diagnostic_only": _criterion_window(exact_samples[1:], True, M=mobility),
        "samples": exact_samples,
        "iterations_max": int(np.max(np.asarray(iterations))) if groups else 0,
        "relative_residual_max": float(np.max(np.asarray(residuals))) if groups else 0.0,
        "converged_all_samples": bool(np.all(np.asarray(converged))),
    }
    directory = out / "diagnostics" / label
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "analysis.json", analysis)
    _write_npz(
        directory / "phase_only_samples.npz",
        steps=np.asarray([row["step"] for row in exact_samples]),
        time=np.asarray([row["time"] for row in exact_samples]),
        angle=np.asarray([np.nan if row.get("measured_angle_deg") is None else row["measured_angle_deg"] for row in exact_samples]),
        free_energy=np.asarray([row["free_energy"] for row in exact_samples]),
        formal_mass=np.asarray([row["conserved_liquid_mass"] for row in exact_samples]),
        phase_rate_l2=np.asarray([row["phase_rate_l2"] for row in exact_samples]),
        cg_iterations=np.concatenate(([0], np.asarray(iterations, dtype=np.int32))),
        cg_residuals=np.concatenate(([0.0], np.asarray(residuals, dtype=np.float64))),
        cg_converged=np.concatenate(([True], np.asarray(converged, dtype=bool))),
    )
    save_forensic_checkpoint(
        directory / "endpoint.npz",
        final_state,
        config,
        step_index=AUTHORITY_END + steps,
        kind=label,
        parent_state_hashes=_state_hashes(endpoint_state),
    )
    return analysis


def _run_momentum_branch(
    out: Path,
    authority: dict[str, Any],
    *,
    label: str,
    steps: int = 5_000,
    sample_every: int = 10,
    freeze_phi: bool = False,
    capillary_scale: float = 1.0,
    velocity_reset: bool = False,
) -> dict[str, Any]:
    p, solid = authority["p"], authority["solid"]
    endpoint = authority["final_state"]
    start = pf.State(
        endpoint.phi,
        jnp.zeros_like(endpoint.u) if velocity_reset else endpoint.u,
        jnp.zeros_like(endpoint.v) if velocity_reset else endpoint.v,
        endpoint.t,
    )
    start_hashes = _state_hashes(start)
    mass_ref = float(pf.liquid_mass(start.phi, solid, p))
    start_phi_hash = _array_hash(start.phi)
    directory = out / "diagnostics" / label
    cached_analysis = directory / "analysis.json"
    cached_samples = directory / "samples.json"
    cached_checkpoint = directory / "endpoint.npz"
    if cached_analysis.exists() and cached_samples.exists() and cached_checkpoint.exists():
        try:
            cached_state, cached_meta = load_forensic_checkpoint(
                cached_checkpoint,
                expected_config=authority["config"],
                expected_step=AUTHORITY_END + steps,
                expected_kind=label,
            )
            cached = _read_json(cached_analysis, {})
            cached_rows = _read_json(cached_samples, [])
            if (
                cached_meta.get("parent_state_hashes") == _state_hashes(endpoint)
                and cached.get("start_state_hashes") == start_hashes
                and cached.get("end_state_hashes") == _state_hashes(cached_state)
                and len(cached_rows) > 1
            ):
                _log(f"reusing compatible diagnostic branch {label} after source/config/state-hash validation")
                return {**cached, "sample_rows": cached_rows}
        except (OSError, ValueError, KeyError, CheckpointError, json.JSONDecodeError) as exc:
            _log(f"rejecting cached diagnostic branch {label}: {exc}")
    rows = []
    initial_fields = _field_components_from_checkpoint(start, solid, p)
    rows.append(
        _sample_row(
            start,
            initial_fields,
            solid,
            p,
            step=AUTHORITY_END,
            mobility=M_REF,
            formal_mass_reference=mass_ref,
            diagnostic_horizon=label,
        )
    )
    rows[0]["phase_rate_l2_dense_sample_interval_over_dt"] = None
    rows[0]["phase_rate_l2_production_200_step_sample_over_dt"] = None
    history = {key: [] for key in ("step_metrics", "substep_metrics", "iterations", "residuals", "converged")}
    previous_dense_phi = np.asarray(start.phi, dtype=np.float64)
    previous_formal_phi = previous_dense_phi.copy()
    state = start
    step = AUTHORITY_END
    if steps % sample_every:
        raise ValueError("momentum diagnostic horizon must be divisible by sample cadence")
    groups_per_block = min(50, steps // sample_every)
    sample_stride = sample_every
    blocks = math.ceil((steps // sample_every) / groups_per_block)
    completed = 0
    _log(
        f"running diagnostic-only {label}: {steps} steps, sample every {sample_every}, "
        f"freeze_phi={freeze_phi}, capillary_scale={capillary_scale}, velocity_reset={velocity_reset}"
    )
    for _ in range(blocks):
        count = min(groups_per_block, steps // sample_every - completed // sample_every)
        if count <= 0:
            break
        final_state, sampled_states, sampled_fields, observations, final_fields = advance_forensic_block(
            state,
            solid,
            p,
            freeze_phi,
            capillary_scale,
            sample_stride,
            count,
        )
        step_hist, substep_hist, iteration_hist, residual_hist, convergence_hist = observations
        _append_history(
            history,
            (step_hist, substep_hist, iteration_hist, residual_hist, convergence_hist),
        )
        step_hist_np = np.asarray(step_hist, dtype=np.float64)
        for frame in range(count):
            current_step = step + sample_stride * (frame + 1)
            current = pf.State(
                sampled_states.phi[frame],
                sampled_states.u[frame],
                sampled_states.v[frame],
                sampled_states.t[frame],
            )
            fields = jax.tree_util.tree_map(lambda item: item[frame], sampled_fields)
            phi_now = np.asarray(current.phi, dtype=np.float64)
            dense_rate = _host_phi_rate(phi_now, previous_dense_phi, p)
            formal_rate = None
            if current_step % 200 == 0:
                formal_rate = _host_phi_rate(phi_now, previous_formal_phi, p)
                previous_formal_phi = phi_now.copy()
            row = _sample_row(
                current,
                fields,
                solid,
                p,
                step=current_step,
                mobility=M_REF,
                formal_mass_reference=mass_ref,
                diagnostic_horizon=label,
                step_metrics=step_hist_np[frame, -1],
                phase_rate_l2_dense=dense_rate,
                phase_rate_l2_formal=formal_rate,
            )
            rows.append(row)
            previous_dense_phi = phi_now.copy()
            if (current_step - AUTHORITY_END) % 1_000 == 0:
                snapshot_dir = out / "fields" / label
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                path = _save_field_snapshot(
                    snapshot_dir,
                    current_step,
                    current,
                    fields,
                    authority["config"],
                    pressure_provenance="actual_last_substep_projection_correction_applied_to_preceding_diagnostic_state",
                )
                _write_json(snapshot_dir / "manifest.json", [{"step": current_step, "path": str(path.relative_to(out))}])
        completed += count * sample_stride
        step += count * sample_stride
        state = final_state
    final_phi_hash = _array_hash(state.phi)
    if freeze_phi and final_phi_hash != start_phi_hash:
        raise RuntimeError(f"{label}: freeze_phi violated the bitwise phase hold")
    if capillary_scale == 0.0 and any(abs(float(row.get("capillary_global_l2") or 0.0)) > 0 for row in rows[1:]):
        raise RuntimeError(f"{label}: capillary-off diagnostic retained a nonzero applied capillary term")
    _attach_contact_line_velocities(rows, Lx=p.Lx)
    history_arrays = _flatten_history(history)
    directory = out / "diagnostics" / label
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "samples.json", rows)
    _write_npz(directory / "production_step_diagnostics.npz", **history_arrays)
    _write_npz(directory / "timeseries.npz", **_records_to_arrays(rows))
    checkpoint_meta = save_forensic_checkpoint(
        directory / "endpoint.npz",
        state,
        authority["config"],
        step_index=AUTHORITY_END + steps,
        kind=label,
        parent_state_hashes=_state_hashes(endpoint),
    )
    summary = {
        "label": label,
        "diagnostic_only": True,
        "production_acceptance_evidence": False,
        "production_solver_files_modified": False,
        "start_authority_step": AUTHORITY_END,
        "end_step": AUTHORITY_END + steps,
        "steps": int(steps),
        "sample_every_steps": int(sample_every),
        "config": authority["config"],
        "config_fingerprint": _canonical_hash(authority["config"]),
        "start_state_hashes": start_hashes,
        "end_state_hashes": checkpoint_meta["state_hashes"],
        "start_phi_sha256": start_phi_hash,
        "end_phi_sha256": final_phi_hash,
        "phi_bitwise_fixed": final_phi_hash == start_phi_hash if freeze_phi else None,
        "velocity_reset_to_zero_bitwise": bool(velocity_reset),
        "freeze_phi": bool(freeze_phi),
        "capillary_scale_diagnostic_only": float(capillary_scale),
        "pressure_policy": (
            "pressure is an algebraic per-substep correction, not stored in State; reset branch starts from u=v=0 "
            "and the unchanged production Poisson projection recomputes its first correction"
            if velocity_reset
            else "pressure is a per-substep correction, not a carried state; every step uses unchanged projection"
        ),
        "samples": len(rows),
        "sample_rows_path": str((directory / "samples.json").relative_to(out)),
        "series_summaries": _series_summaries(rows),
        "production_step_metric_names": list(STEP_METRIC_NAMES),
        "substep_metric_names": list(SUBSTEP_METRIC_NAMES),
        "endpoint_formal_mass_sum_V_phi": float(pf.liquid_mass(state.phi, solid, p)),
        "formal_mass_drift_from_branch_start": abs(float(pf.liquid_mass(state.phi, solid, p)) - mass_ref) / max(abs(mass_ref), 1e-30),
        "diagnostic_horizon": f"{steps} steps after authority endpoint; not a production continuation",
    }
    _write_json(directory / "analysis.json", summary)
    return {**summary, "sample_rows": rows}


def _candidate(label: str, status: str, evidence: Sequence[str], *, evidence_strength: str = "diagnostic") -> dict[str, Any]:
    if label not in PHENOMENOLOGY_LABELS and label not in MECHANISM_LABELS:
        raise ValueError(f"classification label is outside the frozen allowed sets: {label}")
    if status not in MECHANISM_STATUSES:
        raise ValueError(f"invalid evidence status {status!r}")
    return {"label": label, "status": status, "evidence_strength": evidence_strength, "evidence": list(evidence)}


def _aggregate_label(
    candidates: list[dict[str, Any]],
    *,
    inconclusive: str = "INCONCLUSIVE",
    multiple: str = "MULTIPLE_PHENOMENA",
) -> str:
    supported = [item["label"] for item in candidates if item["status"] == "SUPPORTED"]
    supported = [label for label in supported if label not in ("MULTIPLE_PHENOMENA", "MULTIPLE_CONTRIBUTORS", inconclusive)]
    if len(supported) == 1:
        return supported[0]
    if len(supported) > 1:
        return multiple
    return inconclusive


def _stationarity_for_authority(authority: dict[str, Any]) -> dict[str, Any]:
    rows = authority["rows"]
    formal_rows = [row for row in rows if int(row["step"]) % 200 == 0]
    formal_samples = [_formal_sample_from_row(row) for row in formal_rows]
    # Production run_relaxation supplies samples[1:] to its gate. Preserve that exact convention.
    formal_gate_samples = formal_samples[1:] if len(formal_samples) > 1 else formal_samples
    formal = _criterion_window(formal_gate_samples, False, M=M_REF)
    formal["sample_cadence_steps"] = 200
    formal["classifier_source"] = "nwa._window_converged on dense observations subsampled every 200 steps; unchanged CRITERIA"
    dense_samples = []
    for row in rows:
        dense_samples.append(
            {
                "step": row["step"],
                "time": row["time"],
                "mobility_scaled_time": row["mobility_scaled_time"],
                "measured_angle_deg": row["measured_angle_deg"],
                "free_energy": row["free_energy"],
                "phase_rate_l2": row.get("phase_rate_l2_dense_sample_interval_over_dt"),
                "max_speed": row["max_speed"],
            }
        )
    dense_gate_samples = dense_samples[1:] if len(dense_samples) > 1 else dense_samples
    dense = _criterion_window(dense_gate_samples, False, M=M_REF)
    dense["sample_cadence_steps"] = DENSE_CADENCE
    dense["status_role"] = "cadence sensitivity only; not production acceptance"
    persistence = _criterion_persistence(formal_gate_samples, ch_only=False, cadence_steps=200, M=M_REF)
    return {
        "production_cadence_200_step_gate": formal,
        "high_cadence_10_step_sensitivity_not_acceptance": dense,
        "rolling_window_persistence_at_production_cadence": persistence,
        "production_criteria_source": "examples/two_phase/production/nonneutral_wetting_audit.py::CRITERIA and ::_window_converged",
        "criteria_values_unchanged": dict(nwa.CRITERIA),
        "formal_mass_gate": {
            "definition": "sum_i(V_i * phi_i) via pf.liquid_mass; V_i are phase-control-volume cut-cell volumes",
            "max_drift_from_step0": max(float(row["formal_phase_mass_drift_from_reference"]) for row in rows),
            "threshold": MASS_DRIFT_LIMIT,
            "passed": bool(max(float(row["formal_phase_mass_drift_from_reference"]) for row in rows) <= MASS_DRIFT_LIMIT),
            "acceptance_role": "formal conserved phase-mass diagnostic",
        },
        "nonconserved_mass_channels": {
            "dxdy_sum_phi": "recorded as diagnostic_nonconserved; never gates acceptance",
            "legacy_hard_mask_mass": "recorded as diagnostic_nonconserved; never gates acceptance",
        },
    }


def _late_window_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    endpoints = ((40_000, 43_000), (43_000, 46_000), (46_000, 50_000))
    fields = ("measured_angle_deg", "contact_width", "contact_line_mean_speed", "kinetic_energy", "phase_rate_l2_instantaneous_per_dt", "formal_phase_mass_sum_V_phi")
    out = {}
    for start, end in endpoints:
        selected = [row for row in rows if start <= int(row["step"]) <= end]
        out[f"steps_{start}_{end}"] = {
            "n_samples": len(selected),
            "time_start": selected[0]["time"] if selected else None,
            "time_end": selected[-1]["time"] if selected else None,
            "signals": {name: _signal_summary([r["time"] for r in selected], [r.get(name) for r in selected]) for name in fields},
        }
    return out


def _spectral_peak_list(times: Sequence[float], values: Sequence[float | None], max_peaks: int = 5) -> list[dict[str, float]]:
    t = np.asarray(times, dtype=np.float64)
    v = np.asarray([np.nan if item is None else float(item) for item in values], dtype=np.float64)
    valid = np.isfinite(t) & np.isfinite(v)
    t, v = t[valid], v[valid]
    if len(t) < 16:
        return []
    dt = float(np.median(np.diff(t)))
    detrended = signal.detrend(v)
    freq, power = signal.periodogram(detrended, fs=1.0 / dt, window="hann", detrend=False, scaling="spectrum")
    peaks, _ = signal.find_peaks(power)
    peaks = [int(index) for index in peaks if freq[index] > 0.0]
    peaks.sort(key=lambda index: power[index], reverse=True)
    total = float(np.sum(power[freq > 0.0]))
    return [
        {
            "frequency_per_time": float(freq[index]),
            "period_time": float(1.0 / freq[index]),
            "power_fraction": float(power[index] / max(total, 1.0e-300)),
        }
        for index in peaks[:max_peaks]
    ]


def _detect_stick_slip(rows: list[dict[str, Any]], *, dt: float, dx: float) -> dict[str, Any]:
    steps = np.asarray([row["step"] for row in rows], dtype=np.int64)
    left = np.asarray([np.nan if row.get("left_contact_x_wrapped") is None else row["left_contact_x_wrapped"] for row in rows])
    right = np.asarray([np.nan if row.get("right_contact_x_wrapped") is None else row["right_contact_x_wrapped"] for row in rows])
    if len(steps) < 3 or not np.isfinite(left).all() or not np.isfinite(right).all():
        return {"status": "unmeasured", "episodes": 0, "rule": "requires contiguous attached left and right contact lines"}
    left_unwrapped = clk._unwrap_periodic_series(left, LENGTH)
    right_unwrapped = clk._unwrap_periodic_series(right, LENGTH)
    ldisp = np.abs(np.diff(left_unwrapped)) / dx
    rdisp = np.abs(np.diff(right_unwrapped)) / dx
    # Diagnostic pattern detector only: it classifies a resolved dwell (<0.01 cell per sample)
    # followed by a jump (>0.25 cell per sample), never a production acceptance threshold.
    dwell = (ldisp < 0.01) & (rdisp < 0.01)
    jump = (ldisp > 0.25) & (rdisp > 0.25)
    episodes = 0
    for index in np.flatnonzero(jump):
        lo = max(0, index - 20)
        if np.any(dwell[lo:index]):
            episodes += 1
    return {
        "status": "measured",
        "episodes": int(episodes),
        "dwell_intervals": int(np.count_nonzero(dwell)),
        "jump_intervals": int(np.count_nonzero(jump)),
        "left_displacement_cells_per_sample_max": float(np.nanmax(ldisp)) if len(ldisp) else None,
        "right_displacement_cells_per_sample_max": float(np.nanmax(rdisp)) if len(rdisp) else None,
        "sample_step_intervals": int(np.median(np.diff(steps))),
        "diagnostic_pattern_thresholds": {"dwell_cell_displacement": 0.01, "jump_cell_displacement": 0.25, "lookback_samples": 20},
        "rule": "reported as contact-line stick-slip only when repeated bilateral dwell-to-jump episodes appear in contiguous every-step burst",
    }


def _multi_frequency_status(peaks: Sequence[dict[str, Any]], *, minimum_power_fraction: float = 0.05) -> str:
    """Do not label spectral leakage/noise as multi-frequency behavior."""
    substantial = [peak for peak in peaks if float(peak.get("power_fraction", 0.0) or 0.0) >= minimum_power_fraction]
    return "SUSPECTED" if len(substantial) >= 2 else "NOT_TESTED"


def _phase_kinetics_status(angle_delta_deg: float | None, diagnostic_gate_converged: bool) -> str:
    """Only falsify CH-kinetics limitation when the matched phase-only branch actually settles."""
    if angle_delta_deg is None or not math.isfinite(float(angle_delta_deg)):
        return "NOT_TESTED"
    if abs(float(angle_delta_deg)) > float(nwa.CRITERIA["angle_tol_deg"]):
        return "SUPPORTED" if not diagnostic_gate_converged else "SUSPECTED"
    if diagnostic_gate_converged:
        return "FALSIFIED"
    return "SUSPECTED"


def _contact_pinning_status(stick_slip: dict[str, Any]) -> str:
    """Absence of stick-slip does not falsify static contact-line pinning."""
    episodes = int(stick_slip.get("episodes", 0) or 0)
    if episodes >= 3:
        return "SUPPORTED"
    if episodes > 0:
        return "SUSPECTED"
    return "NOT_TESTED"


def _classify_phenomenology(
    authority: dict[str, Any],
    stationarity: dict[str, Any],
    diagnostics: dict[str, Any],
    controls: dict[str, Any],
) -> dict[str, Any]:
    rows = authority["rows"]
    burst = authority["burst_rows"]
    summaries = _series_summaries(rows)
    angle = summaries["measured_angle_deg"]
    kinetic = summaries["kinetic_energy"]
    rate = summaries["phase_rate_l2_instantaneous_per_dt"]
    formal_pass = bool(stationarity["production_cadence_200_step_gate"]["production_gate"].get("converged", False))
    dense_pass = bool(stationarity["high_cadence_10_step_sensitivity_not_acceptance"]["production_gate"].get("converged", False))
    stick = _detect_stick_slip(burst, dt=DT, dx=LENGTH / N)
    candidates: list[dict[str, Any]] = []
    candidates.append(
        _candidate(
            "DECAYING_BUT_SLOW",
            "SUSPECTED" if (not formal_pass and kinetic.get("theil_sen_slope_95pct_high", 0.0) < 0.0 and rate.get("theil_sen_slope_95pct_high", 0.0) < 0.0) else "NOT_TESTED",
            ["rolling 40k-50k kinetic-energy and phase-rate trends are used; this is not inferred from a single endpoint"],
        )
    )
    dominant = angle.get("dominant_spectral_peak", {})
    period = dominant.get("period_time")
    cycles = None if period in (None, 0) else (float(rows[-1]["time"]) - float(rows[0]["time"])) / float(period)
    angle_cycle = bool(
        period is not None
        and angle.get("peak_to_peak", 0.0) is not None
        and angle.get("peak_to_peak", 0.0) > float(nwa.CRITERIA["angle_tol_deg"])
        and dominant.get("power_fraction", 0.0) >= 0.35
        and cycles is not None
        and cycles >= 3.0
    )
    candidates.append(
        _candidate(
            "ANGLE_LIMIT_CYCLE",
            "SUPPORTED" if angle_cycle else ("SUSPECTED" if dominant.get("power_fraction", 0.0) >= 0.25 else "NOT_TESTED"),
            [f"dominant angle period={period}; peak-to-peak={angle.get('peak_to_peak')}; estimated cycles={cycles}; spectral power fraction={dominant.get('power_fraction')}"],
            evidence_strength="high-cadence every-step burst and 10-step series" if angle_cycle else "spectral screen only",
        )
    )
    candidates.append(
        _candidate(
            "CONTACT_LINE_STICK_SLIP",
            "SUPPORTED" if stick.get("episodes", 0) >= 3 else ("SUSPECTED" if stick.get("episodes", 0) > 0 else "NOT_TESTED"),
            [f"every-step burst detector: {stick}"],
            evidence_strength="contiguous 2,000-step burst" if stick.get("status") == "measured" else "unmeasured",
        )
    )
    phase_only = diagnostics.get("freeze_u_phase_only_Mref", {})
    phase_rows = phase_only.get("samples", [])
    phase_angle_delta = None
    phase_energy_delta = None
    if len(phase_rows) >= 2:
        phase_angle_delta = (
            phase_rows[-1].get("measured_angle_deg") - phase_rows[0].get("measured_angle_deg")
            if phase_rows[-1].get("measured_angle_deg") is not None and phase_rows[0].get("measured_angle_deg") is not None
            else None
        )
        phase_energy_delta = phase_rows[-1].get("free_energy", 0.0) - phase_rows[0].get("free_energy", 0.0)
    phase_active = bool(
        phase_angle_delta is not None
        and (abs(phase_angle_delta) > float(nwa.CRITERIA["angle_tol_deg"]) or abs(float(phase_energy_delta)) > 1.0e-8)
    )
    candidates.append(
        _candidate(
            "PHASE_ONLY_NONSTATIONARITY",
            "SUPPORTED" if phase_active and not phase_only.get("production_window_gate_diagnostic_only", {}).get("converged", False) else ("FALSIFIED" if phase_only and not phase_active else "NOT_TESTED"),
            [f"M_ref freeze-u/CH-only branch: angle delta={phase_angle_delta}, free-energy delta={phase_energy_delta}, diagnostic convergence={phase_only.get('production_window_gate_diagnostic_only', {}).get('converged')}"],
        )
    )
    freeze = diagnostics.get("freeze_phi", {})
    freeze_rows = freeze.get("sample_rows", [])
    if len(freeze_rows) > 2:
        k0 = float(freeze_rows[0].get("kinetic_energy", 0.0) or 0.0)
        k1 = float(freeze_rows[-1].get("kinetic_energy", 0.0) or 0.0)
        persistence = k1 / max(k0, 1.0e-30)
        hydro_status = "SUPPORTED" if persistence >= 0.8 else ("FALSIFIED" if persistence <= 0.1 else "SUSPECTED")
    else:
        persistence, hydro_status = None, "NOT_TESTED"
    candidates.append(
        _candidate(
            "HYDRODYNAMIC_NONSTATIONARITY",
            hydro_status,
            [f"freeze-phi 5k-step kinetic-energy retention={persistence}; phase field is checked bitwise fixed"],
        )
    )
    reset = diagnostics.get("velocity_reset", {})
    reset_rows = reset.get("sample_rows", [])
    if reset_rows:
        reset_final_speed = reset_rows[-1].get("max_speed")
        reset_initial_speed = reset_rows[0].get("max_speed")
        reset_generated = reset_final_speed is not None and reset_initial_speed is not None and reset_final_speed > reset_initial_speed + 1.0e-8
        reset_evidence = f"velocity-reset path starts at u=v=0; max speed {reset_initial_speed} -> {reset_final_speed}"
        reset_status = "SUPPORTED" if reset_generated else "SUSPECTED"
    else:
        reset_evidence, reset_status = "velocity-reset branch unmeasured", "NOT_TESTED"
    candidates.append(_candidate("STEADY_RECIRCULATION", "SUSPECTED" if reset_status == "SUPPORTED" and formal_pass else "NOT_TESTED", [reset_evidence, f"production stationarity gate={formal_pass}; see kinetic-energy trend and controls"]))
    candidates.append(
        _candidate(
            "NOISY_STATIONARY_PLATEAU",
            "SUPPORTED" if formal_pass and angle.get("peak_to_peak", math.inf) <= 2.0 * float(nwa.CRITERIA["angle_tol_deg"]) else "NOT_TESTED",
            [f"production 200-step gate={formal_pass}; angle high-cadence peak-to-peak={angle.get('peak_to_peak')}"],
        )
    )
    candidates.append(
        _candidate(
            "CLASSIFIER_FALSE_NEGATIVE",
            "SUSPECTED" if (not formal_pass and dense_pass) else "NOT_TESTED",
            [f"production-cadence gate={formal_pass}; 10-step sensitivity gate={dense_pass}; dense gate is not production acceptance"],
        )
    )
    angle_peaks = _spectral_peak_list([row["time"] for row in rows], [row.get("measured_angle_deg") for row in rows])
    multi_frequency_status = _multi_frequency_status(angle_peaks)
    candidates.append(
        _candidate(
            "MULTI_FREQUENCY_OSCILLATION",
            multi_frequency_status,
            [f"angle-spectrum peaks={angle_peaks}; at least two peaks each need 5% spectral power even to be suspected; stability across windows is required for support"],
        )
    )
    label = _aggregate_label(candidates, inconclusive="INCONCLUSIVE")
    return {
        "allowed_labels": list(PHENOMENOLOGY_LABELS),
        "label": label,
        "status": "ASSIGNED" if label != "INCONCLUSIVE" else "INCONCLUSIVE",
        "candidates": candidates,
        "production_stationarity_gate": formal_pass,
        "high_cadence_gate_is_sensitivity_only": True,
        "high_cadence_sensitivity_gate_converged": dense_pass,
        "contact_line_stick_slip_diagnostic": stick,
        "spectral_peaks_angle_10_step": _spectral_peak_list([row["time"] for row in rows], [row.get("measured_angle_deg") for row in rows]),
        "spectral_peaks_angle_every_step_burst": _spectral_peak_list([row["time"] for row in burst], [row.get("measured_angle_deg") for row in burst]),
        "controls_converged": {key: bool(value.get("converged")) for key, value in controls.items()},
        "caveat": "Diagnostic classifier only; production gate and endpoint remain unchanged. No continuation beyond 50k is used for acceptance.",
    }


def _mechanism_matrix(
    authority: dict[str, Any],
    stationarity: dict[str, Any],
    diagnostics: dict[str, Any],
    controls: dict[str, Any],
    *,
    dt_half: dict[str, Any] | None = None,
) -> dict[str, Any]:
    history = authority["history"]
    values = np.asarray(history.get("substep_metrics", np.zeros((0, 3, len(SUBSTEP_METRIC_NAMES)))), dtype=np.float64)
    index = {name: idx for idx, name in enumerate(SUBSTEP_METRIC_NAMES)}
    if values.size:
        flattened = values.reshape(-1, values.shape[-1])
        proj_before = flattened[:, index["divergence_before_l2"]]
        proj_after = flattened[:, index["divergence_after_l2"]]
        pressure_residual = flattened[:, index["poisson_residual_l2"]]
        recon = np.maximum(flattened[:, index["term_reconstruction_u_linf"]], flattened[:, index["term_reconstruction_v_linf"]])
        projection_stats = {
            "n_internal_substeps": int(len(flattened)),
            "divergence_before_l2_median": float(np.median(proj_before)),
            "divergence_after_l2_median": float(np.median(proj_after)),
            "projection_l2_reduction_median": float(np.median(proj_after / np.maximum(proj_before, 1.0e-30))),
            "projection_l2_reduction_max": float(np.max(proj_after / np.maximum(proj_before, 1.0e-30))),
            "poisson_residual_l2_median": float(np.median(pressure_residual)),
            "poisson_residual_l2_max": float(np.max(pressure_residual)),
            "momentum_term_reconstruction_linf_max": float(np.max(recon)),
            "exact_production_step_transition_validated": True,
        }
    else:
        projection_stats = {"status": "unmeasured"}
    controls_comparison = {
        target: {
            "converged": value.get("converged"),
            "step": value.get("run_record", {}).get("steps"),
            "angle_deg": value.get("endpoint_diagnostics", {}).get("measured_angle_deg"),
            "kinetic_energy": value.get("endpoint_diagnostics", {}).get("kinetic_energy"),
            "max_speed": value.get("endpoint_diagnostics", {}).get("max_speed"),
            "pressure_projection_l2": value.get("endpoint_diagnostics", {}).get("pressure_projection_global_l2"),
            "capillary_l2": value.get("endpoint_diagnostics", {}).get("capillary_global_l2"),
        }
        for target, value in controls.items()
    }
    freeze = diagnostics.get("freeze_phi", {})
    capoff = diagnostics.get("freeze_phi_capillary_off", {})
    phase = diagnostics.get("freeze_u_phase_only_Mref", {})
    reset = diagnostics.get("velocity_reset", {})
    normal_kin = [row.get("kinetic_energy") for row in freeze.get("sample_rows", [])]
    off_kin = [row.get("kinetic_energy") for row in capoff.get("sample_rows", [])]
    capillary_separation = None
    if normal_kin and off_kin:
        capillary_separation = {
            "freeze_phi_normal_final_over_initial_kinetic": float(normal_kin[-1] / max(normal_kin[0], 1.0e-30)),
            "freeze_phi_capillary_off_final_over_initial_kinetic": float(off_kin[-1] / max(off_kin[0], 1.0e-30)),
            "matched_steps": min(len(normal_kin), len(off_kin)) - 1,
            "diagnostic_only": True,
        }
    if projection_stats.get("status") == "unmeasured":
        projection_status = "NOT_TESTED"
        projection_evidence = ["no exact per-substep projection record"]
    else:
        reduction = projection_stats["projection_l2_reduction_median"]
        max_resid = projection_stats["poisson_residual_l2_max"]
        projection_status = "FALSIFIED" if reduction < 1.0e-5 and max_resid < 1.0e-6 else ("SUSPECTED" if reduction > 1.0e-2 or max_resid > 1.0e-3 else "NOT_TESTED")
        projection_evidence = [f"projection reduction median={reduction}; pressure-Poisson residual L2 max={max_resid}; classifier thresholds unchanged"]
    phase_delta = None
    if phase.get("samples") and len(phase["samples"]) > 1:
        p_rows = phase["samples"]
        phase_delta = abs(float(p_rows[-1]["measured_angle_deg"]) - float(p_rows[0]["measured_angle_deg"])) if p_rows[-1].get("measured_angle_deg") is not None and p_rows[0].get("measured_angle_deg") is not None else None
    if phase:
        phase_gate = bool(phase.get("production_window_gate_diagnostic_only", {}).get("converged", False))
        phase_status = _phase_kinetics_status(phase_delta, phase_gate)
        phase_evidence = [f"matched freeze-u M_ref phase-only angle drift over {phase.get('steps')} steps={phase_delta} deg; phase-only convergence={phase_gate}; energy/mass traces and CH solver residuals retained"]
    else:
        phase_status, phase_evidence = "NOT_TESTED", ["freeze-u branch unmeasured"]
    if capillary_separation is None:
        capillary_status, capillary_evidence = "NOT_TESTED", ["matched capillary-off freeze-phi A/B unmeasured"]
    else:
        normal_ratio = capillary_separation["freeze_phi_normal_final_over_initial_kinetic"]
        off_ratio = capillary_separation["freeze_phi_capillary_off_final_over_initial_kinetic"]
        capillary_status = "SUPPORTED" if normal_ratio >= 0.8 and off_ratio <= 0.1 else ("FALSIFIED" if abs(normal_ratio - off_ratio) < 0.1 else "SUSPECTED")
        capillary_evidence = [f"same-state, same-step freeze-phi A/B: {capillary_separation}; exact capillary term/work history is available"]
    if freeze:
        wall_samples = freeze.get("sample_rows", [])
        solid_speed = [float(row["penalized_region_speed_max"]) for row in wall_samples if row.get("penalized_region_speed_max") is not None]
        brinkman_work = [float(row["brinkman_global_work_proxy"]) for row in wall_samples if row.get("brinkman_global_work_proxy") is not None]
        brinkman_l2 = [float(row["brinkman_global_l2"]) for row in wall_samples if row.get("brinkman_global_l2") is not None]
        if not solid_speed or not brinkman_work:
            br_status = "NOT_TESTED"
        elif np.nanmax(solid_speed) < 1.0e-6 and np.nanmax(np.abs(brinkman_work)) < 1.0e-8:
            br_status = "FALSIFIED"
        else:
            br_status = "SUSPECTED"
        br_evidence = [
            "Brinkman damping remains the exact existing implicit factor; term/work, chi>0.5 velocity, wall shells, and contact-line neighborhoods are measured; no parameter change is made",
            f"max chi>0.5 speed={max(solid_speed) if solid_speed else None}; max |Brinkman work proxy|={max(map(abs, brinkman_work)) if brinkman_work else None}; median Brinkman L2={float(np.median(brinkman_l2)) if brinkman_l2 else None}",
        ]
    else:
        br_status, br_evidence = "NOT_TESTED", ["freeze-phi/wall diagnostic unmeasured"]
    stick = _detect_stick_slip(authority["burst_rows"], dt=DT, dx=LENGTH / N)
    contact_status = _contact_pinning_status(stick)
    seam_values = [row.get("y_seam_v_jump_over_rms") for row in authority["rows"] if row.get("y_seam_v_jump_over_rms") is not None]
    topology_values = [row.get("liquid_components_periodic_x") for row in authority["rows"]]
    seam_ratio = float(np.nanmedian(seam_values)) if seam_values else None
    y_status = "FALSIFIED" if seam_ratio is not None and seam_ratio < 0.05 and all(value == 1 for value in topology_values) else ("SUSPECTED" if seam_ratio is not None and seam_ratio > 0.25 else "NOT_TESTED")
    y_evidence = [f"y-periodic seam v-jump/rms median={seam_ratio}; periodic-x liquid component counts={sorted(set(topology_values))}; seam is not a wall"]
    if dt_half is None:
        dt_status, dt_evidence = "NOT_TESTED", ["conditional matched-physical-time dt/2 diagnostic was not triggered or requested"]
    else:
        dt_status, dt_evidence = dt_half.get("mechanism_status", "SUSPECTED"), ["diagnostic full-CHNS dt/2 branch; never production evidence", str(dt_half.get("comparison", {}))]
    candidates = [
        _candidate("PHASE_KINETICS_LIMITED", phase_status, phase_evidence),
        _candidate("CAPILLARY_PRESSURE_IMBALANCE", capillary_status, capillary_evidence),
        _candidate("PRESSURE_PROJECTION_LIMITED", projection_status, projection_evidence),
        _candidate("BRINKMAN_WALL_COUPLING", br_status, br_evidence),
        _candidate("CONTACT_LINE_PINNING", contact_status, [str(stick)]),
        _candidate("Y_PERIODIC_TOPOLOGY_COUPLING", y_status, y_evidence),
        _candidate("TIME_SPLITTING_OR_DT_SENSITIVITY", dt_status, dt_evidence),
        _candidate("MOMENTUM_VISCOSITY_DISCRETIZATION", "NOT_TESTED", ["no viscosity-operator ablation; exact applied nu*lap(u) terms and localization are measured, but no production changes or unsupported inference"]),
        _candidate("VARIABLE_DENSITY_COUPLING", "NOT_TESTED", ["rho(phi) range and overlap are measured; no density-model ablation is permitted in this stage"]),
        _candidate("CAPILLARY_DENSITY_SCALING", "NOT_TESTED", ["production capillary denominator rho_l is preserved; no alternate denominator ablation is run"]),
        _candidate("CLASSIFIER_LOGIC_ONLY", "SUSPECTED" if stationarity["high_cadence_10_step_sensitivity_not_acceptance"]["production_gate"].get("converged") and not stationarity["production_cadence_200_step_gate"]["production_gate"].get("converged") else "NOT_TESTED", ["formal and high-cadence production criterion evaluations are both retained; cadence sensitivity cannot close production acceptance"]),
    ]
    overall = _aggregate_label(candidates, inconclusive="INCONCLUSIVE", multiple="MULTIPLE_CONTRIBUTORS")
    return {
        "allowed_labels": list(MECHANISM_LABELS),
        "label": overall,
        "status": "ASSIGNED" if overall != "INCONCLUSIVE" else "INCONCLUSIVE",
        "candidates": candidates,
        "exact_discrete_projection_and_momentum_reconstruction": projection_stats,
        "matched_converged_controls": controls_comparison,
        "freeze_phi_capillary_ablation_comparison": capillary_separation or {"status": "unmeasured"},
        "freeze_u_phase_angle_drift_deg": phase_delta,
        "pressure_handling": "pressure is not a stored state; diagnostic fields independently reconstruct the per-substep projection correction from the production trajectory state, with a strict restart cross-check; the authority carry is advanced only by pf.step_with_diagnostics",
        "open_blockers_preserved": ["W-CONTACT-ANGLE=open", "N-CH-MASS-PRECISION=resolved_in_contract_v11", "N-WALL-ALIGNMENT-TRANSPORT-DOMAIN=resolved_in_contract_v9; regression confirmed"],
        "control_comparison_note": "90-degree and staged 150-degree production controls are used only at their own converged endpoints; all mechanisms not tested by matched evidence remain NOT_TESTED.",
    }


def _run_dt_half_diagnostic(out: Path, authority: dict[str, Any], *, matched_time: float = 4.0) -> dict[str, Any]:
    """Optional full-CHNS dt/2 diagnostic, matched to an unchanged-dt horizon in physical time."""
    base_p, base_solid = authority["p"], authority["solid"]
    start = authority["final_state"]
    half_steps = int(round(matched_time / (DT / 2.0)))
    base_steps = int(round(matched_time / DT))
    p_half, solid_half, _seed_half, config_half = _make_case(TARGET, N_value=N, dt=DT / 2.0, M=M_REF)
    if not math.isclose(base_steps * DT, half_steps * p_half.dt, rel_tol=0.0, abs_tol=1.0e-12):
        raise AssertionError("dt-halving comparison horizons are not physically matched")
    _log(f"conditional diagnostic dt/2 A/B, matched physical horizon={matched_time:g}")
    base_end = _advance_standard(start, base_solid, base_p, base_steps)
    half_end = _advance_standard(start, solid_half, p_half, half_steps)
    base_diag = forensic_step_jit(base_end, base_solid, base_p)
    half_diag = forensic_step_jit(half_end, solid_half, p_half)
    base_fields = _field_components_from_checkpoint(base_end, base_solid, base_p)
    half_fields = _field_components_from_checkpoint(half_end, solid_half, p_half)
    base_row = _sample_row(
        base_end,
        base_fields,
        base_solid,
        base_p,
        step=AUTHORITY_END + base_steps,
        mobility=M_REF,
        formal_mass_reference=float(pf.liquid_mass(start.phi, base_solid, base_p)),
        diagnostic_horizon="dt_ablation_unchanged_dt_diagnostic",
        step_metrics=np.asarray(base_diag.step_metrics),
    )
    half_row = _sample_row(
        half_end,
        half_fields,
        solid_half,
        p_half,
        step=AUTHORITY_END + half_steps,
        mobility=M_REF,
        formal_mass_reference=float(pf.liquid_mass(start.phi, solid_half, p_half)),
        diagnostic_horizon="dt_ablation_half_dt_diagnostic",
        step_metrics=np.asarray(half_diag.step_metrics),
    )
    angle_delta = None
    if base_row.get("measured_angle_deg") is not None and half_row.get("measured_angle_deg") is not None:
        angle_delta = float(half_row["measured_angle_deg"] - base_row["measured_angle_deg"])
    output = {
        "status": "measured",
        "diagnostic_only": True,
        "production_acceptance_evidence": False,
        "base_dt": DT,
        "half_dt": float(p_half.dt),
        "base_steps": base_steps,
        "half_steps": half_steps,
        "matched_physical_time": matched_time,
        "Brinkman_eta_pen_base": float(base_p.eta_pen),
        "Brinkman_eta_pen_half": float(p_half.eta_pen),
        "note": "dt/2 follows the unchanged parameter-default relation eta_pen=2*dt; this is an explicitly diagnostic combined dt/penalty sensitivity and never production evidence",
        "base_endpoint": base_row,
        "half_endpoint": half_row,
        "comparison": {
            "angle_delta_half_minus_base_deg": angle_delta,
            "max_speed_ratio_half_over_base": half_row["max_speed"] / max(base_row["max_speed"], 1.0e-30),
            "formal_mass_drift_base": base_row["formal_phase_mass_drift_from_reference"],
            "formal_mass_drift_half": half_row["formal_phase_mass_drift_from_reference"],
        },
        "mechanism_status": "SUSPECTED",
        "config_half_dt": config_half,
    }
    directory = out / "diagnostics" / "dt_half"
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "analysis.json", output)
    save_forensic_checkpoint(
        directory / "base_endpoint.npz",
        base_end,
        authority["config"],
        step_index=AUTHORITY_END + base_steps,
        kind="dt_ablation_base_diagnostic",
        parent_state_hashes=_state_hashes(start),
    )
    save_forensic_checkpoint(
        directory / "half_dt_endpoint.npz",
        half_end,
        config_half,
        step_index=AUTHORITY_END + half_steps,
        kind="dt_ablation_half_diagnostic",
        parent_state_hashes=_state_hashes(start),
    )
    return output


def _initial_report(profile: str, out: Path, *, run_dt_half: bool | None = None) -> dict[str, Any]:
    configs = {
        "authority_60": production_config(target_deg=TARGET),
        "control_90": production_config(target_deg=90.0),
        "control_150": production_config(target_deg=150.0),
    }
    return {
        "stage": STAGE,
        "profile": profile,
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "repository_git_sha": get_git_sha(),
        "source_hashes": _source_hashes(),
        "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
        "production_semantics_changed": False,
        "production_configs": configs,
        "exact_production_defaults_preserved": {
            "target_60_N128_eps_2dx_dt_4e-3_M_ref_endpoint_50k": True,
            "M_ref": M_REF,
            "dt": DT,
            "wall_energy": "unchanged surface_energy model",
            "geometry_and_transport": "unchanged contract-11 sdf_cutcell_fv_v1 phase transport",
            "pressure_projection": "unchanged constant-coefficient production Poisson projection",
            "capillary_denominator": "unchanged rho_l",
            "variable_viscosity": "unchanged nu(phi)",
            "Brinkman_eta_pen": 2.0 * DT,
            "solver_contract_version": SOLVER_CONTRACT,
        },
        "production_stationarity_criteria": dict(nwa.CRITERIA),
        "unrun_sections": {
            "authority_60_step_50000": "unmeasured",
            "dense_40k_50k_every_10_steps": "unmeasured",
            "anti_alias_every_step_2000": "unmeasured",
            "sparse_fields_and_manifests": "unmeasured",
            "stationarity_criterion_decomposition_and_persistence": "unmeasured",
            "converged_90_control": "unmeasured",
            "staged_150_control": "unmeasured",
            "freeze_phi": "unmeasured",
            "freeze_u_Mref": "unmeasured",
            "freeze_u_M4_diagnostic": "unmeasured",
            "velocity_reset": "unmeasured",
            "capillary_off_diagnostic": "unmeasured",
            "dt_half_diagnostic": "unmeasured",
            "momentum_projection_reconstruction": "unmeasured",
            "phenomenology_classification": "unmeasured",
            "mechanism_classification": "unmeasured",
        },
        "conditional_dt_half_requested": run_dt_half,
        "formal_mass_definition": "sum_i(V_i*phi_i) = pf.liquid_mass(phi, solid, p)",
        "nonconserved_mass_diagnostics": ["dx*dy*sum(phi)", "obs.liquid_mass hard mask", "legacy total_mass"],
        "blockers": {
            "N-CH-MASS-PRECISION": "resolved_in_contract_v11; preserve",
            "N-WALL-ALIGNMENT-TRANSPORT-DOMAIN": "resolved_in_contract_v9; regression confirmed; preserve",
            "W-CONTACT-ANGLE": "open through L1A-2j absent dedicated closure",
        },
        "artifacts_directory": str(out),
        "status": "running",
        "phenomenology": {"status": "unmeasured", "label": "INCONCLUSIVE"},
        "mechanism": {"status": "unmeasured", "label": "INCONCLUSIVE"},
    }


def _write_report_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# {STAGE}: CHNS nonstationarity forensic audit",
        "",
        f"- **Profile:** `{report.get('profile')}`",
        f"- **Status:** `{report.get('status')}`",
        f"- **Git SHA:** `{report.get('repository_git_sha')}`",
        f"- **Solver contract:** `{report.get('solver_contract_version')}` (required 11)",
        "- **Scope:** Diagnostic/forensic only. Production solver semantics, defaults, contract, thresholds, and acceptance conditions are unchanged.",
        "",
        "## Frozen production configuration",
        "",
        "| Case | Settings |",
        "|---|---|",
    ]
    for name, config in report.get("production_configs", {}).items():
        lines.append(f"| {name} | N={config['N']}, eps={config['eps']:.8g} (2dx), dt={config['dt']:.8g}, M={config['M']:.8g}, target={config['target_deg']:.0f}° |")
    lines.extend(["", "## Classification", ""])
    phenomenology = report.get("phenomenology", {})
    mechanism = report.get("mechanism", {})
    lines.append(f"- **60° late-time phenomenology:** `{phenomenology.get('label', 'INCONCLUSIVE')}` — `{phenomenology.get('status', 'unmeasured')}`")
    lines.append(f"- **Mechanism:** `{mechanism.get('label', 'INCONCLUSIVE')}` — `{mechanism.get('status', 'unmeasured')}`")
    if phenomenology.get("candidates"):
        lines.extend(["", "### Phenomenology evidence labels", "", "| Label | Status | Evidence |", "|---|---|---|"])
        for item in phenomenology["candidates"]:
            lines.append(f"| `{item['label']}` | `{item['status']}` | {'; '.join(item.get('evidence', []))} |")
    if mechanism.get("candidates"):
        lines.extend(["", "### Mechanism evidence matrix", "", "| Mechanism | Status | Evidence |", "|---|---|---|"])
        for item in mechanism["candidates"]:
            lines.append(f"| `{item['label']}` | `{item['status']}` | {'; '.join(item.get('evidence', []))} |")
    lines.extend(["", "## Production stationarity decomposition", ""])
    stationarity = report.get("stationarity", {})
    formal = stationarity.get("production_cadence_200_step_gate", {})
    lines.append(f"- Production-cadence 200-step gate: `{formal.get('production_gate', {}).get('converged', 'unmeasured')}`; dense every-10-step calculation is sensitivity only.")
    criteria = formal.get("criteria", {})
    if criteria:
        lines.extend(["", "| Criterion | Raw | Threshold | Raw / threshold | Pass | Window |", "|---|---:|---:|---:|---|---|"])
        for name, value in criteria.items():
            if name == "strict_energy_observation_non_gating":
                continue
            lines.append(
                f"| {name} | {value.get('raw_value')} | {value.get('threshold')} | {value.get('normalized_value_raw_over_threshold')} | {value.get('passed')} | {value.get('window_samples_used')} samples / {value.get('window_mobility_time')} M·t |"
            )
    else:
        lines.append("- **unmeasured**")
    lines.extend(["", "## Matched controls", ""])
    controls = report.get("controls", {})
    if controls:
        lines.extend(["| Target | Steps | Converged | Angle | Max speed |", "|---:|---:|---|---:|---:|"])
        for target, control in controls.items():
            lines.append(
                f"| {target}° | {control.get('steps', 'unmeasured')} | {control.get('converged', 'unmeasured')} | "
                f"{control.get('final_angle_deg', 'unmeasured')} | {control.get('final_max_speed', 'unmeasured')} |"
            )
    else:
        lines.append("- Controls: **unmeasured**")
    lines.extend(["", "## Diagnostic branches", ""])
    diagnostics = report.get("diagnostics", {})
    if diagnostics:
        for name, result in diagnostics.items():
            lines.append(f"- `{name}`: `{result.get('status', 'measured')}`; diagnostic only; production acceptance evidence = `{result.get('production_acceptance_evidence', False)}`")
    else:
        lines.append("- All diagnostic branches: **unmeasured**")
    lines.extend(["", "## Mass, pressure, and frozen blockers", ""])
    mass = stationarity.get("formal_mass_gate", {})
    lines.append(f"- Formal mass is `sum_i(V_i*phi_i)`: max drift {mass.get('max_drift_from_step0', 'unmeasured')} vs diagnostic limit {mass.get('threshold', MASS_DRIFT_LIMIT)}; pass = `{mass.get('passed', 'unmeasured')}`.")
    lines.append("- `dx*dy*sum(phi)` and legacy hard-mask mass are diagnostic-only and never acceptance gates.")
    for key, value in report.get("blockers", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Unmeasured sections", ""])
    unrun = report.get("unrun_sections", {})
    remaining = [f"- `{key}`: **{value}**" for key, value in unrun.items() if str(value).startswith("unmeasured")]
    lines.extend(remaining if remaining else ["- None"])
    lines.extend(["", "## Artifacts", "", f"- Raw runs, snapshots, manifests, and resumable checkpoints: `{report.get('artifacts_directory')}`", "- Machine-readable report: `chns_nonstationarity_report.json`", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _publish_report(out: Path, report: dict[str, Any]) -> None:
    _write_json(out / "chns_nonstationarity_report.json", report)
    _write_report_markdown(out / "chns_nonstationarity_report.md", report)
    if report.get("profile") == "forensic":
        evidence_dir = Path(__file__).resolve().parent.parent / "evidence" / "l1a2j"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        _write_json(evidence_dir / "chns_nonstationarity_report.json", report)
        _write_report_markdown(evidence_dir / "chns_nonstationarity_report.md", report)


def _run_quick_checks(out: Path) -> dict[str, Any]:
    """CI-safe instrumentation validation; deliberately too short/small to imply physics evidence."""
    p, solid, state, config = _make_case(TARGET, N_value=24, dt=DT, M=M_REF)
    production, production_info = pf.step_with_diagnostics(state, solid, p)
    audited = forensic_step_jit(state, solid, p)
    state_pressure = _field_components_from_checkpoint(state, solid, p).pressure
    production_pressure = pf.pressure_field(state, solid, p)
    pressure_field_equal = np.array_equal(np.asarray(state_pressure), np.asarray(production_pressure))
    state_equal = all(
        np.array_equal(np.asarray(getattr(production, key)), np.asarray(getattr(audited.state, key)))
        for key in ("phi", "u", "v", "t")
    )
    momentum_error = float(np.max(np.asarray(audited.substep_metrics)[:, 14:16]))
    projection_after = np.asarray(audited.substep_metrics)[:, 2]
    projection_before = np.asarray(audited.substep_metrics)[:, 0]
    projection_ratio = float(np.max(projection_after / np.maximum(projection_before, 1.0e-30)))
    phi_hash = _array_hash(state.phi)
    freeze_result = forensic_step_jit(state, solid, p, True, 1.0)
    freeze_phi_bitwise = _array_hash(freeze_result.state.phi) == phi_hash
    zero_state = pf.State(state.phi, jnp.zeros_like(state.u), jnp.zeros_like(state.v), state.t)
    phase_only, _ = pf.phase_only_step_with_diagnostics(zero_state, solid, p)
    freeze_u_bitwise = bool(np.array_equal(np.asarray(phase_only.u), np.zeros_like(np.asarray(phase_only.u))) and np.array_equal(np.asarray(phase_only.v), np.zeros_like(np.asarray(phase_only.v))))
    phase_mass_start = float(pf.liquid_mass(zero_state.phi, solid, p))
    phase_mass_end = float(pf.liquid_mass(phase_only.phi, solid, p))
    formal_mass_relative_drift = abs(phase_mass_end - phase_mass_start) / max(abs(phase_mass_start), 1.0e-30)
    block_final, _, _, _, _ = advance_forensic_block(state, solid, p, False, 1.0, 1, 2)
    loop_final = _advance_standard(state, solid, p, 2)
    scan_observational = all(
        np.array_equal(np.asarray(getattr(block_final, key)), np.asarray(getattr(loop_final, key)))
        for key in ("phi", "u", "v", "t")
    )
    synthetic = []
    for index in range(80):
        synthetic.append(
            {
                "step": index * 200,
                "time": float(index),
                "mobility_scaled_time": 0.01 * index,
                "measured_angle_deg": 60.0,
                "free_energy": 1.0,
                "phase_rate_l2": 0.0,
                "max_speed": 0.0,
            }
        )
    classifier_pass = bool(nwa._window_converged(synthetic[1:], False, dict(nwa.CRITERIA), M=M_REF)["converged"])
    checkpoint_dir = out / "quick_checkpoints"
    checkpoint = checkpoint_dir / "roundtrip.npz"
    metadata = save_forensic_checkpoint(checkpoint, state, config, step_index=0, kind="quick_test")
    loaded, loaded_meta = load_forensic_checkpoint(checkpoint, expected_config=config, expected_step=0, expected_kind="quick_test")
    checkpoint_roundtrip = _state_hashes(loaded) == _state_hashes(state) and metadata["state_hashes"] == loaded_meta["state_hashes"]
    checks = {
        "production_transition_bitwise_equal": state_equal,
        "first_substep_pressure_field_bitwise_equal_to_pf_pressure_field": pressure_field_equal,
        "momentum_addends_reconstruct_rhs_linf": momentum_error,
        "pressure_projection_reduction_max_ratio": projection_ratio,
        "freeze_phi_bitwise_fixed_one_step": freeze_phi_bitwise,
        "freeze_u_zero_velocity_and_disabled_advection_path": freeze_u_bitwise,
        "freeze_u_formal_phase_mass_relative_drift_one_step": formal_mass_relative_drift,
        "observer_scan_matches_standard_integrator": scan_observational,
        "unchanged_production_classifier_synthetic_plateau_passes": classifier_pass,
        "checkpoint_state_hash_roundtrip": checkpoint_roundtrip,
        "production_contract_unchanged": int(pf.SOLVER_CONTRACT_VERSION) == SOLVER_CONTRACT,
        "production_criteria_unchanged": dict(nwa.CRITERIA) == FROZEN_PRODUCTION_CRITERIA,
        "diagnostic_mode_never_claims_production_acceptance": True,
    }
    tolerances = {
        "momentum_addends_reconstruct_rhs_linf": 2.0e-6,
        "pressure_projection_reduction_max_ratio": 5.0e-4,
        "freeze_u_formal_phase_mass_relative_drift_one_step": 1.0e-10,
    }
    passed = (
        state_equal
        and pressure_field_equal
        and momentum_error <= tolerances["momentum_addends_reconstruct_rhs_linf"]
        and projection_ratio <= tolerances["pressure_projection_reduction_max_ratio"]
        and freeze_phi_bitwise
        and freeze_u_bitwise
        and formal_mass_relative_drift <= tolerances["freeze_u_formal_phase_mass_relative_drift_one_step"]
        and scan_observational
        and classifier_pass
        and checkpoint_roundtrip
        and checks["production_contract_unchanged"]
    )
    result = {
        "stage": STAGE,
        "profile": "quick",
        "status": "passed" if passed else "failed",
        "diagnostic_only": True,
        "production_acceptance_evidence": False,
        "checks": checks,
        "tolerances": tolerances,
        "unmeasured": ["production N=128 authority to 50k", "matched controls", "phenomenology", "mechanism", "long-run stationarity"],
    }
    _write_json(out / "quick_checks.json", result)
    return result


def _load_completed_authority(out: Path, config: dict[str, Any], formal_mass_reference: float):
    progress = _read_json(out / "partial" / "progress.json", {})
    if progress.get("status") != "authority_50k_complete":
        return None
    if progress.get("config_fingerprint") != _canonical_hash(config) or progress.get("git_sha") != get_git_sha():
        return None
    if progress.get("source_hashes") not in (None, _source_hashes()):
        return None
    try:
        state, metadata = load_forensic_checkpoint(
            out / "checkpoints" / "authority_060_step_050000.npz",
            expected_config=config,
            expected_step=AUTHORITY_END,
            expected_kind="authority_50k",
        )
        if progress.get("state_hashes") != metadata.get("state_hashes"):
            return None
        rows = _read_json(out / "late_samples.json", [])
        burst_rows = _read_json(out / "anti_alias_burst.json", [])
        if not rows or int(rows[-1]["step"]) != AUTHORITY_END or len(burst_rows) < BURST_STEPS:
            return None
        diag_path = out / "authority_production_step_diagnostics.npz"
        with np.load(diag_path, allow_pickle=False) as archive:
            history = {
                key: [np.array(archive[key], copy=True)]
                for key in ("step_metrics", "substep_metrics", "iterations", "residuals", "converged")
            }
        config_case = _make_case(TARGET, N_value=N, dt=DT, M=M_REF)
        p, solid, seed, _ = config_case
        field_manifest = _read_json(out / "fields" / "authority_060" / "manifest.json", [])
        comparison = _read_json(out / "endpoint_checkpoint_comparison.json", {})
        update_verification = _read_json(out / "production_transition_verification.json", {})
        if update_verification.get("status") != "passed":
            return None
        return {
            "p": p,
            "solid": solid,
            "seed": seed,
            "config": config,
            "final_state": state,
            "rows": rows,
            "burst_rows": burst_rows,
            "history": _flatten_history(history),
            "field_manifest": field_manifest,
            "candidate_comparison": comparison,
            "update_verification": update_verification,
            "resumed": True,
            "formal_mass_reference": formal_mass_reference,
        }
    except (OSError, ValueError, KeyError, CheckpointError, json.JSONDecodeError):
        return None


def _short_control_record(control: dict[str, Any]) -> dict[str, Any]:
    record = control.get("run_record", {})
    return {
        "target_deg": control.get("target_deg"),
        "converged": control.get("converged"),
        "status": control.get("status"),
        "steps": record.get("steps"),
        "stop_reason": record.get("stop_reason"),
        "final_angle_deg": record.get("final_sampled_angle_deg"),
        "final_max_speed": record.get("final_max_speed"),
        "final_kinetic_energy": record.get("E_kin_final"),
        "production_gate": control.get("recorded_production_gate"),
        "stationarity_decomposition": control.get("stationarity_decomposition"),
        "endpoint_diagnostics": control.get("endpoint_diagnostics"),
        "endpoint_state_hashes": control.get("endpoint_state_hashes"),
    }


def _short_diagnostic_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key not in ("sample_rows", "samples")}


def run_forensic_profile(out: str | Path, *, run_dt_half: bool | None = None) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report = _initial_report("forensic", out, run_dt_half=run_dt_half)
    manifest = {
        "stage": STAGE,
        "profile": "forensic",
        "git_sha": get_git_sha(),
        "source_hashes": _source_hashes(),
        "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
        "production_semantics_changed": False,
        "config_fingerprints": {
            "authority_60": _canonical_hash(production_config(target_deg=60.0)),
            "control_90": _canonical_hash(production_config(target_deg=90.0)),
            "control_150": _canonical_hash(production_config(target_deg=150.0)),
        },
        "outputs": {key: "unmeasured" for key in report["unrun_sections"]},
        "resume_policy": "SHA + contract + config fingerprint + source hashes + per-array state hashes; incompatible checkpoints are rejected and rerun from an exact earlier state",
    }
    _write_json(out / "manifest.json", manifest)
    _publish_report(out, report)
    started = time.perf_counter()
    try:
        p, solid, seed, config = _make_case(TARGET, N_value=N, dt=DT, M=M_REF)
        endpoint_candidate, candidate_manifest = _reusable_endpoint(out, config)
        authority = _run_authority(out, endpoint_candidate, candidate_manifest)
        rows = authority["rows"]
        report["authority"] = {
            "status": "measured",
            "steps": AUTHORITY_END,
            "start_step": AUTHORITY_START,
            "config": config,
            "config_fingerprint": _canonical_hash(config),
            "resumed_or_reused": authority["resumed"],
            "endpoint_state_hashes": _state_hashes(authority["final_state"]),
            "production_transition_verification_at_restart": authority["update_verification"],
            "endpoint_checkpoint_comparison": authority["candidate_comparison"],
            "dense_every_10_steps_sample_count": len(rows),
            "dense_series_step_range": [int(rows[0]["step"]), int(rows[-1]["step"])],
            "anti_alias_every_step_sample_count": len(authority["burst_rows"]),
            "anti_alias_step_range": [int(authority["burst_rows"][0]["step"]), int(authority["burst_rows"][-1]["step"])],
            "field_snapshot_count": len(authority["field_manifest"]),
            "field_snapshot_cadence_steps": FIELD_CADENCE,
            "field_manifest": authority["field_manifest"],
            "raw_data": {
                "late_samples_json": "late_samples.json",
                "late_timeseries_npz": "late_timeseries.npz",
                "anti_alias_json": "anti_alias_burst.json",
                "anti_alias_npz": "anti_alias_burst.npz",
                "production_step_diagnostics_npz": "authority_production_step_diagnostics.npz",
                "field_snapshots": "fields/authority_060/",
            },
            "series_summaries": _series_summaries(rows),
            "momentum_spatial_summary": _momentum_spatial_summary(authority),
            "late_window_trends": _late_window_summaries(rows),
            "anti_alias_angle_spectrum": _spectral_peak_list(
                [row["time"] for row in authority["burst_rows"]],
                [row.get("measured_angle_deg") for row in authority["burst_rows"]],
            ),
            "formal_phase_mass_reference_sum_V_phi": authority["formal_mass_reference"],
            "authority_continuation_beyond_50k": False,
        }
        report["unrun_sections"]["authority_60_step_50000"] = "measured"
        report["unrun_sections"]["dense_40k_50k_every_10_steps"] = "measured"
        report["unrun_sections"]["anti_alias_every_step_2000"] = "measured"
        report["unrun_sections"]["sparse_fields_and_manifests"] = "measured"
        report["unrun_sections"]["momentum_projection_reconstruction"] = "measured"
        report["stationarity"] = _stationarity_for_authority(authority)
        report["unrun_sections"]["stationarity_criterion_decomposition_and_persistence"] = "measured"
        manifest["outputs"].update({key: report["unrun_sections"][key] for key in report["unrun_sections"]})
        _write_json(out / "manifest.json", manifest)
        _publish_report(out, report)

        controls = _run_controls(out)
        report["controls"] = {target: _short_control_record(control) for target, control in controls.items()}
        report["unrun_sections"]["converged_90_control"] = "measured" if controls.get("90", {}).get("converged") else "unmeasured"
        report["unrun_sections"]["staged_150_control"] = "measured" if controls.get("150", {}).get("converged") else "unmeasured"
        manifest["outputs"].update({key: report["unrun_sections"][key] for key in report["unrun_sections"]})
        _write_json(out / "manifest.json", manifest)
        _publish_report(out, report)

        diagnostics: dict[str, Any] = {}
        diagnostics["freeze_phi"] = _run_momentum_branch(
            out,
            authority,
            label="freeze_phi",
            steps=5_000,
            sample_every=10,
            freeze_phi=True,
        )
        diagnostics["freeze_phi_capillary_off"] = _run_momentum_branch(
            out,
            authority,
            label="freeze_phi_capillary_off",
            steps=5_000,
            sample_every=10,
            freeze_phi=True,
            capillary_scale=0.0,
        )
        diagnostics["velocity_reset"] = _run_momentum_branch(
            out,
            authority,
            label="velocity_reset",
            steps=5_000,
            sample_every=10,
            velocity_reset=True,
        )
        diagnostics["freeze_u_phase_only_Mref"] = _run_phase_only_branch(
            out, authority["final_state"], mobility=M_REF, steps=7_600, sample_every=200
        )
        diagnostics["freeze_u_phase_only_M4_diagnostic"] = _run_phase_only_branch(
            out, authority["final_state"], mobility=4.0 * M_REF, steps=7_600, sample_every=200
        )
        for key in ("freeze_phi", "capillary_off_diagnostic", "velocity_reset", "freeze_u_Mref", "freeze_u_M4_diagnostic"):
            report["unrun_sections"][key] = "measured"
        report["diagnostics"] = {key: _short_diagnostic_result(value) for key, value in diagnostics.items()}
        manifest["outputs"].update({key: report["unrun_sections"][key] for key in report["unrun_sections"]})
        _write_json(out / "manifest.json", manifest)
        _publish_report(out, report)

        stationarity = report["stationarity"]
        phenomenology = _classify_phenomenology(authority, stationarity, diagnostics, controls)
        mechanism = _mechanism_matrix(authority, stationarity, diagnostics, controls)
        dt_result = None
        # Half-dt is conditional: run only on an explicit request, or when matched freeze tests/controls
        # leave the mechanism inconclusive. It is an ablation, not a route to production acceptance.
        should_run_dt = bool(run_dt_half) if run_dt_half is not None else mechanism["label"] == "INCONCLUSIVE"
        if should_run_dt:
            dt_result = _run_dt_half_diagnostic(out, authority)
            diagnostics["dt_half"] = dt_result
            mechanism = _mechanism_matrix(authority, stationarity, diagnostics, controls, dt_half=dt_result)
            report["diagnostics"] = {key: _short_diagnostic_result(value) for key, value in diagnostics.items()}
            report["unrun_sections"]["dt_half_diagnostic"] = "measured_diagnostic_only"
        else:
            report["unrun_sections"]["dt_half_diagnostic"] = "unmeasured_conditional_not_triggered"
        report["phenomenology"] = phenomenology
        report["mechanism"] = mechanism
        report["unrun_sections"]["phenomenology_classification"] = "measured"
        report["unrun_sections"]["mechanism_classification"] = "measured"
        report["mechanism_falsification_evidence"] = {
            "matched_control_90": controls.get("90", {}).get("converged", False),
            "matched_control_150": controls.get("150", {}).get("converged", False),
            "freeze_phi_and_capillary_off_matched": bool(diagnostics.get("freeze_phi") and diagnostics.get("freeze_phi_capillary_off")),
            "freeze_u_Mref": bool(diagnostics.get("freeze_u_phase_only_Mref")),
            "velocity_reset": bool(diagnostics.get("velocity_reset")),
            "unmeasured_candidates_remain_NOT_TESTED": [item["label"] for item in mechanism.get("candidates", []) if item["status"] == "NOT_TESTED"],
        }
        authority_gate = bool(stationarity["production_cadence_200_step_gate"]["production_gate"].get("converged", False))
        formal_mass_pass = bool(stationarity["formal_mass_gate"]["passed"])
        controls_closed = bool(controls.get("90", {}).get("converged")) and bool(controls.get("150", {}).get("converged"))
        report["acceptance"] = {
            "production_60_degree_stationarity_gate_at_50k": authority_gate,
            "formal_phase_mass_drift_gate": formal_mass_pass,
            "all_solver_acceptance_gates_unchanged": True,
            "authority_endpoint_is_exactly_50k": True,
            "no_post_50k_authority_continuation": True,
            "control_90_and_staged_150_converged": controls_closed,
            "diagnostic_ablation_results_can_close_production_acceptance": False,
            "status": "production_gate_passed" if authority_gate and formal_mass_pass else "production_gate_not_passed_at_50k",
        }
        do_pheno = phenomenology["label"] in PHENOMENOLOGY_LABELS
        do_mech = mechanism["label"] in MECHANISM_LABELS
        matched_evidence = controls_closed and bool(report["mechanism_falsification_evidence"]["freeze_phi_and_capillary_off_matched"])
        authority_quality = (
            report["authority"]["steps"] == AUTHORITY_END
            and report["authority"]["dense_every_10_steps_sample_count"] >= 1000
            and report["authority"]["anti_alias_every_step_sample_count"] >= BURST_STEPS
            and report["authority"]["field_snapshot_count"] >= 40
        )
        report["definition_of_done"] = {
            "authority_ends_exactly_at_50k_with_dense_and_anti_alias_data": authority_quality,
            "phenomenology_assigned_or_explicitly_inconclusive": do_pheno,
            "mechanism_assigned_or_explicitly_inconclusive": do_mech,
            "matched_control_or_falsification_evidence_present": matched_evidence,
            "contract_and_upstream_blockers_unchanged": int(pf.SOLVER_CONTRACT_VERSION) == SOLVER_CONTRACT and report["blockers"]["W-CONTACT-ANGLE"] == "open through L1A-2j absent dedicated closure",
            "all_required_forensic_gates_closed": bool(do_pheno and do_mech and matched_evidence and authority_quality and formal_mass_pass),
        }
        report["elapsed_wall_seconds"] = float(time.perf_counter() - started)
        report["status"] = "complete" if report["definition_of_done"]["all_required_forensic_gates_closed"] else "incomplete_evidence_or_production_gate_open"
        manifest["outputs"].update({key: report["unrun_sections"][key] for key in report["unrun_sections"]})
        manifest["status"] = report["status"]
        manifest["completed_at_local"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
        manifest["definition_of_done"] = report["definition_of_done"]
        _write_json(out / "manifest.json", manifest)
        _write_json(out / "stationarity_decomposition.json", report["stationarity"])
        _write_json(out / "phenomenology_classification.json", report["phenomenology"])
        _write_json(out / "mechanism_matrix.json", report["mechanism"])
        _publish_report(out, report)
        return report
    except Exception as exc:
        report["status"] = "failed_or_incomplete"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        report["elapsed_wall_seconds"] = float(time.perf_counter() - started)
        manifest["status"] = report["status"]
        manifest["failure"] = report["failure"]
        _write_json(out / "manifest.json", manifest)
        _publish_report(out, report)
        raise


def run_controls_profile(out: str | Path) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report = _initial_report("controls", out)
    controls = _run_controls(out)
    report["controls"] = {target: _short_control_record(control) for target, control in controls.items()}
    report["unrun_sections"]["converged_90_control"] = "measured" if controls.get("90", {}).get("converged") else "unmeasured"
    report["unrun_sections"]["staged_150_control"] = "measured" if controls.get("150", {}).get("converged") else "unmeasured"
    report["status"] = "controls_profile_complete"
    _publish_report(out, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("quick", "controls", "forensic"), default="quick")
    parser.add_argument("--target", type=float, choices=(60.0,), default=60.0, help="The authority target is frozen to 60 degrees.")
    parser.add_argument("--out", type=Path, default=Path("artifacts/l1a2j"))
    dt_group = parser.add_mutually_exclusive_group()
    dt_group.add_argument("--run-dt-half", dest="run_dt_half", action="store_true", help="Run the optional matched-horizon diagnostic dt/2 ablation.")
    dt_group.add_argument("--no-dt-half", dest="run_dt_half", action="store_false", help="Do not run dt/2 even if the mechanism remains inconclusive.")
    parser.set_defaults(run_dt_half=None)
    args = parser.parse_args(argv)
    jax.config.update("jax_enable_x64", True)
    if pf.SOLVER_CONTRACT_VERSION != SOLVER_CONTRACT:
        parser.error(f"L1A-2j is pinned to solver contract 11; found {pf.SOLVER_CONTRACT_VERSION}")
    if args.profile == "quick":
        result = _run_quick_checks(args.out)
        print(json.dumps(_json_clean(result), indent=2, sort_keys=True))
        return 0 if result["status"] == "passed" else 1
    if args.profile == "controls":
        report = run_controls_profile(args.out)
        print(f"controls profile: {report['status']} — {args.out / 'chns_nonstationarity_report.md'}")
        return 0 if report["status"] == "controls_profile_complete" else 1
    report = run_forensic_profile(args.out, run_dt_half=args.run_dt_half)
    print(f"forensic profile: {report['status']} — {args.out / 'chns_nonstationarity_report.md'}")
    return 0 if report["status"] == "complete" else 2


def _momentum_spatial_summary(authority: dict[str, Any]) -> dict[str, Any]:
    rows = authority["rows"]
    history = authority["history"]
    step_times = [float(row["time"]) for row in rows]
    regions = ("global", "interface", "wall_0_1dx", "wall_1_2dx", "wall_2_4dx", "wall_gt_4dx", "contact_left", "contact_right")
    terms = {}
    for term in _TERM_FIELD_NAMES:
        regional = {}
        for region in regions:
            regional[region] = {}
            for metric in ("l2", "linf", "work_proxy"):
                values = np.asarray(
                    [np.nan if row.get(f"{term}_{region}_{metric}") is None else float(row[f"{term}_{region}_{metric}"]) for row in rows],
                    dtype=np.float64,
                )
                finite = values[np.isfinite(values)]
                summary = {
                    "n_samples": int(len(finite)),
                    "minimum": None if not len(finite) else float(np.min(finite)),
                    "median": None if not len(finite) else float(np.median(finite)),
                    "maximum": None if not len(finite) else float(np.max(finite)),
                    "p95_abs": None if not len(finite) else float(np.quantile(np.abs(finite), 0.95)),
                }
                if region == "global" and metric in ("l2", "work_proxy"):
                    summary["trend_and_spectrum"] = _signal_summary(step_times, values)
                regional[region][metric] = summary
        terms[term] = {"spatial_norms_every_10_steps": regional}
    matrix = np.asarray(history.get("substep_metrics", np.zeros((0, 3, len(SUBSTEP_METRIC_NAMES)))), dtype=np.float64)
    per_substep = {}
    if matrix.size:
        flat = matrix.reshape(-1, matrix.shape[-1])
        names = tuple(SUBSTEP_METRIC_NAMES)
        for term in ("advection", "viscous", "capillary", "gravity", "brinkman", "pressure_projection"):
            for suffix in ("l2", "linf", "work_proxy"):
                name = f"{term}_{suffix}"
                values = flat[:, names.index(name)]
                per_substep[name] = {
                    "count": int(len(values)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "median": float(np.median(values)),
                    "mean": float(np.mean(values)),
                    "p95_abs": float(np.quantile(np.abs(values), 0.95)),
                }
        for name in (
            "divergence_before_l2",
            "divergence_after_l2",
            "projection_reduction_l2",
            "poisson_residual_l2",
            "pressure_correction_l2",
            "pressure_gradient_l2",
            "term_reconstruction_u_linf",
            "term_reconstruction_v_linf",
        ):
            values = flat[:, names.index(name)]
            per_substep[name] = {
                "count": int(len(values)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "median": float(np.median(values)),
                "mean": float(np.mean(values)),
                "p95": float(np.quantile(values, 0.95)),
            }
    return {
        "norm_weighting": "unweighted momentum-grid cell sums multiplied by dx*dy; region masks localize diagnostics only; cut-cell volume is used only for formal phase mass/phase free energy",
        "force_field_sampling": "momentum terms are reconstructed from the exact production rhs/projection formulas at sampled production states; authority carry advances only through pf.step_with_diagnostics; sampled full-vector snapshots are diagnostic outputs and never feed back",
        "terms": terms,
        "per_internal_substep_global_norms_and_work": per_substep,
        "momentum_substep_metric_names": list(SUBSTEP_METRIC_NAMES),
        "pressure_is_algebraic_not_checkpoint_state": True,
    }


if __name__ == "__main__":
    raise SystemExit(main())
