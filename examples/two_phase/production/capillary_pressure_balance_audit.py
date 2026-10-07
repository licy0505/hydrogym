"""L1A-2k diagnostic-only capillary/pressure subspace audit.

The production solver is observed, never edited or retuned.  This module decomposes the exact
contract-11 capillary acceleration through the production centred-difference divergence/gradient
and the production ``m2_proj`` Poisson solve.  It is a production-operator decomposition, not a
claim that capillary force should be a pressure gradient or that the result explains the 60-degree
stationarity failure.

Run from ``examples/two_phase``::

    JAX_ENABLE_X64=1 python -m production.capillary_pressure_balance_audit --profile quick
    JAX_ENABLE_X64=1 python -m production.capillary_pressure_balance_audit --profile forensic

The forensic profile writes resumable state/report data below ``artifacts/l1a2k`` and the compact
JSON/Markdown evidence under ``evidence/l1a2k``.  Exact old snapshots are reused only after the
contract, Git SHA, source hashes, config fingerprint, array hashes, and step all validate.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

import phasefield as pf
from production import capillary_audit
from production import chns_nonstationarity_audit as chns
from production import contact_line_kinetics as clk
from production import nonneutral_wetting_audit as nwa

# Phase-only float64 storage is part of contract 11. This enables diagnostic reductions without
# promoting the production velocity dtype, which remains float32 in the matched cases.
jax.config.update("jax_enable_x64", True)

STAGE = "L1A-2k"
SOLVER_CONTRACT = 11
CASE_TARGETS = {"authority_060": 60.0, "control_090": 90.0, "control_150": 150.0}
CASE_REQUIRED_STEPS = {"authority_060": 50_000, "control_090": 27_200, "control_150": 100_000}
FROZEN_PHI_STEPS = 500


class AuditValidationError(RuntimeError):
    """Raised when a required production-operator validation fails closed."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_hash(value: Any) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    sources = {
        "phasefield": root / "phasefield.py",
        "capillary_pressure_balance_audit": Path(__file__).resolve(),
        "chns_nonstationarity_audit": root / "production" / "chns_nonstationarity_audit.py",
        "nonneutral_wetting_audit": root / "production" / "nonneutral_wetting_audit.py",
        "contact_line_kinetics": root / "production" / "contact_line_kinetics.py",
        "capillary_audit": root / "production" / "capillary_audit.py",
    }
    return {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in sources.items()}


def _function_source_record(name: str, fn: Callable[..., Any]) -> dict[str, Any]:
    target = getattr(fn, "fget", None) or fn
    try:
        source_lines, start_line = inspect.getsourcelines(target)
        source = "".join(source_lines)
        file_path = Path(inspect.getsourcefile(target) or "").resolve()
        return {
            "name": name,
            "file": str(file_path),
            "line_start": int(start_line),
            "line_end": int(start_line + len(source_lines) - 1),
            "sha256_source": _sha256_bytes(source.encode()),
        }
    except (OSError, TypeError):
        return {"name": name, "file": None, "line_start": None, "line_end": None, "sha256_source": None}


def operator_map() -> dict[str, Any]:
    """Return the exact production source/operator map used by this audit."""
    functions = [
        ("phasefield._ddx", pf._ddx),
        ("phasefield._ddy", pf._ddy),
        ("phasefield._lap", pf._lap),
        ("PhaseFieldParams.m2_proj", pf.PhaseFieldParams.m2_proj),
        ("phasefield.poisson_solve", pf.poisson_solve),
        ("phasefield.chemical_potential", pf.chemical_potential),
        ("phasefield._explicit_chemical_potential", pf._explicit_chemical_potential),
        ("phasefield.rhs", pf.rhs),
        ("phasefield.rho_of", pf.rho_of),
        ("phasefield.nu_of", pf.nu_of),
        ("phasefield.step_with_diagnostics", pf.step_with_diagnostics),
        ("phasefield.pressure_field", pf.pressure_field),
        ("phasefield.wall_energy_derivative", pf.wall_energy_derivative),
        ("phasefield.wall_measure_density", pf.wall_measure_density),
        ("chns_nonstationarity_audit._momentum_components", chns._momentum_components),
        ("chns_nonstationarity_audit._forensic_substep", chns._forensic_substep),
        ("nonneutral_wetting_audit.run_relaxation", nwa.run_relaxation),
        ("nonneutral_wetting_audit._window_converged", nwa._window_converged),
        ("contact_line_kinetics.contact_line_positions", clk.contact_line_positions),
    ]
    return {
        "stage": STAGE,
        "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
        "source_file_sha256": _source_hashes(),
        "source_functions": [_function_source_record(name, fn) for name, fn in functions],
        "production_momentum_and_pressure_operators": {
            "momentum_grid": "uniform cell-centred Nx by Ny periodic grid; momentum/Brinkman/projection are evaluated on every grid cell",  # noqa: E501
            "momentum_area_weight": "dx * dy uniformly on the full momentum grid; cut-cell phase volumes do not weight momentum or pressure projection",  # noqa: E501
            "gradient": "G_h q = (phasefield._ddx(q, dx), phasefield._ddy(q, dy)); centred periodic differences on axes 0 and 1",  # noqa: E501
            "divergence": "D_h a = phasefield._ddx(a_x, dx) + phasefield._ddy(a_y, dy); same centred periodic differences",  # noqa: E501
            "pressure_operator": "D_h G_h with Fourier symbol -PhaseFieldParams.m2_proj; all exact discrete null modes are removed by phasefield.poisson_solve",  # noqa: E501
            "pressure_solve": "q = pf.poisson_solve(D_h a, p.m2_proj), solving the production constant-coefficient periodic operator; no continuum-k^2 surrogate",  # noqa: E501
            "production_projection_stage": "per internal substep: damped predictor, div/dt, poisson_solve(..., p.m2_proj), then subtract dt * G_h pressure",  # noqa: E501
            "production_velocity_dtype": "p.dtype (float32 in all matched cases); phase/chemical-potential intermediates are float64 under phase_only_float64_v1",  # noqa: E501
            "force_density": "(SIGMA_NORM / We) * mu_h * G_h(phi)",
            "acceleration": "force_density / rho_l, with rho_l a spatially constant scalar in the production config",
            "rhs_cast": "pf.rhs sums momentum addends and casts the full u_rhs/v_rhs back to p.dtype; the audit records the raw expression, its production-dtype cast, and the exact marginal addend after the production RHS cast",  # noqa: E501
            "density_treatment": "rho_of(phi) is formed in rhs; with use_gravity=False, the current momentum RHS/projection has no variable-density inertia or variable-coefficient pressure weighting; nu_of(phi) remains the production viscosity model",  # noqa: E501
            "brinkman": "unchanged local damping 1 / (1 + (dt/3) * solid.chi / p.eta_pen) before each pressure projection",  # noqa: E501
            "phase_wall_energy": "the production chemical potential includes the selected wall-energy derivative once; its resulting contribution to mu_h * G_h(phi) is separately decomposed, without modifying wall energy or momentum",  # noqa: E501
        },
        "decomposition_definition": {
            "label": "production_operator_decomposition",
            "equations": [
                "D_h G_h q = D_h a_sigma^h, with q obtained from the production poisson_solve and m2_proj",
                "a_sigma,grad^h = G_h q",
                "a_sigma,perp^h = a_sigma^h - G_h q",
            ],
            "not_claimed": "This report does not call the result an exact orthogonal Hodge decomposition. A separate uniform-grid periodic adjoint/inner-product check is reported; numerical precision and the momentum-grid metric are stated explicitly.",  # noqa: E501
            "curl_label": "diagnostic_curl_v1 (the solver defines no canonical production curl operator)",
        },
        "mask_policy": {
            "interface": "fluid cell centres with 0.05 <= phi <= 0.95",
            "liquid": "fluid cell centres with phi >= 0.95",
            "gas": "fluid cell centres with phi <= 0.05",
            "wall_shells": "signed-distance bins fixed at integer multiples of dx; no post-hoc tuning",
            "contact_line": "contact_line_kinetics.contact_line_positions at level phi=0.5; periodic x distance and fixed wall plane, with radii 2dx and 4dx",  # noqa: E501
            "y_seam": "two- and four-cell strips adjacent to the periodic y seam",
            "overlap": "regional masks intentionally overlap; regional L2 fractions are not additive",
        },
    }


def _as_np64(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _array_hash(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    h = hashlib.sha256()
    h.update(array.dtype.str.encode())
    h.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    h.update(array.view(np.uint8))
    return h.hexdigest()


def _state_hashes(state: pf.State) -> dict[str, str]:
    return chns._state_hashes(state)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _l2(value: np.ndarray, mask: np.ndarray, area: float) -> float:
    array = _as_np64(value)
    active = np.asarray(mask, dtype=bool)
    return float(np.sqrt(np.sum(np.square(array[active]), dtype=np.float64) * float(area))) if np.any(active) else 0.0


def _vector_metrics(u: Any, v: Any, mask: np.ndarray, area: float) -> dict[str, Any]:
    ux, vy = _as_np64(u), _as_np64(v)
    active = np.asarray(mask, dtype=bool)
    n = int(np.count_nonzero(active))
    if not n:
        return {"n_cells": 0, "area": 0.0, "l2": 0.0, "linf": 0.0}
    magnitude = np.hypot(ux, vy)
    return {
        "n_cells": n,
        "area": float(n * area),
        "l2": float(np.sqrt(np.sum((ux[active] ** 2 + vy[active] ** 2), dtype=np.float64) * area)),
        "linf": float(np.max(magnitude[active])),
    }


def _scalar_metrics(value: Any, mask: np.ndarray, area: float) -> dict[str, Any]:
    array = _as_np64(value)
    active = np.asarray(mask, dtype=bool)
    n = int(np.count_nonzero(active))
    if not n:
        return {"n_cells": 0, "area": 0.0, "l2": 0.0, "linf": 0.0}
    return {
        "n_cells": n,
        "area": float(n * area),
        "l2": _l2(array, active, area),
        "linf": float(np.max(np.abs(array[active]))),
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(float(numerator)) or not math.isfinite(float(denominator)):
        return None
    if abs(float(denominator)) <= np.finfo(np.float64).tiny:
        return 0.0 if abs(float(numerator)) <= np.finfo(np.float64).tiny else None
    return float(numerator / denominator)


def _production_operator_arrays(ax: Any, ay: Any, p: pf.PhaseFieldParams) -> dict[str, Any]:
    """Decompose a momentum-dtype vector with the exact production D/G/Poisson operators."""
    source_x_np = np.asarray(ax)
    source_y_np = np.asarray(ay)
    expected_shape = (int(p.Nx), int(p.Ny))
    if source_x_np.shape != expected_shape or source_y_np.shape != expected_shape:
        raise ValueError(f"source vector shape must be {expected_shape}")
    if not np.isfinite(source_x_np).all() or not np.isfinite(source_y_np).all():
        raise AuditValidationError("non-finite source acceleration: refusing to report a decomposition")
    source_x = jnp.asarray(source_x_np, dtype=p.dtype)
    source_y = jnp.asarray(source_y_np, dtype=p.dtype)
    div_source = pf._ddx(source_x, p.dx) + pf._ddy(source_y, p.dy)
    potential = pf.poisson_solve(div_source, p.m2_proj)
    grad_x = pf._ddx(potential, p.dx)
    grad_y = pf._ddy(potential, p.dy)
    residual_x = source_x - grad_x
    residual_y = source_y - grad_y
    lap_potential = pf._ddx(grad_x, p.dx) + pf._ddy(grad_y, p.dy)
    div_residual = pf._ddx(residual_x, p.dx) + pf._ddy(residual_y, p.dy)
    solve_residual = lap_potential - div_source

    source64_x, source64_y = _as_np64(source_x), _as_np64(source_y)
    grad64_x, grad64_y = _as_np64(grad_x), _as_np64(grad_y)
    residual64_x, residual64_y = _as_np64(residual_x), _as_np64(residual_y)
    reconstructed_x = residual64_x + grad64_x
    reconstructed_y = residual64_y + grad64_y
    reconstruction_linf = float(
        max(np.max(np.abs(reconstructed_x - source64_x)), np.max(np.abs(reconstructed_y - source64_y)))
    )
    area = float(p.dx * p.dy)
    source_norm = _vector_metrics(source_x, source_y, np.ones(expected_shape, dtype=bool), area)["l2"]
    grad_norm = _vector_metrics(grad_x, grad_y, np.ones(expected_shape, dtype=bool), area)["l2"]
    residual_norm = _vector_metrics(residual_x, residual_y, np.ones(expected_shape, dtype=bool), area)["l2"]
    divergence_l2 = _scalar_metrics(div_source, np.ones(expected_shape, dtype=bool), area)["l2"]
    residual_divergence_l2 = _scalar_metrics(div_residual, np.ones(expected_shape, dtype=bool), area)["l2"]
    solve_residual_l2 = _scalar_metrics(solve_residual, np.ones(expected_shape, dtype=bool), area)["l2"]
    work_dtype = np.dtype(np.asarray(source_x).dtype)
    eps_work = float(np.finfo(work_dtype).eps)
    derivative_norm_bound = math.sqrt((1.0 / float(p.dx)) ** 2 + (1.0 / float(p.dy)) ** 2)
    residual_scale = max(divergence_l2, derivative_norm_bound * source_norm, solve_residual_l2, 1.0e-300)
    solve_tolerance = float(256.0 * eps_work * residual_scale)
    reconstruction_tolerance = float(8.0 * eps_work * max(float(np.max(np.hypot(source64_x, source64_y))), 1.0e-30))
    div_residual_max = float(np.max(np.abs(_as_np64(div_residual))))
    solve_residual_max = float(np.max(np.abs(_as_np64(solve_residual))))
    source_dot_gradient_residual = float(
        np.sum((source64_x * grad64_x + source64_y * grad64_y), dtype=np.float64) * area - (grad_norm**2)
    )
    orthogonality_dot = float(np.sum((grad64_x * residual64_x + grad64_y * residual64_y), dtype=np.float64) * area)
    checks = {
        "finite_source_and_solution": bool(
            np.isfinite(source64_x).all()
            and np.isfinite(source64_y).all()
            and np.isfinite(_as_np64(potential)).all()
            and np.isfinite(grad64_x).all()
            and np.isfinite(grad64_y).all()
            and np.isfinite(residual64_x).all()
            and np.isfinite(residual64_y).all()
        ),
        "divergence_residual_within_dtype_tolerance": bool(residual_divergence_l2 <= solve_tolerance),
        "poisson_solve_residual_within_dtype_tolerance": bool(solve_residual_l2 <= solve_tolerance),
        "reconstruction_within_dtype_tolerance": bool(reconstruction_linf <= reconstruction_tolerance),
    }
    passed = all(checks.values())
    if not passed:
        raise AuditValidationError(
            "production-operator decomposition failed closed: "
            f"checks={checks}, div_residual_l2={residual_divergence_l2:.6e}, "
            f"solve_residual_l2={solve_residual_l2:.6e}, tolerance={solve_tolerance:.6e}, "
            f"reconstruction_linf={reconstruction_linf:.6e}, reconstruction_tolerance={reconstruction_tolerance:.6e}"
        )
    return {
        "potential": _as_np64(potential),
        "source_x": source64_x,
        "source_y": source64_y,
        "gradient_x": grad64_x,
        "gradient_y": grad64_y,
        "residual_x": residual64_x,
        "residual_y": residual64_y,
        "divergence_source": _as_np64(div_source),
        "divergence_residual": _as_np64(div_residual),
        "poisson_residual": _as_np64(solve_residual),
        "metrics": {
            "source_acceleration_l2": source_norm,
            "projectable_gradient_l2": grad_norm,
            "nonprojectable_residual_l2": residual_norm,
            "projectable_fraction_of_source_l2": _safe_ratio(grad_norm, source_norm),
            "residual_fraction_of_source_l2": _safe_ratio(residual_norm, source_norm),
            "residual_over_projectable_l2": _safe_ratio(residual_norm, grad_norm),
            "source_divergence_l2": divergence_l2,
            "residual_divergence_l2": residual_divergence_l2,
            "residual_divergence_linf": div_residual_max,
            "poisson_solve_residual_l2": solve_residual_l2,
            "poisson_solve_residual_linf": solve_residual_max,
            "solve_tolerance_l2": solve_tolerance,
            "reconstruction_linf": reconstruction_linf,
            "reconstruction_tolerance_linf": reconstruction_tolerance,
            "momentum_cell_area": area,
            "momentum_grid_cell_count": int(p.Nx * p.Ny),
            "work_dtype": work_dtype.name,
            "work_dtype_epsilon": eps_work,
            "mean_free_pressure_potential": float(np.mean(_as_np64(potential), dtype=np.float64)),
            "gradient_residual_inner_product": orthogonality_dot,
            "gradient_residual_cosine": _safe_ratio(orthogonality_dot, max(grad_norm * residual_norm, 1.0e-300)),
            "source_dot_gradient_minus_gradient_norm_squared": source_dot_gradient_residual,
            "checks": checks,
            "passed": passed,
        },
    }


def _diagnostic_curl(u: Any, v: Any, p: pf.PhaseFieldParams) -> np.ndarray:
    """The named diagnostic curl; the production solver has no canonical curl operator."""
    ux, vy = jnp.asarray(u, dtype=p.dtype), jnp.asarray(v, dtype=p.dtype)
    return _as_np64(pf._ddx(vy, p.dx) - pf._ddy(ux, p.dy))


def _manufactured_operator_tests(p: pf.PhaseFieldParams) -> dict[str, Any]:
    """Validate the central periodic D/G pair on independent manufactured fields."""
    nx, ny = int(p.Nx), int(p.Ny)
    x = (np.arange(nx, dtype=np.float64) + 0.5)[:, None] * float(p.dx)
    y = (np.arange(ny, dtype=np.float64) + 0.5)[None, :] * float(p.dy)
    scalar = np.sin(2.0 * np.pi * 3.0 * x / float(p.Lx)) * np.cos(2.0 * np.pi * 2.0 * y / float(p.Ly))
    scalar += 0.25 * np.cos(2.0 * np.pi * 2.0 * x / float(p.Lx) + 2.0 * np.pi * y / float(p.Ly))
    scalar_work = jnp.asarray(scalar, dtype=p.dtype)
    grad_x = pf._ddx(scalar_work, p.dx)
    grad_y = pf._ddy(scalar_work, p.dy)
    curl_gradient = _diagnostic_curl(grad_x, grad_y, p)
    curl_l2 = float(np.sqrt(np.sum(curl_gradient**2, dtype=np.float64) * p.dx * p.dy))
    gradient_scale = float(
        np.sqrt(np.sum((_as_np64(grad_x) ** 2 + _as_np64(grad_y) ** 2), dtype=np.float64) * p.dx * p.dy)
    )
    eps_work = float(np.finfo(np.dtype(np.asarray(scalar_work).dtype)).eps)
    eps64 = float(np.finfo(np.float64).eps)
    curl_tolerance = 256.0 * eps_work * max(gradient_scale / min(float(p.dx), float(p.dy)), 1.0e-30)

    rng = np.random.default_rng(120821)
    ax = jnp.asarray(rng.standard_normal((nx, ny)), dtype=jnp.float64)
    ay = jnp.asarray(rng.standard_normal((nx, ny)), dtype=jnp.float64)
    scalar_j_double = jnp.asarray(scalar, dtype=jnp.float64)
    div = pf._ddx(ax, p.dx) + pf._ddy(ay, p.dy)
    grad = (pf._ddx(scalar_j_double, p.dx), pf._ddy(scalar_j_double, p.dy))
    area = float(p.dx * p.dy)
    lhs = float(np.sum((_as_np64(ax) * _as_np64(grad[0]) + _as_np64(ay) * _as_np64(grad[1])), dtype=np.float64) * area)
    rhs = float(-np.sum(_as_np64(div) * scalar, dtype=np.float64) * area)
    adjoint_error = abs(lhs - rhs)
    adjoint_scale = max(abs(lhs), abs(rhs), 1.0e-30)
    adjoint_relative_error = adjoint_error / adjoint_scale
    adjoint_tolerance = 512.0 * eps64
    return {
        "curl_operator": "diagnostic_curl_v1 = D_x(a_y) - D_y(a_x), using the production centred differences",
        "manufactured_scalar_description": "smooth periodic sum of two non-Nyquist Fourier modes; deterministic grid-centred sample",  # noqa: E501
        "manufactured_gradient_work_dtype": np.dtype(np.asarray(scalar_work).dtype).name,
        "manufactured_gradient_curl_l2": curl_l2,
        "manufactured_gradient_curl_linf": float(np.max(np.abs(curl_gradient))),
        "manufactured_gradient_curl_tolerance": float(curl_tolerance),
        "manufactured_gradient_curl_passed": bool(float(np.max(np.abs(curl_gradient))) <= curl_tolerance),
        "adjoint_inner_product": "<a,Gq>_(dxdy) = -<D a,q>_(dxdy) on the full uniform periodic momentum grid",
        "adjoint_lhs": lhs,
        "adjoint_rhs": rhs,
        "adjoint_absolute_error": adjoint_error,
        "adjoint_relative_error": adjoint_relative_error,
        "adjoint_tolerance_relative": adjoint_tolerance,
        "adjoint_passed": bool(adjoint_relative_error <= adjoint_tolerance),
        "metric_scope": "uniform dx*dy full-grid momentum inner product; not the cut-cell phase-volume metric",
        "not_a_production_curl": True,
    }


def _contact_positions(phi: np.ndarray, solid: pf.Solid, p: pf.PhaseFieldParams) -> dict[str, Any]:
    result = clk.contact_line_positions(
        phi,
        np.asarray(solid.sdf, dtype=np.float64),
        float(p.dx),
        float(p.dy),
        eps=float(p.eps),
        Lx=float(p.Lx),
    )
    return result


def _region_masks(
    phi: Any, solid: pf.Solid, p: pf.PhaseFieldParams, positions: dict[str, Any]
) -> dict[str, np.ndarray]:
    phase = _as_np64(phi)
    sdf = _as_np64(solid.sdf)
    nx, ny = phase.shape
    dx, dy = float(p.dx), float(p.dy)
    x = (np.arange(nx, dtype=np.float64) + 0.5)[:, None] * dx
    y = (np.arange(ny, dtype=np.float64) + 0.5)[None, :] * dy
    X = np.broadcast_to(x, phase.shape)
    Y = np.broadcast_to(y, phase.shape)
    fluid = sdf >= 0.0
    masks: dict[str, np.ndarray] = {
        "all_momentum_cells": np.ones(phase.shape, dtype=bool),
        "fluid_cell_centres": fluid,
        "interface_fluid_phi_005_095": fluid & (phase >= 0.05) & (phase <= 0.95),
        "interface_all_phi_005_095": (phase >= 0.05) & (phase <= 0.95),
        "liquid_phi_ge_095_fluid": fluid & (phase >= 0.95),
        "gas_phi_le_005_fluid": fluid & (phase <= 0.05),
        "near_wall_fluid_0_2dx": fluid & (sdf <= 2.0 * dx),
        "near_wall_fluid_0_4dx": fluid & (sdf <= 4.0 * dx),
        "solid_cell_centres": sdf < 0.0,
        "solid_neighborhood_inside_0_2dx": (sdf < 0.0) & (sdf >= -2.0 * dx),
        "solid_core_deeper_than_2dx": sdf < -2.0 * dx,
        "brinkman_chi_gt_050": _as_np64(solid.chi) > 0.5,
        "brinkman_chi_gt_010": _as_np64(solid.chi) > 0.1,
        "y_periodic_seam_2cells": (Y < 2.0 * dy) | (Y >= float(p.Ly) - 2.0 * dy),
        "y_periodic_seam_4cells": (Y < 4.0 * dy) | (Y >= float(p.Ly) - 4.0 * dy),
    }
    for lo, hi in ((0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, math.inf)):
        suffix = f"{lo:g}_{'inf' if math.isinf(hi) else f'{hi:g}'}dx"
        masks[f"wall_distance_fluid_{suffix}"] = fluid & (sdf >= lo * dx) & (sdf < hi * dx)
        masks[f"wall_distance_solid_{suffix}"] = (
            (sdf < -lo * dx) & (sdf >= -hi * dx) if math.isfinite(hi) else sdf < -lo * dx
        )
    if positions.get("contact_line_exists"):
        wall_y = float(positions.get("wall_y", chns.WALL_HEIGHT))
        # The estimator exposes periodic x positions and uses the known fixed wall plane; preserve
        # its exact locations and use Euclidean radius in x/y, periodic only in x.
        x_values = {
            "left": positions.get("left_contact_x_wrapped"),
            "right": positions.get("right_contact_x_wrapped"),
        }
        periodic_dx = lambda xc: np.abs((X - float(xc) + 0.5 * float(p.Lx)) % float(p.Lx) - 0.5 * float(p.Lx))
        for side, xc in x_values.items():
            if xc is None:
                continue
            distance = np.hypot(periodic_dx(float(xc)), Y - wall_y)
            for cells in (2.0, 4.0):
                key = f"contact_line_{side}_within_{int(cells)}dx"
                masks[key] = distance <= cells * dx
        for cells in (2.0, 4.0):
            union = np.zeros(phase.shape, dtype=bool)
            for side in ("left", "right"):
                key = f"contact_line_{side}_within_{int(cells)}dx"
                if key in masks:
                    union |= masks[key]
            masks[f"contact_line_union_within_{int(cells)}dx"] = union
            masks[f"outside_contact_line_union_{int(cells)}dx"] = ~union
    return masks


def _regional_metrics(
    residual_x: Any,
    residual_y: Any,
    source_x: Any,
    source_y: Any,
    curl_value: Any,
    masks: dict[str, np.ndarray],
    p: pf.PhaseFieldParams,
) -> dict[str, Any]:
    area = float(p.dx * p.dy)
    residual_global = _vector_metrics(residual_x, residual_y, masks["all_momentum_cells"], area)["l2"]
    residual_energy = residual_global**2
    rows: dict[str, Any] = {}
    for name, mask in masks.items():
        residual = _vector_metrics(residual_x, residual_y, mask, area)
        source = _vector_metrics(source_x, source_y, mask, area)
        rows[name] = {
            "mask_definition": name,
            "n_cells": residual["n_cells"],
            "area": residual["area"],
            "residual_l2": residual["l2"],
            "residual_linf": residual["linf"],
            "source_acceleration_l2": source["l2"],
            "residual_to_source_l2_ratio": _safe_ratio(residual["l2"], source["l2"]),
            "residual_energy_fraction_of_global": _safe_ratio(residual["l2"] ** 2, residual_energy),
            "diagnostic_curl_v1_l2": _scalar_metrics(curl_value, mask, area)["l2"],
            "diagnostic_curl_v1_linf": _scalar_metrics(curl_value, mask, area)["linf"],
        }
    return rows


def _capillary_components(state: pf.State, solid: pf.Solid, p: pf.PhaseFieldParams) -> dict[str, Any]:
    """Extract the actual contract-11 force addend and production-RHS cast increment."""
    phi = state.phi
    phi_rhs, rhs_u, rhs_v, mu, _mu_expl = pf.rhs(state, solid, p)
    del phi_rhs
    phi_x, phi_y = pf._ddx(phi, p.dx), pf._ddy(phi, p.dy)
    coefficient = float(pf.SIGMA_NORM / p.We)
    force_density_x = coefficient * mu * phi_x
    force_density_y = coefficient * mu * phi_y
    acceleration_x = force_density_x / p.rho_l
    acceleration_y = force_density_y / p.rho_l

    adv_u = pf.div_upwind(state.u, state.v, state.u, p.dx, p.dy)
    adv_v = pf.div_upwind(state.u, state.v, state.v, p.dx, p.dy)
    nu = pf.nu_of(phi, p)
    lap_u, lap_v = pf._lap(state.u, p.dx, p.dy), pf._lap(state.v, p.dx, p.dy)
    if p.use_gravity:
        rho = pf.rho_of(phi, p)
        gravity_v = -(1.0 / p.Fr**2) * (rho - jnp.mean(rho)) / rho
    else:
        gravity_v = jnp.zeros_like(phi)
    base_u = -adv_u + nu * lap_u
    base_v = -adv_v + nu * lap_v + gravity_v
    reconstructed_rhs_u = (base_u + acceleration_x).astype(p.dtype)
    reconstructed_rhs_v = (base_v + acceleration_y).astype(p.dtype)
    if not np.array_equal(np.asarray(rhs_u), np.asarray(reconstructed_rhs_u)) or not np.array_equal(
        np.asarray(rhs_v), np.asarray(reconstructed_rhs_v)
    ):
        raise AuditValidationError("capillary addend reconstruction disagrees with pf.rhs; refusing analysis")
    effective_x = rhs_u - base_u.astype(p.dtype)
    effective_y = rhs_v - base_v.astype(p.dtype)

    mu_explicit = pf._explicit_chemical_potential(phi, solid, p)
    mu_wall = mu_explicit - pf.fprime(phi) / p.eps
    mu_total = mu
    mu_bulk = mu_total - mu_wall
    wall_acc_x = coefficient * mu_wall * phi_x / p.rho_l
    wall_acc_y = coefficient * mu_wall * phi_y / p.rho_l
    bulk_acc_x = coefficient * mu_bulk * phi_x / p.rho_l
    bulk_acc_y = coefficient * mu_bulk * phi_y / p.rho_l

    mu_x, mu_y = pf._ddx(mu_total, p.dx), pf._ddy(mu_total, p.dy)
    product = mu_total * phi
    product_x, product_y = pf._ddx(product, p.dx), pf._ddy(product, p.dy)
    # E_h = G_h(mu phi) - mu G_h(phi) - phi G_h(mu).  The exact discrete identity is
    # mu G_h(phi) = G_h(mu phi) - phi G_h(mu) - E_h.
    defect_x = product_x - mu_total * phi_x - phi * mu_x
    defect_y = product_y - mu_total * phi_y - phi * mu_y
    alt_b_x = -coefficient * phi * mu_x / p.rho_l
    alt_b_y = -coefficient * phi * mu_y / p.rho_l
    product_gradient_x = coefficient * product_x / p.rho_l
    product_gradient_y = coefficient * product_y / p.rho_l
    defect_acc_x = coefficient * defect_x / p.rho_l
    defect_acc_y = coefficient * defect_y / p.rho_l
    identity_lhs_x = acceleration_x
    identity_lhs_y = acceleration_y
    identity_rhs_x = product_gradient_x + alt_b_x - defect_acc_x
    identity_rhs_y = product_gradient_y + alt_b_y - defect_acc_y
    identity_error = max(
        float(np.max(np.abs(_as_np64(identity_lhs_x) - _as_np64(identity_rhs_x)))),
        float(np.max(np.abs(_as_np64(identity_lhs_y) - _as_np64(identity_rhs_y)))),
    )
    return {
        "mu": _as_np64(mu_total),
        "mu_bulk": _as_np64(mu_bulk),
        "mu_wall": _as_np64(mu_wall),
        "phi_gradient_x": _as_np64(phi_x),
        "phi_gradient_y": _as_np64(phi_y),
        "force_density_x": _as_np64(force_density_x),
        "force_density_y": _as_np64(force_density_y),
        "raw_acceleration_x": _as_np64(acceleration_x),
        "raw_acceleration_y": _as_np64(acceleration_y),
        "working_acceleration_x": _as_np64(acceleration_x.astype(p.dtype)),
        "working_acceleration_y": _as_np64(acceleration_y.astype(p.dtype)),
        "effective_rhs_increment_x": _as_np64(effective_x),
        "effective_rhs_increment_y": _as_np64(effective_y),
        "wall_acceleration_x": _as_np64(wall_acc_x),
        "wall_acceleration_y": _as_np64(wall_acc_y),
        "bulk_acceleration_x": _as_np64(bulk_acc_x),
        "bulk_acceleration_y": _as_np64(bulk_acc_y),
        "alternative_minus_phi_grad_mu_x": _as_np64(alt_b_x),
        "alternative_minus_phi_grad_mu_y": _as_np64(alt_b_y),
        "product_gradient_x": _as_np64(product_gradient_x),
        "product_gradient_y": _as_np64(product_gradient_y),
        "product_rule_defect_acceleration_x": _as_np64(defect_acc_x),
        "product_rule_defect_acceleration_y": _as_np64(defect_acc_y),
        "identity_reconstruction_linf": identity_error,
        "production_rhs_reconstruction_exact": True,
        "coefficient": coefficient,
        "rho_l": float(p.rho_l),
    }


def _force_decomposition(components: dict[str, Any], p: pf.PhaseFieldParams) -> dict[str, Any]:
    area = float(p.dx * p.dy)
    all_cells = np.ones((p.Nx, p.Ny), dtype=bool)
    parts = {
        "force_density": (components["force_density_x"], components["force_density_y"]),
        "raw_acceleration_expression": (components["raw_acceleration_x"], components["raw_acceleration_y"]),
        "production_work_dtype_acceleration": (
            components["working_acceleration_x"],
            components["working_acceleration_y"],
        ),
        "effective_increment_after_rhs_cast": (
            components["effective_rhs_increment_x"],
            components["effective_rhs_increment_y"],
        ),
        "wall_energy_mu_contribution": (components["wall_acceleration_x"], components["wall_acceleration_y"]),
        "bulk_mu_contribution": (components["bulk_acceleration_x"], components["bulk_acceleration_y"]),
        "diagnostic_form_minus_phi_grad_mu": (
            components["alternative_minus_phi_grad_mu_x"],
            components["alternative_minus_phi_grad_mu_y"],
        ),
        "discrete_product_rule_defect": (
            components["product_rule_defect_acceleration_x"],
            components["product_rule_defect_acceleration_y"],
        ),
    }
    result: dict[str, Any] = {}
    arrays: dict[str, dict[str, Any]] = {}
    for name, (ux, vy) in parts.items():
        if name == "force_density":
            # Convert force density to the acceleration scale only for comparison with the
            # pressure projection; the raw force-density norm remains reported independently.
            projection_source_x = (_as_np64(ux) / float(p.rho_l)).astype(p.dtype)
            projection_source_y = (_as_np64(vy) / float(p.rho_l)).astype(p.dtype)
        elif name in ("raw_acceleration_expression",):
            projection_source_x = _as_np64(ux).astype(p.dtype)
            projection_source_y = _as_np64(vy).astype(p.dtype)
        else:
            projection_source_x = _as_np64(ux).astype(p.dtype)
            projection_source_y = _as_np64(vy).astype(p.dtype)
        dec = _production_operator_arrays(projection_source_x, projection_source_y, p)
        g = _vector_metrics(ux, vy, all_cells, area)
        a = _vector_metrics(projection_source_x, projection_source_y, all_cells, area)
        result[name] = {
            "field_l2_in_reported_units": g["l2"],
            "field_linf_in_reported_units": g["linf"],
            "projected_acceleration_l2": a["l2"],
            **dec["metrics"],
        }
        arrays[name] = dec
    # Direct density-scaling audit: rho_l is a constant, so its reciprocal rescales norms but
    # cannot create spatial curl or alter the residual/source ratio, apart from roundoff.
    f_metrics = result["force_density"]
    a_metrics = result["raw_acceleration_expression"]
    ratio_difference = abs(
        float(f_metrics["residual_fraction_of_source_l2"] or 0.0)
        - float(a_metrics["residual_fraction_of_source_l2"] or 0.0)
    )
    result["force_density_vs_acceleration"] = {
        "rho_l_constant": float(p.rho_l),
        "force_density_l2": f_metrics["field_l2_in_reported_units"],
        "raw_acceleration_l2": a_metrics["field_l2_in_reported_units"],
        "expected_force_density_over_acceleration_l2": float(p.rho_l),
        "residual_fraction_difference_force_vs_acceleration": ratio_difference,
        "curl_force_density_l2": float(
            np.sqrt(
                np.sum(
                    _diagnostic_curl(components["force_density_x"], components["force_density_y"], p) ** 2,
                    dtype=np.float64,
                )
                * area
            )
        ),
        "curl_raw_acceleration_l2": float(
            np.sqrt(
                np.sum(
                    _diagnostic_curl(components["raw_acceleration_x"], components["raw_acceleration_y"], p) ** 2,
                    dtype=np.float64,
                )
                * area
            )
        ),
        "interpretation": "constant scalar division by rho_l changes magnitude only; any spatial curl/non-projectability comes from the force field/operator, not constant scaling alone",  # noqa: E501
    }
    # The pressure solve is linear. Verify the discrete identity between the production force's
    # projected residual, the Form-B diagnostic residual, and the projected product-rule defect.
    base = arrays["production_work_dtype_acceleration"]
    alternative = arrays["diagnostic_form_minus_phi_grad_mu"]
    defect = arrays["discrete_product_rule_defect"]
    base_r_x, base_r_y = base["residual_x"], base["residual_y"]
    alt_r_x, alt_r_y = alternative["residual_x"], alternative["residual_y"]
    defect_r_x, defect_r_y = defect["residual_x"], defect["residual_y"]
    identity_residual_x = base_r_x - (alt_r_x - defect_r_x)
    identity_residual_y = base_r_y - (alt_r_y - defect_r_y)
    identity_residual_l2 = _vector_metrics(identity_residual_x, identity_residual_y, all_cells, area)["l2"]
    result["discrete_form_identity"] = {
        "production_force_form": "+(SIGMA_NORM/We) * mu_h * G_h(phi) / rho_l",
        "diagnostic_alternative_not_used_in_production": "-(SIGMA_NORM/We) * phi * G_h(mu_h) / rho_l",
        "product_rule_defect_definition": "E_h = G_h(mu_h*phi) - mu_h*G_h(phi) - phi*G_h(mu_h)",
        "identity": "mu_h G_h(phi) = G_h(mu_h phi) - phi G_h(mu_h) - E_h",
        "production_residual_equals_alternative_minus_defect_residual_l2_error": identity_residual_l2,
        "production_residual_l2": result["production_work_dtype_acceleration"]["nonprojectable_residual_l2"],
        "alternative_residual_l2": result["diagnostic_form_minus_phi_grad_mu"]["nonprojectable_residual_l2"],
        "product_rule_defect_residual_l2": result["discrete_product_rule_defect"]["nonprojectable_residual_l2"],
        "alternative_form_diagnostic_only": True,
        "not_a_force_ablation_or_production_validation": True,
        "mu_product_identity_linf_before_pressure_projection": float(components["identity_reconstruction_linf"]),
    }
    result["_arrays"] = arrays
    return result


def _case_spatial_summary(
    state: pf.State, solid: pf.Solid, p: pf.PhaseFieldParams, components: dict[str, Any], force: dict[str, Any]
) -> dict[str, Any]:
    positions = _contact_positions(np.asarray(state.phi), solid, p)
    masks = _region_masks(state.phi, solid, p, positions)
    base = force["_arrays"]["production_work_dtype_acceleration"]
    curl_residual = _diagnostic_curl(base["residual_x"], base["residual_y"], p)
    curl_source = _diagnostic_curl(base["source_x"], base["source_y"], p)
    area = float(p.dx * p.dy)
    global_curl = _scalar_metrics(curl_source, masks["all_momentum_cells"], area)
    global_residual_curl = _scalar_metrics(curl_residual, masks["all_momentum_cells"], area)
    regional = _regional_metrics(
        base["residual_x"],
        base["residual_y"],
        base["source_x"],
        base["source_y"],
        curl_source,
        masks,
        p,
    )
    curl_regions = {name: _scalar_metrics(curl_source, mask, area) for name, mask in masks.items()}
    residual_energy = (
        _vector_metrics(base["residual_x"], base["residual_y"], masks["all_momentum_cells"], area)["l2"] ** 2
    )
    chi = _as_np64(solid.chi)
    residual_x, residual_y = base["residual_x"], base["residual_y"]
    chi_overlap = {}
    for name, mask in (
        ("chi_gt_001", chi > 0.01),
        ("chi_gt_010", chi > 0.1),
        ("chi_gt_050", chi > 0.5),
        ("chi_gt_090", chi > 0.9),
    ):
        value = _vector_metrics(residual_x, residual_y, mask, area)
        chi_overlap[name] = {
            **value,
            "residual_energy_fraction": _safe_ratio(value["l2"] ** 2, residual_energy),
        }
    wall_dec = force["_arrays"]["wall_energy_mu_contribution"]
    wall_residuals = _regional_metrics(
        wall_dec["residual_x"],
        wall_dec["residual_y"],
        wall_dec["source_x"],
        wall_dec["source_y"],
        _diagnostic_curl(wall_dec["source_x"], wall_dec["source_y"], p),
        masks,
        p,
    )
    return {
        "contact_line_estimator": positions,
        "mask_counts": {name: int(np.count_nonzero(mask)) for name, mask in masks.items()},
        "regional_residuals_and_curl": regional,
        "wall_energy_component_localization": wall_residuals,
        "curl_localization": {
            "curl_label": "diagnostic_curl_v1",
            "production_source_curl_global": global_curl,
            "production_residual_curl_global": global_residual_curl,
            "regional": curl_regions,
            "no_canonical_production_curl": True,
        },
        "brinkman_chi_overlap": chi_overlap,
        "contact_line_exclusion_radii_dx": [2, 4],
        "residual_energy_fraction_masks_overlap": True,
    }


def _projected_one_step_impulse(state: pf.State, solid: pf.Solid, p: pf.PhaseFieldParams) -> dict[str, Any]:
    """Freeze phi and apply one full (three internal-substep) capillary-only momentum step from rest."""
    start = pf.State(
        phi=state.phi,
        u=jnp.zeros_like(state.u),
        v=jnp.zeros_like(state.v),
        t=state.t,
    )
    on = chns.forensic_step_jit(start, solid, p, freeze_phi=True, capillary_scale=1.0)
    off = chns.forensic_step_jit(start, solid, p, freeze_phi=True, capillary_scale=0.0)
    end = on.state
    if not np.array_equal(np.asarray(end.phi), np.asarray(start.phi)):
        raise AuditValidationError("frozen-phi projected impulse changed phi; refusing result")
    if not np.array_equal(np.asarray(off.state.phi), np.asarray(start.phi)):
        raise AuditValidationError("frozen-phi capillary-off impulse changed phi; refusing result")
    area = float(p.dx * p.dy)
    u, v = _as_np64(end.u), _as_np64(end.v)
    off_u, off_v = _as_np64(off.state.u), _as_np64(off.state.v)
    impulse_l2 = _vector_metrics(u, v, np.ones(u.shape, dtype=bool), area)["l2"]
    off_l2 = _vector_metrics(off_u, off_v, np.ones(u.shape, dtype=bool), area)["l2"]
    return {
        "label": "projected_one_step_capillary_impulse_from_rest_frozen_phi",
        "steps": 1,
        "internal_substeps": 3,
        "dt": float(p.dt),
        "initial_velocity": "exact zero",
        "phi_bitwise_fixed": True,
        "capillary_on_final_velocity_l2": impulse_l2,
        "capillary_off_final_velocity_l2": off_l2,
        "capillary_on_minus_off_velocity_l2": _vector_metrics(u - off_u, v - off_v, np.ones(u.shape, dtype=bool), area)[
            "l2"
        ],
        "capillary_on_final_max_speed": float(np.max(np.hypot(u, v))),
        "capillary_off_final_max_speed": float(np.max(np.hypot(off_u, off_v))),
        "uses_production_brinkman_and_projection_each_substep": True,
        "diagnostic_only": True,
        "production_acceptance_evidence": False,
    }


def _frozen_phi_forcing_ab(
    state: pf.State, solid: pf.Solid, p: pf.PhaseFieldParams, *, steps: int = FROZEN_PHI_STEPS
) -> dict[str, Any]:
    """Matched frozen-phi capillary-on/off branch using the L1A-2j exact diagnostic step."""
    if steps <= 0:
        raise ValueError("frozen-phi horizon must be positive")
    rows = {}
    for label, scale in (("capillary_on", 1.0), ("capillary_off", 0.0)):
        final, _states, _fields, observations, _last = chns.advance_forensic_block(
            state,
            solid,
            p,
            True,
            scale,
            int(steps),
            1,
        )
        if not np.array_equal(np.asarray(final.phi), np.asarray(state.phi)):
            raise AuditValidationError(f"{label} frozen-phi branch changed phi")
        area = float(p.dx * p.dy)
        u, v = _as_np64(final.u), _as_np64(final.v)
        speed = np.hypot(u, v)
        rho = _as_np64(pf.rho_of(final.phi, p))
        kinetic = float(0.5 * np.sum(rho * (u**2 + v**2), dtype=np.float64) * area)
        rows[label] = {
            "steps": int(steps),
            "phi_bitwise_fixed": True,
            "final_velocity_l2": _vector_metrics(u, v, np.ones(u.shape, dtype=bool), area)["l2"],
            "final_max_speed": float(np.max(speed)),
            "final_kinetic_energy_rho_phi_weighted_diagnostic": kinetic,
            "step_metric_final": _as_np64(observations[0])[-1, -1, :].tolist(),
            "state_hashes": _state_hashes(final),
        }
    rows["matched_difference"] = {
        "same_start_state": True,
        "same_steps": int(steps),
        "same_phi_geometry_dt_M_wetting_projection_and_Brinkman": True,
        "only_capillary_scale_changed": True,
        "final_velocity_l2_on_over_off": _safe_ratio(
            rows["capillary_on"]["final_velocity_l2"], rows["capillary_off"]["final_velocity_l2"]
        ),
        "diagnostic_only": True,
        "production_acceptance_evidence": False,
    }
    return rows


def _equal_density_one_step_ab(state: pf.State, solid: pf.Solid, p: pf.PhaseFieldParams) -> dict[str, Any]:
    """Evidence-triggered A/B changing only rho_g/rho_l blend, on the identical saved state."""
    equal_params = dataclasses.replace(p, rho_l=float(p.rho_l), rho_g=float(p.rho_l))
    prod_state, _ = pf.step_with_diagnostics(state, solid, p)
    equal_state, _ = pf.step_with_diagnostics(state, solid, equal_params)
    area = float(p.dx * p.dy)
    diffs = {}
    for field in ("phi", "u", "v"):
        diff = _as_np64(getattr(equal_state, field)) - _as_np64(getattr(prod_state, field))
        diffs[field] = {
            "linf": float(np.max(np.abs(diff))),
            "l2": float(np.sqrt(np.sum(diff**2, dtype=np.float64) * area)),
            "bitwise_equal": bool(
                np.array_equal(np.asarray(getattr(equal_state, field)), np.asarray(getattr(prod_state, field)))
            ),
        }
    original_mu = pf.chemical_potential(state.phi, solid, p)
    equal_mu = pf.chemical_potential(state.phi, solid, equal_params)
    original_force = (pf.SIGMA_NORM / p.We) * original_mu * pf._ddx(state.phi, p.dx) / p.rho_l
    equal_force = (pf.SIGMA_NORM / equal_params.We) * equal_mu * pf._ddx(state.phi, p.dx) / equal_params.rho_l
    force_difference = float(np.max(np.abs(_as_np64(equal_force) - _as_np64(original_force))))
    pressure_symbol_difference = float(np.max(np.abs(_as_np64(equal_params.m2_proj) - _as_np64(p.m2_proj))))
    return {
        "status": "measured",
        "evidence_trigger": "P-CAP-RHO/variable-density concern is open and the exact production force/projection balance is the subject of this audit; this minimal same-state A/B isolates only the rho(phi) blend",  # noqa: E501
        "changed_parameters": {
            "rho_g": [float(p.rho_g), float(equal_params.rho_g)],
            "rho_l": [float(p.rho_l), float(equal_params.rho_l)],
        },
        "unchanged": [
            "phi state",
            "geometry",
            "wetting",
            "dt",
            "M",
            "We",
            "force discretization",
            "capillary denominator rho_l",
            "pressure projection symbol",
            "Brinkman",
            "viscosity",
            "phase boundary and transport",
        ],
        "same_start_state_hashes": _state_hashes(state),
        "state_differences_one_production_step": diffs,
        "capillary_acceleration_max_abs_difference": force_difference,
        "pressure_symbol_max_abs_difference": pressure_symbol_difference,
        "gravity_disabled": not bool(p.use_gravity),
        "diagnostic_only": True,
        "production_acceptance_evidence": False,
        "interpretation": "The equal-density A/B is a one-step same-state isolation only. It does not validate a variable-density inertia or pressure model.",  # noqa: E501
    }


def _save_state_snapshot(
    path: Path,
    state: pf.State,
    *,
    case_name: str,
    step: int,
    config: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "stage": STAGE,
        "case": case_name,
        "step": int(step),
        "git_sha": chns.get_git_sha(),
        "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
        "config": config,
        "config_fingerprint": chns._canonical_hash(config),
        "source_hashes": _source_hashes(),
        "state_hashes": _state_hashes(state),
        "origin_metadata": metadata,
        "diagnostic_only": True,
        "production_semantics_changed": False,
    }
    _write_npz(
        path,
        phi=np.asarray(state.phi),
        u=np.asarray(state.u),
        v=np.asarray(state.v),
        t=np.asarray(state.t),
        metadata_json=np.asarray(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)),
    )
    return record


def _load_state_snapshot(
    path: Path,
    *,
    case_name: str,
    step: int,
    config: dict[str, Any],
) -> tuple[pf.State, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        expected = {"phi", "u", "v", "t", "metadata_json"}
        if expected - set(archive.files):
            raise ValueError("L1A-2k snapshot fields are incomplete")
        arrays = {key: np.array(archive[key], copy=True) for key in ("phi", "u", "v", "t")}
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
    checks = {
        "stage": metadata.get("stage") == STAGE,
        "case": metadata.get("case") == case_name,
        "step": metadata.get("step") == int(step),
        "solver_contract": metadata.get("solver_contract_version")
        == SOLVER_CONTRACT
        == int(pf.SOLVER_CONTRACT_VERSION),
        "git_sha": metadata.get("git_sha") == chns.get_git_sha(),
        "config": metadata.get("config") == config,
        "config_fingerprint": metadata.get("config_fingerprint") == chns._canonical_hash(config),
        "source_hashes": metadata.get("source_hashes") == _source_hashes(),
        "state_hashes": metadata.get("state_hashes") == {key: chns._array_hash(value) for key, value in arrays.items()},
        "production_semantics_unchanged": metadata.get("production_semantics_changed") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"L1A-2k snapshot rejected; strict validation failed: {checks}")
    if arrays["phi"].shape != (int(config["Nx"]), int(config["Ny"])):
        raise ValueError("L1A-2k snapshot grid shape mismatch")
    expected_dtypes = {"phi": "float64", "u": "float32", "v": "float32", "t": "float32"}
    if {key: value.dtype.name for key, value in arrays.items()} != expected_dtypes:
        raise ValueError("L1A-2k snapshot dtype mismatch")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("L1A-2k snapshot contains non-finite state values")
    state = pf.State(
        phi=jnp.asarray(arrays["phi"], dtype=jnp.float64),
        u=jnp.asarray(arrays["u"], dtype=jnp.float32),
        v=jnp.asarray(arrays["v"], dtype=jnp.float32),
        t=jnp.asarray(arrays["t"].item(), dtype=jnp.float32),
    )
    return state, metadata


def _save_progress(path: Path, report: dict[str, Any]) -> None:
    _write_json(path, report)


def _load_l1a2j_report(root: Path) -> dict[str, Any] | None:
    path = root / "evidence" / "l1a2j" / "chns_nonstationarity_report.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _initial_l1a2j_reuse_audit(root: Path) -> dict[str, Any]:
    report = _load_l1a2j_report(root)
    artifact_dir = root / "artifacts" / "l1a2j"
    current_chns_hashes = chns._source_hashes()
    if report is None:
        return {
            "report_available": False,
            "raw_artifact_directory": str(artifact_dir),
            "raw_artifact_directory_exists": artifact_dir.exists(),
            "decision": "no prior report metadata; no state is reused without strict checkpoint validation",
            "reusable_state": False,
        }
    posthoc = report.get("posthoc_report_revision", {})
    latest_hashes = posthoc.get("posthoc_classifier_source_hashes", report.get("source_hashes", {}))
    raw_hashes = report.get("source_hashes", {})
    return {
        "report_available": True,
        "report_contract": report.get("solver_contract_version"),
        "report_git_sha": report.get("repository_git_sha"),
        "report_authority_config_fingerprint": report.get("authority", {}).get("config_fingerprint"),
        "report_authority_step": report.get("authority", {}).get("end_step", 50_000),
        "report_authority_state_hashes_present_but_arrays_absent": bool(
            report.get("authority", {}).get("endpoint_state_hashes")
        ),
        "raw_artifact_directory": str(artifact_dir),
        "raw_artifact_directory_exists": artifact_dir.exists(),
        "report_original_run_source_hashes": raw_hashes,
        "report_posthoc_source_hashes": latest_hashes,
        "current_l1a2j_source_hashes": current_chns_hashes,
        "posthoc_source_hashes_match_current": latest_hashes == current_chns_hashes,
        "original_run_source_hashes_match_current": raw_hashes == current_chns_hashes,
        "decision": (
            "report metadata alone is insufficient to reconstruct a state; raw arrays are absent"
            if not artifact_dir.exists()
            else "candidate artifacts will be accepted only by exact checkpoint loader checks (SHA, contract, source hashes, config, state hashes, step)"  # noqa: E501
        ),
        "reusable_state": False,
    }


def _ensure_authority_state(
    root: Path,
    artifact_dir: Path,
    snapshot_path: Path,
    config: dict[str, Any],
    *,
    progress: dict[str, Any],
) -> tuple[pf.State, dict[str, Any]]:
    step = CASE_REQUIRED_STEPS["authority_060"]
    try:
        state, metadata = _load_state_snapshot(snapshot_path, case_name="authority_060", step=step, config=config)
        progress["state_reuse_decisions"].append(
            {
                "case": "authority_060",
                "snapshot": str(snapshot_path),
                "status": "reused_exact_l1a2k_snapshot",
                "step": step,
                "state_hashes": metadata["state_hashes"],
            }
        )
        return state, {"decision": "reused_exact_l1a2k_snapshot", "metadata": metadata}
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        progress["state_reuse_decisions"].append(
            {
                "case": "authority_060",
                "snapshot": str(snapshot_path),
                "status": "rejected_or_missing",
                "reason": str(exc),
            }
        )

    p, solid, seed, expected_config = chns._make_case(60.0, N_value=chns.N, dt=chns.DT, M=chns.M_REF)
    if expected_config != config:
        raise AuditValidationError("authority config differs from the frozen L1A-2j config")
    if pf.SOLVER_CONTRACT_VERSION != SOLVER_CONTRACT or not jax.config.x64_enabled:
        raise AuditValidationError("contract-11/x64 is required for the authority state")

    # Look for a raw L1A-2j 50k state, then an exact 40k restart.  The existing loader is strict:
    # it rejects changed Git SHA, contract, runner/solver hashes, config, state bytes, dtype or step.
    prior_root = root / "artifacts" / "l1a2j"
    direct_candidates = [
        prior_root / "checkpoints" / "authority_060_step_050000.npz",
        prior_root / "checkpoints" / "authority_060_step_040000.npz",
        root / "artifacts" / "contract11_followup" / "chns" / "theta_060_step050000.npz",
        root / "evidence" / "l1a2i" / "theta_060_step050000.npz",
    ]
    selected: tuple[pf.State, dict[str, Any], int] | None = None
    for candidate in direct_candidates:
        if not candidate.exists():
            progress["state_reuse_decisions"].append(
                {"case": "authority_060", "candidate": str(candidate), "status": "not_found"}
            )
            continue
        expected_step = 50_000 if "050000" in candidate.name else 40_000
        expected_kind = (
            "authority_50k" if expected_step == 50_000 and candidate.parent.name == "checkpoints" else "authority_40k"
        )
        try:
            candidate_state, candidate_meta = chns.load_forensic_checkpoint(
                candidate,
                expected_config=config,
                expected_step=expected_step,
                expected_kind=expected_kind,
            )
            progress["state_reuse_decisions"].append(
                {
                    "case": "authority_060",
                    "candidate": str(candidate),
                    "status": "accepted",
                    "step": expected_step,
                    "state_hashes": candidate_meta["state_hashes"],
                }
            )
            selected = candidate_state, candidate_meta, expected_step
            if expected_step == 50_000:
                state_meta = _save_state_snapshot(
                    snapshot_path,
                    candidate_state,
                    case_name="authority_060",
                    step=50_000,
                    config=config,
                    metadata={"source_checkpoint": str(candidate), "source_checkpoint_metadata": candidate_meta},
                )
                return candidate_state, {"decision": "reused_exact_50k_checkpoint", "metadata": state_meta}
            break
        except (OSError, ValueError, KeyError, chns.CheckpointError, json.JSONDecodeError) as exc:
            progress["state_reuse_decisions"].append(
                {"case": "authority_060", "candidate": str(candidate), "status": "rejected", "reason": str(exc)}
            )

    cache_path = artifact_dir / "trajectory_cache" / "authority_060_resume_latest.npz"
    resume: tuple[pf.State, int, dict[str, Any]] | None = None
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as archive:
                arrays = {key: np.array(archive[key], copy=True) for key in ("phi", "u", "v", "t")}
                cache_meta = json.loads(str(np.asarray(archive["metadata_json"]).item()))
            expected_cache = {
                "stage": STAGE,
                "case": "authority_060",
                "step": int(cache_meta.get("step", -1)),
                "git_sha": chns.get_git_sha(),
                "solver_contract_version": SOLVER_CONTRACT,
                "config": config,
                "config_fingerprint": chns._canonical_hash(config),
                "source_hashes": _source_hashes(),
                "state_hashes": {key: chns._array_hash(value) for key, value in arrays.items()},
            }
            valid_cache = all(cache_meta.get(key) == value for key, value in expected_cache.items())
            cache_step = int(cache_meta.get("step", -1))
            if not valid_cache or not (0 <= cache_step <= step) or cache_step % 1_000:
                raise ValueError(
                    f"strict resumable authority checkpoint mismatch; checks={valid_cache}, step={cache_step}"
                )
            resume_state = pf.State(
                jnp.asarray(arrays["phi"], dtype=jnp.float64),
                jnp.asarray(arrays["u"], dtype=jnp.float32),
                jnp.asarray(arrays["v"], dtype=jnp.float32),
                jnp.asarray(arrays["t"].item(), dtype=jnp.float32),
            )
            resume = resume_state, cache_step, cache_meta
            progress["state_reuse_decisions"].append(
                {
                    "case": "authority_060",
                    "candidate": str(cache_path),
                    "status": "accepted",
                    "step": cache_step,
                    "state_hashes": cache_meta["state_hashes"],
                }
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            progress["state_reuse_decisions"].append(
                {"case": "authority_060", "candidate": str(cache_path), "status": "rejected", "reason": str(exc)}
            )

    if selected is not None and (resume is None or selected[2] > resume[1]):
        current, current_step, parent = selected
    elif resume is not None:
        current, current_step, parent = resume
    else:
        current, current_step, parent = seed, 0, None

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_start_step = current_step
    started = time.perf_counter()
    while current_step < step:
        count = min(1_000, step - current_step)
        current = chns._advance_standard(current, solid, p, count)
        current.t.block_until_ready()
        current_step += count
        if current_step % 5_000 == 0 or current_step == step:
            meta = {
                "stage": STAGE,
                "case": "authority_060",
                "step": current_step,
                "git_sha": chns.get_git_sha(),
                "solver_contract_version": SOLVER_CONTRACT,
                "config": config,
                "config_fingerprint": chns._canonical_hash(config),
                "source_hashes": _source_hashes(),
                "state_hashes": _state_hashes(current),
                "parent_state_hashes": None if parent is None else parent.get("state_hashes"),
                "production_semantics_changed": False,
            }
            _write_npz(
                cache_path,
                phi=np.asarray(current.phi),
                u=np.asarray(current.u),
                v=np.asarray(current.v),
                t=np.asarray(current.t),
                metadata_json=np.asarray(json.dumps(meta, sort_keys=True, separators=(",", ":"), allow_nan=False)),
            )
            progress["case_progress"]["authority_060"] = {
                "status": "running",
                "step": current_step,
                "required_step": step,
                "elapsed_seconds": float(time.perf_counter() - started),
                "checkpoint": str(cache_path),
                "state_hashes": meta["state_hashes"],
            }
            _save_progress(artifact_dir / "progress.json", progress)
            print(f"[l1a2k] authority_060 production trajectory {current_step}/{step}", flush=True)
    state_meta = _save_state_snapshot(
        snapshot_path,
        current,
        case_name="authority_060",
        step=step,
        config=config,
        metadata={
            "origin": "production pf.step_with_diagnostics via chns._advance_standard",
            "resumed_from_step": trajectory_start_step,
        },
    )
    return current, {"decision": "reran_or_resumed_exact_production_to_50k", "metadata": state_meta}


def _ensure_control_state(
    root: Path,
    artifact_dir: Path,
    snapshot_path: Path,
    config: dict[str, Any],
    target: float,
    *,
    progress: dict[str, Any],
) -> tuple[pf.State, dict[str, Any], dict[str, Any]]:
    case_name = "control_090" if target == 90.0 else "control_150"
    expected_step = CASE_REQUIRED_STEPS[case_name]
    try:
        state, metadata = _load_state_snapshot(snapshot_path, case_name=case_name, step=expected_step, config=config)
        origin = metadata.get("origin_metadata", {})
        if origin.get("control_converged") is not True or int(origin.get("control_run_steps", -1)) != expected_step:
            raise ValueError("cached control snapshot lacks a validated converged-control lineage")
        control_summary = {
            "converged": True,
            "run_record": {"steps": expected_step, "stop_reason": "validated_cached_converged_control"},
            "recorded_production_gate": "cached control checkpoint lineage validated",
        }
        progress["state_reuse_decisions"].append(
            {
                "case": case_name,
                "snapshot": str(snapshot_path),
                "status": "reused_exact_l1a2k_snapshot",
                "step": expected_step,
                "state_hashes": metadata["state_hashes"],
            }
        )
        return state, metadata, control_summary
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        progress["state_reuse_decisions"].append(
            {"case": case_name, "snapshot": str(snapshot_path), "status": "rejected_or_missing", "reason": str(exc)}
        )

    prior_root = root / "artifacts" / "l1a2j"
    control_dir = prior_root / "controls" / f"theta_{int(target):03d}"
    p, solid, _seed, expected_config = chns._make_case(target, N_value=chns.N, dt=chns.DT, M=chns.M_REF)
    if expected_config != config:
        raise AuditValidationError(f"{case_name} config differs from frozen L1A-2j config")
    # The L1A-2j loader checks each raw state hash, source hash, contract, config, and exact step.
    # If its artifacts are absent/incompatible, _control_case rejects them and performs a fresh run.
    control_analysis = chns._control_case(prior_root, target)
    record = control_analysis.get("run_record", {})
    actual_step = int(record.get("steps", -1))
    if not bool(control_analysis.get("converged")):
        raise AuditValidationError(
            f"{case_name} required converged control; run returned nonconverged at step {actual_step}"
        )
    if actual_step != expected_step:
        # Preserve the required L1A-2j endpoints unless an exact alternate endpoint has a valid
        # convergence record and is explicitly described. The reference run is expected to close
        # at these same steps; a changed stop point is not silently relabelled.
        raise AuditValidationError(f"{case_name} endpoint changed: expected {expected_step}, got {actual_step}")
    endpoint = control_dir / "endpoint.npz"
    state, checkpoint_meta = chns.load_forensic_checkpoint(
        endpoint,
        expected_config=config,
        expected_step=actual_step,
        expected_kind=f"control_{int(target):03d}",
    )
    if checkpoint_meta.get("state_hashes") != control_analysis.get("endpoint_state_hashes"):
        raise AuditValidationError(f"{case_name} run record and checkpoint state hashes disagree")
    state_meta = _save_state_snapshot(
        snapshot_path,
        state,
        case_name=case_name,
        step=actual_step,
        config=config,
        metadata={
            "source_checkpoint": str(endpoint),
            "source_checkpoint_metadata": checkpoint_meta,
            "control_converged": True,
            "control_run_steps": actual_step,
        },
    )
    progress["state_reuse_decisions"].append(
        {
            "case": case_name,
            "candidate": str(endpoint),
            "status": "accepted_after_exact_checkpoint_validation",
            "step": actual_step,
            "state_hashes": checkpoint_meta["state_hashes"],
        }
    )
    return state, state_meta, control_analysis


def _ensure_ch_only_060_state(
    root: Path,
    artifact_dir: Path,
    snapshot_path: Path,
    *,
    progress: dict[str, Any],
    max_steps: int = 300_000,
    chunk_steps: int = 10_000,
) -> tuple[pf.State | None, dict[str, Any]]:
    """Run/resume a strict-convergence CH-only 60-degree reference for the phi comparison."""
    base_p, solid, seed, base_config = chns._make_case(60.0, N_value=chns.N, dt=chns.DT, M=chns.M_REF)
    config = dict(base_config)
    sample_every = 1_000
    config.update(
        {
            "dynamics_mode": "CH_ONLY",
            "diagnostic_only": True,
            "diagnostic_sample_every_steps": sample_every,
            "convergence_criteria": dict(nwa.CRITERIA),
        }
    )
    if chunk_steps <= 0 or max_steps < chunk_steps or chunk_steps % sample_every or max_steps % sample_every:
        raise ValueError(
            "CH-only stage and maximum horizons must be positive multiples of the 1000-step diagnostic sample cadence"
        )
    if max_steps < 10_000:
        raise ValueError("CH-only convergence search requires at least 10,000 diagnostic steps")

    # A converged L1A-2k endpoint can be resumed only if the full snapshot lineage still validates.
    if snapshot_path.exists():
        try:
            with np.load(snapshot_path, allow_pickle=False) as archive:
                cached_metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
            cached_step = int(cached_metadata.get("step", -1))
            cached_state, validated = _load_state_snapshot(
                snapshot_path,
                case_name="ch_only_060_converged",
                step=cached_step,
                config=config,
            )
            origin = validated.get("origin_metadata", {})
            gate = origin.get("convergence_gate", {})
            if origin.get("converged") is not True or gate.get("converged") is not True:
                raise ValueError("cached CH-only state did not close the unchanged convergence criteria")
            progress["state_reuse_decisions"].append(
                {
                    "case": "ch_only_060_converged",
                    "snapshot": str(snapshot_path),
                    "status": "reused_exact_converged_snapshot",
                    "step": cached_step,
                    "state_hashes": validated["state_hashes"],
                }
            )
            return cached_state, {
                "status": "measured_converged",
                "step": cached_step,
                "metadata": validated,
                "convergence_gate": gate,
                "decision": "reused_exact_l1a2k_ch_only_snapshot",
            }
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            progress["state_reuse_decisions"].append(
                {
                    "case": "ch_only_060_converged",
                    "snapshot": str(snapshot_path),
                    "status": "rejected_or_missing",
                    "reason": str(exc),
                }
            )

    branch_dir = artifact_dir / "phase_only_060"
    branch_dir.mkdir(parents=True, exist_ok=True)
    raw_checkpoint = branch_dir / "nwa_resume_latest.npz"
    sidecar_path = branch_dir / "resume_record.json"
    signature = {
        "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
        "N": int(chns.N),
        "target_deg": 60.0,
        "dynamics_mode": "ch_only",
        "M": float(chns.M_REF),
        "dt": float(chns.DT),
        "eps": float(base_p.eps),
        "R": float(chns.RADIUS),
        "wall_height": float(chns.WALL_HEIGHT),
        "wall_gain": 1.0,
        "phase_boundary_model": str(base_p.phase_boundary_model),
        "phase_transport_geometry": str(base_p.phase_transport_geometry),
        "phase_storage_model": str(base_p.phase_storage_model),
        "wall_measure_method": str(base_p.wall_measure),
    }
    resume_record: dict[str, Any] | None = None
    start_step = 0
    if raw_checkpoint.exists() and sidecar_path.exists():
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            with np.load(raw_checkpoint, allow_pickle=False) as archive:
                raw_arrays = {
                    "phi": np.array(archive["phi"], copy=True),
                    "u": np.array(archive["u"], copy=True),
                    "v": np.array(archive["v"], copy=True),
                    "t": np.array(archive["time"], copy=True),
                }
                raw_meta = json.loads(str(np.asarray(archive["metadata_json"]).item()))
            raw_hashes = {key: chns._array_hash(value) for key, value in raw_arrays.items()}
            checks = {
                "stage": sidecar.get("stage") == STAGE,
                "git_sha": sidecar.get("git_sha") == chns.get_git_sha(),
                "contract": sidecar.get("solver_contract_version")
                == SOLVER_CONTRACT
                == int(pf.SOLVER_CONTRACT_VERSION),
                "source_hashes": sidecar.get("source_hashes") == _source_hashes(),
                "config": sidecar.get("config") == config,
                "config_fingerprint": sidecar.get("config_fingerprint") == chns._canonical_hash(config),
                "nwa_signature": all(raw_meta.get(key) == value for key, value in signature.items()),
                "step": int(raw_meta.get("steps", -1))
                == int(sidecar.get("step", -2))
                == int(sidecar.get("run_record", {}).get("steps", -3)),
                "state_hashes": raw_hashes == sidecar.get("state_hashes"),
                "arrays_finite": all(np.isfinite(value).all() for value in raw_arrays.values()),
            }
            if not all(checks.values()):
                raise ValueError(f"strict CH-only resume validation failed: {checks}")
            start_step = int(sidecar["step"])
            resume_record = sidecar
            progress["state_reuse_decisions"].append(
                {
                    "case": "ch_only_060",
                    "candidate": str(raw_checkpoint),
                    "status": "accepted_exact_resumable_checkpoint",
                    "step": start_step,
                    "state_hashes": raw_hashes,
                }
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            progress["state_reuse_decisions"].append(
                {"case": "ch_only_060", "candidate": str(raw_checkpoint), "status": "rejected", "reason": str(exc)}
            )
            start_step, resume_record = 0, None

    run_kwargs = dict(
        ch_only=True,
        N=chns.N,
        eps_factor=chns.EPS_FACTOR,
        M=chns.M_REF,
        dt=chns.DT,
        R=chns.RADIUS,
        wall_height=chns.WALL_HEIGHT,
        sample_every=sample_every,
        keep_samples=2_000,
        label="L1A-2k converged CH-only 60-degree phi reference",
        group="L1A-2k-phase-only-reference",
        dtype="float32",
        ch_solver_rtol=base_p.ch_solver_rtol,
        ch_solver_max_iterations=base_p.ch_solver_max_iterations,
        wall_measure=base_p.wall_measure,
        phase_transport_geometry=base_p.phase_transport_geometry,
    )
    current_step = start_step
    last_record = None if resume_record is None else resume_record["run_record"]
    if last_record is not None:
        last_gate = nwa._window_converged(last_record.get("samples", [])[1:], True, dict(nwa.CRITERIA), M=chns.M_REF)
        if last_gate.get("converged"):
            with np.load(raw_checkpoint, allow_pickle=False) as archive:
                state = pf.State(
                    jnp.asarray(archive["phi"], dtype=jnp.float64),
                    jnp.asarray(archive["u"], dtype=jnp.float32),
                    jnp.asarray(archive["v"], dtype=jnp.float32),
                    jnp.asarray(archive["time"], dtype=jnp.float32),
                )
            snapshot_meta = _save_state_snapshot(
                snapshot_path,
                state,
                case_name="ch_only_060_converged",
                step=current_step,
                config=config,
                metadata={
                    "converged": True,
                    "convergence_gate": last_gate,
                    "nwa_signature": signature,
                    "run_record_steps": current_step,
                    "source_checkpoint": str(raw_checkpoint),
                },
            )
            return state, {
                "status": "measured_converged",
                "step": current_step,
                "metadata": snapshot_meta,
                "convergence_gate": last_gate,
                "decision": "resumed_converged_ch_only_checkpoint",
            }

    while current_step < max_steps:
        count = min(chunk_steps, max_steps - current_step)
        if current_step == 0:
            record = nwa.run_relaxation(
                60.0,
                fixed_steps=count,
                budgets=(count,),
                checkpoint_out=raw_checkpoint,
                **run_kwargs,
            )
        else:
            if last_record is None or not raw_checkpoint.exists():
                raise AuditValidationError("CH-only continuation lacks its exact staged checkpoint or prior samples")
            record = nwa.run_relaxation(
                60.0,
                fixed_steps=count,
                budgets=(current_step + count,),
                checkpoint_in=raw_checkpoint,
                checkpoint_out=raw_checkpoint,
                start_step=current_step,
                mass_reference=float(last_record["mass_reference_initial"]),
                conserved_mass_reference=float(last_record["conserved_mass_reference_initial"]),
                prior_samples=list(last_record["samples"]),
                **run_kwargs,
            )
        current_step = int(record["steps"])
        if current_step % sample_every:
            raise AuditValidationError(
                f"CH-only endpoint {current_step} is not on the {sample_every}-step sample cadence"
            )
        convergence_gate = nwa._window_converged(record.get("samples", [])[1:], True, dict(nwa.CRITERIA), M=chns.M_REF)
        with np.load(raw_checkpoint, allow_pickle=False) as archive:
            raw_arrays = {
                "phi": np.array(archive["phi"], copy=True),
                "u": np.array(archive["u"], copy=True),
                "v": np.array(archive["v"], copy=True),
                "t": np.array(archive["time"], copy=True),
            }
            raw_meta = json.loads(str(np.asarray(archive["metadata_json"]).item()))
        if int(raw_meta.get("steps", -1)) != current_step:
            raise AuditValidationError("NWA CH-only checkpoint step disagrees with the run record")
        state_hashes = {key: chns._array_hash(value) for key, value in raw_arrays.items()}
        if any(not np.isfinite(value).all() for value in raw_arrays.values()):
            raise AuditValidationError("NWA CH-only checkpoint contains non-finite values")
        sidecar = {
            "stage": STAGE,
            "case": "ch_only_060",
            "git_sha": chns.get_git_sha(),
            "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
            "source_hashes": _source_hashes(),
            "config": config,
            "config_fingerprint": chns._canonical_hash(config),
            "nwa_signature": signature,
            "step": current_step,
            "state_hashes": state_hashes,
            "convergence_gate": convergence_gate,
            "converged": bool(convergence_gate.get("converged")),
            "run_record": record,
            "production_semantics_changed": False,
        }
        _write_json(sidecar_path, sidecar)
        last_record = record
        progress["case_progress"]["ch_only_060"] = {
            "status": "converged" if convergence_gate.get("converged") else "running",
            "step": current_step,
            "maximum_steps": max_steps,
            "convergence_gate": convergence_gate,
            "state_hashes": state_hashes,
            "checkpoint": str(raw_checkpoint),
        }
        _save_progress(artifact_dir / "progress.json", progress)
        _write_json(root / "evidence" / "l1a2k" / "phase_only_progress.json", progress["case_progress"]["ch_only_060"])
        print(
            f"[l1a2k] CH-only 60-degree convergence check at step {current_step}: {convergence_gate.get('converged')}",
            flush=True,
        )
        if convergence_gate.get("converged"):
            state = pf.State(
                jnp.asarray(raw_arrays["phi"], dtype=jnp.float64),
                jnp.asarray(raw_arrays["u"], dtype=jnp.float32),
                jnp.asarray(raw_arrays["v"], dtype=jnp.float32),
                jnp.asarray(raw_arrays["t"].item(), dtype=jnp.float32),
            )
            state_meta = _save_state_snapshot(
                snapshot_path,
                state,
                case_name="ch_only_060_converged",
                step=current_step,
                config=config,
                metadata={
                    "converged": True,
                    "convergence_gate": convergence_gate,
                    "nwa_signature": signature,
                    "run_record_steps": current_step,
                    "source_checkpoint": str(raw_checkpoint),
                },
            )
            return state, {
                "status": "measured_converged",
                "step": current_step,
                "metadata": state_meta,
                "convergence_gate": convergence_gate,
                "decision": "ran_or_resumed_exact_ch_only_to_convergence",
                "run_record": {
                    key: record.get(key)
                    for key in (
                        "steps",
                        "converged",
                        "stop_reason",
                        "final_sampled_angle_deg",
                        "final_max_speed",
                        "phase_rate_l2",
                        "free_energy_final",
                    )
                },
            }
    return None, {
        "status": "not_converged_within_maximum_horizon",
        "step": current_step,
        "maximum_steps": max_steps,
        "last_convergence_gate": None
        if last_record is None
        else nwa._window_converged(last_record.get("samples", [])[1:], True, dict(nwa.CRITERIA), M=chns.M_REF),
        "decision": "state_not_used_as_converged_reference",
    }


def _analyse_case(
    case_name: str,
    state: pf.State,
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
    config: dict[str, Any],
    step: int,
    *,
    artifact_dir: Path,
    run_impulse: bool = True,
    run_frozen_branch: bool = True,
) -> dict[str, Any]:
    components = _capillary_components(state, solid, p)
    force = _force_decomposition(components, p)
    spatial = _case_spatial_summary(state, solid, p, components, force)
    manufactured = _manufactured_operator_tests(p)
    if not manufactured["manufactured_gradient_curl_passed"] or not manufactured["adjoint_passed"]:
        raise AuditValidationError(f"manufactured operator checks failed in {case_name}: {manufactured}")
    equal_density = _equal_density_one_step_ab(state, solid, p)
    impulse = _projected_one_step_impulse(state, solid, p) if run_impulse else {"status": "unmeasured"}
    frozen = _frozen_phi_forcing_ab(state, solid, p) if run_frozen_branch else {"status": "unmeasured"}
    pressure_row = force["production_work_dtype_acceleration"]
    if not pressure_row["passed"]:
        raise AuditValidationError(f"production projection decomposition failed for {case_name}")
    rho_field = _as_np64(pf.rho_of(state.phi, p))
    density_context = {
        "rho_l": float(p.rho_l),
        "rho_g": float(p.rho_g),
        "rho_phi_min": float(np.min(rho_field)),
        "rho_phi_max": float(np.max(rho_field)),
        "rho_phi_mean_full_momentum_grid": float(np.mean(rho_field, dtype=np.float64)),
        "rho_phi_span": float(np.max(rho_field) - np.min(rho_field)),
        "rho_phi_is_used_as_variable_pressure_coefficient": False,
        "rho_phi_is_used_as_momentum_inertia_in_current_rhs": False,
        "gravity_disabled": not bool(p.use_gravity),
        "variable_viscosity_nu_phi_remains_active": True,
        "momentum_metric": "uniform dx*dy on full momentum grid; no rho(phi) weighting in the pressure projection",
    }
    save_fields = {
        "state_phi": np.asarray(state.phi),
        "state_u": np.asarray(state.u),
        "state_v": np.asarray(state.v),
        "state_t": np.asarray(state.t),
        "mu": components["mu"],
        "mu_bulk": components["mu_bulk"],
        "mu_wall": components["mu_wall"],
        "force_density_x": components["force_density_x"],
        "force_density_y": components["force_density_y"],
        "raw_acceleration_x": components["raw_acceleration_x"],
        "raw_acceleration_y": components["raw_acceleration_y"],
        "effective_rhs_increment_x": components["effective_rhs_increment_x"],
        "effective_rhs_increment_y": components["effective_rhs_increment_y"],
        "projectable_potential": force["_arrays"]["production_work_dtype_acceleration"]["potential"],
        "capillary_projectable_x": force["_arrays"]["production_work_dtype_acceleration"]["gradient_x"],
        "capillary_projectable_y": force["_arrays"]["production_work_dtype_acceleration"]["gradient_y"],
        "capillary_residual_x": force["_arrays"]["production_work_dtype_acceleration"]["residual_x"],
        "capillary_residual_y": force["_arrays"]["production_work_dtype_acceleration"]["residual_y"],
        "capillary_residual_divergence": force["_arrays"]["production_work_dtype_acceleration"]["divergence_residual"],
        "capillary_poisson_residual": force["_arrays"]["production_work_dtype_acceleration"]["poisson_residual"],
        "capillary_curl_diagnostic_v1": _diagnostic_curl(
            force["_arrays"]["production_work_dtype_acceleration"]["source_x"],
            force["_arrays"]["production_work_dtype_acceleration"]["source_y"],
            p,
        ),
        "product_rule_defect_x": components["product_rule_defect_acceleration_x"],
        "product_rule_defect_y": components["product_rule_defect_acceleration_y"],
    }
    field_path = artifact_dir / "fields" / f"{case_name}.npz"
    _write_npz(field_path, **save_fields)
    field_hash = hashlib.sha256(field_path.read_bytes()).hexdigest()
    # Drop private array payloads from the JSON report; the reproducible compressed field artifact
    # and per-case source/state hashes remain available under artifacts/l1a2k.
    public_force = {key: value for key, value in force.items() if key != "_arrays"}
    return {
        "case": case_name,
        "target_deg": float(config["target_deg"]),
        "step": int(step),
        "time": float(np.asarray(state.t)),
        "config": config,
        "config_fingerprint": chns._canonical_hash(config),
        "state_hashes": _state_hashes(state),
        "state_dtypes": {key: np.asarray(getattr(state, key)).dtype.name for key in ("phi", "u", "v", "t")},
        "source_hashes": _source_hashes(),
        "density_context": density_context,
        "force_terms": {
            "form": "production Form A = +(SIGMA_NORM/We) mu_h * G_h(phi) / rho_l",
            "chemical_potential_includes_production_wall_energy_once": True,
            "production_rhs_addend_reconstruction_exact": components["production_rhs_reconstruction_exact"],
            "production_rhs_cast_increment_is_distinguished": True,
            "identity_reconstruction_linf": components["identity_reconstruction_linf"],
            "scaling": {"SIGMA_NORM": float(pf.SIGMA_NORM), "We": float(p.We), "rho_l": float(p.rho_l)},
        },
        "decomposition": public_force,
        "regional_spatial_summary": spatial,
        "manufactured_operator_checks": manufactured,
        "equal_density_one_step_ab": equal_density,
        "projected_one_step_impulse": impulse,
        "frozen_phi_forcing_ab": frozen,
        "field_artifact": {
            "path": str(field_path),
            "sha256": field_hash,
            "diagnostic_only": True,
            "contains_state_force_potential_residual_curl_and_product_rule_fields": True,
        },
        "diagnostic_only": True,
        "production_acceptance_evidence": False,
    }


def _cross_case_comparison(cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    keys = sorted(cases)
    metrics = [
        "source_acceleration_l2",
        "projectable_gradient_l2",
        "nonprojectable_residual_l2",
        "projectable_fraction_of_source_l2",
        "residual_fraction_of_source_l2",
        "residual_over_projectable_l2",
        "residual_divergence_l2",
        "poisson_solve_residual_l2",
    ]
    regional_names = sorted(
        set.intersection(*(set(cases[key]["regional_spatial_summary"]["regional_residuals_and_curl"]) for key in keys))
    )
    table: dict[str, Any] = {}
    for metric in metrics:
        values = {key: cases[key]["decomposition"]["production_work_dtype_acceleration"].get(metric) for key in keys}
        reference = values.get("authority_060")
        table[metric] = {
            "values": values,
            "authority_60_minus_control_90": None
            if reference is None or values.get("control_090") is None
            else float(reference - values["control_090"]),
            "authority_60_over_control_90": _safe_ratio(reference, values.get("control_090"))
            if reference is not None
            else None,
            "authority_60_minus_control_150": None
            if reference is None or values.get("control_150") is None
            else float(reference - values["control_150"]),
            "authority_60_over_control_150": _safe_ratio(reference, values.get("control_150"))
            if reference is not None
            else None,
        }
    regional: dict[str, Any] = {}
    for name in regional_names:
        regional[name] = {}
        for field in (
            "residual_l2",
            "residual_linf",
            "residual_to_source_l2_ratio",
            "residual_energy_fraction_of_global",
            "diagnostic_curl_v1_l2",
        ):
            values = {
                key: cases[key]["regional_spatial_summary"]["regional_residuals_and_curl"][name].get(field)
                for key in keys
            }
            ref = values.get("authority_060")
            regional[name][field] = {
                "values": values,
                "authority_60_minus_control_90": None
                if ref is None or values.get("control_090") is None
                else float(ref - values["control_090"]),
                "authority_60_over_control_90": _safe_ratio(ref, values.get("control_090"))
                if ref is not None
                else None,
                "authority_60_minus_control_150": None
                if ref is None or values.get("control_150") is None
                else float(ref - values["control_150"]),
                "authority_60_over_control_150": _safe_ratio(ref, values.get("control_150"))
                if ref is not None
                else None,
            }
    return {
        "matched_cases": keys,
        "global_effect_sizes": table,
        "regional_raw_values_and_effect_sizes": regional,
        "regional_masks_are_overlapping_not_partitioned": True,
        "effect_size_convention": "raw absolute difference and 60/control ratio; no fitted multiplier",
    }


def _mechanism_matrix(cases: dict[str, dict[str, Any]], *, static_laplace: dict[str, Any] | None) -> dict[str, Any]:
    all_cases = len(cases) == 3 and all(key in cases for key in CASE_TARGETS)
    residual_rows = [
        cases[key]["decomposition"]["production_work_dtype_acceleration"] for key in CASE_TARGETS if key in cases
    ]
    residual_present = all_cases and all(
        row["residual_fraction_of_source_l2"] is not None
        and row["residual_fraction_of_source_l2"] > 1_024.0 * row["work_dtype_epsilon"]
        for row in residual_rows
    )
    projection_valid = all_cases and all(row["passed"] for row in residual_rows)
    density_equal = all_cases and all(
        cases[key]["equal_density_one_step_ab"]["capillary_acceleration_max_abs_difference"] == 0.0
        and cases[key]["equal_density_one_step_ab"]["pressure_symbol_max_abs_difference"] == 0.0
        for key in CASE_TARGETS
    )
    product_rule_defect_present = all_cases and all(
        cases[key]["decomposition"]["discrete_form_identity"]["product_rule_defect_residual_l2"]
        > 1_024.0
        * cases[key]["decomposition"]["production_work_dtype_acceleration"]["work_dtype_epsilon"]
        * max(cases[key]["decomposition"]["production_work_dtype_acceleration"]["source_acceleration_l2"], 1.0e-30)
        for key in CASE_TARGETS
    )
    matrix = {
        "stage": STAGE,
        "scope": "diagnostic mechanism evidence only; not production validation and not a repair decision",
        "statuses": ["SUPPORTED", "SUSPECTED", "FALSIFIED", "NOT_TESTED"],
        "candidates": {
            "CAPILLARY_PRESSURE_IMBALANCE": {
                "status": "SUPPORTED" if residual_present else ("NOT_TESTED" if not all_cases else "FALSIFIED"),
                "claim_tested": "the exact production-dtype capillary acceleration has a nonzero component outside the exact production pressure-gradient range in all three matched states",  # noqa: E501
                "evidence": {
                    key: cases[key]["decomposition"]["production_work_dtype_acceleration"]
                    for key in CASE_TARGETS
                    if key in cases
                },
                "scope_caveat": "support establishes the force-subspace property only; it does not establish that this component caused the 60-degree contact-angle nonstationarity",  # noqa: E501
            },
            "PRESSURE_PROJECTION_LIMITED": {
                "status": "FALSIFIED" if projection_valid else "NOT_TESTED",
                "claim_tested": "production Poisson solve fails to remove the divergence of a production-gradient field or leaves a solve/reconstruction residual beyond dtype-aware roundoff tolerance",  # noqa: E501
                "evidence": {
                    key: cases[key]["decomposition"]["production_work_dtype_acceleration"]["checks"]
                    for key in CASE_TARGETS
                    if key in cases
                },
                "scope_caveat": "the nonzero solenoidal residual is outside the projectable subspace by definition; it is not a Poisson convergence failure",  # noqa: E501
            },
            "CAPILLARY_DENSITY_SCALING": {
                "status": "FALSIFIED" if all_cases else "NOT_TESTED",
                "claim_tested": "constant division by rho_l alone creates spatial curl or changes the residual/source fraction",  # noqa: E501
                "evidence": {
                    key: cases[key]["decomposition"]["force_density_vs_acceleration"]
                    for key in CASE_TARGETS
                    if key in cases
                },
                "scope_caveat": "does not clear the separate model-choice caveat about density-dependent inertia or projection; local-density scaling was not run",  # noqa: E501
            },
            "VARIABLE_DENSITY_COUPLING": {
                "status": "NOT_TESTED",
                "claim_tested": "whether a variable-density inertial/pressure model would improve the coupled dynamics",
                "evidence": {key: cases[key]["equal_density_one_step_ab"] for key in CASE_TARGETS if key in cases},
                "scope_caveat": "the one-step equal-density blend A/B is identical under the current gravity-off velocity formulation; no variable-density projection or inertia prototype was justified or run",  # noqa: E501
            },
            "DISCRETE_PRODUCT_RULE_IN_CAPILLARY_FORCE": {
                "status": "SUPPORTED" if product_rule_defect_present else ("FALSIFIED" if all_cases else "NOT_TESTED"),
                "claim_tested": "the production centred-difference operator has a measurable product-rule defect in the Form-A/Form-B identity; the projected residual is explicitly split into the Form-B term and defect",  # noqa: E501
                "evidence": {
                    key: cases[key]["decomposition"]["discrete_form_identity"] for key in CASE_TARGETS if key in cases
                },
                "scope_caveat": "this is an exact algebraic decomposition, not evidence that Form B is a better model; no force replacement or variant trajectory was used",  # noqa: E501
            },
            "WALL_ENERGY_MOMENTUM_CONSISTENCY": {
                "status": "NOT_TESTED",
                "claim_tested": "whether wall-energy thermodynamics and the full momentum/wall traction balance are energetically consistent",  # noqa: E501
                "evidence": {
                    key: {"wall_mu_acceleration": cases[key]["decomposition"]["wall_energy_mu_contribution"]}
                    for key in CASE_TARGETS
                    if key in cases
                },
                "scope_caveat": "wall-energy-derived force is measured and localized; no wall-energy, momentum, or contact-angle formulation was modified",  # noqa: E501
            },
            "BRINKMAN_WALL_COUPLING": {
                "status": "NOT_TESTED",
                "claim_tested": "causal contribution of Brinkman damping to the residual/dynamics",
                "evidence": {
                    key: cases[key]["regional_spatial_summary"]["brinkman_chi_overlap"]
                    for key in CASE_TARGETS
                    if key in cases
                },
                "scope_caveat": "spatial overlap is measured; no Brinkman retuning or on/off A/B was run without a separate evidence trigger",  # noqa: E501
            },
            "CONTACT_LINE_PINNING": {
                "status": "NOT_TESTED",
                "claim_tested": "static pinning threshold or unpinning-force mechanism",
                "evidence": {
                    key: cases[key]["regional_spatial_summary"]["contact_line_estimator"]
                    for key in CASE_TARGETS
                    if key in cases
                },
                "scope_caveat": "2dx and 4dx neighborhoods/exclusions are measured; no force-threshold test was run",
            },
            "Y_PERIODIC_TOPOLOGY_COUPLING": {
                "status": "NOT_TESTED",
                "claim_tested": "causal contribution of y-periodic seam coupling",
                "evidence": {
                    key: cases[key]["regional_spatial_summary"]["regional_residuals_and_curl"].get(
                        "y_periodic_seam_2cells"
                    )
                    for key in CASE_TARGETS
                    if key in cases
                },
                "scope_caveat": "seam overlap is localized only; no boundary-topology ablation was run",
            },
            "TIME_SPLITTING_OR_DT_SENSITIVITY": {
                "status": "NOT_TESTED",
                "claim_tested": "matched-physical-time dt/2 effect",
                "evidence": "unmeasured; frozen-state diagnostics were sufficient to localize the algebraic force residual",  # noqa: E501
                "scope_caveat": "dt/2 was not run",
            },
            "MOMENTUM_VISCOSITY_DISCRETIZATION": {
                "status": "NOT_TESTED",
                "claim_tested": "causal effect of the production viscosity operator",
                "evidence": "unmeasured; no viscosity A/B was run",
                "scope_caveat": "production nu(phi) and Laplacian remain unchanged",
            },
            "PHASE_KINETICS_LIMITED": {
                "status": "SUSPECTED",
                "claim_tested": "whether phase kinetics also contributes to the observed L1A-2j nonstationarity",
                "evidence": "L1A-2j retains PHASE_KINETICS_LIMITED=SUSPECTED; L1A-2k does not repeat or upgrade that classification",  # noqa: E501
                "scope_caveat": "no phase-mobility or phase-equation ablation was run",
            },
            "MULTIPLE_CONTRIBUTORS": {
                "status": "NOT_TESTED",
                "claim_tested": "joint causal attribution across multiple mechanisms",
                "evidence": "no matched causal factorial ablation was run",
                "scope_caveat": "mechanism closure is limited to the measured production-force subspace property",
            },
        },
        "equal_density_ablation": {
            "status": "measured_same_state_one_step" if density_equal else "unmeasured",
            "passed_exactly": density_equal,
            "local_density_scaling": "NOT_TESTED; no evidence trigger after rho_l is constant and equal-density blend A/B is unchanged",  # noqa: E501
            "variable_density_projection": "NOT_TESTED; equal-density A/B did not justify a prototype",
        },
        "static_laplace_sign_magnitude": static_laplace or {"status": "unmeasured"},
        "identified_discrete_force_choice": {
            "status": "SUPPORTED" if residual_present and projection_valid else "NOT_TESTED",
            "production_form": "+(SIGMA_NORM/We) * mu_h * G_h(phi) / rho_l, evaluated pointwise in pf.rhs and cast through the production momentum RHS",  # noqa: E501
            "pressure_projectable_subspace": "range(G_h) for the exact centred periodic production gradient and constant-coefficient m2_proj pressure solve",  # noqa: E501
            "discrete_identity": "mu_h G_h(phi) = G_h(mu_h phi) - phi G_h(mu_h) - E_h, where E_h is the measured centred-difference product-rule defect",  # noqa: E501
            "interpretation": "The product-form capillary source is not generally a pure production pressure gradient when mu_h varies spatially. The measured residual is split algebraically into the diagnostic -phi G_h(mu_h) term and E_h. This identifies the force/operator choice behind the subspace residual, but does not show that either term causes the CHNS angle nonstationarity or that an alternate form is preferable.",  # noqa: E501
            "force_variant_or_trajectory_used": False,
        },
        "final_root_cause": "CAPILLARY_PRESSURE_IMBALANCE" if residual_present and projection_valid else "INCONCLUSIVE",
        "final_root_cause_scope": "the measured structural non-projectable capillary-acceleration component only; not a causal explanation of the 60-degree angle-cycle/production-gate failure",  # noqa: E501
        "causal_root_cause_of_l1a2j_nonstationarity": "INCONCLUSIVE",
        "diagnostic_variants_are_not_production_validation": True,
    }
    return matrix


def _report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# L1A-2k: capillary–pressure balance audit",
        "",
        f"- **Profile:** `{report.get('profile')}`",
        f"- **Status:** `{report.get('status')}`",
        f"- **Git SHA:** `{report.get('repository_git_sha')}`",
        f"- **Solver contract:** `{report.get('solver_contract_version')}` (required 11; unchanged)",
        "- **Scope:** Diagnostic/root-cause forensic only. No production force, pressure projection, phase evolution, or defaults were changed.",  # noqa: E501
        "- **Decomposition:** production-operator decomposition via the exact centred periodic D/G and `m2_proj`; not called an exact orthogonal Hodge decomposition.",  # noqa: E501
        "",
        "## Matched snapshot provenance",
        "",
    ]
    reuse = report.get("l1a2j_reuse_audit", {})
    lines += [
        f"- Prior L1A-2j report available: `{reuse.get('report_available')}`; raw artifact directory existed at start: `{reuse.get('raw_artifact_directory_exists')}`.",  # noqa: E501
        f"- Reuse decision: {reuse.get('decision')}",
        "- Every accepted state is checked for contract, Git SHA, source hashes, exact config/fingerprint, state-array hashes, and step.",  # noqa: E501
        "",
        "| Case | Step | State status | State hash (phi) | Residual/source L2 | Projectable/source L2 | D residual L2 | Poisson solve residual L2 |",  # noqa: E501
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for name in ("authority_060", "control_090", "control_150"):
        case = report.get("cases", {}).get(name)
        if not case:
            lines.append(f"| {name} | — | unmeasured | — | — | — | — | — |")
            continue
        m = case["decomposition"]["production_work_dtype_acceleration"]
        lines.append(
            f"| {name} | {case['step']} | measured | `{case['state_hashes']['phi'][:16]}…` | "
            f"{m['residual_fraction_of_source_l2']:.8g} | {m['projectable_fraction_of_source_l2']:.8g} | "
            f"{m['residual_divergence_l2']:.6g} | {m['poisson_solve_residual_l2']:.6g} |"
        )
    lines += ["", "## Force-density and acceleration decomposition", ""]
    for name in ("authority_060", "control_090", "control_150"):
        case = report.get("cases", {}).get(name)
        if not case:
            continue
        force = case["decomposition"]
        base = force["production_work_dtype_acceleration"]
        density = force["force_density_vs_acceleration"]
        identity = force["discrete_form_identity"]
        lines += [
            f"### {name}",
            "",
            f"- Force-density L2: `{force['force_density']['field_l2_in_reported_units']:.9g}`; raw acceleration L2: `{force['raw_acceleration_expression']['field_l2_in_reported_units']:.9g}`.",  # noqa: E501
            f"- Production-work-dtype acceleration residual fraction: `{base['residual_fraction_of_source_l2']:.9g}`; projectable fraction: `{base['projectable_fraction_of_source_l2']:.9g}`.",  # noqa: E501
            f"- Residual divergence L2: `{base['residual_divergence_l2']:.6g}` (tolerance `{base['solve_tolerance_l2']:.6g}`); reconstruction L∞: `{base['reconstruction_linf']:.6g}` (tolerance `{base['reconstruction_tolerance_linf']:.6g}`).",  # noqa: E501
            f"- Constant `rho_l` residual-fraction difference, force density vs acceleration: `{density['residual_fraction_difference_force_vs_acceleration']:.6g}`.",  # noqa: E501
            f"- Product-form identity reconstruction L∞: `{identity['mu_product_identity_linf_before_pressure_projection']:.6g}`; projected identity residual L2 error: `{identity['production_residual_equals_alternative_minus_defect_residual_l2_error']:.6g}`.",  # noqa: E501
            f"- Frozen-φ matched on/off velocity L2 ratio: `{case['frozen_phi_forcing_ab']['matched_difference']['final_velocity_l2_on_over_off']}`; one-step projected impulse: `{case['projected_one_step_impulse']['capillary_on_final_velocity_l2']:.9g}`.",  # noqa: E501
            "",
        ]
    lines += ["## Regional residual and curl localization", ""]
    cross = report.get("cross_case_comparison", {})
    first = report.get("cases", {}).get("authority_060")
    if first:
        regional_names = first["regional_spatial_summary"]["regional_residuals_and_curl"].keys()
        lines += [
            "Masks are deterministic and intentionally overlapping. Norms use float64 reductions on the full uniform momentum-cell area `dx*dy`; residual-energy fractions therefore do not sum to one.",  # noqa: E501
            "",
            "| Region/mask | 60° residual L2 | 90° residual L2 | 150° residual L2 | 60° residual/source | 60/90 L2 | 60/150 L2 | 60° curl L2 |",  # noqa: E501
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name in regional_names:
            vals = []
            for key in ("authority_060", "control_090", "control_150"):
                row = (
                    report.get("cases", {})
                    .get(key, {})
                    .get("regional_spatial_summary", {})
                    .get("regional_residuals_and_curl", {})
                    .get(name)
                )
                vals.append(row)
            if not any(vals):
                continue
            fmt = lambda row, field: (
                "unmeasured" if row is None else f"{row[field]:.6g}" if row.get(field) is not None else "unmeasured"
            )
            effect = cross.get("regional_raw_values_and_effect_sizes", {}).get(name, {}).get("residual_l2", {})
            ratio_90 = effect.get("authority_60_over_control_90")
            ratio_150 = effect.get("authority_60_over_control_150")
            ratio_90_text = "unmeasured" if ratio_90 is None else f"{ratio_90:.6g}"
            ratio_150_text = "unmeasured" if ratio_150 is None else f"{ratio_150:.6g}"
            lines.append(
                f"| `{name}` | {fmt(vals[0], 'residual_l2')} | {fmt(vals[1], 'residual_l2')} | {fmt(vals[2], 'residual_l2')} | "  # noqa: E501
                f"{fmt(vals[0], 'residual_to_source_l2_ratio')} | {ratio_90_text} | {ratio_150_text} | "
                f"{fmt(vals[0], 'diagnostic_curl_v1_l2')} |"
            )
    if cross.get("global_effect_sizes"):
        lines += [
            "",
            "### Global cross-case effect sizes",
            "",
            "| Metric | 60° | 90° | 150° | 60−90 | 60/90 | 60−150 | 60/150 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for metric, values in cross["global_effect_sizes"].items():
            raw = values["values"]
            show = lambda value: "unmeasured" if value is None else f"{value:.6g}"
            lines.append(
                f"| `{metric}` | {show(raw.get('authority_060'))} | {show(raw.get('control_090'))} | {show(raw.get('control_150'))} | "  # noqa: E501
                f"{show(values.get('authority_60_minus_control_90'))} | {show(values.get('authority_60_over_control_90'))} | "  # noqa: E501
                f"{show(values.get('authority_60_minus_control_150'))} | {show(values.get('authority_60_over_control_150'))} |"  # noqa: E501
            )
    lines += ["", "## Mechanism matrix", "", "| Candidate | Status | Scope / evidence note |", "|---|---|---|"]
    matrix = report.get("mechanism_matrix", {}).get("candidates", {})
    for name, item in matrix.items():
        lines.append(f"| `{name}` | **{item['status']}** | {item.get('scope_caveat', item.get('claim_tested', ''))} |")
    if report.get("mechanism_matrix"):
        lines += [
            "",
            f"- **Final root cause of the measured capillary/pressure subspace property:** `{report['mechanism_matrix']['final_root_cause']}`.",  # noqa: E501
            f"- **Causal root cause of the L1A-2j 60° nonstationarity:** `{report['mechanism_matrix']['causal_root_cause_of_l1a2j_nonstationarity']}`.",  # noqa: E501
            f"- Scope: {report['mechanism_matrix']['final_root_cause_scope']}.",
        ]
    lines += ["", "## CHNS-50k versus converged CH-only 60° phi", ""]
    phase_only = report.get("phase_only_comparison", {})
    if phase_only.get("status") == "measured_converged_reference":
        diff = phase_only["phi_difference_chns_minus_ch_only"]
        mass = phase_only["formal_phase_mass_sum_V_phi"]
        lines += [
            f"- CHNS state: step `{phase_only['chns_step']}`; converged CH-only endpoint: step `{phase_only['ch_only_step']}`.",  # noqa: E501
            f"- `phi_CHNS − phi_CH-only`: L2(dxdy) `{diff['l2_dxdy']:.9g}`, L∞ `{diff['linf']:.9g}`, relative L2 vs CHNS `{diff['relative_l2_over_chns_phi']}`.",  # noqa: E501
            f"- Formal `sum_i(V_i*phi_i)` mass: CHNS `{mass['chns_50k']:.12g}`, CH-only `{mass['ch_only_converged']:.12g}`, absolute difference `{mass['absolute_difference']:.6g}` (comparison only, not an acceptance gate).",  # noqa: E501
            "- Full-grid `dx*dy*sum(phi)` is retained as a diagnostic only and is not used as formal mass or an acceptance gate.",  # noqa: E501
            f"- Convergence gate: `{phase_only['ch_only_convergence_gate']}`.",
        ]
    else:
        lines.append(f"- **{phase_only.get('status', 'unmeasured')}:** {phase_only.get('ch_only_run', phase_only)}")
    lines += ["", "## Static Laplace invariant", ""]
    laplace = report.get("static_laplace_audit", {})
    if laplace.get("status") == "measured":
        lines.append(
            f"- Existing `production.capillary_audit` checks rerun without changing any formulation: all passed = `{laplace['all_passed']}`."  # noqa: E501
        )
        for check in laplace.get("checks", []):
            if check.get("name") == "laplace_jump_sign_and_scale":
                lines.append(
                    f"- `laplace_jump_sign_and_scale`: passed `{check['passed']}`, observed `{check.get('observed')}`."
                )
    else:
        lines.append("- **unmeasured**")
    lines += ["", "## Unmeasured sections and scope limits", ""]
    for key, value in report.get("unmeasured_sections", {}).items():
        lines.append(f"- `{key}`: **{value}**")
    lines += [
        "",
        "## Artifacts and source provenance",
        "",
        f"- Field artifacts: `{report.get('artifact_directory')}/fields/` (diagnostic only).",
        "- Machine-readable report: `capillary_pressure_balance_report.json`.",
        "- Operator map: `operator_map.json`; source-function and whole-file SHA256 values are recorded there and in the manifest.",  # noqa: E501
        "- Production contract 11, capillary denominator, pressure projection, wall energy, phase storage, viscosity, Brinkman defaults, and acceptance thresholds remain unchanged.",  # noqa: E501
        "- No diagnostic variant is production validation; no authority continuation beyond step 50,000 was performed.",
    ]
    return "\n".join(lines) + "\n"


def _write_report_pair(report_dir: Path, report: dict[str, Any]) -> None:
    _write_json(report_dir / "capillary_pressure_balance_report.json", report)
    path = report_dir / "capillary_pressure_balance_report.md"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(_report_markdown(report), encoding="utf-8")
    temporary.replace(path)


def _initial_report(profile: str, artifact_dir: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    configs = {key: chns.production_config(target_deg=target) for key, target in CASE_TARGETS.items()}
    return {
        "stage": STAGE,
        "profile": profile,
        "status": "running",
        "repository_git_sha": chns.get_git_sha(),
        "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
        "required_solver_contract_version": SOLVER_CONTRACT,
        "production_semantics_changed": False,
        "diagnostic_only": True,
        "artifact_directory": str(artifact_dir),
        "evidence_directory": str(root / "evidence" / "l1a2k"),
        "source_hashes": _source_hashes(),
        "source_functions": operator_map()["source_functions"],
        "config_fingerprints": {key: chns._canonical_hash(value) for key, value in configs.items()},
        "case_requirements": {
            "authority_060": {"target_deg": 60.0, "step": 50_000, "extension_beyond_50k": False},
            "control_090": {"target_deg": 90.0, "converged": True, "reference_step": 27_200},
            "control_150": {"target_deg": 150.0, "converged": True, "reference_step": 100_000},
            "ch_only_060_comparison": {
                "target_deg": 60.0,
                "dynamics_mode": "CH_ONLY",
                "required": True,
                "must_meet_unchanged_convergence_criteria": True,
            },
        },
        "l1a2j_reuse_audit": _initial_l1a2j_reuse_audit(root),
        "state_reuse_decisions": [],
        "case_progress": {key: {"status": "unmeasured"} for key in CASE_TARGETS},
        "cases": {},
        "cross_case_comparison": {"status": "unmeasured"},
        "phase_only_comparison": {"status": "unmeasured"},
        "static_laplace_audit": {"status": "unmeasured"},
        "operator_validation": {"status": "unmeasured"},
        "mechanism_matrix": {},
        "unmeasured_sections": {
            "local_density_scaling_ab": "unmeasured_not_triggered_by_constant_rho_l_or_equal_density_one_step_ab",
            "variable_density_projection_or_inertia_prototype": "unmeasured_not_justified_by_equal_density_ab",
            "brinkman_on_off_ab": "unmeasured_pending_spatial_overlap_evidence_review",
            "dt_half_matched_physical_time": "unmeasured_frozen_state_diagnostics_not_insufficient",
            "contact_line_unpinning_threshold": "unmeasured_no_threshold_test_requested_or_triggered",
            "production_repair_or_validation": "not_in_scope",
        },
        "resume_policy": "exact contract + SHA + source hashes + config/fingerprint + state-array hashes + step; no migration or silent dtype conversion",  # noqa: E501
    }


def _run_static_laplace_audit() -> dict[str, Any]:
    try:
        result = capillary_audit.run_audit(N=128, R=0.8, We=100.0, eps_factor=2.0)
        record = result.to_dict()
        return {"status": "measured", "all_passed": bool(record["all_passed"]), **record}
    except Exception as exc:
        return {"status": "failed", "all_passed": False, "error": f"{type(exc).__name__}: {exc}"}


def run_quick_profile(artifact_dir: Path, report_dir: Path) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    report = _initial_report("quick", artifact_dir)
    report["quick_tests"] = {"status": "running"}
    _write_json(report_dir / "operator_map.json", operator_map())
    _write_json(
        report_dir / "manifest.json",
        {
            "stage": STAGE,
            "profile": "quick",
            "status": "running",
            "git_sha": chns.get_git_sha(),
            "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
            "production_semantics_changed": False,
            "source_hashes": _source_hashes(),
            "source_functions": operator_map()["source_functions"],
            "outputs": {"small_grid_operator_checks": "running", "matched_60_90_150": "unmeasured_by_design"},
        },
    )
    _write_report_pair(report_dir, report)
    p, solid, state, config = chns._make_case(60.0, N_value=24, dt=chns.DT, M=chns.M_REF)
    if not jax.config.x64_enabled:
        raise AuditValidationError("JAX_ENABLE_X64=1 is required for phase-only float64 contract-11 diagnostics")
    force = _force_decomposition(_capillary_components(state, solid, p), p)
    tests = _manufactured_operator_tests(p)
    if not all(
        (
            tests["manufactured_gradient_curl_passed"],
            tests["adjoint_passed"],
            force["production_work_dtype_acceleration"]["passed"] is True,
        )
    ):
        raise AuditValidationError("quick profile production-operator checks failed")
    report["quick_tests"] = {
        "status": "complete",
        "small_grid": {"N": p.Nx, "target_deg": 60.0, "config_fingerprint": chns._canonical_hash(config)},
        "production_operator_decomposition": {
            key: value for key, value in force["production_work_dtype_acceleration"].items() if key != "checks"
        }
        | {"checks": force["production_work_dtype_acceleration"]["checks"]},
        "manufactured_operator_checks": tests,
        "explicitly_not_matched_authority_or_control_evidence": True,
    }
    report["static_laplace_audit"] = _run_static_laplace_audit()
    report["status"] = "quick_complete"
    report["mechanism_matrix"] = _mechanism_matrix({}, static_laplace=report["static_laplace_audit"])
    _write_json(report_dir / "mechanism_matrix.json", report["mechanism_matrix"])
    _write_json(
        report_dir / "manifest.json",
        {
            "stage": STAGE,
            "profile": "quick",
            "status": "quick_complete",
            "git_sha": chns.get_git_sha(),
            "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
            "production_semantics_changed": False,
            "source_hashes": _source_hashes(),
            "source_functions": operator_map()["source_functions"],
            "outputs": {"small_grid_operator_checks": "complete", "matched_60_90_150": "unmeasured_by_design"},
            "mechanism_matrix": "mechanism_matrix.json",
        },
    )
    _write_report_pair(report_dir, report)
    return report


def run_forensic_profile(
    artifact_dir: Path,
    report_dir: Path,
    *,
    ch_only_max_steps: int = 300_000,
) -> dict[str, Any]:
    if int(pf.SOLVER_CONTRACT_VERSION) != SOLVER_CONTRACT:
        raise AuditValidationError(f"L1A-2k requires solver contract 11; found {pf.SOLVER_CONTRACT_VERSION}")
    if not jax.config.x64_enabled:
        raise AuditValidationError("set JAX_ENABLE_X64=1; contract-11 phase state storage requires x64")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = artifact_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    report = _initial_report("forensic", artifact_dir)
    manifest_path = report_dir / "manifest.json"
    map_data = operator_map()
    _write_json(report_dir / "operator_map.json", map_data)
    manifest = {
        "stage": STAGE,
        "profile": "forensic",
        "status": "running",
        "git_sha": chns.get_git_sha(),
        "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
        "production_semantics_changed": False,
        "source_hashes": _source_hashes(),
        "source_functions": map_data["source_functions"],
        "config_fingerprints": report["config_fingerprints"],
        "required_cases": report["case_requirements"],
        "reuse_validation": report["l1a2j_reuse_audit"],
        "outputs": {
            "snapshots": "running",
            "decomposition": "unmeasured",
            "regions": "unmeasured",
            "mechanisms": "unmeasured",
        },
        "resume_policy": report["resume_policy"],
    }
    _write_json(manifest_path, manifest)
    progress_path = artifact_dir / "progress.json"
    _save_progress(progress_path, report)
    _write_report_pair(report_dir, report)

    try:
        # Re-load strict report progress if it matches current source/config identity. Reports are
        # resumable, but measurements are recomputed from exact state checkpoints if report inputs
        # do not validate; a report is never treated as raw state.
        cached_report_path = report_dir / "capillary_pressure_balance_report.json"
        if cached_report_path.exists():
            try:
                cached = json.loads(cached_report_path.read_text(encoding="utf-8"))
                if (
                    cached.get("source_hashes") == _source_hashes()
                    and cached.get("repository_git_sha") == chns.get_git_sha()
                    and cached.get("solver_contract_version") == SOLVER_CONTRACT
                ):
                    for key, case in cached.get("cases", {}).items():
                        if key in CASE_TARGETS:
                            report["cases"][key] = case
            except (OSError, json.JSONDecodeError):
                pass

        case_states: dict[str, pf.State] = {}
        case_params: dict[str, pf.PhaseFieldParams] = {}
        case_solids: dict[str, pf.Solid] = {}
        for case_name, target in CASE_TARGETS.items():
            config = chns.production_config(target_deg=target)
            snapshot_path = snapshots_dir / f"{case_name}_step_{CASE_REQUIRED_STEPS[case_name]:06d}.npz"
            if case_name == "authority_060":
                state, state_decision = _ensure_authority_state(
                    Path(__file__).resolve().parents[1],
                    artifact_dir,
                    snapshot_path,
                    config,
                    progress=report,
                )
                p, solid, _seed, rebuilt_config = chns._make_case(target, N_value=chns.N, dt=chns.DT, M=chns.M_REF)
                if rebuilt_config != config:
                    raise AuditValidationError("authority case config reconstruction mismatch")
                step = 50_000
                control_record = None
            else:
                p, solid, _seed, rebuilt_config = chns._make_case(target, N_value=chns.N, dt=chns.DT, M=chns.M_REF)
                if rebuilt_config != config:
                    raise AuditValidationError(f"{case_name} config reconstruction mismatch")
                state, state_meta, control_record = _ensure_control_state(
                    Path(__file__).resolve().parents[1],
                    artifact_dir,
                    snapshot_path,
                    config,
                    target,
                    progress=report,
                )
                step = CASE_REQUIRED_STEPS[case_name]
                state_decision = {"decision": "strict_control_checkpoint_or_l1a2k_snapshot", "metadata": state_meta}
            case_states[case_name] = state
            case_params[case_name] = p
            case_solids[case_name] = solid
            current_state_hashes = _state_hashes(state)
            cached_case = report["cases"].get(case_name)
            if cached_case is not None:
                field_record = cached_case.get("field_artifact", {})
                field_path = Path(field_record.get("path", ""))
                valid_cached_case = (
                    cached_case.get("step") == step
                    and cached_case.get("config_fingerprint") == chns._canonical_hash(config)
                    and cached_case.get("state_hashes") == current_state_hashes
                    and cached_case.get("source_hashes") == _source_hashes()
                    and field_path.is_file()
                    and hashlib.sha256(field_path.read_bytes()).hexdigest() == field_record.get("sha256")
                )
                if valid_cached_case:
                    report["case_progress"][case_name] = {
                        "status": "reused_exact_report_and_state",
                        "step": step,
                        "state_hashes": current_state_hashes,
                    }
                    report["state_reuse_decisions"].append(
                        {
                            "case": case_name,
                            "status": "reused_exact_report_and_state",
                            "step": step,
                            "state_hashes": current_state_hashes,
                        }
                    )
                    _save_progress(progress_path, report)
                    _write_report_pair(report_dir, report)
                    print(f"[l1a2k] reused exact report/field/state for {case_name}", flush=True)
                    continue
            report["case_progress"][case_name] = {
                "status": "decomposing",
                "step": step,
                "state_hashes": current_state_hashes,
                "state_decision": state_decision,
            }
            _save_progress(progress_path, report)
            case_result = _analyse_case(
                case_name,
                state,
                solid,
                p,
                config,
                step,
                artifact_dir=artifact_dir,
            )
            if control_record is not None:
                case_result["converged_control_provenance"] = {
                    "converged": bool(control_record.get("converged")),
                    "steps": int(control_record.get("run_record", {}).get("steps", -1)),
                    "stop_reason": control_record.get("run_record", {}).get("stop_reason"),
                    "recorded_gate": control_record.get("recorded_production_gate"),
                }
            report["cases"][case_name] = case_result
            report["case_progress"][case_name] = {
                "status": "measured",
                "step": step,
                "state_hashes": case_result["state_hashes"],
            }
            report["operator_validation"] = {
                "status": "passed"
                if all(
                    case["decomposition"]["production_work_dtype_acceleration"]["passed"]
                    for case in report["cases"].values()
                )
                else "failed",
                "matched_cases_measured": list(report["cases"]),
                "fail_closed": True,
            }
            _save_progress(progress_path, report)
            _write_report_pair(report_dir, report)
            manifest["case_progress"] = report["case_progress"]
            manifest["state_hashes"] = {key: case["state_hashes"] for key, case in report["cases"].items()}
            manifest["case_steps"] = {key: case["step"] for key, case in report["cases"].items()}
            manifest["outputs"]["decomposition"] = (
                "partial_validated" if len(report["cases"]) < 3 else "complete_validated"
            )
            _write_json(manifest_path, manifest)
            print(f"[l1a2k] decomposed {case_name} at exact step {step}", flush=True)

        ch_only_snapshot = snapshots_dir / "ch_only_060_converged.npz"
        ch_only_state, ch_only_info = _ensure_ch_only_060_state(
            Path(__file__).resolve().parents[1],
            artifact_dir,
            ch_only_snapshot,
            progress=report,
            max_steps=ch_only_max_steps,
        )
        if ch_only_state is None or ch_only_info.get("status") != "measured_converged":
            report["phase_only_comparison"] = {
                "status": "unmeasured_ch_only_reference_not_converged",
                "ch_only_run": ch_only_info,
                "required": True,
                "not_used_as_a_matched_chns_control": True,
            }
            report["unmeasured_sections"]["ch_only_060_phi_comparison"] = (
                "unmeasured_ch_only_reference_not_converged_within_configured_horizon"
            )
        else:
            chns_state = case_states["authority_060"]
            p60, solid60 = case_params["authority_060"], case_solids["authority_060"]
            phi_chns = _as_np64(chns_state.phi)
            phi_ch = _as_np64(ch_only_state.phi)
            phi_difference = phi_chns - phi_ch
            cell_area = float(p60.dx * p60.dy)
            mass_chns = float(pf.liquid_mass(chns_state.phi, solid60, p60))
            mass_ch = float(pf.liquid_mass(ch_only_state.phi, solid60, p60))
            diagnostic_mass_chns = float(np.sum(phi_chns, dtype=np.float64) * cell_area)
            diagnostic_mass_ch = float(np.sum(phi_ch, dtype=np.float64) * cell_area)
            phi_chns_l2 = float(np.sqrt(np.sum(phi_chns**2, dtype=np.float64) * cell_area))
            phi_ch_l2 = float(np.sqrt(np.sum(phi_ch**2, dtype=np.float64) * cell_area))
            phase_diff_path = artifact_dir / "fields" / "phi_difference_chns50k_minus_converged_ch_only060.npz"
            _write_npz(
                phase_diff_path,
                phi_chns_at_50000=phi_chns,
                phi_ch_only_converged=phi_ch,
                phi_difference_chns_minus_ch_only=phi_difference,
            )
            report["phase_only_comparison"] = {
                "status": "measured_converged_reference",
                "chns_case": "authority_060",
                "chns_step": 50_000,
                "chns_state_hashes": _state_hashes(chns_state),
                "ch_only_case": "ch_only_060_converged",
                "ch_only_step": int(ch_only_info["step"]),
                "ch_only_state_hashes": _state_hashes(ch_only_state),
                "ch_only_config": ch_only_info["metadata"]["config"],
                "ch_only_convergence_gate": ch_only_info["convergence_gate"],
                "same_phi_grid_geometry_wetting_dt_and_M": True,
                "dynamics_difference": "CHNS-50k phase evolved with coupled velocity versus CH-only phase relaxation at frozen zero velocity",  # noqa: E501
                "phi_difference_chns_minus_ch_only": {
                    "l2_dxdy": float(np.sqrt(np.sum(phi_difference**2, dtype=np.float64) * cell_area)),
                    "linf": float(np.max(np.abs(phi_difference))),
                    "relative_l2_over_chns_phi": _safe_ratio(
                        float(np.sqrt(np.sum(phi_difference**2, dtype=np.float64) * cell_area)), phi_chns_l2
                    ),
                    "relative_l2_over_ch_only_phi": _safe_ratio(
                        float(np.sqrt(np.sum(phi_difference**2, dtype=np.float64) * cell_area)), phi_ch_l2
                    ),
                },
                "formal_phase_mass_sum_V_phi": {
                    "chns_50k": mass_chns,
                    "ch_only_converged": mass_ch,
                    "absolute_difference": abs(mass_chns - mass_ch),
                    "relative_difference": _safe_ratio(abs(mass_chns - mass_ch), max(abs(mass_chns), abs(mass_ch))),
                    "acceptance_gate": False,
                },
                "full_grid_dxdy_sum_phi_diagnostic_only": {
                    "chns_50k": diagnostic_mass_chns,
                    "ch_only_converged": diagnostic_mass_ch,
                    "absolute_difference": abs(diagnostic_mass_chns - diagnostic_mass_ch),
                    "acceptance_gate": False,
                },
                "field_artifact": {
                    "path": str(phase_diff_path),
                    "sha256": hashlib.sha256(phase_diff_path.read_bytes()).hexdigest(),
                    "diagnostic_only": True,
                },
                "production_acceptance_evidence": False,
                "not_a_production_solver_validation": True,
            }
            report["unmeasured_sections"].pop("ch_only_060_phi_comparison", None)
        report["cross_case_comparison"] = _cross_case_comparison(report["cases"])
        report["static_laplace_audit"] = _run_static_laplace_audit()
        if report["static_laplace_audit"].get("status") != "measured" or not report["static_laplace_audit"].get(
            "all_passed"
        ):
            report["unmeasured_sections"]["static_laplace_sign_magnitude_regression"] = (
                "failed_or_unmeasured_existing_production_capillary_audit"
            )
        else:
            report["unmeasured_sections"].pop("static_laplace_sign_magnitude_regression", None)
        report["mechanism_matrix"] = _mechanism_matrix(report["cases"], static_laplace=report["static_laplace_audit"])
        required_comparison_complete = report["phase_only_comparison"].get("status") == "measured_converged_reference"
        report["status"] = "complete" if required_comparison_complete else "incomplete_ch_only_reference"
        report["definition_of_done"] = {
            "all_three_matched_chns_states_exact_and_validated": len(report["cases"]) == 3,
            "all_production_operator_decompositions_passed": report["operator_validation"].get("status") == "passed",
            "converged_ch_only_60_phi_comparison_measured": required_comparison_complete,
            "overall_complete": len(report["cases"]) == 3
            and report["operator_validation"].get("status") == "passed"
            and required_comparison_complete,
        }
        manifest["status"] = report["status"]
        manifest["outputs"] = {
            "snapshots": "complete_exact_matched_states",
            "decomposition": "complete_validated",
            "regions": "complete_all_three_cases",
            "ch_only_phi_comparison": report["phase_only_comparison"].get("status", "unmeasured"),
            "mechanisms": "complete_with_unrun_ablation_sections_marked_unmeasured",
        }
        manifest["state_hashes"] = {key: case["state_hashes"] for key, case in report["cases"].items()}
        manifest["case_steps"] = {key: case["step"] for key, case in report["cases"].items()}
        manifest["mechanism_matrix"] = "mechanism_matrix.json"
        _write_json(manifest_path, manifest)
        _write_json(report_dir / "mechanism_matrix.json", report["mechanism_matrix"])
        _write_report_pair(report_dir, report)
        _save_progress(progress_path, report)
        return report
    except Exception as exc:
        report["status"] = "failed_closed"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        report["mechanism_matrix"] = _mechanism_matrix(
            report.get("cases", {}), static_laplace=report.get("static_laplace_audit")
        )
        manifest["status"] = "failed_closed"
        manifest["failure"] = report["failure"]
        manifest["case_progress"] = report.get("case_progress", {})
        _write_json(manifest_path, manifest)
        _write_json(report_dir / "mechanism_matrix.json", report["mechanism_matrix"])
        _write_report_pair(report_dir, report)
        _save_progress(progress_path, report)
        raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    base = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("quick", "forensic"), default="quick")
    parser.add_argument("--out", type=Path, default=base / "artifacts" / "l1a2k")
    parser.add_argument("--report-dir", type=Path, default=base / "evidence" / "l1a2k")
    parser.add_argument(
        "--ch-only-max-steps",
        type=int,
        default=300_000,
        help="diagnostic CH-only 60-degree convergence search horizon; not an authority continuation",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.profile == "quick":
            result = run_quick_profile(args.out, args.report_dir)
        else:
            result = run_forensic_profile(args.out, args.report_dir, ch_only_max_steps=args.ch_only_max_steps)
    except Exception as exc:
        print(f"L1A-2k {args.profile} failed closed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2
    print(f"L1A-2k {args.profile}: {result['status']}", flush=True)
    print(f"report: {args.report_dir / 'capillary_pressure_balance_report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
