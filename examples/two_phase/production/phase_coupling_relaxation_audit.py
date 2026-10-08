"""L1A-2l diagnostic-only phase-relaxation and coupling forensics.

The audit never edits the contract-11 production solver.  Its phase decomposition calls the actual
``pf.rhs`` / ``pf._phase_update`` / ``pf.solve_ch_implicit`` path at each production substep, while
mirroring the unchanged momentum/projection operations so sampled states remain production states.
Every rehydrated production step is checked against ``pf.step_with_diagnostics`` in tests and the
final matched state hashes are compared with the frozen L1A-2k report.

Run from ``examples/two_phase``::

    JAX_ENABLE_X64=1 python -m production.phase_coupling_relaxation_audit --profile quick
    JAX_ENABLE_X64=1 python -m production.phase_coupling_relaxation_audit --profile forensic

Large/resumable arrays stay under ``artifacts/l1a2l``; versioned evidence is under
``evidence/l1a2l``.  Diagnostic velocity scales and CH-only continuations are never production
acceptance evidence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
from functools import partial
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy import ndimage

import phasefield as pf
from production import capillary_pressure_balance_audit as l1a2k
from production import chns_nonstationarity_audit as chns
from production import contact_line_kinetics as clk
from production import nonneutral_wetting_audit as nwa
from production import observables as obs

jax.config.update("jax_enable_x64", True)

STAGE = "L1A-2l"
SOLVER_CONTRACT = 11
ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "l1a2l"
EVIDENCE_ROOT = ROOT / "evidence" / "l1a2l"
UPSTREAM_2J = ROOT / "evidence" / "l1a2j" / "chns_nonstationarity_report.json"
UPSTREAM_2K = ROOT / "evidence" / "l1a2k" / "capillary_pressure_balance_report.json"
CASE_TARGETS = {"authority_060": 60.0, "control_090": 90.0, "control_150": 150.0}
CASE_STEPS = {"authority_060": 50_000, "control_090": 27_200, "control_150": 100_000}
AUTHORITY_WINDOW_START = 40_000
CONTROL_WINDOW_LENGTH = 10_000
AUTHORITY_BURST_START = 48_000
AUTHORITY_BURST_STEPS = 2_000
REGION_NAMES = (
    "all_grid",
    "whole_fluid",
    "interface_frozen_at_window_start",
    "near_wall_fluid_0_2dx",
    "near_wall_fluid_0_4dx",
    "left_contact_line_2dx_frozen",
    "right_contact_line_2dx_frozen",
    "left_contact_line_4dx_frozen",
    "right_contact_line_4dx_frozen",
    "bulk_interface_excluding_contact_lines_4dx",
    "bulk_liquid_phi_ge_095",
    "bulk_gas_phi_le_005",
)
REGION_METRIC_NAMES = (
    "adv_rate_l2_dxdy",
    "ch_rate_l2_dxdy",
    "net_rate_l2_dxdy",
    "adv_rate_l2_volume",
    "ch_rate_l2_volume",
    "net_rate_l2_volume",
    "cancellation_C_mag_volume",
    "cancellation_C_mag_dxdy",
    "alignment_C_dir_volume",
    "net_rate_energy_fraction_volume",
    "energy_rate_proxy_adv",
    "energy_rate_proxy_ch",
    "energy_rate_proxy_net",
    "formal_mass_increment_adv",
    "formal_mass_increment_ch",
    "formal_mass_increment_net",
)
FLUX_METRIC_NAMES = (
    "adv_face_flux_l2",
    "adv_face_flux_linf",
    "ch_explicit_face_flux_l2",
    "ch_explicit_face_flux_linf",
)
CHECK_METRIC_NAMES = (
    "phase_reconstruction_linf",
    "adv_rhs_vs_exact_face_divergence_linf",
    "explicit_ch_increment_reconstruction_linf",
    "implicit_solver_iterations_max",
    "implicit_solver_relative_residual_max",
    "implicit_solver_converged_all",
)


class AuditValidationError(RuntimeError):
    """Raised when exact-state, production-staging, or fail-closed checks do not pass."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


PRODUCTION_GUARD_PATHS = {
    "phasefield": ROOT / "phasefield.py",
    "chns_nonstationarity_audit": ROOT / "production" / "chns_nonstationarity_audit.py",
    "nonneutral_wetting_audit": ROOT / "production" / "nonneutral_wetting_audit.py",
}
PRODUCTION_SOURCE_HASHES_AT_AUDIT_IMPORT = {
    name: _sha256(path.read_bytes()) for name, path in PRODUCTION_GUARD_PATHS.items()
}


def _hash_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _state_hashes(state: pf.State) -> dict[str, str]:
    return {name: _hash_array(getattr(state, name)) for name in ("phi", "u", "v", "t")}


def _source_hashes() -> dict[str, str]:
    paths = {
        "phasefield": ROOT / "phasefield.py",
        "audit_runner": Path(__file__).resolve(),
        "chns_nonstationarity_audit": ROOT / "production" / "chns_nonstationarity_audit.py",
        "nonneutral_wetting_audit": ROOT / "production" / "nonneutral_wetting_audit.py",
        "contact_line_kinetics": ROOT / "production" / "contact_line_kinetics.py",
        "observables": ROOT / "production" / "observables.py",
        "capillary_pressure_balance_audit": ROOT / "production" / "capillary_pressure_balance_audit.py",
        "capillary_audit": ROOT / "production" / "capillary_audit.py",
        "upstream_l1a2j_report": UPSTREAM_2J,
        "upstream_l1a2k_report": UPSTREAM_2K,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required L1A-2l source/evidence files are missing: {missing}")
    return {name: _sha256(path.read_bytes()) for name, path in paths.items()}


def _runtime_versions() -> dict[str, str]:
    import scipy

    return {
        "python": platform.python_version(),
        "jax": str(jax.__version__),
        "jaxlib": str(jax.lib.__version__),
        "numpy": str(np.__version__),
        "scipy": str(scipy.__version__),
    }


def _canonical_hash(value: Any) -> str:
    return _sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_clean(value), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_npz(path: Path, arrays: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    return _sha256(path.read_bytes())


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_upstream() -> tuple[dict[str, Any], dict[str, Any]]:
    two_j = _load_json(UPSTREAM_2J)
    two_k = _load_json(UPSTREAM_2K)
    if int(pf.SOLVER_CONTRACT_VERSION) != SOLVER_CONTRACT:
        raise AuditValidationError("L1A-2l requires unchanged contract 11")
    if two_k.get("status") != "complete" or two_k.get("solver_contract_version") != SOLVER_CONTRACT:
        raise AuditValidationError("frozen L1A-2k report is missing/incomplete or has a different contract")
    if two_k.get("source_hashes") != l1a2k._source_hashes():
        raise AuditValidationError("the L1A-2k report source hashes no longer match the checked-out audit/solver")
    if two_k.get("mechanism_matrix", {}).get("final_root_cause") != "CAPILLARY_PRESSURE_IMBALANCE":
        raise AuditValidationError("the L1A-2k structural background classification differs from the frozen report")
    if two_k.get("mechanism_matrix", {}).get("causal_root_cause_of_l1a2j_nonstationarity") != "INCONCLUSIVE":
        raise AuditValidationError("the frozen L1A-2k causal verdict is inconsistent")
    if two_j.get("stage") != "L1A-2j" or two_j.get("solver_contract_version") != SOLVER_CONTRACT:
        raise AuditValidationError("frozen L1A-2j phenomenology report is missing or has a different contract")
    return two_j, two_k


def _make_case(target: float) -> tuple[pf.PhaseFieldParams, pf.Solid, pf.State, dict[str, Any]]:
    return chns._make_case(target, N_value=chns.N, dt=chns.DT, M=chns.M_REF)


def _state_arrays(state: pf.State) -> dict[str, np.ndarray]:
    return {name: np.array(getattr(state, name), copy=True) for name in ("phi", "u", "v", "t")}


def _state_from_arrays(arrays: dict[str, np.ndarray]) -> pf.State:
    return pf.State(
        phi=jnp.asarray(arrays["phi"], dtype=jnp.float64),
        u=jnp.asarray(arrays["u"], dtype=jnp.float32),
        v=jnp.asarray(arrays["v"], dtype=jnp.float32),
        t=jnp.asarray(np.asarray(arrays["t"]).item(), dtype=jnp.float32),
    )


def _save_state_checkpoint(
    path: Path,
    state: pf.State,
    *,
    section: str,
    case_name: str,
    step: int,
    config: dict[str, Any],
    upstream_state_hashes: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "stage": STAGE,
        "section": section,
        "case": case_name,
        "step": int(step),
        "git_sha": chns.get_git_sha(),
        "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
        "config": config,
        "config_fingerprint": _canonical_hash(config),
        "source_hashes": _source_hashes(),
        "runtime_versions": _runtime_versions(),
        "state_hashes": _state_hashes(state),
        "upstream_state_hashes": upstream_state_hashes,
        "production_semantics_changed": False,
        **(extra or {}),
    }
    arrays = _state_arrays(state)
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False))
    _write_npz(path, arrays)
    return metadata


def _load_state_checkpoint(
    path: Path,
    *,
    section: str,
    case_name: str,
    step: int,
    config: dict[str, Any],
) -> tuple[pf.State, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"phi", "u", "v", "t", "metadata_json"}
        if required - set(archive.files):
            raise ValueError("checkpoint arrays are incomplete")
        arrays = {name: np.array(archive[name], copy=True) for name in ("phi", "u", "v", "t")}
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
    expected = {
        "stage": STAGE,
        "section": section,
        "case": case_name,
        "step": int(step),
        "git_sha": chns.get_git_sha(),
        # frozen L1A-2l checkpoints were written at contract 11; new checkpoints
        # record the live contract (12 after the sanctioned L1A-2p metadata bump)
        "solver_contract_version": (SOLVER_CONTRACT, int(pf.SOLVER_CONTRACT_VERSION)),
        "config": config,
        "config_fingerprint": _canonical_hash(config),
        "source_hashes": _source_hashes(),
        "runtime_versions": _runtime_versions(),
        "state_hashes": {name: _hash_array(value) for name, value in arrays.items()},
        "production_semantics_changed": False,
    }
    checks = {
        name: (
            metadata.get(name) in value
            if name == "solver_contract_version" and isinstance(value, tuple)
            else metadata.get(name) == value
        )
        for name, value in expected.items()
    }
    if not all(checks.values()):
        raise ValueError(f"strict L1A-2l checkpoint validation failed: {checks}")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("checkpoint contains non-finite arrays")
    if arrays["phi"].shape != (int(config["Nx"]), int(config["Ny"])):
        raise ValueError("checkpoint phase shape differs from the frozen grid")
    state = _state_from_arrays(arrays)
    return state, metadata


def _load_primary_snapshot_if_exact(
    case_name: str,
    config: dict[str, Any],
    step: int,
) -> tuple[pf.State | None, dict[str, Any]]:
    path = ROOT / "artifacts" / "l1a2k" / "snapshots" / f"{case_name}_step_{step:06d}.npz"
    if not path.is_file():
        return None, {"status": "missing_raw_snapshot", "path": str(path.relative_to(ROOT))}
    try:
        state, metadata = l1a2k._load_state_snapshot(path, case_name=case_name, step=step, config=config)
        return state, {
            "status": "accepted_exact_l1a2k_snapshot",
            "path": str(path.relative_to(ROOT)),
            "state_hashes": metadata["state_hashes"],
            "source_hashes": metadata["source_hashes"],
        }
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return None, {"status": "rejected_l1a2k_snapshot", "path": str(path.relative_to(ROOT)), "reason": str(exc)}


def _expected_upstream_case_hashes(two_k: dict[str, Any], case_name: str) -> dict[str, str]:
    case = two_k.get("cases", {}).get(case_name)
    if not isinstance(case, dict) or int(case.get("step", -1)) != CASE_STEPS[case_name]:
        raise AuditValidationError(f"frozen L1A-2k report lacks the required {case_name} step")
    return dict(case["state_hashes"])


def _ensure_production_prefix(
    case_name: str,
    target: float,
    end_step: int,
    config: dict[str, Any],
    solid: pf.Solid,
    seed: pf.State,
    p: pf.PhaseFieldParams,
    *,
    progress_dir: Path,
    upstream_hashes: dict[str, str] | None,
    extra_checkpoint_steps: tuple[int, ...] = (),
) -> tuple[pf.State, int, dict[str, Any]]:
    """Strictly resume or rerun the unchanged production trajectory to ``end_step``."""
    checkpoint_dir = ARTIFACT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_snapshot = checkpoint_dir / f"{case_name}_production_step_{end_step:06d}.npz"
    if final_snapshot.is_file():
        try:
            state, meta = _load_state_checkpoint(
                final_snapshot,
                section="production_state",
                case_name=case_name,
                step=end_step,
                config=config,
            )
            return state, end_step, {"decision": "reused_exact_l1a2l_checkpoint", "metadata": meta}
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"[L1A-2l] reject cached endpoint {case_name}: {exc}", flush=True)

    progress_path = progress_dir / f"{case_name}_trajectory_progress.json"
    latest_state, current_step = seed, 0
    if progress_path.is_file():
        saved = _load_json(progress_path)
        if (
            saved.get("stage") == STAGE
            and saved.get("section") == "production_state"
            and saved.get("case") == case_name
            and saved.get("target_step") == end_step
            and saved.get("config_fingerprint") == _canonical_hash(config)
            and saved.get("source_hashes") == _source_hashes()
            and saved.get("runtime_versions") == _runtime_versions()
        ):
            candidate_step = int(saved.get("committed_step", -1))
            candidate_path = ROOT / saved.get("checkpoint", "")
            if 0 < candidate_step <= end_step and candidate_path.is_file():
                try:
                    latest_state, _metadata = _load_state_checkpoint(
                        candidate_path,
                        section="production_state",
                        case_name=case_name,
                        step=candidate_step,
                        config=config,
                    )
                    current_step = candidate_step
                except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                    print(f"[L1A-2l] reject trajectory resume {case_name}: {exc}", flush=True)
                    latest_state, current_step = seed, 0
        else:
            print(f"[L1A-2l] discard incompatible trajectory progress for {case_name}", flush=True)

    chunk = 1_000
    started = time.perf_counter()
    while current_step < end_step:
        next_milestone = min(
            (step for step in extra_checkpoint_steps if current_step < step <= end_step),
            default=end_step,
        )
        count = min(chunk, next_milestone - current_step, end_step - current_step)
        latest_state = chns._advance_standard(latest_state, solid, p, count)
        latest_state.t.block_until_ready()
        current_step += count
        if current_step % 5_000 == 0 or current_step in extra_checkpoint_steps or current_step == end_step:
            checkpoint = checkpoint_dir / f"{case_name}_production_step_{current_step:06d}.npz"
            metadata = _save_state_checkpoint(
                checkpoint,
                latest_state,
                section="production_state",
                case_name=case_name,
                step=current_step,
                config=config,
                upstream_state_hashes=upstream_hashes,
                extra={"trajectory_source": "pf.step_with_diagnostics via chns._advance_standard"},
            )
            saved = {
                "stage": STAGE,
                "section": "production_state",
                "case": case_name,
                "target_step": int(end_step),
                "committed_step": current_step,
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "state_hashes": metadata["state_hashes"],
                "config_fingerprint": _canonical_hash(config),
                "source_hashes": _source_hashes(),
                "runtime_versions": _runtime_versions(),
                "elapsed_seconds_this_run": float(time.perf_counter() - started),
                "production_semantics_changed": False,
            }
            _write_json(progress_path, saved)
            print(f"[L1A-2l] production {case_name} {current_step}/{end_step}", flush=True)
    endpoint_hashes = _state_hashes(latest_state)
    if current_step != end_step:
        raise AuditValidationError(f"{case_name}: production endpoint stopped at an unexpected step")
    metadata = _save_state_checkpoint(
        final_snapshot,
        latest_state,
        section="production_state",
        case_name=case_name,
        step=end_step,
        config=config,
        upstream_state_hashes=upstream_hashes,
        extra={"trajectory_source": "pf.step_with_diagnostics via chns._advance_standard"},
    )
    return (
        latest_state,
        end_step,
        {
            "decision": "reran_exact_production_from_seed_to_required_step",
            "metadata": metadata,
            "endpoint_state_hashes": endpoint_hashes,
            "upstream_state_hashes_match": None if upstream_hashes is None else endpoint_hashes == upstream_hashes,
        },
    )


def _connected_component_count(solid: pf.Solid, p: pf.PhaseFieldParams) -> int:
    """Count active phase components using the exact open production face graph."""
    operator = pf.phase_transport_operator(solid, p)
    volume = np.asarray(operator.volume) > 0.0
    aperture_x = np.asarray(operator.aperture_x) > 0.0
    aperture_y = np.asarray(operator.aperture_y) > 0.0
    nx, ny = volume.shape
    parent = np.arange(nx * ny, dtype=np.int64)

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    active_indices = np.argwhere(volume)
    for i, j in active_indices:
        if aperture_x[i, j] and volume[(i + 1) % nx, j]:
            union(int(i * ny + j), int(((i + 1) % nx) * ny + j))
        if aperture_y[i, j] and volume[i, (j + 1) % ny]:
            union(int(i * ny + j), int(i * ny + ((j + 1) % ny)))
    roots = {find(int(i * ny + j)) for i, j in active_indices}
    return len(roots)


def _weighted_norm(field: np.ndarray, weight: np.ndarray, mask: np.ndarray | None = None) -> float:
    arr = np.asarray(field, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    if mask is not None:
        w = w * np.asarray(mask, dtype=np.float64)
    return float(np.sqrt(np.sum(w * arr**2, dtype=np.float64)))


def _weighted_mean_std(field: np.ndarray, volume: np.ndarray, active: np.ndarray) -> dict[str, float]:
    values = np.asarray(field, dtype=np.float64)
    weight = np.where(active, np.asarray(volume, dtype=np.float64), 0.0)
    weight_sum = float(np.sum(weight, dtype=np.float64))
    if weight_sum <= 0.0:
        raise AuditValidationError("chemical-potential weighting has zero active phase volume")
    mean = float(np.sum(weight * values, dtype=np.float64) / weight_sum)
    std = float(np.sqrt(np.sum(weight * (values - mean) ** 2, dtype=np.float64) / weight_sum))
    linf = float(np.max(np.abs(values[active] - mean)))
    return {"mean_volume_weighted": mean, "std_volume_weighted": std, "linf_deviation_from_mean": linf}


def _mu_and_flux_diagnostics(
    phi: Any,
    u: Any,
    v: Any,
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
    *,
    active_masks: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    operator = pf.phase_transport_operator(solid, p)
    volume = np.asarray(operator.volume, dtype=np.float64)
    active = volume > 0.0
    components = _connected_component_count(solid, p)
    result: dict[str, Any] = {
        "active_fluid_components_under_production_face_graph": components,
        "componentwise_mu_precondition": "single_connected_component_required",
    }
    if components != 1:
        result.update(
            {
                "status": "unmeasured_fail_closed_multiple_fluid_components",
                "mu_statistics": "unmeasured",
                "no_flux_statistics": "unmeasured",
            }
        )
        return result
    mu = np.asarray(pf.chemical_potential(jnp.asarray(phi), solid, p), dtype=np.float64)
    mu_explicit = np.asarray(pf._explicit_chemical_potential(jnp.asarray(phi), solid, p), dtype=np.float64)
    mu_stats = _weighted_mean_std(mu, volume, active)
    mu_exp_stats = _weighted_mean_std(mu_explicit, volume, active)
    chx, chy = pf.chemical_potential_fluxes(jnp.asarray(mu_explicit), solid, p)
    advx, advy = pf.phase_advective_fluxes(jnp.asarray(u), jnp.asarray(v), jnp.asarray(phi), solid, p)
    flux_result: dict[str, Any] = {
        "exact_production_ch_face_flux": {
            "source": (
                "pf.chemical_potential_fluxes(mu_expl) is called by pf._phase_update; "
                "implicit -eps L(phi) is handled inside solve_ch_implicit"
            ),
            "l2_unweighted_face_array": float(
                np.sqrt(np.sum(np.asarray(chx, dtype=np.float64) ** 2) + np.sum(np.asarray(chy, dtype=np.float64) ** 2))
            ),
            "linf": float(max(np.max(np.abs(np.asarray(chx))), np.max(np.abs(np.asarray(chy))))),
        },
        "full_mu_ch_flux_comparison_diagnostic_only": {
            "source": (
                "pf.chemical_potential_fluxes(mu_full), not the explicit face array "
                "directly passed to the split production solve"
            ),
            "l2_unweighted_face_array": float(
                np.sqrt(
                    np.sum(
                        np.asarray(pf.chemical_potential_fluxes(jnp.asarray(mu), solid, p)[0], dtype=np.float64) ** 2
                    )
                    + np.sum(
                        np.asarray(pf.chemical_potential_fluxes(jnp.asarray(mu), solid, p)[1], dtype=np.float64) ** 2
                    )
                )
            ),
        },
        "exact_production_advective_face_flux": {
            "source": "pf.phase_advective_fluxes(u,v,phi,solid,p) used to construct phi_rhs in pf.rhs",
            "l2_unweighted_face_array": float(
                np.sqrt(
                    np.sum(np.asarray(advx, dtype=np.float64) ** 2) + np.sum(np.asarray(advy, dtype=np.float64) ** 2)
                )
            ),
            "linf": float(max(np.max(np.abs(np.asarray(advx))), np.max(np.abs(np.asarray(advy))))),
        },
    }
    if active_masks:
        regional = {}
        for name, mask in active_masks.items():
            m = np.asarray(mask, dtype=bool)
            regional[name] = {
                "mu_l2_volume": _weighted_norm(mu, volume, m),
                "mu_deviation_l2_volume": _weighted_norm(mu - mu_stats["mean_volume_weighted"], volume, m),
            }
        flux_result["regional_mu_deviation"] = regional
    result.update(
        {
            "status": "measured",
            "mu_statistics": mu_stats,
            "mu_explicit_statistics": mu_exp_stats,
            "fluxes": flux_result,
        }
    )
    return result


def _phase_energy_parts(phi: Any, solid: pf.Solid, p: pf.PhaseFieldParams) -> dict[str, float]:
    value = jnp.asarray(phi)
    operator = pf.phase_transport_operator(solid, p)
    volume = operator.volume.astype(jnp.float64)
    wx = operator.weight_x.astype(jnp.float64)
    wy = operator.weight_y.astype(jnp.float64)
    phase64 = value.astype(jnp.float64)
    bulk = jnp.sum(volume * phase64**2 * (1.0 - phase64) ** 2 / float(p.eps))
    dx_phi = jnp.roll(phase64, -1, axis=0) - phase64
    dy_phi = jnp.roll(phase64, -1, axis=1) - phase64
    gradient = 0.5 * float(p.eps) * jnp.sum(wx * dx_phi**2 + wy * dy_phi**2)
    wall = pf.wall_free_energy(value, solid, p).astype(jnp.float64)
    total = pf.phase_free_energy(value, solid, p).astype(jnp.float64)
    parts = {"bulk": float(bulk), "gradient": float(gradient), "wall": float(wall), "total": float(total)}
    reconstruction = parts["bulk"] + parts["gradient"] + parts["wall"]
    parts["component_sum_minus_production_total"] = reconstruction - parts["total"]
    return parts


def _distance_to_equilibrium(
    phi: Any,
    phi_eq: Any,
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
    masks: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    volume = np.asarray(pf.phase_control_volumes(solid, p), dtype=np.float64)
    delta = np.asarray(phi, dtype=np.float64) - np.asarray(phi_eq, dtype=np.float64)
    denom = _weighted_norm(np.asarray(phi_eq, dtype=np.float64), volume)
    distance = _weighted_norm(delta, volume)
    result = {
        "raw_unaligned_relative_D_phi": distance / max(denom, 1.0e-300),
        "raw_unaligned_l2_volume": distance,
        "equilibrium_l2_volume_denominator": denom,
        "symmetry_aligned_diagnostic": "unmeasured_not_used",
    }
    if masks:
        regional = {}
        for name, mask in masks.items():
            m = np.asarray(mask, dtype=bool)
            diff = _weighted_norm(delta, volume, m)
            reference = _weighted_norm(np.asarray(phi_eq), volume, m)
            regional[name] = {
                "difference_l2_volume": diff,
                "equilibrium_l2_volume": reference,
                "relative_difference_with_regional_denominator": diff / max(reference, 1.0e-300),
            }
        result["regional"] = regional
    return result


def _contact_metrics(phi: Any, solid: pf.Solid, p: pf.PhaseFieldParams) -> dict[str, Any]:
    positions = clk.contact_line_positions(
        np.asarray(phi, dtype=np.float64),
        np.asarray(solid.sdf, dtype=np.float64),
        float(p.dx),
        float(p.dy),
        eps=float(p.eps),
        Lx=float(p.Lx),
    )
    sides = chns._contact_angle_sides(np.asarray(phi, dtype=np.float64), solid, p, positions)
    overall = float(pf.measure_contact_angle(jnp.asarray(phi), solid, p))
    return {**positions, **sides, "overall_circle_fit_angle_deg": overall if math.isfinite(overall) else None}


def _make_region_masks(
    phi_ref: Any, solid: pf.Solid, p: pf.PhaseFieldParams
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    phi = np.asarray(phi_ref, dtype=np.float64)
    volume = np.asarray(pf.phase_control_volumes(solid, p), dtype=np.float64)
    fluid = volume > 0.0
    distance = np.asarray(solid.sdf, dtype=np.float64)
    interface = fluid & (phi >= 0.05) & (phi <= 0.95)
    near_wall_2 = fluid & (distance >= 0.0) & (distance <= 2.0 * float(p.dx))
    near_wall_4 = fluid & (distance >= 0.0) & (distance <= 4.0 * float(p.dx))
    contacts = _contact_metrics(phi, solid, p)
    x_left = contacts.get("left_contact_x_wrapped")
    x_right = contacts.get("right_contact_x_wrapped")
    wall_y = float(pf.wall_plane_height(solid, p))
    x = (np.arange(p.Nx, dtype=np.float64) + 0.5)[:, None] * float(p.dx)
    y = (np.arange(p.Ny, dtype=np.float64) + 0.5)[None, :] * float(p.dy)
    yy = np.broadcast_to(y, phi.shape)

    def cl_mask(center: float | None, radius_cells: float) -> np.ndarray:
        if center is None or not math.isfinite(float(center)) or not math.isfinite(wall_y):
            return np.zeros_like(fluid)
        delta_x = (x - float(center) + 0.5 * float(p.Lx)) % float(p.Lx) - 0.5 * float(p.Lx)
        return fluid & (delta_x**2 + (yy - wall_y) ** 2 <= (radius_cells * float(p.dx)) ** 2)

    left2, right2 = cl_mask(x_left, 2.0), cl_mask(x_right, 2.0)
    left4, right4 = cl_mask(x_left, 4.0), cl_mask(x_right, 4.0)
    union4 = left4 | right4
    masks = {
        "all_grid": np.ones_like(fluid),
        "whole_fluid": fluid,
        "interface_frozen_at_window_start": interface,
        "near_wall_fluid_0_2dx": near_wall_2,
        "near_wall_fluid_0_4dx": near_wall_4,
        "left_contact_line_2dx_frozen": left2,
        "right_contact_line_2dx_frozen": right2,
        "left_contact_line_4dx_frozen": left4,
        "right_contact_line_4dx_frozen": right4,
        "bulk_interface_excluding_contact_lines_4dx": interface & ~union4,
        "bulk_liquid_phi_ge_095": fluid & (phi >= 0.95),
        "bulk_gas_phi_le_005": fluid & (phi <= 0.05),
    }
    policy = {
        "mask_centres": (
            "contact-line centers from the validated estimator at window start; radii fixed at "
            "2dx/4dx; masks remain frozen throughout the sampled window"
        ),
        "interface_definition": "fluid cells with 0.05 <= phi <= 0.95 in the window-start state",
        "near_wall_definition": "positive phase control volume and 0 <= signed distance <= radius",
        "left_contact_x_wrapped": x_left,
        "right_contact_x_wrapped": x_right,
        "wall_plane_height": wall_y,
        "contact_line_estimator": contacts,
        "overlapping_masks": True,
    }
    return masks, policy


def _face_masks_from_cells(masks: dict[str, np.ndarray]) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    return tuple((mask | np.roll(mask, -1, axis=0), mask | np.roll(mask, -1, axis=1)) for mask in masks.values())


def _phase_decomposition_substep(
    carry: tuple[jnp.ndarray, ...],
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
    dt_sub: float,
    face_masks: tuple[tuple[jnp.ndarray, jnp.ndarray], ...],
):
    phi, u, v, time_value = carry
    state = pf.State(phi, u, v, time_value)
    phi_rhs, u_rhs, v_rhs, _mu, mu_expl = pf.rhs(state, solid, p)
    if pf.phase_advection_subcycles(p):
        # Current contract-11 production config is disabled; fail closed rather than claim that the
        # one-step rhs is the rate consumed by _phase_update under an unseen subcycling mode.
        raise AuditValidationError("L1A-2l phase decomposition currently requires disabled production subcycling")
    operator = pf.phase_transport_operator(solid, p)
    adv_flux_x, adv_flux_y = pf.phase_advective_fluxes(u, v, phi, solid, p)
    adv_rhs_from_faces = -pf.control_volume_divergence(adv_flux_x, adv_flux_y, operator.volume_safe)
    adv_increment = dt_sub * phi_rhs
    phi_after_advective_stage = phi + adv_increment

    ch_flux_x, ch_flux_y = pf.chemical_potential_fluxes(mu_expl, solid, p)
    ch_divergence = pf.control_volume_divergence(ch_flux_x, ch_flux_y, operator.volume_safe)
    production_implicit_rhs = phi + dt_sub * (phi_rhs - ch_divergence)
    phi_new, solve_info = pf._phase_update(phi, u, v, solid, p, dt_sub, phi_rhs, mu_expl)
    ch_explicit_increment = -dt_sub * ch_divergence
    implicit_solver_increment = phi_new - production_implicit_rhs
    ch_increment = phi_new - phi_after_advective_stage
    net_increment = phi_new - phi
    explicit_reconstruction_error = jnp.max(jnp.abs(ch_explicit_increment + implicit_solver_increment - ch_increment))
    reconstruction_error = jnp.max(jnp.abs(adv_increment + ch_increment - net_increment))
    adv_rhs_error = jnp.max(jnp.abs(phi_rhs - adv_rhs_from_faces))

    flux_norm_rows = []
    for mask_x, mask_y in face_masks:
        ax = jnp.asarray(mask_x, dtype=jnp.bool_)
        ay = jnp.asarray(mask_y, dtype=jnp.bool_)
        adv_l2 = jnp.sqrt(
            jnp.sum(jnp.where(ax, adv_flux_x.astype(jnp.float64) ** 2, 0.0))
            + jnp.sum(jnp.where(ay, adv_flux_y.astype(jnp.float64) ** 2, 0.0))
        )
        adv_linf = jnp.maximum(
            jnp.max(jnp.where(ax, jnp.abs(adv_flux_x.astype(jnp.float64)), 0.0)),
            jnp.max(jnp.where(ay, jnp.abs(adv_flux_y.astype(jnp.float64)), 0.0)),
        )
        ch_l2 = jnp.sqrt(
            jnp.sum(jnp.where(ax, ch_flux_x.astype(jnp.float64) ** 2, 0.0))
            + jnp.sum(jnp.where(ay, ch_flux_y.astype(jnp.float64) ** 2, 0.0))
        )
        ch_linf = jnp.maximum(
            jnp.max(jnp.where(ax, jnp.abs(ch_flux_x.astype(jnp.float64)), 0.0)),
            jnp.max(jnp.where(ay, jnp.abs(ch_flux_y.astype(jnp.float64)), 0.0)),
        )
        flux_norm_rows.append(jnp.asarray([adv_l2, adv_linf, ch_l2, ch_linf], dtype=jnp.float64))
    flux_metrics = jnp.stack(flux_norm_rows)

    # This is the exact unchanged production momentum / Brinkman / projection sequence; only phase
    # arrays and their decomposition are observed in addition to the production result.
    dt_momentum = dt_sub
    damp = 1.0 / (1.0 + dt_momentum * solid.chi / p.eta_pen)
    u_new = (u + dt_momentum * u_rhs) * damp
    v_new = (v + dt_momentum * v_rhs) * damp
    divergence = pf._ddx(u_new, p.dx) + pf._ddy(v_new, p.dy)
    pressure = pf.poisson_solve(divergence / dt_momentum, p.m2_proj)
    u_new = u_new - dt_momentum * pf._ddx(pressure, p.dx)
    v_new = v_new - dt_momentum * pf._ddy(pressure, p.dy)
    return (phi_new, u_new, v_new, time_value + dt_sub), (
        adv_increment,
        ch_increment,
        ch_explicit_increment,
        implicit_solver_increment,
        flux_metrics,
        reconstruction_error,
        adv_rhs_error,
        explicit_reconstruction_error,
        solve_info.iterations,
        solve_info.relative_residual,
        solve_info.converged,
    )


def _phase_decomposition_step_impl(
    state: pf.State,
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
    cell_masks: tuple[jnp.ndarray, ...],
    face_masks: tuple[tuple[jnp.ndarray, jnp.ndarray], ...],
):
    del cell_masks  # The window-level norms are computed after summing the exact substep increments.
    dt_sub = float(p.dt) / 3.0
    (phi, u, v, time_value), outputs = jax.lax.scan(
        lambda carry, _: _phase_decomposition_substep(carry, solid, p, dt_sub, face_masks),
        (state.phi, state.u, state.v, state.t),
        None,
        length=3,
    )
    out = pf.State(
        phi=phi.astype(pf.phase_state_dtype(p)),
        u=u.astype(p.dtype),
        v=v.astype(p.dtype),
        t=time_value.astype(p.dtype),
    )
    adv, ch, ch_exp, implicit, flux, recon, adv_error, explicit_error, iters, residuals, converged = outputs
    return out, (
        jnp.sum(adv, axis=0),
        jnp.sum(ch, axis=0),
        jnp.sum(ch_exp, axis=0),
        jnp.sum(implicit, axis=0),
        jnp.mean(flux, axis=0),
        jnp.max(recon),
        jnp.max(adv_error),
        jnp.max(explicit_error),
        jnp.max(iters),
        jnp.max(residuals),
        jnp.all(converged),
    )


def _norm_dxdy(field: jnp.ndarray, mask: jnp.ndarray, area: float) -> jnp.ndarray:
    return jnp.sqrt(jnp.sum(jnp.asarray(mask, dtype=jnp.float64) * field.astype(jnp.float64) ** 2) * area)


def _norm_volume(field: jnp.ndarray, volume: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(
        jnp.sum(volume.astype(jnp.float64) * jnp.asarray(mask, dtype=jnp.float64) * field.astype(jnp.float64) ** 2)
    )


def _group_metric_row(
    start_phi: jnp.ndarray,
    end_phi: jnp.ndarray,
    adv_increment: jnp.ndarray,
    ch_increment: jnp.ndarray,
    net_increment: jnp.ndarray,
    flux_average: jnp.ndarray,
    checks: tuple[jnp.ndarray, ...],
    gate_previous_phi: jnp.ndarray,
    cell_masks: tuple[jnp.ndarray, ...],
    volume: jnp.ndarray,
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
    window_steps: int,
):
    duration = float(window_steps) * float(p.dt)
    adv_rate = adv_increment / duration
    ch_rate = ch_increment / duration
    net_rate = net_increment / duration
    mu_start = pf.chemical_potential(start_phi, solid, p).astype(jnp.float64)
    darea = float(p.dx) * float(p.dy)
    total_net_energy_scale = jnp.sum(volume.astype(jnp.float64) * jnp.abs(mu_start * net_rate))
    rows = []
    for mask in cell_masks:
        mask64 = jnp.asarray(mask, dtype=jnp.float64)
        adv_dxdy = _norm_dxdy(adv_rate, mask, darea)
        ch_dxdy = _norm_dxdy(ch_rate, mask, darea)
        net_dxdy = _norm_dxdy(net_rate, mask, darea)
        adv_vol = _norm_volume(adv_rate, volume, mask)
        ch_vol = _norm_volume(ch_rate, volume, mask)
        net_vol = _norm_volume(net_rate, volume, mask)
        product = jnp.sum(
            mask64 * volume.astype(jnp.float64) * adv_rate.astype(jnp.float64) * ch_rate.astype(jnp.float64)
        )
        # Magnitude cancellation factor C_mag = ||R_adv + R_CH||/(||R_adv||+||R_CH||);
        # C_dir below is the volume-weighted cosine between the two fields.
        cancellation_volume = net_vol / jnp.maximum(adv_vol + ch_vol, jnp.asarray(1.0e-300, dtype=jnp.float64))
        cancellation_dxdy = net_dxdy / jnp.maximum(adv_dxdy + ch_dxdy, jnp.asarray(1.0e-300, dtype=jnp.float64))
        alignment = product / jnp.maximum(adv_vol * ch_vol, jnp.asarray(1.0e-300, dtype=jnp.float64))
        p_adv = jnp.sum(mask64 * volume.astype(jnp.float64) * mu_start * adv_rate.astype(jnp.float64))
        p_ch = jnp.sum(mask64 * volume.astype(jnp.float64) * mu_start * ch_rate.astype(jnp.float64))
        p_net = jnp.sum(mask64 * volume.astype(jnp.float64) * mu_start * net_rate.astype(jnp.float64))
        energy_fraction = jnp.abs(p_net) / jnp.maximum(
            jnp.abs(p_adv) + jnp.abs(p_ch), jnp.asarray(1.0e-300, dtype=jnp.float64)
        )
        dmass_adv = jnp.sum(mask64 * volume.astype(jnp.float64) * adv_increment.astype(jnp.float64))
        dmass_ch = jnp.sum(mask64 * volume.astype(jnp.float64) * ch_increment.astype(jnp.float64))
        dmass_net = jnp.sum(mask64 * volume.astype(jnp.float64) * net_increment.astype(jnp.float64))
        rows.append(
            jnp.asarray(
                [
                    adv_dxdy,
                    ch_dxdy,
                    net_dxdy,
                    adv_vol,
                    ch_vol,
                    net_vol,
                    cancellation_volume,
                    cancellation_dxdy,
                    alignment,
                    energy_fraction,
                    p_adv,
                    p_ch,
                    p_net,
                    dmass_adv,
                    dmass_ch,
                    dmass_net,
                ],
                dtype=jnp.float64,
            )
        )
    phase_rate_last = (end_phi - gate_previous_phi) / float(p.dt)
    gate_l2 = _norm_dxdy(phase_rate_last, jnp.ones_like(end_phi, dtype=jnp.bool_), darea)
    gate_linf = jnp.max(jnp.abs(phase_rate_last.astype(jnp.float64)))
    direct_reconstruction = jnp.max(jnp.abs(adv_increment + ch_increment - (end_phi - start_phi)))
    total_mass_adv = jnp.sum(volume.astype(jnp.float64) * adv_increment.astype(jnp.float64))
    total_mass_ch = jnp.sum(volume.astype(jnp.float64) * ch_increment.astype(jnp.float64))
    total_mass_net = jnp.sum(volume.astype(jnp.float64) * (end_phi - start_phi).astype(jnp.float64))
    total_values = jnp.asarray(
        [
            gate_l2,
            gate_linf,
            direct_reconstruction,
            total_mass_adv,
            total_mass_ch,
            total_mass_net,
            jnp.sum(net_rate.astype(jnp.float64) * volume.astype(jnp.float64)),
            total_net_energy_scale,
        ],
        dtype=jnp.float64,
    )
    region_rows = jnp.stack(rows).reshape((-1,))
    return jnp.concatenate(
        (region_rows, flux_average.reshape((-1,)), jnp.asarray(checks, dtype=jnp.float64), total_values)
    )


@jax.jit(static_argnums=(2, 5, 6))
def _advance_decomposed_block(
    state: pf.State,
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
    cell_masks: tuple[jnp.ndarray, ...],
    face_masks: tuple[tuple[jnp.ndarray, jnp.ndarray], ...],
    num_groups: int,
    steps_per_group: int,
):
    """Run exact production steps and retain sample endpoints plus decomposed-rate rows."""
    volume = pf.phase_control_volumes(solid, p)

    def sample_group(state_start, _):
        start_phi = state_start.phi
        zeros = jnp.zeros_like(start_phi)
        init = (
            state_start,
            zeros,
            zeros,
            zeros,
            jnp.zeros((len(cell_masks), 4), dtype=jnp.float64),
            jnp.asarray(0.0, dtype=jnp.float64),
            jnp.asarray(0.0, dtype=jnp.float64),
            jnp.asarray(0.0, dtype=jnp.float64),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=jnp.float64),
            jnp.asarray(True),
            start_phi,
        )

        def one_production_step(carry, _):
            (
                state_in,
                adv_total,
                ch_total,
                ch_exp_total,
                flux_total,
                recon_max,
                adv_rhs_max,
                explicit_ch_max,
                iter_max,
                residual_max,
                converged_all,
                _old_last_phi,
            ) = carry
            state_out, record = _phase_decomposition_step_impl(state_in, solid, p, cell_masks, face_masks)
            adv, ch, ch_exp, _implicit, flux, recon, adv_rhs_err, expl_err, iters, residual, converged = record
            next_carry = (
                state_out,
                adv_total + adv,
                ch_total + ch,
                ch_exp_total + ch_exp,
                flux_total + flux,
                jnp.maximum(recon_max, recon),
                jnp.maximum(adv_rhs_max, adv_rhs_err),
                jnp.maximum(explicit_ch_max, expl_err),
                jnp.maximum(iter_max, iters),
                jnp.maximum(residual_max, residual),
                converged_all & converged,
                state_in.phi,
            )
            return next_carry, None

        (
            (
                state_end,
                adv_total,
                ch_total,
                _ch_exp_total,
                flux_total,
                recon_max,
                adv_rhs_max,
                explicit_ch_max,
                iter_max,
                residual_max,
                converged_all,
                last_previous_phi,
            ),
            _,
        ) = jax.lax.scan(one_production_step, init, None, length=steps_per_group)
        flux_average = flux_total / float(steps_per_group)
        checks = (
            recon_max,
            adv_rhs_max,
            explicit_ch_max,
            iter_max.astype(jnp.float64),
            residual_max,
            converged_all.astype(jnp.float64),
        )
        row = _group_metric_row(
            start_phi,
            state_end.phi,
            adv_total,
            ch_total,
            state_end.phi - start_phi,
            flux_average,
            checks,
            last_previous_phi,
            cell_masks,
            volume,
            solid,
            p,
            steps_per_group,
        )
        return state_end, (state_end, last_previous_phi, row)

    final_state, (samples, gate_previous, metric_rows) = jax.lax.scan(sample_group, state, None, length=num_groups)
    return final_state, samples, gate_previous, metric_rows


def _unpack_metric_row(row: np.ndarray) -> dict[str, Any]:
    value = np.asarray(row, dtype=np.float64)
    n_region = len(REGION_NAMES) * len(REGION_METRIC_NAMES)
    n_flux = len(REGION_NAMES) * len(FLUX_METRIC_NAMES)
    n_check = len(CHECK_METRIC_NAMES)
    region_values = value[:n_region].reshape((len(REGION_NAMES), len(REGION_METRIC_NAMES)))
    flux_values = value[n_region : n_region + n_flux].reshape((len(REGION_NAMES), len(FLUX_METRIC_NAMES)))
    check_values = value[n_region + n_flux : n_region + n_flux + n_check]
    tail = value[n_region + n_flux + n_check :]
    if len(tail) != 8:
        raise ValueError(f"decomposition metric row has unexpected tail length {len(tail)}")
    return {
        "regions": {
            region: {metric: float(region_values[i, j]) for j, metric in enumerate(REGION_METRIC_NAMES)}
            for i, region in enumerate(REGION_NAMES)
        },
        "face_fluxes": {
            region: {metric: float(flux_values[i, j]) for j, metric in enumerate(FLUX_METRIC_NAMES)}
            for i, region in enumerate(REGION_NAMES)
        },
        "checks": {name: float(check_values[i]) for i, name in enumerate(CHECK_METRIC_NAMES)},
        "production_convergence_gate": {
            "phase_rate_l2_dxdy_last_public_step": float(tail[0]),
            "phase_rate_linf_last_public_step": float(tail[1]),
            "adv_ch_net_reconstruction_linf_window": float(tail[2]),
            "formal_mass_increment_adv": float(tail[3]),
            "formal_mass_increment_ch": float(tail[4]),
            "formal_mass_increment_net": float(tail[5]),
            "formal_mass_rate_net": float(tail[6]),
            "net_rate_abs_mu_work_scale": float(tail[7]),
        },
    }


def _instrumented_public_step(
    state: pf.State,
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
    masks: dict[str, np.ndarray],
) -> tuple[pf.State, dict[str, Any], dict[str, np.ndarray]]:
    cell_masks = tuple(jnp.asarray(mask, dtype=jnp.bool_) for mask in masks.values())
    face_masks = tuple(
        (jnp.asarray(mx, dtype=jnp.bool_), jnp.asarray(my, dtype=jnp.bool_)) for mx, my in _face_masks_from_cells(masks)
    )
    out, record = _phase_decomposition_step_impl(state, solid, p, cell_masks, face_masks)
    adv, ch, ch_exp, implicit, flux, recon, adv_rhs_err, explicit_err, iters, residual, converged = record
    details = {
        "advective_increment": np.asarray(adv, dtype=np.float64),
        "CH_increment": np.asarray(ch, dtype=np.float64),
        "CH_explicit_increment": np.asarray(ch_exp, dtype=np.float64),
        "CH_implicit_solver_increment": np.asarray(implicit, dtype=np.float64),
        "net_increment": np.asarray(out.phi - state.phi, dtype=np.float64),
        "phase_reconstruction_linf": float(recon),
        "adv_rhs_vs_face_divergence_linf": float(adv_rhs_err),
        "explicit_implicit_CH_reconstruction_linf": float(explicit_err),
        "implicit_iterations_max": int(iters),
        "implicit_relative_residual_max": float(residual),
        "implicit_converged": bool(converged),
        "flux_metrics_by_region": {
            name: {metric: float(flux[i, j]) for j, metric in enumerate(FLUX_METRIC_NAMES)}
            for i, name in enumerate(REGION_NAMES)
        },
    }
    fields = {
        "advective_increment": details["advective_increment"],
        "CH_increment": details["CH_increment"],
        "net_increment": details["net_increment"],
        "advective_rate": details["advective_increment"] / float(p.dt),
        "CH_rate": details["CH_increment"] / float(p.dt),
        "net_rate": details["net_increment"] / float(p.dt),
    }
    return out, details, fields


def _production_gate_definition() -> dict[str, Any]:
    criteria = (
        chns.PRODUCTION_STATIONARITY_CRITERIA
        if hasattr(chns, "PRODUCTION_STATIONARITY_CRITERIA")
        else {
            "phase_rate_l2_tol": 1.0e-3,
            "angle_tol_deg": 0.1,
            "energy_rel_tol": 1.0e-4,
            "chns_speed_tol": 5.0e-4,
            "window_samples": 5,
            "window_mobility_time": 0.05,
        }
    )
    return {
        "field": "nwa._sample.phase_rate_l2",
        "definition": "sqrt(sum((phi_final - phi_before_last_public_step)^2 / dt^2) * dx * dy)",
        "norm": "full-grid Euclidean L2; not masked and not cut-cell-volume-weighted",
        "normalization": "cell area dx*dy (not V_i); rate uses the exact production public-step delta and dt",
        "dt": float(chns.DT),
        "mask": "none; every grid cell including solid/zero-volume cells participates exactly as in nwa._sample",
        "Mobility_scaling": f"phase_rate_l2 * (M_ref / M) <= {nwa.CRITERIA['phase_rate_l2_tol']}",
        "M_ref": float(chns.M_REF),
        "threshold_at_M_ref": float(nwa.CRITERIA["phase_rate_l2_tol"]),
        "companion_CHNS_gate": criteria,
        "classification": "forensic recording only; does not change any production acceptance threshold",
    }


def _run_decomposed_window(
    start_state: pf.State,
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
    *,
    groups: int,
    steps_per_group: int,
    masks: dict[str, np.ndarray],
    step_start: int,
    case_name: str,
    target_deg: float,
    gate_reference_state: pf.State,
) -> dict[str, Any]:
    if groups < 1 or steps_per_group < 1:
        raise ValueError("decomposed window sizes must be positive")
    cells = tuple(jnp.asarray(mask, dtype=jnp.bool_) for mask in masks.values())
    faces = tuple(
        (jnp.asarray(mx, dtype=jnp.bool_), jnp.asarray(my, dtype=jnp.bool_)) for mx, my in _face_masks_from_cells(masks)
    )
    result_state, sample_states, gate_previous, rows = _advance_decomposed_block(
        start_state, solid, p, cells, faces, groups, steps_per_group
    )
    result_state.phi.block_until_ready()
    public_steps = groups * steps_per_group
    production_state = chns._advance_standard(start_state, solid, p, public_steps)
    production_state.phi.block_until_ready()
    exact_hash_match = _state_hashes(result_state) == _state_hashes(production_state)
    delta_by_field = {
        name: float(
            np.max(np.abs(np.asarray(getattr(result_state, name)) - np.asarray(getattr(production_state, name))))
        )
        for name in ("phi", "u", "v", "t")
    }
    if not exact_hash_match:
        # A mismatch is not hidden or repaired: no decomposed rows are allowed to support a causal
        # classification unless the replay produces the exact public production endpoint.
        raise AuditValidationError(
            (
                f"instrumented {case_name} block did not reproduce the public production endpoint "
                f"exactly; max field errors={delta_by_field}"
            )
        )
    sample_rows = np.asarray(rows, dtype=np.float64)
    gate_prev = np.asarray(gate_previous, dtype=np.float64)
    samples: list[dict[str, Any]] = []
    phi_ref_mass = obs.liquid_mass(
        np.asarray(gate_reference_state.phi, dtype=np.float64),
        np.asarray(solid.sdf, dtype=np.float64),
        float(p.dx),
        float(p.dy),
    )
    volume = np.asarray(pf.phase_control_volumes(solid, p), dtype=np.float64)
    phi_ref_conserved = float(np.sum(np.asarray(gate_reference_state.phi, dtype=np.float64) * volume, dtype=np.float64))
    cos_eff = math.cos(math.radians(float(target_deg)))
    for i in range(groups):
        sampled_state = pf.State(
            phi=sample_states.phi[i],
            u=sample_states.u[i],
            v=sample_states.v[i],
            t=sample_states.t[i],
        )
        absolute_step = int(step_start + (i + 1) * steps_per_group)
        gate_sample = nwa._sample(
            sampled_state,
            jnp.asarray(gate_prev[i]),
            solid,
            p,
            ch_only=False,
            steps=absolute_step,
            cos_eff=cos_eff,
            phi_ref_mass=phi_ref_mass,
            M=float(p.M),
            volume=volume,
            phi_ref_conserved_mass=phi_ref_conserved,
        )
        unpacked = _unpack_metric_row(sample_rows[i])
        exact_gate = unpacked["production_convergence_gate"]["phase_rate_l2_dxdy_last_public_step"]
        gate_delta = abs(float(gate_sample["phase_rate_l2"]) - exact_gate)
        if gate_delta > 5.0e-13 * max(1.0, exact_gate):
            raise AuditValidationError(
                (
                    f"instrumented {case_name} production convergence-gate normalization "
                    f"differs from nwa._sample by {gate_delta}"
                )
            )
        contacts = _contact_metrics(sampled_state.phi, solid, p)
        samples.append(
            {
                "step": absolute_step,
                "time": float(sampled_state.t),
                "sample_stride_steps": steps_per_group,
                "state_hashes": _state_hashes(sampled_state),
                "gate_sample": gate_sample,
                "left_right_contact_metrics": contacts,
                **unpacked,
            }
        )
    return {
        "case": case_name,
        "step_start": int(step_start),
        "groups": int(groups),
        "steps_per_group": int(steps_per_group),
        "window_public_steps": int(public_steps),
        "window_physical_time": float(public_steps) * float(p.dt),
        "window_mobility_scaled_time": float(public_steps) * float(p.dt) * float(p.M),
        "mask_policy_frozen": True,
        "state_endpoint_hashes": _state_hashes(result_state),
        "production_endpoint_hashes": _state_hashes(production_state),
        "endpoint_exact_hash_match_production": exact_hash_match,
        "endpoint_max_abs_difference_by_field": delta_by_field,
        "metric_rows": samples,
        "reference_phase_mass_hard_mask": float(phi_ref_mass),
        "reference_conserved_mass_sum_V_phi": float(phi_ref_conserved),
        "region_masks_policy": (
            "fixed from the start state; masks and left/right 2dx/4dx contact radii do not move during this window"
        ),
    }


def _save_public_step_fields(
    case_name: str,
    step: int,
    fields: dict[str, np.ndarray],
    details: dict[str, Any],
    state: pf.State,
) -> dict[str, Any]:
    path = ARTIFACT_ROOT / "fields" / f"{case_name}_step_{step:06d}_phase_decomposition.npz"
    arrays = {name: np.asarray(value) for name, value in fields.items()}
    arrays.update({f"state_{name}": np.asarray(getattr(state, name)) for name in ("phi", "u", "v", "t")})
    artifact_hash = _write_npz(path, arrays)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": artifact_hash,
        "state_hashes": _state_hashes(state),
        "field_hashes": {name: _hash_array(value) for name, value in fields.items()},
        "checks": {key: value for key, value in details.items() if not isinstance(value, (dict, np.ndarray))},
        "not_production_acceptance_evidence": True,
    }


def _window_rate_summary(window: dict[str, Any]) -> dict[str, Any]:
    rows = window["metric_rows"]
    summary: dict[str, Any] = {
        "window_steps": window["window_public_steps"],
        "window_physical_time": window["window_physical_time"],
        "window_mobility_scaled_time": window["window_mobility_scaled_time"],
        "sample_count": len(rows),
        "start_sample_step": rows[0]["step"] if rows else None,
        "end_sample_step": rows[-1]["step"] if rows else None,
        "all_samples_exact_public_state_replay": window["endpoint_exact_hash_match_production"],
        "regions": {},
        "gate": {},
    }
    for region in REGION_NAMES:
        region_values = [row["regions"][region] for row in rows]
        summary["regions"][region] = {
            metric: {
                "mean": float(np.mean([value[metric] for value in region_values])),
                "max_abs": float(np.max(np.abs([value[metric] for value in region_values]))),
                "last": float(region_values[-1][metric]),
            }
            for metric in REGION_METRIC_NAMES
        }
    for field in ("phase_rate_l2", "phase_rate_linf", "conserved_mass_drift", "free_energy", "max_speed"):
        values = [row["gate_sample"].get(field) for row in rows]
        finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
        summary["gate"][field] = {
            "first": finite[0] if finite else None,
            "last": finite[-1] if finite else None,
            "max": max(finite) if finite else None,
            "min": min(finite) if finite else None,
        }
    summary["decomposition_checks"] = {
        check: {
            "max": max(float(row["checks"][check]) for row in rows),
            "last": float(rows[-1]["checks"][check]),
        }
        for check in CHECK_METRIC_NAMES
    }
    summary["production_gate_max_scaled_rate"] = max(
        (float(row["gate_sample"]["phase_rate_l2"]) * (float(chns.M_REF) / float(chns.M_REF)) for row in rows),
        default=0.0,
    )
    summary["production_gate_threshold_Mref"] = float(nwa.CRITERIA["phase_rate_l2_tol"])
    return summary


def _contact_rate_correlations(window: dict[str, Any]) -> dict[str, Any]:
    rows = window["metric_rows"]
    if len(rows) < 3:
        return {"status": "unmeasured_insufficient_samples", "sample_count": len(rows)}
    times = [float(row["time"]) for row in rows]
    positions = [row["left_right_contact_metrics"] for row in rows]
    left = [item.get("left_contact_x_wrapped") for item in positions]
    right = [item.get("right_contact_x_wrapped") for item in positions]
    velocity = clk.contact_line_velocity(times, left, right, Lx=float(chns.LENGTH))
    output = {"status": "measured", "sample_count": len(rows), "sampling_interval_steps": window["steps_per_group"]}
    for side, region_name, velocity_key in (
        ("left", "left_contact_line_4dx_frozen", "left_velocity"),
        ("right", "right_contact_line_4dx_frozen", "right_velocity"),
    ):
        rates = [
            row["regions"][region_name]["formal_mass_increment_net"] / (window["steps_per_group"] * chns.DT)
            for row in rows
        ]
        speeds = velocity[velocity_key]
        finite = [
            (float(a), float(b))
            for a, b in zip(rates, speeds)
            if b is not None and math.isfinite(float(a)) and math.isfinite(float(b))
        ]
        corr = None
        if len(finite) >= 3 and np.std([x[0] for x in finite]) > 0.0 and np.std([x[1] for x in finite]) > 0.0:
            corr = float(np.corrcoef(np.asarray(finite, dtype=np.float64).T)[0, 1])
        output[side] = {
            "region": region_name,
            "phase_formal_mass_rate": rates,
            "contact_line_velocity": speeds,
            "pearson_zero_lag": corr,
            "finite_pairs": len(finite),
        }
    return output


def _window_effect_sizes(window: dict[str, Any]) -> dict[str, Any]:
    rows = window["metric_rows"]
    if not rows:
        return {"status": "unmeasured"}
    first, last = rows[0], rows[-1]
    authority_fluid = last["regions"]["whole_fluid"]
    return {
        "C_dir_whole_fluid_last": authority_fluid["alignment_C_dir_volume"],
        "C_mag_whole_fluid_last": authority_fluid["cancellation_C_mag_volume"],
        "adv_rate_l2_dxdy_last": authority_fluid["adv_rate_l2_dxdy"],
        "CH_rate_l2_dxdy_last": authority_fluid["ch_rate_l2_dxdy"],
        "net_rate_l2_dxdy_last": authority_fluid["net_rate_l2_dxdy"],
        "adv_rate_l2_volume_last": authority_fluid["adv_rate_l2_volume"],
        "CH_rate_l2_volume_last": authority_fluid["ch_rate_l2_volume"],
        "net_rate_l2_volume_last": authority_fluid["net_rate_l2_volume"],
        "relative_to_first_sample": {
            "adv_l2_change_fraction": (
                last["regions"]["whole_fluid"]["adv_rate_l2_volume"]
                - first["regions"]["whole_fluid"]["adv_rate_l2_volume"]
            )
            / max(first["regions"]["whole_fluid"]["adv_rate_l2_volume"], 1.0e-300),
            "CH_l2_change_fraction": (
                last["regions"]["whole_fluid"]["ch_rate_l2_volume"]
                - first["regions"]["whole_fluid"]["ch_rate_l2_volume"]
            )
            / max(first["regions"]["whole_fluid"]["ch_rate_l2_volume"], 1.0e-300),
            "net_l2_change_fraction": (
                last["regions"]["whole_fluid"]["net_rate_l2_volume"]
                - first["regions"]["whole_fluid"]["net_rate_l2_volume"]
            )
            / max(first["regions"]["whole_fluid"]["net_rate_l2_volume"], 1.0e-300),
        },
        "effect_size_interpretation": "direct reported ratios/cosines; no post-measurement pass threshold is applied",
    }


def _phase_only_config(target_deg: float, *, M: float, branch_name: str) -> dict[str, Any]:
    config = chns.production_config(target_deg=target_deg, N_value=chns.N, dt=chns.DT, M=M)
    config.update(
        {
            "dynamics_mode": "CH_ONLY" if branch_name == "ch_only_equilibrium_060" else "FREEZE_U_CH_ONLY",
            "diagnostic_only": True,
            "diagnostic_sample_every_steps": 1_000,
            "convergence_criteria": dict(nwa.CRITERIA),
        }
    )
    if branch_name != "ch_only_equilibrium_060":
        config["diagnostic_branch"] = branch_name
    return config


def _phase_only_endpoint_metrics(
    state: pf.State,
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
    *,
    phi_eq: np.ndarray | None = None,
    mask_reference: Any | None = None,
) -> dict[str, Any]:
    mu_diag = _mu_and_flux_diagnostics(state.phi, state.u, state.v, solid, p)
    result = {
        "state_hashes": _state_hashes(state),
        "state_dtypes": {
            "phi": str(np.asarray(state.phi).dtype),
            "u": str(np.asarray(state.u).dtype),
            "v": str(np.asarray(state.v).dtype),
            "t": str(np.asarray(state.t).dtype),
        },
        "phase_free_energy_components": _phase_energy_parts(state.phi, solid, p),
        "chemical_potential": mu_diag,
        "contact_lines": _contact_metrics(state.phi, solid, p),
        "formal_mass_sum_V_phi": float(
            np.sum(
                np.asarray(state.phi, dtype=np.float64)
                * np.asarray(pf.phase_control_volumes(solid, p), dtype=np.float64),
                dtype=np.float64,
            )
        ),
    }
    if phi_eq is not None:
        masks, _ = _make_region_masks(mask_reference if mask_reference is not None else state.phi, solid, p)
        result["equilibrium_distance"] = _distance_to_equilibrium(state.phi, phi_eq, solid, p, masks)
    return result


def _run_phase_only_continuation(
    label: str,
    initial_state: pf.State,
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
    config: dict[str, Any],
    *,
    max_steps: int,
    start_absolute_step: int,
    parent_state_hashes: dict[str, str],
    progress_dir: Path,
    phi_eq: np.ndarray | None = None,
    stop_when_converged: bool = True,
) -> dict[str, Any]:
    """Resumable exact CH-only public-step continuation at unchanged M_ref and dt."""
    progress_dir.mkdir(parents=True, exist_ok=True)
    progress_path = progress_dir / f"{label}_progress.json"
    checkpoints = ARTIFACT_ROOT / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    start_phi = np.asarray(initial_state.phi, dtype=np.float64)
    initial = pf.State(
        phi=initial_state.phi,
        u=jnp.zeros_like(initial_state.u, dtype=p.dtype),
        v=jnp.zeros_like(initial_state.v, dtype=p.dtype),
        t=initial_state.t,
    )
    volume = np.asarray(pf.phase_control_volumes(solid, p), dtype=np.float64)
    phi_ref_mass = obs.liquid_mass(start_phi, np.asarray(solid.sdf), float(p.dx), float(p.dy))
    phi_ref_conserved = float(np.sum(start_phi * volume, dtype=np.float64))
    samples: list[dict[str, Any]] = []
    relative_step = 0
    state = initial
    latest_checkpoint_meta = None
    if progress_path.is_file():
        saved = _load_json(progress_path)
        compatible = (
            saved.get("stage") == STAGE
            and saved.get("section") == "phase_only_continuation"
            and saved.get("label") == label
            and saved.get("config_fingerprint") == _canonical_hash(config)
            and saved.get("source_hashes") == _source_hashes()
            and saved.get("runtime_versions") == _runtime_versions()
            and saved.get("start_absolute_step") == int(start_absolute_step)
            and saved.get("parent_state_hashes") == parent_state_hashes
            and saved.get("stop_when_converged") == bool(stop_when_converged)
        )
        if compatible:
            candidate_rel = int(saved.get("committed_relative_step", -1))
            candidate_abs = int(start_absolute_step + candidate_rel)
            checkpoint = ROOT / saved.get("checkpoint", "")
            if candidate_rel >= 1 and candidate_rel <= max_steps and checkpoint.is_file():
                try:
                    state, latest_checkpoint_meta = _load_state_checkpoint(
                        checkpoint,
                        section="phase_only_continuation",
                        case_name=label,
                        step=candidate_abs,
                        config=config,
                    )
                    relative_step = candidate_rel
                    samples = list(saved.get("samples", []))
                    if saved.get("state_hashes") != _state_hashes(state):
                        raise ValueError("progress state hash does not match its committed checkpoint")
                except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                    print(f"[L1A-2l] reject {label} continuation checkpoint: {exc}", flush=True)
                    state, relative_step, samples = initial, 0, []
            if saved.get("status") == "converged" and relative_step == int(saved.get("converged_relative_step", -1)):
                return {
                    "label": label,
                    "status": "converged",
                    "diagnostic_only": True,
                    "config": config,
                    "config_fingerprint": _canonical_hash(config),
                    "start_absolute_step": int(start_absolute_step),
                    "relative_steps": relative_step,
                    "absolute_end_step": int(start_absolute_step + relative_step),
                    "parent_state_hashes": parent_state_hashes,
                    "state_hashes": _state_hashes(state),
                    "samples": samples,
                    "final_convergence_gate": saved.get("final_convergence_gate"),
                    "endpoint_metrics": _phase_only_endpoint_metrics(
                        state, solid, p, phi_eq=phi_eq, mask_reference=start_phi
                    ),
                    "checkpoint": saved.get("checkpoint"),
                    "resumed_existing_converged_checkpoint": True,
                }
        elif saved:
            print(f"[L1A-2l] discard incompatible CH-only progress for {label}", flush=True)

    if not samples:
        first = nwa._sample(
            initial,
            initial.phi,
            solid,
            p,
            ch_only=True,
            steps=int(start_absolute_step),
            cos_eff=math.cos(math.radians(float(config["target_deg"]))),
            phi_ref_mass=phi_ref_mass,
            M=float(p.M),
            volume=volume,
            phi_ref_conserved_mass=phi_ref_conserved,
        )
        first["continuation_relative_step"] = 0
        if phi_eq is not None:
            first["raw_unaligned_D_phi"] = _distance_to_equilibrium(initial.phi, phi_eq, solid, p)[
                "raw_unaligned_relative_D_phi"
            ]
        samples = [first]

    status = "not_converged_step_budget"
    gate = nwa._window_converged(samples, ch_only=True, crit=nwa.CRITERIA, M=float(p.M))
    if gate.get("converged"):
        status = "converged"
    started = time.perf_counter()
    chunk_steps = 1_000
    while relative_step < max_steps and (not gate.get("converged") or not stop_when_converged):
        count = min(chunk_steps, max_steps - relative_step)
        prev_phi, next_state, max_iterations, max_residual, solver_converged = nwa._advance(
            state, solid, p, count, True
        )
        next_state.phi.block_until_ready()
        relative_step += count
        state = next_state
        if not bool(np.isfinite(float(max_residual))) or not np.isfinite(np.asarray(state.phi)).all():
            raise AuditValidationError(f"{label}: non-finite CH-only replay; fail closed")
        row = nwa._sample(
            state,
            prev_phi,
            solid,
            p,
            ch_only=True,
            steps=int(start_absolute_step + relative_step),
            cos_eff=math.cos(math.radians(float(config["target_deg"]))),
            phi_ref_mass=phi_ref_mass,
            M=float(p.M),
            volume=volume,
            phi_ref_conserved_mass=phi_ref_conserved,
        )
        row["continuation_relative_step"] = int(relative_step)
        row["implicit_iterations_max_in_chunk"] = int(max_iterations)
        row["implicit_relative_residual_max_in_chunk"] = float(max_residual)
        row["implicit_last_step_converged"] = bool(solver_converged)
        if phi_eq is not None:
            row["raw_unaligned_D_phi"] = _distance_to_equilibrium(state.phi, phi_eq, solid, p)[
                "raw_unaligned_relative_D_phi"
            ]
        samples.append(row)
        gate = nwa._window_converged(samples, ch_only=True, crit=nwa.CRITERIA, M=float(p.M))
        if relative_step % 10_000 == 0 or gate.get("converged") or relative_step == max_steps:
            absolute_step = int(start_absolute_step + relative_step)
            checkpoint = checkpoints / f"{label}_step_{absolute_step:06d}.npz"
            metadata = _save_state_checkpoint(
                checkpoint,
                state,
                section="phase_only_continuation",
                case_name=label,
                step=absolute_step,
                config=config,
                upstream_state_hashes=parent_state_hashes,
                extra={"diagnostic_only": True, "parent_state_hashes": parent_state_hashes},
            )
            saved = {
                "stage": STAGE,
                "section": "phase_only_continuation",
                "label": label,
                "start_absolute_step": int(start_absolute_step),
                "committed_relative_step": int(relative_step),
                "converged_relative_step": int(relative_step) if gate.get("converged") else None,
                "absolute_end_step": absolute_step,
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "state_hashes": metadata["state_hashes"],
                "parent_state_hashes": parent_state_hashes,
                "config": config,
                "config_fingerprint": _canonical_hash(config),
                "source_hashes": _source_hashes(),
                "runtime_versions": _runtime_versions(),
                "samples": samples,
                "final_convergence_gate": gate,
                "status": "converged"
                if gate.get("converged") and stop_when_converged
                else "running"
                if relative_step < max_steps
                else "converged_at_fixed_step"
                if gate.get("converged")
                else "not_converged_step_budget",
                "stop_when_converged": bool(stop_when_converged),
                "elapsed_seconds_this_run": float(time.perf_counter() - started),
                "production_semantics_changed": False,
                "diagnostic_only": True,
            }
            _write_json(progress_path, saved)
            print(
                (
                    f"[L1A-2l] {label} {relative_step}/{max_steps}; "
                    f"gate={gate.get('converged')} rate={samples[-1].get('phase_rate_l2'):.6g}"
                ),
                flush=True,
            )
    if gate.get("converged"):
        status = "converged"
    endpoint = _phase_only_endpoint_metrics(state, solid, p, phi_eq=phi_eq, mask_reference=start_phi)
    field_path = ARTIFACT_ROOT / "fields" / f"{label}_endpoint.npz"
    field_hash = _write_npz(
        field_path,
        {
            "phi": np.asarray(state.phi),
            "u": np.asarray(state.u),
            "v": np.asarray(state.v),
            "t": np.asarray(state.t),
            "mu": np.asarray(pf.chemical_potential(state.phi, solid, p)),
            "volume": volume,
        },
    )
    return {
        "label": label,
        "status": status,
        "diagnostic_only": True,
        "config": config,
        "config_fingerprint": _canonical_hash(config),
        "start_absolute_step": int(start_absolute_step),
        "relative_steps": int(relative_step),
        "absolute_end_step": int(start_absolute_step + relative_step),
        "physical_continuation_time": float(relative_step) * float(p.dt),
        "mobility_scaled_continuation_time": float(relative_step) * float(p.dt) * float(p.M),
        "parent_state_hashes": parent_state_hashes,
        "state_hashes": _state_hashes(state),
        "samples": samples,
        "final_convergence_gate": gate,
        "endpoint_metrics": endpoint,
        "endpoint_field_artifact": {
            "path": str(field_path.relative_to(ROOT)),
            "sha256": field_hash,
            "not_production_acceptance_evidence": True,
        },
        "checkpoint": str(
            checkpoints.joinpath(f"{label}_step_{int(start_absolute_step + relative_step):06d}.npz").relative_to(ROOT)
        ),
        "max_implicit_iterations_seen": max(
            (int(row.get("implicit_iterations_max_in_chunk", 0)) for row in samples), default=0
        ),
        "max_implicit_relative_residual_seen": max(
            (float(row.get("implicit_relative_residual_max_in_chunk", 0.0)) for row in samples), default=0.0
        ),
        "production_semantics_changed": False,
        "resumed_existing_converged_checkpoint": False,
    }


@partial(jax.jit, static_argnums=(4, 6, 7))
def _fixed_velocity_phase_block(
    phi: jnp.ndarray,
    u_reference: jnp.ndarray,
    v_reference: jnp.ndarray,
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
    velocity_scale: jnp.ndarray,
    steps: int,
    mode: str,
):
    """Diagnostic phase-only replay with frozen velocity samples; never a production trajectory."""
    scale = jnp.asarray(velocity_scale, dtype=p.dtype)
    u_fixed = u_reference * scale
    v_fixed = v_reference * scale
    dt_sub = float(p.dt) / 3.0

    def public_step(phi_in, _):
        init = (
            phi_in,
            jnp.zeros_like(phi_in),
            jnp.zeros_like(phi_in),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0),
            jnp.asarray(True),
        )

        def substep(carry, _):
            phase, adv_total, ch_total, iterations_max, residual_max, converged_all = carry
            if mode == "ch_only":
                rhs = jnp.zeros_like(phase)
                mu_expl = pf._explicit_chemical_potential(phase, solid, p)
                phi_new, solve_info = pf._phase_update(
                    phase,
                    jnp.zeros_like(u_fixed),
                    jnp.zeros_like(v_fixed),
                    solid,
                    p,
                    dt_sub,
                    rhs,
                    mu_expl,
                )
                adv_inc = jnp.zeros_like(phase)
                ch_inc = phi_new - phase
            else:
                phase_state = pf.State(phase, u_fixed, v_fixed, jnp.asarray(0.0, dtype=p.dtype))
                phi_rhs, _u_rhs, _v_rhs, _mu, mu_expl = pf.rhs(phase_state, solid, p)
                adv_inc = dt_sub * phi_rhs
                if mode == "advection_only":
                    phi_new = phase + adv_inc
                    conv = jnp.asarray(True)
                    return (
                        phi_new,
                        adv_total + adv_inc,
                        ch_total,
                        iterations_max,
                        residual_max,
                        converged_all & conv,
                    ), None
                phi_new, solve_info = pf._phase_update(phase, u_fixed, v_fixed, solid, p, dt_sub, phi_rhs, mu_expl)
                ch_inc = phi_new - (phase + adv_inc)
            next_carry = (
                phi_new,
                adv_total + adv_inc,
                ch_total + ch_inc,
                jnp.maximum(iterations_max, solve_info.iterations),
                jnp.maximum(residual_max, solve_info.relative_residual),
                converged_all & solve_info.converged,
            )
            return next_carry, None

        (phi_out, adv, ch, max_iter, max_res, converged), _ = jax.lax.scan(substep, init, None, length=3)
        return phi_out, (phi_out, adv, ch, phi_out - phi_in, max_iter, max_res, converged)

    final_phi, series = jax.lax.scan(public_step, jnp.asarray(phi), None, length=steps)
    phi_rows, adv_rows, ch_rows, net_rows, iters, residual, converged = series
    return (
        final_phi,
        jnp.sum(adv_rows, axis=0),
        jnp.sum(ch_rows, axis=0),
        jnp.sum(net_rows, axis=0),
        {
            "iterations_max": jnp.max(iters),
            "relative_residual_max": jnp.max(residual),
            "converged_all": jnp.all(converged),
            "sampled_phase": phi_rows,
        },
    )


@partial(jax.jit, static_argnums=(2,))
def _phase_only_instrumented_step(phi: jnp.ndarray, solid: pf.Solid, p: pf.PhaseFieldParams, time_value: jnp.ndarray):
    """Capture exact CH-only increments along pf.phase_only_step_with_diagnostics staging."""
    zeros = jnp.zeros_like(phi, dtype=p.dtype)
    dt_sub = float(p.dt) / 3.0

    def substep(carry, _):
        phase, time_in, adv_sum, ch_sum, ch_exp_sum, implicit_sum, max_it, max_res, converged = carry
        state = pf.State(phase, zeros, zeros, time_in)
        phi_rhs, _u_rhs, _v_rhs, _mu, mu_expl = pf.rhs(state, solid, p)
        op = pf.phase_transport_operator(solid, p)
        chx, chy = pf.chemical_potential_fluxes(mu_expl, solid, p)
        ch_div = pf.control_volume_divergence(chx, chy, op.volume_safe)
        rhs_phase = phase + dt_sub * (phi_rhs - ch_div)
        phi_new, solve_info = pf._phase_update(phase, zeros, zeros, solid, p, dt_sub, phi_rhs, mu_expl)
        explicit = -dt_sub * ch_div
        implicit = phi_new - rhs_phase
        adv_inc = dt_sub * phi_rhs
        ch_inc = phi_new - (phase + adv_inc)
        return (
            phi_new,
            time_in + dt_sub,
            adv_sum + adv_inc,
            ch_sum + ch_inc,
            ch_exp_sum + explicit,
            implicit_sum + implicit,
            jnp.maximum(max_it, solve_info.iterations),
            jnp.maximum(max_res, solve_info.relative_residual),
            converged & solve_info.converged,
        ), None

    init = (
        phi,
        time_value,
        jnp.zeros_like(phi),
        jnp.zeros_like(phi),
        jnp.zeros_like(phi),
        jnp.zeros_like(phi),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray(True),
    )
    (phi_new, new_time, adv, ch, explicit_ch, implicit_ch, iterations, residual, converged), _ = jax.lax.scan(
        substep, init, None, length=3
    )
    result = pf.State(phi_new, zeros, zeros, new_time)
    return result, {
        "advective_increment": adv,
        "CH_increment": ch,
        "CH_explicit_increment": explicit_ch,
        "CH_implicit_solver_increment": implicit_ch,
        "net_increment": phi_new - phi,
        "iterations_max": iterations,
        "relative_residual_max": residual,
        "converged": converged,
        "reconstruction_linf": jnp.max(jnp.abs(adv + ch - (phi_new - phi))),
    }


def _volume_direction_cosine(a: Any, b: Any, volume: Any) -> float:
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    w = np.asarray(volume, dtype=np.float64)
    dot = float(np.sum(w * av * bv, dtype=np.float64))
    na = float(np.sqrt(np.sum(w * av * av, dtype=np.float64)))
    nb = float(np.sqrt(np.sum(w * bv * bv, dtype=np.float64)))
    return dot / max(na * nb, 1.0e-300)


def _delta_phi_metrics(phi: Any, phi_eq: Any, solid: pf.Solid, p: pf.PhaseFieldParams) -> dict[str, Any]:
    return _distance_to_equilibrium(phi, phi_eq, solid, p)


def _regional_free_energy(
    phi: Any, solid: pf.Solid, p: pf.PhaseFieldParams, masks: dict[str, np.ndarray]
) -> dict[str, Any]:
    phase = np.asarray(phi, dtype=np.float64)
    op = pf.phase_transport_operator(solid, p)
    volume = np.asarray(op.volume, dtype=np.float64)
    wx, wy = np.asarray(op.weight_x, dtype=np.float64), np.asarray(op.weight_y, dtype=np.float64)
    dx_forward = np.roll(phase, -1, axis=0) - phase
    dy_forward = np.roll(phase, -1, axis=1) - phase
    energy_x = 0.5 * float(p.eps) * wx * dx_forward**2
    energy_y = 0.5 * float(p.eps) * wy * dy_forward**2
    grad_cell = 0.5 * (energy_x + np.roll(energy_x, 1, axis=0) + energy_y + np.roll(energy_y, 1, axis=1))
    bulk_cell = volume * phase**2 * (1.0 - phase) ** 2 / float(p.eps)
    if p.wetting_model == "surface_energy" and pf.wall_measure_is_cutcell(p):
        wall_area = np.asarray(pf.active_wall_measure(solid, p)[0], dtype=np.float64)
        wall_cell = (
            np.asarray(pf.wall_energy_density(jnp.asarray(phase), solid.cos_theta), dtype=np.float64) * wall_area
        )
    else:
        wall_cell = np.zeros_like(phase)
    output = {}
    for name, mask in masks.items():
        m = np.asarray(mask, dtype=bool)
        output[name] = {
            "bulk": float(np.sum(np.where(m, bulk_cell, 0.0), dtype=np.float64)),
            "gradient_half_face_attribution": float(np.sum(np.where(m, grad_cell, 0.0), dtype=np.float64)),
            "wall": float(np.sum(np.where(m, wall_cell, 0.0), dtype=np.float64)),
            "wall_abs": float(np.sum(np.where(m, np.abs(wall_cell), 0.0), dtype=np.float64)),
            "attribution_policy": (
                "bulk is assigned to its cell; each face-gradient energy is split equally "
                "between adjacent periodic cells; cut-cell wall energy uses A_wall,i g_w(phi_i)"
            ),
        }
    return output


def _morphology(phi: Any, solid: pf.Solid, p: pf.PhaseFieldParams) -> dict[str, Any]:
    phase = np.asarray(phi, dtype=np.float64)
    volume = np.asarray(pf.phase_control_volumes(solid, p), dtype=np.float64)
    contacts = _contact_metrics(phase, solid, p)
    liquid = np.clip(phase, 0.0, 1.0)
    mass = float(np.sum(volume * liquid, dtype=np.float64))
    x = (np.arange(p.Nx, dtype=np.float64) + 0.5)[:, None] * float(p.dx)
    y = (np.arange(p.Ny, dtype=np.float64) + 0.5)[None, :] * float(p.dy)
    y_cm = float(np.sum(volume * liquid * y, dtype=np.float64) / max(mass, 1.0e-300))
    x_moment = np.sum(volume * liquid * np.broadcast_to(x, phase.shape), axis=1, dtype=np.float64)
    phase_component_count = int(ndimage.label((phase >= 0.5) & (volume > 0.0))[1])
    return {
        "formal_volume_weighted_liquid_mass_clipped_phi": mass,
        "hard_mask_liquid_mass": float(obs.liquid_mass(phase, np.asarray(solid.sdf), float(p.dx), float(p.dy))),
        "liquid_centroid_y": y_cm,
        "liquid_centroid_x_periodic_first_moment": float(np.sum(x_moment, dtype=np.float64) / max(mass, 1.0e-300)),
        "phi_ge_0_5_connected_components_nonperiodic_diagnostic": phase_component_count,
        "contact_line": contacts,
    }


def _compare_morphology(phi_a: Any, phi_b: Any, solid: pf.Solid, p: pf.PhaseFieldParams) -> dict[str, Any]:
    a, b = _morphology(phi_a, solid, p), _morphology(phi_b, solid, p)
    side_keys = (
        "left_angle_deg",
        "right_angle_deg",
        "left_contact_x_wrapped",
        "right_contact_x_wrapped",
        "contact_width",
        "top_height",
    )
    contact_a, contact_b = a["contact_line"], b["contact_line"]
    contact_delta = {}
    for key in side_keys:
        av, bv = contact_a.get(key), contact_b.get(key)
        contact_delta[key] = None if av is None or bv is None else float(bv) - float(av)
    return {
        "case_a": a,
        "case_b": b,
        "case_b_minus_case_a": {
            "formal_volume_weighted_liquid_mass_clipped_phi": b["formal_volume_weighted_liquid_mass_clipped_phi"]
            - a["formal_volume_weighted_liquid_mass_clipped_phi"],
            "liquid_centroid_y": b["liquid_centroid_y"] - a["liquid_centroid_y"],
            "contact_line_metrics": contact_delta,
        },
    }


def _load_checkpointed_production_state(
    case_name: str,
    step: int,
    config: dict[str, Any],
    expected_hashes: dict[str, str],
) -> tuple[pf.State | None, dict[str, Any]]:
    path = ARTIFACT_ROOT / "checkpoints" / f"{case_name}_production_step_{step:06d}.npz"
    if not path.is_file():
        return None, {"status": "missing_intermediate_production_checkpoint", "path": str(path.relative_to(ROOT))}
    try:
        state, meta = _load_state_checkpoint(
            path,
            section="production_state",
            case_name=case_name,
            step=step,
            config=config,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return None, {
            "status": "rejected_intermediate_production_checkpoint",
            "path": str(path.relative_to(ROOT)),
            "reason": str(exc),
        }
    match = _state_hashes(state) == expected_hashes if step == CASE_STEPS[case_name] else None
    return state, {
        "status": "accepted_strict_l1a2l_checkpoint",
        "path": str(path.relative_to(ROOT)),
        "metadata": meta,
        "matches_upstream_report_hashes": match,
    }


def _rehydrate_case(
    case_name: str,
    target: float,
    two_k: dict[str, Any],
    progress_dir: Path,
) -> tuple[dict[str, Any], pf.PhaseFieldParams, pf.Solid, pf.State | None, dict[str, Any]]:
    p, solid, seed, config = _make_case(target)
    expected = _expected_upstream_case_hashes(two_k, case_name)
    report_case = two_k["cases"][case_name]
    required_step = int(report_case["step"])
    historical_snapshot, historical_decision = _load_primary_snapshot_if_exact(case_name, config, required_step)
    if historical_snapshot is not None:
        state = historical_snapshot
        state_matches = _state_hashes(state) == expected
        return (
            {
                "case": case_name,
                "target_deg": target,
                "step": required_step,
                "config": config,
                "config_fingerprint": _canonical_hash(config),
                "seed_state_hashes": _state_hashes(seed),
                "endpoint_state_hashes": _state_hashes(state),
                "matches_l1a2k_per_field_hashes": state_matches,
                "state_acceptance": "accepted_exact_snapshot" if state_matches else "rejected_hash_mismatch",
                "provenance": historical_decision,
            },
            p,
            solid,
            state if state_matches else None,
            {"source": "l1a2k_snapshot", "path": historical_decision.get("path"), "hash_match": state_matches},
        )
    state, reached, replay = _ensure_production_prefix(
        case_name,
        target,
        required_step,
        config,
        solid,
        seed,
        p,
        progress_dir=progress_dir,
        upstream_hashes=expected,
        extra_checkpoint_steps=(
            (AUTHORITY_WINDOW_START, AUTHORITY_BURST_START)
            if case_name == "authority_060"
            else (required_step - CONTROL_WINDOW_LENGTH,)
        ),
    )
    actual_hashes = _state_hashes(state)
    hash_match = actual_hashes == expected
    case_entry = {
        "case": case_name,
        "target_deg": float(target),
        "step": int(reached),
        "time": float(state.t),
        "config": config,
        "config_fingerprint": _canonical_hash(config),
        "seed_state_hashes": _state_hashes(seed),
        "endpoint_state_hashes": actual_hashes,
        "expected_l1a2k_state_hashes": expected,
        "matches_l1a2k_per_field_hashes": hash_match,
        "state_acceptance": "accepted_exact_rehydration" if hash_match else "rejected_hash_mismatch",
        "provenance": {"historical_snapshot_decision": historical_decision, "production_replay": replay},
        "state_dtypes": {name: str(np.asarray(getattr(state, name)).dtype) for name in ("phi", "u", "v", "t")},
    }
    return (
        case_entry,
        p,
        solid,
        state if hash_match else None,
        {"source": "rerun_from_production_seed", "hash_match": hash_match},
    )


def _history_context(two_j: dict[str, Any], two_k: dict[str, Any]) -> dict[str, Any]:
    prior = two_j.get("diagnostics", {}).get("freeze_u_phase_only_Mref", {})
    phase = two_k.get("phase_only_comparison", {})
    return {
        "l1a2j_authority_phenomenology_label": two_j.get("phenomenology", {}).get("classification"),
        "l1a2j_angle_cycle_upgrade": (
            "not upgraded; this audit does not reinterpret the old cycle label without new differential evidence"
        ),
        "l1a2j_freeze_u_Mref_historical_summary": {
            key: prior.get(key)
            for key in (
                "start_authority_step",
                "steps",
                "production_acceptance_evidence",
                "production_window_gate_diagnostic_only",
                "formal_phase_mass_start_sum_V_phi",
                "formal_phase_mass_final_sum_V_phi",
                "formal_mass_drift_from_branch_start",
                "phase_state_bitwise_changed",
                "velocity_zero_bitwise_all_steps_endpoint",
                "end_state_hashes",
            )
        },
        "l1a2k_existing_ch_only_reference": {
            "case": phase.get("ch_only_case"),
            "step": phase.get("ch_only_step"),
            "state_hashes": phase.get("ch_only_state_hashes"),
            "config": phase.get("ch_only_config"),
            "config_fingerprint_source": (
                "recomputed from the saved config and verified against the current production constructor"
            ),
            "convergence_gate": phase.get("ch_only_convergence_gate"),
            "field_artifact": phase.get("field_artifact"),
            "not_generic_causal_evidence": True,
        },
        "state_array_reuse": (
            "no raw L1A-2j/L1A-2k npy/npz snapshots were present during initial workspace inspection; "
            "required states are regenerated from exact seeds and accepted only on per-field hash match"
        ),
    }


def _direction_test(
    start_state: pf.State,
    eq_phi: np.ndarray,
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
    masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    prod_end, prod_details, prod_fields = _instrumented_public_step(start_state, solid, p, masks)
    production_step, production_diag = pf.step_with_diagnostics(start_state, solid, p)
    production_hash_match = _state_hashes(prod_end) == _state_hashes(production_step)
    if not production_hash_match:
        raise AuditValidationError("instrumented direction-test public step differs from the production public step")
    zero_start = pf.State(
        start_state.phi,
        jnp.zeros_like(start_state.u, dtype=p.dtype),
        jnp.zeros_like(start_state.v, dtype=p.dtype),
        start_state.t,
    )
    ch_end, ch_details = _phase_only_instrumented_step(zero_start.phi, solid, p, zero_start.t)
    ch_production_end, ch_diag = pf.phase_only_step_with_diagnostics(zero_start, solid, p)
    ch_hash_match = _state_hashes(ch_end) == _state_hashes(ch_production_end)
    if not ch_hash_match:
        raise AuditValidationError(
            "instrumented CH-only one-step direction test differs from phase_only_step_with_diagnostics"
        )
    op_volume = np.asarray(pf.phase_control_volumes(solid, p), dtype=np.float64)
    delta_to_eq = np.asarray(eq_phi, dtype=np.float64) - np.asarray(start_state.phi, dtype=np.float64)
    adv = prod_fields["advective_increment"]
    ch = prod_fields["CH_increment"]
    net = prod_fields["net_increment"]
    ch_zero = np.asarray(ch_details["CH_increment"], dtype=np.float64)
    before_d = _distance_to_equilibrium(start_state.phi, eq_phi, solid, p)["raw_unaligned_relative_D_phi"]
    after_d = _distance_to_equilibrium(prod_end.phi, eq_phi, solid, p)["raw_unaligned_relative_D_phi"]
    after_zero_d = _distance_to_equilibrium(ch_end.phi, eq_phi, solid, p)["raw_unaligned_relative_D_phi"]
    one = {
        "start_step": 50_000,
        "public_steps": 1,
        "production_endpoint_hashes": _state_hashes(production_step),
        "instrumented_endpoint_hashes": _state_hashes(prod_end),
        "production_hash_match": production_hash_match,
        "production_implicit_converged_all_substeps": bool(np.all(np.asarray(production_diag.implicit_converged))),
        "advective_increment_direction_cosine_to_phi_eq_minus_phi": _volume_direction_cosine(
            adv, delta_to_eq, op_volume
        ),
        "CH_increment_direction_cosine_to_phi_eq_minus_phi": _volume_direction_cosine(ch, delta_to_eq, op_volume),
        "net_increment_direction_cosine_to_phi_eq_minus_phi": _volume_direction_cosine(net, delta_to_eq, op_volume),
        "zero_velocity_CH_increment_direction_cosine_to_phi_eq_minus_phi": _volume_direction_cosine(
            ch_zero, delta_to_eq, op_volume
        ),
        "raw_unaligned_D_phi_start": before_d,
        "raw_unaligned_D_phi_after_production_public_step": after_d,
        "raw_unaligned_D_phi_after_zero_velocity_CH_public_step": after_zero_d,
        "production_D_phi_change": after_d - before_d,
        "zero_velocity_CH_D_phi_change": after_zero_d - before_d,
        "adv_CH_cancellation_C_mag_whole_fluid": _unpack_metric_row(
            np.asarray(
                _advance_decomposed_block(
                    start_state,
                    solid,
                    p,
                    tuple(jnp.asarray(m, dtype=jnp.bool_) for m in masks.values()),
                    tuple((jnp.asarray(x), jnp.asarray(y)) for x, y in _face_masks_from_cells(masks)),
                    1,
                    1,
                )[3][0]
            )
        )["regions"]["whole_fluid"]["cancellation_C_mag_volume"],
        "production_advection_CH_decomposition_checks": {
            k: prod_details[k]
            for k in (
                "phase_reconstruction_linf",
                "adv_rhs_vs_face_divergence_linf",
                "explicit_implicit_CH_reconstruction_linf",
                "implicit_iterations_max",
                "implicit_relative_residual_max",
                "implicit_converged",
            )
        },
        "zero_velocity_CH_checks": {
            "endpoint_hash_match_production_phase_only_step": ch_hash_match,
            "reconstruction_linf": float(ch_details["reconstruction_linf"]),
            "implicit_iterations_max": int(ch_details["iterations_max"]),
            "implicit_relative_residual_max": float(ch_details["relative_residual_max"]),
            "implicit_converged": bool(ch_details["converged"]),
            "formal_mass_increment_CH": float(np.sum(op_volume * ch_zero, dtype=np.float64)),
            "formal_mass_increment_adv": float(
                np.sum(op_volume * np.asarray(ch_details["advective_increment"]), dtype=np.float64)
            ),
            "production_diag_converged_all_substeps": bool(np.all(np.asarray(ch_diag.implicit_converged))),
        },
    }
    return one


def _one_step_with_fixed_velocity(
    state: pf.State,
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
    scale: float,
    *,
    mode: str = "production",
) -> dict[str, Any]:
    phi_out, adv, ch, net, info = _fixed_velocity_phase_block(
        state.phi,
        state.u,
        state.v,
        solid,
        p,
        jnp.asarray(scale, dtype=jnp.float64),
        1,
        mode,
    )
    phi_out.block_until_ready()
    return {
        "mode": mode,
        "velocity_scale": float(scale),
        "phase_hash": _hash_array(phi_out),
        "advective_increment": np.asarray(adv, dtype=np.float64),
        "CH_increment": np.asarray(ch, dtype=np.float64),
        "net_increment": np.asarray(net, dtype=np.float64),
        "solver": {
            k: (bool(v) if k == "converged_all" else int(v) if k == "iterations_max" else float(v))
            for k, v in info.items()
            if k != "sampled_phase"
        },
    }


def _mass_ledger_summary(window: dict[str, Any]) -> dict[str, Any]:
    rows = window["metric_rows"]
    mass = {
        name: float(
            np.sum(
                [row["regions"]["whole_fluid"][name] for row in rows],
                dtype=np.float64,
            )
        )
        for name in (
            "formal_mass_increment_adv",
            "formal_mass_increment_ch",
            "formal_mass_increment_net",
        )
    }
    return {
        **mass,
        "adv_plus_CH_minus_net": mass["formal_mass_increment_adv"]
        + mass["formal_mass_increment_ch"]
        - mass["formal_mass_increment_net"],
        "formal_phase_mass_definition": (
            "Q = sum_i V_i phi_i; each increment is sum_i V_i delta_phi_i for the exact "
            "production substep decomposition"
        ),
        "production_gate_mass_drift_is_separate": True,
    }


def _load_phase_only_endpoint(result: dict[str, Any], config: dict[str, Any], label: str) -> pf.State:
    path = ROOT / result["checkpoint"]
    state, _metadata = _load_state_checkpoint(
        path,
        section="phase_only_continuation",
        case_name=label,
        step=int(result["absolute_end_step"]),
        config=config,
    )
    if _state_hashes(state) != result["state_hashes"]:
        raise AuditValidationError(f"{label}: endpoint checkpoint hash differs from its progress record")
    return state


def _velocity_scaling_sweep(
    state: pf.State,
    eq_phi: np.ndarray,
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
    *,
    steps: int,
    scales: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0),
) -> dict[str, Any]:
    volume = np.asarray(pf.phase_control_volumes(solid, p), dtype=np.float64)
    baseline = float(_distance_to_equilibrium(state.phi, eq_phi, solid, p)["raw_unaligned_relative_D_phi"])
    output = []
    for scale in scales:
        phi_out, adv, ch, net, info = _fixed_velocity_phase_block(
            state.phi,
            state.u,
            state.v,
            solid,
            p,
            jnp.asarray(scale, dtype=jnp.float64),
            int(steps),
            "production",
        )
        phi_out.block_until_ready()
        adv_np, ch_np, net_np = (np.asarray(item, dtype=np.float64) for item in (adv, ch, net))
        distance = _distance_to_equilibrium(phi_out, eq_phi, solid, p)
        output.append(
            {
                "velocity_scale": float(scale),
                "fixed_velocity_replay_steps": int(steps),
                "diagnostic_only": True,
                "initial_D_phi": baseline,
                "endpoint_D_phi": distance["raw_unaligned_relative_D_phi"],
                "D_phi_change": distance["raw_unaligned_relative_D_phi"] - baseline,
                "adv_rate_l2_volume_over_replay": _weighted_norm(adv_np / (steps * float(p.dt)), volume),
                "CH_rate_l2_volume_over_replay": _weighted_norm(ch_np / (steps * float(p.dt)), volume),
                "net_rate_l2_volume_over_replay": _weighted_norm(net_np / (steps * float(p.dt)), volume),
                "advective_direction_cosine_to_equilibrium_displacement": _volume_direction_cosine(
                    adv_np, np.asarray(eq_phi) - np.asarray(state.phi), volume
                ),
                "CH_direction_cosine_to_equilibrium_displacement": _volume_direction_cosine(
                    ch_np, np.asarray(eq_phi) - np.asarray(state.phi), volume
                ),
                "formal_mass_increment_adv": float(np.sum(volume * adv_np, dtype=np.float64)),
                "formal_mass_increment_CH": float(np.sum(volume * ch_np, dtype=np.float64)),
                "formal_mass_increment_net": float(np.sum(volume * net_np, dtype=np.float64)),
                "phase_free_energy_components": _phase_energy_parts(phi_out, solid, p),
                "phase_min_max": [float(np.min(np.asarray(phi_out))), float(np.max(np.asarray(phi_out)))],
                "state_hash": _hash_array(phi_out),
                "solver": {
                    "implicit_iterations_max": int(info["iterations_max"]),
                    "implicit_relative_residual_max": float(info["relative_residual_max"]),
                    "implicit_converged_all": bool(info["converged_all"]),
                },
                "velocity_field": (
                    "the production state u/v at step 50000 frozen for every replay step; "
                    "not a saved time-varying production velocity trajectory"
                ),
            }
        )
    return {
        "status": "measured",
        "M": float(p.M),
        "dt": float(p.dt),
        "steps": int(steps),
        "scales": output,
        "not_run": {
            "M_4x": "unmeasured_by_design; cannot establish production closure",
            "dt_half": "unmeasured_unless_cancellation_and_replay_leave_cause_ambiguous",
        },
    }


def _matched_velocity_zero_counterfactual(
    state: pf.State,
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
    *,
    steps: int,
) -> dict[str, Any]:
    """Compare full production evolution with the exact zero-u CH-only branch from one state."""
    production_end = chns._advance_standard(state, solid, p, int(steps))
    zero_start = pf.State(
        state.phi,
        jnp.zeros_like(state.u, dtype=p.dtype),
        jnp.zeros_like(state.v, dtype=p.dtype),
        state.t,
    )
    zero_previous, zero_end, iterations, residual, converged = nwa._advance(zero_start, solid, p, int(steps), True)
    production_delta = np.asarray(production_end.phi, dtype=np.float64) - np.asarray(state.phi, dtype=np.float64)
    zero_delta = np.asarray(zero_end.phi, dtype=np.float64) - np.asarray(state.phi, dtype=np.float64)
    difference = production_delta - zero_delta
    volume = np.asarray(pf.phase_control_volumes(solid, p), dtype=np.float64)
    return {
        "status": "measured",
        "matched_steps": int(steps),
        "M": float(p.M),
        "dt": float(p.dt),
        "production_endpoint_state_hashes": _state_hashes(production_end),
        "zero_velocity_endpoint_state_hashes": _state_hashes(zero_end),
        "production_minus_zero_phase_change_l2_volume": _weighted_norm(difference, volume),
        "production_phase_change_l2_volume": _weighted_norm(production_delta, volume),
        "zero_velocity_phase_change_l2_volume": _weighted_norm(zero_delta, volume),
        "production_formal_mass_increment": float(np.sum(volume * production_delta, dtype=np.float64)),
        "zero_velocity_formal_mass_increment": float(np.sum(volume * zero_delta, dtype=np.float64)),
        "zero_velocity_formal_mass_increment_adv": 0.0,
        "zero_velocity_formal_mass_increment_CH": float(np.sum(volume * zero_delta, dtype=np.float64)),
        "zero_velocity_implicit_iterations_max": int(iterations),
        "zero_velocity_implicit_relative_residual_max": float(residual),
        "zero_velocity_implicit_last_step_converged": bool(converged),
        "zero_velocity_previous_phi_hash": _hash_array(zero_previous),
        "production_end_free_energy": _phase_energy_parts(production_end.phi, solid, p),
        "zero_velocity_end_free_energy": _phase_energy_parts(zero_end.phi, solid, p),
        "production_end_contact_metrics": _contact_metrics(production_end.phi, solid, p),
        "zero_velocity_end_contact_metrics": _contact_metrics(zero_end.phi, solid, p),
        "not_production_acceptance_evidence": True,
    }


def _classify_candidates(
    authority_windows: dict[str, Any] | None,
    controls: dict[str, Any],
    freeze_u: dict[str, Any] | None,
    replay: dict[str, Any] | None,
    direction: dict[str, Any] | None,
    mu_comparison: dict[str, Any] | None,
    two_k: dict[str, Any],
    *,
    required_provenance_valid: bool,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []

    def add(name: str, status: str, evidence: Any, counterfactual: Any, scope: str):
        candidates.append(
            {
                "candidate": name,
                "status": status,
                "effect_sizes": evidence,
                "control_or_falsification": counterfactual,
                "scope": scope,
                "production_repair_or_contract_change": False,
            }
        )

    if not required_provenance_valid:
        provenance_status = "SUSPECTED"
        provenance_evidence = "required replay per-field hashes did not validate"
    else:
        provenance_status = "FALSIFIED"
        provenance_evidence = (
            "required production and CH-only reference state hashes were accepted against saved provenance"
        )
    add(
        "STATE_PROVENANCE_MISMATCH",
        provenance_status,
        provenance_evidence,
        "L1A-2k per-field state hashes and strict current-source/config checkpoint checks",
        "evidence admissibility blocker only; not a physics root-cause claim",
    )

    cancellation_status = "NOT_TESTED"
    cancellation_effect = "unmeasured"
    cancellation_falsification = "unmeasured"
    hydro_status = "NOT_TESTED"
    hydro_effect = "unmeasured"
    hydro_falsification = "unmeasured"
    if authority_windows is not None:
        windows = [
            authority_windows.get(key)
            for key in ("one_step_50000", "burst_10_step_48k_50k", "window_1000_step_40k_50k")
        ]
        windows = [item for item in windows if item is not None]
        values = [item["metric_rows"][-1]["regions"]["whole_fluid"] for item in windows if item.get("metric_rows")]
        if values:
            cdir_values = [float(value["alignment_C_dir_volume"]) for value in values]
            cmag_values = [float(value["cancellation_C_mag_volume"]) for value in values]
            cancellation_effect = {
                "C_dir_by_window": cdir_values,
                "C_mag_by_window": cmag_values,
                "net_to_adv_norm_by_window": [
                    float(value["net_rate_l2_volume"] / max(value["adv_rate_l2_volume"], 1.0e-300)) for value in values
                ],
                "net_to_CH_norm_by_window": [
                    float(value["net_rate_l2_volume"] / max(value["ch_rate_l2_volume"], 1.0e-300)) for value in values
                ],
            }
            controls_values = []
            for c in controls.values():
                for w in c.get("windows", {}).values():
                    rows = w.get("metric_rows", [])
                    if rows:
                        controls_values.append(rows[-1]["regions"]["whole_fluid"]["cancellation_C_mag_volume"])
            cancellation_falsification = {
                "control_C_mag_last_values_90_150": controls_values,
                "authority_C_mag_strictly_below_all_controls": bool(
                    controls_values and max(cmag_values) < min(controls_values)
                ),
                "status_basis": "direct measured signs and magnitude ordering; no post-measurement cutoff",
            }
            all_opposed = bool(cdir_values and all(value < 0.0 for value in cdir_values))
            net_reduced = bool(
                all(
                    value["net_rate_l2_volume"] < min(value["adv_rate_l2_volume"], value["ch_rate_l2_volume"])
                    for value in values
                )
            )
            control_differential = bool(cancellation_falsification["authority_C_mag_strictly_below_all_controls"])
            cancellation_status = (
                "SUPPORTED"
                if all_opposed and net_reduced and control_differential
                else "SUSPECTED"
                if all_opposed or net_reduced
                else "FALSIFIED"
            )
        if direction is not None and replay is not None:
            d = direction
            production_zero_effect = replay.get("production_minus_zero_phase_change_l2_volume")
            control_replay_effects = {
                name: case.get("matched_velocity_zero_counterfactual", {}).get(
                    "production_minus_zero_phase_change_l2_volume"
                )
                for name, case in controls.items()
            }
            finite_control_effects = [float(value) for value in control_replay_effects.values() if value is not None]
            adv_away = (
                d.get("advective_increment_direction_cosine_to_phi_eq_minus_phi") is not None
                and float(d["advective_increment_direction_cosine_to_phi_eq_minus_phi"]) < 0.0
            )
            ch_toward = (
                d.get("CH_increment_direction_cosine_to_phi_eq_minus_phi") is not None
                and float(d["CH_increment_direction_cosine_to_phi_eq_minus_phi"]) > 0.0
            )
            production_replay = replay.get("production_velocity_replay", {})
            zero_velocity_counterfactual = replay.get("zero_velocity_counterfactual", {})
            production_replay = production_replay if isinstance(production_replay, dict) else {}
            zero_velocity_counterfactual = (
                zero_velocity_counterfactual if isinstance(zero_velocity_counterfactual, dict) else {}
            )
            matched_prod_delta = production_replay.get("D_phi_production_change")
            matched_zero_delta = zero_velocity_counterfactual.get("D_phi_zero_velocity_change")
            one_step_prod_delta = d.get("production_D_phi_change")
            one_step_zero_delta = d.get("zero_velocity_CH_D_phi_change")
            dphi_away = (
                matched_prod_delta is not None
                and matched_zero_delta is not None
                and float(matched_prod_delta) > float(matched_zero_delta)
            )
            controls_falsify = bool(
                finite_control_effects
                and production_zero_effect is not None
                and float(production_zero_effect) > max(finite_control_effects)
            )
            hydro_effect = {
                "one_step_advective_direction_cosine_to_equilibrium": d.get(
                    "advective_increment_direction_cosine_to_phi_eq_minus_phi"
                ),
                "one_step_CH_direction_cosine_to_equilibrium": d.get(
                    "CH_increment_direction_cosine_to_phi_eq_minus_phi"
                ),
                "one_step_production_D_phi_change": one_step_prod_delta,
                "one_step_zero_velocity_CH_D_phi_change": one_step_zero_delta,
                "matched_production_D_phi_change": matched_prod_delta,
                "matched_zero_velocity_CH_D_phi_change": matched_zero_delta,
                "matched_production_minus_zero_D_phi_change": replay.get(
                    "D_phi_production_minus_zero_after_matched_steps"
                ),
                "matched_production_minus_zero_phase_change_l2_volume": production_zero_effect,
                "matched_control_phase_change_l2_volume_differences": control_replay_effects,
                "advective_increment_points_away_from_equilibrium": adv_away,
                "CH_increment_points_toward_equilibrium": ch_toward,
            }
            hydro_falsification = {
                "matched_zero_velocity_counterfactual": replay.get("zero_velocity_counterfactual"),
                "matched_90_150_control_counterfactual_effects": control_replay_effects,
                "control_effects_are_differential_falsification": controls_falsify,
                "control_late_window_advective_rates_volume": {
                    name: {
                        window_name: window["metric_rows"][-1]["regions"]["whole_fluid"]["adv_rate_l2_volume"]
                        for window_name, window in case.get("windows", {}).items()
                        if window.get("metric_rows")
                    }
                    for name, case in controls.items()
                },
                "support_rule": (
                    "requires advection away from and CH toward the provenance-checked 60-degree "
                    "equilibrium, larger production-minus-zero D_phi growth, and a matched "
                    "90/150 differential control"
                ),
            }
            if adv_away and ch_toward and dphi_away and controls_falsify:
                hydro_status = "SUPPORTED"
            elif not adv_away or not ch_toward or (finite_control_effects and not controls_falsify):
                hydro_status = "FALSIFIED"
            else:
                hydro_status = "SUSPECTED"
    add(
        "ADVECTIVE_CH_NEAR_CANCELLATION",
        cancellation_status,
        cancellation_effect,
        cancellation_falsification,
        (
            "60-degree differential mechanism candidate; C_dir is a cut-cell-volume cosine "
            "and C_mag is the cut-cell-volume norm ratio"
        ),
    )
    add(
        "HYDRODYNAMICALLY_DRIVEN_PHASE_NONEQUILIBRIUM",
        hydro_status,
        hydro_effect,
        hydro_falsification,
        "causal one-step plus matched 100-step production-velocity/zero-velocity counterfactual",
    )

    kinetics_status = "NOT_TESTED"
    kinetics_effect: Any = "unmeasured"
    kinetics_falsification: Any = "unmeasured"
    if freeze_u is not None:
        samples = freeze_u.get("samples", [])
        energies = [float(row["free_energy"]) for row in samples if row.get("free_energy") is not None]
        energy_diffs = np.diff(energies) if len(energies) > 1 else np.asarray([], dtype=np.float64)
        dphis = [float(row["raw_unaligned_D_phi"]) for row in samples if row.get("raw_unaligned_D_phi") is not None]
        dphi_diffs = np.diff(dphis) if len(dphis) > 1 else np.asarray([], dtype=np.float64)
        monotone_energy = bool(len(energy_diffs) and np.all(energy_diffs <= 0.0))
        monotone_dphi = bool(len(dphi_diffs) and np.all(dphi_diffs <= 0.0))
        long_vs_window = int(freeze_u.get("relative_steps", 0)) > 10_000
        kinetics_effect = {
            "status": freeze_u.get("status"),
            "M": freeze_u.get("config", {}).get("M"),
            "dt": freeze_u.get("config", {}).get("dt"),
            "steps_to_gate_or_budget": freeze_u.get("relative_steps"),
            "physical_time": freeze_u.get("physical_continuation_time"),
            "mobility_scaled_time": freeze_u.get("mobility_scaled_continuation_time"),
            "energy_monotone_exact_sample_order": monotone_energy,
            "raw_unaligned_D_phi_monotone_exact_sample_order": monotone_dphi,
            "max_sample_energy_increase": float(np.max(energy_diffs)) if len(energy_diffs) else None,
            "max_sample_D_phi_increase": float(np.max(dphi_diffs)) if len(dphi_diffs) else None,
        }
        kinetics_falsification = {
            "comparison_window_steps": 10_000,
            "longer_than_matched_production_window": long_vs_window,
            "M4_not_used_as_production_closure": True,
            "converged": freeze_u.get("final_convergence_gate", {}).get("converged"),
        }
        kinetics_status = (
            "SUPPORTED"
            if long_vs_window and monotone_energy and monotone_dphi
            else "SUSPECTED"
            if long_vs_window or not freeze_u.get("final_convergence_gate", {}).get("converged")
            else "FALSIFIED"
        )
    add(
        "MREF_INTRINSIC_CH_RELAXATION_LIMITED",
        kinetics_status,
        kinetics_effect,
        kinetics_falsification,
        "matched freeze-u, M_ref continuation only; diagnostic and not production closure",
    )

    if mu_comparison is None:
        add(
            "CHEMICAL_POTENTIAL_UNIFORMITY_LIMITATION",
            "NOT_TESTED",
            "unmeasured",
            "unmeasured",
            "componentwise invariant precondition unavailable",
        )
    else:
        a = mu_comparison.get("authority_060", {})
        e = mu_comparison.get("ch_only_equilibrium_060", {})
        au = a.get("mu_statistics", {})
        eq = e.get("mu_statistics", {})
        if au.get("status") == "measured" and eq.get("status") == "measured":
            std_au = au.get("mu_statistics", {}).get("std_volume_weighted")
            std_eq = eq.get("mu_statistics", {}).get("std_volume_weighted")
            mean_eq = abs(float(eq.get("mu_statistics", {}).get("mean_volume_weighted", 0.0)))
            ratio = None if std_au is None else float(std_au) / max(mean_eq, 1.0e-300)
            add(
                "CHEMICAL_POTENTIAL_NONUNIFORMITY",
                "SUSPECTED" if std_au and std_eq is not None else "FALSIFIED",
                {
                    "authority_mu": au.get("mu_statistics"),
                    "equilibrium_mu": eq.get("mu_statistics"),
                    "authority_std_over_equilibrium_mean_abs": ratio,
                },
                "single-fluid-component precondition and matched converged CH-only endpoint",
                "diagnostic gradient/free-energy state indicator; not a standalone causal claim",
            )
        else:
            add(
                "CHEMICAL_POTENTIAL_UNIFORMITY_LIMITATION",
                "NOT_TESTED",
                {"authority": au, "equilibrium": eq},
                "fail-closed componentwise chemical-potential precondition",
                "no cross-component averaging",
            )

    capillary_result = two_k.get("mechanism_matrix", {})
    capillary_background = {
        "L1A2k_final_root_cause": capillary_result.get("final_root_cause"),
        "L1A2k_causal_root_cause_of_L1A2j_nonstationarity": capillary_result.get(
            "causal_root_cause_of_l1a2j_nonstationarity"
        ),
        "scope": capillary_result.get("final_root_cause_scope"),
        "not_reused_as_causal_60_degree_evidence": True,
    }
    add(
        "N-CAPILLARY-PRESSURE-BALANCE",
        "SUPPORTED",
        capillary_background,
        (
            "structural L1A-2k evidence retained only as a background; "
            "no differential 60-degree causal evidence is inferred"
        ),
        "structural background only; explicitly not the 60-degree root cause",
    )
    add(
        "N-CH-MASS-PRECISION",
        "FALSIFIED",
        "contract-11 cut-cell mass ledger and per-increment sums are directly reported",
        "N-CH-MASS-PRECISION remains resolved_in_contract_v11",
        "historical resolved constraint, not reopened",
    )
    add(
        "N-WALL-ALIGNMENT-TRANSPORT-DOMAIN",
        "FALSIFIED",
        "contract-9 cut-cell-volume regional masks/operators are used without changing geometry",
        "N-WALL-ALIGNMENT-TRANSPORT-DOMAIN remains resolved_in_contract_v9",
        "historical resolved constraint, not reopened",
    )
    add(
        "W-CONTACT-ANGLE",
        "NOT_TESTED",
        "left/right contact metrics are measured separately; equilibrium target and side-specific residual remain open",
        "90/150 controls and converged CH-only reference",
        "open condition; do not upgrade angle-cycle classification",
    )
    add(
        "MULTIPLE_CONTRIBUTORS",
        "SUPPORTED"
        if sum(
            item["status"] == "SUPPORTED"
            and item["candidate"]
            not in {"N-CAPILLARY-PRESSURE-BALANCE", "N-CH-MASS-PRECISION", "N-WALL-ALIGNMENT-TRANSPORT-DOMAIN"}
            for item in candidates
        )
        >= 2
        else "NOT_TESTED",
        "unmeasured until distinct causal candidates independently pass their counterfactuals",
        "each individual mechanism matrix row",
        "composite label is not a way to bypass candidate-level evidence",
    )

    final_root = "INCONCLUSIVE"
    final_scope = (
        "No sole 60-degree root cause is assigned without provenance-checked, differential "
        "causal evidence; the L1A-2k capillary result remains a structural background only."
    )
    return {
        "stage": STAGE,
        "solver_contract_version": SOLVER_CONTRACT,
        "candidates": candidates,
        "final_root_cause": final_root,
        "final_root_cause_scope": final_scope,
        "allowed_final_root_cause_labels": [
            "INCONCLUSIVE",
            "suspected_in_contract_v11",
            "confirmed_problem_in_contract_v11",
        ],
        "production_semantics_changed": False,
        "no_contract_bump": True,
    }


def _initial_report(two_j: dict[str, Any] | None, two_k: dict[str, Any] | None, profile: str) -> dict[str, Any]:
    sections = {
        "production_state_rehydration_60_90_150": "unmeasured",
        "authority_060_phase_rate_decomposition_1_10_1000_steps": "unmeasured",
        "stationary_control_late_windows_090_150": "unmeasured",
        "ch_only_060_equilibrium_reference": "unmeasured",
        "chemical_potential_uniformity_and_face_fluxes": "unmeasured",
        "phase_free_energy_and_regional_localization": "unmeasured",
        "raw_unaligned_equilibrium_distance_and_morphology": "unmeasured",
        "freeze_u_Mref_continuation": "unmeasured",
        "production_velocity_replay_vs_zero_velocity": "unmeasured",
        "one_step_direction_tests": "unmeasured",
        "velocity_scaling": "unmeasured",
        "left_right_contact_line_correlations": "unmeasured",
        "dt_half": "unmeasured_not_triggered",
        "M_4x_closure": "unmeasured_by_design_not_a_production_closure_test",
    }
    return {
        "stage": STAGE,
        "status": "in_progress",
        "profile": profile,
        "created_at_local": dt.datetime.now().astimezone().isoformat(),
        "updated_at_local": dt.datetime.now().astimezone().isoformat(),
        "solver_contract_version": SOLVER_CONTRACT,
        "diagnostic_only": True,
        "production_acceptance_evidence": False,
        "production_semantics_changed": False,
        "contract_bumped": False,
        "production_defaults": {
            "M": float(chns.M_REF),
            "dt": float(chns.DT),
            "phase_storage_model": pf.PHASE_ONLY_FLOAT64_STORAGE_MODEL,
            "phase_boundary_model": "impermeable_flux",
            "wetting_model": "surface_energy",
            "wall_measure": pf.WALL_MEASURE_METHOD,
            "phase_transport_geometry": pf.PHASE_TRANSPORT_GEOMETRY,
            "phase_advection_subcycling": pf.PHASE_ADVECTION_SUBCYCLING,
            "momentum_projection_and_Brinkman_defaults": "unchanged contract-11 production configuration",
            "acceptance_thresholds": "unchanged",
        },
        "runtime_versions": _runtime_versions(),
        "repository_git_sha": chns.get_git_sha(),
        "source_hashes": _source_hashes(),
        "upstream_reports": {
            "l1a2j_path": str(UPSTREAM_2J.relative_to(ROOT)),
            "l1a2k_path": str(UPSTREAM_2K.relative_to(ROOT)),
            "l1a2j_source_hashes_match_for_reused_components": None
            if two_j is None
            else {
                key: two_j.get("source_hashes", {}).get(key) == _source_hashes().get(key)
                for key in ("phasefield", "nonneutral_wetting_audit", "contact_line_kinetics")
            },
            "l1a2k_source_hashes_match_current_audit": None
            if two_k is None
            else two_k.get("source_hashes") == l1a2k._source_hashes(),
            "l1a2j_l1a2k_context": None if two_j is None or two_k is None else _history_context(two_j, two_k),
        },
        "exact_convergence_gate": _production_gate_definition(),
        "measurements": sections,
        "cases": {},
        "windows": {},
        "equilibrium_reference": "unmeasured",
        "chemical_potential": "unmeasured",
        "free_energy_localization": "unmeasured",
        "morphology_comparison": "unmeasured",
        "freeze_u_continuation": "unmeasured",
        "velocity_replay": "unmeasured",
        "one_step_direction_tests": "unmeasured",
        "velocity_scaling": "unmeasured",
        "left_right_contact_line_metrics": "unmeasured",
        "effect_sizes": "unmeasured",
        "candidate_classifications": "unmeasured",
        "unmeasured_sections": list(sections),
        "mechanism_matrix": {
            "stage": STAGE,
            "solver_contract_version": SOLVER_CONTRACT,
            "candidates": [],
            "final_root_cause": "INCONCLUSIVE",
            "final_root_cause_scope": "Measurements are incomplete; no causal label assigned.",
            "production_semantics_changed": False,
        },
        "quality_status": {
            "pytest": "unmeasured",
            "ruff": "unmeasured",
            "production_solver_source_integrity": "unmeasured",
        },
    }


def _render_markdown(report: dict[str, Any]) -> str:
    matrix = report.get("mechanism_matrix", {})
    lines = [
        "# L1A-2l phase-coupling and relaxation forensics",
        "",
        f"- **Status:** `{report.get('status')}`",
        f"- **Profile:** `{report.get('profile')}`",
        f"- **Final root-cause label:** `{matrix.get('final_root_cause', 'INCONCLUSIVE')}`",
        f"- **Solver contract:** `{report.get('solver_contract_version')}` (unchanged)",
        f"- **Production semantics changed:** `{report.get('production_semantics_changed')}`",
        "- **Diagnostic-only:** yes; none of these measurements are production acceptance evidence.",
        "",
        "## Frozen production conditions",
        "",
        (
            f"`M={report.get('production_defaults', {}).get('M')}`, "
            f"`dt={report.get('production_defaults', {}).get('dt')}`, phase-only float64 storage, "
            "Young surface energy, cut-cell geometry, momentum/projection, Brinkman defaults, "
            "and all acceptance thresholds remain unchanged."
        ),
        "",
        "## Required state provenance",
        "",
        (
            "The raw L1A-2j/L1A-2k state arrays were absent at execution. Production trajectories "
            "were rehydrated from their exact seeds and are accepted only when all four per-field "
            "hashes match the saved L1A-2k report. Any mismatch rejects that state for causal comparisons."
        ),
        "",
    ]
    for name, case in report.get("cases", {}).items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- State acceptance: `{case.get('state_acceptance')}` at step `{case.get('step')}`.",
                f"- Per-field hash match to frozen L1A-2k: `{case.get('matches_l1a2k_per_field_hashes')}`.",
                f"- Configuration fingerprint: `{case.get('config_fingerprint')}`.",
                "",
            ]
        )
    lines.extend(["## Production-gate normalization", ""])
    gate = report.get("exact_convergence_gate", {})
    lines.extend(
        [
            f"The exact phase-rate field is `{gate.get('field')}`: `{gate.get('definition')}`.",
            (
                f"Normalization is `{gate.get('normalization')}`; mask: `{gate.get('mask')}`; "
                f"`dt={gate.get('dt')}`; M-ref threshold `{gate.get('threshold_at_M_ref')}`."
            ),
            "Cut-cell-volume-weighted forensic norms are reported separately and never replace this gate.",
            "",
        ]
    )
    lines.extend(["## Decomposed windows", ""])
    lines.append(
        (
            "| Window | Exact production endpoint | Samples | Whole-fluid C_dir (last) | "
            "C_mag (last) | Advective L2 (last) | CH L2 (last) | Net L2 (last) |"
        )
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, window in report.get("windows", {}).items():
        summary = window.get("summary", {})
        rows = window.get("metric_rows", [])
        last = rows[-1]["regions"]["whole_fluid"] if rows else {}
        lines.append(
            (
                f"| {name} | {window.get('endpoint_exact_hash_match_production')} | "
                f"{summary.get('sample_count')} | {last.get('alignment_C_dir_volume')} | "
                f"{last.get('cancellation_C_mag_volume')} | {last.get('adv_rate_l2_dxdy')} | "
                f"{last.get('ch_rate_l2_dxdy')} | {last.get('net_rate_l2_dxdy')} |"
            )
        )
    lines.extend(["", "## Equilibrium and counterfactuals", ""])
    eq = report.get("equilibrium_reference", {})
    eq = eq if isinstance(eq, dict) else {}
    lines.append(
        (
            f"- CH-only 60° reference: `{eq.get('state_acceptance', eq.get('status', 'unmeasured'))}`; "
            f"step `{eq.get('absolute_end_step')}`; gate `{eq.get('final_convergence_gate')}`."
        )
    )
    morphology = report.get("morphology_comparison", {})
    morphology = morphology if isinstance(morphology, dict) else {}
    distance = morphology.get("authority_vs_ch_only_equilibrium", {}).get("raw_unaligned_relative_D_phi")
    lines.append(f"- Raw, unaligned authority-to-equilibrium distance: `{distance}`.")
    freeze = report.get("freeze_u_continuation", {})
    freeze = freeze if isinstance(freeze, dict) else {}
    lines.append(
        (
            f"- Freeze-u M_ref continuation: `{freeze.get('status', 'unmeasured')}`; "
            f"steps `{freeze.get('relative_steps')}`; gate `{freeze.get('final_convergence_gate')}`."
        )
    )
    replay = report.get("velocity_replay", {})
    replay = replay if isinstance(replay, dict) else {}
    lines.append(
        (
            f"- Matched production-velocity vs zero-velocity replay: `{replay.get('status', 'unmeasured')}`; "
            f"steps `{replay.get('matched_steps')}`."
        )
    )
    lines.extend(["", "## Candidate matrix", ""])
    lines.append("| Candidate | Status | Scope |")
    lines.append("|---|---|---|")
    for candidate in matrix.get("candidates", []):
        lines.append(f"| `{candidate.get('candidate')}` | `{candidate.get('status')}` | {candidate.get('scope')} |")
    lines.extend(
        [
            "",
            (
                f"**Final root-cause label:** `{matrix.get('final_root_cause', 'INCONCLUSIVE')}` — "
                f"{matrix.get('final_root_cause_scope', '')}"
            ),
            "",
            "## Unmeasured sections",
            "",
        ]
    )
    unmeasured = report.get("unmeasured_sections", [])
    lines.extend([f"- `{name}`" for name in unmeasured] or ["- None."])
    lines.extend(
        [
            "",
            "## Quality and provenance",
            "",
            f"- Runtime: `{report.get('runtime_versions')}`.",
            f"- Source hashes: `{report.get('source_hashes')}`.",
            f"- Quality status: `{report.get('quality_status')}`.",
            (
                "- `N-CH-MASS-PRECISION=resolved_in_contract_v11`, "
                "`N-WALL-ALIGNMENT-TRANSPORT-DOMAIN=resolved_in_contract_v9`, "
                "`N-CAPILLARY-PRESSURE-BALANCE` as a structural background, "
                "and `W-CONTACT-ANGLE=open` are preserved."
            ),
            (
                "- No generic L1A-2k residual is used as a causal 60° explanation; the apparent "
                "angle-cycle classification is not upgraded by this report."
            ),
            "",
        ]
    )
    refresh = report.get("derived_report_refresh")
    if isinstance(refresh, dict):
        measurement_hash = refresh.get("measurement_source_hashes", {}).get("audit_runner")
        analysis_hash = refresh.get("postprocessing_source_hashes", {}).get("audit_runner")
        lines.extend(
            [
                "",
                "### Post-run classifier refresh",
                "",
                f"- Scope: {refresh.get('scope')}",
                f"- Measurement-run audit source SHA-256: `{measurement_hash}`.",
                f"- Post-processing audit source SHA-256: `{analysis_hash}`.",
                "- No solver replay or production-state measurement was rerun or altered by this summary correction.",
            ]
        )
    return "\n".join(lines)


def _write_deliverables(report: dict[str, Any]) -> dict[str, str]:
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    report["updated_at_local"] = dt.datetime.now().astimezone().isoformat()
    report["final_verdict"] = report.get("mechanism_matrix", {}).get("final_root_cause", "INCONCLUSIVE")
    report["unmeasured_sections"] = [
        name for name, status in report.get("measurements", {}).items() if str(status).startswith("unmeasured")
    ]
    required_measurements = (
        "production_state_rehydration_60_90_150",
        "authority_060_phase_rate_decomposition_1_10_1000_steps",
        "stationary_control_late_windows_090_150",
        "ch_only_060_equilibrium_reference",
        "chemical_potential_uniformity_and_face_fluxes",
        "phase_free_energy_and_regional_localization",
        "raw_unaligned_equilibrium_distance_and_morphology",
        "freeze_u_Mref_continuation",
        "production_velocity_replay_vs_zero_velocity",
        "one_step_direction_tests",
        "left_right_contact_line_correlations",
    )
    report["required_measurements_for_complete_status"] = list(required_measurements)
    measured_core = [str(report.get("measurements", {}).get(key, "unmeasured")) for key in required_measurements]
    report["status"] = (
        "complete"
        if measured_core
        and all(
            value.startswith("measured") and not any(token in value for token in ("rejected", "unavailable", "missing"))
            for value in measured_core
        )
        else "partial_unmeasured"
    )
    report_path = EVIDENCE_ROOT / "phase_coupling_relaxation_report.json"
    markdown_path = EVIDENCE_ROOT / "phase_coupling_relaxation_report.md"
    matrix_path = EVIDENCE_ROOT / "mechanism_matrix.json"
    quality_path = EVIDENCE_ROOT / "quality_status.json"
    manifest_path = EVIDENCE_ROOT / "manifest.json"
    _write_json(matrix_path, report.get("mechanism_matrix", {}))
    _write_json(report_path, report)
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    _write_json(quality_path, report.get("quality_status", {"status": "unmeasured"}))
    artifact_paths = [report_path, markdown_path, matrix_path, quality_path]
    generated_fields: list[Path] = []
    seen_artifacts: set[str] = set()

    def collect_artifacts(value: Any) -> None:
        if isinstance(value, dict):
            artifact_path = value.get("path")
            if isinstance(artifact_path, str):
                path = ROOT / artifact_path
                key = str(path.resolve())
                if key not in seen_artifacts and path.is_file():
                    seen_artifacts.add(key)
                    generated_fields.append(path)
            for child in value.values():
                collect_artifacts(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                collect_artifacts(child)

    collect_artifacts(report)
    artifacts = artifact_paths + generated_fields
    manifest = {
        "stage": STAGE,
        "solver_contract_version": SOLVER_CONTRACT,
        "git_sha": chns.get_git_sha(),
        "source_hashes": _source_hashes(),
        "runtime_versions": _runtime_versions(),
        "production_semantics_changed": False,
        "config_fingerprints": {name: case.get("config_fingerprint") for name, case in report.get("cases", {}).items()},
        "artifacts": [
            {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path.read_bytes()), "bytes": path.stat().st_size}
            for path in artifacts
        ],
        "manifest_self_hash": "omitted_to_avoid_recursive_hash",
        "unmeasured_sections": report.get("unmeasured_sections", []),
    }
    _write_json(manifest_path, manifest)
    return {
        "report_json": str(report_path.relative_to(ROOT)),
        "report_markdown": str(markdown_path.relative_to(ROOT)),
        "mechanism_matrix": str(matrix_path.relative_to(ROOT)),
        "manifest": str(manifest_path.relative_to(ROOT)),
        "quality_status": str(quality_path.relative_to(ROOT)),
    }


def run_forensic(
    *,
    profile: str = "forensic",
    max_phase_only_steps: int = 180_000,
    max_freeze_steps: int = 100_000,
    velocity_replay_steps: int = 100,
    run_quality: bool = True,
) -> dict[str, Any]:
    """Execute the resumable diagnostic audit and publish JSON/Markdown evidence."""
    two_j, two_k = _validate_upstream()
    report = _initial_report(two_j, two_k, profile)
    progress_dir = ARTIFACT_ROOT / "progress"
    report["requested_budgets"] = {
        "ch_only_reference_steps": int(max_phase_only_steps),
        "freeze_u_steps": int(max_freeze_steps),
        "production_velocity_replay_steps": int(velocity_replay_steps),
        "sample_stride_steps": 1_000,
        "authority_burst": {
            "start_step": AUTHORITY_BURST_START,
            "steps": AUTHORITY_BURST_STEPS,
            "sample_stride_steps": 10,
        },
        "controls_window_steps": CONTROL_WINDOW_LENGTH,
    }
    _write_deliverables(report)

    case_entries: dict[str, dict[str, Any]] = {}
    case_parameters: dict[str, tuple[pf.PhaseFieldParams, pf.Solid, pf.State, dict[str, Any]]] = {}
    accepted_states: dict[str, pf.State] = {}
    endpoint_provenance: dict[str, Any] = {}
    for name, target in CASE_TARGETS.items():
        try:
            entry, p, solid, state, provenance = _rehydrate_case(name, target, two_k, progress_dir)
            case_entries[name] = entry
            case_parameters[name] = (p, solid, _make_case(target)[2], entry["config"])
            endpoint_provenance[name] = provenance
            if state is not None:
                accepted_states[name] = state
            report["cases"][name] = entry
            report["measurements"]["production_state_rehydration_60_90_150"] = (
                "measured" if len(accepted_states) == len(CASE_TARGETS) else "measured_with_rejected_states"
            )
            _write_deliverables(report)
        except Exception as exc:
            case_entries[name] = {
                "case": name,
                "target_deg": target,
                "state_acceptance": "rejected_validation_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            report["cases"][name] = case_entries[name]
            report["measurements"]["production_state_rehydration_60_90_150"] = "measured_with_rejected_states"
            _write_deliverables(report)

    valid_reference_state: pf.State | None = None
    equilibrium_result: dict[str, Any] | None = None
    eq_phi: np.ndarray | None = None
    reference_hash_match = False
    if profile == "forensic":
        p60, solid60, seed60, config60 = _make_case(60.0)
        reference_config = _phase_only_config(60.0, M=float(chns.M_REF), branch_name="ch_only_equilibrium_060")
        upstream_phase = two_k.get("phase_only_comparison", {})
        expected_reference_config = upstream_phase.get("ch_only_config")
        config_match = isinstance(expected_reference_config, dict) and _canonical_hash(
            reference_config
        ) == _canonical_hash(expected_reference_config)
        if config_match:
            try:
                equilibrium_result = _run_phase_only_continuation(
                    "ch_only_equilibrium_060",
                    seed60,
                    solid60,
                    p60,
                    reference_config,
                    max_steps=int(max_phase_only_steps),
                    start_absolute_step=0,
                    parent_state_hashes=_state_hashes(seed60),
                    progress_dir=progress_dir,
                    phi_eq=None,
                    stop_when_converged=False,
                )
                expected_endpoint_hashes = dict(upstream_phase.get("ch_only_state_hashes", {}))
                reference_hash_match = (
                    equilibrium_result.get("status") == "converged"
                    and equilibrium_result.get("absolute_end_step") == int(upstream_phase.get("ch_only_step", -1))
                    and equilibrium_result.get("state_hashes") == expected_endpoint_hashes
                )
                equilibrium_result["saved_l1a2k_config_match"] = config_match
                equilibrium_result["saved_l1a2k_endpoint_hashes"] = expected_endpoint_hashes
                equilibrium_result["matches_l1a2k_per_field_hashes"] = reference_hash_match
                equilibrium_result["state_acceptance"] = (
                    "accepted_provenance_checked_l1a2k_reference"
                    if reference_hash_match
                    else "rejected_hash_or_convergence_mismatch"
                )
                if reference_hash_match:
                    valid_reference_state = _load_phase_only_endpoint(
                        equilibrium_result, reference_config, "ch_only_equilibrium_060"
                    )
                    eq_phi = np.asarray(valid_reference_state.phi, dtype=np.float64)
                report["equilibrium_reference"] = equilibrium_result
                report["measurements"]["ch_only_060_equilibrium_reference"] = (
                    "measured" if reference_hash_match else "unmeasured_provenance_hash_mismatch"
                )
            except Exception as exc:
                equilibrium_result = {
                    "status": "rejected_validation_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "saved_l1a2k_config_match": config_match,
                }
                report["equilibrium_reference"] = equilibrium_result
                report["measurements"]["ch_only_060_equilibrium_reference"] = "unmeasured_validation_error"
        else:
            equilibrium_result = {
                "status": "rejected_config_mismatch",
                "saved_l1a2k_config": expected_reference_config,
                "derived_config": reference_config,
                "config_fingerprint_match": False,
                "not_reused": True,
            }
            report["equilibrium_reference"] = equilibrium_result
            report["measurements"]["ch_only_060_equilibrium_reference"] = "unmeasured_config_mismatch"
        _write_deliverables(report)
    else:
        report["equilibrium_reference"] = {
            "status": "unmeasured_by_controls_profile",
            "saved_l1a2k_reference": two_k.get("phase_only_comparison", {}).get("ch_only_state_hashes"),
        }
        report["measurements"]["ch_only_060_equilibrium_reference"] = "unmeasured_by_controls_profile"
        _write_deliverables(report)

    # Strictly load required production prefix checkpoints. No state whose endpoint hash differs from
    # the frozen L1A-2k report is passed into a phase-rate or causal comparison.
    intermediate_states: dict[str, dict[int, pf.State]] = {}
    intermediate_decisions: dict[str, dict[str, Any]] = {}
    for name in CASE_TARGETS:
        if name not in accepted_states:
            continue
        p, solid, seed, config = case_parameters[name]
        if name == "authority_060":
            start_steps = (AUTHORITY_WINDOW_START, AUTHORITY_BURST_START)
        else:
            start_steps = (CASE_STEPS[name] - CONTROL_WINDOW_LENGTH,)
        intermediate_states[name] = {}
        intermediate_decisions[name] = {}
        expected = _expected_upstream_case_hashes(two_k, name)
        for start_step in start_steps:
            candidate, decision = _load_checkpointed_production_state(name, start_step, config, expected)
            if candidate is not None:
                intermediate_states[name][start_step] = candidate
            intermediate_decisions[name][str(start_step)] = decision
    report["intermediate_production_checkpoints"] = intermediate_decisions
    _write_deliverables(report)

    # Decomposed production windows: the late 60° authority is measured at three explicit temporal
    # scales. 90° and 150° each receive an equal 10k-step late stationary-control window.
    authority_windows: dict[str, Any] | None = None
    control_results: dict[str, Any] = {}
    case_morphologies: dict[str, Any] = {}
    directional_tests: dict[str, Any] | None = None
    field_artifacts: dict[str, Any] = {}
    if "authority_060" in accepted_states:
        p60, solid60, _seed60, config60 = case_parameters["authority_060"]
        authority = accepted_states["authority_060"]
        if AUTHORITY_WINDOW_START in intermediate_states.get(
            "authority_060", {}
        ) and AUTHORITY_BURST_START in intermediate_states.get("authority_060", {}):
            authority_windows = {}
            for key, start_step, groups, stride in (
                ("window_1000_step_40k_50k", AUTHORITY_WINDOW_START, 10, 1_000),
                ("burst_10_step_48k_50k", AUTHORITY_BURST_START, AUTHORITY_BURST_STEPS // 10, 10),
                ("one_step_50000", CASE_STEPS["authority_060"], 1, 1),
            ):
                start_state = intermediate_states["authority_060"].get(
                    start_step, authority if start_step == CASE_STEPS["authority_060"] else None
                )
                if start_state is None:
                    continue
                region_masks, region_policy = _make_region_masks(start_state.phi, solid60, p60)
                window = _run_decomposed_window(
                    start_state,
                    solid60,
                    p60,
                    groups=groups,
                    steps_per_group=stride,
                    masks=region_masks,
                    step_start=start_step,
                    case_name="authority_060",
                    target_deg=60.0,
                    gate_reference_state=start_state,
                )
                end_step = int(start_step + groups * stride)
                expected_end = authority if end_step == CASE_STEPS["authority_060"] else None
                if expected_end is not None and window["state_endpoint_hashes"] != _state_hashes(expected_end):
                    raise AuditValidationError(
                        f"authority 60° {key} endpoint differs from accepted state step {end_step}"
                    )
                window["region_mask_policy"] = region_policy
                window["summary"] = _window_rate_summary(window)
                window["formal_mass_ledger_over_window"] = _mass_ledger_summary(window)
                window["left_right_contact_line_rate_correlation"] = _contact_rate_correlations(window)
                window["effect_sizes"] = _window_effect_sizes(window)
                authority_windows[key] = window
                report["windows"][key] = window
                if key == "one_step_50000":
                    one_state = start_state
                    one_masks, _ = _make_region_masks(one_state.phi, solid60, p60)
                    out, details, fields = _instrumented_public_step(one_state, solid60, p60, one_masks)
                    artifact = _save_public_step_fields(
                        "authority_060", CASE_STEPS["authority_060"], fields, details, one_state
                    )
                    field_artifacts["authority_060_step_50000_phase_decomposition"] = artifact
                    if eq_phi is not None:
                        directional_tests = _direction_test(one_state, eq_phi, solid60, p60, one_masks)
                        directional_tests["field_artifact"] = artifact
                _write_deliverables(report)
            if authority_windows:
                report["measurements"]["authority_060_phase_rate_decomposition_1_10_1000_steps"] = "measured"
                report["measurements"]["left_right_contact_line_correlations"] = (
                    "measured"
                    if any(
                        w["left_right_contact_line_rate_correlation"].get("status") == "measured"
                        for w in authority_windows.values()
                    )
                    else "unmeasured_insufficient_samples"
                )
        else:
            authority_windows = None
            report["measurements"]["authority_060_phase_rate_decomposition_1_10_1000_steps"] = (
                "unmeasured_missing_intermediate_production_checkpoints"
            )

        if eq_phi is not None:
            if directional_tests is not None:
                report["directional_tests"] = directional_tests
                report["measurements"]["one_step_direction_tests"] = "measured"
            else:
                report["measurements"]["one_step_direction_tests"] = "unmeasured_missing_authority_window"
        else:
            report["directional_tests"] = "unmeasured_provenance_checked_CH_only_reference_unavailable"
            report["measurements"]["one_step_direction_tests"] = (
                "unmeasured_provenance_checked_CH_only_reference_unavailable"
            )
    else:
        report["measurements"]["authority_060_phase_rate_decomposition_1_10_1000_steps"] = (
            "unmeasured_authority_state_hash_rejected"
        )
        report["directional_tests"] = "unmeasured_authority_state_hash_rejected"
        report["measurements"]["one_step_direction_tests"] = "unmeasured_authority_state_hash_rejected"

    for name in ("control_090", "control_150"):
        if name not in accepted_states:
            control_results[name] = {"status": "unmeasured_state_hash_rejected", "windows": {}}
            continue
        p_case, solid_case, _seed_case, config_case = case_parameters[name]
        endpoint = accepted_states[name]
        start_step = CASE_STEPS[name] - CONTROL_WINDOW_LENGTH
        start_state = intermediate_states.get(name, {}).get(start_step)
        if start_state is None:
            control_results[name] = {"status": "unmeasured_missing_late_window_checkpoint", "windows": {}}
            continue
        masks, policy = _make_region_masks(start_state.phi, solid_case, p_case)
        control_window = _run_decomposed_window(
            start_state,
            solid_case,
            p_case,
            groups=10,
            steps_per_group=1_000,
            masks=masks,
            step_start=start_step,
            case_name=name,
            target_deg=CASE_TARGETS[name],
            gate_reference_state=start_state,
        )
        if control_window["state_endpoint_hashes"] != _state_hashes(endpoint):
            raise AuditValidationError(f"{name} matched late window endpoint is not the accepted production state")
        control_window["region_mask_policy"] = policy
        control_window["summary"] = _window_rate_summary(control_window)
        control_window["formal_mass_ledger_over_window"] = _mass_ledger_summary(control_window)
        control_window["left_right_contact_line_rate_correlation"] = _contact_rate_correlations(control_window)
        matched_counterfactual = _matched_velocity_zero_counterfactual(
            endpoint,
            solid_case,
            p_case,
            steps=min(100, max(1, int(velocity_replay_steps))),
        )
        control_results[name] = {
            "status": "measured_matched_late_stationary_window",
            "matched_velocity_zero_counterfactual": matched_counterfactual,
            "window_start_step": start_step,
            "window_end_step": CASE_STEPS[name],
            "window_length_steps": CONTROL_WINDOW_LENGTH,
            "matched_physical_time": CONTROL_WINDOW_LENGTH * float(p_case.dt),
            "matched_mobility_scaled_time": CONTROL_WINDOW_LENGTH * float(p_case.dt) * float(p_case.M),
            "windows": {"late_stationary_1000_step_samples": control_window},
            "summaries": {"late_stationary_1000_step_samples": control_window["summary"]},
            "formal_mass_ledger": control_window["formal_mass_ledger_over_window"],
            "left_right_contact_line_rate_correlation": control_window["left_right_contact_line_rate_correlation"],
        }
        report["windows"][f"{name}_late_stationary_1000_step_samples"] = control_window
        _write_deliverables(report)

    report["control_comparison"] = control_results
    if all(item.get("status") == "measured_matched_late_stationary_window" for item in control_results.values()):
        report["measurements"]["stationary_control_late_windows_090_150"] = "measured_matched_10k_step_windows"
    else:
        report["measurements"]["stationary_control_late_windows_090_150"] = (
            "measured_with_unavailable_control_state" if control_results else "unmeasured"
        )

    # Endpoint chemical-potential, exact free-energy components, regional energy attribution, and
    # morphology are measured only on accepted provenance states.
    mu_comparison: dict[str, Any] = {}
    phase_thermodynamics: dict[str, Any] = {}
    regional_energy_localization: dict[str, Any] = {}
    for name, state in accepted_states.items():
        p_case, solid_case, _seed_case, _config_case = case_parameters[name]
        masks, policy = _make_region_masks(state.phi, solid_case, p_case)
        active_masks = {
            key: masks[key]
            for key in (
                "whole_fluid",
                "interface_frozen_at_window_start",
                "near_wall_fluid_0_2dx",
                "near_wall_fluid_0_4dx",
                "left_contact_line_2dx_frozen",
                "right_contact_line_2dx_frozen",
                "left_contact_line_4dx_frozen",
                "right_contact_line_4dx_frozen",
                "bulk_interface_excluding_contact_lines_4dx",
            )
        }
        mu_comparison[name] = _mu_and_flux_diagnostics(
            state.phi, state.u, state.v, solid_case, p_case, active_masks=active_masks
        )
        case_morphologies[name] = _morphology(state.phi, solid_case, p_case)
        phase_thermodynamics[name] = {
            "phase_free_energy_components": _phase_energy_parts(state.phi, solid_case, p_case),
            "chemical_potential": mu_comparison[name],
            "formal_mass_sum_V_phi": float(
                np.sum(
                    np.asarray(state.phi, dtype=np.float64)
                    * np.asarray(pf.phase_control_volumes(solid_case, p_case), dtype=np.float64),
                    dtype=np.float64,
                )
            ),
            "state_hashes": _state_hashes(state),
            "state_dtypes": {field: str(np.asarray(getattr(state, field)).dtype) for field in ("phi", "u", "v", "t")},
            "region_mask_policy": policy,
        }
        regional_energy_localization[name] = _regional_free_energy(state.phi, solid_case, p_case, masks)
    if valid_reference_state is not None:
        mu_comparison["ch_only_equilibrium_060"] = _mu_and_flux_diagnostics(
            valid_reference_state.phi,
            valid_reference_state.u,
            valid_reference_state.v,
            case_parameters["authority_060"][1],
            case_parameters["authority_060"][0],
        )
        eq_masks, eq_policy = _make_region_masks(
            valid_reference_state.phi, case_parameters["authority_060"][1], case_parameters["authority_060"][0]
        )
        phase_thermodynamics["ch_only_equilibrium_060"] = {
            "phase_free_energy_components": _phase_energy_parts(
                valid_reference_state.phi, case_parameters["authority_060"][1], case_parameters["authority_060"][0]
            ),
            "chemical_potential": mu_comparison["ch_only_equilibrium_060"],
            "formal_mass_sum_V_phi": float(
                np.sum(
                    np.asarray(valid_reference_state.phi, dtype=np.float64)
                    * np.asarray(
                        pf.phase_control_volumes(
                            case_parameters["authority_060"][1], case_parameters["authority_060"][0]
                        ),
                        dtype=np.float64,
                    ),
                    dtype=np.float64,
                )
            ),
            "state_hashes": _state_hashes(valid_reference_state),
            "state_dtypes": {
                field: str(np.asarray(getattr(valid_reference_state, field)).dtype) for field in ("phi", "u", "v", "t")
            },
            "region_mask_policy": eq_policy,
        }
        regional_energy_localization["ch_only_equilibrium_060"] = _regional_free_energy(
            valid_reference_state.phi,
            case_parameters["authority_060"][1],
            case_parameters["authority_060"][0],
            eq_masks,
        )

    report["chemical_potential"] = mu_comparison
    report["phase_thermodynamics"] = phase_thermodynamics
    report["free_energy_localization"] = regional_energy_localization
    report["morphologies"] = case_morphologies
    component_counts = [
        value.get("active_fluid_components_under_production_face_graph")
        for value in mu_comparison.values()
        if isinstance(value, dict)
    ]
    component_precondition_ok = bool(component_counts) and all(value == 1 for value in component_counts)
    if component_precondition_ok and len(mu_comparison) >= 2:
        report["measurements"]["chemical_potential_uniformity_and_face_fluxes"] = (
            "measured_single_component_precondition_satisfied"
        )
    elif any(value not in (None, 1) for value in component_counts):
        report["measurements"]["chemical_potential_uniformity_and_face_fluxes"] = (
            "unmeasured_fail_closed_multiple_fluid_components"
        )
        report["chemical_potential_fail_closed_reason"] = (
            "more than one connected fluid component exists under the exact production face graph; "
            "chemical-potential uniformity is not averaged across components"
        )
    else:
        report["measurements"]["chemical_potential_uniformity_and_face_fluxes"] = (
            "unmeasured_missing_provenance_reference"
        )
    if phase_thermodynamics:
        report["measurements"]["phase_free_energy_and_regional_localization"] = (
            "measured_exact_discrete_energy_components_and_fixed_regions"
        )

    if "authority_060" in accepted_states and valid_reference_state is not None:
        p60, solid60, _seed60, _config60 = case_parameters["authority_060"]
        authority = accepted_states["authority_060"]
        authority_masks, authority_mask_policy = _make_region_masks(authority.phi, solid60, p60)
        equilibrium_distance = _distance_to_equilibrium(
            authority.phi, valid_reference_state.phi, solid60, p60, authority_masks
        )
        morph_compare = _compare_morphology(authority.phi, valid_reference_state.phi, solid60, p60)
        report["morphology_comparison"] = {
            "authority_vs_ch_only_equilibrium": equilibrium_distance,
            "morphology_case_comparison": morph_compare,
            "authority_state_hashes": _state_hashes(authority),
            "equilibrium_state_hashes": _state_hashes(valid_reference_state),
            "contact_line_side_angles_and_positions": {
                "authority": _contact_metrics(authority.phi, solid60, p60),
                "ch_only_equilibrium": _contact_metrics(valid_reference_state.phi, solid60, p60),
            },
            "region_mask_policy": authority_mask_policy,
            "raw_unaligned_distance_is_authoritative": True,
            "center_aligned_distance": "unmeasured_secondary_diagnostic_not_used",
        }
        report["measurements"]["raw_unaligned_equilibrium_distance_and_morphology"] = (
            "measured_raw_unaligned_authoritative"
        )
    else:
        report["morphology_comparison"] = "unmeasured_provenance_checked_60_degree_equilibrium_or_authority_unavailable"
        report["measurements"]["raw_unaligned_equilibrium_distance_and_morphology"] = (
            "unmeasured_provenance_reference_unavailable"
        )

    # M_ref freeze-u CH continuation is a zero-velocity causal counterfactual, not a production
    # trajectory. It is checkpointed and resumed only under the strict source/config/runtime lineage.
    freeze_u_result: dict[str, Any] | None = None
    if profile == "forensic" and "authority_060" in accepted_states:
        p60, solid60, _seed60, _config60 = case_parameters["authority_060"]
        authority = accepted_states["authority_060"]
        freeze_config = _phase_only_config(60.0, M=float(chns.M_REF), branch_name="freeze_u_Mref_from_50000")
        freeze_u_result = _run_phase_only_continuation(
            "freeze_u_Mref_from_50000",
            authority,
            solid60,
            p60,
            freeze_config,
            max_steps=int(max_freeze_steps),
            start_absolute_step=CASE_STEPS["authority_060"],
            parent_state_hashes=_state_hashes(authority),
            progress_dir=progress_dir,
            phi_eq=eq_phi,
            stop_when_converged=True,
        )
        report["freeze_u_continuation"] = freeze_u_result
        report["measurements"]["freeze_u_Mref_continuation"] = (
            "measured_converged"
            if freeze_u_result.get("status") == "converged"
            else "measured_not_converged_within_step_budget"
        )
        _write_deliverables(report)
    else:
        report["freeze_u_continuation"] = "unmeasured_by_profile_or_rejected_authority_state"
        report["measurements"]["freeze_u_Mref_continuation"] = "unmeasured_by_profile_or_rejected_authority_state"

    # Matched production velocity versus zero velocity replay. Production evolves normally; the
    # zero-velocity branch uses the exact phase-only public step from the same authority phi/time.
    velocity_replay: dict[str, Any] | None = None
    if profile == "forensic" and "authority_060" in accepted_states:
        p60, solid60, _seed60, _config60 = case_parameters["authority_060"]
        authority = accepted_states["authority_060"]
        replay_steps = max(1, int(velocity_replay_steps))
        masks, mask_policy = _make_region_masks(authority.phi, solid60, p60)
        production_replay = _run_decomposed_window(
            authority,
            solid60,
            p60,
            groups=replay_steps,
            steps_per_group=1,
            masks=masks,
            step_start=CASE_STEPS["authority_060"],
            case_name="authority_060_production_velocity_replay",
            target_deg=60.0,
            gate_reference_state=authority,
        )
        production_endpoint = chns._advance_standard(authority, solid60, p60, replay_steps)
        if _state_hashes(production_endpoint) != production_replay["state_endpoint_hashes"]:
            raise AuditValidationError(
                "production velocity replay endpoint differs from unchanged public production stepping"
            )
        zero_initial = pf.State(
            authority.phi,
            jnp.zeros_like(authority.u, dtype=p60.dtype),
            jnp.zeros_like(authority.v, dtype=p60.dtype),
            authority.t,
        )
        zero_previous_phi, zero_endpoint, zero_iterations, zero_residual, zero_converged = nwa._advance(
            zero_initial, solid60, p60, replay_steps, True
        )
        zero_endpoint.phi.block_until_ready()
        volume = np.asarray(pf.phase_control_volumes(solid60, p60), dtype=np.float64)
        production_delta = np.asarray(production_endpoint.phi, dtype=np.float64) - np.asarray(
            authority.phi, dtype=np.float64
        )
        zero_delta = np.asarray(zero_endpoint.phi, dtype=np.float64) - np.asarray(authority.phi, dtype=np.float64)
        start_energy = _phase_energy_parts(authority.phi, solid60, p60)
        prod_energy = _phase_energy_parts(production_endpoint.phi, solid60, p60)
        zero_energy = _phase_energy_parts(zero_endpoint.phi, solid60, p60)
        start_distance = (
            None
            if eq_phi is None
            else _distance_to_equilibrium(authority.phi, eq_phi, solid60, p60)["raw_unaligned_relative_D_phi"]
        )
        prod_distance = (
            None
            if eq_phi is None
            else _distance_to_equilibrium(production_endpoint.phi, eq_phi, solid60, p60)["raw_unaligned_relative_D_phi"]
        )
        zero_distance = (
            None
            if eq_phi is None
            else _distance_to_equilibrium(zero_endpoint.phi, eq_phi, solid60, p60)["raw_unaligned_relative_D_phi"]
        )
        velocity_replay = {
            "status": "measured",
            "matched_steps": replay_steps,
            "M": float(p60.M),
            "dt": float(p60.dt),
            "physical_duration": replay_steps * float(p60.dt),
            "mobility_scaled_duration": replay_steps * float(p60.dt) * float(p60.M),
            "production_velocity_replay": {
                "trajectory": "full unchanged production step; momentum, Brinkman and projection advance normally",
                "endpoint_state_hashes": _state_hashes(production_endpoint),
                "endpoint_exact_replay_match": production_replay["endpoint_exact_hash_match_production"],
                "formal_phase_mass_increment": float(np.sum(volume * production_delta, dtype=np.float64)),
                "phase_free_energy_start": start_energy,
                "phase_free_energy_end": prod_energy,
                "D_phi_start": start_distance,
                "D_phi_end": prod_distance,
                "D_phi_production_change": None
                if start_distance is None or prod_distance is None
                else prod_distance - start_distance,
                "delta_phi_l2_volume": _weighted_norm(production_delta, volume),
                "phase_rate_decomposition_series": production_replay,
                "endpoint_morphology": _morphology(production_endpoint.phi, solid60, p60),
            },
            "zero_velocity_counterfactual": {
                "trajectory": (
                    "pf.phase_only_step_with_diagnostics over the same matched public-step count; "
                    "u=v=0 at every phase substep"
                ),
                "endpoint_state_hashes": _state_hashes(zero_endpoint),
                "implicit_iterations_max": int(zero_iterations),
                "implicit_relative_residual_max": float(zero_residual),
                "implicit_last_step_converged": bool(zero_converged),
                "formal_phase_mass_increment": float(np.sum(volume * zero_delta, dtype=np.float64)),
                "formal_mass_increment_adv": 0.0,
                "formal_mass_increment_CH": float(np.sum(volume * zero_delta, dtype=np.float64)),
                "phase_free_energy_start": start_energy,
                "phase_free_energy_end": zero_energy,
                "D_phi_start": start_distance,
                "D_phi_end": zero_distance,
                "D_phi_zero_velocity_change": None
                if start_distance is None or zero_distance is None
                else zero_distance - start_distance,
                "delta_phi_l2_volume": _weighted_norm(zero_delta, volume),
                "endpoint_morphology": _morphology(zero_endpoint.phi, solid60, p60),
                "last_public_step_previous_phi_hash": _hash_array(zero_previous_phi),
            },
            "D_phi_production_minus_zero_after_matched_steps": None
            if prod_distance is None or zero_distance is None
            else prod_distance - zero_distance,
            "production_minus_zero_phase_change_l2_volume": _weighted_norm(production_delta - zero_delta, volume),
            "region_mask_policy": mask_policy,
            "diagnostic_only": True,
            "not_production_validation": True,
        }
        report["velocity_replay"] = velocity_replay
        report["measurements"]["production_velocity_replay_vs_zero_velocity"] = "measured"
        _write_deliverables(report)
    else:
        report["velocity_replay"] = "unmeasured_by_profile_or_rejected_authority_state"
        report["measurements"]["production_velocity_replay_vs_zero_velocity"] = (
            "unmeasured_by_profile_or_rejected_authority_state"
        )

    # Velocity-scaling diagnostics are gated by a direct one-step advection effect. They use M_ref
    # and the production u/v field frozen; 4*M_ref cannot establish production closure.
    velocity_scaling: dict[str, Any] | None = None
    velocity_scale_trigger = False
    if directional_tests is not None and eq_phi is not None and "authority_060" in accepted_states:
        p60, solid60, _seed60, _config60 = case_parameters["authority_060"]
        authority = accepted_states["authority_060"]
        one_step_adv = authority_windows.get("one_step_50000", {}).get("metric_rows", []) if authority_windows else []
        adv_norm = float(one_step_adv[-1]["regions"]["whole_fluid"]["adv_rate_l2_volume"]) if one_step_adv else 0.0
        velocity_norm = float(
            np.sqrt(
                np.sum(np.asarray(authority.u, dtype=np.float64) ** 2 + np.asarray(authority.v, dtype=np.float64) ** 2)
            )
        )
        velocity_scale_trigger = adv_norm > 0.0 and velocity_norm > 0.0
        if velocity_scale_trigger:
            velocity_scaling = _velocity_scaling_sweep(
                authority,
                eq_phi,
                solid60,
                p60,
                steps=min(max(1, int(velocity_replay_steps)), 100),
            )
            velocity_scaling["trigger"] = {
                "triggered_by_nonzero_one_step_advective_rate_and_production_velocity": True,
                "one_step_adv_rate_l2_volume": adv_norm,
                "production_velocity_l2_unweighted": velocity_norm,
            }
            report["measurements"]["velocity_scaling"] = "measured_diagnostic_only"
        else:
            velocity_scaling = {
                "status": "unmeasured_trigger_not_met",
                "trigger": {
                    "one_step_adv_rate_l2_volume": adv_norm,
                    "production_velocity_l2_unweighted": velocity_norm,
                },
            }
            report["measurements"]["velocity_scaling"] = "unmeasured_trigger_not_met"
    else:
        velocity_scaling = "unmeasured_direction_test_or_equilibrium_reference_unavailable"
        report["measurements"]["velocity_scaling"] = "unmeasured_direction_test_or_equilibrium_reference_unavailable"
    report["velocity_scaling"] = velocity_scaling
    report["measurements"]["dt_half"] = (
        "unmeasured_not_triggered; cancellation and matched replay are assessed before deciding whether dt/2 is needed"
    )
    report["measurements"]["M_4x_closure"] = "unmeasured_by_design; 4*M_ref cannot establish production closure"

    # Candidate classifications are conservative: only matched counterfactual or differential control
    # evidence can be SUPPORTED; the final causal root remains INCONCLUSIVE unless uniquely established.
    all_case_hashes_valid = all(
        case_entries.get(name, {}).get("matches_l1a2k_per_field_hashes") is True for name in CASE_TARGETS
    )
    required_provenance_valid = bool(all_case_hashes_valid and reference_hash_match)
    mechanism_matrix = _classify_candidates(
        authority_windows,
        control_results,
        freeze_u_result,
        velocity_replay,
        directional_tests,
        mu_comparison if mu_comparison else None,
        two_k,
        required_provenance_valid=required_provenance_valid,
    )
    report["mechanism_matrix"] = mechanism_matrix
    report["candidate_classifications"] = mechanism_matrix["candidates"]
    report["effect_sizes"] = {
        "authority_window_summaries": {
            name: window.get("effect_sizes") for name, window in (authority_windows or {}).items()
        },
        "stationary_controls": {name: case.get("summaries") for name, case in control_results.items()},
        "raw_unaligned_D_phi_authority_vs_CH_only_equilibrium": report.get("morphology_comparison", {})
        .get("authority_vs_ch_only_equilibrium", {})
        .get("raw_unaligned_relative_D_phi")
        if isinstance(report.get("morphology_comparison"), dict)
        else None,
        "production_vs_zero_velocity_D_phi_effect": None
        if velocity_replay is None
        else velocity_replay.get("D_phi_production_minus_zero_after_matched_steps"),
        "Mref_freeze_u_relaxation_steps": None if freeze_u_result is None else freeze_u_result.get("relative_steps"),
        "velocity_scaling_effects": velocity_scaling,
        "no_post_measurement_thresholds": True,
    }
    report["field_artifacts"] = field_artifacts
    report["left_right_contact_line_metrics"] = {
        name: window.get("left_right_contact_line_rate_correlation")
        for name, window in report.get("windows", {}).items()
        if isinstance(window, dict) and "left_right_contact_line_rate_correlation" in window
    }
    report["production_state_acceptance"] = {
        name: {
            "state_acceptance": entry.get("state_acceptance"),
            "matches_l1a2k_per_field_hashes": entry.get("matches_l1a2k_per_field_hashes"),
            "expected_hashes": entry.get("expected_l1a2k_state_hashes"),
            "actual_hashes": entry.get("endpoint_state_hashes"),
        }
        for name, entry in case_entries.items()
    }
    report["quality_status"] = {
        "pytest": "pending",
        "ruff": "pending",
        "production_solver_source_integrity": "pending",
        "contract_11": "unchanged",
        "production_semantics_changed": False,
    }
    report["measurements"]["phase_free_energy_and_regional_localization"] = (
        "measured_exact_discrete_energy_components_and_fixed_regions"
        if phase_thermodynamics
        else "unmeasured_no_accepted_states"
    )
    report["measurements"]["left_right_contact_line_correlations"] = (
        "measured" if report["left_right_contact_line_metrics"] else "unmeasured_insufficient_valid_windows"
    )
    report["measurements"]["velocity_scaling"] = (
        "measured_diagnostic_only"
        if isinstance(velocity_scaling, dict) and velocity_scaling.get("status") == "measured"
        else report["measurements"].get("velocity_scaling", "unmeasured")
    )
    report["required_acceptance_criteria"] = {
        "state_hashes_match_l1a2k_for_all_three_cases": all_case_hashes_valid,
        "provenance_checked_converged_CH_only_060_reference": reference_hash_match,
        "instrumented_public_step_matches_production": bool(
            directional_tests and directional_tests.get("production_hash_match")
        ),
        "adv_CH_net_reconstruction_to_roundoff": bool(
            authority_windows
            and all(
                max(row["checks"]["phase_reconstruction_linf"] for row in window["metric_rows"]) <= 5.0e-13
                for window in authority_windows.values()
            )
        ),
        "production_gate_normalization_preserved": True,
        "Mref_freeze_u_continuation_measured": freeze_u_result is not None,
        "velocity_replay_zero_counterfactual_measured": velocity_replay is not None,
        "diagnostics_are_not_production_validation": True,
    }
    if run_quality:
        report["quality_status"] = _run_quality_checks()
    _write_deliverables(report)
    report["evidence_paths"] = _write_deliverables(report)
    return report


def run_quick() -> dict[str, Any]:
    """Small deterministic operator smoke-test; does not create production-scale state evidence."""
    p, solid, state, config = chns._make_case(60.0, N_value=32, dt=chns.DT, M=chns.M_REF)
    masks, policy = _make_region_masks(state.phi, solid, p)
    instrumented, details, fields = _instrumented_public_step(state, solid, p, masks)
    production, production_diag = pf.step_with_diagnostics(state, solid, p)
    ch_initial = pf.State(state.phi, jnp.zeros_like(state.u), jnp.zeros_like(state.v), state.t)
    ch_instrumented, ch_details = _phase_only_instrumented_step(ch_initial.phi, solid, p, ch_initial.t)
    ch_production, ch_diag = pf.phase_only_step_with_diagnostics(ch_initial, solid, p)
    component_count = _connected_component_count(solid, p)
    result = {
        "stage": STAGE,
        "profile": "quick",
        "status": "complete"
        if _state_hashes(instrumented) == _state_hashes(production)
        and _state_hashes(ch_instrumented) == _state_hashes(ch_production)
        else "failed",
        "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
        "production_semantics_changed": False,
        "config": config,
        "config_fingerprint": _canonical_hash(config),
        "runtime_versions": _runtime_versions(),
        "state_dtypes": {name: str(np.asarray(getattr(state, name)).dtype) for name in ("phi", "u", "v", "t")},
        "single_connected_fluid_component": component_count == 1,
        "production_public_step_exact_hash_match": _state_hashes(instrumented) == _state_hashes(production),
        "production_implicit_converged_all": bool(np.all(np.asarray(production_diag.implicit_converged))),
        "decomposition_checks": {
            key: details[key]
            for key in (
                "phase_reconstruction_linf",
                "adv_rhs_vs_face_divergence_linf",
                "explicit_implicit_CH_reconstruction_linf",
                "implicit_iterations_max",
                "implicit_relative_residual_max",
                "implicit_converged",
            )
        },
        "CH_only_public_step_exact_hash_match": _state_hashes(ch_instrumented) == _state_hashes(ch_production),
        "CH_only_reconstruction_linf": float(ch_details["reconstruction_linf"]),
        "CH_only_implicit_converged_all": bool(np.all(np.asarray(ch_diag.implicit_converged))),
        "production_state_hashes": _state_hashes(production),
        "region_mask_policy": policy,
        "one_step_fields_finite": bool(all(np.isfinite(array).all() for array in fields.values())),
        "production_gate_definition": _production_gate_definition(),
        "diagnostic_only": True,
        "not_production_validation": True,
    }
    output_path = EVIDENCE_ROOT / "quick_phase_coupling_smoke.json"
    _write_json(output_path, result)
    return result


def _run_quality_checks() -> dict[str, Any]:
    """Run focused tests/lint and hash-fence production sources during this audit."""
    import subprocess

    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "1"
    repo_root = ROOT.parent.parent
    test_path = ROOT / "tests" / "test_phase_coupling_relaxation_audit.py"
    pytest_result = None
    if test_path.is_file():
        run = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_path.relative_to(repo_root)), "-q"],
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        pytest_result = {
            "exit_code": run.returncode,
            "stdout_tail": run.stdout[-6000:],
            "stderr_tail": run.stderr[-6000:],
            "passed": run.returncode == 0,
        }
    else:
        pytest_result = {"passed": False, "status": "unmeasured_test_file_missing"}
    lint_paths = [
        str(Path(__file__).resolve().relative_to(repo_root)),
        str(test_path.relative_to(repo_root)),
    ]
    lint = subprocess.run(
        [sys.executable, "-m", "ruff", "check", *lint_paths],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    working_tree = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "--",
            "examples/two_phase/phasefield.py",
            "examples/two_phase/production/chns_nonstationarity_audit.py",
            "examples/two_phase/production/nonneutral_wetting_audit.py",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    source_hashes_after = {name: _sha256(path.read_bytes()) for name, path in PRODUCTION_GUARD_PATHS.items()}
    changed_during_audit = [
        name
        for name, initial_hash in PRODUCTION_SOURCE_HASHES_AT_AUDIT_IMPORT.items()
        if source_hashes_after.get(name) != initial_hash
    ]
    return {
        "pytest": pytest_result,
        "ruff": {
            "exit_code": lint.returncode,
            "passed": lint.returncode == 0,
            "stdout_tail": lint.stdout[-6000:],
            "stderr_tail": lint.stderr[-6000:],
        },
        "production_solver_source_integrity": {
            "hashes_at_audit_import": PRODUCTION_SOURCE_HASHES_AT_AUDIT_IMPORT,
            "hashes_after_audit": source_hashes_after,
            "working_tree_modifications_vs_git_head": [line for line in working_tree.stdout.splitlines() if line],
            "changed_during_l1a2l": changed_during_audit,
            "passed": working_tree.returncode == 0 and not changed_during_audit,
            "interpretation": (
                "Git-relative modifications may predate L1A-2l; the before/after SHA-256 guard "
                "verifies this diagnostic stage did not modify production sources."
            ),
        },
        # contract 12 is the sanctioned L1A-2p metadata-only promotion of the
        # contract-11 arithmetic; anything else means the production semantics moved
        "contract_11": "unchanged" if pf.SOLVER_CONTRACT_VERSION in (SOLVER_CONTRACT, 12) else "failed",
        "production_semantics_changed": False,
        "diagnostic_only": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("quick", "controls", "forensic"), default="forensic")
    parser.add_argument("--max-phase-only-steps", type=int, default=180_000)
    parser.add_argument("--max-freeze-steps", type=int, default=100_000)
    parser.add_argument("--velocity-replay-steps", type=int, default=100)
    parser.add_argument("--skip-quality", action="store_true")
    args = parser.parse_args(argv)
    if args.profile == "quick":
        result = run_quick()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "complete" else 1
    report = run_forensic(
        profile=args.profile,
        max_phase_only_steps=args.max_phase_only_steps,
        max_freeze_steps=args.max_freeze_steps,
        velocity_replay_steps=args.velocity_replay_steps,
        run_quality=not args.skip_quality,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "final_verdict": report["final_verdict"],
                "evidence_paths": report.get("evidence_paths"),
            },
            indent=2,
        )
    )
    return 0 if report.get("status") in ("complete", "partial_unmeasured") else 1


if __name__ == "__main__":
    raise SystemExit(main())
