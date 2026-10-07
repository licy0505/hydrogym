"""L1A-2m diagnostic-only stationarity metric-domain forensics (contract 11).

Question: is the 60-degree production phase-rate gate measuring true physical
nonstationarity, classifier-only activity in zero/nonphysical-volume cells, or
hidden coupling from those cells back into the physical solver?

The audit never edits the contract-11 production solver, the production
phase-rate definition, or any acceptance threshold.  It instruments the exact
field the production gate consumes -- ``r_i = (phi_after_public_step -
phi_before_public_step) / dt`` evaluated with the unchanged
``pf.step_with_diagnostics`` -- and decomposes the formal full-grid norm
``R_prod = sqrt(sum_i(r_i^2) * dx * dy)`` over exact contract-11 control-volume
support classes built from ``V_i = pf.phase_control_volumes``.  A shadow
physical-domain metric ``R_V = sqrt(sum_i(V_i * r_i^2))`` is reported next to it
and never replaces the gate.  Inactive-state causality is measured with
counterfactual copies that perturb ``phi`` only on ``V_i == 0`` cells.

Run from ``examples/two_phase``::

    JAX_ENABLE_X64=1 python -m production.stationarity_metric_domain_audit --profile quick
    JAX_ENABLE_X64=1 python -m production.stationarity_metric_domain_audit --profile forensic --out artifacts/l1a2m

Large/resumable arrays stay under ``artifacts/l1a2m``; versioned evidence is
written to ``evidence/l1a2m``.  This stage must not change the production
convergence gate: the production verdict remains authoritative and any future
threshold belongs to a separate classifier-repair stage.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import platform
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import phasefield as pf
from production import capillary_pressure_balance_audit as l1a2k
from production import chns_nonstationarity_audit as chns
from production import contact_line_kinetics as clk
from production import nonneutral_wetting_audit as nwa
from production import observables as obs
from production import phase_coupling_relaxation_audit as l1a2l

jax.config.update("jax_enable_x64", True)

STAGE = "L1A-2m"
SOLVER_CONTRACT = 11
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts" / "l1a2m"
EVIDENCE_ROOT = ROOT / "evidence" / "l1a2m"
UPSTREAM_2J = l1a2l.UPSTREAM_2J
UPSTREAM_2K = l1a2l.UPSTREAM_2K
UPSTREAM_2L_REPORT = ROOT / "evidence" / "l1a2l" / "phase_coupling_relaxation_report.json"
UPSTREAM_2L_MANIFEST = ROOT / "evidence" / "l1a2l" / "manifest.json"

#: Frozen provenance states (section 3).  The three CHNS states are the L1A-2k
#: staged trajectories; the CH-only equilibrium is the L1A-2k converged
#: ``ch_only_060_converged`` reference.
CASE_TARGETS = {"authority_060": 60.0, "control_090": 90.0, "control_150": 150.0}
CASE_STEPS = {"authority_060": 50_000, "control_090": 27_200, "control_150": 100_000}
CH_ONLY_STEP = 180_000
CH_ONLY_MAX_STEPS = 300_000
CH_ONLY_CHUNK_STEPS = 10_000

#: Late-window policy (section 9): the frozen forensic cadence is 1000 steps and
#: the forensic windows are the last 10000 public steps of each trajectory.
WINDOW_LENGTH_STEPS = 10_000
SAMPLE_CADENCE_STEPS = 1_000
#: Denser available sampling for the 60-degree angle-spread audit (section 18).
DENSE_WINDOW_STEPS = 2_000
DENSE_CADENCE_STEPS = 100

#: Recorded, never enforced against R_V (section 17): the 0.001 tolerance belongs
#: to the production full-grid metric only.
PRODUCTION_PHASE_RATE_TOL = float(nwa.CRITERIA["phase_rate_l2_tol"])
PRODUCTION_ANGLE_TOL_DEG = float(nwa.CRITERIA["angle_tol_deg"])

#: Support-class energy-fraction interpretation bands (report-only wording).
ZERO_VOLUME_DOMINANT_FRACTION = 0.5
ZERO_VOLUME_MATERIAL_FRACTION = 0.1
ZERO_VOLUME_SMALL_FRACTION = 0.01
CONTROL_LIKE_RATIO_LIMIT = 3.0
CONTROL_LIKE_EFFECT_SIZE_LIMIT = 3.0

SUPPORT_CLASS_NAMES = ("ZERO_VOLUME", "PARTIAL_VOLUME", "FULL_VOLUME")


class AuditValidationError(RuntimeError):
    """Raised when a fail-closed provenance, admissibility, or schema check does not pass."""


# ---------------------------------------------------------------------------
# provenance helpers (delegated to the frozen L1A-2l helpers where possible)
# ---------------------------------------------------------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return _sha256(array.tobytes() + str(array.dtype).encode() + str(array.shape).encode())


def _state_hashes(state: pf.State) -> dict[str, str]:
    return {name: _hash_array(np.asarray(getattr(state, name))) for name in ("phi", "u", "v", "t")}


def _canonical_hash(value: Any) -> str:
    return l1a2l._canonical_hash(value)


def _json_clean(value: Any) -> Any:
    return l1a2l._json_clean(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_json_clean(value), indent=1, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _runtime_versions() -> dict[str, str]:
    return l1a2l._runtime_versions()


AUDITED_SOURCE_MODULES = {
    "phasefield": ROOT / "phasefield.py",
    "chns_nonstationarity_audit": ROOT / "production" / "chns_nonstationarity_audit.py",
    "nonneutral_wetting_audit": ROOT / "production" / "nonneutral_wetting_audit.py",
    "contact_line_kinetics": ROOT / "production" / "contact_line_kinetics.py",
    "observables": ROOT / "production" / "observables.py",
    "capillary_pressure_balance_audit": ROOT / "production" / "capillary_pressure_balance_audit.py",
    "phase_coupling_relaxation_audit": ROOT / "production" / "phase_coupling_relaxation_audit.py",
}


def _file_sha256(path: Path) -> str:
    return _sha256(Path(path).read_bytes())


def _source_hashes() -> dict[str, str]:
    hashes = {name: _file_sha256(path) for name, path in AUDITED_SOURCE_MODULES.items()}
    hashes["stationarity_metric_domain_audit"] = _file_sha256(Path(__file__).resolve())
    return hashes


def _binding_source_hashes() -> dict[str, str]:
    """The frozen production sources saved states are bound to (the diagnostic module itself evolves)."""
    return {name: _file_sha256(path) for name, path in AUDITED_SOURCE_MODULES.items()}


def _frozen_source_hashes() -> dict[str, str]:
    """Source hashes frozen in the L1A-2l manifest for the shared production sources."""
    manifest = _load_json(UPSTREAM_2L_MANIFEST)
    return dict(manifest["source_hashes"])


def _validate_upstream() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate the frozen L1A-2j/L1A-2k reports and the L1A-2l manifest/report anchoring."""
    two_j, two_k = l1a2l._validate_upstream()
    if not UPSTREAM_2L_MANIFEST.is_file() or not UPSTREAM_2L_REPORT.is_file():
        raise AuditValidationError("frozen L1A-2l manifest/report are required upstream evidence for L1A-2m")
    two_l = _load_json(UPSTREAM_2L_REPORT)
    manifest = _load_json(UPSTREAM_2L_MANIFEST)
    if two_l.get("stage") != "L1A-2l" or int(two_l.get("solver_contract_version", -1)) != SOLVER_CONTRACT:
        raise AuditValidationError("frozen L1A-2l report has an unexpected stage or contract")
    if str(two_l.get("final_verdict")) != "INCONCLUSIVE":
        raise AuditValidationError("the frozen L1A-2l verdict is expected to be INCONCLUSIVE")
    frozen = _frozen_source_hashes()
    current = _source_hashes()
    mismatched = sorted(
        name
        for name in (
            "phasefield",
            "chns_nonstationarity_audit",
            "nonneutral_wetting_audit",
            "contact_line_kinetics",
            "observables",
            "capillary_pressure_balance_audit",
        )
        if frozen.get(name) != current.get(name)
    )
    if mismatched:
        raise AuditValidationError(f"production sources differ from the frozen L1A-2l anchor: {mismatched}")
    for name in ("upstream_l1a2j_report", "upstream_l1a2k_report"):
        path = UPSTREAM_2J if name == "upstream_l1a2j_report" else UPSTREAM_2K
        if manifest["artifacts"] and frozen.get(name) and _file_sha256(path) != frozen[name]:
            raise AuditValidationError(f"the frozen {name} no longer matches the L1A-2l manifest hash")
    return two_j, two_k, two_l


def _expected_upstream_case_hashes(two_k: dict[str, Any], case_name: str) -> dict[str, str]:
    return l1a2l._expected_upstream_case_hashes(two_k, case_name)


def _expected_ch_only_hashes(two_k: dict[str, Any]) -> dict[str, str]:
    phase = two_k.get("phase_only_comparison", {})
    if phase.get("ch_only_case") != "ch_only_060_converged" or int(phase.get("ch_only_step", -1)) != CH_ONLY_STEP:
        raise AuditValidationError("frozen L1A-2k report lacks the converged ch_only_060 reference at step 180000")
    return dict(phase["ch_only_state_hashes"])


def _make_case(target: float) -> tuple[pf.PhaseFieldParams, pf.Solid, pf.State, dict[str, Any]]:
    return l1a2l._make_case(target)


def _state_arrays(state: pf.State) -> dict[str, np.ndarray]:
    return {name: np.array(getattr(state, name), copy=True) for name in ("phi", "u", "v", "t")}


def _state_from_arrays(arrays: dict[str, np.ndarray], p: pf.PhaseFieldParams) -> pf.State:
    return pf.State(
        phi=jnp.asarray(arrays["phi"], dtype=jnp.float64),
        u=jnp.asarray(arrays["u"], dtype=p.dtype),
        v=jnp.asarray(arrays["v"], dtype=p.dtype),
        t=jnp.asarray(np.asarray(arrays["t"]).item(), dtype=p.dtype),
    )


# ---------------------------------------------------------------------------
# strict L1A-2m checkpoint I/O (same discipline as the frozen L1A-2l store)
# ---------------------------------------------------------------------------


def _save_checkpoint(
    path: Path,
    state: pf.State,
    *,
    case_name: str,
    step: int,
    config: dict[str, Any],
    upstream_state_hashes: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "stage": STAGE,
        "section": "production_state",
        "case": case_name,
        "step": int(step),
        "git_sha": chns.get_git_sha(),
        "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
        "config": config,
        "config_fingerprint": _canonical_hash(config),
        "source_hashes": _binding_source_hashes(),
        "runtime_versions": _runtime_versions(),
        "state_hashes": _state_hashes(state),
        "upstream_state_hashes": upstream_state_hashes,
        "production_semantics_changed": False,
        **(extra or {}),
    }
    arrays = _state_arrays(state)
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    return metadata


def _load_checkpoint(
    path: Path,
    *,
    case_name: str,
    step: int,
    config: dict[str, Any],
    p: pf.PhaseFieldParams,
    upstream_state_hashes: dict[str, str] | None = None,
) -> tuple[pf.State, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"phi", "u", "v", "t", "metadata_json"}
        if required - set(archive.files):
            raise ValueError("checkpoint arrays are incomplete")
        arrays = {name: np.array(archive[name], copy=True) for name in ("phi", "u", "v", "t")}
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
    expected = {
        "stage": STAGE,
        "section": "production_state",
        "case": case_name,
        "step": int(step),
        "git_sha": chns.get_git_sha(),
        "solver_contract_version": SOLVER_CONTRACT,
        "config": config,
        "config_fingerprint": _canonical_hash(config),
        "source_hashes": _binding_source_hashes(),
        "runtime_versions": _runtime_versions(),
        "production_semantics_changed": False,
    }
    checks = {name: metadata.get(name) == value for name, value in expected.items()}
    stored_hashes = metadata.get("state_hashes")
    checks["state_hashes"] = stored_hashes == {name: _hash_array(value) for name, value in arrays.items()}
    if upstream_state_hashes is not None and int(step) == int(metadata.get("step", -1)):
        checks["upstream_match"] = stored_hashes == upstream_state_hashes
    if not all(checks.values()):
        raise ValueError(f"strict L1A-2m checkpoint validation failed: {checks}")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("checkpoint contains non-finite arrays")
    if arrays["phi"].shape != (int(config["Nx"]), int(config["Ny"])):
        raise ValueError("checkpoint phase shape differs from the frozen grid")
    return _state_from_arrays(arrays, p), metadata


def _save_sample_arrays(path: Path, row: dict[str, Any], case_name: str) -> None:
    arrays = {name: np.asarray(row[name]) for name in ("phi", "phi_prev", "u", "v")}
    arrays["t"] = np.asarray(row["t"])
    meta = {key: value for key, value in row.items() if key not in arrays}
    meta["stage"] = STAGE
    meta["case"] = case_name
    meta["source_hashes"] = _source_hashes()
    arrays["metadata_json"] = np.asarray(
        json.dumps(_json_clean(meta), sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        np.savez_compressed(handle, **arrays)


def _load_sample_arrays(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"phi", "phi_prev", "u", "v", "t", "metadata_json"}
        if required - set(archive.files):
            raise ValueError("sample arrays are incomplete")
        row = {name: np.array(archive[name], copy=True) for name in ("phi", "phi_prev", "u", "v", "t")}
        row["metadata"] = json.loads(str(np.asarray(archive["metadata_json"]).item()))
    for name in ("phi", "phi_prev", "u", "v", "t"):
        if not np.isfinite(row[name]).all():
            raise ValueError(f"sample array {name} contains non-finite values")
    return row


# ---------------------------------------------------------------------------
# rehydration of the frozen provenance states with integrated window sampling
# ---------------------------------------------------------------------------


def _sample_steps(end: int, *, window: int, cadence: int, dense_window: int = 0, dense_cadence: int = 0) -> list[int]:
    """Cadence samples of the late window, endpoint included; dense extras never replace the cadence."""
    start = end - window
    steps = {step for step in range(start + cadence, end + 1, cadence)}
    steps.add(end)
    if dense_window > 0 and dense_cadence > 0:
        steps |= {step for step in range(end - dense_window, end + 1, dense_cadence)}
    return sorted(steps)


def _rehydrate_chns_case(
    case_name: str,
    target: float,
    two_k: dict[str, Any],
    artifact_root: Path,
    *,
    dense_window: int = 0,
    dense_cadence: int = 0,
) -> dict[str, Any]:
    """Resume or rerun the exact production trajectory and sample its late window.

    The trajectory is advanced exclusively with ``chns._advance_standard`` (the
    unchanged ``pf.step_with_diagnostics`` public step); sampling only reads host
    copies of the sampled states, so a sampled run and a bare run share bitwise
    endpoints.  The endpoint is accepted only when all four per-field hashes match
    the frozen L1A-2k report.
    """
    p, solid, seed, config = _make_case(target)
    expected = _expected_upstream_case_hashes(two_k, case_name)
    end = CASE_STEPS[case_name]
    window_start = end - WINDOW_LENGTH_STEPS
    checkpoint_dir = artifact_root / "checkpoints"
    sample_dir = artifact_root / "samples"
    final_path = checkpoint_dir / f"{case_name}_production_step_{end:06d}.npz"
    progress_path = checkpoint_dir / f"{case_name}_trajectory_progress.json"

    state: pf.State | None = None
    reuse = {"endpoint_checkpoint": "absent", "window_samples": "absent"}
    if final_path.is_file():
        try:
            state, _meta = _load_checkpoint(
                final_path, case_name=case_name, step=end, config=config, p=p, upstream_state_hashes=expected
            )
            reuse["endpoint_checkpoint"] = "reused_strict_l1a2m_checkpoint"
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            reuse["endpoint_checkpoint"] = f"rejected: {exc}"
            state = None

    wanted_steps = _sample_steps(
        end,
        window=WINDOW_LENGTH_STEPS,
        cadence=SAMPLE_CADENCE_STEPS,
        dense_window=dense_window,
        dense_cadence=dense_cadence,
    )
    sample_rows: dict[int, dict[str, Any]] = {}
    missing_samples: list[int] = []
    if state is not None:
        for step in wanted_steps:
            path = sample_dir / f"{case_name}_sample_step_{step:06d}.npz"
            if path.is_file():
                try:
                    sample_rows[step] = _load_sample_arrays(path)
                except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                    print(f"[{STAGE}] reject sample {case_name}@{step}: {exc}", flush=True)
                    missing_samples.append(step)
            else:
                missing_samples.append(step)
        if missing_samples:
            # Samples are recorded along the trajectory only; a partial sample set
            # cannot be reconstructed from the endpoint alone.
            reuse["window_samples"] = f"incomplete_missing_{len(missing_samples)}"
            state = None
        else:
            reuse["window_samples"] = f"reused_{len(wanted_steps)}_samples"

    replay_record: dict[str, Any] = {}
    if state is None:
        started = time.perf_counter()
        latest, current = seed, 0
        if progress_path.is_file():
            saved = _load_json(progress_path)
            try:
                candidate_step = int(saved.get("committed_step", -1))
                candidate_path = ROOT / saved["checkpoint"]
                if (
                    saved.get("stage") == STAGE
                    and saved.get("case") == case_name
                    and saved.get("target_step") == end
                    and saved.get("config_fingerprint") == _canonical_hash(config)
                    and saved.get("source_hashes") == _binding_source_hashes()
                    and saved.get("runtime_versions") == _runtime_versions()
                    and 0 < candidate_step <= window_start
                ):
                    latest, current = _load_checkpoint(
                        candidate_path, case_name=case_name, step=candidate_step, config=config, p=p
                    )
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                print(f"[{STAGE}] discard trajectory resume for {case_name}: {exc}", flush=True)
                latest, current = seed, 0
        replay_record["resumed_from_step"] = int(current)
        while current < window_start:
            chunk = min(1_000, window_start - current)
            latest = chns._advance_standard(latest, solid, p, chunk)
            current += chunk
            if current % 10_000 == 0 or current == window_start:
                meta = _save_checkpoint(
                    checkpoint_dir / f"{case_name}_production_step_{current:06d}.npz",
                    latest,
                    case_name=case_name,
                    step=current,
                    config=config,
                    upstream_state_hashes=expected,
                    extra={"trajectory_source": "chns._advance_standard (pf.step_with_diagnostics)"},
                )
                _write_json(
                    progress_path,
                    {
                        "stage": STAGE,
                        "case": case_name,
                        "target_step": end,
                        "committed_step": current,
                        "checkpoint": str(
                            (checkpoint_dir / f"{case_name}_production_step_{current:06d}.npz").relative_to(ROOT)
                        ),
                        "state_hashes": meta["state_hashes"],
                        "config_fingerprint": _canonical_hash(config),
                        "source_hashes": _binding_source_hashes(),
                        "runtime_versions": _runtime_versions(),
                        "elapsed_seconds_this_run": time.perf_counter() - started,
                    },
                )
                print(f"[{STAGE}] production {case_name} {current}/{end}", flush=True)
        window_start_state = latest
        _save_checkpoint(
            checkpoint_dir / f"{case_name}_window_start_step_{window_start:06d}.npz",
            window_start_state,
            case_name=case_name,
            step=window_start,
            config=config,
            extra={"role": "window_reference_state"},
        )
        prev_phi = np.array(latest.phi, copy=True)
        wanted = set(wanted_steps)
        for step in range(window_start + 1, end + 1):
            latest = chns._advance_standard(latest, solid, p, 1)
            if step in wanted:
                arrays = {
                    "step": int(step),
                    "time": float(latest.t),
                    "phi": np.array(latest.phi, copy=True),
                    "phi_prev": prev_phi,
                    "u": np.array(latest.u, copy=True),
                    "v": np.array(latest.v, copy=True),
                    "t": np.array(latest.t, copy=True),
                }
                sample_rows[step] = arrays
                _save_sample_arrays(sample_dir / f"{case_name}_sample_step_{step:06d}.npz", arrays, case_name)
            if step + 1 in wanted:
                prev_phi = np.array(latest.phi, copy=True)
        replay_record.update(
            {
                "decision": "reran_exact_production_from_seed_with_window_sampling",
                "window_start_step": int(window_start),
                "elapsed_seconds": time.perf_counter() - started,
                "state_hashes": _state_hashes(latest),
            }
        )
        state = latest

    endpoint_hashes = _state_hashes(state)
    hash_match = endpoint_hashes == expected
    if not hash_match:
        print(
            f"[{STAGE}] STATE_PROVENANCE_MISMATCH for {case_name}: {endpoint_hashes} != {expected}",
            flush=True,
        )
    meta = _save_checkpoint(
        final_path,
        state,
        case_name=case_name,
        step=end,
        config=config,
        upstream_state_hashes=expected,
        extra={"trajectory_source": "chns._advance_standard (pf.step_with_diagnostics)"},
    )
    return {
        "case": case_name,
        "target_deg": float(target),
        "step": int(end),
        "time": float(state.t),
        "p": p,
        "solid": solid,
        "seed": seed,
        "config": config,
        "config_fingerprint": _canonical_hash(config),
        "seed_state_hashes": _state_hashes(seed),
        "endpoint_state_hashes": endpoint_hashes,
        "expected_l1a2k_state_hashes": expected,
        "matches_l1a2k_per_field_hashes": hash_match,
        "state_acceptance": "accepted_exact_rehydration" if hash_match else "STATE_PROVENANCE_MISMATCH",
        "state_dtypes": {name: str(np.asarray(getattr(state, name)).dtype) for name in ("phi", "u", "v", "t")},
        "artifact_reuse": reuse,
        "replay": replay_record,
        "checkpoint_metadata": meta,
        "sample_steps": sorted(sample_rows.keys()),
        "sample_rows": sample_rows,
        "window_start_step": int(window_start),
        "state": state,
    }


def _ch_only_case() -> tuple[pf.PhaseFieldParams, pf.Solid, pf.State, dict[str, Any], float, float]:
    """The exact L1A-2k ``ch_only_060`` construction (mirrors nwa.run_relaxation kwargs)."""
    sample_every = 1_000
    p = pf.PhaseFieldParams(
        Nx=chns.N,
        Ny=chns.N,
        Lx=chns.LENGTH,
        Ly=chns.LENGTH,
        Re=200.0,
        We=100.0,
        dt=chns.DT,
        M=chns.M_REF,
        eps=chns.EPS_FACTOR * chns.LENGTH / chns.N,
        dtype=jnp.float32,
        ch_solver_rtol=1.0e-6,
        ch_solver_max_iterations=200,
    )
    cos_eff = math.cos(math.radians(60.0))
    sdf = pf.surface_flat(p, wall_height=chns.WALL_HEIGHT)
    solid = pf.make_solid(sdf, p, cos_theta=cos_eff)
    seed = pf.sessile_initial_state(p, solid, R=chns.RADIUS, wall_height=chns.WALL_HEIGHT)
    config = {
        "target_deg": 60.0,
        "N": int(chns.N),
        "Nx": int(chns.N),
        "Ny": int(chns.N),
        "Lx": float(chns.LENGTH),
        "Ly": float(chns.LENGTH),
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
        "R": float(chns.RADIUS),
        "wall_height": float(chns.WALL_HEIGHT),
        "eps_factor": float(chns.EPS_FACTOR),
        "eps": float(p.eps),
        "wall_delta_width": float(1.5 * chns.LENGTH / chns.N),
        "dt": float(chns.DT),
        "M": float(chns.M_REF),
        "phase_storage_model": str(pf.PHASE_ONLY_FLOAT64_STORAGE_MODEL),
        "phase_state_dtype": "float64",
        "velocity_state_dtype": "float32",
        "phase_boundary_model": "impermeable_flux",
        "wetting_model": "surface_energy",
        "wall_measure": str(pf.WALL_MEASURE_METHOD),
        "phase_transport_geometry": str(pf.PHASE_TRANSPORT_GEOMETRY),
        "phase_advection_subcycling": str(pf.PHASE_ADVECTION_SUBCYCLING),
        "eta_pen": float(2.0 * chns.DT),
        "ch_solver_rtol": 1.0e-6,
        "ch_solver_max_iterations": 200,
        "enforce_solid_phi": False,
        "use_gravity": False,
        "solid_geometry": "surface_flat_wall_height_0.25",
        "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
        "dynamics_mode": "CH_ONLY",
        "diagnostic_only": True,
        "diagnostic_sample_every_steps": sample_every,
        "convergence_criteria": dict(nwa.CRITERIA),
    }
    volume = np.asarray(pf.phase_control_volumes(solid, p), dtype=np.float64)
    mass_ref = float(obs.liquid_mass(np.asarray(seed.phi), np.asarray(sdf), p.dx, p.dy))
    conserved_ref = float(np.sum(np.asarray(seed.phi, dtype=np.float64) * volume))
    return p, solid, seed, config, mass_ref, conserved_ref


def _advance_phase_only(state: pf.State, solid: pf.Solid, p: pf.PhaseFieldParams, steps: int) -> pf.State:
    """Fast exact CH-only advance used between sample boundaries (scan of the unchanged public step)."""

    def body(current, _):
        next_state, _info = pf.phase_only_step_with_diagnostics(current, solid, p)
        return next_state, None

    return jax.lax.scan(body, state, None, length=int(steps))[0]


def _rehydrate_ch_only_case(two_k: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    """Resume or rerun the converged CH-only 60-degree reference with window arrays.

    Replays the exact L1A-2k convergence protocol: 1000-step CH-only sample
    blocks, the unchanged ``nwa._window_converged`` gate evaluated on
    ``samples[1:]`` every 10000 steps, and a hash acceptance against the frozen
    ``ch_only_state_hashes``.
    """
    p, solid, seed, config, mass_ref, conserved_ref = _ch_only_case()
    expected = _expected_ch_only_hashes(two_k)
    checkpoint_dir = artifact_root / "checkpoints"
    sample_dir = artifact_root / "samples"
    final_path = checkpoint_dir / "ch_only_060_production_step_180000.npz"
    wanted_steps = _sample_steps(CH_ONLY_STEP, window=WINDOW_LENGTH_STEPS, cadence=SAMPLE_CADENCE_STEPS)

    state: pf.State | None = None
    reuse = {"endpoint_checkpoint": "absent", "window_samples": "absent"}
    sample_rows: dict[int, dict[str, Any]] = {}
    if final_path.is_file():
        try:
            state, _meta = _load_checkpoint(
                final_path,
                case_name="ch_only_060",
                step=CH_ONLY_STEP,
                config=config,
                p=p,
                upstream_state_hashes=expected,
            )
            reuse["endpoint_checkpoint"] = "reused_strict_l1a2m_checkpoint"
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            reuse["endpoint_checkpoint"] = f"rejected: {exc}"
            state = None
        if state is not None:
            missing = [
                step for step in wanted_steps if not (sample_dir / f"ch_only_060_sample_step_{step:06d}.npz").is_file()
            ]
            if missing:
                reuse["window_samples"] = f"incomplete_missing_{len(missing)}"
                state = None
            else:
                for step in wanted_steps:
                    sample_rows[step] = _load_sample_arrays(sample_dir / f"ch_only_060_sample_step_{step:06d}.npz")
                reuse["window_samples"] = f"reused_{len(wanted_steps)}_samples"

    replay: dict[str, Any] = {}
    if state is None:
        started = time.perf_counter()
        volume = np.asarray(pf.phase_control_volumes(solid, p), dtype=np.float64)
        cos_eff = math.cos(math.radians(60.0))
        samples: list[dict[str, Any]] = [
            nwa._sample(
                seed,
                seed.phi,
                solid,
                p,
                ch_only=True,
                steps=0,
                cos_eff=cos_eff,
                phi_ref_mass=mass_ref,
                M=chns.M_REF,
                volume=volume,
                phi_ref_conserved_mass=conserved_ref,
            )
        ]
        latest, current = seed, 0
        prev_phi = np.array(seed.phi, copy=True)
        gate: dict[str, Any] = {}
        converged = False
        while current < CH_ONLY_MAX_STEPS and not converged:
            for _ in range(CH_ONLY_CHUNK_STEPS // SAMPLE_CADENCE_STEPS):
                # 999 scanned steps + 1 explicit step: bitwise-identical to a 1000-step
                # block while retaining the phi before the sample's last public step.
                latest = _advance_phase_only(latest, solid, p, SAMPLE_CADENCE_STEPS - 1)
                prev_phi = np.array(latest.phi, copy=True)
                latest, _info = pf.phase_only_step_with_diagnostics(latest, solid, p)
                current += SAMPLE_CADENCE_STEPS
                row = nwa._sample(
                    latest,
                    jnp.asarray(prev_phi),
                    solid,
                    p,
                    ch_only=True,
                    steps=current,
                    cos_eff=cos_eff,
                    phi_ref_mass=mass_ref,
                    M=chns.M_REF,
                    volume=volume,
                    phi_ref_conserved_mass=conserved_ref,
                )
                samples.append(row)
                if current in set(wanted_steps):
                    arrays = {
                        "step": int(current),
                        "time": float(latest.t),
                        "phi": np.array(latest.phi, copy=True),
                        "phi_prev": prev_phi,
                        "u": np.array(latest.u, copy=True),
                        "v": np.array(latest.v, copy=True),
                        "t": np.array(latest.t, copy=True),
                    }
                    sample_rows[current] = arrays
                    _save_sample_arrays(
                        sample_dir / f"ch_only_060_sample_step_{current:06d}.npz", arrays, "ch_only_060"
                    )
            gate = nwa._window_converged(samples[1:], True, dict(nwa.CRITERIA), M=chns.M_REF)
            converged = bool(gate.get("converged"))
            print(
                f"[{STAGE}] ch_only_060 step {current}: converged={converged} angle={row['measured_angle_deg']}",
                flush=True,
            )
        replay = {
            "decision": "reran_exact_ch_only_convergence_protocol_from_seed",
            "converged": converged,
            "stop_step": int(current),
            "elapsed_seconds": time.perf_counter() - started,
            "final_gate": gate,
        }
        state = latest

    endpoint_hashes = _state_hashes(state)
    hash_match = endpoint_hashes == expected
    if not hash_match:
        print(f"[{STAGE}] STATE_PROVENANCE_MISMATCH for ch_only_060: {endpoint_hashes} != {expected}", flush=True)
    _save_checkpoint(
        final_path,
        state,
        case_name="ch_only_060",
        step=CH_ONLY_STEP,
        config=config,
        upstream_state_hashes=expected,
        extra={"trajectory_source": "pf.phase_only_step_with_diagnostics (L1A-2k protocol)"},
    )
    return {
        "case": "ch_only_equilibrium_060",
        "target_deg": 60.0,
        "step": int(CH_ONLY_STEP),
        "time": float(state.t),
        "p": p,
        "solid": solid,
        "seed": seed,
        "config": config,
        "config_fingerprint": _canonical_hash(config),
        "endpoint_state_hashes": endpoint_hashes,
        "expected_l1a2k_state_hashes": expected,
        "matches_l1a2k_per_field_hashes": hash_match,
        "state_acceptance": "accepted_exact_rehydration" if hash_match else "STATE_PROVENANCE_MISMATCH",
        "state_dtypes": {name: str(np.asarray(getattr(state, name)).dtype) for name in ("phi", "u", "v", "t")},
        "artifact_reuse": reuse,
        "replay": replay,
        "sample_steps": sorted(sample_rows.keys()),
        "sample_rows": sample_rows,
        "window_start_step": int(CH_ONLY_STEP - WINDOW_LENGTH_STEPS),
        "state": state,
    }


# ---------------------------------------------------------------------------
# section 5: authoritative control-volume support classes
# ---------------------------------------------------------------------------


def _support_partition(solid: pf.Solid, p: pf.PhaseFieldParams) -> dict[str, Any]:
    """Deterministic mutually exclusive support classes from the exact contract-11 ``V_i``.

    ``ZERO_VOLUME: V_i == 0``, ``PARTIAL_VOLUME: 0 < V_i < dx*dy``,
    ``FULL_VOLUME: V_i == dx*dy``.  Exact comparisons are used unless the actual
    geometry arithmetic produces a cell whose volume differs from ``dx*dy`` only by
    roundoff, in which case the tolerance is recorded explicitly.  The partition is
    fail-closed on negative or non-finite volumes.
    """
    volume = np.asarray(pf.phase_control_volumes(solid, p), dtype=np.float64)
    area = float(p.dx) * float(p.dy)
    if not np.isfinite(volume).all() or np.any(volume < 0.0):
        raise AuditValidationError("control volumes must be finite and non-negative")
    zero = volume == 0.0
    full = volume == area
    partial = (volume > 0.0) & (volume < area)
    leftover = ~(zero | partial | full)
    near_full = (volume > 0.0) & (~full) & (np.abs(volume - area) <= 1.0e-12 * area)
    near_zero = (~zero) & (volume <= 1.0e-12 * area)
    tolerance_used = bool(np.any(near_full) or np.any(near_zero))
    if np.any(leftover):
        raise AuditValidationError("support partition is not exhaustive over the grid")
    operator = pf.phase_transport_operator(solid, p)
    aperture_x = np.asarray(operator.aperture_x, dtype=np.float64)
    aperture_y = np.asarray(operator.aperture_y, dtype=np.float64)
    aperture_x_open = aperture_x > 0.0
    aperture_y_open = aperture_y > 0.0
    zero_open_faces = int(
        aperture_x_open[zero].sum()
        + aperture_y_open[zero].sum()
        + np.roll(aperture_x_open, 1, axis=0)[zero].sum()
        + np.roll(aperture_y_open, 1, axis=1)[zero].sum()
    )
    classes = {
        "ZERO_VOLUME": np.ascontiguousarray(zero),
        "PARTIAL_VOLUME": np.ascontiguousarray(partial),
        "FULL_VOLUME": np.ascontiguousarray(full),
    }
    return {
        "volume": volume,
        "cell_area": area,
        "classes": classes,
        "counts": {name: int(mask.sum()) for name, mask in classes.items()} | {"total": int(volume.size)},
        "tolerance_policy": {
            "tolerance_used": tolerance_used,
            "rule": "exact equality against 0 and dx*dy; tolerance only if geometry arithmetic requires it",
            "n_full_within_roundoff_but_not_exact": int(near_full.sum()),
            "n_zero_within_roundoff_but_not_exact": int(near_zero.sum()),
            "tolerance_value": 1.0e-12 * area if tolerance_used else 0.0,
            "justification": (
                "geometry emits exact 0.0 for fully solid cells and exact dx*dy for full cells "
                "(verified by the exact-equality counts); no tolerance was required"
                if not tolerance_used
                else "near-boundary volumes exist; tolerance recorded and applied to class assignment"
            ),
        },
        "diagnostics": {
            "min_positive_volume": float(volume[~zero].min()) if np.any(~zero) else 0.0,
            "max_zero_volume": float(volume[zero].max()) if np.any(zero) else 0.0,
            "sum_volume_over_area": float(volume.sum() / area),
            "n_zero_volume_cells_with_open_phase_face": zero_open_faces,
            "aperture_zero_is_machine_zero": bool(np.all((aperture_x == 0.0) | (aperture_x > 0.0))),
        },
        "aperture_x": aperture_x,
        "aperture_y": aperture_y,
    }


def _distance_to_positive_volume(volume: np.ndarray, dx: float) -> dict[str, Any]:
    """Periodic-in-x 8-neighbour BFS distance (in cells) from every cell to the nearest ``V>0`` cell."""
    ny_shape = volume.shape
    nx, ny = ny_shape
    positive = volume > 0.0
    distance = np.full(positive.shape, 10**9, dtype=np.int64)
    queue: deque[tuple[int, int]] = deque()
    for i in range(nx):
        for j in range(ny):
            if positive[i, j]:
                distance[i, j] = 0
                queue.append((i, j))
    while queue:
        i, j = queue.popleft()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ii, jj = (i + di) % nx, j + dj
                if 0 <= jj < ny and distance[ii, jj] > distance[i, j] + 1:
                    distance[ii, jj] = distance[i, j] + 1
                    queue.append((ii, jj))
    return {
        "distance_cells": distance,
        "max_distance_cells": int(distance[~positive].max()) if np.any(~positive) else 0,
        "distance_in_dx_units_max_over_inactive": float(distance[~positive].max()) * dx if np.any(~positive) else 0.0,
    }


# ---------------------------------------------------------------------------
# sections 4/6/7: exact production gate reconstruction, contribution ledger, shadow metric
# ---------------------------------------------------------------------------


def _production_rate_field(phi: np.ndarray, phi_prev: np.ndarray, dt: float) -> np.ndarray:
    """``r_i = (phi_after_public_step - phi_before_public_step) / dt`` on the full grid (mask: none)."""
    return (np.asarray(phi, dtype=np.float64) - np.asarray(phi_prev, dtype=np.float64)) / float(dt)


def _production_phase_rate(r: np.ndarray, area: float) -> float:
    """The exact production gate norm ``sqrt(sum_i(r_i^2) * dx * dy)`` (full grid, mask: none)."""
    rate = np.asarray(r, dtype=np.float64)
    return float(np.sqrt(np.sum(rate * rate) * float(area)))


def _shadow_volume_metric(r: np.ndarray, volume: np.ndarray) -> dict[str, Any]:
    """Diagnostic-only cut-cell-volume metric ``R_V = sqrt(sum_i(V_i * r_i^2))`` and its square."""
    rate = np.asarray(r, dtype=np.float64)
    vol = np.asarray(volume, dtype=np.float64)
    q_v = float(np.sum(vol * rate * rate))
    r_v = float(np.sqrt(q_v))
    # A computed zero is a measurement (0.0), never missing data: the None-vs-0.0
    # distinction is a required regression after PR #15.
    return {
        "R_V": r_v if math.isfinite(r_v) else None,
        "Q_V": q_v if math.isfinite(q_v) else None,
        "is_exact_zero": bool(q_v == 0.0),
    }


def _support_ledger(
    r: np.ndarray,
    partition: dict[str, Any],
    cross_masks: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Per-class production-metric contribution ledger ``E_S = sum_{i in S}(r_i^2 dx*dy)``."""
    rate = np.asarray(r, dtype=np.float64)
    area = float(partition["cell_area"])
    energy_per_cell = rate * rate * area
    e_all = float(np.sum(energy_per_cell))
    ledger: dict[str, Any] = {
        "E_all": e_all,
        "R_prod": float(np.sqrt(e_all)) if e_all > 0.0 else 0.0,
        "classes": {},
    }
    for name in SUPPORT_CLASS_NAMES:
        mask = partition["classes"][name]
        e_class = float(np.sum(energy_per_cell[mask]))
        class_rate = rate[mask]
        nonzero = int(np.count_nonzero(class_rate))
        rms = float(np.sqrt(np.mean(class_rate**2))) if class_rate.size else 0.0
        ledger["classes"][name] = {
            "E": e_class,
            "f": (e_class / e_all) if e_all > 0.0 else 0.0,
            "cell_count": int(mask.sum()),
            "nonzero_rate_cell_count": nonzero,
            "max_abs_r": float(np.max(np.abs(class_rate))) if class_rate.size else 0.0,
            "rms_r": rms if math.isfinite(rms) else 0.0,
        }
    if cross_masks is not None:
        ledger["cross_mask_energy_fraction"] = {}
        ledger["cross_mask_cell_counts"] = {}
        for mask_name, mask in cross_masks.items():
            ledger["cross_mask_energy_fraction"][mask_name] = (
                float(np.sum(energy_per_cell[mask]) / e_all) if e_all > 0.0 else 0.0
            )
            ledger["cross_mask_cell_counts"][mask_name] = int(np.count_nonzero(mask))
    return ledger


def _state_cross_masks(
    phi: np.ndarray, solid: pf.Solid, p: pf.PhaseFieldParams, partition: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Overlapping cross-masks of section 5 (reported on top of the disjoint support classes)."""
    chi = np.asarray(solid.chi, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    masks, _policy = l1a2l._make_region_masks(phi, solid, p)
    interface = masks["interface_frozen_at_window_start"]
    interface_neighbour = interface.copy()
    interface_neighbour |= np.roll(interface, 1, axis=0) | np.roll(interface, -1, axis=0)
    up = np.zeros_like(interface_neighbour)
    down = np.zeros_like(interface_neighbour)
    up[:, :-1] = interface[:, 1:]
    down[:, 1:] = interface[:, :-1]
    interface_neighbour |= up | down
    wall_y = float(pf.wall_plane_height(solid, p))
    y = (np.arange(p.Ny, dtype=np.float64) + 0.5)[None, :] * float(p.dy)
    yy = np.broadcast_to(y, phi.shape)
    distance_to_wall = np.abs(yy - wall_y)
    column = np.arange(phi.shape[0], dtype=np.int64)[:, None]
    row = np.arange(phi.shape[1], dtype=np.int64)[None, :]
    full_grid = np.zeros(phi.shape, dtype=bool)
    cross = {
        "chi_gt_0p5": chi > 0.5,
        "chi_gt_0p1": chi > 0.1,
        "interface_band_fluid_005_095": interface,
        "interface_neighbour_1cell": interface_neighbour,
        "near_wall_0_2dx_all_cells": distance_to_wall <= 2.0 * float(p.dx),
        "near_wall_0_4dx_all_cells": distance_to_wall <= 4.0 * float(p.dx),
        "near_wall_fluid_0_2dx": masks["near_wall_fluid_0_2dx"],
        "near_wall_fluid_0_4dx": masks["near_wall_fluid_0_4dx"],
        "contact_line_2dx": masks["left_contact_line_2dx_frozen"] | masks["right_contact_line_2dx_frozen"],
        "contact_line_4dx": masks["left_contact_line_4dx_frozen"] | masks["right_contact_line_4dx_frozen"],
        "bulk_liquid_phi_ge_095": masks["bulk_liquid_phi_ge_095"],
        "bulk_gas_phi_le_005": masks["bulk_gas_phi_le_005"],
        "periodic_x_seam_columns": full_grid | (column == 0) | (column == phi.shape[0] - 1),
        "y_domain_edge_rows": full_grid | (row == 0) | (row == phi.shape[1] - 1),
    }
    return cross


def _gate_crosscheck(
    phi: np.ndarray,
    phi_prev: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    t: float,
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
    *,
    step: int,
    target_deg: float,
    volume: np.ndarray,
    phi_ref_mass: float,
    phi_ref_conserved: float,
    ch_only: bool,
    r: np.ndarray,
    R_prod: float,
) -> dict[str, Any]:
    """Verify the reconstructed gate equals ``nwa._sample.phase_rate_l2`` to roundoff (section 4)."""
    state = pf.State(
        phi=jnp.asarray(phi, dtype=jnp.float64),
        u=jnp.asarray(u, dtype=p.dtype),
        v=jnp.asarray(v, dtype=p.dtype),
        t=jnp.asarray(t, dtype=p.dtype),
    )
    sample = nwa._sample(
        state,
        jnp.asarray(phi_prev, dtype=jnp.float64),
        solid,
        p,
        ch_only=ch_only,
        steps=step,
        cos_eff=math.cos(math.radians(float(target_deg))),
        phi_ref_mass=phi_ref_mass,
        M=float(p.M),
        volume=volume,
        phi_ref_conserved_mass=phi_ref_conserved,
    )
    gate_value = float(sample["phase_rate_l2"])
    diff = abs(gate_value - float(R_prod))
    scale = max(1.0, abs(gate_value))
    if diff > 5.0e-13 * scale:
        raise AuditValidationError(
            f"production phase-rate reconstruction differs from nwa._sample by {diff} at step {step}"
        )
    return {
        "source_function": "production.nonneutral_wetting_audit._sample",
        "definition": "sqrt(sum((phi_final - phi_before_last_public_step)^2 / dt^2) * dx * dy)",
        "normalization": "cell area dx*dy (not V_i); exact production public-step delta and dt",
        "mask": "none; every grid cell participates exactly as in nwa._sample",
        "dtype": "float64 phase delta; dt float; accumulation float64 numpy",
        "nwa_phase_rate_l2": gate_value,
        "abs_difference": diff,
        "reconstruction_bitwise_equal": bool(gate_value == float(R_prod)),
        "step": int(step),
        "dt": float(p.dt),
        "sample_row": sample,
    }


