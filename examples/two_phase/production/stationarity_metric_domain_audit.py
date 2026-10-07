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


# ---------------------------------------------------------------------------
# sections 10/11/12: spatial localization and inactive-state counterfactuals
# ---------------------------------------------------------------------------


def _zero_volume_activity_localization(
    r: np.ndarray,
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
    partition: dict[str, Any],
    phi: np.ndarray,
    distance_cells: np.ndarray | None = None,
) -> dict[str, Any]:
    """Section 10: where do nonzero-rate zero-volume cells sit, if any exist?"""
    volume = partition["volume"]
    zero = partition["classes"]["ZERO_VOLUME"]
    rate = np.asarray(r, dtype=np.float64)
    active = zero & (rate != 0.0)
    n_active = int(np.count_nonzero(active))
    distance = (
        distance_cells
        if distance_cells is not None
        else _distance_to_positive_volume(volume, float(p.dx))["distance_cells"]
    )
    stencil_adjacent = np.zeros_like(zero)
    for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        stencil_adjacent |= np.roll(volume > 0.0, shift, axis=axis)
    deep = ~(volume > 0.0) & ~stencil_adjacent
    result: dict[str, Any] = {
        "definition": {
            "stencil_adjacent_inactive": "zero-volume cell with a 5-point (periodic-x) V>0 neighbour",
            "deep_inactive": "zero-volume cell with no V>0 5-point neighbour",
        },
        "zero_volume_cell_count": int(np.count_nonzero(zero)),
        "nonzero_rate_zero_volume_cell_count": n_active,
        "nonzero_rate_zero_volume_energy_fraction": None,
        "stencil_adjacent_inactive_cell_count": int(np.count_nonzero(zero & stencil_adjacent)),
        "deep_inactive_cell_count": int(np.count_nonzero(zero & deep)),
        "distance_to_nearest_positive_volume_cell": {
            "max_over_inactive_cells": int(distance[~(volume > 0.0)].max()) if np.any(~(volume > 0.0)) else 0,
            "median_over_inactive_cells": (
                float(np.median(distance[~(volume > 0.0)])) if np.any(~(volume > 0.0)) else 0.0
            ),
        },
    }
    if n_active == 0:
        result["spatial_overlap"] = "not_applicable_no_active_zero_volume_cells"
        result["interpretation"] = (
            "every zero-volume cell has bitwise-zero phase rate; there is no zero-volume activity to localize"
        )
        return result
    energy = rate * rate * partition["cell_area"]
    e_active = float(np.sum(energy[active]))
    e_all = float(np.sum(energy))
    masks = _state_cross_masks(phi, solid, p, partition)
    overlap = {
        name: {
            "cell_count": int(np.count_nonzero(active & mask)),
            "fraction_of_active": float(np.count_nonzero(active & mask) / n_active),
        }
        for name, mask in masks.items()
    }
    overlap["solid_brinkman_region_chi_gt_0p5"] = overlap.pop("chi_gt_0p5")
    distances = distance[active]
    result["nonzero_rate_zero_volume_energy_fraction"] = (e_active / e_all) if e_all > 0.0 else 0.0
    result["spatial_overlap"] = overlap
    result["distance_cells_active"] = {
        "min": int(distances.min()),
        "median": float(np.median(distances)),
        "max": int(distances.max()),
    }
    result["stencil_adjacent_active_count"] = int(np.count_nonzero(active & stencil_adjacent))
    result["deep_active_count"] = int(np.count_nonzero(active & deep))
    result["interpretation"] = (
        "stencil-adjacent inactive state"
        if result["stencil_adjacent_active_count"] > 0
        else "deep inactive-state noise"
    )
    return result


PERTURBATION_POLICIES: dict[str, dict[str, Any]] = {
    "ZERO_CLAMP": {
        "value": 0.0,
        "description": "phi[V==0] = 0.0, the gas-phase reference of the double well (the seeded deep-solid value)",
    },
    "INACTIVE_FREEZE": {
        "value": "previous_step",
        "description": "phi[V==0] = the exact previous-step (step n-1) values of the authority trajectory",
    },
    "ZERO_CLAMP_LIQUID": {
        "value": 1.0,
        "description": "phi[V==0] = 1.0, the liquid-phase reference of the double well; excites the "
        "nu(phi) momentum and capillary-gradient stencils at wall-adjacent cells",
    },
    "ZERO_CLAMP_MIDPOINT": {
        "value": 0.5,
        "description": "phi[V==0] = 0.5, the double-well maximum; maximizes f'(phi) so mu is nonzero "
        "on inactive storage and the capillary stencil response is largest",
    },
}


def _build_counterfactual(
    state: pf.State,
    phi_prev: np.ndarray,
    partition: dict[str, Any],
    policy_name: str,
    p: pf.PhaseFieldParams,
) -> tuple[pf.State, dict[str, Any]]:
    """Section 11: a diagnostic copy whose phi differs from ``state`` only on ``V_i == 0`` cells."""
    if policy_name not in PERTURBATION_POLICIES:
        raise AuditValidationError(f"unknown perturbation policy {policy_name}")
    zero = partition["classes"]["ZERO_VOLUME"]
    phi_a = np.asarray(state.phi, dtype=np.float64)
    phi_b = np.array(phi_a, copy=True)
    policy = PERTURBATION_POLICIES[policy_name]
    if policy["value"] == "previous_step":
        phi_b[zero] = np.asarray(phi_prev, dtype=np.float64)[zero]
    else:
        phi_b[zero] = float(policy["value"])
    state_b = pf.State(
        phi=jnp.asarray(phi_b, dtype=jnp.float64),
        u=jnp.asarray(state.u, dtype=p.dtype),
        v=jnp.asarray(state.v, dtype=p.dtype),
        t=jnp.asarray(state.t, dtype=p.dtype),
    )
    changed = phi_b != phi_a
    outside = int(np.count_nonzero(changed & ~zero))
    if outside != 0:
        raise AuditValidationError(f"perturbation {policy_name} leaked onto {outside} cells with V>0")
    record = {
        "policy": policy_name,
        "description": policy["description"],
        "changed_cell_count": int(np.count_nonzero(changed)),
        "changed_value_min": float(phi_b[changed].min()) if np.any(changed) else 0.0,
        "changed_value_max": float(phi_b[changed].max()) if np.any(changed) else 0.0,
        "phi_hash": _hash_array(phi_b),
        "state_hashes": _state_hashes(state_b),
        "confined_to_zero_volume": True,
    }
    return state_b, record


def _counterfactual_admissibility(
    state_a: pf.State,
    state_b: pf.State,
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
    partition: dict[str, Any],
    config: dict[str, Any],
    config_b: dict[str, Any],
) -> dict[str, Any]:
    """Section 12: B must equal A bitwise outside the V==0 phi entries before any operator runs."""
    zero = partition["classes"]["ZERO_VOLUME"]
    physical = ~zero
    phi_a = np.asarray(state_a.phi, dtype=np.float64)
    phi_b = np.asarray(state_b.phi, dtype=np.float64)
    phi_physical_equal = bool(np.array_equal(phi_a[physical], phi_b[physical]))
    u_equal = bool(np.array_equal(np.asarray(state_a.u), np.asarray(state_b.u)))
    v_equal = bool(np.array_equal(np.asarray(state_a.v), np.asarray(state_b.v)))
    t_equal = bool(np.array_equal(np.asarray(state_a.t), np.asarray(state_b.t)))
    geometry = {
        "volume": partition["volume"],
        "aperture_x": partition["aperture_x"],
        "aperture_y": partition["aperture_y"],
        "sdf": np.asarray(solid.sdf, dtype=np.float64),
        "chi": np.asarray(solid.chi, dtype=np.float64),
        "wall_area": np.asarray(solid.wall_area, dtype=np.float64),
    }
    geometry_hashes = {name: _hash_array(value) for name, value in geometry.items()}
    config_equal = config == config_b
    admissible = phi_physical_equal and u_equal and v_equal and t_equal and config_equal
    return {
        "phi_v_positive_bitwise_equal": phi_physical_equal,
        "u_bitwise_equal": u_equal,
        "v_bitwise_equal": v_equal,
        "t_bitwise_equal": t_equal,
        "config_bitwise_equal": config_equal,
        "geometry_hashes": geometry_hashes,
        "state_a_hashes": _state_hashes(state_a),
        "state_b_hashes": _state_hashes(state_b),
        "admissible": admissible,
        "pressure_note": (
            "p is not an independent state variable: it is reconstructed from (phi,u,v) by the "
            "unchanged projection, so bitwise u/v/phi equality implies bitwise reconstructed pressure"
        ),
    }


# ---------------------------------------------------------------------------
# section 13/14: operator dependency audit and one-public-step replay
# ---------------------------------------------------------------------------


def _field_delta_report(
    delta: np.ndarray,
    physical_mask: np.ndarray,
    reference_norm: float | None = None,
) -> dict[str, Any]:
    phys = np.asarray(delta)[physical_mask] if physical_mask is not None else np.asarray(delta)
    linf = float(np.max(np.abs(phys))) if phys.size else 0.0
    l2 = float(np.sqrt(np.sum(np.asarray(phys) * np.asarray(phys)))) if phys.size else 0.0
    bitwise = bool(np.all(np.asarray(delta) == 0.0))
    report = {
        "l2_delta_physical": l2,
        "linf_delta_physical": linf,
        "bitwise_equal_physical": bitwise,
        "l2_delta_full_grid": float(np.sqrt(np.sum(np.asarray(delta) * np.asarray(delta)))),
        "linf_delta_full_grid": float(np.max(np.abs(delta))),
    }
    if reference_norm is not None:
        report["l2_delta_over_reference_l2"] = (l2 / reference_norm) if reference_norm > 0.0 else 0.0
    return report


def _operator_dependency_audit(
    state_a: pf.State,
    state_b: pf.State,
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
    partition: dict[str, Any],
) -> dict[str, Any]:
    """Section 13: evaluate the production diagnostic operators on A and B without advancing either."""
    physical = ~(partition["classes"]["ZERO_VOLUME"])
    aperture_x_open = partition["aperture_x"] > 0.0
    aperture_y_open = partition["aperture_y"] > 0.0
    out: dict[str, Any] = {
        "policy": "evaluate unchanged production operators on A and B; compare on physical cells/faces"
    }
    phi_a = jnp.asarray(state_a.phi, dtype=jnp.float64)
    phi_b = jnp.asarray(state_b.phi, dtype=jnp.float64)

    rhs_a = pf.rhs(state_a, solid, p)
    rhs_b = pf.rhs(state_b, solid, p)
    phi_rhs_a, u_rhs_a, v_rhs_a, mu_a, mu_expl_a = (np.asarray(x, dtype=np.float64) for x in rhs_a)
    phi_rhs_b, u_rhs_b, v_rhs_b, mu_b, mu_expl_b = (np.asarray(x, dtype=np.float64) for x in rhs_b)
    u_rhs_a32, v_rhs_a32 = np.asarray(rhs_a[1]), np.asarray(rhs_a[2])

    out["chemical_potential_mu"] = _field_delta_report(mu_a - mu_b, physical)
    out["chemical_potential_mu_explicit"] = _field_delta_report(mu_expl_a - mu_expl_b, physical)
    out["rho_of_phi"] = _field_delta_report(
        np.asarray(pf.rho_of(phi_a, p), dtype=np.float64) - np.asarray(pf.rho_of(phi_b, p), dtype=np.float64),
        physical,
    )
    out["nu_of_phi"] = _field_delta_report(
        np.asarray(pf.nu_of(phi_a, p), dtype=np.float64) - np.asarray(pf.nu_of(phi_b, p), dtype=np.float64),
        physical,
    )

    # Capillary acceleration, decomposed exactly as in pf.rhs: (SIGMA_NORM/We) * mu * grad(phi) / rho_l
    # with the production jax scalar kept unconverted so the decomposition is bitwise the rhs term.
    def _capillary(mu: Any, phi: Any, state: pf.State) -> tuple[Any, Any, Any, Any]:
        cap_x = (pf.SIGMA_NORM / p.We) * mu * pf._ddx(phi, p.dx) / p.rho_l
        cap_y = (pf.SIGMA_NORM / p.We) * mu * pf._ddy(phi, p.dy) / p.rho_l
        u_rhs = (
            -pf.div_upwind(state.u, state.v, state.u, p.dx, p.dy)
            + pf.nu_of(phi, p) * pf._lap(state.u, p.dx, p.dy)
            + cap_x
        ).astype(p.dtype)
        v_rhs = (
            -pf.div_upwind(state.u, state.v, state.v, p.dx, p.dy)
            + pf.nu_of(phi, p) * pf._lap(state.v, p.dx, p.dy)
            + cap_y
        ).astype(p.dtype)
        return cap_x, cap_y, u_rhs, v_rhs

    cap_x_a, cap_y_a, mirror_u, mirror_v = _capillary(rhs_a[3], phi_a, state_a)
    cap_x_b, cap_y_b, _mirror_u_b, _mirror_v_b = _capillary(rhs_b[3], phi_b, state_b)
    out["capillary_acceleration_x"] = _field_delta_report(
        np.asarray(cap_x_a, dtype=np.float64) - np.asarray(cap_x_b, dtype=np.float64), physical
    )
    out["capillary_acceleration_y"] = _field_delta_report(
        np.asarray(cap_y_a, dtype=np.float64) - np.asarray(cap_y_b, dtype=np.float64), physical
    )
    out["momentum_mirror_selfcheck"] = {
        "u_rhs_bitwise_equal": bool(np.array_equal(np.asarray(mirror_u), np.asarray(u_rhs_a32))),
        "v_rhs_bitwise_equal": bool(np.array_equal(np.asarray(mirror_v), np.asarray(v_rhs_a32))),
        "note": "the capillary decomposition above is applied to the same mu the production rhs produced",
    }

    wetting_a = np.asarray(pf.wall_energy_derivative(phi_a, solid.cos_theta), dtype=np.float64) * np.asarray(
        pf.wall_measure_density(solid, p), dtype=np.float64
    )
    wetting_b = np.asarray(pf.wall_energy_derivative(phi_b, solid.cos_theta), dtype=np.float64) * np.asarray(
        pf.wall_measure_density(solid, p), dtype=np.float64
    )
    out["wetting_wall_energy_mu_term"] = _field_delta_report(wetting_a - wetting_b, physical)

    out["phase_rhs_advective"] = _field_delta_report(phi_rhs_a - phi_rhs_b, physical)
    adv_x_a, adv_y_a = pf.phase_advective_fluxes(state_a.u, state_a.v, phi_a, solid, p)
    adv_x_b, adv_y_b = pf.phase_advective_fluxes(state_b.u, state_b.v, phi_b, solid, p)
    out["advective_phase_flux_x_faces"] = _field_delta_report(
        np.asarray(adv_x_a, dtype=np.float64) - np.asarray(adv_x_b, dtype=np.float64), aperture_x_open
    )
    out["advective_phase_flux_y_faces"] = _field_delta_report(
        np.asarray(adv_y_a, dtype=np.float64) - np.asarray(adv_y_b, dtype=np.float64), aperture_y_open
    )
    ch_x_a, ch_y_a = pf.chemical_potential_fluxes(rhs_a[4], solid, p)
    ch_x_b, ch_y_b = pf.chemical_potential_fluxes(rhs_b[4], solid, p)
    out["ch_flux_x_faces"] = _field_delta_report(
        np.asarray(ch_x_a, dtype=np.float64) - np.asarray(ch_x_b, dtype=np.float64), aperture_x_open
    )
    out["ch_flux_y_faces"] = _field_delta_report(
        np.asarray(ch_y_a, dtype=np.float64) - np.asarray(ch_y_b, dtype=np.float64), aperture_y_open
    )
    out["momentum_predictor_u_rhs"] = _field_delta_report(
        u_rhs_a.astype(np.float64) - u_rhs_b.astype(np.float64), physical
    )
    out["momentum_predictor_v_rhs"] = _field_delta_report(
        v_rhs_a.astype(np.float64) - v_rhs_b.astype(np.float64), physical
    )
    pressure_a = np.asarray(pf.pressure_field(state_a, solid, p), dtype=np.float64)
    pressure_b = np.asarray(pf.pressure_field(state_b, solid, p), dtype=np.float64)
    out["pressure_rhs_projection_reconstruction"] = _field_delta_report(pressure_a - pressure_b, physical)
    out["all_physical_bitwise_equal"] = bool(
        all(
            value.get("bitwise_equal_physical", True)
            for value in out.values()
            if isinstance(value, dict) and "bitwise_equal_physical" in value
        )
    )
    out["coupling_channels"] = {
        "phase_transport": (
            "aperture-gated; closed faces carry machine-zero flux so V=0 phi cannot enter V>0 phase updates"
        ),
        "momentum_projection": (
            "full-grid periodic stencils (capillary mu*grad(phi), nu(phi), global FFT projection) read every cell "
            "including V=0"
        ),
    }
    return out


def _one_step_counterfactual(
    state_a: pf.State,
    state_b: pf.State,
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
    partition: dict[str, Any],
) -> dict[str, Any]:
    """Section 14: matched one-public-step replay from A and B; compare physical-domain outputs only."""
    physical = ~(partition["classes"]["ZERO_VOLUME"])
    out_a, _info_a = pf.step_with_diagnostics(state_a, solid, p)
    out_b, _info_b = pf.step_with_diagnostics(state_b, solid, p)
    phi_a = np.asarray(out_a.phi, dtype=np.float64)
    phi_b = np.asarray(out_b.phi, dtype=np.float64)
    volume = partition["volume"]
    mass_a = float(np.sum(volume * phi_a))
    mass_b = float(np.sum(volume * phi_b))
    pressure_a = np.asarray(pf.pressure_field(out_a, solid, p), dtype=np.float64)
    pressure_b = np.asarray(pf.pressure_field(out_b, solid, p), dtype=np.float64)
    angle_a = float(pf.measure_contact_angle(out_a.phi, solid, p))
    angle_b = float(pf.measure_contact_angle(out_b.phi, solid, p))
    return {
        "phi_v_positive": _field_delta_report(phi_a - phi_b, physical),
        "u": _field_delta_report(
            np.asarray(out_a.u, dtype=np.float64) - np.asarray(out_b.u, dtype=np.float64), physical
        ),
        "v": _field_delta_report(
            np.asarray(out_a.v, dtype=np.float64) - np.asarray(out_b.v, dtype=np.float64), physical
        ),
        "pressure_reconstructed": _field_delta_report(pressure_a - pressure_b, physical),
        "formal_phase_mass_sum_V_phi": {
            "A": mass_a,
            "B": mass_b,
            "abs_difference": abs(mass_a - mass_b),
            "bitwise_equal": bool(mass_a == mass_b),
        },
        "contact_angle_deg": {
            "A": angle_a if math.isfinite(angle_a) else None,
            "B": angle_b if math.isfinite(angle_b) else None,
            "abs_difference": abs(angle_a - angle_b) if math.isfinite(angle_a) and math.isfinite(angle_b) else None,
        },
        "next_physical_state_changed": bool(
            not _field_delta_report(phi_a - phi_b, physical)["bitwise_equal_physical"]
            or not _field_delta_report(
                np.asarray(out_a.u, dtype=np.float64) - np.asarray(out_b.u, dtype=np.float64), physical
            )["bitwise_equal_physical"]
            or not _field_delta_report(
                np.asarray(out_a.v, dtype=np.float64) - np.asarray(out_b.v, dtype=np.float64), physical
            )["bitwise_equal_physical"]
        ),
        "policy": (
            "one diagnostic public step per copy via pf.step_with_diagnostics; the authority trajectory is not advanced"
        ),
    }


def _series_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if array.size == 0:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "n": 0}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "std": float(np.std(array)),
        "n": int(array.size),
    }


def _process_sample_row(
    case_name: str,
    target_deg: float,
    row_arrays: dict[str, Any],
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
    partition: dict[str, Any],
    distance_cells: np.ndarray,
    *,
    ch_only: bool,
    phi_ref_mass: float,
    phi_ref_conserved: float,
    with_localization: bool,
) -> dict[str, Any]:
    phi = np.asarray(row_arrays["phi"], dtype=np.float64)
    phi_prev = np.asarray(row_arrays["phi_prev"], dtype=np.float64)
    r = _production_rate_field(phi, phi_prev, p.dt)
    area = partition["cell_area"]
    volume = partition["volume"]
    R_prod = _production_phase_rate(r, area)
    ledger = _support_ledger(r, partition)
    shadow = _shadow_volume_metric(r, volume)
    cross_masks = _state_cross_masks(phi, solid, p, partition)
    cross_ledger = _support_ledger(r, partition, cross_masks)
    crosscheck = _gate_crosscheck(
        phi,
        phi_prev,
        row_arrays["u"],
        row_arrays["v"],
        float(row_arrays["t"]),
        solid,
        p,
        step=int(row_arrays["step"]),
        target_deg=target_deg,
        volume=volume,
        phi_ref_mass=phi_ref_mass,
        phi_ref_conserved=phi_ref_conserved,
        ch_only=ch_only,
        r=r,
        R_prod=R_prod,
    )
    sample = crosscheck["sample_row"]
    contacts = l1a2l._contact_metrics(phi, solid, p)
    processed = {
        "case": case_name,
        "step": int(row_arrays["step"]),
        "time": float(row_arrays["t"]),
        "state_hashes": {
            "phi": _hash_array(phi),
            "u": _hash_array(row_arrays["u"]),
            "v": _hash_array(row_arrays["v"]),
            "t": _hash_array(row_arrays["t"]),
        },
        "R_prod": R_prod,
        "Q_prod": R_prod * R_prod,
        "R_V": shadow["R_V"],
        "Q_V": shadow["Q_V"],
        "R_prod_over_R_V": (R_prod / shadow["R_V"]) if shadow["R_V"] and shadow["R_V"] > 0.0 else None,
        "ledger": ledger,
        "cross_mask_energy_fraction": cross_ledger["cross_mask_energy_fraction"],
        "gate_crosscheck": {key: value for key, value in crosscheck.items() if key != "sample_row"},
        "production_sample": sample,
        "contacts": contacts,
    }
    if with_localization:
        processed["zero_volume_activity_localization"] = _zero_volume_activity_localization(
            r, solid, p, partition, phi, distance_cells
        )
    return processed


def _window_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fractions = {
        "ZERO_VOLUME_fraction": [row["ledger"]["classes"]["ZERO_VOLUME"]["f"] for row in rows],
        "PARTIAL_VOLUME_fraction": [row["ledger"]["classes"]["PARTIAL_VOLUME"]["f"] for row in rows],
        "FULL_VOLUME_fraction": [row["ledger"]["classes"]["FULL_VOLUME"]["f"] for row in rows],
    }
    summary: dict[str, Any] = {
        "steps": [row["step"] for row in rows],
        "R_prod": _series_summary([row["R_prod"] for row in rows]),
        "R_V": _series_summary([row["R_V"] for row in rows]),
    }
    for name, values in fractions.items():
        summary[name] = _series_summary(values)
    summary["R_prod_last"] = rows[-1]["R_prod"] if rows else None
    summary["R_V_last"] = rows[-1]["R_V"] if rows else None
    return summary


def _production_window_angle_audit(
    rows: list[dict[str, Any]],
    dense_rows: list[dict[str, Any]],
    *,
    M: float,
    window_mobility_time: float,
) -> dict[str, Any]:
    """Section 18: audit the late-window angle signal without changing any threshold."""

    def spread(values: list[float]) -> dict[str, float]:
        clean = [value for value in values if value is not None]
        if len(clean) < 2:
            return {
                "spread_deg": 0.0,
                "n": len(clean),
                "min_deg": clean[0] if clean else 0.0,
                "max_deg": clean[0] if clean else 0.0,
            }
        return {
            "spread_deg": float(max(clean) - min(clean)),
            "n": len(clean),
            "min_deg": float(min(clean)),
            "max_deg": float(max(clean)),
        }

    angles = [row["production_sample"]["measured_angle_deg"] for row in rows]
    times = [row["production_sample"]["mobility_scaled_time"] for row in rows]
    last_mt = times[-1]
    production_window_rows = [
        row for row, mt in zip(rows, times) if mt >= last_mt - float(window_mobility_time) - 1.0e-12
    ]
    production_window_spread = spread(
        [row["production_sample"]["measured_angle_deg"] for row in production_window_rows]
    )
    full_window_spread = spread(angles)
    dense_angles = [row["production_sample"]["measured_angle_deg"] for row in dense_rows]
    dense_spread = spread(dense_angles)
    cadence_in_dense = spread(
        [angle for angle, row in zip(dense_angles, dense_rows) if row["step"] % SAMPLE_CADENCE_STEPS == 0]
    )

    def slope_deg_per_step(row_list: list[dict[str, Any]]) -> float | None:
        steps = np.asarray([row["step"] for row in row_list], dtype=np.float64)
        values = np.asarray([row["production_sample"]["measured_angle_deg"] for row in row_list], dtype=np.float64)
        keep = np.isfinite(values)
        if int(keep.sum()) < 2:
            return None
        steps_kept, values_kept = steps[keep], values[keep]
        if float(steps_kept.max() - steps_kept.min()) <= 0.0:
            return None
        return float(np.polyfit(steps_kept, values_kept, 1)[0])

    left = [row["contacts"].get("left_contact_x_wrapped") for row in rows]
    right = [row["contacts"].get("right_contact_x_wrapped") for row in rows]
    left_values = [value for value in left if value is not None]
    right_values = [value for value in right if value is not None]
    result = {
        "threshold_recorded_not_changed_deg": PRODUCTION_ANGLE_TOL_DEG,
        "production_window": {
            "definition": (
                f"last {window_mobility_time} M*t (nwa.CRITERIA window_mobility_time) at the frozen 1000-step cadence"
            ),
            "steps": [row["step"] for row in production_window_rows],
            **production_window_spread,
            "exceeds_recorded_threshold": bool(production_window_spread["spread_deg"] > PRODUCTION_ANGLE_TOL_DEG),
        },
        "full_late_window_1000_step_cadence": {"steps": [row["step"] for row in rows], **full_window_spread},
        "dense_window": {
            "cadence_steps": DENSE_CADENCE_STEPS,
            **dense_spread,
            "cadence_only_subset": cadence_in_dense,
        },
        "slope_deg_per_step_cadence": slope_deg_per_step(rows),
        "slope_deg_per_Mt_cadence": (None if slope_deg_per_step(rows) is None else slope_deg_per_step(rows) / float(M)),
        "peak_to_peak_deg_full_window": full_window_spread["spread_deg"],
        "sampling_sensitivity": {
            "spread_production_window_vs_full_window_deg": abs(
                production_window_spread["spread_deg"] - full_window_spread["spread_deg"]
            ),
            "spread_dense_vs_cadence_in_dense_deg": abs(dense_spread["spread_deg"] - cadence_in_dense["spread_deg"]),
            "cadence_changes_classifier_outcome": bool(
                (production_window_spread["spread_deg"] > PRODUCTION_ANGLE_TOL_DEG)
                != (cadence_in_dense["spread_deg"] > PRODUCTION_ANGLE_TOL_DEG)
            ),
        },
        "left_right_symmetry": {
            "left_contact_x_min": float(min(left_values)) if left_values else None,
            "left_contact_x_max": float(max(left_values)) if left_values else None,
            "right_contact_x_min": float(min(right_values)) if right_values else None,
            "right_contact_x_max": float(max(right_values)) if right_values else None,
            "left_contact_x_spread": (float(max(left_values) - min(left_values)) if len(left_values) >= 2 else None),
            "right_contact_x_spread": (
                float(max(right_values) - min(right_values)) if len(right_values) >= 2 else None
            ),
        },
    }
    return result


def _cross_metric_coupling(rows: list[dict[str, Any]], two_l: dict[str, Any], case_name: str) -> dict[str, Any]:
    """Section 19: correlations over the late window; correlation is not causality."""
    steps = [row["step"] for row in rows]
    r_prod = np.asarray([row["R_prod"] for row in rows], dtype=np.float64)
    r_v = np.asarray([row["R_V"] for row in rows], dtype=np.float64)
    f_zero = np.asarray([row["ledger"]["classes"]["ZERO_VOLUME"]["f"] for row in rows], dtype=np.float64)
    angles = np.asarray(
        [
            row["production_sample"]["measured_angle_deg"]
            if row["production_sample"]["measured_angle_deg"] is not None
            else np.nan
            for row in rows
        ],
        dtype=np.float64,
    )
    left_x = np.asarray(
        [
            row["contacts"].get("left_contact_x_wrapped")
            if row["contacts"].get("left_contact_x_wrapped") is not None
            else np.nan
            for row in rows
        ],
        dtype=np.float64,
    )
    right_x = np.asarray(
        [
            row["contacts"].get("right_contact_x_wrapped")
            if row["contacts"].get("right_contact_x_wrapped") is not None
            else np.nan
            for row in rows
        ],
        dtype=np.float64,
    )
    angle_velocity = np.full_like(angles, np.nan)
    line_speed = np.full_like(angles, np.nan)
    dt = rows[1]["time"] - rows[0]["time"] if len(rows) > 1 else 1.0
    if len(rows) > 1:
        angle_velocity[1:] = (angles[1:] - angles[:-1]) / dt
        line_speed[1:] = np.sqrt((left_x[1:] - left_x[:-1]) ** 2 + (right_x[1:] - right_x[:-1]) ** 2) / dt
    l1a2l_series: dict[int, float] = {}
    for window in two_l.get("windows", {}).values():
        if window.get("case") != case_name:
            continue
        for metric_row in window.get("metric_rows", []):
            value = metric_row.get("regions", {}).get("whole_fluid", {}).get("net_rate_l2_volume")
            if value is not None:
                l1a2l_series[int(metric_row["step"])] = float(value)
    net_physical = np.asarray([l1a2l_series.get(step, np.nan) for step in steps], dtype=np.float64)

    def safe_corr(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
        keep = np.isfinite(a) & np.isfinite(b)
        n = int(keep.sum())
        if n < 3 or float(np.std(a[keep])) == 0.0 or float(np.std(b[keep])) == 0.0:
            return {"pearson_r": None, "spearman_rho": None, "n": n}
        pearson = float(np.corrcoef(a[keep], b[keep])[0, 1])
        from scipy.stats import spearmanr

        rho = float(spearmanr(a[keep], b[keep]).statistic)
        return {
            "pearson_r": pearson if math.isfinite(pearson) else None,
            "spearman_rho": rho if math.isfinite(rho) else None,
            "n": n,
        }

    return {
        "pairs": {
            "R_prod_vs_R_V": safe_corr(r_prod, r_v),
            "R_prod_vs_zero_volume_fraction": safe_corr(r_prod, f_zero),
            "R_prod_vs_angle_velocity": safe_corr(r_prod, angle_velocity),
            "R_prod_vs_contact_line_speed": safe_corr(r_prod, line_speed),
            "R_V_vs_angle_velocity": safe_corr(r_v, angle_velocity),
            "R_prod_vs_l1a2l_whole_fluid_net_rate_volume": safe_corr(r_prod, net_physical),
            "R_V_vs_l1a2l_whole_fluid_net_rate_volume": safe_corr(r_v, net_physical),
        },
        "l1a2l_reference_series_steps": sorted(l1a2l_series.keys()),
        "caveat": "correlation over a late window cannot establish causality; see the counterfactual sections",
    }


def _matched_control_calibration(
    authority_rows: list[dict[str, Any]],
    control_windows: dict[str, dict[str, Any]],
    ch_only_endpoint: dict[str, Any],
) -> dict[str, Any]:
    """Section 17: calibrate R_V against converged controls; never emit a reused threshold."""
    authority_median = _series_summary([row["R_V"] for row in authority_rows])["median"]
    calibration: dict[str, Any] = {
        "policy": (
            "report-only calibration against converged controls; the production 0.001 tolerance is NOT "
            "reapplied to R_V and no new acceptance threshold is proposed in L1A-2m"
        ),
        "authority_060_R_V": _series_summary([row["R_V"] for row in authority_rows]),
        "authority_060_R_prod": _series_summary([row["R_prod"] for row in authority_rows]),
        "controls": {},
        "effect_sizes": {},
    }
    control_medians: list[float] = []
    for name, window in control_windows.items():
        summary_rv = window["summary"]["R_V"]
        summary_rp = window["summary"]["R_prod"]
        calibration["controls"][name] = {"R_V": summary_rv, "R_prod": summary_rp}
        control_medians.append(summary_rv["median"])
    if ch_only_endpoint is not None:
        calibration["controls"]["ch_only_equilibrium_060"] = {
            "R_V_endpoint_only": ch_only_endpoint["R_V"],
            "R_prod_endpoint_only": ch_only_endpoint["R_prod"],
            "R_prod_window_from_l1a2k_samples": ch_only_endpoint.get("R_prod_window"),
            "note": (
                "window arrays were not retained for the CH-only control; endpoint ledger plus the L1A-2k scalar "
                "sample window are reported"
            ),
        }
        if ch_only_endpoint.get("R_V") is not None:
            control_medians.append(float(ch_only_endpoint["R_V"]))
    if control_medians and authority_median is not None:
        pooled_control = float(np.median(control_medians))
        spread = float(np.std(control_medians))
        calibration["effect_sizes"] = {
            "authority_median_R_V_over_pooled_control_median_R_V": (
                authority_median / pooled_control if pooled_control > 0.0 else None
            ),
            "authority_minus_control_median_R_V": authority_median - pooled_control,
            "control_median_R_V_std": spread,
            "effect_size_d": ((authority_median - pooled_control) / spread) if spread > 0.0 else None,
            "control_like": bool(
                pooled_control > 0.0
                and authority_median / pooled_control <= CONTROL_LIKE_RATIO_LIMIT
                and (spread == 0.0 or abs(authority_median - pooled_control) / spread <= CONTROL_LIKE_EFFECT_SIZE_LIMIT)
            ),
        }
    return calibration


