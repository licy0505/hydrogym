"""Contract-11 solver-state checkpoint serialization and fail-closed restart validation.

Dataset samples are not restart files: this module is the only supported serializer for an
authoritative solver state. Contract-10 checkpoints are intentionally rejected; no implicit
float32-to-float64 migration is provided.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import phasefield as pf

CHECKPOINT_SCHEMA_VERSION = 1
RESTART_STATE_VERSION = "contract11_phase_only_float64_v1"


def _dtype_name(dtype: Any) -> str:
    return jnp.dtype(dtype).name


def _json_value(value: Any) -> Any:
    """Convert dataclass configuration values to strict JSON primitives."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("checkpoint configuration contains a non-finite float")
        return value
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("checkpoint config dictionaries must use string keys")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        return _json_value(value.item())
    raise TypeError(f"unsupported checkpoint configuration value {type(value).__name__}")


def params_metadata(p: pf.PhaseFieldParams) -> dict[str, Any]:
    """Return the complete JSON-safe solver parameter block used for restart compatibility."""
    values: dict[str, Any] = {}
    for item in fields(p):
        value = getattr(p, item.name)
        values[item.name] = _dtype_name(value) if item.name == "dtype" else _json_value(value)
    return values


def _config_fingerprint(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def checkpoint_metadata(p: pf.PhaseFieldParams, *, step_index: int) -> dict[str, Any]:
    """Build canonical lineage and config metadata for a contract-11 state."""
    if pf.SOLVER_CONTRACT_VERSION != 11:
        raise RuntimeError(f"solver checkpoints require contract 11, found contract {pf.SOLVER_CONTRACT_VERSION}")
    if p.phase_storage_model != pf.PHASE_ONLY_FLOAT64_STORAGE_MODEL:
        raise RuntimeError(
            "contract-11 solver checkpoints require phase_storage_model='phase_only_float64_v1'"
        )
    if _dtype_name(p.dtype) != "float32":
        raise RuntimeError("contract-11 production checkpoints require float32 velocity/state working dtype")
    pf.validate_x64_for_phase_storage(p.phase_storage_model)
    if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
        raise ValueError("step_index must be a non-negative integer")
    config = params_metadata(p)
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "restart_state_version": RESTART_STATE_VERSION,
        "step_index": int(step_index),
        "grid": {"Nx": int(p.Nx), "Ny": int(p.Ny)},
        "config": config,
        "config_fingerprint": _config_fingerprint(config),
        **pf.phase_transport_metadata(p),
    }


def _validate_state_dtypes(state: pf.State, p: pf.PhaseFieldParams) -> None:
    expected = {
        "phi": _dtype_name(pf.phase_state_dtype(p)),
        "u": _dtype_name(p.dtype),
        "v": _dtype_name(p.dtype),
        "t": _dtype_name(p.dtype),
    }
    actual = {
        "phi": _dtype_name(state.phi.dtype),
        "u": _dtype_name(state.u.dtype),
        "v": _dtype_name(state.v.dtype),
        "t": _dtype_name(jnp.asarray(state.t).dtype),
    }
    wrong = {name: (actual[name], expected[name]) for name in expected if actual[name] != expected[name]}
    if wrong:
        raise TypeError(f"checkpoint state dtype mismatch (actual, expected): {wrong}")
    shape = (p.Nx, p.Ny)
    for name in ("phi", "u", "v"):
        if tuple(getattr(state, name).shape) != shape:
            raise ValueError(f"state.{name} shape {getattr(state, name).shape} does not match grid {shape}")


def save_checkpoint(
    path: str | Path,
    state: pf.State,
    p: pf.PhaseFieldParams,
    *,
    step_index: int,
) -> dict[str, Any]:
    """Atomically save the authoritative float64-phi / float32-velocity solver state."""
    _validate_state_dtypes(state, p)
    metadata = checkpoint_metadata(p, step_index=step_index)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False
        ) as stream:
            tmp_path = Path(stream.name)
            np.savez_compressed(
                stream,
                phi=np.asarray(state.phi),
                u=np.asarray(state.u),
                v=np.asarray(state.v),
                t=np.asarray(state.t),
                metadata=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, destination)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise
    return metadata


def _validate_metadata(metadata: dict[str, Any], expected_params: pf.PhaseFieldParams | None) -> None:
    if metadata.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(
            f"unsupported or stale checkpoint schema {metadata.get('checkpoint_schema_version')!r}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}"
        )
    contract = metadata.get("solver_contract_version")
    if contract != pf.SOLVER_CONTRACT_VERSION:
        raise RuntimeError(
            f"stale solver checkpoint contract {contract!r}; current contract is "
            f"{pf.SOLVER_CONTRACT_VERSION}. Contract-10 to contract-11 migration is not implicit."
        )
    if metadata.get("restart_state_version") != RESTART_STATE_VERSION:
        raise RuntimeError("checkpoint restart-state version does not match contract-11 A1 semantics")
    if metadata.get("phase_storage_model") != pf.PHASE_ONLY_FLOAT64_STORAGE_MODEL:
        raise RuntimeError("checkpoint phase storage model is stale or unsupported")
    if metadata.get("phase_state_dtype") != "float64" or metadata.get("velocity_state_dtype") != "float32":
        raise RuntimeError("checkpoint dtype lineage is not contract-11 phase64/velocity32")
    pf.validate_x64_for_phase_storage(pf.PHASE_ONLY_FLOAT64_STORAGE_MODEL)
    config = metadata.get("config")
    if not isinstance(config, dict) or metadata.get("config_fingerprint") != _config_fingerprint(config):
        raise RuntimeError("checkpoint config fingerprint is missing or invalid")
    if expected_params is not None:
        expected = params_metadata(expected_params)
        if metadata.get("config_fingerprint") != _config_fingerprint(expected):
            raise RuntimeError("checkpoint solver configuration does not match the requested restart config")
        expected_lineage = pf.phase_transport_metadata(expected_params)
        for key, value in expected_lineage.items():
            if metadata.get(key) != value:
                raise RuntimeError(f"checkpoint lineage mismatch for {key}: {metadata.get(key)!r} != {value!r}")


def load_checkpoint(
    path: str | Path,
    *,
    expected_params: pf.PhaseFieldParams | None = None,
) -> tuple[pf.State, dict[str, Any]]:
    """Load a contract-11 checkpoint, rejecting v10 or dtype/config mismatches without conversion."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {"phi", "u", "v", "t", "metadata"}
            missing = sorted(required - set(archive.files))
            if missing:
                raise RuntimeError(f"checkpoint is missing required fields: {missing}")
            metadata = json.loads(str(np.asarray(archive["metadata"]).item()))
            if not isinstance(metadata, dict):
                raise RuntimeError("checkpoint metadata must be a JSON object")
            _validate_metadata(metadata, expected_params)
            arrays = {name: np.array(archive[name], copy=True) for name in ("phi", "u", "v", "t")}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read solver checkpoint {path}: {exc}") from exc

    expected_grid = metadata.get("grid") or {}
    shape = (int(expected_grid.get("Nx", -1)), int(expected_grid.get("Ny", -1)))
    for name in ("phi", "u", "v"):
        if arrays[name].shape != shape:
            raise RuntimeError(f"checkpoint {name} shape {arrays[name].shape} does not match metadata grid {shape}")
    actual_dtypes = {name: arrays[name].dtype.name for name in arrays}
    expected_dtypes = {
        "phi": "float64",
        "u": metadata["velocity_state_dtype"],
        "v": metadata["velocity_state_dtype"],
        "t": metadata["velocity_state_dtype"],
    }
    if actual_dtypes != expected_dtypes:
        raise RuntimeError(f"checkpoint array dtypes do not match contract metadata: {actual_dtypes}")
    if arrays["t"].shape != ():
        raise RuntimeError("checkpoint time must be a scalar")
    state = pf.State(
        phi=jnp.asarray(arrays["phi"], dtype=jnp.float64),
        u=jnp.asarray(arrays["u"], dtype=jnp.float32),
        v=jnp.asarray(arrays["v"], dtype=jnp.float32),
        t=jnp.asarray(arrays["t"].item(), dtype=jnp.float32),
    )
    if expected_params is not None:
        _validate_state_dtypes(state, expected_params)
    return state, metadata
