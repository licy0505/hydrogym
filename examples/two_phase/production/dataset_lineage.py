"""Schema-3 ML sample lineage for contract-11 solver trajectories."""

from __future__ import annotations

from typing import Any

DATASET_SCHEMA_VERSION = 3
DATASET_SAMPLE_CAST_POLICY = "downsample_mean_then_float32_v1"
DATASET_SAMPLE_REPRESENTATION = "derived_training_observable_not_restart_authoritative"


def sample_lineage_metadata(params: Any) -> dict[str, str | int]:
    """Identifiers separating authoritative solver state from the stored training sample."""
    import phasefield as pf

    return {
        "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
        "phase_storage_model": str(params.phase_storage_model),
        "solver_phase_dtype": str(pf.phase_storage_metadata(params)["phase_state_dtype"]),
        "stored_sample_phase_dtype": "float32",
        "sample_cast_policy": DATASET_SAMPLE_CAST_POLICY,
        "sample_representation": DATASET_SAMPLE_REPRESENTATION,
    }


def validate_training_sample_lineage(metadata: dict[str, Any], *, phi_dtype: str | None = None) -> None:
    """Reject schema-3 files that lack current solver precision or have an ambiguous sample cast."""
    import phasefield as pf

    if not isinstance(metadata, dict):
        raise RuntimeError("dataset case metadata must be a JSON object")
    expected = {
        "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
        "phase_storage_model": pf.PHASE_ONLY_FLOAT64_STORAGE_MODEL,
        "phase_state_dtype": "float64",
        "solver_phase_dtype": "float64",
        "stored_sample_phase_dtype": "float32",
        "sample_cast_policy": DATASET_SAMPLE_CAST_POLICY,
        "sample_representation": DATASET_SAMPLE_REPRESENTATION,
    }
    mismatches = {
        key: {"actual": metadata.get(key), "expected": value}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"stale or ambiguous two_phase dataset solver lineage: {mismatches}")
    if phi_dtype is not None and phi_dtype != expected["stored_sample_phase_dtype"]:
        raise RuntimeError(
            f"stored phi dtype {phi_dtype!r} contradicts sample metadata; expected float32 under "
            f"{DATASET_SAMPLE_CAST_POLICY}"
        )
