"""Regression tests for the two-phase conservative surrogate head and schema-3 lineage gate."""

import json

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("flax")
jnp = jax.numpy

import surrogate as S  # noqa: E402
from production.dataset_lineage import (  # noqa: E402
    DATASET_SAMPLE_CAST_POLICY,
    DATASET_SAMPLE_REPRESENTATION,
    DATASET_SCHEMA_VERSION,
)


def test_mass_project_is_bounded_and_fluid_mass_exact():
    phi_in = np.zeros((2, 16, 16), dtype=np.float32)
    phi_in[:, 4:12, 4:12] = 0.8
    phi_pred = phi_in * 0.55
    phi_pred[:, 7:9, 7:9] = 1.4

    chi = np.zeros_like(phi_in)
    chi[:, :, :3] = 1.0

    out, stats = S.mass_project_with_stats(jnp.asarray(phi_pred), jnp.asarray(phi_in), jnp.asarray(chi))
    out = np.asarray(out)
    fluid = chi < 0.5

    assert out.min() >= -1e-7
    assert out.max() <= 1.0 + 1e-7
    assert np.max(np.abs(out[~fluid])) < 1e-7

    m_in = np.sum(np.clip(phi_in, 0.0, 1.0) * fluid, axis=(1, 2))
    m_out = np.sum(out * fluid, axis=(1, 2))
    np.testing.assert_allclose(m_out, m_in, rtol=2e-5, atol=2e-5)
    assert np.all(np.asarray(stats["raw_mass_rel"]) > 0.0)
    assert np.all(np.asarray(stats["projection_l1_rel"]) > 0.0)
    assert np.all(np.asarray(stats["projection_linf"]) > 0.0)


def test_straight_through_projection_keeps_projected_forward_and_raw_gradient():
    raw = jnp.array([[[0.2, 0.4], [0.6, 0.8]]], dtype=jnp.float32)
    projected = jnp.array([[[0.3, 0.3], [0.7, 0.7]]], dtype=jnp.float32)
    out = S.straight_through_project(raw, projected)
    np.testing.assert_allclose(np.asarray(out), np.asarray(projected), rtol=0, atol=1e-7)

    grad = jax.grad(lambda x: jnp.sum(S.straight_through_project(x, projected)))(raw)
    np.testing.assert_allclose(np.asarray(grad), np.ones_like(np.asarray(raw)), rtol=0, atol=1e-7)


def test_surrogate_dataset_loader_accepts_only_current_sample_lineage(tmp_path):
    # The current-state fixture derives its expected contract from the live solver
    # module, so a future contract bump cannot leave this positive fixture stale.
    import phasefield as pf

    current_contract = int(pf.SOLVER_CONTRACT_VERSION)
    assert DATASET_SCHEMA_VERSION == 3

    lineage = {
        "solver_contract_version": current_contract,
        "phase_storage_model": "phase_only_float64_v1",
        "phase_state_dtype": "float64",
        "solver_phase_dtype": "float64",
        "stored_sample_phase_dtype": "float32",
        "sample_cast_policy": DATASET_SAMPLE_CAST_POLICY,
        "sample_representation": DATASET_SAMPLE_REPRESENTATION,
    }
    current = tmp_path / "current.npz"
    np.savez(
        current,
        dataset_schema_version=np.asarray(DATASET_SCHEMA_VERSION, dtype=np.int32),
        dataset_fingerprint=np.asarray("fingerprint"),
        case=np.asarray(json.dumps(lineage)),
        phi=np.zeros((2, 8, 8), dtype=np.float32),
    )
    with np.load(current, allow_pickle=False) as archive:
        S._require_current_dataset(archive, str(current))

    # The immediately previous contract is stale under the current contract.
    stale = tmp_path / "stale-previous-contract.npz"
    stale_lineage = dict(
        lineage,
        solver_contract_version=current_contract - 1,
        phase_storage_model=f"float32_contract_{current_contract - 2}",
    )
    np.savez(
        stale,
        dataset_schema_version=np.asarray(DATASET_SCHEMA_VERSION, dtype=np.int32),
        dataset_fingerprint=np.asarray("stale-fingerprint"),
        case=np.asarray(json.dumps(stale_lineage)),
        phi=np.zeros((2, 8, 8), dtype=np.float32),
    )
    with np.load(stale, allow_pickle=False) as archive, pytest.raises(
        RuntimeError, match=f"invalid solver lineage for contract-{current_contract}"
    ):
        S._require_current_dataset(archive, str(stale))

    # Two contracts back stays stale as well.
    older = tmp_path / "stale-older-contract.npz"
    older_lineage = dict(
        lineage,
        solver_contract_version=current_contract - 2,
        phase_storage_model=f"float32_contract_{current_contract - 2}",
    )
    np.savez(
        older,
        dataset_schema_version=np.asarray(DATASET_SCHEMA_VERSION, dtype=np.int32),
        dataset_fingerprint=np.asarray("older-fingerprint"),
        case=np.asarray(json.dumps(older_lineage)),
        phi=np.zeros((2, 8, 8), dtype=np.float32),
    )
    with np.load(older, allow_pickle=False) as archive, pytest.raises(
        RuntimeError, match=f"invalid solver lineage for contract-{current_contract}"
    ):
        S._require_current_dataset(archive, str(older))

    # Missing lineage fails closed.
    missing = tmp_path / "missing-lineage.npz"
    np.savez(
        missing,
        dataset_schema_version=np.asarray(DATASET_SCHEMA_VERSION, dtype=np.int32),
        dataset_fingerprint=np.asarray("missing-fingerprint"),
        phi=np.zeros((2, 8, 8), dtype=np.float32),
    )
    with np.load(missing, allow_pickle=False) as archive, pytest.raises(
        RuntimeError, match="lacks solver/sample lineage"
    ):
        S._require_current_dataset(archive, str(missing))

    # A stored-sample dtype that contradicts the recorded cast policy fails closed.
    wrong_cast = tmp_path / "wrong-cast.npz"
    np.savez(
        wrong_cast,
        dataset_schema_version=np.asarray(DATASET_SCHEMA_VERSION, dtype=np.int32),
        dataset_fingerprint=np.asarray("wrong-fingerprint"),
        case=np.asarray(json.dumps(lineage)),
        phi=np.zeros((2, 8, 8), dtype=np.float64),
    )
    with np.load(wrong_cast, allow_pickle=False) as archive, pytest.raises(
        RuntimeError, match=f"invalid solver lineage for contract-{current_contract}"
    ):
        S._require_current_dataset(archive, str(wrong_cast))
