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


def test_surrogate_dataset_loader_accepts_only_explicit_v11_sample_lineage(tmp_path):
    lineage = {
        "solver_contract_version": 11,
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

    stale = tmp_path / "stale-v10.npz"
    stale_lineage = dict(lineage, solver_contract_version=10, phase_storage_model="float32_contract_10")
    np.savez(
        stale,
        dataset_schema_version=np.asarray(DATASET_SCHEMA_VERSION, dtype=np.int32),
        dataset_fingerprint=np.asarray("stale-fingerprint"),
        case=np.asarray(json.dumps(stale_lineage)),
        phi=np.zeros((2, 8, 8), dtype=np.float32),
    )
    with np.load(stale, allow_pickle=False) as archive, pytest.raises(RuntimeError, match="contract-11 lineage"):
        S._require_current_dataset(archive, str(stale))

    wrong_cast = tmp_path / "wrong-cast.npz"
    np.savez(
        wrong_cast,
        dataset_schema_version=np.asarray(DATASET_SCHEMA_VERSION, dtype=np.int32),
        dataset_fingerprint=np.asarray("wrong-fingerprint"),
        case=np.asarray(json.dumps(lineage)),
        phi=np.zeros((2, 8, 8), dtype=np.float64),
    )
    with np.load(wrong_cast, allow_pickle=False) as archive, pytest.raises(RuntimeError, match="contract-11 lineage"):
        S._require_current_dataset(archive, str(wrong_cast))
