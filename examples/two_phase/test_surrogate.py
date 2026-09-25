"""Regression tests for the two-phase conservative surrogate head."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("flax")
jnp = jax.numpy

import surrogate as S  # noqa: E402


def test_mass_project_is_bounded_and_fluid_mass_exact():
    phi_in = np.zeros((2, 16, 16), dtype=np.float32)
    phi_in[:, 4:12, 4:12] = 0.8
    phi_pred = phi_in * 0.55
    phi_pred[:, 7:9, 7:9] = 1.4

    chi = np.zeros_like(phi_in)
    chi[:, :, :3] = 1.0

    out = np.asarray(
        S.mass_project(jnp.asarray(phi_pred), jnp.asarray(phi_in), jnp.asarray(chi))
    )
    fluid = chi < 0.5

    assert out.min() >= -1e-7
    assert out.max() <= 1.0 + 1e-7
    assert np.max(np.abs(out[~fluid])) < 1e-7

    m_in = np.sum(np.clip(phi_in, 0.0, 1.0) * fluid, axis=(1, 2))
    m_out = np.sum(out * fluid, axis=(1, 2))
    np.testing.assert_allclose(m_out, m_in, rtol=2e-5, atol=2e-5)
