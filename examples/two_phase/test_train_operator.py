"""Regression tests for validation-gated unrolled fine-tuning."""

import pytest

pytest.importorskip("jax")
pytest.importorskip("optax")

import train_operator as T  # noqa: E402


def test_accept_unroll_candidate_requires_real_validation_improvement():
    baseline = {"loss": 1.0, "raw_mass": 0.01, "projection_l1": 0.01}
    good = {"loss": 0.90, "raw_mass": 0.02, "projection_l1": 0.02}
    assert T._accept_unroll_candidate(baseline, good, 0.002, 0.10, 0.10)

    no_improve = {"loss": 1.01, "raw_mass": 0.01, "projection_l1": 0.01}
    assert not T._accept_unroll_candidate(baseline, no_improve, 0.002, 0.10, 0.10)


def test_accept_unroll_candidate_rejects_projector_exploitation():
    baseline = {"loss": 1.0, "raw_mass": 0.01, "projection_l1": 0.01}
    cheating = {"loss": 0.50, "raw_mass": 3.17, "projection_l1": 3.17}
    assert not T._accept_unroll_candidate(baseline, cheating, 0.002, 0.10, 0.10)
