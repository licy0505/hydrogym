"""Contract-11 checkpoint lineage, dtype preservation and interrupted-run equivalence."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import phasefield as pf  # noqa: E402
from production.a1_promotion_audit import run_restart_smoke  # noqa: E402
from production.restart import load_checkpoint, save_checkpoint  # noqa: E402


def _case():
    p = pf.PhaseFieldParams(Nx=24, Ny=24, Lx=6.0, Ly=6.0, dt=2.0e-3)
    solid = pf.make_solid(pf.surface_flat(p, wall_height=0.25), p, cos_theta=0.0)
    state = pf.sessile_initial_state(p, solid, R=0.8, wall_height=0.25, theta0_deg=90.0)
    return p, solid, state


def test_contract11_checkpoint_roundtrip_preserves_storage_and_parameters(tmp_path):
    p, _solid, state = _case()
    path = tmp_path / "checkpoint.npz"
    metadata = save_checkpoint(path, state, p, step_index=17)
    restored, loaded = load_checkpoint(path, expected_params=p)

    assert metadata["solver_contract_version"] == loaded["solver_contract_version"] == 11
    assert metadata["phase_storage_model"] == loaded["phase_storage_model"] == "phase_only_float64_v1"
    assert metadata["phase_state_dtype"] == "float64"
    assert metadata["velocity_state_dtype"] == "float32"
    assert np.asarray(restored.phi).dtype == np.float64
    assert np.asarray(restored.u).dtype == np.float32
    assert np.asarray(restored.v).dtype == np.float32
    assert np.asarray(restored.t).dtype == np.float32
    for name in ("phi", "u", "v", "t"):
        np.testing.assert_array_equal(np.asarray(getattr(restored, name)), np.asarray(getattr(state, name)))


def test_contract10_checkpoint_is_rejected_without_promotion(tmp_path):
    p, _solid, state = _case()
    current = tmp_path / "current.npz"
    save_checkpoint(current, state, p, step_index=1)
    stale = tmp_path / "contract10.npz"
    with np.load(current, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    metadata = json.loads(str(arrays["metadata"].item()))
    metadata["solver_contract_version"] = 10
    arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    np.savez_compressed(stale, **arrays)

    with pytest.raises(RuntimeError, match="stale solver checkpoint contract"):
        load_checkpoint(stale, expected_params=p)


def test_checkpoint_rejects_legacy_float32_model(tmp_path):
    p = pf.PhaseFieldParams(
        Nx=8,
        Ny=8,
        phase_storage_model=pf.LEGACY_FLOAT32_STORAGE_MODEL,
    )
    state = pf.State(
        phi=jnp.zeros((8, 8), dtype=jnp.float32),
        u=jnp.zeros((8, 8), dtype=jnp.float32),
        v=jnp.zeros((8, 8), dtype=jnp.float32),
        t=jnp.asarray(0.0, dtype=jnp.float32),
    )
    with pytest.raises(RuntimeError, match="require phase_storage_model"):
        save_checkpoint(tmp_path / "legacy.npz", state, p, step_index=0)


def test_split_run_matches_uninterrupted_phase_mass_and_angle_histories():
    report = run_restart_smoke(N=48, first=2, second=2)
    assert report["status"] == "PASS", report
    assert report["checks"]["phase_state_exact"]
    assert report["checks"]["velocity_u_exact"]
    assert report["checks"]["velocity_v_exact"]
    assert report["checks"]["mass_history_exact"]
    assert report["checks"]["angle_history_exact"]
    assert report["checks"]["stale_v10_checkpoint_rejected"]
