"""Regression tests for dataset resolution and exact provenance contracts."""

import json
from argparse import Namespace

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("scipy")

import cases as C  # noqa: E402
import generate_dataset as G  # noqa: E402
import phasefield as pf  # noqa: E402


def _args(ds):
    return Namespace(N=192, ds=ds, dt=4e-3, nsteps=2000, save_every=20)


def test_lowwe_test_geometries_are_resolvable_on_v3_saved_grid():
    saved_dx = 6.0 / 192 * 2
    cells = [G._feature_cells(c, saved_dx) for c in C.lowwe_test_cases()]
    assert min(cells) >= 2.5


def test_dataset_fingerprint_changes_when_saved_grid_changes():
    case = C.lowwe_cases(1)[0]
    a2, a3 = _args(2), _args(3)
    dt2, n2, s2 = G._effective_schedule(case, a2)
    dt3, n3, s3 = G._effective_schedule(case, a3)
    fp2 = G._dataset_fingerprint(case, a2, dt2, n2, s2)
    fp3 = G._dataset_fingerprint(case, a3, dt3, n3, s3)
    assert fp2 != fp3


def test_v6_dataset_is_stale_under_v7(tmp_path, monkeypatch):
    """Schema v3 is retained, but contract-6 phase semantics must fingerprint stale under v7."""
    assert pf.SOLVER_CONTRACT_VERSION == 7
    assert pf.WETTING_MODELS == ("surface_energy", "surface_energy_volume_v6", "legacy_affinity", "none")
    params = pf.PhaseFieldParams(Nx=8, Ny=8, Lx=1.0, Ly=1.0)
    assert params.wetting_model == "surface_energy"
    assert params.phase_boundary_model == "impermeable_flux"
    case = C.lowwe_cases(1)[0]
    args = _args(3)
    dt, nsteps, save_every = G._effective_schedule(case, args)
    expected_v7 = G._dataset_fingerprint(case, args, dt, nsteps, save_every)

    path = tmp_path / "train_lowWe_000_flat.npz"
    with monkeypatch.context() as as_v6:
        as_v6.setattr(pf, "SOLVER_CONTRACT_VERSION", 6)
        saved_v6 = G._dataset_fingerprint(case, args, dt, nsteps, save_every)
        np.savez(
            path,
            dataset_schema_version=np.array(G.DATASET_SCHEMA_VERSION, dtype=np.int32),
            dataset_fingerprint=np.array(saved_v6),
        )
        assert G._saved_case_is_current(path, saved_v6)

    assert saved_v6 != expected_v7
    assert not G._saved_case_is_current(path, expected_v7)
    # The explicit boundary-model key also makes otherwise identical custom cases stale.
    legacy_case = {**case, "phase_boundary_model": "projection_legacy"}
    assert G._dataset_fingerprint(legacy_case, args, dt, nsteps, save_every) != expected_v7


def test_manifest_records_solver_contract_version(tmp_path):
    manifest = G._write_manifest(tmp_path, "smoke", [])
    assert manifest["solver_contract_version"] == pf.SOLVER_CONTRACT_VERSION == 7
    assert manifest["wetting_model"] == "surface_energy"
    assert manifest["phase_boundary_model"] == "impermeable_flux"
    assert json.loads((tmp_path / "manifest.json").read_text())["solver_contract_version"] == 7


def test_coarse_geometry_is_signed_and_shape_correct():
    p = pf.PhaseFieldParams(Nx=96, Ny=96, Lx=6.0, Ly=6.0)
    solid = pf.make_solid(pf.surface_pillars(p, n_pillars=4, width=0.3), p)
    frac, sdf = G._coarsen_geometry(solid, 2, p.dx)
    assert frac.shape == (48, 48)
    assert sdf.shape == (48, 48)
    assert np.any(sdf < 0.0)
    assert np.any(sdf > 0.0)
    assert np.all((frac >= 0.0) & (frac <= 1.0))


# ---------------------------------------------------------------------------
#  ``spreading``: high-inertia impact + dynamic spreading transfer set
# ---------------------------------------------------------------------------


def test_spreading_set_keeps_the_transfer_protocol():
    cs = C.CASE_SETS["spreading"]()
    train = [c for c in cs if c["split"] == "train"]
    test = [c for c in cs if c["split"] == "test"]
    assert train and test
    # Train only ever sees the two simple families; the complex ones stay unseen.
    assert {c["surface"] for c in train} <= set(C.SIMPLE_FAMILIES)
    assert {c["surface"] for c in test} <= set(C.COMPLEX_FAMILIES)
    # Every unseen family is represented.
    assert {c["surface"] for c in test} == set(C.COMPLEX_FAMILIES)
    # Test geometries are disjoint from the training ones (distinct seeds).
    train_seeds = {c.get("seed") for c in train}
    assert not any(c.get("seed") in train_seeds for c in test)


def test_spreading_cases_are_high_inertia_and_cfl_stable():
    for c in C.CASE_SETS["spreading"]():
        assert float(c["u_impact"]) >= 1.5
        # ``Re`` is optional and defaults to 200 inside ``pf.build_case``.
        assert float(c.get("Re", 200.0)) == 200.0
        assert float(c["We"]) >= 150.0
        p = pf.PhaseFieldParams(Nx=192, Ny=192, Lx=6.0, Ly=6.0, dt=float(c["dt"]))
        # The requested dt must already be inside the CFL-stable envelope.
        assert float(c["dt"]) <= float(pf.stable_dt(p, u_max=2.0))


def test_spreading_geometries_are_resolvable_on_v3_saved_grid():
    saved_dx = 6.0 / 192 * 3
    cells = [G._feature_cells(c, saved_dx) for c in C.CASE_SETS["spreading"]()]
    assert min(cells) >= 2.0


def test_spreading_schedule_is_integer_and_physical():
    args = _args(3)
    for c in C.CASE_SETS["spreading"]():
        dt, nsteps, save_every = G._effective_schedule(c, args)
        assert dt > 0.0 and nsteps >= save_every > 0
        assert nsteps % save_every == 0
        assert dt <= float(c["dt"])


def test_smoke_case_set_is_exactly_the_l0_transfer_contract():
    smoke = C.CASE_SETS["smoke"]()
    assert len(smoke) == 8
    train = [c for c in smoke if c["split"] == "train"]
    test = [c for c in smoke if c["split"] == "test"]
    assert len(train) == len(test) == 4
    assert {c["surface"] for c in train} <= {"flat", "pillars"}
    assert {c["surface"] for c in train} == {"flat", "pillars"}
    assert {c["surface"] for c in test} == {"random_pillars", "hierarchical", "grooves", "wedge"}
    assert all(float(c["u_impact"]) == 1.0 for c in smoke)
    assert all(float(c["Re"]) == 120.0 for c in smoke)
    assert all(float(c["R"]) == pytest.approx(0.65) for c in smoke)
    assert all(c["velocity_mode"] == "uniform" for c in smoke)
    assert all(float(c["dt"]) == pytest.approx(2e-3) for c in smoke)


def test_smoke_geometry_features_have_three_saved_grid_cells():
    saved_dx = 6.0 / 64.0
    smoke = C.CASE_SETS["smoke"]()
    cells = [G._feature_cells(c, saved_dx) for c in smoke]
    finite = [n for n in cells if np.isfinite(n)]
    assert finite and min(finite) >= 3.0


def test_diagnose_checks_total_and_fluid_mass_against_raw_initial_state():
    p = pf.PhaseFieldParams(Nx=64, Ny=64, Lx=6.0, Ly=6.0)
    solid = pf.make_solid(pf.surface_flat(p), p)
    raw = np.zeros((p.Nx, p.Ny), dtype=np.float32)
    raw[0, 0] = 0.5  # diffuse tail in the geometric solid at raw t=0
    raw[0, 4] = 0.5
    initial = pf.State(phi=raw, u=np.zeros_like(raw), v=np.zeros_like(raw), t=0.0)

    # Simulate a loss after initialization: both the total phase mass and the
    # fluid-region mass must be compared against the raw t=0 state.
    first_saved = np.zeros_like(raw)
    first_saved[0, 4] = 0.495
    history = first_saved[None, ...]
    ok, diagnostics = G._diagnose(
        initial,
        history,
        np.zeros_like(history),
        np.zeros_like(history),
        solid,
        p,
        max_phi_overshoot=0.02,
        max_solid_leak=5e-4,
        min_total_mass_ratio=0.995,
        max_total_mass_ratio=1.005,
        max_speed=5.0,
    )
    assert not ok
    assert diagnostics["startup_total_mass_ratio"] == pytest.approx(0.495)
    assert diagnostics["min_total_mass_ratio"] == pytest.approx(0.495)
    assert diagnostics["min_fluid_mass_ratio"] == pytest.approx(0.99)
