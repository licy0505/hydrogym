"""Regression tests for dataset resolution and exact provenance contracts."""

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
