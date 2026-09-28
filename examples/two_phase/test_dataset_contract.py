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
