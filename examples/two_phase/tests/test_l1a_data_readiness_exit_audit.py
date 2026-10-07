"""L1A-2o data-readiness exit audit tests (stage specification sections 49-56).

The tests cover ML-contract discovery, schedule/horizon semantics, observable
extraction, refinement comparisons, contact-gap classification, sample
representation fidelity, current-reader compatibility, and the anti-cheating
guarantees (contract 11 stays frozen, no production threshold is touched, no
repair is promoted, the 60 deg authority endpoint is never extended).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import cases as cases_module
import phasefield as pf
import production.l1a_data_readiness_exit_audit as audit
from production import dataset_lineage as lineage
from production import validation as validation_module

FROZEN_PHASEFIELD_FILE_SHA256_PREFIX = "4790c6235dd763db"


@pytest.fixture(scope="session")
def canary_env(tmp_path_factory):
    """One tiny real generator canary in an isolated artifact directory."""

    old_cache, old_canary = audit.CACHE_PATH, audit.CANARY_DIR
    root = tmp_path_factory.mktemp("l1a2o_fixture")
    audit.ARTIFACT_ROOT = root
    audit.CACHE_PATH = root / "cache" / "canaries.json"
    audit.CANARY_DIR = root / "canaries"
    case = cases_module.train_cases()[6]  # flat, We=100, cos_theta=+0.5
    args = audit.GeneratorArgs(N=64, ds=2, dt=2e-3, nsteps=40, save_every=20)
    record = audit.run_canary(case, "flat_test", args, use_cache=False, mandatory=True)
    yield {"record": record, "args": args, "root": root, "case": case}
    audit.ARTIFACT_ROOT, audit.CACHE_PATH, audit.CANARY_DIR = audit.ARTIFACT_ROOT, old_cache, old_canary


@pytest.fixture(scope="session")
def small_export(canary_env):
    """A tiny end-to-end generator export with solver histories still in hand."""

    case = canary_env["case"]
    args = canary_env["args"]
    schedule = audit.effective_schedule(case, args)
    p, solid, initial = pf.build_case(case, N=32, dt=schedule["effective_dt"])
    _final, phi, u, v = pf.rollout(initial, solid, p, 40, save_every=20)
    phi, u, v = np.asarray(phi), np.asarray(u), np.asarray(v)
    path = canary_env["root"] / "small_export.npz"
    from production.dataset_lineage import sample_lineage_metadata

    generator_fingerprint = audit.dataset_fingerprint(case, args, schedule)
    audit.generator._save_case(
        path,
        case,
        p,
        solid,
        phi,
        u,
        v,
        20,
        2,
        {"finite": True},
        generator_fingerprint,
        float("inf"),
        {"solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION)},
    )
    return {
        "path": path,
        "phi": phi,
        "u": u,
        "v": v,
        "p": p,
        "solid": solid,
        "ds": 2,
        "fingerprint": generator_fingerprint,
        "metadata": sample_lineage_metadata(p),
    }


# ---------------------------------------------------------------------------
# section 49: ML contract discovery
# ---------------------------------------------------------------------------


def test_discovers_current_generator_defaults():
    discovered = audit.discover_generator_defaults()
    defaults = discovered["defaults"]
    assert defaults["N"] == 192 and defaults["ds"] == 3
    assert defaults["dt"] == pytest.approx(4e-3)
    assert defaults["nsteps"] == 2000 and defaults["save_every"] == 20
    assert discovered["threshold_status"] == "current_defaults_match_frozen_stage_expectations"


def test_discovers_current_saved_field_dtypes(canary_env):
    record = canary_env["record"]
    stored = audit._dataset_dtype_discovery(Path(record["canary_npz"]))
    assert stored["phi"] == "float32"
    assert stored["u"] == "float16" and stored["v"] == "float16"
    assert stored["chi"] == "float32" and stored["sdf"] == "float32"


def test_discovers_current_surrogate_target_fields(canary_env):
    import surrogate as surrogate_module

    reader_dir = canary_env["root"] / "reader_targets"
    reader_dir.mkdir()
    shutil.copy(canary_env["record"]["canary_npz"], reader_dir / Path(canary_env["record"]["canary_npz"]).name)
    X, Y, _cond, _metas = surrogate_module.load_arrays(str(reader_dir), "train")
    with np.load(canary_env["record"]["canary_npz"], allow_pickle=True) as data:
        phi, u, v = (
            np.asarray(data["phi"]),
            np.asarray(data["u"]).astype(np.float32),
            np.asarray(data["v"]).astype(np.float32),
        )
    assert np.array_equal(Y[..., 0], phi[1:]) and np.array_equal(Y[..., 1], u[1:]) and np.array_equal(Y[..., 2], v[1:])
    assert np.array_equal(X[..., 0], phi[:-1]) and X.shape[-1] == 4 and Y.shape[-1] == 3


def test_pressure_is_not_silently_added_as_training_target(canary_env):
    with np.load(canary_env["record"]["canary_npz"], allow_pickle=True) as data:
        keys = set(data.files)
    assert not ({"p", "pressure"} & keys)
    contract = audit.discover_ml_task_contract(Path(canary_env["record"]["canary_npz"]))
    assert "p" not in contract["dynamic_input_fields"] and "pressure" not in contract["target_fields"]
    assert "pressure" in contract["non_targets"]


def test_ml_task_contract_records_source_hashes(canary_env):
    contract = audit.discover_ml_task_contract(Path(canary_env["record"]["canary_npz"]))
    for name, digest in contract["source_hashes"].items():
        assert digest == audit._file_sha256(Path(name)), name
    assert set(contract["source_hashes"]) >= set(audit.AUDIT_SOURCE_FILES)


# ---------------------------------------------------------------------------
# section 50: schedule / horizon
# ---------------------------------------------------------------------------


def test_effective_schedule_compares_equal_physical_time():
    case = cases_module.train_cases()[6]
    full = audit.effective_schedule(case, audit.GeneratorArgs())
    half = audit.effective_schedule(case, audit.GeneratorArgs(dt=2e-3, nsteps=4000, save_every=40))
    assert full["physical_horizon"] == pytest.approx(half["physical_horizon"]) == pytest.approx(8.0)
    assert full["effective_dt"] != half["effective_dt"]


def test_dt_half_uses_equal_save_times():
    case = cases_module.train_cases()[6]
    full = audit.effective_schedule(case, audit.GeneratorArgs())
    half = audit.effective_schedule(case, audit.GeneratorArgs(dt=2e-3, nsteps=4000, save_every=40))
    assert full["frame_dt"] == pytest.approx(half["frame_dt"])
    assert full["n_frames"] == half["n_frames"]


def test_late60_horizon_is_reported_separately_from_dataset_horizon():
    window = [audit.LATE60_WINDOW_START_STEP * 0.004, audit.LATE60_ENDPOINT_STEP * 0.004]
    horizon = 2000 * 0.004
    assert window == [160.0, 200.0] and horizon == 8.0
    assert horizon < window[0]


def test_no_60deg_authority_extension_beyond_frozen_endpoint(monkeypatch):
    assert audit.LATE60_ENDPOINT_STEP == 50_000
    monkeypatch.setattr(audit, "LATE60_ENDPOINT_STEP", 60_000)
    with pytest.raises(audit.AuditValidationError, match="frozen"):
        audit.late60_replay(force_rerun=True)


# ---------------------------------------------------------------------------
# section 51: observable extraction
# ---------------------------------------------------------------------------


def _tiny_case_context(N=24):
    p = pf.PhaseFieldParams(Nx=N, Ny=N, Lx=6.0, Ly=6.0, dt=2e-3)
    solid = pf.make_solid(np.full((N, N), 1.0), p, cos_theta=0.0)
    return p, solid


def test_formal_mass_uses_sum_V_phi():
    p, solid = _tiny_case_context()
    rng = np.random.default_rng(0)
    phi = rng.random((2, p.Nx, p.Ny))
    u = np.zeros_like(phi)
    volume = np.asarray(pf.phase_control_volumes(solid, p), dtype=np.float64)
    observable = audit.extract_observables(phi, u, u, solid, p, 0.7, save_every=20)
    expected = np.sum(phi[0] * volume) * p.dx * p.dy
    assert observable["rows"][0]["formal_mass"] == pytest.approx(float(expected))


def test_spread_observable_is_deterministic():
    p, solid = _tiny_case_context()
    rng = np.random.default_rng(1)
    phi = (rng.random((2, p.Nx, p.Ny)) > 0.8).astype(np.float64)
    first = audit.extract_observables(phi, np.zeros_like(phi), np.zeros_like(phi), solid, p, 0.7, 20)
    second = audit.extract_observables(phi, np.zeros_like(phi), np.zeros_like(phi), solid, p, 0.7, 20)
    assert first["rows"] == second["rows"]


def test_contact_line_observable_is_left_right_explicit():
    p, solid = _tiny_case_context(N=32)
    phi = np.zeros((1, 32, 32))
    phi[0, 8:20, 10:14] = 1.0
    positions = audit._contact_line_positions(phi[0], float(p.dx))
    assert positions[0] == pytest.approx(8 * p.dx)
    assert positions[1] == pytest.approx(19 * p.dx)


def test_unavailable_force_observable_is_unmeasured_not_zero():
    p, solid = _tiny_case_context()
    phi = np.zeros((1, p.Nx, p.Ny))
    observable = audit.extract_observables(phi, phi, phi, solid, p, 0.7, 20)
    for key in ("F", "F_max", "t_Fmax", "impulse"):
        assert observable["force_observables"][key] == "unmeasured_not_current_contract"


# ---------------------------------------------------------------------------
# section 52: refinement
# ---------------------------------------------------------------------------


def _rows(values_by_key, times):
    return [{"t": t, **{key: values[index] for key, values in values_by_key.items()}} for index, t in enumerate(times)]


def test_refinement_comparison_uses_common_physical_times():
    fine = _rows({"beta": [1.0, 2.0, 3.0]}, [0.08, 0.16, 0.24])
    coarse = _rows({"beta": [1.005, 2.005]}, [0.08, 0.16])
    comparison = audit.compare_scalar_observables(fine, coarse, ("beta",), [0.08, 0.16, 0.24], [0.08, 0.16])
    assert comparison["n_common"] == 2 and comparison["beta"]["at_target"]
    bad_coarse = _rows({"beta": [1.5, 2.5]}, [0.1, 0.2])
    with pytest.raises(audit.AuditValidationError):
        audit.compare_scalar_observables(fine, bad_coarse, ("beta",), [0.08, 0.16, 0.24], [0.1, 0.2])


def test_field_comparison_uses_explicit_grid_mapping():
    fine = np.random.default_rng(2).random((64, 64))
    coarse = np.random.default_rng(3).random((32, 32))
    result = audit.compare_fields_on_common_grid(fine, coarse, (64, 64), (32, 32), (32, 32))
    assert "integer average-pool" in result["mapping_rule"] and result["common_shape"] == [32, 32]
    with pytest.raises(audit.AuditValidationError):
        audit.compare_fields_on_common_grid(fine, coarse, (64, 64), (32, 32), (24, 24))


def test_existing_three_percent_scalar_target_is_not_modified():
    assert audit.KEY_OBSERVABLE_REFINEMENT_CHANGE == 0.03
    assert (
        audit.KEY_OBSERVABLE_REFINEMENT_CHANGE
        == validation_module.PROVISIONAL_READINESS_TARGETS["key_observable_refinement_change"]
    )


def test_unmeasured_refinement_cannot_pass():
    audit_core = {"spatial_refinement": {"scalars": {"beta": {"status": "UNMEASURED"}}}, "temporal_refinement": {}}
    categories = audit.assemble_category_statuses(audit_core, "forensic")
    assert categories["SPATIAL_REFINEMENT"]["status"] == "UNMEASURED"


# ---------------------------------------------------------------------------
# section 53: contact gap
# ---------------------------------------------------------------------------


def test_contact_gap_phi05_and_phi01_are_distinct():
    from production.observables import bottom_gap

    p, _solid = _tiny_case_context(N=32)
    y_axis = (np.arange(32) + 0.5) * p.dy
    sdf = np.repeat(y_axis[None, :], 32, axis=0)  # fluid above a wall at y=0
    phi = np.zeros((32, 32))
    phi = np.where(y_axis[None, :] >= 1.5 * p.dy, 0.3, phi)
    phi = np.where(y_axis[None, :] >= 2.5 * p.dy, 0.9, phi)
    gap05 = bottom_gap(phi, sdf, threshold=0.5)
    gap01 = bottom_gap(phi, sdf, threshold=0.1)
    assert gap05 != gap01 and gap01 < gap05
    assert gap01 == pytest.approx(1.5 * p.dy) and gap05 == pytest.approx(2.5 * p.dy)


def test_gap_is_reported_in_dx_and_eps_units():
    p, _solid = _tiny_case_context(N=32)
    y_axis = (np.arange(32) + 0.5) * p.dy
    sdf = np.repeat(y_axis[None, :], 32, axis=0)
    phi = np.where(y_axis[None, :] >= 1.5 * p.dy, 0.9, np.zeros((32, 32)))
    rows = [{"t": 0.08, "gap05": float(np.min(sdf[phi >= 0.5])), "gap01": float(np.min(sdf[phi >= 0.1]))}]
    metrics = audit.contact_gap_metrics(rows, float(p.dx), float(p.eps))
    assert metrics["gap05_min_over_dx"] == pytest.approx(metrics["gap05_min"] / p.dx)
    assert metrics["gap05_min_over_eps"] == pytest.approx(metrics["gap05_min"] / p.eps)
    assert metrics["gap05_min_over_dx"] != metrics["gap05_min_over_eps"]


def test_contact_gap_refinement_classification_is_fail_closed():
    production = {"n_frames_with_liquid": 5, "n_frames_contact_established": 0, "n_frames_diffuse_only": 0}
    assert audit.contact_gap_classification(production, None, None)["classification"] == "INCONCLUSIVE"
    assert (
        audit.contact_gap_classification(production, {"n_frames_with_liquid": 0}, None)["classification"]
        == "INCONCLUSIVE"
    )
    contact = {"n_frames_with_liquid": 5, "n_frames_contact_established": 2, "n_frames_diffuse_only": 0}
    film = {"n_frames_with_liquid": 5, "n_frames_contact_established": 0, "n_frames_diffuse_only": 0}
    assert audit.contact_gap_classification(contact, contact, contact)["classification"] == "CONTACT_ESTABLISHED"
    assert audit.contact_gap_classification(film, film, film)["classification"] == "FINITE_GAS_FILM_PERSISTS"
    assert audit.contact_gap_classification(contact, film, contact)["classification"] == "FINITE_GAS_FILM_PERSISTS"


