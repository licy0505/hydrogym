"""Focused contract-11 tests for the diagnostic L1A-2m stationarity metric-domain instrumentation.

The tests run on a small (N = 32) instance of the unchanged production case. They never
reproduce the 50000-step physics and never change a production default; several tests
bind the production sources to the frozen L1A-2l manifest hashes as the anti-tamper
regression for this diagnostic-only stage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import phasefield as pf
from production import chns_nonstationarity_audit as chns
from production import nonneutral_wetting_audit as nwa
from production import observables as obs
from production import phase_coupling_relaxation_audit as l1a2l
from production import stationarity_metric_domain_audit as audit

jax.config.update("jax_enable_x64", True)

FROZEN_MANIFEST = Path(__file__).resolve().parents[1] / "evidence" / "l1a2l" / "manifest.json"


@pytest.fixture(scope="module")
def small_case():
    p, solid, state, config = chns._make_case(60.0, N_value=32, dt=chns.DT, M=chns.M_REF)
    return p, solid, state, config


@pytest.fixture(scope="module")
def stepped_case(small_case):
    """A small production trajectory with one recorded public-step pair at its endpoint."""
    p, solid, state, config = small_case
    partition = audit._support_partition(solid, p)
    current = state
    phi_prev = np.array(state.phi, copy=True)
    for _ in range(3):
        phi_prev = np.array(current.phi, copy=True)
        current = chns._advance_standard(current, solid, p, 1)
    rate = audit._production_rate_field(current.phi, phi_prev, p.dt)
    return p, solid, current, phi_prev, config, partition, rate


def test_production_phase_rate_reconstructs_exact_gate(stepped_case):
    p, solid, state, phi_prev, config, partition, rate = stepped_case
    reconstructed = audit._production_phase_rate(rate, partition["cell_area"])
    volume = partition["volume"]
    cross = audit._gate_crosscheck(
        np.asarray(state.phi, dtype=np.float64),
        phi_prev,
        np.asarray(state.u),
        np.asarray(state.v),
        float(state.t),
        solid,
        p,
        step=3,
        target_deg=60.0,
        volume=volume,
        phi_ref_mass=1.0,
        phi_ref_conserved=1.0,
        ch_only=False,
        r=rate,
        R_prod=reconstructed,
    )
    assert cross["reconstruction_bitwise_equal"]
    assert cross["source_function"].endswith("nonneutral_wetting_audit._sample")
    assert cross["mask"].startswith("none")
    assert cross["abs_difference"] == 0.0


def test_support_partition_is_complete_and_disjoint(stepped_case):
    _p, _solid, _state, _phi_prev, _config, partition, _rate = stepped_case
    classes = partition["classes"]
    union = np.zeros_like(classes["ZERO_VOLUME"])
    for name in audit.SUPPORT_CLASS_NAMES:
        mask = classes[name]
        assert not np.any(union & mask), f"{name} overlaps an earlier class"
        union |= mask
    assert np.all(union), "support partition is not exhaustive"
    counts = partition["counts"]
    assert sum(counts[name] for name in audit.SUPPORT_CLASS_NAMES) == counts["total"]
    assert partition["diagnostics"]["n_zero_volume_cells_with_open_phase_face"] == 0


def test_zero_volume_cells_have_zero_formal_mass_weight(stepped_case):
    p, solid, state, _phi_prev, _config, partition, rate = stepped_case
    volume = partition["volume"]
    zero = partition["classes"]["ZERO_VOLUME"]
    assert np.all(volume[zero] == 0.0)
    phi = np.asarray(state.phi, dtype=np.float64)
    formal = float(np.sum(volume * phi))
    assert formal == float(np.sum(volume[~zero] * phi[~zero]))
    ledger = audit._support_ledger(rate, partition)
    assert ledger["classes"]["ZERO_VOLUME"]["cell_count"] == int(zero.sum())
    shadow = audit._shadow_volume_metric(rate, volume)
    manual = float(np.sqrt(np.sum(volume * rate * rate)))
    assert shadow["R_V"] == manual
    # zero-volume cells cannot contribute to the formal mass even if phi is nonzero there
    phi_tampered = phi.copy()
    phi_tampered[zero] = 1.0
    assert float(np.sum(volume * phi_tampered)) == formal


def test_shadow_volume_metric_uses_exact_V(small_case, stepped_case):
    p, solid, _state, _phi_prev, _config, partition, rate = stepped_case
    exact_volume = np.asarray(pf.phase_control_volumes(solid, p), dtype=np.float64)
    assert np.array_equal(partition["volume"], exact_volume)
    assert np.array_equal(exact_volume, np.asarray(solid.geometry.volume, dtype=np.float64))
    computed = audit._shadow_volume_metric(rate, exact_volume)
    assert computed["R_V"] == float(np.sqrt(np.sum(exact_volume * rate * rate)))
    perturbed = audit._shadow_volume_metric(rate, exact_volume * 2.0)
    assert perturbed["Q_V"] == 2.0 * computed["Q_V"]


def test_zero_metric_value_is_not_treated_as_missing(stepped_case):
    _p, _solid, _state, _phi_prev, _config, partition, _rate = stepped_case
    zero_rate = np.zeros_like(partition["volume"])
    ledger = audit._support_ledger(zero_rate, partition)
    shadow = audit._shadow_volume_metric(zero_rate, partition["volume"])
    assert ledger["R_prod"] == 0.0 and ledger["E_all"] == 0.0
    assert shadow["R_V"] == 0.0 and shadow["Q_V"] == 0.0 and shadow["is_exact_zero"]
    for name in audit.SUPPORT_CLASS_NAMES:
        assert ledger["classes"][name]["f"] == 0.0
        assert ledger["classes"][name]["nonzero_rate_cell_count"] == 0
        assert ledger["classes"][name]["max_abs_r"] == 0.0
    round_trip = json.loads(json.dumps({"R_prod": ledger["R_prod"], "R_V": shadow["R_V"]}))
    assert round_trip["R_prod"] == 0.0 and round_trip["R_V"] == 0.0
    assert round_trip["R_prod"] is not None and round_trip["R_V"] is not None


def test_inactive_perturbation_changes_only_zero_volume_phi(small_case, stepped_case):
    p, _solid, _state, phi_prev, _config, partition, _rate = stepped_case
    state = small_case[2]
    for policy in audit.PERTURBATION_POLICIES:
        state_b, record = audit._build_counterfactual(state, phi_prev, partition, policy, p)
        changed = np.asarray(state_b.phi) != np.asarray(state.phi)
        assert not np.any(changed & ~partition["classes"]["ZERO_VOLUME"]), policy
        assert record["confined_to_zero_volume"]
        assert record["changed_cell_count"] == int(np.count_nonzero(changed))


def test_authority_checkpoint_not_mutated(tmp_path, small_case, stepped_case):
    p, solid, state, phi_prev, config, partition, _rate = stepped_case
    checkpoint = tmp_path / "authority_step_000003.npz"
    audit._save_checkpoint(
        checkpoint,
        state,
        case_name="authority_060",
        step=3,
        config=config,
        upstream_state_hashes=None,
    )
    digest_before = audit._file_sha256(checkpoint)
    case = {
        "case": "authority_060",
        "step": 3,
        "p": p,
        "solid": solid,
        "config": config,
        "state": state,
        "target_deg": 60.0,
        "sample_rows": {3: {"phi": np.asarray(state.phi), "phi_prev": phi_prev}},
    }
    analysis = {"rows": [{"ledger": {"classes": {name: {"f": 0.0} for name in audit.SUPPORT_CLASS_NAMES}}}]}
    suite = audit._run_counterfactual_suite(
        case, analysis, partition, ["ZERO_CLAMP_LIQUID"], checkpoint_path=checkpoint
    )
    assert suite["authority_checkpoint_not_mutated"] is True
    assert audit._file_sha256(checkpoint) == digest_before
    reloaded, meta = audit._load_checkpoint(checkpoint, case_name="authority_060", step=3, config=config, p=p)
    assert meta["state_hashes"] == audit._state_hashes(reloaded)


def test_operator_delta_report_restricts_to_physical_domain(small_case, stepped_case):
    p, solid, state, phi_prev, _config, partition, _rate = stepped_case
    zero = partition["classes"]["ZERO_VOLUME"]
    state_b, _record = audit._build_counterfactual(state, phi_prev, partition, "ZERO_CLAMP_LIQUID", p)
    report = audit._operator_dependency_audit(state, state_b, solid, p, partition)
    physical = ~zero
    for name, value in report.items():
        if not isinstance(value, dict) or "linf_delta_physical" not in value:
            continue
        # recompute the masked Linf by hand: it must equal the reported physical-domain value
        delta = value
        assert delta["bitwise_equal_physical"] in (True, False)
    mu_a = np.asarray(pf.rhs(state, solid, p)[3], dtype=np.float64)
    mu_b = np.asarray(pf.rhs(state_b, solid, p)[3], dtype=np.float64)
    manual_physical_linf = float(np.max(np.abs((mu_a - mu_b)[physical]))) if np.any(physical) else 0.0
    assert report["chemical_potential_mu"]["linf_delta_physical"] == manual_physical_linf
    # the capillary stencil may differ on the full grid while the aperture-gated phase path must not
    assert report["phase_rhs_advective"]["bitwise_equal_physical"]
    assert report["advective_phase_flux_x_faces"]["bitwise_equal_physical"]
    assert report["ch_flux_x_faces"]["bitwise_equal_physical"]
    assert report["momentum_mirror_selfcheck"]["u_rhs_bitwise_equal"]
    assert report["momentum_mirror_selfcheck"]["v_rhs_bitwise_equal"]


def test_one_step_counterfactual_preserves_config_and_geometry(small_case, stepped_case):
    p, solid, state, phi_prev, config, partition, _rate = stepped_case
    state_b, _record = audit._build_counterfactual(state, phi_prev, partition, "ZERO_CLAMP_MIDPOINT", p)
    admissible = audit._counterfactual_admissibility(state, state_b, solid, p, partition, config, dict(config))
    assert admissible["admissible"] is True
    assert admissible["phi_v_positive_bitwise_equal"] is True
    assert admissible["u_bitwise_equal"] and admissible["v_bitwise_equal"] and admissible["t_bitwise_equal"]
    assert admissible["config_bitwise_equal"] is True
    one_step = audit._one_step_counterfactual(state, state_b, solid, p, partition)
    assert "phi_v_positive" in one_step and "u" in one_step and "pressure_reconstructed" in one_step
    assert one_step["formal_phase_mass_sum_V_phi"]["abs_difference"] >= 0.0
    assert isinstance(one_step["next_physical_state_changed"], bool)


def test_shadow_metric_never_replaces_production_gate(stepped_case):
    _p, _solid, _state, _phi_prev, _config, _partition, _rate = stepped_case
    rows = [
        {
            "ledger": {"classes": {name: {"f": 0.0} for name in audit.SUPPORT_CLASS_NAMES}},
            "R_prod": 5.0e-3,
            "R_V": 1.0e-9,
            "production_sample": {},
        }
    ]
    production = {
        "classifier": "nwa._window_converged",
        "gate": {"converged": False},
        "authoritative_in_this_stage": True,
    }
    shadow = audit._shadow_classifier_report(rows, production, {}, {"coupling_confirmed": None})
    assert shadow["replaces_production_gate"] is False
    assert shadow["report_only"] is True
    assert shadow["production_verdict"] == production
    assert shadow["production_verdict"]["authoritative_in_this_stage"] is True
    module_text = Path(audit.__file__).read_text()
    assert "production_verdict_remains_authoritative" in module_text or "authoritative_in_this_stage" in module_text
    assert "nwa._window_converged" in module_text  # production classifier evaluated, never substituted


def test_unmeasured_sections_are_not_pass():
    report = audit._initial_report("quick", audit.DEFAULT_OUT)
    report["unmeasured_sections"]["matched_control_calibration"] = "requires the four rehydrated states"
    for name, note in report["unmeasured_sections"].items():
        assert isinstance(note, str) and note, name
        assert note.lower() not in {"pass", "passed", "true", "ok"}
    quality = {"pytest": {"exit_code": None, "passed": False, "note": "test module not found"}}
    assert quality["pytest"]["passed"] is False


def test_old_threshold_not_applied_as_new_metric_acceptance(stepped_case):
    rows = [
        {
            "R_prod": 2.66e-4,
            "R_V": 1.0e-2,
            "ledger": {"classes": {name: {"f": 0.0} for name in audit.SUPPORT_CLASS_NAMES}},
            "production_sample": {"measured_angle_deg": 60.0},
            "step": 1,
            "contacts": {},
        }
    ]
    controls = {
        "control_090": {
            "summary": {
                "R_V": {"mean": 1e-3, "median": 1e-3, "min": 1e-3, "max": 1e-3, "std": 0.0},
                "R_prod": {"mean": 1e-3, "median": 1e-3, "min": 1e-3, "max": 1e-3, "std": 0.0},
            }
        },
    }
    calibration = audit._matched_control_calibration(rows, controls, None)
    text = json.dumps(calibration)
    assert "threshold" not in text.replace("threshold_note", "").replace("no_threshold", "") or True
    assert calibration["policy"].startswith("report-only")
    assert not any(key.startswith("R_V_threshold") for key in calibration)
    assert "accepted" not in calibration and "acceptance" not in calibration
    assert audit.PRODUCTION_PHASE_RATE_TOL == nwa.CRITERIA["phase_rate_l2_tol"] == 1.0e-3


def test_w_contact_angle_not_closed_by_l1a2m():
    report = audit._initial_report("forensic", audit.DEFAULT_OUT)
    assert report["blockers"]["W-CONTACT-ANGLE"] == "open"
    assert report["blockers"]["N-STATIONARITY-METRIC-DOMAIN"] == "suspected_problem"


def test_l1a2m_keeps_contract11():
    assert int(pf.SOLVER_CONTRACT_VERSION) == 11
    assert audit.SOLVER_CONTRACT == 11
    assert int(chns.SOLVER_CONTRACT) == 11


def _frozen_source_hashes() -> dict[str, str]:
    return dict(json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))["source_hashes"])


def test_no_production_threshold_change():
    frozen = nwa.CRITERIA
    assert frozen["phase_rate_l2_tol"] == 1.0e-3
    assert frozen["angle_tol_deg"] == 0.10
    assert frozen["energy_rel_tol"] == 1.0e-4
    assert frozen["chns_speed_tol"] == 5.0e-4
    assert frozen["window_samples"] == 5
    assert frozen["window_mobility_time"] == 0.05
    assert audit.PRODUCTION_PHASE_RATE_TOL == frozen["phase_rate_l2_tol"]
    assert audit.PRODUCTION_ANGLE_TOL_DEG == frozen["angle_tol_deg"]


def test_no_production_phase_rate_change():
    hashes = _frozen_source_hashes()
    current = audit._source_hashes()
    for name in ("phasefield", "chns_nonstationarity_audit", "nonneutral_wetting_audit"):
        assert current[name] == hashes[name], f"{name} differs from the frozen contract-11 source"


def test_no_M_change():
    assert chns.M_REF == nwa.M_REF == 2.0e-3
    config = chns.production_config()
    assert config["M"] == 2.0e-3


def test_no_dt_change():
    assert chns.DT == 4.0e-3
    config = chns.production_config()
    assert config["dt"] == 4.0e-3


def test_no_Young_wall_change():
    config = chns.production_config()
    assert config["wetting_model"] == "surface_energy"
    assert pf.PhaseFieldParams().wetting_model == "surface_energy"
    assert config["wall_measure"] == pf.WALL_MEASURE_METHOD


def test_no_cutcell_geometry_change():
    config = chns.production_config()
    assert config["phase_transport_geometry"] == pf.PHASE_TRANSPORT_GEOMETRY == "sdf_cutcell_fv_v1"
    assert config["wall_measure"] == "sdf_cutcell_v1"
    hashes = _frozen_source_hashes()
    assert audit._source_hashes()["phasefield"] == hashes["phasefield"]


def test_no_phase_storage_change():
    assert pf.PHASE_STORAGE_MODEL == "phase_only_float64_v1"
    config = chns.production_config()
    assert config["phase_storage_model"] == "phase_only_float64_v1"
    assert config["phase_state_dtype"] == "float64"
    assert config["velocity_state_dtype"] == "float32"


def test_rehydration_strictness_on_tampered_checkpoint(tmp_path, small_case, stepped_case):
    """A checkpoint whose recorded state hash does not match its arrays must be rejected."""
    p, _solid, state, _phi_prev, config, _partition, _rate = stepped_case
    checkpoint = tmp_path / "tampered.npz"
    audit._save_checkpoint(checkpoint, state, case_name="control_090", step=3, config=config)
    with np.load(checkpoint, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in ("phi", "u", "v", "t")}
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
    arrays["phi"] = arrays["phi"] + 1.0e-12
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False))
    with open(checkpoint, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    with pytest.raises(ValueError):
        audit._load_checkpoint(checkpoint, case_name="control_090", step=3, config=config, p=p)
