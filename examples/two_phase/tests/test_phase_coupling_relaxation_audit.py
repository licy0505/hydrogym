"""Focused contract-11 tests for the diagnostic L1A-2l instrumentation."""

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
from production import phase_coupling_relaxation_audit as audit

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def small_case():
    p, solid, state, config = chns._make_case(60.0, N_value=32, dt=chns.DT, M=chns.M_REF)
    return p, solid, state, config


def _assert_same_state(left: pf.State, right: pf.State) -> None:
    for name in ("phi", "u", "v", "t"):
        np.testing.assert_array_equal(np.asarray(getattr(left, name)), np.asarray(getattr(right, name)))


def test_instrumented_public_step_matches_production_and_reconstructs(small_case):
    p, solid, state, _config = small_case
    masks, _policy = audit._make_region_masks(state.phi, solid, p)
    instrumented, record, fields = audit._instrumented_public_step(state, solid, p, masks)
    production, info = pf.step_with_diagnostics(state, solid, p)

    _assert_same_state(instrumented, production)
    np.testing.assert_allclose(
        fields["advective_increment"] + fields["CH_increment"],
        fields["net_increment"],
        rtol=0.0,
        atol=5.0e-13,
    )
    assert record["phase_reconstruction_linf"] <= 5.0e-13
    assert record["adv_rhs_vs_face_divergence_linf"] <= 5.0e-13
    assert record["explicit_implicit_CH_reconstruction_linf"] <= 5.0e-13
    assert record["implicit_converged"]
    assert bool(np.all(np.asarray(info.implicit_converged)))
    energy_parts = audit._phase_energy_parts(state.phi, solid, p)
    assert abs(energy_parts["component_sum_minus_production_total"]) <= 5.0e-12


def test_decomposed_window_uses_exact_gate_and_cut_cell_mass_ledger(small_case):
    p, solid, state, _config = small_case
    masks, _policy = audit._make_region_masks(state.phi, solid, p)
    window = audit._run_decomposed_window(
        state,
        solid,
        p,
        groups=2,
        steps_per_group=1,
        masks=masks,
        step_start=0,
        case_name="quick_test",
        target_deg=60.0,
        gate_reference_state=state,
    )
    assert window["endpoint_exact_hash_match_production"]
    assert len(window["metric_rows"]) == 2
    for row in window["metric_rows"]:
        production_rate = row["gate_sample"]["phase_rate_l2"]
        forensic_gate = row["production_convergence_gate"]["phase_rate_l2_dxdy_last_public_step"]
        assert production_rate == pytest.approx(forensic_gate, rel=0.0, abs=5.0e-13)
        for region in audit.REGION_NAMES:
            value = row["regions"][region]
            assert value["cancellation_C_mag_volume"] >= 0.0
            assert value["cancellation_C_mag_volume"] <= 1.0 + 1.0e-12
    ledger = audit._mass_ledger_summary(window)
    assert abs(ledger["adv_plus_CH_minus_net"]) <= 5.0e-13
    assert abs(ledger["formal_mass_increment_adv"]) <= 5.0e-13
    assert abs(ledger["formal_mass_increment_ch"]) <= 5.0e-13


def test_ch_only_instrumentation_matches_exact_phase_only_public_step(small_case):
    p, solid, state, _config = small_case
    zero_state = pf.State(
        state.phi,
        jnp.zeros_like(state.u, dtype=p.dtype),
        jnp.zeros_like(state.v, dtype=p.dtype),
        state.t,
    )
    instrumented, record = audit._phase_only_instrumented_step(zero_state.phi, solid, p, zero_state.t)
    production, info = pf.phase_only_step_with_diagnostics(zero_state, solid, p)
    _assert_same_state(instrumented, production)
    assert float(record["reconstruction_linf"]) <= 5.0e-13
    assert bool(record["converged"])
    assert bool(np.all(np.asarray(info.implicit_converged)))
    volume = np.asarray(pf.phase_control_volumes(solid, p), dtype=np.float64)
    formal_ch_mass = float(np.sum(volume * np.asarray(record["CH_increment"], dtype=np.float64)))
    assert abs(formal_ch_mass) <= 5.0e-13


def test_fixed_regional_masks_have_independent_left_right_contact_lines(small_case):
    p, solid, state, _config = small_case
    masks_a, policy_a = audit._make_region_masks(state.phi, solid, p)
    masks_b, policy_b = audit._make_region_masks(state.phi, solid, p)
    assert tuple(masks_a) == audit.REGION_NAMES
    assert policy_a == policy_b
    for key in masks_a:
        np.testing.assert_array_equal(masks_a[key], masks_b[key])
    assert np.any(masks_a["left_contact_line_2dx_frozen"])
    assert np.any(masks_a["right_contact_line_2dx_frozen"])
    assert not np.array_equal(
        masks_a["left_contact_line_2dx_frozen"],
        masks_a["right_contact_line_2dx_frozen"],
    )


def test_multiple_fluid_components_fail_closed_for_mu_statistics(small_case):
    p, _solid, state, _config = small_case
    x = (np.arange(p.Nx, dtype=np.float64) + 0.5) * float(p.dx)
    barriers = np.minimum(np.abs(x - 1.5) - 0.25, np.abs(x - 4.5) - 0.25)[:, None]
    sdf = jnp.asarray(np.broadcast_to(barriers, (p.Nx, p.Ny)), dtype=p.dtype)
    disconnected = pf.make_solid(sdf, p, cos_theta=0.5)
    assert audit._connected_component_count(disconnected, p) == 2
    diagnostics = audit._mu_and_flux_diagnostics(state.phi, state.u, state.v, disconnected, p)
    assert diagnostics["status"] == "unmeasured_fail_closed_multiple_fluid_components"
    assert diagnostics["mu_statistics"] == "unmeasured"


def test_strict_checkpoint_validation_binds_state_config_source_and_runtime(tmp_path, small_case):
    _p, _solid, state, config = small_case
    path = tmp_path / "strict_checkpoint.npz"
    audit._save_state_checkpoint(
        path,
        state,
        section="test_state",
        case_name="quick_test",
        step=1,
        config=config,
        extra={"diagnostic_only": True},
    )
    loaded, metadata = audit._load_state_checkpoint(
        path,
        section="test_state",
        case_name="quick_test",
        step=1,
        config=config,
    )
    _assert_same_state(loaded, state)
    assert metadata["solver_contract_version"] == 12
    assert metadata["production_semantics_changed"] is False

    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in ("phi", "u", "v", "t")}
        saved_metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
    saved_metadata["config_fingerprint"] = "wrong-config"
    arrays["metadata_json"] = np.asarray(json.dumps(saved_metadata))
    bad_path = tmp_path / "wrong_config_checkpoint.npz"
    np.savez_compressed(bad_path, **arrays)
    with pytest.raises(ValueError, match="strict L1A-2l checkpoint validation failed"):
        audit._load_state_checkpoint(
            bad_path,
            section="test_state",
            case_name="quick_test",
            step=1,
            config=config,
        )


def test_production_gate_preserves_dxdy_not_cut_cell_normalization():
    gate = audit._production_gate_definition()
    assert gate["field"] == "nwa._sample.phase_rate_l2"
    assert gate["dt"] == pytest.approx(4.0e-3)
    assert "dx*dy" in gate["normalization"]
    assert "not V_i" in gate["normalization"]
    assert gate["mask"].startswith("none;")
    assert gate["threshold_at_M_ref"] == pytest.approx(1.0e-3)


def test_classifier_reports_one_step_and_nested_matched_D_phi_effects():
    direction = {
        "advective_increment_direction_cosine_to_phi_eq_minus_phi": 0.2,
        "CH_increment_direction_cosine_to_phi_eq_minus_phi": 0.3,
        "production_D_phi_change": -1.0e-6,
        "zero_velocity_CH_D_phi_change": -2.0e-6,
    }
    replay = {
        "D_phi_production_minus_zero_after_matched_steps": -2.0e-4,
        "production_minus_zero_phase_change_l2_volume": 1.0e-4,
        "production_velocity_replay": {"D_phi_production_change": -3.0e-4},
        "zero_velocity_counterfactual": {"D_phi_zero_velocity_change": -1.0e-4},
    }
    authority_windows = {
        "one_step_50000": {
            "metric_rows": [
                {
                    "regions": {
                        "whole_fluid": {
                            "alignment_C_dir_volume": 0.5,
                            "cancellation_C_mag_volume": 0.5,
                            "net_rate_l2_volume": 0.5,
                            "adv_rate_l2_volume": 1.0,
                            "ch_rate_l2_volume": 1.0,
                        }
                    }
                }
            ]
        }
    }
    matrix = audit._classify_candidates(
        authority_windows,
        {},
        None,
        replay,
        direction,
        None,
        {},
        required_provenance_valid=True,
    )
    hydro = next(
        item for item in matrix["candidates"] if item["candidate"] == "HYDRODYNAMICALLY_DRIVEN_PHASE_NONEQUILIBRIUM"
    )
    assert hydro["effect_sizes"]["one_step_production_D_phi_change"] == pytest.approx(-1.0e-6)
    assert hydro["effect_sizes"]["one_step_zero_velocity_CH_D_phi_change"] == pytest.approx(-2.0e-6)
    assert hydro["effect_sizes"]["matched_production_D_phi_change"] == pytest.approx(-3.0e-4)
    assert hydro["effect_sizes"]["matched_zero_velocity_CH_D_phi_change"] == pytest.approx(-1.0e-4)
    assert hydro["effect_sizes"]["matched_production_minus_zero_D_phi_change"] == pytest.approx(-2.0e-4)
    assert hydro["status"] == "FALSIFIED"
