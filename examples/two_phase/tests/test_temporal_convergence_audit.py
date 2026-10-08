"""Focused L1A-2q tests (B30): fail-closed convergence-measurement machinery."""

import json

import numpy as np
import pytest

import phasefield as pf
from production import temporal_convergence_audit as tca
from production import timestep_policy


def _tiny_case():
    return tca.study_cases()["flat_we100_ct050"]["case"]


def test_initial_state_hashes_identical_across_dt_levels():
    case = _tiny_case()
    hashes = set()
    for dt in (0.002, 0.001, 0.0005):
        _p, _solid, initial = pf.build_case(case, N=24, dt=dt)
        hashes.add(json.dumps(tca._state_hashes(initial), sort_keys=True))
    assert len(hashes) == 1, "the initial state must not depend on the dt level"


def test_equal_physical_observation_times_across_levels():
    times = set()
    for dt in tca.DT_LEVELS:
        schedule = tca.explicit_schedule(192, dt, horizon=8.0, frame_dt=0.08)
        times.add(json.dumps(tca.frame_times(schedule).tolist()))
    assert len(times) == 1, "frame times must be bit-identical across dt levels"


def test_explicit_schedule_mapping_exact_and_fail_closed():
    schedule = tca.explicit_schedule(192, 0.002, horizon=8.0, frame_dt=0.08)
    assert schedule["nsteps"] == 4000 and schedule["save_every"] == 40
    assert schedule["n_frames"] == 100
    schedule = tca.explicit_schedule(192, 0.0005, horizon=8.0, frame_dt=0.08)
    assert schedule["nsteps"] == 16000 and schedule["save_every"] == 160
    with pytest.raises(tca.AuditValidationError):
        tca.explicit_schedule(192, 0.003, horizon=8.0, frame_dt=0.08)
    with pytest.raises(tca.AuditValidationError):
        tca.explicit_schedule(192, 0.002, horizon=8.001, frame_dt=0.08)


def test_error_norm_domains():
    n = 8
    volume = np.full((n, n), 0.5)
    a = {"phi": np.full((n, n), 1.0), "u": np.full((n, n), 2.0), "v": np.zeros((n, n))}
    b = {"phi": np.zeros((n, n)), "u": np.zeros((n, n)), "v": np.zeros((n, n))}
    errors = tca.field_errors(a, b, volume, dx=0.5, dy=0.5)
    # volume-weighted phi L2: sqrt(sum(V dphi^2)/sum(V)) with constant V = 1.0
    assert errors["phi_l2_volume_weighted"] == pytest.approx(1.0)
    # physical-domain RMS over the full Nx*Ny domain
    assert errors["u_l2_physical"] == pytest.approx(2.0)
    assert errors["u_linf"] == pytest.approx(2.0)
    assert errors["v_l2_physical"] == 0.0


def test_observed_order_known_first_order_fixture():
    verdict = tca.observed_order(4.0e-3, 2.0e-3)
    assert verdict["status"] == "FIRST_ORDER_LIKE"
    assert verdict["p_obs"] == pytest.approx(1.0)


def test_observed_order_higher_order_and_asymptotic_band():
    assert tca.observed_order(8.0e-3, 1.0e-3)["status"] == "HIGHER_ORDER_LIKE"
    assert tca.observed_order(1.5e-3, 1.0e-3)["status"] == "NOT_IN_ASYMPTOTIC_REGIME"


def test_observed_order_zero_and_nonmonotone_handling():
    assert tca.observed_order(0.0, 0.0)["status"] == "UNDETERMINED"
    assert tca.observed_order(0.0, 1e-9)["status"] == "UNDETERMINED"
    assert tca.observed_order(1e-3, 0.0)["status"] == "UNDETERMINED"
    verdict = tca.observed_order(1.0e-3, 2.0e-3)
    assert verdict["status"] == "NONMONOTONE" and verdict["p_obs"] is None
    assert tca.observed_order(None, 1e-3)["status"] == "UNDETERMINED"


def _gaussian_series(t_peak: float, width: float, amplitude: float, dt: float, step: float):
    t = np.arange(0.16, 0.48 + 1e-12, step)
    return {
        "t": t.tolist(),
        "max_speed": (amplitude * np.exp(-0.5 * ((t - t_peak) / width) ** 2)).tolist(),
    }


def test_peak_timing_and_amplitude_measured_separately():
    coarse = _gaussian_series(0.24, 0.02, 1.0, 0.002, 0.002)
    fine = _gaussian_series(0.25, 0.02, 1.0, 0.001, 0.001)
    audit = tca.peak_audit(coarse, fine, dt_fine=0.001)
    assert audit["t_peak_shift"] == pytest.approx(-0.01, abs=2e-3)
    assert audit["peak_amplitude_rel_diff"] < 0.05  # amplitude nearly identical
    assert audit["t_peak_shift"] != 0.0  # timing differs — reported separately
    assert audit["peak_aligned_rms_error"] < audit["pointwise_rms_error"]


def test_official_gates_unchanged():
    assert tca.REFINEMENT_GATE == 0.03
    assert tca.PHASE_OVERSHOOT_GATE == 0.02
    args = tca.l1a.GeneratorArgs()
    assert args.max_phi_overshoot == 0.02
    assert args.max_solid_leak == 5e-4
    assert args.min_total_mass_ratio == 0.995
    assert args.max_total_mass_ratio == 1.005
    assert args.max_speed == 5.0


def _fake_analysis(window_gate=None, horizon_gate=None):
    window_gate = {} if window_gate is None else window_gate
    horizon_gate = {} if horizon_gate is None else horizon_gate
    return {
        "order_summary": {"flat_we100_ct050": {"phi@t": tca.observed_order(2e-3, 1e-3)}},
        "observable_convergence_matrix": {
            "flat_we100_ct050": {
                quantity: {"order": tca.observed_order(2e-3, 1e-3), "E2": {"rel_rms_error": value}}
                for quantity, value in window_gate.items()
            }
        }
        if window_gate
        else {},
        "horizon_convergence": {
            "quantities": {
                quantity: {"E2": {"rel_rms_error": value}} for quantity, value in horizon_gate.items()
            }
        },
        "spatial_refinement_recheck": {"quantities": {}},
        "cost_accuracy_matrix": {
            "0.004": {"projected_full_horizon_seconds": 120.0},
            "0.002": {"projected_full_horizon_seconds": 240.0},
            "0.001": {"projected_full_horizon_seconds": 480.0},
            "0.0005": {"projected_full_horizon_seconds": 960.0},
        },
        "peak_timing_amplitude_audit": {},
        "local_step_error_audit": {"frozen_states": []},
    }


def _fake_mechanism():
    return {"TIME_SPLITTING_OR_COUPLING_LIMITATION": {"status": "NOT_TESTED"}}


def test_no_promotion_from_quick_profile():
    args = tca.RunArgs(profile="quick")
    analysis = _fake_analysis(
        window_gate={"spread_width": 0.001, "max_speed": 0.001},
        horizon_gate={"spread_width": 0.001, "max_speed": 0.001},
    )
    verdicts = tca.build_policy_and_verdicts(args, analysis, _fake_mechanism())
    assert verdicts["temporal_verdict"] == "INCONCLUSIVE"
    assert verdicts["selected_action"] == "ADDITIONAL_TARGETED_DIAGNOSTIC"


def test_no_false_readiness_from_incomplete_case_matrix():
    args = tca.RunArgs(profile="forensic")
    analysis = _fake_analysis()  # no gate measurements at all
    verdicts = tca.build_policy_and_verdicts(args, analysis, _fake_mechanism())
    assert verdicts["l1b_exit_verdict"] == "L1B_DATA_NOT_READY"


def test_gate_failure_blocks_ready_even_if_convergence_first_order():
    args = tca.RunArgs(profile="forensic")
    analysis = _fake_analysis(
        window_gate={"spread_width": 0.10, "max_speed": 0.10},
        horizon_gate={"spread_width": 0.10, "max_speed": 0.10},
    )
    verdicts = tca.build_policy_and_verdicts(args, analysis, _fake_mechanism())
    assert verdicts["temporal_verdict"] == "FIRST_ORDER_INTEGRATOR_LIMITATION"
    assert verdicts["l1b_exit_verdict"] == "L1B_DATA_NOT_READY"


def test_gate_met_and_affordable_confirms_convergence():
    args = tca.RunArgs(profile="forensic")
    analysis = _fake_analysis(
        window_gate={"spread_width": 0.005, "max_speed": 0.005},
        horizon_gate={"spread_width": 0.005, "max_speed": 0.005},
    )
    analysis["cost_accuracy_matrix"]["0.0005"]["projected_full_horizon_seconds"] = 400.0
    verdicts = tca.build_policy_and_verdicts(args, analysis, _fake_mechanism())
    assert verdicts["temporal_verdict"] == "TEMPORAL_CONVERGENCE_CONFIRMED"
    assert verdicts["selected_action"] == "PROMOTE_SMALLER_DT_POLICY"


def test_stale_cache_rejected(tmp_path):
    npz_path = tmp_path / "run.npz"
    binding_path = tmp_path / "run.binding.json"
    expect = {"name": "run", "N": 192}
    binding_path.write_text(json.dumps({"name": "run", "N": 64}))  # stale binding
    npz_path.write_bytes(b"placeholder")
    assert tca._load_cache(npz_path, binding_path, expect) is None
    binding_path.write_text(json.dumps(expect))  # matching binding, no npz content check needed
    binding_path.unlink()
    assert tca._load_cache(npz_path, binding_path, expect) is None


def test_contract_remains_12_and_policy_registry_intact():
    assert int(pf.SOLVER_CONTRACT_VERSION) == 12
    assert timestep_policy.DEFAULT_POLICY_NAME == "impact_phase_cap_dx2_v1"


def test_w_contact_angle_stays_open_and_unmeasured_stays_unmeasured():
    args = tca.RunArgs(profile="forensic")
    analysis = _fake_analysis()
    mechanism = _fake_mechanism()
    verdicts = tca.build_policy_and_verdicts(args, analysis, mechanism)
    assert verdicts["blockers"]["W_CONTACT_ANGLE"].startswith("OPEN")
    reaudit = tca.build_exit_reaudit(analysis, verdicts)
    assert reaudit["categories"]["FRESH_TRAINING_AGGREGATE_LINEAGE"].startswith("UNMEASURED")
    assert reaudit["categories"]["EXTERNAL_DYNAMIC_VALIDATION"].startswith("UNMEASURED")


def test_integrator_map_binds_to_live_source():
    mapping = tca.build_integrator_map()
    assert "length=3" in mapping["anchors_bound_to_live_source"]["three_substeps_lax_scan_length3"]
    assert "first-order" in mapping["public_step"]["order_verdict"]


def test_assemble_refuses_non_contract_12(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "SOLVER_CONTRACT_VERSION", 13)
    monkeypatch.setattr(tca, "EVIDENCE_ROOT", tmp_path)
    monkeypatch.setattr(tca, "_load_stage", lambda stage: {})
    with pytest.raises(tca.AuditValidationError, match="contract 12"):
        tca.assemble(tca.RunArgs(profile="quick"))
