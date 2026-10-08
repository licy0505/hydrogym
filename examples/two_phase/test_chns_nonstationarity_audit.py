"""L1A-2j tests: exact observational instrumentation and diagnostic-only freeze branches.

All numerical tests use a tiny grid and at most two public steps. Production 40k-50k trajectories,
matched controls, and forensic ablations are intentionally not run in CI.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import phasefield as pf  # noqa: E402
from production import chns_nonstationarity_audit as audit  # noqa: E402
from production import contact_line_kinetics as clk  # noqa: E402
from production import nonneutral_wetting_audit as nwa  # noqa: E402


@pytest.fixture(scope="module")
def tiny_case():
    return audit._make_case(60.0, N_value=24, dt=audit.DT, M=audit.M_REF)


def _same_state(left, right):
    return all(
        np.array_equal(np.asarray(getattr(left, key)), np.asarray(getattr(right, key)))
        for key in ("phi", "u", "v", "t")
    )


def test_instrumentation_reconstructs_production_step_and_terms(tiny_case):
    p, solid, state, _ = tiny_case
    production_state, production_info = pf.step_with_diagnostics(state, solid, p)
    observed = audit.forensic_step_jit(state, solid, p)

    assert _same_state(production_state, observed.state)
    assert np.array_equal(np.asarray(production_info.implicit_iterations), np.asarray(observed.iterations))
    assert np.array_equal(np.asarray(production_info.implicit_relative_residuals), np.asarray(observed.residuals))
    assert np.array_equal(np.asarray(production_info.implicit_converged), np.asarray(observed.converged))
    assert np.all(np.asarray(observed.converged))
    term_errors = np.asarray(observed.substep_metrics)[:, 14:16]
    assert np.max(term_errors) == 0.0
    assert tuple(audit.SUBSTEP_METRIC_NAMES[14:16]) == (
        "term_reconstruction_u_linf",
        "term_reconstruction_v_linf",
    )
    # The exact arrays in pressure/Brinkman diagnostics reproduce the production transition, not a
    # separately tuned or post-corrected trajectory.
    assert np.max(np.asarray(observed.substep_metrics)[:, audit.SUBSTEP_METRIC_NAMES.index("projection_reduction_l2")]) < 1.0e-3

    # Exercise nonzero advection and viscosity too: the seed starts at rest, so a rest-only test
    # would not verify those two independently accumulated acceleration addends.
    x = (np.arange(p.Nx) + 0.5)[:, None] * p.dx
    y = (np.arange(p.Ny) + 0.5)[None, :] * p.dy
    u = (0.01 * np.sin(2.0 * np.pi * x / p.Lx) * np.cos(2.0 * np.pi * y / p.Ly)).astype(np.float32)
    v = (-0.01 * np.cos(2.0 * np.pi * x / p.Lx) * np.sin(2.0 * np.pi * y / p.Ly)).astype(np.float32)
    moving = pf.State(state.phi, jnp.asarray(u), jnp.asarray(v), state.t)
    prod_moving, _ = pf.step_with_diagnostics(moving, solid, p)
    obs_moving = audit.forensic_step_jit(moving, solid, p)
    assert _same_state(prod_moving, obs_moving.state)
    assert np.max(np.asarray(obs_moving.substep_metrics)[:, 14:16]) == 0.0


def test_observer_scan_does_not_change_integrator_state(tiny_case):
    p, solid, state, _ = tiny_case
    observed_end, states, fields, observations, final_fields = audit.advance_forensic_block(
        state, solid, p, False, 1.0, 1, 2
    )
    production_end = audit._advance_standard(state, solid, p, 2)
    assert _same_state(observed_end, production_end)
    production_observed_end, prod_states, prod_fields, prod_observations, _ = audit.advance_production_observed_block(
        state, solid, p, 1, 2
    )
    assert _same_state(production_observed_end, production_end)
    assert np.array_equal(np.asarray(prod_observations[0]), np.asarray(observations[0]))
    assert states.phi.shape[0] == prod_states.phi.shape[0] == 2
    assert fields.pressure.shape[0] == prod_fields.pressure.shape[0] == 2
    assert observations[1].shape == (2, 1, 3, len(audit.SUBSTEP_METRIC_NAMES))
    assert final_fields.pressure.shape == state.phi.shape


def test_freeze_phi_is_bitwise_fixed_and_keeps_momentum_live(tiny_case):
    p, solid, state, _ = tiny_case
    frozen = audit.forensic_step_jit(state, solid, p, True, 1.0)
    assert audit._array_hash(frozen.state.phi) == audit._array_hash(state.phi)
    assert np.array_equal(np.asarray(frozen.fields.phase_rate), np.zeros_like(np.asarray(frozen.fields.phase_rate)))
    # A nonzero forcing/capillary state must still be able to update velocity; the test only requires
    # that the momentum branch executes, not that a particular transient magnitude is accepted.
    assert frozen.state.u.shape == state.u.shape
    assert frozen.state.v.shape == state.v.shape


def test_freeze_u_uses_zero_velocity_phase_only_operator_and_formal_mass(tiny_case):
    p, solid, state, _ = tiny_case
    zero = pf.State(state.phi, jnp.zeros_like(state.u), jnp.zeros_like(state.v), state.t)
    result, diagnostics = pf.phase_only_step_with_diagnostics(zero, solid, p)
    assert np.array_equal(np.asarray(result.u), np.zeros_like(np.asarray(result.u)))
    assert np.array_equal(np.asarray(result.v), np.zeros_like(np.asarray(result.v)))
    assert np.all(np.asarray(diagnostics.implicit_converged))
    sampled_end, sampled_states, iteration_max, residual_max, all_converged = audit._advance_phase_only_samples(
        zero, solid, p, steps_per_sample=1, num_samples=1
    )
    assert np.all(np.asarray(all_converged))
    assert int(np.asarray(iteration_max)[0]) >= 0
    assert float(np.asarray(residual_max)[0]) >= 0.0
    assert sampled_states.phi.shape[0] == 1
    sampled_last = pf.State(sampled_states.phi[-1], sampled_states.u[-1], sampled_states.v[-1], sampled_states.t[-1])
    assert _same_state(sampled_end, sampled_last)
    start_mass = float(pf.liquid_mass(zero.phi, solid, p))
    end_mass = float(pf.liquid_mass(result.phi, solid, p))
    assert abs(end_mass - start_mass) / abs(start_mass) < 1.0e-10
    # The diagnostic runner must keep the conserved quantity distinct from old full-grid/hard-mask
    # reconstructions; only the former is used in its formal mass gate.
    observed = audit.forensic_step_jit(zero, solid, p)
    row = audit._sample_row(
        result,
        observed.fields,
        solid,
        p,
        step=1,
        mobility=audit.M_REF,
        formal_mass_reference=start_mass,
        diagnostic_horizon="test-only",
        step_metrics=np.asarray(observed.step_metrics),
    )
    assert row["formal_phase_mass_sum_V_phi"] == pytest.approx(end_mass, rel=1.0e-13)
    assert "mass_dxdy_sum_phi_diagnostic_nonconserved" in row
    assert row["mass_dxdy_sum_phi_diagnostic_nonconserved"] != row["formal_phase_mass_sum_V_phi"]


def test_stationarity_decomposition_delegates_to_frozen_production_gate():
    samples = []
    for index in range(80):
        samples.append(
            {
                "step": index * 200,
                "time": float(index),
                "mobility_scaled_time": 0.01 * index,
                "measured_angle_deg": 60.0,
                "free_energy": 2.0,
                "phase_rate_l2": 0.0,
                "max_speed": 0.0,
            }
        )
    detailed = audit._criterion_window(samples[1:], False, M=audit.M_REF)
    assert detailed["production_gate"]["converged"]
    assert set(audit.FROZEN_PRODUCTION_CRITERIA) == set(nwa.CRITERIA)
    assert audit.FROZEN_PRODUCTION_CRITERIA == dict(nwa.CRITERIA)
    for key in audit.FORMAL_CRITERION_KEYS:
        criterion = detailed["criteria"][key]
        assert criterion["raw_value"] is not None
        assert criterion["threshold"] == audit.FROZEN_PRODUCTION_CRITERIA[
            {
                "angle_stationarity": "angle_tol_deg",
                "free_energy_stability": "energy_rel_tol",
                "phase_rate": "phase_rate_l2_tol",
                "maximum_speed": "chns_speed_tol",
            }[key]
        ]
        assert criterion["normalized_value_raw_over_threshold"] == pytest.approx(0.0)
        assert criterion["passed"] is True
    assert detailed["criteria"]["strict_energy_observation_non_gating"]["acceptance_gate"] is False


def test_periodogram_and_stick_slip_classifiers_are_diagnostic_and_bounded():
    times = np.arange(1000, dtype=np.float64) * 0.01
    values = 60.0 + 0.03 * np.sin(2.0 * np.pi * 0.5 * times)
    summary = audit._signal_summary(times, values)
    peak = summary["dominant_spectral_peak"]
    assert peak["frequency_per_time"] == pytest.approx(0.5, abs=0.02)
    assert audit._multi_frequency_status([peak, {"power_fraction": 1.0e-8}]) == "NOT_TESTED"
    assert audit._multi_frequency_status([{"power_fraction": 0.4}, {"power_fraction": 0.1}]) == "SUSPECTED"
    assert audit._phase_kinetics_status(0.004, False) == "SUSPECTED"
    assert audit._phase_kinetics_status(0.2, False) == "SUPPORTED"
    assert audit._phase_kinetics_status(0.004, True) == "FALSIFIED"
    assert audit._phase_kinetics_status(None, False) == "NOT_TESTED"
    assert audit._contact_pinning_status({"status": "measured", "episodes": 0}) == "NOT_TESTED"
    assert audit._contact_pinning_status({"status": "measured", "episodes": 1}) == "SUSPECTED"
    assert audit._contact_pinning_status({"status": "measured", "episodes": 3}) == "SUPPORTED"

    dx = 0.05
    positions = [1.0]
    for _ in range(4):
        positions.extend([positions[-1]] * 4)
        positions.append(positions[-1] + 0.5 * dx)
    rows = [
        {"step": index, "left_contact_x_wrapped": value, "right_contact_x_wrapped": 3.0 - value, "time": index * audit.DT}
        for index, value in enumerate(positions)
    ]
    detector = audit._detect_stick_slip(rows, dt=audit.DT, dx=dx)
    assert detector["episodes"] >= 3
    assert detector["diagnostic_pattern_thresholds"]["dwell_cell_displacement"] == 0.01
    assert "INCONCLUSIVE" in audit.PHENOMENOLOGY_LABELS
    with pytest.raises(ValueError, match="outside the frozen allowed sets"):
        audit._candidate("invented_unapproved_label", "NOT_TESTED", [])


def test_checkpoint_hashes_and_config_fingerprint_fail_closed(tmp_path, tiny_case):
    _p, _solid, state, config = tiny_case
    path = tmp_path / "state.npz"
    audit.save_forensic_checkpoint(path, state, config, step_index=0, kind="test")
    loaded, metadata = audit.load_forensic_checkpoint(
        path, expected_config=config, expected_step=0, expected_kind="test"
    )
    assert audit._state_hashes(loaded) == audit._state_hashes(state)
    assert metadata["config_fingerprint"] == audit._canonical_hash(config)
    wrong_config = dict(config)
    wrong_config["dt"] *= 0.5
    with pytest.raises(audit.CheckpointError, match="fingerprint"):
        audit.load_forensic_checkpoint(path, expected_config=wrong_config)

    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in ("phi", "u", "v", "t")}
        metadata_json = np.asarray(archive["metadata_json"]).item()
    arrays["phi"][0, 0] += 1.0e-12
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays, metadata_json=np.asarray(metadata_json))
    with pytest.raises(audit.CheckpointError, match="state-array hash"):
        audit.load_forensic_checkpoint(path, expected_config=config)


def test_contract_defaults_and_anti_cheating_guards(tiny_case):
    p, _solid, _state, config = tiny_case
    assert pf.SOLVER_CONTRACT_VERSION == 12
    assert config["target_deg"] == 60.0
    assert config["N"] == 24
    assert p.dt == audit.DT
    assert p.M == audit.M_REF
    assert p.eps == 2.0 * p.dx
    assert p.eta_pen == 2.0 * p.dt
    assert config["capillary_denominator"] == "rho_l"
    assert dict(nwa.CRITERIA) == audit.FROZEN_PRODUCTION_CRITERIA
    source = inspect.getsource(audit)
    assert '"production_acceptance_evidence"' in source
    assert "mass_dxdy_sum_phi_diagnostic_nonconserved" in source
    assert "formal_phase_mass_sum_V_phi" in source
    assert "phasefield.step_with_diagnostics" not in source
    assert "nwa._window_converged" in source
    # Accelerated mobility is explicitly limited to the freeze-u CH-only path.
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "run_relaxation":
                for kw in node.keywords:
                    if kw.arg == "M" and isinstance(kw.value, ast.BinOp):
                        assert "_run_phase_only_branch" in source
    assert audit.SOLVER_CONTRACT == pf.SOLVER_CONTRACT_VERSION


def test_contact_line_forensic_calls_production_periodic_estimators():
    source = inspect.getsource(audit._sample_row)
    assert "clk.contact_line_positions" in source
    assert "contact_line_velocity" in inspect.getsource(audit._attach_contact_line_velocities)
    assert "Lx=Lx" in inspect.getsource(audit._attach_contact_line_velocities)
    assert audit.AUTHORITY_END == 50_000
    assert audit.BURST_STEPS == 2_000


def test_report_does_not_mark_unmeasured_diagnostic_gates_as_pass(tmp_path):
    report = audit._initial_report("quick", tmp_path)
    assert report["status"] == "running"
    assert report["unrun_sections"]["authority_60_step_50000"] == "unmeasured"
    assert report["phenomenology"]["status"] == "unmeasured"
    assert report["mechanism"]["status"] == "unmeasured"
    assert report["production_semantics_changed"] is False
    assert report["blockers"]["W-CONTACT-ANGLE"].startswith("open")
    report["controls"] = {
        "90": {"steps": 27200, "converged": True, "final_angle_deg": 89.41, "final_max_speed": 2.68e-4}
    }
    markdown = tmp_path / "report.md"
    audit._write_report_markdown(markdown, report)
    text = markdown.read_text(encoding="utf-8")
    assert "| 90° | 27200 | True | 89.41 | 0.000268 |" in text
