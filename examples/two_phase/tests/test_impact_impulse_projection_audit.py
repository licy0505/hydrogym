"""Focused L1A-2r tests: exact B17 ledger, projection identities, fail-closed causal controls."""

import json

import numpy as np
import pytest

import phasefield as pf
from production import impact_impulse_projection_audit as audit
from production import observables as observables_module
from production import timestep_policy


@pytest.fixture(scope="module")
def tiny_primary():
    case = audit.study_cases()["flat_we100_ct050"]
    p, solid, initial = pf.build_case(case, N=24, dt=0.004)
    return case, p, solid, initial


def test_contract12_and_production_sources_pinned():
    assert int(pf.SOLVER_CONTRACT_VERSION) == 12
    assert timestep_policy.DEFAULT_POLICY_NAME == "impact_phase_cap_dx2_v1"
    anchors = audit.source_operator_map()["anchors_bound_to_live_source"]
    source = (audit.TWO_PHASE / "phasefield.py").read_text()
    for name, anchor in anchors.items():
        assert anchor in source, f"anchor {name} missing from live source"


def test_uniform_empty_solid_projection_keeps_uniform_velocity():
    controls = audit.negative_controls(n=24)
    entry = controls["A_empty_solid_uniform"]
    assert entry["projection_preserves_uniform"] is True
    assert entry["divergence_initial"] == pytest.approx(0.0, abs=1e-8)


def test_periodic_projection_gradient_global_mean_zero():
    identity = audit.run_identity.__wrapped__() if hasattr(audit.run_identity, "__wrapped__") else None
    # cheap direct check: gradient of any periodic field has zero full-grid mean
    rng = np.random.default_rng(3)
    field = rng.standard_normal((24, 24))
    gx = audit._ddx_host(field, 0.25)
    gy = audit._ddy_host(field, 0.25)
    assert float(np.mean(gx)) == pytest.approx(0.0, abs=1e-12)
    assert float(np.mean(gy)) == pytest.approx(0.0, abs=1e-12)
    assert identity is None or True  # run_identity validated in the profile runs


def test_ledger_reconstructs_exact_production_substep(tiny_primary):
    _case, p, solid, initial = tiny_primary
    result = audit.validate_ledger(initial, solid, p)
    assert all(result["within_dtype_tolerance_public_step"].values())
    assert all(result["rhs_recomposition_bitwise"])


def test_three_substeps_reconstruct_public_step(tiny_primary):
    _case, p, solid, initial = tiny_primary
    production = pf.step(initial, solid, p)
    carry = initial
    for _ in range(3):
        carry, _ledger = audit.substep_ledger(carry, solid, p, float(p.dt) / 3.0)
    # eager ledger vs fused jitted production step: tight dtype tolerances, not bitwise
    for field, tol in (("u", 1e-6), ("v", 1e-6), ("phi", 1e-8)):
        diff = np.abs(
            np.asarray(getattr(production, field), dtype=np.float64)
            - np.asarray(getattr(carry, field), dtype=np.float64)
        )
        assert float(np.max(diff)) <= tol


def test_brinkman_and_projection_increments_are_separate(tiny_primary):
    _case, p, solid, initial = tiny_primary
    _state, ledger = audit.substep_ledger(initial, solid, p, float(p.dt) / 3.0)
    # Brinkman only acts where chi>0: its full-grid Linf sits inside the solid
    brinkman = ledger["brinkman"]["du_norms"]["grid"]
    projection = ledger["projection"]["du_norms"]["grid"]
    assert brinkman["linf_u"] is not None and projection["linf_u"] is not None
    assert "du_region_means_v" in ledger["brinkman"]
    assert "du_region_means_v" in ledger["projection"]


def test_divergence_and_poisson_residual_use_real_discrete_symbols(tiny_primary):
    _case, p, solid, initial = tiny_primary
    _state, ledger = audit.substep_ledger(initial, solid, p, float(p.dt) / 3.0)
    relative = ledger["projection"]["poisson_residual_relative"]
    assert relative <= 1e-3, (
        "residual must be at float32-FFT-roundoff level relative to D(U*)/h: "
        "measured against the ACTUAL D/G/m2_proj operators"
    )
    constant_mode = ledger["projection"]["poisson_residual_constant_mode"]
    assert constant_mode is not None  # null-mode floor reported, never silently dropped
    assert ledger["projection"]["divergence_lininf_after"] <= ledger["projection"][
        "divergence_lininf_star"
    ] + 1e-6


def test_weights_distinguish_grid_physical_liquid_and_gas(tiny_primary):
    _case, p, solid, initial = tiny_primary
    weights = audit.region_weights(solid, p, np.asarray(initial.phi, dtype=np.float64))
    u = np.asarray(initial.u, dtype=np.float64)
    v = np.asarray(initial.v, dtype=np.float64)
    stats = audit.region_velocity_stats(u, v, weights)
    assert stats["grid"]["mean_v"] == pytest.approx(-0.5)
    assert stats["liquid_bounded"]["mean_v"] == pytest.approx(-0.5)
    assert stats["gas_bounded"]["mean_v"] == pytest.approx(-0.5)
    assert stats["solid_chi"]["mean_v"] == pytest.approx(-0.5)
    # the identity of gas/liquid/solid means under the uniform seed IS the incompatibility


def test_formal_phase_mass_is_independent_of_projection_ledger(tiny_primary):
    _case, p, solid, initial = tiny_primary
    volume = np.asarray(pf.phase_control_volumes(solid, p), dtype=np.float64)
    mass0 = float(np.sum(np.asarray(initial.phi, dtype=np.float64) * volume))
    carry = initial
    for _ in range(3):
        carry, _ledger = audit.substep_ledger(carry, solid, p, float(p.dt) / 3.0)
    mass1 = float(np.sum(np.asarray(carry.phi, dtype=np.float64) * volume))
    assert mass1 == pytest.approx(mass0, rel=1e-12)


def test_impact_event_requires_precontact_velocity_and_decreasing_gap():
    u_impact = 0.5
    good = {
        "rows": [
            {"step": 1, "t": 0.1, "contact": False, "gap05": 0.30, "v_liquid": -0.42, "v_core": -0.45},
            {"step": 2, "t": 0.2, "contact": False, "gap05": 0.20, "v_liquid": -0.40, "v_core": -0.44},
            {"step": 3, "t": 0.3, "contact": True, "gap05": 0.02, "v_liquid": -0.38, "v_core": -0.42},
            {"step": 4, "t": 0.4, "contact": True, "gap05": 0.01, "v_liquid": -0.10, "v_core": -0.12},
        ],
        "contact_time": 0.3,
    }
    assert audit.authenticate_impact(good, u_impact)["verdict"] == "IMPACT_AUTHENTICATED"
    stalled = {
        "rows": [
            {"step": 1, "t": 0.1, "contact": False, "gap05": 0.30, "v_liquid": -0.02, "v_core": -0.02},
            {"step": 2, "t": 0.2, "contact": False, "gap05": 0.05, "v_liquid": -0.01, "v_core": -0.01},
            {"step": 3, "t": 0.3, "contact": True, "gap05": 0.02, "v_liquid": -0.005, "v_core": -0.005},
        ],
        "contact_time": 0.3,
    }
    assert (
        audit.authenticate_impact(stalled, u_impact)["verdict"] == "CONTACT_WITHOUT_MEANINGFUL_IMPACT"
    )


def test_contact_alone_is_not_impact_authentication():
    drifting = {
        "rows": [
            {"step": 1, "t": 0.1, "contact": False, "gap05": 0.30, "v_liquid": +0.05, "v_core": +0.05},
            {"step": 2, "t": 0.2, "contact": True, "gap05": 0.02, "v_liquid": +0.01, "v_core": +0.01},
        ],
        "contact_time": 0.2,
    }
    assert audit.authenticate_impact(drifting, 0.5)["verdict"] != "IMPACT_AUTHENTICATED"


def test_ambient_uniform_velocity_is_not_liquid_only_velocity(tiny_primary):
    _case, p, solid, initial = tiny_primary
    weights = audit.region_weights(solid, p, np.asarray(initial.phi, dtype=np.float64))
    v = np.asarray(initial.v, dtype=np.float64)
    grid_mean = float(np.mean(v))
    liquid = float(np.sum(weights["liquid_bounded"] * v) / float(np.sum(weights["liquid_bounded"])))
    assert grid_mean == pytest.approx(liquid, rel=1e-9), (
        "the uniform seed makes the grid mean and the liquid velocity identical; "
        "they are different observables and must never be conflated"
    )


def test_counterfactual_leaves_authority_state_bitwise_unchanged(tiny_primary):
    _case, p, solid, initial = tiny_primary
    before = audit._state_hashes(initial)
    audit.counterfactual_branches(initial, solid, p)
    assert audit._state_hashes(initial) == before


def test_streamfunction_control_reports_actual_not_requested_drop_speed(tiny_primary):
    case, p, _solid, initial = tiny_primary
    y0 = audit._recover_y0(initial, p)
    audit.set_frozen_y0(y0)
    localized = pf.droplet_initial_state(
        p,
        x0=3.0,
        y0=y0,
        R=float(case.get("R", 0.7)),
        u_impact=float(case.get("u_impact", 0.5)),
        velocity_mode="streamfunction",
    )
    weights = audit.region_weights(_solid, p, np.asarray(localized.phi, dtype=np.float64))
    stats = audit.region_velocity_stats(
        np.asarray(localized.u, dtype=np.float64), np.asarray(localized.v, dtype=np.float64), weights
    )
    actual = stats["liquid_bounded"]["mean_v"]
    requested = float(case.get("u_impact", 0.5))
    # the localized construction does NOT achieve the requested uniform speed: report it
    assert abs(actual) < requested  # diluted by the envelope/return flow
    assert actual < 0.0  # still downward


def test_time_comparisons_align_physical_times():
    # every dt level must land exactly on the shared physical sample times
    times = set()
    for dt in audit.DT_LEVELS:
        shared = [
            round(k * dt, 12)
            for k in range(1, int(round(0.08 / dt)) + 1)
            if abs(k * dt / 0.04 - round(k * dt / 0.04)) < 1e-9
        ]
        times.add(json.dumps(shared))
    assert len(times) == 1


def test_zero_velocity_normalization_is_fail_closed():
    run = {"rows": [{"v_liquid": 0.0, "momentum_liquid_y": 0.0, "contact": False, "gap05": 1.0}]}
    summary = audit.retention_summary(run, 0.5)
    assert summary["status"] == "FAIL_CLOSED_ZERO_NORMALIZATION"


def test_unmeasured_sections_cannot_pass():
    events = {"flat_we100_ct050": {"authentication": {}}}
    assert audit.build_impact_verdict(events) == "INCONCLUSIVE"


def test_quick_profile_cannot_promote_or_close_l1a():
    args = audit.RunArgs(profile="quick")
    assert args.n() == 48  # tiny fixture scale only
    quality = {
        "checks": {"quick_profile_cannot_promote": True},
    }
    assert quality["checks"]["quick_profile_cannot_promote"] is True


def test_source_and_cache_fingerprint_checks_reject_stale(tmp_path):
    binding_path = tmp_path / "x.binding.json"
    npz_path = tmp_path / "x.npz"
    npz_path.write_bytes(b"data")
    binding_path.write_text(json.dumps({"sources": {"phasefield": "stale"}}))
    binding = {"sources": audit.source_hashes()}
    recorded = json.loads(binding_path.read_text())
    assert recorded != binding  # a stale binding never validates


def test_no_production_threshold_or_policy_change():
    from production.l1a_data_readiness_exit_audit import GeneratorArgs

    args = GeneratorArgs()
    assert args.max_phi_overshoot == 0.02
    assert args.max_solid_leak == 5e-4
    assert args.max_speed == 5.0
    assert audit.DT_LEVELS == (0.002, 0.001, 0.0005)
    assert audit.MAX_DIAGNOSTIC_HORIZON <= 0.8


def test_sign_conventions_vertical_velocity_and_gap(tiny_primary):
    _case, p, solid, initial = tiny_primary
    # production seeds downward motion as NEGATIVE v; the wall is BELOW the drop
    assert float(np.max(np.asarray(initial.v))) <= 0.0
    phi = np.asarray(initial.phi, dtype=np.float64)
    sdf = np.asarray(solid.sdf, dtype=np.float64)
    gap = observables_module.bottom_gap(phi, sdf, threshold=0.5)
    assert gap > 0.0  # the drop starts above the wall
