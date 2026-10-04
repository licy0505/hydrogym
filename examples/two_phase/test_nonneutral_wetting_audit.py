"""L1A-2d tests: CH-only path, Young boundary residual, contact-line kinetics, classification, report schema.

Everything here is tiny (N <= 48, <= 400 steps); the 10k-50k matrices are never run in CI.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import phasefield as pf
import pytest
from production import contact_line_kinetics as clk
from production import nonneutral_wetting_audit as audit

HERE = Path(__file__).resolve().parent


def _tiny(target=150.0, N=32, R=0.6, dt=4.0e-3, M=2.0e-3):
    p = pf.PhaseFieldParams(Nx=N, Ny=N, Lx=6.0, Ly=6.0, dt=dt, M=M, eps=2.0 * 6.0 / N, dtype=jnp.float32)
    solid = pf.make_solid(pf.surface_flat(p, wall_height=0.25), p, cos_theta=math.cos(math.radians(target)))
    state = pf.sessile_initial_state(p, solid, R=R, wall_height=0.25)
    return p, solid, state


# ---------------------------------------------------------------------------
#  CH-only path
# ---------------------------------------------------------------------------
def test_ch_only_step_keeps_velocity_zero():
    p, solid, state = _tiny()
    # Even a non-zero starting velocity must be discarded: momentum is never advanced.
    state = pf.State(phi=state.phi, u=jnp.ones_like(state.phi), v=jnp.ones_like(state.phi), t=0.0)
    for _ in range(3):
        state, info = pf.phase_only_step_with_diagnostics(state, solid, p)
    assert float(jnp.max(jnp.abs(state.u))) == 0.0 and float(jnp.max(jnp.abs(state.v))) == 0.0
    assert bool(jnp.all(info.implicit_converged))
    assert float(state.t) == pytest.approx(3 * p.dt, rel=1e-6)


def _ch_only_versus_substeps(target=120.0, N=32, dtype=jnp.float32):
    """Max |phi| difference between the CH-only public step and three un-fused substeps."""
    p = pf.PhaseFieldParams(Nx=N, Ny=N, Lx=6.0, Ly=6.0, dt=4.0e-3, M=2.0e-3, eps=2.0 * 6.0 / N, dtype=dtype)
    solid = pf.make_solid(pf.surface_flat(p, wall_height=0.25), p, cos_theta=math.cos(math.radians(target)))
    state = pf.sessile_initial_state(p, solid, R=0.6, wall_height=0.25)
    zero = jnp.zeros_like(state.phi)
    expected = state.phi
    for _ in range(3):
        expected, _info = pf.phase_transport_step(expected, zero, zero, solid, p, dt=p.dt / 3.0)
    got, _ = pf.phase_only_step_with_diagnostics(state, solid, p)
    difference = np.asarray(got.phi, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
    return float(np.max(np.abs(difference))), got, expected, p, solid


def test_ch_only_step_reuses_the_exact_production_phase_operator():
    """CH-only == three ``phase_transport_step`` substeps with u = v = 0.

    The two paths are the same code; what is left is XLA fusion round-off. In float32 the
    scan-fused body and the un-fused loop differ by at most one float32 ulp of ``phi`` near the
    wall (measured 1.9e-9 at N=32), which the contract-v8 wall term makes visible because it is
    ~1/f stronger than the v7 kernel's and its ``A_wall/(dx dy)`` factor rounds differently under
    fusion (the v7 expression happened to round identically, 7e-15). In float64 the same
    comparison agrees to 3.4e-21, which is the round-off-immune statement of the identity.
    """
    difference, got, expected, p, solid = _ch_only_versus_substeps(dtype=jnp.float32)
    ulp = float(np.spacing(np.max(np.abs(np.asarray(expected, dtype=np.float32))).astype(np.float32)))
    assert difference <= 4.0 * max(ulp, 1e-9)  # a few float32 ulps of the largest phi
    # The same "a few ulps of phi" bound has to govern the elementwise statement too: one ulp of
    # the largest phi (~0.86) is 6e-8, so a flat 1e-8 is *below* one ulp and cannot be met by two
    # differently fused programs. Contract v10 splits the RHS into its conserved and orthogonal
    # parts, which changes how XLA fuses the recurrence and therefore which of the few-ulp
    # roundings land where; the operator itself is the same code.
    np.testing.assert_allclose(np.asarray(got.phi), np.asarray(expected), rtol=0.0, atol=4.0 * max(ulp, 1e-9))
    # and it differs from the legacy projection path by construction (no redistribution is called)
    assert p.phase_boundary_model == "impermeable_flux" and p.enforce_solid_phi is False

    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        difference64, *_rest = _ch_only_versus_substeps(dtype=jnp.float64)
    finally:
        jax.config.update("jax_enable_x64", previous)
    assert difference64 <= 1.0e-15  # same operator, same math: float64 agrees to round-off


def _ch_only_drift(rtol: float, steps: int = 30):
    """Fluid-mass drift of the CH-only path at a given implicit-solve tolerance."""
    p = pf.PhaseFieldParams(
        Nx=32,
        Ny=32,
        Lx=6.0,
        Ly=6.0,
        dt=4.0e-3,
        M=2.0e-3,
        eps=2.0 * 6.0 / 32,
        dtype=jnp.float32,
        ch_solver_rtol=rtol,
        ch_solver_max_iterations=400,
    )
    solid = pf.make_solid(pf.surface_flat(p, wall_height=0.25), p, cos_theta=math.cos(math.radians(60.0)))
    state = pf.sessile_initial_state(p, solid, R=0.6, wall_height=0.25)
    # contract v9: the conserved quantity is sum_i V_i phi_i on the cut-cell control volumes, and
    # the leak is measured on the cells that own *no* control volume (the centre mask is no longer
    # the transported domain: a cut cell with a solid centre legitimately holds its fluid)
    volume = np.asarray(pf.phase_control_volumes(solid, p), dtype=np.float64)
    mass0 = float(np.sum(np.asarray(state.phi, dtype=np.float64) * volume))
    step = jax.jit(pf.phase_only_step, static_argnums=(2,))
    for _ in range(steps):
        state = step(state, solid, p)
    phi = np.asarray(state.phi, dtype=np.float64)
    zero_volume = volume <= 0.0
    leak = float(np.sum(np.maximum(phi[zero_volume], 0.0))) / float(np.sum(np.maximum(phi, 0.0)))
    return abs(float(np.sum(phi * volume)) - mass0) / mass0, leak


def test_ch_only_mass_conservation():
    """CH-only transport conserves ``sum_i V_i phi_i`` down to the float32 round-off floor.

    Contract v10 poses the weighted solve for the substep exchange and removes the ``sqrt(V)``
    similarity pair, so the residual is no longer a *truncation* term that follows the CG
    tolerance: measured on this fixture the drift is 3.3e-9 at rtol = 1e-6 (v9: 1.4e-5) and
    2.5e-7 at rtol = 1e-8 (v9: 2.1e-6). It is a rounding walk, so it is *not* monotone in the
    tolerance, and the old "a tighter tolerance cannot drift more" premise -- a property of the
    v9 Krylov truncation -- is exactly what the L1A-2g mass-precision audit falsified (the drift
    matrix is non-monotone at every grid). Both rows are therefore bounded by the floor.
    """
    production_drift, production_leak = _ch_only_drift(1.0e-6)
    tight_drift, tight_leak = _ch_only_drift(1.0e-8)
    assert production_drift <= 1.0e-6  # round-off floor; v9 measured 1.4e-5 here
    assert tight_drift <= 1.0e-6
    assert production_leak <= 1e-6 and tight_leak <= 1e-6  # no solid leak at either tolerance


@pytest.mark.parametrize("target", [60.0, 120.0, 150.0])
def test_ch_only_energy_does_not_increase_tiny_case(target):
    p, solid, state = _tiny(target=target)
    step = jax.jit(pf.phase_only_step, static_argnums=(2,))
    energies = [float(pf.phase_free_energy(state.phi, solid, p))]
    for _ in range(25):
        state = step(state, solid, p)
        energies.append(float(pf.phase_free_energy(state.phi, solid, p)))
    scale = max(1.0, abs(energies[0]))
    assert max((b - a) / scale for a, b in zip(energies, energies[1:])) <= 1.0e-5
    assert energies[-1] < energies[0]


# ---------------------------------------------------------------------------
#  Young boundary residual (manufactured)
# ---------------------------------------------------------------------------
def _manufactured(N, theta, eps_factor=2.0):
    dx = 6.0 / N
    eps = eps_factor * dx
    x = (np.arange(N) + 0.5) * dx
    X, Y = np.meshgrid(x, x, indexing="ij")
    sdf = Y - 0.25
    phi = clk.manufactured_young_boundary_field(X, Y, sdf, eps, theta)
    return phi, sdf, dx, eps


def _residual(N, theta, wall_theta=None, eps_factor=2.0):
    phi, sdf, dx, eps = _manufactured(N, theta, eps_factor)
    cos = math.cos(math.radians(theta if wall_theta is None else wall_theta))
    return clk.young_boundary_residual(phi, sdf, dx, dx, eps, cos)


def test_young_boundary_residual_manufactured_flat_wall():
    """90 deg: g_w' == 0 and the manufactured normal derivative is zero -> residual at round-off."""
    res = _residual(128, 90.0)
    assert res["n_points"] >= 8
    assert res["RY_linf"] <= 1.0e-12


@pytest.mark.parametrize("theta", [60.0, 120.0, 150.0])
def test_young_boundary_residual_manufactured_non_neutral(theta):
    coarse, fine = _residual(64, theta), _residual(128, theta)
    assert fine["n_points"] >= 8
    assert fine["RY_normalized_l2"] <= 0.03  # truncation error only
    # The truncation error is O((dx/eps)^2): scale invariant at fixed eps/dx, smaller for a thicker interface.
    assert fine["RY_normalized_l2"] == pytest.approx(coarse["RY_normalized_l2"], rel=0.1)
    assert _residual(128, theta, eps_factor=4.0)["RY_normalized_l2"] < 0.5 * fine["RY_normalized_l2"]
    # not vacuous: the wrong wall angle is detected at O(1)
    wrong = _residual(128, theta, wall_theta=180.0 - theta if theta != 90.0 else 60.0)
    assert wrong["RY_normalized_l2"] >= 0.4


def test_young_boundary_residual_manufactured_hydrophilic():
    assert _residual(128, 60.0)["RY_normalized_l2"] <= 0.03


def test_young_boundary_residual_manufactured_hydrophobic():
    assert _residual(128, 120.0)["RY_normalized_l2"] <= 0.03
    assert _residual(128, 150.0)["RY_normalized_l2"] <= 0.03


def test_young_boundary_residual_manufactured_audit_gate():
    result = clk.audit_young_boundary_residual(N_values=(64, 128))
    assert result["passed"] is True


# ---------------------------------------------------------------------------
#  Contact-line kinematics
# ---------------------------------------------------------------------------
def _cap(p, x0, R=1.1, wall_y=0.25):
    X, Y = (np.asarray(a) for a in pf.grids(p))
    sdf = np.asarray(pf.surface_flat(p, wall_height=wall_y))
    rel = ((X - x0 + 0.5 * p.Lx) % p.Lx) - 0.5 * p.Lx
    r = np.sqrt(rel**2 + (Y - wall_y) ** 2)
    return np.where(sdf >= 0.0, 0.5 * (1.0 - np.tanh((r - R) / (math.sqrt(2.0) * p.eps))), 0.0), sdf


def test_contact_line_positions_periodic_safe():
    p = pf.PhaseFieldParams(Nx=96, Ny=96, Lx=6.0, Ly=6.0, eps=2.0 * 6.0 / 96)
    centre, sdf = _cap(p, 3.0)
    seam, _ = _cap(p, 0.2)  # the drop straddles x = 0 / x = Lx
    a = clk.contact_line_positions(centre, sdf, p.dx, p.dy, eps=p.eps, Lx=p.Lx)
    b = clk.contact_line_positions(seam, sdf, p.dx, p.dy, eps=p.eps, Lx=p.Lx)
    assert a["contact_line_exists"] and b["contact_line_exists"]
    assert b["contact_width"] == pytest.approx(a["contact_width"], abs=2e-3)
    assert a["contact_width"] == pytest.approx(2.2, abs=0.02)  # hemisphere of R = 1.1
    assert b["left_contact_x"] < 0.0 < b["right_contact_x"]  # unwrapped about the periodic centroid
    assert (b["left_contact_x_wrapped"] - 0.2 + 3.0) % 6.0 - 3.0 == pytest.approx(-1.1, abs=0.02)


def test_contact_line_detachment_is_explicit():
    p = pf.PhaseFieldParams(Nx=96, Ny=96, Lx=6.0, Ly=6.0, eps=2.0 * 6.0 / 96)
    X, Y = (np.asarray(a) for a in pf.grids(p))
    sdf = np.asarray(pf.surface_flat(p, wall_height=0.25))
    phi = 0.5 * (1.0 - np.tanh((np.sqrt((X - 3.0) ** 2 + (Y - 3.0) ** 2) - 1.0) / (math.sqrt(2.0) * p.eps)))
    pos = clk.contact_line_positions(phi, sdf, p.dx, p.dy, eps=p.eps, Lx=p.Lx)
    assert pos["contact_line_exists"] is False and pos["detachment_observed"] is True
    assert pos["sessile_angle_deg"] is None  # a circle fit is no longer a sessile angle
    assert pos["bottom_gap"] > 1.0


def test_contact_line_velocity_on_synthetic_translation():
    p = pf.PhaseFieldParams(Nx=96, Ny=96, Lx=6.0, Ly=6.0, eps=2.0 * 6.0 / 96)
    times = [0.0, 1.0, 2.0, 3.0, 4.0]
    speed = 0.3
    left, right = [], []
    for t in times:
        phi, sdf = _cap(p, (5.5 + speed * t) % p.Lx)  # crosses the periodic seam
        pos = clk.contact_line_positions(phi, sdf, p.dx, p.dy, eps=p.eps, Lx=p.Lx)
        left.append(pos["left_contact_x_wrapped"])
        right.append(pos["right_contact_x_wrapped"])
    vel = clk.contact_line_velocity(times, left, right, Lx=p.Lx)
    assert vel["mean_translation_velocity"] == pytest.approx(speed, abs=5e-3)
    assert all(abs(v - speed) < 2e-2 for v in vel["left_velocity"])
    assert all(abs(v - speed) < 2e-2 for v in vel["right_velocity"])
    assert vel["final_mean_speed"] == pytest.approx(speed, abs=2e-2)


def test_contact_line_velocity_synthetic_spreading():
    times = np.linspace(0.0, 4.0, 9)
    left = 3.0 - (1.0 + 0.1 * times)
    right = 3.0 + (1.0 + 0.1 * times)
    vel = clk.contact_line_velocity(times, left, right, Lx=6.0)
    assert np.allclose(vel["spreading_rate"], 0.1, atol=1e-9)
    assert np.allclose(vel["translation_velocity"], 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
#  Trends, fits, classification
# ---------------------------------------------------------------------------
def test_late_time_trend_known_signal():
    t = np.linspace(0.0, 10.0, 101)
    y = np.where(t < 8.0, 50.0, 50.0 - 2.5 * (t - 8.0))  # plateau then linear decay
    trend = clk.late_time_linear_trend(t, y, fraction=0.2)
    assert trend["slope"] == pytest.approx(-2.5, rel=1e-6)
    assert trend["r_squared"] == pytest.approx(1.0, abs=1e-9)
    assert trend["sign"] == -1
    flat = clk.late_time_linear_trend(t, np.full_like(t, 3.0))
    assert flat["sign"] == 0 and flat["slope"] == pytest.approx(0.0, abs=1e-12)


def test_relaxation_fit_does_not_claim_equilibrium():
    t = np.linspace(0.0, 20.0, 60)
    theta = 70.0 + 20.0 * np.exp(-t / 6.0)
    fit = clk.fit_relaxation_asymptote(t, theta, target_deg=60.0)
    assert fit["theta_inf_fit"] == pytest.approx(70.0, abs=0.5)
    assert fit["tau_fit"] == pytest.approx(6.0, rel=0.15)
    assert fit["fit_r_squared"] > 0.999
    assert fit["diagnostic_only"] is True and fit["is_equilibrium_angle"] is False
    assert fit["used_for_convergence_gate"] is False


def test_equilibration_classification_known_synthetic_cases():
    classify = clk.classify_equilibration
    assert classify({"detachment_observed": True})[0] == "TOPOLOGY_CHANGE_OR_DETACHMENT"
    assert classify({"contact_line_exists": False})[0] == "TOPOLOGY_CHANGE_OR_DETACHMENT"
    assert classify({"dt_angle_spread_deg": 6.0})[0] == "TIME_STEP_LIMITED"
    assert classify({"resolution_error_systematically_decreases": True})[0] == "RESOLUTION_LIMITED"
    assert classify({"ch_only_converged_near_target": True, "chns_converged_near_target": False})[0] == (
        "HYDRODYNAMIC_COUPLING_LIMITED"
    )
    # converged near the target after a slow approach, small wall residual, M*t collapse
    kin = classify({"converged": True, "equilibrium_error_deg": 1.2, "RY_normalized_l2": 0.02})
    assert kin[0] == "KINETICS_LIMITED"
    slow = classify(
        {
            "converged": False,
            "relaxing_toward_target": True,
            "mt_curves_collapse": True,
            "RY_normalized_l2": 0.03,
            "higher_m_converges_near_target": True,
        }
    )
    assert slow[0] == "KINETICS_LIMITED"
    # converged to a wrong angle with a small residual -> thermodynamic bias, *not* "not converged"
    biased = classify({"converged": True, "equilibrium_error_deg": 18.0, "RY_normalized_l2": 0.02})
    assert biased[0] == "THERMODYNAMIC_EQUILIBRIUM_BIASED"
    ablation = classify(
        {
            "converged": True,
            "equilibrium_error_deg": -14.0,
            "RY_normalized_l2": 0.4,
            "gain_ablation_restores_target": True,
            "wall_kernel_fluid_fraction": 0.61,
        }
    )
    assert ablation[0] == "BOUNDARY_DISCRETIZATION_LIMITED"
    wall = classify({"converged": False, "RY_normalized_l2": 0.6, "ry_plateau": True})
    assert wall[0] == "BOUNDARY_DISCRETIZATION_LIMITED"
    assert classify({"converged": False, "RY_normalized_l2": 0.6})[0] == "INCONCLUSIVE"  # transient residual only
    for evidence in ({}, {"converged": True, "equilibrium_error_deg": 9.0}, {"relaxing_toward_target": True}):
        assert classify(evidence)[0] in clk.ALLOWED_CLASSIFICATIONS


def _row(i, theta, energy, rate=1e-4, speed=0.0, dmt=2.0e-3):
    return {
        "step": i,
        "mobility_scaled_time": i * dmt,
        "measured_angle_deg": theta,
        "free_energy": energy,
        "phase_rate_l2": rate,
        "max_speed": speed,
    }


def test_stable_wrong_angle_is_converged_with_signed_error():
    n = 40  # window = 0.05 M*t = 25 samples of 2e-3
    rows = [_row(i, 78.0 + 1e-3 * (-1) ** i, 0.7 - 1e-9 * i) for i in range(n)]
    verdict = audit._window_converged(rows, True, audit.CRITERIA)
    assert verdict["converged"] is True  # stable at 78 deg although the target is 60 deg
    drifting = [_row(i, 78.0 - 0.2 * i, 0.7 - 1e-9 * i) for i in range(n)]
    assert audit._window_converged(drifting, True, audit.CRITERIA)["converged"] is False
    energetic = [_row(i, 78.0, 0.7 - 5e-3 * i) for i in range(n)]
    assert audit._window_converged(energetic, True, audit.CRITERIA)["converged"] is False
    # CHNS additionally needs a small speed; CH-only must not use max_speed as a gate
    fast = [_row(i, 78.0, 0.7, speed=1e-2) for i in range(n)]
    assert audit._window_converged(fast, False, audit.CRITERIA)["converged"] is False
    assert audit._window_converged(fast, True, audit.CRITERIA)["converged"] is True


def test_stationarity_window_is_fixed_in_mobility_scaled_time():
    """Regression: a slow drift of 0.02 deg per old 5-sample window (15 deg per unit M*t) must not converge."""
    slow = [_row(i, 80.0 - 0.01 * i, 0.7 - 1e-9 * i, dmt=1.6e-3) for i in range(300)]
    verdict = audit._window_converged(slow, True, audit.CRITERIA)
    assert verdict["angle_ok"] is False and verdict["converged"] is False
    short = [_row(i, 78.0, 0.7) for i in range(10)]  # covers only 0.018 M*t < the window
    assert audit._window_converged(short, True, audit.CRITERIA)["converged"] is False
    # phase rate is judged in M-scaled units: a small-M run cannot pass the rate gate trivially
    quiet = [_row(i, 78.0, 0.7, rate=5e-4) for i in range(40)]
    assert audit._window_converged(quiet, True, audit.CRITERIA, M=audit.M_REF * 0.25)["rate_ok"] is False
    assert audit._window_converged(quiet, True, audit.CRITERIA, M=audit.M_REF)["rate_ok"] is True


def test_wall_kernel_fluid_fraction_is_reported():
    p = pf.PhaseFieldParams(Nx=128, Ny=128, Lx=6.0, Ly=6.0)
    sdf = np.asarray(pf.surface_flat(p, wall_height=0.25))
    info = clk.fluid_wall_delta_integral(sdf, p.dx, p.dy)
    assert info["total_normal_integral"] == pytest.approx(0.99925, abs=1e-4)
    assert info["fluid_side_normal_integral"] == pytest.approx(0.6137, abs=1e-3)
    assert info["fluid_side_normal_integral"] + info["solid_side_normal_integral"] == pytest.approx(
        info["total_normal_integral"], rel=1e-12
    )


# ---------------------------------------------------------------------------
#  Report / contract
# ---------------------------------------------------------------------------
def test_nonneutral_audit_report_schema(tmp_path):
    cfg = audit.PROFILES["quick"]
    kw = dict(N=32, eps_factor=2.0, dt=4e-3, R=0.6, sample_every=20, budgets=[40, 60])
    sections = {
        "primary": [
            audit.run_relaxation(t, ch_only=mode, group="primary", label="primary", **kw)
            for t in (60.0, 150.0)
            for mode in (True, False)
        ],
        "mobility": [audit.run_relaxation(150.0, ch_only=True, M=4e-3, group="mobility", label="M_x2", **kw)],
        "gain_ablation": [
            audit.run_relaxation(
                150.0, ch_only=True, M=4e-3, wall_gain=1.6, group="wall_gain_ablation", label="gain", **kw
            )
        ],
    }
    manufactured = clk.audit_young_boundary_residual(N_values=(64, 128))
    report = audit.assemble_report(sections, "quick", cfg, manufactured)
    assert audit.validate_report(report) == []
    assert report["stage"] == "L1A-2d"
    assert all(row["classification"] in (None, "INCONCLUSIVE") for row in report["classification_summary"].values())
    assert report["recommended_next_stage"]["decision"] == "INCONCLUSIVE"  # smoke budgets never classify
    assert report["solver_contract_version"] == pf.SOLVER_CONTRACT_VERSION == 10
    assert report["trajectory_semantics_changed"] is False  # the L1A-2d stage itself changes no default
    assert len(report["historical_v7_baseline"]) == 4
    assert {row["final_sampled_angle_deg"] for row in report["historical_v7_baseline"]} == {
        75.829,
        89.961,
        102.659,
        113.256,
    }
    for case in report["cases"]:
        assert case["implicit_residual_max"] <= 1.0e-6 + 1e-9
        assert case["solid_phase_fraction_max"] <= 1e-6
        assert case["converged"] is False and case["equilibrium_angle_deg"] is None  # 60 steps cannot converge
    assert json.loads(json.dumps(report, allow_nan=False))["stage"] == "L1A-2d"
    assert "BASELINE" not in json.dumps(report["recommended_next_stage"])  # decision is evidence-based, not canned
    # fail-closed checks of the validator itself
    broken = json.loads(json.dumps(report))
    broken["cases"][0]["equilibrium_angle_deg"] = 70.0
    assert any("without converging" in e for e in audit.validate_report(broken))
    broken = json.loads(json.dumps(report))
    broken["solver_contract_version"] = 7  # a v9 tree cannot produce a v7-contract report
    assert any("contract" in e for e in audit.validate_report(broken))
    broken = json.loads(json.dumps(report))
    # a contract-8 report may not claim the v9 transport geometry: the metadata is part of the lineage
    broken["solver_contract_version"] = 8
    assert any("contract" in e for e in audit.validate_report(broken))
    broken = json.loads(json.dumps(report))
    broken["cases"][0]["classification"] = "VAGUE_NEW_LABEL"
    assert any("classification" in e for e in audit.validate_report(broken))


def test_solver_contract_is_v9_and_l1a2d_stage_is_frozen():
    """The L1A-2d diagnostic stage is unchanged; the *solver* it diagnoses is now contract v9."""
    assert pf.SOLVER_CONTRACT_VERSION == 10
    assert audit.STAGE == "L1A-2d"
    p = pf.PhaseFieldParams(Nx=16, Ny=16)
    assert p.phase_boundary_model == "impermeable_flux" and p.wetting_model == "surface_energy"
    assert p.M == 2.0e-3 and p.eps == pytest.approx(1.5 * p.dx)
    assert pf.WALL_SIGMA0 == pytest.approx(math.sqrt(2.0) / 6.0)
    # the full CHNS public step is untouched: it still advances momentum
    p, solid, state = _tiny(target=60.0)
    out, _ = pf.step_with_diagnostics(state, solid, p)
    assert float(jnp.max(jnp.abs(out.u)) + jnp.max(jnp.abs(out.v))) > 0.0


def test_example_config_documents_the_baseline_profile():
    cfg = json.loads((HERE / "production" / "configs" / "nonneutral_equilibration.example.json").read_text())
    base = audit.PROFILES["baseline"]
    assert cfg["stage"] == "L1A-2d" and cfg["solver_contract_version"] == 7  # historical profile document
    assert cfg["targets_deg"] == base["targets"] and cfg["stage_step_budgets"] == base["budgets"]
    assert cfg["base_case"]["N"] == base["N"] and cfg["base_case"]["dt"] == base["dt"]
    assert cfg["sweeps"]["mobility"]["mobility_factors"] == base["mobility"]["factors"]
    assert cfg["sweeps"]["dt_sensitivity"]["dt_values"] == base["dt_sweep"]["dts"]
    assert cfg["equilibrium_criteria"]["window_samples"] == audit.CRITERIA["window_samples"]
    assert cfg["equilibrium_criteria"]["angle_tol_deg"] == audit.CRITERIA["angle_tol_deg"]
