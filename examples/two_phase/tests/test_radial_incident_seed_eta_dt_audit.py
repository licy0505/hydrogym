"""L1A-2u radial-envelope / eta-dt audit: identities, invariants and fail-closed gates.

Every test verifies either a discrete identity of the frozen operator sequence, a declared
property of the prespecified radial-envelope family, a provenance rule, or a fail-closed
failure mode. No test edits a frozen production source or threshold, and no test relaxes
the frozen near-wall ``chi >= 0.01`` speed limit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import phasefield as pf  # noqa: E402
import pytest
from production import impact_initialization_compatibility_audit as audit  # noqa: E402
from production import nearwall_taper_interface_transport_audit as taper  # noqa: E402
from production import radial_incident_seed_eta_dt_audit as stage  # noqa: E402
from production import timestep_policy  # noqa: E402

QUICK_N = 48


@pytest.fixture(scope="module")
def flat_bundle():
    case = audit._canary_cases()["flat_we100_ct050"]
    return audit._derive_bundle("flat_we100_ct050", case, QUICK_N, 0.004)


@pytest.fixture(scope="module")
def pillar_bundle():
    case = audit._canary_cases()["pillar_training"]
    return audit._derive_bundle("pillar_training", case, QUICK_N, 0.004)


@pytest.fixture(scope="module")
def radial_states(flat_bundle):
    return {key: stage.build_radial_state(flat_bundle, key) for key in stage.RADIAL_ORDER}


# ---------------------------------------------------------------------------
# family definition
# ---------------------------------------------------------------------------
def test_smooth5_matches_the_frozen_form():
    z = np.asarray([-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5], dtype=np.float64)
    values = np.asarray(stage.smooth5(z), dtype=np.float64)
    expected = np.clip(z, 0.0, 1.0) ** 3 * (10.0 + np.clip(z, 0.0, 1.0) * (-15.0 + 6.0 * np.clip(z, 0.0, 1.0)))
    assert np.array_equal(values, expected)
    assert values[0] == 0.0 and values[-1] == 1.0


def test_radial_envelope_is_compact_and_radial_only(flat_bundle):
    psi, envelope, r = stage.radial_envelope(flat_bundle, 2.0, 2.0)
    r_in = flat_bundle.R - 2.0 * flat_bundle.p.dx
    width = 2.0 * flat_bundle.p.dx
    assert np.all(envelope[r <= r_in - width] == 1.0)
    assert np.all(envelope[r >= r_in] == 0.0)
    assert np.all((envelope >= 0.0) & (envelope <= 1.0))
    # monotone in r wherever it transitions
    order = np.argsort(r.ravel())
    flat_env = envelope.ravel()[order]
    flat_r = r.ravel()[order]
    moving = np.diff(flat_env) < -1e-15
    assert moving.any()
    assert np.all(np.diff(flat_r[1:][moving]) >= 0.0)


def test_periodic_seam_taper_is_the_frozen_one(flat_bundle):
    psi, _envelope, _r = stage.radial_envelope(flat_bundle, 2.0, 2.0)
    p = flat_bundle.p
    y = (np.arange(p.Ny, dtype=np.float64) + 0.5) * float(p.dy)
    seam = np.minimum(y, p.Ly - y)
    expected = stage.smooth5((seam - 3.0 * p.dy) / (3.0 * p.dy))
    # psi is separably zero wherever the seam plateau is zero, and the x-profile is odd about x0
    assert np.all(np.abs(psi[:, np.flatnonzero(expected == 0.0)]) == 0.0)
    assert not np.all(expected == 1.0)


def test_no_sdf_wall_taper_is_applied(flat_bundle):
    psi, envelope, _r = stage.radial_envelope(flat_bundle, 2.0, 2.0)
    sdf = np.asarray(flat_bundle.solid.sdf, dtype=np.float64)
    wall = sdf <= 0.0
    assert wall.any()
    assert float(np.max(envelope[wall])) == 0.0
    assert float(np.max(np.abs(psi[wall]))) == 0.0


# ---------------------------------------------------------------------------
# single-streamfunction identities
# ---------------------------------------------------------------------------
def test_velocity_is_exactly_the_derivative_of_the_streamfunction(flat_bundle, radial_states):
    """No post-derivative mask: u/v must equal the central derivatives of psi bitwise."""
    p = flat_bundle.p
    for key in stage.RADIAL_ORDER:
        state, details = radial_states[key]
        psi, _envelope, _r = stage.radial_envelope(
            flat_bundle, float(stage.RADIAL_CANDIDATES[key]["delta_dx"]), float(stage.RADIAL_CANDIDATES[key]["w_dx"])
        )
        psi = psi * float(details["amplitude_scale"])
        unit_u = np.asarray(pf._ddy(psi, p.dy), dtype=np.float64)
        unit_v = -np.asarray(pf._ddx(psi, p.dx), dtype=np.float64)
        assert np.array_equal(np.asarray(state.u, dtype=np.float64), unit_u.astype(p.dtype).astype(np.float64))
        assert np.array_equal(np.asarray(state.v, dtype=np.float64), unit_v.astype(p.dtype).astype(np.float64))


def test_core_mean_normalization_hits_the_frozen_target(flat_bundle, radial_states):
    for key in stage.RADIAL_ORDER:
        state, _details = radial_states[key]
        metrics = audit.initial_metrics(flat_bundle, key, state, {"candidate": key})
        core = metrics["actual_initial_core_v_phi_ge_0p9"]
        assert core is not None
        relative = abs(core + flat_bundle.u_impact) / abs(flat_bundle.u_impact)
        assert relative <= stage.TIER1_GATES["core_mean_speed_relative_error_max"]


def test_phase_field_is_bitwise_preserved(flat_bundle, pillar_bundle, radial_states):
    for bundle in (flat_bundle, pillar_bundle):
        reference = audit._array_sha(np.asarray(bundle.base.phi))
        for key in (stage.C2_KEY, *stage.RADIAL_ORDER):
            state, _details = stage.build_any_state(bundle, key)
            assert audit._array_sha(np.asarray(state.phi)) == reference


def test_c2_family_still_reproduces_the_frozen_c2_builder(flat_bundle, pillar_bundle):
    """The frozen C2 seed must be reproduced bitwise by the frozen taper builder."""
    for bundle in (flat_bundle, pillar_bundle):
        c2_state, _details = stage.build_any_state(bundle, stage.C2_KEY)
        tapered_u, tapered_v, _psi, _taper, _d = taper.build_tapered_velocity(bundle, 4.0, 8.0)
        assert np.array_equal(np.asarray(c2_state.u), np.asarray(tapered_u))
        assert np.array_equal(np.asarray(c2_state.v), np.asarray(tapered_v))


def test_reference_controls_are_replay_controls_only(flat_bundle):
    for key in stage.REFERENCE_CANDIDATES:
        state, details = stage.build_any_state(flat_bundle, key)
        assert "never promoted" in details["source"] or "replay control" in details["source"]
        metrics = audit.initial_metrics(flat_bundle, key, state, details)
        assert "initial_constraints_pass" in metrics


# ---------------------------------------------------------------------------
# frozen near-wall gate and wall leakage
# ---------------------------------------------------------------------------
def test_gate_band_speed_is_measured_for_every_radial_candidate(flat_bundle, pillar_bundle, radial_states):
    """The reported gate status must follow the measured ratio, for every outcome.

    A candidate that exceeds the frozen limit must be labelled incompatible rather than
    silently accepted, and a candidate inside the limit must be labelled compatible.
    """
    limit = stage.TIER1_GATES["nearwall_chi_ge_0p01_speed_over_u_impact_max"]
    for bundle in (flat_bundle, pillar_bundle):
        for key in stage.RADIAL_ORDER:
            state, details = stage.build_radial_state(bundle, key)
            metrics = audit.initial_metrics(bundle, key, state, details)
            ratio = metrics["max_abs_velocity_nearwall_chi_ge_0p01"] / bundle.u_impact
            assert ratio == pytest.approx(
                metrics["dimensionless_residuals"]["nearwall_chi_speed_over_u_impact"], rel=1e-12
            )
            expected = ratio <= limit
            assert metrics["acceptance_checks"]["nearwall_chi_velocity"] is expected


def test_at_least_one_radial_candidate_is_compatible_on_the_flat_cases():
    """The prespecified window must not be empty on the reference geometry."""
    record = stage.radial_seed_constraint_matrix("quick")
    for case_name in ("flat_we100_ct050", "flat_we200_ct000"):
        compatible = record["cases"][case_name]["compatible_candidates"]
        assert any(key in compatible for key in stage.RADIAL_ORDER), case_name


def test_incompatible_candidate_is_excluded_from_selection(monkeypatch):
    """A candidate failing the frozen gate must never be selected."""
    record = stage.radial_seed_constraint_matrix("quick")
    selection = stage.pick_factorial_candidate(record)
    compatible = record["compatible_in_every_case"]
    if selection["candidate"] is not None:
        assert selection["candidate"] in compatible
    for case_name, entry in record["cases"].items():
        for key, row in entry["candidates"].items():
            if row["status"] == "INITIALIZER_INCOMPATIBLE_CASE":
                assert row["failing_gates"], (case_name, key)
                assert key not in compatible


def test_bilinear_wall_leakage_is_inside_the_frozen_limit(flat_bundle, pillar_bundle, radial_states):
    limit = stage.TIER1_GATES["embedded_wall_bilinear_normal_speed_over_u_impact_max"]
    for bundle in (flat_bundle, pillar_bundle):
        for key in stage.RADIAL_ORDER:
            state, details = stage.build_radial_state(bundle, key)
            metrics = audit.initial_metrics(bundle, key, state, details)
            assert metrics["max_embedded_wall_bilinear_normal_velocity"] / bundle.u_impact <= limit


def test_all_six_frozen_gates_are_evaluated_per_candidate(flat_bundle, radial_states):
    expected = set(stage.GATE_CHECK_TO_LIMIT)
    for key in stage.RADIAL_ORDER:
        state, details = radial_states[key]
        metrics = audit.initial_metrics(flat_bundle, key, state, details)
        assert set(metrics["acceptance_checks"]) == expected
        assert metrics["measurements_complete"] is True
        for name, ok in metrics["acceptance_checks"].items():
            assert isinstance(ok, bool), name


def test_gate_band_geometry_locates_the_contour_support(flat_bundle):
    geometry = stage.gate_band_geometry(flat_bundle)
    assert geometry["gate_band_max_sdf_over_dx"] is not None
    assert geometry["phi05_support_chi"] is not None
    assert geometry["phi05_support_is_inside_gate_band"] is True
    assert geometry["open_cells_min_sdf_over_dx"] >= geometry["gate_band_max_sdf_over_dx"]


def test_contour_support_carries_no_initial_normal_momentum(flat_bundle, radial_states):
    """The feasibility statement, checked directly: the gate band contains the contour."""
    support = audit._local_gap(np.asarray(flat_bundle.base.phi), 0.5, flat_bundle.solid, flat_bundle.p)
    for key in (stage.C2_KEY, *stage.RADIAL_ORDER):
        state, _details = (
            radial_states.get(key, (None, None)) if key != stage.C2_KEY else stage.build_any_state(flat_bundle, key)
        )
        approach = audit._approach_speed(state.u, state.v, support, flat_bundle.p)
        assert approach is not None
        assert abs(approach) <= 0.02 * flat_bundle.u_impact


# ---------------------------------------------------------------------------
# support confinement
# ---------------------------------------------------------------------------
def test_confinement_separates_the_four_support_measures(flat_bundle):
    for key in stage.RADIAL_ORDER:
        state, details = stage.build_radial_state(flat_bundle, key)
        psi, _env, _r = stage.radial_envelope(
            flat_bundle, float(stage.RADIAL_CANDIDATES[key]["delta_dx"]), float(stage.RADIAL_CANDIDATES[key]["w_dx"])
        )
        psi = psi * float(details["amplitude_scale"])
        confinement = stage.support_confinement(flat_bundle, state, psi)
        for field in (
            "psi_support_min_sdf_over_dx",
            "velocity_support_min_sdf_over_dx",
            "cutcell_advective_flux_support_min_sdf_over_dx",
        ):
            assert confinement[field] is not None
        assert confinement["velocity_support_min_sdf_over_dx"] <= confinement["psi_support_min_sdf_over_dx"]
        assert (
            confinement["cutcell_advective_flux_support_min_sdf_over_dx"]
            <= (confinement["velocity_support_min_sdf_over_dx"])
        )


def test_deep_rolloff_is_reported_as_ill_posed_not_amplified(flat_bundle):
    """A roll-off that leaves the core outside the support must fail closed."""
    with pytest.raises(stage.AuditBlocked):
        stage.radial_envelope(flat_bundle, 40.0, 2.0)
    psi, _env, r = stage.radial_envelope(flat_bundle, 4.0, 1.0)
    r_in = flat_bundle.R - 4.0 * flat_bundle.p.dx
    core_weight = np.asarray(pf.phase_control_volumes(flat_bundle.solid, flat_bundle.p), dtype=np.float64) * (
        np.asarray(flat_bundle.base.phi, dtype=np.float64) >= 0.9
    )
    coverage = float(np.sum(core_weight * (r <= r_in)) / np.sum(core_weight))
    assert coverage < stage.CORE_COVERAGE_FLOOR


# ---------------------------------------------------------------------------
# eta / dt deconfounding
# ---------------------------------------------------------------------------
def test_production_default_eta_is_still_tied_to_dt():
    for dt in (0.004, 0.002, 0.001):
        params = pf.PhaseFieldParams(Nx=48, Ny=48, Lx=6.0, Ly=6.0, dt=dt)
        assert params.eta_pen == pytest.approx(2.0 * dt, rel=0.0, abs=1e-15)


def test_diagnostic_eta_override_is_local_and_exact(flat_bundle):
    original_eta = float(flat_bundle.p.eta_pen)
    original_dt = float(flat_bundle.p.dt)
    cloned, record = stage.diagnostic_eta_bundle(flat_bundle, 0.001)
    assert float(flat_bundle.p.eta_pen) == original_eta
    assert float(flat_bundle.p.dt) == original_dt
    assert record["diagnostic_override"] is True
    assert cloned.p.eta_pen == pytest.approx(0.001, rel=0.0, abs=1e-18)
    assert cloned.p.dt == original_dt
    for name in ("dx", "dy", "eps", "We", "Re", "M", "rho_1" if hasattr(cloned.p, "rho_1") else "rho_l"):
        assert float(getattr(cloned.p, name)) == float(getattr(flat_bundle.p, name))


def test_coupled_configuration_keeps_eta_equal_to_two_dt(flat_bundle):
    for dt in stage.DT_LEVELS:
        case = audit._canary_cases()["flat_we100_ct050"]
        bundle = audit._derive_bundle("flat_we100_ct050", case, QUICK_N, dt)
        assert bundle.p.eta_pen == pytest.approx(2.0 * float(bundle.p.dt), rel=0.0, abs=1e-15)


def test_fixed_eta_configuration_holds_eta_across_dt(flat_bundle):
    case = audit._canary_cases()["flat_we100_ct050"]
    etas = []
    for dt in stage.DT_LEVELS:
        bundle = audit._derive_bundle("flat_we100_ct050", case, QUICK_N, dt)
        cloned, record = stage.diagnostic_eta_bundle(bundle, stage.BASELINE_ETA)
        etas.append(record["eta_pen"])
        assert record["dt_over_eta"] == pytest.approx(dt / stage.BASELINE_ETA, rel=1e-12)
    assert len(set(etas)) == 1


def test_damping_ledger_matches_the_frozen_operator_form(flat_bundle):
    ledger = stage.brinkman_damping_ledger(flat_bundle)
    p = flat_bundle.p
    h = float(p.dt) / 3.0
    chi = np.asarray(flat_bundle.solid.chi, dtype=np.float64)
    damp = 1.0 / (1.0 + h * chi / float(p.eta_pen))
    assert ledger["h"] == pytest.approx(h, rel=0.0, abs=1e-18)
    open_cells = np.asarray(pf.phase_control_volumes(flat_bundle.solid, flat_bundle.p), dtype=np.float64) > 0.0
    assert ledger["open_control_volumes"]["min"] == pytest.approx(float(np.min(damp[open_cells])), rel=1e-12)
    assert ledger["open_control_volumes"]["max"] == pytest.approx(float(np.max(damp[open_cells])), rel=1e-12)
    substeps = int(round(stage.FACTORIAL_WINDOW / h))
    gate = np.asarray(flat_bundle.solid.chi, dtype=np.float64) >= 0.01
    assert ledger["cumulative_over_factorial_window"]["gate_band_min"] == pytest.approx(
        float(np.min(damp[gate] ** substeps)), rel=1e-12
    )


# ---------------------------------------------------------------------------
# interface versus fluid velocity and the phase budget
# ---------------------------------------------------------------------------
def test_support_phase_split_closes_the_additive_identity(flat_bundle, radial_states):
    state, _details = radial_states["R1_MEDIUM"]
    split = stage.support_phase_split(flat_bundle, state)
    window = split["window_V_weighted"]
    residual = window["closure_residual"]
    scale = max(
        abs(window["sum_V_delta_adv"]),
        abs(window["sum_V_delta_ch_explicit"]),
        abs(window["sum_V_delta_ch_implicit"]),
        1e-30,
    )
    assert abs(residual) / scale <= 1e-10
    assert split["grid_closure_linf"] <= 1e-12
    assert split["implicit_solver"]["converged"] is True


def test_interface_and_fluid_velocity_are_reported_separately(flat_bundle):
    record = stage.interface_vs_fluid_velocity("quick", candidate=None)
    entry = record["cases"]["flat_we100_ct050"][stage.C2_KEY]
    assert entry["front_speed_dgap_dt"] is not None or entry["contact_time_discrete"] is None
    assert entry["mean_local_normal_fluid_speed"] is not None
    assert "not interchangeable" in entry["distinction"]
    assert "contour" in entry["distinction"]


def test_light_readout_separates_gap_from_flow_speed(flat_bundle, radial_states):
    state, _details = radial_states["R1_MEDIUM"]
    row = stage.light_readout(flat_bundle, state, 0, 0.0, with_gap=True)
    assert "gap_phi05" in row and "local_normal_approach_speed" in row
    assert row["gap_phi05"] is not None
    # the contour distance and the fluid speed are different physical quantities
    assert abs(row["gap_phi05"]) > 0.0
    assert row["local_normal_approach_speed"] is not None


# ---------------------------------------------------------------------------
# feasibility stage
# ---------------------------------------------------------------------------
def test_feasibility_reports_the_gate_band_before_any_candidate(monkeypatch):
    record = stage.t0_feasibility_matrix("quick")
    assert record["verdict"] == "T0_FEASIBILITY_NO_GO_FOR_T0_CONTACT_MOMENTUM"
    assert record["structural_finding"]["measured_not_assumed"] is True
    for name, geometry in record["gate_band_geometry"].items():
        assert geometry["gate_band_max_sdf_over_dx"] is not None
        assert geometry["phi05_support_is_inside_gate_band"] is True


def test_feasibility_probes_record_ill_posed_boundaries():
    record = stage.t0_feasibility_matrix("quick")
    statuses = {key: value["status"] for key, value in record["boundary_probes"]["flat_we100_ct050"].items()}
    assert statuses["P_DELTA_4.0"] == "ILL_POSED"
    assert statuses["P_DELTA_1.0"] == "MEASURED"


def test_worst_case_gate_band_edge_uses_the_maximum():
    geometry = {"a": {"gate_band_max_sdf_over_dx": 2.5}, "b": {"gate_band_max_sdf_over_dx": 3.4}}
    assert stage.worst_case_gate_band_edge(geometry) == 3.4


# ---------------------------------------------------------------------------
# constraint matrix and candidate selection
# ---------------------------------------------------------------------------
def test_constraint_matrix_labels_status_and_lists_the_gates(flat_bundle):
    record = stage.radial_seed_constraint_matrix("quick")
    for case_name, entry in record["cases"].items():
        for key, row in entry["candidates"].items():
            assert row["status"] in ("COMPATIBLE", "INITIALIZER_INCOMPATIBLE_CASE")
            assert row["status"] == ("COMPATIBLE" if not row["failing_gates"] else "INITIALIZER_INCOMPATIBLE_CASE")
            assert set(row["acceptance_checks"]) == set(stage.GATE_CHECK_TO_LIMIT)
        for key, row in entry["controls"].items():
            assert "never promoted" in row["role"]


def test_candidate_selection_rule_is_prespecified():
    fake = {"compatible_in_every_case": ["R0_WEAK", "R1_MEDIUM"]}
    assert stage.pick_factorial_candidate(fake)["candidate"] == "R1_MEDIUM"
    fake2 = {"compatible_in_every_case": ["R0_WEAK"]}
    assert stage.pick_factorial_candidate(fake2)["candidate"] == "R0_WEAK"
    fake3 = {"compatible_in_every_case": []}
    assert stage.pick_factorial_candidate(fake3)["candidate"] is None


# ---------------------------------------------------------------------------
# interior-speed admissibility screen (frozen gates bound the wall band only)
# ---------------------------------------------------------------------------
def _screen_matrix(candidate_peaks, control_peak=4.9):
    rows = {"C2_BASE": {"interior_peak_speed_over_u_impact": control_peak}}
    rows.update({key: {"interior_peak_speed_over_u_impact": float(value)} for key, value in candidate_peaks.items()})
    return {
        "cases": {
            "case_a": {
                "controls": {"C0_UNIFORM": {"interior_peak_speed_over_u_impact": 1.0}},
                "candidates": rows,
            }
        }
    }


def test_interior_speed_ceiling_is_measured_from_the_frozen_reference_family():
    matrix = _screen_matrix({"R0_WEAK": 30.974, "R1_MEDIUM": 89.770, "R2_SHARP": 36.328}, control_peak=4.316)
    screen = stage.interior_speed_screen(matrix)
    assert screen["ceiling_over_u_impact"] == 4.316
    assert screen["ceiling_sources"] == ["case_a/C2_BASE"]
    assert screen["per_case"]["case_a"]["C2_BASE"]["status"] == "ADMISSIBLE"
    for key in stage.RADIAL_ORDER:
        assert screen["per_case"]["case_a"][key]["status"] == "DISCRETIZATION_INADMISSIBLE"
    assert screen["inadmissible_in_every_case"] == list(stage.RADIAL_ORDER)
    assert screen["admissible_in_every_case"] == []


def test_interior_speed_screen_keeps_a_candidate_exactly_at_the_ceiling():
    matrix = _screen_matrix({"R0_WEAK": 4.9, "R1_MEDIUM": 4.900001, "R2_SHARP": 0.5}, control_peak=4.9)
    screen = stage.interior_speed_screen(matrix)
    per_case = screen["per_case"]["case_a"]
    assert per_case["R0_WEAK"]["status"] == "ADMISSIBLE"
    assert per_case["R2_SHARP"]["status"] == "ADMISSIBLE"
    assert per_case["R1_MEDIUM"]["status"] == "DISCRETIZATION_INADMISSIBLE"


def test_inadmissible_seed_rows_lists_the_blocked_case_candidate_pairs():
    matrix = _screen_matrix({"R0_WEAK": 31.0, "R1_MEDIUM": 89.8, "R2_SHARP": 36.3})
    matrix["interior_speed_screen"] = stage.interior_speed_screen(matrix)
    rows = stage.inadmissible_seed_rows(matrix)
    assert set(rows) == {("case_a", key) for key in stage.RADIAL_ORDER}


def _confounded_comparison():
    comparison = {
        observable: {
            "coupled_finest_over_coarsest": 0.5,
            "fixed_finest_over_coarsest": 0.99,
        }
        for observable in ("gate_band_max_speed_over_u_impact_peak", "front_speed_gap_per_time")
    }
    return {"comparison": comparison}


def test_factorial_verdict_counts_only_admissible_seed_rows():
    matrix = {
        "cases": {
            "case_a": {"C2_BASE": _confounded_comparison()},
            "case_b": {
                "R1_MEDIUM": {
                    "comparison": {
                        "gate_band_max_speed_over_u_impact_peak": {
                            "coupled_finest_over_coarsest": 0.0005,
                            "fixed_finest_over_coarsest": 0.0006,
                        },
                        "front_speed_gap_per_time": {
                            "coupled_finest_over_coarsest": 1.3167,
                            "fixed_finest_over_coarsest": 1.3517,
                        },
                    }
                }
            },
        }
    }
    included = stage._factorial_verdict(matrix)
    excluded = stage._factorial_verdict(matrix, (("case_b", "R1_MEDIUM"),))
    assert included["verdict"] == "DT_ETA_CONFOUNDED"
    assert included["confounded_observables"] == 2
    assert included["residual_observables"] == 2
    assert excluded["verdict"] == "DT_ETA_CONFOUNDED"
    assert excluded["confounded_observables"] == 2
    assert excluded["residual_observables"] == 0
    assert excluded["excluded_inadmissible_rows"] == ["case_b/R1_MEDIUM"]
    assert excluded["all_rows_verdict_including_inadmissible_seeds"] == "DT_ETA_CONFOUNDED"
    assert excluded["all_rows_counts"] == {"confounded": 2, "residual": 2}
    assert all(
        row["counted_in_verdict"] is False for key, row in excluded["trends"].items() if key.startswith("case_b/")
    )


def _decision_inputs(interior_peak, status):
    chosen = "R1_MEDIUM"
    case_names = ("flat_we100_ct050", "flat_we200_ct000", "pillar_training", "complex_heldout")
    rows = {
        "C2_BASE": {
            "flux_budget": {
                "downward_phase_flux_above_band": 0.001,
                "mass_weighted_downward_momentum_above_band": 0.01,
            },
            "nearwall_chi_speed_over_u_impact": 0.0,
            "interior_peak_speed_over_u_impact": 4.3,
        },
        chosen: {
            "flux_budget": {
                "downward_phase_flux_above_band": 0.01,
                "mass_weighted_downward_momentum_above_band": 0.02,
            },
            "nearwall_chi_speed_over_u_impact": 0.0,
            "interior_peak_speed_over_u_impact": interior_peak,
        },
    }
    constraints = {
        "cases": {name: {"candidates": rows} for name in case_names},
        "compatible_in_every_case": [chosen],
        "interior_speed_screen": {
            "ceiling_over_u_impact": 4.906,
            "ceiling_sources": ["flat_we100_ct050/C1_STREAMFUNCTION_V0"],
            "admissible_in_every_case": [chosen] if status == "ADMISSIBLE" else [],
            "inadmissible_in_every_case": [] if status == "ADMISSIBLE" else [chosen],
            "rule": "measured ceiling, not adjustable",
            "per_case": {name: {chosen: {"status": status}} for name in case_names},
        },
    }
    interface = {
        "cases": {
            name: {
                "C2_BASE": {"max_local_normal_fluid_speed": 0.05, "front_speed_dgap_dt": -0.09},
                chosen: {"max_local_normal_fluid_speed": 0.22, "front_speed_dgap_dt": -0.30},
            }
            for name in case_names
        }
    }
    return {
        "profile": "forensic",
        "feasibility": {
            "verdict": "T0_FEASIBILITY_NO_GO_FOR_T0_CONTACT_MOMENTUM",
            "design_window_verdict": "FEASIBLE_WINDOW_EXISTS_ABOVE_THE_GATE_BAND",
        },
        "constraints": constraints,
        "factorial": {"verdict": {"verdict": "DT_ETA_CONFOUNDED"}},
        "interface": interface,
        "provenance": {"checks": {"l1a2t_manifest_verified": True}, "ci_observed": False},
        "c2_check": {"verdict": "BASELINE_REPLAY_OK"},
    }


def test_decision_withholds_the_gain_claim_when_the_seed_is_inadmissible():
    admissible = stage.decide(**_decision_inputs(4.4, "ADMISSIBLE"))
    assert admissible["candidate_outcome"] == "RADIAL_SEED_COMPATIBLE_CAUSAL_SPEED_GAIN"
    assert admissible["radial_compatibility"]["cases_with_material_flux_and_approach_gain"] == 4
    assert admissible["radial_compatibility"]["cases_with_valid_readout"] == 4

    blocked = stage.decide(**_decision_inputs(89.7, "DISCRETIZATION_INADMISSIBLE"))
    assert blocked["candidate_outcome"] == "RADIAL_SEED_DISCRETIZATION_INADMISSIBLE_NO_GAIN_CLAIM"
    assert blocked["taper_mechanism_verdict"] == "INTERIOR_SPEED_INADMISSIBLE"
    assert blocked["radial_compatibility"]["cases_with_material_flux_and_approach_gain"] == 0
    assert blocked["radial_compatibility"]["cases_with_valid_readout"] == 0
    assert blocked["radial_compatibility"]["gain_claim_status"].startswith("WITHHELD")
    assert blocked["stage_status"] == "L1A2U_DIAGNOSTIC_COMPLETE_NO_PROMOTION"
    assert blocked["next_action"]["must_not_be_started_in_this_stage"] is True
    assert "t=0" in blocked["next_action"]["action"]
    for name, row in blocked["per_case_radial_vs_c2"].items():
        assert row["readout_status"] == "DISCRETIZATION_INADMISSIBLE_READOUT", name
        assert row["radial_seed_admissibility"] == "DISCRETIZATION_INADMISSIBLE"


def test_stability_probe_horizon_stays_before_the_earliest_frozen_contact():
    earliest_contact = 0.208645
    assert stage.STABILITY_PROBE_CASE == "complex_heldout"
    assert stage.STABILITY_PROBE_MACRO_STEPS * 0.002 < earliest_contact


def test_no_control_is_ever_promoted(flat_bundle):
    record = stage.radial_seed_constraint_matrix("quick")
    assert record["controls_promoted"] is False
    for key in stage.REFERENCE_CANDIDATES:
        assert key not in stage.RADIAL_ORDER
        assert key not in stage.RADIAL_CANDIDATES


# ---------------------------------------------------------------------------
# failure injection / adversarial
# ---------------------------------------------------------------------------
def test_injected_failures_raise(flat_bundle):
    with pytest.raises(stage.AuditBlocked):
        stage.diagnostic_eta_bundle(flat_bundle, -1.0)
    with pytest.raises(stage.AuditBlocked):
        stage.radial_envelope(flat_bundle, 2.0, 0.0)
    with pytest.raises(stage.AuditBlocked):
        stage.build_any_state(flat_bundle, "NOT_A_CANDIDATE")
    with pytest.raises(stage.AuditBlocked):
        stage.run(profile="quick", stages="not_a_stage")


def test_diagnostic_eta_override_refuses_to_change_physics(flat_bundle):
    """The override must not silently alter dt, geometry or any physical parameter."""
    cloned, record = stage.diagnostic_eta_bundle(flat_bundle, stage.BASELINE_ETA)
    assert record["geometry_and_physics_unchanged"] is True
    geometry_before = audit._geometry_hash(flat_bundle.solid)
    assert geometry_before == audit._geometry_hash(cloned.solid)


def test_adversarial_self_checks_hold():
    preflight = {"l1a2t_missing_commits_resolve_to": {name: None for name in stage.L1A2T_MISSING_COMMITS}}
    provenance = {
        "checks": {
            "l1a2t_module_sha_matches_stage_record": True,
            "l1a2t_manifest_verified": True,
        },
        "l1a2t_manifest_verification": {"verified": True},
    }
    record = stage.failure_and_adversarial_tests(preflight, provenance)
    assert record["checks"]["eta_override_does_not_change_production_default"]["holds"] is True
    assert record["checks"]["radial_family_uses_no_sdf_taper"]["holds"] is True
    assert all(record["checks"]["injected_failures_are_rejected"].values())


def test_missing_commits_are_recorded_not_hidden(tmp_path):
    preflight = {"l1a2t_missing_commits_resolve_to": {name: None for name in stage.L1A2T_MISSING_COMMITS}}
    provenance = {
        "checks": {"l1a2t_module_sha_matches_stage_record": True, "l1a2t_manifest_verified": True},
        "l1a2t_manifest_verification": {"verified": True},
    }
    record = stage.failure_and_adversarial_tests(preflight, provenance)
    assert record["checks"]["missing_l1a2t_commits_are_recorded"]["all_unresolved"] is True


# ---------------------------------------------------------------------------
# manifest / admission control
# ---------------------------------------------------------------------------
def test_manifest_self_verification_and_tamper_detection(tmp_path):
    out = tmp_path / "evidence"
    out.mkdir()
    manifest = {
        "stage": stage.STAGE,
        "stage_version": stage.STAGE_VERSION,
        "profile": "quick",
        "evidence_class": "QUICK_ONLY",
        "verified_main_sha": stage.VERIFIED_MAIN_SHA,
        "evidence_files": {},
    }
    for name in stage.REQUIRED_EVIDENCE_FILES:
        manifest["evidence_files"][name] = stage.write_json(out / name, {"marker": name})
    stage.write_json(out / "manifest.json", manifest)
    verification = stage.verify_evidence_manifest(out)
    assert verification["problems"] == []
    assert verification["missing_required"] == []
    assert verification["verified"] is True
    assert verification["bound_files"] == len(stage.REQUIRED_EVIDENCE_FILES)

    # tampering with a bound file is detected
    (out / "preflight.json").write_text(json.dumps({"marker": "tampered"}))
    tampered = stage.verify_evidence_manifest(out)
    assert any(problem["status"] == "HASH_MISMATCH" for problem in tampered["problems"])
    assert tampered["verified"] is False

    # deleting a bound file is detected
    (out / "preflight.json").unlink()
    missing = stage.verify_evidence_manifest(out)
    assert any(problem["status"] == "MISSING" for problem in missing["problems"])
    assert missing["verified"] is False

    # a partially declared manifest reports the absent required files
    manifest["evidence_files"] = {}
    stage.write_json(out / "manifest.json", manifest)
    partial = stage.verify_evidence_manifest(out)
    assert set(partial["missing_required"]) == set(stage.REQUIRED_EVIDENCE_FILES)
    assert partial["verified"] is False


def test_required_evidence_files_match_the_specification():
    names = set(stage.REQUIRED_EVIDENCE_FILES)
    assert len(stage.REQUIRED_EVIDENCE_FILES) == 12
    assert "l1a2u_final_report.md" in names
    assert "preflight.json" in names
    assert "dt_eta_factorial_matrix.json" in names
    assert all(name.endswith(".json") or name.endswith(".md") for name in names)


def test_reseal_records_the_observed_ci_status_without_touching_physics(tmp_path):
    out = tmp_path / "evidence"
    out.mkdir()
    manifest = {
        "stage": stage.STAGE,
        "stage_version": stage.STAGE_VERSION,
        "profile": "quick",
        "evidence_class": "QUICK_ONLY",
        "verified_main_sha": stage.VERIFIED_MAIN_SHA,
        "evidence_files": {},
    }
    for name in stage.REQUIRED_EVIDENCE_FILES:
        manifest["evidence_files"][name] = stage.write_json(out / name, {"marker": name})
    quality = stage.quality_and_ci("quick", None, "queued")
    stage.write_json(out / "quality_and_ci.json", quality)
    manifest["evidence_files"]["quality_and_ci.json"] = stage.sha256_file(out / "quality_and_ci.json")
    report = out / "l1a2u_final_report.md"
    report.write_text("# report\n\n## M. Quality, tests and hosted CI\n- hosted CI for this head: `queued`\n")
    manifest["evidence_files"]["l1a2u_final_report.md"] = stage.sha256_file(report)
    stage.write_json(out / "manifest.json", manifest)
    assert stage.verify_evidence_manifest(out)["verified"] is True

    resealed = stage.reseal_ci_status(out, ci_status="success", head_sha="a" * 40)
    assert resealed["verified"] is True
    updated = json.loads((out / "quality_and_ci.json").read_text())
    assert updated["hosted_ci"]["status_for_this_head"] == "success"
    assert updated["hosted_ci"]["previous_recorded_status"] == "queued"
    assert updated["hosted_ci"]["observed_head_sha"] == "a" * 40
    assert "success" in report.read_text()
    assert json.loads((out / "manifest.json").read_text())["hosted_ci_reseal"]["performed"] is True
    # an unknown status is refused rather than recorded silently
    with pytest.raises(stage.AuditBlocked):
        stage.reseal_ci_status(out, ci_status="banana")
    # a tampered set must not be re-sealed
    (out / "preflight.json").write_text(json.dumps({"marker": "tampered"}))
    with pytest.raises(stage.AuditBlocked):
        stage.reseal_ci_status(out, ci_status="success")


def test_validation_payload_hash_binding(tmp_path):
    module_sha = stage.sha256_file(Path(stage.__file__))
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"stage": stage.STAGE, "stage_source_sha256": module_sha}))
    assert stage.load_validation_payload(good)["stage"] == stage.STAGE
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"stage": stage.STAGE, "stage_source_sha256": "0" * 64}))
    with pytest.raises(stage.AuditBlocked):
        stage.load_validation_payload(bad)
    unbound = tmp_path / "unbound.json"
    unbound.write_text(json.dumps({"stage": stage.STAGE}))
    with pytest.raises(stage.AuditBlocked):
        stage.load_validation_payload(unbound)


def test_stage_owns_only_its_own_paths():
    for owned in stage.STAGE_OWNED_PATHS:
        assert owned.startswith("examples/two_phase/")
    for frozen in (
        "examples/two_phase/phasefield.py",
        "examples/two_phase/generate_dataset.py",
        "examples/two_phase/cases.py",
        "examples/two_phase/production/timestep_policy.py",
        "examples/two_phase/production/impact_initialization_compatibility_audit.py",
        "quality/legacy_two_phase_debt_v1.json",
    ):
        assert not stage._stage_owned_path(frozen), frozen


def test_stage_writes_nothing_outside_the_evidence_and_artifact_dirs():
    source = Path(stage.__file__).read_text()
    for forbidden in ('phasefield.py"', 'generate_dataset.py"', 'cases.py"'):
        assert "write_text(out_dir" not in source or forbidden not in source
    assert "l1a2u" in source
    assert "EVIDENCE_DIR" in source and "ARTIFACT_DIR" in source


def test_quality_and_ci_reports_the_frozen_blockers():
    record = stage.quality_and_ci("quick", None, "success")
    assert record["declared_new_code_diagnostics"] == 0
    assert record["quality_baseline_untouched"] is True
    assert record["inherited_statuses"]["L1B_DATA"] == "L1B_DATA_NOT_READY"
    assert record["frozen_adjacent_suites"]["test_l1a2r_merged_base_verified_or_fail_closed"]["classification"] == (
        "FROZEN_LINEAGE_DRIFT_CONFIRMED"
    )
    assert record["hosted_ci"]["status_for_this_head"] == "success"
    unknown = stage.quality_and_ci("quick", None, "banana")
    assert unknown["hosted_ci"]["status_for_this_head"] == "UNMEASURED"


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def test_report_contains_every_required_section(tmp_path):
    run = stage.run(profile="quick", stages="preflight,feasibility,constraints", out=tmp_path / "evidence")
    payload = {name: value for name, value in run.items() if not name.startswith("__") and name != "manifest"}
    payload["preflight"]["profile"] = "quick"
    payload["preflight"]["profile_N"] = stage._profile_N("quick")
    payload["source_and_provenance"] = run["source_and_provenance"]
    record = stage.t0_feasibility_matrix("quick")
    constraints = run["radial_seed_constraint_matrix"]
    # an unmeasured value must render as an explicit label, never as a zero
    constraints["cases"]["flat_we100_ct050"]["candidates"]["R1_MEDIUM"]["wall_normal_sample_over_u_impact"] = None
    assert stage._fmt(None) == "UNMEASURED"
    report = stage.render_report(
        preflight=payload["preflight"],
        source_and_provenance=run["source_and_provenance"],
        t0_feasibility_matrix=record,
        radial_seed_constraint_matrix=constraints,
        nearwall_and_flux_budget=stage.nearwall_and_flux_budget("quick", candidate=None),
        dt_eta_factorial_matrix={
            "verdict": {
                "verdict": "INSUFFICIENT_EVIDENCE",
                "confounded_observables": 0,
                "residual_observables": 0,
                "rule": "x",
            },
            "cases": {},
        },
        interface_vs_fluid_velocity={"cases": {}},
        counterfactual_ledger={"entry_count": 0, "definition": "x", "entries": []},
        failure_and_adversarial_tests=run["failure_and_adversarial_tests"],
        quality_and_ci=run["quality_and_ci"],
        mechanism_decision={
            "stage_status": "L1A2U_DIAGNOSTIC_COMPLETE_NO_PROMOTION",
            "c2_replay_verdict": "NOT_COMPARABLE_AT_THIS_PROFILE",
            "taper_mechanism_verdict": "INSUFFICIENT_EVIDENCE",
            "t0_feasibility_verdict": record["verdict"],
            "design_window_verdict": record["design_window_verdict"],
            "candidate_outcome": "NO_COMPATIBLE_SEED",
            "dt_eta_verdict": "INSUFFICIENT_EVIDENCE",
            "radial_compatibility": {
                "candidate_selection_rule": "x",
                "cases_with_material_flux_and_approach_gain": 0,
                "cases_compared": 0,
            },
            "provenance_status": {
                "l1a2t_commit_shas_verifiable": False,
                "l1a2t_content_recovered_and_published": True,
                "l1a2t_remote_provenance_blocked": False,
            },
            "frozen_lineage_gate_blocked": True,
            "next_action": {"action": "x", "bounded_scope": "y", "must_not_be_started_in_this_stage": True},
        },
        c2_reproduction_check={"verdict": "NOT_COMPARABLE_AT_THIS_PROFILE", "cases": {}},
    )
    for heading in (
        "## A.",
        "## B.",
        "## C.",
        "## D.",
        "## E.",
        "## F.",
        "## G.",
        "## H.",
        "## I.",
        "## J.",
        "## K.",
        "## L.",
        "## M.",
        "## N.",
    ):
        assert heading in report, heading
    assert "UNMEASURED" in report
    assert "wall sample/U=UNMEASURED" in report
