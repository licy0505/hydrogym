#!/usr/bin/env python3
"""Short, diagnostic-only C2 comparison of pillar_training and two flat canaries.

This reuses the frozen L1A-2s geometry, initializer, event times, and saved T=8
outcomes. It advances only to each saved precontact time and records RHS,
Brinkman, and periodic pressure-projection contributions. It does not change any
production source, default, solver input, threshold, or data lineage.

Run from examples/two_phase with JAX_ENABLE_X64=1 and PYTHONPATH=.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import jax
import numpy as np
import phasefield as pf
from production import impact_impulse_projection_audit as impulse
from production import impact_initialization_compatibility_audit as audit

HERE = Path(__file__).resolve().parent
EVIDENCE = Path(__file__).resolve().parents[1]
CASE_NAMES = ("flat_we100_ct050", "flat_we200_ct000", "pillar_training")
CANDIDATE = "SDF_TAPERED_STREAMFUNCTION_V1"
N = 192
REQUESTED_DT = 0.004
ROLLING_PUBLIC_STEPS = 10
CONTACT_GAP_CELLS_FROZEN = 1.5
APPROACH_FLOOR_FRACTION_FROZEN = 0.2


class EvidenceMismatch(RuntimeError):
    """Stop if the saved run cannot be reproduced under the frozen identities."""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def max_abs_delta(left: Any, right: Any) -> float:
    return float(np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))))


def scalar_row(values: Any) -> dict[str, float | bool]:
    raw = np.asarray(values, dtype=np.float64).tolist()
    row: dict[str, float | bool] = {}
    for name, value in zip(impulse.SCALAR_NAMES, raw, strict=True):
        row[name] = bool(value > 0.5) if name == "cg_converged" else float(value)
    return row


def make_stage_capture_kernel(solid: Any, p: Any, h: float):
    """Return actual-dtype stage fields; state result is checked against the fast kernel."""

    @jax.jit
    def capture(state: Any):
        phi, u, v = state.phi, state.u, state.v
        phi_rhs, u_rhs, v_rhs, _mu, mu_expl = pf.rhs(state, solid, p)
        phi_new, _solve_info = pf._phase_update(phi, u, v, solid, p, h, phi_rhs, mu_expl)
        damp = 1.0 / (1.0 + h * solid.chi / p.eta_pen)
        u_rhs_stage = u + h * u_rhs
        v_rhs_stage = v + h * v_rhs
        u_brinkman = u_rhs_stage * damp
        v_brinkman = v_rhs_stage * damp
        div_star = pf._ddx(u_brinkman, p.dx) + pf._ddy(v_brinkman, p.dy)
        pressure = pf.poisson_solve(div_star / h, p.m2_proj)
        u_projection = u_brinkman - h * pf._ddx(pressure, p.dx)
        v_projection = v_brinkman - h * pf._ddy(pressure, p.dy)
        next_state = pf.State(
            phi=phi_new.astype(pf.phase_state_dtype(p)),
            u=u_projection.astype(p.dtype),
            v=v_projection.astype(p.dtype),
            t=state.t + h,
        )
        stages = (u_rhs_stage, v_rhs_stage, u_brinkman, v_brinkman, u_projection, v_projection)
        return next_state, stages

    return capture


def local_stage_row(
    state_before: Any,
    stage_fields: tuple[Any, ...],
    solid: Any,
    p: Any,
    t_before: float,
    public_step: int,
    substep_index: int,
) -> dict[str, Any]:
    support = audit._local_gap(np.asarray(state_before.phi), 0.5, solid, p)
    if support.get("local_wall_normal_into_solid") is None:
        raise EvidenceMismatch("no measured wall normal at the reconstructed local phi=0.5 contour")
    u_rhs, v_rhs, u_brink, v_brink, u_proj, v_proj = stage_fields
    stages = {
        "before": (state_before.u, state_before.v),
        "after_explicit_rhs": (u_rhs, v_rhs),
        "after_brinkman": (u_brink, v_brink),
        "after_pressure_projection": (u_proj, v_proj),
    }
    values = {name: audit._approach_speed(np.asarray(u), np.asarray(v), support, p) for name, (u, v) in stages.items()}
    if any(value is None for value in values.values()):
        raise EvidenceMismatch("local approach component was not measured")
    return {
        "public_step": int(public_step),
        "substep_index_1_based": int(substep_index + 1),
        "t_before": float(t_before),
        "t_after": float(t_before + float(p.dt) / 3.0),
        "support": support,
        "normal_approach_speed_by_velocity_stage": values,
        "normal_approach_delta_by_operator": {
            "explicit_rhs": float(values["after_explicit_rhs"] - values["before"]),
            "brinkman": float(values["after_brinkman"] - values["after_explicit_rhs"]),
            "periodic_pressure_projection": float(values["after_pressure_projection"] - values["after_brinkman"]),
        },
    }


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_per_internal_substep": float(np.mean(array)),
        "median_per_internal_substep": float(np.median(array)),
        "min_per_internal_substep": float(np.min(array)),
        "max_per_internal_substep": float(np.max(array)),
        "max_abs_per_internal_substep": float(np.max(np.abs(array))),
    }


def stage_window_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = (
        "rhs_dv_liquid_mean",
        "brinkman_dv_liquid_mean_exact",
        "projection_dv_liquid_mean_exact",
        "rhs_dv_grid_mean",
        "brinkman_dv_grid_mean_exact",
        "projection_dv_grid_mean_exact",
        "brinkman_dv_grid_l2",
        "proj_dv_grid_l2",
        "proj_dv_grid_linf_v",
        "mean_of_proj_correction",
        "div_before_linf",
        "div_after_rhs_linf",
        "div_after_brinkman_linf",
        "div_after_projection_linf",
    )
    return {
        "public_steps": [rows[0]["public_step"], rows[-1]["public_step"]],
        "public_step_count": len({row["public_step"] for row in rows}),
        "internal_substep_count": len(rows),
        "definitions": {
            "liquid_mean": "stage delta in V_i*phi_i-weighted mean v; weights frozen at substep input phi",
            "grid_mean": "full periodic grid mean v change at the named operator stage",
            "projection_linf_v": "max absolute vertical velocity correction from periodic pressure projection",
            "projection_mean": "mean of the periodic pressure-gradient correction; not a Brinkman mean change",
            "divergence": "periodic central-D Linf at named substep stage; not the open-face FV reconstruction",
        },
        "statistics": {name: summarize([float(row[name]) for row in rows]) for name in names},
        "all_projection_solves_converged": all(bool(row["cg_converged"]) for row in rows),
    }


def main() -> None:
    if int(pf.SOLVER_CONTRACT_VERSION) != 12:
        raise EvidenceMismatch(f"expected frozen solver contract 12, got {pf.SOLVER_CONTRACT_VERSION}")
    if audit.timestep_policy.DEFAULT_POLICY_NAME != "impact_phase_cap_dx2_v1":
        raise EvidenceMismatch("frozen timestep policy changed")
    if not bool(jax.config.jax_enable_x64):
        raise EvidenceMismatch("run requires JAX_ENABLE_X64=1, matching the saved evidence environment")
    if audit.CONTACT_GAP_CELLS != CONTACT_GAP_CELLS_FROZEN:
        raise EvidenceMismatch("contact gap threshold differs from frozen 1.5 dx")
    if audit.APPROACH_FLOOR_FRACTION != APPROACH_FLOOR_FRACTION_FROZEN:
        raise EvidenceMismatch("meaningful-approach floor differs from frozen 0.2*u_impact")

    frozen = audit._frozen_source_gate()
    if not all(frozen["checks"].values()):
        raise EvidenceMismatch(f"L1A-2s source freeze failed: {frozen['checks']}")

    saved_events = load_json(EVIDENCE / "impact_event_matrix.json")
    saved_initial = load_json(EVIDENCE / "initializer_constraint_matrix.json")
    saved_startup = load_json(EVIDENCE / "projection_brinkman_startup_ledger.json")
    saved_decision = load_json(EVIDENCE / "initializer_decision.json")
    cases = audit._canary_cases()
    if tuple(cases) != audit.CASE_NAMES:
        raise EvidenceMismatch("the frozen four-canary registry changed")

    output: dict[str, Any] = {
        "stage": "L1A-2s focused pillar-versus-flat precontact operator diagnosis",
        "diagnostic_only": True,
        "production_lineage_eligible": False,
        "solver_contract_version": 12,
        "timestep_policy": "impact_phase_cap_dx2_v1",
        "candidate_initializer": CANDIDATE,
        "candidate_initializer_role": "isolated diagnostic-only C2; not the production default",
        "N": N,
        "requested_dt": REQUESTED_DT,
        "effective_dt": None,
        "internal_substep_h": None,
        "contact_gap_cells": CONTACT_GAP_CELLS_FROZEN,
        "meaningful_approach_floor": {
            "formula": "0.2*u_impact",
            "fraction": APPROACH_FLOOR_FRACTION_FROZEN,
            "u_impact": 0.5,
            "value": 0.1,
        },
        "production_source_freeze_checks": frozen["checks"],
        "historical_readiness_and_blockers_unchanged": {
            "primary_blocker": saved_decision["primary_blocker"],
            "initializer_verdict": saved_decision["initializer_verdict"],
            "stage_decision": saved_decision["stage_decision"]["decision"],
            "no_l1b_or_contract_promotion": True,
        },
        "scope": {
            "cases": list(CASE_NAMES),
            "geometry_and_drop_placement": (
                "reconstructed from exact frozen canary registry; verified against saved "
                "geometry and initial-state hashes"
            ),
            "contact_metric": "bilinear actual-SDF gap on the periodic linear phi=0.5 contour; gap <= 1.5 dx",
            "postcontact": "reused from saved full-horizon evidence; no postcontact window was rerun",
            "full_horizon": "not rerun; only short segments ending at each saved precontact time were advanced",
            "complex_heldout": "not rerun or replaced; its saved event and readiness status remain unchanged",
        },
        "saved_full_horizon_outcomes_reused": {},
        "cases_detail": {},
    }

    bundles = {}
    for case_name in CASE_NAMES:
        bundle = audit._derive_bundle(case_name, cases[case_name], N, REQUESTED_DT)
        bundles[case_name] = bundle
        output["effective_dt"] = float(bundle.p.dt)
        output["internal_substep_h"] = float(bundle.p.dt / 3.0)

    for case_name in CASE_NAMES:
        bundle = bundles[case_name]
        event_case = saved_events[case_name]
        event = event_case["full_horizon"]
        if event.get("contact_gap_criterion") != "local phi=0.5 gap <= 1.5 dx":
            raise EvidenceMismatch(f"saved contact criterion changed for {case_name}")
        if not event.get("postcontact_dynamics_verified"):
            raise EvidenceMismatch(f"saved postcontact authentication is absent for {case_name}")
        target_time = float(event["precontact_time"])
        floor = APPROACH_FLOOR_FRACTION_FROZEN * float(bundle.u_impact)
        if not math.isclose(float(event["minimum_approach_floor"]), floor, rel_tol=0.0, abs_tol=1e-12):
            raise EvidenceMismatch(f"saved approach floor changed for {case_name}")
        n_steps = int(round(target_time / float(bundle.p.dt)))
        if not math.isclose(n_steps * float(bundle.p.dt), target_time, rel_tol=0.0, abs_tol=1e-10):
            raise EvidenceMismatch(f"saved precontact time is not on the frozen dt grid for {case_name}")
        saved_pre_rows = event_case["precontact_rows"]
        saved_pre = min(saved_pre_rows, key=lambda row: abs(float(row["t"]) - target_time))
        if not math.isclose(float(saved_pre["t"]), target_time, rel_tol=0.0, abs_tol=1e-10):
            raise EvidenceMismatch(f"saved exact precontact row is missing for {case_name}")

        initial_state, initializer_details = audit.build_candidate(bundle, CANDIDATE)
        current_initial = audit.initial_metrics(bundle, CANDIDATE, initial_state, initializer_details)
        saved_init = saved_initial[case_name][CANDIDATE]
        for hash_name in ("initial_geometry_hash", "initial_phi_hash", "initial_u_hash", "initial_v_hash"):
            if current_initial[hash_name] != saved_init[hash_name]:
                raise EvidenceMismatch(f"{case_name}: {hash_name} differs from saved L1A-2s evidence")

        initial_support = audit._local_gap(np.asarray(initial_state.phi), 0.5, bundle.solid, bundle.p)
        initial_local_speed = audit._approach_speed(
            np.asarray(initial_state.u), np.asarray(initial_state.v), initial_support, bundle.p
        )
        if initial_local_speed is None:
            raise EvidenceMismatch(f"initial local approach not measurable for {case_name}")

        h = float(bundle.p.dt) / 3.0
        fast_kernel = impulse.make_fast_substep(bundle.solid, bundle.p, h)
        capture_kernel = make_stage_capture_kernel(bundle.solid, bundle.p, h)
        state = initial_state
        rolling_rows: list[dict[str, Any]] = []
        first_substep: dict[str, Any] | None = None
        first_local_stages: dict[str, Any] | None = None
        precontact_local_stages: list[dict[str, Any]] = []
        precontact_final_scalar_rows: list[dict[str, Any]] = []
        capture_steps = {1, n_steps}
        capture_steps.update(range(max(1, n_steps - ROLLING_PUBLIC_STEPS + 1), n_steps + 1))

        for public_step in range(1, n_steps + 1):
            for substep_index in range(3):
                t_before = float((public_step - 1) * bundle.p.dt + substep_index * h)
                state_before = state
                fast_state, fast_scalars = fast_kernel(state_before)
                needs_scalar = public_step == 1 or public_step >= n_steps - ROLLING_PUBLIC_STEPS + 1
                values = scalar_row(fast_scalars) if needs_scalar else None
                if values is not None:
                    row = {
                        "public_step": public_step,
                        "substep_index_1_based": substep_index + 1,
                        "t_before": t_before,
                        "t_after": t_before + h,
                        **values,
                    }
                    if public_step == 1 and substep_index == 0:
                        first_substep = row
                    if public_step >= n_steps - ROLLING_PUBLIC_STEPS + 1:
                        rolling_rows.append(row)
                    if public_step == n_steps:
                        precontact_final_scalar_rows.append(row)

                if public_step in capture_steps and (
                    (public_step == 1 and substep_index == 0) or public_step == n_steps
                ):
                    captured_state, fields = capture_kernel(state_before)
                    deltas = {
                        "phi": max_abs_delta(captured_state.phi, fast_state.phi),
                        "u": max_abs_delta(captured_state.u, fast_state.u),
                        "v": max_abs_delta(captured_state.v, fast_state.v),
                    }
                    if max(deltas.values()) > 2e-6:
                        raise EvidenceMismatch(
                            f"stage-capture kernel diverges from fast solver for {case_name}: {deltas}"
                        )
                    local = local_stage_row(
                        state_before,
                        fields,
                        bundle.solid,
                        bundle.p,
                        t_before,
                        public_step,
                        substep_index,
                    )
                    local["fast_vs_capture_state_max_abs_delta"] = deltas
                    if public_step == 1:
                        first_local_stages = local
                    if public_step == n_steps:
                        precontact_local_stages.append(local)
                state = fast_state

        if first_substep is None or first_local_stages is None:
            raise EvidenceMismatch(f"initial substep ledger missing for {case_name}")
        if len(precontact_final_scalar_rows) != 3 or len(precontact_local_stages) != 3:
            raise EvidenceMismatch(f"final precontact substep ledger incomplete for {case_name}")

        precontact_row = audit._trajectory_row(
            state.phi,
            state.u,
            state.v,
            bundle.solid,
            bundle.p,
            bundle,
            n_steps,
            target_time,
            include_constraints=True,
        )
        local_speed_rerun = float(precontact_row["local_approach_speed_into_solid"])
        local_speed_saved = float(event["precontact_local_normal_approach_speed_positive_toward_solid"])
        if not math.isclose(local_speed_rerun, local_speed_saved, rel_tol=0.0, abs_tol=2e-6):
            raise EvidenceMismatch(
                f"{case_name}: rerun local approach {local_speed_rerun} disagrees with saved {local_speed_saved}"
            )
        if not math.isclose(
            float(precontact_row["gap_phi05"]), float(saved_pre["gap_phi05"]), rel_tol=0.0, abs_tol=2e-6
        ):
            raise EvidenceMismatch(f"{case_name}: rerun precontact phi=0.5 gap disagrees with saved evidence")
        flux_keys = (
            "D_div_L2_fluid_volume_rms",
            "cutcell_flux_div_Linf",
            "cutcell_flux_div_L2_fluid_volume_rms",
            "embedded_wall_signed_flux_bilinear",
            "embedded_wall_absolute_flux_bilinear",
            "max_embedded_wall_bilinear_normal_velocity",
        )
        for key in flux_keys:
            saved_value = saved_pre.get(key)
            current_value = precontact_row.get(key)
            if (
                saved_value is not None
                and current_value is not None
                and not math.isclose(float(current_value), float(saved_value), rel_tol=0.0, abs_tol=2e-6)
            ):
                raise EvidenceMismatch(f"{case_name}: rerun {key} differs from the saved measured row")

        startup_saved = saved_startup["cases"][case_name][CANDIDATE]["causal_increment_ledger"]
        startup_checks = {
            "rhs_dv_liquid_mean": float(startup_saved["Delta_v_RHS_liquid_weighted"]),
            "brinkman_dv_liquid_mean_exact": float(startup_saved["Delta_v_Brinkman_liquid_weighted"]),
            "projection_dv_liquid_mean_exact": float(startup_saved["Delta_v_projection_liquid_weighted"]),
            "rhs_dv_grid_mean": float(startup_saved["Delta_grid_mean_v_RHS"]),
            "brinkman_dv_grid_mean_exact": float(startup_saved["Delta_grid_mean_v_Brinkman"]),
            "projection_dv_grid_mean_exact": float(startup_saved["Delta_grid_mean_v_projection"]),
        }
        startup_differences = {name: float(first_substep[name] - expected) for name, expected in startup_checks.items()}
        if max(abs(value) for value in startup_differences.values()) > 2e-7:
            raise EvidenceMismatch(
                f"{case_name}: short-run startup ledger differs from saved ledger: {startup_differences}"
            )

        floor = APPROACH_FLOOR_FRACTION_FROZEN * float(bundle.u_impact)
        output["saved_full_horizon_outcomes_reused"][case_name] = {
            "event_verdict": event["verdict"],
            "contact_time_interpolated_gap_threshold": float(event["contact_time_interpolated_gap_threshold"]),
            "contact_time_discrete": float(event["contact_time_discrete"]),
            "precontact_time": target_time,
            "saved_precontact_gap_phi05": float(saved_pre["gap_phi05"]),
            "saved_precontact_local_normal_approach_speed": local_speed_saved,
            "approach_floor": floor,
            "postcontact_dynamics_verified": bool(event["postcontact_dynamics_verified"]),
            "postcontact_interval": event["postcontact_interval"],
            "contact_location": event["contact_location"],
        }
        output["cases_detail"][case_name] = {
            "case_geometry_and_timestep": {
                "geometry_hash": current_initial["initial_geometry_hash"],
                "phi_hash": current_initial["initial_phi_hash"],
                "u_hash": current_initial["initial_u_hash"],
                "v_hash": current_initial["initial_v_hash"],
                "initial_gap_from_case_geometry": float(bundle.initial_gap),
                "local_surface_top": float(bundle.local_surface_top),
                "dx": float(bundle.p.dx),
                "eps": float(bundle.p.eps),
                "eps_over_dx": float(bundle.p.eps / bundle.p.dx),
                "effective_dt": float(bundle.p.dt),
                "internal_substep_h": h,
                "contact_gap_cells": CONTACT_GAP_CELLS_FROZEN,
            },
            "initialized_velocity": {
                "candidate": CANDIDATE,
                "diagnostic_only": True,
                "u_impact": float(bundle.u_impact),
                "core_v_phi_ge_0p9": float(saved_init["actual_initial_core_v_phi_ge_0p9"]),
                "liquid_v_weighted": float(saved_init["actual_initial_liquid_weighted_v"]),
                "initial_local_normal_approach_speed": float(initial_local_speed),
                "initial_phi05_support": initial_support,
                "central_D_div_linf": float(saved_init["initial_D_div_Linf"]),
                "open_face_fv_div_linf": float(saved_init["cutcell_flux_div_Linf"]),
                "cutcell_flux_status": saved_init["cutcell_flux_divergence_status"],
                "wall_flux_status": saved_init["embedded_wall_flux_status"],
            },
            "precontact_velocity": {
                "time": target_time,
                "rerun_row_matches_saved_local_speed": True,
                "local_normal_approach_speed": local_speed_rerun,
                "local_speed_over_u_impact": local_speed_rerun / float(bundle.u_impact),
                "fraction_of_frozen_approach_floor": local_speed_rerun / floor,
                "liquid_v_weighted": float(precontact_row["v_liquid"]),
                "core_v_phi_ge_0p9": float(precontact_row["v_core"]),
                "gap_phi05": float(precontact_row["gap_phi05"]),
                "gap_phi05_over_dx": float(precontact_row["gap_phi05"] / bundle.p.dx),
                "precontact_support": precontact_row["local_support_phi05"],
                "discrete_divergence_and_flux": {
                    "central_D_linf": float(precontact_row["initial_D_div_Linf"]),
                    "central_D_l2_fluid_volume_rms": float(precontact_row["D_div_L2_fluid_volume_rms"]),
                    "cutcell_open_face_FV_linf": float(precontact_row["cutcell_flux_div_Linf"]),
                    "cutcell_open_face_FV_l2_fluid_volume_rms": float(
                        precontact_row["cutcell_flux_div_L2_fluid_volume_rms"]
                    ),
                    "cutcell_flux_status": precontact_row["cutcell_flux_divergence_status"],
                    "cutcell_flux_reconstruction": precontact_row["cutcell_flux_reconstruction"],
                    "saved_precontact_cutcell_measurement_available": saved_pre.get("cutcell_flux_divergence_status")
                    is not None,
                },
                "embedded_wall_flux": {
                    "signed_bilinear_flux": float(precontact_row["embedded_wall_signed_flux_bilinear"]),
                    "absolute_bilinear_flux": float(precontact_row["embedded_wall_absolute_flux_bilinear"]),
                    "max_bilinear_wall_normal_speed": float(
                        precontact_row["max_embedded_wall_bilinear_normal_velocity"]
                    ),
                    "measurement_status": precontact_row["embedded_wall_flux_status"],
                },
            },
            "operator_response_first_internal_substep": {
                "global_scalar_ledger": {
                    name: float(first_substep[name])
                    for name in (
                        "rhs_dv_liquid_mean",
                        "brinkman_dv_liquid_mean_exact",
                        "projection_dv_liquid_mean_exact",
                        "rhs_dv_grid_mean",
                        "brinkman_dv_grid_mean_exact",
                        "projection_dv_grid_mean_exact",
                        "proj_dv_grid_linf_v",
                        "brinkman_dv_grid_l2",
                        "proj_dv_grid_l2",
                        "mean_of_proj_correction",
                        "damp_min",
                        "damp_max",
                        "div_before_linf",
                        "div_after_rhs_linf",
                        "div_after_brinkman_linf",
                        "div_after_projection_linf",
                    )
                },
                "saved_startup_ledger_max_abs_delta": max(abs(value) for value in startup_differences.values()),
                "local_normal_stages": first_local_stages,
            },
            "operator_response_last_10_precontact_public_steps": stage_window_summary(rolling_rows),
            "operator_response_final_precontact_public_step": {
                "local_normal_stages": precontact_local_stages,
                "global_scalar_ledger": [
                    {
                        name: float(row[name]) if name != "cg_converged" else bool(row[name])
                        for name in (
                            "t_before",
                            "t_after",
                            "rhs_dv_liquid_mean",
                            "brinkman_dv_liquid_mean_exact",
                            "projection_dv_liquid_mean_exact",
                            "rhs_dv_grid_mean",
                            "brinkman_dv_grid_mean_exact",
                            "projection_dv_grid_mean_exact",
                            "proj_dv_grid_linf_v",
                            "mean_of_proj_correction",
                            "div_before_linf",
                            "div_after_brinkman_linf",
                            "div_after_projection_linf",
                            "cg_converged",
                        )
                    }
                    for row in precontact_final_scalar_rows
                ],
            },
        }

    output["production_lineage_guard"] = {
        "diagnostic_only": True,
        "production_lineage_eligible": False,
        "official_data_written": False,
        "candidate_promoted": False,
        "solver_or_default_changed": False,
        "geometry_or_contact_threshold_changed": False,
        "contract_13_or_l1b_readiness_claimed": False,
        "historical_readiness_statuses_overwritten": False,
    }
    output["execution"] = {
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_backend": jax.default_backend(),
        "head_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_hashes": frozen["source_hashes"],
        "short_window_run_steps": {
            case: int(
                round(output["saved_full_horizon_outcomes_reused"][case]["precontact_time"] / output["effective_dt"])
            )
            for case in CASE_NAMES
        },
        "long_forensic_t8_calculation_repeated": False,
    }

    target = HERE / "pillar_flat_operator_comparison.json"
    target.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"wrote {target}")
    for case_name in CASE_NAMES:
        record = output["cases_detail"][case_name]
        print(
            case_name,
            "initial_local=",
            record["initialized_velocity"]["initial_local_normal_approach_speed"],
            "precontact_local=",
            record["precontact_velocity"]["local_normal_approach_speed"],
            "floor_fraction=",
            record["precontact_velocity"]["fraction_of_frozen_approach_floor"],
            "projection_mean=",
            record["operator_response_final_precontact_public_step"]["global_scalar_ledger"][0][
                "projection_dv_liquid_mean_exact"
            ],
            "brinkman_mean=",
            record["operator_response_final_precontact_public_step"]["global_scalar_ledger"][0][
                "brinkman_dv_liquid_mean_exact"
            ],
        )


if __name__ == "__main__":
    main()
