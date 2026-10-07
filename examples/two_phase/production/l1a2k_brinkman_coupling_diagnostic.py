"""Evidence-triggered, diagnostic-only Brinkman damping A/B for L1A-2k.

This supplemental run is intentionally separate from the primary audit runner so its source hash
cannot silently invalidate the strictly validated L1A-2k snapshots. It loads those snapshots with
the primary runner's exact contract/SHA/source/config/state/step checks, holds phi fixed, and changes
only ``solid.chi`` (the Brinkman damping multiplier in the forensic momentum substep). Geometry,
wall energy, contact angle, capillary force, dt, M, density, viscosity, and pressure projection are
otherwise identical. The result is not a production trajectory or validation.

Run from ``examples/two_phase``::

    JAX_ENABLE_X64=1 python -m production.l1a2k_brinkman_coupling_diagnostic
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import phasefield as pf
from production import capillary_pressure_balance_audit as audit
from production import chns_nonstationarity_audit as chns

jax.config.update("jax_enable_x64", True)

STEPS = 100
CASES = {
    "authority_060": {"target_deg": 60.0, "step": 50_000},
    "control_090": {"target_deg": 90.0, "step": 27_200},
    "control_150": {"target_deg": 150.0, "step": 100_000},
}
ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "l1a2k"
EVIDENCE_DIR = ROOT / "evidence" / "l1a2k"
REPORT_PATH = EVIDENCE_DIR / "capillary_pressure_balance_report.json"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hash(value: Any) -> str:
    """Hash all array leaves of a pytree, preserving dtype, shape, and bytes."""
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(value):
        array = np.ascontiguousarray(np.asarray(leaf))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _l2_pair(u: np.ndarray, v: np.ndarray, mask: np.ndarray, area: float) -> float:
    u64 = np.asarray(u, dtype=np.float64)
    v64 = np.asarray(v, dtype=np.float64)
    local = np.asarray(mask, dtype=bool)
    return float(np.sqrt(np.sum((u64[local] ** 2 + v64[local] ** 2), dtype=np.float64) * area))


def _linf_pair(u: np.ndarray, v: np.ndarray, mask: np.ndarray) -> float:
    local = np.asarray(mask, dtype=bool)
    magnitude = np.sqrt(np.asarray(u, dtype=np.float64) ** 2 + np.asarray(v, dtype=np.float64) ** 2)
    return float(np.max(magnitude[local])) if np.any(local) else 0.0


def _scalar_l2(value: np.ndarray, area: float) -> float:
    value64 = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.sum(value64**2, dtype=np.float64) * area))


def _function_record(name: str, function: Any) -> dict[str, Any]:
    try:
        lines, start = inspect.getsourcelines(function)
        text = "".join(lines)
        return {
            "name": name,
            "file": str(Path(inspect.getsourcefile(function) or "").resolve()),
            "line_start": int(start),
            "line_end": int(start + len(lines) - 1),
            "sha256_source": hashlib.sha256(text.encode()).hexdigest(),
        }
    except (OSError, TypeError):
        return {"name": name, "file": None, "line_start": None, "line_end": None, "sha256_source": None}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_field_artifact(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    return _file_sha256(path)


def _case_config(target_deg: float) -> tuple[pf.PhaseFieldParams, pf.Solid, dict[str, Any]]:
    p, solid, _seed, config = chns._make_case(target_deg)
    return p, solid, config


def _load_exact_case(
    case_name: str,
    case_request: dict[str, Any],
    base_report: dict[str, Any],
) -> tuple[pf.State, pf.PhaseFieldParams, pf.Solid, dict[str, Any], dict[str, Any]]:
    target_deg = float(case_request["target_deg"])
    step = int(case_request["step"])
    p, solid, config = _case_config(target_deg)
    reported_case = base_report["cases"][case_name]
    if reported_case.get("step") != step or reported_case.get("config") != config:
        raise RuntimeError(f"{case_name}: required step/config does not match the primary report")
    snapshot = ARTIFACT_DIR / "snapshots" / f"{case_name}_step_{step:06d}.npz"
    state, metadata = audit._load_state_snapshot(
        snapshot,
        case_name=case_name,
        step=step,
        config=config,
    )
    if metadata.get("state_hashes") != reported_case.get("state_hashes"):
        raise RuntimeError(f"{case_name}: strict snapshot hashes differ from the measured report")
    if chns._state_hashes(state) != reported_case["state_hashes"]:
        raise RuntimeError(f"{case_name}: loaded state hash differs from the measured report")
    return state, p, solid, config, metadata


def _run_case(
    case_name: str,
    case_request: dict[str, Any],
    base_report: dict[str, Any],
    steps: int,
) -> dict[str, Any]:
    state, p, solid_on, config, metadata = _load_exact_case(case_name, case_request, base_report)
    solid_off = solid_on._replace(chi=jnp.zeros_like(solid_on.chi))
    fields_on = {name: _tree_hash(getattr(solid_on, name)) for name in solid_on._fields}
    fields_off = {name: _tree_hash(getattr(solid_off, name)) for name in solid_off._fields}
    changed_solid_fields = [name for name in solid_on._fields if fields_on[name] != fields_off[name]]
    if changed_solid_fields != ["chi"]:
        raise RuntimeError(f"{case_name}: Brinkman A/B changed solid fields {changed_solid_fields}")

    print(f"[l1a2k Brinkman A/B] {case_name}: running {steps} frozen-phi steps", flush=True)
    result_on = chns.advance_forensic_block(
        state, solid_on, p, True, 1.0, steps, 1
    )
    result_off = chns.advance_forensic_block(
        state, solid_off, p, True, 1.0, steps, 1
    )
    state_on, _samples_on, _sample_fields_on, _observations_on, fields_on_last = result_on
    state_off, _samples_off, _sample_fields_off, _observations_off, fields_off_last = result_off
    state_on.t.block_until_ready()
    state_off.t.block_until_ready()

    phi_start = np.asarray(state.phi)
    phi_on = np.asarray(state_on.phi)
    phi_off = np.asarray(state_off.phi)
    if not np.array_equal(phi_start, phi_on) or not np.array_equal(phi_start, phi_off):
        raise RuntimeError(f"{case_name}: frozen-phi Brinkman A/B changed phase")

    u_start, v_start = np.asarray(state.u), np.asarray(state.v)
    u_on, v_on = np.asarray(state_on.u), np.asarray(state_on.v)
    u_off, v_off = np.asarray(state_off.u), np.asarray(state_off.v)
    du = u_off.astype(np.float64) - u_on.astype(np.float64)
    dv = v_off.astype(np.float64) - v_on.astype(np.float64)
    area = float(p.dx * p.dy)
    chi = np.asarray(solid_on.chi, dtype=np.float64)
    masks = {
        "all_momentum_cells": np.ones_like(chi, dtype=bool),
        "brinkman_chi_gt_010": chi > 0.10,
        "brinkman_chi_gt_050": chi > 0.50,
        "brinkman_chi_gt_090": chi > 0.90,
        "outside_brinkman_chi_gt_050": chi <= 0.50,
    }

    cap_on_u = np.asarray(fields_on_last.capillary_u)
    cap_on_v = np.asarray(fields_on_last.capillary_v)
    cap_off_u = np.asarray(fields_off_last.capillary_u)
    cap_off_v = np.asarray(fields_off_last.capillary_v)
    mu_equal = np.array_equal(np.asarray(fields_on_last.mu), np.asarray(fields_off_last.mu))
    capillary_equal = np.array_equal(cap_on_u, cap_off_u) and np.array_equal(cap_on_v, cap_off_v)
    if not mu_equal or not capillary_equal:
        raise RuntimeError(f"{case_name}: frozen-phi force/chemical-potential changed across the A/B")

    brinkman_on_u = np.asarray(fields_on_last.brinkman_u)
    brinkman_on_v = np.asarray(fields_on_last.brinkman_v)
    brinkman_off_u = np.asarray(fields_off_last.brinkman_u)
    brinkman_off_v = np.asarray(fields_off_last.brinkman_v)
    brinkman_off_l2 = _l2_pair(brinkman_off_u, brinkman_off_v, masks["all_momentum_cells"], area)
    if brinkman_off_l2 != 0.0:
        raise RuntimeError(f"{case_name}: chi=0 diagnostic branch has nonzero Brinkman damping")

    rho = np.asarray(pf.rho_of(state.phi, p), dtype=np.float64)
    kinetic_on = float(0.5 * np.sum(rho * (u_on.astype(np.float64) ** 2 + v_on.astype(np.float64) ** 2)) * area)
    kinetic_off = float(0.5 * np.sum(rho * (u_off.astype(np.float64) ** 2 + v_off.astype(np.float64) ** 2)) * area)
    start_velocity_l2 = _l2_pair(u_start, v_start, masks["all_momentum_cells"], area)
    on_velocity_l2 = _l2_pair(u_on, v_on, masks["all_momentum_cells"], area)
    off_velocity_l2 = _l2_pair(u_off, v_off, masks["all_momentum_cells"], area)
    delta_l2 = _l2_pair(du, dv, masks["all_momentum_cells"], area)
    brinkman_on_l2 = _l2_pair(
        brinkman_on_u,
        brinkman_on_v,
        masks["all_momentum_cells"],
        area,
    )
    delta_by_region = {
        name: {
            "velocity_difference_l2_off_minus_on": _l2_pair(du, dv, mask, area),
            "velocity_difference_linf_off_minus_on": _linf_pair(du, dv, mask),
            "difference_energy_fraction": float(
                _l2_pair(du, dv, mask, area) ** 2 / max(delta_l2**2, 1.0e-300)
            ),
            "n_cells": int(np.count_nonzero(mask)),
        }
        for name, mask in masks.items()
    }
    fields_finite = all(
        np.isfinite(np.asarray(value)).all()
        for value in (
            u_on,
            v_on,
            u_off,
            v_off,
            np.asarray(fields_on_last.divergence_after),
            np.asarray(fields_off_last.divergence_after),
            np.asarray(fields_on_last.poisson_residual),
            np.asarray(fields_off_last.poisson_residual),
        )
    )
    if not fields_finite:
        raise RuntimeError(f"{case_name}: non-finite A/B state or projection diagnostic")

    relative_velocity_difference = delta_l2 / max(on_velocity_l2, 1.0e-300)
    array_path = ARTIFACT_DIR / "fields" / f"brinkman_ab_{case_name}_step_{steps:04d}.npz"
    array_sha = _write_field_artifact(
        array_path,
        {
            "phi": phi_start,
            "u_start": u_start,
            "v_start": v_start,
            "u_on": u_on,
            "v_on": v_on,
            "u_off": u_off,
            "v_off": v_off,
            "delta_u_off_minus_on": du,
            "delta_v_off_minus_on": dv,
            "brinkman_u_on_last_substep": brinkman_on_u,
            "brinkman_v_on_last_substep": brinkman_on_v,
            "brinkman_u_off_last_substep": brinkman_off_u,
            "brinkman_v_off_last_substep": brinkman_off_v,
            "chi": chi,
        },
    )
    print(
        f"[l1a2k Brinkman A/B] {case_name}: delta velocity L2={delta_l2:.8g}, "
        f"off/on={off_velocity_l2 / max(on_velocity_l2, 1.0e-300):.8g}",
        flush=True,
    )
    return {
        "case": case_name,
        "target_deg": float(case_request["target_deg"]),
        "matched_state_step": int(case_request["step"]),
        "matched_state_hashes": metadata["state_hashes"],
        "config_fingerprint": chns._canonical_hash(config),
        "state_hashes_after_on": chns._state_hashes(state_on),
        "state_hashes_after_off": chns._state_hashes(state_off),
        "steps": int(steps),
        "physical_time_advanced": float(steps * float(p.dt)),
        "diagnostic_only": True,
        "production_acceptance_evidence": False,
        "phase_frozen_bitwise": True,
        "same_start_state": True,
        "only_solid_field_changed": changed_solid_fields,
        "all_other_solid_field_hashes_identical": all(
            fields_on[name] == fields_off[name] for name in solid_on._fields if name != "chi"
        ),
        "same_geometry_wall_energy_wetting_contact_angle_density_viscosity_dt_M_force_and_projection": True,
        "same_chemical_potential_and_capillary_force_bitwise": bool(mu_equal and capillary_equal),
        "brinkman_off_last_substep_l2": brinkman_off_l2,
        "brinkman_on_last_substep_l2": brinkman_on_l2,
        "velocity_l2": {
            "start": start_velocity_l2,
            "brinkman_on": on_velocity_l2,
            "brinkman_off": off_velocity_l2,
            "off_minus_on": delta_l2,
            "off_minus_on_over_on": relative_velocity_difference,
            "off_over_on": off_velocity_l2 / max(on_velocity_l2, 1.0e-300),
            "off_minus_on_linf": _linf_pair(du, dv, masks["all_momentum_cells"]),
        },
        "kinetic_energy": {
            "brinkman_on": kinetic_on,
            "brinkman_off": kinetic_off,
            "off_minus_on": kinetic_off - kinetic_on,
        },
        "regional_velocity_difference": delta_by_region,
        "projection_observations_last_internal_substep": {
            "divergence_after_l2_on": _scalar_l2(np.asarray(fields_on_last.divergence_after), area),
            "divergence_after_l2_off": _scalar_l2(np.asarray(fields_off_last.divergence_after), area),
            "poisson_residual_l2_on": _scalar_l2(np.asarray(fields_on_last.poisson_residual), area),
            "poisson_residual_l2_off": _scalar_l2(np.asarray(fields_off_last.poisson_residual), area),
        },
        "field_artifact": {
            "path": str(array_path.relative_to(ROOT)),
            "sha256": array_sha,
        },
    }


def _supplement_metadata(steps: int) -> dict[str, Any]:
    function_map = audit.operator_map()
    relevant = {
        "phasefield._ddx",
        "phasefield._ddy",
        "phasefield._lap",
        "PhaseFieldParams.m2_proj",
        "phasefield.poisson_solve",
        "phasefield.rhs",
        "phasefield.rho_of",
        "phasefield.nu_of",
        "chns_nonstationarity_audit._momentum_components",
        "chns_nonstationarity_audit._forensic_substep",
    }
    source_functions = [item for item in function_map["source_functions"] if item["name"] in relevant]
    source_functions.extend(
        [
            audit._function_source_record(
                "chns_nonstationarity_audit._forensic_step_impl", chns._forensic_step_impl
            ),
            audit._function_source_record(
                "chns_nonstationarity_audit.advance_forensic_block",
                chns.advance_forensic_block.__wrapped__,
            ),
        ]
    )
    return {
        "name": "brinkman_on_off_frozen_phi_ab",
        "status": "measured_diagnostic_only",
        "trigger": (
            "evidence-triggered follow-up: chi>0.50 contains 25.0%, 29.5%, and 39.9% of the production "
            "residual energy in the 60/90/150-degree matched cases"
        ),
        "method": (
            "100-step frozen-phi matched A/B; only solid.chi is zeroed in the Brinkman damping term in "
            "the audit substep; exact L1A-2k snapshots are loaded with strict validation"
        ),
        "steps": int(steps),
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": _file_sha256(Path(__file__).resolve()),
        "result_artifact": "artifacts/l1a2k/diagnostics/brinkman_on_off_ab.json",
        "baseline_audit_source_hashes": audit._source_hashes(),
        "source_functions": source_functions,
        "source_functions_for_supplement": [
            _function_record("l1a2k_brinkman_coupling_diagnostic._run_case", _run_case),
            _function_record("l1a2k_brinkman_coupling_diagnostic._load_exact_case", _load_exact_case),
        ],
    }


def _merge_into_evidence(result: dict[str, Any]) -> None:
    base = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    if base.get("status") != "complete" or base.get("profile") != "forensic":
        raise RuntimeError("refusing to merge supplemental A/B into an incomplete/non-forensic report")
    if base.get("source_hashes") != result["supplemental_diagnostic"]["baseline_audit_source_hashes"]:
        raise RuntimeError("primary audit source hashes changed; supplemental merge is no longer matched")
    if base.get("solver_contract_version") != 11:
        raise RuntimeError("supplemental A/B requires solver contract 11")
    for case_name, observed in result["cases"].items():
        base_case = base["cases"].get(case_name, {})
        if base_case.get("step") != observed["matched_state_step"]:
            raise RuntimeError(f"{case_name}: report step changed before supplemental merge")
        if base_case.get("state_hashes") != observed["matched_state_hashes"]:
            raise RuntimeError(f"{case_name}: report state hash changed before supplemental merge")
        base_case["brinkman_on_off_ab"] = observed

    candidate = base["mechanism_matrix"]["candidates"]["BRINKMAN_WALL_COUPLING"]
    candidate["claim_tested"] = (
        "changing only Brinkman chi damping changes the short frozen-phi momentum response "
        "on all three exact matched states"
    )
    candidate["status"] = "SUPPORTED" if all(
        item["velocity_l2"]["off_minus_on"] > 0.0 for item in result["cases"].values()
    ) else "NOT_TESTED"
    candidate["scope_caveat"] = (
        "the isolated A/B supports a Brinkman effect on the short frozen-phi momentum response; "
        "it does not establish that Brinkman coupling caused the 60-degree angle nonstationarity"
    )
    candidate["evidence"] = {
        "production_residual_energy_fraction_in_chi_gt_050": {
            case_name: base["cases"][case_name]["regional_spatial_summary"]["brinkman_chi_overlap"]["chi_gt_050"][
                "residual_energy_fraction"
            ]
            for case_name in CASES
        },
        "frozen_phi_on_off_ab": result["cases"],
    }
    base.setdefault("supplemental_diagnostics", {})["brinkman_on_off_ab"] = result["supplemental_diagnostic"]
    base["unmeasured_sections"]["brinkman_on_off_ab"] = "measured_100_step_diagnostic_ab_not_production_validation"

    _write_json(REPORT_PATH, base)
    _write_json(EVIDENCE_DIR / "mechanism_matrix.json", base["mechanism_matrix"])

    operator_path = EVIDENCE_DIR / "operator_map.json"
    operator_map = json.loads(operator_path.read_text(encoding="utf-8"))
    operator_map.setdefault("supplemental_diagnostics", {})["brinkman_on_off_ab"] = {
        "script": result["supplemental_diagnostic"]["script"],
        "script_sha256": result["supplemental_diagnostic"]["script_sha256"],
        "source_functions": result["supplemental_diagnostic"]["source_functions"],
        "source_functions_for_supplement": result["supplemental_diagnostic"]["source_functions_for_supplement"],
        "method": result["supplemental_diagnostic"]["method"],
        "diagnostic_only": True,
    }
    _write_json(operator_path, operator_map)

    manifest_path = EVIDENCE_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("supplemental_diagnostics", {})["brinkman_on_off_ab"] = {
        "status": "measured_diagnostic_only",
        "script": result["supplemental_diagnostic"]["script"],
        "script_sha256": result["supplemental_diagnostic"]["script_sha256"],
        "result_artifact": result["supplemental_diagnostic"]["result_artifact"],
        "steps": result["supplemental_diagnostic"]["steps"],
        "cases": {
            name: item["matched_state_step"] for name, item in result["cases"].items()
        },
    }
    manifest.setdefault("outputs", {})["brinkman_on_off_ab"] = "measured_posthoc_diagnostic_only"
    manifest["outputs"]["mechanisms"] = (
        "complete_with_unrun_ablation_sections_marked_unmeasured_and_brinkman_ab_measured"
    )
    _write_json(manifest_path, manifest)

    markdown_path = EVIDENCE_DIR / "capillary_pressure_balance_report.md"
    markdown = markdown_path.read_text(encoding="utf-8")
    mechanism_status = candidate["status"]
    new_row = (
        f"| `BRINKMAN_WALL_COUPLING` | **{mechanism_status}** | isolated frozen-phi A/B changes "
        "the short momentum response; not evidence that Brinkman coupling caused 60-degree nonstationarity |"
    )
    row_found = False
    lines = markdown.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("| `BRINKMAN_WALL_COUPLING` | "):
            lines[index] = new_row
            row_found = True
            break
    if not row_found:
        raise RuntimeError("could not find the Brinkman mechanism row in the Markdown report")
    markdown = "\n".join(lines) + "\n"
    old_unmeasured = "- `brinkman_on_off_ab`: **unmeasured_pending_spatial_overlap_evidence_review**"
    new_unmeasured = "- `brinkman_on_off_ab`: **measured_100_step_diagnostic_ab_not_production_validation**"
    if old_unmeasured in markdown:
        markdown = markdown.replace(old_unmeasured, new_unmeasured, 1)
    elif new_unmeasured not in markdown:
        raise RuntimeError("could not find the expected Brinkman unmeasured entry in the Markdown report")

    table_rows = [
        (
            "| Case | Snapshot step | Steps | chi>0.50 residual-energy fraction | "
            "Velocity delta L2 (off−on) | delta/on L2 | Off/on velocity L2 | "
            "delta L2 inside chi>0.50 |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case_name in CASES:
        item = result["cases"][case_name]
        residual_fraction = base["cases"][case_name]["regional_spatial_summary"]["brinkman_chi_overlap"][
            "chi_gt_050"
        ]["residual_energy_fraction"]
        table_rows.append(
            "| `{}` | {} | {} | {:.6g} | {:.8g} | {:.8g} | {:.8g} | {:.8g} |".format(
                case_name,
                item["matched_state_step"],
                item["steps"],
                residual_fraction,
                item["velocity_l2"]["off_minus_on"],
                item["velocity_l2"]["off_minus_on_over_on"],
                item["velocity_l2"]["off_over_on"],
                item["regional_velocity_difference"]["brinkman_chi_gt_050"]["velocity_difference_l2_off_minus_on"],
            )
        )
    supplement_section = "\n".join(
        [
            "## Evidence-triggered Brinkman damping A/B (diagnostic only)",
            "",
            (
                "The residual-energy overlap in the fixed `chi>0.50` mask (25.0%, 29.5%, 39.9% for 60°, "
                "90°, 150°) triggered this short diagnostic. Each branch starts from the exact matched "
                "snapshot, holds phi fixed, and advances 100 steps. Only `solid.chi` in the Brinkman "
                "damping term changes. Geometry, wall energy, wetting, dt, M, capillary force, density, "
                "viscosity, and projection are otherwise unchanged. This is not a production trajectory, "
                "retuning, or validation."
            ),
            "",
            *table_rows,
            "",
            "- The chemical potential and applied capillary force were bitwise equal between branches; "
            "phi remained bitwise fixed. The `chi=0` branch has zero Brinkman term by construction.",
            "- Full per-case norms, regional differences, projection observations, strict snapshot "
            "provenance, field arrays, and supplemental source/function hashes are in "
            "`capillary_pressure_balance_report.json`, `mechanism_matrix.json`, `operator_map.json`, "
            "`artifacts/l1a2k/diagnostics/brinkman_on_off_ab.json`, and `artifacts/l1a2k/fields/`.",
            f"- `BRINKMAN_WALL_COUPLING` is `{mechanism_status}` only for this short frozen-phi "
            "momentum response. Causal attribution of the 60° angle nonstationarity remains "
            "`INCONCLUSIVE`; no Brinkman retuning or production change is recommended by this audit.",
            "",
        ]
    )
    anchor = "## CHNS-50k versus converged CH-only 60° phi"
    if "## Evidence-triggered Brinkman damping A/B (diagnostic only)" in markdown:
        start = markdown.index("## Evidence-triggered Brinkman damping A/B (diagnostic only)")
        end = markdown.find("\n## ", start + 1)
        if end < 0:
            markdown = markdown[:start] + supplement_section + "\n"
        else:
            markdown = markdown[:start] + supplement_section + "\n" + markdown[end + 1 :]
    elif anchor in markdown:
        markdown = markdown.replace(anchor, supplement_section + "\n" + anchor, 1)
    else:
        raise RuntimeError("could not place the supplemental Brinkman A/B section in Markdown")
    markdown_path.write_text(markdown, encoding="utf-8")


def run(*, steps: int = STEPS, merge: bool = True) -> dict[str, Any]:
    if steps < 1:
        raise ValueError("steps must be positive")
    if not jax.config.x64_enabled:
        raise RuntimeError("L1A-2k diagnostic reductions require JAX x64")
    if not REPORT_PATH.is_file():
        raise FileNotFoundError(f"required complete L1A-2k report not found: {REPORT_PATH}")
    base_report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    if base_report.get("status") != "complete" or base_report.get("profile") != "forensic":
        raise RuntimeError("Brinkman A/B requires a completed forensic L1A-2k report")
    if base_report.get("solver_contract_version") != 11 or pf.SOLVER_CONTRACT_VERSION != 11:
        raise RuntimeError("Brinkman A/B requires the unchanged solver contract 11")
    if base_report.get("source_hashes") != audit._source_hashes():
        raise RuntimeError("primary audit code/source hashes changed; exact snapshot reuse is not allowed")

    case_results = {
        name: _run_case(name, request, base_report, steps) for name, request in CASES.items()
    }
    if len(case_results) != len(CASES):
        raise RuntimeError("not all matched cases were measured")
    result = {
        "stage": "L1A-2k supplemental Brinkman A/B",
        "status": "measured_diagnostic_only",
        "solver_contract_version": 11,
        "git_sha": chns.get_git_sha(),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "diagnostic_only": True,
        "production_semantics_changed": False,
        "production_acceptance_evidence": False,
        "profile": "short_frozen_phi_brinkman_on_off_ab",
        "steps": int(steps),
        "cases": case_results,
        "supplemental_diagnostic": _supplement_metadata(steps),
    }
    out_path = ARTIFACT_DIR / "diagnostics" / "brinkman_on_off_ab.json"
    result["artifact_report"] = str(out_path.relative_to(ROOT))
    _write_json(out_path, result)
    if merge:
        _merge_into_evidence(result)
    print(f"Brinkman A/B complete: {out_path.relative_to(ROOT)}", flush=True)
    if merge:
        print(f"updated report: {REPORT_PATH.relative_to(ROOT)}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--no-merge", action="store_true", help="write diagnostic JSON/fields only")
    args = parser.parse_args()
    run(steps=args.steps, merge=not args.no_merge)


if __name__ == "__main__":
    main()
