"""Contract-11 production-default staged full-CHNS contact-angle closure audit.

This follow-up audit does not modify the solver or its production defaults. It runs all four target
angles to 50,000 CHNS steps before considering any extension. Subsequent milestones are reached only
when the previous state is finite, the cut-cell conserved-mass drift is at most 1e-3, and the
existing mobility-scaled relaxation diagnostic still classifies the case as relaxing. Checkpoints
persist the exact phase, velocity, and time state between milestones.

Run from ``examples/two_phase`` with float64 phase storage enabled::

    JAX_ENABLE_X64=1 python -m production.contract11_chns_stage_audit \\
        --out artifacts/contract11_followup/chns --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np
import phasefield as pf
from production import nonneutral_wetting_audit as nwa

TARGETS = (60.0, 90.0, 120.0, 150.0)
MILESTONES = (50_000, 100_000, 150_000, 200_000)
MASS_DRIFT_LIMIT = 1.0e-3
N = 128
EPS_FACTOR = 2.0
DT = 4.0e-3
R = 1.1
SAMPLE_EVERY = 200


def _code_sha256() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(nwa.__file__), Path(pf.__file__)):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _should_extend(
    milestone: int, *, complete: bool, drift_clean: bool, stationary: bool, still_relaxing: bool
) -> bool:
    return bool(
        milestone < MILESTONES[-1]
        and complete
        and drift_clean
        and not stationary
        and still_relaxing
    )


def _evaluate_milestone(record: dict[str, Any], milestone: int) -> dict[str, Any]:
    """Apply the staged continuation rule to one completed milestone."""
    samples = record.get("samples", [])
    mode = str(record.get("dynamics_mode", "chns"))
    if mode != "chns":
        raise ValueError(f"expected full CHNS record, got dynamics_mode={mode!r}")
    M = float(record.get("M", nwa.M_REF))
    history = samples[1:] if samples else []
    verdict = nwa._window_converged(history, ch_only=False, crit=nwa.CRITERIA, M=M)
    drift = record.get("conserved_mass_drift")
    drift_clean = isinstance(drift, (int, float)) and math.isfinite(float(drift)) and float(drift) <= MASS_DRIFT_LIMIT
    complete = (
        int(record.get("steps", -1)) == int(milestone)
        and record.get("stop_reason") == "budget_exhausted"
        and record.get("implicit_solve_failed") is False
        and record.get("detachment_observed") is False
        and record.get("contact_line_exists") is True
    )
    still_relaxing = False
    if complete and not verdict["converged"]:
        still_relaxing = bool(nwa._still_relaxing(history, nwa.CRITERIA, M, verdict))
    extend = _should_extend(
        milestone,
        complete=complete,
        drift_clean=drift_clean,
        stationary=bool(verdict["converged"]),
        still_relaxing=still_relaxing,
    )
    if not complete:
        reason = "milestone_incomplete_or_run_invalid"
    elif not drift_clean:
        reason = "formal_cutcell_mass_drift_exceeded"
    elif verdict["converged"]:
        reason = "stationary_convergence_window_passed"
    elif not still_relaxing:
        reason = "no_longer_relaxing"
    elif milestone == MILESTONES[-1]:
        reason = "maximum_200k_budget_reached"
    else:
        reason = "drift_clean_and_still_relaxing"
    return {
        "milestone_steps": int(milestone),
        "milestone_complete_and_valid": bool(complete),
        "formal_conserved_mass_drift_max": None if drift is None else float(drift),
        "formal_mass_drift_limit": MASS_DRIFT_LIMIT,
        "formal_mass_drift_clean": bool(drift_clean),
        "hard_mask_fluid_mass_drift_diagnostic": record.get("mass_drift"),
        "stationarity_window": verdict,
        "stationary": bool(verdict["converged"]),
        "still_relaxing": bool(still_relaxing),
        "extend_to_next_milestone": extend,
        "decision_reason": reason,
    }


def _stage_paths(out: Path, target: float, milestone: int) -> tuple[Path, Path]:
    label = f"theta_{int(target):03d}_step_{milestone:06d}"
    return out / "stages" / f"{label}.json", out / "checkpoints" / f"{label}.npz"


def _save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _run_or_load_stage(
    out: Path,
    target: float,
    milestone: int,
    *,
    previous: dict[str, Any] | None,
    overwrite: bool,
) -> dict[str, Any]:
    stage_path, checkpoint_path = _stage_paths(out, target, milestone)
    if not overwrite and stage_path.exists() and checkpoint_path.exists():
        cached = json.loads(stage_path.read_text())
        if cached.get("code_sha256") != _code_sha256():
            raise RuntimeError(f"cached stage {stage_path} was made by different runner code; use --overwrite")
        return cached

    start_step = 0 if previous is None else int(previous["evaluation"]["milestone_steps"])
    n_steps = milestone - start_step
    if n_steps <= 0:
        raise ValueError(f"invalid milestone progression {start_step} -> {milestone}")
    args: dict[str, Any] = {
        "target_deg": float(target),
        "ch_only": False,
        "N": N,
        "eps_factor": EPS_FACTOR,
        "M": nwa.M_REF,
        "dt": DT,
        "R": R,
        "budgets": (milestone,),
        "sample_every": SAMPLE_EVERY,
        "fixed_steps": n_steps,
        "keep_samples": 2_000,
        "group": "contract11_chns_staged",
        "label": f"fixed_to_{milestone}",
        "checkpoint_out": str(checkpoint_path.resolve()),
    }
    if previous is not None:
        previous_record = previous["record"]
        args.update(
            checkpoint_in=str(Path(previous["checkpoint_path"])),
            start_step=start_step,
            mass_reference=float(previous_record["mass_reference_initial"]),
            conserved_mass_reference=float(previous_record["conserved_mass_reference_initial"]),
            prior_samples=previous_record["samples"],
        )
    print(
        f"[contract11-chns {time.strftime('%H:%M:%S')}] target={target:.0f} "
        f"stage={start_step}->{milestone} M/Mref=1 N={N}",
        flush=True,
    )
    record = nwa.run_relaxation(**args)
    evaluation = _evaluate_milestone(record, milestone)
    result = {
        "target_deg": float(target),
        "evaluation": evaluation,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "stage_path": str(stage_path.resolve()),
        "code_sha256": _code_sha256(),
        "record": record,
    }
    _save_json(stage_path, result)
    print(
        f"[contract11-chns {time.strftime('%H:%M:%S')}] target={target:.0f} "
        f"steps={record['steps']} theta={record['final_sampled_angle_deg']} "
        f"formal_drift={evaluation['formal_conserved_mass_drift_max']} "
        f"stationary={evaluation['stationary']} still_relaxing={evaluation['still_relaxing']} "
        f"next={evaluation['extend_to_next_milestone']} ({evaluation['decision_reason']})",
        flush=True,
    )
    return result


def _final_case(target: float, stages: list[dict[str, Any]]) -> dict[str, Any]:
    final = stages[-1]
    record = final["record"]
    evaluation = final["evaluation"]
    angle = (
        record.get("final_sampled_angle_deg")
        if evaluation["milestone_complete_and_valid"]
        and evaluation["stationary"]
        and evaluation["formal_mass_drift_clean"]
        else None
    )
    return {
        "target_deg": float(target),
        "milestone_complete_and_valid": bool(evaluation["milestone_complete_and_valid"]),
        "final_steps": int(record["steps"]),
        "final_sampled_angle_deg": record.get("final_sampled_angle_deg"),
        "equilibrium_angle_deg": angle,
        "stationary": bool(evaluation["stationary"]),
        "formal_conserved_mass_drift_max": evaluation["formal_conserved_mass_drift_max"],
        "formal_mass_drift_clean": bool(evaluation["formal_mass_drift_clean"]),
        "hard_mask_fluid_mass_drift_diagnostic": evaluation["hard_mask_fluid_mass_drift_diagnostic"],
        "solid_phase_fraction_max": record.get("solid_phase_fraction_max"),
        "free_energy_monotonic_violations": record.get("free_energy_monotonic_violations"),
        "implicit_solve_failed": record.get("implicit_solve_failed"),
        "detachment_observed": record.get("detachment_observed"),
        "stage_decisions": [stage["evaluation"] for stage in stages],
        "stage_artifacts": [
            {
                "stage_path": stage["stage_path"],
                "checkpoint_path": stage["checkpoint_path"],
            }
            for stage in stages
        ],
    }


def run_audit(out: str | Path, *, overwrite: bool = False) -> dict[str, Any]:
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=True)

    # Phase 1 is deliberately separate: all four target cases reach/evaluate 50k before any case
    # is considered for a 100k extension.
    stage_50k: dict[float, dict[str, Any]] = {}
    for target in TARGETS:
        stage_50k[target] = _run_or_load_stage(
            root, target, MILESTONES[0], previous=None, overwrite=overwrite
        )

    # Phase 2 starts only after every 50k stage above has completed. At every later milestone the
    # next extension is re-gated on the formal cut-cell mass drift and the same relaxation test.
    all_stages: dict[float, list[dict[str, Any]]] = {target: [stage_50k[target]] for target in TARGETS}
    for target in TARGETS:
        previous = stage_50k[target]
        for milestone in MILESTONES[1:]:
            if not previous["evaluation"]["extend_to_next_milestone"]:
                break
            current = _run_or_load_stage(
                root, target, milestone, previous=previous, overwrite=overwrite
            )
            all_stages[target].append(current)
            previous = current

    cases = {f"{target:g}": _final_case(target, all_stages[target]) for target in TARGETS}
    converged_angles = [
        case["equilibrium_angle_deg"]
        for case in cases.values()
        if case["equilibrium_angle_deg"] is not None and case["formal_mass_drift_clean"]
    ]
    complete_angles = len(converged_angles) == len(TARGETS)
    errors = [abs(float(cases[f"{target:g}"]["equilibrium_angle_deg"]) - target) for target in TARGETS] if complete_angles else []
    measured = [float(cases[f"{target:g}"]["equilibrium_angle_deg"]) for target in TARGETS] if complete_angles else []
    mae = float(sum(errors) / len(errors)) if complete_angles else None
    max_error = float(max(errors)) if complete_angles else None
    monotonic = bool(all(b >= a for a, b in zip(measured, measured[1:]))) if complete_angles else False
    theta90 = cases["90"]["equilibrium_angle_deg"]
    neutral_ok = theta90 is not None and abs(float(theta90) - 90.0) <= 3.0
    angle_gate = (
        complete_angles
        and mae is not None
        and mae <= 5.0
        and max_error is not None
        and max_error <= 10.0
        and neutral_ok
        and monotonic
    )
    mass_gate = all(case["formal_mass_drift_clean"] for case in cases.values())
    solid_gate = all(
        isinstance(case["solid_phase_fraction_max"], (int, float))
        and float(case["solid_phase_fraction_max"]) <= 1.0e-6
        for case in cases.values()
    )
    report = {
        "audit": "contract11_production_chns_staged_closure",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
        "git_sha": _git_sha(),
        "code_sha256": _code_sha256(),
        "production_settings": {
            "dynamics_mode": "chns",
            "N": N,
            "R": R,
            "eps_factor": EPS_FACTOR,
            "dt": DT,
            "M": nwa.M_REF,
            "M_over_M_ref": 1.0,
            "dtype": "float32 velocity, contract-default float64 phase state",
            "ch_solver_rtol": 1.0e-6,
            "phase_transport_geometry": "sdf_cutcell_fv_v1",
            "wall_measure_method": str(pf.WALL_MEASURE_METHOD),
        },
        "staged_protocol": {
            "targets_deg": list(TARGETS),
            "milestones_steps": list(MILESTONES),
            "all_targets_reach_50k_before_extension": True,
            "extension_requires": [
                "complete finite milestone",
                f"sum_i V_i phi_i relative drift <= {MASS_DRIFT_LIMIT:g}",
                "not stationary under the existing CHNS convergence window",
                "existing _still_relaxing diagnostic is true",
            ],
            "mass_gate_quantity": "sum_i V_i phi_i",
            "full_cell_reconstruction_is_diagnostic_only": True,
        },
        "acceptance": {
            "all_four_equilibrium_angles_available": bool(complete_angles),
            "mae_deg": mae,
            "max_absolute_error_deg": max_error,
            "monotonic_target_to_measured": monotonic,
            "neutral_90_error_deg": None if theta90 is None else abs(float(theta90) - 90.0),
            "angle_gate_passed": bool(angle_gate),
            "formal_mass_gate_passed": bool(mass_gate),
            "solid_phase_gate_passed": bool(solid_gate),
            "four_angle_chns_closure_passed": bool(angle_gate and mass_gate and solid_gate),
        },
        "cases": cases,
        "stage_files": {
            f"{target:g}": [stage["stage_path"] for stage in all_stages[target]] for target in TARGETS
        },
        "readiness": {
            "contract_changed": False,
            "production_physics_changed": False,
            "chns_followup_protocol_complete": all(
                all(stage["evaluation"]["milestone_complete_and_valid"] for stage in stages)
                for stages in all_stages.values()
            ),
            "four_angle_production_chns_ready": bool(angle_gate and mass_gate and solid_gate),
        },
    }
    json_path = root / "contract11_chns_staged_report.json"
    _save_json(json_path, report)
    markdown_path = root / "contract11_chns_staged_report.md"
    markdown_path.write_text(format_markdown(report))
    return report


def _git_sha() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def format_markdown(report: dict[str, Any]) -> str:
    acceptance = report["acceptance"]
    lines = [
        "# Contract 11 staged production CHNS contact-angle audit",
        "",
        f"- Generated: `{report['generated_utc']}`",
        f"- Solver contract: **{report['solver_contract_version']}**",
        f"- Git HEAD: `{report['git_sha']}`",
        "- Production physics/defaults changed: **no**",
        "",
        "## Protocol",
        "",
        "All four targets are run to 50,000 steps before any extension. The formal drift is on "
        "`sum_i V_i phi_i`; the historical full-grid reconstruction is diagnostic only. A case may "
        "advance to the next milestone only when it is valid, drift-clean, nonstationary, and the "
        "existing relaxation classifier says it is still relaxing.",
        "",
        "## Per-angle results",
        "",
        "| Target | Steps | Equilibrium angle | Stationary | Conserved-mass drift | Hard-mask fluid-mass diagnostic drift | Stages evaluated |",
        "|---:|---:|---:|:---:|---:|---:|---|",
    ]
    for key, case in report["cases"].items():
        drift = case["formal_conserved_mass_drift_max"]
        diag = case["hard_mask_fluid_mass_drift_diagnostic"]
        extension = ", ".join(str(item["milestone_steps"]) for item in case["stage_decisions"])
        lines.append(
            f"| {case['target_deg']:.0f} | {case['final_steps']} | {case['equilibrium_angle_deg']} | "
            f"{case['stationary']} | {drift} | {diag} | {extension} |"
        )
    lines += [
        "",
        "## Matrix acceptance",
        "",
        f"- Four converged equilibrium angles available: `{acceptance['all_four_equilibrium_angles_available']}`",
        f"- MAE / maximum error: `{acceptance['mae_deg']}` / `{acceptance['max_absolute_error_deg']}` deg",
        f"- Monotonicity: `{acceptance['monotonic_target_to_measured']}`",
        f"- Neutral 90-degree gate: `{acceptance['neutral_90_error_deg']}` deg error",
        f"- Formal conserved-mass gate: `{acceptance['formal_mass_gate_passed']}`",
        f"- Solid-phase leakage gate: `{acceptance['solid_phase_gate_passed']}`",
        f"- Four-angle CHNS closure: **{acceptance['four_angle_chns_closure_passed']}**",
        "",
        "## Readiness",
        "",
        f"- Staged protocol complete: `{report['readiness']['chns_followup_protocol_complete']}`",
        f"- Four-angle production CHNS ready: `{report['readiness']['four_angle_production_chns_ready']}`",
        "",
        "Stage JSON traces and NPZ checkpoints are listed in `stage_files` and the `stage_artifacts` "
        "arrays; checkpoints contain phi, u, v, and time and are validated against the stage settings.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    jax.config.update("jax_enable_x64", True)
    report = run_audit(args.out, overwrite=args.overwrite)
    print(format_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
