"""Command-line entry point for the L1A two-phase validation suite.

Run from ``examples/two_phase`` with ``python -m production.run_validation``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from production.config import (
    compute_validation_code_hash,
    config_sha256,
    get_git_sha,
    get_phasefield_sha256,
    get_runtime_info,
    load_config,
)
from production.report import (
    build_report,
    validate_benchmark_contract,
    validate_report_schema,
    write_report_atomic,
)


def _record_failure(label: str, exc: Exception) -> dict[str, Any]:
    return {"error": f"{type(exc).__name__}: {exc}", "finite": False, "benchmark": label}


def _run_convergence(config: dict[str, Any], dtype: str) -> tuple[dict[str, Any], dict[str, int]]:
    from production.convergence import run_grid_refinement_study, run_interface_thickness_study
    from production.validation import run_convergence_point

    output: dict[str, Any] = {}
    counts: dict[str, int] = {}
    block = config.get("convergence", {})
    metric = lambda case: run_convergence_point(case, dtype=dtype)
    if "grid_refinement" in block:
        sweep = block["grid_refinement"]
        study = run_grid_refinement_study(
            sweep["base_case"],
            sweep["N_values"],
            metric,
            eps_factor=sweep.get("eps_factor"),
            eps_mode=sweep.get("eps_mode", "coupled"),
            fixed_eps=sweep.get("fixed_eps"),
        )
        output["grid_refinement"] = study.to_dict()
        counts["convergence:grid_refinement"] = len(study.points)
    if "interface_thickness" in block:
        sweep = block["interface_thickness"]
        study = run_interface_thickness_study(
            sweep["base_case"],
            sweep["eps_factors"],
            metric,
            N=sweep["N"],
        )
        output["interface_thickness"] = study.to_dict()
        counts["convergence:interface_thickness"] = len(study.points)
    return output, counts


def _run_benchmarks(config, dtype: str) -> tuple[dict[str, Any], dict[str, int], list[str]]:
    from production.validation import run_contact_angle_suite, run_impact_suite, run_static_droplet_suite

    definitions = config.benchmarks
    output: dict[str, Any] = {
        "static_droplet": {"summary": None, "cases": []},
        "contact_angle": {"summary": None, "cases": []},
        "impact": {"cases": []},
        "convergence": {},
    }
    expected: dict[str, int] = {"static_droplet": 0, "contact_angle": 0, "impact": 0}
    errors: list[str] = []

    if "static_droplet" in definitions:
        spec = definitions["static_droplet"]
        expected["static_droplet"] = len(spec["radii"])
        try:
            output["static_droplet"] = run_static_droplet_suite(
                radii=spec["radii"],
                N=spec["N"],
                steps=spec["steps"],
                We=spec.get("We", 100.0),
                Re=spec.get("Re", 200.0),
                eps_factor=spec.get("eps_factor", 1.5),
                save_every=spec.get("save_every", 100),
                dt=spec.get("dt", 2e-3),
                dtype=dtype,
            )
        except Exception as exc:
            errors.append(f"static_droplet suite: {type(exc).__name__}: {exc}")
            output["static_droplet"] = {"summary": None, "cases": [_record_failure("static_droplet", exc)]}

    if "contact_angle" in definitions:
        spec = definitions["contact_angle"]
        expected["contact_angle"] = len(spec["targets"])
        try:
            output["contact_angle"] = run_contact_angle_suite(
                targets=spec["targets"],
                N=spec["N"],
                relaxation_steps=spec["relaxation_steps"],
                We=spec.get("We", 100.0),
                Re=spec.get("Re", 200.0),
                eps_factor=spec.get("eps_factor", 1.5),
                wall_energy_amp=spec.get("wall_energy_amp", 5.0),
                R=spec.get("R", 1.1),
                dt=spec.get("dt", 4e-3),
                dtype=dtype,
                wetting_model=spec.get("wetting_model", "surface_energy"),
                phase_boundary_model=spec.get("phase_boundary_model", "impermeable_flux"),
                enforce_solid_phi=spec.get("enforce_solid_phi", False),
                sample_every=spec.get("sample_every", 200),
                angle_tol_deg=spec.get("angle_tol_deg", 0.25),
                speed_tol=spec.get("speed_tol", 5e-4),
                windows=spec.get("windows", 3),
                max_steps=spec.get("max_steps"),
            )
        except Exception as exc:
            errors.append(f"contact_angle suite: {type(exc).__name__}: {exc}")
            output["contact_angle"] = {"summary": None, "cases": [_record_failure("contact_angle", exc)]}

    if "impact" in definitions:
        spec = definitions["impact"]
        expected["impact"] = len(spec["cases"])
        cases = [dict(wall_height=spec.get("wall_height", 0.25), **case) for case in spec["cases"]]
        try:
            output["impact"] = run_impact_suite(
                cases,
                N=spec["N"],
                eps_factor=spec.get("eps_factor", 1.5),
                steps=spec["steps"],
                save_every=spec.get("save_every", 10),
                dt=spec.get("dt", 2e-3),
                dtype=dtype,
            )
        except Exception as exc:
            errors.append(f"impact suite: {type(exc).__name__}: {exc}")
            output["impact"] = {"cases": [_record_failure("impact", exc)]}

    if "convergence" in definitions:
        try:
            output["convergence"], convergence_counts = _run_convergence(definitions, dtype)
            expected.update(convergence_counts)
        except Exception as exc:
            errors.append(f"convergence suite: {type(exc).__name__}: {exc}")
            output["convergence"] = {"error": f"{type(exc).__name__}: {exc}"}
            for name, sweep in definitions["convergence"].items():
                expected[f"convergence:{name}"] = len(sweep.get("N_values", sweep.get("eps_factors", [])))
    return output, expected, errors


def _strict_target_failures(benchmarks: dict[str, Any]) -> list[str]:
    """Compare the frozen provisional readiness targets; never changes contract PASS."""
    from production.validation import PROVISIONAL_READINESS_TARGETS

    targets = PROVISIONAL_READINESS_TARGETS
    failures: list[str] = []
    mass_drifts = []
    for section, field in (
        ("static_droplet", "mass_relative_drift"),
        ("contact_angle", "mass_relative_drift"),
        ("impact", "mass_drift"),
    ):
        for case in benchmarks[section].get("cases", []):
            value = case.get(field)
            if case.get("finite") is True and isinstance(value, (int, float)):
                mass_drifts.append(float(value))
    if mass_drifts:
        drift = max(mass_drifts)
        if drift > targets["mass_relative_drift"]:
            failures.append(f"mass relative drift {drift:.6g} > {targets['mass_relative_drift']}")

    static_cases = [c for c in benchmarks["static_droplet"].get("cases", []) if c.get("finite") is True]
    if static_cases:
        laplace = max(abs(float(c["laplace_ratio"]) - 1.0) for c in static_cases)
        if laplace > targets["laplace_relative_error"]:
            failures.append(f"Laplace relative error {laplace:.6g} > {targets['laplace_relative_error']}")
    contact = benchmarks["contact_angle"].get("summary") or {}
    mae = contact.get("mae_deg")
    max_error = contact.get("max_absolute_error_deg")
    if mae is not None and float(mae) > targets["contact_angle_mae_deg"]:
        failures.append(f"contact-angle MAE {float(mae):.6g} deg > {targets['contact_angle_mae_deg']} deg")
    if max_error is not None and float(max_error) > targets["contact_angle_max_error_deg"]:
        failures.append(
            f"contact-angle max error {float(max_error):.6g} deg > {targets['contact_angle_max_error_deg']} deg"
        )
    accepted, contact_evidence = _contact_angle_acceptance(benchmarks)
    if not accepted:
        failures.append(
            "four-angle equilibrium contact acceptance not met: "
            f"converged={contact_evidence['all_converged']}, monotonic={contact_evidence['monotonic']}, "
            f"mass_ok={contact_evidence['mass_ok']}, v7_boundary_ok={contact_evidence['v7_boundary_model_ok']}"
        )
    for study_name, study in benchmarks["convergence"].items():
        changes = study.get("relative_changes", {}) if isinstance(study, dict) else {}
        for metric, values in changes.items():
            if metric in {"laplace_ratio", "beta_max", "contact_angle_deg", "mass_drift"}:
                if values and max(values) > targets["key_observable_refinement_change"]:
                    failures.append(
                        f"{study_name}.{metric} refinement change {max(values):.6g} > "
                        f"{targets['key_observable_refinement_change']}"
                    )
    return failures


def _contact_angle_acceptance(benchmarks: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Apply the frozen four-angle acceptance criteria; no partial-matrix closure."""
    from production.validation import PROVISIONAL_READINESS_TARGETS

    expected = (60.0, 90.0, 120.0, 150.0)
    records = benchmarks.get("contact_angle", {}).get("cases", [])
    by_target = {
        round(float(case["target_deg"]), 8): case
        for case in records
        if isinstance(case, dict) and isinstance(case.get("target_deg"), (int, float))
    }
    complete = set(by_target) == {round(value, 8) for value in expected}
    rows = [by_target.get(round(value, 8), {}) for value in expected]
    all_finite = complete and all(row.get("finite") is True for row in rows)
    all_converged = all_finite and all(row.get("converged") is True for row in rows)
    measured = [row.get("measured_deg") for row in rows]
    finite_angles = all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in measured)
    errors = [
        abs(float(row["measured_deg"]) - target)
        for target, row in zip(expected, rows)
        if isinstance(row.get("measured_deg"), (int, float)) and math.isfinite(float(row["measured_deg"]))
    ]
    partial_mae = float(sum(errors) / len(errors)) if errors else None
    partial_max_error = float(max(errors)) if errors else None
    complete_equilibrium_matrix = bool(complete and all_finite and all_converged and finite_angles and len(errors) == 4)
    mae = partial_mae if complete_equilibrium_matrix else None
    max_error = partial_max_error if complete_equilibrium_matrix else None
    monotonic = finite_angles and all(float(b) >= float(a) for a, b in zip(measured, measured[1:]))
    theta90_error = (
        abs(float(rows[1]["measured_deg"]) - 90.0)
        if len(rows) > 1
        and isinstance(rows[1].get("measured_deg"), (int, float))
        and math.isfinite(float(rows[1]["measured_deg"]))
        else None
    )
    fluid_drifts = [
        float(row["mass_relative_drift"])
        for row in rows
        if isinstance(row.get("mass_relative_drift"), (int, float)) and math.isfinite(float(row["mass_relative_drift"]))
    ]
    total_drifts = [
        float(row["total_mass_relative_drift"])
        for row in rows
        if isinstance(row.get("total_mass_relative_drift"), (int, float))
        and math.isfinite(float(row["total_mass_relative_drift"]))
    ]
    max_fluid_drift = max(fluid_drifts) if len(fluid_drifts) == len(rows) and rows else None
    max_total_drift = max(total_drifts) if len(total_drifts) == len(rows) and rows else None
    mass_ok = (
        max_fluid_drift is not None
        and max_total_drift is not None
        and max_fluid_drift <= 1.0e-3
        and max_total_drift <= 1.0e-3
    )
    v7_boundary_ok = (
        all(
            row.get("phase_boundary_model") == "impermeable_flux" and row.get("enforce_solid_phi") is False
            for row in rows
        )
        if complete
        else False
    )
    angle_ok = (
        mae is not None
        and max_error is not None
        and mae <= PROVISIONAL_READINESS_TARGETS["contact_angle_mae_deg"]
        and max_error <= PROVISIONAL_READINESS_TARGETS["contact_angle_max_error_deg"]
        and theta90_error is not None
        and theta90_error <= 3.0
        and monotonic
    )
    accepted = bool(complete and all_finite and all_converged and angle_ok and mass_ok and v7_boundary_ok)
    return accepted, {
        "required_targets_deg": list(expected),
        "case_count": len(records),
        "complete_target_set": bool(complete),
        "all_finite": bool(all_finite),
        "all_converged": bool(all_converged),
        "measured_angles_deg": measured,
        "mae_deg": mae,
        "max_absolute_error_deg": max_error,
        "converged_subset_mae_deg": partial_mae,
        "converged_subset_max_error_deg": partial_max_error,
        "monotonic": bool(monotonic),
        "theta_90_abs_error_deg": theta90_error,
        "max_fluid_mass_drift": max_fluid_drift,
        "max_total_mass_drift": max_total_drift,
        "mass_ok": bool(mass_ok),
        "v7_boundary_model_ok": bool(v7_boundary_ok),
        "accepted": accepted,
    }


def _assessed_blockers(benchmarks: dict[str, Any]) -> list[dict[str, Any]]:
    """Attach evidence-based status; never close wetting on partial/non-equilibrium data."""
    from production.validation import KNOWN_SOLVER_BLOCKERS

    blockers = [dict(item) for item in KNOWN_SOLVER_BLOCKERS]
    indexed = {item["id"]: item for item in blockers}
    accepted, angle_evidence = _contact_angle_acceptance(benchmarks)
    angle_cases = benchmarks.get("contact_angle", {}).get("cases", [])
    all_measured = bool(
        angle_evidence["complete_target_set"] and angle_evidence["all_finite"] and angle_evidence["all_converged"]
    )
    if accepted:
        indexed["W-CONTACT-ANGLE"]["status"] = "resolved_in_contract_v7"
    elif all_measured:
        indexed["W-CONTACT-ANGLE"]["status"] = "confirmed_problem"
    indexed["W-CONTACT-ANGLE"]["evidence"] = angle_evidence

    neutral = next(
        (case for case in angle_cases if isinstance(case, dict) and float(case.get("target_deg", -1.0)) == 90.0),
        {},
    )
    samples = neutral.get("samples", [])
    last_sample = samples[-1] if samples and isinstance(samples[-1], dict) else {}
    neutral_ok = (
        accepted
        and neutral.get("finite") is True
        and neutral.get("converged") is True
        and neutral.get("phase_boundary_model") == "impermeable_flux"
        and neutral.get("enforce_solid_phi") is False
        and isinstance(neutral.get("measured_deg"), (int, float))
        and abs(float(neutral["measured_deg"]) - 90.0) <= 3.0
        and float(neutral.get("mass_relative_drift", math.inf)) <= 1.0e-3
        and float(neutral.get("total_mass_relative_drift", math.inf)) <= 1.0e-3
        and float(neutral.get("max_solid_liquid_fraction", math.inf)) <= 1.0e-6
        and float(last_sample.get("max_speed", math.inf)) <= 5.0e-4
    )
    indexed["P-SOLID-PIN"]["status"] = "resolved_in_contract_v7" if neutral_ok else "confirmed_problem"
    indexed["P-SOLID-PIN"]["evidence"] = {
        "neutral_90_case_present": bool(neutral),
        "neutral_case_converged": neutral.get("converged") is True,
        "phase_boundary_model": neutral.get("phase_boundary_model"),
        "enforce_solid_phi": neutral.get("enforce_solid_phi"),
        "measured_angle_deg": neutral.get("measured_deg"),
        "final_max_speed": last_sample.get("max_speed"),
        "fluid_mass_drift": neutral.get("mass_relative_drift"),
        "total_mass_drift": neutral.get("total_mass_relative_drift"),
        "max_solid_phase_fraction": neutral.get("max_solid_liquid_fraction"),
        "four_angle_acceptance_passed": bool(accepted),
        "closure_criteria_passed": bool(neutral_ok),
    }

    static = benchmarks.get("static_droplet", {})
    static_records = static.get("cases", [])
    static_cases = [
        case
        for case in static_records
        if case.get("finite") is True and isinstance(case.get("laplace_ratio"), (int, float))
    ]
    if static_cases:
        status, evidence = _laplace_sign_assessment(static_cases, len(static_records), static.get("summary"))
        indexed["P-LAPLACE-SIGN"]["status"] = status
        indexed["P-LAPLACE-SIGN"]["evidence"] = evidence
    return blockers


def _laplace_sign_assessment(
    static_cases: list[dict[str, Any]], record_count: int, summary: dict[str, Any] | None
) -> tuple[str, dict[str, Any]]:
    """Evidence-gated status of P-LAPLACE-SIGN (never closed on a hunch or on too little data).

    * any ratio <= 0                         -> ``confirmed_problem`` (sign still wrong)
    * fewer than ``min_radii`` finite radii  -> ``measurement_required`` (scaling not established)
    * positive but 1/R scaling fails         -> ``confirmed_problem``
    * all positive and 1/R scaling holds     -> ``resolved_in_contract_v5``

    The 5 % magnitude goal is *reported* (``provisional_laplace_target_met``) but does not gate this
    sign blocker; small-radius interface sensitivity belongs to the convergence study.
    """
    from production.validation import LAPLACE_SIGN_CLOSURE_CRITERIA as criteria
    from production.validation import PROVISIONAL_READINESS_TARGETS

    ratios = [float(case["laplace_ratio"]) for case in static_cases]
    fit = summary or {}
    r_squared, slope = fit.get("r_squared"), fit.get("slope_delta_p_vs_inv_R")
    all_positive = all(ratio > 0.0 for ratio in ratios)
    complete = len(static_cases) == record_count and len(static_cases) >= int(criteria["min_radii"])
    scaling_correct = (
        complete
        and isinstance(r_squared, (int, float))
        and isinstance(slope, (int, float))
        and float(r_squared) >= float(criteria["min_r_squared"])
        and float(slope) > 0.0
    )
    max_error = max(abs(ratio - 1.0) for ratio in ratios)
    if not all_positive:
        status = "confirmed_problem"
    elif not complete:
        status = "measurement_required"
    elif not scaling_correct:
        status = "confirmed_problem"
    else:
        status = "resolved_in_contract_v5"
    evidence = {
        "laplace_ratio_min": min(ratios),
        "laplace_ratio_max": max(ratios),
        "case_count": len(ratios),
        "expected_case_count": int(record_count),
        "all_ratios_positive": bool(all_positive),
        "r_squared_delta_p_vs_inv_R": None if r_squared is None else float(r_squared),
        "slope_delta_p_vs_inv_R": None if slope is None else float(slope),
        "scaling_correct": bool(scaling_correct),
        "max_abs_error_from_ratio_1": float(max_error),
        "provisional_laplace_target_met": bool(max_error <= PROVISIONAL_READINESS_TARGETS["laplace_relative_error"]),
        "closure_criteria": dict(criteria),
        "pressure_diagnostic_method": "projection_reconstructed",
        "solver_contract_version": pf_contract_version(),
    }
    return status, evidence


def _config_path_for_report(path: str) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(candidate.resolve())


def run_validation(
    config_path: str,
    output_dir: str | None = None,
    *,
    strict_targets: bool = False,
    overwrite: bool = False,
) -> int:
    """Execute configured benchmarks, always emitting a report after benchmark failures."""
    config = load_config(config_path)
    target = output_dir or config.output["directory"]
    out_path = Path(target)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {out_path}; pass --overwrite to reuse it")
    if out_path.exists() and not out_path.is_dir():
        raise NotADirectoryError(f"Output path exists and is not a directory: {out_path}")
    out_path.mkdir(parents=True, exist_ok=True)

    # Select the requested CPU profile before JAX is initialized. Float64 is
    # opt-in; the recommended configs use float32, matching the existing solver.
    if config.profile == "ci":
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax

    if config.dtype == "float64":
        jax.config.update("jax_enable_x64", True)

    from production.validation import PROVISIONAL_READINESS_TARGETS

    benchmark_results, expected, execution_errors = _run_benchmarks(config, config.dtype)
    contract_errors = execution_errors + validate_benchmark_contract(benchmark_results, expected)
    git_sha = get_git_sha()
    notes = [
        (
            "L1A physics status is BASELINE_ONLY; a green contract does not imply validated "
            "physics or production readiness. Solver contract v7 adds conservative impermeable phase-face "
            "fluxes, a matrix-free fail-closed CH solve and the natural Young boundary condition. "
            "P-VARDENS-PROJ, P-CAP-RHO, N-DT, P-VARVISC and the momentum/pressure BC-Y-PERIODIC "
            "blocker are unchanged; contact-angle acceptance remains evidence-gated."
        ),
        (
            "The v7 phase path uses phase_boundary_model='impermeable_flux'; it does not call the "
            "post-step mass redistribution. phase_boundary_model='projection_legacy' exists only for "
            "contract-v6 reproduction and diagnostics."
        ),
        (
            "pressure_field() is a projection-reconstructed diagnostic, not an independently "
            "evolved thermodynamic pressure."
        ),
        (
            "Laplace comparison is a projection-pressure diagnostic: delta_p = p_liquid - p_gas "
            "(phi = 1 liquid) and laplace_ratio = delta_p * R * We, expected +1 for a 2-D circle."
        ),
        (
            "Impact contact signal is diagnostic: g_0.5 <= 1.5*dx; g_0.1 is also recorded to expose "
            "diffuse-interface gap sensitivity."
        ),
        (
            "Static max_speed_peak tracks every solver step; static max_speed(t) and kinetic_energy(t) "
            "are saved at configured cadence."
        ),
    ]
    if git_sha == "unknown":
        notes.append("WARNING: git SHA is unknown; this run is not pinned to a repository commit.")
    for impact_case in benchmark_results["impact"].get("cases", []):
        if impact_case.get("finite") is True and impact_case.get("first_contact_time") is None:
            notes.append(
                f"Impact case {impact_case.get('case_name', 'unknown')}: no g_0.5 contact event "
                "was observed during the configured trajectory."
            )
    if contract_errors:
        notes.extend(f"CONTRACT FAILURE: {message}" for message in contract_errors)
    target_failures = _strict_target_failures(benchmark_results) if strict_targets else []
    if target_failures:
        notes.extend(f"STRICT PHYSICS TARGET NOT MET: {message}" for message in target_failures)

    report = build_report(
        physics_status="BASELINE_ONLY",
        contract_status="FAIL" if contract_errors else "PASS",
        repository={
            "git_sha": git_sha,
            "solver_contract_version": int(pf_contract_version()),
            "phasefield_sha256": get_phasefield_sha256(),
            "validation_code_sha256": compute_validation_code_hash(),
        },
        runtime=get_runtime_info(),
        config={
            "profile": config.profile,
            "sha256": config_sha256(config),
            "path": _config_path_for_report(config_path),
        },
        benchmarks=benchmark_results,
        known_solver_blockers=_assessed_blockers(benchmark_results),
        provisional_readiness_targets=PROVISIONAL_READINESS_TARGETS,
        notes=notes,
    )
    schema_errors = validate_report_schema(report)
    if schema_errors:
        raise RuntimeError("Generated report violated its own schema: " + "; ".join(schema_errors))
    report_path = out_path / "report.json"
    write_report_atomic(report, report_path)
    print(f"Report: {report_path}")
    print(f"Physics status: {report['physics_status']}; contract status: {report['contract_status']}")
    if target_failures:
        print("Strict physics targets not met: " + "; ".join(target_failures))
    return 1 if contract_errors else (2 if target_failures else 0)


def pf_contract_version() -> int:
    import phasefield as pf

    return int(pf.SOLVER_CONTRACT_VERSION)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="L1A-1 JAX two-phase production validation")
    parser.add_argument("--config", required=True, help="Path to a JSON validation profile")
    parser.add_argument("--out", default=None, help="Output directory (defaults to output.directory in config)")
    parser.add_argument(
        "--strict-targets", action="store_true", help="Return nonzero if provisional physics targets are missed"
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow an existing output directory")
    args = parser.parse_args(argv)
    try:
        return run_validation(args.config, args.out, strict_targets=args.strict_targets, overwrite=args.overwrite)
    except (ValueError, FileExistsError, NotADirectoryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
