"""Command-line entry point for the L1A-1 two-phase validation suite.

Run from ``examples/two_phase`` with ``python -m production.run_validation``.
"""

from __future__ import annotations

import argparse
import json
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
                enforce_solid_phi=spec.get("enforce_solid_phi", True),
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


def _assessed_blockers(benchmarks: dict[str, Any]) -> list[dict[str, Any]]:
    """Attach evidence-based baseline status to measurement-dependent blockers only."""
    from production.validation import KNOWN_SOLVER_BLOCKERS, PROVISIONAL_READINESS_TARGETS

    blockers = [dict(item) for item in KNOWN_SOLVER_BLOCKERS]
    indexed = {item["id"]: item for item in blockers}
    angle_summary = benchmarks.get("contact_angle", {}).get("summary") or {}
    mae = angle_summary.get("mae_deg")
    max_error = angle_summary.get("max_absolute_error_deg")
    if mae is not None and max_error is not None:
        angle_blocker = indexed["W-CONTACT-ANGLE"]
        failed = (
            float(mae) > PROVISIONAL_READINESS_TARGETS["contact_angle_mae_deg"]
            or float(max_error) > PROVISIONAL_READINESS_TARGETS["contact_angle_max_error_deg"]
            or angle_summary.get("monotonic_target_to_measured") is not True
        )
        angle_blocker["status"] = "confirmed_problem" if failed else "acceptable_for_next_stage"
        angle_blocker["evidence"] = {
            "mae_deg": float(mae),
            "max_absolute_error_deg": float(max_error),
            "monotonic_target_to_measured": bool(angle_summary.get("monotonic_target_to_measured")),
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
            "physics or production readiness. Solver contract v5 (L1A-2a) corrects only the capillary-force "
            "sign (P-LAPLACE-SIGN); wetting, variable-density projection, the capillary denominator, "
            "timestep, viscosity and boundary blockers remain open."
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
