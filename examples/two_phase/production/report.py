"""Strict JSON validation reports with lineage fields and atomic file replacement."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from production.config import VALIDATION_REPORT_SCHEMA_VERSION

REPORT_SCHEMA_VERSION = VALIDATION_REPORT_SCHEMA_VERSION
_REQUIRED_TOP = {
    "validation_report_schema_version",
    "physics_status",
    "contract_status",
    "repository",
    "runtime",
    "config",
    "benchmarks",
    "known_solver_blockers",
    "provisional_readiness_targets",
    "notes",
}
_REQUIRED_REPOSITORY = {"git_sha", "solver_contract_version", "phasefield_sha256", "validation_code_sha256"}
_REQUIRED_RUNTIME = {"python_version", "jax_version", "jax_backend", "devices", "platform"}
_REQUIRED_CONFIG = {"profile", "sha256", "path"}
_REQUIRED_BENCHMARKS = {"static_droplet", "contact_angle", "impact", "convergence"}


class ValidationReportError(ValueError):
    """Raised when a report is not valid under the machine-readable contract."""


def _is_sha(value: Any, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def _finite_value_errors(value: Any, path: str = "report") -> list[str]:
    errors: list[str] = []
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return errors
    if isinstance(value, float):
        if not math.isfinite(value):
            errors.append(f"non-finite number at {path}")
        return errors
    # Accept NumPy scalar values when the report builder is used outside the CLI.
    if hasattr(value, "item") and callable(value.item):
        try:
            return _finite_value_errors(value.item(), path)
        except (TypeError, ValueError):
            errors.append(f"unsupported scalar at {path}")
            return errors
    if isinstance(value, dict):
        for key, child in value.items():
            errors.extend(_finite_value_errors(child, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            errors.extend(_finite_value_errors(child, f"{path}[{index}]"))
    else:
        errors.append(f"unsupported value type {type(value).__name__} at {path}")
    return errors


def validate_report_schema(report: dict[str, Any]) -> list[str]:
    """Return all schema/strict-JSON errors, or an empty list for a valid report."""
    errors: list[str] = []
    if not isinstance(report, dict):
        return ["report must be a JSON object"]
    missing = sorted(_REQUIRED_TOP - set(report))
    errors.extend(f"missing top-level field: {key}" for key in missing)
    if report.get("validation_report_schema_version") != REPORT_SCHEMA_VERSION:
        errors.append(f"validation_report_schema_version must equal {REPORT_SCHEMA_VERSION}")
    # This framework captures the as-is baseline; it must not label the solver validated.
    if report.get("physics_status") != "BASELINE_ONLY":
        errors.append("physics_status must be BASELINE_ONLY for the L1A-1 framework")
    if report.get("contract_status") not in {"PASS", "FAIL"}:
        errors.append("contract_status must be PASS or FAIL")

    repository = report.get("repository")
    if not isinstance(repository, dict):
        errors.append("repository must be an object")
    else:
        errors.extend(f"missing repository field: {key}" for key in sorted(_REQUIRED_REPOSITORY - set(repository)))
        git_sha = repository.get("git_sha")
        if git_sha != "unknown" and not _is_sha(git_sha, 40):
            errors.append("repository.git_sha must be a 40-character lowercase git SHA or 'unknown'")
        for name in ("phasefield_sha256", "validation_code_sha256"):
            if not _is_sha(repository.get(name), 64):
                errors.append(f"repository.{name} must be a 64-character lowercase SHA-256")
        if repository.get("solver_contract_version") != 4:
            errors.append("repository.solver_contract_version must remain 4")

    runtime = report.get("runtime")
    if not isinstance(runtime, dict):
        errors.append("runtime must be an object")
    else:
        errors.extend(f"missing runtime field: {key}" for key in sorted(_REQUIRED_RUNTIME - set(runtime)))
        for key in ("python_version", "jax_version", "jax_backend", "platform"):
            if key in runtime and not isinstance(runtime[key], str):
                errors.append(f"runtime.{key} must be a string")
        if "devices" in runtime and not isinstance(runtime["devices"], list):
            errors.append("runtime.devices must be a list")

    config = report.get("config")
    if not isinstance(config, dict):
        errors.append("config must be an object")
    else:
        errors.extend(f"missing config field: {key}" for key in sorted(_REQUIRED_CONFIG - set(config)))
        if config.get("profile") not in {"ci", "baseline", "convergence"}:
            errors.append("config.profile is invalid")
        if not _is_sha(config.get("sha256"), 64):
            errors.append("config.sha256 must be a 64-character lowercase SHA-256")
        if "path" in config and not isinstance(config["path"], str):
            errors.append("config.path must be a string")

    benchmarks = report.get("benchmarks")
    if not isinstance(benchmarks, dict):
        errors.append("benchmarks must be an object")
    else:
        errors.extend(f"missing benchmark category: {key}" for key in sorted(_REQUIRED_BENCHMARKS - set(benchmarks)))
        for key in _REQUIRED_BENCHMARKS.intersection(benchmarks):
            if not isinstance(benchmarks[key], dict):
                errors.append(f"benchmarks.{key} must be an object")

    blockers = report.get("known_solver_blockers")
    if not isinstance(blockers, list):
        errors.append("known_solver_blockers must be a list")
    else:
        for index, blocker in enumerate(blockers):
            if not isinstance(blocker, dict):
                errors.append(f"known_solver_blockers[{index}] must be an object")
                continue
            required_blocker = {"id", "severity", "status", "description"}
            errors.extend(
                f"known_solver_blockers[{index}] missing field {key}" for key in sorted(required_blocker - set(blocker))
            )
            if blocker.get("severity") not in {"critical", "high", "medium", "low"}:
                errors.append(f"known_solver_blockers[{index}].severity is invalid")
            if blocker.get("status") not in {
                "open",
                "measurement_required",
                "confirmed_problem",
                "acceptable_for_next_stage",
            }:
                errors.append(f"known_solver_blockers[{index}].status is invalid")
    if not isinstance(report.get("provisional_readiness_targets"), dict):
        errors.append("provisional_readiness_targets must be an object")
    if not isinstance(report.get("notes"), list) or not all(isinstance(note, str) for note in report.get("notes", [])):
        errors.append("notes must be a list of strings")
    errors.extend(_finite_value_errors(report))
    # Also use the encoder itself as the authoritative strict-JSON check.
    try:
        json.dumps(report, allow_nan=False, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        errors.append(f"report is not strict JSON serializable: {exc}")
    return errors


def validate_benchmark_contract(benchmarks: dict[str, Any], expected: dict[str, int]) -> list[str]:
    """Validate benchmark record completeness, array shapes, ranges, and finite values."""
    errors: list[str] = []
    for name in ("static_droplet", "contact_angle", "impact", "convergence"):
        if name not in benchmarks:
            errors.append(f"missing benchmark category {name}")
    static = benchmarks.get("static_droplet", {})
    static_cases = static.get("cases", []) if isinstance(static, dict) else []
    if len(static_cases) != expected.get("static_droplet", 0):
        errors.append("static_droplet has missing benchmark records")
    static_fields = {
        "R",
        "N",
        "dx",
        "eps",
        "eps_over_dx",
        "Cn",
        "steps",
        "save_every",
        "mass_initial",
        "mass_final",
        "mass_relative_drift",
        "max_speed_final",
        "max_speed_peak",
        "kinetic_energy_final",
        "p_inside",
        "p_outside",
        "delta_p",
        "laplace_ratio",
    }
    for i, case in enumerate(static_cases):
        if not isinstance(case, dict):
            errors.append(f"static_droplet.cases[{i}] must be an object")
            continue
        if case.get("finite") is not True:
            errors.append(f"static_droplet.cases[{i}] did not produce finite metrics")
        if not static_fields.issubset(case):
            errors.append(f"static_droplet.cases[{i}] is missing required metrics")
        if case.get("pressure_diagnostic_method") != "projection_reconstructed":
            errors.append(f"static_droplet.cases[{i}] has unexpected pressure diagnostic method")
        series = case.get("time_series")
        if not isinstance(series, dict) or not {"time", "max_speed", "kinetic_energy"}.issubset(series):
            errors.append(f"static_droplet.cases[{i}] is missing time-series diagnostics")
        elif len({len(series[key]) for key in ("time", "max_speed", "kinetic_energy")}) != 1 or not series["time"]:
            errors.append(f"static_droplet.cases[{i}] time-series shapes differ or are empty")
        runtime = case.get("runtime")
        if not isinstance(runtime, dict) or not {"wall_seconds", "steps", "steps_per_second", "N", "dtype"}.issubset(
            runtime
        ):
            errors.append(f"static_droplet.cases[{i}] is missing runtime metadata")
    contact = benchmarks.get("contact_angle", {})
    contact_cases = contact.get("cases", []) if isinstance(contact, dict) else []
    if len(contact_cases) != expected.get("contact_angle", 0):
        errors.append("contact_angle has missing benchmark records")
    for i, case in enumerate(contact_cases):
        if not isinstance(case, dict):
            errors.append(f"contact_angle.cases[{i}] must be an object")
            continue
        measured = case.get("measured_deg")
        if case.get("finite") is not True or not isinstance(measured, (int, float)) or not 0.0 <= measured <= 180.0:
            errors.append(f"contact_angle.cases[{i}] angle is non-finite or outside [0, 180]")
        if not {"target_deg", "absolute_error_deg", "mass_relative_drift", "relaxation_steps"}.issubset(case):
            errors.append(f"contact_angle.cases[{i}] is missing required metrics")
        runtime = case.get("runtime")
        if not isinstance(runtime, dict) or not {"wall_seconds", "steps", "steps_per_second", "N", "dtype"}.issubset(
            runtime
        ):
            errors.append(f"contact_angle.cases[{i}] is missing runtime metadata")
    impact = benchmarks.get("impact", {})
    impact_cases = impact.get("cases", []) if isinstance(impact, dict) else []
    if len(impact_cases) != expected.get("impact", 0):
        errors.append("impact has missing benchmark records")
    required_impact_series = {
        "time",
        "mass",
        "total_phase_mass",
        "x_cm",
        "y_cm",
        "width",
        "beta",
        "g_0.5",
        "g_0.1",
        "top_height",
        "bottom_height",
        "max_speed",
        "contact_signal",
    }
    for i, case in enumerate(impact_cases):
        if not isinstance(case, dict):
            errors.append(f"impact.cases[{i}] must be an object")
            continue
        if case.get("finite") is not True:
            errors.append(f"impact.cases[{i}] did not produce finite metrics")
            continue
        required_impact_fields = {
            "We",
            "Re",
            "Oh_derived",
            "R",
            "D0",
            "theta_deg",
            "cos_theta",
            "eps",
            "eps_over_dx",
            "M",
            "rho_ratio",
            "nu_ratio",
            "dt",
            "N",
            "beta_max",
            "time_to_beta_max",
            "first_contact_time",
            "detachment_observed",
            "mass_drift",
        }
        if not required_impact_fields.issubset(case):
            errors.append(f"impact.cases[{i}] is missing required impact metadata")
        runtime = case.get("runtime")
        if not isinstance(runtime, dict) or not {"wall_seconds", "steps", "steps_per_second", "N", "dtype"}.issubset(
            runtime
        ):
            errors.append(f"impact.cases[{i}] is missing runtime metadata")
        series = case.get("time_series")
        if not isinstance(series, dict) or not required_impact_series.issubset(series):
            errors.append(f"impact.cases[{i}] is missing required time-series arrays")
            continue
        lengths = {len(series[key]) for key in required_impact_series}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
            errors.append(f"impact.cases[{i}] time-series arrays have invalid shapes")
        if not all(isinstance(item, bool) for item in series.get("contact_signal", [])):
            errors.append(f"impact.cases[{i}].contact_signal must contain booleans")
    convergence = benchmarks.get("convergence", {})
    for study_name, count in expected.items():
        if study_name.startswith("convergence:"):
            key = study_name.split(":", 1)[1]
            study = convergence.get(key)
            if not isinstance(study, dict) or len(study.get("points", [])) != count:
                errors.append(f"convergence.{key} has missing sweep points")
                continue
            for index, point in enumerate(study["points"]):
                metrics = point.get("metrics", {}) if isinstance(point, dict) else {}
                required_metrics = {
                    "laplace_ratio",
                    "mass_drift",
                    "spurious_current_peak",
                    "wall_seconds",
                    "dx",
                    "eps",
                    "eps_over_dx",
                    "Cn",
                }
                if not required_metrics.issubset(metrics):
                    errors.append(f"convergence.{key}.points[{index}] is missing required metrics")
                if not isinstance(point, dict) or not isinstance(point.get("parameters"), dict):
                    errors.append(f"convergence.{key}.points[{index}] is missing sweep parameters")
                elif "target_deg" in point["parameters"] and not isinstance(
                    metrics.get("contact_angle_deg"), (int, float)
                ):
                    errors.append(f"convergence.{key}.points[{index}] lacks its requested contact-angle result")
                elif "impact_steps" in point.get("parameters", {}) and not isinstance(
                    metrics.get("beta_max"), (int, float)
                ):
                    errors.append(f"convergence.{key}.points[{index}] lacks its requested beta_max")
    errors.extend(_finite_value_errors(benchmarks, "benchmarks"))
    return errors


def build_report(
    *,
    physics_status: str,
    contract_status: str,
    repository: dict[str, Any],
    runtime: dict[str, Any],
    config: dict[str, Any],
    benchmarks: dict[str, Any],
    known_solver_blockers: list[dict[str, Any]],
    provisional_readiness_targets: dict[str, Any],
    notes: list[str],
) -> dict[str, Any]:
    return {
        "validation_report_schema_version": REPORT_SCHEMA_VERSION,
        "physics_status": physics_status,
        "contract_status": contract_status,
        "repository": repository,
        "runtime": runtime,
        "config": config,
        "benchmarks": benchmarks,
        "known_solver_blockers": known_solver_blockers,
        "provisional_readiness_targets": provisional_readiness_targets,
        "notes": notes,
    }


def write_report_atomic(report: dict[str, Any], output_path: str | Path) -> None:
    """Write UTF-8 indented strict JSON through a sibling temporary file + replace."""
    errors = validate_report_schema(report)
    if errors:
        raise ValidationReportError("Report schema validation failed: " + "; ".join(errors))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            tmp_path = Path(stream.name)
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, destination)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise
