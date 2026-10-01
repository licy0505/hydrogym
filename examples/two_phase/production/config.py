"""Fail-closed configuration loading and deterministic validation fingerprints."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

VALIDATION_REPORT_SCHEMA_VERSION = 1
VALID_PROFILES = {"ci", "baseline", "convergence"}
VALID_DTYPES = {"float32", "float64"}

_TOP_LEVEL_KEYS = {"profile", "dtype", "benchmarks", "output"}
_BENCHMARK_KEYS = {"static_droplet", "contact_angle", "impact", "convergence"}
_OUTPUT_KEYS = {"directory", "formats"}
_BLOCK_KEYS = {
    "static_droplet": {"radii", "N", "steps", "We", "Re", "eps_factor", "dt", "save_every"},
    "contact_angle": {
        "targets",
        "N",
        "relaxation_steps",
        "max_steps",
        "sample_every",
        "angle_tol_deg",
        "speed_tol",
        "windows",
        "wetting_model",
        "phase_boundary_model",
        "enforce_solid_phi",
        "We",
        "Re",
        "eps_factor",
        "wall_energy_amp",
        "R",
        "dt",
    },
    "impact": {"cases", "N", "steps", "save_every", "We", "Re", "eps_factor", "dt", "wall_height"},
    "convergence": {"grid_refinement", "interface_thickness"},
}
_IMPACT_CASE_KEYS = {
    "name",
    "We",
    "Re",
    "R",
    "cos_theta",
    "impact_gap",
    "u_impact",
    "x0",
    "velocity_mode",
    "wetting_model",
    "phase_boundary_model",
    "enforce_solid_phi",
}
_SWEEP_KEYS = {"base_case", "N_values", "eps_factor", "eps_mode", "fixed_eps", "N", "eps_factors"}
_CASE_KEYS = {
    "R",
    "We",
    "Re",
    "eps_factor",
    "dt",
    "save_every",
    "wall_height",
    "target_deg",
    "relaxation_steps",
    "impact_steps",
    "steps",
    "cos_theta",
    "impact_gap",
    "u_impact",
    "velocity_mode",
    "wetting_model",
    "phase_boundary_model",
    "enforce_solid_phi",
}


@dataclass(frozen=True)
class ValidationConfig:
    """Top-level validation profile. Benchmark blocks remain JSON-like mappings."""

    profile: str
    dtype: str
    benchmarks: dict[str, Any]
    output: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _unknown_keys(obj: Mapping[str, Any], allowed: set[str], where: str, errors: list[str]) -> None:
    for key in sorted(set(obj) - allowed):
        errors.append(f"Unknown key at {where}: {key}")


def _positive_number(value: Any, label: str, errors: list[str], *, allow_zero: bool = False) -> None:
    import math

    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        errors.append(f"{label} must be a finite number")
    elif value < 0 if allow_zero else value <= 0:
        errors.append(f"{label} must be {'non-negative' if allow_zero else 'positive'}")


def _upper_bound(value: Any, label: str, maximum: float, errors: list[str]) -> None:
    import math

    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value > maximum
    ):
        errors.append(f"{label} must be <= {maximum:g}")


def _positive_int(value: Any, label: str, errors: list[str], *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{label} must be an integer")
    elif (value < 0) if allow_zero else (value <= 0):
        errors.append(f"{label} must be {'non-negative' if allow_zero else 'positive'}")


def _grid_size(value: Any, label: str, errors: list[str]) -> None:
    _positive_int(value, label, errors)
    if isinstance(value, int) and not isinstance(value, bool) and not 8 <= value <= 4096:
        errors.append(f"{label} must be between 8 and 4096 cells")


def _step_count(value: Any, label: str, errors: list[str], *, allow_zero: bool = False) -> None:
    _positive_int(value, label, errors, allow_zero=allow_zero)
    if isinstance(value, int) and not isinstance(value, bool) and value > 10_000_000:
        errors.append(f"{label} exceeds the safety limit of 10000000 steps")


def _validate_case(case: Any, label: str, errors: list[str]) -> None:
    if not isinstance(case, dict):
        errors.append(f"{label} must be an object")
        return
    _unknown_keys(case, _CASE_KEYS, label, errors)
    for key in ("R", "We", "Re", "eps_factor", "dt", "wall_height", "u_impact"):
        if key in case:
            _positive_number(case[key], f"{label}.{key}", errors)
    for key, bound in (("R", 1.5), ("eps_factor", 10.0), ("We", 1.0e6), ("Re", 1.0e6), ("dt", 1.0), ("u_impact", 10.0)):
        if key in case:
            _upper_bound(case[key], f"{label}.{key}", bound, errors)
    if "impact_gap" in case:
        _positive_number(case["impact_gap"], f"{label}.impact_gap", errors, allow_zero=True)
    for key in ("save_every", "steps", "relaxation_steps", "impact_steps"):
        if key in case:
            _step_count(case[key], f"{label}.{key}", errors, allow_zero=(key == "relaxation_steps"))
    if "target_deg" in case and (
        isinstance(case["target_deg"], bool)
        or not isinstance(case["target_deg"], (int, float))
        or not 0 <= case["target_deg"] <= 180
    ):
        errors.append(f"{label}.target_deg must be in [0, 180]")
    if "cos_theta" in case and (
        isinstance(case["cos_theta"], bool)
        or not isinstance(case["cos_theta"], (int, float))
        or not -1 <= case["cos_theta"] <= 1
    ):
        errors.append(f"{label}.cos_theta must be in [-1, 1]")
    if "velocity_mode" in case and case["velocity_mode"] not in {"uniform", "streamfunction"}:
        errors.append(f"{label}.velocity_mode must be 'uniform' or 'streamfunction'")
    if "wetting_model" in case and case["wetting_model"] not in {
        "surface_energy",
        "surface_energy_volume_v6",
        "legacy_affinity",
        "none",
    }:
        errors.append(f"{label}.wetting_model is unknown")
    if "phase_boundary_model" in case and case["phase_boundary_model"] not in {
        "impermeable_flux",
        "projection_legacy",
    }:
        errors.append(f"{label}.phase_boundary_model must be impermeable_flux or projection_legacy")
    if (
        case.get("enforce_solid_phi", False)
        and case.get("phase_boundary_model", "impermeable_flux") != "projection_legacy"
    ):
        errors.append(f"{label}.enforce_solid_phi is legacy-only and requires phase_boundary_model='projection_legacy'")


def _validate_sweep(sweep: Any, label: str, mode: str, errors: list[str]) -> None:
    if not isinstance(sweep, dict):
        errors.append(f"benchmarks.convergence.{label} must be an object")
        return
    _unknown_keys(sweep, _SWEEP_KEYS, f"benchmarks.convergence.{label}", errors)
    if not isinstance(sweep.get("base_case"), dict):
        errors.append(f"benchmarks.convergence.{label}.base_case is required and must be an object")
    else:
        _validate_case(sweep["base_case"], f"benchmarks.convergence.{label}.base_case", errors)
    if mode == "grid_refinement":
        values = sweep.get("N_values")
        if not isinstance(values, list) or len(values) < 2:
            errors.append(f"{label}.N_values must contain at least two resolutions")
        else:
            for i, n in enumerate(values):
                _grid_size(n, f"{label}.N_values[{i}]", errors)
            if all(isinstance(n, int) and not isinstance(n, bool) for n in values) and values != sorted(set(values)):
                errors.append(f"{label}.N_values must be strictly increasing")
        eps_mode = sweep.get("eps_mode", "coupled")
        if eps_mode not in {"coupled", "fixed_physical"}:
            errors.append(f"{label}.eps_mode must be 'coupled' or 'fixed_physical'")
        if "eps_factor" in sweep:
            _positive_number(sweep["eps_factor"], f"{label}.eps_factor", errors)
        if eps_mode == "fixed_physical" and "fixed_eps" not in sweep:
            errors.append(f"{label}.fixed_eps is required for eps_mode='fixed_physical'")
        if "fixed_eps" in sweep:
            _positive_number(sweep["fixed_eps"], f"{label}.fixed_eps", errors)
    else:
        _grid_size(sweep.get("N"), f"{label}.N", errors)
        values = sweep.get("eps_factors")
        if not isinstance(values, list) or len(values) < 2:
            errors.append(f"{label}.eps_factors must contain at least two values")
        else:
            for i, value in enumerate(values):
                _positive_number(value, f"{label}.eps_factors[{i}]", errors)


def validate_config_data(data: Any) -> list[str]:
    """Validate all supported configuration fields; unknown keys fail closed."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["config root must be a JSON object"]
    _unknown_keys(data, _TOP_LEVEL_KEYS, "config", errors)
    for key in sorted(_TOP_LEVEL_KEYS - set(data)):
        errors.append(f"Missing required field: {key}")

    if data.get("profile") not in VALID_PROFILES:
        errors.append(f"profile must be one of {sorted(VALID_PROFILES)}")
    if data.get("dtype") not in VALID_DTYPES:
        errors.append(f"dtype must be one of {sorted(VALID_DTYPES)}")

    benchmarks = data.get("benchmarks")
    if not isinstance(benchmarks, dict):
        errors.append("benchmarks must be an object")
    else:
        _unknown_keys(benchmarks, _BENCHMARK_KEYS, "benchmarks", errors)
        if not benchmarks:
            errors.append("benchmarks must contain at least one benchmark")
        for name, block in benchmarks.items():
            if not isinstance(block, dict):
                errors.append(f"benchmarks.{name} must be an object")
                continue
            _unknown_keys(block, _BLOCK_KEYS.get(name, set()), f"benchmarks.{name}", errors)
            if name == "static_droplet":
                radii = block.get("radii")
                if not isinstance(radii, list) or not radii:
                    errors.append("benchmarks.static_droplet.radii must be a non-empty list")
                else:
                    for i, value in enumerate(radii):
                        label = f"static_droplet.radii[{i}]"
                        _positive_number(value, label, errors)
                        _upper_bound(value, label, 1.5, errors)
                _grid_size(block.get("N"), "static_droplet.N", errors)
                _step_count(block.get("steps"), "static_droplet.steps", errors)
                _step_count(block.get("save_every"), "static_droplet.save_every", errors)
                for key in ("We", "Re", "eps_factor", "dt"):
                    if key in block:
                        _positive_number(block[key], f"static_droplet.{key}", errors)
                        bound = 10.0 if key == "eps_factor" else (1.0 if key == "dt" else 1.0e6)
                        _upper_bound(block[key], f"static_droplet.{key}", bound, errors)
            elif name == "contact_angle":
                targets = block.get("targets")
                if not isinstance(targets, list) or not targets:
                    errors.append("benchmarks.contact_angle.targets must be a non-empty list")
                else:
                    for i, value in enumerate(targets):
                        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 180:
                            errors.append(f"contact_angle.targets[{i}] must be in [0, 180]")
                _grid_size(block.get("N"), "contact_angle.N", errors)
                _step_count(block.get("relaxation_steps"), "contact_angle.relaxation_steps", errors, allow_zero=True)
                if "max_steps" in block:
                    _step_count(block["max_steps"], "contact_angle.max_steps", errors)
                sample_every = block.get("sample_every")
                if sample_every is not None and (
                    isinstance(sample_every, bool) or not isinstance(sample_every, int) or sample_every < 1
                ):
                    errors.append("contact_angle.sample_every must be a positive integer")
                windows = block.get("windows")
                if windows is not None and (isinstance(windows, bool) or not isinstance(windows, int) or windows < 1):
                    errors.append("contact_angle.windows must be a positive integer")
                for key in ("angle_tol_deg", "speed_tol"):
                    if key in block:
                        _positive_number(block[key], f"contact_angle.{key}", errors)
                        _upper_bound(
                            block[key], f"contact_angle.{key}", 180.0 if key == "angle_tol_deg" else 1e6, errors
                        )
                if "wetting_model" in block and block["wetting_model"] not in {
                    "surface_energy",
                    "surface_energy_volume_v6",
                    "legacy_affinity",
                    "none",
                }:
                    errors.append(
                        "contact_angle.wetting_model must be one of surface_energy, "
                        "surface_energy_volume_v6, legacy_affinity, none"
                    )
                if "phase_boundary_model" in block and block["phase_boundary_model"] not in {
                    "impermeable_flux",
                    "projection_legacy",
                }:
                    errors.append("contact_angle.phase_boundary_model must be impermeable_flux or projection_legacy")
                if "enforce_solid_phi" in block and not isinstance(block["enforce_solid_phi"], bool):
                    errors.append("contact_angle.enforce_solid_phi must be a boolean")
                if (
                    block.get("enforce_solid_phi", False)
                    and block.get("phase_boundary_model", "impermeable_flux") != "projection_legacy"
                ):
                    errors.append(
                        "contact_angle.enforce_solid_phi is legacy-only and requires "
                        "phase_boundary_model='projection_legacy'"
                    )
                for key in ("We", "Re", "eps_factor", "R", "dt"):
                    if key in block:
                        _positive_number(block[key], f"contact_angle.{key}", errors)
                        bound = (
                            1.5 if key == "R" else (10.0 if key == "eps_factor" else (1.0 if key == "dt" else 1.0e6))
                        )
                        _upper_bound(block[key], f"contact_angle.{key}", bound, errors)
                if "wall_energy_amp" in block:
                    _positive_number(block["wall_energy_amp"], "contact_angle.wall_energy_amp", errors, allow_zero=True)
            elif name == "impact":
                cases = block.get("cases")
                if not isinstance(cases, list) or not cases:
                    errors.append("benchmarks.impact.cases must be a non-empty list")
                else:
                    for i, case in enumerate(cases):
                        _unknown_keys(case, _IMPACT_CASE_KEYS, f"impact.cases[{i}]", errors) if isinstance(
                            case, dict
                        ) else None
                        if not isinstance(case, dict):
                            errors.append(f"impact.cases[{i}] must be an object")
                            continue
                        for key in ("We", "Re", "R", "impact_gap", "u_impact"):
                            if key in case:
                                _positive_number(
                                    case[key], f"impact.cases[{i}].{key}", errors, allow_zero=(key == "impact_gap")
                                )
                        if "cos_theta" in case and (
                            isinstance(case["cos_theta"], bool)
                            or not isinstance(case["cos_theta"], (int, float))
                            or not -1 <= case["cos_theta"] <= 1
                        ):
                            errors.append(f"impact.cases[{i}].cos_theta must be in [-1, 1]")
                        if "velocity_mode" in case and case["velocity_mode"] not in {"uniform", "streamfunction"}:
                            errors.append(f"impact.cases[{i}].velocity_mode must be 'uniform' or 'streamfunction'")
                        if "wetting_model" in case and case["wetting_model"] not in {
                            "surface_energy",
                            "surface_energy_volume_v6",
                            "legacy_affinity",
                            "none",
                        }:
                            errors.append(f"impact.cases[{i}].wetting_model is unknown")
                        if "phase_boundary_model" in case and case["phase_boundary_model"] not in {
                            "impermeable_flux",
                            "projection_legacy",
                        }:
                            errors.append(f"impact.cases[{i}].phase_boundary_model is invalid")
                        if "enforce_solid_phi" in case and not isinstance(case["enforce_solid_phi"], bool):
                            errors.append(f"impact.cases[{i}].enforce_solid_phi must be a boolean")
                        if (
                            case.get("enforce_solid_phi", False)
                            and case.get("phase_boundary_model", "impermeable_flux") != "projection_legacy"
                        ):
                            errors.append(
                                f"impact.cases[{i}].enforce_solid_phi requires phase_boundary_model='projection_legacy'"
                            )
                _grid_size(block.get("N"), "impact.N", errors)
                _step_count(block.get("steps"), "impact.steps", errors)
                _step_count(block.get("save_every"), "impact.save_every", errors)
                for key in ("We", "Re", "eps_factor", "dt", "wall_height"):
                    if key in block:
                        _positive_number(block[key], f"impact.{key}", errors)
                        bound = (
                            10.0
                            if key == "eps_factor"
                            else (1.0 if key == "dt" else (1.5 if key == "wall_height" else 1.0e6))
                        )
                        _upper_bound(block[key], f"impact.{key}", bound, errors)
            else:
                _unknown_keys(block, {"grid_refinement", "interface_thickness"}, "benchmarks.convergence", errors)
                for mode in ("grid_refinement", "interface_thickness"):
                    if mode in block:
                        _validate_sweep(block[mode], mode, mode, errors)
                if not any(mode in block for mode in ("grid_refinement", "interface_thickness")):
                    errors.append(
                        "benchmarks.convergence must define a grid_refinement and/or interface_thickness sweep"
                    )

    output = data.get("output")
    if not isinstance(output, dict):
        errors.append("output must be an object")
    else:
        _unknown_keys(output, _OUTPUT_KEYS, "output", errors)
        if not isinstance(output.get("directory"), str) or not output.get("directory"):
            errors.append("output.directory must be a non-empty string")
        if output.get("formats", ["json"]) != ["json"]:
            errors.append("output.formats currently supports only ['json']")

    return errors


def load_config(path: str | Path) -> ValidationConfig:
    """Read and validate a JSON config; malformed or unknown fields are errors."""
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read validation config {source}: {exc}") from exc
    errors = validate_config_data(data)
    if errors:
        raise ValueError(f"Invalid validation config {source}: " + "; ".join(errors))
    return ValidationConfig(
        profile=data["profile"], dtype=data["dtype"], benchmarks=data["benchmarks"], output=data["output"]
    )


def _config_dict(cfg: ValidationConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(cfg, ValidationConfig):
        return cfg.to_dict()
    if isinstance(cfg, Mapping):
        return dict(cfg)
    raise TypeError("cfg must be a ValidationConfig or mapping")


def canonical_config_json(cfg: ValidationConfig | Mapping[str, Any]) -> str:
    """Canonical UTF-8 JSON: mapping order ignored and non-finite values rejected."""
    return json.dumps(_config_dict(cfg), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def config_sha256(cfg: ValidationConfig | Mapping[str, Any]) -> str:
    """SHA-256 of canonical config JSON."""
    return hashlib.sha256(canonical_config_json(cfg).encode("utf-8")).hexdigest()


def compute_file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def get_git_sha() -> str:
    """Return HEAD SHA, falling back to GITHUB_SHA and then the explicit unknown marker."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_repo_root(),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        value = result.stdout.strip()
        if result.returncode == 0 and len(value) == 40:
            return value
    except (OSError, subprocess.SubprocessError):
        pass
    return os.environ.get("GITHUB_SHA", "unknown")


def get_phasefield_sha256() -> str:
    return compute_file_sha256(Path(__file__).resolve().parents[1] / "phasefield.py")


def compute_validation_code_hash() -> str:
    """Fingerprint the required production source files in stable filename order."""
    root = Path(__file__).resolve().parent
    names = (
        "config.py",
        "observables.py",
        "validation.py",
        "convergence.py",
        "report.py",
        "run_validation.py",
        "capillary_audit.py",
        "wetting_audit.py",
        "phase_boundary_audit.py",
        "solid_gas_film_audit.py",
        "contact_angle_matrix.py",
        "contact_line_kinetics.py",
        "nonneutral_wetting_audit.py",
    )
    digest = hashlib.sha256()
    for name in sorted(names):
        path = root / name
        digest.update(name.encode("utf-8") + b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def get_runtime_info() -> dict[str, Any]:
    import platform
    import sys

    import jax

    return {
        "python_version": sys.version.split()[0],
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "platform": platform.platform(),
    }
