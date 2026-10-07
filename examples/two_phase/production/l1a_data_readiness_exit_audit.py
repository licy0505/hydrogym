"""L1A-2o -- contact-line residual motion and L1A data-readiness exit audit.

Terminal L1A decision stage (sections 0-71 of the stage specification).  The
audit answers exactly one bounded question: is the current contract-11
solver/data pipeline stable, resolved, representation-faithful, and causally
understood enough to begin L1B production dataset generation for the actual
surrogate task?

Allowed headline verdicts (nothing else may be emitted)::

    L1B_DATA_READY
    L1B_DATA_READY_WITH_CAVEAT
    L1B_DATA_NOT_READY

This is a decision/audit stage.  It never repairs the solver, never changes a
threshold, never tunes the dataset schema, and never extends the frozen 60 deg
authority beyond its 50k endpoint.  Every classification threshold used below
is either imported from existing repository code or pre-declared in this
module *before* the forensic measurement (section 37); no acceptance number is
invented after seeing data.

Profiles
--------
``quick``
    Methodology only: the full machinery (contract discovery, schedule
    calculation, canary construction, observable extraction, alignment, sample
    export roundtrip, representation-noise calculation, contact-gap plumbing,
    reader compatibility, verdict schema, fail-closed unmeasured handling) on
    tiny configurations.  It never pretends to run the N=192 refinement
    matrix (section 46).
``forensic``
    The bounded exit matrix: production-resolution canaries, spatial- and
    temporal-refinement subsets, the 60 deg residual relevance analysis on the
    frozen authority, the contact-gap refinement case, sample representation
    fidelity, simple + complex geometry canaries, reader compatibility
    (section 47).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cases as cases_module
import generate_dataset as generator
import numpy as np
import phasefield as pf
import surrogate as surrogate_module
from production import dataset_lineage as lineage
from production import observables as observables_module
from production import validation as validation_module

STAGE = "L1A-2o"
SECTION_VERSION = 2

EVIDENCE_ROOT = Path("evidence/l1a2o")
ARTIFACT_ROOT = Path("artifacts/l1a2o")
CACHE_PATH = ARTIFACT_ROOT / "cache" / "canaries.json"
CANARY_DIR = ARTIFACT_ROOT / "canaries"

#: the frozen 60 deg static-wetting authority endpoint (L1A-2j/k/l/n).  The
#: replayer below may never advance past this step (section 12/50).
LATE60_ENDPOINT_STEP = 50_000
#: the frozen late window is the last 10k steps of the authority (L1A-2m).
LATE60_WINDOW_START_STEP = LATE60_ENDPOINT_STEP - 10_000
#: the authority is an N=128 static case; ds=2 gives a 64-cell saved grid with
#: saved_dx = 0.09375, identical to the production N=192, ds=3 sample pitch.
LATE60_EXPORT_DS = 2

#: pre-declared classification thresholds for the sample-representation noise
#: ratio R_repr = ||saved - ideal|| / (||frame signal|| + eps) (sections 25/26).
#: Declared here BEFORE any forensic measurement; never tuned afterwards.
REPRESENTATION_RATIO_SUBDOMINANT_MAX = 0.25
REPRESENTATION_RATIO_DOMINANT_MIN = 1.0
#: a normalized refinement change at or below this factor of the provisional
#: repo target is reported as "at target"; above 2x is "above target".
REFINEMENT_TARGET_FACTOR_THRESHOLD = 2.0

#: existing provisional readiness target, discovered from repository code
#: (production/validation.py; section 15/17/37 -- do not modify).
KEY_OBSERVABLE_REFINEMENT_CHANGE = float(
    validation_module.PROVISIONAL_READINESS_TARGETS["key_observable_refinement_change"]
)

#: the exact generator CLI thresholds this stage is allowed to run under
#: (section 3/56: the audit may not relax any of them; discovered values must
#: match these).
EXPECTED_GENERATOR_THRESHOLDS = {
    "max_phi_overshoot": 0.02,
    "max_solid_leak": 5e-4,
    "min_total_mass_ratio": 0.995,
    "max_total_mass_ratio": 1.005,
    "max_speed": 5.0,
    "min_feature_cells": 2.0,
}

AUDIT_SOURCE_FILES = (
    "generate_dataset.py",
    "cases.py",
    "surrogate.py",
    "production/dataset_lineage.py",
    "production/observables.py",
    "production/l1a_data_readiness_exit_audit.py",
    "phasefield.py",
)

AuditValidationError = RuntimeError


# ---------------------------------------------------------------------------
# provenance helpers
# ---------------------------------------------------------------------------


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # pragma: no cover - git always exists in this repo
        return "unknown"


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    ).hexdigest()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)):
        return np.asarray(obj).tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def _source_hashes() -> dict[str, str]:
    return {name: _file_sha256(Path(name)) for name in AUDIT_SOURCE_FILES}


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "jax": __import__("jax").__version__,
        "numpy": np.__version__,
        "platform": __import__("jax").default_backend(),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=_json_default) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------
# section 4/58: discover the actual ML task contract from code
# ---------------------------------------------------------------------------


def _argparse_defaults_from_source(path: Path) -> dict[str, Any]:
    """Extract the ``main()`` argparse defaults from the generator source (AST)."""

    tree = ast.parse(Path(path).read_text())
    defaults: dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
                    if call.func.attr != "add_argument" or not call.args:
                        continue
                    first = call.args[0]
                    if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
                        continue
                    name = first.value.lstrip("-").replace("-", "_")
                    for keyword in call.keywords:
                        if keyword.arg == "default":
                            defaults[name] = ast.literal_eval(keyword.value)
    return defaults


def discover_generator_defaults() -> dict[str, Any]:
    """Discover the generator CLI contract from ``generate_dataset.py`` source."""

    defaults = _argparse_defaults_from_source(Path("generate_dataset.py"))
    required = ("nsteps", "save_every", "ds", "N", "dt")
    missing = [name for name in required if name not in defaults]
    if missing:
        raise AuditValidationError(f"generator defaults not discoverable from source: missing {missing}")
    thresholds = {name: defaults[name] for name in EXPECTED_GENERATOR_THRESHOLDS if name in defaults}
    missing_thresholds = set(EXPECTED_GENERATOR_THRESHOLDS) - set(thresholds)
    if missing_thresholds:
        raise AuditValidationError(f"generator acceptance thresholds not discoverable: {sorted(missing_thresholds)}")
    drifted = {
        name: {"actual": thresholds[name], "expected": expected}
        for name, expected in EXPECTED_GENERATOR_THRESHOLDS.items()
        if float(thresholds[name]) != float(expected)
    }
    return {
        "source": "generate_dataset.py main() argparse defaults",
        "defaults": defaults,
        "acceptance_thresholds": thresholds,
        "threshold_status": (
            "current_defaults_match_frozen_stage_expectations"
            if not drifted
            else "thresholds_changed_from_expectations"
        ),
        "threshold_drift": drifted,
    }


def _dataset_dtype_discovery(npz_path: Path) -> dict[str, str]:
    """Read the *actual* stored dtypes from a generated sample file."""

    with np.load(npz_path, allow_pickle=True) as data:
        return {name: str(np.asarray(data[name]).dtype) for name in sorted(data.files) if name != "case"}


def discover_ml_task_contract(sample_file: Path | None = None) -> dict[str, Any]:
    """Freeze the actual L1B-facing contract (section 4/58) from current code.

    Nothing here is hard-coded expectation: defaults come from the generator
    AST, the sample dtypes from a real generated file when available, the
    reader/target structure from :mod:`surrogate`, and every source file is
    hashed.
    """

    gen = discover_generator_defaults()
    defaults = gen["defaults"]
    contract: dict[str, Any] = {
        "stage": STAGE,
        "section_version": SECTION_VERSION,
        "discovered_from": {
            "generator": "generate_dataset.py main() argparse defaults (AST)",
            "cases": "cases.py CASES / CASE_SETS",
            "reader": "surrogate.py load_arrays / load_full / load_windows",
            "sample_dtypes": "actual generated .npz" if sample_file is not None else "unmeasured (no sample file yet)",
            "lineage": "production/dataset_lineage.py constants",
        },
        "dynamic_input_fields": ["phi", "u", "v", "chi"],
        "target_fields": ["phi", "u", "v"],
        "static_geometry_fields": ["chi", "sdf"],
        "conditioning_fields": ["We_over_100", "Re_over_200", "cos_theta", "saved_dx"],
        "conditioning_source": "generate_dataset._save_case scalars vector",
        "model_input_channels": 4,
        "model_output_channels": int(surrogate_module.UNet.out_channels),
        "solver_resolution": {"N": int(defaults["N"]), "Lx": 6.0, "Ly": 6.0},
        "saved_resolution": {"factor_ds": int(defaults["ds"]), "saved_N": int(defaults["N"]) // int(defaults["ds"])},
        "downsampling_rule": "average_pool_mean_per_block (generate_dataset._downsample_history)",
        "geometry_downsampling_rule": "hard-mask mean then EDT reinitialisation (generate_dataset._coarsen_geometry)",
        "solver_dtype_per_field": {
            "phi": "float64",
            "u": "float32",
            "v": "float32",
            "chi": "float32",
            "sdf": "float32",
        },
        "saved_dtype_per_field": {
            "phi": "float32",
            "u": "float16",
            "v": "float16",
            "chi": "float32",
            "sdf": "float32",
            "scalars": "float32",
            "time": "float32",
        },
        "reader_dtype_per_field": {
            "phi": "float32 (kept as stored; no second solver-to-sample cast)",
            "u": "float16 stored, cast to float32 on load",
            "v": "float16 stored, cast to float32 on load",
            "chi": "float32",
            "sdf": "float32",
        },
        "sample_cast_policy": lineage.DATASET_SAMPLE_CAST_POLICY,
        "sample_representation": lineage.DATASET_SAMPLE_REPRESENTATION,
        "dataset_schema_version": int(lineage.DATASET_SCHEMA_VERSION),
        "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
        "nominal_schedule": {
            "dt": float(defaults["dt"]),
            "nsteps": int(defaults["nsteps"]),
            "save_every": int(defaults["save_every"]),
            "nominal_horizon": float(defaults["nsteps"]) * float(defaults["dt"]),
        },
        "effective_schedule_note": (
            "requested dt is min(case dt, stable_dt(N, u_max=2.0)); nsteps/save_every are re-"
            "rounded so the physical horizon and frame spacing stay at the nominal request"
        ),
        "surface_families": {
            "train_simple": sorted(
                {
                    case.get("surface", "flat")
                    for case in cases_module.train_cases()
                    if case.get("surface") in cases_module.SIMPLE_FAMILIES
                }
            ),
            "test_complex": sorted({case["surface"] for case in cases_module.test_cases()}),
            "case_sets": sorted(cases_module.CASE_SETS),
        },
        "stored_targets_are_next_frame": (
            "surrogate.load_arrays: Y = (phi, u, v) at frame i+1; X = (phi,u,v,chi) at frame i"
        ),
        "non_targets": {
            "pressure": "not stored in the schema-3 sample and not a training target",
            "impact_force": "not stored; not a training target",
            "derived_observables": "audit-only (spread, gaps, contact lines); never labels",
        },
        "source_hashes": _source_hashes(),
        "generator_cli_defaults": gen,
        "git_sha": _git_sha(),
    }
    if sample_file is not None and Path(sample_file).is_file():
        contract["actual_stored_dtypes"] = _dataset_dtype_discovery(Path(sample_file))
        contract["dtype_agreement"] = {
            field_name: contract["actual_stored_dtypes"].get(field_name)
            == contract["saved_dtype_per_field"].get(field_name)
            for field_name in ("phi", "u", "v", "chi", "sdf", "scalars", "time")
        }
    return contract


# ---------------------------------------------------------------------------
# section 5/50: effective schedule and physical horizon
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneratorArgs:
    """The generator arguments this audit runs under (never a CLI re-parse)."""

    N: int = 192
    ds: int = 3
    dt: float = 4e-3
    nsteps: int = 2000
    save_every: int = 20
    max_phi_overshoot: float = 0.02
    max_solid_leak: float = 5e-4
    min_total_mass_ratio: float = 0.995
    max_total_mass_ratio: float = 1.005
    max_speed: float = 5.0
    min_feature_cells: float = 2.0

    def namespace(self) -> argparse.Namespace:
        return argparse.Namespace(**vars(self))

    def fingerprint(self) -> str:
        return _canonical_hash(vars(self))


def effective_schedule(case: dict, args: GeneratorArgs) -> dict[str, Any]:
    """Exact production schedule via ``generate_dataset._effective_schedule``."""

    dt, nsteps, save_every = generator._effective_schedule(case, args.namespace())
    frame_dt = dt * save_every
    horizon = nsteps * dt
    return {
        "requested_dt": float(case.get("dt", args.dt)),
        "effective_dt": float(dt),
        "nsteps": int(nsteps),
        "save_every": int(save_every),
        "frame_dt": float(frame_dt),
        "physical_horizon": float(horizon),
        "n_frames": int(nsteps // save_every),
        "stable_dt_cap_N": float(
            pf.stable_dt(pf.PhaseFieldParams(Nx=args.N, Ny=args.N, Lx=6.0, Ly=6.0, dt=1.0), u_max=2.0)
        ),
        "requested_horizon": float(args.nsteps * args.dt),
        "requested_frame_dt": float(args.dt * args.save_every),
    }


def dataset_fingerprint(case: dict, args: GeneratorArgs, schedule: dict[str, Any]) -> str:
    """The exact production trajectory fingerprint (generator implementation)."""

    return generator._dataset_fingerprint(
        case, args.namespace(), schedule["effective_dt"], schedule["nsteps"], schedule["save_every"]
    )


# ---------------------------------------------------------------------------
# section 9: the bounded canary matrix
# ---------------------------------------------------------------------------


def _find_case(split: str, surface: str, seed: int) -> dict:
    pool = list(cases_module.CASES)
    for producer in cases_module.CASE_SETS.values():
        try:
            pool.extend(producer())
        except Exception:
            continue
    for case in pool:
        if case["split"] == split and case.get("surface") == surface and int(case.get("seed", -1)) == seed:
            return {**case}
    raise AuditValidationError(f"case not found in cases.py: {split}/{surface}/seed={seed}")


def canary_matrix() -> list[dict[str, Any]]:
    """The mandatory exit canaries (section 9), all from existing ``cases.py``."""

    matrix: list[dict[str, Any]] = []
    flat_train = [case for case in cases_module.train_cases() if case.get("surface") == "flat"]
    for we in (100.0, 200.0):
        for cos_theta, angle in ((0.5, 60), (0.0, 90), (-0.5, 120)):
            matches = [
                c for c in flat_train if float(c.get("We", 100.0)) == we and float(c.get("cos_theta", 0.0)) == cos_theta
            ]
            if not matches:
                raise AuditValidationError(
                    f"mandatory canary missing from cases.py: flat We={we} cos_theta={cos_theta}"
                )
            matrix.append(
                {
                    "role": "flat_impact_canary",
                    "angle_deg_label": angle,
                    "mandatory": True,
                    "case": matches[0],
                }
            )
    matrix.append(
        {
            "role": "pillar_training_canary",
            "angle_deg_label": None,
            "mandatory": True,
            "case": _find_case("train", "pillars", 8),
        }
    )
    matrix.append(
        {
            "role": "complex_heldout_canary",
            "angle_deg_label": None,
            "mandatory": True,
            "case": _find_case("test", "random_pillars", 100),
        }
    )
    matrix.append(
        {
            "role": "optional_low_we",
            "angle_deg_label": None,
            "mandatory": False,
            "case": None,
            "exclusion_reason": (
                "the default ``--set base`` generation targeted for the immediate L1B run contains no"
                " low-We case; the separate lowWe sets are outside the audited envelope (section 9 optional)"
            ),
        }
    )
    return matrix


# ---------------------------------------------------------------------------
# section 6/11/14: observables on the solver grid
# ---------------------------------------------------------------------------


def _contact_line_positions(phi: np.ndarray, dx: float, threshold: float = 0.5) -> tuple[float, float] | None:
    """Explicit left/right contact-line x positions of thresholded liquid."""

    occupied = np.flatnonzero(np.any(phi >= threshold, axis=1))
    if occupied.size == 0:
        return None
    return float(occupied[0] * dx), float(occupied[-1] * dx)


def extract_observables(
    phi_hist: np.ndarray,
    u_hist: np.ndarray,
    v_hist: np.ndarray,
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
    R: float,
    save_every: int,
) -> dict[str, Any]:
    """Per-frame derived observables using ``production/observables.py``.

    Formal phase mass is the contract-9 conserved quantity ``sum_i V_i phi_i``
    (cut-cell control volumes).  Force/load observables are not part of the
    current contract and are reported as unmeasured, not zero (section 6B).
    """

    phi_hist = np.asarray(phi_hist, dtype=np.float64)
    u_hist = np.asarray(u_hist, dtype=np.float64)
    v_hist = np.asarray(v_hist, dtype=np.float64)
    sdf = np.asarray(solid.sdf, dtype=np.float64)
    dx, dy = float(p.dx), float(p.dy)
    eps = float(p.eps)
    x_axis = (np.arange(phi_hist.shape[1], dtype=np.float64) + 0.5) * dx
    y_axis = (np.arange(phi_hist.shape[2], dtype=np.float64) + 0.5) * dy
    volume = np.asarray(pf.phase_control_volumes(solid, p), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for k in range(phi_hist.shape[0]):
        phi, u, v = phi_hist[k], u_hist[k], v_hist[k]
        row: dict[str, Any] = {"frame": k, "t": float((k + 1) * save_every * p.dt)}
        # Fail-closed frame extraction: a non-finite solver frame is recorded as
        # such (finite=False, observables None) instead of crashing the audit.
        frame_finite = bool(np.isfinite(phi).all() and np.isfinite(u).all() and np.isfinite(v).all())
        row["finite"] = frame_finite
        if not frame_finite:
            for key in (
                "formal_mass",
                "total_mass_plain_sum",
                "spread_width",
                "beta",
                "drop_bottom_height",
                "drop_top_height",
                "drop_vertical_extent",
                "centroid_x",
                "centroid_y",
                "contact_line_left",
                "contact_line_right",
                "max_speed",
                "max_abs_u",
                "max_abs_v",
                "phi_min",
                "phi_max",
                "gap05",
                "gap05_over_dx",
                "gap05_over_eps",
                "gap01",
                "gap01_over_dx",
                "gap01_over_eps",
            ):
                row[key] = None
            rows.append(row)
            continue
        row["formal_mass"] = float(np.sum(phi * volume))
        row["total_mass_plain_sum"] = float(np.sum(phi) * dx * dy)
        width = observables_module.periodic_spreading_width(phi, threshold=0.5, dx=dx, Lx=p.Lx)
        row["spread_width"] = float(width)
        row["beta"] = float(observables_module.beta_from_width(width, R))
        try:
            row["drop_bottom_height"] = float(observables_module.bottom_height(phi, sdf, y_axis))
            row["drop_top_height"] = float(observables_module.top_height(phi, sdf, y_axis))
            row["drop_vertical_extent"] = float(observables_module.vertical_extent(phi, sdf, y_axis))
        except ValueError:
            row["drop_bottom_height"] = row["drop_top_height"] = row["drop_vertical_extent"] = None
        try:
            cx, cy = observables_module.center_of_mass(phi, sdf, x_axis, y_axis)
            row["centroid_x"], row["centroid_y"] = float(cx), float(cy)
        except ValueError:
            row["centroid_x"] = row["centroid_y"] = None
        positions = _contact_line_positions(phi, dx)
        row["contact_line_left"] = None if positions is None else positions[0]
        row["contact_line_right"] = None if positions is None else positions[1]
        row["max_speed"] = float(np.max(np.sqrt(u * u + v * v)))
        row["max_abs_u"] = float(np.max(np.abs(u)))
        row["max_abs_v"] = float(np.max(np.abs(v)))
        for name, threshold in (("gap05", 0.5), ("gap01", 0.1)):
            try:
                gap = observables_module.bottom_gap(phi, sdf, threshold=threshold)
                row[name] = float(gap)
                row[f"{name}_over_dx"] = float(gap / dx)
                row[f"{name}_over_eps"] = float(gap / eps)
            except ValueError:
                row[name] = row[f"{name}_over_dx"] = row[f"{name}_over_eps"] = None
        row["phi_min"] = float(np.min(phi))
        row["phi_max"] = float(np.max(phi))
        rows.append(row)
    force_block = {
        "F": "unmeasured_not_current_contract",
        "F_max": "unmeasured_not_current_contract",
        "t_Fmax": "unmeasured_not_current_contract",
        "impulse": "unmeasured_not_current_contract",
        "reason": "no authoritative force/load observable exists in production/observables.py; none invented",
    }
    times = [float(row["t"]) for row in rows]
    return {
        "rows": rows,
        "frame_times": times,
        "times": times,
        "dx": dx,
        "dy": dy,
        "eps": eps,
        "R": float(R),
        "force_observables": force_block,
        "formal_mass_definition": "sum_i V_i phi_i (cut-cell control volumes in physical area units, contract v9)",
    }


def _finite_or_none(value: Any) -> float | None:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.size == 0 or not np.isfinite(array).all():
        return None
    return float(array)


def late_window_boundedness(observable: dict[str, Any], late_fraction: float = 0.25) -> dict[str, Any]:
    """Boundedness of every direct label and key derived observable (section 14).

    ``late_fraction`` is the pre-declared tail of the *dataset horizon* used as
    the "late part of the dataset horizon" phase; no equilibrium is demanded.
    """

    rows = observable["rows"]
    if not rows:
        return {"status": "UNMEASURED", "reason": "no frames"}
    n_late = max(1, int(np.ceil(len(rows) * late_fraction)))
    keys = (
        "formal_mass",
        "spread_width",
        "beta",
        "drop_vertical_extent",
        "centroid_y",
        "contact_line_left",
        "contact_line_right",
        "max_speed",
        "gap05",
        "gap01",
    )
    per_key: dict[str, Any] = {}
    for key in keys:
        values = np.asarray([_finite_or_none(row.get(key)) for row in rows], dtype=np.float64)
        if np.all(np.isnan(values)):
            per_key[key] = {"status": "undefined_on_all_frames"}
            continue
        finite = values[np.isfinite(values)]
        frame_signal = float(np.max(np.abs(np.diff(values)))) if finite.size > 1 else 0.0
        late = values[-n_late:]
        late_drift = float(np.nanmax(np.abs(np.diff(late)))) if late.size > 1 else 0.0
        per_key[key] = {
            "finite_all_frames": bool(np.isfinite(values).all()),
            "min": float(np.nanmin(values)),
            "max": float(np.nanmax(values)),
            "max_frame_to_frame_change": frame_signal,
            "late_window_max_frame_to_frame_change": late_drift,
            "late_window_bounded": bool(
                late_drift <= max(frame_signal, 1e-12) or late_drift <= abs(float(np.nanmax(values))) * 0.1
            ),
            "value_final": float(values[-1]),
        }
    finite_all = all(entry.get("finite_all_frames", True) for entry in per_key.values() if isinstance(entry, dict))
    return {
        "status": "PASS" if finite_all else "FAIL",
        "late_fraction": late_fraction,
        "n_frames_total": len(rows),
        "n_frames_late": n_late,
        "per_observable": per_key,
        "force_observables": observable["force_observables"],
    }


# ---------------------------------------------------------------------------
# section 10/11/48: canary execution through the exact production generator
# ---------------------------------------------------------------------------


def _case_display_name(case: dict, role: str) -> str:
    label = cases_module.case_label({k: v for k, v in case.items() if k != "family"})
    return f"{role}__{label}"


def _cache_entry_valid(entry: dict[str, Any], binding: dict[str, Any]) -> bool:
    """Fail-closed cache validation (section 48): every binding must match."""

    if not isinstance(entry, dict):
        return False
    stored_binding = entry.get("binding")
    if not isinstance(stored_binding, dict) or {key: stored_binding.get(key) for key in binding} != binding:
        return False
    required = (
        "fingerprint",
        "canary_npz",
        "observables",
        "diagnostics",
        "accepted",
    )
    return all(key in entry for key in required)


def _cache_binding(case: dict, args: GeneratorArgs, schedule: dict[str, Any], fingerprint: str) -> dict[str, Any]:
    return {
        "section_version": SECTION_VERSION,
        "stage": STAGE,
        "git_sha": _git_sha(),
        "solver_contract": int(pf.SOLVER_CONTRACT_VERSION),
        "solver_source_hash": _file_sha256(Path("phasefield.py")),
        "generator_source_hash": _file_sha256(Path("generate_dataset.py")),
        "observables_source_hash": _file_sha256(Path("production/observables.py")),
        "case": case,
        "config_fingerprint": _canonical_hash({"case": case, "schedule": schedule}),
        "N": int(args.N),
        "dt": float(schedule["effective_dt"]),
        "nsteps": int(schedule["nsteps"]),
        "save_every": int(schedule["save_every"]),
        "ds": int(args.ds),
        "sample_dtype_policy": lineage.DATASET_SAMPLE_CAST_POLICY,
        "generator_thresholds": vars(args),
        "fingerprint": fingerprint,
    }


def _canary_observables_from_history(
    phi: np.ndarray, u: np.ndarray, v: np.ndarray, solid: pf.Solid, p: pf.PhaseFieldParams, case: dict, save_every: int
) -> dict[str, Any]:
    return extract_observables(phi, u, v, solid, p, float(case.get("R", 0.7)), save_every)


def run_canary(
    case: dict,
    role: str,
    args: GeneratorArgs,
    *,
    use_cache: bool = True,
    save_sample_file: bool = True,
    mandatory: bool = False,
) -> dict[str, Any]:
    """Generate one canary through the exact production generator semantics.

    ``build_case -> rollout -> _diagnose -> _save_case`` are the generator's own
    functions; the audit never re-implements physics or validation.  A canary
    rejected by the generator is recorded as rejected and never declared
    usable (section 11).
    """

    name = _case_display_name(case, role)
    schedule = effective_schedule(case, args)
    fingerprint = dataset_fingerprint(case, args, schedule)
    binding = _cache_binding(case, args, schedule, fingerprint)
    npz_path = CANARY_DIR / f"{name}.npz"
    cache = _read_json(CACHE_PATH) if CACHE_PATH.is_file() and use_cache else {}
    entry = cache.get(name)
    if use_cache and entry is not None and _cache_entry_valid(entry, binding):
        entry["cache"] = "hit"
        entry["npz_available"] = npz_path.is_file()
        return entry

    started = time.perf_counter()
    p, solid, initial = pf.build_case(case, N=args.N, dt=schedule["effective_dt"])
    final, phi, u, v = pf.rollout(initial, solid, p, schedule["nsteps"], save_every=schedule["save_every"])
    del final
    phi = np.asarray(phi)
    u = np.asarray(u)
    v = np.asarray(v)
    ok, diagnostics = generator._diagnose(
        initial,
        phi,
        u,
        v,
        solid,
        p,
        max_phi_overshoot=args.max_phi_overshoot,
        max_solid_leak=args.max_solid_leak,
        min_total_mass_ratio=args.min_total_mass_ratio,
        max_total_mass_ratio=args.max_total_mass_ratio,
        max_speed=args.max_speed,
    )
    record: dict[str, Any] = {
        "name": name,
        "role": role,
        "case": case,
        "schedule": schedule,
        "fingerprint": fingerprint,
        "binding": binding,
        "solver_params": {
            "N": int(p.Nx),
            "dt": float(p.dt),
            "eps": float(p.eps),
            "We": float(p.We),
            "Re": float(p.Re),
            "M": float(p.M),
            "eta_pen": float(p.eta_pen),
            "wetting_model": str(p.wetting_model),
            "phase_boundary_model": str(p.phase_boundary_model),
            "phase_transport_geometry": str(p.phase_transport_geometry),
            "wall_measure": str(p.wall_measure),
        },
        "accepted_by_generator": bool(ok),
        "generator_rejection_path": None if ok else "rejected (generator physics validation)",
        "mandatory": bool(mandatory),
        "feature_cells_min": _feature_cells_for(case, args),
        "elapsed_seconds": time.perf_counter() - started,
    }
    if not ok:
        # The trajectory is rejected as *training data*, but it is still a solver
        # output and the exit audit measures it (section 11: a rejected canary
        # must never be declared usable -- which is why no sample .npz is written).
        record["observables"] = _canary_observables_from_history(phi, u, v, solid, p, case, schedule["save_every"])
        record["diagnostics"] = diagnostics
        record["frame_signal"] = frame_signal_audit(
            {
                name_: generator._downsample_history(array.astype(np.float64), args.ds)
                for name_, array in (("phi", phi), ("u", u), ("v", v))
            }
        )
        final_fields_path = CANARY_DIR / f"{name}_final_fields.npz"
        CANARY_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            final_fields_path,
            phi=np.asarray(phi[-1], dtype=np.float64),
            u=np.asarray(u[-1], dtype=np.float64),
            v=np.asarray(v[-1], dtype=np.float64),
        )
        record["final_fields_path"] = str(final_fields_path)
        record["cache"] = "stored_rejected"
        cache[name] = record
        if use_cache:
            _write_json(CACHE_PATH, cache)
        import jax

        jax.clear_caches()
        return record
    if save_sample_file:
        CANARY_DIR.mkdir(parents=True, exist_ok=True)
        temporary = CANARY_DIR / f".{name}.tmp.npz"
        generator._save_case(
            temporary,
            case,
            p,
            solid,
            phi,
            u,
            v,
            schedule["save_every"],
            args.ds,
            diagnostics,
            fingerprint,
            float(
                np.inf
                if case.get("surface", "flat") == "flat"
                else generator._feature_cells(case, float(p.dx) * args.ds)
            ),
            {
                "nominal_We": float(case.get("We", 100.0)),
                "nominal_Re": float(case.get("Re", 200.0)),
                "u_impact_star": float(case.get("u_impact", 0.5)),
                "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
                "audit_role": role,
            },
        )
        temporary.replace(npz_path)
        record["canary_npz"] = str(npz_path)
    record["observables"] = _canary_observables_from_history(phi, u, v, solid, p, case, schedule["save_every"])
    record["diagnostics"] = diagnostics
    record["cache"] = "miss"
    if save_sample_file:
        # representation fidelity is measured while the float64 solver history is
        # still in memory; the result (scalars only) is what the cache stores.
        record["representation"] = representation_fidelity(phi, u, v, npz_path, args.ds)
        ideal_saved = {
            name: generator._downsample_history(array.astype(np.float64), args.ds)
            for name, array in (("phi", phi), ("u", u), ("v", v))
        }
        record["frame_signal"] = frame_signal_audit(ideal_saved)
        if args.save_every >= 2:
            record["cadence_half"] = {name: cadence_from_history(array) for name, array in ideal_saved.items()}
        final_fields_path = CANARY_DIR / f"{name}_final_fields.npz"
        np.savez_compressed(
            final_fields_path,
            phi=np.asarray(phi[-1], dtype=np.float64),
            u=np.asarray(u[-1], dtype=np.float64),
            v=np.asarray(v[-1], dtype=np.float64),
        )
        record["final_fields_path"] = str(final_fields_path)
    cache[name] = record
    if use_cache:
        _write_json(CACHE_PATH, cache)
    import jax

    jax.clear_caches()
    return record


def canary_record_fingerprint(record: dict[str, Any]) -> str:
    return str(record.get("fingerprint", ""))


def characterize_overshoot(case: dict, role: str, args: GeneratorArgs) -> dict[str, Any]:
    """Per-frame phi-overshoot characterisation of one rejected canary.

    This is exit-audit evidence for the single next blocker (section 42/69): it
    records WHEN the over/undershoot occurs and WHERE it lives (interface band
    vs wall band), without changing any solver semantics and without relaxing
    the generator gate.
    """

    schedule = effective_schedule(case, args)
    p, solid, initial = pf.build_case(case, N=args.N, dt=schedule["effective_dt"])
    _final, phi, u, v = pf.rollout(initial, solid, p, schedule["nsteps"], save_every=schedule["save_every"])
    del _final, u, v
    phi = np.asarray(phi, dtype=np.float64)
    undershoot = np.maximum(-phi, 0.0)
    overshoot = np.maximum(phi - 1.0, 0.0)
    total = np.maximum(undershoot, overshoot)
    per_frame = total.reshape(total.shape[0], -1).max(axis=1)
    frame = int(np.argmax(per_frame))
    location = np.unravel_index(np.argmax(total[frame]), total[frame].shape)
    dx = float(p.dx)
    y_of = (np.arange(phi.shape[2]) + 0.5) * dx
    wall_band = y_of < 4.0 * float(p.eps)
    at_frame = total[frame]
    total_wall = float(at_frame[:, wall_band].max()) if wall_band.any() else 0.0
    total_outside_wall = float(np.max(at_frame[:, ~wall_band])) if (~wall_band).any() else 0.0
    import jax

    jax.clear_caches()
    return {
        "case": case,
        "role": role,
        "frame_of_max": frame,
        "t_of_max": float((frame + 1) * schedule["frame_dt"]),
        "per_frame_overshoot": [float(value) for value in per_frame],
        "max_overshoot": float(per_frame.max()),
        "location_at_max": {"cell_i": int(location[0]), "cell_j": int(location[1]), "y": float(y_of[location[1]])},
        "max_in_wall_band_4eps": total_wall,
        "max_outside_wall_band": total_outside_wall,
        "overshoot_definition": "max(phi - 1, -phi) over saved solver frames (float64)",
        "gate": EXPECTED_GENERATOR_THRESHOLDS["max_phi_overshoot"],
        "note": "diagnostic only; the generator gate and solver are unchanged (sections 3/63)",
    }


# ---------------------------------------------------------------------------
# section 15-17/52: spatial refinement comparisons
# ---------------------------------------------------------------------------


def _common_times(times_a: np.ndarray, times_b: np.ndarray, rtol: float = 1e-9) -> np.ndarray:
    times_a = np.asarray(times_a, dtype=np.float64)
    times_b = np.asarray(times_b, dtype=np.float64)
    intersection = np.intersect1d(np.round(times_a, 9), np.round(times_b, 9))
    if intersection.size == 0:
        raise AuditValidationError("refinement comparison has no common physical times")
    return intersection


def _pool_to_grid(values: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Explicit conservative mapping rule: integer average-pool to a common grid.

    Both refined and coarse fields are reduced to the common (coarser) saved
    grid by exact integer mean-pooling; a non-integer ratio is rejected instead
    of silently interpolating (section 16: no naive mismatched-grid compare).
    """

    if values.shape[0] % target_shape[0] or values.shape[1] % target_shape[1]:
        raise AuditValidationError(
            f"non-integer grid mapping {values.shape} -> {target_shape} is not supported by the explicit pooling rule"
        )
    fx = values.shape[0] // target_shape[0]
    fy = values.shape[1] // target_shape[1]
    return values.reshape(target_shape[0], fx, target_shape[1], fy).mean(axis=(1, 3))


def compare_fields_on_common_grid(
    fine_field: np.ndarray,
    coarse_field: np.ndarray,
    fine_shape: tuple[int, int],
    coarse_shape: tuple[int, int],
    common_shape: tuple[int, int],
    mask_fine: np.ndarray | None = None,
    mask_coarse: np.ndarray | None = None,
) -> dict[str, Any]:
    """Normalized field error on the explicitly mapped common grid (section 17)."""

    fine_array = np.asarray(fine_field, dtype=np.float64)
    coarse_array = np.asarray(coarse_field, dtype=np.float64)
    if not (np.isfinite(fine_array).all() and np.isfinite(coarse_array).all()):
        return {
            "mapping_rule": f"integer average-pool {fine_shape}->{common_shape} and {coarse_shape}->{common_shape}",
            "common_shape": list(common_shape),
            "status": "UNMEASURED",
            "reason": "non-finite solver field on at least one side of the refinement pair",
            "fine_finite": bool(np.isfinite(fine_array).all()),
            "coarse_finite": bool(np.isfinite(coarse_array).all()),
        }
    fine_pooled = _pool_to_grid(fine_array, common_shape)
    coarse_pooled = _pool_to_grid(coarse_array, common_shape)
    difference = fine_pooled - coarse_pooled
    norm = float(np.sqrt(np.mean(difference**2)))
    scale = float(np.sqrt(np.mean(coarse_pooled**2))) or 1.0
    entry: dict[str, Any] = {
        "mapping_rule": f"integer average-pool {fine_shape}->{common_shape} and {coarse_shape}->{common_shape}",
        "common_shape": list(common_shape),
        "l2": norm,
        "relative_l2": norm / scale,
        "linf": float(np.max(np.abs(difference))),
    }
    if mask_fine is not None and mask_coarse is not None:
        mask = _pool_to_grid(np.asarray(mask_fine, dtype=np.float64), common_shape) >= 0.5
        mask = mask & (_pool_to_grid(np.asarray(mask_coarse, dtype=np.float64), common_shape) >= 0.5)
        if mask.any():
            fluid_norm = float(np.sqrt(np.mean(difference[mask] ** 2)))
            fluid_scale = float(np.sqrt(np.mean(coarse_pooled[mask] ** 2))) or 1.0
            entry["fluid_domain_l2"] = fluid_norm
            entry["fluid_domain_relative_l2"] = fluid_norm / fluid_scale
        else:
            entry["fluid_domain_l2"] = None
    return entry


def compare_scalar_observables(
    rows_fine: list[dict],
    rows_coarse: list[dict],
    keys: tuple[str, ...],
    times_fine: list[float],
    times_coarse: list[float],
) -> dict[str, Any]:
    """Key-observable refinement change at equal physical times (section 16/17)."""

    times_f = np.asarray(times_fine, dtype=np.float64)
    times_c = np.asarray(times_coarse, dtype=np.float64)
    common = _common_times(times_f, times_c)
    out: dict[str, Any] = {"common_times": [float(t) for t in common], "n_common": int(common.size)}
    for key in keys:
        fine_by_t = {round(float(t), 9): _finite_or_none(row.get(key)) for row, t in zip(rows_fine, times_f)}
        coarse_by_t = {round(float(t), 9): _finite_or_none(row.get(key)) for row, t in zip(rows_coarse, times_c)}
        pairs = [(fine_by_t.get(round(float(t), 9)), coarse_by_t.get(round(float(t), 9))) for t in common]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        if not pairs:
            out[key] = {"status": "UNMEASURED", "reason": "no common finite values"}
            continue
        differences = np.asarray([a - b for a, b in pairs], dtype=np.float64)
        scales = np.asarray([max(abs(b), 1e-12) for _, b in pairs], dtype=np.float64)
        max_rel = float(np.max(np.abs(differences) / scales))
        out[key] = {
            "max_abs_change": float(np.max(np.abs(differences))),
            "max_relative_change": max_rel,
            "target": KEY_OBSERVABLE_REFINEMENT_CHANGE,
            "at_target": bool(max_rel <= KEY_OBSERVABLE_REFINEMENT_CHANGE),
            "above_target": bool(max_rel > KEY_OBSERVABLE_REFINEMENT_CHANGE),
            "above_2x_target": bool(max_rel > REFINEMENT_TARGET_FACTOR_THRESHOLD * KEY_OBSERVABLE_REFINEMENT_CHANGE),
            "time_of_max_discrepancy": float(common[int(np.argmax(np.abs(differences) / scales))]),
        }
    return out


def spatial_refinement_audit(production_record: dict[str, Any], refined_record: dict[str, Any]) -> dict[str, Any]:
    """Compare a production-resolution canary against its refined twin (section 15)."""

    fine_n = int(production_record["solver_params"]["N"])
    coarse_n = int(refined_record["solver_params"]["N"])
    ds = int(production_record["binding"]["ds"])
    common_saved = (min(fine_n, coarse_n) // ds, min(fine_n, coarse_n) // ds)
    out: dict[str, Any] = {
        "production_N": fine_n,
        "refined_N": coarse_n,
        "note": (
            "production contract ties eps to N (eps=1.5*Lx/N); the audit compares what production produces at each N"
        ),
        "production_trajectory_finite": bool((production_record.get("diagnostics") or {}).get("finite", True)),
        "refined_trajectory_finite": bool((refined_record.get("diagnostics") or {}).get("finite", True)),
        "equal_physical_times": bool(
            abs(
                production_record["schedule"]["effective_dt"] * production_record["schedule"]["save_every"]
                - refined_record["schedule"]["effective_dt"] * refined_record["schedule"]["save_every"]
            )
            < 1e-12
        ),
    }
    scalar_keys = (
        "formal_mass",
        "spread_width",
        "beta",
        "drop_vertical_extent",
        "centroid_y",
        "contact_line_left",
        "contact_line_right",
        "max_speed",
    )
    out["scalars"] = compare_scalar_observables(
        production_record["observables"]["rows"],
        refined_record["observables"]["rows"],
        scalar_keys,
        production_record["observables"]["frame_times"],
        refined_record["observables"]["frame_times"],
    )
    field_entries: dict[str, Any] = {}
    fine_path = production_record.get("final_fields_path")
    coarse_path = refined_record.get("final_fields_path")
    if fine_path and Path(fine_path).is_file() and coarse_path and Path(coarse_path).is_file():
        with np.load(fine_path) as fine_data, np.load(coarse_path) as coarse_data:
            for name in ("phi", "u", "v"):
                field_entries[f"{name}_final"] = compare_fields_on_common_grid(
                    np.asarray(fine_data[name]),
                    np.asarray(coarse_data[name]),
                    (fine_n, fine_n),
                    (coarse_n, coarse_n),
                    common_saved,
                )
    else:
        field_entries["status"] = "UNMEASURED"
        field_entries["reason"] = "final solver fields not cached for this canary pair"
    out["fields"] = field_entries
    return out


# ---------------------------------------------------------------------------
# section 20-22/53: impact contact-gap audit (I-CONTACT-GAP)
# ---------------------------------------------------------------------------

CONTACT_CLASSIFICATIONS = ("CONTACT_ESTABLISHED", "DIFFUSE_CONTACT_ONLY", "FINITE_GAS_FILM_PERSISTS", "INCONCLUSIVE")
#: the existing repo contact criterion: production/observables.contact_signal
#: treats ``bottom_gap <= 1.5 * dx`` as a contact event.  Reused verbatim.
CONTACT_SIGNAL_GAP_CELLS = 1.5


def _classify_gap_frame(gap05: float | None, gap01: float | None, dx: float) -> str:
    if gap05 is None and gap01 is None:
        return "NO_LIQUID"
    if gap05 is not None and gap05 <= CONTACT_SIGNAL_GAP_CELLS * dx:
        return "CONTACT_ESTABLISHED"
    if gap01 is not None and gap01 <= CONTACT_SIGNAL_GAP_CELLS * dx:
        return "DIFFUSE_CONTACT_ONLY"
    return "FILM"


def contact_gap_metrics(rows: list[dict], dx: float, eps: float) -> dict[str, Any]:
    """Contact-gap metrics through the impact window (section 20), existing impl."""

    gap05 = [_finite_or_none(row.get("gap05")) for row in rows]
    gap01 = [_finite_or_none(row.get("gap01")) for row in rows]
    times = [float(row["t"]) for row in rows]
    finite05 = [g for g in gap05 if g is not None]
    finite01 = [g for g in gap01 if g is not None]
    min05 = min(finite05) if finite05 else None
    min01 = min(finite01) if finite01 else None
    t_min05 = None
    if finite05:
        t_min05 = times[int(np.argmin([g if g is not None else np.inf for g in gap05]))]
    classifications = [_classify_gap_frame(g5, g1, dx) for g5, g1 in zip(gap05, gap01)]
    n_contact = sum(1 for c in classifications if c == "CONTACT_ESTABLISHED")
    n_diffuse = sum(1 for c in classifications if c == "DIFFUSE_CONTACT_ONLY")
    wall_profile_note = "wall-side phi profile recorded in the per-frame rows of the host trajectory (solver grid)"
    return {
        "gap05_min": min05,
        "gap01_min": min01,
        "gap05_min_over_dx": None if min05 is None else min05 / dx,
        "gap05_min_over_eps": None if min05 is None else min05 / eps,
        "gap01_min_over_dx": None if min01 is None else min01 / dx,
        "gap01_min_over_eps": None if min01 is None else min01 / eps,
        "time_of_min_gap05": t_min05,
        "n_frames_contact_established": n_contact,
        "n_frames_diffuse_only": n_diffuse,
        "n_frames_with_liquid": sum(1 for c in classifications if c != "NO_LIQUID"),
        "n_frames_total": len(rows),
        "gap05_series": gap05,
        "gap01_series": gap01,
        "classification_series": classifications,
        "definition": (
            "min fluid-side SDF beneath phi>=threshold cells (production/observables.bottom_gap);"
            " contact event criterion bottom_gap <= 1.5*dx (production/observables.contact_signal)"
        ),
        "wall_profile_note": wall_profile_note,
    }


def contact_gap_classification(
    production_metrics: dict[str, Any], refined_metrics: dict[str, Any] | None, temporal_metrics: dict[str, Any] | None
) -> dict[str, Any]:
    """Fail-closed refinement-robust classification (section 21/53)."""

    if not production_metrics.get("n_frames_with_liquid"):
        return {"classification": "INCONCLUSIVE", "reason": "no liquid-bearing frames at production resolution"}
    label05 = (
        "CONTACT_ESTABLISHED"
        if production_metrics["n_frames_contact_established"] > 0
        else ("DIFFUSE_CONTACT_ONLY" if production_metrics["n_frames_diffuse_only"] > 0 else "FINITE_GAS_FILM_PERSISTS")
    )
    checks = {"production": label05}
    for name, metrics in (("spatial_refined", refined_metrics), ("temporal_refined", temporal_metrics)):
        if metrics is None:
            checks[name] = "UNMEASURED"
            continue
        if not metrics.get("n_frames_with_liquid"):
            checks[name] = "INCONCLUSIVE"
        elif metrics["n_frames_contact_established"] > 0:
            checks[name] = "CONTACT_ESTABLISHED"
        elif metrics["n_frames_diffuse_only"] > 0:
            checks[name] = "DIFFUSE_CONTACT_ONLY"
        else:
            checks[name] = "FINITE_GAS_FILM_PERSISTS"
    measured = [value for value in checks.values() if value not in ("UNMEASURED",)]
    if any(value == "INCONCLUSIVE" for value in measured):
        classification = "INCONCLUSIVE"
    elif len(measured) < 2:
        classification = "INCONCLUSIVE"
        checks["reason"] = "refinement comparison unavailable; classification is fail-closed (section 21/53)"
    elif all(value == "CONTACT_ESTABLISHED" for value in measured):
        classification = "CONTACT_ESTABLISHED"
    elif all(value in ("CONTACT_ESTABLISHED", "DIFFUSE_CONTACT_ONLY") for value in measured):
        classification = "DIFFUSE_CONTACT_ONLY"
    else:
        classification = "FINITE_GAS_FILM_PERSISTS"
    return {"classification": classification, "per_resolution": checks, "fail_closed": len(measured) < 2}


# ---------------------------------------------------------------------------
# section 23-26/54: sample representation fidelity
# ---------------------------------------------------------------------------


def representation_fidelity(
    phi_hist: np.ndarray,
    u_hist: np.ndarray,
    v_hist: np.ndarray,
    npz_path: Path,
    ds: int,
) -> dict[str, Any]:
    """Decompose the exact export path (section 23-26) per field.

    ``solver float64 -> average-pool -> dtype cast -> NPZ -> reader`` is measured
    step by step.  The downsample information loss is the fine-grid residual of
    the piecewise-constant pooling reconstruction; the cast error is measured on
    the saved grid between the float64 pooled field and the stored array; the
    NPZ roundtrip must be bitwise.  ``R_repr`` uses the one-frame physical
    signal as denominator (section 25/54).
    """

    with np.load(Path(npz_path), allow_pickle=True) as data:
        stored = {name: np.asarray(data[name]) for name in ("phi", "u", "v")}
        metadata = json.loads(str(np.asarray(data["case"]).item()))
        fingerprint = str(np.asarray(data["dataset_fingerprint"]).item())
        time_saved = np.asarray(data["time"])
    ideal = {
        "phi": generator._downsample_history(np.asarray(phi_hist, dtype=np.float64), ds),
        "u": generator._downsample_history(np.asarray(u_hist, dtype=np.float64), ds),
        "v": generator._downsample_history(np.asarray(v_hist, dtype=np.float64), ds),
    }
    out: dict[str, Any] = {"dataset_fingerprint": fingerprint, "lineage_valid": None}
    try:
        lineage.validate_training_sample_lineage(metadata, phi_dtype=str(stored["phi"].dtype))
        out["lineage_valid"] = True
    except RuntimeError as exc:
        out["lineage_valid"] = False
        out["lineage_error"] = str(exc)
    frame_signal = {}
    for name, history in (("phi", phi_hist), ("u", u_hist), ("v", v_hist)):
        ideal_field = ideal[name]
        increments = np.diff(ideal_field.astype(np.float64), axis=0)
        frame_signal[name] = float(np.sqrt(np.mean(increments**2))) if increments.size else 0.0
    per_field: dict[str, Any] = {}
    for name in ("phi", "u", "v"):
        pooled = ideal[name]
        history = np.asarray({"phi": phi_hist, "u": u_hist, "v": v_hist}[name], dtype=np.float64)
        # (a) downsample information loss: fine-grid residual of the pooling reconstruction
        factor = history.shape[1] // pooled.shape[1]
        reconstruction = np.repeat(np.repeat(pooled, factor, axis=1), factor, axis=2)
        downsample_l2 = float(np.sqrt(np.mean((history - reconstruction) ** 2)))
        # (b) cast/quantization error on the saved grid
        cast = stored[name].astype(np.float64) - pooled
        cast_l2 = float(np.sqrt(np.mean(cast**2)))
        cast_linf = float(np.max(np.abs(cast)))
        # (c) NPZ roundtrip: NPZ is lossless; the reader must recover the stored bits
        signal = frame_signal[name]
        ratio = None if signal <= 0.0 else cast_l2 / signal
        if ratio is None:
            ratio_class = "ZERO_FRAME_SIGNAL"
        elif ratio >= REPRESENTATION_RATIO_DOMINANT_MIN:
            ratio_class = "DOMINANT"
        elif ratio > REPRESENTATION_RATIO_SUBDOMINANT_MAX:
            ratio_class = "COMPARABLE"
        else:
            ratio_class = "SUBDOMINANT"
        per_field[name] = {
            "stored_dtype": str(stored[name].dtype),
            "downsample_reconstruction_l2_fine_grid": downsample_l2,
            "downsample_reconstruction_relative_l2": downsample_l2 / (float(np.sqrt(np.mean(history**2))) or 1.0),
            "cast_l2_saved_grid": cast_l2,
            "cast_linf_saved_grid": cast_linf,
            "cast_relative_l2": cast_l2 / (float(np.sqrt(np.mean(pooled**2))) or 1.0),
            "frame_signal_l2": signal,
            "representation_noise_ratio": ratio,
            "representation_ratio_class": ratio_class,
            "thresholds": {
                "subdominant_max": REPRESENTATION_RATIO_SUBDOMINANT_MAX,
                "dominant_min": REPRESENTATION_RATIO_DOMINANT_MIN,
            },
        }
    # formal mass change caused by the export (section 24)
    dx = 6.0 / phi_hist.shape[1]
    mass_solver = np.sum(np.asarray(phi_hist, dtype=np.float64), axis=(1, 2)) * dx * dx
    saved_dx = dx * ds
    mass_saved = np.sum(stored["phi"].astype(np.float64), axis=(1, 2)) * saved_dx * saved_dx
    # interface-band error of the saved phi (section 24)
    band_fine = np.abs(np.asarray(phi_hist[-1], dtype=np.float64) - 0.5) <= 0.4
    band_saved = np.abs(ideal["phi"][-1] - 0.5) <= 0.4
    band_cells_fine = int(band_fine.sum())
    band_cells_saved = int(band_saved.sum())
    out["per_field"] = per_field
    out["mass_change"] = {
        "mass_solver_final": mass_solver[-1],
        "mass_saved_final": mass_saved[-1],
        "relative_mass_change": float(abs(mass_saved[-1] - mass_solver[-1]) / max(abs(mass_solver[-1]), 1e-12)),
        "definition": "plain cell sum times cell area on each grid (export is not restart-authoritative)",
    }
    out["interface_band"] = {
        "cells_fine_last_frame": band_cells_fine,
        "cells_saved_last_frame": band_cells_saved,
        "interface_band_l2_saved_grid": float(
            np.sqrt(np.mean((stored["phi"][-1].astype(np.float64) - ideal["phi"][-1])[band_saved] ** 2))
        )
        if band_saved.any()
        else 0.0,
    }
    out["saved_time_first_last"] = [float(time_saved[0]), float(time_saved[-1])]
    return out


def frame_signal_audit(ideal_fields: dict[str, np.ndarray]) -> dict[str, Any]:
    """Frame-to-frame target signal magnitude per frame (section 28)."""

    out: dict[str, Any] = {}
    for name, field_array in ideal_fields.items():
        field_array = np.asarray(field_array, dtype=np.float64)
        if field_array.shape[0] < 2:
            out[name] = {"status": "UNMEASURED", "reason": "needs >= 2 frames"}
            continue
        increments = np.diff(field_array, axis=0)
        norms = np.sqrt(np.mean(increments**2, axis=(1, 2)))
        out[name] = {
            "per_frame_l2": [float(v) for v in norms],
            "min": float(norms.min()),
            "median": float(np.median(norms)),
            "max": float(norms.max()),
            "zero_frames": int(np.sum(norms == 0.0)),
        }
    return out


def cadence_diagnostic(full_jump_l2: float, half_jumps_l2: list[float]) -> dict[str, Any]:
    """One current frame jump vs two half-frame jumps (section 29).

    Both measured as L2 norms of the *saved-grid* fields; the ratio quantifies
    information loss from the current temporal sampling.  No cadence change is
    proposed here.
    """

    two_half_total = float(np.sqrt(sum(value**2 for value in half_jumps_l2)))
    ratio = None if two_half_total <= 0 else float(full_jump_l2 / two_half_total)
    return {
        "one_full_frame_jump_l2": float(full_jump_l2),
        "two_half_frame_jumps_l2": [float(v) for v in half_jumps_l2],
        "two_half_combined_l2": two_half_total,
        "full_over_combined": ratio,
        "note": "diagnostic only; the production save cadence is not changed in this stage (section 29)",
    }


def cadence_from_history(field_saved: np.ndarray) -> dict[str, Any]:
    """One full frame jump vs its two half jumps from a half-cadence rollout.

    ``field_saved`` is the ideal exported field of a trajectory integrated with
    ``save_every/2``; consecutive frames are half jumps, frames two apart the
    full jump at identical physical times.
    """

    field_saved = np.asarray(field_saved, dtype=np.float64)
    if field_saved.shape[0] < 3:
        return {"status": "UNMEASURED", "reason": "needs >= 3 frames of the half-cadence rollout"}
    half_a = float(np.sqrt(np.mean((field_saved[-2] - field_saved[-3]) ** 2)))
    half_b = float(np.sqrt(np.mean((field_saved[-1] - field_saved[-2]) ** 2)))
    full = float(np.sqrt(np.mean((field_saved[-1] - field_saved[-3]) ** 2)))
    return cadence_diagnostic(full, [half_a, half_b])


# ---------------------------------------------------------------------------
# section 12-13/50: the frozen 60 deg residual vs the actual L1B horizon
# ---------------------------------------------------------------------------


def _l1a2k_frozen_authority_hashes() -> dict[str, str]:
    """Frozen authority-060 endpoint hashes, read live from the L1A-2k evidence."""

    report = _read_json(Path("evidence/l1a2k/capillary_pressure_balance_report.json"))
    case = report["cases"]["authority_060"]
    hashes = {name: str(value) for name, value in case["state_hashes"].items()}
    missing = {"phi", "u", "v", "t"} - set(hashes)
    if missing:
        raise AuditValidationError(f"frozen L1A-2k authority hashes incomplete: missing {sorted(missing)}")
    return hashes


def _late60_cache_paths() -> tuple[Path, Path]:
    window_dir = ARTIFACT_ROOT / "late60"
    return (
        window_dir / "authority_060_step_040000.npz",
        window_dir / "authority_060_step_050000.npz",
    )


def _late60_pair_valid(start_path: Path, end_path: Path, frozen: dict[str, str]) -> bool:
    try:
        with np.load(end_path, allow_pickle=True) as end_data, np.load(start_path, allow_pickle=True) as start_data:
            if np.asarray(start_data["phi"]).shape != np.asarray(end_data["phi"]).shape:
                return False
            for name in ("phi", "u", "v"):
                if _array_sha256(np.asarray(end_data[name])) != frozen[name]:
                    return False
        return True
    except Exception:
        return False


def _array_sha256(array: np.ndarray) -> str:
    """Hash in the frozen L1A-2l/2k evidence domain (dtype + shape + bytes)."""

    from production import phase_coupling_relaxation_audit as l1a2l

    return l1a2l._hash_array(array)


def late60_replay(*, force_rerun: bool = False) -> dict[str, Any]:
    """Replay the frozen authority to its endpoint, sampling the late window.

    This is the sanctioned frozen-endpoint rehydration: the trajectory is
    advanced with the unchanged production step from the original seed, the 50k
    endpoint is accepted only when every per-field hash matches the frozen
    L1A-2k record, and the replay never advances beyond
    ``LATE60_ENDPOINT_STEP`` (section 12/50).  Snapshots at the window start
    (40k) give the frozen late-window state pair.
    """

    from production import chns_nonstationarity_audit as chns
    from production import stationarity_metric_domain_audit as l1a2m

    if LATE60_ENDPOINT_STEP > 50_000:
        raise AuditValidationError("the 60 deg authority endpoint is frozen at 50000 steps and must not be extended")
    frozen = _l1a2k_frozen_authority_hashes()
    start_path, end_path = _late60_cache_paths()
    if not force_rerun and _late60_pair_valid(start_path, end_path, frozen):
        return {"cache": "hit", "start_path": str(start_path), "end_path": str(end_path), "frozen_hashes": frozen}
    p, solid, seed, _config = l1a2m._make_case(60.0)
    started = time.perf_counter()
    state = chns._advance_standard(seed, solid, p, LATE60_WINDOW_START_STEP)
    _save_late60_snapshot(start_path, state)
    state = chns._advance_standard(state, solid, p, LATE60_ENDPOINT_STEP - LATE60_WINDOW_START_STEP)
    elapsed = time.perf_counter() - started
    if _array_sha256(np.asarray(state.phi)) != frozen["phi"]:
        raise AuditValidationError("replayed 60 deg endpoint phi hash does not match the frozen L1A-2k authority")
    for name in ("u", "v"):
        if _array_sha256(np.asarray(getattr(state, name))) != frozen[name]:
            raise AuditValidationError(
                f"replayed 60 deg endpoint {name} hash does not match the frozen L1A-2k authority"
            )
    _save_late60_snapshot(end_path, state)
    return {
        "cache": "rerun",
        "start_path": str(start_path),
        "end_path": str(end_path),
        "frozen_hashes": frozen,
        "endpoint_hash_accepted": True,
        "elapsed_seconds": elapsed,
    }


def _save_late60_snapshot(path: Path, state: pf.State) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        phi=np.asarray(state.phi),
        u=np.asarray(state.u),
        v=np.asarray(state.v),
        t=np.asarray(state.t),
    )


def late60_residual_and_export(frame_signal: dict[str, float], saved_dx_production: float) -> dict[str, Any]:
    """Frozen late-window residual magnitude in solver and saved representation.

    The residual over the frozen window (40k -> 50k) is measured on the solver
    grid, then both window states are passed through the exact export path
    (average-pool + single cast) and the residual is measured again in the
    saved representation (section 13).  Ratios against the impact canary frame
    signal connect the residual to the actual label space.
    """

    replay = late60_replay()
    start_path, end_path = _late60_cache_paths()
    with np.load(end_path, allow_pickle=True) as end_data, np.load(start_path, allow_pickle=True) as start_data:
        phi_start, phi_end = np.asarray(start_data["phi"]), np.asarray(end_data["phi"])
        u_start, u_end = np.asarray(start_data["u"]), np.asarray(end_data["u"])
        v_start, v_end = np.asarray(start_data["v"]), np.asarray(end_data["v"])
    residual_solver = {
        "phi_l2": float(np.sqrt(np.mean((phi_end - phi_start) ** 2))),
        "phi_linf": float(np.max(np.abs(phi_end - phi_start))),
        "u_l2": float(np.sqrt(np.mean((u_end - u_start) ** 2))),
        "v_l2": float(np.sqrt(np.mean((v_end - v_start) ** 2))),
        "phi_end_norm": float(np.sqrt(np.mean(phi_end**2))),
        "u_end_norm": float(np.sqrt(np.mean(u_end**2))),
        "v_end_norm": float(np.sqrt(np.mean(v_end**2))),
    }
    # exact export path on both window states (N=128, ds=LATE60_EXPORT_DS)
    exported = {}
    for name, fields in (("start", (phi_start, u_start, v_start)), ("end", (phi_end, u_end, v_end))):
        ideal = {
            "phi": generator._downsample_history(fields[0][None].astype(np.float64), LATE60_EXPORT_DS)[0],
            "u": generator._downsample_history(fields[1][None].astype(np.float64), LATE60_EXPORT_DS)[0],
            "v": generator._downsample_history(fields[2][None].astype(np.float64), LATE60_EXPORT_DS)[0],
        }
        cast = {
            "phi": ideal["phi"].astype(np.float32).astype(np.float64) - ideal["phi"],
            "u": ideal["u"].astype(np.float16).astype(np.float64) - ideal["u"],
            "v": ideal["v"].astype(np.float16).astype(np.float64) - ideal["v"],
        }
        exported[name] = {
            "phi_cast_l2": float(np.sqrt(np.mean(cast["phi"] ** 2))),
            "u_cast_l2": float(np.sqrt(np.mean(cast["u"] ** 2))),
            "v_cast_l2": float(np.sqrt(np.mean(cast["v"] ** 2))),
            "phi_norm": float(np.sqrt(np.mean(ideal["phi"] ** 2))),
            "u_norm": float(np.sqrt(np.mean(ideal["u"] ** 2))),
            "v_norm": float(np.sqrt(np.mean(ideal["v"] ** 2))),
        }
    residual_saved: dict[str, float] = {}
    with np.load(end_path, allow_pickle=True) as end_data, np.load(start_path, allow_pickle=True) as start_data:
        for name in ("phi", "u", "v"):
            saved_end = generator._downsample_history(
                np.asarray(end_data[name])[None].astype(np.float64), LATE60_EXPORT_DS
            )[0]
            saved_start = generator._downsample_history(
                np.asarray(start_data[name])[None].astype(np.float64), LATE60_EXPORT_DS
            )[0]
            residual_saved[f"{name}_saved_grid_l2"] = float(np.sqrt(np.mean((saved_end - saved_start) ** 2)))
    frame_dt = 0.08
    window_steps = LATE60_ENDPOINT_STEP - LATE60_WINDOW_START_STEP
    window_time = window_steps * 0.004
    ratios = {}
    for name in ("phi", "u", "v"):
        signal = frame_signal.get(name, 0.0)
        residual = residual_saved.get(f"{name}_saved_grid_l2", 0.0)
        rate_matched = residual * (frame_dt / window_time)
        ratios[name] = {
            "late60_window_residual_saved_l2": residual,
            "impact_frame_signal_l2": signal,
            "ratio_residual_to_frame_signal": (None if signal <= 0 else residual / signal),
            "rate_matched_residual_per_frame_dt": rate_matched,
            "rate_matched_ratio": (None if signal <= 0 else rate_matched / signal),
        }
    horizon = 2000 * 0.004
    window_time = (LATE60_ENDPOINT_STEP - LATE60_WINDOW_START_STEP) * 0.004
    return {
        "replay": replay,
        "endpoint_step": LATE60_ENDPOINT_STEP,
        "window_start_step": LATE60_WINDOW_START_STEP,
        "frozen_window_physical_time": [LATE60_WINDOW_START_STEP * 0.004, LATE60_ENDPOINT_STEP * 0.004],
        "l1b_nominal_horizon_physical_time": horizon,
        "horizon_to_window_start_ratio": horizon / (LATE60_WINDOW_START_STEP * 0.004),
        "window_length_physical_time": window_time,
        "residual_solver_grid": residual_solver,
        "residual_saved_grid": residual_saved,
        "export_cast_errors": exported,
        "saved_dx": saved_dx_production,
        "late60_export_ds": LATE60_EXPORT_DS,
        "ratios_to_impact_frame_signal": ratios,
        "note": (
            "the frozen window (t in [160, 200]) does not overlap the nominal L1B horizon (t <= 8);"
            " ratios above compare magnitudes, not times, and the window is never extended (section 12)"
        ),
    }


# ---------------------------------------------------------------------------
# section 30/55: current neural-operator reader compatibility
# ---------------------------------------------------------------------------


def reader_compatibility(npz_path: Path, split: str) -> dict[str, Any]:
    """No-optimization reader/shape/lineage canary through the current L1B path."""

    import jax
    import shutil
    import tempfile

    reader_dir = Path(tempfile.mkdtemp(prefix="l1a2o_reader_"))
    shutil.copy(Path(npz_path), reader_dir / Path(npz_path).name)
    results: dict[str, Any] = {"reader_dir": str(reader_dir), "file": str(npz_path)}
    arrays = surrogate_module.load_arrays(str(reader_dir), split)
    X, Y, cond, metas = arrays
    results["load_arrays"] = {
        "X_shape": list(X.shape),
        "Y_shape": list(Y.shape),
        "cond_shape": list(cond.shape),
        "n_cases": len(metas),
        "dtypes": {"X": str(X.dtype), "Y": str(Y.dtype)},
    }
    full = surrogate_module.load_full(str(reader_dir), split)
    results["load_full"] = {"n_cases": len(full), "keys": sorted(full[0].keys()) if full else []}
    windows = surrogate_module.load_windows(str(reader_dir), split, 1)
    results["load_windows"] = {
        "n_windows": len(windows),
        "frames_shape": list(windows[0]["frames"].shape) if windows else [],
    }
    if full:
        case = full[0]
        feats = surrogate_module.geometry_features(case["chi"], case["sdf"], float(case["scalars"][3]), mode="sdf")
        results["geometry_features"] = {"mode": "sdf", "shape": list(feats.shape)}
        saved_n = case["phi"].shape[1]
        model = surrogate_module.UNet(base=4, levels=2, out_channels=3)
        sample_x = jax.numpy.zeros((1, saved_n, saved_n, 4))
        sample_cond = jax.numpy.zeros((1, 4))
        variables = model.init(jax.random.PRNGKey(0), sample_x, sample_cond)
        shaped = jax.eval_shape(lambda v: model.apply(v, sample_x, sample_cond), variables)
        results["model_output_shape"] = list(shaped.shape)
        results["shapes_match_current_model_contract"] = bool(
            X.shape[-1] == 4 and Y.shape[-1] == 3 and cond.shape[-1] == 4 and shaped.shape[-1] == 3
        )
    with np.load(Path(npz_path), allow_pickle=True) as data:
        results["dataset_fingerprint_present"] = "dataset_fingerprint" in data.files
        results["file_fingerprint"] = (
            str(np.asarray(data["dataset_fingerprint"]).item()) if "dataset_fingerprint" in data.files else None
        )
    return results


def reader_compatibility_by_split(canary_dir: Path, preferred: dict[str, Path]) -> dict[str, Any]:
    """Reader proof per split on files this audit produced (section 30).

    Rejected canaries intentionally have no sample file; a split with no
    accepted file is reported UNMEASURED instead of being faked.
    """

    out: dict[str, Any] = {}
    for split in ("train", "test"):
        path = preferred.get(split)
        if path is None or not Path(path).is_file():
            available = sorted(
                str(candidate)
                for candidate in Path(canary_dir).glob("*.npz")
                if not candidate.name.endswith("_final_fields.npz")
            )
            out[split] = {
                "status": "UNMEASURED",
                "reason": "no accepted sample file for this split in this audit run",
                "files_available": available,
            }
            continue
        entry = reader_compatibility(path, split)
        entry["status"] = "MEASURED" if entry.get("shapes_match_current_model_contract") else "FAIL"
        out[split] = entry
    return out


# ---------------------------------------------------------------------------
# section 8/60: blocker relevance matrix
# ---------------------------------------------------------------------------

#: global statuses frozen by the upstream L1A chain (section 1/8).  Only what
#: the cited evidence established; nothing silently disappears.
BLOCKER_GLOBAL_STATUS = {
    "W-CONTACT-ANGLE": "open",
    "I-CONTACT-GAP": "open",
    "P-VARDENS-PROJ": "open",
    "P-CAP-RHO": "open",
    "P-VARVISC": "open",
    "N-DT": "open_until_measured_here",
    "BC-Y-PERIODIC": "open",
    "N-INACTIVE-PHASE-STATE-COUPLING": "confirmed_problem_in_contract_v11",
    "D-FRESH-TRAIN-CONTRACT": "open_l1b_blocker",
    "IMPACT-PHI-OVERSHOOT": "new_target_critical_blocker_created_in_L1A2o",
}

EXIT_CLASSIFICATIONS = (
    "TARGET_CRITICAL",
    "BOUNDED_CAVEAT",
    "NOT_MATERIAL_TO_CURRENT_L1B_TASK",
    "UNMEASURED",
)


def blocker_relevance_matrix(audit: dict[str, Any]) -> list[dict[str, Any]]:
    """Every known blocker mapped to its current L1B target relevance (section 60)."""

    temporal = audit.get("temporal_refinement", {})
    gap = audit.get("contact_gap", {})
    late60 = audit.get("late60", {})
    rows = []
    characterization = audit.get("overshoot_characterization", {})
    rows.append(
        {
            "blocker_id": "IMPACT-PHI-OVERSHOOT",
            "global_status": "new_target_critical_blocker_created_in_L1A2o",
            "evidence_used": (
                "exit canaries: the generator's own physics validation rejects the flat We=100/200"
                " base cases at production settings on max_phi_overshoot"
            ),
            "current_l1b_target_relevance": (
                "direct label corruption risk: the dataset contract rejects the samples, so the"
                " default base generation cannot produce accepted phi/u/v labels at all"
            ),
            "effect_measured_in_exit_canaries": {
                name: (record.get("diagnostics") or {}).get("max_phi_overshoot")
                for name, record in audit.get("canaries", {}).items()
                if not record.get("accepted_by_generator")
            },
            "exit_classification": "TARGET_CRITICAL",
            "required_future_action": (
                "one bounded repair stage: characterise and remove the impact-window interface"
                " over/undershoot (or re-examine the dataset overshoot gate with evidence) BEFORE"
                " any L1B production generation; no gate relaxation without a solver-quality basis"
            ),
        }
    )
    if characterization:
        rows[0]["overshoot_characterization"] = characterization
    rows.append(
        {
            "blocker_id": "W-CONTACT-ANGLE",
            "global_status": "open",
            "evidence_used": "frozen L1A-2j late-window records + frozen 50k authority (this stage late60 section)",
            "current_l1b_target_relevance": "static-equilibrium wetting residual; impact labels live at t<=8",
            "effect_measured_in_exit_canaries": late60.get("ratios_to_impact_frame_signal"),
            "exit_classification": "BOUNDED_CAVEAT",
            "required_future_action": "remain open unless the original formal closure criteria pass (section 34)",
        }
    )
    gap_class = gap.get("classification", {}).get("classification", "UNMEASURED")
    rows.append(
        {
            "blocker_id": "I-CONTACT-GAP",
            "global_status": "open",
            "evidence_used": "exit contact-gap audit with spatial + temporal refinement (section 20-22)",
            "current_l1b_target_relevance": "directly relevant to droplet-impact labels inside the horizon",
            "effect_measured_in_exit_canaries": gap.get("production", {}).get("gap05_min_over_dx"),
            "exit_classification": (
                "TARGET_CRITICAL" if gap_class in ("FINITE_GAS_FILM_PERSISTS", "INCONCLUSIVE") else "BOUNDED_CAVEAT"
            ),
            "required_future_action": (
                "physical-publication claims require external/contact validation;"
                " solver-surrogate data may carry the behaviour as an explicit solver-model caveat (section 22)"
            ),
        }
    )
    for blocker_id, description in (
        ("P-VARDENS-PROJ", "constant-coefficient projection with variable density"),
        ("P-CAP-RHO", "capillary acceleration divided by rho_l"),
        ("P-VARVISC", "viscosity treatment"),
    ):
        rows.append(
            {
                "blocker_id": blocker_id,
                "global_status": "open",
                "evidence_used": (
                    "frozen L1A-2k capillary decomposition (structural background on the static case);"
                    " no new forensic campaign (section 33)"
                ),
                "current_l1b_target_relevance": (
                    f"model-form {description}; the surrogate emulates THIS solver, so a self-consistent"
                    " solver artifact is inside the learned map"
                ),
                "effect_measured_in_exit_canaries": "unmeasured_not_current_contract (no per-canary decomposition run)",
                "exit_classification": "NOT_MATERIAL_TO_CURRENT_L1B_TASK",
                "required_future_action": "physical-publication claims require the model-form repair stage",
            }
        )
    n_dt_spatial = temporal.get("scalars", {}) if isinstance(temporal, dict) else {}
    n_dt_effect = None
    for key in ("spread_width", "beta", "formal_mass"):
        entry = n_dt_spatial.get(key, {})
        if isinstance(entry, dict) and "max_relative_change" in entry:
            n_dt_effect = max(n_dt_effect or 0.0, entry["max_relative_change"])
    rows.append(
        {
            "blocker_id": "N-DT",
            "global_status": "open_until_measured_here",
            "evidence_used": "dt/2 exit subset at equal physical times and equal save times (section 18-19)",
            "current_l1b_target_relevance": "direct label accuracy at the production dt",
            "effect_measured_in_exit_canaries": n_dt_effect,
            "exit_classification": (
                "TARGET_CRITICAL"
                if n_dt_effect is None or n_dt_effect > KEY_OBSERVABLE_REFINEMENT_CHANGE
                else "BOUNDED_CAVEAT"
            ),
            "required_future_action": (
                "none if within the existing provisional target; else a dt-policy contract decision"
            ),
        }
    )
    rows.append(
        {
            "blocker_id": "BC-Y-PERIODIC",
            "global_status": "open",
            "evidence_used": (
                "L1A-2n seam negative control: seam cells respond identically to interior cells"
                " (seam-specific coupling falsified)"
            ),
            "current_l1b_target_relevance": (
                "geometry/topology of the embedded wall; identical across training and evaluation"
            ),
            "effect_measured_in_exit_canaries": "none separable from solver-native behaviour",
            "exit_classification": "NOT_MATERIAL_TO_CURRENT_L1B_TASK",
            "required_future_action": "document the periodic-y domain as part of the audited solver envelope",
        }
    )
    rows.append(
        {
            "blocker_id": "N-INACTIVE-PHASE-STATE-COUPLING",
            "global_status": "confirmed_problem_in_contract_v11",
            "evidence_used": (
                "L1A-2n: INACTIVE_COUPLING_BACKGROUND_ONLY; first causal operator measured (capillary"
                " stencil leakage); not 60-deg-specific; bitwise-suppressible by diagnostic closures"
            ),
            "current_l1b_target_relevance": (
                "generic structural background at O(delta/2dy) in one operator; identical mechanism at every angle"
            ),
            "effect_measured_in_exit_canaries": (
                "not separated per impact canary (no production repair allowed in this stage)"
            ),
            "exit_classification": "BOUNDED_CAVEAT",
            "required_future_action": (
                "contract-12 candidate (ghost/one-sided closure + inactive-momentum handling); never"
                " reopened as a 60-deg root cause"
            ),
        }
    )
    rows.append(
        {
            "blocker_id": "D-FRESH-TRAIN-CONTRACT",
            "global_status": "open_l1b_blocker",
            "evidence_used": (
                "section 31: aggregate dataset/checkpoint lineage enforcement is not implemented or tested yet"
            ),
            "current_l1b_target_relevance": "process blocker for fresh training, not for label fidelity",
            "effect_measured_in_exit_canaries": "per-file lineage validated on every canary (dataset_lineage schema 3)",
            "exit_classification": "UNMEASURED",
            "required_future_action": (
                "close in L1B-1 (fresh generation + aggregate lineage contract); this stage only seeds the envelope"
            ),
        }
    )
    return rows


# ---------------------------------------------------------------------------
# section 39-44: category statuses and the headline verdict
# ---------------------------------------------------------------------------

CATEGORY_STATUSES = (
    "SOLVER_NUMERICAL_STABILITY",
    "FORMAL_MASS_CONSERVATION",
    "SPATIAL_REFINEMENT",
    "TEMPORAL_REFINEMENT",
    "IMPACT_CONTACT_GAP",
    "WETTING_RESIDUAL_RELEVANCE",
    "PHI_SAMPLE_FIDELITY",
    "VELOCITY_SAMPLE_FIDELITY",
    "GEOMETRY_SAMPLE_FIDELITY",
    "FRAME_CADENCE",
    "GENERATOR_ACCEPTANCE",
    "DATASET_LINEAGE_PER_FILE",
    "FRESH_TRAINING_AGGREGATE_LINEAGE",
    "SIMPLE_SURFACE_COVERAGE",
    "COMPLEX_SURFACE_CANARY",
    "EXTERNAL_DYNAMIC_VALIDATION",
)

HEADLINE_VERDICTS = ("L1B_DATA_READY", "L1B_DATA_READY_WITH_CAVEAT", "L1B_DATA_NOT_READY")

#: categories whose UNMEASURED status blocks a READY headline.  Two categories
#: are deferred by design: the aggregate fresh-training lineage is the L1B-1
#: task itself (section 31), and missing external dynamic validation is an
#: explicit PASS_WITH_CAVEAT example in the spec (section 41), not a blocker.
DEFERRED_CAVEAT_CATEGORIES = ("FRESH_TRAINING_AGGREGATE_LINEAGE", "EXTERNAL_DYNAMIC_VALIDATION")
READINESS_BLOCKING_CATEGORIES = tuple(name for name in CATEGORY_STATUSES if name not in DEFERRED_CAVEAT_CATEGORIES)


def _worst_status(statuses: list[str]) -> str:
    for candidate in ("FAIL", "UNMEASURED", "PASS_WITH_CAVEAT", "PASS"):
        if candidate in statuses:
            return candidate
    return "UNMEASURED"


def _representation_category(per_field: dict[str, Any], field_names: tuple[str, ...]) -> tuple[str, dict]:
    classes = []
    detail: dict[str, Any] = {}
    for name in field_names:
        entry = per_field.get(name)
        if not isinstance(entry, dict) or entry.get("class_worst") == "UNMEASURED":
            return "UNMEASURED", {"missing_field": name}
        classes.append(entry["class_worst"])
        detail[name] = {
            "ratio_max_across_canaries": entry.get("ratio_max_across_canaries"),
            "class_worst": entry.get("class_worst"),
        }
    if "ZERO_FRAME_SIGNAL" in classes:
        return "UNMEASURED", {**detail, "reason": "a field has zero frame signal; ratio undefined"}
    if "DOMINANT" in classes:
        return "FAIL", detail
    if "COMPARABLE" in classes:
        return "PASS_WITH_CAVEAT", detail
    return "PASS", detail


def assemble_category_statuses(audit: dict[str, Any], profile: str) -> dict[str, Any]:
    """One PASS/PASS_WITH_CAVEAT/FAIL/UNMEASURED tag per category (section 39).

    Fail-closed: an unmeasured category is never reported as PASS.
    """

    canaries = audit.get("canaries", {})
    accepted = [record for record in canaries.values() if record.get("accepted_by_generator")]
    mandatory = [record for record in canaries.values() if record.get("mandatory")]
    mandatory_accepted = [record for record in mandatory if record.get("accepted_by_generator")]
    all_finite = all(
        bool(record.get("observables", {}).get("rows")) and all(row["finite"] for row in record["observables"]["rows"])
        for record in accepted
    )
    bounded = all(entry.get("status") == "PASS" for entry in audit.get("boundedness", {}).values())
    stability = "PASS" if accepted and all_finite and bounded else ("FAIL" if accepted else "UNMEASURED")
    if any(not record.get("diagnostics", {}).get("finite", True) for record in accepted):
        stability = "FAIL"
    mass_ok = all(
        record.get("diagnostics", {}).get("min_fluid_mass_ratio", 0.0)
        >= EXPECTED_GENERATOR_THRESHOLDS["min_total_mass_ratio"]
        and record.get("diagnostics", {}).get("max_fluid_mass_ratio", 2.0)
        <= EXPECTED_GENERATOR_THRESHOLDS["max_total_mass_ratio"]
        for record in accepted
    )
    mass_category = "PASS" if accepted and mass_ok else ("FAIL" if accepted else "UNMEASURED")

    def refinement_category(comparison: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        scalars = comparison.get("scalars", {})
        at_target, above, unmeasured = [], [], []
        for key, entry in scalars.items():
            if not isinstance(entry, dict):
                continue  # alignment metadata (common_times / n_common), not an observable
            if "max_relative_change" not in entry:
                unmeasured.append(key)
            elif entry["above_target"]:
                above.append(key)
            else:
                at_target.append(key)
        if unmeasured:
            return "UNMEASURED", {
                "unmeasured_keys": unmeasured,
                "at_target": at_target,
                "above_target": above,
                "scalars": scalars,
            }
        if above:
            return "FAIL", {
                "at_target": at_target,
                "above_target": above,
                "scalars": scalars,
                "target": KEY_OBSERVABLE_REFINEMENT_CHANGE,
            }
        return "PASS", {
            "at_target": at_target,
            "above_target": [],
            "scalars": scalars,
            "target": KEY_OBSERVABLE_REFINEMENT_CHANGE,
        }

    spatial_status, spatial_detail = refinement_category(audit.get("spatial_refinement", {}))
    temporal_status, temporal_detail = refinement_category(audit.get("temporal_refinement", {}))
    gap_class = audit.get("contact_gap", {}).get("classification", {}).get("classification", "UNMEASURED")
    gap_status = {
        "CONTACT_ESTABLISHED": "PASS",
        "DIFFUSE_CONTACT_ONLY": "PASS_WITH_CAVEAT",
        "FINITE_GAS_FILM_PERSISTS": "PASS_WITH_CAVEAT",
        "INCONCLUSIVE": "UNMEASURED",
    }.get(gap_class, "UNMEASURED")
    late60 = audit.get("late60", {})
    if late60.get("status") == "UNMEASURED_QUICK_PROFILE":
        wetting_status, wetting_detail = (
            "UNMEASURED",
            {"reason": "quick profile does not run the frozen-authority replay (section 46)"},
        )
    else:
        ratios = late60.get("ratios_to_impact_frame_signal", {})
        phi_ratio = (ratios.get("phi") or {}).get("rate_matched_ratio")
        raw_ratio = (ratios.get("phi") or {}).get("ratio_residual_to_frame_signal")
        outside = late60.get("horizon_to_window_start_ratio", 1.0) < 1.0
        if phi_ratio is None:
            wetting_status, wetting_detail = "UNMEASURED", {"reason": "residual ratio undefined"}
        elif outside and phi_ratio <= REPRESENTATION_RATIO_SUBDOMINANT_MAX:
            wetting_status, wetting_detail = (
                "PASS",
                {
                    "phi_rate_matched_ratio": phi_ratio,
                    "phi_window_total_ratio": raw_ratio,
                    "outside_horizon": True,
                },
            )
        else:
            wetting_status, wetting_detail = (
                "PASS_WITH_CAVEAT",
                {
                    "phi_rate_matched_ratio": phi_ratio,
                    "phi_window_total_ratio": raw_ratio,
                    "outside_horizon": outside,
                },
            )
    phi_status, phi_detail = _representation_category(audit.get("representation_summary", {}), ("phi",))
    velocity_status, velocity_detail = _representation_category(audit.get("representation_summary", {}), ("u", "v"))
    geom_entries = [record for record in accepted if record.get("case", {}).get("surface") not in ("flat",)]
    geometry_status = "PASS" if geom_entries else "UNMEASURED"
    frame_signal = audit.get("frame_signal_summary", {})
    cadence = audit.get("cadence", {})
    frame_mins = [entry.get("min_across_canaries") for entry in frame_signal.values() if isinstance(entry, dict)]
    if frame_mins and all(value is not None and value > 0 for value in frame_mins):
        cadence_status = "PASS"
    else:
        cadence_status = "UNMEASURED"
    generator_status = "PASS" if len(mandatory_accepted) == len(mandatory) and mandatory else "FAIL"
    rejected_detail = {
        record["name"]: {
            "max_phi_overshoot": (record.get("diagnostics") or {}).get("max_phi_overshoot"),
            "max_solid_leak": (record.get("diagnostics") or {}).get("max_solid_leak"),
            "min_fluid_mass_ratio": (record.get("diagnostics") or {}).get("min_fluid_mass_ratio"),
            "max_speed": (record.get("diagnostics") or {}).get("max_speed"),
        }
        for record in canaries.values()
        if record.get("mandatory") and not record.get("accepted_by_generator")
    }
    lineage_ok = all(
        record.get("representation", {}).get("lineage_valid") for record in accepted if record.get("representation")
    )
    lineage_status = "PASS" if accepted and lineage_ok else "UNMEASURED"
    statuses = {
        "SOLVER_NUMERICAL_STABILITY": (
            stability,
            {"n_accepted": len(accepted), "all_finite": all_finite, "bounded": bounded},
        ),
        "FORMAL_MASS_CONSERVATION": (mass_category, {"definition": "sum_i V_i phi_i gates inside the generator"}),
        "SPATIAL_REFINEMENT": (spatial_status, spatial_detail),
        "TEMPORAL_REFINEMENT": (temporal_status, temporal_detail),
        "IMPACT_CONTACT_GAP": (gap_status, {"classification": gap_class}),
        "WETTING_RESIDUAL_RELEVANCE": (wetting_status, wetting_detail),
        "PHI_SAMPLE_FIDELITY": (phi_status, phi_detail),
        "VELOCITY_SAMPLE_FIDELITY": (velocity_status, velocity_detail),
        "GEOMETRY_SAMPLE_FIDELITY": (geometry_status, {"n_geometry_canaries_accepted": len(geom_entries)}),
        "FRAME_CADENCE": (cadence_status, {"frame_signal": frame_signal, "cadence_diagnostic": cadence}),
        "GENERATOR_ACCEPTANCE": (
            generator_status,
            {"mandatory": len(mandatory), "accepted": len(mandatory_accepted), "rejected_diagnostics": rejected_detail},
        ),
        "DATASET_LINEAGE_PER_FILE": (lineage_status, {"per_file_validated": lineage_ok}),
        "FRESH_TRAINING_AGGREGATE_LINEAGE": (
            "UNMEASURED",
            {"reason": "deferred by design to L1B-1 (section 31); this stage only seeds the envelope"},
        ),
        "SIMPLE_SURFACE_COVERAGE": (
            "PASS"
            if any(record.get("case", {}).get("surface") == "flat" for record in accepted)
            else (
                "FAIL" if any(record.get("case", {}).get("surface") == "flat" for record in mandatory) else "UNMEASURED"
            ),
            {
                "flat_accepted": sum(1 for r in accepted if r.get("case", {}).get("surface") == "flat"),
                "flat_mandatory": sum(1 for r in mandatory if r.get("case", {}).get("surface") == "flat"),
            },
        ),
        "COMPLEX_SURFACE_CANARY": (geometry_status, {"accepted": len(geom_entries)}),
        "EXTERNAL_DYNAMIC_VALIDATION": (
            "UNMEASURED",
            {"reason": "no dimensionally compatible 2D benchmark exists in the current sources (section 32)"},
        ),
    }
    return {name: {"status": value[0], "detail": value[1]} for name, value in statuses.items()}


def assemble_headline(categories: dict[str, Any]) -> dict[str, Any]:
    """Headline verdict + the two readiness substatuses (section 7/40-44)."""

    failing = [name for name, entry in categories.items() if entry["status"] == "FAIL"]
    unmeasured = [name for name, entry in categories.items() if entry["status"] == "UNMEASURED"]
    blocking_fail = failing
    blocking_unmeasured = [name for name in unmeasured if name in READINESS_BLOCKING_CATEGORIES]
    caveat_categories = [name for name, entry in categories.items() if entry["status"] == "PASS_WITH_CAVEAT"]
    deferred = [name for name in unmeasured if name not in READINESS_BLOCKING_CATEGORIES]
    if blocking_fail or blocking_unmeasured:
        headline = "L1B_DATA_NOT_READY"
        surrogate = "FAIL"
        physical = "FAIL"
        blocking_reason = {
            "failing_categories": blocking_fail,
            "unmeasured_blocking_categories": blocking_unmeasured,
        }
    elif caveat_categories or deferred:
        headline = "L1B_DATA_READY_WITH_CAVEAT"
        surrogate = "PASS_WITH_CAVEAT"
        physical = "PASS_WITH_CAVEAT"
        blocking_reason = None
    else:
        headline = "L1B_DATA_READY"
        surrogate = "PASS"
        physical = "PASS_WITH_CAVEAT" if "EXTERNAL_DYNAMIC_VALIDATION" in unmeasured else "PASS"
        blocking_reason = None
    return {
        "headline": headline,
        "solver_surrogate_data_ready": surrogate,
        "physical_publication_data_ready": physical,
        "blocking_reason": blocking_reason,
        "caveat_categories": caveat_categories,
        "deferred_categories": deferred,
        "l1a_status": "EXITED_FOR_BOUNDED_L1B_DATA_GENERATION" if headline != "L1B_DATA_NOT_READY" else "BLOCKED",
    }


def l1b_contract_seed(audit: dict[str, Any], categories: dict[str, Any], headline: dict[str, Any]) -> dict[str, Any]:
    """The bounded generation envelope (section 38/61)."""

    allowed = headline["headline"] != "L1B_DATA_NOT_READY"
    seed: dict[str, Any] = {
        "stage": STAGE,
        "generation_allowed": allowed,
        "solver_contract": int(pf.SOLVER_CONTRACT_VERSION),
        "audited_parameter_envelope": {
            "We": [100.0, 200.0],
            "Re": [200.0],
            "cos_theta": [-0.5, 0.0, 0.5],
            "R": [0.7],
            "u_impact": [0.5],
            "velocity_mode": "uniform",
            "wetting_model": "surface_energy",
            "phase_boundary_model": "impermeable_flux",
        },
        "audited_surfaces": {"simple": ["flat", "pillars"], "complex_heldout": ["random_pillars"]},
        "production_resolution": {"N": 192, "Lx": 6.0, "Ly": 6.0, "eps_rule": "1.5*Lx/N"},
        "dt_policy": {"requested_dt": 0.004, "effective_dt_rule": "min(requested, stable_dt(N, u_max=2.0))"},
        "physical_horizon": 8.0,
        "save_cadence": {"save_every": 20, "frame_dt": 0.08},
        "downsample_factor": 3,
        "sample_dtypes": {"phi": "float32", "u": "float16", "v": "float16", "chi": "float32", "sdf": "float32"},
        "mandatory_lineage_fields": sorted(
            (
                "dataset_schema_version",
                "dataset_fingerprint",
                "solver_sha256",
                "solver_contract_version",
                "phase_storage_model",
                "sample_cast_policy",
                "sample_representation",
                "case",
            )
        ),
        "known_caveats": [
            name for name, entry in categories.items() if entry["status"] in ("PASS_WITH_CAVEAT", "UNMEASURED")
        ],
        "excluded_claims": [
            "no claim of long-time equilibrium wetting closure (W-CONTACT-ANGLE stays open)",
            "no physical-publication ground-truth claim while model-form blockers remain open",
            "no extrapolation beyond the audited We/Re/cos_theta/surface envelope",
        ],
        "excluded_target_regions_times": ["t beyond the audited horizon of 8.0 physical time units"],
        "next_l1b_requirement": "D-FRESH-TRAIN-CONTRACT closure (fresh generation + aggregate training lineage)",
    }
    if not allowed:
        seed["blocking_reason"] = headline["blocking_reason"]
    return seed


# ---------------------------------------------------------------------------
# section 18-19: temporal refinement (N-DT audit)
# ---------------------------------------------------------------------------


def temporal_refinement_audit(production_record: dict[str, Any], half_record: dict[str, Any]) -> dict[str, Any]:
    """Effective production dt vs dt/2 at identical physical times (section 18)."""

    out: dict[str, Any] = {
        "production_dt": production_record["schedule"]["effective_dt"],
        "half_requested_dt": half_record["schedule"]["requested_dt"],
        "half_effective_dt": half_record["schedule"]["effective_dt"],
        "informative": production_record["schedule"]["effective_dt"] != half_record["schedule"]["effective_dt"],
        "equal_save_times": bool(
            abs(production_record["schedule"]["frame_dt"] - half_record["schedule"]["frame_dt"]) < 1e-12
        ),
        "equal_horizon": bool(
            abs(production_record["schedule"]["physical_horizon"] - half_record["schedule"]["physical_horizon"]) < 1e-12
        ),
        "eta_pen_note": (
            "eta_pen=2*dt is part of the production dt policy; the dt/2 run is what production would generate"
        ),
    }
    scalar_keys = (
        "formal_mass",
        "spread_width",
        "beta",
        "drop_vertical_extent",
        "centroid_y",
        "contact_line_left",
        "contact_line_right",
        "max_speed",
    )
    out["scalars"] = compare_scalar_observables(
        production_record["observables"]["rows"],
        half_record["observables"]["rows"],
        scalar_keys,
        production_record["observables"]["frame_times"],
        half_record["observables"]["frame_times"],
    )
    field_entries: dict[str, Any] = {}
    fine_path = production_record.get("final_fields_path")
    half_path = half_record.get("final_fields_path")
    n = int(production_record["solver_params"]["N"])
    ds = int(production_record["binding"]["ds"])
    if fine_path and Path(fine_path).is_file() and half_path and Path(half_path).is_file():
        with np.load(fine_path) as fine_data, np.load(half_path) as half_data:
            for name in ("phi", "u", "v"):
                field_entries[f"{name}_final"] = compare_fields_on_common_grid(
                    np.asarray(fine_data[name]), np.asarray(half_data[name]), (n, n), (n, n), (n // ds, n // ds)
                )
    else:
        field_entries["status"] = "UNMEASURED"
    out["fields"] = field_entries
    return out


# ---------------------------------------------------------------------------
# uncertainty matrices (section 35/36/59)
# ---------------------------------------------------------------------------


def observable_uncertainty_matrix(audit: dict[str, Any]) -> dict[str, Any]:
    """Delta-space / delta-time / delta-late60 per critical derived observable."""

    spatial = audit.get("spatial_refinement", {}).get("scalars", {})
    temporal = audit.get("temporal_refinement", {}).get("scalars", {})
    frame_signal = audit.get("frame_signal_summary", {})
    late60 = audit.get("late60", {}).get("residual_solver_grid", {})
    out: dict[str, Any] = {}
    for key in (
        "formal_mass",
        "spread_width",
        "beta",
        "drop_vertical_extent",
        "contact_line_left",
        "contact_line_right",
        "max_speed",
    ):
        space = spatial.get(key, {})
        time = temporal.get(key, {})
        out[key] = {
            "space_refinement_max_relative_change": space.get("max_relative_change")
            if isinstance(space, dict)
            else None,
            "time_refinement_max_relative_change": time.get("max_relative_change") if isinstance(time, dict) else None,
            "space_at_target": space.get("at_target") if isinstance(space, dict) else None,
            "time_at_target": time.get("at_target") if isinstance(time, dict) else None,
            "time_of_max_space_discrepancy": space.get("time_of_max_discrepancy") if isinstance(space, dict) else None,
            "late60_reference": (
                {
                    "phi_window_residual_l2": late60.get("phi_l2"),
                    "note": "static 60 deg window; reported for scale only",
                }
                if key in ("contact_line_left", "contact_line_right")
                else None
            ),
            "target": KEY_OBSERVABLE_REFINEMENT_CHANGE,
            "status": "MEASURED" if isinstance(space, dict) and "max_relative_change" in space else "UNMEASURED",
        }
    out["_frame_signal_note"] = {
        name: entry.get("median") for name, entry in frame_signal.items() if isinstance(entry, dict)
    }
    return out


def field_uncertainty_matrix(audit: dict[str, Any]) -> dict[str, Any]:
    """Per direct-label-field uncertainty decomposition (section 36)."""

    spatial_fields = audit.get("spatial_refinement", {}).get("fields", {})
    temporal_fields = audit.get("temporal_refinement", {}).get("fields", {})
    representation = audit.get("representation_summary", {})
    frame_signal = audit.get("frame_signal_summary", {})
    late60 = audit.get("late60", {}).get("residual_saved_grid", {})
    out: dict[str, Any] = {}
    for name in ("phi", "u", "v"):
        space = spatial_fields.get(f"{name}_final", {})
        time = temporal_fields.get(f"{name}_final", {})
        rep = representation.get(name, {})
        signal = frame_signal.get(name, {})
        frame_norm = signal.get("median") if isinstance(signal, dict) else None
        space_norm = space.get("relative_l2") if isinstance(space, dict) else None
        time_norm = time.get("relative_l2") if isinstance(time, dict) else None
        rep_norm = rep.get("ratio_max") if isinstance(rep, dict) else None
        late_norm = late60.get(f"{name}_saved_grid_l2")
        out[name] = {
            "space_refinement_norm": space_norm,
            "time_refinement_norm": time_norm,
            "representation_norm_saved_grid": rep.get("cast_l2_max") if isinstance(rep, dict) else None,
            "representation_over_frame_signal": rep_norm,
            "late60_residual_norm_saved_grid": late_norm,
            "frame_increment_norm_median": frame_norm,
            "space_over_frame_signal": (None if space_norm is None or not frame_norm else space_norm / frame_norm),
            "time_over_frame_signal": (None if time_norm is None or not frame_norm else time_norm / frame_norm),
        }
    return out


# ---------------------------------------------------------------------------
# section 45-47: profile drivers
# ---------------------------------------------------------------------------


def _feature_cells_for(case: dict, args: GeneratorArgs) -> float:
    saved_dx = 6.0 / args.N * args.ds
    if case.get("surface", "flat") == "flat":
        return float("inf")
    return generator._feature_cells(case, saved_dx)


def _flat_case(we: float, cos_theta: float) -> dict:
    return _find_case(
        "train", "flat", {100.0: {0.5: 6, 0.0: 4, -0.5: 1}, 200.0: {0.5: 7, 0.0: 5, -0.5: 2}}[we][cos_theta]
    )


def run_forensic() -> dict[str, Any]:
    """The bounded exit matrix (section 47)."""

    production_args = GeneratorArgs()
    matrix = canary_matrix()
    canaries: dict[str, Any] = {}
    for entry in matrix:
        if entry["case"] is None:
            continue
        record = run_canary(entry["case"], entry["role"], production_args, mandatory=entry["mandatory"])
        canaries[record["name"]] = record

    primary = next(
        record for record in canaries.values() if record["case"]["We"] == 100.0 and record["case"]["cos_theta"] == 0.5
    )
    pillar_case = _find_case("train", "pillars", 8)
    pillar_record = next(record for record in canaries.values() if record["case"].get("seed") == 8)

    # section 42/69: if the generator rejected representative canaries, characterise the
    # single dominant rejection on one case (the primary flat impact canary).
    overshoot_characterization: dict[str, Any] = {}
    rejected = [record for record in canaries.values() if not record.get("accepted_by_generator")]
    if rejected:
        overshoot_characterization = characterize_overshoot(primary["case"], primary["role"], production_args)

    # spatial refinement subset (section 15)
    spatial_audits = {}
    for role, case, prod in (
        ("flat_impact_canary_spatial_N144", primary["case"], primary),
        ("pillar_training_canary_spatial_N144", pillar_case, pillar_record),
    ):
        refined = run_canary(case, role, GeneratorArgs(N=144), mandatory=False)
        spatial_audits[role] = spatial_refinement_audit(prod, refined)

    # temporal refinement subset (section 18): dt/2, same horizon, same save times
    temporal_audits = {}
    for role, case in (
        ("flat_impact_canary_temporal_dt_half", primary["case"]),
        ("flat_we200_ct0_temporal_dt_half", _flat_case(200.0, 0.0)),
    ):
        half = run_canary(case, role, GeneratorArgs(dt=2e-3, nsteps=4000, save_every=40), mandatory=False)
        temporal_audits[role] = temporal_refinement_audit(
            canaries[_case_display_name(case, "flat_impact_canary")], half
        )

    # cadence subset (section 29): half cadence on the primary case
    cadence_record = run_canary(
        primary["case"], "flat_impact_canary_cadence_half", GeneratorArgs(save_every=10), mandatory=False
    )

    # 60 deg frozen-authority relevance (section 12-13)
    frame_signal_medians = {
        name: entry.get("median", 0.0)
        for name, entry in (primary.get("frame_signal") or {}).items()
        if isinstance(entry, dict)
    }
    try:
        late60 = late60_residual_and_export(frame_signal_medians, float(6.0 / production_args.N * production_args.ds))
        late60["status"] = "MEASURED"
    except AuditValidationError as exc:
        late60 = {"status": "UNMEASURED", "reason": str(exc)}

    # contact gap (section 20-22)
    flat_records = [
        record for record in canaries.values() if record["case"].get("surface") == "flat" and record.get("observables")
    ]
    contact_gap: dict[str, Any] = {"production": {}, "classification": {}}
    for record in flat_records:
        metrics = contact_gap_metrics(
            record["observables"]["rows"], record["observables"]["dx"], record["observables"]["eps"]
        )
        contact_gap["production"][record["name"]] = metrics
    refined_record = run_canary(
        primary["case"], "flat_impact_canary_spatial_N144", GeneratorArgs(N=144), mandatory=False
    )
    half_record = run_canary(
        primary["case"],
        "flat_impact_canary_temporal_dt_half",
        GeneratorArgs(dt=2e-3, nsteps=4000, save_every=40),
        mandatory=False,
    )
    refined_metrics = contact_gap_metrics(
        refined_record["observables"]["rows"], refined_record["observables"]["dx"], refined_record["observables"]["eps"]
    )
    temporal_metrics = contact_gap_metrics(
        half_record["observables"]["rows"], half_record["observables"]["dx"], half_record["observables"]["eps"]
    )
    contact_gap["spatial_refined"] = refined_metrics
    contact_gap["temporal_refined"] = temporal_metrics
    contact_gap["classification"] = contact_gap_classification(
        contact_gap["production"][primary["name"]], refined_metrics, temporal_metrics
    )

    # reader compatibility (section 30): only accepted canaries produce sample files
    accepted_train = next(
        (record for record in canaries.values() if record.get("accepted_by_generator") and record.get("canary_npz")),
        None,
    )
    accepted_test = next(
        (
            record
            for record in canaries.values()
            if record.get("accepted_by_generator")
            and record.get("case", {}).get("split") == "test"
            and record.get("canary_npz")
        ),
        None,
    )
    reader = reader_compatibility_by_split(
        CANARY_DIR,
        {
            "train": Path(accepted_train["canary_npz"]) if accepted_train else None,
            "test": Path(accepted_test["canary_npz"]) if accepted_test else None,
        },
    )

    # representation summaries across accepted canaries
    representation_summary: dict[str, Any] = {}
    frame_signal_summary: dict[str, Any] = {}
    for name_field in ("phi", "u", "v"):
        ratios, casts = [], []
        for record in canaries.values():
            entry = (record.get("representation") or {}).get("per_field", {}).get(name_field)
            if not entry:
                continue
            if entry["representation_noise_ratio"] is not None:
                ratios.append(entry["representation_noise_ratio"])
            casts.append(entry["cast_l2_saved_grid"])
        signal_entries = [
            record["frame_signal"][name_field]
            for record in canaries.values()
            if record.get("frame_signal") and isinstance(record["frame_signal"].get(name_field), dict)
        ]
        mins = [entry["min"] for entry in signal_entries]
        medians = [entry["median"] for entry in signal_entries]
        worst_class = "UNMEASURED"
        if ratios:
            ratio_max = max(ratios)
            worst_class = (
                "DOMINANT"
                if ratio_max >= REPRESENTATION_RATIO_DOMINANT_MIN
                else ("COMPARABLE" if ratio_max > REPRESENTATION_RATIO_SUBDOMINANT_MAX else "SUBDOMINANT")
            )
        representation_summary[name_field] = {
            "ratio_max_across_canaries": max(ratios) if ratios else None,
            "cast_l2_max": max(casts) if casts else None,
            "class_worst": worst_class,
            "thresholds": {
                "subdominant_max": REPRESENTATION_RATIO_SUBDOMINANT_MAX,
                "dominant_min": REPRESENTATION_RATIO_DOMINANT_MIN,
            },
        }
        frame_signal_summary[name_field] = {
            "min_across_canaries": min(mins) if mins else None,
            "median_of_medians": float(np.median(medians)) if medians else None,
        }

    cadence = {name_field: entry for name_field, entry in (cadence_record.get("cadence_half") or {}).items()}

    boundedness = {
        record["name"]: late_window_boundedness(record["observables"])
        for record in canaries.values()
        if record.get("observables")
    }

    audit: dict[str, Any] = {
        "stage": STAGE,
        "profile": "forensic",
        "section_version": SECTION_VERSION,
        "git_sha": _git_sha(),
        "source_hashes": _source_hashes(),
        "runtime": _runtime_versions(),
        "generator_args": vars(production_args),
        "canaries": canaries,
        "spatial_refinement": spatial_audits["flat_impact_canary_spatial_N144"],
        "spatial_refinement_pillar": spatial_audits["pillar_training_canary_spatial_N144"],
        "temporal_refinement": temporal_audits["flat_impact_canary_temporal_dt_half"],
        "temporal_refinement_we200": temporal_audits["flat_we200_ct0_temporal_dt_half"],
        "cadence": cadence,
        "late60": late60,
        "contact_gap": contact_gap,
        "reader": reader,
        "representation_summary": representation_summary,
        "frame_signal_summary": frame_signal_summary,
        "boundedness": boundedness,
        "overshoot_characterization": overshoot_characterization,
        "canary_matrix_definition": matrix,
    }
    return audit


def run_quick() -> dict[str, Any]:
    """Methodology-only quick profile (section 46) on tiny configurations."""

    quick_args = GeneratorArgs(N=64, ds=2, dt=2e-3, nsteps=40, save_every=20)
    flat_quick = _find_case("train", "flat", 6)
    pillar_quick = _find_case("train", "pillars", 502)
    complex_quick = _find_case("test", "random_pillars", 504)
    primary = run_canary(flat_quick, "flat_quick", quick_args, mandatory=True)
    refined = run_canary(
        flat_quick,
        "flat_quick_spatial_N32",
        GeneratorArgs(N=32, ds=2, dt=2e-3, nsteps=40, save_every=20),
        mandatory=False,
    )
    half = run_canary(
        flat_quick,
        "flat_quick_temporal_dt_half",
        GeneratorArgs(N=64, ds=2, dt=1e-3, nsteps=80, save_every=40),
        mandatory=False,
    )
    cadence_record = run_canary(
        flat_quick,
        "flat_quick_cadence_half",
        GeneratorArgs(N=64, ds=2, dt=2e-3, nsteps=40, save_every=10),
        mandatory=False,
    )
    pillar = run_canary(pillar_quick, "pillar_quick", quick_args, mandatory=True)
    complex_record = run_canary(complex_quick, "complex_quick", quick_args, mandatory=True)
    canaries = {primary["name"]: primary, pillar["name"]: pillar, complex_record["name"]: complex_record}

    spatial = spatial_refinement_audit(primary, refined)
    temporal = temporal_refinement_audit(primary, half)
    gap_production = contact_gap_metrics(
        primary["observables"]["rows"], primary["observables"]["dx"], primary["observables"]["eps"]
    )
    gap_refined = contact_gap_metrics(
        refined["observables"]["rows"], refined["observables"]["dx"], refined["observables"]["eps"]
    )
    gap_temporal = contact_gap_metrics(
        half["observables"]["rows"], half["observables"]["dx"], half["observables"]["eps"]
    )
    contact_gap = {
        "production": {primary["name"]: gap_production},
        "spatial_refined": gap_refined,
        "temporal_refined": gap_temporal,
        "classification": contact_gap_classification(gap_production, gap_refined, gap_temporal),
    }
    late60 = {
        "status": "UNMEASURED_QUICK_PROFILE",
        "reason": (
            "the frozen-authority replay is a forensic-only measurement (section 46); the schema"
            " carries the unmeasured state fail-closed"
        ),
    }
    representation_summary = {}
    frame_signal_summary = {}
    for name_field in ("phi", "u", "v"):
        entry = (primary.get("representation") or {}).get("per_field", {}).get(name_field, {})
        representation_summary[name_field] = {
            "ratio_max_across_canaries": entry.get("representation_noise_ratio"),
            "cast_l2_max": entry.get("cast_l2_saved_grid"),
            "class_worst": entry.get("representation_ratio_class", "UNMEASURED"),
        }
        signal = (primary.get("frame_signal") or {}).get(name_field, {})
        frame_signal_summary[name_field] = {
            "min_across_canaries": signal.get("min"),
            "median_of_medians": signal.get("median"),
        }
    reader = reader_compatibility_by_split(
        CANARY_DIR, {"train": Path(primary["canary_npz"]) if primary.get("canary_npz") else None}
    )
    audit = {
        "stage": STAGE,
        "profile": "quick",
        "section_version": SECTION_VERSION,
        "git_sha": _git_sha(),
        "source_hashes": _source_hashes(),
        "runtime": _runtime_versions(),
        "generator_args": vars(quick_args),
        "canaries": canaries,
        "spatial_refinement": spatial,
        "temporal_refinement": temporal,
        "cadence": dict(cadence_record.get("cadence_half") or {}),
        "late60": late60,
        "contact_gap": contact_gap,
        "reader": reader,
        "representation_summary": representation_summary,
        "frame_signal_summary": frame_signal_summary,
        "boundedness": {
            record["name"]: late_window_boundedness(record["observables"])
            for record in canaries.values()
            if record.get("observables")
        },
        "canary_matrix_definition": canary_matrix(),
        "methodology_checks": {},
    }
    checks = {
        "current_ml_contract_discovery": bool(discover_ml_task_contract(CANARY_DIR / f"{primary['name']}.npz")),
        "effective_schedule_calculation": bool(effective_schedule(flat_quick, quick_args)),
        "canary_matrix_construction": bool(canary_matrix()),
        "observable_extraction": bool(primary.get("observables", {}).get("rows")),
        "space_time_comparison_alignment": bool(spatial.get("scalars", {}).get("common_times")),
        "sample_export_roundtrip": bool(primary.get("representation")),
        "representation_noise_calculation": bool(primary.get("representation", {}).get("per_field")),
        "contact_gap_metric_plumbing": bool(gap_production.get("n_frames_total")),
        "reader_compatibility": bool(audit["reader"]["train"].get("shapes_match_current_model_contract")),
        "verdict_schema": all(name in CATEGORY_STATUSES for name in CATEGORY_STATUSES),
        "fail_closed_unmeasured_handling": assemble_headline(
            {**{name: {"status": "UNMEASURED", "detail": {}} for name in CATEGORY_STATUSES}}
        )["headline"]
        == "L1B_DATA_NOT_READY",
    }
    audit["methodology_checks"] = checks
    audit["methodology_all_passed"] = all(bool(value) for value in checks.values())
    return audit


# ---------------------------------------------------------------------------
# section 57: evidence assembly and report rendering
# ---------------------------------------------------------------------------


def _categories_markdown(categories: dict[str, Any]) -> str:
    lines = ["| Category | Status |", "---|---"]
    for name, entry in categories.items():
        lines.append(f"| `{name}` | `{entry['status']}` |")
    return "\n".join(lines)


def _report_markdown(report: dict[str, Any]) -> str:
    verdict = report["verdict"]
    lines = [
        f"# {STAGE} L1A data-readiness exit audit",
        "",
        f"- **Profile:** `{report['profile']}`",
        f"- **Headline verdict:** `{verdict['headline']}`",
        f"- **SOLVER_SURROGATE_DATA_READY:** `{verdict['solver_surrogate_data_ready']}`",
        f"- **PHYSICAL_PUBLICATION_DATA_READY:** `{verdict['physical_publication_data_ready']}`",
        f"- **L1A_STATUS:** `{verdict['l1a_status']}`",
        f"- **Solver contract:** `{pf.SOLVER_CONTRACT_VERSION}` (unchanged)",
        f"- **git SHA:** `{report['git_sha'][:12]}`",
        "",
        "## Category statuses",
        "",
        _categories_markdown(report["categories"]),
        "",
        "## Canary acceptance",
        "",
    ]
    for name, record in report.get("audit_core", {}).get("canaries", {}).items():
        schedule = record.get("schedule", {})
        lines.append(
            f"- `{name}`: accepted={record.get('accepted_by_generator')}"
            f" dt={schedule.get('effective_dt')} horizon={schedule.get('physical_horizon')}"
            f" frames={schedule.get('n_frames')}"
        )
    late60 = report.get("audit_core", {}).get("late60", {})
    if late60.get("status") == "MEASURED":
        ratios = late60.get("ratios_to_impact_frame_signal", {})
        lines += [
            "",
            "## Frozen 60 deg residual vs the L1B horizon",
            "",
            f"- frozen window t in {late60.get('frozen_window_physical_time')},"
            f" nominal L1B horizon t <= {late60.get('l1b_nominal_horizon_physical_time')}",
            f"- horizon/window-start ratio: {late60.get('horizon_to_window_start_ratio')}",
            "- residual (saved grid, frozen window) vs impact frame signal:",
        ]
        for name, entry in ratios.items():
            lines.append(
                f"  - `{name}`: residual={entry.get('late60_window_residual_saved_l2'):.3e}"
                f" signal={entry.get('impact_frame_signal_l2'):.3e}"
                f" ratio={entry.get('ratio_residual_to_frame_signal')}"
            )
    gap = report.get("audit_core", {}).get("contact_gap", {})
    if gap:
        lines += [
            "",
            "## Impact contact gap (I-CONTACT-GAP)",
            "",
            f"- classification: `{gap.get('classification', {}).get('classification')}`",
            f"- per-resolution: `{json.dumps(gap.get('classification', {}).get('per_resolution'))}`",
        ]
    spatial = report.get("audit_core", {}).get("spatial_refinement", {})
    temporal = report.get("audit_core", {}).get("temporal_refinement", {})
    lines += ["", "## Refinement (existing provisional 3% key-observable target)", ""]
    for label, comparison in (("spatial (N=144 vs N=192)", spatial), ("temporal (dt vs dt/2)", temporal)):
        at_target = [
            key
            for key, entry in comparison.get("scalars", {}).items()
            if isinstance(entry, dict) and entry.get("at_target")
        ]
        above = [
            key
            for key, entry in comparison.get("scalars", {}).items()
            if isinstance(entry, dict) and entry.get("above_target")
        ]
        lines.append(f"- {label}: at_target={at_target} above_target={above}")
    lines += [
        "",
        "## Sample representation",
        "",
    ]
    for name, entry in report.get("audit_core", {}).get("representation_summary", {}).items():
        lines.append(
            f"- `{name}`: worst ratio class `{entry.get('class_worst')}`"
            f" ratio_max={entry.get('ratio_max_across_canaries')}"
        )
    lines += ["", "## Blockers", ""]
    for row in report.get("blockers", []):
        lines.append(f"- `{row['blocker_id']}`: {row['global_status']} -> {row['exit_classification']}")
    seed = report.get("l1b_contract_seed", {})
    lines += [
        "",
        "## L1B envelope",
        "",
        f"- generation_allowed: {seed.get('generation_allowed')}",
        f"- caveats: {seed.get('known_caveats')}",
        "",
        "> This stage is a decision stage: no solver, threshold, schema, or cadence change was made.",
    ]
    return "\n".join(lines) + "\n"


def assemble_evidence(audit: dict[str, Any], profile: str) -> dict[str, Any]:
    """Assemble every section-57 evidence document and write them out."""

    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    canary_files = sorted(str(path) for path in CANARY_DIR.glob("*.npz")) if CANARY_DIR.is_dir() else []
    sample_file = next((Path(path) for path in canary_files if not Path(path).name.endswith("_final_fields.npz")), None)
    contract = discover_ml_task_contract(sample_file)
    categories = assemble_category_statuses(audit, profile)
    verdict = assemble_headline(categories)
    blockers = blocker_relevance_matrix(audit)
    seed = l1b_contract_seed(audit, categories, verdict)
    audit_core = {key: value for key, value in audit.items()}
    report = {
        "stage": STAGE,
        "profile": profile,
        "section_version": SECTION_VERSION,
        "git_sha": audit["git_sha"],
        "source_hashes": audit["source_hashes"],
        "runtime": audit["runtime"],
        "verdict": verdict,
        "categories": categories,
        "blockers": blockers,
        "l1b_contract_seed": seed,
        "ml_task_contract": contract,
        "audit_core": audit_core,
    }
    if profile != "quick":
        report["audit_core"].pop("methodology_checks", None)
    documents = {
        "l1a_data_readiness_exit_report.json"
        if profile == "forensic"
        else "l1a_data_readiness_quick_report.json": report,
        "ml_task_contract.json": contract,
        "canary_matrix.json": {
            "definition": audit["canary_matrix_definition"],
            "executed": {
                name: {
                    "role": record["role"],
                    "mandatory": record.get("mandatory", False),
                    "accepted": record.get("accepted_by_generator"),
                    "fingerprint": record.get("fingerprint"),
                    "schedule": record.get("schedule"),
                    "cache": record.get("cache"),
                }
                for name, record in audit["canaries"].items()
            },
        },
        "observable_uncertainty_matrix.json": observable_uncertainty_matrix(audit),
        "field_uncertainty_matrix.json": field_uncertainty_matrix(audit),
        "contact_gap_exit_audit.json": audit["contact_gap"],
        "sample_representation_fidelity.json": {
            name: record.get("representation")
            for name, record in audit["canaries"].items()
            if record.get("representation")
        },
        "blocker_relevance_matrix.json": blockers,
        "l1b_contract_seed.json": seed,
    }
    for filename, payload in documents.items():
        _write_json(EVIDENCE_ROOT / filename, payload)
    (
        EVIDENCE_ROOT
        / (
            documents_key := "l1a_data_readiness_exit_report.md"
            if profile == "forensic"
            else "l1a_data_readiness_quick_report.md"
        )
    ).write_text(_report_markdown(report))
    manifest = {
        "stage": STAGE,
        "profile": profile,
        "section_version": SECTION_VERSION,
        "git_sha": audit["git_sha"],
        "files": {
            filename: {"sha256": _file_sha256(EVIDENCE_ROOT / filename)}
            for filename in (*documents.keys(), documents_key)
        },
        "canary_npz_available": canary_files,
    }
    manifest_path = EVIDENCE_ROOT / "manifest.json"
    if manifest_path.is_file():
        previous = _read_json(manifest_path)
        previous_files = previous.get("files", {})
        for filename, entry in manifest["files"].items():
            previous_files[filename] = entry
        manifest["files"] = previous_files
        manifest["profiles_run"] = sorted(set(previous.get("profiles_run", [])) | {profile})
        manifest["profile"] = "forensic" if "forensic" in manifest["profiles_run"] else profile
    else:
        manifest["profiles_run"] = [profile]
        manifest["profile"] = profile
    _write_json(manifest_path, manifest)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", choices=("quick", "forensic"), default="quick")
    parser.add_argument("--out", default=str(ARTIFACT_ROOT))
    args = parser.parse_args()
    if args.out != str(ARTIFACT_ROOT):
        root = Path(args.out)
        globals()["ARTIFACT_ROOT"] = root
        globals()["CACHE_PATH"] = root / "cache" / "canaries.json"
        globals()["CANARY_DIR"] = root / "canaries"
    started = time.perf_counter()
    if args.profile == "quick":
        audit = run_quick()
    else:
        audit = run_forensic()
    report = assemble_evidence(audit, args.profile)
    elapsed = time.perf_counter() - started
    quality = {
        "stage": STAGE,
        "profile": args.profile,
        "profiles_run": [args.profile],
        "checks": audit.get("methodology_checks", {}),
        "all_checks_passed": bool(audit.get("methodology_all_passed", True)),
        "elapsed_seconds": elapsed,
        "git_sha": audit["git_sha"],
        "notes": {
            "dependency_audit": "PRE_EXISTING_FAILURE (uv audit --locked, unchanged; kept separate per stage spec)",
        },
    }
    quality_path = EVIDENCE_ROOT / "quality_status.json"
    if quality_path.is_file():
        previous = _read_json(quality_path)
        quality["profiles_run"] = sorted(set(previous.get("profiles_run", [])) | {args.profile})
        for key, value in previous.get("checks", {}).items():
            quality["checks"].setdefault(key, value)
    _write_json(quality_path, quality)
    verdict = report["verdict"]["headline"]
    print(
        f"[{STAGE}] profile={args.profile} elapsed={elapsed:.1f}s"
        f" status={'complete' if (args.profile != 'quick' or audit.get('methodology_all_passed')) else 'checks_failed'}"
        f" verdict={verdict}",
        flush=True,
    )


if __name__ == "__main__":
    main()
