"""L1A-2s diagnostic-only audit of divergence- and solid-compatible impact seeds.

This module never changes a production initializer or evolution operator. Candidate
C0/C1/C2 velocity fields are injected only into local ``pf.State`` values used by
the audit. C2 is a compact, case-specific SDF-tapered streamfunction; its phase
field, geometry, parameters, and contract-12 solver remain unchanged.

Run from ``examples/two_phase`` with ``JAX_ENABLE_X64=1``. The forensic profile
uses N=192 and is the only profile that can produce physical evidence. The quick
profile is a small-grid methodology smoke and is permanently ineligible to pass
any L1B readiness gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

STAGE = "L1A-2s"
STAGE_VERSION = "l1a2s_v1"
BASE_MAIN_SHA = "002120f3a051e638a7e85f9db4022107dbe45780"
PR21 = {
    "number": 21,
    "state": "MERGED",
    "head_sha": "8220c32c6fed24432788f28bb0d76a027c7e95df",
    "merge_sha": BASE_MAIN_SHA,
    "merged_at_utc": "2026-10-08T16:05:31Z",
    "url": "https://github.com/licy0505/hydrogym/pull/21",
}
# QUALITY-3 integration lineage. BASE_MAIN_SHA remains the original PR #21 base
# fact; this exact second-parent merge is an additional verified main state, not
# a rewrite of the L1A-2s provenance.
PR23_INTEGRATION = {
    "number": 23,
    "state": "MERGED",
    "head_sha": "ea00347c8a6cebf6cfbff36f744d39ccd60784d1",
    "merge_sha": "039f6a0d9a14ff601c76a58f5a66ce0900677de5",
    "base_sha": BASE_MAIN_SHA,
    "merged_at_utc": "2026-10-10T08:04:14Z",
    "url": "https://github.com/licy0505/hydrogym/pull/23",
}
VERIFIED_MAIN_SHAS = frozenset({BASE_MAIN_SHA, PR23_INTEGRATION["merge_sha"]})
EXPECTED_POLICY = "impact_phase_cap_dx2_v1"
CASE_NAMES = ("flat_we100_ct050", "flat_we200_ct000", "pillar_training", "complex_heldout")
CANDIDATES = ("UNIFORM_ALL_DOMAIN", "STREAMFUNCTION_LOCALIZED_V0", "SDF_TAPERED_STREAMFUNCTION_V1")
REQUESTED_DT = 0.004
DT_LEVELS = (0.002, 0.001, 0.0005)
SHORT_HORIZON = 0.24
EVENT_SCREEN_HORIZON = 0.48
DATASET_HORIZON = 8.0
FRAME_CADENCE = 0.08
CONTACT_GAP_CELLS = 1.5
APPROACH_FLOOR_FRACTION = 0.2
POSTCONTACT_WINDOW = 0.24
POSTCONTACT_SPEED_RESPONSE_FRACTION = 0.05

# Declared before observing C2. Central-D tolerance is scaled by u_impact/dx and
# anchored to 64 float32 epsilons (roundoff margin for two central-difference
# applications). Wall thresholds are separate physical no-slip diagnostics,
# chosen far below the C0/C1 control magnitudes and not borrowed from the 3% mesh
# refinement gate. FV divergence is independently scaled by U/dx.
FLOAT32_EPS = float(np.finfo(np.float32).eps)
INITIAL_ACCEPTANCE = {
    "central_D_dimensionless_Linf_max": 64.0 * FLOAT32_EPS,
    "open_face_FV_dimensionless_Linf_max": 0.01,
    "deep_solid_speed_over_u_impact_max": 32.0 * FLOAT32_EPS,
    "nearwall_chi_ge_0p01_speed_over_u_impact_max": 0.02,
    "embedded_wall_bilinear_normal_speed_over_u_impact_max": 0.02,
    "core_mean_speed_relative_error_max": 0.01,
    "region_definitions": {
        "deep_solid": "sdf < -2*dx",
        "nearwall_sdf": "abs(sdf) <= 3*dx",
        "nearwall_chi": "chi >= 0.01",
        "partial_cells": "0 < V_i < dx*dy",
        "wall_samples": "wall_measure > 0, bilinear cell-velocity sample at wall centroid",
        "liquid_mean": "sum(V_i*phi_i*v_i)/sum(V_i*phi_i), raw phi as specified",
        "core_mean": "V_i-weighted mean over phi >= 0.9",
    },
}

REPO = Path(__file__).resolve().parents[3]
TWO_PHASE = REPO / "examples" / "two_phase"
EVIDENCE_DIR = TWO_PHASE / "evidence" / "l1a2s"
ARTIFACT_DIR = TWO_PHASE / "artifacts" / "l1a2s"

import cases as cases_module  # noqa: E402

# ``generate_dataset`` lives one package level above ``production`` in this example.
# Import it by its existing module name without modifying its source.
import generate_dataset as generator  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import phasefield as pf  # noqa: E402
from production import impact_impulse_projection_audit as impulse_audit  # noqa: E402
from production import l1a_data_readiness_exit_audit as l1a_exit  # noqa: E402
from production import observables  # noqa: E402
from production import timestep_policy  # noqa: E402
from production import validation as validation_module  # noqa: E402


class AuditValidationError(RuntimeError):
    """Fail-closed protocol, source, cache, or evidence error."""


@dataclass
class CaseBundle:
    case_name: str
    case: dict[str, Any]
    N: int
    requested_dt: float
    p: Any
    solid: Any
    base: Any
    x0: float
    y0: float
    R: float
    u_impact: float
    local_surface_top: float
    initial_gap: float
    policy: dict[str, Any]


def _json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default).encode()


def _concat_text(*parts: str) -> str:
    """Keep long provenance descriptions readable in source without altering their text."""
    return "".join(parts)


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha_file(path: Path) -> str:
    if not path.is_file():
        raise AuditValidationError(f"pinned source file is missing: {path}")
    return _sha_bytes(path.read_bytes())


def _array_sha(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = _json_bytes({"dtype": array.dtype.str, "shape": list(array.shape)})
    return _sha_bytes(header + array.tobytes())


def _state_hashes(state: Any) -> dict[str, str]:
    return {name: _array_sha(np.asarray(getattr(state, name))) for name in ("phi", "u", "v")}


def _geometry_hash(solid: Any) -> str:
    """Hash all geometry authorities that can affect phase transport or wall metrics."""
    payload = {}
    for name in (
        "sdf",
        "chi",
        "chi_hard",
        "wall_area",
        "wall_normal_x",
        "wall_normal_y",
        "wall_distance",
    ):
        payload[f"solid.{name}"] = _array_sha(np.asarray(getattr(solid, name)))
    for name in (
        "volume",
        "alpha",
        "centroid_x",
        "centroid_y",
        "aperture_x",
        "aperture_y",
        "wall_measure",
        "wall_normal_x",
        "wall_normal_y",
        "wall_centroid_x",
        "wall_centroid_y",
    ):
        payload[f"geometry.{name}"] = _array_sha(np.asarray(getattr(solid.geometry, name)))
    return _sha_bytes(_json_bytes(payload))


def _git(command: list[str], *, required: bool = True) -> str | None:
    try:
        return subprocess.check_output(command, cwd=REPO, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception as exc:
        if required:
            raise AuditValidationError(f"git provenance command failed: {' '.join(command)}: {exc}") from exc
        return None


def _verified_main_lineage(latest_main: str | None) -> bool:
    if latest_main == BASE_MAIN_SHA:
        return True
    if latest_main != PR23_INTEGRATION["merge_sha"]:
        return False
    parents = _git(["git", "show", "-s", "--format=%P", latest_main], required=False)
    if not parents:
        return False
    parent_set = set(parents.split())
    return {BASE_MAIN_SHA, PR23_INTEGRATION["head_sha"]}.issubset(parent_set)


def _current_source_paths() -> dict[str, Path]:
    return {
        "phasefield": TWO_PHASE / "phasefield.py",
        "generate_dataset": TWO_PHASE / "generate_dataset.py",
        "cases": TWO_PHASE / "cases.py",
        "timestep_policy": TWO_PHASE / "production" / "timestep_policy.py",
        "impact_impulse_projection_audit": TWO_PHASE / "production" / "impact_impulse_projection_audit.py",
        "observables": TWO_PHASE / "production" / "observables.py",
        "l1a_data_readiness_exit_audit": TWO_PHASE / "production" / "l1a_data_readiness_exit_audit.py",
        "l1a2r_final_report_md": TWO_PHASE / "evidence" / "l1a2r" / "impact_impulse_causal_report.md",
        "l1a2r_final_report_json": TWO_PHASE / "evidence" / "l1a2r" / "impact_impulse_causal_report.json",
    }


def _source_hashes() -> dict[str, str]:
    hashes = {name: _sha_file(path) for name, path in _current_source_paths().items()}
    prior = json.loads((TWO_PHASE / "evidence" / "l1a2r" / "manifest.json").read_text())
    hashes["impact_impulse_projection_audit_at_l1a2r"] = prior["binding"]["source_hashes"][
        "impact_impulse_projection_audit"
    ]
    hashes["phasefield_at_l1a2r"] = prior["binding"]["source_hashes"]["phasefield"]
    hashes["generate_dataset_at_l1a2r"] = prior["binding"]["source_hashes"]["generate_dataset"]
    hashes["cases_at_l1a2r"] = prior["binding"]["source_hashes"]["cases"]
    hashes["timestep_policy_at_l1a2r"] = prior["binding"]["source_hashes"]["timestep_policy"]
    return hashes


def _frozen_source_gate() -> dict[str, Any]:
    current = _source_hashes()
    prior_manifest = json.loads((TWO_PHASE / "evidence" / "l1a2r" / "manifest.json").read_text())
    prior = prior_manifest["binding"]["source_hashes"]
    checks = {
        "phasefield_unchanged_since_l1a2r": current["phasefield"] == prior["phasefield"],
        "generator_unchanged_since_l1a2r": current["generate_dataset"] == prior["generate_dataset"],
        "cases_unchanged_since_l1a2r": current["cases"] == prior["cases"],
        "timestep_policy_unchanged_since_l1a2r": current["timestep_policy"] == prior["timestep_policy"],
        "contract12": int(pf.SOLVER_CONTRACT_VERSION) == 12,
        "policy": timestep_policy.DEFAULT_POLICY_NAME == EXPECTED_POLICY,
        "policy_matches_contract": pf.SOLVER_CONTRACT_12_TRAJECTORY_POLICY == EXPECTED_POLICY,
        "l1a2r_report_present": current["l1a2r_final_report_md"]
        == "5a94c9d03af09fbcdd57b5721e618340dbb35d3351593a0c881c791ba03a8a71",
    }
    try:
        latest_main = _git(["git", "rev-parse", "origin/main"])
    except AuditValidationError:
        latest_main = None
    checks["origin_main_is_verified_merge_base"] = _verified_main_lineage(latest_main)
    if not all(checks.values()):
        failed = [name for name, ok in checks.items() if not ok]
        raise AuditValidationError(f"L1A-2s prerequisite/source freeze failed: {failed}; latest_main={latest_main}")
    return {
        "checks": checks,
        "source_hashes": current,
        "latest_main_sha": latest_main,
        "integration_lineage": {
            "original_base_main_sha": BASE_MAIN_SHA,
            "accepted_latest_main_shas": sorted(VERIFIED_MAIN_SHAS),
            "pr23": PR23_INTEGRATION,
            "main_lineage_verified": bool(checks["origin_main_is_verified_merge_base"]),
        },
    }


def _canary_cases() -> dict[str, dict[str, Any]]:
    cases = impulse_audit.study_cases()
    if tuple(cases) != CASE_NAMES:
        raise AuditValidationError(f"L1A-2r canary order/identity changed: {tuple(cases)}")
    return cases


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def _progress(message: str) -> None:
    print(f"[{STAGE}] {message}", flush=True)


def _preflight_payload() -> dict[str, Any]:
    frozen = _frozen_source_gate()
    cases = _canary_cases()
    pr = dict(PR21)
    return {
        "stage": STAGE,
        "stage_version": STAGE_VERSION,
        "prerequisite": {
            "pr21": pr,
            "verified_merge_sha": BASE_MAIN_SHA,
            "latest_main_sha_at_execution": frozen["latest_main_sha"],
            "main_is_at_or_after_pr21_merge": frozen["latest_main_sha"] in VERIFIED_MAIN_SHAS,
            "integration_lineage": frozen["integration_lineage"],
            "branch_session_policy": (
                "Arena binds this execution to arena/3d47375a-hydrogym; no l1a-2s branch switch was made"
            ),
            "worktree_branch": _git(["git", "branch", "--show-current"]),
            "head_at_preflight": _git(["git", "rev-parse", "HEAD"]),
        },
        "frozen_l1a2r_findings": {
            "stage": "COMPLETE / PR #21 merged",
            "solver_contract": 12,
            "l1b_data": "L1B_DATA_NOT_READY",
            "l1a_status": "BLOCKED",
            "root_cause": "BRINKMAN_PROJECTION_COUPLING",
            "direct_global_momentum_loss": "FALSIFIED",
            "local_impulse_redistribution": "SUPPORTED",
            "discrete_operator_mismatch_tested_identity": "FALSIFIED",
            "p_vardens_proj": "OPEN",
            "n_dt": "TARGET_CRITICAL",
            "spatial_refinement": "FAIL",
            "w_contact_angle": "OPEN",
            "d_fresh_train_contract": "OPEN",
            "frozen_flat_events": {
                "flat_we100_ct050": {
                    "verdict": "IMPACT_AUTHENTICATED",
                    "contact_t": 0.174,
                    "v_liquid": -0.300847070329903,
                    "retention": 0.5944755277202775,
                },
                "flat_we200_ct000": {
                    "verdict": "IMPACT_AUTHENTICATED",
                    "contact_t": 0.186,
                    "v_liquid": -0.28997363695536366,
                    "retention": 0.5729735846495383,
                },
            },
            "frozen_short_window_only": {
                "pillar_training": "NO_CONTACT_OR_HOVER within short L1A-2r window; late retention 0.20479536401723591",
                "complex_heldout": "NO_CONTACT_OR_HOVER within short L1A-2r window; late retention 0.2020809365247083",
                "not_extrapolated_to_T8": True,
            },
            "negative_controls_language": {
                "STREAMFUNCTION_LOCALIZED": "near-solenoidal control, not verified solid-compatible",
                "SOLID_COMPATIBLE_INITIAL": "zero-in-solid control, not divergence-free physical seed",
            },
            "l1a2r_report_sha256": frozen["source_hashes"]["l1a2r_final_report_md"],
            "l1a2r_report_json_sha256": frozen["source_hashes"]["l1a2r_final_report_json"],
        },
        "case_definitions": cases,
        "source_freeze": frozen,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "jax": jax.__version__,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "numpy": np.__version__,
        },
    }


def _source_operator_map() -> dict[str, Any]:
    source = (TWO_PHASE / "phasefield.py").read_text()
    required_anchors = {
        "central_D": "_ddx(u_new, p.dx) + _ddy(v_new, p.dy)",
        "periodic_D_x": "def _ddx(f, dx):",
        "periodic_D_y": "def _ddy(f, dy):",
        "periodic_poisson": "def poisson_solve(rhs, m2):",
        "brinkman": "damp = 1.0 / (1.0 + dt * solid.chi / p.eta_pen)",
        "case_local_top": "local_surface_top = _local_surface_top(",
        "production_uniform_seed": "v = -u_impact * jnp.ones_like(phi)",
        "production_streamfunction": 'elif velocity_mode == "streamfunction":',
        "cutcell_aperture": "aperture_x=aperture_x,",
    }
    missing = [name for name, anchor in required_anchors.items() if anchor not in source]
    if missing:
        raise AuditValidationError(f"live source operator anchors missing: {missing}")
    return {
        "central_difference_D": "pf._ddx(u,p.dx)+pf._ddy(v,p.dy), actual contract-12 periodic central operators",
        "production_projection": "periodic FFT poisson_solve using p.m2_proj; unchanged",
        "phase_velocity_flux_reconstruction": _concat_text(
            "cell-centered velocity averaged to +axis faces, multiplied by embedded ",
            "aperture length, divided by V_i",
        ),
        "embedded_wall_flux": _concat_text(
            "bilinear interpolation of stored cell-centered u/v to wall_centroid; ",
            "dot with normalized cut-contour normal and integrate wall_measure",
        ),
        "embedded_wall_flux_status": "MEASURED_WITH_DECLARED_BILINEAR_RECONSTRUCTION",
        "phasefield_geometry_authority": _concat_text(
            "EmbeddedFluidGeometry in phasefield.py (volume, aperture, wall ",
            "centroid/normal/measure)",
        ),
        "anchors": required_anchors,
    }


def _derive_bundle(case_name: str, case: dict[str, Any], N: int, requested_dt: float) -> CaseBundle:
    if case.get("velocity_mode", "uniform") != "uniform":
        raise AuditValidationError(f"frozen canary no longer has production uniform initialization: {case_name}")
    policy = timestep_policy.effective_dt_for_case(case, N, requested_dt, EXPECTED_POLICY)
    p, solid, base = pf.build_case(case, N=N, dt=float(policy["effective_dt"]))
    x0 = float(case.get("x0", 3.0))
    radius = float(case.get("R", 0.7))
    local_top = float(np.asarray(pf._local_surface_top(solid.sdf, p, x0=x0, radius=radius)))
    clearance_min = max(float(case.get("impact_gap_eps", 2.0)) * float(p.eps), 0.05)
    y0 = float(case.get("y0", local_top + radius + max(float(case.get("impact_gap", clearance_min)), clearance_min)))
    actual_gap = y0 - radius - local_top
    if actual_gap < clearance_min - 1e-12:
        raise AuditValidationError(
            f"case {case_name} violates unchanged initial clearance: {actual_gap} < {clearance_min}"
        )
    return CaseBundle(
        case_name=case_name,
        case=dict(case),
        N=int(N),
        requested_dt=float(requested_dt),
        p=p,
        solid=solid,
        base=base,
        x0=x0,
        y0=y0,
        R=radius,
        u_impact=float(case.get("u_impact", 0.5)),
        local_surface_top=local_top,
        initial_gap=actual_gap,
        policy=policy,
    )


def _metadata(bundle: CaseBundle) -> dict[str, Any]:
    p = bundle.p
    return {
        "case_name": bundle.case_name,
        "case": bundle.case,
        "N": bundle.N,
        "Lx": float(p.Lx),
        "Ly": float(p.Ly),
        "dx": float(p.dx),
        "dy": float(p.dy),
        "eps": float(p.eps),
        "eps_over_dx": float(p.eps / p.dx),
        "M": float(p.M),
        "We": float(p.We),
        "Re": float(p.Re),
        "cos_theta": float(bundle.case.get("cos_theta", 0.0)),
        "rho_l": float(p.rho_l),
        "rho_g": float(p.rho_g),
        "nu_l": float(p.nu_l),
        "nu_g": float(p.nu_g),
        "eta_pen": float(p.eta_pen),
        "R": bundle.R,
        "x0": bundle.x0,
        "y0": bundle.y0,
        "local_surface_top": bundle.local_surface_top,
        "initial_gap": bundle.initial_gap,
        "u_impact_argument": bundle.u_impact,
        "requested_dt": bundle.requested_dt,
        "effective_dt": float(p.dt),
        "policy": bundle.policy,
        "phase_dtype": str(np.asarray(bundle.base.phi).dtype),
        "velocity_dtype": str(np.asarray(bundle.base.u).dtype),
    }


def _smoothstep5(values: Any):
    z = jnp.clip(values, 0.0, 1.0)
    return z**3 * (10.0 + z * (-15.0 + 6.0 * z))


def build_c2_velocity(
    bundle: CaseBundle, solid: Any | None = None, *, phi: Any | None = None
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Build C2 from the case's actual SDF and actual periodic central operators.

    A C2 compact radial envelope is multiplied by a C2-flat wall taper (4--12 dx)
    and by a y-periodic seam plateau (3--6 dy). No velocity mask or pressure solve
    is applied after differentiation. A single deterministic scalar matches the
    V_i-weighted phi>=0.9 core mean to -u_impact.
    """
    solid = bundle.solid if solid is None else solid
    phi = bundle.base.phi if phi is None else phi
    X, Y = pf.grids(bundle.p, dtype=bundle.p.dtype)
    sx = (X - bundle.x0 + 0.5 * bundle.p.Lx) % bundle.p.Lx - 0.5 * bundle.p.Lx
    sy = (Y - bundle.y0 + 0.5 * bundle.p.Ly) % bundle.p.Ly - 0.5 * bundle.p.Ly
    r2 = sx * sx + sy * sy
    r_cut = bundle.R + 0.45
    q2 = r2 / (r_cut * r_cut)
    envelope = jnp.where(q2 < 1.0, (1.0 - q2) ** 4, 0.0)
    sdf = jnp.asarray(solid.sdf, dtype=bundle.p.dtype)
    wall_z = (sdf - 4.0 * bundle.p.dx) / (8.0 * bundle.p.dx)
    wall_taper = _smoothstep5(wall_z)
    seam_distance = jnp.minimum(Y, bundle.p.Ly - Y)
    seam_z = (seam_distance - 3.0 * bundle.p.dy) / (3.0 * bundle.p.dy)
    y_seam_taper = _smoothstep5(seam_z)
    psi = sx * envelope * wall_taper * y_seam_taper
    # This is the final streamfunction. Deliberately do not mask u/v after these calls.
    unit_u = pf._ddy(psi, bundle.p.dy)
    unit_v = -pf._ddx(psi, bundle.p.dx)
    phi_host = np.asarray(phi, dtype=np.float64)
    volume = np.asarray(pf.phase_control_volumes(solid, bundle.p), dtype=np.float64)
    core_weight = volume * (phi_host >= 0.9)
    denominator = float(np.sum(core_weight))
    if denominator <= 0.0:
        raise AuditValidationError(f"C2 has no phi>=0.9 control-volume core in {bundle.case_name}")
    unit_core_v = float(np.sum(core_weight * np.asarray(unit_v, dtype=np.float64)) / denominator)
    if not math.isfinite(unit_core_v) or unit_core_v >= 0.0:
        raise AuditValidationError(f"C2 unit field does not carry downward core motion: {unit_core_v}")
    scale = -bundle.u_impact / unit_core_v
    u = (scale * unit_u).astype(bundle.p.dtype)
    v = (scale * unit_v).astype(bundle.p.dtype)
    return (
        u,
        v,
        psi,
        {
            "candidate": "SDF_TAPERED_STREAMFUNCTION_V1",
            "implementation_version": "c2_sdf_tapered_streamfunction_v1",
            "formula": _concat_text(
                "psi=A*sx*(1-r2/(R+0.45)^2)^4_+*smooth5((sdf-4dx)/(8dx))*",
                "smooth5((dist_y_seam-3dy)/(3dy)); u=D_y psi; v=-D_x psi",
            ),
            "wall_taper_plateau_and_transition": [
                "sdf<=4dx: zero",
                "4dx<sdf<12dx: quintic smoothstep",
                "sdf>=12dx: one",
            ],
            "periodic_y_seam_taper": [
                "distance<=3dy: zero",
                "3dy<distance<6dy: quintic smoothstep",
                "distance>=6dy: one",
            ],
            "periodic_x": "wrapped sx with compact support r_cut=R+0.45<Lx/2; seam stencils are continuous",
            "post_derivative_masking": False,
            "diagnostic_pressure_projection_applied_to_seed": False,
            "scale_target": "V_i-weighted mean v over phi>=0.9 equals -u_impact",
            "unit_core_v_before_scale": unit_core_v,
            "amplitude_scale": float(scale),
            "geometry_sdf_hash": _array_sha(np.asarray(solid.sdf)),
            "x0": bundle.x0,
            "y0": bundle.y0,
            "R": bundle.R,
        },
    )


def build_candidate(bundle: CaseBundle, candidate: str, *, solid: Any | None = None) -> tuple[Any, dict[str, Any]]:
    """Create one diagnostic state while preserving the production phi bitwise."""
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown candidate {candidate!r}")
    geometry = bundle.solid if solid is None else solid
    phi = bundle.base.phi
    if candidate == "UNIFORM_ALL_DOMAIN":
        state = bundle.base if solid is None else pf.State(phi=phi, u=bundle.base.u, v=bundle.base.v, t=0.0)
        details = {"candidate": candidate, "production_default_replay": solid is None}
    elif candidate == "STREAMFUNCTION_LOCALIZED_V0":
        raw = pf.droplet_initial_state(
            bundle.p,
            x0=bundle.x0,
            y0=bundle.y0,
            R=bundle.R,
            u_impact=bundle.u_impact,
            velocity_mode="streamfunction",
        )
        state = pf.State(phi=phi, u=raw.u, v=raw.v, t=0.0)
        details = {
            "candidate": candidate,
            "source": "exact current pf.droplet_initial_state(..., velocity_mode='streamfunction') u/v",
            "case_specific_x0_y0": True,
            "large_solid_velocity_is_not_relabelled_compatible": True,
        }
    else:
        u, v, _psi, details = build_c2_velocity(bundle, geometry, phi=phi)
        state = pf.State(phi=phi, u=u, v=v, t=0.0)
    if _array_sha(np.asarray(state.phi)) != _array_sha(np.asarray(bundle.base.phi)):
        raise AuditValidationError(f"candidate {candidate} changed phi at t=0")
    return state, details


def _weighted_mean(value: np.ndarray, weight: np.ndarray) -> float | None:
    denominator = float(np.sum(weight, dtype=np.float64))
    if not math.isfinite(denominator) or abs(denominator) <= 1e-30:
        return None
    return float(np.sum(weight * value, dtype=np.float64) / denominator)


def _periodic_bilinear_sample(field: np.ndarray, x: np.ndarray, y: np.ndarray, p: Any) -> np.ndarray:
    """Bilinearly sample a cell-centered periodic velocity at physical coordinates."""
    field = np.asarray(field)
    fx = np.asarray(x, dtype=np.float64) / float(p.dx) - 0.5
    fy = np.asarray(y, dtype=np.float64) / float(p.dy) - 0.5
    ix0 = np.floor(fx).astype(np.int64)
    iy0 = np.floor(fy).astype(np.int64)
    wx = fx - ix0
    wy = fy - iy0
    ix1 = (ix0 + 1) % p.Nx
    iy1 = (iy0 + 1) % p.Ny
    ix0 %= p.Nx
    iy0 %= p.Ny
    return (
        (1.0 - wx) * (1.0 - wy) * field[ix0, iy0]
        + wx * (1.0 - wy) * field[ix1, iy0]
        + (1.0 - wx) * wy * field[ix0, iy1]
        + wx * wy * field[ix1, iy1]
    )


def _divergence_metrics(u: Any, v: Any, solid: Any, p: Any) -> dict[str, Any]:
    u_j = jnp.asarray(u, dtype=p.dtype)
    v_j = jnp.asarray(v, dtype=p.dtype)
    D = pf._ddx(u_j, p.dx) + pf._ddy(v_j, p.dy)
    d_host = np.asarray(D, dtype=np.float64)
    volume = np.asarray(solid.geometry.volume, dtype=np.float64)
    active = volume > 0.0
    if not active.any():
        fv_linf = fv_l2 = None
    else:
        uf = 0.5 * (u_j + jnp.roll(u_j, -1, axis=0))
        vf = 0.5 * (v_j + jnp.roll(v_j, -1, axis=1))
        flux_x = solid.geometry.aperture_x * uf
        flux_y = solid.geometry.aperture_y * vf
        div_fv = pf.control_volume_divergence(flux_x, flux_y, solid.geometry.volume_safe)
        fv_host = np.asarray(div_fv, dtype=np.float64)
        fv_linf = float(np.max(np.abs(fv_host[active])))
        fv_l2 = float(np.sqrt(np.sum(volume[active] * fv_host[active] ** 2) / np.sum(volume[active])))
    return {
        "initial_D_div_Linf": float(np.max(np.abs(d_host))),
        "initial_D_div_L2_grid_rms": float(np.sqrt(np.mean(d_host**2))),
        "D_div_L2_fluid_volume_rms": float(
            np.sqrt(np.sum(volume[active] * d_host[active] ** 2) / np.sum(volume[active]))
        )
        if active.any()
        else None,
        "cutcell_flux_div_Linf": fv_linf,
        "cutcell_flux_div_L2_fluid_volume_rms": fv_l2,
        "cutcell_flux_divergence_status": "MEASURED_OPEN_FACE_RECONSTRUCTION" if fv_linf is not None else "UNMEASURED",
        "cutcell_flux_reconstruction": _concat_text(
            "u_f=0.5*(u_i+u_{i+1}), v_f=0.5*(v_i+v_{i+1}); ",
            "shared contract-9 A_f times face velocity; divide by V_i",
        ),
    }


def _wall_metrics(u: Any, v: Any, solid: Any, p: Any) -> dict[str, Any]:
    u_host = np.asarray(u, dtype=np.float64)
    v_host = np.asarray(v, dtype=np.float64)
    sdf = np.asarray(solid.sdf, dtype=np.float64)
    chi = np.asarray(solid.chi, dtype=np.float64)
    volume = np.asarray(solid.geometry.volume, dtype=np.float64)
    speed = np.hypot(u_host, v_host)
    cell_area = float(p.dx * p.dy)
    deep = sdf < -2.0 * float(p.dx)
    near_sdf = np.abs(sdf) <= 3.0 * float(p.dx)
    near_chi = chi >= 0.01
    deep_chi = chi >= 0.9
    partial = (volume > 0.0) & (volume < cell_area * (1.0 - 1e-6))
    wall_cells = np.asarray(solid.geometry.wall_measure) > 0.0
    wall_area = np.asarray(solid.geometry.wall_measure, dtype=np.float64)
    wall_nx = np.asarray(solid.geometry.wall_normal_x, dtype=np.float64)
    wall_ny = np.asarray(solid.geometry.wall_normal_y, dtype=np.float64)
    wall_x = np.asarray(solid.geometry.wall_centroid_x, dtype=np.float64)
    wall_y = np.asarray(solid.geometry.wall_centroid_y, dtype=np.float64)
    if wall_cells.any():
        uw = _periodic_bilinear_sample(u_host, wall_x[wall_cells], wall_y[wall_cells], p)
        vw = _periodic_bilinear_sample(v_host, wall_x[wall_cells], wall_y[wall_cells], p)
        nx = wall_nx[wall_cells]
        ny = wall_ny[wall_cells]
        norm = np.hypot(nx, ny)
        valid = norm > 1e-14
        normal_speed = np.zeros_like(uw)
        normal_speed[valid] = (uw[valid] * nx[valid] + vw[valid] * ny[valid]) / norm[valid]
        area = wall_area[wall_cells]
        integrated = float(np.sum(area * normal_speed))
        absolute_integrated = float(np.sum(area * np.abs(normal_speed)))
        wall_sample_speed = float(np.max(np.hypot(uw, vw)))
        wall_normal_linf = float(np.max(np.abs(normal_speed)))
        n_wall_cells = int(np.sum(wall_cells))
    else:
        integrated = absolute_integrated = wall_sample_speed = wall_normal_linf = 0.0
        n_wall_cells = 0
    return {
        "max_abs_velocity_deep_solid": float(np.max(speed[deep])) if deep.any() else 0.0,
        "deep_solid_cell_count": int(np.sum(deep)),
        "max_abs_velocity_nearwall_sdf_abs_le_3dx": float(np.max(speed[near_sdf])) if near_sdf.any() else 0.0,
        "max_abs_velocity_nearwall_chi_ge_0p01": float(np.max(speed[near_chi])) if near_chi.any() else 0.0,
        "max_abs_velocity_deep_chi_ge_0p9": float(np.max(speed[deep_chi])) if deep_chi.any() else 0.0,
        "max_abs_velocity_partial_cutcell": float(np.max(speed[partial])) if partial.any() else 0.0,
        "partial_cutcell_count": int(np.sum(partial)),
        "wall_area_cell_count": n_wall_cells,
        "max_embedded_wall_bilinear_normal_velocity": wall_normal_linf,
        "embedded_wall_signed_flux_bilinear": integrated,
        "embedded_wall_absolute_flux_bilinear": absolute_integrated,
        "embedded_wall_flux_status": "MEASURED_WITH_DECLARED_BILINEAR_CELL_VELOCITY_RECONSTRUCTION",
        "wall_normal_convention": _concat_text(
            "normalized length-weighted normal points from fluid into solid; ",
            "positive dot velocity is penetration",
        ),
        "max_speed_wall_sample": wall_sample_speed,
    }


def _energy_budget(phi: np.ndarray, u: np.ndarray, v: np.ndarray, solid: Any, p: Any) -> dict[str, Any]:
    phi = np.asarray(phi, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    V = np.asarray(solid.geometry.volume, dtype=np.float64)
    chi = np.asarray(solid.chi, dtype=np.float64)
    rho = float(p.rho_g) + (float(p.rho_l) - float(p.rho_g)) * phi
    speed2 = u * u + v * v
    density = 0.5 * rho * speed2
    liquid_weight = V * phi
    gas_weight = V * (1.0 - phi)
    energy = {
        "all_grid_plain_cell_area": float(np.sum(density) * p.dx * p.dy),
        "open_fluid_control_volume": float(np.sum(density * V)),
        "liquid_weighted_control_volume": float(np.sum(density * liquid_weight)),
        "gas_weighted_control_volume": float(np.sum(density * gas_weight)),
        "solid_indicator_cell_area": float(np.sum(density * chi) * p.dx * p.dy),
        "deep_solid_cell_area": float(np.sum(density * (np.asarray(solid.sdf) < -2.0 * p.dx)) * p.dx * p.dy),
    }
    totals = {
        "open_fluid_volume": float(np.sum(V)),
        "liquid_weight_sum_Vphi": float(np.sum(liquid_weight)),
        "gas_weight_sum_V1minusphi": float(np.sum(gas_weight)),
        "solid_indicator_volume_dxdy_chi": float(np.sum(chi) * p.dx * p.dy),
    }
    return {"energy": energy, "region_normalization": totals, "rho_definition": "rho_g+(rho_l-rho_g)*phi"}


def _velocity_stats(
    phi: np.ndarray, u: np.ndarray, v: np.ndarray, solid: Any, p: Any, bundle: CaseBundle
) -> dict[str, Any]:
    phi = np.asarray(phi, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    volume = np.asarray(solid.geometry.volume, dtype=np.float64)
    liquid = volume * phi
    gas = volume * (1.0 - phi)
    core = volume * (phi >= 0.9)
    speed = np.hypot(u, v)
    x_axis = (np.arange(p.Nx, dtype=np.float64) + 0.5) * float(p.dx)
    y_axis = (np.arange(p.Ny, dtype=np.float64) + 0.5) * float(p.dy)
    X, Y = np.meshgrid(x_axis, y_axis, indexing="ij")
    sx = (X - bundle.x0 + 0.5 * p.Lx) % p.Lx - 0.5 * p.Lx
    sy = (Y - bundle.y0 + 0.5 * p.Ly) % p.Ly - 0.5 * p.Ly
    radius = np.hypot(sx, sy)
    gas_region = (phi <= 0.1) & (volume > 0.0)
    return_region = gas_region & (radius >= bundle.R + 0.05) & (radius <= bundle.R + 0.45 + 2.0 * p.dx)
    farfield = gas_region & (radius >= bundle.R + 0.75)
    return {
        "actual_initial_liquid_weighted_v": _weighted_mean(v, liquid),
        "actual_initial_liquid_weighted_u": _weighted_mean(u, liquid),
        "actual_initial_core_v_phi_ge_0p9": _weighted_mean(v, core),
        "actual_initial_core_u_phi_ge_0p9": _weighted_mean(u, core),
        "actual_initial_gas_weighted_v": _weighted_mean(v, gas),
        "actual_initial_gas_weighted_u": _weighted_mean(u, gas),
        "liquid_weight_sum_Vphi": float(np.sum(liquid)),
        "gas_weight_sum_V1minusphi": float(np.sum(gas)),
        "core_weight_sum_V_phi_ge_0p9": float(np.sum(core)),
        "normalization": _concat_text(
            "all phase means use the same physical V_i; liquid V_i*phi_i; ",
            "gas V_i*(1-phi_i); core V_i*1(phi>=0.9)",
        ),
        "farfield_gas_velocity": {
            "u": _weighted_mean(u, volume * (1.0 - phi) * farfield),
            "v": _weighted_mean(v, volume * (1.0 - phi) * farfield),
            "mean_speed": _weighted_mean(speed, volume * (1.0 - phi) * farfield),
            "max_speed": float(np.max(speed[farfield])) if farfield.any() else 0.0,
            "definition": "phi<=0.1, V_i>0, periodic distance from case-specific droplet center >= R+0.75",
        },
        "return_flow_peak_speed": float(np.max(speed[return_region])) if return_region.any() else 0.0,
        "return_flow_peak_definition": _concat_text(
            "max speed in phi<=0.1 open-fluid cells with R+0.05 <= periodic center ",
            "distance <= R+0.45+2dx",
        ),
        "max_speed_global": float(np.max(speed)),
        "max_speed_physical_open_volume": float(np.max(speed[volume > 0.0])) if np.any(volume > 0.0) else 0.0,
        "kinetic_energy_regionwise": _energy_budget(phi, u, v, solid, p),
    }


def _projection_adjustment(u: Any, v: Any, phi: Any, solid: Any, p: Any) -> dict[str, Any]:
    u0, v0 = jnp.asarray(u, dtype=p.dtype), jnp.asarray(v, dtype=p.dtype)
    div0 = pf._ddx(u0, p.dx) + pf._ddy(v0, p.dy)
    pressure = pf.poisson_solve(div0, p.m2_proj)
    u1 = u0 - pf._ddx(pressure, p.dx)
    v1 = v0 - pf._ddy(pressure, p.dy)
    du = np.asarray(u1, dtype=np.float64) - np.asarray(u0, dtype=np.float64)
    dv = np.asarray(v1, dtype=np.float64) - np.asarray(v0, dtype=np.float64)
    dmag = np.hypot(du, dv)
    phi_h = np.asarray(phi, dtype=np.float64)
    V = np.asarray(pf.phase_control_volumes(solid, p), dtype=np.float64)
    w = V * phi_h
    return {
        "initial_projection_adjustment_Linf": float(np.max(dmag)),
        "initial_projection_adjustment_liquid_weighted": {
            "mean_delta_v": _weighted_mean(dv, w),
            "rms_delta_speed": float(np.sqrt(np.sum(w * dmag**2) / np.sum(w))) if np.sum(w) > 0 else None,
            "mean_abs_delta_speed": _weighted_mean(dmag, w),
            "normalization_weight_sum_Vphi": float(np.sum(w)),
        },
        "initial_D_div_Linf_after_diagnostic_projection": float(
            np.max(np.abs(np.asarray(pf._ddx(u1, p.dx) + pf._ddy(v1, p.dy), dtype=np.float64)))
        ),
        "projection_was_applied_to_candidate_seed": False,
        "projection": "diagnostic only: subtract G(poisson_solve(D(u),m2_proj)); not fed back into the initial state",
    }


def initial_metrics(bundle: CaseBundle, candidate: str, state: Any, details: dict[str, Any]) -> dict[str, Any]:
    phi = np.asarray(state.phi)
    u = np.asarray(state.u)
    v = np.asarray(state.v)
    divergence = _divergence_metrics(u, v, bundle.solid, bundle.p)
    solid_metrics = _wall_metrics(u, v, bundle.solid, bundle.p)
    velocity = _velocity_stats(phi, u, v, bundle.solid, bundle.p, bundle)
    projection = _projection_adjustment(u, v, phi, bundle.solid, bundle.p)
    metadata = _metadata(bundle)
    U = max(abs(bundle.u_impact), 1e-30)
    dimensionless_D = divergence["initial_D_div_Linf"] * bundle.p.dx / U
    dimensionless_FV = (
        divergence["cutcell_flux_div_Linf"] * bundle.p.dx / U
        if divergence["cutcell_flux_div_Linf"] is not None
        else None
    )
    core_v = velocity["actual_initial_core_v_phi_ge_0p9"]
    checks = {
        "central_D": dimensionless_D <= INITIAL_ACCEPTANCE["central_D_dimensionless_Linf_max"],
        "open_face_FV_divergence": dimensionless_FV is not None
        and dimensionless_FV <= INITIAL_ACCEPTANCE["open_face_FV_dimensionless_Linf_max"],
        "deep_solid_velocity": solid_metrics["max_abs_velocity_deep_solid"] / U
        <= INITIAL_ACCEPTANCE["deep_solid_speed_over_u_impact_max"],
        "nearwall_chi_velocity": solid_metrics["max_abs_velocity_nearwall_chi_ge_0p01"] / U
        <= INITIAL_ACCEPTANCE["nearwall_chi_ge_0p01_speed_over_u_impact_max"],
        "embedded_wall_normal_velocity": solid_metrics["max_embedded_wall_bilinear_normal_velocity"] / U
        <= INITIAL_ACCEPTANCE["embedded_wall_bilinear_normal_speed_over_u_impact_max"],
        "core_target": core_v is not None
        and abs(core_v + bundle.u_impact) / U <= INITIAL_ACCEPTANCE["core_mean_speed_relative_error_max"],
    }
    measured = all(
        value is not None
        for value in (
            divergence["initial_D_div_Linf"],
            divergence["cutcell_flux_div_Linf"],
            solid_metrics["max_abs_velocity_deep_solid"],
            solid_metrics["max_embedded_wall_bilinear_normal_velocity"],
            core_v,
        )
    )
    return {
        "case_name": bundle.case_name,
        "candidate": candidate,
        "diagnostic_only": True,
        "case_identity": metadata,
        "initializer_parameters": details,
        "initial_phi_hash": _array_sha(phi),
        "initial_geometry_hash": _geometry_hash(bundle.solid),
        "initial_u_hash": _array_sha(u),
        "initial_v_hash": _array_sha(v),
        "requested_u_impact": bundle.u_impact,
        **velocity,
        **divergence,
        **solid_metrics,
        **projection,
        "dimensionless_residuals": {
            "dx_D_Linf_over_u_impact": dimensionless_D,
            "dx_FV_Linf_over_u_impact": dimensionless_FV,
            "deep_solid_speed_over_u_impact": solid_metrics["max_abs_velocity_deep_solid"] / U,
            "nearwall_chi_speed_over_u_impact": solid_metrics["max_abs_velocity_nearwall_chi_ge_0p01"] / U,
            "wall_normal_sample_over_u_impact": solid_metrics["max_embedded_wall_bilinear_normal_velocity"] / U,
        },
        "predeclared_acceptance": INITIAL_ACCEPTANCE,
        "acceptance_checks": checks,
        "measurements_complete": measured,
        "initial_constraints_pass": bool(measured and all(checks.values())),
    }


def _case_fingerprint(bundle: CaseBundle) -> str:
    return _sha_bytes(_json_bytes({"case": bundle.case, "geometry": _geometry_hash(bundle.solid), "N": bundle.N}))


def _binding(
    bundle: CaseBundle, candidate: str, state: Any, *, profile: str, t_end: float, frame_dt: float
) -> dict[str, Any]:
    details_version = (
        "c2_sdf_tapered_streamfunction_v1" if candidate == "SDF_TAPERED_STREAMFUNCTION_V1" else candidate.lower()
    )
    current_sources = _source_hashes()
    return {
        "base_main_git_sha": BASE_MAIN_SHA,
        "solver_contract": int(pf.SOLVER_CONTRACT_VERSION),
        "solver_source_sha256": current_sources["phasefield"],
        "initializer": candidate,
        "initializer_version": details_version,
        "case_fingerprint": _case_fingerprint(bundle),
        "geometry_fingerprint": _geometry_hash(bundle.solid),
        "initial_state_hashes": _state_hashes(state),
        "N": bundle.N,
        "eps": float(bundle.p.eps),
        "eps_over_dx": float(bundle.p.eps / bundle.p.dx),
        "requested_dt": bundle.requested_dt,
        "effective_dt": float(bundle.p.dt),
        "timestep_policy": bundle.policy,
        "horizon": float(t_end),
        "frame_cadence": float(frame_dt),
        "velocity_dtype": str(np.asarray(state.u).dtype),
        "phase_dtype": str(np.asarray(state.phi).dtype),
        "diagnostic_sampling_rule": _concat_text(
            "every public step for event/retention rows; three internal-substep scalar ledger; ",
            "fields at t=0 and exact 0.08 cadence",
        ),
        "profile": profile,
        "stage_version": STAGE_VERSION,
        "diagnostic_only": True,
        "production_lineage_eligible": False,
    }


def _metric_record(
    phi: Any, u: Any, v: Any, solid: Any, p: Any, bundle: CaseBundle, *, include_divergence: bool = True
) -> dict[str, Any]:
    phi_h, u_h, v_h = np.asarray(phi), np.asarray(u), np.asarray(v)
    out = _velocity_stats(phi_h, u_h, v_h, solid, p, bundle)
    if include_divergence:
        out.update(_divergence_metrics(u_h, v_h, solid, p))
        out.update(_wall_metrics(u_h, v_h, solid, p))
    return out


def _circular_x_center(phi: np.ndarray, volume: np.ndarray, p: Any) -> float | None:
    weights = np.where(volume > 0.0, np.clip(phi, 0.0, 1.0) * volume, 0.0)
    total = float(np.sum(weights))
    if total <= 0.0:
        return None
    x = (np.arange(p.Nx, dtype=np.float64) + 0.5) * float(p.dx)
    theta = 2.0 * np.pi * x / float(p.Lx)
    cos_m = float(np.sum(weights.sum(axis=1) * np.cos(theta)) / total)
    sin_m = float(np.sum(weights.sum(axis=1) * np.sin(theta)) / total)
    return float((math.atan2(sin_m, cos_m) % (2.0 * math.pi)) * p.Lx / (2.0 * math.pi))


def _periodic_distance_x(x: np.ndarray, x0: float, Lx: float) -> np.ndarray:
    return (x - x0 + 0.5 * Lx) % Lx - 0.5 * Lx


def _threshold_contour_edge_crossings(phi: np.ndarray, threshold: float, p: Any) -> np.ndarray:
    """Linearly reconstruct the periodic phi=threshold contour on grid edges."""
    phi = np.asarray(phi, dtype=np.float64)
    x_axis = (np.arange(p.Nx, dtype=np.float64) + 0.5) * float(p.dx)
    y_axis = (np.arange(p.Ny, dtype=np.float64) + 0.5) * float(p.dy)
    points = []
    for axis, coordinates, spacing, length in (
        (0, x_axis, float(p.dx), float(p.Lx)),
        (1, y_axis, float(p.dy), float(p.Ly)),
    ):
        neighbor = np.roll(phi, -1, axis=axis)
        crosses = ((phi <= threshold) & (neighbor >= threshold)) | ((phi >= threshold) & (neighbor <= threshold))
        crosses &= neighbor != phi
        i, j = np.where(crosses)
        if not len(i):
            continue
        a = phi[i, j]
        b = neighbor[i, j]
        fraction = (threshold - a) / (b - a)
        if axis == 0:
            x = (coordinates[i] + fraction * spacing) % length
            y = y_axis[j]
        else:
            x = x_axis[i]
            y = (coordinates[j] + fraction * spacing) % length
        points.append(np.stack((x, y), axis=1))
    return np.concatenate(points, axis=0) if points else np.empty((0, 2), dtype=np.float64)


def _local_gap(phi: np.ndarray, threshold: float, solid: Any, p: Any) -> dict[str, Any]:
    """Minimum actual-SDF distance evaluated on the reconstructed local phase contour."""
    V = np.asarray(solid.geometry.volume, dtype=np.float64)
    sdf = np.asarray(solid.sdf, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    mask = (phi >= threshold) & (V > 0.0)
    if not mask.any():
        return {
            "gap": None,
            "gap_cell": None,
            "local_wall_normal_into_solid": None,
            "support_xy": None,
            "gap_method": "UNMEASURED_NO_THRESHOLDED_LIQUID",
        }
    contour = _threshold_contour_edge_crossings(phi, threshold, p)
    if contour.size:
        sdf_on_contour = _periodic_bilinear_sample(sdf, contour[:, 0], contour[:, 1], p)
        selected = int(np.argmin(sdf_on_contour))
        point = contour[selected]
        gap = float(sdf_on_contour[selected])
        method = "BILINEAR_SDF_ON_LINEAR_PERIODIC_PHI_THRESHOLD_CONTOUR"
    else:
        # A fully thresholded domain may have no contour; retain an explicitly marked
        # cell-sampled distance rather than manufacturing an interpolated event.
        values = np.where(mask, sdf, np.inf)
        i, j = np.unravel_index(int(np.argmin(values)), values.shape)
        x_axis = (np.arange(p.Nx, dtype=np.float64) + 0.5) * float(p.dx)
        y_axis = (np.arange(p.Ny, dtype=np.float64) + 0.5) * float(p.dy)
        point = np.array([x_axis[i], y_axis[j]], dtype=np.float64)
        gap = float(values[i, j])
        method = "CELL_SAMPLED_SDF_FALLBACK_NO_CONTOUR"
    i = int(math.floor(float(point[0]) / float(p.dx))) % p.Nx
    j = int(math.floor(float(point[1]) / float(p.dy))) % p.Ny
    wall_mask = np.asarray(solid.geometry.wall_measure) > 0.0
    result = {
        "gap": gap,
        "gap_cell": [i, j],
        "support_xy": [float(point[0]), float(point[1])],
        "wall_xy": None,
        "local_wall_normal_into_solid": None,
        "wall_normal_source": "UNMEASURED_NO_EMBEDDED_WALL_CELLS",
        "gap_method": method,
        "threshold_contour_crossing_count": int(contour.shape[0]),
    }
    if not wall_mask.any():
        return result
    wall_x = np.asarray(solid.geometry.wall_centroid_x, dtype=np.float64)[wall_mask]
    wall_y = np.asarray(solid.geometry.wall_centroid_y, dtype=np.float64)[wall_mask]
    nx = np.asarray(solid.geometry.wall_normal_x, dtype=np.float64)[wall_mask]
    ny = np.asarray(solid.geometry.wall_normal_y, dtype=np.float64)[wall_mask]
    dx_wall = _periodic_distance_x(wall_x, float(point[0]), float(p.Lx))
    dy_wall = (wall_y - float(point[1]) + 0.5 * float(p.Ly)) % float(p.Ly) - 0.5 * float(p.Ly)
    nearest = int(np.argmin(dx_wall * dx_wall + dy_wall * dy_wall))
    normal_norm = math.hypot(float(nx[nearest]), float(ny[nearest]))
    normal = [float(nx[nearest] / normal_norm), float(ny[nearest] / normal_norm)] if normal_norm > 1e-14 else None
    result.update(
        {
            "wall_xy": [float(wall_x[nearest]), float(wall_y[nearest])],
            "local_wall_normal_into_solid": normal,
            "wall_normal_source": (
                "nearest wall_measure centroid to reconstructed local phi contour, with periodic x/y distance"
            ),
        }
    )
    return result


def _approach_speed(u: np.ndarray, v: np.ndarray, support: dict[str, Any], p: Any) -> float | None:
    point = support.get("support_xy")
    cell = support.get("gap_cell")
    normal = support.get("local_wall_normal_into_solid")
    if normal is None or (point is None and cell is None):
        return None
    if point is not None:
        u_value = float(_periodic_bilinear_sample(np.asarray(u), np.asarray([point[0]]), np.asarray([point[1]]), p)[0])
        v_value = float(_periodic_bilinear_sample(np.asarray(v), np.asarray([point[0]]), np.asarray([point[1]]), p)[0])
    else:
        i, j = cell
        u_value, v_value = float(u[i, j]), float(v[i, j])
    # Positive is toward the solid because n points from fluid into solid. Lab-frame
    # v remains separately signed and is negative for downward motion on a flat bottom wall.
    return float(u_value * normal[0] + v_value * normal[1])


def _liquid_momentum_energy(phi: np.ndarray, u: np.ndarray, v: np.ndarray, solid: Any, p: Any) -> dict[str, float]:
    V = np.asarray(solid.geometry.volume, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    rho = float(p.rho_g) + (float(p.rho_l) - float(p.rho_g)) * phi
    mass = V * phi * rho
    kinetic = 0.5 * V * phi * rho * (u * u + v * v)
    return {
        "liquid_mass_density_weighted": float(np.sum(mass)),
        "liquid_momentum_x": float(np.sum(mass * u)),
        "liquid_momentum_y": float(np.sum(mass * v)),
        "liquid_physical_kinetic_energy": float(np.sum(kinetic)),
    }


def _height_width(phi: np.ndarray, solid: Any, p: Any, radius: float) -> dict[str, Any]:
    V = np.asarray(solid.geometry.volume, dtype=np.float64)
    mask = (phi >= 0.5) & (V > 0.0)
    if not mask.any():
        return {
            "spread_width": 0.0,
            "beta": 0.0,
            "drop_height": 0.0,
            "centroid_y": None,
            "contact_line_x_left": None,
            "contact_line_x_right": None,
        }
    width = float(observables.periodic_spreading_width(phi * (V > 0.0), threshold=0.5, dx=float(p.dx), Lx=float(p.Lx)))
    y = (np.arange(p.Ny, dtype=np.float64) + 0.5) * float(p.dy)
    rows = np.any(mask, axis=0)
    height = float(y[rows].max() - y[rows].min() + p.dy)
    w = V * np.where(mask, np.clip(phi, 0.0, 1.0), 0.0)
    centroid_y = float(np.sum(w * y[None, :]) / max(float(np.sum(w)), 1e-30))
    columns = np.flatnonzero(np.any(mask, axis=1))
    x = (np.arange(p.Nx, dtype=np.float64) + 0.5) * float(p.dx)
    # Contact-line positions are the outermost occupied columns in the periodic arc
    # centered at the volume-weighted droplet centroid; these are morphology diagnostics,
    # not a substitute for the embedded wall contact geometry.
    xcm = _circular_x_center(phi, V, p)
    offsets = _periodic_distance_x(x[columns], float(xcm or 0.0), float(p.Lx))
    left, right = float(np.min(offsets)), float(np.max(offsets))
    return {
        "spread_width": width,
        "beta": float(width / (2.0 * radius)),
        "drop_height": height,
        "centroid_y": centroid_y,
        "contact_line_x_left_relative_to_centroid": left,
        "contact_line_x_right_relative_to_centroid": right,
        "contact_line_x_center": xcm,
    }


def _trajectory_row(
    phi: Any, u: Any, v: Any, solid: Any, p: Any, bundle: CaseBundle, step: int, t: float, *, include_constraints: bool
) -> dict[str, Any]:
    phi_h = np.asarray(phi, dtype=np.float64)
    u_h = np.asarray(u, dtype=np.float64)
    v_h = np.asarray(v, dtype=np.float64)
    V = np.asarray(solid.geometry.volume, dtype=np.float64)
    stats = _velocity_stats(phi_h, u_h, v_h, solid, p, bundle)
    gap05 = _local_gap(phi_h, 0.5, solid, p)
    gap01 = _local_gap(phi_h, 0.1, solid, p)
    gap05_value = gap05["gap"]
    gap01_value = gap01["gap"]
    normal_speed = _approach_speed(u_h, v_h, gap05, p)
    momentum = _liquid_momentum_energy(phi_h, u_h, v_h, solid, p)
    morphology = _height_width(phi_h, solid, p, bundle.R)
    mass0 = float(np.sum(np.asarray(bundle.base.phi, dtype=np.float64) * V))
    mass = float(np.sum(phi_h * V))
    denom = max(float(np.sum(np.abs(phi_h * V))), 1e-12)
    deep = V <= 0.0
    deep_leak = float(np.sum(np.abs(phi_h) * deep) / denom)
    speed = np.hypot(u_h, v_h)
    row = {
        "step": int(step),
        "t": float(t),
        "v_liquid": stats["actual_initial_liquid_weighted_v"],
        "u_liquid": stats["actual_initial_liquid_weighted_u"],
        "v_core": stats["actual_initial_core_v_phi_ge_0p9"],
        "v_gas": stats["actual_initial_gas_weighted_v"],
        "liquid_weight_sum": stats["liquid_weight_sum_Vphi"],
        "momentum": momentum,
        "kinetic_energy_physical": stats["kinetic_energy_regionwise"]["energy"]["open_fluid_control_volume"],
        "kinetic_energy_liquid": momentum["liquid_physical_kinetic_energy"],
        "max_speed_global": float(np.max(speed)),
        "max_speed_physical": float(np.max(speed[V > 0.0])) if np.any(V > 0.0) else 0.0,
        "gap_phi05": gap05_value,
        "gap_phi01": gap01_value,
        "gap_phi05_over_dx": None if gap05_value is None else float(gap05_value / p.dx),
        "gap_phi01_over_dx": None if gap01_value is None else float(gap01_value / p.dx),
        "gap_phi05_over_eps": None if gap05_value is None else float(gap05_value / p.eps),
        "gap_phi01_over_eps": None if gap01_value is None else float(gap01_value / p.eps),
        "local_approach_speed_into_solid": normal_speed,
        "local_support_phi05": gap05,
        "local_support_phi01": gap01,
        "contact_phi05": bool(gap05_value is not None and gap05_value <= CONTACT_GAP_CELLS * p.dx),
        "contact_phi01": bool(gap01_value is not None and gap01_value <= CONTACT_GAP_CELLS * p.dx),
        "mass_Vphi": mass,
        "formal_mass_ratio": mass / max(mass0, 1e-30),
        "max_phi_overshoot": float(max(np.max(-phi_h), np.max(phi_h - 1.0), 0.0)),
        "deep_solid_leak": deep_leak,
        **morphology,
    }
    if include_constraints:
        row.update(_divergence_metrics(u, v, solid, p))
        row.update(_wall_metrics(u, v, solid, p))
    return row


def _scalar_ledger(scalars: Any) -> dict[str, float | int | bool]:
    values = dict(zip(impulse_audit.SCALAR_NAMES, np.asarray(scalars, dtype=np.float64).tolist()))
    keep = (
        "mean_v_grid_before",
        "rhs_dv_grid_mean",
        "brinkman_dv_grid_mean_exact",
        "projection_dv_grid_mean_exact",
        "v_liquid_before",
        "v_liquid_after_rhs",
        "v_liquid_after_brinkman",
        "v_liquid_after_projection",
        "v_core_before",
        "v_core_after_projection",
        "v_gas_before",
        "v_gas_after_projection",
        "liquid_momentum_y_before",
        "liquid_momentum_y_after_projection",
        "kinetic_energy_physical_before",
        "kinetic_energy_physical_after_rhs",
        "kinetic_energy_physical_after_brinkman",
        "kinetic_energy_physical_after_projection",
        "div_before_linf",
        "div_after_rhs_linf",
        "div_after_brinkman_linf",
        "div_after_projection_linf",
        "mean_of_proj_correction",
        "resid_abs",
        "resid_relative",
        "cg_iterations",
        "cg_converged",
    )
    out: dict[str, float | int | bool] = {key: float(values[key]) for key in keep if key in values}
    out["cg_iterations"] = int(values["cg_iterations"])
    out["cg_converged"] = bool(values["cg_converged"])
    return out


def _write_run_artifacts(
    out_dir: Path, fingerprint: str, summary: dict[str, Any], fields: dict[str, np.ndarray] | None
) -> dict[str, str]:
    run_dir = out_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in fingerprint[:20])
    summary_path = run_dir / f"{safe}.json"
    _write_json(summary_path, summary)
    files = {summary_path.name: _sha_file(summary_path)}
    if fields:
        field_path = run_dir / f"{safe}.npz"
        np.savez_compressed(field_path, **fields)
        files[field_path.name] = _sha_file(field_path)
    return files


def run_trajectory(
    bundle: CaseBundle,
    candidate: str,
    state: Any,
    *,
    t_end: float,
    profile: str,
    frame_dt: float = FRAME_CADENCE,
    include_substep_ledger: bool = True,
    capture_fields: bool = True,
    existing: dict[str, Any] | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run with the reused, JIT-certified contract-12 substep kernel; optionally continue a run."""
    p = bundle.p
    dt = float(p.dt)
    n_end = int(round(t_end / dt))
    if not math.isclose(n_end * dt, t_end, rel_tol=0.0, abs_tol=1e-10):
        raise AuditValidationError(f"t_end={t_end} is not an integer number of public steps at dt={dt}")
    n_frame = max(1, int(round(frame_dt / dt)))
    if not math.isclose(n_frame * dt, frame_dt, rel_tol=0.0, abs_tol=1e-10):
        raise AuditValidationError(f"frame cadence {frame_dt} is not represented exactly by dt={dt}")
    h = dt / 3.0
    kernel = impulse_audit.make_fast_substep(bundle.solid, p, h)
    state0 = state
    if existing is None:
        step0 = 0
        rows: list[dict[str, Any]] = []
        fields: dict[str, list[np.ndarray]] = {"times": [], "phi": [], "u": [], "v": []}
        initial_row = _trajectory_row(
            state.phi, state.u, state.v, bundle.solid, p, bundle, 0, 0.0, include_constraints=True
        )
        rows.append(initial_row)
        if capture_fields:
            fields["times"].append(0.0)
            for name in ("phi", "u", "v"):
                fields[name].append(np.asarray(getattr(state, name)))
    else:
        state = existing["_state"]
        step0 = int(existing["_last_step"])
        rows = existing["rows"]
        fields = existing["_field_lists"]
        if not math.isclose(float(state.t), step0 * dt, rel_tol=0.0, abs_tol=2e-7):
            raise AuditValidationError("continuation state time and integer step do not agree")
    started = time.perf_counter()
    substep_log: list[dict[str, Any]] = []
    start_step = step0
    for public_step in range(step0 + 1, n_end + 1):
        row_substeps: list[dict[str, Any]] = []
        for _ in range(3):
            state, scalars = kernel(state)
            if include_substep_ledger and public_step * dt <= SHORT_HORIZON + 1e-12:
                row_substeps.append(_scalar_ledger(scalars))
        t = public_step * dt
        save_frame = (public_step % n_frame == 0) or public_step == n_end
        sample_constraints = save_frame or (t <= SHORT_HORIZON + 1e-12)
        row = _trajectory_row(
            state.phi,
            state.u,
            state.v,
            bundle.solid,
            p,
            bundle,
            public_step,
            t,
            include_constraints=sample_constraints,
        )
        if include_substep_ledger and row_substeps:
            row["substep_ledger"] = row_substeps
            substep_log.extend(row_substeps)
        if save_frame:
            # Direct solver arrays are captured at exact common physical times. Full-horizon
            # output has 0 plus 100 cadence frames at dt=.002; no interpolation is performed.
            if capture_fields:
                fields["times"].append(float(t))
                fields["phi"].append(np.asarray(state.phi))
                fields["u"].append(np.asarray(state.u))
                fields["v"].append(np.asarray(state.v))
        rows.append(row)
    elapsed = time.perf_counter() - started
    out_fields = None
    if capture_fields:
        out_fields = {
            "times": np.asarray(fields["times"], dtype=np.float64),
            "phi": np.stack(fields["phi"]),
            "u": np.stack(fields["u"]),
            "v": np.stack(fields["v"]),
        }
    binding = _binding(
        bundle,
        candidate,
        state0 if existing is None else existing["_initial_state"],
        profile=profile,
        t_end=t_end,
        frame_dt=frame_dt,
    )
    run_fingerprint = _sha_bytes(_json_bytes(binding))
    prior_elapsed = (
        0.0
        if existing is None
        else float(existing.get("elapsed_seconds_total", existing.get("elapsed_seconds_this_segment", 0.0)))
    )
    run = {
        "case_name": bundle.case_name,
        "candidate": candidate,
        "diagnostic_only": True,
        "production_lineage_eligible": False,
        "N": bundle.N,
        "dt": dt,
        "t_start": float(start_step * dt),
        "t_end": float(t_end),
        "frame_cadence": float(frame_dt),
        "frame_times": [] if out_fields is None else out_fields["times"].tolist(),
        "rows": rows,
        "substep_ledger": substep_log,
        "run_fingerprint": run_fingerprint,
        "binding": binding,
        "elapsed_seconds_this_segment": elapsed,
        "elapsed_seconds_total": prior_elapsed + elapsed,
        "_state": state,
        "_last_step": n_end,
        "_initial_state": state0 if existing is None else existing["_initial_state"],
        "_field_lists": fields,
        "_fields": out_fields,
        "_artifact_files": {},
    }
    if output_dir is not None:
        summary = {key: value for key, value in run.items() if not key.startswith("_")}
        run["_artifact_files"] = _write_run_artifacts(output_dir, run_fingerprint, summary, out_fields)
    return run


def continue_trajectory(
    bundle: CaseBundle, run: dict[str, Any], *, t_end: float, profile: str, output_dir: Path | None = None
) -> dict[str, Any]:
    return run_trajectory(
        bundle,
        run["candidate"],
        run["_initial_state"],
        t_end=t_end,
        profile=profile,
        frame_dt=run["frame_cadence"],
        include_substep_ledger=False,
        capture_fields=True,
        existing=run,
        output_dir=output_dir,
    )


def authenticate_impact(run: dict[str, Any], bundle: CaseBundle, *, observed_horizon: float) -> dict[str, Any]:
    """Local-footprint, local-normal, motion-plus-postresponse impact protocol."""
    rows = run["rows"]
    contact = next((row for row in rows if row["contact_phi05"]), None)
    if contact is None:
        verdict = (
            "NO_CONTACT_OR_HOVER_AT_DATASET_HORIZON"
            if observed_horizon >= DATASET_HORIZON - 1e-10
            else "NO_CONTACT_WITHIN_OBSERVED_WINDOW"
        )
        return {
            "verdict": verdict,
            "observed_horizon": float(observed_horizon),
            "contact_time": None,
            "criterion": f"local phi=0.5 gap <= {CONTACT_GAP_CELLS} dx",
            "phi01_gap_reported_separately": True,
            "reason": "no local-support contact signal was observed within the stated horizon",
        }
    contact_index = rows.index(contact)
    prior = [row for row in rows[:contact_index] if row["gap_phi05"] is not None]
    pre = prior[-1] if prior else None
    pre_window = prior[-4:]
    gaps = [row["gap_phi05"] for row in pre_window]
    approach_values = [row["local_approach_speed_into_solid"] for row in pre_window]
    decreasing = len(gaps) >= 4 and all(gaps[i + 1] < gaps[i] for i in range(len(gaps) - 1))
    meaningful_approach = (
        pre is not None
        and pre["local_approach_speed_into_solid"] is not None
        and pre["local_approach_speed_into_solid"] >= APPROACH_FLOOR_FRACTION * bundle.u_impact
        and all(value is not None and value > 0.0 for value in approach_values)
    )
    post = [row for row in rows[contact_index + 1 :] if 0.0 < row["t"] - contact["t"] <= POSTCONTACT_WINDOW + 1e-12]
    beta_change = max((abs(row["beta"] - contact["beta"]) for row in post), default=0.0)
    height_change = max((abs(row["drop_height"] - contact["drop_height"]) for row in post), default=0.0)
    line_change = max(
        (
            abs(row.get("contact_line_x_center", 0.0) - contact.get("contact_line_x_center", 0.0))
            for row in post
            if row.get("contact_line_x_center") is not None and contact.get("contact_line_x_center") is not None
        ),
        default=0.0,
    )
    liquid_speed_change = max(
        (
            abs(row["v_liquid"] - contact["v_liquid"])
            for row in post
            if row["v_liquid"] is not None and contact["v_liquid"] is not None
        ),
        default=0.0,
    )
    post_response = bool(
        post
        and (
            beta_change >= bundle.p.dx / (2.0 * bundle.R)
            or height_change >= bundle.p.dx
            or line_change >= 0.5 * bundle.p.dx
            or liquid_speed_change >= POSTCONTACT_SPEED_RESPONSE_FRACTION * bundle.u_impact
        )
    )
    if meaningful_approach and decreasing and post_response:
        verdict = "IMPACT_AUTHENTICATED"
    else:
        verdict = "CONTACT_WITHOUT_MEANINGFUL_IMPACT"
    event_time = float(contact["t"])
    if (
        pre is not None
        and pre["gap_phi05"] is not None
        and contact["gap_phi05"] is not None
        and contact["gap_phi05"] != pre["gap_phi05"]
    ):
        threshold = CONTACT_GAP_CELLS * bundle.p.dx
        fraction = (threshold - pre["gap_phi05"]) / (contact["gap_phi05"] - pre["gap_phi05"])
        if 0.0 <= fraction <= 1.0:
            event_time = float(pre["t"] + fraction * (contact["t"] - pre["t"]))
    momentum = None if pre is None else pre["momentum"]
    return {
        "verdict": verdict,
        "observed_horizon": float(observed_horizon),
        "contact_time_discrete": float(contact["t"]),
        "contact_time_interpolated_gap_threshold": event_time,
        "contact_gap_criterion": f"local phi=0.5 gap <= {CONTACT_GAP_CELLS} dx",
        "contact_gap_phi05": contact["gap_phi05"],
        "contact_gap_phi05_over_dx": contact["gap_phi05_over_dx"],
        "contact_gap_phi05_over_eps": contact["gap_phi05_over_eps"],
        "contact_gap_phi01": contact["gap_phi01"],
        "contact_gap_phi01_over_dx": contact["gap_phi01_over_dx"],
        "contact_gap_phi01_over_eps": contact["gap_phi01_over_eps"],
        "contact_location": contact["local_support_phi05"],
        "precontact_time": None if pre is None else pre["t"],
        "precontact_lab_v_liquid_signed": None if pre is None else pre["v_liquid"],
        "precontact_lab_v_core_signed": None if pre is None else pre["v_core"],
        "precontact_local_normal_approach_speed_positive_toward_solid": None
        if pre is None
        else pre["local_approach_speed_into_solid"],
        "contact_local_normal_approach_speed_positive_toward_solid": contact["local_approach_speed_into_solid"],
        "minimum_approach_floor": APPROACH_FLOOR_FRACTION * bundle.u_impact,
        "precontact_gap_samples": gaps,
        "precontact_local_normal_samples": approach_values,
        "several_decreasing_gap_samples": decreasing,
        "meaningful_local_normal_approach": meaningful_approach,
        "incoming_liquid_momentum_and_energy_precontact": momentum,
        "contact_beta": contact["beta"],
        "postcontact_interval": [float(contact["t"]), float(contact["t"] + POSTCONTACT_WINDOW)],
        "postcontact_sample_count": len(post),
        "postcontact_max_abs_beta_change": beta_change,
        "postcontact_max_abs_height_change": height_change,
        "postcontact_max_contactline_center_shift": line_change,
        "postcontact_max_abs_liquid_speed_change": liquid_speed_change,
        "postcontact_dynamics_verified": post_response,
        "not_equal_to_u_impact": True,
    }


def _retention(run: dict[str, Any]) -> dict[str, Any]:
    rows = run["rows"]
    if not rows or rows[0]["v_liquid"] is None or abs(rows[0]["v_liquid"]) <= 1e-14:
        return {"status": "UNMEASURED", "reason": "initial liquid-weighted velocity is zero or unavailable"}
    v0 = float(rows[0]["v_liquid"])
    samples = []
    for row in rows:
        value = row["v_liquid"]
        samples.append(
            {
                "t": row["t"],
                "v_liquid": value,
                "R_retain": None if value is None else float(value / v0),
                "v_core": row["v_core"],
                "local_normal_approach": row["local_approach_speed_into_solid"],
            }
        )
    return {
        "status": "MEASURED",
        "definition": "R_retain(t)=v_liquid(t)/v_liquid(0); signed, direction-consistent",
        "v_liquid_0": v0,
        "samples": samples,
    }


def _relative_difference(a: float | None, b: float | None, scale: float) -> dict[str, Any]:
    if (
        a is None
        or b is None
        or not math.isfinite(a)
        or not math.isfinite(b)
        or not math.isfinite(scale)
        or scale <= 0.0
    ):
        return {"status": "UNMEASURED", "absolute": None, "relative": None, "scale": scale}
    absolute = abs(a - b)
    denom = max(abs(a), abs(b), scale)
    return {
        "status": "MEASURED",
        "absolute": absolute,
        "relative": absolute / denom,
        "scale": scale,
        "denominator": denom,
    }


def _field_error(a: np.ndarray, b: np.ndarray, *, physical_scale: float) -> dict[str, Any]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    delta = a - b
    rms = float(np.sqrt(np.mean(delta**2)))
    ref = float(np.sqrt(np.mean(b**2)))
    scale = max(ref, physical_scale)
    relative = None if scale <= 1e-30 else rms / scale
    return {
        "status": "UNMEASURED" if relative is None else "MEASURED",
        "Linf": float(np.max(np.abs(delta))),
        "L2_rms": rms,
        "reference_L2_rms": ref,
        "relative_L2_with_physical_floor": relative,
        "normalization_scale": scale,
        "near_zero_fail_closed": relative is None,
    }


def _downsample_mean(field: np.ndarray, factor: int = 3) -> np.ndarray:
    field = np.asarray(field)
    if field.shape[-2] % factor or field.shape[-1] % factor:
        raise AuditValidationError(f"downsample factor {factor} does not divide field shape {field.shape}")
    leading = field.shape[:-2]
    nx, ny = field.shape[-2:]
    shaped = field.reshape(*leading, nx // factor, factor, ny // factor, factor)
    return shaped.mean(axis=(-3, -1))


def _map_common_grid(field: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Explicit center-aligned linear mapping using scipy's periodic grid interpolation."""
    from scipy.ndimage import map_coordinates

    field = np.asarray(field)
    nx_t, ny_t = target_shape
    nx_s, ny_s = field.shape[-2:]
    # Cell-center target coordinate (i+.5)/N mapped to source index x/dx-.5.
    ix = (np.arange(nx_t, dtype=np.float64) + 0.5) * nx_s / nx_t - 0.5
    iy = (np.arange(ny_t, dtype=np.float64) + 0.5) * ny_s / ny_t - 0.5
    gx, gy = np.meshgrid(ix, iy, indexing="ij")
    coords = np.stack([gx, gy])
    return map_coordinates(field, coords, order=1, mode="grid-wrap", prefilter=False)


def _initialization_verdict(matrix: dict[str, Any]) -> str:
    c2_rows = [matrix[case]["SDF_TAPERED_STREAMFUNCTION_V1"] for case in CASE_NAMES]
    if any(not row.get("measurements_complete", False) for row in c2_rows):
        return "INCONCLUSIVE"
    passes = [bool(row.get("initial_constraints_pass")) for row in c2_rows]
    if all(passes):
        return "INITIALIZER_COMPATIBLE"
    if any(passes):
        return "INITIALIZER_PARTIALLY_COMPATIBLE"
    return "NO_FEASIBLE_INITIALIZER_UNDER_CONTRACT12"


def _fleet_verdict(events: dict[str, Any]) -> str:
    missing = [case for case in CASE_NAMES if case not in events]
    if missing:
        return "INCONCLUSIVE"
    verdicts = [events[case].get("verdict") for case in CASE_NAMES]
    if all(verdict == "IMPACT_AUTHENTICATED" for verdict in verdicts):
        return "ALL_REQUIRED_IMPACTS_AUTHENTICATED"
    if any(verdict == "IMPACT_AUTHENTICATED" for verdict in verdicts):
        return "PARTIAL_IMPACT_AUTHENTICATION"
    if all(verdict not in (None, "INCONCLUSIVE") for verdict in verdicts):
        return "NO_VALID_IMPACT_FLEET"
    return "INCONCLUSIVE"


def _assemble_stage_gate(stage_results: dict[str, Any], required: tuple[str, ...]) -> dict[str, Any]:
    absent = [
        name for name in required if name not in stage_results or stage_results[name].get("status") == "UNMEASURED"
    ]
    if absent:
        return {"status": "UNMEASURED", "missing_or_unmeasured_stages": absent}
    return {
        "status": "PASS" if all(stage_results[name].get("pass", False) for name in required) else "FAIL",
        "missing_or_unmeasured_stages": [],
    }


def _contact_time_summary(rows: list[dict[str, Any]]) -> float | None:
    contact = next((row for row in rows if row.get("contact_phi05")), None)
    return None if contact is None else float(contact["t"])


def _match_row(run: dict[str, Any], t: float) -> dict[str, Any] | None:
    rows = run["rows"]
    matches = [row for row in rows if math.isclose(float(row["t"]), t, rel_tol=0.0, abs_tol=1e-10)]
    return matches[0] if matches else None


def _validate_frame_times(actual: Any, *, t_end: float, cadence: float) -> dict[str, Any]:
    """Validate saved solver times against the exact integer-step cadence, without interpolation."""
    actual = np.asarray(actual, dtype=np.float64)
    n_frames = int(round(t_end / cadence)) + 1
    expected = np.arange(n_frames, dtype=np.float64) * cadence
    if actual.shape != expected.shape:
        raise AuditValidationError(f"saved frame count {actual.size} != expected {expected.size}")
    errors = np.abs(actual - expected)
    max_error = float(np.max(errors)) if errors.size else 0.0
    if not np.isfinite(actual).all() or max_error > 1e-10:
        raise AuditValidationError(f"saved frame times miss the integer cadence by {max_error}")
    return {
        "status": "MEASURED",
        "expected_frame_count": int(expected.size),
        "actual_frame_count": int(actual.size),
        "maximum_absolute_time_error": max_error,
        "cadence": float(cadence),
        "t_end": float(t_end),
        "interpolation_used": False,
    }


def _measure_initial_matrix(ctx: dict[str, Any]) -> dict[str, Any]:
    matrix: dict[str, Any] = {}
    region_budget: dict[str, Any] = {}
    for case_name, bundle in ctx["bundles"].items():
        matrix[case_name] = {}
        region_budget[case_name] = {}
        for candidate in CANDIDATES:
            state, details = build_candidate(bundle, candidate)
            row = initial_metrics(bundle, candidate, state, details)
            matrix[case_name][candidate] = row
            region_budget[case_name][candidate] = {
                "case_identity": row["case_identity"],
                "initial_hashes": {
                    key: row[key]
                    for key in ("initial_phi_hash", "initial_geometry_hash", "initial_u_hash", "initial_v_hash")
                },
                "liquid_core_gas_farfield_return_flow": {
                    key: row[key]
                    for key in (
                        "actual_initial_liquid_weighted_v",
                        "actual_initial_core_v_phi_ge_0p9",
                        "actual_initial_gas_weighted_v",
                        "farfield_gas_velocity",
                        "return_flow_peak_speed",
                        "max_speed_global",
                    )
                },
                "kinetic_energy_regionwise": row["kinetic_energy_regionwise"],
                "normalization": row["normalization"],
            }
            ctx["candidate_states"][(case_name, candidate)] = state
    return {"constraint_matrix": matrix, "region_budget": region_budget, "verdict": _initialization_verdict(matrix)}


def _stage_velocity_ledger(
    phi: np.ndarray, u: np.ndarray, v: np.ndarray, bundle: CaseBundle, candidate: str
) -> dict[str, Any]:
    return _metric_record(phi, u, v, bundle.solid, bundle.p, bundle, include_divergence=True)


def _first_substep_ledger(ctx: dict[str, Any]) -> dict[str, Any]:
    ledger: dict[str, Any] = {
        "ledger_kernel": "reused production.impact_impulse_projection_audit.substep_ledger(capture_fields=True)",
        "cases": {},
    }
    for case_name, bundle in ctx["bundles"].items():
        ledger["cases"][case_name] = {}
        for candidate in CANDIDATES:
            initial = ctx["candidate_states"][(case_name, candidate)]
            new_state, entry = impulse_audit.substep_ledger(
                initial, bundle.solid, bundle.p, float(bundle.p.dt) / 3.0, light=False, capture_fields=True
            )
            captured = entry.pop("captured_fields")
            phi_w = captured["phi_before"]
            stages = {
                "before": (captured["u_before"], captured["v_before"]),
                "after_explicit_RHS": (captured["u_after_explicit_rhs"], captured["v_after_explicit_rhs"]),
                "after_Brinkman": (captured["u_after_brinkman"], captured["v_after_brinkman"]),
                "after_pressure_projection": (
                    captured["u_after_pressure_projection"],
                    captured["v_after_pressure_projection"],
                ),
            }
            stage_metrics = {
                name: _stage_velocity_ledger(phi_w, u, v, bundle, candidate) for name, (u, v) in stages.items()
            }
            V = np.asarray(pf.phase_control_volumes(bundle.solid, bundle.p), dtype=np.float64)
            liquid = V * np.asarray(phi_w, dtype=np.float64)
            grid_means = {name: float(np.mean(v)) for name, (_u, v) in stages.items()}
            liquid_means = {
                name: _weighted_mean(np.asarray(v, dtype=np.float64), liquid) for name, (_u, v) in stages.items()
            }
            causal = {
                "v_before": grid_means["before"],
                "v_after_explicit_RHS": grid_means["after_explicit_RHS"],
                "v_after_Brinkman": grid_means["after_Brinkman"],
                "v_after_pressure_projection": grid_means["after_pressure_projection"],
                "Delta_grid_mean_v_RHS": grid_means["after_explicit_RHS"] - grid_means["before"],
                "Delta_grid_mean_v_Brinkman": grid_means["after_Brinkman"] - grid_means["after_explicit_RHS"],
                "Delta_grid_mean_v_projection": grid_means["after_pressure_projection"] - grid_means["after_Brinkman"],
                "Delta_v_RHS_liquid_weighted": liquid_means["after_explicit_RHS"] - liquid_means["before"],
                "Delta_v_Brinkman_liquid_weighted": liquid_means["after_Brinkman"] - liquid_means["after_explicit_RHS"],
                "Delta_v_projection_liquid_weighted": liquid_means["after_pressure_projection"]
                - liquid_means["after_Brinkman"],
                "mean_of_periodic_projection_correction_v": float(np.mean(captured["dv_projection"])),
                "projection_correction_Linf": float(
                    np.max(np.hypot(captured["du_projection"], captured["dv_projection"]))
                ),
                "actual_D_G_m2proj_poisson_residual_Linf": entry["projection"]["poisson_residual_lininf"],
                "actual_D_G_m2proj_poisson_residual_relative": entry["projection"]["poisson_residual_relative"],
                "brinkman_and_projection_are_separately_reported": True,
            }
            ledger["cases"][case_name][candidate] = {
                "public_dt": float(bundle.p.dt),
                "internal_substep_h": float(bundle.p.dt / 3.0),
                "rhs_recomposition_bitwise": entry["rhs_recomposition_bitwise"],
                "cg_iterations": entry["cg_iterations"],
                "cg_converged": entry["cg_converged"],
                "stage_metrics": stage_metrics,
                "causal_increment_ledger": causal,
                "legacy_ledger_detail": entry,
                "post_first_internal_substep_state_hashes": _state_hashes(new_state),
            }
    # Explicitly validate the reused JIT ledger against the frozen public step on a real N=192 C2 state.
    primary = ctx["bundles"]["flat_we100_ct050"]
    c2 = ctx["candidate_states"][("flat_we100_ct050", "SDF_TAPERED_STREAMFUNCTION_V1")]
    ledger["jit_matches_pf_step_validation"] = impulse_audit.validate_ledger(c2, primary.solid, primary.p)
    return ledger


def _old_negative_controls(n: int) -> dict[str, Any]:
    controls = impulse_audit.negative_controls(n=n)
    frozen = json.loads((TWO_PHASE / "evidence" / "l1a2r" / "temporal_control_matrix.json").read_text())
    return {
        "reexecuted_l1a2r_negative_controls": controls,
        "frozen_l1a2r_temporal_control_matrix": frozen,
        "control_language": {
            "EMPTY_SOLID_UNIFORM": "uniform translation control; retained to t=0.24 in L1A-2r reference",
            "PROJECTION_ONLY_UNIFORM_WITH_SOLID_PRESENT": "projection-only no-op; not a wall compatibility test",
            "ZERO_VELOCITY_FORCE_FREE": "force-free no-spontaneous-impact control",
            "PURE_GRADIENT_FIELD": "actual central D/G/m2_proj diagnostic projection control",
            "UNIFORM_ALL_DOMAIN": "production frozen negative control; rapid startup impulse decay is retained",
            "STREAMFUNCTION_LOCALIZED_V0": "near-solenoidal control, still separately failed on large solid velocity",
        },
    }


def _no_solid_c2_control(bundle: CaseBundle, initial: Any, profile: str, output_dir: Path) -> dict[str, Any]:
    sdf_empty = jnp.ones((bundle.N, bundle.N), dtype=bundle.p.dtype)
    empty = pf.make_solid(sdf_empty, bundle.p, cos_theta=float(bundle.case.get("cos_theta", 0.0)))
    u, v, _psi, details = build_c2_velocity(bundle, empty, phi=initial.phi)
    c2_empty = pf.State(phi=initial.phi, u=u, v=v, t=0.0)
    no_solid_bundle = CaseBundle(
        case_name=bundle.case_name + "_NO_SOLID",
        case=bundle.case,
        N=bundle.N,
        requested_dt=bundle.requested_dt,
        p=bundle.p,
        solid=empty,
        base=pf.State(phi=initial.phi, u=initial.u, v=initial.v, t=0.0),
        x0=bundle.x0,
        y0=bundle.y0,
        R=bundle.R,
        u_impact=bundle.u_impact,
        local_surface_top=bundle.local_surface_top,
        initial_gap=bundle.initial_gap,
        policy=bundle.policy,
    )
    run = run_trajectory(
        no_solid_bundle,
        "SDF_TAPERED_STREAMFUNCTION_V1",
        c2_empty,
        t_end=SHORT_HORIZON,
        profile=profile,
        output_dir=output_dir,
    )
    initial_speed = run["rows"][0]["max_speed_global"]
    peak_speed = max(row["max_speed_global"] for row in run["rows"])
    return {
        "initializer_parameters": details,
        "initial_max_speed": initial_speed,
        "peak_max_speed_over_0p24": peak_speed,
        "growth_over_initial": peak_speed - initial_speed,
        "all_finite": all(math.isfinite(row["max_speed_global"]) for row in run["rows"]),
        "trajectory_fingerprint": run["run_fingerprint"],
        "elapsed_seconds": run["elapsed_seconds_this_segment"],
        "trajectory": [
            {
                key: row[key]
                for key in (
                    "t",
                    "max_speed_global",
                    "v_liquid",
                    "v_core",
                    "kinetic_energy_physical",
                    "max_phi_overshoot",
                )
            }
            for row in run["rows"]
        ],
    }


def _run_temporal(ctx: dict[str, Any], profile: str, out_dir: Path) -> dict[str, Any]:
    if profile != "forensic":
        return {"status": "UNMEASURED", "reason": "quick profile is methodology-only and not N=192 evidence"}
    cases_to_run = [("flat_we100_ct050", candidate) for candidate in CANDIDATES] + [
        ("flat_we200_ct000", "SDF_TAPERED_STREAMFUNCTION_V1"),
        ("pillar_training", "SDF_TAPERED_STREAMFUNCTION_V1"),
    ]
    runs: dict[tuple[str, str, float], dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for case_name, candidate in cases_to_run:
        summaries.setdefault(case_name, {})[candidate] = {}
        for dt in DT_LEVELS:
            _progress(f"temporal start case={case_name} candidate={candidate} dt={dt:g}")
            bundle = _derive_bundle(case_name, ctx["cases"][case_name], 192, dt)
            state, details = build_candidate(bundle, candidate)
            initial_match = _state_hashes(state)
            run = run_trajectory(
                bundle,
                candidate,
                state,
                t_end=SHORT_HORIZON,
                profile=profile,
                frame_dt=FRAME_CADENCE,
                include_substep_ledger=True,
                capture_fields=True,
                output_dir=out_dir,
            )
            run["initializer_parameters"] = details
            run["initial_hashes"] = initial_match
            _progress(
                f"temporal complete case={case_name} candidate={candidate} dt={dt:g} "
                f"elapsed={run['elapsed_seconds_this_segment']:.1f}s"
            )
            runs[(case_name, candidate, dt)] = run
            summaries[case_name][candidate][f"dt_{dt:g}"] = {
                "dt": dt,
                "effective_dt": float(bundle.p.dt),
                "actual_frame_times": run["frame_times"],
                "initial_hashes": initial_match,
                "retention": _retention(run),
                "short_window_event": authenticate_impact(run, bundle, observed_horizon=SHORT_HORIZON),
                "rows": [
                    {
                        key: row.get(key)
                        for key in (
                            "step",
                            "t",
                            "v_liquid",
                            "v_core",
                            "v_gas",
                            "gap_phi05",
                            "gap_phi01",
                            "gap_phi05_over_dx",
                            "gap_phi05_over_eps",
                            "local_approach_speed_into_solid",
                            "contact_phi05",
                            "max_speed_global",
                            "kinetic_energy_physical",
                            "mass_Vphi",
                            "formal_mass_ratio",
                            "initial_D_div_Linf",
                            "cutcell_flux_div_Linf",
                            "max_embedded_wall_bilinear_normal_velocity",
                            "substep_ledger",
                        )
                    }
                    for row in run["rows"]
                ],
                "elapsed_seconds": run["elapsed_seconds_this_segment"],
                "run_fingerprint": run["run_fingerprint"],
            }
    # Extend the same C2 dt runs to 0.48 so contact authentication has a separate postcontact
    # interval. The equal-time label/retention matrix above remains anchored at 0, 0.08, 0.16, 0.24.
    event_screen_runs: dict[str, Any] = {}
    for case_name in ("flat_we100_ct050", "flat_we200_ct000", "pillar_training"):
        for dt in DT_LEVELS:
            _progress(f"event-screen continuation start case={case_name} dt={dt:g}")
            run = runs[(case_name, "SDF_TAPERED_STREAMFUNCTION_V1", dt)]
            bundle = _derive_bundle(case_name, ctx["cases"][case_name], 192, dt)
            if run["_last_step"] < int(round(EVENT_SCREEN_HORIZON / bundle.p.dt)):
                run = continue_trajectory(bundle, run, t_end=EVENT_SCREEN_HORIZON, profile=profile, output_dir=out_dir)
                runs[(case_name, "SDF_TAPERED_STREAMFUNCTION_V1", dt)] = run
            screen = authenticate_impact(run, bundle, observed_horizon=EVENT_SCREEN_HORIZON)
            _progress(
                f"event-screen complete case={case_name} dt={dt:g} verdict={screen['verdict']} "
                f"contact_time={screen.get('contact_time_interpolated_gap_threshold')}"
            )
            event_screen_runs[f"{case_name}:dt={dt:g}"] = screen
            summaries[case_name]["SDF_TAPERED_STREAMFUNCTION_V1"][f"dt_{dt:g}"]["event_screen_to_0p48"] = screen
            summaries[case_name]["SDF_TAPERED_STREAMFUNCTION_V1"][f"dt_{dt:g}"]["event_screen_horizon"] = (
                EVENT_SCREEN_HORIZON
            )
            summaries[case_name]["SDF_TAPERED_STREAMFUNCTION_V1"][f"dt_{dt:g}"]["actual_frame_times"] = run[
                "frame_times"
            ]
            summaries[case_name]["SDF_TAPERED_STREAMFUNCTION_V1"][f"dt_{dt:g}"]["run_fingerprint_at_0p48"] = run[
                "run_fingerprint"
            ]
            summaries[case_name]["SDF_TAPERED_STREAMFUNCTION_V1"][f"dt_{dt:g}"]["elapsed_seconds_total"] = run[
                "elapsed_seconds_total"
            ]
    # Pairwise observables at exact common physical times; no frame interpolation is used.
    pairwise: dict[str, Any] = {}
    for case_name, candidates in (
        ("flat_we100_ct050", CANDIDATES),
        ("flat_we200_ct000", ("SDF_TAPERED_STREAMFUNCTION_V1",)),
        ("pillar_training", ("SDF_TAPERED_STREAMFUNCTION_V1",)),
    ):
        pairwise[case_name] = {}
        for candidate in candidates:
            coarse = runs[(case_name, candidate, 0.002)]
            fine = runs[(case_name, candidate, 0.0005)]
            pairwise[case_name][candidate] = {}
            for t in (0.0, 0.08, 0.16, 0.24):
                a, b = _match_row(coarse, t), _match_row(fine, t)
                if a is None or b is None:
                    pairwise[case_name][candidate][f"t_{t:.2f}"] = {
                        "status": "UNMEASURED",
                        "reason": "no exact equal-time row",
                    }
                    continue
                pairwise[case_name][candidate][f"t_{t:.2f}"] = {
                    "time_alignment": "EXACT_PUBLIC_TIME_NO_INTERPOLATION",
                    "v_liquid": _relative_difference(
                        a["v_liquid"], b["v_liquid"], 0.2 * ctx["bundles"][case_name].u_impact
                    ),
                    "v_core": _relative_difference(a["v_core"], b["v_core"], 0.2 * ctx["bundles"][case_name].u_impact),
                    "R_retain": _relative_difference(
                        None if a["v_liquid"] is None else a["v_liquid"] / coarse["rows"][0]["v_liquid"],
                        None if b["v_liquid"] is None else b["v_liquid"] / fine["rows"][0]["v_liquid"],
                        0.2,
                    ),
                    "gap_phi05": _relative_difference(a["gap_phi05"], b["gap_phi05"], ctx["bundles"][case_name].p.dx),
                    "beta": _relative_difference(a["beta"], b["beta"], 1.0),
                    "max_speed_global_historical_metric": _relative_difference(
                        a["max_speed_global"], b["max_speed_global"], ctx["bundles"][case_name].u_impact
                    ),
                    "mass_ratio": _relative_difference(a["formal_mass_ratio"], b["formal_mass_ratio"], 1.0),
                }
    direct_fields = _direct_temporal_field_errors(runs)
    # Authenticate each candidate/dt at 0.48, then compare the primary flat We100
    # impact at exact event times. The historical 3% target is applied to changes
    # of impact observables, never to U_contact/u_impact or initial wall slip.
    event_metrics: dict[str, Any] = {}
    for case_name, candidate in cases_to_run:
        if candidate != "SDF_TAPERED_STREAMFUNCTION_V1":
            continue
        for dt in DT_LEVELS:
            run = runs[(case_name, candidate, dt)]
            bundle = _derive_bundle(case_name, ctx["cases"][case_name], 192, dt)
            auth = authenticate_impact(run, bundle, observed_horizon=EVENT_SCREEN_HORIZON)
            contact_row = next((row for row in run["rows"] if row["contact_phi05"]), None)
            initial_v = run["rows"][0]["v_liquid"]
            pre_row = None if auth.get("precontact_time") is None else _match_row(run, float(auth["precontact_time"]))
            contact_values = (
                None
                if contact_row is None
                else {
                    "contact_time": auth.get("contact_time_interpolated_gap_threshold"),
                    "contact_local_normal_speed": auth.get("contact_local_normal_approach_speed_positive_toward_solid"),
                    "precontact_retention": None
                    if pre_row is None or initial_v in (None, 0.0)
                    else pre_row["v_liquid"] / initial_v,
                    "precontact_local_normal_speed": auth.get(
                        "precontact_local_normal_approach_speed_positive_toward_solid"
                    ),
                    "beta_contact": contact_row["beta"],
                    "height_contact": contact_row["drop_height"],
                    "incoming_momentum_y": auth.get("incoming_liquid_momentum_and_energy_precontact", {}).get(
                        "liquid_momentum_y"
                    )
                    if auth.get("incoming_liquid_momentum_and_energy_precontact")
                    else None,
                    "incoming_kinetic_energy": auth.get("incoming_liquid_momentum_and_energy_precontact", {}).get(
                        "liquid_physical_kinetic_energy"
                    )
                    if auth.get("incoming_liquid_momentum_and_energy_precontact")
                    else None,
                }
            )
            event_metrics[f"{case_name}:dt={dt:g}"] = {"authentication": auth, "observables": contact_values}
    event_pairwise: dict[str, Any] = {}
    primary_events = {dt: event_metrics.get(f"flat_we100_ct050:dt={dt:g}", {}) for dt in DT_LEVELS}
    all_primary_authenticated = all(
        item.get("authentication", {}).get("verdict") == "IMPACT_AUTHENTICATED" and item.get("observables") is not None
        for item in primary_events.values()
    )
    max_change = None
    if all_primary_authenticated:
        coarse = primary_events[0.002]["observables"]
        fine = primary_events[0.0005]["observables"]
        event_pairwise["flat_we100_ct050_C2_dt0p002_vs_dt0p0005"] = {
            name: _relative_difference(coarse.get(name), fine.get(name), scale)
            for name, scale in (
                ("contact_time", FRAME_CADENCE),
                ("contact_local_normal_speed", 0.2 * ctx["bundles"]["flat_we100_ct050"].u_impact),
                ("precontact_retention", 0.2),
                ("precontact_local_normal_speed", 0.2 * ctx["bundles"]["flat_we100_ct050"].u_impact),
                ("beta_contact", 1.0),
                ("height_contact", ctx["bundles"]["flat_we100_ct050"].p.dx),
            )
        }
        target_changes = [
            row["relative"]
            for row in event_pairwise["flat_we100_ct050_C2_dt0p002_vs_dt0p0005"].values()
            if row.get("relative") is not None
        ]
        for t in (0.08, 0.16, 0.24):
            sample = pairwise["flat_we100_ct050"]["SDF_TAPERED_STREAMFUNCTION_V1"][f"t_{t:.2f}"]
            for name in ("R_retain", "gap_phi05", "beta"):
                value = sample[name].get("relative")
                if value is not None:
                    target_changes.append(value)
        max_change = max(target_changes, default=None)
    target = float(validation_module.PROVISIONAL_READINESS_TARGETS["key_observable_refinement_change"])
    refinement = "UNMEASURED" if max_change is None else ("PASS" if max_change <= target else "FAIL")
    return {
        "status": "MEASURED",
        "N": 192,
        "dt_levels": list(DT_LEVELS),
        "physical_sample_times": [0.0, 0.08, 0.16, 0.24],
        "event_authentication_extension_horizon": EVENT_SCREEN_HORIZON,
        "frame_cadence": FRAME_CADENCE,
        "runs": summaries,
        "equal_time_pairwise_coarse_dt_0p002_vs_fine_dt_0p0005": pairwise,
        "event_metrics_by_case_dt": event_metrics,
        "impact_observable_pairwise_refinement": event_pairwise,
        "primary_flat_all_dt_impacts_authenticated": all_primary_authenticated,
        "direct_field_errors": direct_fields,
        "existing_key_observable_refinement_target": target,
        "max_measured_retention_contact_morphology_relative_change": max_change,
        "short_window_temporal_target_status": refinement,
        "interpretation": (
            "No claim is made that U_contact equals u_impact. Global max_speed remains the unchanged "
            "historical metric; near-zero values use stated physical floors. Contact time, precontact "
            "retention/normal approach and morphology are compared as observables."
        ),
        "elapsed_seconds_total": float(sum(run["elapsed_seconds_total"] for run in runs.values())),
        "_runs": runs,
    }


def _direct_temporal_field_errors(runs: dict[tuple[str, str, float], dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for case_name, candidate in (("flat_we100_ct050", c) for c in CANDIDATES):
        coarse = runs[(case_name, candidate, 0.002)]["_fields"]
        fine = runs[(case_name, candidate, 0.0005)]["_fields"]
        output.setdefault(case_name, {})[candidate] = {}
        for t in (0.0, 0.08, 0.16, 0.24):
            ic = int(np.argmin(np.abs(coarse["times"] - t)))
            iff = int(np.argmin(np.abs(fine["times"] - t)))
            if not (
                math.isclose(coarse["times"][ic], t, abs_tol=1e-10)
                and math.isclose(fine["times"][iff], t, abs_tol=1e-10)
            ):
                raise AuditValidationError(f"direct field comparison lacks exact matched time t={t}")
            fields = {}
            scales = {"phi": 1.0, "u": 0.2, "v": 0.2}
            for name in ("phi", "u", "v"):
                a, b = coarse[name][ic], fine[name][iff]
                solver_error = _field_error(a, b, physical_scale=scales[name])
                sample_a = _downsample_mean(a, 3)
                sample_b = _downsample_mean(b, 3)
                # Exact generator representation: phi float32, u/v float16 after block mean.
                cast = np.float32 if name == "phi" else np.float16
                sample_a = sample_a.astype(cast)
                sample_b = sample_b.astype(cast)
                export_error = _field_error(sample_a, sample_b, physical_scale=scales[name])
                quantization_floor = float(
                    np.sqrt(np.mean((sample_a.astype(np.float64) - sample_b.astype(np.float64)) ** 2))
                )
                fields[name] = {
                    "solver_grid": solver_error,
                    "export_downsampled_ds3_cast": export_error,
                    "export_dtype": np.dtype(cast).name,
                    "same_shape": list(sample_a.shape) == list(sample_b.shape),
                    "time": float(t),
                    "quantized_export_L2_rms": quantization_floor,
                }
            output[case_name][candidate][f"t_{t:.2f}"] = fields
    return output


def _make_event_runs(
    ctx: dict[str, Any], temporal: dict[str, Any], profile: str, out_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    if profile != "forensic":
        return {
            case: {"verdict": "INCONCLUSIVE", "reason": "quick profile cannot authenticate impact"}
            for case in CASE_NAMES
        }, {}
    temporal_runs = temporal["_runs"]
    all_runs: dict[str, dict[str, Any]] = {}
    events: dict[str, Any] = {}
    for case_name in CASE_NAMES:
        bundle = _derive_bundle(case_name, ctx["cases"][case_name], 192, 0.002)
        if case_name in ("flat_we100_ct050", "flat_we200_ct000", "pillar_training"):
            run = temporal_runs[(case_name, "SDF_TAPERED_STREAMFUNCTION_V1", 0.002)]
            if run["_last_step"] < int(round(EVENT_SCREEN_HORIZON / bundle.p.dt)):
                run = continue_trajectory(bundle, run, t_end=EVENT_SCREEN_HORIZON, profile=profile, output_dir=out_dir)
        else:
            initial, details = build_candidate(bundle, "SDF_TAPERED_STREAMFUNCTION_V1")
            run = run_trajectory(
                bundle,
                "SDF_TAPERED_STREAMFUNCTION_V1",
                initial,
                t_end=EVENT_SCREEN_HORIZON,
                profile=profile,
                frame_dt=FRAME_CADENCE,
                include_substep_ledger=True,
                capture_fields=True,
                output_dir=out_dir,
            )
            run["initializer_parameters"] = details
        screen = authenticate_impact(run, bundle, observed_horizon=EVENT_SCREEN_HORIZON)
        all_runs[case_name] = run
        events[case_name] = {
            "short_event_screen": screen,
            "short_horizon_status": "NO_CONTACT_WITHIN_OBSERVED_WINDOW"
            if screen["verdict"] == "NO_CONTACT_WITHIN_OBSERVED_WINDOW"
            else screen["verdict"],
            "screen_horizon": EVENT_SCREEN_HORIZON,
            "same_run_extended_to_T8": True,
            "precontact_rows": [row for row in run["rows"] if row["t"] <= EVENT_SCREEN_HORIZON],
            "run_fingerprint_at_screen": run["run_fingerprint"],
        }
    return events, all_runs


def _full_horizon_runs(
    ctx: dict[str, Any], event_runs: dict[str, dict[str, Any]], profile: str, out_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    if profile != "forensic":
        return {"status": "UNMEASURED", "reason": "quick profile is methodology-only"}, {}
    outputs: dict[str, Any] = {}
    for case_name in CASE_NAMES:
        _progress(f"T8 continuation start case={case_name}")
        bundle = _derive_bundle(case_name, ctx["cases"][case_name], 192, 0.002)
        run = event_runs[case_name]
        if run["_last_step"] < int(round(DATASET_HORIZON / bundle.p.dt)):
            run = continue_trajectory(bundle, run, t_end=DATASET_HORIZON, profile=profile, output_dir=None)
        # Canonical read-only generator gates. This is NOT an official build_case/export acceptance:
        # the t=0 velocity was injected externally and is permanently marked diagnostic-only.
        frames = run["_fields"]
        nonzero_frames = frames["times"] > 0.0
        history_phi = frames["phi"][nonzero_frames]
        history_u = frames["u"][nonzero_frames]
        history_v = frames["v"][nonzero_frames]
        ok, diagnostics = generator._diagnose(
            run["_initial_state"],
            history_phi,
            history_u,
            history_v,
            bundle.solid,
            bundle.p,
            max_phi_overshoot=0.02,
            max_solid_leak=5e-4,
            min_total_mass_ratio=0.995,
            max_total_mass_ratio=1.005,
            max_speed=5.0,
        )
        frame_schedule = _validate_frame_times(frames["times"], t_end=DATASET_HORIZON, cadence=FRAME_CADENCE)
        event = authenticate_impact(run, bundle, observed_horizon=DATASET_HORIZON)
        row = {
            "status": "MEASURED",
            "case_name": case_name,
            "diagnostic_only": True,
            "production_lineage_eligible": False,
            "contract12": int(pf.SOLVER_CONTRACT_VERSION) == 12,
            "N": bundle.N,
            "effective_dt": float(bundle.p.dt),
            "horizon": DATASET_HORIZON,
            "frame_cadence": FRAME_CADENCE,
            "frame_times": frames["times"].tolist(),
            "frame_schedule_validation": frame_schedule,
            "trajectory_fingerprint": run["run_fingerprint"],
            "binding": run["binding"],
            "case_fingerprint": _case_fingerprint(bundle),
            "geometry_fingerprint": _geometry_hash(bundle.solid),
            "initial_hashes": _state_hashes(run["_initial_state"]),
            "initial_geometry_hash": _geometry_hash(bundle.solid),
            "impact_authentication": event,
            "first_contact_time": event.get("contact_time_interpolated_gap_threshold"),
            "onset_type": "thresholded local SDF gap event; approach and postresponse independently required"
            if event.get("contact_time_discrete") is not None
            else "NO_LOCAL_PHI05_CONTACT",
            "canonical_read_only_generator_diagnose_ok": bool(ok),
            "canonical_generator_diagnostics": diagnostics,
            "direct_field_integrity": {
                "finite_phi_u_v": bool(
                    np.isfinite(frames["phi"]).all()
                    and np.isfinite(frames["u"]).all()
                    and np.isfinite(frames["v"]).all()
                ),
                "phi_solver_dtype": str(frames["phi"].dtype),
                "u_solver_dtype": str(frames["u"].dtype),
                "v_solver_dtype": str(frames["v"].dtype),
                "frame_shape": list(frames["phi"].shape),
                "export_schema_dtypes_unchanged": {
                    "phi": "float32 block mean ds=3",
                    "u": "float16 block mean ds=3",
                    "v": "float16 block mean ds=3",
                },
            },
            "observables_at_frames": [
                {
                    key: rowf.get(key)
                    for key in (
                        "t",
                        "v_liquid",
                        "v_core",
                        "local_approach_speed_into_solid",
                        "gap_phi05",
                        "gap_phi01",
                        "beta",
                        "drop_height",
                        "centroid_y",
                        "max_speed_global",
                        "formal_mass_ratio",
                        "max_phi_overshoot",
                        "deep_solid_leak",
                        "initial_D_div_Linf",
                        "cutcell_flux_div_Linf",
                        "max_embedded_wall_bilinear_normal_velocity",
                    )
                }
                for rowf in run["rows"]
                if math.isclose((rowf["t"] / FRAME_CADENCE) % 1.0, 0.0, abs_tol=1e-10)
            ],
            "elapsed_seconds": float(run["elapsed_seconds_total"]),
        }
        outputs[case_name] = row
        _progress(
            f"T8 complete case={case_name} verdict={event['verdict']} elapsed={run['elapsed_seconds_total']:.1f}s"
        )
        # Save only diagnostic histories in the ignored artifacts tree, never under data/base/train/test.
        fingerprint = run["run_fingerprint"]
        path = out_dir / "full_horizon" / f"{case_name}_diagnostic_only_{fingerprint[:12]}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            times=frames["times"],
            phi=frames["phi"],
            u=frames["u"],
            v=frames["v"],
            diagnostic_only=np.asarray(True),
            production_lineage_eligible=np.asarray(False),
            trajectory_fingerprint=np.asarray(fingerprint),
        )
        row["diagnostic_artifact"] = str(path.relative_to(REPO))
        row["diagnostic_artifact_sha256"] = _sha_file(path)
        row["diagnostic_marker"] = {"diagnostic_only": True, "production_lineage_eligible": False}
        # Drop the in-memory full history after canonical diagnosis/export checks; the compressed
        # trace remains in the ignored artifacts tree. No downstream stage consumes these arrays.
        if "_field_lists" in run:
            for values in run["_field_lists"].values():
                values.clear()
        run["_fields"] = None
        run["_state"] = None
        event_runs.pop(case_name, None)
    return {
        "status": "MEASURED",
        "cases": outputs,
        "horizon": DATASET_HORIZON,
        "frame_cadence": FRAME_CADENCE,
        "elapsed_seconds_total": float(sum(entry["elapsed_seconds"] for entry in outputs.values())),
    }, {}


def _spatial_reaudit(ctx: dict[str, Any], profile: str, out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if profile != "forensic":
        return {"status": "UNMEASURED", "reason": "quick profile is methodology-only"}, {}
    case_name = "flat_we100_ct050"
    case = ctx["cases"][case_name]
    base192 = _derive_bundle(case, case, 192, 0.002)
    y0_192 = base192.y0
    fixed_eps = float(base192.p.eps)
    runs: dict[str, Any] = {}
    records: dict[str, Any] = {
        "case_name": case_name,
        "physical_geometry_identity": case,
        "dt_requested": 0.002,
        "frame_cadence": FRAME_CADENCE,
        "comparison_horizon": 0.48,
        "grid_field_mapping": "center-aligned linear scipy map_coordinates(mode='grid-wrap'); common target N=192",
        "families": {},
    }
    for family in ("fixed_eps_over_dx_production_family", "fixed_physical_eps_diagnostic_family"):
        family_rows: dict[str, Any] = {}
        family_runs = {}
        for n in (144, 192):
            _progress(f"spatial re-audit start family={family} N={n}")
            case_variant = dict(case)
            if family == "fixed_physical_eps_diagnostic_family":
                case_variant["eps"] = fixed_eps
                case_variant["y0"] = y0_192
            bundle = _derive_bundle(case_name, case_variant, n, 0.002)
            state, details = build_candidate(bundle, "SDF_TAPERED_STREAMFUNCTION_V1")
            run = run_trajectory(
                bundle,
                "SDF_TAPERED_STREAMFUNCTION_V1",
                state,
                t_end=0.48,
                profile=profile,
                frame_dt=FRAME_CADENCE,
                include_substep_ledger=False,
                capture_fields=True,
                output_dir=out_dir,
            )
            _progress(
                f"spatial re-audit complete family={family} N={n} elapsed={run['elapsed_seconds_this_segment']:.1f}s"
            )
            family_runs[n] = run
            family_rows[str(n)] = {
                "N": n,
                "eps": float(bundle.p.eps),
                "eps_over_dx": float(bundle.p.eps / bundle.p.dx),
                "R": bundle.R,
                "x0": bundle.x0,
                "y0": bundle.y0,
                "local_surface_top": bundle.local_surface_top,
                "initial_gap": bundle.initial_gap,
                "dt": float(bundle.p.dt),
                "frame_times": run["frame_times"],
                "initial_hashes": _state_hashes(state),
                "initial_geometry_hash": _geometry_hash(bundle.solid),
                "trajectory_fingerprint": run["run_fingerprint"],
                "impact": authenticate_impact(run, bundle, observed_horizon=0.48),
                "observables": [
                    {
                        key: row.get(key)
                        for key in (
                            "t",
                            "gap_phi05",
                            "local_approach_speed_into_solid",
                            "v_liquid",
                            "beta",
                            "drop_height",
                            "centroid_y",
                            "max_speed_global",
                            "formal_mass_ratio",
                        )
                    }
                    for row in run["rows"]
                    if row["t"] in (0.0, 0.08, 0.16, 0.24, 0.32, 0.4, 0.48)
                ],
                "initializer": details,
            }
        mapped: dict[str, Any] = {}
        fields192 = family_runs[192]["_fields"]
        fields144 = family_runs[144]["_fields"]
        for t in fields192["times"]:
            i192 = int(np.argmin(np.abs(fields192["times"] - t)))
            i144 = int(np.argmin(np.abs(fields144["times"] - t)))
            mapped[f"t_{t:.2f}"] = {}
            for name, scale in (("phi", 1.0), ("u", 0.2), ("v", 0.2)):
                n144_field = _map_common_grid(fields144[name][i144], fields192[name][i192].shape)
                mapped[f"t_{t:.2f}"][name] = _field_error(n144_field, fields192[name][i192], physical_scale=scale)
        records["families"][family] = {
            "runs": family_rows,
            "common_grid_field_errors_N144_mapped_to_N192": mapped,
            "interpretation": (
                _concat_text(
                    "fixed eps/dx changes physical diffuse-interface thickness and, in this production case, ",
                    "the derived clearance/y0; combined mesh+interface-width+initial-clearance sensitivity",
                )
                if family == "fixed_eps_over_dx_production_family"
                else _concat_text(
                    "fixed physical eps and a common y0; additional same-diffuse-model spatial ",
                    "discretization diagnosis",
                )
            ),
        }
        runs[family] = family_runs
    target = float(validation_module.PROVISIONAL_READINESS_TARGETS["key_observable_refinement_change"])
    records["spatial_key_observable_target"] = target
    records["status"] = "MEASURED"
    records["spatial_refinement_pass"] = False
    records["reason"] = _concat_text(
        "bounded N=144/N=192 diagnostic results are descriptive; the historical formal ",
        "SPATIAL_REFINEMENT=FAIL is not cleared by this non-official candidate audit",
    )
    return records, runs


def _impact_parameters(events: dict[str, Any], bundles: dict[str, CaseBundle]) -> dict[str, Any]:
    rows = {}
    for case_name in CASE_NAMES:
        bundle = bundles[case_name]
        event = events.get(case_name, {})
        u_contact = event.get("contact_local_normal_approach_speed_positive_toward_solid")
        ratio = None if u_contact is None else float(u_contact / bundle.u_impact)
        rows[case_name] = {
            "input_We": float(bundle.p.We),
            "input_Re": float(bundle.p.Re),
            "R": bundle.R,
            "input_u_impact": bundle.u_impact,
            "nominal_D_reference_from_PhaseFieldParams_doc": 1.0,
            "rho_l": float(bundle.p.rho_l),
            "rho_g": float(bundle.p.rho_g),
            "nu_l": float(bundle.p.nu_l),
            "nu_g": float(bundle.p.nu_g),
            "capillary_force_coefficient": float(pf.SIGMA_NORM / bundle.p.We),
            "kinematic_We_auxiliary_export": float(bundle.p.We * bundle.u_impact**2),
            "kinematic_Re_auxiliary_export": float(bundle.p.Re * abs(bundle.u_impact)),
            "U_contact_local_normal": u_contact,
            "U_contact_over_case_u_impact": ratio,
            "illustrative_We_contact_over_We_times_uimpact_squared": None if ratio is None else ratio**2,
            "illustrative_We_contact_over_solver_We_at_Uref_1": None if u_contact is None else (u_contact**2),
            "illustrative_Re_contact_over_auxiliary_kinematic_Re": None if ratio is None else abs(ratio),
            "nondimensional_semantics": _concat_text(
                "PhaseFieldParams documents D=1, rho_l, and unit reference U; nu_l=1/Re and ",
                "capillary coefficient=SIGMA_NORM/We. generate_dataset separately records ",
                "kinematic_We=We*u_impact^2 and kinematic_Re=Re*abs(u_impact). Diagnostic ",
                "U_contact is local normal speed into solid. These are distinct nominal ",
                "parameter-conditioned and contact-calibrated interpretations.",
            ),
            "solver_or_sample_labels_reparameterized": False,
        }
    return {
        "status": "UNRESOLVED",
        "case_semantics": rows,
        "reason": _concat_text(
            "contact-speed estimates are diagnostic only; no physical calibration establishes a single ",
            "equivalent nominal We/Re mapping across all geometries, and no PDE parameter or ML label is changed",
        ),
        "publication_claim": _concat_text(
            "unqualified physical-impact-strength claims remain blocked; solver-surrogate conditioning is ",
            "distinct from calibrated contact-impact conditioning",
        ),
    }


def _generator_thresholds_unchanged() -> dict[str, Any]:
    thresholds = {
        "max_phi_overshoot": 0.02,
        "max_solid_leak": 5e-4,
        "min_total_mass_ratio": 0.995,
        "max_total_mass_ratio": 1.005,
        "max_speed": 5.0,
        "key_observable_refinement_change": float(
            validation_module.PROVISIONAL_READINESS_TARGETS["key_observable_refinement_change"]
        ),
    }
    expected = {
        "max_phi_overshoot": 0.02,
        "max_solid_leak": 5e-4,
        "min_total_mass_ratio": 0.995,
        "max_total_mass_ratio": 1.005,
        "max_speed": 5.0,
        "key_observable_refinement_change": 0.03,
    }
    if thresholds != expected:
        raise AuditValidationError(f"historical gates changed during diagnostic: {thresholds}")
    return thresholds


def _decision(
    initial_verdict: str, fleet_verdict: str, readiness: dict[str, str], parameter_status: str, profile: str
) -> tuple[str, str, str]:
    if profile != "forensic" or initial_verdict == "INCONCLUSIVE" or fleet_verdict == "INCONCLUSIVE":
        return (
            "D_INCONCLUSIVE_OR_INCOMPLETE",
            "MISSING_OR_UNMEASURED_FORENSIC_EVIDENCE",
            "complete only the first missing bounded forensic stage; do not generate training data",
        )
    if initial_verdict == "NO_FEASIBLE_INITIALIZER_UNDER_CONTRACT12":
        return (
            "C_NO_FEASIBLE_INITIALIZER_WITH_CURRENT_FORMULATION",
            "INITIAL_DIVERGENCE_OR_SOLID_COMPATIBILITY",
            _concat_text(
                "authorize a separately scoped projection/embedded-wall-BC formulation decision; ",
                "do not patch contract-12 evolution",
            ),
        )
    if initial_verdict in ("INITIALIZER_COMPATIBLE", "INITIALIZER_PARTIALLY_COMPATIBLE"):
        failing = [name for name, status in readiness.items() if status != "PASS"]
        if fleet_verdict != "ALL_REQUIRED_IMPACTS_AUTHENTICATED":
            primary = "REQUIRED_PILLAR_OR_COMPLEX_IMPACT_AUTHENTICATION"
            action = (
                "extend or diagnose only the failed local-support event protocol on the named canary; keep L1B blocked"
            )
        elif parameter_status != "RESOLVED":
            primary = "IMPACT_PARAMETER_SEMANTICS"
            action = _concat_text(
                "perform an independently authorized physical nondimensionalization/calibration review ",
                "before publication or promotion claims",
            )
        elif failing:
            # A single target-critical blocker is selected deterministically; other fails remain listed.
            primary = failing[0]
            action = _concat_text(
                "run the smallest separate formal refinement/label audit for this one blocker; ",
                "no contract-13 promotion in 2s",
            )
        else:
            return (
                "A_READY_FOR_SEPARATE_INITIALIZER_PROMOTION_REVIEW",
                "NONE",
                "begin a separately authorized contract-13 promotion review",
            )
        return "B_INITIALIZATION_HELPFUL_BUT_REFINEMENT_OR_GEOMETRY_BLOCKS", primary, action
    return (
        "D_INCONCLUSIVE_OR_INCOMPLETE",
        "INITIALIZER_CLASSIFICATION_UNRESOLVED",
        "resolve missing compatibility measurements without expanding the candidate search",
    )


def _render_markdown(report: dict[str, Any]) -> str:
    status = report["stage_decision"]
    lines = [
        "# L1A-2s — divergence-free, solid-compatible impact initialization re-audit",
        "",
        _concat_text(
            f"- Stage: **{STAGE_VERSION}** · base/main: `{BASE_MAIN_SHA}` · ",
            f"solver contract: **12** · policy: `{EXPECTED_POLICY}`",
        ),
        _concat_text(
            "- Diagnostic-only; production physics/default initialization unchanged: ",
            f"**{report['production_freeze']['production_semantics_changed']}**",
        ),
        f"- Stage decision: **{status['decision']}** · primary blocker: **{status['primary_blocker']}**",
        _concat_text(
            f"- Initialization verdict: **{report['initializer_verdict']}** · ",
            f"fleet impact verdict: **{report['fleet_impact_verdict']}**",
        ),
        "",
        "## Scientific decision",
        "",
        report["decision_summary"],
        "",
        "## Gate summary",
        "",
        "| Gate | Status | Evidence |",
        "|---|---|---|",
    ]
    for key in ("TEMPORAL_REFINEMENT", "SPATIAL_REFINEMENT", "DIRECT_FIELD_LABELS", "GENERATOR_QUALITY_GATES"):
        lines.append(
            _concat_text(
                f"| {key} | **{report['readiness'][key]}** | See `gate_and_blocker_matrix.json` ",
                "and the corresponding audit matrix |",
            )
        )
    lines.extend(
        [
            "",
            "## Initialization compatibility",
            "",
            _concat_text(
                "C2 is measured using the live central-difference operators and the ",
                "actual case SDF/cut-cell geometry. ",
                "The open-face FV divergence uses an explicit arithmetic cell-to-face velocity reconstruction and ",
                "shared embedded apertures; embedded wall flux uses bilinear cell-velocity interpolation at wall ",
                "centroids. Neither is silently identified with the periodic central-D projection.",
            ),
            "",
            f"Predeclared acceptance: `{json.dumps(INITIAL_ACCEPTANCE, sort_keys=True)}`",
            "",
            "## Impact, refinement, and parameter semantics",
            "",
            _concat_text(
                "- Full-horizon verdicts are individual per canary; see `bounded_full_horizon_matrix.json` ",
                "and `impact_event_matrix.json`.",
            ),
            _concat_text(
                "- Nominal We/Re are not relabelled from the diagnostic contact speed: ",
                f"**{report['impact_parameter_semantics']}**.",
            ),
            _concat_text(
                "- The inherited 3% key-observable target is not applied to `U_contact/u_impact`, nor to ",
                "initial divergence or wall slip. Historical max-speed and official spatial/refinement ",
                "blockers remain unchanged.",
            ),
            "",
            "## Frozen lineage and next action",
            "",
            "- `SOLVER_CONTRACT_VERSION=12`; production timestep policy remains `impact_phase_cap_dx2_v1`.",
            _concat_text(
                "- Candidate promotion: **false**; contract-13 promotion: **not performed**; ",
                "official data release: **blocked**.",
            ),
            f"- Exactly one next action: **{status['next_action']}**",
            "",
            _concat_text(
                "See the JSON evidence set and `manifest.json` for per-file/source hashes, cache identity, ",
                "dtypes, and runtime.",
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _assemble_evidence(
    *,
    profile: str,
    out_dir: Path,
    preflight: dict[str, Any],
    initial: dict[str, Any],
    startup: dict[str, Any] | None,
    controls: dict[str, Any] | None,
    temporal: dict[str, Any] | None,
    events: dict[str, Any] | None,
    full: dict[str, Any] | None,
    spatial: dict[str, Any] | None,
    direct_field: dict[str, Any] | None,
    elapsed: float,
) -> dict[str, Any]:
    c_matrix = initial["constraint_matrix"]
    init_verdict = initial["verdict"]
    event_matrix = events or {case: {"verdict": "INCONCLUSIVE", "reason": "stage absent"} for case in CASE_NAMES}
    full_status = full or {"status": "UNMEASURED", "reason": "full horizon stage absent"}
    if profile != "forensic" or full_status.get("status") != "MEASURED":
        # Short-window absences never assemble into a fleet-level no-impact verdict.
        fleet = "INCONCLUSIVE"
    else:
        fleet = _fleet_verdict(
            {
                case: full_status.get("cases", {})
                .get(case, {})
                .get("impact_authentication", event_matrix.get(case, {}).get("short_event_screen", {}))
                for case in CASE_NAMES
            }
        )
    temporal_status = "UNMEASURED"
    if temporal and temporal.get("status") == "MEASURED":
        temporal_status = temporal.get("short_window_temporal_target_status", "UNMEASURED")
    spatial_status = "UNMEASURED"
    if spatial and spatial.get("status") == "MEASURED":
        spatial_status = "FAIL" if not spatial.get("spatial_refinement_pass", False) else "PASS"
    direct_status = "UNMEASURED"
    if direct_field and direct_field.get("status") == "MEASURED":
        direct_status = direct_field.get("status_gate", "FAIL")
    generator_status = "UNMEASURED"
    if full and full.get("status") == "MEASURED":
        outcomes = [
            bool(row.get("canonical_read_only_generator_diagnose_ok")) for row in full.get("cases", {}).values()
        ]
        generator_status = "PASS" if len(outcomes) == len(CASE_NAMES) and all(outcomes) else "FAIL"
    readiness = {
        "TEMPORAL_REFINEMENT": temporal_status,
        "SPATIAL_REFINEMENT": spatial_status,
        "DIRECT_FIELD_LABELS": direct_status,
        "GENERATOR_QUALITY_GATES": generator_status,
    }
    thresholds = _generator_thresholds_unchanged()
    statuses = {
        "L1A_STATUS": "BLOCKED",
        "L1B_DATA": "L1B_DATA_NOT_READY",
        "SOLVER_CONTRACT_VERSION": 12,
        "production_timestep_policy": EXPECTED_POLICY,
        "N_DT": "TARGET_CRITICAL (unchanged; diagnostic improvement does not satisfy formal exit tests)",
        "SPATIAL_REFINEMENT": "FAIL (historical official status unchanged)"
        if spatial_status != "PASS"
        else "FAIL (historical official status unchanged; diagnostic spatial subset does not clear official gate)",
        "W_CONTACT_ANGLE": "OPEN (unchanged)",
        "P_VARDENS_PROJ": "OPEN (unchanged)",
        "D_FRESH_TRAIN_CONTRACT": "OPEN (unchanged; L1B-1 task)",
        "candidate_promoted": False,
        "production_semantics_changed": False,
        "official_dataset_written": False,
        "bulk_training_data_generated": False,
        "SOLVER_SURROGATE_DATA_READY": False,
        "PHYSICAL_PUBLICATION_DATA_READY": False,
    }
    # Contact semantics are assembled by the caller after full trajectory rows exist.
    return {
        "stage": STAGE,
        "stage_version": STAGE_VERSION,
        "profile": profile,
        "base_main_sha": BASE_MAIN_SHA,
        "pr21": PR21,
        "preflight": preflight,
        "initializer_verdict": init_verdict,
        "fleet_impact_verdict": fleet,
        "readiness": readiness,
        "historical_statuses": statuses,
        "thresholds": thresholds,
        "initialization_constraint_summary": {
            case: {
                candidate: {
                    "pass": row.get("initial_constraints_pass"),
                    "checks": row.get("acceptance_checks"),
                    "D_Linf": row.get("initial_D_div_Linf"),
                    "FV_Linf": row.get("cutcell_flux_div_Linf"),
                    "deep_solid": row.get("max_abs_velocity_deep_solid"),
                    "nearwall_chi": row.get("max_abs_velocity_nearwall_chi_ge_0p01"),
                    "wall_normal": row.get("max_embedded_wall_bilinear_normal_velocity"),
                }
                for candidate, row in c_matrix[case].items()
            }
            for case in c_matrix
        },
        "startup_ledger_status": "MEASURED" if startup is not None else "UNMEASURED",
        "control_status": "MEASURED" if controls is not None else "UNMEASURED",
        "temporal_status": temporal_status,
        "event_status": "MEASURED" if events is not None else "UNMEASURED",
        "full_horizon_status": full_status.get("status", "UNMEASURED"),
        "spatial_status": spatial_status,
        "direct_field_status": direct_status,
        "elapsed_seconds": float(elapsed),
        "decision_summary": (
            "L1A-2s is a diagnostic-only qualification. No candidate can promote itself; all unresolved "
            "numerical, geometry, parameter, and official lineage gates stay fail-closed."
        ),
    }


def _direct_field_gate(temporal: dict[str, Any] | None, full: dict[str, Any] | None) -> dict[str, Any]:
    if not temporal or temporal.get("status") != "MEASURED":
        return {"status": "UNMEASURED", "status_gate": "UNMEASURED", "reason": "temporal field comparison absent"}
    errors = temporal.get("direct_field_errors", {})
    entries = []
    for by_candidate in errors.values():
        for by_time in by_candidate.values():
            for by_field in by_time.values():
                for metrics in by_field.values():
                    entries.append(metrics)
    finite = all(
        math.isfinite(metric["solver_grid"]["Linf"]) and math.isfinite(metric["export_downsampled_ds3_cast"]["Linf"])
        for metric in entries
    )
    # There is no universal direct-field 3% gate in the current contract. We therefore
    # report norms and quantization floors but do not invent a pass threshold.
    return {
        "status": "MEASURED",
        "status_gate": "UNMEASURED",
        "finite": finite,
        "field_errors": errors,
        "reason": (
            "direct phi/u/v differences are measured, but the repository has no approved universal field-error "
            "tolerance; morphology cannot waive these label defects"
        ),
        "sample_dtypes": {"phi": "float32", "u": "float16", "v": "float16"},
        "diagnostic_only": True,
    }


def _full_horizon_quality(full: dict[str, Any] | None) -> dict[str, Any]:
    if not full or full.get("status") != "MEASURED":
        return {"status": "UNMEASURED", "cases": {}, "reason": "T=8 canonical diagnostic check did not run"}
    return {
        "status": "MEASURED",
        "diagnostic_only": True,
        "production_build_case_route_used": False,
        "cases": {
            case: {
                "canonical_read_only_generator_diagnose_ok": row["canonical_read_only_generator_diagnose_ok"],
                "diagnostics": row["canonical_generator_diagnostics"],
                "trajectory_fingerprint": row["trajectory_fingerprint"],
                "diagnostic_only": row["diagnostic_only"],
                "production_lineage_eligible": row["production_lineage_eligible"],
            }
            for case, row in full["cases"].items()
        },
        "all_four_cases_measured": len(full["cases"]) == len(CASE_NAMES),
    }


def _write_required_evidence(
    *,
    profile: str,
    out_dir: Path,
    preflight: dict[str, Any],
    initial: dict[str, Any],
    startup: dict[str, Any] | None,
    controls: dict[str, Any] | None,
    temporal: dict[str, Any] | None,
    events: dict[str, Any] | None,
    full: dict[str, Any] | None,
    spatial: dict[str, Any] | None,
    parameter_semantics: dict[str, Any] | None,
    report_base: dict[str, Any],
    elapsed: float,
) -> dict[str, Any]:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    design = {
        "stage": STAGE,
        "stage_version": STAGE_VERSION,
        "candidates": {
            "C0": "UNIFORM_ALL_DOMAIN, exact production build_case negative control; no promotion",
            "C1": (
                "STREAMFUNCTION_LOCALIZED_V0, exact current droplet_initial_state streamfunction u/v, "
                "case-specific x0/y0; solid velocity independently checked"
            ),
            "C2": (
                "SDF_TAPERED_STREAMFUNCTION_V1, isolated diagnostic builder in this module; formula, taper "
                "and amplitude rule included in candidate details"
            ),
            "C3": {
                "status": "NOT_RUN",
                "reason": (
                    "C2 is evaluated first; optional constrained solve is prohibited unless C2 clearly fails "
                    "a declared compatibility property"
                ),
            },
        },
        "acceptance": INITIAL_ACCEPTANCE,
        "physical_parameters_and_geometry": (
            "Derived per exact study_cases() from build_case, local _local_surface_top, actual SDF, "
            "case x0/R, and policy schedule. No shared y0 or geometry mutation."
        ),
        "phase_and_operator_freeze": (
            "C0/C1/C2 share bitwise phi, Solid geometry, M/eps/We/Re/cos_theta, phase transport, wetting, "
            "rho/nu, capillary force, eta_pen and all production operators; C2 changes only t=0 u/v."
        ),
        "initializer_scale": (
            "one deterministic scalar per geometry, set by V_i-weighted phi>=0.9 core mean; "
            "no input-u-impact equality claim beyond this declared target."
        ),
        "return_flow_requirement": (
            "periodic localized downward motion requires a return flow; ambient/farfield and "
            "return-flow peak are reported separately."
        ),
    }
    _write_json(
        EVIDENCE_DIR / "source_and_prerequisite_map.json",
        {
            **preflight,
            "operator_map": _source_operator_map(),
            "case_identities": {name: _metadata(bundle) for name, bundle in initial.get("bundles", {}).items()}
            if "bundles" in initial
            else preflight["case_definitions"],
        },
    )
    _write_json(EVIDENCE_DIR / "initializer_design_contract.json", design)
    _write_json(EVIDENCE_DIR / "initializer_constraint_matrix.json", initial["constraint_matrix"])
    _write_json(EVIDENCE_DIR / "initial_velocity_region_budget.json", initial["region_budget"])
    _write_json(
        EVIDENCE_DIR / "projection_brinkman_startup_ledger.json",
        startup or {"status": "UNMEASURED", "reason": "startup ledger stage absent"},
    )
    _write_json(
        EVIDENCE_DIR / "control_matrix.json",
        controls or {"status": "UNMEASURED", "reason": "negative-control stage absent"},
    )
    _write_json(
        EVIDENCE_DIR / "temporal_retention_matrix.json",
        {
            key: val
            for key, val in (temporal or {"status": "UNMEASURED", "reason": "temporal stage absent"}).items()
            if key != "_runs"
        },
    )
    _write_json(
        EVIDENCE_DIR / "impact_event_matrix.json", events or {"status": "UNMEASURED", "reason": "event stage absent"}
    )
    _write_json(
        EVIDENCE_DIR / "impact_parameter_semantics.json",
        parameter_semantics or {"status": "UNMEASURED", "reason": "full-horizon contact-speed semantics absent"},
    )
    _write_json(
        EVIDENCE_DIR / "spatial_temporal_reaudit.json",
        spatial or {"status": "UNMEASURED", "reason": "spatial stage absent"},
    )
    direct = _direct_field_gate(temporal, full)
    _write_json(EVIDENCE_DIR / "direct_field_quality_matrix.json", direct)
    full_public = (
        {key: value for key, value in full.items() if key != "_runs"}
        if full
        else {"status": "UNMEASURED", "reason": "full-horizon stage absent"}
    )
    _write_json(EVIDENCE_DIR / "bounded_full_horizon_matrix.json", full_public)
    thresholds = _generator_thresholds_unchanged()
    readiness = report_base["readiness"]
    primary_blocker = report_base.get("primary_blocker", "MISSING_OR_UNMEASURED_FORENSIC_EVIDENCE")
    next_action = report_base.get("next_action", "complete the first missing bounded forensic stage")
    gates = {
        "stage": STAGE,
        "quality_gates": {
            "temporal_refinement": readiness["TEMPORAL_REFINEMENT"],
            "spatial_refinement": readiness["SPATIAL_REFINEMENT"],
            "direct_field_labels": readiness["DIRECT_FIELD_LABELS"],
            "generator_quality_gates": readiness["GENERATOR_QUALITY_GATES"],
        },
        "historical_blockers_unchanged": {
            "L1B_DATA_NOT_READY": True,
            "L1A_STATUS_BLOCKED": True,
            "N_DT_TARGET_CRITICAL": True,
            "SPATIAL_REFINEMENT_FAIL": True,
            "W_CONTACT_ANGLE_OPEN": True,
            "P_VARDENS_PROJ_OPEN": True,
            "D_FRESH_TRAIN_CONTRACT_OPEN": True,
        },
        "existing_thresholds": thresholds,
        "initialization": report_base["initialization_constraint_summary"],
        "primary_blocker": primary_blocker,
        "exactly_one_next_action": next_action,
        "no_missing_stage_assembled_as_pass": True,
    }
    _write_json(EVIDENCE_DIR / "gate_and_blocker_matrix.json", gates)
    decision = {
        "initializer_verdict": report_base["initializer_verdict"],
        "fleet_impact_verdict": report_base["fleet_impact_verdict"],
        "readiness": readiness,
        "stage_decision": report_base["stage_decision"],
        "primary_blocker": primary_blocker,
        "smallest_next_action": next_action,
        "candidate_promoted": False,
        "production_semantics_changed": False,
        "contract13_promoted": False,
    }
    _write_json(EVIDENCE_DIR / "initializer_decision.json", decision)
    report = {
        **report_base,
        "elapsed_seconds": float(elapsed),
        "impact_parameter_semantics": (parameter_semantics or {}).get("status", "UNMEASURED"),
        "direct_field_labels": direct["status_gate"],
        "production_freeze": {
            "production_semantics_changed": False,
            "candidate_promoted": False,
            "solver_contract_version": 12,
            "production_timestep_policy": EXPECTED_POLICY,
            "official_dataset_written": False,
            "bulk_training_data_generated": False,
            "training_or_neural_operator_run": False,
            "production_initializer_default": "uniform (unchanged)",
        },
        "stage_decision": report_base["stage_decision"],
        "quality_status": {
            "focused_2s_pytest": "PENDING_LOCAL_VALIDATION",
            "preexisting_two_phase_regression": "PENDING_LOCAL_VALIDATION",
            "ruff": "PENDING_LOCAL_VALIDATION",
            "py_compile": "PENDING_LOCAL_VALIDATION",
            "git_diff_check": "PENDING_LOCAL_VALIDATION",
            "quick_profile": profile,
            "forensic_profile": profile == "forensic",
            "hosted_ci": "PENDING_PR",
            "locked_dependency_audit": "NOT_CLASSIFIED_UNTIL_HOSTED_CHECKS",
        },
    }
    _write_json(EVIDENCE_DIR / "l1a2s_final_report.json", report)
    md = _render_markdown(report)
    (EVIDENCE_DIR / "l1a2s_final_report.md").write_text(md)
    quality = {
        "stage": STAGE,
        "status": "BLOCKED_DIAGNOSTIC_ONLY",
        "L1B_DATA": "L1B_DATA_NOT_READY",
        "L1A_STATUS": "BLOCKED",
        "solver_contract_version": 12,
        "production_timestep_policy": EXPECTED_POLICY,
        "production_semantics_changed": False,
        "candidate_promoted": False,
        "initializer_verdict": report_base["initializer_verdict"],
        "fleet_impact_verdict": report_base["fleet_impact_verdict"],
        "readiness": readiness,
        "stage_decision": report_base["stage_decision"],
        "primary_blocker": primary_blocker,
        "profile": profile,
        "quality_claims": report["quality_status"],
        "artifacts_directory": str(out_dir.relative_to(REPO) if out_dir.is_relative_to(REPO) else out_dir),
        "evidence_file_count": 17,
    }
    _write_json(EVIDENCE_DIR / "quality_status.json", quality)
    source_hashes = _source_hashes()
    evidence_files = {}
    for path in sorted(EVIDENCE_DIR.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            evidence_files[path.name] = {"sha256": _sha_file(path), "bytes": path.stat().st_size}
    run_bindings = {}
    for run in (temporal or {}).get("_runs", {}).values():
        run_bindings[run["run_fingerprint"]] = run["binding"]
    for case_row in (full or {}).get("cases", {}).values():
        if case_row.get("trajectory_fingerprint") and case_row.get("binding"):
            run_bindings[case_row["trajectory_fingerprint"]] = case_row["binding"]
    manifest = {
        "stage": STAGE,
        "stage_version": STAGE_VERSION,
        "profile": profile,
        "base_main_git_sha": BASE_MAIN_SHA,
        "pr21": PR21,
        "solver_contract_version": 12,
        "production_timestep_policy": EXPECTED_POLICY,
        "source_hashes_sha256": source_hashes,
        "source_hash_baselines_from_l1a2r": {
            key: source_hashes[key]
            for key in (
                "phasefield_at_l1a2r",
                "generate_dataset_at_l1a2r",
                "cases_at_l1a2r",
                "timestep_policy_at_l1a2r",
                "impact_impulse_projection_audit_at_l1a2r",
            )
        },
        "source_hashes_current": {key: source_hashes[key] for key in _current_source_paths()},
        "l1a2r_report_sha256": source_hashes["l1a2r_final_report_md"],
        "l1a2r_report_json_sha256": source_hashes["l1a2r_final_report_json"],
        "initializer_version_and_parameters": {
            "C2": "c2_sdf_tapered_streamfunction_v1",
            "acceptance": INITIAL_ACCEPTANCE,
            "implementation_sha256": _sha_file(Path(__file__)),
        },
        "cases": {
            case: {
                "case_fingerprint": value.get("case_fingerprint"),
                "geometry_fingerprint": value.get("geometry_fingerprint"),
                "initial_hashes": value.get("initial_hashes"),
            }
            for case, value in ((full or {}).get("cases", {})).items()
        },
        "run_bindings": run_bindings,
        "dtype_policy": {
            "phase_solver": "float64 current phase_storage_model",
            "velocity_and_geometry": "float32 p.dtype",
            "generator_sample_phi": "float32",
            "generator_sample_u_v": "float16",
        },
        "profile_and_tests": {
            "profile": profile,
            "forensic_N192_completed": profile == "forensic",
            "candidate_promoted": False,
        },
        "runtime": preflight["runtime"],
        "elapsed_seconds": float(elapsed),
        "artifacts": evidence_files,
        "quality_status": quality,
    }
    _write_json(EVIDENCE_DIR / "manifest.json", manifest)
    return report


def run(profile: str = "forensic", out: str | Path | None = None, stages: str = "all") -> dict[str, Any]:
    if profile not in ("quick", "forensic"):
        raise ValueError(profile)
    if not bool(jax.config.jax_enable_x64):
        raise AuditValidationError(
            "run with JAX_ENABLE_X64=1; contract-12 phase storage is float64 while velocity remains float32"
        )
    out_dir = Path(out) if out else ARTIFACT_DIR
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = {part.strip() for part in stages.split(",") if part.strip()}
    if selected == {"all"}:
        selected = {
            "map",
            "initial",
            "ledger",
            "controls",
            "temporal",
            "events",
            "full_horizon",
            "refinement",
            "assemble",
        }
    allowed = {"map", "initial", "ledger", "controls", "temporal", "events", "full_horizon", "refinement", "assemble"}
    unknown = selected - allowed
    if unknown:
        raise AuditValidationError(f"unknown stage(s): {sorted(unknown)}")
    start = time.perf_counter()
    preflight = _preflight_payload()
    _write_json(
        EVIDENCE_DIR / "source_and_prerequisite_map.json", {**preflight, "operator_map": _source_operator_map()}
    )
    if profile == "quick":
        n = 48
        bundles = {name: _derive_bundle(name, case, n, 0.004) for name, case in preflight["case_definitions"].items()}
        ctx: dict[str, Any] = {"cases": preflight["case_definitions"], "bundles": bundles, "candidate_states": {}}
        initial = (
            _measure_initial_matrix(ctx)
            if "initial" in selected
            else {"constraint_matrix": {}, "region_budget": {}, "verdict": "INCONCLUSIVE"}
        )
        initial["methodology_only_N48_verdict"] = initial["verdict"]
        initial["verdict"] = "INCONCLUSIVE"  # quick observations can never classify N=192 compatibility
        initial["bundles"] = bundles
        selected = {"map", "initial", "assemble"}
        startup = controls = temporal = events = full = spatial = parameter_semantics = None
        init_verdict = initial["verdict"]
        readiness = {
            key: "UNMEASURED"
            for key in ("TEMPORAL_REFINEMENT", "SPATIAL_REFINEMENT", "DIRECT_FIELD_LABELS", "GENERATOR_QUALITY_GATES")
        }
        base = {
            "profile": profile,
            "initializer_verdict": init_verdict,
            "fleet_impact_verdict": "INCONCLUSIVE",
            "readiness": readiness,
            "initialization_constraint_summary": {},
            "stage_decision": {
                "decision": "D_INCONCLUSIVE_OR_INCOMPLETE",
                "primary_blocker": "QUICK_PROFILE_NOT_FORENSIC",
                "next_action": (
                    "run the bounded forensic N=192 profile; do not treat quick results as readiness evidence"
                ),
            },
            "decision_summary": (
                "Quick profile measures methodology only at N=48. No N=192 trajectory, event or readiness "
                "claim is available."
            ),
            "impact_parameter_semantics": "UNMEASURED",
            "production_freeze": {"production_semantics_changed": False},
            "primary_blocker": "QUICK_PROFILE_NOT_FORENSIC",
            "next_action": "run the bounded forensic N=192 profile; do not treat quick results as readiness evidence",
        }
        report = _write_required_evidence(
            profile=profile,
            out_dir=out_dir,
            preflight=preflight,
            initial=initial,
            startup=None,
            controls=None,
            temporal=None,
            events=None,
            full=None,
            spatial=None,
            parameter_semantics=None,
            report_base=base,
            elapsed=time.perf_counter() - start,
        )
        return report

    # The complete forensic route is deliberately ordered and fail-closed.
    bundles = {
        name: _derive_bundle(name, case, 192, REQUESTED_DT) for name, case in preflight["case_definitions"].items()
    }
    ctx = {"cases": preflight["case_definitions"], "bundles": bundles, "candidate_states": {}}
    initial = (
        _measure_initial_matrix(ctx)
        if "initial" in selected
        else {"constraint_matrix": {}, "region_budget": {}, "verdict": "INCONCLUSIVE"}
    )
    initial["bundles"] = bundles
    c2_pass = initial["verdict"] == "INITIALIZER_COMPATIBLE"
    _progress(f"N=192 initial matrix verdict={initial['verdict']} C2_compatibility_gate={c2_pass}")
    startup = controls = temporal = events = full = spatial = parameter_semantics = None
    if c2_pass and "ledger" in selected:
        _progress("stage start: first-substep causal ledger")
        startup = _first_substep_ledger(ctx)
        _progress("stage complete: first-substep causal ledger")
    if c2_pass and "controls" in selected:
        _progress("stage start: historical and no-solid controls")
        controls = _old_negative_controls(192)
        controls["NO_SOLID_C2"] = _no_solid_c2_control(
            bundles["flat_we100_ct050"],
            ctx["candidate_states"][("flat_we100_ct050", "SDF_TAPERED_STREAMFUNCTION_V1")],
            profile,
            out_dir,
        )
        controls["C2_DISTINCT_SURFACE"] = {
            case: {
                "geometry_hash": _geometry_hash(bundles[case].solid),
                "candidate_metrics": initial["constraint_matrix"][case]["SDF_TAPERED_STREAMFUNCTION_V1"],
                "case_specific_x0_y0": [bundles[case].x0, bundles[case].y0],
            }
            for case in CASE_NAMES
        }
        _progress("stage complete: historical and no-solid controls")
    if c2_pass and "temporal" in selected:
        _progress("stage start: temporal dt/refinement matrix")
        temporal = _run_temporal(ctx, profile, out_dir)
        _progress("stage complete: temporal dt/refinement matrix")
    if c2_pass and "events" in selected and temporal is not None and temporal.get("status") == "MEASURED":
        _progress("stage start: four-canary event screens")
        events, event_runs = _make_event_runs(ctx, temporal, profile, out_dir)
        _progress("stage complete: four-canary event screens")
    else:
        event_runs = {}
    if c2_pass and "full_horizon" in selected and event_runs:
        _progress("stage start: T=8 canary continuation and data-readiness checks")
        full, full_runs = _full_horizon_runs(ctx, event_runs, profile, out_dir)
        _progress("stage complete: T=8 canary continuation and data-readiness checks")
        # Attach each T=8 event to the event matrix. The screening result stays explicitly short-window.
        for case_name in CASE_NAMES:
            events[case_name]["full_horizon"] = full["cases"][case_name]["impact_authentication"]
            events[case_name]["full_horizon_verdict"] = full["cases"][case_name]["impact_authentication"]["verdict"]
            events[case_name]["same_run_continued_from_screen"] = True
    else:
        full = {
            "status": "UNMEASURED",
            "reason": "conditional T=8 stage not reached because early C2 gate failed or stage omitted",
        }
    if c2_pass and "refinement" in selected:
        _progress("stage start: N=144/N=192 spatial refinement re-audit")
        spatial, spatial_runs = _spatial_reaudit(ctx, profile, out_dir)
        _progress("stage complete: N=144/N=192 spatial refinement re-audit")
    else:
        spatial = {"status": "UNMEASURED", "reason": "conditional refinement stage not reached"}
    parameter_semantics = (
        _impact_parameters(
            {case: (full or {}).get("cases", {}).get(case, {}).get("impact_authentication", {}) for case in CASE_NAMES},
            bundles,
        )
        if full and full.get("status") == "MEASURED"
        else {"status": "UNMEASURED", "reason": "no full-horizon contact-speed matrix"}
    )
    direct_field = _direct_field_gate(temporal, full)
    if temporal is not None:
        # The exported solver-grid and dataset-grid norms are already summarized; discard
        # cached JAX states/arrays before final JSON assembly to keep the forensic footprint bounded.
        temporal.pop("_runs", None)
    event_runs.clear()
    base = _assemble_evidence(
        profile=profile,
        out_dir=out_dir,
        preflight=preflight,
        initial=initial,
        startup=startup,
        controls=controls,
        temporal=temporal,
        events=events,
        full=full,
        spatial=spatial,
        direct_field=direct_field,
        elapsed=time.perf_counter() - start,
    )
    decision, blocker, action = _decision(
        base["initializer_verdict"],
        base["fleet_impact_verdict"],
        base["readiness"],
        "RESOLVED"
        if parameter_semantics.get("status") == "RESOLVED"
        else parameter_semantics.get("status", "UNRESOLVED"),
        profile,
    )
    base.update(
        {
            "stage_decision": {"decision": decision, "primary_blocker": blocker, "next_action": action},
            "primary_blocker": blocker,
            "next_action": action,
            "impact_parameter_semantics": parameter_semantics.get("status", "UNRESOLVED"),
            "decision_summary": (
                f"Initializer={base['initializer_verdict']}; fleet={base['fleet_impact_verdict']}; "
                f"readiness={base['readiness']}. {blocker} is the single primary obstruction selected "
                "for the next bounded action."
            ),
        }
    )
    report = _write_required_evidence(
        profile=profile,
        out_dir=out_dir,
        preflight=preflight,
        initial=initial,
        startup=startup,
        controls=controls,
        temporal=temporal,
        events=events,
        full=full,
        spatial=spatial,
        parameter_semantics=parameter_semantics,
        report_base=base,
        elapsed=time.perf_counter() - start,
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("quick", "forensic"), default="forensic")
    parser.add_argument("--out", default=str(ARTIFACT_DIR))
    parser.add_argument(
        "--stages",
        default="all",
        help="comma list: map,initial,ledger,controls,temporal,events,full_horizon,refinement,assemble",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = run(profile=args.profile, out=args.out, stages=args.stages)
    print(
        f"[{STAGE}] profile={args.profile} initializer={report['initializer_verdict']} "
        f"fleet={report['fleet_impact_verdict']} decision={report['stage_decision']['decision']} "
        f"elapsed={report['elapsed_seconds']:.1f}s"
    )


if __name__ == "__main__":
    main()
